"""Data retrieval for the conversational layer: what each answer is built from.

One function per kind of answer, for one space (current_data, history_data,
forecast_data, anomaly_data) and one for a whole floor or the building
(floor_data), which runs the per-space ones over every space and summarizes.
Every number the user reads comes out of here or out of the GRU, never out of
the LLM.

What is wired and what is not. The GRU and the 3-sigma anomaly detector are
real: the forecast comes from the trained checkpoint in models/, and the
detector uses the mean and sigma stored next to it. The data source is not:
everything reads the cleaned sensor file written by ingestion.cleaning,
loaded at startup, because the Tec database is not reachable yet. So
"current" means the latest reading in that file, not the present moment.
space_rows() is the one place that changes when the database is; it is
marked `TODO E5`.

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-11-26 - Anomaly questions answered by the 3-sigma detector.

Reference:
    Sections 4.3.1 to 4.3.4 of the research document.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from anomaly import detector as det
from common import config as cfg
from common import spaces
from gru import predict as pred
from llm import classifier

# Default span for a historical question when the user names no period.
HISTORY_HOURS = 24


# Process-wide state, filled once by the API at startup. Reloading the
# checkpoint per request would add seconds of latency to every question.
STATE: dict[str, Any] = {
    "net": None,
    "normalizer": None,
    "metadata": None,
    "data": None,
    "model_error": None,
}


# --------------------------------------------------------------------------
# One space
# --------------------------------------------------------------------------
def space_rows(space_id: int) -> pd.DataFrame:
    """Return every row of one space, sorted by time.

    TODO E5: replace with a query to the Tec database. Everything else in
    this module reads the data through here.

    Args:
        space_id: Space identifier.

    Returns:
        The rows for that space, possibly empty.
    """
    data = STATE["data"]
    own_rows = data[data[cfg.SPACE_COLUMN] == space_id]
    return own_rows.sort_values(cfg.TIME_COLUMN).reset_index(drop=True)


def rooms_with_data() -> list[str]:
    """Return the labels of the spaces that have readings, sorted by id."""
    labels = spaces.labels()
    ids = sorted(int(s) for s in STATE["data"][cfg.SPACE_COLUMN].unique())
    return [str(labels.get(s, s)) for s in ids]


def number(value: object) -> float | None:
    """Coerce a pandas scalar to a plain float for JSON.

    Args:
        value: Value taken from a DataFrame.

    Returns:
        The value as a rounded float, or None when it is missing.
    """
    if value is None or pd.isna(value):
        return None
    return round(float(value), 2)


def current_data(space_id: int) -> dict:
    """Retrieve the latest reading of one space.

    Args:
        space_id: Space identifier.

    Returns:
        A dict shaped for context._prompt_current.
    """
    rows = space_rows(space_id)
    if rows.empty:
        return {"space": space_id, "readings": {}, "timestamp": None}

    last = rows.iloc[-1]
    readings = {}
    for variable in cfg.ACTIVE_VARIABLES:
        readings[variable] = number(last[variable])
    return {
        "space": space_id,
        "timestamp": str(last[cfg.TIME_COLUMN]),
        "readings": readings,
    }


def history_data(space_id: int, variable: str, hours: int = HISTORY_HOURS) -> dict:
    """Summarize one variable of one space over a past window.

    Args:
        space_id: Space identifier.
        variable: Which variable to summarize.
        hours: How far back to look from the latest reading.

    Returns:
        A dict shaped for context._prompt_history.
    """
    rows = space_rows(space_id)
    if rows.empty:
        return {"space": space_id, "variable": variable, "summary": {}}

    stamps = pd.to_datetime(rows[cfg.TIME_COLUMN])
    cutoff = stamps.max() - pd.Timedelta(hours=hours)
    series = rows[stamps >= cutoff][variable].dropna()

    has_values = len(series) > 0
    return {
        "space": space_id,
        "variable": variable,
        "since": str(cutoff),
        "until": str(stamps.max()),
        "summary": {
            "max": number(series.max()) if has_values else None,
            "min": number(series.min()) if has_values else None,
            "mean": number(series.mean()) if has_values else None,
            "n_readings": int(len(series)),
        },
    }


def anomaly_data(space_id: int, variables: list[str] | None = None) -> dict:
    """Check the latest reading of one space with the 3-sigma detector.

    Args:
        space_id: Space identifier.
        variables: The variables the question named; None or empty checks
            all of them.

    Returns:
        A dict shaped for context._prompt_anomaly: state "evaluated" with the
        reading and the anomalies found (possibly none), or "unavailable"
        with the reason it could not be checked. The two are never confused:
        reporting "nothing found" when nothing ran would be a false all-clear.
    """
    reason, checked = _anomaly_check_plan(variables)
    if reason:
        return {"space": space_id, "state": "unavailable", "anomalies": [], "detail": reason}

    current = current_data(space_id)
    readings = {v: current["readings"].get(v) for v in checked}
    return {
        "space": space_id,
        "state": "evaluated",
        "timestamp": current["timestamp"],
        "checked": checked,
        "readings": readings,
        "anomalies": det.check(readings, STATE["normalizer"].parameters),
    }


def _anomaly_check_plan(variables: list[str] | None) -> tuple[str | None, list[str]]:
    """Decide which variables the detector can check, or why it cannot.

    Args:
        variables: The variables the question named, possibly none.

    Returns:
        A tuple of (reason it cannot run, or None; variables to check).
    """
    normalizer = STATE["normalizer"]
    if normalizer is None:
        reason = (
            "El detector de anomalías usa la media y la desviación estándar del "
            f"entrenamiento del GRU, y el modelo no está cargado: {STATE['model_error']}"
        )
        return reason, []

    checked = [v for v in normalizer.parameters if not variables or v in variables]
    if not checked:
        # Only motion was asked about, and the detector does not check it.
        return "El detector no revisa el movimiento: es una variable binaria (0 o 1).", []
    return None, checked


def forecast_data(intent: classifier.Intent, space_id: int) -> dict:
    """Run the GRU forward for one space.

    Args:
        intent: The classified question, which carries the horizon.
        space_id: Space identifier.

    Returns:
        A dict shaped for context._prompt_forecast. When the model or a
        usable window is unavailable it carries an "error" instead of
        predictions, so the prompt can say so plainly.
    """
    if STATE["net"] is None:
        return _no_forecast(space_id, f"El modelo GRU no está cargado: {STATE['model_error']}")

    metadata = STATE["metadata"]
    window = pred.last_contiguous_window(space_rows(space_id), metadata["input_steps"])
    if window is None:
        return _no_forecast(
            space_id,
            f"No hay {metadata['input_steps']} lecturas consecutivas recientes, "
            "así que no se puede predecir.",
        )

    try:
        output = pred.predict(
            STATE["net"],
            STATE["normalizer"],
            metadata,
            window,
            space_id=space_id,
            horizon_min=intent.horizon_min,
        )
    except ValueError as e:
        return _no_forecast(space_id, str(e))

    last = window.iloc[-1]
    current = {}
    for variable in metadata["active_variables"]:
        current[variable] = number(last[variable])
    return {
        "space": space_id,
        "current_reading": current,
        "timestamp": str(last[cfg.TIME_COLUMN]),
        **output,
    }


def _no_forecast(space_id: int, error: str) -> dict:
    """The forecast dict when there is no forecast, only the reason."""
    return {"space": space_id, "current_reading": {}, "predictions": [], "error": error}


def retrieve(intent: classifier.Intent, space_id: int) -> dict:
    """Fetch whatever a question about one space needs.

    Args:
        intent: Result of classifier.classify().
        space_id: Space to answer about.

    Returns:
        The data dict for context.build_prompt().
    """
    if intent.kind == "forecast":
        data = forecast_data(intent, space_id)
    elif intent.kind == "history":
        variable = intent.variables[0] if intent.variables else "co2"
        data = history_data(space_id, variable)
    elif intent.kind == "anomaly":
        data = anomaly_data(space_id, intent.variables)
    else:
        data = current_data(space_id)
    # The prompt names the space by its room number: given the id, Llama
    # answered "espacio 308" and, with the old ids, "espacio 5".
    data["space"] = spaces.labels().get(space_id, space_id)
    return data


# --------------------------------------------------------------------------
# A whole floor, or the building
# --------------------------------------------------------------------------
def floor_data(intent: classifier.Intent, floor: int | None) -> dict:
    """Retrieve what a question about a whole floor, or the building, needs.

    Runs the per-space retrieval for every space that has readings and adds a
    summary. Spaces without a sensor are named so the answer can say the
    floor is only partly measured.

    Args:
        intent: The classified question; carries the kind, the variables
            and, for forecasts, the horizon.
        floor: Floor number, or None for every space in the building.

    Returns:
        A dict shaped for floor_prompt.prompt(), with "scope" set to
        "floor" or "building".
    """
    if floor is None:
        space_ids = sorted(s for ids in classifier.FLOOR_SPACES.values() for s in ids)
    else:
        space_ids = sorted(classifier.FLOOR_SPACES.get(floor, []))

    variables = intent.variables or list(cfg.ACTIVE_VARIABLES)
    if intent.kind == "history":
        # Same limit as the single-space path: one variable per question.
        variables = [intent.variables[0] if intent.variables else "co2"]

    data: dict[str, Any] = {
        "scope": "building" if floor is None else "floor",
        "floor": floor,
        "kind": intent.kind,
        "variables": variables,
        "n_spaces": len(space_ids),
        "rooms": [],
        "without_data": [],
        "errors": [],
    }
    if intent.kind == "anomaly":
        reason, checked = _anomaly_check_plan(intent.variables)
        if reason:
            data["state"], data["detail"] = "unavailable", reason
            return data
        data["state"], data["variables"] = "evaluated", checked

    labels = spaces.labels()
    for space_id in space_ids:
        room = labels.get(space_id, str(space_id))
        if space_rows(space_id).empty:
            data["without_data"].append(room)
        elif intent.kind == "forecast":
            _add_forecast_room(data, intent, space_id, room)
        elif intent.kind == "history":
            item = history_data(space_id, variables[0])
            data["rooms"].append({"room": room, "summary": item["summary"]})
            data["since"], data["until"] = item.get("since"), item.get("until")
        elif intent.kind == "anomaly":
            item = anomaly_data(space_id, intent.variables)
            data["rooms"].append({
                "room": room, "timestamp": item["timestamp"], "anomalies": item["anomalies"],
            })
        else:
            item = current_data(space_id)
            readings = {v: item["readings"].get(v) for v in variables}
            data["rooms"].append({"room": room, "timestamp": item["timestamp"], "readings": readings})

    if intent.kind == "forecast":
        data["summary"] = _floor_summary(data["rooms"], "predicted", variables)
    elif intent.kind == "history":
        data["summary"] = _history_summary(data["rooms"], variables[0])
    elif intent.kind == "anomaly":
        with_anomalies = [r["room"] for r in data["rooms"] if r["anomalies"]]
        data["summary"] = {"rooms_with_anomalies": with_anomalies}
    else:
        data["summary"] = _floor_summary(data["rooms"], "readings", variables)

    if intent.kind in ("current", "anomaly"):
        stamps = [r["timestamp"] for r in data["rooms"] if r["timestamp"]]
        data["timestamp"] = max(stamps) if stamps else None
    return data


def _add_forecast_room(
    data: dict, intent: classifier.Intent, space_id: int, room: str
) -> None:
    """Forecast one room of a floor and add it, or its error, to `data`.

    Args:
        data: The floor dict being built; modified in place.
        intent: The classified question.
        space_id: Space to forecast.
        room: Its label.
    """
    item = forecast_data(intent, space_id)
    if item.get("error"):
        # The detail line reads "3304: sin pronóstico (no hay 12 lecturas
        # consecutivas recientes)", so only the reason is kept, lowercased.
        reason = item["error"].split(", así que")[0].rstrip(".")
        data["errors"].append({"room": room, "reason": reason[:1].lower() + reason[1:]})
        return

    variables = data["variables"]
    last = item["predictions"][-1] if item["predictions"] else {}
    data["rooms"].append({
        "room": room,
        "current": {v: item["current_reading"].get(v) for v in variables},
        "predicted": {v: number(last.get(v)) for v in variables},
    })
    # Same for every room of one question; the last one written stays.
    data["timestamp"] = item.get("timestamp")
    data["horizon_min"] = item.get("horizon_min")
    data["method"] = item.get("method")
    data["warning"] = item.get("warning")


def _floor_summary(rooms: list[dict], key: str, variables: list[str]) -> dict:
    """Mean, minimum and maximum of each variable across the rooms of a floor.

    Computed here and not by the LLM, like every other number in the answer.

    Args:
        rooms: One entry per room, each holding a dict of values under `key`.
        key: Which dict of each room to summarize.
        variables: Variables to summarize.

    Returns:
        Per variable: mean, min and max, each extreme with the room it is in,
        and how many rooms had a value.
    """
    summary = {}
    for variable in variables:
        # (value, room) pairs, so min() and max() also say which room.
        pairs = []
        for room in rooms:
            value = room.get(key, {}).get(variable)
            if value is not None:
                pairs.append((value, room["room"]))
        if not pairs:
            continue
        low, high = min(pairs), max(pairs)
        summary[variable] = {
            "mean": round(sum(v for v, _ in pairs) / len(pairs), 2),
            "min": low[0], "min_room": low[1],
            "max": high[0], "max_room": high[1],
            "n_rooms": len(pairs),
        }
    return summary


def _history_summary(rooms: list[dict], variable: str) -> dict:
    """Highest maximum, lowest minimum and mean of means across a floor.

    Args:
        rooms: One entry per room, each with the "summary" of history_data().
        variable: The one variable a history question is about.

    Returns:
        {variable: {max, max_room, min, min_room, mean, n_rooms}}, with None
        where no room had a value.
    """
    maxima, minima, means = [], [], []
    for room in rooms:
        s = room["summary"]
        if s.get("max") is not None:
            maxima.append((s["max"], room["room"]))
        if s.get("min") is not None:
            minima.append((s["min"], room["room"]))
        if s.get("mean") is not None:
            means.append(s["mean"])

    highest = max(maxima) if maxima else (None, None)
    lowest = min(minima) if minima else (None, None)
    return {variable: {
        "max": highest[0], "max_room": highest[1],
        "min": lowest[0], "min_room": lowest[1],
        "mean": round(sum(means) / len(means), 2) if means else None,
        "n_rooms": len(rooms),
    }}
