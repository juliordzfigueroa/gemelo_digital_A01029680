"""Data retrieval for one space: what each answer is built from.

One function per kind of answer (current_data, history_data, forecast_data,
anomaly_data), each for a moment: the latest measured reading by default, or
the date and time the question names. Questions about a whole floor are in
api/floors.py, built on these. Every number the user reads comes out of here,
out of the GRU or out of the generator, never out of the LLM.

Every answer carries its origin, decided by api/sources.py: the sensor's
readings when it measured that moment, the generator's rows when it did not
("simulado") or when the room has no sensor ("estimado"). A question without
a date about a measured room answers exactly as before the generator existed.

What is wired and what is not. The GRU and the 3-sigma detector are real.
The measured data is not live: it is the cleaned sensor file written by
ingestion.cleaning, because the Tec database is not reachable yet, so
"current" means the latest reading in that file. space_rows() is the one
place that changes when the database is; it is marked `TODO E5`.

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-11-26 - Answers for any moment, from the sensors or the generator;
    floors moved to api/floors.py.

Reference:
    Sections 4.3.1 to 4.3.4 of the research document.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from anomaly import detector as det
from api import sources as src
from common import config as cfg
from common import spaces
from gru import predict as pred
from llm import classifier

# Default span for a historical question when the user names no day.
HISTORY_HOURS = 24

# A reading older than this does not describe the moment asked about: the
# same 15 minutes the cleaning is willing to interpolate across.
STALE = pd.Timedelta(seconds=cfg.MAX_INTERPOLABLE_STEPS * cfg.INTERVAL_SECONDS)

STEP = pd.Timedelta(seconds=cfg.INTERVAL_SECONDS)

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
# Rows, and where they come from
# --------------------------------------------------------------------------
def space_rows(space_id: int) -> pd.DataFrame:
    """Return every measured row of one space, sorted by time.

    TODO E5: replace with a query to the Tec database. Everything else in
    this module reads the measured data through here.

    Args:
        space_id: Space identifier.

    Returns:
        The rows for that space, possibly empty.
    """
    data = STATE["data"]
    own_rows = data[data[cfg.SPACE_COLUMN] == space_id]
    return own_rows.sort_values(cfg.TIME_COLUMN).reset_index(drop=True)


def latest_time() -> pd.Timestamp:
    """The newest measured reading: the "now" of a question without a date."""
    return pd.Timestamp(STATE["data"][cfg.TIME_COLUMN].max())


def rooms_with_data() -> list[str]:
    """Return the labels of the spaces that have readings, sorted by id."""
    labels = spaces.labels()
    ids = sorted(int(s) for s in STATE["data"][cfg.SPACE_COLUMN].unique())
    return [str(labels.get(s, s)) for s in ids]


def rows_between(space_id: int, since: pd.Timestamp, until: pd.Timestamp) -> tuple[pd.DataFrame, str | None]:
    """The rows of one space in a period, and their single origin.

    The sensor's rows if it measured anything in the period; otherwise the
    generator's. Never a mix of both.

    Args:
        space_id: Space identifier.
        since: Start of the period, inclusive.
        until: End of the period, inclusive.

    Returns:
        A tuple of (rows, origin), origin None when neither has rows there.
    """
    measured = space_rows(space_id)
    stamps = measured[cfg.TIME_COLUMN]
    inside = measured[(stamps >= since) & (stamps <= until)]
    if not inside.empty:
        return inside, "medido"

    generated = src.generated_rows(space_id)
    if not generated.empty:
        stamps = generated[cfg.TIME_COLUMN]
        inside = generated[(stamps >= since) & (stamps <= until)]
        if not inside.empty:
            return inside, str(inside["origin"].iloc[0])
    return measured.iloc[0:0], None


def latest_reading(space_id: int, at: pd.Timestamp) -> tuple[pd.Series | None, str | None]:
    """The reading that describes one space at a moment, and its origin.

    Args:
        space_id: Space identifier.
        at: The moment.

    Returns:
        A tuple of (row, origin), or (None, None) when there is nothing.
    """
    rows, origin = rows_between(space_id, at - STALE, at)
    if origin:
        return rows.iloc[-1], origin
    if at != latest_time():
        return None, None
    # Asked without a date, a sensor that stopped reporting and has nothing
    # generated for it (the mentors office) still answers with its last
    # reading, dated, as before.
    measured = space_rows(space_id)
    before = measured[measured[cfg.TIME_COLUMN] <= at]
    if not before.empty:
        return before.iloc[-1], "medido"
    return None, None


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


# --------------------------------------------------------------------------
# One kind of answer each
# --------------------------------------------------------------------------
def current_data(space_id: int, at: pd.Timestamp) -> dict:
    """Retrieve the reading of one space at a moment.

    Args:
        space_id: Space identifier.
        at: The moment.

    Returns:
        A dict shaped for context._prompt_current, with its "origin".
    """
    row, origin = latest_reading(space_id, at)
    if row is None:
        return {"space": space_id, "readings": {}, "timestamp": None, "origin": None}

    readings = {}
    for variable in cfg.ACTIVE_VARIABLES:
        readings[variable] = number(row[variable])
    return {
        "space": space_id,
        "timestamp": str(row[cfg.TIME_COLUMN]),
        "readings": readings,
        "origin": origin,
    }


def history_data(space_id: int, variable: str, since: pd.Timestamp, until: pd.Timestamp) -> dict:
    """Summarize one variable of one space over a period.

    Args:
        space_id: Space identifier.
        variable: Which variable to summarize.
        since: Start of the period.
        until: End of the period.

    Returns:
        A dict shaped for context._prompt_history, with its "origin".
    """
    rows, origin = rows_between(space_id, since, until)
    if origin is None and until != latest_time():
        return {"space": space_id, "variable": variable, "summary": {}, "origin": None}
    if origin is None:
        # Asked without a date, a sensor that stopped reporting: the same
        # span, ending at its last reading, as before the generator existed.
        measured = space_rows(space_id)
        measured = measured[measured[cfg.TIME_COLUMN] <= until]
        if measured.empty:
            return {"space": space_id, "variable": variable, "summary": {}, "origin": None}
        span = until - since
        until = measured[cfg.TIME_COLUMN].max()
        since = until - span
        rows, origin = measured[measured[cfg.TIME_COLUMN] >= since], "medido"

    series = rows[variable].dropna()
    has_values = len(series) > 0
    return {
        "space": space_id,
        "variable": variable,
        "since": str(since),
        "until": str(until),
        "summary": {
            "max": number(series.max()) if has_values else None,
            "min": number(series.min()) if has_values else None,
            "mean": number(series.mean()) if has_values else None,
            "n_readings": int(len(series)),
        },
        "origin": origin,
    }


def anomaly_data(space_id: int, variables: list[str] | None, at: pd.Timestamp) -> dict:
    """Check the reading of one space at a moment with the 3-sigma detector.

    Args:
        space_id: Space identifier.
        variables: The variables the question named; None or empty checks
            all of them.
        at: The moment.

    Returns:
        A dict shaped for context._prompt_anomaly: state "evaluated" with the
        reading and the anomalies found (possibly none), or "unavailable"
        with the reason it could not be checked. Only measured readings are
        checked: a generated value has no sensor that could fail.
    """
    reason, checked = _anomaly_check_plan(variables)
    current = current_data(space_id, at)
    if not reason and current["origin"] != "medido":
        reason = (
            "No hay lecturas medidas de ese momento; las anomalías solo se buscan "
            "en lecturas reales, no en valores del generador."
        )
    if reason:
        return {"space": space_id, "state": "unavailable", "anomalies": [], "detail": reason}

    readings = {v: current["readings"].get(v) for v in checked}
    return {
        "space": space_id,
        "state": "evaluated",
        "timestamp": current["timestamp"],
        "checked": checked,
        "readings": readings,
        "anomalies": det.check(readings, STATE["normalizer"].parameters),
        "origin": "medido",
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


def forecast_data(intent: classifier.Intent, space_id: int, at: pd.Timestamp) -> dict:
    """Forecast one space: with the GRU when it can, otherwise with the generator.

    The GRU only knows the rooms it was trained on, and needs an hour of
    consecutive readings. A room without a sensor, or a sensor without that
    hour, gets the generator's rows for the hours ahead, which come from the
    schedule and the weather of those hours.

    Args:
        intent: The classified question, which carries the horizon.
        space_id: Space identifier.
        at: The moment to forecast from.

    Returns:
        A dict shaped for context._prompt_forecast. When neither can forecast
        it carries an "error" instead of predictions.
    """
    reason = None
    metadata = STATE["metadata"]
    if STATE["net"] is None:
        reason = f"El modelo GRU no está cargado: {STATE['model_error']}"
    elif str(space_id) in metadata["space_index"]:
        measured = space_rows(space_id)
        measured = measured[measured[cfg.TIME_COLUMN] <= at]
        window = pred.last_contiguous_window(measured, metadata["input_steps"])
        if window is not None:
            return _gru_forecast(intent, space_id, window)
        reason = (
            f"No hay {metadata['input_steps']} lecturas consecutivas recientes, "
            "así que no se puede predecir."
        )

    generated = _generated_forecast(intent, space_id, at)
    if generated is not None:
        return generated
    return _no_forecast(space_id, reason or "El GRU solo pronostica los salones con los que se entrenó.")


def _gru_forecast(intent: classifier.Intent, space_id: int, window: pd.DataFrame) -> dict:
    """Run the GRU forward from an hour of measured readings."""
    metadata = STATE["metadata"]
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
        "origin": "medido",
    }


def _generated_forecast(intent: classifier.Intent, space_id: int, at: pd.Timestamp) -> dict | None:
    """The generator's rows for the hours after a moment, shaped as a forecast.

    Returns:
        The forecast dict, or None when the generator does not cover them.
    """
    now = current_data(space_id, at)
    rows, origin = rows_between(space_id, at + STEP, at + pd.Timedelta(minutes=intent.horizon_min))
    steps = intent.horizon_min // (cfg.INTERVAL_SECONDS // 60)
    if origin in (None, "medido") or len(rows) < steps or now["origin"] is None:
        return None

    predictions = []
    for i, (_, row) in enumerate(rows.iterrows()):
        step = {"offset_min": (i + 1) * cfg.INTERVAL_SECONDS // 60}
        for variable in cfg.ACTIVE_VARIABLES:
            step[variable] = number(row[variable])
        predictions.append(step)
    return {
        "space": space_id,
        "current_reading": now["readings"],
        "timestamp": now["timestamp"],
        "predictions": predictions,
        "horizon_min": intent.horizon_min,
        "method": "generador",
        "warning": None,
        "origin": origin,
    }


def _no_forecast(space_id: int, error: str) -> dict:
    """The forecast dict when there is no forecast, only the reason."""
    return {"space": space_id, "current_reading": {}, "predictions": [], "error": error}


def retrieve(intent: classifier.Intent, space_id: int, at: pd.Timestamp, since: pd.Timestamp) -> dict:
    """Fetch whatever a question about one space needs.

    Args:
        intent: Result of classifier.classify().
        space_id: Space to answer about.
        at: The moment of the question.
        since: Start of the period of a history question.

    Returns:
        The data dict for context.build_prompt(). A generated answer also
        carries the rooms it comes from and its expected error.
    """
    if intent.kind == "forecast":
        data = forecast_data(intent, space_id, at)
    elif intent.kind == "history":
        variable = intent.variables[0] if intent.variables else "co2"
        data = history_data(space_id, variable, since, at)
    elif intent.kind == "anomaly":
        data = anomaly_data(space_id, intent.variables, at)
    else:
        data = current_data(space_id, at)
    if data.get("origin") in ("simulado", "estimado"):
        data.update(src.provenance(space_id, data["origin"]))
        # The 3304 is estimated too, but it does have a sensor: a dead one.
        data["has_sensor"] = not space_rows(space_id).empty
    # The prompt names the space by its room number: given the id, Llama
    # answered "espacio 308" and, with the old ids, "espacio 5".
    data["space"] = spaces.labels().get(space_id, space_id)
    return data
