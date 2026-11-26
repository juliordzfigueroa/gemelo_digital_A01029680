"""Data retrieval for a whole floor, or the building, at one moment.

Runs the per-space retrieval of api/retrieval.py for every space of the
floor and adds a summary (mean, lowest and highest room), computed here and
not by the LLM. Each room keeps its own origin: a room without a sensor is
included with its estimate and marked "(estimado)" in its name, so the mark
travels into the summary, the prompt and the room-by-room detail without any
of them having to remember it. Spaces with neither readings nor an estimate
(a meeting room) are named apart, so the answer can say the floor is only
partly covered.

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-11-26 - Moved here from retrieval.py; includes the estimated rooms.

Reference:
    Sections 4.3.4 and 4.4 of the research document.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from api import retrieval as ret
from api import sources as src
from common import config as cfg
from common import spaces
from llm import classifier


def floor_data(intent: classifier.Intent, floor: int | None, at: pd.Timestamp, since: pd.Timestamp) -> dict:
    """Retrieve what a question about a whole floor, or the building, needs.

    Args:
        intent: The classified question; carries the kind, the variables
            and, for forecasts, the horizon.
        floor: Floor number, or None for every space in the building.
        at: The moment of the question.
        since: Start of the period of a history question.

    Returns:
        A dict shaped for floor_prompt.prompt(), with "scope" set to "floor"
        or "building", and "origins" naming the generated ones it includes.
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
        "not_checked": [],
        "origins": [],
        "provenance": {},
        "n_measured": 0,
    }
    if intent.kind == "anomaly":
        reason, checked = ret._anomaly_check_plan(intent.variables)
        if reason:
            data["state"], data["detail"] = "unavailable", reason
            return data
        data["state"], data["variables"] = "evaluated", checked

    labels = spaces.labels()
    for space_id in space_ids:
        _add_room(data, intent, space_id, str(labels.get(space_id, space_id)), at, since)

    _summarize(data, intent.kind)
    return data


def _add_room(data: dict, intent, space_id: int, label: str, at, since) -> None:
    """Retrieve one room of the floor and add it to `data`, where it belongs.

    Args:
        data: The floor dict being built; modified in place.
        intent: The classified question.
        space_id: The room.
        label: Its name.
        at: The moment of the question.
        since: Start of the period of a history question.
    """
    if ret.space_rows(space_id).empty and src.generated_rows(space_id).empty:
        data["without_data"].append(label)
        return

    kind = intent.kind
    if kind == "forecast":
        item = ret.forecast_data(intent, space_id, at)
    elif kind == "history":
        item = ret.history_data(space_id, data["variables"][0], since, at)
    elif kind == "anomaly":
        item = ret.anomaly_data(space_id, intent.variables, at)
    else:
        item = ret.current_data(space_id, at)

    if kind == "anomaly" and item["state"] == "unavailable":
        data["not_checked"].append(label)
        return
    if item.get("error"):
        # The detail line reads "3304: sin pronóstico (no hay 12 lecturas
        # consecutivas recientes)", so only the reason is kept, lowercased.
        reason = item["error"].split(", así que")[0].rstrip(".")
        data["errors"].append({"room": label, "reason": reason[:1].lower() + reason[1:]})
        return
    origin = item.get("origin")
    if origin is None:
        data["without_data"].append(label)
        return

    # The origin goes into the name, so every line that names the room
    # carries it: "3201 (estimado)".
    room = label
    if origin == "medido":
        data["n_measured"] += 1
    else:
        room = f"{label} ({origin})"
        if origin not in data["origins"]:
            data["origins"].append(origin)
            data["provenance"][origin] = src.provenance(space_id, origin)
    _append_entry(data, kind, room, item)


def _append_entry(data: dict, kind: str, room: str, item: dict) -> None:
    """Add one room's values to the floor, in the shape its kind uses."""
    variables = data["variables"]
    if kind == "forecast":
        last = item["predictions"][-1] if item["predictions"] else {}
        data["rooms"].append({
            "room": room,
            "current": {v: item["current_reading"].get(v) for v in variables},
            "predicted": {v: ret.number(last.get(v)) for v in variables},
        })
        data["timestamp"] = item.get("timestamp")
        data["horizon_min"] = item.get("horizon_min")
        data.setdefault("methods", [])
        if item.get("method") not in data["methods"]:
            data["methods"].append(item.get("method"))
        data["warning"] = item.get("warning") or data.get("warning")
    elif kind == "history":
        data["rooms"].append({"room": room, "summary": item["summary"]})
        data["since"], data["until"] = item.get("since"), item.get("until")
    elif kind == "anomaly":
        data["rooms"].append({"room": room, "timestamp": item["timestamp"], "anomalies": item["anomalies"]})
    else:
        readings = {v: item["readings"].get(v) for v in variables}
        data["rooms"].append({"room": room, "timestamp": item["timestamp"], "readings": readings})


def _summarize(data: dict, kind: str) -> None:
    """Add the floor's summary, timestamp, method and overall origin."""
    variables = data["variables"]
    if kind == "forecast":
        data["summary"] = _floor_summary(data["rooms"], "predicted", variables)
        data["method"] = ", ".join(data.get("methods", [])) or None
    elif kind == "history":
        data["summary"] = _history_summary(data["rooms"], variables[0])
    elif kind == "anomaly":
        with_anomalies = [r["room"] for r in data["rooms"] if r["anomalies"]]
        data["summary"] = {"rooms_with_anomalies": with_anomalies}
    else:
        data["summary"] = _floor_summary(data["rooms"], "readings", variables)

    if kind in ("current", "anomaly"):
        stamps = [r["timestamp"] for r in data["rooms"] if r["timestamp"]]
        data["timestamp"] = max(stamps) if stamps else None

    if not data["origins"]:
        data["origin"] = "medido"
    elif data["n_measured"]:
        data["origin"] = "mixto"
    else:
        data["origin"] = data["origins"][0] if len(data["origins"]) == 1 else "mixto"


def _floor_summary(rooms: list[dict], key: str, variables: list[str]) -> dict:
    """Mean, minimum and maximum of each variable across the rooms of a floor.

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
