"""Estimates for spaces without a sensor, made by the generator.

Part C of the design: a space without a sensor, of a type that has measured
rooms, is answered with an estimate, labelled as such and with its expected
error. A type never measured (a meeting room) still gets "no sensor": there
is nothing to estimate it from.

The estimate is for the same "now" as the measured rooms, the latest
reading in the data, so a question about the 3201 and one about the 3208
talk about the same moment. The generator runs the room through the last 24
hours with the schedule and the weather of those hours, and the last step is
the estimate; the 24 hours are the history.

The expected error is the one generator/validate.py measured leaving each
room out: the mean, over the measured rooms, of how far the generated
profile of the day was from the measured one. It describes a typical hour,
not the worst reading.

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-11-26 - Initial implementation.

Reference:
    Sections 4.3.4 and 4.4 of the research document.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from common import config as cfg
from common import spaces
from generator.calibrate import load_params
from generator.generate import run_room
from generator.inputs import load_weather
from generator.occupancy import Occupancy
from llm import classifier

VALIDATION_FILE = cfg.MODELS / "generator_validation.json"

# How far back the generator runs to reach "now"; also the history span.
HOURS = 24

# Filled once by the API at startup.
STATE: dict[str, Any] = {
    "calibration": None,
    "occupancy": None,
    "weather": None,
    "error": None,
}


def load(data: pd.DataFrame) -> str | None:
    """Load what the estimates need: calibration, schedule and weather.

    Args:
        data: The cleaned sensor data the API serves.

    Returns:
        None when estimates are available; otherwise the reason they are not,
        and the API keeps answering "no sensor" for those spaces.
    """
    try:
        STATE["calibration"] = load_params()
        STATE["weather"] = load_weather()
    except FileNotFoundError as e:
        return str(e)
    STATE["occupancy"] = Occupancy(data)
    if VALIDATION_FILE.exists():
        STATE["error"] = expected_error(json.loads(VALIDATION_FILE.read_text(encoding="utf-8")))
    return None


def expected_error(validation: dict) -> dict:
    """The typical error of an estimate, per variable.

    Args:
        validation: Content of models/generator_validation.json.

    Returns:
        Per variable, the mean over the rooms of the profile error when the
        room was left out.
    """
    output = {}
    rooms = list(validation.values())
    for variable in rooms[0]["dejando fuera"]:
        output[variable] = float(np.mean([r["dejando fuera"][variable]["profile_mae"] for r in rooms]))
    return output


def available(space_id: int) -> bool:
    """Whether a space without readings can be estimated.

    Args:
        space_id: Space identifier.

    Returns:
        True when the generator was calibrated and gives that space its
        type's parameters.
    """
    calibration = STATE["calibration"]
    if calibration is None:
        return False
    return calibration["rooms"].get(str(space_id), {}).get("origin") == "estimado"


def estimate(intent: classifier.Intent, space_id: int, now: pd.Timestamp) -> dict:
    """Estimate the present state, or the last day, of a space without a sensor.

    Args:
        intent: The classified question: "current" or "history".
        space_id: A space for which available() is True.
        now: The latest reading in the data, the "now" of every answer.

    Returns:
        A dict shaped like the measured ones (readings, or a summary), plus
        "origin": "estimado", the rooms its type was calibrated with and the
        expected error.
    """
    room = STATE["calibration"]["rooms"][str(space_id)]
    stamps = pd.date_range(now - pd.Timedelta(hours=HOURS), now, freq=f"{cfg.INTERVAL_SECONDS}s")
    students = STATE["occupancy"].students(space_id, stamps)
    weather = STATE["weather"].reindex(stamps).ffill()
    # Seeded by space and moment, so the same question gets the same answer.
    rng = np.random.default_rng([cfg.SEED, space_id, int(now.timestamp())])
    frame = run_room(space_id, room["params"], stamps, students, weather, rng)

    labels = spaces.labels()
    based_on = STATE["calibration"]["types"][room["kind"]]["from_rooms"]
    data = {
        "space": labels.get(space_id, space_id),
        "origin": "estimado",
        "based_on": [str(labels.get(s, s)) for s in based_on],
        "expected_error": STATE["error"],
    }
    if intent.kind == "history":
        variable = intent.variables[0] if intent.variables else "co2"
        series = frame[variable]
        data.update({
            "variable": variable,
            "since": str(stamps[0]),
            "until": str(now),
            "summary": {
                "max": round(float(series.max()), 2),
                "min": round(float(series.min()), 2),
                "mean": round(float(series.mean()), 2),
                "n_readings": int(len(series)),
            },
        })
    else:
        last = frame.iloc[-1]
        data["timestamp"] = str(now)
        data["readings"] = {v: round(float(last[v]), 2) for v in cfg.ACTIVE_VARIABLES}
    return data
