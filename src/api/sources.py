"""Where the rows of an answer come from: the sensors, or the generator.

Part C of the design: every value the user sees carries its origin. The rule
is decided per answer, so an answer never mixes origins:

  medido     the sensor of the space measured the moment, or the period,
             the question asks about.
  simulado   a measured room, at a moment its sensor did not measure (before
             the campaign, after it, a dead sensor): the generator, with that
             room's own parameters.
  estimado   a room without a sensor, of a type that has measured rooms: the
             generator, with its type's parameters.

A type that was never measured (a meeting room) has no generated rows, and
the answer stays "no sensor".

The generated rows come from data/processed/generated.parquet, written by
`python -m generator.generate`. The expected error of every generated value
is the leave-one-out error of generator/validate.py. For a simulated value
that is conservative on purpose: the error with a room's own parameters was
measured on the same days it was calibrated with, and would be optimistic
(it gave the temperature ±0.1 °C).

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-11-26 - Initial implementation; replaces the 24-hour estimates.

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

VALIDATION_FILE = cfg.MODELS / "generator_validation.json"

# Filled once by the API at startup.
STATE: dict[str, Any] = {
    "generated": None,  # the generated rows, with their origin
    "calibration": None,  # models/generator_params.json
    "error": None,  # variable -> typical error of a generated value
}


def load() -> str | None:
    """Load the generated rows, the calibration and the expected errors.

    Returns:
        None when the generator can answer; otherwise the reason it cannot,
        and only measured data is used.
    """
    if not cfg.GENERATED_DATA.exists():
        return (
            f"{cfg.GENERATED_DATA} does not exist; generate the semester with "
            "`python -m generator.generate`"
        )
    try:
        STATE["calibration"] = load_params()
    except FileNotFoundError as e:
        return str(e)
    STATE["generated"] = pd.read_parquet(cfg.GENERATED_DATA)
    if VALIDATION_FILE.exists():
        validation = json.loads(VALIDATION_FILE.read_text(encoding="utf-8"))
        STATE["error"] = _mean_error(validation, "dejando fuera")
    return None


def _mean_error(validation: dict, case: str) -> dict:
    """Per variable, the mean over the rooms of the profile error of one case."""
    rooms = list(validation.values())
    output = {}
    for variable in rooms[0][case]:
        output[variable] = float(np.mean([r[case][variable]["profile_mae"] for r in rooms]))
    return output


def generated_rows(space_id: int) -> pd.DataFrame:
    """Every generated row of one space, sorted by time; empty if none."""
    generated = STATE["generated"]
    if generated is None:
        return pd.DataFrame()
    own = generated[generated[cfg.SPACE_COLUMN] == space_id]
    return own.sort_values(cfg.TIME_COLUMN).reset_index(drop=True)


def covered() -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """First and last generated interval, or None without generated rows."""
    generated = STATE["generated"]
    if generated is None or generated.empty:
        return None
    return generated[cfg.TIME_COLUMN].min(), generated[cfg.TIME_COLUMN].max()


def provenance(space_id: int, origin: str) -> dict:
    """What a generated answer must carry: where it comes from and its error.

    Args:
        space_id: Space identifier.
        origin: "simulado" or "estimado".

    Returns:
        {"based_on": labels of the rooms the parameters came from,
         "expected_error": typical error per variable}.
    """
    labels = spaces.labels()
    room = STATE["calibration"]["rooms"].get(str(space_id), {})
    if origin == "simulado":
        based_on = [space_id]
    else:
        based_on = STATE["calibration"]["types"].get(room.get("kind"), {}).get("from_rooms", [])
    return {
        "based_on": [str(labels.get(s, s)) for s in based_on],
        "expected_error": STATE["error"],
    }
