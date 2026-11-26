"""Motion of one space: derived from N(t), with the sensor's error rates.

Section 4.4.2: there is motion when N(t) > 0, with a false-positive and a
false-negative rate estimated from the real readings. The sensor is a PIR,
and after cleaning each interval holds the fraction of its readings that saw
motion, a value in [0, 1]. So the model decides, every 5 minutes, whether
the PIR fired at all, and how much when it did.

Both depend on the situation, because "no class" is not one thing:

  class    a block of the schedule is on: 1 minus the false-negative rate.
  open     a weekday from 7:00 to 22:00 without class: people come in
           between classes, so the PIR fires now and then.
  closed   nights and weekends: the false-positive rate of the sensor alone.

Pooling the last two, as a first version did, put the between-classes
rate on the weekends, and the generated rooms moved on empty Sundays.

When the PIR fires, the fraction is drawn from the ones it measured in that
situation (stored as ten quantiles), not fixed at their mean.

Usage, from the src/ directory:
    python -m generator.motion

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-11-26 - Three situations, and the level drawn from the measured ones.

Reference:
    Section 4.4.2 of the research document.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ingestion.schedule import CLOSING_HOUR, OPENING_HOUR

SITUATIONS = ("class", "open", "closed")

# The ten quantiles kept of the fractions the PIR read when it fired.
QUANTILES = np.arange(5, 100, 10)


def situation(stamp: pd.Timestamp, students: float) -> str:
    """Which of the three situations an interval is in.

    Args:
        stamp: Start of the interval.
        students: Students in the room during it.

    Returns:
        "class", "open" or "closed".
    """
    if students > 0:
        return "class"
    if stamp.dayofweek < 5 and OPENING_HOUR <= stamp.hour < CLOSING_HOUR:
        return "open"
    return "closed"


def calibrate(grid: pd.DataFrame) -> dict:
    """Measure how often and how much the PIR fires in each situation.

    Args:
        grid: Output of generator.inputs.room_grid().

    Returns:
        For each situation, {"p": chance of firing, "levels": ten quantiles
        of the fraction it read when it fired}.
    """
    motion = grid["motion"].to_numpy()
    students = grid["students"].to_numpy()
    labels = np.array([situation(t, n) for t, n in zip(grid.index, students)])
    valid = ~np.isnan(motion)

    params = {}
    for name in SITUATIONS:
        values = motion[valid & (labels == name)]
        fired = values[values > 0]
        params[name] = {
            "p": float(len(fired) / len(values)) if len(values) else 0.0,
            "levels": [float(v) for v in np.percentile(fired, QUANTILES)] if len(fired) else [0.0],
        }
    return params


def step(stamp: pd.Timestamp, students: float, params: dict, rng) -> float:
    """Draw the motion fraction of one 5-minute interval.

    Args:
        stamp: Start of the interval.
        students: Students in the room during it.
        params: Output of calibrate().
        rng: A numpy Generator.

    Returns:
        The fraction of the interval with motion detected, in [0, 1].
    """
    p = params[situation(stamp, students)]
    if rng.random() < p["p"]:
        return float(rng.choice(p["levels"]))
    return 0.0


def main() -> int:
    """Calibrate every room and compare the generated motion with the sensor.

    Returns:
        Process exit code.
    """
    from common import config as cfg
    from common import data as dat
    from generator.inputs import load_weather, room_grid
    from generator.occupancy import Occupancy

    data = dat.load_data()
    occupancy = Occupancy(data)
    weather = load_weather()
    rng = np.random.default_rng(cfg.SEED)

    print(f"{'salón':>6} {'P clase':>8} {'P abierto':>10} {'P cerrado':>10}   "
          f"{'media med/gen':>14} {'fin de semana med/gen':>22}")
    for space in sorted(data[cfg.SPACE_COLUMN].unique()):
        grid = room_grid(space, data, occupancy, weather)
        if grid["motion"].notna().sum() < 24 * 12:
            print(f"{space:>6}  menos de un día de lecturas")
            continue
        params = calibrate(grid)
        measured = grid["motion"].to_numpy()
        generated = np.array([step(t, n, params, rng) for t, n in zip(grid.index, grid["students"])])
        valid = ~np.isnan(measured)
        weekend = valid & (grid.index.dayofweek.to_numpy() >= 5)
        print(
            f"{space:>6} {params['class']['p']:>8.2f} {params['open']['p']:>10.3f} "
            f"{params['closed']['p']:>10.3f}   {measured[valid].mean():>6.3f}/{generated[valid].mean():<7.3f} "
            f"{measured[weekend].mean():>13.4f}/{generated[weekend].mean():<8.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
