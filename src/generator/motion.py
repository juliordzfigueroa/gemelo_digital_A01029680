"""Motion of one space: derived from N(t), with the sensor's error rates.

Section 4.4.2: there is motion when N(t) > 0, with a false-positive and a
false-negative rate estimated from the real readings. The sensor is a PIR,
and after cleaning each interval holds the fraction of its readings that saw
motion, a value in [0, 1]. So the model decides, every 5 minutes, whether
the PIR fired at all, and how much when it did:

    P(fired | class)      1 minus the false-negative rate
    P(fired | no class)   the false-positive rate
    level | fired         the mean fraction when it fires, with and without class

The PIR misses a lot: even with a full class it fires in only some of the
intervals, which is why the rates are measured instead of assumed.

Usage, from the src/ directory:
    python -m generator.motion

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-11-26 - Initial implementation.

Reference:
    Section 4.4.2 of the research document.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def calibrate(grid: pd.DataFrame) -> dict:
    """Measure how often and how much the PIR fires, with and without class.

    Args:
        grid: Output of generator.inputs.room_grid().

    Returns:
        {"p_class", "p_empty", "level_class", "level_empty"}.
    """
    motion = grid["motion"].to_numpy()
    in_class = grid["students"].to_numpy() > 0
    valid = ~np.isnan(motion)

    params = {}
    for name, mask in (("class", valid & in_class), ("empty", valid & ~in_class)):
        values = motion[mask]
        fired = values[values > 0]
        params[f"p_{name}"] = float(len(fired) / len(values)) if len(values) else 0.0
        params[f"level_{name}"] = float(fired.mean()) if len(fired) else 0.0
    return params


def step(students: float, params: dict, rng) -> float:
    """Draw the motion fraction of one 5-minute interval.

    Args:
        students: Students in the room during the interval.
        params: Output of calibrate().
        rng: A numpy Generator.

    Returns:
        The fraction of the interval with motion detected, in [0, 1].
    """
    name = "class" if students > 0 else "empty"
    if rng.random() < params[f"p_{name}"]:
        return params[f"level_{name}"]
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

    print(f"{'salón':>6} {'P(dispara|clase)':>17} {'P(dispara|vacío)':>17} {'nivel clase':>12} "
          f"{'nivel vacío':>12} {'media med/gen':>14}")
    for space in sorted(data[cfg.SPACE_COLUMN].unique()):
        grid = room_grid(space, data, occupancy, weather)
        if grid["motion"].notna().sum() < 24 * 12:
            print(f"{space:>6}  menos de un día de lecturas")
            continue
        params = calibrate(grid)
        measured = grid["motion"].to_numpy()
        valid = ~np.isnan(measured)
        generated = np.array([step(n, params, rng) for n in grid["students"].to_numpy()[valid]])
        print(
            f"{space:>6} {params['p_class']:>17.2f} {params['p_empty']:>17.3f} "
            f"{params['level_class']:>12.3f} {params['level_empty']:>12.3f} "
            f"{measured[valid].mean():>6.3f}/{generated.mean():<6.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
