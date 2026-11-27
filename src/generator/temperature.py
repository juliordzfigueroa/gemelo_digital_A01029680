"""Temperature of one space: an empirical model, not a heat balance.

Section 4.4.2: the air conditioning dominates the room's temperature (its
correlation with the outdoor one was 0.24), so instead of a thermal balance
each room gets its typical temperature by hour of day plus the effect of
its occupants:

    T(t) = T̄(s, h) + β_s N(t) + ε

T̄(s, h) is the typical temperature of room s at hour h, β_s what each
student adds, and ε the residual noise (generator/noise.py).

Calibration is one least-squares fit per room: one column per hour of the
day, which gives the 24 values of T̄, and one column with N(t), which gives
β.

Usage, from the src/ directory:
    python -m generator.temperature

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

from generator import noise


def calibrate(grid: pd.DataFrame) -> dict:
    """Estimate T̄ for every hour, β and ε of one room.

    Args:
        grid: Output of generator.inputs.room_grid().

    Returns:
        {"by_hour": 24 values, "beta", "residual"}.
    """
    temperature = grid["temperature"].to_numpy()
    students = grid["students"].to_numpy()
    hours = grid.index.hour.to_numpy()
    valid = ~np.isnan(temperature)

    # One column per hour (1 when the reading is from that hour) and one with
    # the students: the least-squares solution is T̄ for each hour, then β.
    columns = np.zeros((valid.sum(), 25))
    columns[np.arange(valid.sum()), hours[valid]] = 1.0
    columns[:, 24] = students[valid]
    solution, *_ = np.linalg.lstsq(columns, temperature[valid], rcond=None)

    # An hour with no readings at all has no estimate; it takes the mean of
    # the hours that do.
    by_hour = solution[:24]
    seen = np.bincount(hours[valid], minlength=24) > 0
    by_hour[~seen] = by_hour[seen].mean()

    params = {"by_hour": [float(v) for v in by_hour], "beta": float(solution[24])}
    params["residual"] = noise.fit(temperature - expected(grid.index, students, params))
    return params


def expected(stamps, students, params: dict) -> np.ndarray:
    """T̄(s, h) + β N(t): the temperature the model expects, without ε.

    Args:
        stamps: Interval start times.
        students: Students at each of them.
        params: Output of calibrate().

    Returns:
        One temperature per stamp, °C.
    """
    hours = pd.DatetimeIndex(stamps).hour.to_numpy()
    return np.asarray(params["by_hour"])[hours] + params["beta"] * np.asarray(students)


def main() -> int:
    """Calibrate every room and compare the model with the sensor.

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

    print(f"{'salón':>6} {'T̄ mín':>6} {'T̄ máx':>6} {'β (°C/alumno)':>14} {'ε φ':>6} {'ε σ':>6} "
          f"{'MAE':>5} {'r':>5}")
    for space in sorted(data[cfg.SPACE_COLUMN].unique()):
        grid = room_grid(space, data, occupancy, weather)
        if grid["temperature"].notna().sum() < 24 * 12:
            print(f"{space:>6}  menos de un día de lecturas")
            continue
        params = calibrate(grid)
        model = expected(grid.index, grid["students"].to_numpy(), params)
        measured = grid["temperature"].to_numpy()
        both = ~np.isnan(measured)
        mae = np.abs(model[both] - measured[both]).mean()
        r = np.corrcoef(model[both], measured[both])[0, 1]
        res = params["residual"]
        print(
            f"{space:>6} {min(params['by_hour']):>6.1f} {max(params['by_hour']):>6.1f} "
            f"{params['beta']:>14.3f} {res['phi']:>6.3f} {res['sigma']:>6.3f} {mae:>5.2f} {r:>5.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
