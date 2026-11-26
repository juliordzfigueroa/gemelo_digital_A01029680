"""Humidity of one space: the same mass balance, in absolute humidity.

Section 4.4.2: relative humidity depends on temperature and is not conserved
when air mixes, so the balance is written for the water vapour itself, in
g/m3, with the outdoor air as the boundary condition:

    dρ/dt = λ (ρ_ext − ρ) + g_h N(t)
    reading = ρ + ε, turned back into relative humidity with the room's
              temperature

λ is the CO2's air change rate: the air that carries the CO2 out carries
the water vapour out with it, so the room has one λ, not two. ρ_ext is the
outdoor absolute humidity from Open-Meteo (ingestion.weather). g_h, the
vapour one student adds per hour, and ε are estimated as for the CO2.

The conversions are the Magnus formula of ingestion.weather.absolute_humidity
and its inverse, so the generator and the weather module agree on them.

Usage, from the src/ directory:
    python -m generator.humidity

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-11-26 - Initial implementation.

Reference:
    Section 4.4.2 of the research document.
    WMO (2018). Guide to Instruments and Methods of Observation (WMO-No. 8),
    Annex 4.B.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from generator import noise
from generator.inputs import STEP_HOURS
from ingestion.weather import absolute_humidity


def relative_humidity(rho: np.ndarray, temperature: np.ndarray) -> np.ndarray:
    """Turn absolute humidity back into relative humidity.

    The inverse of ingestion.weather.absolute_humidity.

    Args:
        rho: Water vapour density, g/m3.
        temperature: Air temperature, °C.

    Returns:
        Relative humidity in percent, capped at 100.
    """
    t = np.asarray(temperature, dtype=float)
    saturation_pressure = 6.112 * np.exp(17.62 * t / (243.12 + t))  # hPa
    rh = 100 * np.asarray(rho, dtype=float) * (273.15 + t) / (216.7 * saturation_pressure)
    return np.minimum(rh, 100.0)


def calibrate(grid: pd.DataFrame, lam: float) -> dict:
    """Estimate g_h and ε of one room, given its air change rate.

    Args:
        grid: Output of generator.inputs.room_grid().
        lam: Air change rate per hour, from generator.co2.calibrate().

    Returns:
        {"lambda", "g", "residual"} and how many occupied steps backed g.
    """
    rho = np.asarray(absolute_humidity(grid["temperature"], grid["humidity"]))
    outdoor = grid["abs_humidity_ext"].to_numpy()
    students = grid["students"].to_numpy()
    now, after, people, out = rho[:-1], rho[1:], students[:-1], outdoor[:-1]

    # Least squares for g_h over the occupied steps, as for the CO2.
    decay = np.exp(-lam * STEP_HOURS)
    occupied = ~np.isnan(now) & ~np.isnan(after) & ~np.isnan(out) & (people > 0)
    y = after[occupied] - out[occupied] - (now[occupied] - out[occupied]) * decay
    x = people[occupied] * (1 - decay) / lam
    g = float((x * y).sum() / (x * x).sum()) if occupied.any() else 0.0

    params = {"lambda": lam, "g": max(g, 0.0)}
    params["residual"] = noise.fit(rho - simulate(grid, params))
    params["n_occupied_steps"] = int(occupied.sum())
    return params


def step(rho: float, students: float, outdoor: float, params: dict) -> float:
    """Advance the absolute humidity of one room by one 5-minute step.

    Args:
        rho: Absolute humidity now, g/m3, without ε.
        students: Students in the room during the step.
        outdoor: Outdoor absolute humidity, g/m3.
        params: Output of calibrate().

    Returns:
        Absolute humidity after the step, without ε.
    """
    steady = outdoor + params["g"] * students / params["lambda"]
    decay = np.exp(-params["lambda"] * STEP_HOURS)
    return steady + (rho - steady) * decay


def simulate(grid: pd.DataFrame, params: dict) -> np.ndarray:
    """Run the balance alone over a room's grid, restarting at every gap.

    Args:
        grid: Output of generator.inputs.room_grid().
        params: λ and g, as in calibrate().

    Returns:
        Absolute humidity aligned with the grid, without ε; NaN where there
        was no reading or no weather.
    """
    rho = np.asarray(absolute_humidity(grid["temperature"], grid["humidity"]))
    outdoor = grid["abs_humidity_ext"].to_numpy()
    students = grid["students"].to_numpy()
    output = np.full(len(rho), np.nan)
    value = None
    for t in range(len(rho)):
        if np.isnan(rho[t]):
            value = None
            continue
        if value is None or np.isnan(outdoor[t - 1]):
            # A new stretch, or no weather for the step: restart from the reading.
            value = rho[t]
        else:
            value = step(value, students[t - 1], outdoor[t - 1], params)
        output[t] = value
    return output


def main() -> int:
    """Calibrate every room and compare the balance with the sensor.

    Returns:
        Process exit code.
    """
    from common import config as cfg
    from common import data as dat
    from generator import co2
    from generator.inputs import load_weather, room_grid
    from generator.occupancy import Occupancy

    data = dat.load_data()
    occupancy = Occupancy(data)
    weather = load_weather()

    print(f"{'salón':>6} {'λ (1/h)':>8} {'g_h (g/m3/h/alumno)':>20} {'ε media':>8} "
          f"{'HR medida':>10} {'HR ecuación+μ':>14} {'r':>5}")
    for space in sorted(data[cfg.SPACE_COLUMN].unique()):
        grid = room_grid(space, data, occupancy, weather)
        try:
            lam = co2.calibrate(grid)["lambda"]
        except ValueError:
            print(f"{space:>6}  sin λ del CO2")
            continue
        params = calibrate(grid, lam)
        rho = simulate(grid, params) + params["residual"]["mean"]
        rh = relative_humidity(rho, grid["temperature"].to_numpy())
        measured = grid["humidity"].to_numpy()
        both = ~np.isnan(measured) & ~np.isnan(rh)
        r = np.corrcoef(measured[both], rh[both])[0, 1]
        print(
            f"{space:>6} {lam:>8.2f} {params['g']:>20.4f} {params['residual']['mean']:>8.2f} "
            f"{measured[both].mean():>10.1f} {rh[both].mean():>14.1f} {r:>5.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
