"""CO2 of one space: the mass balance of section 4.4.2.

    dC/dt = λ (C_ext − C) + g N(t)
    reading = C + ε

λ is the air change rate (Q/V in the document) and g is the CO2 that one
student adds per hour, in ppm (G/V in the document). The volumes are not
known, so g is estimated directly from the readings as one effective number.
Being effective, it also absorbs how many enrolled students actually attend
and how the metal-oxide sensor responds. ε is what the equation does not
explain, mostly the drift of that sensor (generator/noise.py).

Calibration, per measured room:

  - λ and C_ext from the decay after the room empties, as in ASTM E741:
    every run of falling CO2 with no one inside is fitted to
    C(t) = C_ext + (C0 − C_ext) e^(−λt), and the median of the good fits is
    kept. C_ext is fitted too, instead of fixed at 400 ppm, because the gas
    sensor drifts.
  - g by least squares over the occupied intervals, once λ is known.
  - ε as an AR(1) fitted to what the equation leaves unexplained.

One step uses the exact solution of the equation over 5 minutes, with N(t)
held constant during the step. It is the same exponential the decay fit
uses, now towards the level the occupants hold the room at:

    C_ss = C_ext + g N / λ
    C(t + Δt) = C_ss + (C(t) − C_ss) e^(−λ Δt)

Usage, from the src/ directory:
    python -m generator.co2

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-11-26 - Initial implementation.

Reference:
    Section 4.4.2 of the research document.
    ASTM International (2024). ASTM E741: Standard test method for
    determining air change in a single zone by means of a tracer gas dilution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from common import config as cfg
from generator import noise
from generator.inputs import STEP_HOURS

# A decay counts if CO2 falls for at least this many steps (30 minutes) in a
# row, by more than MIN_DROP ppm, with no students and no motion.
MIN_DECAY_STEPS = 6
MIN_DROP = 50.0
NO_MOTION = 0.01

# Only decays the exponential describes well enter the median.
MIN_R2 = 0.9

# Physical range for the fitted outdoor level: below 350 ppm is not air, and
# above 800 ppm the "outdoor" level would be the drifting sensor.
C_EXT_RANGE = (350.0, 800.0)

FLOOR = cfg.PHYSICAL_LIMITS["co2"][0]


def calibrate(grid: pd.DataFrame) -> dict:
    """Estimate λ, C_ext, g and ε of one room.

    Args:
        grid: Output of generator.inputs.room_grid().

    Returns:
        The parameters, plus how many decays and occupied steps backed them.

    Raises:
        ValueError: If the room has no usable decay, so λ cannot be measured.
    """
    fits = decay_fits(grid)
    if not fits:
        raise ValueError("no usable CO2 decay: λ cannot be measured for this room")
    lam = float(np.median([f["lambda"] for f in fits]))
    c_ext = float(np.median([f["c_ext"] for f in fits]))

    co2 = grid["co2"].to_numpy()
    students = grid["students"].to_numpy()
    now, after, people = co2[:-1], co2[1:], students[:-1]

    # Least squares for g over the occupied steps: what the step adds beyond
    # the decay, against what N students would add.
    decay = np.exp(-lam * STEP_HOURS)
    occupied = ~np.isnan(now) & ~np.isnan(after) & (people > 0)
    y = after[occupied] - c_ext - (now[occupied] - c_ext) * decay
    x = people[occupied] * (1 - decay) / lam
    g = float((x * y).sum() / (x * x).sum()) if occupied.any() else 0.0

    params = {"lambda": lam, "c_ext": c_ext, "g": max(g, 0.0)}
    params["residual"] = noise.fit(co2 - simulate(grid, params))
    params["n_decays"] = len(fits)
    params["n_occupied_steps"] = int(occupied.sum())
    return params


def decay_fits(grid: pd.DataFrame) -> list[dict]:
    """Fit the exponential to every clean decay of the room.

    Args:
        grid: Output of generator.inputs.room_grid().

    Returns:
        One dict per decay that the exponential describes with R² above
        MIN_R2: its λ (per hour), its C_ext and its R².
    """
    co2 = grid["co2"].to_numpy()
    empty = (grid["students"].to_numpy() == 0) & (grid["motion"].to_numpy() <= NO_MOTION)
    falling = np.zeros(len(co2), dtype=bool)
    falling[1:] = (np.diff(co2) < 0) & empty[1:]

    fits = []
    i = 0
    while i < len(co2):
        if not falling[i]:
            i += 1
            continue
        end = i
        while end < len(co2) and falling[end]:
            end += 1
        # The run starts at the peak, one step before the first fall.
        run = co2[i - 1 : end]
        if len(run) > MIN_DECAY_STEPS and run[0] - run[-1] > MIN_DROP:
            fit = _fit_decay(run)
            if fit and fit["r2"] > MIN_R2:
                fits.append(fit)
        i = end
    return fits


def _fit_decay(run: np.ndarray) -> dict | None:
    """Fit C(t) = C_ext + (C0 − C_ext) e^(−λt) to one decay.

    Args:
        run: CO2 readings of the decay, one per step.

    Returns:
        {"lambda", "c_ext", "r2"}, or None if the fit does not converge.
    """
    t = np.arange(len(run)) * STEP_HOURS

    def curve(t, c_ext, amplitude, lam):
        return c_ext + amplitude * np.exp(-lam * t)

    low, high = C_EXT_RANGE
    start = [min(max(run[-1] - 20, low + 1), high - 1), run[0] - run[-1], 1.0]
    try:
        p, _ = curve_fit(curve, t, run, p0=start, bounds=([low, 0, 0.01], [high, 5000, 20]), maxfev=5000)
    except RuntimeError:
        return None
    residual = run - curve(t, *p)
    r2 = 1 - (residual**2).sum() / ((run - run.mean()) ** 2).sum()
    return {"lambda": float(p[2]), "c_ext": float(p[0]), "r2": float(r2)}


def step(c: float, students: float, params: dict) -> float:
    """Advance the CO2 of one room by one 5-minute step of the equation.

    Args:
        c: CO2 now, ppm, without ε.
        students: Students in the room during the step.
        params: Output of calibrate().

    Returns:
        CO2 after the step, without ε.
    """
    steady = params["c_ext"] + params["g"] * students / params["lambda"]
    decay = np.exp(-params["lambda"] * STEP_HOURS)
    return steady + (c - steady) * decay


def reading(c: float, e: float) -> float:
    """What the sensor reads: the equation plus ε, never below its 400 ppm floor."""
    return max(c + e, FLOOR)


def simulate(grid: pd.DataFrame, params: dict) -> np.ndarray:
    """Run the equation alone over a room's grid, restarting at every gap.

    Each stretch of readings starts from its first measured value and is then
    produced by the equation alone, so the comparison with the sensor shows
    what the model reproduces, not what it copies.

    Args:
        grid: Output of generator.inputs.room_grid().
        params: λ, C_ext and g, as in calibrate().

    Returns:
        CO2 aligned with the grid, without ε; NaN where there was no reading.
    """
    co2 = grid["co2"].to_numpy()
    students = grid["students"].to_numpy()
    output = np.full(len(co2), np.nan)
    c = None
    for t in range(len(co2)):
        if np.isnan(co2[t]):
            c = None
            continue
        c = co2[t] if c is None else step(c, students[t - 1], params)
        output[t] = c
    return output


def main() -> int:
    """Calibrate every room with CO2 and compare what it generates with the sensor.

    Returns:
        Process exit code.
    """
    from common import data as dat
    from generator.inputs import load_weather, room_grid
    from generator.occupancy import Occupancy

    data = dat.load_data()
    occupancy = Occupancy(data)
    weather = load_weather()
    rng = np.random.default_rng(cfg.SEED)

    print("Parámetros por salón")
    print(f"  {'salón':>6} {'caídas':>6} {'λ (1/h)':>8} {'C_ext':>6} {'g (ppm/h/alumno)':>17} "
          f"{'ε media':>8} {'ε φ':>6} {'ε σ':>5}")
    rooms = {}
    for space in sorted(data[cfg.SPACE_COLUMN].unique()):
        grid = room_grid(space, data, occupancy, weather)
        try:
            params = calibrate(grid)
        except ValueError as e:
            print(f"  {space:>6}  {e}")
            continue
        rooms[space] = (grid, params)
        r = params["residual"]
        print(
            f"  {space:>6} {params['n_decays']:>6} {params['lambda']:>8.2f} {params['c_ext']:>6.0f} "
            f"{params['g']:>17.1f} {r['mean']:>8.0f} {r['phi']:>6.3f} {r['sigma']:>5.1f}"
        )

    print("\nMedido contra generado (20 realizaciones con ε), en ppm")
    print(f"  {'salón':>6} {'p10 med/gen':>12} {'p50 med/gen':>12} {'p90 med/gen':>12} "
          f"{'en clase med/gen':>17} {'perfil del día r':>17}")
    for space, (grid, params) in rooms.items():
        measured = grid["co2"].to_numpy()
        runs = [_generate(grid, params, rng) for _ in range(20)]
        generated = np.nanmean(runs, axis=0)
        both = ~np.isnan(measured)
        q_m = np.percentile(measured[both], [10, 50, 90])
        q_g = np.percentile(np.concatenate([r[both] for r in runs]), [10, 50, 90])
        in_class = both & (grid["students"].to_numpy() > 0)
        profile = pd.DataFrame({"m": measured, "g": generated}, index=grid.index).groupby(grid.index.hour).mean()
        print(
            f"  {space:>6} {q_m[0]:5.0f}/{q_g[0]:<5.0f} {q_m[1]:5.0f}/{q_g[1]:<5.0f} {q_m[2]:5.0f}/{q_g[2]:<5.0f} "
            f"{measured[in_class].mean():8.0f}/{generated[in_class].mean():<7.0f} "
            f"{np.corrcoef(profile['m'], profile['g'])[0, 1]:17.2f}"
        )
    return 0


def _generate(grid: pd.DataFrame, params: dict, rng) -> np.ndarray:
    """One realization with ε over the measured stretches, for main()."""
    co2 = grid["co2"].to_numpy()
    students = grid["students"].to_numpy()
    output = np.full(len(co2), np.nan)
    c = e = None
    for t in range(len(co2)):
        if np.isnan(co2[t]):
            c = None
            continue
        if c is None:
            c, e = co2[t], noise.start(params["residual"], rng)
        else:
            c = step(c, students[t - 1], params)
            e = noise.step(e, params["residual"], rng)
        output[t] = reading(c, e)
    return output


if __name__ == "__main__":
    raise SystemExit(main())
