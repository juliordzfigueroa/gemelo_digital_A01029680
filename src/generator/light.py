"""Light of one space: a state model, a stochastic grey box.

Section 4.4.2: light does not accumulate like CO2. It is what comes in
through the windows plus what the lamps give, and the lamps depend on what
people do. So it is a model of states:

    L(t) = ℓ[s(t)] + c(t) k R(t) + ε(t)

s(t)   lamps off or on; a Markov chain whose chance of switching depends on
       the hour and on whether there is class.
ℓ[s]   the reading each state gives.
k      the room's daylight factor, in % of the sensor per kW/m2 of outdoor
       solar radiation R(t), from Open-Meteo.
c(t)   the curtains or projector factor: 1 without class, c during class.
       It multiplies the daylight instead of subtracting from it, so the
       model never produces negative light.
ε(t)   the residual noise (generator/noise.py).

Every parameter comes from the room's own readings, as the document says:

  - ℓ from the nights, when there is no sun and the sensor sees only the
    lamps: the readings split cleanly into "off" (around 0 %) and "on".
  - k from weekends, without classes, as the median of light over
    radiation. The median, because a lamp someone left on would drag a
    least-squares fit, and did: it gave the windowless 3206 a k of 7.6.
  - c from class hours with sun, by least squares, once ℓ and k are known.
  - The switching chances by counting the state changes, per hour, with
    and without class.

Usage, from the src/ directory:
    python -m generator.light

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-11-26 - Initial implementation.

Reference:
    Section 4.4.2 of the research document.
    Page, J., Robinson, D., Morel, N., & Scartezzini, J.-L. (2008). A
    generalised stochastic model for the simulation of occupant presence.
    Energy and Buildings, 40(2), 83-98.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from generator import noise

# Above this reading, with no sun, the lamps are on.
LAMP_THRESHOLD = 10.0

# Readings needed to trust a level or a count; below that, a fallback is used.
MIN_READINGS = 12

# Sun strong enough to measure k and c, in kW/m2.
MIN_SUN = 0.1

# Weight, in steps, of the overall switching chance when an hour has few
# observations: with one week of data, an hour seen a handful of times
# falls back on the overall chance for that situation.
SMOOTHING = 12.0

SITUATIONS = ("class", "empty")


def calibrate(grid: pd.DataFrame) -> dict:
    """Estimate ℓ, k, c, the switching chances and ε of one room.

    Args:
        grid: Output of generator.inputs.room_grid().

    Returns:
        {"levels", "k", "c", "switch_on", "switch_off", "residual"}.
    """
    light = grid["light"].to_numpy()
    sun = grid["radiation_ext"].to_numpy() / 1000
    in_class = grid["students"].to_numpy() > 0
    valid = ~np.isnan(light) & ~np.isnan(sun)

    params = {"levels": lamp_levels(light, sun, valid), "k": daylight_factor(grid, light, sun, valid)}
    params["c"] = 1.0
    params["c"] = curtain_factor(light, sun, in_class, valid, params)

    states = infer_states(light, sun, in_class, params)
    params.update(switching(states, grid.index.hour.to_numpy(), in_class, valid))
    params["residual"] = noise.fit(np.where(valid, light - expected(states, sun, in_class, params), np.nan))
    return params


def lamp_levels(light: np.ndarray, sun: np.ndarray, valid: np.ndarray) -> list[float]:
    """The reading with the lamps off and on, measured at night.

    Returns:
        [off, on]. A room whose lamps are rarely on at night (the 3305) gets
        the threshold itself as its "on" level, the most cautious guess.
    """
    night = valid & (sun == 0)
    off = light[night & (light <= LAMP_THRESHOLD)]
    on = light[night & (light > LAMP_THRESHOLD)]
    level_off = float(np.median(off)) if len(off) else 0.0
    level_on = float(np.median(on)) if len(on) >= MIN_READINGS else LAMP_THRESHOLD
    return [level_off, level_on]


def daylight_factor(grid: pd.DataFrame, light, sun, valid) -> float:
    """k: the median of light over radiation on sunny weekend readings."""
    weekend = valid & (grid.index.dayofweek.to_numpy() >= 5) & (sun > MIN_SUN)
    if weekend.sum() < MIN_READINGS:
        return 0.0
    return float(np.median(light[weekend] / sun[weekend]))


def curtain_factor(light, sun, in_class, valid, params: dict) -> float:
    """c: how much of the daylight reaches the sensor during class.

    Least squares over the sunny class readings of light minus the lamps,
    against the daylight k R. Kept in [0, 1]; a room without windows has
    no daylight to dim and keeps 1.
    """
    sunny = valid & in_class & (sun > MIN_SUN)
    daylight = params["k"] * sun[sunny]
    if sunny.sum() < MIN_READINGS or (daylight**2).sum() < 1.0:
        return 1.0
    states = infer_states(light, sun, in_class, params)[sunny].astype(int)
    lamps = np.asarray(params["levels"])[states]
    c = ((light[sunny] - lamps) * daylight).sum() / (daylight**2).sum()
    return float(min(max(c, 0.0), 1.0))


def infer_states(light, sun, in_class, params: dict) -> np.ndarray:
    """Lamps off (0) or on (1) at each reading: what is left after the daylight.

    Halfway between the two lamp levels is the line. NaN readings come out
    as off; they are excluded wherever states are counted.
    """
    daylight = np.where(in_class, params["c"], 1.0) * params["k"] * sun
    halfway = (params["levels"][0] + params["levels"][1]) / 2
    return np.nan_to_num(light - daylight, nan=0.0) > halfway


def switching(states, hours, in_class, valid) -> dict:
    """The chances, per hour, of the lamps going on and off, with and without class.

    Returns:
        {"switch_on": {"class": 24 values, "empty": 24}, "switch_off": same}.
        switch_on is P(on next | off now); switch_off, P(off next | on now).
    """
    output = {"switch_on": {}, "switch_off": {}}
    pairs = valid[:-1] & valid[1:]
    now, after = states[:-1].astype(int), states[1:].astype(int)
    for situation in SITUATIONS:
        mask = pairs & (in_class[:-1] if situation == "class" else ~in_class[:-1])
        for name, start in (("switch_on", 0), ("switch_off", 1)):
            from_start = mask & (now == start)
            switched = from_start & (after != start)
            overall = switched.sum() / from_start.sum() if from_start.sum() else 0.0
            by_hour = []
            for hour in range(24):
                seen = from_start & (hours[:-1] == hour)
                by_hour.append(float((switched[seen].sum() + SMOOTHING * overall) / (seen.sum() + SMOOTHING)))
            output[name][situation] = by_hour
    return output


def expected(states, sun, in_class, params: dict) -> np.ndarray:
    """ℓ[s] + c k R: the light the model expects, without ε."""
    lamps = np.asarray(params["levels"])[np.asarray(states, dtype=int)]
    return lamps + np.where(in_class, params["c"], 1.0) * params["k"] * sun


def next_state(state: bool, hour: int, in_class: bool, params: dict, rng) -> bool:
    """Draw the lamps' state for the next 5 minutes.

    Args:
        state: True if the lamps are on now.
        hour: Hour of day of the step.
        in_class: Whether there is class during the step.
        params: Output of calibrate().
        rng: A numpy Generator.

    Returns:
        True if the lamps are on after the step.
    """
    situation = "class" if in_class else "empty"
    name = "switch_off" if state else "switch_on"
    if rng.random() < params[name][situation][hour]:
        return not state
    return state


def reading(value: float, e: float) -> float:
    """What the sensor reads: the model plus ε, kept within 0 to 100 %."""
    return min(max(value + e, 0.0), 100.0)


def main() -> int:
    """Calibrate every room and compare the generated light with the sensor.

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

    print(f"{'salón':>6} {'ℓ apag.':>7} {'ℓ enc.':>7} {'k':>6} {'c':>5} {'p50 med/gen':>12} "
          f"{'p90 med/gen':>12} {'encendido med/gen':>18} {'perfil r':>8}")
    for space in sorted(data[cfg.SPACE_COLUMN].unique()):
        grid = room_grid(space, data, occupancy, weather)
        if grid["light"].notna().sum() < 24 * 12:
            print(f"{space:>6}  menos de un día de lecturas")
            continue
        params = calibrate(grid)
        measured = grid["light"].to_numpy()
        runs = [_generate(grid, params, rng) for _ in range(20)]
        both = ~np.isnan(measured)
        q_m = np.percentile(measured[both], [50, 90])
        q_g = np.nanpercentile(np.concatenate([r[both] for r in runs]), [50, 90])
        on_m, on_g = _mean_on_hours(grid, params, measured), np.mean([_mean_on_hours(grid, params, r) for r in runs])
        profile = pd.DataFrame({"m": measured, "g": np.nanmean(runs, axis=0)}, index=grid.index)
        profile = profile.groupby(grid.index.hour).mean()
        print(
            f"{space:>6} {params['levels'][0]:>7.1f} {params['levels'][1]:>7.1f} {params['k']:>6.1f} "
            f"{params['c']:>5.2f} {q_m[0]:5.1f}/{q_g[0]:<5.1f} {q_m[1]:5.1f}/{q_g[1]:<5.1f} "
            f"{on_m:8.1f}/{on_g:<5.1f} h {np.corrcoef(profile['m'], profile['g'])[0, 1]:8.2f}"
        )
    return 0


def _generate(grid: pd.DataFrame, params: dict, rng) -> np.ndarray:
    """One realization over the measured stretches, for main()."""
    measured = grid["light"].to_numpy()
    sun = grid["radiation_ext"].to_numpy() / 1000
    in_class = grid["students"].to_numpy() > 0
    hours = grid.index.hour.to_numpy()
    output = np.full(len(measured), np.nan)
    state = e = None
    for t in range(len(measured)):
        if np.isnan(measured[t]) or np.isnan(sun[t]):
            state = None
            continue
        if state is None:
            state = bool(infer_states(measured[t : t + 1], sun[t : t + 1], in_class[t : t + 1], params)[0])
            e = noise.start(params["residual"], rng)
        else:
            state = next_state(state, hours[t], in_class[t], params, rng)
            e = noise.step(e, params["residual"], rng)
        output[t] = reading(expected([state], sun[t : t + 1], in_class[t : t + 1], params)[0], e)
    return output


def _mean_on_hours(grid: pd.DataFrame, params: dict, light: np.ndarray) -> float:
    """Mean length of the periods with lamps on, in hours, for main()."""
    sun = grid["radiation_ext"].to_numpy() / 1000
    states = infer_states(light, sun, grid["students"].to_numpy() > 0, params) & ~np.isnan(light)
    lengths, run = [], 0
    for on in states:
        if on:
            run += 1
        elif run:
            lengths.append(run)
            run = 0
    return float(np.mean(lengths)) * 5 / 60 if lengths else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
