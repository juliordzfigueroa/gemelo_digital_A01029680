"""Validate the generator by leaving one measured room out.

The design of the generator calls for it: calibrate the type with the other
measured rooms, generate the room left out over the days its sensor
measured, and compare. That error is what backs the "estimated" values of the
rooms without a sensor, which get their type's parameters the same way.

Each room is generated twice, so the cost of not having a sensor is visible:

  propios       with the parameters calibrated on its own readings.
  dejando fuera with the average of the other rooms of its type.

The generator is stochastic, so each case is the average of several
realizations, and it is compared on what a generator should reproduce: the
mean, and the shape of the day (the hourly profile), not the reading at
10:35 of a given Tuesday.

Usage, from the src/ directory:
    python -m generator.validate

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-11-26 - Initial implementation.

Reference:
    Section 4.4 of the research document.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import config as cfg
from common import data as dat
from common import spaces
from generator.calibrate import VARIABLES, average, calibrate_room
from generator.generate import run_room
from generator.inputs import load_weather, room_grid
from generator.occupancy import Occupancy

REALIZATIONS = 10


def compare(grid: pd.DataFrame, generated: list[pd.DataFrame]) -> dict:
    """Compare several realizations with the measured readings of one room.

    Args:
        grid: Output of generator.inputs.room_grid().
        generated: Realizations from run_room() over the grid's stamps.

    Returns:
        Per variable: the measured and generated means, and the correlation
        and mean absolute difference of the hourly profiles.
    """
    output = {}
    hours = grid.index.hour
    for variable in VARIABLES:
        measured = grid[variable].to_numpy()
        mean_run = np.mean([g[variable].to_numpy() for g in generated], axis=0)
        valid = ~np.isnan(measured)
        rows = pd.DataFrame({"m": measured, "g": mean_run, "hour": hours})[valid]
        profile = rows.groupby("hour").mean()
        output[variable] = {
            "measured_mean": float(measured[valid].mean()),
            "generated_mean": float(mean_run[valid].mean()),
            "profile_r": float(np.corrcoef(profile["m"], profile["g"])[0, 1]),
            "profile_mae": float((profile["m"] - profile["g"]).abs().mean()),
        }
    return output


def validate() -> dict:
    """Run both cases for every classroom that supports all five variables.

    Returns:
        Per room, the comparison with its own parameters and left out.
    """
    data = dat.load_data()
    occupancy = Occupancy(data)
    weather = load_weather()
    kinds = {int(s["space_id"]): s["kind"] for s in spaces.read()["spaces"]}

    grids, own = {}, {}
    for space in sorted(int(s) for s in data[cfg.SPACE_COLUMN].unique()):
        grid = room_grid(space, data, occupancy, weather)
        params = calibrate_room(grid)
        if len(params) == len(VARIABLES):
            grids[space], own[space] = grid, params

    results = {}
    for space, grid in grids.items():
        others = [own[s] for s in own if s != space and kinds[s] == kinds[space]]
        if not others:
            continue
        left_out = {v: average([p[v] for p in others]) for v in VARIABLES}
        stamps = grid.index
        students = grid["students"].to_numpy()
        results[space] = {}
        for case, params in (("propios", own[space]), ("dejando fuera", left_out)):
            runs = []
            for r in range(REALIZATIONS):
                rng = np.random.default_rng([cfg.SEED, space, r])
                runs.append(run_room(space, params, stamps, students, weather.reindex(stamps).ffill(), rng))
            results[space][case] = compare(grid, runs)
    return results


def print_report(results: dict) -> None:
    """Print one table per variable."""
    labels = spaces.labels()
    for variable in VARIABLES:
        print(f"\n== {variable} ==")
        print(f"  {'salón':>6} {'media medida':>12} {'propios':>8} {'fuera':>8}   "
              f"{'perfil r propios':>16} {'fuera':>6}   {'perfil MAE propios':>18} {'fuera':>7}")
        for space, cases in results.items():
            o, f = cases["propios"][variable], cases["dejando fuera"][variable]
            print(
                f"  {labels.get(space, space):>6} {o['measured_mean']:>12.2f} {o['generated_mean']:>8.2f} "
                f"{f['generated_mean']:>8.2f}   {o['profile_r']:>16.2f} {f['profile_r']:>6.2f}   "
                f"{o['profile_mae']:>18.2f} {f['profile_mae']:>7.2f}"
            )


def main() -> int:
    """Validate, print the tables and save them.

    Returns:
        Process exit code.
    """
    results = validate()
    print_report(results)
    path = cfg.MODELS / "generator_validation.json"
    path.write_text(json.dumps({str(k): v for k, v in results.items()}, indent=1), encoding="utf-8")
    print(f"\nGuardado en {path.relative_to(cfg.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
