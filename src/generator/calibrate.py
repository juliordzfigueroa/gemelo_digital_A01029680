"""Calibrate the generator for every space of the building.

Three cases, as the design of the generator sets them:

  - A measured room is calibrated with its own readings, variable by
    variable. A variable its sensor could not support (the 3304's dead gas
    sensor) takes the value of its type, and the file says so.
  - A room without a sensor, of a type that has measured rooms, takes the
    parameters of that type: the average of the measured rooms of the same
    kind in config/spaces.json. Its values are "estimated", and how far off
    they can be is what generator/validate.py measures.
  - A type with no measured room at all (laboratories, meeting rooms) is not
    generated: there is nothing to base it on.

Writes models/generator_params.json, which generate.py and the simulation
read.

Usage, from the src/ directory:
    python -m generator.calibrate

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
from generator import co2, humidity, light, motion, temperature
from generator.inputs import load_weather, room_grid
from generator.occupancy import Occupancy

PARAMS_FILE = cfg.MODELS / "generator_params.json"

VARIABLES = ("co2", "humidity", "temperature", "light", "motion")

# Readings a variable needs before its own calibration is trusted: one day.
MIN_READINGS = 24 * 12


def calibrate_room(grid: pd.DataFrame) -> dict:
    """Calibrate every variable a room's readings can support.

    Args:
        grid: Output of generator.inputs.room_grid().

    Returns:
        A dict from variable to its parameters; a variable left out had too
        few readings, or, for CO2, no clean decay to measure λ from.
    """
    params = {}
    if grid["co2"].notna().sum() >= MIN_READINGS:
        try:
            params["co2"] = co2.calibrate(grid)
        except ValueError:
            pass
    if "co2" in params and grid["humidity"].notna().sum() >= MIN_READINGS:
        params["humidity"] = humidity.calibrate(grid, params["co2"]["lambda"])
    if grid["temperature"].notna().sum() >= MIN_READINGS:
        params["temperature"] = temperature.calibrate(grid)
    if grid["light"].notna().sum() >= MIN_READINGS:
        params["light"] = light.calibrate(grid)
    if grid["motion"].notna().sum() >= MIN_READINGS:
        params["motion"] = motion.calibrate(grid)
    return params


def average(values: list):
    """Average a list of parameter sets of the same shape, leaf by leaf.

    Numbers are averaged, lists element by element, dicts key by key; counts
    are summed, since the type is backed by all of them together.

    Args:
        values: Parameter dicts, lists or numbers, all with the same shape.

    Returns:
        One value of that shape.
    """
    first = values[0]
    if isinstance(first, dict):
        output = {}
        for key in first:
            items = [v[key] for v in values]
            output[key] = int(sum(items)) if key.startswith("n_") else average(items)
        return output
    if isinstance(first, list):
        return [average([v[i] for v in values]) for i in range(len(first))]
    return float(np.mean(values))


def type_params(measured: dict, kinds: dict) -> dict:
    """The parameters of each type: the average of its measured rooms.

    Args:
        measured: Output of calibrate_room() for every measured room.
        kinds: Space id to kind, from the registry.

    Returns:
        A dict from kind to {"from_rooms", "params"}; only kinds where every
        variable was calibrated in at least one room.
    """
    output = {}
    for kind in sorted(set(kinds[s] for s in measured)):
        # Only rooms that contributed something: the 3304 has a sensor, but
        # no variable it could support.
        rooms = [s for s in measured if kinds[s] == kind and measured[s]]
        if not rooms:
            continue
        params = {}
        for variable in VARIABLES:
            backed = [measured[s][variable] for s in rooms if variable in measured[s]]
            if backed:
                params[variable] = average(backed)
        if len(params) == len(VARIABLES):
            output[kind] = {"from_rooms": rooms, "params": params}
    return output


def calibrate_all(data: pd.DataFrame | None = None) -> dict:
    """Calibrate the measured rooms, their types, and every space from them.

    Args:
        data: Cleaned sensor data; defaults to data/processed/aulas3.parquet.

    Returns:
        The content of models/generator_params.json.
    """
    data = dat.load_data() if data is None else data
    occupancy = Occupancy(data)
    weather = load_weather()
    kinds = {int(s["space_id"]): s["kind"] for s in spaces.read()["spaces"]}

    measured = {}
    for space in sorted(int(s) for s in data[cfg.SPACE_COLUMN].unique()):
        measured[space] = calibrate_room(room_grid(space, data, occupancy, weather))
    types = type_params(measured, kinds)

    rooms, not_generated = {}, {}
    for space, kind in sorted(kinds.items()):
        if kind not in types:
            not_generated[str(space)] = f"ningún espacio de tipo '{kind}' tiene sensor con datos suficientes"
            continue
        own = measured.get(space, {})
        params = {v: own.get(v, types[kind]["params"][v]) for v in VARIABLES}
        rooms[str(space)] = {
            "kind": kind,
            "origin": "simulado" if own else "estimado",
            "own_variables": [v for v in VARIABLES if v in own],
            "params": params,
        }
    return {
        "types": {k: {"from_rooms": v["from_rooms"]} for k, v in types.items()},
        "rooms": rooms,
        "not_generated": not_generated,
    }


def load_params() -> dict:
    """Read models/generator_params.json.

    Raises:
        FileNotFoundError: If the generator was never calibrated.
    """
    if not PARAMS_FILE.exists():
        raise FileNotFoundError(
            f"{PARAMS_FILE} does not exist; calibrate first with `python -m generator.calibrate`"
        )
    return json.loads(PARAMS_FILE.read_text(encoding="utf-8"))


def main() -> int:
    """Calibrate everything, print what each space got, and save it.

    Returns:
        Process exit code.
    """
    result = calibrate_all()
    labels = spaces.labels()

    for kind, info in result["types"].items():
        rooms = ", ".join(labels.get(s, str(s)) for s in info["from_rooms"])
        print(f"Tipo {kind}: promedio de {rooms}")
    print(f"\n{'espacio':>18} {'tipo':>12} {'origen':>9}  variables propias")
    for space, info in result["rooms"].items():
        own = ", ".join(info["own_variables"]) or "ninguna (todo del tipo)"
        print(f"{labels.get(int(space), space):>18} {info['kind']:>12} {info['origin']:>9}  {own}")
    print(f"\nNo se generan: {len(result['not_generated'])} espacios "
          f"({', '.join(sorted(set(r.split(chr(39))[1] for r in result['not_generated'].values())))})")

    cfg.MODELS.mkdir(parents=True, exist_ok=True)
    PARAMS_FILE.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nGuardado en {PARAMS_FILE.relative_to(cfg.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
