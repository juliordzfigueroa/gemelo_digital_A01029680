"""Generate the five variables for any spaces and dates of the semester.

Reads the calibration (models/generator_params.json), the schedule through
generator.occupancy and the outdoor weather, and runs one RoomGenerator per
space, 5 minutes at a time. The output has the same columns as the cleaned
sensor data, plus "origin": "simulado" for a measured room generated with its
own parameters, "estimado" for a room without a sensor, generated with its
type's.

Generated data simulates and estimates; it never evaluates a model. Scoring
the GRU against data from the generator would measure how well it imitates
the equations, not the building.

Usage, from the src/ directory:
    python -m generator.generate --since 2026-10-05 --until 2026-10-11
    python -m generator.generate --since 2026-11-02 --until 2026-11-08 --spaces 308 201

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-11-26 - Initial implementation.

Reference:
    Section 4.4 of the research document.
"""

from __future__ import annotations

import argparse
from datetime import date

import numpy as np
import pandas as pd

from common import config as cfg
from common import spaces
from generator.calibrate import load_params
from generator.inputs import load_weather
from generator.occupancy import FIRST_DAY, LAST_DAY, Occupancy
from generator.room import RoomGenerator
from ingestion import weather as wth


def weather_for(stamps: pd.DatetimeIndex) -> pd.DataFrame:
    """The outdoor weather for every stamp, from disk or from Open-Meteo.

    Uses the saved weather when it covers the dates; otherwise downloads
    them, which needs internet.

    Args:
        stamps: The intervals to cover.

    Returns:
        The weather, indexed by time, with a row for every stamp.
    """
    try:
        saved = load_weather()
        if stamps.min() >= saved.index.min() and stamps.max() <= saved.index.max():
            return saved.reindex(stamps)
    except FileNotFoundError:
        pass
    hourly = wth.download(stamps.min().date(), stamps.max().date())
    downloaded = wth.to_intervals(hourly).set_index(cfg.TIME_COLUMN)
    # The last hour of the download ends at 23:00; carry it to midnight.
    return downloaded.reindex(stamps).ffill()


def run_room(space_id: int, params: dict, stamps, students, weather: pd.DataFrame, rng) -> pd.DataFrame:
    """Generate one room over the given stamps.

    Args:
        space_id: Space identifier.
        params: That room's parameters, from models/generator_params.json.
        stamps: The intervals, in order.
        students: N(t) for each interval.
        weather: Output of weather_for(), covering the stamps.
        rng: A numpy Generator.

    Returns:
        One row per interval with timestamp, space_id and the five variables.
    """
    room = RoomGenerator(space_id, params, rng)
    rows = []
    for stamp, n in zip(stamps, students):
        values = room.step(stamp, n, weather.loc[stamp])
        rows.append({cfg.TIME_COLUMN: stamp, cfg.SPACE_COLUMN: space_id, **values})
    return pd.DataFrame(rows)


def generate(space_ids: list[int], since: date, until: date, seed: int = cfg.SEED) -> pd.DataFrame:
    """Generate several spaces over a range of days.

    Args:
        space_ids: Spaces to generate; each must have been calibrated.
        since: First day, inclusive.
        until: Last day, inclusive.
        seed: Seed of the random parts, so a run can be repeated.

    Returns:
        The cleaned-data columns plus "origin", one row per space and interval.

    Raises:
        ValueError: If a space cannot be generated or the dates fall outside
            the semester.
    """
    if pd.Timestamp(since) < FIRST_DAY or pd.Timestamp(until) > LAST_DAY:
        raise ValueError(f"the generator covers {FIRST_DAY.date()} to {LAST_DAY.date()}")
    calibration = load_params()
    for space in space_ids:
        if str(space) not in calibration["rooms"]:
            reason = calibration["not_generated"].get(str(space), "no está en el registro")
            raise ValueError(f"space {space} cannot be generated: {reason}")

    stamps = pd.date_range(since, pd.Timestamp(until) + pd.Timedelta(days=1),
                           freq=f"{cfg.INTERVAL_SECONDS}s", inclusive="left")
    weather = weather_for(stamps)
    occupancy = Occupancy()

    frames = []
    for space in space_ids:
        room = calibration["rooms"][str(space)]
        rng = np.random.default_rng([seed, space])
        frame = run_room(space, room["params"], stamps, occupancy.students(space, stamps), weather, rng)
        frame["origin"] = room["origin"]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    """Generate from the command line and save a parquet file.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Generate environmental data.")
    parser.add_argument("--since", type=date.fromisoformat, required=True)
    parser.add_argument("--until", type=date.fromisoformat, required=True)
    parser.add_argument("--spaces", type=int, nargs="*", help="default: every space that can be generated")
    parser.add_argument("--output", default=str(cfg.PROCESSED_DATA / "generated.parquet"))
    args = parser.parse_args()

    space_ids = args.spaces or [int(s) for s in load_params()["rooms"]]
    output = generate(space_ids, args.since, args.until)
    output.to_parquet(args.output, index=False)

    labels = spaces.labels()
    print(f"Generado {args.since} a {args.until}: {len(output):,} filas, {len(space_ids)} espacios")
    summary = output.groupby(cfg.SPACE_COLUMN).agg(
        origin=("origin", "first"), co2=("co2", "mean"), temperature=("temperature", "mean"),
        humidity=("humidity", "mean"), light=("light", "mean"), motion=("motion", "mean"),
    )
    for space, r in summary.iterrows():
        print(
            f"  {labels.get(space, space):>8} {r['origin']:>9}  CO₂ {r['co2']:6.0f}  "
            f"T {r['temperature']:5.1f}  HR {r['humidity']:5.1f}  luz {r['light']:5.1f}  mov {r['motion']:.3f}"
        )
    print(f"Guardado en {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
