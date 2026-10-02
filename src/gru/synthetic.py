"""Synthetic data generator that satisfies the data contract.

Used to develop and debug the pipeline before the Aulas 3 Arduino readings
exist. It is not meant to be a building simulator: it is meant to have the same
*shape* and the same *defects* the real data will have, so that code working
here keeps working there.

Deliberately reproduced defects:
  - daily and weekly occupancy cycles, different per space type
  - CO2 driven by occupancy with ventilation decay
  - the NDIR calibration floor at 400 ppm, censoring the lower tail
  - light levels on wildly different scales per space, and exact zeros at night
  - noisy binary movement, like a real PIR
  - gaps in the series, like WiFi telemetry dropouts

Usage:
    python synthetic.py            # writes data/raw/synthetic.parquet
    python synthetic.py --days 90

Reference:
    docs/hallazgos_robod.md, for where each of these defects was observed.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import config as cfg

# Per-space profile: light scale, typical capacity and opening hours. Light
# scales differ by two orders of magnitude on purpose, which is what was
# measured between a lecture room and a library and what will happen between a
# room with a large window and an interior corridor.
PROFILES = {
    1: {"name": "interior_room", "light_scale": 90.0, "capacity": 30, "abre": 7, "cierra": 21},
    2: {"name": "window_room", "light_scale": 1800.0, "capacity": 35, "abre": 7, "cierra": 21},
    3: {"name": "oficina", "light_scale": 400.0, "capacity": 8, "abre": 8, "cierra": 19},
    4: {"name": "laboratorio", "light_scale": 650.0, "capacity": 20, "abre": 7, "cierra": 22},
    5: {"name": "pasillo", "light_scale": 120.0, "capacity": 5, "abre": 6, "cierra": 23},
}


def _solar_curve(hour: np.ndarray) -> np.ndarray:
    """Compute the fraction of daylight available at each hour.

    Args:
        hora: Hour of day as a float, e.g. 13.5 for 13:30.

    Returns:
        Values in [0, 1]: zero at night, peaking around midday.
    """
    solar = np.sin(np.pi * (hour - 6.5) / 12.0)
    return np.clip(solar, 0.0, None)


def _occupancy(
    index: pd.DatetimeIndex, profile: dict, rng: np.random.Generator
) -> np.ndarray:
    """Estimate how many people are present at each timestamp.

    Args:
        indice: Regular timestamps to generate for.
        perfil: Space profile from PROFILES.
        rng: Seeded random generator.

    Returns:
        Non-negative occupancy counts, one per timestamp.
    """
    hour = index.hour + index.minute / 60.0
    weekday = index.dayofweek < 5

    inside = (hour >= profile["abre"]) & (hour < profile["cierra"])
    # A bell centred on mid-shift rather than a step function: people arrive
    # and leave gradually.
    center = (profile["abre"] + profile["cierra"]) / 2.0
    width = (profile["cierra"] - profile["abre"]) / 3.0
    shape = np.exp(-0.5 * ((hour - center) / width) ** 2)

    base = profile["capacity"] * shape * inside * np.where(weekday, 1.0, 0.25)
    # Multiplicative noise: classes that empty out, groups arriving together.
    noise = rng.gamma(shape=4.0, scale=0.25, size=len(index))
    return np.clip(base * noise, 0.0, profile["capacity"] * 1.3)


def _co2(people: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Generate CO2 with occupancy build-up and ventilation decay.

    Single time-constant model: at each step the excess over outdoor level
    decays by a fixed factor and the people present add an increment.

    Args:
        personas: Occupancy counts per timestep.
        rng: Seeded random generator.

    Returns:
        CO2 in ppm, clipped at the sensor's calibration floor.
    """
    outdoor = 420.0
    # Steady state of this model is rate_per_person * n / (1 - decaimiento). With
    # these values 30 people settle around 800 ppm above outdoor, so roughly
    # 1,220 ppm total: the range measured in a full room with decent
    # ventilation.
    rate_per_person = 1.6  # ppm per person per 5-minute step
    decay = 0.94  # about a 1.3 h time constant at 5-minute steps

    excess = np.zeros(len(people))
    current = 0.0
    for i, n in enumerate(people):
        current = current * decay + rate_per_person * n
        excess[i] = current

    values = outdoor + excess + rng.normal(0, 4.0, len(people))
    # Sensor floor: an NDIR does not report below 400 ppm. The lower tail ends
    # up censored, exactly as in real readings.
    floor = cfg.LOWER_CENSORED["co2"]
    return np.maximum(values, floor)


def _generate_space(
    space_id: int, index: pd.DatetimeIndex, rng: np.random.Generator
) -> pd.DataFrame:
    """Generate the full series of one space.

    Args:
        space_id: Key into PROFILES.
        indice: Regular timestamps to generate for.
        rng: Seeded random generator.

    Returns:
        A frame satisfying the data contract, for this space only.
    """
    profile = PROFILES[space_id]
    hour = index.hour + index.minute / 60.0
    solar = _solar_curve(np.asarray(hour))
    people = _occupancy(index, profile, rng)
    occupied = people > 0.5

    co2 = _co2(people, rng)

    # Temperature: a damped outdoor daily cycle plus heat from the occupants.
    temperature = (
        21.5
        + 2.2 * np.sin(2 * np.pi * (np.asarray(hour) - 9) / 24)
        + 0.045 * people
        + rng.normal(0, 0.18, len(index))
    )

    # Humidity: inverse to temperature, plus a contribution from the occupants.
    humidity = (
        52.0
        - 1.6 * (temperature - 21.5)
        + 0.12 * people
        + rng.normal(0, 1.1, len(index))
    )
    humidity = np.clip(humidity, *cfg.PHYSICAL_LIMITS["humidity"])

    # Light: solar contribution scaled by the space, plus artificial light when
    # occupied. At night and empty it sits at exactly 0, which is what breaks
    # MAPE and has to be handled in the metrics.
    light = profile["light_scale"] * solar * rng.uniform(0.75, 1.0, len(index))
    light += np.where(occupied, profile["light_scale"] * 0.18, 0.0)
    light = np.maximum(light + rng.normal(0, 2.0, len(index)), 0.0)
    light[(solar <= 0.01) & ~occupied] = 0.0

    # Movement: binary and noisy, like a PIR. The probability grows with
    # occupancy relative to capacity and does not saturate immediately, so
    # there is a gradient between a space holding two people and a full one.
    # The 1% of flips simulates both the false negatives of someone sitting
    # still and the spurious false positives every PIR produces.
    prob = np.clip(people / (profile["capacity"] * 0.5), 0.0, 0.92)
    motion = (rng.random(len(index)) < prob).astype(float)
    motion = np.where(rng.random(len(index)) < 0.01, 1.0 - motion, motion)

    return pd.DataFrame(
        {
            cfg.TIME_COLUMN: index,
            cfg.SPACE_COLUMN: space_id,
            "co2": co2,
            "temperature": temperature,
            "humidity": humidity,
            "light": light,
            "motion": motion,
        }
    )


def _open_gaps(
    df: pd.DataFrame, rng: np.random.Generator, n_gaps: int = 6
) -> pd.DataFrame:
    """Delete whole stretches of rows, like WiFi telemetry dropouts.

    Rows are removed rather than set to NaN because that is what actually
    happens: if a node loses the network, the record never arrives. This is
    what forces the pipeline to work in contiguous blocks.

    Args:
        df: Complete frame.
        rng: Seeded random generator.
        n_gaps: How many gaps to open per space.

    Returns:
        The frame with those rows dropped.
    """
    parts = []
    for space, group in df.groupby(cfg.SPACE_COLUMN, sort=True):
        group = group.sort_values(cfg.TIME_COLUMN).reset_index(drop=True)
        outside = np.zeros(len(group), dtype=bool)
        for _ in range(n_gaps):
            # Gaps from 30 minutes to 2 days, mixing brief and long outages.
            length = int(rng.choice([6, 12, 36, 288, 576]))
            if length >= len(group):
                continue
            start = int(rng.integers(0, len(group) - length))
            outside[start : start + length] = True
        parts.append(group[~outside])
    return pd.concat(parts, ignore_index=True)


def generate(
    days: int = 60,
    n_spaces: int = 5,
    seed: int = cfg.SEED,
    with_gaps: bool = True,
) -> pd.DataFrame:
    """Generate the complete synthetic dataset.

    Args:
        dias: Days of coverage. 60 matches the planned Aulas 3 collection.
        n_spaces: How many spaces to generate, from 1 to 5.
        semilla: Random seed, so runs are reproducible.
        with_gaps: Whether to open gaps in the series. Turning this off helps
            isolate problems, but the realistic case has gaps.

    Returns:
        A frame satisfying the data contract, sorted by space and time.

    Raises:
        ValueError: If n_spaces is outside the available profiles.
    """
    if not 1 <= n_spaces <= len(PROFILES):
        raise ValueError(f"n_espacios must be between 1 and {len(PROFILES)}")

    rng = np.random.default_rng(seed)
    steps = days * 24 * 3600 // cfg.INTERVAL_SECONDS
    index = pd.date_range(
        "2026-01-05 00:00",  # a Monday, so the weekly cycle starts clean
        periods=steps,
        freq=f"{cfg.INTERVAL_SECONDS}s",
    )

    df = pd.concat(
        [_generate_space(e, index, rng) for e in range(1, n_spaces + 1)],
        ignore_index=True,
    )
    if with_gaps:
        df = _open_gaps(df, rng)
    return df.sort_values([cfg.SPACE_COLUMN, cfg.TIME_COLUMN]).reset_index(
        drop=True
    )


def main() -> int:
    """Generate a dataset from the command line and summarize it.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Generate synthetic test data.")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--spaces", type=int, default=5)
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    parser.add_argument("--no-gaps", action="store_true")
    args = parser.parse_args()

    df = generate(
        days=args.days,
        n_spaces=args.spaces,
        seed=args.seed,
        with_gaps=not args.no_gaps,
    )

    cfg.RAW_DATA.mkdir(parents=True, exist_ok=True)
    output = cfg.RAW_DATA / "synthetic.parquet"
    df.to_parquet(output, index=False)

    print(f"Escrito {output.relative_to(cfg.ROOT)}")
    print(f"  {len(df):,} filas | {df[cfg.SPACE_COLUMN].nunique()} espacios")
    print(f"  de {df[cfg.TIME_COLUMN].min()} a {df[cfg.TIME_COLUMN].max()}")
    print()
    summary = df.groupby(cfg.SPACE_COLUMN)[cfg.VARIABLES].agg(["mean", "min", "max"])
    print(summary.round(1).to_string())
    print()
    print("Ceros por variable (el MAPE se indefine donde el valor real es 0):")
    for v in cfg.VARIABLES:
        print(f"  {v:12} {100 * (df[v] == 0).mean():5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
