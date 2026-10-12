"""Class schedule as a scheduled-occupancy signal, and its check against sensors.

Loads the registrar's schedule export and turns it into, for each space and
5-minute interval, how many students a class there could bring. Then compares
that against what the sensors saw, to answer whether the schedule explains the
measurements before anything is built on that assumption.

The export has no day-of-week column. It carries the date range, the time
block and the room, and repeats identical rows, most likely one per weekly
meeting with the day dropped when it was exported. So the occupancy produced
here is *possible*, not *scheduled*: a block counts on every weekday of its
date range, because there is no way to tell from the file which days it meets.
When two classes overlap in the same room and time they cannot both happen,
so they are on different days; the larger enrolment is kept.

Comparing the signal day by day against movement and eCO2 is what recovers the
missing information: a block that lights up the sensors on Monday and
Thursday but not on Tuesday tells which days it meets.

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-10-12 - Initial implementation.

Reference:
    Section 4.2.6 of the research document.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import config as cfg

CONFIG_DIR = cfg.ROOT / "config"

COLUMNS = {
    "Clave Materia": "course",
    "v_Total Alum Inscritos (RE)": "students",
    "Periodo": "period",
    "Fecha Inicio": "start_date",
    "Fecha Fin": "end_date",
    "Horario Inicio": "start_time",
    "Horario Fin": "end_time",
    "Salon": "room",
}

# Hours compared against the schedule. Outside them the building is closed and
# the comparison would only measure the night, inflating every contrast.
OPENING_HOUR = 7
CLOSING_HOUR = 22

DAYS = ["lun", "mar", "mie", "jue", "vie", "sab", "dom"]


def _minutes(hhmm: pd.Series) -> pd.Series:
    """Convert times written as HHMM integers into minutes after midnight.

    Args:
        hhmm: Values such as 710 or 1350.

    Returns:
        Minutes after midnight.
    """
    values = hhmm.astype(int)
    return (values // 100) * 60 + values % 100


def load_spaces(path: Path = CONFIG_DIR / "spaces.json") -> dict[str, int]:
    """Map room numbers as written in the schedule to space_id.

    Args:
        ruta: The shared space registry.

    Returns:
        A dict from room string ("3308") to space_id (308).
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    mapping = {}
    for space in data["spaces"]:
        for name in [space.get("room"), *space.get("alias", [])]:
            if name:
                mapping[str(name)] = int(space["space_id"])
    return mapping


def load_schedule(path: Path) -> tuple[pd.DataFrame, dict]:
    """Read the schedule export and attach space_id.

    Args:
        ruta: CSV exported from the registrar.

    Returns:
        A tuple of (one row per distinct class block, report). Identical rows
        are collapsed because, without the day column, they carry no
        information beyond the first.

    Raises:
        ValueError: If a required column is missing.
    """
    raw = pd.read_csv(path, dtype=str)
    missing = set(COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")

    df = raw[list(COLUMNS)].rename(columns=COLUMNS)
    rows = len(df)
    df = df.drop_duplicates()

    df["students"] = df["students"].astype(int)
    df["start_date"] = pd.to_datetime(df["start_date"], format="%d/%m/%Y")
    # The end date is inclusive in the export; the day after is the exclusive
    # bound that interval arithmetic expects.
    df["end_date"] = pd.to_datetime(df["end_date"], format="%d/%m/%Y") + pd.Timedelta(days=1)
    df["start_min"] = _minutes(df["start_time"])
    df["end_min"] = _minutes(df["end_time"])

    mapping = load_spaces()
    df[cfg.SPACE_COLUMN] = df["room"].map(mapping)
    no_space = sorted(df.loc[df[cfg.SPACE_COLUMN].isna(), "room"].unique())
    df = df.dropna(subset=[cfg.SPACE_COLUMN])
    df[cfg.SPACE_COLUMN] = df[cfg.SPACE_COLUMN].astype(int)

    return df.reset_index(drop=True), {
        "rows_read": rows,
        "repeated_rows": rows - len(raw[list(COLUMNS)].drop_duplicates()),
        "distinct_blocks": len(df),
        "rooms_without_space": no_space,
    }


def possible_occupancy(
    schedule: pd.DataFrame, stamps: pd.Series, space_id: int
) -> np.ndarray:
    """Students a class could bring to one space at each timestamp.

    Weekdays only: the export does not say which days a block meets, and the
    weekend data show essentially no motion, so counting Saturday would add
    blocks that are almost certainly not there.

    Args:
        horarios: Output of load_schedule().
        marcas: Interval start times.
        space_id: Space to compute.

    Returns:
        An array aligned with `marcas`: the largest enrolment among the blocks
        active at that time, or 0.
    """
    own_blocks = schedule[schedule[cfg.SPACE_COLUMN] == space_id]
    stamps = pd.to_datetime(stamps).reset_index(drop=True)
    minute = (stamps.dt.hour * 60 + stamps.dt.minute).to_numpy()
    weekday = (stamps.dt.dayofweek < 5).to_numpy()
    output = np.zeros(len(stamps), dtype=float)

    for _, b in own_blocks.iterrows():
        active = (
            weekday
            & (stamps >= b["start_date"]).to_numpy()
            & (stamps < b["end_date"]).to_numpy()
            & (minute >= b["start_min"])
            & (minute < b["end_min"])
        )
        output[active] = np.maximum(output[active], b["students"])
    return output


def compare(data: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Contrast sensor readings inside and outside scheduled blocks, per day.

    Args:
        datos: Cleaned contract frame from ingestion.cleaning.
        horarios: Output of load_schedule().

    Returns:
        One row per space and weekday with the mean movement fraction and eCO2
        inside and outside scheduled blocks during opening hours, plus the
        correlation between possible students and each variable.
    """
    rows = []
    for space, group in data.groupby(cfg.SPACE_COLUMN):
        group = group.sort_values(cfg.TIME_COLUMN).reset_index(drop=True)
        group["students"] = possible_occupancy(schedule, group[cfg.TIME_COLUMN], space)
        hour = group[cfg.TIME_COLUMN].dt.hour
        group = group[(hour >= OPENING_HOUR) & (hour < CLOSING_HOUR)]
        group = group.assign(day=group[cfg.TIME_COLUMN].dt.dayofweek)

        for day, d in group.groupby("day"):
            inside = d["students"] > 0
            row = {
                cfg.SPACE_COLUMN: space,
                "day": DAYS[day],
                "intervals": len(d),
                "with_class": int(inside.sum()),
            }
            for variable, short in (("motion", "mov"), ("co2", "co2")):
                row[f"{short}_inside"] = d.loc[inside, variable].mean()
                row[f"{short}_outside"] = d.loc[~inside, variable].mean()
                # A constant series has no correlation to report; nan says so
                # instead of a misleading 0.
                valid_rows = d[[variable, "students"]].dropna()
                row[f"r_{short}"] = (
                    valid_rows[variable].corr(valid_rows["students"])
                    if valid_rows["students"].nunique() > 1 and valid_rows[variable].nunique() > 1
                    else np.nan
                )
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    """Load the schedule and the cleaned data, and print the comparison.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Check the schedule against sensors.")
    parser.add_argument("schedule", type=Path, help="schedule CSV export")
    parser.add_argument(
        "--data", type=Path, default=cfg.PROCESSED_DATA / "aulas3.parquet"
    )
    args = parser.parse_args()

    schedule, report = load_schedule(args.schedule)
    print(
        f"Horario: {report['rows_read']} filas, {report['repeated_rows']} "
        f"repetidas, {report['distinct_blocks']} bloques distintos con espacio"
    )
    if report["rooms_without_space"]:
        print(f"Salones que no estan en spaces.json: {report['rooms_without_space']}")

    data = pd.read_parquet(args.data)
    table = compare(data, schedule)

    pd.set_option("display.width", 140)
    print(
        f"\nDe {OPENING_HOUR}:00 a {CLOSING_HOUR}:00. 'dentro' = intervalos con una "
        "clase posible segun el horario; r = correlacion con los alumnos posibles.\n"
    )
    print(
        table.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}" if abs(x) < 10 else f"{x:.0f}",
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
