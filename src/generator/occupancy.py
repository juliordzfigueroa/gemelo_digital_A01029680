"""Occupancy N(t): how many students a space holds, every 5 minutes.

Section 4.4.2 of the research document: the five variables share one
occupancy N(t), estimated from the number of students enrolled in each block
of the class schedule. Sharing it is what keeps the generated variables
related the way the GRU has to learn: when a class starts, CO2, humidity,
temperature, light and motion all react to the same N(t).

Three rules on top of the schedule export:

  - Calendar. The first period (Aug 10 to Sep 13) is ignored, so the
    semester starts with Semana Tec 1 on Sep 14 and ends with period 3 on
    Dec 6. Blocks carry their own dates, so the Semana Tec courses fall on
    their exact weeks.
  - Days of the week. The export does not say which days a block meets, so
    a block counts on every weekday of its date range. Where two blocks of
    the same room overlap they cannot both happen; the larger enrolment is
    kept, as ingestion.schedule.possible_occupancy does.
  - Correction with the sensors. In a measured room, a block that fell on a
    weekday when the sensors saw the room empty (no motion and the lamps
    off) is dropped for that weekday. CO2 is not used for this: the gas
    sensor drifts upwards even at night.

Usage, from the src/ directory:
    python -m generator.occupancy

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

from common import config as cfg
from common import data as dat
from ingestion import schedule as sch

# The semester the generator covers: Semana Tec 1 to the end of period 3.
FIRST_DAY = pd.Timestamp(cfg.SEMESTER_START)
LAST_DAY = pd.Timestamp(cfg.SEMESTER_END)

# A block "happened" on a day when, over its readings, the mean motion
# fraction or the mean light reached these. Lamps on read 40 to 65 % in the
# measured rooms; an empty room at night, 0 to 10 %.
MOTION_SEEN = 0.02
LIGHT_SEEN = 20.0

# Readings needed inside a block to judge it: 30 minutes.
MIN_READINGS = 6

WEEKDAYS = ["lun", "mar", "mie", "jue", "vie"]


class Occupancy:
    """Students per space and interval, from the schedule and the sensors."""

    def __init__(self, data: pd.DataFrame | None = None) -> None:
        """Load the schedule and learn from the sensors which blocks to drop.

        Args:
            data: Cleaned sensor data. Defaults to data/processed/aulas3.parquet.
        """
        blocks, _ = sch.load_schedule(cfg.RAW_DATA / "schedule.csv")
        # The first period is ignored: keep only blocks still running on
        # FIRST_DAY or later.
        self.blocks = blocks[blocks["end_date"] > FIRST_DAY].reset_index(drop=True)
        self.data = dat.load_data() if data is None else data
        self.skipped = empty_days(self.blocks, self.data)

    def students(self, space_id: int, stamps) -> np.ndarray:
        """How many students one space holds at each timestamp.

        Args:
            space_id: Space identifier.
            stamps: Interval start times.

        Returns:
            An array aligned with `stamps`: the enrolment of the block in
            class at that time, or 0. Always 0 outside the semester and on
            weekends.
        """
        stamps = pd.to_datetime(pd.Series(stamps)).reset_index(drop=True)
        output = np.zeros(len(stamps), dtype=float)
        in_semester = (stamps >= FIRST_DAY) & (stamps < LAST_DAY + pd.Timedelta(days=1))
        own = self.blocks[self.blocks[cfg.SPACE_COLUMN] == space_id]

        for weekday in range(5):
            # The blocks that meet on this weekday: all but the ones the
            # sensors showed did not happen.
            meeting_rows = []
            for index in own.index:
                if (index, weekday) not in self.skipped:
                    meeting_rows.append(index)
            on_day = ((stamps.dt.dayofweek == weekday) & in_semester).to_numpy()
            if on_day.any() and meeting_rows:
                meeting = own.loc[meeting_rows]
                output[on_day] = sch.possible_occupancy(meeting, stamps[on_day], space_id)
        return output


def empty_days(blocks: pd.DataFrame, data: pd.DataFrame) -> set[tuple[int, int]]:
    """Find the blocks the sensors show did not meet on some weekday.

    A block is dropped for a weekday only if it was observed on that weekday
    and the room looked empty every time. One absent class among several
    weeks does not change the timetable.

    Args:
        blocks: Schedule blocks, as Occupancy.blocks.
        data: Cleaned sensor data.

    Returns:
        A set of (block index, weekday) pairs, Monday being 0.
    """
    seen: dict[tuple[int, int], list[bool]] = {}
    for space, rows in data.groupby(cfg.SPACE_COLUMN):
        own = rows.set_index(cfg.TIME_COLUMN).sort_index()
        first, last = own.index.min().normalize(), own.index.max().normalize()
        for index, block in blocks[blocks[cfg.SPACE_COLUMN] == space].iterrows():
            days = pd.date_range(max(block["start_date"], first), min(block["end_date"], last + pd.Timedelta(days=1)), inclusive="left")
            for day in days:
                if day.dayofweek >= 5:
                    continue
                occupied = _block_occupied(own, day, block)
                if occupied is not None:
                    seen.setdefault((index, day.dayofweek), []).append(occupied)

    skipped = set()
    for key, observations in seen.items():
        if not any(observations):
            skipped.add(key)
    return skipped


def _block_occupied(own: pd.DataFrame, day: pd.Timestamp, block: pd.Series) -> bool | None:
    """Whether the sensors saw people during one block on one day.

    Args:
        own: Readings of the block's room, indexed by time.
        day: The day, at midnight.
        block: One schedule block.

    Returns:
        True or False, or None when there are too few readings to tell.
    """
    start = day + pd.Timedelta(minutes=int(block["start_min"]))
    end = day + pd.Timedelta(minutes=int(block["end_min"]))
    window = own.loc[start : end - pd.Timedelta(seconds=cfg.INTERVAL_SECONDS)]
    if window["motion"].notna().sum() < MIN_READINGS:
        return None
    return bool(window["motion"].mean() >= MOTION_SEEN or window["light"].mean() >= LIGHT_SEEN)


def main() -> int:
    """Print which blocks the sensors ruled out, and the occupied hours per room.

    Returns:
        Process exit code.
    """
    occupancy = Occupancy()
    blocks = occupancy.blocks
    print(f"Bloques del semestre (sin el periodo 1): {len(blocks)}")
    print(f"Bloque y día descartados por los sensores: {len(occupancy.skipped)}\n")
    for index, weekday in sorted(occupancy.skipped, key=lambda k: (blocks.at[k[0], cfg.SPACE_COLUMN], k[1])):
        b = blocks.loc[index]
        print(
            f"  {b['room']}  {WEEKDAYS[weekday]}  {b['course']:9} "
            f"{b['start_time']}-{b['end_time']}  ({int(b['students'])} alumnos)"
        )

    # One measured week, with and without the correction.
    week = pd.date_range("2026-10-05", "2026-10-10", freq=f"{cfg.INTERVAL_SECONDS}s", inclusive="left")
    hours = cfg.INTERVAL_SECONDS / 3600
    print("\nHoras con clase en la semana del 5 al 9 de octubre:")
    print(f"  {'salón':>6} {'solo horario':>13} {'corregido':>10}")
    for space in sorted(occupancy.data[cfg.SPACE_COLUMN].unique()):
        corrected = occupancy.students(space, week)
        possible = sch.possible_occupancy(blocks, pd.Series(week), space)
        print(f"  {space:>6} {(possible > 0).sum() * hours:13.1f} {(corrected > 0).sum() * hours:10.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
