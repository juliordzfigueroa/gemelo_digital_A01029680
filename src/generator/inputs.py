"""What every equation of the generator reads: one room on a 5-minute grid.

Each variable module calibrates from the same frame: the room's readings on
a continuous grid (gaps as NaN, so a step always means 5 minutes), the
students N(t) of generator.occupancy, and the outdoor weather of
ingestion.weather. Building it once keeps the five calibrations reading the
same rows.

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-11-26 - Initial implementation.

Reference:
    Section 4.4.2 of the research document.
"""

from __future__ import annotations

import pandas as pd

from common import config as cfg
from generator.occupancy import Occupancy
from ingestion import weather as wth

# Hours per step, for the equations written per hour.
STEP_HOURS = cfg.INTERVAL_SECONDS / 3600


def load_weather() -> pd.DataFrame:
    """Read the outdoor weather written by `python -m ingestion.weather`.

    Returns:
        One row per 5-minute interval, indexed by time.

    Raises:
        FileNotFoundError: If the weather was never downloaded.
    """
    if not cfg.WEATHER_DATA.exists():
        raise FileNotFoundError(
            f"{cfg.WEATHER_DATA} does not exist; download it with `python -m ingestion.weather`"
        )
    return pd.read_parquet(cfg.WEATHER_DATA).set_index(cfg.TIME_COLUMN)


def room_grid(
    space_id: int, data: pd.DataFrame, occupancy: Occupancy, weather: pd.DataFrame
) -> pd.DataFrame:
    """Put one room's readings, students and weather on a gap-free grid.

    The grid covers whole days, from the midnight before the first reading
    to the midnight after the last one.

    Args:
        space_id: Space identifier.
        data: Cleaned sensor data.
        occupancy: Built once by the caller, since it reads the schedule.
        weather: Output of load_weather().

    Returns:
        A frame indexed by time with the five variables (NaN where there was
        no reading), "students" and the weather columns.
    """
    rows = data[data[cfg.SPACE_COLUMN] == space_id].set_index(cfg.TIME_COLUMN).sort_index()
    index = pd.date_range(
        rows.index.min().normalize(),
        rows.index.max().normalize() + pd.Timedelta(days=1),
        freq=f"{cfg.INTERVAL_SECONDS}s",
        inclusive="left",
    )
    grid = rows[cfg.ACTIVE_VARIABLES].reindex(index)
    grid["students"] = occupancy.students(space_id, index)
    return grid.join(weather[wth.WEATHER_COLUMNS])
