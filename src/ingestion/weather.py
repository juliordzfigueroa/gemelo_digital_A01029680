"""Outdoor weather for the campus, aligned with the sensor data.

Downloads hourly weather for Santa Fe from Open-Meteo, which is free and needs
no key, and puts it on the same 5-minute grid as the cleaned sensor data. It
feeds three uses: the physical generator, which needs outdoor humidity and
solar radiation as boundary conditions; candidate exogenous inputs for the
GRU and SARIMAX; and checks of whether a reading is plausible given the
weather.

Two decisions worth knowing:

  - Humidity is also reported as absolute humidity (g/m3). Relative humidity
    depends on temperature, so the same air reads ~95% outside and ~46% once
    it is warmed indoors. The moisture balance is written in absolute terms,
    and on the first week of data the indoor and outdoor absolute humidity
    matched within 0.3 g/m3.
  - The archive is published with a delay of a few days. Hours it does not
    have yet are taken from the forecast service, which also covers the
    recent past, and every row records which source it came from.

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-10-12 - Initial implementation.

Reference:
    Open-Meteo API: https://open-meteo.com/en/docs
    WMO (2018). Guide to Instruments and Methods of Observation (WMO-No. 8),
    Annex 4.B, for the Magnus formula constants over water.
"""

from __future__ import annotations

import argparse
from datetime import date

import httpx
import numpy as np
import pandas as pd

from common import config as cfg

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Open-Meteo name -> column name in this project.
VARIABLES = {
    "temperature_2m": "temp_ext",
    "relative_humidity_2m": "humidity_ext",
    "shortwave_radiation": "radiation_ext",
    "cloud_cover": "cloud_cover_ext",
}

TIMEOUT_S = 30.0


def absolute_humidity(temperature: np.ndarray, relative_humidity: np.ndarray) -> np.ndarray:
    """Convert temperature and relative humidity into absolute humidity.

    Args:
        temperatura: Air temperature in degrees Celsius.
        relative_humidity: Relative humidity in percent.

    Returns:
        Water vapour density in g/m3.

    Reference:
        WMO (2018), WMO-No. 8, Annex 4.B (Magnus formula over water).
    """
    t = np.asarray(temperature, dtype=float)
    hr = np.asarray(relative_humidity, dtype=float)
    saturation_pressure = 6.112 * np.exp(17.62 * t / (243.12 + t))  # hPa
    return 216.7 * (hr / 100 * saturation_pressure) / (273.15 + t)


def _query(url: str, since: date, until: date) -> pd.DataFrame:
    """Request hourly weather from one Open-Meteo service.

    Args:
        url: Archive or forecast endpoint.
        desde: First day, inclusive.
        hasta: Last day, inclusive.

    Returns:
        One row per hour, with the project column names. Hours the service
        does not have come back as NaN.

    Raises:
        RuntimeError: If the service is unreachable or answers with an error.
    """
    parameters = {
        "latitude": cfg.LATITUDE,
        "longitude": cfg.LONGITUDE,
        "start_date": since.isoformat(),
        "end_date": until.isoformat(),
        "hourly": ",".join(VARIABLES),
        "timezone": cfg.TIMEZONE,
    }
    try:
        response = httpx.get(url, params=parameters, timeout=TIMEOUT_S)
        response.raise_for_status()
    except httpx.HTTPError as e:
        raise RuntimeError(f"Open-Meteo did not answer ({url}): {e}") from e

    hours = response.json()["hourly"]
    df = pd.DataFrame({cfg.TIME_COLUMN: pd.to_datetime(hours["time"])})
    for name, column in VARIABLES.items():
        df[column] = pd.to_numeric(pd.Series(hours[name]), errors="coerce")
    return df


def download(since: date, until: date) -> pd.DataFrame:
    """Download hourly weather, filling the archive's recent gap from the forecast.

    Args:
        desde: First day, inclusive.
        hasta: Last day, inclusive.

    Returns:
        One row per hour with the weather columns, humedad_abs_ext and a
        'fuente' column saying whether each hour came from the archive or the
        forecast service.
    """
    archive = _query(ARCHIVE_URL, since, until)
    archive["source"] = "archive"

    missing = archive[list(VARIABLES.values())].isna().any(axis=1)
    if missing.any():
        forecast = _query(FORECAST_URL, since, until).set_index(cfg.TIME_COLUMN)
        index = archive.loc[missing, cfg.TIME_COLUMN]
        replacement = forecast.reindex(index)
        for column in VARIABLES.values():
            archive.loc[missing, column] = replacement[column].to_numpy()
        archive.loc[missing, "source"] = "forecast"

    archive["abs_humidity_ext"] = absolute_humidity(
        archive["temp_ext"], archive["humidity_ext"]
    )
    return archive


def to_intervals(hourly: pd.DataFrame) -> pd.DataFrame:
    """Put hourly weather on the 5-minute grid used by the sensor data.

    Weather changes smoothly within an hour, so linear interpolation between
    hourly values is a fair estimate; it never extrapolates past the last
    hour available.

    Args:
        horario: Output of download().

    Returns:
        One row per 5-minute interval.
    """
    numeric = [*VARIABLES.values(), "abs_humidity_ext"]
    series = hourly.set_index(cfg.TIME_COLUMN)
    grid = pd.date_range(
        series.index.min(), series.index.max(), freq=f"{cfg.INTERVAL_SECONDS}s"
    )
    output = (
        series[numeric]
        .reindex(series.index.union(grid))
        .interpolate(method="time", limit_area="inside")
        .reindex(grid)
    )
    output["source"] = series["source"].reindex(grid, method="ffill")
    return output.rename_axis(cfg.TIME_COLUMN).reset_index()


def attach(data: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Attach the outdoor weather to every sensor row.

    Args:
        datos: Contract frame.
        clima: Output of to_intervals(), or the saved weather.parquet.

    Returns:
        The sensor rows with the weather columns added; rows outside the
        weather range keep NaN.
    """
    columns = [cfg.TIME_COLUMN, *VARIABLES.values(), "abs_humidity_ext"]
    return data.merge(weather[columns], on=cfg.TIME_COLUMN, how="left")


def summary(data: pd.DataFrame) -> pd.DataFrame:
    """Compare indoor and outdoor conditions per space.

    Args:
        datos: Output of attach().

    Returns:
        One row per space with mean indoor and outdoor temperature and
        absolute humidity, and the hourly correlation between them.
    """
    d = data.dropna(subset=["temp_ext", "temperature", "humidity"]).copy()
    d["abs_humidity"] = absolute_humidity(d["temperature"], d["humidity"])

    rows = []
    for space, g in d.groupby(cfg.SPACE_COLUMN):
        rows.append({
            cfg.SPACE_COLUMN: space,
            "temp_in": g["temperature"].mean(),
            "temp_ext": g["temp_ext"].mean(),
            "r_temp": g["temperature"].corr(g["temp_ext"]),
            "abs_hum_in": g["abs_humidity"].mean(),
            "abs_hum_out": g["abs_humidity_ext"].mean(),
            "r_abs_hum": g["abs_humidity"].corr(g["abs_humidity_ext"]),
        })
    return pd.DataFrame(rows)


def main() -> int:
    """Download the weather for the period covered by the sensor data.

    Returns:
        Process exit code.
    """
    from common import data as dat

    parser = argparse.ArgumentParser(description="Download outdoor weather.")
    parser.add_argument("--data", default=None, help="contract frame whose dates to cover")
    parser.add_argument("--since", type=date.fromisoformat, default=None)
    parser.add_argument("--until", type=date.fromisoformat, default=None)
    args = parser.parse_args()

    data = dat.load_data(args.data)
    since = args.since or data[cfg.TIME_COLUMN].min().date()
    until = args.until or data[cfg.TIME_COLUMN].max().date()

    hourly = download(since, until)
    weather = to_intervals(hourly)

    cfg.WEATHER_DATA.parent.mkdir(parents=True, exist_ok=True)
    weather.to_parquet(cfg.WEATHER_DATA, index=False)

    sources = hourly["source"].value_counts().to_dict()
    print(
        f"Clima {since} a {until}: {len(hourly)} horas "
        f"({', '.join(f'{k} {v}' for k, v in sources.items())}), "
        f"{int(hourly[list(VARIABLES.values())].isna().any(axis=1).sum())} con huecos"
    )
    print(
        f"  Exterior: {hourly['temp_ext'].min():.1f} a {hourly['temp_ext'].max():.1f} °C, "
        f"humedad {hourly['humidity_ext'].min():.0f} a {hourly['humidity_ext'].max():.0f}%"
    )

    table = summary(attach(data, weather))
    print("\nInterior contra exterior (r = correlacion por intervalo):")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\nGuardado en {cfg.WEATHER_DATA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
