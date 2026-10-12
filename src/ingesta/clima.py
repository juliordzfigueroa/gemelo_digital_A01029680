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

from comun import config as cfg

URL_HISTORICO = "https://archive-api.open-meteo.com/v1/archive"
URL_PRONOSTICO = "https://api.open-meteo.com/v1/forecast"

# Open-Meteo name -> column name in this project.
VARIABLES = {
    "temperature_2m": "temp_ext",
    "relative_humidity_2m": "humedad_ext",
    "shortwave_radiation": "radiacion_ext",
    "cloud_cover": "nubosidad_ext",
}

TIMEOUT_SEG = 30.0


def humedad_absoluta(temperatura: np.ndarray, humedad_relativa: np.ndarray) -> np.ndarray:
    """Convert temperature and relative humidity into absolute humidity.

    Args:
        temperatura: Air temperature in degrees Celsius.
        humedad_relativa: Relative humidity in percent.

    Returns:
        Water vapour density in g/m3.

    Reference:
        WMO (2018), WMO-No. 8, Annex 4.B (Magnus formula over water).
    """
    t = np.asarray(temperatura, dtype=float)
    hr = np.asarray(humedad_relativa, dtype=float)
    presion_saturacion = 6.112 * np.exp(17.62 * t / (243.12 + t))  # hPa
    return 216.7 * (hr / 100 * presion_saturacion) / (273.15 + t)


def _consultar(url: str, desde: date, hasta: date) -> pd.DataFrame:
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
    parametros = {
        "latitude": cfg.LATITUD,
        "longitude": cfg.LONGITUD,
        "start_date": desde.isoformat(),
        "end_date": hasta.isoformat(),
        "hourly": ",".join(VARIABLES),
        "timezone": cfg.ZONA_HORARIA,
    }
    try:
        respuesta = httpx.get(url, params=parametros, timeout=TIMEOUT_SEG)
        respuesta.raise_for_status()
    except httpx.HTTPError as e:
        raise RuntimeError(f"Open-Meteo did not answer ({url}): {e}") from e

    horas = respuesta.json()["hourly"]
    df = pd.DataFrame({cfg.COLUMNA_TIEMPO: pd.to_datetime(horas["time"])})
    for nombre, columna in VARIABLES.items():
        df[columna] = pd.to_numeric(pd.Series(horas[nombre]), errors="coerce")
    return df


def descargar(desde: date, hasta: date) -> pd.DataFrame:
    """Download hourly weather, filling the archive's recent gap from the forecast.

    Args:
        desde: First day, inclusive.
        hasta: Last day, inclusive.

    Returns:
        One row per hour with the weather columns, humedad_abs_ext and a
        'fuente' column saying whether each hour came from the archive or the
        forecast service.
    """
    historico = _consultar(URL_HISTORICO, desde, hasta)
    historico["fuente"] = "historico"

    faltan = historico[list(VARIABLES.values())].isna().any(axis=1)
    if faltan.any():
        pronostico = _consultar(URL_PRONOSTICO, desde, hasta).set_index(cfg.COLUMNA_TIEMPO)
        indice = historico.loc[faltan, cfg.COLUMNA_TIEMPO]
        reemplazo = pronostico.reindex(indice)
        for columna in VARIABLES.values():
            historico.loc[faltan, columna] = reemplazo[columna].to_numpy()
        historico.loc[faltan, "fuente"] = "pronostico"

    historico["humedad_abs_ext"] = humedad_absoluta(
        historico["temp_ext"], historico["humedad_ext"]
    )
    return historico


def a_intervalos(horario: pd.DataFrame) -> pd.DataFrame:
    """Put hourly weather on the 5-minute grid used by the sensor data.

    Weather changes smoothly within an hour, so linear interpolation between
    hourly values is a fair estimate; it never extrapolates past the last
    hour available.

    Args:
        horario: Output of descargar().

    Returns:
        One row per 5-minute interval.
    """
    numericas = [*VARIABLES.values(), "humedad_abs_ext"]
    serie = horario.set_index(cfg.COLUMNA_TIEMPO)
    rejilla = pd.date_range(
        serie.index.min(), serie.index.max(), freq=f"{cfg.INTERVALO_SEGUNDOS}s"
    )
    salida = (
        serie[numericas]
        .reindex(serie.index.union(rejilla))
        .interpolate(method="time", limit_area="inside")
        .reindex(rejilla)
    )
    salida["fuente"] = serie["fuente"].reindex(rejilla, method="ffill")
    return salida.rename_axis(cfg.COLUMNA_TIEMPO).reset_index()


def unir(datos: pd.DataFrame, clima: pd.DataFrame) -> pd.DataFrame:
    """Attach the outdoor weather to every sensor row.

    Args:
        datos: Contract frame.
        clima: Output of a_intervalos(), or the saved clima.parquet.

    Returns:
        The sensor rows with the weather columns added; rows outside the
        weather range keep NaN.
    """
    columnas = [cfg.COLUMNA_TIEMPO, *VARIABLES.values(), "humedad_abs_ext"]
    return datos.merge(clima[columnas], on=cfg.COLUMNA_TIEMPO, how="left")


def resumen(datos: pd.DataFrame) -> pd.DataFrame:
    """Compare indoor and outdoor conditions per space.

    Args:
        datos: Output of unir().

    Returns:
        One row per space with mean indoor and outdoor temperature and
        absolute humidity, and the hourly correlation between them.
    """
    d = datos.dropna(subset=["temp_ext", "temperatura", "humedad"]).copy()
    d["humedad_abs"] = humedad_absoluta(d["temperatura"], d["humedad"])

    filas = []
    for espacio, g in d.groupby(cfg.COLUMNA_ESPACIO):
        filas.append({
            cfg.COLUMNA_ESPACIO: espacio,
            "temp_int": g["temperatura"].mean(),
            "temp_ext": g["temp_ext"].mean(),
            "r_temp": g["temperatura"].corr(g["temp_ext"]),
            "h_abs_int": g["humedad_abs"].mean(),
            "h_abs_ext": g["humedad_abs_ext"].mean(),
            "r_h_abs": g["humedad_abs"].corr(g["humedad_abs_ext"]),
        })
    return pd.DataFrame(filas)


def main() -> int:
    """Download the weather for the period covered by the sensor data.

    Returns:
        Process exit code.
    """
    from comun import datos as dat

    parser = argparse.ArgumentParser(description="Download outdoor weather.")
    parser.add_argument("--datos", default=None, help="contract frame whose dates to cover")
    parser.add_argument("--desde", type=date.fromisoformat, default=None)
    parser.add_argument("--hasta", type=date.fromisoformat, default=None)
    args = parser.parse_args()

    datos = dat.cargar_datos(args.datos)
    desde = args.desde or datos[cfg.COLUMNA_TIEMPO].min().date()
    hasta = args.hasta or datos[cfg.COLUMNA_TIEMPO].max().date()

    horario = descargar(desde, hasta)
    clima = a_intervalos(horario)

    cfg.DATOS_CLIMA.parent.mkdir(parents=True, exist_ok=True)
    clima.to_parquet(cfg.DATOS_CLIMA, index=False)

    fuentes = horario["fuente"].value_counts().to_dict()
    print(
        f"Clima {desde} a {hasta}: {len(horario)} horas "
        f"({', '.join(f'{k} {v}' for k, v in fuentes.items())}), "
        f"{int(horario[list(VARIABLES.values())].isna().any(axis=1).sum())} con huecos"
    )
    print(
        f"  Exterior: {horario['temp_ext'].min():.1f} a {horario['temp_ext'].max():.1f} °C, "
        f"humedad {horario['humedad_ext'].min():.0f} a {horario['humedad_ext'].max():.0f}%"
    )

    tabla = resumen(unir(datos, clima))
    print("\nInterior contra exterior (r = correlacion por intervalo):")
    print(tabla.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\nGuardado en {cfg.DATOS_CLIMA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
