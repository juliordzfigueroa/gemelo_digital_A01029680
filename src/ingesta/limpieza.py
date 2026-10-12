"""Clean raw sensor exports into the project data contract.

Turns the CSV the ESP32 nodes produce into the frame every model in this
repository expects: one row per space and 5-minute interval, with co2,
temperatura, humedad, luz and movimiento. It accepts several files at once,
because exports arrive week by week and overlap, and writes a report of every
reading it discarded and why, so the cleaning can be audited in the thesis
instead of taken on trust.

Decisions that are not obvious from the code:

  - Time comes from `formattedTimestamp`. The numeric `timestamp` column was
    saved in scientific notation by a spreadsheet and holds 57 distinct values
    for 370,341 rows, so it is unusable.
  - `tvoc` and `rssi` are dropped, by the advisor's decision to work with the
    five variables only.
  - `eco2` becomes `co2`, but it is the equivalent CO2 a metal-oxide sensor
    estimates from volatile compounds, not a measured NDIR value. It rises
    with cleaning products and markers, and stays above 1,000 ppm on weekends
    with no motion. Downstream code must not treat it as true CO2.
  - `lightPercent` becomes `luz` in percent (0-100), not lux.
  - Movement becomes the fraction of readings in the interval that saw motion,
    a value in [0, 1]. Keeping a single 0/1 per interval would discard almost
    everything: the PIR fires in 1.6-3.6% of weekday readings even in full
    classrooms.
  - Nothing is interpolated here. Gaps stay as missing rows, which is what the
    data contract asks for; the training pipeline decides what to bridge.

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-10-12 - Initial implementation.

Reference:
    Section 4.2.6 of the research document; data contract in config.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from comun import config as cfg

CARPETA_CONFIG = cfg.RAIZ / "config"

# Raw column -> contract column. Anything not listed here is dropped.
COLUMNAS = {
    "temperature": "temperatura",
    "humidity": "humedad",
    "lightPercent": "luz",
    "motion": "movimiento",
    "eco2": "co2",
}

FORMATO_TIEMPO = "%d %b %Y, %H:%M:%S"

# The gas sensor reports -1 while it warms up or when it fails.
CENTINELA = -1

# Readings arrive about every 5 seconds, so a full 5-minute interval holds
# around 60. An interval backed by fewer than this many valid readings is
# dropped for that variable: an average of two readings is noise presented as
# a measurement.
MIN_LECTURAS_POR_INTERVALO = 6


def leer_crudos(rutas: list[Path]) -> tuple[pd.DataFrame, dict]:
    """Read one or more raw exports and merge them.

    Args:
        rutas: CSV files exported from the sensor platform.

    Returns:
        A tuple of (frame with deviceId, tiempo and the five raw variables,
        partial report). Exact repeats of the same device and second, which
        appear when exports overlap, are kept only once.

    Raises:
        ValueError: If a file lacks a required column or holds a timestamp that
            does not match FORMATO_TIEMPO.
    """
    requeridas = ["deviceId", "formattedTimestamp", *COLUMNAS]
    partes = []
    for ruta in rutas:
        crudo = pd.read_csv(ruta)
        faltan = set(requeridas) - set(crudo.columns)
        if faltan:
            raise ValueError(f"{ruta.name} is missing columns: {sorted(faltan)}")
        partes.append(crudo[requeridas])

    df = pd.concat(partes, ignore_index=True)
    filas_leidas = len(df)

    # errors="raise" on purpose: a timestamp in an unexpected format must stop
    # the run, not become NaT and silently vanish with its reading.
    df["tiempo"] = pd.to_datetime(df.pop("formattedTimestamp"), format=FORMATO_TIEMPO)
    df = df.drop_duplicates(subset=["deviceId", "tiempo"], keep="first")

    return df, {
        "archivos": [r.name for r in rutas],
        "filas_leidas": filas_leidas,
        "duplicadas_entre_archivos": filas_leidas - len(df),
    }


def cargar_sensores(ruta: Path = CARPETA_CONFIG / "sensores.csv") -> pd.DataFrame:
    """Load the device-to-space table.

    Args:
        ruta: CSV with dispositivo, espacio_id, desde and hasta.

    Returns:
        The table with parsed dates. Empty dates mean an open interval.
    """
    tabla = pd.read_csv(ruta, dtype={"dispositivo": str})
    tabla["desde"] = pd.to_datetime(tabla["desde"])
    tabla["hasta"] = pd.to_datetime(tabla["hasta"])
    return tabla


def asignar_espacios(
    df: pd.DataFrame, sensores: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """Attach the space each reading belongs to.

    The model learns spaces, not devices, so every reading needs the space its
    device sat in at that moment. The table carries dates so that a sensor
    moved to another room splits correctly instead of merging two rooms into
    one identity.

    Args:
        df: Output of leer_crudos().
        sensores: Output of cargar_sensores().

    Returns:
        A tuple of (readings with an integer espacio_id, count of discarded
        readings per device and reason).
    """
    df = df.copy()
    df[cfg.COLUMNA_ESPACIO] = np.nan

    for _, fila in sensores.dropna(subset=[cfg.COLUMNA_ESPACIO]).iterrows():
        mascara = df["deviceId"] == fila["dispositivo"]
        if pd.notna(fila["desde"]):
            mascara &= df["tiempo"] >= fila["desde"]
        if pd.notna(fila["hasta"]):
            mascara &= df["tiempo"] < fila["hasta"]
        df.loc[mascara, cfg.COLUMNA_ESPACIO] = int(fila[cfg.COLUMNA_ESPACIO])

    conocidos = set(sensores["dispositivo"])
    sin_espacio = df[cfg.COLUMNA_ESPACIO].isna()
    descartes = {}
    for dispositivo, n in df.loc[sin_espacio, "deviceId"].value_counts().items():
        motivo = (
            "dispositivo sin espacio asignado"
            if dispositivo in conocidos
            else "dispositivo no registrado en sensores.csv"
        )
        descartes[dispositivo] = {"filas": int(n), "motivo": motivo}

    df = df[~sin_espacio].copy()
    df[cfg.COLUMNA_ESPACIO] = df[cfg.COLUMNA_ESPACIO].astype(int)
    return df, descartes


def invalidar(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Turn impossible readings into missing values.

    Three rules, each counted separately so the report says which one fired:

      - centinela: the sensor's own error code, -1.
      - lectura_vacia: temperature and humidity both exactly 0, the signature
        of a failed read on this kind of sensor rather than a real reading.
      - fuera_de_rango: outside the instrument limits in config.LIMITES_FISICOS.

    Only the affected variable is blanked; the rest of the reading survives.

    Args:
        df: Readings with the contract column names.

    Returns:
        A tuple of (cleaned copy, counts per device, variable and rule).
    """
    df = df.copy()
    conteo: dict = {}

    def marcar(mascara: pd.Series, variable: str, regla: str) -> None:
        for dispositivo, n in df.loc[mascara, "deviceId"].value_counts().items():
            (
                conteo.setdefault(dispositivo, {})
                .setdefault(variable, {})
                .setdefault(regla, 0)
            )
            conteo[dispositivo][variable][regla] += int(n)
        df.loc[mascara, variable] = np.nan

    for variable in cfg.VARIABLES_ACTIVAS:
        marcar(df[variable] == CENTINELA, variable, "centinela")

    vacia = (df["temperatura"] == 0) & (df["humedad"] == 0)
    for variable in ("temperatura", "humedad"):
        marcar(vacia, variable, "lectura_vacia")

    for variable in cfg.VARIABLES_ACTIVAS:
        minimo, maximo = cfg.LIMITES_FISICOS[variable]
        fuera = pd.Series(False, index=df.index)
        if minimo is not None:
            fuera |= df[variable] < minimo
        if maximo is not None:
            fuera |= df[variable] > maximo
        if variable == "luz":
            # Real nodes report light as a percentage, so it is bounded above
            # even though the synthetic lux scale in config is not.
            fuera |= df[variable] > 100
        marcar(fuera & df[variable].notna(), variable, "fuera_de_rango")

    return df, conteo


def agregar(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate readings into 5-minute intervals per space.

    Continuous variables are averaged. Movement is averaged too, which turns a
    stream of 0/1 into the fraction of the interval with motion detected.
    Intervals are labelled by their start.

    Args:
        df: Cleaned readings carrying espacio_id and tiempo.

    Returns:
        One row per space and interval, satisfying the data contract. Rows
        where every variable is missing are dropped, so gaps stay as absent
        rows.
    """
    intervalo = f"{cfg.INTERVALO_SEGUNDOS}s"
    df = df.assign(
        **{cfg.COLUMNA_TIEMPO: df["tiempo"].dt.floor(intervalo)}
    )
    grupos = df.groupby([cfg.COLUMNA_ESPACIO, cfg.COLUMNA_TIEMPO])

    medias = grupos[cfg.VARIABLES_ACTIVAS].mean()
    validas = grupos[cfg.VARIABLES_ACTIVAS].count()
    medias = medias.where(validas >= MIN_LECTURAS_POR_INTERVALO)

    salida = medias.dropna(how="all").reset_index()
    salida[cfg.COLUMNA_ESPACIO] = salida[cfg.COLUMNA_ESPACIO].astype(int)
    return salida[[cfg.COLUMNA_TIEMPO, cfg.COLUMNA_ESPACIO, *cfg.VARIABLES_ACTIVAS]]


def limpiar(rutas: list[Path]) -> tuple[pd.DataFrame, dict]:
    """Run the whole cleaning, from raw exports to the data contract.

    Args:
        rutas: Raw CSV exports.

    Returns:
        A tuple of (contract frame, full report).
    """
    crudo, reporte = leer_crudos(rutas)
    crudo = crudo.rename(columns=COLUMNAS)

    reporte["filas_por_dispositivo"] = {
        k: int(v) for k, v in crudo["deviceId"].value_counts().sort_index().items()
    }

    asignado, descartes = asignar_espacios(crudo, cargar_sensores())
    reporte["descartadas_por_dispositivo"] = descartes

    limpio, invalidos = invalidar(asignado)
    reporte["valores_invalidados"] = invalidos

    contrato = agregar(limpio)

    resumen = {}
    for espacio, grupo in contrato.groupby(cfg.COLUMNA_ESPACIO):
        completas = grupo[cfg.VARIABLES_ACTIVAS].notna().all(axis=1)
        resumen[str(espacio)] = {
            "desde": str(grupo[cfg.COLUMNA_TIEMPO].min()),
            "hasta": str(grupo[cfg.COLUMNA_TIEMPO].max()),
            "intervalos": int(len(grupo)),
            "intervalos_completos": int(completas.sum()),
            "horas_completas": round(float(completas.sum()) * cfg.INTERVALO_SEGUNDOS / 3600, 1),
            "faltantes_por_variable": {
                v: int(grupo[v].isna().sum()) for v in cfg.VARIABLES_ACTIVAS
            },
        }
    reporte["por_espacio"] = resumen
    reporte["notas"] = [
        "co2 es eCO2 estimado por un sensor MOX, no CO2 medido (NDIR).",
        "luz esta en porcentaje (0-100), no en lux.",
        "movimiento es la fraccion del intervalo con deteccion, en [0, 1].",
        "tvoc y rssi se descartan por decision de la asesora.",
    ]
    return contrato, reporte


def imprimir(reporte: dict) -> None:
    """Print the parts of the report worth reading in the terminal.

    Args:
        reporte: Report returned by limpiar().
    """
    print(f"Archivos: {', '.join(reporte['archivos'])}")
    print(
        f"Filas leidas: {reporte['filas_leidas']:,}  "
        f"(duplicadas entre archivos: {reporte['duplicadas_entre_archivos']:,})"
    )

    if reporte["descartadas_por_dispositivo"]:
        print("\nDescartadas por no tener espacio:")
        for dispositivo, d in reporte["descartadas_por_dispositivo"].items():
            print(f"  {dispositivo:15} {d['filas']:>8,}  {d['motivo']}")

    if reporte["valores_invalidados"]:
        print("\nValores invalidados (la lectura sigue, sin esa variable):")
        for dispositivo, variables in sorted(reporte["valores_invalidados"].items()):
            for variable, reglas in variables.items():
                detalle = ", ".join(f"{r}={n:,}" for r, n in reglas.items())
                print(f"  {dispositivo:15} {variable:12} {detalle}")

    print(
        f"\n{'espacio':>8} {'desde':>17} {'hasta':>17} {'interv.':>8} "
        f"{'completos':>9} {'horas':>6}"
    )
    for espacio, r in reporte["por_espacio"].items():
        print(
            f"{espacio:>8} {r['desde'][5:16]:>17} {r['hasta'][5:16]:>17} "
            f"{r['intervalos']:>8} {r['intervalos_completos']:>9} "
            f"{r['horas_completas']:>6}"
        )


def main() -> int:
    """Clean the given exports and write the contract frame and the report.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Clean raw sensor exports.")
    parser.add_argument("entradas", nargs="+", type=Path, help="raw CSV exports")
    parser.add_argument(
        "--salida",
        type=Path,
        default=cfg.DATOS_PROCESADOS / "aulas3.parquet",
        help="where to write the contract frame",
    )
    args = parser.parse_args()

    contrato, reporte = limpiar(args.entradas)

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    contrato.to_parquet(args.salida, index=False)
    ruta_reporte = args.salida.with_name(args.salida.stem + "_reporte.json")
    ruta_reporte.write_text(
        json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    imprimir(reporte)
    print(f"\nGuardado: {args.salida}\nReporte:  {ruta_reporte}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
