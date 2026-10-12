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

from comun import config as cfg

CARPETA_CONFIG = cfg.RAIZ / "config"

COLUMNAS = {
    "Clave Materia": "clave",
    "v_Total Alum Inscritos (RE)": "alumnos",
    "Periodo": "periodo",
    "Fecha Inicio": "fecha_inicio",
    "Fecha Fin": "fecha_fin",
    "Horario Inicio": "hora_inicio",
    "Horario Fin": "hora_fin",
    "Salon": "salon",
}

# Hours compared against the schedule. Outside them the building is closed and
# the comparison would only measure the night, inflating every contrast.
HORA_APERTURA = 7
HORA_CIERRE = 22

DIAS = ["lun", "mar", "mie", "jue", "vie", "sab", "dom"]


def _minutos(hhmm: pd.Series) -> pd.Series:
    """Convert times written as HHMM integers into minutes after midnight.

    Args:
        hhmm: Values such as 710 or 1350.

    Returns:
        Minutes after midnight.
    """
    valores = hhmm.astype(int)
    return (valores // 100) * 60 + valores % 100


def cargar_espacios(ruta: Path = CARPETA_CONFIG / "espacios.json") -> dict[str, int]:
    """Map room numbers as written in the schedule to espacio_id.

    Args:
        ruta: The shared space registry.

    Returns:
        A dict from room string ("3308") to espacio_id (308).
    """
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    mapa = {}
    for espacio in datos["espacios"]:
        for nombre in [espacio.get("salon"), *espacio.get("alias", [])]:
            if nombre:
                mapa[str(nombre)] = int(espacio["espacio_id"])
    return mapa


def cargar_horarios(ruta: Path) -> tuple[pd.DataFrame, dict]:
    """Read the schedule export and attach espacio_id.

    Args:
        ruta: CSV exported from the registrar.

    Returns:
        A tuple of (one row per distinct class block, report). Identical rows
        are collapsed because, without the day column, they carry no
        information beyond the first.

    Raises:
        ValueError: If a required column is missing.
    """
    crudo = pd.read_csv(ruta, dtype=str)
    faltan = set(COLUMNAS) - set(crudo.columns)
    if faltan:
        raise ValueError(f"{ruta.name} is missing columns: {sorted(faltan)}")

    df = crudo[list(COLUMNAS)].rename(columns=COLUMNAS)
    filas = len(df)
    df = df.drop_duplicates()

    df["alumnos"] = df["alumnos"].astype(int)
    df["fecha_inicio"] = pd.to_datetime(df["fecha_inicio"], format="%d/%m/%Y")
    # The end date is inclusive in the export; the day after is the exclusive
    # bound that interval arithmetic expects.
    df["fecha_fin"] = pd.to_datetime(df["fecha_fin"], format="%d/%m/%Y") + pd.Timedelta(days=1)
    df["min_inicio"] = _minutos(df["hora_inicio"])
    df["min_fin"] = _minutos(df["hora_fin"])

    mapa = cargar_espacios()
    df[cfg.COLUMNA_ESPACIO] = df["salon"].map(mapa)
    sin_espacio = sorted(df.loc[df[cfg.COLUMNA_ESPACIO].isna(), "salon"].unique())
    df = df.dropna(subset=[cfg.COLUMNA_ESPACIO])
    df[cfg.COLUMNA_ESPACIO] = df[cfg.COLUMNA_ESPACIO].astype(int)

    return df.reset_index(drop=True), {
        "filas_leidas": filas,
        "filas_repetidas": filas - len(crudo[list(COLUMNAS)].drop_duplicates()),
        "bloques_distintos": len(df),
        "salones_sin_espacio": sin_espacio,
    }


def ocupacion_posible(
    horarios: pd.DataFrame, marcas: pd.Series, espacio_id: int
) -> np.ndarray:
    """Students a class could bring to one space at each timestamp.

    Weekdays only: the export does not say which days a block meets, and the
    weekend data show essentially no motion, so counting Saturday would add
    blocks that are almost certainly not there.

    Args:
        horarios: Output of cargar_horarios().
        marcas: Interval start times.
        espacio_id: Space to compute.

    Returns:
        An array aligned with `marcas`: the largest enrolment among the blocks
        active at that time, or 0.
    """
    propios = horarios[horarios[cfg.COLUMNA_ESPACIO] == espacio_id]
    marcas = pd.to_datetime(marcas).reset_index(drop=True)
    minuto = (marcas.dt.hour * 60 + marcas.dt.minute).to_numpy()
    habil = (marcas.dt.dayofweek < 5).to_numpy()
    salida = np.zeros(len(marcas), dtype=float)

    for _, b in propios.iterrows():
        activo = (
            habil
            & (marcas >= b["fecha_inicio"]).to_numpy()
            & (marcas < b["fecha_fin"]).to_numpy()
            & (minuto >= b["min_inicio"])
            & (minuto < b["min_fin"])
        )
        salida[activo] = np.maximum(salida[activo], b["alumnos"])
    return salida


def comparar(datos: pd.DataFrame, horarios: pd.DataFrame) -> pd.DataFrame:
    """Contrast sensor readings inside and outside scheduled blocks, per day.

    Args:
        datos: Cleaned contract frame from ingesta.limpieza.
        horarios: Output of cargar_horarios().

    Returns:
        One row per space and weekday with the mean movement fraction and eCO2
        inside and outside scheduled blocks during opening hours, plus the
        correlation between possible students and each variable.
    """
    filas = []
    for espacio, grupo in datos.groupby(cfg.COLUMNA_ESPACIO):
        grupo = grupo.sort_values(cfg.COLUMNA_TIEMPO).reset_index(drop=True)
        grupo["alumnos"] = ocupacion_posible(horarios, grupo[cfg.COLUMNA_TIEMPO], espacio)
        hora = grupo[cfg.COLUMNA_TIEMPO].dt.hour
        grupo = grupo[(hora >= HORA_APERTURA) & (hora < HORA_CIERRE)]
        grupo = grupo.assign(dia=grupo[cfg.COLUMNA_TIEMPO].dt.dayofweek)

        for dia, d in grupo.groupby("dia"):
            dentro = d["alumnos"] > 0
            fila = {
                cfg.COLUMNA_ESPACIO: espacio,
                "dia": DIAS[dia],
                "intervalos": len(d),
                "con_clase": int(dentro.sum()),
            }
            for variable, corta in (("movimiento", "mov"), ("co2", "co2")):
                fila[f"{corta}_dentro"] = d.loc[dentro, variable].mean()
                fila[f"{corta}_fuera"] = d.loc[~dentro, variable].mean()
                # A constant series has no correlation to report; nan says so
                # instead of a misleading 0.
                valido = d[[variable, "alumnos"]].dropna()
                fila[f"r_{corta}"] = (
                    valido[variable].corr(valido["alumnos"])
                    if valido["alumnos"].nunique() > 1 and valido[variable].nunique() > 1
                    else np.nan
                )
            filas.append(fila)
    return pd.DataFrame(filas)


def main() -> int:
    """Load the schedule and the cleaned data, and print the comparison.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Check the schedule against sensors.")
    parser.add_argument("horario", type=Path, help="schedule CSV export")
    parser.add_argument(
        "--datos", type=Path, default=cfg.DATOS_PROCESADOS / "aulas3.parquet"
    )
    args = parser.parse_args()

    horarios, reporte = cargar_horarios(args.horario)
    print(
        f"Horario: {reporte['filas_leidas']} filas, {reporte['filas_repetidas']} "
        f"repetidas, {reporte['bloques_distintos']} bloques distintos con espacio"
    )
    if reporte["salones_sin_espacio"]:
        print(f"Salones que no estan en espacios.json: {reporte['salones_sin_espacio']}")

    datos = pd.read_parquet(args.datos)
    tabla = comparar(datos, horarios)

    pd.set_option("display.width", 140)
    print(
        f"\nDe {HORA_APERTURA}:00 a {HORA_CIERRE}:00. 'dentro' = intervalos con una "
        "clase posible segun el horario; r = correlacion con los alumnos posibles.\n"
    )
    print(
        tabla.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}" if abs(x) < 10 else f"{x:.0f}",
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
