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
    python sintetico.py            # writes data/raw/sintetico.parquet
    python sintetico.py --dias 90

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-10-01 - Recalibrated the CO2 build-up rate and the PIR activation curve.

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
PERFILES = {
    1: {"nombre": "aula_interior", "escala_luz": 90.0, "aforo": 30, "abre": 7, "cierra": 21},
    2: {"nombre": "aula_ventanal", "escala_luz": 1800.0, "aforo": 35, "abre": 7, "cierra": 21},
    3: {"nombre": "oficina", "escala_luz": 400.0, "aforo": 8, "abre": 8, "cierra": 19},
    4: {"nombre": "laboratorio", "escala_luz": 650.0, "aforo": 20, "abre": 7, "cierra": 22},
    5: {"nombre": "pasillo", "escala_luz": 120.0, "aforo": 5, "abre": 6, "cierra": 23},
}


def _curva_solar(hora: np.ndarray) -> np.ndarray:
    """Compute the fraction of daylight available at each hour.

    Args:
        hora: Hour of day as a float, e.g. 13.5 for 13:30.

    Returns:
        Values in [0, 1]: zero at night, peaking around midday.
    """
    solar = np.sin(np.pi * (hora - 6.5) / 12.0)
    return np.clip(solar, 0.0, None)


def _ocupacion(
    indice: pd.DatetimeIndex, perfil: dict, rng: np.random.Generator
) -> np.ndarray:
    """Estimate how many people are present at each timestamp.

    Args:
        indice: Regular timestamps to generate for.
        perfil: Space profile from PERFILES.
        rng: Seeded random generator.

    Returns:
        Non-negative occupancy counts, one per timestamp.
    """
    hora = indice.hour + indice.minute / 60.0
    habil = indice.dayofweek < 5

    dentro = (hora >= perfil["abre"]) & (hora < perfil["cierra"])
    # A bell centred on mid-shift rather than a step function: people arrive
    # and leave gradually.
    centro = (perfil["abre"] + perfil["cierra"]) / 2.0
    ancho = (perfil["cierra"] - perfil["abre"]) / 3.0
    forma = np.exp(-0.5 * ((hora - centro) / ancho) ** 2)

    base = perfil["aforo"] * forma * dentro * np.where(habil, 1.0, 0.25)
    # Multiplicative noise: classes that empty out, groups arriving together.
    ruido = rng.gamma(shape=4.0, scale=0.25, size=len(indice))
    return np.clip(base * ruido, 0.0, perfil["aforo"] * 1.3)


def _co2(personas: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Generate CO2 with occupancy build-up and ventilation decay.

    Single time-constant model: at each step the excess over outdoor level
    decays by a fixed factor and the people present add an increment.

    Args:
        personas: Occupancy counts per timestep.
        rng: Seeded random generator.

    Returns:
        CO2 in ppm, clipped at the sensor's calibration floor.
    """
    exterior = 420.0
    # Steady state of this model is tasa_persona * n / (1 - decaimiento). With
    # these values 30 people settle around 800 ppm above outdoor, so roughly
    # 1,220 ppm total: the range measured in a full room with decent
    # ventilation.
    tasa_persona = 1.6  # ppm per person per 5-minute step
    decaimiento = 0.94  # about a 1.3 h time constant at 5-minute steps

    exceso = np.zeros(len(personas))
    actual = 0.0
    for i, n in enumerate(personas):
        actual = actual * decaimiento + tasa_persona * n
        exceso[i] = actual

    valores = exterior + exceso + rng.normal(0, 4.0, len(personas))
    # Sensor floor: an NDIR does not report below 400 ppm. The lower tail ends
    # up censored, exactly as in real readings.
    piso = cfg.CENSURADAS_POR_ABAJO["co2"]
    return np.maximum(valores, piso)


def _genera_espacio(
    espacio_id: int, indice: pd.DatetimeIndex, rng: np.random.Generator
) -> pd.DataFrame:
    """Generate the full series of one space.

    Args:
        espacio_id: Key into PERFILES.
        indice: Regular timestamps to generate for.
        rng: Seeded random generator.

    Returns:
        A frame satisfying the data contract, for this space only.
    """
    perfil = PERFILES[espacio_id]
    hora = indice.hour + indice.minute / 60.0
    solar = _curva_solar(np.asarray(hora))
    personas = _ocupacion(indice, perfil, rng)
    ocupado = personas > 0.5

    co2 = _co2(personas, rng)

    # Temperature: a damped outdoor daily cycle plus heat from the occupants.
    temperatura = (
        21.5
        + 2.2 * np.sin(2 * np.pi * (np.asarray(hora) - 9) / 24)
        + 0.045 * personas
        + rng.normal(0, 0.18, len(indice))
    )

    # Humidity: inverse to temperature, plus a contribution from the occupants.
    humedad = (
        52.0
        - 1.6 * (temperatura - 21.5)
        + 0.12 * personas
        + rng.normal(0, 1.1, len(indice))
    )
    humedad = np.clip(humedad, *cfg.LIMITES_FISICOS["humedad"])

    # Light: solar contribution scaled by the space, plus artificial light when
    # occupied. At night and empty it sits at exactly 0, which is what breaks
    # MAPE and has to be handled in the metrics.
    luz = perfil["escala_luz"] * solar * rng.uniform(0.75, 1.0, len(indice))
    luz += np.where(ocupado, perfil["escala_luz"] * 0.18, 0.0)
    luz = np.maximum(luz + rng.normal(0, 2.0, len(indice)), 0.0)
    luz[(solar <= 0.01) & ~ocupado] = 0.0

    # Movement: binary and noisy, like a PIR. The probability grows with
    # occupancy relative to capacity and does not saturate immediately, so
    # there is a gradient between a space holding two people and a full one.
    # The 1% of flips simulates both the false negatives of someone sitting
    # still and the spurious false positives every PIR produces.
    prob = np.clip(personas / (perfil["aforo"] * 0.5), 0.0, 0.92)
    movimiento = (rng.random(len(indice)) < prob).astype(float)
    movimiento = np.where(rng.random(len(indice)) < 0.01, 1.0 - movimiento, movimiento)

    return pd.DataFrame(
        {
            cfg.COLUMNA_TIEMPO: indice,
            cfg.COLUMNA_ESPACIO: espacio_id,
            "co2": co2,
            "temperatura": temperatura,
            "humedad": humedad,
            "luz": luz,
            "movimiento": movimiento,
        }
    )


def _abre_huecos(
    df: pd.DataFrame, rng: np.random.Generator, n_huecos: int = 6
) -> pd.DataFrame:
    """Delete whole stretches of rows, like WiFi telemetry dropouts.

    Rows are removed rather than set to NaN because that is what actually
    happens: if a node loses the network, the record never arrives. This is
    what forces the pipeline to work in contiguous blocks.

    Args:
        df: Complete frame.
        rng: Seeded random generator.
        n_huecos: How many gaps to open per space.

    Returns:
        The frame with those rows dropped.
    """
    partes = []
    for espacio, grupo in df.groupby(cfg.COLUMNA_ESPACIO, sort=True):
        grupo = grupo.sort_values(cfg.COLUMNA_TIEMPO).reset_index(drop=True)
        fuera = np.zeros(len(grupo), dtype=bool)
        for _ in range(n_huecos):
            # Gaps from 30 minutes to 2 days, mixing brief and long outages.
            largo = int(rng.choice([6, 12, 36, 288, 576]))
            if largo >= len(grupo):
                continue
            inicio = int(rng.integers(0, len(grupo) - largo))
            fuera[inicio : inicio + largo] = True
        partes.append(grupo[~fuera])
    return pd.concat(partes, ignore_index=True)


def genera(
    dias: int = 60,
    n_espacios: int = 5,
    semilla: int = cfg.SEMILLA,
    con_huecos: bool = True,
) -> pd.DataFrame:
    """Generate the complete synthetic dataset.

    Args:
        dias: Days of coverage. 60 matches the planned Aulas 3 collection.
        n_espacios: How many spaces to generate, from 1 to 5.
        semilla: Random seed, so runs are reproducible.
        con_huecos: Whether to open gaps in the series. Turning this off helps
            isolate problems, but the realistic case has gaps.

    Returns:
        A frame satisfying the data contract, sorted by space and time.

    Raises:
        ValueError: If n_espacios is outside the available profiles.
    """
    if not 1 <= n_espacios <= len(PERFILES):
        raise ValueError(f"n_espacios must be between 1 and {len(PERFILES)}")

    rng = np.random.default_rng(semilla)
    pasos = dias * 24 * 3600 // cfg.INTERVALO_SEGUNDOS
    indice = pd.date_range(
        "2026-01-05 00:00",  # a Monday, so the weekly cycle starts clean
        periods=pasos,
        freq=f"{cfg.INTERVALO_SEGUNDOS}s",
    )

    df = pd.concat(
        [_genera_espacio(e, indice, rng) for e in range(1, n_espacios + 1)],
        ignore_index=True,
    )
    if con_huecos:
        df = _abre_huecos(df, rng)
    return df.sort_values([cfg.COLUMNA_ESPACIO, cfg.COLUMNA_TIEMPO]).reset_index(
        drop=True
    )


def main() -> int:
    """Generate a dataset from the command line and summarize it.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Generate synthetic test data.")
    parser.add_argument("--dias", type=int, default=60)
    parser.add_argument("--espacios", type=int, default=5)
    parser.add_argument("--semilla", type=int, default=cfg.SEMILLA)
    parser.add_argument("--sin-huecos", action="store_true")
    args = parser.parse_args()

    df = genera(
        dias=args.dias,
        n_espacios=args.espacios,
        semilla=args.semilla,
        con_huecos=not args.sin_huecos,
    )

    cfg.DATOS_CRUDOS.mkdir(parents=True, exist_ok=True)
    salida = cfg.DATOS_CRUDOS / "sintetico.parquet"
    df.to_parquet(salida, index=False)

    print(f"Escrito {salida.relative_to(cfg.RAIZ)}")
    print(f"  {len(df):,} filas | {df[cfg.COLUMNA_ESPACIO].nunique()} espacios")
    print(f"  de {df[cfg.COLUMNA_TIEMPO].min()} a {df[cfg.COLUMNA_TIEMPO].max()}")
    print()
    resumen = df.groupby(cfg.COLUMNA_ESPACIO)[cfg.VARIABLES].agg(["mean", "min", "max"])
    print(resumen.round(1).to_string())
    print()
    print("Ceros por variable (el MAPE se indefine donde el valor real es 0):")
    for v in cfg.VARIABLES:
        print(f"  {v:12} {100 * (df[v] == 0).mean():5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
