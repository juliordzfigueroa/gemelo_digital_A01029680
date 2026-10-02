"""Fit the SARIMAX baseline and evaluate it on the GRU's own test split.

One model per space and variable, so five variables across five spaces means 25
independent fits. Each one is trained on the training partition only and then
forecasts 6 steps ahead from the end of every test window, which is exactly the
task the GRU is scored on: see 12 real steps, produce the next 6 unaided.

The comparison is only worth anything if both models are scored on identical
targets, so this script rebuilds the same windows and then asserts that its
target array matches the one the GRU pipeline produces, aborting if it does
not.

Usage, from the src/ directory:
    python -m sarimax.entrenar
    python -m sarimax.entrenar --dias 60 --estacionalidad estacional
    python -m sarimax.entrenar --paso 10        # subsample test windows

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-10-01 - Initial implementation.

Reference:
    Sections 4.2.5 and 4.2.7.1 of the research document.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from comun import config as cfg
from comun import datos as dat
from comun import metricas as met
from sarimax import modelo as mod


def rejilla_por_espacio(df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Place each space on one continuous 5-minute grid, gaps included as NaN.

    SARIMAX assumes evenly spaced observations, so concatenating blocks across
    a two-day outage would misrepresent the time axis. Leaving the gaps as
    missing values is the correct treatment: the Kalman filter handles absent
    observations natively, carrying the state forward without pretending data
    arrived.

    Args:
        df: Frame returned by comun.datos.preprocesar().

    Returns:
        A dict from space id to a frame indexed by a gap-free timestamp grid.
    """
    paso = f"{cfg.INTERVALO_SEGUNDOS}s"
    salida = {}
    for espacio, grupo in df.groupby(cfg.COLUMNA_ESPACIO, sort=True):
        grupo = grupo.sort_values(cfg.COLUMNA_TIEMPO).set_index(cfg.COLUMNA_TIEMPO)
        rejilla = pd.date_range(grupo.index.min(), grupo.index.max(), freq=paso)
        salida[int(espacio)] = grupo.reindex(rejilla)
    return salida


def ventanas_de_prueba(df: pd.DataFrame) -> dict[int, list[pd.Timestamp]]:
    """Enumerate the test windows, using the same rule as the GRU pipeline.

    A window is valid only if its 18 consecutive steps sit in one block, carry
    no missing value in any active variable, and all belong to the test
    partition. Those are the three conditions comun.datos.VentanasAmbientales
    applies, restated here because SARIMAX consumes raw series rather than
    tensors.

    Args:
        df: Frame returned by comun.datos.preprocesar().

    Returns:
        A dict from space id to the origin timestamps, meaning the last
        observed step of each window, in the order the GRU sees them.
    """
    largo = cfg.PASOS_ENTRADA + cfg.PASOS_SALIDA
    origenes: dict[int, list[pd.Timestamp]] = {}

    for _, bloque in df.groupby("bloque_id", sort=True):
        bloque = bloque.sort_values(cfg.COLUMNA_TIEMPO)
        if len(bloque) < largo:
            continue
        espacio = int(bloque[cfg.COLUMNA_ESPACIO].iloc[0])
        completa = bloque[cfg.VARIABLES_ACTIVAS].notna().all(axis=1).to_numpy()
        es_prueba = (bloque["particion"] == "prueba").to_numpy()
        tiempos = bloque[cfg.COLUMNA_TIEMPO].to_numpy()

        valida = completa & es_prueba
        for inicio in range(len(bloque) - largo + 1):
            if not valida[inicio : inicio + largo].all():
                continue
            # The origin is the last step the model is allowed to observe.
            origenes.setdefault(espacio, []).append(
                pd.Timestamp(tiempos[inicio + cfg.PASOS_ENTRADA - 1])
            )
    return origenes


def objetivos_reales(
    rejillas: dict[int, pd.DataFrame], origenes: dict[int, list[pd.Timestamp]]
) -> np.ndarray:
    """Collect the ground truth for every window, in physical units.

    Args:
        rejillas: Output of rejilla_por_espacio().
        origenes: Output of ventanas_de_prueba().

    Returns:
        An array shaped (n_ventanas, PASOS_SALIDA, n_variables), ordered by
        space and then by time.
    """
    bloques = []
    for espacio in sorted(origenes):
        rejilla = rejillas[espacio]
        posicion = {t: i for i, t in enumerate(rejilla.index)}
        valores = rejilla[cfg.VARIABLES_ACTIVAS].to_numpy(dtype=float)
        for origen in origenes[espacio]:
            i = posicion[origen]
            bloques.append(valores[i + 1 : i + 1 + cfg.PASOS_SALIDA])
    return np.stack(bloques)


def verificar_contra_gru(df_crudo: pd.DataFrame, objetivo: np.ndarray) -> None:
    """Abort unless SARIMAX is scored on exactly the GRU's test targets.

    Runs the GRU preprocessing, denormalizes its test targets and compares them
    element by element. Without this check the two models could silently end up
    on different windows, and the whole comparison would be meaningless while
    still printing a tidy table.

    Args:
        df_crudo: The original frame, before preprocessing.
        objetivo: The target array this script assembled.

    Raises:
        RuntimeError: If the shapes or the values do not match.
    """
    conjuntos, normalizador, _, indice_espacios = dat.preparar(df_crudo)
    prueba = conjuntos["prueba"]
    inverso = {v: k for k, v in indice_espacios.items()}
    ids = np.array([inverso[int(i)] for i in prueba.espacio.numpy()])
    objetivo_gru = normalizador.desnormalizar(
        prueba.y.numpy(), cfg.VARIABLES_ACTIVAS, ids
    )

    if objetivo_gru.shape != objetivo.shape:
        raise RuntimeError(
            f"window counts differ: GRU {objetivo_gru.shape}, "
            f"SARIMAX {objetivo.shape}; the two models are not being scored on "
            "the same test set"
        )
    if not np.allclose(objetivo_gru, objetivo, rtol=1e-4, atol=1e-4):
        peor = float(np.abs(objetivo_gru - objetivo).max())
        raise RuntimeError(
            f"targets differ by up to {peor:.6f} despite matching shapes; "
            "the window alignment is wrong"
        )
    print(f"  Verificado: mismas {len(objetivo):,} ventanas de prueba que el GRU.")


def ajustar_todo(
    df: pd.DataFrame,
    rejillas: dict[int, pd.DataFrame],
    origenes: dict[int, list[pd.Timestamp]],
    estacionalidad: str,
    orden: tuple[int, int, int],
) -> tuple[np.ndarray, dict]:
    """Fit one model per space and variable and forecast every test window.

    Args:
        df: Frame returned by comun.datos.preprocesar().
        rejillas: Output of rejilla_por_espacio().
        origenes: Output of ventanas_de_prueba().
        estacionalidad: "fourier" or "estacional".
        orden: Non-seasonal (p, d, q).

    Returns:
        A tuple of (predictions shaped like the targets, per-model diagnostics).
    """
    espacios = sorted(origenes)
    n_ventanas = sum(len(origenes[e]) for e in espacios)
    prediccion = np.empty(
        (n_ventanas, cfg.PASOS_SALIDA, len(cfg.VARIABLES_ACTIVAS)), dtype=float
    )
    diagnostico = {}

    for j, variable in enumerate(cfg.VARIABLES_ACTIVAS):
        fila = 0
        for espacio in espacios:
            rejilla = rejillas[espacio]
            serie = rejilla[variable].to_numpy(dtype=float)
            entrenamiento = serie[(rejilla["particion"] == "entrenamiento").to_numpy()]

            t0 = time.time()
            baseline = mod.BaselineSARIMAX(
                orden=orden, estacionalidad=estacionalidad
            ).ajustar(entrenamiento)

            posicion = {t: i for i, t in enumerate(rejilla.index)}
            indices = np.array([posicion[t] for t in origenes[espacio]])
            salida = baseline.pronosticar_desde(serie, indices, cfg.PASOS_SALIDA)
            prediccion[fila : fila + len(indices), :, j] = salida
            fila += len(indices)

            clave = f"{variable}@espacio{espacio}"
            diagnostico[clave] = {
                "orden": baseline.resumen_orden(),
                "aic": baseline.aic,
                "n_entrenamiento": int(len(entrenamiento)),
                "segundos": round(time.time() - t0, 2),
            }
            print(
                f"    {clave:28} {baseline.resumen_orden():22} "
                f"AIC {baseline.aic:10.1f}  {diagnostico[clave]['segundos']:6.2f} s"
            )
    return prediccion, diagnostico


def main() -> int:
    """Run the full baseline from the command line.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Fit the SARIMAX baseline.")
    parser.add_argument("--datos", default=None)
    parser.add_argument("--dias", type=int, default=20)
    parser.add_argument("--espacios", type=int, default=2)
    parser.add_argument(
        "--estacionalidad",
        choices=("fourier", "estacional"),
        default="fourier",
        help="'fourier' carries the daily cycle as exogenous regressors; "
        "'estacional' uses s=288 as section 4.2.7.1 states, and is far slower",
    )
    parser.add_argument("--orden", default="2,0,2", help="non-seasonal p,d,q")
    parser.add_argument("--etiqueta", default="sarimax")
    args = parser.parse_args()

    orden = tuple(int(x) for x in args.orden.split(","))
    if len(orden) != 3:
        raise SystemExit(f"--orden needs three values, got {args.orden!r}")

    if args.datos:
        df_crudo = (
            pd.read_parquet(args.datos)
            if args.datos.endswith(".parquet")
            else pd.read_csv(args.datos, parse_dates=[cfg.COLUMNA_TIEMPO])
        )
    else:
        from comun import sintetico

        print(f"Sin --datos: generando {args.dias} dias sinteticos.")
        df_crudo = sintetico.genera(dias=args.dias, n_espacios=args.espacios)

    df, _ = dat.preprocesar(df_crudo)
    rejillas = rejilla_por_espacio(df)
    origenes = ventanas_de_prueba(df)
    objetivo = objetivos_reales(rejillas, origenes)

    print(f"\nEstacionalidad: {args.estacionalidad} | orden no estacional: {orden}")
    print(f"Modelos a ajustar: {len(cfg.VARIABLES_ACTIVAS)} variables x "
          f"{len(origenes)} espacios = "
          f"{len(cfg.VARIABLES_ACTIVAS) * len(origenes)}")
    verificar_contra_gru(df_crudo, objetivo)

    print("\n  Ajustando:")
    t0 = time.time()
    prediccion, diagnostico = ajustar_todo(
        df, rejillas, origenes, args.estacionalidad, orden
    )
    duracion = time.time() - t0
    print(f"\n  Total: {duracion:.1f} s")

    resultados = met.evaluar(objetivo, prediccion)
    met.imprimir(resultados, f"SARIMAX ({args.estacionalidad}) sobre el conjunto de prueba")

    cfg.MODELOS.mkdir(parents=True, exist_ok=True)
    ruta = cfg.MODELOS / f"{args.etiqueta}_metadatos.json"
    ruta.write_text(
        json.dumps(
            {
                "estacionalidad": args.estacionalidad,
                "orden": list(orden),
                "n_ventanas": int(len(objetivo)),
                "segundos": round(duracion, 1),
                "metricas_prueba": resultados,
                "mae_por_horizonte": met.por_horizonte(objetivo, prediccion),
                "modelos": diagnostico,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nGuardado en {ruta.relative_to(cfg.RAIZ)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
