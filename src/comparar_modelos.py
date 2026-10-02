"""Compare the GRU against the SARIMAX baseline on the same test windows.

This is the comparison section 4.2.5 calls for. The two models are trained
independently and by different criteria, and only their predictions meet:

    GRU      minimizes MSE by backpropagation, updating weights step by step.
    SARIMAX  maximizes likelihood through a Kalman filter, solving for its
             coefficients in one go.

Neither feeds the other. There is no loss value that travels between them, and
forcing one would be counterproductive: training the GRU to imitate SARIMAX
would cap it at the baseline's accuracy, which is the opposite of what a
baseline is for. What they share is the data, the partitions, the test windows
and the metric code, and that is what makes the numbers comparable.

Usage, from the src/ directory:
    python comparar_modelos.py
    python comparar_modelos.py --dias 60 --espacios 5

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-10-01 - Initial implementation.

Reference:
    Section 4.2.5 of the research document.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from comun import config as cfg
from comun import datos as dat
from comun import metricas as met
from comun import sintetico
from gru import entrenar as ent_gru
from sarimax import entrenar as ent_sarimax


def correr_gru(df, implementacion: str, epocas: int, paciencia: int) -> dict:
    """Train the GRU and return its test metrics.

    Args:
        df: Frame satisfying the data contract.
        implementacion: "cho" or "pytorch".
        epocas: Maximum epochs.
        paciencia: Epochs without improvement before early stopping.

    Returns:
        A dict with the metrics, the horizon breakdown and the elapsed seconds.
    """
    salida = ent_gru.entrenar(
        df.copy(),
        epocas=epocas,
        paciencia=paciencia,
        silencioso=True,
        implementacion=implementacion,
    )
    return {
        "resultados": salida["resultados"],
        "horizonte": salida["horizonte"],
        "segundos": salida["segundos"],
        "detalle": f"GRU [128, 64], implementacion {implementacion}",
    }


def correr_sarimax(df, estacionalidad: str, orden: tuple[int, int, int]) -> dict:
    """Fit the SARIMAX baseline and return its test metrics.

    Args:
        df: Frame satisfying the data contract.
        estacionalidad: "fourier" or "estacional".
        orden: Non-seasonal (p, d, q).

    Returns:
        A dict with the metrics, the horizon breakdown and the elapsed seconds.
    """
    import time

    preprocesado, _ = dat.preprocesar(df.copy())
    rejillas = ent_sarimax.rejilla_por_espacio(preprocesado)
    origenes = ent_sarimax.ventanas_de_prueba(preprocesado)
    objetivo = ent_sarimax.objetivos_reales(rejillas, origenes)

    # Same guard as the standalone script: refuse to compare models that were
    # not scored on identical windows.
    ent_sarimax.verificar_contra_gru(df.copy(), objetivo)

    t0 = time.time()
    prediccion, _ = ent_sarimax.ajustar_todo(
        preprocesado, rejillas, origenes, estacionalidad, orden
    )
    return {
        "resultados": met.evaluar(objetivo, prediccion),
        "horizonte": met.por_horizonte(objetivo, prediccion),
        "segundos": time.time() - t0,
        "detalle": f"SARIMAX {orden}, estacionalidad {estacionalidad}",
    }


def imprimir_comparacion(gru: dict, sarimax: dict) -> None:
    """Print the side-by-side table and say which model wins.

    Args:
        gru: Output of correr_gru().
        sarimax: Output of correr_sarimax().
    """
    print(f"\n{'=' * 78}\n  GRU CONTRA SARIMAX\n{'=' * 78}\n")
    print(f"  {gru['detalle']}")
    print(f"  {sarimax['detalle']}\n")

    print(f"  {'variable':14} {'unidad':7} {'MAE GRU':>12} {'MAE SARIMAX':>13}   mejora")
    print("  " + "-" * 64)
    for variable in cfg.VARIABLES_ACTIVAS:
        a = gru["resultados"]["por_variable"][variable]["mae"]
        b = sarimax["resultados"]["por_variable"][variable]["mae"]
        mejora = 100 * (b - a) / b if b else float("nan")
        unidad = cfg.UNIDADES.get(variable, "")
        print(f"  {variable:14} {unidad:7} {a:12.4f} {b:13.4f}   {mejora:+6.1f}%")

    g, s = gru["resultados"]["global"], sarimax["resultados"]["global"]
    print("  " + "-" * 64)
    mejora_global = 100 * (s["mae"] - g["mae"]) / s["mae"]
    print(f"  {'MAE global':22} {g['mae']:12.4f} {s['mae']:13.4f}   {mejora_global:+6.1f}%")
    print(f"  {'R2 global':22} {g['r2']:12.4f} {s['r2']:13.4f}")
    print(f"  {'segundos':22} {gru['segundos']:12.1f} {sarimax['segundos']:13.1f}")

    print()
    if g["r2"] > s["r2"]:
        print(
            f"  El GRU supera al baseline: R2 de {g['r2']:.4f} contra {s['r2']:.4f}, "
            f"y un MAE global\n  {mejora_global:.1f}% menor. Eso es evidencia de "
            "que la red captura estructura que un\n  modelo lineal univariado no "
            "alcanza."
        )
    else:
        print(
            f"  ATENCION: el baseline iguala o supera al GRU (R2 {s['r2']:.4f} "
            f"contra {g['r2']:.4f}).\n  Con un modelo de ~95 mil parametros "
            "perdiendo contra uno estadistico, lo primero\n  que hay que revisar "
            "es el pipeline, no los hiperparametros."
        )


def main() -> int:
    """Run both models and print the comparison.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Compare the GRU and SARIMAX.")
    parser.add_argument("--dias", type=int, default=20)
    parser.add_argument("--espacios", type=int, default=2)
    parser.add_argument("--epocas", type=int, default=30)
    parser.add_argument("--paciencia", type=int, default=8)
    parser.add_argument("--implementacion", choices=("cho", "pytorch"), default="pytorch")
    parser.add_argument(
        "--estacionalidad", choices=("fourier", "estacional"), default="fourier"
    )
    parser.add_argument("--orden", default="2,0,2")
    args = parser.parse_args()

    orden = tuple(int(x) for x in args.orden.split(","))
    df = sintetico.genera(dias=args.dias, n_espacios=args.espacios)
    print(f"Datos: {len(df):,} filas, {args.espacios} espacios, {args.dias} dias")

    print("\n== Entrenando el GRU ==")
    gru = correr_gru(df, args.implementacion, args.epocas, args.paciencia)
    print(f"  listo en {gru['segundos']:.1f} s")

    print("\n== Ajustando SARIMAX ==")
    sarimax = correr_sarimax(df, args.estacionalidad, orden)

    imprimir_comparacion(gru, sarimax)

    cfg.DATOS_PROCESADOS.mkdir(parents=True, exist_ok=True)
    ruta = cfg.DATOS_PROCESADOS / "comparacion_gru_sarimax.json"
    ruta.write_text(
        json.dumps({"gru": gru, "sarimax": sarimax}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nGuardado en {ruta.relative_to(cfg.RAIZ)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
