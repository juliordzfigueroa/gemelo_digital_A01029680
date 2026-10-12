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
    python compare_models.py
    python compare_models.py --data ../data/processed/aulas3.parquet

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

from common import config as cfg
from common import data as dat
from common import metrics as met
from gru import train as ent_gru
from sarimax import train as ent_sarimax


def run_gru(df, implementation: str, epochs: int, patience: int) -> dict:
    """Train the GRU and return its test metrics.

    Args:
        df: Frame satisfying the data contract.
        implementacion: "cho" or "pytorch".
        epocas: Maximum epochs.
        paciencia: Epochs without improvement before early stopping.

    Returns:
        A dict with the metrics, the horizon breakdown and the elapsed seconds.
    """
    output = ent_gru.train(
        df.copy(),
        epochs=epochs,
        patience=patience,
        quiet=True,
        implementation=implementation,
    )
    return {
        "results": output["results"],
        "horizon": output["horizon"],
        "seconds": output["seconds"],
        "detail": f"GRU [128, 64], implementacion {implementation}",
    }


def run_sarimax(df, seasonality: str, order: tuple[int, int, int]) -> dict:
    """Fit the SARIMAX baseline and return its test metrics.

    Args:
        df: Frame satisfying the data contract.
        estacionalidad: "fourier" or "estacional".
        orden: Non-seasonal (p, d, q).

    Returns:
        A dict with the metrics, the horizon breakdown and the elapsed seconds.
    """
    import time

    preprocessed, _ = dat.preprocess(df.copy())
    grids = ent_sarimax.grid_by_space(preprocessed)
    origins = ent_sarimax.test_windows(preprocessed)
    target = ent_sarimax.real_targets(grids, origins)

    # Same guard as the standalone script: refuse to compare models that were
    # not scored on identical windows.
    ent_sarimax.check_against_gru(df.copy(), target)

    t0 = time.time()
    prediction, _ = ent_sarimax.fit_all(
        preprocessed, grids, origins, seasonality, order
    )
    return {
        "results": met.evaluate(target, prediction),
        "horizon": met.by_horizon(target, prediction),
        "seconds": time.time() - t0,
        "detail": f"SARIMAX {order}, estacionalidad {seasonality}",
    }


def print_comparison(gru: dict, sarimax: dict) -> None:
    """Print the side-by-side table and say which model wins.

    Args:
        gru: Output of run_gru().
        sarimax: Output of run_sarimax().
    """
    print(f"\n{'=' * 78}\n  GRU CONTRA SARIMAX\n{'=' * 78}\n")
    print(f"  {gru['detail']}")
    print(f"  {sarimax['detail']}\n")

    print(f"  {'variable':14} {'unit':7} {'MAE GRU':>12} {'MAE SARIMAX':>13}   mejora")
    print("  " + "-" * 64)
    for variable in cfg.ACTIVE_VARIABLES:
        a = gru["results"]["by_variable"][variable]["mae"]
        b = sarimax["results"]["by_variable"][variable]["mae"]
        improvement = 100 * (b - a) / b if b else float("nan")
        unit = cfg.UNITS.get(variable, "")
        print(f"  {variable:14} {unit:7} {a:12.4f} {b:13.4f}   {improvement:+6.1f}%")

    g, s = gru["results"]["global"], sarimax["results"]["global"]
    print("  " + "-" * 64)
    global_improvement = 100 * (s["mae"] - g["mae"]) / s["mae"]
    print(f"  {'MAE global':22} {g['mae']:12.4f} {s['mae']:13.4f}   {global_improvement:+6.1f}%")
    print(f"  {'R2 global':22} {g['r2']:12.4f} {s['r2']:13.4f}")
    print(f"  {'seconds':22} {gru['seconds']:12.1f} {sarimax['seconds']:13.1f}")

    print()
    if g["r2"] > s["r2"]:
        print(
            f"  El GRU supera al baseline: R2 de {g['r2']:.4f} contra {s['r2']:.4f}, "
            f"y un MAE global\n  {global_improvement:.1f}% menor. Eso es evidencia de "
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
    parser.add_argument(
        "--data", default=None, help="contract frame; defaults to the cleaned sensor data"
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--implementation", choices=("cho", "pytorch"), default="pytorch")
    parser.add_argument(
        "--seasonality", choices=("fourier", "seasonal"), default="fourier"
    )
    parser.add_argument("--order", default="2,0,2")
    args = parser.parse_args()

    order = tuple(int(x) for x in args.order.split(","))
    df = dat.load_data(args.data)
    print(f"Datos: {len(df):,} filas, espacios {sorted(df[cfg.SPACE_COLUMN].unique())}")

    print("\n== Entrenando el GRU ==")
    gru = run_gru(df, args.implementation, args.epochs, args.patience)
    print(f"  listo en {gru['seconds']:.1f} s")

    print("\n== Ajustando SARIMAX ==")
    sarimax = run_sarimax(df, args.seasonality, order)

    print_comparison(gru, sarimax)

    cfg.PROCESSED_DATA.mkdir(parents=True, exist_ok=True)
    path = cfg.PROCESSED_DATA / "comparison_gru_sarimax.json"
    path.write_text(
        json.dumps({"gru": gru, "sarimax": sarimax}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nGuardado en {path.relative_to(cfg.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
