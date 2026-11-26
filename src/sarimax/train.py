"""Fit the SARIMAX baseline and evaluate it on the GRU's own test split.

One model per variable and space with test windows: with the first week of
data, 5 variables in 4 spaces, 20 fits. Each one is trained on the training
partition only and then forecasts 6 steps ahead from the end of every test
window, which is exactly the task the GRU is scored on: see 12 real steps,
produce the next 6 unaided.

The comparison is only worth anything if both models are scored on identical
targets. The test windows and their ground truth come from the same class the
GRU uses (common.data.EnvironmentalWindows), built on the frame in physical
units, so they are the same windows by construction.

Usage, from the src/ directory:
    python -m sarimax.train

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-11-26 - Ground truth taken from the windows; forecasts placed by row.

Reference:
    Sections 4.2.5 and 4.2.7.1 of the research document.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from common import config as cfg
from common import data as dat
from common import metrics as met
from sarimax import model as mod


def grid_by_space(df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Place each space on one continuous 5-minute grid, gaps included as NaN.

    SARIMAX assumes evenly spaced observations, so concatenating blocks across
    a two-day outage would misrepresent the time axis. Leaving the gaps as
    missing values is the correct treatment: the Kalman filter handles absent
    observations natively, carrying the state forward without pretending data
    arrived.

    Args:
        df: Frame returned by common.data.preprocess().

    Returns:
        A dict from space id to a frame indexed by a gap-free timestamp grid.
    """
    step = f"{cfg.INTERVAL_SECONDS}s"
    output = {}
    for space, group in df.groupby(cfg.SPACE_COLUMN, sort=True):
        group = group.sort_values(cfg.TIME_COLUMN).set_index(cfg.TIME_COLUMN)
        grid = pd.date_range(group.index.min(), group.index.max(), freq=step)
        output[int(space)] = group.reindex(grid)
    return output


def test_windows(
    df: pd.DataFrame, feature_columns: list[str]
) -> dat.EnvironmentalWindows:
    """Build the GRU's test windows on the frame in physical units.

    Args:
        df: Frame returned by common.data.preprocess(), in physical units.
        feature_columns: Input columns returned by the same call.

    Returns:
        The windows. `origins` says where each one ends and `y` holds its
        ground truth, already in physical units because the frame is.
    """
    spaces = sorted(int(s) for s in df[cfg.SPACE_COLUMN].unique())
    space_index = {space: i for i, space in enumerate(spaces)}
    return dat.EnvironmentalWindows(df, "test", feature_columns, space_index)


def fit_all(
    grids: dict[int, pd.DataFrame], windows: dat.EnvironmentalWindows
) -> tuple[np.ndarray, dict]:
    """Fit one model per space and variable and forecast every test window.

    Args:
        grids: Output of grid_by_space().
        windows: Output of test_windows().

    Returns:
        A tuple of (predictions shaped like windows.y, per-model diagnostics).
    """
    prediction = np.empty(windows.y.shape, dtype=float)
    diagnostics = {}

    for space in sorted({space for space, _ in windows.origins}):
        grid = grids[space]
        # Which windows belong to this space, and where each one's last
        # observed step sits on the grid.
        rows = [i for i, (s, _) in enumerate(windows.origins) if s == space]
        origins = grid.index.get_indexer([windows.origins[i][1] for i in rows])

        # Non-training steps become NaN instead of being cut out. Training
        # days need not be contiguous under the day-based split, and cutting
        # would both splice distant days together and shift the phase of the
        # Fourier terms, which are indexed by position. The Kalman filter
        # skips NaN natively.
        is_train = (grid["partition"] == "train").to_numpy()

        for j, variable in enumerate(cfg.ACTIVE_VARIABLES):
            series = grid[variable].to_numpy(dtype=float)
            training = np.where(is_train, series, np.nan)

            t0 = time.time()
            baseline = mod.BaselineSARIMAX().fit(training)
            output = baseline.forecast_from(series, origins, cfg.OUTPUT_STEPS)
            prediction[rows, :, j] = output

            key = f"{variable}@space{space}"
            aic = float(baseline.result.aic)
            seconds = round(time.time() - t0, 2)
            diagnostics[key] = {
                "aic": aic,
                "n_train": int(is_train.sum()),
                "seconds": seconds,
            }
            print(f"    {key:28} AIC {aic:10.1f}  {seconds:6.2f} s")
    return prediction, diagnostics


def run(raw_df: pd.DataFrame) -> dict:
    """Fit every model and score it on the test windows.

    Args:
        raw_df: Frame satisfying the data contract.

    Returns:
        A dict with the targets, the predictions, the metrics, the horizon
        breakdown, the per-model diagnostics and the elapsed seconds.
    """
    df, feature_columns = dat.preprocess(raw_df)
    grids = grid_by_space(df)
    windows = test_windows(df, feature_columns)
    target = windows.y.numpy().astype(float)

    n_spaces = len({space for space, _ in windows.origins})
    n_variables = len(cfg.ACTIVE_VARIABLES)
    print(f"\nOrden {mod.ORDER} + Fourier K={mod.HARMONICS}")
    print(
        f"Modelos a ajustar: {n_variables} variables x {n_spaces} espacios "
        f"= {n_variables * n_spaces}"
    )
    print(f"Ventanas de prueba: {len(target):,} (las mismas del GRU)")

    print("\n  Ajustando:")
    t0 = time.time()
    prediction, diagnostics = fit_all(grids, windows)
    return {
        "target": target,
        "prediction": prediction,
        "results": met.evaluate(target, prediction),
        "horizon": met.by_horizon(target, prediction),
        "models": diagnostics,
        "seconds": time.time() - t0,
    }


def main() -> int:
    """Run the full baseline from the command line.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Fit the SARIMAX baseline.")
    parser.add_argument(
        "--data", default=None, help="contract frame; defaults to the cleaned sensor data"
    )
    parser.add_argument("--label", default="sarimax")
    args = parser.parse_args()

    output = run(dat.load_data(args.data))
    print(f"\n  Total: {output['seconds']:.1f} s")
    met.print_report(output["results"], "SARIMAX sobre el conjunto de prueba")

    cfg.MODELS.mkdir(parents=True, exist_ok=True)
    path = cfg.MODELS / f"{args.label}_metadata.json"
    metadata = {
        "order": list(mod.ORDER),
        "harmonics": mod.HARMONICS,
        "n_windows": int(len(output["target"])),
        "seconds": round(output["seconds"], 1),
        "test_metrics": output["results"],
        "mae_by_horizon": output["horizon"],
        "models": output["models"],
    }
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGuardado en {path.relative_to(cfg.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
