"""Fit the SARIMAX baseline and evaluate it on the GRU's own test split.

One model per space and variable, so five variables across five spaces means 25
independent fits. Each one is trained on the training partition only and then
forecasts 6 steps ahead from the end of every test window, which is exactly the
task the GRU is scored on: see 12 real steps, produce the next 6 unaided.

The comparison is only worth anything if both models are scored on identical
targets. The test windows come from the same class the GRU uses
(common.data.EnvironmentalWindows), built on the frame in physical units, so
they are the same windows by construction.

Usage, from the src/ directory:
    python -m sarimax.train

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-11-25 - Took the test windows from the GRU's windows class.

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


def test_origins(
    df: pd.DataFrame, feature_columns: list[str]
) -> dict[int, list[pd.Timestamp]]:
    """List the GRU's test windows by space.

    Args:
        df: Frame returned by common.data.preprocess(), in physical units.
        feature_columns: Input columns returned by the same call.

    Returns:
        A dict from space id to the origin of each test window (its last
        observed step), in the order the GRU sees them.
    """
    spaces = sorted(int(s) for s in df[cfg.SPACE_COLUMN].unique())
    space_index = {space: i for i, space in enumerate(spaces)}
    windows = dat.EnvironmentalWindows(df, "test", feature_columns, space_index)

    origins: dict[int, list[pd.Timestamp]] = {}
    for space, origin in windows.origins:
        origins.setdefault(space, []).append(origin)
    return origins


def real_targets(
    grids: dict[int, pd.DataFrame], origins: dict[int, list[pd.Timestamp]]
) -> np.ndarray:
    """Collect the ground truth for every window, in physical units.

    Args:
        grids: Output of grid_by_space().
        origins: Output of test_origins().

    Returns:
        An array shaped (n_windows, OUTPUT_STEPS, n_variables), ordered by
        space and then by time.
    """
    blocks = []
    for space in sorted(origins):
        grid = grids[space]
        position = {t: i for i, t in enumerate(grid.index)}
        values = grid[cfg.ACTIVE_VARIABLES].to_numpy(dtype=float)
        for origin in origins[space]:
            i = position[origin]
            blocks.append(values[i + 1 : i + 1 + cfg.OUTPUT_STEPS])
    return np.stack(blocks)


def fit_all(
    grids: dict[int, pd.DataFrame], origins: dict[int, list[pd.Timestamp]]
) -> tuple[np.ndarray, dict]:
    """Fit one model per space and variable and forecast every test window.

    Args:
        grids: Output of grid_by_space().
        origins: Output of test_origins().

    Returns:
        A tuple of (predictions shaped like the targets, per-model diagnostics).
    """
    spaces = sorted(origins)
    n_windows = sum(len(origins[space]) for space in spaces)
    prediction = np.empty(
        (n_windows, cfg.OUTPUT_STEPS, len(cfg.ACTIVE_VARIABLES)), dtype=float
    )
    diagnostics = {}

    for j, variable in enumerate(cfg.ACTIVE_VARIABLES):
        row = 0
        for space in spaces:
            grid = grids[space]
            series = grid[variable].to_numpy(dtype=float)
            # Non-training steps become NaN instead of being cut out. Training
            # days need not be contiguous under the day-based split, and
            # cutting would both splice distant days together and shift the
            # phase of the Fourier terms, which are indexed by position. The
            # Kalman filter skips NaN natively.
            is_train = (grid["partition"] == "train").to_numpy()
            training = np.where(is_train, series, np.nan)

            t0 = time.time()
            baseline = mod.BaselineSARIMAX().fit(training)

            position = {t: i for i, t in enumerate(grid.index)}
            indices = np.array([position[t] for t in origins[space]])
            output = baseline.forecast_from(series, indices, cfg.OUTPUT_STEPS)
            prediction[row : row + len(indices), :, j] = output
            row += len(indices)

            key = f"{variable}@space{space}"
            diagnostics[key] = {
                "aic": baseline.aic,
                "n_train": int(is_train.sum()),
                "seconds": round(time.time() - t0, 2),
            }
            print(
                f"    {key:28} AIC {baseline.aic:10.1f}  "
                f"{diagnostics[key]['seconds']:6.2f} s"
            )
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
    origins = test_origins(df, feature_columns)
    target = real_targets(grids, origins)

    print(f"\nOrden {mod.ORDER} + Fourier K={mod.HARMONICS}")
    print(
        f"Modelos a ajustar: {len(cfg.ACTIVE_VARIABLES)} variables x "
        f"{len(origins)} espacios = {len(cfg.ACTIVE_VARIABLES) * len(origins)}"
    )
    print(f"Ventanas de prueba: {len(target):,} (las mismas del GRU)")

    print("\n  Ajustando:")
    t0 = time.time()
    prediction, diagnostics = fit_all(grids, origins)
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
