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
    python -m sarimax.train
    python -m sarimax.train --days 60 --seasonality seasonal
    python -m sarimax.train --paso 10        # subsample test windows

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
        df: Frame returned by common.data.preprocesar().

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


def test_windows(df: pd.DataFrame) -> dict[int, list[pd.Timestamp]]:
    """Enumerate the test windows, using the same rule as the GRU pipeline.

    A window is valid only if its 18 consecutive steps sit in one block, carry
    no missing value in any active variable, and all belong to the test
    partition. Those are the three conditions common.data.EnvironmentalWindows
    applies, restated here because SARIMAX consumes raw series rather than
    tensors.

    Args:
        df: Frame returned by common.data.preprocesar().

    Returns:
        A dict from space id to the origin timestamps, meaning the last
        observed step of each window, in the order the GRU sees them.
    """
    length = cfg.INPUT_STEPS + cfg.OUTPUT_STEPS
    origins: dict[int, list[pd.Timestamp]] = {}

    for _, block in df.groupby("block_id", sort=True):
        block = block.sort_values(cfg.TIME_COLUMN)
        if len(block) < length:
            continue
        space = int(block[cfg.SPACE_COLUMN].iloc[0])
        complete = block[cfg.ACTIVE_VARIABLES].notna().all(axis=1).to_numpy()
        is_test = (block["partition"] == "test").to_numpy()
        times = block[cfg.TIME_COLUMN].to_numpy()

        valid = complete & is_test
        for start in range(len(block) - length + 1):
            if not valid[start : start + length].all():
                continue
            # The origin is the last step the model is allowed to observe.
            origins.setdefault(space, []).append(
                pd.Timestamp(times[start + cfg.INPUT_STEPS - 1])
            )
    return origins


def real_targets(
    grids: dict[int, pd.DataFrame], origins: dict[int, list[pd.Timestamp]]
) -> np.ndarray:
    """Collect the ground truth for every window, in physical units.

    Args:
        rejillas: Output of grid_by_space().
        origenes: Output of test_windows().

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


def check_against_gru(raw_df: pd.DataFrame, target: np.ndarray) -> None:
    """Abort unless SARIMAX is scored on exactly the GRU's test targets.

    Runs the GRU preprocessing, denormalizes its test targets and compares them
    element by element. Without this check the two models could silently end up
    on different windows, and the whole comparison would be meaningless while
    still printing a tidy table.

    Args:
        raw_df: The original frame, before preprocessing.
        objetivo: The target array this script assembled.

    Raises:
        RuntimeError: If the shapes or the values do not match.
    """
    datasets, normalizer, _, space_index = dat.prepare(raw_df)
    test = datasets["test"]
    inverse = {v: k for k, v in space_index.items()}
    ids = np.array([inverse[int(i)] for i in test.space.numpy()])
    gru_target = normalizer.denormalize(
        test.y.numpy(), cfg.ACTIVE_VARIABLES, ids
    )

    if gru_target.shape != target.shape:
        raise RuntimeError(
            f"window counts differ: GRU {gru_target.shape}, "
            f"SARIMAX {target.shape}; the two models are not being scored on "
            "the same test set"
        )
    if not np.allclose(gru_target, target, rtol=1e-4, atol=1e-4):
        worst = float(np.abs(gru_target - target).max())
        raise RuntimeError(
            f"targets differ by up to {worst:.6f} despite matching shapes; "
            "the window alignment is wrong"
        )
    print(f"  Verificado: mismas {len(target):,} ventanas de prueba que el GRU.")


def fit_all(
    df: pd.DataFrame,
    grids: dict[int, pd.DataFrame],
    origins: dict[int, list[pd.Timestamp]],
    seasonality: str,
    order: tuple[int, int, int],
) -> tuple[np.ndarray, dict]:
    """Fit one model per space and variable and forecast every test window.

    Args:
        df: Frame returned by common.data.preprocesar().
        rejillas: Output of grid_by_space().
        origenes: Output of test_windows().
        estacionalidad: "fourier" or "estacional".
        orden: Non-seasonal (p, d, q).

    Returns:
        A tuple of (predictions shaped like the targets, per-model diagnostics).
    """
    spaces = sorted(origins)
    n_windows = sum(len(origins[e]) for e in spaces)
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
            baseline = mod.BaselineSARIMAX(
                order=order, seasonality=seasonality
            ).fit(training)

            position = {t: i for i, t in enumerate(grid.index)}
            indices = np.array([position[t] for t in origins[space]])
            output = baseline.forecast_from(series, indices, cfg.OUTPUT_STEPS)
            prediction[row : row + len(indices), :, j] = output
            row += len(indices)

            key = f"{variable}@space{space}"
            diagnostics[key] = {
                "order": baseline.order_summary(),
                "aic": baseline.aic,
                "n_train": int(is_train.sum()),
                "seconds": round(time.time() - t0, 2),
            }
            print(
                f"    {key:28} {baseline.order_summary():22} "
                f"AIC {baseline.aic:10.1f}  {diagnostics[key]['seconds']:6.2f} s"
            )
    return prediction, diagnostics


def main() -> int:
    """Run the full baseline from the command line.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Fit the SARIMAX baseline.")
    parser.add_argument(
        "--data", default=None, help="contract frame; defaults to the cleaned sensor data"
    )
    parser.add_argument(
        "--seasonality",
        choices=("fourier", "seasonal"),
        default="fourier",
        help="'fourier' carries the daily cycle as exogenous regressors; "
        "'estacional' uses s=288 as section 4.2.7.1 states, and is far slower",
    )
    parser.add_argument("--order", default="2,0,2", help="non-seasonal p,d,q")
    parser.add_argument("--label", default="sarimax")
    args = parser.parse_args()

    order = tuple(int(x) for x in args.order.split(","))
    if len(order) != 3:
        raise SystemExit(f"--orden needs three values, got {args.order!r}")

    raw_df = dat.load_data(args.data)

    df, _ = dat.preprocess(raw_df)
    grids = grid_by_space(df)
    origins = test_windows(df)
    target = real_targets(grids, origins)

    print(f"\nEstacionalidad: {args.seasonality} | orden no estacional: {order}")
    print(f"Modelos a ajustar: {len(cfg.ACTIVE_VARIABLES)} variables x "
          f"{len(origins)} espacios = "
          f"{len(cfg.ACTIVE_VARIABLES) * len(origins)}")
    check_against_gru(raw_df, target)

    print("\n  Ajustando:")
    t0 = time.time()
    prediction, diagnostics = fit_all(
        df, grids, origins, args.seasonality, order
    )
    duration = time.time() - t0
    print(f"\n  Total: {duration:.1f} s")

    results = met.evaluate(target, prediction)
    met.print_report(results, f"SARIMAX ({args.seasonality}) sobre el conjunto de prueba")

    cfg.MODELS.mkdir(parents=True, exist_ok=True)
    path = cfg.MODELS / f"{args.label}_metadata.json"
    path.write_text(
        json.dumps(
            {
                "seasonality": args.seasonality,
                "order": list(order),
                "n_windows": int(len(target)),
                "seconds": round(duration, 1),
                "test_metrics": results,
                "mae_by_horizon": met.by_horizon(target, prediction),
                "modelos": diagnostics,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nGuardado en {path.relative_to(cfg.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
