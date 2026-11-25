"""Training loop for the GRU: Adam at lr=1e-3, MSE loss and early stopping.

Saves the best state by validation loss into models/, together with the
normalization statistics and the metadata needed to reproduce inference. The
model is only useful alongside its normalizer: without the mean and sigma it
was trained with, its outputs cannot be turned back into physical units.

Usage, from the src/ directory:
    python -m gru.train
    python -m gru.train --data ruta.parquet --label otro_nombre

The check that the network can learn a known series lives in gru/check.py.

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-11-25 - Single recurrent implementation; check moved to check.py.

Reference:
    Sections 4.2.6 and 5.5 of the research document.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from common import config as cfg
from common import data as dat
from common import metrics as met
from gru import model as mod


def device() -> torch.device:
    """Pick the compute device.

    Returns:
        The CUDA device when available, otherwise CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = cfg.SEED) -> None:
    """Seed every random source so a run can be reproduced.

    Args:
        seed: Seed shared by torch, numpy and CUDA.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    net: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    dev: torch.device,
) -> float:
    """Run one full pass over a data loader.

    Args:
        net: The network.
        loader: Batches to iterate over.
        loss_fn: Loss function.
        optimizer: Optimizer to step, or None to evaluate without training.
        dev: Device to run on.

    Returns:
        Mean loss per sample over the pass.
    """
    is_training = optimizer is not None
    net.train(is_training)

    total, n = 0.0, 0
    with torch.set_grad_enabled(is_training):
        for features, space, target in loader:
            features = features.to(dev)
            space = space.to(dev)
            target = target.to(dev)

            prediction = net(features, space)
            loss = loss_fn(prediction, target)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping: recurrent networks can produce very large
                # gradients on sequences with abrupt jumps, and a single runaway
                # step ruins weights that were already learned.
                nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()

            total += loss.item() * len(features)
            n += len(features)
    return total / max(n, 1)


def predict_windows(
    net: nn.Module, loader: DataLoader, dev: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Run the network over every window of a loader.

    Not to be confused with gru.predict, which forecasts one space forward
    from its latest readings. This one scores a whole split at once.

    Args:
        net: Trained network.
        loader: Batches to predict.
        dev: Device to run on.

    Returns:
        A tuple of (ground truth, predictions), both still in normalized space.

    Raises:
        RuntimeError: If the loader yielded no windows at all.
    """
    net.eval()
    actual_values, predicted = [], []
    with torch.no_grad():
        for features, space, target in loader:
            prediction = net(features.to(dev), space.to(dev))
            actual_values.append(target.numpy())
            predicted.append(prediction.cpu().numpy())
    if not actual_values:
        raise RuntimeError("the loader produced no windows")
    return np.concatenate(actual_values), np.concatenate(predicted)


def train(
    df: pd.DataFrame,
    epochs: int = cfg.MAX_EPOCHS,
    patience: int = cfg.PATIENCE,
    quiet: bool = False,
) -> dict:
    """Train the network and evaluate it on the test split.

    Args:
        df: Frame satisfying the data contract.
        epochs: Maximum number of epochs.
        patience: Epochs without validation improvement before stopping.
        quiet: Suppress progress output.

    Returns:
        A dict holding the trained network, its normalizer, the training
        history and the test metrics.

    Raises:
        RuntimeError: If any split ended up with no windows.
    """
    set_seed()
    dev = device()

    datasets, normalizer, columns, space_index = dat.prepare(df)
    loaders = dat.loaders(datasets)

    if not quiet:
        print(f"Dispositivo: {dev}")
        if dev.type == "cuda":
            print(f"  {torch.cuda.get_device_name(0)}")
        print(f"Rasgos de entrada ({len(columns)}): {columns}")
        print(
            f"Variables a predecir ({len(cfg.ACTIVE_VARIABLES)}): "
            f"{cfg.ACTIVE_VARIABLES}"
        )
        print("Ventanas por particion:")
        for name, dataset in datasets.items():
            print(f"  {name:14} {len(dataset):7,}")

    empty_splits = [name for name, dataset in datasets.items() if len(dataset) == 0]
    if empty_splits:
        raise RuntimeError(
            f"splits {empty_splits} produced no windows; "
            "more data or longer contiguous blocks are needed"
        )

    net = mod.build(n_spaces=len(space_index), n_continuous_features=len(columns)).to(dev)
    if not quiet:
        print(f"\nParametros entrenables: {net.n_parameters():,}")

    # MSE because this is regression over continuous values with no activation
    # on the output layer: the standard pairing from Table 1, section 5.3.
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.LEARNING_RATE)

    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    no_improvement = 0
    history = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        loss_train = run_epoch(net, loaders["train"], loss_fn, optimizer, dev)
        loss_val = run_epoch(net, loaders["validation"], loss_fn, None, dev)
        history.append({"epoch": epoch, "train": loss_train, "validation": loss_val})

        if loss_val < best_loss - cfg.MIN_IMPROVEMENT:
            best_loss = loss_val
            best_epoch = epoch
            # Copy to CPU: a state dict left on the GPU carries the device with
            # it and fails to load on a machine without CUDA.
            best_state = {
                k: v.detach().cpu().clone() for k, v in net.state_dict().items()
            }
            no_improvement = 0
            stamp = " *"
        else:
            no_improvement += 1
            stamp = ""

        if not quiet and (epoch <= 5 or epoch % 10 == 0 or stamp):
            print(
                f"  epoca {epoch:3d}  tren {loss_train:.6f}  val {loss_val:.6f}"
                f"  (sin mejora: {no_improvement}){stamp}"
            )

        if no_improvement >= patience:
            if not quiet:
                print(
                    f"\nEarly stopping en la epoca {epoch}: "
                    f"{patience} epocas sin mejorar validacion."
                )
            break

    duration = time.time() - t0
    if best_state is not None:
        net.load_state_dict(best_state)
    if not quiet:
        print(
            f"Mejor epoca: {best_epoch} (val {best_loss:.6f}) en {duration:.1f} s"
        )

    # Evaluate in physical units, never in z-score space.
    real_n, pred_n = predict_windows(net, loaders["test"], dev)
    real = normalizer.denormalize(real_n, cfg.ACTIVE_VARIABLES)
    pred = normalizer.denormalize(pred_n, cfg.ACTIVE_VARIABLES)

    return {
        "net": net,
        "normalizer": normalizer,
        "columns": columns,
        "space_index": space_index,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "seconds": duration,
        "results": met.evaluate(real, pred),
        "horizon": met.by_horizon(real, pred),
    }


def save(output: dict, folder: Path = cfg.MODELS, label: str = "gru") -> None:
    """Persist the weights, the normalizer and the run metadata.

    Args:
        output: Dict returned by train().
        folder: Destination directory.
        label: Filename prefix for the three files written.
    """
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(output["net"].state_dict(), folder / f"{label}.pt")
    output["normalizer"].save(folder / f"{label}_normalizer.json")

    metadata = {
        "active_variables": cfg.ACTIVE_VARIABLES,
        "feature_columns": output["columns"],
        "space_index": {str(k): v for k, v in output["space_index"].items()},
        "input_steps": cfg.INPUT_STEPS,
        "output_steps": cfg.OUTPUT_STEPS,
        "hidden_units": cfg.HIDDEN_UNITS,
        "dropout": cfg.DROPOUT,
        "training_seconds": round(output["seconds"], 1),
        "best_epoch": output["best_epoch"],
        "best_val_loss": output["best_val_loss"],
        "test_metrics": output["results"],
        "mae_by_horizon": output["horizon"],
    }
    (folder / f"{label}_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"\nGuardado en {folder.relative_to(cfg.ROOT)}/: {label}.pt, "
        f"{label}_normalizer.json, {label}_metadata.json"
    )


def main() -> int:
    """Parse arguments, train, report and save.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Train the environmental GRU.")
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="parquet or csv satisfying the data contract; defaults to the "
        "cleaned sensor data",
    )
    parser.add_argument("--epochs", type=int, default=cfg.MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=cfg.PATIENCE)
    parser.add_argument("--label", default="gru")
    args = parser.parse_args()

    df = dat.load_data(args.data)
    print(f"Datos: {len(df):,} filas, espacios {sorted(df[cfg.SPACE_COLUMN].unique())}")

    output = train(df, epochs=args.epochs, patience=args.patience)
    met.print_report(output["results"], "Conjunto de prueba")

    print("\n== MAE por horizonte de prediccion ==")
    print(f"  {'horizon':10} " + " ".join(f"{v:>12}" for v in cfg.ACTIVE_VARIABLES))
    for horizon, values in output["horizon"].items():
        print(f"  {horizon:10} " + " ".join(f"{values[v]:12.4f}" for v in cfg.ACTIVE_VARIABLES))

    save(output, label=args.label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
