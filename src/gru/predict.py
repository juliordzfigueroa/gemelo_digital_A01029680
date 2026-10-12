"""Direct and iterative forecasting with an already trained GRU.

train.py answers "how good is this model" against a held-out split. This
module answers the different question the conversational layer asks: given the
last hour of readings for one space, what comes next.

Two modes, picked from the requested horizon:

    direct      One forward pass. The head emits all 6 steps (30 min) at once,
                which is exactly what the network was trained to produce.
    iterative   The predicted block is appended to the context window and the
                pass repeats, up to 2 hours. Error compounds at every
                iteration, so the result carries an explicit warning.

Three things this module does that are easy to get wrong:

  - The space id is not the embedding index. `space_id` 3302 is a room
    number; the network expects the row of the embedding table that space was
    assigned during training, which lives in the run metadata.
  - Time features are recomputed for the real future timestamps on every
    iteration. Carrying over the window's own time features would shift the
    model's idea of the time of day by 30 minutes per iteration, and the
    network leans on that signal.
  - Fed-back values are clipped to the instrument range before they re-enter
    the window. Without it an iteration can hand the next one a negative CO2
    and the error grows from something physically impossible.

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-10-12 - Initial implementation.

Reference:
    Sections 4.2.5 and 4.2.6 of the research document.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import config as cfg
from common.data import Normalizer, add_time_features
from gru import model as mod

# Longest horizon the iterative mode will produce. Beyond two hours the
# compounded error makes the output not worth reporting.
#
# Reference: Section 4.2.6 of the research document.
MAX_HORIZON_MIN = 120

# Minutes per step, derived from the sampling interval rather than hard-coded
# so that changing the interval in config.py cannot desynchronize this module.
MINUTES_PER_STEP = cfg.INTERVAL_SECONDS // 60


def _implementation_from_weights(state: dict) -> str:
    """Infer which recurrent implementation a checkpoint was trained with.

    The two implementations do not have the same parameters: the hand-written
    cell stores three weight matrices per layer, while nn.GRU stores fused
    input and hidden matrices plus two bias vectors per gate. Loading a
    checkpoint into the wrong one fails with an unreadable state_dict error,
    and older runs saved no `implementacion` field at all, so the weights
    themselves are the only reliable source.

    Args:
        estado: State dict loaded from a .pt file.

    Returns:
        Either "cho" or "pytorch".

    Raises:
        RuntimeError: If the checkpoint matches neither layout.
    """
    if any(key.startswith("gru1.cell.") for key in state):
        return "cho"
    if "gru1.weight_ih_l0" in state:
        return "pytorch"
    raise RuntimeError(
        "the checkpoint matches neither recurrent layout; its first keys are "
        f"{list(state)[:4]}"
    )


def load_model(
    weights_path: Path | None = None,
    normalizer_path: Path | None = None,
    metadata_path: Path | None = None,
    device: str | None = None,
) -> tuple[mod.EnvironmentalGRU, Normalizer, dict]:
    """Load weights, normalizer and run metadata from disk.

    The three files are a unit. Without the normalizer the network's outputs
    cannot be turned back into ppm or degrees, and without the metadata there
    is no way to know which embedding row belongs to which space.

    Args:
        weights_path: Path to the .pt file. Defaults to models/gru.pt.
        normalizer_path: Path to the normalizer JSON.
        metadata_path: Path to the metadata JSON.
        device: "cuda", "cpu" or None to autodetect.

    Returns:
        A tuple of (network in eval mode, fitted normalizer, metadata dict).

    Raises:
        FileNotFoundError: If any of the three files is missing.
    """
    weights_path = weights_path or cfg.MODELS / "gru.pt"
    normalizer_path = normalizer_path or cfg.MODELS / "gru_normalizer.json"
    metadata_path = metadata_path or cfg.MODELS / "gru_metadata.json"

    for path in (weights_path, normalizer_path, metadata_path):
        if not Path(path).exists():
            raise FileNotFoundError(
                f"{path} does not exist; train the model first with "
                "`python -m gru.entrenar`"
            )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    state = torch.load(weights_path, map_location=device)

    # Derived from the weights even when the metadata states it: the weights
    # are what has to load, and a metadata field can be stale or absent.
    implementation = _implementation_from_weights(state)

    net = mod.build(
        n_spaces=len(metadata["space_index"]),
        n_continuous_features=len(metadata["feature_columns"]),
        implementation=implementation,
    )
    net.load_state_dict(state)
    net.to(device)
    net.eval()

    normalizer = Normalizer.load(Path(normalizer_path))
    return net, normalizer, metadata


def space_index_of(metadata: dict, space_id: int) -> int:
    """Translate a space id into its row of the embedding table.

    Args:
        metadatos: Metadata dict returned by load_model().
        space_id: Space identifier as it appears in the data.

    Returns:
        The embedding index the network expects.

    Raises:
        ValueError: If that space was not part of the training run, in which
            case the network has no learned representation for it.
    """
    indices = metadata["space_index"]
    key = str(space_id)
    if key not in indices:
        raise ValueError(
            f"space {space_id} was not in the training run; the model only "
            f"knows {sorted(int(k) for k in indices)}"
        )
    return int(indices[key])


def _check_layout(metadata: dict) -> list[str]:
    """Confirm the feature columns are the variables followed by time features.

    The iterative mode rebuilds input rows by concatenating predicted
    variables and freshly computed time features, which is only correct if the
    columns sit in that order. Checking it here turns a silent corruption into
    an error at load time.

    Args:
        metadatos: Metadata dict returned by load_model().

    Returns:
        The names of the trailing time-feature columns, possibly empty.

    Raises:
        RuntimeError: If the variables are not the leading columns.
    """
    columns = metadata["feature_columns"]
    variables = metadata["active_variables"]
    if columns[: len(variables)] != variables:
        raise RuntimeError(
            "the feature layout is not 'variables first, time features after', "
            f"so iterative prediction cannot rebuild a row: {columns}"
        )
    return columns[len(variables) :]


def _time_features(stamps: pd.Series, names: list[str]) -> np.ndarray:
    """Build the cyclical time features for a series of timestamps.

    Delegates to the same function the training pipeline uses, so inference can
    never drift from training on how the time of day is encoded.

    Args:
        marcas: Timestamps to encode.
        nombres: Expected feature names, in order.

    Returns:
        An array shaped (len(marcas), len(nombres)).
    """
    if not names:
        return np.empty((len(stamps), 0), dtype=np.float32)
    frame = pd.DataFrame({cfg.TIME_COLUMN: stamps.reset_index(drop=True)})
    add_time_features(frame)
    return frame[names].to_numpy(dtype=np.float32)


def _to_physical(
    normalizer: Normalizer,
    block: np.ndarray,
    variables: list[str],
    space_id: int,
) -> np.ndarray:
    """Convert a normalized block to physical units and clip it.

    Args:
        normalizador: Fitted normalizer.
        bloque: Normalized predictions shaped (pasos, n_variables).
        variables: Variable names matching the last axis.
        space_id: Space the block belongs to.

    Returns:
        An array of the same shape in physical units, within the instrument
        range declared in config.PHYSICAL_LIMITS.
    """
    spaces = np.array([space_id]) if normalizer.by_space else None
    physical = normalizer.denormalize(block[None, :, :], variables, spaces)[0]

    for i, variable in enumerate(variables):
        lower, upper = cfg.PHYSICAL_LIMITS.get(variable, (None, None))
        if lower is not None:
            physical[:, i] = np.maximum(physical[:, i], lower)
        if upper is not None:
            physical[:, i] = np.minimum(physical[:, i], upper)
    return physical


def _to_normalized(
    normalizer: Normalizer,
    block: np.ndarray,
    variables: list[str],
    space_id: int,
) -> np.ndarray:
    """Convert a physical block back to normalized space.

    Used only by the iterative mode, to feed clipped values back into the
    context window. It goes through the normalizer's public transform instead
    of repeating the affine arithmetic, so the two can never disagree.

    Args:
        normalizador: Fitted normalizer.
        bloque: Values in physical units shaped (pasos, n_variables).
        variables: Variable names matching the last axis.
        space_id: Space the block belongs to.

    Returns:
        An array of the same shape in normalized space.
    """
    frame = pd.DataFrame(block, columns=variables)
    frame[cfg.SPACE_COLUMN] = space_id
    return normalizer.transform(frame)[variables].to_numpy(dtype=np.float32)


def _normalized_window(
    window: pd.DataFrame,
    normalizer: Normalizer,
    metadata: dict,
    time_names: list[str],
) -> np.ndarray:
    """Turn a window of raw readings into the network's input array.

    Args:
        ventana: Exactly INPUT_STEPS rows satisfying the data contract,
            sorted by time, in physical units.
        normalizador: Fitted normalizer.
        metadatos: Metadata dict returned by load_model().
        time_names: Trailing time-feature column names.

    Returns:
        An array shaped (INPUT_STEPS, n_features) in normalized space.

    Raises:
        ValueError: If the window has the wrong length, is missing a contract
            column, or holds a missing value.
    """
    input_steps = metadata["input_steps"]
    variables = metadata["active_variables"]

    if len(window) != input_steps:
        raise ValueError(
            f"the window needs exactly {input_steps} rows, got {len(window)}"
        )

    missing = {cfg.TIME_COLUMN, *variables} - set(window.columns)
    if missing:
        raise ValueError(f"the window is missing columns: {sorted(missing)}")

    if window[variables].isna().to_numpy().any():
        raise ValueError(
            "the window holds missing values; the network cannot consume a "
            "window with gaps, so pick a contiguous one"
        )

    frame = window.sort_values(cfg.TIME_COLUMN).reset_index(drop=True).copy()
    if cfg.SPACE_COLUMN not in frame.columns:
        frame[cfg.SPACE_COLUMN] = 0

    normalized = normalizer.transform(frame)
    features = normalized[variables].to_numpy(dtype=np.float32)

    if time_names:
        ts = _time_features(frame[cfg.TIME_COLUMN], time_names)
        features = np.concatenate([features, ts], axis=1)
    return features


def predict(
    net: mod.EnvironmentalGRU,
    normalizer: Normalizer,
    metadata: dict,
    window: pd.DataFrame,
    space_id: int,
    horizon_min: int = 30,
    device: str | None = None,
) -> dict:
    """Forecast one space forward, choosing direct or iterative mode.

    Args:
        red: Network returned by load_model().
        normalizador: Normalizer returned by load_model().
        metadatos: Metadata dict returned by load_model().
        ventana: The last INPUT_STEPS readings of that space, in physical
            units, with the timestamp column. The timestamps are what let the
            iterative mode know the real time of day of each predicted step.
        space_id: Space identifier as it appears in the data.
        horizon_min: Minutes ahead, a multiple of 5 up to MAX_HORIZON_MIN.
        device: "cuda", "cpu" or None to autodetect.

    Returns:
        A dict with:
          predicciones: one dict per 5-minute step, holding offset_min and one
            entry per variable in physical units.
          horizon_min: the horizon actually served.
          metodo: "directo" or "iterativo".
          advertencia: a sentence about compounded error, or None.

    Raises:
        ValueError: If the horizon is out of range or not a multiple of the
            sampling interval, or if the window is malformed.
    """
    if horizon_min <= 0 or horizon_min > MAX_HORIZON_MIN:
        raise ValueError(
            f"horizonte_min must be between {MINUTES_PER_STEP} and "
            f"{MAX_HORIZON_MIN}, got {horizon_min}"
        )
    if horizon_min % MINUTES_PER_STEP != 0:
        raise ValueError(
            f"horizonte_min must be a multiple of {MINUTES_PER_STEP}, "
            f"got {horizon_min}"
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    variables = metadata["active_variables"]
    output_steps = metadata["output_steps"]
    time_names = _check_layout(metadata)
    index = space_index_of(metadata, space_id)

    features = _normalized_window(window, normalizer, metadata, time_names)
    last_stamp = pd.to_datetime(window[cfg.TIME_COLUMN]).max()

    requested_steps = horizon_min // MINUTES_PER_STEP
    minutes_per_block = output_steps * MINUTES_PER_STEP
    n_iterations = math.ceil(horizon_min / minutes_per_block)

    # Every timestamp the run will predict, computed once so each iteration
    # encodes the real clock time of its own steps.
    step = pd.Timedelta(seconds=cfg.INTERVAL_SECONDS)
    stamps = pd.Series(
        [last_stamp + step * (i + 1) for i in range(n_iterations * output_steps)]
    )
    future_time = _time_features(stamps, time_names)

    space_t = torch.tensor([index], dtype=torch.long, device=device)
    accumulated: list[np.ndarray] = []

    for iteration in range(n_iterations):
        input_tensor = torch.from_numpy(features[None, :, :]).to(device)
        with torch.inference_mode():
            output = net(input_tensor, space_t)

        norm_block = output[0].cpu().numpy()
        physical_block = _to_physical(normalizer, norm_block, variables, space_id)
        accumulated.append(physical_block)

        if iteration + 1 == n_iterations:
            break

        # Slide the window: drop the oldest block, append what was just
        # predicted. The clipped physical values go back, not the raw network
        # output, so an impossible value never becomes the next input.
        feedback = _to_normalized(
            normalizer, physical_block, variables, space_id
        )
        since = iteration * output_steps
        new_rows = np.concatenate(
            [feedback, future_time[since : since + output_steps]], axis=1
        )
        features = np.vstack([features[output_steps:], new_rows]).astype(np.float32)

    prediction = np.vstack(accumulated)[:requested_steps]

    steps = [
        {
            "offset_min": (i + 1) * MINUTES_PER_STEP,
            **{
                variable: round(float(prediction[i, j]), 2)
                for j, variable in enumerate(variables)
            },
        }
        for i in range(len(prediction))
    ]

    iterative = n_iterations > 1
    warning = None
    if iterative:
        warning = (
            f"Predicción iterativa en {n_iterations} pasos de "
            f"{minutes_per_block} min: el error se acumula en cada iteración, "
            "porque las entradas después del primer bloque son salidas del "
            "propio modelo y no lecturas reales."
        )

    return {
        "predictions": steps,
        "horizon_min": horizon_min,
        "method": "iterativo" if iterative else "directo",
        "warning": warning,
    }


def main() -> int:
    """Forecast one space from synthetic data and print the result.

    Exists so the module can be exercised without the API on top, which is how
    a broken checkpoint or a stale metadata file gets caught early.

    Returns:
        Process exit code.
    """
    import argparse

    from common import synthetic

    parser = argparse.ArgumentParser(description="Forecast one space.")
    parser.add_argument("--space", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--days", type=int, default=3)
    args = parser.parse_args()

    net, normalizer, metadata = load_model()
    print(
        f"Modelo cargado: implementacion {net.implementation}, "
        f"{net.n_parameters():,} parametros, "
        f"espacios {sorted(int(k) for k in metadata['space_index'])}"
    )

    df = synthetic.generate(days=args.days, n_spaces=5)
    own = df[df[cfg.SPACE_COLUMN] == args.space].sort_values(
        cfg.TIME_COLUMN
    )
    window = own.tail(metadata["input_steps"])

    output = predict(
        net,
        normalizer,
        metadata,
        window,
        space_id=args.space,
        horizon_min=args.horizon,
    )

    print(
        f"\nEspacio {args.space}, horizonte {output['horizon_min']} min, "
        f"metodo {output['method']}"
    )
    last = window.iloc[-1]
    print(
        "  ahora      "
        + "  ".join(
            f"{v}={float(last[v]):.1f}" for v in metadata["active_variables"]
        )
    )
    for p in output["predictions"]:
        detail = "  ".join(
            f"{v}={p[v]:.1f}" for v in metadata["active_variables"]
        )
        print(f"  +{p['offset_min']:>3} min  {detail}")

    if output["warning"]:
        print(f"\n  {output['warning']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
