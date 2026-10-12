"""Data pipeline: from the data contract to the tensors the GRU consumes.

Responsibilities, in order:

  1. Split each space into contiguous time blocks. Windows never cross a cut:
     joining two stretches separated by days would manufacture examples the
     model would learn as if they were real.
  2. Reindex each block onto the regular 5-minute grid and interpolate short
     gaps. Long gaps already split the block.
  3. Split chronologically into train, validation and test.
  4. Normalize with z-score, fitting mean and sigma on the training split only.
  5. Build the sliding windows of 12 input steps and 6 output steps.

Steps 3 and 4 are where the most common leakage in time-series pipelines
creeps in, so both are deliberately explicit here.

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-10-01 - Split preprocess() out of prepare() so SARIMAX can reuse it.

Reference:
    Section 4.2.6 of the research document.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from common import config as cfg


# --------------------------------------------------------------------------
# Contiguous blocks and regular grid
# --------------------------------------------------------------------------
def mark_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """Label contiguous time blocks within each space.

    A gap longer than MAX_INTERPOLABLE_STEPS ends the current block.

    Args:
        df: Data contract frame, with timestamp and space columns.

    Returns:
        The same rows plus a 'bloque_id' column.
    """
    step = pd.Timedelta(seconds=cfg.INTERVAL_SECONDS)
    tolerance = step * (cfg.MAX_INTERPOLABLE_STEPS + 1)

    parts = []
    for space, group in df.groupby(cfg.SPACE_COLUMN, sort=True):
        group = group.sort_values(cfg.TIME_COLUMN).copy()
        jump = group[cfg.TIME_COLUMN].diff()
        cutoff = (jump > tolerance) | jump.isna()
        numbers = cutoff.cumsum().astype(int)
        group["block_id"] = [f"E{space}-B{n:03d}" for n in numbers]
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


def regularize(df: pd.DataFrame) -> pd.DataFrame:
    """Snap each block onto the 5-minute grid and fill short gaps.

    Interpolation is inside-only, meaning gaps surrounded by valid data.
    Extrapolating at the edges of a block would invent values with nothing to
    support them. Binary variables are forward-filled instead of interpolated,
    so no meaningless intermediate value such as 0.5 is produced for a PIR.

    Args:
        df: Frame already carrying a 'bloque_id' column.

    Returns:
        Frame on a regular grid, with short gaps filled and long ones left as
        missing values.
    """
    step = f"{cfg.INTERVAL_SECONDS}s"
    parts = []
    for _, group in df.groupby("block_id", sort=True):
        group = group.sort_values(cfg.TIME_COLUMN).set_index(cfg.TIME_COLUMN)
        grid = pd.date_range(group.index.min(), group.index.max(), freq=step)
        group = group.reindex(grid)
        group[cfg.SPACE_COLUMN] = group[cfg.SPACE_COLUMN].ffill().bfill()
        group["block_id"] = group["block_id"].ffill().bfill()

        for variable in cfg.ACTIVE_VARIABLES:
            if variable in cfg.BINARY:
                group[variable] = group[variable].ffill(
                    limit=cfg.MAX_INTERPOLABLE_STEPS
                )
            else:
                group[variable] = group[variable].interpolate(
                    method="linear",
                    limit=cfg.MAX_INTERPOLABLE_STEPS,
                    limit_area="inside",
                )
        parts.append(group.rename_axis(cfg.TIME_COLUMN).reset_index())

    output = pd.concat(parts, ignore_index=True)
    output[cfg.SPACE_COLUMN] = output[cfg.SPACE_COLUMN].astype(int)
    return output


def add_time_features(df: pd.DataFrame) -> list[str]:
    """Add time of day and day of week as sine/cosine pairs.

    Encoded cyclically rather than as integers: 23:55 is one step away from
    00:00, but as numbers they sit 23 apart. Sine and cosine together express
    that circularity without a discontinuity.

    Args:
        df: Frame to extend in place.

    Returns:
        Names of the columns that were added, in order.
    """
    t = df[cfg.TIME_COLUMN]
    minutes = t.dt.hour * 60 + t.dt.minute
    day_angle = 2 * np.pi * minutes / (24 * 60)
    week_angle = 2 * np.pi * t.dt.dayofweek / 7

    df["hour_sin"] = np.sin(day_angle)
    df["hour_cos"] = np.cos(day_angle)
    df["week_sin"] = np.sin(week_angle)
    df["week_cos"] = np.cos(week_angle)
    return ["hour_sin", "hour_cos", "week_sin", "week_cos"]


# --------------------------------------------------------------------------
# Chronological split
# --------------------------------------------------------------------------
def split(df: pd.DataFrame) -> pd.DataFrame:
    """Tag every row as train, validation or test by time.

    Cuts run along each space's time axis, never at random. A random split
    would leave step t in training and step t+1 in test, scoring the model
    against data practically identical to what it already saw: the result
    looks excellent and means nothing.

    Args:
        df: Frame on a regular grid.

    Returns:
        The same rows plus a 'particion' column.
    """
    if cfg.PARTITION_MODE == "by_days":
        return split_by_days(df)

    parts = []
    for _, group in df.groupby(cfg.SPACE_COLUMN, sort=True):
        group = group.sort_values(cfg.TIME_COLUMN).copy()
        n = len(group)
        train_end = int(n * cfg.TRAIN_FRACTION)
        val_end = int(n * (cfg.TRAIN_FRACTION + cfg.VALIDATION_FRACTION))
        partition = np.array(["test"] * n, dtype=object)
        partition[:train_end] = "train"
        partition[train_end:val_end] = "validation"
        group["partition"] = partition
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


def choose_days(dates: list) -> dict:
    """Decide which calendar days go to validation and test.

    Test takes the latest weekdays and validation the weekdays just before
    them, as many as the configured fractions ask for and at least one each.
    Weekend days always stay in training.

    That last rule is what fixes short datasets. A plain chronological cut over
    one week puts Saturday and Sunday at the end, so the model is scored and
    early-stopped on an empty building. The prediction that matters is on class
    days, so that is what validation and test must contain.

    Training may then hold days later than the test days. That is the blocked
    form of cross-validation, which stays valid for time series as long as no
    window straddles two blocks; EnvironmentalWindows already rejects any window
    whose steps fall in different partitions, and midnight, the block boundary,
    is when the building is empty.

    Args:
        fechas: Calendar days present in the data.

    Returns:
        A dict from each day to "entrenamiento", "validacion" or "prueba".

    Reference:
        Bergmeir, C., Hyndman, R. J., & Koo, B. (2018). A note on the validity
        of cross-validation for evaluating autoregressive time series
        prediction. Computational Statistics & Data Analysis, 120, 70-83.
    """
    days = sorted(dates)
    n_test = max(1, round(len(days) * (1 - cfg.TRAIN_FRACTION - cfg.VALIDATION_FRACTION)))
    n_val = max(1, round(len(days) * cfg.VALIDATION_FRACTION))

    weekdays = [d for d in reversed(days) if pd.Timestamp(d).dayofweek < 5]
    test = set(weekdays[:n_test])
    validation = set(weekdays[n_test : n_test + n_val])

    return {
        d: "test" if d in test else "validation" if d in validation else "train"
        for d in days
    }


def split_by_days(df: pd.DataFrame) -> pd.DataFrame:
    """Tag every row by the calendar day it belongs to.

    The same days are used for every space, so all spaces are tested on the
    same period and the GRU and SARIMAX see identical splits.

    Args:
        df: Frame on a regular grid.

    Returns:
        The same rows plus a 'particion' column.
    """
    output = df.copy()
    day = output[cfg.TIME_COLUMN].dt.normalize()
    assignment = choose_days(list(day.unique()))
    output["partition"] = day.map(assignment).astype(object)
    return output


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------
class Normalizer:
    """Per-variable z-score fitted on the training split only.

    The variables live on wildly different scales (CO2 in hundreds of ppm,
    movement in 0/1), and without normalization the network would weight the
    large-magnitude ones out of proportion.

    Two explicit decisions:

      - Binary variables are NOT normalized. Applying (x - 0.4) / 0.49 to
        something that only takes 0 or 1 improves nothing and makes the output
        unreadable.
      - `by_space` chooses between global per-variable statistics, as the
        research document specifies, and per-space statistics. With light
        levels differing by two orders of magnitude between spaces, global
        statistics collapse the dark spaces near zero. The default follows the
        document; this is a parameter, not a decision buried in the code.

    Reference:
        Section 4.2.6 of the research document.
    """

    def __init__(self, by_space: bool = False) -> None:
        """Create an unfitted normalizer.

        Args:
            by_space: If True, fit one mean and sigma per space and
                variable instead of one per variable.
        """
        self.by_space = by_space
        self.parameters: dict[str, dict[str, float]] = {}

    @staticmethod
    def _key(variable: str, space: int | None) -> str:
        """Build the lookup key for a variable, optionally scoped to a space.

        Args:
            variable: Variable name.
            espacio: Space id, or None for global statistics.

        Returns:
            The dictionary key under which the statistics are stored.
        """
        return variable if space is None else f"{variable}@{space}"

    def fit(self, training: pd.DataFrame) -> "Normalizer":
        """Compute mean and sigma from the training split.

        Args:
            entrenamiento: Rows belonging to the training partition only.
                Passing the full frame here would leak the future into
                preprocessing and inflate every metric.

        Returns:
            This normalizer, so calls can be chained.
        """
        for variable in cfg.ACTIVE_VARIABLES:
            if variable in cfg.BINARY:
                continue
            if self.by_space:
                for space, group in training.groupby(cfg.SPACE_COLUMN):
                    self._fit_one(variable, group[variable], space)
            else:
                self._fit_one(variable, training[variable], None)
        return self

    def _fit_one(
        self, variable: str, series: pd.Series, space: int | None
    ) -> None:
        """Store mean and sigma for one variable, optionally per space.

        Args:
            variable: Variable name.
            serie: Training values for that variable.
            espacio: Space id, or None for global statistics.
        """
        values = series.dropna()
        sigma = float(values.std())
        # A variable that is constant in training would give sigma 0 and a
        # division by zero. Falling back to 1 turns normalization into a plain
        # shift, which is the harmless behaviour.
        self.parameters[self._key(variable, space)] = {
            "mean": float(values.mean()),
            "sigma": sigma if sigma > 1e-8 else 1.0,
        }

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted z-score to a frame.

        Args:
            df: Frame holding the active variables.

        Returns:
            A copy with the continuous variables normalized.
        """
        output = df.copy()
        for variable in cfg.ACTIVE_VARIABLES:
            if variable in cfg.BINARY:
                continue
            if self.by_space:
                for space in output[cfg.SPACE_COLUMN].unique():
                    p = self.parameters[self._key(variable, int(space))]
                    rows = output[cfg.SPACE_COLUMN] == space
                    output.loc[rows, variable] = (
                        output.loc[rows, variable] - p["mean"]
                    ) / p["sigma"]
            else:
                p = self.parameters[self._key(variable, None)]
                output[variable] = (output[variable] - p["mean"]) / p["sigma"]
        return output

    def denormalize(
        self,
        values: np.ndarray,
        variables: list[str],
        spaces: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return values to their original physical units.

        Args:
            valores: Array shaped (n, pasos, n_variables) in normalized space.
            variables: Variable names in the order of the last axis.
            espacios: Space ids shaped (n,). Required when by_space is True.

        Returns:
            An array of the same shape, in the original units.

        Raises:
            ValueError: If by_space is True and espacios is None.
        """
        output = np.asarray(values, dtype=float).copy()
        for i, variable in enumerate(variables):
            if variable in cfg.BINARY:
                continue
            if self.by_space:
                if spaces is None:
                    raise ValueError("por_espacio=True requires the space ids")
                for space in np.unique(spaces):
                    p = self.parameters[self._key(variable, int(space))]
                    rows = spaces == space
                    output[rows, :, i] = output[rows, :, i] * p["sigma"] + p["mean"]
            else:
                p = self.parameters[self._key(variable, None)]
                output[:, :, i] = output[:, :, i] * p["sigma"] + p["mean"]
        return output

    def save(self, path: Path) -> None:
        """Write the fitted statistics to disk as JSON.

        The model is useless without this file: without the mean and sigma it
        was trained with, its outputs cannot be turned back into ppm or
        degrees.

        Args:
            ruta: Destination path.
        """
        path.write_text(
            json.dumps(
                {"by_space": self.by_space, "parameters": self.parameters},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "Normalizer":
        """Rebuild a normalizer from a saved JSON file.

        Args:
            ruta: Path written by save().

        Returns:
            A fitted Normalizer.
        """
        data = json.loads(path.read_text(encoding="utf-8"))
        norm = cls(by_space=data["by_space"])
        norm.parameters = data["parameters"]
        return norm


# --------------------------------------------------------------------------
# Windows
# --------------------------------------------------------------------------
class EnvironmentalWindows(Dataset):
    """Sliding windows of 12 input steps and 6 output steps.

    A window is valid only if all three conditions hold:
      - its 18 steps fall inside the same contiguous block;
      - none of its steps has a missing value in the active variables;
      - all 18 steps belong to the same partition.

    The third condition is what stops a test window from being fed by training
    steps.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        partition: str,
        feature_columns: list[str],
        space_index: dict[int, int],
    ) -> None:
        """Materialize every valid window of one partition as tensors.

        Args:
            df: Normalized frame carrying 'bloque_id' and 'particion'.
            particion: Which partition to build, e.g. "entrenamiento".
            feature_columns: Input column names, in order.
            space_index: Mapping from space id to embedding index.
        """
        self.feature_columns = feature_columns
        self.variables = cfg.ACTIVE_VARIABLES
        self.space_index = space_index

        inputs, outputs, spaces = [], [], []
        length = cfg.INPUT_STEPS + cfg.OUTPUT_STEPS

        for _, block in df.groupby("block_id", sort=True):
            block = block.sort_values(cfg.TIME_COLUMN)
            if len(block) < length:
                continue

            features = block[feature_columns].to_numpy(dtype=np.float32)
            target = block[self.variables].to_numpy(dtype=np.float32)
            complete = ~np.isnan(features).any(axis=1) & ~np.isnan(target).any(axis=1)
            in_partition = (block["partition"] == partition).to_numpy()
            space = int(block[cfg.SPACE_COLUMN].iloc[0])

            valid = complete & in_partition
            for start in range(len(block) - length + 1):
                end = start + length
                if not valid[start:end].all():
                    continue
                cutoff = start + cfg.INPUT_STEPS
                inputs.append(features[start:cutoff])
                outputs.append(target[cutoff:end])
                spaces.append(space_index[space])

        if inputs:
            self.X = torch.from_numpy(np.stack(inputs))
            self.y = torch.from_numpy(np.stack(outputs))
            self.space = torch.tensor(spaces, dtype=torch.long)
        else:
            n_features, n_vars = len(feature_columns), len(self.variables)
            self.X = torch.empty(0, cfg.INPUT_STEPS, n_features)
            self.y = torch.empty(0, cfg.OUTPUT_STEPS, n_vars)
            self.space = torch.empty(0, dtype=torch.long)

    def __len__(self) -> int:
        """Count the windows this partition produced.

        Returns:
            Number of valid windows available for training or evaluation.
        """
        return len(self.X)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return one training example.

        Args:
            i: Window index.

        Returns:
            A tuple of (features, space index, target).
        """
        return self.X[i], self.space[i], self.y[i]


# --------------------------------------------------------------------------
def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Run every step that precedes normalization.

    Split into blocks, snap to the regular grid, assign partitions and add the
    cyclical time features. Deliberately stops short of normalizing, because
    SARIMAX works in physical units while the GRU works in z-scores: sharing
    this function is what guarantees both models see the same rows in the same
    partitions, which is the precondition for the comparison to mean anything.

    Args:
        df: Frame satisfying the data contract described in config.py.

    Returns:
        A tuple of (preprocessed frame, feature column names in order).

    Raises:
        ValueError: If the frame does not satisfy the data contract.
    """
    missing = {cfg.TIME_COLUMN, cfg.SPACE_COLUMN, *cfg.ACTIVE_VARIABLES} - set(
        df.columns
    )
    if missing:
        raise ValueError(f"frame violates the data contract, missing: {sorted(missing)}")

    df = mark_blocks(df)
    df = regularize(df)
    df = mark_blocks(df)  # the grid may have renumbered things; relabel
    df = split(df)

    time_columns = add_time_features(df) if cfg.USE_TIME_FEATURES else []
    return df, list(cfg.ACTIVE_VARIABLES) + time_columns


def prepare(
    df: pd.DataFrame, by_space: bool = False
) -> tuple[dict[str, EnvironmentalWindows], Normalizer, list[str], dict[int, int]]:
    """Run the whole pipeline, from the data contract to the three splits.

    Args:
        df: Frame satisfying the data contract described in config.py.
        by_space: Whether to normalize per space instead of globally.

    Returns:
        A tuple of (splits, normalizer, feature column names, space index map).
        `splits` is keyed by "entrenamiento", "validacion" and "prueba".

    Raises:
        ValueError: If the frame does not satisfy the data contract.
    """
    df, feature_columns = preprocess(df)

    normalizer = Normalizer(by_space=by_space)
    normalizer.fit(df[df["partition"] == "train"])
    df = normalizer.transform(df)

    spaces = sorted(df[cfg.SPACE_COLUMN].unique())
    space_index = {int(e): i for i, e in enumerate(spaces)}

    datasets = {
        name: EnvironmentalWindows(df, name, feature_columns, space_index)
        for name in ("train", "validation", "test")
    }
    return datasets, normalizer, feature_columns, space_index


def loaders(
    datasets: dict[str, EnvironmentalWindows], batch: int = cfg.BATCH_SIZE
) -> dict[str, DataLoader]:
    """Wrap the splits in DataLoaders.

    Only the training split is shuffled. Shuffling window order there is both
    correct and desirable: each window is already a complete, self-contained
    example, and the order they are shown in should not influence the weights.
    Validation and test keep chronological order so predictions can be plotted
    against time.

    Args:
        conjuntos: Splits returned by prepare().
        lote: Batch size.

    Returns:
        One DataLoader per split, under the same keys.
    """
    return {
        name: DataLoader(
            dataset,
            batch_size=batch,
            shuffle=(name == "train"),
            drop_last=False,
        )
        for name, dataset in datasets.items()
    }


# --------------------------------------------------------------------------
def load_data(path: Path | None = None) -> pd.DataFrame:
    """Load a frame that satisfies the data contract.

    Args:
        ruta: A parquet or CSV file. Defaults to the cleaned sensor data
            written by ingestion.cleaning.

    Returns:
        The frame, with the timestamp column parsed.

    Raises:
        FileNotFoundError: If the file does not exist, naming the command that
            produces it.
    """
    path = Path(path) if path else cfg.REAL_DATA
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Clean the raw exports first with "
            "`python -m ingesta.limpieza ../data/raw/<archivo>.csv`"
        )
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, parse_dates=[cfg.TIME_COLUMN])
