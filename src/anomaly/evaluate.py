"""Measure the 3-sigma detector by injecting anomalies into the test split.

Section 4.2.5 asks for precision and recall, measured by injecting synthetic
anomalies into the test data. The test day has no labelled anomalies of its
own, so the labels come from the injection: a reading is an anomaly exactly
when this script altered it.

Three kinds, each taken from something the real data has shown:

  spike     a sudden jump of a size that happens in a building: a closed
            room filling with people, an air conditioner failing.
  damaged   the values a failing sensor sends: -1 from the gas sensor while
            it fails (13,598 times on the 3304), 0 from an empty read of the
            temperature and humidity sensor, or a value above what the
            instrument can measure.
  stuck     the sensor freezes and repeats its last value for 1 to 3 hours.

Each kind is injected into a fresh copy of the test split, one variable at a
time, so the results do not mix. Precision counts every flag on a reading
that was not injected as a false alarm, real oddities of the test day
included, so it is a lower bound.

Usage, from the src/ directory:
    python -m anomaly.evaluate

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-11-26 - Initial implementation.

Reference:
    Section 4.2.5 of the research document.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from anomaly import detector as det
from common import config as cfg
from common import data as dat

# Injected spikes per variable and kind, and stuck runs per variable.
N_INJECTED = 100
N_STUCK_RUNS = 20

# Size of a spike, in physical units: (smallest, largest) added to the reading.
SPIKE_SIZE = {
    "co2": (1000.0, 3000.0),
    "temperature": (3.0, 8.0),
    "humidity": (10.0, 30.0),
    "light": (40.0, 90.0),
}

# What a failing sensor sends: (its low failure value, a value above the
# instrument's limit). The low ones are the sentinel and the empty read that
# ingestion/cleaning.py finds in the real exports.
DAMAGED_VALUES = {
    "co2": (-1.0, 6000.0),
    "temperature": (0.0, 70.0),
    "humidity": (0.0, 120.0),
    "light": (-1.0, 150.0),
}

# How long a stuck sensor repeats its value, in 5-minute steps (1 to 3 hours).
STUCK_STEPS = (12, 36)


def test_split() -> tuple[pd.DataFrame, dict]:
    """Load the test readings and the statistics fitted on training.

    Returns:
        A tuple of (test rows with every checked variable present, sorted by
        space and time; mean and sigma per variable from the training split).
    """
    df, _ = dat.preprocess(dat.load_data())
    parameters = dat.Normalizer().fit(df[df["partition"] == "train"]).parameters

    test = df[df["partition"] == "test"].dropna(subset=list(parameters))
    test = test.sort_values([cfg.SPACE_COLUMN, cfg.TIME_COLUMN]).reset_index(drop=True)
    return test, parameters


def inject_spikes(test: pd.DataFrame, variable: str, rng) -> tuple[pd.DataFrame, np.ndarray]:
    """Add a sudden jump to randomly chosen readings of one variable."""
    altered = test.copy()
    rows = rng.choice(len(test), size=N_INJECTED, replace=False)
    low, high = SPIKE_SIZE[variable]
    upper_limit = cfg.PHYSICAL_LIMITS[variable][1]
    for row in rows:
        jumped = altered.at[row, variable] + rng.uniform(low, high)
        # A sensor cannot report past its own range, so the jump stops there.
        altered.at[row, variable] = min(jumped, upper_limit)
    return altered, _labels(len(test), rows)


def inject_damaged(test: pd.DataFrame, variable: str, rng) -> tuple[pd.DataFrame, np.ndarray]:
    """Replace randomly chosen readings with what a failing sensor sends."""
    altered = test.copy()
    rows = rng.choice(len(test), size=N_INJECTED, replace=False)
    for row in rows:
        altered.at[row, variable] = rng.choice(DAMAGED_VALUES[variable])
    return altered, _labels(len(test), rows)


def inject_stuck(test: pd.DataFrame, variable: str, rng) -> tuple[pd.DataFrame, np.ndarray]:
    """Freeze one variable at its current value for runs of 1 to 3 hours.

    A run never crosses from one space into the next, because the rows are
    sorted by space and a run that reaches another space is skipped.
    """
    altered = test.copy()
    labels = np.zeros(len(test), dtype=bool)
    spaces = test[cfg.SPACE_COLUMN].to_numpy()

    runs = 0
    while runs < N_STUCK_RUNS:
        length = int(rng.integers(STUCK_STEPS[0], STUCK_STEPS[1] + 1))
        start = int(rng.integers(0, len(test) - length))
        end = start + length
        if spaces[start] != spaces[end - 1] or labels[start:end].any():
            continue
        frozen = altered.at[start, variable]
        altered.loc[start + 1 : end - 1, variable] = frozen
        # The first step is still a real reading; the repeats are the anomaly.
        labels[start + 1 : end] = True
        runs += 1
    return altered, labels


def _labels(n: int, rows: np.ndarray) -> np.ndarray:
    """A boolean array of length n, True at the given rows."""
    labels = np.zeros(n, dtype=bool)
    labels[rows] = True
    return labels


def score(flagged: np.ndarray, labels: np.ndarray) -> dict:
    """Precision and recall of the flags against the injected labels.

    Args:
        flagged: True where the detector marked the reading.
        labels: True where the reading was injected.

    Returns:
        Counts and the two rates; a rate is None when it is undefined.
    """
    true_positive = int((flagged & labels).sum())
    false_positive = int((flagged & ~labels).sum())
    false_negative = int((~flagged & labels).sum())
    marked = true_positive + false_positive
    injected = true_positive + false_negative
    return {
        "injected": injected,
        "detected": true_positive,
        "false_alarms": false_positive,
        "precision": round(true_positive / marked, 3) if marked else None,
        "recall": round(true_positive / injected, 3) if injected else None,
    }


def evaluate() -> dict:
    """Run every kind of injection on every checked variable.

    Returns:
        The false alarms on the untouched test split, and the scores by kind
        and variable.
    """
    test, parameters = test_split()
    rng = np.random.default_rng(cfg.SEED)

    clean = det.flags(test, parameters)
    results = {
        "test_readings": len(test),
        "false_alarms_clean": {v: int(clean[v].sum()) for v in parameters},
        "bands": {v: det.band(p) for v, p in parameters.items()},
        "by_kind": {},
    }

    injections = {"spike": inject_spikes, "damaged": inject_damaged, "stuck": inject_stuck}
    for kind, inject in injections.items():
        results["by_kind"][kind] = {}
        for variable in parameters:
            altered, labels = inject(test, variable, rng)
            flagged = det.flags(altered, parameters)[variable].to_numpy()
            results["by_kind"][kind][variable] = score(flagged, labels)
    return results


def print_report(results: dict) -> None:
    """Print the evaluation as tables."""
    print(f"Lecturas de prueba: {results['test_readings']:,}")
    print("\nBanda normal (media ± 3σ, calculada con entrenamiento) y falsas alarmas")
    print("en el conjunto de prueba sin tocar:")
    for variable, (low, high) in results["bands"].items():
        alarms = results["false_alarms_clean"][variable]
        print(f"  {variable:12} {low:9.1f} a {high:8.1f}   {alarms:4} falsas alarmas")

    for kind, by_variable in results["by_kind"].items():
        print(f"\n== {kind} ==")
        print(f"  {'variable':12} {'inyectadas':>10} {'detectadas':>10} {'recall':>7} {'precision':>9} {'falsas':>7}")
        for variable, s in by_variable.items():
            recall = "-" if s["recall"] is None else f"{s['recall']:.2f}"
            precision = "-" if s["precision"] is None else f"{s['precision']:.2f}"
            print(
                f"  {variable:12} {s['injected']:>10} {s['detected']:>10} "
                f"{recall:>7} {precision:>9} {s['false_alarms']:>7}"
            )


def main() -> int:
    """Evaluate the detector, print the tables and save them.

    Returns:
        Process exit code.
    """
    results = evaluate()
    print_report(results)

    cfg.MODELS.mkdir(parents=True, exist_ok=True)
    path = cfg.MODELS / "anomaly_evaluation.json"
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGuardado en {path.relative_to(cfg.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
