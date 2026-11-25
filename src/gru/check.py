"""Check that the GRU can learn a series whose answer is known.

This is the test that stands in for reproducing a published benchmark: if the
network cannot fit a clean, deterministic series, the bug is in the code and
not in the data. It is not a performance result to report, only evidence that
the pipeline works.

Usage, from the src/ directory:
    python -m gru.check

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-11-25 - Moved here from gru/train.py.

Reference:
    Section 4.2.6 of the research document.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import config as cfg
from common import metrics as met
from gru import train as trn

# Below this R2 on a clean, predictable series, the network is not learning.
MIN_R2 = 0.5


def known_series(days: int = 20, n_spaces: int = 2) -> pd.DataFrame:
    """Build a clean, deterministic series whose structure is known exactly.

    Classes run on weekdays from 8:00 to 14:00. CO2 follows the occupancy with
    a first-order lag, temperature and humidity follow a daily cycle plus the
    occupancy, light follows daylight, and movement is the occupancy itself.
    No noise and no gaps: a network that cannot fit this has a bug.

    Args:
        days: Length of the series in days, starting on a Monday.
        n_spaces: How many spaces, each with a different CO2 amplitude so the
            space embedding has something to learn.

    Returns:
        A frame satisfying the data contract.
    """
    stamps = pd.date_range("2026-01-05", periods=days * 288, freq="5min")
    hour = np.asarray(stamps.hour + stamps.minute / 60)
    weekday = np.asarray(stamps.dayofweek < 5)
    occupancy = (weekday & (hour >= 8) & (hour < 14)).astype(float)
    cycle = np.sin(2 * np.pi * (hour - 9) / 24)
    sun = np.clip(np.sin(np.pi * (hour - 7) / 12), 0, None)

    parts = []
    for space in range(1, n_spaces + 1):
        # CO2 approaches its target a little at every step: a first-order lag, the way a room fills up and empties.
        target = 420 + 600 * (1 + 0.3 * (space - 1)) * occupancy
        co2 = np.empty_like(target)
        level = 420.0
        for i, goal in enumerate(target):
            level += 0.08 * (goal - level)
            co2[i] = level

        parts.append(pd.DataFrame({
            cfg.TIME_COLUMN: stamps,
            cfg.SPACE_COLUMN: space,
            "co2": co2,
            "temperature": 22 + 1.5 * cycle + (co2 - 420) / 600,
            "humidity": 50 - 4 * cycle + 3 * occupancy,
            "light": np.clip(70 * sun + 20 * occupancy, 0, 100),
            "motion": occupancy,
        }))
    return pd.concat(parts, ignore_index=True)


def main() -> int:
    """Train on the known series and report whether the network learned it.

    Returns:
        Process exit code: 0 when the network learns, 1 when it does not.
    """
    print("== Prueba de implementacion: serie conocida sin ruido ni huecos, 20 dias ==")
    output = trn.train(known_series(), epochs=30, patience=8)
    met.print_report(output["results"], "Prueba de implementacion")

    continuous = [v for v in cfg.ACTIVE_VARIABLES if v not in cfg.BINARY]
    r2_by_variable = {v: output["results"]["by_variable"][v]["r2"] for v in continuous}
    r2_text = ", ".join(f"{v}={r:.3f}" for v, r in r2_by_variable.items())
    print(f"\n  R2 global {output['results']['global']['r2']:.4f} | R2 de las continuas: {r2_text}")

    if min(r2_by_variable.values()) < MIN_R2:
        print(f"\n  FALLO: hay variables continuas con R2 < {MIN_R2} sobre una serie")
        print("  limpia y predecible. El problema esta en el codigo, no en los datos.")
        return 1
    print("\n  OK: la red aprende la estructura de una serie conocida.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
