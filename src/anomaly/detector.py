"""Statistical threshold detector: a reading more than 3 sigma away is an anomaly.

Section 4.2.5 of the research document: each reading is compared with the
mean and standard deviation of its variable, computed on the training split,
and a reading that falls more than three standard deviations away is marked.
No trained model is involved.

The mean and sigma are the ones the GRU normalizer already computes on the
same training split (common.data.Normalizer), so the statistics exist in one
place and the detector and the network agree on what "normal" is. Motion is
not checked: it is binary, so the normalizer does not keep statistics for it,
and a 3-sigma band around a 0/1 variable falls outside [0, 1] anyway.

How well this works is measured, not assumed: anomaly/evaluate.py injects
synthetic anomalies into the test split and reports precision and recall.

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-11-26 - Initial implementation.

Reference:
    Section 4.2.5 of the research document.
    Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly detection: A
    survey. ACM Computing Surveys, 41(3), 1-58.
"""

from __future__ import annotations

import pandas as pd

# How many standard deviations away a reading must be to count as an anomaly.
SIGMAS = 3.0


def deviation(value, statistics: dict):
    """How many standard deviations a value is from its variable's mean.

    Works on a single number or on a whole pandas Series, so the API and the
    evaluation apply literally the same arithmetic.

    Args:
        value: A reading, or a Series of readings, in physical units.
        statistics: {"mean": ..., "sigma": ...} of that variable.

    Returns:
        The signed distance in sigmas: positive above the mean.
    """
    return (value - statistics["mean"]) / statistics["sigma"]


def check(readings: dict, parameters: dict, variables: list[str] | None = None) -> list[dict]:
    """Find the anomalies in one reading of one space.

    Args:
        readings: Variable name to value, in physical units. Missing values
            are skipped.
        parameters: Mean and sigma per variable, as Normalizer.parameters.
        variables: Which variables to check. None checks every variable that
            has statistics.

    Returns:
        One dict per anomalous variable, with its value, its deviation in
        sigmas, the normal band and whether it is above or below it. Empty
        when everything is within the band.
    """
    anomalies = []
    for variable, statistics in parameters.items():
        if variables and variable not in variables:
            continue
        value = readings.get(variable)
        if value is None:
            continue
        z = deviation(value, statistics)
        if abs(z) > SIGMAS:
            anomalies.append({
                "variable": variable,
                "value": value,
                "sigma": round(float(z), 1),
                "band": band(statistics),
                "direction": "por encima" if z > 0 else "por debajo",
            })
    return anomalies


def band(statistics: dict) -> list[float]:
    """The normal band of a variable: mean plus and minus SIGMAS sigmas.

    Args:
        statistics: {"mean": ..., "sigma": ...} of that variable.

    Returns:
        [lower, upper], rounded for display.
    """
    spread = SIGMAS * statistics["sigma"]
    return [round(statistics["mean"] - spread, 1), round(statistics["mean"] + spread, 1)]


def flags(frame: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """Mark every anomalous reading of a whole frame at once.

    Used by the evaluation, which checks thousands of readings; check() is
    the same rule applied to one reading.

    Args:
        frame: Rows with one column per variable, in physical units.
        parameters: Mean and sigma per variable, as Normalizer.parameters.

    Returns:
        A boolean frame, one column per checked variable, True where the
        reading is an anomaly. Missing readings are never anomalies.
    """
    output = pd.DataFrame(index=frame.index)
    for variable, statistics in parameters.items():
        z = deviation(frame[variable], statistics)
        output[variable] = z.abs() > SIGMAS
    return output
