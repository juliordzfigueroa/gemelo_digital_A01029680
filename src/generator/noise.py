"""The ε of every equation: what the physics does not explain, as measured.

Section 4.4.2 closes each model with a noise term estimated from the
residuals between the equation and the readings. Those residuals are not
white noise. The CO2 sensor is a metal-oxide one: it drifts and reacts to
cleaning products, so on an empty Sunday it can read 1,500 ppm for hours. A
residual like that is slow, and it has a mean of its own.

So ε is a first-order autoregressive process, AR(1), fitted to the residuals
of each room:

    ε(t + Δt) = μ + φ (ε(t) − μ) + σ η,    η ~ N(0, 1)

μ is the mean of the residuals, φ how much of a residual carries over to
the next 5 minutes, and σ the size of the new part. With φ near 1 the
excursions last hours, as in the readings.

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-11-26 - Initial implementation.

Reference:
    Section 4.4.2 of the research document.
"""

from __future__ import annotations

import numpy as np

# φ is kept just below 1: at 1 the process would wander without a mean.
MAX_PHI = 0.999


def fit(residual: np.ndarray) -> dict:
    """Fit the AR(1) to a series of residuals, measured minus equation.

    Args:
        residual: One value per 5-minute step; NaN where there was no
            reading. Only pairs of consecutive steps enter φ.

    Returns:
        {"mean", "phi", "sigma"}.
    """
    mean = float(np.nanmean(residual))
    now, after = residual[:-1] - mean, residual[1:] - mean
    pairs = ~np.isnan(now) & ~np.isnan(after)
    phi = float((now[pairs] * after[pairs]).sum() / (now[pairs] ** 2).sum())
    phi = min(max(phi, 0.0), MAX_PHI)
    sigma = float(np.nanstd(residual) * np.sqrt(1 - phi**2))
    return {"mean": mean, "phi": phi, "sigma": sigma}


def start(params: dict, rng) -> float:
    """Draw a first value of ε, from the spread the process settles into.

    Args:
        params: Output of fit().
        rng: A numpy Generator.

    Returns:
        A value of ε.
    """
    spread = params["sigma"] / np.sqrt(1 - params["phi"] ** 2)
    return params["mean"] + rng.normal(0.0, spread)


def step(e: float, params: dict, rng) -> float:
    """Advance ε by one 5-minute step.

    Args:
        e: ε now.
        params: Output of fit().
        rng: A numpy Generator.

    Returns:
        ε after the step.
    """
    return params["mean"] + params["phi"] * (e - params["mean"]) + rng.normal(0.0, params["sigma"])
