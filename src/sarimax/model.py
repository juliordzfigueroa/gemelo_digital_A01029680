"""SARIMAX baseline: one univariate model per space and variable.

This is the statistical yardstick of section 4.2.5, not a component of the GRU.
Where the GRU is one global model for every space and variable, SARIMAX needs
one fit per variable and space. compare_models.py explains why nothing passes
from one to the other.

The daily cycle enters as sine and cosine terms through the exogenous part of
the general equation, sum(beta_i X_i,t), instead of the seasonal period s=288
of section 4.2.7.1. Same daily periodicity, but with 2K regressors instead of a
289-dimensional state: measured, 0.82 s per model against 2147 s. The full
measurement is in src/sarimax/README.md.

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-11-26 - Shortened this docstring; AIC is read straight from the fit.

Reference:
    Section 4.2.7.1 of the research document.
    Hyndman, R. J., & Athanasopoulos, G. (2021). Forecasting: principles and
    practice (3rd ed.), section 10.5, dynamic harmonic regression. OTexts.
"""

from __future__ import annotations

import warnings

import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

from common import config as cfg

# Non-seasonal order (p, d, q). The research document calls for choosing it by
# ACF and PACF analysis on the training split; this is a reasonable default
# while that analysis is pending.
ORDER = (2, 0, 2)

# How many sine/cosine pairs model the daily cycle. Four pairs reproduce a
# smooth daily shape; more would start fitting noise.
HARMONICS = 4

# Daily period at the project sampling interval: 288 steps of 5 minutes.
DAILY_PERIOD = 24 * 3600 // cfg.INTERVAL_SECONDS


def fourier_terms(n: int, start: int = 0) -> np.ndarray:
    """Build the sine and cosine regressors of the daily cycle.

    These enter SARIMAX as exogenous variables, which is the sum(beta_i X_i,t)
    term of the general equation. They tell the model where it sits in the
    daily cycle instead of making it look a whole day back.

    Args:
        n: How many consecutive steps to generate.
        start: Index of the first step, so that a forecast continues the same
            phase as the data it follows.

    Returns:
        An array shaped (n, 2 * HARMONICS).
    """
    t = np.arange(start, start + n)
    columns = []
    for k in range(1, HARMONICS + 1):
        angle = 2 * np.pi * k * t / DAILY_PERIOD
        columns.append(np.sin(angle))
        columns.append(np.cos(angle))
    return np.column_stack(columns)


class BaselineSARIMAX:
    """One fitted SARIMAX model for a single variable of a single space."""

    def __init__(self) -> None:
        """Create an unfitted model with the project order and harmonics."""
        self.result = None

    def fit(self, series: np.ndarray) -> "BaselineSARIMAX":
        """Fit the model by maximum likelihood on the training series.

        Args:
            series: Training values in physical units. Missing values are
                allowed: the Kalman filter skips them.

        Returns:
            This object, so calls can be chained.
        """
        # statsmodels warns loudly about non-invertible starting parameters and
        # about convergence on series this long. Neither is actionable here and
        # both would bury the output across every fit.
        #
        # Stationarity is enforced because d=0 assumes it. On the real sensor
        # data, temperature and humidity are persistent enough at 5 minutes
        # that the unconstrained fit landed on AR roots of 1.000 and 1.057, an
        # explosive process whose 30-minute forecasts reached 1e76.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                series,
                exog=fourier_terms(len(series)),
                order=ORDER,
                enforce_stationarity=True,
                enforce_invertibility=False,
            )
            self.result = model.fit(disp=False, maxiter=200)
        return self

    def forecast_from(
        self, full_series: np.ndarray, origins: np.ndarray, steps: int
    ) -> np.ndarray:
        """Forecast `steps` steps ahead from each of several origins.

        Applies the already fitted coefficients to the full series without
        refitting, then produces a dynamic forecast from each origin: values up
        to the origin are the observed ones, and everything after is the
        model's own output. That mirrors how the GRU is evaluated, which also
        sees 12 real steps and must produce the next 6 unaided.

        Args:
            full_series: The whole series in physical units, training split
                included, so the filter arrives at each origin with the same
                history the GRU had.
            origins: Indices of the last observed step of each window.
            steps: Forecast horizon.

        Returns:
            An array shaped (len(origins), steps).

        Raises:
            RuntimeError: If called before fit().
        """
        if self.result is None:
            raise RuntimeError("fit() must be called before forecasting")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            applied = self.result.apply(
                full_series, exog=fourier_terms(len(full_series)), refit=False
            )

        output = np.empty((len(origins), steps), dtype=float)
        for row, origin in enumerate(origins):
            prediction = applied.get_prediction(
                start=int(origin) + 1,
                end=int(origin) + steps,
                dynamic=True,
            )
            output[row] = prediction.predicted_mean
        return output
