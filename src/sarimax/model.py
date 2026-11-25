"""SARIMAX baseline: one univariate model per space and variable.

This is the statistical yardstick of section 4.2.5, not a component of the GRU.
The two models are trained independently, by different criteria, and only their
predictions meet, on the same test split and through the same metrics:

    GRU      minimizes MSE by backpropagation, updating weights step by step.
    SARIMAX  maximizes likelihood through a Kalman filter, solving for its
             coefficients in one go.

There is no loss value that can travel from one to the other, and forcing one
would be counterproductive: training the GRU to match SARIMAX's output would
cap it at SARIMAX's accuracy, which defeats the purpose of having a baseline to
beat.

SARIMAX is also univariate, so a model for 5 variables across 5 spaces means 25
independent fits. That is the second structural difference from the GRU, which
is a single global model.

On seasonality. Section 4.2.7.1 states a seasonal period of s=288, which is one
day at 5-minute sampling. That is correct in principle and intractable in
practice: the state-space representation grows with s, so a seasonal term of
288 drags a 289-dimensional state through every Kalman step. Measured on the
project data, that was 2147 seconds for a single model against 0.82 seconds
for the Fourier alternative, a factor of 2600, and 25 models are needed. The
seasonal mode was removed after that measurement; the numbers are kept in
src/sarimax/README.md.

The Fourier approach uses a part of the same equation that is already there.
Instead of asking the model to look 288 steps back, sine and cosine terms with
a period of 288 enter through the exogenous term, which the general equation
already carries as sum(beta_i X_i,t). It captures the same daily periodicity
with 2K regressors instead of 289 state dimensions.

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-11-25 - Kept only the Fourier mode and removed unused helpers.

Reference:
    Section 4.2.7.1 of the research document.
    Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of
    forecast accuracy. International Journal of Forecasting, 22(4), 679-688.
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
        # both would bury the output across 25 fits.
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

    @property
    def aic(self) -> float:
        """Akaike information criterion of the fit, or nan before fitting."""
        return float(self.result.aic) if self.result is not None else float("nan")
