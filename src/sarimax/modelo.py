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
288 drags a 289-dimensional state through every Kalman step. Measured on one
week of data, that is 104 seconds for a single model against 0.3 seconds for
the Fourier alternative, and 25 models are needed.

The Fourier approach uses a part of the same equation that is already there.
Instead of asking the model to look 288 steps back, sine and cosine terms with
a period of 288 enter through the exogenous term, which the general equation
already carries as sum(beta_i X_i,t). It captures the same daily periodicity
with 2K regressors instead of 289 state dimensions.

Both are available through `estacionalidad`, so the cost can be measured rather
than asserted.

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-10-01 - Initial implementation.

Reference:
    Section 4.2.7.1 of the research document.
    Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of
    forecast accuracy. International Journal of Forecasting, 22(4), 679-688.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from comun import config as cfg

# Non-seasonal order (p, d, q). The research document calls for choosing it by
# ACF and PACF analysis on the training split; this is a reasonable default
# while that analysis is pending, and it is a parameter, not a hard-coded
# decision.
ORDEN_POR_OMISION = (2, 0, 2)

# How many sine/cosine pairs model the daily cycle. Four pairs reproduce a
# smooth daily shape; more would start fitting noise.
ARMONICOS_POR_OMISION = 4

# Daily period at the project sampling interval: 288 steps of 5 minutes.
PERIODO_DIARIO = 24 * 3600 // cfg.INTERVALO_SEGUNDOS


def terminos_fourier(
    n: int, periodo: int = PERIODO_DIARIO, armonicos: int = ARMONICOS_POR_OMISION,
    inicio: int = 0,
) -> np.ndarray:
    """Build sine and cosine regressors for a cycle of a given period.

    These enter SARIMAX as exogenous variables, which is the sum(beta_i X_i,t)
    term of the general equation. They tell the model where it sits in the
    daily cycle instead of making it look a whole day back.

    Args:
        n: How many consecutive steps to generate.
        periodo: Length of the cycle in steps.
        armonicos: Number of sine/cosine pairs. The result has 2 * armonicos
            columns.
        inicio: Index of the first step, so that a forecast continues the same
            phase as the data it follows.

    Returns:
        An array shaped (n, 2 * armonicos).
    """
    t = np.arange(inicio, inicio + n)
    columnas = []
    for k in range(1, armonicos + 1):
        angulo = 2 * np.pi * k * t / periodo
        columnas.append(np.sin(angulo))
        columnas.append(np.cos(angulo))
    return np.column_stack(columnas)


class BaselineSARIMAX:
    """One fitted SARIMAX model for a single variable of a single space."""

    def __init__(
        self,
        orden: tuple[int, int, int] = ORDEN_POR_OMISION,
        estacionalidad: str = "fourier",
        armonicos: int = ARMONICOS_POR_OMISION,
        periodo_estacional: int = PERIODO_DIARIO,
    ) -> None:
        """Configure the model without fitting it yet.

        Args:
            orden: Non-seasonal (p, d, q).
            estacionalidad: "fourier" to carry the daily cycle as exogenous
                regressors, or "estacional" to use a seasonal SARIMA term with
                period `periodo_estacional`, which is what section 4.2.7.1
                states and is far slower.
            armonicos: Sine/cosine pairs, only used with "fourier".
            periodo_estacional: Cycle length in steps.

        Raises:
            ValueError: If estacionalidad is not one of the two accepted names.
        """
        if estacionalidad not in ("fourier", "estacional"):
            raise ValueError(
                f"estacionalidad must be 'fourier' or 'estacional', "
                f"got {estacionalidad!r}"
            )
        self.orden = orden
        self.estacionalidad = estacionalidad
        self.armonicos = armonicos
        self.periodo_estacional = periodo_estacional
        self.resultado = None
        self.n_entrenamiento = 0

    def _exogenas(self, n: int, inicio: int = 0) -> np.ndarray | None:
        """Produce the exogenous block for n steps, or None if unused.

        Args:
            n: How many steps.
            inicio: Index of the first step, to keep the cycle in phase.

        Returns:
            The Fourier regressors, or None under seasonal mode.
        """
        if self.estacionalidad != "fourier":
            return None
        return terminos_fourier(n, self.periodo_estacional, self.armonicos, inicio)

    def ajustar(self, serie: np.ndarray) -> "BaselineSARIMAX":
        """Fit the model by maximum likelihood on the training series.

        Args:
            serie: Training values in physical units, without missing data.

        Returns:
            This object, so calls can be chained.
        """
        self.n_entrenamiento = len(serie)
        estacional = (
            (1, 0, 1, self.periodo_estacional)
            if self.estacionalidad == "estacional"
            else (0, 0, 0, 0)
        )

        # statsmodels warns loudly about non-invertible starting parameters and
        # about convergence on series this long. Neither is actionable here and
        # both would bury the output across 25 fits.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            modelo = SARIMAX(
                serie,
                exog=self._exogenas(len(serie)),
                order=self.orden,
                seasonal_order=estacional,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self.resultado = modelo.fit(disp=False, maxiter=200)
        return self

    def pronosticar_desde(
        self, serie_completa: np.ndarray, origenes: np.ndarray, pasos: int
    ) -> np.ndarray:
        """Forecast `pasos` steps ahead from each of several origins.

        Applies the already fitted coefficients to the full series without
        refitting, then produces a dynamic forecast from each origin: values up
        to the origin are the observed ones, and everything after is the
        model's own output. That mirrors how the GRU is evaluated, which also
        sees 12 real steps and must produce the next 6 unaided.

        Args:
            serie_completa: The whole series in physical units, training split
                included, so the filter arrives at each origin with the same
                history the GRU had.
            origenes: Indices of the last observed step of each window.
            pasos: Forecast horizon.

        Returns:
            An array shaped (len(origenes), pasos).

        Raises:
            RuntimeError: If called before ajustar().
        """
        if self.resultado is None:
            raise RuntimeError("ajustar() must be called before forecasting")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            aplicado = self.resultado.apply(
                serie_completa, exog=self._exogenas(len(serie_completa)), refit=False
            )

        salida = np.empty((len(origenes), pasos), dtype=float)
        for fila, origen in enumerate(origenes):
            prediccion = aplicado.get_prediction(
                start=int(origen) + 1,
                end=int(origen) + pasos,
                dynamic=True,
            )
            salida[fila] = prediccion.predicted_mean
        return salida

    @property
    def aic(self) -> float:
        """Akaike information criterion of the fit.

        Returns:
            The AIC, or nan when the model has not been fitted.
        """
        return float(self.resultado.aic) if self.resultado is not None else float("nan")

    def resumen_orden(self) -> str:
        """Describe the configured order in one line.

        Returns:
            A readable label such as "(2,0,2) + Fourier K=4".
        """
        if self.estacionalidad == "fourier":
            return f"{self.orden} + Fourier K={self.armonicos}"
        return f"{self.orden} x (1,0,1,{self.periodo_estacional})"


def series_por_espacio(
    df: pd.DataFrame, variable: str
) -> dict[int, pd.DataFrame]:
    """Split a preprocessed frame into one ordered series per space.

    Args:
        df: Frame returned by comun.datos.preprocesar().
        variable: Which variable to extract.

    Returns:
        A dict from space id to a frame holding the timestamp, the variable,
        the partition and the block id, sorted by time.
    """
    columnas = [
        cfg.COLUMNA_TIEMPO, variable, "particion", "bloque_id", cfg.COLUMNA_ESPACIO
    ]
    salida = {}
    for espacio, grupo in df[columnas].groupby(cfg.COLUMNA_ESPACIO, sort=True):
        salida[int(espacio)] = grupo.sort_values(cfg.COLUMNA_TIEMPO).reset_index(
            drop=True
        )
    return salida
