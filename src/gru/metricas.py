"""Evaluation metrics: MAE, RMSE, MAPE, sMAPE and R-squared.

Every metric is computed on DENORMALIZED values, in each variable's original
units. A MAE expressed in z-score units cannot be interpreted ("the error is
0.3 sigmas" tells nobody anything) nor compared against the literature.

On MAPE: it divides by the true value, so it is undefined wherever that value
is zero. That is not a rare corner case in this project, since light is exactly
zero at night and movement is zero most of the time. Here MAPE is computed only
over non-zero observations and the covered fraction is reported alongside, so
that a percentage derived from half the sample is never passed off as global.
For those variables the interpretable metric is MAE in physical units, or
sMAPE, which is symmetric and does not blow up at zero.

Reference:
    Section 4.2.6 of the research document.
    Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of
    forecast accuracy. International Journal of Forecasting, 22(4), 679-688.
"""

from __future__ import annotations

import numpy as np

import config as cfg

# Absolute floor below which a value counts as zero.
EPSILON_MAPE = 1e-6

# Fraction of a variable's typical magnitude below which a value counts as zero
# for MAPE purposes.
#
# A RELATIVE threshold is required, not just an absolute one: exact zeros do
# not survive the round trip through normalization. A 0 lux reading becomes
# z = (0 - mean) / sigma and comes back as roughly 1e-5 due to float32 rounding.
# Dividing by 1e-5 yields a MAPE in the millions of percent, which is exactly
# the symptom this guards against. At 1e-4 of the mean absolute value, a
# variable averaging 200 lux discards anything under 0.02 lux: values that are
# zero in any physical sense.
FRACCION_CERO = 1e-4


def _umbral_cero(real: np.ndarray) -> float:
    """Compute the threshold below which a true value counts as zero.

    Args:
        real: Ground-truth values for one variable.

    Returns:
        The larger of the absolute floor and the scale-relative threshold.
    """
    escala = float(np.mean(np.abs(real)))
    return max(EPSILON_MAPE, FRACCION_CERO * escala)


def mae(real: np.ndarray, pred: np.ndarray) -> float:
    """Compute the mean absolute error.

    Args:
        real: Ground-truth values.
        pred: Predicted values, same shape.

    Returns:
        Mean absolute error in the variable's own units.
    """
    return float(np.mean(np.abs(real - pred)))


def rmse(real: np.ndarray, pred: np.ndarray) -> float:
    """Compute the root mean squared error.

    Penalizes large errors more than MAE, which is what makes it the right
    companion metric when a big miss matters more than several small ones,
    such as failing to see a CO2 spike.

    Args:
        real: Ground-truth values.
        pred: Predicted values, same shape.

    Returns:
        Root mean squared error in the variable's own units.
    """
    return float(np.sqrt(np.mean((real - pred) ** 2)))


def mape(real: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    """Compute the mean absolute percentage error over non-zero observations.

    Args:
        real: Ground-truth values.
        pred: Predicted values, same shape.

    Returns:
        A tuple of (mape in percent, covered fraction of observations). If no
        observation is non-zero, returns (nan, 0.0) rather than inventing a
        number.
    """
    utiles = np.abs(real) > _umbral_cero(real)
    cobertura = float(utiles.mean())
    if not utiles.any():
        return float("nan"), 0.0
    error = np.abs((real[utiles] - pred[utiles]) / real[utiles])
    return float(100 * np.mean(error)), cobertura


def smape(real: np.ndarray, pred: np.ndarray) -> float:
    """Compute the symmetric MAPE, which stays finite at zero.

    Divides by the average of the true and predicted magnitudes instead of by
    the true value alone. When both are zero the term is taken as zero, a
    perfect hit, rather than 0/0.

    Args:
        real: Ground-truth values.
        pred: Predicted values, same shape.

    Returns:
        Symmetric mean absolute percentage error, in percent.
    """
    umbral = _umbral_cero(real)
    denominador = (np.abs(real) + np.abs(pred)) / 2
    termino = np.where(denominador > umbral, np.abs(real - pred) / denominador, 0.0)
    return float(100 * np.mean(termino))


def r2(real: np.ndarray, pred: np.ndarray) -> float:
    """Compute the coefficient of determination.

    The result can come out negative, meaning the model is worse than always
    predicting the mean. That is useful information and is not clipped to zero.

    Args:
        real: Ground-truth values.
        pred: Predicted values, same shape.

    Returns:
        R-squared, or nan when the true series is constant and R-squared is
        undefined.

    Reference:
        Kvalseth, T. O. (1985). Cautionary note about R-squared. The American
        Statistician, 39(4), 279-285.
    """
    residual = np.sum((real - pred) ** 2)
    total = np.sum((real - np.mean(real)) ** 2)
    if total < EPSILON_MAPE:
        return float("nan")
    return float(1 - residual / total)


def evaluar(
    real: np.ndarray, pred: np.ndarray, variables: list[str] | None = None
) -> dict:
    """Compute every metric, per variable and aggregated.

    Args:
        real: Denormalized ground truth, shaped (n, pasos_salida, n_variables).
        pred: Denormalized predictions, same shape.
        variables: Variable names in the order of the last axis. Defaults to
            VARIABLES_ACTIVAS.

    Returns:
        A dict with 'por_variable' (one entry per variable) and 'global'.

    Raises:
        ValueError: If the two arrays have different shapes.
    """
    variables = variables or cfg.VARIABLES_ACTIVAS
    if real.shape != pred.shape:
        raise ValueError(f"shape mismatch: real {real.shape}, pred {pred.shape}")

    por_variable = {}
    for i, variable in enumerate(variables):
        r, p = real[:, :, i].ravel(), pred[:, :, i].ravel()
        valor_mape, cobertura = mape(r, p)
        por_variable[variable] = {
            "unidad": cfg.UNIDADES.get(variable, ""),
            "mae": mae(r, p),
            "rmse": rmse(r, p),
            "mape": valor_mape,
            "mape_cobertura": cobertura,
            "smape": smape(r, p),
            "r2": r2(r, p),
            "es_binaria": variable in cfg.BINARIAS,
        }

    # The global MAE is the mean of the per-variable MAEs, not the MAE over
    # everything pooled. Without that, CO2 in hundreds of ppm would completely
    # dominate temperature in tens of degrees and the number would stop
    # meaning anything.
    return {
        "por_variable": por_variable,
        "global": {
            "mae": float(np.mean([m["mae"] for m in por_variable.values()])),
            "rmse": float(np.mean([m["rmse"] for m in por_variable.values()])),
            "r2": float(np.mean([m["r2"] for m in por_variable.values()])),
        },
    }


def por_horizonte(
    real: np.ndarray, pred: np.ndarray, variables: list[str] | None = None
) -> dict:
    """Compute MAE at each forecast step, to show how error grows with horizon.

    This is the number that says whether the 30 minutes of direct prediction
    hold up or whether the error is already unacceptable by the fourth step.

    Args:
        real: Denormalized ground truth, shaped (n, pasos_salida, n_variables).
        pred: Denormalized predictions, same shape.
        variables: Variable names in the order of the last axis.

    Returns:
        A dict keyed by horizon label, e.g. "+15min", each holding the MAE of
        every variable at that step.
    """
    variables = variables or cfg.VARIABLES_ACTIVAS
    salida = {}
    for paso in range(real.shape[1]):
        minutos = (paso + 1) * cfg.INTERVALO_SEGUNDOS // 60
        salida[f"+{minutos}min"] = {
            variable: mae(real[:, paso, i], pred[:, paso, i])
            for i, variable in enumerate(variables)
        }
    return salida


def imprimir(resultados: dict, titulo: str = "Metricas") -> None:
    """Print a metrics table, flagging the variables whose MAPE is unreliable.

    Args:
        resultados: Output of evaluar().
        titulo: Heading shown above the table.
    """
    print(f"\n== {titulo} ==")
    encabezado = (
        f"  {'variable':13} {'unidad':7} {'MAE':>10} {'RMSE':>10} "
        f"{'MAPE':>9} {'cob.':>6} {'sMAPE':>8} {'R2':>8}"
    )
    print(encabezado)
    print("  " + "-" * (len(encabezado) - 2))
    for variable, m in resultados["por_variable"].items():
        texto_mape = "    n/a" if np.isnan(m["mape"]) else f"{m['mape']:8.2f}%"
        nota = "  <- binaria, MAPE y R2 poco informativos" if m["es_binaria"] else ""
        print(
            f"  {variable:13} {m['unidad']:7} {m['mae']:10.4f} {m['rmse']:10.4f} "
            f"{texto_mape} {100 * m['mape_cobertura']:5.1f}% {m['smape']:7.2f}% "
            f"{m['r2']:8.4f}{nota}"
        )
    g = resultados["global"]
    print(f"\n  global: MAE {g['mae']:.4f} | RMSE {g['rmse']:.4f} | R2 {g['r2']:.4f}")

    bajas = [
        v
        for v, m in resultados["por_variable"].items()
        if m["mape_cobertura"] < 0.95 and not np.isnan(m["mape"])
    ]
    if bajas:
        print(
            f"  Aviso: el MAPE de {', '.join(bajas)} se calculo sobre menos del 95% "
            "de las observaciones (el resto vale 0). Usar el MAE o el sMAPE."
        )
