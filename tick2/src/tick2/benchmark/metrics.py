"""Forecasting evaluation metrics.

Includes both point-forecast metrics (MAE, RMSE, MASE) and probabilistic
metrics (CRPS, interval coverage, Winkler score).
"""

import numpy as np
from numpy.typing import NDArray


def mae(y_true: NDArray[np.floating], y_pred: NDArray[np.floating]) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: NDArray[np.floating], y_pred: NDArray[np.floating]) -> float:
    """Root Mean Square Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mase(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
    y_train: NDArray[np.floating],
    seasonality: int = 1,
) -> float:
    """Mean Absolute Scaled Error.

    Scaled by the in-sample naive forecast error (seasonal random walk).

    Args:
        y_true: True values in the test horizon.
        y_pred: Predicted values.
        y_train: Training series (for computing naive error scale).
        seasonality: Seasonal period for the naive baseline.

    Returns:
        MASE score. Values < 1 beat the naive baseline.
    """
    naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])
    scale = np.mean(naive_errors)
    if scale == 0:
        return float("inf")
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def crps_empirical(
    y_true: NDArray[np.floating],
    samples: NDArray[np.floating],
) -> float:
    """Continuous Ranked Probability Score (empirical estimate).

    Computed from forecast samples using the energy form:
    CRPS = E|X - y| - 0.5 * E|X - X'|

    Args:
        y_true: True values, shape (horizon,).
        samples: Forecast samples, shape (n_samples, horizon).

    Returns:
        Mean CRPS over the horizon.
    """
    n_samples = samples.shape[0]

    # E|X - y|: mean absolute error of each sample vs truth
    abs_diff = np.mean(np.abs(samples - y_true[np.newaxis, :]), axis=0)

    # E|X - X'|: mean pairwise absolute difference between samples
    pairwise = 0.0
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            pairwise += np.mean(np.abs(samples[i] - samples[j]))
    pairwise = 2.0 * pairwise / (n_samples * (n_samples - 1)) if n_samples > 1 else 0.0

    crps_per_step = abs_diff - 0.5 * pairwise
    return float(np.mean(crps_per_step))


def crps_quantile(
    y_true: NDArray[np.floating],
    quantile_forecasts: NDArray[np.floating],
    quantile_levels: NDArray[np.floating],
) -> float:
    """CRPS estimated from quantile forecasts via the quantile loss formulation.

    CRPS = 2 * mean over q of: quantile_loss(q, y, y_hat_q)

    Args:
        y_true: True values, shape (horizon,).
        quantile_forecasts: Quantile predictions, shape (n_quantiles, horizon).
        quantile_levels: Quantile levels, shape (n_quantiles,), e.g. [0.1, 0.5, 0.9].

    Returns:
        Approximate CRPS.
    """
    total = 0.0
    for i, q in enumerate(quantile_levels):
        error = y_true - quantile_forecasts[i]
        loss = np.where(error >= 0, q * error, (q - 1) * error)
        total += np.mean(loss)
    return float(2.0 * total / len(quantile_levels))


def coverage(
    y_true: NDArray[np.floating],
    lower: NDArray[np.floating],
    upper: NDArray[np.floating],
) -> float:
    """Prediction interval coverage: fraction of true values within bounds.

    Args:
        y_true: True values, shape (horizon,).
        lower: Lower bound of prediction interval.
        upper: Upper bound of prediction interval.

    Returns:
        Coverage fraction in [0, 1].
    """
    inside = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(inside))


def winkler_score(
    y_true: NDArray[np.floating],
    lower: NDArray[np.floating],
    upper: NDArray[np.floating],
    alpha: float = 0.2,
) -> float:
    """Winkler interval score: measures both sharpness and coverage.

    Lower is better. Penalizes wide intervals and missed coverage.

    Args:
        y_true: True values.
        lower: Lower prediction bound.
        upper: Upper prediction bound.
        alpha: Nominal miscoverage rate (e.g., 0.2 for 80% intervals).

    Returns:
        Mean Winkler score.
    """
    width = upper - lower
    penalty_lo = (2.0 / alpha) * np.maximum(lower - y_true, 0)
    penalty_hi = (2.0 / alpha) * np.maximum(y_true - upper, 0)
    scores = width + penalty_lo + penalty_hi
    return float(np.mean(scores))
