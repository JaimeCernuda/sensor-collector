"""Base protocol and data structures for model wrappers.

All model wrappers implement the ModelWrapper protocol, producing
PredictionResult objects with a unified interface regardless of the
underlying TSFM library.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@dataclass
class PredictionResult:
    """Standardized prediction output from any model.

    Attributes:
        point_forecast: Point prediction, shape (horizon,).
        quantile_lo: Lower bound (e.g., 10th percentile), or None.
        quantile_hi: Upper bound (e.g., 90th percentile), or None.
        quantiles: Full quantile array if available,
            shape (n_quantiles, horizon), or None.
        quantile_levels: Quantile levels corresponding
            to rows of quantiles array.
        inference_time_ms: Wall-clock inference time in milliseconds.
        model_name: Name of the model that produced this prediction.
        device: Device used for inference ("cpu" or "cuda").
    """

    point_forecast: NDArray[np.floating]
    quantile_lo: NDArray[np.floating] | None = None
    quantile_hi: NDArray[np.floating] | None = None
    quantiles: NDArray[np.floating] | None = None
    quantile_levels: NDArray[np.floating] | None = None
    inference_time_ms: float = 0.0
    model_name: str = ""
    device: str = ""


@runtime_checkable
class ModelWrapper(Protocol):
    """Protocol that all model wrappers must implement."""

    @property
    def name(self) -> str:
        """Human-readable model name."""
        ...

    @property
    def supports_covariates(self) -> bool:
        """Whether the model can use exogenous covariate features."""
        ...

    @property
    def supports_quantiles(self) -> bool:
        """Whether the model produces probabilistic (quantile) output."""
        ...

    def load(self, device: str = "auto") -> None:
        """Download and load model weights.

        Args:
            device: Target device ("auto", "cpu", "cuda", "cuda:0", etc.).
        """
        ...

    def predict(
        self,
        context: NDArray[np.floating],
        horizon: int,
        covariates: NDArray[np.floating] | None = None,
        future_covariates: NDArray[np.floating] | None = None,
    ) -> PredictionResult:
        """Generate a forecast.

        Args:
            context: Historical target values, shape (context_len,).
            horizon: Number of steps to predict.
            covariates: Historical covariate values,
                shape (context_len, n_features), or None.
            future_covariates: Future covariate values,
                shape (horizon, n_features), or None.

        Returns:
            PredictionResult with at least point_forecast filled in.
        """
        ...

    def memory_footprint_mb(self) -> float:
        """Estimated GPU/CPU memory usage in MB."""
        ...
