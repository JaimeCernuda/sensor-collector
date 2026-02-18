"""TimesFM 2.5 model wrapper (Google, 200M).

Requires: install from source (not on PyPI for 2.5)
    git clone https://github.com/google-research/timesfm
    pip install -e ".[torch]"       # basic inference
    pip install -e ".[torch,xreg]"  # adds covariate support (requires JAX)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from tick2.models.base import PredictionResult


@dataclass
class TimesFMWrapper:
    """Wrapper for TimesFM 2.5 zero-shot inference.

    Supports univariate forecasting via ``forecast()`` and covariate
    forecasting via ``forecast_with_covariates()`` (requires JAX).
    """

    model_id: str = "google/timesfm-2.5-200m-pytorch"
    model_name: str = "timesfm-2.5"
    max_context: int = 1024
    _model: object = field(default=None, init=False, repr=False)
    _device: str = field(default="", init=False, repr=False)

    @property
    def name(self) -> str:
        return self.model_name

    @property
    def supports_covariates(self) -> bool:
        return True

    @property
    def supports_quantiles(self) -> bool:
        return True

    def load(self, device: str = "auto") -> None:
        """Load TimesFM 2.5 model weights.

        Args:
            device: "auto", "cpu", or "cuda". TimesFM uses PyTorch backend.
        """
        import timesfm

        self._model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.model_id)

        if device == "auto":
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

    def predict(
        self,
        context: NDArray[np.floating],
        horizon: int,
        covariates: NDArray[np.floating] | None = None,
        future_covariates: NDArray[np.floating] | None = None,
    ) -> PredictionResult:
        """Generate a forecast.

        Without covariates, uses the standard ``forecast()`` API.
        With covariates, uses ``forecast_with_covariates()`` which requires
        the xreg extra (JAX dependency).
        """
        if self._model is None:
            raise RuntimeError("Call load() before predict()")

        import timesfm

        # Configure forecast parameters
        self._model.compile(  # type: ignore[union-attr]
            timesfm.ForecastConfig(
                max_context=min(self.max_context, len(context)),
                max_horizon=horizon,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                return_backcast=(covariates is not None),
            )
        )

        t0 = time.perf_counter()

        if covariates is not None and future_covariates is not None:
            # Covariate forecasting
            cov_2d = covariates.reshape(len(context), -1)
            fut_2d = future_covariates.reshape(horizon, -1)
            n_features = cov_2d.shape[1]

            dynamic_covariates: dict[str, list[list[float]]] = {}
            for i in range(n_features):
                full_series = np.concatenate([cov_2d[:, i], fut_2d[:, i]])
                dynamic_covariates[f"cov_{i}"] = [full_series.tolist()]

            point_out, _xreg_out = self._model.forecast_with_covariates(  # type: ignore[union-attr]
                inputs=[context.astype(np.float32).tolist()],
                dynamic_numerical_covariates=dynamic_covariates,
                xreg_mode="xreg + timesfm",
            )
            point = np.array(point_out[0][:horizon], dtype=np.float64)
            quantiles_arr = None
            q_levels = None
        else:
            # Univariate forecasting
            point_out, quantile_out = self._model.forecast(  # type: ignore[union-attr]
                horizon=horizon,
                inputs=[context.astype(np.float32).tolist()],
            )
            point = np.array(point_out[0][:horizon], dtype=np.float64)

            # TimesFM returns 10 quantile levels
            if quantile_out is not None and len(quantile_out) > 0:
                quantiles_arr = np.array(quantile_out[0][:, :horizon], dtype=np.float64)
                q_levels = np.linspace(0.05, 0.95, quantiles_arr.shape[0])
            else:
                quantiles_arr = None
                q_levels = None

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        quantile_lo = None
        quantile_hi = None
        if quantiles_arr is not None and q_levels is not None:
            quantile_lo = quantiles_arr[0]  # lowest quantile
            quantile_hi = quantiles_arr[-1]  # highest quantile

        return PredictionResult(
            point_forecast=point,
            quantile_lo=quantile_lo,
            quantile_hi=quantile_hi,
            quantiles=quantiles_arr,
            quantile_levels=q_levels,
            inference_time_ms=elapsed_ms,
            model_name=self.model_name,
            device=self._device,
        )

    def memory_footprint_mb(self) -> float:
        """Estimated memory: ~200M params * 4 bytes ~ 800 MB."""
        if self._model is None:
            return 0.0
        return 800.0
