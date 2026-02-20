"""Chronos-2 model wrapper (Amazon, 28M/120M).

Requires: pip install "chronos-forecasting[extras]>=2.2"
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from tick2.models.base import PredictionResult


@dataclass
class Chronos2Wrapper:
    """Wrapper for Chronos-2 zero-shot inference.

    Supports both univariate and multivariate (covariate) prediction
    via the ``predict_df`` API that takes long-format DataFrames.
    """

    model_id: str = "amazon/chronos-2"
    model_name: str = "chronos2-base"
    quantile_levels: list[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    _pipeline: object = field(default=None, init=False, repr=False)
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
        """Load model weights.

        Args:
            device: "auto", "cpu", "cuda", or "cuda:N".
        """
        from chronos import BaseChronosPipeline

        device_map = device if device != "auto" else "cuda"
        try:
            import torch

            if device_map == "cuda" and not torch.cuda.is_available():
                device_map = "cpu"
        except ImportError:
            device_map = "cpu"

        self._pipeline = BaseChronosPipeline.from_pretrained(
            self.model_id,
            device_map=device_map,
        )
        self._device = device_map

    def predict(
        self,
        context: NDArray[np.floating],
        horizon: int,
        covariates: NDArray[np.floating] | None = None,
        future_covariates: NDArray[np.floating] | None = None,
    ) -> PredictionResult:
        """Generate a forecast using Chronos-2.

        For multivariate prediction, pass covariates as historical sensor
        values and future_covariates for known-future values. Both must
        have consistent column ordering.
        """
        if self._pipeline is None:
            raise RuntimeError("Call load() before predict()")

        import pandas as pd

        ctx_len = len(context)

        # Build long-format context DataFrame
        ctx_df = pd.DataFrame(
            {
                "id": ["target"] * ctx_len,
                "timestamp": pd.date_range("2024-01-01", periods=ctx_len, freq="1s"),
                "target": context.astype(np.float32),
            }
        )

        future_df = None

        if covariates is not None:
            n_features = covariates.shape[1] if covariates.ndim > 1 else 1
            cov_2d = covariates.reshape(ctx_len, -1).astype(np.float32)
            cov_df = pd.DataFrame(
                cov_2d,
                columns=[f"cov_{i}" for i in range(n_features)],
            )
            ctx_df = pd.concat([ctx_df, cov_df], axis=1)

            # If future covariates provided, build future_df with same columns
            if future_covariates is not None:
                fut_2d = future_covariates.reshape(horizon, -1)
                fut_data: dict[str, object] = {
                    "id": ["target"] * horizon,
                    "timestamp": pd.date_range(
                        ctx_df["timestamp"].iloc[-1] + pd.Timedelta(seconds=1),
                        periods=horizon,
                        freq="1s",
                    ),
                }
                for i in range(n_features):
                    fut_data[f"cov_{i}"] = fut_2d[:, i].astype(np.float32)
                future_df = pd.DataFrame(fut_data)

        t0 = time.perf_counter()

        pred_df = self._pipeline.predict_df(  # type: ignore[union-attr]
            ctx_df,
            future_df=future_df,
            prediction_length=horizon,
            quantile_levels=self.quantile_levels,
            id_column="id",
            timestamp_column="timestamp",
            target="target",
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Extract results from prediction DataFrame
        if "mean" in pred_df.columns:
            point = pred_df["mean"].values
        else:
            point = pred_df["0.5"].values

        quantile_lo = None
        quantile_hi = None
        quantiles_arr = None
        q_levels = None

        q_cols = [str(q) for q in self.quantile_levels]
        available_q = [c for c in q_cols if c in pred_df.columns]
        if available_q:
            quantiles_arr = np.array([pred_df[c].values for c in available_q])
            q_levels = np.array([float(c) for c in available_q])
            # Use first and last quantile as lo/hi bounds
            quantile_lo = quantiles_arr[0]
            quantile_hi = quantiles_arr[-1]

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
        """Estimated memory footprint."""
        if self._pipeline is None:
            return 0.0
        try:
            import torch

            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024 * 1024)
        except ImportError:
            pass
        # Rough estimates based on model size
        if "small" in self.model_id:
            return 120.0  # ~28M params * 4 bytes
        return 500.0  # ~120M params * 4 bytes
