"""Granite TTM r2.1 model wrapper (IBM, 1-5M).

Requires: pip install "granite-tsfm>=0.3.3"
Note: import as ``tsfm_public`` (not ``granite_tsfm``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from tick2.models.base import PredictionResult

# Zero-shot branches for granite-timeseries-ttm-r2.
# Note: 512-96 only exists as fine-tuned (512-96-ft-r2.1), not zero-shot.
# Shortest zero-shot context for 96-step horizon is 1024.
ZERO_SHOT_BRANCHES: dict[tuple[int, int], str] = {
    (1024, 96): "1024-96-r2",
    (1536, 96): "1536-96-r2",
    (512, 192): "512-192-r2",
    (1024, 192): "1024-192-r2",
    (1536, 192): "1536-192-r2",
    (512, 336): "512-336-r2",
    (1024, 336): "1024-336-r2",
    (1536, 336): "1536-336-r2",
    (512, 720): "512-720-r2",
    (1024, 720): "1024-720-r2",
    (1536, 720): "1536-720-r2",
}


@dataclass
class GraniteTTMWrapper:
    """Wrapper for Granite TTM (TinyTimeMixer) zero-shot inference.

    TTM is uniquely small (1-5M params) and CPU-capable. It produces
    point forecasts only (no native probabilistic output).
    """

    model_id: str = "ibm-granite/granite-timeseries-ttm-r2"
    model_name: str = "granite-ttm"
    context_length: int = 1024
    prediction_length: int = 96
    _model: object = field(default=None, init=False, repr=False)
    _device: str = field(default="", init=False, repr=False)

    @property
    def name(self) -> str:
        return self.model_name

    @property
    def supports_covariates(self) -> bool:
        # Channel-mix fine-tuning enables covariates, but zero-shot is univariate
        return False

    @property
    def supports_quantiles(self) -> bool:
        return False

    def _get_branch(self) -> str:
        """Resolve the HuggingFace branch for the requested config."""
        key = (self.context_length, self.prediction_length)
        if key in ZERO_SHOT_BRANCHES:
            return ZERO_SHOT_BRANCHES[key]
        # Fallback: try to find any branch with matching prediction_length
        for (ctx, pl), branch in ZERO_SHOT_BRANCHES.items():
            if pl == self.prediction_length:
                return branch
        raise ValueError(
            f"No zero-shot branch for ctx={self.context_length}, "
            f"pl={self.prediction_length}. Available: {list(ZERO_SHOT_BRANCHES.keys())}"
        )

    def load(self, device: str = "auto") -> None:
        """Load Granite TTM model weights."""
        import torch

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction

        self._model = TinyTimeMixerForPrediction.from_pretrained(
            self.model_id,
            revision=self._get_branch(),
        )
        self._model.to(device)  # type: ignore[union-attr]
        self._model.eval()  # type: ignore[union-attr]

    def predict(
        self,
        context: NDArray[np.floating],
        horizon: int,
        covariates: NDArray[np.floating] | None = None,
        future_covariates: NDArray[np.floating] | None = None,
    ) -> PredictionResult:
        """Generate a point forecast using Granite TTM.

        TTM expects input shape (batch, n_channels, seq_len). For zero-shot
        univariate: (1, 1, context_length). Covariates are ignored in
        zero-shot mode (use channel-mix fine-tuning to enable).
        """
        if self._model is None:
            raise RuntimeError("Call load() before predict()")

        import torch

        # Pad or truncate context to match model's expected length
        ctx = context[-self.context_length :]
        if len(ctx) < self.context_length:
            pad = np.zeros(self.context_length - len(ctx))
            ctx = np.concatenate([pad, ctx])

        # Shape: (batch, seq_len, n_channels) — TTM uses channels-last
        input_tensor = torch.tensor(
            ctx.reshape(1, -1, 1), dtype=torch.float32, device=self._device
        )

        t0 = time.perf_counter()

        with torch.no_grad():
            output = self._model(input_tensor)  # type: ignore[misc]

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Output shape: (1, prediction_length, n_channels)
        pred = output.prediction_outputs  # type: ignore[union-attr]
        if hasattr(pred, "cpu"):
            pred_np = pred[0, :, 0].cpu().numpy()
        else:
            pred_np = np.array(pred).flatten()

        # Truncate or pad to requested horizon
        if len(pred_np) >= horizon:
            point = pred_np[:horizon]
        else:
            # Model can only predict up to prediction_length; pad with last value
            point = np.pad(
                pred_np,
                (0, horizon - len(pred_np)),
                mode="edge",
            )

        return PredictionResult(
            point_forecast=point.astype(np.float64),
            inference_time_ms=elapsed_ms,
            model_name=self.model_name,
            device=self._device,
        )

    def memory_footprint_mb(self) -> float:
        """Estimated memory: 1-5M params * 4 bytes ~ 4-20 MB."""
        if self._model is None:
            return 0.0
        return 20.0
