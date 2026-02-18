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

# Branch naming scheme for granite-timeseries-ttm-r2:
# {context_length}-{prediction_length}[-ft][-mae]-r2[.1]
DEFAULT_BRANCHES: dict[tuple[int, int], str] = {
    (512, 96): "512-96-r2",
    (1024, 96): "1024-96-r2",
    (1536, 96): "1536-96-r2",
    (512, 192): "512-192-r2",
    (512, 336): "512-336-r2",
    (512, 720): "512-720-r2",
}


@dataclass
class GraniteTTMWrapper:
    """Wrapper for Granite TTM (TinyTimeMixer) zero-shot inference.

    TTM is uniquely small (1-5M params) and CPU-capable. It produces
    point forecasts only (no native probabilistic output).
    """

    model_id: str = "ibm-granite/granite-timeseries-ttm-r2"
    model_name: str = "granite-ttm"
    context_length: int = 512
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
        if key in DEFAULT_BRANCHES:
            return DEFAULT_BRANCHES[key]
        # Fallback to closest match
        return f"{self.context_length}-{self.prediction_length}-r2"

    def load(self, device: str = "auto") -> None:
        """Load Granite TTM model weights."""
        import torch

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        from tsfm_public.toolkit.get_model import get_model

        self._model = get_model(
            model_path=self.model_id,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
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

        # Shape: (1, 1, context_length) for univariate
        input_tensor = torch.tensor(
            ctx.reshape(1, 1, -1), dtype=torch.float32, device=self._device
        )

        t0 = time.perf_counter()

        with torch.no_grad():
            output = self._model(input_tensor)  # type: ignore[misc]

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Output shape: (1, 1, prediction_length)
        pred = output.prediction_outputs  # type: ignore[union-attr]
        if hasattr(pred, "cpu"):
            pred_np = pred[0, 0, :].cpu().numpy()
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
