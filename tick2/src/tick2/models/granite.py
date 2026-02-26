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

    For fine-tuned mix_channel models, set ``_n_input_channels`` to the
    number of channels the model expects (1 target + N covariates).
    This enables multivariate inference via :meth:`predict`.
    """

    model_id: str = "ibm-granite/granite-timeseries-ttm-r2"
    model_name: str = "granite-ttm"
    context_length: int = 1024
    prediction_length: int = 96
    _model: object = field(default=None, init=False, repr=False)
    _device: str = field(default="", init=False, repr=False)
    _n_input_channels: int = field(default=1, init=False, repr=False)

    @property
    def name(self) -> str:
        return self.model_name

    @property
    def supports_covariates(self) -> bool:
        """True when the loaded model expects multivariate (mix_channel) input."""
        return self._n_input_channels > 1

    @property
    def supports_quantiles(self) -> bool:
        return False

    def _get_branch(self) -> str:
        """Resolve the HuggingFace branch for the requested config."""
        key = (self.context_length, self.prediction_length)
        if key in ZERO_SHOT_BRANCHES:
            return ZERO_SHOT_BRANCHES[key]
        # Fallback: try to find any branch with matching prediction_length
        for (_ctx, pl), branch in ZERO_SHOT_BRANCHES.items():
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

        TTM expects input shape ``(batch, seq_len, n_channels)``.
        For zero-shot univariate: ``(1, context_length, 1)``.
        For mix_channel fine-tuned models: ``(1, context_length, 1 + n_cov)``
        where covariates are stacked alongside the target channel.
        """
        if self._model is None:
            raise RuntimeError("Call load() before predict()")

        import torch

        # Pad or truncate context to match model's expected length
        ctx = context[-self.context_length :]
        if len(ctx) < self.context_length:
            pad = np.zeros(self.context_length - len(ctx))
            ctx = np.concatenate([pad, ctx])

        # Build input tensor: (batch, seq_len, n_channels)
        n_cov_expected = self._n_input_channels - 1
        if covariates is not None and n_cov_expected > 0:
            cov = covariates[-self.context_length :]
            # Pad covariates if context was padded
            if len(cov) < self.context_length:
                cov_pad = np.zeros((self.context_length - len(cov), cov.shape[1]))
                cov = np.concatenate([cov_pad, cov], axis=0)
            # Ensure correct number of covariate channels
            if cov.shape[1] > n_cov_expected:
                cov = cov[:, :n_cov_expected]
            elif cov.shape[1] < n_cov_expected:
                extra = np.zeros((cov.shape[0], n_cov_expected - cov.shape[1]))
                cov = np.concatenate([cov, extra], axis=1)
            input_data = np.column_stack([ctx.reshape(-1, 1), cov])
        else:
            input_data = ctx.reshape(-1, 1)

        input_tensor = torch.tensor(
            input_data.reshape(1, input_data.shape[0], input_data.shape[1]),
            dtype=torch.float32,
            device=self._device,
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
