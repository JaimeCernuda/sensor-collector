"""Moirai model wrapper (Salesforce, 14M-311M).

Requires: pip install "uni2ts>=2.0"
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from tick2.models.base import PredictionResult


@dataclass
class MoiraiWrapper:
    """Wrapper for Moirai 1.1/2.0 zero-shot inference.

    Uses the uni2ts MoiraiForecast API. Supports any number of variates
    via Any-Variate Attention (variable channel count without schema changes).
    """

    model_id: str = "Salesforce/moirai-1.1-R-small"
    model_name: str = "moirai-1.1-small"
    patch_size: int | str = "auto"
    num_samples: int = 100
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
        """Load Moirai model weights."""
        import torch

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        # Store model_id for use in predict(); Moirai instantiation
        # happens per-call because context/horizon must be known at init time.
        # We do a dry-run import to verify the package is installed.
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule  # noqa: F401

        # Pre-download weights
        self._model = MoiraiModule.from_pretrained(self.model_id)

    def predict(
        self,
        context: NDArray[np.floating],
        horizon: int,
        covariates: NDArray[np.floating] | None = None,
        future_covariates: NDArray[np.floating] | None = None,
    ) -> PredictionResult:
        """Generate a forecast using Moirai.

        Moirai treats all channels as equal variates. The target is the
        first channel; covariates become additional channels.
        """
        if self._model is None:
            raise RuntimeError("Call load() before predict()")

        import torch
        from uni2ts.model.moirai import MoiraiForecast

        # Build multivariate input: (n_variates, seq_len)
        if covariates is not None:
            cov_2d = covariates.reshape(len(context), -1)
            # Target as first channel, covariates as subsequent channels
            data = np.column_stack([context, cov_2d])  # (seq_len, n_variates)
        else:
            data = context.reshape(-1, 1)  # (seq_len, 1)

        n_variates = data.shape[1]
        ctx_len = data.shape[0]

        # Determine patch size
        ps = self.patch_size
        if ps == "auto":
            ps = 32

        # Create the MoiraiForecast inference wrapper
        n_cov = n_variates - 1 if n_variates > 1 else 0
        predictor = MoiraiForecast(
            module=self._model,
            prediction_length=horizon,
            context_length=ctx_len,
            patch_size=ps,
            num_samples=self.num_samples,
            target_dim=1,  # predict only the target (first channel)
            feat_dynamic_real_dim=n_cov,
            past_feat_dynamic_real_dim=n_cov,
        )
        predictor = predictor.to(self._device)

        # Prepare tensors — uni2ts 2.0 uses channels-last: (batch, time, dim)
        target_tensor = torch.tensor(
            data[:, 0:1], dtype=torch.float32, device=self._device
        ).unsqueeze(0)  # (1, ctx_len, 1)

        past_feat = None
        if n_variates > 1:
            past_feat = torch.tensor(
                data[:, 1:], dtype=torch.float32, device=self._device
            ).unsqueeze(0)  # (1, ctx_len, n_cov)

        t0 = time.perf_counter()

        # Build observed mask for covariates (required when past_feat_dynamic_real is set)
        past_observed_feat = None
        if past_feat is not None:
            past_observed_feat = torch.ones_like(past_feat, dtype=torch.bool)

        # Generate samples
        with torch.no_grad():
            samples = predictor(
                past_target=target_tensor,
                past_observed_target=torch.ones_like(target_tensor, dtype=torch.bool),
                past_is_pad=torch.zeros(
                    1, ctx_len, dtype=torch.bool, device=self._device
                ),
                past_feat_dynamic_real=past_feat,
                past_observed_feat_dynamic_real=past_observed_feat,
            )  # (1, num_samples, horizon, tgt)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        samples_np = samples[0].cpu().numpy()  # (num_samples, horizon)

        # Derive point forecast and quantiles from samples
        point = np.median(samples_np, axis=0)
        q_levels = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        quantiles_arr = np.quantile(samples_np, q_levels, axis=0)  # (5, horizon)

        return PredictionResult(
            point_forecast=point,
            quantile_lo=quantiles_arr[0],  # 10th percentile
            quantile_hi=quantiles_arr[-1],  # 90th percentile
            quantiles=quantiles_arr,
            quantile_levels=q_levels,
            inference_time_ms=elapsed_ms,
            model_name=self.model_name,
            device=self._device,
        )

    def memory_footprint_mb(self) -> float:
        """Estimated memory footprint."""
        if self._model is None:
            return 0.0
        size_map = {
            "small": 60.0,
            "base": 370.0,
            "large": 1250.0,
        }
        for key, mb in size_map.items():
            if key in self.model_id:
                return mb
        return 60.0
