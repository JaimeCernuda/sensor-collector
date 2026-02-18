"""Toto model wrapper (Datadog, 151M).

Requires: pip install toto-ts
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from tick2.models.base import PredictionResult


@dataclass
class TotoWrapper:
    """Wrapper for Toto zero-shot inference.

    Toto is natively multivariate: all channels are treated as equal
    variates. The target channel is the first one; all others are covariates.
    Outputs are sample-based (MixtureOfStudentT distribution).
    """

    model_id: str = "Datadog/Toto-Open-Base-1.0"
    model_name: str = "toto"
    num_samples: int = 256
    _model: object = field(default=None, init=False, repr=False)
    _forecaster: object = field(default=None, init=False, repr=False)
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
        """Load Toto model weights."""
        import torch
        from toto.inference.forecaster import TotoForecaster
        from toto.model.toto import Toto

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        toto = Toto.from_pretrained(self.model_id)
        toto.to(device)
        self._model = toto
        self._forecaster = TotoForecaster(toto.model)
        self._device = device

    def predict(
        self,
        context: NDArray[np.floating],
        horizon: int,
        covariates: NDArray[np.floating] | None = None,
        future_covariates: NDArray[np.floating] | None = None,
    ) -> PredictionResult:
        """Generate a forecast using Toto.

        Toto uses MaskedTimeseries as input, treating all channels as
        equal variates. Produces sample-based probabilistic forecasts.
        """
        if self._forecaster is None:
            raise RuntimeError("Call load() before predict()")

        import torch
        from toto.data.util.dataset import MaskedTimeseries

        ctx_len = len(context)

        # Build multivariate series: (n_channels, seq_len)
        if covariates is not None:
            cov_2d = covariates.reshape(ctx_len, -1)
            series_data = np.column_stack([context, cov_2d]).T  # (n_channels, seq_len)
        else:
            series_data = context.reshape(1, -1)  # (1, seq_len)

        n_channels = series_data.shape[0]

        series = torch.tensor(series_data, dtype=torch.float32, device=self._device)
        padding_mask = torch.ones(
            n_channels, ctx_len, dtype=torch.bool, device=self._device
        )
        id_mask = torch.zeros(
            n_channels, ctx_len, dtype=torch.float32, device=self._device
        )
        timestamp_seconds = torch.zeros(
            n_channels, ctx_len, dtype=torch.float32, device=self._device
        )
        time_interval = torch.ones(n_channels, dtype=torch.float32, device=self._device)

        inputs = MaskedTimeseries(
            series=series,
            padding_mask=padding_mask,
            id_mask=id_mask,
            timestamp_seconds=timestamp_seconds,
            time_interval_seconds=time_interval,
        )

        t0 = time.perf_counter()

        with torch.no_grad():
            forecast = self._forecaster.forecast(  # type: ignore[union-attr]
                inputs,
                prediction_length=horizon,
                num_samples=self.num_samples,
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Extract target channel (index 0) samples
        # forecast.samples shape: (num_samples, n_channels, horizon)
        samples_np = forecast.samples[:, 0, :].cpu().numpy()  # (num_samples, horizon)

        point = np.median(samples_np, axis=0)
        q_levels = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        quantiles_arr = np.quantile(samples_np, q_levels, axis=0)

        return PredictionResult(
            point_forecast=point,
            quantile_lo=quantiles_arr[0],
            quantile_hi=quantiles_arr[-1],
            quantiles=quantiles_arr,
            quantile_levels=q_levels,
            inference_time_ms=elapsed_ms,
            model_name=self.model_name,
            device=self._device,
        )

    def memory_footprint_mb(self) -> float:
        """Estimated memory: ~151M params * 4 bytes ~ 600 MB."""
        if self._model is None:
            return 0.0
        return 600.0
