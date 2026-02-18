"""Inference timing utilities for CPU and GPU profiling.

Provides warm-up, repeated timing, and memory tracking for benchmark runs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class TimingResult:
    """Result of a timing profile run."""

    model_name: str
    device: str
    context_length: int
    horizon: int
    n_warmup: int
    n_repeats: int
    times_ms: list[float]

    @property
    def mean_ms(self) -> float:
        return float(np.mean(self.times_ms))

    @property
    def std_ms(self) -> float:
        return float(np.std(self.times_ms))

    @property
    def median_ms(self) -> float:
        return float(np.median(self.times_ms))

    @property
    def p95_ms(self) -> float:
        return float(np.percentile(self.times_ms, 95))


def profile_inference(
    model: object,
    context: NDArray[np.floating],
    horizon: int,
    n_warmup: int = 3,
    n_repeats: int = 10,
) -> TimingResult:
    """Profile inference timing with warm-up and repeated runs.

    Args:
        model: A loaded ModelWrapper instance.
        context: Historical values, shape (context_len,).
        horizon: Number of steps to predict.
        n_warmup: Number of warm-up runs (not timed).
        n_repeats: Number of timed runs.

    Returns:
        TimingResult with per-run timings.
    """
    from tick2.models.base import ModelWrapper

    assert isinstance(model, ModelWrapper), f"Expected ModelWrapper, got {type(model)}"

    # Warm up (primes CUDA kernels, JIT compilation, etc.)
    for _ in range(n_warmup):
        model.predict(context, horizon)

    # Timed runs
    times: list[float] = []
    for _ in range(n_repeats):
        # Synchronize GPU before timing if using CUDA
        _sync_cuda()

        t0 = time.perf_counter()
        model.predict(context, horizon)
        _sync_cuda()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        times.append(elapsed_ms)

    return TimingResult(
        model_name=model.name,
        device=getattr(model, "_device", "unknown"),
        context_length=len(context),
        horizon=horizon,
        n_warmup=n_warmup,
        n_repeats=n_repeats,
        times_ms=times,
    )


def get_gpu_memory_mb() -> float | None:
    """Return current GPU memory usage in MB, or None if no GPU."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
    except ImportError:
        pass
    return None


def get_peak_gpu_memory_mb() -> float | None:
    """Return peak GPU memory usage in MB, or None if no GPU."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except ImportError:
        pass
    return None


def reset_gpu_memory_stats() -> None:
    """Reset GPU memory tracking statistics."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def _sync_cuda() -> None:
    """Synchronize CUDA if available (ensures GPU ops are complete)."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass
