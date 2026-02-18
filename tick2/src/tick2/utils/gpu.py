"""GPU memory management utilities."""

from __future__ import annotations


def clear_gpu_memory() -> None:
    """Clear GPU cache and run garbage collection.

    Useful between model evaluations to prevent OOM errors on Colab T4.
    """
    import gc

    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def get_vram_gb() -> float | None:
    """Return total GPU VRAM in GB, or None if no GPU."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_mem / (1024**3)
    except ImportError:
        pass
    return None


def fits_in_vram(model_name: str) -> bool:
    """Check if a model likely fits in available VRAM.

    Args:
        model_name: Registry model name.

    Returns:
        True if model should fit, False if VRAM is insufficient.
    """
    vram = get_vram_gb()
    if vram is None:
        return False

    # Approximate VRAM requirements (inference only)
    requirements: dict[str, float] = {
        "chronos2-small": 1.0,
        "chronos2-base": 2.0,
        "timesfm-2.5": 3.0,
        "moirai-1.1-small": 1.0,
        "moirai-1.1-base": 2.0,
        "moirai-1.1-large": 4.0,
        "moirai-2.0-small": 1.0,
        "toto": 3.0,
        "granite-ttm": 0.5,
    }

    required = requirements.get(model_name, 4.0)
    return vram >= required * 1.2  # 20% headroom
