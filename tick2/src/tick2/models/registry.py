"""Model registry: name -> wrapper class lookup."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tick2.models.base import ModelWrapper

# Lazy imports to avoid pulling in heavy model deps at import time.
_REGISTRY: dict[str, tuple[str, str]] = {
    # name -> (module_path, class_name)
    "chronos2-small": ("tick2.models.chronos2", "Chronos2Wrapper"),
    "chronos2-base": ("tick2.models.chronos2", "Chronos2Wrapper"),
    "timesfm-2.5": ("tick2.models.timesfm", "TimesFMWrapper"),
    "moirai-2.0-small": ("tick2.models.moirai", "MoiraiWrapper"),
    "moirai-1.1-small": ("tick2.models.moirai", "MoiraiWrapper"),
    "moirai-1.1-base": ("tick2.models.moirai", "MoiraiWrapper"),
    "moirai-1.1-large": ("tick2.models.moirai", "MoiraiWrapper"),
    "toto": ("tick2.models.toto", "TotoWrapper"),
    "granite-ttm": ("tick2.models.granite", "GraniteTTMWrapper"),
}

# Maps short names to HuggingFace model IDs
MODEL_IDS: dict[str, str] = {
    "chronos2-small": "autogluon/chronos-2-small",
    "chronos2-base": "amazon/chronos-2",
    "timesfm-2.5": "google/timesfm-2.5-200m-pytorch",
    "moirai-2.0-small": "Salesforce/moirai-2.0-R-small",
    "moirai-1.1-small": "Salesforce/moirai-1.1-R-small",
    "moirai-1.1-base": "Salesforce/moirai-1.1-R-base",
    "moirai-1.1-large": "Salesforce/moirai-1.1-R-large",
    "toto": "Datadog/Toto-Open-Base-1.0",
    "granite-ttm": "ibm-granite/granite-timeseries-ttm-r2",
}


def list_models() -> list[str]:
    """Return all registered model names."""
    return sorted(_REGISTRY.keys())


def get_model(name: str, **kwargs: object) -> ModelWrapper:
    """Instantiate a model wrapper by name.

    Args:
        name: Registered model name (e.g., "chronos2-small").
        **kwargs: Passed to the wrapper constructor.

    Returns:
        A ModelWrapper instance (not yet loaded).
    """
    if name not in _REGISTRY:
        available = ", ".join(list_models())
        raise ValueError(f"Unknown model {name!r}. Available: {available}")

    module_path, class_name = _REGISTRY[name]

    import importlib

    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    # Inject model_id if not explicitly provided
    if "model_id" not in kwargs:
        kwargs["model_id"] = MODEL_IDS[name]
    if "model_name" not in kwargs:
        kwargs["model_name"] = name

    return cls(**kwargs)  # type: ignore[no-any-return]
