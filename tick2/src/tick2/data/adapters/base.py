"""Base protocol for data adapters.

Each model has its own expected input format. Adapters convert from our
standard DataFrame representation to the model-specific format.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class DataAdapter(Protocol):
    """Protocol for model-specific data format converters."""

    def prepare_context(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: list[str] | None = None,
    ) -> dict[str, Any]:
        """Convert a DataFrame context window to model-specific input.

        Args:
            df: Context DataFrame with datetime index.
            target_col: Name of the target column.
            feature_cols: Covariate column names, or None for univariate.

        Returns:
            Dict with model-specific keys ready to pass to the wrapper.
        """
        ...
