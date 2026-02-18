"""Common adapter utility: extract arrays from DataFrame.

All model adapters need the same basic extraction step: pull the target
column and optional covariate columns from a DataFrame as numpy arrays.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def extract_arrays(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str] | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating] | None]:
    """Extract target and covariate arrays from a DataFrame.

    Args:
        df: Input DataFrame.
        target_col: Column name for the prediction target.
        feature_cols: Covariate column names, or None for univariate.

    Returns:
        Tuple of (target_array shape (n,), covariates shape (n, n_features) or None).
    """
    target = df[target_col].to_numpy(dtype=np.float64)

    covariates = None
    if feature_cols:
        available = [c for c in feature_cols if c in df.columns]
        if available:
            covariates = df[available].to_numpy(dtype=np.float64)

    return target, covariates
