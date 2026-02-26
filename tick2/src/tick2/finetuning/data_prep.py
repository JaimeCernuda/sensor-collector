"""Shared data preparation for fine-tuning pipelines.

Wraps existing tick2 data infrastructure (load_all, temporal_split,
get_feature_cols) and adds feature selection with category diversity
enforcement.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tick2.data.preprocessing import (
    TARGET_COL,
    load_all,
)
from tick2.data.splits import TemporalSplit, temporal_split
from tick2.finetuning.base import FineTuneConfig

# Categories we want to ensure representation of in feature selection.
PRIORITY_CATEGORIES = [
    "CPU Core Temp",
    "CPU Package Temp",
    "CPU Frequency",
    "Power",
    "C-State",
    "Memory",
    "CPU Load",
]


@dataclass
class PreparedData:
    """Preprocessed data for a single machine, ready for fine-tuning.

    Attributes:
        name: Machine name.
        split: Temporal train/val/test split.
        feature_cols: Selected feature column names.
        categories: Mapping of feature column -> category.
    """

    name: str
    split: TemporalSplit
    feature_cols: list[str]
    categories: dict[str, str]


def select_top_features(
    df: pd.DataFrame,
    categories: dict[str, str],
    max_features: int = 30,
) -> list[str]:
    """Select top features by correlation with target, enforcing category diversity.

    Ensures each priority category (temp, freq, power, cstate, etc.) has at
    least one representative before filling remaining slots by absolute
    correlation ranking.

    Args:
        df: DataFrame with target and feature columns.
        categories: Mapping of feature column -> category name.
        max_features: Maximum number of features to select.

    Returns:
        List of selected feature column names.
    """
    feature_cols = [c for c in df.columns if c != TARGET_COL and c in categories]
    if len(feature_cols) <= max_features:
        return feature_cols

    # Compute absolute correlation with target
    correlations: dict[str, float] = {}
    for col in feature_cols:
        corr = df[col].corr(df[TARGET_COL])
        correlations[col] = abs(corr) if pd.notna(corr) else 0.0

    # Group features by category
    cat_groups: dict[str, list[str]] = {}
    for col in feature_cols:
        cat = categories.get(col, "Other")
        cat_groups.setdefault(cat, []).append(col)

    # Sort each group by correlation (descending)
    for cat in cat_groups:
        cat_groups[cat].sort(key=lambda c: correlations.get(c, 0.0), reverse=True)

    selected: list[str] = []
    used: set[str] = set()

    # Phase 1: Guarantee one representative from each priority category
    for cat in PRIORITY_CATEGORIES:
        if cat in cat_groups and len(selected) < max_features:
            best = cat_groups[cat][0]
            if best not in used:
                selected.append(best)
                used.add(best)

    # Phase 2: Fill remaining slots by global correlation ranking
    all_ranked = sorted(
        feature_cols, key=lambda c: correlations.get(c, 0.0), reverse=True
    )
    for col in all_ranked:
        if len(selected) >= max_features:
            break
        if col not in used:
            selected.append(col)
            used.add(col)

    return selected


def prepare_datasets(
    config: FineTuneConfig,
    data_dir: Path | None = None,
    snapshot: str = "24h_snapshot",
) -> dict[str, PreparedData]:
    """Load all machines and prepare temporal splits with feature selection.

    Args:
        config: Fine-tuning configuration.
        data_dir: Override data directory (None = auto-detect).
        snapshot: Which snapshot to load.

    Returns:
        Dict mapping machine name -> PreparedData.
    """
    raw = load_all(data_dir=data_dir, snapshot=snapshot)
    result: dict[str, PreparedData] = {}

    for name, (df, cats) in raw.items():
        features = select_top_features(df, cats, max_features=config.max_covariates)
        split = temporal_split(df, config.train_frac, config.val_frac, config.test_frac)
        result[name] = PreparedData(
            name=name,
            split=split,
            feature_cols=features,
            categories={c: cats[c] for c in features if c in cats},
        )

    return result


def combine_training_data(
    prepared: dict[str, PreparedData],
) -> tuple[pd.DataFrame, list[str]]:
    """Concatenate training splits from all machines for combined fine-tuning.

    Uses the intersection of feature columns across all machines to handle
    differing sensor availability.

    Args:
        prepared: Dict of machine name -> PreparedData.

    Returns:
        Tuple of (combined training DataFrame, shared feature columns).
    """
    # Find feature columns common to all machines
    feature_sets = [set(p.feature_cols) for p in prepared.values()]
    shared_features = sorted(set.intersection(*feature_sets)) if feature_sets else []

    keep_cols = [*shared_features, TARGET_COL]
    train_dfs = []
    for p in prepared.values():
        train_dfs.append(p.split.train[keep_cols].copy())

    combined = pd.concat(train_dfs, ignore_index=True)
    return combined, shared_features


def combine_validation_data(
    prepared: dict[str, PreparedData],
) -> tuple[pd.DataFrame, list[str]]:
    """Concatenate validation splits from all machines.

    Args:
        prepared: Dict of machine name -> PreparedData.

    Returns:
        Tuple of (combined validation DataFrame, shared feature columns).
    """
    feature_sets = [set(p.feature_cols) for p in prepared.values()]
    shared_features = sorted(set.intersection(*feature_sets)) if feature_sets else []

    keep_cols = [*shared_features, TARGET_COL]
    val_dfs = []
    for p in prepared.values():
        val_dfs.append(p.split.val[keep_cols].copy())

    combined = pd.concat(val_dfs, ignore_index=True)
    return combined, shared_features


def get_machine_arrays(
    data: PreparedData,
    split_name: str = "train",
) -> tuple[np.ndarray, np.ndarray | None]:
    """Extract target and covariate numpy arrays from a PreparedData split.

    Args:
        data: PreparedData for one machine.
        split_name: Which split ("train", "val", "test").

    Returns:
        Tuple of (target 1D array, covariates 2D array or None).
    """
    df = getattr(data.split, split_name)
    target = df[TARGET_COL].to_numpy(dtype=np.float64)

    covariates = None
    if data.feature_cols:
        available = [c for c in data.feature_cols if c in df.columns]
        if available:
            covariates = df[available].to_numpy(dtype=np.float64)

    return target, covariates
