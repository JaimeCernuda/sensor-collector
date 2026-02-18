"""SHAP-based feature importance analysis for multivariate clock drift.

The primary deliverable is SHAP feature/category importance from a GBR
fitted on the entire 1-minute-aggregated dataset.  The detrended target
(30-min rolling mean subtracted) captures short-term drift fluctuations
driven by sensor covariates.

R² is reported only as an in-sample supplementary metric, since the
piecewise-constant NTP target defeats standard out-of-sample regression
evaluation.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

# Detrending: 30-row rolling mean on 1-minute data
DETREND_WINDOW = 30


@dataclass
class MachineResults:
    """All analysis results for a single machine."""

    machine: str
    feature_categories: dict[str, str] = field(default_factory=dict)
    # SHAP outputs
    shap_values: np.ndarray | None = None
    shap_feature_names: list[str] = field(default_factory=list)
    shap_category_importance: dict[str, float] = field(default_factory=dict)
    shap_feature_importance: dict[str, float] = field(default_factory=dict)
    # Feature-set cumulative SHAP (for Figure 2)
    feature_set_shap: dict[str, float] = field(default_factory=dict)
    # Supplementary in-sample R2
    insample_r2: dict[str, float] = field(default_factory=dict)
    # Progressive R2: K categories -> R2 (categories added in SHAP-descending order)
    progressive_r2: dict[int, float] = field(default_factory=dict)
    progressive_categories: list[str] = field(default_factory=list)


# -- Target preprocessing -----------------------------------------------------


def _detrend(series: pd.Series) -> pd.Series:
    """Remove slow NTP-servo trend via rolling-mean subtraction."""
    trend = series.rolling(DETREND_WINDOW, center=True, min_periods=5).mean()
    trend = trend.ffill().bfill()
    return series - trend


# -- Feature helpers -----------------------------------------------------------


def _get_features_by_categories(
    feature_categories: dict[str, str],
    target_categories: set[str],
) -> list[str]:
    """Return feature column names in *target_categories*."""
    return [col for col, cat in feature_categories.items() if cat in target_categories]


# -- Model specs for progressive comparison ------------------------------------

MODEL_SPECS: list[tuple[str, set[str]]] = [
    ("M1: CPU Core Temp", {"CPU Core Temp"}),
    ("M2: All Temps", {"CPU Core Temp", "CPU Package Temp", "Non-CPU Temp"}),
    ("M3: Non-CPU Temp", {"Non-CPU Temp"}),
    ("M4: All Features", set()),  # all categories
]


def _make_gbr() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_leaf=5,
        random_state=42,
    )


# -- Core analysis -------------------------------------------------------------


def analyze_machine(
    machine: str,
    df: pd.DataFrame,
    feature_categories: dict[str, str],
) -> MachineResults:
    """Run full analysis pipeline for a single machine.

    1. Detrend target.
    2. Fit GBR on full feature set -> SHAP values.
    3. Compute cumulative SHAP for each progressive feature set (M1-M4).
    4. Compute in-sample R2 for each feature set (supplementary).
    """
    print(f"\nAnalyzing {machine}...")

    results = MachineResults(machine=machine, feature_categories=feature_categories)

    # Prepare data
    df = df.copy()
    df["adj_freq_ppm"] = _detrend(df["adj_freq_ppm"])
    df = df.dropna(subset=["adj_freq_ppm"]).reset_index(drop=True)

    feature_cols = [
        c for c in df.columns if c != "adj_freq_ppm" and c in feature_categories
    ]
    features = df[feature_cols]
    target = df["adj_freq_ppm"]

    # -- SHAP analysis on full model -------------------------------------------
    print("  Fitting full GBR + SHAP...")
    model = _make_gbr()
    model.fit(features, target)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)

    mean_abs_shap: np.ndarray = np.abs(shap_values).mean(axis=0)
    results.shap_values = shap_values
    results.shap_feature_names = feature_cols
    results.shap_feature_importance = {
        col: float(v) for col, v in zip(feature_cols, mean_abs_shap, strict=True)
    }

    # Per-category importance
    cat_imp: dict[str, float] = {}
    for col, shap_val in zip(feature_cols, mean_abs_shap, strict=True):
        cat = feature_categories[col]
        cat_imp[cat] = cat_imp.get(cat, 0.0) + float(shap_val)
    results.shap_category_importance = cat_imp

    # -- Feature-set cumulative SHAP (for progressive comparison) ---------------
    print("  Computing feature-set contributions...")
    all_categories = set(feature_categories.values())
    for name, categories in MODEL_SPECS:
        cats = categories if categories else all_categories
        subset_cols = _get_features_by_categories(feature_categories, cats)
        available = [c for c in subset_cols if c in results.shap_feature_importance]
        total_shap = sum(results.shap_feature_importance[c] for c in available)
        results.feature_set_shap[name] = total_shap
        print(f"    {name}: sum |SHAP| = {total_shap:.4f}  ({len(available)} features)")

    # -- In-sample R2 (supplementary) ------------------------------------------
    print("  In-sample R2:")
    for name, categories in MODEL_SPECS:
        cats = categories if categories else all_categories
        subset_cols = _get_features_by_categories(feature_categories, cats)
        available = [c for c in subset_cols if c in df.columns]
        if not available:
            results.insample_r2[name] = 0.0
            continue
        m = _make_gbr()
        m.fit(df[available], target)
        r2 = float(r2_score(target, m.predict(df[available])))
        results.insample_r2[name] = r2
        print(f"    {name}: R2={r2:.4f}")

    # -- Progressive R2 (add categories in SHAP-descending order) ----------------
    print("  Progressive R2 (cumulative categories):")
    ordered_cats = sorted(cat_imp.items(), key=lambda kv: kv[1], reverse=True)
    ordered_cat_names = [cat for cat, _ in ordered_cats]
    results.progressive_categories = ordered_cat_names

    for k in range(1, len(ordered_cat_names) + 1):
        top_k = set(ordered_cat_names[:k])
        subset = _get_features_by_categories(feature_categories, top_k)
        available = [c for c in subset if c in df.columns]
        if not available:
            results.progressive_r2[k] = 0.0
            continue
        m = _make_gbr()
        m.fit(df[available], target)
        r2 = float(r2_score(target, m.predict(df[available])))
        results.progressive_r2[k] = r2
        cats_str = ", ".join(ordered_cat_names[:k])
        print(f"    K={k} ({cats_str}): R2={r2:.4f}")

    return results
