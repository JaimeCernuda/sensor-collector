"""Evaluation of fine-tuned models against zero-shot baselines.

Uses the same metrics and sample extraction as the zero-shot benchmark
(notebook 02) for direct comparison.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from tick2.benchmark import metrics
from tick2.data.adapters.common import extract_arrays
from tick2.data.preprocessing import TARGET_COL
from tick2.data.splits import extract_samples
from tick2.finetuning.base import FineTuneConfig
from tick2.finetuning.data_prep import PreparedData
from tick2.models.base import ModelWrapper

logger = logging.getLogger(__name__)

# Default evaluation horizons and context lengths matching notebook 02
DEFAULT_CONTEXT_LENGTHS = [512, 1024]
DEFAULT_HORIZONS = [60, 120, 300]


@dataclass
class EvalResult:
    """Result of evaluating one model on one machine at one config."""

    model: str
    machine: str
    context_length: int
    horizon: int
    with_covariates: bool
    device: str
    n_samples: int
    mae: float
    rmse: float
    inference_ms: float
    coverage: float | None = None
    crps: float | None = None
    winkler: float | None = None
    training_mode: str = "zero_shot"
    ft_epochs: int | None = None
    ft_time_s: float | None = None
    ft_train_machines: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Flatten to a dict for DataFrame construction."""
        return {
            "model": self.model,
            "machine": self.machine,
            "context_length": self.context_length,
            "horizon": self.horizon,
            "with_covariates": self.with_covariates,
            "device": self.device,
            "n_samples": self.n_samples,
            "mae": self.mae,
            "rmse": self.rmse,
            "inference_ms": self.inference_ms,
            "coverage": self.coverage,
            "crps": self.crps,
            "winkler": self.winkler,
            "training_mode": self.training_mode,
            "ft_epochs": self.ft_epochs,
            "ft_time_s": self.ft_time_s,
            "ft_train_machines": self.ft_train_machines,
        }


def evaluate_finetuned(
    model: ModelWrapper,
    prepared: dict[str, PreparedData],
    config: FineTuneConfig,
    training_mode: str = "ft_combined",
    ft_epochs: int | None = None,
    ft_time_s: float | None = None,
    ft_train_machines: str = "",
    context_lengths: list[int] | None = None,
    horizons: list[int] | None = None,
    n_samples: int = 25,
    quantile_alpha: float = 0.2,
    progress: bool = True,
) -> pd.DataFrame:
    """Evaluate a fine-tuned model on test splits.

    Uses the same methodology as notebook 02.

    Runs inference on random windows from each machine's test split, computing
    the same metrics (MAE, RMSE, CRPS, coverage, Winkler) for direct comparison
    with zero-shot results.

    Args:
        model: A loaded ModelWrapper instance (fine-tuned).
        prepared: Dict of machine name -> PreparedData.
        config: Fine-tuning configuration (for seed, prediction_length).
        training_mode: Label for the results ("ft_combined", "ft_per_machine").
        ft_epochs: Number of fine-tuning epochs (metadata).
        ft_time_s: Fine-tuning wall-clock time (metadata).
        ft_train_machines: Which machines were in the training set (metadata).
        context_lengths: Context lengths to evaluate (default: [512, 1024]).
        horizons: Prediction horizons to evaluate (default: [60, 120, 300]).
        n_samples: Number of random windows per config.
        quantile_alpha: Alpha for Winkler score (0.2 = 80% interval).
        progress: Show tqdm progress bar.

    Returns:
        DataFrame with one row per (machine, context_length, horizon, cov_mode)
        combination, matching the zero-shot results schema plus FT metadata.
    """
    if context_lengths is None:
        context_lengths = DEFAULT_CONTEXT_LENGTHS
    if horizons is None:
        horizons = DEFAULT_HORIZONS

    use_covariates_modes = [False]
    if model.supports_covariates:
        use_covariates_modes.append(True)

    combos = [
        (machine, ctx_len, hz, use_cov)
        for machine in prepared
        for ctx_len in context_lengths
        for hz in horizons
        for use_cov in use_covariates_modes
    ]

    results: list[EvalResult] = []
    iterator = tqdm(combos, desc=f"Eval {model.name}", disable=not progress)

    for machine, ctx_len, hz, use_cov in iterator:
        prep = prepared[machine]
        test_df = prep.split.test
        feature_cols = prep.feature_cols if use_cov else None

        try:
            samples = extract_samples(
                test_df,
                target_col=TARGET_COL,
                context_len=ctx_len,
                horizon_len=hz,
                n_samples=n_samples,
                seed=config.seed,
            )
        except ValueError as e:
            logger.warning(
                "Skipping %s/%s ctx=%d hz=%d: %s",
                model.name,
                machine,
                ctx_len,
                hz,
                e,
            )
            continue

        sample_maes: list[float] = []
        sample_rmses: list[float] = []
        sample_times: list[float] = []
        sample_coverages: list[float] = []
        sample_crps: list[float] = []
        sample_winklers: list[float] = []

        for sample in samples:
            try:
                target, covariates = extract_arrays(
                    sample.context, TARGET_COL, feature_cols
                )
                y_true = sample.horizon_true.to_numpy(dtype=np.float64)

                cov_arg = covariates if use_cov else None
                fut_cov_arg = None
                if cov_arg is not None and feature_cols:
                    fut_cov_arg = sample.horizon_df[feature_cols].to_numpy(
                        dtype=np.float64
                    )

                pred = model.predict(
                    context=target,
                    horizon=sample.horizon_len,
                    covariates=cov_arg,
                    future_covariates=fut_cov_arg,
                )

                point = pred.point_forecast[: len(y_true)]
                sample_maes.append(metrics.mae(y_true, point))
                sample_rmses.append(metrics.rmse(y_true, point))
                sample_times.append(pred.inference_time_ms)

                if pred.quantile_lo is not None and pred.quantile_hi is not None:
                    lo = pred.quantile_lo[: len(y_true)]
                    hi = pred.quantile_hi[: len(y_true)]
                    sample_coverages.append(metrics.coverage(y_true, lo, hi))
                    sample_winklers.append(
                        metrics.winkler_score(y_true, lo, hi, alpha=quantile_alpha)
                    )

                if pred.quantiles is not None and pred.quantile_levels is not None:
                    sample_crps.append(
                        metrics.crps_quantile(
                            y_true,
                            pred.quantiles[:, : len(y_true)],
                            pred.quantile_levels,
                        )
                    )
            except Exception:
                logger.exception(
                    "Error on sample for %s/%s ctx=%d hz=%d",
                    model.name,
                    machine,
                    ctx_len,
                    hz,
                )

        if not sample_maes:
            continue

        results.append(
            EvalResult(
                model=model.name,
                machine=machine,
                context_length=ctx_len,
                horizon=hz,
                with_covariates=use_cov,
                device=getattr(model, "_device", ""),
                n_samples=len(sample_maes),
                mae=float(np.mean(sample_maes)),
                rmse=float(np.mean(sample_rmses)),
                inference_ms=float(np.mean(sample_times)),
                coverage=float(np.mean(sample_coverages)) if sample_coverages else None,
                crps=float(np.mean(sample_crps)) if sample_crps else None,
                winkler=float(np.mean(sample_winklers)) if sample_winklers else None,
                training_mode=training_mode,
                ft_epochs=ft_epochs,
                ft_time_s=ft_time_s,
                ft_train_machines=ft_train_machines,
            )
        )

        iterator.set_postfix(
            machine=machine,
            mae=f"{results[-1].mae:.4f}" if results else "N/A",
        )

    rows = [r.to_dict() for r in results]
    return pd.DataFrame(rows)


def load_zero_shot_baselines(
    results_dir: Path,
    model_name: str | None = None,
) -> pd.DataFrame:
    """Load zero-shot results from notebook 02 output for comparison.

    Args:
        results_dir: Path to notebook 02 output directory.
        model_name: If set, filter to this model only.

    Returns:
        DataFrame with zero-shot results, with training_mode="zero_shot" added.
    """
    all_dfs: list[pd.DataFrame] = []

    for subdir in ["gpu", "cpu"]:
        sub_path = results_dir / subdir
        if sub_path.is_dir():
            for csv_path in sorted(sub_path.glob("*.csv")):
                if csv_path.name.startswith("zero_shot"):
                    continue
                df = pd.read_csv(csv_path)
                all_dfs.append(df)

    # Also check root-level CSVs
    for csv_path in sorted(results_dir.glob("*.csv")):
        if csv_path.name.startswith("zero_shot"):
            continue
        if csv_path.parent == results_dir:
            all_dfs.append(pd.read_csv(csv_path))

    if not all_dfs:
        return pd.DataFrame()

    result = pd.concat(all_dfs, ignore_index=True)
    result = result.drop_duplicates(
        subset=["model", "machine", "context_length", "horizon", "with_covariates"],
        keep="last",
    )

    if model_name:
        result = result[result["model"] == model_name]

    # Add FT metadata columns for schema compatibility
    result["training_mode"] = "zero_shot"
    result["ft_epochs"] = None
    result["ft_time_s"] = None
    result["ft_train_machines"] = ""

    return result


def compare_ft_vs_zero_shot(
    ft_results: pd.DataFrame,
    zs_results: pd.DataFrame,
) -> pd.DataFrame:
    """Merge fine-tuned and zero-shot results for side-by-side comparison.

    Args:
        ft_results: Fine-tuned model evaluation results.
        zs_results: Zero-shot baseline results.

    Returns:
        Combined DataFrame sorted by machine, context_length, horizon.
    """
    combined = pd.concat([ft_results, zs_results], ignore_index=True)
    candidate_cols = [
        "machine",
        "context_length",
        "horizon",
        "with_covariates",
        "training_mode",
    ]
    sort_cols = [c for c in candidate_cols if c in combined.columns]
    if sort_cols:
        combined = combined.sort_values(sort_cols).reset_index(drop=True)
    return combined
