"""Benchmark runner: orchestrates model x machine x config evaluation.

Runs all combinations of models, machines, context lengths, horizons,
and covariate modes, collecting PredictionResults and computing metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from tick2.benchmark import metrics
from tick2.data.adapters.common import extract_arrays
from tick2.data.preprocessing import TARGET_COL, get_feature_cols
from tick2.data.splits import SampleWindow, extract_samples
from tick2.models.base import ModelWrapper

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    context_lengths: list[int] = field(default_factory=lambda: [512, 1024])
    horizons: list[int] = field(default_factory=lambda: [60, 120])
    n_samples: int = 25
    seed: int = 42
    use_covariates: list[bool] = field(default_factory=lambda: [False, True])
    quantile_alpha: float = 0.2  # for Winkler score (80% intervals)


@dataclass
class SampleResult:
    """Result for a single (context, horizon) sample."""

    sample_idx: int
    start_idx: int
    mae: float
    rmse: float
    inference_time_ms: float
    coverage: float | None = None
    winkler: float | None = None
    crps: float | None = None


@dataclass
class RunResult:
    """Result for a single model x machine x config combination."""

    model_name: str
    machine: str
    context_length: int
    horizon: int
    with_covariates: bool
    device: str
    n_samples: int
    sample_results: list[SampleResult] = field(default_factory=list)

    @property
    def mean_mae(self) -> float:
        return float(np.mean([s.mae for s in self.sample_results]))

    @property
    def mean_rmse(self) -> float:
        return float(np.mean([s.rmse for s in self.sample_results]))

    @property
    def mean_inference_ms(self) -> float:
        return float(np.mean([s.inference_time_ms for s in self.sample_results]))

    @property
    def mean_coverage(self) -> float | None:
        vals = [s.coverage for s in self.sample_results if s.coverage is not None]
        return float(np.mean(vals)) if vals else None

    @property
    def mean_crps(self) -> float | None:
        vals = [s.crps for s in self.sample_results if s.crps is not None]
        return float(np.mean(vals)) if vals else None

    def to_dict(self) -> dict[str, Any]:
        """Flatten to a dict for DataFrame construction."""
        return {
            "model": self.model_name,
            "machine": self.machine,
            "context_length": self.context_length,
            "horizon": self.horizon,
            "with_covariates": self.with_covariates,
            "device": self.device,
            "n_samples": self.n_samples,
            "mae": self.mean_mae,
            "rmse": self.mean_rmse,
            "inference_ms": self.mean_inference_ms,
            "coverage": self.mean_coverage,
            "crps": self.mean_crps,
        }


def _evaluate_sample(
    model: ModelWrapper,
    sample: SampleWindow,
    feature_cols: list[str] | None,
    use_covariates: bool,
    alpha: float,
) -> SampleResult:
    """Run inference on a single sample and compute metrics."""
    target, covariates = extract_arrays(sample.context, TARGET_COL, feature_cols)
    y_true = sample.horizon_true.to_numpy(dtype=np.float64)

    cov_arg = covariates if (use_covariates and model.supports_covariates) else None

    pred = model.predict(
        context=target,
        horizon=sample.horizon_len,
        covariates=cov_arg,
    )

    point = pred.point_forecast[: len(y_true)]

    sample_mae = metrics.mae(y_true, point)
    sample_rmse = metrics.rmse(y_true, point)

    cov_val = None
    winkler_val = None
    crps_val = None

    if pred.quantile_lo is not None and pred.quantile_hi is not None:
        lo = pred.quantile_lo[: len(y_true)]
        hi = pred.quantile_hi[: len(y_true)]
        cov_val = metrics.coverage(y_true, lo, hi)
        winkler_val = metrics.winkler_score(y_true, lo, hi, alpha=alpha)

    if pred.quantiles is not None and pred.quantile_levels is not None:
        crps_val = metrics.crps_quantile(
            y_true,
            pred.quantiles[:, : len(y_true)],
            pred.quantile_levels,
        )

    return SampleResult(
        sample_idx=0,  # filled by caller
        start_idx=sample.start_idx,
        mae=sample_mae,
        rmse=sample_rmse,
        inference_time_ms=pred.inference_time_ms,
        coverage=cov_val,
        winkler=winkler_val,
        crps=crps_val,
    )


def run_benchmark(
    model: ModelWrapper,
    datasets: dict[str, tuple[pd.DataFrame, dict[str, str]]],
    config: BenchmarkConfig,
    progress: bool = True,
) -> list[RunResult]:
    """Run the full benchmark for a single model across all machines and configs.

    Args:
        model: A loaded ModelWrapper instance.
        datasets: Dict of machine_name -> (DataFrame, feature_categories).
        config: Benchmark configuration.
        progress: Show tqdm progress bar.

    Returns:
        List of RunResult objects, one per machine x config combination.
    """
    results: list[RunResult] = []

    combos = [
        (machine, ctx_len, hz, use_cov)
        for machine in datasets
        for ctx_len in config.context_lengths
        for hz in config.horizons
        for use_cov in config.use_covariates
    ]

    iterator = tqdm(combos, desc=f"{model.name}", disable=not progress)

    for machine, ctx_len, hz, use_cov in iterator:
        df, _cats = datasets[machine]
        feature_cols = get_feature_cols(df) if use_cov else None

        # Skip covariate runs if model doesn't support them
        if use_cov and not model.supports_covariates:
            continue

        try:
            samples = extract_samples(
                df,
                target_col=TARGET_COL,
                context_len=ctx_len,
                horizon_len=hz,
                n_samples=config.n_samples,
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

        run = RunResult(
            model_name=model.name,
            machine=machine,
            context_length=ctx_len,
            horizon=hz,
            with_covariates=use_cov,
            device=getattr(model, "_device", ""),
            n_samples=len(samples),
        )

        for i, sample in enumerate(samples):
            try:
                sr = _evaluate_sample(
                    model, sample, feature_cols, use_cov, config.quantile_alpha
                )
                sr.sample_idx = i
                run.sample_results.append(sr)
            except Exception:
                logger.exception(
                    "Error on sample %d for %s/%s ctx=%d hz=%d",
                    i,
                    model.name,
                    machine,
                    ctx_len,
                    hz,
                )

        if run.sample_results:
            results.append(run)

        iterator.set_postfix(
            machine=machine, mae=f"{run.mean_mae:.4f}" if run.sample_results else "N/A"
        )

    return results


def results_to_dataframe(results: list[RunResult]) -> pd.DataFrame:
    """Convert a list of RunResult objects to a summary DataFrame."""
    rows = [r.to_dict() for r in results]
    return pd.DataFrame(rows)
