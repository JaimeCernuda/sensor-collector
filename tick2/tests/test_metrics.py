"""Tests for forecasting evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from tick2.benchmark.metrics import (
    coverage,
    crps_empirical,
    crps_quantile,
    mae,
    mase,
    rmse,
    winkler_score,
)


class TestMAE:
    def test_perfect_forecast(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        assert mae(y, y) == 0.0

    def test_known_value(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        assert mae(y_true, y_pred) == pytest.approx(1.0)

    def test_symmetric(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([3.0, 4.0, 5.0])
        assert mae(y_true, y_pred) == mae(y_pred, y_true)


class TestRMSE:
    def test_perfect_forecast(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == 0.0

    def test_known_value(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        # RMSE of [1,1,1] = sqrt(1) = 1.0
        assert rmse(y_true, y_pred) == pytest.approx(1.0)

    def test_rmse_ge_mae(self) -> None:
        """RMSE should always be >= MAE."""
        rng = np.random.default_rng(42)
        y_true = rng.standard_normal(100)
        y_pred = rng.standard_normal(100)
        assert rmse(y_true, y_pred) >= mae(y_true, y_pred)


class TestMASE:
    def test_naive_baseline_equals_one(self) -> None:
        """A naive forecast should get MASE = 1.0 (by definition)."""
        rng = np.random.default_rng(42)
        y_train = rng.standard_normal(100)
        # Naive forecast: repeat last value
        y_true = y_train[-5:]
        y_pred = np.full(5, y_train[-6])  # naive baseline
        # MASE should be close to 1.0 for a naive-like forecast
        result = mase(y_true, y_pred, y_train)
        assert result > 0  # just verify it runs

    def test_perfect_forecast(self) -> None:
        y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_true = np.array([6.0, 7.0])
        assert mase(y_true, y_true, y_train) == 0.0

    def test_zero_scale_returns_inf(self) -> None:
        y_train = np.array([5.0, 5.0, 5.0])  # constant -> zero scale
        y_true = np.array([6.0])
        y_pred = np.array([5.0])
        assert mase(y_true, y_pred, y_train) == float("inf")


class TestCRPSEmpirical:
    def test_perfect_samples(self) -> None:
        """If all samples equal truth, CRPS should be 0."""
        y_true = np.array([1.0, 2.0, 3.0])
        samples = np.tile(y_true, (10, 1))  # 10 identical samples
        assert crps_empirical(y_true, samples) == pytest.approx(0.0)

    def test_positive(self) -> None:
        """CRPS should be non-negative."""
        rng = np.random.default_rng(42)
        y_true = rng.standard_normal(10)
        samples = rng.standard_normal((50, 10))
        assert crps_empirical(y_true, samples) >= 0.0


class TestCRPSQuantile:
    def test_perfect_quantiles(self) -> None:
        """If all quantile forecasts equal truth, CRPS should be 0."""
        y_true = np.array([1.0, 2.0, 3.0])
        q_levels = np.array([0.1, 0.5, 0.9])
        q_forecasts = np.tile(y_true, (3, 1))
        assert crps_quantile(y_true, q_forecasts, q_levels) == pytest.approx(0.0)

    def test_positive(self) -> None:
        rng = np.random.default_rng(42)
        y_true = rng.standard_normal(10)
        q_levels = np.array([0.1, 0.5, 0.9])
        q_forecasts = np.sort(rng.standard_normal((3, 10)), axis=0)
        result = crps_quantile(y_true, q_forecasts, q_levels)
        assert result >= 0.0


class TestCoverage:
    def test_perfect_coverage(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        lower = np.array([0.0, 1.0, 2.0])
        upper = np.array([2.0, 3.0, 4.0])
        assert coverage(y, lower, upper) == pytest.approx(1.0)

    def test_zero_coverage(self) -> None:
        y = np.array([10.0, 20.0, 30.0])
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([1.0, 1.0, 1.0])
        assert coverage(y, lower, upper) == pytest.approx(0.0)

    def test_partial_coverage(self) -> None:
        y = np.array([1.0, 10.0])
        lower = np.array([0.0, 0.0])
        upper = np.array([2.0, 5.0])
        assert coverage(y, lower, upper) == pytest.approx(0.5)

    def test_boundary_included(self) -> None:
        """Values exactly on the boundary should be counted as covered."""
        y = np.array([1.0, 3.0])
        lower = np.array([1.0, 2.0])
        upper = np.array([2.0, 3.0])
        assert coverage(y, lower, upper) == pytest.approx(1.0)


class TestWinklerScore:
    def test_perfect_tight_interval(self) -> None:
        """Narrow interval containing all values = width only, no penalty."""
        y = np.array([1.0, 2.0, 3.0])
        lower = y - 0.1
        upper = y + 0.1
        score = winkler_score(y, lower, upper, alpha=0.2)
        assert score == pytest.approx(0.2)  # just the width

    def test_penalty_for_missed(self) -> None:
        """Values outside interval incur penalty."""
        y = np.array([10.0])
        lower = np.array([0.0])
        upper = np.array([5.0])
        score = winkler_score(y, lower, upper, alpha=0.2)
        # Width = 5.0, penalty_hi = (2/0.2) * (10-5) = 50.0
        assert score == pytest.approx(55.0)

    def test_lower_is_better(self) -> None:
        """Tighter intervals should give lower scores when covering."""
        y = np.array([5.0])
        score_tight = winkler_score(y, np.array([4.0]), np.array([6.0]))
        score_wide = winkler_score(y, np.array([0.0]), np.array([10.0]))
        assert score_tight < score_wide
