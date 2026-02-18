"""Tests for time-series splitting strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tick2.data.splits import (
    extract_samples,
    leave_one_machine_out,
    rolling_origin_cv,
    temporal_split,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a simple time-indexed DataFrame for testing."""
    n = 10000
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "target": rng.standard_normal(n),
            "feat_a": rng.standard_normal(n),
            "feat_b": rng.standard_normal(n),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="1s"),
    )


class TestTemporalSplit:
    def test_default_fractions(self, sample_df: pd.DataFrame) -> None:
        split = temporal_split(sample_df)
        n = len(sample_df)
        assert len(split.train) == int(n * 0.7)
        assert len(split.val) == int(n * 0.85) - int(n * 0.7)
        assert len(split.test) == n - int(n * 0.85)

    def test_no_overlap(self, sample_df: pd.DataFrame) -> None:
        split = temporal_split(sample_df)
        total = len(split.train) + len(split.val) + len(split.test)
        assert total == len(sample_df)

    def test_custom_fractions(self, sample_df: pd.DataFrame) -> None:
        split = temporal_split(sample_df, train_frac=0.5, val_frac=0.25, test_frac=0.25)
        n = len(sample_df)
        assert len(split.train) == int(n * 0.5)
        assert len(split.val) == int(n * 0.75) - int(n * 0.5)

    def test_temporal_ordering(self, sample_df: pd.DataFrame) -> None:
        split = temporal_split(sample_df)
        assert split.train.index[-1] < split.val.index[0]
        assert split.val.index[-1] < split.test.index[0]


class TestExtractSamples:
    def test_correct_count(self, sample_df: pd.DataFrame) -> None:
        samples = extract_samples(sample_df, "target", 512, 60, n_samples=10)
        assert len(samples) == 10

    def test_window_lengths(self, sample_df: pd.DataFrame) -> None:
        samples = extract_samples(sample_df, "target", 512, 60)
        for s in samples:
            assert len(s.context) == 512
            assert len(s.horizon_true) == 60
            assert s.context_len == 512
            assert s.horizon_len == 60

    def test_reproducibility(self, sample_df: pd.DataFrame) -> None:
        s1 = extract_samples(sample_df, "target", 512, 60, seed=42)
        s2 = extract_samples(sample_df, "target", 512, 60, seed=42)
        for a, b in zip(s1, s2, strict=True):
            assert a.start_idx == b.start_idx

    def test_different_seeds(self, sample_df: pd.DataFrame) -> None:
        s1 = extract_samples(sample_df, "target", 512, 60, seed=42)
        s2 = extract_samples(sample_df, "target", 512, 60, seed=99)
        starts_1 = [s.start_idx for s in s1]
        starts_2 = [s.start_idx for s in s2]
        assert starts_1 != starts_2

    def test_sorted_starts(self, sample_df: pd.DataFrame) -> None:
        samples = extract_samples(sample_df, "target", 512, 60)
        starts = [s.start_idx for s in samples]
        assert starts == sorted(starts)

    def test_too_short_raises(self) -> None:
        short_df = pd.DataFrame({"target": [1, 2, 3]})
        with pytest.raises(ValueError, match="too short"):
            extract_samples(short_df, "target", 512, 60)


class TestRollingOriginCV:
    def test_correct_fold_count(self, sample_df: pd.DataFrame) -> None:
        folds = rolling_origin_cv(
            sample_df, n_folds=5, min_train_rows=3600, horizon_len=60
        )
        assert len(folds) == 5

    def test_expanding_train(self, sample_df: pd.DataFrame) -> None:
        folds = rolling_origin_cv(
            sample_df, n_folds=5, min_train_rows=3600, horizon_len=60
        )
        train_sizes = [len(train) for train, _ in folds]
        assert train_sizes == sorted(train_sizes)  # monotonically increasing

    def test_test_horizon_length(self, sample_df: pd.DataFrame) -> None:
        hz = 60
        folds = rolling_origin_cv(
            sample_df, n_folds=5, min_train_rows=3600, horizon_len=hz
        )
        for _, test in folds:
            assert len(test) == hz

    def test_insufficient_data_raises(self) -> None:
        tiny = pd.DataFrame({"x": range(100)})
        with pytest.raises(ValueError, match="Not enough data"):
            rolling_origin_cv(tiny, n_folds=5, min_train_rows=3600, horizon_len=60)


class TestLeaveOneMachineOut:
    def test_correct_number_of_splits(self) -> None:
        datasets = {
            "a": pd.DataFrame({"x": [1, 2]}),
            "b": pd.DataFrame({"x": [3, 4]}),
            "c": pd.DataFrame({"x": [5, 6]}),
        }
        splits = leave_one_machine_out(datasets)
        assert len(splits) == 3

    def test_test_machine_excluded_from_train(self) -> None:
        datasets = {
            "a": pd.DataFrame({"x": [1]}),
            "b": pd.DataFrame({"x": [2]}),
            "c": pd.DataFrame({"x": [3]}),
        }
        for _test_name, train_df, test_df in leave_one_machine_out(datasets):
            assert len(test_df) == 1
            # Train should have data from other 2 machines
            assert len(train_df) == 2

    def test_all_machines_tested(self) -> None:
        datasets = {"a": pd.DataFrame(), "b": pd.DataFrame(), "c": pd.DataFrame()}
        tested = {name for name, _, _ in leave_one_machine_out(datasets)}
        assert tested == {"a", "b", "c"}
