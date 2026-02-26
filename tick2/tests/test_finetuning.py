"""Tests for fine-tuning infrastructure (base types, data prep, evaluation)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tick2.data.preprocessing import TARGET_COL
from tick2.finetuning.base import FineTuneConfig, FineTuneResult, save_loss_history
from tick2.finetuning.data_prep import (
    PRIORITY_CATEGORIES,
    PreparedData,
    combine_training_data,
    combine_validation_data,
    get_machine_arrays,
    select_top_features,
)
from tick2.finetuning.evaluate import (
    EvalResult,
    compare_ft_vs_zero_shot,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_df(n_rows: int = 1000, n_features: int = 10, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic DataFrame mimicking sensor data."""
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {TARGET_COL: rng.standard_normal(n_rows)}
    for i in range(n_features):
        data[f"feat_{i}"] = rng.standard_normal(n_rows)
    df = pd.DataFrame(data)
    df.index = pd.date_range("2024-01-01", periods=n_rows, freq="1s")
    return df


def _make_categories(n_features: int = 10) -> dict[str, str]:
    """Create synthetic feature categories ensuring priority coverage."""
    cats: dict[str, str] = {}
    cat_cycle = [
        "CPU Core Temp",
        "CPU Frequency",
        "Power",
        "C-State",
        "Memory",
        "CPU Load",
        "CPU Package Temp",
        "I/O",
        "System",
        "Non-CPU Temp",
    ]
    for i in range(n_features):
        cats[f"feat_{i}"] = cat_cycle[i % len(cat_cycle)]
    return cats


def _make_prepared(
    n_rows: int = 1000, n_features: int = 10, name: str = "test"
) -> PreparedData:
    """Create a PreparedData object for testing."""
    from tick2.data.splits import temporal_split

    df = _make_df(n_rows, n_features)
    cats = _make_categories(n_features)
    features = [f"feat_{i}" for i in range(n_features)]
    split = temporal_split(df)
    return PreparedData(
        name=name,
        split=split,
        feature_cols=features,
        categories=cats,
    )


# ---------------------------------------------------------------------------
# FineTuneConfig
# ---------------------------------------------------------------------------


class TestFineTuneConfig:
    def test_defaults(self) -> None:
        cfg = FineTuneConfig()
        assert cfg.train_frac == 0.7
        assert cfg.val_frac == 0.15
        assert cfg.test_frac == 0.15
        assert cfg.context_length == 1024
        assert cfg.prediction_length == 96
        assert cfg.max_covariates == 30
        assert cfg.seed == 42

    def test_fractions_sum_to_one(self) -> None:
        cfg = FineTuneConfig()
        assert cfg.train_frac + cfg.val_frac + cfg.test_frac == pytest.approx(1.0)

    def test_custom_values(self) -> None:
        cfg = FineTuneConfig(context_length=512, prediction_length=192, seed=99)
        assert cfg.context_length == 512
        assert cfg.prediction_length == 192
        assert cfg.seed == 99


# ---------------------------------------------------------------------------
# FineTuneResult
# ---------------------------------------------------------------------------


class TestFineTuneResult:
    def test_creation(self) -> None:
        result = FineTuneResult(
            model_name="test-model",
            machine="homelab",
            train_loss=[1.0, 0.5, 0.3],
            val_loss=[1.1, 0.6, 0.4],
            best_epoch=2,
            training_time_s=120.5,
            checkpoint_path="/tmp/best",
            config={"lr": 0.001},
        )
        assert result.model_name == "test-model"
        assert result.machine == "homelab"
        assert len(result.train_loss) == 3
        assert result.best_epoch == 2
        assert result.config["lr"] == 0.001

    def test_defaults(self) -> None:
        result = FineTuneResult(model_name="m", machine="x")
        assert result.train_loss == []
        assert result.val_loss == []
        assert result.best_epoch == 0
        assert result.training_time_s == 0.0
        assert result.checkpoint_path == ""
        assert result.config == {}


# ---------------------------------------------------------------------------
# select_top_features
# ---------------------------------------------------------------------------


class TestSelectTopFeatures:
    def test_returns_all_when_under_limit(self) -> None:
        df = _make_df(100, 5)
        cats = _make_categories(5)
        selected = select_top_features(df, cats, max_features=10)
        assert len(selected) == 5

    def test_respects_max_features(self) -> None:
        df = _make_df(100, 20)
        cats = _make_categories(20)
        selected = select_top_features(df, cats, max_features=10)
        assert len(selected) == 10

    def test_priority_categories_represented(self) -> None:
        """Priority categories should have at least one representative."""
        df = _make_df(500, 30)
        cats = _make_categories(30)
        selected = select_top_features(df, cats, max_features=15)

        selected_cats = {cats[f] for f in selected if f in cats}
        for priority_cat in PRIORITY_CATEGORIES:
            # Only check if the category exists in our test data
            if any(c == priority_cat for c in cats.values()):
                assert priority_cat in selected_cats, (
                    f"Priority category {priority_cat!r} missing from selection"
                )

    def test_no_duplicates(self) -> None:
        df = _make_df(100, 20)
        cats = _make_categories(20)
        selected = select_top_features(df, cats, max_features=10)
        assert len(selected) == len(set(selected))

    def test_excludes_target_column(self) -> None:
        df = _make_df(100, 10)
        cats = _make_categories(10)
        selected = select_top_features(df, cats, max_features=15)
        assert TARGET_COL not in selected


# ---------------------------------------------------------------------------
# combine_training_data / combine_validation_data
# ---------------------------------------------------------------------------


class TestCombineData:
    def test_combine_training_concatenates(self) -> None:
        p1 = _make_prepared(500, 5, name="m1")
        p2 = _make_prepared(500, 5, name="m2")
        combined_df, _shared = combine_training_data({"m1": p1, "m2": p2})

        # Combined should have 2x individual train sizes
        expected_rows = len(p1.split.train) + len(p2.split.train)
        assert len(combined_df) == expected_rows

    def test_shared_features_intersection(self) -> None:
        """Combined data uses only features common to all machines."""
        p1 = _make_prepared(500, 8, name="m1")
        p2 = _make_prepared(500, 8, name="m2")
        # Give them different feature columns (overlapping)
        p1.feature_cols = ["feat_0", "feat_1", "feat_2", "feat_3"]
        p2.feature_cols = ["feat_2", "feat_3", "feat_4", "feat_5"]

        _, shared = combine_training_data({"m1": p1, "m2": p2})
        assert set(shared) == {"feat_2", "feat_3"}

    def test_combine_validation(self) -> None:
        p1 = _make_prepared(500, 5, name="m1")
        p2 = _make_prepared(500, 5, name="m2")
        val_df, _shared = combine_validation_data({"m1": p1, "m2": p2})

        expected_rows = len(p1.split.val) + len(p2.split.val)
        assert len(val_df) == expected_rows

    def test_target_column_present(self) -> None:
        p = _make_prepared(500, 5)
        df, _ = combine_training_data({"test": p})
        assert TARGET_COL in df.columns


# ---------------------------------------------------------------------------
# get_machine_arrays
# ---------------------------------------------------------------------------


class TestGetMachineArrays:
    def test_train_split(self) -> None:
        p = _make_prepared(500, 5)
        target, cov = get_machine_arrays(p, "train")
        assert target.shape == (len(p.split.train),)
        assert cov is not None
        assert cov.shape == (len(p.split.train), 5)

    def test_val_split(self) -> None:
        p = _make_prepared(500, 5)
        target, _cov = get_machine_arrays(p, "val")
        assert target.shape == (len(p.split.val),)

    def test_test_split(self) -> None:
        p = _make_prepared(500, 5)
        target, _cov = get_machine_arrays(p, "test")
        assert target.shape == (len(p.split.test),)

    def test_no_covariates(self) -> None:
        p = _make_prepared(500, 5)
        p.feature_cols = []
        target, cov = get_machine_arrays(p, "train")
        assert target.shape == (len(p.split.train),)
        assert cov is None


# ---------------------------------------------------------------------------
# EvalResult
# ---------------------------------------------------------------------------


class TestEvalResult:
    def test_to_dict(self) -> None:
        r = EvalResult(
            model="test-ft",
            machine="homelab",
            context_length=512,
            horizon=96,
            with_covariates=True,
            device="cuda",
            n_samples=25,
            mae=0.5,
            rmse=0.7,
            inference_ms=10.0,
            coverage=0.8,
            training_mode="ft_combined",
            ft_epochs=10,
            ft_time_s=60.0,
            ft_train_machines="homelab,chameleon",
        )
        d = r.to_dict()
        assert d["model"] == "test-ft"
        assert d["mae"] == 0.5
        assert d["training_mode"] == "ft_combined"
        assert d["ft_epochs"] == 10
        assert d["ft_train_machines"] == "homelab,chameleon"

    def test_default_training_mode(self) -> None:
        r = EvalResult(
            model="m",
            machine="x",
            context_length=0,
            horizon=0,
            with_covariates=False,
            device="cpu",
            n_samples=0,
            mae=0.0,
            rmse=0.0,
            inference_ms=0.0,
        )
        assert r.training_mode == "zero_shot"


# ---------------------------------------------------------------------------
# compare_ft_vs_zero_shot
# ---------------------------------------------------------------------------


class TestCompareFtVsZeroShot:
    def test_merges_dataframes(self) -> None:
        ft = pd.DataFrame(
            {
                "model": ["m-ft"],
                "machine": ["homelab"],
                "context_length": [512],
                "horizon": [96],
                "with_covariates": [False],
                "mae": [0.5],
                "training_mode": ["ft_combined"],
            }
        )
        zs = pd.DataFrame(
            {
                "model": ["m"],
                "machine": ["homelab"],
                "context_length": [512],
                "horizon": [96],
                "with_covariates": [False],
                "mae": [0.8],
                "training_mode": ["zero_shot"],
            }
        )
        combined = compare_ft_vs_zero_shot(ft, zs)
        assert len(combined) == 2
        assert set(combined["training_mode"]) == {"ft_combined", "zero_shot"}

    def test_sorted_output(self) -> None:
        ft = pd.DataFrame(
            {
                "model": ["m-ft", "m-ft"],
                "machine": ["chameleon", "ares"],
                "context_length": [512, 1024],
                "horizon": [96, 96],
                "with_covariates": [False, False],
                "mae": [0.5, 0.6],
                "training_mode": ["ft_combined", "ft_combined"],
            }
        )
        zs = pd.DataFrame()
        combined = compare_ft_vs_zero_shot(ft, zs)
        # Should be sorted by machine
        machines = combined["machine"].tolist()
        assert machines == sorted(machines)


# ---------------------------------------------------------------------------
# PreparedData
# ---------------------------------------------------------------------------


class TestPreparedData:
    def test_creation(self) -> None:
        p = _make_prepared(500, 5, name="homelab")
        assert p.name == "homelab"
        assert len(p.feature_cols) == 5
        assert len(p.split.train) > 0
        assert len(p.split.val) > 0
        assert len(p.split.test) > 0

    def test_split_proportions(self) -> None:
        p = _make_prepared(1000, 5)
        total = len(p.split.train) + len(p.split.val) + len(p.split.test)
        assert total == 1000
        # Default 70/15/15 split
        assert len(p.split.train) == 700
        assert len(p.split.val) == 150
        assert len(p.split.test) == 150


# ---------------------------------------------------------------------------
# save_loss_history
# ---------------------------------------------------------------------------


class TestSaveLossHistory:
    def test_creates_csv(self, tmp_path: Path) -> None:
        csv_path = save_loss_history(tmp_path, [0.5, 0.4, 0.3], [0.6, 0.5, 0.4])
        assert csv_path.exists()
        assert csv_path.name == "loss_history.csv"

    def test_csv_content(self, tmp_path: Path) -> None:
        save_loss_history(tmp_path, [1.0, 0.5], [1.2, 0.6])
        df = pd.read_csv(tmp_path / "loss_history.csv")
        assert list(df.columns) == ["epoch", "train_loss", "val_loss"]
        assert len(df) == 2
        assert df["epoch"].tolist() == [1, 2]
        assert df["train_loss"].tolist() == [1.0, 0.5]
        assert df["val_loss"].tolist() == [1.2, 0.6]

    def test_incremental_append(self, tmp_path: Path) -> None:
        """Subsequent calls append only new epochs."""
        save_loss_history(tmp_path, [1.0], [1.2])
        save_loss_history(tmp_path, [1.0, 0.5], [1.2, 0.6])
        df = pd.read_csv(tmp_path / "loss_history.csv")
        assert len(df) == 2
        assert df["epoch"].tolist() == [1, 2]

    def test_no_duplicate_rows(self, tmp_path: Path) -> None:
        """Calling with same data twice does not duplicate rows."""
        save_loss_history(tmp_path, [1.0, 0.5], [1.2, 0.6])
        save_loss_history(tmp_path, [1.0, 0.5], [1.2, 0.6])
        df = pd.read_csv(tmp_path / "loss_history.csv")
        assert len(df) == 2

    def test_run_name_prefix(self, tmp_path: Path) -> None:
        csv_path = save_loss_history(tmp_path, [1.0], [1.2], run_name="combined")
        assert csv_path.name == "combined_loss_history.csv"

    def test_creates_output_dir(self, tmp_path: Path) -> None:
        nested = tmp_path / "deep" / "nested"
        csv_path = save_loss_history(nested, [1.0], [1.2])
        assert nested.exists()
        assert csv_path.exists()

    def test_empty_val_loss(self, tmp_path: Path) -> None:
        """Train-only loss (no validation) still works."""
        save_loss_history(tmp_path, [1.0, 0.5, 0.3], [])
        df = pd.read_csv(tmp_path / "loss_history.csv")
        assert len(df) == 3
        assert df["val_loss"].isna().all()
