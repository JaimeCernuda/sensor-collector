"""End-to-end smoke tests for fine-tuning pipelines.

Runs each model's fine-tuning pipeline on a tiny data subset (1 epoch,
~1000 rows) to verify the full flow: data prep -> train -> evaluate -> compare.

Usage:
    python -m tests.smoke_test_finetune granite
    python -m tests.smoke_test_finetune chronos2
    python -m tests.smoke_test_finetune moirai
    python -m tests.smoke_test_finetune timesfm
    python -m tests.smoke_test_finetune all
"""

from __future__ import annotations

import argparse
import contextlib
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tick2.data.preprocessing import TARGET_COL
from tick2.data.splits import TemporalSplit
from tick2.finetuning.base import FineTuneConfig
from tick2.finetuning.data_prep import PreparedData
from tick2.finetuning.evaluate import (
    compare_ft_vs_zero_shot,
    evaluate_finetuned,
)

# Tiny config for smoke tests
SMOKE_CONFIG = FineTuneConfig(
    context_length=512,
    prediction_length=96,
    max_covariates=5,
    seed=42,
)

N_ROWS = 5000  # needs enough rows for ctx+hz windows in test split
N_FEATURES = 8


def _make_synthetic_data() -> dict[str, PreparedData]:
    """Create synthetic sensor-like data for smoke testing."""
    rng = np.random.default_rng(42)

    machines: dict[str, PreparedData] = {}
    for name in ["smoke_machine"]:
        # Generate drift-like target: random walk + seasonal
        t = np.arange(N_ROWS, dtype=np.float64)
        target = np.cumsum(rng.standard_normal(N_ROWS) * 0.1) + 5.0 * np.sin(
            2 * np.pi * t / 300
        )

        data: dict[str, np.ndarray] = {TARGET_COL: target}
        cats: dict[str, str] = {}
        features: list[str] = []

        cat_names = [
            "CPU Core Temp",
            "CPU Frequency",
            "Power",
            "C-State",
            "Memory",
            "CPU Load",
            "I/O",
            "System",
        ]
        for i in range(N_FEATURES):
            col = f"feat_{i}"
            # Correlated with target + noise
            data[col] = target * (0.3 + 0.1 * i) + rng.standard_normal(N_ROWS) * 2
            cats[col] = cat_names[i % len(cat_names)]
            features.append(col)

        df = pd.DataFrame(data)
        df.index = pd.date_range("2024-01-01", periods=N_ROWS, freq="1s")

        # Manual temporal split
        n = len(df)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)

        split = TemporalSplit(
            train=df.iloc[:train_end].copy(),
            val=df.iloc[train_end:val_end].copy(),
            test=df.iloc[val_end:].copy(),
        )

        machines[name] = PreparedData(
            name=name,
            split=split,
            feature_cols=features[: SMOKE_CONFIG.max_covariates],
            categories={f: cats[f] for f in features[: SMOKE_CONFIG.max_covariates]},
        )

    return machines


def _print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _print_result(result: object) -> None:
    from tick2.finetuning.base import FineTuneResult

    if isinstance(result, FineTuneResult):
        print(f"  Model:     {result.model_name}")
        print(f"  Machine:   {result.machine}")
        print(f"  Time:      {result.training_time_s:.1f}s")
        print(f"  Best epoch: {result.best_epoch}")
        if result.train_loss:
            print(f"  Train loss: {result.train_loss[-1]:.6f}")
        if result.val_loss:
            print(f"  Val loss:   {result.val_loss[-1]:.6f}")
        print(f"  Checkpoint: {result.checkpoint_path}")


# ---------------------------------------------------------------------------
# Granite TTM smoke test
# ---------------------------------------------------------------------------


def smoke_test_granite(tmp_dir: Path) -> bool:
    """Smoke test Granite TTM fine-tuning pipeline."""
    _print_section("Granite TTM Fine-Tuning Smoke Test")

    from tick2.finetuning.granite_ft import finetune_granite, load_finetuned_granite
    from tick2.models.granite import GraniteTTMWrapper

    prepared = _make_synthetic_data()
    output_dir = tmp_dir / "granite_ft"

    # 1. Fine-tune (1 epoch, tiny data)
    print("\n[1/4] Fine-tuning (1 epoch)...")
    t0 = time.perf_counter()
    results = finetune_granite(
        prepared=prepared,
        config=SMOKE_CONFIG,
        output_dir=str(output_dir),
        training_mode="combined",
        decoder_mode="common_channel",  # univariate smoke test
        freeze_backbone=True,
        learning_rate=0.001,
        num_epochs=1,
        batch_size=32,
        early_stopping_patience=1,
    )
    elapsed = time.perf_counter() - t0
    print(f"  Fine-tuning completed in {elapsed:.1f}s")

    assert len(results) == 1, f"Expected 1 result, got {len(results)}"
    _print_result(results[0])

    # 2. Load checkpoint
    print("\n[2/4] Loading fine-tuned checkpoint...")
    ckpt_path = results[0].checkpoint_path
    assert Path(ckpt_path).exists(), f"Checkpoint not found: {ckpt_path}"
    ft_model = load_finetuned_granite(
        ckpt_path,
        context_length=SMOKE_CONFIG.context_length,
        prediction_length=SMOKE_CONFIG.prediction_length,
    )
    print(f"  Loaded from {ckpt_path}")

    # 3. Evaluate using wrapper
    print("\n[3/4] Evaluating on test split...")
    wrapper = GraniteTTMWrapper(
        model_name="granite-ttm-ft",
        context_length=SMOKE_CONFIG.context_length,
        prediction_length=SMOKE_CONFIG.prediction_length,
    )
    wrapper._model = ft_model
    wrapper._device = "cpu"

    eval_df = evaluate_finetuned(
        model=wrapper,
        prepared=prepared,
        config=SMOKE_CONFIG,
        training_mode="ft_combined",
        context_lengths=[512],
        horizons=[96],
        n_samples=3,
        progress=False,
    )
    print(f"  Eval rows: {len(eval_df)}")
    if not eval_df.empty:
        print(f"  Mean MAE: {eval_df['mae'].mean():.4f}")

    # 4. Compare with fake zero-shot baseline
    print("\n[4/4] Comparing FT vs zero-shot...")
    zs_df = pd.DataFrame(
        {
            "model": ["granite-ttm"],
            "machine": ["smoke_machine"],
            "context_length": [512],
            "horizon": [96],
            "with_covariates": [False],
            "mae": [10.0],
            "training_mode": ["zero_shot"],
        }
    )
    combined = compare_ft_vs_zero_shot(eval_df, zs_df)
    print(f"  Combined rows: {len(combined)}")
    assert len(combined) >= 2, "Should have both FT and ZS rows"

    print("\n  GRANITE TTM SMOKE TEST PASSED")
    return True


# ---------------------------------------------------------------------------
# Chronos-2 smoke test
# ---------------------------------------------------------------------------


def smoke_test_chronos2(tmp_dir: Path) -> bool:
    """Smoke test Chronos-2 fine-tuning pipeline."""
    _print_section("Chronos-2 Fine-Tuning Smoke Test")

    from tick2.finetuning.chronos2_ft import finetune_chronos2, load_finetuned_chronos2
    from tick2.models.chronos2 import Chronos2Wrapper

    prepared = _make_synthetic_data()
    output_dir = tmp_dir / "chronos2_ft"

    # 1. Fine-tune (minimal steps)
    print("\n[1/4] Fine-tuning (50 steps, LoRA)...")
    t0 = time.perf_counter()
    results = finetune_chronos2(
        prepared=prepared,
        config=SMOKE_CONFIG,
        output_dir=str(output_dir),
        training_mode="combined",
        model_id="autogluon/chronos-2-small",
        finetune_mode="lora",
        with_covariates=False,  # univariate for speed
        learning_rate=1e-5,
        num_steps=50,
        batch_size=32,
        device_map="cpu",
    )
    elapsed = time.perf_counter() - t0
    print(f"  Fine-tuning completed in {elapsed:.1f}s")

    assert len(results) == 1
    _print_result(results[0])

    # 2. Load checkpoint
    print("\n[2/4] Loading fine-tuned checkpoint...")
    ckpt_path = results[0].checkpoint_path
    ft_pipeline = load_finetuned_chronos2(ckpt_path, device_map="cpu")
    print(f"  Loaded from {ckpt_path}")

    # 3. Evaluate
    print("\n[3/4] Evaluating on test split...")
    wrapper = Chronos2Wrapper(model_id=str(ckpt_path), model_name="chronos2-ft")
    wrapper._pipeline = ft_pipeline
    wrapper._device = "cpu"

    eval_df = evaluate_finetuned(
        model=wrapper,
        prepared=prepared,
        config=SMOKE_CONFIG,
        training_mode="ft_combined",
        context_lengths=[512],
        horizons=[96],
        n_samples=3,
        progress=False,
    )
    print(f"  Eval rows: {len(eval_df)}")
    if not eval_df.empty:
        print(f"  Mean MAE: {eval_df['mae'].mean():.4f}")

    # 4. Compare
    print("\n[4/4] Comparing FT vs zero-shot...")
    zs_df = pd.DataFrame(
        {
            "model": ["chronos2-small"],
            "machine": ["smoke_machine"],
            "context_length": [512],
            "horizon": [96],
            "with_covariates": [False],
            "mae": [10.0],
            "training_mode": ["zero_shot"],
        }
    )
    combined = compare_ft_vs_zero_shot(eval_df, zs_df)
    print(f"  Combined rows: {len(combined)}")

    print("\n  CHRONOS-2 SMOKE TEST PASSED")
    return True


# ---------------------------------------------------------------------------
# Moirai smoke test
# ---------------------------------------------------------------------------


def smoke_test_moirai(tmp_dir: Path) -> bool:
    """Smoke test Moirai fine-tuning pipeline."""
    _print_section("Moirai 1.1 Fine-Tuning Smoke Test")

    from tick2.finetuning.moirai_ft import finetune_moirai, load_finetuned_moirai
    from tick2.models.moirai import MoiraiWrapper

    prepared = _make_synthetic_data()
    output_dir = tmp_dir / "moirai_ft"

    # 1. Fine-tune (1 epoch)
    print("\n[1/4] Fine-tuning (1 epoch)...")
    t0 = time.perf_counter()
    results = finetune_moirai(
        prepared=prepared,
        config=SMOKE_CONFIG,
        output_dir=str(output_dir),
        training_mode="combined",
        patch_size=32,
        max_epochs=1,
        learning_rate=1e-4,
        batch_size=16,
        early_stopping_patience=1,
        num_samples=10,
        stride=256,
        device="cpu",
    )
    elapsed = time.perf_counter() - t0
    print(f"  Fine-tuning completed in {elapsed:.1f}s")

    assert len(results) == 1
    _print_result(results[0])

    # 2. Load checkpoint
    print("\n[2/4] Loading fine-tuned checkpoint...")
    ckpt_path = results[0].checkpoint_path
    n_cov = SMOKE_CONFIG.max_covariates
    ft_model = load_finetuned_moirai(
        ckpt_path,
        context_length=SMOKE_CONFIG.context_length,
        prediction_length=SMOKE_CONFIG.prediction_length,
        n_covariates=n_cov,
        num_samples=10,
    )
    print(f"  Loaded from {ckpt_path}")

    # 3. Evaluate
    print("\n[3/4] Evaluating on test split...")
    wrapper = MoiraiWrapper(
        model_name="moirai-1.1-ft",
        max_covariates=n_cov,
        num_samples=10,
    )
    # For eval, we need the module, not the full MoiraiForecast
    if hasattr(ft_model, "module"):
        wrapper._model = ft_model.module
    else:
        wrapper._model = ft_model
    wrapper._device = "cpu"

    eval_df = evaluate_finetuned(
        model=wrapper,
        prepared=prepared,
        config=SMOKE_CONFIG,
        training_mode="ft_combined",
        context_lengths=[512],
        horizons=[96],
        n_samples=3,
        progress=False,
    )
    print(f"  Eval rows: {len(eval_df)}")
    if not eval_df.empty:
        print(f"  Mean MAE: {eval_df['mae'].mean():.4f}")

    # 4. Compare
    print("\n[4/4] Comparing FT vs zero-shot...")
    zs_df = pd.DataFrame(
        {
            "model": ["moirai-1.1-small"],
            "machine": ["smoke_machine"],
            "context_length": [512],
            "horizon": [96],
            "with_covariates": [False],
            "mae": [10.0],
            "training_mode": ["zero_shot"],
        }
    )
    combined = compare_ft_vs_zero_shot(eval_df, zs_df)
    print(f"  Combined rows: {len(combined)}")

    print("\n  MOIRAI SMOKE TEST PASSED")
    return True


# ---------------------------------------------------------------------------
# TimesFM smoke test
# ---------------------------------------------------------------------------


def smoke_test_timesfm(tmp_dir: Path) -> bool:
    """Smoke test TimesFM fine-tuning pipeline."""
    _print_section("TimesFM 2.5 Fine-Tuning Smoke Test")

    from tick2.finetuning.timesfm_ft import finetune_timesfm, load_finetuned_timesfm
    from tick2.models.timesfm import TimesFMWrapper

    prepared = _make_synthetic_data()
    output_dir = tmp_dir / "timesfm_ft"

    # 1. Fine-tune (1 epoch)
    print("\n[1/4] Fine-tuning (1 epoch)...")
    t0 = time.perf_counter()
    results = finetune_timesfm(
        prepared=prepared,
        config=SMOKE_CONFIG,
        output_dir=str(output_dir),
        training_mode="combined",
        learning_rate=1e-4,
        num_epochs=1,
        batch_size=16,
        context_length=4,  # 4 patches x 32 = 128 timesteps
        horizon_length=4,
    )
    elapsed = time.perf_counter() - t0
    print(f"  Fine-tuning completed in {elapsed:.1f}s")

    assert len(results) == 1
    _print_result(results[0])

    # 2. Load checkpoint
    print("\n[2/4] Loading fine-tuned checkpoint...")
    ckpt_path = results[0].checkpoint_path
    ft_model = load_finetuned_timesfm(ckpt_path)
    print(f"  Loaded from {ckpt_path}")

    # 3. Evaluate
    print("\n[3/4] Evaluating on test split...")
    wrapper = TimesFMWrapper(model_name="timesfm-2.5-ft")
    wrapper._model = ft_model
    wrapper._device = "cpu"

    eval_df = evaluate_finetuned(
        model=wrapper,
        prepared=prepared,
        config=SMOKE_CONFIG,
        training_mode="ft_combined",
        context_lengths=[512],
        horizons=[96],
        n_samples=3,
        progress=False,
    )
    print(f"  Eval rows: {len(eval_df)}")
    if not eval_df.empty:
        print(f"  Mean MAE: {eval_df['mae'].mean():.4f}")

    # 4. Compare
    print("\n[4/4] Comparing FT vs zero-shot...")
    zs_df = pd.DataFrame(
        {
            "model": ["timesfm-2.5"],
            "machine": ["smoke_machine"],
            "context_length": [512],
            "horizon": [96],
            "with_covariates": [False],
            "mae": [10.0],
            "training_mode": ["zero_shot"],
        }
    )
    combined = compare_ft_vs_zero_shot(eval_df, zs_df)
    print(f"  Combined rows: {len(combined)}")

    print("\n  TIMESFM SMOKE TEST PASSED")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TESTS = {
    "granite": smoke_test_granite,
    "chronos2": smoke_test_chronos2,
    "moirai": smoke_test_moirai,
    "timesfm": smoke_test_timesfm,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tuning smoke tests")
    parser.add_argument(
        "model",
        choices=[*TESTS.keys(), "all"],
        help="Which model to smoke test",
    )
    args = parser.parse_args()

    models = list(TESTS.keys()) if args.model == "all" else [args.model]

    tmp_dir = Path(tempfile.mkdtemp(prefix="chronotick_smoke_"))
    print(f"Temp dir: {tmp_dir}")

    passed: list[str] = []
    failed: list[str] = []

    for model in models:
        try:
            ok = TESTS[model](tmp_dir)
            if ok:
                passed.append(model)
            else:
                failed.append(model)
        except Exception as e:
            print(f"\n  {model.upper()} SMOKE TEST FAILED: {e}")
            import traceback

            traceback.print_exc()
            failed.append(model)

    # Cleanup
    with contextlib.suppress(Exception):
        shutil.rmtree(tmp_dir)

    # Summary
    _print_section("SMOKE TEST SUMMARY")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
