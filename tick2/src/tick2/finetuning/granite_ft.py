"""Granite TTM channel-mix fine-tuning pipeline.

Fine-tunes the IBM Granite TTM model using HuggingFace Trainer with
channel-mix decoder mode for multivariate covariate utilization.

Requires: pip install "granite-tsfm>=0.3.3" transformers datasets
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tick2.data.preprocessing import TARGET_COL
from tick2.finetuning.base import FineTuneConfig, FineTuneResult, save_loss_history
from tick2.finetuning.data_prep import (
    PreparedData,
    combine_training_data,
    combine_validation_data,
)

logger = logging.getLogger(__name__)


def _patch_tsfm_readonly() -> None:
    """Monkey-patch tsfm_public to fix read-only array bug with pandas 2.x CoW.

    tsfm_public's ForecastDFDataset.__getitem__ calls `self.y.iloc[...].values`
    which returns read-only views in pandas 2.x, then tries to write in-place.
    This patches the __getitem__ to add .copy() to the seq_y creation.
    """
    try:
        from tsfm_public.toolkit.dataset import ForecastDFDataset

        inner_cls = ForecastDFDataset.BaseForecastDFDataset
        original_getitem = inner_cls.__getitem__

        if getattr(original_getitem, "_patched_readonly", False):
            return  # Already patched

        def patched_getitem(self: object, index: int) -> dict:  # type: ignore[type-arg]
            result = original_getitem(self, index)
            return result

        # Instead of wrapping __getitem__, directly patch the y DataFrame
        original_init = inner_cls.__init__

        def patched_init(self: object, *args: object, **kwargs: object) -> None:
            original_init(self, *args, **kwargs)  # type: ignore[misc]
            # Make y DataFrame writable after init
            if hasattr(self, "y") and isinstance(self.y, pd.DataFrame):
                y_df = self.y
                for col in y_df.columns:
                    y_df[col] = y_df[col].to_numpy().copy()

        patched_init._patched_readonly = True  # type: ignore[attr-defined]
        if not getattr(inner_cls.__init__, "_patched_readonly", False):
            inner_cls.__init__ = patched_init  # type: ignore[misc]

    except ImportError:
        pass


# Context/prediction pairs supported by TTM r2 branches
SUPPORTED_PAIRS: list[tuple[int, int]] = [
    (512, 96),
    (1024, 96),
    (1536, 96),
    (512, 192),
    (1024, 192),
    (1536, 192),
]


def finetune_granite(
    prepared: dict[str, PreparedData],
    config: FineTuneConfig,
    output_dir: str | Path,
    training_mode: str = "combined",
    decoder_mode: str = "mix_channel",
    freeze_backbone: bool = True,
    learning_rate: float = 0.001,
    num_epochs: int = 50,
    batch_size: int = 64,
    early_stopping_patience: int = 10,
) -> list[FineTuneResult]:
    """Fine-tune Granite TTM with channel-mix decoder for multivariate prediction.

    Args:
        prepared: Dict of machine name -> PreparedData.
        config: Shared fine-tuning configuration.
        output_dir: Directory to save fine-tuned checkpoints.
        training_mode: "combined" (all machines) or "per_machine".
        decoder_mode: "mix_channel" for multivariate, "common_channel" for univariate.
        freeze_backbone: If True, only train the decoder (few-shot). If False, full FT.
        learning_rate: Learning rate for the optimizer.
        num_epochs: Maximum training epochs.
        batch_size: Per-device training batch size.
        early_stopping_patience: Early stopping patience (epochs).

    Returns:
        List of FineTuneResult objects (one per training run).
    """
    from transformers import EarlyStoppingCallback, Trainer, TrainingArguments
    from tsfm_public import TimeSeriesPreprocessor, get_datasets
    from tsfm_public.toolkit.get_model import get_model

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ctx_len = config.context_length
    pred_len = config.prediction_length

    # Validate context/prediction pair
    pair = (ctx_len, pred_len)
    if pair not in SUPPORTED_PAIRS:
        closest = min(
            SUPPORTED_PAIRS,
            key=lambda p: abs(p[0] - ctx_len) + abs(p[1] - pred_len),
        )
        logger.warning(
            "TTM pair %s not supported, using %s",
            pair,
            closest,
        )
        ctx_len, pred_len = closest

    results: list[FineTuneResult] = []

    if training_mode == "combined":
        runs = [("combined", prepared)]
    else:
        runs = [(name, {name: p}) for name, p in prepared.items()]

    for run_name, run_data in runs:
        logger.info("Granite FT: %s (mode=%s)", run_name, training_mode)
        run_dir = output_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        # Prepare combined DataFrames
        if training_mode == "combined":
            train_df, shared_features = combine_training_data(run_data)
            val_df, _ = combine_validation_data(run_data)
        else:
            p = next(iter(run_data.values()))
            shared_features = p.feature_cols
            train_df = p.split.train[[*shared_features, TARGET_COL]].copy()
            val_df = p.split.val[[*shared_features, TARGET_COL]].copy()

        # Cap features
        if len(shared_features) > config.max_covariates:
            shared_features = shared_features[: config.max_covariates]
            keep = [*shared_features, TARGET_COL]
            train_df = train_df[keep]
            val_df = val_df[keep]

        # Reset index to integer for TTM preprocessor
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

        # Add timestamp column (TTM expects it)
        train_ts = pd.date_range("2024-01-01", periods=len(train_df), freq="1s")
        val_ts = pd.date_range("2024-06-01", periods=len(val_df), freq="1s")
        train_df.insert(0, "datetime", train_ts)
        val_df.insert(0, "datetime", val_ts)

        # Determine covariate columns
        conditional_cols = shared_features if decoder_mode == "mix_channel" else []

        # Create preprocessor
        tsp = TimeSeriesPreprocessor(
            timestamp_column="datetime",
            id_columns=[],
            target_columns=[TARGET_COL],
            conditional_columns=conditional_cols,
            context_length=ctx_len,
            prediction_length=pred_len,
            scaling=True,
            scaler_type="standard",
        )

        # Build datasets — ensure arrays are writable (tsfm_public writes in-place)
        combined_df = pd.concat([train_df, val_df], ignore_index=True)
        for col in combined_df.select_dtypes(include="number").columns:
            combined_df[col] = combined_df[col].to_numpy(copy=True)
        train_frac = len(train_df) / len(combined_df)
        val_start = train_frac

        _patch_tsfm_readonly()
        train_ds, val_ds, _ = get_datasets(
            tsp,
            combined_df,
            {
                "train": [0, train_frac],
                "valid": [val_start, 1.0],
                "test": [val_start, 1.0],  # not used, but required
            },
        )

        # Load model
        model = get_model(
            "ibm-granite/granite-timeseries-ttm-r2",
            context_length=ctx_len,
            prediction_length=pred_len,
            num_input_channels=tsp.num_input_channels,
            prediction_channel_indices=list(tsp.prediction_channel_indices),
            exogenous_channel_indices=list(tsp.exogenous_channel_indices),
            decoder_mode=decoder_mode,
        )

        # Optionally freeze backbone
        if freeze_backbone:
            for param in model.backbone.parameters():
                param.requires_grad = False

        # Training arguments
        args = TrainingArguments(
            output_dir=str(run_dir / "checkpoints"),
            learning_rate=learning_rate,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=3,
            logging_steps=50,
            report_to="none",
            seed=config.seed,
        )

        callbacks = [
            EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)
        ]

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            callbacks=callbacks,
        )

        # Train
        t0 = time.perf_counter()
        trainer.train()
        training_time = time.perf_counter() - t0

        # Extract loss history from trainer log
        train_losses: list[float] = []
        val_losses: list[float] = []
        for entry in trainer.state.log_history:
            if "loss" in entry and "eval_loss" not in entry:
                train_losses.append(float(entry["loss"]))
            if "eval_loss" in entry:
                val_losses.append(float(entry["eval_loss"]))

        best_epoch = int(trainer.state.best_metric) if trainer.state.best_metric else 0
        if val_losses:
            best_epoch = int(np.argmin(val_losses))

        # Persist loss history to CSV
        if train_losses or val_losses:
            save_loss_history(run_dir, train_losses, val_losses, run_name=run_name)

        # Save best model
        best_dir = run_dir / "best"
        best_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(best_dir))

        results.append(
            FineTuneResult(
                model_name="granite-ttm-ft",
                machine=run_name,
                train_loss=train_losses,
                val_loss=val_losses,
                best_epoch=best_epoch,
                training_time_s=training_time,
                checkpoint_path=str(best_dir),
                config={
                    "context_length": ctx_len,
                    "prediction_length": pred_len,
                    "decoder_mode": decoder_mode,
                    "freeze_backbone": freeze_backbone,
                    "learning_rate": learning_rate,
                    "num_epochs": num_epochs,
                    "batch_size": batch_size,
                    "n_features": len(shared_features),
                    "training_mode": training_mode,
                    "train_rows": len(train_df),
                },
            )
        )

        logger.info(
            "Granite FT %s: %.1fs, best_epoch=%d, val_loss=%.6f",
            run_name,
            training_time,
            best_epoch,
            val_losses[best_epoch] if val_losses else float("nan"),
        )

    return results


def load_finetuned_granite(
    checkpoint_path: str | Path,
    context_length: int = 512,
    prediction_length: int = 96,
) -> object:
    """Load a fine-tuned Granite TTM model from checkpoint.

    Args:
        checkpoint_path: Path to saved model directory.
        context_length: Context length the model was trained with.
        prediction_length: Prediction length the model was trained with.

    Returns:
        Loaded TinyTimeMixerForPrediction model.
    """
    from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction

    model = TinyTimeMixerForPrediction.from_pretrained(str(checkpoint_path))
    model.eval()
    return model
