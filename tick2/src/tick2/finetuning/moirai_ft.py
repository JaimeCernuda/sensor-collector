"""Moirai 1.1 fine-tuning pipeline.

Fine-tunes the Salesforce Moirai model using PyTorch Lightning. Since
uni2ts SimpleDatasetBuilder lacks target/covariate separation, we use
wide_multivariate mode where all channels (target + sensors) are predicted
jointly, then extract only the target channel at inference.

Requires: pip install uni2ts pytorch-lightning
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from tick2.data.preprocessing import TARGET_COL
from tick2.finetuning.base import FineTuneConfig, FineTuneResult, save_loss_history
from tick2.finetuning.data_prep import (
    PreparedData,
    combine_training_data,
    combine_validation_data,
)

logger = logging.getLogger(__name__)


class MoiraiWindowDataset(Dataset):  # type: ignore[type-arg]
    """Sliding-window dataset for Moirai fine-tuning.

    Creates (context, horizon) windows from a multivariate DataFrame,
    returning tensors in the format expected by MoiraiForecast._val_loss().

    The full sequence (past+future) is returned as ``target`` so that
    ``_val_loss`` can split context/horizon internally.
    """

    def __init__(
        self,
        data: np.ndarray,
        context_length: int,
        prediction_length: int,
        stride: int = 1,
    ) -> None:
        """Initialize the dataset.

        Args:
            data: Multivariate array, shape (n_timesteps, n_channels).
                  Channel 0 is the target; remaining are covariates.
            context_length: Number of historical timesteps.
            prediction_length: Number of future timesteps to predict.
            stride: Step size between windows.
        """
        self.data = data.astype(np.float32)
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.total_length = context_length + prediction_length

        n_rows = len(data)
        self.starts = list(range(0, n_rows - self.total_length + 1, stride))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = self.starts[idx]
        end = start + self.total_length
        window = self.data[start:end]  # (total_length, n_channels)

        n_channels = window.shape[1]

        # Full sequence target (past + future), shape (total_length, 1)
        target = torch.tensor(window[:, 0:1])
        observed_target = torch.ones_like(target, dtype=torch.bool)
        is_pad = torch.zeros(self.total_length, dtype=torch.bool)

        result: dict[str, torch.Tensor] = {
            "target": target,
            "observed_target": observed_target,
            "is_pad": is_pad,
        }

        # Past-only covariates (channels 1+), shape (context_length, n_cov)
        if n_channels > 1:
            ctx = window[: self.context_length]
            past_feat = torch.tensor(ctx[:, 1:])
            result["past_feat_dynamic_real"] = past_feat
            result["past_observed_feat_dynamic_real"] = torch.ones_like(
                past_feat, dtype=torch.bool
            )

        return result


def _build_multivariate_array(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> np.ndarray:
    """Build multivariate array with target as first column.

    Args:
        df: DataFrame with target and feature columns.
        feature_cols: Feature column names to include.

    Returns:
        Array of shape (n_rows, 1 + n_features).
    """
    target = df[TARGET_COL].to_numpy(dtype=np.float32).reshape(-1, 1)
    available = [c for c in feature_cols if c in df.columns]

    if available:
        features = df[available].to_numpy(dtype=np.float32)
        return np.hstack([target, features])
    return target


def finetune_moirai(
    prepared: dict[str, PreparedData],
    config: FineTuneConfig,
    output_dir: str | Path,
    training_mode: str = "combined",
    model_id: str = "Salesforce/moirai-1.1-R-small",
    patch_size: int = 32,
    max_epochs: int = 20,
    learning_rate: float = 1e-4,
    batch_size: int = 32,
    early_stopping_patience: int = 5,
    num_samples: int = 100,
    stride: int = 64,
    device: str = "auto",
) -> list[FineTuneResult]:
    """Fine-tune Moirai 1.1 using PyTorch Lightning.

    Uses wide_multivariate approach: all channels (target + top-K sensors)
    are treated as joint targets during training. At inference, only the
    target channel prediction is extracted.

    Args:
        prepared: Dict of machine name -> PreparedData.
        config: Shared fine-tuning configuration.
        output_dir: Directory to save fine-tuned checkpoints.
        training_mode: "combined" (all machines) or "per_machine".
        model_id: HuggingFace model ID for Moirai.
        patch_size: Patch size for the model.
        max_epochs: Maximum training epochs.
        learning_rate: Learning rate.
        batch_size: Training batch size.
        early_stopping_patience: Early stopping patience.
        num_samples: Number of forecast samples for probabilistic output.
        stride: Stride for sliding window dataset.
        device: Device for training ("auto", "cuda", "cpu").

    Returns:
        List of FineTuneResult objects.
    """
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ctx_len = config.context_length
    pred_len = config.prediction_length

    results: list[FineTuneResult] = []

    if training_mode == "combined":
        runs = [("combined", prepared)]
    else:
        runs = [(name, {name: p}) for name, p in prepared.items()]

    for run_name, run_data in runs:
        logger.info("Moirai FT: %s (mode=%s)", run_name, training_mode)
        run_dir = output_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        # Build multivariate arrays
        if training_mode == "combined":
            train_df, shared_features = combine_training_data(run_data)
            val_df, _ = combine_validation_data(run_data)
        else:
            p = next(iter(run_data.values()))
            shared_features = p.feature_cols
            train_df = p.split.train[[*shared_features, TARGET_COL]].copy()
            val_df = p.split.val[[*shared_features, TARGET_COL]].copy()

        if len(shared_features) > config.max_covariates:
            shared_features = shared_features[: config.max_covariates]

        train_arr = _build_multivariate_array(train_df, shared_features)
        val_arr = _build_multivariate_array(val_df, shared_features)

        n_channels = train_arr.shape[1]
        n_cov = n_channels - 1

        # Create datasets
        train_ds = MoiraiWindowDataset(train_arr, ctx_len, pred_len, stride=stride)
        val_ds = MoiraiWindowDataset(val_arr, ctx_len, pred_len, stride=stride)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        # Load pretrained module
        module = MoiraiModule.from_pretrained(model_id)

        # Create forecast model
        model = MoiraiForecast(
            module=module,
            prediction_length=pred_len,
            context_length=ctx_len,
            patch_size=patch_size,
            num_samples=num_samples,
            target_dim=1,
            feat_dynamic_real_dim=0,  # no known-future covariates
            past_feat_dynamic_real_dim=n_cov if n_cov > 0 else 0,
        )

        # Configure optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        # Determine patch size for _val_loss
        ps = patch_size if patch_size != "auto" else model.max_patch_size

        # Manual training loop using _val_loss (MoiraiForecast wraps the module)
        model = model.to(device)
        model.train()

        best_val_loss = float("inf")
        patience_counter = 0
        train_losses: list[float] = []
        val_losses: list[float] = []

        t0 = time.perf_counter()

        for epoch in range(max_epochs):
            # Training
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad()

                # _val_loss computes the variational loss over the full sequence
                per_sample_loss = model._val_loss(
                    patch_size=ps,
                    target=batch["target"],
                    observed_target=batch["observed_target"],
                    is_pad=batch["is_pad"],
                    past_feat_dynamic_real=batch.get("past_feat_dynamic_real"),
                    past_observed_feat_dynamic_real=batch.get(
                        "past_observed_feat_dynamic_real"
                    ),
                )
                loss = per_sample_loss.mean()

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)
            train_losses.append(avg_train_loss)

            # Validation
            model.eval()
            val_loss = 0.0
            n_val_batches = 0

            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    per_sample_loss = model._val_loss(
                        patch_size=ps,
                        target=batch["target"],
                        observed_target=batch["observed_target"],
                        is_pad=batch["is_pad"],
                        past_feat_dynamic_real=batch.get("past_feat_dynamic_real"),
                        past_observed_feat_dynamic_real=batch.get(
                            "past_observed_feat_dynamic_real"
                        ),
                    )
                    val_loss += per_sample_loss.mean().item()
                    n_val_batches += 1

            avg_val_loss = val_loss / max(n_val_batches, 1)
            val_losses.append(avg_val_loss)

            logger.info(
                "Epoch %d/%d: train_loss=%.6f, val_loss=%.6f",
                epoch + 1,
                max_epochs,
                avg_train_loss,
                avg_val_loss,
            )

            # Persist loss history incrementally (survives crashes)
            save_loss_history(run_dir, train_losses, val_losses, run_name=run_name)

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best checkpoint
                best_dir = run_dir / "best"
                best_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), best_dir / "model.pt")
                # Also save module weights for loading
                model.module.save_pretrained(str(best_dir / "module"))
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        training_time = time.perf_counter() - t0
        best_epoch = int(np.argmin(val_losses)) if val_losses else 0

        results.append(
            FineTuneResult(
                model_name="moirai-1.1-ft",
                machine=run_name,
                train_loss=train_losses,
                val_loss=val_losses,
                best_epoch=best_epoch,
                training_time_s=training_time,
                checkpoint_path=str(run_dir / "best"),
                config={
                    "model_id": model_id,
                    "patch_size": patch_size,
                    "context_length": ctx_len,
                    "prediction_length": pred_len,
                    "max_epochs": max_epochs,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "n_channels": n_channels,
                    "stride": stride,
                    "training_mode": training_mode,
                    "train_rows": len(train_arr),
                },
            )
        )

        logger.info(
            "Moirai FT %s: %.1fs, best_epoch=%d, val_loss=%.6f",
            run_name,
            training_time,
            best_epoch,
            val_losses[best_epoch] if val_losses else float("nan"),
        )

    return results


def load_finetuned_moirai(
    checkpoint_path: str | Path,
    model_id: str = "Salesforce/moirai-1.1-R-small",
    context_length: int = 1024,
    prediction_length: int = 96,
    patch_size: int = 32,
    num_samples: int = 100,
    n_covariates: int = 0,
) -> object:
    """Load a fine-tuned Moirai model from checkpoint.

    Args:
        checkpoint_path: Path to saved model directory.
        model_id: Base model ID (for architecture config).
        context_length: Context length used during training.
        prediction_length: Prediction length used during training.
        patch_size: Patch size used during training.
        num_samples: Number of forecast samples.
        n_covariates: Number of covariate channels.

    Returns:
        Loaded MoiraiForecast model.
    """
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

    checkpoint_path = Path(checkpoint_path)

    # Load module from saved pretrained weights
    module_path = checkpoint_path / "module"
    if module_path.exists():
        module = MoiraiModule.from_pretrained(str(module_path))
    else:
        module = MoiraiModule.from_pretrained(model_id)

    model = MoiraiForecast(
        module=module,
        prediction_length=prediction_length,
        context_length=context_length,
        patch_size=patch_size,
        num_samples=num_samples,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=n_covariates,
    )

    # Load state dict if available
    state_path = checkpoint_path / "model.pt"
    if state_path.exists():
        model.load_state_dict(torch.load(str(state_path), map_location="cpu"))

    model.eval()
    return model
