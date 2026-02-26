"""TimesFM 2.5 univariate fine-tuning pipeline.

Fine-tunes the Google TimesFM 2.5 model using a direct PyTorch training loop.
Fine-tuning is strictly univariate; sensor covariates can only be used
at inference time via XReg.

The TimesFM v2 API does not include a built-in finetuning module, so this
pipeline implements teacher-forcing training directly on the underlying
nn.Module (TimesFM_2p5_200M_torch_module).

Requires: pip install timesfm (from source for v2 with torch support)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset

from tick2.data.preprocessing import TARGET_COL
from tick2.finetuning.base import FineTuneConfig, FineTuneResult, save_loss_history
from tick2.finetuning.data_prep import PreparedData

logger = logging.getLogger(__name__)

# TimesFM 2.5 constants
PATCH_SIZE = 32
OUTPUT_PATCH_LEN = 128
NUM_QUANTILES = 10
POINT_QUANTILE_IDX = 5  # index for the point forecast (median)


class TimesFMWindowDataset(Dataset):  # type: ignore[type-arg]
    """Sliding-window dataset for TimesFM teacher-forcing training.

    Each sample consists of context patches and a horizon target.
    The model predicts the next output_patch_len timesteps from each
    context patch position.
    """

    def __init__(
        self,
        series: np.ndarray,
        context_patches: int = 4,
        stride_patches: int = 1,
    ) -> None:
        """Initialize the dataset.

        Args:
            series: 1D time series array.
            context_patches: Number of context patches (x32 = timesteps).
            stride_patches: Stride in patches between windows.
        """
        self.patch_size = PATCH_SIZE
        self.output_len = OUTPUT_PATCH_LEN
        self.context_patches = context_patches

        # Total window: context + horizon (output_patch_len timesteps)
        self.context_len = context_patches * PATCH_SIZE
        self.total_len = self.context_len + self.output_len

        series = series.astype(np.float32)
        stride = stride_patches * PATCH_SIZE
        self.starts = list(range(0, len(series) - self.total_len + 1, stride))
        self.series = series

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = self.starts[idx]
        window = self.series[start : start + self.total_len]

        # Context: reshape into patches
        context = window[: self.context_len].reshape(self.context_patches, PATCH_SIZE)
        # Horizon: next output_patch_len timesteps as flat target
        horizon = window[self.context_len : self.context_len + self.output_len]

        return {
            "context": torch.tensor(context),
            "masks": torch.ones_like(torch.tensor(context)),
            "horizon": torch.tensor(horizon),
        }


def finetune_timesfm(
    prepared: dict[str, PreparedData],
    config: FineTuneConfig,
    output_dir: str | Path,
    training_mode: str = "combined",
    model_id: str = "google/timesfm-2.5-200m-pytorch",
    learning_rate: float = 1e-4,
    num_epochs: int = 50,
    batch_size: int = 64,
    context_length: int = 128,
    horizon_length: int = 32,
    weight_decay: float = 0.01,
    quantile_loss: bool = True,
) -> list[FineTuneResult]:
    """Fine-tune TimesFM 2.5 on univariate clock drift data.

    Uses teacher-forcing: for each context window, the model predicts the
    next output_patch_len (128) timesteps, and we compute MSE loss against
    the actual values.

    Args:
        prepared: Dict of machine name -> PreparedData.
        config: Shared fine-tuning configuration.
        output_dir: Directory to save fine-tuned checkpoints.
        training_mode: "combined" (all machines) or "per_machine".
        model_id: HuggingFace model ID for TimesFM.
        learning_rate: Learning rate.
        num_epochs: Number of training epochs.
        batch_size: Training batch size.
        context_length: Context length in patches (x32 = timesteps).
        horizon_length: Not used (horizon fixed at 128 timesteps).
        weight_decay: Weight decay for optimizer.
        quantile_loss: Whether to train the quantile head.

    Returns:
        List of FineTuneResult objects.
    """
    import timesfm

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[FineTuneResult] = []

    if training_mode == "combined":
        runs = [("combined", prepared)]
    else:
        runs = [(name, {name: p}) for name, p in prepared.items()]

    for run_name, run_data in runs:
        logger.info("TimesFM FT: %s (mode=%s)", run_name, training_mode)
        run_dir = output_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        # Collect univariate training/validation series
        train_series: list[np.ndarray] = []
        val_series: list[np.ndarray] = []
        for _name, p in run_data.items():
            train_series.append(p.split.train[TARGET_COL].to_numpy(dtype=np.float32))
            val_series.append(p.split.val[TARGET_COL].to_numpy(dtype=np.float32))

        # Concatenate all series
        train_data = np.concatenate(train_series)
        val_data = np.concatenate(val_series) if val_series else train_data

        # Create datasets
        train_ds = TimesFMWindowDataset(
            train_data,
            context_patches=context_length,
            stride_patches=max(1, context_length // 4),
        )
        val_ds = TimesFMWindowDataset(
            val_data,
            context_patches=context_length,
            stride_patches=max(1, context_length // 2),
        )

        if len(train_ds) == 0:
            logger.warning(
                "Not enough data for TimesFM FT on %s (need %d rows, have %d)",
                run_name,
                train_ds.total_len,
                len(train_data),
            )
            continue

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=0
        )

        # Load base model
        model_wrapper = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_id)
        model = model_wrapper.model  # The underlying nn.Module
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.train()

        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        best_val_loss = float("inf")
        patience_counter = 0
        patience = max(5, num_epochs // 5)
        train_losses: list[float] = []
        val_losses: list[float] = []

        t0 = time.perf_counter()

        for epoch in range(num_epochs):
            # Training
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                ctx = batch["context"].to(device)
                masks = batch["masks"].to(device)
                horizon = batch["horizon"].to(device)

                optimizer.zero_grad()

                # Forward pass: teacher-forcing on context patches
                (_, _, output_ts, _), _ = model(ctx, masks)

                # output_ts shape: (batch, n_patches, 128*10=1280)
                # Reshape to (batch, n_patches, 128, 10) and select point forecast
                bs = output_ts.shape[0]
                output_reshaped = output_ts[:, -1, :].reshape(
                    bs, OUTPUT_PATCH_LEN, NUM_QUANTILES
                )
                predicted = output_reshaped[:, :, POINT_QUANTILE_IDX]  # (batch, 128)

                loss = functional.mse_loss(predicted, horizon)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
                    ctx = batch["context"].to(device)
                    masks = batch["masks"].to(device)
                    horizon = batch["horizon"].to(device)

                    (_, _, output_ts, _), _ = model(ctx, masks)
                    bs = output_ts.shape[0]
                    output_reshaped = output_ts[:, -1, :].reshape(
                        bs, OUTPUT_PATCH_LEN, NUM_QUANTILES
                    )
                    predicted = output_reshaped[:, :, POINT_QUANTILE_IDX]
                    loss = functional.mse_loss(predicted, horizon)
                    val_loss += loss.item()
                    n_val_batches += 1

            avg_val_loss = val_loss / max(n_val_batches, 1)
            val_losses.append(avg_val_loss)

            logger.info(
                "Epoch %d/%d: train_loss=%.6f, val_loss=%.6f",
                epoch + 1,
                num_epochs,
                avg_train_loss,
                avg_val_loss,
            )

            # Persist loss history incrementally (survives crashes)
            save_loss_history(run_dir, train_losses, val_losses, run_name=run_name)

            # Early stopping and checkpointing
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_path = run_dir / "best_model.pt"
                torch.save(model.state_dict(), best_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        training_time = time.perf_counter() - t0
        best_epoch = int(np.argmin(val_losses)) if val_losses else 0

        total_train_rows = sum(len(p.split.train) for p in run_data.values())

        results.append(
            FineTuneResult(
                model_name="timesfm-2.5-ft",
                machine=run_name,
                train_loss=train_losses,
                val_loss=val_losses,
                best_epoch=best_epoch,
                training_time_s=training_time,
                checkpoint_path=str(run_dir),
                config={
                    "model_id": model_id,
                    "learning_rate": learning_rate,
                    "num_epochs": num_epochs,
                    "batch_size": batch_size,
                    "context_length": context_length,
                    "horizon_length": horizon_length,
                    "weight_decay": weight_decay,
                    "quantile_loss": quantile_loss,
                    "freq": 0,
                    "training_mode": training_mode,
                    "train_rows": total_train_rows,
                    "n_series": len(train_series),
                },
            )
        )

        logger.info(
            "TimesFM FT %s: %.1fs, %d series",
            run_name,
            training_time,
            len(train_series),
        )

    return results


def load_finetuned_timesfm(
    checkpoint_path: str | Path,
    model_id: str = "google/timesfm-2.5-200m-pytorch",
) -> object:
    """Load a fine-tuned TimesFM model from checkpoint.

    Args:
        checkpoint_path: Path to saved model directory.
        model_id: Base model ID.

    Returns:
        Loaded TimesFM model with fine-tuned weights.
    """
    import timesfm

    model_wrapper = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_id)

    # Load fine-tuned weights
    checkpoint_path = Path(checkpoint_path)
    ckpt_file = checkpoint_path / "best_model.pt"
    if not ckpt_file.exists():
        # Try other common names
        for name in ["model.pt", "checkpoint.pt"]:
            candidate = checkpoint_path / name
            if candidate.exists():
                ckpt_file = candidate
                break
        # Also check for .pth files
        pth_files = list(checkpoint_path.glob("*.pt")) + list(
            checkpoint_path.glob("*.pth")
        )
        if pth_files:
            ckpt_file = pth_files[0]

    if ckpt_file.exists():
        state_dict = torch.load(str(ckpt_file), map_location="cpu")
        model_wrapper.model.load_state_dict(state_dict, strict=True)
        logger.info("Loaded fine-tuned weights from %s", ckpt_file)

    return model_wrapper
