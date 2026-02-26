"""Common types and configuration for fine-tuning pipelines."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class FineTuneConfig:
    """Shared configuration for all fine-tuning pipelines.

    Attributes:
        train_frac: Fraction of data for training (temporal split).
        val_frac: Fraction of data for validation.
        test_frac: Fraction of data for testing.
        context_length: Number of historical timesteps as model input.
        prediction_length: Number of future timesteps to predict.
        max_covariates: Maximum covariate features to retain (OOM guard).
        seed: Random seed for reproducibility.
    """

    train_frac: float = 0.7
    val_frac: float = 0.15
    test_frac: float = 0.15
    context_length: int = 1024
    prediction_length: int = 96
    max_covariates: int = 30
    seed: int = 42


@dataclass
class FineTuneResult:
    """Result of a single fine-tuning run.

    Attributes:
        model_name: Name of the model that was fine-tuned.
        machine: Machine name(s) the model was trained on.
        train_loss: Per-epoch training loss history.
        val_loss: Per-epoch validation loss history.
        best_epoch: Epoch index with the lowest validation loss.
        training_time_s: Total wall-clock training time in seconds.
        checkpoint_path: Path to the saved fine-tuned checkpoint.
        config: Full hyperparameter dict for reproducibility.
    """

    model_name: str
    machine: str
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_epoch: int = 0
    training_time_s: float = 0.0
    checkpoint_path: str = ""
    config: dict[str, Any] = field(default_factory=dict)


def save_loss_history(
    output_dir: Path,
    train_loss: list[float],
    val_loss: list[float],
    run_name: str = "",
) -> Path:
    """Write train/val loss per epoch to a CSV file.

    Called incrementally (appends one row per epoch) so partial results
    survive crashes.  If the file already exists the header is not
    duplicated.

    Args:
        output_dir: Directory where the CSV will be written.
        train_loss: All training losses accumulated so far.
        val_loss: All validation losses accumulated so far.
        run_name: Optional label prepended to the filename.

    Returns:
        Path to the loss history CSV.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{run_name}_" if run_name else ""
    csv_path = output_dir / f"{prefix}loss_history.csv"

    # Determine how many rows are already on disk
    existing_rows = 0
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            existing_rows = max(sum(1 for _ in f) - 1, 0)  # minus header

    n_new = len(train_loss)
    if n_new <= existing_rows:
        return csv_path  # nothing to append

    write_header = not csv_path.exists() or existing_rows == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["epoch", "train_loss", "val_loss"])
        for i in range(existing_rows, n_new):
            vl = val_loss[i] if i < len(val_loss) else ""
            writer.writerow([i + 1, train_loss[i], vl])

    logger.info("Loss history (%d epochs) written to %s", n_new, csv_path)
    return csv_path
