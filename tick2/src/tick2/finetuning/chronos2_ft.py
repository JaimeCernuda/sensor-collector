"""Chronos-2 fine-tuning pipeline (LoRA and full).

Fine-tunes the Amazon Chronos-2 model using the native ``pipeline.fit()``
API with support for covariates via past/future covariate dicts.

Requires: pip install "chronos-forecasting[extras]>=2.2"
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from tick2.data.preprocessing import TARGET_COL
from tick2.finetuning.base import FineTuneConfig, FineTuneResult
from tick2.finetuning.data_prep import PreparedData

logger = logging.getLogger(__name__)


def _build_inputs(
    prepared: dict[str, PreparedData],
    split_name: str = "train",
    with_covariates: bool = True,
    shared_features: list[str] | None = None,
) -> list[dict[str, object]]:
    """Build Chronos-2 fine-tuning input dicts from prepared data.

    Each input dict has:
    - "target": 1D numpy array of target values
    - "past_covariates" (optional): dict of covariate name -> 1D numpy array

    Args:
        prepared: Dict of machine name -> PreparedData.
        split_name: Which split to use ("train", "val", "test").
        with_covariates: Whether to include sensor covariates.
        shared_features: If set, use only these features (for combined mode).

    Returns:
        List of input dicts for pipeline.fit().
    """
    inputs: list[dict[str, object]] = []

    for _name, p in prepared.items():
        df = getattr(p.split, split_name)
        target = df[TARGET_COL].to_numpy(dtype=np.float32)
        entry: dict[str, object] = {"target": target}

        if with_covariates:
            features = shared_features if shared_features else p.feature_cols
            available = [c for c in features if c in df.columns]
            if available:
                past_covs: dict[str, np.ndarray] = {}
                for col in available:
                    past_covs[col] = df[col].to_numpy(dtype=np.float32)
                entry["past_covariates"] = past_covs

        inputs.append(entry)

    return inputs


def finetune_chronos2(
    prepared: dict[str, PreparedData],
    config: FineTuneConfig,
    output_dir: str | Path,
    training_mode: str = "combined",
    model_id: str = "autogluon/chronos-2-small",
    finetune_mode: str = "lora",
    with_covariates: bool = True,
    learning_rate: float = 1e-5,
    num_steps: int = 1000,
    batch_size: int = 256,
    context_length: int | None = None,
    device_map: str = "auto",
) -> list[FineTuneResult]:
    """Fine-tune Chronos-2 using the native pipeline.fit() API.

    Args:
        prepared: Dict of machine name -> PreparedData.
        config: Shared fine-tuning configuration.
        output_dir: Directory to save fine-tuned checkpoints.
        training_mode: "combined" (all machines) or "per_machine".
        model_id: HuggingFace model ID for Chronos-2.
        finetune_mode: "lora" or "full".
        with_covariates: Whether to include sensor covariates during FT.
        learning_rate: Learning rate (1e-5 for LoRA, 1e-6 for full).
        num_steps: Number of training steps.
        batch_size: Training batch size.
        context_length: Override context length (None = auto).
        device_map: Device for training ("auto", "cuda", "cpu").

    Returns:
        List of FineTuneResult objects.
    """
    from chronos import BaseChronosPipeline

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_len = config.prediction_length
    ctx_len = context_length or config.context_length

    results: list[FineTuneResult] = []

    if training_mode == "combined":
        runs = [("combined", prepared)]
    else:
        runs = [(name, {name: p}) for name, p in prepared.items()]

    for run_name, run_data in runs:
        logger.info(
            "Chronos-2 FT: %s (mode=%s, ft=%s)",
            run_name,
            training_mode,
            finetune_mode,
        )
        run_dir = output_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        # Find shared features for combined mode
        shared_features = None
        if training_mode == "combined":
            feature_sets = [set(p.feature_cols) for p in run_data.values()]
            common = set.intersection(*feature_sets) if feature_sets else set()
            shared_features = sorted(common)
            if len(shared_features) > config.max_covariates:
                shared_features = shared_features[: config.max_covariates]

        # Build training inputs
        inputs = _build_inputs(
            run_data,
            split_name="train",
            with_covariates=with_covariates,
            shared_features=shared_features,
        )

        # Load base pipeline
        pipeline = BaseChronosPipeline.from_pretrained(
            model_id,
            device_map=device_map,
        )

        # Fine-tune
        t0 = time.perf_counter()

        ckpt_name = f"{run_name}_best"
        pipeline = pipeline.fit(
            inputs=inputs,
            prediction_length=pred_len,
            finetune_mode=finetune_mode,
            context_length=ctx_len,
            learning_rate=learning_rate,
            num_steps=num_steps,
            batch_size=batch_size,
            output_dir=str(run_dir),
            finetuned_ckpt_name=ckpt_name,
        )

        training_time = time.perf_counter() - t0

        # Determine checkpoint path
        ckpt_path = run_dir / ckpt_name

        # pipeline.fit() may only save LoRA adapter weights, which
        # BaseChronosPipeline.from_pretrained() cannot reload standalone.
        # Explicitly save the full (merged) model + tokenizer so the
        # checkpoint is self-contained and loadable.
        ckpt_path.mkdir(parents=True, exist_ok=True)
        if hasattr(pipeline, "model") and hasattr(pipeline.model, "save_pretrained"):
            pipeline.model.save_pretrained(str(ckpt_path))
            logger.info("Saved full model to %s", ckpt_path)
        if hasattr(pipeline, "tokenizer") and hasattr(
            pipeline.tokenizer, "save_pretrained"
        ):
            pipeline.tokenizer.save_pretrained(str(ckpt_path))

        total_train_rows = sum(len(p.split.train) for p in run_data.values())

        results.append(
            FineTuneResult(
                model_name=f"chronos2-ft-{finetune_mode}",
                machine=run_name,
                train_loss=[],  # Chronos doesn't expose per-step loss easily
                val_loss=[],
                best_epoch=0,
                training_time_s=training_time,
                checkpoint_path=str(ckpt_path),
                config={
                    "model_id": model_id,
                    "finetune_mode": finetune_mode,
                    "with_covariates": with_covariates,
                    "learning_rate": learning_rate,
                    "num_steps": num_steps,
                    "batch_size": batch_size,
                    "context_length": ctx_len,
                    "prediction_length": pred_len,
                    "training_mode": training_mode,
                    "train_rows": total_train_rows,
                },
            )
        )

        logger.info(
            "Chronos-2 FT %s: %.1fs, ckpt=%s",
            run_name,
            training_time,
            ckpt_path,
        )

    return results


def load_finetuned_chronos2(
    checkpoint_path: str | Path,
    device_map: str = "auto",
) -> object:
    """Load a fine-tuned Chronos-2 pipeline from checkpoint.

    Args:
        checkpoint_path: Path to saved fine-tuned model directory.
        device_map: Device mapping ("auto", "cuda", "cpu").

    Returns:
        Loaded BaseChronosPipeline.
    """
    from chronos import BaseChronosPipeline

    return BaseChronosPipeline.from_pretrained(
        str(checkpoint_path),
        device_map=device_map,
    )
