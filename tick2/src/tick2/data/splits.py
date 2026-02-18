"""Time-series-aware data splitting strategies.

All splits are contiguous (no random shuffling) to respect temporal ordering.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TemporalSplit:
    """Result of a temporal train/val/test split."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def temporal_split(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> TemporalSplit:
    """Split a DataFrame into contiguous train/val/test blocks.

    Args:
        df: Input DataFrame (must be sorted by time).
        train_frac: Fraction for training.
        val_frac: Fraction for validation.
        test_frac: Fraction for testing.

    Returns:
        TemporalSplit with train, val, test DataFrames.
    """
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    return TemporalSplit(
        train=df.iloc[:train_end].copy(),
        val=df.iloc[train_end:val_end].copy(),
        test=df.iloc[val_end:].copy(),
    )


@dataclass
class SampleWindow:
    """A single (context, horizon) window for benchmarking."""

    context: pd.DataFrame
    horizon_true: pd.Series  # target values over the horizon
    horizon_df: pd.DataFrame  # full horizon DataFrame (for future covariates)
    start_idx: int
    context_len: int
    horizon_len: int


def extract_samples(
    df: pd.DataFrame,
    target_col: str,
    context_len: int,
    horizon_len: int,
    n_samples: int = 25,
    seed: int = 42,
) -> list[SampleWindow]:
    """Extract random (context, horizon) windows for benchmarking.

    Matches Tick 1 Section 4.2 methodology: random starting points, fixed
    context and horizon lengths.

    Args:
        df: Input DataFrame sorted by time.
        target_col: Name of the target column.
        context_len: Number of timesteps in context window.
        horizon_len: Number of timesteps to predict.
        n_samples: Number of random windows to extract.
        seed: Random seed for reproducibility.

    Returns:
        List of SampleWindow objects.
    """
    n = len(df)
    min_start = 0
    max_start = n - context_len - horizon_len

    if max_start <= min_start:
        raise ValueError(
            f"DataFrame too short ({n} rows) for context_len={context_len} "
            f"+ horizon_len={horizon_len}"
        )

    rng = np.random.default_rng(seed)
    starts = rng.integers(min_start, max_start, size=n_samples)
    starts = np.sort(starts)  # sort for cache-friendly access

    samples: list[SampleWindow] = []
    for start in starts:
        ctx_end = start + context_len
        hz_end = ctx_end + horizon_len

        context = df.iloc[start:ctx_end].copy()
        horizon_true = df[target_col].iloc[ctx_end:hz_end].copy()
        horizon_df = df.iloc[ctx_end:hz_end].copy()

        samples.append(
            SampleWindow(
                context=context,
                horizon_true=horizon_true,
                horizon_df=horizon_df,
                start_idx=int(start),
                context_len=context_len,
                horizon_len=horizon_len,
            )
        )

    return samples


def rolling_origin_cv(
    df: pd.DataFrame,
    n_folds: int = 5,
    min_train_rows: int = 3600,
    horizon_len: int = 60,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Expanding-window cross-validation splits.

    Each fold uses all data up to a cutpoint for training, and the next
    horizon_len rows for testing. Cutpoints are evenly spaced.

    Args:
        df: Input DataFrame sorted by time.
        n_folds: Number of cross-validation folds.
        min_train_rows: Minimum number of training rows.
        horizon_len: Number of rows in each test fold.

    Returns:
        List of (train_df, test_df) tuples.
    """
    n = len(df)
    max_cutpoint = n - horizon_len
    step = (max_cutpoint - min_train_rows) // n_folds

    if step <= 0:
        raise ValueError(
            f"Not enough data ({n} rows) for {n_folds} folds with "
            f"min_train={min_train_rows} and horizon={horizon_len}"
        )

    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for i in range(n_folds):
        cutpoint = min_train_rows + i * step
        train = df.iloc[:cutpoint].copy()
        test = df.iloc[cutpoint : cutpoint + horizon_len].copy()
        folds.append((train, test))

    return folds


def leave_one_machine_out(
    datasets: dict[str, pd.DataFrame],
) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """Leave-one-machine-out splits for cross-machine generalization.

    Args:
        datasets: Dict mapping machine name -> DataFrame.

    Returns:
        List of (test_machine_name, train_df, test_df) tuples.
        train_df is the concatenation of all other machines.
    """
    machines = list(datasets.keys())
    splits: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []

    for test_machine in machines:
        train_dfs = [datasets[m] for m in machines if m != test_machine]
        train_df = pd.concat(train_dfs, ignore_index=True)
        test_df = datasets[test_machine].copy()
        splits.append((test_machine, train_df, test_df))

    return splits
