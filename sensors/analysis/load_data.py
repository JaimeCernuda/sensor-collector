"""Data loading, column deduplication, counter differencing, and feature categorization.

The raw 1 Hz CSVs are aggregated to 1-minute means after preprocessing.
This de-quantises the piecewise-constant NTP target and captures the
minute-scale thermal dynamics relevant to clock drift.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "8h_snapshot"

MACHINES = ["homelab", "chameleon", "ares", "ares-comp-10"]

# Temporal aggregation window (seconds)
AGG_SECONDS = 60

# Columns that are cumulative counters -> first-difference to get rates
CUMULATIVE_PATTERNS = [
    r"^cstate_",
    r"^rapl_",
    r"^net_",
    r"^disk_",
    r"^ctxt$",
    r"^intr$",
]

# Feature category assignment (order matters -- first match wins)
CATEGORY_RULES: list[tuple[str, str]] = [
    ("CPU Core Temp", r"temp_coretemp_Core|turbo_core_tmp"),
    ("CPU Package Temp", r"temp_coretemp_Package|turbo_pkg_tmp"),
    (
        "Non-CPU Temp",
        r"temp_(pch|nvme|acpitz)_|tz_|ipmi_(exhaust|inlet)_temp|ipmi_temp$",
    ),
    ("CPU Frequency", r"cpu\d+_freq_mhz|turbo_(avg|bzy|tsc)_mhz"),
    ("CPU Load", r"cpu_.*_pct|loadavg_|turbo_busy_pct|turbo_ipc"),
    ("Power", r"rapl_|ipmi_pwr|turbo_.*_watt"),
    ("C-State", r"cstate_|turbo_pkg_pc\d"),
    ("Memory", r"mem_"),
    ("I/O", r"disk_|net_"),
    ("System", r"ctxt|intr|procs_|turbo_(irq|smi)|ipmi_(fan|current|voltage)"),
    # Peer Clock excluded from analysis: correlated with target by construction
    # ("Peer Clock", r"peer_"),
]

# Columns excluded from features (targets or metadata)
EXCLUDE_PATTERNS = [
    r"^ts_",
    r"^mono_minus_raw",
    r"^adj_",
    r"^chrony_",
    r"_age_ms$",
]


def _deduplicate_columns(columns: list[str]) -> list[str]:
    """Rename duplicate column names -- second occurrence gets ``_sock1``."""
    seen: dict[str, int] = {}
    result: list[str] = []
    for col in columns:
        if col in seen:
            seen[col] += 1
            suffix = "_sock1" if seen[col] == 1 else f"_dup{seen[col]}"
            result.append(f"{col}{suffix}")
        else:
            seen[col] = 0
            result.append(col)
    return result


def _is_cumulative(col: str) -> bool:
    """Check whether a column holds cumulative counter data."""
    return any(re.search(p, col) for p in CUMULATIVE_PATTERNS)


def _is_excluded(col: str) -> bool:
    """Check whether a column should be excluded from features."""
    return any(re.search(p, col) for p in EXCLUDE_PATTERNS)


def categorize_column(col: str) -> str | None:
    """Assign a feature column to a category.  Returns ``None`` if unmatched."""
    for category, pattern in CATEGORY_RULES:
        if re.search(pattern, col):
            return category
    return None


def load_machine(
    name: str,
    data_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Load and preprocess data for a single machine.

    Pipeline: read CSV -> dedup columns -> derive target -> diff counters
    -> ffill sparse sensors -> select features -> aggregate to 1-min means
    -> drop zero-variance / all-NaN -> clean.

    Returns:
        Tuple of (DataFrame with features + target, dict mapping column -> category).
    """
    data_dir = data_dir or DEFAULT_DATA_DIR
    path = data_dir / f"{name}.csv"

    # Read raw header and deduplicate before pandas ingests it
    with open(path) as f:
        raw_header = f.readline().strip().split(",")
    deduped_header = _deduplicate_columns(raw_header)

    df = pd.read_csv(path, header=0, names=deduped_header, skiprows=1)

    # Derive target: adj_freq_ppm (kernel NTP frequency correction in PPM)
    df = pd.concat(
        [df, pd.Series(df["adj_freq"] / 65536.0, name="adj_freq_ppm")], axis=1
    )

    # First-difference cumulative counters -> per-second rates
    cumulative_cols = [c for c in df.columns if _is_cumulative(c)]
    for col in cumulative_cols:
        df[col] = df[col].diff()

    # Drop first row (NaN from differencing)
    df = df.iloc[1:].reset_index(drop=True)

    # Forward-fill then backward-fill sparse periodic data (turbostat, IPMI)
    df = df.ffill().bfill()

    # Identify feature columns and their categories
    feature_categories: dict[str, str] = {}
    feature_cols: list[str] = []

    for col in df.columns:
        if col == "adj_freq_ppm":
            continue
        if _is_excluded(col):
            continue
        cat = categorize_column(col)
        if cat is not None:
            feature_categories[col] = cat
            feature_cols.append(col)

    # Keep only features + target
    keep_cols = [*feature_cols, "adj_freq_ppm"]
    df = df[keep_cols].copy()

    # -- Temporal aggregation to 1-minute means --------------------------------
    df["_bin"] = np.arange(len(df)) // AGG_SECONDS
    df = df.groupby("_bin", sort=True).mean()
    df = df.reset_index(drop=True)

    # Drop columns that are entirely NaN (even after ffill/bfill)
    all_nan_cols = [c for c in df.columns if df[c].isna().all()]
    if all_nan_cols:
        df = df.drop(columns=all_nan_cols)
        for col in all_nan_cols:
            feature_categories.pop(col, None)

    # Drop columns with zero variance (constant -- no predictive value)
    zero_var_cols = [c for c in df.columns if c != "adj_freq_ppm" and df[c].std() == 0]
    if zero_var_cols:
        df = df.drop(columns=zero_var_cols)
        for col in zero_var_cols:
            feature_categories.pop(col, None)

    # Sync feature_categories with surviving columns
    feature_categories = {
        c: cat for c, cat in feature_categories.items() if c in df.columns
    }

    # Drop remaining rows with any NaN
    df = df.dropna().reset_index(drop=True)

    return df, feature_categories


def load_all(
    data_dir: Path | None = None,
) -> dict[str, tuple[pd.DataFrame, dict[str, str]]]:
    """Load data for all machines.

    Returns:
        Dict mapping machine name -> (DataFrame, feature_categories).
    """
    data_dir = data_dir or DEFAULT_DATA_DIR
    result: dict[str, tuple[pd.DataFrame, dict[str, str]]] = {}
    for name in MACHINES:
        print(f"Loading {name}...")
        df, cats = load_machine(name, data_dir=data_dir)
        n_features = len([c for c in df.columns if c != "adj_freq_ppm"])
        categories_used = sorted(set(cats.values()))
        n_cats = len(categories_used)
        print(
            f"  {len(df)} rows ({AGG_SECONDS}s agg), {n_features} feat, {n_cats} cats"
        )
        print(f"  Categories: {', '.join(categories_used)}")
        result[name] = (df, cats)
    return result
