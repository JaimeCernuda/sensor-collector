"""Benchmark reporting: LaTeX tables and CSV export."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def results_to_latex(
    df: pd.DataFrame,
    caption: str = "Zero-shot benchmark results",
    label: str = "tab:benchmark",
) -> str:
    """Convert benchmark results DataFrame to an IEEEtran LaTeX table.

    Args:
        df: Summary DataFrame from ``results_to_dataframe()``.
        caption: Table caption.
        label: LaTeX label for cross-referencing.

    Returns:
        LaTeX table string.
    """
    # Pivot: rows = model x machine, columns = metric
    pivot_cols = ["model", "machine", "context_length", "horizon", "with_covariates"]
    metric_cols = ["mae", "rmse", "inference_ms"]

    # Only keep available columns
    available_pivot = [c for c in pivot_cols if c in df.columns]
    available_metrics = [c for c in metric_cols if c in df.columns]

    # Add probabilistic metrics if present
    for col in ["coverage", "crps"]:
        if col in df.columns and df[col].notna().any():
            available_metrics.append(col)

    display = df[available_pivot + available_metrics].copy()

    # Format numeric columns
    def _fmt(x: object, spec: str) -> str:
        return f"{x:{spec}}" if pd.notna(x) else "--"

    for col in available_metrics:
        if col == "inference_ms":
            display[col] = display[col].apply(_fmt, spec=".1f")
        elif col == "coverage":
            display[col] = display[col].apply(_fmt, spec=".1%")
        else:
            display[col] = display[col].apply(_fmt, spec=".4f")

    # Rename columns for display
    col_renames = {
        "model": "Model",
        "machine": "Machine",
        "context_length": "Ctx",
        "horizon": "Hz",
        "with_covariates": "Cov",
        "mae": "MAE",
        "rmse": "RMSE",
        "inference_ms": "Time (ms)",
        "coverage": "Coverage",
        "crps": "CRPS",
    }
    display = display.rename(columns=col_renames)

    # Generate LaTeX
    col_spec = "l" * min(len(available_pivot), 3) + "r" * len(available_metrics)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        " & ".join(display.columns) + r" \\",
        r"\midrule",
    ]

    for _, row in display.iterrows():
        vals = [str(v) for v in row.values]
        lines.append(" & ".join(vals) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )

    return "\n".join(lines)


def save_results(
    df: pd.DataFrame,
    output_dir: Path,
    prefix: str = "benchmark",
) -> tuple[Path, Path]:
    """Save benchmark results as CSV and LaTeX.

    Args:
        df: Summary DataFrame.
        output_dir: Directory to save files.
        prefix: Filename prefix.

    Returns:
        Tuple of (csv_path, latex_path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{prefix}_results.csv"
    df.to_csv(csv_path, index=False)

    latex_str = results_to_latex(df)
    latex_path = output_dir / f"{prefix}_table.tex"
    latex_path.write_text(latex_str, encoding="utf-8")

    return csv_path, latex_path


def format_summary(df: pd.DataFrame) -> str:
    """Format a human-readable summary of benchmark results.

    Args:
        df: Summary DataFrame from results_to_dataframe().

    Returns:
        Formatted string summary.
    """
    lines: list[str] = ["=" * 60, "BENCHMARK SUMMARY", "=" * 60, ""]

    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        lines.append(f"Model: {model}")
        lines.append("-" * 40)

        for _, row in model_df.iterrows():
            cov = "cov" if row.get("with_covariates") else "uni"
            mae_val = row.get("mae")
            mae_str = f"MAE={mae_val:.4f}" if pd.notna(mae_val) else ""
            t_val = row.get("inference_ms")
            time_str = f"Time={t_val:.1f}ms" if pd.notna(t_val) else ""
            cov_str = ""
            if pd.notna(row.get("coverage")):
                cov_str = f"Cov={row['coverage']:.1%}"
            mach = row["machine"]
            ctx = row["context_length"]
            hz = row["horizon"]
            lines.append(
                f"  {mach:16s} ctx={ctx:5d} "
                f"hz={hz:4d} {cov:3s}  "
                f"{mae_str:16s} {time_str:16s} {cov_str}"
            )

        lines.append("")

    return "\n".join(lines)
