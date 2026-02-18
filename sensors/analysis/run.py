"""Run the full drift analysis pipeline.

Usage::

    uv run --group analysis python -m analysis.run
"""

from pathlib import Path

from analysis.figures import DEFAULT_FIGURE_DIR, generate_all_figures
from analysis.load_data import DEFAULT_DATA_DIR, load_all
from analysis.models import MachineResults, analyze_machine

# Datasets to analyse: (label, data_dir, figure_dir)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS: list[tuple[str, Path, Path]] = [
    (
        "8h snapshot",
        DEFAULT_DATA_DIR,
        DEFAULT_FIGURE_DIR,
    ),
    (
        "16h snapshot",
        _PROJECT_ROOT / "data" / "16h_snapshot",
        Path(__file__).resolve().parent / "figures_16h",
    ),
    (
        "24h snapshot",
        _PROJECT_ROOT / "data" / "24h_snapshot",
        Path(__file__).resolve().parent / "figures_24h",
    ),
]


def _print_summary(
    label: str,
    all_results: dict[str, MachineResults],
) -> None:
    """Print summary and verification for one dataset."""
    print(f"\n{'=' * 72}")
    print(f"SUMMARY — {label}")
    print("=" * 72)

    for machine, res in all_results.items():
        print(f"\n{'-' * 40}")
        print(f"  {machine}")
        print(f"{'-' * 40}")

        m4_total = res.feature_set_shap.get("M4: All Features", 1.0) or 1.0
        print("  Feature set SHAP contribution:")
        for name, val in res.feature_set_shap.items():
            pct = val / m4_total * 100
            print(f"    {name}: {val:.4f} ({pct:.1f}%)")

        print("  Top-5 SHAP categories:")
        sorted_cats = sorted(
            res.shap_category_importance.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )
        total = sum(v for _, v in sorted_cats) or 1.0
        for cat, val in sorted_cats[:5]:
            print(f"    {cat}: {val:.4f} ({val / total * 100:.1f}%)")

        print("  In-sample R2 (GBR):")
        for name, r2 in res.insample_r2.items():
            print(f"    {name}: {r2:.4f}")

    # -- Verification ---------------------------------------------------------
    print(f"\n{'=' * 72}")
    print(f"VERIFICATION — {label}")
    print("=" * 72)

    n_machines = len(all_results)

    m4_gt_m2 = sum(
        1
        for res in all_results.values()
        if res.feature_set_shap.get("M4: All Features", 0)
        > res.feature_set_shap.get("M2: All Temps", 0)
    )
    print(f"  M4 > M2 (multivariate > temp-only): {m4_gt_m2}/{n_machines}")

    m3_ge_m1 = sum(
        1
        for res in all_results.values()
        if res.feature_set_shap.get("M3: Non-CPU Temp", 0)
        >= res.feature_set_shap.get("M1: CPU Core Temp", 0)
    )
    print(f"  M3 >= M1 (non-CPU temp >= CPU core temp): {m3_ge_m1}/{n_machines}")

    non_temp_cats = {
        "CPU Frequency",
        "Power",
        "C-State",
        "CPU Load",
        "Memory",
        "I/O",
        "System",
    }
    for machine, res in all_results.items():
        total = sum(res.shap_category_importance.values()) or 1.0
        non_temp_shap = sum(
            v for k, v in res.shap_category_importance.items() if k in non_temp_cats
        )
        pct = non_temp_shap / total * 100
        print(f"  {machine}: non-temp = {pct:.1f}% of SHAP")

    for machine, res in all_results.items():
        total = sum(res.shap_category_importance.values()) or 1.0
        significant = [
            cat
            for cat in non_temp_cats
            if res.shap_category_importance.get(cat, 0) / total > 0.05
        ]
        sig_str = ", ".join(significant)
        print(f"  {machine}: {len(significant)} non-temp cats >5%: {sig_str}")


def _run_dataset(
    label: str,
    data_dir: Path,
    figure_dir: Path,
) -> dict[str, MachineResults]:
    """Run the full pipeline for a single dataset."""
    print(f"\n{'#' * 72}")
    print(f"# DATASET: {label}  ({data_dir})")
    print(f"{'#' * 72}")

    data = load_all(data_dir=data_dir)

    all_results: dict[str, MachineResults] = {}
    for machine, (df, feature_categories) in data.items():
        all_results[machine] = analyze_machine(machine, df, feature_categories)

    generate_all_figures(all_results, figure_dir=figure_dir)
    _print_summary(label, all_results)
    return all_results


def main() -> None:
    """Load data, fit models, compute SHAP, and generate figures for all datasets."""
    for label, data_dir, figure_dir in DATASETS:
        if not data_dir.exists():
            print(f"\nSkipping {label}: {data_dir} not found")
            continue
        _run_dataset(label, data_dir, figure_dir)


if __name__ == "__main__":
    main()
