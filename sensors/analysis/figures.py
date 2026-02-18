"""Publication-quality figures for multivariate clock drift analysis."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch

from analysis.load_data import categorize_column
from analysis.models import MachineResults

matplotlib.use("Agg")

DEFAULT_FIGURE_DIR = Path(__file__).resolve().parent / "figures"

# -- Publication defaults -----------------------------------------------------

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    }
)

# -- Consistent category colours and ordering ---------------------------------

CATEGORY_COLORS: dict[str, str] = {
    "Non-CPU Temp": "#d62728",
    "CPU Core Temp": "#ff7f0e",
    "CPU Package Temp": "#ff9896",
    "CPU Frequency": "#2ca02c",
    "CPU Load": "#9467bd",
    "Power": "#8c564b",
    "C-State": "#e377c2",
    "Memory": "#7f7f7f",
    "I/O": "#bcbd22",
    "System": "#17becf",
}

CATEGORY_ORDER = [
    "Non-CPU Temp",
    "CPU Core Temp",
    "CPU Package Temp",
    "CPU Frequency",
    "Power",
    "C-State",
    "CPU Load",
    "Memory",
    "I/O",
    "System",
]

MACHINE_ORDER = ["homelab", "chameleon", "ares", "ares-comp-10"]

_MACHINE_LABELS: dict[str, str] = {
    "homelab": "Commodity (idle)",
    "chameleon": "Commodity (stressed)",
    "ares": "HPC (idle)",
    "ares-comp-10": "HPC (stressed)",
}


def _ensure_dir(figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, stem: str, figure_dir: Path) -> None:
    """Save a figure as both PDF and PNG, then close it."""
    fig.savefig(figure_dir / f"{stem}.pdf")
    fig.savefig(figure_dir / f"{stem}.png")
    plt.close(fig)


# -- Figure 1: SHAP Category Importance Heatmap ------------------------------


def figure1_shap_category_heatmap(
    all_results: dict[str, MachineResults],
    figure_dir: Path = DEFAULT_FIGURE_DIR,
) -> None:
    """SHAP category importance heatmap (%-normalised per machine)."""
    _ensure_dir(figure_dir)

    categories = [
        c
        for c in CATEGORY_ORDER
        if any(c in r.shap_category_importance for r in all_results.values())
    ]
    machines = MACHINE_ORDER

    raw = np.zeros((len(categories), len(machines)))
    for j, machine in enumerate(machines):
        res = all_results[machine]
        for i, cat in enumerate(categories):
            raw[i, j] = res.shap_category_importance.get(cat, 0.0)

    col_sums = raw.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1.0
    pct = raw / col_sums * 100.0

    # Build annotation array: mark 0.0 as "N/A", otherwise show value
    annot_arr = np.empty_like(pct, dtype=object)
    for i in range(pct.shape[0]):
        for j in range(pct.shape[1]):
            annot_arr[i, j] = "N/A" if pct[i, j] == 0.0 else f"{pct[i, j]:.1f}"

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        pct,
        annot=annot_arr,
        fmt="",
        xticklabels=[_MACHINE_LABELS[m] for m in machines],
        yticklabels=categories,
        cmap="YlOrRd",
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        vmin=0,
        cbar_kws={"label": "% of total mean"},
    )

    ax.set_xlabel("Machine")
    ax.set_ylabel("")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)

    _save(fig, "fig1_shap_heatmap", figure_dir)
    print("  Saved figure 1: SHAP category heatmap")


# -- Figure 2: Feature Set Contribution (SHAP-based) -------------------------

_MODEL_NAMES = [
    "M1: CPU Core Temp",
    "M2: All Temps",
    "M3: Non-CPU Temp",
    "M4: All Features",
]
_MODEL_LABELS = [
    "M1 CPU Core Temp",
    "M2 All Temps",
    "M3 Non-CPU Temp",
    "M4 All Features",
]
_MODEL_COLORS = ["#ff7f0e", "#d62728", "#9467bd", "#2ca02c"]


def figure2_feature_set_contribution(
    all_results: dict[str, MachineResults],
    figure_dir: Path = DEFAULT_FIGURE_DIR,
) -> None:
    """Feature set contribution bar chart (cumulative SHAP, % of M4)."""
    _ensure_dir(figure_dir)

    machines = MACHINE_ORDER
    n_machines = len(machines)
    n_models = len(_MODEL_NAMES)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(n_machines)
    width = 0.18

    for i, (model_name, label, color) in enumerate(
        zip(_MODEL_NAMES, _MODEL_LABELS, _MODEL_COLORS, strict=True)
    ):
        pct_values = []
        for machine in machines:
            res = all_results[machine]
            m4_total = res.feature_set_shap.get("M4: All Features", 1.0) or 1.0
            model_total = res.feature_set_shap.get(model_name, 0.0)
            pct_values.append(model_total / m4_total * 100.0)

        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            pct_values,
            width,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.5,
        )
        for bar, val in zip(bars, pct_values, strict=True):
            if val > 3:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{val:.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    ax.set_xlabel("Machine")
    ax.set_ylabel("% of Total SHAP Importance")
    ax.set_xticks(x)
    ax.set_xticklabels([_MACHINE_LABELS[m] for m in machines])
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    ax.set_ylim(0, 115)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.axhline(100, color="black", linewidth=0.5, linestyle="--", alpha=0.3)
    ax.grid(axis="y", alpha=0.3)

    _save(fig, "fig2_feature_set_contribution", figure_dir)
    print("  Saved figure 2: Feature set contribution")


# -- Figure 3: Top-20 SHAP Feature Bar Charts (4-panel) ----------------------


def figure3_top_features(
    all_results: dict[str, MachineResults],
    figure_dir: Path = DEFAULT_FIGURE_DIR,
) -> None:
    """Top-20 features by SHAP importance per machine (4-panel)."""
    _ensure_dir(figure_dir)

    machines = MACHINE_ORDER
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes_flat: list[plt.Axes] = axes.flatten().tolist()

    for machine, ax in zip(machines, axes_flat, strict=True):
        res = all_results[machine]

        sorted_feats = sorted(
            res.shap_feature_importance.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )[:20]

        if not sorted_feats:
            ax.set_title(f"{_MACHINE_LABELS[machine]} (no features)")
            continue

        names = [f[0] for f in reversed(sorted_feats)]
        values = [f[1] for f in reversed(sorted_feats)]

        colors = []
        for name in names:
            cat = res.feature_categories.get(name) or categorize_column(name)
            colors.append(CATEGORY_COLORS.get(cat, "#999999") if cat else "#999999")

        ax.barh(
            range(len(names)),
            values,
            color=colors,
            edgecolor="black",
            linewidth=0.3,
        )
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("Mean |SHAP value| (PPM)")
        ax.set_title(_MACHINE_LABELS[machine])

    legend_handles = [
        Patch(
            facecolor=CATEGORY_COLORS[cat],
            edgecolor="black",
            linewidth=0.3,
            label=cat,
        )
        for cat in CATEGORY_ORDER
        if cat in CATEGORY_COLORS
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.tight_layout(rect=(0, 0.04, 1, 1.0))

    _save(fig, "fig3_top_features", figure_dir)
    print("  Saved figure 3: Top-20 features per machine")


# -- Figure 4: Evaluation (Progressive R² + Category Importance) --------------

TEMP_CATEGORIES = {"CPU Core Temp", "CPU Package Temp", "Non-CPU Temp"}

_MACHINE_COLORS = {
    "homelab": "#1f77b4",
    "chameleon": "#ff7f0e",
    "ares": "#2ca02c",
    "ares-comp-10": "#d62728",
}


def figure_eval(
    all_results: dict[str, MachineResults],
    figure_dir: Path = DEFAULT_FIGURE_DIR,
) -> None:
    """Two-panel evaluation: progressive R² + category importance stacked bars."""
    _ensure_dir(figure_dir)

    machines = MACHINE_ORDER
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10, 4.5))

    # ── Panel (a): Progressive R² ──────────────────────────────────────────
    # Collect "temperature-only" R² band (M2 from insample_r2)
    m2_values = [all_results[m].insample_r2.get("M2: All Temps", 0.0) for m in machines]
    m2_min, m2_max = min(m2_values), max(m2_values)

    ax_a.axhspan(m2_min, m2_max, alpha=0.15, color="#ff7f0e", zorder=0)
    ax_a.text(
        0.98,
        (m2_min + m2_max) / 2,
        "Temp.\nonly",
        transform=ax_a.get_yaxis_transform(),
        ha="right",
        va="center",
        fontsize=8,
        fontstyle="italic",
        color="#cc6600",
    )

    for machine in machines:
        res = all_results[machine]
        ks = sorted(res.progressive_r2.keys())
        r2s = [res.progressive_r2[k] for k in ks]
        color = _MACHINE_COLORS[machine]

        ax_a.plot(ks, r2s, "-o", color=color, markersize=4, linewidth=1.5, zorder=2)

        # Direct label at the rightmost point
        ax_a.annotate(
            _MACHINE_LABELS[machine],
            xy=(ks[-1], r2s[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=8,
            color=color,
            va="center",
            fontweight="bold",
        )

    ax_a.set_xlabel("Number of feature categories (SHAP-ranked)")
    ax_a.set_ylabel("In-sample R²")
    ax_a.set_title("(a)")
    ax_a.set_ylim(0.35, 1.02)
    ax_a.set_xlim(0.5, None)
    ax_a.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax_a.grid(axis="y", alpha=0.3)

    # ── Panel (b): Category Importance Stacked Bars ────────────────────────
    y_positions = np.arange(len(machines))
    bar_height = 0.55

    for idx, machine in enumerate(machines):
        res = all_results[machine]
        total = sum(res.shap_category_importance.values()) or 1.0

        # Separate into temp and non-temp, each sorted by importance descending
        cat_imp = res.shap_category_importance
        temp_cats = sorted(
            ((c, v) for c, v in cat_imp.items() if c in TEMP_CATEGORIES),
            key=lambda kv: kv[1],
            reverse=True,
        )
        nontemp_cats = sorted(
            ((c, v) for c, v in cat_imp.items() if c not in TEMP_CATEGORIES),
            key=lambda kv: kv[1],
            reverse=True,
        )

        # Build ordered segments: temp first, then non-temp
        segments = temp_cats + nontemp_cats
        temp_pct = sum(v for _, v in temp_cats) / total * 100.0

        left = 0.0
        for cat, val in segments:
            width = val / total * 100.0
            color = CATEGORY_COLORS.get(cat, "#999999")
            ax_b.barh(
                idx,
                width,
                left=left,
                height=bar_height,
                color=color,
                edgecolor="white",
                linewidth=0.5,
            )

            # Annotate inside segment
            if width > 10:
                ax_b.text(
                    left + width / 2,
                    idx,
                    cat,
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    fontweight="bold",
                    color="white",
                )
            if width > 8:
                ax_b.text(
                    left + width / 2,
                    idx + 0.22,
                    f"{width:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white",
                )
            left += width

        # Vertical dashed line at temp/non-temp boundary
        if temp_pct > 0:
            ax_b.axvline(
                temp_pct,
                color="black",
                linewidth=0.8,
                linestyle="--",
                alpha=0.5,
                ymin=(idx - 0.35) / len(machines),
                ymax=(idx + 0.35) / len(machines),
            )

    ax_b.set_yticks(y_positions)
    ax_b.set_yticklabels([_MACHINE_LABELS[m] for m in machines])
    ax_b.set_xlabel("% of total mean")
    ax_b.set_title("(b)")
    ax_b.set_xlim(0, 100)
    ax_b.invert_yaxis()

    # ── Shared legend ──────────────────────────────────────────────────────
    present_cats = set()
    for res in all_results.values():
        present_cats.update(res.shap_category_importance.keys())

    legend_handles = [
        Patch(
            facecolor=CATEGORY_COLORS[cat],
            edgecolor="white",
            linewidth=0.5,
            label=cat,
        )
        for cat in CATEGORY_ORDER
        if cat in present_cats
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(len(legend_handles), 6),
        fontsize=8,
        bbox_to_anchor=(0.5, -0.04),
        frameon=False,
    )

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _save(fig, "fig_eval", figure_dir)
    print("  Saved figure: Multivariate evaluation (fig_eval)")


# -- Public entry point -------------------------------------------------------


def generate_all_figures(
    all_results: dict[str, MachineResults],
    figure_dir: Path = DEFAULT_FIGURE_DIR,
) -> None:
    """Generate all publication figures."""
    print(f"\nGenerating figures to {figure_dir}/...")
    figure1_shap_category_heatmap(all_results, figure_dir)
    figure2_feature_set_contribution(all_results, figure_dir)
    figure3_top_features(all_results, figure_dir)
    figure_eval(all_results, figure_dir)
    print(f"All figures saved to {figure_dir}/")
