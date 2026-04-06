"""
Figure generation for the paper.
Parameterized matplotlib figures from benchmark results.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from scipy import stats
from scipy.stats import chi2_contingency


# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    "methods": ["DQv1", "DQv2", "GNINA", "Smina"],
    "method_colors": {
        "DQv1": "#1f77b4",
        "DQv2": "#ff7f0e",
        "GNINA": "#2ca02c",
        "Smina": "#d62728",
    },
    "method_order": ["DQv1", "DQv2", "GNINA", "Smina"],
    "figure_dpi": 300,
    "scatter_size": 8,
    "scatter_alpha": 1.0,
    "scatter_alpha_transparent": 0.4,
    "scatter_edgecolor": "black",
    "scatter_linewidth": 0.5,
    "threshold_lines": [0.25, 0.5, 1, 2],
    "threshold_color": "red",
    "exclude_classes": ["transcription", "isomerase"],
    "success_thresholds": [
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.25,
        1.5,
        1.75,
        2.0,
        2.25,
        2.5,
        2.75,
        3.0,
        3.5,
        4.0,
        4.5,
        5.0,
    ],
    "outlier_pdb": "2g8r",
    "outlier_threshold": 80,
    "max_time_seconds": 90,
}


def prepare_unified_data(csv_path: str) -> pd.DataFrame:
    """Load and prepare unified dataframe."""
    df = pd.read_csv(csv_path)

    rmsd_col = "top_rmsd" if "top_rmsd" in df.columns else "rmsd"
    if rmsd_col in df.columns:
        df = df[
            ~(
                (df["pdb_id"] == CONFIG["outlier_pdb"])
                & (df[rmsd_col] > CONFIG["outlier_threshold"])
            )
        ]

    if "target_class" in df.columns and CONFIG.get("exclude_classes"):
        df = df[~df["target_class"].isin(CONFIG["exclude_classes"])]

    max_time = CONFIG.get("max_time_seconds", 90)
    if "time_s" in df.columns:
        df["time_s"] = df["time_s"].clip(upper=max_time)

    if "top_rmsd" in df.columns:
        df = df.rename(columns={"top_rmsd": "rmsd"})

    return df


def compute_statistics(unified_df: pd.DataFrame) -> dict:
    """Compute all pairwise statistics."""
    results = {
        "pairwise": {},
        "pairwise_by_class": {},
        "summary": {},
        "success_rate": {},
    }

    methods = CONFIG["method_order"]
    for method in methods:
        subset = unified_df[unified_df["method"] == method]
        rmsd_vals = subset["rmsd"].dropna()
        time_vals = subset["time_s"].dropna()

        results["summary"][method] = {
            "n": len(rmsd_vals),
            "mean_rmsd": rmsd_vals.mean(),
            "median_rmsd": rmsd_vals.median(),
            "mean_time": time_vals.mean(),
            "success_at_2A": (rmsd_vals < 2.0).sum() / len(rmsd_vals) * 100,
        }

    for threshold in CONFIG["success_thresholds"]:
        results["success_rate"][threshold] = {}
        for method in methods:
            subset = unified_df[unified_df["method"] == method]
            rmsd_vals = subset["rmsd"].dropna()
            success_count = (rmsd_vals < threshold).sum()
            results["success_rate"][threshold][method] = {
                "success": success_count,
                "total": len(rmsd_vals),
                "rate": success_count / len(rmsd_vals) * 100
                if len(rmsd_vals) > 0
                else 0,
            }

        method_a = "DQv2"
        method_b = "DQv1"
        data_a = unified_df[unified_df["method"] == method_a].set_index("pdb_id")
        data_b = unified_df[unified_df["method"] == method_b].set_index("pdb_id")
        common = data_a.index.intersection(data_b.index)

        rmsd_a = data_a.loc[common, "rmsd"].values
        rmsd_b = data_b.loc[common, "rmsd"].values
        valid = ~(np.isnan(rmsd_a) | np.isnan(rmsd_b))

        success_a = (rmsd_a[valid] < threshold).astype(int)
        success_b = (rmsd_b[valid] < threshold).astype(int)

        # Paired comparison of success/failure for DQv2 vs DQv1 using McNemar exact (binomial) test
        b = ((success_a == 0) & (success_b == 1)).sum()  # DQv2 fail, DQv1 success
        c = ((success_a == 1) & (success_b == 0)).sum()  # DQv2 success, DQv1 fail
        discordant = b + c

        p_val = np.nan
        try:
            if discordant > 0:
                # Use exact binomial test on the smaller discordant count
                # scipy.stats.binomtest (new) or binom_test (older)
                try:
                    p_val = stats.binomtest(min(b, c), n=discordant, p=0.5).pvalue
                except Exception:
                    try:
                        p_val = stats.binom_test(min(b, c), discordant, p=0.5)
                    except Exception:
                        p_val = np.nan
            else:
                p_val = np.nan
        except Exception:
            p_val = np.nan

        results["success_rate"][threshold]["DQv2_vs_DQv1"] = {
            "p": p_val,
            "b": int(b),
            "c": int(c),
            "n": int(discordant),
        }

        # Also compute per-class McNemar for later use/inspection
        if "target_class" in unified_df.columns:
            results["success_rate"][threshold]["by_class"] = {}
            classes = sorted(
                unified_df[unified_df["target_class"].notna()]["target_class"].unique()
            )
            for cls in classes:
                # Safely select PDBs for this class in each method and intersect indices
                if "target_class" in data_a.columns:
                    idx_a_cls = data_a.index[data_a["target_class"] == cls]
                else:
                    idx_a_cls = data_a.index
                if "target_class" in data_b.columns:
                    idx_b_cls = data_b.index[data_b["target_class"] == cls]
                else:
                    idx_b_cls = data_b.index

                paired_idx = idx_a_cls.intersection(idx_b_cls).intersection(common)

                if len(paired_idx) > 0:
                    rmsd_a_cls = data_a.loc[paired_idx, "rmsd"].values
                    rmsd_b_cls = data_b.loc[paired_idx, "rmsd"].values
                else:
                    rmsd_a_cls = np.array([])
                    rmsd_b_cls = np.array([])

                valid_cls = (
                    ~(np.isnan(rmsd_a_cls) | np.isnan(rmsd_b_cls))
                    if len(rmsd_a_cls) > 0 and len(rmsd_b_cls) > 0
                    else np.array([], dtype=bool)
                )
                success_a_cls = (
                    (rmsd_a_cls[valid_cls] < threshold).astype(int)
                    if len(rmsd_a_cls) > 0 and len(rmsd_b_cls) > 0
                    else np.array([], dtype=int)
                )
                success_b_cls = (
                    (rmsd_b_cls[valid_cls] < threshold).astype(int)
                    if len(rmsd_a_cls) > 0 and len(rmsd_b_cls) > 0
                    else np.array([], dtype=int)
                )

                b_cls_cnt = (
                    int(((success_a_cls == 0) & (success_b_cls == 1)).sum())
                    if len(success_a_cls) > 0
                    else 0
                )
                c_cls_cnt = (
                    int(((success_a_cls == 1) & (success_b_cls == 0)).sum())
                    if len(success_a_cls) > 0
                    else 0
                )
                discordant_cls = b_cls_cnt + c_cls_cnt

                p_cls = np.nan
                if discordant_cls > 0:
                    try:
                        p_cls = stats.binomtest(
                            min(b_cls_cnt, c_cls_cnt), n=discordant_cls, p=0.5
                        ).pvalue
                    except Exception:
                        try:
                            p_cls = stats.binom_test(
                                min(b_cls_cnt, c_cls_cnt), discordant_cls, p=0.5
                            )
                        except Exception:
                            p_cls = np.nan

                results["success_rate"][threshold]["by_class"][cls] = {
                    "p": p_cls,
                    "b": b_cls_cnt,
                    "c": c_cls_cnt,
                    "n": discordant_cls,
                }

    dqv1 = unified_df[unified_df["method"] == "DQv1"].set_index("pdb_id")
    dqv2 = unified_df[unified_df["method"] == "DQv2"].set_index("pdb_id")
    gnina = unified_df[unified_df["method"] == "GNINA"].set_index("pdb_id")
    smina = unified_df[unified_df["method"] == "Smina"].set_index("pdb_id")

    comparisons = [
        ("DQv2", "DQv1", "accuracy"),
        ("DQv1", "DQv2", "speed"),
        ("DQv2", "GNINA", "accuracy"),
        ("DQv2", "Smina", "accuracy"),
        ("DQv1", "GNINA", "accuracy"),
        ("DQv1", "Smina", "accuracy"),
        ("DQv1", "GNINA", "speed"),
        ("DQv1", "Smina", "speed"),
    ]

    for method_a, method_b, metric in comparisons:
        data_a = unified_df[unified_df["method"] == method_a].set_index("pdb_id")
        data_b = unified_df[unified_df["method"] == method_b].set_index("pdb_id")
        common = data_a.index.intersection(data_b.index)

        col = "rmsd" if metric == "accuracy" else "time_s"
        vals_a = data_a.loc[common, col].values
        vals_b = data_b.loc[common, col].values
        valid = ~(np.isnan(vals_a) | np.isnan(vals_b))
        vals_a = vals_a[valid]
        vals_b = vals_b[valid]

        if len(vals_a) > 0:
            stat, p = stats.wilcoxon(vals_a, vals_b, alternative="less")
        else:
            stat, p = np.nan, np.nan

        key = f"{method_a}_vs_{method_b}_{metric}"
        results["pairwise"][key] = {"stat": stat, "p_value": p}

    # Per-class pairwise tests (including an "all" aggregate)
    if "target_class" in unified_df.columns:
        classes = sorted(
            unified_df[unified_df["target_class"].notna()]["target_class"].unique()
        )
    else:
        classes = []
    classes = ["all"] + classes

    for cls in classes:
        results["pairwise_by_class"][cls] = {}
        for method_a, method_b, metric in comparisons:
            col = "rmsd" if metric == "accuracy" else "time_s"

            if cls == "all":
                data_a = unified_df[unified_df["method"] == method_a].set_index(
                    "pdb_id"
                )
                data_b = unified_df[unified_df["method"] == method_b].set_index(
                    "pdb_id"
                )
            else:
                data_a = unified_df[
                    (unified_df["method"] == method_a)
                    & (unified_df["target_class"] == cls)
                ].set_index("pdb_id")
                data_b = unified_df[
                    (unified_df["method"] == method_b)
                    & (unified_df["target_class"] == cls)
                ].set_index("pdb_id")

            common = data_a.index.intersection(data_b.index)
            vals_a = data_a.loc[common, col].values if len(common) > 0 else np.array([])
            vals_b = data_b.loc[common, col].values if len(common) > 0 else np.array([])
            valid = (
                ~(np.isnan(vals_a) | np.isnan(vals_b))
                if len(vals_a) > 0
                else np.array([], dtype=bool)
            )
            vals_a = vals_a[valid]
            vals_b = vals_b[valid]

            p_val = np.nan
            stat_val = np.nan
            try:
                if len(vals_a) > 0:
                    # Use Wilcoxon for paired continuous comparisons. Use two-sided fallback if SciPy requires it.
                    stat_val, p_val = stats.wilcoxon(vals_a, vals_b, alternative="less")
                else:
                    stat_val, p_val = np.nan, np.nan
            except Exception:
                # Wilcoxon can fail on zero differences or too small samples; record NaN
                stat_val, p_val = np.nan, np.nan

            key = f"{method_a}_vs_{method_b}_{metric}"
            results["pairwise_by_class"][cls][key] = {
                "stat": stat_val,
                "p_value": p_val,
                "n": len(vals_a),
            }

    return results


def add_bar_significance(
    ax, x_positions, method_order, stats_results, value_col, classes=None, bar_data=None
):
    """Add significance brackets above each bar group using per-class tests.

    This implementation uses axis-fraction stacking to avoid collisions between
    brackets and plotted points. Brackets are positioned above the maximum
    observed data for the two methods being compared (using `bar_data` if
    available), converted to axes fraction coordinates, and then stacked so
    overlapping x-ranges don't share the same vertical level.
    """
    pairwise_by_class = stats_results.get("pairwise_by_class", {})

    if value_col == "rmsd":
        comparisons = [
            (1, 0, "DQv2_vs_DQv1_accuracy"),
            (1, 2, "DQv2_vs_GNINA_accuracy"),
            (1, 3, "DQv2_vs_Smina_accuracy"),
            (0, 2, "DQv1_vs_GNINA_accuracy"),
            (0, 3, "DQv1_vs_Smina_accuracy"),
        ]
    else:
        comparisons = [
            (0, 1, "DQv1_vs_DQv2_speed"),
            (0, 2, "DQv1_vs_GNINA_speed"),
            (0, 3, "DQv1_vs_Smina_speed"),
        ]

    n_methods = len(method_order)
    width = 0.2

    # helpers to convert between data coords and axes fraction coords
    def data_to_axes(x, y):
        disp = ax.transData.transform((x, y))
        return ax.transAxes.inverted().transform(disp)

    def axes_to_data(xf, yf):
        disp = ax.transAxes.transform((xf, yf))
        return ax.transData.inverted().transform(disp)

    y_min, y_max = ax.get_ylim()

    # Build candidate brackets list
    candidates = []
    for class_idx, x_pos in enumerate(x_positions):
        cls_label = (
            classes[class_idx]
            if (classes is not None and class_idx < len(classes))
            else None
        )

        for pos_a, pos_b, key in comparisons:
            p = np.nan
            if cls_label is not None:
                p = pairwise_by_class.get(cls_label, {}).get(key, {}).get("p_value")
            if p is None or np.isnan(p):
                continue

            if p < 0.001:
                sig = "***"
            elif p < 0.01:
                sig = "**"
            elif p < 0.05:
                sig = "*"
            else:
                continue

            x1 = x_pos + (pos_a - n_methods / 2 + 0.5) * width
            x2 = x_pos + (pos_b - n_methods / 2 + 0.5) * width

            # conservative base: max observed among the two methods for this class
            base_y = y_min
            if bar_data is not None and cls_label in bar_data:
                for m_idx in (pos_a, pos_b):
                    if 0 <= m_idx < len(method_order):
                        method = method_order[m_idx]
                        mv = bar_data.get(cls_label, {}).get(method, {})
                        if mv is not None:
                            vals = mv.get("values")
                            if vals is not None and len(vals) > 0:
                                try:
                                    base_y = max(
                                        base_y, float(np.nanmax(np.asarray(vals)))
                                    )
                                except Exception:
                                    pass
                            elif mv.get("mean") is not None:
                                try:
                                    base_y = max(base_y, float(mv.get("mean", 0)))
                                except Exception:
                                    pass

            if base_y == y_min:
                base_y = y_min + 0.02 * (y_max - y_min if (y_max > y_min) else 1.0)

            # Convert endpoints to axis fraction coords (for horizontal overlap testing)
            x1f, _ = data_to_axes(x1, base_y)
            x2f, _ = data_to_axes(x2, base_y)
            _, y_frac_base = data_to_axes((x1 + x2) / 2.0, base_y)

            candidates.append(
                {
                    "class_idx": class_idx,
                    "cls": cls_label,
                    "x1_data": x1,
                    "x2_data": x2,
                    "x1f": min(x1f, x2f),
                    "x2f": max(x1f, x2f),
                    "xcenterf": (x1f + x2f) / 2.0,
                    "p": p,
                    "sig": sig,
                    "y_frac_base": y_frac_base,
                }
            )

    if len(candidates) == 0:
        return

    # sort by base vertical position so lower brackets lock in first
    candidates.sort(key=lambda c: c["y_frac_base"])

    placed = []  # each item: {x1f,x2f,y_frac,sig,x1_data,x2_data}

    # fractions used for spacing and margins (axes fraction units)
    margin_frac = 0.02
    step_frac = 0.05
    max_frac = 0.92

    for cand in candidates:
        a1f, a2f = cand["x1f"], cand["x2f"]
        yf = cand["y_frac_base"] + margin_frac

        # find lowest yf that doesn't collide with already placed brackets that overlap in x
        placed_ok = False
        while not placed_ok:
            placed_ok = True
            for p in placed:
                # check horizontal overlap (with small padding)
                pad = 0.005
                if not (a2f + pad < p["x1f"] or a1f - pad > p["x2f"]):
                    # overlapping horizontally -> ensure vertical separation
                    if yf <= p["y_frac"] + 0.005:
                        yf = p["y_frac"] + step_frac
                        placed_ok = False
                        break

        placed.append(
            {
                "x1f": a1f,
                "x2f": a2f,
                "y_frac": yf,
                "sig": cand["sig"],
                "x1_data": cand["x1_data"],
                "x2_data": cand["x2_data"],
                "xcenterf": cand["xcenterf"],
            }
        )

    # If any bracket needs to be beyond the drawable area, expand y-limits
    highest_frac = max(p["y_frac"] for p in placed) + 0.03
    if highest_frac > max_frac:
        # compute the needed data y for the highest_frac and expand ylim
        needed_y = axes_to_data(0.0, highest_frac)[1]
        cur_ymin, cur_ymax = ax.get_ylim()
        if needed_y > cur_ymax:
            ax.set_ylim(cur_ymin, needed_y * 1.04)

    # Draw brackets in data coords and text in axes fraction coords
    for p in placed:
        y_data = axes_to_data(0.0, p["y_frac"])[1]
        # small tick in axes fraction converted to data for bracket ends
        tick_frac = 0.008
        y_low = axes_to_data(0.0, p["y_frac"] - tick_frac)[1]
        y_coords = [y_low, y_data, y_data, y_low]

        ax.plot(
            [p["x1_data"], p["x1_data"], p["x2_data"], p["x2_data"]],
            y_coords,
            lw=1.2,
            c="black",
            zorder=5,
        )
        # place sig text in axes fraction to avoid overlap with plotted points
        ax.text(
            p["xcenterf"],
            p["y_frac"] + 0.003,
            p["sig"],
            ha="center",
            va="bottom",
            fontsize=8,
            transform=ax.transAxes,
            bbox=dict(
                boxstyle="round,pad=0.12",
                facecolor="white",
                edgecolor="none",
                alpha=0.95,
            ),
            zorder=6,
        )


def add_success_rate_significance(ax, unified_df: pd.DataFrame, stats_results: dict):
    """Add significance annotations to success rate plot.

    Shows significance of DQv2 vs DQv1 comparison (chi-square test on paired samples).
    Stars are colored blue (DQv1's color) to indicate the comparison.
    """
    if "success_rate" not in stats_results:
        return

    success_data = stats_results["success_rate"]
    thresholds = [t for t in CONFIG["success_thresholds"] if t in success_data]

    dqv1_color = CONFIG["method_colors"].get("DQv1", "#1f77b4")

    for thresh in thresholds:
        p = success_data[thresh].get("DQv2_vs_DQv1", {}).get("p")
        if p is None or np.isnan(p):
            continue

        if p < 0.001:
            sig = "***"
        elif p < 0.01:
            sig = "**"
        elif p < 0.05:
            sig = "*"
        else:
            sig = "ns"

        dqv2_rate = success_data[thresh].get("DQv2", {}).get("rate", 0)

        ax.annotate(
            sig,
            xy=(thresh, dqv2_rate),
            xytext=(0, -20),
            textcoords="offset points",
            fontsize=9,
            ha="center",
            va="top",
            color=dqv1_color,
            fontweight="bold" if sig != "ns" else "normal",
            arrowprops=dict(arrowstyle="-", color=dqv1_color, lw=0.8, shrinkB=5),
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                edgecolor=dqv1_color,
                alpha=0.8,
            ),
        )

    ax.text(
        0.98,
        0.02,
        "* DQv2 vs DQv1 (McNemar's test)",
        transform=ax.transAxes,
        fontsize=7,
        ha="right",
        va="bottom",
        style="italic",
        color="gray",
    )


def add_stats_annotation(ax, unified_df: pd.DataFrame, stats_results: dict):
    """Add statistical significance annotation to figure."""
    summary = stats_results["summary"]
    pairwise = stats_results["pairwise"]

    accuracy_items = [
        ("DQv2 vs DQv1", pairwise.get("DQv2_vs_DQv1_accuracy", {}).get("p_value")),
        ("DQv2 vs GNINA", pairwise.get("DQv2_vs_GNINA_accuracy", {}).get("p_value")),
        ("DQv2 vs Smina", pairwise.get("DQv2_vs_Smina_accuracy", {}).get("p_value")),
        ("DQv1 vs GNINA", pairwise.get("DQv1_vs_GNINA_accuracy", {}).get("p_value")),
        ("DQv1 vs Smina", pairwise.get("DQv1_vs_Smina_accuracy", {}).get("p_value")),
    ]

    speed_items = [
        ("DQv1 vs DQv2", pairwise.get("DQv1_vs_DQv2_speed", {}).get("p_value")),
        ("DQv1 vs GNINA", pairwise.get("DQv1_vs_GNINA_speed", {}).get("p_value")),
        ("DQv1 vs Smina", pairwise.get("DQv1_vs_Smina_speed", {}).get("p_value")),
    ]

    def format_p(p):
        if np.isnan(p):
            return "N/A"
        if p < 0.001:
            return f"p={p:.2e}"
        return f"p={p:.4f}"

    lines = []
    lines.append("Accuracy comparisons:")
    for label, p in accuracy_items:
        sig = (
            "***"
            if (not np.isnan(p) and p < 0.001)
            else "**"
            if (not np.isnan(p) and p < 0.01)
            else "*"
            if (not np.isnan(p) and p < 0.05)
            else "ns"
        )
        lines.append(f"  {label}: {format_p(p)} {sig}")

    lines.append("Speed comparisons:")
    for label, p in speed_items:
        sig = (
            "***"
            if (not np.isnan(p) and p < 0.001)
            else "**"
            if (not np.isnan(p) and p < 0.01)
            else "*"
            if (not np.isnan(p) and p < 0.05)
            else "ns"
        )
        lines.append(f"  {label}: {format_p(p)} {sig}")

    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=7,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )


def add_threshold_lines(ax, thresholds=None, color=None, horizontal=False):
    """Add threshold lines to a plot."""
    if thresholds is None:
        thresholds = CONFIG["threshold_lines"]
    if color is None:
        color = CONFIG["threshold_color"]
    for t in thresholds:
        if horizontal:
            ax.axhline(y=t, color=color, linestyle="--", alpha=0.7, linewidth=1)
        else:
            ax.axvline(x=t, color=color, linestyle="--", alpha=0.7, linewidth=1)


def figure_rmsd_distribution(unified_df: pd.DataFrame, output_path: str):
    """RMSD distribution - cumulative lines for all methods."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping")
        return

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Calculate cumulative distribution for each method
    x_range = np.linspace(0, 12, 200)

    for method in CONFIG["method_order"]:
        data = unified_df[unified_df["method"] == method]["rmsd"].dropna()
        if len(data) > 0:
            # Sort for cumulative
            sorted_data = np.sort(data)
            cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data) * 100

            ax.plot(
                sorted_data,
                cumulative,
                label=f"{method} (n={len(data)})",
                color=CONFIG["method_colors"].get(method),
                linewidth=2,
            )

    ax.axvline(x=2.0, color="red", linestyle="--", alpha=0.7, label="2Å threshold")
    ax.set_xlabel("RMSD (Å)")
    ax.set_ylabel("Cumulative Percentage (%)")
    ax.set_title("Cumulative Distribution of RMSD")
    ax.legend()
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=CONFIG["figure_dpi"], bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def figure_success_rate(
    unified_df: pd.DataFrame, output_path: str, stats_results: dict = None
):
    """Success rate scatter plot with log scale x-axis."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping")
        return

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for method in CONFIG["method_order"]:
        subset = unified_df[unified_df["method"] == method]
        subset_data = []

        for thresh in CONFIG["success_thresholds"]:
            success = (subset["rmsd"] < thresh).sum()
            total = subset["rmsd"].notna().sum()
            subset_data.append(
                {
                    "threshold": thresh,
                    "success_rate": success / total * 100 if total > 0 else 0,
                }
            )

        df_method = pd.DataFrame(subset_data)
        ax.scatter(
            df_method["threshold"],
            df_method["success_rate"],
            s=80,
            label=method,
            color=CONFIG["method_colors"].get(method),
            alpha=CONFIG["scatter_alpha"],
            edgecolor=CONFIG["scatter_edgecolor"],
            linewidths=CONFIG["scatter_linewidth"],
        )
        ax.plot(
            df_method["threshold"],
            df_method["success_rate"],
            "-",
            color=CONFIG["method_colors"].get(method),
            alpha=0.5,
            linewidth=1,
        )

    ax.set_xlabel("RMSD Threshold (Å)")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Success Rate by RMSD Threshold")
    ax.set_xscale("log")
    ax.set_xlim(0.08, 6)
    ax.set_xticks(CONFIG["success_thresholds"])
    ax.set_xticklabels(
        [str(t) for t in CONFIG["success_thresholds"]], rotation=45, ha="right"
    )
    add_threshold_lines(ax)
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    ax.set_ylim(0, 105)

    if stats_results is not None and "success_rate" in stats_results:
        add_success_rate_significance(ax, unified_df, stats_results)

    plt.tight_layout()
    plt.savefig(output_path, dpi=CONFIG["figure_dpi"], bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def figure_timing_distribution(unified_df: pd.DataFrame, output_path: str):
    """Timing distribution - cumulative lines for all methods."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping")
        return

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for method in CONFIG["method_order"]:
        data = unified_df[unified_df["method"] == method]["time_s"].dropna()
        if len(data) > 0:
            sorted_data = np.sort(data)
            cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data) * 100
            ax.plot(
                sorted_data,
                cumulative,
                label=f"{method} (n={len(data)})",
                color=CONFIG["method_colors"].get(method),
                linewidth=2,
            )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cumulative Percentage (%)")
    ax.set_title("Cumulative Runtime Distribution")
    ax.legend()
    ax.ticklabel_format(style="plain", axis="x")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(output_path, dpi=CONFIG["figure_dpi"], bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def figure_rmsd_by_target_class(
    unified_df: pd.DataFrame,
    output_path: str,
    stats_results: dict = None,
    value_col: str = "rmsd",
    use_log_y: bool = False,
):
    """Multi-bar chart: mean RMSD by enzyme class for each method."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping")
        return

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if "target_class" not in unified_df.columns:
        print("No target_class column, skipping")
        return

    df = unified_df[unified_df["target_class"].notna()].copy()
    classes = sorted(df["target_class"].dropna().unique())
    # Add "all" category at the beginning
    classes = ["all"] + classes
    methods = CONFIG["method_order"]

    bar_data = {}
    for cls in classes:
        bar_data[cls] = {}
        if cls == "all":
            # Use all data for "all" category
            for method in methods:
                subset = df[df["method"] == method]
                values = subset[value_col].dropna()
                if len(values) > 0:
                    bar_data[cls][method] = {
                        "mean": values.mean(),
                        "std": values.std(),
                        "n": len(values),
                        "values": values.values,
                    }
        else:
            for method in methods:
                subset = df[(df["target_class"] == cls) & (df["method"] == method)]
                values = subset[value_col].dropna()
                if len(values) > 0:
                    bar_data[cls][method] = {
                        "mean": values.mean(),
                        "std": values.std(),
                        "n": len(values),
                        "values": values.values,
                    }

    n_classes = len(classes)
    n_methods = len(methods)

    fig, ax = plt.subplots(1, 1, figsize=(max(10, n_classes * 2), 6))

    x = np.arange(n_classes)
    width = 0.2

    for i, method in enumerate(methods):
        means = []
        for cls in classes:
            if method in bar_data[cls]:
                means.append(bar_data[cls][method]["mean"])
            else:
                means.append(0)

        offset = (i - n_methods / 2 + 0.5) * width
        ax.bar(
            x + offset,
            means,
            width,
            label=method,
            color=CONFIG["method_colors"].get(method),
            alpha=0.8,
        )

        # Add scatter points for individual data
        dq_methods = ["DQv1", "DQv2"]
        for j, cls in enumerate(classes):
            if method in bar_data[cls]:
                values = bar_data[cls][method]["values"]
                jitter = np.random.uniform(-width / 3, width / 3, len(values))
                if method not in dq_methods and value_col == "rmsd":
                    alphas = np.where(
                        values > 2,
                        CONFIG.get("scatter_alpha_transparent", 0.4),
                        CONFIG["scatter_alpha"],
                    )
                else:
                    alphas = CONFIG["scatter_alpha"]
                ax.scatter(
                    x[j] + offset + jitter,
                    values,
                    color=CONFIG["method_colors"].get(method),
                    alpha=alphas,
                    s=CONFIG["scatter_size"],
                    edgecolor=CONFIG["scatter_edgecolor"],
                    linewidths=CONFIG["scatter_linewidth"],
                )

    unit_label = "Å" if value_col == "rmsd" else "s"
    if use_log_y:
        ax.set_yscale("log")
        ax.set_title(f"Mean {value_col.upper()} by Enzyme Class and Method (Log Scale)")
        from matplotlib.ticker import ScalarFormatter

        ax.yaxis.set_major_formatter(ScalarFormatter())
    else:
        ax.set_title(f"Mean {value_col.upper()} by Enzyme Class and Method")
    ax.set_xlabel("Enzyme Class")
    ax.set_ylabel(f"Mean {value_col.upper()} ({unit_label})")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    if value_col == "rmsd":
        add_threshold_lines(ax, horizontal=True)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    if stats_results is not None:
        add_bar_significance(
            ax,
            x,
            CONFIG["method_order"],
            stats_results,
            value_col,
            classes=classes,
            bar_data=bar_data,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=CONFIG["figure_dpi"], bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def figure_time_by_target_class(
    unified_df: pd.DataFrame,
    output_path: str,
    stats_results: dict = None,
    use_log_y: bool = False,
):
    figure_rmsd_by_target_class(
        unified_df, output_path, stats_results, value_col="time_s", use_log_y=use_log_y
    )


def figure_rmsd_scatter_comparison(
    unified_df: pd.DataFrame, output_path: str, stats_results: dict = None
):
    """Scatter plot comparing DQ RMSD vs competitors."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping")
        return

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    dq_methods = ["DQv1", "DQv2"]
    competitors = ["GNINA", "Smina"]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Plot each DQ vs each competitor
    for dq in dq_methods:
        dq_data = unified_df[unified_df["method"] == dq].set_index("pdb_id")

        for comp in competitors:
            comp_data = unified_df[unified_df["method"] == comp].set_index("pdb_id")

            # Get common PDBs
            common = dq_data.index.intersection(comp_data.index)

            x = dq_data.loc[common, "rmsd"].values
            y = comp_data.loc[common, "rmsd"].values

            # Different markers for DQ version
            marker_map = {
                ("DQv1", "GNINA"): "o",
                ("DQv1", "Smina"): "^",
                ("DQv2", "GNINA"): "s",
                ("DQv2", "Smina"): "D",
            }
            # Different colors/shades for each combo
            color_map = {
                ("DQv1", "GNINA"): "#90EE90",  # light green
                ("DQv1", "Smina"): "#FFB6C1",  # light red/pink
                ("DQv2", "GNINA"): "#228B22",  # forest green
                ("DQv2", "Smina"): "#CD5C5C",  # indian red
            }

            marker = marker_map.get((dq, comp), "o")
            color = color_map.get((dq, comp), "gray")

            ax.scatter(
                x,
                y,
                alpha=CONFIG["scatter_alpha"],
                s=30,
                c=color,
                marker=marker,
                edgecolor=CONFIG["scatter_edgecolor"],
                linewidths=CONFIG["scatter_linewidth"],
                label=f"{comp} vs {dq} (n={len(common)})",
            )

    limit = unified_df["rmsd"].max()
    limit = max(limit, 10)
    ax.plot([0, limit], [0, limit], "k--", alpha=0.3, label="y=x")
    ax.set_xlabel("DQ RMSD (Å)")
    ax.set_ylabel("Competitor RMSD (Å)")
    ax.set_title("DQ vs Competitors")
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)

    # Add threshold lines - only 2 extends across intersection, others stay limited
    thresholds = CONFIG["threshold_lines"]
    for t in thresholds:
        if t == 2:
            # Full line across the graph
            ax.axvline(x=t, color="red", linestyle="--", alpha=0.7, linewidth=1)
            ax.axhline(y=t, color="red", linestyle="--", alpha=0.7, linewidth=1)
        else:
            # Lines only from origin to intersection
            if t <= limit:
                ax.vlines(
                    x=t,
                    ymin=0,
                    ymax=t,
                    color="red",
                    linestyle="--",
                    alpha=0.7,
                    linewidth=1,
                )
                ax.hlines(
                    y=t,
                    xmin=0,
                    xmax=t,
                    color="red",
                    linestyle="--",
                    alpha=0.7,
                    linewidth=1,
                )

    ax.legend(fontsize=8)
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(output_path, dpi=CONFIG["figure_dpi"], bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def figure_time_vs_rmsd(
    unified_df: pd.DataFrame, output_path: str, stats_results: dict = None
):
    """Scatter plot of runtime vs RMSD for all methods."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping")
        return

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    dq_methods = ["DQv1", "DQv2"]

    for method in CONFIG["method_order"]:
        subset = unified_df[unified_df["method"] == method]

        # Make non-DQ points transparent if RMSD > 2 (not time)
        if method not in dq_methods:
            mask = subset["rmsd"] > 2
            alpha = np.where(
                mask,
                CONFIG.get("scatter_alpha_transparent", 0.4),
                CONFIG["scatter_alpha"],
            )
        else:
            alpha = CONFIG["scatter_alpha"]

        ax.scatter(
            subset["time_s"],
            subset["rmsd"],
            alpha=alpha,
            s=20,
            label=method,
            color=CONFIG["method_colors"].get(method),
            edgecolor=CONFIG["scatter_edgecolor"],
            linewidths=CONFIG["scatter_linewidth"],
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RMSD (Å)")
    ax.set_title("Runtime vs RMSD")
    add_threshold_lines(ax, horizontal=True)
    ax.legend()
    ax.ticklabel_format(style="plain", axis="x")

    plt.tight_layout()
    plt.savefig(output_path, dpi=CONFIG["figure_dpi"], bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def figure_time_scatter_comparison(
    unified_df: pd.DataFrame, output_path: str, stats_results: dict = None
):
    """Scatter plot comparing DQ time vs competitors."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping")
        return

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    dq_methods = ["DQv1", "DQv2"]
    competitors = ["GNINA", "Smina"]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    for dq in dq_methods:
        dq_data = unified_df[unified_df["method"] == dq].set_index("pdb_id")

        for comp in competitors:
            comp_data = unified_df[unified_df["method"] == comp].set_index("pdb_id")

            common = dq_data.index.intersection(comp_data.index)

            x = dq_data.loc[common, "time_s"].values
            y = comp_data.loc[common, "time_s"].values

            marker_map = {
                ("DQv1", "GNINA"): "o",
                ("DQv1", "Smina"): "^",
                ("DQv2", "GNINA"): "s",
                ("DQv2", "Smina"): "D",
            }
            color_map = {
                ("DQv1", "GNINA"): "#90EE90",
                ("DQv1", "Smina"): "#FFB6C1",
                ("DQv2", "GNINA"): "#228B22",
                ("DQv2", "Smina"): "#CD5C5C",
            }

            marker = marker_map.get((dq, comp), "o")
            color = color_map.get((dq, comp), "gray")

            ax.scatter(
                x,
                y,
                alpha=CONFIG["scatter_alpha"],
                s=30,
                c=color,
                marker=marker,
                edgecolor=CONFIG["scatter_edgecolor"],
                linewidths=CONFIG["scatter_linewidth"],
                label=f"{comp} vs {dq} (n={len(common)})",
            )

    limit = unified_df["time_s"].max()
    limit = max(limit, 10)
    ax.plot([0, limit], [0, limit], "k--", alpha=0.3, label="y=x")
    ax.set_xlabel("DQ Time (s)")
    ax.set_ylabel("Competitor Time (s)")
    ax.set_title("DQ vs Competitors (Runtime)")
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.legend(fontsize=8)
    ax.ticklabel_format(style="plain", axis="both")
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(output_path, dpi=CONFIG["figure_dpi"], bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def generate_all_figures(csv_path: str, output_dir: str):
    """Generate all figures."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("Preparing unified data...")
    unified_df = prepare_unified_data(csv_path)

    print("Computing statistics...")
    stats_results = compute_statistics(unified_df)

    print(f"Generated unified data with {len(unified_df)} rows")
    print(f"Methods: {unified_df['method'].value_counts().to_dict()}")

    figure_success_rate(unified_df, f"{output_dir}/success_rate.pdf", stats_results)
    figure_rmsd_by_target_class(
        unified_df, f"{output_dir}/rmsd_by_class.pdf", stats_results
    )
    figure_rmsd_by_target_class(
        unified_df, f"{output_dir}/rmsd_by_class_log.pdf", stats_results, use_log_y=True
    )
    figure_time_by_target_class(
        unified_df, f"{output_dir}/time_by_class.pdf", stats_results
    )
    figure_time_by_target_class(
        unified_df, f"{output_dir}/time_by_class_log.pdf", stats_results, use_log_y=True
    )
    figure_rmsd_scatter_comparison(
        unified_df, f"{output_dir}/rmsd_scatter.pdf", stats_results
    )
    figure_time_scatter_comparison(
        unified_df, f"{output_dir}/time_scatter.pdf", stats_results
    )
    figure_time_vs_rmsd(unified_df, f"{output_dir}/time_vs_rmsd.pdf", stats_results)

    print(f"All figures saved to {output_dir}/")


if __name__ == "__main__":
    import sys
    import os

    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = os.path.join(os.path.dirname(__file__), "combined_results.csv")

    output_dir = os.path.join(os.path.dirname(__file__), "figures")
    generate_all_figures(csv_path, output_dir)
