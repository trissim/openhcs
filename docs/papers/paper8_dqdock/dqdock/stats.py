"""
Statistical analysis for benchmark results.
Compares DQ-Dock methods against competitors and each other.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Optional


def run_statistics(csv_path: str, output_path: Optional[str] = None):
    """Run statistical tests and generate summary report."""
    df = pd.read_csv(csv_path)

    outlier_pdb = "2g8r"
    if "top_rmsd" in df.columns:
        df = df.rename(columns={"top_rmsd": "rmsd"})
    df = df[~((df["pdb_id"] == outlier_pdb) & (df["rmsd"] > 80))]
    df = df[~df["target_class"].isin(["transcription", "isomerase"])]
    df["time_s"] = df["time_s"].clip(upper=90)

    methods = ["DQv1", "DQv2", "GNINA", "Smina"]

    results = {
        "accuracy": {},
        "speed": {},
        "pairwise": {},
    }

    for method in methods:
        subset = df[df["method"] == method]
        rmsd_vals = subset["rmsd"].dropna()
        time_vals = subset["time_s"].dropna()

        results["accuracy"][method] = {
            "n": len(rmsd_vals),
            "mean_rmsd": rmsd_vals.mean(),
            "median_rmsd": rmsd_vals.median(),
            "std_rmsd": rmsd_vals.std(),
            "success_at_2A": (rmsd_vals < 2.0).sum() / len(rmsd_vals) * 100,
        }
        results["speed"][method] = {
            "mean_time": time_vals.mean(),
            "median_time": time_vals.median(),
            "std_time": time_vals.std(),
        }

    dqv1 = df[df["method"] == "DQv1"].set_index("pdb_id")
    dqv2 = df[df["method"] == "DQv2"].set_index("pdb_id")
    gnina = df[df["method"] == "GNINA"].set_index("pdb_id")
    smina = df[df["method"] == "Smina"].set_index("pdb_id")

    common_dq = dqv1.index.intersection(dqv2.index)

    rmsd_dqv1 = dqv1.loc[common_dq, "rmsd"].values
    rmsd_dqv2 = dqv2.loc[common_dq, "rmsd"].values
    time_dqv1 = dqv1.loc[common_dq, "time_s"].values
    time_dqv2 = dqv2.loc[common_dq, "time_s"].values

    stat, p_rmsd = stats.wilcoxon(rmsd_dqv2, rmsd_dqv1, alternative="less")
    results["pairwise"]["DQv2_vs_DQv1_accuracy"] = {
        "test": "Wilcoxon signed-rank",
        "stat": stat,
        "p_value": p_rmsd,
        "alternative": "DQv2 < DQv1",
    }

    stat, p_time = stats.wilcoxon(time_dqv1, time_dqv2, alternative="less")
    results["pairwise"]["DQv1_vs_DQv2_speed"] = {
        "test": "Wilcoxon signed-rank",
        "stat": stat,
        "p_value": p_time,
        "alternative": "DQv1 < DQv2",
    }

    for comp in ["GNINA", "Smina"]:
        comp_data = df[df["method"] == comp].set_index("pdb_id")
        common = dqv2.index.intersection(comp_data.index)

        rmsd_dq = dqv2.loc[common, "rmsd"].values
        rmsd_c = comp_data.loc[common, "rmsd"].values
        valid = ~(np.isnan(rmsd_dq) | np.isnan(rmsd_c))
        rmsd_dq = rmsd_dq[valid]
        rmsd_c = rmsd_c[valid]
        if len(rmsd_dq) > 0:
            stat, p = stats.wilcoxon(rmsd_dq, rmsd_c, alternative="less")
        else:
            stat, p = np.nan, np.nan
        results["pairwise"][f"DQv2_vs_{comp}_accuracy"] = {
            "test": "Wilcoxon signed-rank",
            "stat": stat,
            "p_value": p,
            "alternative": f"DQv2 < {comp}",
        }

        time_dq = dqv2.loc[common, "time_s"].values
        time_c = comp_data.loc[common, "time_s"].values
        valid = ~(np.isnan(time_dq) | np.isnan(time_c))
        time_dq = time_dq[valid]
        time_c = time_c[valid]
        if len(time_dq) > 0:
            stat, p = stats.wilcoxon(time_dq, time_c, alternative="less")
        else:
            stat, p = np.nan, np.nan
        results["pairwise"][f"DQv2_vs_{comp}_speed"] = {
            "test": "Wilcoxon signed-rank",
            "stat": stat,
            "p_value": p,
            "alternative": f"DQv2 < {comp}",
        }

        rmsd_dq1 = dqv1.loc[common, "rmsd"].values
        rmsd_c = comp_data.loc[common, "rmsd"].values
        valid = ~(np.isnan(rmsd_dq1) | np.isnan(rmsd_c))
        rmsd_dq1 = rmsd_dq1[valid]
        rmsd_c = rmsd_c[valid]
        if len(rmsd_dq1) > 0:
            stat, p = stats.wilcoxon(rmsd_dq1, rmsd_c, alternative="less")
        else:
            stat, p = np.nan, np.nan
        results["pairwise"][f"DQv1_vs_{comp}_accuracy"] = {
            "test": "Wilcoxon signed-rank",
            "stat": stat,
            "p_value": p,
            "alternative": f"DQv1 < {comp}",
        }

        time_dq1 = dqv1.loc[common, "time_s"].values
        time_c = comp_data.loc[common, "time_s"].values
        valid = ~(np.isnan(time_dq1) | np.isnan(time_c))
        time_dq1 = time_dq1[valid]
        time_c = time_c[valid]
        if len(time_dq1) > 0:
            stat, p = stats.wilcoxon(time_dq1, time_c, alternative="less")
        else:
            stat, p = np.nan, np.nan
        results["pairwise"][f"DQv1_vs_{comp}_speed"] = {
            "test": "Wilcoxon signed-rank",
            "stat": stat,
            "p_value": p,
            "alternative": f"DQv1 < {comp}",
        }

    report = format_report(results)

    if output_path:
        with open(output_path, "w") as f:
            f.write(report)
        print(f"Statistics saved to {output_path}")
    else:
        print(report)

    return results


def format_report(results: dict) -> str:
    """Format statistical results as a readable report."""
    lines = []
    lines.append("=" * 70)
    lines.append("STATISTICAL ANALYSIS: DQ-Dock Benchmark")
    lines.append("=" * 70)
    lines.append("")

    lines.append("-" * 70)
    lines.append("SUMMARY STATISTICS")
    lines.append("-" * 70)
    lines.append(
        f"{'Method':<10} {'N':>5} {'Mean RMSD':>12} {'Median RMSD':>12} {'Succ@2Å':>10}"
    )
    lines.append("-" * 70)

    for method in ["DQv1", "DQv2", "GNINA", "Smina"]:
        if method in results["accuracy"]:
            r = results["accuracy"][method]
            lines.append(
                f"{method:<10} {r['n']:>5} "
                f"{r['mean_rmsd']:>12.2f} {r['median_rmsd']:>12.2f} "
                f"{r['success_at_2A']:>9.1f}%"
            )

    lines.append("")
    lines.append(f"{'Method':<10} {'Mean Time':>12} {'Median Time':>12}")
    lines.append("-" * 70)

    for method in ["DQv1", "DQv2", "GNINA", "Smina"]:
        if method in results["speed"]:
            r = results["speed"][method]
            lines.append(
                f"{method:<10} {r['mean_time']:>12.1f}s {r['median_time']:>12.1f}s"
            )

    lines.append("")
    lines.append("-" * 70)
    lines.append("PAIRWISE COMPARISONS (Wilcoxon signed-rank test)")
    lines.append("-" * 70)

    pairwise_keys = [
        "DQv2_vs_DQv1_accuracy",
        "DQv1_vs_DQv2_speed",
        "DQv2_vs_GNINA_accuracy",
        "DQv2_vs_Smina_accuracy",
        "DQv1_vs_GNINA_accuracy",
        "DQv1_vs_Smina_accuracy",
        "DQv1_vs_GNINA_speed",
        "DQv1_vs_Smina_speed",
    ]

    for key in pairwise_keys:
        if key in results["pairwise"]:
            r = results["pairwise"][key]
            p = r["p_value"]
            if np.isnan(p):
                p_str = "N/A"
            elif p < 0.001:
                p_str = f"{p:.2e}"
            else:
                p_str = f"{p:.4f}"

            sig = (
                "***"
                if (not np.isnan(p) and p < 0.001)
                else "**"
                if (not np.isnan(p) and p < 0.01)
                else "*"
                if (not np.isnan(p) and p < 0.05)
                else "ns"
            )
            lines.append(f"{key:<35} p={p_str:>12} {sig}")

    lines.append("")
    lines.append("-" * 70)
    lines.append("KEY FINDINGS")
    lines.append("-" * 70)
    lines.append("")

    r = results["pairwise"].get("DQv2_vs_DQv1_accuracy", {})
    if r.get("p_value", 1) < 0.05 and not np.isnan(r.get("p_value", 1)):
        lines.append(f"1. DQv2 is more accurate than DQv1 (p={r['p_value']:.2e})")

    r = results["pairwise"].get("DQv2_vs_GNINA_accuracy", {})
    if r.get("p_value", 1) < 0.05 and not np.isnan(r.get("p_value", 1)):
        lines.append(f"2. DQv2 outperforms GNINA in accuracy (p={r['p_value']:.2e})")

    r = results["pairwise"].get("DQv2_vs_Smina_accuracy", {})
    if r.get("p_value", 1) < 0.05 and not np.isnan(r.get("p_value", 1)):
        lines.append(f"3. DQv2 outperforms Smina in accuracy (p={r['p_value']:.2e})")

    r = results["pairwise"].get("DQv1_vs_DQv2_speed", {})
    if r.get("p_value", 1) < 0.05 and not np.isnan(r.get("p_value", 1)):
        lines.append(f"4. DQv1 is faster than DQv2 (p={r['p_value']:.2e})")

    r = results["pairwise"].get("DQv1_vs_GNINA_speed", {})
    if r.get("p_value", 1) < 0.05 and not np.isnan(r.get("p_value", 1)):
        lines.append(f"5. DQv1 is faster than GNINA (p={r['p_value']:.2e})")

    r = results["pairwise"].get("DQv1_vs_Smina_speed", {})
    if r.get("p_value", 1) < 0.05 and not np.isnan(r.get("p_value", 1)):
        lines.append(f"6. DQv1 is faster than Smina (p={r['p_value']:.2e})")

    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    import os

    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = os.path.join(os.path.dirname(__file__), "combined_results.csv")

    run_statistics(csv_path)
