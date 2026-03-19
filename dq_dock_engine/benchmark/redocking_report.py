from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _graph_label(pdb_id: str, target_name: str) -> str:
    wrapped_target = textwrap.fill(target_name, width=16)
    return f"{pdb_id}\n{wrapped_target}"


def _load_payload(json_path: Path) -> dict:
    with open(json_path) as f:
        return json.load(f)


def _write_markdown_report(payload: dict, report_path: Path) -> None:
    summary = payload["summary"]
    dq_rows = payload["dq_dock"]
    competitor_rows = payload["competitors"]
    excluded_rows = payload["excluded"]

    lines = [
        "# PDB Redocking Benchmark",
        "",
        f"- Phase: `{summary['phase']}`",
        f"- Charge method: `{summary['charge_method']}`",
        f"- Requested complexes: `{summary['n_complexes_requested']}`",
        f"- Completed DQ-Dock complexes: `{summary['n_complexes_run']}`",
        f"- Excluded complexes: `{summary['n_complexes_excluded']}`",
        f"- Poses: `{summary['n_poses']}`",
        f"- Optimization steps: `{summary['n_opt_steps']}`",
        f"- Pocket-guided sampling: `{summary['use_pocket_guided']}`",
        f"- Competitors: `{', '.join(item['display_name'] for item in summary['competitors'])}`",
        "",
        "## Summary",
        "",
        f"- DQ-Dock avg RMSD: `{summary['dq_avg_rmsd']}`",
        f"- DQ-Dock total time (s): `{summary['dq_total_time_s']}`",
        f"- Total benchmark time (s): `{summary['total_benchmark_time_s']}`",
        "",
        "## DQ-Dock",
        "",
        "| PDB | Target | RMSD | Time (s) | Energy | Gap Proof | Native Rank | Energy Gap |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |",
    ]

    for row in dq_rows:
        lines.append(
            f"| {row['pdb_id']} | {row['target_name']} | {row['rmsd']:.3f} | {row['time_s']:.3f} | {row['energy']:.3f} | {row['gap_proof']} | {row['native_rank']} | {row['energy_gap']} |"
        )

    for competitor in summary["competitors"]:
        engine_id = competitor["engine_id"]
        stats = summary["competitor_stats"][engine_id]
        rows = [row for row in competitor_rows if row["engine_id"] == engine_id]
        lines.extend(
            [
                "",
                f"## {competitor['display_name']}",
                "",
                f"- Score: `{competitor['score_name']}`",
                f"- Prep: `{competitor['prep_protocol']}`",
                f"- Completed complexes: `{stats['n_completed']}`",
                f"- Successful complexes: `{stats['n_successful']}`",
                f"- Avg top RMSD: `{stats['avg_top_rmsd']}`",
                f"- Avg best-returned RMSD: `{stats['avg_best_mode_rmsd']}`",
                f"- Total time (s): `{stats['total_time_s']}`",
                "",
                "| PDB | Target | Status | Top RMSD | Best Returned RMSD | Time (s) | Score | Error |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in rows:
            score = row["score"] if row["score"] is not None else "nan"
            top_rmsd = row["top_rmsd"] if row["top_rmsd"] is not None else "nan"
            best_rmsd = (
                row["best_returned_mode_rmsd"]
                if row["best_returned_mode_rmsd"] is not None
                else "nan"
            )
            lines.append(
                f"| {row['pdb_id']} | {row['target_name']} | {'success' if row['success'] else 'failure'} | {top_rmsd} | {best_rmsd} | {row['time_s']:.3f} | {score} | {row['error']} |"
            )

    if excluded_rows:
        lines.extend(
            [
                "",
                "## Excluded",
                "",
                "| PDB | Reason |",
                "| --- | --- |",
            ]
        )
        for row in excluded_rows:
            lines.append(f"| {row['pdb_id']} | {row['reason']} |")

    report_path.write_text("\n".join(lines) + "\n")


def _plot_rmsd(payload: dict, output_path: Path) -> None:
    dq_df = pd.DataFrame(payload["dq_dock"])
    competitor_df = pd.DataFrame(payload["competitors"])
    if dq_df.empty:
        return

    dq_plot = dq_df[["pdb_id", "target_name", "rmsd"]].copy()
    dq_plot["Method"] = "DQ-Dock"
    dq_plot["RMSD"] = dq_plot.pop("rmsd")
    frames = [dq_plot[["pdb_id", "target_name", "Method", "RMSD"]]]
    if not competitor_df.empty:
        successful_competitors = competitor_df[competitor_df["success"]].copy()
        if not successful_competitors.empty:
            top_plot = successful_competitors[
                ["pdb_id", "target_name", "display_name", "top_rmsd"]
            ].copy()
            top_plot["Method"] = top_plot["display_name"] + " top"
            top_plot["RMSD"] = top_plot.pop("top_rmsd")
            best_plot = successful_competitors[
                ["pdb_id", "target_name", "display_name", "best_returned_mode_rmsd"]
            ].copy()
            best_plot["Method"] = best_plot["display_name"] + " best"
            best_plot["RMSD"] = best_plot.pop("best_returned_mode_rmsd")
            frames.extend(
                [
                    top_plot[["pdb_id", "target_name", "Method", "RMSD"]],
                    best_plot[["pdb_id", "target_name", "Method", "RMSD"]],
                ]
            )
    plot_df = cast(pd.DataFrame, pd.concat(frames, ignore_index=True))
    plot_df["label"] = plot_df.apply(
        lambda row: _graph_label(str(row["pdb_id"]), str(row["target_name"])), axis=1
    )
    plot_df = cast(pd.DataFrame, plot_df.dropna(subset=["RMSD"]))
    if plot_df.empty:
        return

    label_count = len(set(plot_df["label"].tolist()))
    plt.figure(figsize=(max(12, 1.5 * label_count), 7.5))
    sns.barplot(data=cast(pd.DataFrame, plot_df), x="label", y="RMSD", hue="Method")
    plt.axhline(2.0, color="red", linestyle="--", linewidth=1, label="2A success")
    plt.ylabel("RMSD (A)")
    plt.xlabel("Complex")
    plt.title("Redocking RMSD by Complex")
    plt.xticks(rotation=0, ha="center", fontsize=8)
    plt.subplots_adjust(bottom=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def _plot_timing(payload: dict, output_path: Path, *, log_scale: bool) -> None:
    dq_df = pd.DataFrame(payload["dq_dock"])
    competitor_df = pd.DataFrame(payload["competitors"])
    if dq_df.empty:
        return

    dq_plot = dq_df[["pdb_id", "time_s"]].copy()
    dq_plot["Method"] = "DQ-Dock"
    dq_plot["Time"] = dq_plot.pop("time_s")
    dq_plot["label"] = dq_df.apply(
        lambda row: _graph_label(str(row["pdb_id"]), str(row["target_name"])), axis=1
    )
    frames = [dq_plot]
    if not competitor_df.empty:
        successful_competitors = competitor_df[competitor_df["success"]].copy()
        if not successful_competitors.empty:
            competitor_plot = successful_competitors[
                ["pdb_id", "target_name", "display_name", "time_s"]
            ].copy()
            competitor_plot["Method"] = competitor_plot.pop("display_name")
            competitor_plot["Time"] = competitor_plot.pop("time_s")
            competitor_plot["label"] = competitor_plot.apply(
                lambda row: _graph_label(str(row["pdb_id"]), str(row["target_name"])),
                axis=1,
            )
            frames.append(competitor_plot[["pdb_id", "Method", "Time", "label"]])
    plot_df = cast(pd.DataFrame, pd.concat(frames, ignore_index=True))

    label_count = len(set(plot_df["label"].tolist()))
    plt.figure(figsize=(max(12, 1.5 * label_count), 7.5))
    sns.barplot(data=cast(pd.DataFrame, plot_df), x="label", y="Time", hue="Method")
    if log_scale:
        plt.yscale("log")
        plt.ylabel("Runtime (s, log scale)")
        plt.title("Runtime by Complex (Log Scale)")
    else:
        plt.ylabel("Runtime (s)")
        plt.title("Runtime by Complex")
    plt.xlabel("Complex")
    plt.xticks(rotation=0, ha="center", fontsize=8)
    plt.subplots_adjust(bottom=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def _plot_scatter(payload: dict, output_path: Path) -> None:
    dq_df = pd.DataFrame(payload["dq_dock"])
    competitor_df = pd.DataFrame(payload["competitors"])
    if dq_df.empty or competitor_df.empty:
        return

    successful_competitors = competitor_df[competitor_df["success"]].copy()
    merged = dq_df[["pdb_id", "rmsd"]].merge(
        successful_competitors[["pdb_id", "display_name", "best_returned_mode_rmsd"]],
        on="pdb_id",
        how="inner",
    )
    if merged.empty:
        return

    limit = max(
        float(merged["rmsd"].max()), float(merged["best_returned_mode_rmsd"].max()), 2.0
    )
    plt.figure(figsize=(6.5, 6.5))
    sns.scatterplot(
        data=merged,
        x="best_returned_mode_rmsd",
        y="rmsd",
        hue="display_name",
        s=90,
    )
    for _, row in merged.iterrows():
        plt.text(
            float(row["best_returned_mode_rmsd"]),
            float(row["rmsd"]),
            str(row["pdb_id"]),
            fontsize=8,
        )
    plt.plot([0, limit], [0, limit], linestyle="--", color="gray")
    plt.xlabel("Competitor best returned RMSD (A)")
    plt.ylabel("DQ-Dock RMSD (A)")
    plt.title("DQ-Dock vs Competitor RMSD")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def render_redocking_report(json_path: Path) -> dict[str, Path]:
    payload = _load_payload(json_path)
    base = json_path.with_suffix("")
    markdown_path = base.with_name(base.name + "_report.md")
    rmsd_path = base.with_name(base.name + "_rmsd.png")
    timing_path = base.with_name(base.name + "_timing.png")
    timing_log_path = base.with_name(base.name + "_timing_log.png")
    scatter_path = base.with_name(base.name + "_scatter.png")

    _write_markdown_report(payload, markdown_path)
    _plot_rmsd(payload, rmsd_path)
    _plot_timing(payload, timing_path, log_scale=False)
    _plot_timing(payload, timing_log_path, log_scale=True)
    _plot_scatter(payload, scatter_path)

    return {
        "markdown": markdown_path,
        "rmsd": rmsd_path,
        "timing": timing_path,
        "timing_log": timing_log_path,
        "scatter": scatter_path,
    }
