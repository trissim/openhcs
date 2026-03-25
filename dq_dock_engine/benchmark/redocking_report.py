from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import Callable, cast

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

csv.field_size_limit(max(csv.field_size_limit(), 10_000_000))

try:
    import seaborn as sns
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    sns = None

from dq_dock_engine.docking.pdb_io import parse_structure


PYMOL_METHOD_STYLES: dict[str, tuple[str, tuple[int, int, int]]] = {
    "native": ("black", (20, 20, 20)),
    "DQ-Dock": ("tv_orange", (217, 119, 6)),
    "GNINA": ("marine", (5, 150, 105)),
}


@dataclass(frozen=True)
class PyMOLPoseMethod:
    label: str
    object_name: str
    pose_path: Path
    color_name: str
    color_rgb: tuple[int, int, int]
    rmsd_text: str


@dataclass(frozen=True)
class PyMOLPoseScene:
    pdb_id: str
    target_name: str
    receptor_path: Path
    native_path: Path
    methods: tuple[PyMOLPoseMethod, ...]
    script_path: Path
    front_png_path: Path
    side_png_path: Path
    pse_path: Path


def _barplot(*, data: pd.DataFrame, x: str, y: str, hue: str) -> None:
    if sns is not None:
        sns.barplot(data=data, x=x, y=y, hue=hue)
        return
    pivot = data.pivot_table(index=x, columns=hue, values=y, aggfunc="mean")
    pivot.plot(kind="bar", ax=plt.gca())


def _scatterplot(*, data: pd.DataFrame, x: str, y: str, hue: str, s: int) -> None:
    if sns is not None:
        sns.scatterplot(data=data, x=x, y=y, hue=hue, s=s)
        return
    for label, group in data.groupby(hue):
        plt.scatter(group[x], group[y], s=s, label=str(label))


def _fmt_number(value: object, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if pd.isna(numeric):
        return "n/a"
    return f"{numeric:.{digits}f}"


def _fmt_text(value: object) -> str:
    return "n/a" if value is None else str(value)


def _graph_label(pdb_id: str, target_name: str) -> str:
    wrapped_target = textwrap.fill(target_name, width=20)
    return f"{wrapped_target}\n({pdb_id})"


def _load_payload(json_path: Path) -> dict:
    with open(json_path) as f:
        payload = json.load(f)
    for row in payload.get("dq_dock", []):
        if "success" not in row:
            row["success"] = row.get("status", "success") == "success"
        row.setdefault("status", "success" if row["success"] else "failure")
        row.setdefault("error", None)
    return payload


def _read_structure_coords(pdb_path: Path) -> np.ndarray:
    try:
        coords, _ = cast(
            tuple[np.ndarray, np.ndarray],
            parse_structure(pdb_path, strip_hydrogens=True),
        )
    except ValueError:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(coords, dtype=np.float64)


def _focus_receptor_coords(
    receptor_coords: np.ndarray, native_coords: np.ndarray, padding: float = 6.0
) -> np.ndarray:
    if len(receptor_coords) == 0 or len(native_coords) == 0:
        return receptor_coords
    lower = native_coords.min(axis=0) - padding
    upper = native_coords.max(axis=0) + padding
    mask = np.all((receptor_coords >= lower) & (receptor_coords <= upper), axis=1)
    focused = receptor_coords[mask]
    return focused if len(focused) > 0 else receptor_coords


def _plot_pose_trace(ax: plt.Axes, coords: np.ndarray, color: str, label: str) -> None:
    if len(coords) == 0:
        return
    ax.plot(coords[:, 0], coords[:, 1], color=color, linewidth=1.2, alpha=0.8)
    ax.scatter(coords[:, 0], coords[:, 1], s=24, color=color, label=label, zorder=3)


def _render_projection_panel(
    ax: plt.Axes,
    receptor_xy: np.ndarray,
    native_xy: np.ndarray,
    method_xy: dict[str, np.ndarray],
    axis_labels: tuple[str, str],
    title: str,
) -> None:
    if len(receptor_xy) > 0:
        ax.scatter(
            receptor_xy[:, 0],
            receptor_xy[:, 1],
            s=6,
            color="#c7ced6",
            alpha=0.45,
            label="receptor pocket",
            zorder=1,
        )
    _plot_pose_trace(ax, native_xy, "#0f172a", "native")
    palette = {
        "DQ-Dock": "#d97706",
        "GNINA": "#059669",
    }
    for label, coords in method_xy.items():
        _plot_pose_trace(ax, coords, palette.get(label, "#2563eb"), label)
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")


def generate_pose_comparison_figures(
    csv_path: Path,
    *,
    output_dir: Path | None = None,
    pdb_ids: set[str] | None = None,
    warning_sink: Callable[[str], None] | None = None,
) -> dict[str, Path]:
    del warning_sink
    if not csv_path.exists():
        return {}

    rows_by_pdb: dict[str, list[dict[str, str]]] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            pdb_id = row.get("pdb_id")
            if not pdb_id:
                continue
            if pdb_ids is not None and pdb_id not in pdb_ids:
                continue
            rows_by_pdb.setdefault(pdb_id, []).append(row)

    if output_dir is None:
        output_dir = csv_path.with_suffix("").with_name(csv_path.stem + "_pose_figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, Path] = {}
    for pdb_id, rows in rows_by_pdb.items():
        dq_row = next((r for r in rows if r.get("engine_id") == "dq_dock"), None)
        if dq_row is None:
            continue
        receptor_path_text = dq_row.get("receptor_pdb")
        native_path_text = dq_row.get("native_ligand_pdb")
        if not receptor_path_text or not native_path_text:
            continue
        receptor_path = Path(receptor_path_text)
        native_path = Path(native_path_text)
        if not receptor_path.exists() or not native_path.exists():
            continue

        receptor_coords = _read_structure_coords(receptor_path)
        native_coords = _read_structure_coords(native_path)
        if len(native_coords) == 0:
            continue
        receptor_focus = _focus_receptor_coords(receptor_coords, native_coords)

        method_coords: dict[str, np.ndarray] = {}
        method_titles: list[str] = []
        for row in rows:
            if row.get("status") != "success":
                continue
            pose_path_text = row.get("pose_pdb")
            if not pose_path_text:
                continue
            pose_path = Path(pose_path_text)
            if not pose_path.exists():
                continue
            method_name = row.get("method") or row.get("engine_id") or "method"
            coords = _read_structure_coords(pose_path)
            if len(coords) == 0:
                continue
            method_coords[method_name] = coords
            rmsd_text = row.get("top_rmsd") or "n/a"
            method_titles.append(f"{method_name}: RMSD {rmsd_text}")

        if not method_coords:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))
        projections = [
            ((0, 1), ("X (A)", "Y (A)"), "XY"),
            ((0, 2), ("X (A)", "Z (A)"), "XZ"),
            ((1, 2), ("Y (A)", "Z (A)"), "YZ"),
        ]
        for ax, (dims, axis_labels, panel_title) in zip(axes, projections):
            receptor_xy = receptor_focus[:, dims]
            native_xy = native_coords[:, dims]
            method_xy = {
                label: coords[:, dims] for label, coords in method_coords.items()
            }
            _render_projection_panel(
                ax,
                receptor_xy,
                native_xy,
                method_xy,
                axis_labels,
                panel_title,
            )

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)))
        target_name = dq_row.get("target_name") or pdb_id
        fig.suptitle(
            f"{pdb_id} - {target_name}\n" + " | ".join(method_titles),
            fontsize=11,
            y=1.04,
        )
        fig.tight_layout()
        output_path = output_dir / f"{pdb_id}_pose_compare.png"
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        outputs[pdb_id] = output_path

    return outputs


def _pymol_safe_name(label: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in label).strip("_") or "pose"


def _flatten_png_background(png_path: Path) -> None:
    if not png_path.exists():
        return
    image = Image.open(png_path).convert("RGBA")
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    flattened = Image.alpha_composite(background, image).convert("RGB")
    flattened.save(png_path)


def _compute_pocket_view_direction(
    receptor_path: Path, native_path: Path
) -> np.ndarray:
    receptor_coords = _read_structure_coords(receptor_path)
    native_coords = _read_structure_coords(native_path)
    if len(receptor_coords) == 0 or len(native_coords) == 0:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    receptor_center = receptor_coords.mean(axis=0)
    pocket_center = native_coords.mean(axis=0)
    direction = pocket_center - receptor_center
    norm = float(np.linalg.norm(direction))
    if norm < 1e-8:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return direction / norm


def _group_csv_rows_by_pdb(
    csv_path: Path, pdb_ids: set[str] | None = None
) -> dict[str, list[dict[str, str]]]:
    rows_by_pdb: dict[str, list[dict[str, str]]] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            pdb_id = row.get("pdb_id")
            if not pdb_id:
                continue
            if pdb_ids is not None and pdb_id not in pdb_ids:
                continue
            rows_by_pdb.setdefault(pdb_id, []).append(row)
    return rows_by_pdb


def _build_pymol_methods(rows: list[dict[str, str]]) -> tuple[PyMOLPoseMethod, ...]:
    methods: list[PyMOLPoseMethod] = []
    for row in rows:
        if row.get("status") != "success":
            continue
        pose_path_text = row.get("pose_pdb")
        if not pose_path_text:
            continue
        pose_path = Path(pose_path_text)
        if not pose_path.exists():
            continue
        label = row.get("method") or row.get("engine_id") or "method"
        color_name, color_rgb = PYMOL_METHOD_STYLES.get(label, ("cyan", (37, 99, 235)))
        methods.append(
            PyMOLPoseMethod(
                label=label,
                object_name=_pymol_safe_name(label.lower()),
                pose_path=pose_path,
                color_name=color_name,
                color_rgb=color_rgb,
                rmsd_text=row.get("top_rmsd") or "n/a",
            )
        )
    return tuple(methods)


def _build_pymol_scene(
    *,
    pdb_id: str,
    rows: list[dict[str, str]],
    output_dir: Path,
) -> PyMOLPoseScene | None:
    dq_row = next((r for r in rows if r.get("engine_id") == "dq_dock"), None)
    if dq_row is None:
        return None
    receptor_path_text = dq_row.get("receptor_pdb")
    native_path_text = dq_row.get("native_ligand_pdb")
    if not receptor_path_text or not native_path_text:
        return None
    receptor_path = Path(receptor_path_text)
    native_path = Path(native_path_text)
    if not receptor_path.exists() or not native_path.exists():
        return None
    methods = _build_pymol_methods(rows)
    if not methods:
        return None
    return PyMOLPoseScene(
        pdb_id=pdb_id,
        target_name=dq_row.get("target_name") or pdb_id,
        receptor_path=receptor_path,
        native_path=native_path,
        methods=methods,
        script_path=output_dir / f"{pdb_id}_pose_compare.pml",
        front_png_path=output_dir / f"{pdb_id}_pose_compare_3d.png",
        side_png_path=output_dir / f"{pdb_id}_pose_compare_3d_side.png",
        pse_path=output_dir / f"{pdb_id}_pose_compare.pse",
    )


def _font() -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 24)
    except OSError:
        return ImageFont.load_default()


def _annotate_pose_render(
    image_path: Path,
    *,
    scene: PyMOLPoseScene,
    view_label: str,
) -> None:
    if not image_path.exists():
        return
    image = Image.open(image_path).convert("RGB")
    title_font = _font()
    body_font = _font()
    banner_height = 72 + 30 * max(1, len(scene.methods))
    canvas = Image.new(
        "RGB", (image.width, image.height + banner_height), (255, 255, 255)
    )
    canvas.paste(image, (0, banner_height))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (28, 18),
        f"{scene.pdb_id} - {scene.target_name} - {view_label}",
        fill=(20, 20, 20),
        font=title_font,
    )
    draw.text(
        (28, 48),
        "black = native reference; colored = predicted poses",
        fill=(90, 90, 90),
        font=body_font,
    )
    legend_y = 82
    for method in scene.methods:
        draw.rectangle(
            (28, legend_y + 4, 50, legend_y + 26),
            fill=method.color_rgb,
            outline=(80, 80, 80),
        )
        draw.text(
            (62, legend_y),
            f"{method.label} RMSD {method.rmsd_text} A",
            fill=(20, 20, 20),
            font=body_font,
        )
        legend_y += 30
    canvas.save(image_path)


def _write_pymol_pose_script(
    scene: PyMOLPoseScene,
) -> None:
    receptor_abs = scene.receptor_path.resolve().as_posix()
    native_abs = scene.native_path.resolve().as_posix()
    front_png_abs = scene.front_png_path.resolve().as_posix()
    side_png_abs = scene.side_png_path.resolve().as_posix()
    pse_abs = scene.pse_path.resolve().as_posix()
    view_dir = _compute_pocket_view_direction(scene.receptor_path, scene.native_path)
    view_dir_list = ", ".join(f"{value:.8f}" for value in view_dir.tolist())
    pose_lines: list[str] = []
    pose_names = [method.object_name for method in scene.methods]
    for method in scene.methods:
        pose_lines.extend(
            [
                f"load {method.pose_path.resolve().as_posix()}, {method.object_name}",
                f"show sticks, {method.object_name}",
                f"color {method.color_name}, {method.object_name}",
            ]
        )

    if not pose_names:
        return

    focus_selection = (
        f"(pocket_wall or native_{scene.pdb_id} or {' or '.join(pose_names)})"
    )

    script = [
        "reinitialize",
        f"load {receptor_abs}, receptor",
        f"load {native_abs}, native",
        "hide everything, all",
        "remove hydro",
        "select pocket_wall, byres (receptor within 6.0 of native)",
        "show sticks, native",
        "color black, native",
        *pose_lines,
        "show surface, pocket_wall",
        "set transparency, 0.14, pocket_wall",
        "set surface_color, lightblue, pocket_wall",
        "show sticks, byres ((receptor within 4.0 of native) and sidechain)",
        "color gray60, byres ((receptor within 4.0 of native) and sidechain)",
        "set surface_quality, 1",
        "set stick_radius, 0.22",
        "set ray_opaque_background, on",
        "set orthoscopic, on",
        "set depth_cue, off",
        "set ray_shadows, off",
        "set two_sided_lighting, on",
        "set antialias, 2",
        "bg_color white",
        f"set_name native, native_{scene.pdb_id}",
        f"group poses, {' '.join(pose_names)}",
        f"orient {focus_selection}",
        f"zoom {focus_selection}, 2",
        "python",
        "from pymol import cmd",
        "import numpy as np",
        f"view_dir = np.array([{view_dir_list}], dtype=float)",
        "view = np.array(cmd.get_view()[:9], dtype=float).reshape(3, 3)",
        "view_dir_camera = view.T @ view_dir",
        "yaw = -np.degrees(np.arctan2(view_dir_camera[0], view_dir_camera[2]))",
        "cmd.turn('y', float(yaw))",
        "view = np.array(cmd.get_view()[:9], dtype=float).reshape(3, 3)",
        "view_dir_camera = view.T @ view_dir",
        "pitch = np.degrees(np.arctan2(view_dir_camera[1], view_dir_camera[2]))",
        "cmd.turn('x', float(pitch))",
        "cmd.turn('z', 15.0)",
        f"cmd.zoom('{focus_selection}', 2.0)",
        "python end",
        "viewport 1600, 1200",
        f"png {front_png_abs}, width=1600, height=1200, dpi=200, ray=1",
        "turn y, 90",
        f"zoom {focus_selection}, 2",
        f"png {side_png_abs}, width=1600, height=1200, dpi=200, ray=1",
        f"save {pse_abs}",
        "quit",
    ]
    scene.script_path.write_text("\n".join(script) + "\n")


def generate_pymol_pose_comparisons(
    csv_path: Path,
    *,
    output_dir: Path | None = None,
    pdb_ids: set[str] | None = None,
    render_png: bool = True,
    warning_sink: Callable[[str], None] | None = None,
) -> dict[str, Path]:
    if not csv_path.exists():
        if warning_sink is not None:
            warning_sink(f"PyMOL render skipped because CSV is missing: {csv_path}")
        return {}
    pymol_bin = shutil.which("pymol")
    if pymol_bin is None:
        if warning_sink is not None:
            warning_sink("PyMOL render skipped because `pymol` is not on PATH")
        return {}
    rows_by_pdb = _group_csv_rows_by_pdb(csv_path, pdb_ids)

    if output_dir is None:
        output_dir = csv_path.with_suffix("").with_name(csv_path.stem + "_pymol")
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, Path] = {}
    for pdb_id, rows in rows_by_pdb.items():
        scene = _build_pymol_scene(pdb_id=pdb_id, rows=rows, output_dir=output_dir)
        if scene is None:
            continue
        _write_pymol_pose_script(scene)
        if not scene.script_path.exists():
            if warning_sink is not None:
                warning_sink(
                    f"PyMOL render skipped for {pdb_id}: script was not written to {scene.script_path}"
                )
            continue

        if render_png:
            try:
                subprocess.run(
                    [pymol_bin, "-cq", str(scene.script_path)],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
            except subprocess.TimeoutExpired:
                if warning_sink is not None:
                    warning_sink(f"PyMOL render timed out for {pdb_id} after 120s")
                continue
            except subprocess.CalledProcessError as exc:
                if warning_sink is not None:
                    stderr = exc.stderr.strip() if exc.stderr is not None else ""
                    suffix = f": {stderr}" if stderr else ""
                    warning_sink(f"PyMOL render failed for {pdb_id}{suffix}")
                continue
            _flatten_png_background(scene.front_png_path)
            _flatten_png_background(scene.side_png_path)
            _annotate_pose_render(
                scene.front_png_path, scene=scene, view_label="Pocket View"
            )
            _annotate_pose_render(
                scene.side_png_path, scene=scene, view_label="Side View"
            )
        outputs[pdb_id] = (
            scene.front_png_path if scene.front_png_path.exists() else scene.script_path
        )

    return outputs


def _write_markdown_report(payload: dict, report_path: Path) -> None:
    summary = payload["summary"]
    dq_rows = payload["dq_dock"]
    competitor_rows = payload["competitors"]
    excluded_rows = payload["excluded"]
    successful_dq_rows = [row for row in dq_rows if row.get("success", True)]

    lines = [
        "# PDB Redocking Benchmark",
        "",
        f"- Phase: `{summary['phase']}`",
        f"- Charge method: `{summary['charge_method']}`",
        f"- Requested complexes: `{summary['n_complexes_requested']}`",
        f"- Completed DQ-Dock complexes: `{summary['n_complexes_run']}`",
        f"- Successful DQ-Dock complexes: `{len(successful_dq_rows)}`",
        f"- Excluded complexes: `{summary['n_complexes_excluded']}`",
        f"- Pocket-guided sampling: `{summary['use_pocket_guided']}`",
        f"- Competitors: `{', '.join(item['display_name'] for item in summary['competitors'])}`",
        "",
        "## Summary",
        "",
        f"- DQ-Dock avg RMSD: `{summary['dq_avg_rmsd']}`",
        f"- DQ-Dock total time (s): `{summary['dq_total_time_s']}`",
        f"- Total benchmark time (s): `{summary['total_benchmark_time_s']}`",
        "",
    ]

    # Diversity summary — derived entirely from the per-row classification fields
    classified_rows = [r for r in dq_rows if r.get("target_class") is not None]
    if classified_rows:
        from collections import Counter

        target_counts: Counter[str] = Counter(
            r["target_class"] for r in classified_rows
        )
        mode_counts: Counter[str] = Counter(r["binding_mode"] for r in classified_rows)
        lines.extend(
            [
                "## Benchmark Diversity",
                "",
                "### By Target Class",
                "",
                "| Target Class | Count |",
                "| --- | ---: |",
            ]
        )
        for tc, n in sorted(target_counts.items(), key=lambda x: -x[1]):
            lines.append(f"| {tc} | {n} |")
        lines.extend(
            [
                "",
                "### By Binding Mode",
                "",
                "| Binding Mode | Count |",
                "| --- | ---: |",
            ]
        )
        for bm, n in sorted(mode_counts.items(), key=lambda x: -x[1]):
            lines.append(f"| {bm} | {n} |")
        lines.append("")

    lines.extend(
        [
            "## DQ-Dock",
            "",
            "| PDB | Target | Class | Mode | Status | RMSD | Time (s) | Energy | Gap Proof | Returned Contract | Native Rank | Energy Gap | Error |",
            "| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | --- |",
        ]
    )

    for row in dq_rows:
        lines.append(
            f"| {row['pdb_id']} | {row['target_name']} | {_fmt_text(row.get('target_class'))} | {_fmt_text(row.get('binding_mode'))} | {'success' if row.get('success', True) else 'failure'} | {_fmt_number(row.get('rmsd'))} | {_fmt_number(row['time_s'])} | {_fmt_number(row.get('energy'))} | {_fmt_text(row.get('gap_proof'))} | {_fmt_text(row.get('returned_pose_contract_summary'))} | {_fmt_text(row.get('native_rank'))} | {_fmt_number(row.get('energy_gap'))} | {_fmt_text(row.get('error'))} |"
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
                f"| {row['pdb_id']} | {row['target_name']} | {'success' if row['success'] else 'failure'} | {_fmt_text(top_rmsd)} | {_fmt_text(best_rmsd)} | {_fmt_number(row['time_s'])} | {_fmt_text(score)} | {_fmt_text(row['error'])} |"
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

    if "success" in dq_df.columns:
        dq_df = dq_df[dq_df["success"]].copy()
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
    _barplot(data=cast(pd.DataFrame, plot_df), x="label", y="RMSD", hue="Method")
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
    _barplot(data=cast(pd.DataFrame, plot_df), x="label", y="Time", hue="Method")
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

    if "success" in dq_df.columns:
        dq_df = dq_df[dq_df["success"]].copy()
    if dq_df.empty:
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
    _scatterplot(
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


def render_redocking_report(
    json_path: Path,
    *,
    report_pdb_ids: set[str] | None = None,
    warning_sink: Callable[[str], None] | None = None,
) -> dict[str, Path]:
    payload = _load_payload(json_path)
    base = json_path.with_suffix("")
    csv_path = json_path.with_suffix(".csv")
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

    outputs = {
        "markdown": markdown_path,
        "rmsd": rmsd_path,
        "timing": timing_path,
        "timing_log": timing_log_path,
        "scatter": scatter_path,
    }

    pose_outputs = generate_pose_comparison_figures(
        csv_path,
        pdb_ids=report_pdb_ids,
        warning_sink=warning_sink,
    )
    if pose_outputs:
        outputs["pose_figures_dir"] = next(iter(pose_outputs.values())).parent
    pymol_outputs = generate_pymol_pose_comparisons(
        csv_path,
        pdb_ids=report_pdb_ids,
        warning_sink=warning_sink,
    )
    if pymol_outputs:
        outputs["pymol_figures_dir"] = next(iter(pymol_outputs.values())).parent

    return outputs
