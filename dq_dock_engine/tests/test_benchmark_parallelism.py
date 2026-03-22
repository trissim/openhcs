from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
import tempfile
import jax.numpy as jnp

from dq_dock_engine.benchmark import benchmark_pdb
from dq_dock_engine.benchmark.redocking_report import (
    generate_pose_comparison_figures,
    render_redocking_report,
)
from dq_dock_engine.docking.charges import ChargeAssigner, ChargeMethod


def test_benchmark_parallelism_rejects_non_positive_workers() -> None:
    try:
        benchmark_pdb.BenchmarkParallelism(max_workers=0)
    except ValueError as exc:
        assert "max_workers must be positive" in str(exc)
        return
    raise AssertionError("expected BenchmarkParallelism to reject max_workers=0")


def test_execute_benchmark_jobs_runs_sequentially_when_single_worker() -> None:
    seen: list[int] = []

    def run_job(value: int) -> int:
        seen.append(value)
        return value * 2

    results = list(
        benchmark_pdb.execute_benchmark_jobs(
            (1, 2, 3),
            run_job=run_job,
            parallelism=benchmark_pdb.BenchmarkParallelism(max_workers=1),
        )
    )

    assert results == [2, 4, 6]
    assert seen == [1, 2, 3]


def test_execute_benchmark_jobs_uses_process_executor_for_multi_worker(
    monkeypatch,
) -> None:
    submitted: list[int] = []

    @dataclass(frozen=True)
    class FakeFuture:
        value: int

        def result(self) -> int:
            return self.value * 10

    class FakeExecutor:
        def __init__(self, *, max_workers: int, mp_context):
            self.max_workers = max_workers
            self.mp_context = mp_context
            self.shutdown_calls: list[tuple[bool, bool | None]] = []

        def __enter__(self) -> "FakeExecutor":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def submit(self, run_job, job: int) -> FakeFuture:
            submitted.append(job)
            return FakeFuture(job)

        def shutdown(self, *, wait: bool, cancel_futures: bool | None = None) -> None:
            self.shutdown_calls.append((wait, cancel_futures))

    monkeypatch.setattr(benchmark_pdb, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(benchmark_pdb, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(
        benchmark_pdb.mp,
        "get_context",
        lambda method: f"ctx:{method}",
    )

    results = list(
        benchmark_pdb.execute_benchmark_jobs(
            (4, 5),
            run_job=lambda value: value,
            parallelism=benchmark_pdb.BenchmarkParallelism(max_workers=2),
        )
    )

    assert submitted == [4, 5]
    assert results == [40, 50]


def test_render_redocking_report_tolerates_missing_gap_metrics() -> None:
    payload = {
        "summary": {
            "phase": "dq_dock",
            "charge_method": "gasteiger",
            "n_complexes_requested": 1,
            "n_complexes_run": 1,
            "n_complexes_excluded": 0,
            "n_poses": 10,
            "n_opt_steps": 5,
            "use_pocket_guided": True,
            "competitors": [],
            "dq_avg_rmsd": 1.23,
            "dq_total_time_s": 2.34,
            "total_benchmark_time_s": 3.45,
            "competitor_stats": {},
        },
        "dq_dock": [
            {
                "pdb_id": "1m0n",
                "target_name": "test target",
                "rmsd": 10.18,
                "time_s": 16.09,
                "energy": None,
                "gap_proof": "inconclusive",
                "native_rank": 1,
                "energy_gap": None,
            }
        ],
        "competitors": [],
        "excluded": [],
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "report.json"
        json_path.write_text(json.dumps(payload))
        outputs = render_redocking_report(json_path)
        markdown = outputs["markdown"].read_text()

    assert "1m0n" in markdown
    assert "n/a" in markdown


def test_prepare_benchmark_electrostatics_surfaces_charge_failures() -> None:
    class FailingChargeAssigner(ChargeAssigner):
        @property
        def method(self) -> ChargeMethod:
            return ChargeMethod.GASTEIGER

        def assign(self, source):
            raise RuntimeError(
                "RDKit produced non-finite Gasteiger charge at atom index 0"
            )

    try:
        benchmark_pdb.prepare_benchmark_electrostatics(
            assigner=FailingChargeAssigner(),
            pdb_id="1m0n",
            ligand_pdb=Path("ligand.pdb"),
            ligand_elements=("C",),
            ligand_atom_count=1,
            pocket_receptor_pdb=Path("pocket.pdb"),
            pocket_elements=("N",),
            pocket_atom_count=1,
        )
    except RuntimeError as exc:
        assert "non-finite Gasteiger charge" in str(exc)
        return
    raise AssertionError("expected raw charge preparation failure to bubble up")


def test_generate_pose_comparison_figures_writes_png() -> None:
    def pdb_line(
        record: str,
        serial: int,
        atom_name: str,
        resname: str,
        chain: str,
        resseq: int,
        x: float,
        y: float,
        z: float,
        element: str,
    ) -> str:
        return (
            f"{record:<6}{serial:>5} {atom_name:^4} {resname:>3} {chain:1}{resseq:>4}    "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{1.00:>6.2f}{20.00:>6.2f}          {element:>2}\n"
        )

    def write_pdb(
        path: Path, coords: list[tuple[float, float, float]], prefix: str
    ) -> None:
        lines = []
        for idx, (x, y, z) in enumerate(coords, start=1):
            lines.append(
                pdb_line("HETATM", idx, f"C{idx}", "LIG", "A", 1, x, y, z, "C")
            )
        path.write_text("".join(lines))

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        receptor = tmp / "receptor.pdb"
        native = tmp / "native.pdb"
        dq_pose = tmp / "dq_pose.pdb"
        gnina_pose = tmp / "gnina_pose.pdb"
        csv_path = tmp / "results.csv"

        receptor.write_text(
            "".join(
                [
                    pdb_line("ATOM", 1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C"),
                    pdb_line("ATOM", 2, "CA", "ALA", "A", 2, 2.0, 1.5, 0.5, "C"),
                    pdb_line("ATOM", 3, "CA", "ALA", "A", 3, -1.5, 2.0, 1.0, "C"),
                ]
            )
        )
        write_pdb(native, [(0.0, 0.0, 0.0), (1.2, 0.4, 0.3), (1.8, 1.1, 0.7)], "N")
        write_pdb(dq_pose, [(0.2, 0.1, 0.0), (1.3, 0.6, 0.4), (1.9, 1.3, 0.8)], "D")
        write_pdb(
            gnina_pose, [(-0.1, -0.2, 0.1), (1.0, 0.3, 0.2), (1.7, 1.0, 0.6)], "G"
        )

        fieldnames = [
            "method",
            "engine_id",
            "pdb_id",
            "target_name",
            "center_x",
            "center_y",
            "center_z",
            "box_x",
            "box_y",
            "box_z",
            "ligand_atoms",
            "pocket_atoms",
            "score_name",
            "score",
            "top_rmsd",
            "best_mode_rmsd",
            "time_s",
            "pose_pdb",
            "gap_proof",
            "native_rank",
            "energy_gap",
            "receptor_pdb",
            "native_ligand_pdb",
            "status",
            "error",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(
                {
                    "method": "DQ-Dock",
                    "engine_id": "dq_dock",
                    "pdb_id": "1abc",
                    "target_name": "example target",
                    "center_x": 0,
                    "center_y": 0,
                    "center_z": 0,
                    "box_x": 12,
                    "box_y": 12,
                    "box_z": 12,
                    "ligand_atoms": 3,
                    "pocket_atoms": 10,
                    "score_name": "energy",
                    "score": -1.0,
                    "top_rmsd": 0.4,
                    "best_mode_rmsd": 0.4,
                    "time_s": 1.2,
                    "pose_pdb": str(dq_pose),
                    "gap_proof": "proved",
                    "native_rank": 1,
                    "energy_gap": 0.1,
                    "receptor_pdb": str(receptor),
                    "native_ligand_pdb": str(native),
                    "status": "success",
                    "error": "",
                }
            )
            writer.writerow(
                {
                    "method": "GNINA",
                    "engine_id": "gnina",
                    "pdb_id": "1abc",
                    "target_name": "example target",
                    "center_x": 0,
                    "center_y": 0,
                    "center_z": 0,
                    "box_x": 12,
                    "box_y": 12,
                    "box_z": 12,
                    "score_name": "cnn_affinity",
                    "score": -2.0,
                    "top_rmsd": 0.8,
                    "best_mode_rmsd": 0.8,
                    "time_s": 2.4,
                    "pose_pdb": str(gnina_pose),
                    "receptor_pdb": str(receptor),
                    "native_ligand_pdb": str(native),
                    "status": "success",
                    "error": "",
                }
            )

        outputs = generate_pose_comparison_figures(csv_path)

        assert "1abc" in outputs
        assert outputs["1abc"].exists()


def test_load_explicit_pdb_ids_combines_cli_and_file_sources(tmp_path) -> None:
    pdb_file = tmp_path / "pdb_ids.txt"
    pdb_file.write_text("# comment\n3ghi 4JKL\n")

    pdb_ids = benchmark_pdb.load_explicit_pdb_ids("1ABC, 2Def", pdb_file)

    assert pdb_ids == ("1abc", "2def", "3ghi", "4jkl")


def test_load_explicit_pdb_ids_rejects_duplicates(tmp_path) -> None:
    pdb_file = tmp_path / "pdb_ids.txt"
    pdb_file.write_text("2def\n")

    try:
        benchmark_pdb.load_explicit_pdb_ids("1abc, 2def", pdb_file)
    except ValueError as exc:
        assert "Duplicate explicit PDB IDs are not allowed" in str(exc)
        assert "2def" in str(exc)
        return
    raise AssertionError("expected duplicate explicit PDB IDs to fail loudly")


def test_derive_benchmark_target_selection_prefers_explicit_ids(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)

    selection = benchmark_pdb.derive_benchmark_target_selection(
        dataset="casf2007",
        n_complexes=99,
        explicit_pdb_ids=("1abc", "2def"),
    )

    assert selection.mode == benchmark_pdb.BenchmarkTargetSelectionMode.EXPLICIT_PDB_IDS
    assert selection.label == "explicit PDB IDs"
    assert selection.cache_dir == Path("pdb_cache")
    assert selection.n_complexes_requested == 2
    assert [entry.pdb_id for entry in selection.entries] == ["1abc", "2def"]
    assert [entry.target_name for entry in selection.entries] == ["1abc", "2def"]


def test_run_benchmark_explicit_pdb_ids_override_dataset_and_use_qc(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    seen_screened_entries: list[tuple[str, str]] = []
    save_calls: list[tuple[str, int, list[str]]] = []

    def fail_dataset_path(*args, **kwargs):
        raise AssertionError("dataset selection path should not be used")

    def fake_download_pdb(pdb_id: str, cache_dir: Path) -> Path:
        cache_dir.mkdir(exist_ok=True)
        pdb_path = cache_dir / f"{pdb_id}.pdb"
        pdb_path.write_text(f"TITLE     Example target {pdb_id.upper()}\n")
        return pdb_path

    def fake_screen_complex(entry, pdb_path: Path, protein_path: Path):
        seen_screened_entries.append((entry.pdb_id, entry.target_name))
        return benchmark_pdb.ScreeningDecision(
            accepted=False,
            reasons=("qc failed",),
        )

    def fake_save(self, phase, context, excluded_rows, *, render_report=False):
        save_calls.append(
            (
                phase,
                context.n_complexes_requested,
                [row.pdb_id for row in excluded_rows],
            )
        )
        return (tmp_path / "results.json", tmp_path / "results.csv")

    monkeypatch.setattr(benchmark_pdb, "get_casf_2007_ids", fail_dataset_path)
    monkeypatch.setattr(benchmark_pdb, "download_pdb", fake_download_pdb)
    monkeypatch.setattr(
        benchmark_pdb,
        "prepare_protein",
        lambda pdb_path: pdb_path.with_name(f"{pdb_path.stem}_protein.pdb"),
    )
    monkeypatch.setattr(benchmark_pdb, "screen_complex", fake_screen_complex)
    monkeypatch.setattr(benchmark_pdb, "create_charge_assigner", lambda method: None)
    monkeypatch.setattr(benchmark_pdb.BenchmarkState, "save", fake_save)

    benchmark_pdb.run_benchmark(
        n_complexes=99,
        dataset="casf2007",
        pdb_ids=("1abc", "2def"),
        competitors="none",
        results_dir=tmp_path / "benchmark_results",
    )

    assert seen_screened_entries == [
        ("1abc", "Example target 1ABC"),
        ("2def", "Example target 2DEF"),
    ]
    assert save_calls == [("curation", 2, ["1abc", "2def"])]
