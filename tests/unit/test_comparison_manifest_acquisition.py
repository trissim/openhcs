from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmark.cellprofiler_comparison import load_comparison_cases
from benchmark.contracts.comparison_manifest import ComparisonManifest
from benchmark.contracts.manifest_acquisition import GitSparseRootAcquisitionStrategy


def test_manifest_git_sparse_root_materializes_missing_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENHCS_BENCHMARK_AUTO_ACQUIRE", "1")
    root = tmp_path / "examples"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "path_roots": {
                    "examples": {
                        "default": str(root),
                        "acquisition": {
                            "kind": "git_sparse",
                            "git_url": "https://example.invalid/examples.git",
                            "git_ref": "abc123",
                        },
                    }
                },
                "cases": [
                    {
                        "name": "ExampleFly",
                        "dataset_path_root": "examples",
                        "dataset_path": "ExampleFly/images",
                        "cppipe_path_root": "examples",
                        "cppipe_path": "ExampleFly/ExampleFly.cppipe",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    git_commands: list[tuple[tuple[str, ...], Path | None]] = []

    def fake_run_git(args: list[str], cwd: Path | None) -> None:
        git_commands.append((tuple(args), cwd))
        if args[0] == "clone":
            root.mkdir(parents=True)
            (root / ".git").mkdir()
        if args[:2] == ["sparse-checkout", "set"]:
            (root / "ExampleFly" / "images").mkdir(parents=True)
            (root / "ExampleFly" / "ExampleFly.cppipe").write_text(
                "CellProfiler Pipeline: http://www.cellprofiler.org\n",
                encoding="utf-8",
            )

    monkeypatch.setattr(
        GitSparseRootAcquisitionStrategy,
        "_run_git",
        staticmethod(fake_run_git),
    )

    cases = load_comparison_cases(manifest_path)

    assert cases[0].dataset_path == root / "ExampleFly/images"
    assert cases[0].cppipe_path == root / "ExampleFly/ExampleFly.cppipe"
    assert ("clone", "--depth", "1", "--filter=blob:none", "--sparse") == git_commands[
        0
    ][0][:5]
    assert git_commands[1][0] == ("fetch", "--depth", "1", "origin", "abc123")
    assert git_commands[2][0] == (
        "sparse-checkout",
        "set",
        "ExampleFly",
    )
    assert git_commands[3][0] == ("checkout", "FETCH_HEAD")


def test_manifest_dataset_registry_root_materializes_missing_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENHCS_BENCHMARK_AUTO_ACQUIRE", "1")
    root = tmp_path / "cache"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "path_roots": {
                    "dataset_cache": {
                        "default": str(root),
                        "acquisition": {
                            "kind": "dataset_registry",
                            "dataset_ids": ["CellProfiler4_benchmark_supplement"],
                        },
                    }
                },
                "cases": [
                    {
                        "name": "combine",
                        "dataset_id": "CellProfiler4_benchmark_supplement",
                        "dataset_path_root": "dataset_cache",
                        "dataset_path": (
                            "CellProfiler4_benchmark_supplement/data/CombineObjects"
                        ),
                        "cppipe_path_root": "dataset_cache",
                        "cppipe_path": (
                            "CellProfiler4_benchmark_supplement/data/CombineObjects/"
                            "CombineObjectsDemo.cppipe"
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    acquired: list[str] = []

    def fake_acquire_dataset(spec: Any, *, cache_base: Path | None = None) -> object:
        acquired.append(spec.id)
        assert cache_base == root
        case_root = (
            root / "CellProfiler4_benchmark_supplement" / "data" / "CombineObjects"
        )
        case_root.mkdir(parents=True)
        (case_root / "CombineObjectsDemo.cppipe").write_text(
            "CellProfiler Pipeline: http://www.cellprofiler.org\n",
            encoding="utf-8",
        )
        return object()

    monkeypatch.setattr(
        "benchmark.datasets.acquire.acquire_dataset",
        fake_acquire_dataset,
    )

    cases = load_comparison_cases(manifest_path)

    assert acquired == ["CellProfiler4_benchmark_supplement"]
    assert cases[0].dataset_path == (
        root / "CellProfiler4_benchmark_supplement/data/CombineObjects"
    )
    assert cases[0].cppipe_path == (
        root
        / "CellProfiler4_benchmark_supplement/data/CombineObjects/"
        "CombineObjectsDemo.cppipe"
    )


def test_manifest_root_materialization_can_be_disabled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "examples"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "path_roots": {
                    "examples": {
                        "default": str(root),
                        "acquisition": {
                            "kind": "git_sparse",
                            "git_url": "https://example.invalid/examples.git",
                        },
                    }
                },
                "cases": [
                    {
                        "name": "ExampleFly",
                        "dataset_path_root": "examples",
                        "dataset_path": "ExampleFly/images",
                        "cppipe_path_root": "examples",
                        "cppipe_path": "ExampleFly/ExampleFly.cppipe",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("OPENHCS_BENCHMARK_AUTO_ACQUIRE", "0")
    manifest = ComparisonManifest.load(manifest_path)

    assert manifest.path_resolver.resolve(
        manifest.payload["cases"][0],
        "cppipe_path",
    ) == root / "ExampleFly/ExampleFly.cppipe"
