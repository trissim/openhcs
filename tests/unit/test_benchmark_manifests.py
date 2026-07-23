from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.cellprofiler_comparison import load_comparison_cases
from benchmark.datasets.registry import DATASET_REGISTRY


def test_official30_portable_manifest_declares_roots_without_absolute_cases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENHCS_BENCHMARK_AUTO_ACQUIRE", "0")
    manifest_path = Path("benchmark/manifests/official30_portable_axis1.json")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert len(payload["cases"]) == 30
    assert set(payload["path_roots"]) == {
        "axis_one_subsets",
        "cellprofiler_examples",
        "dataset_cache",
    }
    assert payload["path_roots"]["cellprofiler_examples"]["acquisition"] == {
        "git_ref": "4972b59e670a4ae96c3d453803c92eeff378d054",
        "git_url": "https://github.com/CellProfiler/examples.git",
        "kind": "git_sparse",
        "sparse_paths": [
            "CellProfiler3Pipelines",
            "ExampleColocalization",
            "ExampleCometAssay",
            "ExampleFly",
            "ExampleHuman",
            "ExampleIlluminationCorrection",
            "ExampleImagingFlowCytometryObjectsInGrid",
            "ExampleNeighbors",
            "ExamplePercentPositive",
            "ExampleSpeckles",
            "ExampleStraightenWorms",
            "ExampleTrackObjects",
            "ExampleTumor",
            "ExampleUntangleWorms",
            "ExampleUntangleWormsBrightField",
            "ExampleVitraImages",
            "ExampleWoundHealing",
            "ExampleYeastColonies",
            "ExampleYeastPatches",
        ],
    }
    assert payload["path_roots"]["dataset_cache"]["acquisition"] == {
        "dataset_ids": [
            "CellProfiler_tutorials",
            "CellProfiler4_benchmark_supplement",
        ],
        "kind": "dataset_registry",
    }
    assert all(
        not Path(case["dataset_path"]).is_absolute() for case in payload["cases"]
    )
    assert all(not Path(case["cppipe_path"]).is_absolute() for case in payload["cases"])
    cases = load_comparison_cases(manifest_path)

    assert len(cases) == 30
    assert cases[0].name == "ExampleColocalization"
    assert cases[-1].name == "cp_tutorial_translocation_start"


def test_official30_dataset_registry_pins_all_git_sources() -> None:
    expected_revisions = {
        "CellProfiler_tutorials": "264a8155da21a2d468051f78211bed2e580a8934",
        "CellProfiler4_benchmark_supplement": (
            "40abc2e600fd46b74c213999dd25c5245048dc92"
        ),
    }

    for dataset_id, expected_revision in expected_revisions.items():
        source = DATASET_REGISTRY[dataset_id].source
        assert source is not None
        assert source.git_ref == expected_revision
