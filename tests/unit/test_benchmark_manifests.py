from __future__ import annotations

import json
from pathlib import Path

from benchmark.cellprofiler_comparison import load_comparison_cases


def test_official30_portable_manifest_declares_roots_without_absolute_cases() -> None:
    manifest_path = Path("benchmark/manifests/official30_portable_axis1.json")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert len(payload["cases"]) == 30
    assert set(payload["path_roots"]) == {
        "axis_one_subsets",
        "cellprofiler_examples",
        "dataset_cache",
    }
    assert all(not Path(case["dataset_path"]).is_absolute() for case in payload["cases"])
    assert all(not Path(case["cppipe_path"]).is_absolute() for case in payload["cases"])

    cases = load_comparison_cases(manifest_path)

    assert len(cases) == 30
    assert cases[0].name == "ExampleColocalization"
    assert cases[-1].name == "cp_tutorial_translocation_start"
