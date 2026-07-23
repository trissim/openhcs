"""Typed expectations for in-tree CellProfiler .cppipe fixtures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path

from benchmark.contracts.comparison_manifest import ComparisonManifest
from benchmark.datasets.cache import (
    CELLPROFILER_EXAMPLES_ROOT_ENV,
    default_cellprofiler_examples_root,
)


DEFAULT_CELLPROFILER_EXAMPLES_ROOT = default_cellprofiler_examples_root()


class CPPipeCorpusStatus(str, Enum):
    """Compatibility status for one shipped .cppipe fixture."""

    SUPPORTED = "supported"
    KNOWN_INVALID = "known_invalid"


@dataclass(frozen=True, slots=True)
class CPPipeCorpusCase:
    """Authoritative compatibility expectation for one shipped .cppipe file."""

    name: str
    cppipe_path: Path
    status: CPPipeCorpusStatus
    source_root: Path | None = None
    expected_error_substring: str | None = None


def in_tree_cppipe_corpus() -> tuple[CPPipeCorpusCase, ...]:
    """Return the tracked in-tree .cppipe corpus with explicit expectations."""

    pipelines_dir = Path(__file__).resolve().parents[1] / "cellprofiler_pipelines"
    return (
        CPPipeCorpusCase(
            name="BBBC021Analysis",
            cppipe_path=pipelines_dir / "BBBC021_analysis.cppipe",
            status=CPPipeCorpusStatus.SUPPORTED,
        ),
        CPPipeCorpusCase(
            name="BBBC021Illumination",
            cppipe_path=pipelines_dir / "BBBC021_illum.cppipe",
            status=CPPipeCorpusStatus.SUPPORTED,
        ),
        CPPipeCorpusCase(
            name="ExampleFly",
            cppipe_path=pipelines_dir / "ExampleFly.cppipe",
            status=CPPipeCorpusStatus.KNOWN_INVALID,
            expected_error_substring=(
                "No CellProfiler module declaration is registered for 'LoadData'"
            ),
        ),
        CPPipeCorpusCase(
            name="ExampleHuman",
            cppipe_path=pipelines_dir / "ExampleHuman.cppipe",
            status=CPPipeCorpusStatus.SUPPORTED,
        ),
    )


def official_cellprofiler3_cppipe_corpus(
    examples_root: Path | None = None,
) -> tuple[CPPipeCorpusCase, ...]:
    """Return discovered official CellProfiler3 example pipelines when available."""

    root = examples_root or Path(
        os.environ.get(
            CELLPROFILER_EXAMPLES_ROOT_ENV,
            str(DEFAULT_CELLPROFILER_EXAMPLES_ROOT),
        )
    )
    cppipe_dir = root / "CellProfiler3Pipelines"
    if not cppipe_dir.exists():
        return ()
    return tuple(
        CPPipeCorpusCase(
            name=cppipe_path.stem,
            cppipe_path=cppipe_path,
            status=CPPipeCorpusStatus.SUPPORTED,
        )
        for cppipe_path in sorted(cppipe_dir.glob("*.cppipe"))
    )


def default_cppipe_corpus() -> tuple[CPPipeCorpusCase, ...]:
    """Return all locally available .cppipe acceptance corpus cases."""

    return (
        *in_tree_cppipe_corpus(),
        *official_cellprofiler3_cppipe_corpus(),
    )


def comparison_manifest_cppipe_corpus(
    manifest_path: Path,
) -> tuple[CPPipeCorpusCase, ...]:
    """Project a benchmark comparison manifest into .cppipe coverage cases."""

    manifest = ComparisonManifest.load(manifest_path)
    payload = manifest.payload
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, Sequence):
        raise ValueError("Benchmark manifest must contain a 'cases' sequence.")

    cases: list[CPPipeCorpusCase] = []
    for raw_case in raw_cases:
        if not isinstance(raw_case, Mapping):
            raise ValueError(f"Benchmark case must be an object: {raw_case!r}")
        cases.append(
            CPPipeCorpusCase(
                name=str(raw_case["name"]),
                cppipe_path=manifest.path_resolver.resolve(raw_case, "cppipe_path"),
                status=CPPipeCorpusStatus.SUPPORTED,
                source_root=manifest.path_resolver.resolve(raw_case, "dataset_path"),
            )
        )
    return tuple(cases)


def comparison_manifests_cppipe_corpus(
    manifest_paths: Sequence[Path],
) -> tuple[CPPipeCorpusCase, ...]:
    """Project multiple benchmark manifests into one .cppipe coverage corpus."""

    return tuple(
        case
        for manifest_path in manifest_paths
        for case in comparison_manifest_cppipe_corpus(manifest_path)
    )
