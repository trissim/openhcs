"""Typed expectations for in-tree CellProfiler .cppipe fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path


CELLPROFILER_EXAMPLES_ROOT_ENV = "CELLPROFILER_EXAMPLES_ROOT"
DEFAULT_CELLPROFILER_EXAMPLES_ROOT = Path("/tmp/cellprofiler_examples")


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
            status=CPPipeCorpusStatus.SUPPORTED,
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
