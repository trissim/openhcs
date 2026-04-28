"""Typed expectations for in-tree CellProfiler .cppipe fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


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
            name="ExampleFly",
            cppipe_path=pipelines_dir / "ExampleFly.cppipe",
            status=CPPipeCorpusStatus.SUPPORTED,
        ),
        CPPipeCorpusCase(
            name="ExampleHuman",
            cppipe_path=pipelines_dir / "ExampleHuman.cppipe",
            status=CPPipeCorpusStatus.KNOWN_INVALID,
            expected_error_substring=(
                "Module MeasureObjectIntensity(10) references unknown objects "
                "symbol 'Cytoplasm'. No prior module produces it."
            ),
        ),
    )
