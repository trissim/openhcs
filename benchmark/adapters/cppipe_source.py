"""Shared .cppipe source resolution for benchmark adapters."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen

from benchmark.contracts.tool_adapter import ToolExecutionError
from benchmark.datasets.registry import get_dataset_spec


CPPipeReferenceMaterializer = Callable[[str, Path], Path]


@dataclass(frozen=True, slots=True)
class CPPipeSourceRequest:
    """Typed request for resolving a CellProfiler pipeline source."""

    dataset_id: str
    output_dir: Path
    cppipe_path: Path | None = None
    reference_url: str | None = None
    reference_index: int | None = None

    @classmethod
    def from_pipeline_params(
        cls,
        *,
        dataset_id: str,
        output_dir: Path,
        pipeline_params: Mapping[str, Any],
    ) -> "CPPipeSourceRequest":
        cppipe_value = pipeline_params.get("cppipe_path") or pipeline_params.get(
            "cppipe_file"
        )
        reference_url = pipeline_params.get("cppipe_reference_url")
        reference_index = pipeline_params.get("cppipe_reference_index")
        return cls(
            dataset_id=dataset_id,
            output_dir=Path(output_dir),
            cppipe_path=Path(cppipe_value) if cppipe_value else None,
            reference_url=str(reference_url) if reference_url is not None else None,
            reference_index=(
                int(reference_index) if reference_index is not None else None
            ),
        )

    def __post_init__(self) -> None:
        if not self.dataset_id:
            raise ValueError("CPPipeSourceRequest.dataset_id cannot be empty.")
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.reference_url == "":
            raise ValueError("CPPipeSourceRequest.reference_url cannot be empty.")


@dataclass(frozen=True, slots=True)
class CPPipeSourceResolution:
    """Resolved CellProfiler pipeline source."""

    path: Path
    reference_url: str | None = None


def resolve_cppipe_source(
    request: CPPipeSourceRequest,
    *,
    materialize_reference: CPPipeReferenceMaterializer | None = None,
) -> CPPipeSourceResolution:
    """Resolve a local or dataset-owned .cppipe path."""
    materializer = materialize_reference or materialize_cppipe_reference
    if request.cppipe_path is not None:
        if not request.cppipe_path.exists():
            raise ToolExecutionError(f".cppipe file not found: {request.cppipe_path}")
        return CPPipeSourceResolution(request.cppipe_path)

    reference_url = request.reference_url
    if reference_url is None and request.reference_index is not None:
        reference_url = reference_cppipe_url(
            request.dataset_id,
            request.reference_index,
        )
    if reference_url is None:
        raise ToolExecutionError(
            "CellProfiler pipeline execution requires cppipe_path, cppipe_file, "
            "cppipe_reference_url, or cppipe_reference_index."
        )

    return CPPipeSourceResolution(
        materializer(reference_url, request.output_dir / "cppipe_references"),
        reference_url=reference_url,
    )


def reference_cppipe_url(dataset_id: str, reference_index: int) -> str:
    """Resolve one canonical .cppipe URL from the dataset registry."""
    try:
        dataset_spec = get_dataset_spec(dataset_id)
    except KeyError as exc:
        raise ToolExecutionError(
            f"Unknown dataset id {dataset_id!r} for cppipe reference lookup."
        ) from exc
    try:
        return dataset_spec.reference_cppipe_urls[reference_index]
    except IndexError as exc:
        raise ToolExecutionError(
            f"Dataset {dataset_id!r} exposes "
            f"{len(dataset_spec.reference_cppipe_urls)} cppipe references; "
            f"index {reference_index} is out of range."
        ) from exc


def materialize_cppipe_reference(
    reference_url: str,
    target_dir: Path,
) -> Path:
    """Download one canonical .cppipe file into a stable local path."""
    target_dir.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(reference_url)
    filename = Path(parsed.path).name or "reference.cppipe"
    target_path = target_dir / filename
    if target_path.exists():
        return target_path
    with urlopen(reference_url) as response:  # noqa: S310
        target_path.write_bytes(response.read())
    return target_path
