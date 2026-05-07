"""Native CellProfiler tool adapter."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
from abc import ABC, abstractmethod
from contextlib import ExitStack
from dataclasses import dataclass
from os import environ
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import quote

from metaclass_registry import AutoRegisterMeta
from benchmark.adapters.cppipe_source import (
    CPPipeSourceRequest,
    CPPipeSourceResolution,
    resolve_cppipe_source,
)
from benchmark.contracts.metric import MetricCollector
from benchmark.contracts.tool_adapter import (
    BenchmarkResult,
    ToolAdapter,
    ToolExecutionError,
    ToolNotInstalledError,
)
from benchmark.timing import BenchmarkPhase, PhaseTimingTrace
from openhcs.core.source_schema_workspace import (
    SourceSchemaWorkspaceMaterialization,
    materialize_source_schema_workspace,
)
from openhcs.core.runtime_equivalence import RuntimeOutputSnapshot
from openhcs.interop.cellprofiler.parser import CPPipeParser
from openhcs.interop.cellprofiler.source_schema import compile_image_schema


BENCHMARK_CACHE_DOMAINS = frozenset({"native_reference"})
CELLPROFILER_EXECUTABLE_ENV = "CELLPROFILER_EXECUTABLE"
PYTHONHASHSEED_ENV = "PYTHONHASHSEED"
DETERMINISTIC_PYTHONHASHSEED = "0"
NATIVE_CELLPROFILER_SUCCESS_MARKER = ".cellprofiler_benchmark_reference.json"


@dataclass(frozen=True, slots=True)
class CellProfilerRunRequest:
    """Authoritative native CellProfiler run request."""

    dataset_path: Path
    pipeline_name: str
    pipeline_params: dict[str, Any]
    metrics: tuple[MetricCollector, ...]
    output_dir: Path

    @property
    def dataset_id(self) -> str:
        return str(self.pipeline_params.get("dataset_id", self.dataset_path.name))

    @property
    def timeout_seconds(self) -> float | None:
        value = self.pipeline_params.get("cellprofiler_timeout_seconds")
        if value is None:
            return None
        return float(value)

    @property
    def cppipe_source(self) -> CPPipeSourceRequest:
        return CPPipeSourceRequest.from_pipeline_params(
            dataset_id=self.dataset_id,
            output_dir=self.output_dir,
            pipeline_params=self.pipeline_params,
        )


@dataclass(frozen=True, slots=True)
class NativeCellProfilerInputDomain:
    """Concrete native-CellProfiler input domain for one run."""

    cppipe_path: Path
    input_dir: Path
    provenance: dict[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "cppipe_path", Path(self.cppipe_path))
        object.__setattr__(self, "input_dir", Path(self.input_dir))
        object.__setattr__(self, "provenance", dict(self.provenance))


class NativeCellProfilerInputDomainStrategy(ABC, metaclass=AutoRegisterMeta):
    """Prepare the native CellProfiler pipeline/input domain from typed source semantics."""

    __registry_key__ = "strategy_key"
    __skip_if_no_key__ = True
    strategy_key: ClassVar[str | None] = None

    @classmethod
    def prepare_for(
        cls,
        request: CellProfilerRunRequest,
        source: CPPipeSourceResolution,
        execution_cppipe_path: Path,
    ) -> NativeCellProfilerInputDomain:
        candidate_strategies = tuple(
            strategy_type() for strategy_type in cls.__registry__.values()
        )
        matching_strategies = tuple(
            strategy
            for strategy in candidate_strategies
            if strategy.accepts(request, source)
        )
        if len(matching_strategies) > 1:
            names = tuple(strategy.strategy_key for strategy in matching_strategies)
            raise ToolExecutionError(
                "Native CellProfiler input domain is ambiguous for "
                f"{source.path}: {names!r}."
            )
        if matching_strategies:
            return matching_strategies[0].prepare(
                request,
                source,
                execution_cppipe_path,
            )
        return DefaultNativeCellProfilerInputDomainStrategy().prepare(
            request,
            source,
            execution_cppipe_path,
        )

    @abstractmethod
    def accepts(
        self,
        request: CellProfilerRunRequest,
        source: CPPipeSourceResolution,
    ) -> bool:
        """Return whether this strategy owns the source semantics."""

    @abstractmethod
    def prepare(
        self,
        request: CellProfilerRunRequest,
        source: CPPipeSourceResolution,
        execution_cppipe_path: Path,
    ) -> NativeCellProfilerInputDomain:
        """Return the concrete native CellProfiler input domain."""


class EmbeddedImagePlaneNativeCellProfilerInputDomainStrategy(
    NativeCellProfilerInputDomainStrategy
):
    """Run embedded image-plane pipelines against a closed local source universe."""

    strategy_key = "embedded_image_planes"

    def accepts(
        self,
        request: CellProfilerRunRequest,
        source: CPPipeSourceResolution,
    ) -> bool:
        del request
        modules = CPPipeParser().parse(source.path)
        return bool(compile_image_schema(modules).image_plane_sources)

    def prepare(
        self,
        request: CellProfilerRunRequest,
        source: CPPipeSourceResolution,
        execution_cppipe_path: Path,
    ) -> NativeCellProfilerInputDomain:
        modules = CPPipeParser().parse(source.path)
        schema = compile_image_schema(modules)
        workspace = materialize_source_schema_workspace(
            request.dataset_path,
            request.output_dir / "native_cellprofiler_source_workspace",
            schema,
        )
        patched_cppipe_path = self._rewrite_embedded_image_plane_sources(
            execution_cppipe_path,
            request.output_dir / "native_cellprofiler_headless",
            workspace,
        )
        input_dir = request.output_dir / "native_cellprofiler_empty_input"
        if input_dir.exists():
            shutil.rmtree(input_dir)
        input_dir.mkdir(parents=True, exist_ok=True)
        return NativeCellProfilerInputDomain(
            cppipe_path=patched_cppipe_path,
            input_dir=input_dir,
            provenance={
                "native_input_domain_strategy": self.strategy_key,
                "native_source_workspace": str(workspace.workspace_root),
                "native_source_plane_count": len(workspace.primary_mappings),
            },
        )

    def _rewrite_embedded_image_plane_sources(
        self,
        cppipe_path: Path,
        target_dir: Path,
        workspace: SourceSchemaWorkspaceMaterialization,
    ) -> Path:
        source_text = cppipe_path.read_text(encoding="utf-8")
        lines = source_text.splitlines()
        source_uris = tuple(
            self._file_uri((workspace.workspace_root / real_path).resolve())
            for real_path in workspace.primary_mappings.values()
        )
        if not source_uris:
            raise ToolExecutionError(
                "Embedded image-plane native input strategy requires at least one "
                "materialized source mapping."
            )
        patched_lines = self._replace_image_plane_rows(lines, source_uris)
        target_dir.mkdir(parents=True, exist_ok=True)
        patched_path = target_dir / cppipe_path.name
        patched_path.write_text("\n".join(patched_lines) + "\n", encoding="utf-8")
        return patched_path

    def _replace_image_plane_rows(
        self,
        lines: list[str],
        source_uris: tuple[str, ...],
    ) -> list[str]:
        parser = CPPipeParser()
        for index, line in enumerate(lines):
            version_match = parser.IMAGE_PLANE_DETAILS_PATTERN.match(line.strip())
            if version_match is None:
                continue
            expected_count = int(version_match.group("count"))
            if expected_count != len(source_uris):
                raise ToolExecutionError(
                    "Embedded image-plane count does not match materialized local "
                    f"source count: cppipe={expected_count}, local={len(source_uris)}."
                )
            header_index = index + 1
            row_start = header_index + 1
            row_stop = row_start + expected_count
            if header_index >= len(lines) or row_stop > len(lines):
                raise ToolExecutionError("Malformed embedded image-plane table.")
            header = self._csv_image_plane_row(lines[header_index])
            if header[:4] != ["URL", "Series", "Index", "Channel"]:
                raise ToolExecutionError(
                    "Embedded image-plane table has unsupported header "
                    f"{header!r}."
                )
            replacement_rows = [
                f'"{source_uri}",,,'
                for source_uri in source_uris
            ]
            return [
                *lines[:row_start],
                *replacement_rows,
                *lines[row_stop:],
            ]
        raise ToolExecutionError("Pipeline has no embedded image-plane table to rewrite.")

    def _file_uri(self, path: Path) -> str:
        return "file://" + quote(str(path))

    def _csv_image_plane_row(self, line: str) -> list[str]:
        return next(csv.reader([line]))


class DefaultNativeCellProfilerInputDomainStrategy(
    NativeCellProfilerInputDomainStrategy
):
    """Run native CellProfiler against the visible dataset directory."""

    strategy_key = None

    def accepts(
        self,
        request: CellProfilerRunRequest,
        source: CPPipeSourceResolution,
    ) -> bool:
        del request, source
        return False

    def prepare(
        self,
        request: CellProfilerRunRequest,
        source: CPPipeSourceResolution,
        execution_cppipe_path: Path,
    ) -> NativeCellProfilerInputDomain:
        del source
        return NativeCellProfilerInputDomain(
            cppipe_path=execution_cppipe_path,
            input_dir=request.dataset_path,
            provenance={"native_input_domain_strategy": "dataset_folder"},
        )


class CellProfilerAdapter(ToolAdapter):
    """Run a native CellProfiler `.cppipe` as the semantic reference tool."""

    name = "CellProfiler"

    def __init__(self, executable: str | Path | None = None) -> None:
        configured_executable = executable or environ.get(CELLPROFILER_EXECUTABLE_ENV)
        self._configured_executable = (
            Path(configured_executable) if configured_executable else None
        )
        self.version = "unknown"

    def validate_installation(self) -> None:
        """Check that the CellProfiler command-line runner is available."""
        executable = self._cellprofiler_executable()
        try:
            result = subprocess.run(
                [str(executable), "--version"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except FileNotFoundError as exc:
            raise ToolNotInstalledError(
                f"CellProfiler executable not found: {executable}"
            ) from exc
        if result.returncode != 0:
            raise ToolExecutionError(
                "Failed to query CellProfiler version:\n"
                + _subprocess_output(result)
            )
        self.version = (result.stdout or result.stderr).strip() or "unknown"

    def run(
        self,
        dataset_path: Path,
        pipeline_name: str,
        pipeline_params: dict[str, Any],
        metrics: list[Any],
        output_dir: Path,
    ) -> BenchmarkResult:
        """Execute a native CellProfiler pipeline headlessly."""
        request = CellProfilerRunRequest(
            dataset_path=Path(dataset_path),
            pipeline_name=pipeline_name,
            pipeline_params=dict(pipeline_params),
            metrics=self._validated_metric_collectors(metrics),
            output_dir=Path(output_dir),
        )
        request.output_dir.mkdir(parents=True, exist_ok=True)
        phase_timing = PhaseTimingTrace(
            run_id=f"{request.dataset_id}:{request.pipeline_name}:native_cellprofiler",
            pipeline_name=request.pipeline_name,
            tool=self.name,
        )
        with phase_timing.phase(BenchmarkPhase.RESOLVE_SOURCE):
            source = resolve_cppipe_source(request.cppipe_source)
            execution_cppipe_path = _headless_cellprofiler_cppipe_path(
                source.path,
                request.output_dir,
            )
            native_input_domain = NativeCellProfilerInputDomainStrategy.prepare_for(
                request,
                source,
                execution_cppipe_path,
            )
        native_output_root = native_cellprofiler_output_root(request)
        if native_output_root.exists():
            shutil.rmtree(native_output_root)
        native_output_root.mkdir(parents=True, exist_ok=True)
        command = (
            str(self._cellprofiler_executable()),
            "-c",
            "-r",
            "-p",
            str(native_input_domain.cppipe_path),
            "-i",
            str(native_input_domain.input_dir),
            "-o",
            str(native_output_root),
        )
        subprocess_env = {
            **environ,
            PYTHONHASHSEED_ENV: environ.get(
                PYTHONHASHSEED_ENV,
                DETERMINISTIC_PYTHONHASHSEED,
            ),
        }

        with ExitStack() as stack:
            for metric in request.metrics:
                stack.enter_context(metric)
            try:
                with phase_timing.phase(BenchmarkPhase.EXECUTE_NATIVE_CP):
                    result = subprocess.run(
                        command,
                        cwd=native_output_root,
                        env=subprocess_env,
                        capture_output=True,
                        text=True,
                        timeout=request.timeout_seconds,
                        check=False,
                    )
            except subprocess.TimeoutExpired as exc:
                raise ToolExecutionError(
                    "Native CellProfiler execution timed out "
                    f"after {request.timeout_seconds}s:\n"
                    + " ".join(command)
                ) from exc
            except FileNotFoundError as exc:
                raise ToolNotInstalledError(
                    f"CellProfiler executable not found: {command[0]}"
                ) from exc
        if result.returncode != 0:
            raise ToolExecutionError(
                "Native CellProfiler execution failed:\n"
                + _subprocess_output(result)
            )

        with phase_timing.phase(BenchmarkPhase.SNAPSHOT_OUTPUTS):
            snapshot = RuntimeOutputSnapshot.from_output_root(native_output_root)
        provenance: dict[str, Any] = {
            "cellprofiler_version": self.version,
            "pipeline_source": "native_cppipe",
            "cppipe_path": str(source.path),
            "execution_cppipe_path": str(native_input_domain.cppipe_path),
            "native_input_dir": str(native_input_domain.input_dir),
            "csv_output_count": len(snapshot.tables),
            "image_output_count": len(snapshot.images),
            "pythonhashseed": subprocess_env[PYTHONHASHSEED_ENV],
            "phase_timing_records": phase_timing.payloads(),
            **native_input_domain.provenance,
        }
        if source.reference_url is not None:
            provenance["cppipe_reference_url"] = source.reference_url
        _write_native_reference_success_marker(native_output_root, provenance)
        return BenchmarkResult(
            tool_name=self.name,
            dataset_id=request.dataset_id,
            pipeline_name=request.pipeline_name,
            metrics={
                metric.name: metric.get_result()
                for metric in request.metrics
            },
            output_path=native_output_root,
            success=True,
            error_message=None,
            provenance=provenance,
        )

    def _cellprofiler_executable(self) -> Path:
        if self._configured_executable is not None:
            return self._configured_executable
        executable = shutil.which("cellprofiler")
        if executable is None:
            raise ToolNotInstalledError(
                "CellProfiler executable not configured and not found in PATH. "
                f"Set {CELLPROFILER_EXECUTABLE_ENV} or pass an executable path."
            )
        return Path(executable)

    def _validated_metric_collectors(
        self,
        metrics: list[Any],
    ) -> tuple[MetricCollector, ...]:
        validated_metrics: list[MetricCollector] = []
        for metric in metrics:
            if not isinstance(metric, MetricCollector):
                raise ToolExecutionError(
                    f"Metric {metric} does not extend MetricCollector"
                )
            validated_metrics.append(metric)
        return tuple(validated_metrics)


def _subprocess_output(result: subprocess.CompletedProcess[str]) -> str:
    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    return "\n".join(part for part in (stdout, stderr) if part)


def _headless_cellprofiler_cppipe_path(cppipe_path: Path, output_dir: Path) -> Path:
    """Return a CellProfiler pipeline safe for non-interactive native execution."""
    source_text = Path(cppipe_path).read_text(encoding="utf-8")
    patched_text = source_text.replace(
        "Overwrite existing files without warning?:No",
        "Overwrite existing files without warning?:Yes",
    )
    if patched_text == source_text:
        return cppipe_path
    patched_path = output_dir / "native_cellprofiler_headless" / cppipe_path.name
    patched_path.parent.mkdir(parents=True, exist_ok=True)
    patched_path.write_text(patched_text, encoding="utf-8")
    return patched_path


def native_cellprofiler_output_root(request: CellProfilerRunRequest) -> Path:
    """Return the output directory owned by one native CellProfiler run."""
    return (
        request.output_dir
        / f"{request.dataset_path.name}_{request.pipeline_name}_native_cellprofiler"
    )


def native_cellprofiler_reference_is_complete(reference_output_dir: Path) -> bool:
    """Return whether a native reference has a registered completeness proof."""
    reference = Path(reference_output_dir)
    if (reference / NATIVE_CELLPROFILER_SUCCESS_MARKER).is_file():
        return NativeCellProfilerSuccessMarkerReferenceCompletenessStrategy().is_complete(
            reference
        )
    return any(
        strategy_type().is_complete(reference)
        for strategy_type in NativeCellProfilerReferenceCompletenessStrategy.__registry__.values()
    )


class NativeCellProfilerReferenceCompletenessStrategy(
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Nominal proof that a native CellProfiler reference can be reused."""

    __registry_key__ = "proof_name"
    __skip_if_no_key__ = True
    proof_name: ClassVar[str | None] = None

    @abstractmethod
    def is_complete(self, reference_output_dir: Path) -> bool:
        """Return whether this proof accepts the reference directory."""


class NativeCellProfilerSuccessMarkerReferenceCompletenessStrategy(
    NativeCellProfilerReferenceCompletenessStrategy
):
    """Accept references explicitly marked by the native adapter."""

    proof_name = "success_marker"

    def is_complete(self, reference_output_dir: Path) -> bool:
        marker = reference_output_dir / NATIVE_CELLPROFILER_SUCCESS_MARKER
        if not marker.is_file():
            return False
        provenance = native_cellprofiler_reference_provenance(reference_output_dir)
        cppipe_path = provenance.get("cppipe_path")
        if not isinstance(cppipe_path, str):
            return True
        source_path = Path(cppipe_path)
        if not source_path.exists():
            return True
        modules = CPPipeParser().parse(source_path)
        source_schema = compile_image_schema(modules)
        if not source_schema.image_plane_sources:
            return True
        return (
            provenance.get("native_input_domain_strategy")
            == EmbeddedImagePlaneNativeCellProfilerInputDomainStrategy.strategy_key
        )


class NativeCellProfilerSemanticSnapshotReferenceCompletenessStrategy(
    NativeCellProfilerReferenceCompletenessStrategy
):
    """Accept references with loadable semantic output artifacts."""

    proof_name = "semantic_snapshot"

    def is_complete(self, reference_output_dir: Path) -> bool:
        if not reference_output_dir.exists():
            return False
        try:
            snapshot = RuntimeOutputSnapshot.from_output_root(reference_output_dir)
        except (OSError, ValueError):
            return False
        return bool(snapshot.tables or snapshot.images)


def native_cellprofiler_reference_provenance(
    reference_output_dir: Path,
) -> dict[str, Any]:
    """Load successful native-reference provenance, if present."""
    marker = Path(reference_output_dir) / NATIVE_CELLPROFILER_SUCCESS_MARKER
    if not marker.is_file():
        return {}
    payload = json.loads(marker.read_text(encoding="utf-8"))
    provenance = payload.get("provenance")
    return dict(provenance) if isinstance(provenance, dict) else {}


def _write_native_reference_success_marker(
    reference_output_dir: Path,
    provenance: dict[str, Any],
) -> None:
    marker = reference_output_dir / NATIVE_CELLPROFILER_SUCCESS_MARKER
    marker.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "provenance": provenance,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
