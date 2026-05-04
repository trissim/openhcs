"""Native CellProfiler tool adapter."""

from __future__ import annotations

import json
import shutil
import subprocess
from contextlib import ExitStack
from dataclasses import dataclass
from os import environ
from pathlib import Path
from typing import Any

from benchmark.adapters.cppipe_source import (
    CPPipeSourceRequest,
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
from openhcs.core.runtime_equivalence import RuntimeOutputSnapshot


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
        native_output_root = native_cellprofiler_output_root(request)
        if native_output_root.exists():
            shutil.rmtree(native_output_root)
        native_output_root.mkdir(parents=True, exist_ok=True)
        command = (
            str(self._cellprofiler_executable()),
            "-c",
            "-r",
            "-p",
            str(source.path),
            "-i",
            str(request.dataset_path),
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
            "csv_output_count": len(snapshot.tables),
            "image_output_count": len(snapshot.images),
            "pythonhashseed": subprocess_env[PYTHONHASHSEED_ENV],
            "phase_timing_records": phase_timing.payloads(),
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


def native_cellprofiler_output_root(request: CellProfilerRunRequest) -> Path:
    """Return the output directory owned by one native CellProfiler run."""
    return (
        request.output_dir
        / f"{request.dataset_path.name}_{request.pipeline_name}_native_cellprofiler"
    )


def native_cellprofiler_reference_is_complete(reference_output_dir: Path) -> bool:
    """Return whether a native reference was explicitly marked successful."""
    return (Path(reference_output_dir) / NATIVE_CELLPROFILER_SUCCESS_MARKER).is_file()


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
