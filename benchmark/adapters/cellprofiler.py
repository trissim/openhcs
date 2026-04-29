"""Native CellProfiler tool adapter."""

from __future__ import annotations

import shutil
import subprocess
from contextlib import ExitStack
from dataclasses import dataclass
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
from openhcs.core.runtime_equivalence import RuntimeOutputSnapshot


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
        self._configured_executable = Path(executable) if executable else None
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
        source = resolve_cppipe_source(request.cppipe_source)
        native_output_root = (
            request.output_dir
            / f"{request.dataset_path.name}_{request.pipeline_name}_native_cellprofiler"
        )
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

        with ExitStack() as stack:
            for metric in request.metrics:
                stack.enter_context(metric)
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=request.timeout_seconds,
                    check=False,
                )
            except FileNotFoundError as exc:
                raise ToolNotInstalledError(
                    f"CellProfiler executable not found: {command[0]}"
                ) from exc
        if result.returncode != 0:
            raise ToolExecutionError(
                "Native CellProfiler execution failed:\n"
                + _subprocess_output(result)
            )

        snapshot = RuntimeOutputSnapshot.from_output_root(native_output_root)
        provenance: dict[str, Any] = {
            "cellprofiler_version": self.version,
            "pipeline_source": "native_cppipe",
            "cppipe_path": str(source.path),
            "csv_output_count": len(snapshot.tables),
            "image_output_count": len(snapshot.images),
        }
        if source.reference_url is not None:
            provenance["cppipe_reference_url"] = source.reference_url
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
                "CellProfiler executable not found in PATH."
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
