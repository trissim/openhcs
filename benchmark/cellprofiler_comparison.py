"""CellProfiler versus OpenHCS benchmark result collection."""

from __future__ import annotations

import csv
import json
import platform
import shutil
import statistics
import sys
import time
import traceback
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmark.contracts.tool_adapter import BenchmarkResult
from benchmark.contracts.tool_adapter import ToolExecutionError
from benchmark.adapters.cellprofiler import (
    native_cellprofiler_reference_is_complete,
    native_cellprofiler_reference_scope_slugs,
)
from benchmark.adapters.openhcs import OPENHCS_AXIS_FILTER_PARAM
from benchmark.adapters.openhcs import OPENHCS_MAX_AXIS_COUNT_PARAM
from benchmark.adapters.openhcs import OPENHCS_NUM_WORKERS_PARAM
from benchmark.adapters.openhcs import OPENHCS_START_METHOD_PARAM
from benchmark.adapters.openhcs import OPENHCS_USE_THREADING_PARAM
from benchmark.datasets.visible_source import resolve_visible_source_path
from benchmark.metrics.memory import MemoryMetric
from benchmark.metrics.time import TimeMetric
from benchmark.runner import CellProfilerCompatibilityResult
from benchmark.runner import run_cellprofiler_cppipe_parity
from benchmark.contracts.metric import MetricCollector
from benchmark.contracts.comparison_manifest import ComparisonManifest

if TYPE_CHECKING:
    from benchmark.converter.compatibility_matrix import CellProfilerCompatibilityReport


BENCHMARK_CACHE_DOMAINS = frozenset({"harness"})
SUITE_ID_FIELD = "suite_id"
CASE_NAME_FIELD = "case_name"
REPETITION_FIELD = "repetition"
DATASET_ID_FIELD = "dataset_id"
ASSAY_CATEGORY_FIELD = "assay_category"
MODULE_CATEGORY_FIELD = "module_category"
EQUIVALENT_FIELD = "equivalent"
DIFFERENCE_COUNT_FIELD = "difference_count"
NUMERIC_ABS_TOLERANCE_FIELD = "numeric_abs_tolerance"
NUMERIC_REL_TOLERANCE_FIELD = "numeric_rel_tolerance"
NATIVE_EXECUTION_SECONDS_FIELD = "native_execution_seconds"
OPENHCS_EXECUTION_SECONDS_FIELD = "openhcs_execution_seconds"
NATIVE_TOTAL_PHASE_SECONDS_FIELD = "native_total_phase_seconds"
OPENHCS_TOTAL_PHASE_SECONDS_FIELD = "openhcs_total_phase_seconds"
NATIVE_PEAK_MEMORY_MB_FIELD = "native_peak_memory_mb"
OPENHCS_PEAK_MEMORY_MB_FIELD = "openhcs_peak_memory_mb"
SPEEDUP_FIELD = "speedup"
TOTAL_PHASE_SPEEDUP_FIELD = "total_phase_speedup"
SPEEDUP_TARGET_FIELD = "speedup_target"
MEETS_SPEEDUP_TARGET_FIELD = "meets_speedup_target"
MEETS_EXECUTION_SPEEDUP_TARGET_FIELD = "meets_execution_speedup_target"
MEETS_TOTAL_PHASE_SPEEDUP_TARGET_FIELD = "meets_total_phase_speedup_target"
PARITY_ACCURACY_FIELD = "parity_accuracy"
NATIVE_CACHED_FIELD = "native_cached"
OPENHCS_CACHED_FIELD = "openhcs_cached"
NATIVE_ERROR_MESSAGE_FIELD = "native_error_message"
OPENHCS_ERROR_MESSAGE_FIELD = "openhcs_error_message"
NATIVE_OUTPUT_PATH_FIELD = "native_output_path"
OPENHCS_OUTPUT_PATH_FIELD = "openhcs_output_path"
TOOL_FIELD = "tool"
PHASE_FIELD = "phase"
SECONDS_FIELD = "seconds"
N_FIELD = "n"
EQUIVALENT_COUNT_FIELD = "equivalent_count"
MEDIAN_NATIVE_EXECUTION_SECONDS_FIELD = "median_native_execution_seconds"
MEDIAN_OPENHCS_EXECUTION_SECONDS_FIELD = "median_openhcs_execution_seconds"
MEDIAN_NATIVE_TOTAL_PHASE_SECONDS_FIELD = "median_native_total_phase_seconds"
MEDIAN_OPENHCS_TOTAL_PHASE_SECONDS_FIELD = "median_openhcs_total_phase_seconds"
MEDIAN_NATIVE_PEAK_MEMORY_MB_FIELD = "median_native_peak_memory_mb"
MEDIAN_OPENHCS_PEAK_MEMORY_MB_FIELD = "median_openhcs_peak_memory_mb"
MEDIAN_SPEEDUP_FIELD = "median_speedup"
MEDIAN_TOTAL_PHASE_SPEEDUP_FIELD = "median_total_phase_speedup"
MIN_PARITY_ACCURACY_FIELD = "min_parity_accuracy"
MODULE_NAME_FIELD = "module_name"
CORPUS_COVERAGE_FIELD = "corpus_coverage"
ABSORPTION_COVERAGE_FIELD = "absorption_coverage"
CPPIPE_CASE_NAMES_FIELD = "cppipe_case_names"
IMPORTABLE_FIELD = "importable"
PROCESSING_CONTRACT_FIELD = "processing_contract"
PROCESSING_CONTRACT_SOURCE_FIELD = "processing_contract_source"
ARTIFACT_CONTRACT_COVERAGE_FIELD = "artifact_contract_coverage"
SOURCE_COVERAGE_FIELD = "source_coverage"
SEMANTIC_FAMILY_FIELD = "semantic_family"
FAMILY_COVERAGE_FIELD = "family_coverage"
FAMILY_SUPPORTED_MODULES_FIELD = "family_supported_modules"
FAMILY_ABSORBED_MODULES_FIELD = "family_absorbed_modules"
CATEGORY_FIELD = "category"
DIMENSIONALITY_FIELD = "dimensionality"
RESPECTS_MASKS_FIELD = "respects_masks"
DEFAULT_SPEEDUP_TARGET = 5.0
OPENHCS_BENCHMARK_CACHE_MARKER = ".openhcs_benchmark_cache.json"
MODULE_COVERAGE_SUMMARY_JSON = "module_coverage_summary.json"
MODULE_COVERAGE_CPPIPE_MODULES_CSV = "module_coverage_cppipe_modules.csv"
MODULE_COVERAGE_CPPIPE_SETTINGS_CSV = "module_coverage_cppipe_settings.csv"
MODULE_COVERAGE_ABSORBED_MODULES_CSV = "module_coverage_absorbed_modules.csv"
MODULE_COVERAGE_SOURCE_MODULES_CSV = "module_coverage_source_modules.csv"
MODULE_COVERAGE_SEMANTIC_FAMILIES_CSV = "module_coverage_semantic_families.csv"
CsvRow = Mapping[str, object]
CsvRowBuilder = Callable[
    [Sequence["CellProfilerComparisonObservation"]],
    Iterable[CsvRow],
]
@dataclass(frozen=True, slots=True)
class CsvTableSpec:
    """Authoritative CSV table projection."""

    fieldnames: tuple[str, ...]
    rows: CsvRowBuilder


@dataclass(frozen=True, slots=True)
class ModuleCoverageSummaryPayload:
    """Serializable benchmark-manifest CellProfiler module coverage summary."""

    manifest_path: str
    cppipe_case_count: int
    supported_cppipe_case_count: int
    known_invalid_cppipe_case_count: int
    module_instance_count: int
    unique_cppipe_module_count: int
    supported_absorbed_processing_module_count: int
    known_invalid_absorbed_processing_module_count: int
    untested_absorbed_processing_module_count: int
    infrastructure_cppipe_module_count: int
    missing_processing_cppipe_module_count: int
    cppipe_setting_row_count: int
    covered_cppipe_setting_row_count: int
    unmapped_cppipe_setting_row_count: int
    supported_absorbed_processing_modules: tuple[str, ...]
    known_invalid_absorbed_processing_modules: tuple[str, ...]
    untested_absorbed_processing_modules: tuple[str, ...]
    infrastructure_cppipe_modules: tuple[str, ...]
    missing_processing_cppipe_modules: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CPPipeModuleCoverageRow:
    """CSV row for one module observed in benchmark .cppipe files."""

    module_name: str
    corpus_coverage: str
    absorption_coverage: str
    cppipe_case_names: str


@dataclass(frozen=True, slots=True)
class AbsorbedModuleCoverageRow:
    """CSV row for one absorbed CellProfiler library module."""

    module_name: str
    corpus_coverage: str
    importable: bool
    processing_contract: str
    processing_contract_source: str
    artifact_contract_coverage: str


@dataclass(frozen=True, slots=True)
class SourceModuleCoverageRow:
    """CSV row for one checked-in CellProfiler source module."""

    module_name: str
    source_coverage: str


@dataclass(frozen=True, slots=True)
class SemanticFamilyCoverageRow:
    """CSV row for semantic-family coverage evidence."""

    module_name: str
    semantic_family: str
    family_coverage: str
    corpus_coverage: str
    category: str
    dimensionality: str
    respects_masks: bool
    family_supported_modules: str
    family_absorbed_modules: str


@dataclass(frozen=True, slots=True)
class CPPipeSettingCoverageRow:
    """CSV row for one concrete setting observed in benchmark .cppipe files."""

    case_name: str
    module_name: str
    canonical_module_name: str
    module_num: int
    setting_name: str
    normalized_setting_name: str
    coverage: str
    value: str


@dataclass(frozen=True, slots=True)
class CsvRowsArtifact:
    """One materialized CSV artifact with nominal row records."""

    filename: str
    fieldnames: tuple[str, ...]
    rows: tuple[object, ...]

    def write_to(self, output_root: Path) -> None:
        path = output_root / self.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(asdict(row) for row in self.rows)


@dataclass(frozen=True, slots=True)
class ObservationCsvArtifact:
    """One benchmark observation CSV artifact backed by a table spec."""

    path: Path
    table: CsvTableSpec
    observations: Sequence["CellProfilerComparisonObservation"]

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.table.fieldnames)
            writer.writeheader()
            writer.writerows(self.table.rows(self.observations))


@dataclass(frozen=True, slots=True)
class ModuleCoverageArtifacts:
    """Complete module-coverage artifact set for one benchmark manifest."""

    summary: ModuleCoverageSummaryPayload
    cppipe_modules: tuple[CPPipeModuleCoverageRow, ...]
    cppipe_settings: tuple[CPPipeSettingCoverageRow, ...]
    absorbed_modules: tuple[AbsorbedModuleCoverageRow, ...]
    source_modules: tuple[SourceModuleCoverageRow, ...]
    semantic_families: tuple[SemanticFamilyCoverageRow, ...]

    @classmethod
    def from_report(
        cls,
        report: "CellProfilerCompatibilityReport",
        *,
        manifest_path: Path,
    ) -> "ModuleCoverageArtifacts":
        coverage = report.benchmark_coverage
        return cls(
            summary=ModuleCoverageSummaryPayload(
                manifest_path=str(manifest_path),
                cppipe_case_count=coverage.cppipe_case_count,
                supported_cppipe_case_count=coverage.supported_cppipe_case_count,
                known_invalid_cppipe_case_count=(
                    coverage.known_invalid_cppipe_case_count
                ),
                module_instance_count=coverage.module_instance_count,
                unique_cppipe_module_count=coverage.unique_cppipe_module_count,
                supported_absorbed_processing_module_count=(
                    coverage.supported_absorbed_processing_module_count
                ),
                known_invalid_absorbed_processing_module_count=(
                    coverage.known_invalid_absorbed_processing_module_count
                ),
                untested_absorbed_processing_module_count=(
                    coverage.untested_absorbed_processing_module_count
                ),
                infrastructure_cppipe_module_count=(
                    coverage.infrastructure_cppipe_module_count
                ),
                missing_processing_cppipe_module_count=(
                    coverage.missing_processing_cppipe_module_count
                ),
                cppipe_setting_row_count=len(report.cppipe_settings),
                covered_cppipe_setting_row_count=sum(
                    1
                    for setting in report.cppipe_settings
                    if setting.coverage.is_covered
                ),
                unmapped_cppipe_setting_row_count=sum(
                    1
                    for setting in report.cppipe_settings
                    if setting.coverage.value == "unmapped"
                ),
                supported_absorbed_processing_modules=(
                    coverage.supported_absorbed_processing_modules
                ),
                known_invalid_absorbed_processing_modules=(
                    coverage.known_invalid_absorbed_processing_modules
                ),
                untested_absorbed_processing_modules=(
                    coverage.untested_absorbed_processing_modules
                ),
                infrastructure_cppipe_modules=coverage.infrastructure_cppipe_modules,
                missing_processing_cppipe_modules=(
                    coverage.missing_processing_cppipe_modules
                ),
            ),
            cppipe_modules=tuple(
                CPPipeModuleCoverageRow(
                    module_name=module.module_name,
                    corpus_coverage=module.corpus_coverage.value,
                    absorption_coverage=module.absorption_coverage.value,
                    cppipe_case_names=";".join(module.cppipe_case_names),
                )
                for module in report.cppipe_modules
            ),
            cppipe_settings=tuple(
                CPPipeSettingCoverageRow(
                    case_name=setting.case_name,
                    module_name=setting.module_name,
                    canonical_module_name=setting.canonical_module_name,
                    module_num=setting.module_num,
                    setting_name=setting.setting_name,
                    normalized_setting_name=setting.normalized_setting_name,
                    coverage=setting.coverage.value,
                    value=setting.value,
                )
                for setting in report.cppipe_settings
            ),
            absorbed_modules=tuple(
                AbsorbedModuleCoverageRow(
                    module_name=module.module_name,
                    corpus_coverage=module.corpus_coverage.value,
                    importable=module.importable,
                    processing_contract=(
                        module.processing_contract.value
                        if module.processing_contract is not None
                        else ""
                    ),
                    processing_contract_source=(
                        module.processing_contract_source.value
                        if module.processing_contract_source is not None
                        else ""
                    ),
                    artifact_contract_coverage=(
                        module.artifact_contract_coverage.value
                    ),
                )
                for module in report.modules
            ),
            source_modules=tuple(
                SourceModuleCoverageRow(
                    module_name=module.module_name,
                    source_coverage=module.coverage.value,
                )
                for module in report.source_modules
            ),
            semantic_families=tuple(
                SemanticFamilyCoverageRow(
                    module_name=family.module_name,
                    semantic_family=family.family_name,
                    family_coverage=family.family_coverage.value,
                    corpus_coverage=family.corpus_coverage.value,
                    category=(
                        family.category.value if family.category is not None else ""
                    ),
                    dimensionality=(
                        family.dimensionality.name
                        if family.dimensionality is not None
                        else ""
                    ),
                    respects_masks=family.respects_masks,
                    family_supported_modules=";".join(
                        family.family_supported_modules
                    ),
                    family_absorbed_modules=";".join(family.family_absorbed_modules),
                )
                for family in report.semantic_families
            ),
        )

    def write_to(self, output_root: Path) -> None:
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / MODULE_COVERAGE_SUMMARY_JSON).write_text(
            json.dumps(asdict(self.summary), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        for artifact in self.csv_artifacts():
            artifact.write_to(output_root)

    def csv_artifacts(self) -> tuple[CsvRowsArtifact, ...]:
        return (
            CsvRowsArtifact(
                filename=MODULE_COVERAGE_CPPIPE_MODULES_CSV,
                fieldnames=(
                    MODULE_NAME_FIELD,
                    CORPUS_COVERAGE_FIELD,
                    ABSORPTION_COVERAGE_FIELD,
                    CPPIPE_CASE_NAMES_FIELD,
                ),
                rows=self.cppipe_modules,
            ),
            CsvRowsArtifact(
                filename=MODULE_COVERAGE_ABSORBED_MODULES_CSV,
                fieldnames=(
                    MODULE_NAME_FIELD,
                    CORPUS_COVERAGE_FIELD,
                    IMPORTABLE_FIELD,
                    PROCESSING_CONTRACT_FIELD,
                    PROCESSING_CONTRACT_SOURCE_FIELD,
                    ARTIFACT_CONTRACT_COVERAGE_FIELD,
                ),
                rows=self.absorbed_modules,
            ),
            CsvRowsArtifact(
                filename=MODULE_COVERAGE_CPPIPE_SETTINGS_CSV,
                fieldnames=(
                    CASE_NAME_FIELD,
                    MODULE_NAME_FIELD,
                    "canonical_module_name",
                    "module_num",
                    "setting_name",
                    "normalized_setting_name",
                    "coverage",
                    "value",
                ),
                rows=self.cppipe_settings,
            ),
            CsvRowsArtifact(
                filename=MODULE_COVERAGE_SOURCE_MODULES_CSV,
                fieldnames=(MODULE_NAME_FIELD, SOURCE_COVERAGE_FIELD),
                rows=self.source_modules,
            ),
            CsvRowsArtifact(
                filename=MODULE_COVERAGE_SEMANTIC_FAMILIES_CSV,
                fieldnames=(
                    MODULE_NAME_FIELD,
                    SEMANTIC_FAMILY_FIELD,
                    FAMILY_COVERAGE_FIELD,
                    CORPUS_COVERAGE_FIELD,
                    CATEGORY_FIELD,
                    DIMENSIONALITY_FIELD,
                    RESPECTS_MASKS_FIELD,
                    FAMILY_SUPPORTED_MODULES_FIELD,
                    FAMILY_ABSORBED_MODULES_FIELD,
                ),
                rows=self.semantic_families,
            ),
        )


@dataclass(frozen=True, slots=True)
class ComparisonMetricPolicy:
    """Metric collection policy for parity-vs-speed benchmark runs."""

    collect_memory: bool = True

    def collectors(self) -> list[MetricCollector]:
        collectors: list[MetricCollector] = [TimeMetric()]
        if self.collect_memory:
            collectors.append(MemoryMetric())
        return collectors


@dataclass(frozen=True, slots=True)
class ComparisonSuiteRunContext:
    """Shared execution/provenance context for one comparison suite run."""

    suite_id: str
    speedup_target: float
    reuse_openhcs_cache: bool
    native_reference_root: Path | None
    require_native_reference: bool
    discard_openhcs_outputs: bool
    continue_on_error: bool
    openhcs_axis_filter: tuple[str, ...]
    openhcs_max_axis_count: int | None
    openhcs_num_workers: int
    openhcs_start_method: str
    openhcs_use_threading: bool
    metric_policy: ComparisonMetricPolicy

    def validate(self) -> None:
        if self.speedup_target <= 0:
            raise ValueError("speedup_target must be positive.")
        if self.openhcs_max_axis_count is not None and self.openhcs_max_axis_count <= 0:
            raise ValueError("openhcs_max_axis_count must be positive.")
        if self.openhcs_num_workers <= 0:
            raise ValueError("openhcs_num_workers must be positive.")


@dataclass(frozen=True, slots=True)
class CellProfilerComparisonCase:
    """One native-CellProfiler versus OpenHCS benchmark case."""

    name: str
    dataset_path: Path
    cppipe_path: Path
    dataset_id: str | None = None
    microscope_type: str | None = None
    assay_category: str | None = None
    module_category: str | None = None
    value_only: bool = False
    equivalence_reference_output_dir: Path | None = None
    cellprofiler_timeout_seconds: float | None = None
    pipeline_params: Mapping[str, object] = field(default_factory=dict)

    @property
    def resolved_dataset_id(self) -> str:
        return self.dataset_id or self.dataset_path.name


@dataclass(frozen=True, slots=True)
class ToolExecutionSummary:
    """Execution and phase timing summary for one tool run."""

    tool: str
    success: bool
    output_path: str
    execution_seconds: float | None
    total_metric_seconds: float | None
    peak_memory_mb: float | None
    cached: bool
    error_message: str | None
    phase_seconds: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class NativeReferenceLocation:
    """Resolved native CellProfiler reference location for a benchmark case."""

    output_dir: Path | None
    reference_output_dir: Path | None


@dataclass(frozen=True, slots=True)
class NativeCellProfilerReferenceScope:
    """Native CellProfiler reference directory identity for one benchmark case."""

    case: CellProfilerComparisonCase
    native_reference_root: Path
    pipeline_params: Mapping[str, object]

    @property
    def output_dir(self) -> Path:
        reference_scope_parts = [self.case.resolved_dataset_id, self.case.name]
        effective_pipeline_params = {
            "dataset_id": self.case.resolved_dataset_id,
            "cppipe_path": str(self.case.cppipe_path),
            **dict(self.pipeline_params),
        }
        reference_scope_parts.extend(
            native_cellprofiler_reference_scope_slugs(
                dataset_path=self.case.dataset_path,
                pipeline_name=self.case.name,
                pipeline_params=effective_pipeline_params,
                output_dir=Path(self.native_reference_root).resolve()
                / _benchmark_path_slug(
                    "_".join([self.case.resolved_dataset_id, self.case.name])
                ),
            )
        )
        return Path(self.native_reference_root).resolve() / _benchmark_path_slug(
            "_".join(reference_scope_parts)
        )

    @property
    def expected_reference(self) -> Path:
        resolved_dataset_path = resolve_visible_source_path(self.case.dataset_path)
        return (
            self.output_dir
            / f"{resolved_dataset_path.name}_{self.case.name}_native_cellprofiler"
        )

    def resolve(self) -> NativeReferenceLocation:
        expected_reference = self.expected_reference
        if native_cellprofiler_reference_is_complete(expected_reference):
            return NativeReferenceLocation(
                output_dir=self.output_dir,
                reference_output_dir=expected_reference,
            )

        discovered_reference = self._unique_completed_reference()
        return NativeReferenceLocation(
            output_dir=self.output_dir,
            reference_output_dir=discovered_reference,
        )

    def _unique_completed_reference(self) -> Path | None:
        if not self.output_dir.is_dir():
            return None
        candidates = tuple(
            path
            for path in sorted(self.output_dir.iterdir())
            if path.is_dir()
            and path.name.endswith("_native_cellprofiler")
            and native_cellprofiler_reference_is_complete(path)
        )
        if len(candidates) > 1:
            raise RuntimeError(
                "Native CellProfiler reference scope is ambiguous for "
                f"{self.case.name!r}: {candidates!r}."
            )
        return candidates[0] if candidates else None


@dataclass(frozen=True, slots=True)
class CellProfilerComparisonObservation:
    """Serializable observation for one case/repetition."""

    suite_id: str
    case_name: str
    repetition: int
    dataset_id: str
    assay_category: str | None
    module_category: str | None
    cppipe_path: str
    equivalent: bool
    difference_count: int | None
    numeric_abs_tolerance: float
    numeric_rel_tolerance: float
    native_cellprofiler: ToolExecutionSummary
    openhcs: ToolExecutionSummary
    observed_at_epoch_seconds: float = field(default_factory=time.time)

    @property
    def speedup(self) -> float | None:
        native_seconds = self.native_cellprofiler.execution_seconds
        openhcs_seconds = self.openhcs.execution_seconds
        if native_seconds is None or openhcs_seconds is None or openhcs_seconds <= 0:
            return None
        return native_seconds / openhcs_seconds

    @property
    def total_phase_speedup(self) -> float | None:
        native_seconds = self.native_cellprofiler.total_metric_seconds
        openhcs_seconds = self.openhcs.total_metric_seconds
        if native_seconds is None or openhcs_seconds is None or openhcs_seconds <= 0:
            return None
        return native_seconds / openhcs_seconds

    @property
    def parity_accuracy(self) -> float:
        return 1.0 if self.equivalent else 0.0

    def as_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["speedup"] = self.speedup
        payload["total_phase_speedup"] = self.total_phase_speedup
        payload["parity_accuracy"] = self.parity_accuracy
        return payload


@dataclass(frozen=True, slots=True)
class CachedNativeReferenceTimingPolicy:
    """Timing contract for reused native references with timeout-backed evidence."""

    case: CellProfilerComparisonCase
    summary: ToolExecutionSummary

    @property
    def has_timeout_lower_bound(self) -> bool:
        return (
            self.summary.success
            and self.summary.cached
            and self.summary.execution_seconds is None
            and self.case.cellprofiler_timeout_seconds is not None
        )

    def apply(self) -> ToolExecutionSummary:
        if not self.has_timeout_lower_bound:
            return self.summary
        timeout_seconds = float(self.case.cellprofiler_timeout_seconds)
        return ToolExecutionSummary(
            tool=self.summary.tool,
            success=self.summary.success,
            output_path=self.summary.output_path,
            execution_seconds=timeout_seconds,
            total_metric_seconds=(
                self.summary.total_metric_seconds
                if self.summary.total_metric_seconds is not None
                else timeout_seconds
            ),
            peak_memory_mb=self.summary.peak_memory_mb,
            cached=self.summary.cached,
            error_message=self.summary.error_message,
            phase_seconds=self.summary.phase_seconds,
        )


def load_comparison_cases(path: Path) -> tuple[CellProfilerComparisonCase, ...]:
    """Load benchmark cases from a JSON manifest."""
    manifest = ComparisonManifest.load(path)
    payload = manifest.payload
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, Sequence):
        raise ValueError("Benchmark manifest must contain a 'cases' sequence.")
    default_pipeline_params = payload.get("default_pipeline_params", {})
    if not isinstance(default_pipeline_params, Mapping):
        raise ValueError("Benchmark manifest default_pipeline_params must be an object.")
    cases: list[CellProfilerComparisonCase] = []
    for raw_case in raw_cases:
        if not isinstance(raw_case, Mapping):
            raise ValueError(f"Benchmark case must be an object: {raw_case!r}")
        raw_pipeline_params = raw_case.get("pipeline_params", {})
        if not isinstance(raw_pipeline_params, Mapping):
            raise ValueError(
                f"Benchmark case pipeline_params must be an object: {raw_pipeline_params!r}"
            )
        cases.append(
            CellProfilerComparisonCase(
                name=str(raw_case["name"]),
                dataset_path=manifest.path_resolver.resolve(raw_case, "dataset_path"),
                cppipe_path=manifest.path_resolver.resolve(raw_case, "cppipe_path"),
                dataset_id=(
                    str(raw_case["dataset_id"])
                    if raw_case.get("dataset_id") is not None
                    else None
                ),
                microscope_type=(
                    str(raw_case["microscope_type"])
                    if raw_case.get("microscope_type") is not None
                    else None
                ),
                assay_category=(
                    str(raw_case["assay_category"])
                    if raw_case.get("assay_category") is not None
                    else None
                ),
                module_category=(
                    str(raw_case["module_category"])
                    if raw_case.get("module_category") is not None
                    else None
                ),
                value_only=bool(raw_case.get("value_only", False)),
                equivalence_reference_output_dir=(
                    Path(str(raw_case["equivalence_reference_output_dir"]))
                    if raw_case.get("equivalence_reference_output_dir") is not None
                    else None
                ),
                cellprofiler_timeout_seconds=(
                    float(raw_case["cellprofiler_timeout_seconds"])
                    if raw_case.get("cellprofiler_timeout_seconds") is not None
                    else None
                ),
                pipeline_params={
                    **dict(default_pipeline_params),
                    **dict(raw_pipeline_params),
                },
            )
        )
    return tuple(cases)


def run_comparison_suite(
    cases: Iterable[CellProfilerComparisonCase],
    *,
    output_root: Path,
    suite_id: str,
    repeats: int = 1,
    reuse_openhcs_cache: bool = True,
    speedup_target: float = DEFAULT_SPEEDUP_TARGET,
    native_reference_root: Path | None = None,
    require_native_reference: bool = False,
    discard_openhcs_outputs: bool = False,
    continue_on_error: bool = False,
    openhcs_axis_filter: Sequence[str] = (),
    openhcs_max_axis_count: int | None = None,
    openhcs_num_workers: int = 1,
    openhcs_start_method: str = "fork",
    openhcs_use_threading: bool = False,
    metric_policy: ComparisonMetricPolicy = ComparisonMetricPolicy(),
    coverage_manifest_path: Path | None = None,
) -> tuple[CellProfilerComparisonObservation, ...]:
    """Run all cases and write raw benchmark observations."""
    if repeats < 1:
        raise ValueError("repeats must be at least 1.")
    context = ComparisonSuiteRunContext(
        suite_id=suite_id,
        speedup_target=speedup_target,
        reuse_openhcs_cache=reuse_openhcs_cache,
        native_reference_root=native_reference_root,
        require_native_reference=require_native_reference,
        discard_openhcs_outputs=discard_openhcs_outputs,
        continue_on_error=continue_on_error,
        openhcs_axis_filter=tuple(openhcs_axis_filter),
        openhcs_max_axis_count=openhcs_max_axis_count,
        openhcs_num_workers=openhcs_num_workers,
        openhcs_start_method=openhcs_start_method,
        openhcs_use_threading=openhcs_use_threading,
        metric_policy=metric_policy,
    )
    context.validate()
    output_root.mkdir(parents=True, exist_ok=True)
    observations: list[CellProfilerComparisonObservation] = []
    for repetition in range(1, repeats + 1):
        for case in cases:
            try:
                result = _run_comparison_case(
                    case,
                    output_root=output_root,
                    repetition=repetition,
                    context=context,
                )
            except Exception as exc:
                if not context.continue_on_error:
                    raise
                result = _failed_comparison_observation(
                    case,
                    suite_id=context.suite_id,
                    repetition=repetition,
                    error=exc,
                )
            observations.append(result)
            append_observations_jsonl(
                output_root / "observations.jsonl",
                (result,),
            )
            write_observations_csv(output_root / "observations.csv", observations)
            write_phase_timing_csv(output_root / "phase_timing.csv", observations)
            write_summary_csv(
                output_root / "summary.csv",
                observations,
                speedup_target=context.speedup_target,
            )
            write_suite_metadata(
                output_root / "suite_metadata.json",
                context=context,
            )
    if coverage_manifest_path is not None:
        write_module_coverage_artifacts(
            output_root,
            manifest_path=coverage_manifest_path,
        )
    return tuple(observations)


def load_observations_jsonl(
    path: Path,
) -> tuple[dict[str, Any], ...]:
    """Load raw observation payloads from a JSONL file."""
    observations: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            observations.append(json.loads(stripped))
    return tuple(observations)


def append_observations_jsonl(
    path: Path,
    observations: Iterable[CellProfilerComparisonObservation],
) -> None:
    """Append raw observations as JSON lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for observation in observations:
            handle.write(json.dumps(observation.as_payload(), sort_keys=True) + "\n")


def write_observations_csv(
    path: Path,
    observations: Sequence[CellProfilerComparisonObservation],
) -> None:
    """Write one-row-per-observation benchmark results."""
    ObservationCsvArtifact(path, _OBSERVATION_TABLE, observations).write()


def write_phase_timing_csv(
    path: Path,
    observations: Sequence[CellProfilerComparisonObservation],
) -> None:
    """Write long-form phase timing rows for all observations."""
    ObservationCsvArtifact(path, _PHASE_TIMING_TABLE, observations).write()


def write_summary_csv(
    path: Path,
    observations: Sequence[CellProfilerComparisonObservation],
    *,
    speedup_target: float = DEFAULT_SPEEDUP_TARGET,
) -> None:
    """Write per-case aggregate medians for plotting."""
    ObservationCsvArtifact(path, _summary_table(speedup_target), observations).write()


def write_module_coverage_artifacts(
    output_root: Path,
    *,
    manifest_path: Path,
) -> None:
    """Write CellProfiler module coverage artifacts for one benchmark manifest."""
    from benchmark.converter.compatibility_matrix import (
        build_cellprofiler_compatibility_report_for_manifest,
    )

    report = build_cellprofiler_compatibility_report_for_manifest(manifest_path)
    ModuleCoverageArtifacts.from_report(report, manifest_path=manifest_path).write_to(
        output_root
    )


def _observation_csv_rows(
    observations: Sequence[CellProfilerComparisonObservation],
) -> Iterable[CsvRow]:
    for observation in observations:
        yield _observation_csv_row(observation)


def _phase_timing_csv_rows(
    observations: Sequence[CellProfilerComparisonObservation],
) -> Iterable[CsvRow]:
    for observation in observations:
        for tool_summary in (
            observation.native_cellprofiler,
            observation.openhcs,
        ):
            for phase, seconds in tool_summary.phase_seconds.items():
                yield {
                    SUITE_ID_FIELD: observation.suite_id,
                    CASE_NAME_FIELD: observation.case_name,
                    REPETITION_FIELD: observation.repetition,
                    TOOL_FIELD: tool_summary.tool,
                    PHASE_FIELD: phase,
                    SECONDS_FIELD: seconds,
                }


def _summary_csv_rows(
    observations: Sequence[CellProfilerComparisonObservation],
    *,
    speedup_target: float,
) -> Iterable[CsvRow]:
    grouped: dict[str, list[CellProfilerComparisonObservation]] = defaultdict(list)
    for observation in observations:
        grouped[observation.case_name].append(observation)
    for case_name in sorted(grouped):
        case_observations = grouped[case_name]
        median_speedup = _median_present(
            observation.speedup for observation in case_observations
        )
        median_total_phase_speedup = _median_present(
            observation.total_phase_speedup for observation in case_observations
        )
        meets_execution_speedup_target = (
            median_speedup is not None and median_speedup >= speedup_target
        )
        meets_total_phase_speedup_target = (
            median_total_phase_speedup is not None
            and median_total_phase_speedup >= speedup_target
        )
        yield {
            CASE_NAME_FIELD: case_name,
            ASSAY_CATEGORY_FIELD: _common_value(
                observation.assay_category for observation in case_observations
            ),
            MODULE_CATEGORY_FIELD: _common_value(
                observation.module_category for observation in case_observations
            ),
            N_FIELD: len(case_observations),
            EQUIVALENT_COUNT_FIELD: sum(
                1 for observation in case_observations if observation.equivalent
            ),
            MEDIAN_NATIVE_EXECUTION_SECONDS_FIELD: _median_present(
                observation.native_cellprofiler.execution_seconds
                for observation in case_observations
            ),
            MEDIAN_OPENHCS_EXECUTION_SECONDS_FIELD: _median_present(
                observation.openhcs.execution_seconds
                for observation in case_observations
            ),
            MEDIAN_NATIVE_TOTAL_PHASE_SECONDS_FIELD: _median_present(
                observation.native_cellprofiler.total_metric_seconds
                for observation in case_observations
            ),
            MEDIAN_OPENHCS_TOTAL_PHASE_SECONDS_FIELD: _median_present(
                observation.openhcs.total_metric_seconds
                for observation in case_observations
            ),
            MEDIAN_NATIVE_PEAK_MEMORY_MB_FIELD: _median_present(
                observation.native_cellprofiler.peak_memory_mb
                for observation in case_observations
            ),
            MEDIAN_OPENHCS_PEAK_MEMORY_MB_FIELD: _median_present(
                observation.openhcs.peak_memory_mb
                for observation in case_observations
            ),
            MEDIAN_SPEEDUP_FIELD: median_speedup,
            MEDIAN_TOTAL_PHASE_SPEEDUP_FIELD: median_total_phase_speedup,
            SPEEDUP_TARGET_FIELD: speedup_target,
            MEETS_EXECUTION_SPEEDUP_TARGET_FIELD: meets_execution_speedup_target,
            MEETS_TOTAL_PHASE_SPEEDUP_TARGET_FIELD: meets_total_phase_speedup_target,
            MEETS_SPEEDUP_TARGET_FIELD: (
                meets_execution_speedup_target and meets_total_phase_speedup_target
            ),
            MIN_PARITY_ACCURACY_FIELD: min(
                observation.parity_accuracy for observation in case_observations
            ),
        }


_OBSERVATION_TABLE = CsvTableSpec(
    (
        SUITE_ID_FIELD,
        CASE_NAME_FIELD,
        REPETITION_FIELD,
        DATASET_ID_FIELD,
        ASSAY_CATEGORY_FIELD,
        MODULE_CATEGORY_FIELD,
        EQUIVALENT_FIELD,
        DIFFERENCE_COUNT_FIELD,
        NUMERIC_ABS_TOLERANCE_FIELD,
        NUMERIC_REL_TOLERANCE_FIELD,
        NATIVE_EXECUTION_SECONDS_FIELD,
        OPENHCS_EXECUTION_SECONDS_FIELD,
        NATIVE_TOTAL_PHASE_SECONDS_FIELD,
        OPENHCS_TOTAL_PHASE_SECONDS_FIELD,
        NATIVE_PEAK_MEMORY_MB_FIELD,
        OPENHCS_PEAK_MEMORY_MB_FIELD,
        SPEEDUP_FIELD,
        TOTAL_PHASE_SPEEDUP_FIELD,
        PARITY_ACCURACY_FIELD,
        NATIVE_CACHED_FIELD,
        OPENHCS_CACHED_FIELD,
        NATIVE_ERROR_MESSAGE_FIELD,
        OPENHCS_ERROR_MESSAGE_FIELD,
        NATIVE_OUTPUT_PATH_FIELD,
        OPENHCS_OUTPUT_PATH_FIELD,
    ),
    _observation_csv_rows,
)
_PHASE_TIMING_TABLE = CsvTableSpec(
    (
        SUITE_ID_FIELD,
        CASE_NAME_FIELD,
        REPETITION_FIELD,
        TOOL_FIELD,
        PHASE_FIELD,
        SECONDS_FIELD,
    ),
    _phase_timing_csv_rows,
)
def _summary_table(speedup_target: float) -> CsvTableSpec:
    return CsvTableSpec(
        (
            CASE_NAME_FIELD,
            ASSAY_CATEGORY_FIELD,
            MODULE_CATEGORY_FIELD,
            N_FIELD,
            EQUIVALENT_COUNT_FIELD,
            MEDIAN_NATIVE_EXECUTION_SECONDS_FIELD,
            MEDIAN_OPENHCS_EXECUTION_SECONDS_FIELD,
            MEDIAN_NATIVE_TOTAL_PHASE_SECONDS_FIELD,
            MEDIAN_OPENHCS_TOTAL_PHASE_SECONDS_FIELD,
            MEDIAN_NATIVE_PEAK_MEMORY_MB_FIELD,
            MEDIAN_OPENHCS_PEAK_MEMORY_MB_FIELD,
            MEDIAN_SPEEDUP_FIELD,
            MEDIAN_TOTAL_PHASE_SPEEDUP_FIELD,
            SPEEDUP_TARGET_FIELD,
            MEETS_EXECUTION_SPEEDUP_TARGET_FIELD,
            MEETS_TOTAL_PHASE_SPEEDUP_TARGET_FIELD,
            MEETS_SPEEDUP_TARGET_FIELD,
            MIN_PARITY_ACCURACY_FIELD,
        ),
        lambda observations: _summary_csv_rows(
            observations,
            speedup_target=speedup_target,
        ),
    )


def write_suite_metadata(
    path: Path,
    *,
    context: ComparisonSuiteRunContext,
) -> None:
    """Write reproducibility metadata for the benchmark suite."""
    payload = {
        "suite_id": context.suite_id,
        "speedup_target": context.speedup_target,
        "created_at_epoch_seconds": time.time(),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "native_reference_root": (
            str(context.native_reference_root)
            if context.native_reference_root is not None
            else None
        ),
        "require_native_reference": context.require_native_reference,
        "discard_openhcs_outputs": context.discard_openhcs_outputs,
        "continue_on_error": context.continue_on_error,
        "collect_memory_metric": context.metric_policy.collect_memory,
        OPENHCS_AXIS_FILTER_PARAM: context.openhcs_axis_filter,
        OPENHCS_MAX_AXIS_COUNT_PARAM: context.openhcs_max_axis_count,
        OPENHCS_NUM_WORKERS_PARAM: context.openhcs_num_workers,
        OPENHCS_START_METHOD_PARAM: context.openhcs_start_method,
        OPENHCS_USE_THREADING_PARAM: context.openhcs_use_threading,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _run_comparison_case(
    case: CellProfilerComparisonCase,
    *,
    output_root: Path,
    repetition: int,
    context: ComparisonSuiteRunContext,
) -> CellProfilerComparisonObservation:
    pipeline_params: dict[str, object] = {
        **case.pipeline_params,
        "dataset_id": case.resolved_dataset_id,
        "cppipe_path": str(case.cppipe_path),
        "compare_image_outputs": not case.value_only,
        "raise_on_equivalence_failure": False,
        "cache_candidate_measurement_snapshot": not context.discard_openhcs_outputs,
    }
    if case.value_only:
        pipeline_params.setdefault("materialize_runtime_artifacts", False)
    if case.cellprofiler_timeout_seconds is not None:
        pipeline_params["cellprofiler_timeout_seconds"] = (
            case.cellprofiler_timeout_seconds
        )
    if context.openhcs_axis_filter:
        pipeline_params[OPENHCS_AXIS_FILTER_PARAM] = context.openhcs_axis_filter
    if context.openhcs_max_axis_count is not None:
        pipeline_params[OPENHCS_MAX_AXIS_COUNT_PARAM] = context.openhcs_max_axis_count
    pipeline_params[OPENHCS_NUM_WORKERS_PARAM] = context.openhcs_num_workers
    pipeline_params[OPENHCS_START_METHOD_PARAM] = context.openhcs_start_method
    pipeline_params[OPENHCS_USE_THREADING_PARAM] = context.openhcs_use_threading
    native_reference = _native_reference_location(
        case,
        context.native_reference_root,
        pipeline_params,
    )
    if (
        context.require_native_reference
        and context.native_reference_root is not None
        and case.equivalence_reference_output_dir is None
        and native_reference.reference_output_dir is None
    ):
        expected_reference = None
        if native_reference.output_dir is not None:
            resolved_dataset_path = resolve_visible_source_path(case.dataset_path)
            expected_reference = (
                native_reference.output_dir
                / f"{resolved_dataset_path.name}_{case.name}_native_cellprofiler"
            )
        raise FileNotFoundError(
            "Required cached native CellProfiler reference is missing or incomplete"
            f" for case {case.name!r}: {expected_reference}"
        )
    result = run_cellprofiler_cppipe_parity(
        case.dataset_path,
        case.cppipe_path,
        metrics=context.metric_policy.collectors(),
        dataset_id=case.dataset_id,
        pipeline_name=case.name,
        microscope_type=case.microscope_type,
        pipeline_params=pipeline_params,
        output_root=output_root / "tool_outputs",
        equivalence_reference_output_dir=native_reference.reference_output_dir,
        native_cellprofiler_output_dir=native_reference.output_dir,
        reuse_openhcs_cache=context.reuse_openhcs_cache,
    )
    observation = comparison_observation_from_result(
        result,
        case=case,
        suite_id=context.suite_id,
        repetition=repetition,
    )
    if context.discard_openhcs_outputs:
        _discard_successful_openhcs_benchmark_tree(
            observation,
            suite_output_root=output_root,
        )
    return observation


def _native_reference_location(
    case: CellProfilerComparisonCase,
    native_reference_root: Path | None,
    pipeline_params: Mapping[str, object] | None = None,
) -> NativeReferenceLocation:
    if case.equivalence_reference_output_dir is not None:
        return NativeReferenceLocation(
            output_dir=None,
            reference_output_dir=case.equivalence_reference_output_dir,
        )
    if native_reference_root is None:
        return NativeReferenceLocation(output_dir=None, reference_output_dir=None)
    effective_pipeline_params = pipeline_params or case.pipeline_params
    return NativeCellProfilerReferenceScope(
        case=case,
        native_reference_root=Path(native_reference_root),
        pipeline_params=effective_pipeline_params,
    ).resolve()


def _failed_comparison_observation(
    case: CellProfilerComparisonCase,
    *,
    suite_id: str,
    repetition: int,
    error: Exception,
) -> CellProfilerComparisonObservation:
    error_traceback = _exception_traceback_message(error)
    return CellProfilerComparisonObservation(
        suite_id=suite_id,
        case_name=case.name,
        repetition=repetition,
        dataset_id=case.resolved_dataset_id,
        assay_category=case.assay_category,
        module_category=case.module_category,
        cppipe_path=str(case.cppipe_path),
        equivalent=False,
        difference_count=None,
        numeric_abs_tolerance=1e-6,
        numeric_rel_tolerance=1e-6,
        native_cellprofiler=ToolExecutionSummary(
            "CellProfiler",
            False,
            "",
            None,
            None,
            None,
            False,
            error_traceback,
            {},
        ),
        openhcs=ToolExecutionSummary(
            "OpenHCS",
            False,
            "",
            None,
            None,
            None,
            False,
            "skipped after benchmark case failure",
            {},
        ),
    )


def _exception_traceback_message(error: BaseException) -> str:
    """Return the full traceback for a benchmark case failure."""

    return "".join(
        traceback.format_exception(type(error), error, error.__traceback__)
    ).rstrip()


def comparison_observation_from_result(
    result: CellProfilerCompatibilityResult,
    *,
    case: CellProfilerComparisonCase,
    suite_id: str,
    repetition: int,
) -> CellProfilerComparisonObservation:
    """Convert adapter results into a stable observation payload."""
    openhcs_provenance = result.openhcs_converted.provenance or {}
    native_summary = CachedNativeReferenceTimingPolicy(
        case=case,
        summary=_tool_execution_summary(
            result.native_cellprofiler,
            execution_phase="EXECUTE_NATIVE_CP",
        ),
    ).apply()
    return CellProfilerComparisonObservation(
        suite_id=suite_id,
        case_name=case.name,
        repetition=repetition,
        dataset_id=case.resolved_dataset_id,
        assay_category=case.assay_category,
        module_category=case.module_category,
        cppipe_path=str(case.cppipe_path),
        equivalent=result.is_equivalent,
        difference_count=_difference_count(result),
        numeric_abs_tolerance=1e-6,
        numeric_rel_tolerance=1e-6,
        native_cellprofiler=native_summary,
        openhcs=_tool_execution_summary(
            result.openhcs_converted,
            execution_phase="EXECUTE_OPENHCS",
            cached=bool(
                openhcs_provenance.get("reused_cached_output")
                or openhcs_provenance.get("reused_runtime_execution_cache")
            ),
        ),
    )


def _tool_execution_summary(
    result: BenchmarkResult,
    *,
    execution_phase: str,
    cached: bool | None = None,
) -> ToolExecutionSummary:
    phase_seconds = _phase_seconds(result)
    metric_seconds = result.metrics.get("execution_time_seconds")
    peak_memory_mb = result.metrics.get("peak_memory_mb")
    total_phase_seconds = sum(phase_seconds.values()) if phase_seconds else None
    return ToolExecutionSummary(
        tool=result.tool_name,
        success=result.success,
        output_path=str(result.output_path),
        execution_seconds=phase_seconds.get(execution_phase),
        total_metric_seconds=total_phase_seconds
        if total_phase_seconds is not None
        else (float(metric_seconds) if metric_seconds is not None else None),
        peak_memory_mb=(
            float(peak_memory_mb) if peak_memory_mb is not None else None
        ),
        cached=bool(cached) if cached is not None else _result_is_cached(result),
        error_message=result.error_message,
        phase_seconds=phase_seconds,
    )


def _phase_seconds(result: BenchmarkResult) -> dict[str, float]:
    phase_totals: dict[str, float] = defaultdict(float)
    provenance = result.provenance or {}
    for raw_record in provenance.get("phase_timing_records", ()):
        if not isinstance(raw_record, Mapping):
            continue
        phase = raw_record.get("phase")
        seconds = raw_record.get("seconds")
        if phase is None or seconds is None:
            continue
        phase_totals[str(phase)] += float(seconds)
    return dict(phase_totals)


def _difference_count(result: CellProfilerCompatibilityResult) -> int | None:
    provenance = result.openhcs_converted.provenance or {}
    value = provenance.get("equivalence_difference_count")
    return int(value) if value is not None else None


def _result_is_cached(result: BenchmarkResult) -> bool:
    provenance = result.provenance or {}
    return bool(
        provenance.get("reused_reference_output")
        or provenance.get("reused_cached_output")
        or provenance.get("reused_runtime_execution_cache")
    )


def _observation_csv_row(
    observation: CellProfilerComparisonObservation,
) -> dict[str, object]:
    return {
        SUITE_ID_FIELD: observation.suite_id,
        CASE_NAME_FIELD: observation.case_name,
        REPETITION_FIELD: observation.repetition,
        DATASET_ID_FIELD: observation.dataset_id,
        ASSAY_CATEGORY_FIELD: observation.assay_category,
        MODULE_CATEGORY_FIELD: observation.module_category,
        EQUIVALENT_FIELD: observation.equivalent,
        DIFFERENCE_COUNT_FIELD: observation.difference_count,
        NUMERIC_ABS_TOLERANCE_FIELD: observation.numeric_abs_tolerance,
        NUMERIC_REL_TOLERANCE_FIELD: observation.numeric_rel_tolerance,
        NATIVE_EXECUTION_SECONDS_FIELD: (
            observation.native_cellprofiler.execution_seconds
        ),
        OPENHCS_EXECUTION_SECONDS_FIELD: observation.openhcs.execution_seconds,
        NATIVE_TOTAL_PHASE_SECONDS_FIELD: (
            observation.native_cellprofiler.total_metric_seconds
        ),
        OPENHCS_TOTAL_PHASE_SECONDS_FIELD: observation.openhcs.total_metric_seconds,
        NATIVE_PEAK_MEMORY_MB_FIELD: observation.native_cellprofiler.peak_memory_mb,
        OPENHCS_PEAK_MEMORY_MB_FIELD: observation.openhcs.peak_memory_mb,
        SPEEDUP_FIELD: observation.speedup,
        TOTAL_PHASE_SPEEDUP_FIELD: observation.total_phase_speedup,
        PARITY_ACCURACY_FIELD: observation.parity_accuracy,
        NATIVE_CACHED_FIELD: observation.native_cellprofiler.cached,
        OPENHCS_CACHED_FIELD: observation.openhcs.cached,
        NATIVE_ERROR_MESSAGE_FIELD: observation.native_cellprofiler.error_message,
        OPENHCS_ERROR_MESSAGE_FIELD: observation.openhcs.error_message,
        NATIVE_OUTPUT_PATH_FIELD: observation.native_cellprofiler.output_path,
        OPENHCS_OUTPUT_PATH_FIELD: observation.openhcs.output_path,
    }


def _median_present(values: Iterable[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return statistics.median(present)


def _common_value(values: Iterable[str | None]) -> str | None:
    present = {value for value in values if value}
    if not present:
        return None
    if len(present) > 1:
        return "Mixed"
    return next(iter(present))


def _benchmark_path_slug(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def _discard_openhcs_benchmark_tree(
    output_path: Path,
    *,
    suite_output_root: Path,
) -> None:
    """Delete one OpenHCS output tree only when benchmark ownership is proven."""
    target = _marked_openhcs_output_tree(Path(output_path), suite_output_root)
    suite_root = Path(suite_output_root).resolve()
    if not target.exists():
        return
    if not target.is_dir():
        raise NotADirectoryError(f"OpenHCS discard target is not a directory: {target}")
    if target == Path(".").resolve() or target == suite_root or target.parent == target:
        raise ToolExecutionError(f"Refusing unsafe OpenHCS discard target: {target}")
    try:
        target.relative_to(suite_root)
    except ValueError as exc:
        raise ToolExecutionError(
            "Refusing OpenHCS discard target outside suite output root: "
            f"{target} not under {suite_root}"
        ) from exc
    shutil.rmtree(target)


def _discard_successful_openhcs_benchmark_tree(
    observation: CellProfilerComparisonObservation,
    *,
    suite_output_root: Path,
) -> None:
    """Delete successful OpenHCS outputs while preserving failed debug artifacts."""
    if not observation.openhcs.success:
        return
    _discard_openhcs_benchmark_tree(
        Path(observation.openhcs.output_path),
        suite_output_root=suite_output_root,
    )


def _marked_openhcs_output_tree(output_path: Path, suite_output_root: Path) -> Path:
    """Find the benchmark-owned OpenHCS tree containing an output path."""
    suite_root = Path(suite_output_root).resolve()
    start = Path(output_path).resolve()
    candidates = (start, *start.parents)
    for candidate in candidates:
        if candidate == suite_root:
            break
        try:
            candidate.relative_to(suite_root)
        except ValueError:
            break
        if (candidate / OPENHCS_BENCHMARK_CACHE_MARKER).is_file():
            return candidate
    raise ToolExecutionError(
        "Refusing to discard OpenHCS output because no benchmark cache marker "
        f"was found between {start} and {suite_root}."
    )
