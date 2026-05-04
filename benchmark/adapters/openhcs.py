"""OpenHCS tool adapter."""

from __future__ import annotations

import json
import logging
import pickle
import hashlib
import importlib.util
import time
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from benchmark.adapters.cppipe_source import (
    CPPipeSourceRequest,
    CPPipeSourceResolution,
    materialize_cppipe_reference,
    resolve_cppipe_source,
)
from benchmark.converter.runtime_pipeline import (
    execute_pipeline_direct,
    prepare_generated_pipeline,
)
from benchmark.converter.execution_validation import (
    CPPipeExecutionValidation,
    CPPipeExecutionValidationError,
    validate_cppipe_execution,
)
from openhcs.interop.cellprofiler.measurement_dialect import (
    cellprofiler_runtime_equivalence_policy,
)
from benchmark.contracts.tool_adapter import (
    BenchmarkResult,
    ToolAdapter,
    ToolExecutionError,
    ToolNotInstalledError,
)
from benchmark.contracts.metric import MetricCollector
from openhcs.constants.constants import Microscope
from openhcs.core.artifacts import ArtifactKind
from openhcs.core.runtime_equivalence import (
    RuntimeEquivalencePolicy,
    RuntimeMeasurementSnapshot,
    RuntimeOutputSnapshot,
    image_paths,
    runtime_artifact_measurement_source_names,
    runtime_measurement_projection_cache_identity,
    runtime_measurement_equivalence,
    runtime_reference_artifact_equivalence,
    table_paths,
)
from openhcs.core.runtime_execution_validation import runtime_output_roots
from openhcs.core.runtime_exports import RuntimeExportObservation
from openhcs.core.source_schema_workspace import materialize_source_schema_workspace

logger = logging.getLogger(__name__)


_RUNTIME_EXECUTION_CACHE_SCHEMA_VERSION = 1
_RUNTIME_EXECUTION_OBSERVATION_PICKLE_NAME = "runtime_execution_observation.pkl"
_RUNTIME_EXECUTION_NON_IMAGE_OBSERVATION_PICKLE_NAME = (
    "runtime_execution_non_image_observation.pkl"
)
_RUNTIME_MEASUREMENT_SNAPSHOT_CACHE_SCHEMA_VERSION = 2
_RUNTIME_MEASUREMENT_SNAPSHOT_CACHE_DIR = ".openhcs_measurement_snapshot_cache"
_RUNTIME_REFERENCE_MEASUREMENT_SNAPSHOT_PREFIX = "runtime_reference_measurement_snapshot"
_RUNTIME_CANDIDATE_MEASUREMENT_SNAPSHOT_PREFIX = "runtime_candidate_measurement_snapshot"
_RUNTIME_EXECUTION_CACHE_IGNORED_PARAM_KEYS = frozenset(
    {
        "equivalence_reference_output_dir",
        "runtime_execution_cache_manifest",
        "runtime_execution_cache_key",
        "reuse_runtime_execution_cache",
    }
)
_RUNTIME_EXECUTION_CACHE_HELPER_KEYS = frozenset({"legacy_source_tree"})


_MICROSCOPES_BY_NORMALIZED_LITERAL = {
    member.value.lower(): member for member in Microscope
}


@dataclass(frozen=True, slots=True)
class OpenHCSRunRequest:
    """Authoritative benchmark run request for one OpenHCS execution."""

    dataset_path: Path
    pipeline_name: str
    pipeline_params: dict[str, Any]
    metrics: tuple[MetricCollector, ...]
    output_dir: Path

    @property
    def dataset_id(self) -> str:
        return str(self.pipeline_params.get("dataset_id", self.dataset_path.name))

    @property
    def microscope_type(self) -> str | None:
        value = self.pipeline_params.get("microscope_type")
        if value is None:
            return None
        return str(value)

    @property
    def cppipe_source(self) -> CPPipeSourceRequest:
        return CPPipeSourceRequest.from_pipeline_params(
            dataset_id=self.dataset_id,
            output_dir=self.output_dir,
            pipeline_params=self.pipeline_params,
        )

    @property
    def equivalence_reference_output_dir(self) -> Path | None:
        value = self.pipeline_params.get("equivalence_reference_output_dir")
        if value is None:
            return None
        return Path(value)

    @property
    def runtime_execution_cache_manifest(self) -> Path | None:
        value = self.pipeline_params.get("runtime_execution_cache_manifest")
        if value is None:
            return None
        return Path(value)

    @property
    def runtime_execution_cache_key(self) -> object | None:
        return self.pipeline_params.get("runtime_execution_cache_key")

    @property
    def reuse_runtime_execution_cache(self) -> bool:
        return bool(self.pipeline_params.get("reuse_runtime_execution_cache", True))

    @property
    def compare_image_outputs(self) -> bool:
        return bool(self.pipeline_params.get("compare_image_outputs", True))


@dataclass(frozen=True, slots=True)
class _RuntimeExecutionCacheHit:
    """Cached OpenHCS execution state, before external equivalence comparison."""

    validation: CPPipeExecutionValidation
    output_roots: tuple[Path, ...]
    execution_output_root: Path
    source_workspace_path: Path | None
    axis_count: int


def _runtime_execution_cache_key_matches(
    cached_key: object,
    expected_key: object,
) -> bool:
    """Return whether a runtime execution cache key is valid for this request."""
    if cached_key == expected_key:
        return True
    if not isinstance(cached_key, Mapping) or not isinstance(expected_key, Mapping):
        return False

    if "execution_source_tree" in cached_key:
        return _runtime_execution_cache_identity(cached_key) == (
            _runtime_execution_cache_identity(expected_key)
        )

    if "source_tree" not in cached_key:
        return False
    expected_legacy_source_tree = expected_key.get("legacy_source_tree")
    if expected_legacy_source_tree is None:
        expected_legacy_source_tree = expected_key.get("source_tree")
    if cached_key.get("source_tree") != expected_legacy_source_tree:
        return False

    return _legacy_runtime_execution_cache_identity(cached_key) == (
        _legacy_runtime_execution_cache_identity(expected_key)
    )


def _runtime_execution_cache_identity(cache_key: Mapping[str, Any]) -> dict[str, Any]:
    """Return the stable execution-cache identity for current cache keys."""
    return {
        key: cache_key[key]
        for key in cache_key
        if key not in _RUNTIME_EXECUTION_CACHE_HELPER_KEYS
    }


def _legacy_runtime_execution_cache_identity(
    cache_key: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare legacy broad cache keys using only execution-defining fields."""
    return {
        "schema_version": cache_key.get("schema_version"),
        "tool_name": cache_key.get("tool_name"),
        "tool_version": cache_key.get("tool_version"),
        "pipeline_name": cache_key.get("pipeline_name"),
        "pipeline_params": _runtime_execution_pipeline_params(
            cache_key.get("pipeline_params")
        ),
        "dataset_tree": cache_key.get("dataset_tree"),
        "cppipe_file": cache_key.get("cppipe_file"),
    }


def _runtime_execution_pipeline_params(value: object) -> dict[str, Any]:
    """Return pipeline params that can affect OpenHCS execution outputs."""
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): value[key]
        for key in value
        if str(key) not in _RUNTIME_EXECUTION_CACHE_IGNORED_PARAM_KEYS
    }


class OpenHCSAdapter(ToolAdapter):
    """OpenHCS tool adapter."""

    name = "OpenHCS"

    def __init__(self):
        import openhcs

        self.version = openhcs.__version__

    def validate_installation(self) -> None:
        """Check OpenHCS is importable."""
        if importlib.util.find_spec("openhcs") is None:
            raise ToolNotInstalledError("OpenHCS not installed")
        import openhcs  # noqa: F401

    def _run_converted_cppipe_pipeline(
        self,
        request: OpenHCSRunRequest,
    ) -> BenchmarkResult:
        """Execute a converted CellProfiler pipeline through the OpenHCS orchestrator."""
        from openhcs.config_framework.lazy_factory import ensure_global_config_context
        from openhcs.core.config import (
            GlobalPipelineConfig,
            LazyPathPlanningConfig,
            MaterializationBackend,
            PipelineConfig,
            VFSConfig,
        )
        from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator

        cppipe_source = self._resolve_cppipe_source(request)
        cppipe_path = cppipe_source.path
        reference_url = cppipe_source.reference_url

        output_suffix = f"_{request.pipeline_name}_converted_cppipe"
        output_plate_root = request.output_dir / f"{request.dataset_path.name}{output_suffix}"
        generated_module_path = request.output_dir / f"{cppipe_path.stem}_openhcs.py"
        try:
            prepared = prepare_generated_pipeline(
                cppipe_path,
                output_path=generated_module_path,
                prune_dead_unmaterialized_artifact_steps=(
                    not request.compare_image_outputs
                ),
            )
        except ValueError as exc:
            raise ToolExecutionError(
                f"Failed to prepare converted .cppipe pipeline {cppipe_path.name}: "
                f"{exc}"
            ) from exc
        source_workspace_path = (
            request.output_dir
            / f"{request.dataset_path.name}_{cppipe_path.stem}_source_workspace"
            if not prepared.source_schema.is_empty
            else None
        )

        cache_hit = self._load_runtime_execution_cache(request)
        if cache_hit is not None:
            validation = cache_hit.validation
            output_roots = cache_hit.output_roots
            execution_output_root = cache_hit.execution_output_root
            source_workspace_path = cache_hit.source_workspace_path
            axis_count = cache_hit.axis_count
            reused_runtime_execution_cache = True
        else:
            execution_plate_path = request.dataset_path
            execution_microscope = self._configured_microscope(request.microscope_type)
            if source_workspace_path is not None:
                try:
                    source_workspace = materialize_source_schema_workspace(
                        request.dataset_path,
                        source_workspace_path,
                        prepared.source_schema,
                    )
                except Exception as exc:
                    raise ToolExecutionError(
                        f"Failed to materialize CellProfiler source schema for "
                        f"{cppipe_path.name}: {exc}"
                    ) from exc
                execution_plate_path = source_workspace.workspace_root
                execution_microscope = Microscope.AUTO

            global_config = GlobalPipelineConfig(
                num_workers=1,
                use_threading=True,
                materialization_results_path=output_plate_root / "results",
                microscope=execution_microscope,
            )
            ensure_global_config_context(GlobalPipelineConfig, global_config)
            pipeline_config = PipelineConfig(
                path_planning_config=LazyPathPlanningConfig(
                    global_output_folder=request.output_dir,
                    output_dir_suffix=output_suffix,
                ),
                vfs_config=VFSConfig(
                    materialization_backend=MaterializationBackend.DISK,
                ),
            )
            orchestrator = PipelineOrchestrator(
                execution_plate_path,
                pipeline_config=pipeline_config,
            )
            orchestrator.initialize()

            with ExitStack() as stack:
                for metric in request.metrics:
                    stack.enter_context(metric)
                execution = execute_pipeline_direct(orchestrator, prepared.pipeline)
            output_roots = runtime_output_roots(
                execution.compiled_contexts,
                output_plate_root,
            )
            execution_output_root = (
                output_roots[0] if len(output_roots) == 1 else request.output_dir
            )
            try:
                validation = validate_cppipe_execution(
                    prepared,
                    execution,
                    execution_output_root,
                )
            except CPPipeExecutionValidationError as exc:
                raise ToolExecutionError(str(exc)) from exc
            axis_count = len(execution.execution_results)
            reused_runtime_execution_cache = False
            self._write_runtime_execution_cache(
                request,
                validation=validation,
                output_roots=output_roots,
                execution_output_root=execution_output_root,
                source_workspace_path=source_workspace_path,
                axis_count=axis_count,
            )
        equivalence_reference = request.equivalence_reference_output_dir
        equivalence_report = None
        if equivalence_reference is not None:
            if not equivalence_reference.exists():
                raise ToolExecutionError(
                    f"Equivalence reference output directory does not exist: "
                    f"{equivalence_reference}"
                )
            equivalence_policy = cellprofiler_runtime_equivalence_policy(
                numeric_abs_tolerance=1e-6,
                numeric_rel_tolerance=1e-6,
                allow_tie_sensitive_location_mismatches=True,
                allow_unstable_shape_descriptors=True,
                threshold_entropy_abs_tolerance=0.5,
                threshold_sensitive_pair_abs_tolerance=0.025,
                image_max_different_fraction=0.02,
            )
            if (
                not request.compare_image_outputs
                or _reference_has_no_images(equivalence_reference)
            ):
                equivalence_report = (
                    self._cached_table_only_reference_artifact_equivalence(
                        request,
                        equivalence_reference=equivalence_reference,
                        validation=validation,
                        policy=equivalence_policy,
                    )
                )
            else:
                equivalence_report = runtime_reference_artifact_equivalence(
                    RuntimeOutputSnapshot.from_output_root(equivalence_reference),
                    validation.observation,
                    policy=equivalence_policy,
                    candidate_image_artifact_names=(
                        validation.expectation.exports.image_artifact_names
                    ),
                    candidate_image_export_specs=(
                        validation.expectation.exports.image_export_specs
                    ),
                    candidate_image_snapshots=_candidate_image_snapshots_for_equivalence(
                        validation
                    ),
                )
            if not equivalence_report.is_equivalent:
                raise ToolExecutionError(
                    "Converted CellProfiler output did not match semantic "
                    f"reference output {equivalence_reference}:\n"
                    + "\n".join(
                        f"- {message}"
                        for message in equivalence_report.failure_messages()
                    )
                )

        metric_results = self._metric_results(request.metrics)
        output_plate_root.mkdir(parents=True, exist_ok=True)
        execution_output_root.mkdir(parents=True, exist_ok=True)

        provenance = {
            "openhcs_version": self.version,
            "microscope_type": request.microscope_type,
            "pipeline_source": "converted_cppipe",
            "cppipe_path": str(cppipe_path),
            "generated_pipeline_module": prepared.module_name,
            "axis_count": axis_count,
            "csv_output_count": len(validation.observation.exports.table_outputs),
            "image_output_count": len(validation.observation.exports.image_outputs),
            "compiled_output_roots": tuple(str(root) for root in output_roots),
            "reused_runtime_execution_cache": reused_runtime_execution_cache,
        }
        if request.runtime_execution_cache_manifest is not None:
            provenance["runtime_execution_cache_manifest"] = str(
                request.runtime_execution_cache_manifest
            )
        if equivalence_reference is not None:
            provenance["equivalence_reference_output_dir"] = str(equivalence_reference)
            provenance["equivalence_difference_count"] = len(
                equivalence_report.differences if equivalence_report else ()
            )
        if source_workspace_path is not None:
            provenance["source_workspace"] = str(source_workspace_path)
        if reference_url is not None:
            provenance["cppipe_reference_url"] = reference_url

        return BenchmarkResult(
            tool_name=self.name,
            dataset_id=request.dataset_id,
            pipeline_name=request.pipeline_name,
            metrics=metric_results,
            output_path=execution_output_root,
            success=True,
            error_message=None,
            provenance=provenance,
        )

    def _load_runtime_execution_cache(
        self,
        request: OpenHCSRunRequest,
    ) -> _RuntimeExecutionCacheHit | None:
        """Load a validated OpenHCS execution snapshot when cache identity matches."""
        manifest_path = request.runtime_execution_cache_manifest
        cache_key = request.runtime_execution_cache_key
        if (
            manifest_path is None
            or cache_key is None
            or not request.reuse_runtime_execution_cache
            or not manifest_path.exists()
        ):
            return None
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        if manifest.get("schema_version") != _RUNTIME_EXECUTION_CACHE_SCHEMA_VERSION:
            return None
        if not _runtime_execution_cache_key_matches(
            manifest.get("cache_key"),
            cache_key,
        ):
            return None
        prefer_non_image_payload = _reference_has_no_images(
            request.equivalence_reference_output_dir
        )
        non_image_validation_path = _cache_payload_path(
            manifest_path,
            manifest.get("non_image_validation_pickle_path"),
        )
        validation_path = (
            non_image_validation_path
            if (
                prefer_non_image_payload
                and non_image_validation_path is not None
                and non_image_validation_path.exists()
            )
            else _cache_payload_path(
                manifest_path,
                manifest.get("validation_pickle_path"),
            )
        )
        if validation_path is None or not validation_path.exists():
            return None
        output_roots = tuple(Path(path) for path in manifest.get("output_roots", ()))
        execution_output_root_value = manifest.get("execution_output_root")
        if not execution_output_root_value:
            return None
        execution_output_root = Path(str(execution_output_root_value))
        if not execution_output_root.exists():
            return None
        if any(not root.exists() for root in output_roots):
            return None
        try:
            with validation_path.open("rb") as handle:
                validation_payload = pickle.load(handle)
            validation = _validation_from_cache_payload(validation_payload)
            if (
                prefer_non_image_payload
                and (
                    non_image_validation_path is None
                    or validation_path != non_image_validation_path
                )
            ):
                self._write_runtime_execution_non_image_cache(
                    manifest_path,
                    validation=validation,
                )
        except Exception:
            logger.exception(
                "Failed to load OpenHCS runtime execution cache %s",
                validation_path,
            )
            return None
        source_workspace_value = manifest.get("source_workspace")
        return _RuntimeExecutionCacheHit(
            validation=validation,
            output_roots=output_roots,
            execution_output_root=execution_output_root,
            source_workspace_path=(
                Path(str(source_workspace_value)) if source_workspace_value else None
            ),
            axis_count=int(manifest.get("axis_count", 0)),
        )

    def _cached_table_only_reference_artifact_equivalence(
        self,
        request: OpenHCSRunRequest,
        *,
        equivalence_reference: Path,
        validation: CPPipeExecutionValidation,
        policy: RuntimeEquivalencePolicy,
    ):
        """Compare table-only references through cached semantic measurements."""
        started_at = time.perf_counter()
        known_source_names = runtime_artifact_measurement_source_names(
            validation.observation
        )
        cache_root = _measurement_snapshot_cache_root(request)
        reference_key = _reference_measurement_snapshot_cache_key(
            equivalence_reference,
            policy=policy,
            known_source_names=known_source_names,
        )
        reference_measurements = _load_or_create_measurement_snapshot(
            cache_root,
            prefix=_RUNTIME_REFERENCE_MEASUREMENT_SNAPSHOT_PREFIX,
            cache_key=reference_key,
            create=lambda: RuntimeMeasurementSnapshot.from_output_snapshot(
                RuntimeOutputSnapshot.from_output_root(equivalence_reference),
                policy=policy,
                known_source_names=known_source_names,
            ),
        )
        required_measurement_keys = frozenset(
            reference_measurements.values_by_feature
        )
        candidate_observation_fingerprint = (
            _runtime_measurement_observation_fingerprint(validation)
        )
        candidate_key = _candidate_measurement_snapshot_cache_key(
            request,
            policy=policy,
            known_source_names=known_source_names,
            required_measurement_keys=required_measurement_keys,
            candidate_observation_fingerprint=candidate_observation_fingerprint,
        )
        candidate_measurements = _load_or_create_measurement_snapshot(
            cache_root,
            prefix=_RUNTIME_CANDIDATE_MEASUREMENT_SNAPSHOT_PREFIX,
            cache_key=candidate_key,
            create=lambda: RuntimeMeasurementSnapshot.from_artifact_execution_observation(
                validation.observation,
                policy=policy,
                known_source_names=known_source_names,
                required_measurement_keys=required_measurement_keys,
            ),
        )
        if reference_measurements.is_empty and candidate_measurements.is_empty:
            logger.info(
                "Semantic measurement projection was empty; falling back to "
                "generic table comparison."
            )
            return runtime_reference_artifact_equivalence(
                _reference_snapshot_for_equivalence_fallback(
                    equivalence_reference,
                    compare_image_outputs=request.compare_image_outputs,
                ),
                validation.observation,
                policy=policy,
            )
        report = runtime_measurement_equivalence(
            reference_measurements,
            candidate_measurements,
            policy=policy,
        )
        logger.info(
            "Semantic table equivalence completed in %.3fs "
            "(reference_features=%d, candidate_features=%d, differences=%d).",
            time.perf_counter() - started_at,
            len(reference_measurements.values_by_feature),
            len(candidate_measurements.values_by_feature),
            len(report.differences),
        )
        return report

    def _write_runtime_execution_cache(
        self,
        request: OpenHCSRunRequest,
        *,
        validation: CPPipeExecutionValidation,
        output_roots: tuple[Path, ...],
        execution_output_root: Path,
        source_workspace_path: Path | None,
        axis_count: int,
    ) -> None:
        """Persist completed OpenHCS execution state before equivalence comparison."""
        manifest_path = request.runtime_execution_cache_manifest
        cache_key = request.runtime_execution_cache_key
        if manifest_path is None or cache_key is None:
            return
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        validation_path = (
            manifest_path.parent / _RUNTIME_EXECUTION_OBSERVATION_PICKLE_NAME
        )
        non_image_validation_path = (
            manifest_path.parent
            / _RUNTIME_EXECUTION_NON_IMAGE_OBSERVATION_PICKLE_NAME
        )
        with validation_path.open("wb") as handle:
            pickle.dump(
                _validation_cache_payload(validation),
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        with non_image_validation_path.open("wb") as handle:
            pickle.dump(
                _validation_cache_payload(validation, include_image_records=False),
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": _RUNTIME_EXECUTION_CACHE_SCHEMA_VERSION,
                    "cache_key": cache_key,
                    "validation_pickle_path": validation_path.name,
                    "non_image_validation_pickle_path": (
                        non_image_validation_path.name
                    ),
                    "output_roots": tuple(str(root) for root in output_roots),
                    "execution_output_root": str(execution_output_root),
                    "source_workspace": (
                        str(source_workspace_path)
                        if source_workspace_path is not None
                        else None
                    ),
                    "axis_count": axis_count,
                },
                indent=2,
                sort_keys=True,
            )
        )

    def _write_runtime_execution_non_image_cache(
        self,
        manifest_path: Path,
        *,
        validation: CPPipeExecutionValidation,
    ) -> None:
        """Backfill a compact runtime cache payload for table-only equivalence."""
        non_image_validation_path = (
            manifest_path.parent
            / _RUNTIME_EXECUTION_NON_IMAGE_OBSERVATION_PICKLE_NAME
        )
        with non_image_validation_path.open("wb") as handle:
            pickle.dump(
                _validation_cache_payload(validation, include_image_records=False),
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            return
        manifest["non_image_validation_pickle_path"] = non_image_validation_path.name
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    def _metric_results(
        self,
        metrics: tuple[MetricCollector, ...],
    ) -> dict[str, Any]:
        """Return metric results, skipping metrics unused by cached execution."""
        results: dict[str, Any] = {}
        for metric in metrics:
            try:
                results[metric.name] = metric.get_result()
            except RuntimeError:
                continue
        return results

    def _configured_microscope(
        self,
        microscope_type: str | None,
    ) -> Microscope:
        """Normalize benchmark microscope literals onto the OpenHCS enum SSOT."""
        if microscope_type is None:
            return Microscope.AUTO
        normalized = microscope_type.strip().lower()
        try:
            return _MICROSCOPES_BY_NORMALIZED_LITERAL[normalized]
        except KeyError as exc:
            raise ToolExecutionError(
                f"Unsupported OpenHCS microscope_type {microscope_type!r}."
            ) from exc

    def _resolve_cppipe_source(
        self,
        request: OpenHCSRunRequest,
    ) -> CPPipeSourceResolution:
        """Resolve .cppipe source metadata through the shared adapter helper."""
        return resolve_cppipe_source(
            request.cppipe_source,
            materialize_reference=self._materialize_cppipe_reference,
        )

    def _materialize_cppipe_reference(
        self,
        reference_url: str,
        target_dir: Path,
    ) -> Path:
        """Download one canonical .cppipe file into a stable local path."""
        return materialize_cppipe_reference(reference_url, target_dir)

    def run(
        self,
        dataset_path: Path,
        pipeline_name: str,
        pipeline_params: dict[str, Any],
        metrics: list[Any],
        output_dir: Path,
    ) -> BenchmarkResult:
        """Execute OpenHCS pipeline with metrics."""
        output_dir.mkdir(parents=True, exist_ok=True)

        request = OpenHCSRunRequest(
            dataset_path=dataset_path,
            pipeline_name=pipeline_name,
            pipeline_params=pipeline_params,
            metrics=self._validated_metric_collectors(metrics),
            output_dir=output_dir,
        )
        return self._run_converted_cppipe_pipeline(request)

    def _validated_metric_collectors(
        self,
        metrics: list[Any],
    ) -> tuple[MetricCollector, ...]:
        """Validate metric collectors once and return a typed immutable bundle."""
        validated_metrics: list[MetricCollector] = []
        for metric in metrics:
            if not isinstance(metric, MetricCollector):
                raise ToolExecutionError(
                    f"Metric {metric} does not extend MetricCollector"
                )
            validated_metrics.append(metric)
        return tuple(validated_metrics)


def _validation_cache_payload(
    validation: CPPipeExecutionValidation,
    *,
    include_image_records: bool = True,
) -> dict[str, Any]:
    """Return a pickle-safe payload for runtime execution validation."""
    exports = validation.observation.exports
    return {
        "expectation": validation.expectation,
        "records_by_axis": {
            axis: tuple(_cacheable_runtime_records(records, include_image_records))
            for axis, records in validation.observation.records_by_axis.items()
        },
        "exports": {
            "table_outputs": tuple(str(path) for path in exports.table_outputs),
            "image_outputs": tuple(str(path) for path in exports.image_outputs),
            "table_headers_by_path": {
                str(path): tuple(headers)
                for path, headers in exports.table_headers_by_path.items()
            },
            "table_row_counts_by_path": {
                str(path): int(row_count)
                for path, row_count in exports.table_row_counts_by_path.items()
            },
        },
    }


def _cacheable_runtime_records(
    records: tuple[Any, ...],
    include_image_records: bool,
) -> tuple[Any, ...]:
    """Return runtime records appropriate for the requested cache payload."""
    if include_image_records:
        return tuple(records)
    return tuple(record for record in records if record.key.kind is not ArtifactKind.IMAGE)


def _reference_has_no_images(reference_output_dir: Path | None) -> bool:
    """Return whether an external reference has no image outputs to compare."""
    if reference_output_dir is None:
        return False
    try:
        return not image_paths(reference_output_dir)
    except OSError:
        return False


def _candidate_image_snapshots_for_equivalence(
    validation: CPPipeExecutionValidation,
) -> tuple[Any, ...] | None:
    """Return candidate export snapshots when exports are the authoritative images.

    SaveImages declares exact runtime image artifacts and export encodings.  In
    that case equivalence must use the typed artifact records, not incidental
    final-step image files that OpenHCS may also materialize.
    """
    if validation.expectation.exports.image_export_specs:
        return None
    if not validation.observation.exports.image_outputs:
        return None
    return RuntimeOutputSnapshot.from_export_observation(
        validation.observation.exports
    ).images


def _reference_snapshot_for_equivalence_fallback(
    reference_output_dir: Path,
    *,
    compare_image_outputs: bool,
) -> RuntimeOutputSnapshot:
    """Build the reference snapshot for generic fallback equivalence."""
    snapshot = RuntimeOutputSnapshot.from_output_root(reference_output_dir)
    if compare_image_outputs:
        return snapshot
    return RuntimeOutputSnapshot(tables=snapshot.tables)


def _validation_from_cache_payload(
    payload: object,
) -> CPPipeExecutionValidation:
    """Rebuild runtime execution validation from a pickle-safe payload."""
    if not isinstance(payload, Mapping):
        raise TypeError(
            "OpenHCS runtime execution cache payload must be a mapping, "
            f"got {type(payload).__name__}."
        )
    exports_payload = payload.get("exports")
    if not isinstance(exports_payload, Mapping):
        raise TypeError("OpenHCS runtime execution cache exports are missing.")
    exports = RuntimeExportObservation(
        table_outputs=tuple(
            Path(path) for path in exports_payload.get("table_outputs", ())
        ),
        image_outputs=tuple(
            Path(path) for path in exports_payload.get("image_outputs", ())
        ),
        table_headers_by_path={
            Path(path): tuple(headers)
            for path, headers in (
                exports_payload.get("table_headers_by_path", {}) or {}
            ).items()
        },
        table_row_counts_by_path={
            Path(path): int(row_count)
            for path, row_count in (
                exports_payload.get("table_row_counts_by_path", {}) or {}
            ).items()
        },
    )
    from openhcs.core.runtime_execution_validation import (
        RuntimeArtifactExecutionObservation,
    )

    return CPPipeExecutionValidation(
        expectation=payload["expectation"],
        observation=RuntimeArtifactExecutionObservation(
            records_by_axis={
                str(axis): tuple(records)
                for axis, records in (payload.get("records_by_axis", {}) or {}).items()
            },
            exports=exports,
        ),
    )


def _cache_payload_path(
    manifest_path: Path,
    value: object,
) -> Path | None:
    if value is None:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    return manifest_path.parent / path


def _measurement_snapshot_cache_root(request: OpenHCSRunRequest) -> Path:
    """Return the directory that owns semantic equivalence snapshot caches."""
    manifest_path = request.runtime_execution_cache_manifest
    if manifest_path is not None:
        return manifest_path.parent / _RUNTIME_MEASUREMENT_SNAPSHOT_CACHE_DIR
    return request.output_dir / _RUNTIME_MEASUREMENT_SNAPSHOT_CACHE_DIR


def _load_or_create_measurement_snapshot(
    cache_root: Path,
    *,
    prefix: str,
    cache_key: object,
    create: Callable[[], RuntimeMeasurementSnapshot],
) -> RuntimeMeasurementSnapshot:
    """Load or populate a generic semantic measurement snapshot cache."""
    path = _measurement_snapshot_cache_path(cache_root, prefix, cache_key)
    snapshot = _load_measurement_snapshot_cache(path, cache_key)
    if snapshot is not None:
        logger.info("Loaded semantic measurement snapshot cache %s", path)
        return snapshot

    started_at = time.perf_counter()
    snapshot = create()
    _write_measurement_snapshot_cache(path, cache_key, snapshot)
    logger.info(
        "Wrote semantic measurement snapshot cache %s in %.3fs "
        "(features=%d).",
        path,
        time.perf_counter() - started_at,
        len(snapshot.values_by_feature),
    )
    return snapshot


def _load_measurement_snapshot_cache(
    path: Path,
    cache_key: object,
) -> RuntimeMeasurementSnapshot | None:
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        logger.exception("Failed to load semantic measurement snapshot cache %s", path)
        return None
    if not isinstance(payload, Mapping):
        return None
    if (
        payload.get("schema_version")
        != _RUNTIME_MEASUREMENT_SNAPSHOT_CACHE_SCHEMA_VERSION
    ):
        return None
    if payload.get("cache_key") != _cache_jsonable(cache_key):
        return None
    snapshot_payload = payload.get("snapshot")
    if snapshot_payload is None:
        return None
    return RuntimeMeasurementSnapshot.from_cache_payload(snapshot_payload)


def _write_measurement_snapshot_cache(
    path: Path,
    cache_key: object,
    snapshot: RuntimeMeasurementSnapshot,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(
            {
                "schema_version": _RUNTIME_MEASUREMENT_SNAPSHOT_CACHE_SCHEMA_VERSION,
                "cache_key": _cache_jsonable(cache_key),
                "snapshot": snapshot.to_cache_payload(),
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def _measurement_snapshot_cache_path(
    cache_root: Path,
    prefix: str,
    cache_key: object,
) -> Path:
    digest = _cache_key_digest(cache_key)
    return cache_root / f"{prefix}_{digest}.pkl"


def _reference_measurement_snapshot_cache_key(
    reference_output_dir: Path,
    *,
    policy: RuntimeEquivalencePolicy,
    known_source_names: tuple[str, ...],
) -> dict[str, object]:
    return {
        "schema_version": _RUNTIME_MEASUREMENT_SNAPSHOT_CACHE_SCHEMA_VERSION,
        "kind": "reference_output_measurements",
        "reference_tables": _table_output_fingerprint(reference_output_dir),
        "semantic_measurement_projection": (
            runtime_measurement_projection_cache_identity()
        ),
        "known_source_names": tuple(known_source_names),
        "policy": policy,
    }


def _candidate_measurement_snapshot_cache_key(
    request: OpenHCSRunRequest,
    *,
    policy: RuntimeEquivalencePolicy,
    known_source_names: tuple[str, ...],
    required_measurement_keys: frozenset[object],
    candidate_observation_fingerprint: str,
) -> dict[str, object]:
    return {
        "schema_version": _RUNTIME_MEASUREMENT_SNAPSHOT_CACHE_SCHEMA_VERSION,
        "kind": "artifact_execution_measurements",
        "runtime_execution_cache_key": _runtime_execution_cache_key_for_snapshot(
            request.runtime_execution_cache_key
        ),
        "runtime_measurement_observation": candidate_observation_fingerprint,
        "semantic_measurement_projection": (
            runtime_measurement_projection_cache_identity()
        ),
        "required_measurement_keys": tuple(
            key.to_cache_payload()
            for key in sorted(
                required_measurement_keys,
                key=lambda measurement_key: measurement_key.sort_key,
            )
        ),
        "known_source_names": tuple(known_source_names),
        "policy": policy,
    }


def _runtime_measurement_observation_fingerprint(
    validation: CPPipeExecutionValidation,
) -> str:
    """Fingerprint non-image runtime observations that feed measurement parity."""
    payload = _validation_cache_payload(validation, include_image_records=False)
    return hashlib.sha256(
        json.dumps(
            _cache_jsonable(payload),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _runtime_execution_cache_key_for_snapshot(cache_key: object) -> object:
    """Drop cache-helper fields that do not affect runtime measurement outputs."""
    if not isinstance(cache_key, Mapping):
        return cache_key
    return {
        key: value
        for key, value in cache_key.items()
        if key not in _RUNTIME_EXECUTION_CACHE_HELPER_KEYS
    }


def _table_output_fingerprint(output_dir: Path) -> tuple[dict[str, object], ...]:
    root = Path(output_dir)
    fingerprint: list[dict[str, object]] = []
    for path in table_paths(root):
        stat = path.stat()
        fingerprint.append(
            {
                "path": str(path.relative_to(root)),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return tuple(fingerprint)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_key_digest(cache_key: object) -> str:
    return hashlib.sha256(
        json.dumps(
            _cache_jsonable(cache_key),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:24]


def _cache_jsonable(value: object) -> object:
    """Return a deterministic JSON-compatible payload for cache identity."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {
            field.name: _cache_jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return tuple(
            (
                _cache_jsonable(key),
                _cache_jsonable(item_value),
            )
            for key, item_value in sorted(
                value.items(),
                key=lambda item: repr(_cache_jsonable(item[0])),
            )
        )
    if isinstance(value, (tuple, list)):
        return tuple(_cache_jsonable(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_cache_jsonable(item) for item in value), key=repr))
    return repr(value)
