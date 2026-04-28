"""OpenHCS tool adapter."""

from __future__ import annotations

import logging
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen

import numpy as np
from skimage import filters, morphology, measure
from tqdm import tqdm

from benchmark.converter.runtime_pipeline import (
    execute_pipeline_direct,
    prepare_generated_pipeline,
)
from benchmark.datasets.registry import get_dataset_spec
from benchmark.contracts.tool_adapter import (
    BenchmarkResult,
    ToolAdapter,
    ToolExecutionError,
    ToolNotInstalledError,
)
from benchmark.contracts.metric import MetricCollector
from openhcs.constants.constants import Microscope

logger = logging.getLogger(__name__)


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
    def cppipe_path(self) -> Path | None:
        cppipe_value = self.pipeline_params.get("cppipe_path") or self.pipeline_params.get(
            "cppipe_file"
        )
        if not cppipe_value:
            return None
        return Path(cppipe_value)

    @property
    def cppipe_reference_url(self) -> str | None:
        value = self.pipeline_params.get("cppipe_reference_url")
        if value is None:
            return None
        return str(value)

    @property
    def cppipe_reference_index(self) -> int | None:
        value = self.pipeline_params.get("cppipe_reference_index")
        if value is None:
            return None
        return int(value)


class OpenHCSAdapter(ToolAdapter):
    """OpenHCS tool adapter."""

    name = "OpenHCS"

    def __init__(self):
        import openhcs

        self.version = openhcs.__version__

    def validate_installation(self) -> None:
        """Check OpenHCS is importable."""
        try:
            import openhcs  # noqa: F401
        except ImportError as exc:
            raise ToolNotInstalledError(f"OpenHCS not installed: {exc}") from exc

    def _prepare_filemanager(self):
        """Initialize FileManager and microscope handler."""
        from openhcs.io.filemanager import FileManager
        from openhcs.io.base import storage_registry, ensure_storage_registry

        ensure_storage_registry()
        return FileManager(storage_registry)

    def _load_microscope(self, filemanager, dataset_path: Path, microscope_type: str):
        """Create microscope handler for dataset."""
        from openhcs.microscopes import create_microscope_handler

        return create_microscope_handler(
            microscope_type=microscope_type or "auto",
            plate_folder=dataset_path,
            filemanager=filemanager,
            allowed_auto_types=[microscope_type] if microscope_type else None,
        )

    def _run_minimal_pipeline(self, image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
        """Blur → threshold → label segmentation pipeline."""
        method = params.get("threshold_method")
        if method not in (None, "Otsu"):
            raise ToolExecutionError(f"Unsupported threshold_method '{method}'")

        scope = params.get("threshold_scope")
        if scope not in (None, "Global"):
            raise ToolExecutionError(f"Unsupported threshold_scope '{scope}'")

        declump = params.get("declump_method")
        if declump not in (None, "Shape"):
            raise ToolExecutionError(f"Unsupported declump_method '{declump}'")

        diameter_range = params.get("diameter_range")
        if diameter_range is not None and (
            not isinstance(diameter_range, tuple)
            or len(diameter_range) != 2
            or not all(isinstance(x, (int, float)) for x in diameter_range)
        ):
            raise ToolExecutionError("diameter_range must be a (min, max) tuple")

        # Convert to float for processing while preserving dynamic range
        if image.dtype != np.float32:
            image = image.astype(np.float32)

        # Gaussian blur
        blurred = filters.gaussian(image, sigma=1)

        # Threshold
        threshold_value = filters.threshold_otsu(blurred)
        mask = blurred > threshold_value

        # Optional morphological opening to denoise
        radius = params.get("opening_radius", 0)
        if radius and radius > 0:
            selem = morphology.disk(radius)
            mask = morphology.opening(mask, selem)

        # Fill small holes if requested
        if params.get("fill_holes", False):
            mask = morphology.remove_small_holes(mask)

        labels = measure.label(mask)

        # Apply size filtering derived from diameter_range if provided
        if diameter_range:
            min_d, max_d = diameter_range
            min_area = np.pi * (min_d / 2) ** 2
            max_area = np.pi * (max_d / 2) ** 2
            props = measure.regionprops(labels)
            remove_ids = [
                prop.label
                for prop in props
                if prop.area < min_area or prop.area > max_area
            ]
            if remove_ids:
                mask = np.isin(labels, remove_ids, invert=True)
                labels = measure.label(mask)
        return labels.astype(np.uint16)

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

        reference_url = request.cppipe_reference_url
        if reference_url is None and request.cppipe_reference_index is not None:
            reference_url = self._dataset_reference_cppipe_url(
                request.dataset_id,
                request.cppipe_reference_index,
            )
        cppipe_path = self._resolve_cppipe_path(request)

        output_suffix = f"_{request.pipeline_name}_converted_cppipe"
        output_plate_root = request.output_dir / f"{request.dataset_path.name}{output_suffix}"
        generated_module_path = request.output_dir / f"{cppipe_path.stem}_openhcs.py"
        try:
            prepared = prepare_generated_pipeline(
                cppipe_path,
                output_path=generated_module_path,
            )
        except ValueError as exc:
            raise ToolExecutionError(
                f"Failed to prepare converted .cppipe pipeline {cppipe_path.name}: "
                f"{exc}"
            ) from exc

        global_config = GlobalPipelineConfig(
            num_workers=1,
            use_threading=True,
            materialization_results_path=output_plate_root / "results",
            microscope=self._configured_microscope(request.microscope_type),
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
            request.dataset_path,
            pipeline_config=pipeline_config,
        )
        orchestrator.initialize()

        with ExitStack() as stack:
            for metric in request.metrics:
                stack.enter_context(metric)
            execution = execute_pipeline_direct(orchestrator, prepared.pipeline)

        metric_results: dict[str, Any] = {
            metric.name: metric.get_result() for metric in request.metrics
        }
        output_plate_root.mkdir(parents=True, exist_ok=True)

        provenance = {
            "openhcs_version": self.version,
            "microscope_type": request.microscope_type,
            "pipeline_source": "converted_cppipe",
            "cppipe_path": str(cppipe_path),
            "generated_pipeline_module": prepared.module_name,
            "axis_count": len(execution.execution_results),
        }
        if reference_url is not None:
            provenance["cppipe_reference_url"] = reference_url

        return BenchmarkResult(
            tool_name=self.name,
            dataset_id=request.dataset_id,
            pipeline_name=request.pipeline_name,
            metrics=metric_results,
            output_path=output_plate_root,
            success=True,
            error_message=None,
            provenance=provenance,
        )

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

    def _resolve_cppipe_path(self, request: OpenHCSRunRequest) -> Path:
        """Resolve either a local or dataset-owned canonical .cppipe path."""
        cppipe_path = request.cppipe_path
        if cppipe_path is not None:
            if not cppipe_path.exists():
                raise ToolExecutionError(f".cppipe file not found: {cppipe_path}")
            return cppipe_path

        reference_url = request.cppipe_reference_url
        if reference_url is None and request.cppipe_reference_index is not None:
            reference_url = self._dataset_reference_cppipe_url(
                request.dataset_id,
                request.cppipe_reference_index,
            )
        if reference_url is None:
            raise ToolExecutionError(
                "Converted pipeline execution requires cppipe_path, cppipe_file, "
                "cppipe_reference_url, or cppipe_reference_index."
            )

        return self._materialize_cppipe_reference(
            reference_url,
            request.output_dir / "cppipe_references",
        )

    def _dataset_reference_cppipe_url(
        self,
        dataset_id: str,
        reference_index: int,
    ) -> str:
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

    def _materialize_cppipe_reference(
        self,
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
        microscope_type = request.microscope_type
        if microscope_type in (None, "auto"):
            raise ToolExecutionError(
                "microscope_type must be explicit (e.g., 'bbbc021'); auto-detect is not allowed."
            )

        if (
            request.cppipe_path is not None
            or request.cppipe_reference_url is not None
            or request.cppipe_reference_index is not None
        ):
            return self._run_converted_cppipe_pipeline(request)

        filemanager = self._prepare_filemanager()

        try:
            microscope_handler = self._load_microscope(
                filemanager,
                request.dataset_path,
                microscope_type,
            )
        except Exception as exc:
            raise ToolExecutionError(f"Failed to create microscope handler: {exc}") from exc

        # Enumerate image files via FileManager (leveraging OpenHCS discovery)
        try:
            from openhcs.constants.constants import Backend
            image_paths = filemanager.list_image_files(
                request.dataset_path,
                Backend.DISK.value,
                recursive=True,
            )
        except Exception as exc:
            raise ToolExecutionError(f"Failed to list dataset images: {exc}") from exc

        if not image_paths:
            raise ToolExecutionError(
                f"No image files found in dataset path: {request.dataset_path}"
            )

        with ExitStack() as stack:
            for metric in request.metrics:
                stack.enter_context(metric)

            for img_path in tqdm(image_paths, desc="OpenHCS pipeline", leave=False):
                image = filemanager.load(img_path, "disk", content_type="image")
                labels = self._run_minimal_pipeline(image, request.pipeline_params)

                output_path = request.output_dir / f"{Path(img_path).stem}_labels.tif"
                filemanager.save(labels, output_path, "disk")

        # Collect metrics after contexts have exited
        metric_results: dict[str, Any] = {
            metric.name: metric.get_result() for metric in request.metrics
        }

        return BenchmarkResult(
            tool_name=self.name,
            dataset_id=request.dataset_id,
            pipeline_name=request.pipeline_name,
            metrics=metric_results,
            output_path=request.output_dir,
            success=True,
            error_message=None,
            provenance={
                "openhcs_version": self.version,
                "microscope_type": microscope_handler.microscope_type,
                "image_count": len(image_paths),
            },
        )

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
