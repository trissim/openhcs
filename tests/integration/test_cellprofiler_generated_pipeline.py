from __future__ import annotations

import csv
from dataclasses import replace
import os
from pathlib import Path
import re

from openhcs.interop.cellprofiler.runtime_pipeline import (
    DirectPipelineExecution,
    execute_pipeline_direct,
    prepare_generated_pipeline,
)
from openhcs.interop.cellprofiler.source_schema_ingestion import (
    CellProfilerSourceSchemaWorkspaceRequest,
    prepare_cellprofiler_source_schema_workspace,
)
from openhcs.interop.cellprofiler.execution_validation import validate_cppipe_execution
import numpy as np
import pytest
import tifffile
from openhcs.config_framework.lazy_factory import ensure_global_config_context
from openhcs.constants import Microscope
from openhcs.constants.constants import AllComponents
from openhcs.core.artifacts import (
    ArtifactType,
    ImageArtifactType,
    ObjectLabelsArtifactType,
    MeasurementsArtifactType,
    RelationshipsArtifactType,
    SpatialGridArtifactType,
)
from openhcs.core.config import (
    AnalysisConsolidationConfig,
    GlobalPipelineConfig,
    LazyWellFilterConfig,
    LazyPathPlanningConfig,
    MaterializationBackend,
    PipelineConfig,
    VFSConfig,
)
from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
from openhcs.core.runtime_artifact_queries import (
    RuntimeArtifactQueryContext,
    runtime_relationship,
)
from openhcs.core.runtime_values import ObjectRelationship, object_label_dense_array
from openhcs.core.source_bindings import (
    ComponentSelector,
    SourceBindingOrigin,
    SourceFilterMatchType,
    SourceFilterSubject,
)
from openhcs.core.source_schema_workspace import (
    SourceSchemaImageSetSelection,
    SourceSchemaWorkspaceMaterialization,
    materialize_source_schema_workspace,
)
from openhcs.tests.generators.generate_synthetic_data import (
    SyntheticMicroscopyGenerator,
)
from PIL import Image
from scipy.io import savemat


def _generated_pipeline_config(
    prepared,
    *,
    path_planning_config: LazyPathPlanningConfig,
    vfs_config: VFSConfig,
    well_filter_config: LazyWellFilterConfig | None = None,
) -> PipelineConfig:
    config = prepared.generated_pipeline.pipeline_config
    overrides = {
        "path_planning_config": path_planning_config,
        "vfs_config": vfs_config,
    }
    if well_filter_config is not None:
        overrides["well_filter_config"] = well_filter_config
    return replace(config, **overrides)


def test_cppipe_generated_pipeline_executes_through_orchestrator(
    tmp_path: Path,
) -> None:
    plate_path = _generate_plate(tmp_path / "plate")
    cppipe_path = _write_cppipe(tmp_path / "identify_primary_objects.cppipe")
    prepared = prepare_generated_pipeline(
        cppipe_path,
        output_path=tmp_path / "generated_cellprofiler_pipeline.py",
    )

    global_config = GlobalPipelineConfig(num_workers=1, use_threading=True)
    ensure_global_config_context(GlobalPipelineConfig, global_config)
    pipeline_config = _generated_pipeline_config(
        prepared,
        path_planning_config=LazyPathPlanningConfig(
            output_dir_suffix="_generated_cppipe",
        ),
        vfs_config=VFSConfig(
            materialization_backend=MaterializationBackend.DISK,
        ),
    )
    orchestrator = PipelineOrchestrator(plate_path, pipeline_config=pipeline_config)
    orchestrator.initialize()

    execution = execute_pipeline_direct(orchestrator, prepared.runtime_pipeline_steps)

    assert prepared.infrastructure_modules
    assert prepared.registered_functions
    assert all(
        result.is_success()
        for result in execution.execution_results.values()
    )

    nuclei_records = execution.compiled_contexts["A01"].runtime_value_store.find(
        name="Nuclei",
        artifact_type=ObjectLabelsArtifactType,
        axis_id="A01",
    )
    assert len(nuclei_records) == 1
    assert nuclei_records[0].value.data.max() > 0


def test_bbbc021_cppipe_generated_pipeline_executes_named_channel_bindings(
    tmp_path: Path,
) -> None:
    plate_path = _generate_bbbc021_plate(tmp_path / "Week1_22123")
    cppipe_path = _write_bbbc021_cppipe(tmp_path / "bbbc021_multichannel.cppipe")
    prepared = prepare_generated_pipeline(
        cppipe_path,
        output_path=tmp_path / "generated_bbbc021_cellprofiler_pipeline.py",
    )

    global_config = GlobalPipelineConfig(
        num_workers=1,
        use_threading=True,
        microscope=Microscope.BBBC021,
    )
    ensure_global_config_context(GlobalPipelineConfig, global_config)
    pipeline_config = _generated_pipeline_config(
        prepared,
        path_planning_config=LazyPathPlanningConfig(
            output_dir_suffix="_generated_cppipe",
        ),
        vfs_config=VFSConfig(
            materialization_backend=MaterializationBackend.DISK,
        ),
    )
    orchestrator = PipelineOrchestrator(plate_path, pipeline_config=pipeline_config)
    orchestrator.initialize()

    execution = execute_pipeline_direct(orchestrator, prepared.runtime_pipeline_steps)

    assert all(
        result.is_success()
        for result in execution.execution_results.values()
    )
    nuclei_records = execution.compiled_contexts["A01"].runtime_value_store.find(
        name="Nuclei",
        artifact_type=ObjectLabelsArtifactType,
        axis_id="A01",
    )
    composite_records = execution.compiled_contexts["A01"].runtime_value_store.find(
        name="Composite",
        artifact_type=ImageArtifactType,
        axis_id="A01",
    )
    assert len(nuclei_records) == 1
    assert nuclei_records[0].value.data.max() > 0
    assert len(composite_records) == 1


def test_bbbc021_canonical_illum_cppipe_executes_real_pipeline_shape(
    tmp_path: Path,
) -> None:
    plate_path = _generate_bbbc021_plate(tmp_path / "Week1_22123")
    _write_bbbc021_image(
        plate_path / "fields" / "A01_s1_w4F00DBABE-17A7-4AA1-9C50-123456789ABC.tif",
        seed=3,
        signal=2400,
    )
    cppipe_path = (
        Path(__file__).resolve().parents[2]
        / "benchmark"
        / "cellprofiler_pipelines"
        / "BBBC021_illum.cppipe"
    )
    prepared = prepare_generated_pipeline(
        cppipe_path,
        output_path=tmp_path / "generated_bbbc021_illum_pipeline.py",
    )

    global_config = GlobalPipelineConfig(
        num_workers=1,
        use_threading=True,
        microscope=Microscope.BBBC021,
    )
    ensure_global_config_context(GlobalPipelineConfig, global_config)
    pipeline_config = _generated_pipeline_config(
        prepared,
        path_planning_config=LazyPathPlanningConfig(
            output_dir_suffix="_generated_cppipe",
        ),
        vfs_config=VFSConfig(
            materialization_backend=MaterializationBackend.DISK,
        ),
    )
    orchestrator = PipelineOrchestrator(plate_path, pipeline_config=pipeline_config)
    orchestrator.initialize()

    execution = execute_pipeline_direct(orchestrator, prepared.runtime_pipeline_steps)

    assert all(
        result.is_success()
        for result in execution.execution_results.values()
    )
    generated_images = sorted(
        (_generated_output_root(plate_path) / "images").glob("*.tif")
    )
    assert [path.name for path in generated_images] == [
        "A01_s1_w1_z001_t001.tif",
        "A01_s1_w2_z001_t001.tif",
        "A01_s1_w4_z001_t001.tif",
    ]
    generated_artifacts = sorted(
        (_generated_output_root(plate_path) / "images_results").glob("*_slice_000.tif")
    )
    assert [path.name for path in generated_artifacts] == [
        "A01_s1_w1_z001_t001_IllumDAPI_step0_slice_000.tif",
        "A01_s1_w2_z001_t001_IllumActin_step0_slice_000.tif",
        "A01_s1_w4_z001_t001_IllumTubulin_step0_slice_000.tif",
    ]


def test_loadimages_cppipe_executes_pipeline_start_mat_illumination_binding(
    tmp_path: Path,
) -> None:
    plate_path = _generate_loadimages_mat_illum_plate(tmp_path / "mat_illum_plate")
    cppipe_path = _write_loadimages_mat_illum_cppipe(
        tmp_path / "loadimages_mat_illum.cppipe"
    )
    prepared = prepare_generated_pipeline(
        cppipe_path,
        output_path=tmp_path / "generated_loadimages_mat_illum_pipeline.py",
    )

    raw_assignment = prepared.source_schema.resolved_assignment_for_alias("Raw")
    illum_assignment = prepared.source_schema.resolved_source_artifact_for_alias(
        "Illum",
        ImageArtifactType,
    )
    assert raw_assignment is not None
    assert raw_assignment.origin is SourceBindingOrigin.PIPELINE_START
    assert illum_assignment is not None
    assert illum_assignment.origin is SourceBindingOrigin.PIPELINE_START

    global_config = GlobalPipelineConfig(
        num_workers=1,
        use_threading=True,
        microscope=Microscope.IMAGEXPRESS,
    )
    ensure_global_config_context(GlobalPipelineConfig, global_config)
    pipeline_config = _generated_pipeline_config(
        prepared,
        path_planning_config=LazyPathPlanningConfig(
            output_dir_suffix="_generated_cppipe",
        ),
        vfs_config=VFSConfig(
            materialization_backend=MaterializationBackend.DISK,
        ),
    )
    orchestrator = PipelineOrchestrator(plate_path, pipeline_config=pipeline_config)
    orchestrator.initialize()

    execution = execute_pipeline_direct(orchestrator, prepared.runtime_pipeline_steps)

    assert all(
        result.is_success()
        for result in execution.execution_results.values()
    )
    corrected_records = execution.compiled_contexts["A01"].runtime_value_store.find(
        name="CorrectedRaw",
        artifact_type=ImageArtifactType,
        axis_id="A01",
    )
    assert len(corrected_records) == 1
    assert np.asarray(corrected_records[0].value.data).shape[-2:] == (64, 64)
    assert sorted(
        path.name
        for path in (_generated_output_root(plate_path) / "images").glob("*.tif")
    ) == ["A01_s001_w1_z001_t001.tif"]


def test_examplefly_cppipe_generated_pipeline_executes_real_pipeline_shape(
    tmp_path: Path,
) -> None:
    plate_path = _generate_two_channel_plate(tmp_path / "examplefly_plate")
    cppipe_path = (
        Path(__file__).resolve().parents[2]
        / "benchmark"
        / "cellprofiler_pipelines"
        / "ExampleFly.cppipe"
    )
    prepared = prepare_generated_pipeline(
        cppipe_path,
        output_path=tmp_path / "generated_examplefly_cellprofiler_pipeline.py",
    )

    blue_assignment = prepared.source_schema.resolved_assignment_for_alias("OrigBlue")
    green_assignment = prepared.source_schema.resolved_assignment_for_alias("OrigGreen")
    assert blue_assignment is not None
    assert blue_assignment.selector.components == (
        ComponentSelector(AllComponents.CHANNEL, "1"),
    )
    assert green_assignment is not None
    assert green_assignment.selector.components == (
        ComponentSelector(AllComponents.CHANNEL, "2"),
    )
    assert any(
        module.name == "ExportToSpreadsheet"
        for module in prepared.infrastructure_modules
    )

    global_config = GlobalPipelineConfig(num_workers=1, use_threading=True)
    ensure_global_config_context(GlobalPipelineConfig, global_config)
    pipeline_config = _generated_pipeline_config(
        prepared,
        path_planning_config=LazyPathPlanningConfig(
            output_dir_suffix="_generated_cppipe",
        ),
        vfs_config=VFSConfig(
            materialization_backend=MaterializationBackend.DISK,
        ),
    )
    orchestrator = PipelineOrchestrator(plate_path, pipeline_config=pipeline_config)
    orchestrator.initialize()

    execution = execute_pipeline_direct(orchestrator, prepared.runtime_pipeline_steps)

    assert all(
        result.is_success()
        for result in execution.execution_results.values()
    )
    runtime_store = execution.compiled_contexts["A01"].runtime_value_store
    assert runtime_store.find(
        name="Cells",
        artifact_type=ObjectLabelsArtifactType,
        axis_id="A01",
    )
    assert runtime_store.find(
        name="Cytoplasm",
        artifact_type=ObjectLabelsArtifactType,
        axis_id="A01",
    )
    assert runtime_store.find(
        artifact_type=MeasurementsArtifactType,
        axis_id="A01",
    )
    csv_outputs = sorted(
        path
        for path in _generated_results_dir(plate_path).rglob("*.csv")
        if "summary" not in path.name.lower()
    )
    assert len(csv_outputs) >= 6
    assert all(path.stat().st_size > 0 for path in csv_outputs)
    headers_by_name = {path.name: _csv_header(path) for path in csv_outputs}
    assert _matching_header(
        headers_by_name,
        "MeasureObjectSizeShape",
    )[:4] == ["slice_index", "object_label", "area", "perimeter"]
    assert "contrast" in _matching_header(headers_by_name, "MeasureTexture")
    assert any(
        "correlation_manders" in column
        for column in _matching_header(headers_by_name, "MeasureColocalization")
    )
    assert all(
        "slice_index" in header
        for name, header in headers_by_name.items()
        if any(prefix in name for prefix in ("MeasureObjectSizeShape", "MeasureTexture"))
    )


def test_examplehuman_cppipe_executes_via_source_schema_workspace(
    tmp_path: Path,
) -> None:
    source_root = _generate_examplehuman_source_folder(tmp_path / "ExampleHuman")
    cppipe_path = (
        Path(__file__).resolve().parents[2]
        / "benchmark"
        / "cellprofiler_pipelines"
        / "ExampleHuman.cppipe"
    )
    prepared = prepare_generated_pipeline(
        cppipe_path,
        output_path=tmp_path / "generated_examplehuman_cellprofiler_pipeline.py",
    )
    workspace = materialize_source_schema_workspace(
        source_root,
        tmp_path / "examplehuman_openhcs_workspace",
        prepared.source_schema,
    )

    global_config = GlobalPipelineConfig(
        num_workers=1,
        use_threading=True,
        microscope=Microscope.AUTO,
    )
    ensure_global_config_context(GlobalPipelineConfig, global_config)
    pipeline_config = _generated_pipeline_config(
        prepared,
        path_planning_config=LazyPathPlanningConfig(
            output_dir_suffix="_generated_cppipe",
        ),
        vfs_config=VFSConfig(
            materialization_backend=MaterializationBackend.DISK,
        ),
    )
    orchestrator = PipelineOrchestrator(
        workspace.workspace_root,
        pipeline_config=pipeline_config,
    )
    orchestrator.initialize()

    execution = execute_pipeline_direct(orchestrator, prepared.runtime_pipeline_steps)

    assert all(
        result.is_success()
        for result in execution.execution_results.values()
    )
    axis_id = _single_execution_axis(execution)
    runtime_store = execution.compiled_contexts[axis_id].runtime_value_store
    cytoplasm_records = runtime_store.find(
        name="Cytoplasm",
        artifact_type=ObjectLabelsArtifactType,
        axis_id=axis_id,
    )
    measurement_records = runtime_store.find(
        name="MeasureObjectIntensity_10_measurements",
        artifact_type=MeasurementsArtifactType,
        axis_id=axis_id,
    )
    assert len(cytoplasm_records) == 1
    cytoplasm_labels = object_label_dense_array(cytoplasm_records[0].value.data)
    assert (
        cytoplasm_labels.ndim == 2
        or cytoplasm_labels.ndim == 3
        and cytoplasm_labels.shape[0] == 1
    )
    assert measurement_records


def test_official_example_untangleworms_cppipe_executes_via_source_schema_workspace(
    tmp_path: Path,
) -> None:
    examples_root = _official_cellprofiler_examples_root()
    cppipe_path = (
        examples_root
        / "CellProfiler3Pipelines"
        / "ExampleUntangleWorms.cppipe"
    )
    source_root = examples_root / "ExampleUntangleWorms"
    if not cppipe_path.exists() or not source_root.exists():
        pytest.skip(
            "Official CellProfiler ExampleUntangleWorms files are not available. "
            f"Set CELLPROFILER_EXAMPLES_ROOT to a local examples checkout; "
            f"looked under {examples_root}."
        )

    prepared = prepare_generated_pipeline(
        cppipe_path,
        output_path=tmp_path / "generated_official_untangleworms_pipeline.py",
    )
    workspace = materialize_source_schema_workspace(
        source_root,
        tmp_path / "official_untangleworms_openhcs_workspace",
        prepared.source_schema,
    )
    effective_well_filter = _effective_source_schema_well_filter(
        workspace,
        requested_well_filter=("A01",),
    )

    global_config = GlobalPipelineConfig(
        num_workers=1,
        use_threading=True,
        microscope=Microscope.AUTO,
    )
    ensure_global_config_context(GlobalPipelineConfig, global_config)
    pipeline_config = _generated_pipeline_config(
        prepared,
        path_planning_config=LazyPathPlanningConfig(
            output_dir_suffix="_generated_cppipe",
        ),
        vfs_config=VFSConfig(
            materialization_backend=MaterializationBackend.DISK,
        ),
        well_filter_config=LazyWellFilterConfig(
            well_filter=list(effective_well_filter),
        ),
    )
    orchestrator = PipelineOrchestrator(
        workspace.workspace_root,
        pipeline_config=pipeline_config,
    )
    orchestrator.initialize()

    execution = execute_pipeline_direct(
        orchestrator,
        prepared.runtime_pipeline_steps,
    )

    assert all(
        result.is_success()
        for result in execution.execution_results.values()
    )
    axis_id = _single_execution_axis(execution)
    runtime_store = execution.compiled_contexts[axis_id].runtime_value_store
    assert runtime_store.find(
        name="OverlappingWorms",
        artifact_type=ObjectLabelsArtifactType,
        axis_id=axis_id,
    )
    assert runtime_store.find(
        name="NonOverlappingWorms",
        artifact_type=ObjectLabelsArtifactType,
        axis_id=axis_id,
    )
    overlay_records = runtime_store.find(
        name="OrigOverlay",
        artifact_type=ImageArtifactType,
        axis_id=axis_id,
    )
    assert len(overlay_records) == 1
    overlay = np.asarray(overlay_records[0].value.data)
    assert overlay.ndim == 4
    assert overlay.shape[0] == 2
    assert overlay.shape[-1] == 3
    assert runtime_store.find(
        name="MeasureObjectIntensity_17_measurements",
        artifact_type=MeasurementsArtifactType,
        axis_id=axis_id,
    )


def test_official_examplefly_cppipe_executes_measurement_math_classification(
    tmp_path: Path,
) -> None:
    examples_root = _official_cellprofiler_examples_root()
    source_root = examples_root / "ExampleFly"
    cppipe_path = source_root / "ExampleFly.cppipe"
    if not cppipe_path.exists() or not source_root.exists():
        pytest.skip(
            "Official CellProfiler ExampleFly files are not available. "
            f"Set CELLPROFILER_EXAMPLES_ROOT to a local examples checkout; "
            f"looked under {examples_root}."
        )

    prepared = prepare_generated_pipeline(
        cppipe_path,
        output_path=tmp_path / "generated_official_examplefly_pipeline.py",
    )
    workspace = materialize_source_schema_workspace(
        source_root,
        tmp_path / "official_examplefly_openhcs_workspace",
        prepared.source_schema,
    )

    global_config = GlobalPipelineConfig(
        num_workers=1,
        use_threading=True,
        microscope=Microscope.AUTO,
    )
    ensure_global_config_context(GlobalPipelineConfig, global_config)
    pipeline_config = _generated_pipeline_config(
        prepared,
        path_planning_config=LazyPathPlanningConfig(
            output_dir_suffix="_generated_cppipe",
        ),
        vfs_config=VFSConfig(
            materialization_backend=MaterializationBackend.DISK,
        ),
        well_filter_config=LazyWellFilterConfig(
            well_filter=["A01"],
        ),
    )
    orchestrator = PipelineOrchestrator(
        workspace.workspace_root,
        pipeline_config=pipeline_config,
    )
    orchestrator.initialize()

    execution = execute_pipeline_direct(
        orchestrator,
        prepared.runtime_pipeline_steps,
    )

    assert all(
        result.is_success()
        for result in execution.execution_results.values()
    )
    runtime_store = execution.compiled_contexts["A01"].runtime_value_store
    assert runtime_store.find(
        name="CalculateMath_18_measurements",
        artifact_type=MeasurementsArtifactType,
        axis_id="A01",
    )
    assert runtime_store.find(
        name="ClassifyObjects_19_measurements",
        artifact_type=MeasurementsArtifactType,
        axis_id="A01",
    )
    assert runtime_store.find(
        name="RGBImage",
        artifact_type=ImageArtifactType,
        axis_id="A01",
    )


def test_official_examplefly_cppipe_executes_through_zmq_server(
    tmp_path: Path,
) -> None:
    from openhcs.runtime.zmq_execution_client import (
        OpenHCSExecutionSubmission,
        ZMQExecutionClient,
    )

    examples_root = _official_cellprofiler_examples_root()
    source_root = examples_root / "ExampleFly"
    cppipe_path = source_root / "ExampleFly.cppipe"
    if not cppipe_path.exists() or not source_root.exists():
        pytest.skip(
            "Official CellProfiler ExampleFly files are not available. "
            f"Set CELLPROFILER_EXAMPLES_ROOT to a local examples checkout; "
            f"looked under {examples_root}."
        )

    workspace = prepare_cellprofiler_source_schema_workspace(
        CellProfilerSourceSchemaWorkspaceRequest.from_paths(
            source_root=source_root,
            cppipe_path=cppipe_path,
            workspace_root=tmp_path / "official_examplefly_zmq_openhcs_workspace",
            generated_pipeline_path=(
                tmp_path / "generated_official_examplefly_zmq_pipeline.py"
            ),
            image_set_selection=SourceSchemaImageSetSelection(
                max_image_set_count=1,
            ),
        )
    )
    prepared = workspace.prepared_pipeline

    global_config = GlobalPipelineConfig(
        num_workers=1,
        use_threading=False,
        microscope=Microscope.AUTO,
        analysis_consolidation_config=AnalysisConsolidationConfig(enabled=False),
    )
    ensure_global_config_context(GlobalPipelineConfig, global_config)
    pipeline_config = _generated_pipeline_config(
        prepared,
        path_planning_config=LazyPathPlanningConfig(
            global_output_folder=str(tmp_path / "zmq_output"),
            output_dir_suffix="_generated_cppipe_zmq",
        ),
        vfs_config=VFSConfig(
            materialization_backend=MaterializationBackend.DISK,
        ),
    )

    client = ZMQExecutionClient(
        port=18000 + (os.getpid() % 20000),
        persistent=False,
    )
    try:
        client.connect(timeout=30)
        response = client.execute_pipeline(
            OpenHCSExecutionSubmission(
                plate_id=str(workspace.source_root),
                execution_plate_id=str(workspace.execution_plate_path),
                pipeline_steps=prepared.runtime_pipeline_steps,
                global_config=global_config,
                pipeline_config=pipeline_config,
                selected_pipeline_path=cppipe_path,
            )
        )
    finally:
        client.disconnect()

    assert response["status"] == "complete"
    assert response["results"]["well_count"] == 1
    assert response["results"]["wells"] == ["A01"]


def test_official_example_untangleworms_brightfield_cppipe_executes_overlay(
    tmp_path: Path,
) -> None:
    examples_root = _official_cellprofiler_examples_root()
    cppipe_path = (
        examples_root
        / "CellProfiler3Pipelines"
        / "ExampleUntangleWormsBrightField.cppipe"
    )
    source_root = examples_root / "ExampleUntangleWormsBrightField"
    if not cppipe_path.exists() or not source_root.exists():
        pytest.skip(
            "Official CellProfiler ExampleUntangleWormsBrightField files are not "
            f"available. Set CELLPROFILER_EXAMPLES_ROOT to a local examples "
            f"checkout; looked under {examples_root}."
        )

    prepared = prepare_generated_pipeline(
        cppipe_path,
        output_path=tmp_path / "generated_official_brightfield_pipeline.py",
    )
    workspace = materialize_source_schema_workspace(
        source_root,
        tmp_path / "official_brightfield_openhcs_workspace",
        prepared.source_schema,
    )

    global_config = GlobalPipelineConfig(
        num_workers=1,
        use_threading=True,
        microscope=Microscope.AUTO,
    )
    ensure_global_config_context(GlobalPipelineConfig, global_config)
    pipeline_config = _generated_pipeline_config(
        prepared,
        path_planning_config=LazyPathPlanningConfig(
            output_dir_suffix="_generated_cppipe",
        ),
        vfs_config=VFSConfig(
            materialization_backend=MaterializationBackend.DISK,
        ),
        well_filter_config=LazyWellFilterConfig(
            well_filter=["A01"],
        ),
    )
    orchestrator = PipelineOrchestrator(
        workspace.workspace_root,
        pipeline_config=pipeline_config,
    )
    orchestrator.initialize()

    execution = execute_pipeline_direct(
        orchestrator,
        prepared.runtime_pipeline_steps,
    )

    assert all(
        result.is_success()
        for result in execution.execution_results.values()
    )
    overlay_outputs = sorted(
        (_generated_output_root(workspace.workspace_root) / "images").glob("*.png")
    )
    assert [path.name for path in overlay_outputs] == [
        "A01_s001_w1_z001_t001.png",
        "A01_s002_w1_z001_t001.png",
    ]
    overlay = np.asarray(Image.open(overlay_outputs[0]))
    assert overlay.dtype == np.uint8
    assert overlay.ndim == 3
    red = overlay[..., 0].astype(np.int16)
    blue = overlay[..., 2].astype(np.int16)
    assert np.count_nonzero(blue > red + 32) > 0


def test_official_example_cometassay_cppipe_executes_mask_geometry(
    tmp_path: Path,
) -> None:
    examples_root = _official_cellprofiler_examples_root()
    cppipe_path = (
        examples_root
        / "CellProfiler3Pipelines"
        / "ExampleCometAssay.cppipe"
    )
    source_root = examples_root / "ExampleCometAssay"
    if not cppipe_path.exists() or not source_root.exists():
        pytest.skip(
            "Official CellProfiler ExampleCometAssay files are not available. "
            f"Set CELLPROFILER_EXAMPLES_ROOT to a local examples checkout; "
            f"looked under {examples_root}."
        )

    prepared = prepare_generated_pipeline(
        cppipe_path,
        output_path=tmp_path / "generated_official_comet_pipeline.py",
    )
    workspace = materialize_source_schema_workspace(
        source_root,
        tmp_path / "official_comet_openhcs_workspace",
        prepared.source_schema,
    )

    global_config = GlobalPipelineConfig(
        num_workers=1,
        use_threading=True,
        microscope=Microscope.AUTO,
    )
    ensure_global_config_context(GlobalPipelineConfig, global_config)
    pipeline_config = _generated_pipeline_config(
        prepared,
        path_planning_config=LazyPathPlanningConfig(
            output_dir_suffix="_generated_cppipe",
        ),
        vfs_config=VFSConfig(
            materialization_backend=MaterializationBackend.DISK,
        ),
        well_filter_config=LazyWellFilterConfig(
            well_filter=["A01"],
        ),
    )
    orchestrator = PipelineOrchestrator(
        workspace.workspace_root,
        pipeline_config=pipeline_config,
    )
    orchestrator.initialize()

    execution = execute_pipeline_direct(
        orchestrator,
        prepared.runtime_pipeline_steps,
    )

    assert all(
        result.is_success()
        for result in execution.execution_results.values()
    )
    image_outputs = sorted(
        (_generated_output_root(workspace.workspace_root) / "images").glob("*.tif")
    )
    assert [path.name for path in image_outputs] == [
        "A01_s001_w1_z001_t001.tif",
        "A01_s002_w1_z001_t001.tif",
    ]
    overlay = tifffile.imread(image_outputs[0])
    assert overlay.shape[:2] == (1040, 1388)
    assert overlay.ndim == 3


def test_official_example_colocalization_cppipe_executes_relationship_exports(
    tmp_path: Path,
) -> None:
    examples_root = _official_cellprofiler_examples_root()
    cppipe_path = (
        examples_root
        / "CellProfiler3Pipelines"
        / "ExampleColocalization.cppipe"
    )
    source_root = examples_root / "ExampleColocalization"
    if not cppipe_path.exists() or not source_root.exists():
        pytest.skip(
            "Official CellProfiler ExampleColocalization files are not available. "
            f"Set CELLPROFILER_EXAMPLES_ROOT to a local examples checkout; "
            f"looked under {examples_root}."
        )

    prepared = prepare_generated_pipeline(
        cppipe_path,
        output_path=tmp_path / "generated_official_colocalization_pipeline.py",
    )
    workspace = materialize_source_schema_workspace(
        source_root,
        tmp_path / "official_colocalization_openhcs_workspace",
        prepared.source_schema,
    )

    global_config = GlobalPipelineConfig(
        num_workers=1,
        use_threading=True,
        microscope=Microscope.AUTO,
    )
    ensure_global_config_context(GlobalPipelineConfig, global_config)
    pipeline_config = _generated_pipeline_config(
        prepared,
        path_planning_config=LazyPathPlanningConfig(
            output_dir_suffix="_generated_cppipe",
        ),
        vfs_config=VFSConfig(
            materialization_backend=MaterializationBackend.DISK,
        ),
        well_filter_config=LazyWellFilterConfig(
            well_filter=["A01"],
        ),
    )
    orchestrator = PipelineOrchestrator(
        workspace.workspace_root,
        pipeline_config=pipeline_config,
    )
    orchestrator.initialize()

    execution = execute_pipeline_direct(
        orchestrator,
        prepared.runtime_pipeline_steps,
    )

    assert all(
        result.is_success()
        for result in execution.execution_results.values()
    )
    validate_cppipe_execution(
        prepared,
        execution,
        _generated_output_root(workspace.workspace_root),
    )
    runtime_store = execution.compiled_contexts["A01"].runtime_value_store
    relationship_records = runtime_store.find(
        artifact_type=RelationshipsArtifactType,
        axis_id="A01",
    )
    assert {
        record.key.name
        for record in relationship_records
    } == {
        "Objects1_Objects2_relationships",
        "ExpandedObjects1_ExpandedObjects2_relationships",
        "Objects1_ColocalizedObjects_relationships",
        "Objects1_ColocalizedRegion_relationships",
    }
    relationships = tuple(
        runtime_relationship(
            RuntimeArtifactQueryContext(
                runtime_store,
                "A01",
                group_key=record.key.scope.group_key,
                match_group=True,
            ),
            record.key.name,
        )
        for record in relationship_records
    )
    assert {relationship.source.role for relationship in relationships} == {"parent"}
    assert {relationship.target.role for relationship in relationships} == {"child"}
    measurement_records = runtime_store.find(
        artifact_type=MeasurementsArtifactType,
        axis_id="A01",
    )
    measurement_names = {record.key.name for record in measurement_records}
    assert any(
        name.startswith("MeasureColocalization_")
        and name.endswith("_measurements")
        for name in measurement_names
    )
    assert any(
        name.startswith("CalculateMath_")
        and name.endswith("_measurements")
        for name in measurement_names
    )

    csv_outputs = sorted(
        _generated_results_dir(workspace.workspace_root).glob("*.csv")
    )
    assert any("relationships" in path.name for path in csv_outputs)
    assert any("MeasureColocalization" in path.name for path in csv_outputs)
    assert _matching_header(
        {path.name: _csv_header(path) for path in csv_outputs},
        "relationships",
    ) == [
        "relationship_type",
        "source_role",
        "target_role",
        "source_object",
        "target_object",
        "parent_id",
        "child_id",
        "slice_index",
        "slice_count",
    ]


def test_official_example_neighbors_cppipe_executes_neighbor_exports(
    tmp_path: Path,
) -> None:
    examples_root = _official_cellprofiler_examples_root()
    cppipe_path = (
        examples_root
        / "CellProfiler3Pipelines"
        / "ExampleNeighbors.cppipe"
    )
    source_root = examples_root / "ExampleNeighbors"
    if not cppipe_path.exists() or not source_root.exists():
        pytest.skip(
            "Official CellProfiler ExampleNeighbors files are not available. "
            f"Set CELLPROFILER_EXAMPLES_ROOT to a local examples checkout; "
            f"looked under {examples_root}."
        )

    prepared = prepare_generated_pipeline(
        cppipe_path,
        output_path=tmp_path / "generated_official_neighbors_pipeline.py",
    )
    workspace = materialize_source_schema_workspace(
        source_root,
        tmp_path / "official_neighbors_openhcs_workspace",
        prepared.source_schema,
    )

    global_config = GlobalPipelineConfig(
        num_workers=1,
        use_threading=True,
        microscope=Microscope.AUTO,
    )
    ensure_global_config_context(GlobalPipelineConfig, global_config)
    pipeline_config = _generated_pipeline_config(
        prepared,
        path_planning_config=LazyPathPlanningConfig(
            output_dir_suffix="_generated_cppipe",
        ),
        vfs_config=VFSConfig(
            materialization_backend=MaterializationBackend.DISK,
        ),
        well_filter_config=LazyWellFilterConfig(
            well_filter=["A01"],
        ),
    )
    orchestrator = PipelineOrchestrator(
        workspace.workspace_root,
        pipeline_config=pipeline_config,
    )
    orchestrator.initialize()

    execution = execute_pipeline_direct(
        orchestrator,
        prepared.runtime_pipeline_steps,
    )

    assert all(
        result.is_success()
        for result in execution.execution_results.values()
    )
    runtime_store = execution.compiled_contexts["A01"].runtime_value_store
    cells_records = runtime_store.find(
        name="Cells",
        artifact_type=ObjectLabelsArtifactType,
        axis_id="A01",
    )
    assert cells_records
    assert np.asarray(cells_records[0].value.data).max() > 0
    assert runtime_store.find(
        name="MeasureObjectNeighbors_10_measurements",
        artifact_type=MeasurementsArtifactType,
        axis_id="A01",
    )

    csv_outputs = sorted(
        _generated_results_dir(workspace.workspace_root).glob("*.csv")
    )
    assert _matching_header(
        {path.name: _csv_header(path) for path in csv_outputs},
        "MeasureObjectNeighbors",
    ) == [
        "slice_index",
        "object_id",
        "scale",
        "number_of_neighbors",
        "percent_touching",
        "first_closest_object_number",
        "first_closest_distance",
        "second_closest_object_number",
        "second_closest_distance",
        "angle_between_neighbors",
        "image_number",
    ]
    image_outputs = sorted(
        _generated_results_dir(workspace.workspace_root).glob("*.tif")
    )
    assert any("ColorNeighbors" in path.name for path in image_outputs)
    assert any("InvertedRedOutlines" in path.name for path in image_outputs)


def test_official_example_illumination_example1_uses_rule_row_binding(
    tmp_path: Path,
) -> None:
    examples_root = _official_cellprofiler_examples_root()
    cppipe_path = (
        examples_root
        / "CellProfiler3Pipelines"
        / "ExampleIlluminationCorrection_Example1_AllMethod.cppipe"
    )
    source_root = examples_root / "ExampleIlluminationCorrection"
    if not cppipe_path.exists() or not source_root.exists():
        pytest.skip(
            "Official CellProfiler ExampleIlluminationCorrection files are not "
            f"available. Set CELLPROFILER_EXAMPLES_ROOT to a local examples "
            f"checkout; looked under {examples_root}."
        )

    prepared = prepare_generated_pipeline(
        cppipe_path,
        output_path=tmp_path / "generated_official_illumination_pipeline.py",
    )
    orig_green = prepared.source_schema.assignment_for_alias("OrigGreen")
    assert orig_green is not None
    assert prepared.source_schema.assignment_for_alias("DNA") is None
    assert orig_green.origin is SourceBindingOrigin.PIPELINE_START
    assert orig_green.selector.components == ()
    assert len(orig_green.selector.filters) == 1
    assert orig_green.selector.filters[0].subject is SourceFilterSubject.FILE
    assert (
        orig_green.selector.filters[0].match_type
        is SourceFilterMatchType.CONTAINS
    )
    assert orig_green.selector.filters[0].value == "AS_09047_"

    workspace = materialize_source_schema_workspace(
        source_root,
        tmp_path / "official_illumination_openhcs_workspace",
        prepared.source_schema,
    )
    effective_well_filter = _effective_source_schema_well_filter(
        workspace,
        requested_well_filter=("A01",),
    )

    global_config = GlobalPipelineConfig(
        num_workers=1,
        use_threading=True,
        microscope=Microscope.AUTO,
    )
    ensure_global_config_context(GlobalPipelineConfig, global_config)
    pipeline_config = _generated_pipeline_config(
        prepared,
        path_planning_config=LazyPathPlanningConfig(
            output_dir_suffix="_generated_cppipe",
        ),
        vfs_config=VFSConfig(
            materialization_backend=MaterializationBackend.DISK,
        ),
        well_filter_config=LazyWellFilterConfig(
            well_filter=list(effective_well_filter),
        ),
    )
    orchestrator = PipelineOrchestrator(
        workspace.workspace_root,
        pipeline_config=pipeline_config,
    )
    orchestrator.initialize()

    execution = execute_pipeline_direct(
        orchestrator,
        prepared.runtime_pipeline_steps,
    )

    assert all(
        result.is_success()
        for result in execution.execution_results.values()
    )
    axis_id = _single_execution_axis(execution)
    image_outputs = sorted(
        (_generated_output_root(workspace.workspace_root) / "images").glob("*.TIF")
    )
    assert [path.name for path in image_outputs] == [
        f"{axis_id}_s001_w1_z001_t001.TIF",
        f"{axis_id}_s002_w1_z001_t001.TIF",
        f"{axis_id}_s003_w1_z001_t001.TIF",
    ]
    corrected = tifffile.imread(image_outputs[0])
    assert corrected.ndim == 2
    assert corrected.shape[0] > 0
    assert corrected.shape[1] > 0


def test_official_example_woundhealing_cppipe_executes_disk_outputs(
    tmp_path: Path,
) -> None:
    examples_root = _official_cellprofiler_examples_root()
    cppipe_path = (
        examples_root
        / "CellProfiler3Pipelines"
        / "ExampleWoundHealing.cppipe"
    )
    source_root = examples_root / "ExampleWoundHealing"
    if not cppipe_path.exists() or not source_root.exists():
        pytest.skip(
            "Official CellProfiler ExampleWoundHealing files are not available. "
            f"Set CELLPROFILER_EXAMPLES_ROOT to a local examples checkout; "
            f"looked under {examples_root}."
        )

    prepared = prepare_generated_pipeline(
        cppipe_path,
        output_path=tmp_path / "generated_official_woundhealing_pipeline.py",
    )
    workspace = materialize_source_schema_workspace(
        source_root,
        tmp_path / "official_woundhealing_openhcs_workspace",
        prepared.source_schema,
    )

    global_config = GlobalPipelineConfig(
        num_workers=1,
        use_threading=True,
        microscope=Microscope.AUTO,
    )
    ensure_global_config_context(GlobalPipelineConfig, global_config)
    pipeline_config = _generated_pipeline_config(
        prepared,
        path_planning_config=LazyPathPlanningConfig(
            output_dir_suffix="_generated_cppipe",
        ),
        vfs_config=VFSConfig(
            materialization_backend=MaterializationBackend.DISK,
        ),
    )
    orchestrator = PipelineOrchestrator(
        workspace.workspace_root,
        pipeline_config=pipeline_config,
    )
    orchestrator.initialize()

    execution = execute_pipeline_direct(orchestrator, prepared.runtime_pipeline_steps)

    assert all(
        result.is_success()
        for result in execution.execution_results.values()
    )
    csv_outputs = sorted(
        _generated_results_dir(workspace.workspace_root).glob(
            "*MeasureImageAreaOccupied_8_measurements_step3.csv"
        )
    )
    assert len(csv_outputs) == 1
    assert _csv_header(csv_outputs[0]) == [
        "slice_index",
        "area_occupied",
        "perimeter",
        "total_area",
        "source_image_name",
        "image_number",
    ]


@pytest.mark.parametrize(
    (
        "pipeline_name",
        "source_name",
        "expected_records",
        "csv_fragments",
        "image_suffixes",
    ),
    (
        pytest.param(
            "ExamplePercentPositive",
            "ExamplePercentPositive",
            (
                ("PH3PosNuclei", ObjectLabelsArtifactType),
                ("Nuclei_PH3_relationships", RelationshipsArtifactType),
                ("CalculateMath_13_measurements", MeasurementsArtifactType),
                ("DisplayImage", ImageArtifactType),
            ),
            ("relationships", "ClassifyObjects", "CalculateMath"),
            (".tif",),
            id="percent-positive",
        ),
        pytest.param(
            "ExampleSpeckles",
            "ExampleSpeckles",
            (
                ("h2ax", ObjectLabelsArtifactType),
                ("Nuclei_h2ax_relationships", RelationshipsArtifactType),
                ("MeasureObjectIntensity_10_measurements", MeasurementsArtifactType),
            ),
            ("relationships", "MeasureObjectIntensity", "RelateObjects"),
            (),
            id="speckles",
        ),
        pytest.param(
            "ExampleTumor",
            "ExampleTumor",
            (
                ("tumor", ObjectLabelsArtifactType),
                ("TumorOutline", ImageArtifactType),
                ("MeasureObjectSizeShape_8_measurements", MeasurementsArtifactType),
            ),
            ("MeasureObjectSizeShape",),
            (".jpg",),
            id="tumor",
        ),
        pytest.param(
            "ExampleUntangleAndStraightenWorms",
            "ExampleStraightenWorms",
            (
                ("StraightenedWorms", ObjectLabelsArtifactType),
                (
                    "NonOverlappingWorms_HeadMarkers_relationships",
                    RelationshipsArtifactType,
                ),
                ("StraightenWorms_11_measurements", MeasurementsArtifactType),
                ("StraightenedRG", ImageArtifactType),
            ),
            ("relationships", "StraightenWorms", "UntangleWorms"),
            (".tif",),
            id="untangle-and-straighten",
        ),
        pytest.param(
            "ExampleYeastColonies",
            "ExampleYeastColonies",
            (
                ("Colonies", ObjectLabelsArtifactType),
                ("OutlinedColonies", ImageArtifactType),
                ("ClassifyObjects_18_measurements", MeasurementsArtifactType),
            ),
            (
                "MeasureObjectIntensity",
                "ClassifyObjects",
            ),
            (".jpg", ".tif"),
            id="yeast-colonies",
        ),
        pytest.param(
            "ExampleYeastPatches",
            "ExampleYeastPatches",
            (
                ("Prespots", ObjectLabelsArtifactType),
                ("FilterObjects", ObjectLabelsArtifactType),
                ("NaturalSpots", ObjectLabelsArtifactType),
                ("ForcedSpots", ObjectLabelsArtifactType),
                ("Grid", SpatialGridArtifactType),
                ("MeasureObjectIntensity_18_measurements", MeasurementsArtifactType),
            ),
            (
                "FilterObjects",
                "Grid",
                "IdentifyObjectsInGrid",
            ),
            (".JPG",),
            id="yeast-patches-grid-illumination",
        ),
        pytest.param(
            "ExampleImagingFlowCytometryObjectsInGrid",
            "ExampleImagingFlowCytometryObjectsInGrid",
            (
                ("BF_cells_on_grid", ObjectLabelsArtifactType),
                (
                    "Non_empty_tile_FilteredBF_relationships",
                    RelationshipsArtifactType,
                ),
                ("MeasureGranularity_24_measurements", MeasurementsArtifactType),
                ("MeasureTexture_25_measurements", MeasurementsArtifactType),
                (
                    "MeasureObjectIntensityDistribution_30_measurements",
                    MeasurementsArtifactType,
                ),
            ),
            (
                "relationships",
                "FilterObjects_19",
                "MeasureGranularity",
                "MeasureTexture",
                "MeasureObjectIntensityDistribution",
            ),
            (".tif",),
            id="imaging-flow-cytometry-grid",
        ),
        pytest.param(
            "ExampleTrackObjects",
            "ExampleTrackObjects",
            (
                ("TrackedCells", ImageArtifactType),
                ("TrackObjects_9_measurements", MeasurementsArtifactType),
                ("OutlineImage", ImageArtifactType),
                ("AdjacentImage", ImageArtifactType),
            ),
            ("TrackObjects",),
            (".tif",),
            id="track-objects",
        ),
        pytest.param(
            "ExampleVitra",
            "ExampleVitraImages",
            (
                ("CorrProtein", ImageArtifactType),
                ("Cells", ObjectLabelsArtifactType),
                ("Cytoplasm", ObjectLabelsArtifactType),
                ("Outlined", ImageArtifactType),
                ("MeasureObjectIntensity_9_measurements", MeasurementsArtifactType),
                ("CalculateMath_10_measurements", MeasurementsArtifactType),
                ("CalculateMath_11_measurements", MeasurementsArtifactType),
            ),
            (
                "MeasureObjectIntensity",
                "CalculateMath_10",
                "CalculateMath_11",
            ),
            (".tif",),
            id="vitra-npy-illumination",
        ),
    ),
)
def test_official_cellprofiler3_additional_representative_pipelines_execute(
    tmp_path: Path,
    pipeline_name: str,
    source_name: str,
    expected_records: tuple[tuple[str, ArtifactType], ...],
    csv_fragments: tuple[str, ...],
    image_suffixes: tuple[str, ...],
) -> None:
    workspace, execution = _execute_official_cellprofiler3_pipeline(
        tmp_path,
        pipeline_name,
        source_name,
        well_filter=("A01",),
    )

    assert all(
        result.is_success()
        for result in execution.execution_results.values()
    )
    axis_id = _single_execution_axis(execution)
    runtime_store = execution.compiled_contexts[axis_id].runtime_value_store
    for name, kind in expected_records:
        assert _runtime_store_has_semantic_record(
            runtime_store,
            name=name,
            artifact_type=kind,
            axis_id=axis_id,
        )

    result_outputs = sorted(
        path
        for path in _generated_results_dir(workspace.workspace_root).rglob("*")
        if path.is_file()
    )
    assert result_outputs
    result_names = tuple(path.name for path in result_outputs)
    for fragment in csv_fragments:
        assert any(fragment in name for name in result_names)

    image_outputs = sorted(
        path
        for path in _generated_output_root(workspace.workspace_root).rglob("*")
        if path.is_file()
    )
    image_names = tuple(path.name for path in image_outputs if path.is_file())
    for suffix in image_suffixes:
        assert any(name.endswith(suffix) for name in image_names)


def _runtime_store_has_semantic_record(
    runtime_store,
    *,
    name: str,
    artifact_type: type[ArtifactType],
    axis_id: str,
) -> bool:
    """Return whether the runtime store contains the public semantic artifact."""
    if runtime_store.find(name=name, artifact_type=artifact_type, axis_id=axis_id):
        return True
    if artifact_type is not MeasurementsArtifactType:
        return False
    match = re.fullmatch(r"(?P<prefix>.+)_\d+_measurements", name)
    if match is None:
        return False
    prefix = match.group("prefix")
    return any(
        record.key.name.startswith(f"{prefix}_")
        and record.key.name.endswith("_measurements")
        for record in runtime_store.find(
            artifact_type=artifact_type,
            axis_id=axis_id,
        )
    )


def test_official_cellprofiler3_cppipe_corpus_prepares(
    tmp_path: Path,
) -> None:
    examples_root = _official_cellprofiler_examples_root()
    cppipe_dir = examples_root / "CellProfiler3Pipelines"
    if not cppipe_dir.exists():
        pytest.skip(
            "Official CellProfiler3 pipeline corpus is not available. "
            f"Set CELLPROFILER_EXAMPLES_ROOT to a local examples checkout; "
            f"looked under {examples_root}."
        )

    cppipe_paths = tuple(sorted(cppipe_dir.glob("*.cppipe")))
    assert cppipe_paths
    failures: list[str] = []
    for cppipe_path in cppipe_paths:
        try:
            prepared = prepare_generated_pipeline(
                cppipe_path,
                output_path=tmp_path / f"{cppipe_path.stem}_openhcs.py",
            )
        except Exception as exc:  # pragma: no cover - assertion includes details
            failures.append(f"{cppipe_path.name}: {type(exc).__name__}: {exc}")
            continue
        assert prepared.pipeline.steps

    assert not failures


def test_official_cellprofiler3_cppipe_corpus_executes_when_enabled(
    tmp_path: Path,
) -> None:
    exhaustive_execution_env = (
        "OPENHCS_RUN_OFFICIAL_CELLPROFILER3_CORPUS_EXECUTION"
    )
    if os.environ.get(exhaustive_execution_env) != "1":
        pytest.skip(
            "Official corpus execution is intentionally opt-in because it runs "
            f"every discovered CellProfiler3 .cppipe. Set "
            f"{exhaustive_execution_env}=1 to enable it."
        )

    examples_root = _official_cellprofiler_examples_root()
    cppipe_dir = examples_root / "CellProfiler3Pipelines"
    if not cppipe_dir.exists():
        pytest.skip(
            "Official CellProfiler3 pipeline corpus is not available. "
            f"Set CELLPROFILER_EXAMPLES_ROOT to a local examples checkout; "
            f"looked under {examples_root}."
        )

    failures: list[str] = []
    cppipe_paths = tuple(sorted(cppipe_dir.glob("*.cppipe")))
    assert cppipe_paths
    for cppipe_path in cppipe_paths:
        pipeline_name = cppipe_path.stem
        try:
            workspace, execution = _execute_official_cellprofiler3_pipeline(
                tmp_path,
                pipeline_name,
                _official_cellprofiler3_source_name_for_pipeline(
                    examples_root,
                    pipeline_name,
                ),
                well_filter=("A01",),
            )
        except Exception as exc:  # pragma: no cover - assertion includes details
            failures.append(
                f"{pipeline_name}: {type(exc).__name__}: {exc}"
            )
            continue

        unsuccessful_results = {
            axis: result
            for axis, result in execution.execution_results.items()
            if not result.is_success()
        }
        if unsuccessful_results:
            failures.append(
                f"{pipeline_name}: unsuccessful execution results: "
                f"{unsuccessful_results!r} in {workspace.workspace_root}"
            )

    assert not failures


def test_cppipe_generated_pipeline_materializes_relationship_outputs(
    tmp_path: Path,
) -> None:
    plate_path = _generate_plate(tmp_path / "relationship_plate")
    cppipe_path = _write_relationship_cppipe(tmp_path / "relate_objects.cppipe")
    prepared = prepare_generated_pipeline(
        cppipe_path,
        output_path=tmp_path / "generated_relationship_pipeline.py",
    )

    global_config = GlobalPipelineConfig(num_workers=1, use_threading=True)
    ensure_global_config_context(GlobalPipelineConfig, global_config)
    pipeline_config = _generated_pipeline_config(
        prepared,
        path_planning_config=LazyPathPlanningConfig(
            output_dir_suffix="_generated_cppipe",
        ),
        vfs_config=VFSConfig(
            materialization_backend=MaterializationBackend.DISK,
        ),
    )
    orchestrator = PipelineOrchestrator(plate_path, pipeline_config=pipeline_config)
    orchestrator.initialize()

    execution = execute_pipeline_direct(orchestrator, prepared.runtime_pipeline_steps)

    assert all(
        result.is_success()
        for result in execution.execution_results.values()
    )
    validate_cppipe_execution(
        prepared,
        execution,
        _generated_output_root(plate_path),
    )

    runtime_store = execution.compiled_contexts["A01"].runtime_value_store
    relationship_records = runtime_store.find(
        artifact_type=RelationshipsArtifactType,
        axis_id="A01",
    )
    measurement_records = runtime_store.find(
        artifact_type=MeasurementsArtifactType,
        axis_id="A01",
    )
    assert relationship_records
    assert measurement_records
    relationship_record = relationship_records[0]
    relationship = ObjectRelationship.from_runtime_value(relationship_record.value)
    assert relationship.source.name == "Nuclei"
    assert relationship.target.name == "Cells"
    assert relationship.relationship_type == "parent_child"

    csv_outputs = sorted(_generated_results_dir(plate_path).rglob("*.csv"))
    assert csv_outputs
    assert any("relationships" in path.name for path in csv_outputs)
    assert any("measurements" in path.name for path in csv_outputs)
    headers_by_name = {path.name: _csv_header(path) for path in csv_outputs}
    assert _matching_header(
        headers_by_name,
        "relationships",
    ) == [
        "relationship_type",
        "source_role",
        "target_role",
        "source_object",
        "target_object",
        "parent_id",
        "child_id",
        "slice_index",
        "slice_count",
    ]
    assert _matching_header(
        headers_by_name,
        "RelateObjects_4_measurements",
    ) == [
        "slice_index",
        "parent_object_count",
        "child_object_count",
        "children_with_parents_count",
        "mean_children_per_parent",
        "mean_centroid_distance",
        "mean_minimum_distance",
        "object_name",
        "object_label",
        "children_cells_count",
        "parent_nuclei",
        "distance_centroid_nuclei",
        "distance_minimum_nuclei",
        "image_number",
    ]


def test_percent_positive_cppipe_executes_relationship_measurement_consumers(
    tmp_path: Path,
) -> None:
    source_root = _generate_percent_positive_source_folder(
        tmp_path / "ExamplePercentPositive"
    )
    cppipe_path = _write_percent_positive_cppipe(
        tmp_path / "percent_positive.cppipe"
    )
    prepared = prepare_generated_pipeline(
        cppipe_path,
        output_path=tmp_path / "generated_percent_positive_pipeline.py",
    )
    workspace = materialize_source_schema_workspace(
        source_root,
        tmp_path / "percent_positive_openhcs_workspace",
        prepared.source_schema,
    )

    global_config = GlobalPipelineConfig(
        num_workers=1,
        use_threading=True,
        microscope=Microscope.AUTO,
    )
    ensure_global_config_context(GlobalPipelineConfig, global_config)
    pipeline_config = _generated_pipeline_config(
        prepared,
        path_planning_config=LazyPathPlanningConfig(
            output_dir_suffix="_generated_cppipe",
        ),
        vfs_config=VFSConfig(
            materialization_backend=MaterializationBackend.DISK,
        ),
    )
    orchestrator = PipelineOrchestrator(
        workspace.workspace_root,
        pipeline_config=pipeline_config,
    )
    orchestrator.initialize()

    execution = execute_pipeline_direct(orchestrator, prepared.runtime_pipeline_steps)

    assert all(
        result.is_success()
        for result in execution.execution_results.values()
    )
    validate_cppipe_execution(
        prepared,
        execution,
        _generated_output_root(workspace.workspace_root),
    )
    runtime_store = execution.compiled_contexts["A01"].runtime_value_store
    assert runtime_store.find(
        name="PH3PosNuclei",
        artifact_type=ObjectLabelsArtifactType,
        axis_id="A01",
    )
    assert runtime_store.find(
        name="DisplayImage",
        artifact_type=ImageArtifactType,
        axis_id="A01",
    )
    calculate_math_records = runtime_store.find(
        name="CalculateMath_11_measurements",
        artifact_type=MeasurementsArtifactType,
        axis_id="A01",
    )
    assert calculate_math_records
    assert calculate_math_records[0].value.data[0]["output_name"] == "PercentPositive"


def _generate_plate(plate_path: Path) -> Path:
    generator = SyntheticMicroscopyGenerator(
        output_dir=str(plate_path),
        grid_size=(1, 1),
        tile_size=(128, 128),
        wavelengths=1,
        z_stack_levels=1,
        num_cells=12,
        cell_size_range=(8, 12),
        cell_intensity_range=(28000, 42000),
        background_intensity=200,
        noise_level=10,
        wells=["A01"],
        format="ImageXpress",
        random_seed=7,
    )
    generator.generate_dataset()
    return plate_path


def _generate_two_channel_plate(plate_path: Path) -> Path:
    generator = SyntheticMicroscopyGenerator(
        output_dir=str(plate_path),
        grid_size=(1, 1),
        tile_size=(128, 128),
        wavelengths=2,
        z_stack_levels=1,
        num_cells=12,
        cell_size_range=(8, 12),
        cell_intensity_range=(28000, 42000),
        background_intensity=200,
        noise_level=10,
        wells=["A01"],
        format="ImageXpress",
        random_seed=11,
    )
    generator.generate_dataset()
    return plate_path


def _generate_bbbc021_plate(plate_path: Path) -> Path:
    fields_dir = plate_path / "fields"
    fields_dir.mkdir(parents=True)
    _write_bbbc021_image(
        fields_dir / "A01_s1_w1BEDC2073-A983-4B98-95E9-84466707A25D.tif",
        seed=1,
        signal=3200,
    )
    _write_bbbc021_image(
        fields_dir / "A01_s1_w242F8F7B1-17A7-4AA1-9C50-123456789ABC.tif",
        seed=2,
        signal=1800,
    )
    return plate_path


def _generate_loadimages_mat_illum_plate(plate_path: Path) -> Path:
    generator = SyntheticMicroscopyGenerator(
        output_dir=str(plate_path),
        grid_size=(1, 1),
        tile_size=(64, 64),
        wavelengths=1,
        z_stack_levels=1,
        num_cells=4,
        cell_size_range=(6, 8),
        cell_intensity_range=(28000, 42000),
        background_intensity=200,
        noise_level=10,
        wells=["A01"],
        format="ImageXpress",
        random_seed=17,
    )
    generator.generate_dataset()
    savemat(
        plate_path / "illum_Channel2.mat",
        {"Image": np.full((64, 64), 2.0, dtype=np.float32)},
    )
    return plate_path


def _generate_examplehuman_source_folder(source_root: Path) -> Path:
    images_dir = source_root / "images"
    images_dir.mkdir(parents=True)
    base_name = "AS_09125_050116030001_D03f00d"
    for channel, seed in enumerate((23, 29, 31)):
        _write_examplehuman_image(
            images_dir / f"{base_name}{channel}.tif",
            seed=seed,
            signal=2200 + channel * 400,
        )
    return source_root


def _generate_percent_positive_source_folder(source_root: Path) -> Path:
    images_dir = source_root / "images"
    images_dir.mkdir(parents=True)
    _write_percent_positive_image(
        images_dir / "PercentPositive_A01_s001_w0d0.tif",
        seed=37,
        spots=((32, 32, 12, 46000), (76, 54, 10, 42000), (84, 92, 11, 44000)),
    )
    _write_percent_positive_image(
        images_dir / "PercentPositive_A01_s001_w1d1.tif",
        seed=41,
        spots=((32, 32, 7, 52000),),
    )
    return source_root


def _write_bbbc021_image(path: Path, *, seed: int, signal: int) -> None:
    rng = np.random.default_rng(seed)
    image = rng.normal(900, 40, size=(64, 64)).clip(0, 65535).astype(np.uint16)
    image[20:44, 20:44] = np.clip(
        image[20:44, 20:44].astype(np.int32) + signal,
        0,
        65535,
    ).astype(np.uint16)
    tifffile.imwrite(path, image, description="spatial-calibration-x: 1.0")


def _write_examplehuman_image(path: Path, *, seed: int, signal: int) -> None:
    rng = np.random.default_rng(seed)
    image = rng.normal(650, 35, size=(128, 128)).clip(0, 65535).astype(np.uint16)
    for center_y, center_x in ((40, 44), (84, 86), (46, 92)):
        y0, y1 = center_y - 8, center_y + 8
        x0, x1 = center_x - 8, center_x + 8
        image[y0:y1, x0:x1] = np.clip(
            image[y0:y1, x0:x1].astype(np.int32) + signal,
            0,
            65535,
        ).astype(np.uint16)
    Image.fromarray(image).save(path)


def _write_percent_positive_image(
    path: Path,
    *,
    seed: int,
    spots: tuple[tuple[int, int, int, int], ...],
) -> None:
    rng = np.random.default_rng(seed)
    image = rng.normal(300, 15, size=(128, 128)).clip(0, 65535).astype(np.uint16)
    yy, xx = np.ogrid[:128, :128]
    for center_y, center_x, radius, signal in spots:
        mask = (yy - center_y) ** 2 + (xx - center_x) ** 2 <= radius**2
        image[mask] = np.clip(
            image[mask].astype(np.int32) + signal,
            0,
            65535,
        ).astype(np.uint16)
    Image.fromarray(image).save(path)


def _generated_output_root(plate_path: Path) -> Path:
    return plate_path.parent / f"{plate_path.name}_generated_cppipe"


def _generated_results_dir(plate_path: Path) -> Path:
    return _generated_output_root(plate_path) / "images_results"


def _official_cellprofiler_examples_root() -> Path:
    return Path(
        os.environ.get(
            "CELLPROFILER_EXAMPLES_ROOT",
            "/tmp/cellprofiler_examples",
        )
    )


def _official_cellprofiler3_source_name_for_pipeline(
    examples_root: Path,
    pipeline_name: str,
) -> str:
    candidate_names = (
        pipeline_name,
        pipeline_name.removesuffix("URL"),
        pipeline_name.split("_", maxsplit=1)[0],
        f"{pipeline_name}Images",
        pipeline_name.replace("ExampleUntangleAnd", "Example"),
    )
    for candidate_name in candidate_names:
        if candidate_name and (examples_root / candidate_name).exists():
            return candidate_name
    raise FileNotFoundError(
        f"No source directory found for official pipeline {pipeline_name!r} "
        f"under {examples_root}."
    )


def _single_execution_axis(execution: DirectPipelineExecution) -> str:
    """Return the only compiled axis for one-sample generated-pipeline tests."""
    axis_ids = tuple(str(axis_id) for axis_id in execution.compiled_contexts)
    assert len(axis_ids) == 1
    return axis_ids[0]


def _execute_official_cellprofiler3_pipeline(
    tmp_path: Path,
    pipeline_name: str,
    source_name: str,
    *,
    well_filter: tuple[str, ...],
) -> tuple[SourceSchemaWorkspaceMaterialization, DirectPipelineExecution]:
    examples_root = _official_cellprofiler_examples_root()
    cppipe_path = (
        examples_root
        / "CellProfiler3Pipelines"
        / f"{pipeline_name}.cppipe"
    )
    source_root = examples_root / source_name
    if not cppipe_path.exists() or not source_root.exists():
        pytest.skip(
            f"Official CellProfiler {pipeline_name} files are not available. "
            f"Set CELLPROFILER_EXAMPLES_ROOT to a local examples checkout; "
            f"looked under {examples_root}."
        )

    prepared = prepare_generated_pipeline(
        cppipe_path,
        output_path=tmp_path / f"generated_{pipeline_name}_pipeline.py",
    )
    workspace = materialize_source_schema_workspace(
        source_root,
        tmp_path / f"{pipeline_name}_openhcs_workspace",
        prepared.source_schema,
    )
    effective_well_filter = _effective_source_schema_well_filter(
        workspace,
        requested_well_filter=well_filter,
    )

    global_config = GlobalPipelineConfig(
        num_workers=1,
        use_threading=True,
        microscope=Microscope.AUTO,
    )
    ensure_global_config_context(GlobalPipelineConfig, global_config)
    pipeline_config = _generated_pipeline_config(
        prepared,
        path_planning_config=LazyPathPlanningConfig(
            output_dir_suffix="_generated_cppipe",
        ),
        vfs_config=VFSConfig(
            materialization_backend=MaterializationBackend.DISK,
        ),
        well_filter_config=LazyWellFilterConfig(
            well_filter=list(effective_well_filter),
        ),
    )
    orchestrator = PipelineOrchestrator(
        workspace.workspace_root,
        pipeline_config=pipeline_config,
    )
    orchestrator.initialize()

    execution = execute_pipeline_direct(
        orchestrator,
        prepared.runtime_pipeline_steps,
    )
    validate_cppipe_execution(
        prepared,
        execution,
        _generated_output_root(workspace.workspace_root),
    )
    return workspace, execution


def _effective_source_schema_well_filter(
    workspace: SourceSchemaWorkspaceMaterialization,
    *,
    requested_well_filter: tuple[str, ...],
) -> tuple[str, ...]:
    """Resolve a requested one-sample filter against source-schema identity."""
    available_wells = workspace.primary_wells()
    requested_available = tuple(
        well for well in requested_well_filter if well in available_wells
    )
    if requested_available:
        return requested_available
    return available_wells[:1]


def _csv_header(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        return next(csv.reader(handle))


def _matching_header(
    headers_by_name: dict[str, list[str]],
    name_fragment: str,
) -> list[str]:
    for filename, header in headers_by_name.items():
        if name_fragment in filename:
            return header
    raise AssertionError(
        f"No CSV output filename contained {name_fragment!r}: "
        f"{sorted(headers_by_name)}"
    )


def _write_cppipe(cppipe_path: Path) -> Path:
    cppipe_path.write_text(
        "\n".join(
            (
                "CellProfiler Pipeline: http://www.cellprofiler.org",
                "Version:3",
                "DateRevision:300",
                "GitHash:",
                "ModuleCount:3",
                "HasImagePlaneDetails:False",
                (
                    "LoadData:[module_num:1|svn_version:'Unknown'|"
                    "enabled:True|wants_pause:False]"
                ),
                "    Input data file location:Elsewhere...",
                (
                    "IdentifyPrimaryObjects:[module_num:2|svn_version:'Unknown'|"
                    "enabled:True|wants_pause:False]"
                ),
                "    Select the input image:OrigBlue",
                "    Name the primary objects to be identified:Nuclei",
                (
                    "ExportToSpreadsheet:[module_num:3|svn_version:'Unknown'|"
                    "enabled:True|wants_pause:False]"
                ),
                "    Select measurements to export:No",
                "",
            )
        )
    )
    return cppipe_path


def _write_bbbc021_cppipe(cppipe_path: Path) -> Path:
    cppipe_path.write_text(
        "\n".join(
            (
                "CellProfiler Pipeline: http://www.cellprofiler.org",
                "Version:3",
                "DateRevision:300",
                "GitHash:",
                "ModuleCount:6",
                "HasImagePlaneDetails:False",
                (
                    "Images:[module_num:1|svn_version:'Unknown'|"
                    "enabled:True|wants_pause:False]"
                ),
                "    Filter images?:Images only",
                '    Select the rule criteria:or (file does containregexp "A01")',
                (
                    "Metadata:[module_num:2|svn_version:'Unknown'|"
                    "enabled:True|wants_pause:False]"
                ),
                "    Metadata extraction method:Extract from file/folder names",
                "    Metadata source:File name",
                (
                    "    Regular expression to extract from file name:"
                    "^.*(?P<well>[A-Z]\\d+)_s(?P<site>\\d+)_w(?P<channel>\\d).*$"
                ),
                (
                    "NamesAndTypes:[module_num:3|svn_version:'Unknown'|"
                    "enabled:True|wants_pause:False]"
                ),
                "    Assign a name to:Images matching rules",
                "    Select the image type:Grayscale image",
                "    Name to assign these images:DNA",
                "    Match metadata:[{'DNA': 'well'}, {'DNA': 'site'}]",
                "    Image set matching method:Metadata",
                '    Select the rule criteria:and (metadata does channel "1")',
                "    Assign a name to:Images matching rules",
                "    Select the image type:Grayscale image",
                "    Name to assign these images:Actin",
                "    Match metadata:[{'Actin': 'well'}, {'Actin': 'site'}]",
                "    Image set matching method:Metadata",
                '    Select the rule criteria:and (metadata does channel "2")',
                (
                    "IdentifyPrimaryObjects:[module_num:4|svn_version:'Unknown'|"
                    "enabled:True|wants_pause:False]"
                ),
                "    Select the input image:DNA",
                "    Name the primary objects to be identified:Nuclei",
                (
                    "GrayToColor:[module_num:5|svn_version:'Unknown'|"
                    "enabled:True|wants_pause:False]"
                ),
                "    Select the image to be colored green:Actin",
                "    Select the image to be colored blue:DNA",
                "    Name the output image:Composite",
                (
                    "ExportToSpreadsheet:[module_num:6|svn_version:'Unknown'|"
                    "enabled:True|wants_pause:False]"
                ),
                "    Select measurements to export:No",
                "",
            )
        )
    )
    return cppipe_path


def _write_loadimages_mat_illum_cppipe(cppipe_path: Path) -> Path:
    cppipe_path.write_text(
        "\n".join(
            (
                "CellProfiler Pipeline: http://www.cellprofiler.org",
                "Version:3",
                "DateRevision:300",
                "GitHash:",
                "ModuleCount:2",
                "HasImagePlaneDetails:False",
                (
                    "LoadImages:[module_num:1|svn_version:'Unknown'|"
                    "enabled:True|wants_pause:False]"
                ),
                "    What type of files are you loading?:individual images",
                "    How do you want to load these files?:Text-Exact match",
                "    Do you want to exclude certain files?:No",
                "    Type the text that these images have in common (case-sensitive):w1",
                "    What do you want to call this image in CellProfiler?:Raw",
                "    What is the position of this image in each group?:1",
                (
                    "    Do you want to extract metadata from the file name, "
                    "the subfolder path or both?:None"
                ),
                (
                    "    Type the text that these images have in common "
                    "(case-sensitive):illum_Channel2"
                ),
                "    What do you want to call this image in CellProfiler?:Illum",
                "    What is the position of this image in each group?:2",
                (
                    "    Do you want to extract metadata from the file name, "
                    "the subfolder path or both?:None"
                ),
                (
                    "CorrectIlluminationApply:[module_num:2|svn_version:'Unknown'|"
                    "enabled:True|wants_pause:False]"
                ),
                "    Select the input image:Raw",
                "    Name the output image:CorrectedRaw",
                "    Select the illumination function:Illum",
                "    Select how the illumination function is applied:Divide",
                "",
            )
        )
    )
    return cppipe_path


def _write_relationship_cppipe(cppipe_path: Path) -> Path:
    cppipe_path.write_text(
        "\n".join(
            (
                "CellProfiler Pipeline: http://www.cellprofiler.org",
                "Version:3",
                "DateRevision:300",
                "GitHash:",
                "ModuleCount:5",
                "HasImagePlaneDetails:False",
                (
                    "LoadData:[module_num:1|svn_version:'Unknown'|"
                    "enabled:True|wants_pause:False]"
                ),
                "    Input data file location:Elsewhere...",
                (
                    "IdentifyPrimaryObjects:[module_num:2|svn_version:'Unknown'|"
                    "enabled:True|wants_pause:False]"
                ),
                "    Select the input image:OrigBlue",
                "    Name the primary objects to be identified:Nuclei",
                (
                    "IdentifySecondaryObjects:[module_num:3|svn_version:'Unknown'|"
                    "enabled:True|wants_pause:False]"
                ),
                "    Select the input objects:Nuclei",
                "    Name the objects to be identified:Cells",
                "    Select the method to identify the secondary objects:Propagation",
                "    Select the input image:OrigBlue",
                "    Name the new primary objects:FilteredNuclei",
                (
                    "RelateObjects:[module_num:4|svn_version:'Unknown'|"
                    "enabled:True|wants_pause:False]"
                ),
                "    Select the parent objects:Nuclei",
                "    Select the child objects:Cells",
                (
                    "ExportToSpreadsheet:[module_num:5|svn_version:'Unknown'|"
                    "enabled:True|wants_pause:False]"
                ),
                "    Select measurements to export:No",
                "",
            )
        )
    )
    return cppipe_path


def _write_percent_positive_cppipe(cppipe_path: Path) -> Path:
    cppipe_path.write_text(
        "\n".join(
            (
                "CellProfiler Pipeline: http://www.cellprofiler.org",
                "Version:3",
                "DateRevision:300",
                "GitHash:",
                "ModuleCount:12",
                "HasImagePlaneDetails:False",
                (
                    "Images:[module_num:1|svn_version:'Unknown'|"
                    "enabled:True|wants_pause:False]"
                ),
                "    Filter images?:Images only",
                "    Select the rule criteria:and (extension does isimage)",
                (
                    "Metadata:[module_num:2|svn_version:'Unknown'|"
                    "enabled:True|wants_pause:False]"
                ),
                "    Metadata extraction method:Extract from file/folder names",
                "    Metadata source:File name",
                (
                    "    Regular expression to extract from file name:"
                    "^(?P<Plate>.*)_(?P<Well>[A-P][0-9]{2})_s"
                    "(?P<Site>[0-9])_w(?P<ChannelNumber>[0-9])"
                ),
                "    Select the filtering criteria:and (file does contain \"\")",
                (
                    "NamesAndTypes:[module_num:3|svn_version:'Unknown'|"
                    "enabled:True|wants_pause:False]"
                ),
                "    Assign a name to:Images matching rules",
                "    Select the image type:Grayscale image",
                "    Name to assign these images:DNA",
                "    Image set matching method:Order",
                "    Assignments count:2",
                "    Single images count:0",
                "    Process as 3D?:No",
                "    Select the rule criteria:and (file does contain \"d0.tif\")",
                "    Name to assign these images:OrigBlue",
                "    Name to assign these objects:Cell",
                "    Select the image type:Grayscale image",
                "    Select the rule criteria:and (file does contain \"d1.tif\")",
                "    Name to assign these images:OrigGreen",
                "    Name to assign these objects:Cell",
                "    Select the image type:Grayscale image",
                (
                    "IdentifyPrimaryObjects:[module_num:4|svn_version:'Unknown'|"
                    "enabled:True|wants_pause:False]"
                ),
                "    Select the input image:OrigBlue",
                "    Name the primary objects to be identified:Nuclei",
                (
                    "IdentifyPrimaryObjects:[module_num:5|svn_version:'Unknown'|"
                    "enabled:True|wants_pause:False]"
                ),
                "    Select the input image:OrigGreen",
                "    Name the primary objects to be identified:PH3",
                (
                    "RelateObjects:[module_num:6|svn_version:'Unknown'|"
                    "enabled:True|wants_pause:False]"
                ),
                "    Parent objects:Nuclei",
                "    Child objects:PH3",
                (
                    "FilterObjects:[module_num:7|svn_version:'Unknown'|"
                    "enabled:True|wants_pause:False]"
                ),
                "    Select the objects to filter:Nuclei",
                "    Name the output objects:PH3PosNuclei",
                "    Select the filtering mode:Measurements",
                "    Select the filtering method:Limits",
                "    Measurement count:1",
                "    Additional object count:0",
                "    Select the measurement to filter by:Children_PH3_Count",
                "    Filter using a minimum measurement value?:Yes",
                "    Minimum value:1",
                "    Filter using a maximum measurement value?:No",
                "    Maximum value:1.0",
                (
                    "MeasureObjectIntensity:[module_num:8|svn_version:'Unknown'|"
                    "enabled:True|wants_pause:False]"
                ),
                "    Select images to measure:OrigGreen, OrigBlue",
                "    Select objects to measure:Nuclei",
                (
                    "OverlayOutlines:[module_num:9|svn_version:'Unknown'|"
                    "enabled:True|wants_pause:False]"
                ),
                "    Display outlines on a blank image?:No",
                "    Select image on which to display outlines:OrigGreen",
                "    Name the output image:OrigGreenOverlay",
                "    Outline display mode:Color",
                "    Select method to determine brightness of outlines:Max of image",
                "    How to outline:Inner",
                "    Select outline color:#00FF40",
                "    Select objects to display:Nuclei",
                (
                    "DisplayDataOnImage:[module_num:10|svn_version:'Unknown'|"
                    "enabled:True|wants_pause:False]"
                ),
                "    Display object or image measurements?:Object",
                "    Select the input objects:Nuclei",
                "    Measurement to display:Intensity_MaxIntensity_OrigGreen",
                "    Select the image on which to display the measurements:OrigGreenOverlay",
                "    Name the output image that has the measurements displayed:DisplayImage",
                (
                    "CalculateMath:[module_num:11|svn_version:'Unknown'|"
                    "enabled:True|wants_pause:False]"
                ),
                "    Name the output measurement:PercentPositive",
                "    Operation:Divide",
                "    Select the numerator measurement type:Image",
                "    Select the numerator objects:None",
                "    Select the numerator measurement:Count_PH3PosNuclei",
                "    Multiply the above operand by:1.0",
                "    Raise the power of above operand by:1.0",
                "    Select the denominator measurement type:Image",
                "    Select the denominator objects:Nuclei",
                "    Select the denominator measurement:Count_Nuclei",
                "    Multiply the above operand by:1.0",
                "    Raise the power of above operand by:1.0",
                "    Take log10 of result?:No",
                "    Multiply the result by:100",
                "    Raise the power of result by:1.0",
                "    Add to the result:0.0",
                "    How should the output value be rounded?:Not rounded",
                "    Enter how many decimal places the value should be rounded to:0",
                "    Constrain the result to a lower bound?:No",
                "    Enter the lower bound:0.0",
                "    Constrain the result to an upper bound?:No",
                "    Enter the upper bound:1.0",
                (
                    "ExportToSpreadsheet:[module_num:12|svn_version:'Unknown'|"
                    "enabled:True|wants_pause:False]"
                ),
                "    Select measurements to export:No",
                "",
            )
        )
    )
    return cppipe_path
