from __future__ import annotations
from openhcs.core.pipeline_document import PipelineDocumentAuthority

import ast
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import tifffile
from objectstate import replace_raw
from objectstate.lazy_factory import ensure_global_config_context
from PIL import Image
from polystore.base import _create_storage_registry
from polystore.filemanager import FileManager
from scipy.io import savemat
from zmqruntime.execution.responses import (
    ExecutionSubmissionResponse,
    ExecutionWaitResult,
)

from benchmark.cellprofiler_comparison import (
    CellProfilerComparisonCase,
    load_comparison_cases,
)
from benchmark.datasets.cache import (
    CELLPROFILER_EXAMPLES_ROOT_ENV,
    BenchmarkPathRootKind,
    resolve_benchmark_path_root,
)
from openhcs.constants import Backend, Microscope
from openhcs.constants.constants import AllComponents
from openhcs.constants.input_source import InputSource
from openhcs.core.artifacts import (
    ArtifactType,
    ImageArtifactType,
    MeasurementsArtifactType,
    ObjectLineageArtifactType,
    ObjectLabelsArtifactType,
    RelationshipsArtifactType,
    SpatialGridArtifactType,
)
from openhcs.core.config import (
    AnalysisConsolidationConfig,
    GlobalPipelineConfig,
    LazyPathPlanningConfig,
    LazyProcessingConfig,
    LazyVFSConfig,
    LazyWellFilterConfig,
    MaterializationBackend,
    PipelineConfig,
)
from openhcs.core.function_step_transport import FunctionStepTransportAuthority
from openhcs.core.runtime_object_labels import (
    ObjectLabelValue,
    object_label_dense_array,
)
from openhcs.core.runtime_measurements import MeasurementTable
from openhcs.core.runtime_relationships import (
    ObjectRelationship,
)
from openhcs.core.runtime_plane_projection import RuntimePlaneAxis
from openhcs.core.runtime_tabular_values import measurement_row_mapping
from openhcs.core.runtime_stores import StoredRuntimeValue
from openhcs.core.source_binding_workspace import (
    SourceBindingWorkspaceMaterialization,
    materialize_source_binding_workspace,
)
from openhcs.core.source_bindings import (
    ComponentSelector,
    LazySourceBindingsConfig,
    NamedSourceBinding,
    SourceBindingOrigin,
    SourceFilterClause,
    SourceFilterMatchType,
    SourceFilterSubject,
    SourceSelector,
    source_bindings_defaults_to_base,
)
from openhcs.core.steps.function_step import FunctionStep
from openhcs.interop.cellprofiler.pipeline_import import import_cellprofiler_pipeline
from openhcs.microscopes.source_schema import SourceSchemaFilenameParser
from openhcs.processing.backends.cellprofiler import align
from openhcs.runtime.zmq_execution_client import (
    OpenHCSExecutionSubmission,
    ZMQExecutionClient,
)
from openhcs.runtime.zmq_execution_observation import (
    ZMQRuntimeExecutionObservationExport,
)
from openhcs.demo.synthetic_data import (
    SyntheticMicroscopyGenerator,
)


def _materialize_imported_sources(
    *,
    cppipe_path: Path,
    source_root: Path,
    workspace_root: Path,
    pipeline_config: PipelineConfig,
) -> SourceBindingWorkspaceMaterialization | None:
    source_bindings = source_bindings_defaults_to_base(
        pipeline_config.source_bindings_config
    ).resolved_imported_metadata_locations(
        source_root,
        portable_roots=(cppipe_path.parent,),
    )
    if source_bindings.is_empty:
        return None
    filemanager = FileManager(_create_storage_registry())
    return materialize_source_binding_workspace(
        source_root,
        workspace_root,
        source_bindings,
        filemanager=filemanager,
        source_backend=Backend.DISK,
        workspace_backend=Backend.DISK,
        source_files=tuple(
            filemanager.list_files(
                source_root,
                Backend.DISK.value,
                recursive=True,
            )
        ),
        parser=SourceSchemaFilenameParser(),
    )


def _execute_imported_cppipe_via_zmq(
    tmp_path: Path,
    *,
    cppipe_path: Path,
    source_root: Path,
    microscope: Microscope = Microscope.AUTO,
    well_filter: tuple[str, ...] | int | None = None,
    materialize_runtime_artifacts: bool = True,
) -> tuple[
    SourceBindingWorkspaceMaterialization | None,
    ZMQRuntimeExecutionObservationExport,
]:
    pipeline_steps, imported_config = import_cellprofiler_pipeline(
        cppipe_path,
        source_root=source_root,
    )
    assert pipeline_steps

    workspace = _materialize_imported_sources(
        cppipe_path=cppipe_path,
        source_root=source_root,
        workspace_root=tmp_path / f"{cppipe_path.stem}_source_workspace",
        pipeline_config=imported_config,
    )
    execution_plate_path = (
        source_root if workspace is None else workspace.workspace_root
    )
    global_config = GlobalPipelineConfig(
        num_workers=1,
        use_threading=False,
        microscope=microscope,
        materialize_runtime_artifacts=materialize_runtime_artifacts,
        analysis_consolidation_config=AnalysisConsolidationConfig(enabled=False),
    )
    ensure_global_config_context(GlobalPipelineConfig, global_config)
    pipeline_config = replace_raw(
        imported_config,
        path_planning_config=LazyPathPlanningConfig(
            global_output_folder=tmp_path / "zmq_outputs",
            output_dir_suffix="_imported_cppipe_zmq",
        ),
        vfs_config=LazyVFSConfig(
            materialization_backend=MaterializationBackend.DISK,
        ),
        well_filter_config=(
            LazyWellFilterConfig()
            if well_filter is None
            else LazyWellFilterConfig(
                well_filter=(
                    well_filter if isinstance(well_filter, int) else list(well_filter)
                )
            )
        ),
    )
    observation_path = tmp_path / f"{cppipe_path.stem}_runtime_observation.pkl"
    config_params = {
        "runtime_observation_export_path": str(observation_path),
    }
    submission = OpenHCSExecutionSubmission(
        plate_id=source_root,
        execution_plate_id=execution_plate_path,
        selected_pipeline_path=cppipe_path,
        pipeline_document=PipelineDocumentAuthority.from_values(
            pipeline_config=pipeline_config, pipeline_steps=pipeline_steps
        ),
        global_config=global_config,
        config_params=config_params,
    )
    assert submission.pipeline_code() == PipelineDocumentAuthority.render(
        submission.pipeline_document
    )

    client = ZMQExecutionClient(
        port=18000 + os.getpid() % 20000,
        persistent=False,
    )
    try:
        assert client.connect(timeout=30)
        compile_response = ExecutionSubmissionResponse.from_wire(
            client.submit_compile(submission)
        )
        compile_id = compile_response.require_execution_id(
            "CellProfiler integration compilation"
        )
        ExecutionWaitResult.from_wire(
            client.wait_for_completion(compile_id)
        ).require_complete("CellProfiler integration compilation")

        execution_submission = OpenHCSExecutionSubmission(
            plate_id=source_root,
            execution_plate_id=execution_plate_path,
            selected_pipeline_path=cppipe_path,
            pipeline_document=PipelineDocumentAuthority.from_values(
                pipeline_config=pipeline_config, pipeline_steps=pipeline_steps
            ),
            global_config=global_config,
            config_params=config_params,
            compile_artifact_id=compile_id,
        )
        execution_response = ExecutionSubmissionResponse.from_wire(
            client.submit_pipeline(execution_submission)
        )
        execution_id = execution_response.require_execution_id(
            "CellProfiler integration execution"
        )
        ExecutionWaitResult.from_wire(
            client.wait_for_completion(execution_id)
        ).require_complete("CellProfiler integration execution")
    finally:
        client.disconnect()

    export = ZMQRuntimeExecutionObservationExport.read(observation_path)
    export.require_valid_observation()
    return workspace, export


def _runtime_records(
    export: ZMQRuntimeExecutionObservationExport,
    *,
    name: str | None = None,
    artifact_type: type[ArtifactType] | None = None,
    axis_id: str | None = None,
) -> tuple[StoredRuntimeValue, ...]:
    records = tuple(
        record
        for current_axis, axis_records in export.records_by_axis.items()
        if axis_id is None or current_axis == axis_id
        for record in axis_records
        if name is None or record.key.name == name
        if artifact_type is None or record.key.artifact_type is artifact_type
    )
    return records


def _single_execution_axis(
    export: ZMQRuntimeExecutionObservationExport,
) -> str:
    axis_ids = tuple(str(axis_id) for axis_id in export.records_by_axis)
    assert len(axis_ids) == 1
    return axis_ids[0]


def _runtime_export_has_semantic_record(
    export: ZMQRuntimeExecutionObservationExport,
    *,
    name: str | tuple[str, str],
    artifact_type: type[ArtifactType],
    axis_id: str,
) -> bool:
    if isinstance(name, tuple):
        if artifact_type is not RelationshipsArtifactType:
            raise TypeError(
                "Structured endpoint identity is only valid for relationship records."
            )
        return any(
            isinstance(record.value.data, ObjectRelationship)
            and (
                record.value.data.declaration.source.name,
                record.value.data.declaration.target.name,
            )
            == name
            for record in _runtime_records(
                export,
                artifact_type=artifact_type,
                axis_id=axis_id,
            )
        )
    if _runtime_records(
        export,
        name=name,
        artifact_type=artifact_type,
        axis_id=axis_id,
    ):
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
        for record in _runtime_records(
            export,
            artifact_type=artifact_type,
            axis_id=axis_id,
        )
    )


def test_cellprofiler_integration_uses_public_two_stage_zmq_boundary() -> None:
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert "submit_compile" in called_attributes
    assert "submit_pipeline" in called_attributes


def test_invalid_public_cellprofiler_step_fails_during_zmq_compilation(
    tmp_path: Path,
) -> None:
    source_root = _generate_plate(tmp_path / "single_channel_plate")
    selected_pipeline_path = _write_cppipe(tmp_path / "source_config.cppipe")
    pipeline_steps = [
        FunctionStep(
            func=align,
            name="Align",
            processing_config=LazyProcessingConfig(
                input_source=InputSource.PIPELINE_START,
            ),
        )
    ]
    global_config = GlobalPipelineConfig(
        num_workers=1,
        use_threading=False,
        microscope=Microscope.SOURCE_BINDINGS,
        analysis_consolidation_config=AnalysisConsolidationConfig(enabled=False),
    )
    ensure_global_config_context(GlobalPipelineConfig, global_config)
    pipeline_config = replace_raw(
        PipelineConfig(
            microscope=Microscope.SOURCE_BINDINGS,
            source_bindings_config=LazySourceBindingsConfig(
                bindings=(
                    NamedSourceBinding(
                        alias="OnlyImage",
                        selector=SourceSelector(
                            filters=(
                                SourceFilterClause(
                                    subject=SourceFilterSubject.FILE,
                                    match_type=SourceFilterMatchType.CONTAINS,
                                    value="w1",
                                ),
                            ),
                        ),
                        origin=SourceBindingOrigin.PIPELINE_START,
                        component_identity=(
                            ComponentSelector(AllComponents.CHANNEL, "1"),
                        ),
                    ),
                ),
            ),
        ),
        path_planning_config=LazyPathPlanningConfig(
            global_output_folder=tmp_path / "invalid_compile_outputs",
        ),
        vfs_config=LazyVFSConfig(
            materialization_backend=MaterializationBackend.DISK,
        ),
    )
    observation_path = tmp_path / "invalid_compile_observation.pkl"
    submission = OpenHCSExecutionSubmission(
        plate_id=source_root,
        execution_plate_id=source_root,
        selected_pipeline_path=selected_pipeline_path,
        pipeline_document=PipelineDocumentAuthority.from_values(
            pipeline_config=pipeline_config, pipeline_steps=pipeline_steps
        ),
        global_config=global_config,
        config_params={"runtime_observation_export_path": str(observation_path)},
    )
    assert submission.pipeline_code() == PipelineDocumentAuthority.render(
        submission.pipeline_document
    )

    client = ZMQExecutionClient(
        port=18000 + os.getpid() % 20000,
        persistent=False,
    )
    try:
        assert client.connect(timeout=30)
        compile_response = ExecutionSubmissionResponse.from_wire(
            client.submit_compile(submission)
        )
        compile_id = compile_response.require_execution_id(
            "Invalid CellProfiler integration compilation"
        )
        wait_result = ExecutionWaitResult.from_wire(
            client.wait_for_completion(compile_id)
        )
    finally:
        client.disconnect()

    assert not wait_result.complete
    diagnostic = wait_result.diagnostic.require_text(
        "Invalid CellProfiler integration compilation"
    )
    assert "step 0" in diagnostic
    assert "Align" in diagnostic
    assert "AlignModule" in diagnostic
    assert "cannot reconstruct an exact module block" in diagnostic
    assert not observation_path.exists()


def test_cppipe_import_returns_only_public_openhcs_declarations(
    tmp_path: Path,
) -> None:
    cppipe_path = _write_bbbc021_cppipe(tmp_path / "public_import.cppipe")

    pipeline_steps, pipeline_config = import_cellprofiler_pipeline(cppipe_path)

    assert pipeline_steps
    assert all(isinstance(step, FunctionStep) for step in pipeline_steps)
    assert isinstance(pipeline_config, PipelineConfig)
    source = FunctionStepTransportAuthority.source_from_pipeline(pipeline_steps)
    assert "pipeline_steps = [" in source
    assert "runtime_pipeline" not in source


def test_cppipe_import_executes_through_canonical_zmq_path(
    tmp_path: Path,
) -> None:
    plate_path = _generate_plate(tmp_path / "plate")
    cppipe_path = _write_cppipe(tmp_path / "identify_primary_objects.cppipe")

    _, export = _execute_imported_cppipe_via_zmq(
        tmp_path,
        cppipe_path=cppipe_path,
        source_root=plate_path,
    )

    assert export.axis_count == 1
    nuclei_records = _runtime_records(
        export,
        name="Nuclei",
        artifact_type=ObjectLabelsArtifactType,
        axis_id="A01",
    )
    assert len(nuclei_records) == 1
    assert object_label_dense_array(nuclei_records[0].value.data).max() > 0


def test_bbbc021_cppipe_executes_named_channel_bindings_through_zmq(
    tmp_path: Path,
) -> None:
    plate_path = _generate_bbbc021_plate(tmp_path / "Week1_22123")
    cppipe_path = _write_bbbc021_cppipe(tmp_path / "bbbc021_multichannel.cppipe")

    _, export = _execute_imported_cppipe_via_zmq(
        tmp_path,
        cppipe_path=cppipe_path,
        source_root=plate_path,
        microscope=Microscope.SOURCE_BINDINGS,
    )

    nuclei_records = _runtime_records(
        export,
        name="Nuclei",
        artifact_type=ObjectLabelsArtifactType,
        axis_id="A01",
    )
    composite_records = _runtime_records(
        export,
        name="Composite",
        artifact_type=ImageArtifactType,
        axis_id="A01",
    )
    assert len(nuclei_records) == 1
    assert object_label_dense_array(nuclei_records[0].value.data).max() > 0
    assert len(composite_records) == 1


def test_bbbc021_canonical_illum_cppipe_materializes_declared_images_through_zmq(
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

    _, export = _execute_imported_cppipe_via_zmq(
        tmp_path,
        cppipe_path=cppipe_path,
        source_root=plate_path,
        microscope=Microscope.SOURCE_BINDINGS,
    )

    image_names = {path.name for path in export.exports.image_outputs}
    assert image_names == {
        "fields_IllumActin.npy",
        "fields_IllumActinAvg.npy",
        "fields_IllumDAPI.npy",
        "fields_IllumDAPIAvg.npy",
        "fields_IllumTubulin.npy",
        "fields_IllumTubulinAvg.npy",
    }


def test_loadimages_cppipe_preserves_source_artifact_bindings_through_zmq(
    tmp_path: Path,
) -> None:
    plate_path = _generate_loadimages_mat_illum_plate(tmp_path / "mat_illum_plate")
    cppipe_path = _write_loadimages_mat_illum_cppipe(
        tmp_path / "loadimages_mat_illum.cppipe"
    )
    _, imported_config = import_cellprofiler_pipeline(cppipe_path)
    source_bindings = source_bindings_defaults_to_base(
        imported_config.source_bindings_config
    )
    raw_assignment = source_bindings.binding_for_alias("Raw")
    illum_assignment = source_bindings.binding_for_alias("Illum")
    assert raw_assignment is not None
    assert illum_assignment is not None
    assert raw_assignment.origin is SourceBindingOrigin.PIPELINE_START
    assert illum_assignment.origin is SourceBindingOrigin.PIPELINE_START

    _, export = _execute_imported_cppipe_via_zmq(
        tmp_path,
        cppipe_path=cppipe_path,
        source_root=plate_path,
        microscope=Microscope.IMAGEXPRESS,
    )

    corrected_records = _runtime_records(
        export,
        name="CorrectedRaw",
        artifact_type=ImageArtifactType,
        axis_id="A01",
    )
    assert len(corrected_records) == 1
    assert np.asarray(corrected_records[0].value.data).shape[-2:] == (64, 64)


def test_legacy_examplefly_load_data_cppipe_fails_at_nominal_import_boundary() -> None:
    cppipe_path = (
        Path(__file__).resolve().parents[2]
        / "benchmark"
        / "cellprofiler_pipelines"
        / "ExampleFly.cppipe"
    )

    with pytest.raises(
        KeyError,
        match="No CellProfiler module declaration.*LoadData",
    ):
        import_cellprofiler_pipeline(cppipe_path)


def test_cppipe_relationship_outputs_are_compiler_derived_over_zmq(
    tmp_path: Path,
) -> None:
    plate_path = _generate_plate(tmp_path / "relationship_plate")
    cppipe_path = _write_relationship_cppipe(tmp_path / "relate_objects.cppipe")

    _, export = _execute_imported_cppipe_via_zmq(
        tmp_path,
        cppipe_path=cppipe_path,
        source_root=plate_path,
    )

    relationship_records = _runtime_records(
        export,
        artifact_type=RelationshipsArtifactType,
        axis_id="A01",
    )
    measurement_records = _runtime_records(
        export,
        artifact_type=MeasurementsArtifactType,
        axis_id="A01",
    )
    assert relationship_records
    assert measurement_records
    relationship = cast(ObjectRelationship, relationship_records[0].value.data)
    assert relationship.declaration.source.name == "Nuclei"
    assert relationship.declaration.target.name == "Cells"
    assert relationship.declaration.relationship_type == "Parent"

    headers_by_name = {
        path.name: list(export.exports.table_headers_by_path[path])
        for path in export.exports.table_outputs
    }
    assert _matching_header(headers_by_name, "relationships") == [
        "relationship_type",
        "source_role",
        "target_role",
        "source_object",
        "target_object",
        "producer_module_number",
        "parent_id",
        "child_id",
        "image_number",
        "slice_count",
    ]
    assert _matching_header(headers_by_name, "Image.csv") == [
        "image_number",
        "Count_Nuclei",
        "Threshold_FinalThreshold_Nuclei",
        "Threshold_OrigThreshold_Nuclei",
        "Threshold_WeightedVariance_Nuclei",
        "Threshold_SumOfEntropies_Nuclei",
        "Count_Cells",
        "Threshold_FinalThreshold_Cells",
        "Threshold_OrigThreshold_Cells",
        "Threshold_WeightedVariance_Cells",
        "Threshold_SumOfEntropies_Cells",
    ]


def test_percent_positive_cppipe_executes_measurement_consumers_over_zmq(
    tmp_path: Path,
) -> None:
    source_root = _generate_percent_positive_source_folder(
        tmp_path / "ExamplePercentPositive"
    )
    cppipe_path = _write_percent_positive_cppipe(tmp_path / "percent_positive.cppipe")

    _, export = _execute_imported_cppipe_via_zmq(
        tmp_path,
        cppipe_path=cppipe_path,
        source_root=source_root,
        well_filter=1,
    )

    filtered_object_records = _runtime_records(
        export,
        name="PH3PosNuclei",
        artifact_type=ObjectLabelsArtifactType,
    )
    assert len(filtered_object_records) == 1
    assert _runtime_records(
        export,
        name="DisplayImage",
        artifact_type=ImageArtifactType,
    )
    calculate_math_records = tuple(
        record
        for record in _runtime_records(
            export,
            artifact_type=MeasurementsArtifactType,
        )
        if isinstance(record.value.data, MeasurementTable)
        if any(
            row_mapping["output_name"] == "PercentPositive"
            for row in record.value.data.rows.iter_row_mappings()
            for row_mapping in (measurement_row_mapping(row),)
            if "output_name" in row_mapping
        )
    )
    assert calculate_math_records
    calculate_math_table = cast(
        MeasurementTable,
        calculate_math_records[0].value.data,
    )
    calculate_math_rows = tuple(
        measurement_row_mapping(row)
        for row in calculate_math_table.rows.iter_row_mappings()
    )
    assert calculate_math_rows[0]["output_name"] == "PercentPositive"


def test_examplehuman_cppipe_executes_source_bound_objects_over_zmq(
    tmp_path: Path,
) -> None:
    source_root = _generate_examplehuman_source_folder(tmp_path / "ExampleHuman")
    cppipe_path = (
        Path(__file__).resolve().parents[2]
        / "benchmark"
        / "cellprofiler_pipelines"
        / "ExampleHuman.cppipe"
    )

    workspace, export = _execute_imported_cppipe_via_zmq(
        tmp_path,
        cppipe_path=cppipe_path,
        source_root=source_root,
        well_filter=1,
    )

    assert workspace is not None
    axis_id = _single_execution_axis(export)
    cytoplasm_records = _runtime_records(
        export,
        name="Cytoplasm",
        artifact_type=ObjectLabelsArtifactType,
        axis_id=axis_id,
    )
    assert len(cytoplasm_records) == 1
    cytoplasm = object_label_dense_array(cytoplasm_records[0].value.data)
    assert cytoplasm.ndim == 2 or cytoplasm.ndim == 3 and cytoplasm.shape[0] == 1
    assert _runtime_export_has_semantic_record(
        export,
        name="MeasureObjectIntensity_10_measurements",
        artifact_type=MeasurementsArtifactType,
        axis_id=axis_id,
    )


def test_official_untangleworms_preserves_overlay_shape_over_zmq(
    tmp_path: Path,
) -> None:
    _, export = _execute_official_cellprofiler3_pipeline(
        tmp_path,
        "ExampleUntangleWorms",
    )
    axis_id = _single_execution_axis(export)

    assert _runtime_records(
        export,
        name="OverlappingWorms",
        artifact_type=ObjectLabelsArtifactType,
        axis_id=axis_id,
    )
    assert _runtime_records(
        export,
        name="NonOverlappingWorms",
        artifact_type=ObjectLabelsArtifactType,
        axis_id=axis_id,
    )
    overlay_records = _runtime_records(
        export,
        name="OrigOverlay",
        artifact_type=ImageArtifactType,
        axis_id=axis_id,
    )
    assert len(overlay_records) == 1
    overlay = np.asarray(overlay_records[0].value.data)
    assert overlay.ndim == 4
    assert overlay.shape[0] == 2
    assert overlay.shape[-1] == 3


def test_official_untangleworms_brightfield_preserves_overlay_pixels_over_zmq(
    tmp_path: Path,
) -> None:
    _, export = _execute_official_cellprofiler3_pipeline(
        tmp_path,
        "ExampleUntangleWormsBrightField",
    )

    overlay_outputs = sorted(
        path
        for output_root in export.output_roots
        for path in output_root.rglob("*OrigOverlay.tif")
    )
    assert [path.name for path in overlay_outputs] == [
        "A01_s001_w1_z001_t001_OrigOverlay.tif",
        "A01_s002_w1_z001_t001_OrigOverlay.tif",
    ]
    overlay = np.asarray(tifffile.imread(overlay_outputs[0]))
    assert overlay.dtype == np.float32
    assert overlay.ndim == 3
    red = overlay[..., 0]
    blue = overlay[..., 2]
    assert np.count_nonzero(blue > red + (32 / 255)) > 0


def test_official_cometassay_preserves_mask_geometry_over_zmq(
    tmp_path: Path,
) -> None:
    _, export = _execute_official_cellprofiler3_pipeline(
        tmp_path,
        "ExampleCometAssay",
    )

    image_outputs = sorted(export.exports.image_outputs)
    assert [path.name for path in image_outputs] == [
        "A01_s001_w1_z001_t001_CometHeadOutline.png",
        "A01_s002_w1_z001_t001_CometHeadOutline.png",
    ]
    overlay = np.asarray(Image.open(image_outputs[0]))
    assert overlay.shape[:2] == (1040, 1388)
    assert overlay.ndim == 3


def test_official_colocalization_preserves_relationships_and_configured_export(
    tmp_path: Path,
) -> None:
    _, export = _execute_official_cellprofiler3_pipeline(
        tmp_path,
        "ExampleColocalization",
        materialize_runtime_artifacts=False,
    )
    axis_id = _single_execution_axis(export)
    object_records = _runtime_records(
        export,
        artifact_type=ObjectLabelsArtifactType,
        axis_id=axis_id,
    )
    object_records_by_name = {
        name: tuple(record for record in object_records if record.key.name == name)
        for name in ("Objects1", "Objects2", "ColocalizedRegion")
    }
    assert {
        name: tuple(
            (
                record.key.scope.component,
                record.key.scope.value_text,
                object_label_dense_array(record.value.data).shape,
                cast(ObjectLabelValue, record.value.data).plane_axis,
            )
            for record in records
        )
        for name, records in object_records_by_name.items()
    } == {
        "Objects1": (
            (
                AllComponents.CHANNEL,
                "1",
                (2, 1040, 1392),
                RuntimePlaneAxis.RUNTIME_SLICE,
            ),
        ),
        "Objects2": (
            (
                AllComponents.CHANNEL,
                "2",
                (2, 1040, 1392),
                RuntimePlaneAxis.RUNTIME_SLICE,
            ),
        ),
        "ColocalizedRegion": (
            (
                AllComponents.CHANNEL,
                "1",
                (2, 1040, 1392),
                RuntimePlaneAxis.RUNTIME_SLICE,
            ),
        ),
    }
    relationship_records = _runtime_records(
        export,
        artifact_type=RelationshipsArtifactType,
        axis_id=axis_id,
    )
    assert {
        (
            record.value.data.declaration.source.name,
            record.value.data.declaration.target.name,
        )
        for record in relationship_records
    } == {
        ("Objects1", "Objects2"),
        ("Objects2", "Objects1"),
        ("ExpandedObjects1", "ExpandedObjects2"),
        ("ExpandedObjects2", "ExpandedObjects1"),
    }
    lineage_records = _runtime_records(
        export,
        artifact_type=ObjectLineageArtifactType,
        axis_id=axis_id,
    )
    assert {
        (
            record.value.data.declaration.source.name,
            record.value.data.declaration.target.name,
        )
        for record in lineage_records
    } == {
        ("Objects1", "ColocalizedObjects"),
        ("Objects1", "ColocalizedRegion"),
    }
    relationships = tuple(
        cast(ObjectRelationship, record.value.data) for record in relationship_records
    )
    assert {
        (
            relationship.declaration.source_role,
            relationship.declaration.target_role,
        )
        for relationship in relationships
    } == {
        ("parent", "child"),
        ("child", "parent"),
    }
    measurement_names = {
        record.key.name
        for record in _runtime_records(
            export,
            artifact_type=MeasurementsArtifactType,
            axis_id=axis_id,
        )
    }
    assert any(
        name.startswith("MeasureColocalization_") and name.endswith("_measurements")
        for name in measurement_names
    )
    assert any(
        name.startswith("CalculateMath_") and name.endswith("_measurements")
        for name in measurement_names
    )
    headers_by_name = {
        path.name: list(export.exports.table_headers_by_path[path])
        for path in export.exports.table_outputs
    }
    assert tuple(headers_by_name) == ("Image.csv",)


def test_official_neighbors_preserves_measurement_and_image_exports_over_zmq(
    tmp_path: Path,
) -> None:
    _, export = _execute_official_cellprofiler3_pipeline(
        tmp_path,
        "ExampleNeighbors",
    )
    axis_id = _single_execution_axis(export)
    cells_records = _runtime_records(
        export,
        name="Cells",
        artifact_type=ObjectLabelsArtifactType,
        axis_id=axis_id,
    )
    assert cells_records
    assert object_label_dense_array(cells_records[0].value.data).max() > 0
    assert _runtime_export_has_semantic_record(
        export,
        name="MeasureObjectNeighbors_6_measurements",
        artifact_type=MeasurementsArtifactType,
        axis_id=axis_id,
    )
    headers_by_name = {
        path.name: list(export.exports.table_headers_by_path[path])
        for path in export.exports.table_outputs
    }
    assert [
        field
        for field in headers_by_name["Cells.csv"]
        if field.startswith("Neighbors_")
    ] == [
        "Neighbors_NumberOfNeighbors_Expanded",
        "Neighbors_PercentTouching_Expanded",
        "Neighbors_FirstClosestObjectNumber_Expanded",
        "Neighbors_FirstClosestDistance_Expanded",
        "Neighbors_SecondClosestObjectNumber_Expanded",
        "Neighbors_SecondClosestDistance_Expanded",
        "Neighbors_AngleBetweenNeighbors_Expanded",
    ]
    image_names = {path.name for path in export.exports.image_outputs}
    assert image_names == {
        "A01_s001_w1_z001_t001_ColorNeighbors.png",
        "A01_s001_w1_z001_t001_InvertedRed.png",
    }


def test_official_illumination_preserves_rule_row_binding_over_zmq(
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
            "Official CellProfiler illumination example is not available under "
            f"{examples_root}."
        )

    _, pipeline_config = import_cellprofiler_pipeline(cppipe_path)
    source_bindings = source_bindings_defaults_to_base(
        pipeline_config.source_bindings_config
    )
    orig_green = source_bindings.binding_for_alias("OrigGreen")
    assert orig_green is not None
    assert source_bindings.binding_for_alias("DNA") is None
    assert orig_green.origin is SourceBindingOrigin.PIPELINE_START
    assert orig_green.component_identity == (
        ComponentSelector(AllComponents.CHANNEL, "1"),
    )
    assert len(orig_green.selector.filters) == 1
    source_filter = orig_green.selector.filters[0]
    assert source_filter.subject is SourceFilterSubject.FILE
    assert source_filter.match_type is SourceFilterMatchType.CONTAINS
    assert source_filter.value == "AS_09047_"

    _, export = _execute_imported_cppipe_via_zmq(
        tmp_path,
        cppipe_path=cppipe_path,
        source_root=source_root,
        well_filter=1,
    )
    assert tuple(path.name for path in export.exports.image_outputs) == ("Illum.npy",)
    illumination = np.load(export.exports.image_outputs[0])
    assert illumination.shape == (512, 512)
    assert illumination.dtype == np.float32
    assert np.isfinite(illumination).all()
    assert float(illumination.max()) > float(illumination.min())


def test_official_woundhealing_preserves_area_table_shape_over_zmq(
    tmp_path: Path,
) -> None:
    _, export = _execute_official_cellprofiler3_pipeline(
        tmp_path,
        "ExampleWoundHealing",
    )
    headers_by_name = {
        path.name: list(export.exports.table_headers_by_path[path])
        for path in export.exports.table_outputs
    }
    assert [
        field
        for field in headers_by_name["Image.csv"]
        if field.startswith("AreaOccupied_")
    ] == [
        "AreaOccupied_AreaOccupied_Tissue",
        "AreaOccupied_Perimeter_Tissue",
        "AreaOccupied_TotalArea_Tissue",
    ]


@pytest.mark.parametrize(
    (
        "pipeline_name",
        "expected_records",
    ),
    (
        pytest.param(
            "ExamplePercentPositive",
            (
                ("PH3PosNuclei", ObjectLabelsArtifactType),
                (("Nuclei", "PH3"), RelationshipsArtifactType),
                ("CalculateMath_13_measurements", MeasurementsArtifactType),
                ("DisplayImage", ImageArtifactType),
            ),
            id="percent-positive",
        ),
        pytest.param(
            "ExampleSpeckles",
            (
                ("h2ax", ObjectLabelsArtifactType),
                (("Nuclei", "h2ax"), RelationshipsArtifactType),
                ("MeasureObjectIntensity_10_measurements", MeasurementsArtifactType),
            ),
            id="speckles",
        ),
        pytest.param(
            "ExampleTumor",
            (
                ("tumor", ObjectLabelsArtifactType),
                ("TumorOutline", ImageArtifactType),
                ("MeasureObjectSizeShape_8_measurements", MeasurementsArtifactType),
            ),
            id="tumor",
        ),
        pytest.param(
            "ExampleUntangleAndStraightenWorms",
            (
                ("StraightenedWorms", ObjectLabelsArtifactType),
                (
                    ("NonOverlappingWorms", "HeadMarkers"),
                    RelationshipsArtifactType,
                ),
                ("StraightenWorms_11_measurements", MeasurementsArtifactType),
                ("Straightened_GFP", ImageArtifactType),
            ),
            id="untangle-and-straighten",
        ),
        pytest.param(
            "ExampleYeastColonies",
            (
                ("Colonies", ObjectLabelsArtifactType),
                ("OutlinedColonies", ImageArtifactType),
                ("ClassifyObjects_18_measurements", MeasurementsArtifactType),
            ),
            id="yeast-colonies",
        ),
        pytest.param(
            "ExampleYeastPatches",
            (
                ("Prespots", ObjectLabelsArtifactType),
                ("FilterObjects", ObjectLabelsArtifactType),
                ("NaturalSpots", ObjectLabelsArtifactType),
                ("ForcedSpots", ObjectLabelsArtifactType),
                ("Grid", SpatialGridArtifactType),
                ("MeasureObjectIntensity_18_measurements", MeasurementsArtifactType),
            ),
            id="yeast-patches-grid-illumination",
        ),
        pytest.param(
            "ExampleImagingFlowCytometryObjectsInGrid",
            (
                ("BF_cells_on_grid", ObjectLabelsArtifactType),
                (
                    ("Non_empty_tile", "FilteredBF"),
                    RelationshipsArtifactType,
                ),
                ("MeasureGranularity_24_measurements", MeasurementsArtifactType),
                ("MeasureTexture_25_measurements", MeasurementsArtifactType),
                (
                    "MeasureObjectIntensityDistribution_30_measurements",
                    MeasurementsArtifactType,
                ),
            ),
            id="imaging-flow-cytometry-grid",
        ),
        pytest.param(
            "ExampleTrackObjects",
            (
                ("TrackedCells", ImageArtifactType),
                ("TrackObjects_9_measurements", MeasurementsArtifactType),
                ("OutlineImage", ImageArtifactType),
                ("AdjacentImage", ImageArtifactType),
            ),
            id="track-objects",
        ),
        pytest.param(
            "ExampleVitra",
            (
                ("CorrProtein", ImageArtifactType),
                ("Cells", ObjectLabelsArtifactType),
                ("Cytoplasm", ObjectLabelsArtifactType),
                ("Outlined", ImageArtifactType),
                ("MeasureObjectIntensity_9_measurements", MeasurementsArtifactType),
                ("CalculateMath_10_measurements", MeasurementsArtifactType),
                ("CalculateMath_11_measurements", MeasurementsArtifactType),
            ),
            id="vitra-npy-illumination",
        ),
    ),
)
def test_official_cellprofiler3_representative_pipelines_execute_over_zmq(
    tmp_path: Path,
    pipeline_name: str,
    expected_records: tuple[tuple[str | tuple[str, str], type[ArtifactType]], ...],
) -> None:
    workspace, export = _execute_official_cellprofiler3_pipeline(
        tmp_path,
        pipeline_name,
    )
    assert workspace is not None
    axis_id = _single_execution_axis(export)
    for name, kind in expected_records:
        assert _runtime_export_has_semantic_record(
            export,
            name=name,
            artifact_type=kind,
            axis_id=axis_id,
        )


def test_official_cellprofiler3_cppipe_corpus_imports_public_declarations() -> None:
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
    cases_by_name = {case.name: case for case in _official_cellprofiler3_cases()}
    missing_cases = tuple(
        cppipe_path.stem
        for cppipe_path in cppipe_paths
        if cppipe_path.stem not in cases_by_name
    )
    assert not missing_cases
    for discovered_path in cppipe_paths:
        case = cases_by_name[discovered_path.stem]
        try:
            pipeline_steps, pipeline_config = import_cellprofiler_pipeline(
                case.cppipe_path,
                source_root=case.dataset_path,
            )
            source = FunctionStepTransportAuthority.source_from_pipeline(pipeline_steps)
        except Exception as exc:
            failures.append(f"{discovered_path.name}: {type(exc).__name__}: {exc}")
            continue
        assert pipeline_steps
        assert isinstance(pipeline_config, PipelineConfig)
        assert "pipeline_steps = [" in source

    assert not failures


@pytest.mark.parametrize(
    "pipeline_name",
    ("ExampleFly", "ExampleYeastColonies"),
)
def test_official_classify_repeated_rows_import_as_one_object_role(
    pipeline_name: str,
) -> None:
    cppipe_path = (
        _official_cellprofiler_examples_root()
        / "CellProfiler3Pipelines"
        / f"{pipeline_name}.cppipe"
    )
    if not cppipe_path.exists():
        pytest.skip(f"Official corpus pipeline is unavailable: {cppipe_path}")

    pipeline_steps, pipeline_config = import_cellprofiler_pipeline(cppipe_path)

    assert isinstance(pipeline_config, PipelineConfig)
    assert sum(step.name == "ClassifyObjects" for step in pipeline_steps) == 1


def test_official_cellprofiler3_cppipe_corpus_executes_over_zmq_when_enabled(
    tmp_path: Path,
) -> None:
    exhaustive_execution_env = "OPENHCS_RUN_OFFICIAL_CELLPROFILER3_CORPUS_EXECUTION"
    if os.environ.get(exhaustive_execution_env) != "1":
        pytest.skip(
            "Official corpus execution is intentionally opt-in because it runs "
            "every discovered CellProfiler3 .cppipe. Set "
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
            _, export = _execute_official_cellprofiler3_pipeline(
                tmp_path / pipeline_name,
                pipeline_name,
            )
        except Exception as exc:
            failures.append(f"{pipeline_name}: {type(exc).__name__}: {exc}")
            continue
        if export.execution_failures():
            failures.append(f"{pipeline_name}: {export.execution_failures()!r}")

    assert not failures


def _execute_official_cellprofiler3_pipeline(
    tmp_path: Path,
    pipeline_name: str,
    *,
    materialize_runtime_artifacts: bool = True,
) -> tuple[
    SourceBindingWorkspaceMaterialization | None,
    ZMQRuntimeExecutionObservationExport,
]:
    case = _official_cellprofiler3_case(pipeline_name)
    cppipe_path = case.cppipe_path
    source_root = case.dataset_path
    if not cppipe_path.exists() or not source_root.exists():
        pytest.skip(
            f"Official CellProfiler {pipeline_name} files are not available. "
            f"Manifest paths: cppipe={cppipe_path}, source={source_root}."
        )
    tmp_path.mkdir(parents=True, exist_ok=True)
    return _execute_imported_cppipe_via_zmq(
        tmp_path,
        cppipe_path=cppipe_path,
        source_root=source_root,
        well_filter=1,
        materialize_runtime_artifacts=materialize_runtime_artifacts,
    )


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


def _official_cellprofiler_examples_root() -> Path:
    return resolve_benchmark_path_root(
        BenchmarkPathRootKind.CELLPROFILER_EXAMPLES,
        env_name=CELLPROFILER_EXAMPLES_ROOT_ENV,
    )


@lru_cache(maxsize=1)
def _official_cellprofiler3_cases() -> tuple[CellProfilerComparisonCase, ...]:
    """Load the canonical pipeline and source roots for official cases."""

    return load_comparison_cases(
        Path(__file__).parents[2]
        / "benchmark"
        / "manifests"
        / "official30_portable_axis1.json"
    )


def _official_cellprofiler3_case(
    pipeline_name: str,
) -> CellProfilerComparisonCase:
    """Return one exact manifest-owned official case."""

    matches = tuple(
        case for case in _official_cellprofiler3_cases() if case.name == pipeline_name
    )
    if len(matches) != 1:
        raise ValueError(
            f"Official manifest requires one {pipeline_name!r} case, got {matches!r}."
        )
    return matches[0]


def _matching_header(
    headers_by_name: dict[str, list[str]],
    name_fragment: str,
) -> list[str]:
    for filename, header in headers_by_name.items():
        if name_fragment in filename:
            return header
    raise AssertionError(
        f"No CSV output filename contained {name_fragment!r}: {sorted(headers_by_name)}"
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
                    "LoadImages:[module_num:1|svn_version:'Unknown'|"
                    "enabled:True|wants_pause:False]"
                ),
                "    What type of files are you loading?:individual images",
                "    How do you want to load these files?:Text-Exact match",
                "    Do you want to exclude certain files?:No",
                "    Type the text that these images have in common (case-sensitive):w1",
                "    What do you want to call this image in CellProfiler?:OrigBlue",
                "    What is the position of this image in each group?:1",
                (
                    "    Do you want to extract metadata from the file name, "
                    "the subfolder path or both?:None"
                ),
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
                "ModuleCount:5",
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
                "    Extract metadata?:Yes",
                "    Metadata extraction method:Extract from file/folder names",
                "    Metadata source:File name",
                "    Extract metadata from:All images",
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
                "    Image set matching method:Order",
                "    Assignments count:2",
                "    Single images count:0",
                '    Select the rule criteria:and (file does contain "_w1")',
                "    Name to assign these images:DNA",
                "    Name to assign these objects:UnusedObjects1",
                "    Select the image type:Grayscale image",
                '    Select the rule criteria:and (file does contain "_w2")',
                "    Name to assign these images:Actin",
                "    Name to assign these objects:UnusedObjects2",
                "    Select the image type:Grayscale image",
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
                    "LoadImages:[module_num:1|svn_version:'Unknown'|"
                    "enabled:True|wants_pause:False]"
                ),
                "    What type of files are you loading?:individual images",
                "    How do you want to load these files?:Text-Exact match",
                "    Do you want to exclude certain files?:No",
                "    Type the text that these images have in common (case-sensitive):w1",
                "    What do you want to call this image in CellProfiler?:OrigBlue",
                "    What is the position of this image in each group?:1",
                (
                    "    Do you want to extract metadata from the file name, "
                    "the subfolder path or both?:None"
                ),
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
                    "variable_revision_number:5|enabled:True|wants_pause:False]"
                ),
                "    Parent objects:Nuclei",
                "    Child objects:Cells",
                "    Calculate child-parent distances?:None",
                "    Calculate per-parent means for all child measurements?:No",
                "    Calculate distances to other parents?:No",
                "    Do you want to save the children with parents as a new object set?:No",
                "    Name the output object:None",
                "    Parent name:None",
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
                "    Extract metadata?:Yes",
                "    Metadata extraction method:Extract from file/folder names",
                "    Metadata source:File name",
                "    Extract metadata from:All images",
                (
                    "    Regular expression to extract from file name:"
                    "^(?P<Plate>.*)_(?P<Well>[A-P][0-9]{2})_s"
                    "(?P<Site>[0-9])_w(?P<ChannelNumber>[0-9])"
                ),
                '    Select the filtering criteria:and (file does contain "")',
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
                '    Select the rule criteria:and (file does contain "d0.tif")',
                "    Name to assign these images:OrigBlue",
                "    Name to assign these objects:Cell",
                "    Select the image type:Grayscale image",
                '    Select the rule criteria:and (file does contain "d1.tif")',
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
                    "variable_revision_number:5|enabled:True|wants_pause:False]"
                ),
                "    Parent objects:Nuclei",
                "    Child objects:PH3",
                "    Calculate child-parent distances?:None",
                "    Calculate per-parent means for all child measurements?:No",
                "    Calculate distances to other parents?:No",
                "    Do you want to save the children with parents as a new object set?:No",
                "    Name the output object:None",
                "    Parent name:None",
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
