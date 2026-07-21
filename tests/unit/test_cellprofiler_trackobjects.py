from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactOutputPlan,
    ArtifactSpec,
    ArtifactSpecCollection,
    ImageArtifactType,
    ObjectLabelsArtifactType,
)
from openhcs.core.function_patterns import FunctionInvocationKey, normalize_function_pattern
from openhcs.core.function_step_transport import FunctionStepTransportAuthority
from openhcs.core.invocation_artifacts import ArtifactDeclarationStepContext
from openhcs.core.pipeline.artifact_planning import artifact_producers_for_outputs
from openhcs.core.runtime_measurements import MeasurementRowAxisField, MeasurementRowValueField
from openhcs.core.runtime_object_label_domains import ObjectLabelDomainScope, PresentObjectLabelIdsDomainDeclaration
from openhcs.core.runtime_plane_projection import RuntimePlaneAxis, RuntimePlaneAxisValueProjection
from openhcs.core.runtime_object_labels import (
    ObjectLabelVariantData,
    ObjectLabelPayload,
)
import math
from typing import get_type_hints

import numpy as np
from inspect import unwrap

from openhcs.processing.backends.cellprofiler.tracking import track_objects
from openhcs.constants.constants import MemoryType

from openhcs.processing.backends.cellprofiler.tracking import (
    NumbaNumpyObjectTrackingBackendStrategy,
    ObjectTrackingBackendStrategy,
    TrackObjectsModule,
    TrackingImageMeasurement,
    TrackingObjectMeasurement,
)
from openhcs.core.steps.function_step import FunctionStep
from openhcs.processing.backends.cellprofiler.image_geometry import TileModule, tile


def _measurement_value(rows, *, slice_index, feature_name, object_label=None):
    for row in rows:
        if row.get("slice_index") != slice_index:
            continue
        if row.get(MeasurementRowAxisField.FEATURE_NAME.value) != feature_name:
            continue
        if (
            object_label is not None
            and row.get(MeasurementRowAxisField.OBJECT_LABEL.value) != object_label
        ):
            continue
        return row[MeasurementRowValueField.MEASUREMENT_VALUE.value]
    raise AssertionError(f"missing measurement row {slice_index=} {feature_name=}")


def _projected_measurement_rows(result, *, object_name="Cells"):
    return TrackObjectsModule.MeasurementRows(
        result.tracking_measurements,
        module_type=TrackObjectsModule,
        object_name=object_name,
    ).rows()


def _timepoint_labels(labels: np.ndarray, *, axis_size: int) -> ObjectLabelPayload:
    projection = RuntimePlaneAxisValueProjection.preserve(
        axis=RuntimePlaneAxis.SOURCE_BINDING,
        axis_size=axis_size,
    )
    return ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
        domain=PresentObjectLabelIdsDomainDeclaration(
            scope=ObjectLabelDomainScope.PLANE,
            plane_projection=projection,
        ).declared_domain(None, labels),
    )


def test_public_roundtrip_publishes_default_tracked_image_for_tile() -> None:
    initial_artifacts = (
        ArtifactSpec.output("OrigColor", ImageArtifactType),
        ArtifactSpec.output("OutlineImage", ImageArtifactType),
        ArtifactSpec.output("Embryos", ObjectLabelsArtifactType),
    )
    output_parameter = (
        TrackObjectsModule.output_image_binding.require_parameter_name()
    )
    track_step = FunctionStep(
        func=(
            track_objects,
            {
                TrackObjectsModule.tracked_objects_binding.require_parameter_name(): (
                    "Embryos"
                ),
                "save_color_coded_image": True,
                output_parameter: TrackObjectsModule.default_output_image_name,
            },
        ),
        name="TrackObjects",
    )

    source = FunctionStepTransportAuthority.source_from_pipeline([track_step])
    assert output_parameter not in source
    namespace: dict[str, object] = {}
    exec(compile(source, "<track-objects-public>", "exec"), namespace)
    (restored_step,) = FunctionStepTransportAuthority.pipeline_steps_from_namespace(
        namespace
    )
    track_invocation = next(
        normalize_function_pattern(restored_step.func).iter_items()
    )
    assert output_parameter not in track_invocation.kwargs_dict

    producer_key = FunctionInvocationKey(
        "fixture_producer",
        track_invocation.key.group_key,
        0,
    )
    track_context = ArtifactDeclarationStepContext(
        step_name="TrackObjects",
        step_index=4,
        available_artifacts=ArtifactSpecCollection(initial_artifacts),
        main_flow_artifacts=ArtifactSpecCollection(
            spec.for_plan_type(ArtifactInputPlan) for spec in initial_artifacts
        ),
        available_artifact_producers=artifact_producers_for_outputs(
            initial_artifacts,
            groups=(None,),
            invocation_keys=(producer_key,),
        ),
    )
    track_blocks, consumed = TrackObjectsModule.module_blocks_for_invocation(
        invocation=track_invocation,
        step_context=track_context,
    )
    (track_blocks,), _next_module_num = (
        TrackObjectsModule.number_step_invocation_blocks(
            (track_blocks,),
            first_module_num=5,
        )
    )
    track_contract, _consumed = TrackObjectsModule.invocation_callable_contract(
        invocation=track_invocation,
        numbered_module_blocks=track_blocks,
        consumed_kwarg_names=consumed,
        step_context=track_context,
    )
    tracked_output = (
        track_contract.artifact_outputs.require_by_name_and_artifact_type(
            TrackObjectsModule.default_output_image_name,
            ImageArtifactType,
        )
    )
    assert tracked_output.plan_type is ArtifactOutputPlan

    available_artifacts = (*initial_artifacts, *track_contract.artifact_outputs.specs)
    tile_step = FunctionStep(
        func=(
            tile,
            {
                TileModule.input_image_binding.require_parameter_name(): "OrigColor",
                TileModule.additional_image_binding.require_parameter_name(): (
                    "OutlineImage",
                    TrackObjectsModule.default_output_image_name,
                ),
                TileModule.output_image_binding.require_parameter_name(): "TiledImage",
            },
        ),
        name="Tile",
    )
    tile_invocation = next(normalize_function_pattern(tile_step.func).iter_items())
    tile_context = ArtifactDeclarationStepContext(
        step_name="Tile",
        step_index=6,
        available_artifacts=ArtifactSpecCollection(available_artifacts),
        main_flow_artifacts=ArtifactSpecCollection(
            spec.for_plan_type(ArtifactInputPlan) for spec in available_artifacts
        ),
        available_artifact_producers=(
            *track_context.available_artifact_producers,
            *artifact_producers_for_outputs(
                track_contract.artifact_outputs.for_plan_type(
                    ArtifactOutputPlan
                ).specs,
                groups=(None,),
                invocation_keys=(track_invocation.key,),
            ),
        ),
    )
    tile_blocks, tile_consumed = TileModule.module_blocks_for_invocation(
        invocation=tile_invocation,
        step_context=tile_context,
    )
    (tile_blocks,), _next_module_num = TileModule.number_step_invocation_blocks(
        (tile_blocks,),
        first_module_num=7,
    )
    tile_contract, _consumed = TileModule.invocation_callable_contract(
        invocation=tile_invocation,
        numbered_module_blocks=tile_blocks,
        consumed_kwarg_names=tile_consumed,
        step_context=tile_context,
    )
    tracked_input = tile_contract.artifact_inputs.require_by_name_and_artifact_type(
        TrackObjectsModule.default_output_image_name,
        ImageArtifactType,
    )

    assert (
        tracked_output.ref().for_plan_type(ArtifactInputPlan) == tracked_input.ref()
    )


def test_track_objects_uses_numba_tracking_backend_by_default():
    assert type(ObjectTrackingBackendStrategy.for_memory_type(MemoryType.NUMPY)) is (
        NumbaNumpyObjectTrackingBackendStrategy
    )


def test_tracking_backend_overlap_transition_counts_preserves_sparse_ids():
    previous_labels = np.array([[2, 2, 5, 8, 11, 13, 0, 0]], dtype=np.int32)
    current_labels = np.array([[3, 3, 7, 7, 0, 0, 9, 9]], dtype=np.int32)
    previous_track_labels = np.zeros(13, dtype=np.int32)
    previous_track_labels[[1, 4, 7, 10, 12]] = [101, 202, 303, 404, 505]
    current_track_labels = np.zeros(9, dtype=np.int32)
    current_track_labels[[2, 6, 8]] = [101, 202, 404]

    backend = ObjectTrackingBackendStrategy.for_memory_type(MemoryType.NUMPY)

    assert backend.overlap_transition_counts(
        previous_labels,
        current_labels,
        previous_track_labels,
        current_track_labels,
    ) == (1, 1)


def test_tracking_backend_overlap_transition_counts_handles_empty_frames():
    backend = ObjectTrackingBackendStrategy.for_memory_type(MemoryType.NUMPY)
    empty = np.zeros((1, 4), dtype=np.int32)
    previous_labels = np.array([[0, 2, 0, 5]], dtype=np.int32)

    assert backend.overlap_transition_counts(
        empty,
        previous_labels,
        np.zeros(0, dtype=np.int32),
        np.zeros(5, dtype=np.int32),
    ) == (0, 0)
    assert backend.overlap_transition_counts(
        previous_labels,
        empty,
        np.zeros(5, dtype=np.int32),
        np.zeros(0, dtype=np.int32),
    ) == (2, 0)


def test_track_objects_declares_native_scalar_fields():
    annotations = get_type_hints(TrackingObjectMeasurement, include_extras=True)

    assert annotations["final_age"] is int
    assert annotations["parent_image_number"] is int
    assert annotations["trajectory_x"] is float
    assert annotations["trajectory_y"] is float


def test_track_objects_emits_stack_tracking_measurements():
    labels = np.zeros((2, 5, 5), dtype=np.int32)
    labels[0, 1:3, 1:3] = 1
    labels[1, 1:3, 2:4] = 1
    image = np.zeros(labels.shape, dtype=np.float32)

    result = unwrap(track_objects)(
        image,
        labels=_timepoint_labels(labels, axis_size=2),
        tracking_method="overlap",
        pixel_radius=50,
    )
    output = result.output_image
    assert tuple(
        batch.row_type for batch in result.tracking_measurements.row_batches
    ) == (TrackingObjectMeasurement, TrackingImageMeasurement)
    rows = _projected_measurement_rows(result)

    np.testing.assert_array_equal(output, image)
    assert (
        _measurement_value(
            rows,
            slice_index=0,
            feature_name="TrackObjects_NewObjectCount_Cells_50",
        )
        == 1
    )
    assert (
        _measurement_value(
            rows,
            slice_index=1,
            feature_name="TrackObjects_NewObjectCount_Cells_50",
        )
        == 0
    )
    assert (
        _measurement_value(
            rows,
            slice_index=1,
            object_label=1,
            feature_name="TrackObjects_Label_50",
        )
        == 1
    )
    assert (
        _measurement_value(
            rows,
            slice_index=1,
            object_label=1,
            feature_name="TrackObjects_DistanceTraveled_50",
        )
        == 1.0
    )
    assert (
        _measurement_value(
            rows,
            slice_index=1,
            feature_name="Mean_Cells_TrackObjects_DistanceTraveled_50",
        )
        == 1.0
    )


def test_track_objects_parent_image_number_is_axis_local():
    labels = np.zeros((2, 5, 5), dtype=np.int32)
    labels[0, 1:3, 1:3] = 1
    labels[1, 1:3, 2:4] = 1
    image = np.zeros(labels.shape, dtype=np.float32)

    result = unwrap(track_objects)(
        image,
        labels=_timepoint_labels(labels, axis_size=2),
        tracking_method="overlap",
        pixel_radius=50,
    )
    rows = _projected_measurement_rows(result)

    assert (
        _measurement_value(
            rows,
            slice_index=0,
            feature_name="TrackObjects_NewObjectCount_Cells_50",
        )
        == 1
    )
    parent_image_number = _measurement_value(
        rows,
        slice_index=1,
        object_label=1,
        feature_name="TrackObjects_ParentImageNumber_50",
    )
    trajectory_x = _measurement_value(
        rows,
        slice_index=1,
        object_label=1,
        feature_name="TrackObjects_TrajectoryX_50",
    )
    trajectory_y = _measurement_value(
        rows,
        slice_index=1,
        object_label=1,
        feature_name="TrackObjects_TrajectoryY_50",
    )

    assert parent_image_number == 1
    assert type(parent_image_number) is int
    assert type(trajectory_x) is float
    assert type(trajectory_y) is float


def test_track_objects_preserves_fractional_trajectory_measurements():
    labels = np.zeros((2, 5, 5), dtype=np.int32)
    labels[0, 1:3, 1:3] = 1
    labels[1, 1, 1:4] = 1
    labels[1, 2, 1] = 1
    image = np.zeros(labels.shape, dtype=np.float32)

    result = unwrap(track_objects)(
        image,
        labels=_timepoint_labels(labels, axis_size=2),
        tracking_method="overlap",
        pixel_radius=50,
    )
    rows = _projected_measurement_rows(result)

    trajectory_x = _measurement_value(
        rows,
        slice_index=1,
        object_label=1,
        feature_name="TrackObjects_TrajectoryX_50",
    )
    trajectory_y = _measurement_value(
        rows,
        slice_index=1,
        object_label=1,
        feature_name="TrackObjects_TrajectoryY_50",
    )
    mean_trajectory_x = _measurement_value(
        rows,
        slice_index=1,
        feature_name="Mean_Cells_TrackObjects_TrajectoryX_50",
    )
    mean_trajectory_y = _measurement_value(
        rows,
        slice_index=1,
        feature_name="Mean_Cells_TrackObjects_TrajectoryY_50",
    )

    assert trajectory_x == 0.25
    assert trajectory_y == -0.25
    assert mean_trajectory_x == 0.25
    assert mean_trajectory_y == -0.25


def test_track_objects_overlap_allows_split_children_to_inherit_parent_label():
    labels = np.zeros((3, 7, 8), dtype=np.int32)
    labels[0, 1:5, 1:5] = 1
    labels[1, 1:5, 1:3] = 1
    labels[1, 1:5, 3:5] = 2
    labels[2, 2:6, 1:3] = 1
    image = np.zeros(labels.shape, dtype=np.float32)

    result = unwrap(track_objects)(
        image,
        labels=_timepoint_labels(labels, axis_size=3),
        tracking_method="overlap",
        pixel_radius=50,
    )
    rows = _projected_measurement_rows(result)

    assert (
        _measurement_value(
            rows,
            slice_index=1,
            object_label=1,
            feature_name="TrackObjects_Label_50",
        )
        == 1
    )
    assert (
        _measurement_value(
            rows,
            slice_index=1,
            object_label=2,
            feature_name="TrackObjects_Label_50",
        )
        == 1
    )
    assert (
        _measurement_value(
            rows,
            slice_index=1,
            feature_name="TrackObjects_NewObjectCount_Cells_50",
        )
        == 0
    )
    assert (
        _measurement_value(
            rows,
            slice_index=1,
            feature_name="TrackObjects_SplitObjectCount_Cells_50",
        )
        == 1
    )
    assert (
        _measurement_value(
            rows,
            slice_index=2,
            feature_name="TrackObjects_LostObjectCount_Cells_50",
        )
        == 0
    )
    assert (
        _measurement_value(
            rows,
            slice_index=2,
            feature_name="TrackObjects_MergedObjectCount_Cells_50",
        )
        == 1
    )
    assert (
        _measurement_value(
            rows,
            slice_index=2,
            object_label=1,
            feature_name="TrackObjects_DistanceTraveled_50",
        )
        == 1.0
    )


def test_track_objects_overlap_counts_distinct_parent_merge_not_loss():
    labels = np.zeros((2, 6, 8), dtype=np.int32)
    labels[0, 1:4, 1:3] = 1
    labels[0, 1:4, 4:6] = 2
    labels[1, 1:4, 1:6] = 1
    image = np.zeros(labels.shape, dtype=np.float32)

    result = unwrap(track_objects)(
        image,
        labels=_timepoint_labels(labels, axis_size=2),
        tracking_method="overlap",
        pixel_radius=50,
    )
    rows = _projected_measurement_rows(result)

    assert (
        _measurement_value(
            rows,
            slice_index=1,
            feature_name="TrackObjects_LostObjectCount_Cells_50",
        )
        == 0
    )
    assert (
        _measurement_value(
            rows,
            slice_index=1,
            feature_name="TrackObjects_MergedObjectCount_Cells_50",
        )
        == 1
    )


def test_track_objects_motion_state_follows_split_parent_object():
    labels = np.zeros((3, 7, 8), dtype=np.int32)
    labels[0, 1:5, 1:5] = 1
    labels[1, 1:5, 1:3] = 1
    labels[1, 1:5, 3:5] = 2
    labels[2, 2:6, 3:5] = 1
    image = np.zeros(labels.shape, dtype=np.float32)

    result = unwrap(track_objects)(
        image,
        labels=_timepoint_labels(labels, axis_size=3),
        tracking_method="overlap",
        pixel_radius=50,
    )
    rows = _projected_measurement_rows(result)

    assert (
        _measurement_value(
            rows,
            slice_index=2,
            object_label=1,
            feature_name="TrackObjects_Label_50",
        )
        == 1
    )
    assert (
        _measurement_value(
            rows,
            slice_index=2,
            object_label=1,
            feature_name="TrackObjects_ParentObjectNumber_50",
        )
        == 2
    )
    assert (
        _measurement_value(
            rows,
            slice_index=2,
            object_label=1,
            feature_name="TrackObjects_DistanceTraveled_50",
        )
        == 1.0
    )


def test_track_objects_final_age_marks_terminal_track_labels():
    labels = np.zeros((3, 7, 8), dtype=np.int32)
    labels[0, 1:5, 1:5] = 1
    labels[1, 1:5, 1:3] = 1
    labels[1, 1:5, 3:5] = 2
    labels[2, 2:6, 1:3] = 1
    image = np.zeros(labels.shape, dtype=np.float32)

    result = unwrap(track_objects)(
        image,
        labels=_timepoint_labels(labels, axis_size=3),
        tracking_method="overlap",
        pixel_radius=50,
    )
    rows = _projected_measurement_rows(result)

    assert math.isnan(
        _measurement_value(
            rows,
            slice_index=1,
            object_label=2,
            feature_name="TrackObjects_FinalAge_50",
        )
    )
    final_age = _measurement_value(
        rows,
        slice_index=2,
        object_label=1,
        feature_name="TrackObjects_FinalAge_50",
    )
    assert final_age == 3
    assert type(final_age) is int
    assert (
        _measurement_value(
            rows,
            slice_index=2,
            feature_name="Mean_Cells_TrackObjects_FinalAge_50",
        )
        == 3.0
    )
