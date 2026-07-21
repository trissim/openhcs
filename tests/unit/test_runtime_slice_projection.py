from dataclasses import dataclass

import numpy as np
import pytest
from arraybridge.decorators import PRESERVE_INPUT_DTYPE_CONFIG
from typing import Any, get_args, get_origin

from openhcs.constants.constants import VariableComponents
from openhcs.core.aligned_image_payload import (
    ImagePayloadComposition,
    ImagePayloadExecutionMode,
    AlignedImageSliceContext,
    AlignedImageStack,
    ImageOutputBundle,
)
from openhcs.core.callable_contract import KeywordRuntimeParameter
from openhcs.core.runtime_measurements import MeasurementScope, MeasurementSubject
from openhcs.core.runtime_object_label_domains import (
    ObjectLabelDomain,
    ObjectLabelDomainScope,
)
from openhcs.core.runtime_plane_projection import (
    RuntimePlaneAxis,
    RuntimePlaneAxisValueProjection,
)
from openhcs.core.runtime_slice_projection import (
    ObjectLabelValueRuntimeSliceProjectionStrategy,
    RuntimeProjectionSourceIdentityRequest,
    RuntimeProjectionSourceIdentityRequirement,
    RuntimeSliceProjection,
    RuntimeSliceProjectionDeclarationError,
    RuntimeSliceProjectionStrategy,
)
from openhcs.core.runtime_slice_alignment import RuntimeSliceAlignedValues
from openhcs.core.runtime_image_values import (
    ImagePayloadMetadata,
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
)
from openhcs.core.runtime_measurements import (
    MeasurementTable,
)
from openhcs.core.measurement_row_materialization import (
    DataclassMeasurementColumnarRows,
    MeasurementSparseColumnarRows,
)
from openhcs.core.runtime_tabular_values import ColumnarRows
from openhcs.core.runtime_tabular_values import FieldSpec
from openhcs.core.runtime_object_labels import (
    ObjectLabelPayload,
    ObjectLabelVariantData,
    ObjectLabelSet,
    object_label_dense_array,
)
from openhcs.core.runtime_spatial_grid import (
    SpatialGrid,
)
from openhcs.core.source_image_provenance import SourceImageProvenancePlanes
import openhcs.processing.backends.cellprofiler  # noqa: F401


def _subclasses(root: type) -> tuple[type, ...]:
    children = root.__subclasses__()
    return tuple(children) + tuple(
        descendant for child in children for descendant in _subclasses(child)
    )


def _annotation_leaf_types(annotation: object) -> tuple[type, ...]:
    if annotation is Any:
        return ()
    origin = get_origin(annotation)
    if origin is None:
        return (annotation,) if isinstance(annotation, type) else ()
    return tuple(
        leaf
        for argument in get_args(annotation)
        if argument is not Ellipsis
        for leaf in _annotation_leaf_types(argument)
    )


def test_cellprofiler_runtime_parameter_types_declare_slice_projection() -> None:
    missing: list[str] = []
    for parameter_type in _subclasses(KeywordRuntimeParameter):
        if not parameter_type.__module__.startswith(
            ("openhcs.processing.backends.cellprofiler", "openhcs.interop.cellprofiler")
        ):
            continue
        for value_type in _annotation_leaf_types(parameter_type.annotation_type):
            if not RuntimeSliceProjectionStrategy.strategy_types_for_nominal_type(
                value_type
            ):
                missing.append(f"{parameter_type.__name__}: {value_type.__name__}")

    assert not missing, (
        "CellProfiler runtime parameter types lack projection: "
        + ", ".join(sorted(missing))
    )


def _declared_plane_metadata(count: int) -> ImagePayloadMetadata:
    return ImagePayloadMetadata(
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=tuple(f"/tmp/source_{index}.tif" for index in range(count)),
            component_metadata=tuple(
                {"site": str(index + 1)} for index in range(count)
            ),
        ),
    )


def test_runtime_slice_projection_rejects_undeclared_value_type() -> None:
    class UndeclaredValue:
        pass

    with pytest.raises(RuntimeSliceProjectionDeclarationError, match="no nominal"):
        RuntimeSliceProjectionStrategy.strategy_for_value(UndeclaredValue())


def test_object_label_payload_selects_nominal_object_label_projection() -> None:
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.zeros((2, 3), dtype=np.int32),
        ),
    )

    assert type(RuntimeSliceProjectionStrategy.strategy_for_value(payload)) is (
        ObjectLabelValueRuntimeSliceProjectionStrategy
    )


def test_runtime_slice_projection_preserves_nominal_dtype_config() -> None:
    projected = RuntimeSliceProjection.kwargs_for_slice(
        {"dtype_config": PRESERVE_INPUT_DTYPE_CONFIG},
        RuntimePlaneAxisValueProjection.from_selected_plane(
            axis=RuntimePlaneAxis.RUNTIME_SLICE, plane_index=1, axis_size=2
        ),
    )

    assert projected["dtype_config"] is PRESERVE_INPUT_DTYPE_CONFIG


def test_runtime_slice_projection_projects_all_columnar_row_carriers() -> None:
    @dataclass(frozen=True, slots=True)
    class ProjectionRow:
        slice_index: int
        value: float

    rows = DataclassMeasurementColumnarRows((ProjectionRow(0, 2.5),))

    projected = RuntimeSliceProjection.value_for_slice(
        rows,
        RuntimePlaneAxisValueProjection.from_selected_plane(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            plane_index=1,
            axis_size=2,
        ),
    )

    assert isinstance(projected, ColumnarRows)
    assert tuple(projected.column_values("slice_index")) == (1,)
    assert tuple(projected.column_values("value")) == (2.5,)


def test_runtime_slice_projection_selects_runtime_plane_projection_parameter() -> None:
    preserved = RuntimePlaneAxisValueProjection.preserve(
        axis=RuntimePlaneAxis.RUNTIME_SLICE,
        axis_size=2,
    )
    selected = RuntimePlaneAxisValueProjection.from_selected_plane(
        axis=RuntimePlaneAxis.RUNTIME_SLICE,
        plane_index=1,
        axis_size=2,
    )

    projected = RuntimeSliceProjection.kwargs_for_slice(
        {"runtime_plane_projection": preserved},
        selected,
    )

    assert projected["runtime_plane_projection"] == selected


def test_runtime_slice_projection_does_not_infer_high_rank_array_axis() -> None:
    value = np.zeros((3, 2, 4, 5), dtype=np.float32)

    assert RuntimeSliceProjection.slice_count_from_values((value,)) is None
    assert (
        RuntimeSliceProjection.value_for_slice(
            value,
            RuntimePlaneAxisValueProjection.from_selected_plane(
                axis=RuntimePlaneAxis.RUNTIME_SLICE, plane_index=2, axis_size=3
            ),
        )
        is value
    )


def test_runtime_slice_projection_projects_declared_color_image_stack() -> None:
    data = np.stack(
        tuple(np.full((4, 5, 3), index, dtype=np.float32) for index in range(2))
    )
    payload = ImagePayloadMetadata(
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        source_channel_axis=-1,
    ).payload_with(data, None)

    projected = RuntimeSliceProjection.value_for_slice(
        payload,
        RuntimePlaneAxisValueProjection.from_selected_plane(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            plane_index=1,
            axis_size=2,
        ),
    )

    np.testing.assert_array_equal(image_payload_data(projected), data[1])
    assert image_payload_metadata(projected).plane_axis is None
    assert RuntimeSliceProjection.preserved_context_for_value(payload) == (
        RuntimePlaneAxisValueProjection.preserve(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            axis_size=2,
        )
    )


def test_aligned_image_stack_projects_inner_source_axis_across_runtime_slices() -> None:
    runtime_slices = tuple(
        ImagePayloadMetadata(
            plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
            source_image_provenance_planes=(
                SourceImageProvenancePlanes.from_components(
                    component_metadata=(
                        {"channel": "1", "timepoint": str(runtime_index)},
                        {"channel": "2", "timepoint": str(runtime_index)},
                    )
                )
            ),
        ).payload_with(
            np.stack(
                (
                    np.full((2, 3), runtime_index, dtype=np.float32),
                    np.full((2, 3), runtime_index + 10, dtype=np.float32),
                )
            ),
            None,
        )
        for runtime_index in range(2)
    )
    aligned = AlignedImageStack(runtime_slices)

    selected_source = RuntimeSliceProjection.value_for_slice(
        aligned,
        RuntimePlaneAxisValueProjection.from_selected_plane(
            axis=RuntimePlaneAxis.SOURCE_BINDING,
            plane_index=0,
            axis_size=2,
        ),
    )

    assert isinstance(selected_source, AlignedImageStack)
    assert RuntimeSliceProjection.slice_count_from_values((selected_source,)) == 2
    for runtime_index, payload in enumerate(selected_source.slices):
        np.testing.assert_array_equal(
            image_payload_data(payload),
            np.full((2, 3), runtime_index, dtype=np.float32),
        )
        assert image_payload_metadata(payload).source_component_metadata == {
            "channel": "1",
            "timepoint": str(runtime_index),
        }


def test_image_output_bundle_projects_shared_runtime_axis_not_output_count() -> None:
    outputs = tuple(
        _declared_plane_metadata(3).payload_with(
            np.stack(
                tuple(
                    np.full((2, 3), output_index * 10 + plane_index, dtype=np.float32)
                    for plane_index in range(3)
                )
            ),
            None,
        )
        for output_index in range(2)
    )
    contexts = tuple(
        AlignedImageSliceContext.main_flow(output_name)
        for output_name in ("First", "Second")
    )
    bundle = ImageOutputBundle(outputs, contexts)

    assert RuntimeSliceProjection.slice_count_from_values((bundle,)) == 3

    projected = RuntimeSliceProjection.value_for_slice(
        bundle,
        RuntimePlaneAxisValueProjection.from_selected_plane(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            plane_index=1,
            axis_size=3,
        ),
    )

    assert isinstance(projected, ImageOutputBundle)
    assert projected.slice_contexts == contexts
    for output_index, output in enumerate(projected.slices):
        np.testing.assert_array_equal(
            image_payload_data(output),
            np.full((2, 3), output_index * 10 + 1, dtype=np.float32),
        )
        assert image_payload_metadata(output).plane_axis is None


def test_image_output_bundle_uses_outer_axis_without_inner_declaration() -> None:
    outputs = tuple(
        ImagePayloadMetadata().payload_with(
            np.full((2, 3), output_index, dtype=np.float32),
            None,
        )
        for output_index in range(2)
    )
    bundle = ImageOutputBundle(
        outputs,
        tuple(
            AlignedImageSliceContext.main_flow(output_name)
            for output_name in ("First", "Second")
        ),
    )
    composition = ImagePayloadComposition(
        payload=bundle,
        execution_mode=ImagePayloadExecutionMode.ALIGNED_MULTI_IMAGE_STACK,
    )

    assert composition.plane_axis is RuntimePlaneAxis.RUNTIME_SLICE
    assert RuntimeSliceProjection.slice_count_from_values((bundle,)) == 2
    assert RuntimeSliceProjection.preserved_context_for_value(bundle) == (
        RuntimePlaneAxisValueProjection.preserve(
            axis=composition.plane_axis,
            axis_size=2,
        )
    )
    assert (
        RuntimeSliceProjection.value_for_slice(
            bundle,
            RuntimePlaneAxisValueProjection.from_selected_plane(
                axis=RuntimePlaneAxis.RUNTIME_SLICE,
                plane_index=1,
                axis_size=2,
            ),
        )
        is outputs[1]
    )


def test_selected_image_projection_stays_consumed_on_derived_output() -> None:
    data = np.stack(
        tuple(np.full((4, 5), index, dtype=np.float32) for index in range(2))
    )
    payload = _declared_plane_metadata(2).payload_with(data, None)
    projection = RuntimePlaneAxisValueProjection.from_selected_plane(
        axis=RuntimePlaneAxis.RUNTIME_SLICE,
        plane_index=1,
        axis_size=2,
    )
    projected = RuntimeSliceProjection.value_for_slice(payload, projection)

    derived = image_payload_metadata(projected).derive_payload(
        projected,
        np.ones((4, 5), dtype=np.float32),
        plane_projection=projection,
    )

    assert image_payload_metadata(projected).plane_axis is None
    assert image_payload_metadata(derived).plane_axis is None
    assert image_payload_metadata(derived).source_path == "/tmp/source_1.tif"


def test_runtime_slice_projection_validates_declared_image_cardinality() -> None:
    payload = ImagePayloadMetadata(
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    ).payload_with(np.zeros((3, 4, 5), dtype=np.float32), None)

    with pytest.raises(ValueError, match="axis of size 2"):
        RuntimeSliceProjection.value_for_slice(
            payload,
            RuntimePlaneAxisValueProjection.from_selected_plane(
                axis=RuntimePlaneAxis.RUNTIME_SLICE,
                plane_index=1,
                axis_size=2,
            ),
        )


@pytest.mark.parametrize(
    ("axis_size", "shape", "expected"),
    (
        (1, (5, 7), False),
        (1, (1, 5, 7), True),
        (2, (2, 5, 7), True),
        (2, (1, 5, 7), False),
    ),
)
def test_runtime_plane_projection_owns_dense_axis_shape_validation(
    axis_size: int,
    shape: tuple[int, ...],
    expected: bool,
) -> None:
    projection = RuntimePlaneAxisValueProjection.preserve(
        axis=RuntimePlaneAxis.RUNTIME_SLICE,
        axis_size=axis_size,
    )

    assert projection.dense_shape_carries_axis(shape) is expected
    if expected:
        projection.validate_shape(shape, value_name="Dense image")
    else:
        with pytest.raises(ValueError, match=f"axis of size {axis_size}"):
            projection.validate_shape(shape, value_name="Dense image")


def test_runtime_slice_projection_projects_aligned_kwargs_from_one_coordinate() -> None:
    grids = RuntimeSliceAlignedValues(
        (
            SpatialGrid(
                name="Grid",
                rows=1,
                columns=1,
                x_spacing=1.0,
                y_spacing=1.0,
                x_origin=1.0,
                y_origin=1.0,
            ),
            SpatialGrid(
                name="Grid",
                rows=1,
                columns=1,
                x_spacing=1.0,
                y_spacing=1.0,
                x_origin=2.0,
                y_origin=2.0,
            ),
        )
    )
    projected = RuntimeSliceProjection.kwargs_for_slice(
        {
            "grid": grids,
            "values": RuntimeSliceAlignedValues((np.asarray([1.0]), np.asarray([2.0]))),
            "shape_choice": "natural_shape_and_location",
        },
        RuntimePlaneAxisValueProjection.from_selected_plane(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            plane_index=1,
            axis_size=2,
        ),
    )

    assert projected["grid"].x_origin == 2.0
    np.testing.assert_array_equal(projected["values"], np.asarray([2.0]))
    assert projected["shape_choice"] == "natural_shape_and_location"


def test_runtime_slice_projection_rejects_aligned_kwarg_cardinality_mismatch() -> None:
    with pytest.raises(ValueError, match="must exactly match"):
        RuntimeSliceProjection.kwargs_for_slice(
            {"values": RuntimeSliceAlignedValues((np.asarray([1.0]),))},
            RuntimePlaneAxisValueProjection.from_selected_plane(
                axis=RuntimePlaneAxis.RUNTIME_SLICE,
                plane_index=1,
                axis_size=2,
            ),
        )


def test_runtime_slice_projection_projects_declared_object_label_plane() -> None:
    label_planes = np.stack(
        (
            np.full((5, 5), 3, dtype=np.int32),
            np.full((5, 5), 7, dtype=np.int32),
        )
    )
    labels = ObjectLabelSet(
        name="Cells",
        variant_data=ObjectLabelVariantData(labels=label_planes),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
            declared_object_id_domains=((3,), (7,)),
        ),
    )

    projected = RuntimeSliceProjection.value_for_slice(
        labels,
        RuntimePlaneAxisValueProjection.from_selected_plane(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            plane_index=1,
            axis_size=2,
        ),
    )

    assert isinstance(projected, ObjectLabelSet)
    assert projected.domain.scope is ObjectLabelDomainScope.PAYLOAD
    assert projected.plane_axis is None
    np.testing.assert_array_equal(projected.labels, label_planes[1])


def test_runtime_slice_projection_preserves_image_mask_and_plane_metadata() -> None:
    data = np.stack(
        tuple(np.full((4, 5), index, dtype=np.float32) for index in range(2))
    )
    mask = np.stack(
        tuple(np.full((4, 5), index == 1, dtype=bool) for index in range(2))
    )
    payload = _declared_plane_metadata(2).payload_with(data, mask)

    projected = RuntimeSliceProjection.value_for_slice(
        payload,
        RuntimePlaneAxisValueProjection.from_selected_plane(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            plane_index=1,
            axis_size=2,
        ),
    )

    np.testing.assert_array_equal(image_payload_data(projected), data[1])
    np.testing.assert_array_equal(image_payload_mask(projected), mask[1])
    assert image_payload_metadata(projected).source_path == "/tmp/source_1.tif"
    assert image_payload_metadata(projected).plane_axis is None


def test_variable_component_projection_requires_declared_plane_provenance() -> None:
    request = RuntimeProjectionSourceIdentityRequest(
        value=np.zeros((2, 4, 5), dtype=np.float32),
        source_description="undeclared image stack",
        variable_components=(VariableComponents.SITE,),
    )

    with pytest.raises(
        RuntimeSliceProjectionDeclarationError,
        match="nominal payload.*RUNTIME_SLICE",
    ):
        RuntimeProjectionSourceIdentityRequirement.OPTIONAL.project_payload_items(
            request
        )


def test_variable_component_projection_validates_declared_cardinality() -> None:
    payload = _declared_plane_metadata(2).payload_with(
        np.zeros((3, 4, 5), dtype=np.float32), None
    )
    request = RuntimeProjectionSourceIdentityRequest(
        value=payload,
        source_description="mismatched image stack",
        variable_components=(VariableComponents.SITE,),
        plane_projection=RuntimePlaneAxisValueProjection.preserve(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            axis_size=2,
        ),
    )

    with pytest.raises(ValueError, match="declared 'runtime_slice' axis of size 2"):
        RuntimeProjectionSourceIdentityRequirement.OPTIONAL.project_payload_items(
            request
        )


def test_variable_component_projection_rejects_expanded_source_provenance() -> None:
    payload = _declared_plane_metadata(9).payload_with(
        np.zeros((3, 4, 5), dtype=np.float32), None
    )
    request = RuntimeProjectionSourceIdentityRequest(
        value=payload,
        source_description="expanded-provenance image stack",
        variable_components=(VariableComponents.SITE,),
        plane_projection=RuntimePlaneAxisValueProjection.preserve(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            axis_size=3,
        ),
    )

    with pytest.raises(
        ValueError,
        match="source provenance must exactly match.*9 != 3",
    ):
        RuntimeProjectionSourceIdentityRequirement.OPTIONAL.project_payload_items(
            request
        )


def test_variable_component_projection_uses_declared_plane_axis() -> None:
    stack = np.stack(
        tuple(np.full((4, 5), index, dtype=np.float32) for index in range(2))
    )
    payload = _declared_plane_metadata(2).payload_with(stack, None)

    projected = (
        RuntimeProjectionSourceIdentityRequirement.OPTIONAL.project_payload_items(
            RuntimeProjectionSourceIdentityRequest(
                value=payload,
                source_description="declared image stack",
                variable_components=(VariableComponents.SITE,),
                plane_projection=RuntimePlaneAxisValueProjection.preserve(
                    axis=RuntimePlaneAxis.RUNTIME_SLICE,
                    axis_size=2,
                ),
            )
        )
    )

    assert len(projected) == 2
    np.testing.assert_array_equal(projected[0].data, stack[0])
    np.testing.assert_array_equal(projected[1].data, stack[1])
    assert projected[0].runtime_plane_metadata is not None
    assert projected[0].runtime_plane_metadata.plane_shape == (2,)
    assert projected[1].runtime_plane_metadata is not None
    assert projected[1].runtime_plane_metadata.source_plane_indices == (1,)


def test_source_binding_projection_projects_object_labels_through_same_axis() -> None:
    planes = np.array(
        (
            ((1, 0), (0, 0)),
            ((0, 0), (0, 2)),
        ),
        dtype=np.int32,
    )
    labels = ObjectLabelSet(
        name="Objects",
        variant_data=ObjectLabelVariantData(labels=planes),
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
        source_image_names=("First", "Second"),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )
    projection = RuntimePlaneAxisValueProjection.preserve(
        axis=RuntimePlaneAxis.SOURCE_BINDING,
        axis_size=2,
        source_aliases=("First", "Second"),
    ).selected_plane(1)

    projected = RuntimeSliceProjection.value_for_slice(labels, projection)

    assert isinstance(projected, ObjectLabelSet)
    np.testing.assert_array_equal(projected.labels, planes[1])
    assert projected.plane_axis is None


def test_source_identity_request_projects_explicit_source_binding_label_planes() -> (
    None
):
    planes = np.array(
        (
            ((1, 0), (0, 0)),
            ((0, 0), (0, 2)),
        ),
        dtype=np.int32,
    )
    labels = ObjectLabelSet(
        name="Objects",
        variant_data=ObjectLabelVariantData(labels=planes),
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
        source_image_names=("First", "Second"),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )

    projected = (
        RuntimeProjectionSourceIdentityRequirement.OPTIONAL.project_payload_items(
            RuntimeProjectionSourceIdentityRequest(
                value=labels,
                source_description="source-bound labels",
                plane_projection=labels.declared_plane_projection(),
            )
        )
    )

    assert len(projected) == 2
    np.testing.assert_array_equal(
        object_label_dense_array(projected[0].value),
        planes[0],
    )
    np.testing.assert_array_equal(
        object_label_dense_array(projected[1].value),
        planes[1],
    )
    assert projected[0].runtime_plane_metadata is not None
    assert projected[0].runtime_plane_metadata.plane_indices == (0,)
    assert projected[1].runtime_plane_metadata is not None
    assert projected[1].runtime_plane_metadata.plane_indices == (1,)


def test_measurement_row_names_and_order_do_not_declare_slice_count() -> None:
    table = MeasurementTable(
        name="Measurements",
        rows=MeasurementSparseColumnarRows.from_rows(
            (
                {"slice_index": 0, "value": 1.0},
                {"slice_index": 1, "value": 2.0},
            ),
            fields=(FieldSpec("slice_index", int), FieldSpec("value", float)),
        ),
        subject=MeasurementSubject(MeasurementScope.OBJECT, name="Objects"),
    )

    with pytest.raises(
        RuntimeSliceProjectionDeclarationError,
        match="source-plane provenance",
    ):
        RuntimeSliceProjection.slice_count_from_values((table,))


def test_axisless_object_measurement_is_not_runtime_slice_projected() -> None:
    table = MeasurementTable(
        name="Measurements",
        rows=MeasurementSparseColumnarRows.from_rows(
            ({"object_label": 1, "value": 1.0},),
            fields=(FieldSpec("object_label", int), FieldSpec("value", float)),
        ),
        subject=MeasurementSubject(MeasurementScope.OBJECT, name="Objects"),
    )
    projection = RuntimePlaneAxisValueProjection.from_selected_plane(
        axis=RuntimePlaneAxis.RUNTIME_SLICE,
        plane_index=0,
        axis_size=1,
    )

    assert RuntimeSliceProjection.slice_count_from_values((table,)) is None
    assert RuntimeSliceProjection.value_for_slice(table, projection) is table


def test_measurement_table_source_provenance_declares_slice_count() -> None:
    table = MeasurementTable(
        name="Measurements",
        rows=MeasurementSparseColumnarRows.from_rows(
            (
                {"slice_index": 0, "value": 1.0},
                {"slice_index": 1, "value": 2.0},
            ),
            fields=(FieldSpec("slice_index", int), FieldSpec("value", float)),
        ),
        subject=MeasurementSubject(MeasurementScope.OBJECT, name="Objects"),
        source_image_provenance_planes=_declared_plane_metadata(
            2
        ).source_image_provenance_planes,
    )

    assert RuntimeSliceProjection.slice_count_from_values((table,)) == 2


def test_measurement_table_scalar_source_provenance_is_already_projected() -> None:
    table = MeasurementTable(
        name="Measurements",
        rows=MeasurementSparseColumnarRows.from_rows(
            ({"slice_index": 1, "value": 2.0},),
            fields=(FieldSpec("slice_index", int), FieldSpec("value", float)),
        ),
        subject=MeasurementSubject(MeasurementScope.OBJECT, name="Objects"),
        source_path="/tmp/source_1.tif",
        source_component_metadata={"site": "2"},
    )

    assert RuntimeSliceProjection.slice_count_from_values((table,)) is None
