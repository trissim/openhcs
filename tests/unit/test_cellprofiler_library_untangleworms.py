from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import numpy as np
import centrosome.cpmorphology
from skimage.segmentation import find_boundaries

from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactOutputPlan,
    ArtifactSpec,
    ObjectLabelsArtifactType,
)
from openhcs.core.callable_contract import CallableContract
from openhcs.core.component_group_scope import (
    ComponentGroupScope,
    RuntimeExecutionAxisScope,
)
from openhcs.core.pipeline.function_contracts import special_input_names_from_callable
from openhcs.processing.backends.cellprofiler.worms import (
    OverlapStyle,
    StraightenWormsModule,
    WormControlPointGeometry,
    _overlapping_worm_outline,
    _reconstructed_worm_pixels,
    _worm_descriptor_row,
    straighten_worms,
    untangle_worms,
)
import openhcs.processing.backends.cellprofiler.worms as worm_backend
from openhcs.core.runtime_image_values import (
    ImagePayloadMetadata,
    ImageMetadataPayload,
)
from openhcs.core.runtime_object_labels import (
    ObjectLabelRepresentation,
    ObjectLabelVariantData,
    ObjectLabelPayload,
    object_label_dense_array,
)
from openhcs.core.runtime_object_label_building import (
    SourceImageObjectLabelBuildRequest,
)
from openhcs.core.runtime_object_label_domains import (
    ObjectLabelDomain,
    ObjectLabelDomainScope,
)
from openhcs.core.runtime_plane_projection import (
    RuntimePlaneAxis,
    RuntimePlaneProjection,
)
from openhcs.core.runtime_artifact_values import RuntimeValue
from openhcs.core.runtime_stores import RuntimeValueStore
from openhcs.core.source_spatial_domain import SourceSpatialDomain
from openhcs.core.runtime_sparse_labels import SparseIJVLabelRows
from openhcs.processing.backends.cellprofiler.object_images import (
    object_label_colormap,
)
from openhcs.processing.backends.cellprofiler.worm_geometry import (
    branchpoints,
    calculate_cumulative_lengths,
    endpoints,
    rebuild_worm_from_control_points_approx,
    sample_control_points,
    skeletonize_worm_mask,
)
from openhcs.interop.cellprofiler.runtime.artifact_binding import (
    RuntimeInputBindingRequest,
)
from tests.unit.cellprofiler_runtime_test_support import (
    cellprofiler_runtime_adapter_for_test,
    cellprofiler_runtime_input_edge_for_test,
)


def _straighten_worms_binding_request(
    labels: ObjectLabelPayload,
    *,
    plane_index: int | None,
) -> RuntimeInputBindingRequest:
    object_spec = ArtifactSpec.input(
        "Worms",
        ObjectLabelsArtifactType,
        parameter_name=(
            StraightenWormsModule.input_objects_binding.require_runtime_parameter_name()
        ),
    )
    callable_contract = CallableContract.from_callable(straighten_worms)
    path = "/memory/Worms.pkl"
    output_plan = ArtifactOutputPlan(
        name=object_spec.name,
        path=path,
        artifact_type=object_spec.artifact_type,
    )
    store = RuntimeValueStore()
    store.record(
        RuntimeValue.normalize(output_plan, labels, axis_id="A01"),
        path=path,
        backend="memory",
    )
    plane_count = labels.declared_plane_count()
    edge = cellprofiler_runtime_input_edge_for_test(
        ArtifactInputPlan(
            name=object_spec.name,
            path=path,
            artifact_type=object_spec.artifact_type,
        ),
        spec=object_spec,
        invocation_scope=ComponentGroupScope.ungrouped(),
        producer_selection_scope=ComponentGroupScope.ungrouped(),
        component_scopes=(),
        consumer_variable_components=(),
    )
    adapter = cellprofiler_runtime_adapter_for_test(
        runtime_value_store=store,
        callable_contract=replace(
            callable_contract,
            metadata=replace(
                callable_contract.metadata,
                artifact_inputs=(object_spec,),
            ),
        ),
        axis_scope=RuntimeExecutionAxisScope.from_raw(
            "A01",
            component=None,
            value=None,
        ),
        artifact_inputs={edge.key: edge},
        artifact_outputs={},
        plane_projection=(
            RuntimePlaneProjection.stack(plane_count)
            if plane_index is None
            else RuntimePlaneProjection.selected(plane_index, plane_count)
        ),
    )
    return RuntimeInputBindingRequest(
        adapter=adapter,
        kwargs={},
        current_image=np.zeros((2, 2), dtype=np.float32),
    )


def _straighten_worm_labels(bound: dict[str, object]) -> object:
    parameter_names = special_input_names_from_callable(straighten_worms)
    assert len(parameter_names) == 1
    return bound[parameter_names[0]]


def test_untangle_worms_single_component_does_not_apply_cluster_min_path_length() -> (
    None
):
    image = np.zeros((24, 24), dtype=np.uint8)
    image[8:11, 4:20] = 1

    canonical_sample_control_points = worm_backend.sample_control_points
    with patch.object(
        worm_backend,
        "sample_control_points",
        wraps=canonical_sample_control_points,
    ) as sample_control_points:
        _image, measurements, nonoverlap = untangle_worms.__wrapped__(
            image,
            overlap_style=OverlapStyle.WITHOUT_OVERLAP,
            min_worm_area=1.0,
            max_worm_area=1_000.0,
            min_path_length=100.0,
            max_path_length=200.0,
            cost_threshold=1_000.0,
            num_control_points=5,
            mean_angles=(),
        )

    assert sample_control_points.call_count == 2
    assert len(measurements) == 1
    assert measurements[0]["object_number"] == 1
    assert np.max(object_label_dense_array(nonoverlap)) == 1


def test_reused_worm_geometry_matches_fresh_45_degree_orientation_tie() -> None:
    path_coords = np.array(
        [
            (3, 3),
            (4, 4),
            (5, 5),
            (6, 6),
            (7, 7),
        ],
        dtype=np.int32,
    )
    radii = np.ones(5, dtype=float)
    cumulative_lengths = calculate_cumulative_lengths(path_coords)
    shared_geometry = WormControlPointGeometry.from_path_coords(
        path_coords,
        cumulative_lengths,
        num_control_points=5,
    )
    shared_control_coords = shared_geometry.control_coords.copy()

    actual_pixels = _reconstructed_worm_pixels(
        shared_geometry,
        image_shape=(12, 12),
        radii_from_training=radii,
    )
    actual_row = _worm_descriptor_row(shared_geometry, object_number=1)

    canonical_control_coords = sample_control_points(
        path_coords,
        calculate_cumulative_lengths(path_coords),
        5,
    )
    canonical_geometry = WormControlPointGeometry(
        canonical_control_coords,
        float(cumulative_lengths[-1]),
    )
    expected_pixels = rebuild_worm_from_control_points_approx(
        canonical_control_coords,
        radii,
        (12, 12),
    )
    expected_row = _worm_descriptor_row(canonical_geometry, object_number=1)

    np.testing.assert_array_equal(shared_geometry.control_coords, shared_control_coords)
    np.testing.assert_array_equal(shared_control_coords, canonical_control_coords)
    np.testing.assert_array_equal(actual_pixels[0], expected_pixels[0])
    np.testing.assert_array_equal(actual_pixels[1], expected_pixels[1])
    assert actual_row.keys() == expected_row.keys()
    np.testing.assert_array_equal(
        np.asarray(tuple(actual_row.values()), dtype=np.float64).view(np.uint64),
        np.asarray(tuple(expected_row.values()), dtype=np.float64).view(np.uint64),
    )


def test_untangle_worms_labels_preserve_source_image_spatial_metadata() -> None:
    image = ImageMetadataPayload(
        data=np.zeros((24, 24), dtype=np.uint8),
        metadata=ImagePayloadMetadata(
            source_spatial_domain=SourceSpatialDomain(
                origin_yx=(5, 7),
                source_shape_yx=(40, 50),
            ),
        ),
    )
    image.data[8:11, 4:20] = 1

    _image, _measurements, nonoverlap = untangle_worms.__wrapped__(
        image,
        overlap_style=OverlapStyle.WITHOUT_OVERLAP,
        min_worm_area=1.0,
        max_worm_area=1_000.0,
        min_path_length=100.0,
        max_path_length=200.0,
        cost_threshold=1_000.0,
        num_control_points=5,
        mean_angles=(),
    )

    assert isinstance(nonoverlap, ObjectLabelPayload)
    assert nonoverlap.spatial_origin_yx == (5, 7)
    assert nonoverlap.source_spatial_shape_yx == (40, 50)


def test_straighten_worms_rejects_unprojected_object_label_stack() -> None:
    image = np.zeros((8, 8), dtype=np.float32)
    labels = np.zeros((2, 8, 8), dtype=np.int32)

    with np.testing.assert_raises_regex(
        ValueError,
        "one runtime-projected 2-D object-label plane",
    ):
        straighten_worms.__wrapped__(
            image,
            ObjectLabelPayload(variant_data=ObjectLabelVariantData(labels=labels)),
            num_control_points=3,
        )


def test_straighten_worms_policy_consumes_declared_singleton_label_plane() -> None:
    label_stack = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.array([[[0, 1], [0, 0]]], dtype=np.int32)
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,),),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )
    request = _straighten_worms_binding_request(label_stack, plane_index=0)

    bound = StraightenWormsModule.bind_runtime_inputs(request)

    projected_labels = _straighten_worm_labels(bound)
    np.testing.assert_array_equal(
        projected_labels.labels,
        np.array([[0, 1], [0, 0]], dtype=np.int32),
    )
    assert projected_labels.plane_axis is None
    assert projected_labels.domain.scope is ObjectLabelDomainScope.PAYLOAD


def test_straighten_worms_callable_rejects_unselected_label_stack() -> None:
    label_stack = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.array(
                [
                    [[0, 1], [0, 0]],
                    [[0, 0], [2, 0]],
                ],
                dtype=np.int32,
            )
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )
    request = _straighten_worms_binding_request(
        label_stack,
        plane_index=None,
    )
    bound = StraightenWormsModule.bind_runtime_inputs(request)

    with np.testing.assert_raises_regex(
        ValueError,
        "one runtime-projected 2-D object-label plane",
    ):
        straighten_worms.__wrapped__(
            np.zeros((2, 2), dtype=np.float32),
            _straighten_worm_labels(bound),
        )


def test_worm_geometry_matches_centrosome_for_interior_skeletons() -> None:
    rng = np.random.RandomState(11)

    for shape in ((8, 8), (16, 16), (24, 24)):
        for _index in range(8):
            mask = rng.rand(*shape) > 0.72
            mask[[0, -1], :] = False
            mask[:, [0, -1]] = False

            np.testing.assert_array_equal(
                skeletonize_worm_mask(mask),
                centrosome.cpmorphology.skeletonize(mask),
            )


def test_worm_lookup_geometry_matches_centrosome() -> None:
    rng = np.random.RandomState(13)

    for shape in ((8, 8), (16, 16)):
        for _index in range(8):
            skeleton = rng.rand(*shape) > 0.82

            np.testing.assert_array_equal(
                branchpoints(skeleton),
                centrosome.cpmorphology.branchpoints(skeleton),
            )
            np.testing.assert_array_equal(
                endpoints(skeleton),
                centrosome.cpmorphology.endpoints(skeleton),
            )


def test_overlapping_worm_outline_crops_without_changing_pixels() -> None:
    image = np.zeros((12, 14), dtype=np.uint8)
    ijv = np.array(
        [
            (0, 0, 1),
            (0, 1, 1),
            (1, 0, 1),
            (1, 1, 1),
            (1, 1, 3),
            (1, 2, 3),
            (2, 1, 3),
            (2, 2, 3),
            (10, 12, 5),
            (10, 13, 5),
            (11, 12, 5),
            (11, 13, 5),
        ],
        dtype=np.int32,
    )
    labels = SourceImageObjectLabelBuildRequest(
        image=image,
        labels=SparseIJVLabelRows(ijv),
    ).payload(representation=ObjectLabelRepresentation.SPARSE_IJV)

    colors = object_label_colormap("viridis", 5)
    expected = np.zeros((*image.shape, 3), dtype=np.float32)
    for label_id in (1, 3, 5):
        label_rows = ijv[ijv[:, 2] == label_id]
        mask = np.zeros(image.shape, dtype=bool)
        mask[label_rows[:, 0], label_rows[:, 1]] = True
        expected[find_boundaries(mask, mode="inner")] = colors[label_id]

    with patch(
        "skimage.segmentation.find_boundaries",
        wraps=find_boundaries,
    ) as cropped_find_boundaries:
        actual = _overlapping_worm_outline(image, labels, "viridis")

    np.testing.assert_array_equal(np.asarray(actual), expected)
    assert cropped_find_boundaries.call_count == 3
    assert all(
        call.args[0].shape != image.shape
        for call in cropped_find_boundaries.call_args_list
    )
