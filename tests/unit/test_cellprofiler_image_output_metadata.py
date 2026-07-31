from __future__ import annotations

import inspect

import numpy as np
import pytest

from openhcs.core.aligned_image_payload import (
    AlignedImageStack,
    ImagePayloadBundleContext,
)
from openhcs.core.runtime_image_values import (
    ImagePayloadMetadata,
    image_payload_data,
    image_payload_metadata,
)
from openhcs.core.runtime_object_label_building import (
    SourceImageObjectLabelBuildRequest,
)
from openhcs.core.runtime_object_labels import (
    ObjectLabelPayload,
    ObjectLabelVariantData,
)
from openhcs.core.runtime_plane_projection import RuntimePlaneAxis
from openhcs.core.source_image_provenance import SourceImageProvenancePlanes
from openhcs.core.source_spatial_domain import SourceSpatialDomain
from openhcs.processing.backends.cellprofiler.colocalization import (
    ColocalizationThresholdMaskGroup,
    ColocalizationThresholdMaskRuntimeOutput,
    measure_colocalization,
)
from openhcs.processing.backends.cellprofiler.illumination import (
    CalculationScope,
    RescaleOption,
    SmoothingMethod,
    correct_illumination_calculate,
)
from openhcs.processing.backends.cellprofiler.image_math import (
    ImageMathOperation,
    image_math,
)
from openhcs.processing.backends.cellprofiler.skeleton import (
    measure_object_skeleton_with_branchpoint_image,
)
from openhcs.processing.backends.cellprofiler.worms import (
    _nonoverlapping_worm_outline,
    _overlapping_worm_outline,
    straighten_worms,
)


def _source_binding_payload(
    data: np.ndarray,
    *,
    scalar_path: str = "/stale/input.tif",
    plane_path: str = "/input/shared.tif",
) -> object:
    plane_count = data.shape[0]
    return ImagePayloadMetadata(
        source_path=scalar_path,
        source_image_names=tuple(f"Source{index}" for index in range(plane_count)),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=(plane_path,) * plane_count,
            component_metadata=tuple(
                {"channel": str(index)} for index in range(plane_count)
            ),
        ),
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
    ).payload_with(data, None)


def test_colocalization_saved_mask_projects_its_declared_source_plane() -> None:
    dna = np.arange(20, dtype=np.float32).reshape(4, 5)
    rna = np.flipud(dna)
    payload = ImagePayloadBundleContext.from_payloads(
        (
            ImagePayloadMetadata(
                source_path="/input/dna.tif",
                source_image_names=("DNA",),
            ).payload_with(dna, None),
            ImagePayloadMetadata(
                source_path="/input/rna.tif",
                source_image_names=("RNA",),
            ).payload_with(rna, None),
        )
    ).compose()
    group = ColocalizationThresholdMaskGroup("RNA", "RNAThreshold", 25.0)

    output, _rows = inspect.unwrap(measure_colocalization)(
        payload,
        do_correlation=False,
        do_manders=False,
        do_rwc=False,
        do_overlap=False,
        do_costes=False,
        threshold_mask_groups=(group,),
        threshold_mask_outputs=(
            ColocalizationThresholdMaskRuntimeOutput(group, 1),
        ),
    )

    metadata = image_payload_metadata(output)
    assert image_payload_data(output).shape == rna.shape
    assert metadata.plane_axis is None
    assert metadata.source_channel_axis is None
    assert metadata.source_image_paths == ("/input/rna.tif",)


def test_image_math_collapses_all_source_binding_contributors() -> None:
    payload = _source_binding_payload(
        np.stack(
            (
                np.full((3, 4), 0.25, dtype=np.float32),
                np.full((3, 4), 0.5, dtype=np.float32),
            )
        )
    )

    output = image_math(
        payload,
        operation=ImageMathOperation.ADD,
        factors=(1.0, 1.0),
        truncate_high=False,
    )

    metadata = image_payload_metadata(output)
    assert metadata.plane_axis is None
    assert metadata.source_path == "/input/shared.tif"
    assert len(metadata.source_provenance.represented_source_identities) == 2


def test_all_image_illumination_collapses_all_source_plane_contributors() -> None:
    payload = _source_binding_payload(
        np.stack(
            (
                np.full((4, 5), 0.25, dtype=np.float32),
                np.full((4, 5), 0.75, dtype=np.float32),
            )
        )
    )

    output = inspect.unwrap(correct_illumination_calculate)(
        payload,
        calculation_scope=CalculationScope.ALL_FIRST_CYCLE,
        smoothing_method=SmoothingMethod.NONE,
        rescale_option=RescaleOption.NO,
    )

    metadata = image_payload_metadata(output)
    assert image_payload_data(output).shape == (4, 5)
    assert metadata.plane_axis is None
    assert metadata.source_path == "/input/shared.tif"
    assert len(metadata.source_provenance.represented_source_identities) == 2


def test_skeleton_branchpoint_image_declares_rgb_channel_axis() -> None:
    skeleton = np.zeros((9, 9), dtype=bool)
    skeleton[4, 2:7] = True
    labels = np.zeros((9, 9), dtype=np.int32)
    labels[3:6, 3:6] = 1
    image = ImagePayloadMetadata(
        source_path="/input/skeleton.tif",
        source_image_names=("Skeleton",),
    ).payload_with(skeleton, None)

    branchpoints, _rows = inspect.unwrap(
        measure_object_skeleton_with_branchpoint_image
    )(
        image,
        ObjectLabelPayload(variant_data=ObjectLabelVariantData(labels=labels)),
        branchpoint_image_name="Branches",
    )

    metadata = image_payload_metadata(branchpoints)
    assert image_payload_data(branchpoints).shape == (9, 9, 3)
    assert metadata.source_channel_axis == -1
    assert metadata.plane_axis is None


def test_worm_outline_images_declare_rgb_and_scalar_channel_semantics() -> None:
    source = ImagePayloadMetadata(
        source_path="/input/worms.tif",
        source_image_names=("Worms",),
    ).payload_with(np.zeros((7, 8), dtype=np.float32), None)
    dense_labels = np.zeros((7, 8), dtype=np.int32)
    dense_labels[2:5, 2:6] = 1
    labels = SourceImageObjectLabelBuildRequest(
        image=source,
        labels=dense_labels,
    ).payload()

    overlapping = _overlapping_worm_outline(source, labels, "Default")
    assert image_payload_data(overlapping).shape == (7, 8, 3)
    assert image_payload_metadata(overlapping).source_channel_axis == -1

    rgb_source = ImagePayloadMetadata(source_channel_axis=-1).payload_with(
        np.zeros((7, 8, 3), dtype=np.float32),
        None,
    )
    nonoverlapping = _nonoverlapping_worm_outline(rgb_source, labels)
    assert image_payload_data(nonoverlapping).shape == (7, 8)
    assert image_payload_metadata(nonoverlapping).source_channel_axis is None


@pytest.mark.parametrize(
    "plane_axis",
    (RuntimePlaneAxis.SOURCE_BINDING, RuntimePlaneAxis.RUNTIME_SLICE),
)
def test_straighten_worms_projects_sources_into_warped_spatial_domain(
    plane_axis: RuntimePlaneAxis,
) -> None:
    source_payloads = tuple(
        ImagePayloadMetadata(
            source_path=f"/input/channel_{index}.tif",
            source_image_names=(f"Channel{index}",),
            source_spatial_domain=SourceSpatialDomain(
                origin_yx=(2, 3),
                source_shape_yx=(20, 20),
            ),
        ).payload_with(np.full((8, 8), index, dtype=np.float32), None)
        for index in (1, 2)
    )
    image = (
        ImagePayloadBundleContext.from_payloads(source_payloads).compose()
        if plane_axis is RuntimePlaneAxis.SOURCE_BINDING
        else ImagePayloadMetadata.compose(source_payloads).replace_fields(
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE
        ).payload_with(
            np.stack(tuple(image_payload_data(payload) for payload in source_payloads)),
            None,
        )
    )
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.zeros((8, 8), dtype=np.int32)
        )
    )

    output, _rows, _labels = inspect.unwrap(straighten_worms)(
        image,
        labels,
        control_points=np.empty((0, 2, 3), dtype=float),
        worm_width=2,
        num_control_points=3,
    )

    assert isinstance(output, AlignedImageStack)
    assert len(output.slices) == 2
    for index, warped in enumerate(output.slices, start=1):
        metadata = image_payload_metadata(warped)
        assert metadata.plane_axis is None
        assert metadata.source_image_paths == (f"/input/channel_{index}.tif",)
        assert metadata.source_spatial_domain.origin_yx == (0, 0)
        assert metadata.source_spatial_domain.source_shape_yx == (
            image_payload_data(warped).shape[-2:]
        )
