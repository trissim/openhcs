import numpy as np
import pytest

from openhcs.core.aligned_image_payload import AlignedImageStack
from openhcs.core.runtime_object_label_domains import ObjectLabelDomain, ObjectLabelDomainScope
from openhcs.core.runtime_plane_projection import RuntimePlaneAxis
from openhcs.core.runtime_object_labels import (
    ObjectLabelVariantData,
    ObjectLabelSet,
)
from openhcs.processing.backends.cellprofiler.image_geometry import (
    align_label_plane_to_shape,
    aligned_image_mask_planes,
)


def test_image_mask_planes_require_exact_aligned_owner_cardinality() -> None:
    images = AlignedImageStack(
        tuple(np.zeros((4, 5), dtype=np.float32) for _index in range(2))
    )
    masks = AlignedImageStack(tuple(np.ones((4, 5), dtype=bool) for _index in range(4)))

    with pytest.raises(ValueError, match="cardinalities must exactly match"):
        aligned_image_mask_planes(images, masks)


def test_object_label_runtime_axis_owns_exact_mask_plane_selection() -> None:
    images = AlignedImageStack(
        tuple(np.zeros((4, 5), dtype=np.float32) for _index in range(2))
    )
    labels = np.zeros((2, 4, 5), dtype=np.int32)
    labels[0, 0, 0] = 1
    labels[1, 1, 1] = 2
    objects = ObjectLabelSet(
        name="Nuclei",
        variant_data=ObjectLabelVariantData(labels=labels),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )

    planes = aligned_image_mask_planes(images, objects, labels=True)

    assert len(planes) == 2
    np.testing.assert_array_equal(planes[0].mask, labels[0] > 0)
    np.testing.assert_array_equal(planes[1].mask, labels[1] > 0)


def test_label_shape_mismatch_fails_instead_of_resizing() -> None:
    with pytest.raises(ValueError, match="must exactly match"):
        align_label_plane_to_shape(
            np.zeros((2, 3), dtype=np.int32),
            (4, 5),
        )
