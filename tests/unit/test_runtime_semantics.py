from __future__ import annotations

import numpy as np
import pytest

from openhcs.core.runtime_semantics import (
    aligned_dense_object_label_arrays,
    dense_object_label_id_domain,
    object_label_parent_child_payload,
)
from openhcs.core.runtime_values import ObjectLabelPayload


def test_aligned_dense_object_label_arrays_projects_unambiguous_stack() -> None:
    first_plane = np.array(
        [
            [1, 1, 0],
            [0, 0, 2],
        ],
        dtype=np.int32,
    )
    stack = np.stack((first_plane, np.zeros_like(first_plane)))
    reference = np.array(
        [
            [10, 10, 0],
            [0, 0, 20],
        ],
        dtype=np.int32,
    )

    aligned_stack, aligned_reference = aligned_dense_object_label_arrays(
        stack,
        reference,
    )

    np.testing.assert_array_equal(aligned_stack, first_plane)
    np.testing.assert_array_equal(aligned_reference, reference)


def test_aligned_dense_object_label_arrays_rejects_conflicting_stack_projection() -> None:
    first_plane = np.array([[1, 0]], dtype=np.int32)
    second_plane = np.array([[2, 0]], dtype=np.int32)

    with pytest.raises(ValueError, match="conflicting positive labels"):
        aligned_dense_object_label_arrays(
            np.stack((first_plane, second_plane)),
            np.array([[1, 0]], dtype=np.int32),
        )


def test_object_label_parent_child_payload_aligns_parent_stack_to_child_plane() -> None:
    parent_plane = np.array(
        [
            [1, 1, 0],
            [0, 2, 2],
        ],
        dtype=np.int32,
    )
    child_plane = np.array(
        [
            [1, 1, 0],
            [0, 2, 2],
        ],
        dtype=np.int32,
    )
    parent_stack = np.stack((parent_plane, parent_plane))

    payload = object_label_parent_child_payload(parent_stack, child_plane)

    assert payload.parent_ids == (1, 2)
    assert payload.child_ids == (1, 2)


def test_dense_object_label_id_domain_uses_declared_count_for_empty_labels() -> None:
    payload = ObjectLabelPayload(
        labels=np.zeros((3, 3), dtype=np.int32),
        declared_object_count=4,
    )

    assert dense_object_label_id_domain(payload) == (1, 2, 3, 4)


def test_dense_object_label_id_domain_preserves_missing_dense_ids() -> None:
    labels = np.array([[1, 0, 3]], dtype=np.int32)

    assert dense_object_label_id_domain(labels) == (1, 2, 3)
