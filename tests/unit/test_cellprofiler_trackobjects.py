import math

import numpy as np
from inspect import unwrap

from benchmark.cellprofiler_library.functions.trackobjects import track_objects
from openhcs.constants.constants import MemoryType
from openhcs.core.runtime_artifact_queries import (
    MEASUREMENT_FEATURE_NAME_FIELD,
    MEASUREMENT_MEASUREMENT_VALUE_FIELD,
    MEASUREMENT_OBJECT_LABEL_FIELD,
)
from openhcs.processing.backends.cellprofiler.tracking import (
    NumbaNumpyObjectTrackingBackendStrategy,
    ObjectTrackingBackendStrategy,
)


def _measurement_value(rows, *, image_number, feature_name, object_label=None):
    for row in rows:
        if row.get("image_number") != image_number:
            continue
        if row.get(MEASUREMENT_FEATURE_NAME_FIELD) != feature_name:
            continue
        if (
            object_label is not None
            and row.get(MEASUREMENT_OBJECT_LABEL_FIELD) != object_label
        ):
            continue
        return row[MEASUREMENT_MEASUREMENT_VALUE_FIELD]
    raise AssertionError(f"missing measurement row {image_number=} {feature_name=}")


def test_track_objects_uses_numba_tracking_backend_by_default():
    assert type(ObjectTrackingBackendStrategy.for_memory_type(MemoryType.NUMPY)) is (
        NumbaNumpyObjectTrackingBackendStrategy
    )


def test_track_objects_emits_stack_tracking_measurements():
    labels = np.zeros((2, 5, 5), dtype=np.int32)
    labels[0, 1:3, 1:3] = 1
    labels[1, 1:3, 2:4] = 1
    image = np.zeros(labels.shape, dtype=np.float32)

    output, rows = unwrap(track_objects)(
        image,
        labels=labels,
        object_name="Cells",
        tracking_method="overlap",
        pixel_radius=50,
    )

    np.testing.assert_array_equal(output, image)
    assert _measurement_value(
        rows,
        image_number=1,
        feature_name="TrackObjects_NewObjectCount_Cells_50",
    ) == 1
    assert _measurement_value(
        rows,
        image_number=2,
        feature_name="TrackObjects_NewObjectCount_Cells_50",
    ) == 0
    assert _measurement_value(
        rows,
        image_number=2,
        object_label=1,
        feature_name="TrackObjects_Label_50",
    ) == 1
    assert _measurement_value(
        rows,
        image_number=2,
        object_label=1,
        feature_name="TrackObjects_DistanceTraveled_50",
    ) == 1.0
    assert _measurement_value(
        rows,
        image_number=2,
        feature_name="Mean_Cells_TrackObjects_DistanceTraveled_50",
    ) == 1.0


def test_track_objects_uses_global_image_number_start_for_measurements():
    labels = np.zeros((2, 5, 5), dtype=np.int32)
    labels[0, 1:3, 1:3] = 1
    labels[1, 1:3, 2:4] = 1
    image = np.zeros(labels.shape, dtype=np.float32)

    _output, rows = unwrap(track_objects)(
        image,
        labels=labels,
        object_name="Cells",
        tracking_method="overlap",
        pixel_radius=50,
        image_number_start=22,
    )

    assert _measurement_value(
        rows,
        image_number=22,
        feature_name="TrackObjects_NewObjectCount_Cells_50",
    ) == 1
    assert _measurement_value(
        rows,
        image_number=23,
        object_label=1,
        feature_name="TrackObjects_ParentImageNumber_50",
    ) == 22


def test_track_objects_overlap_allows_split_children_to_inherit_parent_label():
    labels = np.zeros((3, 7, 8), dtype=np.int32)
    labels[0, 1:5, 1:5] = 1
    labels[1, 1:5, 1:3] = 1
    labels[1, 1:5, 3:5] = 2
    labels[2, 2:6, 1:3] = 1
    image = np.zeros(labels.shape, dtype=np.float32)

    _output, rows = unwrap(track_objects)(
        image,
        labels=labels,
        object_name="Cells",
        tracking_method="overlap",
        pixel_radius=50,
    )

    assert _measurement_value(
        rows,
        image_number=2,
        object_label=1,
        feature_name="TrackObjects_Label_50",
    ) == 1
    assert _measurement_value(
        rows,
        image_number=2,
        object_label=2,
        feature_name="TrackObjects_Label_50",
    ) == 1
    assert _measurement_value(
        rows,
        image_number=2,
        feature_name="TrackObjects_NewObjectCount_Cells_50",
    ) == 0
    assert _measurement_value(
        rows,
        image_number=2,
        feature_name="TrackObjects_SplitObjectCount_Cells_50",
    ) == 1
    assert _measurement_value(
        rows,
        image_number=3,
        feature_name="TrackObjects_LostObjectCount_Cells_50",
    ) == 0
    assert _measurement_value(
        rows,
        image_number=3,
        feature_name="TrackObjects_MergedObjectCount_Cells_50",
    ) == 1
    assert _measurement_value(
        rows,
        image_number=3,
        object_label=1,
        feature_name="TrackObjects_DistanceTraveled_50",
    ) == 1.0


def test_track_objects_overlap_counts_distinct_parent_merge_not_loss():
    labels = np.zeros((2, 6, 8), dtype=np.int32)
    labels[0, 1:4, 1:3] = 1
    labels[0, 1:4, 4:6] = 2
    labels[1, 1:4, 1:6] = 1
    image = np.zeros(labels.shape, dtype=np.float32)

    _output, rows = unwrap(track_objects)(
        image,
        labels=labels,
        object_name="Cells",
        tracking_method="overlap",
        pixel_radius=50,
    )

    assert _measurement_value(
        rows,
        image_number=2,
        feature_name="TrackObjects_LostObjectCount_Cells_50",
    ) == 0
    assert _measurement_value(
        rows,
        image_number=2,
        feature_name="TrackObjects_MergedObjectCount_Cells_50",
    ) == 1


def test_track_objects_motion_state_follows_split_parent_object():
    labels = np.zeros((3, 7, 8), dtype=np.int32)
    labels[0, 1:5, 1:5] = 1
    labels[1, 1:5, 1:3] = 1
    labels[1, 1:5, 3:5] = 2
    labels[2, 2:6, 3:5] = 1
    image = np.zeros(labels.shape, dtype=np.float32)

    _output, rows = unwrap(track_objects)(
        image,
        labels=labels,
        object_name="Cells",
        tracking_method="overlap",
        pixel_radius=50,
    )

    assert _measurement_value(
        rows,
        image_number=3,
        object_label=1,
        feature_name="TrackObjects_Label_50",
    ) == 1
    assert _measurement_value(
        rows,
        image_number=3,
        object_label=1,
        feature_name="TrackObjects_ParentObjectNumber_50",
    ) == 2
    assert _measurement_value(
        rows,
        image_number=3,
        object_label=1,
        feature_name="TrackObjects_DistanceTraveled_50",
    ) == 1.0


def test_track_objects_final_age_marks_terminal_track_labels():
    labels = np.zeros((3, 7, 8), dtype=np.int32)
    labels[0, 1:5, 1:5] = 1
    labels[1, 1:5, 1:3] = 1
    labels[1, 1:5, 3:5] = 2
    labels[2, 2:6, 1:3] = 1
    image = np.zeros(labels.shape, dtype=np.float32)

    _output, rows = unwrap(track_objects)(
        image,
        labels=labels,
        object_name="Cells",
        tracking_method="overlap",
        pixel_radius=50,
    )

    assert math.isnan(
        _measurement_value(
            rows,
            image_number=2,
            object_label=2,
            feature_name="TrackObjects_FinalAge_50",
        )
    )
    assert _measurement_value(
        rows,
        image_number=3,
        object_label=1,
        feature_name="TrackObjects_FinalAge_50",
    ) == 3.0
    assert _measurement_value(
        rows,
        image_number=3,
        feature_name="Mean_Cells_TrackObjects_FinalAge_50",
    ) == 3.0
