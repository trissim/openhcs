import json

import numpy as np
import pytest

import openhcs  # noqa: F401
from polystore.filemanager import FileManager
from polystore.memory import MemoryStorageBackend

from openhcs.processing.materialization import (
    JsonOptions,
    MaterializationSpec,
    ROIOptions,
    csv_only,
    json_materializer,
    json_only,
    materialize,
    tabular_field_names_from_materialization,
    tiff_stack,
)
from openhcs.core.runtime_values import ObjectLabelPayload


@pytest.mark.unit
def test_materialize_strips_roi_zip_compound_suffix_for_json() -> None:
    fm = FileManager({"memory": MemoryStorageBackend()})

    spec = MaterializationSpec(JsonOptions(filename_suffix=".json"))
    out = materialize(
        spec,
        data={"ok": True},
        path="/tmp/A01_test_output.roi.zip",
        filemanager=fm,
        backends=["memory"],
        backend_kwargs={},
    )

    assert out == "/tmp/A01_test_output.json"
    assert fm.load(out, "memory") == json.dumps({"ok": True}, indent=2, default=str)


@pytest.mark.unit
def test_csv_materialization_preserves_declared_fields_for_empty_rows() -> None:
    fm = FileManager({"memory": MemoryStorageBackend()})

    out = materialize(
        csv_only(fields=["object_label", "area"]),
        data=[],
        path="/tmp/A01_measurements",
        filemanager=fm,
        backends=["memory"],
        backend_kwargs={},
    )

    assert out == "/tmp/A01_measurements_details.csv"
    assert fm.load(out, "memory").splitlines()[0] == "object_label,area"


@pytest.mark.unit
def test_csv_materialization_preserves_declared_field_order_for_dict_rows() -> None:
    fm = FileManager({"memory": MemoryStorageBackend()})

    out = materialize(
        csv_only(fields=["object_label", "area"]),
        data=[{"area": 9, "object_label": 1, "ignored": 99}],
        path="/tmp/A01_measurements",
        filemanager=fm,
        backends=["memory"],
        backend_kwargs={},
    )

    assert out == "/tmp/A01_measurements_details.csv"
    assert fm.load(out, "memory").splitlines() == ["object_label,area", "1,9"]


@pytest.mark.unit
def test_json_materialization_filters_to_declared_fields() -> None:
    fm = FileManager({"memory": MemoryStorageBackend()})

    out = materialize(
        json_materializer(fields=["object_label", "area"]),
        data=[{"object_label": 1, "area": 9, "ignored": 99}],
        path="/tmp/A01_measurements",
        filemanager=fm,
        backends=["memory"],
        backend_kwargs={},
    )

    assert out == "/tmp/A01_measurements.json"
    assert json.loads(fm.load(out, "memory")) == [{"object_label": 1, "area": 9}]


@pytest.mark.unit
def test_tabular_field_names_from_materialization_reads_csv_and_json_options() -> None:
    assert tabular_field_names_from_materialization(
        csv_only(fields=["object_label", "area"])
    ) == ("object_label", "area")
    assert tabular_field_names_from_materialization(
        json_only(fields=["object_label", "area"])
    ) == ("object_label", "area")


@pytest.mark.unit
def test_tiff_stack_preserves_channels_last_color_image_as_single_file() -> None:
    fm = FileManager({"memory": MemoryStorageBackend()})
    image = np.zeros((5, 7, 3), dtype=np.uint8)

    out = materialize(
        tiff_stack(),
        data=image,
        path="/tmp/A01_rgb",
        filemanager=fm,
        backends=["memory"],
        backend_kwargs={},
    )

    assert out == "/tmp/A01_rgb_slice_000.tif"
    assert fm.exists(out, "memory")
    assert not fm.exists("/tmp/A01_rgb_slice_001.tif", "memory")


@pytest.mark.unit
def test_tiff_stack_splits_scalar_3d_stack_by_plane() -> None:
    fm = FileManager({"memory": MemoryStorageBackend()})
    stack = np.zeros((3, 5, 7), dtype=np.uint8)

    out = materialize(
        tiff_stack(),
        data=stack,
        path="/tmp/A01_stack",
        filemanager=fm,
        backends=["memory"],
        backend_kwargs={},
    )

    assert out == "/tmp/A01_stack_slice_000.tif"
    assert fm.exists("/tmp/A01_stack_slice_000.tif", "memory")
    assert fm.exists("/tmp/A01_stack_slice_001.tif", "memory")
    assert fm.exists("/tmp/A01_stack_slice_002.tif", "memory")
    assert not fm.exists("/tmp/A01_stack_slice_003.tif", "memory")


@pytest.mark.unit
def test_roi_materialization_offsets_object_label_payload_geometry() -> None:
    fm = FileManager({"memory": MemoryStorageBackend()})
    labels = np.zeros((8, 8), dtype=np.int32)
    labels[2:6, 3:7] = 1
    payload = ObjectLabelPayload(labels=labels, spatial_origin_yx=(10, 20))

    out = materialize(
        MaterializationSpec(ROIOptions(min_area=0)),
        data=payload,
        path="/tmp/A01_Nuclei_step3.roi.zip",
        filemanager=fm,
        backends=["memory"],
        backend_kwargs={},
    )

    rois = fm.load(out, "memory")
    assert out == "/tmp/A01_Nuclei_step3_rois.roi.zip"
    assert rois[0].metadata["bbox"] == (12, 23, 16, 27)
    assert rois[0].metadata["centroid"] == (13.5, 24.5)
    assert float(rois[0].shapes[0].coordinates[:, 0].min()) >= 11.5
    assert float(rois[0].shapes[0].coordinates[:, 1].min()) >= 22.5


@pytest.mark.unit
def test_roi_materialization_extracts_each_plane_from_object_label_stack() -> None:
    fm = FileManager({"memory": MemoryStorageBackend()})
    labels = np.zeros((2, 8, 8), dtype=np.int32)
    labels[0, 1:4, 1:4] = 1
    labels[1, 4:7, 4:7] = 2
    payload = ObjectLabelPayload(labels=labels)

    out = materialize(
        MaterializationSpec(ROIOptions(min_area=0)),
        data=payload,
        path="/tmp/A01_Nuclei_step3.roi.zip",
        filemanager=fm,
        backends=["memory"],
        backend_kwargs={},
    )

    rois = fm.load(out, "memory")
    assert out == "/tmp/A01_Nuclei_step3_rois.roi.zip"
    assert [roi.metadata["label"] for roi in rois] == [1, 2]
    assert [roi.metadata["plane_indices"] for roi in rois] == [(0,), (1,)]
    assert all(roi.metadata["plane_shape"] == (2,) for roi in rois)


@pytest.mark.unit
def test_roi_materialization_treats_non_spatial_label_payload_as_empty() -> None:
    fm = FileManager({"memory": MemoryStorageBackend()})
    payload = ObjectLabelPayload(labels=np.asarray(0, dtype=np.int32))

    out = materialize(
        MaterializationSpec(ROIOptions(min_area=0)),
        data=payload,
        path="/tmp/A01_Worms_step3.roi.zip",
        filemanager=fm,
        backends=["memory"],
        backend_kwargs={},
    )

    assert out == "/tmp/A01_Worms_step3_segmentation_summary.txt"
    assert "No ROIs extracted" in fm.load(out, "memory")
