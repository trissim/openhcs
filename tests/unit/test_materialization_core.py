import json

import numpy as np
import pytest

from polystore.filemanager import FileManager
from polystore.fiji_stream import FijiStreamingBackend
from polystore.memory import MemoryStorageBackend
from polystore.napari_stream import NapariStreamingBackend

from openhcs.processing.materialization import (
    AlignedROIMask,
    AlignedROIMasks,
    JsonOptions,
    MaterializationSpec,
    ROIOptions,
    materialize,
)


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
def test_aligned_roi_masks_preserve_channel_and_role_through_stream_payload() -> None:
    fm = FileManager({"memory": MemoryStorageBackend()})
    w1_mask = np.zeros((32, 32), dtype=np.int32)
    w1_mask[3:8, 3:8] = 1
    w2_mask = np.zeros((32, 32), dtype=np.int32)
    w2_mask[20:26, 20:26] = 1
    payload = AlignedROIMasks(
        (
            AlignedROIMask(
                w1_mask,
                source_index=0,
                role="w1_nuclei",
                label_metadata={1: {"w2_positive": False}},
            ),
            AlignedROIMask(w2_mask, source_index=1, role="w2_stain"),
        )
    )
    source_paths = [
        "/plate/images/A01_s001_w1_z001_t001.tif",
        "/plate/images/A01_s001_w2_z001_t001.tif",
    ]
    spec = MaterializationSpec(ROIOptions())

    primary_path = materialize(
        spec,
        data=payload,
        path="/tmp/A01_s001_w1_z001_t001_colocalization_masks_step0.roi.zip",
        filemanager=fm,
        backends=["memory"],
        source_paths=source_paths,
        output_key="colocalization_masks",
        step_index=0,
    )

    w1_path = (
        "/tmp/A01_s001_w1_z001_t001_colocalization_masks_"
        "w1_nuclei_step0_rois.roi.zip"
    )
    w2_path = (
        "/tmp/A01_s001_w2_z001_t001_colocalization_masks_" "w2_stain_step0_rois.roi.zip"
    )
    assert primary_path == w1_path
    w1_rois = fm.load(w1_path, "memory")
    w2_rois = fm.load(w2_path, "memory")
    assert w1_rois[0].metadata["roi_role"] == "w1_nuclei"
    assert w1_rois[0].metadata["w2_positive"] is False
    assert w2_rois[0].metadata["roi_role"] == "w2_stain"

    class _Parser:
        @staticmethod
        def parse_filename(filename):
            channel = 1 if "_w1_" in filename else 2
            return {"well": "A01", "site": 1, "channel": channel}

    class _MicroscopeHandler:
        parser = _Parser()

    streaming_backend = NapariStreamingBackend()
    batch_items, _ = streaming_backend._prepare_batch_items(
        [w1_rois, w2_rois],
        [w1_path, w2_path],
        _MicroscopeHandler(),
        "Dual channel simple count",
        streaming_backend._prepare_batch_item,
    )

    assert [item["metadata"]["channel"] for item in batch_items] == [1, 2]
    assert [item["data_type"] for item in batch_items] == ["shapes", "shapes"]
    assert batch_items[0]["shapes"][0]["metadata"]["roi_role"] == "w1_nuclei"
    assert batch_items[1]["shapes"][0]["metadata"]["roi_role"] == "w2_stain"

    fiji_backend = FijiStreamingBackend()
    fiji_items, _ = fiji_backend._prepare_batch_items(
        [w1_rois, w2_rois],
        [w1_path, w2_path],
        _MicroscopeHandler(),
        "Dual channel simple count",
        fiji_backend._prepare_batch_item,
    )
    assert [item["metadata"]["channel"] for item in fiji_items] == [1, 2]
    assert [item["data_type"] for item in fiji_items] == ["rois", "rois"]
    assert all(item["rois"] for item in fiji_items)
