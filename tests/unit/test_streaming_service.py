from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from openhcs.core.config import (
    FijiDimensionMode,
    FijiStreamingConfig,
    NapariDimensionMode,
    NapariStreamingConfig,
    StreamingConfig,
)
from openhcs.ui.shared.streaming_service import (
    ImageStreamingRequest,
    RoiStreamingRequest,
    StreamingService,
)
from polystore.streaming.viewer_transport import ViewerStreamKwarg
from zmqruntime.viewer_protocol import ViewerBatchWireField


class FakeFileManager:
    def __init__(self) -> None:
        self.saved_batches: list[tuple[list[object], list[str], str, dict]] = []

    def load(self, path: str, read_backend: str) -> str:
        return f"{read_backend}:{Path(path).name}"

    def save_batch(
        self,
        data_list: list[object],
        file_paths: list[str],
        backend: str,
        **metadata,
    ) -> None:
        self.saved_batches.append((data_list, file_paths, backend, metadata))


class FakeRuntimeEndpoint:
    def wait_ready(self, *, timeout: float, require_ready: bool) -> bool:
        del timeout, require_ready
        return True


class FakeViewer:
    port = 5565
    runtime_endpoint = FakeRuntimeEndpoint()


class FakeMetadataHandler:
    def find_metadata_file(self, root: Path) -> Path:
        return root

    def get_component_values(
        self,
        root: Path,
        component_name: str,
    ) -> dict[str, str]:
        return {}


def test_streaming_config_separates_registry_key_from_viewer_identity() -> None:
    assert set(StreamingConfig.__registry__) == {
        "napari_streaming_config",
        "fiji_streaming_config",
    }

    assert NapariStreamingConfig().streaming_config_key == "napari_streaming_config"
    assert NapariStreamingConfig().viewer_type == "napari"
    assert FijiStreamingConfig().streaming_config_key == "fiji_streaming_config"
    assert FijiStreamingConfig().viewer_type == "fiji"
    assert "source" not in NapariStreamingConfig().component_modes()
    assert "source" not in FijiStreamingConfig().component_modes()

    assert StreamingConfig.supported_config_keys() == (
        "fiji_streaming_config",
        "napari_streaming_config",
    )
    assert StreamingConfig.display_name_for_config_key("fiji_streaming_config") == "Fiji"
    assert NapariStreamingConfig().display_name == "Napari"


def test_streaming_config_component_modes_apply_display_defaults() -> None:
    assert NapariStreamingConfig().component_modes() == {
        component: NapariDimensionMode.STACK.value
        for component in NapariStreamingConfig.COMPONENT_ORDER
    }
    assert FijiStreamingConfig().component_modes() == {
        "site": FijiDimensionMode.FRAME.value,
        "timepoint": FijiDimensionMode.FRAME.value,
        "channel": FijiDimensionMode.CHANNEL.value,
        "z_index": FijiDimensionMode.SLICE.value,
        "well": FijiDimensionMode.FRAME.value,
    }


def test_stream_images_uses_resolved_config_backend_not_viewer_name(monkeypatch) -> None:
    monkeypatch.setattr(
        "openhcs.ui.shared.streaming_service.spawn_thread_with_context",
        lambda worker, name: worker(),
    )
    filemanager = FakeFileManager()
    config = FijiStreamingConfig(enabled=True)
    statuses: list[str] = []
    errors: list[str] = []

    service = StreamingService(
        filemanager=filemanager,
        microscope_handler=SimpleNamespace(
            parser=SimpleNamespace(
                parse_filename=lambda filename: {
                    "well": "A01",
                    "site": 1,
                    "channel": 1,
                }
                if filename == "img.tif"
                else None
            ),
            metadata_handler=FakeMetadataHandler(),
        ),
        plate_path=Path("/plate"),
    )
    service.stream_images_async(
        ImageStreamingRequest(
            viewer=FakeViewer(),
            config=config,
            status_callback=statuses.append,
            error_callback=errors.append,
            filenames=("A01/img.tif",),
            read_backend="disk",
        )
    )

    assert errors == []
    assert filemanager.saved_batches
    _data, _paths, backend, metadata = filemanager.saved_batches[0]
    assert backend == config.backend.value
    stream_request = metadata[ViewerStreamKwarg.STREAM_REQUEST.value]
    assert stream_request.display_config is config
    assert stream_request.host == config.host
    assert stream_request.transport_mode is config.transport_mode
    assert stream_request.source.metadata.metadata_by_path == {
        "A01/img.tif": {
            "well": "A01",
            "site": 1,
            "channel": 1,
        }
    }
    assert stream_request.message_extra[
        ViewerBatchWireField.COMPONENT_VALUE_DOMAIN.value
    ] == {
        "site": [1],
        "channel": [1],
        "well": ["A01"],
    }
    assert stream_request.producer.identity.to_payload() == {
        "origin": "manual",
        "output_kind": "manual",
        "output_key": "selected_images",
        "step_name": None,
        "pipeline_position": None,
        "step_scope_id": None,
        "invocation_key": None,
        "artifact_kind": None,
    }


def test_stream_rois_supplies_per_path_component_metadata_from_artifact_name(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "openhcs.ui.shared.streaming_service.spawn_thread_with_context",
        lambda worker, name: worker(),
    )
    monkeypatch.setattr(
        "polystore.roi.load_rois_from_zip",
        lambda _path: [object()],
    )
    filemanager = FakeFileManager()
    config = FijiStreamingConfig(enabled=True)
    roi_filename = "A01_s001_w1_z001_t001_Nuclei_step3_rois.roi.zip"
    microscope_handler = SimpleNamespace(
        parser=SimpleNamespace(
            parse_filename=lambda filename: {
                "well": "A01",
                "site": 1,
                "channel": 1,
                "z_index": 1,
                "timepoint": 1,
            }
            if filename == "A01_s001_w1_z001_t001.tif"
            else None
        ),
        metadata_handler=FakeMetadataHandler(),
    )

    service = StreamingService(
        filemanager=filemanager,
        microscope_handler=microscope_handler,
        plate_path=Path("/plate"),
    )
    service.stream_rois_async(
        RoiStreamingRequest(
            viewer=FakeViewer(),
            config=config,
            status_callback=lambda _status: None,
            error_callback=lambda error: (_ for _ in ()).throw(AssertionError(error)),
            roi_filenames=(roi_filename,),
        )
    )

    assert filemanager.saved_batches
    _data, _paths, _backend, metadata = filemanager.saved_batches[0]
    stream_request = metadata[ViewerStreamKwarg.STREAM_REQUEST.value]
    assert stream_request.source.metadata.metadata_by_path[roi_filename] == {
        "well": "A01",
        "site": 1,
        "channel": 1,
        "z_index": 1,
        "timepoint": 1,
    }
    assert stream_request.message_extra[
        ViewerBatchWireField.COMPONENT_VALUE_DOMAIN.value
    ] == {
        "site": [1],
        "timepoint": [1],
        "channel": [1],
        "z_index": [1],
        "well": ["A01"],
    }


def test_stream_rois_rejects_unresolved_source_plane_metadata() -> None:
    service = StreamingService(
        filemanager=FakeFileManager(),
        microscope_handler=SimpleNamespace(
            parser=SimpleNamespace(parse_filename=lambda _filename: None)
        ),
        plate_path=Path("/plate"),
    )

    with pytest.raises(ValueError, match="Could not resolve source-plane metadata"):
        service.source.roi_component_metadata_by_path(
            ["nuclei1_out_c00_dr90_image_Watershed_step3_rois.roi.zip"],
        )
