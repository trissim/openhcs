from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from openhcs.core.config import (
    FijiStreamingConfig,
    NapariStreamingConfig,
    StreamingConfig,
)
from openhcs.ui.shared.streaming_service import (
    ImageStreamingRequest,
    RoiStreamingRequest,
    StreamingService,
    ViewerStreamingContext,
)


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


class FakeViewer:
    port = 5565

    def wait_for_ready(self, timeout: float) -> bool:
        return True


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

    assert StreamingService.supported_viewer_types() == [
        "fiji_streaming_config",
        "napari_streaming_config",
    ]
    assert StreamingService.display_name_for_viewer_type("fiji_streaming_config") == "Fiji"
    assert StreamingService.display_name_for_viewer_type("napari") == "Napari"


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
        microscope_handler=object(),
        plate_path=Path("/plate"),
    )
    service.stream_images_async(
        ImageStreamingRequest(
            context=ViewerStreamingContext(
                viewer=FakeViewer(),
                plate_path=Path("/plate"),
                config=config,
                viewer_type=config.viewer_type,
                status_callback=statuses.append,
                error_callback=errors.append,
            ),
            filenames=("A01/img.tif",),
            read_backend="disk",
        )
    )

    assert errors == []
    assert filemanager.saved_batches
    _data, _paths, backend, metadata = filemanager.saved_batches[0]
    assert backend == config.backend.value
    assert metadata["display_config"] is config
    assert metadata["host"] == config.host
    assert metadata["transport_mode"] is config.transport_mode
    assert metadata["producer_identity"].to_payload() == {
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
        )
    )

    service = StreamingService(
        filemanager=filemanager,
        microscope_handler=microscope_handler,
        plate_path=Path("/plate"),
    )
    service.stream_rois_async(
        RoiStreamingRequest(
            context=ViewerStreamingContext(
                viewer=FakeViewer(),
                plate_path=Path("/plate"),
                config=config,
                viewer_type=config.viewer_type,
                status_callback=lambda _status: None,
                error_callback=lambda error: (_ for _ in ()).throw(AssertionError(error)),
            ),
            roi_filenames=(roi_filename,),
        )
    )

    assert filemanager.saved_batches
    _data, _paths, _backend, metadata = filemanager.saved_batches[0]
    assert metadata["component_metadata_by_path"][roi_filename] == {
        "well": "A01",
        "site": 1,
        "channel": 1,
        "z_index": 1,
        "timepoint": 1,
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
        service._roi_component_metadata_by_path(
            ["nuclei1_out_c00_dr90_image_Watershed_step3_rois.roi.zip"],
        )
