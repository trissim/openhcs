from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace

import pytest

from openhcs.core.config import (
    FijiDimensionMode,
    FijiStreamingConfig,
    GlobalPipelineConfig,
    NapariDimensionMode,
    NapariStreamingConfig,
    PipelineConfig,
    StreamingConfig,
    get_all_streaming_ports,
)
from openhcs.core.viewer_streaming_service import (
    ImageStreamingRequest,
    RoiStreamingRequest,
    StreamingService,
    StreamingViewerLifecycle,
)
from openhcs.runtime.fiji_stream_visualizer import FijiStreamVisualizer
from openhcs.runtime.napari_stream_visualizer import NapariStreamVisualizer
from openhcs.runtime.viewer_protocol import (
    DetachedViewerLaunchFailure,
    DetachedViewerServerEntrypointSpec,
    ViewerLaunchContext,
)
from openhcs.runtime.zmq_config import OPENHCS_ZMQ_CONFIG
from polystore.zmq_config import POLYSTORE_ZMQ_CONFIG
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

    def __init__(self, *, settlement_succeeds: bool = True) -> None:
        self.settlement_succeeds = settlement_succeeds
        self.settlement_calls = 0

    def settle_viewer_state(self) -> bool:
        self.settlement_calls += 1
        return self.settlement_succeeds


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
    assert (
        StreamingConfig.display_name_for_config_key("fiji_streaming_config") == "Fiji"
    )
    assert NapariStreamingConfig().display_name == "Napari"


@pytest.mark.parametrize("config", (GlobalPipelineConfig(), PipelineConfig()))
def test_all_streaming_ports_read_declared_registry_fields(config) -> None:
    ports = get_all_streaming_ports(config, num_ports_per_type=1)

    assert set(ports) == {5555, 5565}


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


def test_stream_images_uses_resolved_config_backend_not_viewer_name(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "openhcs.core.viewer_streaming_service.spawn_thread_with_context",
        lambda worker, name: worker(),
    )
    filemanager = FakeFileManager()
    config = FijiStreamingConfig(enabled=True)
    statuses: list[str] = []
    errors: list[str] = []
    transport_config = replace(
        OPENHCS_ZMQ_CONFIG,
        ipc_socket_prefix="test-openhcs-zmq",
    )

    service = StreamingService(
        filemanager=filemanager,
        microscope_handler=SimpleNamespace(
            parser=SimpleNamespace(
                parse_filename=lambda filename: (
                    {
                        "well": "A01",
                        "site": 1,
                        "channel": 1,
                        "z_index": 1,
                        "timepoint": 1,
                    }
                    if filename == "img.tif"
                    else None
                )
            ),
            metadata_handler=FakeMetadataHandler(),
        ),
        plate_path=Path("/plate"),
        transport_config=transport_config,
    )
    viewer = FakeViewer()
    service.stream_images_async(
        ImageStreamingRequest(
            viewer=viewer,
            config=config,
            status_callback=statuses.append,
            error_callback=errors.append,
            filenames=("A01/img.tif",),
            read_backend="disk",
        )
    )

    assert errors == []
    assert viewer.settlement_calls == 1
    assert filemanager.saved_batches
    _data, _paths, backend, metadata = filemanager.saved_batches[0]
    assert backend == config.backend.value
    stream_request = metadata[ViewerStreamKwarg.STREAM_REQUEST.value]
    assert stream_request.display_config is config
    assert stream_request.host == config.host
    assert stream_request.transport_mode is config.transport_mode
    assert (
        stream_request.transport_config.resolve(POLYSTORE_ZMQ_CONFIG)
        is transport_config
    )
    assert stream_request.source.metadata.metadata_by_path == {
        "A01/img.tif": {
            "well": "A01",
            "site": 1,
            "channel": 1,
            "z_index": 1,
            "timepoint": 1,
        }
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
    assert stream_request.producer.identities[0].to_payload() == {
        "origin": "manual",
        "output_kind": "manual",
        "output_key": "selected_images",
        "projection_key": "selected_images",
        "step_name": None,
        "pipeline_position": None,
        "step_scope_id": None,
        "invocation_key": None,
        "artifact_kind": None,
    }


def test_stream_images_reports_success_only_after_viewer_state_is_settled() -> None:
    events: list[str] = []

    class SettledStateViewer(FakeViewer):
        layer_state: dict[str, object] | None = None

        def settle_viewer_state(self) -> bool:
            events.append("settle")
            self.layer_state = {
                "mounted": True,
                "pending_update": False,
                "axis_labels": ("channel", "y", "x"),
                "payload_nonzero_counts": (4, 5),
                "component_values": (
                    {"channel": 1},
                    {"channel": 2},
                ),
            }
            return super().settle_viewer_state()

    viewer = SettledStateViewer()
    statuses: list[str] = []

    def record_status(message: str) -> None:
        statuses.append(message)
        if not message.startswith("Streamed "):
            return
        events.append("success")
        state = viewer.layer_state
        assert state is not None
        assert state["mounted"] is True
        assert state["pending_update"] is False
        assert state["axis_labels"] == ("channel", "y", "x")
        assert all(state["payload_nonzero_counts"])
        coordinates = tuple(
            tuple(sorted(values.items())) for values in state["component_values"]
        )
        assert len(coordinates) == len(set(coordinates))

    filemanager = FakeFileManager()
    service = StreamingService(
        filemanager=filemanager,
        microscope_handler=SimpleNamespace(
            parser=SimpleNamespace(
                parse_filename=lambda filename: {
                    "well": "A01",
                    "site": 1,
                    "channel": int(filename[1]),
                    "z_index": 1,
                    "timepoint": 1,
                }
            ),
            metadata_handler=FakeMetadataHandler(),
        ),
        plate_path=Path("/plate"),
    )

    result = service.stream_images(
        ImageStreamingRequest(
            viewer=viewer,
            config=NapariStreamingConfig(enabled=True),
            status_callback=record_status,
            error_callback=lambda error: (_ for _ in ()).throw(AssertionError(error)),
            filenames=("w1.tif", "w2.tif"),
            read_backend="disk",
        )
    )

    assert result.streamed_count == 2
    assert events == ["settle", "success"]
    assert viewer.settlement_calls == 1


def test_stream_images_does_not_report_success_when_viewer_settlement_fails() -> None:
    statuses: list[str] = []
    viewer = FakeViewer(settlement_succeeds=False)
    service = StreamingService(
        filemanager=FakeFileManager(),
        microscope_handler=SimpleNamespace(
            parser=SimpleNamespace(
                parse_filename=lambda _filename: {
                    "well": "A01",
                    "site": 1,
                    "channel": 1,
                    "z_index": 1,
                    "timepoint": 1,
                }
            ),
            metadata_handler=FakeMetadataHandler(),
        ),
        plate_path=Path("/plate"),
    )

    with pytest.raises(RuntimeError, match="Failed to settle streamed updates"):
        service.stream_images(
            ImageStreamingRequest(
                viewer=viewer,
                config=NapariStreamingConfig(enabled=True),
                status_callback=statuses.append,
                error_callback=lambda _error: None,
                filenames=("w1.tif",),
                read_backend="disk",
            )
        )

    assert viewer.settlement_calls == 1
    assert all(not status.startswith("Streamed ") for status in statuses)


def test_stream_rois_supplies_per_path_component_metadata_from_artifact_name(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "openhcs.core.viewer_streaming_service.spawn_thread_with_context",
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
            parse_filename=lambda filename: (
                {
                    "well": "A01",
                    "site": 1,
                    "channel": 1,
                    "z_index": 1,
                    "timepoint": 1,
                }
                if filename == "A01_s001_w1_z001_t001.tif"
                else None
            )
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


def test_stream_rois_uses_explicit_component_metadata(monkeypatch) -> None:
    monkeypatch.setattr(
        "polystore.roi.load_rois_from_zip",
        lambda _path: [object()],
    )
    filemanager = FakeFileManager()
    config = FijiStreamingConfig(enabled=True)
    roi_path = "/output/images_results/A01_w2_rois.roi.zip"
    service = StreamingService(
        filemanager=filemanager,
        microscope_handler=SimpleNamespace(
            parser=SimpleNamespace(
                parse_filename=lambda _filename: (_ for _ in ()).throw(
                    AssertionError("parser fallback used")
                )
            ),
            metadata_handler=FakeMetadataHandler(),
        ),
        plate_path=Path("/plate"),
    )

    viewer = FakeViewer()
    service.stream_rois(
        RoiStreamingRequest(
            viewer=viewer,
            config=config,
            status_callback=lambda _status: None,
            error_callback=lambda error: (_ for _ in ()).throw(AssertionError(error)),
            roi_filenames=(roi_path,),
            component_metadata_by_path={
                roi_path: {
                    "well": "A01",
                    "site": 1,
                    "channel": 2,
                    "z_index": 1,
                    "timepoint": 1,
                }
            },
        )
    )

    assert viewer.settlement_calls == 1
    assert filemanager.saved_batches
    _data, _paths, _backend, metadata = filemanager.saved_batches[0]
    stream_request = metadata[ViewerStreamKwarg.STREAM_REQUEST.value]
    assert stream_request.source.metadata.metadata_by_path[roi_path] == {
        "well": "A01",
        "site": 1,
        "channel": 2,
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
        service.source.roi_component_metadata_by_path(
            ["nuclei1_out_c00_dr90_image_Watershed_step3_rois.roi.zip"],
        )


def test_streaming_viewer_lifecycle_attaches_existing_viewer_without_restart(
    monkeypatch,
) -> None:
    class FakeManager:
        def get_viewer(self, viewer_type: str, port: int):
            del viewer_type, port
            return None

        def release_viewer(
            self, viewer_type: str, port: int, *, stop: bool, force: bool
        ):
            raise AssertionError("fresh release should not run for non-fresh attach")

    monkeypatch.setattr(
        "zmqruntime.ViewerStateManager.get_instance",
        lambda: FakeManager(),
    )
    monkeypatch.setattr(
        "zmqruntime.get_or_create_viewer",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("viewer restart path used")
        ),
    )
    monkeypatch.setattr(
        NapariStreamVisualizer,
        "existing_viewer_is_ready",
        lambda self: True,
    )

    viewer = StreamingViewerLifecycle.get_or_create_visualizer(
        filemanager=FakeFileManager(),
        config=NapariStreamingConfig(enabled=True, port=5563, persistent=True),
        fresh=False,
        launch_context=ViewerLaunchContext.headless(),
    )

    assert isinstance(viewer, NapariStreamVisualizer)
    assert viewer.lifecycle_state.is_connected_external


@pytest.mark.parametrize(
    ("config", "visualizer_type"),
    (
        (
            NapariStreamingConfig(enabled=True, port=5563, persistent=True),
            NapariStreamVisualizer,
        ),
        (
            FijiStreamingConfig(enabled=True, port=5564, persistent=True),
            FijiStreamVisualizer,
        ),
    ),
)
def test_streaming_viewer_lifecycle_projects_launch_context_for_every_viewer(
    monkeypatch,
    config,
    visualizer_type,
) -> None:
    class FakeManager:
        def get_viewer(self, viewer_type: str, port: int):
            del viewer_type, port
            return None

        def release_viewer(
            self, viewer_type: str, port: int, *, stop: bool, force: bool
        ):
            raise AssertionError("fresh release should not run for non-fresh attach")

    monkeypatch.setattr(
        "zmqruntime.ViewerStateManager.get_instance",
        lambda: FakeManager(),
    )
    monkeypatch.setattr(
        visualizer_type,
        "existing_viewer_is_ready",
        lambda self: True,
    )
    launch_context = ViewerLaunchContext.projected_graphical_session(
        {
            "DISPLAY": ":31",
            "XAUTHORITY": "/run/user/1000/xauth",
            "XDG_RUNTIME_DIR": "/run/user/1000",
        }
    )

    viewer = StreamingViewerLifecycle.get_or_create_visualizer(
        filemanager=FakeFileManager(),
        config=config,
        fresh=False,
        launch_context=launch_context,
    )

    assert isinstance(viewer, visualizer_type)
    assert viewer.detached_launch_request().launch_context is launch_context


def test_streaming_viewer_lifecycle_reports_bounded_launch_log(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class FakeManager:
        def release_viewer(
            self, viewer_type: str, port: int, *, stop: bool, force: bool
        ):
            del viewer_type, port, stop, force

    def fail_after_factory(*, factory, **_kwargs):
        viewer = factory()
        log_file = viewer.detached_launch_request().log_file
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text(
            "\n".join(f"startup-{index:03d}" for index in range(100)),
            encoding="utf-8",
        )
        raise RuntimeError("napari process terminated unexpectedly during startup")

    monkeypatch.setattr(
        "zmqruntime.ViewerStateManager.get_instance",
        lambda: FakeManager(),
    )
    monkeypatch.setattr("zmqruntime.get_or_create_viewer", fail_after_factory)
    monkeypatch.setattr(
        DetachedViewerServerEntrypointSpec,
        "log_file_for",
        lambda self, port: tmp_path / f"{self.viewer_type.value}_{port}.log",
    )

    with pytest.raises(DetachedViewerLaunchFailure) as error:
        StreamingViewerLifecycle.get_or_create_visualizer(
            filemanager=FakeFileManager(),
            config=NapariStreamingConfig(enabled=True, port=5563, persistent=True),
        )

    assert error.value.log_file == tmp_path / "napari_5563.log"
    assert error.value.log_tail.endswith("startup-099")
    assert "startup-000" not in error.value.log_tail
    assert str(error.value.log_file) in str(error.value)


def test_streaming_viewer_lifecycle_reuses_manager_owned_viewer(monkeypatch) -> None:
    existing_viewer = FakeViewer()

    class FakeManager:
        def get_viewer(self, viewer_type: str, port: int):
            assert viewer_type == "napari"
            assert port == 5563
            return existing_viewer

        def release_viewer(
            self, viewer_type: str, port: int, *, stop: bool, force: bool
        ):
            raise AssertionError("fresh release should not run for non-fresh reuse")

    monkeypatch.setattr(
        "zmqruntime.ViewerStateManager.get_instance",
        lambda: FakeManager(),
    )
    monkeypatch.setattr(
        NapariStreamingConfig,
        "create_visualizer",
        lambda self, filemanager, visualizer_config=None: (_ for _ in ()).throw(
            AssertionError("external viewer probe should not run")
        ),
    )

    viewer = StreamingViewerLifecycle.get_or_create_visualizer(
        filemanager=FakeFileManager(),
        config=NapariStreamingConfig(enabled=True, port=5563, persistent=True),
        fresh=False,
    )

    assert viewer is existing_viewer
