from pathlib import Path

from openhcs.agent.dto.execution import ExecutionConnectionSpec
from openhcs.agent.dto.plate import (
    PlateFileStreamRequest,
)
from openhcs.agent.services.plate_streaming_service import PlateStreamingService
from openhcs.agent.services.plate_inspection_service import PlateInspectionContext
from openhcs.constants.constants import FileFormat
from openhcs.core.plate_image_inventory import (
    PlateFileInventory,
    PlateFileKind,
    PlateImageRecord,
    PlateResultFileRecord,
)
from openhcs.runtime.viewer_protocol import DetachedViewerLaunchFailure, ViewerType


class FakeHandler:
    microscope_type = "openhcsdata"

    def get_primary_backend(self, plate_path, filemanager):
        del plate_path, filemanager
        return "disk"


class FakeInspectionService:
    def __init__(self, inventory: PlateFileInventory) -> None:
        self.inventory = inventory
        self.open_requests = []
        self.resolve_requests = []
        self.inventory_contexts = []
        self.inventory_kinds = []

    def open_context(self, request):
        self.open_requests.append(request)
        return PlateInspectionContext(
            plate_path=Path(request.plate_path),
            filemanager=object(),
            handler=FakeHandler(),
            parser=None,
        ), (), ()

    def resolve_plate_path(self, plate_path):
        self.resolve_requests.append(plate_path)
        return Path(plate_path), ()

    def file_inventory(self, context, *, kind):
        self.inventory_contexts.append(context)
        self.inventory_kinds.append(kind)
        return self.inventory, ()


class FakeViewer:
    port = 5565


def test_plate_streaming_service_projects_launch_environment(monkeypatch):
    captured = {}
    inventory = PlateFileInventory(
        plate_path=Path("/plate"),
        image_records=(
            PlateImageRecord(
                virtual_path="A01_s001_w1_z001_t001.tif",
                full_virtual_path="/plate/A01_s001_w1_z001_t001.tif",
                backend="virtual_workspace",
                source_path="/raw/source_A01.tif",
                metadata={"well": "A01"},
            ),
        ),
        result_records=(),
    )

    def fake_viewer(**kwargs):
        captured["launch_environment"] = kwargs["launch_environment"]
        return FakeViewer()

    monkeypatch.setattr(
        "openhcs.agent.services.plate_streaming_service."
        "StreamingViewerLifecycle.get_or_create_visualizer",
        fake_viewer,
    )
    monkeypatch.setattr(
        "openhcs.agent.services.plate_streaming_service.StreamingService.stream_images",
        lambda self, request: None,
    )
    environment = {
        "DISPLAY": ":41",
        "XAUTHORITY": "/run/user/1000/xauth",
        "XDG_RUNTIME_DIR": "/run/user/1000",
    }

    result = PlateStreamingService(FakeInspectionService(inventory)).stream_files(
        PlateFileStreamRequest(
            plate_path="/plate",
            file_paths=("A01_s001_w1_z001_t001.tif",),
        ),
        launch_environment=environment,
    )

    assert result.errors == ()
    assert captured["launch_environment"] == environment


def test_plate_streaming_service_projects_detached_log_diagnostics(monkeypatch, tmp_path):
    inventory = PlateFileInventory(
        plate_path=Path("/plate"),
        image_records=(
            PlateImageRecord(
                virtual_path="A01_s001_w1_z001_t001.tif",
                full_virtual_path="/plate/A01_s001_w1_z001_t001.tif",
                backend="virtual_workspace",
                source_path="/raw/source_A01.tif",
                metadata={"well": "A01"},
            ),
        ),
        result_records=(),
    )
    log_file = tmp_path / "napari_detached_port_5555.log"
    failure = DetachedViewerLaunchFailure(
        viewer_type=ViewerType.NAPARI,
        port=5555,
        cause=RuntimeError("Qt/xcb could not connect to display"),
        log_file=log_file,
        log_tail="qt.qpa.plugin: Could not load the Qt platform plugin xcb",
    )
    monkeypatch.setattr(
        "openhcs.agent.services.plate_streaming_service."
        "StreamingViewerLifecycle.get_or_create_visualizer",
        lambda **_kwargs: (_ for _ in ()).throw(failure),
    )

    result = PlateStreamingService(FakeInspectionService(inventory)).stream_files(
        PlateFileStreamRequest(
            plate_path="/plate",
            file_paths=("A01_s001_w1_z001_t001.tif",),
        )
    )

    error = result.errors[0]
    assert error.code == "plate_file_stream_failed"
    assert error.path == str(log_file)
    assert "Qt platform plugin xcb" in error.message
    assert "bounded tail" in error.hint


def test_plate_streaming_service_streams_virtual_image_path(monkeypatch):
    captured = {}
    inventory = PlateFileInventory(
        plate_path=Path("/plate"),
        image_records=(
            PlateImageRecord(
                virtual_path="A01_s001_w1_z001_t001.tif",
                full_virtual_path="/plate/A01_s001_w1_z001_t001.tif",
                backend="virtual_workspace",
                source_path="/raw/source_A01.tif",
                metadata={"well": "A01"},
            ),
        ),
        result_records=(),
    )

    monkeypatch.setattr(
        "openhcs.agent.services.plate_streaming_service."
        "StreamingViewerLifecycle.get_or_create_visualizer",
        lambda **_kwargs: FakeViewer(),
    )

    def fake_stream_images(self, request):
        del self
        captured["filenames"] = request.filenames
        captured["read_backend"] = request.read_backend
        request.status_callback("streamed image")

    monkeypatch.setattr(
        "openhcs.agent.services.plate_streaming_service."
        "StreamingService.stream_images",
        fake_stream_images,
    )

    result = PlateStreamingService(FakeInspectionService(inventory)).stream_files(
        PlateFileStreamRequest(
            plate_path="/plate",
            file_paths=("A01_s001_w1_z001_t001.tif",),
            connection=ExecutionConnectionSpec(port=5565, transport_mode="ipc"),
        )
    )

    assert result.errors == ()
    assert result.viewer_type == "napari"
    assert result.connection.port == 5565
    assert result.streamed_image_paths == ("A01_s001_w1_z001_t001.tif",)
    assert result.streamed_roi_paths == ()
    assert captured == {
        "filenames": ("A01_s001_w1_z001_t001.tif",),
        "read_backend": "disk",
    }
    assert result.status_messages == ("streamed image",)
    assert result.resolved_records[0].virtual_path == "A01_s001_w1_z001_t001.tif"


def test_plate_streaming_service_rejects_non_roi_result_files(monkeypatch):
    inventory = PlateFileInventory(
        plate_path=Path("/plate"),
        image_records=(),
        result_records=(
            PlateResultFileRecord(
                relative_path="images_results/A01_counts.csv",
                full_path="/plate/images_results/A01_counts.csv",
                file_format=FileFormat.CSV,
                metadata={"well": "A01"},
            ),
        ),
    )
    monkeypatch.setattr(
        "openhcs.agent.services.plate_streaming_service."
        "StreamingViewerLifecycle.get_or_create_visualizer",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("viewer launched")),
    )

    result = PlateStreamingService(FakeInspectionService(inventory)).stream_files(
        PlateFileStreamRequest(
            plate_path="/plate",
            file_paths=("images_results/A01_counts.csv",),
            kind=PlateFileKind.RESULT,
        )
    )

    assert result.errors[0].code == "plate_file_stream_no_streamable_records"
    assert result.skipped_records[0].relative_path == "images_results/A01_counts.csv"
    assert result.streamed_image_paths == ()
    assert result.streamed_roi_paths == ()


def test_plate_streaming_service_streams_roi_result_full_path_and_metadata(monkeypatch):
    captured = {}
    roi_full_path = "/plate_openhcs/images_results/A01_w1_rois.roi.zip"
    inventory = PlateFileInventory(
        plate_path=Path("/plate"),
        image_records=(),
        result_records=(
            PlateResultFileRecord(
                relative_path="images_results/A01_w1_rois.roi.zip",
                full_path=roi_full_path,
                file_format=FileFormat.ROI,
                metadata={
                    "well": "A01",
                    "site": 1,
                    "channel": 1,
                    "z_index": 1,
                    "timepoint": 1,
                },
            ),
        ),
    )
    monkeypatch.setattr(
        "openhcs.agent.services.plate_streaming_service."
        "StreamingViewerLifecycle.get_or_create_visualizer",
        lambda **_kwargs: FakeViewer(),
    )

    def fake_stream_rois(self, request):
        del self
        captured["roi_filenames"] = request.roi_filenames
        captured["component_metadata_by_path"] = request.component_metadata_by_path
        request.status_callback("streamed rois")

    monkeypatch.setattr(
        "openhcs.agent.services.plate_streaming_service."
        "StreamingService.stream_rois",
        fake_stream_rois,
    )

    result = PlateStreamingService(FakeInspectionService(inventory)).stream_files(
        PlateFileStreamRequest(
            plate_path="/plate",
            file_paths=("images_results/A01_w1_rois.roi.zip",),
            kind=PlateFileKind.RESULT,
        )
    )

    assert result.errors == ()
    assert result.streamed_image_paths == ()
    assert result.streamed_roi_paths == (roi_full_path,)
    assert captured == {
        "roi_filenames": (roi_full_path,),
        "component_metadata_by_path": {
            roi_full_path: {
                "well": "A01",
                "site": 1,
                "channel": 1,
                "z_index": 1,
                "timepoint": 1,
            }
        },
    }
    assert result.status_messages == ("streamed rois",)


def test_plate_streaming_service_reports_explicit_result_excluded_by_default_kind(
    monkeypatch,
):
    roi_full_path = "/plate/images_results/A01_w1_rois.roi.zip"
    inventory = PlateFileInventory(
        plate_path=Path("/plate"),
        image_records=(),
        result_records=(
            PlateResultFileRecord(
                relative_path="images_results/A01_w1_rois.roi.zip",
                full_path=roi_full_path,
                file_format=FileFormat.ROI,
                metadata={"well": "A01", "channel": 1},
            ),
        ),
    )
    inspection_service = FakeInspectionService(inventory)
    monkeypatch.setattr(
        "openhcs.agent.services.plate_streaming_service."
        "StreamingViewerLifecycle.get_or_create_visualizer",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("viewer launched")),
    )

    result = PlateStreamingService(inspection_service).stream_files(
        PlateFileStreamRequest(
            plate_path="/plate",
            file_paths=("images_results/A01_w1_rois.roi.zip",),
        )
    )

    assert inspection_service.inventory_kinds == [None]
    assert result.errors[0].code == "plate_file_stream_failed"
    assert "exists as kind 'result'" in result.errors[0].message
    assert "excluded by the requested kind filter (image)" in result.errors[0].message


def test_plate_streaming_service_query_limit_counts_streamable_roi_results(monkeypatch):
    captured = {}
    roi_one_path = "/plate_openhcs/checkpoints_step7_results/A01_w1_rois.roi.zip"
    roi_two_path = "/plate_openhcs/checkpoints_step7_results/A01_w2_rois.roi.zip"
    inventory = PlateFileInventory(
        plate_path=Path("/plate_openhcs"),
        image_records=(),
        result_records=(
            PlateResultFileRecord(
                relative_path="checkpoints_step7_results/A01_measurements.json",
                full_path="/plate_openhcs/checkpoints_step7_results/A01_measurements.json",
                file_format=FileFormat.JSON,
                metadata={"well": "A01"},
            ),
            PlateResultFileRecord(
                relative_path="checkpoints_step7_results/A01_summary.csv",
                full_path="/plate_openhcs/checkpoints_step7_results/A01_summary.csv",
                file_format=FileFormat.CSV,
                metadata={"well": "A01"},
            ),
            PlateResultFileRecord(
                relative_path="checkpoints_step7_results/A01_w1_rois.roi.zip",
                full_path=roi_one_path,
                file_format=FileFormat.ROI,
                metadata={"well": "A01", "channel": 1},
            ),
            PlateResultFileRecord(
                relative_path="checkpoints_step7_results/A01_w2_rois.roi.zip",
                full_path=roi_two_path,
                file_format=FileFormat.ROI,
                metadata={"well": "A01", "channel": 2},
            ),
        ),
    )
    monkeypatch.setattr(
        "openhcs.agent.services.plate_streaming_service."
        "StreamingViewerLifecycle.get_or_create_visualizer",
        lambda **_kwargs: FakeViewer(),
    )

    def fake_stream_rois(self, request):
        del self
        captured["roi_filenames"] = request.roi_filenames
        captured["component_metadata_by_path"] = request.component_metadata_by_path
        request.status_callback("streamed query rois")

    monkeypatch.setattr(
        "openhcs.agent.services.plate_streaming_service."
        "StreamingService.stream_rois",
        fake_stream_rois,
    )

    result = PlateStreamingService(FakeInspectionService(inventory)).stream_files(
        PlateFileStreamRequest(
            plate_path="/plate_openhcs",
            kind=PlateFileKind.RESULT,
            path_contains="checkpoints_step7_results",
            limit=2,
        )
    )

    assert result.errors == ()
    assert result.streamed_image_paths == ()
    assert result.streamed_roi_paths == (roi_one_path, roi_two_path)
    assert result.skipped_records == ()
    assert tuple(record.relative_path for record in result.resolved_records) == (
        "checkpoints_step7_results/A01_w1_rois.roi.zip",
        "checkpoints_step7_results/A01_w2_rois.roi.zip",
    )
    assert captured == {
        "roi_filenames": (roi_one_path, roi_two_path),
        "component_metadata_by_path": {
            roi_one_path: {"well": "A01", "channel": 1},
            roi_two_path: {"well": "A01", "channel": 2},
        },
    }
    assert result.status_messages == ("streamed query rois",)


def test_plate_streaming_service_uses_context_plate_for_output_roi_stream(monkeypatch):
    captured = {}
    roi_full_path = "/plate_openhcs/images_results/A01_w1_rois.roi.zip"
    inventory = PlateFileInventory(
        plate_path=Path("/plate_openhcs"),
        image_records=(),
        result_records=(
            PlateResultFileRecord(
                relative_path="images_results/A01_w1_rois.roi.zip",
                full_path=roi_full_path,
                file_format=FileFormat.ROI,
                metadata={},
            ),
        ),
    )
    inspection_service = FakeInspectionService(inventory)
    monkeypatch.setattr(
        "openhcs.agent.services.plate_streaming_service."
        "StreamingViewerLifecycle.get_or_create_visualizer",
        lambda **_kwargs: FakeViewer(),
    )

    def fake_stream_rois(self, request):
        captured["stream_plate_path"] = self.source.plate_path
        captured["roi_filenames"] = request.roi_filenames
        captured["component_metadata_by_path"] = request.component_metadata_by_path
        request.status_callback("streamed output rois")

    monkeypatch.setattr(
        "openhcs.agent.services.plate_streaming_service."
        "StreamingService.stream_rois",
        fake_stream_rois,
    )

    result = PlateStreamingService(inspection_service).stream_files(
        PlateFileStreamRequest(
            plate_path="/plate_openhcs",
            context_plate_path="/plate",
            file_paths=("images_results/A01_w1_rois.roi.zip",),
            kind=PlateFileKind.RESULT,
        )
    )

    assert result.errors == ()
    assert result.plate_path == "/plate_openhcs"
    assert inspection_service.open_requests[0].plate_path == "/plate"
    assert inspection_service.resolve_requests == ["/plate_openhcs"]
    assert inspection_service.inventory_contexts[0].plate_path == Path("/plate_openhcs")
    assert inspection_service.inventory_kinds == [None]
    assert captured == {
        "stream_plate_path": Path("/plate_openhcs"),
        "roi_filenames": (roi_full_path,),
        "component_metadata_by_path": {},
    }
    assert result.status_messages == ("streamed output rois",)
