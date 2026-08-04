"""Tests for ImageBrowser filter projection without constructing the widget."""

from pathlib import Path
from types import SimpleNamespace
from polystore.disk import DiskStorageBackend
from polystore.filemanager import FileManager
from openhcs.constants.constants import FileFormat
from openhcs.core.config import GlobalPipelineConfig
from openhcs.core.pipeline.path_planner import PathPlannerPathAuthority
from openhcs.core.plate_image_inventory import (
    PlateImageInventory,
    PlateFileInventoryQuery,
    PlateFileKind,
    PlateFileInventory,
    PlateResultFileInventory,
    PlateResultFileRecord,
)
from openhcs.microscopes.microscope_interfaces import AnalysisResultDirectory
from openhcs.pyqt_gui.widgets.image_browser import (
    ImageBrowserFilterController,
    ImageBrowserWidget,
    ImageBrowserItem,
)


class _FolderItem:
    def __init__(self, folder_path: str | None):
        self.folder_path = folder_path

    def data(self, *_):
        return self.folder_path


class _FolderTree:
    def __init__(self, folder_path: str | None = None):
        self.folder_path = folder_path

    def selectedItems(self):
        return [] if self.folder_path is None else [_FolderItem(self.folder_path)]


class _SearchInput:
    def text(self):
        return ""


class _ImageTableBrowser:
    def __init__(self, browser: "_Browser"):
        self.browser = browser

    def search_items(self, _search_term: str):
        return self.browser.file_items


class _Browser:
    def __init__(self):
        self.folder_tree = _FolderTree("PlateA")
        self.selected_wells = {"A01"}
        self.file_items = {
            key: ImageBrowserItem(key=key, metadata=metadata)
            for key, metadata in {
                "PlateA/A01_DAPI.tif": {
                    "filename": "PlateA/A01_DAPI.tif",
                    "well": "A01",
                    "_display_channel": "DAPI",
                },
                "PlateA/A02_DAPI.tif": {
                    "filename": "PlateA/A02_DAPI.tif",
                    "well": "A02",
                    "_display_channel": "DAPI",
                },
                "PlateA_results/A01_DAPI.csv": {
                    "filename": "PlateA_results/A01_DAPI.csv",
                    "well": "A01",
                    "_display_channel": "DAPI",
                },
                "PlateB/A01_DAPI.tif": {
                    "filename": "PlateB/A01_DAPI.tif",
                    "well": "A01",
                    "_display_channel": "DAPI",
                },
                "PlateA/A01_FITC.tif": {
                    "filename": "PlateA/A01_FITC.tif",
                    "well": "A01",
                    "_display_channel": "FITC",
                },
            }.items()
        }
        self.search_input = _SearchInput()
        self.image_table_browser = _ImageTableBrowser(self)
        self.updated = {}

    def _extract_well_id(self, metadata: dict) -> str:
        return metadata["well"]

    def _set_visible_files(self, files_dict, *, rebuild_index: bool):
        assert rebuild_index is False
        self.updated = files_dict


class _ResultOnlyOrchestrator:
    def __init__(self, plate_path: Path, config: GlobalPipelineConfig) -> None:
        self.plate_path = plate_path
        self.microscope_handler = None
        self._config = config

    def get_effective_config(self) -> GlobalPipelineConfig:
        return self._config

    def initialize_microscope_handler(self) -> None:
        return None


class _InventoryParser:
    def parse_filename(self, filename: str):
        if "A01" not in filename:
            return {}
        return {"well": "A01"}


class _InventoryMetadataHandler:
    def __init__(self, handler_result_dir: Path) -> None:
        self._handler_result_dir = handler_result_dir

    def get_image_files(self, _plate_path: Path, *, all_subdirs: bool):
        assert all_subdirs is True
        return ("images/A01_w1.tif",)

    def source_workspace_metadata_document(self, _plate_path: Path):
        return None

    def source_dataset(self, _plate_path: Path):
        return None

    def analysis_result_directories(self, _plate_path: Path):
        return (
            AnalysisResultDirectory(
                subdirectory_name="results",
                path=self._handler_result_dir,
            ),
        )


class _InventoryMicroscopeHandler:
    def __init__(self, handler_result_dir: Path) -> None:
        self.metadata_handler = _InventoryMetadataHandler(handler_result_dir)
        self.parser = _InventoryParser()

    def get_primary_backend(self, _plate_path: Path, _filemanager: FileManager) -> str:
        return "disk"


class _LazyInventoryOrchestrator:
    def __init__(self, plate_path: Path, handler_result_dir: Path) -> None:
        self.plate_path = plate_path
        self.microscope_handler = None
        self._handler = _InventoryMicroscopeHandler(handler_result_dir)
        self._config = GlobalPipelineConfig()
        self.filemanager = FileManager({"disk": DiskStorageBackend()})
        self.initialized = False

    def initialize_microscope_handler(self) -> None:
        self.initialized = True
        self.microscope_handler = self._handler

    def get_effective_config(self) -> GlobalPipelineConfig:
        return self._config


def test_image_browser_filter_controller_applies_folder_and_well_filters():
    browser = _Browser()
    controller = ImageBrowserFilterController(browser)

    controller.apply_combined_filters()

    assert set(browser.updated) == {
        "PlateA/A01_DAPI.tif",
        "PlateA_results/A01_DAPI.csv",
        "PlateA/A01_FITC.tif",
    }


def test_image_browser_background_cycle_uses_synchronous_settled_stream_calls():
    calls: list[tuple[str, tuple[str, ...]]] = []

    class StreamingService:
        def stream_images(self, request) -> None:
            calls.append(("images", request.filenames))

        def stream_rois(self, request) -> None:
            calls.append(("rois", request.roi_filenames))

        def stream_images_async(self, _request) -> None:
            raise AssertionError("nested image streaming thread used")

        def stream_rois_async(self, _request) -> None:
            raise AssertionError("nested ROI streaming thread used")

    browser = SimpleNamespace(
        _prepare_streaming=lambda _config_key: (
            object(),
            "disk",
            SimpleNamespace(display_name="Napari"),
        ),
        streaming_service=StreamingService(),
        _status_update_signal=SimpleNamespace(emit=lambda _message: None),
        _show_streaming_error=lambda _viewer, _error: None,
    )

    ImageBrowserWidget._stream_rois_to_viewer(
        browser,
        ["A01_w1_rois.roi.zip"],
        "napari_streaming_config",
    )
    ImageBrowserWidget._stream_images_to_viewer(
        browser,
        ["A01_w1.tif", "A01_w2.tif"],
        "napari_streaming_config",
    )

    assert calls == [
        ("rois", ("A01_w1_rois.roi.zip",)),
        ("images", ("A01_w1.tif", "A01_w2.tif")),
    ]


def test_image_browser_projects_shared_result_inventory_records(tmp_path: Path):
    roi_path = tmp_path / "plate" / "results" / "A01.roi.zip"
    text_path = tmp_path / "plate" / "results" / "summary.txt"
    roi_path.parent.mkdir(parents=True)
    roi_path.write_bytes(b"roi")
    text_path.write_text("summary", encoding="utf-8")
    inventory = PlateResultFileInventory(
        plate_path=tmp_path / "plate",
        scanned_file_count=2,
        records=(
            PlateResultFileRecord(
                relative_path="results/A01.roi.zip",
                full_path=str(roi_path),
                file_format=FileFormat.ROI,
                metadata={
                    "filename": "results/A01.roi.zip",
                    "type": "ROI",
                    "well": "A01",
                },
            ),
            PlateResultFileRecord(
                relative_path="results/summary.txt",
                full_path=str(text_path),
                file_format=FileFormat.TEXT,
                metadata={
                    "filename": "results/summary.txt",
                    "type": "TEXT",
                    "well": "A01",
                },
            ),
        ),
    )

    result_items = ImageBrowserWidget._result_items_from_inventory(inventory)

    item = result_items["results/A01.roi.zip"]
    assert item.result_file_type is FileFormat.ROI
    assert item.full_path == roi_path
    assert item.metadata["type"] == "ROI"
    text_item = result_items["results/summary.txt"]
    assert text_item.result_file_type is FileFormat.TEXT
    assert text_item.full_path == text_path
    assert text_item.metadata["type"] == "TEXT"


def test_plate_file_inventory_reads_path_planned_result_only_output(
    tmp_path: Path,
) -> None:
    plate_root = tmp_path / "source_plate_openhcs"
    config = GlobalPipelineConfig()
    result_dir = PathPlannerPathAuthority.analysis_results_dir_for(
        plate_root / config.path_planning_config.sub_dir,
    )
    result_dir.mkdir(parents=True)
    detail_csv = result_dir / "A01_w1_cell_counts_step0_details.csv"
    summary_text = result_dir / "A01_w1_segmentation_masks_step0_summary.txt"
    detail_csv.write_text("slice_index,cell_count\n0,11\n", encoding="utf-8")
    summary_text.write_text("Segmentation ROIs: 11 cells\n", encoding="utf-8")
    orchestrator = _ResultOnlyOrchestrator(plate_root, config)

    inventory = PlateFileInventory.from_orchestrator(orchestrator)

    assert inventory.image_records == ()
    assert inventory.scanned_result_file_count == 2
    assert [record.relative_path for record in inventory.result_records] == [
        "images_results/A01_w1_cell_counts_step0_details.csv",
        "images_results/A01_w1_segmentation_masks_step0_summary.txt",
    ]


def test_plate_file_inventory_from_lazy_orchestrator_uses_public_inventory_path(
    tmp_path: Path,
) -> None:
    plate_root = tmp_path / "plate"
    image_path = plate_root / "images" / "A01_w1.tif"
    handler_result_dir = plate_root / "results"
    image_path.parent.mkdir(parents=True)
    handler_result_dir.mkdir(parents=True)
    image_path.write_bytes(b"image")
    (handler_result_dir / "A01_summary.txt").write_text(
        "handler result\n",
        encoding="utf-8",
    )
    orchestrator = _LazyInventoryOrchestrator(plate_root, handler_result_dir)

    image_inventory = PlateImageInventory.from_orchestrator(orchestrator)
    file_inventory = PlateFileInventory.from_orchestrator(orchestrator)

    assert orchestrator.initialized is True
    assert [record.virtual_path for record in image_inventory.records] == [
        "images/A01_w1.tif"
    ]
    assert [record.virtual_path for record in file_inventory.image_records] == [
        "images/A01_w1.tif"
    ]
    assert [record.relative_path for record in file_inventory.result_records] == [
        "results/A01_summary.txt"
    ]


def test_plate_file_inventory_from_handler_matches_browser_file_shape(
    tmp_path: Path,
) -> None:
    plate_root = tmp_path / "plate"
    image_path = plate_root / "images" / "A01_w1.tif"
    handler_result_dir = plate_root / "results"
    config = GlobalPipelineConfig()
    output_plate_root = PathPlannerPathAuthority.build_output_plate_root(
        plate_root,
        config.path_planning_config,
    )
    output_result_dir = PathPlannerPathAuthority.analysis_results_dir_for(
        output_plate_root / config.path_planning_config.sub_dir,
    )
    image_path.parent.mkdir(parents=True)
    handler_result_dir.mkdir(parents=True)
    output_result_dir.mkdir(parents=True)
    image_path.write_bytes(b"image")
    (handler_result_dir / "A01_handler_summary.txt").write_text(
        "handler result\n",
        encoding="utf-8",
    )
    (output_result_dir / "A01_output_counts.csv").write_text(
        "well,count\nA01,3\n",
        encoding="utf-8",
    )

    inventory = PlateFileInventory.from_handler(
        plate_path=plate_root,
        metadata_handler=_InventoryMetadataHandler(handler_result_dir),
        parser=_InventoryParser(),
        filemanager=FileManager({"disk": DiskStorageBackend()}),
        backend="disk",
        path_config=config.path_planning_config,
        all_subdirs=True,
    )

    assert [record.virtual_path for record in inventory.image_records] == [
        "images/A01_w1.tif"
    ]
    assert inventory.image_records[0].metadata["well"] == "A01"
    assert [record.relative_path for record in inventory.result_records] == [
        "images_results/A01_output_counts.csv",
        "results/A01_handler_summary.txt",
    ]
    assert inventory.result_records[0].full_path == str(
        output_result_dir / "A01_output_counts.csv"
    )
    assert inventory.scanned_result_file_count == 2
    assert inventory.result_inventory.records == inventory.result_records


def test_plate_file_inventory_query_returns_unified_browser_records(
    tmp_path: Path,
) -> None:
    plate_root = tmp_path / "plate"
    image_path = plate_root / "images" / "A01_w1.tif"
    handler_result_dir = plate_root / "results"
    image_path.parent.mkdir(parents=True)
    handler_result_dir.mkdir(parents=True)
    image_path.write_bytes(b"image")
    (handler_result_dir / "A01_summary.txt").write_text(
        "handler result\n",
        encoding="utf-8",
    )

    inventory = PlateFileInventory.from_handler(
        plate_path=plate_root,
        metadata_handler=_InventoryMetadataHandler(handler_result_dir),
        parser=_InventoryParser(),
        filemanager=FileManager({"disk": DiskStorageBackend()}),
        backend="disk",
        all_subdirs=True,
    )

    query_result = inventory.query_files(
        PlateFileInventoryQuery(kinds=(PlateFileKind.IMAGE,), well="A01")
    )
    image_items, result_items = ImageBrowserWidget._items_from_file_records(
        inventory.file_records()
    )

    assert query_result.total_count == 1
    assert query_result.records[0].key == "images/A01_w1.tif"
    assert image_items["images/A01_w1.tif"].metadata["well"] == "A01"
    assert result_items["results/A01_summary.txt"].result_file_type is FileFormat.TEXT
