from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import tifffile

from openhcs.constants import Backend
from openhcs.constants.constants import FileFormat
from openhcs.core.config import GlobalPipelineConfig
from openhcs.core.plate_file_inventory import PlateFileInventoryQuery, PlateFileKind
from openhcs.core.plate_image_inventory import (
    PlateFileInventory,
    PlateImageInventory,
    PlateResultFileInventory,
    PlateResultFilePreviewReader,
)
from openhcs.microscopes import create_microscope_handler
from openhcs.microscopes.microscope_interfaces import AnalysisResultDirectory
from openhcs.tests.generators.generate_synthetic_data import SyntheticMicroscopyGenerator
from polystore.base import ensure_storage_registry, storage_registry
from polystore.filemanager import FileManager


class _ImageBrowserOrchestrator:
    def __init__(self, plate_path: Path, microscope_handler, filemanager: FileManager):
        self.plate_path = plate_path
        self.microscope_handler = microscope_handler
        self.filemanager = filemanager
        self._config = GlobalPipelineConfig()

    def get_effective_config(self) -> GlobalPipelineConfig:
        return self._config


def test_image_browser_inventory_uses_declared_virtual_workspace_address(
    tmp_path: Path,
) -> None:
    plate = tmp_path / "plate"
    with redirect_stdout(StringIO()):
        SyntheticMicroscopyGenerator(
            output_dir=str(plate),
            grid_size=(1, 1),
            tile_size=(32, 32),
            wavelengths=1,
            z_stack_levels=2,
            num_cells=4,
            wells=["A01"],
            format="ImageXpress",
            random_seed=7,
        ).generate_dataset()

    ensure_storage_registry()
    filemanager = FileManager(dict(storage_registry))
    handler = create_microscope_handler(
        "imagexpress",
        plate_folder=plate,
        filemanager=filemanager,
    )
    handler.initialize_workspace(plate, filemanager)
    assert handler.get_primary_backend(plate, filemanager) == Backend.VIRTUAL_WORKSPACE.value

    inventory = PlateFileInventory.from_orchestrator(
        _ImageBrowserOrchestrator(plate, handler, filemanager)
    )

    record = inventory.image_records[0]
    assert len(inventory.image_records) == 2
    assert record.virtual_path == "A01_s001_w1_z001_t001.tif"
    assert record.full_virtual_path == str(plate / record.virtual_path)
    assert record.backend == Backend.VIRTUAL_WORKSPACE.value
    assert record.source_path == str(
        plate / "TimePoint_1/ZStep_1/A01_s001_w1.tif"
    )
    np.testing.assert_array_equal(
        filemanager.load(record.full_virtual_path, record.backend),
        tifffile.imread(record.source_path),
    )


def test_plate_result_inventory_exposes_swc_as_text_result(
    tmp_path: Path,
) -> None:
    plate = tmp_path / "plate"
    result_directory = plate / "images_results"
    result_directory.mkdir(parents=True)
    swc_path = result_directory / "A01_neurite_morphology_step1.swc"
    swc_path.write_text(
        "# OpenHCS morphology\n1 1 0 0 0 1 -1\n2 3 1 0 0 1 1\n",
        encoding="utf-8",
    )

    results = PlateResultFileInventory.from_directories(
        plate_path=plate,
        result_directories=(
            AnalysisResultDirectory(
                subdirectory_name="images_results",
                path=result_directory,
            ),
        ),
        parser=None,
    )
    inventory = PlateFileInventory.from_inventories(
        PlateImageInventory(plate_path=plate, records=()),
        results,
    )
    query = inventory.query_files(
        PlateFileInventoryQuery(kinds=(PlateFileKind.RESULT,))
    )

    assert query.total_count == 1
    assert query.records[0].relative_path == "images_results/A01_neurite_morphology_step1.swc"
    assert query.records[0].file_format is FileFormat.TEXT
    preview = PlateResultFilePreviewReader.preview(results.records[0])
    assert preview is not None
    assert preview.text_lines == (
        "# OpenHCS morphology",
        "1 1 0 0 0 1 -1",
        "2 3 1 0 0 1 1",
    )
