from pathlib import Path

import numpy as np

from openhcs.constants.constants import Backend
from openhcs.core.steps.function_io import bulk_preload_step_images
from openhcs.microscopes.bioformats import BioFormatsHandler
from tests.unit.bioformats_fixture import (
    bioformats_filemanager,
    write_bioformats_manifest_fixture,
)


def test_bioformats_handler_preloads_planes_through_runtime_path(tmp_path: Path) -> None:
    stack = write_bioformats_manifest_fixture(tmp_path)
    filemanager = bioformats_filemanager()
    handler = BioFormatsHandler(filemanager)
    handler.initialize_workspace(tmp_path, filemanager)

    bulk_preload_step_images(
        step_input_dir=tmp_path,
        axis_id="A01",
        read_backend=Backend.VIRTUAL_WORKSPACE.value,
        filemanager=filemanager,
        microscope_handler=handler,
        patterns_to_preload=("A01_s001_w1_z001_t001.tif",),
    )

    loaded = filemanager.load_batch(
        [str(tmp_path / "A01_s001_w1_z001_t001.tif")],
        Backend.MEMORY.value,
    )
    np.testing.assert_array_equal(loaded[0].data, stack[0, 0, 0])
    assert loaded[0].metadata.source_path.endswith("A01_s001_w1_z001_t001.tif")
