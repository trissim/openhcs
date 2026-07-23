from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from openhcs.tests.generators.generate_synthetic_data import SyntheticMicroscopyGenerator


class BioFormatsCompatibleImageXpressPlateFactory:
    """Test authority for generating a small Bio-Formats-compatible ImageXpress plate."""

    def create(self, plate: Path) -> None:
        with redirect_stdout(StringIO()):
            SyntheticMicroscopyGenerator(
                output_dir=plate,
                grid_size=(1, 2),
                tile_size=(32, 32),
                overlap_percent=10,
                stage_error_px=1,
                wavelengths=2,
                z_stack_levels=2,
                num_cells=4,
                wells=["A01"],
                format="ImageXpress",
                random_seed=7,
                imagexpress_bioformats_compatible=True,
            ).generate_dataset()


IMAGE_XPRESS_PLATE_FACTORY = BioFormatsCompatibleImageXpressPlateFactory()
