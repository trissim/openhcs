#!/usr/bin/env python3
"""
Generate synthetic microscopy images for OpenHCS demonstrations and tests.

This module generates synthetic microscopy images with the following features:
- Multiple wavelengths (channels)
- Z-stack support with varying focus levels
- Cell-like structures (circular particles with varying eccentricity)
- Proper tiling with configurable overlap
- Realistic stage positioning errors
- HTD file generation for metadata
- Automatic image size calculation based on grid and tile parameters

Usage:
    from openhcs.demo.synthetic_data import SyntheticMicroscopyGenerator

    generator = SyntheticMicroscopyGenerator(
        output_dir="path/to/output",
        grid_size=(3, 3),
        wavelengths=2,
        z_stack_levels=3,
        auto_image_size=True
    )
    generator.generate_dataset()
"""

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import tifffile
from skimage import draw, filters

from openhcs.microscopes.imagexpress import ImageXpressFilenameParser
from openhcs.microscopes.opera_phenix import OperaPhenixFilenameParser
from polystore.constants import Backend
from polystore.virtual_workspace import SourcePixelRef


class SyntheticMicroscopyGenerator:
    """Generate synthetic microscopy images for testing."""

    PROMINENT_WAVELENGTH_CELL_INTENSITIES = {
        1: 25000,
        2: 10000,
    }
    ADDITIONAL_WAVELENGTH_BASE_INTENSITY = 5000
    ADDITIONAL_WAVELENGTH_INTENSITY_STEP = 1000

    def __init__(self,
                 output_dir: str,
                 grid_size: Tuple[int, int] = (3, 3),
                 tile_size: Tuple[int, int] = (256, 256),  # Updated to match common test defaults
                 overlap_percent: int = 10,
                 stage_error_px: int = 2,
                 wavelengths: int = 2,
                 z_stack_levels: int = 3,  # Updated to match common test defaults (3D data)
                 z_step_size: float = 0.1,  # Reduced by 10x for more subtle blur effect
                 num_cells: int = 50,
                 cell_size_range: Tuple[int, int] = (3, 6),  # Updated to match common test defaults
                 cell_eccentricity_range: Tuple[float, float] = (0.1, 0.5),
                 cell_intensity_range: Tuple[int, int] = (5000, 20000),
                 background_intensity: int = 500,
                 noise_level: int = 100,
                 wavelength_params: Optional[Dict[int, Dict[str, Any]]] = None,
                 shared_cell_fraction: float = 0.95,  # Fraction of cells shared between wavelengths
                 wavelength_intensities: Optional[Dict[int, Tuple[int, int]]] = None,  # Fixed intensities for each wavelength
                 wavelength_backgrounds: Optional[Dict[int, int]] = None,  # Background intensities for each wavelength
                 wells: List[str] = None,  # Updated to match common test defaults (4 wells)
                 format: str = 'ImageXpress',  # Format of the filenames ('ImageXpress' or 'OperaPhenix')
                 openhcs_format: bool = False,  # If True, generate OpenHCS native format with metadata
                 random_seed: Optional[int] = None,
                 include_all_components: bool = False,  # Include all filename components (for OpenHCS)
                 imagexpress_bioformats_compatible: bool = False,  # Emit MetaXpress names Bio-Formats can discover
                 skip_files: Optional[List[str]] = None):  # List of filenames to skip (for testing missing image handling)
        """
        Initialize the synthetic microscopy generator.

        Args:
            output_dir: Directory to save generated images
            grid_size: Tuple of (rows, cols) for the grid of tiles
            tile_size: Size of each tile
            overlap_percent: Percentage of overlap between tiles
            stage_error_px: Random error in stage positioning (pixels)
            wavelengths: Number of wavelength channels to generate
            z_stack_levels: Number of Z-stack levels to generate
            z_step_size: Spacing between Z-steps in microns
            num_cells: Number of cells to generate
            cell_size_range: Range of cell sizes (min, max)
            cell_eccentricity_range: Range of cell eccentricity (min, max)
            cell_intensity_range: Range of cell intensity (min, max)
            background_intensity: Background intensity level
            noise_level: Amount of noise to add
            wavelength_params: Optional dictionary of parameters for each wavelength
                Example: {
                    1: {
                        'num_cells': 100,
                        'cell_size_range': (10, 30),
                        'cell_eccentricity_range': (0.1, 0.5),
                        'cell_intensity_range': (5000, 20000),
                        'background_intensity': 500
                    },
                    2: {
                        'num_cells': 50,
                        'cell_size_range': (5, 15),
                        'cell_eccentricity_range': (0.3, 0.8),
                        'cell_intensity_range': (3000, 12000),
                        'background_intensity': 300
                    }
                }
            shared_cell_fraction: Fraction of cells shared between wavelengths (0.0-1.0)
                0.0 means all cells are unique to each wavelength
                1.0 means all cells are shared between wavelengths
                Default is 0.95 (95% shared)
            wavelength_intensities: Dictionary mapping wavelength indices to fixed intensities
                Example: {1: 20000, 2: 10000}
            wavelength_backgrounds: Dictionary mapping wavelength indices to background intensities
                Example: {1: 800, 2: 400}
            wells: List of well IDs to generate (e.g., ['A01', 'A02'])
            format: Format of the filenames ('ImageXpress' or 'OperaPhenix')
            openhcs_format: If True, generate OpenHCS native format with openhcs_metadata.json
            random_seed: Random seed for reproducibility
            include_all_components: If True, include all filename components (timepoint, z-index) even for flat plates
            imagexpress_bioformats_compatible: If True, emit ImageXpress TIFF names with the plate-name
                prefix and unpadded site/channel axes expected by Bio-Formats' MetaXpress reader.
            skip_files: List of filenames to skip during generation (for testing missing image handling)
        """
        self.output_dir = Path(output_dir)
        self.skip_files = set(skip_files) if skip_files else set()
        self.grid_size = grid_size
        self.tile_size = tile_size
        self.overlap_percent = overlap_percent
        self.stage_error_px = stage_error_px
        self.include_all_components = include_all_components
        self.imagexpress_bioformats_compatible = imagexpress_bioformats_compatible
        self.openhcs_format = openhcs_format

        # Create parser instances for filename construction
        self.imagexpress_parser = ImageXpressFilenameParser()
        self.operaphenix_parser = OperaPhenixFilenameParser()

        # Always auto-calculate image size from grid and tile parameters
        self.image_size = self._calculate_image_size(grid_size, tile_size, overlap_percent, stage_error_px)
        self.image_shape = (self.image_size[1], self.image_size[0])
        print(f"Auto-calculated image size: {self.image_size[0]}x{self.image_size[1]}")
        self.wavelengths = wavelengths
        self.z_stack_levels = z_stack_levels
        self.z_step_size = z_step_size
        self.num_cells = num_cells
        self.cell_size_range = cell_size_range
        self.cell_eccentricity_range = cell_eccentricity_range
        self.cell_intensity_range = cell_intensity_range
        self.background_intensity = background_intensity
        self.noise_level = noise_level
        self.wavelength_params = wavelength_params or {}
        self.shared_cell_fraction = shared_cell_fraction

        # Set default wavelength intensities if not provided
        if wavelength_intensities is None:
            self.wavelength_intensities = {1: 20000, 2: 10000}
            # Add defaults for additional wavelengths if needed
            for w in range(3, wavelengths + 1):
                self.wavelength_intensities[w] = 15000
        else:
            self.wavelength_intensities = wavelength_intensities

        # Set default wavelength backgrounds if not provided
        if wavelength_backgrounds is None:
            self.wavelength_backgrounds = {1: 800, 2: 400}
            # Add defaults for additional wavelengths if needed
            for w in range(3, wavelengths + 1):
                self.wavelength_backgrounds[w] = 600
        else:
            self.wavelength_backgrounds = wavelength_backgrounds

        # Store the wells to generate (use default if None)
        self.wells = wells if wells is not None else ['A01', 'D02', 'B03', 'B06']

        # Store the format
        self.format = format

        # Store the base random seed
        self.base_random_seed = random_seed

        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)

        # Create output directory structure
        # For ImageXpress, create TimePoint_1 directory
        # For Opera Phenix, create Images directory
        if self.format == 'ImageXpress':
            self.timepoint_dir = self.output_dir / "TimePoint_1"
            self.timepoint_dir.mkdir(parents=True, exist_ok=True)
        else:  # OperaPhenix
            # Create the Images directory for Opera Phenix
            self.images_dir = self.output_dir / "Images"
            self.images_dir.mkdir(parents=True, exist_ok=True)
            self.timepoint_dir = self.images_dir  # Store images in the Images directory

        # Calculate effective step size with overlap
        self.step_x = int(tile_size[0] * (1 - overlap_percent / 100))
        self.step_y = int(tile_size[1] * (1 - overlap_percent / 100))

        # First, generate a common set of base cells for all wavelengths
        # This ensures consistent patterns across wavelengths for better registration
        base_cells = []
        max_num_cells = max([
            self.wavelength_params.get(w+1, {}).get('num_cells', self.num_cells)
            for w in range(wavelengths)
        ])

        # Generate shared cell positions and common attributes
        for i in range(max_num_cells):
            # Generate cell in a way that ensures good features in overlap regions
            # Bias cell positions to increase density in overlap regions for better registration

            # Calculate overlap regions (where tiles will overlap)
            overlap_x = int(tile_size[0] * (overlap_percent / 100))
            overlap_y = int(tile_size[1] * (overlap_percent / 100))

            # Very strongly favor overlap regions with 80% probability
            # to ensure very high density of features in the 10% overlap region for reliable registration
            if np.random.random() < 0.8:
                # Position in an overlap region between tiles
                col = np.random.randint(0, grid_size[1])
                row = np.random.randint(0, grid_size[0])

                # Calculate base tile position
                base_x = col * self.step_x
                base_y = row * self.step_y

                # Position cells in right/bottom overlapping regions
                if np.random.random() < 0.5:
                    # Right overlap region
                    x = base_x + tile_size[0] - overlap_x + np.random.randint(0, overlap_x)
                    y = base_y + np.random.randint(0, tile_size[1])
                else:
                    # Bottom overlap region
                    x = base_x + np.random.randint(0, tile_size[0])
                    y = base_y + tile_size[1] - overlap_y + np.random.randint(0, overlap_y)

                # Ensure we're within image bounds
                x = min(x, self.image_size[0] - 1)
                y = min(y, self.image_size[1] - 1)
            else:
                # Random position anywhere in the image
                x = np.random.randint(0, self.image_size[0])
                y = np.random.randint(0, self.image_size[1])

            # Common cell attributes
            size = np.random.uniform(*self.cell_size_range)
            eccentricity = np.random.uniform(*self.cell_eccentricity_range)
            rotation = np.random.uniform(0, 2*np.pi)

            base_cells.append({
                'x': x,
                'y': y,
                'size': size,
                'eccentricity': eccentricity,
                'rotation': rotation
            })

        # We'll generate cell parameters for each well and wavelength on demand
        # This is just a placeholder initialization
        self.cell_params = {}

        # Store wavelength-specific parameters for later use
        self.wavelength_specific_params = []
        for w in range(wavelengths):
            wavelength_idx = w + 1  # 1-based wavelength index

            # Get wavelength-specific parameters or use defaults
            w_params = self.wavelength_params.get(wavelength_idx, {})
            w_num_cells = w_params.get('num_cells', self.num_cells)
            w_cell_size_range = w_params.get('cell_size_range', self.cell_size_range)
            w_cell_intensity_range = w_params.get('cell_intensity_range', self.cell_intensity_range)

            self.wavelength_specific_params.append({
                'wavelength_idx': wavelength_idx,
                'num_cells': w_num_cells,
                'cell_size_range': w_cell_size_range,
                'cell_intensity_range': w_cell_intensity_range
            })

            # We'll generate cells on demand in generate_cell_image

    def generate_cell_image(self, wavelength, z_level, well=None):
        """
        Generate a full image with cells for a specific wavelength and Z level.

        Args:
            wavelength: Wavelength channel index
            z_level: Z-stack level index
            well: Well identifier (e.g., 'A01')

        Returns:
            Full image with cells
        """
        # Generate a unique key for this well and wavelength
        key = f"{well}_{wavelength}" if well else f"default_{wavelength}"

        # Get wavelength-specific parameters
        wavelength_idx = wavelength + 1  # Convert to 1-based index for params lookup
        w_params = self.wavelength_params.get(wavelength_idx, {})

        # Generate cells for this well and wavelength if not already generated
        if key not in self.cell_params:
            # Get parameters for cell generation
            w_num_cells = w_params.get('num_cells', self.num_cells)
            w_cell_size_range = w_params.get('cell_size_range', self.cell_size_range)
            w_cell_intensity_range = w_params.get('cell_intensity_range', self.cell_intensity_range)

            # Generate cells for this wavelength
            cells = []
            for i in range(w_num_cells):
                # Generate random position for this wavelength
                x = np.random.randint(0, self.image_size[0])
                y = np.random.randint(0, self.image_size[1])

                # Generate random cell properties
                size = np.random.uniform(w_cell_size_range[0], w_cell_size_range[1])
                eccentricity = np.random.uniform(self.cell_eccentricity_range[0], self.cell_eccentricity_range[1])
                rotation = np.random.uniform(0, 2*np.pi)

                intensity = self.cell_intensity_for_wavelength(wavelength_idx)

                cells.append({
                    'x': x,
                    'y': y,
                    'size': size,
                    'eccentricity': eccentricity,
                    'rotation': rotation,
                    'intensity': intensity
                })

            # Store cells for this well and wavelength
            self.cell_params[key] = cells

        # Get cells for this well and wavelength
        cells = self.cell_params[key]

        # Get cell parameters for this well and wavelength
        cells = self.cell_params[key]

        # Calculate Z-focus factor (1.0 at center Z, decreasing toward edges)
        if self.z_stack_levels > 1:
            z_center = (self.z_stack_levels - 1) / 2
            z_distance = abs(z_level - z_center)
            z_factor = 1.0 - (z_distance / z_center) if z_center > 0 else 1.0
        else:
            z_factor = 1.0

        # STEP 1: Create uniform background
        # Get background intensity from wavelength_backgrounds or use default
        w_background = self.wavelength_backgrounds.get(wavelength_idx, self.background_intensity)
        image = np.ones(self.image_shape, dtype=np.uint16) * w_background

        # STEP 2: Create cells on black background for blur processing
        cell_image = np.zeros(self.image_shape, dtype=np.uint16)

        # Draw each cell on black background
        for cell in cells:
            # Adjust intensity based on Z level (cells are brightest at focus)
            # Keep cells visible even when out of focus (minimum 30% intensity)
            intensity_factor = 0.3 + 0.7 * z_factor  # Range from 0.3 to 1.0
            intensity = cell['intensity'] * intensity_factor

            # Calculate ellipse parameters
            a = cell['size']
            b = cell['size'] * (1 - cell['eccentricity'])

            # Generate ellipse coordinates
            rr, cc = draw.ellipse(
                cell['y'], cell['x'],
                b, a,
                rotation=cell['rotation'],
                shape=self.image_shape
            )

            # Add cell to black background
            cell_image[rr, cc] = intensity

        # STEP 3: Apply blur to cells on black background (optical defocus)
        if self.z_stack_levels > 1:
            # More blur for Z levels further from center
            # Use a fixed scaling factor that works well regardless of z_step_size
            # Base blur sigma ranges from 0.5 (in focus) to 2.0 (out of focus)
            blur_sigma = 0.5 + 1.5 * (1.0 - z_factor)
            print(f"  Z-level {z_level}: blur_sigma={blur_sigma:.2f} (z_factor={z_factor:.2f}, z_step_size={self.z_step_size})")
            if blur_sigma > 0.1:  # Only apply blur if sigma is meaningful
                # Convert to float for processing, then back to preserve range properly
                cell_image_float = cell_image.astype(np.float64)
                cell_image_float = filters.gaussian(cell_image_float, sigma=blur_sigma)
                cell_image = cell_image_float.astype(np.uint16)

        # STEP 4: Add blurred cells to uniform background
        # This preserves uniform background while adding blurred cell signal
        image = image + cell_image
        image = np.clip(image, 0, 65535).astype(np.uint16)

        # Use wavelength-specific noise level if provided (add noise AFTER blur)
        w_noise_level = w_params.get('noise_level', self.noise_level)
        if w_noise_level > 0:
            noise = np.random.normal(0, w_noise_level, self.image_shape)
            image = image.astype(np.float64) + noise
            image = np.clip(image, 0, 65535).astype(np.uint16)
        else:
            # Ensure valid pixel values even without noise
            image = np.clip(image, 0, 65535).astype(np.uint16)

        return image

    def cell_intensity_for_wavelength(self, wavelength_idx: int) -> int:
        """Return synthetic cell contrast for a one-based wavelength index."""
        configured_intensity = self.PROMINENT_WAVELENGTH_CELL_INTENSITIES.get(
            wavelength_idx
        )
        if configured_intensity is not None:
            return configured_intensity
        return (
            self.ADDITIONAL_WAVELENGTH_BASE_INTENSITY
            + wavelength_idx * self.ADDITIONAL_WAVELENGTH_INTENSITY_STEP
        )

    # We've replaced the generate_tiles method with position pre-generation in generate_dataset

    def generate_htd_file(self):
        """Generate HTD file with metadata in the format expected by openhcs."""
        # Derive plate name from output directory name
        plate_name = self.output_dir.name

        if self.format == 'OperaPhenix':
            # Generate Index.xml for Opera Phenix
            return self.generate_opera_phenix_index_xml(plate_name)
        else:
            # Generate HTD file for ImageXpress
            htd_filename = f"{plate_name}.HTD"

            # Generate the main HTD file in the plate dir
            htd_path = self.output_dir / htd_filename

            # Basic HTD file content matching the format of real HTD files
            htd_content = f""""HTSInfoFile", Version 1.0
"Description", "Synthetic microscopy data for testing"
"PlateType", 6
"TimePoints", 1
"ZSeries", {"TRUE" if self.z_stack_levels > 1 else "FALSE"}
"ZSteps", {self.z_stack_levels}
"ZProjection", FALSE
"XWells", 4
"YWells", 3"""

            # Add wells selection (only the wells we're using are TRUE)
            for y in range(3):  # 3 rows (A, B, C)
                row_wells = []
                for x in range(4):  # 4 columns (1, 2, 3, 4)
                    well = f"{chr(65+y)}{x+1:02d}"  # A01, A02, etc.
                    row_wells.append("TRUE" if well in self.wells else "FALSE")
                htd_content += f"\n\"WellsSelection{y+1}\", {', '.join(row_wells)}"

            # Add sites information
            htd_content += "\n\"Sites\", TRUE"
            htd_content += f"\n\"XSites\", {self.grid_size[1]}"
            htd_content += f"\n\"YSites\", {self.grid_size[0]}"

            # Add site selection rows (all set to FALSE except the ones we're using)
            for y in range(self.grid_size[0]):
                row = []
                for x in range(self.grid_size[1]):
                    row.append("TRUE")  # All sites are used in our synthetic data
                htd_content += f"\n\"SiteSelection{y+1}\", {', '.join(row)}"

            # Add wavelength information
            htd_content += "\n\"Waves\", TRUE"
            htd_content += f"\n\"NWavelengths\", {self.wavelengths}"
            htd_content += "\n\"PixelSizeUM\", 0.65"

            # Add wavelength names and collection flags
            for w in range(self.wavelengths):
                htd_content += f"\n\"WaveName{w+1}\", \"W{w+1}\""
                htd_content += f"\n\"WaveCollect{w+1}\", 1"

            # Add unique identifier and end file marker
            htd_content += f"\n\"UniquePlateIdentifier\", \"{plate_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}\""
            htd_content += "\n\"EndFile\""

            # Write HTD file in plate root directory
            with open(htd_path, 'w') as f:
                f.write(htd_content)

            # For ImageXpress, also create a copy in the TimePoint directory
            timepoint_htd_path = self.timepoint_dir / htd_filename
            with open(timepoint_htd_path, 'w') as f:
                f.write(htd_content)

            return htd_path

    @staticmethod
    def generate_opera_phenix_field_pattern(grid_rows: int, grid_cols: int) -> Dict[int, Tuple[int, int]]:
        """
        Generate Opera Phenix center-first snake pattern for field IDs.

        Real Opera Phenix microscopes acquire fields starting from the center,
        then follow a snake/boustrophedon pattern skipping the center position.

        For 3x3 grid:
        2  3  4
        6  1  5
        7  8  9

        Args:
            grid_rows: Number of rows in the grid
            grid_cols: Number of columns in the grid

        Returns:
            dict: Mapping of field_id -> (row_idx, col_idx) in grid coordinates
        """
        # Calculate center position
        center_row = grid_rows // 2
        center_col = grid_cols // 2

        # Field 1 is always at center
        field_pattern = {1: (center_row, center_col)}

        field_id = 2

        # Snake pattern: even rows go left-to-right, odd rows go right-to-left
        for row in range(grid_rows):
            if row % 2 == 0:
                # Even row: left to right
                for col in range(grid_cols):
                    if (row, col) != (center_row, center_col):  # Skip center
                        field_pattern[field_id] = (row, col)
                        field_id += 1
            else:
                # Odd row: right to left
                for col in range(grid_cols - 1, -1, -1):
                    if (row, col) != (center_row, center_col):  # Skip center
                        field_pattern[field_id] = (row, col)
                        field_id += 1

        return field_pattern

    def generate_opera_phenix_index_xml(self, plate_name):
        """Generate Index.xml file for Opera Phenix format."""
        # Create the Index.xml file in the Images directory
        index_xml_path = self.images_dir / "Index.xml"

        # Get current date and time for the measurement ID
        current_time = datetime.now().strftime("%Y-%m-%dT%H_%M_%S")
        measurement_id = f"{plate_name}__{current_time}-Measurement 1"

        # Create a unique ID for the measurement
        import uuid
        unique_id = str(uuid.uuid4())

        # Extract row and column numbers from wells
        # Convert well names like 'A01' to row and column indices
        well_indices = []
        for well in self.wells:
            row = ord(well[0]) - ord('A') + 1  # A -> 1, B -> 2, etc.
            col = int(well[1:3])
            well_indices.append((row, col))

        # Calculate pixel size in meters (for ImageResolutionX/Y)
        # Default is 0.65 µm, but we'll use a more realistic value for Opera Phenix
        pixel_size_meters = 1.1867525298988041E-06  # ~1.19 µm

        # Calculate Z-step size in meters
        z_step_size_meters = self.z_step_size * 1e-6  # Convert from µm to m

        # Start building the XML content
        xml_content = f"""<?xml version="1.0" encoding="utf-8"?>
<EvaluationInputData xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" Version="1" xmlns="http://www.perkinelmer.com/PEHH/HarmonyV6">
  <User>Synthetic</User>
  <InstrumentType>Phenix</InstrumentType>
  <Plates>
    <Plate>
      <PlateID>{plate_name}</PlateID>
      <MeasurementID>{unique_id}</MeasurementID>
      <MeasurementStartTime>{datetime.now().isoformat()}-04:00</MeasurementStartTime>
      <n>{plate_name}</n>
      <PlateTypeName>96well</PlateTypeName>
      <PlateRows>8</PlateRows>
      <PlateColumns>12</PlateColumns>"""

        # Add wells
        for row, col in well_indices:
            xml_content += f"\n      <Well id=\"{row:02d}{col:02d}\" />"

        xml_content += "\n    </Plate>\n  </Plates>\n  <Wells>"

        # Add well details
        for row, col in well_indices:
            xml_content += f"\n    <Well>\n      <id>{row:02d}{col:02d}</id>\n      <Row>{row}</Row>\n      <Col>{col}</Col>"

            # Add images for each site and channel
            for site in range(1, self.grid_size[0] * self.grid_size[1] + 1):
                for channel in range(1, self.wavelengths + 1):
                    for z in range(1, self.z_stack_levels + 1):
                        xml_content += f"\n      <Image id=\"{row:02d}{col:02d}K1F{site}P{z}R{channel}\" />"

            xml_content += "\n    </Well>"

        # Add image details section
        xml_content += """
  </Wells>
  <Images>
    <Map>
      <Entry ChannelID="1">
        <ChannelName>HOECHST 33342</ChannelName>
        <ImageType>Signal</ImageType>
        <AcquisitionType>NonConfocal</AcquisitionType>
        <IlluminationType>Epifluorescence</IlluminationType>
        <ChannelType>Fluorescence</ChannelType>
        <ImageResolutionX Unit="m">{}</ImageResolutionX>
        <ImageResolutionY Unit="m">{}</ImageResolutionY>
        <ImageSizeX>{}</ImageSizeX>
        <ImageSizeY>{}</ImageSizeY>
        <BinningX>2</BinningX>
        <BinningY>2</BinningY>
        <MaxIntensity>65536</MaxIntensity>
        <CameraType>AndorZylaCam</CameraType>
        <MainExcitationWavelength Unit="nm">375</MainExcitationWavelength>
        <MainEmissionWavelength Unit="nm">456</MainEmissionWavelength>
        <ObjectiveMagnification Unit="">10</ObjectiveMagnification>
        <ObjectiveNA Unit="">0.3</ObjectiveNA>
        <ExposureTime Unit="s">0.03</ExposureTime>
        <OrientationMatrix>[[1.009457,0,0,34.3],[0,-1.009457,0,-15.1],[0,0,1.33,-6.014]]</OrientationMatrix>
        <CropArea>[[0,0],[2160,2160],[2160,2160]]</CropArea>
      </Entry>""".format(pixel_size_meters, pixel_size_meters, self.image_size[0], self.image_size[1])

        # Add entries for each channel
        channel_names = ["Calcein", "Alexa 647", "FITC", "TRITC", "Cy5"]
        excitation_wavelengths = [488, 647, 488, 561, 647]
        emission_wavelengths = [525, 665, 525, 590, 665]
        exposure_times = [0.05, 0.1, 0.05, 0.08, 0.1]

        for channel in range(2, self.wavelengths + 1):
            channel_idx = min(channel - 2, len(channel_names) - 1)  # Ensure we don't go out of bounds
            xml_content += f"""
      <Entry ChannelID="{channel}">
        <ChannelName>{channel_names[channel_idx]}</ChannelName>
        <ImageType>Signal</ImageType>
        <AcquisitionType>NonConfocal</AcquisitionType>
        <IlluminationType>Epifluorescence</IlluminationType>
        <ChannelType>Fluorescence</ChannelType>
        <ImageResolutionX Unit="m">{pixel_size_meters}</ImageResolutionX>
        <ImageResolutionY Unit="m">{pixel_size_meters}</ImageResolutionY>
        <ImageSizeX>{self.image_size[0]}</ImageSizeX>
        <ImageSizeY>{self.image_size[1]}</ImageSizeY>
        <BinningX>2</BinningX>
        <BinningY>2</BinningY>
        <MaxIntensity>65536</MaxIntensity>
        <CameraType>AndorZylaCam</CameraType>
        <MainExcitationWavelength Unit="nm">{excitation_wavelengths[channel_idx]}</MainExcitationWavelength>
        <MainEmissionWavelength Unit="nm">{emission_wavelengths[channel_idx]}</MainEmissionWavelength>
        <ObjectiveMagnification Unit="">10</ObjectiveMagnification>
        <ObjectiveNA Unit="">0.3</ObjectiveNA>
        <ExposureTime Unit="s">{exposure_times[channel_idx]}</ExposureTime>
        <OrientationMatrix>[[1.009457,0,0,34.3],[0,-1.009457,0,-15.1],[0,0,1.33,-6.014]]</OrientationMatrix>
        <CropArea>[[0,0],[2160,2160],[2160,2160]]</CropArea>
      </Entry>"""

        # Add image information section
        xml_content += f"""
    </Map>
    <PixelSizeCalibration>
      <PixelSize Unit="µm">{pixel_size_meters * 1e6:.4f}</PixelSize>
      <MagnificationRatio>1.0</MagnificationRatio>
    </PixelSizeCalibration>"""

        # Generate Opera Phenix center-first snake pattern for field IDs
        field_pattern = self.generate_opera_phenix_field_pattern(self.grid_size[0], self.grid_size[1])

        # Add detailed image information for each image
        for row, col in well_indices:
            for site in range(1, self.grid_size[0] * self.grid_size[1] + 1):
                # Get grid position for this field ID using Opera Phenix pattern
                site_row, site_col = field_pattern[site]

                # Calculate position in meters (typical Opera Phenix values)
                # These are arbitrary values for demonstration
                # NOTE: In microscope stage coordinates, Y increases upward (higher Y = top row)
                # But in array coordinates, row index increases downward (higher row = bottom)
                # So we need to invert: stage_y = (grid_rows - 1 - array_row)
                pos_x = 0.000576762 + site_col * 0.001  # X: left to right
                pos_y = 0.000576762 + (self.grid_size[0] - 1 - site_row) * 0.001  # Y: inverted (higher = top)

                for z in range(1, self.z_stack_levels + 1):
                    # Calculate Z position based on Z level
                    pos_z = 0.0001 + (z - 1) * z_step_size_meters
                    abs_pos_z = 0.135809004 + (z - 1) * z_step_size_meters  # Arbitrary base Z position

                    for channel in range(1, self.wavelengths + 1):
                        # Create image ID
                        image_id = f"{row:02d}{col:02d}K1F{site}P{z}R{channel}"

                        # Create URL (filename)
                        url = f"r{row:02d}c{col:02d}f{site}p{z:02d}-ch{channel}sk1fk1fl1.tiff"

                        # Add image element
                        xml_content += f"""
    <Image Version="1">
      <id>{image_id}</id>
      <State>Ok</State>
      <URL>{url}</URL>
      <Row>{row}</Row>
      <Col>{col}</Col>
      <FieldID>{site}</FieldID>
      <PlaneID>{z}</PlaneID>
      <TimepointID>1</TimepointID>
      <SequenceID>1</SequenceID>
      <GroupID>1</GroupID>
      <ChannelID>{channel}</ChannelID>
      <FlimID>1</FlimID>
      <PositionX Unit="m">{pos_x}</PositionX>
      <PositionY Unit="m">{pos_y}</PositionY>
      <PositionZ Unit="m">{pos_z}</PositionZ>
      <AbsPositionZ Unit="m">{abs_pos_z}</AbsPositionZ>
      <MeasurementTimeOffset Unit="s">0</MeasurementTimeOffset>
      <AbsTime>{datetime.now().isoformat()}-04:00</AbsTime>
    </Image>"""

        # Close the XML
        xml_content += """
  </Images>
</EvaluationInputData>"""

        # Write the XML file
        with open(index_xml_path, 'w', encoding='utf-8') as f:
            f.write(xml_content)

        return index_xml_path

    def _calculate_image_size(self, grid_size, tile_size, overlap_percent, stage_error_px):
        """
        Calculate the appropriate image size based on grid dimensions, tile size, and overlap.

        Args:
            grid_size: Tuple of (rows, cols) for the grid of tiles
            tile_size: Size of each tile (width, height)
            overlap_percent: Percentage of overlap between tiles
            stage_error_px: Random error in stage positioning (pixels)

        Returns:
            tuple: (width, height) of the calculated image size
        """
        # Calculate effective step size with overlap
        step_x = int(tile_size[0] * (1 - overlap_percent / 100))
        step_y = int(tile_size[1] * (1 - overlap_percent / 100))

        # Calculate minimum required size
        min_width = step_x * (grid_size[1] - 1) + tile_size[0]
        min_height = step_y * (grid_size[0] - 1) + tile_size[1]

        # Add margin for stage positioning errors
        margin = stage_error_px * 2
        width = min_width + margin
        height = min_height + margin

        return (width, height)

    def generate_dataset(self):
        """Generate the complete dataset."""
        print(f"Generating synthetic microscopy dataset in {self.output_dir}")
        print(f"Grid size: {self.grid_size[0]}x{self.grid_size[1]}")
        print(f"Wavelengths: {self.wavelengths}")
        print(f"Z-stack levels: {self.z_stack_levels}")
        print(f"Wells: {', '.join(self.wells)}")

        # Generate HTD file
        htd_path = self.generate_htd_file()
        print(f"Generated HTD file: {htd_path}")

        # Process each well
        for well_index, well in enumerate(self.wells):
            print(f"\nGenerating data for well {well}...")

            # Use a different random seed for each well if base seed is provided
            if self.base_random_seed is not None:
                well_seed = self.base_random_seed + well_index
                np.random.seed(well_seed)
                random.seed(well_seed)
                print(f"Using random seed {well_seed} for well {well}")

            # Pre-generate the positions for each site to ensure consistency across Z-levels
            # This creates a mapping of site_index -> (base_x_pos, base_y_pos)
            site_positions = {}

            # For Opera Phenix, use center-first snake pattern
            # For ImageXpress, use simple raster pattern
            if self.format == 'OperaPhenix':
                field_pattern = self.generate_opera_phenix_field_pattern(self.grid_size[0], self.grid_size[1])
                for site_index in range(1, self.grid_size[0] * self.grid_size[1] + 1):
                    # Get grid position for this field ID
                    grid_row, grid_col = field_pattern[site_index]

                    # Calculate base position
                    x = grid_col * self.step_x
                    y = grid_row * self.step_y

                    # Add random stage positioning error
                    x_error = np.random.randint(-self.stage_error_px, self.stage_error_px)
                    y_error = np.random.randint(-self.stage_error_px, self.stage_error_px)

                    x_pos = x + x_error
                    y_pos = y + y_error

                    # Ensure we don't go out of bounds
                    x_pos = max(0, min(x_pos, self.image_size[0] - self.tile_size[0]))
                    y_pos = max(0, min(y_pos, self.image_size[1] - self.tile_size[1]))

                    site_positions[site_index] = (x_pos, y_pos)
            else:
                # ImageXpress: simple raster pattern
                site_index = 1
                for row in range(self.grid_size[0]):
                    for col in range(self.grid_size[1]):
                        # Calculate base position
                        x = col * self.step_x
                        y = row * self.step_y

                        # Add random stage positioning error
                        x_error = np.random.randint(-self.stage_error_px, self.stage_error_px)
                        y_error = np.random.randint(-self.stage_error_px, self.stage_error_px)

                        x_pos = x + x_error
                        y_pos = y + y_error

                        # Ensure we don't go out of bounds
                        x_pos = max(0, min(x_pos, self.image_size[0] - self.tile_size[0]))
                        y_pos = max(0, min(y_pos, self.image_size[1] - self.tile_size[1]))

                        site_positions[site_index] = (x_pos, y_pos)
                        site_index += 1

            # For multiple Z-stack levels
            if self.z_stack_levels > 1:
                # Handle differently based on format
                if self.format == 'ImageXpress':
                    # For ImageXpress, create ZStep folders
                    # Make sure all ZStep folders are created first
                    for z in range(self.z_stack_levels):
                        z_level = z + 1  # 1-based Z level index
                        zstep_dir = self.timepoint_dir / f"ZStep_{z_level}"
                        zstep_dir.mkdir(exist_ok=True)
                        print(f"Created ZStep folder: {zstep_dir}")
                else:  # OperaPhenix
                    # Opera Phenix doesn't use ZStep folders - all images go directly in the Images folder
                    print("Opera Phenix format: all Z-stack images will be placed directly in the Images folder")

                # Now generate images for each Z-level
                for z in range(self.z_stack_levels):
                    z_level = z + 1  # 1-based Z level index

                    # For ImageXpress, use ZStep folders; for Opera Phenix, use the Images folder directly
                    if self.format == 'ImageXpress':
                        target_dir = self.timepoint_dir / f"ZStep_{z_level}"
                    else:  # OperaPhenix
                        target_dir = self.timepoint_dir  # This is already set to self.images_dir for Opera Phenix

                    # Generate images for each wavelength at this Z level
                    for w in range(self.wavelengths):
                        wavelength = w + 1  # 1-based wavelength index

                        # Generate full image
                        print(f"Generating full image for wavelength {wavelength}, Z level {z_level}...")
                        full_image = self.generate_cell_image(w, z, well=well)

                        # Save tiles for this Z level using the pre-generated positions
                        site_index = 1
                        for row in range(self.grid_size[0]):
                            for col in range(self.grid_size[1]):
                                # Get the pre-generated position
                                x_pos, y_pos = site_positions[site_index]

                                # Extract tile
                                tile = full_image[
                                    y_pos:y_pos + self.tile_size[1],
                                    x_pos:x_pos + self.tile_size[0]
                                ]

                                # Create filename based on format
                                if self.format == 'ImageXpress':
                                    filename = self.imagexpress_filename(
                                        well=well,
                                        site=site_index,
                                        channel=wavelength,
                                        z_index=z_level,
                                    )
                                else:  # OperaPhenix
                                    # Opera Phenix format: rXXcYYfZZZpWW-chVskNfkNflN.tiff
                                    # Extract row and column from well ID (e.g., 'A01' -> row=1, col=1)
                                    row = ord(well[0]) - ord('A') + 1
                                    col = int(well[1:3])
                                    filename = f"r{row:02d}c{col:02d}f{site_index}p{z_level:02d}-ch{wavelength}sk1fk1fl1.tiff"
                                filepath = target_dir / filename

                                # Skip this file if it's in the skip list
                                if filename in self.skip_files:
                                    print(f"  Skipped tile: {filename} (simulating missing image)")
                                    site_index += 1
                                    continue

                                # Save image without compression
                                tifffile.imwrite(filepath, tile, compression=None)

                                # Print progress with full path for debugging
                                print(f"  Saved tile: {target_dir.name}/{filename} (position: {x_pos}, {y_pos})")
                                print(f"  Full path: {filepath.resolve()}")
                                site_index += 1
            else:
                # For single Z level (no Z-stack), just save files directly in TimePoint_1
                for w in range(self.wavelengths):
                    wavelength = w + 1  # 1-based wavelength index

                    # Generate full image for the single Z level
                    print(f"Generating full image for wavelength {wavelength} (no Z-stack)...")
                    full_image = self.generate_cell_image(w, 0, well=well)

                    # Save tiles without Z-stack index
                    site_index = 1
                    for row in range(self.grid_size[0]):
                        for col in range(self.grid_size[1]):
                            # Get the pre-generated position
                            x_pos, y_pos = site_positions[site_index]

                            # Extract tile
                            tile = full_image[
                                y_pos:y_pos + self.tile_size[1],
                                x_pos:x_pos + self.tile_size[0]
                            ]

                            # Create filename based on format
                            if self.format == 'ImageXpress':
                                filename = self.imagexpress_filename(
                                    well=well,
                                    site=site_index,
                                    channel=wavelength,
                                    z_index=1,
                                )
                            else:  # OperaPhenix
                                # Opera Phenix format: rXXcYYfZZZpWW-chVskNfkNflN.tiff
                                # Extract row and column from well ID (e.g., 'A01' -> row=1, col=1)
                                row = ord(well[0]) - ord('A') + 1
                                col = int(well[1:3])
                                filename = f"r{row:02d}c{col:02d}f{site_index}p01-ch{wavelength}sk1fk1fl1.tiff"
                            filepath = self.timepoint_dir / filename

                            # Skip this file if it's in the skip list
                            if filename in self.skip_files:
                                print(f"  Skipped tile: {filename} (simulating missing image)")
                                site_index += 1
                                continue

                            # Save image without compression
                            tifffile.imwrite(filepath, tile, compression=None)

                            # Print progress
                            print(f"  Saved tile: {filename} (position: {x_pos}, {y_pos})")
                            site_index += 1

        print("Dataset generation complete!")

        # If OpenHCS format requested, generate metadata file
        if self.openhcs_format:
            # Determine subdirectory based on format
            if self.format == "ImageXpress":
                sub_dir = "TimePoint_1"
            else:  # OperaPhenix
                sub_dir = "Images"

            self.generate_openhcs_metadata(sub_dir=sub_dir, pixel_size=0.65)

    def imagexpress_filename(
        self,
        *,
        well: str,
        site: int,
        channel: int,
        z_index: int,
    ) -> str:
        """Return an ImageXpress synthetic filename for the configured compatibility mode."""
        if self.imagexpress_bioformats_compatible:
            plate_name = self.output_dir.name
            parts = [f"{plate_name}_{well}"]
            if self.grid_size[0] * self.grid_size[1] > 1:
                parts.append(f"_s{site}")
            if self.wavelengths > 1:
                parts.append(f"_w{channel}")
            return "".join(parts) + ".tif"
        if self.include_all_components:
            return self.imagexpress_parser.construct_filename(
                well=well,
                site=site,
                channel=channel,
                z_index=z_index,
                timepoint=1,
                extension='.tif',
            )
        return self.imagexpress_parser.construct_filename(
            well=well,
            site=site,
            channel=channel,
            extension='.tif',
        )

    def generate_openhcs_metadata(self, sub_dir: str = "images", pixel_size: float = 0.65):
        """
        Generate OpenHCS metadata file for the synthetic dataset.

        This converts the generated ImageXpress or OperaPhenix data into OpenHCS format
        by adding an openhcs_metadata.json file with the subdirectory-keyed structure.

        Args:
            sub_dir: Subdirectory name where images are located (default: "images")
            pixel_size: Pixel size in microns (default: 0.65)
        """
        print("\nGenerating OpenHCS metadata...")

        # Collect all image files
        image_files = []
        workspace_mapping: Dict[str, SourcePixelRef] = {}
        if self.format == 'ImageXpress':
            # ImageXpress stores images in TimePoint_1/ZStep_X/ or TimePoint_1/ for flat plates
            timepoint_dir = self.output_dir / "TimePoint_1"
            if self.z_stack_levels > 1:
                # Z-stack: images in ZStep folders
                for z in range(1, self.z_stack_levels + 1):
                    zstep_dir = timepoint_dir / f"ZStep_{z}"
                    if zstep_dir.exists():
                        for img_file in sorted(zstep_dir.glob("*.tif")):
                            virtual_path = f"{sub_dir}/{img_file.name}"
                            image_files.append(virtual_path)
                            workspace_mapping[virtual_path] = SourcePixelRef(
                                backend=Backend.DISK.value,
                                backend_address=img_file.relative_to(
                                    self.output_dir
                                ).as_posix(),
                            )
            else:
                # Flat: images directly in TimePoint_1
                if timepoint_dir.exists():
                    for img_file in sorted(timepoint_dir.glob("*.tif")):
                        rel_path = f"{sub_dir}/{img_file.name}"
                        image_files.append(rel_path)
        else:  # OperaPhenix
            # OperaPhenix stores images in Images/ folder
            images_dir = self.output_dir / "Images"
            if images_dir.exists():
                for img_file in sorted(images_dir.glob("*.tiff")):
                    rel_path = f"{sub_dir}/{img_file.name}"
                    image_files.append(rel_path)

        # Build component metadata dictionaries
        channels = {str(i): str(i) for i in range(1, self.wavelengths + 1)}
        wells = {well: well for well in self.wells}

        # Calculate number of sites from grid
        num_sites = self.grid_size[0] * self.grid_size[1]
        sites = {str(i): str(i) for i in range(1, num_sites + 1)}

        # Z-indexes
        z_indexes = {str(i): str(i) for i in range(1, self.z_stack_levels + 1)} if self.z_stack_levels > 1 else None

        # Timepoints (always 1 for synthetic data)
        timepoints = {"1": "1"}

        # Determine parser name based on format
        parser_name = "ImageXpressFilenameParser" if self.format == "ImageXpress" else "OperaPhenixFilenameParser"
        microscope_handler = "imagexpress" if self.format == "ImageXpress" else "operaphenix"

        # Build metadata structure
        metadata = {
            "subdirectories": {
                sub_dir: {
                    "microscope_handler_name": microscope_handler,
                    "source_filename_parser_name": parser_name,
                    "grid_dimensions": list(self.grid_size),
                    "pixel_size": pixel_size,
                    "image_files": image_files,
                    "channels": channels,
                    "wells": wells,
                    "sites": sites,
                    "z_indexes": z_indexes,
                    "timepoints": timepoints,
                    "available_backends": {
                        "disk": True,
                        **({"virtual_workspace": True} if workspace_mapping else {}),
                    },
                    **(
                        {
                            "workspace_mapping": {
                                virtual_path: source_ref.to_workspace_mapping()
                                for virtual_path, source_ref in workspace_mapping.items()
                            }
                        }
                        if workspace_mapping
                        else {}
                    ),
                    "main": True
                }
            }
        }

        # Write metadata file
        metadata_path = self.output_dir / "openhcs_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Generated OpenHCS metadata: {metadata_path}")
        print(f"  - {len(image_files)} image files")
        print(f"  - {len(wells)} wells")
        print(f"  - {len(channels)} channels")
        print(f"  - {len(sites)} sites")
        if z_indexes:
            print(f"  - {len(z_indexes)} z-levels")

        return metadata_path
