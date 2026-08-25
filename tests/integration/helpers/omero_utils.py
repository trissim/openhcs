#!/usr/bin/env python3
"""
Generate and upload synthetic test data to OMERO as a Plate.

This script:
1. Generates synthetic microscopy images using SyntheticMicroscopyGenerator
2. Uploads them to OMERO as a proper Plate/Well/WellSample structure
3. Returns plate ID for use in demo
"""

import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import omero.gateway
import omero.model
from omero.gateway import BlitzGateway
from omero.rtypes import rint, rstring
from polystore import OMEROWellAddress


def _plane_sequence(
    planes_by_cz: dict[tuple[int, int], np.ndarray],
    *,
    size_c: int,
    size_z: int,
    size_t: int,
):
    """Yield one rectangular OMERO plane sequence in T/C/Z order."""

    empty_plane = np.zeros_like(next(iter(planes_by_cz.values())))
    for _timepoint in range(size_t):
        for channel in range(1, size_c + 1):
            for z_index in range(1, size_z + 1):
                yield planes_by_cz.get((channel, z_index), empty_plane)


def upload_plate_to_omero(
    conn,
    data_dir: str,
    plate_name: str = "OpenHCS_Synthetic_Plate",
    grid_dimensions: tuple[int, int] | None = None,
):
    """
    Upload existing microscopy data directory to OMERO as a Plate.

    Args:
        conn: OMERO BlitzGateway connection
        data_dir: Directory containing microscopy images
        plate_name: Name for the OMERO plate
        grid_dimensions: Optional (rows, cols) grid dimensions to store as metadata

    Returns:
        int: OMERO plate ID
    """
    # Create Plate in OMERO
    print("\n[2/4] Creating OMERO Plate...")
    update_service = conn.getUpdateService()

    plate = omero.model.PlateI()
    plate.setName(rstring(plate_name))
    plate.setColumnNamingConvention(rstring("number"))
    plate.setRowNamingConvention(rstring("letter"))
    plate = update_service.saveAndReturnObject(plate)
    plate_id = plate.getId().getValue()

    print(f"✓ Created plate: {plate_name} (ID: {plate_id})")

    # Add the OpenHCS-owned grid projection when the fixture declares it.
    if grid_dimensions:
        from openhcs.microscopes.omero import OMEROMetadataHandler

        map_ann = omero.model.MapAnnotationI()
        map_ann.setNs(rstring("polystore.metadata"))
        map_ann.setMapValue(
            [
                omero.model.NamedValue(
                    OMEROMetadataHandler.GRID_DIMENSIONS_METADATA_KEY,
                    f"{grid_dimensions[0]},{grid_dimensions[1]}",
                )
            ]
        )
        map_ann = update_service.saveAndReturnObject(map_ann)

        link = omero.model.PlateAnnotationLinkI()
        link.setParent(plate)
        link.setChild(map_ann)
        update_service.saveAndReturnObject(link)
        print("✓ Added OpenHCS grid metadata to plate")

    # Upload images and create Wells
    print("\n[3/4] Uploading images and creating Wells...")

    # Group images by well and site
    image_files = sorted(Path(data_dir).rglob("*.tif"))
    print(f"  Found {len(image_files)} images to upload")

    # Parse well positions from filenames
    # ImageXpress format: <well>_s<site>_w<channel>_z<z>.tif
    wells_data = defaultdict(lambda: defaultdict(list))

    for img_path in image_files:
        # Extract well and site from filename (e.g., "A01_s001_w1_z001.tif" or "A01_s1_w1.tif")
        filename = img_path.name
        parts = filename.split("_")
        well_id = parts[0]  # "A01"
        site_match = re.search(r"s(\d+)", filename)
        channel_match = re.search(r"w(\d+)", filename)
        z_match = re.search(r"z(\d+)", filename)

        if site_match and channel_match:
            site = int(site_match.group(1))
            channel = int(channel_match.group(1))
            z = (
                int(z_match.group(1)) if z_match else 1
            )  # Default to z=1 if no z in filename
            wells_data[well_id][site].append((channel, z, img_path))

    # Create Wells and upload images
    for well_id, sites_data in sorted(wells_data.items()):
        well_address = OMEROWellAddress.from_label(well_id)

        # Create Well
        well = omero.model.WellI()
        well.setPlate(omero.model.PlateI(plate_id, False))
        well.setColumn(rint(well_address.column_index))
        well.setRow(rint(well_address.row_index))

        # Create one WellSample per site
        for site, planes_data in sorted(sites_data.items()):
            # Group planes by channel and z
            import tifffile

            planes_by_cz = {}
            max_channel = 0
            max_z = 0

            for channel, z, img_path in planes_data:
                img_data = tifffile.imread(img_path)
                # Ensure 2D
                if img_data.ndim == 3:
                    img_data = img_data[0]  # Take first plane if 3D
                planes_by_cz[(channel, z)] = img_data
                max_channel = max(max_channel, channel)
                max_z = max(max_z, z)

            sizeC = max_channel
            sizeZ = max_z
            sizeT = 1

            # Upload image to OMERO
            image = conn.createImageFromNumpySeq(
                zctPlanes=_plane_sequence(
                    planes_by_cz,
                    size_c=sizeC,
                    size_z=sizeZ,
                    size_t=sizeT,
                ),
                imageName=f"{well_id}_s{site:03d}",
                sizeZ=sizeZ,
                sizeC=sizeC,
                sizeT=sizeT,
                description=f"Site {site} of well {well_id}",
            )

            # Create WellSample linking image to well
            ws = omero.model.WellSampleI()
            ws.setImage(omero.model.ImageI(image.getId(), False))
            ws.setWell(well)
            well.addWellSample(ws)

        # Save well with all its samples
        update_service.saveObject(well)
        print(
            f"  Created well {well_id} "
            f"(row={well_address.row_index}, col={well_address.column_index}) "
            f"with {len(sites_data)} sites"
        )

    print(f"✓ Created {len(wells_data)} wells with proper site structure")

    return plate_id


def generate_and_upload_synthetic_plate(
    conn,
    plate_name: str = "OpenHCS_Synthetic_Plate",
    grid_size=(2, 2),
    tile_size=(128, 128),
    wavelengths=2,
    z_stack_levels=3,
    wells=("A01", "A02", "B01", "B02"),
):
    """
    Generate synthetic data and upload to OMERO as a Plate.

    This creates a proper HCS Plate structure with Wells and WellSamples,
    preserving the plate organization that OpenHCS expects.
    """

    print("[1/4] Generating synthetic microscopy data...")
    print(
        f"  Grid: {grid_size[0]}x{grid_size[1]}, Tile: {tile_size}, Channels: {wavelengths}, Z-levels: {z_stack_levels}"
    )
    print(f"  Wells: {wells}")

    # Generate synthetic data to temp directory
    from openhcs.demo.synthetic_data import SyntheticMicroscopyGenerator

    with tempfile.TemporaryDirectory() as tmpdir:
        generator = SyntheticMicroscopyGenerator(
            output_dir=tmpdir,
            grid_size=grid_size,
            tile_size=tile_size,
            overlap_percent=10,
            wavelengths=wavelengths,
            z_stack_levels=z_stack_levels,
            wells=wells,
            format="ImageXpress",
            auto_image_size=True,
        )
        generator.generate_dataset()

        print(f"✓ Generated synthetic data for {len(wells)} wells")

        # Upload to OMERO with grid dimensions
        return upload_plate_to_omero(
            conn, tmpdir, plate_name, grid_dimensions=grid_size
        )


def main():
    """Generate synthetic plate and upload to OMERO."""

    # Connect to OMERO
    print("Connecting to OMERO...")
    conn = BlitzGateway(
        "root", "omero-root-password", host="localhost", port=4064, secure=False
    )
    if not conn.connect():
        print("❌ Failed to connect to OMERO")
        print("   Make sure OMERO is running: docker-compose up -d")
        sys.exit(1)

    print("✓ Connected to OMERO\n")

    try:
        plate_id = generate_and_upload_synthetic_plate(
            conn, wells=["A01", "A02", "B01", "B02"]
        )

        print("\n✅ Setup complete!")
        print(f"   Plate ID: {plate_id}")
        print(f"\n   Plate ID: {plate_id}")
        print(
            f"\n   View in OMERO.web: http://localhost:4080/webclient/?show=plate-{plate_id}"
        )

        return plate_id

    finally:
        conn.close()


if __name__ == "__main__":
    main()
