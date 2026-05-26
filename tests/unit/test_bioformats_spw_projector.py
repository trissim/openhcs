from pathlib import Path

import pytest

from openhcs.microscopes.bioformats_adapter import (
    BioFormatsImage,
    BioFormatsMetadata,
    BioFormatsPixels,
    BioFormatsPlane,
    BioFormatsPlate,
    BioFormatsWell,
    BioFormatsWellSample,
)
from openhcs.microscopes.bioformats_spw_projector import (
    BioFormatsProjectionError,
    BioFormatsSPWProjector,
)


def test_projector_maps_ome_spw_to_openhcs_axes(tmp_path: Path) -> None:
    metadata = BioFormatsMetadata(
        root=tmp_path,
        plates=(
            BioFormatsPlate(
                wells=(
                    BioFormatsWell(row=0, column=0, samples=()),
                    BioFormatsWell(
                        row=0,
                        column=1,
                        samples=(BioFormatsWellSample(image_id="image:1", index=2),),
                    ),
                )
            ),
        ),
        images=(
            BioFormatsImage(
                image_id="image:1",
                source_path=tmp_path / "stack.npy",
                series_index=4,
                pixels=BioFormatsPixels(
                    size_c=2,
                    size_z=1,
                    size_t=2,
                    planes=tuple(
                        BioFormatsPlane(c=c, z=z, t=t, index=index)
                        for c, z, t, index in (
                            (1, 1, 1, 0),
                            (2, 1, 1, 1),
                            (1, 1, 2, 2),
                            (2, 1, 2, 3),
                        )
                    ),
                ),
                channel_names=("DAPI", "GFP"),
                pixel_size=0.65,
                reader="npy",
            ),
        ),
    )

    dataset = BioFormatsSPWProjector().project(metadata)

    assert [(entry.well, entry.site, entry.channel, entry.z_index, entry.timepoint) for entry in dataset.entries] == [
        ("A02", 1, 1, 1, 1),
        ("A02", 1, 2, 1, 1),
        ("A02", 1, 1, 1, 2),
        ("A02", 1, 2, 1, 2),
    ]
    assert [entry.plane_index for entry in dataset.entries] == [0, 1, 2, 3]
    assert dataset.entries[1].channel_name == "GFP"
    assert dataset.entries[0].pixel_size == 0.65


def test_projector_numbers_sites_per_well_not_from_global_ome_well_sample_index(
    tmp_path: Path,
) -> None:
    metadata = BioFormatsMetadata(
        root=tmp_path,
        plates=(
            BioFormatsPlate(
                wells=(
                    BioFormatsWell(
                        row=0,
                        column=0,
                        samples=(
                            BioFormatsWellSample(image_id="image:1", index=12),
                            BioFormatsWellSample(image_id="image:2", index=13),
                        ),
                    ),
                    BioFormatsWell(
                        row=1,
                        column=0,
                        samples=(BioFormatsWellSample(image_id="image:3", index=44),),
                    ),
                )
            ),
        ),
        images=tuple(
            BioFormatsImage(
                image_id=f"image:{index}",
                source_path=tmp_path / f"image-{index}.tif",
                series_index=index - 1,
                pixels=BioFormatsPixels(
                    size_c=1,
                    size_z=1,
                    size_t=1,
                    planes=(BioFormatsPlane(c=1, z=1, t=1, index=0),),
                ),
            )
            for index in range(1, 4)
        ),
    )

    dataset = BioFormatsSPWProjector().project(metadata)

    assert [(entry.well, entry.site) for entry in dataset.entries] == [
        ("A01", 1),
        ("A01", 2),
        ("B01", 1),
    ]


def test_projector_fails_when_plate_has_no_well_samples(tmp_path: Path) -> None:
    metadata = BioFormatsMetadata(
        root=tmp_path,
        plates=(BioFormatsPlate(wells=(BioFormatsWell(row=0, column=0, samples=()),)),),
        images=(),
    )

    with pytest.raises(BioFormatsProjectionError, match="no image-plane entries"):
        BioFormatsSPWProjector().project(metadata)


def test_projector_fails_without_spw_metadata(tmp_path: Path) -> None:
    metadata = BioFormatsMetadata(root=tmp_path, plates=(), images=())

    with pytest.raises(BioFormatsProjectionError, match="no OME Plate"):
        BioFormatsSPWProjector().project(metadata)


def test_projector_fails_on_incomplete_plane_mapping(tmp_path: Path) -> None:
    metadata = BioFormatsMetadata(
        root=tmp_path,
        plates=(
            BioFormatsPlate(
                wells=(
                    BioFormatsWell(
                        row="B",
                        column="03",
                        samples=(BioFormatsWellSample(image_id="image:1"),),
                    ),
                )
            ),
        ),
        images=(
            BioFormatsImage(
                image_id="image:1",
                source_path=tmp_path / "stack.npy",
                series_index=0,
                pixels=BioFormatsPixels(
                    size_c=2,
                    size_z=1,
                    size_t=1,
                    planes=(BioFormatsPlane(c=1, z=1, t=1, index=0),),
                ),
            ),
        ),
    )

    with pytest.raises(BioFormatsProjectionError, match="complete stable C/Z/T"):
        BioFormatsSPWProjector().project(metadata)
