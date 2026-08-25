from pathlib import Path

import pytest

from openhcs.constants.constants import AllComponents
from openhcs.core.source_projection import OpenHCSPlaneAddress
from openhcs.microscopes.bioformats_adapter import (
    BioFormatsDatasetAmbiguityError,
    BioFormatsImage,
    BioFormatsPixels,
    BioFormatsPlane,
    BioFormatsPlate,
    BioFormatsStoreMetadata,
    BioFormatsWell,
    BioFormatsWellSample,
)


def _image(
    root: Path,
    image_id: str,
    series_index: int,
    *,
    size_c: int = 1,
    size_t: int = 1,
    source_name: str | None = None,
) -> BioFormatsImage:
    source = root / (source_name or f"{image_id}.npy")
    return BioFormatsImage(
        image_id=image_id,
        image_name=f"Sample {image_id}",
        source_path=source,
        source_files=(source,),
        series_index=series_index,
        pixels=BioFormatsPixels(
            size_c=size_c,
            size_z=1,
            size_t=size_t,
            planes=tuple(
                BioFormatsPlane(c=channel, z=1, t=timepoint, index=index)
                for index, (timepoint, channel) in enumerate(
                    (t, c) for t in range(1, size_t + 1) for c in range(1, size_c + 1)
                )
            ),
        ),
        channel_names=tuple(f"Channel {channel}" for channel in range(1, size_c + 1)),
        pixel_size=0.65,
        reader="npy",
    )


def test_store_metadata_emits_exact_plate_planes(tmp_path: Path) -> None:
    metadata = BioFormatsStoreMetadata(
        root=tmp_path,
        plates=(
            BioFormatsPlate(
                plate_id="Plate:0",
                name="fixture",
                wells=(
                    BioFormatsWell(
                        well_id="Well:0:1",
                        row=0,
                        column=1,
                        samples=(
                            BioFormatsWellSample(
                                sample_id="WellSample:0:1:2",
                                image_id="Image:1",
                                index=2,
                            ),
                        ),
                    ),
                ),
            ),
        ),
        images=(_image(tmp_path, "Image:1", 4, size_c=2, size_t=2),),
    )

    dataset = metadata.source_dataset()

    assert dataset.identity.value == "Plate:0"
    assert [candidate.declared_address for candidate in dataset.candidates] == [
        OpenHCSPlaneAddress.from_values("A02", "3", "1", "1", "1"),
        OpenHCSPlaneAddress.from_values("A02", "3", "2", "1", "1"),
        OpenHCSPlaneAddress.from_values("A02", "3", "1", "1", "2"),
        OpenHCSPlaneAddress.from_values("A02", "3", "2", "1", "2"),
    ]


def test_nonplate_images_map_to_distinct_well_samples(tmp_path: Path) -> None:
    dataset = BioFormatsStoreMetadata(
        root=tmp_path,
        declared_dataset_id="Dataset:nonplate",
        images=(
            _image(tmp_path, "Image:sample-a", 0),
            _image(tmp_path, "Image:sample-b", 1),
        ),
    ).source_dataset()

    assert [
        candidate.declared_address.value_for(AllComponents.WELL)
        for candidate in dataset.candidates
    ] == [
        "Image%3Asample-a.npy",
        "Image%3Asample-b.npy",
    ]
    assert [
        candidate.declared_address.value_for(AllComponents.SITE)
        for candidate in dataset.candidates
    ] == [
        "1",
        "2",
    ]


def test_one_nonplate_czi_maps_many_scenes_to_exact_series_sites(
    tmp_path: Path,
) -> None:
    dataset = BioFormatsStoreMetadata(
        root=tmp_path,
        images=(
            _image(tmp_path, "Image:scene-a", 0, source_name="many-scenes.czi"),
            _image(tmp_path, "Image:scene-b", 3, source_name="many-scenes.czi"),
        ),
    ).source_dataset()

    assert dataset.identity.value == tmp_path.resolve().as_uri()
    assert [candidate.declared_address for candidate in dataset.candidates] == [
        OpenHCSPlaneAddress.from_values("many-scenes.czi", "1", "1", "1", "1"),
        OpenHCSPlaneAddress.from_values("many-scenes.czi", "4", "1", "1", "1"),
    ]
    assert {
        candidate.store_identity.container_key for candidate in dataset.candidates
    } == {((tmp_path / "many-scenes.czi").resolve(),)}
    assert {
        candidate.store_identity.sample_group_id for candidate in dataset.candidates
    } == {"Image:scene-a", "Image:scene-b"}


def test_one_plate_czi_preserves_many_wells_and_sparse_sample_indexes(
    tmp_path: Path,
) -> None:
    dataset = BioFormatsStoreMetadata(
        root=tmp_path,
        plates=(
            BioFormatsPlate(
                plate_id="Plate:many-samples",
                name="fixture",
                wells=(
                    BioFormatsWell(
                        well_id="Well:A01",
                        row=0,
                        column=0,
                        samples=(
                            BioFormatsWellSample("Sample:0", "Image:0", 0),
                            BioFormatsWellSample("Sample:4", "Image:4", 4),
                        ),
                    ),
                    BioFormatsWell(
                        well_id="Well:B02",
                        row=1,
                        column=1,
                        samples=(BioFormatsWellSample("Sample:9", "Image:9", 9),),
                    ),
                ),
            ),
        ),
        images=(
            _image(tmp_path, "Image:0", 0, source_name="plate.czi"),
            _image(tmp_path, "Image:4", 1, source_name="plate.czi"),
            _image(tmp_path, "Image:9", 2, source_name="plate.czi"),
        ),
    ).source_dataset()

    assert dataset.identity.value == "Plate:many-samples"
    assert [candidate.declared_address for candidate in dataset.candidates] == [
        OpenHCSPlaneAddress.from_values("A01", "1", "1", "1", "1"),
        OpenHCSPlaneAddress.from_values("A01", "5", "1", "1", "1"),
        OpenHCSPlaneAddress.from_values("B02", "10", "1", "1", "1"),
    ]
    assert {
        candidate.store_identity.sample_group_id for candidate in dataset.candidates
    } == {"Sample:0", "Sample:4", "Sample:9"}


def test_one_container_rejects_multiple_embedded_plates_actionably(
    tmp_path: Path,
) -> None:
    metadata = BioFormatsStoreMetadata(
        root=tmp_path,
        plates=(
            BioFormatsPlate("Plate:first", None, ()),
            BioFormatsPlate("Plate:second", None, ()),
        ),
        images=(_image(tmp_path, "Image:0", 0, source_name="two-plates.czi"),),
    )

    with pytest.raises(
        BioFormatsDatasetAmbiguityError,
        match=r"Plate:first.*Plate:second.*one embedded dataset identity.*Source bindings",
    ):
        metadata.source_dataset()


def test_store_metadata_rejects_incomplete_planes(tmp_path: Path) -> None:
    source = tmp_path / "incomplete.npy"
    with pytest.raises(ValueError, match="complete exact C/Z/T"):
        BioFormatsImage(
            image_id="Image:0",
            image_name=None,
            source_path=source,
            source_files=(source,),
            series_index=0,
            pixels=BioFormatsPixels(
                size_c=2,
                size_z=1,
                size_t=1,
                planes=(BioFormatsPlane(c=1, z=1, t=1, index=0),),
            ),
            channel_names=("DAPI", "GFP"),
            pixel_size=0.65,
        )


def test_store_metadata_rejects_duplicate_sample_identity(tmp_path: Path) -> None:
    metadata = BioFormatsStoreMetadata(
        root=tmp_path,
        plates=(
            BioFormatsPlate(
                plate_id="Plate:0",
                name=None,
                wells=(
                    BioFormatsWell(
                        well_id="Well:0:0",
                        row=0,
                        column=0,
                        samples=(
                            BioFormatsWellSample("Sample:0", "Image:1", 0),
                            BioFormatsWellSample("Sample:0", "Image:2", 1),
                        ),
                    ),
                ),
            ),
        ),
        images=(
            _image(tmp_path, "Image:1", 0),
            _image(tmp_path, "Image:2", 1),
        ),
    )

    with pytest.raises(RuntimeError, match="Duplicate OME WellSample.ID"):
        metadata.source_dataset()
