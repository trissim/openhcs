from pathlib import Path

from benchmark.bioformats_hcs_validation import (
    bioformats_hcs_validation_specs,
    validate_acquired_bioformats_hcs_dataset,
)
from benchmark.contracts.dataset import (
    AcquiredDataset,
    ArchiveFormat,
    BenchmarkDatasetTag,
    DatasetSourceKind,
    DatasetSourceSpec,
    DatasetSpec,
    DatasetValidationRule,
)
from benchmark.datasets.registry import DATASET_REGISTRY
from tests.unit.bioformats_fixture import write_bioformats_manifest_fixture


def test_bioformats_hcs_validation_specs_are_tagged_url_file_sources() -> None:
    specs = {spec.id: spec for spec in bioformats_hcs_validation_specs()}

    assert set(specs) == {
        "ome_tiff_hcs_companion",
        "cellomics_bbbc001_a03",
        "cellomics_bbbc017_nirhta001_a01",
        "cellomics_bbbc017_nirhta002_a01",
        "cellomics_bbbc017_nirhta003_a01",
        "cellomics_bbbc017_nirhtaplus001_a01",
        "cellomics_bbbc017_nirhtaplus002_a01",
        "incell2000_59223_a01_site1",
        "incell2000_59224_a01_site1",
        "incell2000_zenodo14777242_c05_site1_zstack",
        "incell3000_bbbc013_first3",
        "cv7000_cpg0016_a01_subset",
        "olympus_scanr_idr0009_w00002_subset",
        "columbus_zenodo6327496_tif",
        "columbus_zenodo6327496_flex",
        "operetta_zenodo7841360_single",
        "operetta_omer_r01c02_f01_p01",
        "metaxpress_idr0081_a01",
        "metaxpress_idr0008_a01_site1",
        "metaxpress_idr0006_a01_site10",
        "incell2000_zenodo14769820_ok_a01_site1",
        "incell2000_zenodo14769820_fail_b02_site1",
        "cv7000_idr0088_b02_two_fields",
        "cv7000_idr0093_b02_site1_three_channels",
        "cv7000_idr0093_b03_site1_three_channels",
    }
    assert DATASET_REGISTRY["ome_tiff_hcs_companion"].tags == frozenset(
        {BenchmarkDatasetTag.BIOFORMATS_HCS_VALIDATION}
    )
    for spec in specs.values():
        assert spec.microscope_type == "bioformats"
        assert BenchmarkDatasetTag.BIOFORMATS_HCS_VALIDATION in spec.tags
        assert spec.acquisition_source().kind is DatasetSourceKind.URL_FILES
        assert spec.acquisition_source().urls


def test_validate_acquired_bioformats_hcs_dataset_reports_projection_metrics(
    tmp_path: Path,
) -> None:
    write_bioformats_manifest_fixture(tmp_path)
    spec = DatasetSpec(
        id="fixture",
        urls=[],
        size_bytes=128,
        archive_format=ArchiveFormat.ZIP,
        microscope_type="bioformats",
        validation_rule=DatasetValidationRule.NON_EMPTY,
        source=DatasetSourceSpec(kind=DatasetSourceKind.URL_FILES),
    )
    acquired = AcquiredDataset(
        id="fixture",
        path=tmp_path,
        microscope_type="bioformats",
        image_count=1,
        metadata={"cached": True},
    )

    result = validate_acquired_bioformats_hcs_dataset(
        spec,
        acquired,
        load_sample_count=1,
    )

    assert result.status == "passed"
    assert result.cached is True
    assert result.virtual_file_count == 2
    assert result.well_count == 1
    assert result.site_count == 1
    assert result.channel_count == 2
    assert result.z_count == 1
    assert result.timepoint_count == 1
    assert result.grid_dimensions == (1, 1)
    assert result.wells == ("A01",)
    assert result.sites == ("1",)
    assert result.channels == ("1", "2")
    assert result.z_indexes == ("1",)
    assert result.timepoints == ("1",)
    assert result.loaded_plane_count == 1
    assert result.load_shapes == ("3x4",)
    assert result.load_dtypes == ("uint16",)
