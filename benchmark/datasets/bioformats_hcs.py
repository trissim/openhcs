"""Public Bio-Formats HCS sample dataset declarations."""

from __future__ import annotations

from typing import ClassVar

from openhcs.constants.constants import Microscope

from benchmark.contracts.dataset import (
    BenchmarkDatasetTag,
    DatasetSourceKind,
    DatasetSourceSpec,
)
from benchmark.datasets.registry import (
    BenchmarkDatasetDeclaration,
    ImageCountValidatedDatasetMixin,
)


class BioFormatsHcsValidationDatasetMixin(ImageCountValidatedDatasetMixin):
    """Dataset declaration mixin for public Bio-Formats HCS validation samples."""

    microscope_type: ClassVar[str] = Microscope.BIOFORMATS.value
    tags: ClassVar[frozenset[BenchmarkDatasetTag]] = frozenset(
        {BenchmarkDatasetTag.BIOFORMATS_HCS_VALIDATION}
    )


def _url_files(source_page: str, files: tuple[str, ...]) -> DatasetSourceSpec:
    """Build a URL-files source from a public Bio-Formats directory."""
    return DatasetSourceSpec(
        kind=DatasetSourceKind.URL_FILES,
        urls=tuple(f"{source_page}{file_name}" for file_name in files),
    )


class OmeTiffHcsCompanionDataset(
    BioFormatsHcsValidationDatasetMixin, BenchmarkDatasetDeclaration
):
    """OME-TIFF HCS companion fileset."""

    id = "ome_tiff_hcs_companion"
    public_alias = "OME_TIFF_HCS_COMPANION"
    size_bytes = 64000
    expected_count = 5
    source = _url_files(
        "https://downloads.openmicroscopy.org/images/OME-TIFF/2016-06/plate-companion/",
        (
            "hcs.companion.ome",
            "well-A2.ome.tiff",
            "well-B1.ome.tiff",
            "well-B3.ome.tiff",
            "well-C2.ome.tiff",
            "well-C2-2.ome.tiff",
        ),
    )


class CellomicsBbbc001A03Dataset(
    BioFormatsHcsValidationDatasetMixin, BenchmarkDatasetDeclaration
):
    """Cellomics BBBC001 A03."""

    id = "cellomics_bbbc001_a03"
    size_bytes = 3146040
    expected_count = 6
    source = _url_files(
        "https://downloads.openmicroscopy.org/images/Cellomics/BBBC001/",
        (
            "AS_09125_050118150001_A03f00d0.DIB",
            "AS_09125_050118150001_A03f01d0.DIB",
            "AS_09125_050118150001_A03f02d0.DIB",
            "AS_09125_050118150001_A03f03d0.DIB",
            "AS_09125_050118150001_A03f04d0.DIB",
            "AS_09125_050118150001_A03f05d0.DIB",
        ),
    )


class CellomicsBbbc017Nirhta001A01Dataset(
    BioFormatsHcsValidationDatasetMixin, BenchmarkDatasetDeclaration
):
    """Cellomics BBBC017 NIRHTa-001 A01."""

    id = "cellomics_bbbc017_nirhta001_a01"
    size_bytes = 9438120
    expected_count = 18
    source = _url_files(
        "https://downloads.openmicroscopy.org/images/Cellomics/BBBC017/NIRHTa-001/",
        (
            "AS_09125_050116110001_A01f00d0.DIB",
            "AS_09125_050116110001_A01f00d1.DIB",
            "AS_09125_050116110001_A01f00d2.DIB",
            "AS_09125_050116110001_A01f01d0.DIB",
            "AS_09125_050116110001_A01f01d1.DIB",
            "AS_09125_050116110001_A01f01d2.DIB",
            "AS_09125_050116110001_A01f02d0.DIB",
            "AS_09125_050116110001_A01f02d1.DIB",
            "AS_09125_050116110001_A01f02d2.DIB",
            "AS_09125_050116110001_A01f03d0.DIB",
            "AS_09125_050116110001_A01f03d1.DIB",
            "AS_09125_050116110001_A01f03d2.DIB",
            "AS_09125_050116110001_A01f04d0.DIB",
            "AS_09125_050116110001_A01f04d1.DIB",
            "AS_09125_050116110001_A01f04d2.DIB",
            "AS_09125_050116110001_A01f05d0.DIB",
            "AS_09125_050116110001_A01f05d1.DIB",
            "AS_09125_050116110001_A01f05d2.DIB",
        ),
    )


class CellomicsBbbc017Nirhta002A01Dataset(
    BioFormatsHcsValidationDatasetMixin, BenchmarkDatasetDeclaration
):
    """Cellomics BBBC017 NIRHTa-002 A01."""

    id = "cellomics_bbbc017_nirhta002_a01"
    size_bytes = 9438120
    expected_count = 18
    source = _url_files(
        "https://downloads.openmicroscopy.org/images/HCS/BBBC/BBBC017/NIRHTa-002/",
        (
            "AS_09125_050115070001_A01f00d0.DIB",
            "AS_09125_050115070001_A01f00d1.DIB",
            "AS_09125_050115070001_A01f00d2.DIB",
            "AS_09125_050115070001_A01f01d0.DIB",
            "AS_09125_050115070001_A01f01d1.DIB",
            "AS_09125_050115070001_A01f01d2.DIB",
            "AS_09125_050115070001_A01f02d0.DIB",
            "AS_09125_050115070001_A01f02d1.DIB",
            "AS_09125_050115070001_A01f02d2.DIB",
            "AS_09125_050115070001_A01f03d0.DIB",
            "AS_09125_050115070001_A01f03d1.DIB",
            "AS_09125_050115070001_A01f03d2.DIB",
            "AS_09125_050115070001_A01f04d0.DIB",
            "AS_09125_050115070001_A01f04d1.DIB",
            "AS_09125_050115070001_A01f04d2.DIB",
            "AS_09125_050115070001_A01f05d0.DIB",
            "AS_09125_050115070001_A01f05d1.DIB",
            "AS_09125_050115070001_A01f05d2.DIB",
        ),
    )


class CellomicsBbbc017Nirhta003A01Dataset(
    BioFormatsHcsValidationDatasetMixin, BenchmarkDatasetDeclaration
):
    """Cellomics BBBC017 NIRHTa-003 A01."""

    id = "cellomics_bbbc017_nirhta003_a01"
    size_bytes = 9438120
    expected_count = 18
    source = _url_files(
        "https://downloads.openmicroscopy.org/images/HCS/BBBC/BBBC017/NIRHTa-003/",
        (
            "AS_09125_050116130001_A01f00d0.DIB",
            "AS_09125_050116130001_A01f00d1.DIB",
            "AS_09125_050116130001_A01f00d2.DIB",
            "AS_09125_050116130001_A01f01d0.DIB",
            "AS_09125_050116130001_A01f01d1.DIB",
            "AS_09125_050116130001_A01f01d2.DIB",
            "AS_09125_050116130001_A01f02d0.DIB",
            "AS_09125_050116130001_A01f02d1.DIB",
            "AS_09125_050116130001_A01f02d2.DIB",
            "AS_09125_050116130001_A01f03d0.DIB",
            "AS_09125_050116130001_A01f03d1.DIB",
            "AS_09125_050116130001_A01f03d2.DIB",
            "AS_09125_050116130001_A01f04d0.DIB",
            "AS_09125_050116130001_A01f04d1.DIB",
            "AS_09125_050116130001_A01f04d2.DIB",
            "AS_09125_050116130001_A01f05d0.DIB",
            "AS_09125_050116130001_A01f05d1.DIB",
            "AS_09125_050116130001_A01f05d2.DIB",
        ),
    )


class CellomicsBbbc017Nirhtaplus001A01Dataset(
    BioFormatsHcsValidationDatasetMixin, BenchmarkDatasetDeclaration
):
    """Cellomics BBBC017 NIRHTa+001 A01."""

    id = "cellomics_bbbc017_nirhtaplus001_a01"
    size_bytes = 9438120
    expected_count = 18
    source = _url_files(
        "https://downloads.openmicroscopy.org/images/HCS/BBBC/BBBC017/NIRHTa%2B001/",
        (
            "AS_09125_050116000001_A01f00d0.DIB",
            "AS_09125_050116000001_A01f00d1.DIB",
            "AS_09125_050116000001_A01f00d2.DIB",
            "AS_09125_050116000001_A01f01d0.DIB",
            "AS_09125_050116000001_A01f01d1.DIB",
            "AS_09125_050116000001_A01f01d2.DIB",
            "AS_09125_050116000001_A01f02d0.DIB",
            "AS_09125_050116000001_A01f02d1.DIB",
            "AS_09125_050116000001_A01f02d2.DIB",
            "AS_09125_050116000001_A01f03d0.DIB",
            "AS_09125_050116000001_A01f03d1.DIB",
            "AS_09125_050116000001_A01f03d2.DIB",
            "AS_09125_050116000001_A01f04d0.DIB",
            "AS_09125_050116000001_A01f04d1.DIB",
            "AS_09125_050116000001_A01f04d2.DIB",
            "AS_09125_050116000001_A01f05d0.DIB",
            "AS_09125_050116000001_A01f05d1.DIB",
            "AS_09125_050116000001_A01f05d2.DIB",
        ),
    )


class CellomicsBbbc017Nirhtaplus002A01Dataset(
    BioFormatsHcsValidationDatasetMixin, BenchmarkDatasetDeclaration
):
    """Cellomics BBBC017 NIRHTa+002 A01."""

    id = "cellomics_bbbc017_nirhtaplus002_a01"
    size_bytes = 9438120
    expected_count = 18
    source = _url_files(
        "https://downloads.openmicroscopy.org/images/HCS/BBBC/BBBC017/NIRHTa%2B002/",
        (
            "AS_09125_050117080001_A01f00d0.DIB",
            "AS_09125_050117080001_A01f00d1.DIB",
            "AS_09125_050117080001_A01f00d2.DIB",
            "AS_09125_050117080001_A01f01d0.DIB",
            "AS_09125_050117080001_A01f01d1.DIB",
            "AS_09125_050117080001_A01f01d2.DIB",
            "AS_09125_050117080001_A01f02d0.DIB",
            "AS_09125_050117080001_A01f02d1.DIB",
            "AS_09125_050117080001_A01f02d2.DIB",
            "AS_09125_050117080001_A01f03d0.DIB",
            "AS_09125_050117080001_A01f03d1.DIB",
            "AS_09125_050117080001_A01f03d2.DIB",
            "AS_09125_050117080001_A01f04d0.DIB",
            "AS_09125_050117080001_A01f04d1.DIB",
            "AS_09125_050117080001_A01f04d2.DIB",
            "AS_09125_050117080001_A01f05d0.DIB",
            "AS_09125_050117080001_A01f05d1.DIB",
            "AS_09125_050117080001_A01f05d2.DIB",
        ),
    )


class Incell200059223A01Site1Dataset(
    BioFormatsHcsValidationDatasetMixin, BenchmarkDatasetDeclaration
):
    """InCell 2000 INMAC384 59223 A01 site 1."""

    id = "incell2000_59223_a01_site1"
    size_bytes = 25167226
    expected_count = 3
    source = _url_files(
        "https://downloads.openmicroscopy.org/images/HCS/INCELL2000/INMAC384-DAPI-CM-eGFP_59223_1/",
        (
            "A%20-%201%28fld%201%20wv%20Cy5%20-%20Cy5%29.tif",
            "A%20-%201%28fld%201%20wv%20DAPI%20-%20DAPI%29.tif",
            "A%20-%201%28fld%201%20wv%20FITC%20-%20FITC%29.tif",
        ),
    )


class Incell200059224A01Site1Dataset(
    BioFormatsHcsValidationDatasetMixin, BenchmarkDatasetDeclaration
):
    """InCell 2000 INMAC384 59224 A01 site 1."""

    id = "incell2000_59224_a01_site1"
    size_bytes = 25167226
    expected_count = 3
    source = _url_files(
        "https://downloads.openmicroscopy.org/images/HCS/INCELL2000/INMAC384-DAPI-CM-eGFP_59224_1/",
        (
            "A%20-%201%28fld%201%20wv%20Cy5%20-%20Cy5%29.tif",
            "A%20-%201%28fld%201%20wv%20DAPI%20-%20DAPI%29.tif",
            "A%20-%201%28fld%201%20wv%20FITC%20-%20FITC%29.tif",
        ),
    )


class Incell2000Zenodo14777242C05Site1ZstackDataset(
    BioFormatsHcsValidationDatasetMixin, BenchmarkDatasetDeclaration
):
    """InCell 2000 zenodo-14777242 C05 site 1 Z-stack."""

    id = "incell2000_zenodo14777242_c05_site1_zstack"
    size_bytes = 39853630
    expected_count = 19
    source = _url_files(
        "https://downloads.openmicroscopy.org/images/HCS/INCELL2000/zenodo-14777242/",
        (
            "C%20-%2005%28fld%201%20wv%20Cy3%20-%20Cy3%29.tif",
            "C%20-%2005%28fld%201%20wv%20DAPI%20-%20DAPI%20z%2001%29.tif",
            "C%20-%2005%28fld%201%20wv%20DAPI%20-%20DAPI%20z%2002%29.tif",
            "C%20-%2005%28fld%201%20wv%20DAPI%20-%20DAPI%20z%2003%29.tif",
            "C%20-%2005%28fld%201%20wv%20DAPI%20-%20DAPI%20z%2004%29.tif",
            "C%20-%2005%28fld%201%20wv%20DAPI%20-%20DAPI%20z%2005%29.tif",
            "C%20-%2005%28fld%201%20wv%20DAPI%20-%20DAPI%20z%2006%29.tif",
            "C%20-%2005%28fld%201%20wv%20DAPI%20-%20DAPI%20z%2007%29.tif",
            "C%20-%2005%28fld%201%20wv%20DAPI%20-%20DAPI%20z%2008%29.tif",
            "C%20-%2005%28fld%201%20wv%20DAPI%20-%20DAPI%20z%2009%29.tif",
            "C%20-%2005%28fld%201%20wv%20DAPI%20-%20DAPI%20z%2010%29.tif",
            "C%20-%2005%28fld%201%20wv%20DAPI%20-%20DAPI%20z%2011%29.tif",
            "C%20-%2005%28fld%201%20wv%20DAPI%20-%20DAPI%20z%2012%29.tif",
            "C%20-%2005%28fld%201%20wv%20DAPI%20-%20DAPI%20z%2013%29.tif",
            "C%20-%2005%28fld%201%20wv%20DAPI%20-%20DAPI%20z%2014%29.tif",
            "C%20-%2005%28fld%201%20wv%20DAPI%20-%20DAPI%20z%2015%29.tif",
            "C%20-%2005%28fld%201%20wv%20DAPI%20-%20DAPI%20z%2016%29.tif",
            "C%20-%2005%28fld%201%20wv%20FITC%20-%20FITC%29.tif",
            "C%20-%2005%28fld%201%20wv%20TL-Brightfield%20-%20Cy3%29.tif",
        ),
    )


class Incell3000Bbbc013First3Dataset(
    BioFormatsHcsValidationDatasetMixin, BenchmarkDatasetDeclaration
):
    """InCell 3000 BBBC013 first three BMP images."""

    id = "incell3000_bbbc013_first3"
    size_bytes = 1232034
    expected_count = 3
    source = _url_files(
        "https://downloads.openmicroscopy.org/images/InCell3000/BBBC013/BBBC013_v1_images_bmp/",
        (
            "Channel1-01-A-01.BMP",
            "Channel1-02-A-02.BMP",
            "Channel1-03-A-03.BMP",
        ),
    )


class Cv7000Cpg0016A01SubsetDataset(
    BioFormatsHcsValidationDatasetMixin, BenchmarkDatasetDeclaration
):
    """Yokogawa CV7000 cpg0016 A01 subset."""

    id = "cv7000_cpg0016_a01_subset"
    size_bytes = 6008832
    expected_count = 3
    source = _url_files(
        "https://downloads.openmicroscopy.org/images/CV7000/cpg0016/Dest21053D1-15214/",
        (
            "Dest210531-152149_A01_T0001F001L01A01Z01C01.tif",
            "Dest210531-152149_A01_T0001F001L01A01Z01C05.tif",
            "Dest210531-152149_A01_T0001F001L01A02Z01C02.tif",
        ),
    )


class OlympusScanrIdr0009W00002SubsetDataset(
    BioFormatsHcsValidationDatasetMixin, BenchmarkDatasetDeclaration
):
    """Olympus ScanR idr0009 W00002 subset."""

    id = "olympus_scanr_idr0009_w00002_subset"
    size_bytes = 8258634
    expected_count = 3
    source = _url_files(
        "https://downloads.openmicroscopy.org/images/ScanR/idr0009/0307-10--2007-05-30/data/",
        (
            "--W00002--P00001--Z00000--T00000--nucleus-dapi.tif",
            "--W00002--P00001--Z00000--T00000--pm-647.tif",
            "--W00002--P00001--Z00000--T00000--vsvg-cfp.tif",
        ),
    )


class ColumbusZenodo6327496TifDataset(
    BioFormatsHcsValidationDatasetMixin, BenchmarkDatasetDeclaration
):
    """PerkinElmer Columbus zenodo-6327496 TIFF."""

    id = "columbus_zenodo6327496_tif"
    size_bytes = 18638103
    expected_count = 2
    source = _url_files(
        "https://downloads.openmicroscopy.org/images/PerkinElmer-Columbus/zenodo-6327496/tif/",
        (
            "001001-1.tif",
            "002001-1.tif",
            "ImageIndex.ColumbusIDX.csv",
            "ImageIndex.ColumbusIDX.xml",
            "MeasurementIndex.ColumbusIDX.xml",
        ),
    )


class ColumbusZenodo6327496FlexDataset(
    BioFormatsHcsValidationDatasetMixin, BenchmarkDatasetDeclaration
):
    """PerkinElmer Columbus zenodo-6327496 FLEX."""

    id = "columbus_zenodo6327496_flex"
    size_bytes = 33465961
    expected_count = 2
    source = _url_files(
        "https://downloads.openmicroscopy.org/images/PerkinElmer-Columbus/zenodo-6327496/flex/",
        (
            "001001.flex",
            "002001.flex",
            "ImageIndex.ColumbusIDX.csv",
            "ImageIndex.ColumbusIDX.xml",
            "MeasurementIndex.ColumbusIDX.xml",
        ),
    )


class OperettaZenodo7841360SingleDataset(
    BioFormatsHcsValidationDatasetMixin, BenchmarkDatasetDeclaration
):
    """PerkinElmer Operetta zenodo-7841360 single plane."""

    id = "operetta_zenodo7841360_single"
    size_bytes = 2299104
    expected_count = 1
    source = _url_files(
        "https://downloads.openmicroscopy.org/images/PerkinElmer-Operetta/zenodo-7841360/Hoechst__2023-03-07T10_17_54-Measurement%202/Images/",
        ("r03c07f01p01-ch1sk1fk1fl1.tiff",),
    )


class OperettaOmerR01c02F01P01Dataset(
    BioFormatsHcsValidationDatasetMixin, BenchmarkDatasetDeclaration
):
    """PerkinElmer Operetta omer 006P_M3 r01c02 field 1."""

    id = "operetta_omer_r01c02_f01_p01"
    size_bytes = 4605888
    expected_count = 3
    source = _url_files(
        "https://downloads.openmicroscopy.org/images/PerkinElmer-Operetta/omer/006P_M3/006P__2017-08-19T12_42_59-Measurement%203/Images/",
        (
            "r01c02f01p01-ch1sk1fk1fl1.tiff",
            "r01c02f01p01-ch2sk1fk1fl1.tiff",
            "r01c02f01p01-ch3sk1fk1fl1.tiff",
        ),
    )


class MetaxpressIdr0081A01Dataset(
    BioFormatsHcsValidationDatasetMixin, BenchmarkDatasetDeclaration
):
    """MetaXpress idr0081 BSF018292-1A A01."""

    id = "metaxpress_idr0081_a01"
    size_bytes = 16799169
    expected_count = 2
    source = _url_files(
        "https://downloads.openmicroscopy.org/images/MetaXpress/idr0081/BSF018292-1A/",
        (
            "BSF018292-1A_A01_w1.TIF",
            "BSF018292-1A_A01_w2.TIF",
        ),
    )


class MetaxpressIdr0008A01Site1Dataset(
    BioFormatsHcsValidationDatasetMixin, BenchmarkDatasetDeclaration
):
    """MetaXpress idr0008 Plate10 A01 site 1."""

    id = "metaxpress_idr0008_a01_site1"
    size_bytes = 2178231
    expected_count = 3
    source = _url_files(
        "https://downloads.openmicroscopy.org/images/MetaXpress/idr0008/Plate10_Actinome1/",
        (
            "Act1_Plate10-SP-A8_A01_s1_w1.TIF",
            "Act1_Plate10-SP-A8_A01_s1_w2.TIF",
            "Act1_Plate10-SP-A8_A01_s1_w3.TIF",
        ),
    )


class MetaxpressIdr0006A01Site10Dataset(
    BioFormatsHcsValidationDatasetMixin, BenchmarkDatasetDeclaration
):
    """MetaXpress idr0006 plate 11001 A01 site 10."""

    id = "metaxpress_idr0006_a01_site10"
    size_bytes = 5802138
    expected_count = 2
    source = _url_files(
        "https://downloads.openmicroscopy.org/images/MetaXpress/idr0006/plate%2011001_Plate_136/TimePoint_1/",
        (
            "plate%2011001_A01_s10_w1.TIF",
            "plate%2011001_A01_s10_w2.TIF",
        ),
    )


class Incell2000Zenodo14769820OkA01Site1Dataset(
    BioFormatsHcsValidationDatasetMixin, BenchmarkDatasetDeclaration
):
    """InCell 2000 zenodo-14769820 Dataset_ok A01 site 1."""

    id = "incell2000_zenodo14769820_ok_a01_site1"
    size_bytes = 8390536
    expected_count = 4
    source = _url_files(
        "https://downloads.openmicroscopy.org/images/InCell2000/zenodo-14769820/Dataset_ok/",
        (
            "A%20-%2001%28fld%201%20wv%20Cy3%20-%20Cy3%29.tif",
            "A%20-%2001%28fld%201%20wv%20DAPI%20-%20DAPI%29.tif",
            "A%20-%2001%28fld%201%20wv%20FITC%20-%20FITC%29.tif",
            "A%20-%2001%28fld%201%20wv%20TL-Brightfield%20-%20Cy3%29.tif",
        ),
    )


class Incell2000Zenodo14769820FailB02Site1Dataset(
    BioFormatsHcsValidationDatasetMixin, BenchmarkDatasetDeclaration
):
    """InCell 2000 zenodo-14769820 Dataset_Fail B02 site 1."""

    id = "incell2000_zenodo14769820_fail_b02_site1"
    size_bytes = 10488106
    expected_count = 5
    source = _url_files(
        "https://downloads.openmicroscopy.org/images/InCell2000/zenodo-14769820/Dataset_Fail/",
        (
            "B%20-%2002%28fld%201%20wv%20Cy3%20-%20Cy3%29.tif",
            "B%20-%2002%28fld%201%20wv%20Cy5%20-%20Cy5%20wix%204%29.tif",
            "B%20-%2002%28fld%201%20wv%20Cy5%20-%20Cy5%20wix%205%29.tif",
            "B%20-%2002%28fld%201%20wv%20DAPI%20-%20DAPI%29.tif",
            "B%20-%2002%28fld%201%20wv%20FITC%20-%20FITC%29.tif",
        ),
    )


class Cv7000Idr0088B02TwoFieldsDataset(
    BioFormatsHcsValidationDatasetMixin, BenchmarkDatasetDeclaration
):
    """Yokogawa CV7000 idr0088 B02 two fields."""

    id = "cv7000_idr0088_b02_two_fields"
    size_bytes = 15763740
    expected_count = 6
    source = _url_files(
        "https://downloads.openmicroscopy.org/images/CV7000/idr0088/110000251230/",
        (
            "110000251230_B02_T0001F001L01A01Z01C01.tif",
            "110000251230_B02_T0001F001L01A02Z01C02.tif",
            "110000251230_B02_T0001F001L01A03Z01C03.tif",
            "110000251230_B02_T0001F002L01A01Z01C01.tif",
            "110000251230_B02_T0001F002L01A02Z01C02.tif",
            "110000251230_B02_T0001F002L01A03Z01C03.tif",
        ),
    )


class Cv7000Idr0093B02Site1ThreeChannelsDataset(
    BioFormatsHcsValidationDatasetMixin, BenchmarkDatasetDeclaration
):
    """Yokogawa CV7000 idr0093 B02 site 1 three channels."""

    id = "cv7000_idr0093_b02_site1_three_channels"
    size_bytes = 33230358
    expected_count = 3
    source = _url_files(
        "https://downloads.openmicroscopy.org/images/CV7000/idr0093/020/",
        (
            "020_B02_T0001F001L01A01Z01C01.tif",
            "020_B02_T0001F001L01A01Z01C03.tif",
            "020_B02_T0001F001L01A01Z01C05.tif",
        ),
    )


class Cv7000Idr0093B03Site1ThreeChannelsDataset(
    BioFormatsHcsValidationDatasetMixin, BenchmarkDatasetDeclaration
):
    """Yokogawa CV7000 idr0093 B03 site 1 three channels."""

    id = "cv7000_idr0093_b03_site1_three_channels"
    size_bytes = 33230358
    expected_count = 3
    source = _url_files(
        "https://downloads.openmicroscopy.org/images/CV7000/idr0093/020/",
        (
            "020_B03_T0001F001L01A01Z01C01.tif",
            "020_B03_T0001F001L01A01Z01C03.tif",
            "020_B03_T0001F001L01A01Z01C05.tif",
        ),
    )
