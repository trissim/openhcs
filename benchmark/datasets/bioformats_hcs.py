"""Public Bio-Formats HCS sample datasets for validation evidence."""

from __future__ import annotations

from dataclasses import dataclass

from benchmark.contracts.dataset import (
    ArchiveFormat,
    DatasetSourceKind,
    DatasetSourceSpec,
    DatasetSpec,
    DatasetValidationRule,
)


OME_DOWNLOADS_ROOT = "https://downloads.openmicroscopy.org/images"


@dataclass(frozen=True)
class BioFormatsHcsAxisExpectation:
    """Expected OpenHCS source-axis keys for one declared Bio-Formats subset."""

    wells: tuple[str, ...]
    sites: tuple[str, ...]
    channels: tuple[str, ...]
    z_indexes: tuple[str, ...]
    timepoints: tuple[str, ...] = ("1",)


def _axes(
    *,
    wells: tuple[str, ...],
    sites: tuple[int | str, ...],
    channels: tuple[int | str, ...],
    z_indexes: tuple[int | str, ...] = (1,),
    timepoints: tuple[int | str, ...] = (1,),
) -> BioFormatsHcsAxisExpectation:
    return BioFormatsHcsAxisExpectation(
        wells=wells,
        sites=tuple(str(value) for value in sites),
        channels=tuple(str(value) for value in channels),
        z_indexes=tuple(str(value) for value in z_indexes),
        timepoints=tuple(str(value) for value in timepoints),
    )


@dataclass(frozen=True)
class BioFormatsHcsDatasetLabel:
    """Shared display metadata for a public Bio-Formats HCS dataset."""

    display_name: str
    vendor: str
    format_name: str
    source_page: str
    notes: str


@dataclass(frozen=True)
class BioFormatsHcsCatalogRow(BioFormatsHcsDatasetLabel):
    """One public HCS dataset small enough for routine Bio-Formats validation."""

    spec: DatasetSpec
    axes: BioFormatsHcsAxisExpectation


@dataclass(frozen=True)
class BioFormatsHcsDatasetDeclaration(BioFormatsHcsDatasetLabel):
    """Authoritative declaration for one catalog-backed Bio-Formats HCS subset."""

    dataset_id: str
    files: tuple[str, ...]
    size_bytes: int
    expected_count: int
    axes: BioFormatsHcsAxisExpectation

    def catalog_row(self) -> BioFormatsHcsCatalogRow:
        return BioFormatsHcsCatalogRow(
            display_name=self.display_name,
            vendor=self.vendor,
            format_name=self.format_name,
            source_page=self.source_page,
            notes=self.notes,
            axes=self.axes,
            spec=DatasetSpec(
                id=self.dataset_id,
                urls=[],
                size_bytes=self.size_bytes,
                archive_format=ArchiveFormat.ZIP,
                microscope_type="bioformats",
                validation_rule=DatasetValidationRule.IMAGE_COUNT,
                expected_count=self.expected_count,
                source=DatasetSourceSpec(
                    kind=DatasetSourceKind.URL_FILES,
                    urls=tuple(f"{self.source_page}{file_name}" for file_name in self.files),
                ),
            ),
        )


def _cellomics_bbbc017_files(prefix: str) -> tuple[str, ...]:
    return tuple(
        f"{prefix}f{site:02d}d{channel}.DIB"
        for site in range(6)
        for channel in range(3)
    )


BIOFORMATS_HCS_DECLARATIONS: tuple[BioFormatsHcsDatasetDeclaration, ...] = (
    BioFormatsHcsDatasetDeclaration(
        dataset_id="ome_tiff_hcs_companion",
        source_page=f"{OME_DOWNLOADS_ROOT}/OME-TIFF/2016-06/plate-companion/",
        files=(
            "hcs.companion.ome",
            "well-A2.ome.tiff",
            "well-B1.ome.tiff",
            "well-B3.ome.tiff",
            "well-C2.ome.tiff",
            "well-C2-2.ome.tiff",
        ),
        size_bytes=64_000,
        expected_count=5,
        axes=_axes(
            wells=("A02", "B01", "B03", "C02"),
            sites=(1, 2),
            channels=(1,),
        ),
        display_name="OME-TIFF HCS companion fileset",
        vendor="Open Microscopy Environment",
        format_name="OME-TIFF companion OME-XML",
        notes="Small OME-SPW positive-control plate companion fileset.",
    ),
    BioFormatsHcsDatasetDeclaration(
        dataset_id="cellomics_bbbc001_a03",
        source_page=f"{OME_DOWNLOADS_ROOT}/Cellomics/BBBC001/",
        files=tuple(
            f"AS_09125_050118150001_A03f{site:02d}d0.DIB"
            for site in range(6)
        ),
        size_bytes=3_146_040,
        expected_count=6,
        axes=_axes(wells=("A03",), sites=tuple(range(1, 7)), channels=(1,)),
        display_name="Cellomics BBBC001 A03",
        vendor="Thermo/Cellomics",
        format_name="Cellomics DIB",
        notes="Single-well six-site subset from the Bio-Formats Cellomics sample.",
    ),
    BioFormatsHcsDatasetDeclaration(
        dataset_id="cellomics_bbbc017_nirhta001_a01",
        source_page=f"{OME_DOWNLOADS_ROOT}/Cellomics/BBBC017/NIRHTa-001/",
        files=_cellomics_bbbc017_files("AS_09125_050116110001_A01"),
        size_bytes=9_438_120,
        expected_count=18,
        axes=_axes(wells=("A01",), sites=tuple(range(1, 7)), channels=(1, 2, 3)),
        display_name="Cellomics BBBC017 NIRHTa-001 A01",
        vendor="Thermo/Cellomics",
        format_name="Cellomics DIB",
        notes="Single-well six-site, three-channel subset from BBBC017.",
    ),
    BioFormatsHcsDatasetDeclaration(
        dataset_id="cellomics_bbbc017_nirhta002_a01",
        source_page=f"{OME_DOWNLOADS_ROOT}/HCS/BBBC/BBBC017/NIRHTa-002/",
        files=_cellomics_bbbc017_files("AS_09125_050115070001_A01"),
        size_bytes=9_438_120,
        expected_count=18,
        axes=_axes(wells=("A01",), sites=tuple(range(1, 7)), channels=(1, 2, 3)),
        display_name="Cellomics BBBC017 NIRHTa-002 A01",
        vendor="Thermo/Cellomics",
        format_name="Cellomics DIB",
        notes="Second single-well six-site, three-channel BBBC017 subset.",
    ),
    BioFormatsHcsDatasetDeclaration(
        dataset_id="cellomics_bbbc017_nirhta003_a01",
        source_page=f"{OME_DOWNLOADS_ROOT}/HCS/BBBC/BBBC017/NIRHTa-003/",
        files=_cellomics_bbbc017_files("AS_09125_050116130001_A01"),
        size_bytes=9_438_120,
        expected_count=18,
        axes=_axes(wells=("A01",), sites=tuple(range(1, 7)), channels=(1, 2, 3)),
        display_name="Cellomics BBBC017 NIRHTa-003 A01",
        vendor="Thermo/Cellomics",
        format_name="Cellomics DIB",
        notes="Third single-well six-site, three-channel BBBC017 subset.",
    ),
    BioFormatsHcsDatasetDeclaration(
        dataset_id="cellomics_bbbc017_nirhtaplus001_a01",
        source_page=f"{OME_DOWNLOADS_ROOT}/HCS/BBBC/BBBC017/NIRHTa%2B001/",
        files=_cellomics_bbbc017_files("AS_09125_050116000001_A01"),
        size_bytes=9_438_120,
        expected_count=18,
        axes=_axes(wells=("A01",), sites=tuple(range(1, 7)), channels=(1, 2, 3)),
        display_name="Cellomics BBBC017 NIRHTa+001 A01",
        vendor="Thermo/Cellomics",
        format_name="Cellomics DIB",
        notes="Positive-control single-well six-site, three-channel BBBC017 subset.",
    ),
    BioFormatsHcsDatasetDeclaration(
        dataset_id="cellomics_bbbc017_nirhtaplus002_a01",
        source_page=f"{OME_DOWNLOADS_ROOT}/HCS/BBBC/BBBC017/NIRHTa%2B002/",
        files=_cellomics_bbbc017_files("AS_09125_050117080001_A01"),
        size_bytes=9_438_120,
        expected_count=18,
        axes=_axes(wells=("A01",), sites=tuple(range(1, 7)), channels=(1, 2, 3)),
        display_name="Cellomics BBBC017 NIRHTa+002 A01",
        vendor="Thermo/Cellomics",
        format_name="Cellomics DIB",
        notes="Second positive-control single-well six-site, three-channel BBBC017 subset.",
    ),
    BioFormatsHcsDatasetDeclaration(
        dataset_id="incell2000_59223_a01_site1",
        source_page=f"{OME_DOWNLOADS_ROOT}/HCS/INCELL2000/INMAC384-DAPI-CM-eGFP_59223_1/",
        files=(
            "A%20-%201%28fld%201%20wv%20Cy5%20-%20Cy5%29.tif",
            "A%20-%201%28fld%201%20wv%20DAPI%20-%20DAPI%29.tif",
            "A%20-%201%28fld%201%20wv%20FITC%20-%20FITC%29.tif",
        ),
        size_bytes=25_167_226,
        expected_count=3,
        axes=_axes(wells=("A01",), sites=(1,), channels=(1, 2, 3)),
        display_name="InCell 2000 INMAC384 59223 A01 site 1",
        vendor="GE/Cytiva InCell",
        format_name="InCell 2000 TIFF filename layout",
        notes="Single well/site, three-channel subset with HCS axes encoded in filenames.",
    ),
    BioFormatsHcsDatasetDeclaration(
        dataset_id="incell2000_59224_a01_site1",
        source_page=f"{OME_DOWNLOADS_ROOT}/HCS/INCELL2000/INMAC384-DAPI-CM-eGFP_59224_1/",
        files=(
            "A%20-%201%28fld%201%20wv%20Cy5%20-%20Cy5%29.tif",
            "A%20-%201%28fld%201%20wv%20DAPI%20-%20DAPI%29.tif",
            "A%20-%201%28fld%201%20wv%20FITC%20-%20FITC%29.tif",
        ),
        size_bytes=25_167_226,
        expected_count=3,
        axes=_axes(wells=("A01",), sites=(1,), channels=(1, 2, 3)),
        display_name="InCell 2000 INMAC384 59224 A01 site 1",
        vendor="GE/Cytiva InCell",
        format_name="InCell 2000 TIFF filename layout",
        notes="Second single well/site, three-channel InCell 2000 subset.",
    ),
    BioFormatsHcsDatasetDeclaration(
        dataset_id="incell2000_zenodo14777242_c05_site1_zstack",
        source_page=f"{OME_DOWNLOADS_ROOT}/HCS/INCELL2000/zenodo-14777242/",
        files=(
            "C%20-%2005%28fld%201%20wv%20Cy3%20-%20Cy3%29.tif",
            *tuple(
                f"C%20-%2005%28fld%201%20wv%20DAPI%20-%20DAPI%20z%20{z_index:02d}%29.tif"
                for z_index in range(1, 17)
            ),
            "C%20-%2005%28fld%201%20wv%20FITC%20-%20FITC%29.tif",
            "C%20-%2005%28fld%201%20wv%20TL-Brightfield%20-%20Cy3%29.tif",
        ),
        size_bytes=39_853_630,
        expected_count=19,
        axes=_axes(
            wells=("C05",),
            sites=(1,),
            channels=(1, 2, 3, 4),
            z_indexes=tuple(range(1, 17)),
        ),
        display_name="InCell 2000 zenodo-14777242 C05 site 1 Z-stack",
        vendor="GE/Cytiva InCell",
        format_name="InCell 2000 TIFF filename layout",
        notes="Single well/site subset spanning three channels plus a 16-plane DAPI Z-stack.",
    ),
    BioFormatsHcsDatasetDeclaration(
        dataset_id="incell3000_bbbc013_first3",
        source_page=f"{OME_DOWNLOADS_ROOT}/InCell3000/BBBC013/BBBC013_v1_images_bmp/",
        files=(
            "Channel1-01-A-01.BMP",
            "Channel1-02-A-02.BMP",
            "Channel1-03-A-03.BMP",
        ),
        size_bytes=1_232_034,
        expected_count=3,
        axes=_axes(wells=("A01", "A02", "A03"), sites=(1, 2, 3), channels=(1,)),
        display_name="InCell 3000 BBBC013 first three BMP images",
        vendor="GE/Cytiva InCell",
        format_name="InCell 3000 BMP filename layout",
        notes="Three single-channel BMP images with HCS axes encoded in filenames.",
    ),
    BioFormatsHcsDatasetDeclaration(
        dataset_id="cv7000_cpg0016_a01_subset",
        source_page=f"{OME_DOWNLOADS_ROOT}/CV7000/cpg0016/Dest21053D1-15214/",
        files=(
            "Dest210531-152149_A01_T0001F001L01A01Z01C01.tif",
            "Dest210531-152149_A01_T0001F001L01A01Z01C05.tif",
            "Dest210531-152149_A01_T0001F001L01A02Z01C02.tif",
        ),
        size_bytes=6_008_832,
        expected_count=3,
        axes=_axes(wells=("A01",), sites=(1, 2), channels=(1, 2, 3)),
        display_name="Yokogawa CV7000 cpg0016 A01 subset",
        vendor="Yokogawa CV7000",
        format_name="CV7000 TIFF filename layout",
        notes="Single-well subset with CV7000 well, field, tile, Z, T, and C axes encoded in filenames.",
    ),
    BioFormatsHcsDatasetDeclaration(
        dataset_id="olympus_scanr_idr0009_w00002_subset",
        source_page=f"{OME_DOWNLOADS_ROOT}/ScanR/idr0009/0307-10--2007-05-30/data/",
        files=(
            "--W00002--P00001--Z00000--T00000--nucleus-dapi.tif",
            "--W00002--P00001--Z00000--T00000--pm-647.tif",
            "--W00002--P00001--Z00000--T00000--vsvg-cfp.tif",
        ),
        size_bytes=8_258_634,
        expected_count=3,
        axes=_axes(wells=("W00002",), sites=(1,), channels=(1, 2, 3)),
        display_name="Olympus ScanR idr0009 W00002 subset",
        vendor="Olympus ScanR",
        format_name="ScanR TIFF filename layout",
        notes="Single-well, single-position, three-channel ScanR subset with HCS axes encoded in filenames.",
    ),
    BioFormatsHcsDatasetDeclaration(
        dataset_id="columbus_zenodo6327496_tif",
        source_page=f"{OME_DOWNLOADS_ROOT}/PerkinElmer-Columbus/zenodo-6327496/tif/",
        files=(
            "001001-1.tif",
            "002001-1.tif",
            "ImageIndex.ColumbusIDX.csv",
            "ImageIndex.ColumbusIDX.xml",
            "MeasurementIndex.ColumbusIDX.xml",
        ),
        size_bytes=18_638_103,
        expected_count=2,
        axes=_axes(
            wells=("A01", "B01"),
            sites=(1,),
            channels=(1, 2),
            z_indexes=(1, 2, 3),
        ),
        display_name="PerkinElmer Columbus zenodo-6327496 TIFF",
        vendor="PerkinElmer/Revvity Columbus",
        format_name="Columbus indexed TIFF",
        notes="Two-well Columbus indexed TIFF sample with OME-SPW metadata from Bio-Formats.",
    ),
    BioFormatsHcsDatasetDeclaration(
        dataset_id="columbus_zenodo6327496_flex",
        source_page=f"{OME_DOWNLOADS_ROOT}/PerkinElmer-Columbus/zenodo-6327496/flex/",
        files=(
            "001001.flex",
            "002001.flex",
            "ImageIndex.ColumbusIDX.csv",
            "ImageIndex.ColumbusIDX.xml",
            "MeasurementIndex.ColumbusIDX.xml",
        ),
        size_bytes=33_465_961,
        expected_count=2,
        axes=_axes(
            wells=("A01", "B01"),
            sites=(1,),
            channels=(1, 2),
            z_indexes=(1, 2, 3),
        ),
        display_name="PerkinElmer Columbus zenodo-6327496 FLEX",
        vendor="PerkinElmer/Revvity Columbus",
        format_name="Columbus indexed FLEX",
        notes="Two-well Columbus indexed FLEX sample with OME-SPW metadata from Bio-Formats.",
    ),
    BioFormatsHcsDatasetDeclaration(
        dataset_id="operetta_zenodo7841360_single",
        source_page=f"{OME_DOWNLOADS_ROOT}/PerkinElmer-Operetta/zenodo-7841360/Hoechst__2023-03-07T10_17_54-Measurement%202/Images/",
        files=("r03c07f01p01-ch1sk1fk1fl1.tiff",),
        size_bytes=2_299_104,
        expected_count=1,
        axes=_axes(wells=("C07",), sites=(1,), channels=(1,)),
        display_name="PerkinElmer Operetta zenodo-7841360 single plane",
        vendor="PerkinElmer/Revvity Operetta",
        format_name="Operetta TIFF filename layout",
        notes="Single Operetta image with row, column, field, plane, and channel axes encoded in filename.",
    ),
    BioFormatsHcsDatasetDeclaration(
        dataset_id="operetta_omer_r01c02_f01_p01",
        source_page=f"{OME_DOWNLOADS_ROOT}/PerkinElmer-Operetta/omer/006P_M3/006P__2017-08-19T12_42_59-Measurement%203/Images/",
        files=(
            "r01c02f01p01-ch1sk1fk1fl1.tiff",
            "r01c02f01p01-ch2sk1fk1fl1.tiff",
            "r01c02f01p01-ch3sk1fk1fl1.tiff",
        ),
        size_bytes=4_605_888,
        expected_count=3,
        axes=_axes(wells=("A02",), sites=(1,), channels=(1, 2, 3)),
        display_name="PerkinElmer Operetta omer 006P_M3 r01c02 field 1",
        vendor="PerkinElmer/Revvity Operetta",
        format_name="Operetta TIFF filename layout",
        notes="Three-channel Operetta subset with row, column, field, plane, and channel axes encoded in filenames.",
    ),
    BioFormatsHcsDatasetDeclaration(
        dataset_id="metaxpress_idr0081_a01",
        source_page=f"{OME_DOWNLOADS_ROOT}/MetaXpress/idr0081/BSF018292-1A/",
        files=(
            "BSF018292-1A_A01_w1.TIF",
            "BSF018292-1A_A01_w2.TIF",
        ),
        size_bytes=16_799_169,
        expected_count=2,
        axes=_axes(wells=("A01",), sites=(1,), channels=(1, 2)),
        display_name="MetaXpress idr0081 BSF018292-1A A01",
        vendor="Molecular Devices MetaXpress",
        format_name="MetaXpress TIFF filename layout",
        notes="Single-well, two-channel MetaXpress subset with well and wavelength axes encoded in filenames.",
    ),
    BioFormatsHcsDatasetDeclaration(
        dataset_id="metaxpress_idr0008_a01_site1",
        source_page=f"{OME_DOWNLOADS_ROOT}/MetaXpress/idr0008/Plate10_Actinome1/",
        files=(
            "Act1_Plate10-SP-A8_A01_s1_w1.TIF",
            "Act1_Plate10-SP-A8_A01_s1_w2.TIF",
            "Act1_Plate10-SP-A8_A01_s1_w3.TIF",
        ),
        size_bytes=2_178_231,
        expected_count=3,
        axes=_axes(wells=("A01",), sites=(1,), channels=(1, 2, 3)),
        display_name="MetaXpress idr0008 Plate10 A01 site 1",
        vendor="Molecular Devices MetaXpress",
        format_name="MetaXpress TIFF filename layout",
        notes="Single-well/site, three-channel MetaXpress subset with well, site, and wavelength axes encoded in filenames.",
    ),
    BioFormatsHcsDatasetDeclaration(
        dataset_id="metaxpress_idr0006_a01_site10",
        source_page=f"{OME_DOWNLOADS_ROOT}/MetaXpress/idr0006/plate%2011001_Plate_136/TimePoint_1/",
        files=(
            "plate%2011001_A01_s10_w1.TIF",
            "plate%2011001_A01_s10_w2.TIF",
        ),
        size_bytes=5_802_138,
        expected_count=2,
        axes=_axes(wells=("A01",), sites=(10,), channels=(1, 2)),
        display_name="MetaXpress idr0006 plate 11001 A01 site 10",
        vendor="Molecular Devices MetaXpress",
        format_name="MetaXpress TIFF filename layout",
        notes="Single-well/site, two-channel MetaXpress subset with well, site, and wavelength axes encoded in filenames.",
    ),
    BioFormatsHcsDatasetDeclaration(
        dataset_id="incell2000_zenodo14769820_ok_a01_site1",
        source_page=f"{OME_DOWNLOADS_ROOT}/InCell2000/zenodo-14769820/Dataset_ok/",
        files=(
            "A%20-%2001%28fld%201%20wv%20Cy3%20-%20Cy3%29.tif",
            "A%20-%2001%28fld%201%20wv%20DAPI%20-%20DAPI%29.tif",
            "A%20-%2001%28fld%201%20wv%20FITC%20-%20FITC%29.tif",
            "A%20-%2001%28fld%201%20wv%20TL-Brightfield%20-%20Cy3%29.tif",
        ),
        size_bytes=8_390_536,
        expected_count=4,
        axes=_axes(wells=("A01",), sites=(1,), channels=(1, 2, 3, 4)),
        display_name="InCell 2000 zenodo-14769820 Dataset_ok A01 site 1",
        vendor="GE/Cytiva InCell",
        format_name="InCell 2000 TIFF filename layout",
        notes="Four-channel InCell 2000 subset from Dataset_ok with well, field, and wavelength axes encoded in filenames.",
    ),
    BioFormatsHcsDatasetDeclaration(
        dataset_id="incell2000_zenodo14769820_fail_b02_site1",
        source_page=f"{OME_DOWNLOADS_ROOT}/InCell2000/zenodo-14769820/Dataset_Fail/",
        files=(
            "B%20-%2002%28fld%201%20wv%20Cy3%20-%20Cy3%29.tif",
            "B%20-%2002%28fld%201%20wv%20Cy5%20-%20Cy5%20wix%204%29.tif",
            "B%20-%2002%28fld%201%20wv%20Cy5%20-%20Cy5%20wix%205%29.tif",
            "B%20-%2002%28fld%201%20wv%20DAPI%20-%20DAPI%29.tif",
            "B%20-%2002%28fld%201%20wv%20FITC%20-%20FITC%29.tif",
        ),
        size_bytes=10_488_106,
        expected_count=5,
        axes=_axes(wells=("B02",), sites=(1,), channels=(1, 2, 3, 4, 5)),
        display_name="InCell 2000 zenodo-14769820 Dataset_Fail B02 site 1",
        vendor="GE/Cytiva InCell",
        format_name="InCell 2000 TIFF filename layout",
        notes="Five-channel InCell 2000 subset preserving two distinct wavelength-indexed Cy5 channels.",
    ),
    BioFormatsHcsDatasetDeclaration(
        dataset_id="cv7000_idr0088_b02_two_fields",
        source_page=f"{OME_DOWNLOADS_ROOT}/CV7000/idr0088/110000251230/",
        files=(
            "110000251230_B02_T0001F001L01A01Z01C01.tif",
            "110000251230_B02_T0001F001L01A02Z01C02.tif",
            "110000251230_B02_T0001F001L01A03Z01C03.tif",
            "110000251230_B02_T0001F002L01A01Z01C01.tif",
            "110000251230_B02_T0001F002L01A02Z01C02.tif",
            "110000251230_B02_T0001F002L01A03Z01C03.tif",
        ),
        size_bytes=15_763_740,
        expected_count=6,
        axes=_axes(
            wells=("B02",),
            sites=(1, 2, 3, 1001, 1002, 1003),
            channels=(1, 2, 3),
        ),
        display_name="Yokogawa CV7000 idr0088 B02 two fields",
        vendor="Yokogawa CV7000",
        format_name="CV7000 TIFF filename layout",
        notes="Single-well CV7000 subset spanning two field IDs, three area/site IDs, and three channels.",
    ),
    BioFormatsHcsDatasetDeclaration(
        dataset_id="cv7000_idr0093_b02_site1_three_channels",
        source_page=f"{OME_DOWNLOADS_ROOT}/CV7000/idr0093/020/",
        files=(
            "020_B02_T0001F001L01A01Z01C01.tif",
            "020_B02_T0001F001L01A01Z01C03.tif",
            "020_B02_T0001F001L01A01Z01C05.tif",
        ),
        size_bytes=33_230_358,
        expected_count=3,
        axes=_axes(wells=("B02",), sites=(1,), channels=(1, 2, 3)),
        display_name="Yokogawa CV7000 idr0093 B02 site 1 three channels",
        vendor="Yokogawa CV7000",
        format_name="CV7000 TIFF filename layout",
        notes="Single-well CV7000 subset from idr0093 with three non-contiguous source channel IDs.",
    ),
    BioFormatsHcsDatasetDeclaration(
        dataset_id="cv7000_idr0093_b03_site1_three_channels",
        source_page=f"{OME_DOWNLOADS_ROOT}/CV7000/idr0093/020/",
        files=(
            "020_B03_T0001F001L01A01Z01C01.tif",
            "020_B03_T0001F001L01A01Z01C03.tif",
            "020_B03_T0001F001L01A01Z01C05.tif",
        ),
        size_bytes=33_230_358,
        expected_count=3,
        axes=_axes(wells=("B03",), sites=(1,), channels=(1, 2, 3)),
        display_name="Yokogawa CV7000 idr0093 B03 site 1 three channels",
        vendor="Yokogawa CV7000",
        format_name="CV7000 TIFF filename layout",
        notes="Second idr0093 CV7000 well subset with three non-contiguous source channel IDs.",
    ),
)


BIOFORMATS_HCS_CATALOG: tuple[BioFormatsHcsCatalogRow, ...] = tuple(
    declaration.catalog_row() for declaration in BIOFORMATS_HCS_DECLARATIONS
)

BIOFORMATS_HCS_REGISTRY: dict[str, BioFormatsHcsCatalogRow] = {
    row.spec.id: row for row in BIOFORMATS_HCS_CATALOG
}
