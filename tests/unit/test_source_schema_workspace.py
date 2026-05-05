from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from openhcs.constants import AllComponents
from openhcs.core.pipeline_image_schema import (
    GroupingPlan,
    ImageAssignment,
    ImagePlaneSource,
    ImagesRule,
    ImportedMetadataJoin,
    ImportedMetadataTable,
    PipelineImageSchema,
    SOURCE_IMAGE_TYPE_METADATA_FIELD,
)
from openhcs.core.source_bindings import (
    ComponentSelector,
    MetadataExtractionRule,
    MetadataSource,
    MetadataSelector,
    SourceBindingMatchDimension,
    SourceBindingMatchField,
    SourceBindingMatchMethod,
    SourceBindingMatchPlan,
    SourceBindingOrigin,
    SourceFilterClause,
    SourceFilterMatchType,
    SourceFilterSubject,
    SourceSelector,
)
from openhcs.core.source_schema_workspace import (
    SOURCE_SCHEMA_WORKSPACE_SOURCE_DIR,
    materialize_source_schema_workspace,
)
from openhcs.microscopes.source_schema import SourceSchemaFilenameParser


def test_source_schema_filename_parser_handles_artifact_suffixes() -> None:
    parser = SourceSchemaFilenameParser()

    parsed = parser.parse_filename(
        "source_s001_w1_z001_t001_CorrectIlluminationCalculate_7_measurements_step2.csv"
    )

    assert parsed == {
        "well": "source",
        "site": 1,
        "channel": 1,
        "z_index": 1,
        "timepoint": 1,
        "extension": ".csv",
    }


def test_materialize_source_schema_workspace_projects_cellprofiler_sources(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "cellprofiler_source"
    source_root.mkdir()
    _write_image(source_root / "Channel1-01-A-01.tif", value=1)
    _write_image(source_root / "Channel2-01-A-01.tif", value=2)
    (source_root / "Channel2ILLUM.mat").write_bytes(b"mat payload")
    workspace_root = tmp_path / "openhcs_workspace"

    result = materialize_source_schema_workspace(
        source_root,
        workspace_root,
        _example_sbs_source_schema(),
    )

    metadata = json.loads(result.metadata_path.read_text())
    primary = metadata["subdirectories"]["."]
    auxiliary = metadata["subdirectories"][SOURCE_SCHEMA_WORKSPACE_SOURCE_DIR]

    assert result.workspace_root == workspace_root
    assert set(primary["workspace_mapping"]) == {
        "A01_s001_w1_z001_t001.tif",
        "A01_s001_w2_z001_t001.tif",
    }
    assert primary["channels"] == {"1": "rawGFP", "2": "rawDNA"}
    assert primary["wells"] == {"A01": None}
    assert primary["sites"] == {"1": None}
    assert primary["source_filename_parser_name"] == "SourceSchemaFilenameParser"
    assert primary["available_backends"]["virtual_workspace"] is True
    assert set(auxiliary["workspace_mapping"]) == {
        f"{SOURCE_SCHEMA_WORKSPACE_SOURCE_DIR}/IllumGFP/001_Channel2ILLUM.mat"
    }
    assert not (source_root / "openhcs_metadata.json").exists()


def test_materialize_source_schema_workspace_derives_well_match_field(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "cellprofiler_vitra_source"
    source_root.mkdir()
    _write_image(source_root / "Channel 1-01-A-01-00.tif", value=1)
    _write_image(source_root / "Channel 2-01-A-01-00.tif", value=2)
    workspace_root = tmp_path / "openhcs_workspace"

    result = materialize_source_schema_workspace(
        source_root,
        workspace_root,
        _well_row_column_match_source_schema(),
    )

    metadata = json.loads(result.metadata_path.read_text())
    primary = metadata["subdirectories"]["."]

    assert set(primary["workspace_mapping"]) == {
        "A01_s001_w1_z001_t001.tif",
        "A01_s001_w2_z001_t001.tif",
    }
    assert primary["wells"] == {"A01": None}


def test_materialize_source_schema_workspace_applies_images_rule(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    _write_image(source_root / "Channel1-A01.tif", value=1)
    _write_image(source_root / "Channel2-A01.tif", value=2)
    _write_image(source_root / "Channel1-B01.tif", value=3)
    _write_image(source_root / "Channel2-B01.tif", value=4)

    result = materialize_source_schema_workspace(
        source_root,
        tmp_path / "workspace",
        _filtered_source_schema(),
    )

    metadata = json.loads(result.metadata_path.read_text())
    primary = metadata["subdirectories"]["."]

    assert set(primary["workspace_mapping"]) == {
        "source_s001_w1_z001_t001.tif",
        "source_s001_w2_z001_t001.tif",
    }
    assert primary["wells"] == {"source": None}


def test_materialize_source_schema_workspace_uses_single_default_well_for_ordered_sets(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    _write_image(source_root / "field-001.tif", value=1)
    _write_image(source_root / "field-002.tif", value=2)

    result = materialize_source_schema_workspace(
        source_root,
        tmp_path / "workspace",
        PipelineImageSchema(
            assignments_by_alias={
                "OrigGreen": ImageAssignment(
                    alias="OrigGreen",
                    image_type="Grayscale image",
                    selector=SourceSelector(
                        filters=(
                            SourceFilterClause(
                                SourceFilterSubject.FILE,
                                SourceFilterMatchType.CONTAINS,
                                "field-",
                            ),
                        )
                    ),
                    origin=SourceBindingOrigin.PIPELINE_START,
                ),
            },
            match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
        ),
    )

    metadata = json.loads(result.metadata_path.read_text())
    primary = metadata["subdirectories"]["."]

    assert set(primary["workspace_mapping"]) == {
        "source_s001_w1_z001_t001.tif",
        "source_s002_w1_z001_t001.tif",
    }
    assert primary["wells"] == {"source": None}
    assert primary["sites"] == {"1": None, "2": None}
    for path in primary["workspace_mapping"]:
        assert (
            primary["source_metadata"][path][SOURCE_IMAGE_TYPE_METADATA_FIELD]
            == "Grayscale image"
        )


def test_materialize_source_schema_workspace_disambiguates_duplicate_site_metadata(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    _write_image(source_root / "Plate_A01_s1_w1_GUID1.tif", value=1)
    _write_image(source_root / "Plate_A01_s1_w1_GUID2.tif", value=2)

    result = materialize_source_schema_workspace(
        source_root,
        tmp_path / "workspace",
        PipelineImageSchema(
            metadata_rules=(
                MetadataExtractionRule(
                    source=MetadataSource.FILE_NAME,
                    pattern=r"^Plate_(?P<Well>[A-Z][0-9]{2})_s(?P<Site>[0-9]+)_w(?P<ChannelNumber>[0-9]+)",
                ),
            ),
            assignments_by_alias={
                "DNA": ImageAssignment(
                    alias="DNA",
                    image_type="Grayscale image",
                    selector=SourceSelector(),
                    origin=SourceBindingOrigin.PIPELINE_START,
                ),
            },
            match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
        ),
    )

    metadata = json.loads(result.metadata_path.read_text())
    primary = metadata["subdirectories"]["."]

    assert set(primary["workspace_mapping"]) == {
        "A01_s001_w1_z001_t001.tif",
        "A01_s002_w1_z001_t001.tif",
    }
    assert primary["source_metadata"]["A01_s002_w1_z001_t001.tif"]["Site"] == "1"


def test_materialize_source_schema_workspace_matches_numeric_component_values(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    _write_image(source_root / "Sample_ch00.tif", value=1)

    result = materialize_source_schema_workspace(
        source_root,
        tmp_path / "workspace",
        PipelineImageSchema(
            metadata_rules=(
                MetadataExtractionRule(
                    source=MetadataSource.FILE_NAME,
                    pattern=r"^Sample_ch(?P<ChannelNumber>[0-9]+)",
                ),
            ),
            assignments_by_alias={
                "typeI": ImageAssignment(
                    alias="typeI",
                    image_type="Grayscale image",
                    selector=SourceSelector(
                        components=(ComponentSelector(AllComponents.CHANNEL, "0"),),
                    ),
                    origin=SourceBindingOrigin.PIPELINE_START,
                ),
            },
            match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
        ),
    )

    metadata = json.loads(result.metadata_path.read_text())
    primary = metadata["subdirectories"]["."]

    assert primary["workspace_mapping"]["source_s001_w1_z001_t001.tif"].endswith(
        "source/Sample_ch00.tif"
    )


def test_materialize_source_schema_workspace_applies_source_filters_relative_to_root(
    tmp_path: Path,
) -> None:
    hidden_parent = tmp_path / ".cache" / "dataset"
    hidden_parent.mkdir(parents=True)
    _write_image(hidden_parent / "Sample_ch00.tif", value=1)

    result = materialize_source_schema_workspace(
        hidden_parent,
        tmp_path / "workspace",
        PipelineImageSchema(
            images_rule=ImagesRule(
                filters=(
                    SourceFilterClause(
                        subject=SourceFilterSubject.EXTENSION,
                        match_type=SourceFilterMatchType.IS_IMAGE,
                    ),
                    SourceFilterClause(
                        subject=SourceFilterSubject.DIRECTORY,
                        match_type=SourceFilterMatchType.DOES_NOT_CONTAIN_REGEX,
                        value=r"[\\/]\.",
                    ),
                ),
            ),
            metadata_rules=(
                MetadataExtractionRule(
                    source=MetadataSource.FILE_NAME,
                    pattern=r"^Sample_ch(?P<ChannelNumber>[0-9]+)",
                ),
            ),
            assignments_by_alias={
                "typeI": ImageAssignment(
                    alias="typeI",
                    image_type="Grayscale image",
                    selector=SourceSelector(
                        components=(ComponentSelector(AllComponents.CHANNEL, "00"),),
                    ),
                    origin=SourceBindingOrigin.PIPELINE_START,
                ),
            },
            match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
        ),
    )

    metadata = json.loads(result.metadata_path.read_text())
    primary = metadata["subdirectories"]["."]

    assert primary["workspace_mapping"]["source_s001_w1_z001_t001.tif"].endswith(
        ".cache/dataset/Sample_ch00.tif"
    )


def test_materialize_source_schema_workspace_includes_embedded_image_planes(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    external_root = tmp_path / "external"
    external_root.mkdir()
    _write_image(source_root / "local_D.TIF", value=1)
    _write_image(source_root / "local_F.TIF", value=2)
    _write_image(external_root / "url_D.TIF", value=3)
    _write_image(external_root / "url_F.TIF", value=4)

    result = materialize_source_schema_workspace(
        source_root,
        tmp_path / "workspace",
        PipelineImageSchema(
            image_plane_sources=(
                ImagePlaneSource(uri=(external_root / "url_D.TIF").as_uri()),
                ImagePlaneSource(uri=(external_root / "url_F.TIF").as_uri()),
            ),
            assignments_by_alias={
                "OrigBlue": ImageAssignment(
                    alias="OrigBlue",
                    image_type="Grayscale image",
                    selector=SourceSelector(
                        filters=(
                            SourceFilterClause(
                                SourceFilterSubject.FILE,
                                SourceFilterMatchType.CONTAINS,
                                "D.TIF",
                            ),
                        )
                    ),
                    origin=SourceBindingOrigin.PIPELINE_START,
                ),
                "OrigGreen": ImageAssignment(
                    alias="OrigGreen",
                    image_type="Grayscale image",
                    selector=SourceSelector(
                        filters=(
                            SourceFilterClause(
                                SourceFilterSubject.FILE,
                                SourceFilterMatchType.CONTAINS,
                                "F.TIF",
                            ),
                        )
                    ),
                    origin=SourceBindingOrigin.PIPELINE_START,
                ),
            },
            match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
        ),
    )

    metadata = json.loads(result.metadata_path.read_text())
    primary = metadata["subdirectories"]["."]

    assert set(primary["workspace_mapping"]) == {
        "source_s001_w1_z001_t001.TIF",
        "source_s001_w2_z001_t001.TIF",
        "source_s002_w1_z001_t001.TIF",
        "source_s002_w2_z001_t001.TIF",
    }
    assert primary["sites"] == {"1": None, "2": None}
    assert primary["workspace_mapping"]["source_s002_w1_z001_t001.TIF"].endswith(
        "external/url_D.TIF"
    )


def test_materialize_source_schema_workspace_projects_groups_to_well_axis(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    (source_root / "Sequence1").mkdir(parents=True)
    (source_root / "Sequence2").mkdir()
    _write_image(source_root / "Sequence1" / "Embryo_GFP_0000.tif", value=1)
    _write_image(source_root / "Sequence1" / "Embryo_GFP_0001.tif", value=2)
    _write_image(source_root / "Sequence2" / "Embryo_GFP_0000.tif", value=3)
    _write_image(source_root / "Sequence2" / "Embryo_GFP_0001.tif", value=4)

    result = materialize_source_schema_workspace(
        source_root,
        tmp_path / "workspace",
        _grouped_order_source_schema(),
    )

    metadata = json.loads(result.metadata_path.read_text())
    primary = metadata["subdirectories"]["."]

    assert set(primary["workspace_mapping"]) == {
        "Sequence1_s001_w1_z001_t001.tif",
        "Sequence1_s002_w1_z001_t001.tif",
        "Sequence2_s001_w1_z001_t001.tif",
        "Sequence2_s002_w1_z001_t001.tif",
    }
    assert primary["wells"] == {"Sequence1": None, "Sequence2": None}
    assert primary["sites"] == {"1": None, "2": None}
    assert (
        primary["source_metadata"]["Sequence1_s001_w1_z001_t001.tif"]["FrameNumber"]
        == "0000"
    )


def test_materialize_source_schema_workspace_joins_imported_metadata(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    _write_image(source_root / "Channel1-A-01.tif", value=1)
    _write_image(source_root / "Channel2-A-01.tif", value=2)
    (source_root / "metadata.csv").write_text(
        "Row,Compound\nA,DMSO\n",
        encoding="utf-8",
    )

    result = materialize_source_schema_workspace(
        source_root,
        tmp_path / "workspace",
        _imported_metadata_source_schema(),
    )

    metadata = json.loads(result.metadata_path.read_text())
    primary = metadata["subdirectories"]["."]

    assert set(primary["workspace_mapping"]) == {
        "A01_s001_w1_z001_t001.tif",
        "A01_s001_w2_z001_t001.tif",
    }
    assert primary["wells"] == {"A01": None}
    assert primary["source_metadata"]["A01_s001_w1_z001_t001.tif"]["Compound"] == (
        "DMSO"
    )
    assert result.source_metadata["A01_s001_w2_z001_t001.tif"]["Compound"] == "DMSO"


def _example_sbs_source_schema() -> PipelineImageSchema:
    metadata_rule = MetadataExtractionRule(
        source=MetadataSource.FILE_NAME,
        pattern=r".*-(?P<ImageNumber>\d*)-(?P<WellRow>.*)-(?P<WellColumn>\d*)",
    )
    return PipelineImageSchema(
        metadata_rules=(metadata_rule,),
        assignments_by_alias={
            "rawGFP": ImageAssignment(
                alias="rawGFP",
                image_type="Grayscale image",
                selector=SourceSelector(
                    filters=(
                        SourceFilterClause(
                            SourceFilterSubject.FILE,
                            SourceFilterMatchType.CONTAINS,
                            "Channel1-",
                        ),
                    )
                ),
                origin=SourceBindingOrigin.STEP_INPUT,
            ),
            "rawDNA": ImageAssignment(
                alias="rawDNA",
                image_type="Grayscale image",
                selector=SourceSelector(
                    filters=(
                        SourceFilterClause(
                            SourceFilterSubject.FILE,
                            SourceFilterMatchType.CONTAINS,
                            "Channel2-",
                        ),
                    )
                ),
                origin=SourceBindingOrigin.STEP_INPUT,
            ),
            "IllumGFP": ImageAssignment(
                alias="IllumGFP",
                image_type="Illumination function",
                selector=SourceSelector(
                    filters=(
                        SourceFilterClause(
                            SourceFilterSubject.FILE,
                            SourceFilterMatchType.ENDS_WITH,
                            ".mat",
                        ),
                    )
                ),
                origin=SourceBindingOrigin.PIPELINE_START,
            ),
        },
        match_plan=SourceBindingMatchPlan(
            method=SourceBindingMatchMethod.METADATA,
            dimensions=(
                SourceBindingMatchDimension(
                    fields=(
                        SourceBindingMatchField("rawGFP", "WellRow"),
                        SourceBindingMatchField("rawDNA", "WellRow"),
                    )
                ),
                SourceBindingMatchDimension(
                    fields=(
                        SourceBindingMatchField("rawGFP", "WellColumn"),
                        SourceBindingMatchField("rawDNA", "WellColumn"),
                    )
                ),
            ),
        ),
    )


def _filtered_source_schema() -> PipelineImageSchema:
    return PipelineImageSchema(
        images_rule=ImagesRule(
            filters=(
                SourceFilterClause(
                    SourceFilterSubject.FILE,
                    SourceFilterMatchType.CONTAINS,
                    "A01",
                ),
            ),
        ),
        assignments_by_alias={
            "rawGFP": ImageAssignment(
                alias="rawGFP",
                image_type="Grayscale image",
                selector=SourceSelector(
                    filters=(
                        SourceFilterClause(
                            SourceFilterSubject.FILE,
                            SourceFilterMatchType.CONTAINS,
                            "Channel1",
                        ),
                    )
                ),
                origin=SourceBindingOrigin.PIPELINE_START,
            ),
            "rawDNA": ImageAssignment(
                alias="rawDNA",
                image_type="Grayscale image",
                selector=SourceSelector(
                    filters=(
                        SourceFilterClause(
                            SourceFilterSubject.FILE,
                            SourceFilterMatchType.CONTAINS,
                            "Channel2",
                        ),
                    )
                ),
                origin=SourceBindingOrigin.PIPELINE_START,
            ),
        },
    )


def _imported_metadata_source_schema() -> PipelineImageSchema:
    return PipelineImageSchema(
        metadata_rules=(
            MetadataExtractionRule(
                source=MetadataSource.FILE_NAME,
                pattern=r"^Channel(?P<ChannelNumber>[0-9])-(?P<WellRow>[A-Z])-(?P<WellColumn>[0-9]{2})",
            ),
        ),
        imported_metadata_tables=(
            ImportedMetadataTable(
                location="metadata.csv",
                joins=(
                    ImportedMetadataJoin(
                        image_metadata_field="WellRow",
                        imported_metadata_field="Row",
                    ),
                ),
            ),
        ),
        assignments_by_alias={
            "rawGFP": ImageAssignment(
                alias="rawGFP",
                image_type="Grayscale image",
                selector=SourceSelector(
                    filters=(
                        SourceFilterClause(
                            SourceFilterSubject.FILE,
                            SourceFilterMatchType.CONTAINS,
                            "Channel1",
                        ),
                    ),
                    metadata=(MetadataSelector("Compound", "DMSO"),),
                ),
                origin=SourceBindingOrigin.PIPELINE_START,
            ),
            "rawDNA": ImageAssignment(
                alias="rawDNA",
                image_type="Grayscale image",
                selector=SourceSelector(
                    filters=(
                        SourceFilterClause(
                            SourceFilterSubject.FILE,
                            SourceFilterMatchType.CONTAINS,
                            "Channel2",
                        ),
                    ),
                    metadata=(MetadataSelector("Compound", "DMSO"),),
                ),
                origin=SourceBindingOrigin.PIPELINE_START,
            ),
        },
        match_plan=SourceBindingMatchPlan(
            method=SourceBindingMatchMethod.METADATA,
            dimensions=(
                SourceBindingMatchDimension(
                    fields=(
                        SourceBindingMatchField("rawGFP", "Compound"),
                        SourceBindingMatchField("rawDNA", "Compound"),
                    )
                ),
            ),
        ),
    )


def _grouped_order_source_schema() -> PipelineImageSchema:
    return PipelineImageSchema(
        metadata_rules=(
            MetadataExtractionRule(
                source=MetadataSource.FILE_NAME,
                pattern=r"^(?P<Specimen>.*)_(?P<Stain>.*)_(?P<FrameNumber>[0-9]*)",
            ),
            MetadataExtractionRule(
                source=MetadataSource.FOLDER_NAME,
                pattern=r".*[\\/](?P<Run>.*)$",
            ),
        ),
        assignments_by_alias={
            "OrigColor": ImageAssignment(
                alias="OrigColor",
                image_type="Grayscale image",
                selector=SourceSelector(
                    filters=(
                        SourceFilterClause(
                            SourceFilterSubject.FILE,
                            SourceFilterMatchType.CONTAINS,
                            "GFP",
                        ),
                    )
                ),
                origin=SourceBindingOrigin.PIPELINE_START,
            ),
        },
        match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
        grouping=GroupingPlan(metadata_fields=("Run",)),
    )


def _well_row_column_match_source_schema() -> PipelineImageSchema:
    metadata_rule = MetadataExtractionRule(
        source=MetadataSource.FILE_NAME,
        pattern=(
            r"^Channel (?P<ChannelNumber>[0-9])-[0-9]{2}-"
            r"(?P<WellRow>[A-P])-(?P<WellCol>[0-9]{2})"
        ),
    )
    return PipelineImageSchema(
        metadata_rules=(metadata_rule,),
        assignments_by_alias={
            "OrigProtein": ImageAssignment(
                alias="OrigProtein",
                image_type="Grayscale image",
                selector=SourceSelector(
                    filters=(
                        SourceFilterClause(
                            SourceFilterSubject.FILE,
                            SourceFilterMatchType.CONTAINS,
                            "Channel 1",
                        ),
                    )
                ),
                origin=SourceBindingOrigin.PIPELINE_START,
            ),
            "OrigDNA": ImageAssignment(
                alias="OrigDNA",
                image_type="Color image",
                selector=SourceSelector(
                    filters=(
                        SourceFilterClause(
                            SourceFilterSubject.FILE,
                            SourceFilterMatchType.CONTAINS,
                            "Channel 2",
                        ),
                    )
                ),
                origin=SourceBindingOrigin.PIPELINE_START,
            ),
        },
        match_plan=SourceBindingMatchPlan(
            method=SourceBindingMatchMethod.METADATA,
            dimensions=(
                SourceBindingMatchDimension(
                    fields=(
                        SourceBindingMatchField("OrigProtein", "Well"),
                        SourceBindingMatchField("OrigDNA", "Well"),
                    )
                ),
            ),
        ),
    )


def _write_image(path: Path, *, value: int) -> None:
    image = np.full((8, 8), value, dtype=np.uint16)
    Image.fromarray(image).save(path)
