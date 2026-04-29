from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from openhcs.core.pipeline_image_schema import (
    ImageAssignment,
    ImagesRule,
    ImportedMetadataJoin,
    ImportedMetadataTable,
    PipelineImageSchema,
)
from openhcs.core.source_bindings import (
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
    assert primary["source_filename_parser_name"] == "ImageXpressFilenameParser"
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
        "A01_s001_w1_z001_t001.tif",
        "A01_s001_w2_z001_t001.tif",
    }
    assert primary["wells"] == {"A01": None}


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
