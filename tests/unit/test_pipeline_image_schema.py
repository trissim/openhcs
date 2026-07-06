import pytest

from openhcs.core.artifacts import ArtifactType, ImageArtifactType, ObjectLabelsArtifactType
from openhcs.constants.constants import AllComponents
from openhcs.core.pipeline_image_schema import (
    GroupingPlan,
    GrayscaleImageTypeSourceRole,
    ImageAssignment,
    ImagePlaneSource,
    ImagesRule,
    ImportedMetadataJoin,
    ImportedMetadataTable,
    PipelineImageSchemaBuilder,
    PipelineImageSchema,
    SourceImageStackPlan,
    SourceArtifactAssignment,
    image_type_artifact_kind,
    image_type_loads_as_monochrome,
    image_type_materializes_source_mask,
    image_type_participates_in_image_stack,
)
from openhcs.core.source_bindings import (
    SourceBindingMatchMethod,
    SourceBindingMatchPlan,
    SourceFilterClause,
    SourceFilterMatchType,
    SourceFilterSubject,
    MetadataExtractionRule,
    MetadataSource,
    SourceBindingOrigin,
    SourceSelector,
)
from openhcs.constants.constants import Microscope
from openhcs.interop.cellprofiler.pipeline_generator import PipelineGeneratorBuildStage


def test_pipeline_image_schema_builder_deduplicates_metadata_rules():
    builder = PipelineImageSchemaBuilder()
    rule = MetadataExtractionRule(
        source=MetadataSource.FILE_NAME,
        pattern=r"(?P<well>[A-H]\d{2})\.tif",
    )

    builder.add_metadata_rule(rule)
    builder.add_metadata_rule(rule)

    assert builder.build().metadata_rules == (rule,)


def test_pipeline_image_schema_builder_rejects_alias_kind_conflicts():
    builder = PipelineImageSchemaBuilder()
    builder.declare_assignment(
        ImageAssignment(
            alias="Nuclei",
            image_type="Grayscale image",
            selector=SourceSelector(),
            origin=SourceBindingOrigin.STEP_INPUT,
        )
    )

    with pytest.raises(ValueError, match="already declared as an image assignment"):
        builder.declare_source_artifact(
            SourceArtifactAssignment(
                alias="Nuclei",
                artifact_kind=ObjectLabelsArtifactType,
                selector=SourceSelector(),
                origin=SourceBindingOrigin.PIPELINE_START,
            )
        )


def test_pipeline_image_schema_measurement_source_names_exclude_object_artifacts():
    builder = PipelineImageSchemaBuilder()
    builder.declare_assignment(
        ImageAssignment(
            alias="OrigColor",
            image_type="Color image",
            selector=SourceSelector(),
            origin=SourceBindingOrigin.PIPELINE_START,
        )
    )
    builder.declare_source_artifact(
        SourceArtifactAssignment(
            alias="Embryos",
            artifact_kind=ObjectLabelsArtifactType,
            selector=SourceSelector(),
            origin=SourceBindingOrigin.STEP_INPUT,
        )
    )

    assert builder.build().measurement_source_names == ("OrigColor",)


def test_pipeline_image_schema_projects_representable_source_bindings_config():
    filter_clause = SourceFilterClause(
        SourceFilterSubject.FILE,
        SourceFilterMatchType.ENDS_WITH,
        ".tif",
    )
    metadata_rule = MetadataExtractionRule(
        source=MetadataSource.FILE_NAME,
        pattern=r"(?P<well>[A-H]\d{2})\.tif",
    )
    match_plan = SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER)
    schema = PipelineImageSchema(
        images_rule=ImagesRule(filters=(filter_clause,)),
        metadata_rules=(metadata_rule,),
        assignments_by_alias={
            "DNA": ImageAssignment(
                alias="DNA",
                image_type=GrayscaleImageTypeSourceRole.image_type(),
                selector=SourceSelector(),
                origin=SourceBindingOrigin.PIPELINE_START,
            )
        },
        source_artifacts_by_alias={
            "Nuclei": SourceArtifactAssignment(
                alias="Nuclei",
                artifact_kind=ObjectLabelsArtifactType,
                selector=SourceSelector(),
                origin=SourceBindingOrigin.PIPELINE_START,
            )
        },
        match_plan=match_plan,
    )

    config = schema.to_source_bindings_config()

    assert config.source_filters == (filter_clause,)
    assert config.metadata_rules == (metadata_rule,)
    assert config.match_plan == match_plan
    assert tuple(binding.alias for binding in config.bindings) == ("DNA", "Nuclei")
    assert config.bindings[0].artifact_kind is ImageArtifactType
    assert config.bindings[1].artifact_kind is ObjectLabelsArtifactType


def test_pipeline_image_schema_lowers_to_source_bindings_pipeline_config():
    filter_clause = SourceFilterClause(
        SourceFilterSubject.FILE,
        SourceFilterMatchType.ENDS_WITH,
        ".tif",
    )
    schema = PipelineImageSchema(
        images_rule=ImagesRule(filters=(filter_clause,)),
        assignments_by_alias={
            "DNA": ImageAssignment(
                alias="DNA",
                image_type=GrayscaleImageTypeSourceRole.image_type(),
                selector=SourceSelector(),
                origin=SourceBindingOrigin.PIPELINE_START,
            )
        },
    )

    pipeline_config = PipelineGeneratorBuildStage._pipeline_config(schema)

    assert pipeline_config is not None
    assert pipeline_config.microscope is Microscope.SOURCE_BINDINGS
    assert pipeline_config.source_bindings_config == schema.to_source_bindings_config()


@pytest.mark.parametrize(
    ("schema", "field_name"),
    (
        (
            PipelineImageSchema(
                image_plane_sources=(ImagePlaneSource(uri="file:///tmp/image.tif"),),
            ),
            "image_plane_sources",
        ),
        (
            PipelineImageSchema(
                imported_metadata_tables=(
                    ImportedMetadataTable(
                        location="metadata.csv",
                        joins=(ImportedMetadataJoin("Well", "Well"),),
                    ),
                ),
            ),
            "imported_metadata_tables",
        ),
        (
            PipelineImageSchema(
                source_image_stack=SourceImageStackPlan(
                    components=(AllComponents.Z_INDEX,),
                ),
            ),
            "source_image_stack",
        ),
        (
            PipelineImageSchema(grouping=GroupingPlan(metadata_fields=("Plate",))),
            "grouping",
        ),
        (
            PipelineImageSchema(
                assignments_by_alias={
                    "Color": ImageAssignment(
                        alias="Color",
                        image_type="Color image",
                        selector=SourceSelector(),
                        origin=SourceBindingOrigin.PIPELINE_START,
                    )
                },
            ),
            "source_assignment_payloads",
        ),
        (
            PipelineImageSchema(
                source_artifacts_by_alias={
                    "Illum": SourceArtifactAssignment(
                        alias="Illum",
                        artifact_kind=ImageArtifactType,
                        selector=SourceSelector(),
                        origin=SourceBindingOrigin.PIPELINE_START,
                        payload_type="Illumination function",
                    )
                },
            ),
            "source_assignment_payloads",
        ),
    ),
)
def test_pipeline_image_schema_source_bindings_projection_rejects_unrepresented_fields(
    schema: PipelineImageSchema,
    field_name: str,
):
    with pytest.raises(ValueError, match=field_name):
        schema.to_source_bindings_config()


@pytest.mark.parametrize(
    ("image_type", "artifact_kind", "participates_in_stack"),
    (
        ("Grayscale image", ImageArtifactType, True),
        ("Illumination function", ImageArtifactType, False),
        ("Objects", ObjectLabelsArtifactType, False),
    ),
)
def test_image_type_roles_define_artifact_kind_and_stack_participation(
    image_type: str,
    artifact_kind: ArtifactType,
    participates_in_stack: bool,
):
    assert image_type_artifact_kind(image_type) is artifact_kind
    assert image_type_participates_in_image_stack(image_type) is participates_in_stack


def test_image_type_role_class_owns_schema_label():
    image_type = GrayscaleImageTypeSourceRole.image_type()

    assert image_type_artifact_kind(image_type) is ImageArtifactType
    assert image_type_participates_in_image_stack(image_type)


@pytest.mark.parametrize(
    ("image_type", "loads_as_monochrome"),
    (
        ("Grayscale image", True),
        ("Binary image", True),
        ("Binary mask", True),
        ("Mask", True),
        ("Color image", False),
        ("Objects", False),
    ),
)
def test_image_type_roles_define_source_monochrome_loading(
    image_type: str,
    loads_as_monochrome: bool,
):
    assert image_type_loads_as_monochrome(image_type) is loads_as_monochrome


@pytest.mark.parametrize(
    ("image_type", "materializes_source_mask"),
    (
        ("Grayscale image", True),
        ("Color image", True),
        ("Binary image", True),
        ("Binary mask", True),
        ("Mask", True),
        ("Illumination function", False),
        ("Objects", False),
    ),
)
def test_image_type_roles_define_source_mask_materialization(
    image_type: str,
    materializes_source_mask: bool,
):
    assert image_type_materializes_source_mask(image_type) is materializes_source_mask
