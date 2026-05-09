import pytest

from openhcs.core.artifacts import ArtifactKind
from openhcs.core.pipeline_image_schema import (
    ImageAssignment,
    PipelineImageSchemaBuilder,
    SourceArtifactAssignment,
    image_type_artifact_kind,
    image_type_loads_as_monochrome,
    image_type_materializes_source_mask,
    image_type_participates_in_image_stack,
)
from openhcs.core.source_bindings import (
    MetadataExtractionRule,
    MetadataSource,
    SourceBindingOrigin,
    SourceSelector,
)


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
                kind=ArtifactKind.OBJECT_LABELS,
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
            kind=ArtifactKind.OBJECT_LABELS,
            selector=SourceSelector(),
            origin=SourceBindingOrigin.STEP_INPUT,
        )
    )

    assert builder.build().measurement_source_names == ("OrigColor",)


@pytest.mark.parametrize(
    ("image_type", "artifact_kind", "participates_in_stack"),
    (
        ("Grayscale image", ArtifactKind.IMAGE, True),
        ("Illumination function", ArtifactKind.IMAGE, False),
        ("Objects", ArtifactKind.OBJECT_LABELS, False),
    ),
)
def test_image_type_roles_define_artifact_kind_and_stack_participation(
    image_type: str,
    artifact_kind: ArtifactKind,
    participates_in_stack: bool,
):
    assert image_type_artifact_kind(image_type) is artifact_kind
    assert image_type_participates_in_image_stack(image_type) is participates_in_stack


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
