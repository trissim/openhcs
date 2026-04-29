import pytest

from openhcs.core.artifacts import ArtifactKind
from openhcs.core.pipeline_image_schema import (
    ImageAssignment,
    PipelineImageSchemaBuilder,
    SourceArtifactAssignment,
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
