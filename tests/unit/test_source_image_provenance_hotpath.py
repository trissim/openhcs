"""Exact regressions for source-provenance reconstruction."""

from openhcs.core.runtime_image_values import ImagePayloadMetadata
from openhcs.core.source_image_provenance import (
    SourceImageProvenance,
    SourceImageProvenanceFields,
)


def test_empty_explicit_provenance_reuses_authoritative_value() -> None:
    retained = SourceImageProvenance(
        source_path="/input/A01_s001_w1.tif",
        source_image_names=("DNA",),
    )
    fields = SourceImageProvenanceFields(source_provenance=retained)

    fields.absorb_explicit_source_provenance(SourceImageProvenance())

    assert fields.source_provenance is retained


def test_populated_explicit_provenance_preserves_merge_precedence() -> None:
    fallback = SourceImageProvenance(
        source_path="/input/A01_s001_w1.tif",
        source_image_names=("Fallback",),
    )
    explicit = SourceImageProvenance(
        source_component_metadata={"channel": "2"},
        source_image_names=("Explicit",),
    )
    fields = SourceImageProvenanceFields(source_provenance=fallback)

    fields.absorb_explicit_source_provenance(explicit)

    assert fields.source_provenance == SourceImageProvenance(
        source_path="/input/A01_s001_w1.tif",
        source_component_metadata={"channel": "2"},
        source_image_names=("Explicit",),
    )
    assert fields.source_provenance is not fallback
    assert fields.source_provenance is not explicit


def test_replace_fields_preserves_final_provenance_copy_isolation() -> None:
    metadata = ImagePayloadMetadata(
        source_provenance=SourceImageProvenance(
            source_path="/input/A01_s001_w1.tif",
            source_component_metadata={"well": "A01", "channel": "1"},
            source_image_names=("DNA",),
        )
    )
    retained = metadata.source_provenance

    replaced = metadata.replace_fields()

    assert replaced.source_provenance == retained
    assert replaced.source_provenance is not retained
