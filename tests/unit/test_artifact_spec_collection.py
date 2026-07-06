from __future__ import annotations

import pytest

from openhcs.core.artifacts import (
    ArtifactSpec,
    ArtifactSpecCollection,
    ImageArtifactType,
    ObjectLabelsArtifactType,
)


def test_artifact_spec_collection_queries_ordered_artifact_contracts() -> None:
    image = ArtifactSpec.output("DNA", ImageArtifactType)
    objects = ArtifactSpec.output("Nuclei", ObjectLabelsArtifactType)

    collection = ArtifactSpecCollection((image, objects))

    assert collection.of_artifact_type(ImageArtifactType) == (image,)
    assert collection.by_name("Nuclei") == objects
    assert collection.by_name_and_artifact_type("Nuclei", ObjectLabelsArtifactType) == objects
    assert collection.by_name_and_artifact_type("Nuclei", ImageArtifactType) is None


def test_artifact_spec_collection_deduplicates_or_fails_loudly() -> None:
    image = ArtifactSpec.output("DNA", ImageArtifactType)

    assert ArtifactSpecCollection((image, image)).unique() == (image,)

    with pytest.raises(ValueError, match="Conflicting runtime declarations"):
        ArtifactSpecCollection(
            (
                image,
                ArtifactSpec.output("DNA", ImageArtifactType, required=False),
            )
        ).unique(conflict_context="runtime")
