from __future__ import annotations

import pytest

from openhcs.core.artifacts import ArtifactKind, ArtifactSpec, ArtifactSpecCollection


def test_artifact_spec_collection_queries_ordered_artifact_contracts() -> None:
    image = ArtifactSpec("DNA", ArtifactKind.IMAGE)
    objects = ArtifactSpec("Nuclei", ArtifactKind.OBJECT_LABELS)

    collection = ArtifactSpecCollection((image, objects))

    assert collection.of_kind(ArtifactKind.IMAGE) == (image,)
    assert collection.by_name("Nuclei") == objects
    assert collection.by_name_and_kind("Nuclei", ArtifactKind.OBJECT_LABELS) == objects
    assert collection.by_name_and_kind("Nuclei", ArtifactKind.IMAGE) is None


def test_artifact_spec_collection_deduplicates_or_fails_loudly() -> None:
    image = ArtifactSpec("DNA", ArtifactKind.IMAGE)

    assert ArtifactSpecCollection((image, image)).unique() == (image,)

    with pytest.raises(ValueError, match="Conflicting runtime declarations"):
        ArtifactSpecCollection(
            (
                image,
                ArtifactSpec("DNA", ArtifactKind.IMAGE, required=False),
            )
        ).unique(conflict_context="runtime")
