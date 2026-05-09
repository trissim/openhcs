from __future__ import annotations

from openhcs.core.artifacts import ArtifactKind, ArtifactSpec
from openhcs.core.runtime_output_matching import RuntimeReturnedOutputMatcher


def test_runtime_output_matcher_uses_declared_main_output_position() -> None:
    image = ArtifactSpec("Image", ArtifactKind.IMAGE)
    measurements = ArtifactSpec("Measurements", ArtifactKind.MEASUREMENTS)

    resolved = RuntimeReturnedOutputMatcher(
        retained_specs=(measurements,),
        declared_specs=(image, measurements),
        main_output="image-payload",
        artifact_values=("measurement-table",),
    ).resolve()

    assert resolved == {"Measurements": "measurement-table"}


def test_runtime_output_matcher_uses_returned_specs_semantics_when_names_differ() -> None:
    retained = ArtifactSpec("FilteredObjects", ArtifactKind.OBJECT_LABELS)
    returned = ArtifactSpec("Objects", ArtifactKind.OBJECT_LABELS)

    resolved = RuntimeReturnedOutputMatcher(
        retained_specs=(retained,),
        declared_specs=(),
        main_output="image-payload",
        artifact_values=("label-payload",),
        returned_specs=(returned,),
    ).resolve()

    assert resolved == {"FilteredObjects": "label-payload"}


def test_runtime_output_matcher_preserves_positional_image_main_flow() -> None:
    image = ArtifactSpec("Corrected", ArtifactKind.IMAGE)
    measurements = ArtifactSpec("Diagnostics", ArtifactKind.MEASUREMENTS)

    resolved = RuntimeReturnedOutputMatcher(
        retained_specs=(image, measurements),
        declared_specs=(),
        main_output="image-payload",
        artifact_values=("measurement-table",),
    ).resolve()

    assert resolved == {
        "Corrected": "image-payload",
        "Diagnostics": "measurement-table",
    }
