from __future__ import annotations

from openhcs.core.artifacts import (
    ArtifactSpec,
    ImageArtifactType,
    ObjectLabelsArtifactType,
    MeasurementsArtifactType,
)
from openhcs.core.runtime_output_matching import RuntimeReturnedOutputMatcher


def test_runtime_output_matcher_uses_declared_main_output_position() -> None:
    image = ArtifactSpec.output("Image", ImageArtifactType)
    measurements = ArtifactSpec.output("Measurements", MeasurementsArtifactType)

    resolved = RuntimeReturnedOutputMatcher(
        retained_specs=(measurements,),
        declared_specs=(image, measurements),
        main_output="image-payload",
        artifact_values=("measurement-table",),
    ).resolve()

    assert resolved == {"Measurements": "measurement-table"}


def test_runtime_output_matcher_uses_declared_main_output_position_at_end() -> None:
    image = ArtifactSpec.output("Image", ImageArtifactType)
    measurements = ArtifactSpec.output("Measurements", MeasurementsArtifactType)

    resolved = RuntimeReturnedOutputMatcher(
        retained_specs=(measurements, image),
        declared_specs=(measurements, image),
        main_output="image-payload",
        artifact_values=("measurement-table",),
    ).resolve()

    assert resolved == {
        "Measurements": "measurement-table",
        "Image": "image-payload",
    }


def test_runtime_output_matcher_uses_returned_specs_semantics_when_names_differ() -> None:
    retained = ArtifactSpec.output("FilteredObjects", ObjectLabelsArtifactType)
    returned = ArtifactSpec.output("Objects", ObjectLabelsArtifactType)

    resolved = RuntimeReturnedOutputMatcher(
        retained_specs=(retained,),
        declared_specs=(),
        main_output="image-payload",
        artifact_values=("label-payload",),
        returned_specs=(returned,),
    ).resolve()

    assert resolved == {"FilteredObjects": "label-payload"}


def test_runtime_output_matcher_single_image_uses_main_output_before_sidecars() -> None:
    image = ArtifactSpec.output("IlluminationFunction", ImageArtifactType)

    resolved = RuntimeReturnedOutputMatcher(
        retained_specs=(image,),
        declared_specs=(image,),
        main_output="image-payload",
        artifact_values=("illumination-stats",),
    ).resolve()

    assert resolved == {"IlluminationFunction": "image-payload"}


def test_runtime_output_matcher_preserves_positional_image_main_flow() -> None:
    image = ArtifactSpec.output("Corrected", ImageArtifactType)
    measurements = ArtifactSpec.output("Diagnostics", MeasurementsArtifactType)

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
