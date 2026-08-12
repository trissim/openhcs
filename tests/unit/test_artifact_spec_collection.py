from __future__ import annotations

import pytest

from openhcs.core.artifacts import (
    ArtifactOutputPlan,
    ArtifactSpec,
    ArtifactSpecAccumulator,
    ArtifactSpecCollection,
    GroupLineageSourceRelation,
    ImageArtifactType,
    MeasurementsArtifactType,
    ObjectLabelsArtifactType,
)
from openhcs.core.runtime_measurements import RuntimeMeasurementFeatureOwner


class FirstMeasurementFeatureOwner(RuntimeMeasurementFeatureOwner):
    @classmethod
    def owns_measurement_feature_name(cls, feature_name: str) -> bool:
        return feature_name == "first"

    @classmethod
    def owns_primary_measurement_feature_name(cls, feature_name: str) -> bool:
        return cls.owns_measurement_feature_name(feature_name)


class SecondMeasurementFeatureOwner(RuntimeMeasurementFeatureOwner):
    @classmethod
    def owns_measurement_feature_name(cls, feature_name: str) -> bool:
        return feature_name == "second"

    @classmethod
    def owns_primary_measurement_feature_name(cls, feature_name: str) -> bool:
        return cls.owns_measurement_feature_name(feature_name)


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


def test_measurement_artifact_preserves_its_nominal_feature_owner() -> None:
    measurement = ArtifactSpec.output(
        "Measurements",
        MeasurementsArtifactType,
        measurement_feature_owner=FirstMeasurementFeatureOwner,
    )

    assert measurement.measurement_feature_owner is FirstMeasurementFeatureOwner
    assert (
        measurement.for_plan_type(ArtifactOutputPlan).measurement_feature_owner
        is FirstMeasurementFeatureOwner
    )

    with pytest.raises(TypeError, match="measurements artifact"):
        ArtifactSpec.output(
            "DNA",
            ImageArtifactType,
            measurement_feature_owner=FirstMeasurementFeatureOwner,
        )


def test_artifact_accumulator_rejects_conflicting_measurement_feature_owners() -> None:
    accumulator = ArtifactSpecAccumulator.empty("producer")
    accumulator.add(
        ArtifactSpec.output(
            "Measurements",
            MeasurementsArtifactType,
            measurement_feature_owner=FirstMeasurementFeatureOwner,
        )
    )

    with pytest.raises(ValueError, match="measurement feature owner"):
        accumulator.add(
            ArtifactSpec.output(
                "Measurements",
                MeasurementsArtifactType,
                measurement_feature_owner=SecondMeasurementFeatureOwner,
            )
        )


def test_artifact_spec_collection_rejects_ambiguous_active_lookup() -> None:
    with pytest.raises(ValueError, match="Conflicting active image artifact"):
        ArtifactSpecCollection(
            (
                ArtifactSpec.input("DNA", ImageArtifactType),
                ArtifactSpec.output("DNA", ImageArtifactType),
            )
        ).by_name_and_artifact_type("DNA", ImageArtifactType)


def test_artifact_spec_collection_selects_occurrences_in_declaration_order() -> None:
    primary = ArtifactSpec.input("Primary", ImageArtifactType)
    mask = ArtifactSpec.input("Mask", ImageArtifactType)
    collection = ArtifactSpecCollection((primary, mask, primary))

    assert collection.select_declared_occurrences((mask, primary)).specs == (
        primary,
        mask,
    )
    assert collection.select_declared_occurrences((primary, primary)).specs == (
        primary,
        primary,
    )

    with pytest.raises(ValueError, match="occurrence cardinality"):
        collection.select_declared_occurrences((mask, mask))


def test_artifact_spec_collection_selects_exact_refs_in_declaration_order() -> None:
    first = ArtifactSpec.output("First", ImageArtifactType)
    second = ArtifactSpec.output("Second", ImageArtifactType)
    collection = ArtifactSpecCollection((first, second))

    assert collection.select_refs((second.ref(),)).specs == (second,)

    with pytest.raises(ValueError, match="duplicate identities"):
        collection.select_refs((first.ref(), first.ref()))
    with pytest.raises(ValueError, match="one exact declared occurrence"):
        collection.select_refs(
            (ArtifactSpec.output("Missing", ImageArtifactType).ref(),)
        )


def test_image_output_plan_requires_one_exact_source_context() -> None:
    first = ArtifactSpec.input("First", ImageArtifactType).ref()
    second = ArtifactSpec.input("Second", ImageArtifactType).ref()
    relations = (
        GroupLineageSourceRelation(first),
        GroupLineageSourceRelation(second),
    )

    with pytest.raises(ValueError, match="multiple runtime-context sources"):
        ArtifactOutputPlan(
            name="Combined",
            path="/memory/Combined.pkl",
            artifact_type=ImageArtifactType,
            relations=relations,
        )

    measurements = ArtifactOutputPlan(
        name="Measurements",
        path="/memory/Measurements.pkl",
        artifact_type=MeasurementsArtifactType,
        relations=relations,
    )
    assert measurements.source_context_source() is None
