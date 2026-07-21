"""Normalization of artifact-fed callable parameter declarations."""

from dataclasses import replace

import pytest

from openhcs.core.artifacts import ArtifactSpec, SpecialArtifactType
from openhcs.core.callable_contract import CallableContract
from openhcs.core.function_patterns import compile_function_pattern
from openhcs.core.pipeline.function_contracts import (
    artifact_inputs,
    special_input_names_from_callable,
    special_inputs,
)


def test_legacy_only_artifact_parameter_declaration_compiles() -> None:
    @special_inputs("labels")
    def consume(image, *, labels):
        del labels
        return image

    compiled = compile_function_pattern(consume, {}, {})
    contract = compiled.default_group.invocations[0].contract

    assert contract.artifact_input_parameter_names == ("labels",)
    assert contract.artifact_inputs.names() == ()


def test_artifact_spec_only_parameter_declaration_compiles() -> None:
    labels = ArtifactSpec.input(
        "StoredLabels",
        SpecialArtifactType,
        parameter_name="labels",
    )

    @artifact_inputs(labels)
    def consume(image, *, labels):
        del labels
        return image

    compiled = compile_function_pattern(consume, {}, {})
    contract = compiled.default_group.invocations[0].contract

    assert contract.artifact_input_parameter_names == ("labels",)
    assert special_input_names_from_callable(consume) == ("labels",)


def test_matching_legacy_and_artifact_spec_declarations_compile() -> None:
    labels = ArtifactSpec.input(
        "StoredLabels",
        SpecialArtifactType,
        parameter_name="labels",
    )

    @artifact_inputs(labels)
    @special_inputs("labels")
    def consume(image, *, labels):
        del labels
        return image

    compiled = compile_function_pattern(consume, {}, {})

    assert (
        compiled.default_group.invocations[0].contract.artifact_input_parameter_names
        == ("labels",)
    )


def test_conflicting_legacy_and_artifact_spec_declarations_fail_compilation() -> None:
    labels = ArtifactSpec.input(
        "StoredLabels",
        SpecialArtifactType,
        parameter_name="labels",
    )

    @artifact_inputs(labels)
    @special_inputs("mask")
    def consume(image, *, labels, mask):
        del labels, mask
        return image

    with pytest.raises(ValueError, match="artifact-fed parameter declarations disagree"):
        compile_function_pattern(consume, {}, {})


def test_legacy_declaration_agrees_with_compiled_exact_artifact_binding() -> None:
    @special_inputs("labels")
    def consume(image, *, labels):
        del labels
        return image

    contract = CallableContract.from_callable(consume)
    labels = ArtifactSpec.input(
        "StoredLabels",
        SpecialArtifactType,
        parameter_name="labels",
    )
    compiled_contract = replace(
        contract,
        metadata=replace(contract.metadata, artifact_inputs=(labels,)),
    )

    compiled_contract.validate_artifact_input_parameter_bindings()
    assert compiled_contract.artifact_input_parameter_names == ("labels",)
