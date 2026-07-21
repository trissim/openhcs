from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from openhcs.core.aligned_image_payload import (
    AlignedImageSliceContext,
    AlignedImageStack,
    ImageOutputBundle,
    ImagePayloadExecutionMode,
)
from openhcs.core.artifacts import (
    ArtifactOutputPlan,
    ArtifactSidecarRole,
    ArtifactSpec,
    ImageArtifactType,
    MeasurementsArtifactType,
    ObjectLabelsArtifactType,
)
from openhcs.core.callable_contract import CallableContract, CallableMetadata
from openhcs.core.pipeline.function_contracts import artifact_inputs, artifact_outputs
from openhcs.core.runtime_image_values import image_payload_data
from openhcs.core.runtime_object_labels import ObjectLabelValue
from openhcs.core.runtime_output_matching import (
    RuntimeOutputBundle,
    RuntimeReturnedOutputMatcher,
)
from openhcs.core.runtime_tabular_values import ColumnarRows
from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule
from openhcs.interop.cellprofiler.runtime.function_contract_execution import (
    CellProfilerFunctionContractExecutor,
)
from openhcs.processing.backends.lib_registry.unified_registry import (
    ProcessingContract,
)
from tests.unit.test_function_artifact_outputs import (
    ContextStub,
    CoreExecutionRequest,
    _execute_function_core,
)


def _cellprofiler_contract(func) -> CallableContract:
    return replace(CallableContract.from_callable(func), module_name="TestModule")


class _GenericRuntimeOutputs(RuntimeOutputBundle):
    def as_runtime_tuple(
        self,
    ) -> tuple[ObjectLabelValue, ColumnarRows]:
        raise AssertionError("ABI validation must not execute the output bundle")


def test_callable_abi_is_canonical_slot_followed_by_trailing_outputs() -> None:
    source = ArtifactSpec.input("Input", ImageArtifactType)
    labels = ArtifactSpec.output_preserving_source_stack_scope(
        "Labels",
        ObjectLabelsArtifactType,
        source,
    )
    measurements = ArtifactSpec.output("Measurements", MeasurementsArtifactType)

    @artifact_inputs(source)
    @artifact_outputs(labels, measurements)
    def process(
        value: np.ndarray,
    ) -> tuple[ObjectLabelValue, ColumnarRows]:
        raise AssertionError("ABI validation must not execute the callable")

    contract = _cellprofiler_contract(process)

    assert contract.artifact_outputs.specs == (labels, measurements)
    assert contract.canonical_return_output_specs.specs == (labels,)
    assert contract.trailing_return_output_specs.specs == (measurements,)
    CellProfilerModule.validate_callable_artifact_abi(process, contract)


def test_callable_abi_reads_slots_from_nominal_runtime_output_bundle() -> None:
    source = ArtifactSpec.input("Input", ImageArtifactType)
    labels = ArtifactSpec.output_preserving_source_stack_scope(
        "Labels",
        ObjectLabelsArtifactType,
        source,
    )
    measurements = ArtifactSpec.output("Measurements", MeasurementsArtifactType)

    @artifact_inputs(source)
    @artifact_outputs(labels, measurements)
    def process(value: np.ndarray) -> _GenericRuntimeOutputs:
        raise AssertionError("ABI validation must not execute the callable")

    CellProfilerModule.validate_callable_artifact_abi(
        process,
        _cellprofiler_contract(process),
    )


def test_object_label_after_image_is_a_trailing_return_slot() -> None:
    source = ArtifactSpec.input("Input", ImageArtifactType)
    image = ArtifactSpec.output_preserving_source_stack_scope(
        "Image",
        ImageArtifactType,
        source,
    )
    labels = ArtifactSpec.output_preserving_source_stack_scope(
        "Labels",
        ObjectLabelsArtifactType,
        source,
    )

    @artifact_inputs(source)
    @artifact_outputs(image, labels)
    def process(value: np.ndarray) -> tuple[np.ndarray, ObjectLabelValue]:
        raise AssertionError("ABI validation must not execute the callable")

    contract = _cellprofiler_contract(process)

    assert contract.canonical_return_output_specs.specs == (image,)
    assert contract.trailing_return_output_specs.specs == (labels,)
    CellProfilerModule.validate_callable_artifact_abi(process, contract)


def test_callable_abi_accepts_multiple_canonical_names_in_one_stack_slot() -> None:
    source = ArtifactSpec.input("Input", ImageArtifactType)
    first = ArtifactSpec.output_preserving_source_stack_scope(
        "First",
        ImageArtifactType,
        source,
    )
    second = ArtifactSpec.output_preserving_source_stack_scope(
        "Second",
        ImageArtifactType,
        source,
    )
    measurements = ArtifactSpec.output("Measurements", MeasurementsArtifactType)

    @artifact_inputs(source)
    @artifact_outputs(first, second, measurements)
    def process(value: np.ndarray) -> tuple[AlignedImageStack, ColumnarRows]:
        raise AssertionError("ABI validation must not execute the callable")

    CellProfilerModule.validate_callable_artifact_abi(
        process,
        _cellprofiler_contract(process),
    )


def test_cellprofiler_execution_names_each_multi_canonical_stack_slice() -> None:
    first = ArtifactSpec.output("First", ImageArtifactType)
    second = ArtifactSpec.output("Second", ImageArtifactType)

    def process(value: np.ndarray) -> AlignedImageStack:
        return AlignedImageStack((value, value + 1))

    contract = CallableContract(
        func=process,
        function_name="process",
        module_name="TestModule",
        metadata=CallableMetadata(
            artifact_outputs=(first, second),
            processing_contract=ProcessingContract.PURE_3D,
        ),
    )
    returned = CellProfilerFunctionContractExecutor().execute(
        contract,
        process,
        np.zeros((1, 2, 2), dtype=np.float32),
        {},
        execution_mode=ImagePayloadExecutionMode.NATURAL,
    )

    resolved = RuntimeReturnedOutputMatcher(
        callable_contract=contract,
        returned_output=returned,
    ).resolve()

    assert tuple(resolved) == (first.ref(), second.ref())
    np.testing.assert_array_equal(resolved[first.ref()], returned.slices[0])
    np.testing.assert_array_equal(resolved[second.ref()], returned.slices[1])


def test_callable_abi_rejects_missing_trailing_artifact_slot() -> None:
    measurements = ArtifactSpec.output("Measurements", MeasurementsArtifactType)

    @artifact_outputs(measurements)
    def process(value: np.ndarray) -> np.ndarray:
        raise AssertionError("ABI validation must not execute the callable")

    with pytest.raises(
        ValueError,
        match="canonical return followed by exactly 1 trailing",
    ):
        CellProfilerModule.validate_callable_artifact_abi(
            process,
            _cellprofiler_contract(process),
        )


def test_output_matcher_does_not_duplicate_named_canonical_in_trailing_slot() -> None:
    image = ArtifactSpec.output("Image", ImageArtifactType)
    measurements = ArtifactSpec.output("Measurements", MeasurementsArtifactType)
    contract = _cellprofiler_contract(
        artifact_outputs(image, measurements)(lambda value: value)
    )

    matcher = RuntimeReturnedOutputMatcher(
        callable_contract=contract,
        returned_output=("image", "measurements"),
    )

    assert matcher.canonical_output == "image"
    assert matcher.resolve() == {
        image.ref(): "image",
        measurements.ref(): "measurements",
    }


def test_generic_function_save_records_canonical_and_trailing_outputs() -> None:
    first = ArtifactSpec.output("First", ImageArtifactType)
    second = ArtifactSpec.output("Second", ImageArtifactType)
    mask = ArtifactSpec.output(
        "Second__crop_mask",
        ImageArtifactType,
        sidecar_role=ArtifactSidecarRole.CROP_MASK,
    )
    first_value = np.full((2, 3), 1, dtype=np.float32)
    second_value = np.full((2, 3), 2, dtype=np.float32)
    mask_value = np.full((2, 3), 3, dtype=np.float32)

    @artifact_outputs(first, second, mask)
    def process(_image: np.ndarray) -> tuple[ImageOutputBundle, np.ndarray]:
        return (
            ImageOutputBundle(
                (first_value, second_value),
                (
                    AlignedImageSliceContext.main_flow(
                        first.name,
                        artifact_kind=first.artifact_type.value,
                    ),
                    AlignedImageSliceContext.main_flow(
                        second.name,
                        artifact_kind=second.artifact_type.value,
                    ),
                ),
            ),
            mask_value,
        )

    context = ContextStub()
    result = _execute_function_core(
        CoreExecutionRequest(
            func_callable=process,
            main_data_arg=np.zeros((2, 3), dtype=np.float32),
            base_kwargs={},
            context=context,
            artifact_inputs={},
            artifact_outputs={
                spec.ref(): ArtifactOutputPlan(
                    name=spec.name,
                    path=f"/memory/{spec.name}.pkl",
                    artifact_type=spec.artifact_type,
                    sidecar_role=spec.sidecar_role,
                )
                for spec in (first, second, mask)
            },
        )
    )

    assert isinstance(result, ImageOutputBundle)
    np.testing.assert_array_equal(image_payload_data(result.slices[0]), first_value)
    np.testing.assert_array_equal(image_payload_data(result.slices[1]), second_value)
    for spec, expected in (
        (first, first_value),
        (second, second_value),
        (mask, mask_value),
    ):
        records = context.runtime_value_store.find(name=spec.name, axis_id="A01")
        assert len(records) == 1
        np.testing.assert_array_equal(
            image_payload_data(records[0].value.data),
            expected,
        )
