from openhcs.core.artifacts import ArtifactSpec, ImageArtifactType
from openhcs.core.callable_contract import CallableContract, CallableMetadata
from openhcs.core.function_patterns import (
    CompiledFunctionInvocation,
    FunctionInvocationKey,
)
from openhcs.core.function_reference import FunctionReference
from openhcs.core.steps.function_runtime import FunctionInvocationCallableResolver


def _callable_contract_for_reference(
    reference: FunctionReference,
    output_spec: ArtifactSpec,
) -> CallableContract:
    return CallableContract(
        func=reference,
        function_name=reference.function_name,
        module_name=reference.original_module,
        metadata=CallableMetadata(
            input_memory_type="python",
            output_memory_type="python",
            artifact_outputs=(output_spec,),
        ),
    )


def test_function_reference_cache_key_distinguishes_compiled_callable_contracts():
    reference = FunctionReference(
        function_name="image_math",
        registry_name="cellprofiler",
        memory_type="python",
        composite_key="cellprofiler:image_math",
        original_module="openhcs.processing.backends.cellprofiler",
    )
    first_contract = _callable_contract_for_reference(
        reference,
        ArtifactSpec.output("CombinedImage", ImageArtifactType),
    )
    second_contract = _callable_contract_for_reference(
        reference,
        ArtifactSpec.output("SubtractedRed", ImageArtifactType),
    )

    first_invocation = CompiledFunctionInvocation(
        key=FunctionInvocationKey.from_contract(first_contract, "default", 0),
        contract=first_contract,
    )
    second_invocation = CompiledFunctionInvocation(
        key=FunctionInvocationKey.from_contract(second_contract, "default", 0),
        contract=second_contract,
    )

    assert FunctionInvocationCallableResolver.cache_key(
        first_invocation
    ) != FunctionInvocationCallableResolver.cache_key(second_invocation)
