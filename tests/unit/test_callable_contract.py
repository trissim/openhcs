from openhcs.core.aligned_image_payload import ImagePayloadExecutionMode
from openhcs.core.callable_contract import (
    CallableContract,
    RUNTIME_IMAGE_EXECUTION_MODE_ATTR,
    runtime_image_execution_mode,
)
from openhcs.core.pipeline.compiler import FunctionReference


def test_callable_contract_reads_runtime_image_execution_mode() -> None:
    @runtime_image_execution_mode(ImagePayloadExecutionMode.FULL_STACK)
    def process(image):
        return image

    contract = CallableContract.from_callable(process)

    assert contract.runtime_image_execution_mode is ImagePayloadExecutionMode.FULL_STACK


def test_callable_contract_reads_runtime_image_execution_mode_from_function_reference() -> None:
    reference = FunctionReference(
        function_name="process",
        registry_name="test",
        memory_type="numpy",
        composite_key="numpy:process",
        original_module=__name__,
        preserved_attrs={
            RUNTIME_IMAGE_EXECUTION_MODE_ATTR: ImagePayloadExecutionMode.FULL_STACK,
        },
    )

    contract = CallableContract.from_callable(reference)

    assert contract.runtime_image_execution_mode is ImagePayloadExecutionMode.FULL_STACK
