from openhcs.core.aligned_image_payload import ImagePayloadExecutionMode
from openhcs.core.callable_contract import (
    CallableContract,
    CompilerPreparedAutoRegisterFamily,
    PROCESSING_CONTRACT_ATTR,
    RUNTIME_IMAGE_EXECUTION_MODE_ATTR,
    attach_callable_contract_metadata,
    prepare_processing_callable,
    runtime_image_execution_mode,
)
from openhcs.core.pipeline.compiler import FunctionReference
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract


_PREPARED_TEST_FAMILY_CALLS = 0


class _TestCompilerPreparedFamily(CompilerPreparedAutoRegisterFamily):
    @classmethod
    def prepare_registered_family(cls) -> None:
        global _PREPARED_TEST_FAMILY_CALLS
        _PREPARED_TEST_FAMILY_CALLS += 1


def _function_with_imported_prepared_family(image):
    return image


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


def test_callable_contract_metadata_preserves_explicit_nominal_processing_contract() -> None:
    def process(image):
        return image

    process.__processing_contract__ = ProcessingContract.FLEXIBLE

    attach_callable_contract_metadata(
        process,
        declared_processing_contract=ProcessingContract.PURE_2D.name,
    )

    contract = CallableContract.from_callable(process)
    assert getattr(process, PROCESSING_CONTRACT_ATTR) is ProcessingContract.FLEXIBLE
    assert contract.processing_contract is ProcessingContract.FLEXIBLE
    assert contract.declared_processing_contract == ProcessingContract.PURE_2D.name


def test_prepare_processing_callable_warms_imported_registered_families() -> None:
    prepare_processing_callable(_function_with_imported_prepared_family)

    assert _PREPARED_TEST_FAMILY_CALLS == 1
