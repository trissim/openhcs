import pickle
import sys
from types import MappingProxyType, ModuleType

from metaclass_registry import AutoRegisterMeta

from openhcs.core.aligned_image_payload import ImagePayloadExecutionMode
from openhcs.core.autoregister_preparation import AutoRegisterRegistryPreparation
from openhcs.core.callable_contract import (
    CallableContract,
    CallableMetadata,
    CompilerPreparedAutoRegisterFamily,
    attach_callable_contract_metadata,
    prepare_processing_callable,
    prepare_module_autoregister_families,
    reset_processing_callable_preparation_cache,
    runtime_image_execution_mode,
)
from openhcs.constants.constants import VariableComponents
from openhcs.core.config import LazyDtypeConfig
from openhcs.core.function_contract_metadata import FunctionContractAttribute
from openhcs.core.function_reference import FunctionReference
from openhcs.core.memory.decorators import numpy
from openhcs.core.pipeline.function_contracts import special_inputs
from openhcs.core.pipeline.function_contracts import (
    required_variable_components,
    runtime_bound_parameters,
)
from openhcs.core.runtime_batch_contracts import SliceIndexRuntimeParameter
from openhcs.core.runtime_batch_contracts import (
    RuntimeBatchExecutionDomain,
    RuntimePure2DSliceBatchRequest,
    pure_2d_batch_executor,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from python_introspect import parameter_exclusions


_AUTOREGISTER_PREPARED_TEST_FAMILY_CALLS = 0


class _PreparedAutoRegisterFamily(
    CompilerPreparedAutoRegisterFamily,
    metaclass=AutoRegisterMeta,
):
    __registry_key__ = "registry_key"
    __skip_if_no_key__ = True
    registry_key = None

    @classmethod
    def prepare_registered_family(cls) -> None:
        global _AUTOREGISTER_PREPARED_TEST_FAMILY_CALLS
        _AUTOREGISTER_PREPARED_TEST_FAMILY_CALLS += 1


class _PreparedAutoRegisterImplementation(_PreparedAutoRegisterFamily):
    registry_key = "prepared"


def _function_with_imported_prepared_family(image):
    return image


def _batch_executor(request):
    return request


@pure_2d_batch_executor(_batch_executor)
def _function_with_runtime_batch_executor(image):
    return image


def test_callable_contract_reads_runtime_image_execution_mode() -> None:
    @runtime_image_execution_mode(ImagePayloadExecutionMode.FULL_STACK)
    def process(image):
        return image

    contract = CallableContract.from_callable(process)

    assert contract.runtime_image_execution_mode is ImagePayloadExecutionMode.FULL_STACK


def test_callable_contract_reads_runtime_bound_parameters() -> None:
    @runtime_bound_parameters(SliceIndexRuntimeParameter)
    def process(image, *, slice_index: int = 0):
        del slice_index
        return image

    contract = CallableContract.from_callable(process)

    assert contract.runtime_bound_parameters == ("slice_index",)
    assert contract.runtime_bound_parameter_types == (SliceIndexRuntimeParameter,)
    assert CallableMetadata.from_callable(process).as_namespace()[
        FunctionContractAttribute.runtime_bound_parameters
    ] == (SliceIndexRuntimeParameter,)


def test_callable_contract_reads_wrapper_declared_config_parameters() -> None:
    @numpy(contract=ProcessingContract.PURE_3D)
    def process(image):
        return image

    contract = CallableContract.from_callable(process)

    assert contract.config_bound_parameter_names == ("dtype_config",)
    assert contract.runtime_owned_parameter_names == frozenset({"dtype_config"})
    (parameter,) = contract.config_bound_parameters
    assert parameter.annotation is LazyDtypeConfig


def test_memory_wrapper_preserves_declared_parameter_exclusions() -> None:
    @numpy(contract=ProcessingContract.PURE_2D)
    @special_inputs("mask")
    def process(image, *, mask):
        return image

    assert "mask" in parameter_exclusions(process)


def test_callable_contract_reads_required_variable_components() -> None:
    @required_variable_components(VariableComponents.TIMEPOINT)
    def process(image):
        return image

    contract = CallableContract.from_callable(process)

    assert contract.required_variable_components == (VariableComponents.TIMEPOINT,)
    assert CallableMetadata.from_callable(process).as_namespace()[
        FunctionContractAttribute.required_variable_components
    ] == (VariableComponents.TIMEPOINT,)


def test_callable_contract_reads_runtime_image_execution_mode_from_function_reference() -> None:
    reference = FunctionReference(
        function_name="process",
        registry_name="test",
        memory_type="numpy",
        composite_key="numpy:process",
        original_module=__name__,
        metadata=CallableMetadata(
            runtime_image_execution_mode=ImagePayloadExecutionMode.FULL_STACK,
        ),
    )

    contract = CallableContract.from_callable(reference)

    assert contract.runtime_image_execution_mode is ImagePayloadExecutionMode.FULL_STACK


def test_callable_contract_metadata_preserves_explicit_nominal_processing_contract() -> None:
    def process(image):
        return image

    vars(process)[FunctionContractAttribute.processing_contract] = (
        ProcessingContract.FLEXIBLE
    )

    attach_callable_contract_metadata(
        process,
        declared_processing_contract=ProcessingContract.PURE_2D.name,
    )

    contract = CallableContract.from_callable(process)
    assert (
        vars(process)[FunctionContractAttribute.processing_contract]
        is ProcessingContract.FLEXIBLE
    )
    assert contract.processing_contract is ProcessingContract.FLEXIBLE
    assert contract.declared_processing_contract == ProcessingContract.PURE_2D.name


def test_prepare_processing_callable_warms_imported_registered_families() -> None:
    global _AUTOREGISTER_PREPARED_TEST_FAMILY_CALLS
    _AUTOREGISTER_PREPARED_TEST_FAMILY_CALLS = 0
    reset_processing_callable_preparation_cache()

    prepare_processing_callable(_function_with_imported_prepared_family)

    assert _AUTOREGISTER_PREPARED_TEST_FAMILY_CALLS == 1


def test_prepare_processing_callable_does_not_warm_unrelated_loaded_families() -> None:
    calls = {"related": 0, "unrelated": 0}

    class RelatedFamily(CompilerPreparedAutoRegisterFamily, metaclass=AutoRegisterMeta):
        __registry_key__ = "registry_key"
        __skip_if_no_key__ = True
        registry_key = None

        @classmethod
        def prepare_registered_family(cls) -> None:
            calls["related"] += 1

    class RelatedImplementation(RelatedFamily):
        registry_key = "related"

    class UnrelatedFamily(
        CompilerPreparedAutoRegisterFamily,
        metaclass=AutoRegisterMeta,
    ):
        __registry_key__ = "registry_key"
        __skip_if_no_key__ = True
        registry_key = None

        @classmethod
        def prepare_registered_family(cls) -> None:
            calls["unrelated"] += 1

    class UnrelatedImplementation(UnrelatedFamily):
        registry_key = "unrelated"

    def process(image):
        return image

    related_module = ModuleType("tests.unit._warmup_related_module")
    unrelated_module = ModuleType("tests.unit._warmup_unrelated_module")
    related_module.RelatedFamily = RelatedFamily
    related_module.RelatedImplementation = RelatedImplementation
    related_module.process = process
    unrelated_module.UnrelatedFamily = UnrelatedFamily
    unrelated_module.UnrelatedImplementation = UnrelatedImplementation
    process.__module__ = related_module.__name__

    reset_processing_callable_preparation_cache()
    sys.modules[related_module.__name__] = related_module
    sys.modules[unrelated_module.__name__] = unrelated_module
    try:
        prepare_processing_callable(process)
    finally:
        sys.modules.pop(related_module.__name__, None)
        sys.modules.pop(unrelated_module.__name__, None)
        reset_processing_callable_preparation_cache()

    assert calls == {"related": 1, "unrelated": 0}


def test_prepare_processing_callable_caches_equivalent_bound_method_hooks() -> None:
    module_name = __name__

    class PreparedCallable:
        __name__ = "prepared_callable"
        __module__ = module_name

        calls = 0

        def __call__(self, image):
            return image

        def prepare(self) -> None:
            type(self).calls += 1

    first = PreparedCallable()
    second = PreparedCallable()
    first.__dict__[FunctionContractAttribute.processing_prepare] = first.prepare
    second.__dict__[FunctionContractAttribute.processing_prepare] = second.prepare

    reset_processing_callable_preparation_cache()

    prepare_processing_callable(first)
    prepare_processing_callable(second)

    assert PreparedCallable.calls == 1


def test_prepare_module_autoregister_families_skips_cellprofiler_backend_mixin_root() -> None:
    prepare_module_autoregister_families(
        "openhcs.processing.backends.cellprofiler.crop"
    )


def test_module_registered_family_preparation_runs_compiler_prepared_family_hook() -> None:
    global _AUTOREGISTER_PREPARED_TEST_FAMILY_CALLS
    _AUTOREGISTER_PREPARED_TEST_FAMILY_CALLS = 0
    AutoRegisterRegistryPreparation.cached_module_registry_families.cache_clear()

    report = AutoRegisterRegistryPreparation.prepare_module_registered_families(
        (sys.modules[__name__],)
    )

    assert _AUTOREGISTER_PREPARED_TEST_FAMILY_CALLS == 1
    assert report.prepared_family_count == 1
    assert _PreparedAutoRegisterImplementation in (
        _PreparedAutoRegisterFamily.__registry__.values()
    )


def test_callable_contract_preserves_immutable_runtime_batch_executors() -> None:
    contract = CallableContract.from_callable(_function_with_runtime_batch_executor)

    assert isinstance(contract.runtime_batch_executors, MappingProxyType)
    assert (
        contract.runtime_batch_executor(RuntimeBatchExecutionDomain.PURE_2D_SLICES)
        is _batch_executor
    )


def test_callable_contract_pickles_runtime_batch_executors() -> None:
    restored = pickle.loads(
        pickle.dumps(
            CallableContract.from_callable(_function_with_runtime_batch_executor)
        )
    )

    assert isinstance(restored.runtime_batch_executors, MappingProxyType)
    assert (
        restored.runtime_batch_executor(RuntimeBatchExecutionDomain.PURE_2D_SLICES)
        is _batch_executor
    )


def test_runtime_slice_batch_request_exposes_callable_defaults() -> None:
    def process(image, *, method="otsu", threshold=0.5):
        return image, method, threshold

    def execute_slice(func, image, kwargs, slice_index, slice_count):
        del slice_index, slice_count
        return func(image, **kwargs)

    request = RuntimePure2DSliceBatchRequest(
        func=process,
        slices_2d=("image",),
        kwargs={"threshold": 0.75},
        execute_slice=execute_slice,
    )

    assert request.kwargs["method"] == "otsu"
    assert request.kwargs["threshold"] == 0.75
    assert request.execute_one(0) == ("image", "otsu", 0.75)


def test_runtime_slice_batch_request_preserves_callable_result_identity() -> None:
    class MeasurementRows:
        pass

    rows = MeasurementRows()
    result = ("image", rows)

    request = RuntimePure2DSliceBatchRequest(
        func=lambda image: image,
        slices_2d=("image",),
        kwargs={},
        execute_slice=lambda func, image, kwargs, slice_index, slice_count: result,
    )

    assert request.execute_one(0) is result
