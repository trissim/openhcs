import pickle
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import openhcs.processing.backends.cellprofiler as cellprofiler_backend
import openhcs.serialization.pycodify_formatters  # noqa: F401
from pycodify import Assignment, generate_python_source

from openhcs.constants.constants import GroupBy, VariableComponents
from openhcs.constants.input_source import InputSource
from openhcs.core.callable_contract import CallableContract
from openhcs.core.artifact_materialization_policy import (
    NO_ARTIFACT_MATERIALIZATION,
)
from openhcs.core.artifacts import ArtifactKind, ArtifactSpec
from openhcs.core.compiled_step_plan import CompiledStepPlan
from openhcs.core.function_patterns import (
    CompiledFunctionGroup,
    CompiledFunctionInvocation,
    CompiledFunctionPattern,
    FunctionInvocationKey,
)
from openhcs.core.config import LazyProcessingConfig
from openhcs.core.function_step_transport import FunctionStepTransportAuthority
from openhcs.core.module_artifact_contract import ModuleArtifactContract
from openhcs.core.pipeline.compiler import FunctionReference
from openhcs.core.pipeline.compiler import FunctionReferenceTransportAuthority
from openhcs.core.source_bindings import (
    GroupedSourceBindings,
    NamedSourceBinding,
    SourceBindingOrigin,
    StepSourceBindingsConfig,
)
from openhcs.core.steps.function_runtime import FunctionInvocationCallableResolver
from openhcs.core.steps.function_step import FunctionStep
from openhcs.interop.cellprofiler.runtime.module_execution import (
    cellprofiler_module_callable,
)
from openhcs.interop.cellprofiler.runtime.generated_pipeline import (
    CellProfilerPipelineRuntimeRebinder,
)
from openhcs.processing.materialization.core import MaterializationSpec
from openhcs.processing.materialization.options import TiffStackOptions
from openhcs.runtime.zmq_pipeline_transport import ZMQPipelineCodeTransport


def test_transport_authority_accepts_stripped_compiled_function_steps():
    step = FunctionStep(func=cellprofiler_backend.crop, name="Crop")
    for field_name in tuple(vars(step)):
        delattr(step, field_name)

    normalized = FunctionStepTransportAuthority.normalize_pipeline([step])

    assert normalized == [step]


def test_no_artifact_materialization_survives_pickle_by_identity():
    restored = pickle.loads(pickle.dumps(NO_ARTIFACT_MATERIALIZATION))

    assert restored is NO_ARTIFACT_MATERIALIZATION


def test_transport_authority_normalizes_unstripped_pipeline_steps():
    crop = cellprofiler_backend.crop
    cellprofiler_backend._cellprofiler_function_maps.cache_clear()

    normalized = FunctionStepTransportAuthority.normalize_pipeline(
        [FunctionStep(func=crop, name="Crop")]
    )

    assert normalized[0].func is cellprofiler_backend.crop
    pickle.dumps(normalized)


def test_zmq_pipeline_transport_source_rebinds_cellprofiler_contracts():
    import inspect

    contract = ModuleArtifactContract(
        module_name="Crop",
        outputs=(ArtifactSpec("CropBlue", ArtifactKind.IMAGE),),
    )
    runtime_callable = cellprofiler_module_callable(
        cellprofiler_backend.crop,
        contract,
        declared_processing_contract="PURE_2D",
    )
    pipeline = [FunctionStep(func=runtime_callable, name="Crop")]
    pipeline_source = generate_python_source(
        Assignment("pipeline_steps", pipeline),
        clean_mode=True,
    )

    source = ZMQPipelineCodeTransport.from_pipeline_source(
        source=pipeline_source,
        pipeline_steps=pipeline,
    ).source
    namespace: dict[str, object] = {}
    exec(source, namespace)

    restored = ZMQPipelineCodeTransport.pipeline_from_namespace(namespace)
    rebound = CellProfilerPipelineRuntimeRebinder(
        generated_module_name="generated_test_pipeline",
        contracts_by_module_num={1: contract},
    ).rebind(restored)
    restored_func = rebound[0].func
    restored_contract = CallableContract.from_callable(restored_func)

    assert "__openhcs_zmq_pipeline_payload__" not in source
    assert "ModuleArtifactContract(" not in source
    assert "cellprofiler_module_callable" not in source
    assert restored_func.__name__ == "crop"
    assert "cellprofiler_runtime" in inspect.signature(restored_func).parameters
    assert restored_contract.module_artifact_contract == contract
    pickle.dumps(rebound)


def test_function_step_source_emits_source_bindings_once():
    source_bindings = StepSourceBindingsConfig(
        groups=(
            GroupedSourceBindings(
                bindings=(
                    NamedSourceBinding(
                        alias="OrigBlue",
                        origin=SourceBindingOrigin.PIPELINE_START,
                    ),
                ),
            ),
        ),
    )
    pipeline = [
        FunctionStep(
            func=cellprofiler_backend.crop,
            name="Crop",
            source_bindings=source_bindings,
        )
    ]

    pipeline_source = generate_python_source(
        Assignment("pipeline_steps", pipeline),
        clean_mode=True,
    )

    compile(pipeline_source, "<pipeline>", "exec")
    assert pipeline_source.count("source_bindings=") == 1


def test_zmq_pipeline_transport_preserves_explicit_lazy_processing_defaults():
    pipeline = [
        FunctionStep(
            func=cellprofiler_backend.identify_primary_objects,
            name="IdentifyPrimaryObjects",
            processing_config=LazyProcessingConfig(
                variable_components=[VariableComponents.SITE],
                group_by=GroupBy.NONE,
                input_source=InputSource.PREVIOUS_STEP,
            ),
        )
    ]
    pipeline_source = generate_python_source(
        Assignment("pipeline_steps", pipeline),
        clean_mode=True,
    )

    source = ZMQPipelineCodeTransport.from_pipeline_source(
        source=pipeline_source,
        pipeline_steps=pipeline,
    ).source
    namespace: dict[str, object] = {}
    exec(source, namespace)
    restored = ZMQPipelineCodeTransport.pipeline_from_namespace(namespace)
    restored_processing_config = vars(restored[0])["processing_config"]

    assert "group_by=GroupBy.NONE" in source
    assert "input_source=InputSource.PREVIOUS_STEP" in source
    assert object.__getattribute__(restored_processing_config, "group_by") is GroupBy.NONE
    assert (
        object.__getattribute__(restored_processing_config, "input_source")
        is InputSource.PREVIOUS_STEP
    )


def test_function_step_parameter_order_uses_abstract_step_declaration_order():
    from python_introspect import UnifiedParameterAnalyzer

    parameter_names = list(
        UnifiedParameterAnalyzer.analyze(
            FunctionStep(func=cellprofiler_backend.crop)
        )
    )

    assert parameter_names.index("source_bindings") > parameter_names.index(
        "processing_config"
    )
    assert parameter_names.index("source_bindings") < parameter_names.index(
        "step_well_filter_config"
    )


def test_cellprofiler_runtime_callable_is_function_step_picklable():
    contract = ModuleArtifactContract(
        module_name="IdentifyTertiaryObjects",
        outputs=(ArtifactSpec("Tertiary", ArtifactKind.OBJECT_LABELS),),
    )
    runtime_callable = cellprofiler_module_callable(
        cellprofiler_backend.identify_tertiary_objects,
        contract,
    )

    restored_step = pickle.loads(
        pickle.dumps(FunctionStep(func=runtime_callable, name="IdentifyTertiaryObjects"))
    )
    restored_contract = CallableContract.from_callable(restored_step.func)

    assert restored_step.func.__name__ == "identify_tertiary_objects"
    assert restored_contract.module_artifact_contract == contract


def test_cellprofiler_runtime_callable_source_derives_materialization_contract():
    contract = ModuleArtifactContract(
        module_name="Crop",
        outputs=(
            ArtifactSpec(
                "CropBlue",
                ArtifactKind.IMAGE,
                materialization=MaterializationSpec(
                    TiffStackOptions(normalize_uint8=True),
                ),
            ),
        ),
    )
    runtime_callable = cellprofiler_module_callable(
        cellprofiler_backend.crop,
        contract,
        declared_processing_contract="PURE_2D",
    )
    pipeline = [FunctionStep(func=runtime_callable, name="Crop")]
    pipeline_source = generate_python_source(
        Assignment("pipeline_steps", pipeline),
        clean_mode=True,
    )
    namespace: dict[str, object] = {}
    exec(pipeline_source, namespace)

    restored = ZMQPipelineCodeTransport.pipeline_from_namespace(namespace)
    rebound = CellProfilerPipelineRuntimeRebinder(
        generated_module_name="generated_test_pipeline",
        contracts_by_module_num={1: contract},
    ).rebind(restored)
    restored_contract = CallableContract.from_callable(rebound[0].func)
    restored_materialization = restored_contract.module_artifact_contract.outputs[
        0
    ].materialization

    assert "MaterializationSpec(" not in pipeline_source
    assert "ModuleArtifactContract(" not in pipeline_source
    assert restored_materialization == contract.outputs[0].materialization


def test_zmq_pipeline_transport_uses_source_only_cellprofiler_catalog_identity_after_reload():
    import importlib

    crop = cellprofiler_backend.crop
    crop_module = importlib.import_module("openhcs.processing.backends.cellprofiler.crop")
    importlib.reload(crop_module)
    pipeline = [FunctionStep(func=(crop, {"crop_shape": "Rectangle"}), name="Crop")]
    pipeline_source = generate_python_source(
        Assignment("pipeline_steps", pipeline),
        clean_mode=True,
    )

    source = ZMQPipelineCodeTransport.from_pipeline_source(
        source=pipeline_source,
        pipeline_steps=pipeline,
    ).source
    namespace: dict[str, object] = {}
    exec(source, namespace)

    restored = ZMQPipelineCodeTransport.pipeline_from_namespace(namespace)
    restored_func = restored[0].func

    assert "__openhcs_zmq_pipeline_payload__" not in source
    assert restored_func is cellprofiler_backend.crop
    pickle.dumps(restored)


def test_registered_custom_function_is_function_step_picklable():
    from openhcs.processing.custom_functions import CustomFunctionManager
    import openhcs.processing.custom_functions as custom_functions

    code = """
@numpy
def codex_pickle_probe(image):
    return image
"""
    with TemporaryDirectory() as tmp_dir:
        manager = CustomFunctionManager()
        manager.storage_dir = Path(tmp_dir)
        manager.register_from_code(code, persist=False)

        custom_func = custom_functions.codex_pickle_probe
        restored_step = pickle.loads(
            pickle.dumps(FunctionStep(func=custom_func, name="CustomProbe"))
        )

    assert restored_step.func is custom_functions.codex_pickle_probe


def test_function_reference_transport_preserves_runtime_callable_contracts():
    contract = ModuleArtifactContract(
        module_name="Crop",
        outputs=(ArtifactSpec("CropBlue", ArtifactKind.IMAGE),),
    )
    runtime_callable = cellprofiler_module_callable(
        cellprofiler_backend.crop,
        contract,
        declared_processing_contract="PURE_2D",
    )

    referenced = FunctionReferenceTransportAuthority.reference_pipeline(
        [FunctionStep(func=(runtime_callable, {"crop_shape": "Rectangle"}), name="Crop")]
    )

    referenced_func = referenced[0].func[0]
    referenced_contract = CallableContract.from_callable(referenced_func)

    assert isinstance(referenced_func, FunctionReference)
    assert referenced_contract.module_artifact_contract == contract
    pickle.dumps(referenced)


def test_transport_authority_normalizes_compiled_context_callable_contracts():
    crop = cellprofiler_backend.crop
    cellprofiler_backend._cellprofiler_function_maps.cache_clear()

    contract = CallableContract.from_callable(crop)
    step_plan = CompiledStepPlan(
        step_index=0,
        step_name="Crop",
        step_type="FunctionStep",
        axis_id="A01",
        func=(crop, {"crop_shape": "Rectangle"}),
        compiled_function_pattern=CompiledFunctionPattern(
            groups=(
                CompiledFunctionGroup(
                    group_key="default",
                    invocations=(
                        CompiledFunctionInvocation(
                            key=FunctionInvocationKey.from_contract(
                                contract,
                                "default",
                                0,
                            ),
                            contract=contract,
                        ),
                    ),
                ),
            ),
            is_grouped=False,
        ),
    )
    context = SimpleNamespace(step_plans={0: step_plan})

    normalized_contexts = FunctionStepTransportAuthority.normalize_contexts(
        {"A01": context}
    )

    normalized_plan = normalized_contexts["A01"].step_plans[0]
    normalized_invocation = next(
        normalized_plan.compiled_function_pattern.iter_invocations()
    )
    assert normalized_plan.func[0] is cellprofiler_backend.crop
    assert normalized_invocation.contract.func is cellprofiler_backend.crop
    assert (
        normalized_invocation.contract.raw_processing_function
        is cellprofiler_backend.crop
    )
    pickle.dumps(normalized_contexts)


def test_transport_authority_normalizes_cellprofiler_submodule_raw_contract():
    from openhcs.processing.backends.cellprofiler.crop import crop as raw_crop

    contract = CallableContract.from_callable(cellprofiler_backend.crop)
    contract = replace(contract, raw_processing_function=raw_crop)
    step_plan = CompiledStepPlan(
        step_index=0,
        step_name="Crop",
        step_type="FunctionStep",
        axis_id="A01",
        func=cellprofiler_backend.crop,
        compiled_function_pattern=CompiledFunctionPattern(
            groups=(
                CompiledFunctionGroup(
                    group_key="default",
                    invocations=(
                        CompiledFunctionInvocation(
                            key=FunctionInvocationKey.from_contract(
                                contract,
                                "default",
                                0,
                            ),
                            contract=contract,
                        ),
                    ),
                ),
            ),
            is_grouped=False,
        ),
    )

    normalized_contexts = FunctionStepTransportAuthority.normalize_contexts(
        {"A01": SimpleNamespace(step_plans={0: step_plan})}
    )
    normalized_invocation = next(
        normalized_contexts["A01"]
        .step_plans[0]
        .compiled_function_pattern
        .iter_invocations()
    )

    assert normalized_invocation.contract.raw_processing_function is (
        cellprofiler_backend.crop
    )
    pickle.dumps(normalized_contexts)


def test_transport_authority_normalizes_function_reference_preserved_callables():
    from openhcs.core.callable_contract import (
        PROCESSING_PREPARE_ATTR,
        RAW_PROCESSING_FUNCTION_ATTR,
    )
    from openhcs.processing.backends.cellprofiler.crop import crop as raw_crop

    def local_prepare():
        pass

    reference = FunctionReference(
        function_name="crop",
        registry_name="openhcs",
        memory_type="numpy",
        composite_key="openhcs:openhcs.processing.backends.cellprofiler:crop",
        original_module="openhcs.processing.backends.cellprofiler",
        preserved_attrs={
            RAW_PROCESSING_FUNCTION_ATTR: raw_crop,
            PROCESSING_PREPARE_ATTR: local_prepare,
        },
    )

    normalized = FunctionStepTransportAuthority.normalize_function_spec(reference)

    assert PROCESSING_PREPARE_ATTR not in normalized.preserved_attrs
    assert (
        normalized.preserved_attrs[RAW_PROCESSING_FUNCTION_ATTR]
        is cellprofiler_backend.crop
    )
    pickle.dumps(normalized)


def test_function_reference_resolver_preserves_cellprofiler_module_contracts(
    monkeypatch,
):
    raw_crop = cellprofiler_backend.crop
    blue_callable = cellprofiler_module_callable(
        raw_crop,
        ModuleArtifactContract(
            module_name="Crop",
            inputs=(ArtifactSpec("OrigBlue", ArtifactKind.IMAGE),),
        ),
    )
    green_callable = cellprofiler_module_callable(
        raw_crop,
        ModuleArtifactContract(
            module_name="Crop",
            inputs=(ArtifactSpec("OrigGreen", ArtifactKind.IMAGE),),
        ),
    )
    blue_contract = CallableContract.from_callable(blue_callable)
    green_contract = CallableContract.from_callable(green_callable)
    blue_reference = FunctionReference(
        function_name="crop",
        registry_name="openhcs",
        memory_type="numpy",
        composite_key="openhcs:openhcs.processing.backends.cellprofiler:crop",
        original_module="openhcs.processing.backends.cellprofiler",
        preserved_attrs={},
    )
    green_reference = FunctionReference(
        function_name="crop",
        registry_name="openhcs",
        memory_type="numpy",
        composite_key="openhcs:openhcs.processing.backends.cellprofiler:crop",
        original_module="openhcs.processing.backends.cellprofiler",
        preserved_attrs={},
    )
    blue_contract = replace(blue_contract, func=blue_reference)
    green_contract = replace(green_contract, func=green_reference)
    monkeypatch.setattr(FunctionReference, "resolve", lambda _self: raw_crop)
    FunctionInvocationCallableResolver._cache.clear()

    blue_resolved = FunctionInvocationCallableResolver.resolve(
        CompiledFunctionInvocation(
            key=FunctionInvocationKey.from_contract(blue_contract, "default", 0),
            contract=blue_contract,
        )
    )
    green_resolved = FunctionInvocationCallableResolver.resolve(
        CompiledFunctionInvocation(
            key=FunctionInvocationKey.from_contract(green_contract, "default", 0),
            contract=green_contract,
        )
    )

    assert blue_resolved is not green_resolved
    assert tuple(
        spec.name
        for spec in CallableContract.from_callable(
            blue_resolved
        ).module_artifact_contract.inputs
    ) == ("OrigBlue",)
    assert tuple(
        spec.name
        for spec in CallableContract.from_callable(
            green_resolved
        ).module_artifact_contract.inputs
    ) == ("OrigGreen",)
