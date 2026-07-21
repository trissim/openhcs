"""Transport boundaries for public FunctionStep declarations."""

from __future__ import annotations
from openhcs.core.pipeline_document import PipelineDocumentAuthority

import ast
import importlib
import inspect
import pickle
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from zmqruntime.messages import MessageFields

from openhcs.constants.constants import GroupBy, VariableComponents
from openhcs.core.config import (
    GlobalPipelineConfig,
    LazyProcessingConfig,
    PipelineConfig,
)
from openhcs.core.callable_contract import CallableContract
from openhcs.core.function_reference import (
    FunctionReference,
    FunctionReferenceTransportAuthority,
)
from openhcs.core.function_step_transport import FunctionStepTransportAuthority
from openhcs.core.source_bindings import (
    NamedSourceBinding,
    SourceBindingOrigin,
    StepSourceBindingsConfig,
)
from openhcs.core.steps.function_step import FunctionStep
from openhcs.processing.backends import cellprofiler as cellprofiler_backend
from openhcs.processing.backends.lib_registry.registry_service import RegistryService
from openhcs.processing.backends.cellprofiler.illumination import (
    IlluminationCorrectionMethod,
)
from openhcs.runtime.zmq_execution_client import (
    OpenHCSExecutionSubmission,
    ZMQExecutionClient,
    ZMQExecutionRequestBuilder,
)


def _public_step() -> FunctionStep:
    return FunctionStep(
        func=(
            cellprofiler_backend.correct_illumination_apply,
            {
                "method": IlluminationCorrectionMethod.SUBTRACT,
                "truncate_low": False,
            },
        ),
        name="CorrectIlluminationApply",
        processing_config=LazyProcessingConfig(
            variable_components=[VariableComponents.CHANNEL],
            group_by=GroupBy.SITE,
        ),
    )


def test_transport_authority_accepts_stripped_compiled_function_steps() -> None:
    step = FunctionStep(func=cellprofiler_backend.crop, name="Crop")
    for field_name in tuple(vars(step)):
        delattr(step, field_name)

    normalized = FunctionStepTransportAuthority.normalize_pipeline([step])

    assert normalized == [step]


def test_function_reference_transport_rejects_three_member_leaf() -> None:
    with pytest.raises(TypeError, match="exactly two"):
        FunctionReferenceTransportAuthority.reference_function_spec(
            (cellprofiler_backend.crop, {}, object())
        )


def test_transport_round_trip_preserves_public_steps_and_typed_kwargs_only() -> None:
    source = FunctionStepTransportAuthority.source_from_pipeline([_public_step()])
    namespace: dict[str, object] = {}
    exec(compile(source, "<pipeline>", "exec"), namespace)

    restored = FunctionStepTransportAuthority.pipeline_steps_from_namespace(namespace)

    assert type(restored) is list
    assert len(restored) == 1
    restored_func, restored_kwargs = restored[0].func
    assert restored_func is RegistryService.registered_callable(
        cellprofiler_backend.correct_illumination_apply
    )
    assert restored_kwargs == {
        "method": IlluminationCorrectionMethod.SUBTRACT,
        "truncate_low": False,
    }


def test_compiler_referenced_steps_render_only_public_callables() -> None:
    pipeline = [
        _public_step(),
        FunctionStep(
            func={
                "1": cellprofiler_backend.crop,
                "2": cellprofiler_backend.crop,
            },
            name="Crop",
        ),
    ]
    FunctionReferenceTransportAuthority.reference_pipeline_in_place(pipeline)

    first_reference = pipeline[0].func[0]
    assert isinstance(first_reference, FunctionReference)
    assert all(
        isinstance(reference, FunctionReference)
        for reference in pipeline[1].func.values()
    )

    source = FunctionStepTransportAuthority.source_from_pipeline(pipeline)
    namespace: dict[str, object] = {}
    exec(compile(source, "<pipeline>", "exec"), namespace)
    restored = FunctionStepTransportAuthority.pipeline_steps_from_namespace(namespace)

    assert "FunctionReference" not in source
    assert "CallableMetadata" not in source
    assert restored[0].func[0] is RegistryService.registered_callable(
        cellprofiler_backend.correct_illumination_apply
    )
    assert tuple(restored[1].func) == ("1", "2")
    assert all(
        func is RegistryService.registered_callable(cellprofiler_backend.crop)
        for func in restored[1].func.values()
    )
    assert FunctionStepTransportAuthority.source_from_pipeline(restored) == source


@pytest.mark.parametrize(
    ("namespace", "message"),
    (
        ({}, "pipeline_steps"),
        ({"pipeline_steps": ()}, "list"),
        ({"pipeline_steps": [object()]}, "FunctionStep"),
    ),
)
def test_pipeline_namespace_requires_direct_function_step_list(
    namespace: dict[str, object],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        FunctionStepTransportAuthority.pipeline_steps_from_namespace(namespace)


def test_submission_namespace_and_server_payload_use_direct_step_lists() -> None:
    step = _public_step()
    global_config = GlobalPipelineConfig()
    pipeline_config = PipelineConfig()
    submission = OpenHCSExecutionSubmission(
        plate_id="/tmp/plate",
        pipeline_document=PipelineDocumentAuthority.from_values(
            pipeline_config=pipeline_config, pipeline_steps=[step]
        ),
        global_config=global_config,
    )

    assert type(submission.pipeline_steps) is list
    submitted_func, submitted_kwargs = submission.pipeline_steps[0].func
    assert submitted_func is RegistryService.registered_callable(
        cellprofiler_backend.correct_illumination_apply
    )
    assert submitted_kwargs == step.func[1]
    assert submission.global_pipeline_config is global_config
    assert submission.pipeline_config is pipeline_config
    assert submission.pipeline_code() == PipelineDocumentAuthority.render(
        submission.pipeline_document
    )
    assert not hasattr(submission, "submission_pipeline")
    assert "pipeline_steps_boundary" not in dir(submission)


def test_request_builder_stores_explicit_pipeline_code_directly() -> None:
    pipeline_document = PipelineDocumentAuthority.from_values(
        pipeline_config=PipelineConfig(),
        pipeline_steps=[],
    )
    pipeline_source = PipelineDocumentAuthority.render(pipeline_document)
    submission = OpenHCSExecutionSubmission(
        plate_id="/tmp/plate",
        pipeline_document=PipelineDocumentAuthority.from_source(pipeline_source),
        global_config=GlobalPipelineConfig(),
    )
    compile_submission = submission.compile_request()
    builder = ZMQExecutionRequestBuilder.from_task(compile_submission)
    artifact_submission = OpenHCSExecutionSubmission(
        plate_id="/tmp/plate",
        pipeline_document=PipelineDocumentAuthority.from_source(pipeline_source),
        global_config=submission.global_pipeline_config,
        compile_artifact_id="compile-1",
    )
    artifact_builder = ZMQExecutionRequestBuilder.from_task(artifact_submission)

    assert compile_submission.pipeline_steps is submission.pipeline_steps
    assert compile_submission.pipeline_code() == pipeline_source
    assert compile_submission.compile_only is True
    assert builder.pipeline_code == pipeline_source
    assert builder.request_payload.pipeline_code == pipeline_source
    assert builder.request().values[MessageFields.PIPELINE_CODE] == pipeline_source
    assert not hasattr(builder, "pipeline_transport")
    assert builder.request().values[MessageFields.COMPILE_ONLY] is True
    assert artifact_builder.pipeline_code == pipeline_source
    assert artifact_builder.request().values[MessageFields.COMPILE_ARTIFACT_ID] == (
        "compile-1"
    )
    assert MessageFields.COMPILE_ONLY not in artifact_builder.request().values


def test_zmq_execution_submission_serializes_default_plate_config_on_client() -> None:
    submission = OpenHCSExecutionSubmission(
        plate_id="/tmp/plate",
        pipeline_document=PipelineDocumentAuthority.from_values(
            pipeline_config=PipelineConfig(), pipeline_steps=[]
        ),
        global_config=GlobalPipelineConfig(),
    )

    payload = ZMQExecutionClient().serialize_task(submission)
    document = PipelineDocumentAuthority.from_source(
        payload[MessageFields.PIPELINE_CODE]
    )

    assert MessageFields.CONFIG_CODE in payload
    assert MessageFields.PIPELINE_CONFIG_CODE not in payload
    assert isinstance(document.pipeline_config, PipelineConfig)


def test_function_step_source_emits_source_bindings_once() -> None:
    source_bindings = StepSourceBindingsConfig(
        bindings=(
            NamedSourceBinding(
                alias="OrigBlue",
                origin=SourceBindingOrigin.PIPELINE_START,
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

    source = FunctionStepTransportAuthority.source_from_pipeline(pipeline)

    compile(source, "<pipeline>", "exec")
    assert source.count("source_bindings=") == 1


def test_transport_preserves_explicit_lazy_processing_defaults() -> None:
    source = FunctionStepTransportAuthority.source_from_pipeline([_public_step()])
    namespace: dict[str, object] = {}
    exec(compile(source, "<pipeline>", "exec"), namespace)
    restored = FunctionStepTransportAuthority.pipeline_steps_from_namespace(namespace)

    assert restored[0].processing_config.variable_components == [
        VariableComponents.CHANNEL
    ]
    assert restored[0].processing_config.group_by is GroupBy.SITE


def test_cellprofiler_callable_uses_registered_function_reference() -> None:
    func = cellprofiler_backend.crop
    registered = RegistryService.registered_callable(func)
    reference = FunctionReferenceTransportAuthority.function_reference(func)

    assert reference.function_name == "crop"
    assert reference.original_module == func.__module__
    assert reference.resolve() is registered
    assert reference.metadata == CallableContract.from_callable(registered).metadata


def test_module_objects_are_rejected_as_function_specs() -> None:
    crop_module = importlib.import_module(
        "openhcs.processing.backends.cellprofiler.crop"
    )

    with pytest.raises(TypeError, match="module object"):
        FunctionStepTransportAuthority.normalize_function_spec(crop_module)


def test_generated_source_imports_underlying_cellprofiler_callable() -> None:
    source = FunctionStepTransportAuthority.source_from_pipeline(
        [FunctionStep(func=cellprofiler_backend.crop, name="Crop")]
    )
    module = ast.parse(source)
    imported_modules = {
        node.module for node in module.body if isinstance(node, ast.ImportFrom)
    }

    assert cellprofiler_backend.crop.__module__ in imported_modules
    assert (
        "openhcs.interop.cellprofiler.runtime.module_execution" not in imported_modules
    )


def test_registered_custom_function_is_function_step_picklable() -> None:
    from openhcs.processing.custom_functions import CustomFunctionManager
    import openhcs.processing.custom_functions as custom_functions
    from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule

    canonical_crop = cellprofiler_backend.crop
    code = """
@numpy
def codex_pickle_probe(image):
    return image
"""
    with TemporaryDirectory() as tmp_dir:
        manager = CustomFunctionManager()
        manager.storage_dir = Path(tmp_dir)
        manager.register_from_code(code, persist=False, emit_signal=False)

        custom_func = custom_functions.codex_pickle_probe
        assert inspect.unwrap(custom_func).__module__ == (
            "openhcs.processing.custom_functions"
        )

        reference = FunctionReferenceTransportAuthority.function_reference(custom_func)
        assert reference.original_module == "openhcs.processing.custom_functions"
        assert reference.resolve() is custom_func

        compiler_pipeline = [FunctionStep(func=custom_func, name="CustomProbe")]
        FunctionReferenceTransportAuthority.reference_pipeline_in_place(
            compiler_pipeline
        )
        assert isinstance(compiler_pipeline[0].func, FunctionReference)
        assert compiler_pipeline[0].func.resolve() is custom_func

        restored_step = pickle.loads(
            pickle.dumps(FunctionStep(func=custom_func, name="CustomProbe"))
        )

    assert restored_step.func is custom_functions.codex_pickle_probe
    assert (
        CellProfilerModule.require_module("Crop").require_callable() is canonical_crop
    )


def test_persisted_custom_function_is_importable_from_package(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import openhcs.processing.custom_functions as custom_functions
    import openhcs.processing.custom_functions.manager as manager_module

    func_name = "codex_lazy_import_probe"
    storage_dir = tmp_path / "custom_functions"
    storage_dir.mkdir()
    (storage_dir / f"{func_name}.py").write_text(
        """
@numpy
def codex_lazy_import_probe(image):
    return image
""",
        encoding="utf-8",
    )
    vars(custom_functions).pop(func_name, None)
    monkeypatch.setattr(manager_module, "get_data_file_path", lambda _name: storage_dir)

    namespace: dict[str, object] = {}
    try:
        exec(
            f"from openhcs.processing.custom_functions import {func_name}\n"
            f"imported = {func_name}",
            namespace,
        )

        imported = namespace["imported"]
        assert imported.__name__ == func_name
        assert imported.__module__ == "openhcs.processing.custom_functions"
        assert vars(custom_functions)[func_name] is imported
    finally:
        vars(custom_functions).pop(func_name, None)
