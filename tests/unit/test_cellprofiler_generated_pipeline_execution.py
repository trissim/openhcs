from pathlib import Path

import numpy as np

from benchmark.converter.parser import ModuleBlock
from benchmark.converter.pipeline_generator import GeneratedPipeline, PipelineGenerator
from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactKind,
    ArtifactOutputPlan,
)
from openhcs.core.config import DtypeConfig
from openhcs.core.runtime_adapters import runtime_adapter_spec_from_callable
from openhcs.core.runtime_stores import RuntimeValueStore
from openhcs.core.source_bindings import CompiledSourceBindingPlan
from openhcs.core.steps.function_runtime import (
    FunctionExecutionRequest,
    _execute_function_core,
)


AXIS_ID = "A01"
SOURCE_IMAGE = "OrigBlue"
NUCLEI = "Nuclei"
NUCLEI_IMAGE = "NucleiImage"
OPENED_NUCLEI_IMAGE = "OpenedNucleiImage"
OVERLAY_IMAGE = "OverlayImage"
IDENTIFY_PRIMARY_OBJECTS = "IdentifyPrimaryObjects"
MEASURE_OBJECT_SIZE_SHAPE = "MeasureObjectSizeShape"
CONVERT_OBJECTS_TO_IMAGE = "ConvertObjectsToImage"
OPENING = "Opening"
OVERLAY_OUTLINES = "OverlayOutlines"


class MemoryBackend:
    def __init__(self):
        self._memory_store = {}


class FileManagerStub:
    def __init__(self):
        self.memory = MemoryBackend()
        self.saved = {}
        self.loaded = []
        self.directories = set()

    def _get_backend(self, backend):
        return self.memory

    def ensure_directory(self, path, backend):
        self.directories.add((path, backend))

    def save(self, value, path, backend):
        self.saved[(path, backend)] = value
        self.memory._memory_store[path] = value

    def load(self, path, backend):
        self.loaded.append((path, backend))
        return self.memory._memory_store[path]


class ContextStub:
    def __init__(self):
        self.axis_id = AXIS_ID
        self.filemanager = FileManagerStub()
        self.runtime_value_store = RuntimeValueStore()


def _module(module_num: int, name: str, settings: dict[str, str]) -> ModuleBlock:
    return ModuleBlock(name=name, module_num=module_num, settings=settings)


def _generated_pipeline(modules: list[ModuleBlock]) -> GeneratedPipeline:
    return PipelineGenerator().generate_from_registry(
        pipeline_name="cellprofiler_generated_runtime_smoke",
        source_cppipe=Path("cellprofiler_generated_runtime_smoke.cppipe"),
        modules=modules,
    )


def _pipeline_namespace(generated: GeneratedPipeline) -> dict:
    namespace: dict = {}
    exec(
        compile(generated.code, "<generated-cellprofiler-pipeline>", "exec"),
        namespace,
    )
    return namespace


def _synthetic_nuclei_image() -> np.ndarray:
    image = np.zeros((64, 64), dtype=np.float32)
    image[18:28, 18:28] = 0.95
    image[40:50, 40:50] = 0.85
    return image


def _artifact_output_plans(contract) -> dict[str, ArtifactOutputPlan]:
    return {
        spec.name: ArtifactOutputPlan(
            name=spec.name,
            path=_artifact_path(spec.name),
            kind=spec.kind,
        )
        for spec in contract.outputs
    }


def _artifact_input_plans(contract) -> dict[str, ArtifactInputPlan]:
    return {
        spec.name: ArtifactInputPlan(
            name=spec.name,
            path=_artifact_path(spec.name),
            kind=spec.kind,
        )
        for spec in contract.runtime_artifact_inputs
    }


def _artifact_path(name: str) -> str:
    return f"/memory/{name}.pkl"


def _step_function_and_kwargs(step) -> tuple:
    if isinstance(step.func, tuple):
        return step.func[0], dict(step.func[1])
    return step.func, {}


def _run_generated_step(step, contract, image, context):
    func, kwargs = _step_function_and_kwargs(step)
    kwargs["dtype_config"] = DtypeConfig()
    return _execute_function_core(
        FunctionExecutionRequest(
            func_callable=func,
            main_data_arg=image,
            base_kwargs=kwargs,
            context=context,
            artifact_inputs=_artifact_input_plans(contract),
            artifact_outputs=_artifact_output_plans(contract),
            runtime_adapter=runtime_adapter_spec_from_callable(func),
            source_binding_plan=CompiledSourceBindingPlan.from_config(
                step.source_bindings
            ),
        )
    )


def _measurement_pipeline_modules() -> list[ModuleBlock]:
    return [
        _module(
            1,
            IDENTIFY_PRIMARY_OBJECTS,
            {
                "Select the input image": SOURCE_IMAGE,
                "Name the primary objects to be identified": NUCLEI,
            },
        ),
        _module(
            2,
            MEASURE_OBJECT_SIZE_SHAPE,
            {"Select object sets to measure": NUCLEI},
        ),
    ]


def _image_artifact_pipeline_modules() -> list[ModuleBlock]:
    return [
        _module(
            1,
            IDENTIFY_PRIMARY_OBJECTS,
            {
                "Select the input image": SOURCE_IMAGE,
                "Name the primary objects to be identified": NUCLEI,
            },
        ),
        _module(
            2,
            CONVERT_OBJECTS_TO_IMAGE,
            {
                "Select the input objects": NUCLEI,
                "Name the output image": NUCLEI_IMAGE,
            },
        ),
        _module(
            3,
            OPENING,
            {
                "Select the input image": NUCLEI_IMAGE,
                "Name the output image": OPENED_NUCLEI_IMAGE,
                "Size": "2",
            },
        ),
        _module(
            4,
            OVERLAY_OUTLINES,
            {
                "Select image on which to display outlines": OPENED_NUCLEI_IMAGE,
                "Select objects to display": NUCLEI,
                "Name the output image": OVERLAY_IMAGE,
            },
        ),
    ]


def test_generated_cellprofiler_pipeline_executes_runtime_artifact_flow():
    generated = _generated_pipeline(_measurement_pipeline_modules())
    namespace = _pipeline_namespace(generated)
    context = ContextStub()
    image = _synthetic_nuclei_image()

    for step, contract in zip(
        namespace["pipeline_steps"],
        generated.artifact_contracts,
        strict=True,
    ):
        image = _run_generated_step(step, contract, image, context)

    nuclei_records = context.runtime_value_store.find(
        name=NUCLEI,
        kind=ArtifactKind.OBJECT_LABELS,
        axis_id=AXIS_ID,
    )
    measurement_name = generated.artifact_contracts[1].outputs[0].name
    measurement_records = context.runtime_value_store.find(
        name=measurement_name,
        kind=ArtifactKind.MEASUREMENTS,
        axis_id=AXIS_ID,
    )

    assert len(nuclei_records) == 1
    assert nuclei_records[0].value.data.max() == 2
    assert len(measurement_records) == 1
    assert measurement_records[0].value.schema.object_name == NUCLEI
    assert len(measurement_records[0].value.data) == 2
    assert context.filemanager.loaded == []


def test_generated_cellprofiler_pipeline_executes_runtime_image_artifact_flow():
    generated = _generated_pipeline(_image_artifact_pipeline_modules())
    namespace = _pipeline_namespace(generated)
    context = ContextStub()
    image = _synthetic_nuclei_image()

    for step, contract in zip(
        namespace["pipeline_steps"],
        generated.artifact_contracts,
        strict=True,
    ):
        image = _run_generated_step(step, contract, image, context)

    nuclei_image_records = context.runtime_value_store.find(
        name=NUCLEI_IMAGE,
        kind=ArtifactKind.IMAGE,
        axis_id=AXIS_ID,
    )
    opened_image_records = context.runtime_value_store.find(
        name=OPENED_NUCLEI_IMAGE,
        kind=ArtifactKind.IMAGE,
        axis_id=AXIS_ID,
    )
    overlay_image_records = context.runtime_value_store.find(
        name=OVERLAY_IMAGE,
        kind=ArtifactKind.IMAGE,
        axis_id=AXIS_ID,
    )

    assert len(nuclei_image_records) == 1
    assert nuclei_image_records[0].value.schema.source_image_name == SOURCE_IMAGE
    assert len(opened_image_records) == 1
    assert opened_image_records[0].value.schema.source_image_name == SOURCE_IMAGE
    assert len(overlay_image_records) == 1
    assert overlay_image_records[0].value.schema.source_image_name == SOURCE_IMAGE
    assert overlay_image_records[0].value.data.shape[-1] == 3
