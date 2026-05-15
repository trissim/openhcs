import json
from pathlib import Path
from types import SimpleNamespace

import imageio.v3 as imageio
import numpy as np
import tifffile

from openhcs.interop.cellprofiler.runtime import (
    CellProfilerModuleContractRegistry,
    cellprofiler_runtime_adapter_factory,
)
from openhcs.interop.cellprofiler.runtime.generated_pipeline import (
    bind_generated_pipeline_runtime,
    GeneratedPipelineContractSidecar,
    materialize_generated_pipeline_import_module,
)
from openhcs.interop.cellprofiler.parser import ModuleBlock
from openhcs.interop.cellprofiler.pipeline_generator import GeneratedPipeline, PipelineGenerator
from openhcs.constants import Backend
from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactKind,
    ArtifactOutputPlan,
    ArtifactSpec,
)
from openhcs.core.config import DtypeConfig
from openhcs.core.runtime_adapters import runtime_adapter_spec_from_callable
from openhcs.core.runtime_stores import RuntimeValueStore
from openhcs.core.runtime_semantics import parent_child_relationship_artifact_name
from openhcs.core.source_bindings import (
    ComponentSelector,
    CompiledSourceBindingPlan,
    GroupedSourceBindings,
    NamedSourceBinding,
    SourceBindingRuntimeContext,
    SourceSelector,
    StepSourceBindingsConfig,
)
from openhcs.core.runtime_adapters import runtime_adapter
from openhcs.core.runtime_values import (
    ImagePayloadMetadata,
    ObjectLabelSet,
    ObjectRelationship,
    image_payload_with_context,
)
from openhcs.core.steps.function_runtime import (
    FunctionExecutionRequest,
    _execute_function_core,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.microscopes.imagexpress import ImageXpressFilenameParser
from openhcs.constants.constants import AllComponents, GroupBy, VariableComponents


AXIS_ID = "A01"
SOURCE_IMAGE = "OrigBlue"
NUCLEI = "Nuclei"
CELLS = "Cells"
NUCLEI_IMAGE = "NucleiImage"
OPENED_NUCLEI_IMAGE = "OpenedNucleiImage"
OVERLAY_IMAGE = "OverlayImage"
COLOR_IMAGE = "ColorImage"
IDENTIFY_PRIMARY_OBJECTS = "IdentifyPrimaryObjects"
IDENTIFY_SECONDARY_OBJECTS = "IdentifySecondaryObjects"
MEASURE_OBJECT_SIZE_SHAPE = "MeasureObjectSizeShape"
THRESHOLD = "Threshold"
CONVERT_OBJECTS_TO_IMAGE = "ConvertObjectsToImage"
OPENING = "Opening"
EROSION = "Erosion"
OVERLAY_OUTLINES = "OverlayOutlines"
GRAY_TO_COLOR = "GrayToColor"
RELATE_OBJECTS = "RelateObjects"
MASK_OBJECTS = "MaskObjects"
MASKED_NUCLEI = "MaskedNuclei"
FILTER_OBJECTS = "FilterObjects"
FILTERED_NUCLEI = "FilteredNuclei"
FILTERED_CELLS = "FilteredCells"
UNTANGLE_WORMS = "UntangleWorms"


def _object_labels(record):
    return ObjectLabelSet.from_runtime_value(record.value).labels


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

    def exists(self, path, backend):
        return path in self.memory._memory_store

    def delete(self, path, backend):
        del self.memory._memory_store[path]
        self.saved.pop((path, backend), None)

    def load(self, path, backend):
        self.loaded.append((path, backend))
        return self.memory._memory_store[path]


class ContextStub:
    def __init__(self):
        self.axis_id = AXIS_ID
        self.filemanager = FileManagerStub()
        self.runtime_value_store = RuntimeValueStore()
        self.input_dir = "/plate/Images"
        self.global_config = SimpleNamespace(zarr_config=None)
        self.microscope_handler = SimpleNamespace(
            parser=ImageXpressFilenameParser(),
            get_primary_backend=lambda plate_path, filemanager: "memory",
        )


def _module(module_num: int, name: str, settings: dict[str, str]) -> ModuleBlock:
    return ModuleBlock(name=name, module_num=module_num, settings=settings)


def _module_from_cppipe(
    module_num: int,
    name: str,
    settings: dict[str, str],
    cppipe_path: Path,
) -> ModuleBlock:
    return ModuleBlock(
        name=name,
        module_num=module_num,
        settings=settings,
        cppipe_path=cppipe_path,
    )


def _generated_pipeline(
    modules: list[ModuleBlock],
    *,
    prune_dead_unmaterialized_artifact_steps: bool = False,
) -> GeneratedPipeline:
    return PipelineGenerator().generate_from_registry(
        pipeline_name="cellprofiler_generated_runtime_smoke",
        source_cppipe=Path("cellprofiler_generated_runtime_smoke.cppipe"),
        modules=modules,
        prune_dead_unmaterialized_artifact_steps=(
            prune_dead_unmaterialized_artifact_steps
        ),
    )


def _pipeline_namespace(generated: GeneratedPipeline) -> dict:
    namespace: dict = {"__name__": "test_generated_cellprofiler_pipeline"}
    runtime_contracts = generated.runtime_module_contracts_by_module_num
    CellProfilerModuleContractRegistry.register(
        namespace["__name__"],
        runtime_contracts,
    )
    exec(
        compile(generated.code, "<generated-cellprofiler-pipeline>", "exec"),
        namespace,
    )
    bind_generated_pipeline_runtime(SimpleNamespace(**namespace), runtime_contracts)
    return namespace


def test_generated_pipeline_save_can_use_explicit_filemanager_vfs(
    tmp_path: Path,
) -> None:
    generated = GeneratedPipeline(
        name="Example",
        code="# generated",
        source_cppipe="example.cppipe",
        converted_modules=[],
        failed_modules=[],
    )
    filemanager = FileManagerStub()
    output_path = tmp_path / "generated.py"

    generated.save(output_path, filemanager=filemanager, backend=Backend.MEMORY)

    assert filemanager.directories == {(str(tmp_path), "memory")}
    assert filemanager.saved[(str(output_path), "memory")] == "# generated"
    assert not output_path.exists()


def test_materialized_generated_pipeline_contract_sidecar_is_versioned_json(
    tmp_path: Path,
) -> None:
    generated = _generated_pipeline(_image_artifact_pipeline_modules())
    module_name = "test_generated_contract_sidecar"

    import_module_path = materialize_generated_pipeline_import_module(
        generated.code,
        module_name=module_name,
        output_dir=tmp_path,
        artifact_contracts=generated.runtime_module_contracts_by_module_num,
    )

    sidecar_path = tmp_path / f"{module_name}.cellprofiler_contracts.json"
    assert sidecar_path.exists()
    assert not (tmp_path / f"{module_name}.cellprofiler_contracts.pkl").exists()

    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "openhcs.cellprofiler.generated_contracts"
    assert payload["version"] == 1
    assert [contract["module_num"] for contract in payload["contracts"]] == [
        1,
        2,
        3,
        4,
    ]

    restored = GeneratedPipelineContractSidecar.read(sidecar_path)
    assert restored[3].module_name == "Opening"
    assert restored[3].outputs[0].name == OPENED_NUCLEI_IMAGE
    assert "GeneratedPipelineContractSidecar" in import_module_path.read_text(
        encoding="utf-8"
    )


def _synthetic_nuclei_image() -> np.ndarray:
    image = np.zeros((64, 64), dtype=np.float32)
    image[18:28, 18:28] = 0.95
    image[40:50, 40:50] = 0.85
    return image


def test_generator_uses_absorbed_function_contract_for_unknown_registry_contract():
    generated = _generated_pipeline(_image_artifact_pipeline_modules())

    assert "from openhcs.processing.backends.cellprofiler import (" in generated.code
    assert "opening," in generated.code


def test_generator_scopes_artifact_managed_callables_to_pattern_group():
    generated = _generated_pipeline(_image_artifact_pipeline_modules())

    assert "CellProfilerModuleRuntimeBinding" not in generated.code
    assert "func=(opening," in generated.code


def test_generator_scopes_runtime_artifact_only_step_once_per_axis():
    generated = _generated_pipeline(_measurement_pipeline_modules())
    namespace = _pipeline_namespace(generated)
    measurement_step = namespace["pipeline_steps"][1]

    assert measurement_step.name == MEASURE_OBJECT_SIZE_SHAPE
    assert measurement_step.processing_config.variable_components == [
        VariableComponents.SITE
    ]
    assert measurement_step.processing_config.group_by is GroupBy.NONE


def test_generator_binds_canonical_morphology_alias_structuring_element():
    generated = _generated_pipeline(
        [
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
                EROSION,
                {
                    "Select the input image": NUCLEI_IMAGE,
                    "Name the output image": "ErodedNucleiImage",
                    "Structuring element": "disk,5",
                },
            ),
        ]
    )

    assert "erode_image," in generated.code
    assert "CellProfilerModuleRuntimeBinding" not in generated.code
    assert "'structuring_element': 'disk'" in generated.code
    assert "'size': 5" in generated.code


def test_generator_binds_untangle_worms_overlap_style():
    generated = _generated_pipeline(
        [
            _module(
                1,
                UNTANGLE_WORMS,
                {
                    "Select the input binary image": SOURCE_IMAGE,
                    "Overlap style": "Both",
                    "Name the output overlapping worm objects": "OverlappingWorms",
                    "Name the output non-overlapping worm objects": (
                        "NonOverlappingWorms"
                    ),
                },
            ),
        ]
    )

    assert "'overlap_style': 'both'" in generated.code


def test_generator_binds_untangle_worms_training_xml(tmp_path: Path):
    images = tmp_path / "images"
    images.mkdir()
    (images / "WormModel.xml").write_text(
        """<?xml version="1.0"?>
<training-set>
  <min-area>12.5</min-area>
  <max-area>30.0</max-area>
  <cost-threshold>4.5</cost-threshold>
  <num-control-points>9</num-control-points>
  <min-path-length>6.5</min-path-length>
  <max-path-length>70.0</max-path-length>
  <overlap-weight>2.0</overlap-weight>
  <leftover-weight>8.0</leftover-weight>
</training-set>
""",
        encoding="utf-8",
    )

    generated = PipelineGenerator().generate_from_registry(
        pipeline_name="cellprofiler_generated_runtime_smoke",
        source_cppipe=tmp_path / "worm.cppipe",
        modules=[
            _module_from_cppipe(
                1,
                UNTANGLE_WORMS,
                {
                    "Select the input binary image": SOURCE_IMAGE,
                    "Overlap style": "Both",
                    "Name the output overlapping worm objects": "OverlappingWorms",
                    "Name the output non-overlapping worm objects": (
                        "NonOverlappingWorms"
                    ),
                    "Training set file name": "WormModel.xml",
                    "Number of control points": "21",
                },
                tmp_path / "worm.cppipe",
            )
        ],
    )

    assert "'min_worm_area': 12.5" in generated.code
    assert "'max_worm_area': 30.0" in generated.code
    assert "'num_control_points': 9" in generated.code


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


def _run_generated_step(
    step,
    contract,
    image,
    context,
    *,
    source_binding_context=SourceBindingRuntimeContext.empty(),
):
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
            source_binding_context=source_binding_context,
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


def _gray_to_color_pipeline_modules() -> list[ModuleBlock]:
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
            GRAY_TO_COLOR,
            {
                "Select a color scheme": "RGB",
                "Select the image to be colored red": "Leave this black",
                "Select the image to be colored green": NUCLEI_IMAGE,
                "Select the image to be colored blue": SOURCE_IMAGE,
                "Name the output image": COLOR_IMAGE,
                "Relative weight for the red image": "1.0",
                "Relative weight for the green image": "1.0",
                "Relative weight for the blue image": "1.0",
            },
        ),
    ]


def _relationship_pipeline_modules() -> list[ModuleBlock]:
    return [
        _module(
            1,
            IDENTIFY_PRIMARY_OBJECTS,
            {
                "Select the input image": "OrigBlue",
                "Name the primary objects to be identified": NUCLEI,
            },
        ),
        _module(
            2,
            IDENTIFY_SECONDARY_OBJECTS,
            {
                "Select the input objects": NUCLEI,
                "Select the input image": "OrigGreen",
                "Name the objects to be identified": CELLS,
                "Name the new primary objects": "FilteredNuclei",
            },
        ),
        _module(
            3,
            RELATE_OBJECTS,
            {
                "Select the parent objects": CELLS,
                "Select the child objects": NUCLEI,
            },
        ),
    ]


def _mask_objects_pipeline_modules() -> list[ModuleBlock]:
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
            MASK_OBJECTS,
            {
                "Select the input objects": NUCLEI,
                "Select the masking image": SOURCE_IMAGE,
                "Name the output objects": MASKED_NUCLEI,
                "Handling of objects that are partially masked": (
                    "Keep overlapping region"
                ),
            },
        ),
    ]


def _filter_objects_pipeline_modules() -> list[ModuleBlock]:
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
            IDENTIFY_PRIMARY_OBJECTS,
            {
                "Select the input image": SOURCE_IMAGE,
                "Name the primary objects to be identified": CELLS,
            },
        ),
        _module(
            3,
            FILTER_OBJECTS,
            {
                "Name the output objects": FILTERED_NUCLEI,
                "Select the object to filter": NUCLEI,
                "Filter using classifier rules or measurements?": "Measurements",
                "Select the filtering method": "Limits",
                "Filter using a minimum measurement value?": "No",
                "Filter using a maximum measurement value?": "No",
                "Select additional object to relabel": CELLS,
                "Name the relabeled objects": FILTERED_CELLS,
                "Save outlines of relabeled objects?": "No",
            },
        ),
    ]


def _filter_objects_measurement_pipeline_modules() -> list[ModuleBlock]:
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
        _module(
            3,
            FILTER_OBJECTS,
            {
                "Name the output objects": FILTERED_NUCLEI,
                "Select the object to filter": NUCLEI,
                "Filter using classifier rules or measurements?": "Measurements",
                "Select the filtering method": "Limits",
                "Select the measurement to filter by": "AreaShape_Area",
                "Filter using a minimum measurement value?": "Yes",
                "Minimum value": "200",
                "Filter using a maximum measurement value?": "No",
                "Maximum value": "10000",
            },
        ),
    ]


def _single_channel_source_binding_context() -> SourceBindingRuntimeContext:
    return SourceBindingRuntimeContext(
        step_input_files=("A01_s001_w1_z001_t001.tif",)
    )


def test_generated_cellprofiler_pipeline_executes_runtime_artifact_flow():
    generated = _generated_pipeline(_measurement_pipeline_modules())
    namespace = _pipeline_namespace(generated)
    context = ContextStub()
    image = _synthetic_nuclei_image()
    source_binding_context = _single_channel_source_binding_context()

    for step, contract in zip(
        namespace["pipeline_steps"],
        generated.artifact_contracts,
        strict=True,
    ):
        image = _run_generated_step(
            step,
            contract,
            image,
            context,
            source_binding_context=source_binding_context,
        )

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
    assert _object_labels(nuclei_records[0]).max() == 2
    assert len(measurement_records) == 1
    assert measurement_records[0].value.schema.object_name == NUCLEI
    assert len(measurement_records[0].value.data) == 2
    assert context.filemanager.loaded == []


def test_generated_cellprofiler_pipeline_executes_runtime_image_artifact_flow():
    generated = _generated_pipeline(_image_artifact_pipeline_modules())
    namespace = _pipeline_namespace(generated)
    context = ContextStub()
    image = _synthetic_nuclei_image()
    source_binding_context = _single_channel_source_binding_context()

    for step, contract in zip(
        namespace["pipeline_steps"],
        generated.artifact_contracts,
        strict=True,
    ):
        image = _run_generated_step(
            step,
            contract,
            image,
            context,
            source_binding_context=source_binding_context,
        )

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


def test_generator_prunes_dead_unmaterialized_image_artifacts_when_requested():
    generated = _generated_pipeline(
        _image_artifact_pipeline_modules(),
        prune_dead_unmaterialized_artifact_steps=True,
    )

    assert 'name="IdentifyPrimaryObjects"' in generated.code
    assert 'name="ConvertObjectsToImage"' not in generated.code
    assert 'name="Opening"' not in generated.code
    assert 'name="OverlayOutlines"' not in generated.code
    assert [contract.module_name for contract in generated.artifact_contracts] == [
        IDENTIFY_PRIMARY_OBJECTS
    ]


def test_generator_prunes_dead_outputs_from_retained_modules():
    generated = _generated_pipeline(
        [
            _module(
                1,
                THRESHOLD,
                {
                    "Select the input image": SOURCE_IMAGE,
                    "Name the output image": "UnusedThresholdImage",
                },
            ),
        ],
        prune_dead_unmaterialized_artifact_steps=True,
    )

    (contract,) = generated.artifact_contracts

    assert contract.module_name == THRESHOLD
    assert [output.kind for output in contract.outputs] == [ArtifactKind.MEASUREMENTS]
    assert "image:UnusedThresholdImage" not in generated.code


def test_generator_retains_observable_object_label_outputs_when_pruning():
    generated = _generated_pipeline(
        _filter_objects_measurement_pipeline_modules(),
        prune_dead_unmaterialized_artifact_steps=True,
    )

    filter_contract = next(
        contract
        for contract in generated.artifact_contracts
        if contract.module_name == FILTER_OBJECTS
    )

    assert 'name="FilterObjects"' in generated.code
    assert ArtifactSpec(FILTERED_NUCLEI, ArtifactKind.OBJECT_LABELS) in (
        filter_contract.outputs
    )


def test_generator_keeps_unmaterialized_image_artifacts_required_by_saveimages():
    generated = PipelineGenerator().generate_from_registry(
        pipeline_name="cellprofiler_generated_runtime_smoke",
        source_cppipe=Path("cellprofiler_generated_runtime_smoke.cppipe"),
        modules=_image_artifact_pipeline_modules(),
        skipped_modules=[
            _module(
                5,
                "SaveImages",
                {"Select the image to save": OVERLAY_IMAGE},
            )
        ],
        prune_dead_unmaterialized_artifact_steps=True,
    )

    assert 'name="ConvertObjectsToImage"' in generated.code
    assert 'name="Opening"' in generated.code
    assert 'name="OverlayOutlines"' in generated.code
    overlay_contract = generated.runtime_module_contracts_by_module_num[4]
    assert overlay_contract.outputs[0].name == OVERLAY_IMAGE
    assert overlay_contract.outputs[0].materialization is not None
    assert "ArtifactSpec(" not in generated.code
    assert "tiff_stack(" not in generated.code
    assert "NO_ARTIFACT_MATERIALIZATION" not in generated.code
    assert [contract.module_name for contract in generated.artifact_contracts] == [
        IDENTIFY_PRIMARY_OBJECTS,
        CONVERT_OBJECTS_TO_IMAGE,
        OPENING,
        OVERLAY_OUTLINES,
    ]


def test_generator_can_ignore_saveimages_artifacts_for_value_only_runs():
    generated = PipelineGenerator().generate_from_registry(
        pipeline_name="cellprofiler_generated_runtime_smoke",
        source_cppipe=Path("cellprofiler_generated_runtime_smoke.cppipe"),
        modules=_image_artifact_pipeline_modules(),
        skipped_modules=[
            _module(
                5,
                "SaveImages",
                {"Select the image to save": OVERLAY_IMAGE},
            )
        ],
        prune_dead_unmaterialized_artifact_steps=True,
        materialize_skipped_save_images=False,
    )

    assert 'name="ConvertObjectsToImage"' not in generated.code
    assert 'name="Opening"' not in generated.code
    assert 'name="OverlayOutlines"' not in generated.code
    assert [contract.module_name for contract in generated.artifact_contracts] == [
        IDENTIFY_PRIMARY_OBJECTS
    ]


def test_generated_cellprofiler_pipeline_executes_gray_to_color_module():
    generated = _generated_pipeline(_gray_to_color_pipeline_modules())
    namespace = _pipeline_namespace(generated)
    context = ContextStub()
    image = _synthetic_nuclei_image()
    source_binding_context = _single_channel_source_binding_context()

    for step, contract in zip(
        namespace["pipeline_steps"],
        generated.artifact_contracts,
        strict=True,
    ):
        image = _run_generated_step(
            step,
            contract,
            image,
            context,
            source_binding_context=source_binding_context,
        )

    color_image_records = context.runtime_value_store.find(
        name=COLOR_IMAGE,
        kind=ArtifactKind.IMAGE,
        axis_id=AXIS_ID,
    )

    assert len(color_image_records) == 1
    assert color_image_records[0].value.schema.source_image_name == (
        f"{NUCLEI_IMAGE}__{SOURCE_IMAGE}"
    )
    assert color_image_records[0].value.data.shape == (64, 64, 3)
    assert image.shape == color_image_records[0].value.data.shape


def test_identify_primary_objects_uses_runtime_image_intensity_scale():
    from benchmark.cellprofiler_library.functions.identifyprimaryobjects import (
        UnclumpMethod,
        WatershedMethod,
        identify_primary_objects,
    )
    from openhcs.interop.cellprofiler.thresholding import (
        CellProfilerThresholdMethod as ThresholdMethod,
    )

    image = np.full((16, 16), 128, dtype=np.uint16)
    image[4:12, 4:12] = 512
    source_scaled_image = image_payload_with_context(
        image,
        metadata=ImagePayloadMetadata(
            intensity_scale=4095.0,
            source_dtype="uint16",
        ),
    )

    _raw_image, stats, labels = identify_primary_objects(
        source_scaled_image,
        min_diameter=2,
        max_diameter=20,
        exclude_size=False,
        exclude_border_objects=False,
        unclump_method=UnclumpMethod.NONE,
        watershed_method=WatershedMethod.NONE,
        use_advanced_settings=True,
        threshold_method=ThresholdMethod.OTSU,
        threshold_smoothing_scale=0.0,
        dtype_config=DtypeConfig(),
    )

    assert stats.object_count == 1
    assert labels.labels[8, 8] == 1


def test_runtime_image_metadata_uses_declared_tiff_intensity_scale(tmp_path):
    from openhcs.core.runtime_values import image_payload_metadata_from_source

    path = tmp_path / "source_12bit.tif"
    image = np.array([[0, 4095]], dtype=np.uint16)
    tifffile.imwrite(
        path,
        image,
        extratags=(
            (280, "H", 1, 0, False),
            (281, "H", 1, 4095, False),
        ),
    )
    readback = imageio.imread(path)

    metadata = image_payload_metadata_from_source(readback, source_path=str(path))

    assert metadata.source_dtype == "uint16"
    assert metadata.intensity_scale == 4095.0


def test_runtime_adapter_receives_step_input_source_binding_context():
    @runtime_adapter(
        "cellprofiler_runtime",
        cellprofiler_runtime_adapter_factory,
        manages_artifact_inputs=True,
    )
    def select_named_input(image, *, cellprofiler_runtime):
        return cellprofiler_runtime.resolve_source_image(SOURCE_IMAGE, image)

    context = ContextStub()
    input_stack = np.stack(
        [
            np.full((8, 8), 4.0, dtype=np.float32),
            np.full((8, 8), 2.0, dtype=np.float32),
        ]
    )
    source_binding_context = SourceBindingRuntimeContext(
        step_input_files=(
            "A01_s001_w1_z001_t001.tif",
            "A01_s001_w2_z001_t001.tif",
        )
    )
    source_binding_plan = CompiledSourceBindingPlan.from_config(
        StepSourceBindingsConfig(
            groups=(
                GroupedSourceBindings(
                    bindings=(
                        NamedSourceBinding(
                            alias=SOURCE_IMAGE,
                            selector=SourceSelector(
                                components=(
                                    ComponentSelector(AllComponents.CHANNEL, "1"),
                                ),
                            ),
                        ),
                    ),
                ),
            )
        )
    )
    selected_output = _execute_function_core(
        FunctionExecutionRequest(
            func_callable=select_named_input,
            main_data_arg=input_stack,
            base_kwargs={},
            context=context,
            artifact_inputs={},
            artifact_outputs={},
            runtime_adapter=runtime_adapter_spec_from_callable(select_named_input),
            source_binding_plan=source_binding_plan,
            source_binding_context=source_binding_context,
        )
    )

    assert selected_output.shape == (8, 8)


def test_generated_cellprofiler_pipeline_records_relationship_artifacts():
    generated = _generated_pipeline(_relationship_pipeline_modules())
    namespace = _pipeline_namespace(generated)
    context = ContextStub()
    input_stack = np.stack(
        [
            _synthetic_nuclei_image(),
            np.clip(_synthetic_nuclei_image() + 0.05, 0.0, 1.0),
        ]
    )
    source_binding_context = SourceBindingRuntimeContext(
        step_input_files=(
            "A01_s001_w1_z001_t001.tif",
            "A01_s001_w2_z001_t001.tif",
        )
    )

    image = input_stack
    for step, contract in zip(
        namespace["pipeline_steps"],
        generated.artifact_contracts,
        strict=True,
    ):
        image = _run_generated_step(
            step,
            contract,
            image,
            context,
            source_binding_context=source_binding_context,
        )

    relationship_name = generated.artifact_contracts[2].outputs[0].name
    measurement_name = generated.artifact_contracts[2].outputs[1].name
    relationship_records = context.runtime_value_store.find(
        name=relationship_name,
        kind=ArtifactKind.RELATIONSHIPS,
        axis_id=AXIS_ID,
    )
    measurement_records = context.runtime_value_store.find(
        name=measurement_name,
        kind=ArtifactKind.MEASUREMENTS,
        axis_id=AXIS_ID,
    )

    assert len(relationship_records) == 1
    assert relationship_records[0].value.schema.relationship is not None
    assert len(measurement_records) == 1


def test_generated_cellprofiler_pipeline_executes_generic_mask_objects_contract():
    generated = _generated_pipeline(_mask_objects_pipeline_modules())
    namespace = _pipeline_namespace(generated)
    context = ContextStub()
    image = _synthetic_nuclei_image()
    source_binding_context = _single_channel_source_binding_context()

    for step, contract in zip(
        namespace["pipeline_steps"],
        generated.artifact_contracts,
        strict=True,
    ):
        image = _run_generated_step(
            step,
            contract,
            image,
            context,
            source_binding_context=source_binding_context,
        )

    masked_records = context.runtime_value_store.find(
        name=MASKED_NUCLEI,
        kind=ArtifactKind.OBJECT_LABELS,
        axis_id=AXIS_ID,
    )
    measurement_records = context.runtime_value_store.find(
        name="MaskObjects_2_measurements",
        kind=ArtifactKind.MEASUREMENTS,
        axis_id=AXIS_ID,
    )
    relationship_records = context.runtime_value_store.find(
        name=f"{NUCLEI}_{MASKED_NUCLEI}_relationships",
        kind=ArtifactKind.RELATIONSHIPS,
        axis_id=AXIS_ID,
    )

    assert len(masked_records) == 1
    assert _object_labels(masked_records[0]).max() > 0
    assert len(measurement_records) == 1
    assert len(relationship_records) == 1
    assert relationship_records[0].value.schema.relationship is not None


def test_generated_cellprofiler_pipeline_executes_filterobjects_relabel_outputs():
    generated = _generated_pipeline(_filter_objects_pipeline_modules())
    namespace = _pipeline_namespace(generated)
    context = ContextStub()
    image = _synthetic_nuclei_image()
    source_binding_context = _single_channel_source_binding_context()

    for step, contract in zip(
        namespace["pipeline_steps"],
        generated.artifact_contracts,
        strict=True,
    ):
        image = _run_generated_step(
            step,
            contract,
            image,
            context,
            source_binding_context=source_binding_context,
        )

    filtered_nuclei_records = context.runtime_value_store.find(
        name=FILTERED_NUCLEI,
        kind=ArtifactKind.OBJECT_LABELS,
        axis_id=AXIS_ID,
    )
    filtered_cells_records = context.runtime_value_store.find(
        name=FILTERED_CELLS,
        kind=ArtifactKind.OBJECT_LABELS,
        axis_id=AXIS_ID,
    )
    measurement_records = context.runtime_value_store.find(
        name="FilterObjects_3_measurements",
        kind=ArtifactKind.MEASUREMENTS,
        axis_id=AXIS_ID,
    )
    relationship_records = context.runtime_value_store.find(
        name=parent_child_relationship_artifact_name(NUCLEI, FILTERED_NUCLEI),
        kind=ArtifactKind.RELATIONSHIPS,
        axis_id=AXIS_ID,
    )

    assert len(filtered_nuclei_records) == 1
    assert _object_labels(filtered_nuclei_records[0]).max() > 0
    assert len(filtered_cells_records) == 1
    assert _object_labels(filtered_cells_records[0]).max() > 0
    assert len(measurement_records) == 1
    assert len(relationship_records) == 1
    relationship = ObjectRelationship.from_runtime_value(relationship_records[0].value)
    assert relationship.source_ids
    assert relationship.target_ids


def test_generated_cellprofiler_pipeline_filters_objects_by_prior_measurements():
    generated = _generated_pipeline(_filter_objects_measurement_pipeline_modules())
    namespace = _pipeline_namespace(generated)
    context = ContextStub()
    image = _synthetic_nuclei_image()
    source_binding_context = _single_channel_source_binding_context()

    for step, contract in zip(
        namespace["pipeline_steps"],
        generated.artifact_contracts,
        strict=True,
    ):
        image = _run_generated_step(
            step,
            contract,
            image,
            context,
            source_binding_context=source_binding_context,
        )

    filtered_records = context.runtime_value_store.find(
        name=FILTERED_NUCLEI,
        kind=ArtifactKind.OBJECT_LABELS,
        axis_id=AXIS_ID,
    )
    relationship_records = context.runtime_value_store.find(
        name=parent_child_relationship_artifact_name(NUCLEI, FILTERED_NUCLEI),
        kind=ArtifactKind.RELATIONSHIPS,
        axis_id=AXIS_ID,
    )

    assert len(filtered_records) == 1
    assert _object_labels(filtered_records[0]).max() == 0
    assert len(relationship_records) == 1
    relationship = ObjectRelationship.from_runtime_value(relationship_records[0].value)
    assert tuple(relationship.source_ids) == ()
    assert tuple(relationship.target_ids) == ()
