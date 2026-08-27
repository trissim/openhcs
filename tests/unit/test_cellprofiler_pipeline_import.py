"""Focused public-boundary tests for direct CellProfiler import."""

from __future__ import annotations

import ast
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

import pytest
from objectstate import config_context

from openhcs.constants import AllComponents, Backend, GroupBy, VariableComponents
from openhcs.constants.input_source import InputSource
from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactSpec,
    ArtifactSpecCollection,
    ImageArtifactType,
    ObjectLabelsArtifactType,
)
from openhcs.core.callable_contract import CallableContract, FunctionStepExecutionScope
from openhcs.core.config import PipelineConfig, ProcessingConfig
from openhcs.core.function_patterns import (
    DEFAULT_GROUP_KEY,
    FunctionInvocationKey,
    normalize_function_pattern,
)
from openhcs.core.function_step_transport import FunctionStepTransportAuthority
from openhcs.core.invocation_artifacts import ArtifactDeclarationStepContext
from openhcs.core.pipeline.artifact_planning import (
    artifact_producers_for_outputs,
)
from openhcs.core.source_bindings import (
    ComponentSelector,
    NamedSourceBinding,
    SourceBindingMatchMethod,
    SourceProjectionRole,
    StepSourceBindingsConfig,
)
from openhcs.core.steps.function_step import FunctionStep
from openhcs.core.vfs_protocol import FileManagerLike
from openhcs.interop.cellprofiler.module_declarations import (
    CellProfilerModule,
)
from openhcs.interop.cellprofiler.parser import ModuleBlock, ModuleSetting
from openhcs.interop.cellprofiler.pipeline_import import (
    _ParsedTargetUnit,
    _public_kwargs_for_target,
    _public_step_source_bindings,
    _SelectedInputBindingOccurrence,
    import_cellprofiler_pipeline,
)
from openhcs.interop.cellprofiler.setting_names import (
    SettingNameFamily,
    setting_values,
)
from openhcs.processing.backends.cellprofiler.morphology import (
    dilate_objects_3d,
    erode_image,
    remove_holes_3d,
)
from openhcs.processing.backends.cellprofiler.neighbors import (
    DistanceMethod,
    MeasureObjectNeighborsModule,
    measure_object_neighbors,
)
from openhcs.processing.backends.cellprofiler.outlines import OverlayObjectsModule

PIPELINE_IMPORT_PATH = (
    Path(__file__).parents[2]
    / "openhcs"
    / "interop"
    / "cellprofiler"
    / "pipeline_import.py"
)


def _contract_for(
    func: Callable[..., object],
    *,
    artifact_inputs: tuple[ArtifactSpec, ...] = (),
    artifact_outputs: tuple[ArtifactSpec, ...] = (),
    execution_scope: FunctionStepExecutionScope | None = None,
) -> CallableContract:
    contract = CallableContract.from_callable(func)
    return replace(
        contract,
        metadata=replace(
            contract.metadata,
            artifact_inputs=artifact_inputs,
            artifact_outputs=artifact_outputs,
            execution_scope=(
                contract.execution_scope if execution_scope is None else execution_scope
            ),
        ),
    )


class _MemoryFileManager(FileManagerLike):
    def __init__(self, files: dict[Path, str]) -> None:
        self.files = dict(files)
        self.saved: list[tuple[Path, str, str]] = []

    def list_files(
        self, directory: str | Path, backend: str, **kwargs: object
    ) -> list[str]:
        del directory, backend, kwargs
        return []

    def exists(self, path: str | Path, backend: str) -> bool:
        del backend
        return Path(path) in self.files

    def is_dir(self, path: str | Path, backend: str) -> bool:
        del backend
        return Path(path) == Path("pipelines")

    def load(self, file_path: str | Path, backend: str, **kwargs: object) -> object:
        del backend, kwargs
        return self.files[Path(file_path)]

    def load_batch(
        self,
        file_paths: Sequence[str | Path],
        backend: str,
        **kwargs: object,
    ) -> tuple[object, ...]:
        return tuple(self.load(path, backend, **kwargs) for path in file_paths)

    def resolve_address(
        self,
        backend_address: str | Path,
        backend: str,
        *,
        base_path: str | Path,
    ) -> str | Path:
        del backend, base_path
        return Path(backend_address)

    physical_source_path = resolve_address

    def save(
        self,
        data: object,
        output_path: str | Path,
        backend: str,
        **kwargs: object,
    ) -> None:
        del kwargs
        assert isinstance(data, str)
        self.files[Path(output_path)] = data
        self.saved.append((Path(output_path), backend, data))

    def ensure_directory(self, directory: str | Path, backend: str) -> str:
        del backend
        return str(directory)


def test_direct_import_returns_public_steps_and_generic_pycodify_source() -> None:
    cppipe_path = Path("pipelines/direct.cppipe")
    filemanager = _MemoryFileManager(
        {cppipe_path: """CellProfiler Pipeline: https://cellprofiler.org
NamesAndTypes:[module_num:1|enabled:True]
    Assignments count:1
    Select the image type:Grayscale image
    Name to assign these images:DNA
    Select the rule criteria:and (file does contain \"DNA\")
MedianFilter:[module_num:2|enabled:True]
    Select the input image:DNA
    Name the output image:FilteredDNA
    Window:5
SaveImages:[module_num:3|enabled:True]
    Select the image to save:FilteredDNA
"""}
    )

    pipeline_steps, pipeline_config = import_cellprofiler_pipeline(
        cppipe_path,
        filemanager=filemanager,
        backend=Backend.MEMORY,
    )

    assert isinstance(pipeline_config, PipelineConfig)
    assert type(pipeline_steps) is list
    assert [step.name for step in pipeline_steps] == [
        "MedianFilter",
        "SaveImages",
    ]
    assert all(isinstance(step, FunctionStep) for step in pipeline_steps)
    assert pipeline_config.source_bindings_config.bindings[0].alias == "DNA"

    source = FunctionStepTransportAuthority.source_from_pipeline(pipeline_steps)
    assert filemanager.saved == []
    assert "pipeline_steps" in source
    assert "'select_the_image_to_save': 'FilteredDNA'" not in source
    assert "'image_to_save':" not in source
    namespace: dict[str, object] = {}
    exec(compile(source, "direct.py", "exec"), namespace)
    reconstructed_steps = FunctionStepTransportAuthority.pipeline_steps_from_namespace(
        namespace
    )
    assert [step.name for step in reconstructed_steps] == [
        "MedianFilter",
        "SaveImages",
    ]
    save_invocation = next(
        normalize_function_pattern(reconstructed_steps[-1].func).iter_items()
    )
    assert "select_the_image_to_save" not in save_invocation.kwargs_dict


def test_direct_import_rejects_interactive_manual_identification() -> None:
    cppipe_path = Path("pipelines/interactive.cppipe")
    filemanager = _MemoryFileManager(
        {
            cppipe_path: """CellProfiler Pipeline: https://cellprofiler.org
IdentifyObjectsManually:[module_num:1|enabled:True]
    Select the input image:DNA
    Name the objects to be identified:ManualObjects
"""
        }
    )

    with pytest.raises(ValueError, match="requires interactive desktop input"):
        import_cellprofiler_pipeline(
            cppipe_path,
            filemanager=filemanager,
            backend=Backend.MEMORY,
        )


def test_image_intensity_without_object_mask_lowers_to_image_callable(
    tmp_path: Path,
) -> None:
    cppipe_path = tmp_path / "image-intensity.cppipe"
    cppipe_path.write_text(
        """CellProfiler Pipeline: https://cellprofiler.org
NamesAndTypes:[module_num:1|enabled:True]
    Assignments count:1
    Select the image type:Grayscale image
    Name to assign these images:DNA
    Select the rule criteria:and (file does contain \"DNA\")
MeasureImageIntensity:[module_num:2|enabled:True]
    Select images to measure:DNA
    Measure the intensity only from areas enclosed by objects?:No
    Select input object sets:
    Calculate custom percentiles:No
    Specify percentiles to measure:10,90
""",
        encoding="utf-8",
    )

    steps, _pipeline_config = import_cellprofiler_pipeline(cppipe_path)

    assert [step.name for step in steps] == ["MeasureImageIntensity"]
    (invocation,) = tuple(normalize_function_pattern(steps[0].func).iter_items())
    assert invocation.key.function_name == "measure_image_intensity"
    assert invocation.kwargs_dict == {}


@pytest.mark.parametrize(
    ("module_name", "module_block", "expected_function_name"),
    (
        (
            "MaskObjects",
            """MaskObjects:[module_num:4|enabled:True]
    Select objects to be masked:Nuclei
    Name the masked objects:MaskedNuclei
    Mask using a region defined by other objects or by binary image?:Objects
    Select the masking object:MaskingObjects
    Select the masking image:None
    Handling of objects that are partially masked:Keep overlapping region
    Fraction of object that must overlap:0.5
    Numbering of resulting objects:Renumber
    Invert the mask?:No
""",
            "mask_objects",
        ),
        (
            "ResizeObjects",
            """ResizeObjects:[module_num:4|enabled:True]
    Select the input object:Nuclei
    Name the output object:ResizedNuclei
    Method:Factor
    X Factor:0.5
    Y Factor:0.5
    Z Factor:0.5
    Width (X):100
    Height (Y):100
    Planes (Z):10
    Select the image with the desired dimensions:None
""",
            "resize_objects_3d",
        ),
    ),
)
def test_direct_import_preserves_object_lineage_output_abi(
    module_name: str,
    module_block: str,
    expected_function_name: str,
) -> None:
    cppipe_path = Path(f"pipelines/{module_name}.cppipe")
    filemanager = _MemoryFileManager(
        {cppipe_path: f"""CellProfiler Pipeline: https://cellprofiler.org
NamesAndTypes:[module_num:1|enabled:True]
    Assignments count:1
    Select the image type:Grayscale image
    Name to assign these images:DNA
    Select the rule criteria:and (file does contain "DNA")
IdentifyPrimaryObjects:[module_num:2|enabled:True]
    Select the input image:DNA
    Name the primary objects to be identified:Nuclei
IdentifyPrimaryObjects:[module_num:3|enabled:True]
    Select the input image:DNA
    Name the primary objects to be identified:MaskingObjects
{module_block}"""}
    )

    pipeline_steps, pipeline_config = import_cellprofiler_pipeline(
        cppipe_path,
        filemanager=filemanager,
        backend=Backend.MEMORY,
    )

    assert pipeline_steps[-1].name == module_name
    invocation = next(normalize_function_pattern(pipeline_steps[-1].func).iter_items())
    assert invocation.key.function_name == expected_function_name


def test_direct_import_preserves_external_source_declarations_without_io() -> None:
    cppipe_path = Path("pipelines/external-metadata.cppipe")
    metadata_location = "/unavailable/source/metadata.csv"
    filemanager = _MemoryFileManager(
        {cppipe_path: f"""CellProfiler Pipeline: https://cellprofiler.org
Metadata:[module_num:1|enabled:True]
    Extract metadata?:Yes
    Metadata extraction method:Import from file
    Metadata file location:{metadata_location}
    Metadata file name:
    Match file and image metadata:[]
NamesAndTypes:[module_num:2|enabled:True]
    Assignments count:1
    Select the image type:Grayscale image
    Name to assign these images:DNA
    Select the rule criteria:and (file does contain "DNA")
MedianFilter:[module_num:3|enabled:True]
    Select the input image:DNA
    Name the output image:FilteredDNA
    Window:5
"""}
    )

    _steps, pipeline_config = import_cellprofiler_pipeline(
        cppipe_path,
        filemanager=filemanager,
        backend=Backend.MEMORY,
    )

    assert (
        pipeline_config.source_bindings_config.imported_metadata_tables[0].location
        == metadata_location
    )


def test_direct_import_resolves_measure_image_quality_all_loaded_images() -> None:
    cppipe_path = Path("pipelines/image-quality.cppipe")
    filemanager = _MemoryFileManager(
        {cppipe_path: """CellProfiler Pipeline: https://cellprofiler.org
NamesAndTypes:[module_num:1|enabled:True]
    Assignments count:2
    Select the image type:Grayscale image
    Name to assign these images:DNA
    Select the rule criteria:and (file does contain "DNA")
    Select the image type:Grayscale image
    Name to assign these images:RNA
    Select the rule criteria:and (file does contain "RNA")
MeasureImageQuality:[module_num:2|variable_revision_number:6|enabled:True]
    Calculate metrics for which images?:All loaded images
    Select the images to measure:
    Include the image rescaling value?:No
    Calculate blur metrics?:Yes
    Spatial scale for blur measurements:20
    Calculate saturation metrics?:No
    Calculate intensity metrics?:No
    Calculate thresholds?:No
"""}
    )

    pipeline_steps, pipeline_config = import_cellprofiler_pipeline(
        cppipe_path,
        filemanager=filemanager,
        backend=Backend.MEMORY,
    )

    assert tuple(
        binding.alias for binding in pipeline_config.source_bindings_config.bindings
    ) == ("DNA", "RNA")
    assert [step.name for step in pipeline_steps] == ["MeasureImageQuality"]
    invocation = next(normalize_function_pattern(pipeline_steps[0].func).iter_items())
    assert invocation.key.function_name == "measure_image_quality"
    assert "select_images_to_measure" not in invocation.kwargs_dict


def test_adjacent_save_images_with_distinct_exports_share_one_invocation_chain() -> (
    None
):
    cppipe_path = Path("pipelines/distinct-save-images.cppipe")
    filemanager = _MemoryFileManager(
        {cppipe_path: """CellProfiler Pipeline: https://cellprofiler.org
NamesAndTypes:[module_num:1|enabled:True]
    Assignments count:1
    Select the image type:Grayscale image
    Name to assign these images:DNA
    Select the rule criteria:and (file does contain "DNA")
MedianFilter:[module_num:2|enabled:True]
    Select the input image:DNA
    Name the output image:FilteredDNA
    Window:5
SaveImages:[module_num:3|enabled:True]
    Select the image to save:FilteredDNA
    Saved file format:tiff
SaveImages:[module_num:4|enabled:True]
    Select the image to save:FilteredDNA
    Saved file format:png
"""}
    )

    pipeline_steps, pipeline_config = import_cellprofiler_pipeline(
        cppipe_path,
        filemanager=filemanager,
        backend=Backend.MEMORY,
    )

    assert [step.name for step in pipeline_steps] == ["MedianFilter", "SaveImages"]
    save_invocations = tuple(
        invocation
        for step in pipeline_steps[1:]
        for invocation in normalize_function_pattern(step.func).iter_items()
    )
    assert len(save_invocations) == 2
    assert save_invocations[0].kwargs_dict == {}
    assert save_invocations[1].kwargs_dict["file_format"].value == ".png"


def test_adjacent_save_images_retain_exact_distinct_runtime_image_identities() -> None:
    cppipe_path = Path("pipelines/distinct-save-images-inputs.cppipe")
    filemanager = _MemoryFileManager(
        {cppipe_path: """CellProfiler Pipeline: https://cellprofiler.org
NamesAndTypes:[module_num:1|enabled:True]
    Assignments count:1
    Select the image type:Grayscale image
    Name to assign these images:DNA
    Select the rule criteria:and (file does contain "DNA")
MedianFilter:[module_num:2|enabled:True]
    Select the input image:DNA
    Name the output image:FilteredDNA
    Window:3
MedianFilter:[module_num:3|enabled:True]
    Select the input image:DNA
    Name the output image:AlternateDNA
    Window:5
SaveImages:[module_num:4|enabled:True]
    Select the image to save:FilteredDNA
    Saved file format:tiff
SaveImages:[module_num:5|enabled:True]
    Select the image to save:AlternateDNA
    Saved file format:png
"""}
    )

    pipeline_steps, pipeline_config = import_cellprofiler_pipeline(
        cppipe_path,
        filemanager=filemanager,
        backend=Backend.MEMORY,
    )
    selected_images = tuple(
        invocation.kwargs_dict["select_the_image_to_save"]
        for step in pipeline_steps
        if step.name == "SaveImages"
        for invocation in normalize_function_pattern(step.func).iter_items()
    )

    assert selected_images == ("FilteredDNA", "AlternateDNA")


def test_adjacent_same_module_output_to_input_lowers_as_separate_steps() -> None:
    cppipe_path = Path("pipelines/sequential-median-filters.cppipe")
    filemanager = _MemoryFileManager(
        {cppipe_path: """CellProfiler Pipeline: https://cellprofiler.org
NamesAndTypes:[module_num:1|enabled:True]
    Assignments count:1
    Select the image type:Grayscale image
    Name to assign these images:DNA
    Select the rule criteria:and (file does contain "DNA")
MedianFilter:[module_num:2|enabled:True]
    Select the input image:DNA
    Name the output image:FilteredDNA
    Window:3
MedianFilter:[module_num:3|enabled:True]
    Select the input image:FilteredDNA
    Name the output image:SmoothedDNA
    Window:5
"""}
    )

    pipeline_steps, _pipeline_config = import_cellprofiler_pipeline(
        cppipe_path,
        filemanager=filemanager,
        backend=Backend.MEMORY,
    )

    assert [step.name for step in pipeline_steps] == ["MedianFilter", "MedianFilter"]
    assert all(
        len(tuple(normalize_function_pattern(step.func).iter_items())) == 1
        for step in pipeline_steps
    )


def test_adjacent_same_module_independent_inputs_lower_as_one_exact_dict_step() -> None:
    cppipe_path = Path("pipelines/independent-median-filters.cppipe")
    filemanager = _MemoryFileManager(
        {cppipe_path: """CellProfiler Pipeline: https://cellprofiler.org
NamesAndTypes:[module_num:1|enabled:True]
    Assignments count:2
    Select the image type:Grayscale image
    Name to assign these images:DNA
    Select the rule criteria:and (file does contain "DNA")
    Select the image type:Grayscale image
    Name to assign these images:RNA
    Select the rule criteria:and (file does contain "RNA")
MedianFilter:[module_num:2|enabled:True]
    Select the input image:DNA
    Name the output image:FilteredDNA
    Window:3
MedianFilter:[module_num:3|enabled:True]
    Select the input image:RNA
    Name the output image:FilteredRNA
    Window:5
"""}
    )

    pipeline_steps, _pipeline_config = import_cellprofiler_pipeline(
        cppipe_path,
        filemanager=filemanager,
        backend=Backend.MEMORY,
    )

    assert [step.name for step in pipeline_steps] == ["MedianFilter"]
    pattern = normalize_function_pattern(pipeline_steps[0].func)
    assert pattern.is_grouped
    invocations = tuple(pattern.iter_items())
    assert tuple(invocation.key.group_key for invocation in invocations) == ("1", "2")
    assert tuple(invocation.kwargs_dict for invocation in invocations) == (
        {"name_the_output_image": "FilteredDNA"},
        {
            "window_size": 5,
            "name_the_output_image": "FilteredRNA",
        },
    )


def test_direct_import_preserves_3d_image_variant_for_derived_input() -> None:
    cppipe_path = Path("pipelines/volumetric.cppipe")
    filemanager = _MemoryFileManager(
        {cppipe_path: """CellProfiler Pipeline: https://cellprofiler.org
NamesAndTypes:[module_num:1|enabled:True]
    Assignments count:1
    Select the image type:Grayscale image
    Name to assign these images:DNA
    Select the rule criteria:and (file does contain \"DNA\")
    Process as 3D?:Yes
MedianFilter:[module_num:2|enabled:True]
    Select the input image:DNA
    Name the output image:FilteredDNA
    Window:5
RemoveHoles:[module_num:3|enabled:True]
    Select the input image:FilteredDNA
    Name the output image:FilledDNA
    Size of holes to fill:5
"""}
    )

    pipeline_steps, pipeline_config = import_cellprofiler_pipeline(
        cppipe_path,
        filemanager=filemanager,
        backend=Backend.MEMORY,
    )

    assert pipeline_config.source_bindings_config.source_stack_components == (
        AllComponents.Z_INDEX,
    )
    assert pipeline_config.processing_config.variable_components == [
        VariableComponents.Z_INDEX
    ]
    with config_context(pipeline_config):
        assert pipeline_steps[0].processing_config.variable_components == [
            VariableComponents.Z_INDEX
        ]
        assert (
            pipeline_steps[1].processing_config.input_source
            is InputSource.PREVIOUS_STEP
        )
    remove_holes_invocation = next(
        normalize_function_pattern(pipeline_steps[1].func).iter_items()
    )
    assert remove_holes_invocation.func is remove_holes_3d


def test_direct_import_preserves_3d_object_variant_for_derived_input() -> None:
    cppipe_path = Path("pipelines/volumetric-objects.cppipe")
    filemanager = _MemoryFileManager(
        {cppipe_path: """CellProfiler Pipeline: https://cellprofiler.org
NamesAndTypes:[module_num:1|enabled:True]
    Assignments count:1
    Select the image type:Grayscale image
    Name to assign these images:DNA
    Select the rule criteria:and (file does contain \"DNA\")
    Process as 3D?:Yes
IdentifyPrimaryObjects:[module_num:2|enabled:True]
    Select the input image:DNA
    Name the primary objects to be identified:Nuclei
DilateObjects:[module_num:3|enabled:True]
    Select the input objects:Nuclei
    Name the output objects:DilatedNuclei
    Structuring element:Ball,1
"""}
    )

    pipeline_steps, pipeline_config = import_cellprofiler_pipeline(
        cppipe_path,
        filemanager=filemanager,
        backend=Backend.MEMORY,
    )

    assert [step.name for step in pipeline_steps] == [
        "IdentifyPrimaryObjects",
        "DilateObjects",
    ]
    with config_context(pipeline_config):
        assert (
            pipeline_steps[1].processing_config.input_source
            is InputSource.PREVIOUS_STEP
        )
    dilate_invocation = next(
        normalize_function_pattern(pipeline_steps[1].func).iter_items()
    )
    assert dilate_invocation.func is dilate_objects_3d


def test_direct_import_derives_image_morphology_execution_from_footprint() -> None:
    cppipe_path = Path("pipelines/volumetric-image-morphology.cppipe")
    filemanager = _MemoryFileManager(
        {cppipe_path: """CellProfiler Pipeline: https://cellprofiler.org
NamesAndTypes:[module_num:1|enabled:True]
    Assignments count:1
    Select the image type:Grayscale image
    Name to assign these images:DNA
    Select the rule criteria:and (file does contain \"DNA\")
    Process as 3D?:Yes
ErodeImage:[module_num:2|enabled:True]
    Select the input image:DNA
    Name the output image:ErodedDNA
    Structuring element:Ball,1
"""}
    )

    pipeline_steps, _pipeline_config = import_cellprofiler_pipeline(
        cppipe_path,
        filemanager=filemanager,
        backend=Backend.MEMORY,
    )

    assert [step.name for step in pipeline_steps] == ["ErodeImage"]
    invocation = next(normalize_function_pattern(pipeline_steps[0].func).iter_items())
    assert invocation.func is erode_image
    assert invocation.kwargs_dict["slice_by_slice"] is False


def test_required_module_axis_is_local_to_its_step() -> None:
    cppipe_path = Path("pipelines/tracking.cppipe")
    filemanager = _MemoryFileManager(
        {cppipe_path: """CellProfiler Pipeline: https://cellprofiler.org
NamesAndTypes:[module_num:1|enabled:True]
    Assignments count:1
    Select the image type:Objects
    Name to assign these objects:Cells
    Select the rule criteria:and (file does contain "Cells")
    Process as 3D?:No
TrackObjects:[module_num:2|enabled:True]
    Choose a tracking method:Overlap
    Select the objects to track:Cells
    Save color-coded image?:Yes
    Name the output image:TrackedNuclei
SaveImages:[module_num:3|enabled:True]
    Select the image to save:TrackedNuclei
    Saved file format:tiff
SaveImages:[module_num:4|enabled:True]
    Select the image to save:TrackedNuclei
    Saved file format:png
"""}
    )

    pipeline_steps, pipeline_config = import_cellprofiler_pipeline(
        cppipe_path,
        filemanager=filemanager,
        backend=Backend.MEMORY,
    )

    assert pipeline_config.processing_config.variable_components == [
        VariableComponents.SITE
    ]
    assert [step.name for step in pipeline_steps] == ["TrackObjects", "SaveImages"]
    track_invocation = next(
        normalize_function_pattern(pipeline_steps[0].func).iter_items()
    )
    assert track_invocation.kwargs_dict["save_color_coded_image"] is True
    assert track_invocation.kwargs_dict["name_the_output_image"] == "TrackedNuclei"
    with config_context(pipeline_config):
        assert pipeline_steps[0].processing_config.variable_components == [
            VariableComponents.TIMEPOINT
        ]
        assert pipeline_steps[1].processing_config.variable_components == [
            VariableComponents.SITE
        ]
    assert (
        len(tuple(normalize_function_pattern(pipeline_steps[1].func).iter_items())) == 2
    )


def test_grouped_source_pipeline_inherits_declared_tracking_axis() -> None:
    cppipe_path = Path("pipelines/grouped-tracking.cppipe")
    filemanager = _MemoryFileManager(
        {cppipe_path: """CellProfiler Pipeline: https://cellprofiler.org
Metadata:[module_num:1|enabled:True]
    Extract metadata?:Yes
    Metadata extraction method:Extract from file/folder names
    Metadata source:File name
    Regular expression to extract from file name:^(?P<FrameNumber>[0-9]+)
    Extract metadata from:All images
NamesAndTypes:[module_num:2|enabled:True]
    Assignments count:1
    Select the image type:Objects
    Name to assign these objects:Cells
    Select the rule criteria:and (file does contain "Cells")
    Process as 3D?:No
Groups:[module_num:3|enabled:True]
    Do you want to group your images?:Yes
    grouping metadata count:1
    Metadata category:Run
TrackObjects:[module_num:4|enabled:True]
    Choose a tracking method:Overlap
    Select the objects to track:Cells
    Save color-coded image?:Yes
    Name the output image:TrackedCells
SaveImages:[module_num:5|enabled:True]
    Select the image to save:TrackedCells
    Saved file format:tiff
"""}
    )

    pipeline_steps, pipeline_config = import_cellprofiler_pipeline(
        cppipe_path,
        filemanager=filemanager,
        backend=Backend.MEMORY,
    )

    assert pipeline_config.processing_config.variable_components == [
        VariableComponents.TIMEPOINT
    ]
    with config_context(pipeline_config):
        assert all(
            step.processing_config.variable_components == [VariableComponents.TIMEPOINT]
            for step in pipeline_steps
        )


def test_classification_measurement_selector_remains_public_behavior() -> None:
    cppipe_path = Path("pipelines/classification.cppipe")
    filemanager = _MemoryFileManager(
        {cppipe_path: """CellProfiler Pipeline: https://cellprofiler.org
NamesAndTypes:[module_num:1|enabled:True]
    Assignments count:1
    Select the image type:Grayscale image
    Name to assign these images:DNA
    Select the rule criteria:and (file does contain \"DNA\")
IdentifyPrimaryObjects:[module_num:2|enabled:True]
    Select the input image:DNA
    Name the primary objects to be identified:Nuclei
MeasureObjectSizeShape:[module_num:3|enabled:True]
    Select object sets to measure:Nuclei
ClassifyObjects:[module_num:4|enabled:True]
    Make each classification decision on how many measurements?:Single measurement
    Select the object to be classified:Nuclei
    Select the object to be classified:Nuclei
    Select the measurement to classify by:AreaShape_Area
    Select bin spacing:Custom-defined bins
    Enter the custom thresholds separating the values between bins:0.5
    Use a bin for objects below the threshold?:Yes
    Use a bin for objects above the threshold?:Yes
    Select the object name:Nuclei
"""}
    )

    pipeline_steps, _pipeline_config = import_cellprofiler_pipeline(
        cppipe_path,
        filemanager=filemanager,
        backend=Backend.MEMORY,
    )

    classify_step = next(
        step for step in pipeline_steps if step.name == "ClassifyObjects"
    )
    invocation = tuple(normalize_function_pattern(classify_step.func).iter_items())[0]
    assert dict(invocation.kwargs)["measurement_feature"] == "AreaShape_Area"
    source = FunctionStepTransportAuthority.source_from_pipeline(pipeline_steps)
    assert "'measurement_feature': 'AreaShape_Area'" in source


def test_threshold_measurement_subject_retains_exact_output_identity() -> None:
    cppipe_path = Path("pipelines/threshold-observed-output.cppipe")
    filemanager = _MemoryFileManager(
        {cppipe_path: """CellProfiler Pipeline: https://cellprofiler.org
NamesAndTypes:[module_num:1|enabled:True]
    Assignments count:1
    Select the image type:Grayscale image
    Name to assign these images:phase
    Select the rule criteria:and (metadata does channel \"1\")
Threshold:[module_num:2|enabled:True]
    Select the input image:phase
    Name the output image:phaseThresh
    Threshold strategy:Global
    Thresholding method:Minimum Cross-Entropy
    Threshold smoothing scale:1.3488
"""}
    )

    pipeline_steps, _pipeline_config = import_cellprofiler_pipeline(
        cppipe_path,
        filemanager=filemanager,
        backend=Backend.MEMORY,
    )

    threshold_step = next(step for step in pipeline_steps if step.name == "Threshold")
    invocation = next(normalize_function_pattern(threshold_step.func).iter_items())
    assert invocation.kwargs_dict["name_the_output_image"] == "phaseThresh"
    assert "select_the_input_image" not in invocation.kwargs_dict


def test_shared_input_ref_retains_only_the_mismatching_binding() -> None:
    invocation_key = FunctionInvocationKey(
        function_name="measure_object_neighbors",
        group_key=DEFAULT_GROUP_KEY,
        position=0,
    )
    available_outputs = (ArtifactSpec.output("Nuclei", ObjectLabelsArtifactType),)
    available_producers = artifact_producers_for_outputs(
        available_outputs,
        groups=(None,),
        invocation_keys=(invocation_key,),
    )
    context = ArtifactDeclarationStepContext(
        step_name="MeasureObjectNeighbors",
        step_index=0,
        available_artifacts=ArtifactSpecCollection(available_outputs),
        available_artifact_producers=available_producers,
    )
    module = ModuleBlock(
        name="MeasureObjectNeighbors",
        module_num=4,
        setting_records=[
            ModuleSetting("Select objects to measure", "Nuclei"),
            ModuleSetting("Select neighboring objects to measure", "Nuclei"),
            ModuleSetting("Method to determine neighbors", "Expand until adjacent"),
            ModuleSetting("Neighbor distance", "5"),
        ],
    )
    target_contract = MeasureObjectNeighborsModule.callable_contract(
        module=module,
        invocation_key=invocation_key,
        step_context=context,
    )
    neighbor_runtime_parameter = (
        MeasureObjectNeighborsModule.neighbor_objects_binding.runtime_parameter_name
    )
    assert neighbor_runtime_parameter is not None
    target_unit = _ParsedTargetUnit(
        module=module,
        invocation_key=invocation_key,
        contract=target_contract,
        raw_callable=measure_object_neighbors,
        behavior_kwargs={
            "distance_method": DistanceMethod.EXPAND,
            "neighbor_distance": 5,
        },
        compile_kwargs={},
        identity_kwargs={
            MeasureObjectNeighborsModule.measured_objects_binding.require_parameter_name(): "Nuclei",
            MeasureObjectNeighborsModule.neighbor_objects_binding.require_parameter_name(): "Nuclei",
        },
        processing_config=ProcessingConfig(),
        context=context,
        step_source_bindings=StepSourceBindingsConfig(),
        output_producers=artifact_producers_for_outputs(
            target_contract.artifact_outputs,
            groups=(None,),
            invocation_keys=(invocation_key,),
        ),
        selected_input_bindings=(
            _SelectedInputBindingOccurrence(
                binding=MeasureObjectNeighborsModule.neighbor_objects_binding,
                refs=tuple(
                    spec.ref()
                    for spec in target_contract.artifact_inputs
                    if spec.parameter_name == neighbor_runtime_parameter
                ),
            ),
        ),
        target_position=0,
    )

    projection = _public_kwargs_for_target(
        MeasureObjectNeighborsModule,
        (target_unit,),
        candidate_group_keys=(),
        step_context=context,
    )

    assert projection is not None
    measured_parameter = (
        MeasureObjectNeighborsModule.measured_objects_binding.require_parameter_name()
    )
    neighbor_parameter = (
        MeasureObjectNeighborsModule.neighbor_objects_binding.require_parameter_name()
    )
    assert measured_parameter not in projection.kwargs
    assert projection.kwargs[neighbor_parameter] == "Nuclei"
    assert tuple(
        selection.binding for selection in projection.units[0].selected_input_bindings
    ) == (MeasureObjectNeighborsModule.neighbor_objects_binding,)


def test_full_domain_modules_preserve_group_owned_output_identity() -> None:
    cppipe_path = Path("pipelines/grouped.cppipe")
    filemanager = _MemoryFileManager(
        {cppipe_path: """CellProfiler Pipeline: https://cellprofiler.org
NamesAndTypes:[module_num:1|enabled:True]
    Assignments count:2
    Select the image type:Grayscale image
    Name to assign these images:DNA
    Select the rule criteria:and (metadata does channel \"1\")
    Select the image type:Grayscale image
    Name to assign these images:RNA
    Select the rule criteria:and (metadata does channel \"2\")
IdentifyPrimaryObjects:[module_num:2|enabled:True]
    Select the input image:RNA
    Name the primary objects to be identified:Cells
IdentifyPrimaryObjects:[module_num:3|enabled:True]
    Select the input image:DNA
    Name the primary objects to be identified:Nuclei
"""}
    )

    pipeline_steps, _pipeline_config = import_cellprofiler_pipeline(
        cppipe_path,
        filemanager=filemanager,
        backend=Backend.MEMORY,
    )

    assert len(pipeline_steps) == 1
    pattern = normalize_function_pattern(pipeline_steps[0].func)
    assert pattern.is_grouped
    invocations = tuple(pattern.iter_items())
    assert tuple(invocation.key.group_key for invocation in invocations) == ("2", "1")
    assert tuple(invocation.kwargs_dict for invocation in invocations) == (
        {"name_the_primary_objects_to_be_identified": "Cells"},
        {"name_the_primary_objects_to_be_identified": "Nuclei"},
    )


def test_observed_channel_outputs_lower_to_one_exact_dict_step() -> None:
    cppipe_path = Path("pipelines/grouped-observed.cppipe")
    filemanager = _MemoryFileManager(
        {cppipe_path: """CellProfiler Pipeline: https://cellprofiler.org
NamesAndTypes:[module_num:1|enabled:True]
    Assignments count:2
    Select the image type:Grayscale image
    Name to assign these images:DNA
    Select the rule criteria:and (metadata does channel \"1\")
    Select the image type:Grayscale image
    Name to assign these images:RNA
    Select the rule criteria:and (metadata does channel \"2\")
IdentifyPrimaryObjects:[module_num:2|enabled:True]
    Select the input image:DNA
    Name the primary objects to be identified:Nuclei
IdentifyPrimaryObjects:[module_num:3|enabled:True]
    Select the input image:RNA
    Name the primary objects to be identified:Cells
MeasureObjectSizeShape:[module_num:4|enabled:True]
    Select object sets to measure:Nuclei
MeasureObjectSizeShape:[module_num:5|enabled:True]
    Select object sets to measure:Cells
"""}
    )

    pipeline_steps, _pipeline_config = import_cellprofiler_pipeline(
        cppipe_path,
        filemanager=filemanager,
        backend=Backend.MEMORY,
    )

    identify_step = next(
        step for step in pipeline_steps if step.name == "IdentifyPrimaryObjects"
    )
    pattern = normalize_function_pattern(identify_step.func)
    assert pattern.is_grouped
    assert all(len(group.items) == 1 for group in pattern.groups)
    invocations = tuple(pattern.iter_items())
    assert tuple(invocation.key.group_key for invocation in invocations) == ("1", "2")
    assert tuple(
        invocation.kwargs_dict["name_the_primary_objects_to_be_identified"]
        for invocation in invocations
    ) == ("Nuclei", "Cells")
    assert all(
        "select_the_input_image" not in invocation.kwargs_dict
        for invocation in invocations
    )


def test_independent_nonpreserving_modules_lower_to_separate_steps() -> None:
    cppipe_path = Path("pipelines/independent-gray-to-color.cppipe")
    filemanager = _MemoryFileManager(
        {cppipe_path: """CellProfiler Pipeline: https://cellprofiler.org
NamesAndTypes:[module_num:1|enabled:True]
    Assignments count:4
    Select the image type:Grayscale image
    Name to assign these images:Straightened_mCherry
    Select the rule criteria:and (metadata does channel "1")
    Select the image type:Grayscale image
    Name to assign these images:Straightened_GFP
    Select the rule criteria:and (metadata does channel "2")
    Select the image type:Grayscale image
    Name to assign these images:mCherry
    Select the rule criteria:and (metadata does channel "3")
    Select the image type:Grayscale image
    Name to assign these images:GFP
    Select the rule criteria:and (metadata does channel "4")
GrayToColor:[module_num:2|variable_revision_number:4|enabled:True]
    Select a color scheme:RGB
    Rescale intensity:No
    Select the image to be colored red:Straightened_mCherry
    Select the image to be colored green:Straightened_GFP
    Select the image to be colored blue:Leave this black
    Name the output image:StraightenedRG
GrayToColor:[module_num:3|variable_revision_number:4|enabled:True]
    Select a color scheme:RGB
    Rescale intensity:No
    Select the image to be colored red:mCherry
    Select the image to be colored green:GFP
    Select the image to be colored blue:Leave this black
    Name the output image:OrigRG
SaveImages:[module_num:4|enabled:True]
    Select the image to save:StraightenedRG
    Saved file format:tiff
SaveImages:[module_num:5|enabled:True]
    Select the image to save:OrigRG
    Saved file format:tiff
"""}
    )

    pipeline_steps, _pipeline_config = import_cellprofiler_pipeline(
        cppipe_path,
        filemanager=filemanager,
        backend=Backend.MEMORY,
    )

    gray_steps = tuple(step for step in pipeline_steps if step.name == "GrayToColor")
    assert len(gray_steps) == 2
    assert all(
        len(tuple(normalize_function_pattern(step.func).iter_items())) == 1
        for step in gray_steps
    )


def test_active_channel_outputs_preserve_declared_identity_without_consumers() -> None:
    cppipe_path = Path("pipelines/full-channel-plain.cppipe")
    filemanager = _MemoryFileManager(
        {cppipe_path: """CellProfiler Pipeline: https://cellprofiler.org
NamesAndTypes:[module_num:1|enabled:True]
    Assignments count:2
    Select the image type:Grayscale image
    Name to assign these images:OrigStain1
    Select the rule criteria:and (metadata does channel "1")
    Select the image type:Grayscale image
    Name to assign these images:OrigStain2
    Select the rule criteria:and (metadata does channel "2")
CorrectIlluminationCalculate:[module_num:2|enabled:True]
    Select the input image:OrigStain1
    Name the output image:IllumStain1
    Select how the illumination function is calculated:Regular
    Dilate objects in the final averaged image?:No
    Dilation radius:1
    Block size:60
    Rescale the illumination function?:Yes
    Calculate function for each image individually, or based on all images?:Each
    Smoothing method:Fit Polynomial
    Method to calculate smoothing filter size:Automatic
    Approximate object diameter:10
    Smoothing filter size:10
    Retain the averaged image?:No
    Name the averaged image:IllumBlueAvg
    Retain the dilated image?:No
    Name the dilated image:IllumBlueDilated
    Automatically calculate spline parameters?:Yes
    Background mode:auto
    Number of spline points:5
    Background threshold:2.0
    Image resampling factor:2.0
    Maximum number of iterations:40
    Residual value for convergence:0.001
CorrectIlluminationCalculate:[module_num:3|enabled:True]
    Select the input image:OrigStain2
    Name the output image:IllumStain2
    Select how the illumination function is calculated:Regular
    Dilate objects in the final averaged image?:No
    Dilation radius:1
    Block size:60
    Rescale the illumination function?:Yes
    Calculate function for each image individually, or based on all images?:Each
    Smoothing method:Fit Polynomial
    Method to calculate smoothing filter size:Automatic
    Approximate object diameter:10
    Smoothing filter size:10
    Retain the averaged image?:No
    Name the averaged image:IllumBlueAvg
    Retain the dilated image?:No
    Name the dilated image:IllumBlueDilated
    Automatically calculate spline parameters?:Yes
    Background mode:auto
    Number of spline points:5
    Background threshold:2.0
    Image resampling factor:2.0
    Maximum number of iterations:40
    Residual value for convergence:0.001
CorrectIlluminationApply:[module_num:4|enabled:True]
    Select the input image:OrigStain1
    Name the output image:CorrectedStain1
    Select the illumination function:IllumStain1
    Select how the illumination function is applied:Divide
    Set output image values less than 0 equal to 0?:Yes
    Set output image values greater than 1 equal to 1?:Yes
CorrectIlluminationApply:[module_num:5|enabled:True]
    Select the input image:OrigStain2
    Name the output image:CorrectedStain2
    Select the illumination function:IllumStain2
    Select how the illumination function is applied:Divide
    Set output image values less than 0 equal to 0?:Yes
    Set output image values greater than 1 equal to 1?:Yes
"""}
    )

    pipeline_steps, _pipeline_config = import_cellprofiler_pipeline(
        cppipe_path,
        filemanager=filemanager,
        backend=Backend.MEMORY,
    )

    assert len(pipeline_steps) == 2
    calculate_pattern = normalize_function_pattern(pipeline_steps[0].func)
    apply_pattern = normalize_function_pattern(pipeline_steps[1].func)
    assert calculate_pattern.is_grouped
    assert apply_pattern.is_grouped
    calculate_invocations = tuple(calculate_pattern.iter_items())
    apply_invocations = tuple(apply_pattern.iter_items())
    assert tuple(
        invocation.kwargs_dict["name_the_output_image"]
        for invocation in calculate_invocations
    ) == ("IllumStain1", "IllumStain2")
    assert tuple(
        invocation.kwargs_dict["name_the_output_image"]
        for invocation in apply_invocations
    ) == ("CorrectedStain1", "CorrectedStain2")


def test_repeated_natural_measurement_images_use_group_local_source_identity() -> None:
    cppipe_path = Path("pipelines/grouped-measurements.cppipe")
    filemanager = _MemoryFileManager(
        {cppipe_path: """CellProfiler Pipeline: https://cellprofiler.org
NamesAndTypes:[module_num:1|enabled:True]
    Assignments count:3
    Select the image type:Grayscale image
    Name to assign these images:DNA
    Select the rule criteria:and (metadata does channel "1")
    Select the image type:Grayscale image
    Name to assign these images:PH3
    Select the rule criteria:and (metadata does channel "2")
    Select the image type:Grayscale image
    Name to assign these images:Mito
    Select the rule criteria:and (metadata does channel "3")
IdentifyPrimaryObjects:[module_num:2|enabled:True]
    Select the input image:DNA
    Name the primary objects to be identified:Nuclei
MeasureObjectIntensity:[module_num:3|enabled:True]
    Select images to measure:DNA, PH3
    Select object sets to measure:Nuclei
"""}
    )

    pipeline_steps, pipeline_config = import_cellprofiler_pipeline(
        cppipe_path,
        filemanager=filemanager,
        backend=Backend.MEMORY,
    )

    measurement_step = next(
        step for step in pipeline_steps if step.name == "MeasureObjectIntensity"
    )
    invocations = tuple(normalize_function_pattern(measurement_step.func).iter_items())
    assert tuple(invocation.key.group_key for invocation in invocations) == ("1", "2")
    assert all(
        "select_images_to_measure" not in invocation.kwargs_dict
        for invocation in invocations
    )
    assert tuple(invocation.kwargs_dict for invocation in invocations) == (
        {},
        {"select_object_sets_to_measure": "Nuclei"},
    )
    with config_context(pipeline_config):
        assert measurement_step.processing_config.group_by is GroupBy.CHANNEL


def test_repeated_object_measurements_retain_each_scalar_object_selection() -> None:
    cppipe_path = Path("pipelines/repeated-object-measurements.cppipe")
    filemanager = _MemoryFileManager(
        {cppipe_path: """CellProfiler Pipeline: https://cellprofiler.org
NamesAndTypes:[module_num:1|enabled:True]
    Assignments count:1
    Select the image type:Grayscale image
    Name to assign these images:DNA
    Select the rule criteria:and (metadata does channel "1")
MedianFilter:[module_num:2|enabled:True]
    Select the input image:DNA
    Name the output image:CorrDNA
    Window:3
IdentifyPrimaryObjects:[module_num:3|enabled:True]
    Select the input image:CorrDNA
    Name the primary objects to be identified:Nuclei
IdentifySecondaryObjects:[module_num:4|enabled:True]
    Select the input image:CorrDNA
    Select the input objects:Nuclei
    Name the objects to be identified:Cells
IdentifyTertiaryObjects:[module_num:5|enabled:True]
    Select the larger identified objects:Cells
    Select the smaller identified objects:Nuclei
    Name the tertiary objects to be identified:Cytoplasm
MeasureObjectIntensity:[module_num:6|enabled:True]
    Select images to measure:CorrDNA
    Select objects to measure:Nuclei, Cells, Cytoplasm
"""}
    )

    pipeline_steps, _pipeline_config = import_cellprofiler_pipeline(
        cppipe_path,
        filemanager=filemanager,
        backend=Backend.MEMORY,
    )

    measurement_steps = tuple(
        step for step in pipeline_steps if step.name == "MeasureObjectIntensity"
    )
    assert len(measurement_steps) == 1
    measurement_invocations = tuple(
        normalize_function_pattern(measurement_steps[0].func).iter_items()
    )
    assert tuple(
        invocation.kwargs_dict["select_object_sets_to_measure"]
        for invocation in measurement_invocations
    ) == ("Nuclei", "Cells", "Cytoplasm")

    source = FunctionStepTransportAuthority.source_from_pipeline(pipeline_steps)
    namespace: dict[str, object] = {}
    exec(compile(source, "<repeated-object-measurements>", "exec"), namespace)
    reconstructed = FunctionStepTransportAuthority.pipeline_steps_from_namespace(
        namespace
    )
    reconstructed_measurements = tuple(
        step for step in reconstructed if step.name == "MeasureObjectIntensity"
    )
    assert FunctionStepTransportAuthority.source_from_pipeline(reconstructed) == source
    assert tuple(
        invocation.kwargs_dict["select_object_sets_to_measure"]
        for step in reconstructed_measurements
        for invocation in normalize_function_pattern(step.func).iter_items()
    ) == ("Nuclei", "Cells", "Cytoplasm")


def test_mixed_source_and_produced_measurements_share_one_step() -> None:
    cppipe_path = Path("pipelines/mixed-source-produced-measurements.cppipe")
    filemanager = _MemoryFileManager(
        {cppipe_path: """CellProfiler Pipeline: https://cellprofiler.org
NamesAndTypes:[module_num:1|enabled:True]
    Assignments count:2
    Select the image type:Grayscale image
    Name to assign these images:DNA
    Select the rule criteria:and (metadata does channel "1")
    Select the image type:Grayscale image
    Name to assign these images:RNA
    Select the rule criteria:and (metadata does channel "2")
MedianFilter:[module_num:2|enabled:True]
    Select the input image:DNA
    Name the output image:CorrDNA
    Window:3
IdentifyPrimaryObjects:[module_num:3|enabled:True]
    Select the input image:CorrDNA
    Name the primary objects to be identified:Nuclei
MeasureObjectIntensityDistribution:[module_num:4|enabled:True]
    Select images to measure:CorrDNA, RNA
    Hidden:1
    Hidden:1
    Hidden:0
    Calculate intensity Zernikes?:None
    Maximum zernike moment:9
    Select objects to measure:Nuclei
    Object to use as center?:These objects
    Select objects to use as centers:None
    Scale the bins?:Yes
    Number of bins:4
    Maximum radius:100
"""}
    )

    pipeline_steps, pipeline_config = import_cellprofiler_pipeline(
        cppipe_path,
        filemanager=filemanager,
        backend=Backend.MEMORY,
    )

    measurement_steps = tuple(
        step
        for step in pipeline_steps
        if step.name == "MeasureObjectIntensityDistribution"
    )
    assert len(measurement_steps) == 1
    measurement_invocations = tuple(
        normalize_function_pattern(measurement_steps[0].func).iter_items()
    )
    assert len(measurement_invocations) == 1
    assert measurement_invocations[0].kwargs_dict["select_images_to_measure"] == (
        "CorrDNA",
        "RNA",
    )
    source = FunctionStepTransportAuthority.source_from_pipeline(pipeline_steps)
    namespace: dict[str, object] = {}
    exec(compile(source, "<mixed-source-produced-measurements>", "exec"), namespace)
    reconstructed = FunctionStepTransportAuthority.pipeline_steps_from_namespace(
        namespace
    )
    assert FunctionStepTransportAuthority.source_from_pipeline(reconstructed) == source
    with config_context(pipeline_config):
        assert (
            measurement_steps[0].processing_config.input_source
            is InputSource.PIPELINE_START
        )


def test_multi_image_source_contract_keeps_every_source_input_exact() -> None:
    bindings = tuple(
        NamedSourceBinding(
            alias=alias,
            component_identity=(ComponentSelector(AllComponents.CHANNEL, channel),),
        )
        for alias, channel in (
            ("origDNA", "2"),
            ("origMito", "1"),
            ("origMemb", "0"),
        )
    )
    source_bindings = StepSourceBindingsConfig(enabled=True, bindings=bindings)
    step_context = ArtifactDeclarationStepContext(
        step_name="ImageMath",
        step_index=13,
        source_bindings=source_bindings,
        group_by=GroupBy.CHANNEL,
        input_source=InputSource.PIPELINE_START,
    ).with_source_declarations(binding.input_spec() for binding in bindings)
    module = ModuleBlock(
        name="ImageMath",
        module_num=14,
        setting_records=[
            ModuleSetting("Operation", "Add"),
            ModuleSetting("Select the first image", "origDNA"),
            ModuleSetting("Select the second image", "origMemb"),
            ModuleSetting("Select the third image", "origMito"),
            ModuleSetting("Name the output image", "Monolayer"),
        ],
    )

    contract = CellProfilerModule.require_module("ImageMath").callable_contract(
        module=module,
        invocation_key=FunctionInvocationKey(
            "image_math",
            DEFAULT_GROUP_KEY,
            0,
        ),
        step_context=step_context,
    )

    assert tuple(spec.name for spec in contract.artifact_inputs) == (
        "origDNA",
        "origMemb",
        "origMito",
    )


def test_previous_step_lowering_enables_only_contract_selected_source_bindings() -> (
    None
):
    selected = NamedSourceBinding(alias="DNA")
    unrelated = NamedSourceBinding(alias="RNA")
    source_bindings = StepSourceBindingsConfig(
        bindings=(selected, unrelated),
    )

    def mixed_inputs(image: object) -> object:
        return image

    contract = _contract_for(
        mixed_inputs,
        artifact_inputs=(
            selected.input_spec(),
            ArtifactSpec.input("ProducedImage", ImageArtifactType),
        ),
    )

    projected = _public_step_source_bindings(
        source_bindings,
        contract.artifact_inputs,
        InputSource.PREVIOUS_STEP,
    )

    assert projected.enabled is True
    assert projected.binding_declarations == (selected,)


def test_order_matched_3d_sources_retain_declared_order_match_plan() -> None:
    cppipe_path = Path("pipelines/ordered-volumes.cppipe")
    filemanager = _MemoryFileManager(
        {cppipe_path: """CellProfiler Pipeline: https://cellprofiler.org
NamesAndTypes:[module_num:1|enabled:True]
    Assignments count:3
    Select the image type:Grayscale image
    Name to assign these images:origDNA
    Select the rule criteria:and (metadata does ChannelNumber "2")
    Select the image type:Grayscale image
    Name to assign these images:origMito
    Select the rule criteria:and (metadata does ChannelNumber "1")
    Select the image type:Grayscale image
    Name to assign these images:origMemb
    Select the rule criteria:and (metadata does ChannelNumber "0")
    Image set matching method:Order
    Process as 3D?:Yes
ImageMath:[module_num:2|enabled:True]
    Operation:Add
    Select the first image:origDNA
    Select the second image:origMemb
    Select the third image:origMito
    Name the output image:Monolayer
"""}
    )

    _steps, pipeline_config = import_cellprofiler_pipeline(
        cppipe_path,
        filemanager=filemanager,
        backend=Backend.MEMORY,
    )

    match_plan = pipeline_config.source_bindings_config.match_plan
    assert match_plan is not None
    assert match_plan.method is SourceBindingMatchMethod.ORDER
    assert match_plan.dimensions == ()


def test_save_images_target_preserves_cross_group_source_and_main_flow() -> None:
    cppipe_path = Path("pipelines/example-tumor-source-lineage.cppipe")
    filemanager = _MemoryFileManager(
        {cppipe_path: """CellProfiler Pipeline: https://cellprofiler.org
NamesAndTypes:[module_num:1|enabled:True]
    Assignments count:2
    Select the image type:Color image
    Name to assign these images:ColorFluor
    Select the rule criteria:and (file does contain "f.jpg")
    Select the image type:Color image
    Name to assign these images:ColorLung
    Select the rule criteria:and (file does contain "b.jpg")
ColorToGray:[module_num:2|enabled:True]
    Select the input image:ColorFluor
    Conversion method:Split
    Image type:RGB
    Name the output image:OrigGray
    Relative weight of the red channel:1.0
    Relative weight of the green channel:1.0
    Relative weight of the blue channel:1.0
    Convert red to gray?:No
    Name the output image:OrigRed
    Convert green to gray?:Yes
    Name the output image:GrayTumor
    Convert blue to gray?:No
    Name the output image:OrigBlue
IdentifyPrimaryObjects:[module_num:3|enabled:True]
    Select the input image:GrayTumor
    Name the primary objects to be identified:tumor
    Typical diameter of objects, in pixel units (Min,Max):4,99999
SaveImages:[module_num:4|enabled:True]
    Select the type of image to save:Image
    Select the image to save:GrayTumor
    Select method for constructing file names:From image filename
    Select image name for file prefix:ColorLung
    Enter single file name:OrigBlue
    Append a suffix to the image file name?:Yes
    Text to append to the image name:_Tumors
    Saved file format:png
    Image bit depth:8-bit integer
    Overwrite existing files without warning?:Yes
    Record the file and path information to the saved image?:No
    Base image folder:Elsewhere...|
"""}
    )

    pipeline_steps, pipeline_config = import_cellprofiler_pipeline(
        cppipe_path,
        filemanager=filemanager,
        backend=Backend.MEMORY,
    )

    gray_pattern = normalize_function_pattern(pipeline_steps[0].func)
    assert tuple(group.group_key for group in gray_pattern.groups) == ("1",)
    gray_invocation = next(gray_pattern.iter_items())
    assert dict(gray_invocation.kwargs)["name_the_output_image"] == "GrayTumor"
    object_pattern = normalize_function_pattern(pipeline_steps[1].func)
    assert tuple(group.group_key for group in object_pattern.groups) == (
        DEFAULT_GROUP_KEY,
    )
    assert pipeline_steps[2].name == "SaveImages"
    with config_context(pipeline_config):
        color_lung_binding = pipeline_steps[2].source_bindings.binding_for_alias(
            "ColorLung"
        )
    assert color_lung_binding is not None
    assert color_lung_binding.projection_role is SourceProjectionRole.PRIMARY_PLANE
    save_invocation = next(
        normalize_function_pattern(pipeline_steps[2].func).iter_items()
    )
    assert "select_the_image_to_save" not in save_invocation.kwargs_dict
    assert save_invocation.kwargs_dict["select_image_name_for_file_prefix"] == (
        "ColorLung"
    )


def test_cross_group_runtime_input_is_not_compressed_to_plain_dispatch() -> None:
    cppipe_path = Path("pipelines/crop.cppipe")
    filemanager = _MemoryFileManager(
        {cppipe_path: """CellProfiler Pipeline: https://cellprofiler.org
NamesAndTypes:[module_num:1|enabled:True]
    Assignments count:2
    Select the image type:Grayscale image
    Name to assign these images:OrigBlue
    Select the rule criteria:and (file does contain "D.TIF")
    Select the image type:Grayscale image
    Name to assign these images:OrigGreen
    Select the rule criteria:and (file does contain "F.TIF")
Crop:[module_num:2|enabled:True]
    Select the input image:OrigBlue
    Name the output image:CropBlue
    Select the cropping shape:Rectangle
    Select the cropping method:Coordinates
    Left and right rectangle positions:1,10
    Top and bottom rectangle positions:1,10
    Coordinates of ellipse center:5,5
    Ellipse radius, X direction:5
    Ellipse radius, Y direction:5
    Remove empty rows and columns?:No
    Select the masking image:None
    Select the image with a cropping mask:None
    Select the objects:None
Crop:[module_num:3|enabled:True]
    Select the input image:OrigGreen
    Name the output image:CropGreen
    Select the cropping shape:Previous cropping
    Select the cropping method:Coordinates
    Left and right rectangle positions:1,10
    Top and bottom rectangle positions:1,10
    Coordinates of ellipse center:5,5
    Ellipse radius, X direction:5
    Ellipse radius, Y direction:5
    Remove empty rows and columns?:No
    Select the masking image:None
    Select the image with a cropping mask:CropBlue
    Select the objects:None
"""}
    )

    pipeline_steps, _pipeline_config = import_cellprofiler_pipeline(
        cppipe_path,
        filemanager=filemanager,
        backend=Backend.MEMORY,
    )

    assert len(pipeline_steps) == 2
    second_invocation = next(
        normalize_function_pattern(pipeline_steps[1].func).iter_items()
    )
    assert second_invocation.key.group_key == "2"
    assert "select_the_image_with_a_cropping_mask" not in second_invocation.kwargs_dict


def test_grouped_public_step_reconstructs_cross_group_input_edge() -> None:
    cppipe_path = Path("pipelines/grouped-flow.cppipe")
    filemanager = _MemoryFileManager(
        {cppipe_path: """CellProfiler Pipeline: https://cellprofiler.org
NamesAndTypes:[module_num:1|enabled:True]
    Assignments count:3
    Select the image type:Grayscale image
    Name to assign these images:OrigBlue
    Select the rule criteria:and (metadata does channel "1")
    Select the image type:Grayscale image
    Name to assign these images:OrigGreen
    Select the rule criteria:and (metadata does channel "2")
    Select the image type:Grayscale image
    Name to assign these images:OrigRed
    Select the rule criteria:and (metadata does channel "3")
Crop:[module_num:2|enabled:True]
    Select the input image:OrigBlue
    Name the output image:CropBlue
    Select the cropping shape:Rectangle
    Select the cropping method:Coordinates
    Left and right rectangle positions:1,10
    Top and bottom rectangle positions:1,10
    Coordinates of ellipse center:5,5
    Ellipse radius, X direction:5
    Ellipse radius, Y direction:5
    Remove empty rows and columns?:No
    Select the masking image:None
    Select the image with a cropping mask:None
    Select the objects:None
Crop:[module_num:3|enabled:True]
    Select the input image:OrigGreen
    Name the output image:CropGreen
    Select the cropping shape:Previous cropping
    Select the cropping method:Coordinates
    Left and right rectangle positions:1,10
    Top and bottom rectangle positions:1,10
    Coordinates of ellipse center:5,5
    Ellipse radius, X direction:5
    Ellipse radius, Y direction:5
    Remove empty rows and columns?:No
    Select the masking image:None
    Select the image with a cropping mask:CropBlue
    Select the objects:None
Crop:[module_num:4|enabled:True]
    Select the input image:OrigRed
    Name the output image:CropRed
    Select the cropping shape:Previous cropping
    Select the cropping method:Coordinates
    Left and right rectangle positions:1,10
    Top and bottom rectangle positions:1,10
    Coordinates of ellipse center:5,5
    Ellipse radius, X direction:5
    Ellipse radius, Y direction:5
    Remove empty rows and columns?:No
    Select the masking image:None
    Select the image with a cropping mask:CropBlue
    Select the objects:None
IdentifyPrimaryObjects:[module_num:5|enabled:True]
    Select the input image:CropBlue
    Name the primary objects to be identified:Nuclei
"""}
    )

    pipeline_steps, _pipeline_config = import_cellprofiler_pipeline(
        cppipe_path,
        filemanager=filemanager,
        backend=Backend.MEMORY,
    )

    grouped_invocations = tuple(
        normalize_function_pattern(pipeline_steps[1].func).iter_items()
    )
    assert tuple(invocation.key.group_key for invocation in grouped_invocations) == (
        "2",
        "3",
    )
    assert all(
        "select_the_image_with_a_cropping_mask" not in invocation.kwargs_dict
        for invocation in grouped_invocations
    )


def test_setting_name_aliases_do_not_duplicate_repeated_role_occurrences() -> None:
    setting_family = SettingNameFamily(
        "Select the object to be classified",
        aliases=("Select the object name",),
    )
    module = ModuleBlock(
        name="ClassifyObjects",
        module_num=19,
        setting_records=[
            ModuleSetting("Select the object to be classified", "Nuclei"),
            ModuleSetting("Select the object to be classified", "Cells"),
            ModuleSetting("Select the object name", "Nuclei"),
        ],
    )

    assert setting_values(module, setting_family) == ("Nuclei", "Cells")


def test_direct_import_uses_generic_config_composition_without_field_mirrors() -> None:
    tree = ast.parse(PIPELINE_IMPORT_PATH.read_text(encoding="utf-8"))

    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "vars"
        for node in ast.walk(tree)
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id
        in {
            "LazyProcessingConfig",
            "LazySourceBindingsConfig",
            "LazyStepSourceBindingsConfig",
        }
        and node.keywords
        for node in ast.walk(tree)
    )
    projected_types = {
        node.func.value.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "from_config"
        and isinstance(node.func.value, ast.Name)
    }
    assert projected_types == {
        "LazyProcessingConfig",
        "LazyStepSourceBindingsConfig",
        "PipelineConfig",
    }
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "Microscope" not in names
    attribute_names = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert "group_scope_inputs" in attribute_names
    assert "group_scope_sources" not in attribute_names
    assert "invocation_callable_contract" in attribute_names
    assert "module_blocks_for_invocation" in attribute_names
    assert "number_step_invocation_blocks" in attribute_names
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "replace"
        and any(keyword.arg == "group_key" for keyword in node.keywords)
        for node in ast.walk(tree)
    )


def test_special_input_resolves_from_the_exact_prior_artifact_producer() -> None:
    module = ModuleBlock(
        name="OverlayObjects",
        module_num=1,
        enabled=True,
        setting_records=[
            ModuleSetting("Select the input image", "Derived"),
            ModuleSetting("Select objects to display", "Objects"),
            ModuleSetting("Name the output image", "Overlay"),
        ],
    )
    image = ArtifactSpec.input("Derived", ImageArtifactType)
    labels = ArtifactSpec.output("Objects", ObjectLabelsArtifactType)
    label_producers = artifact_producers_for_outputs(
        (labels,),
        groups=(None,),
        invocation_keys=(
            FunctionInvocationKey(
                function_name="identify_primary_objects",
                group_key=DEFAULT_GROUP_KEY,
                position=0,
            ),
        ),
    )

    contract = OverlayObjectsModule.callable_contract(
        module=module,
        invocation_key=FunctionInvocationKey(
            function_name="overlay_objects",
            group_key=DEFAULT_GROUP_KEY,
            position=0,
        ),
        step_context=ArtifactDeclarationStepContext(
            step_index=0,
            available_artifacts=ArtifactSpecCollection((image, labels)),
            main_flow_artifacts=ArtifactSpecCollection((image,)),
            available_artifact_producers=label_producers,
        ),
    )

    runtime_labels = labels.for_plan_type(ArtifactInputPlan)
    assert contract.artifact_inputs.specs == (image, runtime_labels)
