from __future__ import annotations

import csv
from pathlib import Path

from benchmark.converter.runtime_pipeline import (
    execute_pipeline_direct,
    prepare_generated_pipeline,
)
import numpy as np
from openhcs.config_framework.lazy_factory import ensure_global_config_context
from openhcs.constants import Microscope
from openhcs.constants.constants import AllComponents
from openhcs.core.artifacts import ArtifactKind
from openhcs.core.config import (
    GlobalPipelineConfig,
    LazyPathPlanningConfig,
    MaterializationBackend,
    PipelineConfig,
    VFSConfig,
)
from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
from openhcs.core.source_bindings import ComponentSelector, SourceBindingOrigin
from openhcs.tests.generators.generate_synthetic_data import (
    SyntheticMicroscopyGenerator,
)
from PIL import Image
from scipy.io import savemat


def test_cppipe_generated_pipeline_executes_through_orchestrator(
    tmp_path: Path,
) -> None:
    plate_path = _generate_plate(tmp_path / "plate")
    cppipe_path = _write_cppipe(tmp_path / "identify_primary_objects.cppipe")
    prepared = prepare_generated_pipeline(
        cppipe_path,
        output_path=tmp_path / "generated_cellprofiler_pipeline.py",
    )

    global_config = GlobalPipelineConfig(num_workers=1, use_threading=True)
    ensure_global_config_context(GlobalPipelineConfig, global_config)
    pipeline_config = PipelineConfig(
        path_planning_config=LazyPathPlanningConfig(
            output_dir_suffix="_generated_cppipe",
        ),
        vfs_config=VFSConfig(
            materialization_backend=MaterializationBackend.DISK,
        ),
    )
    orchestrator = PipelineOrchestrator(plate_path, pipeline_config=pipeline_config)
    orchestrator.initialize()

    execution = execute_pipeline_direct(orchestrator, prepared.pipeline)

    assert prepared.infrastructure_modules
    assert prepared.registered_functions
    assert all(
        result.is_success()
        for result in execution.execution_results.values()
    )

    nuclei_records = execution.compiled_contexts["A01"].runtime_value_store.find(
        name="Nuclei",
        kind=ArtifactKind.OBJECT_LABELS,
        axis_id="A01",
    )
    assert len(nuclei_records) == 1
    assert nuclei_records[0].value.data.max() > 0


def test_bbbc021_cppipe_generated_pipeline_executes_named_channel_bindings(
    tmp_path: Path,
) -> None:
    plate_path = _generate_bbbc021_plate(tmp_path / "Week1_22123")
    cppipe_path = _write_bbbc021_cppipe(tmp_path / "bbbc021_multichannel.cppipe")
    prepared = prepare_generated_pipeline(
        cppipe_path,
        output_path=tmp_path / "generated_bbbc021_cellprofiler_pipeline.py",
    )

    global_config = GlobalPipelineConfig(
        num_workers=1,
        use_threading=True,
        microscope=Microscope.BBBC021,
    )
    ensure_global_config_context(GlobalPipelineConfig, global_config)
    pipeline_config = PipelineConfig(
        path_planning_config=LazyPathPlanningConfig(
            output_dir_suffix="_generated_cppipe",
        ),
        vfs_config=VFSConfig(
            materialization_backend=MaterializationBackend.DISK,
        ),
    )
    orchestrator = PipelineOrchestrator(plate_path, pipeline_config=pipeline_config)
    orchestrator.initialize()

    execution = execute_pipeline_direct(orchestrator, prepared.pipeline)

    assert all(
        result.is_success()
        for result in execution.execution_results.values()
    )
    nuclei_records = execution.compiled_contexts["A01"].runtime_value_store.find(
        name="Nuclei",
        kind=ArtifactKind.OBJECT_LABELS,
        axis_id="A01",
    )
    composite_records = execution.compiled_contexts["A01"].runtime_value_store.find(
        name="Composite",
        kind=ArtifactKind.IMAGE,
        axis_id="A01",
    )
    assert len(nuclei_records) == 1
    assert nuclei_records[0].value.data.max() > 0
    assert len(composite_records) == 1


def test_bbbc021_canonical_illum_cppipe_executes_real_pipeline_shape(
    tmp_path: Path,
) -> None:
    plate_path = _generate_bbbc021_plate(tmp_path / "Week1_22123")
    _write_bbbc021_image(
        plate_path / "fields" / "A01_s1_w4F00DBABE-17A7-4AA1-9C50-123456789ABC.tif",
        seed=3,
        signal=2400,
    )
    cppipe_path = (
        Path(__file__).resolve().parents[2]
        / "benchmark"
        / "cellprofiler_pipelines"
        / "BBBC021_illum.cppipe"
    )
    prepared = prepare_generated_pipeline(
        cppipe_path,
        output_path=tmp_path / "generated_bbbc021_illum_pipeline.py",
    )

    global_config = GlobalPipelineConfig(
        num_workers=1,
        use_threading=True,
        microscope=Microscope.BBBC021,
    )
    ensure_global_config_context(GlobalPipelineConfig, global_config)
    pipeline_config = PipelineConfig(
        path_planning_config=LazyPathPlanningConfig(
            output_dir_suffix="_generated_cppipe",
        ),
        vfs_config=VFSConfig(
            materialization_backend=MaterializationBackend.DISK,
        ),
    )
    orchestrator = PipelineOrchestrator(plate_path, pipeline_config=pipeline_config)
    orchestrator.initialize()

    execution = execute_pipeline_direct(orchestrator, prepared.pipeline)

    assert all(
        result.is_success()
        for result in execution.execution_results.values()
    )
    generated_images = sorted(
        (_generated_output_root(plate_path) / "images").glob("*.tif")
    )
    assert [path.name for path in generated_images] == [
        "A01_s1_w1_z001_t001.tif",
        "A01_s1_w2_z001_t001.tif",
        "A01_s1_w4_z001_t001.tif",
    ]


def test_loadimages_cppipe_executes_pipeline_start_mat_illumination_binding(
    tmp_path: Path,
) -> None:
    plate_path = _generate_loadimages_mat_illum_plate(tmp_path / "mat_illum_plate")
    cppipe_path = _write_loadimages_mat_illum_cppipe(
        tmp_path / "loadimages_mat_illum.cppipe"
    )
    prepared = prepare_generated_pipeline(
        cppipe_path,
        output_path=tmp_path / "generated_loadimages_mat_illum_pipeline.py",
    )

    raw_assignment = prepared.source_schema.resolved_assignment_for_alias("Raw")
    illum_assignment = prepared.source_schema.resolved_assignment_for_alias("Illum")
    assert raw_assignment is not None
    assert raw_assignment.origin is SourceBindingOrigin.PIPELINE_START
    assert illum_assignment is not None
    assert illum_assignment.origin is SourceBindingOrigin.PIPELINE_START

    global_config = GlobalPipelineConfig(
        num_workers=1,
        use_threading=True,
        microscope=Microscope.IMAGEXPRESS,
    )
    ensure_global_config_context(GlobalPipelineConfig, global_config)
    pipeline_config = PipelineConfig(
        path_planning_config=LazyPathPlanningConfig(
            output_dir_suffix="_generated_cppipe",
        ),
        vfs_config=VFSConfig(
            materialization_backend=MaterializationBackend.DISK,
        ),
    )
    orchestrator = PipelineOrchestrator(plate_path, pipeline_config=pipeline_config)
    orchestrator.initialize()

    execution = execute_pipeline_direct(orchestrator, prepared.pipeline)

    assert all(
        result.is_success()
        for result in execution.execution_results.values()
    )
    corrected_records = execution.compiled_contexts["A01"].runtime_value_store.find(
        name="CorrectedRaw",
        kind=ArtifactKind.IMAGE,
        axis_id="A01",
    )
    assert len(corrected_records) == 1
    assert np.asarray(corrected_records[0].value.data).shape[-2:] == (64, 64)
    assert sorted(
        path.name
        for path in (_generated_output_root(plate_path) / "images").glob("*.tif")
    ) == ["A01_s001_w1_z001_t001.tif"]


def test_examplefly_cppipe_generated_pipeline_executes_real_pipeline_shape(
    tmp_path: Path,
) -> None:
    plate_path = _generate_two_channel_plate(tmp_path / "examplefly_plate")
    cppipe_path = (
        Path(__file__).resolve().parents[2]
        / "benchmark"
        / "cellprofiler_pipelines"
        / "ExampleFly.cppipe"
    )
    prepared = prepare_generated_pipeline(
        cppipe_path,
        output_path=tmp_path / "generated_examplefly_cellprofiler_pipeline.py",
    )

    blue_assignment = prepared.source_schema.resolved_assignment_for_alias("OrigBlue")
    green_assignment = prepared.source_schema.resolved_assignment_for_alias("OrigGreen")
    assert blue_assignment is not None
    assert blue_assignment.selector.components == (
        ComponentSelector(AllComponents.CHANNEL, "1"),
    )
    assert green_assignment is not None
    assert green_assignment.selector.components == (
        ComponentSelector(AllComponents.CHANNEL, "2"),
    )
    assert any(
        module.name == "ExportToSpreadsheet"
        for module in prepared.infrastructure_modules
    )

    global_config = GlobalPipelineConfig(num_workers=1, use_threading=True)
    ensure_global_config_context(GlobalPipelineConfig, global_config)
    pipeline_config = PipelineConfig(
        path_planning_config=LazyPathPlanningConfig(
            output_dir_suffix="_generated_cppipe",
        ),
        vfs_config=VFSConfig(
            materialization_backend=MaterializationBackend.DISK,
        ),
    )
    orchestrator = PipelineOrchestrator(plate_path, pipeline_config=pipeline_config)
    orchestrator.initialize()

    execution = execute_pipeline_direct(orchestrator, prepared.pipeline)

    assert all(
        result.is_success()
        for result in execution.execution_results.values()
    )
    runtime_store = execution.compiled_contexts["A01"].runtime_value_store
    assert runtime_store.find(
        name="Cells",
        kind=ArtifactKind.OBJECT_LABELS,
        axis_id="A01",
    )
    assert runtime_store.find(
        name="Cytoplasm",
        kind=ArtifactKind.OBJECT_LABELS,
        axis_id="A01",
    )
    assert runtime_store.find(
        kind=ArtifactKind.MEASUREMENTS,
        axis_id="A01",
    )
    csv_outputs = sorted(_generated_results_dir(plate_path).rglob("*.csv"))
    assert len(csv_outputs) >= 6
    assert all(path.stat().st_size > 0 for path in csv_outputs)
    headers_by_name = {path.name: _csv_header(path) for path in csv_outputs}
    assert _matching_header(
        headers_by_name,
        "MeasureObjectSizeShape",
    )[:4] == ["slice_index", "object_label", "area", "perimeter"]
    assert "contrast" in _matching_header(headers_by_name, "MeasureTexture")
    assert "manders_m1" in _matching_header(headers_by_name, "MeasureColocalization")
    assert all("slice_index" in header for header in headers_by_name.values())


def test_cppipe_generated_pipeline_materializes_relationship_outputs(
    tmp_path: Path,
) -> None:
    plate_path = _generate_plate(tmp_path / "relationship_plate")
    cppipe_path = _write_relationship_cppipe(tmp_path / "relate_objects.cppipe")
    prepared = prepare_generated_pipeline(
        cppipe_path,
        output_path=tmp_path / "generated_relationship_pipeline.py",
    )

    global_config = GlobalPipelineConfig(num_workers=1, use_threading=True)
    ensure_global_config_context(GlobalPipelineConfig, global_config)
    pipeline_config = PipelineConfig(
        path_planning_config=LazyPathPlanningConfig(
            output_dir_suffix="_generated_cppipe",
        ),
        vfs_config=VFSConfig(
            materialization_backend=MaterializationBackend.DISK,
        ),
    )
    orchestrator = PipelineOrchestrator(plate_path, pipeline_config=pipeline_config)
    orchestrator.initialize()

    execution = execute_pipeline_direct(orchestrator, prepared.pipeline)

    assert all(
        result.is_success()
        for result in execution.execution_results.values()
    )

    runtime_store = execution.compiled_contexts["A01"].runtime_value_store
    relationship_records = runtime_store.find(
        kind=ArtifactKind.RELATIONSHIPS,
        axis_id="A01",
    )
    measurement_records = runtime_store.find(
        kind=ArtifactKind.MEASUREMENTS,
        axis_id="A01",
    )
    assert relationship_records
    assert measurement_records

    csv_outputs = sorted(_generated_results_dir(plate_path).rglob("*.csv"))
    assert csv_outputs
    assert any("relationships" in path.name for path in csv_outputs)
    assert any("measurements" in path.name for path in csv_outputs)
    headers_by_name = {path.name: _csv_header(path) for path in csv_outputs}
    assert _matching_header(
        headers_by_name,
        "relationships",
    ) == [
        "relationship_type",
        "source_role",
        "target_role",
        "source_object",
        "target_object",
        "parent_id",
        "child_id",
        "slice_index",
    ]
    assert _matching_header(
        headers_by_name,
        "measurements",
    ) == [
        "slice_index",
        "parent_object_count",
        "child_object_count",
        "children_with_parents_count",
        "mean_children_per_parent",
        "mean_centroid_distance",
        "mean_minimum_distance",
    ]


def _generate_plate(plate_path: Path) -> Path:
    generator = SyntheticMicroscopyGenerator(
        output_dir=str(plate_path),
        grid_size=(1, 1),
        tile_size=(128, 128),
        wavelengths=1,
        z_stack_levels=1,
        num_cells=12,
        cell_size_range=(8, 12),
        cell_intensity_range=(28000, 42000),
        background_intensity=200,
        noise_level=10,
        wells=["A01"],
        format="ImageXpress",
        random_seed=7,
    )
    generator.generate_dataset()
    return plate_path


def _generate_two_channel_plate(plate_path: Path) -> Path:
    generator = SyntheticMicroscopyGenerator(
        output_dir=str(plate_path),
        grid_size=(1, 1),
        tile_size=(128, 128),
        wavelengths=2,
        z_stack_levels=1,
        num_cells=12,
        cell_size_range=(8, 12),
        cell_intensity_range=(28000, 42000),
        background_intensity=200,
        noise_level=10,
        wells=["A01"],
        format="ImageXpress",
        random_seed=11,
    )
    generator.generate_dataset()
    return plate_path


def _generate_bbbc021_plate(plate_path: Path) -> Path:
    fields_dir = plate_path / "fields"
    fields_dir.mkdir(parents=True)
    _write_bbbc021_image(
        fields_dir / "A01_s1_w1BEDC2073-A983-4B98-95E9-84466707A25D.tif",
        seed=1,
        signal=3200,
    )
    _write_bbbc021_image(
        fields_dir / "A01_s1_w242F8F7B1-17A7-4AA1-9C50-123456789ABC.tif",
        seed=2,
        signal=1800,
    )
    return plate_path


def _generate_loadimages_mat_illum_plate(plate_path: Path) -> Path:
    generator = SyntheticMicroscopyGenerator(
        output_dir=str(plate_path),
        grid_size=(1, 1),
        tile_size=(64, 64),
        wavelengths=1,
        z_stack_levels=1,
        num_cells=4,
        cell_size_range=(6, 8),
        cell_intensity_range=(28000, 42000),
        background_intensity=200,
        noise_level=10,
        wells=["A01"],
        format="ImageXpress",
        random_seed=17,
    )
    generator.generate_dataset()
    savemat(
        plate_path / "illum_Channel2.mat",
        {"Image": np.full((64, 64), 2.0, dtype=np.float32)},
    )
    return plate_path


def _write_bbbc021_image(path: Path, *, seed: int, signal: int) -> None:
    rng = np.random.default_rng(seed)
    image = rng.normal(900, 40, size=(64, 64)).clip(0, 65535).astype(np.uint16)
    image[20:44, 20:44] = np.clip(
        image[20:44, 20:44].astype(np.int32) + signal,
        0,
        65535,
    ).astype(np.uint16)
    Image.fromarray(image).save(path)


def _generated_output_root(plate_path: Path) -> Path:
    return plate_path.parent / f"{plate_path.name}_generated_cppipe"


def _generated_results_dir(plate_path: Path) -> Path:
    return _generated_output_root(plate_path) / "images_results"


def _csv_header(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        return next(csv.reader(handle))


def _matching_header(
    headers_by_name: dict[str, list[str]],
    name_fragment: str,
) -> list[str]:
    for filename, header in headers_by_name.items():
        if name_fragment in filename:
            return header
    raise AssertionError(
        f"No CSV output filename contained {name_fragment!r}: "
        f"{sorted(headers_by_name)}"
    )


def _write_cppipe(cppipe_path: Path) -> Path:
    cppipe_path.write_text(
        "\n".join(
            (
                "CellProfiler Pipeline: http://www.cellprofiler.org",
                "Version:3",
                "DateRevision:300",
                "GitHash:",
                "ModuleCount:3",
                "HasImagePlaneDetails:False",
                (
                    "LoadData:[module_num:1|svn_version:'Unknown'|"
                    "enabled:True|wants_pause:False]"
                ),
                "    Input data file location:Elsewhere...",
                (
                    "IdentifyPrimaryObjects:[module_num:2|svn_version:'Unknown'|"
                    "enabled:True|wants_pause:False]"
                ),
                "    Select the input image:OrigBlue",
                "    Name the primary objects to be identified:Nuclei",
                (
                    "ExportToSpreadsheet:[module_num:3|svn_version:'Unknown'|"
                    "enabled:True|wants_pause:False]"
                ),
                "    Select measurements to export:No",
                "",
            )
        )
    )
    return cppipe_path


def _write_bbbc021_cppipe(cppipe_path: Path) -> Path:
    cppipe_path.write_text(
        "\n".join(
            (
                "CellProfiler Pipeline: http://www.cellprofiler.org",
                "Version:3",
                "DateRevision:300",
                "GitHash:",
                "ModuleCount:6",
                "HasImagePlaneDetails:False",
                (
                    "Images:[module_num:1|svn_version:'Unknown'|"
                    "enabled:True|wants_pause:False]"
                ),
                "    Filter images?:Images only",
                '    Select the rule criteria:or (file does containregexp "A01")',
                (
                    "Metadata:[module_num:2|svn_version:'Unknown'|"
                    "enabled:True|wants_pause:False]"
                ),
                "    Metadata extraction method:Extract from file/folder names",
                "    Metadata source:File name",
                (
                    "    Regular expression to extract from file name:"
                    "^.*(?P<well>[A-Z]\\d+)_s(?P<site>\\d+)_w(?P<channel>\\d).*$"
                ),
                (
                    "NamesAndTypes:[module_num:3|svn_version:'Unknown'|"
                    "enabled:True|wants_pause:False]"
                ),
                "    Assign a name to:Images matching rules",
                "    Select the image type:Grayscale image",
                "    Name to assign these images:DNA",
                "    Match metadata:[{'DNA': 'well'}, {'DNA': 'site'}]",
                "    Image set matching method:Metadata",
                '    Select the rule criteria:and (metadata does channel "1")',
                "    Assign a name to:Images matching rules",
                "    Select the image type:Grayscale image",
                "    Name to assign these images:Actin",
                "    Match metadata:[{'Actin': 'well'}, {'Actin': 'site'}]",
                "    Image set matching method:Metadata",
                '    Select the rule criteria:and (metadata does channel "2")',
                (
                    "IdentifyPrimaryObjects:[module_num:4|svn_version:'Unknown'|"
                    "enabled:True|wants_pause:False]"
                ),
                "    Select the input image:DNA",
                "    Name the primary objects to be identified:Nuclei",
                (
                    "GrayToColor:[module_num:5|svn_version:'Unknown'|"
                    "enabled:True|wants_pause:False]"
                ),
                "    Select the image to be colored green:Actin",
                "    Select the image to be colored blue:DNA",
                "    Name the output image:Composite",
                (
                    "ExportToSpreadsheet:[module_num:6|svn_version:'Unknown'|"
                    "enabled:True|wants_pause:False]"
                ),
                "    Select measurements to export:No",
                "",
            )
        )
    )
    return cppipe_path


def _write_loadimages_mat_illum_cppipe(cppipe_path: Path) -> Path:
    cppipe_path.write_text(
        "\n".join(
            (
                "CellProfiler Pipeline: http://www.cellprofiler.org",
                "Version:3",
                "DateRevision:300",
                "GitHash:",
                "ModuleCount:2",
                "HasImagePlaneDetails:False",
                (
                    "LoadImages:[module_num:1|svn_version:'Unknown'|"
                    "enabled:True|wants_pause:False]"
                ),
                "    What type of files are you loading?:individual images",
                "    How do you want to load these files?:Text-Exact match",
                "    Do you want to exclude certain files?:No",
                "    Type the text that these images have in common (case-sensitive):w1",
                "    What do you want to call this image in CellProfiler?:Raw",
                "    What is the position of this image in each group?:1",
                (
                    "    Do you want to extract metadata from the file name, "
                    "the subfolder path or both?:None"
                ),
                (
                    "    Type the text that these images have in common "
                    "(case-sensitive):illum_Channel2"
                ),
                "    What do you want to call this image in CellProfiler?:Illum",
                "    What is the position of this image in each group?:2",
                (
                    "    Do you want to extract metadata from the file name, "
                    "the subfolder path or both?:None"
                ),
                (
                    "CorrectIlluminationApply:[module_num:2|svn_version:'Unknown'|"
                    "enabled:True|wants_pause:False]"
                ),
                "    Select the input image:Raw",
                "    Name the output image:CorrectedRaw",
                "    Select the illumination function:Illum",
                "    Select how the illumination function is applied:Divide",
                "",
            )
        )
    )
    return cppipe_path


def _write_relationship_cppipe(cppipe_path: Path) -> Path:
    cppipe_path.write_text(
        "\n".join(
            (
                "CellProfiler Pipeline: http://www.cellprofiler.org",
                "Version:3",
                "DateRevision:300",
                "GitHash:",
                "ModuleCount:5",
                "HasImagePlaneDetails:False",
                (
                    "LoadData:[module_num:1|svn_version:'Unknown'|"
                    "enabled:True|wants_pause:False]"
                ),
                "    Input data file location:Elsewhere...",
                (
                    "IdentifyPrimaryObjects:[module_num:2|svn_version:'Unknown'|"
                    "enabled:True|wants_pause:False]"
                ),
                "    Select the input image:OrigBlue",
                "    Name the primary objects to be identified:Nuclei",
                (
                    "IdentifySecondaryObjects:[module_num:3|svn_version:'Unknown'|"
                    "enabled:True|wants_pause:False]"
                ),
                "    Select the input objects:Nuclei",
                "    Name the objects to be identified:Cells",
                "    Select the method to identify the secondary objects:Propagation",
                "    Select the input image:OrigBlue",
                "    Name the new primary objects:FilteredNuclei",
                (
                    "RelateObjects:[module_num:4|svn_version:'Unknown'|"
                    "enabled:True|wants_pause:False]"
                ),
                "    Select the parent objects:Nuclei",
                "    Select the child objects:Cells",
                (
                    "ExportToSpreadsheet:[module_num:5|svn_version:'Unknown'|"
                    "enabled:True|wants_pause:False]"
                ),
                "    Select measurements to export:No",
                "",
            )
        )
    )
    return cppipe_path
