from __future__ import annotations

from pathlib import Path

from benchmark.converter.runtime_pipeline import (
    execute_pipeline_direct,
    prepare_generated_pipeline,
)
import numpy as np
from openhcs.config_framework.lazy_factory import ensure_global_config_context
from openhcs.constants import Microscope
from openhcs.core.artifacts import ArtifactKind
from openhcs.core.config import (
    GlobalPipelineConfig,
    LazyPathPlanningConfig,
    MaterializationBackend,
    PipelineConfig,
    VFSConfig,
)
from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
from openhcs.tests.generators.generate_synthetic_data import (
    SyntheticMicroscopyGenerator,
)
from PIL import Image


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


def _write_bbbc021_image(path: Path, *, seed: int, signal: int) -> None:
    rng = np.random.default_rng(seed)
    image = rng.normal(900, 40, size=(64, 64)).clip(0, 65535).astype(np.uint16)
    image[20:44, 20:44] = np.clip(
        image[20:44, 20:44].astype(np.int32) + signal,
        0,
        65535,
    ).astype(np.uint16)
    Image.fromarray(image).save(path)


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
