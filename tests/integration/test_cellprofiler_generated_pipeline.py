from __future__ import annotations

from pathlib import Path

from benchmark.converter.runtime_pipeline import (
    execute_pipeline_direct,
    prepare_generated_pipeline,
)
from openhcs.config_framework.lazy_factory import ensure_global_config_context
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
