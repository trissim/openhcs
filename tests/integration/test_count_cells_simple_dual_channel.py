import csv
import io
import logging
import queue
import time
from contextlib import redirect_stderr, redirect_stdout

import openhcs  # noqa: F401 - prefer repository submodules before direct imports
import numpy as np
import tifffile
from objectstate import ObjectStateRegistry
from skimage.draw import disk

from openhcs.config_framework.lazy_factory import ensure_global_config_context
from openhcs.constants import Microscope, VariableComponents
from openhcs.core.config import (
    AnalysisConsolidationConfig,
    GlobalPipelineConfig,
    LazyPathPlanningConfig,
    LazyProcessingConfig,
    MaterializationBackend,
    PathPlanningConfig,
    PipelineConfig,
    VFSConfig,
)
from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
from openhcs.core.progress import set_progress_queue
from openhcs.core.steps import FunctionStep
from openhcs.processing.backends.analysis.count_cells_simple import (
    MetaXpressW2Settings,
    MetaXpressWavelengthSettings,
    StainedArea,
    count_cells_simple_dual_channel,
)
from openhcs.tests.generators.generate_synthetic_data import (
    SyntheticMicroscopyGenerator,
)


def _write_known_dual_channel_images(plate_dir):
    """Replace generated pixels while preserving the synthetic plate layout."""

    channel_1 = np.full((64, 64), 500, dtype=np.uint16)
    channel_2 = np.full((64, 64), 1000, dtype=np.uint16)

    for center in ((20, 20), (45, 45)):
        rows, columns = disk(center, 5, shape=channel_1.shape)
        channel_1[rows, columns] = 20000

    for center in ((20, 20), (20, 45)):
        rows, columns = disk(center, 5, shape=channel_2.shape)
        channel_2[rows, columns] = 5000

    image_dir = plate_dir / "TimePoint_1"
    tifffile.imwrite(image_dir / "A01_s001_w1_z001_t001.tif", channel_1)
    tifffile.imwrite(image_dir / "A01_s001_w2_z001_t001.tif", channel_2)


def test_dual_channel_count_runs_on_synthetic_plate_with_channel_stack(
    tmp_path, caplog
):
    """Exercise the dual counter through parsing, compilation, VFS, and CSV output."""

    caplog.set_level(logging.CRITICAL)
    plate_dir = tmp_path / "synthetic_dual_channel_plate"
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        SyntheticMicroscopyGenerator(
            output_dir=str(plate_dir),
            grid_size=(1, 1),
            tile_size=(64, 64),
            overlap_percent=10,
            stage_error_px=1,
            wavelengths=2,
            z_stack_levels=1,
            num_cells=2,
            wells=["A01"],
            format="ImageXpress",
            random_seed=7,
        ).generate_dataset()
    _write_known_dual_channel_images(plate_dir)

    suffix = "_dual_channel_test"
    vfs_config = VFSConfig(materialization_backend=MaterializationBackend.DISK)
    global_config = GlobalPipelineConfig(
        num_workers=1,
        microscope=Microscope.IMAGEXPRESS,
        use_threading=True,
        path_planning_config=PathPlanningConfig(output_dir_suffix=suffix),
        vfs_config=vfs_config,
        analysis_consolidation_config=AnalysisConsolidationConfig(enabled=False),
    )
    pipeline_config = PipelineConfig(
        path_planning_config=LazyPathPlanningConfig(output_dir_suffix=suffix),
        vfs_config=vfs_config,
    )

    w1 = MetaXpressWavelengthSettings(
        channel_index=0,
        approx_min_width=6.0,
        approx_max_width=14.0,
        intensity_above_local_background=5000.0,
    )
    w2 = MetaXpressW2Settings(
        channel_index=1,
        approx_min_width=6.0,
        approx_max_width=14.0,
        intensity_above_local_background=2000.0,
        stained_area=StainedArea.NUCLEUS,
    )
    step = FunctionStep(
        name="Dual channel simple count",
        func=(
            count_cells_simple_dual_channel,
            {
                "w1": w1,
                "w2": w2,
                "minimum_stained_area": 20.0,
            },
        ),
        processing_config=LazyProcessingConfig(
            variable_components=[VariableComponents.CHANNEL]
        ),
    )

    assert step.processing_config.variable_components == [VariableComponents.CHANNEL]

    ObjectStateRegistry.clear()
    progress_queue = queue.Queue()
    try:
        ensure_global_config_context(GlobalPipelineConfig, global_config)
        orchestrator = PipelineOrchestrator(
            plate_dir, pipeline_config=pipeline_config
        ).initialize()
        assert orchestrator.get_component_keys(VariableComponents.CHANNEL) == [
            "1",
            "2",
        ]

        set_progress_queue(progress_queue)
        compilation = orchestrator.compile_pipelines(
            pipeline_definition=[step], well_filter=["A01"]
        )
        compiled_context = compilation["compiled_contexts"]["A01"]
        assert compiled_context.step_plans[0]["variable_components"] == [
            VariableComponents.CHANNEL
        ]
        _, compiled_kwargs = compiled_context.step_plans[0]["func"]
        assert compiled_kwargs["pixel_size"] == 1.0

        results = orchestrator.execute_compiled_plate(
            pipeline_definition=[step],
            compiled_contexts={"A01": compiled_context},
            progress_queue=progress_queue,
            progress_context={
                "execution_id": f"test::{time.time_ns()}",
                "plate_id": str(plate_dir),
                "axis_id": "",
            },
        )
        assert results["A01"].is_success(), results["A01"].error_message

        csv_paths = list(tmp_path.rglob("*dual_channel_counts*.csv"))
        assert len(csv_paths) == 1
        with csv_paths[0].open(newline="") as csv_file:
            rows = list(csv.DictReader(csv_file))

        assert len(rows) == 1
        assert int(rows[0]["total_cell_count"]) == 2
        assert int(rows[0]["w2_positive_cell_count"]) == 1
        assert int(rows[0]["w2_negative_cell_count"]) == 1
        assert rows[0]["w2_stained_area"] == "nucleus"
    finally:
        set_progress_queue(None)
        ObjectStateRegistry.clear()
