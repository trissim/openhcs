"""Synthetic-plate integration coverage for 2D neurite outgrowth."""

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
from polystore.roi import PolylineShape, load_rois_from_zip
from skimage.draw import disk, line

from objectstate.lazy_factory import ensure_global_config_context
from openhcs.constants import GroupBy, Microscope, VariableComponents
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
from openhcs.processing.backends.analysis.neurite_outgrowth import (
    MetaXpressCellBodySettings,
    MetaXpressNuclearSettings,
    MetaXpressOutgrowthSettings,
    neurite_outgrowth_metaxpress,
)
from openhcs.tests.generators.generate_synthetic_data import (
    SyntheticMicroscopyGenerator,
)


def _write_known_neurite_images(plate_dir):
    neurites = np.zeros((96, 128), dtype=np.uint16)
    rows, columns = disk((48, 20), 9, shape=neurites.shape)
    neurites[rows, columns] = 1000
    rows, columns = line(48, 28, 48, 110)
    neurites[rows, columns] = 700
    rows, columns = line(48, 70, 25, 95)
    neurites[rows, columns] = 700

    nuclei = np.zeros_like(neurites)
    rows, columns = disk((48, 20), 5, shape=nuclei.shape)
    nuclei[rows, columns] = 1200

    image_dir = plate_dir / "TimePoint_1"
    for site in (1, 2):
        tifffile.imwrite(image_dir / f"A01_s{site:03d}_w1_z001_t001.tif", neurites)
        tifffile.imwrite(image_dir / f"A01_s{site:03d}_w2_z001_t001.tif", nuclei)


def test_neurite_outgrowth_runs_on_synthetic_plate_as_2d_channel_stack(
    tmp_path, caplog
):
    caplog.set_level(logging.CRITICAL)
    plate_dir = tmp_path / "synthetic_neurite_plate"
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        SyntheticMicroscopyGenerator(
            output_dir=str(plate_dir),
            grid_size=(1, 1),
            tile_size=(96, 128),
            overlap_percent=10,
            stage_error_px=1,
            wavelengths=2,
            z_stack_levels=1,
            num_cells=1,
            wells=["A01"],
            format="ImageXpress",
            random_seed=11,
        ).generate_dataset()
    _write_known_neurite_images(plate_dir)

    suffix = "_neurite_test"
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
    step = FunctionStep(
        name="MetaXpress-style neurite outgrowth",
        func=(
            neurite_outgrowth_metaxpress,
            {
                "cell_body": MetaXpressCellBodySettings(
                    approximate_max_width=30.0,
                    minimum_area=100.0,
                    intensity_above_local_background=100.0,
                ),
                "outgrowth": MetaXpressOutgrowthSettings(
                    maximum_width=3.0,
                    intensity_above_local_background=100.0,
                    minimum_cell_growth_to_log_as_significant=20.0,
                ),
                "use_nuclear_stain": True,
                "nuclear_stain": MetaXpressNuclearSettings(
                    channel_index=1,
                    approx_min_width=6.0,
                    approx_max_width=14.0,
                    intensity_above_local_background=200.0,
                ),
            },
        ),
        processing_config=LazyProcessingConfig(
            variable_components=[VariableComponents.CHANNEL],
            group_by=GroupBy.SITE,
        ),
    )

    ObjectStateRegistry.clear()
    progress_queue = queue.Queue()
    try:
        ensure_global_config_context(GlobalPipelineConfig, global_config)
        orchestrator = PipelineOrchestrator(
            plate_dir, pipeline_config=pipeline_config
        ).initialize()
        set_progress_queue(progress_queue)
        compilation = orchestrator.compile_pipelines(
            pipeline_definition=[step], well_filter=["A01"]
        )
        compiled_context = compilation["execution_bundle"].runtime_contexts["A01"]
        compiled_plan = compiled_context.step_plans[0]
        assert compiled_plan.variable_components == [VariableComponents.CHANNEL]
        assert compiled_plan.group_by is GroupBy.SITE
        compiled_pattern = compiled_plan.compiled_function_pattern
        assert compiled_pattern is not None
        compiled_invocation = next(compiled_pattern.iter_invocations())
        assert compiled_invocation.kwargs_dict["pixel_size"] == 0.65

        results = orchestrator.execute_compiled_plate(
            execution_bundle=compilation["execution_bundle"],
            progress_queue=progress_queue,
            progress_context={
                "execution_id": f"test::{time.time_ns()}",
                "plate_id": str(plate_dir),
                "axis_id": "",
            },
        )
        assert results["A01"].is_success(), results["A01"].error_message

        summary_paths = list(tmp_path.rglob("*neurite_outgrowth_summary*.csv"))
        cell_paths = list(tmp_path.rglob("*neurite_outgrowth_cells*.csv"))
        assert len(summary_paths) == 2
        assert len(cell_paths) == 2
        summary_rows = []
        for path in summary_paths:
            with path.open(newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))
            assert len(rows) == 1
            summary_rows.extend(rows)
        cell_rows = []
        for path in cell_paths:
            with path.open(newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))
            assert len(rows) == 1
            cell_rows.extend(rows)
        assert len(summary_rows) == 2
        assert len(cell_rows) == 2
        assert all(int(row["number_of_cells"]) == 1 for row in summary_rows)
        assert all(int(row["total_processes"]) == 1 for row in summary_rows)
        assert all(int(row["total_branches"]) == 1 for row in summary_rows)
        assert all(int(row["cell_body_channel_index"]) == 0 for row in summary_rows)
        assert all(int(row["nuclear_channel_index"]) == 1 for row in summary_rows)
        assert all(row["significant_growth"] == "True" for row in cell_rows)

        roi_paths = sorted(
            (
                *tmp_path.rglob("*cell_bodies*rois.roi.zip"),
                *tmp_path.rglob("*neurite_outgrowth*rois.roi.zip"),
                *tmp_path.rglob("*neurons*rois.roi.zip"),
                *tmp_path.rglob("*nuclei*rois.roi.zip"),
            )
        )
        assert len(roi_paths) == 8
        assert len([path for path in roi_paths if "cell_bodies" in path.name]) == 2
        assert (
            len(
                [path for path in roi_paths if "_neurite_outgrowth_step0_" in path.name]
            )
            == 2
        )
        assert len([path for path in roi_paths if "_neurons_step0_" in path.name]) == 2
        assert len([path for path in roi_paths if "nuclei" in path.name]) == 2
        assert all(
            "_w1_" in path.name for path in roi_paths if "nuclei" not in path.name
        )
        assert all("_w2_" in path.name for path in roi_paths if "nuclei" in path.name)
        assert all(load_rois_from_zip(path) for path in roi_paths)

        swc_paths = sorted(tmp_path.rglob("*neurite_morphology*.swc"))
        graph_roi_paths = sorted(
            tmp_path.rglob("*neurite_morphology*.graph.roi.zip")
        )
        assert len(swc_paths) == 2
        assert len(graph_roi_paths) == 2
        assert all(
            "# OpenHCS spatial graph: neurite_morphology" in path.read_text()
            for path in swc_paths
        )
        for graph_roi_path in graph_roi_paths:
            branch_rois = load_rois_from_zip(graph_roi_path)
            assert branch_rois
            for branch_roi in branch_rois:
                assert len(branch_roi.shapes) == 1
                assert isinstance(branch_roi.shapes[0], PolylineShape)
                assert branch_roi.metadata["label"] == 1
                assert branch_roi.metadata["neuron_label"] == 1
                assert branch_roi.metadata["branch_distance_um"] > 0
                assert branch_roi.metadata["euclidean_distance_um"] > 0
                assert branch_roi.metadata["tortuosity"] >= 1.0
                assert branch_roi.metadata["distance_from_soma_um"] >= 0
                assert "branch_type" in branch_roi.metadata

        summaries = sorted(
            (
                *tmp_path.rglob("*cell_bodies*segmentation_summary.txt"),
                *tmp_path.rglob("*neurite_outgrowth*segmentation_summary.txt"),
                *tmp_path.rglob("*neurons*segmentation_summary.txt"),
                *tmp_path.rglob("*nuclei*segmentation_summary.txt"),
            )
        )
        assert len(summaries) == 8
        assert all("Spatial dimensions: 2D" in path.read_text() for path in summaries)
    finally:
        set_progress_queue(None)
        ObjectStateRegistry.clear()
