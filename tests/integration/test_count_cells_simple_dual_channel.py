from openhcs.core.pipeline_document import PipelineDocumentAuthority
import csv
import io
import logging
import os
from contextlib import redirect_stderr, redirect_stdout

import openhcs  # noqa: F401 - prefer repository submodules before direct imports
import numpy as np
import tifffile
from polystore.roi import load_rois_from_zip
from skimage.draw import disk
from zmqruntime.execution.responses import (
    ExecutionSubmissionResponse,
    ExecutionWaitResult,
)

from openhcs.config_framework.lazy_factory import ensure_global_config_context
from openhcs.constants import Microscope, VariableComponents
from openhcs.core.artifacts import MeasurementsArtifactType, ObjectLabelsArtifactType
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
from openhcs.core.function_step_transport import FunctionStepTransportAuthority
from openhcs.core.steps import FunctionStep
from openhcs.processing.backends.analysis.count_cells_simple import (
    MetaXpressW2Settings,
    MetaXpressWavelengthSettings,
    StainedArea,
    count_cells_simple_dual_channel,
)
from openhcs.runtime.zmq_execution_client import (
    OpenHCSExecutionSubmission,
    ZMQExecutionClient,
)
from openhcs.runtime.zmq_execution_observation import (
    ZMQRuntimeExecutionObservationExport,
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
    tifffile.imwrite(image_dir / "A01_s002_w1_z001_t001.tif", channel_1)
    tifffile.imwrite(image_dir / "A01_s002_w2_z001_t001.tif", channel_2)


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
            variable_components=[VariableComponents.CHANNEL],
        ),
    )

    assert step.processing_config.variable_components == [VariableComponents.CHANNEL]

    observation_path = tmp_path / "dual_channel_runtime_observation.pkl"
    ensure_global_config_context(GlobalPipelineConfig, global_config)
    submission = OpenHCSExecutionSubmission(
        plate_id=plate_dir,
        pipeline_document=PipelineDocumentAuthority.from_values(
            pipeline_config=pipeline_config, pipeline_steps=[step]
        ),
        global_config=global_config,
        config_params={
            "runtime_observation_export_path": str(observation_path),
        },
    )
    assert submission.pipeline_code() == (
        FunctionStepTransportAuthority.source_from_pipeline([step])
    )
    client = ZMQExecutionClient(
        port=18000 + os.getpid() % 20000,
        persistent=False,
    )
    try:
        assert client.connect(timeout=30)
        compile_response = ExecutionSubmissionResponse.from_wire(
            client.submit_compile(submission)
        )
        compile_id = compile_response.require_execution_id(
            "native analysis integration compilation"
        )
        ExecutionWaitResult.from_wire(
            client.wait_for_completion(compile_id)
        ).require_complete("native analysis integration compilation")

        execution_response = ExecutionSubmissionResponse.from_wire(
            client.submit_pipeline(
                OpenHCSExecutionSubmission(
                    plate_id=plate_dir,
                    pipeline_document=PipelineDocumentAuthority.from_values(
                        pipeline_config=pipeline_config, pipeline_steps=[step]
                    ),
                    global_config=global_config,
                    config_params={
                        "runtime_observation_export_path": str(observation_path),
                    },
                    compile_artifact_id=compile_id,
                )
            )
        )
        execution_id = execution_response.require_execution_id(
            "native analysis integration execution"
        )
        ExecutionWaitResult.from_wire(
            client.wait_for_completion(execution_id)
        ).require_complete("native analysis integration execution")

        observation = ZMQRuntimeExecutionObservationExport.read(observation_path)
        observation.require_valid_observation()
        runtime_identities = {
            (record.key.name, record.key.artifact_type)
            for records in observation.records_by_axis.values()
            for record in records
        }
        assert {
            ("dual_channel_counts", MeasurementsArtifactType),
            ("dual_channel_cells", MeasurementsArtifactType),
            ("w1_nuclei", ObjectLabelsArtifactType),
            ("w2_stain", ObjectLabelsArtifactType),
        } <= runtime_identities

        csv_paths = list(tmp_path.rglob("*dual_channel_counts*.csv"))
        assert len(csv_paths) == 1
        with csv_paths[0].open(newline="") as csv_file:
            rows = list(csv.DictReader(csv_file))

        assert len(rows) == 2
        for row in rows:
            assert int(row["total_cell_count"]) == 2
            assert int(row["w2_positive_cell_count"]) == 1
            assert int(row["w2_negative_cell_count"]) == 1
            assert row["w2_stained_area"] == "nucleus"

        cell_paths = list(tmp_path.rglob("*dual_channel_cells*.csv"))
        assert len(cell_paths) == 1
        with cell_paths[0].open(newline="") as csv_file:
            cell_rows = list(csv.DictReader(csv_file))
        assert len(cell_rows) == 4
        assert {int(row["object_label"]) for row in cell_rows} == {1, 2}

        roi_paths = sorted(
            (
                *tmp_path.rglob("*w1_nuclei*rois.roi.zip"),
                *tmp_path.rglob("*w2_stain*rois.roi.zip"),
            )
        )
        assert len(roi_paths) == 4
        w1_roi_paths = [path for path in roi_paths if "w1_nuclei" in path.name]
        w2_roi_paths = [path for path in roi_paths if "w2_stain" in path.name]
        assert len(w1_roi_paths) == 2
        assert len(w2_roi_paths) == 2
        assert {"s001", "s002"} == {
            "s001" if "_s001_" in path.name else "s002" for path in w1_roi_paths
        }
        assert all("_w1_" in path.name for path in w1_roi_paths)
        assert all("_w2_" in path.name for path in w2_roi_paths)
        assert all(len(load_rois_from_zip(path)) == 2 for path in roi_paths)

        roi_summaries = sorted(
            (
                *tmp_path.rglob("*w1_nuclei*segmentation_summary.txt"),
                *tmp_path.rglob("*w2_stain*segmentation_summary.txt"),
            )
        )
        assert len(roi_summaries) == 4
        for summary_path in roi_summaries:
            summary = summary_path.read_text()
            assert "Spatial dimensions: 2D" in summary
            assert "Z-planes" not in summary
    finally:
        client.disconnect()
