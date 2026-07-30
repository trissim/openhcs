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

from objectstate.lazy_factory import ensure_global_config_context
from openhcs.constants import Microscope, VariableComponents
from openhcs.core.artifacts import (
    MeasurementsArtifactType,
    ObjectLabelsArtifactType,
    SpecialArtifactType,
)
from openhcs.core.config import (
    AnalysisConsolidationConfig,
    GlobalPipelineConfig,
    LazyPathPlanningConfig,
    LazyProcessingConfig,
    LazyVFSConfig,
    MaterializationBackend,
    PathPlanningConfig,
    PipelineConfig,
    VFSConfig,
)
from openhcs.core.pipeline.path_planner import PathPlannerPathAuthority
from openhcs.core.steps import FunctionStep
from openhcs.processing.backends.analysis.count_cells_simple import (
    MetaXpressW2Settings,
    MetaXpressWavelengthSettings,
    SimpleCellSegmentationConfig,
    StainedArea,
    ThresholdMethod,
    count_cells_simple,
    count_cells_simple_dual_channel,
)
from openhcs.processing.custom_functions import CustomFunctionManager
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
    for well in ("A01", "B01"):
        tifffile.imwrite(
            image_dir / f"{well}_s001_w1_z001_t001.tif",
            channel_1,
        )
        tifffile.imwrite(
            image_dir / f"{well}_s001_w2_z001_t001.tif",
            channel_2,
        )
        tifffile.imwrite(
            image_dir / f"{well}_s002_w1_z001_t001.tif",
            channel_1,
        )
        tifffile.imwrite(
            image_dir / f"{well}_s002_w2_z001_t001.tif",
            channel_2,
        )


def test_dual_channel_count_runs_on_synthetic_plate_with_channel_stack(
    tmp_path, caplog, monkeypatch
):
    """Exercise built-in and persisted callables through spawned plate workers."""

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
            wells=["A01", "B01"],
            format="ImageXpress",
            random_seed=7,
        ).generate_dataset()
    _write_known_dual_channel_images(plate_dir)

    suffix = "_dual_channel_test"
    vfs_config = VFSConfig(materialization_backend=MaterializationBackend.DISK)
    global_config = GlobalPipelineConfig(
        num_workers=2,
        microscope=Microscope.IMAGEXPRESS,
        use_threading=False,
        path_planning_config=PathPlanningConfig(output_dir_suffix=suffix),
        vfs_config=vfs_config,
        analysis_consolidation_config=AnalysisConsolidationConfig(enabled=True),
    )
    pipeline_config = PipelineConfig(
        path_planning_config=LazyPathPlanningConfig(output_dir_suffix=suffix),
        vfs_config=LazyVFSConfig.from_config(vfs_config),
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
    aggregate_step = FunctionStep(
        name="Aggregate site and channel count",
        func=(
            count_cells_simple,
            {
                "segmentation_settings": SimpleCellSegmentationConfig(
                    threshold_method=ThresholdMethod.MANUAL,
                    threshold=1500.0,
                    min_size=20,
                    max_size=1000,
                ),
            },
        ),
        processing_config=LazyProcessingConfig(
            variable_components=[
                VariableComponents.SITE,
                VariableComponents.CHANNEL,
            ],
        ),
    )
    custom_function_data_home = tmp_path / "custom-function-data"
    monkeypatch.setenv("XDG_DATA_HOME", str(custom_function_data_home))
    custom_function_manager = CustomFunctionManager()
    [persisted_special_output_probe] = custom_function_manager.register_from_code(
        """
from openhcs.core.memory import numpy
from openhcs.core.pipeline.function_contracts import special_outputs
from openhcs.processing.materialization import CsvOptions, MaterializationSpec, ROIOptions

import numpy as np


@numpy
@special_outputs(
    (
        "legacy_counts",
        MaterializationSpec(CsvOptions(fields=["slice_index", "cell_count"])),
    ),
    ("legacy_masks", MaterializationSpec(ROIOptions())),
)
def persisted_special_output_probe(image):
    counts = []
    masks = []
    for slice_index, plane in enumerate(image):
        mask = plane > 1500
        counts.append(
            {
                "slice_index": slice_index,
                "cell_count": int(mask.any()),
            }
        )
        masks.append(mask.astype(np.int32))
    return image, counts, masks
""",
        persist=True,
        emit_signal=False,
    )
    assert custom_function_manager.source_path_for_function(
        persisted_special_output_probe
    ).exists()
    custom_step = FunctionStep(
        name="Persisted special-output compatibility",
        func=persisted_special_output_probe,
        processing_config=LazyProcessingConfig(
            variable_components=[
                VariableComponents.SITE,
                VariableComponents.CHANNEL,
            ],
        ),
    )
    pipeline_steps = [step, aggregate_step, custom_step]

    assert step.processing_config.variable_components == [VariableComponents.CHANNEL]
    assert aggregate_step.processing_config.variable_components == [
        VariableComponents.SITE,
        VariableComponents.CHANNEL,
    ]

    output_plate_root = PathPlannerPathAuthority.build_output_plate_root(
        plate_dir,
        global_config.path_planning_config,
    )
    analysis_results_dir = PathPlannerPathAuthority.analysis_results_dir_for(
        output_plate_root / global_config.path_planning_config.sub_dir
    )
    analysis_results_dir.mkdir(parents=True)
    stale_csv_path = (
        analysis_results_dir / "A01_stale_counts_step0_details.csv"
    )
    stale_csv_path.write_text(
        "slice_index,cell_count,stale_signal\n0,999,1\n",
        encoding="utf-8",
    )

    observation_path = tmp_path / "dual_channel_runtime_observation.pkl"
    ensure_global_config_context(GlobalPipelineConfig, global_config)
    submission = OpenHCSExecutionSubmission(
        plate_id=plate_dir,
        pipeline_document=PipelineDocumentAuthority.from_values(
            pipeline_config=pipeline_config, pipeline_steps=pipeline_steps
        ),
        global_config=global_config,
        config_params={
            "runtime_observation_export_path": str(observation_path),
        },
    )
    assert submission.pipeline_code() == PipelineDocumentAuthority.render(
        submission.pipeline_document
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
                        pipeline_config=pipeline_config, pipeline_steps=pipeline_steps
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
            ("cell_counts", MeasurementsArtifactType),
            ("segmentation_masks", ObjectLabelsArtifactType),
            ("legacy_counts", SpecialArtifactType),
            ("legacy_masks", SpecialArtifactType),
        } <= runtime_identities

        csv_paths = sorted(tmp_path.rglob("*dual_channel_counts*.csv"))
        assert len(csv_paths) == 4
        assert {"site-1", "site-2"} == {
            "site-1" if "_site-1_" in path.name else "site-2"
            for path in csv_paths
        }
        assert all("_w1_" not in path.name and "_w2_" not in path.name for path in csv_paths)
        rows = []
        for csv_path in csv_paths:
            with csv_path.open(newline="") as csv_file:
                site_rows = list(csv.DictReader(csv_file))
            assert len(site_rows) == 1
            expected_site = "1" if "_site-1_" in csv_path.name else "2"
            assert site_rows[0]["site"] == expected_site
            rows.extend(site_rows)

        assert len(rows) == 4
        for row in rows:
            assert int(row["total_cell_count"]) == 2
            assert int(row["w2_positive_cell_count"]) == 1
            assert int(row["w2_negative_cell_count"]) == 1
            assert row["w2_stained_area"] == "nucleus"

        cell_paths = sorted(tmp_path.rglob("*dual_channel_cells*.csv"))
        assert len(cell_paths) == 4
        cell_rows = []
        for cell_path in cell_paths:
            with cell_path.open(newline="") as csv_file:
                site_cell_rows = list(csv.DictReader(csv_file))
            assert len(site_cell_rows) == 2
            expected_site = "1" if "_site-1_" in cell_path.name else "2"
            assert {row["site"] for row in site_cell_rows} == {expected_site}
            assert {int(row["object_label"]) for row in site_cell_rows} == {1, 2}
            cell_rows.extend(site_cell_rows)
        assert len(cell_rows) == 8
        assert {int(row["object_label"]) for row in cell_rows} == {1, 2}

        aggregate_csv_paths = sorted(
            tmp_path.rglob("*cell_counts_step1_details.csv")
        )
        assert len(aggregate_csv_paths) == 2
        aggregate_rows = []
        for aggregate_csv_path in aggregate_csv_paths:
            with aggregate_csv_path.open(newline="") as csv_file:
                well_rows = list(csv.DictReader(csv_file))
            assert len(well_rows) == 4
            expected_well = aggregate_csv_path.name.split("_", maxsplit=1)[0]
            assert {row["well"] for row in well_rows} == {expected_well}
            assert {
                (row["site"], row["channel"])
                for row in well_rows
            } == {
                ("1", "1"),
                ("1", "2"),
                ("2", "1"),
                ("2", "2"),
            }
            assert {int(row["cell_count"]) for row in well_rows} == {2}
            aggregate_rows.extend(well_rows)
        assert len(aggregate_rows) == 8

        consolidated_summary = (
            analysis_results_dir
            / global_config.analysis_consolidation_config.output_filename
        )
        assert consolidated_summary.exists()
        summary_lines = consolidated_summary.read_text(encoding="utf-8").splitlines()
        summary_rows = list(csv.DictReader(summary_lines[6:]))
        assert len(summary_rows) == 2
        assert {row["Well"] for row in summary_rows} == {"A01", "B01"}
        assert not any(
            "stale" in field_name.lower()
            for field_name in summary_rows[0]
        )
        assert stale_csv_path.exists()

        roi_paths = sorted(
            (
                *tmp_path.rglob("*w1_nuclei*rois.roi.zip"),
                *tmp_path.rglob("*w2_stain*rois.roi.zip"),
            )
        )
        assert len(roi_paths) == 8
        w1_roi_paths = [path for path in roi_paths if "w1_nuclei" in path.name]
        w2_roi_paths = [path for path in roi_paths if "w2_stain" in path.name]
        assert len(w1_roi_paths) == 4
        assert len(w2_roi_paths) == 4
        assert {"s001", "s002"} == {
            "s001" if "_s001_" in path.name else "s002" for path in w1_roi_paths
        }
        assert all("_w1_" in path.name for path in w1_roi_paths)
        assert all("_w2_" in path.name for path in w2_roi_paths)
        assert all(len(load_rois_from_zip(path)) == 2 for path in roi_paths)

        aggregate_roi_paths = sorted(
            tmp_path.rglob("*segmentation_masks_step1_rois.roi.zip")
        )
        assert len(aggregate_roi_paths) == 8
        assert {
            (
                path.name.split("_", maxsplit=1)[0],
                "1" if "_s001_" in path.name else "2",
                "1" if "_w1_" in path.name else "2",
            )
            for path in aggregate_roi_paths
        } == {
            (well, site, channel)
            for well in ("A01", "B01")
            for site in ("1", "2")
            for channel in ("1", "2")
        }
        assert all(
            len(load_rois_from_zip(path)) == 2
            for path in aggregate_roi_paths
        )

        aggregate_roi_summaries = sorted(
            tmp_path.rglob("*segmentation_masks_step1_segmentation_summary.txt")
        )
        assert len(aggregate_roi_summaries) == 2

        custom_csv_paths = sorted(
            tmp_path.rglob("*legacy_counts_step2_details.csv")
        )
        assert len(custom_csv_paths) == 2
        assert all(
            "_z_index-1_timepoint-1_" in path.name
            and "_s001_" not in path.name
            and "_w1_" not in path.name
            for path in custom_csv_paths
        )
        for custom_csv_path in custom_csv_paths:
            with custom_csv_path.open(newline="") as csv_file:
                custom_rows = list(csv.DictReader(csv_file))
            assert len(custom_rows) == 4
            assert {int(row["slice_index"]) for row in custom_rows} == {
                0,
                1,
                2,
                3,
            }
            assert {int(row["cell_count"]) for row in custom_rows} == {1}

        custom_roi_paths = sorted(
            tmp_path.rglob("*legacy_masks_step2_rois.roi.zip")
        )
        assert len(custom_roi_paths) == 2
        assert all(
            "_z_index-1_timepoint-1_" in path.name
            and "_s001_" not in path.name
            and "_w1_" not in path.name
            for path in custom_roi_paths
        )
        assert all(load_rois_from_zip(path) for path in custom_roi_paths)

        roi_summaries = sorted(
            (
                *tmp_path.rglob("*w1_nuclei*segmentation_summary.txt"),
                *tmp_path.rglob("*w2_stain*segmentation_summary.txt"),
            )
        )
        assert len(roi_summaries) == 8
        for summary_path in roi_summaries:
            summary = summary_path.read_text()
            assert "Spatial dimensions: 2D" in summary
            assert "Z-planes" not in summary
    finally:
        client.disconnect()
