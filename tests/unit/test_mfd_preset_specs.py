import importlib.util
from pathlib import Path

from openhcs.constants.constants import VariableComponents
from openhcs.constants.input_source import InputSource
from openhcs.core.config import LazyDtypeConfig
from openhcs.core.memory import DtypeConversion
from openhcs.processing.backends.analysis.cell_counting_cpu import (
    DetectionMethod,
    count_cells_single_channel,
)
from openhcs.processing.backends.analysis.multi_template_matching import (
    multi_template_crop_reference_channel,
)
from openhcs.processing.backends.analysis.skan_axon_analysis import (
    AnalysisDimension,
    OutputMode,
    skan_axon_skeletonize_and_analyze,
)
from openhcs.processing.backends.assemblers.assemble_stack_cupy import (
    assemble_stack_cupy,
)
from openhcs.processing.backends.pos_gen.ashlar_main_cpu import (
    ashlar_compute_tile_positions_cpu,
)
from openhcs.processing.backends.pos_gen.ashlar_main_gpu import (
    ashlar_compute_tile_positions_gpu,
)
from openhcs.processing.backends.processors.cupy_processor import (
    create_composite,
    crop,
    sobel,
    stack_percentile_normalize,
    tophat,
)
from openhcs.processing.presets.mfd_specs import (
    MFD_WHOLE_DEVICE_TEMPLATE_PATH,
    MfdPresetKey,
    build_mfd_preset,
)

PIPELINE_DIR = (
    Path(__file__).resolve().parents[2]
    / "openhcs"
    / "processing"
    / "presets"
    / "pipelines"
)


def _load_pipeline_file(filename: str):
    module_path = PIPELINE_DIR / filename
    spec = importlib.util.spec_from_file_location(filename.replace("-", "_"), module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.pipeline_steps


def _normalize_call():
    return (
        stack_percentile_normalize,
        {
            "low_percentile": 0.1,
            "high_percentile": 99.9,
        },
    )


def test_crop_analyze_spec_matches_expected_step_contract():
    steps = build_mfd_preset(MfdPresetKey.CROP_ANALYZE)

    assert [step.name for step in steps] == [
        "crop_device",
        "crop_compartments",
        "analysis",
    ]
    assert steps[0].processing_config.variable_components == [VariableComponents.CHANNEL]
    assert steps[0].func == (
        multi_template_crop_reference_channel,
        {
            "score_threshold": 0.1,
            "method": 1,
            "template_path": MFD_WHOLE_DEVICE_TEMPLATE_PATH,
            "rotate_result": False,
        },
    )
    assert steps[1].func["3"] == (
        crop,
        {
            "width": 5046,
            "height": 3694,
            "start_x": 5253,
            "dtype_config": LazyDtypeConfig(
                default_dtype_conversion=DtypeConversion.UINT16
            ),
        },
    )
    assert steps[1].func["4"] == [
        (crop, {"width": 5046, "height": 3694}),
        tophat,
    ]
    assert steps[2].func["2"] == (
        count_cells_single_channel,
        {
            "min_cell_area": 40,
            "max_cell_area": 200,
            "enable_preprocessing": False,
            "return_segmentation_mask": True,
            "detection_method": DetectionMethod.WATERSHED,
            "dtype_config": LazyDtypeConfig(
                default_dtype_conversion=DtypeConversion.UINT8
            ),
        },
    )
    assert steps[2].func["3"] == (
        skan_axon_skeletonize_and_analyze,
        {
            "analysis_dimension": AnalysisDimension.TWO_D,
            "return_skeleton_visualizations": True,
            "skeleton_visualization_mode": OutputMode.SKELETON,
            "min_branch_length": 20.0,
            "dtype_config": LazyDtypeConfig(
                default_dtype_conversion=DtypeConversion.UINT8
            ),
        },
    )
    assert steps[2].func["4"] == []


def test_crop_analyze_cy5_overlay_adds_channel_4_cell_count():
    base_steps = build_mfd_preset(MfdPresetKey.CROP_ANALYZE)
    overlay_steps = build_mfd_preset(MfdPresetKey.CROP_ANALYZE_DAPI_FITC_CY5)

    assert base_steps[0].func == overlay_steps[0].func
    assert base_steps[1].func == overlay_steps[1].func
    assert overlay_steps[2].func["4"] == (
        count_cells_single_channel,
        {
            "min_cell_area": 100,
            "max_cell_area": 1000,
            "enable_preprocessing": False,
            "return_segmentation_mask": True,
            "detection_method": DetectionMethod.WATERSHED,
            "dtype_config": LazyDtypeConfig(
                default_dtype_conversion=DtypeConversion.UINT8
            ),
        },
    )


def test_stitch_specs_share_structure_and_vary_backend():
    cpu_steps = build_mfd_preset(MfdPresetKey.STITCH_ASHLAR_CPU)
    gpu_steps = build_mfd_preset(MfdPresetKey.STITCH_GPU)

    assert [step.name for step in cpu_steps] == [
        "process",
        "composite",
        "gpu_stitch",
        "process_2",
        "assemble",
    ]
    assert cpu_steps[0].func == gpu_steps[0].func
    assert cpu_steps[1].func == gpu_steps[1].func == create_composite
    assert cpu_steps[1].processing_config.variable_components == [VariableComponents.CHANNEL]
    assert cpu_steps[2].func == (ashlar_compute_tile_positions_cpu, {"stitch_alpha": 0.2})
    assert gpu_steps[2].func == (ashlar_compute_tile_positions_gpu, {"stitch_alpha": 0.2})
    assert cpu_steps[3].processing_config.input_source == InputSource.PIPELINE_START
    assert gpu_steps[3].processing_config.input_source == InputSource.PIPELINE_START
    assert cpu_steps[4].func is assemble_stack_cupy
    assert gpu_steps[4].func is assemble_stack_cupy
    assert cpu_steps[0].func["1"] == [_normalize_call(), (sobel, {"slice_by_slice": True}), _normalize_call()]
    assert cpu_steps[3].func["4"] == [_normalize_call(), tophat, _normalize_call()]


def test_preset_wrapper_files_materialize_pipeline_steps():
    expected_builders = {
        "10x_mfd_crop_analyze.py": MfdPresetKey.CROP_ANALYZE,
        "10x_mfd_crop_analyze_dapi-fitc-cy5.py": MfdPresetKey.CROP_ANALYZE_DAPI_FITC_CY5,
        "10x_mfd_stitch_ashlar_cpu.py": MfdPresetKey.STITCH_ASHLAR_CPU,
        "10x_mfd_stitch_gpu.py": MfdPresetKey.STITCH_GPU,
    }

    for filename, preset_key in expected_builders.items():
        wrapper_steps = _load_pipeline_file(filename)
        direct_steps = build_mfd_preset(preset_key)
        assert [step.name for step in wrapper_steps] == [step.name for step in direct_steps]
        assert [step.func for step in wrapper_steps] == [step.func for step in direct_steps]
