# Edit this pipeline and save to apply changes

# Automatically collected imports
from pathlib import Path

from openhcs.constants.constants import VariableComponents
from openhcs.constants.input_source import InputSource
from openhcs.core.config import LazyDtypeConfig, LazyStepMaterializationConfig
from arraybridge.decorators import DtypeConversion
from openhcs.core.steps.function_step import FunctionStep
from openhcs.processing.backends.analysis.cell_counting_cpu import (
    DetectionMethod,
    count_cells_single_channel,
)
from openhcs.processing.backends.analysis.multi_template_matching import (
    OpenCVTemplateMatchMethod,
    multi_template_crop_reference_channel,
)
from openhcs.processing.backends.analysis.skan_axon_analysis import (
    AnalysisDimension,
    OutputMode,
    skan_axon_skeletonize_and_analyze,
)
from openhcs.processing.backends.processors.cupy_processor import crop, tophat

# Pipeline steps
pipeline_steps = []

# Step 1: crop_device
step_1 = FunctionStep(
    func=(
        multi_template_crop_reference_channel,
        {
            "score_threshold": 0.1,
            "method": OpenCVTemplateMatchMethod.SQDIFF_NORMED,
            "template_path": Path("templates/mfd_96_sobel_10x_whole_device.tif"),
            "rotate_result": False,
        },
    ),
    name="crop_device",
    variable_components=[VariableComponents.CHANNEL],
)
pipeline_steps.append(step_1)

# Step 2: crop_compartments_cell_body_cy5
step_2 = FunctionStep(
    func=[
        (crop, {"start_y": 160, "width": 2812, "height": 3250, "start_x": 2000}),
        tophat,
    ],
    name="crop_compartments_cell_body_cy5",
    step_materialization_config=LazyStepMaterializationConfig(sub_dir="cellbody"),
)
pipeline_steps.append(step_2)

# Step 3: analysis_cellbody_cy5_dapi
step_3 = FunctionStep(
    func={
        "4": (
            count_cells_single_channel,
            {
                "min_cell_area": 40,
                "max_cell_area": 300,
                "enable_preprocessing": False,
                "detection_method": DetectionMethod.WATERSHED,
                "dtype_config": LazyDtypeConfig(
                    default_dtype_conversion=DtypeConversion.UINT8
                ),
            },
        )
    },
    name="analysis_cellbody_cy5_dapi",
)
pipeline_steps.append(step_3)

# Step 4: dapi_cellbody_count
step_4 = FunctionStep(
    func={
        "2": (
            count_cells_single_channel,
            {
                "min_cell_area": 40,
                "max_cell_area": 200,
                "enable_preprocessing": False,
                "detection_method": DetectionMethod.WATERSHED,
                "dtype_config": LazyDtypeConfig(
                    default_dtype_conversion=DtypeConversion.UINT8
                ),
            },
        )
    },
    name="dapi_cellbody_count",
)
pipeline_steps.append(step_4)

# Step 5: crop_device
step_5 = FunctionStep(
    func=(
        multi_template_crop_reference_channel,
        {
            "score_threshold": 0.1,
            "method": OpenCVTemplateMatchMethod.SQDIFF_NORMED,
            "template_path": Path("templates/mfd_96_sobel_10x_whole_device.tif"),
            "rotate_result": False,
        },
    ),
    name="crop_device",
    variable_components=[VariableComponents.CHANNEL],
    input_source=InputSource.PIPELINE_START,
)
pipeline_steps.append(step_5)

# Step 6: crop_axon_cy5
step_6 = FunctionStep(
    func=[
        (crop, {"width": 1000, "height": 3294, "start_x": 5253, "start_y": 200}),
        tophat,
    ],
    name="crop_axon_cy5",
)
pipeline_steps.append(step_6)

# Step 7: axon_cy_5_analysis
step_7 = FunctionStep(
    func={
        "4": (
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
    },
    name="axon_cy_5_analysis",
)
pipeline_steps.append(step_7)
