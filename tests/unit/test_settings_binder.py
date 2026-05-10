from enum import Enum

import pytest

from benchmark.converter.settings_binder import (
    SettingToKeywordBinding,
    SettingsBinder,
    normalize_cellprofiler_setting_name,
    parse_cellprofiler_int,
)
from benchmark.converter.parser import ModuleBlock, ModuleSetting
from benchmark.converter.module_settings_binding import ModuleUnmappedSettingIgnore
from benchmark.converter.module_settings_binding import ModuleSettingsBindingStrategy
from benchmark.converter.module_settings_binding import UnmappedModuleSettingsError
from benchmark.converter.resize_objects_settings import RESIZE_OBJECTS_SETTINGS
from benchmark.cellprofiler_library.functions.imagemath import MathOperation
from openhcs.interop.cellprofiler.setting_names import setting_names


class ThresholdMethod(Enum):
    OTSU = "otsu"
    MANUAL = "manual"


def test_normalize_cellprofiler_setting_name_is_shared_authority():
    assert (
        normalize_cellprofiler_setting_name(
            "Typical diameter of objects, in pixel units (Min,Max)?"
        )
        == "typical_diameter_of_objects_in_pixel_units"
    )


def test_settings_binder_binds_typed_values_and_skips_infrastructure_keys():
    binder = SettingsBinder(
        enum_mappings={"threshold_method": ThresholdMethod}
    )

    assert binder.bind(
        {
            "Show window": "Yes",
            "Threshold method": "Otsu",
            "Typical diameter": "8, 80",
            "Object names": "Nuclei, Cells",
            "Smoothing radius": "1.5",
            "Iterations": "3",
        }
    ) == {
        "threshold_method": ThresholdMethod.OTSU,
        "typical_diameter": (8, 80),
        "object_names": ["Nuclei", "Cells"],
        "smoothing_radius": 1.5,
        "iterations": 3,
    }


def test_settings_binder_preserves_binding_provenance():
    details = SettingsBinder().bind_with_details(
        {"Use advanced settings?": "No"}
    )

    assert len(details) == 1
    assert details[0].name == "use_advanced_settings"
    assert details[0].value is False
    assert details[0].original_key == "Use advanced settings?"
    assert details[0].original_value == "No"


def test_settings_binder_keeps_numeric_one_zero_as_numbers_without_declared_bool():
    binder = SettingsBinder()

    assert binder.parse_value("Row number of the first cell", "1") == 1
    assert binder.parse_value("Column number of the first cell", "0") == 0


def test_settings_binder_binds_declared_setting_to_keyword():
    module = ModuleBlock(
        name="Example",
        module_num=1,
        settings={
            "Block size": "40.0",
            "Use correction?": "Yes",
        },
    )

    assert SettingsBinder().bind_declared(
        module,
        (
            SettingToKeywordBinding(
                "Block size",
                "block_size",
                parse_cellprofiler_int,
            ),
            SettingToKeywordBinding("Use correction?", "use_correction"),
        ),
    ) == {
        "block_size": 40,
        "use_correction": True,
    }


def test_watershed_settings_bind_nominal_method_enums():
    module = ModuleBlock(
        name="Watershed",
        module_num=1,
        settings={
            "Use advanced settings?": "No",
            "Generate from": "Markers",
            "Declump method": "Intensity",
            "Connectivity": "2",
            "Compactness": "0.25",
            "Maximum number of seeds": "15",
        },
    )

    bound = ModuleSettingsBindingStrategy.for_module("Watershed").bind(
        module,
        binder=SettingsBinder(),
        param_mapping={},
    )

    assert bound.kwargs["use_advanced_settings"] is False
    assert bound.kwargs["watershed_method"] == "markers"
    assert bound.kwargs["declump_method"] == "intensity"
    assert bound.kwargs["connectivity"] == 2
    assert bound.kwargs["compactness"] == 0.25
    assert bound.kwargs["max_seeds"] == 15
    assert bound.kwargs["structuring_element"] == "disk"
    assert bound.kwargs["structuring_element_size"] == 1


def test_watershed_settings_bind_seed_dilation_structuring_element():
    module = ModuleBlock(
        name="Watershed",
        module_num=1,
        settings={
            "Use advanced settings?": "Yes",
            "Generate from": "Distance",
            "Declump method": "Shape",
            "Structuring element for seed dilation": "Ball,5",
        },
    )

    bound = ModuleSettingsBindingStrategy.for_module("Watershed").bind(
        module,
        binder=SettingsBinder(),
        param_mapping={},
    )

    assert bound.kwargs["structuring_element"] == "ball"
    assert bound.kwargs["structuring_element_size"] == 5


def test_combine_objects_binds_overlap_policy_nominally():
    module = ModuleBlock(
        name="CombineObjects",
        module_num=5,
        settings={
            "Select initial object set": "A",
            "Select object set to combine": "B",
            "Select how to handle overlapping objects": "Merge",
            "Name the combined object set": "CombinedObjects",
        },
    )

    bound = ModuleSettingsBindingStrategy.for_module("CombineObjects").bind(
        module,
        binder=SettingsBinder(),
        param_mapping={},
    )

    assert bound.kwargs["method"] == "merge"
    assert not bound.unmapped_kwargs


def test_watershed_runtime_family_follows_module_revision():
    legacy_module = ModuleBlock(
        name="Watershed",
        module_num=1,
        settings={"Use advanced settings?": "No"},
        metadata={"variable_revision_number": "3"},
    )
    current_module = ModuleBlock(
        name="Watershed",
        module_num=1,
        settings={"Use advanced settings?": "No"},
        metadata={"variable_revision_number": "4"},
    )

    strategy = ModuleSettingsBindingStrategy.for_module("Watershed")

    assert strategy.bind(
        legacy_module,
        binder=SettingsBinder(),
        param_mapping={},
    ).kwargs["runtime_family"] == "cellprofiler4"
    assert strategy.bind(
        current_module,
        binder=SettingsBinder(),
        param_mapping={},
    ).kwargs["runtime_family"] == "library"


def test_erode_objects_binds_preservation_settings():
    module = ModuleBlock(
        name="ErodeObjects",
        module_num=1,
        settings={
            "Structuring element": "Ball,5",
            "Prevent object removal": "Yes",
            "Relabel resulting objects": "No",
        },
    )

    bound = ModuleSettingsBindingStrategy.for_module("ErodeObjects").bind(
        module,
        binder=SettingsBinder(),
        param_mapping={},
    )

    assert bound.kwargs["structuring_element"] == "ball"
    assert bound.kwargs["size"] == 5
    assert bound.kwargs["preserve_midpoints"] is True
    assert bound.kwargs["relabel_objects"] is False


def test_gray_to_color_rescale_default_follows_cellprofiler_revision_upgrade():
    v3_module = ModuleBlock(
        name="GrayToColor",
        module_num=5,
        settings={
            "Select a color scheme": "RGB",
            "Select the image to be colored red": "OrigRed",
            "Select the image to be colored green": "OrigGreen",
            "Select the image to be colored blue": "OrigBlue",
        },
        metadata={"variable_revision_number": "3"},
    )
    current_module = ModuleBlock(
        name="GrayToColor",
        module_num=5,
        settings={
            "Select a color scheme": "RGB",
            "Select the image to be colored red": "OrigRed",
            "Select the image to be colored green": "OrigGreen",
            "Select the image to be colored blue": "OrigBlue",
        },
        metadata={"variable_revision_number": "4"},
    )
    explicit_module = ModuleBlock(
        name="GrayToColor",
        module_num=5,
        settings={
            "Select a color scheme": "RGB",
            "Rescale intensity": "Yes",
            "Select the image to be colored red": "OrigRed",
            "Select the image to be colored green": "OrigGreen",
            "Select the image to be colored blue": "OrigBlue",
        },
        metadata={"variable_revision_number": "3"},
    )

    strategy = ModuleSettingsBindingStrategy.for_module("GrayToColor")

    assert strategy.bind(
        v3_module,
        binder=SettingsBinder(),
        param_mapping={},
    ).kwargs["rescale_intensity"] is False
    assert strategy.bind(
        current_module,
        binder=SettingsBinder(),
        param_mapping={},
    ).kwargs["rescale_intensity"] is True
    assert strategy.bind(
        explicit_module,
        binder=SettingsBinder(),
        param_mapping={},
    ).kwargs["rescale_intensity"] is True


def test_identify_primary_objects_binds_threshold_semantics():
    module = ModuleBlock(
        name="IdentifyPrimaryObjects",
        module_num=2,
        settings={
            "Typical diameter of objects, in pixel units (Min,Max)": "10,40",
            "Use advanced settings?": "Yes",
            "Threshold strategy": "Global",
            "Thresholding method": "Otsu",
            "Threshold smoothing scale": "1.3488",
            "Lower and upper bounds on threshold": "0,1",
            "Two-class or three-class thresholding?": "Three classes",
            "Assign pixels in the middle intensity class to the foreground or the background?": "Background",
            "Log transform before thresholding?": "No",
            "Size of adaptive window": "10",
            "Threshold setting version": "12",
        },
    )

    bound = ModuleSettingsBindingStrategy.for_module(
        "IdentifyPrimaryObjects"
    ).bind(
        module,
        binder=SettingsBinder(),
        param_mapping={
            "typical_diameter_of_objects_in_pixel_units": [
                "min_diameter",
                "max_diameter",
            ],
        },
    )

    assert bound.kwargs["min_diameter"] == 10
    assert bound.kwargs["max_diameter"] == 40
    assert bound.kwargs["use_advanced_settings"] is True
    assert bound.kwargs["threshold_scope"] == "Global"
    assert bound.kwargs["threshold_method"] == "Otsu"
    assert bound.kwargs["otsu_class_count"] == "Three classes"
    assert bound.kwargs["assign_middle_to_foreground"] == "Background"
    assert bound.kwargs["threshold_min"] == 0
    assert bound.kwargs["threshold_max"] == 1
    assert "threshold_setting_version" not in bound.unmapped_kwargs
    assert "lower_and_upper_bounds_on_threshold" not in bound.unmapped_kwargs


def test_threshold_module_binds_shared_threshold_semantics():
    module = ModuleBlock(
        name="Threshold",
        module_num=8,
        settings={
            "Select the input image": "MedianFiltDNA",
            "Name the output image": "maskDNA",
            "Threshold strategy": "Global",
            "Thresholding method": "Minimum Cross-Entropy",
            "Threshold smoothing scale": "1.25",
            "Threshold correction factor": "0.9",
            "Lower and upper bounds on threshold": "0.1,0.8",
            "Manual threshold": "0.2",
            "Select the measurement to threshold with": "None",
            "Two-class or three-class thresholding?": "Two classes",
            "Log transform before thresholding?": "No",
            "Assign pixels in the middle intensity class to the foreground or the background?": "Foreground",
            "Size of adaptive window": "25",
            "Lower outlier fraction": "0.05",
            "Upper outlier fraction": "0.05",
            "Averaging method": "Mean",
            "Variance method": "Standard deviation",
            "# of deviations": "2.0",
        },
    )

    bound = ModuleSettingsBindingStrategy.for_module("Threshold").bind(
        module,
        binder=SettingsBinder(),
        param_mapping={},
    )

    assert bound.kwargs["threshold_scope"] == "Global"
    assert bound.kwargs["threshold_method"] == "Minimum Cross-Entropy"
    assert bound.kwargs["smoothing"] == 1.25
    assert bound.kwargs["threshold_correction_factor"] == 0.9
    assert bound.kwargs["threshold_min"] == 0.1
    assert bound.kwargs["threshold_max"] == 0.8
    assert bound.kwargs["window_size"] == 25
    assert "manual_threshold" not in bound.kwargs
    assert "threshold_smoothing_scale" not in bound.kwargs
    assert "adaptive_window_size" not in bound.kwargs
    assert not bound.unmapped_kwargs


def test_identify_secondary_objects_binds_global_threshold_method_semantics():
    module = ModuleBlock(
        name="IdentifySecondaryObjects",
        module_num=9,
        settings={
            "Select the method to identify the secondary objects": "Propagation",
            "Regularization factor": "0.05",
        },
        setting_records=[
            ModuleSetting("Select the input objects", "Nuclei"),
            ModuleSetting("Name the objects to be identified", "Cells"),
            ModuleSetting(
                "Select the method to identify the secondary objects",
                "Propagation",
            ),
            ModuleSetting("Regularization factor", "0.05"),
            ModuleSetting("Threshold strategy", "Global"),
            ModuleSetting("Thresholding method", "Minimum Cross-Entropy"),
            ModuleSetting("Threshold smoothing scale", "0"),
            ModuleSetting("Threshold correction factor", "1"),
            ModuleSetting("Lower and upper bounds on threshold", "0,1"),
            ModuleSetting(
                "Two-class or three-class thresholding?",
                "Two classes",
            ),
            ModuleSetting(
                "Assign pixels in the middle intensity class to the foreground or the background?",
                "Foreground",
            ),
            ModuleSetting("Thresholding method", "Otsu"),
        ],
    )

    bound = ModuleSettingsBindingStrategy.for_module(
        "IdentifySecondaryObjects"
    ).bind(
        module,
        binder=SettingsBinder(),
        param_mapping={},
    )

    assert bound.kwargs["method"] == "Propagation"
    assert bound.kwargs["regularization_factor"] == 0.05
    assert bound.kwargs["threshold_scope"] == "Global"
    assert bound.kwargs["threshold_method"] == "Minimum Cross-Entropy"
    assert bound.kwargs["threshold_smoothing_scale"] == 0
    assert bound.kwargs["threshold_min"] == 0
    assert bound.kwargs["threshold_max"] == 1
    assert "thresholding_method" not in bound.unmapped_kwargs


def test_identify_secondary_objects_binds_adaptive_threshold_method_semantics():
    module = ModuleBlock(
        name="IdentifySecondaryObjects",
        module_num=9,
        settings={
            "Select the method to identify the secondary objects": "Propagation",
            "Regularization factor": "0.05",
        },
        setting_records=[
            ModuleSetting("Select the input objects", "Nuclei"),
            ModuleSetting("Name the objects to be identified", "Cells"),
            ModuleSetting(
                "Select the method to identify the secondary objects",
                "Propagation",
            ),
            ModuleSetting("Regularization factor", "0.05"),
            ModuleSetting("Threshold strategy", "Adaptive"),
            ModuleSetting("Thresholding method", "Minimum Cross-Entropy"),
            ModuleSetting("Threshold smoothing scale", "0"),
            ModuleSetting("Threshold correction factor", "1"),
            ModuleSetting("Lower and upper bounds on threshold", "0,1"),
            ModuleSetting(
                "Two-class or three-class thresholding?",
                "Two classes",
            ),
            ModuleSetting(
                "Assign pixels in the middle intensity class to the foreground or the background?",
                "Foreground",
            ),
            ModuleSetting("Thresholding method", "Otsu"),
        ],
    )

    bound = ModuleSettingsBindingStrategy.for_module(
        "IdentifySecondaryObjects"
    ).bind(
        module,
        binder=SettingsBinder(),
        param_mapping={},
    )

    assert bound.kwargs["threshold_scope"] == "Adaptive"
    assert bound.kwargs["threshold_method"] == "Otsu"
    assert "thresholding_method" not in bound.unmapped_kwargs


def test_legacy_three_class_otsu_synthesizes_log_transform_upgrade():
    module = ModuleBlock(
        name="IdentifySecondaryObjects",
        module_num=8,
        settings={
            "Select the method to identify the secondary objects": "Propagation",
        },
        setting_records=[
            ModuleSetting("Select the input objects", "Nuclei"),
            ModuleSetting("Name the objects to be identified", "Cells"),
            ModuleSetting(
                "Select the method to identify the secondary objects",
                "Propagation",
            ),
            ModuleSetting("Threshold setting version", "10"),
            ModuleSetting("Threshold strategy", "Global"),
            ModuleSetting("Thresholding method", "Otsu"),
            ModuleSetting("Threshold smoothing scale", "0"),
            ModuleSetting("Threshold correction factor", "1"),
            ModuleSetting("Lower and upper bounds on threshold", "0,1"),
            ModuleSetting(
                "Two-class or three-class thresholding?",
                "Three classes",
            ),
            ModuleSetting(
                "Assign pixels in the middle intensity class to the foreground or the background?",
                "Foreground",
            ),
        ],
    )

    bound = ModuleSettingsBindingStrategy.for_module(
        "IdentifySecondaryObjects"
    ).bind(
        module,
        binder=SettingsBinder(),
        param_mapping={},
    )

    assert bound.kwargs["threshold_method"] == "Otsu"
    assert bound.kwargs["otsu_class_count"] == "Three classes"
    assert bound.kwargs["log_transform"] is True
    assert "threshold_setting_version" not in bound.unmapped_kwargs


def test_legacy_duplicate_threshold_method_keeps_ordered_active_method():
    module = ModuleBlock(
        name="IdentifyPrimaryObjects",
        module_num=10,
        settings={"Use advanced settings?": "Yes"},
        setting_records=[
            ModuleSetting("Use advanced settings?", "Yes"),
            ModuleSetting("Threshold setting version", "10"),
            ModuleSetting("Threshold strategy", "Global"),
            ModuleSetting("Thresholding method", "Minimum cross entropy"),
            ModuleSetting("Threshold smoothing scale", "1.3488"),
            ModuleSetting("Threshold correction factor", "1"),
            ModuleSetting("Lower and upper bounds on threshold", "0,1"),
            ModuleSetting(
                "Two-class or three-class thresholding?",
                "Two classes",
            ),
            ModuleSetting(
                "Assign pixels in the middle intensity class to the foreground or the background?",
                "Foreground",
            ),
            ModuleSetting("Thresholding method", "Otsu"),
        ],
    )

    bound = ModuleSettingsBindingStrategy.for_module(
        "IdentifyPrimaryObjects"
    ).bind(
        module,
        binder=SettingsBinder(),
        param_mapping={},
    )

    assert bound.kwargs["threshold_method"] == "Minimum Cross-Entropy"
    assert bound.kwargs["otsu_class_count"] == "Two classes"
    assert bound.kwargs["log_transform"] is False
    assert "thresholding_method" not in bound.unmapped_kwargs


def test_legacy_threshold_method_names_are_upgraded_to_current_spellings():
    module = ModuleBlock(
        name="IdentifyPrimaryObjects",
        module_num=7,
        settings={
            "Use advanced settings?": "Yes",
            "Threshold setting version": "10",
            "Threshold strategy": "Global",
            "Thresholding method": "RobustBackground",
            "Threshold smoothing scale": "1.3488",
            "Threshold correction factor": "1",
            "Lower and upper bounds on threshold": "0,1",
            "Two-class or three-class thresholding?": "Two classes",
        },
    )

    bound = ModuleSettingsBindingStrategy.for_module(
        "IdentifyPrimaryObjects"
    ).bind(
        module,
        binder=SettingsBinder(),
        param_mapping={},
    )

    assert bound.kwargs["threshold_method"] == "Robust Background"
    assert bound.kwargs["log_transform"] is False
    assert "threshold_setting_version" not in bound.unmapped_kwargs


def test_enhance_or_suppress_features_settings_bind_to_runtime_kwargs():
    module = ModuleBlock(
        name="EnhanceOrSuppressFeatures",
        module_num=6,
        settings={
            "Select the input image": "OrigGreen",
            "Name the output image": "EnhancedGreen",
            "Select the operation": "Enhance",
            "Feature size": "10",
            "Feature type": "Speckles",
            "Range of hole sizes": "1,10",
            "Smoothing scale": "2.0",
            "Shear angle": "0.0",
            "Decay": "0.95",
            "Enhancement method": "Tubeness",
            "Speed and accuracy": "Fast",
        },
    )

    bound = ModuleSettingsBindingStrategy.for_module(
        "EnhanceOrSuppressFeatures"
    ).bind(
        module,
        binder=SettingsBinder(),
        param_mapping={},
    )

    assert bound.kwargs["method"] == "Enhance"
    assert bound.kwargs["radius"] == 5
    assert bound.kwargs["enhance_method"] == "Speckles"
    assert bound.kwargs["dark_hole_radius_min"] == 1
    assert bound.kwargs["dark_hole_radius_max"] == 10
    assert bound.kwargs["smoothing_value"] == 2.0
    assert bound.kwargs["dic_decay"] == 0.95
    assert bound.kwargs["neurite_method"] == "Tubeness"
    assert bound.kwargs["speckle_accuracy"] == "Fast"
    assert not bound.unmapped_kwargs


def test_smooth_settings_bind_to_runtime_kwargs():
    module = ModuleBlock(
        name="Smooth",
        module_num=10,
        settings={
            "Select the input image": "MaskBF",
            "Name the output image": "SmoothedBF",
            "Select smoothing method": "Gaussian Filter",
            "Calculate artifact diameter automatically?": "No",
            "Typical artifact diameter": "3.0",
            "Edge intensity difference": "0.1",
            "Clip intensities to 0 and 1?": "Yes",
        },
    )

    bound = ModuleSettingsBindingStrategy.for_module("Smooth").bind(
        module,
        binder=SettingsBinder(),
        param_mapping={},
    )

    assert bound.kwargs == {
        "smoothing_method": "Gaussian Filter",
        "auto_object_size": False,
        "object_size": 3.0,
        "edge_intensity_difference": 0.1,
        "clip_polynomial": True,
    }
    assert not bound.unmapped_kwargs


def test_enhance_edges_settings_bind_to_runtime_kwargs():
    module = ModuleBlock(
        name="EnhanceEdges",
        module_num=11,
        settings={
            "Select the input image": "SmoothedBF",
            "Name the output image": "EdgedImage",
            "Automatically calculate the threshold?": "Yes",
            "Absolute threshold": "0.2",
            "Threshold adjustment factor": "1.0",
            "Select an edge-finding method": "Sobel",
            "Select edge direction to enhance": "All",
            "Calculate Gaussian's sigma automatically?": "No",
            "Gaussian's sigma value": "10.0",
            "Calculate value for low threshold automatically?": "Yes",
            "Low threshold value": "0.1",
        },
    )

    bound = ModuleSettingsBindingStrategy.for_module("EnhanceEdges").bind(
        module,
        binder=SettingsBinder(),
        param_mapping={},
    )

    assert bound.kwargs == {
        "automatic_threshold": True,
        "manual_threshold": 0.2,
        "threshold_adjustment_factor": 1.0,
        "method": "Sobel",
        "direction": "All",
        "automatic_gaussian": False,
        "sigma": 10.0,
        "automatic_low_threshold": True,
        "low_threshold": 0.1,
    }
    assert not bound.unmapped_kwargs


def test_color_to_gray_split_kwargs_use_enabled_rgb_channels():
    module = ModuleBlock(
        name="ColorToGray",
        module_num=6,
        setting_records=[
            ModuleSetting("Select the input image", "ColorFluor"),
            ModuleSetting("Conversion method", "Split"),
            ModuleSetting("Image type", "RGB"),
            ModuleSetting("Name the output image", "OrigGray"),
            ModuleSetting("Relative weight of the red channel", "1.0"),
            ModuleSetting("Relative weight of the green channel", "1.0"),
            ModuleSetting("Relative weight of the blue channel", "1.0"),
            ModuleSetting("Convert red to gray?", "No"),
            ModuleSetting("Name the output image", "OrigRed"),
            ModuleSetting("Convert green to gray?", "Yes"),
            ModuleSetting("Name the output image", "GrayTumor"),
            ModuleSetting("Convert blue to gray?", "No"),
            ModuleSetting("Name the output image", "OrigBlue"),
        ],
    )

    bound = ModuleSettingsBindingStrategy.for_module("ColorToGray").bind(
        module,
        binder=SettingsBinder(),
        param_mapping={},
    )

    assert bound.kwargs["mode"] == "split"
    assert bound.kwargs["channel_indices"] == (1,)
    assert bound.kwargs["contributions"] == (1.0,)


def test_robust_background_threshold_binds_fractional_deviations():
    module = ModuleBlock(
        name="IdentifyPrimaryObjects",
        module_num=7,
        settings={
            "Use advanced settings?": "Yes",
            "Threshold setting version": "10",
            "Threshold strategy": "Global",
            "Thresholding method": "RobustBackground",
            "Threshold smoothing scale": "1.3488",
            "Threshold correction factor": "1",
            "Lower and upper bounds on threshold": "0,1",
            "Two-class or three-class thresholding?": "Two classes",
            "# of deviations": "0.75",
        },
    )

    bound = ModuleSettingsBindingStrategy.for_module(
        "IdentifyPrimaryObjects"
    ).bind(
        module,
        binder=SettingsBinder(),
        param_mapping={},
    )

    assert bound.kwargs["number_of_deviations"] == 0.75


def test_mask_objects_binds_masking_policy_semantics():
    module = ModuleBlock(
        name="MaskObjects",
        module_num=10,
        settings={
            "Handling of objects that are partially masked": "Keep overlapping region",
            "Fraction of object that must overlap": "0.5",
            "Numbering of resulting objects": "Renumber",
            "Invert the mask?": "Yes",
        },
    )

    bound = ModuleSettingsBindingStrategy.for_module("MaskObjects").bind(
        module,
        binder=SettingsBinder(),
        param_mapping={},
    )

    assert bound.kwargs == {
        "overlap_handling": "keep_overlapping_region",
        "overlap_fraction": 0.5,
        "numbering": "renumber",
        "invert_mask": True,
    }


def test_measure_texture_binds_repeated_texture_scales():
    module = ModuleBlock(
        name="MeasureTexture",
        module_num=12,
        settings={
            "Measure images or objects?": "Both",
        },
        setting_records=[
            ModuleSetting("Measure images or objects?", "Both"),
            ModuleSetting("Texture scale to measure", "5"),
            ModuleSetting("Texture scale to measure", "10"),
            ModuleSetting("Texture scale to measure", "20"),
            ModuleSetting(
                "Enter how many gray levels to measure the texture at",
                "128",
            ),
        ],
    )

    bound = ModuleSettingsBindingStrategy.for_module("MeasureTexture").bind(
        module,
        binder=SettingsBinder(),
        param_mapping={},
    )

    assert bound.kwargs["scale"] == (5, 10, 20)
    assert bound.kwargs["gray_levels"] == 128


def test_measure_texture_ignores_legacy_gabor_ui_settings():
    module = ModuleBlock(
        name="MeasureTexture",
        module_num=12,
        settings={
            "Angles to measure": "Horizontal, Vertical, Diagonal, Anti-diagonal",
            "Measure Gabor features?": "Yes",
            "Number of angles to compute for Gabor": "4",
        },
    )

    bound = ModuleSettingsBindingStrategy.for_module("MeasureTexture").bind(
        module,
        binder=SettingsBinder(),
        param_mapping={},
    )

    assert bound.unmapped_kwargs == {}


def test_measure_granularity_binds_shared_spectrum_settings():
    setting_records = [
        ModuleSetting("Select an image to measure", "BF_image"),
        ModuleSetting("Subsampling factor for granularity measurements", "1"),
        ModuleSetting("Subsampling factor for background reduction", "0.25"),
        ModuleSetting("Radius of structuring element", "10"),
        ModuleSetting("Range of the granular spectrum", "5"),
        ModuleSetting("Select objects to measure", "Cells"),
        ModuleSetting("Select an image to measure", "Marker_image"),
        ModuleSetting("Subsampling factor for granularity measurements", "1"),
        ModuleSetting("Subsampling factor for background reduction", "0.25"),
        ModuleSetting("Radius of structuring element", "10"),
        ModuleSetting("Range of the granular spectrum", "5"),
        ModuleSetting("Select objects to measure", "Cells"),
    ]
    module = ModuleBlock(
        name="MeasureGranularity",
        module_num=24,
        settings={setting.name: setting.value for setting in setting_records},
        setting_records=setting_records,
    )

    bound = ModuleSettingsBindingStrategy.for_module("MeasureGranularity").bind(
        module,
        binder=SettingsBinder(),
        param_mapping={},
    )

    assert bound.kwargs == {
        "subsample_size": 1.0,
        "background_subsample_size": 0.25,
        "element_radius": 10,
        "spectrum_length": 5,
    }


def test_measure_image_quality_ignores_inactive_all_images_selector():
    module = ModuleBlock(
        name="MeasureImageQuality",
        module_num=5,
        settings={
            "Calculate metrics for which images?": "All loaded images",
            "Select the images to measure": "",
        },
        setting_records=[
            ModuleSetting(
                "Calculate metrics for which images?",
                "All loaded images",
            ),
            ModuleSetting("Select the images to measure", ""),
        ],
    )

    bound = ModuleSettingsBindingStrategy.for_module("MeasureImageQuality").bind(
        module,
        binder=SettingsBinder(),
        param_mapping={},
    )

    assert bound.unmapped_kwargs == {}


def test_measure_image_quality_rejects_active_unmapped_image_selector():
    module = ModuleBlock(
        name="MeasureImageQuality",
        module_num=5,
        settings={
            "Calculate metrics for which images?": "Select...",
            "Select the images to measure": "",
        },
        setting_records=[
            ModuleSetting("Calculate metrics for which images?", "Select..."),
            ModuleSetting("Select the images to measure", ""),
        ],
    )

    with pytest.raises(UnmappedModuleSettingsError) as exc:
        ModuleSettingsBindingStrategy.for_module("MeasureImageQuality").bind(
            module,
            binder=SettingsBinder(),
            param_mapping={},
        )

    assert "select_the_images_to_measure" in str(exc.value)


def test_relate_objects_binds_distance_setting():
    module = ModuleBlock(
        name="RelateObjects",
        module_num=18,
        settings={
            "Calculate child-parent distances?": "None",
        },
        setting_records=[
            ModuleSetting("Select the parent objects", "Tiles"),
            ModuleSetting("Select the child objects", "Cells"),
            ModuleSetting("Calculate child-parent distances?", "None"),
        ],
    )

    bound = ModuleSettingsBindingStrategy.for_module("RelateObjects").bind(
        module,
        binder=SettingsBinder(),
        param_mapping={},
    )

    assert bound.kwargs["calculate_distances"] == "none"
    assert "calculate_child_parent_distances" not in bound.unmapped_kwargs


def test_correct_illumination_binds_legacy_object_size_alias():
    module = ModuleBlock(
        name="CorrectIlluminationCalculate",
        module_num=5,
        settings={
            "Approximate object size": "10",
        },
        setting_records=[
            ModuleSetting("Approximate object size", "10"),
        ],
    )

    bound = ModuleSettingsBindingStrategy.for_module(
        "CorrectIlluminationCalculate"
    ).bind(
        module,
        binder=SettingsBinder(),
        param_mapping={},
    )

    assert bound.kwargs["object_width"] == 10
    assert "approximate_object_size" not in bound.unmapped_kwargs


def test_identify_primary_objects_consumes_legacy_input_output_aliases():
    module = ModuleBlock(
        name="IdentifyPrimaryObjects",
        module_num=7,
        settings={
            "Input": "CorrGray",
            "Object": "Comet",
        },
        setting_records=[
            ModuleSetting("Input", "CorrGray"),
            ModuleSetting("Object", "Comet"),
        ],
    )

    bound = ModuleSettingsBindingStrategy.for_module("IdentifyPrimaryObjects").bind(
        module,
        binder=SettingsBinder(),
        param_mapping={},
    )

    assert bound.unmapped_kwargs == {}


def test_measure_image_intensity_ignores_blank_legacy_object_selector():
    module = ModuleBlock(
        name="MeasureImageIntensity",
        module_num=16,
        settings={
            "Select the input objects": "None",
        },
        setting_records=[
            ModuleSetting("Select the input objects", "None"),
        ],
    )

    bound = ModuleSettingsBindingStrategy.for_module("MeasureImageIntensity").bind(
        module,
        binder=SettingsBinder(),
        param_mapping={},
    )

    assert bound.unmapped_kwargs == {}


def test_measure_colocalization_binds_legacy_rank_weighted_typo():
    module = ModuleBlock(
        name="MeasureColocalization",
        module_num=9,
        settings={
            "Calculate the Rank Weighted Coloalization coefficients?": "Yes",
        },
        setting_records=[
            ModuleSetting(
                "Calculate the Rank Weighted Coloalization coefficients?",
                "Yes",
            ),
        ],
    )

    bound = ModuleSettingsBindingStrategy.for_module(
        "MeasureColocalization"
    ).bind(
        module,
        binder=SettingsBinder(),
        param_mapping={},
    )

    assert bound.kwargs["do_rwc"] is True
    assert (
        "calculate_the_rank_weighted_coloalization_coefficients"
        not in bound.unmapped_kwargs
    )


def test_measure_colocalization_ignores_inactive_legacy_object_selector():
    module = ModuleBlock(
        name="MeasureColocalization",
        module_num=9,
        settings={
            "Select an object to measure": "None",
            "Hidden": "1",
        },
        setting_records=[
            ModuleSetting("Select an object to measure", "None"),
            ModuleSetting("Hidden", "1"),
        ],
    )

    bound = ModuleSettingsBindingStrategy.for_module(
        "MeasureColocalization"
    ).bind(
        module,
        binder=SettingsBinder(),
        param_mapping={},
    )

    assert bound.unmapped_kwargs == {}


def test_measure_colocalization_preserves_active_legacy_object_selector_semantics():
    module = ModuleBlock(
        name="MeasureColocalization",
        module_num=9,
        settings={
            "Select an object to measure": "Nuclei",
        },
        setting_records=[
            ModuleSetting("Select an object to measure", "Nuclei"),
        ],
    )

    ignored = ModuleUnmappedSettingIgnore.ignored_setting_names_for_module(module)
    bound = ModuleSettingsBindingStrategy.for_module("MeasureColocalization").bind(
        module,
        binder=SettingsBinder(),
        param_mapping={},
    )

    assert "select_an_object_to_measure" not in ignored
    assert bound.unmapped_kwargs == {}


def test_measure_object_size_shape_binds_zernike_toggle():
    module = ModuleBlock(
        name="MeasureObjectSizeShape",
        module_num=7,
        settings={
            "Select objects to measure": "Embryos",
            "Calculate the Zernike features?": "No",
            "Calculate the advanced features?": "No",
        },
    )

    bound = ModuleSettingsBindingStrategy.for_module("MeasureObjectSizeShape").bind(
        module,
        binder=SettingsBinder(),
        param_mapping={},
    )

    assert bound.kwargs == {
        "calculate_zernikes": False,
        "calculate_advanced": False,
    }
    assert bound.unmapped_kwargs == {}


def test_measure_object_intensity_distribution_binds_scalar_settings():
    module = ModuleBlock(
        name="MeasureObjectIntensityDistribution",
        module_num=29,
        settings={
            "Calculate intensity Zernikes?": "Magnitudes and phase",
            "Maximum zernike moment": "9",
            "Object to use as center?": "These objects",
            "Scale the bins?": "Yes",
            "Number of bins": "4",
            "Maximum radius": "100",
        },
    )

    bound = ModuleSettingsBindingStrategy.for_module(
        "MeasureObjectIntensityDistribution"
    ).bind(
        module,
        binder=SettingsBinder(),
        param_mapping={},
    )

    assert bound.kwargs == {
        "wants_zernikes": "magnitudes_and_phase",
        "zernike_degree": 9,
        "wants_scaled": True,
        "bin_count": 4,
        "maximum_radius": 100,
        "center_choice": "self",
    }


def test_measure_object_neighbors_binds_distance_semantics():
    setting_records = [
        ModuleSetting("Select objects to measure", "Nuclei"),
        ModuleSetting("Select neighboring objects to measure", "Nuclei"),
        ModuleSetting("Method to determine neighbors", "Within a specified distance"),
        ModuleSetting("Neighbor distance", "4"),
        ModuleSetting(
            "Consider objects discarded for touching image border?",
            "Yes",
        ),
        ModuleSetting(
            "Retain the image of objects colored by numbers of neighbors?",
            "Yes",
        ),
        ModuleSetting("Name the output image", "ColorNeighbors"),
        ModuleSetting("Select colormap", "hot"),
        ModuleSetting(
            "Retain the image of objects colored by percent of touching pixels?",
            "No",
        ),
        ModuleSetting("Name the output image", "PercentTouching"),
        ModuleSetting("Select colormap", "Default"),
    ]
    module = ModuleBlock(
        name="MeasureObjectNeighbors",
        module_num=14,
        settings={
            setting.name: setting.value for setting in setting_records
        },
        setting_records=setting_records,
    )

    bound = ModuleSettingsBindingStrategy.for_module(
        "MeasureObjectNeighbors"
    ).bind(
        module,
        binder=SettingsBinder(),
        param_mapping={
            "select_objects_to_measure": None,
            "select_neighboring_objects_to_measure": None,
            "method_to_determine_neighbors": "distance_method",
            "neighbor_distance": "neighbor_distance",
            "consider_objects_discarded_for_touching_image_border": (
                "consider_discarded_objects"
            ),
        },
    )

    assert bound.kwargs == {
        "distance_method": "Within a specified distance",
        "neighbor_distance": 4,
        "consider_discarded_objects": True,
        "retain_neighbor_count_image": True,
        "neighbor_count_colormap": "hot",
        "retain_percent_touching_image": False,
        "percent_touching_colormap": "Default",
    }
    assert bound.unmapped_kwargs == {}


def test_image_math_binds_operation_semantics():
    module = ModuleBlock(
        name="ImageMath",
        module_num=7,
        settings={
            "Operation": "Invert",
            "Raise the power of the result by": "1.0",
            "Multiply the result by": "1.0",
            "Add to result": "0.0",
            "Set values less than 0 equal to 0?": "Yes",
            "Set values greater than 1 equal to 1?": "Yes",
            "Replace invalid values with 0?": "Yes",
            "Ignore the image masks?": "No",
            "Multiply the first image by": "1.0",
            "Multiply the second image by": "1.0",
        },
    )

    bound = ModuleSettingsBindingStrategy.for_module("ImageMath").bind(
        module,
        binder=SettingsBinder(),
        param_mapping={},
    )

    assert bound.kwargs == {
        "operation": MathOperation.INVERT,
        "exponent": 1.0,
        "after_factor": 1.0,
        "addend": 0.0,
        "truncate_low": True,
        "truncate_high": True,
        "replace_nan": True,
        "ignore_masks": False,
        "factors": (1.0, 1.0),
    }
    assert bound.unmapped_kwargs == {}


def test_expand_or_shrink_objects_binds_operation_semantics():
    module = ModuleBlock(
        name="ExpandOrShrinkObjects",
        module_num=17,
        settings={
            "Select the input objects": "Filtered_tiles",
            "Name the output objects": "Non_empty_tile",
            "Select the operation": "Shrink objects by a specified number of pixels",
            "Number of pixels by which to expand or shrink": "1",
            "Fill holes in objects so that all objects shrink to a single point?": "No",
        },
    )

    bound = ModuleSettingsBindingStrategy.for_module("ExpandOrShrinkObjects").bind(
        module,
        binder=SettingsBinder(),
        param_mapping={},
    )

    assert bound.kwargs == {
        "mode": "shrink_defined_pixels",
        "iterations": 1,
        "fill_holes": False,
    }
    assert bound.unmapped_kwargs == {}


def test_measure_colocalization_binds_metric_and_costes_semantics():
    module = ModuleBlock(
        name="MeasureColocalization",
        module_num=15,
        settings={
            "Select images to measure": "CropBlue, CropGreen",
            "Set threshold as percentage of maximum intensity for the images": "15.0",
            "Select where to measure correlation": "Both",
            "Select objects to measure": "Nuclei",
            "Run all metrics?": "Accurate",
            "Calculate correlation and slope metrics?": "No",
            "Calculate the Manders coefficients?": "No",
            "Calculate the Rank Weighted Colocalization coefficients?": "No",
            "Calculate the Overlap coefficients?": "No",
            "Calculate the Manders coefficients using Costes auto threshold?": "No",
            "Method for Costes thresholding": "Fast",
        },
    )

    bound = ModuleSettingsBindingStrategy.for_module(
        "MeasureColocalization"
    ).bind(
        module,
        binder=SettingsBinder(),
        param_mapping={
            "select_images_to_measure": None,
            "select_objects_to_measure": None,
        },
    )

    assert bound.kwargs == {
        "_cellprofiler_measurement_target_scope": "both",
        "threshold_percent": 15.0,
        "do_correlation": True,
        "do_manders": True,
        "do_rwc": True,
        "do_overlap": True,
        "do_costes": True,
        "costes_method": "fast",
    }
    assert bound.unmapped_kwargs == {}


def test_tile_binds_within_cycles_to_row_montage_geometry():
    module = ModuleBlock(
        name="Tile",
        module_num=11,
        settings={
            "Select an input image": "OrigColor",
            "Name the output image": "AdjacentImage",
            "Tile assembly method": "Within cycles",
        },
    )

    bound = ModuleSettingsBindingStrategy.for_module("Tile").bind(
        module,
        binder=SettingsBinder(),
        param_mapping={},
    )

    assert bound.kwargs == {
        "rows": 1,
        "columns": 1,
        "place_first": "top_left",
        "tile_style": "row",
        "meander": False,
        "auto_rows": False,
        "auto_columns": True,
    }
    assert bound.unmapped_kwargs == {}


def test_tile_rejects_unsupported_assembly_method():
    module = ModuleBlock(
        name="Tile",
        module_num=11,
        settings={
            "Tile assembly method": "Across cycles",
        },
    )

    with pytest.raises(NotImplementedError, match="Across cycles"):
        ModuleSettingsBindingStrategy.for_module("Tile").bind(
            module,
            binder=SettingsBinder(),
            param_mapping={},
        )


def test_track_objects_binds_tracking_identity_settings():
    module = ModuleBlock(
        name="TrackObjects",
        module_num=9,
        settings={
            "Choose a tracking method": "Overlap",
            "Select the objects to track": "Embryos",
            "Maximum pixel distance to consider matches": "50",
        },
    )

    bound = ModuleSettingsBindingStrategy.for_module("TrackObjects").bind(
        module,
        binder=SettingsBinder(),
        param_mapping={},
    )

    assert bound.kwargs["object_name"] == "Embryos"
    assert bound.kwargs["tracking_method"] == "overlap"
    assert bound.kwargs["pixel_radius"] == 50


def test_resize_binds_factor_and_interpolation_settings():
    module = ModuleBlock(
        name="Resize",
        module_num=5,
        settings={
            "Resizing method": "Resize by a fraction or multiple of the original size",
            "Resizing factor": "2.0",
            "Width of the final image": "100",
            "Height of the final image": "100",
            "Interpolation method": "Nearest Neighbor",
        },
    )

    bound = ModuleSettingsBindingStrategy.for_module("Resize").bind(
        module,
        binder=SettingsBinder(),
        param_mapping={},
    )

    assert bound.kwargs == {
        "resize_method": "by_factor",
        "resizing_factor_x": 2.0,
        "resizing_factor_y": 2.0,
        "specific_width": 100,
        "specific_height": 100,
        "interpolation": "nearest_neighbor",
    }


def test_resize_objects_binds_volumetric_factor_settings():
    setting_values = {
        "method": "Factor",
        "factor_x": "2",
        "factor_y": "2",
        "factor_z": "1.0",
        "width": "100",
        "height": "100",
        "planes": "10",
    }
    module = ModuleBlock(
        name="ResizeObjects",
        module_num=11,
        settings={
            setting_names(binding.setting_name)[0]: setting_values[
                binding.parameter_name
            ]
            for binding in RESIZE_OBJECTS_SETTINGS
        },
    )

    bound = ModuleSettingsBindingStrategy.for_module("ResizeObjects").bind(
        module,
        binder=SettingsBinder(),
        param_mapping={},
    )

    assert bound.kwargs == {
        "method": "factor",
        "factor_x": 2.0,
        "factor_y": 2.0,
        "factor_z": 1.0,
        "width": 100,
        "height": 100,
        "planes": 10,
    }


def test_median_filter_binds_window_size():
    module = ModuleBlock(
        name="MedianFilter",
        module_num=7,
        settings={"Window": "5"},
    )

    bound = ModuleSettingsBindingStrategy.for_module("MedianFilter").bind(
        module,
        binder=SettingsBinder(),
        param_mapping={},
    )

    assert bound.kwargs == {"window_size": 5}


def test_rescale_intensity_binds_source_range_and_nominal_modes():
    module = ModuleBlock(
        name="RescaleIntensity",
        module_num=5,
        settings={
            "Select the input image": "origDNA",
            "Name the output image": "RescaledDNA",
            "Rescaling method": "Stretch each image to use the full intensity range",
            "Method to calculate the minimum intensity": "Custom",
            "Method to calculate the maximum intensity": "Custom",
            "Lower intensity limit for the input image": "0.0",
            "Upper intensity limit for the input image": "1.0",
            "Intensity range for the input image": "0.0,1.0",
            "Intensity range for the output image": "0.0,1.0",
            "Select image to match in maximum intensity": "None",
            "Divisor value": "1.0",
            "Divisor measurement": "None",
        },
    )

    bound = ModuleSettingsBindingStrategy.for_module("RescaleIntensity").bind(
        module,
        binder=SettingsBinder(),
        param_mapping={},
    )

    assert bound.kwargs == {
        "rescale_method": "stretch",
        "automatic_low": "custom",
        "automatic_high": "custom",
        "source_low": 0.0,
        "source_high": 1.0,
        "dest_low": 0.0,
        "dest_high": 1.0,
        "divisor_value": 1.0,
    }
    assert not bound.unmapped_kwargs


def test_mask_image_binds_mask_source_and_inversion():
    module = ModuleBlock(
        name="MaskImage",
        module_num=23,
        settings={
            "Select the input image": "MembInvertRemoveHoles",
            "Name the output image": "MembMasked",
            "Use objects or an image as a mask?": "Image",
            "Select object for mask": "None",
            "Select image for mask": "MonolayerMask",
            "Invert the mask?": "No",
        },
    )

    bound = ModuleSettingsBindingStrategy.for_module("MaskImage").bind(
        module,
        binder=SettingsBinder(),
        param_mapping={},
    )

    assert bound.kwargs == {
        "mask_source": "image",
        "invert_mask": False,
    }
    assert not bound.unmapped_kwargs


def test_mask_objects_ignores_inactive_outline_output_name():
    module = ModuleBlock(
        name="MaskObjects",
        module_num=10,
        settings={
            "Retain outlines of the resulting objects?": "No",
            "Name the outline image": "MaskedOutlines",
        },
        setting_records=[
            ModuleSetting("Retain outlines of the resulting objects?", "No"),
            ModuleSetting("Name the outline image", "MaskedOutlines"),
        ],
    )

    bound = ModuleSettingsBindingStrategy.for_module("MaskObjects").bind(
        module,
        binder=SettingsBinder(),
        param_mapping={},
    )

    assert bound.unmapped_kwargs == {}


def test_overlay_objects_binds_opacity_and_ignores_contract_routing():
    module = ModuleBlock(
        name="OverlayObjects",
        module_num=28,
        settings={
            "Input": "RescaledDNA",
            "Name the output image": "NucleiOverlay",
            "Objects": "Nuclei",
            "Opacity": "0.2",
        },
    )

    bound = ModuleSettingsBindingStrategy.for_module("OverlayObjects").bind(
        module,
        binder=SettingsBinder(),
        param_mapping={},
    )

    assert bound.kwargs == {"opacity": 0.2}
    assert not bound.unmapped_kwargs


def test_measure_object_intensity_declares_contract_routing_settings():
    module = ModuleBlock(
        name="MeasureObjectIntensity",
        module_num=26,
        settings={
            "Select images to measure": "origDNA,origMemb",
            "Select objects to measure": "Nuclei,Cells",
        },
    )

    bound = ModuleSettingsBindingStrategy.for_module("MeasureObjectIntensity").bind(
        module,
        binder=SettingsBinder(),
        param_mapping={},
    )

    assert not bound.kwargs
    assert not bound.unmapped_kwargs


def test_module_settings_binding_rejects_unmapped_settings():
    module = ModuleBlock(
        name="Generic",
        module_num=3,
        settings={"Unexpected setting": "42"},
    )

    with pytest.raises(UnmappedModuleSettingsError) as exc:
        ModuleSettingsBindingStrategy.for_module("Generic").bind(
            module,
            binder=SettingsBinder(),
            param_mapping={},
        )

    assert "Generic(3).unexpected_setting=42" in str(exc.value)


def test_module_settings_binding_allows_derived_dead_output_setting_ignore():
    module = ModuleBlock(
        name="Generic",
        module_num=3,
        settings={"Name the output image": "DeadOutput"},
    )

    bound = ModuleSettingsBindingStrategy.for_module("Generic").bind(
        module,
        binder=SettingsBinder(),
        param_mapping={},
        ignored_unmapped_settings=frozenset({"name_the_output_image"}),
    )

    assert not bound.unmapped_kwargs


def test_remove_holes_binds_hole_diameter():
    module = ModuleBlock(
        name="RemoveHoles",
        module_num=9,
        settings={"Size of holes to fill": "20.0"},
    )

    bound = ModuleSettingsBindingStrategy.for_module("RemoveHoles").bind(
        module,
        binder=SettingsBinder(),
        param_mapping={},
    )

    assert bound.kwargs == {"diameter": 20.0}
