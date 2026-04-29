from enum import Enum

from benchmark.converter.settings_binder import (
    SettingToKeywordBinding,
    SettingsBinder,
    normalize_cellprofiler_setting_name,
    parse_cellprofiler_int,
)
from benchmark.converter.parser import ModuleBlock


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
