"""Shared CellProfiler Watershed setting identifiers."""

from __future__ import annotations

from openhcs.interop.cellprofiler.setting_names import SettingNameFamily


WATERSHED_METHOD_SETTING = SettingNameFamily("Generate from")
WATERSHED_USE_ADVANCED_SETTINGS_SETTING = SettingNameFamily("Use advanced settings?")
WATERSHED_DECLUMP_METHOD_SETTING = SettingNameFamily("Declump method")
WATERSHED_MARKERS_SETTING = SettingNameFamily("Markers")
WATERSHED_MASK_SETTING = SettingNameFamily("Mask")
WATERSHED_INTENSITY_IMAGE_SETTING = SettingNameFamily(
    "Intensity image",
    aliases=("Reference Image",),
)
WATERSHED_CONNECTIVITY_SETTING = SettingNameFamily("Connectivity")
WATERSHED_COMPACTNESS_SETTING = SettingNameFamily("Compactness")
WATERSHED_FOOTPRINT_SETTING = SettingNameFamily("Footprint")
WATERSHED_DOWNSAMPLE_SETTING = SettingNameFamily("Downsample")
WATERSHED_LABEL_SEPARATION_SETTING = SettingNameFamily("Separate watershed labels")
WATERSHED_BORDER_EXCLUSION_SETTING = SettingNameFamily(
    "Discard objects touching the border of the image?",
    aliases=("Pixels from border to exclude",),
)
WATERSHED_MINIMUM_SEED_DISTANCE_SETTING = SettingNameFamily(
    "Minimum distance between seeds"
)
WATERSHED_MINIMUM_INTERNAL_DISTANCE_SETTING = SettingNameFamily(
    "Minimum absolute internal distance"
)
WATERSHED_SMOOTHING_FACTOR_SETTING = SettingNameFamily(
    "Segmentation distance transform smoothing factor"
)
WATERSHED_MAX_SEEDS_SETTING = SettingNameFamily("Maximum number of seeds")
WATERSHED_STRUCTURING_ELEMENT_SETTING = SettingNameFamily(
    "Structuring element for seed dilation"
)


def parse_watershed_border_exclusion(value: str) -> bool:
    """Parse legacy and current Watershed border-exclusion settings."""
    normalized = value.strip().lower()
    if normalized in {"yes", "true", "1", "on"}:
        return True
    if normalized in {"no", "false", "0", "off"}:
        return False
    pixel_count = int(value)
    if pixel_count == 0:
        return False
    raise ValueError(
        "OpenHCS Watershed only supports CellProfiler border exclusion as a "
        f"boolean edge clear, got pixel border width {pixel_count!r}."
    )
