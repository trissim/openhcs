"""CellProfiler Smooth module setting lowering."""

from __future__ import annotations

from .settings_binder import SettingToKeywordBinding


SMOOTH_SETTINGS: tuple[SettingToKeywordBinding, ...] = (
    SettingToKeywordBinding("Select smoothing method", "smoothing_method"),
    SettingToKeywordBinding(
        "Calculate artifact diameter automatically?",
        "auto_object_size",
    ),
    SettingToKeywordBinding("Typical artifact diameter", "object_size"),
    SettingToKeywordBinding(
        "Edge intensity difference",
        "edge_intensity_difference",
    ),
    SettingToKeywordBinding(
        "Clip intensities to 0 and 1?",
        "clip_polynomial",
    ),
)

__all__ = ("SMOOTH_SETTINGS",)
