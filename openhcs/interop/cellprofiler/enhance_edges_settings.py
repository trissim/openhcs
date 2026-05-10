"""CellProfiler EnhanceEdges module setting lowering."""

from __future__ import annotations

from .settings_binder import SettingToKeywordBinding


ENHANCE_EDGES_SETTINGS: tuple[SettingToKeywordBinding, ...] = (
    SettingToKeywordBinding(
        "Automatically calculate the threshold?",
        "automatic_threshold",
    ),
    SettingToKeywordBinding("Absolute threshold", "manual_threshold"),
    SettingToKeywordBinding(
        "Threshold adjustment factor",
        "threshold_adjustment_factor",
    ),
    SettingToKeywordBinding("Select an edge-finding method", "method"),
    SettingToKeywordBinding("Select edge direction to enhance", "direction"),
    SettingToKeywordBinding(
        "Calculate Gaussian's sigma automatically?",
        "automatic_gaussian",
    ),
    SettingToKeywordBinding("Gaussian's sigma value", "sigma"),
    SettingToKeywordBinding(
        "Calculate value for low threshold automatically?",
        "automatic_low_threshold",
    ),
    SettingToKeywordBinding("Low threshold value", "low_threshold"),
)

__all__ = ("ENHANCE_EDGES_SETTINGS",)
