"""
Centralized config preview formatters for UI widgets.

Single source of truth for how config fields are formatted in preview labels
across PipelineEditor, PlateManager, and other widgets.

Preview labels are now declared on config classes via @global_pipeline_config(preview_label="NAP")
and registered in PREVIEW_LABEL_REGISTRY. Field abbreviations are declared via
@global_pipeline_config(field_abbreviations={'well_filter': 'wf'}) and registered in
FIELD_ABBREVIATIONS_REGISTRY. This file provides formatting utilities only.
"""

from dataclasses import dataclass
from typing import Any, Optional, Callable

from pyqt_reactive.utils.preview_formatters import (
    check_enabled_field as _check_enabled_field,
    format_preview_value,
)
from objectstate.lazy_factory import PREVIEW_LABEL_REGISTRY, FIELD_ABBREVIATIONS_REGISTRY


@dataclass(frozen=True)
class PreviewLabelAuthority:
    """Resolves preview labels from the configured type registry."""

    labels_by_type: dict[type, str]

    def label_for(self, config: Any) -> Optional[str]:
        return self.label_for_type(type(config))

    def label_for_type(self, config_type: type) -> Optional[str]:
        if config_type in self.labels_by_type:
            return self.labels_by_type[config_type]

        for base in config_type.__mro__[1:]:
            if base in self.labels_by_type:
                return self.labels_by_type[base]

        return None


PREVIEW_LABELS = PreviewLabelAuthority(PREVIEW_LABEL_REGISTRY)


@dataclass(frozen=True)
class ConfigPreviewIndicatorAuthority:
    """Builds preview indicators from label metadata and enabled policy."""

    labels: PreviewLabelAuthority

    def indicator_for(
        self,
        config: Any,
        resolve_attr: Optional[Callable] = None,
    ) -> Optional[str]:
        indicator = self.labels.label_for(config)
        if indicator is None:
            return None
        if not _check_enabled_field(config, resolve_attr):
            return None
        return indicator


PREVIEW_INDICATORS = ConfigPreviewIndicatorAuthority(PREVIEW_LABELS)


def format_generic_config(config_attr: str, config: Any, resolve_attr: Optional[Callable] = None) -> Optional[str]:
    """Format any config with an indicator for preview display.

    GENERAL RULE: Any config with an 'enabled: bool' parameter will only show
    if it resolves to True.

    Args:
        config_attr: Config attribute name (e.g., 'napari_streaming_config')
        config: Config object
        resolve_attr: Optional function to resolve lazy config attributes
                     Signature: resolve_attr(parent_obj, config_obj, attr_name, context) -> value

    Returns:
        Formatted indicator string (e.g., 'NAP') or None if config is disabled
    """
    return PREVIEW_INDICATORS.indicator_for(config, resolve_attr)


def format_well_filter_config(config_attr: str, config: Any, resolve_attr: Optional[Callable] = None) -> Optional[str]:
    """Format well filter config for preview display.

    GENERAL RULE: Any config with an 'enabled: bool' parameter will only show
    if it resolves to True. This applies to streaming configs (NAP/FIJI/MAT) which
    inherit from WellFilterConfig but also have an 'enabled' field.

    Args:
        config_attr: Config attribute name (e.g., 'step_well_filter_config')
        config: Config object (must be WellFilterConfig)
        resolve_attr: Optional function to resolve lazy config attributes

    Returns:
        Formatted indicator string (e.g., 'NAP+5', 'FIJI', 'FILT+5') or None if disabled
    """
    from openhcs.core.config import WellFilterConfig, WellFilterMode

    if not isinstance(config, WellFilterConfig):
        return None

    # CRITICAL: Check enabled field first (for streaming configs that inherit from WellFilterConfig)
    # This ensures NAP/FIJI/MAT only show if enabled=True
    is_enabled = _check_enabled_field(config, resolve_attr)
    if not is_enabled:
        return None

    # Resolve well_filter value
    if resolve_attr:
        well_filter = resolve_attr(None, config, 'well_filter', None)
        mode = resolve_attr(None, config, 'well_filter_mode', None)
    else:
        well_filter = config.well_filter
        mode = config.well_filter_mode

    # Get indicator from registry (NAP, FIJI, MAT) or default to FILT
    indicator = PREVIEW_LABELS.label_for(config)
    if indicator is None:
        indicator = 'FILT'
    has_registered_label = indicator != 'FILT'

    # If well_filter is None, show just the indicator (for configs with registered labels)
    # or return None (for generic well filter configs)
    if well_filter is None:
        # Configs with registered labels (NAP/FIJI/MAT) show indicator even without well_filter
        if has_registered_label:
            return indicator
        # Generic well filter configs require well_filter to be set
        return None

    # Format well_filter for display
    if isinstance(well_filter, list):
        wf_display = str(len(well_filter))
    elif isinstance(well_filter, int):
        wf_display = str(well_filter)
    else:
        wf_display = str(well_filter)

    # Add +/- prefix for INCLUDE/EXCLUDE mode
    mode_prefix = '-' if mode == WellFilterMode.EXCLUDE else '+'

    return f"{indicator}{mode_prefix}{wf_display}"


def format_config_indicator(config_attr: str, config: Any, resolve_attr: Optional[Callable] = None) -> Optional[str]:
    """Format any config for preview display (dispatcher function).

    GENERAL RULE: Any config with an 'enabled: bool' parameter will only show
    if it resolves to True.

    Args:
        config_attr: Config attribute name
        config: Config object
        resolve_attr: Optional function to resolve lazy config attributes

    Returns:
        Formatted indicator string or None if config should not be shown
    """
    from openhcs.core.config import WellFilterConfig

    # Dispatch to specific formatter based on config type
    if isinstance(config, WellFilterConfig):
        return format_well_filter_config(config_attr, config, resolve_attr)
    else:
        # All other configs use generic formatter (checks enabled field automatically)
        return format_generic_config(config_attr, config, resolve_attr)


def get_field_abbreviation(field_name: str, config_type: type = None) -> str:
    """Look up field abbreviation from registry.

    Searches FIELD_ABBREVIATIONS_REGISTRY for the field name, checking:
    1. The specific config_type (if provided)
    2. All registered config types (as fallback for shared field names)

    Args:
        field_name: The field name to abbreviate (e.g., 'well_filter')
        config_type: Optional config class to check first

    Returns:
        Abbreviated name if found, otherwise the original field_name
    """
    # Check specific config type first
    if config_type is not None:
        # Direct lookup
        if config_type in FIELD_ABBREVIATIONS_REGISTRY:
            abbrevs = FIELD_ABBREVIATIONS_REGISTRY[config_type]
            if field_name in abbrevs:
                return abbrevs[field_name]
        # Check base classes (for lazy wrapper types)
        for base in config_type.__mro__[1:]:
            if base in FIELD_ABBREVIATIONS_REGISTRY:
                abbrevs = FIELD_ABBREVIATIONS_REGISTRY[base]
                if field_name in abbrevs:
                    return abbrevs[field_name]

    # Fallback: search all registered configs for this field name
    for abbrevs in FIELD_ABBREVIATIONS_REGISTRY.values():
        if field_name in abbrevs:
            return abbrevs[field_name]

    return field_name
