"""OpenHCS config preview projections derived from config declarations."""

from typing import TYPE_CHECKING

from pyqt_reactive.utils.preview_formatters import (
    check_enabled_field,
    resolve_preview_label,
)

if TYPE_CHECKING:
    from openhcs.core.config import WellFilterConfig


def format_well_filter_config(
    config: "WellFilterConfig",
    _config_attr: str,
) -> str | None:
    """Project one well-filter declaration into its compact preview text."""
    if not check_enabled_field(config):
        return None

    label = resolve_preview_label(config)
    if label is None:
        raise TypeError(f"{type(config).__name__} has no preview-label declaration")

    well_filter = config.well_filter
    if well_filter is None:
        return label.label if config.show_preview_without_well_filter else None

    if isinstance(well_filter, list):
        well_display = str(len(well_filter))
    else:
        well_display = str(well_filter)

    return f"{label.label}{config.well_filter_mode.preview_prefix}{well_display}"
