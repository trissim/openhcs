"""Shared OpenHCS manager-widget declaration mixins."""

from __future__ import annotations


class OpenHCSSingleRowActionManagerMixin:
    """Manager widgets whose button bar is rendered as one row."""

    BUTTON_GRID_COLUMNS = 0
    ACTION_REGISTRY = {}
