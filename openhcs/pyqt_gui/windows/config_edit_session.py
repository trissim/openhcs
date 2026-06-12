"""Nominal edit-session state for ConfigWindow."""

from __future__ import annotations

import logging
import copy
from dataclasses import dataclass, field
from typing import Any, Type

from openhcs.config_framework import is_global_config_type
from openhcs.config_framework.global_config import (
    set_global_config_for_editing,
    set_live_global_config,
    set_saved_global_config,
)
from openhcs.config_framework.object_state import ObjectState, ObjectStateRegistry

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ConfigEditSession:
    """Own non-visual config edit state for ConfigWindow."""

    config_class: Type
    state: ObjectState
    original_config: Any
    global_context_dirty: bool = False
    saving: bool = False
    _original_global_config_snapshot: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._original_global_config_snapshot = (
            copy.deepcopy(self.original_config)
            if self.is_global_config
            else None
        )

    @property
    def is_global_config(self) -> bool:
        return is_global_config_type(self.config_class)

    def to_object(self) -> Any:
        """Reconstruct the current config object from ObjectState storage."""
        return self.state.to_object()

    def begin_save_callback(self, window_id: int) -> None:
        self.saving = True
        logger.info("SAVE_CONFIG: Set saving=True before callback (id=%s)", window_id)

    def end_save_callback(self, window_id: int) -> None:
        self.saving = False
        logger.info("SAVE_CONFIG: Reset saving=False (id=%s)", window_id)

    def mark_global_field_changed(self) -> None:
        """Track unsaved global-context changes unless a save is in progress."""
        if self.saving:
            return
        self.global_context_dirty = True

    def apply_code_edit_context(self, new_config: Any) -> None:
        """Apply immediate thread-local context updates for edited code."""
        if not self.is_global_config:
            return
        set_global_config_for_editing(self.config_class, new_config)
        self.global_context_dirty = True
        logger.debug("Updated thread-local %s context", self.config_class.__name__)

    def publish_saved_global_config(self, new_config: Any) -> None:
        """Publish a saved global config to the saved/live thread-local stores."""
        if not self.is_global_config:
            return
        set_saved_global_config(self.config_class, new_config)
        set_live_global_config(self.config_class, new_config)
        logger.debug(
            "Updated SAVED and LIVE thread-local %s on SAVE",
            self.config_class.__name__,
        )
        ObjectStateRegistry.increment_token(notify=True)
        logger.debug("Invalidated descendant caches after global config save")
        self._original_global_config_snapshot = copy.deepcopy(new_config)
        self.global_context_dirty = False

    def restore_global_context_if_dirty(self) -> bool:
        """Restore the saved global edit context if this session dirtied it."""
        if (
            not self.is_global_config
            or not self.global_context_dirty
            or self._original_global_config_snapshot is None
        ):
            return False
        set_global_config_for_editing(
            self.config_class,
            copy.deepcopy(self._original_global_config_snapshot),
        )
        self.global_context_dirty = False
        return True
