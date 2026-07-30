"""Nominal edit-session state for ConfigWindow."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Generic, TypeVar, cast

from objectstate import (
    ObjectState,
    ObjectStateEditSession,
    get_live_global_config,
    get_saved_global_config,
    is_global_config_type,
    set_live_global_config,
    set_saved_global_config,
)
from objectstate.lazy_factory import LazyDataclass

from openhcs.serialization.pycodify_formatters import (
    LazyDataclassFieldEmissionState,
)

logger = logging.getLogger(__name__)


ConfigT = TypeVar("ConfigT")


@dataclass(frozen=True, slots=True)
class GlobalConfigContextCheckpoint(Generic[ConfigT]):
    """Exact external and edit-session context at one transaction boundary."""

    saved: ConfigT | None
    live: ConfigT | None
    original_live: ConfigT | None
    dirty: bool


@dataclass(slots=True)
class ConfigEditSession(Generic[ConfigT]):
    """Own non-visual config edit state for ConfigWindow."""

    state: ObjectState
    original_config: ConfigT
    global_context_dirty: bool = False
    saving: bool = False
    _original_live_global_config: ConfigT | None = field(init=False, repr=False)
    _object_session: ObjectStateEditSession[ConfigT] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._original_live_global_config = None
        if self.is_global_config:
            self._original_live_global_config = get_live_global_config(
                self.config_class
            )
        self._object_session = ObjectStateEditSession(
            state_provider=lambda: self.state,
            fallback_object=self.original_config,
            expected_type=self.config_class,
        )

    @property
    def config_class(self) -> type[ConfigT]:
        return type(self.original_config)

    @property
    def is_global_config(self) -> bool:
        return is_global_config_type(self.config_class)

    def to_object(self) -> ConfigT:
        """Reconstruct the current config object from ObjectState storage."""
        return self._object_session.to_object()

    def to_code_document_object(self) -> ConfigT:
        """Reconstruct config while retaining only authored lazy field paths.

        Flattened ``ObjectState`` storage reconstructs every nested lazy
        dataclass so raw inheritance markers remain intact.  Clean code mode
        must not mistake those reconstructed containers for user-authored
        overrides, so project the object through ObjectState's authoritative
        signature-diff paths before serialization.
        """

        config = self.to_object()
        if not isinstance(config, LazyDataclass):
            return config
        return cast(
            ConfigT,
            LazyDataclassFieldEmissionState.retain_only_authored_paths(
                config,
                self.state.signature_diff_fields,
            ),
        )

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

    def apply_code_edit_context(self, new_config: ConfigT) -> None:
        """Publish an unsaved code edit to the live UI context only."""
        if not self.is_global_config:
            return
        set_live_global_config(self.config_class, new_config)
        self.global_context_dirty = True
        logger.debug(
            "Updated LIVE thread-local %s context",
            self.config_class.__name__,
        )

    def capture_global_context(self) -> GlobalConfigContextCheckpoint[ConfigT] | None:
        """Capture saved and live stores independently before a fallible save."""

        if not self.is_global_config:
            return None
        return GlobalConfigContextCheckpoint(
            saved=get_saved_global_config(self.config_class),
            live=get_live_global_config(self.config_class),
            original_live=self._original_live_global_config,
            dirty=self.global_context_dirty,
        )

    def restore_global_context(
        self,
        checkpoint: GlobalConfigContextCheckpoint[ConfigT] | None,
    ) -> None:
        """Restore the exact saved/live split captured before a save."""

        if not self.is_global_config or checkpoint is None:
            return
        set_saved_global_config(self.config_class, checkpoint.saved)
        set_live_global_config(self.config_class, checkpoint.live)
        self._original_live_global_config = checkpoint.original_live
        self.global_context_dirty = checkpoint.dirty

    def publish_saved_global_config(self, new_config: ConfigT) -> None:
        """Publish a saved global config to the saved/live thread-local stores."""
        if not self.is_global_config:
            return
        set_saved_global_config(self.config_class, new_config)
        set_live_global_config(self.config_class, new_config)
        logger.debug(
            "Updated SAVED and LIVE thread-local %s on SAVE",
            self.config_class.__name__,
        )
        self._original_live_global_config = new_config
        self.global_context_dirty = False

    def restore_global_context_if_dirty(self) -> bool:
        """Restore only the live context that preceded this edit session."""
        if (
            not self.is_global_config
            or not self.global_context_dirty
        ):
            return False
        set_live_global_config(
            self.config_class,
            self._original_live_global_config,
        )
        self.global_context_dirty = False
        return True
