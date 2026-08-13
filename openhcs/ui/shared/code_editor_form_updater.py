"""Shared utility for updating forms from code editor changes.

This module provides utilities for parsing edited code and updating form
managers with only the explicitly set fields, preserving ``None`` values for
unspecified fields.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from typing import Any, get_args, get_origin

from objectstate import semantic_values_equal


logger = logging.getLogger(__name__)


class CodeEditorFormUpdater:
    """Utility for updating forms from code editor changes.

    The code editor always produces a *concrete* instance (dataclass or
    regular object). For lazy configs, we rely on the pattern that raw
    attribute access (``object.__getattribute__``) returns ``None`` for
    inherited / unset fields. The form layer then interprets ``None`` as
    "show placeholder", while concrete values override placeholders.
    """

    # REMOVED: extract_explicitly_set_fields() method
    # Raw None = inherited, raw concrete = user-set (same pattern as
    # the pycodify serializer and the rest of the config IO stack).

    @staticmethod
    def update_form_from_instance(
        form_manager,
        new_instance: Any,
        broadcast_callback=None,
    ) -> None:
        """Update a form manager with values from a new instance.

        Args:
            form_manager: ParameterFormManager instance.
            new_instance: New object/dataclass instance with updated values.
            broadcast_callback: Optional callback to broadcast the new
                instance (e.g., to an event bus).
        """

        is_instance_dataclass = is_dataclass(new_instance)
        if is_instance_dataclass:
            instance_fields = [field.name for field in fields(new_instance)]
        else:
            # Fallback: drive off whatever the form currently knows about.
            instance_fields = list(form_manager.parameters.keys())

        for field_name in instance_fields:
            if field_name not in form_manager.parameters:
                logger.debug(
                    "CodeEditorFormUpdater: field %s missing from form_manager %s",
                    field_name,
                    getattr(form_manager, "field_id", "<unknown>"),
                )
                continue

            if is_instance_dataclass:
                new_value = CodeEditorFormUpdater._get_raw_field_value(
                    new_instance, field_name
                )
            else:
                new_value = getattr(new_instance, field_name, None)

            if is_dataclass(new_value) and not isinstance(new_value, type):
                CodeEditorFormUpdater._update_nested_dataclass(
                    form_manager, field_name, new_value
                )
            else:
                logger.debug(
                    "CodeEditorFormUpdater: updating %s to %r", field_name, new_value
                )
                CodeEditorFormUpdater._update_parameter_if_changed(
                    form_manager,
                    field_name,
                    new_value,
                )

	        # NOTE:
	        # Cross-window propagation and placeholder refresh are handled by
	        # ParameterFormManager.update_parameter together with the
	        # FieldChangeDispatcher / LiveContextService stack. Callers usually
	        # do NOT need to trigger a global refresh after using this helper;
	        # only special flows (e.g., global-config snapshot restore/cancel)
	        # should call ParameterFormManager.trigger_global_cross_window_refresh().

        if broadcast_callback:
            broadcast_callback(new_instance)

    # ------------------------------------------------------------------
    # Nested dataclass handling
    # ------------------------------------------------------------------

    @staticmethod
    def _update_nested_dataclass(
        form_manager,
        field_name: str,
        new_value: Any,
    ) -> None:
        """Recursively update a nested dataclass field and all its children.

        ``new_value`` is a concrete dataclass instance coming from the code
        editor. We walk its fields and drive the *nested* manager entirely via
        ``update_parameter`` so that FieldChangeDispatcher and
        ValueCollectionService can do the right thing (parent reconstruction,
        _user_set_fields tracking, cross-window propagation, etc.).
        """

        nested_manager = form_manager.nested_managers.get(field_name)
        if not nested_manager:
            # No dedicated nested manager – treat as a regular field on the
            # parent. This is still routed through FieldChangeDispatcher.
            CodeEditorFormUpdater._update_parameter_if_changed(
                form_manager,
                field_name,
                new_value,
            )
            return

        for field in fields(new_value):
            nested_field_value = CodeEditorFormUpdater._get_raw_field_value(
                new_value, field.name
            )
            if field.name not in nested_manager.parameters:
                continue

            if is_dataclass(nested_field_value) and not isinstance(
                nested_field_value, type
            ):
                CodeEditorFormUpdater._update_nested_dataclass_in_manager(
                    nested_manager, field.name, nested_field_value
                )
            else:
                CodeEditorFormUpdater._update_parameter_if_changed(
                    nested_manager,
                    field.name,
                    nested_field_value,
                )

    @staticmethod
    def _update_nested_dataclass_in_manager(
        manager,
        field_name: str,
        new_value: Any,
    ) -> None:
        """Helper to update nested dataclass within a specific manager."""

        nested_manager = manager.nested_managers.get(field_name)
        if not nested_manager:
            CodeEditorFormUpdater._update_parameter_if_changed(
                manager,
                field_name,
                new_value,
            )
            return

        for field in fields(new_value):
            nested_field_value = CodeEditorFormUpdater._get_raw_field_value(
                new_value, field.name
            )
            if field.name not in nested_manager.parameters:
                continue

            if is_dataclass(nested_field_value) and not isinstance(
                nested_field_value, type
            ):
                CodeEditorFormUpdater._update_nested_dataclass_in_manager(
                    nested_manager, field.name, nested_field_value
                )
            else:
                CodeEditorFormUpdater._update_parameter_if_changed(
                    nested_manager,
                    field.name,
                    nested_field_value,
                )

    @staticmethod
    def _update_parameter_if_changed(
        form_manager,
        field_name: str,
        new_value: Any,
    ) -> bool:
        """Dispatch one code-mode field only when its raw value changed.

        ``ObjectState`` already rejects identical assignments before cache
        invalidation, but ``ParameterFormManager.update_parameter`` performs
        widget synchronization and emits root/cross-window signals before the
        caller can benefit from that no-op.  Code documents reconstruct a whole
        object, so filtering against the manager's raw, path-scoped parameters
        avoids turning a one-field edit into one dispatch per dataclass field.

        Returns:
            ``True`` when the field was sent through the normal form dispatcher.
        """

        current_parameters = form_manager.parameters
        if field_name not in current_parameters:
            return False
        if semantic_values_equal(
            current_parameters[field_name],
            new_value,
        ):
            logger.debug(
                "CodeEditorFormUpdater: skipping unchanged field %s",
                field_name,
            )
            return False

        form_manager.update_parameter(field_name, new_value)
        return True

    # ------------------------------------------------------------------
    # Lazy-constructor patching and helpers
    # ------------------------------------------------------------------

    @staticmethod
    @contextmanager
    def patch_lazy_constructors():
        """Patch lazy dataclass constructors during ``exec``.

        Ensures exec()-created instances only set explicitly provided kwargs,
        allowing unspecified fields to remain ``None``.
        """

        from objectstate import patch_lazy_constructors

        with patch_lazy_constructors():
            yield

    @staticmethod
    def _get_raw_field_value(obj: Any, field_name: str):
        """Fetch field without triggering lazy ``__getattr__`` logic."""

        try:
            return object.__getattribute__(obj, field_name)
        except AttributeError:
            return getattr(obj, field_name, None)

    @staticmethod
    def _is_dataclass(field_type: Any) -> bool:
        """Check if a field type represents (or wraps) a dataclass."""

        origin = get_origin(field_type)
        if origin is not None:
            return any(
                CodeEditorFormUpdater._is_dataclass(arg)
                for arg in get_args(field_type)
                if arg is not type(None)
            )
        try:
            return is_dataclass(field_type)
        except TypeError:
            return False

    @staticmethod
    def _get_dataclass_field_value(instance: Any, field_obj) -> Any:
        """Get a field value from a dataclass, preserving raw ``None``.

        For nested dataclasses we want to avoid triggering any lazy resolution
        logic, so we go through ``_get_raw_field_value``. For primitives and
        non-dataclass fields we fall back to normal ``getattr``.
        """

        if CodeEditorFormUpdater._is_dataclass(field_obj.type):
            return CodeEditorFormUpdater._get_raw_field_value(
                instance, field_obj.name
            )
        return getattr(instance, field_obj.name, None)
