====================================
Cross-Window Update Optimization
====================================

*Module: openhcs.pyqt_gui.widgets.shared.parameter_form_manager*
*Status: STABLE*

---

Overview
========

OpenHCS configuration windows update placeholders in real-time as users edit values in other windows. The cross-window update system uses type-based inheritance filtering and targeted field refresh to achieve <10ms update latency (down from ~200ms) while maintaining correct MRO-based inheritance semantics.

**Key Performance Optimizations:**

- **Type-based filtering**: Only refresh configs that inherit from the changed type via MRO
- **Targeted field refresh**: Only refresh the specific field that changed, not all fields
- **Field path extraction**: Parse full paths to extract relevant field names for nested managers
- **Widget signature checks**: Skip UI updates when placeholder text hasn't changed
- **None value propagation**: Pass None values through to override saved concrete values during reset
- **Trailing debounce**: 100ms debounce on placeholder refreshes to batch updates during rapid typing
- **Correct signal emission**: Nested managers and reset operations emit signals with proper object instances

Problem Context
===============

Without optimization, every keystroke in a configuration field triggered:

1. **Global token increment** → invalidates all caches
2. **Refresh ALL configs** → including completely unrelated ones (ZarrConfig, VFSConfig, etc.)
3. **Refresh ALL fields** → even fields that didn't change
4. **Redundant UI updates** → even when placeholder text unchanged

**Performance impact**: ~200ms per keystroke, with 5-28ms spent refreshing each unrelated config.

Solution Architecture
=====================

Type-Based Inheritance Filtering
---------------------------------

Only refresh configs that inherit from the changed config type using Python's MRO:

.. code-block:: python

    def _is_affected_by_context_change(self, editing_object, context_object):
        """Check if this form should refresh based on MRO inheritance."""

        # Check if this config type inherits from the changed config type
        if self.dataclass_type:
            editing_type = type(editing_object)
            try:
                if issubclass(self.dataclass_type, editing_type):
                    # This config inherits from changed type → refresh
                    return True
            except TypeError:
                pass

        return False

**Example**: When ``StepWellFilterConfig`` changes:

- ✅ Refresh ``StreamingDefaults`` (inherits from ``StepWellFilterConfig``)
- ✅ Refresh ``StepMaterializationConfig`` (inherits from ``StepWellFilterConfig``)
- ❌ Skip ``ZarrConfig`` (unrelated, doesn't inherit)
- ❌ Skip ``VFSConfig`` (unrelated, doesn't inherit)

Targeted Field Refresh
-----------------------

Only refresh placeholders for fields that inherit from the changed field's type:

.. code-block:: python

    def _refresh_specific_placeholder(self, field_name: str = None, live_context: dict = None):
        """Refresh placeholder for a specific field, or all fields if field_name is None."""

        if field_name is None:
            # No specific field - refresh all placeholders
            self._refresh_all_placeholders(live_context=live_context)
            return

        # Check if this exact field exists
        if field_name in self._placeholder_candidates:
            self._refresh_single_field_placeholder(field_name, live_context)
            return

        # Field doesn't exist with exact name - find fields that inherit from same base type
        # Example: PipelineConfig.well_filter_config → Step.step_well_filter_config
        fields_to_refresh = self._find_fields_inheriting_from_changed_field(field_name, live_context)

        # Refresh only the matching fields
        for matching_field in fields_to_refresh:
            self._refresh_single_field_placeholder(matching_field, live_context)

**Type matching logic**:

.. code-block:: python

    def _find_fields_inheriting_from_changed_field(self, changed_field_name, live_context):
        """Find fields that inherit from the same base type as the changed field."""

        # Get the type of the changed field from live context
        changed_field_type = self._get_field_type_from_live_context(changed_field_name, live_context)

        # Find fields in this form with matching types
        matching_fields = []
        for field in dataclass_fields(self.dataclass_type):
            field_type = field.type

            # Check if types match or share inheritance
            if field_type == changed_field_type:
                matching_fields.append(field.name)
            elif issubclass(field_type, changed_field_type):
                matching_fields.append(field.name)
            elif issubclass(changed_field_type, field_type):
                matching_fields.append(field.name)

        return matching_fields

**Example**: When ``PipelineConfig.well_filter_config`` changes:

- ✅ Refresh ``Step.step_well_filter_config`` (both inherit from ``WellFilterConfig``)
- ❌ Skip ``Step.dtype_config`` (unrelated type)
- ❌ Skip ``Step.processing_config`` (unrelated type)

Field Path Extraction
----------------------

Parse full field paths to extract relevant field names at each manager level:

.. code-block:: python

    def _do_cross_window_refresh(self, emit_signal=True, changed_field_path=None):
        """Perform cross-window refresh with field path extraction."""

        # Extract the relevant field name for this manager level
        # Example: "PipelineConfig.well_filter_config.well_filter"
        #   → extract "well_filter_config" for root manager
        changed_field_name = None
        if changed_field_path:
            path_parts = changed_field_path.split('.')
            if len(path_parts) > 1:
                # For root manager: use the first field name
                changed_field_name = path_parts[1]

        # Refresh this manager's specific field
        self._refresh_specific_placeholder(changed_field_name, live_context)

        # Extract remaining path for nested managers
        # "PipelineConfig.well_filter_config.well_filter" → "well_filter"
        nested_field_path = None
        if changed_field_path and changed_field_name:
            path_parts = changed_field_path.split('.')
            if len(path_parts) > 2:
                nested_field_path = '.'.join(path_parts[2:])

        # Pass remaining path to nested managers
        self._apply_to_nested_managers(
            lambda name, manager: manager._refresh_specific_placeholder_from_path(
                parent_field_name=changed_field_name,
                remaining_path=nested_field_path,
                live_context=live_context
            )
        )

**Path extraction example**:

- Full path: ``"PipelineConfig.well_filter_config.well_filter"``
- Root manager extracts: ``"well_filter_config"`` (first field after type name)
- Nested manager extracts: ``"well_filter"`` (remaining path)

Widget Signature Checks
------------------------

Widget strategies check if placeholder text changed before updating UI:

.. code-block:: python

    def _apply_lineedit_placeholder(widget, text):
        """Apply placeholder to line edit with signature check."""

        # Create signature from text
        signature = f"lineedit:{text}"

        # Check if signature changed
        if widget.property("placeholder_signature") == signature:
            return  # Skip update - placeholder text unchanged

        # Update widget
        widget.clear()
        widget.setPlaceholderText(text)
        widget.setProperty("placeholder_signature", signature)

**Performance impact**: Eliminates redundant ``setPlaceholderText()`` calls when resolved value hasn't changed.

Reset Propagation
=================

None Value Semantics
--------------------

In OpenHCS lazy configs, ``None`` has special meaning:

- **In saved configs**: ``None`` means "inherit from parent context via MRO"
- **In live context**: ``None`` means "field was reset, override saved value"

The system must distinguish between these two cases to enable proper reset propagation.

Reset Field Tracking
--------------------

When a field is reset, it's added to the ``reset_fields`` set:

.. code-block:: python

    def reset_parameter(self, param_name):
        """Reset parameter to inherit from parent context."""

        # Set value to None
        self.parameters[param_name] = None

        # Track that this field was explicitly reset
        self.reset_fields.add(param_name)

        # Increment token to invalidate caches
        type(self)._live_context_token_counter += 1

Live Context Inclusion
----------------------

Reset fields are included in live context even though their value is ``None``:

.. code-block:: python

    def get_user_modified_values(self):
        """Get only values that were explicitly set by the user.

        CRITICAL: Includes fields that were explicitly reset to None.
        This ensures cross-window updates see reset operations.
        """
        user_modified = {}
        current_values = self.get_current_values()

        for field_name, value in current_values.items():
            # Include None values if they were explicitly reset
            is_explicitly_reset = field_name in self.reset_fields

            if value is not None or is_explicitly_reset:
                user_modified[field_name] = value

        return user_modified

None Value Propagation
----------------------

When merging live context, ``None`` values are **passed through** to override saved concrete values:

.. code-block:: python

    def _merge_live_values(self, base_obj, live_values):
        """Merge live values into base object.

        CRITICAL: Passes None values through to dataclasses.replace(). When a field is reset
        to None in a form, the None value should override the saved concrete value in the
        base object. This allows the lazy resolution system to walk up the MRO to find the
        inherited value from parent context.
        """
        if live_values is None or not is_dataclass(base_obj):
            return base_obj

        # Reconstruct nested dataclasses recursively
        reconstructed_values = self.reconstruct_live_values(live_values)

        # Merge into base object (including None values to override saved concrete values)
        if reconstructed_values:
            return dataclass_replace(base_obj, **reconstructed_values)
        else:
            return base_obj

**Why pass None through?** When a field is reset to None in PipelineConfig, we need to override the saved concrete value in ``orchestrator.pipeline_config`` with None. This triggers MRO resolution which walks up to GlobalPipelineConfig to find the inherited value.

Trailing Debounce for Performance
==================================

To prevent expensive ``collect_live_context`` calls on every keystroke during rapid typing, the system uses **trailing debounce** with 100ms delay.

Debounce Constants
------------------

.. code-block:: python

    class ParameterFormManager:
        # Trailing debounce delays (ms) - timer restarts on each change
        PARAMETER_CHANGE_DEBOUNCE_MS = 100      # Same-window placeholder refreshes
        CROSS_WINDOW_REFRESH_DELAY_MS = 100     # Cross-window placeholder refreshes

    class CrossWindowPreviewMixin:
        PREVIEW_UPDATE_DEBOUNCE_MS = 100        # Pipeline editor preview label updates

Trailing Debounce Behavior
---------------------------

**Trailing debounce** means the timer **restarts** on each keystroke, only executing after typing stops:

- User types "abc" rapidly (3 keystrokes in 50ms)
- Timer starts at 0ms, restarts at 20ms, restarts at 50ms
- Timer fires at 150ms (50ms + 100ms delay)
- Only **one** refresh happens, not three

**Key properties**:

- ✅ Never blocks user input (timer runs in background)
- ✅ Always waits for user to finish typing
- ✅ Batches rapid changes into single update
- ✅ QTimer.start() on existing timer automatically restarts it

**Contrast with leading debounce** (NOT used):

- ❌ Executes immediately on first keystroke
- ❌ Blocks subsequent updates for fixed duration
- ❌ Can feel laggy if user types during block period

Performance Impact
------------------

**Before debounce**: Every keystroke triggered:

- 1x ``collect_live_context`` in typing window
- Nx ``collect_live_context`` in other open windows (N = number of windows)
- 1x ``collect_live_context`` in pipeline editor
- Token increment invalidates all caches

**After debounce**: Typing "hello" (5 keystrokes in 200ms):

- 0 refreshes during typing (timers keep restarting)
- 1 refresh at 300ms (200ms + 100ms delay)
- **5x reduction** in expensive operations

Implementation
--------------

**Same-window refresh** (``parameter_form_manager.py``):

.. code-block:: python

    def _on_parameter_changed_root(self, param_name: str, value: Any) -> None:
        """Debounce placeholder refreshes originating from this root manager."""
        if self._parameter_change_timer is None:
            self._run_debounced_placeholder_refresh()
        else:
            # Restart timer (trailing debounce)
            self._parameter_change_timer.start(self.PARAMETER_CHANGE_DEBOUNCE_MS)

**Cross-window refresh** (``parameter_form_manager.py``):

.. code-block:: python

    def _schedule_cross_window_refresh(self, emit_signal: bool = True, changed_field_path: str = None):
        """Schedule a debounced placeholder refresh for cross-window updates."""
        # Cancel existing timer if any (trailing debounce)
        if self._cross_window_refresh_timer is not None:
            self._cross_window_refresh_timer.stop()

        # Schedule new refresh after configured delay
        self._cross_window_refresh_timer = QTimer()
        self._cross_window_refresh_timer.setSingleShot(True)
        self._cross_window_refresh_timer.timeout.connect(
            lambda: self._do_cross_window_refresh(emit_signal=emit_signal, changed_field_path=changed_field_path)
        )
        self._cross_window_refresh_timer.start(self.CROSS_WINDOW_REFRESH_DELAY_MS)

**Pipeline editor preview** (``cross_window_preview_mixin.py``):

.. code-block:: python

    def _schedule_preview_update(self, full_refresh: bool = False) -> None:
        """Schedule a debounced preview update (trailing debounce)."""
        # Cancel existing timer if any
        if self._preview_update_timer is not None:
            self._preview_update_timer.stop()

        # Schedule new update after configured delay
        self._preview_update_timer = QTimer()
        self._preview_update_timer.setSingleShot(True)

        if full_refresh:
            self._preview_update_timer.timeout.connect(self._handle_full_preview_refresh)
        else:
            self._preview_update_timer.timeout.connect(self._process_pending_preview_updates)

        self._preview_update_timer.start(self.PREVIEW_UPDATE_DEBOUNCE_MS)

Signal Emission Correctness
============================

Nested Manager Signal Fix
--------------------------

**Problem**: Nested managers (e.g., ``LazyStepWellFilterConfig`` inside ``PipelineConfig``) were emitting signals with ``self.object_instance`` instead of ``root.object_instance``. This broke type-based filtering because other windows check if they inherit from ``PipelineConfig``, not ``LazyStepWellFilterConfig``.

**Solution**: Always emit signals from root manager with root's object instance:

.. code-block:: python

    def _on_parameter_changed_nested(self, param_name: str, value: Any) -> None:
        """Handle parameter changes in nested managers."""

        # Find root manager
        root = self._parent_manager
        while root._parent_manager is not None:
            root = root._parent_manager

        # Build full field path
        path_parts = [param_name]
        current = self
        while current._parent_manager is not None:
            parent_param_name = self._find_param_name_in_parent(current)
            if parent_param_name:
                path_parts.insert(0, parent_param_name)
            current = current._parent_manager

        path_parts.insert(0, root.field_id)
        field_path = '.'.join(path_parts)

        # CRITICAL: Use root.object_instance, not self.object_instance
        # This ensures type-based filtering works correctly
        root.context_value_changed.emit(field_path, value,
                                       root.object_instance, root.context_obj)

**Example**: Changing ``well_filter_config.enabled`` in PipelineConfig:

- ❌ **Before**: Signal emitted with ``editing_object=LazyStepWellFilterConfig()``
- ✅ **After**: Signal emitted with ``editing_object=PipelineConfig()``
- **Result**: Other windows correctly identify this as a PipelineConfig change

Reset Button Signal Emission
-----------------------------

**Problem**: Reset buttons set ``_in_reset`` flag to prevent infinite loops, but this flag also blocks normal ``parameter_changed`` handlers from emitting cross-window signals. Result: reset operations didn't trigger cross-window updates.

**Solution**: Manually emit ``context_value_changed`` signal during reset operations:

.. code-block:: python

    def reset_parameter(self, param_name: str) -> None:
        """Reset parameter to inherit from parent context."""

        # Set flag to block normal handlers
        self._in_reset = True

        try:
            # Reset value to None
            self.parameters[param_name] = None
            self.reset_fields.add(param_name)

            # Increment token to invalidate caches
            type(self)._live_context_token_counter += 1

            # CRITICAL: Manually emit cross-window signal
            # The _in_reset flag blocks normal handlers, so we must emit manually
            reset_value = self.parameters.get(param_name)
            if self._parent_manager is None:
                # Root manager: emit directly
                field_path = f"{self.field_id}.{param_name}"
                self.context_value_changed.emit(field_path, reset_value,
                                               self.object_instance, self.context_obj)
            else:
                # Nested manager: build full path and emit from root
                # (same logic as nested manager signal fix above)
                ...

            # Refresh placeholders
            self._refresh_with_live_context()
        finally:
            self._in_reset = False

**Same logic applies to** ``reset_all_parameters()``:

- Sets ``_block_cross_window_updates`` flag to prevent per-parameter signals
- Manually emits ``context_value_changed`` for each reset field after batch reset
- Ensures pipeline editor and other windows see all reset operations

Cross-Window Update Flow
=========================

Complete update flow when user changes a field:

1. **User edits field** in PipelineConfig window
2. **Debounce timer starts/restarts** (100ms trailing debounce)
3. **User stops typing** for 100ms
4. **Timer fires**: Form manager emits signal ``context_value_changed.emit(field_path, new_value, editing_object, context_object)``
5. **Other windows receive signal**: All active form managers check ``_is_affected_by_context_change()``
6. **Type-based filtering**: Only windows with configs inheriting from ``PipelineConfig`` proceed
7. **Debounce timer starts/restarts** in receiving windows (100ms trailing debounce)
8. **Timer fires**: Field path extraction to extract relevant field name for this manager level
9. **Type matching**: Find fields that inherit from the changed field's type
10. **Targeted refresh**: Refresh only matching fields, not all fields
11. **Widget signature check**: Skip UI update if placeholder text unchanged
12. **Nested propagation**: Pass remaining path to nested managers for recursive refresh

Performance Impact
==================

Measured improvements from optimization:

- **Cross-window refresh**: ~200ms → <10ms (20x faster)
- **Unaffected configs**: No longer refresh at all (was 5-28ms each)
- **Widget updates**: Skipped when placeholder text unchanged
- **Type-based filtering**: Only configs inheriting from changed type are refreshed
- **Trailing debounce**: Reduces refresh calls from every keystroke to once per typing burst

**Example scenario**: Changing ``well_filter`` in PipelineConfig

- **Before optimization**: Refreshed 8 configs (ZarrConfig, VFSConfig, DtypeConfig, ProcessingConfig, etc.) = ~200ms per keystroke
- **After type-based filtering**: Refreshed 1 config (StepWellFilterConfig) = <10ms per keystroke
- **After trailing debounce**: Typing "hello" (5 keystrokes) = 1 refresh at end instead of 5 during typing

**Combined impact**: Typing "hello" in well_filter field:

- **Before**: 5 keystrokes × 200ms = 1000ms of UI blocking
- **After**: 1 refresh × <10ms = <10ms total (100x improvement)

Implementation Notes
====================

**Source Code**:

- Type-based filtering: ``parameter_form_manager.py::_is_affected_by_context_change()``
- Targeted field refresh: ``parameter_form_manager.py::_refresh_specific_placeholder()``
- Field path extraction: ``parameter_form_manager.py::_do_cross_window_refresh()``
- Widget signature checks: ``widget_strategies.py::_apply_*_placeholder()``
- None value propagation: ``live_context_resolver.py::_merge_live_values()``
- Trailing debounce (same-window): ``parameter_form_manager.py::_on_parameter_changed_root()``
- Trailing debounce (cross-window): ``parameter_form_manager.py::_schedule_cross_window_refresh()``
- Trailing debounce (pipeline editor): ``cross_window_preview_mixin.py::_schedule_preview_update()``
- Nested manager signals: ``parameter_form_manager.py::_on_parameter_changed_nested()``
- Reset button signals: ``parameter_form_manager.py::reset_parameter()`` and ``reset_all_parameters()``

**Debounce Constants**:

- ``ParameterFormManager.PARAMETER_CHANGE_DEBOUNCE_MS = 100``
- ``ParameterFormManager.CROSS_WINDOW_REFRESH_DELAY_MS = 100``
- ``CrossWindowPreviewMixin.PREVIEW_UPDATE_DEBOUNCE_MS = 100``

**Related Documentation**:

- :doc:`configuration_framework` - Dual-axis resolution and MRO inheritance
- :doc:`../development/scope_hierarchy_live_context` - Scope isolation and live context
- :doc:`../development/placeholder_inheritance_debugging` - Debugging inheritance chains
