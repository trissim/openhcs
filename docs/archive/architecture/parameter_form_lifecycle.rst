Parameter Form Lifecycle Management
===================================

**Complete lifecycle of parameter forms from creation to context synchronization.**

*Status: STABLE (describes main branch implementation)*
*Module: openhcs.pyqt_gui.widgets.shared.parameter_form_manager*

.. note::
   This document describes the **main branch** monolithic implementation. For the refactored service-oriented architecture currently in development, see :doc:`parameter_form_service_architecture`.

Overview
--------
Parameter forms must maintain consistency between widget state, internal parameters, and thread-local context. Traditional forms lose this synchronization during operations like reset, causing placeholder bugs. The lifecycle management system ensures all three states remain synchronized.

:py:meth:`~openhcs.pyqt_gui.widgets.shared.parameter_form_manager.ParameterFormManager.reset_parameter` orchestrates the complete reset lifecycle. It first determines the reset value (None for lazy configs), updates the internal parameter dictionary, updates the thread-local context to match, then updates the widget display and applies placeholder logic. This four-step process ensures that form state, context state, and widget display all stay in sync.

This prevents the reset placeholder bug where forms show stale values instead of current defaults.

Widget State Management
-----------------------
The form manager coordinates widget updates with context behavior application.

Value Update Coordination
~~~~~~~~~~~~~~~~~~~~~~~~~
:py:meth:`~openhcs.pyqt_gui.widgets.shared.parameter_form_manager.ParameterFormManager.update_widget_value` acts as the central coordinator for widget updates. It first blocks signals to prevent infinite loops, updates the widget's displayed value using type-specific dispatch, then applies context behavior (like placeholder text for None values). This ensures widgets show the right value without triggering cascading updates.

Context Behavior Application
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:py:meth:`~openhcs.pyqt_gui.widgets.shared.parameter_form_manager.ParameterFormManager._apply_context_behavior` decides whether to show placeholder text. If the value is None and we're in a lazy dataclass context, it calls the placeholder resolution system. If the value is not None, it clears any existing placeholder state. This creates the dynamic "Pipeline default: X" behavior.

Rendering Optimizations
~~~~~~~~~~~~~~~~~~~~~~~
Several recent improvements keep typing responsive even with large forms:

* :py:meth:`~openhcs.pyqt_gui.widgets.shared.parameter_form_manager.ParameterFormManager._store_parameter_value` mirrors each edit into an in-memory cache so :py:meth:`get_current_values` no longer rereads every widget.
* ``_placeholder_candidates`` tracks only the parameters that currently resolve to ``None``. Placeholder refreshes iterate over this set instead of the entire form.
* Each widget stores a ``placeholder_signature`` (see :mod:`openhcs.pyqt_gui.widgets.shared.widget_strategies`) so placeholders that have not changed are skipped entirely, avoiding redundant repaints.

Parameter Change Propagation
----------------------------
Changes flow from widgets through internal state to context synchronization.

:py:meth:`~openhcs.pyqt_gui.widgets.shared.parameter_form_manager.ParameterFormManager._emit_parameter_change` handles the flow when users change widget values. It converts the raw widget value to the correct type, updates the internal parameter dictionary, then emits a signal so other components can react. This is the normal path for user edits (as opposed to programmatic updates like reset).

Thread-Local Context Synchronization
------------------------------------
Critical synchronization patterns ensure context reflects current form state.

Reset Context Update Pattern
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:py:meth:`~openhcs.pyqt_gui.widgets.shared.parameter_form_manager.ParameterFormManager.reset_parameter` updates thread-local context during reset using :py:func:`~dataclasses.replace` to prevent placeholder bugs.

UI Component Lifecycle Patterns
-------------------------------
Different UI components have different form lifecycle requirements.

Step Editor Lifecycle
~~~~~~~~~~~~~~~~~~~~~
Step editors show step configurations with isolated context. Forms are created with custom context providers that resolve against their parent pipeline configuration. Reset operations update the step-specific context without affecting other UI components.

Pipeline Editor Lifecycle
~~~~~~~~~~~~~~~~~~~~~~~~~
Pipeline editors (plate manager style) handle pipeline-level configuration editing. Forms use standard thread-local context resolution and coordinate with the plate manager's save/load operations.

Pipeline Config Lifecycle
~~~~~~~~~~~~~~~~~~~~~~~~~
Pipeline config editing (accessed from plate manager) creates forms that resolve against the current pipeline's thread-local context. Save operations update both the pipeline configuration and the thread-local context to maintain consistency.

Global Config Lifecycle
~~~~~~~~~~~~~~~~~~~~~~~
Global config editing (accessed from main window) creates forms that show static defaults. Reset operations restore base class default values since there's no higher-level context to resolve against.

Cross-Window Placeholder Updates
---------------------------------
When multiple configuration dialogs are open simultaneously, they share live values for placeholder resolution. This enables real-time preview of configuration changes across windows.

Live Context Collection
~~~~~~~~~~~~~~~~~~~~~~~~
:py:meth:`~openhcs.pyqt_gui.widgets.shared.parameter_form_manager.ParameterFormManager._collect_live_context_from_other_windows` gathers current user-modified values from all active form managers. When a user types in one window, other windows immediately see the updated value in their placeholders. This creates a live preview system where configuration changes are visible before saving.

Live Context Snapshots
~~~~~~~~~~~~~~~~~~~~~~
:py:class:`~openhcs.pyqt_gui.widgets.shared.parameter_form_manager.LiveContextSnapshot` wraps the collected values together with a monotonically increasing ``token``. As long as the token remains unchanged, :py:meth:`_build_context_stack` reuses cached GlobalPipelineConfig and PipelineConfig overlays instead of rebuilding them on every keystroke.

Async Placeholder Resolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Once the initial load completes, :py:meth:`~openhcs.pyqt_gui.widgets.shared.parameter_form_manager.ParameterFormManager._schedule_async_placeholder_refresh` offloads placeholder work to :class:`~openhcs.pyqt_gui.widgets.shared.parameter_form_manager._PlaceholderRefreshTask`. The worker receives the parameter snapshot, the ``_placeholder_candidates`` list, and the current :class:`LiveContextSnapshot`, resolves placeholders off the UI thread, then emits the results back to the main thread for application. This keeps the UI responsive even when dozens of placeholders participate.

Active Manager Registry
~~~~~~~~~~~~~~~~~~~~~~~~
:py:attr:`~openhcs.pyqt_gui.widgets.shared.parameter_form_manager.ParameterFormManager._active_form_managers` maintains a class-level list of all active form manager instances. When a form manager is created, it registers itself in this list. When a dialog closes, it must unregister to prevent ghost references that cause infinite refresh loops.

Signal-Based Synchronization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Form managers emit :py:attr:`~openhcs.pyqt_gui.widgets.shared.parameter_form_manager.ParameterFormManager.context_value_changed` and :py:attr:`~openhcs.pyqt_gui.widgets.shared.parameter_form_manager.ParameterFormManager.context_refreshed` signals when values change. Other active managers listen to these signals and refresh their placeholders accordingly. This creates a reactive system where all windows stay synchronized.

Cross-Window Debounce
~~~~~~~~~~~~~~~~~~~~~
:py:meth:`~openhcs.pyqt_gui.widgets.shared.parameter_form_manager.ParameterFormManager._schedule_cross_window_refresh` debounces placeholder refreshes with :pyattr:`~openhcs.pyqt_gui.widgets.shared.parameter_form_manager.ParameterFormManager.CROSS_WINDOW_REFRESH_DELAY_MS` (currently 60 ms). Multiple rapid edits are coalesced into a single refresh burst without sacrificing the “live preview” feel.

External Listener Pattern
~~~~~~~~~~~~~~~~~~~~~~~~~~
Some UI components need to react to configuration changes but are not themselves form managers (e.g., the pipeline editor's preview labels showing "MAT", "NAP", "FIJI" indicators). The external listener pattern allows these components to register for cross-window notifications without participating in the full form manager lifecycle.

**Registration**

External listeners register via the class method :py:meth:`~openhcs.pyqt_gui.widgets.shared.parameter_form_manager.ParameterFormManager.register_external_listener`:

.. code-block:: python

    # pipeline_editor.py
    ParameterFormManager.register_external_listener(
        self,
        self._on_cross_window_context_changed,
        self._on_cross_window_context_refreshed
    )

The registration stores a tuple ``(listener, value_changed_handler, refresh_handler)`` in the class-level ``_external_listeners`` list. Unlike form managers, external listeners:

- Do not participate in the mesh signal topology (no bidirectional connections)
- Do not emit signals themselves
- Receive notifications via direct method calls, not Qt signals
- Are not scoped (they receive all notifications regardless of ``scope_id``)

**Notification Points**

External listeners are notified at three critical points:

1. **Global refresh** (:py:meth:`~openhcs.pyqt_gui.widgets.shared.parameter_form_manager.ParameterFormManager.trigger_global_cross_window_refresh`): Called when global config changes or when all managers are unregistered (e.g., after cancel). Notifies all external listeners with ``refresh_handler(None, None)``.

2. **Reset operations** (:py:meth:`~openhcs.pyqt_gui.widgets.shared.parameter_form_manager.ParameterFormManager.reset_all_parameters`): Called when a form is reset (either root or nested manager). Notifies external listeners with ``refresh_handler(object_instance, context_obj)`` via :py:meth:`~openhcs.pyqt_gui.widgets.shared.parameter_form_manager.ParameterFormManager._notify_external_listeners_refreshed`.

3. **Dialog cancel** (``reject()`` in :py:class:`~openhcs.pyqt_gui.windows.base_form_dialog.BaseFormDialog` subclasses): Called when a dialog is cancelled (Cancel button or Escape key). Must call :py:meth:`~openhcs.pyqt_gui.widgets.shared.parameter_form_manager.ParameterFormManager.trigger_global_cross_window_refresh` AFTER unregistration to ensure external listeners receive the refresh notification.

**Critical Implementation Requirements**

For external listeners to work correctly:

1. Cancel operations (``reject()``) must call :py:meth:`~openhcs.pyqt_gui.widgets.shared.parameter_form_manager.ParameterFormManager.trigger_global_cross_window_refresh` AFTER ``super().reject()`` to ensure external listeners are notified after the window is unregistered.

2. Reset operations must call :py:meth:`~openhcs.pyqt_gui.widgets.shared.parameter_form_manager.ParameterFormManager._notify_external_listeners_refreshed` for both root and nested managers to ensure external listeners update immediately.

3. External listeners must implement both ``value_changed_handler(field_path, new_value, editing_object, context_object)`` and ``refresh_handler(editing_object, context_object)`` to handle both incremental changes and full refreshes.

**Example: Pipeline Editor Preview Labels**

The pipeline editor uses external listeners to update preview labels (MAT, NAP, FIJI) when configuration changes occur in other windows:

.. code-block:: python

    # openhcs/pyqt_gui/widgets/pipeline_editor.py
    def setup_connections(self):
        """Setup signal connections."""
        # Register as external listener for cross-window updates
        ParameterFormManager.register_external_listener(
            self,
            self._on_cross_window_context_changed,
            self._on_cross_window_context_refreshed
        )

    def _on_cross_window_context_refreshed(self, editing_object, context_object):
        """Handle cross-window context refresh (e.g., after cancel or reset)."""
        logger.info(f"🔄 Pipeline editor: Context refreshed, refreshing preview labels")
        self._update_step_list()  # Refresh all preview labels

    def closeEvent(self, event):
        """Handle widget close event."""
        # Unregister external listener
        ParameterFormManager.unregister_external_listener(self)
        super().closeEvent(event)

This pattern ensures that preview labels stay synchronized with configuration changes across all open windows, including when dialogs are cancelled or reset.

Dialog Lifecycle Management
----------------------------
Proper dialog cleanup is critical to prevent ghost form managers that cause infinite refresh loops and runaway CPU usage.

The Ghost Manager Problem
~~~~~~~~~~~~~~~~~~~~~~~~~~
When a dialog closes without unregistering its form manager, it remains in the ``_active_form_managers`` registry as a "ghost". When a new dialog opens and the user types, the system collects context from the ghost manager, which triggers a refresh in the ghost, which collects context from the new manager, creating an infinite ping-pong loop.

Qt Dialog Lifecycle Quirk
~~~~~~~~~~~~~~~~~~~~~~~~~~
Qt's ``QDialog.accept()`` and ``QDialog.reject()`` methods do NOT trigger ``closeEvent()`` - they just hide the dialog and emit signals. This means cleanup code in ``closeEvent()`` is never called when users click Save or Cancel buttons. Dialogs must explicitly unregister in ``accept()``, ``reject()``, and ``closeEvent()`` to ensure cleanup happens regardless of how the dialog closes.

BaseFormDialog Pattern
~~~~~~~~~~~~~~~~~~~~~~
:py:class:`~openhcs.pyqt_gui.windows.base_form_dialog.BaseFormDialog` solves the cleanup problem by providing a base class that automatically handles unregistration. It overrides ``accept()``, ``reject()``, and ``closeEvent()`` to call ``_unregister_all_form_managers()`` before closing. Subclasses implement ``_get_form_managers()`` to return their form manager instances, and the base class handles all cleanup automatically.

This pattern ensures that every dialog using ParameterFormManager properly cleans up, preventing ghost manager bugs without requiring developers to remember manual cleanup in multiple methods.

Form State Synchronization
--------------------------
The three-state synchronization pattern ensures consistency across all UI components.

Internal Parameter State
~~~~~~~~~~~~~~~~~~~~~~~~
:py:attr:`~openhcs.pyqt_gui.widgets.shared.parameter_form_manager.ParameterFormManager.parameters` stores the form's internal parameter dictionary. This represents the current user edits and serves as the source of truth for widget display.

Thread-Local Context State
~~~~~~~~~~~~~~~~~~~~~~~~~~
:py:class:`~openhcs.core.context.global_config._global_config_contexts` maintains thread-local context that affects placeholder resolution. This must be kept synchronized with form state during operations like reset.

Widget Display State
~~~~~~~~~~~~~~~~~~~~
Widget values and placeholder text reflect the combination of internal parameters and context resolution. The form manager ensures widgets always display the correct state based on current parameters and context.

Example: BaseFormDialog Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from openhcs.pyqt_gui.windows.base_form_dialog import BaseFormDialog
   from openhcs.pyqt_gui.widgets.shared.parameter_form_manager import ParameterFormManager

   class MyConfigDialog(BaseFormDialog):
       """Configuration dialog with automatic cleanup."""

       def __init__(self, config, parent=None):
           super().__init__(parent)

           # Create form manager
           self.form_manager = ParameterFormManager(
               field_id="my_config",
               dataclass_type=type(config),
               initial_values=config
           )

       def _get_form_managers(self):
           """Return form managers to unregister (required by BaseFormDialog)."""
           return [self.form_manager]

       # No need to override accept(), reject(), or closeEvent()
       # BaseFormDialog handles all cleanup automatically!

This pattern ensures proper cleanup regardless of how the dialog closes (Save button → ``accept()``, Cancel button → ``reject()``, X button → ``closeEvent()``).

See Also
--------
- :doc:`parameter_form_service_architecture` - Refactored service-oriented architecture (in development)
- :doc:`context_system` - Thread-local context management patterns
- :doc:`service-layer-architecture` - Service layer integration with forms
- :doc:`code_ui_interconversion` - Code/UI interconversion patterns
- :py:class:`~openhcs.pyqt_gui.windows.base_form_dialog.BaseFormDialog` - Base class for dialog cleanup
- :py:class:`~openhcs.pyqt_gui.windows.config_window.ConfigWindow` - Example BaseFormDialog implementation
- :py:class:`~openhcs.pyqt_gui.windows.dual_editor_window.DualEditorWindow` - Example BaseFormDialog implementation
