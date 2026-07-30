===============================
Placeholder Refresh Threading
===============================

*Module: openhcs.pyqt_gui.widgets.shared.parameter_form_manager*  
*Status: STABLE*

---

Overview
========

PyQt6 worker threads don't inherit thread-local storage from the main thread. The placeholder refresh system must explicitly capture and restore GlobalPipelineConfig context to ensure worker threads resolve placeholders with the correct configuration.

Problem Context
===============

Thread-local storage is process-specific and doesn't propagate to worker threads:

.. code-block:: python

    # Main thread
    set_global_config_for_editing(GlobalPipelineConfig, config)
    
    # Worker thread (different thread-local storage)
    config = get_current_global_config(GlobalPipelineConfig)  # Returns None!

Without explicit propagation, worker threads resolve placeholders against empty context, producing incorrect placeholder text.

Solution: Context Snapshot
===========================

The ``_PlaceholderRefreshTask`` captures GlobalPipelineConfig in the main thread and restores it in the worker thread:

.. code-block:: python

    class _PlaceholderRefreshTask:
        def __init__(self, generation, parameters_snapshot, placeholder_plan, live_context_snapshot):
            self._generation = generation
            self._parameters_snapshot = parameters_snapshot
            self._placeholder_plan = placeholder_plan
            self._live_context_snapshot = live_context_snapshot
            
            # CRITICAL: Capture thread-local GlobalPipelineConfig from main thread
            # Worker threads don't inherit thread-local storage
            from objectstate.context_manager import get_base_global_config
            self._global_config_snapshot = get_base_global_config()
        
        def run(self):
            """Execute in worker thread with restored context."""
            # Restore GlobalPipelineConfig in worker thread
            from objectstate.context_manager import set_base_global_config
            if self._global_config_snapshot is not None:
                set_base_global_config(self._global_config_snapshot)
            
            # Now placeholder resolution sees correct context
            resolved_placeholders = self._resolve_placeholders()
            return resolved_placeholders

This ensures worker threads resolve placeholders with the same GlobalPipelineConfig as the main thread.

Threading Architecture
======================

Main Thread Responsibilities
-----------------------------

1. **Capture context**: Snapshot GlobalPipelineConfig before creating task
2. **Create task**: Package context with placeholder resolution plan
3. **Submit to thread pool**: QThreadPool executes task asynchronously
4. **Handle results**: Update UI with resolved placeholder text

Worker Thread Responsibilities
-------------------------------

1. **Restore context**: Set thread-local GlobalPipelineConfig from snapshot
2. **Resolve placeholders**: Execute placeholder resolution with correct context
3. **Return results**: Send resolved text back to main thread via signals

Context Propagation Flow
=========================

.. code-block:: python

    # 1. Main thread: User edits GlobalPipelineConfig
    config_window.save()  # Updates thread-local storage
    
    # 2. Main thread: Trigger placeholder refresh
    form_manager._refresh_with_live_context()
    
    # 3. Main thread: Create task with context snapshot
    task = _PlaceholderRefreshTask(
        generation=self._generation,
        parameters_snapshot=params,
        placeholder_plan=plan,
        live_context_snapshot=live_context
    )
    # task._global_config_snapshot captured here
    
    # 4. Worker thread: Restore context
    task.run()  # Sets thread-local storage in worker
    
    # 5. Worker thread: Resolve placeholders
    # Placeholder resolution sees correct GlobalPipelineConfig
    
    # 6. Main thread: Update UI
    # Signal emitted, main thread updates placeholder text

Implementation Details
======================

Context Capture (Main Thread)
------------------------------

.. code-block:: python

    from objectstate.context_manager import get_base_global_config
    
    # Capture current GlobalPipelineConfig
    self._global_config_snapshot = get_base_global_config()

**Why get_base_global_config?**

Returns the actual config object, not a lazy wrapper. This ensures the snapshot is serializable and can be transferred to worker threads.

Context Restoration (Worker Thread)
------------------------------------

.. code-block:: python

    from objectstate.context_manager import set_base_global_config
    
    # Restore GlobalPipelineConfig in worker thread
    if self._global_config_snapshot is not None:
        set_base_global_config(self._global_config_snapshot)

**Why check for None?**

Initial application startup may not have GlobalPipelineConfig set. Gracefully handle this case.

Signal-Based Result Delivery
-----------------------------

.. code-block:: python

    class _PlaceholderRefreshSignals(QObject):
        """Signals for communicating from worker thread to main thread."""
        finished = pyqtSignal(int, dict)  # (generation, resolved_placeholders)
        error = pyqtSignal(int, str)      # (generation, error_message)

Worker threads cannot directly update UI. Signals marshal results back to main thread for UI updates.

Common Patterns
===============

Async Placeholder Refresh
--------------------------

.. code-block:: python

    def _refresh_with_live_context(self, live_context=None, exclude_param=None):
        """Trigger async placeholder refresh with context propagation."""
        
        # Collect live context from other open windows
        if live_context is None:
            live_context = self._collect_live_context_from_other_windows()
        
        # Create task with context snapshot
        task = _PlaceholderRefreshTask(
            generation=self._generation,
            parameters_snapshot=self._get_current_parameters(),
            placeholder_plan=self._build_placeholder_plan(),
            live_context_snapshot=live_context
        )
        
        # Submit to thread pool
        QThreadPool.globalInstance().start(task)

Synchronous Placeholder Refresh
--------------------------------

.. code-block:: python

    def _perform_placeholder_refresh_sync(self, live_context, exclude_param=None):
        """Synchronous refresh (no worker thread, no context propagation needed)."""
        
        # Already in main thread, thread-local storage accessible
        resolved_placeholders = self._resolve_placeholders_sync(live_context)
        
        # Update UI directly
        self._apply_resolved_placeholders(resolved_placeholders)

Use synchronous refresh during initialization when UI responsiveness is not critical.

Debugging Thread Issues
=======================

Symptom: Incorrect Placeholder Text
------------------------------------

**Cause**: Worker thread resolving against empty GlobalPipelineConfig

**Diagnosis**:

.. code-block:: python

    # In worker thread
    config = get_current_global_config(GlobalPipelineConfig)
    logger.debug(f"Worker thread config: {config}")  # Should not be None

**Fix**: Verify ``_global_config_snapshot`` is captured and restored

Symptom: UI Not Updating
-------------------------

**Cause**: Worker thread trying to update UI directly (not allowed)

**Diagnosis**: Check for direct widget updates in worker thread

**Fix**: Use signals to marshal results back to main thread

Implementation Notes
====================

**🔬 Source Code**: 

- Task definition: ``openhcs/pyqt_gui/widgets/shared/parameter_form_manager.py`` (line 173)
- Context capture: ``openhcs/pyqt_gui/widgets/shared/parameter_form_manager.py`` (line 176-180)
- Context restoration: ``openhcs/pyqt_gui/widgets/shared/parameter_form_manager.py`` (line 200-203)

**🏗️ Architecture**: 

- :doc:`../architecture/configuration-management-system` - Configuration hierarchy
- :doc:`parameter_form_manager_live_context` - Live context collection

**📊 Performance**: 

- Context snapshot is lightweight (single config object)
- Worker threads prevent UI blocking during placeholder resolution
- Signal overhead is negligible compared to placeholder resolution cost

Key Design Decisions
====================

**Why not use QThread directly?**

QThreadPool provides automatic thread management and reuse. Creating QThread instances for each refresh would be wasteful.

**Why snapshot instead of passing config as parameter?**

Thread-local storage is the canonical source of truth for GlobalPipelineConfig. Passing as parameter would create two sources of truth.

**Why restore in worker thread instead of using locks?**

Thread-local storage is thread-safe by design. Locks would add complexity and potential deadlocks.

Common Gotchas
==============

- **Don't access thread-local storage from worker threads**: Always capture in main thread and restore in worker
- **Don't update UI from worker threads**: Use signals to marshal results back to main thread
- **Don't assume GlobalPipelineConfig exists**: Check for None before restoring
- **Generation numbers prevent stale updates**: Worker threads may complete out of order; generation numbers ensure only latest results are applied
