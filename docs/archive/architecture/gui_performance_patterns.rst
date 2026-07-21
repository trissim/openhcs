GUI Performance Patterns
========================

OpenHCS GUI implements several performance optimization patterns to maintain responsiveness when editing complex pipelines with many steps and cross-window dependencies.

Cross-Window Preview System
---------------------------

The cross-window preview system enables real-time preview updates in list widgets (like pipeline editor step lists) when users edit configuration values in other windows (like step editor dialogs).

**Problem**

Traditional approach: When a user edits a step's configuration in a dialog, the pipeline editor must refresh its entire step list to show updated preview text. With 20+ steps, this causes:

- Redundant context collection (gathering live values from all open forms)
- Redundant context resolution (building context stacks 20+ times)
- Full widget list rebuilds (destroying and recreating all list items)
- Measured latency: 60ms per keystroke

**Solution Architecture**

The cross-window preview system uses three components:

1. **Token-based caching**: Global token counter invalidates all caches when any value changes
2. **Scope-based routing**: Changes routed to specific items via hierarchical scope IDs
3. **Incremental updates**: Only affected items refresh, not entire lists

**CrossWindowPreviewMixin**

Reusable mixin for widgets that consume cross-window updates. The mixin provides:

1. Scope-based routing for targeted updates
2. Debounced preview updates (100ms trailing debounce)
3. Incremental updates (only affected items refresh)
4. **Configurable preview fields** (per-widget control over which fields show previews)

.. code-block:: python

   from openhcs.pyqt_gui.widgets.mixins import CrossWindowPreviewMixin

   class PipelineEditorWidget(QWidget, CrossWindowPreviewMixin):
       def __init__(self):
           super().__init__()
           self._init_cross_window_preview_mixin()

           # Map editing objects to hierarchical scope ids
           self.register_preview_scope(
               root_name='step',
               editing_types=(FunctionStep,),
               scope_resolver=lambda step, ctx: self._build_step_scope_id(step),
               aliases=('FunctionStep',),
           )
           self.register_preview_scope(
               root_name='global_config',
               editing_types=(GlobalPipelineConfig,),
               scope_resolver=lambda obj, ctx: self.ALL_ITEMS_SCOPE,
               aliases=('GlobalPipelineConfig',),
               process_all_fields=True,
           )

           # Configure which fields to show in previews
           self.enable_preview_for_field(
               'napari_streaming_config.enabled',
               lambda v: 'N:✓' if v else 'N:✗',
               scope_root='step'
           )
           self.enable_preview_for_field(
               'fiji_streaming_config.enabled',
               lambda v: 'F:✓' if v else 'F:✗',
               scope_root='step'
           )
           self.enable_preview_for_field(
               'roi_streaming_config.enabled',
               lambda v: 'R:✓' if v else 'R:✗',
               scope_root='step'
           )

**Configurable Preview Fields**

The mixin provides methods to control which configuration fields are shown in preview labels:

.. code-block:: python

   # Enable preview for a field with custom formatter
   self.enable_preview_for_field(
       'global_config.num_workers',
       lambda v: f'Workers: {v}',
       scope_root='global_config'
   )

   # Enable preview with default str() formatter
   self.enable_preview_for_field(
       'pipeline_config.well_filter',
       scope_root='pipeline_config'
   )

   # Disable preview for a field
   self.disable_preview_for_field('global_config.num_workers')

   # Check if preview is enabled
   if self.is_preview_enabled('napari_streaming_config.enabled'):
       # ...

   # Format a value using registered formatter
   formatted = self.format_preview_value('napari_streaming_config.enabled', True)
   # Returns: 'N:✓'

   # Get all enabled preview fields
   enabled_fields = self.get_enabled_preview_fields()
   # Returns: {'napari_streaming_config.enabled', 'fiji_streaming_config.enabled', ...}

**Scope registration**

``register_preview_scope`` wires editing objects to scope ids used for incremental updates:

.. code-block:: python

   self.register_preview_scope(
       root_name='step',
       editing_types=(FunctionStep,),
       scope_resolver=lambda step, ctx: self._build_step_scope_id(step),
       aliases=('FunctionStep', 'step'),
   )

   self.register_preview_scope(
       root_name='global_config',
       editing_types=(GlobalPipelineConfig,),
       scope_resolver=lambda obj, ctx: self.ALL_ITEMS_SCOPE,
       aliases=('global_config', 'GlobalPipelineConfig'),
       process_all_fields=True,  # Refresh even if field not explicitly registered
   )

Key details:

- ``aliases`` lets you support both lowercase and class-name prefixes in ``field_path``.
- ``scope_root`` in ``enable_preview_for_field`` links a field to the corresponding scope registration.
- ``process_all_fields=True`` tells the mixin to refresh items for any change under that root, even if the field is not explicitly registered (useful for pipeline/global configs that affect everything).
- ``ALL_ITEMS_SCOPE`` refreshes every registered item; ``FULL_REFRESH_SCOPE`` triggers ``_handle_full_preview_refresh``; ``ROOTLESS_SCOPE`` tracks fields without a declared root.

**Centralized Config Formatters**

For consistency across widgets, use the centralized formatters in ``config_preview_formatters.py``:

.. code-block:: python

   from openhcs.pyqt_gui.widgets.config_preview_formatters import (
       CONFIG_INDICATORS,
       format_config_indicator
   )

   # Use centralized indicators (single source of truth)
   # CONFIG_INDICATORS = {
   #     'step_materialization_config': 'MAT',
   #     'napari_streaming_config': 'NAP',
   #     'fiji_streaming_config': 'FIJI',
   # }

   # Format config using centralized formatter
   indicator = format_config_indicator('napari_streaming_config', config, resolve_attr)
   # Returns: 'NAP' (if enabled) or None (if disabled)

   # Both PipelineEditor and PlateManager use these formatters
   # to ensure consistent preview labels (e.g., 'NAP', 'FIJI', 'MAT')

**Enabled Field Checking Rule**

**ARCHITECTURAL RULE**: Any config with an ``enabled: bool`` parameter should only display its preview label if the value resolves to ``True``.

This rule is enforced by the centralized formatters:

.. code-block:: python

   def _check_enabled_field(config: Any, resolve_attr: Optional[Callable] = None) -> bool:
       """Check if a config object is enabled.

       GENERAL RULE: Any config with an 'enabled: bool' parameter should only show
       if it resolves to True.
       """
       import dataclasses

       # Check if config has 'enabled' field
       has_enabled = dataclasses.is_dataclass(config) and 'enabled' in {f.name for f in dataclasses.fields(config)}

       if has_enabled:
           # Resolve enabled field if resolver provided
           if resolve_attr:
               enabled = resolve_attr(None, config, 'enabled', None)
           else:
               enabled = getattr(config, 'enabled', False)

           return bool(enabled)

       # No enabled field - assume enabled
       return True

**Examples**:

- ``NapariStreamingConfig(enabled=True)`` → Shows ``'NAP'`` label
- ``NapariStreamingConfig(enabled=False)`` → Shows nothing (returns ``None``)
- ``FijiStreamingConfig(enabled=True)`` → Shows ``'FIJI'`` label
- ``StepMaterializationConfig(enabled=False)`` → Shows nothing (returns ``None``)

This ensures that disabled configs don't clutter the UI with misleading preview labels.

**Well Filter Handling**:

The formatters correctly handle ``None`` values for ``well_filter`` fields. When a config
has a specific indicator (e.g., ``'NAP'``, ``'FIJI'``, ``'MAT'``) and the ``enabled`` field
is ``True``, the indicator is shown even if ``well_filter`` is ``None``. This preserves
visual consistency in preview labels across different config states.

**Reset Button Refresh Behavior**

``CrossWindowPreviewMixin`` automatically responds to reset button clicks via the ``refresh_handler``:

.. code-block:: python

   def _init_cross_window_preview_mixin(self):
       """Initialize cross-window preview mixin."""
       # ...

       # CRITICAL: Register as external listener for cross-window refresh signals
       # This makes preview labels reactive to live context changes
       # Listen to both value changes AND refresh events (e.g., reset button clicks)
       from openhcs.pyqt_gui.widgets.shared.parameter_form_manager import ParameterFormManager
       ParameterFormManager.register_external_listener(
           self,
           value_changed_handler=self.handle_cross_window_preview_change,
           refresh_handler=self.handle_cross_window_preview_refresh  # Listen to refresh events
       )

   def handle_cross_window_preview_refresh(
       self,
       editing_object: Any,
       context_object: Any,
   ) -> None:
       """Handle cross-window refresh events (e.g., reset button clicks).

       This is called when a ParameterFormManager emits context_refreshed signal,
       which happens when:
       - User clicks Reset button (reset_all_parameters or reset_parameter)
       - User cancels a config editor window (trigger_global_cross_window_refresh)

       Unlike handle_cross_window_preview_change which does incremental updates,
       this triggers a full refresh since reset can affect multiple fields.
       """
       # Extract scope ID and refresh affected items
       # Same logic as handle_cross_window_preview_change

**When refresh_handler is called**:

1. **Reset All button**: User clicks "Reset All" in a config window → all preview labels refresh
2. **Reset Field button**: User clicks reset icon next to a field → affected preview labels refresh
3. **Cancel button**: User cancels a config editor → preview labels revert to saved values

This ensures that preview labels stay synchronized with the actual config state, even when users reset values to defaults.

**Scope IDs**

Hierarchical scope identifiers enable targeted updates:

.. code-block:: python

   # Format: "plate_path::step_token"
   scope_id = f"{orchestrator.plate_path}::{step._pipeline_scope_token}"

   # Example: "/path/to/plate::step_001"
   # Enables routing changes to specific step in specific plate

**Scope Mapping**

Map scope IDs to item keys for incremental updates:

.. code-block:: python

   def _build_scope_index_map(self) -> Dict[str, int]:
       """Map scope IDs to step indices."""
       scope_map = {}
       for idx, step in enumerate(self.pipeline_steps):
           token = getattr(step, '_pipeline_scope_token', None)
           if token:
               scope_id = f"{self.current_plate}::{token}"
               scope_map[scope_id] = idx
       return scope_map

**Implementing Mixin Hooks**

After registering scopes/fields, subclasses still implement the operational hooks:

.. code-block:: python

   def _process_pending_preview_updates(self) -> None:
       """Apply incremental updates for pending keys."""
       # Collect live context ONCE
       # Refresh only items in self._pending_preview_keys

   def _handle_full_preview_refresh(self) -> None:
       """Fallback when incremental updates not possible."""
       # Call update_step_list() or equivalent

   def _merge_with_live_values(...):
       """Merge live overrides into objects returned by _get_preview_instance."""
       # Widget-specific merge logic

**Performance Impact**

- Context collection: 20+ calls → 1 call (cached via token)
- Context resolution: 20+ operations → 1 operation (incremental update)
- Widget updates: Full rebuild → Text-only update on existing widgets
- Measured latency: 60ms → 1ms per keystroke

Dispatch Cycle Caching System
------------------------------

**Problem**: When a user types a single character in a form field, the system triggers multiple expensive operations:

1. Collect live context from all open forms (~2ms)
2. Build context stack with GLOBAL layer resolution (~2ms)
3. Refresh sibling placeholders (5-10 siblings × ~2ms each)
4. Cross-window updates to other windows

With 6 sibling refreshes per keystroke, this totals ~20-30ms per keystroke, making typing feel sluggish.

**Solution**: Dispatch Cycle Caching

The dispatch cycle caching system uses ``contextvars`` to cache expensive computations within a single keystroke's dispatch cycle:

.. code-block:: python

   from openhcs.config_framework.context_manager import dispatch_cycle

   # In FieldChangeDispatcher.dispatch():
   with dispatch_cycle():
       # All operations within this cycle share the same cache
       # First sibling refresh: computes and caches live_context + GLOBAL layer
       # Subsequent siblings: get cache hits (O(1) lookup)
       for sibling_manager in sibling_managers:
           sibling_manager.refresh_with_live_context()

**How It Works**

1. **Context Variable Storage**: ``dispatch_cycle()`` creates a thread-local cache dict
2. **Cache Keys**: Operations use deterministic keys like ``('live_context', scope, type)``
3. **Automatic Invalidation**: Token increments on next keystroke, invalidating all caches
4. **Zero Overhead**: Cache lookups are O(1) dict operations

**Cache Layers**

The system caches at multiple levels:

1. **Live Context Cache** (``collect_live_context()``)
   - Key: ``('live_context', scope_filter, for_type_name)``
   - Value: Dict of all form values for the given scope/type
   - Hit rate: ~90% (same scope/type queried multiple times per keystroke)

2. **GLOBAL Layer Cache** (``build_context_stack()``)
   - Key: ``('global_layer', is_global_config_editing, global_config_type)``
   - Value: Resolved GLOBAL layer for lazy placeholder resolution
   - Hit rate: ~95% (GLOBAL layer same for all siblings)

3. **Placeholder Text Cache** (``apply_placeholder_text()``)
   - Key: Widget instance + placeholder text
   - Value: Cached placeholder text
   - Hit rate: ~80% (same placeholder text for unchanged fields)

**Usage Example**

.. code-block:: python

   from openhcs.config_framework.context_manager import dispatch_cycle, get_dispatch_cache

   def my_operation():
       # Check if we're in a dispatch cycle
       cache = get_dispatch_cache()
       if cache is not None:
           # We're in a dispatch cycle - use the cache
           cache_key = ('my_operation', param1, param2)
           if cache_key in cache:
               return cache[cache_key]  # Cache hit!

           # Cache miss - compute and store
           result = expensive_computation()
           cache[cache_key] = result
           return result
       else:
           # Not in a dispatch cycle - compute directly
           return expensive_computation()

**Performance Impact**

Before dispatch cycle caching:
- 94 keystrokes → 163 ``collect_live_context`` COMPUTING calls
- Each keystroke: ~20-30ms (6 siblings × ~3-5ms each)

After dispatch cycle caching:
- 94 keystrokes → 47 ``collect_live_context`` COMPUTING calls (369 cache hits)
- Each keystroke: ~3-5ms (6 siblings × ~0.5-1ms each)
- **Improvement: 4-6x faster typing**

**Implementation Details**

The dispatch cycle is implemented in ``openhcs/config_framework/context_manager.py``:

.. code-block:: python

   from contextvars import ContextVar

   _dispatch_cycle_cache: ContextVar[Optional[dict]] = ContextVar(
       'dispatch_cycle_cache', default=None
   )

   @contextmanager
   def dispatch_cycle():
       """Context manager for a dispatch cycle. Enables caching of computed values."""
       cache: dict = {}
       token = _dispatch_cycle_cache.set(cache)
       try:
           yield cache
       finally:
           _dispatch_cycle_cache.reset(token)

   def get_dispatch_cache() -> Optional[dict]:
       """Get the current dispatch cycle cache, or None if not in a cycle."""
       return _dispatch_cycle_cache.get()

**Integration Points**

The dispatch cycle is automatically entered at the top level:

1. **FieldChangeDispatcher.dispatch()** - Wraps entire field change handling
2. **LiveContextService.collect_live_context()** - Checks cache before computing
3. **build_context_stack()** - Caches GLOBAL layer resolution
4. **apply_placeholder_text()** - Caches placeholder text by string comparison

**Thread Safety**

``contextvars`` are thread-safe by design:

- Each thread has its own context variable values
- No locks needed
- Safe to use in async code (each async task gets its own context)

**When NOT to Use Dispatch Cycle Caching**

Don't use dispatch cycle caching for:

- Operations that must always reflect current state (e.g., file I/O)
- Operations with side effects (e.g., database writes)
- Long-running operations (cache should be short-lived)

**Debugging Dispatch Cycle Issues**

Enable debug logging to see cache hits/misses:

.. code-block:: python

   import logging
   logging.getLogger('openhcs.config_framework.context_manager').setLevel(logging.DEBUG)
   logging.getLogger('openhcs.config_framework.object_state').setLevel(logging.DEBUG)

Log output will show:

.. code-block:: text

   📦 collect_live_context: DISPATCH CACHE HIT (token=76, scope=None, for_type=GlobalPipelineConfig)
   📦 collect_live_context: COMPUTING (token=76, scope=/path/to/scope, for_type=PipelineConfig)
   🚀 GLOBAL layer CACHE HIT

Eliminating Redundant Cross-Window Refreshes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: The ``config_window.py`` was calling ``trigger_global_cross_window_refresh()`` on every keystroke, which:

1. Called ``refresh_with_live_context()`` for ALL active form managers
2. Triggered full placeholder refresh for every manager
3. Caused O(n) work where n = number of open windows

This was completely redundant because ``FieldChangeDispatcher`` already handles cross-window updates via:

- Sibling refresh (nested managers with same field name)
- Cross-window signals (``context_value_changed``)
- Listener notification (``LiveContextService._notify_change()``)

**Solution**: Remove the redundant ``trigger_global_cross_window_refresh()`` call

.. code-block:: python

   # BEFORE (slow):
   def _sync_global_context_with_current_values(self, source_param: str = None):
       current_values = self.form_manager.get_current_values()
       updated_config = self.config_class(**current_values)
       self.current_config = updated_config
       set_global_config_for_editing(self.config_class, updated_config)
       self._global_context_dirty = True
       ParameterFormManager.trigger_global_cross_window_refresh()  # ❌ REDUNDANT!

   # AFTER (fast):
   def _sync_global_context_with_current_values(self, source_param: str = None):
       current_values = self.form_manager.get_current_values()
       updated_config = self.config_class(**current_values)
       self.current_config = updated_config
       set_global_config_for_editing(self.config_class, updated_config)
       self._global_context_dirty = True
       # FieldChangeDispatcher already handles cross-window updates

**Performance Impact**

- Removed O(n) refresh of all managers per keystroke
- Measured improvement: ~10-15ms per keystroke

Optimizing get_user_modified_values()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: ``get_user_modified_values()`` was calling ``get_current_values()`` which:

1. Reads ALL widget values (expensive)
2. Recursively collects nested manager values
3. Happens on every keystroke during ``collect_live_context()``

But for lazy dataclasses, we only need values for fields in ``_user_set_fields``, not all fields.

**Solution**: Read directly from ``self.parameters`` instead of calling ``get_current_values()``

.. code-block:: python

   # BEFORE (slow):
   def get_user_modified_values(self) -> Dict[str, Any]:
       if not is_lazy_dataclass(self.object_instance):
           return self.get_current_values()  # ❌ Reads ALL widgets

       user_modified = {}
       current_values = self.get_current_values()  # ❌ Expensive!

       for field_name in self._user_set_fields:
           value = current_values.get(field_name)
           # ...

   # AFTER (fast):
   def get_user_modified_values(self) -> Dict[str, Any]:
       if not is_lazy_dataclass(self.object_instance):
           return self.get_current_values()

       user_modified = {}

       # Fast path: if no user-set fields, return empty dict
       if not self._user_set_fields:
           return user_modified

       for field_name in self._user_set_fields:
           # ✅ Read directly from self.parameters (already updated by FieldChangeDispatcher)
           value = self.parameters.get(field_name)
           # ...

**Why This Works**

- ``FieldChangeDispatcher`` updates ``self.parameters`` BEFORE calling any refresh
- For user-set fields, ``self.parameters`` is always the source of truth
- We only need values for fields in ``_user_set_fields``, not all fields
- No need to read widgets or recursively collect nested values

**Performance Impact**

- Eliminated expensive ``get_current_values()`` calls from ``collect_live_context()`` path
- Measured improvement: ~5-10ms per keystroke
- Reduced from 109 ``get_current_values`` calls to ~20 calls per typing session

Live Context Collection
-----------------------

``ParameterFormManager.collect_live_context()`` provides cached access to live form values:

.. code-block:: python

   from openhcs.pyqt_gui.widgets.shared.parameter_form_manager import (
       ParameterFormManager
   )

   # Collect live context with scope filtering
   snapshot = ParameterFormManager.collect_live_context(
       scope_filter=self.current_plate
   )

   # Use snapshot for resolution
   for step_index in indices_to_refresh:
       display_text = self.format_item_for_display(
           step,
           live_context_snapshot=snapshot
       )

**Caching Behavior**

- Token-based: Snapshot cached until token changes
- Scope-filtered: Separate cache entries per scope filter
- Automatic invalidation: Token increments on any form value change
- Type aliasing: Maps lazy/base types for flexible matching

**Token Lifecycle**

1. User edits form field → ``_emit_cross_window_context_changed()``
2. Token incremented → ``_live_context_token_counter += 1``
3. All caches invalidated globally
4. Next ``collect_live_context()`` call recomputes snapshot
5. Subsequent calls with same token return cached snapshot

Async Operations in GUI
----------------------

Heavy operations (file I/O, network requests, blocking waits) must run in background threads to prevent UI freezes.

**Problem**

Blocking operations on the UI thread cause:

- Frozen interface (no repaints, no event processing)
- Unresponsive buttons and menus
- Poor user experience (appears crashed)
- Cannot cancel long-running operations

**Solution: Background Workers**

Move heavy operations to daemon threads:

.. code-block:: python

   import threading

   def on_user_action(self):
       """UI thread: Lightweight checks only."""
       # Check preconditions (cheap)
       if not self.is_valid():
           return

       # Spawn background worker
       threading.Thread(
           target=self._heavy_operation_async,
           args=(param1, param2),
           daemon=True
       ).start()

   def _heavy_operation_async(self, param1, param2):
       """Background thread: Heavy operations."""
       try:
           # Load from disk (blocking I/O)
           data = load_from_file(path)

           # Wait for external service (blocking)
           if not service.wait_for_ready(timeout=15.0):
               raise RuntimeError("Service not ready")

           # Process data (CPU-intensive)
           result = process_data(data)

           # Update UI via signal (thread-safe)
           self._status_update_signal.emit(f"Completed: {result}")

       except Exception as e:
           # Show error dialog on UI thread
           QTimer.singleShot(0, lambda: QMessageBox.warning(
               self, "Error", str(e)
           ))

**Thread-Safe UI Updates**

Never call UI methods directly from background threads. Use Qt signals or QTimer:

.. code-block:: python

   class MyWidget(QWidget):
       # Define signal for cross-thread communication
       _status_update_signal = pyqtSignal(str)

       def __init__(self):
           super().__init__()
           # Connect signal to UI update method
           self._status_update_signal.connect(self._update_status_label)

       def _update_status_label(self, text: str):
           """UI thread: Safe to update widgets."""
           self.status_label.setText(text)

       def _background_worker(self):
           """Background thread: Emit signal instead of direct update."""
           # ❌ WRONG: self.status_label.setText("Loading...")
           # ✅ CORRECT: Emit signal
           self._status_update_signal.emit("Loading...")

**QTimer for One-Shot UI Operations**

Use ``QTimer.singleShot()`` to schedule UI operations from background threads:

.. code-block:: python

   def _background_worker(self):
       """Background thread."""
       try:
           result = expensive_operation()
       except Exception as e:
           # Schedule dialog on UI thread
           QTimer.singleShot(0, lambda: QMessageBox.warning(
               self, "Error", f"Operation failed: {e}"
           ))
           return

       # Schedule success dialog on UI thread
       QTimer.singleShot(0, lambda: QMessageBox.information(
           self, "Success", f"Result: {result}"
       ))

**Daemon Threads**

Always use ``daemon=True`` for background workers:

- Daemon threads automatically terminate when app exits
- Non-daemon threads prevent app from closing
- User doesn't have to wait for background operations to finish

**Example: Async ROI Streaming**

From ``image_browser.py``:

.. code-block:: python

   def _stream_roi_file(self, roi_zip_path: Path):
       """UI thread: Lightweight checks only."""
       # Check which viewers are enabled (cheap)
       napari_enabled = self.napari_enable_checkbox.isChecked()
       fiji_enabled = self.fiji_enable_checkbox.isChecked()

       if not napari_enabled and not fiji_enabled:
           QMessageBox.information(self, "No Viewers", "Enable at least one viewer")
           return

       # Resolve configs on UI thread (cheap)
       napari_config = self._resolve_napari_config()
       fiji_config = self._resolve_fiji_config()

       # Spawn background workers
       if napari_enabled:
           threading.Thread(
               target=self._stream_single_roi_async,
               args=(napari_viewer, roi_zip_path, napari_config),
               daemon=True
           ).start()

       if fiji_enabled:
           threading.Thread(
               target=self._stream_single_roi_async,
               args=(fiji_viewer, roi_zip_path, fiji_config),
               daemon=True
           ).start()

   def _stream_single_roi_async(self, viewer, roi_zip_path, config):
       """Background thread: Heavy operations."""
       try:
           # Load ROIs from disk (blocking I/O)
           self._status_update_signal.emit(f"Loading {roi_zip_path.name}...")
           rois = load_rois_from_zip(roi_zip_path)

           # Wait for viewer (blocking, up to 15s)
           if not viewer.wait_for_ready(timeout=15.0):
               raise RuntimeError("Viewer not ready")

           # Stream to viewer (blocking I/O)
           self._status_update_signal.emit(f"Streaming to viewer...")
           filemanager.save(rois, roi_zip_path, backend, **metadata)

           # Success message on UI thread
           msg = f"Streamed {len(rois)} ROIs"
           self._status_update_signal.emit(msg)

       except Exception as e:
           # Error dialog on UI thread
           QTimer.singleShot(0, lambda: QMessageBox.warning(
               self, "Error", str(e)
           ))

Best Practices
-------------

**When to Use Incremental Updates**

Use incremental updates when:

- List has many items (10+)
- Updates are frequent (per-keystroke)
- Items have stable identities (indices, IDs)
- Preview computation is expensive

**When to Use Full Refresh**

Use full refresh when:

- List structure changes (items added/removed/reordered)
- Scope mapping is invalid or stale
- Incremental update complexity outweighs benefits

**When to Use Background Threads**

Use background threads when:

- Operation blocks for >100ms
- File I/O or network requests
- Waiting for external services
- CPU-intensive processing

**Threading Safety Checklist**

1. ✅ Use ``daemon=True`` for all background threads
2. ✅ Never call UI methods from background threads
3. ✅ Use Qt signals for cross-thread communication
4. ✅ Use ``QTimer.singleShot()`` for one-shot UI operations
5. ✅ Handle exceptions in background threads
6. ✅ Show errors via dialogs on UI thread

**Optimization Checklist**

1. ✅ Collect live context ONCE per refresh cycle
2. ✅ Use token caching for expensive operations
3. ✅ Update existing widgets instead of rebuilding
4. ✅ Batch multiple changes before processing
5. ✅ Use scope filtering to limit context collection
6. ✅ Implement incremental updates for large lists
7. ✅ Move blocking operations to background threads

Log Viewer Performance Optimizations
-------------------------------------

The log viewer implements several performance patterns to minimize UI impact when running in the background while users work in other windows.

Background Syntax Highlighting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Regex-based syntax highlighting is expensive (~1-2ms per line). Running it on the UI thread during paint events causes lag when scrolling or when new log lines arrive.

**Solution**: Move regex parsing to background thread pool, cache results, paint plain text as fallback.

**Architecture**:

.. code-block:: python

   class LogItemDelegate(QStyledItemDelegate):
       def __init__(self):
           self._thread_pool = QThreadPool.globalInstance()
           self._segment_cache: Dict[Tuple[str, str, int], List[HighlightedSegment]] = {}
           self._pending_highlights: Set[Tuple[str, str, int]] = set()

       def paint(self, painter, option, index):
           text = index.data(Qt.DisplayRole)

           # Try to get cached formatting segments (async, may return None)
           segments = self._get_or_request_segments(text, option.font)

           # Create document on main thread (fast)
           doc = QTextDocument()
           doc.setPlainText(text)

           if segments is not None:
               # Formatting ready - apply it (fast)
               self._apply_segments_to_document(doc, segments)
           # else: Paint plain text (still readable while parsing)

           # Paint the document
           doc.drawContents(painter)

       def _get_or_request_segments(self, text, font):
           cache_key = (text, font.family(), font.pointSize())

           # Check cache
           if cache_key in self._segment_cache:
               return self._segment_cache[cache_key]

           # Not in cache - start async parsing if not already pending
           if cache_key not in self._pending_highlights:
               self._pending_highlights.add(cache_key)
               worker = HighlightWorker(text, cache_key, self._color_scheme, self._signals)
               self._thread_pool.start(worker)

           return None  # Caller will paint plain text

**Benefits**:

- UI thread never blocks on regex parsing
- Progressive enhancement: plain text → highlighted text
- Cache provides instant highlighting on subsequent paints
- Scrolling remains smooth even with complex highlighting rules

**Performance**:

- Regex parsing: 1-2ms per line (background thread)
- Format application: <1ms per line (main thread)
- Cache hit: <0.1ms per line
- UI impact: 0ms (async)

Update Throttling
~~~~~~~~~~~~~~~~~

**Problem**: Log tailer checks for new content every 50ms. When new lines arrive, they immediately trigger model updates which cause the entire QListView to repaint. When typing rapidly in pipeline config, these frequent repaints compete for UI thread time.

**Solution**: Buffer incoming log lines and flush at most every 50ms, defer updates when window is hidden.

**Architecture**:

.. code-block:: python

   class LogViewerWindow(QMainWindow):
       def __init__(self):
           self._pending_lines: List[str] = []
           self._update_timer = QTimer()
           self._update_timer.setSingleShot(True)
           self._update_timer.timeout.connect(self._flush_pending_lines)
           self._update_throttle_ms = 50

       def _on_new_content(self, new_content: str, new_file_position: int):
           # Defer updates if window is hidden
           if self.isMinimized() or not self.isVisible():
               self.current_file_position = new_file_position
               return

           lines = new_content.splitlines()

           # Add to pending buffer
           self._pending_lines.extend(lines)

           # Start throttle timer if not already running
           if not self._update_timer.isActive():
               self._update_timer.start(self._update_throttle_ms)

       def _flush_pending_lines(self):
           """Flush pending lines to UI (called by throttle timer)."""
           if not self._pending_lines:
               return

           lines = self._pending_lines
           self._pending_lines = []

           # Insert lines into model
           self.log_model.append_lines(lines)

**Benefits**:

- Multiple log lines arriving within 50ms are batched into single UI update
- Reduces number of model updates and QListView repaints
- UI thread has more time to handle user input in other windows
- Hidden windows don't consume UI resources

**Performance**:

- Before throttling: 10 updates/second = 10 repaints/second
- After throttling: 1 update per 50ms burst = 1 repaint per burst
- Typing latency improvement: ~40ms (measured in pipeline config)

Type-Based Inheritance Filtering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: When typing in a nested config field (e.g., ``WellFilterConfig.well_filter``), the cross-window update system was refreshing ALL sibling nested configs (e.g., ``VFSConfig``, ``NapariStreamingConfig``) even though only configs inheriting from ``WellFilterConfig`` could be affected.

**Solution**: Use ``isinstance()`` checks to only refresh sibling configs whose object instances inherit from the changed config type.

**Architecture**:

.. code-block:: python

   def _on_nested_parameter_changed(self, emitting_manager_name: str):
       # Get the emitting manager's type
       emitting_manager = self.nested_managers.get(emitting_manager_name)
       emitting_type = emitting_manager.dataclass_type if emitting_manager else None

       def should_refresh_sibling(name: str, manager) -> bool:
           if name == emitting_manager_name:
               return False  # Don't refresh the emitting manager itself
           if not emitting_type:
               return True  # Conservative: refresh if we can't determine
           # Check if sibling's object instance inherits from emitting type
           return isinstance(manager.object_instance, emitting_type)

       # Only refresh affected siblings
       self._apply_to_nested_managers(
           lambda name, manager: (
               manager._refresh_all_placeholders(live_context=live_context)
               if should_refresh_sibling(name, manager)
               else None
           )
       )

**Example**:

When editing ``WellFilterConfig.well_filter`` in ``PipelineConfig``:

- ✅ Refresh ``NapariStreamingConfig`` (inherits from ``WellFilterConfig`` via ``StreamingDefaults`` → ``StepWellFilterConfig``)
- ❌ Skip ``VFSConfig`` (doesn't inherit from ``WellFilterConfig``)

**Benefits**:

- Eliminates unnecessary placeholder refreshes
- Reduces cross-window update overhead
- Cleaner logs (no more "Skipping cross-window update" spam)

**Performance**:

- Before: 3-5 sibling refreshes per keystroke (all siblings)
- After: 0-2 sibling refreshes per keystroke (only affected siblings)
- Measured improvement: ~5-10ms per keystroke in complex configs
