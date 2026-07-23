====================================
Scope Hierarchy for Live Context
====================================

*Module: openhcs.pyqt_gui.widgets.shared.parameter_form_manager*  
*Status: STABLE*

---

Overview
========

Multiple orchestrators can exist simultaneously (different plates, different pipelines). The scope hierarchy system prevents cross-orchestrator parameter bleed-through by ensuring step and function editors only see parameters from their own orchestrator context.

Problem Context
===============

Without scope isolation, parameter forms collect live context from all open windows:

.. code-block:: python

    # Orchestrator A: Plate 1, Step 1
    step_editor_A.get_parameter('gaussian_sigma')  # Should see Step 1 values
    
    # Orchestrator B: Plate 2, Step 2 (different plate!)
    step_editor_B.get_parameter('gaussian_sigma')  # Should NOT see Step 1 values
    
    # Without scope isolation:
    # Both editors see each other's parameters → incorrect placeholder text

Scope isolation ensures each editor only sees parameters from its own orchestrator.

Solution: Hierarchical Scope IDs
=================================

Scope IDs create a hierarchy that matches the orchestrator → step relationship.

**Current Implementation** (uses ``::`` separator and plate_path):

.. code-block:: python

    # Orchestrator/Plate scope (unique per plate)
    plate_scope = str(orchestrator.plate_path)
    # Example: "/data/plates/plate_001"

    # Step scope (inherits plate scope)
    step_token = getattr(step, '_pipeline_scope_token', step.name)
    step_scope = f"{plate_scope}::{step_token}"
    # Example: "/data/plates/plate_001::step_0"

**Note**: The ``::`` (double colon) separator is used for hierarchical scoping, not ``.`` (period).

Editors with matching scope prefixes share live context. Editors with different scopes are isolated.

Scope Hierarchy Architecture
=============================

Two-Level Hierarchy
--------------------

.. code-block:: python

    # Level 1: Plate/Orchestrator (uses actual plate_path)
    scope_id = "/data/plates/plate_001"
    # Shared by: All config editors for this plate

    # Level 2: Step (inherits plate scope with :: separator)
    scope_id = "/data/plates/plate_001::step_0"
    # Shared by: Step editor and its function editor for this specific step

**Real Examples from Code**:

.. code-block:: python

    # From dual_editor_window.py:240-245
    def _build_step_scope_id(self, fallback_name: str) -> str:
        plate_scope = getattr(self.orchestrator, 'plate_path', 'no_orchestrator')
        token = getattr(self.editing_step, '_pipeline_scope_token', None)
        if token:
            return f"{plate_scope}::{token}"
        return f"{plate_scope}::{fallback_name}"

    # From plate_manager.py:419, 1141
    scope_id = str(orchestrator.plate_path)  # Plate-level config editing

**Key insight**: Step and function editors share the same scope prefix (including step token), enabling them to see each other's live parameters while remaining isolated from other orchestrators and other steps.

Scope Matching Logic
--------------------

**Actual Implementation** (from ``parameter_form_manager.py:375-393``):

.. code-block:: python

    @staticmethod
    def _is_scope_visible_static(manager_scope: str, filter_scope) -> bool:
        """
        Check if scopes match (prefix matching for hierarchical scopes).
        Supports generic hierarchical scope strings like 'x::y::z'.
        """
        # Convert filter_scope to string if it's a Path
        filter_scope_str = str(filter_scope) if not isinstance(filter_scope, str) else filter_scope

        return (
            manager_scope == filter_scope_str or
            manager_scope.startswith(f"{filter_scope_str}::") or
            filter_scope_str.startswith(f"{manager_scope}::")
        )

**Examples**:

.. code-block:: python

    # Same plate: plate scope matches step scope (parent-child)
    _is_scope_visible_static(
        "/data/plates/plate_001::step_0",     # manager_scope (step)
        "/data/plates/plate_001"               # filter_scope (plate)
    )
    # → True (step scope starts with "plate_scope::")

    # Different plates: no match
    _is_scope_visible_static(
        "/data/plates/plate_001::step_0",     # manager_scope
        "/data/plates/plate_002"               # filter_scope
    )
    # → False (different plate prefixes)

This ensures step and function editors see each other's parameters (same plate/step) while remaining isolated from other orchestrators.

Implementation Patterns
=======================

Dual Editor Window
------------------

Step and function editors share scope to enable parameter synchronization.

**Actual Implementation** (from ``dual_editor_window.py:240-267``):

.. code-block:: python

    class DualEditorWindow(BaseFormDialog):
        def _build_step_scope_id(self, fallback_name: str) -> str:
            plate_scope = getattr(self.orchestrator, 'plate_path', 'no_orchestrator')
            token = getattr(self.editing_step, '_pipeline_scope_token', None)
            if token:
                return f"{plate_scope}::{token}"
            return f"{plate_scope}::{fallback_name}"

        def create_step_tab(self):
            step_name = getattr(self.editing_step, 'name', 'unknown_step')
            scope_id = self._build_step_scope_id(step_name)
            # Result: "/data/plates/plate_001::step_0"

            # Step editor uses step scope
            self.step_editor = StepParameterEditorWidget(
                self.editing_step,
                scope_id=scope_id
            )

        def create_function_tab(self):
            step_name = getattr(self.editing_step, 'name', 'unknown_step')
            scope_id = self._build_step_scope_id(step_name)
            # Same scope as step editor!

            # Function editor uses same step scope
            self.func_editor = FunctionListEditorWidget(
                scope_id=scope_id  # Same as step editor
            )

**Why same scope?** Function editor needs to see step parameters (e.g., ``processing_config.group_by``) for placeholder resolution.

Function List Editor
--------------------

.. code-block:: python

    class FunctionListEditor(QWidget):
        def __init__(self, step, scope_id):
            self.step = step
            self.scope_id = scope_id  # Inherited from dual editor
            
        def refresh_from_step_context(self):
            """Refresh function editor when step parameters change."""
            # Collect live context from step editor (same scope)
            live_context = self._collect_live_context()
            
            # Resolve group_by placeholder using step's processing_config
            group_by_value = self._resolve_group_by_placeholder(live_context)
            
            # Update UI
            self.group_by_selector.setCurrentValue(group_by_value)

Config Window
-------------

Config windows use plate-scoped or global scope depending on the config being edited.

**Actual Implementation** (from ``config_window.py:59,77,117`` and ``plate_manager.py:1141-1148``):

.. code-block:: python

    class ConfigWindow(BaseFormDialog):
        def __init__(self, config_class, current_config, ...,
                     scope_id: Optional[str] = None):
            # scope_id passed from caller
            self.scope_id = scope_id

            self.form_manager = ParameterFormManager.from_dataclass_instance(
                dataclass_instance=current_config,
                scope_id=self.scope_id  # Plate-scoped or None for global
            )

    # From plate_manager.py - creating plate-scoped config window
    scope_id = str(orchestrator.plate_path) if orchestrator else None
    config_window = ConfigWindow(
        config_class,
        current_config,
        scope_id=scope_id  # "/data/plates/plate_001" or None
    )

**Scope Semantics**:

- ``scope_id=None``: Global config (GlobalPipelineConfig) - visible to all orchestrators
- ``scope_id="/data/plates/plate_001"``: Plate-scoped config (PipelineConfig) - only visible to this plate's editors

Scope Isolation Examples
=========================

Isolated Orchestrators
-----------------------

.. code-block:: python

    # Orchestrator A: Plate 1
    orchestrator_A = PipelineOrchestrator(plate_path=Path("/data/plates/plate_001"))
    scope_A = str(orchestrator_A.plate_path)  # "/data/plates/plate_001"

    # Orchestrator B: Plate 2
    orchestrator_B = PipelineOrchestrator(plate_path=Path("/data/plates/plate_002"))
    scope_B = str(orchestrator_B.plate_path)  # "/data/plates/plate_002"

    # Step editors are isolated (using actual implementation)
    step_editor_A = StepParameterEditor(step_A, scope_id=f"{scope_A}::step_0")
    step_editor_B = StepParameterEditor(step_B, scope_id=f"{scope_B}::step_0")
    # step_editor_A: "/data/plates/plate_001::step_0"
    # step_editor_B: "/data/plates/plate_002::step_0"

    # step_editor_A does NOT see step_editor_B's parameters
    # Different scope prefixes: "/data/plates/plate_001" vs "/data/plates/plate_002"

Shared Step/Function Context
-----------------------------

.. code-block:: python

    # Same plate, same step
    plate_scope = "/data/plates/plate_001"
    step_scope = f"{plate_scope}::step_0"  # "/data/plates/plate_001::step_0"

    # Step editor
    step_editor = StepParameterEditor(step, scope_id=step_scope)

    # Function editor (same scope!)
    function_editor = FunctionListEditor(step, scope_id=step_scope)

    # function_editor DOES see step_editor's parameters
    # Same scope: "/data/plates/plate_001::step_0"

Cross-Window Synchronization
=============================

The scope system enables selective cross-window synchronization:

.. code-block:: python

    # User edits step parameter in step editor
    step_editor.update_parameter('processing_config.group_by', GroupBy.WELL)
    
    # Triggers cross-window refresh
    ParameterFormManager.trigger_global_cross_window_refresh()
    
    # Function editor receives refresh signal (same scope)
    function_editor.refresh_from_step_context()
    
    # Function editor sees updated group_by value
    # Updates group_by selector to match step editor

**Key insight**: Scope matching ensures only related windows refresh, preventing unnecessary updates.

Recursive Live Context Collection
==================================

The ``collect_live_context()`` method recursively collects values from all managers
AND their nested managers via ``_collect_from_manager_tree()``:

.. code-block:: python

   @classmethod
   def _collect_from_manager_tree(cls, manager, result: dict, scoped_result: dict = None):
       """Recursively collect values from manager and all nested managers."""
       if manager.dataclass_type:
           result[manager.dataclass_type] = manager.get_user_modified_values()
           if scoped_result is not None and manager.scope_id:
               scoped_result.setdefault(manager.scope_id, {})[manager.dataclass_type] = result[manager.dataclass_type]

       # Recurse into nested managers
       for nested in manager.nested_managers.values():
           cls._collect_from_manager_tree(nested, result, scoped_result)

This enables sibling inheritance: when ``live_context`` contains both
``LazyStepWellFilterConfig`` and ``LazyWellFilterConfig`` values,
``_find_live_values_for_type()`` can use ``issubclass()`` matching to find
``StepWellFilterConfig`` values when resolving ``WellFilterConfig`` placeholders.

**Example**: Step form has two nested config managers:

- ``step_well_filter_config`` → ``LazyStepWellFilterConfig(well_filter=123)``
- ``well_filter_config`` → ``LazyWellFilterConfig(well_filter=None)``

Old behavior: Only root manager's values collected (nested values missed).

New behavior: Both nested managers' values collected, enabling:

1. ``well_filter_config.well_filter`` needs placeholder
2. ``_find_live_values_for_type(LazyWellFilterConfig, live_context)`` called
3. Finds ``LazyStepWellFilterConfig`` via ``issubclass(StepWellFilterConfig, WellFilterConfig)``
4. Returns ``step_well_filter_config`` values with ``well_filter=123``
5. Placeholder shows "Pipeline default: 123"

See :doc:`../architecture/field_change_dispatcher` for how changes trigger sibling refresh.

Implementation Notes
====================

**🔬 Source Code**:

- Scope matching: ``openhcs/pyqt_gui/widgets/shared/parameter_form_manager.py``
- Recursive collection: ``_collect_from_manager_tree()`` in same file
- Dual editor scope setup: ``openhcs/pyqt_gui/windows/dual_editor_window.py``
- Function editor scope: ``openhcs/pyqt_gui/widgets/function_list_editor.py``

**🏗️ Architecture**:

- :doc:`../architecture/field_change_dispatcher` - Unified field change handling
- :doc:`../architecture/context_system` - Configuration context and inheritance

**📊 Performance**:

- Scope matching is O(n) where n = number of active form managers
- Typically < 10 managers active, so overhead is negligible
- Scope string comparison is fast (prefix matching)
- Recursive collection adds minimal overhead (tree depth typically < 5)

Key Design Decisions
====================

**Why use plate_path for orchestrator scope instead of object ID?**

- Plate path is semantically meaningful (shows which plate the editor is for)
- Plate path is stable across sessions (object ID changes each run)
- Plate path enables future scope persistence/serialization
- In practice, only one orchestrator per plate is active at a time

**Why use ``::`` separator instead of ``.``?**

- Avoids conflicts with file paths (which use ``.`` for extensions)
- More visually distinct in logs and debugging
- Consistent with other path-like separators in the codebase

**Why share scope between step and function editors?**

Function editor needs step parameters for placeholder resolution (e.g., ``group_by`` selector). Sharing scope enables this without manual parameter passing.

**Why use strings for scope IDs?**

Scope IDs are strings, enabling serialization and comparison without object reference issues.

Common Gotchas
==============

- **Don't use global scope (``None``) for plate-specific editors**: Each plate must have unique scope (``plate_path``) to prevent parameter bleed-through
- **Step and function editors must share scope**: Function editor needs step parameters for placeholder resolution
- **Scope IDs are immutable**: Don't change ``scope_id`` after form manager creation
- **Scope matching uses ``::`` separator**: ``"/data/plates/plate_001"`` matches ``"/data/plates/plate_001::step_0"`` but not ``"/data/plates/plate_002"``
- **Separator matters**: Use ``::`` (double colon), not ``.`` (period) or ``:`` (single colon)

Debugging Scope Issues
======================

Symptom: Function Editor Not Syncing
-------------------------------------

**Cause**: Step and function editors have different scopes

**Diagnosis**:

.. code-block:: python

    # Check scope IDs
    logger.debug(f"Step editor scope: {step_editor.form_manager.scope_id}")
    logger.debug(f"Function editor scope: {function_editor.scope_id}")
    # Should be identical (including step token)

**Fix**: Ensure both editors use same ``scope_id`` from ``_build_step_scope_id()``

Symptom: Cross-Orchestrator Bleed-Through
------------------------------------------

**Cause**: Multiple orchestrators sharing same scope prefix

**Diagnosis**:

.. code-block:: python

    # Check orchestrator/plate scopes
    logger.debug(f"Orchestrator A scope: {scope_A}")
    logger.debug(f"Orchestrator B scope: {scope_B}")
    # Should have different plate_path prefixes

**Fix**: Ensure each orchestrator uses unique ``plate_path`` as scope prefix


Pipeline Editor Preview Labels
===============================

The pipeline editor displays real-time preview labels (MAT, NAP, FIJI, FILT) that show which configurations are enabled for each step. These labels must update immediately when fields are changed in the step editor, including when fields are reset to None.

Critical Implementation Details
--------------------------------

**Problem**: Preview labels were not updating when resetting fields that had concrete saved values.

**Root Cause**: The pipeline editor was resolving config objects from the original saved step instead of from the merged step with live values.

When a step is saved with a concrete value (e.g., ``napari_streaming_config.enabled=True``), and then reset to None in the step editor:

1. The live form manager has ``enabled=None`` in its current values
2. The pipeline editor collects live context and merges it into a new step object
3. **BUG**: The config object being resolved was from the ORIGINAL saved step, not the merged step
4. Result: Lazy resolution sees the saved concrete value instead of the live None value

**Solution**: Resolve config from merged step, not original step

.. code-block:: python

    # WRONG: Resolve from original step's config
    config = getattr(step, 'napari_streaming_config')  # Has saved enabled=True
    # ... merge live values into step ...
    resolved = config.enabled  # Still resolves to True!

    # CORRECT: Resolve from merged step's config
    step_to_use = merge_live_values(step, live_values)  # Has enabled=None
    config = getattr(step_to_use, 'napari_streaming_config')  # Has live enabled=None
    resolved = config.enabled  # Correctly resolves to None, walks up context

**Implementation** (``openhcs/pyqt_gui/widgets/pipeline_editor.py``):

.. code-block:: python

    # Build merged step with live values
    step_to_use = step
    if step_live_values:
        step_to_use = self._merge_live_values(step, step_live_values)

    # CRITICAL: Get config from merged step, not original step!
    config_to_resolve = getattr(step_to_use, config_attr_name, config)

    # Now resolve through context stack
    with config_context(global_config):
        with config_context(pipeline_config):
            with config_context(step_to_use):
                resolved_value = config_to_resolve.enabled

This ensures that when a field is reset to None in the step editor, the pipeline editor sees the None value and correctly resolves it through the context hierarchy (GlobalPipelineConfig → PipelineConfig → Step).

Scope Matching for Step Editors
--------------------------------

The pipeline editor must match step editors by scope_id to collect the correct live values:

.. code-block:: python

    # Build step-specific scope
    step_scope = f"{plate_path}::{step.name}"

    # Only collect live context from:
    # 1. Global scope (None)
    # 2. Exact plate scope match
    # 3. Exact step scope match (for THIS specific step)
    is_visible = (
        manager.scope_id is None or
        manager.scope_id == plate_scope or
        manager.scope_id == step_scope
    )

This prevents collecting live values from other step editors in the same plate, ensuring each step's preview labels only reflect its own editor's state.
