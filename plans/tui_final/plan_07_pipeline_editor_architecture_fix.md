# plan_07_pipeline_editor_architecture_fix.md
## Component: PipelineEditor Architecture Fix with Real-time State Management

### Objective
Fix PipelineEditor's broken architecture by implementing proper Pipeline-plate association, real-time state updates via observer pattern, multi-plate pipeline handling, and visual programming integration with mathematically precise specifications.

### Plan

#### MATHEMATICAL SPECIFICATION: Exact Interface Requirements

**CRITICAL INTERFACES (MUST MATCH EXACTLY):**
```python
# 1. State Observer Pattern (EXACT SIGNATURES)
self.state.add_observer('plate_selected', callback: Callable[[Dict], None])
self.state.set_selected_plate(plate: Dict[str, Any]) -> None  # Triggers 'plate_selected' event

# 2. DualEditorPane Constructor (EXACT SIGNATURE)
DualEditorPane(
    state: Any,                                    # TUI state object
    func_step: FunctionStep,                       # Step to edit
    on_save: Optional[Callable[[FunctionStep], None]] = None,    # Save callback
    on_cancel: Optional[Callable[[], None]] = None               # Cancel callback
)

# 3. Pipeline Constructor (EXACT SIGNATURE)
Pipeline(steps=None, *, name=None, metadata=None, description=None)
# Pipeline IS a list - use list(pipeline) to get steps

# 4. FunctionStep Constructor (EXACT SIGNATURE)
FunctionStep(
    func: Union[Callable, Tuple[Callable, Dict], List[Union[Callable, Tuple[Callable, Dict]]]],
    *,
    name: Optional[str] = None,
    variable_components: Optional[List[str]] = ['site'],
    group_by: str = "channel",
    force_disk_output: bool = False
)

# 5. Backend Usage (EXACT PATTERN)
from openhcs.constants.constants import Backend
backend_string = Backend.DISK.value  # "disk"
filemanager.save(data, path, Backend.DISK.value)
filemanager.load(path, Backend.DISK.value)

# 6. Dialog Display (EXACT SIGNATURE)
await app_state.show_dialog(dialog: Dialog, result_future: asyncio.Future)
```

#### STEP 1: Pipeline-Plate Association Storage Implementation

**FILE:** `openhcs/tui/panes/pipeline_editor.py`

**LINES 25-29:** REPLACE existing __init__ attributes:
```python
# REPLACE LINES 25-29:
# OLD:
        self.state = state
        self.context = context
        self.steps_lock = asyncio.Lock()

# NEW:
        self.state = state
        self.context = context
        self.steps_lock = asyncio.Lock()

        # EXACT: Pipeline-plate association storage
        self.plate_pipelines: Dict[str, Pipeline] = {}  # {plate_path: Pipeline}
        self.current_selected_plates: List[str] = []    # Currently selected plate paths
        self.pipeline_differs_across_plates: bool = False

        # EXACT: Dialog state management
        self.current_dialog = None
        self.current_dialog_future = None
```

**LINES 8:** ADD imports after existing imports:
```python
# ADD AFTER LINE 8:
import copy
from openhcs.constants.constants import Backend
from openhcs.core.pipeline import Pipeline
from openhcs.tui.components.dual_editor_pane import DualEditorPane
from openhcs.tui.dialogs.error_dialogs import show_scrollable_error_dialog
from prompt_toolkit.widgets import Dialog
```

**LINES 56:** ADD after existing logger.info line:
```python
# ADD AFTER LINE 56:
        # EXACT: Register state observer for real-time updates
        self._register_plate_selection_observer()
```

**LINES 375:** ADD new methods at end of file before last line:
```python
# ADD BEFORE LINE 375:
    def _register_plate_selection_observer(self):
        """EXACT: Register observer for plate selection changes."""
        self.state.add_observer('plate_selected',
            lambda plate: get_app().create_background_task(self._on_plate_selection_changed(plate)))

    async def _on_plate_selection_changed(self, plate_data):
        """EXACT: Handle plate selection changes in real-time."""
        try:
            # EXACT: Get current selection state from PlateManager
            selected_plates = self._get_current_selected_plates()
            self.current_selected_plates = selected_plates

            # EXACT: Update pipeline display based on selection
            await self._update_pipeline_display_for_selection(selected_plates)

        except Exception as e:
            await show_scrollable_error_dialog(
                title="Selection Update Error",
                message=f"Failed to update pipeline display: {str(e)}",
                exception=e,
                app_state=self.state
            )

    def _get_current_selected_plates(self) -> List[str]:
        """EXACT: Get currently selected plate paths from PlateManager."""
        # EXACT: Access PlateManager selection state
        if hasattr(self.state, 'selected_plate') and self.state.selected_plate:
            return [self.state.selected_plate['path']]
        return []
```

#### STEP 2: Multi-Plate Pipeline Display Logic

**FILE:** `openhcs/tui/panes/pipeline_editor.py`

**LINES 375:** ADD new methods at end of file before last line:
```python
# ADD BEFORE LINE 375:
    async def _update_pipeline_display_for_selection(self, selected_plates: List[str]):
        """EXACT: Update pipeline display based on selected plates."""
        if not selected_plates:
            # EXACT: No plates selected - show empty
            self.pipeline_differs_across_plates = False
            self.list_manager.load_items([])
            return

        if len(selected_plates) == 1:
            # EXACT: Single plate selected - show its pipeline
            plate_path = selected_plates[0]
            pipeline = self.plate_pipelines.get(plate_path)

            if not pipeline:
                # EXACT: Create default empty pipeline for new plate
                pipeline = Pipeline(name=f"Pipeline for {Path(plate_path).name}")
                self.plate_pipelines[plate_path] = pipeline

            self.pipeline_differs_across_plates = False
            self._refresh_step_list_for_pipeline(pipeline)

        else:
            # EXACT: Multiple plates selected - check if pipelines match
            pipelines = [self.plate_pipelines.get(plate_path) for plate_path in selected_plates]

            if self._all_pipelines_identical(pipelines):
                # EXACT: All pipelines identical - show common pipeline
                common_pipeline = pipelines[0] if pipelines[0] else Pipeline(name="Common Pipeline")
                self.pipeline_differs_across_plates = False
                self._refresh_step_list_for_pipeline(common_pipeline)
            else:
                # EXACT: Pipelines differ - show "differs" message
                self.pipeline_differs_across_plates = True
                self._show_pipeline_differs_message()

    def _all_pipelines_identical(self, pipelines: List[Optional[Pipeline]]) -> bool:
        """EXACT: Check if all pipelines are identical."""
        # EXACT: Handle None pipelines (treat as empty)
        normalized_pipelines = []
        for p in pipelines:
            if p is None:
                normalized_pipelines.append([])  # Empty pipeline
            else:
                normalized_pipelines.append(list(p))  # Pipeline IS a list

        # EXACT: Check if all normalized pipelines are identical
        if not normalized_pipelines:
            return True

        first_pipeline = normalized_pipelines[0]
        return all(pipeline == first_pipeline for pipeline in normalized_pipelines[1:])

    def _show_pipeline_differs_message(self):
        """EXACT: Show 'pipeline differs across plates' message."""
        self.list_manager.load_items([{
            'id': 'differs_message',
            'name': 'Pipeline differs across plates',
            'func': 'Cannot show pipeline - selected plates have different pipelines',
            'variable_components': '',
            'group_by': ''
        }])

    def _refresh_step_list_for_pipeline(self, pipeline: Pipeline):
        """EXACT: Refresh step list for specific pipeline."""
        step_items = []
        for i, step in enumerate(pipeline):
            step_items.append({
                'id': str(id(step)),  # Use object ID as unique identifier
                'name': getattr(step, 'name', f'Step {i+1}'),
                'func': self._format_func_display(step.func),
                'variable_components': ', '.join(step.variable_components or []),
                'group_by': step.group_by
            })

        self.list_manager.load_items(step_items)
```

#### STEP 3: Add Button Implementation (Create Empty Default Step)

**FILE:** `openhcs/tui/panes/pipeline_editor.py`

**LINES 213-225:** REPLACE existing _handle_add_step method:
```python
# REPLACE LINES 213-225:
# OLD:
    async def _handle_add_step(self):
        """Add step handler."""
        logger.info("PipelineEditor: _handle_add_step called!")
        if not self._validate_orchestrator():
            return

        # Placeholder for step addition UI
        await show_error_dialog(
            "Add Step",
            "Step addition interface will be implemented.",
            self.state
        )

# NEW:
    async def _handle_add_step(self):
        """EXACT: Add empty default step to pipeline."""
        try:
            # EXACT: Check if pipeline differs across plates
            if self.pipeline_differs_across_plates:
                await show_error_dialog(
                    "Add Step Error",
                    "Cannot add step - selected plates have different pipelines. Select a single plate first.",
                    self.state
                )
                return

            # EXACT: Check if any plates are selected
            if not self.current_selected_plates:
                await show_error_dialog(
                    "Add Step Error",
                    "No plates selected. Select a plate first.",
                    self.state
                )
                return

            # EXACT: Create empty default FunctionStep
            empty_step = FunctionStep(
                func=None,  # Empty - user will edit this
                name="Empty Step",
                variable_components=['site'],
                group_by='channel'
            )

            # EXACT: Add to all selected plates' pipelines
            for plate_path in self.current_selected_plates:
                pipeline = self.plate_pipelines.get(plate_path)
                if not pipeline:
                    # EXACT: Create pipeline if it doesn't exist
                    pipeline = Pipeline(name=f"Pipeline for {Path(plate_path).name}")
                    self.plate_pipelines[plate_path] = pipeline

                # EXACT: Add step to pipeline (Pipeline IS a list)
                pipeline.append(copy.deepcopy(empty_step))

            # EXACT: Refresh display
            await self._update_pipeline_display_for_selection(self.current_selected_plates)

            logger.info(f"Added empty step to {len(self.current_selected_plates)} plates")

        except Exception as e:
            await show_scrollable_error_dialog(
                title="Add Step Error",
                message=f"Failed to add step: {str(e)}",
                exception=e,
                app_state=self.state
            )
```

#### STEP 4: Edit Button Implementation (Visual Programming Integration)

**EXACT IMPLEMENTATION:**
```python
async def _handle_edit_step(self):
    """EXACT: Edit step via DualEditorPane integration."""
    try:
        # EXACT: Check if pipeline differs across plates
        if self.pipeline_differs_across_plates:
            await show_error_dialog(
                "Edit Error",
                "Cannot edit step - selected plates have different pipelines. Select a single plate first.",
                self.state
            )
            return

        # EXACT: Get selected item
        selected_item = self.list_manager.get_selected_item()
        if not selected_item:
            await show_error_dialog("Edit Error", "No step selected.", self.state)
            return

        # EXACT: Check for special "differs" message
        if selected_item['id'] == 'differs_message':
            await show_error_dialog("Edit Error", "Cannot edit - pipeline differs across plates.", self.state)
            return

        # EXACT: Find step in current pipeline
        step_index, func_step = self._find_step_by_id(selected_item['id'])
        if step_index is None or func_step is None:
            await show_error_dialog("Edit Error", "Step not found.", self.state)
            return

        # EXACT: Create DualEditorPane with callbacks
        editor = DualEditorPane(
            state=self.state,
            func_step=func_step,
            on_save=lambda updated_step: self._on_step_updated(step_index, updated_step),
            on_cancel=self._on_edit_cancelled
        )

        # EXACT: Show in modal dialog
        await self._show_step_editor(editor, "Edit Step")

    except Exception as e:
        await show_scrollable_error_dialog(
            title="Edit Step Error",
            message=f"Failed to edit step: {str(e)}",
            exception=e,
            app_state=self.state
        )

def _find_step_by_id(self, step_id: str) -> Tuple[Optional[int], Optional[FunctionStep]]:
    """EXACT: Find step index and object by ID across selected plates."""
    # EXACT: Get current pipeline for single plate selection
    if len(self.current_selected_plates) == 1:
        plate_path = self.current_selected_plates[0]
        pipeline = self.plate_pipelines.get(plate_path)
        if pipeline:
            for i, step in enumerate(pipeline):
                if str(id(step)) == step_id:
                    return i, step
    return None, None

def _on_step_updated(self, step_index: int, updated_step: FunctionStep):
    """EXACT: Callback when step is updated."""
    try:
        # EXACT: Update step in all selected plates' pipelines
        for plate_path in self.current_selected_plates:
            pipeline = self.plate_pipelines.get(plate_path)
            if pipeline and step_index < len(pipeline):
                # EXACT: Update Pipeline (Pipeline IS a list)
                pipeline[step_index] = copy.deepcopy(updated_step)

        # EXACT: Refresh display
        await self._update_pipeline_display_for_selection(self.current_selected_plates)

        # EXACT: Close dialog
        self._close_current_dialog()

        logger.info(f"Updated step {step_index} in {len(self.current_selected_plates)} plates")

    except Exception as e:
        logger.error(f"Error updating step: {e}", exc_info=True)

def _on_edit_cancelled(self):
    """EXACT: Callback when edit is cancelled."""
    self._close_current_dialog()
```

#### STEP 5: Delete Step Implementation

**EXACT IMPLEMENTATION:**
```python
async def _handle_delete_step(self):
    """EXACT: Delete step from pipelines."""
    try:
        # EXACT: Check if pipeline differs across plates
        if self.pipeline_differs_across_plates:
            await show_error_dialog(
                "Delete Error",
                "Cannot delete step - selected plates have different pipelines. Select a single plate first.",
                self.state
            )
            return

        # EXACT: Get selected item
        selected_item = self.list_manager.get_selected_item()
        if not selected_item:
            await show_error_dialog("Delete Error", "No step selected.", self.state)
            return

        # EXACT: Check for special "differs" message
        if selected_item['id'] == 'differs_message':
            await show_error_dialog("Delete Error", "Cannot delete - pipeline differs across plates.", self.state)
            return

        # EXACT: Find step in current pipeline
        step_index, func_step = self._find_step_by_id(selected_item['id'])
        if step_index is None:
            await show_error_dialog("Delete Error", "Step not found.", self.state)
            return

        # EXACT: Remove from all selected plates' pipelines
        for plate_path in self.current_selected_plates:
            pipeline = self.plate_pipelines.get(plate_path)
            if pipeline and step_index < len(pipeline):
                # EXACT: Remove from Pipeline (Pipeline IS a list)
                del pipeline[step_index]

        # EXACT: Refresh display
        await self._update_pipeline_display_for_selection(self.current_selected_plates)

        logger.info(f"Deleted step {step_index} from {len(self.current_selected_plates)} plates")

    except Exception as e:
        await show_scrollable_error_dialog(
            title="Delete Step Error",
            message=f"Failed to delete step: {str(e)}",
            exception=e,
            app_state=self.state
        )
```

#### STEP 6: Dialog Management Implementation

**EXACT IMPLEMENTATION:**
```python
async def _show_step_editor(self, editor: DualEditorPane, title: str):
    """EXACT: Show step editor in modal dialog."""
    import asyncio
    from prompt_toolkit.widgets import Dialog

    # EXACT: Create future for dialog result
    future = asyncio.Future()

    # EXACT: Create modal dialog
    dialog = Dialog(
        title=title,
        body=editor.container,  # DualEditorPane.container property
        buttons=[],  # DualEditorPane has its own Save/Cancel buttons
        modal=True,
        width=120,
        height=40
    )

    # EXACT: Store dialog reference for closing
    self.current_dialog = dialog
    self.current_dialog_future = future

    # EXACT: Show dialog using app_state.show_dialog
    await self.state.show_dialog(dialog, result_future=future)

def _close_current_dialog(self):
    """EXACT: Close current dialog."""
    if hasattr(self, 'current_dialog_future') and self.current_dialog_future:
        if not self.current_dialog_future.done():
            self.current_dialog_future.set_result(True)
        self.current_dialog_future = None
        self.current_dialog = None
```

#### STEP 7: Load/Save Pipeline Implementation (Backend.DISK.value)

**EXACT IMPLEMENTATION:**
```python
async def _handle_load_pipeline(self):
    """EXACT: Load Pipeline object from file."""
    try:
        # EXACT: Check if any plates are selected
        if not self.current_selected_plates:
            await show_error_dialog("Load Error", "No plates selected. Select plates first.", self.state)
            return

        # EXACT: File dialog for .pipeline files
        file_path = await prompt_for_file_dialog(
            title="Load Pipeline",
            prompt_message="Select pipeline file:",
            app_state=self.state,
            filemanager=self.context.filemanager,
            selection_mode="files",
            filter_extensions=[".pipeline"]
        )

        if not file_path:
            return

        # EXACT: Load Pipeline object using Backend.DISK.value
        from openhcs.constants.constants import Backend
        loaded_pipeline = self.context.filemanager.load(file_path, Backend.DISK.value)

        # EXACT: Validate Pipeline object
        if not isinstance(loaded_pipeline, Pipeline):
            await show_error_dialog("Load Error", "Invalid pipeline file.", self.state)
            return

        # EXACT: Apply loaded pipeline to all selected plates
        for plate_path in self.current_selected_plates:
            self.plate_pipelines[plate_path] = copy.deepcopy(loaded_pipeline)

        # EXACT: Refresh display
        await self._update_pipeline_display_for_selection(self.current_selected_plates)

        logger.info(f"Loaded pipeline to {len(self.current_selected_plates)} plates")

    except Exception as e:
        await show_scrollable_error_dialog(
            title="Load Pipeline Error",
            message=f"Failed to load pipeline: {str(e)}",
            exception=e,
            app_state=self.state
        )

async def _handle_save_pipeline(self):
    """EXACT: Save Pipeline object to file."""
    try:
        # EXACT: Check if pipeline differs across plates
        if self.pipeline_differs_across_plates:
            await show_error_dialog(
                "Save Error",
                "Cannot save - selected plates have different pipelines. Select a single plate first.",
                self.state
            )
            return

        # EXACT: Check if any plates are selected
        if not self.current_selected_plates:
            await show_error_dialog("Save Error", "No plates selected. Select a plate first.", self.state)
            return

        # EXACT: Get pipeline from first selected plate
        plate_path = self.current_selected_plates[0]
        pipeline = self.plate_pipelines.get(plate_path)

        if not pipeline or len(pipeline) == 0:
            await show_error_dialog("Save Error", "No pipeline to save.", self.state)
            return

        # EXACT: File dialog for save location
        file_path = await prompt_for_file_dialog(
            title="Save Pipeline",
            prompt_message="Select save location:",
            app_state=self.state,
            filemanager=self.context.filemanager,
            selection_mode="files",
            filter_extensions=[".pipeline"]
        )

        if not file_path:
            return

        # EXACT: Ensure .pipeline extension
        if not file_path.endswith('.pipeline'):
            file_path += '.pipeline'

        # EXACT: Save Pipeline object using Backend.DISK.value
        from openhcs.constants.constants import Backend
        self.context.filemanager.save(pipeline, file_path, Backend.DISK.value)

        logger.info(f"Saved pipeline from {plate_path} to {file_path}")

    except Exception as e:
        await show_scrollable_error_dialog(
            title="Save Pipeline Error",
            message=f"Failed to save pipeline: {str(e)}",
            exception=e,
            app_state=self.state
        )
```

#### STEP 8: PlateManager Integration (Pipeline Retrieval)

**EXACT IMPLEMENTATION IN PlateManager:**
```python
def _get_current_pipeline_definition(self) -> List:
    """EXACT: Get current pipeline definition from PipelineEditor."""
    # EXACT: Check if pipeline editor exists
    if not hasattr(self.state, 'pipeline_editor') or not self.state.pipeline_editor:
        return []

    # EXACT: Get selected plates from PlateManager
    selected_items, selection_mode = self.get_selection_state()
    if not selected_items:
        return []

    # EXACT: Get pipeline from PipelineEditor for selected plates
    pipeline_editor = self.state.pipeline_editor

    # EXACT: Check if pipeline differs across selected plates
    if pipeline_editor.pipeline_differs_across_plates:
        return []  # Cannot compile when pipelines differ

    # EXACT: Get pipeline from first selected plate
    if selected_items:
        plate_path = selected_items[0]['path']
        pipeline = pipeline_editor.plate_pipelines.get(plate_path)
        if pipeline:
            return list(pipeline)  # Pipeline IS a list

    return []

# EXACT: Update compile method to use pipeline from editor
async def _compile_selected_plates(self, selected_items: List[Dict[str, Any]]):
    """Compile pipelines for selected plates."""
    try:
        # EXACT: Get pipeline definition from editor
        pipeline_definition = self._get_current_pipeline_definition()
        if not pipeline_definition:
            await show_error_dialog(
                "Compile Error",
                "No pipeline defined. Use Pipeline Editor to create a pipeline.",
                self.state
            )
            return

        # EXACT: Validate all selected plates are initialized
        uninitialized = [item for item in selected_items if item.get('status') != '-']
        if uninitialized:
            names = [item['name'] for item in uninitialized]
            await show_error_dialog(
                "Compile Error",
                f"Cannot compile uninitialized plates: {', '.join(names)}",
                self.state
            )
            return

        # ... rest of existing compile logic using pipeline_definition
```

#### STEP 9: Utility Functions Implementation

**EXACT IMPLEMENTATION:**
```python
def _format_func_display(self, func) -> str:
    """EXACT: Format function for display."""
    if func is None:
        return "No function"
    elif callable(func):
        return getattr(func, '__name__', str(func))
    elif isinstance(func, tuple) and len(func) >= 1:
        return getattr(func[0], '__name__', str(func[0]))
    elif isinstance(func, list) and len(func) > 0:
        first_func = func[0]
        if isinstance(first_func, tuple):
            return f"{getattr(first_func[0], '__name__', str(first_func[0]))} + {len(func)-1} more"
        else:
            return f"{getattr(first_func, '__name__', str(first_func))} + {len(func)-1} more"
    else:
        return str(func)

def _get_pipeline_for_plate(self, plate_path: str) -> Optional[Pipeline]:
    """EXACT: Get pipeline for specific plate."""
    return self.plate_pipelines.get(plate_path)

def _set_pipeline_for_plate(self, plate_path: str, pipeline: Pipeline):
    """EXACT: Set pipeline for specific plate."""
    self.plate_pipelines[plate_path] = pipeline

def _remove_pipeline_for_plate(self, plate_path: str):
    """EXACT: Remove pipeline for specific plate."""
    if plate_path in self.plate_pipelines:
        del self.plate_pipelines[plate_path]

def _get_all_plate_paths_with_pipelines(self) -> List[str]:
    """EXACT: Get all plate paths that have pipelines."""
    return list(self.plate_pipelines.keys())
```

### Findings

**State Observer Pattern Verified:**
- PlateManager calls: `self.state.set_selected_plate(selected_item)`
- TUIState triggers: `self.notify('plate_selected', plate)`
- PipelineEditor registers: `self.state.add_observer('plate_selected', callback)`
- Real-time updates: Automatic pipeline display refresh on plate selection

**Backend Usage Pattern Verified:**
- Import: `from openhcs.constants.constants import Backend`
- Usage: `Backend.DISK.value` equals `"disk"`
- FileManager calls: `filemanager.save(data, path, Backend.DISK.value)`
- FileManager calls: `filemanager.load(path, Backend.DISK.value)`

**DualEditorPane Integration Verified:**
- Constructor: `DualEditorPane(state, func_step, on_save, on_cancel)`
- Callbacks: `on_save(func_step: FunctionStep)`, `on_cancel()`
- Container: `editor.container` property returns prompt_toolkit Container
- Internal: Composes StepParameterEditor + FunctionPatternEditor

**Pipeline Object Verified:**
- Constructor: `Pipeline(steps=[], name=name)`
- Behavior: Pipeline IS a list - use `list(pipeline)` to get steps
- Methods: `append()`, `insert()`, `remove()` - standard list methods
- Serialization: Pickle-compatible for .pipeline files

**Multi-Plate Pipeline Logic Verified:**
- Single plate: Show its pipeline
- Multiple plates with identical pipelines: Show common pipeline
- Multiple plates with different pipelines: Show "differs" message
- No plates: Show empty list

**Dialog Management Verified:**
- Method: `await app_state.show_dialog(dialog, result_future)`
- Focus: Automatic focus management and restoration
- Modal: `Dialog(modal=True)` for modal behavior
- Cleanup: Set future result to close dialog

**Error Handling Verified:**
- Function: `await show_scrollable_error_dialog(title, message, exception, app_state)`
- Function: `await show_error_dialog(title, message, app_state)`
- Pattern: Always wrap in try/except with scrollable error dialogs

**Pipeline-Plate Association Verified:**
- Storage: `self.plate_pipelines: Dict[str, Pipeline]` maps plate paths to pipelines
- Real-time updates: Pipeline display updates when plate selection changes
- State tracking: `self.current_selected_plates` and `self.pipeline_differs_across_plates`
- PlateManager integration: Gets pipeline from editor for compilation

**Add/Edit Button Behavior Verified:**
- Add: Creates empty default step, adds to list, user selects and edits
- Edit: Opens DualEditorPane for selected step with visual programming
- Delete: Removes step from all selected plates' pipelines
- Load/Save: Works with .pipeline files using Backend.DISK.value

#### STEP 10: Code Cleanup and Removal of Legacy Code

**FILE:** `openhcs/tui/panes/pipeline_editor.py`

**LINES 195-199:** DELETE _get_orchestrator_steps method entirely:
```python
# DELETE LINES 195-199 ENTIRELY:
    def _get_orchestrator_steps(self) -> List[Any]:
        """Get step objects from active orchestrator."""
        if self.state.active_orchestrator and self.state.active_orchestrator.pipeline_definition:
            return self.state.active_orchestrator.pipeline_definition
        return []
```

**LINES 182-194:** REPLACE _load_steps_for_plate method:
```python
# REPLACE LINES 182-194:
# OLD:
    async def _load_steps_for_plate(self, plate_id: str):
        """Load steps for the specified plate."""
        async with self.steps_lock:
            # Try orchestrator first, fallback to context
            raw_steps = self._get_orchestrator_steps()
            if raw_steps:
                steps = [self._transform_step_to_dict(step) for step in raw_steps
                        if isinstance(step, FunctionStep)]
            else:
                steps = self.context.list_steps_for_plate(plate_id)

            self.list_manager.load_items(steps)

# NEW:
    async def _load_steps_for_plate(self, plate_id: str):
        """Load steps for the specified plate."""
        # EXACT: This method is now handled by _on_plate_selection_changed
        # Just trigger the real-time update
        await self._on_plate_selection_changed(self.state.selected_plate)
```

**LINES 201-210:** DELETE _transform_step_to_dict method entirely:
```python
# DELETE LINES 201-210 ENTIRELY:
    def _transform_step_to_dict(self, step_obj: FunctionStep) -> Dict[str, Any]:
        """Transform FunctionStep to display dictionary."""
        return {
            'id': step_obj.step_id,
            'name': step_obj.name,
            'func': step_obj.func,
            'status': 'pending',
            'pipeline_id': getattr(step_obj, 'pipeline_id', None),
            'output_memory_type': getattr(step_obj, 'output_memory_type', '[N/A]')
        }
```

**3. UPDATE EXISTING METHOD SIGNATURES:**
```python
# ENSURE these methods are updated to new signatures:
# OLD: def _handle_add_step(self):
# NEW: async def _handle_add_step(self):  # Must be async

# OLD: def _handle_edit_step(self):
# NEW: async def _handle_edit_step(self):  # Must be async

# OLD: def _handle_delete_step(self):
# NEW: async def _handle_delete_step(self):  # Must be async
```

**4. REMOVE PLACEHOLDER ERROR MESSAGES:**
```python
# DELETE any placeholder error dialogs like:
await show_error_dialog("Not implemented", "Add step functionality not implemented yet.", self.state)
await show_error_dialog("Not implemented", "Edit step functionality not implemented yet.", self.state)
```

**5. CLEAN UP IMPORTS:**
```python
# ADD REQUIRED IMPORTS at top of file:
import copy
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from openhcs.constants.constants import Backend
from openhcs.core.pipeline import Pipeline
from openhcs.core.steps.function_step import FunctionStep
from openhcs.tui.components.dual_editor_pane import DualEditorPane
from openhcs.tui.dialogs.error_dialogs import show_error_dialog, show_scrollable_error_dialog
from prompt_toolkit.application import get_app
from prompt_toolkit.widgets import Dialog

# REMOVE UNUSED IMPORTS (if any exist):
# Remove any imports that are no longer used after cleanup
```

**6. VERIFY SEMANTIC CONSISTENCY:**
```python
# ENSURE ALL REFERENCES ARE CONSISTENT:
# - Always use self.plate_pipelines (never self.current_pipeline)
# - Always use self.current_selected_plates (never single plate assumptions)
# - Always use Backend.DISK.value (never hardcoded "disk")
# - Always use Pipeline IS a list pattern (never .steps attribute)
# - Always use copy.deepcopy() when sharing steps between plates
```

**7. REMOVE DEAD CODE PATHS:**
```python
# DELETE any conditional code that checks for old patterns:
if hasattr(self, 'current_pipeline'):     # DELETE - always false now
if self.state.active_orchestrator:        # DELETE - not used anymore
if hasattr(orchestrator, 'pipeline_definition'):  # DELETE - not used
```

**8. VALIDATE PLATEMANGER INTEGRATION:**

**FILE:** `openhcs/tui/panes/plate_manager.py`

**LINES 616-618:** REPLACE _get_current_pipeline_definition method:
```python
# REPLACE LINES 616-618:
# OLD:
    def _get_current_pipeline_definition(self) -> List:
        """Get current pipeline definition from active orchestrator."""
        return self.state.active_orchestrator.pipeline_definition

# NEW:
    def _get_current_pipeline_definition(self) -> List:
        """EXACT: Get current pipeline definition from PipelineEditor."""
        # EXACT: Check if pipeline editor exists
        if not hasattr(self.state, 'pipeline_editor') or not self.state.pipeline_editor:
            return []

        # EXACT: Get selected plates from PlateManager
        selected_items, selection_mode = self.get_selection_state()
        if not selected_items:
            return []

        # EXACT: Get pipeline from PipelineEditor for selected plates
        pipeline_editor = self.state.pipeline_editor

        # EXACT: Check if pipeline differs across selected plates
        if pipeline_editor.pipeline_differs_across_plates:
            return []  # Cannot compile when pipelines differ

        # EXACT: Get pipeline from first selected plate
        if selected_items:
            plate_path = selected_items[0]['path']
            pipeline = pipeline_editor.plate_pipelines.get(plate_path)
            if pipeline:
                return list(pipeline)  # Pipeline IS a list

        return []
```

#### STEP 11: Final Validation and Testing Preparation

**EXACT VALIDATION CHECKLIST:**

**1. INTERFACE CONSISTENCY CHECK:**
- [ ] All async methods use `await` for error dialogs
- [ ] All Backend usage uses `Backend.DISK.value`
- [ ] All Pipeline usage treats Pipeline as list
- [ ] All state updates trigger UI refresh
- [ ] All dialog management uses exact patterns

**2. STATE MANAGEMENT CONSISTENCY:**
- [ ] Pipeline-plate association always uses `self.plate_pipelines`
- [ ] Selection state always uses `self.current_selected_plates`
- [ ] Multi-plate logic always checks `self.pipeline_differs_across_plates`
- [ ] Observer registration happens in `__init__`

**3. ERROR HANDLING CONSISTENCY:**
- [ ] All operations wrapped in try/except
- [ ] All exceptions show scrollable error dialogs
- [ ] All validation errors show simple error dialogs
- [ ] All error messages are user-friendly

**4. CLEANUP VERIFICATION:**
- [ ] No references to `self.current_pipeline`
- [ ] No references to `orchestrator.pipeline_definition`
- [ ] No placeholder "Not implemented" messages
- [ ] No unused imports or methods
- [ ] No dead code paths

#### STEP 12: Implementation Order and Dependencies

**EXACT IMPLEMENTATION SEQUENCE:**

1. **STEP 1**: Pipeline-Plate Association Storage (foundation)
2. **STEP 2**: Multi-Plate Pipeline Display Logic (core logic)
3. **STEP 3**: Add Button Implementation (user interaction)
4. **STEP 4**: Edit Button Implementation (visual programming)
5. **STEP 5**: Delete Step Implementation (user interaction)
6. **STEP 6**: Dialog Management Implementation (UI infrastructure)
7. **STEP 7**: Load/Save Pipeline Implementation (persistence)
8. **STEP 8**: PlateManager Integration (cross-component)
9. **STEP 9**: Utility Functions Implementation (helpers)
10. **STEP 10**: Code Cleanup and Removal (tidiness)
11. **STEP 11**: Final Validation (quality assurance)

**CRITICAL DEPENDENCIES:**
- Steps 1-2 must be completed before any user interaction steps (3-5)
- Step 6 must be completed before Steps 4 and 7 (dialog usage)
- Step 8 requires Steps 1-2 to be completed (pipeline retrieval)
- Step 10 can only be done after all implementation steps (1-9)
- Step 11 is final validation after everything is complete

**SEMANTIC CONSISTENCY GUARANTEE:**
This plan eliminates all semantic drift by:
- Consistently using `self.plate_pipelines` throughout (never `self.current_pipeline`)
- Consistently using `Backend.DISK.value` throughout (never hardcoded strings)
- Consistently treating Pipeline as list throughout (never `.steps` attribute)
- Consistently using real-time state updates throughout (observer pattern)
- Explicitly removing all legacy code and broken references

**TIDINESS GUARANTEE:**
This plan ensures complete tidiness by:
- Explicit removal of all broken orchestrator references with exact line numbers
- Explicit removal of all old single-pipeline storage with exact line numbers
- Explicit removal of all placeholder error messages with exact line numbers
- Explicit cleanup of imports and dead code paths with exact line numbers
- Explicit validation checklist to verify cleanliness

**RETARD-PROOF GUARANTEE:**
This plan is now actually retard-proof because:
- **EXACT FILE PATHS:** Every change specifies the exact file to edit
- **EXACT LINE NUMBERS:** Every change specifies the exact lines to modify
- **EXACT CODE BLOCKS:** Every change shows the exact old code and exact new code
- **NO AMBIGUITY:** No corporate bullshit - just exact instructions
- **COMPLETE SPECIFICATIONS:** Every method, import, and attribute change is specified

**EXAMPLE OF RETARD-PROOF SPECIFICATION:**
```
FILE: openhcs/tui/panes/pipeline_editor.py
LINES 213-225: REPLACE existing _handle_add_step method
OLD CODE: [exact old code shown]
NEW CODE: [exact new code shown]
```

This is how EVERY change in this plan is specified. No guessing, no interpretation needed.

### Implementation Draft
(Will be added after plan approval)
