# plan_04_plate_manager_selection_behavior.md
## Component: PlateManager Selection-Based Button Behavior

### Objective
Implement mathematically precise selection-based button behavior for PlateManager operations. Each button (Del, Edit, Init, Compile, Run) must behave differently based on whether checkboxes are selected or only cursor highlighting is active, following the two-phase execution model and orchestrator lifecycle.

### Plan

#### Step 1: Understand Current Architecture and Requirements

**OpenHCS Two-Phase Execution Model:**
1. **PHASE 1: Compilation** → `orchestrator.compile_pipelines()` → Creates frozen ProcessingContexts
2. **PHASE 2: Execution** → `orchestrator.execute_compiled_plate()` → Runs against frozen contexts

**Orchestrator Lifecycle:**
1. **Created** (`?` gray): PipelineOrchestrator instantiated from plate path
2. **Initialized** (`-` yellow): `orchestrator.initialize()` called (discovers wells, setup workspace)
3. **Compiled** (`o` green): `orchestrator.compile_pipelines()` called (frozen ProcessingContexts created)
4. **Running/Error** (`!` red): `orchestrator.execute_compiled_plate()` called

**Selection Behavior Requirements:**
- **No checkboxes selected**: Operation applies to cursor-highlighted item only
- **Checkboxes selected**: Operation applies to all checked items, ignoring cursor position

#### Step 2: Define Selection State Logic (BULLETPROOF)

**ALL Possible Selection States:**
1. **Empty list**: No items exist → All operations should show "No plates available"
2. **No selection, no checks**: Cursor not on item, no checkboxes currently checked → Operations apply to ALL items
3. **Cursor selection, no checks**: Cursor on item, no checkboxes currently checked → Operations apply to highlighted item only
4. **Checkboxes checked**: One or more checkboxes currently checked → Operations apply to checked items only, ignore cursor position

**Selection State Determination (BULLETPROOF):**
```python
def get_selection_state(self) -> Tuple[List[Dict], str]:
    """Get current selection state with bulletproof validation.

    Returns:
        Tuple[List[Dict], str]: (selected_items, selection_mode)
        - selected_items: List of item dictionaries to operate on
        - selection_mode: "empty" | "all" | "cursor" | "checkbox"
    """
    # VALIDATION 1: Check if list exists and has items
    if not hasattr(self, 'list_manager') or not self.list_manager:
        raise RuntimeError("list_manager not initialized")
    if not hasattr(self.list_manager, 'model') or not self.list_manager.model:
        raise RuntimeError("list_manager.model not initialized")

    all_items = self.list_manager.model.get_all_items()
    if not all_items:
        return [], "empty"

    # VALIDATION 2: Check if any checkboxes are currently checked (highest priority)
    checked_items = self.list_manager.model.get_checked_items()
    if checked_items:
        # Validate checked items are still in the list
        valid_checked = [item for item in checked_items if item in all_items]
        if not valid_checked:
            # Checked items were removed - clear checks and fall through to cursor/all logic
            self.list_manager.model.clear_all_checks()
        else:
            # At least one checkbox is currently checked - use checkbox mode
            return valid_checked, "checkbox"

    # VALIDATION 3: Check for cursor selection
    highlighted_item = self.list_manager.get_selected_item()
    if highlighted_item and highlighted_item in all_items:
        return [highlighted_item], "cursor"

    # VALIDATION 4: No specific selection - default to all items
    return all_items, "all"

def get_operation_description(self, selected_items: List[Dict], selection_mode: str, operation: str) -> str:
    """Generate human-readable description of what will be operated on."""
    count = len(selected_items)
    if selection_mode == "empty":
        return f"No plates available for {operation}"
    elif selection_mode == "all":
        return f"{operation.title()} ALL {count} plates"
    elif selection_mode == "cursor":
        item_name = selected_items[0].get('name', 'Unknown')
        return f"{operation.title()} highlighted plate: {item_name}"
    elif selection_mode == "checkbox":
        if count == 1:
            item_name = selected_items[0].get('name', 'Unknown')
            return f"{operation.title()} checked plate: {item_name}"
        else:
            return f"{operation.title()} {count} checked plates"
    else:
        raise ValueError(f"Unknown selection_mode: {selection_mode}")
```

#### Step 3: Implement Button Behavior Logic (MATHEMATICALLY PRECISE)

**Del Button Logic:**
```python
async def _handle_delete_plates(self):
    """Delete plates based on selection state."""
    selected_items, is_checkbox_mode = self.get_selection_state()
    
    if not selected_items:
        # No selection - show error dialog
        await show_error_dialog("No plates selected for deletion")
        return
    
    # Confirm deletion
    item_names = [item['name'] for item in selected_items]
    mode_text = "checked" if is_checkbox_mode else "highlighted"
    confirm_msg = f"Delete {len(selected_items)} {mode_text} plate(s):\n" + "\n".join(item_names)
    
    if await show_confirmation_dialog(confirm_msg):
        # Remove items from model
        for item in selected_items:
            self.list_manager.model.remove_item_by_data(item)
        
        # Clear checkboxes if in checkbox mode
        if is_checkbox_mode:
            self.list_manager.model.clear_all_checks()
```

**Edit Button Logic:**
```python
async def _handle_edit_plate(self):
    """Edit plate configuration based on selection state."""
    selected_items, is_checkbox_mode = self.get_selection_state()
    
    if not selected_items:
        await show_error_dialog("No plates selected for editing")
        return
    
    if is_checkbox_mode:
        # Multi-edit mode: apply config to all checked plates
        await self._edit_multiple_plate_configs(selected_items)
    else:
        # Single edit mode: edit highlighted plate only
        await self._edit_single_plate_config(selected_items[0])
```

**Init Button Logic:**
```python
async def _handle_initialize_plates(self):
    """Initialize plates based on selection state."""
    selected_items, is_checkbox_mode = self.get_selection_state()
    
    if not selected_items:
        # No selection: initialize ALL plates in list
        all_items = self.list_manager.model.get_all_items()
        if not all_items:
            await show_error_dialog("No plates available for initialization")
            return
        selected_items = all_items
        mode_text = "all"
    else:
        mode_text = "checked" if is_checkbox_mode else "highlighted"
    
    # Confirm initialization
    confirm_msg = f"Initialize {len(selected_items)} {mode_text} plate(s)?"
    if await show_confirmation_dialog(confirm_msg):
        await self._initialize_selected_plates(selected_items)
```

**Compile Button Logic:**
```python
async def _handle_compile_plates(self):
    """Compile plates based on selection state."""
    selected_items, is_checkbox_mode = self.get_selection_state()
    
    if not selected_items:
        # No selection: compile ALL plates in list
        all_items = self.list_manager.model.get_all_items()
        selected_items = all_items
        mode_text = "all"
    else:
        mode_text = "checked" if is_checkbox_mode else "highlighted"
    
    # Validate all selected plates are initialized
    uninitialized = [item for item in selected_items if item.get('status') == '?']
    if uninitialized:
        names = [item['name'] for item in uninitialized]
        await show_error_dialog(f"Cannot compile uninitialized plates:\n" + "\n".join(names))
        return
    
    await self._compile_selected_plates(selected_items)
```

**Run Button Logic:**
```python
async def _handle_run_plates(self):
    """Run plates based on selection state."""
    selected_items, is_checkbox_mode = self.get_selection_state()
    
    if not selected_items:
        # No selection: run ALL plates in list
        all_items = self.list_manager.model.get_all_items()
        selected_items = all_items
        mode_text = "all"
    else:
        mode_text = "checked" if is_checkbox_mode else "highlighted"
    
    # Validate all selected plates are compiled
    uncompiled = [item for item in selected_items if item.get('status') != 'o']
    if uncompiled:
        names = [item['name'] for item in uncompiled]
        await show_error_dialog(f"Cannot run uncompiled plates:\n" + "\n".join(names))
        return
    
    await self._run_selected_plates(selected_items)
```

#### Step 4: Implement Missing Model Methods (CRITICAL GAPS)

**MISSING METHODS that must be implemented in ListModel:**
```python
def get_checked_items(self) -> List[Dict]:
    """Get all checked items. MUST BE IMPLEMENTED."""
    if not hasattr(self, '_checked_indices'):
        self._checked_indices = set()
    return [self.items[i] for i in self._checked_indices if i < len(self.items)]

def get_all_items(self) -> List[Dict]:
    """Get all items in the list. MUST BE IMPLEMENTED."""
    return self.items.copy() if hasattr(self, 'items') and self.items else []

def clear_all_checks(self) -> None:
    """Clear all checkbox selections. MUST BE IMPLEMENTED."""
    if hasattr(self, '_checked_indices'):
        self._checked_indices.clear()
        self.notify_observers()

def remove_item_by_data(self, item_data: Dict) -> bool:
    """Remove item by data reference. MUST BE IMPLEMENTED."""
    try:
        index = self.items.index(item_data)
        return self.remove_item(index)
    except ValueError:
        return False
```

**MISSING METHODS that must be implemented in PlateManagerPane:**
```python
def _get_orchestrator_for_item(self, item: Dict) -> PipelineOrchestrator:
    """Get or create orchestrator for plate item. MUST BE IMPLEMENTED."""
    # Check if orchestrator already exists
    if 'orchestrator' in item and item['orchestrator']:
        return item['orchestrator']

    # Create new orchestrator
    plate_path = item.get('path')
    if not plate_path:
        raise ValueError(f"Item {item.get('name', 'Unknown')} has no path")

    # Get global config from state
    global_config = self.state.get_global_config()
    if not global_config:
        raise ValueError("No global configuration available")

    # Create orchestrator
    orchestrator = PipelineOrchestrator(plate_path, global_config, self.filemanager)

    # Store in item for reuse
    item['orchestrator'] = orchestrator

    return orchestrator

async def show_error_dialog(self, message: str) -> None:
    """Show error dialog. MUST BE IMPLEMENTED."""
    # Implementation depends on existing dialog system
    pass

async def show_confirmation_dialog(self, message: str) -> bool:
    """Show confirmation dialog. MUST BE IMPLEMENTED."""
    # Implementation depends on existing dialog system
    return False
```

#### Step 5: Implement Orchestrator Operations (LIFECYCLE COMPLIANT)

**Initialization Operation:**
```python
async def _initialize_selected_plates(self, selected_items: List[Dict]):
    """Initialize orchestrators for selected plates."""
    for item in selected_items:
        try:
            # Get or create orchestrator
            orchestrator = self._get_orchestrator_for_item(item)
            
            # Call orchestrator.initialize()
            orchestrator.initialize()
            
            # Update status to initialized
            item['status'] = '-'  # Yellow: initialized but not compiled
            
        except Exception as e:
            item['status'] = '!'  # Red: error
            logger.error(f"Failed to initialize {item['name']}: {e}")
    
    # Refresh UI
    self.list_manager.model.notify_observers()
```

**Compilation Operation:**
```python
async def _compile_selected_plates(self, selected_items: List[Dict]):
    """Compile pipelines for selected plates."""
    for item in selected_items:
        try:
            # Get orchestrator (must be initialized)
            orchestrator = self._get_orchestrator_for_item(item)
            
            # Get current pipeline definition from state
            pipeline_definition = self.state.get_current_pipeline()
            if not pipeline_definition:
                raise ValueError("No pipeline defined")
            
            # Call orchestrator.compile_pipelines()
            compiled_contexts = orchestrator.compile_pipelines(pipeline_definition)
            
            # Store compiled contexts in item for execution
            item['compiled_contexts'] = compiled_contexts
            item['status'] = 'o'  # Green: compiled and ready
            
        except Exception as e:
            item['status'] = '!'  # Red: error
            logger.error(f"Failed to compile {item['name']}: {e}")
    
    # Refresh UI
    self.list_manager.model.notify_observers()
```

**Execution Operation:**
```python
async def _run_selected_plates(self, selected_items: List[Dict]):
    """Execute compiled plates."""
    for item in selected_items:
        try:
            # Get orchestrator and compiled contexts
            orchestrator = self._get_orchestrator_for_item(item)
            compiled_contexts = item.get('compiled_contexts')
            
            if not compiled_contexts:
                raise ValueError("No compiled contexts found")
            
            # Get pipeline definition
            pipeline_definition = self.state.get_current_pipeline()
            
            # Set status to running
            item['status'] = '!'  # Red: running
            self.list_manager.model.notify_observers()
            
            # Call orchestrator.execute_compiled_plate()
            results = orchestrator.execute_compiled_plate(
                pipeline_definition, 
                compiled_contexts
            )
            
            # Store results and update status
            item['execution_results'] = results
            item['status'] = 'o'  # Green: completed successfully
            
        except Exception as e:
            item['status'] = '!'  # Red: error
            logger.error(f"Failed to run {item['name']}: {e}")
    
    # Refresh UI
    self.list_manager.model.notify_observers()
```

#### Step 6: Comprehensive Error Handling and Validation (BULLETPROOF)

**ALL Possible Error Conditions:**
1. **Invalid orchestrator state**: Calling operations out of order
2. **Missing dependencies**: Pipeline not defined, global config missing
3. **File system errors**: Plate path doesn't exist, permission issues
4. **Compilation failures**: Invalid pipeline, missing functions
5. **Execution failures**: GPU errors, out of memory, processing errors
6. **UI state corruption**: Items removed during operation, invalid selections

**Error Handling Strategy:**
```python
async def _safe_operation_wrapper(self, operation_name: str, operation_func, selected_items: List[Dict], selection_mode: str):
    """Bulletproof wrapper for all plate operations."""
    try:
        # Pre-operation validation
        if not selected_items:
            await self.show_error_dialog(f"No plates selected for {operation_name}")
            return False

        # Show progress dialog for long operations
        progress_msg = self.get_operation_description(selected_items, selection_mode, operation_name)
        # TODO: Implement progress dialog

        # Execute operation
        await operation_func(selected_items)

        # Post-operation cleanup
        if selection_mode == "checkbox":
            self.list_manager.model.clear_all_checks()

        # Refresh UI
        self.list_manager.model.notify_observers()

        return True

    except Exception as e:
        # Log error
        logger.error(f"{operation_name} failed: {e}")

        # Show user-friendly error
        await self.show_error_dialog(f"{operation_name} failed:\n{str(e)}")

        # Reset any corrupted state
        self._reset_corrupted_items(selected_items)

        return False

def _reset_corrupted_items(self, items: List[Dict]):
    """Reset items that may be in corrupted state."""
    for item in items:
        # Remove any partial state
        item.pop('compiled_contexts', None)
        item.pop('execution_results', None)

        # Reset status to last known good state
        if 'orchestrator' in item and item['orchestrator']:
            try:
                if item['orchestrator'].is_initialized():
                    item['status'] = '-'  # Initialized
                else:
                    item['status'] = '?'  # Created but not initialized
            except:
                item['status'] = '?'  # Unknown state, reset to created
        else:
            item['status'] = '?'  # No orchestrator, reset to created
```

**Validation Requirements (MATHEMATICAL PRECISION):**
1. **State Transitions**: Only allow valid state transitions (? → - → o → !)
2. **Dependency Validation**: Check all prerequisites before operations
3. **Data Integrity**: Validate all data structures before use
4. **Resource Management**: Proper cleanup of orchestrators and contexts
5. **UI Consistency**: Ensure UI state matches actual orchestrator state

### Findings

#### Cross-Validation Results
1. **Orchestrator Lifecycle**: Confirmed three-phase lifecycle (initialize → compile → execute)
2. **ProcessingContext**: Frozen after compilation, contains step_plans and execution state
3. **Two-Phase Model**: Compile-all creates frozen contexts, execute-all runs against them
4. **Status Progression**: `?` → `-` → `o` → `!` matches orchestrator lifecycle exactly
5. **Selection Model**: ListView supports multi-select but MISSING critical methods

#### Critical Implementation Gaps Identified
1. **ListModel Methods**: `get_checked_items()`, `get_all_items()`, `clear_all_checks()` NOT IMPLEMENTED
2. **Dialog System**: Error and confirmation dialogs need implementation
3. **Orchestrator Management**: No existing pattern for storing orchestrators in items
4. **State Management**: TUIState methods for pipeline and config access undefined
5. **Progress Feedback**: No progress indication for long-running operations

#### Architecture Validation
- **Direct Orchestrator Calls**: TUI uses direct `orchestrator.method()` calls (no MVC layers)
- **VFS Compliance**: All operations must use FileManager abstraction
- **Error Handling**: Comprehensive user feedback through dialog system (TO BE IMPLEMENTED)
- **State Management**: Pipeline definition stored in TUIState, accessed by PlateManager (TO BE VERIFIED)

#### Implementation Requirements (BULLETPROOF)
1. **Selection State Logic**: Bulletproof checkbox vs cursor selection detection ✓
2. **Missing Model Methods**: Implement all required ListModel methods ✓
3. **Orchestrator Management**: Create/store orchestrators per plate item ✓
4. **Status Tracking**: Update visual status indicators based on lifecycle phase ✓
5. **Error Handling**: User-friendly dialogs for all error conditions ✓
6. **Context Storage**: Store compiled ProcessingContexts in item data for execution ✓
7. **Validation Logic**: Comprehensive pre/post operation validation ✓

### Implementation Draft

*Implementation will be added after plan approval via smell loop*
