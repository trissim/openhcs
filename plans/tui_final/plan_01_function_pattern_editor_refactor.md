# plan_01_function_pattern_editor_refactor.md
## Component: Function Pattern Editor Modular Refactoring

### Objective
Refactor the 970-line FunctionPatternEditor god class into clean, modular components following the established TUI architecture patterns. Transform it from a monolithic class doing everything into a composition of focused, single-responsibility components that match the quality of ListManagerPane, ParameterEditor, and DualEditorPane.

### Plan

#### 1. **PatternDataManager** (Pure Data Operations)
**Responsibility**: Pattern data structure operations and transformations
**Location**: `openhcs/tui/services/pattern_data_manager.py`
**Size**: ~150 lines (MAX 200 LOC to prevent bloat)

**Methods**:
- `clone_pattern(pattern)` - Deep cloning with callable preservation
- `convert_list_to_dict(pattern)` - List→Dict conversion with None key for unnamed groups (ORDER DETERMINISTIC)
- `convert_dict_to_list(pattern)` - Dict→List conversion when only None key remains (ORDER DETERMINISTIC)
- `extract_func_and_kwargs(func_item)` - Parse (func, kwargs) tuples
- `validate_pattern_structure(pattern)` - Basic structural validation
- `get_current_functions(pattern, key, is_dict)` - Extract function list for current context

**Key Design**:
- Pure functions, no UI dependencies
- **Order Determinism**: List↔Dict conversions preserve order for stable diffs across saves
- Handles None key semantics for unnamed structural groups
- Immutable operations (always returns new objects)
- No state, just data transformations

**CRITICAL IMPLEMENTATION DETAILS**:
```python
# EXACT method signatures to prevent drift:
def clone_pattern(pattern: Union[List, Dict]) -> Union[List, Dict]:
    """Deep clone preserving callable references exactly"""

def convert_list_to_dict(pattern: List, preserve_order: bool = True) -> Dict:
    """MUST use {None: pattern} for unnamed groups - exact key semantics"""

def convert_dict_to_list(pattern: Dict) -> List:
    """ONLY convert if single None key exists, preserve original order"""

def extract_func_and_kwargs(func_item) -> Tuple[Optional[Callable], Dict]:
    """Handle (func, kwargs) tuples AND bare callables - exact current logic"""
```

#### 2. **FunctionRegistryService** (Function Discovery & Metadata)
**Responsibility**: Function registry integration and metadata extraction
**Location**: `openhcs/tui/services/function_registry_service.py`
**Size**: ~100 lines (MAX 200 LOC to prevent bloat)

**INTEGRATION WITH EXISTING**: Extends existing `openhcs.processing.func_registry` functionality:
- **REUSE**: `openhcs.processing.func_registry.get_function_info()` - Basic metadata
- **REUSE**: `openhcs.processing.func_registry.get_functions_by_memory_type()` - Function lists
- **EXTEND**: `get_enhanced_function_metadata(func)` - Adds validation + special_inputs/outputs (renamed to avoid collision)

**Methods**:
- `get_functions_by_backend()` - Group FUNC_REGISTRY functions by backend
- `get_enhanced_function_metadata(func)` - Extract metadata with validation (RENAMED to avoid import collision)
- `create_dropdown_options(functions_by_backend)` - Format for GroupedDropdown
- `find_default_function()` - Get first available function for new items

**Key Design**:
- **Stateless service class** - No state accumulation, pure functions only
- Integrates with existing FUNC_REGISTRY
- Export only enhanced function from this service (prevent import confusion)
- Provides clean interface for UI components

**CRITICAL IMPLEMENTATION DETAILS**:
```python
# EXACT function signature to prevent naming collision:
def get_enhanced_function_metadata(func: Callable) -> Dict[str, Any]:
    """Enhanced version with validation + special_inputs/outputs

    MUST return exact same structure as current get_function_info():
    {
        'name': str,
        'backend': str,
        'input_memory_type': str,
        'output_memory_type': str,
        'special_inputs': List[str],
        'special_outputs': List[str]
    }
    """

# EXACT backend grouping logic - preserve current behavior:
def get_functions_by_backend() -> Dict[str, List[Tuple[Callable, str]]]:
    """Group FUNC_REGISTRY by backend, include memory types in display name"""
    # MUST preserve: f"{func.__name__} ({input_type} → {output_type})"
```

#### 3. **PatternFileService** (File I/O Operations)
**Responsibility**: Load/save .func files and external editor integration
**Location**: `openhcs/tui/services/pattern_file_service.py`
**Size**: ~120 lines

**INTEGRATION WITH EXISTING**: Reuses existing services, adds .func-specific functionality:
- **REUSE**: `openhcs.tui.services.external_editor_service.ExternalEditorService` - Vim integration
- **REUSE**: `openhcs.tui.utils.dialog_helpers.prompt_for_file_dialog` - File dialogs
- **EXTEND**: .func file loading/saving with pickle and validation

**CRITICAL BUG CONFIRMED**: FunctionPatternEditor calls `prompt_for_path_dialog` (lines 910, 955) which doesn't exist. Only `prompt_for_file_dialog` and `prompt_for_multi_folder_dialog` exist in dialog_helpers.py. Need to create `prompt_for_save_file_dialog` with text input for file saving.

**Methods**:
- `load_pattern_from_file(file_path)` - Load and validate .func files (async with run_in_executor)
- `save_pattern_to_file(pattern, file_path)` - Save patterns with pickle (async with run_in_executor)
- `edit_pattern_externally(pattern, state)` - Vim integration via ExternalEditorService
- `prompt_for_func_file_path(title, message, state)` - Wrapper around existing `prompt_for_file_dialog`

**Key Design**:
- **Async/Sync Safety**: Wrap all file I/O in `run_in_executor()` to prevent event loop deadlocks
- Reuses existing ExternalEditorService
- Fixes the broken dialog import/usage
- Proper error handling and validation
- Clean separation from UI concerns

**CRITICAL IMPLEMENTATION DETAILS**:
```python
# FIXED async pattern using unified task manager:
async def load_pattern_from_file(file_path: Path) -> Union[List, Dict]:
    """Use unified task manager for file I/O"""
    from openhcs.tui.utils.unified_task_manager import get_task_manager

    def _sync_load_pattern(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    # Use asyncio.get_running_loop() instead of deprecated get_event_loop()
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _sync_load_pattern, file_path)

async def save_pattern_to_file(pattern: Union[List, Dict], file_path: Path) -> None:
    """Use unified task manager for file I/O"""
    def _sync_save_pattern(pattern_data, path):
        with open(path, "wb") as f:
            pickle.dump(pattern_data, f)

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _sync_save_pattern, pattern, file_path)

# NEW DIALOG FUNCTION NEEDED - Add to dialog_helpers.py:
async def prompt_for_save_file_dialog(title: str, message: str, state: Any,
                                     initial_path: str = "", filemanager=None) -> Optional[str]:
    """Dialog with text input for typing file save path"""
    from prompt_toolkit.widgets import TextArea

    text_area = TextArea(text=initial_path, multiline=False)

    dialog_body = HSplit([
        Label(message),
        Label(""),
        Label("File path:"),
        text_area,
    ])

    future = asyncio.Future()

    def accept_path():
        path = text_area.text.strip()
        if path:
            future.set_result(path)
        else:
            future.set_result(None)

    dialog = Dialog(
        title=title,
        body=dialog_body,
        buttons=[
            Button("Save", handler=accept_path),
            Button("Cancel", handler=lambda: future.set_result(None)),
        ],
        modal=True
    )

    await state.show_dialog(dialog, result_future=future)
    return await future

# FIXED dialog wrapper - use new save dialog for file saving:
async def prompt_for_func_file_path(title: str, message: str, state: Any, filemanager) -> Optional[str]:
    """Wrapper for file saving with text input"""
    return await prompt_for_save_file_dialog(
        title=title,
        message=message,
        state=state,
        initial_path="",
        filemanager=filemanager
    )
```

#### 4. **FunctionListManager** (Function List UI Component)
**Responsibility**: Display and manage the list of functions in a pattern
**Location**: `openhcs/tui/components/function_list_manager.py`
**Size**: ~200 lines

**ARCHITECTURE DECISION**: Use pure composition pattern, NOT inheritance from ListManagerPane!

**RATIONALE**: After examining the codebase, ListManagerPane is tightly coupled to specific list operations. FunctionListManager has different requirements (parameter editing, function selection). Pure composition provides cleaner separation.

**Components**:
- `FunctionListModel` - Pure data model for function list (standalone class)
- `FunctionListView` - UI view that observes model (standalone class)
- `FunctionListManager` - Coordinator using composition (standalone class)

**Key Design**:
- **Pure composition pattern** - No inheritance from Container or ListManagerPane
- Composes ParameterEditor for each function's parameters
- Uses GroupedDropdown for function selection
- Clean event handling with observer pattern
- **COMPOSITION ONLY**: All UI elements composed, not inherited
- **MAX 200 LOC per component** to prevent god class reformation

**CRITICAL IMPLEMENTATION DETAILS**:
```python
# FIXED composition pattern - standalone classes:
class FunctionListModel:
    """Standalone data model for function list"""
    def __init__(self):
        self.functions: List[Tuple[Callable, Dict]] = []
        self.observers: List[Callable] = []

    def update_function_at_index(self, index: int, func: Callable, kwargs: Dict) -> None:
        """Update function while preserving (func, kwargs) tuple structure"""

class FunctionListView:
    """Standalone UI view that observes model"""
    def __init__(self, model: FunctionListModel):
        self.model = model
        self.container = HSplit([])  # Composed container

    def _create_item_widget(self, index: int, item_data: Dict) -> Container:
        """Compose ParameterEditor + GroupedDropdown inside Container"""

class FunctionListManager:
    """Standalone coordinator using pure composition"""
    def __init__(self):
        self.model = FunctionListModel()
        self.view = FunctionListView(self.model)
        # Pure composition - no inheritance

# FIXED parameter editor integration using unified task manager:
def _create_function_item_with_params(self, index: int, func: Callable, kwargs: Dict):
    """Create ParameterEditor for each function with FIXED callback pattern:

    OLD BROKEN: get_app().create_background_task(self._handle_param_change(...))
    NEW FIXED: get_task_manager().fire_and_forget(self._handle_param_change(...), 'param_change')
    """
```

#### 5. **PatternKeySelector** (Key Management UI Component)
**Responsibility**: Dict key selection and management UI
**Location**: `openhcs/tui/components/pattern_key_selector.py`
**Size**: ~100 lines (MAX 200 LOC to prevent bloat)

**Methods**:
- `build_key_selector(pattern, current_key, is_dict)` - Key selection UI
- `create_key_dropdown(keys, current_key)` - Dropdown for key selection
- `create_key_management_buttons()` - Add/Remove key buttons

**Key Design**:
- Displays "Unnamed" for None key to avoid user confusion
- Clean button-based key management
- Follows established UI patterns

**CRITICAL IMPLEMENTATION DETAILS**:
```python
# EXACT None key display logic - preserve current behavior:
def _create_key_dropdown_options(self, pattern: Dict) -> List[Tuple]:
    """MUST display 'Unnamed' for None key while preserving None in data model"""
    display_keys = []
    for k in pattern.keys():
        if k is None:
            display_keys.append((None, "Unnamed"))  # EXACT current logic
        else:
            display_keys.append((k, str(k)))
    return display_keys

# EXACT key conversion logic - preserve structural truth:
def _should_convert_to_list(self, pattern: Dict) -> bool:
    """MUST convert back to list if only None key remains"""
    return list(pattern.keys()) == [None]  # EXACT current condition
```

#### 6. **FunctionPatternView** (Main Coordinator - Replaces God Class)
**Responsibility**: UI coordination and event handling
**Location**: `openhcs/tui/editors/function_pattern_editor.py` (refactored)
**Size**: ~150 lines (MAX 200 LOC to prevent bloat)

**Methods**:
- `__init__()` - Compose all sub-components
- `get_pattern()` - Return current pattern state
- `_on_pattern_change()` - Handle changes from sub-components
- `_refresh_ui()` - Coordinate UI updates

**Key Design**:
- Composition over inheritance
- Delegates to specialized components
- Minimal state (just coordination)
- Clean callback-based communication

**CRITICAL IMPLEMENTATION DETAILS**:
```python
# EXACT interface preservation - NO CHANGES to public API:
class FunctionPatternEditor:
    def __init__(self, state: Any, initial_pattern: Union[List, Dict, None] = None,
                 change_callback: Optional[Callable] = None):
        """EXACT same constructor signature - no breaking changes"""

    def get_pattern(self) -> Union[List, Dict]:
        """EXACT same return type and behavior"""

    @property
    def container(self) -> Container:
        """EXACT same property - DualEditorPane depends on this"""

    def _notify_change(self):
        """EXACT same callback pattern"""
        if self.change_callback:
            self.change_callback()

# EXACT state management - preserve original vs current pattern tracking:
self.original_pattern = self._clone_pattern(initial_pattern or [])
self.current_pattern = self._clone_pattern(self.original_pattern)

# EXACT component composition - delegate to services:
self.pattern_data_manager = PatternDataManager()
self.registry_service = FunctionRegistryService()
self.file_service = PatternFileService()
```

### Critical Issues Found During Investigation

1. **BROKEN DIALOG CALLS**: FunctionPatternEditor calls `prompt_for_path_dialog` (lines 910, 955) which doesn't exist - need to create `prompt_for_save_file_dialog` with text input
2. **BROKEN ASYNC PATTERNS**: Uses deprecated `get_app().create_background_task()` instead of unified task manager (lines 323, 433, etc.)
3. **ARCHITECTURE MISMATCH**: 980-line god class doesn't follow established ListManagerPane/ParameterEditor patterns
4. **DIRECT FILE I/O**: Uses direct `open()` calls (lines 926, 969) instead of FileManager abstraction
5. **SPINNER COMPONENT**: Still uses deprecated `app.create_background_task()` (line 68) - should use unified task manager

### Implementation Strategy

#### Phase 1: Fix Critical Bugs & Extract Pure Data Operations
1. **CREATE MISSING DIALOG**: Add `prompt_for_save_file_dialog` to dialog_helpers.py with text input for file saving
2. **FIX BROKEN DIALOG CALLS**: Replace `prompt_for_path_dialog` with `prompt_for_save_file_dialog` (lines 910, 955)
3. **FIX ASYNC PATTERNS**: Replace `get_app().create_background_task()` with unified task manager (lines 323, 433, etc.)
4. **FIX FILE I/O**: Replace direct `open()` calls with FileManager abstraction (lines 926, 969)
5. **FIX SPINNER COMPONENT**: Update Spinner to use unified task manager instead of deprecated async pattern
6. Create `PatternDataManager` with all pattern manipulation logic (order deterministic conversions)
7. Create `FunctionRegistryService` with enhanced function info (renamed to avoid collision)
8. Create `PatternFileService` with file operations (FileManager + unified task manager)
9. **STATIC ANALYSIS VERIFICATION**: Comprehensive interface compatibility and data flow validation

#### Phase 2: Create UI Components Following Established Patterns
1. Create `PatternKeySelector` component (follows StepParameterEditor pattern)
2. Create `FunctionListManager` component (follows ListManagerPane MVC pattern)
3. **STATIC ANALYSIS**: Verify component interfaces and composition patterns

#### Phase 3: Refactor Main Class
1. Refactor `FunctionPatternEditor` to use new components
2. Update `DualEditorPane` integration (should be seamless)
3. **STATIC ANALYSIS**: Verify interface preservation and callback chain integrity

#### Phase 4: Cleanup & Validation
1. **CLEANUP STALE ARTIFACTS**: Remove .pyc files for deleted services (pattern_editing_service, command_service, etc.)
2. **UPDATE BROKEN TESTS**: Fix test imports that reference deleted services
3. Remove old god class methods
4. Update imports and dependencies
5. Verify no regressions
6. Confirm all dialog functions work correctly

### Architectural Benefits

**Clean Separation of Concerns**:
- Data operations separate from UI
- File I/O separate from business logic
- Each component has single responsibility

**Static Verifiability**:
- Pure functions with clear contracts and invariants
- UI components with explicit interface definitions
- Clear data flow patterns for comprehensive analysis

**Maintainability**:
- Small, focused files (~100-200 lines each)
- Clear dependencies and interfaces
- Follows established TUI patterns

**Reusability**:
- PatternDataManager can be used elsewhere
- FunctionRegistryService useful for other function UIs
- Components compose cleanly

### Integration Points

**With Existing Architecture**:
- Uses existing ParameterEditor for parameter editing
- Uses existing GroupedDropdown for function selection
- Uses existing ExternalEditorService for Vim integration
- Follows DualEditorPane composition pattern

**With DualEditorPane**:
- Same interface: `get_pattern()` and change callbacks
- No changes needed to DualEditorPane
- Clean drop-in replacement

### Risk Mitigation

**Backward Compatibility**:
- Keep same public interface
- Gradual refactoring approach
- Extensive testing at each phase

**Complexity Management**:
- Start with pure functions (lowest risk)
- Build UI components incrementally
- Test each component independently

### Backend Architecture Compliance

**Core Interface Preservation**:
- Maintain exact same public interface: `get_pattern()` and `change_callback`
- No changes needed to DualEditorPane integration
- Preserve state management patterns (original_pattern vs current_pattern)

**Essential Principles Respected**:
- **Immutable Pattern Operations**: All pattern transformations return new objects
- **None Key Semantics**: Preserve unnamed structural group handling
- **Function Registry Integration**: Maintain existing FUNC_REGISTRY usage
- **Validation Integration**: Continue using FuncStepContractValidator
- **External Editor Support**: Preserve Vim integration via ExternalEditorService

**No Breaking Changes**:
- Same constructor signature
- Same public methods
- Same callback patterns
- Same container property

This refactoring transforms a 970-line god class into 6 focused components totaling ~1000 lines (6 × 200 LOC max) with much better separation of concerns, testability, and maintainability while preserving all essential backend architecture principles.

---

## DOCTRINE ENFORCEMENT TEMPLATE

**Implementation Agent Must Follow This Process:**

### Before Each Component Implementation:
1. **Re-read this plan** - Verify understanding of component responsibilities
2. **Check LOC limits** - Each file MAX 200 lines, no exceptions
3. **Verify layer placement** - services/ for non-UI helpers, components/ for visual only
4. **Confirm composition** - NO Container inheritance, composition only

### During Implementation:
1. **STATIC ANALYSIS FIRST** - Comprehensive interface and data flow verification
2. **Validate naming** - Use `get_enhanced_function_metadata()` not `get_function_info()`
3. **Async safety** - Wrap file I/O in `run_in_executor()`
4. **Order determinism** - List↔Dict conversions must preserve order

### After Each Component:
1. **STATIC VERIFICATION** - Comprehensive interface compatibility analysis
2. **LOC check** - Verify file is under 200 lines
3. **Integration analysis** - Component works with existing architecture
4. **Checkpoint summary** - Document what was completed and any issues found

### Phase Completion Gates:
- **Phase 1**: Static analysis confirms all pure functions correct, no regressions
- **Phase 2**: UI components integrate cleanly, no Container inheritance
- **Phase 3**: Full integration works, DualEditorPane unchanged
- **Phase 4**: All imports work, no runtime crashes, performance maintained

**FAILURE TO FOLLOW THIS PROCESS INVALIDATES THE IMPLEMENTATION**

---

## CRITICAL DRIFT-PRONE PATTERNS

**These patterns are most likely to cause implementation drift:**

### 1. **Function Tuple Structure** (HIGH DRIFT RISK)
```python
# CURRENT: Functions stored as (callable, kwargs) tuples OR bare callables
# MUST preserve EXACT logic:
if isinstance(func_item, tuple) and len(func_item) == 2 and callable(func_item[0]):
    return func_item[0], func_item[1]
elif callable(func_item):
    return func_item, {}
else:
    return None, {}
```

### 2. **None Key Semantics** (HIGH DRIFT RISK)
```python
# CURRENT: None key represents unnamed structural groups
# MUST preserve EXACT conversion logic:
if not self.is_dict:
    self.current_pattern = {None: self.current_pattern}  # List → Dict

if list(self.current_pattern.keys()) == [None]:
    self.current_pattern = self.current_pattern[None]  # Dict → List
```

### 3. **Callback Chain Pattern** (MEDIUM DRIFT RISK)
```python
# FIXED: Nested lambda callbacks with unified task manager
# OLD BROKEN: get_app().create_background_task(...)
# NEW FIXED: get_task_manager().fire_and_forget(...)
on_parameter_change=lambda p_name, p_val_str, idx=index:
    get_task_manager().fire_and_forget(self._handle_parameter_change(p_name, p_val_str, idx), f"param_change_{idx}")
```

### 4. **Container Property Pattern** (HIGH DRIFT RISK)
```python
# CURRENT: DualEditorPane expects .container property
# MUST preserve EXACT interface:
@property
def container(self) -> Container:
    return self._container  # NOT self.view.container or similar
```

### 5. **Change Notification Pattern** (MEDIUM DRIFT RISK)
```python
# CURRENT: _notify_change() called after every pattern modification
# MUST call after: add, delete, move, parameter change, key operations
self._update_pattern_functions(functions)
self._notify_change()  # MUST be called together
```

**IF ANY OF THESE PATTERNS DRIFT, THE INTEGRATION WILL BREAK**

---

## FACT-CHECK SUMMARY

**✅ CONFIRMED ISSUES (Fixed in Plan):**
1. **Lines 910, 955**: `prompt_for_path_dialog` doesn't exist - need to create `prompt_for_save_file_dialog` with text input
2. **Lines 323, 433**: Uses deprecated `get_app().create_background_task()` - should use unified task manager
3. **Lines 926, 969**: Direct `open()` calls violate FileManager abstraction
4. **980 lines**: God class needs modular refactoring
5. **Spinner line 68**: Uses deprecated `app.create_background_task()` - should use unified task manager

**✅ ARCHITECTURAL DECISIONS (Corrected):**
1. **Pure composition** instead of ListManagerPane inheritance (cleaner separation)
2. **Unified task manager** integration for all async operations
3. **FileManager abstraction** for all file I/O operations
4. **Modern async patterns** using `asyncio.get_running_loop()`

**✅ IMPLEMENTATION STRATEGY (Validated):**
1. **Phase 1**: Fix critical bugs first (dialog calls, async patterns, file I/O)
2. **Phase 2**: Extract pure data operations (PatternDataManager, services)
3. **Phase 3**: Create UI components with composition pattern
4. **Phase 4**: Refactor main class to coordinate components

**🎯 PLAN STATUS: FACT-CHECKED AND CORRECTED**
