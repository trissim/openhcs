# plan_01_dialog_system_investigation.md
## Component: Dialog System Investigation and Repair

### Objective
Fix the PlateManager Add button dialog system and correct architectural violations. The PlateManager should use FileManager to browse folders, create orchestrators from selected paths, and display them as interactive list items. Current violations include improper ProcessingContext dependency and VFS abstraction breaks.

### Plan
1. **Architectural Violations Documentation** - Identify improper dependencies and VFS breaks
2. **Clean Architecture Implementation** - Fix PlateManager to use only FileManager
3. **Dialog System Repair** - Fix FileManagerBrowser constructor and VFS compliance
4. **Orchestrator Creation Workflow** - Implement clean folder → orchestrator → list item flow
5. **End-to-End Verification** - User clicks Add → selects folders → sees orchestrators in list

### Findings

#### ARCHITECTURAL VIOLATIONS

**Violation 1: PlateManager ProcessingContext Dependency**
```python
# CURRENT (wrong):
class PlateManagerPane:
    def __init__(self, state, context: ProcessingContext, storage_registry: Any):
        self.context = context
        filemanager = self.context.filemanager  # Wrong layer!
```

**Clean Architecture Should Be:**
```python
# CORRECT:
class PlateManagerPane:
    def __init__(self, state, filemanager: FileManager):
        self.filemanager = filemanager  # Direct FileManager dependency
        # No ProcessingContext needed - orchestrators create their own contexts
```

**Violation 2: VFS Abstraction Breaks**
- `Path.cwd()` calls bypass FileManager VFS abstraction
- Direct filesystem access violates backend independence
- Initial directory should come from config, not filesystem

#### Clean Workflow Architecture
**Correct execution chain:**
```
User clicks Add button
  → PlateManager._add_plates()
  → prompt_for_multi_folder_dialog(filemanager, initial_path_from_config)
  → FileManagerBrowser(filemanager, Backend.DISK, ...)
  → User selects folders
  → Create orchestrators: PipelineOrchestrator(config, folder_path)
  → Display as interactive list items with status symbols
  → User clicks Init/Compile/Run buttons
  → Call orchestrator methods directly
```

**Current violations blocking this workflow:**
1. PlateManager has ProcessingContext dependency (wrong layer)
2. FileManagerBrowser constructor parameter misalignment
3. VFS violations with Path.cwd() calls
4. Missing config-based initial directory

#### ROOT CAUSE: Backend Abstraction Violation

**Critical Issue: FileManagerBrowser Constructor Parameter Misalignment**

OpenHCS's elegance comes from its VFS (Virtual File System) abstraction where ALL file operations go through FileManager with explicit backend specification. The FileManagerBrowser is designed to browse files THROUGH this VFS abstraction.

**Constructor Signature:**
```python
FileManagerBrowser(file_manager, backend, on_path_selected, on_cancel, initial_path, ...)
```

**Current BROKEN Call:**
```python
FileManagerBrowser(
    file_manager=filemanager,           # ✅ Position 1: Correct
    on_path_selected=on_folder_selected, # ❌ Position 2: Goes to backend!
    on_cancel=on_browser_cancel,        # ❌ Position 3: Goes to on_path_selected!
    initial_path=None,                  # ❌ Position 4: Goes to on_cancel!
    backend=Backend.DISK,               # ❌ Position 5: Goes to initial_path!
    ...
)
```

**The Problem:** FileManagerBrowser is trying to use `on_folder_selected` function as the backend, completely breaking the VFS abstraction that makes OpenHCS elegant.

**Future Vision:** The same FileManagerBrowser could browse Memory backend to show images being processed in real-time through the VFS - but only if the backend abstraction works correctly.

#### Implementation Fixes Required

**Fix 1: PlateManager Constructor**
```python
# CURRENT (wrong):
def __init__(self, state, context: ProcessingContext, storage_registry: Any):

# CORRECT:
def __init__(self, state, filemanager: FileManager):
```

**Fix 2: Remove ProcessingContext Usage**
```python
# CURRENT (wrong):
filemanager=self.context.filemanager

# CORRECT:
filemanager=self.filemanager
```

**Fix 3: Config-Based Initial Directory**
```python
# CURRENT (VFS violation):
initial_path=Path.cwd()

# CORRECT:
initial_path=config.initial_browse_directory
```

**Fix 4: FileManagerBrowser Parameter Order**
```python
# CURRENT (wrong parameter positions):
FileManagerBrowser(file_manager, on_path_selected, on_cancel, initial_path, backend, ...)

# CORRECT:
FileManagerBrowser(file_manager, backend, on_path_selected, on_cancel, initial_path, ...)
```

### Implementation Strategy

#### Phase 1: Architectural Cleanup
1. **Fix PlateManager constructor** - Remove ProcessingContext dependency
2. **Update PlateManager calls** - Use self.filemanager instead of self.context.filemanager
3. **Fix VFS violations** - Remove Path.cwd() calls, use config-based paths
4. **Fix FileManagerBrowser constructor** - Correct parameter order and add missing parameters

#### Phase 2: Orchestrator Creation Workflow
1. **Update dialog helpers** - Accept initial_path parameter from caller
2. **Implement config-based initial directory** - Get from global config or state
3. **Fix orchestrator creation** - Use PipelineOrchestrator(config, folder_path)
4. **Ensure list display** - Selected folders appear as interactive list items

#### Phase 3: User Testing
1. **User clicks Add button** - Test if dialog appears
2. **User selects folders** - Test if selection works
3. **Check orchestrator creation** - Test if list items appear with correct status
4. **Test button actions** - Init/Compile/Run should call orchestrator methods

### Expected Outcomes

#### Immediate Goals
1. **Clean Architecture** - PlateManager uses only FileManager, no ProcessingContext
2. **VFS Compliance** - All file operations through FileManager with proper backend specification
3. **Working Dialog** - Add button shows file browser for folder selection
4. **Orchestrator Creation** - Selected folders become PipelineOrchestrator instances
5. **Interactive List** - Orchestrators appear as list items with status symbols

#### Long-term Benefits
1. **Architectural Elegance** - Clean separation between TUI and pipeline layers
2. **Backend Flexibility** - Future Memory backend overlay for real-time visualization
3. **Maintainable Code** - Clear dependencies and responsibilities
4. **Extensible System** - Easy to add new storage backends or TUI features

### Implementation Priorities

#### Priority 1: Fix Architectural Violations
1. **Remove ProcessingContext dependency from PlateManager:**
   ```python
   # Change constructor signature:
   def __init__(self, state, filemanager: FileManager):
       self.filemanager = filemanager
       # Remove: self.context = context
   ```

2. **Fix VFS abstraction violations:**
   ```python
   # CORRECT FileManagerBrowser constructor:
   FileManagerBrowser(
       file_manager=filemanager,
       backend=Backend.DISK,  # ← Explicit backend in correct position
       on_path_selected=on_folder_selected,
       on_cancel=on_browser_cancel,
       initial_path=config.initial_browse_directory,  # ← From config, not filesystem
       selection_mode=SelectionMode.DIRECTORIES_ONLY,
       allow_multiple=True,  # ← Add missing parameter
       show_hidden_files=False
   )
   ```

3. **Clean orchestrator creation workflow:**
   - User selects folders via FileManagerBrowser
   - Create orchestrators: `PipelineOrchestrator(folder_path)` (uses global config by default)
   - Display as interactive list items with status symbols

4. **Plate-specific config override pattern:**
   - Edit button opens same config editor as global config
   - User can set plate-specific overrides for selected plates
   - Orchestrators created with merged config: `PipelineOrchestrator(folder_path, global_config=merged_config)`

#### Priority 2: User Testing Cycle
1. **User clicks Add button** - Reports what happens
2. **Static analysis of reported issue** - Find obvious code problem
3. **Simple fix** - No elaborate testing, just fix the code
4. **Repeat** - User clicks button again

#### Priority 3: Minimal Logging for Debugging
Only add logging if user reports "nothing happens":
- Log when `_handle_dialog_request()` is called
- Log when dialog is added to FloatContainer
- Log when buttons are clicked

### Expected Outcomes

#### Immediate Goals
1. **Dialog Visibility** - Add button shows file browser dialog
2. **Folder Selection** - User can browse and select multiple folders
3. **Orchestrator Creation** - Selected folders become plate entries
4. **List Display** - New plates appear as interactive list items below buttons

#### Long-term Goals
1. **Robust Dialog System** - Reliable modal dialog infrastructure for all TUI components
2. **File Browser Integration** - Reusable file/folder selection across application
3. **Error Handling** - Graceful failure modes with user feedback
4. **Performance** - Responsive dialog display and interaction

### Implementation Checklist

#### Files to Modify
1. **openhcs/tui/panes/plate_manager.py** - Constructor and filemanager usage
2. **openhcs/tui/utils/dialog_helpers.py** - FileManagerBrowser constructor and imports
3. **Caller of PlateManagerPane** - Update constructor call to pass FileManager
4. **Config system** - Add initial_browse_directory setting
5. **Config editor integration** - Enable plate-specific config overrides using same editor as global config

#### Validation Steps
1. **No ProcessingContext in PlateManager** - Only FileManager dependency
2. **VFS compliance** - No direct filesystem access
3. **Working file browser** - Dialog appears and allows folder selection
4. **Orchestrator creation** - Selected folders become interactive list items
5. **Status progression** - ? → - → o → ! workflow functions correctly
6. **Config editor integration** - Edit button opens config editor for plate-specific overrides

This plan provides a complete roadmap for implementing the clean architecture and fixing the dialog system.
