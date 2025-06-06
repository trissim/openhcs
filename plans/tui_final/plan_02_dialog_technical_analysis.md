# plan_02_dialog_technical_analysis.md
## Component: Dialog System Technical Implementation Analysis

### Objective
Fix architectural violations in PlateManager and implement clean folder selection workflow. PlateManager should use only FileManager (not ProcessingContext), create orchestrators from selected folder paths, and display them as interactive list items. Remove VFS abstraction violations and implement proper backend specification.

### Plan
1. **PlateManager Architecture Cleanup** - Remove ProcessingContext dependency
2. **VFS Compliance Implementation** - Fix FileManagerBrowser constructor and config-based paths
3. **Orchestrator Creation Workflow** - Implement folder → orchestrator → list item flow
4. **Dialog System Integration** - Ensure FileManagerBrowser works with clean architecture
5. **Global Storage Registry Pattern** - Single registry shared by all FileManager instances

### Findings

#### Current Architectural Violations

**Violation 1: PlateManager ProcessingContext Dependency**
```python
# CURRENT (wrong layer mixing):
from openhcs.core.context.processing_context import ProcessingContext

class PlateManagerPane:
    def __init__(self, state, context: ProcessingContext, storage_registry: Any):
        self.context = context
        filemanager = self.context.filemanager  # Wrong!
```

**Why this is wrong:**
- ProcessingContext is for stateless pipeline execution
- PlateManager is a TUI component that manages orchestrators
- Orchestrators create their own processing contexts internally
- TUI layer should not depend on pipeline execution layer

**Violation 2: VFS Abstraction Breaks**
```python
# CURRENT (VFS violation):
initial_path=Path.cwd()  # Direct filesystem access!

# CORRECT (VFS compliant):
initial_path=config.initial_browse_directory  # Through config system
```

#### Clean Architecture Implementation

**Correct PlateManager Architecture:**
```python
# CLEAN ARCHITECTURE:
class PlateManagerPane:
    def __init__(self, state, filemanager: FileManager):
        self.state = state
        self.filemanager = filemanager  # Direct FileManager dependency
        self.orchestrators: Dict[str, PipelineOrchestrator] = {}

    async def _add_plates(self):
        # Get folders through VFS-compliant file browser
        folder_paths = await prompt_for_multi_folder_dialog(
            filemanager=self.filemanager,
            initial_path=self.state.config.initial_browse_directory
        )

        # Create orchestrators from selected folders
        for folder_path in folder_paths:
            orchestrator = PipelineOrchestrator(
                config=self.state.global_config,
                plate_path=folder_path
            )
            self.orchestrators[folder_path] = orchestrator

        # Display as interactive list items
        self._update_plate_list()
```

**Global Storage Registry Pattern:**
```python
# CURRENT: Global registry created at module import time in openhcs.io.base
from openhcs.io.base import storage_registry  # Global singleton

# Registry contains: {"disk": DiskStorageBackend(), "memory": MemoryStorageBackend(), "zarr": ZarrStorageBackend()}

# CURRENT: Application startup in simple_launcher.py
self.shared_storage_registry = storage_registry  # Use global singleton
self.filemanager = FileManager(self.shared_storage_registry)

# CURRENT: PlateManager gets storage_registry parameter
plate_manager = PlateManagerPane(state, context, storage_registry)

# FIXED: PlateManager should get FileManager directly
plate_manager = PlateManagerPane(state, filemanager)
```

#### FileManagerBrowser Constructor Fix

**Current BROKEN Parameter Order:**
```python
# WRONG - parameters in wrong positions:
FileManagerBrowser(
    file_manager=filemanager,           # ✅ Position 1: Correct
    on_path_selected=on_folder_selected, # ❌ Position 2: Goes to backend!
    on_cancel=on_browser_cancel,        # ❌ Position 3: Goes to on_path_selected!
    initial_path=Path.cwd(),            # ❌ Position 4: Goes to on_cancel! + VFS violation!
    backend=Backend.DISK,               # ❌ Position 5: Goes to initial_path!
    ...
)
```

**CORRECT Parameter Order:**
```python
# FIXED - correct parameter positions and VFS compliance:
FileManagerBrowser(
    file_manager=filemanager,
    backend=Backend.DISK,               # ✅ Position 2: Explicit backend
    on_path_selected=on_folder_selected, # ✅ Position 3: Correct callback
    on_cancel=on_browser_cancel,        # ✅ Position 4: Correct callback
    initial_path=config.initial_browse_directory, # ✅ Position 5: Config-based, VFS compliant
    selection_mode=SelectionMode.DIRECTORIES_ONLY,
    allow_multiple=True,                # ✅ Add missing parameter
    show_hidden_files=False
)
```

### Implementation Strategy

#### Phase 1: Architectural Cleanup
1. **Remove ProcessingContext from PlateManager**
   - Change constructor signature: `def __init__(self, state, filemanager: FileManager)`
   - Remove import: `from openhcs.core.context.processing_context import ProcessingContext`
   - Update all usage: `self.filemanager` instead of `self.context.filemanager`

2. **Fix VFS Violations**
   - Add import: `from pathlib import Path` to dialog_helpers.py
   - Replace `Path.cwd()` with config-based initial directory
   - Update dialog helpers to accept `initial_path` parameter

3. **Fix FileManagerBrowser Constructor**
   - Correct parameter order: `file_manager, backend, on_path_selected, on_cancel, initial_path, ...`
   - Add missing `allow_multiple=True` parameter
   - Ensure backend is explicitly specified in correct position

#### Phase 2: Global Storage Registry Implementation
1. **Single Registry Pattern**
   - Initialize one storage registry at application startup
   - All FileManager instances constructed with same registry
   - Remove storage_registry parameter from PlateManager

2. **Update Caller Code**
   - Modify code that creates PlateManagerPane
   - Pass FileManager instead of ProcessingContext and storage_registry
   - Ensure FileManager has global storage registry

#### Phase 3: Orchestrator Creation Workflow
1. **Clean Folder Selection**
   - User clicks Add button
   - FileManagerBrowser dialog appears with config-based initial directory
   - User selects multiple folders through VFS-compliant interface
   - Dialog returns selected folder paths

2. **Orchestrator Creation**
   ```python
   # Clean orchestrator creation pattern:
   for folder_path in selected_folders:
       # Use global config by default, or merged config for plate-specific overrides
       orchestrator = PipelineOrchestrator(plate_path=folder_path)
       # For plates with specific config overrides:
       # orchestrator = PipelineOrchestrator(plate_path=folder_path, global_config=merged_config)
       self.orchestrators[folder_path] = orchestrator
   ```

3. **Interactive List Display**
   - Create plate data dictionaries with name/path/status
   - Display as interactive list items with status symbols
   - Status progression: ? (added) → - (initialized) → o (compiled) → ! (running)

4. **Plate-Specific Config Override System**
   - Edit button opens same config editor as global config
   - Config editor operates in "plate-specific override" mode for selected plates
   - Overrides stored per plate, merged with global config when creating orchestrators
   - Consistent UI pattern between global and plate-specific configuration

#### Phase 4: Validation and Testing
1. **Architectural Validation**
   - PlateManager has no ProcessingContext dependency
   - All file operations go through FileManager VFS abstraction
   - No direct filesystem access (no Path.cwd() calls)
   - Single global storage registry shared by all FileManager instances

2. **Functional Testing**
   - User clicks Add button → dialog appears
   - User selects folders → orchestrators created
   - Orchestrators appear as interactive list items
   - Init/Compile/Run buttons call orchestrator methods correctly

### Summary

This plan provides a complete roadmap for fixing the PlateManager architectural violations and implementing the clean folder selection workflow. The key changes are:

1. **Remove ProcessingContext dependency** - PlateManager uses only FileManager
2. **Fix VFS violations** - Config-based paths, proper backend specification
3. **Correct constructor parameters** - FileManagerBrowser parameter order and missing parameters
4. **Implement clean workflow** - Folder selection → orchestrator creation → interactive list display
5. **Global storage registry** - Single registry shared by all FileManager instances

The result will be a clean, architecturally sound system where:
- TUI layer (PlateManager) is properly separated from pipeline execution layer
- All file operations respect the VFS abstraction
- Scientists can easily add plates through folder selection
- Orchestrators manage their own processing contexts internally
- The system maintains OpenHCS's architectural elegance and extensibility

### Technical Implementation Issues

#### Issue 1: FileManagerBrowser Constructor Fix
**Problem:** Parameter mismatches causing potential silent failures
**Solution:**
```python
# Fixed constructor call:
file_browser = FileManagerBrowser(
    file_manager=filemanager,
    backend=Backend.DISK,  # Move to correct position
    on_path_selected=on_folder_selected,
    on_cancel=on_browser_cancel,
    initial_path=Path.cwd(),  # Use current directory instead of None
    selection_mode=SelectionMode.DIRECTORIES_ONLY,
    allow_multiple=True,  # Add missing parameter
    show_hidden_files=False
)
```

#### Issue 2: Float Positioning and Modal Behavior
**Problem:** Dialog may not be visible or properly modal
**Solution:**
```python
# Enhanced Float creation:
float_dialog = Float(
    content=dialog,
    xcursor=True,  # Center on screen
    ycursor=True,
    z_index=1000,  # Ensure on top
    transparent=False
)
```

#### Issue 3: Focus Management
**Problem:** Dialog may not receive keyboard focus
**Solution:**
```python
# After adding to floats:
layout.container.floats.append(float_dialog)
get_app().layout.focus(dialog)  # Explicitly focus dialog
get_app().invalidate()
```

#### Issue 4: Async Exception Handling
**Problem:** Unhandled exceptions may cause silent failures
**Solution:**
```python
# Enhanced error handling:
try:
    await file_browser.start_load()
    await app_state.show_dialog(dialog, result_future=future)
    return await future
except Exception as e:
    logger.error(f"Dialog error: {e}", exc_info=True)
    return None
```

### Implementation Strategy: Static Analysis + User Testing

#### No Elaborate Testing Frameworks
**User testing is simple:** Click the Add button, report what happens. That's it.

**Real work is static analysis:** Find obvious code issues and fix them.

#### Static Analysis Fixes Needed
1. **FileManagerBrowser constructor parameters** - Fix obvious mismatches
2. **Float positioning** - Add xcursor/ycursor for visibility
3. **Future completion** - Ensure buttons complete the result_future
4. **Minimal logging** - Only if user reports "nothing happens"

#### User Testing Cycle
1. **Fix obvious static analysis issue**
2. **User clicks Add button**
3. **User reports what happens** ("dialog appears", "nothing happens", "error message", etc.)
4. **Static analysis of reported issue**
5. **Repeat until working**

### Implementation Checklist

#### Files to Modify
1. **openhcs/tui/panes/plate_manager.py** - Remove ProcessingContext, use FileManager
2. **openhcs/tui/utils/dialog_helpers.py** - Fix constructor calls and add Path import
3. **Application startup code** - Create global storage registry and FileManager
4. **PlateManager caller** - Update constructor to pass FileManager

#### Success Criteria
1. **Clean Architecture** - PlateManager depends only on FileManager
2. **VFS Compliance** - No direct filesystem access, config-based paths
3. **Working Dialog** - Add button shows file browser successfully
4. **Orchestrator Creation** - Selected folders become interactive list items
5. **Status Workflow** - ? → - → o → ! progression works correctly

This plan is now complete and ready for implementation.



