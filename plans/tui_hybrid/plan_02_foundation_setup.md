# plan_01_foundation_setup.md
## Component: Foundation Setup and Core Utilities

### Objective
Establish the hybrid TUI foundation by creating the folder structure, porting core utilities, and setting up component interfaces that will support the merged architecture.

### Plan
1. Create hybrid TUI folder structure based on TUI2 MVC pattern
2. Port essential utilities from TUI (dialogs, error handling)
3. Create component interface standards
4. Port file operation utilities
5. Establish static analysis helpers

### Findings

**Hybrid Folder Structure:**
```
openhcs/tui_hybrid/
├── __init__.py
├── app_controller.py              ← From TUI2, adapted
├── async_manager.py               ← From TUI2
├── controllers/
│   ├── __init__.py
│   ├── dual_editor_controller.py  ← From TUI2, enhanced
│   ├── pipeline_editor_controller.py
│   └── plate_manager_controller.py
├── components/
│   ├── __init__.py
│   ├── function_pattern_editor.py ← From TUI, adapted
│   ├── step_settings_editor.py    ← From TUI2, completed
│   ├── parameter_editor.py        ← From TUI, adapted
│   ├── plate_list_view.py         ← From TUI2
│   └── step_list_view.py          ← From TUI2
├── utils/
│   ├── __init__.py
│   ├── dialogs.py                 ← From TUI utils.py
│   ├── static_analysis.py         ← New - schema replacement
│   └── file_operations.py         ← From TUI, extracted
└── interfaces/
    ├── __init__.py
    └── component_interfaces.py    ← New - standardized interfaces
```

**Core Utilities to Port (from TUI):**
- `show_error_dialog()` - Error display
- `prompt_for_path_dialog()` - File dialogs
- External editor integration
- File load/save operations

**Static Analysis Helpers (New):**
```python
# static_analysis.py
def get_abstractstep_parameters() -> Dict[str, Any]:
    """Extract AbstractStep.__init__ parameters via introspection"""

def get_function_signature(func: Callable) -> Dict[str, Any]:
    """Extract function signature for parameter forms"""

def get_function_registry_grouped() -> Dict[str, List[Callable]]:
    """Get FUNC_REGISTRY organized by backend"""
```

**Component Interface Standards:**
```python
# component_interfaces.py
class ComponentInterface(Protocol):
    @property
    def container(self) -> Container:
        """Return prompt_toolkit container"""

    def update_data(self, data: Any) -> None:
        """Update component with new data"""

class EditorComponentInterface(ComponentInterface):
    def get_current_value(self) -> Any:
        """Get current edited value"""

    def set_change_callback(self, callback: Callable) -> None:
        """Set callback for value changes"""
```

**File Operations to Extract:**
- `.func` pattern file load/save
- `.step` file operations
- `.pipeline` file operations
- Pickle serialization helpers

**Schema Removal Targets:**
- Remove all `ParamSchema` references
- Remove `CoreStepData` schema dependencies
- Replace with direct `FunctionStep` and `AbstractStep` usage
- Use `inspect.signature()` for all parameter introspection

### Implementation Draft

**Step 1: Create Hybrid Folder Structure**

First, let me create the hybrid TUI directory structure:

```bash
mkdir -p openhcs/tui_hybrid/{controllers,components,utils,interfaces}
```

**Step 2: Create Foundation Files**

✅ **COMPLETED**: Created hybrid TUI foundation structure:

```
openhcs/tui_hybrid/
├── __init__.py                     ✅ Main module with version info
├── controllers/
│   └── __init__.py                 ✅ Controller module placeholder
├── components/
│   └── __init__.py                 ✅ Component module placeholder
├── utils/
│   ├── __init__.py                 ✅ Utils module exports
│   ├── static_analysis.py          ✅ Schema replacement functions
│   ├── dialogs.py                  ✅ Error/file dialogs with async
│   └── file_operations.py          ✅ Load/save .func/.step/.pipeline
└── interfaces/
    ├── __init__.py                 ✅ Interface module exports
    └── component_interfaces.py     ✅ Component interface protocols
```

**Key Features Implemented:**
- ✅ Static analysis functions to replace schema dependencies
- ✅ Async dialog utilities (error, file selection, confirmation)
- ✅ File operations for .func, .step, .pipeline files
- ✅ Component interface protocols for clean architecture
- ✅ Proper error handling and logging throughout

**Next Steps:**
Ready to proceed to Phase 2 (Function Pattern Editor Port)
