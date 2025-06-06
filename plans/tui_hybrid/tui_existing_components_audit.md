# TUI Existing Components Audit - What Can Be Reused

## ✅ **EXISTING COMPONENTS - DON'T REWRITE**

### **Parameter Editing System - COMPLETE**
- **ParameterEditor** (`openhcs/tui/components/parameter_editor.py`)
  - Auto-generates forms from function signatures using `inspect.signature()`
  - Handles parameter validation and type conversion
  - Supports reset functionality and callbacks
  - **REUSE**: Connect to Add Step dialog instead of building new parameter editor

### **Function Selection System - COMPLETE**
- **GroupedDropdown** (`openhcs/tui/components/grouped_dropdown.py`)
  - Groups options by category with headers
  - Handles selection callbacks and validation
  - **REUSE**: Use for function selection from FUNC_REGISTRY

### **Visual Programming Editor - EXISTS**
- **DualStepFuncEditor** (`openhcs/tui/dual_step_func_editor.py`)
  - Step/Function dual view editor
  - Parameter editing integration
  - Function pattern editing
  - **REUSE**: This IS the visual programming interface

### **Command System - COMPLETE**
- **Command Registry** (`openhcs/tui/commands/registry.py`)
- **Base Commands** (`openhcs/tui/commands/base_command.py`)
- **Existing Commands**:
  - `InitializePlatesCommand` - Real orchestrator integration
  - `CompilePlatesCommand` - Real orchestrator integration  
  - `RunPlatesCommand` - Real orchestrator integration
  - `ShowHelpCommand` - Working help dialog
  - `ShowGlobalSettingsDialogCommand` - Config editor
  - `ShowAddPlateDialogCommand` - File browser integration
- **REUSE**: Replace all custom button handlers with command system

### **Dialog Infrastructure - COMPLETE**
- **DialogManager** (`openhcs/tui/dialogs/manager.py`)
- **BaseDialog** (`openhcs/tui/dialogs/base.py`)
- **Working Dialogs**:
  - `HelpDialog` (`openhcs/tui/dialogs/help_dialog.py`)
  - `GlobalSettingsEditorDialog` (`openhcs/tui/dialogs/global_settings_editor.py`)
  - `PlateDialogManager` (`openhcs/tui/dialogs/plate_dialog_manager.py`)
- **REUSE**: Use existing dialog system instead of custom _show_dialog()

### **File Browser - WORKING**
- **FileManagerBrowser** (`openhcs/tui/file_browser.py`)
  - Real file selection with callbacks
  - Directory and file selection modes
  - VFS integration
- **REUSE**: Already integrated in Add Plate, working correctly

## 🚨 **CRITICAL ISSUE: canonical_layout.py IGNORES EXISTING SYSTEM**

### **Problem**: Duplicate Implementation
The `canonical_layout.py` file reimplements everything from scratch instead of using the existing, working components:

1. **Custom button handlers** instead of Command system
2. **Custom dialog management** instead of DialogManager
3. **Placeholder Add Step** instead of existing DualStepFuncEditor
4. **Fake status updates** instead of real orchestrator commands
5. **Custom help dialog** instead of existing HelpDialog

### **Solution**: Integration, Not Rewriting
Replace canonical_layout.py handlers with existing components:

```python
# WRONG - Current approach
def _handle_add_step(self):
    self._update_status("Add Step: Function picker not yet implemented")

# RIGHT - Use existing system  
def _handle_add_step(self):
    command = command_registry.get('add_step')
    await command.execute(self.state, self.context)
```

## 🔧 **SPECIFIC INTEGRATION TASKS**

### **1. Replace Button Handlers with Commands**
```python
# Current canonical_layout.py
self._handle_init_plate()
self._handle_compile_plate() 
self._handle_run_plate()

# Should be
await command_registry.execute('initialize_plates', self.state, self.context)
await command_registry.execute('compile_plates', self.state, self.context)
await command_registry.execute('run_plates', self.state, self.context)
```

### **2. Use Existing Dialog System**
```python
# Current canonical_layout.py
self._show_dialog(help_dialog)

# Should be
dialog_manager = self.state.dialog_manager
await dialog_manager.show('help_dialog')
```

### **3. Connect to Real Function Registry**
```python
# Current - hardcoded
mock_functions = ["gaussian_blur", "enhance_contrast"]

# Should be
from openhcs.processing.func_registry import get_functions_by_memory_type
numpy_functions = get_functions_by_memory_type('numpy')
```

### **4. Use Existing Parameter Editor**
```python
# Current - placeholder
self._update_status("Add Step: Function picker not implemented")

# Should be
param_editor = ParameterEditor(
    func=selected_function,
    current_kwargs={},
    on_parameter_change=self._handle_parameter_change
)
```

## 🎯 **IMMEDIATE ACTION PLAN**

### **Phase 1: Stop Reinventing**
1. **Audit canonical_layout.py** - Identify all custom implementations
2. **Map to existing components** - Find existing equivalent for each custom piece
3. **Replace incrementally** - One handler at a time

### **Phase 2: Integration**
1. **Connect to command system** - Replace all _handle_* methods
2. **Use dialog manager** - Replace custom dialog code
3. **Connect to FUNC_REGISTRY** - Replace hardcoded function lists
4. **Use existing editors** - Replace placeholder Add Step

### **Phase 3: Cleanup**
1. **Remove duplicate code** - Delete custom implementations
2. **Test integration** - Verify existing components work together
3. **Fix any gaps** - Address missing pieces without rewriting existing ones

## 📋 **EXISTING COMPONENTS INVENTORY**

### **Working and Ready to Use:**
- ✅ ParameterEditor - Function signature → UI forms
- ✅ GroupedDropdown - Categorized function selection  
- ✅ DualStepFuncEditor - Visual programming interface
- ✅ Command system - All orchestrator integration
- ✅ Dialog system - Modal dialogs and management
- ✅ File browser - Directory/file selection
- ✅ FUNC_REGISTRY - Function discovery
- ✅ Real orchestrator - All methods exist and work

### **Missing/Incomplete:**
- ❌ Integration between canonical_layout.py and existing system
- ❌ Function picker dialog (but components exist to build it)
- ❌ Pipeline visualization (but data structures exist)
- ❌ Selection management in lists (but InteractiveListItem exists)

## 🎉 **THE GOOD NEWS**

**90% of the TUI functionality already exists and works.** The problem is not missing features - it's that canonical_layout.py was written without using the existing architecture.

**The fix is integration, not implementation.**
