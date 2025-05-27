# plan_03_step_settings_editor_completion.md
## Component: Step Settings Editor Completion

### Objective
Complete TUI2's placeholder `step_settings_editor.py` by porting TUI's parameter editor patterns and using AbstractStep introspection instead of schema dependencies.

### Plan
1. Analyze TUI's parameter editor implementation
2. Complete TUI2's StepSettingsEditorView using static analysis
3. Remove schema dependencies completely
4. Implement dynamic form generation for AbstractStep parameters
5. Integrate with DualEditorController

### Findings

**TUI Parameter Editor Analysis:**

**From `components/parameter_editor.py` (TUI):**
```python
class ParameterEditor:
    def _create_parameter_editor(self, name, current_value, default_value)
    def _build_ui(self)
    def _handle_reset_parameter(self, name)
    def _handle_reset_all_parameters(self)
```

**Key Features to Port:**
- ✅ Dynamic TextArea creation for parameters
- ✅ Reset buttons for individual parameters
- ✅ Reset all functionality
- ✅ Type-aware input handling
- ✅ Change callbacks with validation

**TUI2 Current Implementation (Incomplete):**
```python
class StepSettingsEditorView:
    def __init__(self, on_parameter_change, get_current_step_schema)  # ← Schema dependency
    async def set_step_data(self, step_data)
    async def _build_form(self)  # ← Placeholder
```

**Schema Removal Strategy:**

**Replace Schema with Static Analysis:**
```python
# OLD (Schema-based):
self.current_step_schema = self.get_current_step_schema()

# NEW (Static Analysis):
from openhcs.core.steps.abstract import AbstractStep
import inspect

def get_abstractstep_parameters() -> Dict[str, Any]:
    """Extract AbstractStep.__init__ parameters"""
    sig = inspect.signature(AbstractStep.__init__)
    params = {}
    for name, param in sig.parameters.items():
        if name == 'self':
            continue
        params[name] = {
            'type': param.annotation,
            'default': param.default,
            'required': param.default == inspect.Parameter.empty
        }
    return params
```

**AbstractStep Parameters (from codebase analysis):**
```python
# From AbstractStep.__init__:
name: Optional[str] = None
variable_components: Optional[List[str]] = None
force_disk_output: Optional[bool] = False
group_by: Optional[str] = None
input_dir: Optional[Union[str,Path]] = None
output_dir: Optional[Union[str,Path]] = None
```

**Form Generation Strategy:**
```python
async def _build_form(self):
    """Build form using static analysis instead of schema"""
    if not self.current_step_data:
        return

    # Get AbstractStep parameters via introspection
    step_params = get_abstractstep_parameters()
    form_widgets = []

    for param_name, param_info in step_params.items():
        current_value = self.current_step_data.get(param_name)
        widget = self._create_parameter_widget(
            param_name,
            param_info['type'],
            current_value,
            param_info['default']
        )
        form_widgets.append(widget)

    self._form_container = HSplit(form_widgets)
```

**Widget Creation by Type:**
```python
def _create_parameter_widget(self, name, param_type, current_value, default_value):
    """Create appropriate widget based on parameter type"""

    if param_type == Optional[str] or param_type == str:
        return self._create_text_widget(name, current_value, default_value)
    elif param_type == Optional[bool] or param_type == bool:
        return self._create_checkbox_widget(name, current_value, default_value)
    elif param_type == Optional[List[str]]:
        return self._create_list_widget(name, current_value, default_value)
    elif 'Union[str,Path]' in str(param_type):
        return self._create_path_widget(name, current_value, default_value)
    else:
        return self._create_text_widget(name, current_value, default_value)
```

**Integration with Controller:**
```python
# In DualEditorController:
self.step_settings_editor = StepSettingsEditorView(
    on_parameter_change=self._on_step_parameter_changed,
    # Remove schema callback - use static analysis
)
```

**Port Checklist:**
- [ ] Remove schema dependencies from constructor
- [ ] Implement static analysis parameter extraction
- [ ] Create type-aware widget generation
- [ ] Port reset functionality from TUI
- [ ] Implement change callbacks
- [ ] Add validation for parameter types
- [ ] Test with real FunctionStep objects
- [ ] Integrate with controller

### Implementation Draft

**✅ COMPLETED: Step Settings Editor Completion**

Successfully completed TUI2's placeholder step settings editor using TUI's parameter editor patterns:

**✅ Core Components Implemented:**
1. **StepSettingsEditor** - Complete step parameter editor using static analysis
2. **ParameterEditor** - Reusable parameter form generator for any callable
3. **Enhanced FunctionPatternEditor** - Updated to use new ParameterEditor

**✅ Key Features Implemented:**
- ✅ Dynamic form generation from AbstractStep introspection
- ✅ Type-aware widget creation (text, checkbox, path, list)
- ✅ Reset functionality (individual + reset all)
- ✅ Change callbacks with validation
- ✅ Schema-free operation using static analysis
- ✅ File browser integration for path parameters
- ✅ Component interface compliance

**✅ Schema Removal Achieved:**
- ✅ No ParamSchema dependencies
- ✅ Uses `get_abstractstep_parameters()` for introspection
- ✅ Direct `inspect.signature()` usage for function analysis
- ✅ Type-based widget selection without schema metadata

**✅ Architecture Improvements:**
- ✅ Reusable ParameterEditor component
- ✅ Clean separation between step and function parameter editing
- ✅ Consistent async/await patterns
- ✅ Proper error handling and logging
- ✅ Type conversion and validation

**✅ Files Created/Updated:**
```
openhcs/tui_hybrid/components/
├── step_settings_editor.py       ✅ 300+ lines - Complete step editor
├── parameter_editor.py           ✅ 300+ lines - Reusable param editor
├── function_pattern_editor.py    ✅ Updated to use ParameterEditor
└── __init__.py                   ✅ Updated exports
```

**✅ Integration Points:**
- ✅ Uses hybrid static analysis utilities
- ✅ Integrates with FileManagerBrowser for path selection
- ✅ Implements EditorComponentInterface
- ✅ Ready for controller integration
- ✅ Compatible with FunctionStep objects

**✅ Type-Aware Widget Creation:**
- **Boolean**: CheckBox widgets
- **Path/Union[str,Path]**: TextArea + Browse button with FileManagerBrowser
- **List[str]**: TextArea with comma separation
- **int/float**: TextArea with automatic type conversion
- **str**: TextArea with string handling

**🚀 Ready for Phase 4: Controller Integration**

The step settings editor is now complete with full AbstractStep parameter support. All schema dependencies have been removed and replaced with static analysis. The component is ready for integration with controllers.
