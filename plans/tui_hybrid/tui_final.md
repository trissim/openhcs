# OpenHCS TUI Final Specification
**CANONICAL REFERENCE - IMPLEMENTATION REQUIRED**

## 🎯 CRITICAL UNDERSTANDING
This document defines the EXACT layout and behavior the OpenHCS TUI MUST implement. Any deviation from this specification is a bug.

## 📐 LAYOUT STRUCTURE

### **TOP BAR (Global Controls)**
```
| [Global Settings] [Help]                                    OpenHCS V1.0 |
```
- **Global Settings**: Opens modal dialog for default config inspection/editing
- **Help**: Opens modal dialog with help text and OK button
- **OpenHCS V1.0**: Right-aligned application title


### **2 MAIN PANES (Plate Manager & Pipeline Editor and Dual Editor)**

- **Plate Manager**: Left pane for orchestrator/plate management
- **Pipeline Editor**: Right pane for pipeline step management
- **Dual Editor**: Replaces Plate Manager when editing steps (see way further below)
- Platemanager and Pipeline Editor are built using the same structure as each other. See below

### **TITLE BARS (Pane Titles)**
```
| Plate Manager                          | Pipeline Editor                      |
```
- **Left**: "Plate Manager" title
- **Right**: "Pipeline Editor" title (formerly "Step Viewer")



### **BUTTON BARS (Pane Controls)**
```
| [add] [del] [edit] [init] [compile] [run] | [add] [del] [edit] [load] [save] |
```
- **Left**: Plate Manager buttons
- **Right**: Pipeline Editor buttons

### **CONTENT AREAS (Interactive Lists)**
```
| Plate List (Orchestrators)             | Step List (Pipeline Components)     |
```
- **Left**: Interactive list of plates/orchestrators
- **Right**: Interactive list of pipeline steps

## 🔧 DETAILED FUNCTIONALITY

### **PLATE MANAGER BUTTONS**

#### **[add]** - Create New Plate
- Opens file select dialog for folder selection
- Multiple folders may be selected at once
- Creates orchestrator with default config (from global settings)
- Adds new entry to plate list with **?** status symbol (not initialized)

#### **[del]** - Delete Plate
- Removes currently selected plate(s) from list
- Confirms deletion with user

#### **[edit]** - Edit Plate Config
- Opens custom config editor for selected plates
- Reflects current global settings as starting point
- Allows parameter editing and saving as plate-specific config
- Uses static reflection similar to function pattern editor

#### **[init]** - Initialize Plate
- Runs `initialize()` method on selected plate(s)
- Updates status symbol to **-** (yellow) if successful (initialized but not compiled)
- Shows error if initialization fails

#### **[compile]** - Compile Pipeline
- Runs pipeline compiler using current step list
- Updates status symbol to **o** (green) if successful (ready to run)

#### **[run]** - Execute Pipeline
- Runs the compiled pipeline on selected plate(s)
- Updates status symbol to **!** (red) during execution

### **PIPELINE EDITOR BUTTONS**

#### **[add]** - Add New Step
- Opens step creation dialog
- Allows selection of step type and configuration

#### **[del]** - Delete Step
- Removes currently selected step(s) from pipeline

#### **[edit]** - Edit Step
- Opens dual step/func editor (replaces plate manager pane)
- Allows editing of step parameters and function patterns

#### **[load]** - Load Pipeline
- Opens file dialog to load saved pipeline configuration

#### **[save]** - Save Pipeline
- Opens file dialog to save current pipeline configuration

## 🎨 VISUAL LAYOUT REFERENCE
```
__________________________________________________________________________________________
| |_[_Global_Settings_]_[_Help]_[Quit]_|_OpenHCS_V1.0__________________________________|
| |_____________Plate_Manager___________________|_|__Pipeline_Editor___________________|
|_|[add]_[del]_[edit]_[init]_[compile]_[run]___|_|[add]_[del]_[edit]_[load]_[save]_____|
|o| ^/v 1: axotomy_n1_03-02-24 | (...)/nvme_usb/    |o| ^/v 1: pos_gen_pattern          |
|!| ^/v 2: axotomy_n2_03-03-24 | (...)/nvme_usb/    |o| ^/v 2: enhance + assemble       |
|?| ^/v 3: axotomy_n3_03-04-24 | (...)/nvme_usb/    |o| ^/v 3: trace analysis           |
| |                                                  | |                                 |
| |                                                  | |                                 |
| |                                                  | |                                 |
| |                                                  | |                                 |
| |                                                  | |                                 |
|_|__________________________________________________|_|_________________________________|
|_Status:_...________________________________________|_|_________________________________|
```

### **STATUS SYMBOLS**
- **?** = Not initialized (gray)
- **-** = Initialized but not compiled (yellow)
- **o** = Compiled and ready (green)
- **!** = Running/Error (red)

## 🔄 DUAL STEP/FUNC EDITOR SYSTEM

### **ACTIVATION**
When clicking **[edit]** on a step in the pipeline editor, the **Plate Manager pane is replaced** with the dual editor.

### **EDITOR STRUCTURE**
- **Scrollable pane** with menu bar containing two toggle buttons
- **Step tab**: Abstract step parameters (all optional)
- **Func tab**: Function pattern configuration for FuncStep implementation
- **[save]** button: Grayed out until changes made, updates step with new configuration
- **[close]** button: Returns to Plate Manager view

### **STEP EDITOR FUNCTIONALITY**
- Initializes using static inspection of current FuncStep
- Displays all parameters from AbstractStep (inherited) and FuncStep
- Uses static reflection for dynamic UI generation
- All parameters are optional and configurable

### **FUNC EDITOR FUNCTIONALITY**
- Configures the single FuncStep parameter (callable enhancement)
- Supports multiple callable formats:
  - Single callable
  - (callable, dict kwargs) tuple
  - List of enhanced callables
  - Dict of any combination above
- Uses function register and static analysis for dropdown population

### **SAVE BEHAVIOR**
- **[save]** button grayed out initially
- Becomes available when any change is made
- Constructs new FuncStep with all collected parameters
- Returns to grayed out state after successful save
- Re-enables if further changes made

## 📋 STEP EDITOR LAYOUT
```
_________________________________________________
|_X_Step_X_|___Func___|_[save]__[close]__________| ← Tab bar with save/close
|^|_Step_Settings_Editor__[load]_[save_as]_______| ← Settings with file operations
|X| [reset] Name: [...]                          | ← Step name (optional)
|X| [reset] input_dir: [...]                     | ← Input directory selector
|X| [reset] output_dir: [...]                    | ← Output directory selector
|X| [reset] force_disk_output: [ ]               | ← Checkbox for disk output
|X| [reset] variable_components: |site|V|        | ← Enum dropdown
|X| [reset] group_by: |channel|V|                | ← Enum dropdown
|X|                                              |
|X|                                              |
|X|                                              |
|X|                                              |
|X|                                              |
|X|______________________________________________|
```

### **STEP PARAMETERS (All Optional)**
- **Name**: Text field for step identification
- **input_dir**: File dialog within orchestrator plate_path
- **output_dir**: File dialog within orchestrator plate_path
- **force_disk_output**: Boolean checkbox
- **variable_components**: Enum dropdown (DEFAULT_VARIABLE_COMPONENTS)
- **group_by**: Enum dropdown (DEFAULT_GROUP_BY)
- **[reset]** buttons: Reset individual parameters to None/default
- **[load]/[save_as]**: Load/save pickled .step objects via file dialog


## 🔧 FUNC EDITOR LAYOUT
```
_________________________________________________
|___Step___|_X_Func_X_|_[save]__[close]_________| ← Tab bar with save/close
|^|_Func_Pattern_Editor_[add]_[load]_[save_as]__| ← Pattern editor with file operations
|X|_dict_keys:_|None|V|_+/-__|__[edit_in_vim]_?_| ← Dict key selector with vim editing
|X| Func 1: |percentile_stack_normalize|V|      | ← Function dropdown from register
|X| --------------------------------------------|
|X|  move  [reset] percentile_low:  0.1 ...     | ← Auto-generated parameter fields
|X|   /\   [reset] percentile_high: 99.9 ...    |
|X|   \/   [add]                                |
|X|        [delete]                             |
|X| ____________________________________________|
|X| Func 2: |n2v2|V|                            | ← Additional function
|X|---------------------------------------------|
|X|         [reset] random_seed: 42 ...         | ← Function-specific parameters
|X|   move  [reset] device: cuda ...            |
|X|    /\   [reset] blindspot_prob: 0.05 ...    |
|X|    \/   [reset] max_epochs: 10 ...          |
|X|         [reset] batch_size: 4 ...           |
|X|         [reset] patch_size: 64 ...          |
|X|         [reset] learning_rate: 1e-4 ...     |
|X|         [reset] save_model_path: ...        |
|X|         [reset_all]                         |
|V|         [add]                               |
| |         [delete]                            |
| | ____________________________________________|
| | Func 3: |3d_deconv|V|                       | ← Scrollable for more functions
| |         [reset] random_seed:  42 ...        |
| |    move [reset] device: cuda  ...           |
| |     /\  [reset] blindspot_prob: 0.05 ...    |
|_| vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv| ← Scroll indicator
```

### **FUNC PATTERN FEATURES - VISUAL FUNCTION PATTERN BUILDER**

**🎯 CORE CONCEPT**: The func editor is a **visual programming interface** for building processing pipelines. It automatically discovers functions and generates UI from their signatures - no hardcoded forms!

#### **🔍 FUNCTION DISCOVERY & INTROSPECTION**
- **FUNC_REGISTRY**: Global registry `Dict[str, List[Callable]]` organized by memory type
  - **Structure**: `{"numpy": [func1, func2], "cupy": [func3], "torch": [func4, func5]}`
  - **Memory types**: `["numpy", "cupy", "torch", "tensorflow", "jax"]`
  - **Auto-discovery**: Functions decorated with `@memory_types(input_type="numpy", output_type="numpy")`
- **Function metadata**: Each registered function has attributes:
  - `func.input_memory_type` (e.g., "numpy")
  - `func.output_memory_type` (e.g., "numpy")
  - `func.backend` (same as input_memory_type)
- **Signature introspection**: Uses `inspect.signature(func).parameters` to discover:
  - Parameter names, defaults, type hints, required vs optional
  - Special parameters (those in function's special I/O schema)
- **Registry access**: `get_functions_by_memory_type("numpy")` returns list of numpy functions

#### **🏗️ FUNCTION PATTERN DATA STRUCTURES**
Function patterns are the core data structure for configuring processing pipelines:

1. **Single Function**: `func = some_processing_function`
2. **Function with Parameters**: `func = (some_processing_function, {'sigma': 2.0, 'contrast': 1.8})`
3. **List of Functions (Sequential)**: `func = [gaussian_blur, (contrast_enhance, {'factor': 1.5}), edge_detection]`
4. **Dict of Functions (Component-specific)**: `func = {'channel_1': gaussian_blur, 'channel_2': (edge_detection, {'threshold': 0.5})}`

#### **🎨 UI GENERATION PROCESS**
1. **Memory type selection**: Choose target backend (numpy/cupy/torch/tensorflow/jax)
2. **Function dropdowns**: Populated from `FUNC_REGISTRY[memory_type]`
   - Display: `func.__name__` (e.g., "gaussian_blur", "contrast_enhance")
   - Grouped by backend for organization
3. **Parameter fields**: Auto-generated from `inspect.signature(func).parameters`
   - **Text fields**: For string/numeric parameters
   - **Checkboxes**: For boolean parameters
   - **Dropdowns**: For enum parameters or limited choices
4. **Default values**: Extracted from parameter defaults in function signatures
5. **Validation**: Uses `FuncStepContractValidator` to ensure:
   - Memory type compatibility between chained functions
   - Required parameters are provided
   - Pattern structure is valid for execution

#### **🔧 INTERACTIVE FEATURES**
- **dict_keys dropdown**:
  - **None** = Single function or sequential list (no component grouping)
  - **Component keys** = Component-specific functions (e.g., 'channel_1', 'site_A', 'timepoint_0')
  - **Switching keys** changes which function list is being edited
- **Function dropdowns**: Pick from registry, automatically grouped by backend/memory type
- **Parameter fields**: Dynamically generated from function signatures with live validation
- **Move buttons**: Reorder functions in sequential lists (up/down arrows)
- **[add]/[delete]**: Build complex nested patterns dynamically
- **[reset]/[reset_all]**: Restore parameter defaults from function signatures
- **[edit_in_vim]**: External editor integration for complex pattern editing
- **[load]/[save_as]**: Serialize/deserialize patterns to/from .func files

#### **💡 THE GENIUS**:
This is **visual programming for image processing pipelines** where:
- **No hardcoded UI forms** - everything discovered dynamically
- **Function signatures drive UI** - add new functions, UI updates automatically
- **Complex data structures built visually** - lists, dicts, tuples via drag-and-drop
- **Type safety enforced** - contract validator ensures memory type compatibility
- **Serializable patterns** - save/load complex configurations as files

## ✅ IMPLEMENTATION REQUIREMENTS

### **🚨 CRITICAL FIXES NEEDED (IMMEDIATE)**
1. **Remove broken `class Button(Button)` from file_browser.py** - This causes infinite recursion
2. **Revert all SafeButton changes** - They break the UI rendering
3. **Use FramedButton for top bar** - As specified in layout_components.py
4. **Implement proper text sanitization** - Only where original formatting errors occur
5. **Ensure InteractiveListItem works** - For plate and step lists
6. **Test button rendering** - All buttons must display text correctly

### **🏗️ ARCHITECTURAL IMPLEMENTATION (FOLLOW PLANS)**
Implement according to the **5 sequential plans** in `/plans/tui_hybrid/`:

1. **plan_SIMPLE_01_copy_missing_components.md** (6.5h) - Copy FramedButton + StatusBar from archive
2. **plan_SIMPLE_02_create_layout_components.md** (12h) - Create exact canonical layout structure
3. **plan_SIMPLE_03_fix_status_symbols.md** (8h) - Implement `?`/`-`/`o`/`!` progression
4. **plan_SIMPLE_04_fix_file_browser.md** (11h) - Orchestrator integration + multi-folder selection
5. **plan_SIMPLE_05_integration_testing.md** (14h) - End-to-end verification

**Total: ~51.5 hours of systematic implementation**

### **🔍 FUNCTION PATTERN EDITOR IMPLEMENTATION**
- **Use existing `function_pattern_editor.py`** - Already implements the visual programming interface
- **Leverage `FUNC_REGISTRY`** - Functions auto-discovered by memory type
- **Use `inspect.signature()`** - Parameter fields auto-generated from function signatures
- **Implement `FuncStepContractValidator`** - Ensures pattern correctness and memory type compatibility
- **Support all 4 pattern types** - Single function, function+kwargs, list, dict
- **Dynamic UI generation** - No hardcoded forms, everything discovered from function metadata

### **🎯 ORCHESTRATOR INTEGRATION REQUIREMENTS**

#### **ORCHESTRATOR LIFECYCLE & STATUS PROGRESSION**
The TUI manages `PipelineOrchestrator` instances with a clear 4-state progression:

1. **CREATION** (`?` gray): `orchestrator = PipelineOrchestrator(plate_path, global_config=config)`
2. **INITIALIZATION** (`-` yellow): `orchestrator.initialize()` → workspace setup + microscope handler
3. **COMPILATION** (`o` green): `compiled_contexts = orchestrator.compile_pipelines(pipeline_definition)`
4. **EXECUTION** (`!` red): `results = orchestrator.execute_compiled_plate(pipeline_definition, compiled_contexts)`

#### **BUTTON IMPLEMENTATIONS**
- **[add]**: Multi-folder selection → create `PipelineOrchestrator(plate_path)` → status `?`
- **[init]**: Call `orchestrator.initialize()` → status `?` → `-` (yellow) if successful
- **[compile]**: Call `orchestrator.compile_pipelines(pipeline_definition)` → status `-` → `o` (green)
  - **Note**: `pipeline_definition` is `List[FunctionStep]` (TUI only uses FunctionStep, though method accepts List[AbstractStep])
- **[run]**: Call `orchestrator.execute_compiled_plate()` → status `o` → `!` (red during execution)
- **[edit]**: Static reflection editor for `orchestrator.global_config` (GlobalPipelineConfig)

#### **PIPELINE STEP MANAGEMENT**
- **Pipeline steps**: List of `FunctionStep` objects (concrete implementation, not abstract base)
- **Step creation**: `FunctionStep(func=pattern, name=name, variable_components=components, group_by=group_by)`
- **Function patterns**: The `func` parameter uses the 4 pattern types (single, tuple, list, dict)
- **Step persistence**: Save/load as pickled `.step` files via file dialogs
- **Future extensibility**: `AbstractStep` exists for future concrete step types, but pipelines currently use only `FunctionStep`

#### **ERROR HANDLING & STATE MANAGEMENT**
- **Error Recovery Rule**: DON'T update status flag if operation fails - keep at previous state
- **Initialization errors**: Show in status bar + modal dialog, keep status at `?` (gray)
- **Compilation errors**: Show detailed error, keep status at `-` (yellow)
- **Execution errors**: Show error details, keep status at `o` (green, ready to retry)
- **Error Display**: Both status bar message AND modal dialog with OK button
- **State persistence**: TUI tracks orchestrator state independently of orchestrator internal state

### **📋 VERIFICATION CHECKLIST**
- [ ] **UI Layout**: Top bar shows `[Global Settings] [Help] | OpenHCS V1.0`
- [ ] **Title Bar**: Shows `Plate Manager | Pipeline Editor`
- [ ] **Button Bars**: Show proper button text (not "Window to Window...")
- [ ] **Interactive Lists**: Display plate/step entries with status symbols
- [ ] **Status Symbols**: `?`/`-`/`o`/`!` appear in left column with correct colors
- [ ] **Function Editor**: Dropdowns populated from FUNC_REGISTRY
- [ ] **Parameter Fields**: Auto-generated from function signatures
- [ ] **Pattern Building**: Can create all 4 pattern types visually
- [ ] **Orchestrator Integration**: Buttons call actual orchestrator methods
- [ ] **File Operations**: Multi-folder selection, save/load patterns
- [ ] **Error Handling**: No crashes, proper error dialogs
- [ ] **Performance**: No infinite recursion, clean button rendering

### **🧠 CONTEXT PRESERVATION**
**This specification is CANONICAL and PERMANENT.** Any future work on the TUI must:
1. **Reference this document first** - Before making any changes
2. **Follow the sequential plans** - Don't skip steps or make assumptions
3. **Understand the function pattern system** - Visual programming interface concept
4. **Respect the orchestrator integration** - Status symbols reflect actual state
5. **Maintain architectural integrity** - No shortcuts or band-aid fixes

**The TUI is a visual programming interface for image processing pipelines, not just a generic UI. The function pattern editor is the crown jewel - it automatically discovers functions and builds UI from their signatures. This is the mental model that must be preserved.**

## 🏗️ ARCHITECTURAL UNDERSTANDING

### **CORE DATA FLOW**
1. **Plate Creation**: User selects folders → `PipelineOrchestrator(plate_path)` created
2. **Initialization**: `orchestrator.initialize()` → workspace setup + microscope handler discovery
3. **Pipeline Building**: User creates `FunctionStep` objects with function patterns via visual editor
4. **Compilation**: `orchestrator.compile_pipelines(steps)` → creates frozen `ProcessingContext` per well
5. **Execution**: `orchestrator.execute_compiled_plate()` → parallel execution across wells

### **KEY ARCHITECTURAL COMPONENTS**

#### **PipelineOrchestrator** (Core Engine)
- **Purpose**: Manages entire pipeline lifecycle for a single plate
- **Key Methods**:
  - `__init__(plate_path, global_config)` - Create orchestrator for plate
  - `initialize()` - Setup workspace, microscope handler, file manager
  - `compile_pipelines(steps)` - Create execution plans for all wells
  - `execute_compiled_plate()` - Run pipeline across wells in parallel
- **State**: Tracks initialization status, not execution status (TUI tracks that)

#### **FunctionStep** (Pipeline Unit)
- **Purpose**: The ONLY concrete step type currently used in pipelines (wraps function patterns)
- **Constructor**: `FunctionStep(func=pattern, name=name, variable_components=['site'], group_by='channel')`
- **Function Patterns**: The `func` parameter accepts all 4 pattern types
- **Pipeline Usage**: All pipeline lists contain `List[FunctionStep]`, not `List[AbstractStep]`
- **Execution**: Stateless during execution - all config comes from `ProcessingContext.step_plans`

#### **FUNC_REGISTRY** (Function Discovery)
- **Purpose**: Auto-discovery of available processing functions
- **Structure**: `{"numpy": [func1, func2], "cupy": [func3], ...}`
- **Population**: Functions decorated with `@memory_types(input_type="numpy", output_type="numpy")`
- **Access**: `get_functions_by_memory_type("numpy")` for TUI dropdowns

#### **ProcessingContext** (Execution State)
- **Purpose**: Immutable execution context for each well
- **Lifecycle**: Created → populated → frozen → used for execution
- **Contents**: `step_plans` dict, `filemanager`, `microscope_handler`, `global_config`
- **Immutability**: Frozen after compilation, read-only during execution

#### **GlobalPipelineConfig** (System Configuration)
- **Purpose**: System-wide settings (num_workers, VFS config, path planning)
- **Immutability**: Frozen dataclass, replaced entirely when changed
- **Usage**: Passed to orchestrator constructor, affects all operations

### **CRITICAL INTEGRATION POINTS**
1. **Multi-folder selection** → Multiple `PipelineOrchestrator` instances
2. **Function pattern editor** → Builds `FunctionStep.func` patterns
3. **Step/Func dual editor** → Edits `AbstractStep` + `FunctionStep` parameters
4. **Status progression** → TUI tracks orchestrator state transitions
5. **Error handling** → Catch orchestrator exceptions, show in UI, maintain state
6. **Config editing** → Modify `GlobalPipelineConfig`, apply to orchestrators

### **EXECUTION MODEL**
- **Two-phase**: Compile-all-then-execute-all (not step-by-step)
- **Parallelization**: Multiple wells processed simultaneously
- **State isolation**: Each well has independent `ProcessingContext`
- **Error isolation**: Well failures don't affect other wells
- **Memory management**: VFS handles intermediate results, configurable backends

## 🔧 CRITICAL IMPLEMENTATION PATTERNS

### **FunctionStep Construction**
```python
# When saving from dual editor (Step + Func):
function_step = FunctionStep(
    func=pattern_editor_output,      # Object returned by FunctionPatternEditor
    name=step_name,                  # From step editor text field
    variable_components=step_components,  # From step editor list
    group_by=step_group_by,          # From step editor dropdown
    force_disk_output=step_force_disk,    # From step editor checkbox
    input_dir=step_input_dir,        # From step editor path field
    output_dir=step_output_dir       # From step editor path field
)
```

### **File Operations (Pickle)**
```python
# Loading .pipeline, .step, .func files:
with open(filepath, 'rb') as f:
    loaded_object = pickle.load(f)
    # Then assign to appropriate variable:
    # - .pipeline → List[FunctionStep] → pipeline_list
    # - .step → FunctionStep → individual step
    # - .func → function pattern → pattern_editor.current_pattern

# Saving follows same pattern with pickle.dump()
```

### **Error Recovery Pattern**
```python
# Button click handlers - DON'T update status flag if operation fails:
def handle_init_button():
    try:
        orchestrator.initialize()
        tui_state.orchestrator_status[plate_path] = "-"  # yellow - only if successful
    except Exception as e:
        show_error_message(f"Initialization failed: {str(e)}")
        # Status remains "?" (gray) - DON'T update on failure

def handle_compile_button():
    try:
        orchestrator.compile_pipelines(pipeline_definition)  # List[FunctionStep]
        tui_state.orchestrator_status[plate_path] = "o"  # green - only if successful
    except Exception as e:
        show_error_message(f"Compilation failed: {str(e)}")
        # Status remains "-" (yellow) - DON'T update on failure
```

### **Multi-Folder Orchestrator Creation**
```python
def handle_add_plates_button():
    selected_folders = file_dialog.select_multiple_directories()
    for folder_path in selected_folders:
        orchestrator = PipelineOrchestrator(folder_path, global_config)
        tui_state.orchestrators[folder_path] = orchestrator
        tui_state.orchestrator_status[folder_path] = "?"  # gray - created but not initialized
        tui_state.current_pipelines[folder_path] = []     # empty List[FunctionStep]
```

