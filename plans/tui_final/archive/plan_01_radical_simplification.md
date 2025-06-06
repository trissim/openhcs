# plan_01_radical_simplification.md
## Component: Radical TUI Simplification - Preserve Visual Programming, Delete Over-Architecture

### Objective
Eliminate over-architected MVC/Command/Bridge layers while preserving the excellent visual programming components (FunctionPatternEditor, DualStepFuncEditorPane, FUNC_REGISTRY) and implementing simple, direct orchestrator integration that matches the tui_final.md specification.

### Plan - STATIC ANALYSIS DRIVEN IMPLEMENTATION

**🧠 CORE MINDSET: Static Analysis > Testing**
- **NO TESTING until absolute end** - rely exclusively on static analysis
- **Sidetracking encouraged** when foundational issues discovered
- **Architectural consistency** verified through code inspection
- **Import dependency tracking** to prevent broken references

#### **Phase 1: Safe Deletion via Static Analysis (45 minutes)**
- **Static import analysis** before each deletion batch
- Remove MVC plate management system (562 lines)
- Delete complex command system and bridges
- Remove redundant orchestrator management
- Delete monolithic dead code
- **Verify**: No broken imports, no missing dependencies

#### **Phase 2: State Enhancement via Interface Analysis (15 minutes)**
- Add `show_dialog()` method to SimpleTUIState
- **Observer Pattern Setup**: Connect state dialog events to layout dialog display
  - State emits `'show_dialog_requested'` events via `notify()`
  - Layout observes these events and calls `_show_dialog()` method
  - Leverages existing dialog infrastructure in canonical_layout.py
- **Static verification**: Visual programming components can instantiate
- **Interface contract check**: Required methods exist
- **Verify**: State interface matches component expectations

#### **Phase 3: Integration Verification via Dependency Analysis (30 minutes)**
- **Static analysis**: canonical_layout.py import chain
- **Code flow tracing**: Visual programming component loading
- **Registry analysis**: FUNC_REGISTRY population paths
- **Verify**: All dependencies satisfied, no circular imports

#### **Phase 4: Direct Handler Implementation via Code Flow Analysis (45 minutes)**
- Replace complex command system with direct handlers
- **Static integration**: DualStepFuncEditorPane into layout
- **Method call tracing**: Orchestrator operations
- **Verify**: Button handlers call correct orchestrator methods

#### **Phase 5: Architectural Consistency Verification (30 minutes)**
- **Static workflow analysis**: add plate → edit step → compile → run
- **Interface verification**: Visual programming integration
- **Error path analysis**: Exception handling flows
- **Verify**: Complete system coherence without execution

#### **Phase 6: Button Implementation Completion + Backend Compliance (60 minutes)**
- **Replace minimal stubs with functional implementations**
- **Low-hanging fruit first**: Use existing dialog utilities
- **Menu handlers**: Global Settings, Help, Exit
- **🚨 CRITICAL: Fix backend architecture violations**
- **Pipeline save/load**: Replace direct file I/O with FileManager abstraction
- **Step commands**: Replace notification-only with actual orchestrator method calls
- **VFS compliance**: Ensure all I/O uses proper backend abstraction
- **Fallback button handlers**: Quit, Global Settings, Help in canonical_layout.py
- **Verify**: All buttons use proper OpenHCS backend architecture

**Button Implementation Inventory:**
```
MINIMAL STUBS (Need Real Implementation):
- canonical_layout.py: _handle_quit, _handle_global_settings, _handle_help
- menu_structure.py: show_global_settings, show_help, exit_application
- menu_handlers.py: handle_global_settings, handle_help, handle_exit

EXISTING IMPLEMENTATIONS (🚨 BACKEND VIOLATIONS FOUND):
- pipeline_editor.py: Add, Delete, Edit, Load, Save buttons
  ❌ Save/Load: Direct file I/O bypasses FileManager abstraction
  ❌ Add/Delete: Notification-only, don't call orchestrator methods
- menu_bar.py: Menu activation and navigation ✅ (Working correctly)

BACKEND ARCHITECTURE VIOLATIONS TO FIX:
- Replace open()/pickle.dump() with filemanager.save()
- Replace state.notify() with orchestrator method calls
- Add proper VFS backend parameter handling
- Use ProcessingContext for all operations

AVAILABLE UTILITIES:
- dialog_helpers.py: show_error_dialog, prompt_for_path_dialog
- state.show_dialog(): Dialog integration infrastructure
- FileManager: Proper VFS abstraction for all I/O operations
```

## 🔒 PRESERVE EXISTING IMPLEMENTATIONS (Do NOT Reimplement)

**Pipeline Editor Button System (`openhcs/tui/pipeline_editor.py`):**
- **Add/Delete/Edit/Load/Save buttons** - Use `FramedButton` with command integration
- **Command classes**: `AddStepCommand`, `DeleteSelectedStepsCommand`, `ShowEditStepDialogCommand`, `LoadPipelineCommand`, `SavePipelineCommand`
- **Backend integration**: Commands call `state.notify()` with orchestrator operations
- **Step editing**: `_handle_edit_step_request()` activates `DualStepFuncEditorPane` (crown jewel!)
- **Pipeline save/load**: Direct `state.active_orchestrator.pipeline_definition` access

**Dialog Infrastructure (`openhcs/tui/utils/dialog_helpers.py`):**
- **`show_error_dialog(title, message, app_state)`** - Modal error dialogs using `state.show_dialog()`
- **`prompt_for_path_dialog(title, prompt, app_state)`** - Path input dialogs with validation
- **`SafeButton`** - Button component with proper error handling
- **Backend integration**: Uses `app_state.show_dialog()` method for consistent display

**Menu Bar System (`openhcs/tui/menu_bar.py`):**
- **Menu activation**: `_activate_menu()`, `_close_menu()` with mouse/keyboard handlers
- **Key bindings**: Alt+F, Alt+H navigation via `_create_key_bindings()`
- **Observer integration**: Responds to `operation_status_changed`, `plate_selected`, `is_compiled_changed`
- **Backend integration**: Menu handlers call `state.notify()` for orchestrator operations

**Visual Programming Components:**
- **`ParameterEditor` (`openhcs/tui/components/parameter_editor.py`)**: Function parameter editing with signature inspection, callbacks for `on_parameter_change`, `on_reset_parameter`
- **`GroupedDropdown` (`openhcs/tui/components/grouped_dropdown.py`)**: Category-grouped options with header separation
- **`FramedButton` (`openhcs/tui/components/framed_button.py`)**: Custom styled buttons with mouse handlers
- **Backend integration**: All connect to `FUNC_REGISTRY` for function discovery

**Function Registry System (`openhcs/processing/func_registry.py`):**
- **`FUNC_REGISTRY`** - Global singleton with 16 functions across 5 backends
- **`get_functions_by_memory_type(memory_type)`** - Backend-specific function access
- **Auto-initialization**: Scans processing directory and registers functions by backend
- **Backend integration**: Functions tagged with `.backend` attribute for identification

**State Dialog Integration (`openhcs/tui/simple_launcher.py`):**
- **`SimpleTUIState.show_dialog(dialog, result_future)`** - Async dialog display
- **Observer pattern**: Emits `'show_dialog_requested'` events
- **Layout integration**: `CanonicalTUILayout._handle_dialog_request()` displays dialogs

**Layout Dialog Display (`openhcs/tui/canonical_layout.py`):**
- **`_show_dialog(dialog_container)`** - FloatContainer dialog display
- **`_hide_dialog()`** - Dialog cleanup and layout restoration
- **Backend integration**: Integrates with application layout and invalidation system

#### **🚨 SIDETRACKING PROTOCOL**
**WHEN FOUNDATIONAL ISSUES DISCOVERED:**
1. **STOP current phase immediately**
2. **Analyze root cause** through static inspection
3. **Address foundational issue** before continuing
4. **Update plan** if architectural changes needed
5. **Resume original phase** only after foundation solid

#### **🔍 STATIC ANALYSIS TOOLKIT**
**Before Each Tool Call:**
- **State current phase** and specific objective
- **Declare architectural context** (what preserving vs changing)
- **Predict expected outcome** and verification criteria

**After Each Tool Call:**
- **Verify actual changes** against expectations
- **Trace import dependencies** for broken references
- **Check interface contracts** for missing methods
- **Assess architectural impact** on overall system

**Static Verification Methods:**
- **Import chain analysis** - Follow import statements to verify dependencies
- **Method signature inspection** - Ensure required interfaces exist
- **Code flow tracing** - Follow execution paths through static inspection
- **Dependency graph mapping** - Understand component relationships
- **Interface contract verification** - Check method/property requirements

### Findings

#### ✅ DEPENDENCY ANALYSIS COMPLETE - PLAN VERIFIED

**🔍 Critical Discovery**: Visual programming components are COMPLETELY INDEPENDENT of over-architecture!
- FunctionPatternEditor & DualStepFuncEditorPane: Only import core OpenHCS + prompt_toolkit
- Utils & Services: Self-contained, no MVC dependencies
- FUNC_REGISTRY: Auto-initialized in __main__.py, no TUI dependencies
- State Interface: Minimal requirements (notify() method + optional show_dialog())

**⚠️ Integration Fix Required**: canonical_layout.py imports plate_manager_refactored.py
- **REVISED PLAN**: Keep plate_manager_refactored.py as facade (158 lines)
- Delete the underlying MVC system it wraps
- Simplify its dependencies

**🎯 Confidence Level**: 95% - All major assumptions verified, plan is solid!

#### Files to DELETE (4,085+ lines eliminated):

**MVC Plate Management System (562 lines):**
- `services/plate_manager_service.py` (197 lines)
- `controllers/plate_manager_controller.py` (272 lines)
- `views/plate_manager_view.py` (251 lines)
- **KEEP**: `plate_manager_refactored.py` (158 lines) - Used by canonical_layout.py

**Complex Command System:**
- `commands/` directory (all files)
- `enhanced_compilation_command.py`
- Command registration and coordination code

**Redundant Orchestrator Management (523 lines):**
- `orchestrator_manager.py` (254 lines)
- `plate_orchestrator_bridge.py` (269 lines)

**Monolithic Dead Code (3,000+ lines):**
- `tui_architecture.py` (1,165 lines)
- `plate_manager_core.py` (1,000+ lines)
- `commands.py` (916 lines)
- `plate_manager_refactored.py` (158 lines)

**Complex Infrastructure:**
- Most of `components/`, `controllers/`, `dialogs/`, `services/`, `utils/`, `views/`
- Event coordination and observer pattern implementations
- Complex async initialization systems

#### Files to KEEP (Excellent Visual Programming Code):

**🎯 CROWN JEWEL - Visual Programming Interface (1,400+ lines of sophisticated code):**
- `function_pattern_editor.py` (800+ lines) - Auto-discovers functions, generates UI from signatures
- `dual_step_func_editor.py` (600+ lines) - Complete dual Step/Func editor with tabs
- `func_registry.py` - Function discovery system with FUNC_REGISTRY
- Supporting components: ParameterEditor, GroupedDropdown, ExternalEditorService

**Core Layout & State (simplified):**
- `canonical_layout.py` (756 lines) - Working 3-bar layout, uses production components
- `simple_launcher.py` (SimpleTUIState) - Clean state management with observer pattern
- `plate_manager_refactored.py` (158 lines) - Facade wrapper (required by canonical_layout)

**Production UI Components:**
- `menu_bar.py` - Top bar with [Global Settings] [Help]
- `status_bar.py` - Bottom status display
- `pipeline_editor.py` - Pipeline step management
- Basic interactive list components

#### New Simple Architecture:

**Enhanced State Management (100 lines total):**
```python
class SimpleTUIState:
    def __init__(self):
        # Current properties (already exist)
        self.selected_plate = None
        self.selected_step = None
        self.is_compiled = False
        self.is_running = False
        self.error_message = None
        self.current_pipeline_definition = []

        # New properties for enhanced functionality
        self.orchestrators: Dict[str, PipelineOrchestrator] = {}
        self.status: Dict[str, str] = {}  # "?", "-", "o", "!"
        self.pipelines: Dict[str, List[FunctionStep]] = {}
        self.global_config: GlobalPipelineConfig = default_config()

        # Observer pattern (already exists)
        self.observers = {}

    async def notify(self, event, data):
        # Already implemented

    async def show_dialog(self, dialog, result_future):
        # Emit dialog event for layout to handle
        await self.notify('show_dialog_requested', {
            'dialog': dialog,
            'result_future': result_future
        })
```

**Dialog Integration Pattern:**
```python
# In CanonicalTUILayout.__init__():
self.state.add_observer('show_dialog_requested', self._handle_dialog_request)

# In CanonicalTUILayout:
async def _handle_dialog_request(self, event_data):
    dialog = event_data['dialog']
    result_future = event_data['result_future']

    # Show dialog using existing infrastructure
    self._show_dialog(dialog)

    # Wait for dialog completion and hide
    await result_future
    self._hide_dialog()
```

**Button Handlers (300 lines total):**
```python
def handle_add_plates():
    folders = file_dialog.select_multiple_directories()
    for folder in folders:
        orchestrator = PipelineOrchestrator(folder, state.global_config)
        state.orchestrators[folder] = orchestrator
        state.status[folder] = "?"
        state.pipelines[folder] = []

def handle_init_plates():
    for plate_path in get_selected_plates():
        try:
            state.orchestrators[plate_path].initialize()
            state.status[plate_path] = "-"
        except Exception as e:
            show_error(f"Init failed for {plate_path}: {e}")

def handle_compile_plates():
    for plate_path in get_selected_plates():
        try:
            pipeline = state.pipelines[plate_path]  # List[FunctionStep]
            state.orchestrators[plate_path].compile_pipelines(pipeline)
            state.status[plate_path] = "o"
        except Exception as e:
            show_error(f"Compile failed for {plate_path}: {e}")

def handle_edit_step(step_index):
    # Use existing DualStepFuncEditorPane!
    func_step = state.pipelines[state.selected_plate][step_index]
    editor = DualStepFuncEditorPane(state, func_step)
    # Replace plate manager pane with editor
    layout.replace_left_pane(editor.container)
```

#### Architecture Comparison:

**BEFORE (Over-Architected):**
```
User clicks [init] 
  → Command system parameter passing
  → OrchestratorManager.get_selected()
  → PlateOrchestratorCoordinationBridge coordination
  → Event emission and observer notification
  → Complex error handling coordination
  → Status update through multiple layers
```

**AFTER (Direct & Simple with Visual Programming):**
```
User clicks [init]
  → handle_init_plates()
  → orchestrator.initialize()
  → status[plate] = "-"
  → UI refresh

User clicks [edit] on step
  → handle_edit_step()
  → DualStepFuncEditorPane(state, func_step)
  → FunctionPatternEditor auto-discovers from FUNC_REGISTRY
  → Parameter fields generated from function signatures
  → Visual pipeline building interface
```

### Implementation Draft
*Implementation will be added after smell loop approval*
