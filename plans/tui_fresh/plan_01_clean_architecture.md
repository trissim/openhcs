# plan_01_clean_architecture.md
## Component: Clean TUI Architecture

### Objective
Build a minimal, working TUI that matches the canonical specification without over-engineering. Focus on **direct implementation** rather than abstract layers.

### Plan

#### **1. IMMEDIATE FIXES (2 hours)**
- **Remove mock plates** from `plate_manager_refactored.py` 
- **Fix MenuBar key binding error** - ensure `self.kb` is properly initialized
- **Add missing pipeline editor buttons** - ensure button bar renders
- **Remove broken components** - eliminate infinite recursion and SafeButton issues

#### **2. SIMPLIFIED ARCHITECTURE (4 hours)**
Create a **single-file TUI** that implements the canonical layout:

```
simple_tui.py:
├── SimpleTUI class (main coordinator)
├── PlatePane class (left pane)
├── PipelinePane class (right pane)  
└── StatusPane class (bottom bar)
```

**Key Principles:**
- **No services/controllers/handlers** - direct method calls
- **No complex state management** - simple dictionaries
- **No dynamic containers** - static layout with content updates
- **No abstract base classes** - concrete implementations only

#### **3. CORE DATA MODEL (1 hour)**
```python
# Simple state management
tui_state = {
    'plates': [],  # List[dict] with name, path, status
    'current_plate': None,  # Currently selected plate
    'pipeline_steps': [],  # List[dict] with step data
    'current_step': None,  # Currently selected step
}
```

#### **4. BUTTON IMPLEMENTATIONS (3 hours)**
Direct orchestrator integration without abstraction layers:

**Plate Buttons:**
- `add_plate()` → file dialog → create orchestrator → add to list
- `init_plate()` → `orchestrator.initialize()` → update status
- `compile_plate()` → `orchestrator.compile_pipelines()` → update status
- `run_plate()` → `orchestrator.execute_compiled_plate()` → update status

**Pipeline Buttons:**
- `add_step()` → create FunctionStep → add to list
- `edit_step()` → open dual editor → update step
- `save_pipeline()` → pickle.dump(steps) → file dialog
- `load_pipeline()` → pickle.load() → update list

### Findings

#### **Current Architecture Problems:**
1. **MenuBar.py** (1,255 lines) - Massive over-engineering with complex key binding system
2. **canonical_layout.py** (756 lines) - Too many abstraction layers
3. **plate_manager_refactored.py** (500+ lines) - Unnecessary complexity for simple list management
4. **Mock data contamination** - Sample plates hardcoded in production code

#### **What Actually Works:**
- `InteractiveListItem` component for list rendering
- `FramedButton` for button display
- Basic prompt_toolkit layout containers (HSplit, VSplit, Frame)
- Direct orchestrator method calls (when not wrapped in abstractions)

#### **Critical Insights:**
- The canonical spec is **simple** - just 3 panes with buttons and lists
- Current implementation has **10x more complexity** than needed
- **Direct orchestrator calls work** - the abstraction layers are the problem
- **Static layout** is sufficient - no need for dynamic containers

### Implementation Draft

*[Code will be written here after smell loop approval]*
