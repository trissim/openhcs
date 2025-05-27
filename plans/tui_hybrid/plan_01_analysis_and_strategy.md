# plan_00_analysis_and_strategy.md
## Component: Analysis and Hybrid Strategy

### Objective
Define the systematic approach to merge TUI2's clean MVC architecture with TUI's working components, removing schema dependencies and completing placeholder implementations.

### Plan
1. Map salvageable components from TUI to TUI2 architecture
2. Identify schema removal requirements
3. Define component porting strategy
4. Create implementation phases with clear deliverables
5. Establish tracking system for progress

### Findings

**Architecture Decision: TUI2 MVC + TUI Components**

**TUI2 Architecture (Keep):**
```
controllers/
├── dual_editor_controller.py      ← Clean MVC pattern
├── pipeline_editor_controller.py  ← Controller separation
└── plate_manager_controller.py    ← State management

components/
├── step_settings_editor.py        ← Needs completion
├── func_pattern_view.py           ← Placeholder - needs TUI port
├── plate_list_view.py             ← Working
└── step_list_view.py              ← Working
```

**TUI Components (Port):**
```
function_pattern_editor.py         ← 909 lines - COMPLETE dict key mgmt
dual_step_func_editor.py          ← 616 lines - Working step/func toggle
components/parameter_editor.py     ← 200+ lines - Dynamic form generation
utils.py                          ← File dialogs, error handling
```

**Schema Removal Strategy:**
- Replace `ParamSchema` with `inspect.signature()` analysis
- Use `AbstractStep.__init__` introspection for step settings
- Use `FUNC_REGISTRY` + function signatures for func patterns
- Remove all TYPE_CHECKING schema imports

**Component Porting Strategy:**
1. **Direct Port**: Copy working TUI components to TUI2 structure
2. **Controller Integration**: Wrap ported components in TUI2 controllers
3. **Schema Elimination**: Replace schema calls with static analysis
4. **Interface Standardization**: Ensure consistent component interfaces

**Implementation Phases:**

**Phase 0: Analysis and Strategy** (This plan)
- Component mapping and architecture decisions
- Schema removal strategy
- Implementation roadmap

**Phase 1: Foundation Setup**
- Create hybrid folder structure
- Port core utilities (dialogs, error handling)
- Establish component interfaces

**Phase 2: Function Pattern Editor Port**
- Port `function_pattern_editor.py` to TUI2 components
- Remove schema dependencies
- Integrate with `DualEditorController`

**Phase 3: Step Settings Editor Completion**
- Complete `step_settings_editor.py` using TUI's parameter editor patterns
- Use `AbstractStep` introspection instead of schema
- Dynamic form generation

**Phase 4: Controller Integration**
- Update controllers to use ported components
- Remove schema dependencies from controllers
- Ensure proper state management

**Phase 5: Static Analysis and Cleanup**
- Static analysis validation using meta tools
- Error reduction through static analysis
- Architecture purity verification
- Documentation

**Phase 6: Integration Testing (Final)**
- End-to-end testing only after static validation
- Performance verification
- Final cleanup

### Implementation Draft
(Only after smell loop passes)
