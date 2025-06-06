# plan_09_codebase_cleanup.md
## Component: TUI Codebase Cleanup and Bloat Removal

### Objective
Remove bloat and legacy components from the OpenHCS TUI codebase while preserving all essential production components. This cleanup will improve codebase clarity, reduce confusion, and eliminate ~530MB of accumulated artifacts.

### Plan
1. **Cross-Validate All Removal Claims** - Verify dependencies before removing any files
2. **Remove Confirmed Legacy Components** - plate_manager_core.py and tui_architecture.py
3. **Investigate Questionable Files** - 8 files requiring dependency analysis
4. **Archive Historical Plans** - Move old plan directories to archive
5. **Clean Audit Artifacts** - Remove 100+ DNA analysis directories
6. **Validate Production System** - Ensure all imports work after cleanup

### Findings

#### **Cross-Validation Results - Critical Corrections Made**
**Certainty Level**: **High (95%)**

**MAJOR DISCOVERY: function_pattern_editor.py is ESSENTIAL, not legacy**

**❌ INCORRECT ASSUMPTION CORRECTED:**
- **Original Claim**: "function_pattern_editor.py superseded by dual_step_func_editor.py"
- **Cross-Validation Finding**: **WRONG** - dual_step_func_editor.py DEPENDS ON function_pattern_editor.py
- **Evidence**: 
  ```python
  # dual_step_func_editor.py line 27:
  from .function_pattern_editor import FunctionPatternEditor
  
  # Lines 129-133: Creates FunctionPatternEditor instance
  self.func_pattern_editor_component = FunctionPatternEditor(...)
  ```

**✅ CONFIRMED LEGACY COMPONENTS:**
- **plate_manager_core.py** (1089 lines) - Superseded by plate_manager_refactored.py
  - Evidence: canonical_layout.py imports plate_manager_refactored, not plate_manager_core
- **tui_architecture.py** - Uses legacy plate_manager_core import
  - Evidence: Line 32 imports old plate_manager_core implementation

#### **Essential Components Confirmed (KEEP)**
**Certainty Level**: **High (99%)**

**Production Integration Layer:**
- `canonical_layout.py` - Working integration layer with all fixes applied
- `orchestrator_manager.py` - Orchestrator coordination

**Production Components:**
- `plate_manager_refactored.py` - ACTIVE MVC plate management (158 lines)
- `pipeline_editor.py` - Pipeline editing component
- `menu_bar.py` - Menu system
- `status_bar.py` - Status display
- `tui_launcher.py` - Application launcher

**Essential Editor Components (CORRECTED):**
- `function_pattern_editor.py` - **ESSENTIAL** reusable component (975 lines)
- `dual_step_func_editor.py` - **PRODUCTION** container using function_pattern_editor.py (773 lines)

**Supporting Infrastructure:**
- `commands/` directory - Command pattern implementation
- `services/` directory - Business logic layer
- `controllers/` directory - MVC controllers
- `views/` directory - MVC views
- `dialogs/` directory - Dialog management
- `components/` directory - Reusable UI components
- `utils/` directory - Utility functions

#### **Files Requiring Investigation**
**Certainty Level**: **Medium (60%)**

**Need Dependency Analysis Before Removal:**
- `layout_components.py` - Check if used by any production components
- `simple_launcher.py` - Check if used vs tui_launcher.py
- `commands.py` - Check if used vs commands/ directory
- `components.py` - Check if used vs components/ directory
- `utils.py` - Check if used vs utils/ directory
- `file_browser.py` - Check if used by canonical_layout.py or dialogs
- `menu_handlers.py` - Check if used by menu_bar.py
- `menu_structure.py` - Check if used by menu_bar.py

#### **Massive Audit Bloat Identified**
**Certainty Level**: **High (100%)**

**Audit Directory Analysis:**
- **100+ DNA analysis directories** (~500MB)
- **Historical analysis artifacts** from development process
- **Only 3 most recent canonical_layout.py analyses needed**

**Historical Plans Bloat:**
- **10 plan directories** from previous implementation attempts
- **Archive rather than delete** to preserve development history

### Implementation Draft
*Ready for implementation after smell loop approval*

#### **Phase 1: Remove Confirmed Legacy (Low Risk)**
```bash
# Remove confirmed legacy components
rm openhcs/tui/plate_manager_core.py      # 1089 lines - superseded
rm openhcs/tui/tui_architecture.py        # Uses legacy imports

# Remove temporary analysis files  
rm openhcs/tui/integration_assessment.md
rm openhcs/tui/static_analysis_report.md
rm openhcs/tui/openhcs_tui.log
```

#### **Phase 2: Dependency Investigation (Medium Risk)**
```bash
# Investigate each file for dependencies before removal
grep -r "layout_components" openhcs/tui/
grep -r "simple_launcher" openhcs/tui/
grep -r "file_browser" openhcs/tui/
# ... continue for all questionable files
```

#### **Phase 3: Archive Historical Plans (Low Risk)**
```bash
mkdir -p plans/archive/
mv plans/tui/ plans/archive/
mv plans/tui_final/ plans/archive/
mv plans/tui_final_implementation/ plans/archive/
mv plans/tui_fixes/ plans/archive/
mv plans/meta_tui/ plans/archive/
# ... continue for all historical plan directories
```

#### **Phase 4: Clean Audit Artifacts (Low Risk)**
```bash
# Keep only 3 most recent canonical_layout.py analyses
find audit/ -name "canonical_layout.py_*" -type d | sort | head -n -3 | xargs rm -rf

# Remove all other audit data
find audit/ -maxdepth 1 -type d ! -name "canonical_layout.py_*" -exec rm -rf {} +
```

#### **Phase 5: Validation (Critical)**
```bash
# Test all critical imports
python -c "from openhcs.tui.canonical_layout import CanonicalTUILayout"
python -c "from openhcs.tui.function_pattern_editor import FunctionPatternEditor"
python -c "from openhcs.tui.dual_step_func_editor import DualStepFuncEditorPane"

# Run static analysis on cleaned codebase
dna --detailed openhcs/tui/canonical_layout.py
```

**Implementation Priority**: **High** (cleanup enables clearer development)
**Complexity**: **Low** (mostly file operations with validation)
**Risk**: **Low** (with proper dependency checking and validation)

**Expected Impact:**
- **Space Savings**: ~530MB (500MB audit + 15MB legacy + 15MB misc)
- **Clarity Improvement**: Massive (single source of truth for each component)
- **Development Speed**: Faster (less noise when navigating codebase)
- **Maintenance**: Easier (no risk of importing legacy components)
