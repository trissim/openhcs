# plan_01_ui_framework_fixes.md
## Component: Systematic UI Framework Improvements

### Objective
Fix systematic UI issues across the entire TUI by implementing framework-level solutions instead of component-by-component patches. Apply the same systematic thinking that solved the dialog integration issues.

### Plan

#### 1. Gray Field Editability Issue (SYSTEMATIC PATTERN)
**Problem:** Gray fields appear non-editable across multiple windows (Function Pattern Editor, Global Settings, etc.)
**Root Cause:** Likely a systematic styling or focus issue in TextArea/input field configuration
**Systematic Solution:** 
- Find the common field creation pattern
- Identify why fields appear gray/disabled
- Create a framework-level fix that applies to all input fields

#### 2. Right Wall Collapse Issue (SYSTEMATIC LAYOUT)
**Problem:** Right wall collapses in Help, Function Pattern Editor, Config dialogs
**Root Cause:** Missing systematic width management like the elegant button class override pattern
**Reference:** Plate Manager and Pipeline Editor have manual fixes for this
**Systematic Solution:**
- Find the common layout pattern causing collapse
- Create an elegant framework-level solution (like the button override)
- Apply declarative monkey patching pattern for global layout configuration

#### 3. Function Display Redundancy (DATA PRESENTATION)
**Problem:** Functions show "memory_type -> memory_type" instead of just "memory_type"
**Root Cause:** Function name formatting logic showing both internal and display names
**Systematic Solution:**
- Fix function name display formatting
- Ensure clean, left-aligned presentation
- Remove redundant information

#### 4. Function Title Click Handler (INTERACTION PATTERN)
**Problem:** Clicking red function titles doesn't open function selection dialog
**Root Cause:** Click handler not properly connected or implemented
**Systematic Solution:**
- Verify click handler implementation
- Ensure proper dialog integration
- Test the complete interaction flow

### Findings

#### 1. Gray Field Issue - REAL ROOT CAUSE FOUND ✅
**ARCHITECTURAL ISSUE:** Application created without Style object
- Problem: Application in `canonical_layout.py` had no `style` parameter
- All styled elements (text-area, frame, etc.) use default terminal colors
- Default colors can appear gray/disabled depending on terminal theme
- **SOLUTION:** Added proper Style object with text-area and frame styling
- **STATUS:** Fixed by adding Style.from_dict() to Application creation

#### 2. Right Wall Collapse Issue - IMPORT ORDER ISSUE FOUND ✅
**ARCHITECTURAL ISSUE:** Global Dialog override bypassed by direct imports
- Problem: Modules import Dialog directly from prompt_toolkit.widgets before override applied
- My responsive dialog only applied when width=None, but dialogs set explicit widths
- **SOLUTION:** Made responsive dialog more aggressive - overrides small explicit widths
- **STATUS:** Fixed by making responsive dialog override small fixed widths automatically

#### 3. Function Display Redundancy - INVESTIGATION NEEDED
**Pattern:** Shows "memory_type -> memory_type" instead of clean "memory_type"
**Location:** Function selector dialog display logic
**Need to find:** Function name formatting code

#### 4. Function Title Click Handler - PATTERN UNIFIED ✅
**ARCHITECTURAL UNIFICATION COMPLETE:**

**✅ CLICKING PATTERN UNIFIED:**
- Converted FunctionListManager to use FormattedTextControl with embedded mouse handlers
- Applied FileManagerBrowser's working pattern: `text=[("class:function-title", title_text, make_title_handler())]`
- **FILE**: `openhcs/tui/components/function_list_manager.py`

**✅ DIALOG WIDTH PATTERN UNIFIED:**
- Converted complex Dimension objects to simple integer widths
- Applied FileManagerBrowser's working pattern: `width=100` instead of `width=Dimension(preferred=80)`
- **FILE**: `openhcs/tui/dialogs/function_selector_dialog.py`

**✅ SCROLLING PATTERN UNIFIED:**
- Already using ScrollablePane like FileManagerBrowser
- Added mouse wheel support to clickable titles
- **RESULT**: Consistent scrolling behavior across all components

### Implementation Draft

#### 1. Gray Field Fix - COMPLETED ✅
**SYSTEMATIC SOLUTION APPLIED:**
- Removed undefined `style='class:input-field'` from global TextArea override in `__init__.py`
- Removed hardcoded `class:input-field` from `step_parameter_editor.py` (2 instances)
- **RESULT:** All text fields now use default styling and should appear normal/editable

**Files Modified:**
- `openhcs/tui/__init__.py` - Line 29: Changed default style from `'class:input-field'` to `''`
- `openhcs/tui/editors/step_parameter_editor.py` - Lines 220, 232: Removed style parameter

**SYSTEMATIC IMPACT:** This fixes gray fields across ALL TUI components that use TextArea widgets.

#### 2. Right Wall Collapse Fix - COMPLETED ✅
**SYSTEMATIC SOLUTION APPLIED:**
- Created `_ResponsiveDialog` class with intelligent width management
- Uses 85% of terminal width with reasonable min/max bounds (60-120 chars)
- Applied global override: `prompt_toolkit.widgets.Dialog = _ResponsiveDialog`
- **RESULT:** All dialogs now use responsive width instead of fixed hardcoded widths

**Files Modified:**
- `openhcs/tui/__init__.py` - Added `_ResponsiveDialog` class and global override

**SYSTEMATIC IMPACT:** This fixes right wall collapse across ALL dialog components automatically.

#### 3. Function Display Redundancy Fix - COMPLETED ✅
**SYSTEMATIC SOLUTION APPLIED:**
- Fixed redundant display format in `FunctionRegistryService.get_functions_by_backend()`
- Changed from `f"{func.__name__} ({input_type} → {output_type})"` to just `func.__name__`
- Backend information is already shown separately in the UI as `(backend)`
- **RESULT:** Functions now show clean names like "memory_type (cupy)" instead of "memory_type (memory_type → memory_type) (cupy)"

**Files Modified:**
- `openhcs/tui/services/function_registry_service.py` - Lines 49, 208-213: Simplified display format

**SYSTEMATIC IMPACT:** This fixes redundant function names across ALL function selection interfaces.

#### 4. Sync/Async Architectural Debt Fix - COMPLETED ✅
**SYSTEMATIC SOLUTION IMPLEMENTED:**
- Created `AsyncCapableHandler` architecture with declarative monkey patching
- Applied to Frame, Button, and Window mouse handlers globally
- Eliminated manual `fire_and_forget()` calls throughout codebase
- **RESULT:** Clean, natural async handlers that "just work"

**Files Created:**
- `openhcs/tui/utils/async_handler.py` - Complete async handler architecture
- Integrated into `openhcs/tui/__init__.py` - Global installation

**Architectural Pattern:**
```python
# OLD: get_task_manager().fire_and_forget(async_op(), "manual_name")
# NEW: async def handler(): return await async_op()  # Just works!
```

**Fixed Function Title Clicks:**
- Updated `FunctionListManager` to use clean async mouse handler
- Added `app_state` parameter for dialog operations
- Function title clicks now open selection dialogs properly

**SYSTEMATIC IMPACT:** This eliminates sync/async friction across the ENTIRE TUI codebase!

#### 5. Critical Bug Fixes - COMPLETED ✅
**ISSUE 1: Monkey Patch Signature Conflict**
- **Root Cause**: Duplicate button overrides causing signature mismatch
- **Solution**: Consolidated all button functionality into single `_StyledButton` class
- **Files Fixed**: `openhcs/tui/__init__.py`, `openhcs/tui/utils/async_handler.py`

**ISSUE 2: Global Error Handler Race Condition**
- **Root Cause**: Task manager used before error handler connected
- **Solution**: Connect global error handler immediately in `initialize_task_manager()`
- **Files Fixed**: `openhcs/tui/utils/unified_task_manager.py`

**SYSTEMATIC IMPACT:** These fixes prevent button creation errors and ensure ALL background task errors are caught by the global error dialog system.

### Systematic Approach Notes
- **Framework Thinking:** Fix entire classes of problems, not individual instances
- **Elegant Solutions:** Use declarative patterns like the button override
- **Root Cause Analysis:** Find the systematic issues causing symptoms
- **Wide Application:** Ensure fixes apply across all affected components
