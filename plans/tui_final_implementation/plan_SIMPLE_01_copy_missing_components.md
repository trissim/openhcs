# plan_SIMPLE_01_copy_missing_components.md
## Component: Copy and Adapt Missing Components from Archive

### Objective
Copy and adapt the required components from archive to support the canonical TUI layout: FramedButton for all buttons, StatusBar for bottom status display. Note: MenuBar not needed - canonical uses specific button layout.

### Plan
1. **CRITICAL: Adapt FramedButton Component (2 hours)**
   - Copy `archive/tui_old/components/framed_button.py` → `openhcs/tui/components/framed_button.py`
   - **Verify prompt_toolkit 3.0.51 compatibility** (current canonical version)
   - **Add missing `container` property** to match ComponentInterface
   - **Add missing `update_data()` method** implementation
   - **Keep `__pt_container__()` for backward compatibility**
   - **Verify async/await correctness** in event handlers
   - **Ensure proper import management** to avoid circular imports
   - Update import paths to match current TUI structure
   - Add to `openhcs/tui/components/__init__.py` exports

2. **CRITICAL: Adapt StatusBar Component (4 hours)**
   - Copy `archive/tui_old/status_bar.py` → `openhcs/tui/components/status_bar.py`
   - **Verify prompt_toolkit 3.0.51 compatibility** (current canonical version)
   - **Add missing `container` property** to match ComponentInterface
   - **Add missing `update_data()` method** implementation
   - **Update constructor signature** to match current TUIState interface
   - **Fix state management integration** with current TUIState
   - **Enforce proper state subscription** (all components must subscribe correctly)
   - **Verify async/await correctness** in event handlers
   - **Ensure proper import management** to avoid circular imports
   - **Canonical integration**: Display error messages and status updates as specified
   - **Error dialog integration**: Show OK dialogs for errors as per specification

3. **Update Component Exports (30 minutes)**
   - Add FramedButton and StatusBar to `openhcs/tui/components/__init__.py`
   - Ensure proper import structure for integration
   - Remove MenuBar references (not needed for canonical layout)
   - Test that all components can be imported without errors

### Findings
**CANONICAL SPECIFICATION REQUIREMENTS:**
- **FramedButton**: Needed for all buttons in canonical layout (`[add]`, `[del]`, `[Global Settings]`, etc.)
- **StatusBar**: Needed for bottom status display (`Status: ...`) with error message integration
- **MenuBar**: NOT needed - canonical uses specific button layout, not traditional menu

**COMPONENT REQUIREMENTS FROM tui_final.md:**
- **Button styling**: All buttons should be framed/boxed style: `[add]`, `[del]`, `[edit]`, etc.
- **Status display**: Bottom bar shows status messages and error dialogs with OK button
- **Error handling**: Validation errors display in status bar with modal OK dialog
- **Integration**: Components must work with orchestrator state management

**INTERFACE COMPATIBILITY ISSUES:**
- **Archive components use OLD interface pattern**: `__pt_container__()` + Container delegation methods
- **Current TUI uses NEW interface pattern**: `container` property + `update_data()` method
- **Major adaptation required** for FramedButton and StatusBar

**COMPONENT INVENTORY RESULTS:**
- **FramedButton**: ✅ Available in archive, needs interface adaptation
- **StatusBar**: ✅ Available in archive, needs interface adaptation
- **MenuBar**: ❌ Not needed for canonical layout

**Archive Locations:**
- `archive/tui_old/components/framed_button.py`
- `archive/tui_old/status_bar.py`

**Target Locations:**
- `openhcs/tui/components/framed_button.py`
- `openhcs/tui/components/status_bar.py`

**ESTIMATED EFFORT:**
- **FramedButton adaptation**: ~2 hours (interface updates)
- **StatusBar adaptation**: ~4 hours (interface + state integration + error dialogs)
- **Component exports**: ~30 minutes (update __init__.py files)
- **Total**: ~6.5 hours (was 11 hours before removing MenuBar requirement)

**INTELLECTUAL HONESTY WIN:**
- Canonical specification clarified exact component requirements
- MenuBar not needed - saves 4 hours of unnecessary work
- Proper specification adherence prevents over-implementation

### Implementation Draft
*Implementation will be added after smell loop approval*
