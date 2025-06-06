# plan_01a_foundation_components.md
## Component: Foundation Components from Archive

### Objective
Copy and adapt essential foundation components from archive that are required for the 3-bar layout structure and basic dialog functionality specified in the TUI spec.

### Plan
1. **Copy Essential Foundation Components (No Duplicates)**
   - Copy `framed_button.py` from `archive/tui_old/components/` → `openhcs/tui/components/`
   - Copy `help_dialog.py` from `archive/tui_old/dialogs/` → `openhcs/tui/dialogs/`
   - Create `openhcs/tui/dialogs/` directory structure
   - Update `__init__.py` files to export new components

2. **Adapt FramedButton for Spec Requirements**
   - Ensure buttons integrate with toolbar layout (Bar 3)
   - Verify button styling matches spec visual requirements
   - Test button responsiveness and click handling

3. **Adapt HelpDialog for Global Menu**
   - Ensure dialog opens from top menu bar (Bar 1)
   - Verify modal behavior and proper focus management
   - Test "Help" button integration with menu system

4. **Update Import Paths and Dependencies**
   - Fix import statements to match current TUI structure
   - Update TYPE_CHECKING imports to use current interfaces
   - Ensure compatibility with existing command system

### Findings
**Archive Audit Results:**
- **TUI2_OLD**: Contains working action toolbars (already copied to current TUI)
- **TUI_OLD**: Contains complete dialog system, menu bar, status bar, and framed buttons
- **Total Reusable Code**: ~1400 lines of tested UI components (excluding duplicates)
- **Components to Copy**: FramedButton, GlobalSettingsEditor, HelpDialog, MenuBar, StatusBar

**Current TUI State:**
- Most core components already exist (FunctionPatternEditor, StepSettingsEditor, etc.)
- Action toolbars already present (PlateActionsToolbar, PipelineActionsToolbar)
- File browser exists but has DNA-identified issues (complexity 16 & 11 in button handlers)
- Missing: Dialog system, menu bar, status bar, framed buttons

**Duplicate Prevention:**
- ✅ SAFE: framed_button.py, global_settings_editor.py, help_dialog.py, menu_bar.py, status_bar.py
- ❌ SKIP: plate_actions_toolbar.py, pipeline_actions_toolbar.py (already exist)

### Implementation Draft
*Implementation will be added after smell loop approval*
