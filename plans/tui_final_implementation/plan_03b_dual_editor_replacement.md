# plan_03b_dual_editor_replacement.md
## Component: Dual Editor Pane Replacement System

### Objective
Implement the dual editor pane replacement system where clicking "edit" on a pipeline step replaces the plate manager pane with the dual STEP/FUNC editor, as specified in the TUI spec.

### Plan
1. **Understand Dual Editor Architecture from Spec**
   - Dual editor has internal toggle: `|_X_Step_X_|___Func___|_[save]__[close]__________|`
   - Step mode: Shows AbstractStep parameters (name, input_dir, output_dir, etc.)
   - Func mode: Shows function pattern editor with dynamic parameter inspection
   - Save/Close buttons: `[save]` and `[close]` in top-right of editor pane

2. **Implement Pane Replacement Logic**
   - When "edit" clicked on pipeline step → replace left pane (plate manager) with dual editor
   - Keep right pane (pipeline editor) visible and functional
   - Dual editor occupies full left pane space
   - Original layout restored when dual editor closed

3. **Create Dual Editor Container**
   - Design `DualEditorContainer` that wraps existing `DualStepFuncEditor`
   - Add internal toggle bar: `|_X_Step_X_|___Func___|_[save]__[close]__________|`
   - Implement Step/Func mode switching within the editor
   - Add save/close buttons in editor header

4. **Implement Step/Func Toggle System**
   - **Toggle Bar**: `|_X_Step_X_|___Func___|` with clickable tabs
   - **Step Mode**: Show `StepSettingsEditor` with AbstractStep parameters
   - **Func Mode**: Show `FunctionPatternEditor` with function-specific parameters
   - **Data Preservation**: All parameters preserved when switching between modes
   - **Free Switching**: No validation when switching - validation only at compile time

5. **Implement Save/Close Workflow**
   - **Save Button**: Construct new FunctionStep with all parameters → update pipeline
   - **Save Button State**: Gray out after saving until next change
   - **Close Button**: Show confirm dialog if unsaved changes → restore original layout
   - **No Validation Required**: Compiler will flag issues later, can save partial/invalid configs
   - **Sequence**: Construct new step object → update step in pipeline → refresh display

6. **Integrate with Layout Manager**
   - Modify `ThreeBarLayout` to support pane replacement
   - Implement `replace_left_pane(new_component)` method
   - Implement `restore_original_layout()` method
   - Ensure proper focus management during transitions

7. **Function Registry Integration**
   - Show all functions in FUNC_REGISTRY regardless of backend
   - No backend filtering needed - compiler handles memory type conversion
   - Allow any function in any order (except special function rules)
   - Populate function dropdown with complete registry

### Findings
**Specification Requirements:**
- "when clicking edit on a step in the pipeline editor, it replaces the plate_manager window"
- Dual editor has internal Step/Func toggle with save/close buttons
- Step mode shows AbstractStep parameters, Func mode shows function pattern editor
- Save constructs new FunctionStep, Close restores original layout

**Existing Components to Leverage:**
- `DualStepFuncEditor` - Current dual editor implementation
- `StepSettingsEditor` - For Step mode parameter editing
- `FunctionPatternEditor` - For Func mode parameter editing
- `PlateListView` - Component that gets replaced/restored

**Key Integration Points:**
- Pipeline editor "edit" button triggers pane replacement
- Dual editor needs access to selected step data
- Save operation needs to update pipeline in TUI state
- Close operation needs to restore plate manager visibility

### Implementation Draft
*Implementation will be added after smell loop approval*
