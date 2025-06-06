# plan_05b_pipeline_workflow.md
## Component: Pipeline Editing Workflow Integration

### Objective
Implement the complete pipeline editing workflow with step management, dual editor integration, and file operations as specified in the TUI spec.

### Plan
1. **Implement Add Step Workflow**
   - `AddStepCommand` creates new `FunctionStep` instance
   - Add step to current pipeline in TUI state
   - Update pipeline editor display with new step
   - Assign default step name and parameters
   - Focus on newly added step for immediate editing

2. **Implement Edit Step Workflow**
   - `EditStepCommand` triggers dual editor pane replacement
   - Replace plate manager pane with dual STEP/FUNC editor
   - Load selected step data into dual editor
   - Implement Step/Func toggle functionality within editor
   - Provide save/close buttons in editor header

3. **Implement Step Save/Close Workflow**
   - **Save**: Construct new `FunctionStep` with all parameters from dual editor
   - Update step in pipeline and refresh pipeline editor display
   - Validate all required parameters before saving
   - **Close**: Restore original layout (show plate manager pane)
   - Handle unsaved changes with confirmation dialog

4. **Implement Delete Step Workflow**
   - `DeleteStepCommand` removes selected steps from pipeline
   - Update pipeline editor display after deletion
   - Confirm deletion with user before proceeding
   - Handle dependencies and step ordering after deletion

5. **Implement Load/Save Pipeline Workflow**
   - **Load**: `LoadPipelineCommand` opens file dialog for .pipeline files
   - Use `pickle.load()` to deserialize list of steps from file
   - Update pipeline editor display with loaded steps
   - **Save**: `SavePipelineCommand` opens file dialog for save location
   - Use `pickle.dump()` to serialize current step list to .pipeline file
   - Provide user feedback for successful/failed operations

6. **Integrate with Dual Editor System**
   - Ensure dual editor has access to function registry
   - Implement proper parameter validation and type checking
   - Handle function selection and parameter updates
   - Maintain step data consistency during editing

### Findings
**Specification Requirements:**
- Complete pipeline editing: add, edit, delete, load, save steps
- Dual editor pane replacement for step editing
- Step/Func toggle within dual editor
- File operations: .pipeline files are pickled lists of steps
- Always start with blank pipeline (no default loading)
- Proper validation and error handling

**Existing Components to Leverage:**
- `FunctionStep` and `AbstractStep` for step data
- `DualStepFuncEditor` for step editing interface
- `StepListView` for pipeline display
- File dialog system for load/save operations

**Integration Points:**
- Pipeline editor buttons trigger appropriate commands
- Dual editor needs step data and function registry access
- Save/load operations need file format handling
- Step validation needs function registry integration

### Implementation Draft
*Implementation will be added after smell loop approval*
