# plan_04_workflow_integration.md
## Component: Complete Workflow Integration and State Management

### Objective
Implement the complete plate management and pipeline editing workflows according to the TUI specification, including proper state management, status indicators, and command execution.

### Plan
1. **Implement Plate Management Workflow**
   - **Add Plate**: File dialog → directory selection → create PipelineOrchestrator → add to state with `?` status
   - **Delete Plate**: Remove selected plates from state and clean up resources
   - **Edit Plate**: Open plate-specific configuration editor using static reflection
   - **Initialize Plate**: Call `orchestrator.initialize()` → update status to `!` (yellow)
   - **Compile Plate**: Call `orchestrator.compile_pipelines()` → update status to `o` (green)
   - **Run Plate**: Execute compiled pipeline → show progress and results

2. **Implement Pipeline Editing Workflow**
   - **Add Step**: Create new FunctionStep and add to pipeline
   - **Delete Step**: Remove selected steps from pipeline
   - **Edit Step**: Replace plate manager pane with dual STEP/FUNC editor
   - **Load Pipeline**: File dialog → load pipeline from .pipeline file
   - **Save Pipeline**: File dialog → save current pipeline to .pipeline file

3. **Implement Status Symbol Management**
   - `?` = not initialized yet (red/default)
   - `!` = initialized but not compiled (yellow)
   - `o` = compiled/ready (green)
   - Update symbols based on orchestrator state changes
   - Ensure visual feedback for all operations

4. **Implement Command Integration**
   - Wire all toolbar buttons to proper command instances
   - Ensure commands have access to current state and context
   - Implement proper error handling and user feedback
   - Add command validation and enablement logic

5. **Implement State Synchronization**
   - Ensure TUIState reflects actual orchestrator states
   - Implement proper event handling for state changes
   - Synchronize UI updates with backend operations
   - Handle concurrent operations and state conflicts

### Findings
**Workflow Requirements from Spec:**
- Complete plate lifecycle: add → initialize → compile → run
- Pipeline editing with step management
- Dual editor integration for step configuration
- Status indicators for plate states
- File operations for load/save

**Available Domain Objects:**
- `PipelineOrchestrator` with `initialize()`, `compile_pipelines()`, `execute_compiled_plate()`
- `AbstractStep` and `FunctionStep` for pipeline construction
- `GlobalPipelineConfig` for configuration management
- `FUNC_REGISTRY` for function discovery

**State Management Needs:**
- Track plate orchestrators and their states
- Manage current pipeline definition
- Handle selected plates and steps
- Coordinate UI updates with backend changes

**Integration Points:**
- Commands need access to orchestrators and state
- UI components need state change notifications
- File dialogs need proper callback handling
- Status updates need visual feedback

### Implementation Draft
*Implementation will be added after smell loop approval*
