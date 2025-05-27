# plan_05a_workflow_integration.md
## Component: Complete Plate Management Workflow Integration

### Objective
Implement the complete plate management workflow with proper state management, command integration, and user feedback as specified in the TUI spec.

### Plan
1. **Implement Add Plate Workflow**
   - Multi-folder selection dialog → multiple directory selection
   - Create `PipelineOrchestrator` instance for each selected directory
   - Add plates to TUI state with `?` status (not initialized)
   - Generate unique plate IDs and meaningful names from directory names
   - Provide user feedback for successful batch creation

2. **Implement Initialize Plate Workflow**
   - `InitializePlatesCommand` calls `orchestrator.initialize()` on selected plates
   - Update plate status from `?` to `!` (initialized but not compiled)
   - Handle initialization errors with proper user feedback
   - Show progress for batch initialization operations
   - Update plate list display with new status symbols

3. **Implement Compile Plate Workflow**
   - `CompilePlatesCommand` calls `orchestrator.compile_pipelines()` on selected plates
   - Update plate status from `!` to `o` (compiled/ready)
   - Validate that plates are initialized before allowing compilation
   - Handle compilation errors with detailed error messages
   - Show compilation progress and results

4. **Implement Run Plate Workflow**
   - `RunPlatesCommand` calls `orchestrator.execute_compiled_plate()` on selected plates
   - Validate that plates are compiled before allowing execution
   - Show execution progress with real-time updates
   - Handle execution errors and provide detailed feedback
   - Maintain `o` status after successful execution

5. **Implement Delete Plate Workflow**
   - `DeleteSelectedPlatesCommand` removes plates from TUI state
   - Clean up orchestrator resources and temporary files
   - Confirm deletion with user before proceeding
   - Update plate list display after deletion
   - Handle errors during cleanup gracefully

6. **Integrate Error Handling and Feedback**
   - Show error dialogs for operation failures
   - Display progress indicators for long-running operations
   - Update status bar with operation status and messages
   - Provide detailed error information in expandable log drawer

### Findings
**Specification Requirements:**
- Complete plate lifecycle: add → initialize → compile → run
- Multi-folder selection for batch plate creation
- Status progression with visual feedback (`?` → `!` → `o`)
- Proper error handling and user feedback
- Progress indicators for long-running operations

**Available Domain Objects:**
- `PipelineOrchestrator` with `initialize()`, `compile_pipelines()`, `execute_compiled_plate()`
- Command classes for each operation
- TUI state management for plate tracking
- Status symbol system for visual feedback

**Integration Points:**
- Commands need access to selected plates and orchestrators
- Status updates need to trigger UI refresh
- Error handling needs to show user-friendly messages
- Progress tracking needs real-time UI updates

### Implementation Draft
*Implementation will be added after smell loop approval*
