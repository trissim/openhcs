# plan_04b_plate_config_editor.md
## Component: Plate Configuration Editor with Static Reflection

### Objective
Implement plate-specific configuration editing using static reflection pattern (similar to function pattern editor) as specified in the TUI spec for the "edit" button functionality.

### Plan
1. **Understand Static Reflection Pattern from Spec**
   - "very similar logic in gui static reflection as in function pattern editor"
   - Use same introspection approach as `FunctionPatternEditor`
   - Generate dynamic UI elements based on orchestrator configuration structure
   - Allow editing of plate-specific settings that override global defaults

2. **Analyze Orchestrator Configuration Structure**
   - Examine `PipelineOrchestrator` configuration parameters
   - Identify configurable fields: paths, backends, processing options
   - Determine which settings can be overridden per-plate
   - Map configuration types to appropriate UI widgets

3. **Create PlateConfigEditor Component**
   - Design `PlateConfigEditor` class following `FunctionPatternEditor` pattern
   - Use static analysis to inspect orchestrator configuration fields
   - Generate dynamic form elements for each configurable parameter
   - Implement validation and type checking for configuration values

4. **Implement Dynamic UI Generation**
   - **Path fields**: Directory browser buttons for input/output paths
   - **Enum fields**: Dropdowns for backend selection, microscope types
   - **Boolean fields**: Checkboxes for feature flags
   - **Numeric fields**: Input boxes with validation
   - **List fields**: Dynamic list editors for multi-value settings

5. **Integrate with Edit Plate Command**
   - Modify `ShowEditPlateConfigDialogCommand` to open PlateConfigEditor
   - Pass selected plate's current configuration to editor
   - Implement save/cancel workflow with proper validation
   - Update plate configuration in TUI state and orchestrator

6. **Implement Configuration Persistence**
   - Save plate-specific configurations to appropriate storage
   - Load existing configurations when editing plates
   - Handle configuration inheritance from global defaults
   - Provide reset-to-defaults functionality

### Findings
**Specification Requirements:**
- "edit" button opens plate configuration editor
- Uses "very similar logic in gui static reflection as in function pattern editor"
- Allows customization of plate-specific settings
- Should override global defaults for specific plates

**Existing Pattern to Follow:**
- `FunctionPatternEditor` provides the static reflection pattern
- Uses `get_function_registry_by_backend()` for dynamic discovery
- Generates forms based on function signatures and types
- Implements proper validation and error handling

**Configuration Areas to Support:**
- Storage backend selection (DISK, MEMORY, ZARR)
- Input/output directory paths
- Processing parameters and options
- Microscope-specific settings
- Performance and worker settings

**Integration Points:**
- `ShowEditPlateConfigDialogCommand` triggers the editor
- `PipelineOrchestrator` stores plate-specific configuration
- TUI state manages configuration changes
- Global settings provide default values

### Implementation Draft
*Implementation will be added after smell loop approval*
