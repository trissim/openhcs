# plan_01b_global_settings_dialog.md
## Component: Global Settings Dialog with Static Reflection

### Objective
Implement the Global Settings dialog that opens from Bar 1 menu, using static introspection of GlobalPipelineConfig to create dynamic UI elements. This follows the spec requirement for "static introspection of default config."

### Plan
1. **Copy and Adapt Global Settings Editor**
   - Copy `global_settings_editor.py` from `archive/tui_old/dialogs/` → `openhcs/tui/dialogs/`
   - Adapt to use current `GlobalPipelineConfig` structure
   - Ensure compatibility with current TUI state management

2. **Implement Static Reflection Pattern**
   - Use same pattern as function pattern editor for dynamic UI generation
   - Inspect `GlobalPipelineConfig` fields using dataclass/Pydantic introspection
   - Generate appropriate widgets for each field type (dropdowns for enums, checkboxes for booleans)

3. **Create Dynamic UI Elements**
   - **Microscope dropdown**: `Microscope.AUTO`, `Microscope.IMAGEXPRESS`, `Microscope.OPERAPHENIX`
   - **Backend dropdowns**: For VFS configuration (DISK, MEMORY, ZARR)
   - **Numeric inputs**: For worker counts, timeouts, etc.
   - **Boolean checkboxes**: For feature flags and options

4. **Implement Save/Cancel Logic**
   - Validate configuration changes before applying
   - Update global configuration in TUI state
   - Provide user feedback for successful/failed updates
   - Handle configuration validation errors gracefully

5. **Integration with Menu System**
   - Ensure dialog opens from "Global Settings" button in Bar 1
   - **Modal Behavior**: Block all interaction with main TUI, take most of screen
   - **Single Dialog Rule**: Only one dialog can be open at once
   - Handle focus management and keyboard navigation

### Findings
**Specification Requirements:**
- "global setting open a window for static introspection of default config and allows it to be changed"
- Must use static reflection pattern similar to function pattern editor
- Should allow changing defaults that affect all orchestrators

**Available Infrastructure:**
- `GlobalPipelineConfig` dataclass with all configurable fields
- Existing static analysis utilities in `openhcs/tui/utils/static_analysis.py`
- Function pattern editor as reference implementation for static reflection

**Key Configuration Areas:**
- Microscope type selection (AUTO, IMAGEXPRESS, OPERAPHENIX)
- Storage backend defaults (DISK, MEMORY, ZARR)
- Worker and performance settings
- Path planning and VFS configuration

### Implementation Draft
*Implementation will be added after smell loop approval*
