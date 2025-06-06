# plan_02b_multi_folder_selection.md
## Component: Multi-Folder Selection for Add Plate Workflow

### Objective
Implement multi-folder selection capability in the file browser to support the spec requirement: "file select dialog that allows multiple folders to be selected." This is critical for the add plate workflow.

### Plan
1. **Extend File Browser for Multi-Selection**
   - Add `select_multiple` parameter to FileManagerBrowser constructor
   - Implement multi-selection UI with checkboxes or selection indicators
   - Add keyboard shortcuts for multi-select (Ctrl+click, Shift+click)
   - Display selected folder count in dialog status

2. **Implement Multi-Folder Selection Logic**
   - Track multiple selected directories in internal state
   - Validate that all selections are directories (not files)
   - Provide visual feedback for selected vs unselected folders
   - Allow deselection of previously selected folders

3. **Update Add Plate Dialog Integration**
   - Modify `prompt_for_directory_dialog()` to support multi-selection mode
   - Update add plate command to handle multiple directory results
   - Create multiple orchestrators from selected directories
   - Add all plates to state with proper naming and IDs

4. **Implement Batch Plate Creation**
   - Create unique plate IDs for each selected directory
   - Generate meaningful plate names from directory names
   - Initialize all plates with `?` status (not initialized)
   - Provide user feedback for successful batch creation

5. **Add Multi-Selection UI Elements**
   - "Select All" / "Clear All" buttons for convenience
   - Selected folder count display: "3 folders selected"
   - Preview list of selected folders before confirmation
   - Proper validation before allowing OK button

### Findings
**Specification Requirements:**
- "file select dialog that allows multiple folders to be selected"
- Each selected folder becomes a separate plate (orchestrator)
- All plates start with `?` status (not initialized)
- User should be able to batch-initialize multiple plates

**Current File Browser Limitations:**
- Only supports single directory selection
- No multi-selection UI elements
- Add plate command expects single directory result
- No batch creation workflow

**Integration Points:**
- `ShowAddPlateDialogCommand` needs to handle multiple results
- `TUIState.add_plate()` may need batch operation support
- Plate list view needs to display multiple new plates
- Status management needs to handle batch operations

### Implementation Draft
*Implementation will be added after smell loop approval*
