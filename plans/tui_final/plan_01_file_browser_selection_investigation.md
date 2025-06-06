# plan_01_file_browser_selection_investigation.md
## Component: File Browser Selection System Investigation

### Objective
Systematically investigate why file browser folder selection has no visual feedback when clicking. This investigation will uncover the complete architecture, identify all potential failure points, and document the exact flow from user click to visual feedback.

### Plan
1. **Map Complete Architecture Flow**
   - Document the complete chain: PlateManager → dialog_helpers → FileManagerBrowser → InteractiveListItem
   - Identify all components involved in selection state management
   - Map the UI update cycle and invalidation flow
   - Document mouse event handling chain

2. **Analyze Selection State Management**
   - Trace how `selected_paths` set is managed in FileManagerBrowser
   - Verify `_toggle_selection()` logic and state updates
   - Check `_update_ui()` and `get_app().invalidate()` calls
   - Analyze `DynamicContainer` rebuild behavior

3. **Investigate Mouse Event Handling**
   - Trace mouse events from InteractiveListItem to FileManagerBrowser
   - Verify `_item_clicked()` → `_set_focus()` → `_on_activate()` flow
   - Check if mouse events are properly reaching the handlers
   - Analyze prompt_toolkit mouse handler registration patterns

4. **Examine Visual Feedback System**
   - Verify InteractiveListItem styling system
   - Check if `is_selected` parameter is properly passed and used
   - Analyze style application in `_build_container()`
   - Test if styles are defined and working

5. **Identify Architectural Issues**
   - Look for state synchronization problems
   - Check for UI update timing issues
   - Identify any defensive programming patterns that mask errors
   - Find any architectural rot or inconsistencies

### Findings

#### 1. Complete Architecture Flow Analysis

**User Action Flow:**
1. User clicks "Add" in PlateManager → `_add_plates()`
2. Calls `prompt_for_multi_folder_dialog()` in dialog_helpers.py
3. Creates `FileManagerBrowser` with `allow_multiple=True`, `selection_mode=DIRECTORIES_ONLY`
4. Dialog shows with file browser embedded
5. User clicks on folder item → Should trigger selection with visual feedback

**Component Chain:**
- **PlateManager** (panes/plate_manager.py) - Initiates folder selection
- **dialog_helpers.py** - Creates dialog with FileManagerBrowser
- **FileManagerBrowser** (editors/file_browser.py) - Main browser logic
- **InteractiveListItem** (components/interactive_list_item.py) - Individual clickable items
- **prompt_toolkit** - Underlying UI framework

#### 2. Selection State Management Analysis

**FileManagerBrowser State:**
- `selected_paths: set[Path] = set()` - Tracks selected folder paths
- `focused_index: int = 0` - Current keyboard focus
- `allow_multiple: bool = True` - Multi-selection mode enabled

**Selection Logic Flow:**
1. `_item_clicked(index)` → `_set_focus(index)` → `_on_activate()`
2. `_on_activate()` → `_toggle_selection()` (when `allow_multiple=True`)
3. `_toggle_selection()` → Updates `selected_paths` set → `_update_ui()`
4. `_update_ui()` → `get_app().invalidate()` → Should trigger UI rebuild

**Critical Issue Found:** The selection logic appears correct, but visual feedback is not working.

#### 3. Mouse Event Handling Analysis

**InteractiveListItem Mouse Handling:**
- Uses `FormattedTextControl` with direct mouse handler assignment (lines 67-79)
- Mouse handler checks for `MouseEventType.MOUSE_UP` and calls `self.on_select(self.item_index)`
- Follows same pattern as `FramedButton` (lines 49-55 in framed_button.py)

**Mouse Event Flow:**
1. User clicks on InteractiveListItem
2. prompt_toolkit routes mouse event to `FormattedTextControl.mouse_handler`
3. Mouse handler calls `self._run_callback(self.on_select, self.item_index)`
4. This calls `FileManagerBrowser._item_clicked(index)`
5. `_item_clicked()` calls `_set_focus(index)` then `_on_activate()`

**Potential Issue:** Mouse events may not be reaching the InteractiveListItem at all.

#### 4. Visual Feedback System Analysis

**InteractiveListItem Styling:**
- `is_selected` parameter passed to constructor (line 454 in file_browser.py)
- Style applied in `_build_container()` method (lines 88-95)
- Uses `selected_style="reverse"` and `unselected_style=""` (lines 461-462)
- Style applied to `Box` wrapper around the text content

**DynamicContainer Rebuild:**
- FileManagerBrowser uses `DynamicContainer(self._build_item_list)` (line 131)
- `_update_ui()` calls `get_app().invalidate()` (line 255)
- This should trigger `_build_item_list()` to be called again
- New InteractiveListItem objects created with updated `is_selected` values

**Critical Discovery:** The visual feedback system should work if mouse events are reaching the handlers.

#### 5. Application Mouse Support Analysis

**CRITICAL FINDING:** Mouse support IS enabled in the main TUI application!
- `canonical_layout.py` line 61: `mouse_support=True`
- This means mouse events should be reaching the components
- Other components like `FramedButton` and `StatusBar` have working mouse handlers
- The issue is NOT at the application level

#### 6. Architectural Issues Identified

**Primary Suspect: DynamicContainer State Synchronization**
- `FileManagerBrowser._build_item_list()` creates new `InteractiveListItem` objects on each call
- Each `InteractiveListItem` captures `is_selected` state at creation time
- When `_toggle_selection()` updates `selected_paths`, `_update_ui()` calls `get_app().invalidate()`
- `DynamicContainer` should rebuild, but new items may not reflect updated selection state

**Potential Race Condition:**
1. User clicks → `_item_clicked()` → `_set_focus()` → `_on_activate()` → `_toggle_selection()`
2. `_toggle_selection()` updates `selected_paths` set
3. `_update_ui()` calls `get_app().invalidate()`
4. `DynamicContainer` rebuilds by calling `_build_item_list()`
5. New `InteractiveListItem` objects created with updated `is_selected` values
6. **BUT**: The mouse click might be processed before the UI rebuild completes

**Lambda Closure Issue:**
- Line 455 in file_browser.py: `display_text_func=lambda data, selected: data['display_text']`
- This lambda captures variables by reference, which could cause issues

**Architectural Rot Detected:**
- Debug logging still present in production code (lines 276, 280, 285, 290, 293)
- Mixed concerns: FileManagerBrowser handles both file system operations AND UI state
- No separation between model (file listing) and view (UI representation)

### Implementation Draft
*Only after smell loop passes - no code until architectural issues are identified and planned*
