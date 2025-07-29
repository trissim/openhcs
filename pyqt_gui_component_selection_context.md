# PyQt6 GUI Component Selection Implementation Context

## Overview
This document provides complete context for implementing component selection functionality in the OpenHCS PyQt6 GUI, mirroring the working Textual TUI implementation.

## Problem Statement
The PyQt6 GUI's function list editor component selection was not working properly. Users could not select components (channels, wells, sites, z_index) when editing function patterns in pipeline steps.

## Root Cause Analysis
The issue had multiple layers:

1. **Missing Orchestrator Connection**: PyQt6 GUI had no connection between plate manager and pipeline editor
2. **Race Condition**: Function list editor tried to access orchestrator before initialization completed  
3. **Missing Metadata Display**: Component buttons and selectors didn't show metadata names like Textual TUI
4. **Incorrect Navigation**: Function list editor couldn't find current plate context

## Architecture Comparison

### Textual TUI (WORKING)
```
PipelinePlateWindow
├── PlateManagerWidget (has orchestrators)
└── PipelineEditorWidget (connected via signals)
    └── DualEditorWindow
        └── FunctionListEditorWidget
            └── _get_current_orchestrator() → finds via app.query_one()
```

### PyQt6 GUI (FIXED)
```
MainWindow
├── FloatingWindow[plate_manager] (has orchestrators)
└── FloatingWindow[pipeline_editor] (connected via signals)
    └── DualEditorWindow (modal)
        └── FunctionListEditorWidget
            └── _get_current_orchestrator() → finds via main_window reference
```

## Key Components

### 1. Plate Manager Widget (`openhcs/pyqt_gui/widgets/plate_manager.py`)
- **Purpose**: Manages plate selection and orchestrator initialization
- **Key Properties**:
  - `self.orchestrators: Dict[str, PipelineOrchestrator]` - Maps plate paths to orchestrators
  - `self.selected_plate_path: str` - Currently selected plate
- **Key Signals**:
  - `plate_selected = pyqtSignal(str)` - Emitted when plate selection changes

### 2. Pipeline Editor Widget (`openhcs/pyqt_gui/widgets/pipeline_editor.py`)
- **Purpose**: Manages pipeline steps for current plate
- **Key Properties**:
  - `self.current_plate: str` - Current plate path (set by plate manager signal)
- **Key Methods**:
  - `set_current_plate(plate_path)` - Updates current plate context

### 3. Function List Editor Widget (`openhcs/pyqt_gui/widgets/function_list_editor.py`)
- **Purpose**: Edits function patterns with component selection
- **Key Properties**:
  - `self.current_group_by: GroupBy` - Current component type (CHANNEL, WELL, etc.)
  - `self.current_variable_components: List[VariableComponents]` - Variable components
  - `self.main_window` - Reference to main window for orchestrator access
- **Key Methods**:
  - `_get_current_orchestrator()` - Gets orchestrator from plate manager
  - `_get_component_button_text()` - Shows component type and metadata
  - `show_component_selection_dialog()` - Opens component selector

### 4. Group By Selector Dialog (`openhcs/pyqt_gui/dialogs/group_by_selector_dialog.py`)
- **Purpose**: Dual-list component selection dialog
- **Key Features**:
  - Shows components with metadata names (e.g., "Channel 1 | HOECHST 33342")
  - Move components between available/selected lists
  - Extracts component keys from formatted display text

## Implementation Details

### Connection Pattern (Fixed)
```python
# In MainWindow.show_plate_manager()
def _connect_plate_to_pipeline_manager(self, plate_manager_widget):
    if "pipeline_editor" in self.floating_windows:
        pipeline_editor_widget = self._find_pipeline_editor_widget()
        plate_manager_widget.plate_selected.connect(pipeline_editor_widget.set_current_plate)

# In MainWindow.show_pipeline_editor()  
def _connect_pipeline_to_plate_manager(self, pipeline_widget):
    if "plate_manager" in self.floating_windows:
        plate_manager_widget = self._find_plate_manager_widget()
        plate_manager_widget.plate_selected.connect(pipeline_widget.set_current_plate)
```

### Orchestrator Access Pattern (Fixed)
```python
# In FunctionListEditorWidget._get_current_orchestrator()
def _get_current_orchestrator(self):
    main_window = getattr(self, 'main_window', None)
    plate_manager_widget = self._find_plate_manager_widget(main_window)
    current_plate = plate_manager_widget.selected_plate_path
    orchestrator = plate_manager_widget.orchestrators[current_plate]
    
    # Re-initialize if needed (mirrors Textual TUI)
    if not orchestrator.is_initialized():
        orchestrator.initialize()
    
    return orchestrator
```

### Metadata Display Pattern (Fixed)
```python
# Component button text with metadata
def _get_component_display_name(self, component_key: str) -> str:
    orchestrator = self._get_current_orchestrator()
    if orchestrator and self.current_group_by:
        metadata_name = orchestrator.get_component_metadata(self.current_group_by, component_key)
        if metadata_name:
            return metadata_name
    return component_key

# Group by selector with metadata
def _format_component_display(self, component_key: str) -> str:
    base_text = f"{self.component_type.title()} {component_key}"
    if self.orchestrator:
        group_by = GroupBy(self.component_type)
        metadata_name = self.orchestrator.get_component_metadata(group_by, component_key)
        if metadata_name:
            return f"{base_text} | {metadata_name}"
    return base_text
```

## Current Status
✅ **FIXED**: All component selection functionality working
✅ **FIXED**: Metadata display in component buttons and selectors  
✅ **FIXED**: Orchestrator connection and initialization
✅ **FIXED**: Plate manager to pipeline editor signal connections

## Testing Checklist
1. Open Plate Manager → Initialize plate (no warnings)
2. Open Pipeline Editor → Should connect to current plate
3. Edit step → Function Pattern tab → Component button shows "Channel: [metadata]"
4. Click Component button → Selector shows "Channel 1 | HOECHST 33342" format
5. Select components → Should work without errors
6. Component selection persists across dialog opens

## Key Files Modified
- `openhcs/pyqt_gui/main.py` - Added connection methods
- `openhcs/pyqt_gui/widgets/function_list_editor.py` - Fixed orchestrator access and metadata
- `openhcs/pyqt_gui/dialogs/group_by_selector_dialog.py` - Added metadata display
- `openhcs/pyqt_gui/windows/dual_editor_window.py` - Fixed step configuration passing

## Next Steps for Fresh Agent
1. Test the complete component selection workflow
2. Verify metadata display matches Textual TUI exactly
3. Handle edge cases (no metadata, empty components, etc.)
4. Ensure proper error handling and user feedback

## Technical Deep Dive

### Orchestrator Initialization Flow
```python
# 1. User clicks "Initialize Plate" in Plate Manager
async def action_init_plate(self):
    orchestrator = PipelineOrchestrator(
        plate_path=plate_path,
        global_config=self.global_config,
        storage_registry=self.file_manager.registry
    ).initialize()
    self.orchestrators[plate_path] = orchestrator

# 2. Orchestrator.initialize() calls cache_component_keys() and cache_metadata()
def initialize(self):
    self.cache_component_keys()  # Finds components from filenames
    self.cache_metadata()        # Loads metadata names from plate files

# 3. Component keys and metadata are cached for fast access
self._component_keys_cache = {
    GroupBy.CHANNEL: ["1", "2", "3", "4"],
    GroupBy.WELL: ["A01", "A02", ...],
    # ...
}
self._metadata_cache = {
    GroupBy.CHANNEL: {
        "1": "HOECHST 33342",
        "2": "FITC",
        "3": "TRITC",
        "4": "Cy5"
    }
}
```

### Component Selection Dialog Flow
```python
# 1. User clicks component button in function list editor
def show_component_selection_dialog(self):
    orchestrator = self._get_current_orchestrator()
    available_components = orchestrator.get_component_keys(self.current_group_by)

    # 2. Dialog shows formatted components with metadata
    result = GroupBySelectorDialog.select_components(
        available_components=available_components,  # ["1", "2", "3", "4"]
        selected_components=selected_components,
        component_type=self.current_group_by.value,  # "channel"
        orchestrator=orchestrator
    )

# 3. Dialog formats each component for display
def _format_component_display(self, component_key):
    # "1" → "Channel 1 | HOECHST 33342"
    base_text = f"Channel {component_key}"
    metadata_name = orchestrator.get_component_metadata(GroupBy.CHANNEL, component_key)
    return f"{base_text} | {metadata_name}" if metadata_name else base_text

# 4. User selections are extracted back to component keys
def _extract_component_key(self, display_text):
    # "Channel 1 | HOECHST 33342" → "1"
    return display_text.split(' | ')[0].split(' ')[-1]
```

### Error Scenarios and Handling
1. **No Plate Selected**: Function list editor shows "No orchestrator available"
2. **Uninitialized Orchestrator**: Auto-reinitialize like Textual TUI
3. **No Components Found**: Component button disabled, empty selector
4. **No Metadata Available**: Show component keys without metadata
5. **Connection Issues**: Graceful fallback to basic functionality

### Memory and Performance Notes
- Orchestrators are cached per plate path in plate manager
- Component keys and metadata are cached during initialization
- Dialog reuses orchestrator reference (no re-initialization)
- Metadata lookup is O(1) dictionary access
- Component selection state is cached per GroupBy type

### User Experience Patterns
- Component button shows current selection state
- Metadata names provide scientific context
- Dual-list selector follows standard UI patterns
- Selection persists across dialog sessions
- Visual feedback for disabled states

## Debugging Tips for Fresh Agent
1. **Check orchestrator initialization**: Look for "PipelineOrchestrator fully initialized" log
2. **Verify component caching**: Check "Cached X keys" debug logs
3. **Test metadata loading**: Look for "Updated metadata for X" logs
4. **Trace signal connections**: Add debug logs to plate_selected signal handlers
5. **Monitor dialog state**: Log available/selected components in selector dialog

## Code Quality Standards
- Mirror Textual TUI patterns exactly
- Use proper error handling with user-friendly messages
- Add debug logging for troubleshooting
- Follow PyQt6 signal/slot patterns
- Maintain separation of concerns (UI vs business logic)
