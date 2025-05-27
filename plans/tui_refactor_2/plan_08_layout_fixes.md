# plan_08_layout_fixes.md
## Component: TUI Layout Fixes and Pane Swapping

### Objective
Fix the TUI layout issues where the boxes don't properly include their respective menu bars, and implement the swapping functionality between Plate Manager and Dual Step/Func Editor panes when the Edit button is pressed in the Pipeline Editor.

### Findings
Based on analysis of the codebase:

1. **Layout Structure Issues**:
   - The current layout uses separate horizontal bars for titles and buttons, but they're not visually connected to their respective content panes
   - The main content area is created separately from the title and button bars, causing a disconnected appearance
   - The Frame containers are applied at the wrong level in the hierarchy, causing the separation

2. **Current Layout Implementation**:
   - In `tui_architecture.py`, the `_create_root_container()` method creates a layout with:
     - Top bar with Global Settings, Help buttons, and OpenHCS label
     - Titles bar with separate frames for "1_plate_manager" and "2_Pipeline_editor"
     - Buttons bar with dynamic containers for plate manager and pipeline editor buttons
     - Main content area with left pane and pipeline editor pane
   - This structure doesn't visually group the title, buttons, and content of each pane together

3. **Pane Swapping Mechanism**:
   - The `_get_left_pane()` method in `tui_architecture.py` already has logic to dynamically show different panes based on state
   - It currently handles swapping between PlateManagerPane, PlateConfigEditorPane, and DualStepFuncEditorPane
   - However, it doesn't have a direct connection to the Pipeline Editor's Edit button

4. **Affected Components**:
   - `openhcs/tui/tui_architecture.py`: Main layout definition
   - `openhcs/tui/pipeline_editor.py`: Edit button handler
   - `openhcs/tui/plate_manager_core.py`: Container structure

### Plan
1. **Restructure Main Layout**:
   - Modify `_create_root_container()` to create a layout that properly groups each pane's components
   - Create two main columns, each containing a VSplit with:
     - Title bar
     - Buttons bar
     - Content area
   - Apply Frame containers at the appropriate level to visually connect the components

2. **Update Left Pane Container Structure**:
   - Modify `_get_left_pane()` to return a complete container with title, buttons, and content
   - Ensure the Frame is applied to the entire container, not just the content

3. **Update Pipeline Editor Container Structure**:
   - Modify `_get_pipeline_editor_pane()` to return a complete container with title, buttons, and content
   - Ensure the Frame is applied to the entire container, not just the content

4. **Implement Pane Swapping on Edit Button Press**:
   - Update the Pipeline Editor's Edit button handler to:
     - Set the appropriate state variables to trigger showing the DualStepFuncEditorPane
     - Notify observers of the change
   - Ensure the `_get_left_pane()` method responds to this state change

5. **Update Title Bars**:
   - Ensure title bars show the correct title based on the current pane
   - Update the title when swapping between Plate Manager and Dual Step/Func Editor

6. **Static Analysis**:
   - Use the meta tool to analyze the updated layout structure
   - Verify that all containers are properly nested
   - Check for any potential issues with the dynamic container swapping

### Implementation Draft
```python
# In openhcs/tui/tui_architecture.py

def _create_root_container(self) -> Container:
    """
    Creates the root container for the TUI application with proper grouping of components.
    """
    # Top Bar (Global bar)
    top_bar = VSplit([
        Button("Global Settings", handler=self.show_global_settings_command.execute),
        Button("Help", handler=self.show_help_command.execute),
        Window(width=Dimension(weight=1), char=' '),  # Flexible spacer
        Label("OpenHCS_V1.0", style="class:app-title", dont_extend_width=True)
    ], height=Dimension.exact(1), padding=0)

    # Main content area with two columns, each with its own frame
    main_content_area = VSplit([
        # Left column (Plate Manager or Dual Step/Func Editor)
        self._get_left_pane_with_frame(),

        # Right column (Pipeline Editor)
        self._get_pipeline_editor_pane_with_frame(),
    ], height=Dimension(weight=1, min=10))

    # Status Bar
    status_bar_container = self._get_status_bar()

    # Create the main layout
    main_layout = HSplit([
        top_bar,
        main_content_area,
        status_bar_container,
    ])

    # Wrap in a FloatContainer to allow the loading screen to float on top
    return FloatContainer(
        content=main_layout,
        floats=[
            Float(
                content=self.loading_screen,
                transparent=False,
            )
        ]
    )

def _get_left_pane_with_frame(self) -> Container:
    """
    Get the left pane with its title and buttons, wrapped in a frame.
    """
    # Determine which pane to show based on state
    is_editing_step_config = getattr(self.state, 'editing_step_config', False)
    is_editing_plate_config = getattr(self.state, 'editing_plate_config', False)

    # Set the title based on the current pane
    if is_editing_step_config:
        title = "Step Editor"
        buttons_container = self._get_dual_step_func_editor_buttons()
        content_container = self._get_dual_step_func_editor_content()
    elif is_editing_plate_config:
        title = "Plate Config Editor"
        buttons_container = self._get_plate_config_editor_buttons()
        content_container = self._get_plate_config_editor_content()
    else:
        title = "Plate Manager"
        buttons_container = DynamicContainer(
            lambda: self.plate_manager.get_buttons_container()
            if self.plate_manager and hasattr(self.plate_manager, 'get_buttons_container')
            else Box(Label("[Plate Buttons Placeholder]"), padding_left=1)
        )
        content_container = self._get_plate_manager_content()

    # Create a container with title, buttons, and content
    return Frame(
        HSplit([
            buttons_container,
            content_container
        ]),
        title=title
    )

def _get_pipeline_editor_pane_with_frame(self) -> Container:
    """
    Get the pipeline editor pane with its title and buttons, wrapped in a frame.
    """
    # Create a container with title, buttons, and content
    return Frame(
        HSplit([
            DynamicContainer(
                lambda: self.pipeline_editor.get_buttons_container()
                if self.pipeline_editor and hasattr(self.pipeline_editor, 'get_buttons_container')
                else Box(Label("[Pipeline Buttons Placeholder]"), padding_left=1)
            ),
            self._get_pipeline_editor_content()
        ]),
        title="Pipeline Editor"
    )

def _get_pipeline_editor_content(self) -> Container:
    """
    Get the pipeline editor content without the frame.

    Returns:
        Container with the pipeline editor content
    """
    if self.pipeline_editor is None:
        return Label("Pipeline Editor initializing...")

    if hasattr(self.pipeline_editor, '_dynamic_step_list_wrapper'):
        return self.pipeline_editor._dynamic_step_list_wrapper

    return Label("Pipeline Editor content not ready.")

def _get_plate_manager_content(self) -> Container:
    """
    Get the plate manager content without the frame.

    Returns:
        Container with the plate manager content
    """
    if self.plate_manager is None:
        return Label("Plate Manager initializing...")

    if hasattr(self.plate_manager, '_dynamic_plate_list_wrapper'):
        return self.plate_manager._dynamic_plate_list_wrapper

    return Label("Plate Manager content not ready.")

# In openhcs/tui/pipeline_editor.py

async def _edit_step(self) -> None:
    """
    Edit the currently selected step.
    This will trigger showing the DualStepFuncEditorPane in the left pane.
    """
    async with self.steps_lock:
        if not self.steps or self.selected_index >= len(self.steps):
            return

        # Get the selected step
        step = self.steps[self.selected_index]

        # Set the state to trigger showing the DualStepFuncEditorPane
        self.state.step_to_edit_config = step
        self.state.editing_step_config = True

        # Notify observers of the change
        await self.state.notify('edit_step_dialog_requested', step)
```
