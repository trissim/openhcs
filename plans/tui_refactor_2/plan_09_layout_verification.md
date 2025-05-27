# plan_09_layout_verification.md
## Component: TUI Layout Verification and Refinement

### Objective
Verify that the current TUI layout implementation matches the canonical sketch provided, and make any necessary adjustments to ensure the layout logic is correct. Focus on the menu bar structure, pane organization, and the step/func editor toggle functionality.

### Findings
Based on analysis of the current implementation and the canonical sketch:

1. **Main Layout Structure**:
   - The current implementation has a top-level global menu bar with Global Settings and Help buttons
   - The main content area is divided into two columns (left and right)
   - Each column has its own frame with a title and buttons
   - The status bar is at the bottom

2. **Menu Bar Implementation**:
   - The original MenuBar class was monolithic and is no longer used for the top bar
   - Instead, individual button containers are created for each pane
   - The top global bar is implemented directly in `_create_root_container()`
   - Each pane has its own buttons container

3. **Pane Swapping Mechanism**:
   - The left pane can switch between Plate Manager, Dual Step/Func Editor, and Plate Config Editor
   - The swapping is triggered by state changes and handled in `_get_left_pane_with_frame()`
   - The Edit button in the Pipeline Editor triggers showing the Dual Step/Func Editor

4. **Step/Func Editor Implementation**:
   - The Dual Step/Func Editor has two views: Step Settings and Func Pattern Editor
   - The toggle between these views is handled by buttons in the editor's menu bar
   - Each view has its own set of controls and parameters

5. **Discrepancies with Canonical Sketch**:
   - The title bars in the current implementation are part of the Frame, not separate bars
   - The buttons are inside the Frame, not between the title and content
   - The layout of the Dual Step/Func Editor may not exactly match the sketch

### Plan
1. **Verify Top-Level Layout**:
   - Confirm that the top-level layout has:
     - A global menu bar with Global Settings and Help buttons
     - Two main columns (left and right) with their own frames
     - A status bar at the bottom

2. **Verify Left Pane (Plate Manager)**:
   - Confirm that the Plate Manager pane has:
     - A title "Plate Manager"
     - Buttons: Add, Del, Edit, Init, Compile, Run
     - A list of plates with status indicators

3. **Verify Right Pane (Pipeline Editor)**:
   - Confirm that the Pipeline Editor pane has:
     - A title "Pipeline Editor"
     - Buttons: Add, Del, Edit, Load, Save
     - A list of steps

4. **Verify Pane Swapping**:
   - Confirm that clicking the Edit button in the Pipeline Editor:
     - Hides the Plate Manager
     - Shows the Dual Step/Func Editor in the left pane
     - The Dual Step/Func Editor has its own title and buttons

5. **Verify Dual Step/Func Editor**:
   - Confirm that the Dual Step/Func Editor has:
     - A toggle between Step and Func views
     - Save and Close buttons
     - Appropriate controls for each view

6. **Adjust Frame and Button Layout**:
   - Modify the Frame implementation to match the canonical sketch:
     - Title bar at the top
     - Buttons bar below the title
     - Content area below the buttons

7. **Update Step/Func Editor Layout**:
   - Ensure the Step view has:
     - A title "Step Settings Editor"
     - Load and Save As buttons
     - Fields for Name, input_dir, output_dir, etc.
   - Ensure the Func view has:
     - A title "Func Pattern Editor"
     - Add, Load, Save As buttons
     - Dict keys dropdown and controls
     - List of functions with parameters

8. **Static Analysis**:
   - Use the meta tool to analyze the layout structure
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

# In openhcs/tui/dual_step_func_editor.py

def _create_container(self) -> Container:
    """
    Create the container for the Dual Step/Func Editor.
    """
    # Create the toggle buttons for Step/Func views
    toggle_buttons = VSplit([
        Button("Step", handler=self._show_step_view, width=10),
        Button("Func", handler=self._show_func_view, width=10),
        Window(width=Dimension(weight=1), char=' '),  # Flexible spacer
        Button("Save", handler=self._save_step),
        Button("Close", handler=self._close_editor)
    ], height=Dimension.exact(1), padding=0)
    
    # Create the content container based on the current view
    if self.current_view == "step":
        title = "Step Settings Editor"
        buttons = VSplit([
            Label(title, dont_extend_width=True),
            Window(width=Dimension(weight=1), char=' '),  # Flexible spacer
            Button("Load", handler=self._load_step),
            Button("Save As", handler=self._save_step_as)
        ], height=Dimension.exact(1), padding=0)
        
        content = self._create_step_view()
    else:  # func view
        title = "Func Pattern Editor"
        buttons = VSplit([
            Label(title, dont_extend_width=True),
            Window(width=Dimension(weight=1), char=' '),  # Flexible spacer
            Button("Add", handler=self._add_func),
            Button("Load", handler=self._load_func),
            Button("Save As", handler=self._save_func_as)
        ], height=Dimension.exact(1), padding=0)
        
        content = self._create_func_view()
    
    # Combine everything
    return HSplit([
        toggle_buttons,
        buttons,
        content
    ])

def _create_step_view(self) -> Container:
    """
    Create the Step Settings view.
    """
    # Create fields for step parameters
    name_field = HSplit([
        Label("[reset] Name:"),
        TextArea(text=self.func_step.name or "")
    ])
    
    input_dir_field = HSplit([
        Label("[reset] input_dir:"),
        TextArea(text=self.func_step.input_dir or "")
    ])
    
    output_dir_field = HSplit([
        Label("[reset] output_dir:"),
        TextArea(text=self.func_step.output_dir or "")
    ])
    
    force_disk_output_field = HSplit([
        Label("[reset] force_disk_output:"),
        Checkbox(checked=self.func_step.force_disk_output or False)
    ])
    
    # Combine all fields
    return HSplit([
        name_field,
        input_dir_field,
        output_dir_field,
        force_disk_output_field
    ])

def _create_func_view(self) -> Container:
    """
    Create the Func Pattern Editor view.
    """
    # Create dict keys dropdown
    dict_keys_container = HSplit([
        VSplit([
            Label("dict_keys:"),
            Dropdown(values=["None"]),
            Button("+", width=3),
            Button("-", width=3),
            Window(width=Dimension(weight=1), char=' '),  # Flexible spacer
            Button("edit_in_vim", width=10),
            Label("?", width=2)
        ]),
    ])
    
    # Create function list
    func_list = []
    for i, func in enumerate(self.func_step.func or []):
        func_container = self._create_func_container(i, func)
        func_list.append(func_container)
    
    # Combine everything
    return HSplit([
        dict_keys_container,
        *func_list
    ])

def _create_func_container(self, index: int, func: Dict) -> Container:
    """
    Create a container for a single function.
    """
    # Create function header
    header = VSplit([
        Label(f"Func {index + 1}:"),
        Dropdown(values=["percentile stack normalize", "n2v2", "3d_deconv"]),
    ])
    
    # Create parameter fields
    params = []
    for key, value in func.items():
        param_container = HSplit([
            VSplit([
                Label(f"[reset] {key}:"),
                TextArea(text=str(value))
            ])
        ])
        params.append(param_container)
    
    # Create buttons
    buttons = VSplit([
        Button("move /\\", width=10),
        Button("move \\/", width=10),
        Button("add", width=10),
        Button("delete", width=10)
    ])
    
    # Combine everything
    return HSplit([
        header,
        *params,
        buttons
    ])
```
