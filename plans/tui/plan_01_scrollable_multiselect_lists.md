# plan_01_scrollable_multiselect_lists.md
## Component: Scrollable Multi-Select Interactive Lists

### Objective
Convert ListView from HSplit of individual containers to FormattedTextControl approach for scrolling and multi-selection support.

### Mathematical Implementation Plan

#### Step 1: Add Selection State to ListModel
**File**: `openhcs/tui/components/list_manager.py`
**Action**: Replace line 54 `self.selected_index: int = 0` with these exact lines:

```python
        self.focused_index: int = 0  # Currently focused item (independent of selection)
        self.selected_indices: set[int] = set()  # Track selected items by index
```

**Mathematical Change**:
- OLD: `selected_index` (single selection)
- NEW: `focused_index` (focus) + `selected_indices` (multi-selection)

**Action**: Update existing methods to use `focused_index` instead of `selected_index`:

**Line 72**: Change `self.selected_index = min(self.selected_index, len(items) - 1) if items else 0` to:
```python
        self.focused_index = min(self.focused_index, len(items) - 1) if items else 0
```

**Line 75-79**: Replace entire `select_item` method with:
```python
    def select_item(self, index: int):
        """Select item by index (single selection for compatibility)."""
        if 0 <= index < len(self.items):
            self.focused_index = index
            self._notify_observers()
```

**Line 85**: Change `self.selected_index = index - 1` to:
```python
            self.focused_index = index - 1
```

**Line 94**: Change `self.selected_index = index + 1` to:
```python
            self.focused_index = index + 1
```

**Line 103-104**: Change the selected_index logic to:
```python
            if self.focused_index >= len(self.items) and self.items:
                self.focused_index = len(self.items) - 1
```

**Action**: Add these exact NEW methods after line 107 (after the `remove_item` method):

```python
    def toggle_selection(self, index: int) -> None:
        """Toggle selection state of item at index."""
        if not (0 <= index < len(self.items)):
            return

        if index in self.selected_indices:
            self.selected_indices.remove(index)
        else:
            self.selected_indices.add(index)
        self._notify_observers()

    def get_selected_items(self) -> List[Dict[str, Any]]:
        """Get list of currently selected items."""
        selected_items = []
        for index in self.selected_indices:
            if 0 <= index < len(self.items):
                selected_items.append(self.items[index])
        return selected_items

    def clear_selection(self) -> None:
        """Clear all selections."""
        if self.selected_indices:
            self.selected_indices.clear()
            self._notify_observers()

    def set_focused_index(self, index: int) -> None:
        """Set focused index with bounds checking."""
        if 0 <= index < len(self.items):
            self.focused_index = index
            self._notify_observers()
```

**Validation**: After this step, ListModel should have 4 new methods and 1 new field.

#### Step 2: Add Required Imports to ListView
**File**: `openhcs/tui/components/list_manager.py`
**Action**: Add these exact imports after line 8 (after existing imports):

```python
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.containers import ScrollablePane
from prompt_toolkit.mouse_events import MouseEventType
from prompt_toolkit.filters import Condition
from typing import List, Tuple
```

**Validation**: Import section should now have 6 additional import lines.

#### Step 3: Add Instance Variables to ListView.__init__
**File**: `openhcs/tui/components/list_manager.py`
**Action**: Add these exact lines after line 120 (after `self.config = config`):

```python
        self.list_control: Optional[FormattedTextControl] = None  # Will be created in _create_current_list
        self.allow_multi_select: bool = getattr(config, 'allow_multi_select', False)
```

**Validation**: ListView.__init__ should have 2 new instance variables.

#### Step 4: Replace ListView._create_current_list() Method
**File**: `openhcs/tui/components/list_manager.py`
**Action**: Replace the entire `_create_current_list` method (lines 184-212) with this exact code:

```python
    def _create_current_list(self) -> Container:
        """Create scrollable list using FormattedTextControl approach."""
        if not self.model.items:
            # Return existing empty message logic unchanged
            from prompt_toolkit.layout import Window
            from prompt_toolkit.layout.controls import FormattedTextControl
            from prompt_toolkit.layout.containers import WindowAlign, VSplit, HorizontalAlign

            # Create empty message window with center alignment
            empty_window = Window(
                FormattedTextControl(
                    self.config.empty_message,
                    focusable=False,
                ),
                align=WindowAlign.CENTER,  # Center each line horizontally
                wrap_lines=True,  # Support newlines and wrapping
                dont_extend_width=True,  # Don't expand beyond allocated width
            )

            # Use VSplit to center horizontally
            centered_content = VSplit([
                empty_window
            ], align=HorizontalAlign.CENTER)

            # Use explicit padding for vertical centering
            return HSplit([
                Window(height=Dimension(weight=1)),  # Top padding (flexible)
                centered_content,                     # Content (fixed size)
                Window(height=Dimension(weight=1)),  # Bottom padding (flexible)
            ])

        # Create FormattedTextControl for non-empty list
        self.list_control = FormattedTextControl(
            text=self._generate_list_text,
            focusable=True,
            key_bindings=self._create_list_key_bindings()
        )

        # Attach mouse handler to the control
        self.list_control.mouse_handler = self._create_mouse_handler()

        # Wrap in ScrollablePane for scrolling support
        scrollable_window = Window(content=self.list_control)
        return ScrollablePane(
            scrollable_window,
            height=Dimension(weight=1),
            show_scrollbar=True,
            display_arrows=True
        )
```

**Validation**: Method should return ScrollablePane for non-empty lists, HSplit for empty lists.

#### Step 5: Add Text Generation Method to ListView
**File**: `openhcs/tui/components/list_manager.py`
**Action**: Add this exact method after the `_create_current_list` method:

```python
    def _generate_list_text(self) -> List[Tuple[str, str]]:
        """Generate formatted text for all list items.

        Returns:
            List of (style, text) tuples for FormattedTextControl
        """
        if not self.model.items:
            return [("", "")]  # Empty list fallback

        lines = []

        for i, item_data in enumerate(self.model.items):
            is_selected = i in self.model.selected_indices
            is_focused = i == self.model.focused_index

            # Build checkbox: [x] or [ ]
            checkbox = "[x]" if is_selected else "[ ]"

            # Get display text using config function or fallback to str
            if self.config.display_func:
                display_text = self.config.display_func(item_data, is_selected)
            else:
                display_text = str(item_data)

            # Truncate display text to exactly 45 characters
            if len(display_text) > 45:
                display_text = display_text[:42] + "..."

            # Move buttons: ^ and v (only show if movement is possible)
            up_btn = "^" if self._can_move_up(i, item_data) else " "
            down_btn = "v" if self._can_move_down(i, item_data) else " "

            # Format line with exact spacing:
            # Position 0-3: checkbox "[x] "
            # Position 4-49: display text (45 chars + padding)
            # Position 50-51: up button "^ "
            # Position 52-53: down button "v "
            line_text = f"{checkbox} {display_text:<45} {up_btn} {down_btn}"

            # Add focus indicator and styling
            if is_focused:
                line_text = f"> {line_text}"
                style = "reverse"
            else:
                line_text = f"  {line_text}"
                style = ""

            lines.append((style, line_text))

        return lines
```

**Validation**: Method should return list of (style, text) tuples with exact character positioning.

#### Step 6: Add Key Bindings Method to ListView
**File**: `openhcs/tui/components/list_manager.py`
**Action**: Add this exact method after the `_generate_list_text` method:

```python
    def _create_list_key_bindings(self) -> KeyBindings:
        """Create key bindings for list navigation and selection.

        Returns:
            KeyBindings object with navigation and selection keys
        """
        kb = KeyBindings()

        @kb.add('up')
        def move_up(event):
            """Move focus up one item."""
            if self.model.focused_index > 0:
                self.model.set_focused_index(self.model.focused_index - 1)

        @kb.add('down')
        def move_down(event):
            """Move focus down one item."""
            if self.model.focused_index < len(self.model.items) - 1:
                self.model.set_focused_index(self.model.focused_index + 1)

        @kb.add('space')
        def toggle_selection(event):
            """Toggle selection of focused item."""
            if self.allow_multi_select:
                self.model.toggle_selection(self.model.focused_index)

        @kb.add('enter')
        def select_item(event):
            """Select/activate focused item."""
            self._handle_select(self.model.focused_index)

        return kb
```

**Validation**: Method should return KeyBindings with 4 key handlers (up, down, space, enter).

#### Step 7: Add Mouse Handler Method to ListView
**File**: `openhcs/tui/components/list_manager.py`
**Action**: Add this exact method after the `_create_list_key_bindings` method:

```python
    def _create_mouse_handler(self):
        """Create mouse handler for list interactions.

        Returns:
            Mouse handler function for FormattedTextControl
        """
        def mouse_handler(mouse_event):
            """Handle mouse clicks on list items.

            Click zones (character positions):
            0-3: Checkbox area "[x] "
            4-49: Item text area (45 chars)
            50-51: Up button "^ "
            52-53: Down button "v "
            """
            if mouse_event.event_type != MouseEventType.MOUSE_UP:
                return False

            line_index = mouse_event.position.y
            x_pos = mouse_event.position.x

            # Validate line index
            if not (0 <= line_index < len(self.model.items)):
                return False

            # Update focused index
            self.model.set_focused_index(line_index)

            # Handle click based on x position
            if x_pos <= 3:  # Checkbox area
                if self.allow_multi_select:
                    self.model.toggle_selection(line_index)
            elif 50 <= x_pos <= 51:  # Up button area
                if self._can_move_up(line_index, self.model.items[line_index]):
                    self.model.move_item_up(line_index)
            elif 52 <= x_pos <= 53:  # Down button area
                if self._can_move_down(line_index, self.model.items[line_index]):
                    self.model.move_item_down(line_index)
            else:  # Item text area (4-49)
                self._handle_select(line_index)

            return True

        return mouse_handler
```

**Validation**: Method should return function that handles clicks in 4 distinct zones.

#### Step 8: Add Multi-Select Support to ListConfig
**File**: `openhcs/tui/components/list_manager.py`
**Action**: Add these exact fields to the ListConfig dataclass (after line 46):

```python
    allow_multi_select: bool = False  # Enable checkbox multi-selection
    bulk_button_configs: List[ButtonConfig] = field(default_factory=list)  # Bulk operation buttons
```

**Action**: Update ListView._create_button_bar() method to support bulk buttons.
**Replace lines 148-182** (entire `_create_button_bar` method) with this exact code:

```python
    def _create_button_bar(self) -> Container:
        """Create button bar with regular and bulk operation buttons."""
        from openhcs.tui.components.framed_button import FramedButton
        from prompt_toolkit.layout import Window
        from prompt_toolkit.layout.containers import ConditionalContainer
        from prompt_toolkit.filters import Condition

        regular_buttons = self._create_regular_buttons()

        if self.allow_multi_select and self.config.bulk_button_configs:
            bulk_buttons = self._create_bulk_buttons()
            # Show bulk buttons only when items are selected
            bulk_container = ConditionalContainer(
                bulk_buttons,
                filter=Condition(lambda: len(self.model.selected_indices) > 0)
            )
            return HSplit([regular_buttons, bulk_container])

        return regular_buttons

    def _create_regular_buttons(self) -> Container:
        """Create regular button bar."""
        from openhcs.tui.components.framed_button import FramedButton
        from prompt_toolkit.layout import Window

        if not self.config.button_configs:
            # Return empty container with height 3 if no buttons
            return Window(height=Dimension.exact(3), char=' ')

        buttons = []
        for config in self.config.button_configs:
            # Create framed button (height 3)
            framed_button = FramedButton(
                text=config.text,
                handler=self._wrap_handler(config.handler),
                width=config.width
            )
            if not config.is_enabled():
                framed_button.disabled = True
            buttons.append(framed_button)

        # Add spacers between buttons for even distribution
        spaced_buttons = []
        for i, button in enumerate(buttons):
            if i > 0:
                # Add flexible spacer between buttons
                spaced_buttons.append(Window(width=Dimension(weight=1), char=' '))
            spaced_buttons.append(button)

        # Add spacers at start and end for centering
        if spaced_buttons:
            spaced_buttons.insert(0, Window(width=Dimension(weight=1), char=' '))
            spaced_buttons.append(Window(width=Dimension(weight=1), char=' '))

        return VSplit(spaced_buttons, height=Dimension.exact(3))

    def _create_bulk_buttons(self) -> Container:
        """Create bulk operation buttons."""
        from openhcs.tui.components.framed_button import FramedButton
        from prompt_toolkit.layout import Window

        buttons = []
        for config in self.config.bulk_button_configs:
            # Create button with selection count
            selection_count = len(self.model.selected_indices)
            button_text = f"{config.text} ({selection_count})"

            framed_button = FramedButton(
                text=button_text,
                handler=self._wrap_handler(config.handler),
                width=len(button_text) + 2
            )
            if not config.is_enabled():
                framed_button.disabled = True
            buttons.append(framed_button)

        # Add spacers between buttons for even distribution
        spaced_buttons = []
        for i, button in enumerate(buttons):
            if i > 0:
                spaced_buttons.append(Window(width=Dimension(weight=1), char=' '))
            spaced_buttons.append(button)

        # Add spacers at start and end for centering
        if spaced_buttons:
            spaced_buttons.insert(0, Window(width=Dimension(weight=1), char=' '))
            spaced_buttons.append(Window(width=Dimension(weight=1), char=' '))

        return VSplit(spaced_buttons, height=Dimension.exact(3))
```

**Validation**: ListView should have 3 button-related methods and support conditional bulk buttons.

### Implementation Order (Execute in This Exact Sequence)
1. **Step 1**: Add selection state to ListModel (4 methods + 1 field)
2. **Step 2**: Add imports to ListView (6 import lines)
3. **Step 3**: Add instance variables to ListView.__init__ (2 variables)
4. **Step 4**: Replace ListView._create_current_list() method (complete replacement)
5. **Step 5**: Add _generate_list_text() method (after _create_current_list)
6. **Step 6**: Add _create_list_key_bindings() method (after _generate_list_text)
7. **Step 7**: Add _create_mouse_handler() method (after _create_list_key_bindings)
8. **Step 8**: Add multi-select fields to ListConfig (2 fields)

### File Changes Required
- `openhcs/tui/components/list_manager.py`: Lines 54, 72, 75-79, 85, 94, 103-104, 107, 8, 120, 184-212, 46, 148-182 (model changes + imports + variables + method replacements + config fields)
- `openhcs/tui/panes/plate_manager.py`: Lines 37-49, 377 (config update + bulk operation methods)

### Validation After Each Step
- **Step 1**: `python -c "from openhcs.tui.components.list_manager import ListModel; m=ListModel(); print(hasattr(m, 'focused_index') and hasattr(m, 'selected_indices') and hasattr(m, 'toggle_selection'))"`
- **Step 2**: `python -c "from openhcs.tui.components.list_manager import FormattedTextControl, ScrollablePane, KeyBindings"`
- **Step 3**: `python -c "from openhcs.tui.components.list_manager import ListView, ListConfig; v=ListView(ListModel(), ListConfig('test')); print(hasattr(v, 'list_control') and hasattr(v, 'allow_multi_select'))"`
- **Step 4**: `python -c "from openhcs.tui.components.list_manager import *; m=ListModel(); m.set_items([{'test': 1}]); v=ListView(m, ListConfig('test')); result=v._create_current_list(); print(type(result).__name__)"`
- **Step 5**: `python -c "from openhcs.tui.components.list_manager import *; m=ListModel(); m.set_items([{'test': 1}]); v=ListView(m, ListConfig('test')); result=v._generate_list_text(); print(type(result).__name__ == 'list' and len(result) > 0)"`
- **Step 6**: `python -c "from openhcs.tui.components.list_manager import *; v=ListView(ListModel(), ListConfig('test')); kb=v._create_list_key_bindings(); print(type(kb).__name__)"`
- **Step 7**: `python -c "from openhcs.tui.components.list_manager import *; v=ListView(ListModel(), ListConfig('test')); handler=v._create_mouse_handler(); print(callable(handler))"`
- **Step 8**: `python -c "from openhcs.tui.components.list_manager import ListConfig; c=ListConfig('test'); print(hasattr(c, 'allow_multi_select') and hasattr(c, 'bulk_button_configs'))"`
- **Step 9**: `python -c "from openhcs.tui.panes.plate_manager import PlateManagerPane; print('bulk_delete' in dir(PlateManagerPane))"`

### Testing Commands
```bash
# Test basic functionality
cd /home/ts/code/projects/openhcs
python -m openhcs.tui.main

# Test with multiple items (create test data)
# Verify scrolling with arrow keys
# Verify multi-selection with space key
# Verify mouse clicks on checkboxes, move buttons, and item text
# Verify empty state centering still works
```

#### Step 9: Update PlateManager for Bulk Operations
**File**: `openhcs/tui/panes/plate_manager.py`
**Action**: Replace lines 37-49 (ListConfig creation) with this exact code:

```python
        # Create list manager configuration with multi-selection support
        config = ListConfig(
            title="Plate Manager",
            allow_multi_select=True,  # Enable multi-selection
            button_configs=[
                ButtonConfig("Add", self._handle_add_plates, width=len("Add") + 2),
                ButtonConfig("Del", self._handle_delete_plates, width=len("Del") + 2),
                ButtonConfig("Edit", self._handle_edit_plate, width=len("Edit") + 2),
                ButtonConfig("Init", self._handle_initialize_plates, width=len("Init") + 2),
                ButtonConfig("Compile", self._handle_compile_plates, width=len("Compile") + 2),
                ButtonConfig("Run", self._handle_run_plates, width=len("Run") + 2),
            ],
            bulk_button_configs=[
                ButtonConfig("Del Selected", self._handle_bulk_delete, width=len("Del Selected") + 2),
                ButtonConfig("Init Selected", self._handle_bulk_initialize, width=len("Init Selected") + 2),
                ButtonConfig("Compile Selected", self._handle_bulk_compile, width=len("Compile Selected") + 2),
                ButtonConfig("Run Selected", self._handle_bulk_run, width=len("Run Selected") + 2),
            ],
            display_func=self._get_display_text,
            empty_message="Click 'Add' to add plates.\n\nStatus: ? = added, - = initialized, o = compiled, ! = running"
        )
```

**Action**: Add these exact bulk operation methods after line 377 (after `_get_current_pipeline_definition`):

```python
    # Bulk operation handlers
    async def _handle_bulk_delete(self):
        """Delete all selected plates."""
        selected_items = self.list_manager.model.get_selected_items()
        if not selected_items:
            return

        # Remove selected items from model
        remaining_items = []
        selected_paths = {item['path'] for item in selected_items}

        for item in self.list_manager.model.items:
            if item['path'] not in selected_paths:
                remaining_items.append(item)
            else:
                # Clean up orchestrator if exists
                if item['path'] in self.orchestrators:
                    del self.orchestrators[item['path']]
                if item['path'] in self.pipelines:
                    del self.pipelines[item['path']]
                if item['path'] in self.plate_configs:
                    del self.plate_configs[item['path']]

        self.list_manager.load_items(remaining_items)
        self.list_manager.model.clear_selection()
        logger.info(f"Deleted {len(selected_items)} selected plates")

    async def _handle_bulk_initialize(self):
        """Initialize all selected plates."""
        selected_items = self.list_manager.model.get_selected_items()
        if not selected_items:
            return
        await self._initialize_plates_list(selected_items)

    async def _handle_bulk_compile(self):
        """Compile all selected plates."""
        selected_items = self.list_manager.model.get_selected_items()
        if not selected_items:
            return
        await self._compile_plates_list(selected_items)

    async def _handle_bulk_run(self):
        """Run all selected plates."""
        selected_items = self.list_manager.model.get_selected_items()
        if not selected_items:
            return
        await self._run_plates_list(selected_items)

    # Helper methods for bulk operations
    async def _initialize_plates_list(self, plates: List[Dict[str, Any]]):
        """Initialize specific list of plates."""
        try:
            for plate_data in plates:
                try:
                    plate_path = plate_data['path']
                    plate_config = self.plate_configs.get(plate_path, self.state.global_config)

                    loop = asyncio.get_event_loop()
                    orchestrator = await loop.run_in_executor(
                        None,
                        lambda: PipelineOrchestrator(plate_path, global_config=plate_config).initialize()
                    )

                    self.orchestrators[plate_path] = orchestrator
                    plate_data['status'] = '-'

                except Exception as e:
                    logger.error(f"Error initializing {plate_data['name']}: {e}")
                    plate_data['status'] = '!'

            self.list_manager.load_items(self.list_manager.model.items)
            logger.info(f"Bulk initialized {len(plates)} plates")

        except Exception as e:
            logger.error(f"Error in bulk initialize: {e}", exc_info=True)

    async def _compile_plates_list(self, plates: List[Dict[str, Any]]):
        """Compile specific list of plates."""
        try:
            for plate_data in plates:
                try:
                    plate_path = plate_data['path']
                    if plate_path not in self.orchestrators:
                        continue  # Skip uninitialized plates

                    orchestrator = self.orchestrators[plate_path]
                    pipeline_definition = self._get_current_pipeline_definition()

                    loop = asyncio.get_event_loop()
                    wells = await loop.run_in_executor(None, orchestrator.get_wells)
                    compiled_contexts = await loop.run_in_executor(
                        None,
                        lambda: orchestrator.compile_pipelines(
                            pipeline_definition=pipeline_definition,
                            well_filter=wells
                        )
                    )

                    self.pipelines[plate_path] = {
                        'pipeline_definition': pipeline_definition,
                        'compiled_contexts': compiled_contexts
                    }
                    plate_data['status'] = 'o'

                except Exception as e:
                    logger.error(f"Error compiling {plate_data['name']}: {e}")
                    plate_data['status'] = '!'

            self.list_manager.load_items(self.list_manager.model.items)
            logger.info(f"Bulk compiled {len(plates)} plates")

        except Exception as e:
            logger.error(f"Error in bulk compile: {e}", exc_info=True)

    async def _run_plates_list(self, plates: List[Dict[str, Any]]):
        """Run specific list of plates."""
        try:
            for plate_data in plates:
                try:
                    plate_path = plate_data['path']
                    if plate_path not in self.pipelines:
                        continue  # Skip uncompiled plates

                    orchestrator = self.orchestrators[plate_path]
                    pipeline_data = self.pipelines[plate_path]

                    plate_data['status'] = '!'  # Running
                    self.list_manager.load_items(self.list_manager.model.items)

                    loop = asyncio.get_event_loop()
                    results = await loop.run_in_executor(
                        None,
                        lambda: orchestrator.execute_compiled_plate(
                            pipeline_definition=pipeline_data['pipeline_definition'],
                            compiled_contexts=pipeline_data['compiled_contexts']
                        )
                    )

                    if results and all(r.get('status') != 'error' for r in results.values()):
                        plate_data['status'] = 'o'
                    else:
                        plate_data['status'] = '!'

                except Exception as e:
                    logger.error(f"Error executing {plate_data['name']}: {e}")
                    plate_data['status'] = '!'

            self.list_manager.load_items(self.list_manager.model.items)
            logger.info(f"Bulk executed {len(plates)} plates")

        except Exception as e:
            logger.error(f"Error in bulk run: {e}", exc_info=True)
```

### Expected Behavior After Implementation
1. **Empty lists**: Centered message as before
2. **Non-empty lists**: Scrollable FormattedTextControl with checkboxes and move buttons
3. **Keyboard navigation**: Up/down arrows move focus, space toggles selection
4. **Mouse interaction**: Click zones work for checkbox, move buttons, item selection
5. **Multi-selection**: Selected items show [x], unselected show [ ]
6. **Move buttons**: ^ and v appear inline, only clickable when movement possible
7. **Bulk operations**: Bulk buttons appear when items selected, operate on selected items only

uldit be hard 