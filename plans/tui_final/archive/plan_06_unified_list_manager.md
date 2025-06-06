# plan_06_unified_list_manager.md
## Component: Unified List Manager Pane Architecture

### Objective
Create a unified `ListManagerPane` base class to eliminate code duplication between `PlateManagerPane` and `PipelineEditorPane`. Both components follow identical architectural patterns and can be abstracted into a reusable list management interface.

### Plan
1. **Create ListManagerPane Base Class**
   - Abstract the common list management patterns
   - Provide configurable button bars and display functions
   - Handle all InteractiveListItem coordination
   - Manage selection, movement, and container rebuilding

2. **Refactor PlateManagerPane and PipelineEditorPane**
   - Inherit from ListManagerPane
   - Implement abstract methods for customization
   - Remove duplicated infrastructure code
   - Preserve existing functionality and interfaces

3. **Create ButtonConfig System**
   - Declarative button configuration
   - Support for async handlers
   - Conditional button enabling/disabling
   - Consistent button styling and layout

4. **Implement UI Abstraction Layer**
   - Common container building logic
   - Shared selection and movement handling
   - Unified update and refresh mechanisms
   - Consistent styling and layout patterns

### Findings

#### **Current Code Duplication Analysis**

Both `PlateManagerPane` and `PipelineEditorPane` implement nearly identical patterns:

**Shared Infrastructure (500+ lines each):**
- Data management: `self.items: List[Dict[str, Any]]`
- Selection tracking: `self.selected_index: int`
- Container building: `_build_items_container() -> HSplit`
- Item widget creation: `_create_item_widget(index, data) -> InteractiveListItem`
- Selection callbacks: `_handle_item_select(index)`
- Movement callbacks: `_handle_item_move_up/down(index)`
- Update mechanism: `_update_selection()` rebuilds container

**Identical UI Structure:**
```
Frame(
    HSplit([
        Label("Title"),                    # Title bar
        VSplit([Button(...), ...]),        # Button bar  
        DynamicContainer(get_items_list)   # Interactive list
    ]),
    title="Component Name"
)
```

**Only Differences:**
- Button configurations (Add/Del/Edit vs Add/Del/Edit/Load/Save)
- Display text formatting functions
- Item-specific business logic in handlers
- Movement validation rules

#### **InteractiveListItem Integration**

Both components use `InteractiveListItem` identically:
- Pass `item_data`, `item_index`, `is_selected`
- Provide `display_text_func` callback
- Handle `on_select`, `on_move_up`, `on_move_down` callbacks
- Manage `can_move_up`, `can_move_down` validation

#### **Architectural Opportunity**

**Code Reduction Estimate:**
- **Current**: 500 lines × 2 components = 1000 lines
- **With abstraction**: 300 lines base + 100 lines × 2 = 500 lines
- **50% reduction** while improving maintainability

### Implementation Draft

#### **1. ButtonConfig System**

```python
@dataclass
class ButtonConfig:
    """Configuration for a button in the button bar."""
    text: str
    handler: Callable[[], Any]
    width: Optional[int] = None
    enabled_func: Optional[Callable[[], bool]] = None
    
    def is_enabled(self) -> bool:
        """Check if button should be enabled."""
        return self.enabled_func() if self.enabled_func else True
```

#### **2. Clean Architecture Components**

```python
@dataclass
class ListConfig:
    """Configuration for a list manager pane."""
    title: str
    frame_title: Optional[str] = None
    button_configs: List[ButtonConfig] = field(default_factory=list)
    display_func: Optional[Callable[[Dict[str, Any], bool], str]] = None
    can_move_up_func: Optional[Callable[[int, Dict[str, Any]], bool]] = None
    can_move_down_func: Optional[Callable[[int, Dict[str, Any]], bool]] = None
    empty_message: str = "No items available."

class ListModel:
    """Model for list state - pure data, no UI concerns."""

    def __init__(self):
        self.items: List[Dict[str, Any]] = []
        self.selected_index: int = 0
        self._observers: List[Callable] = []

    def add_observer(self, callback: Callable):
        """Add observer for model changes."""
        self._observers.append(callback)

    def _notify_observers(self):
        """Notify all observers of model changes."""
        for callback in self._observers:
            callback()

    def set_items(self, items: List[Dict[str, Any]]):
        """Set items and notify observers."""
        self.items = items
        self.selected_index = min(self.selected_index, len(items) - 1) if items else 0
        self._notify_observers()

    def select_item(self, index: int):
        """Select item by index."""
        if 0 <= index < len(self.items):
            self.selected_index = index
            self._notify_observers()

    def move_item_up(self, index: int) -> bool:
        """Move item up, return True if moved."""
        if index > 0 and index < len(self.items):
            self.items[index], self.items[index - 1] = self.items[index - 1], self.items[index]
            self.selected_index = index - 1
            self._notify_observers()
            return True
        return False

    def move_item_down(self, index: int) -> bool:
        """Move item down, return True if moved."""
        if index >= 0 and index < len(self.items) - 1:
            self.items[index], self.items[index + 1] = self.items[index + 1], self.items[index]
            self.selected_index = index + 1
            self._notify_observers()
            return True
        return False

    def remove_item(self, index: int) -> bool:
        """Remove item, return True if removed."""
        if 0 <= index < len(self.items):
            self.items.pop(index)
            if self.selected_index >= len(self.items) and self.items:
                self.selected_index = len(self.items) - 1
            self._notify_observers()
            return True
        return False

class ListView:
    """View for list display - observes model, updates UI automatically."""

    def __init__(self, model: ListModel, config: ListConfig):
        self.model = model
        self.config = config

        # UI components (built once, updated automatically)
        self.container = self._build_container()

        # Observe model changes
        self.model.add_observer(self._on_model_changed)

    def _build_container(self) -> Container:
        """Build container once - updates automatically via DynamicContainer."""
        # Title bar
        title_bar = Label(self.config.title)

        # Button bar
        button_bar = self._create_button_bar()

        # Dynamic list that updates when model changes
        def get_current_list():
            return self._create_current_list()

        dynamic_list = DynamicContainer(get_current_list)

        return Frame(
            HSplit([title_bar, button_bar, dynamic_list]),
            title=self.config.frame_title or self.config.title
        )

    def _create_button_bar(self) -> VSplit:
        """Create button bar from configs."""
        buttons = []
        for config in self.config.button_configs:
            button = Button(
                text=config.text,
                handler=self._wrap_handler(config.handler),
                width=config.width
            )
            if not config.is_enabled():
                button.disabled = True
            buttons.append(button)

        return VSplit(buttons, padding=1)

    def _create_current_list(self) -> HSplit:
        """Create current list state - called automatically when model changes."""
        if not self.model.items:
            return HSplit([Label(self.config.empty_message)])

        item_widgets = []
        for index, item_data in enumerate(self.model.items):
            widget = self._create_item_widget(index, item_data)
            item_widgets.append(widget)

        return HSplit(item_widgets)

    def _create_item_widget(self, index: int, item_data: Dict[str, Any]) -> InteractiveListItem:
        """Create InteractiveListItem for an item."""
        is_selected = (index == self.model.selected_index)

        return InteractiveListItem(
            item_data=item_data,
            item_index=index,
            is_selected=is_selected,
            display_text_func=self.config.display_func,
            on_select=self._handle_select,
            on_move_up=self._handle_move_up,
            on_move_down=self._handle_move_down,
            can_move_up=self._can_move_up(index, item_data),
            can_move_down=self._can_move_down(index, item_data)
        )

    def _can_move_up(self, index: int, item_data: Dict[str, Any]) -> bool:
        """Check if item can move up."""
        if self.config.can_move_up_func:
            return self.config.can_move_up_func(index, item_data)
        return index > 0

    def _can_move_down(self, index: int, item_data: Dict[str, Any]) -> bool:
        """Check if item can move down."""
        if self.config.can_move_down_func:
            return self.config.can_move_down_func(index, item_data)
        return index < len(self.model.items) - 1

    def _handle_select(self, index: int):
        """Handle item selection."""
        self.model.select_item(index)

    def _handle_move_up(self, index: int):
        """Handle move up."""
        self.model.move_item_up(index)

    def _handle_move_down(self, index: int):
        """Handle move down."""
        self.model.move_item_down(index)

    def _wrap_handler(self, handler: Callable) -> Callable:
        """Wrap handler for async support."""
        def wrapped():
            if asyncio.iscoroutinefunction(handler):
                get_app().create_background_task(handler())
            else:
                handler()
        return wrapped

    def _on_model_changed(self):
        """Called when model changes - triggers UI update."""
        get_app().invalidate()

class ListManagerPane:
    """
    Clean list manager pane - pure composition, no inheritance.

    Coordinates model, view, and business logic without architectural anti-patterns.
    """

    def __init__(self, config: ListConfig, backend: Any):
        """Pure setup - no work in constructor."""
        self.config = config
        self.backend = backend

        # Clean MVC components
        self.model = ListModel()
        self.view = ListView(self.model, config)

        # Observe model for business logic
        self.model.add_observer(self._on_model_changed)

    @property
    def container(self) -> Container:
        """Get the UI container."""
        return self.view.container

    def load_items(self, items: List[Dict[str, Any]]):
        """Load items - just update model, view updates automatically."""
        self.model.set_items(items)

    def get_selected_item(self) -> Optional[Dict[str, Any]]:
        """Get currently selected item."""
        if 0 <= self.model.selected_index < len(self.model.items):
            return self.model.items[self.model.selected_index]
        return None

    def _on_model_changed(self):
        """Handle model changes for business logic."""
        # Override in subclasses for business logic
        pass
```

#### **3. Clean PlateManagerPane**

```python
class PlateManagerPane:
    """Clean plate manager using composition."""

    def __init__(self, state, context: ProcessingContext, storage_registry: Any):
        self.state = state
        self.context = context
        self.storage_registry = storage_registry

        # Business logic state
        self.orchestrators: Dict[str, Any] = {}
        self.status: Dict[str, str] = {}
        self.pipelines: Dict[str, List] = {}

        # Create configuration
        config = ListConfig(
            title="Plate Manager",
            button_configs=[
                ButtonConfig("Add", self._handle_add_plates),
                ButtonConfig("Del", self._handle_delete_plates,
                            enabled_func=lambda: len(self.list_manager.model.items) > 0),
                ButtonConfig("Edit", self._handle_edit_plate,
                            enabled_func=lambda: self.list_manager.get_selected_item() is not None),
                ButtonConfig("Init", self._handle_initialize_plates),
                ButtonConfig("Compile", self._handle_compile_plates),
                ButtonConfig("Run", self._handle_run_plates),
            ],
            display_func=self._get_display_text,
            empty_message="Click 'Add' to add plates. Status: ? = added, - = initialized, o = compiled, ! = running"
        )

        # Create list manager
        self.list_manager = ListManagerPane(config, storage_registry)
        self.list_manager._on_model_changed = self._on_selection_changed

    @property
    def container(self) -> Container:
        """Get the UI container."""
        return self.list_manager.container

    def _get_display_text(self, plate_data: Dict[str, Any], is_selected: bool) -> str:
        """Generate display text for a plate."""
        status_symbol = self._get_status_symbol(plate_data.get('status', '?'))
        name = plate_data.get('name', 'Unknown Plate')
        path = plate_data.get('path', 'Unknown Path')
        return f"{status_symbol} {name} | {path}"

    def _get_status_symbol(self, status: str) -> str:
        """Get status symbol for plate."""
        status_map = {
            '?': '?',  # Added but not initialized
            '-': '-',  # Initialized but not compiled
            'o': 'o',  # Compiled and ready
            '!': '!',  # Running or error
        }
        return status_map.get(status, '?')

    def _on_selection_changed(self):
        """Handle selection changes."""
        selected_item = self.list_manager.get_selected_item()
        if selected_item and hasattr(self.state, 'set_selected_plate'):
            self.state.set_selected_plate(selected_item)

    # Business logic handlers (unchanged)
    async def _handle_add_plates(self):
        """Handle Add Plates button."""
        # Existing implementation unchanged
        # When plates are added, call: self.list_manager.load_items(new_plates)
        pass

    async def _handle_delete_plates(self):
        """Handle Delete Plates button."""
        selected_item = self.list_manager.get_selected_item()
        if selected_item:
            # Remove from business logic
            # Update model: self.list_manager.model.remove_item(selected_index)
            pass

    # ... other handlers unchanged
```

#### **4. Clean PipelineEditorPane**

```python
class PipelineEditorPane:
    """Clean pipeline editor using composition."""

    def __init__(self, state, context: ProcessingContext):
        self.state = state
        self.context = context
        self.steps_lock = asyncio.Lock()

        # Create configuration
        config = ListConfig(
            title="Pipeline Editor",
            frame_title="Steps",
            button_configs=[
                ButtonConfig("Add", self._handle_add_step),
                ButtonConfig("Del", self._handle_delete_step,
                            enabled_func=lambda: len(self.list_manager.model.items) > 0),
                ButtonConfig("Edit", self._handle_edit_step,
                            enabled_func=lambda: self.list_manager.get_selected_item() is not None),
                ButtonConfig("Load", self._handle_load_pipeline),
                ButtonConfig("Save", self._handle_save_pipeline,
                            enabled_func=lambda: len(self.list_manager.model.items) > 0),
            ],
            display_func=self._get_display_text,
            can_move_up_func=self._can_move_up,
            can_move_down_func=self._can_move_down,
            empty_message="No steps available. Select a plate first."
        )

        # Create list manager
        self.list_manager = ListManagerPane(config, context)
        self.list_manager._on_model_changed = self._on_selection_changed

        # Load initial steps
        initial_steps = getattr(state, 'current_pipeline_definition', [])
        self.list_manager.load_items(initial_steps)

    @property
    def container(self) -> Container:
        """Get the UI container."""
        return self.list_manager.container

    def _get_display_text(self, step_data: Dict[str, Any], is_selected: bool) -> str:
        """Generate display text for a step."""
        status_icon = self._get_status_icon(step_data.get('status', 'unknown'))
        name = step_data.get('name', 'Unknown Step')
        func_name = self._get_function_name(step_data)
        output_memory_type = step_data.get('output_memory_type', '[N/A]')
        return f"{status_icon} {name} | {func_name} → {output_memory_type}"

    def _can_move_up(self, index: int, step_data: Dict[str, Any]) -> bool:
        """Check if step can be moved up (within same pipeline)."""
        if index <= 0:
            return False
        items = self.list_manager.model.items
        current_pipeline_id = step_data.get('pipeline_id')
        prev_pipeline_id = items[index - 1].get('pipeline_id')
        return current_pipeline_id == prev_pipeline_id

    def _can_move_down(self, index: int, step_data: Dict[str, Any]) -> bool:
        """Check if step can be moved down (within same pipeline)."""
        items = self.list_manager.model.items
        if index >= len(items) - 1:
            return False
        current_pipeline_id = step_data.get('pipeline_id')
        next_pipeline_id = items[index + 1].get('pipeline_id')
        return current_pipeline_id == next_pipeline_id

    def _on_selection_changed(self):
        """Handle selection changes."""
        selected_item = self.list_manager.get_selected_item()
        if selected_item and hasattr(self.state, 'set_selected_step'):
            self.state.set_selected_step(selected_item)

    # Business logic helpers
    def _get_status_icon(self, status: str) -> str:
        """Get status icon for step."""
        # Existing implementation
        pass

    def _get_function_name(self, step_data: Dict[str, Any]) -> str:
        """Get function name for step."""
        # Existing implementation
        pass

    # Business logic handlers (unchanged)
    async def _handle_add_step(self):
        """Handle Add Step button."""
        # Existing implementation unchanged
        # When steps are added, call: self.list_manager.load_items(updated_steps)
        pass

    # ... other handlers unchanged
```

### UI Sketches

#### **Current Duplicated Structure**
```
┌─ PlateManagerPane ──────────────────┐  ┌─ PipelineEditorPane ────────────────┐
│ Plate Manager                       │  │ Pipeline Editor                     │
│ [Add][Del][Edit][Init][Compile][Run]│  │ [Add][Del][Edit][Load][Save]        │
│ ┌─ Steps ─────────────────────────┐ │  │ ┌─ Steps ─────────────────────────┐ │
│ │ ? plate1 | /path/to/plate1      │ │  │ │ o step1 | load_images → zarr    │ │
│ │ - plate2 | /path/to/plate2      │ │  │ │ o step2 | stitch → tiff         │ │
│ │ o plate3 | /path/to/plate3      │ │  │ │ ? step3 | segment → mask        │ │
│ └─────────────────────────────────┘ │  │ └─────────────────────────────────┘ │
└─────────────────────────────────────┘  └─────────────────────────────────────┘
```

#### **Unified ListManagerPane Structure**
```
┌─ ListManagerPane Base ──────────────────────────────────────────────────────┐
│ ┌─ Title Bar ─────────────────────────────────────────────────────────────┐ │
│ │ {configurable_title}                                                    │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│ ┌─ Button Bar ────────────────────────────────────────────────────────────┐ │
│ │ {ButtonConfig[]} → [Btn1][Btn2][Btn3][...]                             │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│ ┌─ Interactive List ──────────────────────────────────────────────────────┐ │
│ │ ┌─ InteractiveListItem ─────────────────────────────────────────────┐   │ │
│ │ │ {display_text_func(item_data, is_selected)} [^][v]               │   │ │
│ │ └───────────────────────────────────────────────────────────────────┘   │ │
│ │ ┌─ InteractiveListItem ─────────────────────────────────────────────┐   │ │
│ │ │ {display_text_func(item_data, is_selected)} [^][v]               │   │ │
│ │ └───────────────────────────────────────────────────────────────────┘   │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### **Specialized Implementations**
```
PlateManagerPane(ListManagerPane):
├─ title="Plate Manager"
├─ buttons=[Add, Del, Edit, Init, Compile, Run]
├─ display_text_func=lambda: f"{status} {name} | {path}"
└─ business_logic=[orchestrator management, multi-folder selection, etc.]

PipelineEditorPane(ListManagerPane):
├─ title="Pipeline Editor", frame_title="Steps"
├─ buttons=[Add, Del, Edit, Load, Save]
├─ display_text_func=lambda: f"{status} {name} | {func} → {output}"
└─ business_logic=[step management, pipeline serialization, etc.]
```

### Architectural Reasoning

#### **1. Separation of Concerns**

**Infrastructure vs Business Logic:**
- **ListManagerPane**: Handles UI infrastructure (layout, selection, movement, InteractiveListItem coordination)
- **Subclasses**: Handle domain-specific business logic (orchestrators, steps, file operations)

**Benefits:**
- Clear separation between UI patterns and domain logic
- Reusable infrastructure for future list-based components
- Easier testing (mock business logic, test UI patterns separately)

#### **2. Configuration Over Code**

**ButtonConfig System:**
- Declarative button configuration instead of hardcoded UI building
- Conditional enabling/disabling through `enabled_func`
- Consistent styling and behavior across all list managers

**Benefits:**
- No duplicated button creation code
- Easy to add/remove/modify buttons
- Consistent UX patterns across components

#### **3. Template Method Pattern**

**Abstract Methods for Customization:**
- `get_display_text()`: Format item display
- `can_move_up/down()`: Movement validation rules
- `load_initial_items()`: Data loading strategy
- `on_item_selected()`: Selection handling

**Benefits:**
- Enforces consistent interface while allowing customization
- Prevents accidental breaking of shared infrastructure
- Clear extension points for new functionality

#### **4. Composition Over Inheritance**

**InteractiveListItem Integration:**
- ListManagerPane composes InteractiveListItem widgets
- Provides callbacks and configuration
- Handles coordination and state management

**Benefits:**
- Leverages existing InteractiveListItem functionality
- No need to modify InteractiveListItem for new features
- Clean separation between list item and list manager concerns

#### **5. Migration Strategy**

**Backward Compatibility:**
- Existing public interfaces preserved
- Internal implementation replaced with base class
- Business logic handlers unchanged

**Benefits:**
- Zero breaking changes to existing code
- Gradual migration possible
- Immediate code reduction benefits

#### **6. Future Extensibility**

**New List-Based Components:**
- File browser lists
- Configuration option lists
- Log/history viewers
- Any component following title+buttons+list pattern

**Benefits:**
- Consistent UX patterns across entire TUI
- Rapid development of new list-based features
- Maintainable codebase with shared infrastructure
