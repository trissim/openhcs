# plan_04_generic_list_manager.md
## Component: Generic List Manager Base Component

### Objective
Build a reusable base component that handles the common pattern of [title bar + button toolbar + interactive list]. Both plate manager and pipeline editor share this identical structure with only different data objects and button handlers.

### Plan
1. Create generic ListManagerComponent with configurable title, buttons, and item renderers
2. Implement generic selection and multi-selection logic
3. Build configurable button toolbar system with action dispatch
4. Create pluggable item renderer system for different display formats
5. Add generic keyboard navigation and focus management
6. Implement status update system with configurable status indicators

### Findings
Pattern analysis from existing TUI1 components:
- PlateManagerPane and PipelineEditorPane follow identical structure
- Only differences: title text, button configs, item rendering, action handlers
- Both use same UI patterns: HSplit([title, buttons, scrollable_list])
- Both need selection, keyboard navigation, async action dispatch
- Both use status indicators (plates: o/!/✓/?, steps: position arrows)

**Crystal Clear Generic Component Structure:**
```
┌─────────────────────────┐
│ Configurable Title      │ ← Title text varies
├─────────────────────────┤
│ [Configurable Buttons]  │ ← Button configs vary
├─────────────────────────┤
│ icon item1         ↑↓   │ ← Item rendering varies
│ icon item2         ↑↓   │ ← Icons on LEFT side
│ icon item3         ↑↓   │ ← Scrollbar when needed
│ │                       │
│ ▼                       │
└─────────────────────────┘
```

**Configuration Architecture:**
```python
@dataclass
class ListManagerConfig:
    title: str
    buttons: List[ButtonConfig]
    item_renderer: ItemRenderer
    action_dispatcher: ActionDispatcher

@dataclass
class ButtonConfig:
    text: str
    width: int
    action_handler: Callable

class ItemRenderer:
    def render_item(self, item: Any) -> str
    def get_status_icon(self, item: Any) -> str
```

**Icon Positioning Requirements:**
- Icons always on LEFT side of items
- Plates: `o plate1    ↑↓` (status + arrows)
- Steps: `↑↓ step1` (just arrows)
- Scrollbars appear automatically when content overflows

### Implementation Draft
(Only after smell loop passes)
