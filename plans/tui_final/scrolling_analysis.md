# ScrollablePane Architecture Analysis

## Current Understanding

### ScrollablePane Design
From source analysis of `/prompt_toolkit/layout/scrollable_pane.py`:

1. **Core Architecture**: ScrollablePane renders content to a virtual screen, then copies visible portion to real screen
2. **Automatic Scrolling**: Handles cursor visibility and focused window visibility automatically
3. **Built-in Scrollbar**: Has native scrollbar support with `show_scrollbar=True`
4. **Mouse Wheel Support**: Should work automatically via mouse event system

### Mouse Wheel Architecture
From `/prompt_toolkit/key_binding/bindings/mouse.py`:

1. **Mouse Events**: SCROLL_UP and SCROLL_DOWN are handled at the mouse event level
2. **Key Mapping**: ScrollUp/ScrollDown events map to `<scroll-up>` and `<scroll-down>` keys
3. **Fallback Behavior**: When no specific handler exists, scroll events become Up/Down arrow keys

### Current Implementation Issues

#### 1. Container Architecture Problem
**Current (BROKEN)**:
```python
ScrollablePane(
    Window(content=FormattedTextControl(text=dynamic_text))  # Single control
)
```

**Our Implementation (PROBLEMATIC)**:
```python
ScrollablePane(
    Window(content=DynamicContainer(self._build_item_list))  # Dynamic container
)
```

**Issue**: DynamicContainer returns HSplit with multiple Windows, breaking ScrollablePane's expectation of single content.

#### 2. Focus and Mouse Event Routing
**Problem**: FormattedTextControl in our implementation has `focusable=False`, preventing mouse events from reaching it.

**Solution**: Use single FormattedTextControl with dynamic text generation.

#### 3. Key Binding Integration
**Missing**: No key bindings for scroll events in our file browser.

## Clean Solution Architecture

### 1. Proper ScrollablePane Usage
```python
# Single FormattedTextControl with dynamic text
self.item_list_control = FormattedTextControl(
    text=self._get_item_list_text,  # Dynamic text function
    focusable=True,  # CRITICAL: Must be focusable for mouse events
    key_bindings=self._create_key_bindings()  # Include scroll bindings
)

self.scrollable_pane = ScrollablePane(
    Window(
        content=self.item_list_control,
        height=Dimension(min=10, max=30)  # Bounded height
    ),
    show_scrollbar=True,  # Native scrollbar
    display_arrows=True   # Up/down arrows
)
```

### 2. Mouse Event Handling
```python
def _get_item_list_text(self):
    """Generate formatted text with mouse handlers."""
    if not self.listing:
        return "Loading..." if self.loading else "(empty directory)"

    lines = []
    max_name_width = min(60, max(20, max(len(item.name) for item in self.listing) + 6))

    for i, item in enumerate(self.listing):
        is_selected = item.path in self.selected_paths
        is_focused = i == self.focused_index

        # Build display text
        prefix = ""
        if self.allow_multiple:
            prefix = "[x] " if is_selected else "[ ] "

        icon = "📁" if item.is_dir else "📄"
        name_part = f"{prefix}{icon} {item.name}"

        display = f"{name_part:<{max_name_width}}"
        if not item.is_dir:
            display += f"{item.display_size:>10}  "
        else:
            display += f"{'':>10}  "
        display += item.display_mtime

        # Add selection/focus styling
        style = ""
        if is_focused:
            style = "reverse"
            display = f"> {display}"
        else:
            display = f"  {display}"

        # Create mouse handler for this line
        def make_handler(index):
            def handler(mouse_event):
                if mouse_event.event_type == MouseEventType.MOUSE_UP:
                    current_time = time.time()

                    # Check for double-click
                    is_double_click = (
                        index == self.last_click_index and
                        current_time - self.last_click_time < 0.5
                    )

                    # Update click tracking
                    self.last_click_time = current_time
                    self.last_click_index = index

                    # Handle click based on position and type
                    x_pos = mouse_event.position.x

                    if self.allow_multiple and x_pos <= 4:  # Checkbox area
                        self._set_focus(index)
                        self._toggle_selection()
                    elif is_double_click:
                        # Double click - navigate or select
                        item = self.listing[index]
                        if item.is_dir:
                            self._navigate_to(item.path)
                        else:
                            self._set_focus(index)
                            self._on_activate()
                    else:
                        # Single click - just focus
                        self._set_focus(index)

                    return True  # Event handled
                return False  # Event not handled
            return handler

        # Add line with mouse handler - format: (style, text, mouse_handler)
        lines.append((style, display, make_handler(i)))
        lines.append(("", "\n"))  # Newline

    # Remove last newline
    if lines:
        lines.pop()

    return lines
```

### 3. Key Bindings for Scrolling
```python
def _create_key_bindings(self):
    kb = KeyBindings()
    
    # Standard navigation
    @kb.add('up')
    def _(event): self._move_focus(-1)
    
    @kb.add('down') 
    def _(event): self._move_focus(1)
    
    # Mouse wheel support
    @kb.add('<scroll-up>')
    def _(event): self._move_focus(-3)  # Scroll 3 lines up
    
    @kb.add('<scroll-down>')
    def _(event): self._move_focus(3)   # Scroll 3 lines down
    
    # Page navigation
    @kb.add('pageup')
    def _(event): self._move_focus(-10)
    
    @kb.add('pagedown')
    def _(event): self._move_focus(10)
    
    return kb
```

## Implementation Changes Needed

### 1. Replace DynamicContainer with FormattedTextControl
- Remove `_build_item_list()` method that returns Container
- Replace with `_get_item_list_text()` that returns formatted text
- Make FormattedTextControl focusable

### 2. Simplify Mouse Handling
- Remove manual scrollbar click handling (ScrollablePane handles this)
- Remove manual scroll position tracking
- Use FormattedTextControl's built-in mouse support

### 3. Add Proper Key Bindings
- Add scroll-up/scroll-down bindings for mouse wheel
- Ensure key bindings are attached to the control

## Benefits of Clean Solution

1. **Native Scrolling**: ScrollablePane handles all scroll logic automatically
2. **Mouse Wheel Works**: Built-in support via key binding system
3. **Simpler Code**: Remove manual scroll tracking and scrollbar handling
4. **Better Performance**: Single control vs multiple windows
5. **Consistent Behavior**: Follows prompt_toolkit patterns

## Implementation Plan

### Phase 1: Clean Architecture Refactor
1. **Replace DynamicContainer with FormattedTextControl**
   - Remove `_build_item_list()` method
   - Add `_get_item_list_text()` method returning formatted text with mouse handlers
   - Make FormattedTextControl focusable=True

2. **Simplify ScrollablePane Usage**
   - Remove manual scrollbar click handling code
   - Remove manual scroll position tracking
   - Use native ScrollablePane scrolling

3. **Add Mouse Wheel Key Bindings**
   - Add `<scroll-up>` and `<scroll-down>` key bindings
   - Map to focus movement for smooth scrolling

### Phase 2: Testing and Validation
1. **Test Mouse Wheel Scrolling**
   - Verify mouse wheel events are received
   - Test smooth scrolling behavior
   - Validate focus tracking during scroll

2. **Test Mouse Click Handling**
   - Single click focus
   - Double click navigation
   - Checkbox area clicking (multi-select mode)

3. **Test Keyboard Navigation**
   - Arrow keys
   - Page up/down
   - Home/end keys

### Phase 3: Performance Optimization
1. **Optimize Text Generation**
   - Cache formatted text when listing unchanged
   - Minimize string operations
   - Efficient mouse handler creation

2. **Memory Management**
   - Proper cleanup of mouse handlers
   - Avoid memory leaks in closures

## Next Steps

1. **IMMEDIATE**: Implement Phase 1 refactor in file_browser.py
2. **TEST**: Verify mouse wheel and click functionality
3. **OPTIMIZE**: Performance improvements if needed
4. **DOCUMENT**: Update architecture documentation
