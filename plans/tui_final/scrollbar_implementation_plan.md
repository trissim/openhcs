# Scrollbar Click Implementation Plan

## Current Issues Analysis

### 1. Key Bindings Not Working
**Problem**: Mouse wheel `<scroll-up>` and `<scroll-down>` bindings aren't being triggered.

**Root Cause Investigation Needed**:
- Are key bindings attached to the right component?
- Is the focus on the correct element?
- Are mouse wheel events being converted to key events properly?

### 2. No Native Scrollbar Click Support
**Confirmed**: ScrollablePane and ScrollbarMargin only draw visual scrollbars - no mouse handlers.

**Need to Implement**:
- Scrollbar area detection
- Click-to-position mapping
- Arrow button click handling

## Architecture Requirements

### 1. Clean Overlay System
```
ScrollablePane
├── Content Area (existing FormattedTextControl)
└── Scrollbar Overlay (new mouse handler system)
    ├── Up Arrow Button
    ├── Scrollbar Track
    │   ├── Above Thumb (page up)
    │   ├── Thumb (drag - future)
    │   └── Below Thumb (page down)
    └── Down Arrow Button
```

### 2. Coordinate Mapping System
```python
class ScrollbarGeometry:
    """Calculate scrollbar positions and map clicks to actions."""
    
    def __init__(self, scrollable_pane, write_position):
        self.scrollable_pane = scrollable_pane
        self.write_position = write_position
        self.calculate_geometry()
    
    def calculate_geometry(self):
        """Calculate all scrollbar positions."""
        # Up arrow: (x, y) = (write_pos.width-1, write_pos.y)
        # Down arrow: (x, y) = (write_pos.width-1, write_pos.y + height - 1)
        # Track area: between arrows
        # Thumb position: based on scroll percentage
    
    def handle_click(self, mouse_event) -> bool:
        """Map click position to scroll action."""
        x, y = mouse_event.position.x, mouse_event.position.y
        
        if self.is_up_arrow(x, y):
            return self.scroll_up()
        elif self.is_down_arrow(x, y):
            return self.scroll_down()
        elif self.is_track_above_thumb(x, y):
            return self.page_up()
        elif self.is_track_below_thumb(x, y):
            return self.page_down()
        
        return False
```

### 3. Integration Points
1. **Mouse Handler Injection**: Add scrollbar mouse handler to ScrollablePane
2. **Coordinate Translation**: Map screen coordinates to scrollbar actions
3. **Scroll Action Execution**: Update focus/scroll position cleanly

## Implementation Strategy

### Phase 1: Debug Key Bindings
1. **Investigate why mouse wheel bindings don't work**
   - Check key binding attachment
   - Test with simple key bindings first
   - Verify focus and event routing

### Phase 2: Scrollbar Geometry System
1. **Create ScrollbarGeometry class**
   - Calculate arrow positions
   - Calculate track and thumb positions
   - Map coordinates to actions

### Phase 3: Mouse Handler Integration
1. **Add scrollbar mouse handler to ScrollablePane**
   - Overlay on existing mouse handlers
   - Don't break content mouse handling
   - Clean coordinate mapping

### Phase 4: Testing and Refinement
1. **Test all scrollbar interactions**
   - Arrow clicks
   - Track clicks
   - Mouse wheel (if fixed)
   - Keyboard navigation

## Questions for Investigation

### 1. Key Binding Attachment
- Where should key bindings be attached for mouse wheel to work?
- Is the FormattedTextControl receiving focus?
- Are mouse wheel events being generated?

### 2. ScrollablePane Mouse Handler Access
- How to add mouse handlers to ScrollablePane without breaking existing ones?
- Can we override the `write_to_screen` method cleanly?
- Should we subclass ScrollablePane or use composition?

### 3. Coordinate System
- How does ScrollablePane's coordinate translation affect mouse positions?
- What's the exact geometry of the scrollbar relative to the content?
- How to handle different terminal sizes?

## Next Steps

1. **IMMEDIATE**: Debug why key bindings aren't working
2. **PLAN**: Design clean scrollbar geometry system
3. **IMPLEMENT**: Scrollbar click handling
4. **TEST**: All interaction modes

## Architecture Principles

1. **Don't Break Existing**: Content mouse handling must continue working
2. **Clean Separation**: Scrollbar logic separate from content logic
3. **Accurate Mapping**: Precise coordinate-to-action mapping
4. **Performance**: Minimal overhead for mouse event processing
