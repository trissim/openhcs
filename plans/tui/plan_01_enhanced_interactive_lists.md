# plan_01_enhanced_interactive_lists.md
## Component: Enhanced Interactive List Items with Multi-Selection and Scrolling

### Objective
Enhance the interactive list items in list_manager.py to support:
1. Inline move buttons (^/v) on the same line as each item
2. Multi-selection support like file browser (checkboxes, space to toggle)
3. Scrollable lists using ScrollablePane
4. Bulk operations on multiple selected items
5. Reuse file browser patterns without code duplication

### Plan
1. **Extract Common Patterns**: Create shared components for multi-selection and scrolling
2. **Enhance InteractiveListItem**: Add inline move buttons and selection checkbox
3. **Update ListView**: Add ScrollablePane and FormattedTextControl approach
4. **Add Multi-Selection Model**: Track selected items in ListModel
5. **Update PlateManager**: Support bulk operations on selected orchestrators

### Findings

#### File Browser Multi-Selection Pattern
- Uses `selected_paths: set[Path]` for tracking selections
- Checkbox UI: `[x]` for selected, `[ ]` for unselected  
- Space key toggles selection
- Mouse click on checkbox area (x_pos <= 4) toggles selection
- `allow_multiple` flag controls behavior

#### File Browser Scrolling Pattern
- Uses `ScrollablePane` wrapper around `Window(FormattedTextControl)`
- `FormattedTextControl` with `text=self._get_item_list_text` function
- Mouse handlers attached to FormattedTextControl
- `_ensure_focused_visible()` sets cursor position for auto-scroll

#### Current InteractiveListItem Architecture
- Individual Container per item with Box styling
- Move buttons (^/v) in separate VSplit
- Mouse handler on main text area for selection
- Compatible interface with list_manager.py

#### Key Architectural Insights
1. **Two Approaches**: File browser uses single FormattedTextControl for all items, InteractiveListItem uses individual Containers
2. **Scrolling**: ScrollablePane works with FormattedTextControl, not with HSplit of individual items
3. **Multi-Selection**: Needs model-level tracking, not just UI state
4. **SOLID Compliance**: Can extract common selection/scrolling logic into mixins or composition

### Implementation Strategy

#### Option A: FormattedTextControl Approach (File Browser Style)
- Replace HSplit of InteractiveListItems with single FormattedTextControl
- Generate text representation with checkboxes and move buttons
- Handle mouse clicks by position calculation
- Pros: Native scrolling, proven pattern
- Cons: Loses individual item styling, complex position logic

#### Option B: Enhanced Container Approach (Current Style)
- Keep individual InteractiveListItem containers
- Add ScrollablePane wrapper around HSplit
- Add selection state to each item
- Pros: Maintains current architecture, cleaner item logic
- Cons: May have scrolling limitations

#### Option C: Hybrid Approach
- Extract common selection/scrolling logic into shared components
- Create SelectableScrollableList component
- Use composition to combine with existing InteractiveListItem
- Pros: Reusable, SOLID compliant, maintains flexibility
- Cons: More complex architecture

### Recommended Approach: Option C (Hybrid)

1. **Create SelectionMixin**: Handle multi-selection state and logic
2. **Create ScrollableListContainer**: Wrapper for ScrollablePane functionality  
3. **Enhance InteractiveListItem**: Add checkbox and inline move buttons
4. **Update ListView**: Use new scrollable container
5. **Update ListModel**: Add selection tracking

### Next Steps
1. Analyze current ListView scrolling limitations
2. Design SelectionMixin interface
3. Create ScrollableListContainer component
4. Enhance InteractiveListItem with selection support
5. Update ListModel for multi-selection
6. Test with PlateManager bulk operations

### Questions for User
1. Do you prefer the FormattedTextControl approach (like file browser) or enhanced Container approach?
2. Should move buttons be always visible or only on hover/selection?
3. What bulk operations should be supported? (edit config, delete, init, compile, run)
4. Should selection state persist across UI updates?
