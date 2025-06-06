# TUI Width Sizing Investigation

## Problem Statement

The OpenHCS TUI layout has incorrect pane width distribution. The left pane (Plate Manager) and right pane (Pipeline Editor) should split the available width according to their VSplit weights, but instead they are sizing based on their content text width.

## Observed Behavior

From the user's screenshot:
- Left pane (Plate Manager) is much narrower than expected
- Right pane (Pipeline Editor) takes up most of the width
- The panes appear to be sized to fit their empty message text content rather than respecting the VSplit weight distribution

## Root Cause Analysis

The issue is that the empty message Labels in each pane have different text lengths:

**Plate Manager empty message**: 
```
"Click 'Add' to add plates. Status: ? = added, - = initialized, o = compiled, ! = running"
```
(≈88 characters)

**Pipeline Editor empty message**:
```
"No steps available. Select a plate first."
```
(≈39 characters)

The longer Plate Manager message is driving the width calculation, causing uneven pane distribution.

## What I've Tried

### Attempt 1: Button Width Calculation
Initially misdiagnosed the problem as button width issues. Tried calculating button widths dynamically based on text length + padding. This was incorrect - the buttons are not the issue.

### Attempt 2: Label `dont_extend_width=True`
Tried setting `dont_extend_width=True` on the empty message Labels:

```python
empty_label = Label(self.config.empty_message, dont_extend_width=True)
```

**Result**: This should theoretically work according to prompt_toolkit documentation, but testing was interrupted.

### Attempt 3: Window Wrapper (Failed)
Tried wrapping the Label in a Window container:

```python
empty_label = Label(self.config.empty_message)
empty_window = Window(content=empty_label, dont_extend_width=True, wrap_lines=True)
```

**Result**: Failed with error `'Label' object has no attribute 'reset'` because Label doesn't implement the required container interface for Window content.

## What Should Work (Theory)

According to prompt_toolkit source code analysis:

1. **Label with `dont_extend_width=True`**: This should prevent the Label from requesting more width than its preferred width, allowing the VSplit weight distribution to take precedence.

2. **VSplit weight distribution**: The canonical_layout.py uses:
   ```python
   VSplit([
       plate_manager_frame,  # Should get proportional width
       pipeline_editor_frame  # Should get proportional width  
   ])
   ```

3. **Frame behavior**: Frames should respect their content's width preferences but also allow weight-based distribution when content doesn't have strong width requirements.

## What I Don't Understand

1. **Why `dont_extend_width=True` isn't working**: The prompt_toolkit documentation suggests this should solve the problem, but the behavior persists.

2. **VSplit weight calculation**: How exactly does prompt_toolkit calculate width distribution when children have different preferred widths? Is there a way to force equal distribution regardless of content?

3. **Frame width behavior**: Do Frame containers pass through their content's width preferences, or do they have their own width calculation logic?

4. **Alternative approaches**: Are there other prompt_toolkit patterns for ensuring equal width distribution in VSplit regardless of content width?

## Next Steps to Investigate

1. **Test the `dont_extend_width=True` fix properly**: Complete the testing that was interrupted.

2. **Examine VSplit source code**: Look at how prompt_toolkit's VSplit calculates width distribution when children have different preferred widths.

3. **Consider explicit width constraints**: Maybe use explicit `width` parameters or `Dimension` objects to force equal distribution.

4. **Investigate Frame behavior**: Check if Frame containers are interfering with width distribution.

5. **Alternative layout patterns**: Research other prompt_toolkit applications to see how they handle equal width distribution with variable content.

## Code Locations

- **Main layout**: `openhcs/tui/layout/canonical_layout.py`
- **List manager**: `openhcs/tui/components/list_manager.py` (line 167 - empty message creation)
- **Empty messages**: Defined in ListManagerConfig objects

## Resolution ✅

**SOLVED**: The width distribution issue has been resolved through proper architectural fixes.

### Root Cause Confirmed
The issue was that VSplit weight distribution only works when children don't have strong width preferences. The long empty message Label (88 characters) was requesting its preferred width, which overrode the weight system.

### Successful Fix
Applied explicit `width=Dimension(weight=1)` constraints at multiple levels:

1. **DynamicContainer level** (canonical_layout.py lines 283, 315):
   ```python
   dynamic_container.width = Dimension(weight=1)  # Equal weight for 50/50 split
   ```

2. **Frame level** (list_manager.py line 146):
   ```python
   return Frame(
       # ... content ...
       width=Dimension(weight=1)  # Force equal width distribution
   )
   ```

### Result
- ✅ Both panes now have equal width distribution (50/50 split)
- ✅ Content width preferences no longer override layout weights
- ✅ Layout is stable regardless of empty message text length

### Key Insight
The fix belonged in the **layout architecture**, not the component code. Setting `dont_extend_width=True` on Labels only prevents expansion beyond preferred width, but doesn't prevent them from requesting their preferred width in the first place.

### Testing Confirmed
Visual inspection of the TUI shows proper equal width distribution between Plate Manager and Pipeline Editor panes.
