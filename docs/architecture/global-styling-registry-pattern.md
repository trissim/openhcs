# Global Styling with Exception Registry Pattern

## Problem Statement

OpenHCS TUI uses global class overrides to apply consistent styling across all prompt_toolkit components. While this successfully eliminates repetition (e.g., changing all buttons from `<text>` to `[text]`), it creates conflicts with components that have specific behavioral requirements.

**Specific Issue**: ScrollablePane is invisible because our global override automatically adds `height=Dimension(weight=1)`, but ScrollablePane requires `height=None` to scale based on its content (documented prompt_toolkit requirement).

## Architectural Solution: Exception Registry Pattern

Instead of abandoning global styling (which would require repetitive kwargs everywhere) or creating workarounds in every usage, we implement an **exception registry** that documents and handles special cases.

### Core Principle

> Most UI components benefit from consistent auto-styling, but some have specific requirements that must be respected. The registry makes these exceptions explicit and maintainable.

## Implementation

### 1. Create Exception Registry

```python
# In openhcs/tui/__init__.py

# Registry of components with special styling requirements
UI_STYLING_EXCEPTIONS = {
    'ScrollablePane': {
        'skip_auto_style': ['height'],
        'reason': 'ScrollablePane must calculate height from content - see prompt_toolkit docs',
        'documented_behavior': 'Requires height=None to scale naturally'
    },
    # Add other exceptions as discovered
}
```

### 2. Update Global Overrides to Check Registry

```python
class _StyledScrollablePane(OriginalScrollablePane):
    """ScrollablePane with OpenHCS styling that respects behavioral requirements."""
    
    def __init__(self, content, height=None, show_scrollbar=True, 
                 display_arrows=True, **kwargs):
        # Check if this parameter should be auto-styled
        exceptions = UI_STYLING_EXCEPTIONS.get('ScrollablePane', {})
        skip_params = exceptions.get('skip_auto_style', [])
        
        # Only apply auto-styling if parameter not in skip list
        if height is None and 'height' not in skip_params:
            # This block will NOT execute for ScrollablePane
            height = Dimension(weight=1)
        
        # Pass through original height=None for ScrollablePane
        super().__init__(
            content=content,
            height=height,  # Will be None, allowing natural scaling
            show_scrollbar=show_scrollbar,
            display_arrows=display_arrows,
            **kwargs
        )
```

### 3. Fix for Frame (Optional Enhancement)

```python
class _StyledFrame(OriginalFrame):
    """Frame with intelligent auto-styling."""
    
    def __init__(self, body, title=None, style='', width=None, height=None, **kwargs):
        # Check for special child components
        child_type = type(body).__name__
        child_exceptions = UI_STYLING_EXCEPTIONS.get(child_type, {})
        
        # Apply normal auto-styling for frames
        if width is None and self._should_auto_manage_width(title, style):
            width = Dimension(weight=1)
            
        if height is None and self._should_auto_manage_height(title, style):
            # Check if child has special requirements
            if 'height' not in child_exceptions.get('skip_auto_style', []):
                height = Dimension(weight=1)
        
        super().__init__(body, title, style, width, height, **kwargs)
```

## Immediate Fix for ScrollablePane Visibility

With the registry pattern implemented, the fix is simple:

```python
# In FunctionListManager._build_function_list
def _build_function_list(self):
    function_items = self._build_function_items()
    
    if not function_items:
        function_items = [Label("No functions defined")]
    
    # ScrollablePane will now correctly receive height=None
    scrollable_content = ScrollablePane(
        HSplit(function_items, padding=0)
        # No height parameter - will get None from improved override
    )
    
    # Frame can have explicit height
    return Frame(
        scrollable_content,
        title="Function Pattern Editor",
        height=Dimension(min=5, max=20, preferred=10)
    )
```

## Benefits of This Pattern

1. **DRY Principle Maintained**: Global styling still applies to 95% of components
2. **Explicit Exceptions**: Special cases are documented with reasons
3. **Framework Contracts Respected**: Components work as documented
4. **Easy to Extend**: Just add to registry when discovering new exceptions
5. **Self-Documenting**: Registry explains WHY each exception exists
6. **No Workarounds Needed**: Components work correctly without special handling at usage sites

## When to Add to Registry

Add a component to the exception registry when:

1. **Global styling breaks documented behavior** (like ScrollablePane)
2. **Component has specific requirements** incompatible with defaults
3. **Framework documentation explicitly states requirements** that conflict with auto-styling

Do NOT add to registry for:
- Style preferences (use normal parameters)
- One-off customizations (pass parameters explicitly)
- Components that work fine with auto-styling

## Testing the Fix

After implementing the registry pattern:

1. ScrollablePane should become visible without extra Frame wrapping
2. Existing styled components should continue working
3. No changes needed at usage sites
4. Future exceptions can be added to registry without modifying usage code

## Architectural Principle

> "Global defaults with explicit exceptions is better than either extreme (all manual or all automatic). It acknowledges that UI frameworks have both patterns and special cases."

This pattern provides the benefits of DRY styling while respecting the behavioral contracts of sophisticated components like ScrollablePane.

## Related: Semantic vs Visual Styling

**IMPORTANT**: This registry pattern handles behavioral styling (layout, dimensions). For visual styling (colors, fonts), see `docs/architecture/styling-guidelines.md`.

**Key distinction:**
- **Behavioral styling** (this pattern): `height=Dimension(weight=1)`, `width=Dimension(min=5)`
- **Visual styling** (guidelines doc): `style='class:frame'` (✅ semantic) vs `style='#fff bg:#000'` (❌ hardcoded)

The registry pattern should NEVER handle visual styling - only behavioral properties that affect layout and component contracts.