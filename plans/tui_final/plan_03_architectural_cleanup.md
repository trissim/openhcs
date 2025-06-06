# plan_03_architectural_cleanup.md
## Component: File Browser Architectural Cleanup

### Objective
Clean up architectural rot in the file browser system to make it retard-proof and maintainable. Remove all defensive programming, debug pollution, and architectural inconsistencies identified in the mathematical analysis.

### Plan
1. **Remove Debug Pollution**
   - Eliminate all debug logging from file_browser.py
   - Remove logger imports and debug statements
   - Clean up production code

2. **Implement Mathematical Fix**
   - Replace InteractiveListItem with direct prompt_toolkit containers
   - Use proven FramedButton pattern for mouse handling
   - Ensure mathematical invariants are maintained

3. **Extract Display Logic**
   - Create `_build_display_text()` helper method
   - Separate display formatting from container creation
   - Make the code more readable and maintainable

4. **Architectural Constraints**
   - Ensure no duck typing in container hierarchy
   - Use only direct prompt_toolkit containers
   - Make future modifications impossible to break

### Findings

#### Architectural Rot Inventory
1. **Debug logging pollution** (lines 276, 280, 285, 290, 293 in file_browser.py)
2. **InteractiveListItem duck typing** breaking mouse event routing
3. **Mixed concerns** in `_build_item_list()` method
4. **Lambda closure issues** in display_text_func

#### Mathematical Correctness Requirements
- State invariant: `∀i: item_style[i] ⟺ listing[i].path ∈ selected_paths`
- Mouse routing: `HSplit → Window → FormattedTextControl → mouse_handler`
- No intermediate duck-typed objects in container hierarchy

### Implementation Draft

#### Step 1: Clean Debug Pollution
Remove all debug logging statements and logger references from `_on_activate()` method.

#### Step 2: Extract Display Logic
```python
def _build_display_text(self, index: int, item: FileItem) -> str:
    """Build display text for file item - pure function."""
    max_name_width = min(60, max(20, max(len(i.name) for i in self.listing) + 6))
    
    # Build display text
    prefix = ""
    if self.allow_multiple:
        is_selected = item.path in self.selected_paths
        prefix = "[x] " if is_selected else "[ ] "
    
    icon = "📁" if item.is_dir else "📄"
    name_part = f"{prefix}{icon} {item.name}"
    
    display = f"{name_part:<{max_name_width}}"
    if not item.is_dir:
        display += f"{item.display_size:>10}  "
    else:
        display += f"{'':>10}  "
    display += item.display_mtime
    
    return display
```

#### Step 3: Implement Mathematical Fix
Replace entire `_build_item_list()` method with mathematically correct implementation using direct prompt_toolkit containers.

#### Step 4: Remove InteractiveListItem Import
Remove `from openhcs.tui.components.interactive_list_item import InteractiveListItem` since it's no longer needed for file browser.

#### Architectural Guarantees
1. **No duck typing** - Only prompt_toolkit containers in hierarchy
2. **Mathematical correctness** - State invariants maintained
3. **Proven patterns** - Uses same pattern as working FramedButton
4. **Retard-proof** - Future modifications cannot break mouse routing
5. **Clean separation** - Display logic separated from container creation
