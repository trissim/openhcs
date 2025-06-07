# OpenHCS TUI Styling Guidelines

## Core Principle: Semantic vs Visual Styling

**SEMANTIC STYLING** (✅ KEEP):
- Class names that identify element purpose: `class:frame`, `class:button`, `class:dialog`
- Provides structure and accessibility
- Enables theming without forcing specific colors
- Examples: `class:frame`, `class:button.focused`, `class:error-text`

**VISUAL STYLING** (❌ REMOVE):
- Hardcoded colors, backgrounds, fonts
- Forces specific appearance regardless of user preferences
- Examples: `#ffffff bg:#000000`, `ansired bold`, `bg:ansiblack`

## Decision Framework

### ✅ ALWAYS KEEP
```python
# Semantic class names
style='class:frame'
style='class:button'
style='class:button.focused'
style='class:dialog'
style='class:error-text'
style='class:log.warning'
```

### ❌ ALWAYS REMOVE
```python
# Hardcoded colors in Style.from_dict()
Style.from_dict({
    'text-area': '#ffffff bg:#000000',  # ❌ REMOVE
    'frame': '#ffffff bg:#000000',      # ❌ REMOVE
})

# Hardcoded colors in component definitions
style='ansired bold'                    # ❌ REMOVE
style='ansigreen bg:ansiblack'         # ❌ REMOVE
```

### 🤔 CASE-BY-CASE (Functional Colors)
```python
# Error dialogs - colors serve functional purpose
if 'Error' in token_str:
    style = 'ansired bold'              # 🤔 FUNCTIONAL - probably keep

# Log levels - colors distinguish severity
LogLevel.ERROR: ("class:log.error", "ansired")  # 🤔 FUNCTIONAL - probably keep
```

## Implementation Rules

### 1. Global Overrides (openhcs/tui/__init__.py)
- ✅ Keep semantic class defaults: `style='class:frame'`
- ❌ Never set hardcoded colors in overrides
- ✅ Focus on layout/behavior, not appearance

### 2. Application Setup (canonical_layout.py)
- ❌ Never create Style.from_dict() with hardcoded colors
- ✅ Use Application() without custom style parameter
- ✅ Let prompt_toolkit use terminal defaults

### 3. Component Definitions
- ✅ Use semantic classes: `style='class:button.focused'`
- ❌ Never hardcode colors in component style parameters
- ✅ Let global overrides handle semantic styling

### 4. Functional Color Exceptions
- ✅ Error dialogs can use colors for clarity
- ✅ Log levels can use colors for severity distinction
- ✅ Syntax highlighting can use colors for readability
- ❌ But prefer semantic classes even here when possible

## Validation Checklist

Before committing styling changes, check:

1. **Are you removing semantic classes?** ❌ DON'T
   - `class:frame` → `''` is WRONG
   - `class:button` → `''` is WRONG

2. **Are you adding hardcoded colors?** ❌ DON'T
   - `style='#ffffff bg:#000000'` is WRONG
   - `Style.from_dict({'frame': '#fff'})` is WRONG

3. **Are you keeping functional colors?** 🤔 EVALUATE
   - Error dialogs: probably keep
   - Log levels: probably keep
   - General UI: probably remove

4. **Does it respect user terminal theme?** ✅ GOAL
   - User has dark theme → should work
   - User has light theme → should work
   - User has custom colors → should work

## Examples of Correct Changes

### ✅ CORRECT: Remove hardcoded colors, keep semantic classes
```python
# Before
Style.from_dict({
    'frame': '#ffffff bg:#000000',      # ❌ Hardcoded
})

# After  
# No Style.from_dict() at all              # ✅ Use terminal defaults

# Keep semantic classes
style='class:frame'                     # ✅ Semantic
```

### ❌ INCORRECT: Remove semantic classes
```python
# Before
style='class:frame'                     # ✅ Semantic

# After (WRONG)
style=''                                # ❌ Lost semantic meaning
```

## Testing Strategy

1. **Test with different terminal themes**
   - Dark theme
   - Light theme  
   - High contrast theme
   - Custom color schemes

2. **Verify semantic structure**
   - Screen readers can identify elements
   - Future theming hooks are preserved
   - prompt_toolkit conventions followed

3. **Check functional colors**
   - Error messages still distinguishable
   - Log levels still clear
   - Syntax highlighting still readable

## Future Prevention

1. **Code review checklist** - Include styling guidelines
2. **Automated checks** - Scan for hardcoded color patterns
3. **Documentation** - This file as reference
4. **Testing** - Multiple terminal themes in CI

## Summary

**The goal is terminal-native styling that respects user preferences while maintaining semantic structure and functional clarity.**
