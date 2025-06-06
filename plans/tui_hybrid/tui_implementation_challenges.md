# TUI Implementation Challenges and Requirements Analysis

## What You Are Asking Me to Implement

### Core Request
You want me to fix the OpenHCS TUI (Text User Interface) so that:
1. **Buttons are clickable** - All buttons in the interface should respond to mouse clicks and keyboard navigation
2. **Scrolling works** - File lists and content areas should be scrollable when content exceeds visible area
3. **Framing is correct** - The layout should display properly with clean borders and proper spacing
4. **File browser functions** - The file manager/browser component should allow navigation and file selection

### Specific Components Mentioned
1. **File Browser** (`openhcs/tui/file_browser.py`)
   - Should allow directory navigation
   - Should display files and folders with proper icons
   - Should support file/directory selection
   - Should be scrollable when content is long
   - Should have clickable buttons (Up, Refresh, Select, Cancel)

2. **Main TUI Layout** (`openhcs/tui/canonical_layout.py`)
   - 3-bar layout: Top bar | Main content | Status bar
   - Left pane: Plate Manager with buttons (add, del, edit, init, compile, run)
   - Right pane: Pipeline Editor with buttons (add, del, edit, load, save)
   - All buttons should be functional

3. **Integration Requirements**
   - File browser should integrate properly with dialogs
   - Components should work together without import/runtime errors
   - Should follow prompt_toolkit best practices

## What I Am Finding Difficult

### 1. **Environment/Import Issues**
**Problem**: Basic Python imports are hanging or failing
- Even simple `import prompt_toolkit` commands hang indefinitely
- This suggests environment corruption or dependency conflicts
- Makes it impossible to test actual functionality

**Uncertainty**: 
- Is this a virtual environment issue?
- Are there conflicting dependencies?
- Is there a circular import somewhere I'm missing?

### 2. **prompt_toolkit API Knowledge Gaps**
**Problem**: I've been hallucinating prompt_toolkit APIs
- Used non-existent `ScrollablePane` class
- Incorrect parameter names and usage patterns
- Need to cross-reference actual prompt_toolkit source

**Uncertainty**:
- What's the correct way to implement scrollable content in prompt_toolkit?
- What are the proper container types and their parameters?
- How should focus management work between components?

### 3. **Button Handler Implementation**
**Problem**: Buttons appear to render but don't respond to clicks
- Handlers are defined but may not be properly connected
- Focus management might be preventing click events
- Mouse support might not be properly configured

**Uncertainty**:
- Are the button handlers using the correct signature?
- Is the Application configured with proper mouse_support?
- How should async handlers be implemented for background tasks?

### 4. **Container and Layout Structure**
**Problem**: Layout framing appears broken
- Components may not be using proper container hierarchy
- Frame, Box, HSplit, VSplit usage might be incorrect
- Height/width specifications might be wrong

**Uncertainty**:
- What's the correct container hierarchy for a 3-pane layout?
- How should dynamic content (like file lists) be properly contained?
- What are the correct dimension specifications?

### 5. **File Browser Integration**
**Problem**: File browser has import blocking issues
- FileManager calls during `__init__` cause hanging
- Integration with dialog system is unclear
- Async/sync boundary handling is problematic

**Uncertainty**:
- How should the file browser be properly integrated into dialogs?
- What's the correct way to handle async file operations?
- How should the `__pt_container__()` method work?

## Specific Technical Questions

### 1. **Scrolling Implementation**
- What's the correct prompt_toolkit way to make content scrollable?
- Should I use `Window` with `scrollbar=True` or a different approach?
- How do I handle dynamic content that changes size?

### 2. **Button Event Handling**
- What's the correct handler signature for prompt_toolkit buttons?
- How do I properly handle async operations in button handlers?
- How do I ensure buttons receive focus and mouse events?

### 3. **Layout Architecture**
- What's the proper container hierarchy for complex layouts?
- How do I ensure proper sizing and spacing?
- How do I handle dialog overlays correctly?

### 4. **Integration Patterns**
- How should components like FileManagerBrowser integrate with the main TUI?
- What's the correct way to handle component lifecycle (init, show, hide)?
- How do I prevent import-time blocking operations?

## What I Need to Succeed

1. **Working Environment**: Confirmation that the Python environment can actually run prompt_toolkit
2. **API Reference**: Clear examples of correct prompt_toolkit usage patterns
3. **Architecture Guidance**: Proper way to structure TUI components and their interactions
4. **Testing Strategy**: How to verify that fixes actually work without hanging

## Current Status

- ✅ Fixed import blocking issues in FileManagerBrowser
- ✅ Removed non-existent ScrollablePane usage
- ✅ Simplified button creation to use standard Button widgets
- ❌ Still uncertain about proper scrolling implementation
- ❌ Still uncertain about button click handling
- ❌ Still uncertain about layout framing
- ❌ Cannot test due to environment issues

The core challenge is that I'm working blind - I can't test changes because even basic imports hang, so I'm making educated guesses about what might be wrong without being able to verify the fixes work.
