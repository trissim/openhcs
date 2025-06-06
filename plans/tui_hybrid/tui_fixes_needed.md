# TUI Fixes Needed - Based on Actual OpenHCS Codebase

## 🚨 CRITICAL ARCHITECTURAL ISSUES

### 1. **Function Discovery System EXISTS - Integration Missing**
**Reality**: FUNC_REGISTRY exists in `openhcs/processing/func_registry.py` with auto-discovery
**Fix Required**:
- Connect TUI to existing `get_functions_by_memory_type()` API
- Use existing `@memory_types(input_type="numpy", output_type="numpy")` decorators
- Integrate with existing `initialize_registry()` system
- Build function picker dialog using discovered functions from FUNC_REGISTRY

### 2. **Real Orchestrator Methods EXIST - Connection Missing**
**Reality**: `PipelineOrchestrator` has actual methods:
- `orchestrator.initialize()` - exists and works
- `orchestrator.compile_pipelines(pipeline_definition, well_filter)` - exists and works
- `orchestrator.execute_compiled_plate(pipeline_definition, compiled_contexts)` - exists and works
**Fix Required**:
- Connect TUI buttons to real orchestrator methods instead of fake status updates
- Handle `List[AbstractStep]` pipeline definitions properly
- Manage `compiled_contexts` dictionary from compile to execute phases
- Implement proper error handling for real orchestrator failures

### 3. **FunctionStep Integration EXISTS - TUI Missing**
**Reality**: `FunctionStep(func=...)` exists and supports the 4 sacred patterns:
- Single: `FunctionStep(func=my_function)`
- Parameterized: `FunctionStep(func=(my_function, {'param': value}))`
- Sequential: `FunctionStep(func=[func1, func2, func3])`
- Component-specific: Not directly in FunctionStep but supported by pipeline structure
**Fix Required**:
- Create Add Step dialog that builds real FunctionStep objects
- Connect to FUNC_REGISTRY for function selection
- Generate parameter forms from function signatures using `inspect.signature()`
- Build `List[AbstractStep]` pipeline definitions for orchestrator

## 🎯 UI/UX FUNCTIONAL ISSUES

### 4. **Selection System Broken**
**Problem**: No way to select plates/steps in lists
**Fix Required**:
- Implement clickable list items with visual selection feedback
- Add keyboard navigation (arrow keys) for list selection
- Update `selected_plate_index` and `selected_step_index` on user interaction
- Show selection with `>` prefix as designed

### 5. **ScrollablePane Import Fixed But Usage Incomplete**
**Problem**: ScrollablePane imported correctly but not fully utilized
**Fix Required**:
- Ensure proper `from prompt_toolkit.layout import ScrollablePane` usage
- Add proper height dimensions and scroll offsets
- Implement `cursorline=True` for visual feedback
- Test scrolling with large lists (>20 items)

### 6. **Mouse Support Enabled But Focus Issues Remain**
**Problem**: `mouse_support=True` added but focus management incomplete
**Fix Required**:
- Implement proper tab navigation between panes
- Ensure buttons are focusable and receive click events
- Add keyboard shortcuts for common actions
- Fix focus order: top bar → left pane → right pane → status

## 🔧 COMPONENT INTEGRATION ISSUES

### 7. **File Browser Integration Incomplete**
**Problem**: File browser opens but selection callback might not work properly
**Fix Required**:
- Test end-to-end file browser workflow (Add Plate)
- Ensure `on_path_selected` callback properly updates state
- Verify dialog hiding/showing works correctly
- Test with actual directory selection and state updates

### 8. **Pipeline Serialization Needs Validation**
**Problem**: Save/Load uses pickle but no validation of pipeline structure
**Fix Required**:
- Validate pipeline structure before saving
- Handle loading of incompatible pipeline versions
- Add error handling for corrupted .pipeline files
- Implement pipeline metadata (version, creation date, etc.)

### 9. **Status Symbol Logic Incomplete**
**Problem**: Status symbols update but don't reflect real TUI memory properly
**Fix Required**:
- Implement proper TUI memory tracking (what TUI has done vs. system state)
- Preserve status on errors (critical for retry capability)
- Add status persistence across TUI sessions
- Implement status validation (can't compile without init, etc.)

## 🎨 VISUAL PROGRAMMING FEATURES MISSING

### 10. **Parameter Editor Missing**
**Problem**: Edit Step/Edit Plate buttons show placeholder
**Fix Required**:
- Create parameter editor dialog that auto-generates from function signatures
- Support all Python types: int, float, str, bool, lists, dicts
- Implement default value handling and validation
- Enable real-time parameter preview

### 11. **Function Pattern Editor Missing**
**Problem**: No visual pipeline builder interface
**Fix Required**:
- Create dual Step/Func editor as described in intuition doc
- Implement visual pipeline composition (drag & drop)
- Support the 4 sacred patterns of function combination
- Enable pipeline structure visualization

### 12. **Global Settings Dialog Missing**
**Problem**: Global Settings button shows placeholder
**Fix Required**:
- Create configuration editor for global OpenHCS settings
- Support backend selection (numpy/cupy/torch)
- Implement memory management settings
- Add logging level and output directory configuration

## 🚀 DISCOVERY SYSTEM IMPLEMENTATION

### 13. **Auto-Generated UI Missing**
**Problem**: All forms are hardcoded, no signature-based generation
**Fix Required**:
- Implement function signature parsing for UI generation
- Create automatic form generation from type hints
- Support complex types (numpy arrays, custom objects)
- Enable dynamic UI updates when functions change

### 14. **Memory Type Validation Missing**
**Problem**: No GPU/CPU compatibility checking
**Fix Required**:
- Implement memory backend detection and validation
- Prevent incompatible function combinations
- Add automatic memory type conversion where possible
- Show memory requirements in function picker

### 15. **Component-Specific Processing Missing**
**Problem**: No support for channel-specific pipelines
**Fix Required**:
- Implement component-specific function assignment
- Support dictionary-based pipeline definitions
- Enable per-channel parameter customization
- Add component detection from microscope metadata

## 🔄 LIFECYCLE AND STATE MANAGEMENT

### 16. **Lazy Initialization Partially Implemented**
**Problem**: Some components still do I/O in __init__
**Fix Required**:
- Complete lazy initialization for all components
- Use `@property` decorators for expensive operations
- Implement proper component lifecycle (initialize/show/hide/cleanup)
- Prevent import-time blocking completely

### 17. **Error Handling and Recovery Incomplete**
**Problem**: Errors might corrupt TUI state
**Fix Required**:
- Implement proper error isolation
- Preserve TUI memory on failures
- Add retry mechanisms without state loss
- Show errors without changing status symbols

### 18. **Multi-Folder Batch Processing Missing**
**Problem**: No support for processing multiple experiment folders
**Fix Required**:
- Implement multi-folder selection in Add Plate
- Create independent orchestrator instances per folder
- Support shared pipeline across multiple experiments
- Add batch progress tracking

## 📊 TESTING AND VALIDATION NEEDED

### 19. **End-to-End Workflow Testing**
**Problem**: Individual buttons work but full workflow untested
**Fix Required**:
- Test complete workflow: Add Plate → Add Steps → Init → Compile → Run
- Validate state transitions and error conditions
- Test with real microscopy data and pipelines
- Verify parallel execution and progress tracking

### 20. **Performance and Scalability**
**Problem**: Unknown performance with large datasets
**Fix Required**:
- Test with 384-well plates and large image datasets
- Optimize scrolling performance with many items
- Implement progress indicators for long operations
- Add memory usage monitoring and limits

## 🎯 PRIORITY ORDER

**Phase 1 (Critical)**: Items 1, 2, 3, 4 - Core functionality
**Phase 2 (Essential)**: Items 5, 6, 7, 8, 9 - UI/UX completion  
**Phase 3 (Advanced)**: Items 10, 11, 12, 13, 14, 15 - Visual programming
**Phase 4 (Polish)**: Items 16, 17, 18, 19, 20 - Production readiness

Each item should be tackled methodically, one at a time, with proper testing before moving to the next.
