# plan_06_integration_testing.md
## Component: Integration Testing (Final Phase)

### Objective
Perform final integration testing of the hybrid TUI after static analysis validation, verify performance, and complete final cleanup.

### Plan
1. End-to-end integration testing (after static validation)
2. Performance verification and optimization
3. Final cleanup and documentation
4. Deployment preparation
5. User acceptance validation

### Findings

**Integration Testing Strategy (Final Phase Only):**

**Prerequisites:**
- Static analysis validation complete (Phase 5)
- Error density < 0.05 achieved
- Architecture purity verified
- All components pass static validation

**1. End-to-End Integration Tests:**
```python
# test_hybrid_tui_integration.py
async def test_complete_step_editing_workflow():
    """Test full workflow: select step -> edit -> save -> verify"""

async def test_pipeline_editor_integration():
    """Test pipeline editor -> step editor -> back to pipeline"""

async def test_file_operations_integration():
    """Test load/save operations with real files"""
```

**2. Performance Integration Tests:**
```python
# test_performance_integration.py
async def test_large_function_list_performance():
    """Test UI responsiveness with large function lists"""

async def test_memory_usage_patterns():
    """Test memory usage during extended editing sessions"""

async def test_file_io_performance():
    """Test load/save performance with large patterns"""
```

**3. User Workflow Tests:**
```python
# test_user_workflows.py
async def test_typical_step_creation_workflow():
    """Test creating new step from scratch"""

async def test_complex_pattern_editing_workflow():
    """Test editing complex multi-function patterns"""

async def test_error_recovery_workflows():
    """Test recovery from various error conditions"""
```

**Note: No Unit Tests During Development**
- Static analysis provides better validation than unit tests
- Integration tests only after static validation complete
- Focus on architecture purity over test coverage during development

**Performance Testing:**

**1. Component Load Times:**
- Function pattern editor initialization
- Step settings form generation
- Large function list handling

**2. Memory Usage:**
- Component lifecycle management
- State cleanup on editor close
- Memory leaks in async operations

**3. UI Responsiveness:**
- Large parameter form rendering
- Function registry loading
- File I/O operations

**Integration Test Scenarios:**

**Scenario 1: Basic Step Editing**
```python
async def test_basic_step_editing():
    # Create FunctionStep
    step = FunctionStep(func=some_function, name="Test Step")

    # Open editor
    controller = DualEditorController(ui_state, async_mgr, step)
    await controller.initialize_controller()

    # Modify step parameters
    await controller.step_settings_editor.update_parameter('name', 'Modified Step')

    # Save and verify
    await controller.save_changes()
    assert step.name == 'Modified Step'
```

**Scenario 2: Function Pattern Editing**
```python
async def test_function_pattern_editing():
    # Create step with complex pattern
    pattern = [func1, (func2, {'param': 'value'})]
    step = FunctionStep(func=pattern)

    # Open editor, switch to func view
    controller = DualEditorController(ui_state, async_mgr, step)
    await controller.initialize_controller()

    # Modify pattern
    new_pattern = {'key1': [func1, func2]}
    await controller.func_pattern_editor.update_data(new_pattern)

    # Save and verify
    await controller.save_changes()
    assert step.func == new_pattern
```

**Cleanup Checklist:**
- [ ] Remove all schema-related code
- [ ] Clean up unused imports
- [ ] Fix circular import issues
- [ ] Add comprehensive docstrings
- [ ] Remove dead code from porting
- [ ] Optimize component initialization
- [ ] Add error handling for edge cases
- [ ] Performance profiling
- [ ] Memory leak testing
- [ ] Documentation updates

**Success Criteria:**
- ✅ All component tests pass
- ✅ Integration tests pass
- ✅ Error density < 0.1
- ✅ No memory leaks
- ✅ UI responsive under load
- ✅ Complete feature parity with TUI
- ✅ Clean MVC architecture maintained

### Implementation Draft
(Only after smell loop passes)
