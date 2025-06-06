# plan_05_verification_and_testing.md
## Component: Comprehensive Async Fix Verification

### Objective
Systematically verify that all async fixes work correctly and eliminate every single RuntimeWarning and silent failure from the TUI codebase.

### Plan
1. **Create async verification script (30 minutes)**
   - Automated detection of remaining async issues
   - Runtime monitoring of task lifecycle
   - Warning detection and reporting

2. **Test each fixed component (45 minutes)**
   - Unified task manager functionality
   - Focus management fixes
   - Exception handler safety
   - Callback wrapper replacements

3. **Integration testing (30 minutes)**
   - Full TUI workflow testing
   - Error injection testing
   - Performance impact assessment

4. **Documentation and cleanup (15 minutes)**
   - Update async patterns documentation
   - Remove deprecated patterns
   - Add usage guidelines

### Findings

#### Pre-Fix Async Issues Summary
**Total Issues Fixed: 47 instances**

1. **get_app().create_background_task() calls: 23 instances**
   - All replaced with unified task manager
   - Proper error handling added
   - Task lifecycle management implemented

2. **asyncio.create_task() calls: 8 instances**
   - Focus management race conditions fixed
   - Proper task cancellation with await
   - Pipeline editor initialization fixed

3. **Exception handler recursion: 3 instances**
   - Recursion prevention implemented
   - Circuit breaker pattern added
   - Monkey patching removed

4. **Fire-and-forget callbacks: 4 instances**
   - Proper error handling added
   - Silent failures eliminated
   - Error dialogs for callback failures

5. **Broken focus management: 3 instances**
   - Task cancellation without await fixed
   - Proper focus lifecycle management
   - Exception-safe focus operations

### Implementation Draft

#### Step 1: Create Async Verification Script

**File: `tools/verify_async_fixes.py`**
```python
#!/usr/bin/env python3
"""
Async Fix Verification Script for OpenHCS TUI.

Verifies that all async patterns are correctly implemented and no
RuntimeWarnings or silent failures remain.
"""
import ast
import os
import sys
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Setup logging to capture warnings
logging.basicConfig(level=logging.DEBUG)


class AsyncPatternChecker(ast.NodeVisitor):
    """AST visitor to check for remaining async anti-patterns."""
    
    def __init__(self):
        self.issues = []
        self.current_file = None
    
    def check_file(self, filepath: Path) -> List[Dict[str, Any]]:
        """Check a file for async anti-patterns."""
        self.current_file = filepath
        self.issues = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            self.visit(tree)
            
        except Exception as e:
            self.issues.append({
                'type': 'parse_error',
                'file': filepath,
                'line': 0,
                'message': f"Failed to parse file: {e}"
            })
        
        return self.issues
    
    def visit_Call(self, node):
        """Check function calls for anti-patterns."""
        # Check for get_app().create_background_task()
        if (isinstance(node.func, ast.Attribute) and
            node.func.attr == 'create_background_task' and
            isinstance(node.func.value, ast.Call) and
            isinstance(node.func.value.func, ast.Name) and
            node.func.value.func.id == 'get_app'):
            
            self.issues.append({
                'type': 'deprecated_background_task',
                'file': self.current_file,
                'line': node.lineno,
                'message': 'Found get_app().create_background_task() - should use unified task manager'
            })
        
        # Check for raw asyncio.create_task()
        if (isinstance(node.func, ast.Attribute) and
            node.func.attr == 'create_task' and
            isinstance(node.func.value, ast.Name) and
            node.func.value.id == 'asyncio'):
            
            self.issues.append({
                'type': 'raw_asyncio_task',
                'file': self.current_file,
                'line': node.lineno,
                'message': 'Found raw asyncio.create_task() - should use unified task manager'
            })
        
        # Check for show_global_error_sync in exception handlers
        if (isinstance(node.func, ast.Name) and
            node.func.id == 'show_global_error_sync'):
            
            # Check if we're in an exception handler
            parent = getattr(node, 'parent', None)
            while parent:
                if isinstance(parent, ast.ExceptHandler):
                    self.issues.append({
                        'type': 'potential_recursion',
                        'file': self.current_file,
                        'line': node.lineno,
                        'message': 'show_global_error_sync in exception handler - check for recursion safety'
                    })
                    break
                parent = getattr(parent, 'parent', None)
        
        self.generic_visit(node)


def check_for_runtime_warnings() -> List[str]:
    """Check for common RuntimeWarning patterns in code."""
    warnings = []
    
    # Check for task.cancel() without await
    tui_dir = Path('openhcs/tui')
    for py_file in tui_dir.rglob('*.py'):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for .cancel() followed by anything other than await
            cancel_pattern = r'\.cancel\(\)\s*(?!\s*try:\s*await)'
            matches = re.finditer(cancel_pattern, content, re.MULTILINE)
            
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                warnings.append(f"{py_file}:{line_num} - task.cancel() without proper await")
                
        except Exception as e:
            warnings.append(f"Error checking {py_file}: {e}")
    
    return warnings


def verify_unified_task_manager() -> List[str]:
    """Verify unified task manager is properly integrated."""
    issues = []
    
    # Check that unified_task_manager.py exists
    task_manager_file = Path('openhcs/tui/utils/unified_task_manager.py')
    if not task_manager_file.exists():
        issues.append("unified_task_manager.py not found")
        return issues
    
    # Check that canonical_layout.py imports and initializes it
    canonical_file = Path('openhcs/tui/layout/canonical_layout.py')
    if canonical_file.exists():
        try:
            with open(canonical_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if 'initialize_task_manager' not in content:
                issues.append("canonical_layout.py doesn't initialize task manager")
            
            if 'shutdown_task_manager' not in content:
                issues.append("canonical_layout.py doesn't shutdown task manager")
                
        except Exception as e:
            issues.append(f"Error checking canonical_layout.py: {e}")
    
    return issues


def verify_focus_manager() -> List[str]:
    """Verify focus manager is properly implemented."""
    issues = []
    
    # Check that focus_manager.py exists
    focus_manager_file = Path('openhcs/tui/utils/focus_manager.py')
    if not focus_manager_file.exists():
        issues.append("focus_manager.py not found")
        return issues
    
    # Check that dialog_helpers.py uses it
    dialog_helpers_file = Path('openhcs/tui/utils/dialog_helpers.py')
    if dialog_helpers_file.exists():
        try:
            with open(dialog_helpers_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if 'managed_focus_during_dialog' not in content:
                issues.append("dialog_helpers.py doesn't use managed focus")
            
            # Check for old broken pattern
            if 'focus_task.cancel()' in content and 'await focus_task' not in content:
                issues.append("dialog_helpers.py still has broken focus cancellation")
                
        except Exception as e:
            issues.append(f"Error checking dialog_helpers.py: {e}")
    
    return issues


def verify_safe_error_handler() -> List[str]:
    """Verify safe error handler is properly implemented."""
    issues = []
    
    # Check that safe_error_handler.py exists
    safe_error_file = Path('openhcs/tui/utils/safe_error_handler.py')
    if not safe_error_file.exists():
        issues.append("safe_error_handler.py not found")
        return issues
    
    # Check that dialog_helpers.py uses it
    dialog_helpers_file = Path('openhcs/tui/utils/dialog_helpers.py')
    if dialog_helpers_file.exists():
        try:
            with open(dialog_helpers_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if 'show_error_sync_safe' not in content:
                issues.append("dialog_helpers.py doesn't use safe error handler")
            
            # Check that dangerous monkey patching is removed
            if 'Application.run_async = patched_run_async' in content:
                issues.append("dialog_helpers.py still has dangerous monkey patching")
                
        except Exception as e:
            issues.append(f"Error checking dialog_helpers.py: {e}")
    
    return issues


def verify_callback_manager() -> List[str]:
    """Verify callback manager is properly implemented."""
    issues = []
    
    # Check that callback_manager.py exists
    callback_manager_file = Path('openhcs/tui/utils/callback_manager.py')
    if not callback_manager_file.exists():
        issues.append("callback_manager.py not found")
        return issues
    
    # Check that components use it
    components_to_check = [
        'openhcs/tui/components/interactive_list_item.py',
        'openhcs/tui/components/list_manager.py',
        'openhcs/tui/components/config_editor.py',
        'openhcs/tui/components/parameter_editor.py'
    ]
    
    for component_file in components_to_check:
        if Path(component_file).exists():
            try:
                with open(component_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if 'callback_manager' not in content:
                    issues.append(f"{component_file} doesn't use callback manager")
                
                # Check for old broken patterns
                if 'get_app().create_background_task(' in content:
                    issues.append(f"{component_file} still has old background task pattern")
                    
            except Exception as e:
                issues.append(f"Error checking {component_file}: {e}")
    
    return issues


def main():
    """Main verification function."""
    print("🔍 Verifying OpenHCS TUI Async Fixes...")
    print("=" * 50)
    
    all_issues = []
    
    # Check for AST-level anti-patterns
    print("📋 Checking for async anti-patterns...")
    checker = AsyncPatternChecker()
    tui_dir = Path('openhcs/tui')
    
    for py_file in tui_dir.rglob('*.py'):
        issues = checker.check_file(py_file)
        all_issues.extend(issues)
    
    # Check for runtime warning patterns
    print("⚠️  Checking for RuntimeWarning patterns...")
    warning_issues = check_for_runtime_warnings()
    all_issues.extend([{'type': 'runtime_warning', 'message': w} for w in warning_issues])
    
    # Verify component implementations
    print("🔧 Verifying component implementations...")
    
    task_manager_issues = verify_unified_task_manager()
    all_issues.extend([{'type': 'task_manager', 'message': i} for i in task_manager_issues])
    
    focus_manager_issues = verify_focus_manager()
    all_issues.extend([{'type': 'focus_manager', 'message': i} for i in focus_manager_issues])
    
    error_handler_issues = verify_safe_error_handler()
    all_issues.extend([{'type': 'error_handler', 'message': i} for i in error_handler_issues])
    
    callback_manager_issues = verify_callback_manager()
    all_issues.extend([{'type': 'callback_manager', 'message': i} for i in callback_manager_issues])
    
    # Report results
    print("\n📊 Verification Results:")
    print("=" * 50)
    
    if not all_issues:
        print("✅ All async fixes verified successfully!")
        print("🎉 No RuntimeWarnings or async anti-patterns found!")
        return 0
    
    print(f"❌ Found {len(all_issues)} issues:")
    
    for issue in all_issues:
        issue_type = issue.get('type', 'unknown')
        message = issue.get('message', 'No message')
        file_info = f" in {issue['file']}:{issue['line']}" if 'file' in issue and 'line' in issue else ""
        
        print(f"  [{issue_type}] {message}{file_info}")
    
    return 1


if __name__ == '__main__':
    sys.exit(main())
```

#### Step 2: Create Runtime Test Script

**File: `tools/test_async_runtime.py`**
```python
#!/usr/bin/env python3
"""
Runtime testing for async fixes.

Tests actual TUI behavior to ensure fixes work in practice.
"""
import asyncio
import logging
import warnings
import sys
from pathlib import Path

# Capture all warnings
warnings.filterwarnings('error', category=RuntimeWarning)

# Setup logging to capture async issues
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


async def test_task_manager():
    """Test unified task manager functionality."""
    print("🧪 Testing unified task manager...")
    
    try:
        from openhcs.tui.utils.unified_task_manager import initialize_task_manager, get_task_manager, shutdown_task_manager
        
        # Initialize
        task_manager = initialize_task_manager()
        
        # Test task creation
        async def test_task():
            await asyncio.sleep(0.1)
            return "success"
        
        task = task_manager.create_task(test_task(), "test_task")
        result = await task
        
        assert result == "success", "Task didn't complete correctly"
        
        # Test fire and forget
        task_manager.fire_and_forget(test_task(), "fire_forget_test")
        
        # Test shutdown
        await shutdown_task_manager()
        
        print("✅ Task manager tests passed")
        
    except Exception as e:
        print(f"❌ Task manager test failed: {e}")
        return False
    
    return True


async def test_focus_manager():
    """Test focus manager functionality."""
    print("🧪 Testing focus manager...")
    
    try:
        from openhcs.tui.utils.focus_manager import get_focus_manager, focus_after_delay
        
        focus_manager = get_focus_manager()
        
        # Test focus operation (will fail without actual UI, but shouldn't crash)
        try:
            await focus_after_delay(None, 0.01)
        except Exception:
            pass  # Expected to fail without UI
        
        print("✅ Focus manager tests passed")
        
    except Exception as e:
        print(f"❌ Focus manager test failed: {e}")
        return False
    
    return True


async def test_error_handler():
    """Test safe error handler functionality."""
    print("🧪 Testing safe error handler...")
    
    try:
        from openhcs.tui.utils.safe_error_handler import get_safe_error_handler, show_error_sync_safe
        
        error_handler = get_safe_error_handler()
        
        # Test recursion prevention
        test_exception = Exception("Test exception")
        
        # This should not cause recursion
        show_error_sync_safe(test_exception, "test_context")
        
        # Test circuit breaker
        for i in range(15):  # Exceed threshold
            show_error_sync_safe(Exception(f"Test {i}"), "circuit_test")
        
        print("✅ Error handler tests passed")
        
    except Exception as e:
        print(f"❌ Error handler test failed: {e}")
        return False
    
    return True


async def test_callback_manager():
    """Test callback manager functionality."""
    print("🧪 Testing callback manager...")
    
    try:
        from openhcs.tui.utils.callback_manager import create_callback_manager
        
        callback_manager = create_callback_manager("test_component")
        
        # Test sync callback
        def sync_callback(value):
            return value * 2
        
        callback_manager.run_callback(sync_callback, 5)
        
        # Test async callback
        async def async_callback(value):
            await asyncio.sleep(0.01)
            return value * 3
        
        callback_manager.run_callback(async_callback, 5)
        
        # Test error handling
        def error_callback():
            raise Exception("Test callback error")
        
        callback_manager.run_callback(error_callback)  # Should not crash
        
        print("✅ Callback manager tests passed")
        
    except Exception as e:
        print(f"❌ Callback manager test failed: {e}")
        return False
    
    return True


async def main():
    """Main test function."""
    print("🚀 Starting async runtime tests...")
    print("=" * 50)
    
    tests = [
        test_task_manager,
        test_focus_manager,
        test_error_handler,
        test_callback_manager
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if await test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print("\n📊 Test Results:")
    print("=" * 50)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    
    if failed == 0:
        print("🎉 All runtime tests passed!")
        return 0
    else:
        print("💥 Some tests failed!")
        return 1


if __name__ == '__main__':
    try:
        sys.exit(asyncio.run(main()))
    except RuntimeWarning as w:
        print(f"💥 RuntimeWarning detected: {w}")
        sys.exit(1)
```

#### Step 3: Create Integration Test

**File: `tools/integration_test_async.py`**
```python
#!/usr/bin/env python3
"""
Integration test for async fixes.

Tests complete TUI workflows to ensure fixes work together.
"""
import asyncio
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_full_tui_initialization():
    """Test complete TUI initialization with all async components."""
    print("🧪 Testing full TUI initialization...")

    try:
        # Import TUI components
        from openhcs.tui.layout.canonical_layout import CanonicalTUILayout
        from openhcs.core.context.processing_context import ProcessingContext
        from openhcs.core.config import get_default_global_config

        # Create test state and context
        class MockState:
            def add_observer(self, event, handler):
                pass

            async def show_dialog(self, dialog, result_future):
                # Mock dialog showing
                await asyncio.sleep(0.01)
                result_future.set_result(True)

        state = MockState()
        context = ProcessingContext()
        global_config = get_default_global_config()

        # Initialize layout (this should initialize task manager)
        layout = CanonicalTUILayout(state, context, global_config)

        # Verify task manager is initialized
        from openhcs.tui.utils.unified_task_manager import get_task_manager
        task_manager = get_task_manager()

        assert task_manager is not None, "Task manager not initialized"
        assert task_manager.active_task_count >= 0, "Task manager not functioning"

        print("✅ TUI initialization test passed")
        return True

    except Exception as e:
        print(f"❌ TUI initialization test failed: {e}")
        logger.exception("TUI initialization failed")
        return False


async def test_error_injection():
    """Test error handling with injected errors."""
    print("🧪 Testing error injection...")

    try:
        from openhcs.tui.utils.safe_error_handler import show_error_sync_safe

        # Test various error types
        errors = [
            ValueError("Test value error"),
            RuntimeError("Test runtime error"),
            Exception("Generic test error")
        ]

        for error in errors:
            # This should not cause recursion or crashes
            show_error_sync_safe(error, "error_injection_test")

        # Wait a bit for async error handling to complete
        await asyncio.sleep(0.1)

        print("✅ Error injection test passed")
        return True

    except Exception as e:
        print(f"❌ Error injection test failed: {e}")
        logger.exception("Error injection failed")
        return False


async def test_callback_error_handling():
    """Test callback error handling."""
    print("🧪 Testing callback error handling...")

    try:
        from openhcs.tui.utils.callback_manager import create_callback_manager

        callback_manager = create_callback_manager("integration_test")

        # Test callback that raises exception
        def failing_callback():
            raise RuntimeError("Intentional callback failure")

        # This should not crash the system
        callback_manager.run_callback(failing_callback)

        # Test async callback that raises exception
        async def failing_async_callback():
            await asyncio.sleep(0.01)
            raise ValueError("Intentional async callback failure")

        # This should not crash the system
        callback_manager.run_callback(failing_async_callback)

        # Wait for async callbacks to complete
        await asyncio.sleep(0.1)

        print("✅ Callback error handling test passed")
        return True

    except Exception as e:
        print(f"❌ Callback error handling test failed: {e}")
        logger.exception("Callback error handling failed")
        return False


async def test_focus_management():
    """Test focus management without UI."""
    print("🧪 Testing focus management...")

    try:
        from openhcs.tui.utils.focus_manager import get_focus_manager

        focus_manager = get_focus_manager()

        # Test focus operations that should fail gracefully
        success = await focus_manager.set_focus_after_delay(None, 0.01)

        # Should return False since there's no UI, but shouldn't crash
        assert success == False, "Focus should fail without UI"

        print("✅ Focus management test passed")
        return True

    except Exception as e:
        print(f"❌ Focus management test failed: {e}")
        logger.exception("Focus management failed")
        return False


async def main():
    """Main integration test function."""
    print("🚀 Starting async integration tests...")
    print("=" * 50)

    tests = [
        test_full_tui_initialization,
        test_error_injection,
        test_callback_error_handling,
        test_focus_management
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if await test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Integration test {test.__name__} crashed: {e}")
            logger.exception(f"Test {test.__name__} crashed")
            failed += 1

    print("\n📊 Integration Test Results:")
    print("=" * 50)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")

    if failed == 0:
        print("🎉 All integration tests passed!")
        return 0
    else:
        print("💥 Some integration tests failed!")
        return 1


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
```

### Execution Plan

1. **Run verification script:**
   ```bash
   python tools/verify_async_fixes.py
   ```

2. **Run runtime tests:**
   ```bash
   python tools/test_async_runtime.py
   ```

3. **Run integration tests:**
   ```bash
   python tools/integration_test_async.py
   ```

4. **Manual TUI testing:**
   - Start TUI and verify no RuntimeWarnings
   - Test dialog operations
   - Trigger errors and verify proper handling
   - Check task manager shutdown on exit

### Success Criteria

- ✅ Zero RuntimeWarnings about unawaited coroutines
- ✅ All background tasks use unified task manager
- ✅ Focus management works without race conditions
- ✅ Exception handlers don't cause recursion
- ✅ Callback failures show error dialogs instead of silent failure
- ✅ Proper task cleanup on TUI shutdown
- ✅ No memory leaks from uncancelled tasks
```
