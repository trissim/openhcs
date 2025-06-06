"""
Strategic Test Suite for PatternEditorController.

DNA-guided tests targeting critical pattern editor functionality.
Coverage Gap: E=0, C=5 (FIFTH HIGHEST COVERAGE PRIORITY)
Entropy: 109, Complexity: 5, Errors: 0

🔬 Test Strategy:
- Pattern editor coordination
- Service integration patterns
- UI state management
- Pattern validation workflows
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.test_framework import (
    test_framework, TestMetrics, TestResult
)
from openhcs.tui.controllers.pattern_editor_controller import PatternEditorController


class TestPatternEditorController:
    """
    Strategic test suite for PatternEditorController.
    
    Tests prioritized by DNA analysis focusing on:
    - Pattern editor coordination
    - Service integration complexity
    - UI state synchronization
    """
    
    @pytest.fixture
    def mock_state(self):
        """Create mock TUIState."""
        return test_framework.create_mock_state()
    
    @pytest.fixture
    def mock_pattern_service(self):
        """Create mock PatternEditingService."""
        service = Mock()
        service.clone_pattern = Mock(side_effect=lambda x: x.copy() if x else [])
        service.is_dict_pattern = Mock(return_value=False)
        service.get_pattern_keys = Mock(return_value=[])
        service.get_pattern_functions = Mock(return_value=[])
        service.get_available_functions = Mock(return_value=[])
        service.get_function_info = Mock(return_value={'name': 'test_func', 'backend': 'cpu'})
        service.add_pattern_key = Mock()
        service.remove_pattern_key = Mock()
        service.convert_list_to_dict_pattern = Mock(return_value={})
        service.add_function_to_pattern = Mock()
        service.remove_function_from_pattern = Mock()
        service.move_function_in_pattern = Mock()
        service.update_function_kwargs = Mock()
        service.validate_pattern = Mock(return_value=(True, None))
        return service
    
    @pytest.fixture
    def mock_external_editor_service(self):
        """Create mock ExternalEditorService."""
        service = Mock()
        service.edit_pattern_in_external_editor = AsyncMock(return_value=(True, [], None))
        return service
    
    @pytest.fixture
    def sample_pattern(self):
        """Create sample pattern for testing."""
        return [
            {'func': 'test_func_1', 'kwargs': {'param1': 'value1'}},
            {'func': 'test_func_2', 'kwargs': {'param2': 42}}
        ]
    
    @pytest.fixture
    def controller(self, mock_state, sample_pattern):
        """Create PatternEditorController instance."""
        with patch('openhcs.tui.controllers.pattern_editor_controller.PatternEditingService') as mock_ps:
            with patch('openhcs.tui.controllers.pattern_editor_controller.ExternalEditorService') as mock_es:
                # Setup service mocks
                mock_pattern_service = self.mock_pattern_service()
                mock_ps.return_value = mock_pattern_service
                
                mock_external_service = self.mock_external_editor_service()
                mock_es.return_value = mock_external_service
                
                # Create controller
                controller = PatternEditorController(
                    mock_state,
                    initial_pattern=sample_pattern,
                    change_callback=None
                )
                
                # Assign mocked services
                controller.pattern_service = mock_pattern_service
                controller.external_editor_service = mock_external_service
                
                return controller
    
    @pytest.mark.high_priority
    def test_get_pattern(self, controller, sample_pattern):
        """
        Test getting current pattern.
        
        DNA Priority: MEDIUM (pattern access complexity)
        Entropy Target: Pattern state access
        """
        test_metrics = TestMetrics(
            entropy=109.0,
            complexity=5,
            error_count=0,
            refactoring_vector=7.0,
            spectral_rank=0.6,
            topological_importance=0.5
        )
        
        def test_logic():
            # Test getting pattern
            pattern = controller.get_pattern()
            
            # Verify pattern returned
            assert pattern == sample_pattern
        
        result = test_framework.run_test_with_metrics(
            test_logic, "test_get_pattern", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.high_priority
    def test_get_original_pattern(self, controller, sample_pattern):
        """
        Test getting original pattern.
        
        DNA Priority: MEDIUM (pattern state complexity)
        Entropy Target: Original state preservation
        """
        test_metrics = TestMetrics(
            entropy=109.0,
            complexity=5,
            error_count=0,
            refactoring_vector=7.0,
            spectral_rank=0.6,
            topological_importance=0.5
        )
        
        def test_logic():
            # Test getting original pattern
            original = controller.get_original_pattern()
            
            # Verify original pattern returned
            assert original == sample_pattern
        
        result = test_framework.run_test_with_metrics(
            test_logic, "test_get_original_pattern", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.high_priority
    def test_has_changes_false(self, controller):
        """
        Test checking for changes when none exist.
        
        DNA Priority: MEDIUM (change detection complexity)
        Entropy Target: State comparison
        """
        test_metrics = TestMetrics(
            entropy=110.0,
            complexity=7,
            error_count=0,
            refactoring_vector=7.2,
            spectral_rank=0.6,
            topological_importance=0.5
        )
        
        def test_logic():
            # Test checking for changes (should be false initially)
            has_changes = controller.has_changes()
            
            # Verify no changes detected
            assert has_changes is False
        
        result = test_framework.run_test_with_metrics(
            test_logic, "test_has_changes_false", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.high_priority
    def test_has_changes_true(self, controller):
        """
        Test checking for changes when they exist.
        
        DNA Priority: MEDIUM (change detection complexity)
        Entropy Target: State comparison
        """
        test_metrics = TestMetrics(
            entropy=110.0,
            complexity=7,
            error_count=0,
            refactoring_vector=7.2,
            spectral_rank=0.6,
            topological_importance=0.5
        )
        
        def test_logic():
            # Modify current pattern
            controller.current_pattern.append({'func': 'new_func', 'kwargs': {}})
            
            # Test checking for changes
            has_changes = controller.has_changes()
            
            # Verify changes detected
            assert has_changes is True
        
        result = test_framework.run_test_with_metrics(
            test_logic, "test_has_changes_true", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.high_priority
    def test_get_pattern_keys(self, controller):
        """
        Test getting pattern keys.
        
        DNA Priority: MEDIUM (key access complexity)
        Entropy Target: Key management
        """
        test_metrics = TestMetrics(
            entropy=111.0,
            complexity=8,
            error_count=0,
            refactoring_vector=7.3,
            spectral_rank=0.7,
            topological_importance=0.6
        )
        
        def test_logic():
            # Test getting pattern keys
            keys = controller.get_pattern_keys()
            
            # Verify service was called
            controller.pattern_service.get_pattern_keys.assert_called_once_with(controller.current_pattern)
            
            # Verify keys returned
            assert keys == []
        
        result = test_framework.run_test_with_metrics(
            test_logic, "test_get_pattern_keys", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.high_priority
    def test_set_current_key(self, controller):
        """
        Test setting current key.
        
        DNA Priority: MEDIUM (key management complexity)
        Entropy Target: Key state management
        """
        test_metrics = TestMetrics(
            entropy=111.0,
            complexity=8,
            error_count=0,
            refactoring_vector=7.3,
            spectral_rank=0.7,
            topological_importance=0.6
        )
        
        def test_logic():
            with patch.object(controller, '_notify_ui_update') as mock_notify:
                # Test setting current key
                controller.set_current_key('test_key')
                
                # Verify key was set
                assert controller.current_key == 'test_key'
                
                # Verify UI update was triggered
                mock_notify.assert_called_once()
        
        result = test_framework.run_test_with_metrics(
            test_logic, "test_set_current_key", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.high_priority
    def test_get_current_functions(self, controller):
        """
        Test getting current functions.
        
        DNA Priority: MEDIUM (function access complexity)
        Entropy Target: Function state access
        """
        test_metrics = TestMetrics(
            entropy=112.0,
            complexity=9,
            error_count=0,
            refactoring_vector=7.4,
            spectral_rank=0.7,
            topological_importance=0.6
        )
        
        def test_logic():
            # Test getting current functions
            functions = controller.get_current_functions()
            
            # Verify service was called
            controller.pattern_service.get_pattern_functions.assert_called_once_with(
                controller.current_pattern, controller.current_key
            )
            
            # Verify functions returned
            assert functions == []
        
        result = test_framework.run_test_with_metrics(
            test_logic, "test_get_current_functions", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.high_priority
    def test_get_available_functions(self, controller):
        """
        Test getting available functions.
        
        DNA Priority: MEDIUM (function registry access complexity)
        Entropy Target: Function registry integration
        """
        test_metrics = TestMetrics(
            entropy=110.0,
            complexity=6,
            error_count=0,
            refactoring_vector=7.1,
            spectral_rank=0.6,
            topological_importance=0.5
        )
        
        def test_logic():
            # Test getting available functions
            functions = controller.get_available_functions()
            
            # Verify functions returned
            assert functions == []
        
        result = test_framework.run_test_with_metrics(
            test_logic, "test_get_available_functions", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.high_priority
    def test_get_function_info(self, controller):
        """
        Test getting function information.
        
        DNA Priority: MEDIUM (function introspection complexity)
        Entropy Target: Function metadata access
        """
        test_metrics = TestMetrics(
            entropy=111.0,
            complexity=7,
            error_count=0,
            refactoring_vector=7.2,
            spectral_rank=0.6,
            topological_importance=0.5
        )
        
        def test_logic():
            # Create mock function
            mock_func = Mock()
            
            # Test getting function info
            info = controller.get_function_info(mock_func)
            
            # Verify service was called
            controller.pattern_service.get_function_info.assert_called_once_with(mock_func)
            
            # Verify info returned
            assert info == {'name': 'test_func', 'backend': 'cpu'}
        
        result = test_framework.run_test_with_metrics(
            test_logic, "test_get_function_info", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
