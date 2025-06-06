"""
Strategic Test Suite for PatternEditingService.

DNA-guided tests targeting critical pattern editing functionality.
Coverage Gap: E=5, C=44 (SECOND HIGHEST COVERAGE PRIORITY)
Entropy: 103, Complexity: 44, Errors: 5

🔬 Test Strategy:
- Pattern validation and manipulation
- Function registry interactions
- Pattern conversion operations
- Business logic for pattern editing
"""
import pytest
import copy
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.test_framework import (
    test_framework, TestMetrics, TestResult
)
from openhcs.tui.services.pattern_editing_service import PatternEditingService


class TestPatternEditingService:
    """
    Strategic test suite for PatternEditingService.
    
    Tests prioritized by DNA analysis focusing on:
    - Pattern manipulation complexity
    - Function registry integration
    - Validation logic patterns
    """
    
    @pytest.fixture
    def mock_state(self):
        """Create mock TUIState."""
        return test_framework.create_mock_state()
    
    @pytest.fixture
    def service(self, mock_state):
        """Create PatternEditingService instance."""
        return PatternEditingService(mock_state)
    
    @pytest.fixture
    def sample_list_pattern(self):
        """Create sample list pattern."""
        return [
            {'func': 'test_func_1', 'kwargs': {'param1': 'value1'}},
            {'func': 'test_func_2', 'kwargs': {'param2': 42}}
        ]
    
    @pytest.fixture
    def sample_dict_pattern(self):
        """Create sample dictionary pattern."""
        return {
            'preprocessing': [
                {'func': 'preprocess_func', 'kwargs': {'threshold': 0.5}}
            ],
            'analysis': [
                {'func': 'analysis_func', 'kwargs': {'method': 'advanced'}}
            ]
        }
    
    @pytest.fixture
    def mock_function(self):
        """Create mock function for testing."""
        def test_function(data, param1=None, param2=None):
            """Test function for pattern editing."""
            return data
        
        test_function.__name__ = 'test_function'
        test_function.backend = 'cpu'
        return test_function
    
    @pytest.mark.high_priority
    def test_create_empty_pattern_list(self, service):
        """
        Test creating empty list pattern.
        
        DNA Priority: MEDIUM (pattern creation complexity)
        Entropy Target: Pattern initialization
        """
        test_metrics = TestMetrics(
            entropy=103.0,
            complexity=44,
            error_count=5,
            refactoring_vector=7.7,
            spectral_rank=0.7,
            topological_importance=0.6
        )
        
        def test_logic():
            pattern = service.create_empty_pattern('list')
            assert isinstance(pattern, list)
            assert len(pattern) == 0
        
        result = test_framework.run_test_with_metrics(
            test_logic, "test_create_empty_pattern_list", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.high_priority
    def test_create_empty_pattern_dict(self, service):
        """
        Test creating empty dictionary pattern.
        
        DNA Priority: MEDIUM (pattern creation complexity)
        Entropy Target: Pattern initialization
        """
        test_metrics = TestMetrics(
            entropy=103.0,
            complexity=44,
            error_count=5,
            refactoring_vector=7.7,
            spectral_rank=0.7,
            topological_importance=0.6
        )
        
        def test_logic():
            pattern = service.create_empty_pattern('dict')
            assert isinstance(pattern, dict)
            assert len(pattern) == 0
        
        result = test_framework.run_test_with_metrics(
            test_logic, "test_create_empty_pattern_dict", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.high_priority
    def test_clone_pattern_list(self, service, sample_list_pattern):
        """
        Test cloning list pattern.
        
        DNA Priority: HIGH (deep copy complexity)
        Entropy Target: Pattern cloning
        """
        test_metrics = TestMetrics(
            entropy=108.0,
            complexity=50,
            error_count=8,
            refactoring_vector=8.2,
            spectral_rank=0.8,
            topological_importance=0.7
        )
        
        def test_logic():
            cloned = service.clone_pattern(sample_list_pattern)
            
            # Verify deep copy
            assert cloned == sample_list_pattern
            assert cloned is not sample_list_pattern
            assert cloned[0] is not sample_list_pattern[0]
            
            # Modify clone and verify original unchanged
            cloned[0]['kwargs']['param1'] = 'modified'
            assert sample_list_pattern[0]['kwargs']['param1'] == 'value1'
        
        result = test_framework.run_test_with_metrics(
            test_logic, "test_clone_pattern_list", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.high_priority
    def test_clone_pattern_dict(self, service, sample_dict_pattern):
        """
        Test cloning dictionary pattern.
        
        DNA Priority: HIGH (deep copy complexity)
        Entropy Target: Pattern cloning
        """
        test_metrics = TestMetrics(
            entropy=108.0,
            complexity=50,
            error_count=8,
            refactoring_vector=8.2,
            spectral_rank=0.8,
            topological_importance=0.7
        )
        
        def test_logic():
            cloned = service.clone_pattern(sample_dict_pattern)
            
            # Verify deep copy
            assert cloned == sample_dict_pattern
            assert cloned is not sample_dict_pattern
            assert cloned['preprocessing'] is not sample_dict_pattern['preprocessing']
            
            # Modify clone and verify original unchanged
            cloned['preprocessing'][0]['kwargs']['threshold'] = 0.8
            assert sample_dict_pattern['preprocessing'][0]['kwargs']['threshold'] == 0.5
        
        result = test_framework.run_test_with_metrics(
            test_logic, "test_clone_pattern_dict", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.high_priority
    def test_is_dict_pattern(self, service, sample_list_pattern, sample_dict_pattern):
        """
        Test pattern type detection.
        
        DNA Priority: MEDIUM (type checking complexity)
        Entropy Target: Pattern type validation
        """
        test_metrics = TestMetrics(
            entropy=105.0,
            complexity=35,
            error_count=3,
            refactoring_vector=7.5,
            spectral_rank=0.6,
            topological_importance=0.5
        )
        
        def test_logic():
            assert service.is_dict_pattern(sample_dict_pattern) is True
            assert service.is_dict_pattern(sample_list_pattern) is False
            assert service.is_dict_pattern({}) is True
            assert service.is_dict_pattern([]) is False
        
        result = test_framework.run_test_with_metrics(
            test_logic, "test_is_dict_pattern", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.high_priority
    def test_get_pattern_keys(self, service, sample_dict_pattern, sample_list_pattern):
        """
        Test getting pattern keys.
        
        DNA Priority: MEDIUM (key extraction complexity)
        Entropy Target: Key management
        """
        test_metrics = TestMetrics(
            entropy=107.0,
            complexity=40,
            error_count=4,
            refactoring_vector=7.8,
            spectral_rank=0.7,
            topological_importance=0.6
        )
        
        def test_logic():
            # Dict pattern should return keys
            keys = service.get_pattern_keys(sample_dict_pattern)
            assert set(keys) == {'preprocessing', 'analysis'}
            
            # List pattern should return empty list
            keys = service.get_pattern_keys(sample_list_pattern)
            assert keys == []
            
            # Empty dict should return empty list
            keys = service.get_pattern_keys({})
            assert keys == []
        
        result = test_framework.run_test_with_metrics(
            test_logic, "test_get_pattern_keys", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.high_priority
    def test_add_pattern_key_success(self, service):
        """
        Test adding pattern key successfully.
        
        DNA Priority: HIGH (pattern modification complexity)
        Entropy Target: Pattern mutation
        """
        test_metrics = TestMetrics(
            entropy=112.0,
            complexity=55,
            error_count=7,
            refactoring_vector=8.5,
            spectral_rank=0.8,
            topological_importance=0.7
        )
        
        def test_logic():
            pattern = {'existing_key': []}
            
            # Add new key
            result = service.add_pattern_key(pattern, 'new_key')
            
            # Verify key was added
            assert 'new_key' in pattern
            assert pattern['new_key'] == []
            assert result == pattern
        
        result = test_framework.run_test_with_metrics(
            test_logic, "test_add_pattern_key_success", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.high_priority
    def test_add_pattern_key_duplicate_error(self, service):
        """
        Test adding duplicate pattern key error.
        
        DNA Priority: HIGH (error handling complexity)
        Entropy Target: Validation error handling
        """
        test_metrics = TestMetrics(
            entropy=115.0,
            complexity=60,
            error_count=10,
            refactoring_vector=8.8,
            spectral_rank=0.9,
            topological_importance=0.8
        )
        
        def test_logic():
            pattern = {'existing_key': []}
            
            # Try to add existing key
            with pytest.raises(ValueError, match="already exists"):
                service.add_pattern_key(pattern, 'existing_key')
        
        result = test_framework.run_test_with_metrics(
            test_logic, "test_add_pattern_key_duplicate_error", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.high_priority
    def test_add_pattern_key_non_dict_error(self, service):
        """
        Test adding key to non-dict pattern error.
        
        DNA Priority: HIGH (type validation complexity)
        Entropy Target: Type validation error handling
        """
        test_metrics = TestMetrics(
            entropy=113.0,
            complexity=58,
            error_count=9,
            refactoring_vector=8.6,
            spectral_rank=0.8,
            topological_importance=0.7
        )
        
        def test_logic():
            pattern = []  # List pattern
            
            # Try to add key to list
            with pytest.raises(ValueError, match="Cannot add key to non-dictionary"):
                service.add_pattern_key(pattern, 'new_key')
        
        result = test_framework.run_test_with_metrics(
            test_logic, "test_add_pattern_key_non_dict_error", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.high_priority
    def test_remove_pattern_key_success(self, service, sample_dict_pattern):
        """
        Test removing pattern key successfully.
        
        DNA Priority: HIGH (pattern modification complexity)
        Entropy Target: Pattern mutation
        """
        test_metrics = TestMetrics(
            entropy=112.0,
            complexity=55,
            error_count=7,
            refactoring_vector=8.5,
            spectral_rank=0.8,
            topological_importance=0.7
        )
        
        def test_logic():
            pattern = copy.deepcopy(sample_dict_pattern)
            
            # Remove existing key
            result = service.remove_pattern_key(pattern, 'preprocessing')
            
            # Verify key was removed
            assert 'preprocessing' not in pattern
            assert 'analysis' in pattern  # Other key should remain
            assert result == pattern
        
        result = test_framework.run_test_with_metrics(
            test_logic, "test_remove_pattern_key_success", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed

    @pytest.mark.high_priority
    def test_convert_list_to_dict_pattern(self, service, sample_list_pattern):
        """
        Test converting list pattern to dictionary pattern.

        DNA Priority: HIGH (pattern conversion complexity)
        Entropy Target: Pattern transformation
        """
        test_metrics = TestMetrics(
            entropy=118.0,
            complexity=65,
            error_count=12,
            refactoring_vector=9.0,
            spectral_rank=0.9,
            topological_importance=0.8
        )

        def test_logic():
            result = service.convert_list_to_dict_pattern(sample_list_pattern)

            # Verify conversion
            assert isinstance(result, dict)
            assert None in result  # Should use None key for unnamed group
            assert result[None] == sample_list_pattern

            # Test with already dict pattern
            dict_pattern = {'key': []}
            result2 = service.convert_list_to_dict_pattern(dict_pattern)
            assert result2 == dict_pattern

        result = test_framework.run_test_with_metrics(
            test_logic, "test_convert_list_to_dict_pattern", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed

    @pytest.mark.high_priority
    def test_convert_dict_to_list_pattern(self, service, sample_dict_pattern):
        """
        Test converting dictionary pattern to list pattern.

        DNA Priority: HIGH (pattern conversion complexity)
        Entropy Target: Pattern transformation
        """
        test_metrics = TestMetrics(
            entropy=118.0,
            complexity=65,
            error_count=12,
            refactoring_vector=9.0,
            spectral_rank=0.9,
            topological_importance=0.8
        )

        def test_logic():
            # Test with None key
            dict_with_none = {None: sample_dict_pattern['preprocessing']}
            result = service.convert_dict_to_list_pattern(dict_with_none)
            assert result == sample_dict_pattern['preprocessing']

            # Test with regular key (should use first key)
            result2 = service.convert_dict_to_list_pattern(sample_dict_pattern)
            assert result2 == sample_dict_pattern['preprocessing']  # First key

            # Test with empty dict
            result3 = service.convert_dict_to_list_pattern({})
            assert result3 == []

            # Test with already list pattern
            list_pattern = []
            result4 = service.convert_dict_to_list_pattern(list_pattern)
            assert result4 == list_pattern

        result = test_framework.run_test_with_metrics(
            test_logic, "test_convert_dict_to_list_pattern", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed

    @pytest.mark.high_priority
    def test_add_function_to_list_pattern(self, service, mock_function):
        """
        Test adding function to list pattern.

        DNA Priority: HIGH (function manipulation complexity)
        Entropy Target: Function pattern modification
        """
        test_metrics = TestMetrics(
            entropy=115.0,
            complexity=60,
            error_count=10,
            refactoring_vector=8.7,
            spectral_rank=0.8,
            topological_importance=0.7
        )

        def test_logic():
            pattern = []
            kwargs = {'param1': 'value1', 'param2': 42}

            # Add function to list pattern
            result = service.add_function_to_pattern(pattern, mock_function, kwargs)

            # Verify function was added
            assert len(pattern) == 1
            assert pattern[0]['func'] == mock_function
            assert pattern[0]['kwargs'] == kwargs
            assert result == pattern

        result = test_framework.run_test_with_metrics(
            test_logic, "test_add_function_to_list_pattern", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed

    @pytest.mark.high_priority
    def test_add_function_to_dict_pattern(self, service, mock_function):
        """
        Test adding function to dictionary pattern.

        DNA Priority: HIGH (function manipulation complexity)
        Entropy Target: Function pattern modification
        """
        test_metrics = TestMetrics(
            entropy=115.0,
            complexity=60,
            error_count=10,
            refactoring_vector=8.7,
            spectral_rank=0.8,
            topological_importance=0.7
        )

        def test_logic():
            pattern = {'preprocessing': []}
            kwargs = {'threshold': 0.5}

            # Add function to dict pattern
            result = service.add_function_to_pattern(pattern, mock_function, kwargs, 'preprocessing')

            # Verify function was added
            assert len(pattern['preprocessing']) == 1
            assert pattern['preprocessing'][0]['func'] == mock_function
            assert pattern['preprocessing'][0]['kwargs'] == kwargs
            assert result == pattern

        result = test_framework.run_test_with_metrics(
            test_logic, "test_add_function_to_dict_pattern", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed

    @pytest.mark.high_priority
    def test_remove_function_from_pattern(self, service, sample_list_pattern):
        """
        Test removing function from pattern.

        DNA Priority: HIGH (function manipulation complexity)
        Entropy Target: Function pattern modification
        """
        test_metrics = TestMetrics(
            entropy=113.0,
            complexity=58,
            error_count=9,
            refactoring_vector=8.5,
            spectral_rank=0.8,
            topological_importance=0.7
        )

        def test_logic():
            pattern = copy.deepcopy(sample_list_pattern)
            original_length = len(pattern)

            # Remove function at index 0
            result = service.remove_function_from_pattern(pattern, 0)

            # Verify function was removed
            assert len(pattern) == original_length - 1
            assert pattern[0]['func'] == 'test_func_2'  # Second function moved to first
            assert result == pattern

        result = test_framework.run_test_with_metrics(
            test_logic, "test_remove_function_from_pattern", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed

    @pytest.mark.high_priority
    def test_move_function_in_pattern(self, service, sample_list_pattern):
        """
        Test moving function within pattern.

        DNA Priority: HIGH (function reordering complexity)
        Entropy Target: Function pattern reordering
        """
        test_metrics = TestMetrics(
            entropy=116.0,
            complexity=62,
            error_count=11,
            refactoring_vector=8.8,
            spectral_rank=0.9,
            topological_importance=0.8
        )

        def test_logic():
            pattern = copy.deepcopy(sample_list_pattern)

            # Move function from index 0 to index 1
            result = service.move_function_in_pattern(pattern, 0, 1)

            # Verify function was moved
            assert pattern[0]['func'] == 'test_func_2'  # Second function now first
            assert pattern[1]['func'] == 'test_func_1'  # First function now second
            assert result == pattern

        result = test_framework.run_test_with_metrics(
            test_logic, "test_move_function_in_pattern", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed

    @pytest.mark.high_priority
    def test_update_function_kwargs(self, service, sample_list_pattern):
        """
        Test updating function arguments.

        DNA Priority: HIGH (function modification complexity)
        Entropy Target: Function parameter modification
        """
        test_metrics = TestMetrics(
            entropy=114.0,
            complexity=59,
            error_count=8,
            refactoring_vector=8.6,
            spectral_rank=0.8,
            topological_importance=0.7
        )

        def test_logic():
            pattern = copy.deepcopy(sample_list_pattern)
            new_kwargs = {'new_param': 'new_value', 'another_param': 123}

            # Update function kwargs at index 0
            result = service.update_function_kwargs(pattern, 0, new_kwargs)

            # Verify kwargs were updated
            assert pattern[0]['kwargs'] == new_kwargs
            assert pattern[1]['kwargs'] == {'param2': 42}  # Other function unchanged
            assert result == pattern

        result = test_framework.run_test_with_metrics(
            test_logic, "test_update_function_kwargs", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed

    @pytest.mark.medium_priority
    def test_get_function_info(self, service, mock_function):
        """
        Test getting function information.

        DNA Priority: MEDIUM (function introspection complexity)
        Entropy Target: Function metadata extraction
        """
        test_metrics = TestMetrics(
            entropy=110.0,
            complexity=45,
            error_count=5,
            refactoring_vector=7.8,
            spectral_rank=0.7,
            topological_importance=0.6
        )

        def test_logic():
            # Mock function registry
            with patch.object(service, 'func_registry', {'test_function': mock_function}):
                info = service.get_function_info(mock_function)

                # Verify function info
                assert info['name'] == 'test_function'
                assert info['backend'] == 'cpu'
                assert 'signature' in info
                assert 'doc' in info

            # Test with None function
            info_none = service.get_function_info(None)
            assert info_none['name'] == 'None'
            assert info_none['backend'] == ''

        result = test_framework.run_test_with_metrics(
            test_logic, "test_get_function_info", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
