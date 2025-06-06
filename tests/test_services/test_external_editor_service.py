"""
Strategic Test Suite for ExternalEditorService.

DNA-guided tests targeting critical external editor functionality.
Coverage Gap: E=0, C=3 (SEVENTH HIGHEST COVERAGE PRIORITY)
Entropy: 108, Complexity: 3, Errors: 0

🔬 Test Strategy:
- External editor integration
- File I/O operations
- Pattern parsing and validation
- Error handling for external processes
"""
import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock, AsyncMock, patch, mock_open
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.test_framework import (
    test_framework, TestMetrics, TestResult
)
from openhcs.tui.services.external_editor_service import ExternalEditorService


class TestExternalEditorService:
    """
    Strategic test suite for ExternalEditorService.
    
    Tests prioritized by DNA analysis focusing on:
    - External process coordination
    - File I/O complexity
    - Pattern parsing and validation
    """
    
    @pytest.fixture
    def mock_state(self):
        """Create mock TUIState."""
        return test_framework.create_mock_state()
    
    @pytest.fixture
    def service(self, mock_state):
        """Create ExternalEditorService instance."""
        return ExternalEditorService(mock_state)
    
    @pytest.fixture
    def sample_pattern_content(self):
        """Create sample pattern content for testing."""
        return """pattern = [
    {'func': 'test_func_1', 'kwargs': {'param1': 'value1'}},
    {'func': 'test_func_2', 'kwargs': {'param2': 42}}
]"""
    
    @pytest.fixture
    def sample_dict_pattern_content(self):
        """Create sample dictionary pattern content for testing."""
        return """pattern = {
    'preprocessing': [
        {'func': 'preprocess_func', 'kwargs': {'threshold': 0.5}}
    ],
    'analysis': [
        {'func': 'analysis_func', 'kwargs': {'method': 'advanced'}}
    ]
}"""
    
    @pytest.mark.asyncio
    @pytest.mark.high_priority
    async def test_edit_pattern_in_external_editor_success(self, service, sample_pattern_content):
        """
        Test successful external editor pattern editing.
        
        DNA Priority: HIGH (external process complexity)
        Entropy Target: External process coordination
        """
        test_metrics = TestMetrics(
            entropy=108.0,
            complexity=3,
            error_count=0,
            refactoring_vector=7.5,
            spectral_rank=0.7,
            topological_importance=0.6
        )
        
        async def test_logic():
            with patch('tempfile.NamedTemporaryFile') as mock_temp:
                with patch('subprocess.run') as mock_subprocess:
                    with patch('builtins.open', mock_open(read_data=sample_pattern_content)) as mock_file:
                        # Setup mocks
                        mock_temp_file = Mock()
                        mock_temp_file.name = '/tmp/test_pattern.py'
                        mock_temp_file.__enter__ = Mock(return_value=mock_temp_file)
                        mock_temp_file.__exit__ = Mock(return_value=None)
                        mock_temp.return_value = mock_temp_file
                        
                        # Mock successful subprocess
                        mock_subprocess.return_value.returncode = 0
                        
                        # Test external editor
                        success, pattern, error = await service.edit_pattern_in_external_editor(sample_pattern_content)
                        
                        # Verify success
                        assert success is True
                        assert pattern is not None
                        assert error is None
                        
                        # Verify subprocess was called
                        mock_subprocess.assert_called_once()
                        
                        # Verify file operations
                        mock_file.assert_called()
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_edit_pattern_in_external_editor_success", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.asyncio
    @pytest.mark.high_priority
    async def test_edit_pattern_in_external_editor_subprocess_error(self, service, sample_pattern_content):
        """
        Test external editor with subprocess error.
        
        DNA Priority: HIGH (error handling complexity)
        Entropy Target: External process error handling
        """
        test_metrics = TestMetrics(
            entropy=112.0,
            complexity=6,
            error_count=1,
            refactoring_vector=8.0,
            spectral_rank=0.8,
            topological_importance=0.7
        )
        
        async def test_logic():
            with patch('tempfile.NamedTemporaryFile') as mock_temp:
                with patch('subprocess.run') as mock_subprocess:
                    # Setup mocks
                    mock_temp_file = Mock()
                    mock_temp_file.name = '/tmp/test_pattern.py'
                    mock_temp_file.__enter__ = Mock(return_value=mock_temp_file)
                    mock_temp_file.__exit__ = Mock(return_value=None)
                    mock_temp.return_value = mock_temp_file
                    
                    # Mock subprocess error
                    mock_subprocess.return_value.returncode = 1
                    mock_subprocess.return_value.stderr = "Editor failed"
                    
                    # Test external editor
                    success, pattern, error = await service.edit_pattern_in_external_editor(sample_pattern_content)
                    
                    # Verify failure
                    assert success is False
                    assert pattern is None
                    assert error is not None
                    assert "Editor failed" in error
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_edit_pattern_in_external_editor_subprocess_error", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.asyncio
    @pytest.mark.high_priority
    async def test_edit_pattern_in_external_editor_parse_error(self, service):
        """
        Test external editor with pattern parsing error.
        
        DNA Priority: HIGH (parsing complexity)
        Entropy Target: Pattern validation
        """
        test_metrics = TestMetrics(
            entropy=115.0,
            complexity=8,
            error_count=2,
            refactoring_vector=8.3,
            spectral_rank=0.8,
            topological_importance=0.7
        )
        
        async def test_logic():
            invalid_content = "pattern = invalid python syntax ["
            
            with patch('tempfile.NamedTemporaryFile') as mock_temp:
                with patch('subprocess.run') as mock_subprocess:
                    with patch('builtins.open', mock_open(read_data=invalid_content)) as mock_file:
                        # Setup mocks
                        mock_temp_file = Mock()
                        mock_temp_file.name = '/tmp/test_pattern.py'
                        mock_temp_file.__enter__ = Mock(return_value=mock_temp_file)
                        mock_temp_file.__exit__ = Mock(return_value=None)
                        mock_temp.return_value = mock_temp_file
                        
                        # Mock successful subprocess
                        mock_subprocess.return_value.returncode = 0
                        
                        # Test external editor
                        success, pattern, error = await service.edit_pattern_in_external_editor("initial_content")
                        
                        # Verify parsing failure
                        assert success is False
                        assert pattern is None
                        assert error is not None
                        assert "parsing" in error.lower() or "syntax" in error.lower()
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_edit_pattern_in_external_editor_parse_error", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.asyncio
    @pytest.mark.high_priority
    async def test_edit_pattern_dict_pattern(self, service, sample_dict_pattern_content):
        """
        Test external editor with dictionary pattern.
        
        DNA Priority: MEDIUM (pattern type complexity)
        Entropy Target: Dictionary pattern handling
        """
        test_metrics = TestMetrics(
            entropy=110.0,
            complexity=5,
            error_count=0,
            refactoring_vector=7.7,
            spectral_rank=0.7,
            topological_importance=0.6
        )
        
        async def test_logic():
            with patch('tempfile.NamedTemporaryFile') as mock_temp:
                with patch('subprocess.run') as mock_subprocess:
                    with patch('builtins.open', mock_open(read_data=sample_dict_pattern_content)) as mock_file:
                        # Setup mocks
                        mock_temp_file = Mock()
                        mock_temp_file.name = '/tmp/test_pattern.py'
                        mock_temp_file.__enter__ = Mock(return_value=mock_temp_file)
                        mock_temp_file.__exit__ = Mock(return_value=None)
                        mock_temp.return_value = mock_temp_file
                        
                        # Mock successful subprocess
                        mock_subprocess.return_value.returncode = 0
                        
                        # Test external editor
                        success, pattern, error = await service.edit_pattern_in_external_editor("initial_content")
                        
                        # Verify success with dict pattern
                        assert success is True
                        assert isinstance(pattern, dict)
                        assert error is None
                        assert 'preprocessing' in pattern
                        assert 'analysis' in pattern
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_edit_pattern_dict_pattern", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.asyncio
    @pytest.mark.medium_priority
    async def test_get_editor_command_default(self, service):
        """
        Test getting default editor command.
        
        DNA Priority: MEDIUM (configuration complexity)
        Entropy Target: Editor configuration
        """
        test_metrics = TestMetrics(
            entropy=107.0,
            complexity=2,
            error_count=0,
            refactoring_vector=7.0,
            spectral_rank=0.5,
            topological_importance=0.4
        )
        
        async def test_logic():
            with patch.dict(os.environ, {}, clear=True):
                # Test default editor
                editor_cmd = service._get_editor_command()
                
                # Verify default editor
                assert editor_cmd in ['nano', 'vi', 'vim', 'code', 'notepad']
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_get_editor_command_default", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.asyncio
    @pytest.mark.medium_priority
    async def test_get_editor_command_from_env(self, service):
        """
        Test getting editor command from environment variable.
        
        DNA Priority: MEDIUM (configuration complexity)
        Entropy Target: Environment configuration
        """
        test_metrics = TestMetrics(
            entropy=108.0,
            complexity=3,
            error_count=0,
            refactoring_vector=7.2,
            spectral_rank=0.6,
            topological_importance=0.5
        )
        
        async def test_logic():
            with patch.dict(os.environ, {'EDITOR': 'custom_editor'}, clear=True):
                # Test custom editor from environment
                editor_cmd = service._get_editor_command()
                
                # Verify custom editor
                assert editor_cmd == 'custom_editor'
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_get_editor_command_from_env", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.asyncio
    @pytest.mark.medium_priority
    async def test_parse_pattern_from_content_list(self, service, sample_pattern_content):
        """
        Test parsing list pattern from content.
        
        DNA Priority: MEDIUM (parsing complexity)
        Entropy Target: Pattern parsing
        """
        test_metrics = TestMetrics(
            entropy=111.0,
            complexity=6,
            error_count=0,
            refactoring_vector=7.8,
            spectral_rank=0.7,
            topological_importance=0.6
        )
        
        async def test_logic():
            # Test parsing list pattern
            pattern = service._parse_pattern_from_content(sample_pattern_content)
            
            # Verify parsed pattern
            assert isinstance(pattern, list)
            assert len(pattern) == 2
            assert pattern[0]['func'] == 'test_func_1'
            assert pattern[1]['func'] == 'test_func_2'
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_parse_pattern_from_content_list", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.asyncio
    @pytest.mark.medium_priority
    async def test_parse_pattern_from_content_dict(self, service, sample_dict_pattern_content):
        """
        Test parsing dictionary pattern from content.
        
        DNA Priority: MEDIUM (parsing complexity)
        Entropy Target: Pattern parsing
        """
        test_metrics = TestMetrics(
            entropy=111.0,
            complexity=6,
            error_count=0,
            refactoring_vector=7.8,
            spectral_rank=0.7,
            topological_importance=0.6
        )
        
        async def test_logic():
            # Test parsing dict pattern
            pattern = service._parse_pattern_from_content(sample_dict_pattern_content)
            
            # Verify parsed pattern
            assert isinstance(pattern, dict)
            assert 'preprocessing' in pattern
            assert 'analysis' in pattern
            assert len(pattern['preprocessing']) == 1
            assert len(pattern['analysis']) == 1
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_parse_pattern_from_content_dict", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.asyncio
    @pytest.mark.high_priority
    async def test_parse_pattern_from_content_invalid(self, service):
        """
        Test parsing invalid pattern content.
        
        DNA Priority: HIGH (error handling complexity)
        Entropy Target: Parsing error handling
        """
        test_metrics = TestMetrics(
            entropy=114.0,
            complexity=7,
            error_count=1,
            refactoring_vector=8.1,
            spectral_rank=0.8,
            topological_importance=0.7
        )
        
        async def test_logic():
            invalid_content = "pattern = invalid syntax ["
            
            # Test parsing invalid content
            with pytest.raises(Exception):
                service._parse_pattern_from_content(invalid_content)
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_parse_pattern_from_content_invalid", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.asyncio
    @pytest.mark.medium_priority
    async def test_create_temp_file_with_content(self, service, sample_pattern_content):
        """
        Test creating temporary file with content.
        
        DNA Priority: MEDIUM (file I/O complexity)
        Entropy Target: File operations
        """
        test_metrics = TestMetrics(
            entropy=109.0,
            complexity=4,
            error_count=0,
            refactoring_vector=7.5,
            spectral_rank=0.6,
            topological_importance=0.5
        )
        
        async def test_logic():
            with patch('tempfile.NamedTemporaryFile') as mock_temp:
                mock_temp_file = Mock()
                mock_temp_file.name = '/tmp/test_pattern.py'
                mock_temp_file.write = Mock()
                mock_temp_file.flush = Mock()
                mock_temp_file.__enter__ = Mock(return_value=mock_temp_file)
                mock_temp_file.__exit__ = Mock(return_value=None)
                mock_temp.return_value = mock_temp_file
                
                # Test creating temp file
                with service._create_temp_file_with_content(sample_pattern_content) as temp_file:
                    # Verify temp file operations
                    assert temp_file == mock_temp_file
                    mock_temp_file.write.assert_called()
                    mock_temp_file.flush.assert_called()
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_create_temp_file_with_content", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.asyncio
    @pytest.mark.medium_priority
    async def test_run_editor_command_success(self, service):
        """
        Test running editor command successfully.
        
        DNA Priority: MEDIUM (subprocess complexity)
        Entropy Target: External process execution
        """
        test_metrics = TestMetrics(
            entropy=110.0,
            complexity=5,
            error_count=0,
            refactoring_vector=7.6,
            spectral_rank=0.7,
            topological_importance=0.6
        )
        
        async def test_logic():
            with patch('subprocess.run') as mock_subprocess:
                # Mock successful subprocess
                mock_subprocess.return_value.returncode = 0
                mock_subprocess.return_value.stderr = ""
                
                # Test running editor command
                success, error = await service._run_editor_command('nano', '/tmp/test.py')
                
                # Verify success
                assert success is True
                assert error is None
                
                # Verify subprocess was called
                mock_subprocess.assert_called_once()
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_run_editor_command_success", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.asyncio
    @pytest.mark.medium_priority
    async def test_run_editor_command_failure(self, service):
        """
        Test running editor command with failure.
        
        DNA Priority: MEDIUM (error handling complexity)
        Entropy Target: External process error handling
        """
        test_metrics = TestMetrics(
            entropy=112.0,
            complexity=6,
            error_count=1,
            refactoring_vector=7.9,
            spectral_rank=0.7,
            topological_importance=0.6
        )
        
        async def test_logic():
            with patch('subprocess.run') as mock_subprocess:
                # Mock failed subprocess
                mock_subprocess.return_value.returncode = 1
                mock_subprocess.return_value.stderr = "Command not found"
                
                # Test running editor command
                success, error = await service._run_editor_command('invalid_editor', '/tmp/test.py')
                
                # Verify failure
                assert success is False
                assert error is not None
                assert "Command not found" in error
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_run_editor_command_failure", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
