"""
Strategic Test Suite for SimplifiedCommands.

DNA-guided tests targeting critical command functionality.
Refactoring Vector: 8.3 (HIGH PRIORITY)
Entropy: 109, Complexity: 31, Errors: 0

🔬 Test Strategy:
- Command execution patterns
- Service integration
- Error handling and validation
- State management consistency
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
from openhcs.tui.commands.dialog_commands import ShowHelpCommand, ShowGlobalSettingsDialogCommand
from openhcs.tui.commands.pipeline_commands import (
    InitializePlatesCommand, CompilePlatesCommand, RunPlatesCommand, DeleteSelectedPlatesCommand
)
from openhcs.tui.commands.pipeline_step_commands import (
    AddStepCommand, RemoveStepCommand, ValidatePipelineCommand
)


class TestSimplifiedCommands:
    """
    Strategic test suite for SimplifiedCommands.
    
    Tests prioritized by DNA analysis focusing on:
    - Command execution patterns
    - Service integration complexity
    - Error handling consistency
    """
    
    @pytest.fixture
    def mock_state(self):
        """Create mock TUIState."""
        return test_framework.create_mock_state()
    
    @pytest.fixture
    def mock_context(self):
        """Create mock ProcessingContext."""
        return test_framework.create_mock_context()
    
    @pytest.fixture
    def mock_service(self):
        """Create mock service."""
        service = Mock()
        service.initialize_plates = AsyncMock(return_value={
            'successful': ['plate1', 'plate2'],
            'failed': [],
            'errors': []
        })
        service.compile_plates = AsyncMock(return_value={
            'successful': ['plate1', 'plate2'],
            'failed': [],
            'errors': [],
            'compiled_contexts': {'plate1': {}, 'plate2': {}}
        })
        service.run_plates = AsyncMock(return_value={
            'successful': ['plate1', 'plate2'],
            'failed': [],
            'errors': []
        })
        service.delete_plates = AsyncMock(return_value={
            'successful': ['plate1'],
            'failed': [],
            'errors': []
        })
        return service
    
    @pytest.mark.asyncio
    async def test_show_help_command(self, mock_state):
        """
        Test ShowHelpCommand execution.
        
        DNA Priority: LOW (simple dialog complexity)
        Entropy Target: Dialog coordination
        """
        test_metrics = TestMetrics(
            entropy=105.0,
            complexity=20,
            error_count=0,
            refactoring_vector=5.0,
            spectral_rank=0.3,
            topological_importance=0.2
        )
        
        async def test_logic():
            command = ShowHelpCommand(mock_state)

            # Test execution - the command shows a dialog directly
            # We can't easily test the dialog display without mocking the dialog system
            # So we just test that the command executes without error
            await command.execute()

            # Verify the command can be executed
            assert command.can_execute(mock_state) is True
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_show_help_command", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.asyncio
    async def test_show_settings_command(self, mock_state):
        """
        Test ShowSettingsCommand execution.
        
        DNA Priority: LOW (simple dialog complexity)
        Entropy Target: Dialog coordination
        """
        test_metrics = TestMetrics(
            entropy=105.0,
            complexity=20,
            error_count=0,
            refactoring_vector=5.0,
            spectral_rank=0.3,
            topological_importance=0.2
        )
        
        async def test_logic():
            command = ShowGlobalSettingsDialogCommand(mock_state)

            # Test execution
            await command.execute()

            # Verify dialog request was sent
            mock_state.notify.assert_called_once()
            call_args = mock_state.notify.call_args[0]
            assert call_args[0] == 'show_dialog_requested'
            assert call_args[1]['type'] == 'info'
            assert 'Global Settings' in call_args[1]['data']['title']
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_show_settings_command", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.asyncio
    async def test_initialize_plates_command_success(self, mock_state, mock_service):
        """
        Test InitializePlatesCommand successful execution.
        
        DNA Priority: HIGH (service integration complexity)
        Entropy Target: Service coordination
        """
        test_metrics = TestMetrics(
            entropy=115.0,
            complexity=80,
            error_count=5,
            refactoring_vector=8.0,
            spectral_rank=0.8,
            topological_importance=0.7
        )
        
        async def test_logic():
            command = InitializePlatesCommand(mock_state, service=mock_service)
            
            # Create mock orchestrators
            orchestrators = [
                test_framework.create_mock_orchestrator("plate1"),
                test_framework.create_mock_orchestrator("plate2")
            ]
            
            # Test execution
            await command.execute(orchestrators_to_initialize=orchestrators)
            
            # Verify service was called
            mock_service.initialize_plates.assert_called_once_with(orchestrators)
            
            # Verify success notification
            mock_state.notify.assert_called()
            # Check for success notification
            success_calls = [call for call in mock_state.notify.call_args_list 
                           if call[0][0] == 'operation_status_changed' and 
                           call[0][1].get('status') == 'success']
            assert len(success_calls) > 0
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_initialize_plates_command_success", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.asyncio
    async def test_initialize_plates_command_no_selection(self, mock_state, mock_service):
        """
        Test InitializePlatesCommand with no plates selected.
        
        DNA Priority: MEDIUM (validation complexity)
        Entropy Target: Input validation
        """
        test_metrics = TestMetrics(
            entropy=110.0,
            complexity=60,
            error_count=3,
            refactoring_vector=7.0,
            spectral_rank=0.6,
            topological_importance=0.5
        )
        
        async def test_logic():
            command = InitializePlatesCommand(mock_state, service=mock_service)
            
            # Test execution with no orchestrators
            await command.execute(orchestrators_to_initialize=[])
            
            # Verify service was not called
            mock_service.initialize_plates.assert_not_called()
            
            # Verify info notification
            mock_state.notify.assert_called()
            info_calls = [call for call in mock_state.notify.call_args_list 
                         if call[0][0] == 'operation_status_changed' and 
                         call[0][1].get('status') == 'info']
            assert len(info_calls) > 0
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_initialize_plates_command_no_selection", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.asyncio
    async def test_compile_plates_command_success(self, mock_state, mock_service):
        """
        Test CompilePlatesCommand successful execution.
        
        DNA Priority: HIGH (compilation complexity)
        Entropy Target: Pipeline compilation
        """
        test_metrics = TestMetrics(
            entropy=118.0,
            complexity=100,
            error_count=8,
            refactoring_vector=8.5,
            spectral_rank=0.9,
            topological_importance=0.8
        )
        
        async def test_logic():
            command = CompilePlatesCommand(mock_state, service=mock_service)
            
            # Create mock orchestrators
            orchestrators = [
                test_framework.create_mock_orchestrator("plate1"),
                test_framework.create_mock_orchestrator("plate2")
            ]
            
            # Test execution
            await command.execute(orchestrators_to_compile=orchestrators)
            
            # Verify service was called
            mock_service.compile_plates.assert_called_once_with(orchestrators)
            
            # Verify state was updated
            assert mock_state.compiled_contexts == {'plate1': {}, 'plate2': {}}
            assert mock_state.is_compiled is True
            
            # Verify success notification
            mock_state.notify.assert_called()
            success_calls = [call for call in mock_state.notify.call_args_list 
                           if call[0][0] == 'operation_status_changed' and 
                           call[0][1].get('status') == 'success']
            assert len(success_calls) > 0
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_compile_plates_command_success", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.asyncio
    async def test_run_plates_command_success(self, mock_state, mock_service):
        """
        Test RunPlatesCommand successful execution.
        
        DNA Priority: HIGH (execution complexity)
        Entropy Target: Pipeline execution
        """
        test_metrics = TestMetrics(
            entropy=122.0,
            complexity=120,
            error_count=10,
            refactoring_vector=9.0,
            spectral_rank=1.0,
            topological_importance=0.9
        )
        
        async def test_logic():
            command = RunPlatesCommand(mock_state, service=mock_service)
            
            # Set up compiled contexts
            mock_state.compiled_contexts = {'plate1': {}, 'plate2': {}}
            
            # Create mock orchestrators
            orchestrators = [
                test_framework.create_mock_orchestrator("plate1"),
                test_framework.create_mock_orchestrator("plate2")
            ]
            
            # Test execution
            await command.execute(orchestrators_to_run=orchestrators)
            
            # Verify service was called
            mock_service.run_plates.assert_called_once_with(orchestrators, mock_state.compiled_contexts)
            
            # Verify running state was managed
            assert mock_state.is_running is False  # Should be reset after execution
            
            # Verify success notification
            mock_state.notify.assert_called()
            success_calls = [call for call in mock_state.notify.call_args_list 
                           if call[0][0] == 'operation_status_changed' and 
                           call[0][1].get('status') == 'success']
            assert len(success_calls) > 0
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_run_plates_command_success", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.asyncio
    async def test_run_plates_command_not_compiled(self, mock_state, mock_service):
        """
        Test RunPlatesCommand with no compiled plates.
        
        DNA Priority: MEDIUM (validation complexity)
        Entropy Target: State validation
        """
        test_metrics = TestMetrics(
            entropy=115.0,
            complexity=90,
            error_count=6,
            refactoring_vector=7.5,
            spectral_rank=0.7,
            topological_importance=0.6
        )
        
        async def test_logic():
            command = RunPlatesCommand(mock_state, service=mock_service)
            
            # No compiled contexts
            mock_state.compiled_contexts = {}
            
            # Create mock orchestrators
            orchestrators = [test_framework.create_mock_orchestrator("plate1")]
            
            # Test execution
            await command.execute(orchestrators_to_run=orchestrators)
            
            # Verify service was not called
            mock_service.run_plates.assert_not_called()
            
            # Verify error notification
            mock_state.notify.assert_called()
            error_calls = [call for call in mock_state.notify.call_args_list 
                          if call[0][0] == 'error']
            assert len(error_calls) > 0
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_run_plates_command_not_compiled", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.asyncio
    async def test_add_step_command_success(self, mock_state):
        """
        Test AddStepCommand successful execution.
        
        DNA Priority: MEDIUM (pipeline editing complexity)
        Entropy Target: Pipeline modification
        """
        test_metrics = TestMetrics(
            entropy=112.0,
            complexity=70,
            error_count=4,
            refactoring_vector=7.2,
            spectral_rank=0.6,
            topological_importance=0.5
        )
        
        async def test_logic():
            command = AddStepCommand(mock_state)
            
            # Set up active orchestrator
            mock_state.active_orchestrator = test_framework.create_mock_orchestrator("test_plate")
            
            # Test execution
            await command.execute()
            
            # Verify notification was sent
            mock_state.notify.assert_called_once()
            call_args = mock_state.notify.call_args[0]
            assert call_args[0] == 'add_step_requested'
            assert call_args[1]['orchestrator'] == mock_state.active_orchestrator
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_add_step_command_success", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.asyncio
    async def test_validate_pipeline_command_success(self, mock_state):
        """
        Test ValidatePipelineCommand successful execution.
        
        DNA Priority: MEDIUM (validation complexity)
        Entropy Target: Pipeline validation
        """
        test_metrics = TestMetrics(
            entropy=110.0,
            complexity=65,
            error_count=3,
            refactoring_vector=7.0,
            spectral_rank=0.5,
            topological_importance=0.4
        )
        
        async def test_logic():
            command = ValidatePipelineCommand(mock_state)
            
            # Set up active orchestrator with pipeline
            mock_orchestrator = test_framework.create_mock_orchestrator("test_plate")
            mock_orchestrator.pipeline_definition = [{"step": "test_step"}]
            mock_state.active_orchestrator = mock_orchestrator
            
            # Test execution
            await command.execute()
            
            # Verify success notification
            mock_state.notify.assert_called()
            success_calls = [call for call in mock_state.notify.call_args_list 
                           if call[0][0] == 'operation_status_changed' and 
                           call[0][1].get('status') == 'success']
            assert len(success_calls) > 0
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_validate_pipeline_command_success", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
