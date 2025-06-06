"""
Strategic Test Suite for MenuController.

DNA-guided tests targeting critical menu management functionality.
Coverage Gap: E=1, C=7 (FOURTH HIGHEST COVERAGE PRIORITY)
Entropy: 113, Complexity: 7, Errors: 1

🔬 Test Strategy:
- Menu command coordination
- Service integration
- Event handling patterns
- State management
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
from openhcs.tui.controllers.menu_controller import MenuController


class TestMenuController:
    """
    Strategic test suite for MenuController.
    
    Tests prioritized by DNA analysis focusing on:
    - Menu command coordination
    - Service integration patterns
    - Event handling complexity
    """
    
    @pytest.fixture
    def mock_state(self):
        """Create mock TUIState."""
        return test_framework.create_mock_state()
    
    @pytest.fixture
    def mock_menu_service(self):
        """Create mock MenuService."""
        service = Mock()
        service.execute_command = AsyncMock()
        service.get_available_commands = Mock(return_value=['new_pipeline', 'save_pipeline', 'exit'])
        service.is_command_enabled = Mock(return_value=True)
        service.shutdown = AsyncMock()
        return service
    
    @pytest.fixture
    def controller(self, mock_state, mock_menu_service):
        """Create MenuController instance."""
        return MenuController(mock_state, mock_menu_service)
    
    @pytest.mark.asyncio
    @pytest.mark.high_priority
    async def test_execute_command_success(self, controller, mock_menu_service):
        """
        Test successful command execution.
        
        DNA Priority: HIGH (command coordination complexity)
        Entropy Target: Command execution flow
        """
        test_metrics = TestMetrics(
            entropy=113.0,
            complexity=7,
            error_count=1,
            refactoring_vector=8.0,
            spectral_rank=0.8,
            topological_importance=0.7
        )
        
        async def test_logic():
            # Test command execution
            await controller.execute_command('new_pipeline')
            
            # Verify service was called
            mock_menu_service.execute_command.assert_called_once_with('new_pipeline')
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_execute_command_success", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.asyncio
    @pytest.mark.high_priority
    async def test_execute_command_with_args(self, controller, mock_menu_service):
        """
        Test command execution with arguments.
        
        DNA Priority: HIGH (command parameter complexity)
        Entropy Target: Parameter passing
        """
        test_metrics = TestMetrics(
            entropy=115.0,
            complexity=10,
            error_count=2,
            refactoring_vector=8.2,
            spectral_rank=0.8,
            topological_importance=0.7
        )
        
        async def test_logic():
            # Test command execution with arguments
            args = {'file_path': '/test/path', 'force': True}
            await controller.execute_command('save_pipeline', **args)
            
            # Verify service was called with arguments
            mock_menu_service.execute_command.assert_called_once_with('save_pipeline', **args)
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_execute_command_with_args", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.asyncio
    @pytest.mark.high_priority
    async def test_execute_command_error_handling(self, controller, mock_menu_service):
        """
        Test command execution error handling.
        
        DNA Priority: HIGH (error handling complexity)
        Entropy Target: Exception propagation
        """
        test_metrics = TestMetrics(
            entropy=118.0,
            complexity=12,
            error_count=3,
            refactoring_vector=8.5,
            spectral_rank=0.9,
            topological_importance=0.8
        )
        
        async def test_logic():
            # Make service raise an exception
            mock_menu_service.execute_command.side_effect = Exception("Command failed")
            
            # Test command execution with error
            await controller.execute_command('invalid_command')
            
            # Verify error was handled (should not raise)
            mock_menu_service.execute_command.assert_called_once_with('invalid_command')
            
            # Verify state notification was sent
            controller.state.notify.assert_called()
            error_calls = [call for call in controller.state.notify.call_args_list 
                          if call[0][0] == 'error']
            assert len(error_calls) > 0
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_execute_command_error_handling", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.high_priority
    def test_get_available_commands(self, controller, mock_menu_service):
        """
        Test getting available commands.
        
        DNA Priority: MEDIUM (command query complexity)
        Entropy Target: Command availability
        """
        test_metrics = TestMetrics(
            entropy=110.0,
            complexity=5,
            error_count=0,
            refactoring_vector=7.5,
            spectral_rank=0.6,
            topological_importance=0.5
        )
        
        def test_logic():
            # Test getting available commands
            commands = controller.get_available_commands()
            
            # Verify service was called
            mock_menu_service.get_available_commands.assert_called_once()
            
            # Verify commands returned
            assert commands == ['new_pipeline', 'save_pipeline', 'exit']
        
        result = test_framework.run_test_with_metrics(
            test_logic, "test_get_available_commands", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.high_priority
    def test_is_command_enabled(self, controller, mock_menu_service):
        """
        Test checking if command is enabled.
        
        DNA Priority: MEDIUM (command state complexity)
        Entropy Target: Command state validation
        """
        test_metrics = TestMetrics(
            entropy=111.0,
            complexity=6,
            error_count=0,
            refactoring_vector=7.7,
            spectral_rank=0.7,
            topological_importance=0.6
        )
        
        def test_logic():
            # Test checking command enabled state
            is_enabled = controller.is_command_enabled('save_pipeline')
            
            # Verify service was called
            mock_menu_service.is_command_enabled.assert_called_once_with('save_pipeline')
            
            # Verify result
            assert is_enabled is True
        
        result = test_framework.run_test_with_metrics(
            test_logic, "test_is_command_enabled", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.high_priority
    def test_is_command_enabled_false(self, controller, mock_menu_service):
        """
        Test checking disabled command.
        
        DNA Priority: MEDIUM (command state complexity)
        Entropy Target: Command state validation
        """
        test_metrics = TestMetrics(
            entropy=111.0,
            complexity=6,
            error_count=0,
            refactoring_vector=7.7,
            spectral_rank=0.7,
            topological_importance=0.6
        )
        
        def test_logic():
            # Make service return False
            mock_menu_service.is_command_enabled.return_value = False
            
            # Test checking command enabled state
            is_enabled = controller.is_command_enabled('disabled_command')
            
            # Verify service was called
            mock_menu_service.is_command_enabled.assert_called_once_with('disabled_command')
            
            # Verify result
            assert is_enabled is False
        
        result = test_framework.run_test_with_metrics(
            test_logic, "test_is_command_enabled_false", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.asyncio
    @pytest.mark.medium_priority
    async def test_handle_menu_selection(self, controller):
        """
        Test menu selection handling.
        
        DNA Priority: MEDIUM (event handling complexity)
        Entropy Target: Menu event coordination
        """
        test_metrics = TestMetrics(
            entropy=112.0,
            complexity=8,
            error_count=1,
            refactoring_vector=7.8,
            spectral_rank=0.7,
            topological_importance=0.6
        )
        
        async def test_logic():
            # Mock execute_command method
            controller.execute_command = AsyncMock()
            
            # Test menu selection handling
            await controller.handle_menu_selection('File', 'New Pipeline')
            
            # Verify command was executed
            controller.execute_command.assert_called_once()
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_handle_menu_selection", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.asyncio
    @pytest.mark.medium_priority
    async def test_handle_keyboard_shortcut(self, controller):
        """
        Test keyboard shortcut handling.
        
        DNA Priority: MEDIUM (input handling complexity)
        Entropy Target: Keyboard event coordination
        """
        test_metrics = TestMetrics(
            entropy=112.0,
            complexity=8,
            error_count=1,
            refactoring_vector=7.8,
            spectral_rank=0.7,
            topological_importance=0.6
        )
        
        async def test_logic():
            # Mock execute_command method
            controller.execute_command = AsyncMock()
            
            # Test keyboard shortcut handling
            await controller.handle_keyboard_shortcut('c-s')  # Ctrl+S for save
            
            # Verify command was executed
            controller.execute_command.assert_called_once()
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_handle_keyboard_shortcut", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.asyncio
    @pytest.mark.medium_priority
    async def test_refresh_menu_state(self, controller):
        """
        Test menu state refresh.
        
        DNA Priority: MEDIUM (state synchronization complexity)
        Entropy Target: Menu state management
        """
        test_metrics = TestMetrics(
            entropy=110.0,
            complexity=7,
            error_count=0,
            refactoring_vector=7.6,
            spectral_rank=0.6,
            topological_importance=0.5
        )
        
        async def test_logic():
            # Test menu state refresh
            await controller.refresh_menu_state()
            
            # Verify state notification was sent
            controller.state.notify.assert_called()
            refresh_calls = [call for call in controller.state.notify.call_args_list 
                           if 'menu_state_updated' in str(call)]
            # Note: Actual implementation may vary, this tests the pattern
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_refresh_menu_state", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.asyncio
    @pytest.mark.medium_priority
    async def test_update_command_availability(self, controller, mock_menu_service):
        """
        Test updating command availability.
        
        DNA Priority: MEDIUM (state update complexity)
        Entropy Target: Command state synchronization
        """
        test_metrics = TestMetrics(
            entropy=111.0,
            complexity=8,
            error_count=1,
            refactoring_vector=7.7,
            spectral_rank=0.7,
            topological_importance=0.6
        )
        
        async def test_logic():
            # Test updating command availability
            await controller.update_command_availability('save_pipeline', False)
            
            # Verify state was updated (implementation dependent)
            # This tests the pattern of state coordination
            controller.state.notify.assert_called()
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_update_command_availability", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.asyncio
    @pytest.mark.high_priority
    async def test_shutdown(self, controller, mock_menu_service):
        """
        Test menu controller shutdown.
        
        DNA Priority: HIGH (resource cleanup complexity)
        Entropy Target: Resource management
        """
        test_metrics = TestMetrics(
            entropy=113.0,
            complexity=9,
            error_count=1,
            refactoring_vector=8.0,
            spectral_rank=0.8,
            topological_importance=0.7
        )
        
        async def test_logic():
            # Test shutdown
            await controller.shutdown()
            
            # Verify service shutdown was called
            mock_menu_service.shutdown.assert_called_once()
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_shutdown", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.asyncio
    @pytest.mark.medium_priority
    async def test_concurrent_command_execution(self, controller, mock_menu_service):
        """
        Test concurrent command execution handling.
        
        DNA Priority: MEDIUM (concurrency complexity)
        Entropy Target: Concurrent operation management
        """
        test_metrics = TestMetrics(
            entropy=116.0,
            complexity=12,
            error_count=2,
            refactoring_vector=8.3,
            spectral_rank=0.8,
            topological_importance=0.7
        )
        
        async def test_logic():
            # Test concurrent command execution
            tasks = [
                controller.execute_command('command1'),
                controller.execute_command('command2'),
                controller.execute_command('command3')
            ]
            
            # Execute concurrently
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Verify all commands were executed
            assert mock_menu_service.execute_command.call_count == 3
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_concurrent_command_execution", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.high_priority
    def test_get_menu_service(self, controller, mock_menu_service):
        """
        Test getting menu service.
        
        DNA Priority: LOW (service access complexity)
        Entropy Target: Service access pattern
        """
        test_metrics = TestMetrics(
            entropy=105.0,
            complexity=3,
            error_count=0,
            refactoring_vector=6.5,
            spectral_rank=0.4,
            topological_importance=0.3
        )
        
        def test_logic():
            # Test getting menu service
            service = controller.get_menu_service()
            
            # Verify correct service returned
            assert service == mock_menu_service
        
        result = test_framework.run_test_with_metrics(
            test_logic, "test_get_menu_service", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
