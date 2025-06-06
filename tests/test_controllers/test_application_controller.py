"""
Strategic Test Suite for ApplicationController.

DNA-guided tests targeting critical application lifecycle functionality.
Refactoring Vector: 8.5 (HIGH PRIORITY)
Entropy: 113, Complexity: 100, Errors: 18

🔬 Test Strategy:
- Application lifecycle management
- Component initialization coordination
- Error handling and recovery
- Resource cleanup and shutdown
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
from openhcs.tui.controllers.application_controller import ApplicationController


class TestApplicationController:
    """
    Strategic test suite for ApplicationController.
    
    Tests prioritized by DNA analysis focusing on:
    - High complexity initialization sequences
    - Error-prone component coordination
    - Resource management consistency
    """
    
    @pytest.fixture
    def mock_context(self):
        """Create mock ProcessingContext."""
        return test_framework.create_mock_context()
    
    @pytest.fixture
    def mock_state(self):
        """Create mock TUIState."""
        return test_framework.create_mock_state()
    
    @pytest.fixture
    def mock_global_config(self):
        """Create mock GlobalPipelineConfig."""
        config = Mock()
        config.vfs = Mock()
        config.vfs.default_storage_backend = 'disk'
        return config
    
    @pytest.fixture
    def controller(self, mock_context, mock_state, mock_global_config):
        """Create ApplicationController instance."""
        with patch('openhcs.tui.controllers.application_controller.LayoutController') as mock_layout:
            with patch('openhcs.tui.controllers.application_controller.DialogService') as mock_dialog:
                mock_layout_instance = Mock()
                mock_layout_instance.initialize = AsyncMock()
                mock_layout_instance.get_root_container = Mock()
                mock_layout_instance.get_key_bindings = Mock()
                mock_layout_instance.initialize_components = AsyncMock()
                mock_layout_instance.shutdown = AsyncMock()
                mock_layout.return_value = mock_layout_instance
                
                mock_dialog_instance = Mock()
                mock_dialog_instance.shutdown = AsyncMock()
                mock_dialog.return_value = mock_dialog_instance
                
                controller = ApplicationController(mock_context, mock_state, mock_global_config)
                controller.layout_controller = mock_layout_instance
                controller.dialog_service = mock_dialog_instance
                
                return controller
    
    @pytest.mark.asyncio
    async def test_initialization_success(self, controller):
        """
        Test successful application initialization.
        
        DNA Priority: HIGH (initialization complexity)
        Entropy Target: Component coordination
        """
        test_metrics = TestMetrics(
            entropy=113.0,
            complexity=100,
            error_count=18,
            refactoring_vector=8.5,
            spectral_rank=0.9,
            topological_importance=0.8
        )
        
        async def test_logic():
            with patch('openhcs.tui.controllers.application_controller.Application') as mock_app_class:
                mock_app = Mock()
                mock_app_class.return_value = mock_app
                
                # Test initialization
                await controller.initialize()
                
                # Verify initialization state
                assert controller.is_initialized is True
                assert controller.application is not None
                
                # Verify layout controller was initialized
                controller.layout_controller.initialize.assert_called_once()
                controller.layout_controller.initialize_components.assert_called_once()
                
                # Verify application was created
                mock_app_class.assert_called_once()
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_initialization_success", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.asyncio
    async def test_initialization_failure_recovery(self, controller):
        """
        Test initialization failure handling.
        
        DNA Priority: HIGH (error handling complexity)
        Entropy Target: Exception propagation
        """
        test_metrics = TestMetrics(
            entropy=120.0,
            complexity=130,
            error_count=25,
            refactoring_vector=9.0,
            spectral_rank=1.0,
            topological_importance=0.9
        )
        
        async def test_logic():
            # Make layout controller initialization fail
            controller.layout_controller.initialize.side_effect = Exception("Layout init failed")
            
            # Test initialization failure
            with pytest.raises(Exception, match="Layout init failed"):
                await controller.initialize()
            
            # Verify state remains uninitialized
            assert controller.is_initialized is False
            assert controller.application is None
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_initialization_failure_recovery", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.asyncio
    async def test_run_application_lifecycle(self, controller):
        """
        Test complete application run lifecycle.
        
        DNA Priority: HIGH (lifecycle complexity)
        Entropy Target: Application coordination
        """
        test_metrics = TestMetrics(
            entropy=118.0,
            complexity=140,
            error_count=22,
            refactoring_vector=8.8,
            spectral_rank=0.9,
            topological_importance=0.8
        )
        
        async def test_logic():
            with patch('openhcs.tui.controllers.application_controller.Application') as mock_app_class:
                mock_app = Mock()
                mock_app.run_async = AsyncMock()
                mock_app_class.return_value = mock_app
                
                # Mock run_async to complete immediately
                async def mock_run():
                    pass
                mock_app.run_async.side_effect = mock_run
                
                # Test run
                await controller.run()
                
                # Verify initialization occurred
                assert controller.is_initialized is True
                
                # Verify application was run
                mock_app.run_async.assert_called_once()
                
                # Verify shutdown was called
                controller.layout_controller.shutdown.assert_called_once()
                controller.dialog_service.shutdown.assert_called_once()
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_run_application_lifecycle", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.asyncio
    async def test_exit_request_handling(self, controller):
        """
        Test exit request handling with confirmation.
        
        DNA Priority: MEDIUM (dialog coordination complexity)
        Entropy Target: User interaction flow
        """
        test_metrics = TestMetrics(
            entropy=110.0,
            complexity=90,
            error_count=12,
            refactoring_vector=7.5,
            spectral_rank=0.7,
            topological_importance=0.6
        )
        
        async def test_logic():
            # Mock dialog service confirmation
            controller.dialog_service.show_confirmation_dialog = AsyncMock(return_value=True)
            
            # Mock application
            mock_app = Mock()
            mock_app.exit = Mock()
            controller.application = mock_app
            
            # Test exit request
            await controller._handle_exit_request({'force': False})
            
            # Verify confirmation dialog was shown
            controller.dialog_service.show_confirmation_dialog.assert_called_once()
            
            # Verify application exit was called
            mock_app.exit.assert_called_once()
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_exit_request_handling", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.asyncio
    async def test_exit_request_cancelled(self, controller):
        """
        Test exit request cancellation.
        
        DNA Priority: MEDIUM (user interaction complexity)
        Entropy Target: Cancellation flow
        """
        test_metrics = TestMetrics(
            entropy=108.0,
            complexity=85,
            error_count=10,
            refactoring_vector=7.0,
            spectral_rank=0.6,
            topological_importance=0.5
        )
        
        async def test_logic():
            # Mock dialog service confirmation (user cancels)
            controller.dialog_service.show_confirmation_dialog = AsyncMock(return_value=False)
            
            # Mock application
            mock_app = Mock()
            mock_app.exit = Mock()
            controller.application = mock_app
            
            # Test exit request
            await controller._handle_exit_request({'force': False})
            
            # Verify confirmation dialog was shown
            controller.dialog_service.show_confirmation_dialog.assert_called_once()
            
            # Verify application exit was NOT called
            mock_app.exit.assert_not_called()
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_exit_request_cancelled", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.asyncio
    async def test_error_handling(self, controller):
        """
        Test application error handling.
        
        DNA Priority: HIGH (error handling complexity)
        Entropy Target: Error propagation
        """
        test_metrics = TestMetrics(
            entropy=115.0,
            complexity=120,
            error_count=20,
            refactoring_vector=8.3,
            spectral_rank=0.8,
            topological_importance=0.7
        )
        
        async def test_logic():
            # Mock dialog service error dialog
            controller.dialog_service.show_error_dialog = AsyncMock()
            
            # Test error handling
            error_data = {
                'message': 'Test error message',
                'source': 'TestComponent',
                'details': 'Detailed error information'
            }
            
            await controller._handle_error(error_data)
            
            # Verify error dialog was shown
            controller.dialog_service.show_error_dialog.assert_called_once_with(
                title="Error - TestComponent",
                message="Test error message",
                details="Detailed error information"
            )
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_error_handling", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.asyncio
    async def test_dialog_request_handling(self, controller):
        """
        Test dialog request coordination.
        
        DNA Priority: MEDIUM (dialog coordination complexity)
        Entropy Target: Dialog management
        """
        test_metrics = TestMetrics(
            entropy=112.0,
            complexity=95,
            error_count=15,
            refactoring_vector=7.8,
            spectral_rank=0.7,
            topological_importance=0.6
        )
        
        async def test_logic():
            # Mock dialog service methods
            controller.dialog_service.show_error_dialog = AsyncMock()
            controller.dialog_service.show_info_dialog = AsyncMock()
            controller.dialog_service.show_confirmation_dialog = AsyncMock(return_value=True)
            
            # Test error dialog request
            await controller._handle_show_dialog_request({
                'type': 'error',
                'data': {
                    'title': 'Test Error',
                    'message': 'Error message',
                    'details': 'Error details'
                }
            })
            
            # Test info dialog request
            await controller._handle_show_dialog_request({
                'type': 'info',
                'data': {
                    'title': 'Test Info',
                    'message': 'Info message'
                }
            })
            
            # Test confirmation dialog request
            callback_called = False
            async def test_callback(result):
                nonlocal callback_called
                callback_called = True
                assert result is True
            
            await controller._handle_show_dialog_request({
                'type': 'confirmation',
                'data': {
                    'title': 'Test Confirm',
                    'message': 'Confirm message',
                    'default_yes': False
                },
                'callback': test_callback
            })
            
            # Verify dialogs were called
            controller.dialog_service.show_error_dialog.assert_called_once()
            controller.dialog_service.show_info_dialog.assert_called_once()
            controller.dialog_service.show_confirmation_dialog.assert_called_once()
            assert callback_called is True
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_dialog_request_handling", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.asyncio
    async def test_shutdown_cleanup(self, controller):
        """
        Test proper shutdown and cleanup.
        
        DNA Priority: HIGH (resource management complexity)
        Entropy Target: Resource cleanup
        """
        test_metrics = TestMetrics(
            entropy=114.0,
            complexity=105,
            error_count=16,
            refactoring_vector=8.1,
            spectral_rank=0.8,
            topological_importance=0.7
        )
        
        async def test_logic():
            # Set up initialized state
            controller.is_initialized = True
            
            # Test shutdown
            await controller.shutdown()
            
            # Verify shutdown was called on components
            controller.layout_controller.shutdown.assert_called_once()
            controller.dialog_service.shutdown.assert_called_once()
            
            # Verify shutdown state
            assert controller.is_shutting_down is True
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_shutdown_cleanup", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
