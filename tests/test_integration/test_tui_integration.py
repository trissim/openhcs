"""
Strategic Integration Test Suite for TUI Components.

DNA-guided integration tests targeting critical component interactions.
Based on spectral analysis and topological connectivity patterns.

🔬 Test Strategy:
- Service-Controller integration
- Controller-View coordination
- Command-Service pipelines
- Error propagation chains
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


class TestTUIIntegration:
    """
    Strategic integration test suite for TUI components.
    
    Tests prioritized by DNA spectral analysis focusing on:
    - High connectivity component interactions
    - Critical data flow paths
    - Error propagation patterns
    """
    
    @pytest.fixture
    def mock_dependencies(self):
        """Create comprehensive mock dependencies."""
        deps = {}
        
        # Mock core dependencies
        deps['context'] = test_framework.create_mock_context()
        deps['state'] = test_framework.create_mock_state()
        deps['global_config'] = Mock()
        deps['global_config'].vfs = Mock()
        deps['global_config'].vfs.default_storage_backend = 'disk'
        
        # Mock storage registry
        deps['storage_registry'] = Mock()
        
        return deps
    
    @pytest.mark.asyncio
    async def test_plate_manager_service_controller_integration(self, mock_dependencies):
        """
        Test PlateManagerService-Controller integration.
        
        DNA Priority: CRITICAL (highest connectivity)
        Entropy Target: Service-Controller data flow
        """
        test_metrics = TestMetrics(
            entropy=125.0,
            complexity=180,
            error_count=35,
            refactoring_vector=10.1,
            spectral_rank=1.0,
            topological_importance=1.0
        )
        
        async def test_logic():
            with patch('openhcs.tui.services.plate_manager_service.PipelineOrchestrator') as mock_orch_class:
                with patch('openhcs.tui.controllers.plate_manager_controller.PlateManagerController') as mock_controller_class:
                    # Import after patching to avoid import errors
                    from openhcs.tui.services.plate_manager_service import PlateManagerService
                    
                    # Create service
                    service = PlateManagerService(
                        mock_dependencies['context'],
                        mock_dependencies['storage_registry']
                    )
                    
                    # Mock orchestrator creation
                    mock_orchestrator = test_framework.create_mock_orchestrator("test_plate")
                    mock_orch_class.return_value = mock_orchestrator
                    
                    # Test service-controller integration
                    plate = await service.add_plate("/test/path", mock_dependencies['global_config'])
                    
                    # Verify plate structure
                    assert 'id' in plate
                    assert 'orchestrator' in plate
                    assert plate['orchestrator'] == mock_orchestrator
                    
                    # Test status update
                    success = await service.update_plate_status(plate['id'], 'ready')
                    assert success is True
                    
                    # Verify updated plate
                    updated_plate = await service.get_plate_by_id(plate['id'])
                    assert updated_plate['status'] == 'ready'
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_plate_manager_service_controller_integration", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.asyncio
    async def test_command_service_integration_pipeline(self, mock_dependencies):
        """
        Test Command-Service integration pipeline.
        
        DNA Priority: HIGH (command execution complexity)
        Entropy Target: Command-Service coordination
        """
        test_metrics = TestMetrics(
            entropy=120.0,
            complexity=150,
            error_count=28,
            refactoring_vector=9.5,
            spectral_rank=0.9,
            topological_importance=0.9
        )
        
        async def test_logic():
            from openhcs.tui.services.command_service import CommandService
            from openhcs.tui.commands.simplified_commands import InitializePlatesCommand
            
            # Create service and command
            command_service = CommandService(mock_dependencies['state'], mock_dependencies['context'])
            command = InitializePlatesCommand(mock_dependencies['state'], service=command_service)
            
            # Create mock orchestrators
            orchestrators = [
                test_framework.create_mock_orchestrator("plate_1"),
                test_framework.create_mock_orchestrator("plate_2")
            ]
            
            # Test command execution through service
            await command.execute(orchestrators_to_initialize=orchestrators)
            
            # Verify service operations were tracked
            active_ops = command_service.get_active_operations()
            assert isinstance(active_ops, dict)
            
            # Verify state notifications
            mock_dependencies['state'].notify.assert_called()
            
            # Cleanup
            await command_service.shutdown()
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_command_service_integration_pipeline", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.asyncio
    async def test_application_controller_lifecycle_integration(self, mock_dependencies):
        """
        Test ApplicationController full lifecycle integration.
        
        DNA Priority: HIGH (application coordination complexity)
        Entropy Target: Component lifecycle coordination
        """
        test_metrics = TestMetrics(
            entropy=118.0,
            complexity=140,
            error_count=25,
            refactoring_vector=8.8,
            spectral_rank=0.9,
            topological_importance=0.8
        )
        
        async def test_logic():
            with patch('openhcs.tui.controllers.application_controller.LayoutController') as mock_layout:
                with patch('openhcs.tui.controllers.application_controller.DialogService') as mock_dialog:
                    from openhcs.tui.controllers.application_controller import ApplicationController
                    
                    # Mock layout controller
                    mock_layout_instance = Mock()
                    mock_layout_instance.initialize = AsyncMock()
                    mock_layout_instance.get_root_container = Mock()
                    mock_layout_instance.get_key_bindings = Mock()
                    mock_layout_instance.initialize_components = AsyncMock()
                    mock_layout_instance.shutdown = AsyncMock()
                    mock_layout.return_value = mock_layout_instance
                    
                    # Mock dialog service
                    mock_dialog_instance = Mock()
                    mock_dialog_instance.shutdown = AsyncMock()
                    mock_dialog.return_value = mock_dialog_instance
                    
                    # Create controller
                    controller = ApplicationController(
                        mock_dependencies['context'],
                        mock_dependencies['state'],
                        mock_dependencies['global_config']
                    )
                    
                    # Test initialization
                    with patch('openhcs.tui.controllers.application_controller.Application') as mock_app_class:
                        mock_app = Mock()
                        mock_app_class.return_value = mock_app
                        
                        await controller.initialize()
                        
                        # Verify initialization chain
                        assert controller.is_initialized is True
                        mock_layout_instance.initialize.assert_called_once()
                        mock_layout_instance.initialize_components.assert_called_once()
                    
                    # Test shutdown
                    await controller.shutdown()
                    
                    # Verify shutdown chain
                    mock_layout_instance.shutdown.assert_called_once()
                    mock_dialog_instance.shutdown.assert_called_once()
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_application_controller_lifecycle_integration", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.asyncio
    async def test_error_propagation_chain(self, mock_dependencies):
        """
        Test error propagation through component chain.
        
        DNA Priority: HIGH (error handling complexity)
        Entropy Target: Error propagation patterns
        """
        test_metrics = TestMetrics(
            entropy=122.0,
            complexity=160,
            error_count=30,
            refactoring_vector=9.2,
            spectral_rank=0.9,
            topological_importance=0.8
        )
        
        async def test_logic():
            from openhcs.tui.services.command_service import CommandService
            from openhcs.tui.commands.simplified_commands import InitializePlatesCommand
            
            # Create service and command
            command_service = CommandService(mock_dependencies['state'], mock_dependencies['context'])
            command = InitializePlatesCommand(mock_dependencies['state'], service=command_service)
            
            # Create orchestrator that will fail
            failing_orchestrator = test_framework.create_mock_orchestrator("failing_plate")
            failing_orchestrator.initialize_plate.side_effect = Exception("Initialization failed")
            
            # Test error propagation
            await command.execute(orchestrators_to_initialize=[failing_orchestrator])
            
            # Verify error was handled and propagated
            mock_dependencies['state'].notify.assert_called()
            
            # Check for error notifications
            error_calls = [call for call in mock_dependencies['state'].notify.call_args_list 
                          if call[0][0] == 'error']
            assert len(error_calls) > 0
            
            # Cleanup
            await command_service.shutdown()
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_error_propagation_chain", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.asyncio
    async def test_concurrent_operations_integration(self, mock_dependencies):
        """
        Test concurrent operations across multiple components.
        
        DNA Priority: CRITICAL (concurrency complexity)
        Entropy Target: Thread safety and coordination
        """
        test_metrics = TestMetrics(
            entropy=130.0,
            complexity=200,
            error_count=40,
            refactoring_vector=10.1,
            spectral_rank=1.0,
            topological_importance=1.0
        )
        
        async def test_logic():
            with patch('openhcs.tui.services.plate_manager_service.PipelineOrchestrator') as mock_orch_class:
                from openhcs.tui.services.plate_manager_service import PlateManagerService
                from openhcs.tui.services.command_service import CommandService
                
                # Create services
                plate_service = PlateManagerService(
                    mock_dependencies['context'],
                    mock_dependencies['storage_registry']
                )
                command_service = CommandService(mock_dependencies['state'], mock_dependencies['context'])
                
                # Mock orchestrator creation
                mock_orch_class.side_effect = lambda **kwargs: test_framework.create_mock_orchestrator(
                    f"plate_{kwargs['plate_path'].split('/')[-1]}"
                )
                
                # Create concurrent operations
                tasks = []
                
                # Concurrent plate additions
                for i in range(5):
                    task = plate_service.add_plate(f"/test/path{i}", mock_dependencies['global_config'])
                    tasks.append(task)
                
                # Execute concurrently
                plates = await asyncio.gather(*tasks)
                
                # Verify all operations completed
                assert len(plates) == 5
                
                # Verify thread safety - all plates in service
                all_plates = await plate_service.get_plates()
                assert len(all_plates) == 5
                
                # Test concurrent command operations
                orchestrators = [plate['orchestrator'] for plate in plates]
                
                # Concurrent initialization
                init_results = await command_service.initialize_plates(orchestrators)
                assert len(init_results['successful']) == 5
                
                # Cleanup
                await plate_service.shutdown()
                await command_service.shutdown()
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_concurrent_operations_integration", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.asyncio
    async def test_pattern_editing_service_integration(self, mock_dependencies):
        """
        Test PatternEditingService integration with controllers.
        
        DNA Priority: MEDIUM (pattern editing complexity)
        Entropy Target: Pattern manipulation coordination
        """
        test_metrics = TestMetrics(
            entropy=115.0,
            complexity=120,
            error_count=20,
            refactoring_vector=8.0,
            spectral_rank=0.7,
            topological_importance=0.6
        )
        
        async def test_logic():
            from openhcs.tui.services.pattern_editing_service import PatternEditingService
            from openhcs.tui.controllers.pattern_editor_controller import PatternEditorController
            
            # Create service and controller
            pattern_service = PatternEditingService(mock_dependencies['state'])
            controller = PatternEditorController(
                mock_dependencies['state'],
                initial_pattern=[],
                change_callback=None
            )
            
            # Test pattern operations
            pattern = controller.get_pattern()
            assert isinstance(pattern, list)
            
            # Test pattern conversion
            success = await controller.convert_to_dict_pattern()
            assert success is True
            
            # Verify pattern was converted
            pattern = controller.get_pattern()
            assert isinstance(pattern, dict)
            
            # Test adding pattern key
            success = await controller.add_pattern_key("test_key")
            assert success is True
            
            # Verify key was added
            keys = controller.get_pattern_keys()
            assert "test_key" in keys
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_pattern_editing_service_integration", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
