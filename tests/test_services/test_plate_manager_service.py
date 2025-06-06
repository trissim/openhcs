"""
Strategic Test Suite for PlateManagerService.

DNA-guided tests targeting critical functionality based on mathematical analysis.
Refactoring Vector: 10.1 (HIGHEST PRIORITY)
Entropy: 112, Complexity: 135, Errors: 22

🔬 Test Strategy:
- Async operations and thread safety
- Orchestrator lifecycle management
- Error handling and validation
- Resource cleanup and shutdown
"""
import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.test_framework import (
    test_framework, TestMetrics, TestResult
)
from openhcs.tui.services.plate_manager_service import PlateManagerService


class TestPlateManagerService:
    """
    Strategic test suite for PlateManagerService.
    
    Tests are prioritized based on DNA analysis:
    - High entropy operations (async/thread safety)
    - High complexity methods (orchestrator management)
    - Error-prone areas (file I/O, validation)
    """
    
    @pytest.fixture
    def mock_context(self):
        """Create mock ProcessingContext."""
        return test_framework.create_mock_context()
    
    @pytest.fixture
    def mock_storage_registry(self):
        """Create mock storage registry."""
        return Mock()
    
    @pytest.fixture
    def service(self, mock_context, mock_storage_registry):
        """Create PlateManagerService instance."""
        return PlateManagerService(mock_context, mock_storage_registry)
    
    @pytest.fixture
    def mock_global_config(self):
        """Create mock global configuration."""
        config = Mock()
        config.vfs = Mock()
        config.vfs.default_storage_backend = 'disk'
        return config
    
    @pytest.mark.asyncio
    async def test_add_plate_success(self, service, mock_global_config):
        """
        Test successful plate addition.
        
        DNA Priority: HIGH (orchestrator creation complexity)
        Entropy Target: Orchestrator initialization
        """
        test_metrics = TestMetrics(
            entropy=112.0,
            complexity=135,
            error_count=22,
            refactoring_vector=10.1,
            spectral_rank=1.0,
            topological_importance=0.9
        )
        
        async def test_logic():
            # Mock PipelineOrchestrator creation
            with patch('openhcs.tui.services.plate_manager_service.PipelineOrchestrator') as mock_orchestrator_class:
                mock_orchestrator = test_framework.create_mock_orchestrator("test_plate")
                mock_orchestrator_class.return_value = mock_orchestrator
                
                # Test plate addition
                result = await service.add_plate("/test/path", mock_global_config)
                
                # Validate result structure
                assert isinstance(result, dict)
                assert 'id' in result
                assert 'name' in result
                assert 'path' in result
                assert 'status' in result
                assert 'orchestrator' in result
                assert 'backend' in result
                
                # Validate values
                assert result['path'] == "/test/path"
                assert result['status'] == 'not_initialized'
                assert result['backend'] == 'disk'
                assert result['orchestrator'] == mock_orchestrator
                
                # Verify orchestrator was created correctly
                mock_orchestrator_class.assert_called_once_with(
                    plate_path="/test/path",
                    config=mock_global_config,
                    storage_registry=service.registry
                )
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_add_plate_success", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.asyncio
    async def test_add_plate_duplicate_error(self, service, mock_global_config):
        """
        Test duplicate plate addition error handling.
        
        DNA Priority: HIGH (error handling complexity)
        Entropy Target: Validation logic
        """
        test_metrics = TestMetrics(
            entropy=115.0,
            complexity=140,
            error_count=25,
            refactoring_vector=10.1,
            spectral_rank=1.0,
            topological_importance=0.8
        )
        
        async def test_logic():
            with patch('openhcs.tui.services.plate_manager_service.PipelineOrchestrator') as mock_orchestrator_class:
                mock_orchestrator = test_framework.create_mock_orchestrator("test_plate")
                mock_orchestrator_class.return_value = mock_orchestrator
                
                # Add plate first time
                await service.add_plate("/test/path", mock_global_config)
                
                # Try to add same plate again - should raise ValueError
                with pytest.raises(ValueError, match="already exists"):
                    await service.add_plate("/test/path", mock_global_config)
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_add_plate_duplicate_error", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.asyncio
    async def test_remove_plates_success(self, service, mock_global_config):
        """
        Test successful plate removal.
        
        DNA Priority: MEDIUM (list manipulation complexity)
        Entropy Target: Thread-safe operations
        """
        test_metrics = TestMetrics(
            entropy=110.0,
            complexity=120,
            error_count=15,
            refactoring_vector=8.5,
            spectral_rank=0.8,
            topological_importance=0.7
        )
        
        async def test_logic():
            with patch('openhcs.tui.services.plate_manager_service.PipelineOrchestrator') as mock_orchestrator_class:
                mock_orchestrator = test_framework.create_mock_orchestrator("test_plate")
                mock_orchestrator_class.return_value = mock_orchestrator
                
                # Add plates
                plate1 = await service.add_plate("/test/path1", mock_global_config)
                plate2 = await service.add_plate("/test/path2", mock_global_config)
                
                # Remove one plate
                removed_count = await service.remove_plates([plate1['id']])
                
                assert removed_count == 1
                
                # Verify plate was removed
                remaining_plates = await service.get_plates()
                assert len(remaining_plates) == 1
                assert remaining_plates[0]['id'] == plate2['id']
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_remove_plates_success", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.asyncio
    async def test_update_plate_status(self, service, mock_global_config):
        """
        Test plate status updates.
        
        DNA Priority: MEDIUM (state management complexity)
        Entropy Target: Status synchronization
        """
        test_metrics = TestMetrics(
            entropy=108.0,
            complexity=100,
            error_count=10,
            refactoring_vector=7.0,
            spectral_rank=0.6,
            topological_importance=0.6
        )
        
        async def test_logic():
            with patch('openhcs.tui.services.plate_manager_service.PipelineOrchestrator') as mock_orchestrator_class:
                mock_orchestrator = test_framework.create_mock_orchestrator("test_plate")
                mock_orchestrator_class.return_value = mock_orchestrator
                
                # Add plate
                plate = await service.add_plate("/test/path", mock_global_config)
                plate_id = plate['id']
                
                # Update status to ready
                success = await service.update_plate_status(plate_id, 'ready')
                assert success is True
                
                # Verify status was updated
                updated_plate = await service.get_plate_by_id(plate_id)
                assert updated_plate['status'] == 'ready'
                
                # Update status to error with message
                success = await service.update_plate_status(plate_id, 'error', 'Test error')
                assert success is True
                
                # Verify error status and message
                updated_plate = await service.get_plate_by_id(plate_id)
                assert updated_plate['status'] == 'error'
                assert updated_plate['error_message'] == 'Test error'
                
                # Test non-existent plate
                success = await service.update_plate_status('nonexistent', 'ready')
                assert success is False
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_update_plate_status", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.asyncio
    async def test_refresh_plates_from_directory(self, service):
        """
        Test directory-based plate refresh.
        
        DNA Priority: HIGH (file I/O complexity)
        Entropy Target: FileManager integration
        """
        test_metrics = TestMetrics(
            entropy=118.0,
            complexity=150,
            error_count=20,
            refactoring_vector=9.5,
            spectral_rank=0.9,
            topological_importance=0.8
        )
        
        async def test_logic():
            # Mock filemanager responses
            service.filemanager.exists.return_value = True
            service.filemanager.list_dir.return_value = ["/test/plate1", "/test/plate2"]
            service.filemanager.is_dir.return_value = True
            
            # Test refresh
            plates = await service.refresh_plates_from_directory("/test/output")
            
            # Verify results
            assert len(plates) == 2
            assert plates[0]['id'] == 'plate1'
            assert plates[1]['id'] == 'plate2'
            
            # Verify filemanager calls
            service.filemanager.exists.assert_called_once_with("/test/output", backend='disk')
            service.filemanager.list_dir.assert_called_once_with("/test/output", backend='disk')
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_refresh_plates_from_directory", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.asyncio
    async def test_thread_safety_concurrent_operations(self, service, mock_global_config):
        """
        Test thread safety with concurrent operations.
        
        DNA Priority: CRITICAL (async safety complexity)
        Entropy Target: Race condition prevention
        """
        test_metrics = TestMetrics(
            entropy=125.0,
            complexity=180,
            error_count=30,
            refactoring_vector=10.1,
            spectral_rank=1.0,
            topological_importance=1.0
        )
        
        async def test_logic():
            with patch('openhcs.tui.services.plate_manager_service.PipelineOrchestrator') as mock_orchestrator_class:
                mock_orchestrator_class.side_effect = lambda **kwargs: test_framework.create_mock_orchestrator(
                    f"plate_{kwargs['plate_path'].split('/')[-1]}"
                )
                
                # Create multiple concurrent operations
                tasks = []
                for i in range(10):
                    task = service.add_plate(f"/test/path{i}", mock_global_config)
                    tasks.append(task)
                
                # Execute concurrently
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Verify all operations completed successfully
                successful_results = [r for r in results if isinstance(r, dict)]
                assert len(successful_results) == 10
                
                # Verify thread safety - all plates should be in the list
                all_plates = await service.get_plates()
                assert len(all_plates) == 10
                
                # Verify unique IDs
                plate_ids = [p['id'] for p in all_plates]
                assert len(set(plate_ids)) == 10  # All unique
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_thread_safety_concurrent_operations", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.asyncio
    async def test_resource_cleanup_shutdown(self, service):
        """
        Test proper resource cleanup during shutdown.
        
        DNA Priority: HIGH (resource management complexity)
        Entropy Target: Memory and thread cleanup
        """
        test_metrics = TestMetrics(
            entropy=115.0,
            complexity=130,
            error_count=18,
            refactoring_vector=8.8,
            spectral_rank=0.8,
            topological_importance=0.7
        )
        
        async def test_logic():
            # Verify executor exists
            assert service.io_executor is not None
            
            # Shutdown service
            await service.shutdown()
            
            # Verify executor was cleaned up
            assert service.io_executor is None
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_resource_cleanup_shutdown", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
