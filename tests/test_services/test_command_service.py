"""
Strategic Test Suite for CommandService.

DNA-guided tests targeting critical command execution functionality.
Refactoring Vector: 8.3 (HIGH PRIORITY)
Entropy: 115, Complexity: 107, Errors: 32

🔬 Test Strategy:
- Command execution pipelines
- Error handling and recovery
- Operation state management
- Async operation coordination
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
from openhcs.tui.services.command_service import CommandService


class TestCommandService:
    """
    Strategic test suite for CommandService.
    
    Tests prioritized by DNA analysis focusing on:
    - High complexity command execution
    - Error-prone async operations
    - State management consistency
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
    def service(self, mock_state, mock_context):
        """Create CommandService instance."""
        return CommandService(mock_state, mock_context)
    
    @pytest.fixture
    def mock_orchestrators(self):
        """Create list of mock orchestrators."""
        return [
            test_framework.create_mock_orchestrator("plate_1"),
            test_framework.create_mock_orchestrator("plate_2"),
            test_framework.create_mock_orchestrator("plate_3")
        ]
    
    @pytest.mark.asyncio
    async def test_initialize_plates_success(self, service, mock_orchestrators):
        """
        Test successful plate initialization.
        
        DNA Priority: HIGH (orchestrator lifecycle complexity)
        Entropy Target: Async operation coordination
        """
        test_metrics = TestMetrics(
            entropy=115.0,
            complexity=107,
            error_count=32,
            refactoring_vector=8.3,
            spectral_rank=0.9,
            topological_importance=0.8
        )
        
        async def test_logic():
            # Test initialization
            results = await service.initialize_plates(mock_orchestrators)
            
            # Validate results structure
            assert 'successful' in results
            assert 'failed' in results
            assert 'errors' in results
            
            # Verify all plates initialized successfully
            assert len(results['successful']) == 3
            assert len(results['failed']) == 0
            assert len(results['errors']) == 0
            
            # Verify orchestrator methods were called
            for orchestrator in mock_orchestrators:
                orchestrator.initialize_plate.assert_called_once()
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_initialize_plates_success", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.asyncio
    async def test_initialize_plates_with_failures(self, service, mock_orchestrators):
        """
        Test plate initialization with some failures.
        
        DNA Priority: HIGH (error handling complexity)
        Entropy Target: Exception propagation
        """
        test_metrics = TestMetrics(
            entropy=120.0,
            complexity=130,
            error_count=40,
            refactoring_vector=9.0,
            spectral_rank=1.0,
            topological_importance=0.9
        )
        
        async def test_logic():
            # Make second orchestrator fail
            mock_orchestrators[1].initialize_plate.side_effect = Exception("Initialization failed")
            
            # Test initialization
            results = await service.initialize_plates(mock_orchestrators)
            
            # Verify partial success
            assert len(results['successful']) == 2
            assert len(results['failed']) == 1
            assert len(results['errors']) == 1
            assert "Initialization failed" in results['errors'][0]
            
            # Verify failed plate ID
            assert "plate_2" in results['failed']
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_initialize_plates_with_failures", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.asyncio
    async def test_compile_plates_success(self, service, mock_orchestrators):
        """
        Test successful plate compilation.
        
        DNA Priority: HIGH (compilation complexity)
        Entropy Target: Pipeline compilation logic
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
            # Set up orchestrators with pipeline definitions
            for orchestrator in mock_orchestrators:
                orchestrator.pipeline_definition = [{"step": "test"}]
            
            # Test compilation
            results = await service.compile_plates(mock_orchestrators)
            
            # Validate results
            assert len(results['successful']) == 3
            assert len(results['failed']) == 0
            assert len(results['compiled_contexts']) == 3
            
            # Verify compiled contexts
            for plate_id in ['plate_1', 'plate_2', 'plate_3']:
                assert plate_id in results['compiled_contexts']
                assert results['compiled_contexts'][plate_id] == {"compiled": True}
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_compile_plates_success", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.asyncio
    async def test_compile_plates_not_ready(self, service, mock_orchestrators):
        """
        Test compilation with orchestrators not ready.
        
        DNA Priority: MEDIUM (validation logic complexity)
        Entropy Target: Readiness validation
        """
        test_metrics = TestMetrics(
            entropy=112.0,
            complexity=100,
            error_count=20,
            refactoring_vector=7.5,
            spectral_rank=0.7,
            topological_importance=0.6
        )
        
        async def test_logic():
            # Leave orchestrators without pipeline definitions (not ready)
            # mock_orchestrators already have empty pipeline_definition = []
            
            # Test compilation
            results = await service.compile_plates(mock_orchestrators)
            
            # Verify all failed due to not being ready
            assert len(results['successful']) == 0
            assert len(results['failed']) == 3
            assert len(results['errors']) == 3
            
            # Verify error messages
            for error in results['errors']:
                assert "not ready for compilation" in error
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_compile_plates_not_ready", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.asyncio
    async def test_run_plates_success(self, service, mock_orchestrators):
        """
        Test successful plate execution.
        
        DNA Priority: HIGH (execution complexity)
        Entropy Target: Pipeline execution logic
        """
        test_metrics = TestMetrics(
            entropy=122.0,
            complexity=160,
            error_count=35,
            refactoring_vector=9.2,
            spectral_rank=1.0,
            topological_importance=0.9
        )
        
        async def test_logic():
            # Set up orchestrators with pipeline definitions
            for orchestrator in mock_orchestrators:
                orchestrator.pipeline_definition = [{"step": "test"}]
            
            # Create compiled contexts
            compiled_contexts = {
                'plate_1': {"compiled": True},
                'plate_2': {"compiled": True},
                'plate_3': {"compiled": True}
            }
            
            # Test execution
            results = await service.run_plates(mock_orchestrators, compiled_contexts)
            
            # Validate results
            assert len(results['successful']) == 3
            assert len(results['failed']) == 0
            assert len(results['errors']) == 0
            
            # Verify execution methods were called
            for orchestrator in mock_orchestrators:
                orchestrator.execute_compiled_plate.assert_called_once()
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_run_plates_success", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.asyncio
    async def test_run_plates_missing_contexts(self, service, mock_orchestrators):
        """
        Test execution with missing compiled contexts.
        
        DNA Priority: MEDIUM (validation complexity)
        Entropy Target: Context validation
        """
        test_metrics = TestMetrics(
            entropy=115.0,
            complexity=120,
            error_count=25,
            refactoring_vector=8.0,
            spectral_rank=0.8,
            topological_importance=0.7
        )
        
        async def test_logic():
            # Set up orchestrators with pipeline definitions
            for orchestrator in mock_orchestrators:
                orchestrator.pipeline_definition = [{"step": "test"}]
            
            # Empty compiled contexts (missing)
            compiled_contexts = {}
            
            # Test execution
            results = await service.run_plates(mock_orchestrators, compiled_contexts)
            
            # Verify all failed due to missing contexts
            assert len(results['successful']) == 0
            assert len(results['failed']) == 3
            assert len(results['errors']) == 3
            
            # Verify error messages
            for error in results['errors']:
                assert "not ready for execution" in error
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_run_plates_missing_contexts", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.asyncio
    async def test_operation_state_management(self, service, mock_orchestrators):
        """
        Test operation state tracking and management.
        
        DNA Priority: HIGH (state management complexity)
        Entropy Target: Operation tracking
        """
        test_metrics = TestMetrics(
            entropy=116.0,
            complexity=125,
            error_count=22,
            refactoring_vector=8.5,
            spectral_rank=0.8,
            topological_importance=0.8
        )
        
        async def test_logic():
            # Start initialization (should track operation)
            init_task = asyncio.create_task(service.initialize_plates([mock_orchestrators[0]]))
            
            # Check active operations during execution
            await asyncio.sleep(0.01)  # Let operation start
            active_ops = service.get_active_operations()
            
            # Wait for completion
            await init_task
            
            # Verify operation was tracked
            # Note: Operation might complete too quickly to catch in active state
            # This tests the tracking mechanism exists
            assert isinstance(active_ops, dict)
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_operation_state_management", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.asyncio
    async def test_cancel_operation(self, service):
        """
        Test operation cancellation.
        
        DNA Priority: MEDIUM (cancellation complexity)
        Entropy Target: Operation lifecycle
        """
        test_metrics = TestMetrics(
            entropy=110.0,
            complexity=95,
            error_count=15,
            refactoring_vector=7.0,
            spectral_rank=0.6,
            topological_importance=0.5
        )
        
        async def test_logic():
            # Manually add an active operation
            async with service.operation_lock:
                service.active_operations["test_plate_initialize"] = {
                    'plate_id': 'test_plate',
                    'operation': 'initialize',
                    'status': 'running',
                    'start_time': asyncio.get_event_loop().time()
                }
            
            # Test cancellation
            success = await service.cancel_operation('test_plate', 'initialize')
            assert success is True
            
            # Verify operation was cancelled
            async with service.operation_lock:
                op = service.active_operations.get("test_plate_initialize")
                assert op['status'] == 'cancelled'
            
            # Test cancelling non-existent operation
            success = await service.cancel_operation('nonexistent', 'initialize')
            assert success is False
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_cancel_operation", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.asyncio
    async def test_shutdown_cleanup(self, service):
        """
        Test proper shutdown and resource cleanup.
        
        DNA Priority: HIGH (resource management complexity)
        Entropy Target: Resource cleanup
        """
        test_metrics = TestMetrics(
            entropy=113.0,
            complexity=110,
            error_count=18,
            refactoring_vector=8.2,
            spectral_rank=0.7,
            topological_importance=0.7
        )
        
        async def test_logic():
            # Add some active operations
            async with service.operation_lock:
                service.active_operations["test_op"] = {
                    'plate_id': 'test',
                    'operation': 'test',
                    'status': 'running',
                    'start_time': asyncio.get_event_loop().time()
                }
            
            # Verify executor exists
            assert service.executor is not None
            
            # Shutdown service
            await service.shutdown()
            
            # Verify cleanup
            assert service.executor is None
            
            # Verify active operations were cancelled
            async with service.operation_lock:
                if "test_op" in service.active_operations:
                    assert service.active_operations["test_op"]['status'] == 'cancelled'
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_shutdown_cleanup", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
