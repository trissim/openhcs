"""
Strategic Test Suite for LayoutController.

DNA-guided tests targeting critical layout management functionality.
Coverage Gap: E=0, C=17 (THIRD HIGHEST COVERAGE PRIORITY)
Entropy: 109, Complexity: 17, Errors: 0

🔬 Test Strategy:
- Layout structure management
- Component coordination
- Key binding management
- UI component lifecycle
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
from openhcs.tui.controllers.layout_controller import LayoutController


class TestLayoutController:
    """
    Strategic test suite for LayoutController.
    
    Tests prioritized by DNA analysis focusing on:
    - Layout coordination complexity
    - Component initialization patterns
    - UI structure management
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
    def mock_global_config(self):
        """Create mock GlobalPipelineConfig."""
        config = Mock()
        config.vfs = Mock()
        config.vfs.default_storage_backend = 'disk'
        return config
    
    @pytest.fixture
    def controller(self, mock_state, mock_context, mock_global_config):
        """Create LayoutController instance with mocked dependencies."""
        with patch('openhcs.tui.controllers.layout_controller.PlateManagerService') as mock_pm_service:
            with patch('openhcs.tui.controllers.layout_controller.MenuService') as mock_menu_service:
                with patch('openhcs.tui.controllers.layout_controller.PlateManagerController') as mock_pm_controller:
                    with patch('openhcs.tui.controllers.layout_controller.MenuController') as mock_menu_controller:
                        with patch('openhcs.tui.controllers.layout_controller.PlateManagerView') as mock_pm_view:
                            with patch('openhcs.tui.controllers.layout_controller.MenuView') as mock_menu_view:
                                # Setup service mocks
                                mock_pm_service_instance = Mock()
                                mock_pm_service.return_value = mock_pm_service_instance
                                
                                mock_menu_service_instance = Mock()
                                mock_menu_service.return_value = mock_menu_service_instance
                                
                                # Setup controller mocks
                                mock_pm_controller_instance = Mock()
                                mock_pm_controller.return_value = mock_pm_controller_instance
                                
                                mock_menu_controller_instance = Mock()
                                mock_menu_controller.return_value = mock_menu_controller_instance
                                
                                # Setup view mocks
                                mock_pm_view_instance = Mock()
                                mock_pm_view_instance.get_container = Mock()
                                mock_pm_view.return_value = mock_pm_view_instance
                                
                                mock_menu_view_instance = Mock()
                                mock_menu_view.return_value = mock_menu_view_instance
                                
                                # Create controller
                                controller = LayoutController(mock_state, mock_context, mock_global_config)
                                
                                # Assign mocked instances
                                controller.plate_manager_service = mock_pm_service_instance
                                controller.menu_service = mock_menu_service_instance
                                controller.plate_manager_controller = mock_pm_controller_instance
                                controller.menu_controller = mock_menu_controller_instance
                                controller.plate_manager_view = mock_pm_view_instance
                                controller.menu_view = mock_menu_view_instance
                                
                                return controller
    
    @pytest.mark.asyncio
    @pytest.mark.high_priority
    async def test_initialize_success(self, controller):
        """
        Test successful layout controller initialization.
        
        DNA Priority: HIGH (initialization complexity)
        Entropy Target: Component coordination
        """
        test_metrics = TestMetrics(
            entropy=109.0,
            complexity=17,
            error_count=0,
            refactoring_vector=8.0,
            spectral_rank=0.8,
            topological_importance=0.7
        )
        
        async def test_logic():
            with patch('openhcs.tui.controllers.layout_controller.storage_registry') as mock_storage_reg:
                with patch('openhcs.tui.controllers.layout_controller.PlateValidationService') as mock_validation:
                    mock_storage_reg.return_value = Mock()
                    mock_validation.return_value = Mock()
                    
                    # Test initialization
                    await controller.initialize()
                    
                    # Verify initialization state
                    assert controller.is_initialized is True
                    assert controller.root_container is not None
                    assert controller.key_bindings is not None
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_initialize_success", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.asyncio
    @pytest.mark.high_priority
    async def test_initialize_services(self, controller):
        """
        Test service initialization.
        
        DNA Priority: HIGH (service coordination complexity)
        Entropy Target: Service lifecycle management
        """
        test_metrics = TestMetrics(
            entropy=112.0,
            complexity=20,
            error_count=0,
            refactoring_vector=8.2,
            spectral_rank=0.8,
            topological_importance=0.7
        )
        
        async def test_logic():
            with patch('openhcs.tui.controllers.layout_controller.storage_registry') as mock_storage_reg:
                mock_storage_reg.return_value = Mock()
                
                # Test service initialization
                await controller._initialize_services()
                
                # Verify services were created
                assert controller.plate_manager_service is not None
                assert controller.menu_service is not None
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_initialize_services", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.asyncio
    @pytest.mark.high_priority
    async def test_initialize_controllers(self, controller):
        """
        Test controller initialization.
        
        DNA Priority: HIGH (controller coordination complexity)
        Entropy Target: Controller lifecycle management
        """
        test_metrics = TestMetrics(
            entropy=112.0,
            complexity=20,
            error_count=0,
            refactoring_vector=8.2,
            spectral_rank=0.8,
            topological_importance=0.7
        )
        
        async def test_logic():
            with patch('openhcs.tui.controllers.layout_controller.PlateValidationService') as mock_validation:
                mock_validation.return_value = Mock()
                
                # Test controller initialization
                await controller._initialize_controllers()
                
                # Verify controllers were created
                assert controller.plate_manager_controller is not None
                assert controller.menu_controller is not None
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_initialize_controllers", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.asyncio
    @pytest.mark.high_priority
    async def test_initialize_views(self, controller):
        """
        Test view initialization.
        
        DNA Priority: HIGH (view coordination complexity)
        Entropy Target: View lifecycle management
        """
        test_metrics = TestMetrics(
            entropy=110.0,
            complexity=18,
            error_count=0,
            refactoring_vector=8.0,
            spectral_rank=0.7,
            topological_importance=0.6
        )
        
        async def test_logic():
            # Test view initialization
            await controller._initialize_views()
            
            # Verify views were created
            assert controller.plate_manager_view is not None
            assert controller.menu_view is not None
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_initialize_views", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.high_priority
    def test_create_layout(self, controller):
        """
        Test layout creation.
        
        DNA Priority: HIGH (layout structure complexity)
        Entropy Target: UI structure creation
        """
        test_metrics = TestMetrics(
            entropy=115.0,
            complexity=25,
            error_count=0,
            refactoring_vector=8.5,
            spectral_rank=0.9,
            topological_importance=0.8
        )
        
        def test_logic():
            with patch('openhcs.tui.controllers.layout_controller.VSplit') as mock_vsplit:
                with patch('openhcs.tui.controllers.layout_controller.HSplit') as mock_hsplit:
                    with patch('openhcs.tui.controllers.layout_controller.Frame') as mock_frame:
                        with patch('openhcs.tui.controllers.layout_controller.FramedButton') as mock_button:
                            # Setup mocks
                            mock_vsplit.return_value = Mock()
                            mock_hsplit.return_value = Mock()
                            mock_frame.return_value = Mock()
                            mock_button.return_value = Mock()
                            
                            # Ensure views are initialized
                            controller.plate_manager_view = Mock()
                            controller.plate_manager_view.get_container = Mock()
                            
                            # Test layout creation
                            controller._create_layout()
                            
                            # Verify layout was created
                            assert controller.root_container is not None
                            
                            # Verify layout components were used
                            mock_vsplit.assert_called()
                            mock_hsplit.assert_called()
                            mock_frame.assert_called()
        
        result = test_framework.run_test_with_metrics(
            test_logic, "test_create_layout", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.high_priority
    def test_create_key_bindings(self, controller):
        """
        Test key binding creation.
        
        DNA Priority: HIGH (key binding complexity)
        Entropy Target: Input handling setup
        """
        test_metrics = TestMetrics(
            entropy=113.0,
            complexity=22,
            error_count=0,
            refactoring_vector=8.3,
            spectral_rank=0.8,
            topological_importance=0.7
        )
        
        def test_logic():
            with patch('openhcs.tui.controllers.layout_controller.KeyBindings') as mock_keybindings:
                mock_kb_instance = Mock()
                mock_kb_instance.add = Mock()
                mock_keybindings.return_value = mock_kb_instance
                
                # Test key binding creation
                controller._create_key_bindings()
                
                # Verify key bindings were created
                assert controller.key_bindings is not None
                
                # Verify key bindings were added
                assert mock_kb_instance.add.call_count >= 3  # At least 3 key bindings
        
        result = test_framework.run_test_with_metrics(
            test_logic, "test_create_key_bindings", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.asyncio
    @pytest.mark.medium_priority
    async def test_handle_settings(self, controller):
        """
        Test settings button handler.
        
        DNA Priority: MEDIUM (event handling complexity)
        Entropy Target: Event coordination
        """
        test_metrics = TestMetrics(
            entropy=108.0,
            complexity=15,
            error_count=0,
            refactoring_vector=7.5,
            spectral_rank=0.6,
            topological_importance=0.5
        )
        
        async def test_logic():
            # Test settings handler
            await controller._handle_settings()
            
            # Verify state notification was sent
            controller.state.notify.assert_called_once()
            call_args = controller.state.notify.call_args[0]
            assert call_args[0] == 'show_dialog_requested'
            assert call_args[1]['type'] == 'info'
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_handle_settings", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.asyncio
    @pytest.mark.medium_priority
    async def test_handle_help(self, controller):
        """
        Test help button handler.
        
        DNA Priority: MEDIUM (event handling complexity)
        Entropy Target: Event coordination
        """
        test_metrics = TestMetrics(
            entropy=108.0,
            complexity=15,
            error_count=0,
            refactoring_vector=7.5,
            spectral_rank=0.6,
            topological_importance=0.5
        )
        
        async def test_logic():
            # Test help handler
            await controller._handle_help()
            
            # Verify state notification was sent
            controller.state.notify.assert_called_once()
            call_args = controller.state.notify.call_args[0]
            assert call_args[0] == 'show_dialog_requested'
            assert call_args[1]['type'] == 'info'
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_handle_help", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.asyncio
    @pytest.mark.high_priority
    async def test_initialize_components(self, controller):
        """
        Test component initialization.
        
        DNA Priority: HIGH (component coordination complexity)
        Entropy Target: Component lifecycle
        """
        test_metrics = TestMetrics(
            entropy=114.0,
            complexity=23,
            error_count=0,
            refactoring_vector=8.4,
            spectral_rank=0.8,
            topological_importance=0.7
        )
        
        async def test_logic():
            # Mock plate manager controller
            controller.plate_manager_controller.refresh_plates = AsyncMock()
            
            # Test component initialization
            await controller.initialize_components()
            
            # Verify plate manager was refreshed
            controller.plate_manager_controller.refresh_plates.assert_called_once()
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_initialize_components", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.high_priority
    def test_get_root_container(self, controller):
        """
        Test getting root container.
        
        DNA Priority: MEDIUM (container access complexity)
        Entropy Target: Container management
        """
        test_metrics = TestMetrics(
            entropy=106.0,
            complexity=12,
            error_count=0,
            refactoring_vector=7.0,
            spectral_rank=0.5,
            topological_importance=0.4
        )
        
        def test_logic():
            # Set up root container
            mock_container = Mock()
            controller.root_container = mock_container
            
            # Test getting root container
            result = controller.get_root_container()
            
            # Verify correct container returned
            assert result == mock_container
        
        result = test_framework.run_test_with_metrics(
            test_logic, "test_get_root_container", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.high_priority
    def test_get_key_bindings(self, controller):
        """
        Test getting key bindings.
        
        DNA Priority: MEDIUM (key binding access complexity)
        Entropy Target: Input handling access
        """
        test_metrics = TestMetrics(
            entropy=106.0,
            complexity=12,
            error_count=0,
            refactoring_vector=7.0,
            spectral_rank=0.5,
            topological_importance=0.4
        )
        
        def test_logic():
            # Set up key bindings
            mock_bindings = Mock()
            controller.key_bindings = mock_bindings
            
            # Test getting key bindings
            result = controller.get_key_bindings()
            
            # Verify correct bindings returned
            assert result == mock_bindings
        
        result = test_framework.run_test_with_metrics(
            test_logic, "test_get_key_bindings", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
    
    @pytest.mark.asyncio
    @pytest.mark.high_priority
    async def test_shutdown(self, controller):
        """
        Test layout controller shutdown.
        
        DNA Priority: HIGH (resource cleanup complexity)
        Entropy Target: Resource management
        """
        test_metrics = TestMetrics(
            entropy=111.0,
            complexity=19,
            error_count=0,
            refactoring_vector=8.1,
            spectral_rank=0.7,
            topological_importance=0.6
        )
        
        async def test_logic():
            # Mock controller shutdown methods
            controller.plate_manager_controller.shutdown = AsyncMock()
            controller.menu_controller.shutdown = AsyncMock()
            
            # Test shutdown
            await controller.shutdown()
            
            # Verify controllers were shut down
            controller.plate_manager_controller.shutdown.assert_called_once()
            controller.menu_controller.shutdown.assert_called_once()
        
        result = await test_framework.run_test_with_metrics(
            test_logic, "test_shutdown", test_metrics
        )
        test_framework.test_results.append(result)
        assert result.passed
