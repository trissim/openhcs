"""
Simple, direct tests for TUI services.
No frameworks, no bullshit, just testing what actually exists.
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from openhcs.tui.services.menu_service import MenuService
from openhcs.tui.services.dialog_service import DialogService
from openhcs.tui.services.command_service import CommandService
from openhcs.tui.services.pattern_editing_service import PatternEditingService
from openhcs.tui.services.plate_manager_service import PlateManagerService


class TestMenuService:
    """Test MenuService - what actually works."""
    
    def setup_method(self):
        self.state = Mock()
        self.state.active_orchestrator = None
        self.state.is_compiled = False
        self.state.is_running = False
        self.state.notify = AsyncMock()
        self.context = Mock()
        self.service = MenuService(self.state, self.context)
    
    @pytest.mark.asyncio
    async def test_new_pipeline(self):
        result = await self.service.execute_command('new_pipeline')
        assert result is True
        self.state.notify.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_save_pipeline_no_orchestrator(self):
        result = await self.service.execute_command('save_pipeline')
        assert result is True
        # Should send error notification
        args = self.state.notify.call_args[0]
        assert args[0] == 'error'
        assert 'No active pipeline' in args[1]['message']
    
    @pytest.mark.asyncio
    async def test_save_pipeline_with_orchestrator(self):
        orchestrator = Mock()
        orchestrator.save_pipeline = AsyncMock(return_value=True)
        self.state.active_orchestrator = orchestrator
        
        result = await self.service.execute_command('save_pipeline')
        assert result is True
        orchestrator.save_pipeline.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_unknown_command(self):
        result = await self.service.execute_command('fake_command')
        assert result is False
        args = self.state.notify.call_args[0]
        assert args[0] == 'error'
        assert 'Unknown command' in args[1]['message']
    
    def test_command_availability(self):
        # No orchestrator
        assert self.service.is_command_enabled('save_pipeline') is False
        
        # With orchestrator
        self.state.active_orchestrator = Mock()
        assert self.service.is_command_enabled('save_pipeline') is True
        
        # Run pipeline logic
        self.state.is_compiled = False
        assert self.service.is_command_enabled('run_pipeline') is False
        
        self.state.is_compiled = True
        self.state.is_running = False
        assert self.service.is_command_enabled('run_pipeline') is True
        
        self.state.is_running = True
        assert self.service.is_command_enabled('run_pipeline') is False


class TestDialogService:
    """Test DialogService - what actually works."""
    
    def setup_method(self):
        self.state = Mock()
        self.state.notify = AsyncMock()
        self.service = DialogService(self.state)
    
    @pytest.mark.asyncio
    async def test_show_info_dialog(self):
        dialog_id = await self.service.show_info_dialog("Test", "Message")
        assert dialog_id is not None
        self.state.notify.assert_called()
    
    @pytest.mark.asyncio
    async def test_show_error_dialog(self):
        dialog_id = await self.service.show_error_dialog("Error", "Error message")
        assert dialog_id is not None
        self.state.notify.assert_called()
    
    @pytest.mark.asyncio
    async def test_show_confirmation_dialog(self):
        dialog_id = await self.service.show_confirmation_dialog("Confirm", "Are you sure?")
        assert dialog_id is not None
        self.state.notify.assert_called()


class TestCommandService:
    """Test CommandService - what actually works."""
    
    def setup_method(self):
        self.state = Mock()
        self.context = Mock()
        self.service = CommandService(self.state, self.context)
    
    def test_register_command(self):
        def test_cmd():
            return "test"
        
        self.service.register_command("test", test_cmd)
        assert "test" in self.service.commands
    
    def test_execute_registered_command(self):
        def test_cmd():
            return "success"
        
        self.service.register_command("test", test_cmd)
        result = self.service.execute_command("test")
        assert result == "success"
    
    def test_execute_unknown_command(self):
        result = self.service.execute_command("unknown")
        assert result is None


class TestPatternEditingService:
    """Test PatternEditingService - what actually works."""
    
    def setup_method(self):
        self.service = PatternEditingService()
    
    def test_clone_pattern_list(self):
        pattern = [{'func': 'test', 'kwargs': {'a': 1}}]
        cloned = self.service.clone_pattern(pattern)
        
        assert cloned == pattern
        assert cloned is not pattern
        assert cloned[0] is not pattern[0]
    
    def test_clone_pattern_dict(self):
        pattern = {'key1': [{'func': 'test'}]}
        cloned = self.service.clone_pattern(pattern)
        
        assert cloned == pattern
        assert cloned is not pattern
        assert cloned['key1'] is not pattern['key1']
    
    def test_is_dict_pattern(self):
        dict_pattern = {'key': []}
        list_pattern = [{'func': 'test'}]
        
        assert self.service.is_dict_pattern(dict_pattern) is True
        assert self.service.is_dict_pattern(list_pattern) is False
    
    def test_get_pattern_keys(self):
        dict_pattern = {'key1': [], 'key2': []}
        list_pattern = []
        
        keys = self.service.get_pattern_keys(dict_pattern)
        assert set(keys) == {'key1', 'key2'}
        
        keys = self.service.get_pattern_keys(list_pattern)
        assert keys == []
    
    def test_add_function_to_pattern(self):
        pattern = []
        func_data = {'func': 'test_func', 'kwargs': {}}
        
        self.service.add_function_to_pattern(pattern, func_data)
        assert len(pattern) == 1
        assert pattern[0] == func_data


class TestPlateManagerService:
    """Test PlateManagerService - what actually works."""
    
    def setup_method(self):
        self.state = Mock()
        self.context = Mock()
        self.context.filemanager = Mock()
        self.global_config = Mock()
        self.service = PlateManagerService(self.state, self.context, self.global_config)
    
    @pytest.mark.asyncio
    async def test_add_plate(self):
        self.context.filemanager.exists.return_value = True
        self.context.filemanager.is_dir.return_value = True
        
        plate_id = await self.service.add_plate("/test/path", "Test Plate")
        assert plate_id is not None
        assert plate_id in self.service.plates
    
    @pytest.mark.asyncio
    async def test_add_plate_invalid_path(self):
        self.context.filemanager.exists.return_value = False
        
        with pytest.raises(ValueError, match="does not exist"):
            await self.service.add_plate("/invalid/path", "Test Plate")
    
    @pytest.mark.asyncio
    async def test_remove_plate(self):
        # Add a plate first
        self.context.filemanager.exists.return_value = True
        self.context.filemanager.is_dir.return_value = True
        plate_id = await self.service.add_plate("/test/path", "Test Plate")
        
        # Remove it
        result = await self.service.remove_plate(plate_id)
        assert result is True
        assert plate_id not in self.service.plates
    
    def test_get_plates(self):
        plates = self.service.get_plates()
        assert isinstance(plates, list)
    
    def test_get_plate(self):
        # Non-existent plate
        plate = self.service.get_plate("fake_id")
        assert plate is None


# Simple coverage test
def test_import_all_services():
    """Test that all services can be imported without errors."""
    from openhcs.tui.services.menu_service import MenuService
    from openhcs.tui.services.dialog_service import DialogService
    from openhcs.tui.services.command_service import CommandService
    from openhcs.tui.services.pattern_editing_service import PatternEditingService
    from openhcs.tui.services.plate_manager_service import PlateManagerService
    
    # Just verify they can be imported
    assert MenuService is not None
    assert DialogService is not None
    assert CommandService is not None
    assert PatternEditingService is not None
    assert PlateManagerService is not None
