"""
Test only MenuService - the one we know works.
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from openhcs.tui.services.menu_service import MenuService


class TestMenuServiceWorking:
    """Test only the MenuService functionality that we know works."""
    
    def setup_method(self):
        self.state = Mock()
        self.state.active_orchestrator = None
        self.state.is_compiled = False
        self.state.is_running = False
        self.state.notify = AsyncMock()
        self.context = Mock()
        self.service = MenuService(self.state, self.context)
    
    @pytest.mark.asyncio
    async def test_new_pipeline_command(self):
        """Test new_pipeline command - we know this works."""
        result = await self.service.execute_command('new_pipeline')
        assert result is True
        self.state.notify.assert_called_once()
        
        # Check the actual notification
        args = self.state.notify.call_args[0]
        assert args[0] == 'operation_status_changed'
        assert args[1]['operation'] == 'new_pipeline'
    
    @pytest.mark.asyncio
    async def test_save_pipeline_no_orchestrator(self):
        """Test save_pipeline without orchestrator."""
        result = await self.service.execute_command('save_pipeline')
        assert result is True
        
        # Should send error notification
        args = self.state.notify.call_args[0]
        assert args[0] == 'error'
        assert 'No active pipeline' in args[1]['message']
    
    @pytest.mark.asyncio
    async def test_save_pipeline_with_orchestrator(self):
        """Test save_pipeline with orchestrator."""
        orchestrator = Mock()
        orchestrator.save_pipeline = AsyncMock(return_value=True)
        self.state.active_orchestrator = orchestrator
        
        result = await self.service.execute_command('save_pipeline')
        assert result is True
        orchestrator.save_pipeline.assert_called_once()
        
        # Should send success notification
        args = self.state.notify.call_args[0]
        assert args[0] == 'operation_status_changed'
        assert args[1]['status'] == 'success'
    
    @pytest.mark.asyncio
    async def test_exit_command(self):
        """Test exit command."""
        result = await self.service.execute_command('exit')
        assert result is True
        
        args = self.state.notify.call_args[0]
        assert args[0] == 'exit_requested'
    
    @pytest.mark.asyncio
    async def test_validate_pipeline_no_orchestrator(self):
        """Test validate_pipeline without orchestrator."""
        result = await self.service.execute_command('validate_pipeline')
        assert result is True
        
        # Should send error
        args = self.state.notify.call_args[0]
        assert args[0] == 'error'
        assert 'No active pipeline' in args[1]['message']
    
    @pytest.mark.asyncio
    async def test_validate_pipeline_with_orchestrator(self):
        """Test validate_pipeline with orchestrator."""
        orchestrator = Mock()
        orchestrator.validate_pipeline = AsyncMock(return_value=True)
        self.state.active_orchestrator = orchestrator
        
        result = await self.service.execute_command('validate_pipeline')
        assert result is True
        orchestrator.validate_pipeline.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_unknown_command(self):
        """Test unknown command handling."""
        result = await self.service.execute_command('fake_command')
        assert result is False
        
        args = self.state.notify.call_args[0]
        assert args[0] == 'error'
        assert 'Unknown command' in args[1]['message']
    
    def test_save_pipeline_availability_no_orchestrator(self):
        """Test save_pipeline availability without orchestrator."""
        self.state.active_orchestrator = None
        assert self.service.is_command_enabled('save_pipeline') is False
    
    def test_save_pipeline_availability_with_orchestrator(self):
        """Test save_pipeline availability with orchestrator."""
        self.state.active_orchestrator = Mock()
        assert self.service.is_command_enabled('save_pipeline') is True
    
    def test_run_pipeline_availability_not_compiled(self):
        """Test run_pipeline availability when not compiled."""
        self.state.active_orchestrator = Mock()
        self.state.is_compiled = False
        self.state.is_running = False
        assert self.service.is_command_enabled('run_pipeline') is False
    
    def test_run_pipeline_availability_compiled_not_running(self):
        """Test run_pipeline availability when compiled but not running."""
        self.state.active_orchestrator = Mock()
        self.state.is_compiled = True
        self.state.is_running = False
        assert self.service.is_command_enabled('run_pipeline') is True
    
    def test_run_pipeline_availability_already_running(self):
        """Test run_pipeline availability when already running."""
        self.state.active_orchestrator = Mock()
        self.state.is_compiled = True
        self.state.is_running = True
        assert self.service.is_command_enabled('run_pipeline') is False
    
    def test_stop_pipeline_availability_not_running(self):
        """Test stop_pipeline availability when not running."""
        self.state.is_running = False
        assert self.service.is_command_enabled('stop_pipeline') is False
    
    def test_stop_pipeline_availability_running(self):
        """Test stop_pipeline availability when running."""
        self.state.is_running = True
        assert self.service.is_command_enabled('stop_pipeline') is True
    
    def test_validate_pipeline_availability_no_orchestrator(self):
        """Test validate_pipeline availability without orchestrator."""
        self.state.active_orchestrator = None
        assert self.service.is_command_enabled('validate_pipeline') is False
    
    def test_validate_pipeline_availability_with_orchestrator(self):
        """Test validate_pipeline availability with orchestrator."""
        self.state.active_orchestrator = Mock()
        assert self.service.is_command_enabled('validate_pipeline') is True
    
    def test_new_pipeline_always_enabled(self):
        """Test that new_pipeline is always enabled."""
        assert self.service.is_command_enabled('new_pipeline') is True
    
    def test_exit_always_enabled(self):
        """Test that exit is always enabled."""
        assert self.service.is_command_enabled('exit') is True
    
    def test_unknown_command_disabled(self):
        """Test that unknown commands are disabled."""
        assert self.service.is_command_enabled('fake_command') is False


# Test what commands are actually implemented
def test_implemented_commands():
    """Document what commands are actually implemented in MenuService."""
    state = Mock()
    state.notify = AsyncMock()
    context = Mock()
    service = MenuService(state, context)
    
    # These should return True or False (implemented)
    implemented_commands = [
        'new_pipeline',
        'save_pipeline', 
        'exit',
        'validate_pipeline',
        'run_pipeline',
        'stop_pipeline'
    ]
    
    for cmd in implemented_commands:
        # Should not raise an exception
        try:
            enabled = service.is_command_enabled(cmd)
            assert isinstance(enabled, bool)
        except Exception as e:
            pytest.fail(f"Command {cmd} availability check failed: {e}")
    
    # These should return False (not implemented)
    unimplemented_commands = [
        'load_pipeline',
        'compile_pipeline',
        'fake_command'
    ]
    
    for cmd in unimplemented_commands:
        enabled = service.is_command_enabled(cmd)
        assert enabled is False
