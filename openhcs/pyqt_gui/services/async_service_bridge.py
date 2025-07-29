"""
Async Service Bridge

Bridges async OpenHCS services to PyQt6 threading model,
converting async/await patterns to Qt signals/slots.
"""

import logging
from typing import Any, Callable, Optional
import asyncio

from PyQt6.QtCore import QObject, QThread, pyqtSignal

logger = logging.getLogger(__name__)


class AsyncServiceBridge(QObject):
    """
    Bridge for converting async service operations to Qt threading model.
    
    Handles the conversion of async/await patterns used in OpenHCS services
    to Qt's signal/slot threading model.
    """
    
    # Signals for async operation results
    operation_completed = pyqtSignal(object)  # result
    operation_failed = pyqtSignal(str)  # error_message
    operation_progress = pyqtSignal(int)  # progress_percentage
    
    def __init__(self, service_adapter):
        """
        Initialize async service bridge.
        
        Args:
            service_adapter: PyQtServiceAdapter instance
        """
        super().__init__()
        self.service_adapter = service_adapter
        self.active_threads = []
    
    def execute_async_operation(self, async_func: Callable, *args, **kwargs) -> None:
        """
        Execute async operation in Qt thread.
        
        Args:
            async_func: Async function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
        """
        thread = AsyncOperationThread(async_func, *args, **kwargs)
        
        # Connect thread signals
        thread.result_ready.connect(self.operation_completed.emit)
        thread.error_occurred.connect(self.operation_failed.emit)
        thread.finished.connect(lambda: self._cleanup_thread(thread))
        
        # Track active thread
        self.active_threads.append(thread)
        
        # Start thread
        thread.start()
    
    def _cleanup_thread(self, thread: QThread) -> None:
        """
        Clean up completed thread.
        
        Args:
            thread: Completed thread to clean up
        """
        if thread in self.active_threads:
            self.active_threads.remove(thread)
        thread.deleteLater()
    
    def wait_for_all_operations(self, timeout_ms: int = 30000) -> bool:
        """
        Wait for all active async operations to complete.
        
        Args:
            timeout_ms: Timeout in milliseconds
            
        Returns:
            True if all operations completed, False if timeout
        """
        for thread in self.active_threads[:]:  # Copy list to avoid modification during iteration
            if not thread.wait(timeout_ms):
                logger.warning(f"Async operation timed out after {timeout_ms}ms")
                return False
        return True


class AsyncOperationThread(QThread):
    """
    Thread for executing async operations.
    
    Converts async/await patterns to Qt threading model.
    """
    
    result_ready = pyqtSignal(object)
    error_occurred = pyqtSignal(str)
    progress_updated = pyqtSignal(int)
    
    def __init__(self, async_func: Callable, *args, **kwargs):
        super().__init__()
        self.async_func = async_func
        self.args = args
        self.kwargs = kwargs
    
    def run(self):
        """Execute async function in thread with new event loop."""
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Execute async function
                result = loop.run_until_complete(
                    self.async_func(*self.args, **self.kwargs)
                )
                self.result_ready.emit(result)
                
            finally:
                # Clean up event loop
                loop.close()
                
        except Exception as e:
            logger.error(f"Async operation failed: {e}")
            self.error_occurred.emit(str(e))


class PatternFileServiceBridge:
    """
    Bridge for PatternFileService async operations.
    
    Adapts PatternFileService to work with PyQt6 service adapter.
    """
    
    def __init__(self, service_adapter):
        """
        Initialize pattern file service bridge.
        
        Args:
            service_adapter: PyQtServiceAdapter instance
        """
        self.service_adapter = service_adapter
        self.async_bridge = AsyncServiceBridge(service_adapter)
        
        # Import and adapt the original service
        from openhcs.textual_tui.services.pattern_file_service import PatternFileService
        self.original_service = PatternFileService(service_adapter)
    
    def load_pattern_from_file(self, file_path, callback: Callable = None):
        """
        Load pattern from file using Qt threading.
        
        Args:
            file_path: Path to pattern file
            callback: Optional callback for result
        """
        if callback:
            self.async_bridge.operation_completed.connect(callback)
            self.async_bridge.operation_failed.connect(
                lambda error: self.service_adapter.show_error_dialog(f"Load failed: {error}")
            )
        
        self.async_bridge.execute_async_operation(
            self.original_service.load_pattern_from_file,
            file_path
        )
    
    def save_pattern_to_file(self, pattern, file_path, callback: Callable = None):
        """
        Save pattern to file using Qt threading.
        
        Args:
            pattern: Pattern to save
            file_path: Path to save to
            callback: Optional callback for completion
        """
        if callback:
            self.async_bridge.operation_completed.connect(callback)
            self.async_bridge.operation_failed.connect(
                lambda error: self.service_adapter.show_error_dialog(f"Save failed: {error}")
            )
        
        self.async_bridge.execute_async_operation(
            self.original_service.save_pattern_to_file,
            pattern,
            file_path
        )


class ExternalEditorServiceBridge:
    """
    Bridge for ExternalEditorService with PyQt6 integration.
    
    Replaces prompt_toolkit dependencies with Qt equivalents.
    """
    
    def __init__(self, service_adapter):
        """
        Initialize external editor service bridge.
        
        Args:
            service_adapter: PyQtServiceAdapter instance
        """
        self.service_adapter = service_adapter
        self.async_bridge = AsyncServiceBridge(service_adapter)
    
    def edit_pattern_in_external_editor(self, initial_content: str, callback: Callable = None):
        """
        Edit pattern in external editor using Qt process management.
        
        Args:
            initial_content: Initial content for editor
            callback: Optional callback for result
        """
        if callback:
            self.async_bridge.operation_completed.connect(callback)
            self.async_bridge.operation_failed.connect(
                lambda error: self.service_adapter.show_error_dialog(f"Editor failed: {error}")
            )
        
        # Use Qt-based external editor implementation
        self.async_bridge.execute_async_operation(
            self._qt_external_editor_operation,
            initial_content
        )
    
    async def _qt_external_editor_operation(self, initial_content: str):
        """
        Qt-based external editor operation.
        
        Args:
            initial_content: Initial content for editor
            
        Returns:
            Tuple of (success, pattern, error_message)
        """
        import tempfile
        import os
        from pathlib import Path
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, mode='w+', suffix='.py', encoding='utf-8') as tmp_file:
            tmp_file.write(initial_content)
            tmp_file_path = Path(tmp_file.name)
        
        try:
            # Get editor command
            editor = os.environ.get('EDITOR', 'vim')
            command = f"{editor} {tmp_file_path}"
            
            # Use service adapter to run command
            success = self.service_adapter.run_system_command(command, wait_for_finish=True)
            
            if not success:
                return False, None, "Editor command failed"
            
            # Read modified content
            with open(tmp_file_path, "r", encoding="utf-8") as f:
                modified_content = f.read()
            
            # Validate content (simplified validation)
            try:
                import ast
                tree = ast.parse(modified_content)
                
                # Extract pattern assignment
                for node in tree.body:
                    if isinstance(node, ast.Assign) and len(node.targets) == 1:
                        target = node.targets[0]
                        if isinstance(target, ast.Name) and target.id == 'pattern':
                            pattern = ast.literal_eval(ast.unparse(node.value))
                            return True, pattern, None
                
                return False, None, "No valid pattern assignment found"
                
            except Exception as e:
                return False, None, f"Pattern validation failed: {e}"
                
        finally:
            # Clean up temporary file
            if tmp_file_path.exists():
                tmp_file_path.unlink()
