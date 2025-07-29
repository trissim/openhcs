"""
PyQt6 Service Adapter

Bridges OpenHCS services to PyQt6 context, replacing prompt_toolkit dependencies
with Qt equivalents while preserving all business logic.
"""

import logging
from typing import Any, Optional
from pathlib import Path

from PyQt6.QtWidgets import QMessageBox, QFileDialog, QApplication, QWidget
from PyQt6.QtCore import QProcess, QThread, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtCore import QUrl

from openhcs.pyqt_gui.utils.path_cache import PathCacheKey, get_cached_dialog_path, cache_dialog_path

logger = logging.getLogger(__name__)


class PyQtServiceAdapter:
    """
    Adapter to bridge OpenHCS services to PyQt6 context.
    
    Replaces prompt_toolkit dependencies (dialogs, system commands, etc.)
    with PyQt6 equivalents while maintaining the same interface for services.
    """
    
    def __init__(self, main_window: QWidget):
        """
        Initialize the service adapter.

        Args:
            main_window: Main PyQt6 window for dialog parenting
        """
        self.main_window = main_window
        self.app = QApplication.instance()
        logger.debug("PyQt6 service adapter initialized")

    def execute_async_operation(self, async_func, *args, **kwargs):
        """
        Execute async operation using ThreadPoolExecutor (simpler and more reliable).

        Args:
            async_func: Async function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
        """
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        def run_async_in_thread():
            """Run async function in thread with its own event loop."""
            try:
                # Create new event loop for this thread (like TUI executor)
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                # Run the async function
                result = loop.run_until_complete(async_func(*args, **kwargs))

                # Clean up
                loop.close()

                return result

            except Exception as e:
                logger.error(f"Async operation failed: {e}", exc_info=True)
                raise

        # Use ThreadPoolExecutor (simpler than Qt threading)
        if not hasattr(self, '_thread_pool'):
            self._thread_pool = ThreadPoolExecutor(max_workers=4)

        # Submit to thread pool (non-blocking like TUI executor)
        future = self._thread_pool.submit(run_async_in_thread)

    def show_dialog(self, content: str, title: str = "OpenHCS") -> bool:
        """
        Replace prompt_toolkit dialogs with QMessageBox.
        
        Args:
            content: Dialog content text
            title: Dialog title
            
        Returns:
            True if user clicked OK, False otherwise
        """
        msg = QMessageBox(self.main_window)
        msg.setWindowTitle(title)
        msg.setText(content)
        msg.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
        msg.setDefaultButton(QMessageBox.StandardButton.Ok)
        
        result = msg.exec()
        return result == QMessageBox.StandardButton.Ok
    
    def show_error_dialog(self, error_message: str, title: str = "Error") -> None:
        """
        Show error dialog with error icon.
        
        Args:
            error_message: Error message to display
            title: Dialog title
        """
        msg = QMessageBox(self.main_window)
        msg.setWindowTitle(title)
        msg.setText(error_message)
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()
    
    def show_info_dialog(self, info_message: str, title: str = "Information") -> None:
        """
        Show information dialog.
        
        Args:
            info_message: Information message to display
            title: Dialog title
        """
        msg = QMessageBox(self.main_window)
        msg.setWindowTitle(title)
        msg.setText(info_message)
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()

    def show_cached_file_dialog(
        self,
        cache_key: PathCacheKey,
        title: str = "Select File",
        file_filter: str = "All Files (*)",
        mode: str = "open",
        fallback_path: Optional[Path] = None
    ) -> Optional[Path]:
        """
        Show file dialog with path caching (mirrors Textual TUI pattern).

        Args:
            cache_key: Cache key for remembering last used path
            title: Dialog title
            file_filter: File filter string (e.g., "Pipeline Files (*.pipeline)")
            mode: "open" or "save"
            fallback_path: Fallback path if no cached path exists

        Returns:
            Selected file path or None if cancelled
        """
        # Get cached initial directory
        initial_dir = str(get_cached_dialog_path(cache_key, fallback_path))

        try:
            if mode == "save":
                file_path, _ = QFileDialog.getSaveFileName(
                    self.main_window,
                    title,
                    initial_dir,
                    file_filter
                )
            else:  # mode == "open"
                file_path, _ = QFileDialog.getOpenFileName(
                    self.main_window,
                    title,
                    initial_dir,
                    file_filter
                )

            if file_path:
                selected_path = Path(file_path)
                # Cache the parent directory for future dialogs
                cache_dialog_path(cache_key, selected_path.parent)
                return selected_path

            return None

        except Exception as e:
            logger.error(f"File dialog failed: {e}")
            raise

    def show_cached_directory_dialog(
        self,
        cache_key: PathCacheKey,
        title: str = "Select Directory",
        fallback_path: Optional[Path] = None
    ) -> Optional[Path]:
        """
        Show directory dialog with path caching.

        Args:
            cache_key: Cache key for remembering last used path
            title: Dialog title
            fallback_path: Fallback path if no cached path exists

        Returns:
            Selected directory path or None if cancelled
        """
        # Get cached initial directory
        initial_dir = str(get_cached_dialog_path(cache_key, fallback_path))

        try:
            dir_path = QFileDialog.getExistingDirectory(
                self.main_window,
                title,
                initial_dir
            )

            if dir_path:
                selected_path = Path(dir_path)
                # Cache the selected directory
                cache_dialog_path(cache_key, selected_path)
                return selected_path

            return None

        except Exception as e:
            logger.error(f"Directory dialog failed: {e}")
            raise

    def run_system_command(self, command: str, wait_for_finish: bool = True) -> bool:
        """
        Replace prompt_toolkit system command with QProcess.
        
        Args:
            command: System command to execute
            wait_for_finish: Whether to wait for command completion
            
        Returns:
            True if command executed successfully, False otherwise
        """
        try:
            process = QProcess(self.main_window)
            
            if wait_for_finish:
                process.start(command)
                success = process.waitForFinished(30000)  # 30 second timeout
                return success and process.exitCode() == 0
            else:
                # Start detached process
                return process.startDetached(command)
                
        except Exception as e:
            logger.error(f"System command failed: {command} - {e}")
            self.show_error_dialog(f"Command failed: {e}")
            return False
    
    def open_external_editor(self, file_path: Path) -> bool:
        """
        Open file in external editor using system default.
        
        Args:
            file_path: Path to file to edit
            
        Returns:
            True if editor opened successfully, False otherwise
        """
        try:
            url = QUrl.fromLocalFile(str(file_path))
            return QDesktopServices.openUrl(url)
        except Exception as e:
            logger.error(f"Failed to open external editor: {e}")
            self.show_error_dialog(f"Failed to open editor: {e}")
            return False
    
    def get_global_config(self):
        """
        Get global configuration from application.
        
        Returns:
            Global configuration object
        """
        # Access global config through application property
        if hasattr(self.app, 'global_config'):
            return self.app.global_config
        else:
            # Fallback to default config
            from openhcs.core.config import get_default_global_config
            return get_default_global_config()
    
    def set_global_config(self, config):
        """
        Set global configuration on application.
        
        Args:
            config: Global configuration object
        """
        if hasattr(self.app, 'global_config'):
            self.app.global_config = config
        else:
            # Set as application property
            setattr(self.app, 'global_config', config)
    
    def get_file_manager(self):
        """
        Get FileManager instance from application.
        
        Returns:
            FileManager instance
        """
        if hasattr(self.app, 'file_manager'):
            return self.app.file_manager
        else:
            # Create default FileManager
            from openhcs.io.filemanager import FileManager
            from openhcs.io.base import storage_registry
            file_manager = FileManager(storage_registry)
            setattr(self.app, 'file_manager', file_manager)
            return file_manager


class ExternalEditorProcess(QThread):
    """
    Thread for handling external editor processes.
    
    Replaces prompt_toolkit's run_system_command for external editor integration.
    """
    
    finished = pyqtSignal(bool, str)  # success, error_message
    
    def __init__(self, command: str, file_path: Path):
        super().__init__()
        self.command = command
        self.file_path = file_path
    
    def run(self):
        """Execute external editor command in thread."""
        try:
            process = QProcess()
            process.start(self.command)
            
            success = process.waitForFinished(300000)  # 5 minute timeout
            
            if success and process.exitCode() == 0:
                self.finished.emit(True, "")
            else:
                error_msg = process.readAllStandardError().data().decode()
                self.finished.emit(False, f"Editor failed: {error_msg}")
                
        except Exception as e:
            self.finished.emit(False, f"Editor process failed: {e}")


class AsyncOperationThread(QThread):
    """
    Generic thread for async operations.
    
    Converts async operations to Qt thread-based operations.
    """
    
    result_ready = pyqtSignal(object)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, async_func, *args, **kwargs):
        super().__init__()
        self.async_func = async_func
        self.args = args
        self.kwargs = kwargs
    
    def run(self):
        """Execute async function in thread with event loop."""
        try:
            import asyncio
            
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Run async function
            result = loop.run_until_complete(
                self.async_func(*self.args, **self.kwargs)
            )
            
            self.result_ready.emit(result)
            
        except Exception as e:
            logger.error(f"Async operation failed: {e}")
            self.error_occurred.emit(str(e))
        finally:
            # Clean up event loop
            loop.close()
