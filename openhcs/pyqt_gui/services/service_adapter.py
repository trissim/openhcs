"""
PyQt6 Service Adapter

Bridges OpenHCS services to PyQt6 context, replacing prompt_toolkit dependencies
with Qt equivalents while preserving all business logic.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from pathlib import Path

from PyQt6.QtWidgets import QMessageBox, QFileDialog, QApplication, QWidget
from PyQt6.QtCore import QProcess, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtCore import QUrl

from openhcs.core.path_cache import (
    PathCacheKey,
    get_cached_dialog_path,
    cache_dialog_path,
)
from pyqt_reactive.theming import ThemeManager
from pyqt_reactive.theming import ColorScheme

logger = logging.getLogger(__name__)


class GlobalEventBus(QObject):
    """
    Centralized event bus for cross-window communication.

    ARCHITECTURE: OpenHCS "Set and Forget" Pattern
    ===============================================

    Instead of manually connecting every window to every other window's signals,
    all windows register with this global event bus once during initialization.

    When ANY window modifies pipeline/config (via code editor, UI, etc.), it
    broadcasts to the event bus, which automatically notifies ALL registered windows.

    Benefits:
    - No manual cross-connections between windows
    - Adding a new window type automatically works with all existing windows
    - Single source of truth for cross-window events
    - Eliminates scattered signal connection code

    Usage:
    1. Windows inherit from BaseFormDialog (automatic registration)
    2. Windows call _broadcast_pipeline_changed() or _broadcast_config_changed()
    3. Windows implement _on_pipeline_changed() to receive updates

    This is the OpenHCS "set and forget" pattern - add a new window type once,
    and it automatically receives updates from all other windows.
    """

    # Global signals that all windows can emit/receive
    pipeline_changed = pyqtSignal(
        list
    )  # List[FunctionStep] - emitted when pipeline changes
    config_changed = pyqtSignal(
        object
    )  # Config object - emitted when any config changes
    step_changed = pyqtSignal(object)  # FunctionStep - emitted when a step is modified

    def __init__(self):
        super().__init__()
        self._registered_windows = []
        logger.debug("Global event bus initialized")

    def register_window(self, window):
        """Register a window to receive global events.

        Args:
            window: Window instance to register
        """
        if window not in self._registered_windows:
            self._registered_windows.append(window)
            logger.debug(f"Registered window: {window.__class__.__name__}")

    def unregister_window(self, window):
        """Unregister a window from receiving global events.

        Args:
            window: Window instance to unregister
        """
        if window in self._registered_windows:
            self._registered_windows.remove(window)
            logger.debug(f"Unregistered window: {window.__class__.__name__}")

    def emit_pipeline_changed(self, pipeline_steps: list):
        """Emit pipeline changed event to all registered windows.

        Args:
            pipeline_steps: Updated list of FunctionStep objects
        """
        logger.debug(
            f"Broadcasting pipeline_changed to {len(self._registered_windows)} windows"
        )
        self.pipeline_changed.emit(pipeline_steps)

    def emit_config_changed(self, config):
        """Emit config changed event to all registered windows.

        Args:
            config: Updated config object
        """
        logger.debug(
            f"Broadcasting config_changed to {len(self._registered_windows)} windows"
        )
        self.config_changed.emit(config)

    def emit_step_changed(self, step):
        """Emit step changed event to all registered windows.

        Args:
            step: Updated FunctionStep object
        """
        logger.debug(
            f"Broadcasting step_changed to {len(self._registered_windows)} windows"
        )
        self.step_changed.emit(step)


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
        self._thread_pool = ThreadPoolExecutor(max_workers=4)

        # Initialize theme manager for centralized color management
        self.theme_manager = ThemeManager()

        # Initialize global event bus for cross-window communication
        self.event_bus = GlobalEventBus()

        # Apply dark theme globally to ensure consistent dialog styling
        self._apply_dark_theme()

        logger.debug("PyQt6 service adapter initialized")

    def _apply_dark_theme(self):
        """Apply dark theme globally for consistent dialog styling."""
        try:
            # Create dark color scheme (same as enhanced path widget)
            dark_scheme = ColorScheme()

            # Apply the dark theme globally so all dialogs use consistent styling
            self.theme_manager.apply_color_scheme(dark_scheme)

            logger.debug("Applied dark theme globally for consistent dialog styling")
        except Exception as e:
            logger.warning(f"Failed to apply dark theme: {e}")

    def execute_async_operation(self, async_func, *args, **kwargs):
        """
        Execute async operation using ThreadPoolExecutor (simpler and more reliable).

        Args:
            async_func: Async function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
        """
        import asyncio
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

        # Submit to thread pool (non-blocking like TUI executor)
        self._thread_pool.submit(run_async_in_thread)

    def show_dialog(self, content: str, title: str = "OpenHCS") -> bool:
        """
        Replace prompt_toolkit dialogs with QMessageBox.

        Args:
            content: Dialog content text
            title: Dialog title

        Returns:
            True if user clicked OK, False otherwise
        """
        msg = self.create_message_box(
            icon=QMessageBox.Icon.Question,
            title=title,
            text=content,
            buttons=(
                QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel
            ),
            default_button=QMessageBox.StandardButton.Ok,
        )
        result = QMessageBox.StandardButton(msg.exec())
        return result == QMessageBox.StandardButton.Ok

    def create_message_box(
        self,
        *,
        icon: QMessageBox.Icon,
        title: str,
        text: str,
        buttons: QMessageBox.StandardButton,
        default_button: QMessageBox.StandardButton,
    ) -> QMessageBox:
        """Build one message box with the current shared application theme."""

        message_box = QMessageBox(self.main_window)
        message_box.setIcon(icon)
        message_box.setWindowTitle(title)
        message_box.setText(text)
        message_box.setStandardButtons(buttons)
        message_box.setDefaultButton(default_button)
        styles = self.get_style_generator()
        message_box.setStyleSheet(
            styles.generate_dialog_style()
            + "\n"
            + styles.generate_button_style()
        )
        return message_box

    def show_error_dialog(self, error_message: str, title: str = "Error") -> None:
        """
        Show error dialog with error icon.

        Args:
            error_message: Error message to display
            title: Dialog title
        """
        self.create_message_box(
            icon=QMessageBox.Icon.Critical,
            title=title,
            text=error_message,
            buttons=QMessageBox.StandardButton.Ok,
            default_button=QMessageBox.StandardButton.Ok,
        ).exec()

    def show_info_dialog(self, info_message: str, title: str = "Information") -> None:
        """
        Show information dialog.

        Args:
            info_message: Information message to display
            title: Dialog title
        """
        self.create_message_box(
            icon=QMessageBox.Icon.Information,
            title=title,
            text=info_message,
            buttons=QMessageBox.StandardButton.Ok,
            default_button=QMessageBox.StandardButton.Ok,
        ).exec()

    def show_warning_dialog(
        self,
        warning_message: str,
        title: str = "Warning",
    ) -> None:
        """Show a warning dialog through the shared themed owner."""

        self.create_message_box(
            icon=QMessageBox.Icon.Warning,
            title=title,
            text=warning_message,
            buttons=QMessageBox.StandardButton.Ok,
            default_button=QMessageBox.StandardButton.Ok,
        ).exec()

    def show_cached_file_dialog(
        self,
        cache_key: PathCacheKey,
        title: str = "Select File",
        file_filter: str = "All Files (*)",
        mode: str = "open",
        fallback_path: Optional[Path] = None,
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
                    self.main_window, title, initial_dir, file_filter
                )
            else:  # mode == "open"
                file_path, _ = QFileDialog.getOpenFileName(
                    self.main_window, title, initial_dir, file_filter
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
        fallback_path: Optional[Path] = None,
        allow_multiple: bool = False,
    ) -> Optional[Path | list[Path]]:
        """
        Show directory dialog with path caching.

        Args:
            cache_key: Cache key for remembering last used path
            title: Dialog title
            fallback_path: Fallback path if no cached path exists
            allow_multiple: If True, allow selecting multiple directories

        Returns:
            Selected directory path(s) or None if cancelled
            - Single Path if allow_multiple=False
            - List[Path] if allow_multiple=True
        """
        # Get cached initial directory
        initial_path = get_cached_dialog_path(cache_key, fallback_path)
        initial_dir = str(initial_path)

        try:
            if allow_multiple:
                # Use custom QFileDialog for multi-directory selection.
                # NOTE: DontUseNativeDialog is required for multi-select.
                dialog = QFileDialog(self.main_window, title, initial_dir)
                dialog.setFileMode(QFileDialog.FileMode.Directory)
                dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)

                # Enable multi-selection in the list/tree views.
                list_view = dialog.findChild(QWidget, "listView")
                if list_view:
                    from PyQt6.QtWidgets import QAbstractItemView

                    list_view.setSelectionMode(
                        QAbstractItemView.SelectionMode.ExtendedSelection
                    )

                tree_view = dialog.findChild(QWidget, "treeView")
                if tree_view:
                    from PyQt6.QtWidgets import QAbstractItemView

                    tree_view.setSelectionMode(
                        QAbstractItemView.SelectionMode.ExtendedSelection
                    )

                # Make the path bar editable so users can paste paths.
                # In the non-native dialog, the path widget is a QComboBox ("lookInCombo").
                try:
                    from PyQt6.QtWidgets import QComboBox

                    look_in = dialog.findChild(QComboBox, "lookInCombo")
                    if look_in:
                        look_in.setEditable(True)
                        line_edit = look_in.lineEdit()
                        if line_edit:
                            line_edit.setPlaceholderText("Paste a path and press Enter")

                            def _jump_to_typed_path() -> None:
                                raw = line_edit.text().strip().strip('"').strip("'")
                                if not raw:
                                    return
                                try:
                                    p = Path(raw).expanduser()
                                    if p.exists() and p.is_dir():
                                        dialog.setDirectory(str(p))
                                except Exception:
                                    # Leave the dialog as-is if the path is invalid.
                                    return

                            line_edit.returnPressed.connect(_jump_to_typed_path)
                except Exception:
                    # If the internal widget names differ across platforms, fail gracefully.
                    pass

                if dialog.exec():
                    selected_paths = [Path(p) for p in dialog.selectedFiles()]
                    if selected_paths:
                        # Cache the first selected directory
                        cache_dialog_path(cache_key, selected_paths[0])
                        return selected_paths
                return None
            else:
                # Single directory selection (native dialog)
                dir_path = QFileDialog.getExistingDirectory(
                    self.main_window, title, initial_dir
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
        return self.main_window.pipeline_runtime_config

    def set_global_config(self, config):
        """
        Set global configuration on application.

        Args:
            config: Global configuration object
        """
        self.main_window.set_pipeline_runtime_config(config)

    # ========== THEME MANAGEMENT METHODS ==========

    def get_theme_manager(self) -> ThemeManager:
        """
        Get the theme manager for color scheme management.

        Returns:
            ThemeManager: Current theme manager instance
        """
        return self.theme_manager

    def get_current_color_scheme(self) -> ColorScheme:
        """
        Get the current color scheme.

        Returns:
            ColorScheme: Current color scheme
        """
        return self.theme_manager.color_scheme

    def get_style_generator(self):
        """
        Get the style generator for creating consistent widget styles.

        Returns:
            StyleSheetGenerator: Style generator instance
        """
        return self.theme_manager.style_generator

    def apply_color_scheme(self, color_scheme: ColorScheme):
        """
        Apply a new color scheme to the entire application.

        Args:
            color_scheme: New color scheme to apply
        """
        self.theme_manager.apply_color_scheme(color_scheme)

    def switch_to_dark_theme(self):
        """Switch to dark theme variant."""
        self.theme_manager.switch_to_dark_theme()

    def switch_to_light_theme(self):
        """Switch to light theme variant."""
        self.theme_manager.switch_to_light_theme()

    def load_theme_from_config(self, config_path: str) -> bool:
        """
        Load and apply theme from configuration file.

        Args:
            config_path: Path to JSON configuration file

        Returns:
            bool: True if successful, False otherwise
        """
        return self.theme_manager.load_theme_from_config(config_path)

    def save_current_theme(self, config_path: str) -> bool:
        """
        Save current theme to configuration file.

        Args:
            config_path: Path to save JSON configuration file

        Returns:
            bool: True if successful, False otherwise
        """
        return self.theme_manager.save_current_theme(config_path)

    def get_current_style_sheet(self) -> str:
        """
        Get the current complete application style sheet.

        Returns:
            str: Complete QStyleSheet for current theme
        """
        return self.theme_manager.get_current_style_sheet()

    def register_theme_change_callback(self, callback):
        """
        Register a callback to be called when theme changes.

        Args:
            callback: Function to call with new color scheme
        """
        self.theme_manager.register_theme_change_callback(callback)

    def get_file_manager(self):
        """
        Get FileManager instance from application.

        Returns:
            FileManager instance
        """
        return self.main_window.file_manager

    def get_event_bus(self) -> GlobalEventBus:
        """Get the global event bus for cross-window communication.

        Returns:
            GlobalEventBus instance
        """
        return self.event_bus


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
            result = loop.run_until_complete(self.async_func(*self.args, **self.kwargs))

            self.result_ready.emit(result)

        except Exception as e:
            logger.error(f"Async operation failed: {e}")
            self.error_occurred.emit(str(e))
        finally:
            # Clean up event loop
            loop.close()
