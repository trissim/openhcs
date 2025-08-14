"""Window service to break circular imports between widgets and windows."""

from typing import Any, Callable, List, Optional, Type
from pathlib import Path
from textual.css.query import NoMatches

from openhcs.constants.constants import Backend
from openhcs.textual_tui.services.file_browser_service import SelectionMode
from openhcs.core.path_cache import PathCacheKey


class WindowService:
    """Service to handle window creation and management, breaking circular imports."""
    
    def __init__(self, app):
        self.app = app
    
    async def open_file_browser(
        self,
        file_manager,
        initial_path: Path,
        backend: Backend = Backend.DISK,
        title: str = "Select Directory",
        mode: str = "load",  # "load" or "save"
        selection_mode: SelectionMode = SelectionMode.DIRECTORIES_ONLY,
        filter_extensions: Optional[List[str]] = None,
        default_filename: str = "",
        cache_key: Optional[PathCacheKey] = None,
        on_result_callback: Optional[Callable] = None,
        caller_id: str = "unknown",
        enable_multi_selection: bool = False,
    ):
        """Open file browser window without circular imports."""
        # Lazy import to avoid circular dependency
        from openhcs.textual_tui.windows.file_browser_window import (
            open_file_browser_window, BrowserMode
        )
        
        browser_mode = BrowserMode.SAVE if mode == "save" else BrowserMode.LOAD
        
        return await open_file_browser_window(
            app=self.app,
            file_manager=file_manager,
            initial_path=initial_path,
            backend=backend,
            title=title,
            mode=browser_mode,
            selection_mode=selection_mode,
            filter_extensions=filter_extensions,
            default_filename=default_filename,
            cache_key=cache_key,
            on_result_callback=on_result_callback,
            caller_id=caller_id,
            enable_multi_selection=enable_multi_selection,
        )
    
    async def open_config_window(
        self,
        config_class: Type,
        current_config: Any,
        on_save_callback: Optional[Callable] = None
    ):
        """
        Open config window with separate config_class and current_config parameters.

        Supports both GlobalPipelineConfig (global) and PipelineConfig (per-orchestrator).
        """
        try:
            window = self.app.query_one(ConfigWindow)
            window.open_state = True
        except NoMatches:
            window = ConfigWindow(
                config_class=config_class,
                current_config=current_config,
                on_save_callback=on_save_callback
            )
            await self.app.mount(window)
            window.open_state = True
        return window
    
    async def open_multi_orchestrator_config(
        self, 
        orchestrators: List, 
        on_save_callback: Optional[Callable] = None
    ):
        """Open multi-orchestrator config window without circular imports."""
        # Lazy import to avoid circular dependency
        from openhcs.textual_tui.windows.multi_orchestrator_config_window import (
            show_multi_orchestrator_config
        )
        
        return await show_multi_orchestrator_config(
            app=self.app,
            orchestrators=orchestrators,
            on_save_callback=on_save_callback
        )
