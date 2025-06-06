"""
Simple TUI Launcher for OpenHCS.

A minimal launcher that uses the canonical layout instead of the complex
architecture. This provides a working TUI that matches the tui_final.md
specification without the initialization race conditions and dynamic
container complexity.
"""
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict

from openhcs.core.config import GlobalPipelineConfig, get_default_global_config
from openhcs.core.context.processing_context import ProcessingContext

from openhcs.io.filemanager import FileManager
from openhcs.io.base import storage_registry

from openhcs.tui.layout.canonical_layout import CanonicalTUILayout

logger = logging.getLogger(__name__)


class SimpleTUIState:
    """Simplified TUI state for the canonical layout."""
    
    def __init__(self):
        """Initialize the simple TUI state."""
        self.selected_plate = None
        self.selected_step = None
        self.is_compiled = False
        self.is_running = False
        self.error_message = None
        self.current_pipeline_definition = []
        
        # Observer pattern (simplified)
        self.observers = {}
    
    def add_observer(self, event_type: str, callback):
        """Add an observer for an event type."""
        if event_type not in self.observers:
            self.observers[event_type] = []
        self.observers[event_type].append(callback)

    def remove_observer(self, event_type: str, callback):
        """Remove an observer for an event type."""
        if event_type in self.observers:
            try:
                self.observers[event_type].remove(callback)
                # Clean up empty observer lists
                if not self.observers[event_type]:
                    del self.observers[event_type]
            except ValueError:
                # Callback not found, ignore
                pass

    async def notify(self, event_type: str, data=None):
        """Notify observers of an event."""
        if event_type in self.observers:
            for callback in self.observers[event_type]:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)

    def set_selected_step(self, step):
        """Set the selected step and notify observers."""
        self.selected_step = step
        # Use synchronous notify for compatibility
        if 'step_selected' in self.observers:
            for callback in self.observers['step_selected']:
                if asyncio.iscoroutinefunction(callback):
                    from openhcs.tui.utils.unified_task_manager import get_task_manager
                    get_task_manager().fire_and_forget(callback(step), "simple_launcher_step_selected")
                else:
                    callback(step)

    def set_selected_plate(self, plate):
        """Set the selected plate and notify observers."""
        self.selected_plate = plate
        # Reset compilation state when plate changes
        self.is_compiled = False
        self.error_message = None
        # Use synchronous notify for compatibility
        if 'plate_selected' in self.observers:
            for callback in self.observers['plate_selected']:
                if asyncio.iscoroutinefunction(callback):
                    from openhcs.tui.utils.unified_task_manager import get_task_manager
                    get_task_manager().fire_and_forget(callback(plate), "simple_launcher_plate_selected")
                else:
                    callback(plate)

    async def show_dialog(self, dialog, result_future):
        """
        Show a dialog by emitting a dialog event for the layout to handle.

        Args:
            dialog: The prompt_toolkit Dialog object to display
            result_future: asyncio.Future to track dialog completion
        """
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"SimpleTUIState.show_dialog called with dialog: {dialog}")
        await self.notify('show_dialog_requested', {
            'dialog': dialog,
            'result_future': result_future
        })
        logger.info("SimpleTUIState.show_dialog: Event emitted")


class SimpleOpenHCSTUILauncher:
    """
    Simple launcher for the OpenHCS TUI.
    
    Uses the canonical layout without complex initialization or dynamic containers.
    """
    
    def __init__(self, 
                 core_global_config: Optional[GlobalPipelineConfig] = None,
                 common_output_directory: Optional[str] = None):
        """
        Initialize the simple launcher.
        
        Args:
            core_global_config: Global configuration (uses default if None)
            common_output_directory: Output directory (uses default if None)
        """
        self.logger = logger
        self.core_global_config = core_global_config or get_default_global_config()
        self.common_output_root = Path(common_output_directory) if common_output_directory else Path("./openhcs_tui_outputs")
        
        # Create output directory
        try:
            self.common_output_root.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Output directory: {self.common_output_root.resolve()}")
        except Exception as e:
            self.logger.error(f"Failed to create output directory: {e}")
        
        # Create shared components
        self.shared_storage_registry = storage_registry
        self.filemanager = FileManager(self.shared_storage_registry)
        self.state = SimpleTUIState()
        
        # Create processing context
        self.context = ProcessingContext(
            global_config=self.core_global_config,
            filemanager=self.filemanager,
            well_id="TUI_GlobalContext"
        )

        # Direct orchestrator management (no complex manager needed)
        self.orchestrator_manager = None  # Use direct orchestrator storage in PlateManagerPane

        self.logger.info("SimpleOpenHCSTUILauncher initialized successfully")
    
    async def run(self):
        """Run the TUI application."""
        self.logger.info("Starting Simple OpenHCS TUI...")
        
        try:
            # Create the canonical layout with orchestrator manager and storage registry
            tui_layout = CanonicalTUILayout(
                state=self.state,
                context=self.context,
                global_config=self.core_global_config,
                orchestrator_manager=self.orchestrator_manager,
                storage_registry=self.shared_storage_registry
            )
            
            self.logger.info("Canonical layout created, starting application...")
            
            # Run the application
            await tui_layout.run_async()
            
        except Exception as e:
            self.logger.error(f"Error running TUI: {e}", exc_info=True)

            # Show error dialog if possible
            try:
                from openhcs.tui.utils.dialog_helpers import show_scrollable_error_dialog
                await show_scrollable_error_dialog(
                    title="TUI Application Error",
                    message="A critical error occurred in the TUI application",
                    exception=e,
                    app_state=self.state
                )
            except Exception as dialog_error:
                self.logger.error(f"Failed to show error dialog: {dialog_error}")
                # Fallback to console output
                print(f"CRITICAL ERROR: {e}")
                print("Check the log file for details.")

            raise
        finally:
            self.logger.info("TUI application finished")
    
    async def shutdown(self):
        """Shutdown the launcher."""
        self.logger.info("Shutting down Simple TUI Launcher...")

        # No orchestrator manager to shutdown (using direct orchestrator storage)

        # Close filemanager if it has a close method
        if hasattr(self.filemanager, 'close'):
            try:
                if asyncio.iscoroutinefunction(self.filemanager.close):
                    await self.filemanager.close()
                else:
                    self.filemanager.close()
            except Exception as e:
                self.logger.error(f"Error closing filemanager: {e}")

        self.logger.info("Simple TUI Launcher shutdown complete")


async def main():
    """Main entry point for the simple TUI."""
    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting Simple OpenHCS TUI...")
    
    try:
        # Create and run the launcher
        launcher = SimpleOpenHCSTUILauncher()
        await launcher.run()
        
    except KeyboardInterrupt:
        logger.info("TUI terminated by user")
    except Exception as e:
        logger.error(f"Unhandled error: {e}", exc_info=True)
        print(f"ERROR: {e}")
    finally:
        logger.info("Simple TUI main finished")


if __name__ == "__main__":
    asyncio.run(main())
