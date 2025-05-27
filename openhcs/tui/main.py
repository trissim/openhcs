"""
Main entry point for OpenHCS Hybrid TUI.

Provides a simple way to launch the hybrid TUI application with
proper initialization and error handling.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

from .controllers.app_controller import AppController
from .state import TUIState
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.config import GlobalPipelineConfig

# Setup logging - only to file during TUI to avoid screen pollution
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('openhcs_tui.log')
        # Removed StreamHandler to prevent console output during TUI
    ]
)

logger = logging.getLogger(__name__)

class HybridTUIApp:
    """
    Main application class for the hybrid TUI.

    Handles initialization, startup, and shutdown of the TUI application.
    """

    def __init__(self):
        """Initialize the hybrid TUI application."""
        self.app_controller: Optional[AppController] = None
        self.is_initialized = False
        self.state: Optional[TUIState] = None
        self.context: Optional[ProcessingContext] = None

    async def initialize(self) -> None:
        """Initialize the application."""
        try:
            logger.info("Initializing OpenHCS Hybrid TUI...")

            # Create TUI state
            self.state = TUIState()

            # Create processing context with minimal config
            global_config = GlobalPipelineConfig()
            self.context = ProcessingContext(global_config=global_config)

            # Create and initialize app controller with state and context
            self.app_controller = AppController(state=self.state, context=self.context)
            await self.app_controller.initialize_controller()

            self.is_initialized = True
            logger.info("Hybrid TUI initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize hybrid TUI: {e}")
            raise

    async def run(self) -> None:
        """Run the application."""
        try:
            if not self.is_initialized:
                await self.initialize()

            if not self.app_controller:
                raise RuntimeError("App controller not initialized")

            logger.info("Starting OpenHCS Hybrid TUI...")
            await self.app_controller.run_async()

        except KeyboardInterrupt:
            logger.info("Application interrupted by user")
        except Exception as e:
            logger.error(f"Application error: {e}")
            raise
        finally:
            await self.cleanup()

    async def cleanup(self) -> None:
        """Clean up application resources."""
        try:
            if self.app_controller:
                await self.app_controller.cleanup_controller()
                self.app_controller = None

            self.is_initialized = False
            logger.info("Hybrid TUI cleanup complete")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

async def main() -> None:
    """Main entry point for the hybrid TUI."""
    app = HybridTUIApp()

    try:
        await app.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

def run_tui() -> None:
    """Synchronous entry point for the hybrid TUI."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
    except Exception as e:
        logger.error(f"Failed to run TUI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_tui()
