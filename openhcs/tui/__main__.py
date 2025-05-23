"""
OpenHCS TUI entry point.

This module provides a standalone entry point for the OpenHCS TUI,
creating separate orchestrators per plate.
"""
import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
import yaml # For loading persisted config
import dataclasses # For asdict, if needed for complex merging (not used in this simple version)

# Imports for new configuration system
from openhcs.core.config import GlobalPipelineConfig, VFSConfig, PathPlanningConfig, get_default_global_config
from openhcs.core.orchestrator.gpu_scheduler import setup_global_gpu_registry

from openhcs.tui.tui_launcher import OpenHCSTUILauncher


async def main():
    """Main entry point for OpenHCS TUI."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="OpenHCS Terminal User Interface")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--workspace", type=str, default=None,
                        help="Default workspace directory for plates")
    # The --tui-config argument is for any TUI-specific settings, not GlobalPipelineConfig.
    parser.add_argument("--tui-config", type=str, default=None, 
                        help="Path to TUI-specific configuration file (if any)")
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    if not logging.getLogger().hasHandlers(): # Ensure basicConfig is set up once
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            stream=sys.stderr 
        )
    logging.getLogger("openhcs").setLevel(log_level) # Set level for openhcs root logger
    logger = logging.getLogger("openhcs.tui.main") 
    logger.info(f"OpenHCS TUI starting with log level: {logging.getLevelName(log_level)}")

    # 1. Obtain Global Pipeline Configuration
    USER_CONFIG_DIR = Path.home() / ".config" / "openhcs"
    USER_GLOBAL_CONFIG_FILE = USER_CONFIG_DIR / "global_pipeline_config.yaml"
    
    core_global_config: Optional[GlobalPipelineConfig] = None
    
    if USER_GLOBAL_CONFIG_FILE.exists():
        logger.info(f"Attempting to load user-defined GlobalPipelineConfig from {USER_GLOBAL_CONFIG_FILE}")
        try:
            with open(USER_GLOBAL_CONFIG_FILE, 'r', encoding='utf-8') as f:
                loaded_data = yaml.safe_load(f)
            
            if loaded_data and isinstance(loaded_data, dict):
                # Reconstruct nested dataclasses if they are present as dicts
                vfs_data = loaded_data.pop('vfs', {})
                pp_data = loaded_data.pop('path_planning', {})
                
                # Create default instances to get default values for nested fields
                default_vfs = VFSConfig()
                default_pp = PathPlanningConfig()

                # Merge loaded data with defaults for nested dataclasses
                final_vfs_args = {**dataclasses.asdict(default_vfs), **vfs_data}
                final_pp_args = {**dataclasses.asdict(default_pp), **pp_data}

                current_vfs = VFSConfig(**final_vfs_args)
                current_pp = PathPlanningConfig(**final_pp_args)

                # Get defaults for top-level fields from a default GlobalPipelineConfig instance
                default_global_conf_for_top_level = get_default_global_config()

                core_global_config = GlobalPipelineConfig(
                    num_workers=loaded_data.get('num_workers', default_global_conf_for_top_level.num_workers),
                    path_planning=current_pp,
                    vfs=current_vfs
                    # Add any other top-level fields here, getting from loaded_data or default_global_conf
                )
                logger.info(f"Successfully loaded and applied user-defined GlobalPipelineConfig.")
            else:
                logger.warning(f"User config file {USER_GLOBAL_CONFIG_FILE} is empty or not a valid structure. Using default config.")
        except yaml.YAMLError as e:
            logger.warning(f"Error parsing YAML from {USER_GLOBAL_CONFIG_FILE}: {e}. Using default config.")
        except TypeError as e: # Handles issues with dataclass construction from dict if structure is wrong
            logger.warning(f"Error constructing GlobalPipelineConfig from {USER_GLOBAL_CONFIG_FILE} (likely due to mismatched fields/types): {e}. Using default config.")
        except Exception as e:
            logger.error(f"Unexpected error loading config from {USER_GLOBAL_CONFIG_FILE}: {e}. Using default config.", exc_info=True)

    if core_global_config is None: # Fallback if loading failed or file didn't exist
        core_global_config = get_default_global_config()
        logger.info("Using default GlobalPipelineConfig.")
    
    logger.info(f"Final core global config: num_workers={core_global_config.num_workers}, VFS intermediate backend='{core_global_config.vfs.default_intermediate_backend}'")

    # 2. Setup Global GPU Registry (must be done once, early)
    try:
        setup_global_gpu_registry(global_config=core_global_config)
        logger.info("GPU registry setup initiated using global config.")
    except Exception as e:
        logger.error(f"Error during GPU registry setup: {e}", exc_info=True)
        # Depending on severity, might exit or continue with GPU features potentially disabled.

    # 3. Initialize launcher, passing the core global config
    # OpenHCSTUILauncher will need to be updated to accept core_global_config
    launcher = OpenHCSTUILauncher(
        common_output_directory=args.workspace,
        # tui_config_path=args.tui_config, # If tui_launcher uses its own config file
        core_global_config=core_global_config # Pass the obtained config
    )
    
    # Run the TUI application
    await launcher.run()


if __name__ == "__main__":
    if sys.platform == "win32" and sys.version_info >= (3, 8):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        logger.info("OpenHCS TUI terminated by user.")
        sys.exit(0)
    except Exception as e:
        logging.getLogger("openhcs.tui.main").critical(f"Unhandled exception in TUI main: {e}", exc_info=True)
        sys.exit(1)