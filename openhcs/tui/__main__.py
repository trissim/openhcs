from typing import Optional
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
import time
import traceback
from pathlib import Path
import yaml # For loading persisted config
import dataclasses # For asdict, if needed for complex merging (not used in this simple version)

# Imports for new configuration system
from openhcs.core.config import GlobalPipelineConfig, VFSConfig, PathPlanningConfig, get_default_global_config
from openhcs.core.orchestrator.gpu_scheduler import setup_global_gpu_registry

from openhcs.tui.tui_launcher import OpenHCSTUILauncher


async def main():
    """Main entry point for OpenHCS TUI."""
    args = _parse_command_line_arguments()
    logger = _setup_logging(args.debug)

    core_global_config = await _load_global_configuration(logger)
    await _setup_gpu_registry(core_global_config, logger)

    launcher = _create_launcher(args, core_global_config)
    await _run_tui_application(launcher, logger)

def _parse_command_line_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="OpenHCS Terminal User Interface")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--workspace", type=str, default=None,
                        help="Default workspace directory for plates")
    parser.add_argument("--tui-config", type=str, default=None,
                        help="Path to TUI-specific configuration file (if any)")
    return parser.parse_args()

def _setup_logging(debug_mode: bool):
    """Setup logging configuration."""
    log_level = logging.DEBUG if debug_mode else logging.INFO

    log_dir = Path.home() / ".local" / "share" / "openhcs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"openhcs_tui_{time.strftime('%Y%m%d_%H%M%S')}.log"

    if not logging.getLogger().hasHandlers():
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        root_logger.addHandler(file_handler)

    logging.getLogger("openhcs").setLevel(log_level)
    logger = logging.getLogger("openhcs.tui.main")
    logger.info(f"OpenHCS TUI starting with log level: {logging.getLevelName(log_level)}")
    return logger

async def _load_global_configuration(logger) -> GlobalPipelineConfig:
    """Load global pipeline configuration."""
    USER_CONFIG_DIR = Path.home() / ".config" / "openhcs"
    USER_GLOBAL_CONFIG_FILE = USER_CONFIG_DIR / "global_pipeline_config.yaml"

    if USER_GLOBAL_CONFIG_FILE.exists():
        config = await _load_config_from_file(USER_GLOBAL_CONFIG_FILE, logger)
        if config:
            return config

    config = get_default_global_config()
    logger.info("Using default GlobalPipelineConfig.")
    return config

async def _load_config_from_file(config_file: Path, logger) -> Optional[GlobalPipelineConfig]:
    """Load configuration from file with error handling."""
    logger.info(f"Attempting to load user-defined GlobalPipelineConfig from {config_file}")

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            loaded_data = yaml.safe_load(f)

        if not loaded_data or not isinstance(loaded_data, dict):
            logger.warning(f"User config file {config_file} is empty or not a valid structure. Using default config.")
            return None

        return _construct_config_from_data(loaded_data, logger)

    except yaml.YAMLError as e:
        logger.warning(f"Error parsing YAML from {config_file}: {e}. Using default config.")
        return None
    except TypeError as e:
        logger.warning(f"Error constructing GlobalPipelineConfig from {config_file} (likely due to mismatched fields/types): {e}. Using default config.")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading config from {config_file}: {e}. Using default config.", exc_info=True)
        return None

def _construct_config_from_data(loaded_data: dict, logger) -> GlobalPipelineConfig:
    """Construct configuration from loaded data."""
    vfs_data = loaded_data.pop('vfs', {})
    pp_data = loaded_data.pop('path_planning', {})

    # Create default instances and merge with loaded data
    default_vfs = VFSConfig()
    default_pp = PathPlanningConfig()

    final_vfs_args = {**dataclasses.asdict(default_vfs), **vfs_data}
    final_pp_args = {**dataclasses.asdict(default_pp), **pp_data}

    current_vfs = VFSConfig(**final_vfs_args)
    current_pp = PathPlanningConfig(**final_pp_args)

    default_global_conf = get_default_global_config()

    config = GlobalPipelineConfig(
        num_workers=loaded_data.get('num_workers', default_global_conf.num_workers),
        path_planning=current_pp,
        vfs=current_vfs
    )

    logger.info("Successfully loaded and applied user-defined GlobalPipelineConfig.")
    return config

async def _setup_gpu_registry(core_global_config: GlobalPipelineConfig, logger):
    """Setup global GPU registry."""
    try:
        setup_global_gpu_registry(global_config=core_global_config)
        logger.info("GPU registry setup initiated using global config.")
    except Exception as e:
        logger.error(f"Error during GPU registry setup: {e}", exc_info=True)

def _create_launcher(args, core_global_config: GlobalPipelineConfig) -> OpenHCSTUILauncher:
    """Create and configure the TUI launcher."""
    return OpenHCSTUILauncher(
        common_output_directory=args.workspace,
        core_global_config=core_global_config
    )

async def _run_tui_application(launcher: OpenHCSTUILauncher, logger):
    """Run the TUI application with error handling."""
    try:
        logger.info("Starting TUI launcher.run()")
        await launcher.run()
        logger.info("TUI launcher.run() completed successfully")
    except Exception as e:
        logger.critical(f"Error in launcher.run(): {e}", exc_info=True)
        print(f"ERROR: TUI crashed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    if sys.platform == "win32" and sys.version_info >= (3, 8):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        print("Starting OpenHCS TUI...")
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("OpenHCS TUI terminated by user.")
        logger.info("OpenHCS TUI terminated by user.")
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: Unhandled exception in TUI main: {e}")
        print("Check the log file for details.")
        print(f"Traceback: {traceback.format_exc()}")
        logging.getLogger("openhcs.tui.main").critical(f"Unhandled exception in TUI main: {e}", exc_info=True)
        # Wait for user to see the error
        input("Press Enter to exit...")
        sys.exit(1)