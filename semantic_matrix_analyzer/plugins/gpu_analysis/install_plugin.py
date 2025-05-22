#!/usr/bin/env python3
"""
Installation script for the GPU Analysis Plugin.

This script installs the GPU Analysis Plugin into SMA's plugin directory.
"""

import os
import sys
import shutil
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("install_plugin")

def install_plugin(sma_dir: Path = None, plugin_name: str = "gpu_analysis_plugin"):
    """
    Install the GPU Analysis Plugin into SMA's plugin directory.

    Args:
        sma_dir: Path to the SMA directory
        plugin_name: Name of the plugin file (without .py extension)

    Returns:
        True if installation was successful, False otherwise
    """
    try:
        # Determine SMA directory
        if sma_dir is None:
            # Try to find SMA directory
            sma_dir = Path("/home/ts/code/projects/brain/semantic_matrix_analyzer")
            if not sma_dir.exists():
                logger.error(f"SMA directory not found: {sma_dir}")
                return False

        # Determine plugin directories
        src_dir = Path(__file__).parent
        dst_dir = sma_dir / "semantic_matrix_analyzer" / "plugins"

        if not dst_dir.exists():
            logger.error(f"SMA plugin directory not found: {dst_dir}")
            return False

        # Copy plugin files
        src_file = src_dir / f"{plugin_name}.py"
        dst_file = dst_dir / f"{plugin_name}.py"

        if not src_file.exists():
            logger.error(f"Plugin file not found: {src_file}")
            return False

        # Copy the plugin file
        shutil.copy2(src_file, dst_file)
        logger.info(f"Copied {src_file} to {dst_file}")

        # Copy required modules
        required_modules = [
            "ast_adapter.py",
            "ast_tensor.py",
            "analyzers/semantic_analyzer.py",
            "analyzers/complexity_analyzer.py",
            "analyzers/dependency_analyzer.py",
            "analyzers/__init__.py",
            "pattern_matcher.py",
            "config_manager.py",
            "batch_processor.py",
            "utils/gpu_utils.py",
            "utils/__init__.py",
            "logging_integration.py",
            "error_handling.py",
            "config_integration.py",
        ]

        # Create required directories
        (dst_dir / "analyzers").mkdir(exist_ok=True)
        (dst_dir / "utils").mkdir(exist_ok=True)

        for module in required_modules:
            src_module = src_dir / module
            dst_module = dst_dir / module

            if not src_module.exists():
                logger.warning(f"Required module not found: {src_module}")
                continue

            # Create parent directory if it doesn't exist
            dst_module.parent.mkdir(exist_ok=True)

            # Copy the module
            shutil.copy2(src_module, dst_module)
            logger.info(f"Copied {src_module} to {dst_module}")

        logger.info("GPU Analysis Plugin installed successfully")
        return True

    except Exception as e:
        logger.error(f"Error installing GPU Analysis Plugin: {e}")
        return False

if __name__ == "__main__":
    # Parse command line arguments
    import argparse

    parser = argparse.ArgumentParser(description="Install the GPU Analysis Plugin")
    parser.add_argument("--sma-dir", type=str, help="Path to the SMA directory")
    args = parser.parse_args()

    sma_dir = Path(args.sma_dir) if args.sma_dir else None

    # Install the plugin
    success = install_plugin(sma_dir)

    sys.exit(0 if success else 1)
