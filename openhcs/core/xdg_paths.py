"""
XDG Base Directory Specification utilities for OpenHCS.

Provides standardized paths following XDG Base Directory Specification:
- Data: ~/.local/share/openhcs/
- Cache: ~/.local/share/openhcs/cache/
- Config: ~/.local/share/openhcs/config/

Includes migration utilities to move existing cache files from legacy ~/.openhcs/ location.
"""

import logging
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def get_openhcs_data_dir() -> Path:
    """
    Get the OpenHCS data directory following XDG Base Directory Specification.
    
    Returns:
        Path to ~/.local/share/openhcs/
    """
    data_dir = Path.home() / ".local" / "share" / "openhcs"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_openhcs_cache_dir() -> Path:
    """
    Get the OpenHCS cache directory following XDG Base Directory Specification.
    
    Returns:
        Path to ~/.local/share/openhcs/cache/
    """
    cache_dir = get_openhcs_data_dir() / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_openhcs_config_dir() -> Path:
    """
    Get the OpenHCS config directory following XDG Base Directory Specification.
    
    Returns:
        Path to ~/.local/share/openhcs/config/
    """
    config_dir = get_openhcs_data_dir() / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_legacy_openhcs_dir() -> Path:
    """
    Get the legacy OpenHCS directory (~/.openhcs/).
    
    Returns:
        Path to ~/.openhcs/
    """
    return Path.home() / ".openhcs"


def migrate_cache_file(legacy_path: Path, new_path: Path, description: str) -> bool:
    """
    Migrate a cache file from legacy location to XDG-compliant location.
    
    Args:
        legacy_path: Path to the legacy cache file
        new_path: Path to the new XDG-compliant location
        description: Human-readable description of the cache file
        
    Returns:
        True if migration was performed, False if no migration needed
    """
    if not legacy_path.exists():
        logger.debug(f"No legacy {description} found at {legacy_path}")
        return False
        
    if new_path.exists():
        logger.debug(f"XDG {description} already exists at {new_path}, skipping migration")
        return False
    
    try:
        # Ensure target directory exists
        new_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy the file to new location
        shutil.copy2(legacy_path, new_path)
        logger.info(f"Migrated {description} from {legacy_path} to {new_path}")
        
        # Remove the legacy file
        legacy_path.unlink()
        logger.debug(f"Removed legacy {description} at {legacy_path}")
        
        return True
        
    except Exception as e:
        logger.warning(f"Failed to migrate {description} from {legacy_path} to {new_path}: {e}")
        return False


def migrate_cache_directory(legacy_dir: Path, new_dir: Path, description: str) -> bool:
    """
    Migrate an entire cache directory from legacy location to XDG-compliant location.
    
    Args:
        legacy_dir: Path to the legacy cache directory
        new_dir: Path to the new XDG-compliant directory
        description: Human-readable description of the cache directory
        
    Returns:
        True if migration was performed, False if no migration needed
    """
    if not legacy_dir.exists():
        logger.debug(f"No legacy {description} found at {legacy_dir}")
        return False
        
    if new_dir.exists() and any(new_dir.iterdir()):
        logger.debug(f"XDG {description} already exists and is not empty at {new_dir}, skipping migration")
        return False
    
    try:
        # Ensure target parent directory exists
        new_dir.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy the entire directory to new location
        if new_dir.exists():
            shutil.rmtree(new_dir)
        shutil.copytree(legacy_dir, new_dir)
        logger.info(f"Migrated {description} from {legacy_dir} to {new_dir}")
        
        # Remove the legacy directory
        shutil.rmtree(legacy_dir)
        logger.debug(f"Removed legacy {description} at {legacy_dir}")
        
        return True
        
    except Exception as e:
        logger.warning(f"Failed to migrate {description} from {legacy_dir} to {new_dir}: {e}")
        return False


def migrate_all_legacy_cache_files() -> None:
    """
    Migrate all known legacy cache files to XDG-compliant locations.
    
    This function should be called during application startup to ensure
    smooth transition from legacy cache locations.
    """
    legacy_dir = get_legacy_openhcs_dir()
    
    if not legacy_dir.exists():
        logger.debug("No legacy ~/.openhcs directory found, no migration needed")
        return
    
    logger.info("Checking for legacy cache files to migrate to XDG locations...")
    
    # Migrate individual cache files
    migrations = [
        (legacy_dir / "path_cache.json", get_openhcs_data_dir() / "path_cache.json", "path cache"),
        (legacy_dir / "global_config.config", get_openhcs_config_dir() / "global_config.config", "global config cache"),
    ]
    
    migrated_files = 0
    for legacy_path, new_path, description in migrations:
        if migrate_cache_file(legacy_path, new_path, description):
            migrated_files += 1
    
    # Migrate cache directory
    legacy_cache_dir = legacy_dir / "cache"
    new_cache_dir = get_openhcs_cache_dir()
    if migrate_cache_directory(legacy_cache_dir, new_cache_dir, "cache directory"):
        migrated_files += 1
    
    # Clean up empty legacy directory
    try:
        if legacy_dir.exists() and not any(legacy_dir.iterdir()):
            legacy_dir.rmdir()
            logger.info(f"Removed empty legacy directory {legacy_dir}")
    except Exception as e:
        logger.debug(f"Could not remove legacy directory {legacy_dir}: {e}")
    
    if migrated_files > 0:
        logger.info(f"Successfully migrated {migrated_files} cache files/directories to XDG locations")
    else:
        logger.debug("No legacy cache files found to migrate")


def get_cache_file_path(filename: str, legacy_filename: Optional[str] = None) -> Path:
    """
    Get path for a cache file, with automatic migration from legacy location.
    
    Args:
        filename: Name of the cache file in XDG location
        legacy_filename: Optional different filename in legacy location
        
    Returns:
        Path to the cache file in XDG location
    """
    new_path = get_openhcs_cache_dir() / filename
    
    # Check if we need to migrate from legacy location
    if not new_path.exists():
        legacy_filename = legacy_filename or filename
        legacy_path = get_legacy_openhcs_dir() / "cache" / legacy_filename
        
        if legacy_path.exists():
            migrate_cache_file(legacy_path, new_path, f"cache file {filename}")
    
    return new_path


def get_config_file_path(filename: str, legacy_filename: Optional[str] = None) -> Path:
    """
    Get path for a config file, with automatic migration from legacy location.
    
    Args:
        filename: Name of the config file in XDG location
        legacy_filename: Optional different filename in legacy location
        
    Returns:
        Path to the config file in XDG location
    """
    new_path = get_openhcs_config_dir() / filename
    
    # Check if we need to migrate from legacy location
    if not new_path.exists():
        legacy_filename = legacy_filename or filename
        legacy_path = get_legacy_openhcs_dir() / legacy_filename
        
        if legacy_path.exists():
            migrate_cache_file(legacy_path, new_path, f"config file {filename}")
    
    return new_path


def get_data_file_path(filename: str, legacy_filename: Optional[str] = None) -> Path:
    """
    Get path for a data file, with automatic migration from legacy location.
    
    Args:
        filename: Name of the data file in XDG location
        legacy_filename: Optional different filename in legacy location
        
    Returns:
        Path to the data file in XDG location
    """
    new_path = get_openhcs_data_dir() / filename
    
    # Check if we need to migrate from legacy location
    if not new_path.exists():
        legacy_filename = legacy_filename or filename
        legacy_path = get_legacy_openhcs_dir() / legacy_filename
        
        if legacy_path.exists():
            migrate_cache_file(legacy_path, new_path, f"data file {filename}")
    
    return new_path
