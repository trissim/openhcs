"""
OpenHCS Legacy Metadata Migration Utilities

This module provides utilities to migrate old OpenHCS metadata files from the flat format
with absolute paths to the new subdirectory-keyed format with relative paths.

The migration handles:
- Converting flat metadata structure to subdirectory-keyed format
- Converting absolute paths to relative paths
- Renaming .zarr directories to clean names
- Detecting and preserving backend information (disk vs zarr)
- Creating atomic backups during migration

Usage as module:
    from openhcs.io.metadata_migration import migrate_plate_metadata, detect_legacy_format

    # Check if migration is needed
    if detect_legacy_format(metadata_dict):
        success = migrate_plate_metadata(plate_dir)

Usage as script:
    python -m openhcs.io.metadata_migration /path/to/plate/directory
    python -m openhcs.io.metadata_migration /path/to/plate/directory --dry-run
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

from .metadata_writer import METADATA_CONFIG

logger = logging.getLogger(__name__)

# Use the centralized metadata filename constant
METADATA_FILENAME = METADATA_CONFIG.METADATA_FILENAME


def detect_legacy_format(metadata_dict: Dict[str, Any]) -> bool:
    """
    Detect if metadata is in legacy format.
    
    Legacy format characteristics:
    - No 'subdirectories' key
    - 'image_files' contains absolute paths
    
    Args:
        metadata_dict: Loaded metadata dictionary
        
    Returns:
        True if legacy format detected, False otherwise
    """
    # New format has subdirectories key
    if "subdirectories" in metadata_dict:
        return False
    
    # Check if image_files contains absolute paths
    image_files = metadata_dict.get("image_files", [])
    if image_files and isinstance(image_files[0], str):
        # If first file path is absolute, assume legacy format
        return Path(image_files[0]).is_absolute()
    
    return False



def _rename_zarr_directories(plate_root: Path, dry_run: bool = False) -> Dict[str, str]:
    """
    Rename any directories containing '.zarr' in their name to remove the suffix.

    Args:
        plate_root: Root directory of the plate
        dry_run: If True, only simulate the renames

    Returns:
        Dictionary mapping old names to new names
    """
    renames = {}

    for item in plate_root.iterdir():
        if item.is_dir() and '.zarr' in item.name:
            old_name = item.name
            new_name = old_name.replace('.zarr', '')
            new_path = plate_root / new_name

            # Only rename if target doesn't already exist
            if not new_path.exists():
                if dry_run:
                    logger.info(f"DRY RUN: Would rename directory: {old_name} → {new_name}")
                else:
                    logger.info(f"Renaming directory: {old_name} → {new_name}")
                    item.rename(new_path)
                renames[old_name] = new_name
            else:
                logger.warning(f"Cannot rename {old_name} to {new_name}: target already exists")

    return renames


def migrate_legacy_metadata(legacy_metadata: Dict[str, Any], plate_root: Path, dry_run: bool = False) -> Dict[str, Any]:
    """
    Migrate legacy flat metadata format to new subdirectory-keyed format.

    Args:
        legacy_metadata: Legacy metadata dictionary
        plate_root: Root directory of the plate

    Returns:
        Migrated metadata in new format
    """
    # Step 1: Rename any .zarr directories to clean names
    renames = _rename_zarr_directories(plate_root, dry_run)

    # Step 2: Determine subdirectory and backend from renames or find data directories
    has_zarr = bool(renames)  # If we renamed .zarr directories, this is zarr storage

    if renames:
        # Use the first renamed directory as the subdirectory
        sub_dir = next(iter(renames.values()))
    else:
        # Look for existing data directories
        potential_dirs = ["images", "data", "raw"]
        sub_dir = None
        for potential_dir in potential_dirs:
            if (plate_root / potential_dir).exists():
                sub_dir = potential_dir
                break
        if sub_dir is None:
            sub_dir = "images"  # Default fallback

    # Step 3: Build relative paths using the subdirectory
    image_files = legacy_metadata.get("image_files", [])
    relative_files = []

    for legacy_path_str in image_files:
        # Extract filename from legacy path
        filename = Path(legacy_path_str).name
        # Create relative path with subdirectory prefix
        relative_files.append(f"{sub_dir}/{filename}")

    
    # Create new subdirectory-keyed structure
    migrated_metadata = {
        "subdirectories": {
            sub_dir: {
                "microscope_handler_name": legacy_metadata.get("microscope_handler_name"),
                "source_filename_parser_name": legacy_metadata.get("source_filename_parser_name"),
                "grid_dimensions": legacy_metadata.get("grid_dimensions"),
                "pixel_size": legacy_metadata.get("pixel_size"),
                "image_files": relative_files,
                "channels": legacy_metadata.get("channels"),
                "wells": legacy_metadata.get("wells"),
                "sites": legacy_metadata.get("sites"),
                "z_indexes": legacy_metadata.get("z_indexes"),
                "available_backends": {"zarr": True} if has_zarr else {"disk": True}
            }
        }
    }
    
    return migrated_metadata


def migrate_plate_metadata(plate_dir: Path, dry_run: bool = False, backup_suffix: str = ".backup") -> bool:
    """
    Migrate metadata file in a plate directory.
    
    Args:
        plate_dir: Path to plate directory
        dry_run: If True, only show what would be done
        backup_suffix: Suffix for backup file
        
    Returns:
        True if migration was needed and successful, False otherwise
    """
    metadata_file = plate_dir / METADATA_FILENAME
    
    if not metadata_file.exists():
        logger.error(f"Metadata file not found: {metadata_file}")
        return False
    
    # Load existing metadata
    try:
        with open(metadata_file, 'r') as f:
            metadata_dict = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Failed to load metadata from {metadata_file}: {e}")
        return False
    
    # Check if migration is needed
    if not detect_legacy_format(metadata_dict):
        logger.info(f"Metadata file {metadata_file} is already in new format - no migration needed")
        return False
    
    logger.info(f"Legacy format detected in {metadata_file}")
    
    # Perform migration
    try:
        migrated_metadata = migrate_legacy_metadata(metadata_dict, plate_dir, dry_run)
    except Exception as e:
        logger.error(f"Failed to migrate metadata: {e}")
        return False
    
    if dry_run:
        logger.info(f"DRY RUN: Would migrate {metadata_file}")
        logger.info(f"DRY RUN: Would create backup {metadata_file}{backup_suffix}")
        logger.info(f"DRY RUN: Migrated metadata would have {len(migrated_metadata['subdirectories'])} subdirectories")
        return True
    
    # Create backup
    backup_file = metadata_file.with_suffix(f"{metadata_file.suffix}{backup_suffix}")
    try:
        metadata_file.rename(backup_file)
        logger.info(f"Created backup: {backup_file}")
    except OSError as e:
        logger.error(f"Failed to create backup: {e}")
        return False
    
    # Write migrated metadata
    try:
        with open(metadata_file, 'w') as f:
            json.dump(migrated_metadata, f, indent=2)
        logger.info(f"Successfully migrated metadata file: {metadata_file}")
        return True
    except IOError as e:
        logger.error(f"Failed to write migrated metadata: {e}")
        # Restore backup
        try:
            backup_file.rename(metadata_file)
            logger.info(f"Restored original file from backup")
        except OSError:
            logger.error(f"Failed to restore backup - original file is at {backup_file}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Migrate OpenHCS legacy metadata files")
    parser.add_argument("plate_directory", type=Path, help="Path to plate directory containing openhcs_metadata.json")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument("--backup-suffix", default=".backup", help="Suffix for backup files (default: .backup)")
    
    args = parser.parse_args()
    
    plate_dir = args.plate_directory
    
    if not plate_dir.exists():
        logger.error(f"Plate directory does not exist: {plate_dir}")
        sys.exit(1)
    
    if not plate_dir.is_dir():
        logger.error(f"Path is not a directory: {plate_dir}")
        sys.exit(1)
    
    success = migrate_plate_metadata(plate_dir, args.dry_run, args.backup_suffix)
    
    if success:
        if args.dry_run:
            logger.info("Dry run completed - no changes made")
        else:
            logger.info("Migration completed successfully")
        sys.exit(0)
    else:
        logger.error("Migration failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
