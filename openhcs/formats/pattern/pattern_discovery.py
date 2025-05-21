"""
Pattern discovery engine for OpenHCS.

This module provides a dedicated engine for discovering and grouping patterns
in microscopy image files, separating this responsibility from FilenameParser.
"""

# Standard Library
import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from openhcs.constants.constants import DEFAULT_IMAGE_EXTENSION
from openhcs.io.filemanager import FileManager
# Core OpenHCS Interfaces
from openhcs.microscopes.microscope_interfaces_base import FilenameParser

logger = logging.getLogger(__name__)

# PatternPath class for pattern handling
class PatternPath(str):
    """
    A string subclass that represents a pattern path.

    This class replaces the original PatternPath from io/virtual_path/pattern_path.py
    which has been removed as part of the rot cleanup.
    """

    def __new__(cls, pattern_string):
        return super().__new__(cls, pattern_string)

    def is_fully_instantiated(self):
        """Check if the pattern is fully instantiated."""
        return '{' not in self and '}' not in self

    def get_pattern(self):
        """Get the pattern string."""
        return str(self)


class PatternDiscoveryEngine:
    """
    Engine for discovering and grouping patterns in microscopy image files.

    This class is responsible for:
    - Finding image files in directories
    - Filtering files based on well IDs
    - Generating patterns from files
    - Grouping patterns by components

    It works with a FilenameParser to parse individual filenames and a
    FileManager to access the file system.
    """

    # Constants
    PLACEHOLDER_PATTERN = '{iii}'

    def __init__(self, parser: FilenameParser, filemanager: FileManager):
        """
        Initialize the pattern discovery engine.

        Args:
            parser: Parser for microscopy filenames
            filemanager: FileManager for file system operations
        """
        self.parser = parser
        self.filemanager = filemanager

    def path_list_from_pattern(self, directory: Union[str, Path], pattern: Union[str, 'PatternPath'], backend: str) -> List[str]:
        """
        Get a list of filenames matching a pattern in a directory.

        Args:
            directory: Directory to search (string or Path object)
            pattern: Pattern to match (PatternPath for patterns, str for literal filenames)
            backend: Backend to use for file operations (required)

        Returns:
            List of matching filenames

        Raises:
            TypeError: If a string with braces is passed (should be a PatternPath instance)
            ValueError: If directory does not exist
        """
        # Ensure directory is a valid path
        from openhcs.io.filemanager.exceptions import \
            DirectoryNotFoundError

        # Convert to string path if it's a Path object
        directory_path = str(directory) if isinstance(directory, Path) else directory

        try:
            if not self.filemanager.is_dir(directory_path, backend):
                raise DirectoryNotFoundError(f"Directory does not exist or is not accessible: {directory_path}")
        except Exception as e:
            raise DirectoryNotFoundError(f"Failed to validate directory existence for {directory_path}") from e

        if not isinstance(directory, (str, Path)):
            raise TypeError(f"Expected string or Path object, got {type(directory).__name__}")

        # Handle literal filenames (non-pattern strings)
        if not isinstance(pattern, PatternPath):
            pattern_str = str(pattern)
            if '{' in pattern_str and '}' in pattern_str:
                raise TypeError(
                    f"Clause 88 Violation: String with braces must be a PatternPath instance. "
                    f"Got '{pattern_str}'. Use PatternPath(pattern_str) to create pattern paths."
                )

            # Use FileManager to check if file exists
            file_exists = self.filemanager.exists(
                str(directory_path / pattern_str),
                backend
            )
            if file_exists:
                return [pattern_str]
            return []

        # 🔒 Clause 92 — Structural Validation First
        # For PatternPath objects, validate they are fully instantiated
        pattern_str = pattern.get_pattern()
        logger.debug("Using pattern template: %s", pattern_str)

        # Validate that pattern is properly instantiated
        if not pattern.is_fully_instantiated():
            raise ValueError(f"Clause 93 Violation: Pattern '{pattern}' contains uninstantiated template fields")

        # Use the pattern string directly
        pattern_template = pattern_str.replace(self.PLACEHOLDER_PATTERN, '001')

        pattern_metadata = self.parser.parse_filename(pattern_template)
        if not pattern_metadata:
            logger.error("Failed to parse pattern template: %s", pattern_template)
            return []

        matching_files = []
        all_images = self.filemanager.list_image_files(directory_path, backend)
        for file_path in all_images:
            # FileManager should return strings, but handle Path objects too
            if isinstance(file_path, str):
                filename = os.path.basename(file_path)
            elif isinstance(file_path, Path):
                filename = file_path.name
            else:
                # Skip any unexpected types
                logger.warning(f"Unexpected file path type: {type(file_path).__name__}")
                continue

            file_metadata = self.parser.parse_filename(filename)
            if not file_metadata:
                continue

            is_match = True
            for key, value in pattern_metadata.items():
                if value is None:
                    continue
                if key not in file_metadata or file_metadata[key] != value:
                    is_match = False
                    break

            if is_match:
                matching_files.append(filename)

        return self._natural_sort(matching_files)

    def _natural_sort(self, file_list: List[str]) -> List[str]:
        """
        Sort filenames naturally.
        """
        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

        return sorted(file_list, key=natural_sort_key)

    def group_patterns_by_component(
        self,
        patterns: List[Union[str, 'PatternPath']],
        component: str
    ) -> Dict[str, List[Union[str, 'PatternPath']]]:
        """
        Group patterns by a required component.

        Args:
            patterns: List of patterns to group
            component: Component to group by

        Returns:
            Dictionary mapping component values to lists of patterns

        Raises:
            TypeError: If patterns are not of the expected type
            ValueError: If component is not present in a pattern
        """
        grouped_patterns = defaultdict(list)
        # 🔒 Clause 92 — Structural Validation First
        # Validate component parameter
        if not component or not isinstance(component, str):
            raise ValueError(f"Clause 92 Violation: Component must be a non-empty string, got {component}")

        # 🔒 Clause 92 — Structural Validation First
        # Strictly enforce that all patterns must be PatternPath instances
        if not all(isinstance(p, PatternPath) for p in patterns):
            raise TypeError("Clause 92 Violation: All patterns must be PatternPath instances")

        for pattern in patterns:
            pattern_str = pattern.get_pattern()

            # Validate that pattern is fully instantiated
            if not pattern.is_fully_instantiated():
                raise ValueError(f"Clause 93 Violation: Pattern '{pattern}' contains uninstantiated template fields")

            pattern_template = pattern_str.replace(self.PLACEHOLDER_PATTERN, '001')
            metadata = self.parser.parse_filename(pattern_template)

            if not metadata or component not in metadata or metadata[component] is None:
                raise ValueError(
                    f"Missing required component '{component}' in pattern: {pattern_str}"
                )

            value = str(metadata[component])
            grouped_patterns[value].append(pattern)

        return grouped_patterns

    def auto_detect_patterns(
        self,
        folder_path: Union[str, Path],
        well_filter: List[str],
        extensions: List[str],
        group_by: Optional[str],
        variable_components: List[str],
        backend: str
    ) -> Dict[str, Any]:
        """
        Automatically detect image patterns in a folder.
        """
        files_by_well = self._find_and_filter_images(
            folder_path, well_filter, extensions, True, backend
        )

        if not files_by_well:
            return {}

        result = {}
        for well, files in files_by_well.items():
            patterns = self._generate_patterns_for_files(files, variable_components)

            for pattern in patterns:
                if not isinstance(pattern, PatternPath):
                    raise TypeError(f"Clause 92 Violation: Pattern generator returned invalid type: {type(pattern).__name__}")

                if not pattern.is_fully_instantiated():
                    raise ValueError(f"Clause 93 Violation: Pattern '{pattern}' contains uninstantiated template fields")

            result[well] = (
                self.group_patterns_by_component(patterns, component=group_by)
                if group_by else patterns
            )

        return result

    def _find_and_filter_images(
        self,
        folder_path: Union[str, Path],
        well_filter: List[str],
        extensions: List[str],
        recursive: bool,
        backend: str
    ) -> Dict[str, List[Any]]:
        """
        Find all image files in a directory and filter by well.

        Args:
            folder_path: Path to the folder to search (string or Path object)
            well_filter: List of wells to include
            extensions: List of file extensions to include
            recursive: Whether to search recursively
            backend: Backend to use for file operations (required)

        Returns:
            Dictionary mapping wells to lists of image paths

        Raises:
            TypeError: If folder_path is not a string or Path object
            ValueError: If well_filter is empty or folder_path does not exist
        """
        # Ensure folder_path is a valid path
        if isinstance(folder_path, str):
            folder_path = Path(folder_path)
        elif not isinstance(folder_path, Path):
            raise TypeError(f"Expected string or Path object, got {type(folder_path).__name__}")

        # Ensure the path exists
        if not folder_path.exists():
            raise ValueError(f"Folder path does not exist: {folder_path}")

        # 🔒 Clause 92 — Structural Validation First
        # Validate well_filter
        if not well_filter:
            raise ValueError("Clause 92 Violation: well_filter cannot be empty")

        extensions = extensions or ['.tif', '.TIF', '.tiff', '.TIFF']

        image_paths = self.filemanager.list_image_files(folder_path, backend, extensions=extensions, recursive=recursive)

        files_by_well = defaultdict(list)
        for img_path in image_paths:
            # FileManager should return strings, but handle Path objects too
            if isinstance(img_path, str):
                filename = os.path.basename(img_path)
            elif isinstance(img_path, Path):
                filename = img_path.name
            else:
                # Skip any unexpected types
                logger.warning(f"Unexpected file path type: {type(img_path).__name__}")
                continue

            metadata = self.parser.parse_filename(filename)
            if not metadata:
                continue

            well = metadata['well']
            if well not in well_filter:
                continue

            files_by_well[well].append(img_path)

        return files_by_well

    def _generate_patterns_for_files(
        self,
        files: List[Any],
        variable_components: List[str]
    ) -> List['PatternPath']:
        """
        Generate patterns for a list of files.

        Args:
            files: List of file path objects representing files
            variable_components: List of components that can vary in the pattern

        Returns:
            List of PatternPath objects

        Raises:
            TypeError: If files list is not a list
            ValueError: If pattern templates cannot be instantiated
        """
        # 🔒 Clause 92 — Structural Validation First
        # Validate input parameters
        if not isinstance(files, list):
            raise TypeError(f"Clause 92 Violation: Expected list of file path objects, got {type(files).__name__}")

        if not isinstance(variable_components, list):
            raise TypeError(f"Clause 92 Violation: Expected list of variable components, got {type(variable_components).__name__}")


        component_combinations = defaultdict(list)
        for file_path in files:
            # FileManager should return strings, but handle Path objects too
            if isinstance(file_path, str):
                filename = os.path.basename(file_path)
            elif isinstance(file_path, Path):
                filename = file_path.name
            else:
                # Skip any unexpected types
                logger.warning(f"Unexpected file path type: {type(file_path).__name__}")
                continue

            metadata = self.parser.parse_filename(filename)
            if not metadata:
                continue

            key_parts = []
            for comp in self.parser.FILENAME_COMPONENTS:
                if comp in metadata and comp not in variable_components and metadata[comp] is not None:
                    key_parts.append(f"{comp}={metadata[comp]}")

            key = ",".join(key_parts)
            component_combinations[key].append((file_path, metadata))

        patterns = []
        for _, files_metadata in component_combinations.items():
            if not files_metadata:
                continue

            _, template_metadata = files_metadata[0]
            pattern_args = {}
            for comp in self.parser.FILENAME_COMPONENTS:
                if comp in template_metadata:
                    if comp in variable_components:
                        pattern_args[comp] = self.PLACEHOLDER_PATTERN
                    else:
                        pattern_args[comp] = template_metadata[comp]

            # 🔒 Clause 93 — Declarative Execution Enforcement
            # Ensure all required components are present
            if 'well' not in pattern_args or pattern_args['well'] is None:
                raise ValueError("Clause 93 Violation: 'well' is a required component for pattern templates")

            pattern_str = self.parser.construct_filename(
                well=pattern_args['well'],
                site=pattern_args.get('site'),
                channel=pattern_args.get('channel'),
                z_index=pattern_args.get('z_index'),
                extension=pattern_args.get('extension') or DEFAULT_IMAGE_EXTENSION
            )

            # Create pattern and validate it can be instantiated
            # Use FileManager to create a pattern
            pattern = PatternPath(pattern_str)

            # 🔒 Clause 92 — Structural Validation First
            # Validate that the pattern is a PatternPath
            if not isinstance(pattern, PatternPath):
                raise TypeError(f"Clause 92 Violation: Expected PatternPath, got {type(pattern).__name__}")

            # Validate that the pattern can be instantiated
            test_instance = pattern_str.replace(self.PLACEHOLDER_PATTERN, '001')
            if not self.parser.parse_filename(test_instance):
                raise ValueError(f"Clause 93 Violation: Pattern template '{pattern_str}' cannot be instantiated")

            # Validate that the pattern is properly registered
            if not hasattr(pattern, 'is_fully_instantiated'):
                raise TypeError(f"Clause 93 Violation: Pattern '{pattern}' does not implement is_fully_instantiated()")

            patterns.append(pattern)

        # 🔒 Clause 92 — Structural Validation First
        # Validate the final pattern list
        if not patterns:
            logger.warning("No patterns generated from files")

        return patterns
