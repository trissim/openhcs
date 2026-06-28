"""
Pattern discovery engine for OpenHCS.

This module provides a dedicated engine for discovering and grouping patterns
in microscopy image files, separating this responsibility from FilenameParser.
"""

# Standard Library
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from openhcs.constants.constants import DEFAULT_IMAGE_EXTENSION
from polystore.filemanager import FileManager
# Core OpenHCS Interfaces
from openhcs.microscopes.microscope_interfaces import FilenameParser

# Note: Previously used GenericPatternEngine, but now we always use microscope-specific parsers

logger = logging.getLogger(__name__)

# Pattern utility functions
def has_placeholders(pattern: str) -> bool:
    """Check if pattern contains placeholder variables."""
    return '{' in pattern and '}' in pattern


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

    def path_list_from_pattern(self, directory: Union[str, Path], pattern: str, backend: str, variable_components: Optional[List[str]] = None) -> List[str]:
        """
        Get a list of filenames matching a pattern in a directory.

        Args:
            directory: Directory to search (string or Path object)
            pattern: Pattern to match (string with optional {iii} placeholders)
            backend: Backend to use for file operations (required)
            variable_components: List of components that can vary (will be ignored during matching)

        Returns:
            List of matching filenames

        Raises:
            ValueError: If directory does not exist
        """
        directory_path = str(directory)  # Keep as string for FileManager consistency
        if not self.filemanager.is_dir(directory_path, backend):
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        pattern_str = str(pattern)

        # Handle literal filenames (patterns without placeholders)
        if not has_placeholders(pattern_str):
            # Use FileManager to check if file exists
            file_path = os.path.join(directory_path, pattern_str)  # Use os.path.join instead of /
            file_exists = self.filemanager.exists(file_path, backend)
            if file_exists:
                return [pattern_str]
            return []

        # Handle pattern strings with placeholders
        logger.debug("Using pattern template: %s", pattern_str)

        # Parse pattern template to get expected structure
        pattern_metadata = self.parser.parse_filename(pattern_str)
        if not pattern_metadata:
            logger.error("Failed to parse pattern template: %s", pattern_str)
            return []

        # Get all image files in directory using FileManager
        all_files = self.filemanager.list_image_files(str(directory_path), backend)

        matching_files = []

        for file_path in all_files:
            # Extract filename from path
            if isinstance(file_path, str):
                filename = os.path.basename(file_path)
            elif isinstance(file_path, Path):
                filename = file_path.name
            else:
                continue

            # Parse the actual filename
            file_metadata = self.parser.parse_filename(filename)
            if not file_metadata:
                continue

            # Check if file matches pattern structure
            if self._matches_pattern_structure(file_metadata, pattern_metadata, variable_components or []):
                matching_files.append(filename)

        return matching_files

    def _matches_pattern_structure(self, file_metadata: Dict[str, Any], pattern_metadata: Dict[str, Any], variable_components: List[str]) -> bool:
        """
        Check if a file's metadata matches a pattern's structure.

        Args:
            file_metadata: Metadata extracted from actual filename
            pattern_metadata: Metadata extracted from pattern template
            variable_components: List of components that can vary

        Returns:
            True if file matches pattern structure, False otherwise
        """
        # Check all components in the pattern
        for component in self.parser.FILENAME_COMPONENTS:
            if component not in pattern_metadata:
                continue

            pattern_value = pattern_metadata[component]
            file_value = file_metadata.get(component)

            # Variable components can have any value
            if component in variable_components:
                # File must have a value for this component, but it can be anything
                if file_value is None:
                    return False
                continue

            # Fixed components must match exactly
            if pattern_value != file_value:
                return False

        return True

    def group_patterns_by_component(
        self,
        patterns: List[str],
        component: str
    ) -> Dict[str, List[str]]:
        """
        Group patterns by a required component.

        Args:
            patterns: List of pattern strings to group
            component: Component to group by

        Returns:
            Dictionary mapping component values to lists of patterns

        Raises:
            TypeError: If patterns are not strings
            ValueError: If component is not present in a pattern
        """
        grouped_patterns = defaultdict(list)
        # Validate inputs
        if not component or not isinstance(component, str):
            raise ValueError(f"Component must be a non-empty string, got {component}")

        if not all(isinstance(p, str) for p in patterns):
            raise TypeError("All patterns must be strings")

        for pattern in patterns:
            pattern_str = str(pattern)

            # Note: Patterns with template fields (like {iii}) are EXPECTED for pattern discovery
            # The has_placeholders() check is only relevant when using patterns as concrete filenames
            # For pattern discovery and grouping, we WANT patterns with placeholders

            metadata = self.parser.parse_filename(pattern_str)

            if not metadata or component not in metadata or metadata[component] is None:
                raise ValueError(
                    f"Missing required component '{component}' in pattern: {pattern_str}"
                )

            value = str(metadata[component])
            grouped_patterns[value].append(pattern)

        return grouped_patterns

    def subdivide_patterns_by_components(
        self,
        patterns: List[str],
        components: List[str]
    ) -> Dict[tuple, List[str]]:
        """
        Subdivide patterns by multiple component values.

        Args:
            patterns: List of pattern strings
            components: List of component names to subdivide by

        Returns:
            Dictionary mapping component value tuples to pattern lists
            Example: {('001', '1'): [...], ('001', '2'): [...]}
        """
        if not components:
            return {(): patterns}

        subdivided = defaultdict(list)
        for pattern in patterns:
            metadata = self.parser.parse_filename(str(pattern))
            if not metadata:
                raise ValueError(f"Failed to parse pattern: {pattern}")
            key = tuple(str(metadata[comp]) for comp in components if comp in metadata and metadata[comp] is not None)
            subdivided[key].append(pattern)
        return dict(subdivided)

    def auto_detect_patterns(
        self,
        folder_path: Union[str, Path],
        variable_components: List[str],
        backend: str,
        extensions: List[str] = None,
        group_by=None,  # Accept GroupBy enum or None
        recursive: bool = False,
        **kwargs  # Dynamic filter parameters (e.g., well_filter, site_filter)
    ) -> Dict[str, Any]:
        """
        Automatically detect image patterns in a folder.
        """
        # Extract axis_filter from dynamic kwargs
        from openhcs.constants import MULTIPROCESSING_AXIS
        axis_name = MULTIPROCESSING_AXIS.value
        axis_filter = kwargs.get(f"{axis_name}_filter")

        files_by_axis = self._find_and_filter_images(
            folder_path, axis_filter, extensions, recursive, backend
        )

        if not files_by_axis:
            return {}

        return self._patterns_for_files_by_axis(
            files_by_axis,
            variable_components,
            group_by,
        )

    def auto_detect_patterns_from_files(
        self,
        image_paths: List[Union[str, Path]],
        variable_components: List[str],
        group_by=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Automatically detect image patterns from an authoritative file list."""

        from openhcs.constants import MULTIPROCESSING_AXIS

        axis_name = MULTIPROCESSING_AXIS.value
        axis_filter = kwargs.get(f"{axis_name}_filter")
        files_by_axis = self._filter_images_by_axis(image_paths, axis_filter)
        if not files_by_axis:
            return {}
        return self._patterns_for_files_by_axis(
            files_by_axis,
            variable_components,
            group_by,
        )

    def _patterns_for_files_by_axis(
        self,
        files_by_axis: Dict[str, List[Any]],
        variable_components: List[str],
        group_by=None,
    ) -> Dict[str, Any]:
        result = {}
        for axis_value, files in files_by_axis.items():
            patterns = self._generate_patterns_for_files(files, variable_components, axis_value)

            # Validate patterns
            for pattern in patterns:
                if not isinstance(pattern, str):
                    raise TypeError(f"Pattern generator returned invalid type: {type(pattern).__name__}")

            if group_by:
                # Extract string value from GroupBy enum for pattern grouping
                component_string = group_by.value if group_by.value else None
                if component_string:
                    result[axis_value] = self.group_patterns_by_component(patterns, component=component_string)
                else:
                    result[axis_value] = patterns
            else:
                result[axis_value] = patterns

        return result

    def _find_and_filter_images(
        self,
        folder_path: Union[str, Path],
        axis_filter: List[str],
        extensions: List[str],
        recursive: bool,
        backend: str
    ) -> Dict[str, List[Any]]:
        """
        Find all image files in a directory and filter by multiprocessing axis.

        Args:
            folder_path: Path to the folder to search (string or Path object)
            axis_filter: List of axis values to include
            extensions: List of file extensions to include
            recursive: Whether to search recursively
            backend: Backend to use for file operations (required)

        Returns:
            Dictionary mapping axis values to lists of image paths

        Raises:
            TypeError: If folder_path is not a string or Path object
            ValueError: If axis_filter is empty or folder_path does not exist
        """
        # Convert to Path and validate using FileManager abstraction
        folder_path = Path(folder_path)
        if not self.filemanager.exists(str(folder_path), backend):
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        # Validate inputs
        if not axis_filter:
            raise ValueError("axis_filter cannot be empty")

        extensions = extensions or ['.tif', '.TIF', '.tiff', '.TIFF']

        image_paths = self.filemanager.list_image_files(folder_path, backend, extensions=extensions, recursive=recursive)
        return self._filter_images_by_axis(image_paths, axis_filter)

    def _filter_images_by_axis(
        self,
        image_paths: List[Any],
        axis_filter: List[str],
    ) -> Dict[str, List[Any]]:
        if not axis_filter:
            raise ValueError("axis_filter cannot be empty")

        files_by_axis = defaultdict(list)
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

            # Get multiprocessing axis dynamically from configuration
            from openhcs.constants import MULTIPROCESSING_AXIS
            axis_key = MULTIPROCESSING_AXIS.value
            axis_value = metadata.get(axis_key)
            if not axis_value or axis_value not in axis_filter:
                continue

            files_by_axis[axis_value].append(img_path)

        return files_by_axis

    def _generate_patterns_for_files(
        self,
        files: List[Any],
        variable_components: List[str],
        axis_value: str
    ) -> List[str]:
        """
        Generate patterns for a list of files.

        Args:
            files: List of file path objects representing files
            variable_components: List of components that can vary in the pattern

        Returns:
            List of pattern strings

        Raises:
            TypeError: If files list is not a list
            ValueError: If pattern templates cannot be instantiated
        """
        # Validate input parameters
        if not isinstance(files, list):
            raise TypeError(f"Expected list of file path objects, got {type(files).__name__}")

        if not isinstance(variable_components, list):
            raise TypeError(f"Expected list of variable components, got {type(variable_components).__name__}")

        # Use microscope-specific parser for pattern generation


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
            # Generate pattern arguments for all discovered components
            pattern_args = {}
            for comp in self.parser.FILENAME_COMPONENTS:
                if comp in template_metadata:
                    if comp in variable_components:
                        pattern_args[comp] = self.PLACEHOLDER_PATTERN
                    else:
                        pattern_args[comp] = template_metadata[comp]

            # Ensure pattern generation succeeded
            if not pattern_args:
                raise ValueError("Clause 93 Violation: No components found in template metadata for pattern generation")

            # Use metaprogramming approach - pass all components dynamically
            extension = pattern_args.get('extension') or DEFAULT_IMAGE_EXTENSION
            component_kwargs = {comp: pattern_args.get(comp) for comp in self.parser.get_component_names() if comp in pattern_args}

            pattern_str = self.parser.construct_filename(
                extension=extension,
                **component_kwargs
            )

            # Validate that the pattern can be instantiated
            if not self.parser.parse_filename(pattern_str):
                raise ValueError(
                    f"Clause 93 Violation: Pattern template '{pattern_str}' cannot be instantiated"
                )

            patterns.append(pattern_str)

        # Validate the final pattern list
        if not patterns:
            raise ValueError(
                "No patterns generated from files. This indicates either: "
                "(1) no image files found in the directory, "
                "(2) files don't match the expected naming convention, or "
                "(3) pattern generation logic failed. "
                "Check that image files exist and follow the expected naming pattern."
            )

        return patterns
