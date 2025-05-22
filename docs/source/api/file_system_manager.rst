File System Manager
==================

.. module:: ezstitcher.core.file_system_manager

This module provides a class for managing file system operations.

FileSystemManager
---------------

.. py:class:: FileSystemManager

   Manages file system operations for ezstitcher.
   Abstracts away direct file system interactions for improved testability.

   .. py:attribute:: default_extensions
      :type: list
      :value: ['.tif', '.TIF', '.tiff', '.TIFF', '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']

      Default file extensions for image files.

   .. py:staticmethod:: ensure_directory(directory)

      Ensure a directory exists, creating it if necessary.

      :param directory: Directory path to ensure exists
      :type directory: str or Path
      :return: Path object for the directory
      :rtype: Path

   .. py:staticmethod:: list_image_files(directory, extensions=None, recursive=False, flatten=False)

      List all image files in a directory with specified extensions.

      :param directory: Directory to search
      :type directory: str or Path
      :param extensions: List of file extensions to include
      :type extensions: list, optional
      :param recursive: Whether to search recursively
      :type recursive: bool
      :param flatten: Whether to flatten Z-stack directories (implies recursive)
      :type flatten: bool
      :return: List of Path objects for image files
      :rtype: list

   .. py:staticmethod:: load_image(file_path)

      Load an image. Only 2D images are supported.

      :param file_path: Path to the image file
      :type file_path: str or Path
      :return: 2D image or None if loading fails
      :rtype: numpy.ndarray or None
      :raises: ValueError: If the image is 3D (not supported)

   .. py:staticmethod:: save_image(file_path, image, compression=None)

      Save an image to disk.

      :param file_path: Path to save the image
      :type file_path: str or Path
      :param image: Image to save
      :type image: numpy.ndarray
      :param compression: Compression method
      :type compression: str, optional
      :return: True if successful, False otherwise
      :rtype: bool

   .. py:staticmethod:: copy_file(source_path, dest_path)

      Copy a file from source to destination, preserving metadata.

      :param source_path: Source file path
      :type source_path: str or Path
      :param dest_path: Destination file path
      :type dest_path: str or Path
      :return: True if successful, False otherwise
      :rtype: bool

   .. py:staticmethod:: remove_directory(directory_path, recursive=True)

      Remove a directory and optionally all its contents.

      :param directory_path: Path to the directory to remove
      :type directory_path: str or Path
      :param recursive: Whether to remove the directory recursively
      :type recursive: bool
      :return: True if successful, False otherwise
      :rtype: bool

   .. py:staticmethod:: clean_temp_folders(parent_dir, base_name, keep_suffixes=None)

      Clean up temporary folders created during processing.

      :param parent_dir: Parent directory
      :type parent_dir: str or Path
      :param base_name: Base name of the plate folder
      :type base_name: str
      :param keep_suffixes: List of suffixes to keep
      :type keep_suffixes: list, optional

   .. py:staticmethod:: create_output_directories(plate_path, suffixes)

      Create output directories for a plate.

      :param plate_path: Path to plate folder
      :type plate_path: str or Path
      :param suffixes: Dictionary mapping directory types to suffixes
      :type suffixes: dict
      :return: Dictionary mapping directory types to Path objects
      :rtype: dict

   .. py:staticmethod:: find_file_recursive(directory, filename)

      Recursively search for a file by name in a directory and its subdirectories.
      Returns the first instance found.

      :param directory: Directory to search in
      :type directory: str or Path
      :param filename: Name of the file to find
      :type filename: str
      :return: Path to the first instance of the file, or None if not found
      :rtype: Path or None

   .. py:staticmethod:: rename_files_with_consistent_padding(directory, parser=None, width=3, force_suffixes=False)

      Rename files in a directory to have consistent site number and Z-index padding.
      Optionally force the addition of missing optional suffixes (site, channel, z-index).

      :param directory: Directory containing files to rename
      :type directory: str or Path
      :param parser: Parser to use for filename parsing and padding
      :type parser: FilenameParser, optional
      :param width: Width to pad site numbers to
      :type width: int, optional
      :param force_suffixes: If True, add missing optional suffixes with default values
      :type force_suffixes: bool, optional
      :return: Dictionary mapping original filenames to new filenames
      :rtype: dict

   .. py:staticmethod:: find_z_stack_dirs(root_dir, pattern="ZStep_\\d+", recursive=True)

      Find directories matching a pattern (default: ZStep_#) recursively.

      :param root_dir: Root directory to start the search
      :type root_dir: str or Path
      :param pattern: Regex pattern to match directory names (default: pattern for Z-step folders)
      :type pattern: str
      :param recursive: Whether to search recursively in subdirectories
      :type recursive: bool
      :return: List of (z_index, directory) tuples where z_index is extracted from the pattern
      :rtype: list

   .. py:staticmethod:: find_image_directory(plate_folder, extensions=None)

      Find the directory where images are actually located.

      Handles both cases:
      1. Images directly in a folder (returns that folder)
      2. Images split across Z-step folders (returns parent of Z-step folders)

      :param plate_folder: Base directory to search
      :type plate_folder: str or Path
      :param extensions: List of file extensions to include. If None, uses default_extensions.
      :type extensions: list, optional
      :return: Path to the directory containing images
      :rtype: Path

   .. py:staticmethod:: detect_zstack_folders(plate_folder, pattern=None)

      Detect Z-stack folders in a plate folder.

      :param plate_folder: Path to the plate folder
      :type plate_folder: str or Path
      :param pattern: Regex pattern to match Z-stack folders
      :type pattern: str or Pattern, optional
      :return: Tuple of (has_zstack, z_folders) where z_folders is a list of (z_index, folder_path) tuples
      :rtype: tuple

   .. py:staticmethod:: organize_zstack_folders(plate_folder, filename_parser=None)

      Organize Z-stack folders by moving files to the plate folder with proper naming.

      :param plate_folder: Path to the plate folder
      :type plate_folder: str or Path
      :param filename_parser: Parser for microscopy filenames
      :type filename_parser: FilenameParser, optional
      :return: True if Z-stack was organized, False otherwise
      :rtype: bool

   .. py:staticmethod:: cleanup_processed_files(processed_files, output_files)

      Clean up processed files after they've been used to create output files.

      :param processed_files: Set or list of file paths to clean up
      :type processed_files: set or list
      :param output_files: List of output file paths to preserve
      :type output_files: list
      :return: Number of files successfully removed
      :rtype: int
