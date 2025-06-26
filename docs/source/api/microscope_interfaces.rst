Microscope Interfaces
=====================

.. module:: ezstitcher.core.microscope_interfaces

This module provides abstract base classes for handling microscope-specific functionality, including filename parsing and metadata handling.

For conceptual explanation, see :doc:`../concepts/architecture_overview`.
For information about extending EZStitcher with support for new microscope types, see :ref:`extending-microscope-types` in the :doc:`../development/extending` guide.

MicroscopeHandler
-----------------

.. py:class:: MicroscopeHandler(plate_folder=None, parser=None, metadata_handler=None, microscope_type='auto')

   Composed class for handling microscope-specific functionality.

   :param plate_folder: Path to the plate folder
   :type plate_folder: str or Path, optional
   :param parser: Parser for microscopy filenames
   :type parser: FilenameParser, optional
   :param metadata_handler: Handler for microscope metadata
   :type metadata_handler: MetadataHandler, optional
   :param microscope_type: Type of microscope ('auto', 'ImageXpress', 'OperaPhenix', etc.)
   :type microscope_type: str, optional

   .. py:attribute:: DEFAULT_MICROSCOPE
      :type: str
      :value: 'ImageXpress'

      Default microscope type to use if auto-detection fails.

   .. py:method:: parse_filename(filename)

      Delegate to parser.

      :param filename: Filename to parse
      :type filename: str
      :return: Dictionary with extracted components or None if parsing fails
      :rtype: dict or None

   .. py:method:: construct_filename(well, site=None, channel=None, z_index=None, extension='.tif', site_padding=3, z_padding=3)

      Delegate to parser.

      :param well: Well ID (e.g., 'A01')
      :type well: str
      :param site: Site number or placeholder string (e.g., '{iii}')
      :type site: int or str, optional
      :param channel: Channel/wavelength number
      :type channel: int, optional
      :param z_index: Z-index or placeholder string (e.g., '{zzz}')
      :type z_index: int or str, optional
      :param extension: File extension
      :type extension: str, optional
      :param site_padding: Width to pad site numbers to
      :type site_padding: int, optional
      :param z_padding: Width to pad Z-index numbers to
      :type z_padding: int, optional
      :return: Constructed filename
      :rtype: str

   .. py:method:: auto_detect_patterns(folder_path, well_filter=None, extensions=None, group_by='channel', variable_components=None)

      Delegate to parser.

      :param folder_path: Path to the folder
      :type folder_path: str or Path
      :param well_filter: Optional list of wells to include
      :type well_filter: list, optional
      :param extensions: Optional list of file extensions to include
      :type extensions: list, optional
      :param group_by: How to group patterns ('channel' or 'z_index')
      :type group_by: str, optional
      :param variable_components: List of components to make variable (e.g., ['site', 'z_index'])
      :type variable_components: list, optional
      :return: Dictionary mapping wells to patterns grouped by channel or z-index
      :rtype: dict

   .. py:method:: path_list_from_pattern(directory, pattern)

      Delegate to parser.

      :param directory: Directory to search
      :type directory: str or Path
      :param pattern: Pattern to match with {iii} placeholder for site index
      :type pattern: str
      :return: List of matching filenames
      :rtype: list

   .. py:method:: find_metadata_file(plate_path)

      Delegate to metadata handler.

      :param plate_path: Path to the plate folder
      :type plate_path: str or Path
      :return: Path to the metadata file, or None if not found
      :rtype: Path or None

   .. py:method:: get_grid_dimensions(plate_path)

      Delegate to metadata handler.

      :param plate_path: Path to the plate folder
      :type plate_path: str or Path
      :return: Tuple of (grid_size_x, grid_size_y)
      :rtype: tuple

   .. py:method:: get_pixel_size(plate_path)

      Delegate to metadata handler.

      :param plate_path: Path to the plate folder
      :type plate_path: str or Path
      :return: Pixel size in micrometers, or None if not available
      :rtype: float or None

FilenameParser
--------------

.. py:class:: FilenameParser
   :noindex:

   Abstract base class for parsing microscopy image filenames.

   .. py:attribute:: FILENAME_COMPONENTS
      :type: list
      :value: ['well', 'site', 'channel', 'z_index', 'extension']

      List of components that can be extracted from filenames.

   .. py:attribute:: PLACEHOLDER_PATTERN
      :type: str
      :value: '{iii}'

      Placeholder pattern for variable components.

   .. py:classmethod:: can_parse(cls, filename)

      Check if this parser can parse the given filename.

      :param filename: Filename to check
      :type filename: str
      :return: True if this parser can parse the filename, False otherwise
      :rtype: bool

   .. py:method:: parse_filename(filename)

      Parse a microscopy image filename to extract all components.

      :param filename: Filename to parse
      :type filename: str
      :return: Dictionary with extracted components or None if parsing fails
      :rtype: dict or None

   .. py:method:: construct_filename(well, site=None, channel=None, z_index=None, extension='.tif', site_padding=3, z_padding=3)

      Construct a filename from components.

      :param well: Well ID (e.g., 'A01')
      :type well: str
      :param site: Site number or placeholder string (e.g., '{iii}')
      :type site: int or str, optional
      :param channel: Channel/wavelength number
      :type channel: int, optional
      :param z_index: Z-index or placeholder string (e.g., '{zzz}')
      :type z_index: int or str, optional
      :param extension: File extension
      :type extension: str, optional
      :param site_padding: Width to pad site numbers to
      :type site_padding: int, optional
      :param z_padding: Width to pad Z-index numbers to
      :type z_padding: int, optional
      :return: Constructed filename
      :rtype: str

   .. py:method:: path_list_from_pattern(directory, pattern)

      Get a list of filenames matching a pattern in a directory.

      :param directory: Directory to search
      :type directory: str or Path
      :param pattern: Pattern to match with {iii} placeholder for site index
      :type pattern: str
      :return: List of matching filenames
      :rtype: list

   .. py:method:: group_patterns_by_component(patterns, component='channel', default_value='1')

      Group patterns by a specific component (channel, z_index, site, well, etc.)

      :param patterns: List of patterns to group
      :type patterns: list
      :param component: Component to group by (e.g., 'channel', 'z_index', 'site', 'well')
      :type component: str, optional
      :param default_value: Default value to use if component is not found
      :type default_value: str, optional
      :return: Dictionary mapping component values to patterns
      :rtype: dict

   .. py:method:: auto_detect_patterns(folder_path, well_filter=None, extensions=None, group_by='channel', variable_components=None)

      Automatically detect image patterns in a folder.

      :param folder_path: Path to the folder
      :type folder_path: str or Path
      :param well_filter: Optional list of wells to include
      :type well_filter: list, optional
      :param extensions: Optional list of file extensions to include
      :type extensions: list, optional
      :param group_by: How to group patterns ('channel' or 'z_index')
      :type group_by: str, optional
      :param variable_components: List of components to make variable (e.g., ['site', 'z_index'])
      :type variable_components: list, optional
      :return: Dictionary mapping wells to patterns grouped by channel or z-index
      :rtype: dict

MetadataHandler
---------------

.. py:class:: MetadataHandler
   :noindex:

   Abstract base class for handling microscope metadata.

   .. py:method:: find_metadata_file(plate_path)

      Find the metadata file for a plate.

      :param plate_path: Path to the plate folder
      :type plate_path: str or Path
      :return: Path to the metadata file, or None if not found
      :rtype: Path or None

   .. py:method:: get_grid_dimensions(plate_path)

      Get grid dimensions for stitching from metadata.

      :param plate_path: Path to the plate folder
      :type plate_path: str or Path
      :return: Tuple of (grid_size_x, grid_size_y)
      :rtype: tuple

   .. py:method:: get_pixel_size(plate_path)

      Get the pixel size from metadata.

      :param plate_path: Path to the plate folder
      :type plate_path: str or Path
      :return: Pixel size in micrometers, or None if not available
      :rtype: float or None

Functions
--------

.. py:function:: create_microscope_handler(microscope_type='auto', **kwargs)

   Create the appropriate microscope handler.

   :param microscope_type: Type of microscope ('auto', 'ImageXpress', 'OperaPhenix', etc.)
   :type microscope_type: str, optional
   :param kwargs: Additional keyword arguments to pass to MicroscopeHandler
   :type kwargs: dict
   :return: Microscope handler
   :rtype: MicroscopeHandler
