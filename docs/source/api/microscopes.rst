Microscope Implementations
========================

This module contains implementations of microscope-specific functionality.

ImageXpress
----------

.. module:: ezstitcher.microscopes.imagexpress

.. py:class:: ImageXpressFilenameParser

   Filename parser for ImageXpress microscopes.

   .. py:classmethod:: can_parse(cls, filename)

      Check if this parser can parse the given filename.

      :param filename: Filename to check
      :type filename: str
      :return: True if this parser can parse the filename, False otherwise
      :rtype: bool

   .. py:method:: parse_filename(filename)

      Parse an ImageXpress filename into its components.

      :param filename: Filename to parse
      :type filename: str
      :return: Dictionary with extracted components or None if parsing fails
      :rtype: dict or None

   .. py:method:: construct_filename(well, site=None, channel=None, z_index=None, extension='.tif', site_padding=3, z_padding=3)

      Construct an ImageXpress filename from components.

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

.. py:class:: ImageXpressMetadataHandler

   Metadata handler for ImageXpress microscopes.

   .. py:method:: find_metadata_file(plate_path)

      Find the metadata file for an ImageXpress plate.

      :param plate_path: Path to the plate folder
      :type plate_path: str or Path
      :return: Path to the metadata file, or None if not found
      :rtype: Path or None

   .. py:method:: get_grid_dimensions(plate_path)

      Get grid dimensions for stitching from ImageXpress metadata.

      :param plate_path: Path to the plate folder
      :type plate_path: str or Path
      :return: Tuple of (grid_size_x, grid_size_y)
      :rtype: tuple

   .. py:method:: get_pixel_size(plate_path)

      Get the pixel size from ImageXpress metadata.

      :param plate_path: Path to the plate folder
      :type plate_path: str or Path
      :return: Pixel size in micrometers, or None if not available
      :rtype: float or None

Opera Phenix
-----------

.. module:: ezstitcher.microscopes.opera_phenix

.. py:class:: OperaPhenixFilenameParser

   Filename parser for Opera Phenix microscopes.

   .. py:classmethod:: can_parse(cls, filename)

      Check if this parser can parse the given filename.

      :param filename: Filename to check
      :type filename: str
      :return: True if this parser can parse the filename, False otherwise
      :rtype: bool

   .. py:method:: parse_filename(filename)

      Parse an Opera Phenix filename into its components.

      :param filename: Filename to parse
      :type filename: str
      :return: Dictionary with extracted components or None if parsing fails
      :rtype: dict or None

   .. py:method:: construct_filename(well, site=None, channel=None, z_index=None, extension='.tiff', site_padding=1, z_padding=1)

      Construct an Opera Phenix filename from components.

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

.. py:class:: OperaPhenixMetadataHandler

   Metadata handler for Opera Phenix microscopes.

   .. py:method:: find_metadata_file(plate_path)

      Find the metadata file for an Opera Phenix plate.

      :param plate_path: Path to the plate folder
      :type plate_path: str or Path
      :return: Path to the metadata file, or None if not found
      :rtype: Path or None

   .. py:method:: get_grid_dimensions(plate_path)

      Get grid dimensions for stitching from Opera Phenix metadata.

      :param plate_path: Path to the plate folder
      :type plate_path: str or Path
      :return: Tuple of (grid_size_x, grid_size_y)
      :rtype: tuple

   .. py:method:: get_pixel_size(plate_path)

      Get the pixel size from Opera Phenix metadata.

      :param plate_path: Path to the plate folder
      :type plate_path: str or Path
      :return: Pixel size in micrometers, or None if not available
      :rtype: float or None
