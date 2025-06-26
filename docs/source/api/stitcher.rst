Stitcher
========

.. module:: ezstitcher.core.stitcher

This module contains the Stitcher class for handling image stitching operations.

For comprehensive information about the stitching process, including:

* Position generation
* Image assembly
* Typical stitching workflows
* Best practices for stitching

See the ImageStitchingStep class in the :doc:`steps` documentation.

Stitcher
-------

.. py:class:: Stitcher(config=None, filename_parser=None)

   Class for handling image stitching operations.

   :param config: Configuration for stitching
   :type config: :class:`~ezstitcher.core.config.StitcherConfig`, optional
   :param filename_parser: Parser for microscopy filenames
   :type filename_parser: :class:`~ezstitcher.core.microscope_interfaces.FilenameParser`, optional

   .. py:method:: generate_positions(image_dir, image_pattern, positions_path, grid_size_x, grid_size_y)

      Generate positions for stitching using Ashlar.

      :param image_dir: Directory containing images
      :type image_dir: str or Path
      :param image_pattern: Pattern with '{iii}' placeholder
      :type image_pattern: str
      :param positions_path: Path to save positions CSV
      :type positions_path: str or Path
      :param grid_size_x: Number of tiles horizontally
      :type grid_size_x: int
      :param grid_size_y: Number of tiles vertically
      :type grid_size_y: int
      :return: True if successful, False otherwise
      :rtype: bool

   .. py:method:: assemble_image(positions_path, images_dir, output_path, override_names=None)

      Assemble a stitched image using subpixel positions from a CSV file.

      :param positions_path: Path to the CSV with subpixel positions
      :type positions_path: str or Path
      :param images_dir: Directory containing image tiles
      :type images_dir: str or Path
      :param output_path: Path to save final stitched image
      :type output_path: str or Path
      :param override_names: Optional list of filenames to use instead of those in CSV
      :type override_names: list, optional
      :return: True if successful, False otherwise
      :rtype: bool

   .. py:method:: generate_positions_df(image_dir, image_pattern, positions, grid_size_x, grid_size_y)

      Given an image_dir, an image_pattern (with '{iii}' or similar placeholder)
      and a list of (x, y) tuples 'positions', build a DataFrame with lines like:
      file: <filename>; position: (x, y); grid: (col, row);

      :param image_dir: Directory containing images
      :type image_dir: str or Path
      :param image_pattern: Pattern with '{iii}' placeholder
      :type image_pattern: str
      :param positions: List of (x, y) positions
      :type positions: list
      :param grid_size_x: Number of tiles horizontally
      :type grid_size_x: int
      :param grid_size_y: Number of tiles vertically
      :type grid_size_y: int
      :return: DataFrame with positions
      :rtype: pandas.DataFrame

   .. py:staticmethod:: parse_positions_csv(csv_path)

      Parse a CSV file with lines of the form:
      file: <filename>; grid: (col, row); position: (x, y)

      :param csv_path: Path to the CSV file
      :type csv_path: str or Path
      :return: List of (filename, x, y) tuples
      :rtype: list

   .. py:staticmethod:: save_positions_df(df, positions_path)

      Save a positions DataFrame to CSV.

      :param df: DataFrame to save
      :type df: pandas.DataFrame
      :param positions_path: Path to save the CSV file
      :type positions_path: str or Path
      :return: True if successful, False otherwise
      :rtype: bool

StitcherConfig
------------

.. py:class:: StitcherConfig

   Configuration for the Stitcher class.

   .. py:attribute:: tile_overlap
      :type: float
      :value: 10.0

      Percentage overlap between tiles.

   .. py:attribute:: max_shift
      :type: int
      :value: 50

      Maximum allowed shift in pixels.

   .. py:attribute:: margin_ratio
      :type: float
      :value: 0.1

      Ratio of image size to use as margin for blending.

   .. py:attribute:: pixel_size
      :type: float
      :value: 1.0

      Pixel size in micrometers.
