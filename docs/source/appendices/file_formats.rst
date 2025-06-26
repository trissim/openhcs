.. _file-formats:

File Formats
===========

This appendix provides technical specifications for file formats and directory structures supported by EZStitcher.

.. _image-file-formats:

Image File Formats
------------------

EZStitcher supports the following image file formats:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Format
     - Extensions
     - Description
   * - TIFF
     - .tif, .tiff
     - Tagged Image File Format, the primary format for microscopy images. EZStitcher currently works with 16-bit TIFF images only.
   * - JPEG
     - .jpg, .jpeg
     - Joint Photographic Experts Group format, a compressed image format. Not recommended for scientific images due to lossy compression.
   * - PNG
     - .png
     - Portable Network Graphics format, a lossless compressed image format.

Bit Depth Support
~~~~~~~~~~~~~~~~~~

EZStitcher currently supports only 16-bit images (uint16, values from 0-65535). Support for 8-bit and 32-bit images may be added in future versions.

.. _position-files:

Position Files
--------------

Position files are CSV files with the following format:

.. code-block:: text

    filename,x,y
    A01_s1_w1.tif,0.0,0.0
    A01_s2_w1.tif,1024.5,0.0
    A01_s3_w1.tif,2049.2,0.0
    A01_s4_w1.tif,0.0,1024.3
    ...

Where:

- **filename**: The filename of the tile
- **x, y**: Pixel coordinates in the final stitched image (floating-point values for subpixel precision)

Alternative format with grid positions:

.. code-block:: text

    file,i,j,x,y
    A01_s1_w1.tif,0,0,0.0,0.0
    A01_s2_w1.tif,1,0,1024.5,0.0
    A01_s3_w1.tif,0,1,0.0,1024.5
    A01_s4_w1.tif,1,1,1024.5,1024.5
    ...

Where:

- **file**: The filename of the tile
- **i, j**: Grid coordinates (column, row)
- **x, y**: Pixel coordinates in the final stitched image

.. _metadata-formats:

Metadata Formats
---------------

EZStitcher extracts metadata from microscope-specific files:

ImageXpress Metadata
^^^^^^^^^^^^^^^^^^^

ImageXpress metadata is stored in HTD files (text-based) or XML files with the following structure:

.. code-block:: xml

    <MetaData>
      <PlateType>
        <SiteRows>3</SiteRows>
        <SiteColumns>3</SiteColumns>
      </PlateType>
      <ImageSize>
        <PixelWidthUM>0.65</PixelWidthUM>
      </ImageSize>
    </MetaData>

HTD files have a similar structure but in a text-based format:

.. code-block:: text

    [General]
    Plate Type=96 Well
    ...
    [Sites]
    SiteCount=9
    GridRows=3
    GridColumns=3
    ...
    [Wavelengths]
    WavelengthCount=3
    ...
    [Scale]
    PixelSize=0.65
    ...

Opera Phenix Metadata
^^^^^^^^^^^^^^^^^^^

Opera Phenix metadata is stored in the Index.xml file:

.. code-block:: xml

    <EvaluationInputData>
      <Plates>
        <Plate>
          <PlateID>plate_name</PlateID>
          <PlateTypeName>96well</PlateTypeName>
        </Plate>
      </Plates>
      <Images>
        <Image id="r01c01f001p01-ch1sk1fk1fl1">
          <URL>Images/r01c01f001p01-ch1sk1fk1fl1.tiff</URL>
          <PositionX>0.0</PositionX>
          <PositionY>0.0</PositionY>
          <ImageResolutionX>0.65</ImageResolutionX>
          <ImageResolutionY>0.65</ImageResolutionY>
        </Image>
      </Images>
    </EvaluationInputData>

.. _output-file-structure:

Output File Structure
-------------------

EZStitcher creates a dynamic directory structure during processing. By default, it follows this pattern:

.. code-block:: text

    plate_folder/                 # Original data
    plate_folder_workspace/       # Workspace with symlinks to original images
    plate_folder_workspace_out/   # Processed individual tiles
    plate_folder_workspace_positions/  # CSV files with stitching positions
    plate_folder_workspace_stitched/   # Final stitched images

For comprehensive information on directory structure and management in EZStitcher, including:

- Default directory structure
- Directory resolution logic
- Step initialization best practices
- Custom directory structures
- When to specify directories explicitly
- Common mistakes to avoid

See :doc:`../concepts/directory_structure`.

However, the actual directory structure is determined by the specific steps in your pipeline. Each step can specify its own input and output directories, and the pipeline will create them as needed.

File Naming Conventions
---------------------

For detailed information about file naming conventions for different microscope types, see the :doc:`microscope_formats` appendix.
