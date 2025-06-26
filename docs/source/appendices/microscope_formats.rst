.. _microscope-formats:

Microscope Formats
==================

This page describes the file naming conventions and directory structures for different microscope types supported by EZStitcher.

.. _microscope-imagexpress:

ImageXpress
-----------

ImageXpress microscopes use the following file naming convention:

.. code-block:: text

    <well>_s<site>_w<channel>[_z<z_index>].tif

For example:

- ``A01_s1_w1.tif``: Well A01, Site 1, Channel 1
- ``A01_s1_w2.tif``: Well A01, Site 1, Channel 2
- ``A01_s2_w1.tif``: Well A01, Site 2, Channel 1
- ``A01_s1_w1_z1.tif``: Well A01, Site 1, Channel 1, Z-index 1

Directory Structure
^^^^^^^^^^^^^^^^^^^

ImageXpress typically organizes files in the following structure:

.. code-block:: text

    plate_folder/
    ├── TimePoint_1/
    │   ├── A01_s1_w1.tif
    │   ├── A01_s1_w2.tif
    │   ├── A01_s2_w1.tif
    │   └── ...
    └── ...

For Z-stacks, the structure is typically:

.. code-block:: text

    plate_folder/
    ├── TimePoint_1/
    │   ├── ZStep_1/
    │   │   ├── A01_s1_w1.tif
    │   │   └── ...
    │   ├── ZStep_2/
    │   │   ├── A01_s1_w1.tif
    │   │   └── ...
    │   └── ...
    └── ...

Alternatively, Z-stacks may be organized with Z-index in the filename:

.. code-block:: text

    plate_folder/
    ├── TimePoint_1/
    │   ├── A01_s1_w1_z1.tif
    │   ├── A01_s1_w1_z2.tif
    │   ├── A01_s1_w2_z1.tif
    │   └── ...
    └── ...

Metadata
^^^^^^^^

ImageXpress metadata is stored in HTD (Hardware Test Definition) files with names like:

- ``<plate_name>.HTD``
- ``<plate_name>_meta.HTD``
- ``MetaData/<plate_name>.HTD``

The metadata contains information about:

- Grid dimensions (number of sites in x and y directions)
- Acquisition settings

Pixel size information is typically stored in the TIFF files themselves.

.. _microscope-opera-phenix:

Opera Phenix
------------

Opera Phenix microscopes use the following file naming convention:

.. code-block:: text

    r<row>c<col>f<field>p<plane>-ch<channel>sk1fk1fl1.tiff

For example:

- ``r01c01f001p01-ch1sk1fk1fl1.tiff``: Well R01C01, Channel 1, Field 1, Plane 1
- ``r01c01f001p01-ch2sk1fk1fl1.tiff``: Well R01C01, Channel 2, Field 1, Plane 1
- ``r01c01f002p01-ch1sk1fk1fl1.tiff``: Well R01C01, Channel 1, Field 2, Plane 1
- ``r01c01f001p02-ch1sk1fk1fl1.tiff``: Well R01C01, Channel 1, Field 1, Plane 2

Components:

- ``r<row>c<col>``: Well identifier (r01c01 = R01C01, r02c03 = R02C03, etc.)
- ``f<field>``: Field/site number (f001, f002, etc.)
- ``p<plane>``: Z-plane number (p01, p02, etc.)
- ``ch<channel>``: Channel number (ch1, ch2, etc.)

Note: The prefixes ``r``, ``c``, ``f``, ``p``, and ``ch`` are fixed parts of the filename format and should always be lowercase. The suffixes ``sk1``, ``fk1``, and ``fl1`` are fixed values that represent sequence ID, timepoint ID, and flim ID respectively. These are always expected to be 1 and are not supported as variable components.

Directory Structure
^^^^^^^^^^^^^^^^^^^

Opera Phenix typically organizes files in the following structure:

.. code-block:: text

    plate_folder/
    ├── Images/
    │   ├── r01c01f001p01-ch1sk1fk1fl1.tiff
    │   ├── r01c01f002p01-ch1sk1fk1fl1.tiff
    │   ├── r01c01f003p01-ch1sk1fk1fl1.tiff
    │   └── ...
    ├── Index.xml
    └── ...

Metadata
^^^^^^^^

Opera Phenix metadata is stored in XML files with names like:

- ``Index.xml``
- ``MeasurementDetail.xml``

The metadata contains information about:

- Image resolution (pixel size)
- Position coordinates for each field
- Acquisition settings

.. _microscope-automatic-detection:

Automatic Detection
-------------------

EZStitcher can automatically detect the microscope type based on the file structure and naming conventions:

.. code-block:: python

    from ezstitcher.core.microscope_interfaces import MicroscopeHandler
    from pathlib import Path

    plate_folder = Path("path/to/plate_folder")
    handler = MicroscopeHandler(plate_folder=plate_folder)
    print(f"Detected microscope type: {handler.__class__.__name__}")

The detection algorithm:

1. Examines the directory structure
2. Checks for characteristic metadata files
3. Samples image filenames and tries to parse them with different parsers
4. Selects the most likely microscope type based on the results

.. _microscope-adding-support:

Adding Support for New Microscopes
----------------------------------

To add support for a new microscope type:

1. Create a new file in the `ezstitcher/microscopes/` directory
2. Implement the `FilenameParser` and `MetadataHandler` interfaces
3. Register the new microscope type in `ezstitcher/microscopes/__init__.py`

See the :doc:`../development/extending` section for details.

.. _microscope-comparison:

Comparison of Microscope Formats
--------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Feature
     - ImageXpress
     - Opera Phenix
   * - File Extension
     - .tif
     - .tiff
   * - Well Format
     - A01, B02, etc.
     - R01C01, R02C02, etc. (stored as r01c01, r02c02 in filenames)
   * - Channel Identifier
     - w1, w2, etc.
     - ch1, ch2, etc. (lowercase 'ch')
   * - Site/Field Identifier
     - s1, s2, etc.
     - f1, f2, etc. (lowercase 'f')
   * - Z-Stack Organization
     - ZStep folders or _z suffix
     - p1, p2, etc. in filename (lowercase 'p')
   * - Metadata Format
     - HTD files with SiteRows/SiteColumns
     - XML with PositionX/Y coordinates
   * - Pixel Size Location
     - TIFF file metadata
     - ImageResolutionX/Y elements in XML
