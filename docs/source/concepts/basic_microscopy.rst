=============================
Basic Microscopy Concepts
=============================

Introduction to Microscopy Image Stitching
------------------------------------------

Microscopy image stitching is the process of combining multiple overlapping images (tiles) into a single larger image. This is necessary because:

1. The field of view of a microscope is limited
2. High-resolution imaging often requires capturing multiple tiles
3. Stitching allows visualization and analysis of larger areas

EZStitcher addresses these challenges by:

- Automatically detecting tile positions
- Aligning tiles with subpixel precision
- Blending overlapping regions smoothly
- Handling multi-channel fluorescence and Z-stacks

Key Microscopy Concepts
-----------------------

Plate-Based Experiments
~~~~~~~~~~~~~~~~~~~~~~~

In high-content screening, samples are typically organized in plates:

.. image:: ../_static/plate_diagram.png
   :width: 400
   :alt: Plate Diagram

Key concepts:

- **Plate**: A container with multiple wells (e.g., 96-well plate)
- **Well**: A single compartment in a plate, identified by a row letter and column number (e.g., A01, B02)
- **Site**: A specific location within a well where an image is captured
- **Grid**: The arrangement of sites within a well (e.g., 3×3 grid)

EZStitcher processes images on a per-well basis, stitching together all sites within each well.

Multi-Channel Fluorescence
~~~~~~~~~~~~~~~~~~~~~~~~~

Fluorescence microscopy captures images at different wavelengths to visualize different structures:

.. image:: ../_static/multichannel_diagram.png
   :width: 400
   :alt: Multi-Channel Diagram

Key concepts:

- **Channel**: A specific wavelength or color used for imaging
- **Composite**: A combined image created from multiple channels

EZStitcher can:

- Process each channel independently
- Create composite images from multiple channels
- Use one channel as a reference for stitching all channels

Z-Stacks
~~~~~~~~

Z-stacks are 3D image stacks captured at different focal planes:

.. image:: ../_static/zstack_diagram.png
   :width: 400
   :alt: Z-Stack Diagram

Key concepts:

- **Z-Stack**: A series of images captured at different focal planes
- **Z-Plane**: A single image at a specific focal depth
- **Projection**: A 2D representation of a 3D stack (e.g., maximum intensity projection)
- **Best Focus**: The plane with the highest focus quality

EZStitcher provides several options for handling Z-stacks:

- Maximum intensity projection
- Mean projection
- Best focus selection
- Per-plane stitching

For detailed information about Z-stack processing, see the :doc:`../user_guide/intermediate_usage` guide.

Tiled Images
~~~~~~~~~~~

Tiled images are multiple overlapping images that cover a larger area:

.. image:: ../_static/tiling_diagram.png
   :width: 400
   :alt: Tiling Diagram

Key concepts:

- **Tile**: A single image captured at a specific position
- **Overlap**: The region where adjacent tiles overlap
- **Grid Size**: The number of tiles in X and Y directions
- **Position**: The coordinates of a tile in the final stitched image

EZStitcher handles tiled images by:

1. Determining the relative positions of tiles
2. Aligning tiles with subpixel precision
3. Blending overlapping regions
4. Assembling the final stitched image
