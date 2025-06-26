Glossary
========

.. glossary::
   :sorted:

   Ashlar
      A Python library for stitching microscopy images with subpixel precision.

   Best Focus
      The plane in a Z-stack that has the highest focus quality according to a focus metric.

   Channel
      A specific wavelength or color channel in a microscopy image, often corresponding to a specific fluorophore or stain.

   Composite Image
      An image created by combining multiple channels, often with different weights.

   Focus Metric
      A mathematical measure of image focus quality, such as normalized variance, Laplacian energy, or Tenengrad variance.

   Grid Size
      The number of tiles in the x and y directions in a microscopy acquisition.

   ImageXpress
      A high-content screening platform from Molecular Devices that generates microscopy images with a specific file naming convention.

   Max Projection
      A method of combining a Z-stack by taking the maximum intensity value at each pixel position across all planes.

   Mean Projection
      A method of combining a Z-stack by taking the average intensity value at each pixel position across all planes.

   Metadata
      Information about the microscopy acquisition, such as pixel size, grid dimensions, and acquisition settings.

   Microscope Handler
      A component in EZStitcher that handles microscope-specific functionality, such as filename parsing and metadata extraction.

   Opera Phenix
      A high-content screening platform from PerkinElmer that generates microscopy images with a specific file naming convention.

   Pixel Size
      The physical size of a pixel in the microscopy image, typically measured in micrometers.

   Plate
      A container for samples in high-content screening, typically with multiple wells arranged in a grid.

   Position Generation
      The process of determining the relative positions of tiles for stitching.

   Preprocessing
      Image processing operations applied to individual tiles before stitching, such as background subtraction or histogram equalization.

   Reference Channel
      The channel used for position generation in stitching.

   Site
      A specific field of view or tile in a microscopy acquisition.

   Stitching
      The process of combining multiple overlapping images into a single larger image.

   Subpixel Precision
      Alignment of images with precision finer than a single pixel, achieved through interpolation.

   Tile
      A single image in a grid of images that will be stitched together.

   Tile Overlap
      The percentage of overlap between adjacent tiles in a microscopy acquisition.

   Well
      A specific location in a plate where a sample is placed, typically identified by a letter-number combination (e.g., A01).

   Z-Stack
      A series of images taken at different focal planes along the z-axis, capturing 3D information about a sample.

   Z-Index
      The index or position of a plane in a Z-stack.
