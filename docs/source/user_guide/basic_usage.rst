===========
Basic Usage
===========

This page provides an overview of how to use EZStitcher for basic image stitching tasks. If you're looking for a quick start guide, see :doc:`/getting_started/getting_started`.

Three-Tier Approach
------------------

EZStitcher offers three main approaches for creating stitching pipelines, each designed for a different level of user experience and need for control:

1. **EZ Module (Beginner Level)**
   
   A simplified, one-liner interface for beginners and non-coders:

   * "I just want to stitch my images quickly"
   * Uses sensible defaults and auto-detection
   * Handles common use cases with a single function call
   * Example: ``stitch_plate("path/to/plate")``


2. **Custom Pipelines with Steps (Intermediate Level)**
   
   More flexibility and control using pre-defined steps:

   * "I need more control over the processing steps"
   * Uses pre-defined steps that provide a clean interface for common operations
   * Allows customization of processing steps and parameters
   * See :doc:`intermediate_usage` for details
   

3. **Library Extension with Base Step (Advanced Level)**
   
   For advanced users who need to understand implementation details:

   * "I need to understand how the steps work under the hood"
   * Uses the base Step class to create custom processing functions
   * Provides maximum flexibility and control
   * See :doc:`advanced_usage` for details


This guide focuses on the EZ Module approach, which is recommended for most users.

Getting Started with EZStitcher
------------------------------

The simplest way to use EZStitcher is through the EZ module, which provides a one-liner function for stitching microscopy images:

.. code-block:: python

   from ezstitcher import stitch_plate

   # Stitch a plate with default settings
   stitch_plate("path/to/plate")

That's it! This single line will:

1. Automatically detect the plate format
2. Process all channels and Z-stacks appropriately
3. Generate positions and stitch images
4. Save the output to a new directory

Key Parameters
--------------

While the default settings work well for most cases, you can customize the behavior with a few key parameters:

.. code-block:: python

   stitch_plate(
       "path/to/plate",                    # Input directory with microscopy images
       output_path="path/to/output",       # Where to save results (optional)
       normalize=True,                     # Apply intensity normalization (default: True)
       flatten_z=True,                     # Flatten Z-stacks to 2D (auto-detected if None)
       z_method="max",                     # How to flatten Z-stacks: "max", "mean", "focus"
       channel_weights=[0.7, 0.3, 0],      # Weights for position finding (auto-detected if None)
       well_filter=["A01", "B02"]          # Process only specific wells (optional)
   )

Z-Stack Processing
------------------

For plates with Z-stacks, you can control how they're flattened:

.. code-block:: python

   # Maximum intensity projection (brightest pixel from each Z-stack)
   stitch_plate("path/to/plate", flatten_z=True, z_method="max")

   # Focus-based projection (selects best-focused plane)
   stitch_plate("path/to/plate", flatten_z=True, z_method="focus")

   # Mean projection (average across Z-planes)
   stitch_plate("path/to/plate", flatten_z=True, z_method="mean")

More Control
------------

For slightly more control while keeping things simple, use the ``EZStitcher`` class:

.. code-block:: python

   from ezstitcher import EZStitcher

   # Create a stitcher
   stitcher = EZStitcher("path/to/plate")

   # Set options
   stitcher.set_options(
       normalize=True,
       z_method="focus"
   )

   # Run stitching
   stitcher.stitch()

Troubleshooting
---------------

**Common issues:**

- **No output**: Check that the input path exists and contains microscopy images
- **Z-stacks not detected**: Explicitly set ``flatten_z=True``
- **Poor quality**: Try different ``z_method`` values or adjust ``channel_weights``

Understanding Key Concepts
------------------------

Here are the key concepts you need to understand for basic usage:

**Plates and Wells**

EZStitcher processes microscopy data organized in plates and wells. A plate contains multiple wells, and each well contains multiple images.

**Images and Channels**

Microscopy images can have multiple channels (e.g., DAPI, GFP, RFP) and Z-stacks (multiple focal planes).

**Processing Steps**

Behind the scenes, EZStitcher processes images through a series of steps:

- Z-flattening: Converting 3D Z-stacks into 2D images
- Normalization: Adjusting image intensity for consistent visualization
- Channel compositing: Combining multiple channels into a single image
- Position generation: Finding the relative positions of tiles
- Image stitching: Combining tiles into a complete image

These steps are organized into two standard pipelines:

1. **Position Generation Pipeline**: Z-flattening → Normalization → Channel compositing → Position generation
2. **Assembly Pipeline**: Normalization → Image stitching

The EZ module handles all these steps automatically, so you don't need to worry about them unless you need more control.

For more detailed information about EZStitcher's architecture and concepts, see :doc:`../concepts/architecture_overview` and the :doc:`../concepts/index` section.

When You Need More Control
------------------------

If you need more flexibility than the EZ module provides:

1. First, explore all the options available in the EZ module (see the Key Parameters section above)
2. If you still need more control, see :doc:`intermediate_usage` to learn how to create custom pipelines with steps
3. For even more advanced usage, see :doc:`advanced_usage` for understanding implementation details

For detailed API documentation of the EZ module, see :doc:`../api/ez`.
