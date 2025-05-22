Architecture Overview
=====================

EZStitcher is built on a modular, object-oriented architecture that separates concerns and enables flexible workflows.

Core Components
---------------

.. image:: ../_static/architecture_overview.png
   :alt: EZStitcher Architecture Overview
   :width: 600px

PipelineOrchestrator
^^^^^^^^^^^^^^^^^^^^^

The central coordinator that manages the execution of multiple pipelines across wells. It:

- Initializes and configures all other components
- Manages the flow of data through the pipelines
- Handles high-level operations like well filtering and multithreading
- Coordinates the execution of pipelines for each well

For detailed information on the pipeline architecture, see :doc:`../concepts/architecture_overview`.

MicroscopeHandler
^^^^^^^^^^^^^^^^^

Handles microscope-specific functionality through composition. It:

- Detects the microscope type automatically
- Parses filenames according to microscope-specific patterns
- Extracts metadata from microscope-specific files
- Provides a unified interface for different microscope types

Stitcher
^^^^^^^^

Performs image stitching with subpixel precision. It:

- Generates positions for stitching using Ashlar
- Assembles images using the generated positions
- Handles blending of overlapping regions
- Supports different stitching strategies

FocusAnalyzer
^^^^^^^^^^^^^

Provides multiple focus detection algorithms for Z-stacks as static utility methods. It:

- Implements various focus quality metrics
- Selects the best focused plane in a Z-stack
- Combines multiple metrics for robust focus detection
- Supports both string-based metrics and custom weight dictionaries

ImageProcessor
^^^^^^^^^^^^^^

Handles image normalization, filtering, and compositing. It:

- Provides a library of image processing functions
- Supports both single-image and stack-processing functions
- Creates composite images from multiple channels
- Generates projections from Z-stacks

FileSystemManager
^^^^^^^^^^^^^^^^^

Manages file operations and directory structure. It:

- Handles file loading and saving
- Creates and manages directory structure
- Renames files with consistent padding
- Organizes Z-stack folders
- Locates and organizes images in various directory structures
- Finds images in directories
- Detects Z-stack folders
- Finds images matching patterns
- Determines the main image directory

Design Principles
-----------------

EZStitcher follows these design principles:

1. **Separation of Concerns**: Each component has a specific responsibility
2. **Composition over Inheritance**: Components are composed rather than inherited
3. **Configuration Objects**: Each component has a corresponding configuration class
4. **Static Utility Methods**: Common operations are implemented as static methods
5. **Dependency Injection**: Components are injected into each other
6. **Fail Fast**: Errors are detected and reported as early as possible
7. **Sensible Defaults**: Components have sensible default configurations

Data Flow
---------

The data flow through the pipeline is as follows:

1. **Input**: Raw microscopy images
2. **Preprocessing**: Apply preprocessing functions to individual tiles
3. **Channel Selection/Composition**: Select or compose channels for reference
4. **Z-Stack Flattening**: Flatten Z-stacks using projections or best focus
5. **Position Generation**: Generate stitching positions
6. **Stitching**: Stitch images using the generated positions
7. **Output**: Stitched images

Directory Structure
-------------------

The directory structure of the EZStitcher codebase is as follows:

.. code-block:: text

    ezstitcher/
    ├── core/                  # Core components
    │   ├── __init__.py
    │   ├── config.py          # Configuration classes
    │   ├── file_system_manager.py
    │   ├── focus_analyzer.py
    │   ├── image_processor.py
    │   ├── main.py            # Main entry point
    │   ├── microscope_interfaces.py
    │   ├── pipeline_orchestrator.py
    │   └── stitcher.py
    ├── microscopes/           # Microscope-specific implementations
    │   ├── __init__.py
    │   ├── imagexpress.py
    │   └── opera_phenix.py
    ├── __init__.py
    └── __main__.py            # Command-line entry point

Extension Points
----------------

EZStitcher is designed to be extended in several ways:

1. **New Microscope Types**: Add new microscope types by implementing the FilenameParser and MetadataHandler interfaces
2. **New Preprocessing Functions**: Add new preprocessing functions to the ImagePreprocessor class
3. **New Focus Detection Methods**: Add new focus detection methods to the FocusAnalyzer class
4. **New Stitching Strategies**: Add new stitching strategies to the Stitcher class
5. **New Pipeline Components**: Add new components to the PipelineOrchestrator
