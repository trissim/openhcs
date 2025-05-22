Extending EZStitcher
==================

This guide explains how to extend EZStitcher with new functionality.

.. _extending-autopipelinefactory:

Extending AutoPipelineFactory
-------------------------

.. note::
   Extending AutoPipelineFactory is primarily for contributors to the core library or for organization-wide standardization. 
   For most advanced use cases, creating custom pipelines with the Pipeline/Step abstraction is recommended instead.

The AutoPipelineFactory class can be extended to create custom factory classes that build on the standard pipeline creation logic. This is useful when:

1. You want to create a standardized pipeline structure for your organization
2. You need to add additional pipelines to the standard position generation and assembly pipelines
3. You're contributing new functionality to the core EZStitcher library

Here's an example of extending AutoPipelineFactory to add a quality control pipeline:

.. code-block:: python

   from ezstitcher.factories import AutoPipelineFactory
   from ezstitcher.core.steps import Step
   from ezstitcher.core.pipeline import Pipeline

   class QCFactory(AutoPipelineFactory):
       """Adds an analysis pipeline after stitching."""

       def create_pipelines(self):
           # Get the standard pipelines from the parent class
           pipelines = super().create_pipelines()

           # Get the output directory from the assembly pipeline
           stitched_dir = pipelines[1].output_dir

           # Create a new analysis pipeline
           analysis = Pipeline(
               input_dir=stitched_dir,
               steps=[Step(func=self.simple_qc)],
               name="QC",
           )

           # Add the analysis pipeline to the list
           pipelines.append(analysis)
           return pipelines

       @staticmethod
       def simple_qc(images):
           from skimage.exposure import histogram
           return [histogram(im)[0] for im in images]

   # Usage
   factory = QCFactory(
       input_dir=orchestrator.workspace_path,
       normalize=True
   )
   pipelines = factory.create_pipelines()
   orchestrator.run(pipelines=pipelines)

When to Extend vs. Create Custom Pipelines
---------------------------------------

**Use custom pipelines (recommended for most users) when:**

- You need a one-off solution for a specific dataset
- You want maximum flexibility and control
- You want transparent, explicit code
- You're prototyping or experimenting

**Extend AutoPipelineFactory (for contributors) when:**

- You're adding a new feature to the EZStitcher library
- You need to standardize pipeline creation across an organization
- You're creating a reusable component that builds on the standard pipelines
- You need to maintain backward compatibility with existing code

In most cases, creating custom pipelines with the Pipeline/Step abstraction provides more flexibility and transparency than extending AutoPipelineFactory.

.. _extending-microscope-types:

Adding a New Microscope Type
-------------------------

EZStitcher is designed to be easily extended with support for new microscope types. For detailed information about the microscope formats currently supported by EZStitcher, see :ref:`microscope-formats`. For a comparison of different microscope formats, see :ref:`microscope-comparison`.

For comprehensive information about the microscope interfaces, including:

* MicroscopeHandler class
* FilenameParser interface
* MetadataHandler interface
* Available methods and attributes
* Function signatures and parameters

See the :doc:`../api/microscope_interfaces` documentation.

There are two approaches to adding a new microscope type:

1. **Using the BaseMicroscopeHandler class** (recommended for most cases)
2. **Implementing the FilenameParser and MetadataHandler interfaces separately** (for more complex cases)

Both approaches are described below.

Approach 1: Using BaseMicroscopeHandler
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The simplest way to add a new microscope type is to subclass the ``BaseMicroscopeHandler`` class:

.. code-block:: python

    from ezstitcher.core.microscope_interfaces import BaseMicroscopeHandler
    import re
    from pathlib import Path

    class CustomMicroscopeHandler(BaseMicroscopeHandler):
        """Handler for a custom microscope format."""

        # Regular expression for parsing file names
        # Example: Sample_A01_s3_w2_z1.tif
        FILE_PATTERN = re.compile(
            r'(?P<prefix>.+)_'
            r'(?P<well>[A-Z][0-9]{2})_'
            r's(?P<site>[0-9]+)_'
            r'w(?P<channel>[0-9]+)_'
            r'z(?P<z_index>[0-9]+)'
            r'\.tif$'
        )

        def __init__(self, plate_path):
            """Initialize the handler."""
            super().__init__(plate_path)

        def get_wells(self):
            """Get list of wells in the plate."""
            wells = set()
            for file_path in Path(self.plate_path).glob('**/*.tif'):
                match = self.FILE_PATTERN.match(file_path.name)
                if match:
                    wells.add(match.group('well'))
            return sorted(list(wells))

        def get_sites(self, well):
            """Get list of sites for a well."""
            sites = set()
            for file_path in Path(self.plate_path).glob(f'**/*_{well}_*.tif'):
                match = self.FILE_PATTERN.match(file_path.name)
                if match:
                    sites.add(match.group('site'))
            return sorted(list(sites))

        def get_channels(self, well, site=None):
            """Get list of channels for a well/site."""
            channels = set()
            pattern = f'**/*_{well}_s{site}_*.tif' if site else f'**/*_{well}_*.tif'
            for file_path in Path(self.plate_path).glob(pattern):
                match = self.FILE_PATTERN.match(file_path.name)
                if match:
                    channels.add(match.group('channel'))
            return sorted(list(channels))

        def get_z_indices(self, well, site=None, channel=None):
            """Get list of z-indices for a well/site/channel."""
            z_indices = set()
            pattern = f'**/*_{well}_s{site}_w{channel}_*.tif'
            for file_path in Path(self.plate_path).glob(pattern):
                match = self.FILE_PATTERN.match(file_path.name)
                if match:
                    z_indices.add(match.group('z_index'))
            return sorted(list(z_indices))

        def get_image_path(self, well, site, channel, z_index=None):
            """Get path to a specific image."""
            z_part = f'_z{z_index}' if z_index else ''
            pattern = f'**/*_{well}_s{site}_w{channel}{z_part}.tif'
            for file_path in Path(self.plate_path).glob(pattern):
                if self.FILE_PATTERN.match(file_path.name):
                    return str(file_path)
            return None

        def parse_file_name(self, file_path):
            """Parse components from a file name."""
            match = self.FILE_PATTERN.match(Path(file_path).name)
            if match:
                return {
                    'well': match.group('well'),
                    'site': match.group('site'),
                    'channel': match.group('channel'),
                    'z_index': match.group('z_index')
                }
            return None

        @classmethod
        def can_handle(cls, plate_path):
            """Check if this handler can handle the given plate."""
            # Check if any files match the pattern
            for file_path in Path(plate_path).glob('**/*.tif'):
                if cls.FILE_PATTERN.match(file_path.name):
                    return True
            return False

To register your custom handler with EZStitcher:

.. code-block:: python

    from ezstitcher.core.microscope_interfaces import register_microscope_handler

    # Register the custom handler
    register_microscope_handler(CustomMicroscopeHandler)

    # Now EZStitcher will automatically detect and use your handler
    orchestrator = PipelineOrchestrator(
        config=config,
        plate_path="/path/to/custom/plate"
    )

You can also explicitly specify which handler to use:

.. code-block:: python

    # Create orchestrator with specific handler
    orchestrator = PipelineOrchestrator(
        config=config,
        plate_path="/path/to/plate",
        microscope_handler=CustomMicroscopeHandler
    )

Approach 2: Implementing FilenameParser and MetadataHandler
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For more complex cases, you can implement the ``FilenameParser`` and ``MetadataHandler`` interfaces separately:

.. code-block:: python

    """
    NewMicroscope implementations for ezstitcher.

    This module provides concrete implementations of FilenameParser and MetadataHandler
    for NewMicroscope microscopes.
    """

    import re
    import logging
    from pathlib import Path
    from typing import Dict, List, Optional, Union, Any, Tuple

    from ezstitcher.core.microscope_interfaces import FilenameParser, MetadataHandler

    logger = logging.getLogger(__name__)


    class NewMicroscopeFilenameParser(FilenameParser):
        """Filename parser for NewMicroscope microscopes."""

        # Define the regex pattern as a class attribute
        FILENAME_PATTERN = r'([A-Z]\d{2})_s(\d+)_w(\d+)(?:_z(\d+))?\.(?:tif|tiff)'

        @classmethod
        def can_parse(cls, filename: str) -> bool:
            """Check if this parser can parse the given filename."""
            # Use the class attribute pattern
            return bool(re.match(cls.FILENAME_PATTERN, filename))

        def parse_filename(self, filename: str) -> Optional[Dict[str, Any]]:
            """Parse a NewMicroscope filename into its components."""
            match = re.match(self.FILENAME_PATTERN, filename)

            if not match:
                return None

            well, site, channel, z_index = match.groups()

            return {
                'well': well,
                'site': int(site),
                'channel': int(channel),
                'z_index': int(z_index) if z_index else None,
                'extension': Path(filename).suffix
            }

        def construct_filename(self, well: str, site: Optional[Union[int, str]] = None,
                              channel: Optional[int] = None,
                              z_index: Optional[Union[int, str]] = None,
                              extension: str = '.tif',
                              site_padding: int = 3, z_padding: int = 3) -> str:
            """Construct a NewMicroscope filename from components."""
            # Format site number with padding
            if site is None:
                site_str = ""
            elif isinstance(site, str) and site == self.PLACEHOLDER_PATTERN:
                site_str = f"_s{site}"
            else:
                site_str = f"_s{int(site):0{site_padding}d}"

            # Format channel number
            if channel is None:
                channel_str = ""
            else:
                channel_str = f"_w{int(channel)}"

            # Format z-index with padding
            if z_index is None:
                z_str = ""
            elif isinstance(z_index, str) and z_index == self.PLACEHOLDER_PATTERN:
                z_str = f"_z{z_index}"
            else:
                z_str = f"_z{int(z_index):0{z_padding}d}"

            # Ensure extension starts with a dot
            if not extension.startswith('.'):
                extension = f".{extension}"

            return f"{well}{site_str}{channel_str}{z_str}{extension}"


    class NewMicroscopeMetadataHandler(MetadataHandler):
        """Metadata handler for NewMicroscope microscopes."""

        def find_metadata_file(self, plate_path: Union[str, Path]) -> Optional[Path]:
            """Find the metadata file for a NewMicroscope plate."""
            plate_path = Path(plate_path)

            # Look for metadata file
            metadata_file = plate_path / "metadata.xml"
            if metadata_file.exists():
                return metadata_file

            return None

        def get_grid_dimensions(self, plate_path: Union[str, Path]) -> Tuple[int, int]:
            """Get grid dimensions for stitching from NewMicroscope metadata."""
            metadata_file = self.find_metadata_file(plate_path)
            if not metadata_file:
                # Default grid size if metadata file not found
                return (3, 3)

            # Parse metadata file to extract grid dimensions
            # This is just an example, implement your own parsing logic
            try:
                # Parse XML or other format
                # ...

                # Return grid dimensions
                return (4, 4)
            except Exception as e:
                logger.error(f"Error parsing metadata file: {e}")
                return (3, 3)

        def get_pixel_size(self, plate_path: Union[str, Path]) -> Optional[float]:
            """Get the pixel size from NewMicroscope metadata."""
            metadata_file = self.find_metadata_file(plate_path)
            if not metadata_file:
                return None

            # Parse metadata file to extract pixel size
            # This is just an example, implement your own parsing logic
            try:
                # Parse XML or other format
                # ...

                # Return pixel size in micrometers
                return 0.65
            except Exception as e:
                logger.error(f"Error parsing metadata file: {e}")
                return None

Then, register the new microscope type in `ezstitcher/microscopes/__init__.py`:

.. code-block:: python

    """
    Microscope-specific implementations for ezstitcher.

    This package contains modules for different microscope types, each providing
    concrete implementations of FilenameParser and MetadataHandler interfaces.
    """

    # Import microscope handlers for easier access
    from ezstitcher.microscopes.imagexpress import ImageXpressFilenameParser, ImageXpressMetadataHandler
    from ezstitcher.microscopes.opera_phenix import OperaPhenixFilenameParser, OperaPhenixMetadataHandler
    from ezstitcher.microscopes.new_microscope import NewMicroscopeFilenameParser, NewMicroscopeMetadataHandler
