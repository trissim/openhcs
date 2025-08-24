Microscope Handler Integration
==============================

OpenHCS achieves microscope-agnostic processing through a handler system that abstracts the unique characteristics of different imaging platforms while providing a unified interface to the pipeline system.

Why Microscope Abstraction Matters
-----------------------------------

High-content screening involves diverse microscope platforms (Opera Phenix, ImageXpress, etc.), each with distinct:

- **Directory structures**: Flat vs hierarchical organization
- **Filename patterns**: Different field, well, and channel encoding schemes
- **Metadata formats**: XML, proprietary formats, embedded TIFF tags
- **File organization**: Single files vs multi-file series

Without abstraction, pipelines would need platform-specific logic throughout, making them brittle and hard to maintain. The handler system isolates these differences behind a clean interface.

Architecture: Composition Over Inheritance
-------------------------------------------

The handler system uses composition rather than monolithic inheritance, separating concerns into specialized components:

.. code:: python

   class MicroscopeHandler(ABC):
       @property
       @abstractmethod
       def parser(self) -> FilenameParser:
           """Extracts well, field, channel from filenames."""

       @property
       @abstractmethod
       def metadata_handler(self) -> MetadataHandler:
           """Reads acquisition parameters and plate layout."""

This design enables:

- **Independent evolution**: Parser and metadata logic can change separately
- **Testability**: Each component can be tested in isolation
- **Reusability**: Common parsing logic can be shared across similar formats
- **Extensibility**: New microscope formats require minimal code

Filename Parsers and Metadata Handlers
---------------------------------------

The core of microscope abstraction lies in two critical components that handle format-specific details:

Filename Parser Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each microscope format has unique filename conventions. Parsers extract semantic information from these patterns:

.. code-block:: python

   class ImageXpressParser(FilenameParser):
       """Parser for ImageXpress filename format: A01_s1_w1.tif"""

       def parse_filename(self, filename: str) -> ParsedFilename:
           # Extract well, site, wavelength from filename
           match = re.match(r'([A-Z]\d{2})_s(\d+)_w(\d+)', filename)
           if not match:
               raise ValueError(f"Invalid ImageXpress filename: {filename}")

           return ParsedFilename(
               well=match.group(1),        # "A01"
               site=int(match.group(2)),   # 1
               wavelength=int(match.group(3))  # 1
           )

       def construct_filename(self, well: str, site: int, wavelength: int) -> str:
           """Reverse operation: construct filename from components."""
           return f"{well}_s{site}_w{wavelength}.tif"

   class OperaPhenixParser(FilenameParser):
       """Parser for Opera Phenix format: r01c01f01p01-ch1sk1fk1fl1.tiff"""

       def parse_filename(self, filename: str) -> ParsedFilename:
           # More complex pattern with row/col encoding
           pattern = r'r(\d{2})c(\d{2})f(\d{2})p(\d{2})-ch(\d+)sk(\d+)fk(\d+)fl(\d+)'
           match = re.match(pattern, filename)
           if not match:
               raise ValueError(f"Invalid Opera Phenix filename: {filename}")

           row, col = int(match.group(1)), int(match.group(2))
           well = f"{chr(64 + row)}{col:02d}"  # Convert to A01 format

           return ParsedFilename(
               well=well,
               site=int(match.group(3)),
               channel=int(match.group(5))
           )

Metadata Handler Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Metadata handlers extract acquisition parameters and plate layout information:

.. code-block:: python

   class ImageXpressMetadataHandler(MetadataHandler):
       """Handles ImageXpress .HTD and .MES files."""

       def read_plate_metadata(self, plate_dir: Path) -> PlateMetadata:
           htd_file = plate_dir / f"{plate_dir.name}.HTD"
           if not htd_file.exists():
               raise FileNotFoundError(f"HTD file not found: {htd_file}")

           # Parse HTD file for plate layout
           with open(htd_file, 'r') as f:
               htd_data = self._parse_htd_format(f.read())

           return PlateMetadata(
               plate_name=htd_data['PlateName'],
               wells=htd_data['Wells'],
               sites_per_well=htd_data['SitesPerWell'],
               channels=htd_data['Wavelengths'],
               acquisition_date=htd_data['AcquisitionDate']
           )

       def read_acquisition_metadata(self, image_path: Path) -> AcquisitionMetadata:
           """Extract metadata from MES files or TIFF tags."""
           mes_file = image_path.with_suffix('.MES')
           if mes_file.exists():
               return self._parse_mes_file(mes_file)
           else:
               # Fallback to TIFF metadata
               return self._extract_tiff_metadata(image_path)

   class OperaPhenixMetadataHandler(MetadataHandler):
       """Handles Opera Phenix XML metadata files."""

       def read_plate_metadata(self, plate_dir: Path) -> PlateMetadata:
           # Opera Phenix uses XML files for metadata
           xml_files = list(plate_dir.glob("*.xml"))
           if not xml_files:
               raise FileNotFoundError("No XML metadata files found")

           metadata_xml = xml_files[0]  # Usually Index.idx.xml
           tree = ET.parse(metadata_xml)
           root = tree.getroot()

           # Extract plate information from XML structure
           wells = self._extract_wells_from_xml(root)
           channels = self._extract_channels_from_xml(root)

           return PlateMetadata(
               plate_name=root.get('PlateName', 'Unknown'),
               wells=wells,
               channels=channels,
               acquisition_date=self._parse_xml_timestamp(root)
           )

Key Architectural Components
----------------------------

Workspace Preparation
~~~~~~~~~~~~~~~~~~~~~

Each microscope format requires different workspace preparation to normalize directory structures for pipeline processing:

.. code-block:: python

   class ImageXpressHandler(MicroscopeHandler):
       def _prepare_workspace(self, input_dir: Path, workspace_dir: Path):
           """Flatten nested Z-step directories into flat structure."""
           # ImageXpress organizes files like: TimePoint_1/ZStep_1/A01_s1_w1.tif
           # We need to flatten this to: workspace/A01_s1_w1.tif

           for timepoint_dir in input_dir.glob("TimePoint_*"):
               if timepoint_dir.is_dir():
                   # Check for Z-step subdirectories
                   z_dirs = list(timepoint_dir.glob("ZStep_*"))
                   if z_dirs:
                       # Flatten Z-step structure
                       for z_dir in z_dirs:
                           for image_file in z_dir.glob("*.tif"):
                               # Create symlink in flat workspace
                               workspace_link = workspace_dir / image_file.name
                               workspace_link.symlink_to(image_file)
                   else:
                       # No Z-steps, process files directly
                       for image_file in timepoint_dir.glob("*.tif"):
                           workspace_link = workspace_dir / image_file.name
                           workspace_link.symlink_to(image_file)

   class OperaPhenixHandler(MicroscopeHandler):
       def _prepare_workspace(self, input_dir: Path, workspace_dir: Path):
           """Handle Opera Phenix multi-level organization."""
           # Opera Phenix may have: Images/r01c01f01p01-ch1sk1fk1fl1.tiff
           images_dir = input_dir / "Images"
           if images_dir.exists():
               for image_file in images_dir.rglob("*.tiff"):
                   workspace_link = workspace_dir / image_file.name
                   workspace_link.symlink_to(image_file)
           else:
               # Direct structure, create symlinks
               for image_file in input_dir.glob("*.tiff"):
                   workspace_link = workspace_dir / image_file.name
                   workspace_link.symlink_to(image_file)

This workspace preparation ensures pipelines always see a consistent flat structure regardless of the original microscope organization.

Pattern Detection and File Discovery
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Handlers implement automatic pattern detection to identify image files and extract metadata:

.. code-block:: python

   class MicroscopeHandler(ABC):
       def auto_detect_patterns(self, input_dir: Path, well_id: str) -> List[ImagePattern]:
           """Detect all image patterns for a specific well."""
           patterns = []

           # Use parser to identify files belonging to this well
           for image_file in input_dir.glob("*.tif*"):
               try:
                   parsed = self.parser.parse_filename(image_file.name)
                   if parsed.well == well_id:
                       # Group by site and channel to create patterns
                       pattern = ImagePattern(
                           well=parsed.well,
                           site=parsed.site,
                           channel=parsed.channel,
                           file_path=image_file
                       )
                       patterns.append(pattern)
               except ValueError:
                   # Skip files that don't match expected pattern
                   continue

           return self._group_patterns_by_acquisition(patterns)

       def path_list_from_pattern(self, pattern: ImagePattern, input_dir: Path) -> List[Path]:
           """Generate file paths matching a specific pattern."""
           file_paths = []

           # Use parser to construct expected filenames
           for site in pattern.sites:
               for channel in pattern.channels:
                   filename = self.parser.construct_filename(
                       well=pattern.well,
                       site=site,
                       channel=channel
                   )
                   file_path = input_dir / filename
                   if file_path.exists():
                       file_paths.append(file_path)

           return file_paths

This abstraction allows pipelines to discover images without knowing the underlying filename conventions or directory structures.

Integration with Pipeline System
---------------------------------

Handler Factory and Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OpenHCS automatically selects the appropriate handler based on directory structure analysis:

.. code-block:: python

   class MicroscopeHandlerFactory:
       @staticmethod
       def create_handler(input_dir: Path, microscope_type: Optional[str] = None) -> MicroscopeHandler:
           """Create appropriate handler based on directory structure or explicit type."""

           if microscope_type:
               # Explicit selection
               return MicroscopeHandlerFactory._create_explicit_handler(microscope_type)

           # Automatic detection based on directory structure
           if MicroscopeHandlerFactory._is_imagexpress_format(input_dir):
               return ImageXpressHandler()
           elif MicroscopeHandlerFactory._is_opera_phenix_format(input_dir):
               return OperaPhenixHandler()
           elif MicroscopeHandlerFactory._is_openhcs_format(input_dir):
               return OpenHCSHandler()
           else:
               # Fallback to generic handler
               return GenericHandler()

       @staticmethod
       def _is_imagexpress_format(input_dir: Path) -> bool:
           """Detect ImageXpress format by looking for TimePoint directories and .HTD files."""
           has_timepoint_dirs = any(input_dir.glob("TimePoint_*"))
           has_htd_file = any(input_dir.glob("*.HTD"))
           return has_timepoint_dirs or has_htd_file

       @staticmethod
       def _is_opera_phenix_format(input_dir: Path) -> bool:
           """Detect Opera Phenix format by looking for XML metadata and filename patterns."""
           has_xml_metadata = any(input_dir.glob("*.xml"))
           has_opera_filenames = any(input_dir.glob("*r??c??f??p??-ch*.tiff"))
           return has_xml_metadata and has_opera_filenames

       @staticmethod
       def _is_openhcs_format(input_dir: Path) -> bool:
           """Detect OpenHCS format by looking for openhcsmetadata.json."""
           return (input_dir / "openhcsmetadata.json").exists()

FileManager Integration
~~~~~~~~~~~~~~~~~~~~~~~

Handlers work seamlessly with OpenHCS's VFS system, supporting both disk and memory backends:

- **Workspace preparation** operates through FileManager abstraction
- **Pattern detection** works across different storage backends
- **File discovery** respects backend-specific optimizations

Metaclass Registration System
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OpenHCS uses a metaclass-based registration system that automatically registers new handler classes:

.. code-block:: python

   class MicroscopeHandlerMeta(ABCMeta):
       """Metaclass that automatically registers handler classes."""

       _registry: Dict[str, Type[MicroscopeHandler]] = {}

       def __new__(mcs, name, bases, namespace, **kwargs):
           # Create the class
           cls = super().__new__(mcs, name, bases, namespace, **kwargs)

           # Register non-abstract handlers
           if not getattr(cls, '__abstractmethods__', None):
               # Extract handler type from class name (e.g., "ImageXpress" from "ImageXpressHandler")
               handler_type = name.replace('Handler', '').lower()
               mcs._registry[handler_type] = cls
               print(f"Registered microscope handler: {handler_type} -> {cls}")

           return cls

       @classmethod
       def get_handler_class(mcs, handler_type: str) -> Type[MicroscopeHandler]:
           """Get handler class by type name."""
           return mcs._registry.get(handler_type.lower())

       @classmethod
       def list_available_handlers(mcs) -> List[str]:
           """List all registered handler types."""
           return list(mcs._registry.keys())

   class MicroscopeHandler(ABC, metaclass=MicroscopeHandlerMeta):
       """Base class with automatic registration."""

The metaclass automatically:

- **Registers handlers** upon class definition (no manual registration needed)
- **Validates implementation** of required abstract methods
- **Maintains handler registry** for factory pattern selection
- **Enables automatic detection** based on handler capabilities

This design ensures that new microscope formats are automatically available to the system once their handler class is defined.

OpenHCS Native Handler
~~~~~~~~~~~~~~~~~~~~~~

The OpenHCS handler represents a special case that leverages existing handler components while using OpenHCS-specific metadata:

.. code-block:: python

   class OpenHCSMicroscopeHandler(MicroscopeHandler):
       """Handler for OpenHCS pre-processed format with JSON metadata."""

       def __init__(self, filemanager: FileManager, pattern_format: Optional[str] = None):
           self.filemanager = filemanager
           self.metadata_handler = OpenHCSMetadataHandler(filemanager)
           self._parser: Optional[FilenameParser] = None
           self.plate_folder: Optional[Path] = None
           self.pattern_format = pattern_format

           # Parser is loaded dynamically based on metadata
           super().__init__(parser=None, metadata_handler=self.metadata_handler)

       @property
       def parser(self) -> FilenameParser:
           """Dynamically load parser based on metadata."""
           if self._parser is None:
               parser_name = self.metadata_handler.get_source_filename_parser_name(self.plate_folder)
               available_parsers = _get_available_filename_parsers()
               ParserClass = available_parsers.get(parser_name)

               if not ParserClass:
                   raise ValueError(f"Unknown parser '{parser_name}' in metadata")

               self._parser = ParserClass(pattern_format=self.pattern_format)

           return self._parser

       def _prepare_workspace(self, workspace_path: Path, filemanager: FileManager) -> Path:
           """OpenHCS format is already normalized, no preparation needed."""
           # Ensure plate_folder is set for dynamic parser loading
           if self.plate_folder is None:
               self.plate_folder = Path(workspace_path)
           return workspace_path

   class OpenHCSMetadataHandler(MetadataHandler):
       """Handles OpenHCS JSON metadata format."""

       METADATA_FILENAME = "openhcs_metadata.json"

       def get_source_filename_parser_name(self, plate_path: Path) -> str:
           """Get the original filename parser used for this plate."""
           metadata = self._load_metadata(plate_path)
           return metadata.get("source_filename_parser_name")

       def determine_main_subdirectory(self, plate_path: Path) -> str:
           """Determine which subdirectory contains the main input images."""
           metadata_dict = self._load_metadata_dict(plate_path)

           # Handle subdirectory-keyed format
           if subdirs := metadata_dict.get("subdirectories"):
               # Find subdirectory marked as main, or use first available
               for subdir, subdir_metadata in subdirs.items():
                   if subdir_metadata.get("main", False):
                       return subdir
               return next(iter(subdirs.keys()))  # Fallback to first

           # Legacy format fallback
           return "images"

**Key Architectural Features**:

- **Component reuse**: Leverages existing parser and metadata handler infrastructure
- **JSON-based metadata**: Uses `openhcsmetadata.json` instead of microscope-specific formats
- **Structured metadata**: Standardized JSON schema for plate layout, acquisition parameters, and file organization
- **Self-describing datasets**: Datasets carry their own metadata, making them portable and self-contained

**OpenHCS Metadata Structure**:
The `openhcs_metadata.json` file uses a subdirectory-keyed format to organize metadata by processing step:

.. code-block:: json

   {
     "subdirectories": {
       "images": {
         "microscope_handler_name": "imagexpress",
         "source_filename_parser_name": "ImageXpressFilenameParser",
         "grid_dimensions": [2048, 2048],
         "pixel_size": 0.325,
         "image_files": [
           "images/A01_s1_w1.tif",
           "images/A01_s1_w2.tif",
           "images/A01_s2_w1.tif"
         ],
         "channels": {"1": "DAPI", "2": "GFP"},
         "wells": {"A01": "Control", "A02": "Treatment"},
         "sites": {"1": "Site1", "2": "Site2"},
         "z_indexes": null,
         "available_backends": {"disk": true},
         "main": true
       },
       "processed": {
         "microscope_handler_name": "imagexpress",
         "source_filename_parser_name": "ImageXpressFilenameParser",
         "grid_dimensions": [2048, 2048],
         "pixel_size": 0.325,
         "image_files": [
           "processed/A01_s1_w1_filtered.tif",
           "processed/A01_s1_w2_filtered.tif"
         ],
         "channels": {"1": "DAPI", "2": "GFP"},
         "wells": {"A01": "Control"},
         "sites": {"1": "Site1"},
         "z_indexes": null,
         "available_backends": {"disk": true},
         "main": false
       }
     }
   }

This approach enables OpenHCS to create fully self-describing datasets that can be processed consistently regardless of the original microscope platform.

Extensibility: Adding New Microscope Formats
---------------------------------------------

The handler architecture makes adding support for new microscope formats straightforward:

1. Implement the ABC Contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a new handler class implementing the required abstract methods:

.. code:: python

   class NewMicroscopeHandler(MicroscopeHandler):
       @property
       def parser(self) -> FilenameParser:
           return NewMicroscopeParser()

       @property
       def metadata_handler(self) -> MetadataHandler:
           return NewMicroscopeMetadataHandler()

2. Define Format-Specific Logic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Directory structure**: What directories indicate this format?
- **Workspace preparation**: What transformations are needed?
- **Filename patterns**: How are wells, fields, channels encoded?
- **Metadata sources**: XML files, embedded TIFF tags, etc.?

3. Register with Factory
~~~~~~~~~~~~~~~~~~~~~~~~

The handler factory automatically detects and uses new handlers based on directory structure patterns.

Design Benefits
---------------

**Separation of Concerns**
- **Parser**: Handles filename pattern extraction and construction
- **Metadata Handler**: Manages acquisition parameters and plate layout
- **Workspace Preparation**: Normalizes directory structures
- **Handler**: Orchestrates components and provides unified interface

**Testability and Maintainability**
- Each component can be tested independently
- Format-specific logic is isolated and contained
- Changes to one microscope format don't affect others
- Common functionality can be shared across similar formats

**Pipeline Integration**
- Pipelines remain microscope-agnostic
- Automatic format detection reduces user configuration
- Consistent interface regardless of underlying complexity
- Seamless integration with VFS and memory management systems

This architecture enables OpenHCS to process data from any supported microscope platform through a single, consistent pipeline interface, while handling the complex format-specific details transparently.

