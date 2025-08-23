Microscope Handler Integration System
=====================================

Overview
--------

The microscope handler system provides format-specific processing for
different microscope platforms. Each handler understands the unique
directory structures, filename patterns, and metadata formats of its
target microscope, providing a unified interface for the pipeline
system.

Handler Architecture
--------------------

Base Handler Interface
~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   class MicroscopeHandler(ABC, metaclass=MicroscopeHandlerMeta):
       """Base class for all microscope handlers."""

       def __init__(self, filemanager: FileManager, pattern_format: Optional[str] = None):
           # Handler creates parser and metadata handler internally
           # Actual implementation uses dependency injection pattern
           self.plate_folder: Optional[Path] = None

       @property
       @abstractmethod
       def parser(self) -> FilenameParser:
           """Format-specific filename parser."""
           pass

       @property
       @abstractmethod
       def metadata_handler(self) -> MetadataHandler:
           """Metadata extraction handler."""
           pass

       @abstractmethod
       def _prepare_workspace(self, workspace_path: Path, filemanager: FileManager) -> Path:
           """Format-specific workspace preparation."""
           pass

       @property
       @abstractmethod
       def common_dirs(self) -> List[str]:
           """Common directory names for this format."""
           pass

       def post_workspace(self, workspace_path: Union[str, Path], filemanager: FileManager, width: int = 3):
           """Unified post-processing workflow."""
           # 1. Apply format-specific preparation
           prepared_dir = self._prepare_workspace(workspace_path, filemanager)

           # 2. Find image directory based on common_dirs
           image_dir = self._find_image_directory(workspace_path, filemanager)

           # 3. Apply filename padding
           self._apply_filename_padding(image_dir, filemanager, width)

           return image_dir

Handler Lifecycle
~~~~~~~~~~~~~~~~~

The microscope handler follows a specific lifecycle during pipeline
execution:

::

   1. Creation → 2. Workspace Preparation → 3. Pattern Detection → 4. Execution Support
        ↓                    ↓                      ↓                    ↓
   Handler Factory    post_workspace()     auto_detect_patterns()  path_list_from_pattern()

Format-Specific Implementations
-------------------------------

ImageXpress Handler
~~~~~~~~~~~~~~~~~~~

Handles Molecular Devices ImageXpress microscope format with Z-step
directory flattening:

.. code:: python

   class ImageXpressHandler(MicroscopeHandler):
       """Handler for ImageXpress microscope format."""
       
       @property
       def common_dirs(self) -> List[str]:
           return ['TimePoint_1']
       
       def _prepare_workspace(self, workspace_path, filemanager):
           """Flatten ImageXpress Z-step directory structure."""
           
           # 1. Find TimePoint directories
           entries = filemanager.list_dir(workspace_path, "disk")
           subdirs = [Path(workspace_path) / entry for entry in entries 
                     if (Path(workspace_path) / entry).is_dir()]
           
           # 2. Process directories containing TimePoint_1
           for subdir in subdirs:
               if self.common_dirs in subdir.name:
                   self._flatten_zsteps(subdir, filemanager)
           
           return workspace_path
       
       def _flatten_zsteps(self, directory, filemanager):
           """Flatten ZStep_N subdirectories into parent directory."""
           
           # Pattern for Z-step directories: ZStep_1, ZStep_2, etc.
           zstep_pattern = re.compile(r"ZStep[_-]?(\d+)", re.IGNORECASE)
           
           # Find Z-step subdirectories
           entries = filemanager.list_dir(directory, "disk")
           zstep_dirs = []
           
           for entry in entries:
               entry_path = Path(directory) / entry
               if entry_path.is_dir():
                   match = zstep_pattern.match(entry_path.name)
                   if match:
                       z_index = int(match.group(1))
                       zstep_dirs.append((entry_path, z_index))
           
           # Process each Z-step directory
           for zstep_dir, z_index in zstep_dirs:
               # Get all image files in Z-step directory
               img_files = filemanager.list_image_files(zstep_dir, "disk")
               
               for img_file in img_files:
                   # Parse original filename
                   components = self.parser.parse_filename(img_file.name)
                   if not components:
                       continue
                   
                   # Update z_index component
                   components['z_index'] = z_index
                   
                   # Construct new filename with correct z_index
                   new_name = self.parser.construct_filename(
                       well=components['well'],
                       site=components['site'],
                       channel=components['channel'],
                       z_index=z_index,
                       extension=components['extension']
                   )
                   
                   # Move file to parent directory with new name
                   new_path = directory / new_name
                   filemanager.move(img_file, new_path, "disk")
               
               # Remove empty Z-step directory
               filemanager.delete(zstep_dir, "disk", recursive=True)

**Key Features**: - **Directory Flattening**: Moves images from
``ZStep_N/`` subdirectories to parent - **Z-Index Correction**: Updates
filenames with correct z_index values - **Cleanup**: Removes empty
Z-step directories after processing

Opera Phenix Handler
~~~~~~~~~~~~~~~~~~~~

Handles PerkinElmer Opera Phenix format with spatial layout remapping:

.. code:: python

   class OperaPhenixHandler(MicroscopeHandler):
       """Handler for Opera Phenix microscope format."""
       
       @property
       def common_dirs(self) -> List[str]:
           return ['Images']
       
       def _prepare_workspace(self, workspace_path, filemanager):
           """Rename Opera Phenix images based on spatial layout."""
           
           # 1. Find and parse Index.xml for spatial mapping
           index_xml = self.metadata_handler.find_metadata_file(workspace_path)
           spatial_mapping = self._parse_spatial_layout(index_xml)
           
           # 2. Find image directory
           image_dir = self._find_image_directory(workspace_path, filemanager)
           
           # 3. Apply spatial remapping to filenames
           img_files = filemanager.list_image_files(image_dir, "disk")
           
           for img_file in img_files:
               # Parse original filename
               components = self.parser.parse_filename(img_file.name)
               if not components:
                   continue
               
               # Apply spatial remapping if available
               original_site = components['site']
               if original_site in spatial_mapping:
                   components['site'] = spatial_mapping[original_site]
                   
                   # Construct new filename with remapped site
                   new_name = self.parser.construct_filename(**components)
                   new_path = img_file.parent / new_name
                   
                   # Rename file if name changed
                   if new_path != img_file:
                       filemanager.move(img_file, new_path, "disk")
           
           return image_dir
       
       def _parse_spatial_layout(self, index_xml_path):
           """Parse Index.xml to extract spatial field mapping."""
           
           spatial_mapping = {}
           
           try:
               import xml.etree.ElementTree as ET
               tree = ET.parse(index_xml_path)
               root = tree.getroot()
               
               # Extract field positions from XML
               for field in root.findall('.//Field'):
                   field_id = field.get('ID')
                   row = field.get('Row')
                   col = field.get('Col')
                   
                   if field_id and row and col:
                       # Map original field ID to spatial position
                       spatial_position = int(row) * 12 + int(col)  # Assuming 12-column layout
                       spatial_mapping[int(field_id)] = spatial_position
           
           except Exception as e:
               logger.warning(f"Failed to parse spatial layout: {e}")
               # Return empty mapping - files will keep original names
           
           return spatial_mapping

**Key Features**: - **Spatial Remapping**: Reorders site indices based
on physical plate layout - **Metadata Integration**: Uses Index.xml for
spatial information - **Flexible Directory Finding**: Handles various
Opera Phenix directory structures

Directory Resolution System
---------------------------

Image Directory Detection
~~~~~~~~~~~~~~~~~~~~~~~~~

The ``post_workspace()`` method uses a standardized approach to find
image directories:

.. code:: python

   def post_workspace(self, workspace_path, filemanager):
       """Unified post-processing workflow."""
       
       # 1. Apply format-specific preparation
       prepared_dir = self._prepare_workspace(workspace_path, filemanager)
       
       # 2. Find image directory based on common_dirs
       entries = filemanager.list_dir(workspace_path, "disk")
       subdirs = [Path(workspace_path) / entry for entry in entries 
                 if (Path(workspace_path) / entry).is_dir()]
       
       # Look for directory containing common_dirs string
       image_dir = workspace_path  # Default fallback
       
       for subdir in subdirs:
           if self.common_dirs in subdir.name:
               image_dir = subdir
               break
       
       # 3. Apply filename padding for consistency
       self._apply_filename_padding(image_dir, filemanager)
       
       return image_dir

   def _apply_filename_padding(self, directory, filemanager, width=3):
       """Apply consistent filename padding."""
       
       img_files = filemanager.list_image_files(directory, "disk")
       
       for img_file in img_files:
           components = self.parser.parse_filename(img_file.name)
           if not components:
               continue
           
           # Apply padding to numeric components
           padded_components = {}
           for key, value in components.items():
               if key in ['site', 'z_index'] and isinstance(value, int):
                   padded_components[key] = f"{value:0{width}d}"
               else:
                   padded_components[key] = value
           
           # Construct padded filename
           new_name = self.parser.construct_filename(**padded_components)
           
           if new_name != img_file.name:
               new_path = img_file.parent / new_name
               filemanager.move(img_file, new_path, "disk")

Handler Factory System
----------------------

Auto-Detection
~~~~~~~~~~~~~~

The factory system automatically detects microscope formats:

.. code:: python

   def create_microscope_handler(microscope_type: str = 'auto',
                                 plate_folder: Optional[Union[str, Path]] = None,
                                 filemanager: Optional[FileManager] = None,
                                 pattern_format: Optional[str] = None,
                                 allowed_auto_types: Optional[List[str]] = None) -> MicroscopeHandler:
       """Factory function to create appropriate microscope handler."""

       if filemanager is None:
           raise ValueError("FileManager must be provided to create_microscope_handler")

       if microscope_type == 'auto':
           microscope_type = _auto_detect_microscope_type(plate_folder, filemanager, allowed_auto_types)

       # Get handler class from registry
       handler_class = MICROSCOPE_HANDLERS.get(microscope_type.lower())
       if not handler_class:
           raise ValueError(f"Unsupported microscope type: {microscope_type}")

       # Create handler instance with dependency injection
       handler = handler_class(filemanager, pattern_format=pattern_format)

       # Set plate_folder for handlers that need it (e.g., OpenHCS)
       if plate_folder and hasattr(handler, 'plate_folder'):
           handler.plate_folder = Path(plate_folder) if isinstance(plate_folder, str) else plate_folder

       return handler

   def _auto_detect_microscope_type(plate_folder: Path, filemanager: FileManager,
                                   allowed_types: Optional[List[str]] = None) -> str:
       """Auto-detect microscope type using metadata handler registry."""

       # Build detection order: openhcsdata first, then filtered/ordered list
       detection_order = ['openhcsdata']  # Always first, always included

       if allowed_types is None:
           # Use all registered handlers in registration order
           detection_order.extend([name for name in METADATA_HANDLERS.keys() if name != 'openhcsdata'])
       else:
           # Use filtered list, but ensure openhcsdata is first
           filtered_types = [name for name in allowed_types if name != 'openhcsdata' and name in METADATA_HANDLERS]
           detection_order.extend(filtered_types)

       # Try detection in order using metadata handlers
       for handler_name in detection_order:
           handler_class = METADATA_HANDLERS.get(handler_name)
           if handler_class and _try_metadata_detection(handler_class, filemanager, plate_folder):
               logger.info(f"Auto-detected {handler_name} microscope type")
               return handler_name

       # No handler succeeded
       raise ValueError(f"Could not auto-detect microscope type in {plate_folder}")

Handler Registry
~~~~~~~~~~~~~~~~

.. code:: python

   # Global registry of available handlers
   MICROSCOPE_HANDLERS = {
       'imagexpress': ImageXpressHandler,
       'opera_phenix': OperaPhenixHandler,
       # Additional handlers can be registered here
   }

   def register_microscope_handler(name, handler_class):
       """Register a new microscope handler."""
       MICROSCOPE_HANDLERS[name.lower()] = handler_class

Integration with Pipeline System
--------------------------------

Orchestrator Integration
~~~~~~~~~~~~~~~~~~~~~~~~

The orchestrator uses microscope handlers throughout the pipeline
lifecycle:

.. code:: python

   class PipelineOrchestrator:
       """Pipeline orchestrator with microscope handler integration."""
       
       def __init__(self, microscope_type='auto', plate_folder=None):
           # Create microscope handler
           self.microscope_handler = create_microscope_handler(
               microscope_type=microscope_type,
               plate_folder=plate_folder,
               filemanager=self.filemanager
           )
       
       def compile_pipelines(self):
           """Compile pipelines with microscope-specific processing."""
           
           # 1. Create workspace symlinks
           self.create_workspace_symlinks()
           
           # 2. Process workspace with microscope handler
           actual_input_dir = self.microscope_handler.post_workspace(
               workspace_path=self.workspace_path,
               filemanager=self.filemanager
           )
           
           # 3. Update input directory to processed location
           self.input_dir = actual_input_dir
           
           # 4. Detect wells and patterns
           self.wells = self._detect_wells()
           
           # 5. Compile pipeline for each well
           for well_id in self.wells:
               context = self.create_context(well_id)
               # Add microscope handler to context
               context.microscope_handler = self.microscope_handler
               # Continue compilation...

Context Integration
~~~~~~~~~~~~~~~~~~~

The microscope handler is made available to pipeline steps through the
context:

.. code:: python

   class ProcessingContext:
       """Processing context with microscope handler access."""
       
       def __init__(self, global_config, well_id=None, filemanager=None):
           self.global_config = global_config
           self.well_id = well_id
           self.filemanager = filemanager
           self.microscope_handler = None  # Set by orchestrator
           self.step_plans = {}
       
       def get_microscope_handler(self):
           """Get the microscope handler for this context."""
           if self.microscope_handler is None:
               raise RuntimeError("Microscope handler not available in context")
           return self.microscope_handler

FunctionStep Integration
~~~~~~~~~~~~~~~~~~~~~~~~

Function steps use the microscope handler for pattern resolution:

.. code:: python

   def process(self, context):
       """Execute function step with microscope handler support."""
       
       # Get microscope handler from context
       microscope_handler = context.get_microscope_handler()
       
       # Detect patterns for this well
       patterns_by_well = microscope_handler.auto_detect_patterns(
           folder_path=step_input_dir,
           well_filter=[well_id],
           extensions=DEFAULT_IMAGE_EXTENSIONS,
           group_by=group_by,
           variable_components=variable_components,
           backend=read_backend
       )
       
       # Process patterns
       for pattern_group in patterns_by_well[well_id]:
           # Get matching files for pattern
           matching_files = microscope_handler.path_list_from_pattern(
               str(step_input_dir), pattern_group, read_backend
           )
           
           # Process files...

Error Handling and Validation
-----------------------------

Format Validation
~~~~~~~~~~~~~~~~~

.. code:: python

   def validate_microscope_format(plate_folder, expected_format, filemanager):
       """Validate that directory matches expected microscope format."""
       
       if expected_format == 'imagexpress':
           # Check for required ImageXpress files
           htd_files = filemanager.list_files(
               plate_folder, extensions={'.htd', '.HTD'}, 
               recursive=True, backend="disk"
           )
           if not htd_files:
               raise ValueError("ImageXpress format requires .HTD metadata files")
       
       elif expected_format == 'opera_phenix':
           # Check for required Opera Phenix files
           index_xml = filemanager.find_file_recursive(
               plate_folder, "disk", filename="Index.xml"
           )
           if not index_xml:
               raise ValueError("Opera Phenix format requires Index.xml file")

Processing Validation
~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   def validate_workspace_processing(workspace_path, filemanager):
       """Validate workspace processing was successful."""
       
       # Check that image files exist
       img_files = filemanager.list_image_files(workspace_path, "disk")
       if not img_files:
           raise ValueError(f"No image files found after processing: {workspace_path}")
       
       # Check filename consistency
       inconsistent_files = []
       for img_file in img_files:
           if not _is_consistent_filename(img_file.name):
               inconsistent_files.append(img_file.name)
       
       if inconsistent_files:
           raise ValueError(f"Inconsistent filenames after processing: {inconsistent_files}")

Performance Considerations
--------------------------

Lazy Processing
~~~~~~~~~~~~~~~

.. code:: python

   def lazy_workspace_processing(workspace_path, filemanager):
       """Process workspace lazily for large datasets."""
       
       # Process directories on-demand
       for subdir in workspace_path.iterdir():
           if subdir.is_dir():
               yield from self._process_directory_lazy(subdir, filemanager)

   def _process_directory_lazy(self, directory, filemanager):
       """Process single directory lazily."""
       
       # Yield files as they're processed
       for img_file in filemanager.list_image_files(directory, "disk"):
           processed_file = self._process_single_file(img_file, filemanager)
           yield processed_file

Caching Strategy
~~~~~~~~~~~~~~~~

.. code:: python

   class CachedMicroscopeHandler:
       """Microscope handler with caching for repeated operations."""
       
       def __init__(self, base_handler):
           self.base_handler = base_handler
           self._pattern_cache = {}
           self._metadata_cache = {}
       
       def auto_detect_patterns(self, folder_path, **kwargs):
           """Cached pattern detection."""
           cache_key = (str(folder_path), tuple(sorted(kwargs.items())))
           
           if cache_key not in self._pattern_cache:
               patterns = self.base_handler.auto_detect_patterns(folder_path, **kwargs)
               self._pattern_cache[cache_key] = patterns
           
           return self._pattern_cache[cache_key]

Future Enhancements
-------------------

Additional Microscope Support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The primary enhancement goal is expanding microscope format support:

- **Additional Vendors**: Support for more microscope manufacturers
- **Format Variants**: Handle vendor-specific format variations
- **Legacy Formats**: Support for older microscope file formats
- **Custom Formats**: Framework for laboratory-specific formats
