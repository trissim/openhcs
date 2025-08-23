Pattern Detection and Microscope Integration System
===================================================

Overview
--------

OpenHCS implements a pattern detection system that automatically
discovers image file patterns across different microscope formats. This
system coordinates filename parsing, directory structure analysis, and
pattern grouping to enable flexible pipeline processing.

Architecture Components
-----------------------

Core Components
~~~~~~~~~~~~~~~

::

   ┌─────────────────────────────────────────────────────────────┐
   │                    MicroscopeHandler                        │
   │  • Format-specific directory flattening                    │
   │  • Filename parsing and pattern detection                  │
   │  • Metadata extraction and validation                      │
   │  • Post-workspace processing coordination                  │
   └─────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
   ┌─────────────────────────────────────────────────────────────┐
   │                PatternDiscoveryEngine                       │
   │  • Auto-detection of file patterns                         │
   │  • Pattern grouping by components                          │
   │  • Pattern validation and instantiation                    │
   │  • Cross-well pattern coordination                         │
   └─────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
   ┌─────────────────────────────────────────────────────────────┐
   │                 FilenameParser                              │
   │  • Format-specific regex patterns                          │
   │  • Component extraction (well, site, channel, z_index)     │
   │  • Filename construction and validation                    │
   │  • Pattern template generation                             │
   └─────────────────────────────────────────────────────────────┘

Pattern Detection Flow
----------------------

Phase 1: Directory Structure Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each microscope handler implements format-specific directory processing:

ImageXpress Handler
^^^^^^^^^^^^^^^^^^^

.. code:: python

   def _prepare_workspace(self, workspace_path, filemanager):
       """Flatten ImageXpress Z-step directory structure."""
       
       # 1. Find TimePoint_1 directories
       entries = filemanager.list_dir(workspace_path, "disk")
       for entry in entries:
           if "TimePoint_1" in entry:
               # 2. Process Z-step subdirectories
               self._flatten_zsteps(entry, filemanager)
       
       return workspace_path

   def _flatten_zsteps(self, directory, filemanager):
       """Flatten ZStep_N directories into parent directory."""
       
       # 1. Find ZStep directories (ZStep_1, ZStep_2, etc.)
       zstep_pattern = re.compile(r"ZStep[_-]?(\d+)", re.IGNORECASE)
       subdirs = filemanager.list_dir(directory, "disk")
       
       for subdir in subdirs:
           match = zstep_pattern.match(subdir.name)
           if match:
               z_index = int(match.group(1))
               
               # 2. Move images to parent with updated z_index
               img_files = filemanager.list_image_files(subdir, "disk")
               for img_file in img_files:
                   # Parse original filename
                   components = self.parser.parse_filename(img_file.name)
                   
                   # Update z_index component
                   components['z_index'] = z_index
                   
                   # Construct new filename with correct z_index
                   new_name = self.parser.construct_filename(**components)
                   
                   # Move file to parent directory
                   new_path = directory / new_name
                   filemanager.move(img_file, new_path, "disk")

Opera Phenix Handler
^^^^^^^^^^^^^^^^^^^^

.. code:: python

   def _prepare_workspace(self, workspace_path, filemanager):
       """Rename Opera Phenix images based on spatial layout."""
       
       # 1. Find Index.xml for spatial mapping
       index_xml = self.metadata_handler.find_metadata_file(workspace_path)
       spatial_mapping = self._parse_spatial_layout(index_xml)
       
       # 2. Find image directory
       image_dir = workspace_path / "Images"
       if not image_dir.exists():
           # Look for other common image directories
           image_dir = self._find_image_directory(workspace_path)
       
       # 3. Rename files based on spatial layout
       img_files = filemanager.list_image_files(image_dir, "disk")
       for img_file in img_files:
           # Parse original filename
           components = self.parser.parse_filename(img_file.name)
           
           # Apply spatial remapping
           if components['site'] in spatial_mapping:
               components['site'] = spatial_mapping[components['site']]
           
           # Construct new filename
           new_name = self.parser.construct_filename(**components)
           new_path = img_file.parent / new_name
           
           if new_path != img_file:
               filemanager.move(img_file, new_path, "disk")
       
       return image_dir

Phase 2: Pattern Discovery
~~~~~~~~~~~~~~~~~~~~~~~~~~

The PatternDiscoveryEngine analyzes the flattened directory structure:

.. code:: python

   def auto_detect_patterns(
       self,
       folder_path: Union[str, Path],
       well_filter: List[str],
       extensions: List[str],
       group_by: Optional[str],
       variable_components: List[str],
       backend: str
   ) -> Dict[str, Any]:
       """Automatically detect image patterns in a folder."""

       # 1. Find and filter images by well
       files_by_well = self._find_and_filter_images(
           folder_path, well_filter, extensions, True, backend
       )

       if not files_by_well:
           return {}

       # 2. Generate patterns for each well
       result = {}
       for well, files in files_by_well.items():
           # Generate patterns from file list
           patterns = self._generate_patterns_for_files(files, variable_components)

           # Group patterns by component if requested
           result[well] = (
               self.group_patterns_by_component(patterns, component=group_by)
               if group_by else patterns
           )

       return result

   def _find_and_filter_images(self, folder_path, well_filter, extensions, 
                              recursive, backend):
       """Find all image files and filter by well."""
       
       # 1. Get all image files from directory
       image_paths = self.filemanager.list_image_files(
           folder_path, backend, extensions=extensions, recursive=recursive
       )
       
       # 2. Parse filenames and group by well
       files_by_well = defaultdict(list)
       for img_path in image_paths:
           filename = os.path.basename(img_path)
           
           # Parse filename to extract metadata
           metadata = self.parser.parse_filename(filename)
           if not metadata:
               continue
           
           # Filter by well
           well = metadata['well']
           if well not in well_filter:
               continue
           
           files_by_well[well].append(img_path)
       
       return files_by_well

Phase 3: Pattern Grouping
~~~~~~~~~~~~~~~~~~~~~~~~~

Patterns are grouped by components for processing:

.. code:: python

   def group_patterns_by_component(self, patterns, component):
       """Group patterns by a specific component (channel, site, etc.)."""
       
       grouped_patterns = defaultdict(list)
       
       for pattern in patterns:
           # Extract pattern template
           pattern_str = pattern.get_pattern()
           pattern_template = pattern_str.replace(self.PLACEHOLDER_PATTERN, '001')
           
           # Parse template to get component value
           metadata = self.parser.parse_filename(pattern_template)
           if not metadata or component not in metadata:
               raise ValueError(f"Missing component '{component}' in pattern: {pattern_str}")
           
           # Group by component value
           value = str(metadata[component])
           grouped_patterns[value].append(pattern)
       
       return grouped_patterns

Filename Parsing System
-----------------------

Parser Architecture
~~~~~~~~~~~~~~~~~~~

Each microscope format has a specialized parser:

ImageXpress Parser
^^^^^^^^^^^^^^^^^^

.. code:: python

   class ImageXpressFilenameParser(FilenameParser):
       """Parser for ImageXpress filename format."""
       
       def __init__(self, filemanager, pattern_format=None):
           # Default ImageXpress pattern
           self._pattern = re.compile(
               r"([A-Z]\d{2})_s(\d+)_w(\d+)_z(\d+)\.(\w+)$"
           )
           # Groups: well, site, channel, z_index, extension
       
       def parse_filename(self, filename):
           """Parse ImageXpress filename into components."""
           basename = Path(filename).name
           match = self._pattern.match(basename)
           
           if match:
               well, site_str, channel_str, z_str, ext = match.groups()
               
               # Handle placeholder components
               parse_comp = lambda s: None if not s or '{' in s else int(s)
               
               return {
                   'well': well,
                   'site': parse_comp(site_str),
                   'channel': parse_comp(channel_str),
                   'z_index': parse_comp(z_str),
                   'extension': ext or '.tif'
               }
           
           return None
       
       def construct_filename(self, well, site, channel, z_index, extension):
           """Construct filename from components."""
           return f"{well}_s{site:03d}_w{channel}_z{z_index:03d}{extension}"

Opera Phenix Parser
^^^^^^^^^^^^^^^^^^^

.. code:: python

   class OperaPhenixFilenameParser(FilenameParser):
       """Parser for Opera Phenix filename format."""
       
       def __init__(self, filemanager, pattern_format=None):
           # Opera Phenix pattern: r(\d+)c(\d+)f(\d+)p(\d+)-ch(\d+)sk(\d+)fk(\d+)fl(\d+)\.(\w+)
           self._pattern = re.compile(
               r"r(\d+)c(\d+)f(\d+)p(\d+)-ch(\d+)sk(\d+)fk(\d+)fl(\d+)\.(\w+)$"
           )
       
       def parse_filename(self, filename):
           """Parse Opera Phenix filename into components."""
           basename = Path(filename).name
           match = self._pattern.match(basename)
           
           if match:
               row, col, site_str, z_str, channel_str, ext = match.groups()
               
               # Create well ID from row and column
               well = f"R{int(row):02d}C{int(col):02d}"
               
               # Parse components
               parse_comp = lambda s: None if not s or '{' in s else int(s)
               
               return {
                   'well': well,
                   'site': parse_comp(site_str),
                   'channel': parse_comp(channel_str),
                   'wavelength': parse_comp(channel_str),  # Backward compatibility
                   'z_index': parse_comp(z_str),
                   'extension': ext or '.tif'
               }
           
           return None

Integration with Pipeline System
--------------------------------

Post-Workspace Processing
~~~~~~~~~~~~~~~~~~~~~~~~~

The orchestrator calls ``post_workspace()`` after creating symlinks:

.. code:: python

   # In orchestrator.compile_pipelines()
   def compile_pipelines(self):
       """Compile pipelines for all detected wells."""
       
       # 1. Create workspace symlinks
       self.create_workspace_symlinks()
       
       # 2. Process workspace with microscope handler
       actual_input_dir = self.microscope_handler.post_workspace(
           workspace_path=self.workspace_path,
           filemanager=self.filemanager
       )
       
       # 3. Update input directory to flattened location
       self.input_dir = actual_input_dir
       
       # 4. Detect patterns in processed directory
       patterns_by_well = self.microscope_handler.auto_detect_patterns(
           folder_path=self.input_dir,
           well_filter=self.wells,
           extensions=DEFAULT_IMAGE_EXTENSIONS,
           group_by="channel",
           variable_components=["site"],
           backend="disk"
       )
       
       # 5. Compile pipeline for each well
       for well_id in self.wells:
           if well_id in patterns_by_well:
               context = self.create_context(well_id)
               # ... compilation continues

Pattern Usage in FunctionStep
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Patterns are used during step execution:

.. code:: python

   # In FunctionStep.process()
   def process(self, context):
       """Execute function step using detected patterns."""
       
       # 1. Get patterns for this well
       patterns_by_well = context.microscope_handler.auto_detect_patterns(
           str(step_input_dir), [well_id], DEFAULT_IMAGE_EXTENSIONS,
           group_by, variable_components, read_backend
       )
       
       # 2. Resolve function patterns
       grouped_patterns, comp_to_funcs, comp_to_base_args = prepare_patterns_and_functions(
           patterns_by_well[well_id], self.func, component=group_by
       )
       
       # 3. Process each component group
       for comp_val, current_pattern_list in grouped_patterns.items():
           exec_func_or_chain = comp_to_funcs[comp_val]
           base_kwargs = comp_to_base_args[comp_val]
           
           for pattern_item in current_pattern_list:
               # Get matching files for this pattern
               matching_files = context.microscope_handler.path_list_from_pattern(
                   str(step_input_dir), pattern_item, read_backend
               )
               
               # Load, stack, process, unstack, save
               _process_single_pattern_group(...)

Pattern Data Structures
-----------------------

PatternPath Objects
~~~~~~~~~~~~~~~~~~~

Patterns are represented as PatternPath objects:

.. code:: python

   class PatternPath:
       """Represents a file pattern with component placeholders."""
       
       def __init__(self, pattern_string):
           self.pattern = pattern_string
       
       def get_pattern(self):
           """Get the pattern string."""
           return self.pattern
       
       def is_fully_instantiated(self):
           """Check if pattern has no uninstantiated placeholders."""
           return '{' not in self.pattern and '}' not in self.pattern

Pattern Grouping Results
~~~~~~~~~~~~~~~~~~~~~~~~

Pattern detection returns nested dictionaries:

.. code:: python

   # Example result structure
   patterns_by_well = {
       'A01': {
           'channel_1': [
               PatternPath("A01_s{site}_w1_z{z_index}.tif"),
               # ... more patterns for channel 1
           ],
           'channel_2': [
               PatternPath("A01_s{site}_w2_z{z_index}.tif"),
               # ... more patterns for channel 2
           ]
       },
       'D02': {
           # ... patterns for well D02
       }
   }

Error Handling and Validation
-----------------------------

Pattern Validation
~~~~~~~~~~~~~~~~~~

.. code:: python

   def validate_patterns(patterns):
       """Validate pattern structure and instantiation."""
       
       for pattern in patterns:
           # Check type
           if not isinstance(pattern, PatternPath):
               raise TypeError(f"Invalid pattern type: {type(pattern)}")
           
           # Check instantiation
           if not pattern.is_fully_instantiated():
               raise ValueError(f"Pattern contains placeholders: {pattern}")
           
           # Check pattern syntax
           pattern_str = pattern.get_pattern()
           if not _is_valid_pattern_syntax(pattern_str):
               raise ValueError(f"Invalid pattern syntax: {pattern_str}")

Directory Structure Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   def validate_directory_structure(workspace_path, microscope_type):
       """Validate directory structure matches expected format."""
       
       if microscope_type == "imagexpress":
           # Check for TimePoint directories
           timepoint_dirs = list(workspace_path.glob("*TimePoint*"))
           if not timepoint_dirs:
               raise ValueError("ImageXpress format requires TimePoint directories")
       
       elif microscope_type == "opera_phenix":
           # Check for Index.xml
           index_files = list(workspace_path.glob("**/Index.xml"))
           if not index_files:
               raise ValueError("Opera Phenix format requires Index.xml file")

Performance Considerations
--------------------------

Performance Characteristics
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # Pattern detection performance considerations:
   class PatternDiscoveryEngine:
       """Pattern discovery engine with performance optimizations."""

       def __init__(self, parser: FilenameParser, filemanager: FileManager):
           self.parser = parser
           self.filemanager = filemanager

       def auto_detect_patterns(self, folder_path, **kwargs):
           """Auto-detect patterns with efficient file operations."""

           # Use FileManager for efficient directory listing
           # Breadth-first traversal for consistent ordering
           # Filter files by extension early to reduce parsing overhead

           return self._detect_patterns_optimized(folder_path, **kwargs)

Current Implementation Status
-----------------------------

Implemented Features
~~~~~~~~~~~~~~~~~~~~

-  ✅ MicroscopeHandler architecture with format-specific processing
-  ✅ PatternDiscoveryEngine for automatic pattern detection
-  ✅ FilenameParser interface with ImageXpress and Opera Phenix
   implementations
-  ✅ Directory structure flattening (ImageXpress Z-steps, Opera Phenix
   spatial remapping)
-  ✅ Pattern grouping by components (channel, site, z_index)
-  ✅ Integration with pipeline orchestrator and FunctionStep execution
-  ✅ post_workspace workflow for microscope-specific preprocessing

Future Enhancements
~~~~~~~~~~~~~~~~~~~

1. **Pattern Caching**: Cache pattern detection results for performance
2. **Dynamic Parser Registration**: Runtime registration of new
   microscope formats
3. **Parallel Pattern Detection**: Multi-threaded pattern discovery for
   large datasets
4. **Advanced Pattern Validation**: Enhanced validation of pattern
   consistency
5. **Lazy Pattern Loading**: On-demand pattern detection for large
   datasets
