Microscope Handler Integration
==============================

OpenHCS achieves microscope-agnostic processing through a handler system that abstracts the unique characteristics of different imaging platforms while providing a unified interface to the pipeline system.

## Why Microscope Abstraction Matters

High-content screening involves diverse microscope platforms (Opera Phenix, ImageXpress, etc.), each with distinct:

- **Directory structures**: Flat vs hierarchical organization
- **Filename patterns**: Different field, well, and channel encoding schemes
- **Metadata formats**: XML, proprietary formats, embedded TIFF tags
- **File organization**: Single files vs multi-file series

Without abstraction, pipelines would need platform-specific logic throughout, making them brittle and hard to maintain. The handler system isolates these differences behind a clean interface.

## Architecture: Composition Over Inheritance

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

## Key Architectural Components

### Workspace Preparation

Each microscope format requires different workspace preparation to normalize directory structures for pipeline processing:

- **ImageXpress**: Flattens nested Z-step directories into flat structure
- **Opera Phenix**: Handles multi-level plate/field organization
- **Generic formats**: Minimal preparation, direct processing

The `_prepare_workspace()` method encapsulates these format-specific transformations, ensuring pipelines always see a consistent structure regardless of the original microscope organization.

### Pattern Detection and File Discovery

Handlers implement automatic pattern detection to identify image files and extract metadata:

.. code:: python

   # Each handler provides format-specific pattern detection
   patterns = handler.auto_detect_patterns(input_dir, well_id)
   file_paths = handler.path_list_from_pattern(pattern, input_dir)

This abstraction allows pipelines to discover images without knowing the underlying filename conventions or directory structures.

## Integration with Pipeline System

### Handler Factory and Selection

OpenHCS automatically selects the appropriate handler based on directory structure analysis:

- **Automatic detection**: Scans for format-specific markers (directory names, file patterns)
- **Explicit selection**: Users can override with specific handler types
- **Fallback handling**: Generic handler for unknown formats

### FileManager Integration

Handlers work seamlessly with OpenHCS's VFS system, supporting both disk and memory backends:

- **Workspace preparation** operates through FileManager abstraction
- **Pattern detection** works across different storage backends
- **File discovery** respects backend-specific optimizations

## Extensibility: Adding New Microscope Formats

The handler architecture makes adding support for new microscope formats straightforward:

### 1. **Implement the ABC Contract**

Create a new handler class implementing the required abstract methods:

.. code:: python

   class NewMicroscopeHandler(MicroscopeHandler):
       @property
       def parser(self) -> FilenameParser:
           return NewMicroscopeParser()

       @property
       def metadata_handler(self) -> MetadataHandler:
           return NewMicroscopeMetadataHandler()

### 2. **Define Format-Specific Logic**

- **Directory structure**: What directories indicate this format?
- **Workspace preparation**: What transformations are needed?
- **Filename patterns**: How are wells, fields, channels encoded?
- **Metadata sources**: XML files, embedded TIFF tags, etc.?

### 3. **Register with Factory**

The handler factory automatically detects and uses new handlers based on directory structure patterns.

## Design Benefits

### **Separation of Concerns**
- **Parser**: Handles filename pattern extraction and construction
- **Metadata Handler**: Manages acquisition parameters and plate layout
- **Workspace Preparation**: Normalizes directory structures
- **Handler**: Orchestrates components and provides unified interface

### **Testability and Maintainability**
- Each component can be tested independently
- Format-specific logic is isolated and contained
- Changes to one microscope format don't affect others
- Common functionality can be shared across similar formats

### **Pipeline Integration**
- Pipelines remain microscope-agnostic
- Automatic format detection reduces user configuration
- Consistent interface regardless of underlying complexity
- Seamless integration with VFS and memory management systems
This architecture enables OpenHCS to process data from any supported microscope platform through a single, consistent pipeline interface, while handling the complex format-specific details transparently.

