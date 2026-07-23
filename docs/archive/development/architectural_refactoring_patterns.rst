Architectural Refactoring Patterns
===================================

Six fundamental architectural principles for OpenHCS development.

Six Fundamental Principles
--------------------------

1. ABC Contract Enforcement
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ABCs to enforce explicit contracts and enable polymorphism.

**Examples**: ``StorageBackend``, ``FilenameParser``, ``MetadataHandler``, ``LibraryRegistryBase``

2. Explicit Dependency Injection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dependencies are explicitly provided, never implicitly created.

**Examples**: ``create_microscope_handler(filemanager=filemanager)``, factory functions require all dependencies

3. Indirection Minimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Eliminate unnecessary layers; prefer direct method calls.

**Examples**: ``getattr(self, method_name)`` instead of dispatch tables, direct enum usage

4. Genericism Enforcement
~~~~~~~~~~~~~~~~~~~~~~~~~

Create truly generic systems that work with any valid configuration.

**Examples**: ``ComponentConfiguration[T]``, ``DynamicParserMeta`` generates methods from any enum

5. Fail-Loud Error Handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Specific exceptions with context; no silent failures or defensive programming.

**Examples**: ``MemoryConversionError``, ``StorageResolutionError``, ``allow_cpu_roundtrip=False`` by default

6. Consistent Interface Design
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Uniform patterns across all subsystems.

**Examples**: ABC + Factory pattern, consistent method naming, enum-driven behavior

Refactoring Patterns
--------------------

Pattern 1: ABC Contract Enforcement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Before**: Inconsistent interfaces across similar classes

.. code-block:: python

   class ImageXpressParser:
       def parse_well(self, filename): pass

   class OperaPhenixParser:
       def extract_well(self, filename): pass  # Different method name

**After**: ABC enforces consistent contracts

.. code-block:: python

   class FilenameParser(ABC):
       @abstractmethod
       def parse_well(self, filename): pass

   class ImageXpressParser(FilenameParser):
       def parse_well(self, filename): pass  # Contract enforced

   class OperaPhenixParser(FilenameParser):
       def parse_well(self, filename): pass  # Must match ABC

Pattern 2: Explicit Dependency Injection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Before**: Hidden dependencies and global state

.. code-block:: python

   class MicroscopeHandler:
       def __init__(self, microscope_type: str):
           self.filemanager = get_global_filemanager()  # Hidden dependency
           self.parser = create_parser(microscope_type)  # Hidden creation

**After**: Explicit dependency injection

.. code-block:: python

   class MicroscopeHandler:
       def __init__(self, parser: FilenameParser, filemanager: FileManager):
           self.parser = parser
           self.filemanager = filemanager

   # Factory function with explicit dependencies
   def create_microscope_handler(microscope_type: str, 
                               filemanager: FileManager) -> MicroscopeHandler:
       parser = create_parser(microscope_type)
       return MicroscopeHandler(parser, filemanager)

Pattern 3: Indirection Minimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Before**: Unnecessary dispatch tables and routing layers

.. code-block:: python

   class ComponentProcessor:
       def __init__(self):
           self.dispatch_table = {
               'DAPI': self._process_dapi,
               'GFP': self._process_gfp,
               'RFP': self._process_rfp
           }
       
       def process(self, component: str, data):
           handler = self.dispatch_table.get(component)
           if handler:
               return handler(data)

**After**: Direct method calls with enum-driven behavior

.. code-block:: python

   class ComponentProcessor:
       def process(self, component: Component, data):
           method_name = f"_process_{component.value.lower()}"
           method = getattr(self, method_name)
           return method(data)

Pattern 4: Genericism Enforcement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Before**: Pseudo-generic code with hardcoded assumptions

.. code-block:: python

   class ConfigValidator:
       def validate_pipeline_config(self, config):
           # Hardcoded for PipelineConfig only
           if not config.steps:
               return False
           return True
       
       def validate_zarr_config(self, config):
           # Separate method for each config type
           if not config.compression:
               return False
           return True

**After**: Truly generic validation using metaprogramming

.. code-block:: python

   class ConfigValidator(Generic[T]):
       def validate(self, config: T) -> ValidationResult:
           errors = []
           for field in fields(config):
               if field.metadata.get('required') and getattr(config, field.name) is None:
                   errors.append(f"Required field {field.name} is None")
           return ValidationResult(errors)

Pattern 5: Fail-Loud Error Handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Before**: Defensive programming with silent failures

.. code-block:: python

   def process_image(image_path: str):
       try:
           image = load_image(image_path)
           if image is not None:
               return process(image)
       except Exception:
           pass  # Silent failure
       return default_image()  # Fallback masks problems

**After**: Explicit error handling with specific exceptions

.. code-block:: python

   def process_image(image_path: str) -> ProcessedImage:
       try:
           image = load_image(image_path)
       except FileNotFoundError as e:
           raise ImageLoadError(f"Image not found: {image_path}") from e
       except PermissionError as e:
           raise ImageLoadError(f"Permission denied: {image_path}") from e
       
       if image.ndim != 3:
           raise ImageValidationError(f"Expected 3D image, got {image.ndim}D")
       
       return process(image)

Pattern 6: Consistent Interface Design
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Before**: Inconsistent interfaces across subsystems

.. code-block:: python

   class StorageBackend:
       def load_file(self, path): pass  # Different method name
       def write_file(self, data, path): pass

   class MicroscopeHandler:
       def get_data(self, path): pass  # Different method name
       def put_data(self, data, path): pass

**After**: Consistent interface pattern

.. code-block:: python

   class StorageBackend(ABC):
       @abstractmethod
       def load(self, path): pass  # Consistent naming

       @abstractmethod
       def save(self, data, path): pass

   class MicroscopeHandler(ABC):
       @abstractmethod
       def load(self, path): pass  # Same interface pattern

       @abstractmethod
       def save(self, data, path): pass

   # Factory pattern used consistently
   def create_storage_backend(backend_type: str) -> StorageBackend: pass
   def create_microscope_handler(microscope_type: str) -> MicroscopeHandler: pass

Refactoring Methodology
-----------------------

Step 1: Identify Violations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **ABC violations**: Similar classes with inconsistent interfaces
- **Dependency violations**: Hidden object creation or global state access
- **Indirection excess**: Unnecessary dispatch tables or routing layers
- **Pseudo-genericism**: Hardcoded assumptions in "generic" code
- **Defensive programming**: hasattr checks, silent fallbacks
- **Interface inconsistency**: Different method names for similar operations

Step 2: Apply Patterns
~~~~~~~~~~~~~~~~~~~~~~

1. **Create ABCs** for similar functionality
2. **Extract dependencies** to constructor parameters
3. **Remove dispatch layers** in favor of direct method calls
4. **Use Generic[T]** and metaprogramming for true genericism
5. **Replace defensive code** with explicit error handling
6. **Standardize interfaces** across subsystems

Step 3: Validate Consistency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- All subsystems follow ABC + Factory pattern
- Dependencies are explicitly injected
- Error handling is fail-loud with specific exceptions
- Systems work with any valid configuration
- Interfaces are consistent across similar functionality

Evidence from OpenHCS
---------------------

**Stable Systems** (Storage, Microscope): Established foundational patterns
**Refactored Systems** (Memory, Component): Applied same principles to new domains
**Cross-System Integration**: All systems work together due to consistent architecture

Summary
-------

OpenHCS refactoring follows six fundamental principles that create architectural consistency:

1. **ABC Contract Enforcement** - Explicit interfaces enable polymorphism
2. **Explicit Dependency Injection** - No hidden dependencies or object creation
3. **Indirection Minimization** - Direct method calls, fewer layers
4. **Genericism Enforcement** - True genericism via metaprogramming
5. **Fail-Loud Error Handling** - Specific exceptions, no silent failures
6. **Consistent Interface Design** - Uniform patterns across all subsystems

These principles appear consistently across both stable legacy code and newly refactored systems, creating a coherent architectural methodology.
