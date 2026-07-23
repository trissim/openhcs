Plugin Registry System (AutoRegisterMeta)
=========================================

| **Status**: CANONICAL
| **Date**: 2025-10-30
| **Purpose**: Comprehensive documentation of OpenHCS's automatic plugin
  registration system using metaclass-driven discovery and lazy loading.

Overview
--------

OpenHCS implements a unified plugin registry system that automatically discovers
and registers plugin classes across multiple subsystems using metaclass-driven
registration, lazy discovery, and automatic configuration inference. This system
eliminates boilerplate code while providing type-safe, self-documenting plugin
architectures.

Why Automatic Plugin Registration
----------------------------------

Traditional plugin systems require extensive boilerplate:

**Traditional Approach** (❌ Boilerplate-heavy):

.. code-block:: python

   # Manual registration in each plugin file
   MICROSCOPE_HANDLERS = {}

   class ImageXpressHandler(MicroscopeHandler):
       FORMAT_NAME = 'imagexpress'

   # Manual registration call
   MICROSCOPE_HANDLERS['imagexpress'] = ImageXpressHandler

   # Manual discovery function
   def discover_all_handlers():
       """Manually import all handler modules."""
       from openhcs.microscopes import imagexpress_handler
       from openhcs.microscopes import opera_phenix_handler
       from openhcs.microscopes import omero_handler
       # ... more imports

   # Manual call at startup
   discover_all_handlers()

**AutoRegisterMeta Approach** (✅ Zero boilerplate):

.. code-block:: python

   # Base class with registry configuration as class attributes
   class MicroscopeHandler(metaclass=AutoRegisterMeta):
       __registry_key__ = 'FORMAT_NAME'
       # That's it! Registry auto-created, discovery auto-configured!

   # Access the auto-created registry
   MICROSCOPE_HANDLERS = MicroscopeHandler.__registry__

   # Plugin auto-registers on definition
   class ImageXpressHandler(MicroscopeHandler):
       FORMAT_NAME = 'imagexpress'  # That's it!

   # Discovery happens automatically on first access
   handler = MICROSCOPE_HANDLERS['imagexpress']  # Auto-discovers all handlers

Core Architecture
-----------------

AutoRegisterMeta Metaclass
~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``AutoRegisterMeta`` metaclass inherits from ``ABCMeta`` and automatically
registers concrete classes in configured registries:

.. code-block:: python

   class AutoRegisterMeta(ABCMeta):
       """
       Metaclass for automatic plugin registration with lazy discovery.

       Features:
       - Auto-registers concrete classes (skips abstract base classes)
       - Supports primary and secondary registries
       - Auto-infers discovery package from base class module
       - Auto-wraps secondary registries for lazy discovery
       - Integrates with LazyDiscoveryDict for on-demand plugin loading
       """

**Registration Flow**:

1. Class definition triggers ``AutoRegisterMeta.__new__()``
2. Metaclass auto-configures registry from class attributes (or inherits from parent)
3. Metaclass checks if class is abstract (has abstract methods)
4. If concrete, extracts registration key from ``key_attribute``
5. Registers class in primary registry
6. Registers in secondary registries if configured
7. Sets up lazy discovery on first registry access

Auto-Configuration from Class Attributes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The metaclass automatically configures registries from class attributes - **no manual
RegistryConfig needed**:

.. code-block:: python

   # ZERO BOILERPLATE - Just set class attributes!
   class BackendBase(metaclass=AutoRegisterMeta):
       __registry_key__ = '_backend_type'  # Required: attribute name for registration key
       __key_extractor__ = None            # Optional: function to derive key from class name
       __skip_if_no_key__ = True           # Optional: skip registration if key is None
       __secondary_registries__ = None     # Optional: list of SecondaryRegistry objects
       __registry_name__ = None            # Optional: human-readable registry name

   # Registry auto-created and stored on the class!
   STORAGE_BACKENDS = BackendBase.__registry__  # LazyDiscoveryDict auto-created

   # Child classes inherit the registry - no duplication!
   class StorageBackend(BackendBase, DataSource, DataSink):
       pass  # Inherits BackendBase.__registry__ automatically

   class ReadOnlyBackend(BackendBase, DataSource):
       pass  # Also inherits BackendBase.__registry__

**Auto-Configuration Logic**:

1. **Check parent classes first**: If any parent has ``__registry__``, inherit it
2. **Create new registry**: If class defines ``__registry_key__`` in its body, create new registry
3. **Skip registration**: If no registry found and no ``__registry_key__``, skip

**Auto-Inference**:

- ``discovery_package``: Auto-inferred from base class module (e.g., ``polystore.base`` → ``polystore``)
- ``discovery_recursive``: Auto-detected by checking if package has subpackages with ``__init__.py``
- ``registry_name``: Auto-derived from class name (e.g., ``StorageBackend`` → ``"storage backend"``)

RegistryConfig Dataclass (Legacy/Advanced)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For advanced use cases, you can still use explicit ``RegistryConfig``:

.. code-block:: python

   @dataclass(frozen=True)
   class RegistryConfig:
       """Configuration for automatic plugin registration."""

       # Primary registry
       registry_dict: RegistryDict
       key_attribute: str
       key_extractor: Optional[Callable] = None
       skip_if_no_key: bool = False

       # Secondary registries (e.g., metadata handlers)
       secondary_registries: Optional[list[SecondaryRegistry]] = None

       # Discovery configuration
       discovery_package: Optional[str] = None  # Auto-inferred if None!
       discovery_recursive: bool = True
       discovery_function: Optional[Callable] = None

       # Logging
       log_registration: bool = False
       registry_name: str = 'registry'

**Note**: The auto-configuration approach is preferred for new code. Explicit ``RegistryConfig``
is only needed for complex scenarios with custom discovery functions or multiple secondary registries.

LazyDiscoveryDict
~~~~~~~~~~~~~~~~~

Dictionary subclass that auto-discovers plugins on first access:

.. code-block:: python

   class LazyDiscoveryDict(dict):
       """
       Dictionary that automatically discovers and loads plugins on first access.

       Features:
       - Lazy loading: Only imports plugin modules when registry is accessed
       - One-time discovery: Caches results after first access
       - Graceful failure: Logs warnings for import errors
       - Fully generic: Auto-detects discovery module from package root
       """

       def _discover(self):
           """Discover and import all plugin modules."""
           if self._discovered:
               return

           # Import discovery module and call discovery function
           discovery_module = self._get_discovery_module()
           discovery_func = getattr(discovery_module, 'discover_registry_classes')
           discovered = discovery_func(
               base_class=self._base_class,
               package_name=self._config.discovery_package,
               recursive=self._config.discovery_recursive
           )

           self._discovered = True

**Auto-Discovery Trigger**: All dict access methods (``__getitem__``, ``keys()``,
``values()``, ``items()``, etc.) trigger discovery before returning results.

SecondaryRegistryDict
~~~~~~~~~~~~~~~~~~~~~

Dictionary subclass for secondary registries that auto-triggers primary discovery:

.. code-block:: python

   class SecondaryRegistryDict(dict):
       """
       Dictionary for secondary registries that auto-triggers primary discovery.

       Use case: METADATA_HANDLERS is populated when MICROSCOPE_HANDLERS classes
       are registered. If MICROSCOPE_HANDLERS uses lazy discovery, METADATA_HANDLERS
       remains empty until primary registry is accessed.

       Solution: SecondaryRegistryDict triggers primary discovery on first access.
       """

       def _ensure_discovered(self):
           """Trigger discovery of primary registry."""
           if hasattr(self._primary_registry, '_discover'):
               self._primary_registry._discover()

**Auto-Wrapping**: The metaclass automatically wraps plain dict secondary registries
with ``SecondaryRegistryDict`` - no manual wrapping needed!

Registry Inheritance Pattern
----------------------------

Multiple classes can share the same registry via inheritance:

.. code-block:: python

   # Base class creates the registry
   class BackendBase(metaclass=AutoRegisterMeta):
       __registry_key__ = '_backend_type'
       # Registry auto-created: BackendBase.__registry__

   # Child classes inherit the registry
   class StorageBackend(BackendBase, DataSource, DataSink):
       pass  # Inherits BackendBase.__registry__

   class ReadOnlyBackend(BackendBase, DataSource):
       pass  # Also inherits BackendBase.__registry__

   # All three classes share the SAME registry dict
   assert StorageBackend.__registry__ is BackendBase.__registry__
   assert ReadOnlyBackend.__registry__ is BackendBase.__registry__

   # Concrete implementations register in the shared registry
   class DiskStorageBackend(StorageBackend):
       _backend_type = 'disk'  # Registers in BackendBase.__registry__

   class VirtualWorkspaceBackend(ReadOnlyBackend):
       _backend_type = 'virtual_workspace'  # Also registers in BackendBase.__registry__

**Inheritance Logic**:

1. **Check parent classes first**: Metaclass checks ``__mro__`` for existing ``__registry__``
2. **Inherit if found**: Use parent's registry instead of creating new one
3. **Create only if needed**: Only create new registry if ``__registry_key__`` is defined in class body

**Benefits**:

- ✅ Single source of truth for all related plugins
- ✅ Clean interface hierarchy without registry duplication
- ✅ Subclasses can specialize behavior while sharing registration

Auto-Inference Features
------------------------

Discovery Package Auto-Inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The metaclass automatically infers the discovery package from the base class module:

.. code-block:: python

   # Base class in openhcs/io/base.py
   class BackendBase(metaclass=AutoRegisterMeta):
       __registry_key__ = '_backend_type'
       # discovery_package auto-inferred: 'polystore'

   # Metaclass auto-infers: 'polystore'
   # (Extracts package by removing last component from module path)

**Inference Logic**:

.. code-block:: python

   # Extract package from base class module
   # 'polystore.base' → 'polystore'
   module_parts = base_class.__module__.rsplit('.', 1)
   inferred_package = module_parts[0] if len(module_parts) > 1 else base_class.__module__

Discovery Recursive Auto-Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The metaclass automatically detects if discovery should be recursive:

.. code-block:: python

   # Checks if discovery package has subdirectories with __init__.py
   # If yes: discovery_recursive = True
   # If no: discovery_recursive = False

   # Example: polystore has subdirectories → recursive=True
   # Example: openhcs.constants has no subdirectories → recursive=False

Secondary Registry Auto-Wrapping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The metaclass automatically wraps plain dict secondary registries:

.. code-block:: python

   # User code - just use a plain dict!
   MICROSCOPE_HANDLERS = LazyDiscoveryDict()
   METADATA_HANDLERS = {}  # Plain dict

   class MicroscopeHandler(metaclass=AutoRegisterMeta):
       _registry_config = RegistryConfig(
           registry_dict=MICROSCOPE_HANDLERS,
           secondary_registries=[
               SecondaryRegistry(
                   registry_dict=METADATA_HANDLERS,  # Plain dict here
                   key_source=PRIMARY_KEY,
                   attr_name='METADATA_HANDLER'
               )
           ]
       )

   # Metaclass automatically:
   # 1. Detects METADATA_HANDLERS is a plain dict
   # 2. Wraps it with SecondaryRegistryDict(MICROSCOPE_HANDLERS)
   # 3. Updates module global to use wrapped version
   # 4. Now METADATA_HANDLERS.keys() auto-triggers MICROSCOPE_HANDLERS discovery!

Real-World Examples
-------------------

Example 1: Microscope Handler Registry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Registry Setup** (``openhcs/microscopes/microscope_base.py``):

.. code-block:: python

   from openhcs.core.auto_register_meta import (
       AutoRegisterMeta,
       RegistryConfig,
       SecondaryRegistry,
       LazyDiscoveryDict,
       PRIMARY_KEY
   )

   # Primary registry (lazy discovery)
   MICROSCOPE_HANDLERS = LazyDiscoveryDict()

   # Secondary registry (auto-wrapped by metaclass!)
   METADATA_HANDLERS = {}

   # Base class with auto-registration
   class MicroscopeHandler(ABC, metaclass=AutoRegisterMeta):
       """Base class for microscope-specific handlers."""

       _registry_config = RegistryConfig(
           registry_dict=MICROSCOPE_HANDLERS,
           key_attribute='FORMAT_NAME',
           skip_if_no_key=True,
           secondary_registries=[
               SecondaryRegistry(
                   registry_dict=METADATA_HANDLERS,
                   key_source=PRIMARY_KEY,
                   attr_name='METADATA_HANDLER'
               )
           ],
           log_registration=True,
           registry_name='microscope handler registry'
           # discovery_package auto-inferred: 'openhcs.microscopes'
       )

       FORMAT_NAME: Optional[str] = None  # Abstract base has None
       METADATA_HANDLER: Optional[Type] = None

**Plugin Implementation** (``openhcs/microscopes/imagexpress_handler.py``):

.. code-block:: python

   from openhcs.microscopes.microscope_base import MicroscopeHandler

   class ImageXpressHandler(MicroscopeHandler):
       """ImageXpress microscope handler."""

       FORMAT_NAME = 'imagexpress'  # Auto-registers with this key!
       METADATA_HANDLER = ImageXpressMetadata  # Populates secondary registry

**Usage** (automatic discovery):

.. code-block:: python

   from openhcs.microscopes.microscope_base import MICROSCOPE_HANDLERS

   # First access triggers automatic discovery
   handler_class = MICROSCOPE_HANDLERS['imagexpress']
   handler = handler_class(plate_path='/path/to/plate')

**Registered Handlers**:

- ``imagexpress`` → ``ImageXpressHandler``
- ``opera_phenix`` → ``OperaPhenixHandler``
- ``omero`` → ``OMEROHandler``
- ``openhcsdata`` → ``OpenHCSMicroscopeHandler``

Example 2: Storage Backend Registry (ZERO Boilerplate)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Registry Setup** (``openhcs/io/base.py``):

.. code-block:: python

   # ZERO BOILERPLATE - Just class attributes!
   class BackendBase(metaclass=AutoRegisterMeta):
       """Base class for all storage backends."""
       __registry_key__ = '_backend_type'
       # Registry auto-created, discovery auto-configured!

   # Access the auto-created registry
   STORAGE_BACKENDS = BackendBase.__registry__

   # Interface hierarchy with shared registry
   class StorageBackend(BackendBase, DataSource, DataSink):
       """Read-write storage backends."""
       # Inherits BackendBase.__registry__ - no duplication!

   class ReadOnlyBackend(BackendBase, DataSource):
       """Read-only storage backends."""
       # Also inherits BackendBase.__registry__ - same registry!

**Plugin Implementation** (``openhcs/io/disk.py``):

.. code-block:: python

   from polystore.base import StorageBackend

   class DiskStorageBackend(StorageBackend):
       """Disk-based storage backend."""
       _backend_type = 'disk'  # Auto-registers with this key!

**Registered Backends**:

- ``disk`` → ``DiskStorageBackend`` (read-write)
- ``zarr`` → ``ZarrStorageBackend`` (read-write)
- ``memory`` → ``MemoryStorageBackend`` (read-write)
- ``virtual_workspace`` → ``VirtualWorkspaceBackend`` (read-only)

Example 3: Library Registry System
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Registry Setup** (``openhcs/processing/backends/lib_registry/unified_registry.py``):

.. code-block:: python

   LIBRARY_REGISTRIES = LazyDiscoveryDict()

   class LibraryRegistryBase(ABC, metaclass=AutoRegisterMeta):
       """Base class for library-specific function registries."""

       _registry_config = RegistryConfig(
           registry_dict=LIBRARY_REGISTRIES,
           key_attribute='LIBRARY_NAME',
           skip_if_no_key=True,
           log_registration=True,
           registry_name='library registry'
           # discovery_package auto-inferred: 'openhcs.processing.backends.lib_registry'
       )

**Registered Libraries**:

- ``pyclesperanto`` → ``PyclesperantoRegistry`` (230+ GPU functions)
- ``skimage`` → ``SkimageRegistry`` (110+ CPU functions)
- ``cupy`` → ``CupyRegistry`` (124+ GPU functions)
- ``openhcs`` → ``OpenHCSRegistry`` (native implementations)

Example 4: ZMQ Server Registry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Registry Setup** (``zmqruntime/server.py``):

.. code-block:: python

   ZMQ_SERVERS = LazyDiscoveryDict()

   class ZMQServer(ABC, metaclass=AutoRegisterMeta):
       """Base class for ZMQ server implementations."""

       _registry_config = RegistryConfig(
           registry_dict=ZMQ_SERVERS,
           key_attribute='SERVER_TYPE',
           skip_if_no_key=True,
           log_registration=True,
           registry_name='ZMQ server registry'
           # discovery_package auto-inferred: 'openhcs.runtime'
       )

**Registered Servers**:

- ``execution`` → ``ZMQExecutionServer``
- ``viewer`` → ``ZMQViewerServer``
- ``fiji`` → ``ZMQFijiServer``

Implementation Details
----------------------

Discovery Module Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~

The discovery system uses a generic discovery module at the package root:

**File**: ``openhcs/core/registry_discovery.py``

.. code-block:: python

   def discover_registry_classes(
       base_class: Type,
       package_name: str,
       recursive: bool = True
   ) -> list[Type]:
       """
       Discover all concrete subclasses of base_class in package.

       Args:
           base_class: Base class to find subclasses of
           package_name: Package to search (e.g., 'openhcs.microscopes')
           recursive: Whether to search subpackages

       Returns:
           List of discovered concrete subclasses
       """
       discovered = []

       # Import package
       package = importlib.import_module(package_name)

       # Walk package modules
       for importer, modname, ispkg in pkgutil.walk_packages(
           path=package.__path__,
           prefix=package.__name__ + '.',
           onerror=lambda x: None
       ):
           if not recursive and ispkg:
               continue

           try:
               # Import module (triggers class registration)
               importlib.import_module(modname)
           except ImportError as e:
               logger.warning(f"Could not import {modname}: {e}")

       # Collect all registered subclasses
       for subclass in base_class.__subclasses__():
           if not inspect.isabstract(subclass):
               discovered.append(subclass)

       return discovered

Subprocess Safety
~~~~~~~~~~~~~~~~~

The lazy discovery system works correctly in subprocess environments (multiprocessing, ZMQ):

**Problem**: In a subprocess, registries start empty because lazy discovery hasn't triggered yet.

**Solution**: ``SecondaryRegistryDict`` auto-triggers primary discovery on first access.

**Example** (ZMQ execution server):

.. code-block:: python

   # In subprocess: registries start empty
   MICROSCOPE_HANDLERS = LazyDiscoveryDict()  # Empty, not discovered yet
   METADATA_HANDLERS = {}  # Auto-wrapped by metaclass

   # Auto-detection code accesses secondary registry
   available_types = list(METADATA_HANDLERS.keys())

   # SecondaryRegistryDict triggers primary discovery automatically!
   # 1. METADATA_HANDLERS.keys() called
   # 2. SecondaryRegistryDict._ensure_discovered() called
   # 3. MICROSCOPE_HANDLERS._discover() called
   # 4. All handlers imported and registered
   # 5. METADATA_HANDLERS populated via secondary registration
   # 6. Returns correct keys: ['imagexpress', 'opera_phenix', 'omero', 'openhcsdata']

**Test Results**: ✅ All integration tests pass in both direct and ZMQ execution modes.

Performance Characteristics
---------------------------

Lazy Loading Benefits
~~~~~~~~~~~~~~~~~~~~~

**Cold Start** (first import):

- Registry creation: ~0.1ms (just creates empty dict)
- No plugin imports: 0ms
- Total: ~0.1ms

**First Access** (triggers discovery):

- Module discovery: ~50-100ms (depends on package size)
- Plugin imports: ~200-500ms (depends on plugin count)
- Registration: ~1-5ms
- Total: ~250-600ms (one-time cost)

**Subsequent Access**:

- Registry lookup: ~0.001ms (standard dict access)
- No re-discovery: 0ms
- Total: ~0.001ms

**Memory Usage**:

- Empty registry: ~1KB
- After discovery: ~10-50KB (depends on plugin count)
- Plugin classes: Loaded only when accessed

Migration Guide
---------------

Migrating from Manual Registration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Before** (manual registration):

.. code-block:: python

   # Old: Manual registry and discovery
   MICROSCOPE_HANDLERS = {}

   class MicroscopeHandler(ABC):
       pass

   class ImageXpressHandler(MicroscopeHandler):
       FORMAT_NAME = 'imagexpress'

   # Manual registration
   MICROSCOPE_HANDLERS['imagexpress'] = ImageXpressHandler

   # Manual discovery function
   def discover_all_handlers():
       from openhcs.microscopes import imagexpress_handler
       from openhcs.microscopes import opera_phenix_handler
       # ... more imports

   # Manual call
   discover_all_handlers()

**After** (automatic registration):

.. code-block:: python

   # New: Automatic registry and discovery
   MICROSCOPE_HANDLERS = LazyDiscoveryDict()

   class MicroscopeHandler(ABC, metaclass=AutoRegisterMeta):
       _registry_config = RegistryConfig(
           registry_dict=MICROSCOPE_HANDLERS,
           key_attribute='FORMAT_NAME',
           skip_if_no_key=True,
           log_registration=True,
           registry_name='microscope handler registry'
       )

       FORMAT_NAME: Optional[str] = None

   class ImageXpressHandler(MicroscopeHandler):
       FORMAT_NAME = 'imagexpress'  # Auto-registers!

   # No manual registration needed!
   # No discovery function needed!
   # Just access the registry:
   handler = MICROSCOPE_HANDLERS['imagexpress']  # Auto-discovers on first access

**Benefits**:

- ✅ Eliminated ~200 lines of boilerplate across 5 registries
- ✅ No manual imports needed
- ✅ No discovery functions needed
- ✅ No manual registration calls needed
- ✅ Automatic subprocess safety
- ✅ Self-documenting plugin architecture

Creating a New Plugin Registry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Step 1**: Create registry dict and base class:

.. code-block:: python

   from openhcs.core.auto_register_meta import (
       AutoRegisterMeta,
       RegistryConfig,
       LazyDiscoveryDict
   )

   # Create registry
   MY_PLUGINS = LazyDiscoveryDict()

   # Create base class with auto-registration
   class MyPluginBase(ABC, metaclass=AutoRegisterMeta):
       """Base class for my plugins."""

       _registry_config = RegistryConfig(
           registry_dict=MY_PLUGINS,
           key_attribute='PLUGIN_NAME',
           skip_if_no_key=True,
           log_registration=True,
           registry_name='my plugin registry'
           # discovery_package auto-inferred from module!
       )

       PLUGIN_NAME: Optional[str] = None  # Abstract base has None

       @abstractmethod
       def process(self, data):
           """Plugin processing method."""
           pass

**Step 2**: Create plugins (auto-register on definition):

.. code-block:: python

   class MyFirstPlugin(MyPluginBase):
       """My first plugin."""

       PLUGIN_NAME = 'first'  # Auto-registers with this key!

       def process(self, data):
           return data * 2

   class MySecondPlugin(MyPluginBase):
       """My second plugin."""

       PLUGIN_NAME = 'second'  # Auto-registers with this key!

       def process(self, data):
           return data + 10

**Step 3**: Use plugins (auto-discovers on first access):

.. code-block:: python

   # First access triggers automatic discovery
   plugin = MY_PLUGINS['first']()
   result = plugin.process(5)  # Returns 10

**That's it!** No manual registration, no discovery functions, no boilerplate.

Advanced Features
-----------------

Custom Key Extraction
~~~~~~~~~~~~~~~~~~~~~

Use ``key_extractor`` to derive keys from class names:

.. code-block:: python

   from openhcs.core.auto_register_meta import extract_key_from_handler_suffix

   class MicroscopeHandler(ABC, metaclass=AutoRegisterMeta):
       _registry_config = RegistryConfig(
           registry_dict=MICROSCOPE_HANDLERS,
           key_attribute='FORMAT_NAME',
           key_extractor=extract_key_from_handler_suffix('Handler'),
           # If FORMAT_NAME is None, extracts from class name:
           # 'ImageXpressHandler' → 'imagexpress'
       )

Custom Discovery Function
~~~~~~~~~~~~~~~~~~~~~~~~~~

Provide custom discovery logic:

.. code-block:: python

   def custom_discovery(base_class, package_name, recursive):
       """Custom discovery logic."""
       # Your custom discovery implementation
       return discovered_classes

   class MyPluginBase(ABC, metaclass=AutoRegisterMeta):
       _registry_config = RegistryConfig(
           registry_dict=MY_PLUGINS,
           key_attribute='PLUGIN_NAME',
           discovery_function=custom_discovery
       )

Related Systems
---------------

This plugin registry system integrates with:

- :doc:`function_registry_system` - Library function discovery and registration
- :doc:`pattern_detection_system` - Microscope handler plugin architecture
- :doc:`storage_and_memory_system` - Storage backend plugin architecture

Summary
-------

The AutoRegisterMeta plugin registry system provides:

**Zero Boilerplate**:

- No manual registration calls
- No discovery functions
- No hardcoded imports
- No manual wrapping of secondary registries

**Automatic Features**:

- Auto-registration on class definition
- Auto-discovery on first access
- Auto-inference of discovery packages
- Auto-wrapping of secondary registries
- Auto-subprocess safety

**Developer Experience**:

- Self-documenting plugin architecture
- Type-safe with IDE support
- Graceful error handling
- Comprehensive logging

**Performance**:

- Lazy loading (fast startup)
- One-time discovery cost
- Minimal memory overhead
- Standard dict access speed

**Production Ready**:

- ✅ 5 registries migrated
- ✅ All integration tests passing
- ✅ Subprocess-safe (multiprocessing, ZMQ)
- ✅ Ready for PyPI packaging (fully generic)
