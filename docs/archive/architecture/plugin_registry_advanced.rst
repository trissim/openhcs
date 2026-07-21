Plugin Registry System - Advanced Topics
=========================================

| **Status**: CANONICAL
| **Date**: 2025-10-31
| **Purpose**: Advanced topics, troubleshooting, and best practices for the
  OpenHCS plugin registry system.

This document covers advanced usage, performance optimization, troubleshooting,
and third-party plugin development.

.. contents:: Table of Contents
   :local:
   :depth: 2

Caching System
--------------

Overview
~~~~~~~~

OpenHCS includes a unified persistent caching system that dramatically improves startup
performance across multiple subsystems:

1. **Plugin Registries** - Microscope handlers, storage backends, format registries
2. **Enum Generation** - Colormap enums, component enums
3. **Function Registries** - Already implemented with custom caching

**Performance Impact**:

- Plugin registries: 155ms → 16ms (**9.7x faster**)
- Colormap enums: 1400ms → 0.5ms (**2800x faster!**)
- Component enums: Cached persistently across processes
- **Combined startup improvement**: ~500ms → ~50ms (**10x faster**)

**Benefits**:

- **Faster startup**: 10-2800x faster depending on subsystem
- **Reduced I/O**: Avoids scanning filesystem for modules
- **Version-aware**: Automatically invalidates on version changes
- **Mtime-aware**: Detects file modifications and rebuilds cache (optional)

How It Works
~~~~~~~~~~~~

.. code-block:: python

   # Caching is enabled by default
   MICROSCOPE_HANDLERS = LazyDiscoveryDict(enable_cache=True)

   # First access: Full discovery + cache save
   handlers = list(MICROSCOPE_HANDLERS.keys())  # ~50ms

   # Subsequent runs: Load from cache
   handlers = list(MICROSCOPE_HANDLERS.keys())  # ~5ms (10x faster!)

Cache Location
~~~~~~~~~~~~~~

Caches are stored in XDG-compliant locations:

.. code-block:: bash

   ~/.local/share/openhcs/cache/
   ├── microscope_handler_registry.json
   ├── storage_backend_registry.json
   ├── zmq_server_registry.json
   ├── library_registry_registry.json
   ├── microscope_format_registry_registry.json
   ├── colormap_enum.json
   └── component_enums.json

Cache Validation
~~~~~~~~~~~~~~~~

Caches are automatically invalidated when:

1. **Version changes**: OpenHCS version differs from cached version
2. **Age exceeds limit**: Cache older than 7 days (configurable)
3. **File modifications**: Any plugin file modified since cache creation
4. **Corruption**: Cache file is corrupted or unreadable

Manual Cache Management
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from openhcs.core.registry_cache import RegistryCacheManager

   # Clear all caches
   import shutil
   from openhcs.core.xdg_paths import get_openhcs_cache_dir
   shutil.rmtree(get_openhcs_cache_dir())

   # Disable caching for a specific registry
   MICROSCOPE_HANDLERS = LazyDiscoveryDict(enable_cache=False)

Thread Safety
-------------

Discovery Thread Safety
~~~~~~~~~~~~~~~~~~~~~~~

**Discovery is NOT thread-safe** but this is acceptable because:

1. **GIL Protection**: Discovery happens during module import (GIL-protected)
2. **Idempotent**: Multiple discoveries produce identical results
3. **Worst Case**: Discovery runs twice (harmless, just slower)

**Recommendation**: Trigger discovery in main thread before spawning workers:

.. code-block:: python

   # In main thread before multiprocessing
   _ = list(MICROSCOPE_HANDLERS.keys())  # Force discovery

   # Now safe to use in worker processes
   with multiprocessing.Pool() as pool:
       pool.map(process_with_handlers, items)

Registry Access Thread Safety
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Registry access IS thread-safe** after discovery:

- Registries are plain dicts (thread-safe for reads)
- No writes after discovery completes
- Safe to access from multiple threads

Subprocess Behavior
-------------------

How Subprocesses Work
~~~~~~~~~~~~~~~~~~~~~

When a subprocess starts:

1. **Fresh Python interpreter**: No shared state with parent
2. **Empty registries**: All registries start empty
3. **Lazy discovery**: Discovery happens on first access
4. **Independent caches**: Each process has its own cache

**This is why** ``SecondaryRegistryDict`` is critical - it ensures secondary
registries trigger primary discovery in subprocesses.

Multiprocessing Best Practices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # ❌ BAD: Discovery in each worker (slow)
   def worker(item):
       handler = MICROSCOPE_HANDLERS[item.type]  # Discovery per worker!
       return handler.process(item)

   # ✅ GOOD: Pre-discover in main process
   def main():
       _ = list(MICROSCOPE_HANDLERS.keys())  # Discover once

       with multiprocessing.Pool() as pool:
           results = pool.map(worker, items)

ZMQ Subprocess Mode
~~~~~~~~~~~~~~~~~~~

ZMQ execution servers run in separate processes. The registry system handles
this automatically:

.. code-block:: python

   # In ZMQ subprocess: registries start empty
   available_types = list(METADATA_HANDLERS.keys())

   # SecondaryRegistryDict auto-triggers primary discovery
   # Result: All handlers discovered and available!

Enum Generation Caching
-----------------------

Overview
~~~~~~~~

OpenHCS dynamically generates several enums at runtime. These are now cached for
dramatic performance improvements:

1. **Colormap Enums** (``openhcs.utils.enum_factory``)
2. **Component Enums** (``openhcs.constants.constants``)

Colormap Enum Caching
~~~~~~~~~~~~~~~~~~~~~

The ``create_colormap_enum()`` function generates an enum of all available napari
colormaps. This is expensive (~1400ms) because it imports napari and introspects
all colormap plugins.

**With caching**:

.. code-block:: python

   from openhcs.utils.enum_factory import create_colormap_enum

   # First run: Full discovery + cache save (~1400ms)
   NapariColormap = create_colormap_enum()

   # Subsequent runs: Load from cache (~0.5ms) - 2800x faster!
   NapariColormap = create_colormap_enum()

**Cache invalidation**:

- OpenHCS version changes
- Cache age > 30 days
- Manual deletion of ``~/.local/share/openhcs/cache/colormap_enum.json``

**Disabling cache**:

.. code-block:: python

   # Disable caching (always discover fresh)
   NapariColormap = create_colormap_enum(enable_cache=False)

   # Lazy mode (no caching, deferred discovery)
   NapariColormap = create_colormap_enum(lazy=True)

Component Enum Caching
~~~~~~~~~~~~~~~~~~~~~~

The ``_create_enums()`` function in ``openhcs.constants.constants`` generates three
enums from the config system:

- ``AllComponents`` - All available microscope components
- ``VariableComponents`` - Components that can be varied in experiments
- ``GroupBy`` - Components that can be used for grouping

**With caching**:

.. code-block:: python

   from openhcs.constants import AllComponents, VariableComponents, GroupBy

   # First import: Full config parsing + cache save
   # Subsequent imports: Load from cache (instant!)

**Cache invalidation**:

- OpenHCS version changes
- Cache age > 7 days (shorter than colormap cache)
- Manual deletion of ``~/.local/share/openhcs/cache/component_enums.json``

**Implementation details**:

The cache stores all three enums as a single JSON structure:

.. code-block:: json

   {
     "cache_version": "1.0",
     "version": "0.3.7",
     "timestamp": 1730419200,
     "items": {
       "enums": {
         "all_components": {"Laser": "Laser", "Camera": "Camera", ...},
         "variable_components": {"Laser": "Laser", ...},
         "group_by": {"NONE": null, "Laser": "Laser", ...}
       }
     }
   }

Custom methods (like ``GroupBy.component``) are restored after deserialization.

Environment Variables
---------------------

``OPENHCS_SUBPROCESS_NO_GPU``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Controls GPU backend discovery in subprocess mode:

.. code-block:: bash

   # Disable GPU backends in subprocesses (faster startup)
   export OPENHCS_SUBPROCESS_NO_GPU=1

Used by ``STORAGE_BACKENDS`` custom discovery function to skip GPU-dependent
backends (``fiji_stream``, ``napari_stream``) in subprocess mode.

``OPENHCS_DISABLE_REGISTRY_CACHE``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Disable all registry caching:

.. code-block:: bash

   # Disable caching (useful for development)
   export OPENHCS_DISABLE_REGISTRY_CACHE=1

Troubleshooting
---------------

Empty Registry After Discovery
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom**: Registry is empty even after accessing it

**Causes**:

1. **No plugins found**: Check discovery package path
2. **Import errors**: Check logs for import failures
3. **Wrong base class**: Plugins don't inherit from expected base

**Solution**:

.. code-block:: python

   import logging
   logging.basicConfig(level=logging.DEBUG)

   # Check discovery package
   print(f"Discovery package: {MICROSCOPE_HANDLERS._config.discovery_package}")

   # Force discovery and check logs
   _ = list(MICROSCOPE_HANDLERS.keys())

Plugin Not Discovered
~~~~~~~~~~~~~~~~~~~~~~

**Symptom**: Specific plugin missing from registry

**Causes**:

1. **Import error**: Plugin module has syntax/import errors
2. **No key attribute**: Plugin missing required attribute (e.g., ``_microscope_type``)
3. **Abstract class**: Plugin is abstract (has ``@abstractmethod``)
4. **Wrong location**: Plugin not in discovery package

**Solution**:

.. code-block:: python

   # Check if plugin can be imported
   from openhcs.microscopes import my_plugin  # Does this work?

   # Check if plugin has required attribute
   print(hasattr(MyPlugin, '_microscope_type'))

   # Check if plugin is abstract
   import inspect
   print(inspect.isabstract(MyPlugin))

Cache Not Invalidating
~~~~~~~~~~~~~~~~~~~~~~~

**Symptom**: Changes to plugin code not reflected

**Causes**:

1. **Mtime not updated**: File saved but mtime unchanged
2. **Cache validation disabled**: ``check_mtimes=False``
3. **Version unchanged**: OpenHCS version not bumped

**Solution**:

.. code-block:: bash

   # Clear cache manually
   rm -rf ~/.local/share/openhcs/cache/

   # Or touch the file to update mtime
   touch openhcs/microscopes/my_plugin.py

Slow Discovery Performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom**: Discovery takes several seconds

**Causes**:

1. **Caching disabled**: ``enable_cache=False``
2. **Cache invalidated**: Version/mtime changes
3. **Large plugin set**: Many modules to scan
4. **Slow imports**: Plugins import heavy dependencies

**Solution**:

.. code-block:: python

   # Enable caching (default)
   MICROSCOPE_HANDLERS = LazyDiscoveryDict(enable_cache=True)

   # Profile discovery
   import time
   start = time.time()
   _ = list(MICROSCOPE_HANDLERS.keys())
   print(f"Discovery took {time.time() - start:.3f}s")

   # Check cache status
   if MICROSCOPE_HANDLERS._cache_manager:
       print(f"Cache path: {MICROSCOPE_HANDLERS._cache_manager._cache_path}")

Third-Party Plugin Development
-------------------------------

Creating a Plugin Package
~~~~~~~~~~~~~~~~~~~~~~~~~

Third-party packages can extend OpenHCS by providing plugins:

.. code-block:: python

   # my_openhcs_plugin/microscopes.py
   from openhcs.microscopes.microscope_base import MicroscopeHandler

   class MyMicroscopeHandler(MicroscopeHandler):
       _microscope_type = 'my_microscope'

       def parse_filename(self, filename):
           # Implementation
           pass

**Installation**:

.. code-block:: bash

   pip install my-openhcs-plugin

**Discovery**: Plugins are automatically discovered if they:

1. Inherit from the correct base class
2. Are installed in the Python environment
3. Are in a package that gets imported

Plugin Discovery Hooks
~~~~~~~~~~~~~~~~~~~~~~~

For third-party plugins in separate packages, add a discovery hook:

.. code-block:: python

   # In your plugin package's __init__.py
   def register_plugins():
       """Register third-party plugins with OpenHCS."""
       # Import your plugin modules to trigger registration
       from . import microscopes
       from . import backends

   # Auto-register on import
   register_plugins()

Then users just need to import your package:

.. code-block:: python

   import my_openhcs_plugin  # Auto-registers plugins

   # Now available in registries
   handler = MICROSCOPE_HANDLERS['my_microscope']

Best Practices
--------------

1. **Use caching in production**: Significantly faster startup
2. **Pre-discover in main thread**: Before multiprocessing
3. **Handle import errors gracefully**: Don't crash on missing dependencies
4. **Provide clear error messages**: Help users debug plugin issues
5. **Document required attributes**: Make it clear what plugins need
6. **Test in subprocess mode**: Ensure plugins work in ZMQ/multiprocessing
7. **Version your plugins**: Use semantic versioning
8. **Minimize import-time side effects**: Keep imports fast

Performance Benchmarks
-----------------------

Typical discovery times (on modern hardware):

==================  ==============  ===============
Registry            Without Cache   With Cache
==================  ==============  ===============
Microscope (4)      ~30ms           ~3ms (10x)
Storage (6)         ~40ms           ~4ms (10x)
ZMQ Servers (3)     ~20ms           ~2ms (10x)
Library (4)         ~50ms           ~5ms (10x)
Format (2)          ~15ms           ~2ms (7x)
==================  ==============  ===============

**Total startup improvement**: ~155ms → ~16ms (9.7x faster)

Related Documentation
---------------------

- :doc:`plugin_registry_system` - Core architecture and usage
- :doc:`../api/core/auto_register_meta` - API reference
- :doc:`../api/core/registry_cache` - Caching API reference
