---
title: 'metaclass-registry: Zero-Boilerplate Plugin Registration via Python Metaclasses'
tags:
  - Python
  - plugin systems
  - metaclasses
  - software architecture
authors:
  - name: Tristan Simas
    orcid: 0000-0002-6526-3149
    affiliation: 1
affiliations:
  - name: McGill University
    index: 1
date: 13 January 2026
bibliography: paper.bib
---

# Summary

`metaclass-registry` provides automatic class registration for Python plugin systems. When a developer defines a class inheriting from a base with `AutoRegisterMeta`, the class is immediately available in the registry—no configuration files, no decorator calls, no manual dictionary insertion. The complete registration for a storage backend is:

```python
class ZarrBackend(StorageBackend):
    _backend_type = "zarr"
```

The class is now discoverable via `StorageBackend.__registry__["zarr"]`. The library handles lazy discovery (plugins aren't imported until the registry is accessed), disk caching with version and mtime validation, thread-safe access, secondary registries for related metadata, and automatic key extraction from class names.

# Statement of Need

Applications with extensible architectures—storage backends, file format handlers, hardware drivers, GUI widgets—require plugin registration. The standard Python approaches each impose friction:

**Setuptools entry points** require `pyproject.toml` configuration per plugin and package installation. For application-internal plugins where the developer controls all code, this adds deployment complexity without benefit.

**Explicit registration** (`BACKENDS["zarr"] = ZarrBackend` or `@register`) is error-prone. Add a class, forget the registration call, and the plugin silently fails to appear. In a codebase with 40+ plugin classes across 6 registries, this becomes a recurring source of bugs.

**Manual discovery** (`pkgutil.iter_modules` + `importlib.import_module`) requires ~20 lines of boilerplate per registry, repeated identically across each plugin system.

`metaclass-registry` eliminates all three. The base class definition specifies the registry configuration once; every subclass auto-registers. In OpenHCS (a microscopy image analysis platform), this pattern is used for storage backends, microscope format handlers, metadata parsers, array converters, widget adapters, and parameter type handlers—6 distinct registries, 40+ plugin classes, zero manual registration.

# State of the Field

**Stevedore** [@stevedore] manages plugins via setuptools entry points. It provides driver managers and extension managers but requires entry point configuration in `pyproject.toml` for each plugin class. This is appropriate for third-party extensions but creates friction for internal plugins.

**Pluggy** [@pluggy] powers pytest's hook system. It provides sophisticated hook calling with ordering and wrappers but requires `@hookimpl` decorators and explicit `register()` calls. It targets hook-based extensibility rather than class registration.

**Entry points** (via `importlib.metadata`) are the standard for cross-package discovery. They require package installation and metadata configuration.

`metaclass-registry` targets a different use case: **application-internal plugins where all code is controlled by the developer**. No configuration files. No installation steps. No registration calls. Define the class, set the key attribute, done.

| Feature | metaclass-registry | Stevedore | pluggy |
|---------|-------------------|-----------|--------|
| Zero config files | ✓ | ✗ | ✗ |
| Zero registration calls | ✓ | ✗ | ✗ |
| Lazy discovery | ✓ | ✓ | ✗ |
| Disk caching with mtime | ✓ | ✗ | ✗ |
| Secondary registries | ✓ | ✗ | ✗ |
| Works without install | ✓ | ✗ | ✓ |

# Software Design

The base class declares registry configuration via class attributes:

```python
class BackendBase(metaclass=AutoRegisterMeta):
    __registry_key__ = '_backend_type'
    __registry_config__ = RegistryConfig(
        registry_dict=LazyDiscoveryDict(),
        discovery_package='openhcs.io',
    )
```

When Python processes a subclass definition, `AutoRegisterMeta.__new__` extracts the key from the `_backend_type` attribute and inserts the class into the registry. Abstract classes (with unimplemented abstract methods) are skipped.

**Lazy discovery** defers plugin imports until first registry access. `LazyDiscoveryDict` overrides `__getitem__`, `__contains__`, and iteration methods to trigger `pkgutil.iter_modules` scanning on first use. Results are cached to `~/.cache/metaclass-registry/` with version validation and file modification time checking—subsequent application starts load from cache without re-scanning.

**Example**: An OpenHCS application with 40 storage backends doesn't import all 40 at startup. When the user selects "Zarr" from the UI, the registry is accessed, lazy discovery scans the `openhcs.io` package, imports only the Zarr backend, and caches the result. On the next application start, the cache is used instead of re-scanning. If a developer adds a new backend, the cache is invalidated (via mtime checking) and the scan runs again.

**Secondary registries** handle related metadata. A microscope handler registers both itself and its metadata parser:

```python
class MicroscopeHandler(metaclass=AutoRegisterMeta):
    __registry_key__ = '_microscope_type'
    __secondary_registries__ = [
        SecondaryRegistry(
            registry_dict=METADATA_HANDLERS,
            key_source=PRIMARY_KEY,
            attr_name='_metadata_handler_class'
        )
    ]

class ImageXpressHandler(MicroscopeHandler):
    _microscope_type = 'imagexpress'
    _metadata_handler_class = ImageXpressMetadata
```

Both `MicroscopeHandler.__registry__['imagexpress']` and `METADATA_HANDLERS['imagexpress']` are populated automatically.

**Thread safety** uses `threading.RLock` around discovery. The reentrant lock handles recursive access when module imports during discovery trigger registry lookups.

# Research Impact Statement

`metaclass-registry` is used across the OpenHCS microscopy platform:

- **openhcs**: `BackendBase` (storage backends), `MicroscopeHandler` (format handlers), `METADATA_HANDLERS` (secondary registry)
- **arraybridge**: `ConverterBase` (6 GPU framework converters auto-generated from config)
- **pyqt-reactor**: `WidgetMeta` (widget adapters), `ParameterInfoMeta` (parameter type handlers)

The library consolidated ~200 lines of duplicated metaclass code into a single, tested implementation. Adding a new storage backend or microscope handler requires only the class definition—no registration code, no configuration updates.

# AI Usage Disclosure

Generative AI (Claude) assisted with code generation and documentation. All content was reviewed and tested against the implementation.

# Acknowledgements

This work was supported in part by the Fournier lab at the Montreal Neurological Institute, McGill University.

# References
