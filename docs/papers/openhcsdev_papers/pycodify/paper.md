---
title: 'pycodify: Python Source Code as a Serialization Format'
tags:
  - Python
  - serialization
  - code generation
  - reproducibility
  - configuration
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

`pycodify` serializes Python objects to executable Python source code. The output includes correct imports, handles name collisions via aliasing, and can be `exec()`'d to recreate the original object:

```python
# In-memory object → executable source with imports
from openhcs.constants import DtypeConversion
from openhcs.core.config import PathPlanningConfig, PipelineConfig

config = PipelineConfig(
    path_planning=PathPlanningConfig(output_dir_suffix="_custom"),
    dtype=DtypeConversion.PRESERVE_INPUT,
)
```

The key insight: **Python source code is a serialization format.** Rather than inventing a format and writing loaders, `pycodify` emits code that Python itself interprets. The import system becomes the deserializer.

# Statement of Need

Serialization formats occupy a spectrum:

| Format | Diffable | Inspectable | Editable | Type-preserving | Cross-version |
|--------|:--------:|:-----------:|:--------:|:---------------:|:-------------:|
| pickle | ✗ | ✗ | ✗ | ✓ | ✗ |
| JSON/YAML | ✓ | ✓ | ✓ | ✗ | ✓ |
| Python source | ✓ | ✓ | ✓ | ✓ | ✓ |

Binary formats like `pickle` [@pickle] cannot be diffed, inspected, or edited without execution. Text formats like JSON lose type information—an Enum becomes a string, a Path becomes a string, a callable cannot be represented. `repr()` produces code fragments but not complete programs.

Python source code has all desired properties simultaneously: it is diffable (text), inspectable (readable), editable (valid syntax), type-preserving (Enums, Paths, callables serialize as themselves), and cross-version stable (if the code runs, deserialization succeeded).

The challenge is generating *complete* source—not just expressions, but the imports required to make those expressions executable. This requires solving import collisions: two classes named `Config` from different modules cannot both be imported without aliasing.

# State of the Field

`pickle` [@pickle], `dill` [@dill], and `cloudpickle` [@cloudpickle] serialize to opaque binary. `repr()` and `ast.unparse` [@ast] produce code fragments without imports. No existing tool produces complete, executable Python source with automatic import resolution and collision handling.

# Software Design

## Two-Pass Algorithm

Generating executable source requires knowing import aliases before emitting code. But aliases depend on detecting collisions, which requires visiting all types first. This creates a dependency: code generation requires alias resolution, but alias resolution requires traversing the object graph.

`pycodify` solves this with two passes:

1. **Collection pass**: Traverse the object, emit code fragments, collect `(module, name)` import pairs
2. **Resolution**: Detect collisions (same name, different modules), generate deterministic aliases
3. **Regeneration pass**: Re-traverse with resolved `name_mappings`, emit final code

This is not an optimization—it is structurally necessary. A single-pass algorithm cannot know whether `Config` needs aliasing until it has seen all types that might also be named `Config`.

**Example**: Consider a configuration object with two fields:
```python
config = PipelineConfig(
    path_planning=PathPlanningConfig(...),
    dtype=DtypeConversion.PRESERVE_INPUT,
)
```

The first pass collects:
- `(openhcs.core.config, PipelineConfig)`
- `(openhcs.core.config, PathPlanningConfig)`
- `(openhcs.constants, DtypeConversion)`

No collisions detected. The second pass emits:
```python
from openhcs.core.config import PipelineConfig, PathPlanningConfig
from openhcs.constants import DtypeConversion

config = PipelineConfig(
    path_planning=PathPlanningConfig(...),
    dtype=DtypeConversion.PRESERVE_INPUT,
)
```

If the object also contained a `PathPlanningConfig` from a different module (e.g., `custom_config.PathPlanningConfig`), the resolution pass would detect the collision and generate aliases:
```python
from openhcs.core.config import PathPlanningConfig as PathPlanningConfig_1
from custom_config import PathPlanningConfig as PathPlanningConfig_2
```

## Extensible Formatter Registry

Each type maps to a `SourceFormatter` that emits a `SourceFragment(code, imports)`. Formatters register via `__init_subclass__`—defining a formatter class automatically adds it to the registry:

```python
class EnumFormatter(SourceFormatter):
    priority = 70

    def can_format(self, value: Any) -> bool:
        return isinstance(value, Enum)

    def format(self, value: Enum, context: FormatContext) -> SourceFragment:
        cls = value.__class__
        import_pair = (cls.__module__, cls.__name__)
        name = context.name_mappings.get(import_pair, cls.__name__)
        return SourceFragment(f"{name}.{value.name}", frozenset([import_pair]))
```

Priority-based dispatch selects the most specific formatter. Domain extensions add formatters without modifying core code.

**Clean mode** omits fields matching defaults; **explicit mode** includes all fields for complete reproducibility.

# Research Impact Statement

`pycodify` powers code-based serialization in OpenHCS [@openhcs], enabling:

- **GUI round-trip editing**: Pipeline Editor serializes steps to Python; users edit code directly; changes reload into the GUI via `pyqt-reactor` [@pyqtreactor]
- **Remote execution**: ZMQ clients serialize pipeline configurations as Python code, avoiding pickle versioning issues across nodes
- **Reproducibility**: Pipeline scripts are human-readable records of exact processing parameters

The formatter registry enables domain extensions without modifying pycodify's core. OpenHCS adds formatters for:

- **FunctionStep**: Pipeline step objects with function patterns and processing configuration
- **Virtual module rewrites**: External library functions are rewritten to virtual module paths that include OpenHCS decorators:

```python
class OpenHCSCallableFormatter(SourceFormatter):
    def format(self, value: Any, context: FormatContext) -> SourceFragment:
        module = getattr(value, "__module__", None)
        name = getattr(value, "__name__", None)
        # Rewrite skimage.filters.gaussian → openhcs.skimage.filters.gaussian
        if _is_external_registered_function(value):
            module = f"openhcs.{module}"
        return SourceFragment(name, frozenset([(module, name)]))
```

- **Lazy dataclass bypass**: For dataclasses with `__getattribute__` interception (used for hierarchical config inheritance), formatters use `object.__getattribute__` to access raw field values without triggering lazy resolution:

```python
# In DataclassFormatter
if hasattr(instance, "_resolve_field_value"):
    # Bypass __getattribute__ to get raw None vs concrete value
    current_value = object.__getattribute__(instance, field_name)
else:
    current_value = getattr(instance, field_name)
```

This distinguishes explicitly-set values from inherited ones during serialization.

# AI Usage Disclosure

Generative AI assisted with drafting documentation. All content was reviewed and tested by human developers.

# Acknowledgements

This work was supported in part by the Fournier lab at the Montreal Neurological Institute, McGill University.

# References
