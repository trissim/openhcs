Serialization boundaries
========================

OpenHCS uses three distinct serialization authorities. They are not fallback
formats for one another.

Python source
-------------

pycodify owns generic executable-Python generation, import construction,
formatter registration, collision handling, and clean versus explicit output.
OpenHCS owns formatters and round-trip policy for its domain declarations. Code
documents are user-editable source and must pass the normal validation boundary
before application.

JSON transport
--------------

``openhcs.serialization.json.to_jsonable`` is the OpenHCS-owned nominal JSON
projection for agent capabilities and UI-bridge transport. Its singledispatch
implementations cover JSON scalars, collections, paths, enums, callables,
dataclass instances, and registered nominal types. A registered type projects
through its declared registry key; a callable projects through a stable import
identity.

This representation is for bounded transport and inspection. It does not
replace typed Python declarations or compiled plans, and consumers must not add
parallel class-name/enum tables.

Image files
-----------

``ImageFileFormat`` owns OpenHCS image-container semantics: suffix ownership,
pixel preparation, reading/writing, source dtype/intensity scale, and declared
pixel-band axes. PolyStore owns the backend and generic persistence mechanism,
not the microscopy meaning of loaded pixels.

Extension rule
--------------

Extend the declaration that owns the representation:

- a pycodify formatter for new Python-source syntax;
- a ``to_jsonable.register`` implementation for a stable transport value;
- an ``ImageFileFormat`` subclass for image-container pixel semantics.

Do not add a generic catch-all that stringifies unknown values. Unsupported
values fail at the boundary so missing public semantics remain visible.
