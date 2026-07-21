Function library
================

OpenHCS integrates image-processing callables from CPU and GPU libraries plus
native and CellProfiler-compatible implementations. A function is usable when
its callable declaration provides the metadata needed by the compiler and UI.

Callable contract
-----------------

``CallableContract`` is the compiler-visible authority for a function. It can
declare:

- input and output memory types;
- artifact inputs and outputs;
- runtime-bound parameters;
- required variable components and allowed grouping;
- ``ProcessingContract``;
- runtime adapter and image execution mode;
- axis- or plate-level execution scope;
- compiler preparation hooks.

Image transport
---------------

Image inputs use 3D transport even for a single logical plane. The leading axis
is a declared variable-component plane axis, not always physical Z.
``ProcessingContract`` determines whether the callable runs per slice, over the
full stack, flexibly, or as a volumetric-to-slice operation.

Functions may return images, object labels, measurements, relationships,
tables, grids, files, or structured combinations. “All functions return 3D
arrays” is not part of the current contract.

Memory backends
---------------

ArrayBridge owns memory-type detection and conversion for NumPy, CuPy, PyTorch,
JAX, TensorFlow, pyclesperanto, and other registered frameworks. OpenHCS
decorators attach memory and processing metadata to callables; compilation plans
the required conversions. Generic converter internals belong in ArrayBridge
documentation.

Function patterns
-----------------

Functions appear in a ``FunctionStep`` as a callable, callable/kwargs tuple,
chain, or dictionary pattern. Signature analysis validates user kwargs while
runtime-bound parameters remain owned by the contract and adapter.

Finding functions
-----------------

The GUI and agent interfaces query the current function registry. Prefer those
surfaces over copying backend module paths into scripts: processing-library
versions and discovered functions can vary by installed extras and hardware.

Custom functions
----------------

A custom callable should declare its memory type and processing contract using
the supported OpenHCS decorator. Add artifact declarations when the callable
consumes or produces semantic values beyond main-flow image data. The compiler
will then expose the same declaration to UI forms, code transport, memory
planning, and runtime execution.

See :doc:`../user_guide/custom_functions`,
:doc:`../architecture/processing_semantics`, and
:doc:`../architecture/artifact_contract_system`.
