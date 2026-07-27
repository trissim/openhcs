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

Choosing preprocessing
----------------------

Choose preprocessing from the image defect and downstream measurement, not from
a generic recipe. Search the live registry with the intended operation, inspect
the returned callable's full description and contract, and keep the untreated
image available for comparison.

Percentile normalization remaps intensity endpoints. Per-plane normalization is
useful when each plane needs its own robust contrast scale, but it removes real
intensity differences between those planes. Stack normalization uses one pair of
endpoints for the assembled input and therefore preserves relative differences
within that invocation, but independently fitting each well or site can still
erase between-sample differences. Neither form preserves absolute intensity
calibration.

White top-hat filtering estimates local background from a spatial
structuring-element scale. It is appropriate for bright targets smaller than
that scale on slowly varying background. A radius near the target size can erase
or distort the target, while a radius that is too large may leave background
variation. For quantitative assays, it is often safer to use the transformed
image for detection and measure intensity on a raw or separately validated
illumination-corrected image.

Illumination or flat-field correction addresses repeatable acquisition bias.
Division models multiplicative shading; subtraction models additive background.
Estimate a field from comparable images, keep acquisition channels separate, and
avoid pooling conditions whose real spatial patterns could be learned as
background. Local top-hat subtraction and global flat-field correction solve
different problems and are not interchangeable.

Smoothing and denoising trade noise suppression for edge, texture, and
small-object fidelity. Choose a spatial scale below the smallest structure that
must remain detectable. A method that improves one segmentation preview may
still invalidate texture, morphology, or intensity measurements.

Validate preprocessing on a bounded, representative set spanning plate
positions, controls, weak and strong signals, and expected acquisition
variation. Compare raw and processed images and distributions; inspect clipping,
halos, residual gradients, lost small structures, and changed object intensity.
Then validate the downstream segmentation or measurement against assay-specific
expectations supplied by the domain expert. OpenHCS can expose and execute these
checks, but it cannot infer the expected biology from pixels alone.

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
