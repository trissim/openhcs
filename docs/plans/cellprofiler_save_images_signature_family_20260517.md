# CellProfiler SaveImages Signature Family Plan

## Goal

Resolve the remaining advisor finding for the public `SaveImages` processing
signature family without breaking OpenHCS callable discovery or generated
pipeline compatibility.

Current advisor finding:

- "Repeated threaded semantic parameter family"
- evidence:
  `openhcs/interop/cellprofiler/image_export.py:save_images`
  and `save_images_3d`
- repeated parameters:
  `bit_depth`, `file_format`, `filename_prefix`, `image`, `use_compression`

## Current State

`SaveImagesRequest` now exists internally and owns:

- image,
- filename prefix,
- file format,
- bit depth,
- compression flag,
- conversion strategy,
- filename projection,
- metadata projection.

The advisor finding remains because the duplicated parameter family is still
present on the public processing function signatures. This is expected: those
signatures are part of the OpenHCS function contract surface and should not be
changed casually.

## Constraint

Do not simply replace public parameters with `request: SaveImagesRequest` in
`save_images` and `save_images_3d`.

That would likely break:

- generated pipeline kwargs,
- CellProfiler settings binding,
- function contract introspection,
- user-facing processing API shape,
- materialization metadata expectations.

The correct fix needs a wrapper-generation or invocation-adapter path.

## Refactor Options

### Option A: Generated Wrapper Migration

Keep ergonomic/internal implementation as:

```python
def save_images_impl(request: SaveImagesRequest, image_type: ImageType) -> ...
def save_images_3d_impl(request: SaveImagesRequest) -> ...
```

Expose OpenHCS-compatible generated wrappers:

```python
def save_images(image, filename_prefix, file_format, bit_depth, image_type, use_compression):
    return save_images_impl(SaveImagesRequest(...), image_type)
```

This alone may still trigger advisor wrapper findings unless the wrapper is
produced by an accepted generated-public-signature substrate.

Required supporting abstraction:

- a general "public processing signature adapter" declaration,
- tests proving the generated wrapper is the public contract while the request
  object owns implementation semantics.

### Option B: Function Contract Request Binding

Teach OpenHCS function contract introspection how to bind kwargs into a request
record before invoking the underlying function.

Target shape:

```python
@processing_request(SaveImagesRequest)
def save_images(request: SaveImagesRequest, image_type: ImageType = ImageType.IMAGE):
    ...
```

The callable contract would expose the request fields as public kwargs while the
Python implementation receives the nominal request.

This is the cleanest long-term option, but it touches core callable-contract
machinery and must be planned as a broader OpenHCS feature.

### Option C: Accept Public Signature Duplication

Keep the current public signatures and document this as accepted noise because:

- it is a deliberate public API compatibility boundary,
- internal implementation already uses `SaveImagesRequest`,
- changing public signatures is higher risk than the remaining duplication.

This is acceptable only if the team agrees that advisor count is not the sole
goal at public OpenHCS function boundaries.

## Recommended Sequence

### 1. Pin Current Public Behavior

Add focused tests for:

- 2D image conversion metadata,
- 2D mask/cropping binary output,
- 3D format validation,
- 3D metadata shape,
- generated pipeline compatibility for SaveImages settings.

Suggested files:

- `tests/unit/test_cellprofiler_generated_pipeline_execution.py`
- new `tests/unit/test_cellprofiler_save_images_export.py` if needed

### 2. Audit Function Contract Introspection

Inspect:

- `openhcs/core/callable_contract.py`
- `openhcs/core/pipeline/function_contracts.py`
- `openhcs/core/steps/function_step.py`
- CellProfiler generator callsites for SaveImages settings.

Determine whether request-record expansion can be generic.

### 3. Choose A or B

Prefer Option B if request-record binding can be implemented generically without
breaking existing function patterns.

Prefer Option A if generated wrappers are already accepted elsewhere in the
codebase and can be made advisor-visible as generated compatibility surfaces.

Do not implement either in `image_export.py` alone.

### 4. Migrate SaveImages as First Consumer

Once the generic path exists:

- make `SaveImagesRequest` the implementation boundary,
- expose the same public CellProfiler kwargs through the adapter/generator,
- keep `save_images` and `save_images_3d` behavior identical.

## Verification

Focused:

```bash
.venv/bin/python -m nominal_refactor_advisor \
  openhcs/interop/cellprofiler/image_export.py \
  openhcs/core/callable_contract.py \
  openhcs/core/pipeline/function_contracts.py
```

```bash
.venv/bin/python -m pytest \
  tests/unit/test_cellprofiler_generated_pipeline_execution.py \
  tests/unit/test_cellprofiler_symbol_table.py \
  tests/unit/test_runner_cellprofiler_compatibility.py -q
```

Full:

```bash
.venv/bin/python -m pytest tests/unit -q
```

## Completion Criteria

- The repeated public SaveImages signature finding is gone or explicitly
  documented as accepted public API noise.
- Any signature migration is generic, tested, and not CellProfiler-only duct
  tape.
- SaveImages generated pipeline compatibility remains intact.

## Execution Note

The current checkpoint keeps the public `save_images` and `save_images_3d`
signatures unchanged and treats the remaining advisor finding as an accepted
public function-contract boundary. Internally, `SaveImagesRequest` already owns
conversion and metadata semantics. Eliminating the public threaded parameters
cleanly requires a generic request-record expansion feature in OpenHCS callable
contract introspection, not a one-off CellProfiler wrapper.
