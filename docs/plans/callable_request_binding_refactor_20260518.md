# Callable Request Binding Refactor Plan

## Goal

Add a generic OpenHCS callable-contract feature that lets a public processing
function expose ordinary keyword parameters while its implementation receives a
typed request/context record.

Primary proof case:

- `openhcs/interop/cellprofiler/image_export.py`
- `SaveImagesRequest`
- `save_images`
- `save_images_3d`

Current pressure:

- CP-wide advisor still reports a repeated threaded parameter family across
  `save_images` and `save_images_3d`.
- The repeated family is real, but the public signatures are also real OpenHCS
  function-contract surface.
- A one-off wrapper in `image_export.py` would be duct tape. The durable fix is
  a core callable-contract request-binding substrate.

## Verified Current Seams

### Callable Metadata

`openhcs/core/callable_contract.py` owns compiler-visible callable metadata.

Relevant existing fields and attributes:

- `CallableContract.from_callable(func)`
- `CallableProjection.from_callable(func)`
- `CallableMetadataReader`
- `attach_callable_contract_metadata(...)`
- `PROCESSING_CONTRACT_ATTR`
- `DECLARED_PROCESSING_CONTRACT_ATTR`
- `RAW_PROCESSING_FUNCTION_ATTR`
- `PROCESSING_PREPARE_ATTR`

This is the correct location for a new metadata attribute such as:

- `CALLABLE_REQUEST_BINDING_ATTR = "__openhcs_callable_request_binding__"`

### Function Pattern Compilation

`openhcs/core/function_patterns.py` lowers public function-pattern syntax into
runtime invocation records:

- `NormalizedFunctionItem`
- `CompiledFunctionInvocation`
- `_split_function_item(...)`
- `_freeze_runtime_kwargs(...)`

This layer should preserve the user-facing kwargs unchanged. It should not
eagerly instantiate request records because runtime artifact injection and
special input injection can still add values later.

### Runtime Kwarg Filtering

CellProfiler runtime currently filters and coerces invocation kwargs in:

- `CallableInvocationKwargSpec.from_callable(func)`
- `CallableInvocationKwargSpec.coerce_kwargs(...)`
- `_callable_parameters(func)`
- `_callable_type_hints(func)`

This is currently CP-local in
`openhcs/interop/cellprofiler/runtime/module_execution.py`.

Request binding should not remain CP-local. The generic request-binding object
should expose a runtime method that accepts a callable plus raw kwargs and
returns:

- positional image argument unchanged,
- implementation kwargs,
- request object injected if configured,
- enum coercion preserved.

### Runtime Invocation

Actual calls eventually flow through:

- `RuntimeCallablePolicy.call(...)`
- `ProcessingContract.execute(...)`
- `CellProfilerFunctionContractExecutor.execute_pure_2d(...)`
- `CellProfilerFunctionContractExecutor.execute_pure_3d(...)`
- `_execute_pure_2d_slice(...)`

The request binding must be applied before the callable is invoked, including
per-slice execution. If `image` is a request field, the per-slice image must be
the value used in that slice's request object.

## Target Architecture

### Request Binding Declaration

Add a core declaration API:

```python
@callable_request(
    SaveImagesRequest,
    public_fields=("image", "filename_prefix", "file_format", "bit_depth", "use_compression"),
    request_parameter="request",
)
def save_images_impl(request: SaveImagesRequest, image_type: ImageType = ImageType.IMAGE):
    ...
```

The implementation callable receives `request`.

The public callable contract still exposes expanded fields:

- `image`
- `filename_prefix`
- `file_format`
- `bit_depth`
- `use_compression`
- plus any local parameters such as `image_type`.

### Runtime Binding Record

Add a nominal record in core, likely in `openhcs/core/callable_contract.py` or a
new `openhcs/core/callable_request_binding.py`:

```python
@dataclass(frozen=True, slots=True)
class CallableRequestBinding:
    request_type: type
    request_parameter: str
    public_fields: tuple[str, ...]

    def bind_kwargs(self, *, image: object, kwargs: Mapping[str, object]) -> Mapping[str, object]:
        ...
```

Required behavior:

- fail loudly if `request_parameter` is absent from the implementation
  signature;
- fail loudly if a public field cannot be populated;
- allow defaults from the request dataclass or public wrapper declaration;
- preserve non-request local kwargs;
- be picklable for multiprocessing and compiled pipeline execution;
- not depend on CellProfiler.

### Public Signature Projection

The hard part is signature projection. The implementation should not manually
spell a compatibility wrapper that the advisor sees as a transport shell.

Preferred options:

1. Attach `__signature__` to the decorated callable so introspection sees the
   expanded public signature while the callable body receives a request object.
2. Generate the public wrapper from a core decorator and mark it as generated
   callable-contract surface.
3. Extend `CallableContract` with a `public_signature`/`request_binding` view and
   update all code that inspects parameters to use that view.

Option 3 is the cleanest but touches the most call sites. Option 1 is smaller
but must be tested with `inspect.signature`, function-pattern parsing, and CP
runtime kwarg filtering.

## Implementation Sequence

### Phase 1: Pin Current SaveImages Behavior

Add focused tests before changing signatures:

- `save_images` converts normal 2D images with metadata.
- `save_images` converts mask/cropping outputs to binary.
- `save_images_3d` accepts `TIFF`, `NPY`, and `H5`.
- `save_images_3d` rejects non-volumetric file formats.
- generated CP pipelines still bind SaveImages settings into kwargs.

Suggested test file:

- `tests/unit/test_cellprofiler_save_images_export.py`

### Phase 2: Add Core Request Binding Metadata

Add:

- `CALLABLE_REQUEST_BINDING_ATTR`
- `CallableRequestBinding`
- `callable_request(...)` decorator or equivalent metadata helper
- `CallableContract.request_binding`

Verification:

- `CallableContract.from_callable` sees the binding.
- binding metadata survives `FunctionReference` preservation.
- bad declarations fail loudly.

Suggested tests:

- `tests/unit/test_callable_contract.py`
- new `tests/unit/test_callable_request_binding.py`

### Phase 3: Add Runtime Kwarg Projection

Add a core projection object, not CP-local:

```python
CallableInvocationKwargs.from_callable(func).bind(image, kwargs)
```

Then migrate CP `CallableInvocationKwargSpec` to delegate to it or to consume
the new binding record.

Critical cases:

- enum coercion still works;
- unsupported kwargs still drop unless the callable accepts `**kwargs`;
- request fields are removed from implementation kwargs after request
  construction;
- local kwargs remain as direct kwargs.

### Phase 4: Migrate SaveImages as First Consumer

Target shape:

```python
@numpy(contract=ProcessingContract.PURE_2D)
@special_outputs(...)
@callable_request(SaveImagesRequest, request_parameter="request")
def save_images(
    request: SaveImagesRequest,
    image_type: ImageType = ImageType.IMAGE,
) -> tuple[np.ndarray, SaveMetadata]:
    ...
```

For `save_images_3d`, either:

- reuse `SaveImagesRequest`, or
- introduce `SaveImages3DRequest` only if 3D validation belongs in the request
  type.

Do not duplicate implementation logic between 2D and 3D.

### Phase 5: Remove Public Signature Advisor Finding

Run the advisor on:

```bash
.venv/bin/python -m nominal_refactor_advisor \
  openhcs/core/callable_contract.py \
  openhcs/core/function_patterns.py \
  openhcs/interop/cellprofiler/image_export.py \
  openhcs/interop/cellprofiler/runtime/module_execution.py
```

If the advisor flags generated wrapper transport shells, move the generation
marker into the core request-binding substrate rather than hand-editing
`image_export.py`.

## Risk Points

- `inspect.signature` is used in several places. Any signature override must be
  verified against those exact call sites.
- Function references may preserve `__dict__` metadata but not arbitrary runtime
  wrapper state. Request binding metadata must be serializable.
- PURE_2D per-slice execution must build request objects with the sliced image,
  not the original stack.
- The public function contract must remain understandable to GUI/config editors.
- Generated CP pipelines may serialize enums and defaults; request binding must
  preserve those values.

## Verification Gates

Focused:

```bash
.venv/bin/python -m pytest \
  tests/unit/test_callable_contract.py \
  tests/unit/test_function_step_execution_plan.py \
  tests/unit/test_cellprofiler_generated_pipeline_execution.py \
  tests/unit/test_runner_cellprofiler_compatibility.py \
  tests/unit/test_cellprofiler_save_images_export.py -q
```

Advisor:

```bash
.venv/bin/python -m nominal_refactor_advisor \
  openhcs/core/callable_contract.py \
  openhcs/core/function_patterns.py \
  openhcs/interop/cellprofiler/image_export.py \
  openhcs/interop/cellprofiler/runtime/module_execution.py
```

Full:

```bash
.venv/bin/python -m pytest tests/unit -q
```

## Completion Criteria

- `SaveImages` implementation receives a typed request record.
- Public callable contract still exposes the same user/generated-pipeline
  kwargs.
- CP generated pipeline compatibility remains green.
- The SaveImages repeated threaded-parameter advisor finding is gone without a
  CellProfiler-only wrapper.
- The request-binding substrate is generic and documented through tests.

## Execution Note

Implemented a generic `callable_request(...)` substrate in
`openhcs/core/callable_contract.py`:

- `CallableRequestBinding` stores request metadata;
- public signatures are projected through `__signature__`;
- dataclass defaults are accepted as authoritative default records;
- `CallableContract` now exposes `request_binding`.

`SaveImages` now uses `SaveImagesRequest` as its implementation boundary while
retaining the public kwargs expected by generated pipelines and runtime
introspection. Focused tests cover public signature projection, metadata
visibility, 2D conversion, mask/cropping conversion, and 3D validation.

The SaveImages repeated threaded-parameter advisor finding is removed. Remaining
advisor findings in `callable_contract.py` are older callable metadata probing
debt and should be handled as a separate core metadata-authority campaign.
