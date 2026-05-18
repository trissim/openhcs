# Backend Parameter Request Records Refactor - 2026-05-18

## Advisor Evidence

Full-repo scan flagged repeated threaded semantic parameter families. A clear
example is Ashlar position generation:

- `processing/backends/pos_gen/ashlar_main_cpu.py`
- `ArrayEdgeAligner.__init__`
- `ashlar_compute_tile_positions_cpu`
- analogous GPU family in `ashlar_main_gpu.py`

Repeated parameter family includes:

- `image_stack`
- `max_error`
- `max_permutation_tries`
- `max_shift`
- `min_permutation_samples`
- `permutation_samples`
- `permutation_upsample`
- `pixel_size`
- `randomize`
- `upsample_factor`
- `verbose`
- `window_size_factor`

## Current Problem

The public processing function and internal aligner constructors thread a large
semantic bundle by positional/keyword parameters. CPU and GPU versions duplicate
the same conceptual configuration.

This increases drift risk and makes generated function signatures noisy.

## Target Shape

Introduce request/config records:

```python
@dataclass(frozen=True, slots=True)
class AshlarAlignmentConfig:
    pixel_size: float = 1.0
    max_shift: float = 30.0
    stitch_alpha: float = 0.05
    max_error: float | None = None
    randomize: bool = False
    verbose: bool = False
    upsample_factor: int = 50
    permutation_upsample: int = 1
    permutation_samples: int = 1000
    min_permutation_samples: int = 10
    max_permutation_tries: int = 100
    window_size_factor: float = 0.15

@dataclass(frozen=True, slots=True)
class AshlarPositionRequest:
    image_stack: np.ndarray
    grid_dimensions: tuple[int, int]
    overlap_ratio: float
    alignment: AshlarAlignmentConfig
```

Use the existing callable request-binding substrate only if the processing
function must keep the public expanded kwargs for GUI/generated pipeline
compatibility.

## Phase 1: CPU Characterization

Add tests around `ashlar_compute_tile_positions_cpu` for a small synthetic image
stack:

- output array shape;
- positions list shape/order;
- default config path;
- one non-default parameter affects internal config construction.

If full algorithm tests are too expensive, mock the aligner construction and
verify request/config projection.

## Phase 2: CPU Internal Config

Refactor `ArrayEdgeAligner.__init__` to accept:

```python
alignment_config: AshlarAlignmentConfig
```

Keep a compatibility constructor or classmethod if necessary:

```python
ArrayEdgeAligner.from_kwargs(...)
```

## Phase 3: Public Function Compatibility

Keep `ashlar_compute_tile_positions_cpu(...)` public signature stable unless
there is a deliberate GUI-breaking migration.

If stable signature is required, use `@callable_request` or a local request
builder so the implementation boundary receives `AshlarPositionRequest`.

## Phase 4: GPU Mirror

Repeat CPU pattern for `ashlar_main_gpu.py`, but only after CPU tests pass.

Extract any shared config record into a common module:

- `processing/backends/pos_gen/ashlar_config.py`

Do not create two nearly identical CPU/GPU config records.

## Phase 5: Broader Parameter Families

After Ashlar, use full-repo advisor output to identify the next repeated
parameter families. Prefer high-value backend families over public GUI wrappers.

## Risks

- Processing function signatures are GUI-visible. Do not collapse public kwargs
  without compatibility support.
- Numpy/CuPy array types may differ; keep data payload outside pure config.
- The CPU/GPU modules may intentionally diverge in some defaults. Verify before
  sharing defaults.

## Verification Gates

```bash
.venv/bin/python -m pytest \
  tests/unit/test_callable_contract.py \
  tests/unit/test_function_step_execution_plan.py \
  tests/unit/test_cellprofiler_generated_pipeline_execution.py -q

python -m nominal_refactor_advisor \
  openhcs/processing/backends/pos_gen/ashlar_main_cpu.py \
  openhcs/processing/backends/pos_gen/ashlar_main_gpu.py
```

## Completion Criteria

- Ashlar parameter bundle has one typed config/request authority.
- Public processing signatures remain compatible or are explicitly migrated.
- CPU/GPU defaults are shared where semantically identical.
- Advisor repeated-parameter findings for Ashlar are removed.
