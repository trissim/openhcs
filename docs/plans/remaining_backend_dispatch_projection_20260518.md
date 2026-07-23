# Remaining Backend Dispatch And Projection Refactor - 2026-05-18

## Evidence

From `/tmp/advisor_openhcs_remaining_after_cp_public_20260518.txt`, active
backend dispatch/projection findings include:

- `openhcs/processing/backends/processors/pyclesperanto_processor.py`
  `create_projection`: inline dispatch over `max_projection` and
  `mean_projection`.
- `openhcs/processing/backends/processors/cupy_processor.py`
  `spatial_bin_2d`, `spatial_bin_3d`, `create_projection`, `edge_magnitude`.
- `openhcs/processing/backends/processors/numpy_processor.py`
  `spatial_bin_2d`, `spatial_bin_3d`, `create_projection`.
- `openhcs/processing/backends/processors/jax_processor.py`
  `create_projection`.
- `openhcs/processing/backends/processors/tensorflow_processor.py`
  `create_projection`.
- `openhcs/processing/backends/processors/torch_processor.py`
  `create_projection`.
- `openhcs/processing/backends/assemblers/assemble_stack_cpu.py` and
  `assemble_stack_cupy.py`: stack assembly mode dispatch.
- `openhcs/processing/backends/enhance/jax_nlm_processor.py` and
  `pyclesperanto_processor.py` had top numeric dispatch findings already
  cleared in checkpoints `dc94c6e9` and `f6280c62`.

## Problem

Backend behavior is still selected through local literal ladders. That scatters
domain axes like projection method, binning mode, edge method, and assembly mode
across each backend implementation. The same semantic operation families appear
in multiple memory backends, so local branch ladders make parity and backend
extension harder.

## Target Shape

- Introduce explicit closed-domain enums for shared backend axes:
  `ProjectionMethod`, `SpatialBinningMode`, `EdgeMagnitudeMode`,
  `StackAssemblyMode` where they do not already exist.
- Use nominal strategy families or typed operation tables when behavior is
  backend-specific but the axis is shared.
- Keep backend-specific implementation details local, but make the dispatch key
  and supported surface common.
- Avoid generic dict dispatch when cases own behavior; prefer class methods,
  dataclass operation specs, or registered strategies.

## Phases

1. Inventory backend processor literal axes and group by semantic operation:
   projection, spatial binning, edge magnitude, mask application, assembly.
2. Start with `create_projection` across NumPy/CuPy/JAX/TensorFlow/Torch/
   pyclesperanto because the advisor reports the same inline dispatch family in
   six files.
3. Add one shared projection-method enum/spec and backend-local adapters.
4. Refactor `spatial_bin_2d`/`spatial_bin_3d` into a shared mode authority for
   CPU/GPU backends.
5. Refactor stack assembly mode dispatch in CPU/CuPy assemblers.
6. Re-run focused advisor on each touched backend file and update this plan with
   per-checkpoint deltas.

## Verification Gates

```bash
.venv/bin/python -m py_compile \
  openhcs/processing/backends/processors/numpy_processor.py \
  openhcs/processing/backends/processors/cupy_processor.py \
  openhcs/processing/backends/processors/jax_processor.py \
  openhcs/processing/backends/processors/tensorflow_processor.py \
  openhcs/processing/backends/processors/torch_processor.py \
  openhcs/processing/backends/processors/pyclesperanto_processor.py

timeout 120 .venv/bin/python -m nominal_refactor_advisor \
  openhcs/processing/backends/processors/numpy_processor.py \
  openhcs/processing/backends/processors/cupy_processor.py \
  openhcs/processing/backends/processors/jax_processor.py \
  openhcs/processing/backends/processors/tensorflow_processor.py \
  openhcs/processing/backends/processors/torch_processor.py \
  openhcs/processing/backends/processors/pyclesperanto_processor.py
```

Use targeted backend tests when present; otherwise add characterization tests
around projection/binning behavior before changing implementation.

## Risks

- GPU backend imports may be unavailable in CI/dev shells; use import-smoke tests
  that tolerate optional dependencies and direct py_compile gates.
- Some branches may be public string API. Preserve accepted string inputs until a
  typed settings layer owns migration.
