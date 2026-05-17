# Advisor-Guided Adjacent Refactor Overview

Date: 2026-05-16

## Purpose

This plan pack captures the current deep-refactor checkpoint and the remaining adjacent systems that should be cleaned up before more benchmark/runtime work is layered on top.

The goal is not to chase analyzer count mechanically. The goal is to use the advisor as a pressure test while removing real architectural debt at load-bearing seams.

## Current Checkpoint

- Stable pushed checkpoint: `9802b42e Register remaining runtime value semantics`.
- Full unit suite at checkpoint: `1485 passed, 10 warnings`.
- Targeted advisor scan files:
  - `openhcs/core/runtime_semantics.py`
  - `openhcs/core/runtime_values.py`
  - `openhcs/interop/cellprofiler/runtime/module_execution.py`
  - `openhcs/interop/cellprofiler/runtime/invocation.py`
- Current targeted advisor finding count: `10`.

Current finding profile:

- `repeated_hardcoded_strings`: `6`
- `trivial_forwarding_wrapper`: `2`
- `under_amortized_infrastructure`: `2`

This is the post-cleanup baseline. Any follow-up branch work should compare against this baseline before claiming architectural improvement.

## Hard Constraints

- Keep every code slice independently revertible and pushed after verification.
- Do not replace registry-key literals with constants unless the metaclass/advisor path is changed intentionally. A prior simple-constant attempt broke CellProfiler behavior and made advisor output worse.
- Do not introduce dangling private helpers to hide complexity. Extract named semantic collaborators or value objects instead.
- Do not use fallback/compatibility helpers as a permanent endpoint. Temporary bridges must have deletion criteria.
- Do not combine planner/runtime parity changes with broad GUI decomposition in the same commit.
- Run focused tests first, then the full non-GUI unit suite before committing risky runtime/planner changes.

## Plan Files In This Pack

- `registry_key_declaration_refactor_20260516.md`
  - Normalizes registry declaration debt without breaking `AutoRegisterMeta`.
- `cellprofiler_runtime_boundary_decomposition_20260516.md`
  - Splits remaining runtime/executor boundaries and removes forwarding bridges only when ownership actually moves.
- `runtime_artifact_semantics_consolidation_20260516.md`
  - Consolidates runtime semantic and artifact payload families where the advisor reports under-amortized infrastructure.
- `debug_source_binding_followthrough_20260516.md`
  - Finishes adjacent debug/source-binding UX hardening without mixing it into runtime parity work.

## Execution Order

1. Establish this plan pack as a committed checkpoint.
2. Investigate registry declaration substrate first, because the remaining string findings span multiple registered families.
3. If registry work requires changing metaclass/advisor expectations, add focused registry tests before implementation.
4. Move to runtime boundary decomposition only after registry behavior is stable.
5. Treat under-amortized runtime semantic findings as fanout questions, not automatic collapse targets.
6. Finish debug/source-binding polish only after runtime/planner tests are green.

## Verification Gates

For docs-only plan commits:

- `git diff --check`

For targeted registry/runtime changes:

- `.venv/bin/python -m pytest tests/unit/test_cellprofiler_module_execution.py tests/unit/test_cellprofiler_strategy_registries.py -q --tb=short --disable-warnings`
- `.venv/bin/python -m pytest tests/unit -q --tb=short --disable-warnings`
- `.venv/bin/python -m nominal_refactor_advisor <touched files> --json`

For planner/runtime behavior changes:

- Add focused parity tests around the changed seam.
- Re-run official30 parity only after unit parity is restored.

## Known Unsafe Approaches

- Simple constants for `__registry_key__` declarations.
- Moving trivial wrappers from one class to another without changing ownership.
- Collapsing metadata leaves into behaviorful bases when the class family is intentionally nominal.
- Replacing explicit policy families with loose dict dispatch.
- Treating object-label/image/table runtime semantics as interchangeable just because they share storage shape.

