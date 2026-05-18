# Orchestrator Stage Split Continuation - 2026-05-18

## Full-Scan Evidence

Source scan:

```bash
timeout 180 .venv/bin/python -m nominal_refactor_advisor openhcs
```

Relevant findings:

- `PipelineOrchestrator.execute_compiled_plate` remains an oversized
  orchestration hub after the first staged split.
- `openhcs/core/orchestrator/orchestrator.py` still has repeated
  attribute-probe sites.
- The side-effecting `pipeline_config` property alias is recorded as known
  noise until a typed config-sync descriptor exists.

## Current State

Already extracted:

- request validation/defaults;
- visualizer bootstrap;
- worker-lane identity/plan/executor;
- progress step-name authority;
- state/cache alias descriptors.

Remaining method responsibilities:

- worker-assignment planning;
- lane context projection;
- executor construction;
- process/thread/fork execution selection;
- future submission and result collection;
- runtime observation merge;
- executor shutdown and broken-pool handling;
- GPU cleanup;
- analysis consolidation;
- visualizer cleanup;
- final orchestrator state projection.

## Target Shape

Add semantic collaborators, not line-range helpers:

- `CompiledContextLanePlanner`
- `WorkerAssignmentPlan`
- `ExecutorFactory`
- `WorkerSubmissionPlan`
- `WorkerResultCollector`
- `ExecutionCleanupPlan`
- `AnalysisConsolidationPlan`
- `ExecutionStateProjector`

`execute_compiled_plate` should become request assembly plus stage sequencing.

## Phases

1. Characterize worker-assignment behavior:
   - default assignment distribution;
   - duplicate assignment failure;
   - missing assignment failure;
   - `__combo_` context grouping.
2. Extract lane planning:
   - `contexts_by_axis`;
   - `worker_assignments`;
   - `axis_to_worker`;
   - `lane_axis_contexts`.
3. Extract executor construction:
   - inline/thread/process/fork mode decision;
   - worker initializer args;
   - log-file base handling.
4. Extract result collection:
   - future mapping;
   - runtime observation merge;
   - progress error emission;
   - fail-fast behavior.
5. Extract cleanup/finalization:
   - executor shutdown;
   - fork inherited state cleanup;
   - GPU cleanup;
   - analysis consolidation;
   - visualizer cleanup;
   - orchestrator state update.

## Verification Gates

Focused:

```bash
.venv/bin/python -m pytest \
  tests/unit/test_orchestrator*.py \
  tests/unit/test_debug*.py \
  tests/unit/test_runner_cellprofiler_compatibility.py -q
timeout 120 .venv/bin/python -m nominal_refactor_advisor openhcs/core/orchestrator/orchestrator.py
```

Full:

```bash
git diff --check
.venv/bin/python -m pytest tests/unit -q
timeout 180 .venv/bin/python -m nominal_refactor_advisor openhcs
```

## Completion Criteria

- `execute_compiled_plate` no longer appears as an oversized hub.
- Worker planning, submission, result collection, and cleanup are typed
  boundaries.
- No behavior-changing progress payload or runtime observation regressions.

