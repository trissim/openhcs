# Full Repo Orchestration Hubs Refactor - 2026-05-18

## Advisor Evidence

Full-repo advisor scan flagged 16 oversized orchestration hubs. The highest-risk
OpenHCS runtime example is:

- `openhcs/core/orchestrator/orchestrator.py`
- `PipelineOrchestrator.execute_compiled_plate`
- Advisor summary: 556 lines, 43 branches, 142 calls, 85 callee families

Other representative hubs:

- `runtime/fiji_viewer_server.py`: `FijiViewerServer._add_dimension_change_listener`
- `microscopes/opera_phenix.py`: `OperaPhenixHandler._fill_missing_images`
- `pyqt_gui/widgets/shared/plate_view_widget.py`: `PlateViewWidget.eventFilter`

This plan focuses first on `PipelineOrchestrator.execute_compiled_plate`, because
it owns runtime execution semantics and interacts with recent debug/runtime work.

## Current Problem

`execute_compiled_plate` currently owns too many lifecycle responsibilities:

- resolving compiled pipeline definitions;
- validating execution inputs;
- deriving worker counts;
- creating visualizers from compiled contexts;
- constructing progress identity and event data;
- managing worker assignments;
- submitting work;
- handling cancellation;
- aggregating results;
- reconciling debug execution policy;
- shaping public return dictionaries.

The method is load-bearing and should not be split by moving arbitrary blocks
into private helpers. The split must create semantic subsystems with typed
requests and responses.

## Target Shape

Introduce a narrow execution facade that delegates to typed collaborators:

- `CompiledPlateExecutionRequest`
- `ExecutionInputValidator`
- `ExecutionWorkerPlan`
- `ExecutionVisualizerBootstrap`
- `ExecutionProgressEnvelope`
- `CompiledPlateWorkSubmission`
- `CompiledPlateResultAggregator`
- `ExecutionCancellationState`

The public `PipelineOrchestrator.execute_compiled_plate(...)` method should
remain as the compatibility entrypoint, but it should become request assembly +
facade invocation.

## Phase 1: Characterization Before Moving Code

Add focused tests around current externally visible behavior:

- missing `progress_queue` raises the current invariant error;
- missing `progress_context` raises the current invariant error;
- empty compiled contexts returns `{}`;
- `max_workers <= 0` normalizes to 1;
- resolved pipeline override still wins when `_resolved_pipeline_definition`
  exists;
- cancellation flag is respected by submission loops if currently testable.

Do not refactor before these tests exist.

## Phase 2: Request Record Extraction

Create a request record that carries the current parameter set plus resolved
defaults:

```python
@dataclass(frozen=True, slots=True)
class CompiledPlateExecutionRequest:
    pipeline_definition: tuple[AbstractStep, ...]
    compiled_contexts: Mapping[str, ProcessingContext]
    max_workers: int
    visualizer: NapariVisualizerType | None
    log_file_base: str | None
    progress_queue: object
    progress_context: Mapping[str, object]
    worker_assignments: Mapping[str, tuple[str, ...]] | None
    execution_bundle: CompiledExecutionBundle | None
    runtime_observation_mode: RuntimeObservationMode
    debug_execution_policy: DebugExecutionPolicy
```

Keep the compatibility method signature unchanged. Build the request internally.

## Phase 3: Validation and Defaults

Extract `ExecutionInputValidator`:

- initialized orchestrator check;
- non-empty pipeline check;
- compiled-context empty handling;
- progress queue/context invariants;
- max worker normalization;
- execution id / plate id extraction.

This should return a typed `ValidatedExecutionInputs` record, not mutate the
request.

## Phase 4: Visualizer Bootstrap

Extract visualizer creation from compiled contexts into:

- `ExecutionVisualizerRequirement`
- `ExecutionVisualizerBootstrap`
- `ExecutionVisualizerLaunchResult`

The bootstrapper should be injectable/testable and should not depend on the
entire orchestrator object. It can receive the file manager/config dependencies
it actually needs.

## Phase 5: Worker Submission and Result Aggregation

Extract work submission into a service with one responsibility:

- convert validated execution inputs into submitted worker jobs;
- expose cancellation state explicitly;
- emit progress through a typed envelope;
- return raw worker outcomes.

Extract result aggregation separately:

- map raw worker outcomes to the existing return dictionary shape;
- preserve current error messages/status labels;
- keep debug observation behavior unchanged.

## Phase 6: Compatibility Facade

After collaborators are tested, reduce `execute_compiled_plate` to:

1. resolve compiled pipeline;
2. build `CompiledPlateExecutionRequest`;
3. call `CompiledPlateExecutionFacade.execute(request)`;
4. return existing public result shape.

## Risks

- Multiprocessing behavior is hard to unit test; preserve existing process
  boundaries until typed seams are characterized.
- Progress events are part of GUI/debug behavior; do not change payload shape
  without focused tests.
- Visualizer creation may import optional GUI packages; keep imports lazy.
- Debug execution policy is recent and should remain explicit in records.

## Verification Gates

Focused:

```bash
.venv/bin/python -m pytest \
  tests/unit/test_orchestrator*.py \
  tests/unit/test_debug*.py \
  tests/unit/test_runner_cellprofiler_compatibility.py -q
```

Full:

```bash
.venv/bin/python -m pytest tests/unit -q
python -m nominal_refactor_advisor openhcs/core/orchestrator openhcs/runtime > /tmp/advisor_orchestration_after.txt
```

## Completion Criteria

- `execute_compiled_plate` no longer appears as a giant orchestration hub.
- Worker lifecycle, progress, visualizer bootstrap, and result aggregation have
  typed boundaries.
- Existing runtime tests pass.
- No new fake helpers that simply mirror method chunks by line range.
