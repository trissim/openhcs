# Architectural Debt Refactor Sequence

Date: 2026-05-15

## Purpose

This document captures the next sequence of deep refactors after the CellProfiler boilerplate-elimination, source-binding GUI, and debug-mode checkpoint. The current checkpoint should be treated as the stable restore point before starting these larger structural changes.

The debug/source-binding implementation files were scanned with `nominal_refactor_advisor` after cleanup and report no findings. The remaining debt is concentrated in older, larger GUI and workflow orchestration classes plus a few compatibility paths that should be retired only after benchmark/debug stability is preserved.

## Sequence 1: Split GUI Workflow Orchestration

Primary files:

- `openhcs/pyqt_gui/widgets/shared/services/batch_workflow_service.py`
- `openhcs/pyqt_gui/widgets/plate_manager.py`
- `openhcs/pyqt_gui/widgets/pipeline_editor.py`

Problem:

`BatchWorkflowService` is now a broad facade over compile, run, debug run, progress, snapshot readback, server status, and failure handling. Advisor output identifies a method-role quotient with many cohesive private method groups behind a smaller public facade. This is a real architectural signal: the class is carrying several subsystem algebras.

Target shape:

- `CompileWorkflowService`: compile submission transport, compile wait behavior, and stable pipeline fingerprints.
- `PlatePipelineRequestBuilder`: compile/run request construction from host plate state and fail-loud pipeline validation.
- `ExecutionSubmissionService`: normal run submission, execution-id bookkeeping, and completion polling.
- `DebugWorkflowService`: debug compile/run, paused-worker commands, artifact export, and warm debug compile artifact reuse.
- `DebugProgressNotificationService`: debug snapshot notification parsing and server readback from progress events.
- `ProgressWorkflowService`: progress registration, runtime projection invalidation, server status polling, and coalesced progress UI updates.
- `TerminalExecutionResultBuilder`: host-facing terminal result payload construction.
- `ExecutionControlService`: stop/force-kill orchestration, cancellation fanout, disconnect, terminal failure handling, and completion checks.
- `BatchWorkflowService`: thin public facade that owns sequencing and composes the services.

Migration strategy:

1. Extract pure dataclass request/result records first, without changing behavior. Done for compile/run/debug progress/terminal result records.
2. Move methods by role with tests kept at the facade boundary. Done for compile transport, plate request building, normal execution submission/completion polling, debug compile/run orchestration, paused-worker commands, artifact export, progress/server-status polling, debug snapshot fanout, terminal result mapping, and stop/disconnect/failure handling.
3. Keep `BatchWorkflowService` public methods stable until GUI callers are migrated. Current pass preserves `compile_plates`, `run_plates`, debug run/control/export methods, and debug snapshot listener registration.
4. Add focused tests for each service before deleting facade internals.
5. Remaining Sequence 1 work is optional facade slimming only; the touched service set is currently advisor-clean and independently covered by focused tests plus full `tests/unit`.

## Sequence 2: Normalize Pipeline/Plate GUI Command Families

Primary files:

- `openhcs/pyqt_gui/widgets/pipeline_editor.py`
- `openhcs/pyqt_gui/widgets/plate_manager.py`

Problem:

The broad advisor scan flags string-key dispatch tables, repeated attribute probing, and dangling private methods in the large GUI widgets. These are mostly pre-existing structural issues, but the debug work now exercises those classes more heavily.

Target shape:

- Replace config-attribute string dispatch with a typed config-detail projection table or registered strategy family.
- Replace widget action dispatch and inherited item hook mappings with nominal declarations plus compatibility projection.
- Replace structural `hasattr`/`getattr` GUI role probes with explicit host protocols or nominal service interfaces.
- Delete truly unreferenced private methods after verifying no Qt/framework hook depends on them.
- Convert one-step setter wrappers into properties or direct state updates.

Migration strategy:

1. Add tests around currently observed GUI behavior before deleting private methods. Done for action dispatch and workflow/progress seams.
2. Introduce nominal protocols for host/editor services used by tests and workflow services. Partially done through typed action routes, typed item hooks, direct projection state, and extracted workflow services.
3. Replace string-key dispatch one closed family at a time. Done for preview config details, widget actions, and widget item hooks.
4. Only then remove dead private methods. Done for verified dead partial-refresh helpers; remaining private-method advisor entries are inherited `AbstractManagerWidget` template hooks or broad class-decomposition candidates.
5. Remaining Sequence 2 work is broad GUI class decomposition and structural `hasattr`/`getattr` cleanup. Keep that separate from runtime/planner parity work.

## Sequence 3: Retire CellProfiler Runtime Compatibility Paths

Primary files:

- `openhcs/interop/cellprofiler/pipeline_generator.py`
- `openhcs/interop/cellprofiler/runtime/generated_pipeline.py`
- `openhcs/interop/cellprofiler/runtime/module_execution.py`
- `openhcs/interop/cellprofiler/runtime_pipeline.py`
- `openhcs/processing/backends/cellprofiler/__init__.py`

Problem:

Generated pipelines now emit normal backend callable declarations plus source/settings bindings, while product runtime owns artifact contracts, wrappers, adapter preparation, and sidecar materialization. The next risk is stale compatibility paths that preserve old generated-boilerplate assumptions.

Target shape:

- Generated code is declarative: imports backend callables, declares `FunctionStep`s, declares source schema/settings sidecars.
- Runtime code is authoritative for CellProfiler module contracts and wrapper binding.
- Import/materialization sidecars are versioned and validated before registry hydration.
- No product-owned runtime object literals are emitted into generated pipeline modules.

Migration strategy:

1. Add a compatibility-path inventory with tests proving which paths are still exercised. Done for the generated runtime import/registration shell.
2. Delete one legacy path at a time after parity benchmarks and generated-pipeline tests pass. Done for the trivial `.cppipe` generation facade and generated function-registration facade; benchmark conversion now calls `CPPipePipelineGenerationRequest` directly.
3. Keep sidecar schema versioning fail-loud. Done, with contract/spec/materialization sidecar codecs split under the versioned sidecar facade.
4. Re-run official30 parity after each deletion batch. Still required before deleting deeper CP runtime compatibility paths.
5. Remaining Sequence 3 work is the broad `PipelineGenerator` stage split and any deeper CP runtime compatibility deletion. Do not mix that with path-planner decomposition.

## Sequence 4: Tighten Artifact/Invocation Planning Boundaries

Primary files:

- `openhcs/core/invocation_artifacts.py`
- `openhcs/core/pipeline/artifact_planning.py`
- `openhcs/core/pipeline/path_planner.py`
- `openhcs/core/module_artifact_contract.py`
- `openhcs/core/function_patterns.py`

Problem:

The new invocation-aware artifact declaration seam is the correct abstraction, but it is still adjacent to older callable-contract and path-planning APIs. Future refactors should make the invocation declaration provider the only artifact declaration authority visible to path planning.

Target shape:

- `ArtifactDeclarationStepContext` is constructed once from step snapshots.
- `InvocationArtifactDeclarations` is the only provider return type.
- Callable-contract artifact projection remains only the default provider implementation.
- CellProfiler module contract projection remains a provider, not a generated-code concern.
- Group/axis artifact key selection is owned by typed artifact-key selection models.

Migration strategy:

1. Add provider-level tests for native functions, CP module contracts, grouped function patterns, and disabled invocations. Existing `tests/unit/test_function_patterns.py` covers these seams and was rerun after each cleanup.
2. Remove direct callable-contract artifact extraction from path-planner internals. Current path planning consumes `extract_artifact_declarations(...)` through the invocation declaration provider; callable-contract projection remains the default provider.
3. Keep backwards-compatible decorators as metadata producers, not planner logic. Preserved.
4. Done in this pass: artifact spec merging moved into `ArtifactSpecAccumulator`, group-key de-duplication moved onto `ArtifactGraph`, compiled invocations now extend normalized invocation items, function-pattern group lookup/kwarg merge are nominalized, and `PathPlanner` construction uses `PathPlanningContext`.
5. Remaining Sequence 4 work is the broad `PathPlanner` staged subsystem split. The touched artifact/function-pattern/path-planner scan now reports only that broad class decomposition.

## Sequence 5: Complete Debug UX Integration

Primary files:

- `openhcs/core/debug.py`
- `openhcs/core/debug_views.py`
- `openhcs/pyqt_gui/windows/debug_inspector_window.py`
- `openhcs/pyqt_gui/widgets/pipeline_editor.py`
- `openhcs/runtime/zmq_execution_client.py`
- `openhcs/runtime/zmq_execution_server.py`

Problem:

The debug substrate is now structurally clean and tested at unit/control-channel boundaries. The remaining work is true GUI workflow integration, not core model cleanup.

Target shape:

- Live PyQt + ZMQ paused-worker end-to-end tests for pause, step, continue, stop, inspect, export.
- Viewer open requests wired through existing streaming services for local readable artifacts.
- Export destination UX reports resulting paths and failures clearly.
- Richer CP module renderers add previews only through generic snapshot refs and table models.

Migration strategy:

1. Build a minimal live debug workflow harness around a real ZMQ server.
2. Test command ordering and worker lifetime across multiple commands.
3. Add viewer/export behavior behind typed host services.
4. Extend CP renderers by module family, not one-off module conditionals.

## Verification Gates

Every sequence should preserve:

- `python -m py_compile` on touched modules.
- Focused tests for the moved subsystem.
- Full `tests/unit`.
- `nominal_refactor_advisor` on touched implementation files.
- CellProfiler official30 parity/speed benchmark only after runtime/planner changes.

## Current Checkpoint Notes

The current checkpoint includes:

- Invocation-aware artifact declarations and module artifact contracts.
- Generated CellProfiler boilerplate elimination through runtime-owned wrappers and sidecars, with generated import/registration compatibility facades reduced to nominal runtime-module authorities.
- Product-owned direct generated-pipeline execution, with benchmark timing kept as a thin facade.
- Artifact planning merge/group-key behavior owned by nominal artifact planning authorities.
- Source-binding inline editor and preview model.
- Debug snapshot store, ZMQ read/export controls, paused-worker command path, warm artifact replay validation.
- Debug/source-binding implementation files cleaned to advisor zero-findings for the targeted scan.
- Current remaining advisor findings are broad staged-orchestration splits: `PipelineGenerator`, `PathPlanner`, and older large GUI widgets.

Do not start broad GUI class decomposition in the same commit as benchmark/runtime parity changes. Keep each sequence independently revertible.
