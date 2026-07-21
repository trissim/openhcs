# ExampleYeastColonies Source-Context Identity Fix

## Status

Complete. The generic compiled input edge now carries exclusive current-flow
authority, source and produced artifacts retain their exact owners, focused gates
pass, and the canonical YeastColonies pipeline executes without a provenance
failure. Cached-reference comparison still reports six explicitly out-of-scope
residuals: four missing Align shift measurement features and two object
orientation-value differences owned by output-recording and shape-kernel parity.

## Ownership

- Own the `ExampleYeastColonies` source-context identity failure only.
- May edit the nominal artifact relation/projector owner and focused regression
  tests required to fix that class of provenance error.
- Do not edit `pipeline_import.py` or FilterObjects declarations.
- Do not alter metadata validation, add pipeline/module-name branches, infer by
  string matching, or introduce fallback source-resolution paths.

## Owner Inventory

- `openhcs/core/artifacts.py`: `ArtifactSpecRelation` and compiled
  `ArtifactOutputPlan` are the authoritative source-context relation owners.
- `openhcs/core/steps/function_runtime.py`: generic output contextualization and
  exact declared-source payload projection.
- `openhcs/core/function_patterns.py`: `InvocationArtifactInputEdgePlan` is the
  compiled per-occurrence authority for runtime artifact input selection.
- `openhcs/core/pipeline/path_planner.py`: compiles exact input edges from the
  current `ArtifactDeclarationStepContext`, including its named main flow.
- `openhcs/interop/cellprofiler/runtime/artifact_binding.py`: consumes compiled
  input edges when binding module inputs and output source contexts.
- `openhcs/interop/cellprofiler/module_callable_abi.py`: CellProfiler module
  output/source payload selection hooks.
- `openhcs/interop/cellprofiler/runtime/output_recording.py`: records the exact
  matched output plan and delegates image/object context to nominal owners.
- Module declarations and callable artifact declarations: exact producer and
  consumer identities for the YeastColonies graph.

## Active Peers

- `.agents/artifact-contract-collapse.md`: owns broader declaration/import and
  runtime migration; this task changes only the exact compiled input-edge owner
  and its runtime consumer.
- `.agents/unify-object-label-measurement-runtime.md`: owns generic output
  normalization; this task does not change output carriers or contextualization.
- `.agents/cp-output-plan-availability.md`: completed exact output selection;
  this task does not reopen output matching or recording.
- `.agents/special-input-artifact-normalization.md`: owns artifact-fed parameter
  normalization and explicitly excludes runtime payload projection.
- `.agents/next-batch-parity-and-performance.md`: owns suite-wide acceptance;
  this task runs only the assigned canonical YeastColonies case under the shared
  runtime lock.

## Root Cause

`Align` is the producing module. Its compiled output order and exact relations
are correct:

1. `AlignedPlate` has `SourceStackLineageSourceRelation(PlateTemplate)`.
2. The second image output is related to `CorrRed`.
3. The third image output is related to `CombinedImage`.

The step's main-input dependency is the preceding `ImageMath` output,
`CombinedImage`. `PlateTemplate` is separately source-bound. The compiled
invocation edge records no storage plan for `PlateTemplate`, but does not record
that `PlateTemplate` is *not* the invocation's main-flow input. At runtime,
`RuntimeInputBindingRequest.artifact_request()` therefore treats the inherited
workspace `PRIMARY_PLANE` role as main-flow ownership and projects
`PlateTemplate` from the `CombinedImage` payload. Strict metadata validation
correctly rejects that projection because the payload represents
`CombinedImage`, `OrigColor`, `OrigBlue`, and `CorrBlue`.

The workspace projection role and invocation input authority are distinct.
`ArtifactDeclarationStepContext.main_flow_artifacts` already owns the latter at
compile time; `InvocationArtifactInputEdgePlan` must persist it so runtime code
does not reconstruct it from source-binding roles.

## Executable Plan

- [x] Reproduce the current canonical ZMQ failure for `ExampleYeastColonies`.
- [x] Dump the compiled step/artifact graph around `PlateTemplate`,
  `CombinedImage`, `OrigColor`, `OrigBlue`, and `CorrBlue`.
- [x] Trace the failing output through `ArtifactOutputPlan.relations`, source
  payload selection, output contextualization, and output recording.
- [x] Fix the exact relation owner or generic projector owner.
- [x] Add a focused regression that fails on the wrong source identity while
  retaining strict metadata validation.
- [x] Run focused unit suites and the one canonical ZMQ case.
- [x] Review the scoped production diff for mirrored or fallback semantics.

## Progress

- 2026-07-19: Started owner inventory. Existing tests cover producer-anchored
  `OrigColor`/`PlateTemplate` source projection, but not the failing output's
  declared artifact relation.
- 2026-07-19: Canonical compile-then-execute over ZMQ reproduced the strict
  `PlateTemplate` metadata failure. Compiled graph inspection confirmed the
  `Align` output relation is correct and isolated the fault to runtime
  reconstruction of invocation input ownership.
- 2026-07-19 06:27 EDT: Read `AGENTS.md` and every `.agents/*.md` peer plan,
  audited the dirty worktree and current diffs, and completed the required
  registry/MRO search. No new nominal family is justified. The exact edit scope
  is `InvocationArtifactInputEdgePlan`, its `PathPlannerArtifactStage` compiler,
  `RuntimeInputBindingRequest.artifact_request()`, and one disjoint focused test
  file. Next step is implementation.
- 2026-07-19 06:32 EDT: Added `consumes_main_flow` to the compiled input-edge
  owner, populated it from `ArtifactDeclarationStepContext.main_flow_artifacts`
  for storage-backed and storage-free occurrences, and made runtime source
  selection consume that fact directly. Added focused compiler/runtime coverage
  proving a `PRIMARY_PLANE` workspace role cannot override compiled non-main-flow
  ownership and that strict metadata rejects the wrong current payload. Next
  step is focused testing.
- 2026-07-19 06:35 EDT: Focused compiler/runtime provenance regression passes
  (`2 passed`). The next phase is broader existing edge, planner, binding, and
  source-projection verification.
- 2026-07-19 06:37 EDT: The first broader six-file batch stopped during
  collection because active peer work references an undefined
  `_artifact_input_parameter_names_from_projection` in `callable_contract.py`;
  this prevents CellProfiler backend imports and is outside this plan's write
  boundary. No owned test failed. Next step is the four generic suites that do
  not import the blocked backend package.
- 2026-07-19 06:39 EDT: Four generic suites ran after another concurrent
  callable-contract edit and reported `84 failed, 134 passed`. Sampled failures
  all stop before input-edge compilation because `function_patterns.py` passes
  `artifact_input_parameter_names` to a peer-owned `validate_public_kwargs()`
  signature that no longer accepts it. This is outside the owned diff; next step
  is to monitor the active normalization peer and rerun after that boundary is
  internally consistent.
- 2026-07-19 06:43 EDT: Scoped `py_compile`, Ruff, forbidden-name search, and
  `git diff --check` pass. Extended repeated-occurrence authority comparison to
  include `consumes_main_flow` and tightened the focused regression so an
  intentionally wrong compiled bit fails through `artifact_request()` with the
  unchanged strict metadata error. Next step is rerunning the focused test and
  monitoring the peer-owned normalization boundary.
- 2026-07-19 06:45 EDT: The tightened focused regression passes again
  (`2 passed`). The active normalization peer has completed its implementation
  batch and removed the inconsistent compiler call, though its own verification
  remains in progress. Next step is rerunning the four generic suites.
- 2026-07-19 06:47 EDT: Four generic suites improved to `207 passed, 11
  failed`. Ten failures are the active normalization peer's new artifact-fed
  parameter annotation checks; one is the completed output-plan worker's runtime
  projection assertion. No failure reaches or references the new main-flow edge
  authority. Next step is isolated green edge/source suites and canonical runner
  preparation while peers settle.
- 2026-07-19 06:49 EDT: Exact input-edge cardinality and source-projection suites
  pass together (`79 passed`). Next step is focused CellProfiler runtime-binding
  verification followed by the locked canonical YeastColonies case.
- 2026-07-19 06:51 EDT: Focused CellProfiler runtime-input binding and identity
  suites pass (`11 passed`). All owned and directly adjacent gates are green;
  next step is canonical CLI/lock inspection and the single locked ZMQ case.
- 2026-07-19: The first locked canonical case reached the unchanged strict
  `PlateTemplate` metadata failure. The client connected to orphaned ZMQ server
  processes that started before this plan's production edits, so this is not
  valid acceptance evidence for the current worktree. Next step is to confirm
  process/source timestamps and rerun the same locked case against a fresh
  current-worktree server before revising the nominal-owner implementation.
- 2026-07-19: Confirmed both orphaned servers predated the edited compiler and
  runtime files, stopped them after verifying no benchmark/profiler work or lock
  holder, and reran the one canonical manifest case under the shared lock. A
  fresh current-worktree server compiles successfully and clears the original
  `PlateTemplate` failure, then fails one step earlier when
  `CorrectIlluminationCalculate` resolves declared `OrigRed` from an
  `OrigColor` current payload. This is the same generic compiled-input authority
  class, not acceptance. Next step is to inspect the exact `OrigRed` edge and
  the lifecycle payload selected for its step before the next edit batch.
- 2026-07-19: Direct current-worktree compilation proved the incomplete class:
  `OrigRed`, `CorrGreen`, and `CombinedImage` each have exact storage-backed
  input edges but were also stamped `consumes_main_flow=True`; only
  storage-free `OrigColor` and `PlateTemplate` require the source-versus-current
  payload distinction. A storage-backed edge already owns the exact produced
  payload and must never be replaced by whichever lifecycle image is current.
  Next edit will make `consumes_main_flow` an exclusive storage-free authority,
  enforce that invariant on the nominal edge, and add a produced-artifact
  regression using a deliberately wrong current payload.
- 2026-07-19: Implemented exclusive storage-free main-flow ownership and the
  nominal edge invariant. The first focused run failed before compilation
  because the new test helper locally shadowed the imported `artifact_inputs`
  decorator; this is test-only construction, not a production failure. Renamed
  the local plan mapping; next step is rerunning the focused regression.
- 2026-07-19: Focused source-context coverage now passes `3/3`, including the
  storage-backed produced-artifact case with unrelated current-image metadata.
  Next step is the adjacent edge and CellProfiler binding suites, followed by a
  fresh locked canonical case after confirming the host and server state.
- 2026-07-19: The combined owned and adjacent edge/source/runtime-binding batch
  passes `93/93`. No synthetic compiled edge violates the new exclusivity
  invariant. Next step is scoped static/diff review and canonical host checks.
- 2026-07-19: Scoped Ruff, `py_compile`, forbidden concrete-name/branch search,
  and `git diff --check` pass. The production path contains no YeastColonies,
  artifact-name, module-name, or callable-name branch. Next step is the fresh
  locked canonical case after rereading plan steering and confirming no host
  contention or stale server.
- 2026-07-19: Fresh locked canonical ZMQ compile and execution complete all
  19 submitted steps with no source-context metadata exception; execution time
  is `2.764349s`. Cached-reference comparison remains non-equivalent for four
  missing Align shift measurement features and two object orientation values.
  Those are output-recording and shape-kernel parity classes outside this
  plan's input identity/provenance owner, and no validation or tolerance was
  weakened. Next step is final scoped ownership/diff review.
- 2026-07-19: Final scoped review confirms one generic authority path with no
  mirrored registry, fallback chain, workspace-role reconstruction, concrete
  artifact/pipeline/module branch, or weakened validation. Produced inputs use
  exact compiled storage edges; storage-free current-flow inputs use the
  compiled edge bit; remaining storage-free inputs require an exact source
  binding. Plan complete; the six comparison-only differences remain assigned
  to their separate output-recording and orientation owners.

## Parent Notes

- Preserve strict one-authority runtime validation and declared-source metadata
  validation.
- Do not infer invocation ownership from workspace projection roles, artifact
  names, module names, storage availability, or callable identity.
- Do not edit importer/source lowering, output matching/recording, module
  declarations, or backend leaves.
- Canonical ZMQ acceptance must use the shared runtime lock and wait until the
  host is free of overlapping benchmark/profiler work.
- Do not classify the first post-edit canonical failure as an implementation
  failure until the pre-edit orphaned ZMQ server has been excluded.

## Tests

- `.venv/bin/pytest -q tests/unit/test_invocation_input_source_context_identity.py`
  - `2 passed in 2.68s`
  - tightened runtime-path rerun: `2 passed in 1.82s`
  - full-class produced-artifact regression: `3 passed in 1.15s`
- Six-file broader batch including CellProfiler binding suites
  - collection blocked by peer-owned `callable_contract.py` `NameError` before
    tests executed
- Four generic edge/planner/source suites
  - `134 passed, 84 failed`; sampled failures share the peer-owned
    `CallableContract.validate_public_kwargs()` signature mismatch before the
    edited input-edge compiler executes
  - rerun after peer implementation: `207 passed, 11 failed`; ten peer-owned
    parameter-normalization failures and one pre-existing output-plan projection
    failure, with no owned failure
- `.venv/bin/pytest -q tests/unit/test_artifact_input_edge_cardinality.py tests/unit/test_function_runtime_source_projection.py`
  - `79 passed in 2.60s`
- `.venv/bin/pytest -q tests/unit/test_cellprofiler_runtime_input_binding_contract.py tests/unit/test_cellprofiler_runtime_binding_identity.py`
  - `11 passed in 3.00s`
- Combined owned and adjacent five-file batch
  - `93 passed in 3.10s`
- `.venv/bin/python -m py_compile ...` over all owned production/test files
  - passed
- `.venv/bin/ruff check ...` over all owned production/test files
  - all checks passed
- Scoped forbidden-name search and `git diff --check`
  - passed
  - rerun after full-class fix passed with no concrete artifact or pipeline names
- First locked canonical rerun
  - reached the unchanged strict `PlateTemplate` metadata error through ZMQ
  - invalid as current-worktree acceptance evidence because the connected
    orphaned server processes predated the production edits
- Fresh-server locked canonical rerun
  - compile succeeded and the original `PlateTemplate` failure did not recur
  - execution reached `CorrectIlluminationCalculate` and strict validation
    rejected an `OrigColor` payload for declared `OrigRed`
  - result is a valid current-worktree failure and requires completing the same
    generic source-context ownership class
- Final locked canonical ZMQ case
  - compile and execution completed through all 19 submitted steps
  - no `Image metadata does not represent declared source image` failure
  - `EXECUTE_OPENHCS=2.7643494606018066s`
  - cached-reference comparison reports six outside-scope parity differences:
    four missing Align shift features and two orientation-value differences

## Changed Files

- `docs/plans/yeast_colonies_source_context_identity_fix_20260719.md`
- `openhcs/core/function_patterns.py`
- `openhcs/core/pipeline/path_planner.py`
- `openhcs/interop/cellprofiler/runtime/artifact_binding.py`
- `tests/unit/test_invocation_input_source_context_identity.py`
