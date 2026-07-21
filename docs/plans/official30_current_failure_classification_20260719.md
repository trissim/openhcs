# Official-30 Current Failure Classification

## Status

Complete. The existing read-only classification process stopped after writing
all 30 observations; no production fixes are owned here.

Start: `2026-07-19T06:08:54-04:00`

Tree fingerprint:

- `HEAD`: `f5e7c73d2d37ab6ecb6c973bcdd44fd754b1be49`
- porcelain path/status SHA-256: `99b2515445322bca4d30cd371eadb2b916b7f414c9b335a6ab2fcb4d0cb86a90`
- tracked binary diff SHA-256: `0f9a8ff641ead2cd747cd53c3da3c57d41182eac5b61fda72fd5f0e36599132d`

Canonical run fingerprint (`2026-07-19T06:11:20-04:00`):

- `HEAD`: `f5e7c73d2d37ab6ecb6c973bcdd44fd754b1be49`
- porcelain path/status SHA-256: `70f0e18cba1ce9352f717078242c2ec661419612d344869c6a0bad2208ea6692`
- tracked binary diff SHA-256: `e4b567dc1772bf9f9b30d2f2f5c0a9d4d3af6eec14699dad6f00691d9004cad4`
- isolated pytest root:
  `/home/ts/.cache/openhcs/official30-classification-20260719-0615/pytest-tmp`
- Timing validity: **diagnostic only**. A peer worker's
  NominalRefactorAdvisor process consumed one full CPU during this run. Failure
  classifications and parity observations remain valid; no timing from this run
  is an acceptance result.

Observation completion: `2026-07-19T06:23:45-04:00`.

Post-run inspection fingerprint (`2026-07-19T06:29:03-04:00`):

- `HEAD`: `f5e7c73d2d37ab6ecb6c973bcdd44fd754b1be49`
- porcelain path/status SHA-256: `0409656c433661e5141c9ccb5f3c26f71875913f9e3647001c79ae784ef3820e`
- tracked binary diff SHA-256: `f6c0d0ea034695063961f8720c2aa971219228102d765624cf8ea71866fa31bd`
- Stable-tree validity: **invalid**. Source/UI production files changed during
  the run, and `function_patterns.py`, `path_planner.py`, and runtime
  `artifact_binding.py` changed after the final observation. The exact recorded
  failures remain classification evidence, but this is not a current-tree
  acceptance baseline. F1, F2, F3, F6, and F7 cross those changed owners and
  require targeted confirmation before being claimed as current failures.

## Ownership

- Own broad official-30 failure-surface classification only.
- Do not edit production code.
- Do not generate native CellProfiler references.
- Do not create another benchmark harness.
- Retain classifications from an invalidated run; rerun only affected failures
  after recording a new tree fingerprint.

## Authority Inventory

- Canonical integration harness:
  `tests/integration/test_cellprofiler_official30_zmq.py`
- Case manifest:
  `benchmark/manifests/official30_portable_axis1.json`
- Cached native references:
  `benchmark/native_refs/official30_scoped_rows`
- Canonical path: generated public `PipelineConfig + FunctionStep` source,
  compile, then execute over ZMQ using the compiled artifact.
- Baseline config: one well, Napari disabled and non-persistent.
- Strict numeric comparison tolerance: absolute and relative `1e-6`.

## Active Peers

- `docs/plans/czi_source_bindings_and_zarr_audit_20260719.md`: source-store
  aggregation and source-binding ownership; production changes may invalidate
  source/import classifications.
- `docs/plans/global_ui_zmq_config_tabs_plan_20260719.md`: nominal UI and ZMQ
  config ownership; production changes may invalidate harness import or server
  lifecycle behavior.

## Executable Plan

- [x] Record start time and tree fingerprint.
- [x] Confirm exactly 30 manifest cases and cached-reference-only harness.
- [x] Establish an import-coherent baseline test collector.
- [x] Run baseline official-30 through the canonical ZMQ harness in an isolated
  output root, serializing canonical execution with
  `/home/ts/.cache/openhcs/official30-runtime.lock`.
- [x] Update each case incrementally with import, compile, execute, strict parity,
  and OpenHCS execution timing.
- [x] Group failures by exact shared root cause and owning production files.
- [x] Record the ending tree fingerprint and identify invalidated observations.

## Case Ledger

| Case | Import | Compile | Execute | Strict parity | OpenHCS execute (s) | Root cause |
|---|---|---|---|---|---:|---|
| ExampleColocalization | pass | pass | fail | not run |  | F1 output-plan availability |
| ExampleCometAssay | pass | pass | pass | pass | 4.571 diagnostic |  |
| ExampleFly | pass | pass | fail | not run |  | F2 duplicate compiled input occurrence |
| ExampleFlyURL | pass | pass | pass | pass | 1.682 diagnostic |  |
| ExampleHuman | pass | fail | not run | not run |  | F3 typed same-name artifact collision |
| ExampleIlluminationCorrection_Example1_AllMethod | pass | pass | pass | pass | 6.815 diagnostic |  |
| ExampleIlluminationCorrection_Example1_EachMethod | pass | pass | pass | pass | 0.215 diagnostic |  |
| ExampleIlluminationCorrection_Example2 | pass | pass | pass | pass | 1.839 diagnostic |  |
| ExampleIlluminationCorrection_Example3 | pass | pass | pass | pass | 0.865 diagnostic |  |
| ExampleImagingFlowCytometryObjectsInGrid | pass | pass | fail | not run |  | F4 dict-group source identity routing |
| ExampleNeighbors | pass | pass | fail | not run |  | F2 input occurrence identity collapse |
| ExamplePercentPositive | pass | pass | fail | not run |  | F4 dict-group source identity routing |
| ExampleSpeckles | pass | pass | pass | pass | 1.432 diagnostic |  |
| ExampleTrackObjects | pass | pass | pass | pass | 19.139 diagnostic |  |
| ExampleTumor | pass | pass | pass | pass | 11.209 diagnostic |  |
| ExampleUntangleAndStraightenWorms | pass | not reached | not run | not run |  | F5 ZMQ cold-start readiness race |
| ExampleUntangleWorms | pass | pass | pass | pass | 8.974 diagnostic |  |
| ExampleUntangleWormsBrightField | pass | pass | pass | pass | 10.253 diagnostic |  |
| ExampleVitra | pass | fail | not run | not run |  | F6 scalar artifact occurrence partition loss |
| ExampleWoundHealing | pass | pass | pass | pass | 7.170 diagnostic |  |
| ExampleYeastColonies | pass | pass | fail | not run |  | F7 invocation main-flow ownership loss |
| ExampleYeastPatches | pass | pass | pass | pass | 5.130 diagnostic |  |
| cp_tutorial_3d_monolayer | pass | pass | fail | not run |  | F8 unprojected aligned image input |
| cp_tutorial_advanced_segmentation_final | pass | fail | not run | not run |  | F6 scalar artifact occurrence partition loss |
| cp_tutorial_quality_control | pass | pass | pass | pass | 3.417 diagnostic |  |
| cp_tutorial_beginner_segmentation_final | pass | fail | not run | not run |  | F6 scalar artifact occurrence partition loss |
| cp_tutorial_pixel_based_classification | pass | pass | pass | fail | 1.099 diagnostic | F9 externally observed output identity omission |
| cp_tutorial_translocation_final | pass | pass | fail | not run |  | F10 relationship endpoint domain projection |
| cp4_supplement_combine_objects | pass | pass | pass | pass | 0.057 diagnostic |  |
| cp_tutorial_translocation_start | pass | pass | pass | pass | 0.147 diagnostic |  |

## Failure Classes

### F1 - Compiled output plan unavailable at runtime

- Cases: `ExampleColocalization` (first observed).
- Exact failure: compiled `correct_illumination_calculate` invocation output
  `CorrectIlluminationCalculate_1_image_2` is unavailable in the step.
- Stage: import pass; compile pass; execute fail before parity/timing.
- Owning boundary: `CompiledFunctionInvocation.select_outputs()` in
  `openhcs/core/function_patterns.py`, plus the compiler/runtime owner that
  determines which compiled `ArtifactOutputPlan` values the step carries.
- Architectural question for the fixing owner: determine why an invocation-owned
  output survived compilation while its exact step runtime plan was pruned; do not
  weaken exact identity validation or add a module-name exception.

### F2 - Duplicate compiled artifact input occurrence

- Cases: `ExampleFly`, `ExampleNeighbors`.
- Exact failure: runtime requested `object_labels:Nuclei` and found two compiled
  artifact-input occurrences where the consumer requires one.
- Stage: import pass; compile pass; execute fail before parity/timing.
- Owning boundary: invocation artifact-input edge construction in the compiler and
  exact occurrence consumption in `CellProfilerRuntimeAdapter._artifact_input()` at
  `openhcs/interop/cellprofiler/runtime/adapter.py`.
- Architectural question for the fixing owner: identify why two semantic input
  roles lower to the same undifferentiated artifact name/type occurrence. Preserve
  invocation-position identity; do not deduplicate by name or select the first.
- Second exact manifestation (`ExampleNeighbors`): the same `Cells` object artifact
  intentionally fills multiple callable roles; runtime-context selection calls
  `ArtifactSpecCollection.unique()` and rejects the role-specific declarations as
  conflicting. Output/source relations must select the compiled occurrence/role,
  not collapse roles to an artifact ref and then guess.

### F3 - Typed same-name artifact collision

- Cases: `ExampleHuman` (first observed).
- Exact failure: source image `PH3` and produced object labels `PH3` share a legal
  external name; compilation reports object-label production consumed as image.
- Stage: public pipeline import pass; compile fail; execution/parity not run.
- Owning boundary: artifact graph storage and lookup in
  `openhcs/core/pipeline/path_planner.py`, where declarations are still selected by
  plain name despite the existing typed `ArtifactSpecRef` identity.
- Architectural question for the fixing owner: migrate graph identity to exact
  `(artifact type, name)` refs throughout the owning generic graph. Do not rename
  the public artifacts or add a CellProfiler-specific collision rule.

### F4 - Dict-group source identity routing drift

- Cases: `ExampleImagingFlowCytometryObjectsInGrid`, `ExamplePercentPositive`.
- Exact failure: the dict-pattern invocation for source group/channel `2` requests
  declared image `DF_image`, but its runtime payload carries only `BF_image`
  provenance from source group/channel `1`.
- Stage: import pass; compile pass; execute fail before parity/timing.
- Owning boundary: generic function-pattern group routing and compiled source-plan
  projection in `openhcs/core/steps/function_runtime.py`, with exact provenance
  validation in `openhcs/core/runtime_image_values.py`.
- Architectural question for the fixing owner: retain the invocation's exact
  source group/key through compiled loading. Do not relabel BF pixels as DF, infer
  aliases from position, or weaken provenance validation.

### F5 - ZMQ cold-start readiness race

- Cases: `ExampleUntangleAndStraightenWorms` (infrastructure-only observation).
- Exact failure: the client exhausted its connection attempt before compilation
  submission. The spawned server logged ready at `06:19:21.545`, approximately
  `0.24s` after the failed observation at `06:19:21.305`; the next case used a
  live server and continued the suite.
- Stage: public import pass; compile not reached; execution/parity not run.
- Owning boundary: benchmark server lifecycle/readiness in
  `benchmark/adapters/openhcs.py` and ZMQ launch/client readiness signaling, not
  the `ExampleUntangleAndStraightenWorms` pipeline.
- Architectural question for the fixing owner: synchronize submission with the
  server's authoritative ready state across cold registry startup. Do not add a
  pipeline retry or classify this as a worm-module defect.

### F6 - Scalar artifact occurrence partition lost during reconstruction

- Cases: `ExampleVitra`, `cp_tutorial_advanced_segmentation_final`, and
  `cp_tutorial_beginner_segmentation_final`.
- Exact failure: compilation of a scalar callable parameter named `labels`
  receives respectively three, five, and four exact object-label occurrences
  instead of the one occurrence selected for that reconstructed invocation.
- Stage: public import pass; compile fail; execution/parity not run.
- Owning boundary: occurrence-preserving contract reconstruction in
  `CellProfilerModuleArtifactContracts` and `pipeline_import.py`, followed by
  exact compiled parameter-edge construction. The active registry-wide audit
  already identifies flattened candidate aggregation as erasing scalar
  occurrence partitions.
- Architectural question for the fixing owner: retain each scalar binding's
  exact ordered occurrence through public reconstruction and compilation. Do not
  choose the first object, branch on module names, or weaken scalar cardinality.

### F7 - Invocation main-flow ownership absent from compiled input edge

- Cases: `ExampleYeastColonies` (first observed).
- Exact failure: `Align` correctly declares `AlignedPlate` as sourced from
  `PlateTemplate`, but the compiled input edge does not retain that
  `PlateTemplate` is not the invocation main-flow input. Runtime reconstructs
  main-flow ownership from the inherited workspace `PRIMARY_PLANE` role and
  projects `PlateTemplate` from the `CombinedImage` payload; strict provenance
  validation rejects the mismatch.
- Stage: import pass; compile pass; execute fail before parity/timing.
- Owning boundary: compile-time
  `ArtifactDeclarationStepContext.main_flow_artifacts` projection into
  `InvocationArtifactInputEdgePlan`, consumed by generic runtime input binding.
- Architectural question for the fixing owner: persist invocation main-flow
  ownership on the exact compiled occurrence. Do not infer it from workspace
  projection roles or weaken image provenance validation.

### F8 - Aligned image input not projected to the exact invocation source

- Cases: `cp_tutorial_3d_monolayer` (first observed).
- Exact failure: PURE_3D `rescale_intensity` receives an `AlignedImageStack` as
  `source_data`; `CellProfilerFunctionContractExecutor.execute_pure_3d()` keeps
  the multi-named main-flow carrier unchanged and the leaf fails when converting
  it to `float32`.
- Stage: import pass; compile pass; execute fail before parity/timing.
- Owning boundary: invocation-source selection and runtime slice/context
  projection across `function_runtime.py`, `runtime_slice_projection.py`, and
  `function_contract_execution.py`.
- Architectural question for the fixing owner: project the exact declared image
  source/context before entering the scalar image callable. Do not make
  `rescale_intensity` unwrap a generic runtime carrier or coerce the carrier with
  `np.asarray`.

### F9 - Externally observed output identity omitted during reconstruction

- Cases: `cp_tutorial_pixel_based_classification` (first observed).
- Exact failure: execution succeeds, but the generated public Threshold step
  omits parsed output identity `ThreshPhase`. Candidate rows are emitted under
  generated subject `Threshold_1_image_1`, while the native schema observes
  `ThreshPhase`; strict comparison therefore reports the four Threshold
  diagnostics missing under the expected subject.
- Stage: import pass; compile pass; execute pass; strict parity fail with four
  missing measurement features.
- Owning boundary: output-identity retention in `pipeline_import.py` and the
  registered `SettingToKeywordBinding.output` reconstruction contract. The
  threshold row owner correctly emits all four diagnostic fields under the
  identity it receives.
- Architectural question for the fixing owner: retain output identity when
  measurement vocabulary externally observes it, using the generic occurrence
  proof. Do not add Threshold-specific names or rename rows during comparison.

### F10 - Relationship endpoint lacks payload-scoped object-ID projection

- Cases: `cp_tutorial_translocation_final` (first observed).
- Exact failure: `IdentifyTertiaryObjects` reaches relationship row
  materialization with endpoint `Cells` spanning more than one object-ID domain;
  `object_numbers_by_label_id()` requires one payload-scoped domain and rejects
  the unprojected value.
- Stage: import pass; compile pass; execute fail before parity/timing.
- Owning boundary: exact endpoint payload/domain projection before
  `relationship_measurement_rows.py` materializes parent/child rows, using the
  nominal object-label domain owners in `runtime_object_label_domains.py`.
- Architectural question for the fixing owner: select the endpoint's exact
  compiled occurrence and current payload domain before row construction. Do not
  choose one domain by order or weaken the one-domain invariant.

## Progress

- `2026-07-19T06:08:54-04:00`: Confirmed the existing official-30 manifest has
  exactly 30 cases. Confirmed the integration test uses cached references,
  `WellFilterConfig(well_filter=1)`, Napari baseline/disabled, public generated
  pipeline declarations, ZMQ compile-then-execute, and incremental observation
  files. Next: establish import coherence, then execute baseline.
- `2026-07-19T06:13:00-04:00`: The exact baseline integration node collected
  successfully (`1 test collected in 1.84s`), confirming the current tree is
  import-coherent. Next: serialized canonical baseline execution.
- `2026-07-19T06:11:20-04:00`: The representative ZMQ run released the host.
  Recorded a fresh tree fingerprint, acquired
  `/home/ts/.cache/openhcs/official30-runtime.lock`, and started the exact
  official-30 baseline integration node with a persistent isolated pytest root.
  Next: read incremental observations and classify completed cases.
- `2026-07-19T06:12:00-04:00`: Classified `ExampleColocalization` as F1. Public
  pipeline import and compilation completed; execution failed in exact compiled
  output-plan selection before parity. The 30-case run remains active so the
  shared failure surface can be measured before production changes.
- `2026-07-19T06:13:00-04:00`: `ExampleCometAssay` passed import, compile,
  execution, and strict parity in `4.571s`. Classified `ExampleFly` as F2: two
  compiled input occurrences for `object_labels:Nuclei` reach a consumer that
  requires one. The run remains active.
- `2026-07-19T06:15:00-04:00`: `ExampleFlyURL` and both first illumination cases
  passed strict parity. Classified `ExampleHuman` as F3: the legal image/object
  same-name pair `PH3` collides at compile-time because the artifact graph lookup
  is still name-keyed instead of typed-ref-keyed. The run remains active.
- `2026-07-19T06:16:00-04:00`: Classified
  `ExampleImagingFlowCytometryObjectsInGrid` as F4. Its channel-2 dict invocation
  correctly asks for `DF_image`, but execution loads channel-1 `BF_image`
  provenance. Exact metadata validation caught the routing drift. The run remains
  active.
- `2026-07-19T06:17:00-04:00`: Parent reported a concurrent
  NominalRefactorAdvisor process consuming one full CPU. Per `AGENTS.md`, retained
  this run for breadth/failure/parity classification and marked every execution
  time diagnostic/non-acceptance. The run will not be restarted; quiet-host
  acceptance timing is deferred until classification and fixes finish.
- `2026-07-19T06:18:00-04:00`: `ExampleSpeckles` passed strict parity.
  `ExamplePercentPositive` reproduced F4 with `OrigGreen` requested but
  channel-1 `OrigBlue` provenance loaded. `ExampleNeighbors` reproduced the F2
  occurrence-identity collapse through runtime-context source selection for the
  same `Cells` artifact used in multiple roles. The run remains active.
- `2026-07-19T06:25:00-04:00`: Adopted read-only monitoring after the process had
  emitted all 30 observations. Confirmed no official-30 pytest process remains.
  Cases 14-22 include six strict-parity passes, the F5 server-readiness race, F6
  scalar occurrence-partition loss in `ExampleVitra`, and F7 compiled main-flow
  ownership loss in `ExampleYeastColonies`. All six execution times are
  diagnostic because the advisor process remained active.
- `2026-07-19T06:29:00-04:00`: Classified the final eight records. Three strict
  parity passes complete the passing set. The remaining failures reduce to F8
  exact-source projection of an aligned image carrier, F6 scalar occurrence
  partition loss in two more pipelines, F9 externally observed Threshold output
  identity omission, and F10 payload-scoped relationship endpoint projection.
  The observation file contains exactly 30 records; all 17 recorded execution
  times are diagnostic/non-acceptance.
- `2026-07-19T06:29:03-04:00`: Confirmed the official-30 pytest process is no
  longer present. The final observation and suite metadata were written at
  `06:23:45`. Two orphaned port-7777 ZMQ launcher processes (`1818179` and
  `1818396`) remain, but neither is an official-30 pytest process. Recorded the
  post-run fingerprint and tree drift above; no process was stopped and no new
  workload was launched by this monitoring lane.
- `2026-07-19T06:30:40-04:00`: Finalized the read-only ledger. All 30 public
  sources converted/imported. Compilation totals are 25 pass, four fail, and one
  unknown because F5 failed before submission. Execution totals are 17 pass,
  eight fail, and five not reached. Strict parity totals are 16 pass, one fail,
  and 13 not reached. The 14 nonpassing cases map to ten shared classes: 13
  product failures across F1-F4 and F6-F10, plus the infrastructure-only F5.

## Parent Notes

- Preserve partial evidence across concurrent production edits.
- Baseline only before Napari.
- Failure classification must identify exact shared causes, not one local symptom
  per pipeline.
- This lane is read-only and has no repeated structural migration. Any production
  fixing lane derived from a failure class must first inventory the verified owner
  and all call sites with AST/CST tooling, reread peer ownership, and use a codemod
  for the mechanical portion. NominalRefactorAdvisor findings are evidence only,
  never semantic authority.

## Changed Files

- `docs/plans/official30_current_failure_classification_20260719.md` (ledger only)

## Verification

- Collector: PASS
  (`test_official30_compile_execute_and_match_native_references_over_zmq[baseline]`,
  one node collected).
- Public-source conversion/import: 30 pass, zero fail.
- Compile: 25 pass, four fail, one unknown/not submitted (F5).
- Execute: 17 pass, eight fail, five not reached.
- Strict parity: 16 pass, one fail, 13 not reached.
- Root-cause ownership: F1 compiled output-plan availability in
  `function_patterns.py` and compiler/runtime plan retention; F2 compiled input
  occurrence identity in compiler edges and the runtime adapter; F3 typed graph
  identity in `path_planner.py`; F4 dict-group source routing in
  `function_runtime.py`; F5 benchmark/ZMQ readiness lifecycle; F6 scalar
  occurrence reconstruction in module contracts and `pipeline_import.py`; F7
  compiled main-flow ownership in invocation edges and runtime artifact binding;
  F8 aligned-image invocation projection in function runtime/slice projection and
  function-contract execution; F9 externally observed output identity retention
  in `pipeline_import.py`; F10 endpoint domain projection in relationship-row and
  object-label-domain owners.
- Nonpassing cases: 13 product failures and one infrastructure-only pre-compile
  failure (F5), grouped into ten shared root-cause classes.
- Timing acceptance: INVALIDATED BY CONCURRENT CPU LOAD; every numeric value in
  the ledger is marked diagnostic.
- Stable-tree acceptance: INVALIDATED BY CONCURRENT PRODUCTION EDITS; failure
  classes are retained as exact historical observations, with changed-owner
  classes identified in Status.
- Process state: official-30 pytest stopped after 30/30 records; two orphaned ZMQ
  launchers remain and were not modified by this lane.
