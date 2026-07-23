# Registered CellProfiler Artifact Identity Reconstruction

**Date:** 2026-07-19
**Status:** Implementation complete; parent integration pending

## Objective

Make automatic input/output identity reconstruction complete across every
registered CellProfiler module. Preserve a parsed identity in public
`FunctionStep.func` kwargs only when exact nominal contracts prove subset/order
selection, same-type ambiguity, or externally observed vocabulary.

## Ownership

- `SettingToKeywordBinding` is the nominal owner of the public identity parameter
  and its declared artifact binding.
- `ArtifactSpec.parameter_name` is the execution authority for whether a compiled
  artifact input is projected into a callable parameter. Existing
  `special_inputs` metadata validates compatibility with that compiled contract;
  this migration does not delete or replace `special_inputs`.
- CellProfiler module contract reconstruction is the nominal owner of the exact
  candidate contract produced when that identity is omitted.
- `pipeline_import.py` owns occurrence-aware comparison of the parsed target
  binding with that reconstructed candidate and lowers only a proven selection
  into public kwargs.
- `FunctionStep` plus `PipelineConfig` remain the complete compiler input. No
  `.cppipe`, importer sidecar, or runtime workspace state participates afterward.
- `CellProfilerModuleArtifactContracts` and the `CellProfilerModule` MRO own
  shared derivation, occurrence partitioning, and reconstruction templates.
- Leaf modules own only semantic differences that cannot be expressed by the
  existing binding declarations and artifact relations.

## AST And Registry Inventory

- Parsed 119 production files.
- Found 181 artifact binding declarations: 111 inputs, 70 outputs.
- Found 13 repeated inputs, 7 repeated outputs, 47 runtime-parameter inputs, and
  one sidecar input.
- Inspected 91 concrete `CellProfilerModule.__registry__` values.
- Root `module_blocks_for_invocation` owns 81 modules; 10 leaves override it.
- Root `artifact_inputs_for_binding` owns 90 modules; one leaf override remains.
- Root `_artifact_input_record_groups` has four leaf overrides: Align, Tile,
  ImageMath, and MeasureObjectNeighbors.
- Root `artifact_names_for_binding` has four conditional/repeated output leaves.
- Vitra's defect is exact: importer candidate aggregation flattened three scalar
  `labels` occurrences into the same ref sequence as one repeated occurrence.

## Structured Migration Scope

- Add exact binding-occurrence reconstruction and comparison to the existing
  artifact-contract owner; no new policy or registry.
- Use CST to migrate repeated leaf `module_blocks_for_invocation` wrappers onto a
  root template hook, preserving only each leaf's semantic transformation body.
- Generalize ordered mixed scalar/repeated input assignment at the root and
  delete Align/Tile/ImageMath copies when registry tests prove exact equivalence.
- Retain genuinely distinct leaf behavior, including conditional active bindings,
  relation construction, and neighbor-role defaulting, as polymorphic hooks.
- Inventory every registered module before production edits, then use a structured
  AST/CST codemod for every repeated leaf shape. Manual edits are limited to
  verified semantic exceptions on the module MRO.
- Use `/home/ts/code/projects/NominalRefactorAdvisor` at `e8a3c50` only as
  navigation and ownership-audit evidence. Verify every finding against the
  OpenHCS MRO, registries, and nominal declarations before changing code.

## Constraints

- Fix the generic binding-occurrence/contract reconstruction path, never Vitra or
  `MeasureObjectIntensity` by name.
- No candidate counts, name heuristics, exception-driven fallback, mirrored
  tables, `getattr`, or `setattr`.
- Do not touch FilterObjects callable declarations, runtime output matching, or
  source-context execution.
- Do not perform the separate `special_inputs` deletion in this task.

## Executable Plan

- [x] Capture the exact Vitra target contract, canonical omission candidate, and
  generated public kwargs.
- [x] Trace selected-binding occurrence creation and projection through the
  importer using AST and exact nominal references.
- [x] Run NominalRefactorAdvisor plus AST/registry ownership inventories over the
  complete declaration/reconstruction/import path and record commands, coverage,
  and verified findings before production edits.
- [x] Repair the lost occurrence identity at its existing generic owner.
- [x] Add unambiguous-omission and ambiguous-explicit-selection regressions.
- [x] Round-trip generated source and compile it using only public declarations.
- [x] Run focused importer, transport, compiler, and static architecture tests.
- [x] Add root-owned occurrence reconstruction/equivalence and delete importer
  flattening plus unordered `Counter` matching. The comparison must accept both
  existing `special_inputs` metadata and compiled artifact inputs while taking
  execution projection only from `ArtifactSpec.parameter_name`.
- [x] Execute the CST leaf migration recorded above and verify every migrated
  module against pre-migration contracts.
- [x] Add registry-wide omission/explicit-selection/output-observation tests over
  every declared binding on every registered module.
- [x] Import and pycodify all official 30 pipelines; derive one exact nominal
  selection/ambiguity/observation proof for every retained identity kwarg.
- [x] Remove unproven identities through the same generic lowering and repeat the
  public import/transport/compile audit.
- [x] Record before/after step, dict-step, artifact-identity-kwarg, and authored
  step-local `source_bindings` AST counts plus representative generated source.

## Progress

- 2026-07-19 EDT: Read the governing automatic naming plan. The intended public
  API already distinguishes omission from exact selection; this fix completes
  that implementation rather than adding naming policy or module metadata.
- 2026-07-19 EDT: Expanded post-fix acceptance to the official 30. Supplied static
  baseline is 387 steps, 59 dict steps, 634 authored artifact-identity kwargs, and
  zero import failures. The audit will not presume any retained identity is valid;
  every one must have a transient contract-derived proof.
- 2026-07-19 EDT: Broadened implementation from Vitra to the complete registered
  module class. AST and runtime-registry inventory established the exact counts
  and override surface above before root or leaf migration.
- 2026-07-19 EDT: Corrected the execution boundary before implementation:
  `ArtifactSpec.parameter_name` is authoritative for compiled artifact-to-callable
  projection. `special_inputs` remains compatibility validation only and is not
  being removed in this batch. The omission decision belongs to the module
  artifact-contract ABC/MRO, not an importer policy or module-name table.
- 2026-07-19 EDT: Resume audit read all live peer plans and confirmed this task's
  narrow automatic identity slice while preserving the concurrent generic
  `special_inputs` normalization boundary. NominalRefactorAdvisor `e8a3c50` was
  run evidence-only on the four owner files with context expansion disabled; its
  repeated record-group and selection-carrier findings agree with the verified
  MRO owner inventory and do not define the implementation.
- 2026-07-19 EDT: Current focused behavior is already green (`2 passed`) for both
  canonical repeated-image omission and exact scalar retention across `Nuclei`,
  `Cells`, and `Cytoplasm`. The implementation is still incomplete because
  importer candidates are flattened by binding and unordered `Counter`
  equivalence remains. The next production batch moves occurrence shape and
  equivalence to `CellProfilerModuleArtifactContracts`, then performs the
  structured leaf migration and registry audit.
- 2026-07-19 EDT: AST classified all ten leaf reconstruction overrides. Nine are
  post-reconstruction transformations for the root template hook; GrayToColor is
  the one pre-reconstruction semantic override. The three rootable ordered input
  assignments are Align, Tile, and ImageMath; neighbor-role defaulting remains a
  leaf hook. Baseline topology/repeated-row/import coverage is green (`63 passed`).
  Retained identities remain visible by correlating public invocation kwargs with
  existing exact contracts, compiled edges/plans, source bindings, and scopes,
  without persistent reason metadata.
- 2026-07-19 EDT: Implemented nested occurrence reconstruction and comparison on
  the module artifact-contract root, then replaced importer flattening and
  unordered `Counter` matching with that API. Runtime-parameter partitions are
  strict; main-flow equivalence accepts only exact ordered occurrences, with the
  outer occurrence order immaterial.
- 2026-07-19 EDT: Completed the structured LibCST migration of all nine
  post-reconstruction leaf wrappers onto the root finalization hook. Moved the
  shared Align/Tile/ImageMath ordered assignment into the root while retaining
  ImageMath's conditional binding selection and the independent neighbor-role
  semantic hook. Post-migration topology, repeated-row, and importer coverage is
  green (`63 passed in 5.27s`). Registry and public generated-source proof is next.
- 2026-07-19 EDT: Added live-registry structured binding/MRO gates plus a
  compile-only authored-identity graph test. The graph test uses only existing
  public invocation, contract plan, typed artifact input/edge, and source-binding
  objects to expose authored identity, resolved ref/runtime parameter, source
  alias, and component identity. No reason metadata or sidecar was introduced.
- 2026-07-19 07:04 EDT: Made omission proof stable under its own public result.
  The importer now reconstructs all declared input bindings to a fixed point after
  each newly retained sibling identity and compares exact root-owned occurrence
  partitions before accepting omission. The permissive measurement-contract
  experiment used to diagnose the RelateObjects failure has been removed; the
  existing strict conflict gate is unchanged. Focused tests and the generated
  all-30 compile-only API audit remain next.
- 2026-07-19 07:06 EDT: Initial all-30 generated-Python gate passed exact public
  source/config round-trip and compile-only reconstruction for 26 pipelines. It
  rejected ExampleFly `GrayToColor` and three `RelateObjects` lowerings before
  source emission. Passing-subset AST totals are 297 steps, 44 dict steps, 448
  authored identities, 113 step configs, 15 step source bindings, and zero exact
  pipeline source-binding copies. Binding-level root diagnosis is in progress;
  no official30 execution occurred.
- 2026-07-19 07:12 EDT: Corrected the fixed point to compare only still-omitted
  bindings. Retained sibling identities are already exact public invocation-plan
  inputs; omitted alternatives may duplicate them across candidate blocks and do
  not invalidate that authored identity. A reversed parent/child producer-order
  importer regression plus occurrence merge/conflict tests pass (`6 passed`).
- 2026-07-19 07:12 EDT: AST method inventory and repository merge/occurrence
  search found no `ArtifactSpec`, `ArtifactSpecCollection`, or `CallableContract`
  owner for composing multiple dynamic contracts with duplicate-ref occurrence
  alignment. `ArtifactSpecCollection.select_declared_occurrences()` selects from
  one declaration and does not compose contracts. The CP-specific
  `combine_callable_contracts()` therefore remains root-owned on
  `CellProfilerModule`; no generic-core move or forwarding helper is warranted.
  Exact search also confirms the reported duplicate consumed-name assignment is
  reduced to one live assignment.
- 2026-07-19 07:17 EDT: Strengthened transient AST/API proof passed lazy
  processing/source field-delta assertions on every completed case, then exposed
  two harness classification errors: variant callables were keyed by the module's
  canonical callable, and the pre-existing invocation-owned CP source-position
  field was treated as new reason metadata. Both are corrected; no production
  sidecar or config was introduced. Complete rerun is next.
- 2026-07-19 07:19 EDT: Final generated-Python gate is green for all 30 public
  pipelines. Exact source/config regeneration and compile-only contract planning
  pass for 399 steps and 456 CP invocations. AST inventory reports 63 dict steps,
  695 authored identity kwargs, 167 lazy processing configs/262 inherited-delta
  fields, 17 lazy step-source configs/28 fields, and zero copied pipeline source
  bindings. Every generated identity has a transient exact input-occurrence or
  observed-output occurrence proof; no unproved identity remains. Nine grouped
  patterns with identical syntax have distinct group contracts, so they are not
  collapsible full-domain duplicates. Parent baseline 387/59/634 versus live
  399/63/695 is recorded as +12/+4/+61 on the concurrent current tree. No
  official30 execution occurred.
- 2026-07-19 07:22 EDT: Final review passes scoped Ruff, migrated-file
  `py_compile`, scoped `git diff --check`, and 136 focused tests. Live registry
  postconditions remain 91 concrete values with only GrayToColor and
  MeasureObjectNeighbors at the two documented leaf override points.
  `special_inputs` declarations and projection remain intact. Representative
  generated ExampleFly source is `/tmp/official30_generated_examplefly.py`.
  Parent integration and the separately serialized official30 execution are next.
- 2026-07-19 07:23 EDT: Tightened CP root occurrence composition to compare
  `ArtifactSpec.parameter_name` explicitly because that execution binding is not
  part of dataclass equality. The swap-only regression passes with all existing
  merge/conflict gates (`4 passed`); ownership remains on the CP MRO root.

## Changed Files

- Root owners: `settings_binder.py`, `module_artifact_contracts.py`,
  `module_declarations.py`, and `pipeline_import.py`.
- Ordered-assignment leaves: `alignment.py`, `image_geometry.py`, and
  `image_math.py`.
- Structured finalization-hook leaves: `area_occupied.py`, `classification.py`,
  `colocalization.py`, `export_to_database.py`, `intensity_distribution.py`,
  `morphology.py`, `outlines.py`, `save_images.py`, and `worms.py`.
- This plan and `.agents/measure-object-intensity-selection-retention.md`.
- `tests/unit/test_cellprofiler_artifact_identity_reconstruction.py`.

## Verification

- Fixed-point omission, registry graph, and strict conflict selection:
  `11 passed in 3.15s`; owner/test Ruff, `py_compile`, and scoped
  `git diff --check` pass.
- Post-fixed-point topology/repeated-row/import batch:
  `63 passed in 5.44s`.
- Initial all-30 generated-Python command:
  `.venv/bin/python /tmp/official30_public_api_gate.py > /tmp/official30_public_api_gate_20260719.jsonl`
  -> 26/30 compile-only passes; four importer lowering failures listed in
  Progress. No ZMQ or pipeline execution.
- Omitted-sibling/root-composition focused selection: `6 passed in 2.97s`;
  scoped Ruff and `py_compile` pass. Exact consumed-name assignment count: one.
- Strengthened generated-source diagnostic: audit exit 1 after 11 completed cases
  due to two corrected harness assumptions; completed lazy inheritance and source
  duplication assertions passed. No pipeline execution.
- Final all-30 generated-Python gate command:
  `.venv/bin/python /tmp/official30_public_api_gate.py > /tmp/official30_public_api_gate_20260719_final2.jsonl`
  -> exit 0 with 30/30 round-trip/import/compile-only passes and the exact AST
  totals recorded in Progress. No pipeline execution.
- Final post-review all-30 rerun writes
  `/tmp/official30_public_api_gate_20260719_final3.jsonl` and exits 0 with the
  same exact 30/30 totals.
- Final focused repository batch: `136 passed in 48.33s` across reconstruction,
  importer, topology, repeated-row, artifact composition, and static deletion
  gates.
- Final post-review focused rerun: `137 passed in 51.75s`.
- Final Ruff, syntax, diff, registry, and `special_inputs` compatibility
  postconditions pass. Representative source generation exits 0.
- Exact runtime-parameter occurrence composition selection: `4 passed in 4.26s`.
- Resume focused baseline: `2 passed in 4.01s` for the two exact
  `MeasureObjectIntensity` omission/selection tests.
- Post-migration exact reconstruction regressions: `4 passed in 4.68s`.
- Post-migration topology, repeated-row, and importer batch:
  `63 passed in 5.27s`.
- Registry/MRO/compiled-graph gates: `4 passed in 3.01s`; combined focused
  registry and importer selection batch: `5 passed in 3.34s`.
- Focused lint and Python compilation pass for the root implementation and new
  test. Broader touched-leaf lint remains blocked by pre-existing dirty-tree
  errors unrelated to the structured method migration.
- Registry-wide reconstruction, public transport/compile, and official-30 proof
  pending.
