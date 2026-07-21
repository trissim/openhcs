# CellProfiler Artifact-Parameter Declaration Contract Fix

## Status

Complete at the artifact-fed callable-parameter declaration boundary. The full
registry invariant and all three canonical ZMQ compilations pass. Two cases expose
separate source-image provenance failures during execution, recorded below.

## Ownership

- Scope: every registered CellProfiler callable parameter fed by a declared
  artifact input.
- Production owners: the affected callables in
  `openhcs/processing/backends/cellprofiler/object_filtering.py` and
  `openhcs/processing/backends/cellprofiler/area_occupied.py`.
- Contract consumer: generic artifact-parameter validation in
  `openhcs/core/pipeline/function_contracts.py`.
- Module binding owner: `FilterObjectsModule` and `FilterObjectsInputPolicy` in
  `openhcs/processing/backends/cellprofiler/object_filtering.py`.
- Non-goals: pipeline import, source context, source bindings, artifact selection,
  and module-name-specific compiler branches.

## Authority Inventory

- `special_inputs(...)` is the canonical callable declaration for parameters fed
  from compiled artifact inputs.
- `runtime_bound_parameters(...)` declares non-artifact values supplied by runtime
  infrastructure.
- `SettingToKeywordBinding.input(..., runtime_parameter_name=...)` records the exact
  callable parameter consumed by an artifact input.
- `validate_artifact_input_parameter_bindings(...)` enforces that adapter-managed
  artifact parameters are present in the callable's `special_inputs` declaration.
- `ObjectLabelsInputBindingMixin` binds declared object artifacts to the exact
  `object_labels` callable parameter; it does not own whether that parameter is an
  artifact input or runtime-infrastructure input.
- Existing object-label callables such as `overlay_outlines`, `straighten_worms`,
  and `measure_object_intensity_distribution` declare their artifact-fed label
  parameters with `special_inputs(...)`.
- `CellProfilerModule.__registry__`, `declared_artifact_bindings(...)`, callable
  signatures, `special_input_names_from_callable(...)`, and
  `runtime_bound_parameter_names_from_callable(...)` together expose the complete
  declaration surface without a copied module/parameter inventory.

## Exact Root Cause

`FilterObjectsModule.input_binding` and `additional_input_binding` correctly bind
`ObjectLabelsArtifactType` inputs to `object_labels`. `FilterObjectsInputPolicy`
then binds those exact compiled object artifacts to the same callable parameter.
However, `filter_objects` currently includes `ObjectLabelsRuntimeParameter` in
`runtime_bound_parameters(...)` and does not declare `object_labels` with
`special_inputs(...)`.

The callable declaration therefore disagrees with the module-owned artifact
binding. Generic compile validation rejects Nuclei, Prespots, and Tile_of_grid for
the same reason: each artifact binds `object_labels`, which the callable has not
declared as an artifact-fed special input. The compiler check is correct and must
not be weakened.

Registry traversal found two additional edges in the same mismatch class:

- FilterObjects' `enclosing_object_labels` is artifact-fed by
  `enclosing_object_binding` but was declared as runtime infrastructure.
- `measure_image_volume_occupied_objects` declared a `labels` special input while
  the owning `objects_binding` targets the canonical `object_labels` parameter.

The complete fix is therefore three callable-parameter declaration corrections,
not three pipeline-specific exceptions.

## Executable Plan

- [x] Inventory the nominal callable, module binding, object-input policy, and
  generic validation owners.
- [x] Traverse every registered module/callable and AST-audit backend decorators to
  enumerate the complete mismatch class.
- [x] Move `object_labels` to the callable's `special_inputs(...)` declaration and
  remove `ObjectLabelsRuntimeParameter` from `runtime_bound_parameters(...)`.
- [x] Move FilterObjects `enclosing_object_labels` to `special_inputs(...)` and
  remove it from `runtime_bound_parameters(...)`.
- [x] Align the AreaOccupied 3-D object callable with its owning
  `objects_binding` canonical parameter.
- [x] Replace the one-direction registry test with a bidirectional invariant:
  applicable artifact binding parameters equal special inputs and are disjoint
  from runtime-bound parameters.
- [x] Add focused public `FunctionStep` reconstruction/compile regressions for the
  primary object identities Nuclei, Prespots, and Tile_of_grid.
- [x] Prove the same three reconstructed steps compile with their public
  `PipelineConfig` over canonical ZMQ.
- [x] Run focused callable-contract, reconstruction, and compile suites.
- [x] Run canonical ZMQ compile checks for ExamplePercentPositive,
  ExampleYeastPatches, and ExampleImagingFlowCytometryObjectsInGrid when fixtures
  are available.

## Progress

- 2026-07-19: Audited registries, callable decorators, module artifact bindings,
  object-label input policies, and generic artifact-parameter validation. Confirmed
  one declaration mismatch shared by all three failures; no compiler or source
  binding exception is required.
- 2026-07-19: Corrected the callable declaration at its owner. `object_labels` now
  uses `special_inputs(...)` through the existing nominal parameter declaration and
  is no longer classified as non-artifact runtime infrastructure.
- 2026-07-19: Added one parametrized public FunctionStep reconstruction regression
  for Nuclei, Prespots, and Tile_of_grid. The first run intentionally reached a
  distinct FilterObjects output-cardinality ABI failure when it used the broad
  module validator; the test now invokes the exact generic artifact-input compiler
  gate that produced the assigned failures.
- 2026-07-19: Runtime binding regressions passed (`3 passed`). Ruff and scoped
  diff checks pass. Subsequent collection is blocked because concurrent
  `bioformats_adapter.py` edits temporarily removed `BioFormatsCompositeAdapter`
  while eager microscope discovery still imports it. This task does not own or
  modify that boundary.
- 2026-07-19: The registry-wide special-input gate independently reports
  `measure_image_volume_occupied_objects: labels`; this is an existing AreaOccupied
  declaration mismatch now included in the expanded ownership boundary.
- 2026-07-19: Registry traversal across all 96 registered CellProfiler modules
  found exactly two remaining callable rows before the expanded fix: FilterObjects
  (`enclosing_object_labels` artifact-bound but runtime-bound) and
  MeasureImageAreaOccupiedBinary's 3-D object variant (`labels` special input versus
  `object_labels` binding). Both are corrected at their callable declarations.
- 2026-07-19: The registry regression now derives all applicable parameter names
  from registered bindings and signatures. It contains no module names, parameter
  allowlist, or copied exception table.
- 2026-07-19: Post-fix AST/registry audit parsed 69 backend files, traversed 91
  registered module entries and every declared callable variant, and reported
  `MISMATCHES=[]`.
- 2026-07-19: Focused verification passes: registry invariant `1 passed`, three
  exact FilterObjects identities `3 passed`, and combined AreaOccupied plus
  FilterObjects runtime coverage `9 passed`.
- 2026-07-19: Canonical serialized ZMQ integration compiled all three assigned
  public `PipelineConfig + FunctionStep` pipelines. ExampleYeastPatches executed
  successfully. ExamplePercentPositive then failed because an `OrigBlue` payload
  was used where `OrigGreen` was declared; ImagingFlowCytometryObjectsInGrid then
  failed because a `BF_image` payload was used where `DF_image` was declared.
  These are downstream source-image provenance defects, not artifact-parameter
  declaration or FilterObjects compile failures.

## Changed Files

- `docs/plans/filter_objects_special_input_contract_fix_20260719.md` (this tracker)
- `openhcs/processing/backends/cellprofiler/object_filtering.py` (corrected callable
  input declaration)
- `openhcs/processing/backends/cellprofiler/area_occupied.py` (aligned the 3-D
  object-volume callable with its nominal artifact binding)
- `tests/unit/test_cellprofiler_generated_pipeline_execution.py` (three exact
  public reconstruction regressions)
- `tests/unit/test_cellprofiler_generic_special_input_binding.py` (registry-wide
  bidirectional declaration invariant)

## Verification

- `.venv/bin/ruff check openhcs/processing/backends/cellprofiler/object_filtering.py tests/unit/test_cellprofiler_generated_pipeline_execution.py`
  -> passed.
- `.venv/bin/pytest -q tests/unit/test_cellprofiler_module_execution.py -k 'filterobjects_runtime_plan or object_input_binding_resolves_exact_compiled_label_artifact' --tb=short`
  -> `3 passed, 375 deselected` before the concurrent import surface changed.
- `git diff --check --` over the three owned files -> passed.
- `.venv/bin/pytest -q tests/unit/test_cellprofiler_generic_special_input_binding.py::test_registered_module_artifact_parameters_are_exact_special_inputs --tb=short`
  -> `1 passed`.
- `.venv/bin/pytest -q tests/unit/test_cellprofiler_generated_pipeline_execution.py::test_filter_objects_function_step_reconstructs_exact_public_topology --tb=short`
  -> `3 passed`.
- `.venv/bin/pytest -q tests/unit/test_cellprofiler_area_occupied.py tests/unit/test_cellprofiler_module_execution.py -k 'area_occupied or filterobjects_runtime_plan or object_input_binding_resolves_exact_compiled_label_artifact' --tb=short`
  -> `9 passed, 375 deselected`.
- `.venv/bin/pytest -q tests/unit/test_cellprofiler_generic_special_input_binding.py --tb=short`
  -> `6 passed`.
- `.venv/bin/pytest -q tests/unit/test_cellprofiler_generated_pipeline_execution.py -k 'filter_objects_function_step_reconstructs_exact_public_topology or filter_objects_contract' --tb=short`
  -> `5 passed, 17 deselected`.
- `CELLPROFILER_EXAMPLES_ROOT=/home/ts/.cache/openhcs/cellprofiler_examples .venv/bin/pytest -q tests/integration/test_cellprofiler_generated_pipeline.py::test_official_cellprofiler3_representative_pipelines_execute_over_zmq -k 'percent-positive or yeast-patches-grid-illumination or imaging-flow-cytometry-grid' --tb=short`
  -> all three compile over ZMQ; ExampleYeastPatches passes execution; the other
  two reach the separate source-image provenance failures recorded above
  (`1 passed, 2 failed, 6 deselected`).
- Final scoped Ruff and `git diff --check` -> passed.
