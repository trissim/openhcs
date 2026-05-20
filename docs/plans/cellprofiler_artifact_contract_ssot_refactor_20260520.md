# CellProfiler Artifact Contract SSOT Refactor

## Problem

CellProfiler pipeline import currently lowers one parsed `.cppipe` module into
two related projections:

- `StepSourceBindingsConfig`, stored on the generated `FunctionStep`, describes
  how external aliases such as `OrigBlue` resolve from source images or step
  inputs.
- `ModuleArtifactContract`, attached to the runtime callable, describes the
  module's artifact inputs and outputs, including sidecars such as crop masks
  and measurement tables.

Both are produced from the same parsed CellProfiler symbol model. That is safe
while they remain generated projections from a single authority. It becomes an
SSOT risk once generated pipeline Python, JSON sidecars, ObjectState, and the UI
can persist or edit one projection without rebuilding or validating the other.

The immediate symptoms are:

- Source bindings are editable in the UI, but artifact contracts are hidden
  runtime metadata. Drift can occur if a binding alias/kind changes while a
  callable still carries stale `ArtifactSpec` inputs.
- CellProfiler runtime wrappers are visible to generic function editing, so UI
  introspection can see runtime adapter parameters and degraded annotations
  instead of the raw backend function signature and documentation.
- Generated code can show internal artifact-contract structure that looks like
  user-editable pipeline configuration even though users should never manually
  maintain it.

## Verified Code Paths

- `openhcs/interop/cellprofiler/symbol_table.py`
  - `CellProfilerSymbol.artifact_spec()` converts one parsed CP symbol into an
    `ArtifactSpec`.
  - `_SymbolTableBuilder.source_bindings_for(...)` converts external CP symbols
    into `StepSourceBindingsConfig`.
  - `CellProfilerContractAssemblyMixin.assemble_contract(...)` builds one
    `ModuleArtifactContracts` value containing both symbol-derived artifact
    declarations and symbol-derived source bindings.
  - `ModuleArtifactContracts.module_contract` projects the import-time symbol
    contract into a runtime `ModuleArtifactContract`.

- `openhcs/interop/cellprofiler/pipeline_generator.py`
  - `PipelineGeneratorRuntimeContractProjector.by_module_num(...)` already
    projects symbol-table contracts into runtime-only `ModuleArtifactContract`
    values without serializing them into the generated pipeline code.
  - `PipelineGeneratorStepEmitter.generate_steps_from_registry(...)` writes
    `FunctionStep(..., source_bindings=...)` into generated Python when a module
    has external source symbols.
  - `ModuleProcessingComponentStrategy` currently derives processing components
    from `contract.source_bindings` and `contract.runtime_artifact_inputs`.

- `openhcs/interop/cellprofiler/runtime/generated_pipeline.py`
  - `GeneratedPipelineContractSidecar` persists runtime contracts to JSON.
  - `GeneratedPipelineRuntimeModule.load_from_source(...)` registers runtime
    contracts, imports generated code, then calls
    `bind_generated_pipeline_runtime(...)`.
  - `GeneratedPipelineRuntimeBindings.apply()` mutates imported `FunctionStep`
    objects by replacing backend callables with `CellProfilerRuntimeCallable`
    instances resolved by module name/order.

- `openhcs/interop/cellprofiler/runtime/module_execution.py`
  - `CellProfilerRuntimeCallable` attaches `ModuleArtifactContract` metadata,
    injects hidden runtime adapter parameters, and preserves the raw processing
    function through `attach_callable_contract_metadata(...,
    raw_processing_function=raw_func, ...)`.

- `openhcs/core/invocation_artifacts.py`
  - `InvocationArtifactDeclarations.from_contract(...)` treats
    `ModuleArtifactContract.runtime_artifact_inputs` and `.outputs` as the
    runtime artifact declaration authority for an invocation.

- `external/python-introspect/src/python_introspect/signature_analyzer.py`
  - `SignatureAnalyzer.analyze(...)` uses `inspect.signature(...)` and
    `typing.get_type_hints(...)` on the object it receives. When it receives a
    runtime wrapper instead of the raw backend callable, it can expose injected
    runtime params or fall back to string annotations.

## SSOT Rule

The durable authority must be the parsed CellProfiler module descriptor plus its
symbol table, not the generated runtime wrapper.

The split should be:

- `CellProfilerSymbolTable` owns semantic module inputs, outputs, sidecars,
  measurement artifacts, and external source aliases.
- `StepSourceBindingsConfig` is the only user-editable representation of how
  external inputs are matched to OpenHCS sources.
- `ModuleArtifactContract` is a compiled/runtime projection, not editable user
  state.
- Generated Python should contain user-facing pipeline configuration only:
  `FunctionStep`, bound module settings, `source_bindings`, processing config,
  and streaming/materialization config.
- Runtime contracts may be cached for performance, but caches must be validated
  against the current pipeline/source-binding state or regenerated before
  compile/execution.

## Target Architecture

### 1. Preserve Import-Time Semantic Authority

Introduce an explicit import artifact that keeps the complete CP semantic model
available after generation:

```python
@dataclass(frozen=True, slots=True)
class CellProfilerPipelineSemanticModel:
    provenance: CellProfilerPipelineProvenance
    source_schema: PipelineImageSchema
    symbol_table: CellProfilerSymbolTable
    module_contracts: tuple[ModuleArtifactContracts, ...]
```

This model should be the source for:

- generated `FunctionStep.source_bindings`
- runtime `ModuleArtifactContract`
- read-only UI artifact previews
- validation of edited source bindings against expected module inputs

The existing `CellProfilerPipelineImportResult` can carry this model or a
versioned equivalent. The current `artifact_contracts:
tuple[ModuleArtifactContract, ...]` is insufficient because it has already lost
the source-binding side of the symbol table.

### 2. Keep Generated Python User-Facing

Generated pipeline code should not inline full `ModuleArtifactContract` literals
or expose `ArtifactSpec` as normal step config. The code should continue to
render `source_bindings=StepSourceBindingsConfig(...)`, because that is
user-facing editable pipeline state.

Runtime contracts should be produced by one of these authorities, in priority
order:

1. Recompute from `CellProfilerPipelineSemanticModel` at load/compile time.
2. Read a sidecar cache that includes a semantic-model fingerprint and validate
   it against the current generated pipeline/module-source state.
3. Reject stale sidecars with a clear error, never silently execute stale
   contracts.

### 3. Rebuild Runtime Contracts After Source-Binding Edits

Editing `StepSourceBindingsConfig` must not require manual `ArtifactSpec` edits.
Instead, compile/load should derive runtime declarations from:

- expected module input/output symbols from the semantic model
- current step source bindings for external inputs
- module settings that affect declared outputs

The invariant should be:

```text
runtime_contract.inputs external aliases == step.source_bindings declared aliases
runtime_contract.runtime_artifact_inputs == prior-module artifacts only
```

If the user renames an external alias in source bindings, the compiler should
either:

- update the source-bound input projection when the semantic model permits the
  alias, or
- reject the step as invalid because the CP module still semantically expects
  the original alias.

It should not execute with a stale `ModuleArtifactContract` that still declares
the old input name.

### 4. Separate Runtime Callable From UI Callable

`CellProfilerRuntimeCallable` is an execution adapter. The UI should inspect the
raw backend callable, not the runtime adapter object.

Add a central callable-introspection projection in `python-introspect`, not in
PyQt and not in a CellProfiler-specific widget branch. The generic shape should
be a small registry API:

```python
set_signature_analysis_target(wrapper, raw_callable)
signature_analysis_target(wrapper)
```

`SignatureAnalyzer` should resolve that analysis target once at the beginning of
callable analysis. Wrapper types can then expose their product callable without
changing every UI call site. `CellProfilerRuntimeCallable` should set the
projection to `raw_func`. This keeps:

- raw parameter names
- resolved enum/type annotations
- docstrings and parameter descriptions
- hidden runtime adapter params out of the form

This is deliberately not a widget hack. It belongs in the introspection package
because the problem class is generic: runtime/proxy/wrapper callables need one
nominal way to declare what should be inspected for user-facing parameter
editing.

### 5. Add Read-Only Artifact Preview UI

The UI should not edit `ModuleArtifactContract`, but it should be able to show a
compact read-only summary:

- inputs: alias, kind, origin/source-bound vs artifact-bound
- outputs: name, kind, materialization, sidecar role
- warnings: stale/missing source binding, unresolved runtime input, pruned
  output

This preview should be derived from the same semantic model/runtime projection
used by compile. It must not become another editable copy.

## Implementation Phases

### Phase 1: Python-Introspect Projection Guardrails

- Add a generic `SIGNATURE_ANALYSIS_TARGET_ATTR` extension point to
  `python-introspect`.
- Add `set_signature_analysis_target(...)` and
  `signature_analysis_target(...)` helpers so products do not hand-roll private
  attribute names or structural probes.
- Update `SignatureAnalyzer._analyze_callable(...)` and wrapper fallback
  extraction to use that projection before reading signatures, docstrings, or
  type hints.
- Export the helpers from `python_introspect.__init__`.
- Add tests proving a wrapper with runtime-only parameters analyzes as its raw
  target.

### Phase 2: CellProfiler Runtime Callable Projection

- Set the `python-introspect` analysis target on `CellProfilerRuntimeCallable`
  to the raw absorbed backend function.
- Keep the runtime signature on `CellProfilerRuntimeCallable` intact for
  execution. Only UI/introspection analysis should project to the raw callable.
- Add tests that `SignatureAnalyzer.analyze(...)` on a CP runtime callable uses
  the raw backend callable for UI-facing parameters and does not expose
  `cellprofiler_runtime` or `runtime_invocation_options`.

### Phase 3: Contract Drift Guardrails and Tests

- Add tests that import a CP pipeline, edit a source binding alias/kind, and
  prove compile either regenerates matching runtime contracts or rejects the
  drift.
- Add tests that artifact preview data is read-only and derived from the same
  projection used by `InvocationArtifactDeclarations`.

### Phase 4: Semantic Model Persistence

- Extend `CellProfilerPipelineImportResult` to preserve a semantic-model
  projection rather than only runtime `ModuleArtifactContract` values.
- Add a generated Python semantic-contract module, via pycodify, so
  `ModuleArtifactContracts` remain nominal Python objects instead of a second
  hand-written JSON schema.
- Add fingerprints over source `.cppipe`, module order, module settings, and
  generated source-binding declarations.

### Phase 5: Compile-Time Contract Projection

- Move runtime contract binding behind a nominal authority that accepts:
  current `FunctionStep`, current `StepSourceBindingsConfig`, and the semantic
  model entry for that module.
- Make `GeneratedPipelineRuntimeBindings` call that authority instead of binding
  stale sidecar contracts directly by module-name order.
- Ensure `InvocationArtifactDeclarations.from_contract(...)` still sees a
  `ModuleArtifactContract`, but that contract is regenerated/validated for the
  current step.

### Phase 6: Read-Only Artifact Preview

- Add a compact read-only artifact contract widget/panel for `FunctionStep`
  editors.
- Source the preview from the same compile-time projection, not from ObjectState
  editable fields.
- Highlight drift/errors instead of offering editable `ArtifactSpec` tables.

## Non-Goals

- Do not make users manually edit `ArtifactSpec`.
- Do not duplicate `ArtifactSpec` fields into ObjectState form models.
- Do not make microscopes or generic OpenHCS metadata handlers understand
  CellProfiler module semantics.
- Do not special-case CellProfiler in individual PyQt widgets when a central
  callable/introspection authority can solve the same class of problem.

## Verification Gates

- `.cppipe` import still auto-loads generated `FunctionStep` objects into the
  GUI.
- Editing source bindings updates ObjectState and either compiles with matching
  regenerated runtime contracts or fails with a specific drift error.
- ExampleFly compile and ZMQ execution no longer depends on pickled stale
  runtime wrapper identity.
- Function editor for a CP module shows raw backend parameters, real enum/type
  hints, and docstrings; runtime adapter params remain hidden.
- Advisor run on touched files should not report new SSOT, hardcoded family, or
  widget-special-case regressions.

## Advisor Calibration

Scoped advisor run:

```bash
nominal-refactor-advisor \
  openhcs/interop/cellprofiler/symbol_table.py \
  openhcs/interop/cellprofiler/pipeline_generator.py \
  openhcs/interop/cellprofiler/runtime/generated_pipeline.py \
  openhcs/interop/cellprofiler/runtime/module_execution.py \
  openhcs/core/invocation_artifacts.py \
  external/python-introspect/src/python_introspect/signature_analyzer.py \
  --include-plans
```

Findings were concentrated in
`external/python-introspect/src/python_introspect/signature_analyzer.py`:

- `SignatureAnalyzer` and `DocstringExtractor` still recover semantic roles
  through attribute probes and string-section dispatch.
- The advisor recommends a nominal template-method/dispatch authority for that
  subsystem.

That supports Phase 4: UI callable selection should be centralized as an
introspection projection authority, not added as a PyQt widget-level
CellProfiler special case.

## Implementation Progress

Completed:

- Added a generic `python-introspect` signature-analysis projection registry:
  `set_signature_analysis_target(...)` and `signature_analysis_target(...)`.
- Updated `SignatureAnalyzer` to analyze explicit projection targets before
  reading signatures, type hints, or docstrings.
- Updated `CellProfilerRuntimeCallable` to project UI/signature analysis to the
  absorbed raw backend function recorded by `CallableContract`, while preserving
  its runtime adapter signature for execution.
- Added focused tests proving CellProfiler runtime callables expose raw backend
  parameter types such as `CropShape | str` and do not expose
  `cellprofiler_runtime` / `runtime_invocation_options` to UI analysis.
- Added `SourceBindingRuntimeContractGuard` during generated runtime binding so
  edited `StepSourceBindingsConfig` values cannot silently drift from
  source-bound `ModuleArtifactContract.inputs`.
- Updated `bind_generated_pipeline_runtime(...)` to register direct contract
  inputs before `CellProfilerRuntimeStepBinding` resolves module-number
  bindings, so direct and sidecar-backed import paths share one authority.
- Preserved full `ModuleArtifactContracts` semantic contracts on
  `CellProfilerPipelineImportResult.semantic_contracts` so product import
  callers no longer receive only flattened runtime `ModuleArtifactContract`
  values.
- Added `GeneratedPipelineSemanticContractsModule`, which persists full
  `ModuleArtifactContracts` values as importable pycodify-generated Python and
  exposes them from generated modules as `CELLPROFILER_SEMANTIC_CONTRACTS`.
- Added `GeneratedPipelineSemanticContractsFingerprint`, hashing source `.cppipe`
  bytes/path, generated Python, and pycodify-rendered semantic contracts; the
  generated import module now passes the expected fingerprint when loading the
  semantic sidecar and rejects mismatches.
- Replaced runtime binding's module-name bucket matching with an ordered
  module-number contract stream that validates each generated step callable
  against its exact runtime contract before wrapping.
- Added `ArtifactContractPreview` as a frozen core projection for read-only UI
  display of artifact inputs, outputs, sidecar roles, and source-binding vs
  runtime-artifact origins.
- Added a PyQt `ArtifactContractPreviewWidget` tab in the step editor. It is
  read-only and derives rows from `CallableContract.module_artifact_contract`,
  not editable ObjectState fields.
- Moved source-binding/runtime-contract alignment into the core
  `SourceBindingRuntimeContractGuard`, which now exposes both fail-loud
  validation for compile/runtime binding and a non-throwing alignment report for
  the artifact preview. The PyQt preview shows the same missing/unexpected
  source-binding drift before compile.

Verified:

```bash
PYTHONPATH=src ../../.venv/bin/python -m pytest \
  tests/test_signature_analyzer.py::TestSignatureAnalyzer::test_analyze_uses_declared_signature_analysis_target \
  tests/test_init.py::TestPublicAPI::test_all_exports -q

PYTHONPATH=external/python-introspect/src .venv/bin/python -m pytest \
  tests/unit/test_cellprofiler_runtime_callable_introspection.py -q

PYTHONPATH=external/python-introspect/src .venv/bin/python -m pytest \
  tests/unit/test_cellprofiler_symbol_table.py::test_pipeline_generator_emits_compiled_artifact_contracts -q

PYTHONPATH=external/python-introspect/src .venv/bin/python -m pytest \
  tests/unit/test_artifact_contract_preview.py \
  tests/unit/test_cellprofiler_runtime_callable_introspection.py \
  tests/unit/test_cellprofiler_interop_import_records.py \
  tests/unit/test_cppipe_corpus.py::test_in_tree_cppipe_corpus_prepare_expectations -q

PYTHONPATH=external/python-introspect/src .venv/bin/python -m pytest \
  tests/unit/test_artifact_contract_preview.py \
  tests/unit/test_cellprofiler_generated_pipeline_execution.py::test_materialized_generated_pipeline_exports_semantic_contracts_as_python \
  tests/unit/test_cellprofiler_generated_pipeline_execution.py::test_semantic_contract_sidecar_rejects_fingerprint_mismatch \
  tests/unit/test_cellprofiler_generated_pipeline_execution.py::test_materialized_generated_pipeline_contract_sidecar_is_versioned_json -q

nominal-refactor-advisor \
  openhcs/core/artifact_contract_preview.py \
  openhcs/interop/cellprofiler/runtime/generated_pipeline.py \
  openhcs/pyqt_gui/widgets/artifact_contract_preview.py \
  --include-plans
```

Remaining:

- Refactor broader `python-introspect` debt flagged by the advisor:
  docstring-section string dispatch, attribute-probe-heavy callable analysis,
  and the oversized docstring parsing hub.
