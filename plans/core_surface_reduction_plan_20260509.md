# Core Surface Reduction Plan

Date: 2026-05-09
Status: newest architecture plan
Scope: reduce `benchmark` and CellProfiler-specific surface area by moving reusable runtime/compiler semantics into OpenHCS core and leaving thin benchmark/dialect adapters.

## Fact-Checked Snapshot

Checked on 2026-05-09 against the current worktree.

- `openhcs` currently has one production import from `benchmark`: `openhcs/processing/backends/cellprofiler/__init__.py` imports `benchmark.cellprofiler_library`.
- Generated example pipelines currently still import `benchmark.cellprofiler_library.get_function`, so the generated-code import path must be part of the migration, not only hand-written product code.
- `contracts.json` is not accessed only through one API. Converter maintenance scripts and pipeline generation read it directly, so the initial library move needs a broad compatibility shim or a deliberate generator migration in the same slice.
- `benchmark/cellprofiler_compat/module_execution.py` is 10,040 lines and is the largest mixed-responsibility surface.
- `benchmark/cellprofiler_compat/runtime_adapter.py` is 3,902 lines and should be reviewed after `module_execution.py`, but it is less obviously the first extraction point.
- `benchmark/converter/symbol_table.py` is 2,467 lines and still owns generic-looking artifact contract/codegen/lowering concerns.
- `benchmark/converter/module_settings_binding.py` is 2,786 lines and should remain mostly CP dialect, but its generic binding mechanics should be inspected during the compiler move.
- `openhcs/core/runtime_semantics.py` is 3,986 lines, `runtime_slice_projection.py` is 943 lines, and `runtime_artifact_queries.py` is 1,709 lines. These are large but already core-owned; the goal is not to move them out, but to stop CP/benchmark code from duplicating their responsibilities.
- Existing core primitives already cover much of the target design: `RuntimeOutputBundle`, `ArtifactKind`, `ArtifactSpec`, `ModuleArtifactContract`, `RuntimeSliceProjection`, `MeasurementTableRowLayout`, `SourceBindingMatchPlan`, `StepSourceBindingsConfig`, `MaterializationSpec`, `ObjectLabelPayload`, `ObjectLabelSet`, `MeasurementTable`, and `ParentChildRelationshipPayload`.

## Spine

The benchmark platform should not own runtime semantics. It should own benchmark declarations, tool process orchestration, result collection, and reporting. CellProfiler support should not own generic artifact execution, source binding, object-label domain handling, measurement-table projection, materialization, or slice aggregation. It should own only CellProfiler dialect facts: `.cppipe` parsing/lowering, UI literal interpretation, module setting schemas, and CP-compatible function implementations.

The target shape is:

1. `openhcs.core` owns runtime values, artifact contracts, source selection, slice/batch execution, measurement semantics, object-label semantics, and equivalence.
2. `openhcs.processing` owns generic backend/provider dispatch and reusable algorithms.
3. `openhcs.interop.cellprofiler` owns CellProfiler dialect parsing, setting names, module semantics, and lowering into core OpenHCS contracts.
4. `benchmark` owns datasets, manifests, external tool adapters, timing, parity checks, and figure/report generation.
5. `benchmark.cellprofiler_library` either becomes a real OpenHCS CellProfiler backend package or remains an external absorbed-function fixture, but core must not import it from `benchmark`.

## Core Principle

Do not add another registry or compatibility layer unless it removes a boundary violation. The desired migration path is:

1. Find an existing nominal OpenHCS abstraction.
2. Move CP/benchmark-specific data into a dialect declaration when the semantics are CP-specific.
3. Move reusable runtime mechanics into core when the semantics are not CP-specific.
4. Leave benchmark code as declarations and adapters only.

If a fix requires hardcoded string matching, local image extension lists, module-name switch tables, or ad hoc output-shape scoring, it is probably the wrong layer unless it is explicitly modeling CellProfiler UI syntax.

## Current Boundary Violations

### 1. Core imports benchmark code

Evidence:

- `openhcs/processing/backends/cellprofiler/__init__.py` imports `benchmark.cellprofiler_library`.

Why it matters:

This inverts the architecture. A reusable OpenHCS backend cannot depend on benchmark fixtures. It also makes the installed OpenHCS surface depend on a benchmark package name, which is the opposite of making benchmark code thin.

Target:

- Move absorbed CellProfiler function inventory/contract loading to an OpenHCS-owned package, likely `openhcs/interop/cellprofiler/functions` or `openhcs/processing/backends/cellprofiler/functions`.
- Keep `benchmark.cellprofiler_library` as a compatibility import shim only, or delete it after imports migrate.
- `openhcs/processing/backends/cellprofiler/__init__.py` should import only from `openhcs.*`.

Preferred shape:

- `openhcs.processing.backends.cellprofiler.library` owns absorbed function registration, contract JSON loading, and contract coercion.
- `benchmark.cellprofiler_library` re-exports that package temporarily for old tests/scripts.
- The benchmark runner imports OpenHCS product code in the same way users do.
- An import-boundary test fails if any production file under `openhcs/` imports `benchmark`.

### 2. `module_execution.py` is a generic runtime hidden inside CP compatibility

Evidence:

- `benchmark/cellprofiler_compat/module_execution.py` is about 10k lines.
- It owns artifact-kind runtime dispatch, primary image input selection, object-label binding, measurement row projection, pure-2D output aggregation, object-label slice aggregation, returned-output matching, execution-mode policy, main-flow replacement policy, and CP-specific bits.

What belongs in core:

- `RuntimeArtifactKindStrategy`
- `_specs_of_kind`, `_single_spec_of_kind`, `_unique_specs`
- returned output matching against `ArtifactSpec`
- pure-2D tuple/output aggregation
- object-label slice aggregation
- generic runtime invocation lowering through `RuntimeOutputBundle`
- object-label source-binding projection
- relationship endpoint resolution from artifact contracts
- measurement row ownership/completion projection

What remains CP-specific:

- CP module-specific policy leaves
- CP measurement feature-name rendering
- CP source-pair feature projection
- CP special-input module policies
- CP object/image measurement row dialect quirks

Important extraction seam:

- Treat `RuntimeOutputBundle.as_runtime_tuple()` as the generic ABI for multi-output callable returns.
- Treat `ArtifactSpec` and `ModuleArtifactContract` as the authoritative declaration of expected outputs.
- CP may decide what a CP module means, but it should not own the generic algorithm that matches returned values to declared artifact specs.

### 3. Converter owns generic compiler/lowering concepts

Evidence:

- `benchmark/converter/symbol_table.py` owns `CellProfilerSymbol`, `ModuleArtifactContracts`, source-binding literal rendering, module contract builders, inferred contract patterns, and artifact-spec output code generation.
- Existing plans already moved some image schema/source-binding concepts into core; this file still mixes CP workspace symbols, OpenHCS artifact contracts, and generated-Python literal rendering.

What belongs in core or interop:

- Generic `CompiledModuleArtifactContract` / `ModuleArtifactContracts` should live with `ModuleArtifactContract`.
- Artifact-contract builder families should live under `openhcs.interop.cellprofiler` if CP-specific, not under `benchmark.converter`.
- Source-binding literal rendering should be a generic pycodify/codegen concern, not ad hoc converter string construction.
- `CellProfilerSymbolKind` can remain CP dialect, but the output should lower into core artifact contracts as early as possible.

Current smell to remove:

- `benchmark.converter` is acting like a product compiler package. The benchmark can keep command-line wrappers and corpus fixtures, but the compiler itself belongs under `openhcs.interop.cellprofiler`.
- Generated pipelines should import OpenHCS product packages, not benchmark helpers.

### 4. Materialization schema is almost core but still underused

Evidence:

- `openhcs/processing/materialization` has `MaterializationSpec`, options, and presets.
- CP functions still spell many export schemas locally and the benchmark/converter still classifies special outputs by inspecting materialization shapes.

Target:

- Add a core `TabularRowSchema` / `RuntimeTableSchema` abstraction that owns dataclass field extraction, row identity fields, source fields, and export intent.
- `csv_dataclass_materializer` should become one surface over that schema, not the only consumer.
- Converter artifact classification should ask the schema/materialization object for artifact kind instead of reclassifying by local rules.

### 5. Runtime equivalence is doing dialect parsing and generic row semantics together

Evidence:

- `openhcs/core/runtime_equivalence.py` is large and owns row subject resolution, feature canonicalization, source-name extraction, pair-feature directionality, and equivalence policy.
- Some of that is generic measurement semantics; some is CellProfiler dialect-specific naming.

Target:

- Keep generic equivalence pipeline in core.
- Move dialect-specific feature parsing into pluggable `RuntimeMeasurementDialect` families.
- Make CP feature parsing live in `openhcs.interop.cellprofiler.measurement_dialect`, then inject it into core equivalence.
- Core should not know CP strings like `Correlation_*`; it should know pair-feature semantics.

Design note:

- This should be a dialect interface, not a CellProfiler special case. The same core equivalence pipeline should be able to compare Fiji-, napari-, OMERO-, or custom-function-derived tables when a dialect tells it how to parse feature/source semantics.

### 6. Runtime value and slice projection are converging but need cleaner ownership

Evidence:

- `openhcs/core/runtime_values.py`, `runtime_semantics.py`, and `runtime_slice_projection.py` now own many reusable concepts.
- CP compatibility still owns object-label aggregation, relationship slicing, and endpoint coercion in `module_execution.py`.

Target:

- Core owns all runtime value projection and slice aggregation for `ImagePayload`, `ObjectLabelPayload`, `ObjectLabelSet`, `MeasurementTable`, `ParentChildRelationshipPayload`, and `ObjectRelationship`.
- CP compatibility calls core projection APIs only.
- The aggregation API should be backend-generic and arraybridge-aware; CP should not choose NumPy-specific paths except inside CP CPU algorithms.

Existing anchors:

- `runtime_slice_projection.py` already has strategy classes for measurement tables, object-label sets, and object-label payloads.
- `runtime_values.py` already owns `ObjectLabelPayload`, `ObjectLabelSet`, `MeasurementTable`, and relationship payload types.
- `processing.backends.lib_registry.unified_registry` already has pure-2D auxiliary output aggregation machinery. CP-specific batch executors should plug into that shape instead of making a parallel CP-only path.

### 7. Benchmark dataset/source declarations are not yet pure benchmark contracts

Evidence:

- `benchmark/contracts/dataset.py` has good dataset specs, but acquisition/source layout and CP example layout still leak into converter/runtime logic.
- Earlier audit flagged image extension and official CP layout detection as duplicated source semantics.

Target:

- Benchmark dataset declarations should say what dataset/cases exist, where to fetch them, and how to validate their acquired layout.
- OpenHCS source matching/VFS should decide what files are loadable and how sources are matched.
- CP-specific official example layouts should be declared as dataset layout strategies, not hardcoded path checks.

VFS/source rule:

- Dataset acquisition may know archive paths and URLs.
- Source discovery should ask OpenHCS/VFS/source-binding abstractions what files are loadable and how aliases match.
- Benchmark code should not define its own image suffix universe or infer payload roots by scanning filenames unless that logic is packaged as a reusable dataset layout strategy.

## Existing Abstractions To Reuse First

| Existing abstraction | Current owner | Use it for | Do not replace with |
| --- | --- | --- | --- |
| `ArtifactKind`, `ArtifactSpec` | `openhcs.core.artifacts` | Typed artifact declarations and payload category semantics | module-local string categories |
| `ModuleArtifactContract` | `openhcs.core.module_artifact_contract` | Callable/module input-output contracts | converter-local contract records |
| `RuntimeOutputBundle` | `openhcs.core.runtime_invocation` | Nominal multi-output callable returns | CP-only tuple unpacking conventions |
| `RuntimeSliceProjection` | `openhcs.core.runtime_slice_projection` | Projection of runtime values across slices/planes | CP-only slice aggregation helpers |
| `ObjectLabelPayload`, `ObjectLabelSet` | `openhcs.core.runtime_values` | Object-label runtime values and domains | raw ndarray label conventions in CP glue |
| `MeasurementTable`, `MeasurementTableRowLayout` | `openhcs.core.runtime_values`, `openhcs.core.runtime_semantics` | Generic table values and long/wide row semantics | ad hoc row dict inspection |
| `ParentChildRelationshipPayload` | `openhcs.core.runtime_semantics` | Relationship runtime payloads | CP-specific relationship tuple conventions |
| `SourceBindingMatchPlan`, `StepSourceBindingsConfig` | `openhcs.core.source_bindings` | Source alias matching and execution plans | dataset/path heuristics |
| `MaterializationSpec` | `openhcs.processing.materialization` | Declared output materialization | local CSV writer-shape inference |
| `AutoRegisterMeta` strategy families | `openhcs.core` / shared utils | Semantic object registries keyed by nominal type/enum | hand-maintained dicts |

## PyQt UI Integration Notes

Checked on 2026-05-09 against the PyQt GUI path.

The PyQt editor stack already has the right high-level shape for CellProfiler integration:

- `PipelineEditorWidget` owns the pipeline list. It already treats `.cppipe` as an importable pipeline file and calls `get_cellprofiler_dialect_compiler().compile_pipeline(...)`, then stores the result as ordinary `FunctionStep` objects.
- `DualEditorWindow` owns the per-step editing dialog. It creates two tabs: `StepParameterEditorWidget` for step-level settings and `FunctionListEditorWidget` for the function pattern.
- `StepParameterEditorWidget` is generated from `AbstractStep.__init__`/`FunctionStep` state through `ObjectState` and `ParameterFormManager`; it should only expose user-editable step config, not runtime compiler artifacts.
- `FunctionListEditorWidget` and `FunctionPaneWidget` come from `pyqt-reactive`. They normalize callable/list/dict function patterns, keep stable per-function ObjectState tokens, and render each callable's signature as editable parameters.
- `FunctionSelectorDialog` reads `RegistryService.get_all_functions_with_metadata()` and already exposes registry/backend/contract/tags filters. This is the right place for CellProfiler-compatible functions to appear once they are first-class registry functions.

Implication:

- CellProfiler should enter the UI in two ways: importing a `.cppipe` into normal OpenHCS pipeline state, and selecting CellProfiler-compatible processing functions from the normal function selector.
- It should not require a parallel CellProfiler step editor, parallel CellProfiler function pane, or CellProfiler-specific pipeline object model.

What should surface in the UI:

- Imported pipeline provenance: original `.cppipe` path, generated pipeline path, module count, warnings/skipped modules, and source schema summary.
- Function selector metadata: registry/source = CellProfiler-compatible, memory/backend = NumPy/other backend, processing contract, module category/tags, and concise CP module name.
- Read-only artifact contract summary per function or step: inputs, outputs, measurements, relationships, object labels, and materialization intent.
- Validation/provenance messages when a CP module lowered to multiple OpenHCS functions or when a module was pruned as dead/unmaterialized.

What should stay internal:

- `RuntimeOutputBundle`, output matching, slice aggregation, object-label projection internals, relationship payload mechanics, and equivalence dialect parsing.
- These are compiler/runtime contracts. They should appear only as debug/advanced read-only inspection, not as editable UI controls.

Current UI fit risks:

- The function selector sees functions through `RegistryService`. If CellProfiler-compatible wrappers are not registered through the normal registry path in all modes, imported `.cppipe` pipelines will work but manual CP function selection will be incomplete.
- `FunctionPaneWidget` renders callable signatures. Absorbed CP function signatures must stay clean and nominally typed; otherwise CP modules become noisy in the UI.
- Runtime artifact declarations are currently mostly compiler metadata. If the UI needs to show them, add a read-only contract/provenance pane rather than exposing low-level artifact types as editable fields.

Recommended UI-facing work:

1. Add a `.cppipe` import result summary panel or status detail in `PipelineEditorWidget`, backed by `CellProfilerPipelineImportResult`.
2. Ensure CellProfiler-compatible wrappers appear in `FunctionSelectorDialog` through `RegistryService`, with tags/categories that make sense to biologists.
3. Add an optional read-only "Contracts" expander to function panes or the selector detail table, sourced from `CallableContract`/`ArtifactSpec`.
4. Keep typed runtime infrastructure invisible by default; expose it only for debugging, provenance, and validation.

## LLM Integration Notes

Checked on 2026-05-09 against the PyQt LLM service and `pyqt-reactive` integration.

The LLM integration already has the right architectural entry point:

- `LLMPipelineService` builds runtime prompts from the real function registry, not a static handwritten function list.
- The pipeline prompt tells the model to use only registered functions, imports each function from its discovered module path, and emits normal `FunctionStep` code.
- The custom-function prompt already teaches OpenHCS memory decorators, backend-conversion rules, artifact outputs, materialization presets, pyclesperanto, and CuPy/cuCIM examples.
- `register_reactor_providers()` registers `LLMPipelineService`, `OpenHCSCodegenProvider`, and `OpenHCSFunctionRegistry` together, so the code editor, function selector, and LLM assistant are all meant to share the same provider surface.
- `LLMChatPanel` is generic over `code_type` and inserts generated code into the existing editor. This is a good fit for CellProfiler because generated output should remain ordinary OpenHCS pipeline/function code.

Implication:

- Do not create a separate CellProfiler LLM assistant.
- Feed CellProfiler support into the existing registry, compiler, codegen, and provenance surfaces.
- The LLM should be good at CellProfiler module names because CellProfiler is public and well documented, but OpenHCS must still constrain it to actual supported functions/import results.

What CellProfiler should add to LLM context:

- A compact CellProfiler module index generated from the same module semantic declarations used by the compiler.
- Module aliases and familiar CP UI names mapped to OpenHCS callable names.
- Support status per module: supported, partially supported, import-only, benchmark-only, skipped, or unsupported.
- Typed setting schemas with CP display labels, enum values, defaults, and target OpenHCS parameter names.
- Artifact contract summaries: consumed images/objects/measurements, produced images/objects/measurements/relationships, and materialization intent.
- Import provenance summaries for `.cppipe` files: original module list, lowered OpenHCS steps/functions, warnings, skipped modules, and source schema.
- Parity/performance status as read-only metadata when benchmark evidence exists, not as model-inferred claims.

Preferred LLM behavior:

- If the user provides an existing `.cppipe`, import it through `CellProfilerPipelineImportRequest` and explain the lowered OpenHCS pipeline instead of asking the model to recreate the module graph from memory.
- If the user asks for a "CellProfiler-style" workflow, generate normal OpenHCS `FunctionStep` code using CellProfiler-compatible functions discovered from `RegistryService`.
- If the user asks why an imported module changed shape, answer from compiler provenance and artifact contracts.
- If the user asks to optimize a CP-derived pipeline, prefer backend-generic OpenHCS equivalents and preserve declared artifact semantics.
- If support is missing, report the missing CP module/setting from compiler support metadata instead of hallucinating an implementation.

Guardrails:

- The model must not infer file/source matching from path strings; source matching should remain `PipelineImageSchema`/source-binding driven.
- The model must not generate benchmark imports. Generated code should import from `openhcs.*` product packages.
- The model must not add manual NumPy/CuPy/pyclesperanto conversions unless the function contract explicitly requires them.
- The model must not expose `RuntimeOutputBundle`, object-label projection internals, or equivalence dialect internals as editable user code.
- The model should use CellProfiler names as user-facing vocabulary, but generate nominal OpenHCS objects.

Required architecture work:

1. Add an LLM/documentation projection for the CellProfiler semantic registry once that registry is productized.
2. Extend `LLMPipelineService._get_function_documentation()` with provider-supplied documentation fragments instead of hardcoding CellProfiler into the prompt builder.
3. Add a CellProfiler context provider that emits compact module/setting/support/provenance docs from the compiler declarations.
4. Ensure CellProfiler-compatible wrappers are registered with tags/categories that make selector and LLM filtering useful.
5. Add UI affordances that reuse the existing LLM panel: "explain this imported CellProfiler pipeline", "summarize module lowering", and "rewrite this CP-derived step with faster OpenHCS functions".
6. Keep prompt size bounded by grouping CP modules by category and including detailed setting docs only for modules relevant to the current pipeline or user request.

## Ownership Matrix

| Surface | Current responsibility | Target owner | Migration note |
| --- | --- | --- | --- |
| `benchmark.cellprofiler_library` | Absorbed CP functions, inventory, contracts | `openhcs.processing.backends.cellprofiler.library` | First move; benchmark becomes shim |
| `benchmark.cellprofiler_compat.module_execution` | CP policies plus generic runtime execution | Split between `openhcs.core.runtime_*` and `openhcs.interop.cellprofiler.runtime` | Extract generic mechanics before shrinking policy leaves |
| `benchmark.cellprofiler_compat.runtime_adapter` | Adapter from benchmark/native CP world into OpenHCS runtime | Mostly benchmark/interop boundary | Audit after `module_execution`; avoid moving process/timing concerns |
| `benchmark.converter.symbol_table` | CP symbols, artifact contracts, codegen | `openhcs.interop.cellprofiler.compiler` plus core contracts | Move CP symbol dialect with compiler; generic contract types stay core |
| `benchmark.converter.module_settings_binding` | CP setting binding and some generic binding machinery | Mostly `openhcs.interop.cellprofiler.compiler` | Keep setting names CP-specific; abstract reusable binding skeleton only if another dialect pays rent |
| `openhcs.core.runtime_equivalence` | Generic equivalence plus CP-like naming logic | Core engine plus injected dialects | Move CP feature/source parsing into CP dialect |
| `benchmark.contracts.dataset` | Benchmark cases, URLs, expected layouts | Benchmark | Keep, but layout strategies must call core source/VFS APIs |
| `benchmark/figures`, reports | Plotting and lab/paper output | Benchmark | Do not move into OpenHCS |

## What Should Stay Thin

Benchmark should keep:

- Native CellProfiler process execution.
- OpenHCS benchmark process execution.
- Dataset download/cache policy.
- Run manifests and benchmark case selection.
- Timing, memory sampling, raw result storage, parity report assembly, and figures.
- Compatibility shims needed while old imports migrate.

CellProfiler interop should keep:

- `.cppipe` parsing and UI literal interpretation.
- CP module setting declarations.
- CP module-role semantics.
- CP measurement feature-name rendering/parsing.
- CP-specific quirks needed for parity.

OpenHCS core should keep:

- Runtime artifact contracts.
- Runtime value types.
- Source binding/matching.
- Slice and batch projection.
- Generic table/object-label/relationship semantics.
- Equivalence over typed rows, subjects, sources, tolerances, and dialect hooks.

## Target Package Boundaries

### `openhcs.core`

Owns:

- `ArtifactKind`, `ArtifactSpec`, `ModuleArtifactContract`
- runtime artifact input/output matching
- runtime value payloads and schemas
- object-label domains, variants, sparse/dense representation, relationships
- source binding and source matching
- slice projection and pure-2D/batch aggregation
- measurement table schema, row identity, source identity, and generic row queries
- equivalence engine with injected dialects
- generic codegen/literal helpers if needed by generated pipelines

Must not own:

- CellProfiler setting names
- `.cppipe` parser details
- benchmark dataset manifests
- CellProfiler native UI literals except through injected dialect objects

### `openhcs.interop.cellprofiler`

Owns:

- `.cppipe` parser and import records
- CP module roles/categories
- CP setting-name families and literal policies
- CP measurement dialects and feature renderers
- CP module semantic declarations
- CP artifact-contract builders and setting binders, after they are no longer benchmark-only
- CP source schema lowering into core `PipelineImageSchema`

Must not own:

- benchmark timing/parity/figures
- generic source matching
- generic artifact execution
- generic object-label aggregation

### `openhcs.processing.backends.cellprofiler`

Owns:

- reusable CP-compatible algorithm implementations
- backend strategies for CPU/GPU/arraybridge variants
- function inventory and contract declarations if CP functions are first-class OpenHCS processing functions

Must not own:

- benchmark package imports
- `.cppipe` conversion
- parity report logic

### `benchmark`

Owns:

- dataset manifests and acquisition
- tool adapters for native CellProfiler and OpenHCS
- benchmark runner, timing, memory tracking
- parity comparison output
- figure generation and paper/lab-meeting artifacts
- fixture corpora used to validate CP integration

Must not own:

- runtime artifact semantics
- generic materialization schemas
- OpenHCS backend function registry
- CP parser/lowering once it is productized

## Migration Phases

### Phase 1: Stop core from importing benchmark

Goal:

Remove the upward dependency from `openhcs/processing/backends/cellprofiler` into `benchmark.cellprofiler_library`.

Moves:

1. Create an OpenHCS-owned absorbed function package.
2. Move function inventory, contract JSON access, and contract coercion there.
3. Keep benchmark import paths as shims for tests and old scripts.
4. Add an import-boundary test: no `openhcs/**` file may import `benchmark`.
5. Update generated-pipeline import rendering to target the new OpenHCS package.
6. Migrate direct `contracts.json` consumers or route them through the new package API.

Verification:

- `rg "from benchmark|import benchmark" openhcs -g '*.py'` returns no production imports.
- CellProfiler backend function loading tests still pass.
- New import-boundary test blocks regressions.
- Generated pipeline smoke output contains no `benchmark.` imports except in explicit benchmark-only fixtures.

Exit criteria:

- The one known production dependency inversion is gone.
- Benchmark code can still use the old import path through a shim until callers migrate.
- The new OpenHCS-owned package has the same contract inventory API as the old benchmark package.
- Existing generated example pipelines either import the new package or are explicitly marked as stale fixtures.

### Phase 2: Extract generic runtime execution substrate from `module_execution.py`

Goal:

Make CP module execution a thin dialect adapter over core runtime execution.

Core modules to introduce or extend:

- `openhcs/core/runtime_artifact_execution.py`
- `openhcs/core/runtime_output_matching.py`
- `openhcs/core/runtime_batch_aggregation.py`
- `openhcs/core/runtime_object_label_projection.py`

Moves:

1. Move returned-output matching to core and type it against `ArtifactSpec`.
2. Move pure-2D tuple output aggregation to core.
3. Move object-label slice aggregation to core.
4. Make CP result bundles implement `RuntimeOutputBundle`.
5. Replace CP-local helpers with core calls.

Verification:

- CP module execution tests pass unchanged.
- Add non-CP tests for core returned-output matching and pure-2D aggregation.

Suggested PR slices:

1. Extract returned-output matching into `openhcs.core.runtime_output_matching`.
2. Extract pure-2D auxiliary aggregation into `openhcs.core.runtime_batch_aggregation`, reusing existing lib-registry aggregation concepts.
3. Extract object-label slice aggregation into `openhcs.core.runtime_object_label_projection`.
4. Convert CP result wrapper classes to implement or lower through `RuntimeOutputBundle`.

Do not do:

- Do not add a CP-only batch executor registry if the existing runtime/function registry can express the behavior.
- Do not introduce a dict keyed by CP module names for generic output matching.

### Phase 3: Promote measurement-row schema and materialization contracts

Goal:

Stop treating tabular output fields as local CSV/materializer details.

Moves:

1. Add `RuntimeTableSchema` or `MeasurementRowSchema` in core.
2. Let dataclass row types lower into that schema.
3. Make `MaterializationSpec` carry or reference row schema where applicable.
4. Move object/image/source row identity handling into schema objects.
5. Make CP row policies render CP names but rely on generic row ownership/completion.

Verification:

- Existing CSV materialization tests pass.
- CP measurement row projection tests pass.
- Converter artifact classification no longer inspects materialization internals ad hoc.

Suggested shape:

- `RuntimeTableSchema`: declares row layout, identity fields, feature fields, source fields, and required/completed fields.
- `RuntimeRowIdentity`: declares image/object/relationship subject identity independent of CP column names.
- `RuntimeFeatureField`: declares numeric/string feature payload semantics.
- CP dialect maps those schema concepts to CP column names.
- Materializers consume the schema; they do not define the schema.

Payoff:

- CSV, parquet, in-memory comparison, and figures can all consume the same table contract.
- Equivalence no longer needs to reverse-engineer intent from column strings.

### Phase 4: Move CP converter from `benchmark.converter` to `openhcs.interop.cellprofiler`

Goal:

Benchmark should call the product converter, not own it.

Moves:

1. Move parser wrappers, setting binders, module function resolution, symbol table, source schema lowering, and pipeline generator into `openhcs.interop.cellprofiler.compiler`.
2. Keep `benchmark.converter` as CLI/import compatibility wrappers.
3. Replace converter-local code literal rendering with pycodify or core codegen helpers.
4. Ensure generated pipelines import from OpenHCS and CP backend packages, not benchmark.

Verification:

- Existing converter tests pass through compatibility shims.
- New tests import the real compiler from `openhcs.interop.cellprofiler.compiler`.

Package target:

- `openhcs.interop.cellprofiler.compiler.symbols`
- `openhcs.interop.cellprofiler.compiler.settings`
- `openhcs.interop.cellprofiler.compiler.contracts`
- `openhcs.interop.cellprofiler.compiler.pipeline_generator`
- `benchmark.converter.*` as temporary re-export/CLI wrappers only.

Guardrail:

- The compiler may import OpenHCS core and CP interop.
- The compiler must not import benchmark manifests, benchmark adapters, timing, figures, or result directories.

### Phase 5: Split CP dialect semantics from generic equivalence

Goal:

Core equivalence works for CP, Fiji, napari-derived outputs, OMERO-backed runs, and future dialects.

Moves:

1. Make all feature-name canonicalization strategy-driven through `RuntimeMeasurementDialect`.
2. Move CP pair-feature and source-feature naming into CP dialect modules.
3. Keep core equivalence over typed subjects, features, source pairs, numeric tolerances, and row schemas.

Verification:

- CP parity tests pass.
- Add dialect-level tests for CP feature parsing without running equivalence.

Suggested interface:

- `RuntimeMeasurementDialect.subject_for_row(row, schema)`
- `RuntimeMeasurementDialect.feature_for_column(column, schema)`
- `RuntimeMeasurementDialect.source_pair_for_feature(feature_name)`
- `RuntimeMeasurementDialect.canonical_feature_key(feature)`

The exact names can change, but the ownership should not: CP strings in CP dialect, generic comparison in core.

### Phase 6: Make dataset/source layout declarations declarative

Goal:

Benchmark manifests declare datasets; OpenHCS source infrastructure discovers and matches files.

Moves:

1. Add dataset layout strategies for archive-root, official CP examples, and explicit cppipe/data pair layouts.
2. Use OpenHCS source matching/loadable-file semantics instead of local image extension sets.
3. Keep VFS boundary explicit in acquisition and adapter code.

Verification:

- Dataset acquisition can prepare all declared benchmark cases without converter-local path heuristics.
- Unit tests cover layout strategies on synthetic directory trees.

Suggested dataset layout strategies:

- `ArchiveRootLayout`: payload is already at the extracted root.
- `OfficialCellProfilerExampleLayout`: payload root is declared by the example manifest, not inferred by suffix heuristics.
- `CppipeAdjacentDataLayout`: `.cppipe` and data live under a known relative pair.
- `ExplicitSourceAliasLayout`: manifest maps aliases to source roots directly.

These strategies belong to benchmark acquisition/source declaration code, but they should call OpenHCS source matching and VFS primitives.

## Concrete Extraction Queue

This is the queue another agent can safely consume. The lanes are ordered by architectural leverage and by how likely they are to create conflicts.

### Lane A: Dependency inversion guardrail

Write scope:

- `openhcs/processing/backends/cellprofiler/**`
- `benchmark/cellprofiler_library/**`
- generated import-rendering code in `benchmark/converter/**`
- import-boundary tests

Tasks:

1. Move the absorbed CellProfiler function inventory and contract loading into an OpenHCS-owned backend package.
2. Keep `benchmark.cellprofiler_library` as a re-export shim during migration.
3. Update generated pipeline imports to use the OpenHCS-owned package.
4. Add a test that fails on any production `openhcs` import from `benchmark`.

Validation:

- `rg -n "from benchmark|import benchmark" openhcs -g '*.py'` returns nothing.
- Existing CellProfiler backend function loading tests pass.
- At least one regenerated pipeline imports the new package.

Conflict risk:

- Medium. This touches import paths that runtime/compiler work may also touch.

### Lane B: Runtime output matching

Write scope:

- new or existing `openhcs/core/runtime_output_matching.py`
- `benchmark/cellprofiler_compat/module_execution.py`
- focused tests for output matching

Tasks:

1. Extract generic returned-output matching from CP execution code.
2. Type it against `ArtifactSpec`, `ArtifactKind`, and `ModuleArtifactContract`.
3. Lower any CP multi-output result wrappers through `RuntimeOutputBundle`.
4. Leave CP module-name policy decisions in CP code.

Validation:

- Core unit tests prove matching works without importing CP.
- CP module execution tests pass unchanged.

Conflict risk:

- High if another agent is editing `module_execution.py`.

### Lane C: Slice and batch aggregation

Write scope:

- `openhcs/core/runtime_slice_projection.py`
- `openhcs/core/runtime_values.py`
- `openhcs/processing/backends/lib_registry/unified_registry.py`
- `benchmark/cellprofiler_compat/module_execution.py`

Tasks:

1. Move pure-2D auxiliary aggregation to the existing backend/function aggregation system where possible.
2. Move object-label slice aggregation into core runtime projection/value strategies.
3. Keep array handling backend-aware; do not add NumPy-only core paths.
4. Make CP execution call the generic aggregator instead of owning a parallel path.

Validation:

- Object-label, relationship, and measurement projection tests pass.
- CP parity tests for pipelines using 2D/3D aggregation remain green.

Conflict risk:

- High if another agent is working on parity fixes in object labels or relationships.

### Lane D: Measurement schema and materialization

Write scope:

- `openhcs/core/runtime_semantics.py`
- `openhcs/core/runtime_values.py`
- `openhcs/processing/materialization/**`
- CP row-policy code in `benchmark/cellprofiler_compat/module_execution.py`

Tasks:

1. Promote row identity/source/feature semantics into a core table schema.
2. Let `MaterializationSpec` reference or expose schema information for tabular outputs.
3. Move CP column rendering/parsing to a CP dialect layer.
4. Stop converter/runtime code from inferring artifact meaning from CSV writer internals.

Validation:

- CSV materialization tests pass.
- Equivalence tests still pass.
- New tests verify schema can be consumed without CP naming.

Conflict risk:

- Medium-high. This is broad and should start after Lane B unless another agent is not touching runtime tables.

### Lane E: Compiler package migration

Write scope:

- `benchmark/converter/**`
- `openhcs/interop/cellprofiler/**`
- generated pipeline tests

Tasks:

1. Move product compiler code under `openhcs.interop.cellprofiler.compiler`.
2. Keep benchmark converter modules as thin re-export/CLI wrappers.
3. Move source-binding and artifact-contract lowering to product interop/core boundaries.
4. Replace converter-local literal rendering with pycodify or a core codegen helper where that already exists.

Validation:

- Old benchmark import paths still work through shims.
- New product import paths are tested directly.
- Generated pipelines do not depend on benchmark internals.

Conflict risk:

- Medium. This is mostly separate from runtime execution but touches generated imports from Lane A.

### Lane F: Dataset/source layout cleanup

Write scope:

- `benchmark/contracts/**`
- `benchmark/datasets/**`
- benchmark acquisition/setup helpers
- source/VFS integration points

Tasks:

1. Represent official examples, archive roots, adjacent cppipe/data pairs, and explicit alias maps as dataset layout strategy objects.
2. Remove image suffix/path-root heuristics from benchmark setup code.
3. Route loadability and source matching through OpenHCS source/VFS abstractions.

Validation:

- Synthetic directory-layout tests pass.
- Existing benchmark dataset setup still prepares all declared cases.

Conflict risk:

- Low unless another agent is adding datasets.

### Highest priority

1. `openhcs/processing/backends/cellprofiler/__init__.py` dependency inversion.
2. Runtime output matching and `RuntimeOutputBundle` lowering from `module_execution.py` to core.
3. Pure-2D/batch output aggregation from `module_execution.py` to core.
4. Object-label slice aggregation from `module_execution.py` to core.
5. CP converter package move from `benchmark.converter` to `openhcs.interop.cellprofiler.compiler`.

### First two commits should be boring

1. Move absorbed function inventory out of `benchmark` and keep an import shim.
2. Add the import-boundary test.

Reason:

This creates a hard architectural guardrail before deeper extraction begins. It is also low risk because it should be mostly file movement and import rewriting, not semantic changes.

### Medium priority

1. Measurement row schema ownership in core.
2. CP measurement feature rendering as dialect-owned semantics.
3. Source/dataset layout strategies.
4. Generic module-policy metaclass/helper in core if more dialects use module-name keyed families.
5. Code literal rendering through pycodify/core codegen instead of converter-local string functions.

### Defer

1. Cosmetic tuple/sorted helper cleanup.
2. Viewer display abbreviations unless working in viewer code.
3. Benchmark graph/report refactors unless they block lab/paper output.

## Validation Gates

Run these after each extraction slice when practical:

- Import boundary: `rg -n "from benchmark|import benchmark" openhcs -g '*.py'`.
- Generated pipeline boundary: inspect generated OpenHCS pipelines for `benchmark.` imports.
- Focused backend loading tests for CellProfiler function inventory.
- Focused `benchmark/cellprofiler_compat` tests for runtime output matching, object labels, relationships, and measurement rows.
- Focused converter tests through both new product import paths and old benchmark shim paths.
- Full CP parity benchmark only after behavior-changing slices, not after pure documentation or import-shim edits.
- Nominal refactor advisor folder scans on touched packages, using the latest advisor checkout.

Add these tests if missing:

- `test_openhcs_does_not_import_benchmark`.
- `test_generated_pipeline_does_not_import_benchmark`.
- `test_runtime_output_bundle_matching_is_not_cp_specific`.
- `test_materialization_schema_declares_tabular_artifact_kind`.
- `test_cp_measurement_dialect_parses_pair_features_without_core_cp_strings`.

## Risk Register

| Risk | Why it matters | Mitigation |
| --- | --- | --- |
| Circular imports after moving CP library into OpenHCS | Backend package may import interop/compiler and vice versa | Keep function inventory under processing backend; compiler depends on contracts, not runtime backend initialization |
| Compatibility shims become permanent | Old benchmark imports hide the real boundary | Add removal issue/date and keep shims small/re-export-only |
| Runtime extraction changes parity | `module_execution.py` encodes many CP edge cases | Extract generic algorithms with black-box tests before changing CP policy leaves |
| Generic abstractions become fake generic registries | Renaming CP dicts does not reduce surface area | Require at least one non-CP or core-level test for every new generic API |
| Dataset layouts recreate VFS badly | Hardcoded suffix/path logic already caused pushback | Route all loadable-file and alias matching through source/VFS abstractions |
| Other agent edits conflict | Same repo is being modified concurrently | Only edit this plan during planning passes; before code edits, re-read touched files and `git status` |

## Open Questions To Fact Check During Implementation

- Should absorbed CP functions live in `openhcs.processing.backends.cellprofiler.library` or `openhcs.interop.cellprofiler.functions`? Current usage by `openhcs.processing.backends.cellprofiler.__init__` points toward backend ownership, while compiler contract generation may need a small interop-facing API.
- Is `benchmark.cellprofiler_library/contracts.json` currently consumed only through the function inventory API, or do converter paths read it directly? The answer determines how broad the shim must be.
- Answer from this pass: converter maintenance/generation code reads `contracts.json` directly in several places, including `pipeline_generator.py`, `add_parameter_mappings.py`, `backfill_parameter_mappings.py`, `recategorize_functions.py`, and `fix_registry.py`.
- Which `module_execution.py` policy classes are truly CP-specific and which are generic artifact/source policies with CP names?
- Can `MaterializationSpec.tabular_field_names()` be extended to return a schema object without breaking existing CSV materializer callers?
- Does generated pipeline code already have a central import renderer that can be switched once, or are imports scattered through `symbol_table.py` and module binders?

## Fact Check Log

- `rg -n "from benchmark|import benchmark" openhcs -g '*.py'` found only `openhcs/processing/backends/cellprofiler/__init__.py`.
- Broader `rg` found generated example pipelines and converter code still import `benchmark.cellprofiler_library`; those are acceptable as current benchmark-surface facts but must be removed or shimmed during productization.
- `rg` over core abstractions confirmed existing nominal types for artifacts, runtime invocation bundles, source bindings, runtime slice projection, object labels, measurement tables, relationships, and materialization.
- `wc -l` confirmed the largest mixed surfaces: `module_execution.py` at 10,040 lines, `runtime_adapter.py` at 3,902 lines, `symbol_table.py` at 2,467 lines, and `module_settings_binding.py` at 2,786 lines.
- `MaterializationSpec` already validates writer option types and exposes `tabular_field_names()`, making it a plausible bridge for row schema promotion rather than a thing to replace.
- `SourceBindingMatchPlan` and related source binding types already model cross-alias matching, so benchmark dataset layout logic should reuse them instead of suffix/path heuristics.

## Anti-Goals

- Do not move benchmark timing, native CellProfiler process execution, or paper figures into OpenHCS core.
- Do not move CP UI literals into core enums.
- Do not create giant generic registries that merely rename CP-specific tables.
- Do not preserve compatibility shims indefinitely once imports are migrated.
- Do not make NumPy the core runtime array abstraction; generic core APIs should operate through existing memory/arraybridge mechanisms.

## Definition of Done

The refactor is successful when:

1. `openhcs` has zero production imports from `benchmark`.
2. Generated CP pipelines do not import from `benchmark`.
3. `benchmark/cellprofiler_compat/module_execution.py` is reduced to CP dialect policies and adapter glue, not generic runtime mechanics.
4. `benchmark/converter` is only a compatibility wrapper around `openhcs.interop.cellprofiler.compiler`.
5. CP parity and benchmark code exercise the same OpenHCS core APIs that Fiji/napari/OMERO/future integrations can use.
6. The benchmark package can be deleted or excluded without removing CellProfiler support from OpenHCS itself.
