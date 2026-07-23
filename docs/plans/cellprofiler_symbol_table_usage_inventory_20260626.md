# CellProfiler Symbol Table Usage Inventory

Date: 2026-06-26

Purpose: identify every current dependency on `openhcs.interop.cellprofiler.symbol_table` so the table can be replaced by a generic artifact-flow compiler that queries `CellProfilerModule` declarations.

## Architectural Target

The symbol table currently mixes two concerns:

- pipeline-wide artifact environment: prior producer lookup, source binding lookup, and validation that a referenced image/object/measurement exists;
- module-specific artifact semantics: which CP settings are inputs, outputs, sidecars, measurements, relationships, and retained artifacts.

Only the first concern is generic compiler state. The second belongs on AutoRegisterMeta-backed module declaration classes and inherited nominal policy families.

Replacement direction:

- module declarations expose typed artifact requirements and outputs, preferably using existing `ArtifactSpec`, `ArtifactKind`, `ArtifactSidecarRole`, and inherited declaration methods;
- common declaration families handle repeated patterns such as image-to-image, object-to-object, measurement-producing, relationship-producing, retained-image-output, and source-artifact input modules;
- the compiler performs one generic pass over ordered modules: query the module declaration, resolve declared inputs against prior producers or source schema, declare outputs, attach source bindings, and produce runtime `ModuleArtifactContract` values;
- no parallel `ModuleContractBuilder` registry and no symbol-table-owned module-specific setting catalog.

## Production Imports

| File | Imports from `symbol_table` | Current use | Swap-out target |
|---|---|---|---|
| `openhcs/interop/cellprofiler/pipeline_generator.py` | `CellProfilerSymbolTable`, `ModuleArtifactContracts`, `module_contract_literal` | Compiles ordered modules, fetches contracts by module number, prunes dead artifact steps, computes terminal images, projects runtime `ModuleArtifactContract`, emits semantic sidecar literals, carries `source_schema`. | Replace with a generic `CellProfilerArtifactFlowCompiler` result: `source_schema`, contracts by module num, and runtime contracts. Literal rendering should move to sidecar/persistence code or disappear if runtime contracts are stored directly. |
| `openhcs/interop/cellprofiler/module_processing_components.py` | `ModuleArtifactContracts`, `source_bindings_literal` | Uses contract inputs/outputs/runtime inputs/source bindings to choose variable components, group-by, pairwise object scope, and generated `source_bindings=` lines. Traverses `input_symbols[*].producer_module_num` for lineage. | New compiler contract must preserve producer metadata separately from module-specific semantics. `source_bindings_literal` belongs with source-binding serialization, not symbol table. |
| `openhcs/interop/cellprofiler/runtime/generated_pipeline.py` | `ModuleArtifactContracts` | Persists and reloads semantic contracts in generated Python sidecars; also injects semantic contracts into loaded generated modules. Runtime execution mainly uses `ModuleArtifactContract`. | Replace semantic sidecar with new generic artifact-flow DTO or remove once runtime contracts are enough. Generated legacy sidecars may need one migration shim. |
| `openhcs/interop/cellprofiler/import_records.py` | `ModuleArtifactContracts` | Stores `semantic_contracts` on `CellProfilerPipelineImportResult` and validates tuple member type. | Store runtime `ModuleArtifactContract` plus optional new artifact-flow debug record, not symbol-table contracts. |
| `openhcs/interop/cellprofiler/module_roles.py` | `ModuleArtifactContracts` under `TYPE_CHECKING` | Infrastructure modules ask for `contracts_by_module_num` to decide which artifacts remain live, especially skipped SaveImages/export-style modules. | Keep generic compiler contract map, but retained-artifact behavior belongs on the infrastructure module declaration. |
| `openhcs/interop/cellprofiler/module_settings_binding.py` | `INPUT_IMAGE_SETTING`, `INPUT_OBJECTS_SETTING`, `OUTPUT_IMAGE_SETTING`, `OUTPUT_OBJECTS_SETTING`, `IDENTIFY_PRIMARY_OUTPUT_OBJECTS_SETTING` | Uses symbol-table setting-name constants while deciding setting coverage / ignored artifact settings. | Move common CP setting-name families to module declarations or a neutral setting-name module. Do not keep them on artifact compiler state. |
| `openhcs/processing/backends/cellprofiler/module_classes.py` | `ModuleArtifactContracts` under `TYPE_CHECKING`; imports `INPUT_IMAGE_SETTING`, `INPUT_OBJECTS_SETTING`, `OUTPUT_IMAGE_SETTING` inside `OverlayObjectsModule.ignored_settings_for` | Type annotations for infrastructure retained artifacts and a concrete module declaration reaching into symbol-table constants. | Type annotations should point at the generic compiler contract map or core `ModuleArtifactContract`. OverlayObjects setting names should be class attributes or inherited common names. |
| `openhcs/agent/services/architecture_projection_service.py` | `CellProfilerSymbolTable` | MCP/agent architecture exposition lists symbol table as an internal symbol and describes it as validating CP symbol flow. | Update exposition to the new artifact-flow compiler and declaration-owned artifact semantics. |
| `benchmark/converter/__init__.py` | `CellProfilerSymbol`, `CellProfilerSymbolKind`, `CellProfilerSymbolTable`, `ModuleArtifactContracts` | Compatibility reexports. | Remove or reexport replacement compiler/result types after downstream imports are updated. |
| `benchmark/converter/compatibility_matrix.py` | `ModuleContractBuilder` | Reports artifact-contract coverage by checking `ModuleContractBuilder.__registry__`. | Coverage should query module declaration capability or inherited artifact-flow family, not a second registry. |

## Generated / Persistent Artifacts

Generated files under `benchmark/results/napari_streaming_validator/**` import `CellProfilerSymbol`, `CellProfilerSymbolKind`, and `ModuleArtifactContracts` in semantic contract sidecars.

Replacement options:

- regenerate these artifacts after the new compiler result exists;
- keep a narrow loader migration for old sidecars only, isolated from production generation;
- prefer not to preserve `CellProfilerSymbol*` as public compatibility types.

## Test Imports

| File | Current dependency | Purpose |
|---|---|---|
| `tests/unit/test_cellprofiler_symbol_table.py` | `CellProfilerSymbolKind`, `CellProfilerSymbolTable` | Direct tests for symbol table compilation, symbol kinds, producer linkage, contracts, sidecars, and error cases. |
| `tests/unit/test_cellprofiler_source_schema.py` | `CellProfilerSymbolTable` | Integration checks that setup/source schema data reaches contract/source-binding behavior. |
| `tests/unit/test_cellprofiler_interop_import_records.py` | `ModuleArtifactContracts` | Import-record type validation. |

These tests become migration acceptance tests for the replacement compiler. Most assertions should shift from symbol-table names to:

- typed module declaration artifact specs;
- compiled producer/source-binding metadata;
- runtime `ModuleArtifactContract` projection;
- pipeline generation and execution behavior.

## Raw Query Log

This section is the mechanical inventory requested before rewriting the symbol-table path. It records current callers of `openhcs.interop.cellprofiler.symbol_table`; generated benchmark sidecars are separated from source files because they should be regenerated or migrated, not treated as production semantic owners.

### Production Source

| File | Imported names | Use lines |
|---|---|---|
| `openhcs/agent/services/architecture_projection_service.py` | `CellProfilerSymbolTable` at L206 | L226 |
| `openhcs/interop/cellprofiler/import_records.py` | `ModuleArtifactContracts` at L12 | L104, L147 |
| `openhcs/interop/cellprofiler/module_processing_components.py` | `ModuleArtifactContracts` at L40; `source_bindings_literal` at L41 | `ModuleArtifactContracts`: L517, L908, L1107, L1123, L1126, L1132, L1140, L1146, L1177, L1209. `source_bindings_literal`: L1114 |
| `openhcs/interop/cellprofiler/module_roles.py` | `ModuleArtifactContracts` at L24 under `TYPE_CHECKING` | L119, L161 |
| `openhcs/interop/cellprofiler/module_settings_binding.py` | `IDENTIFY_PRIMARY_OUTPUT_OBJECTS_SETTING` at L83; `INPUT_IMAGE_SETTING` at L84; `INPUT_OBJECTS_SETTING` at L85; `OUTPUT_IMAGE_SETTING` at L86; `OUTPUT_OBJECTS_SETTING` at L87 | `INPUT_IMAGE_SETTING`: L1164, L2034. `IDENTIFY_PRIMARY_OUTPUT_OBJECTS_SETTING`: L1165. `INPUT_OBJECTS_SETTING`: L2035. `OUTPUT_IMAGE_SETTING`: L2036. No explicit use found for `OUTPUT_OBJECTS_SETTING` after import |
| `openhcs/interop/cellprofiler/pipeline_generator.py` | `CellProfilerSymbolTable` at L70; `ModuleArtifactContracts` at L71; `module_contract_literal` at L72 | `ModuleArtifactContracts`: L150, L168, L399, L474, L490, L513, L548, L571, L640, L658, L662, L672, L675, L688, L691, L703, L706, L719, L722, L738, L760, L969. `module_contract_literal`: L891. `CellProfilerSymbolTable`: L1018. Direct local symbol-table result APIs: `contract_for` L1020, L1024; `contracts_by_module_num` L1033; `source_schema` L1109, L1137 |
| `openhcs/interop/cellprofiler/runtime/generated_pipeline.py` | `ModuleArtifactContracts` at L43 | L277, L286, L317, L355, L438, L459, L1100, L1125 |
| `openhcs/processing/backends/cellprofiler/module_classes.py` | `ModuleArtifactContracts` at L52 under `TYPE_CHECKING`; local `INPUT_IMAGE_SETTING` at L3571; local `INPUT_OBJECTS_SETTING` at L3572; local `OUTPUT_IMAGE_SETTING` at L3573 | `ModuleArtifactContracts`: L333, L4070. `INPUT_IMAGE_SETTING`: L3577. `INPUT_OBJECTS_SETTING`: L3578. `OUTPUT_IMAGE_SETTING`: L3579 |

### Benchmark Source

| File | Imported names | Use lines |
|---|---|---|
| `benchmark/converter/__init__.py` | `CellProfilerSymbol` at L20; `CellProfilerSymbolKind` at L21; `CellProfilerSymbolTable` at L22; `ModuleArtifactContracts` at L23 | No direct per-name references after import; exported through `globals()` / `__all__` construction at L74-L78 |
| `benchmark/converter/compatibility_matrix.py` | `ModuleContractBuilder` at L50 | L640 checks `ModuleContractBuilder.__registry__` |

### Generated Benchmark Artifacts

All generated hits are under `benchmark/results/napari_streaming_validator/**` and import `CellProfilerSymbol`, `CellProfilerSymbolKind`, and `ModuleArtifactContracts` from `symbol_table` at import lines L16-L20.

| File | `ModuleArtifactContracts(` | `CellProfilerSymbol(` | `CellProfilerSymbolKind.` |
|---|---:|---:|---:|
| `benchmark/results/napari_streaming_validator/20260624_204718/ExampleColocalization/benchmark_generated_generated_pipeline_fd0343a28e63.cellprofiler_semantic_contracts.py` | 18 | 101 | 101 |
| `benchmark/results/napari_streaming_validator/20260624_205319/ExampleCometAssay/benchmark_generated_generated_pipeline_acfa1b7d224e.cellprofiler_semantic_contracts.py` | 10 | 51 | 51 |
| `benchmark/results/napari_streaming_validator/20260624_225715/cp_tutorial_3d_monolayer/benchmark_generated_generated_pipeline_11488036d8a2.cellprofiler_semantic_contracts.py` | 28 | 117 | 117 |
| `benchmark/results/napari_streaming_validator/20260624_225745/cp_tutorial_3d_monolayer/benchmark_generated_generated_pipeline_307884f50d29.cellprofiler_semantic_contracts.py` | 28 | 117 | 117 |

Totals: 4 files, 84 `ModuleArtifactContracts(` uses, 386 `CellProfilerSymbol(` uses, and 386 `CellProfilerSymbolKind.` uses.

### Direct Test Imports

| File | Imported names | Use lines |
|---|---|---|
| `tests/unit/test_cellprofiler_interop_import_records.py` | `ModuleArtifactContracts` at L11 | L71 constructs `semantic_contracts=(ModuleArtifactContracts(...),)`; L78 asserts semantic contract data |
| `tests/unit/test_cellprofiler_source_schema.py` | `CellProfilerSymbolTable` at L12 | L776, L860, L989 call `CellProfilerSymbolTable.compile(...)` |
| `tests/unit/test_cellprofiler_symbol_table.py` | `CellProfilerSymbolKind` at L16; `CellProfilerSymbolTable` at L17 | L158, L160, L161, L164, L167, L168, L171, L174, L175, L179, L181, L247, L307, L331, L359, L390, L395, L419, L450, L513, L575, L631, L682, L736, L763, L784, L786, L815, L817, L818, L844, L877, L1060, L1100, L1147, L1178, L1210, L1244, L1291, L1503, L1562, L1601, L1626, L1663, L1724, L1758, L1776, L1802, L1820, L1877, L1906, L1939, L1975, L2089, L2147, L2192 |

No `tests/**/*.py` file imports `openhcs.interop.cellprofiler.symbol_table` as a module alias or calls direct `symbol_table.*` APIs.

### Indirect Test Surfaces

These tests do not import `symbol_table`, but they depend on generated/imported semantic-contract objects that currently come from `ModuleArtifactContracts`.

| File | Lines | Dependency surface |
|---|---|---|
| `tests/unit/test_cellprofiler_generated_pipeline_execution.py` | L452, L461, L469, L481, L489, L497, L499 | Fingerprinting, materialization, sidecar restore, and mismatch tests use `generated.artifact_contracts` / `semantic_contracts` |
| `tests/unit/test_cppipe_corpus.py` | L138-L140 | Asserts `import_result.semantic_contracts == prepared.generated_pipeline.artifact_contracts` |

## Internal APIs To Replace

`CellProfilerSymbolTable`

- Current responsibilities: `compile(modules)`, `contract_for(module)`, `contracts_by_module_num`, `symbol_for(...)`, `source_schema`.
- Replacement: compiler result with `source_schema`, contract map, and optional debug lookup. `symbol_for` is test/debug-only and should not drive production.

`ModuleArtifactContracts`

- Current responsibilities: stores `CellProfilerSymbol` inputs/outputs with producer/source-bound flags, projects to core `ModuleArtifactContract`, carries `source_bindings`.
- Replacement: separate generic compiled artifact-flow record that contains core `ArtifactSpec`s plus producer/source-binding provenance. Runtime should consume core `ModuleArtifactContract`.

`CellProfilerSymbol` and `CellProfilerSymbolKind`

- Current responsibilities: CellProfiler string-name plus kind plus producer/source-bound metadata.
- Replacement: use `ArtifactSpec`/`ArtifactKind` for declared module facts, and a compiler-owned provenance record for producer/source-bound metadata.

`ModuleContractBuilder` and `InferredModuleContractPattern`

- Current responsibilities: second registry of per-module and fallback artifact semantics.
- Replacement: module declaration inheritance. Generic fallback patterns can become inherited declaration families or a compiler fallback for truly conventional one-input/one-output modules.

`module_contract_literal`

- Current responsibility: pycodify serializer for semantic contracts.
- Replacement: move serialization to generated-pipeline sidecar code, and serialize the new compiler result only if still needed.

`source_bindings_literal`

- Current responsibility: generated-code serializer for `StepSourceBindingsConfig`.
- Replacement: source-binding serialization helper outside the artifact compiler.

`SymbolTableSettingNameCatalog` and exported setting constants

- Current responsibility: common setting-name families used by artifact builders and settings binder.
- Replacement: common setting-name declarations on module classes or neutral setting-name families. Module-specific constants must not live on the compiler.

## Replacement Work Order

1. Define the declaration-facing contract using existing core types: module declarations return typed input/output `ArtifactSpec` data and inherited policy markers.
2. Build a generic artifact-flow compiler that resolves those specs against prior producers and `PipelineImageSchema`.
3. Move source-binding derivation and producer lineage into that compiler result.
4. Rewire `pipeline_generator.py` to consume the new compiler result instead of `CellProfilerSymbolTable.compile`.
5. Rewire `module_processing_components.py` to consume the new compiler result/provenance, not `CellProfilerSymbol`.
6. Replace semantic sidecar persistence in `runtime/generated_pipeline.py`.
7. Move setting-name constants out of `symbol_table.py`.
8. Delete `ModuleContractBuilder`, `InferredModuleContractPattern`, `CellProfilerSymbolTable`, `CellProfilerSymbol`, and `CellProfilerSymbolKind` once tests and generated artifacts are migrated.

## Scan Commands Used

```bash
rg -n "from openhcs\\.interop\\.cellprofiler\\.symbol_table import|import openhcs\\.interop\\.cellprofiler\\.symbol_table" openhcs tests benchmark -g '*.py'
rg -n "CellProfilerSymbolTable\\.compile|\\.contract_for\\(|contracts_by_module_num|module_contracts|source_bindings_literal\\(" openhcs tests benchmark -g '*.py'
rg -n "symbol table|symbol-table|SymbolTable|ModuleArtifactContracts" docs openhcs tests benchmark -g '*.md' -g '*.rst' -g '*.py'
```
