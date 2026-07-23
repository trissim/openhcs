# CellProfiler Production Boilerplate Inventory

Date: 2026-06-26

Purpose: inventory remaining CellProfiler Python production boilerplate that mirrors module semantics outside AutoRegisterMeta-backed `CellProfilerModule` declarations. This is meant to feed the next refactor pass and, where mechanical, NominalRefactorAdvisor DSL recipes.

## Scope

Production scope:

- `openhcs/interop/cellprofiler/**/*.py`
- `openhcs/processing/backends/cellprofiler/*.py`

Excluded from this file:

- tests, benchmark generated artifacts, and external CellProfiler source;
- full symbol-table callsite details, already recorded in `docs/plans/cellprofiler_symbol_table_usage_inventory_20260626.md`;
- primary module-execution and adapter policy details, already recorded in `docs/plans/cellprofiler_runtime_declaration_semantics_inventory_20260626.md`.

The target rule remains:

```text
CellProfiler module declarations own module-specific facts.
Compiler/generator/runtime/adapter code query those declarations.
Registries and catalogs are derived views or generic execution mechanisms.
```

## Classification Rules

`Move to declaration`

- module-name keyed tables or policy leaves;
- CP setting labels/defaults/aliases tied to a module;
- function-resolution rules tied to a module;
- artifact input/output/special-output facts tied to a module;
- measurement row schema/ownership tied to a module;
- output provenance/main-flow semantics tied to a module.

`Keep as generic executor`

- parser mechanics;
- source-binding resolution mechanics;
- artifact-kind dispatch that is not module-name keyed;
- algorithm/backend strategy dispatch inside one callable implementation;
- profiling, cache, serialization, and persistence mechanics that do not declare module-specific rules.

`Already declaration-owned`

- helpers whose only job is to query `CellProfilerModule.for_module(...).<fact>(...)`.

## Companion Inventories

| Inventory | What it covers |
|---|---|
| `cellprofiler_symbol_table_usage_inventory_20260626.md` | `symbol_table.py` import/callsite ledger, generated semantic sidecars, tests, and replacement work order. |
| `cellprofiler_runtime_declaration_semantics_inventory_20260626.md` | `module_execution.py`, `adapter.py`, object-input policies, measurement-row policies, output context, main-flow, and runtime declaration semantics. |
| This file | Remaining production boilerplate: non-runtime interop, residual runtime policy surfaces, backend declaration/wrapper mirrors, and NRA automation shape. |

## Focused Live Pass: Generator, Runtime, And Processing Components

Rechecked current production code on 2026-06-26 with focus on
`pipeline_generator.py`, `module_processing_components.py`, `runtime/adapter.py`,
and `runtime/module_execution.py`.

| File | Current state | Follow-up boundary |
|---|---|---|
| `openhcs/interop/cellprofiler/pipeline_generator.py` | Mostly generic orchestration now. Registry loading, setting binding, function resolution, and processing-component lowering route through the selected `CellProfilerModule` class. | Keep it a renderer/orchestrator. Remaining dependencies are the artifact-flow compiler result (`CellProfilerSymbolTable`, `ModuleArtifactContracts`), infrastructure retained-artifact policy, and source-binding emission from contracts. |
| `openhcs/interop/cellprofiler/module_processing_components.py` | Generic source-axis algebra and runtime-artifact lineage lowering. No obvious concrete module-name list remains. | The category-to-axis registry keyed by `image_operation`, `z_projection`, and `channel_operation` should stay tied to declaration-owned module categories and should not grow into a parallel module registry. |
| `openhcs/interop/cellprofiler/runtime/adapter.py` | Generic source-binding, runtime artifact, object-label provenance, and metadata mechanics. No current module-name dispatch found in the focused scan. | Feed declaration-owned provenance/source-domain requirements through compiled plans. Do not add module-specific adapter branches. |
| `openhcs/interop/cellprofiler/runtime/module_execution.py` | Main remaining runtime semantic mirror. `CellProfilerModuleRuntimePlan.build(...)` centrally selects policy families via module-name lookup, and many policy leaves still declare `module_name`. | Move selection facts to `CellProfilerModule` declarations or inherited declaration mixins while keeping runtime request/projection mechanics generic. |

## Coverage Audit For Previously Unmentioned Paths

A follow-up path coverage pass compared every production CellProfiler Python file against the file paths explicitly named in these three reports. Allowing line-qualified references such as `file.py:123`, the reports explicitly path-mentioned 57 of 181 production files, leaving 126 files not explicitly named. Some of those were already logically covered by a companion report, especially `symbol_table.py`; most of the rest are backend algorithm implementations or generic runtime/provenance helpers. The leak does go further in the specific files below.

### Newly Flagged Declaration Leaks

| File | Evidence | Classification | Declaration-owned target |
|---|---|---|---|
| `openhcs/interop/cellprofiler/debug_views.py` | L262-L350, L364-L383, L402-L505 | `Move to declaration`. It has module-name keyed debug renderer tables, an AutoRegisterMeta registry keyed by `module_name`, and generated specialized renderer classes for individual modules. | Debug renderer/section family should be declared on the module class or inherited debug traits; debug rendering should query the module declaration. |
| `openhcs/interop/cellprofiler/settings_binder.py` | L176-L205, L226-L247 | `Move to shared declaration base / keep mechanics generic`. `SettingToKeywordBinding` is generic, but `SKIP_SETTINGS` is a standalone all-module setting policy outside the module declaration family. | CP infrastructure setting ignore policy belongs on the shared `CellProfilerModule` base or a nominal infrastructure-setting family; module-specific binding rows belong on module declarations. |
| `openhcs/interop/cellprofiler/runtime_pipeline.py` | L149-L155, L367-L391, L456-L462 | `Move to compiler/declaration result`. It carries generated semantic contracts and defaults `infrastructure_module_names` from a parallel module-name set. | Pipeline import should consume declaration-derived artifact contracts and module-role facts; infrastructure membership should be a module declaration role query. |
| `openhcs/interop/cellprofiler/runtime/processing_contracts.py` plus `runtime/image_execution_strategies.py` and `runtime/runtime_plane_kwargs.py` | `processing_contracts.py` L68-L99; `image_execution_strategies.py` L76-L87; `runtime_plane_kwargs.py` L100-L111 | `Rewire to declaration-derived invocation contract`. These are generic execution helpers, but currently re-authoritize runtime behavior from static callable metadata instead of an invocation/module declaration contract. | Runtime should receive or query the compiled module invocation contract derived from `CellProfilerModule`; callable metadata remains implementation metadata, not the semantic SSOT. |

### Reviewed But Not A New Module-Specific SSOT

| File or group | Evidence | Classification | Notes |
|---|---|---|---|
| `openhcs/interop/cellprofiler/parser.py` | L51-L62, L188-L193 | `No leak`. | Parses `.cppipe` module names into `ModuleBlock`; it does not declare module semantics. |
| `openhcs/interop/cellprofiler/runtime/artifact_binding.py` | L138-L147, L298-L405, L521-L535 | `Generic executor`. | Artifact-kind dispatch and source-binding mechanics can stay generic, but any module-specific artifact-role facts it consumes must come from declaration-derived contracts. |
| `openhcs/interop/cellprofiler/runtime/output_value_resolution.py` | L32-L54 | `Generic executor needing declaration input`. | It projects callable special outputs; after declaration-owned artifacts mature, this should consume the compiled declaration contract rather than independently rediscovering output semantics from the callable. |
| `openhcs/interop/cellprofiler/runtime/output_record_request.py` | L113-L175, L221-L278, L294-L363 | `Generic executor / provenance glue`. | Relationship-derived object-label source selection is downstream of artifact contracts and endpoint declarations. It should not become a new `source_identity_stack_axes`-style declaration surface. |
| `openhcs/interop/cellprofiler/runtime/source_binding_runtime.py`, `runtime/source_candidates.py`, `runtime/source_identity.py` | `source_binding_runtime.py` L193-L246, L594-L742; `source_identity.py` L206-L538, L768-L845 | `Generic source/provenance mechanics`. | These are not module-name keyed semantic mirrors. They should derive source identity from actual stack/provenance metadata and source-binding plans, not from authored `FunctionStep` source identity fields. |
| Backend algorithm strategy registries such as `classification.py`, `grid.py`, `morphology.py`, and `object_filtering.py` | `classification.py` L75; `grid.py` L135, L568, L1129; `morphology.py` L389, L584, L1040; `object_filtering.py` L420, L555, L735 | `Actual backend implementation`. | These `__registry__` hits are implementation strategy families keyed by algorithm method/label, not CP module-name registries. Declarations should own the CP setting domain that selects a strategy; the executable strategy family can stay in the backend file. |
| Backend wrapper files not individually named above | e.g. `flagging.py` L20-L47, `feature_enhancement.py` L58-L113, `object_overlap.py` L201-L354 | `Broad wrapper ABI surface`. | Covered by the `Wrapper modules broadly` row below. The leak is duplicated callable/module contract ABI when it mirrors module declarations; algorithm code and payload schemas stay with the wrapper. |

## Interop Non-Runtime Boilerplate

| File | Evidence | Boilerplate / semantic mirror | Declaration-owned fact |
|---|---|---|---|
| `openhcs/interop/cellprofiler/module_semantics.py` | L172-L390 | Large module category, dimensionality, infrastructure, and mask-support table keyed by CP module names. | `CellProfilerModule` traits: category, dimensionality, mask support, infrastructure role. |
| `openhcs/interop/cellprofiler/module_roles.py` | L35-L46, L73-L90, L133-L145 | Infrastructure role derivation and per-module import notes for `LoadData` / `ExportToSpreadsheet`. | Module role and `infrastructure_import_note`; retained artifact behavior should remain a declaration query. |
| `openhcs/interop/cellprofiler/module_function_resolution.py` | L41-L54, L139-L195, L229-L279 | Function-resolution registry keyed by module name; embeds scope settings, defaults, object function names, volumetric variants. | Function variant resolver on the module declaration: target function, scope setting, object variant, volumetric variant settings. |
| `openhcs/interop/cellprofiler/module_settings_binding.py` | L582-L617, L1156-L1232, L1263-L1405, L1408-L1580, L1650-L2221 | Main per-module settings-to-kwargs registry; embeds module names, setting labels, ignored settings, aliases, unsupported settings, and parsing rules. | Module declaration setting bindings, ignored settings, unsupported settings, setting aliases, and semantic binding hooks. |
| `openhcs/interop/cellprofiler/setting_names.py` | L95-L102, L172-L182 | Shared setting-name constants and blank-symbol sentinel values. | Module declarations' artifact/measurement setting-name families and blank/default symbol policy. |
| `openhcs/interop/cellprofiler/artifact_semantics.py` | L104-L149, L157-L235, L238-L260 | Generic setting-label classifiers infer artifact roles from CP setting text patterns. | Module declaration artifact input/output specs and setting-role declarations. |
| `openhcs/interop/cellprofiler/module_artifact_inputs.py` | L27-L39 | Artifact-input helper queries `CellProfilerModule.for_module(...).artifact_inputs(...)`. | Already declaration-owned; keep as generic query helper or inline later. |
| `openhcs/interop/cellprofiler/module_processing_components.py` | L414-L475, L615-L620 | Category-to-axis execution role registry keyed by absorbed category strings such as `image_operation`, `z_projection`, `channel_operation`. | Module declaration processing category and execution-axis/lineage semantics. |
| `openhcs/interop/cellprofiler/processing_contract_resolution.py` | L32-L54 | Processing contract resolver ignores `declared_contract` and relies on callable metadata; error text references catalog declaration coercion. | Module/function declaration processing contract or callable contract metadata. |
| `openhcs/interop/cellprofiler/module_runtime_semantics.py` | L17-L26, L38-L64 | Runtime-semantics registry for module revisions; `Watershed` revision interval selects CP4 vs library runtime family. | Module declaration runtime semantics by revision/schema version. |
| `openhcs/interop/cellprofiler/semantic_defaults.py` | L158-L184, L254-L277, L280-L356 | Source-vs-absorbed semantic default contracts keyed by module/function; encodes `MedianFilter`, `Watershed`, `Threshold` default/execution-domain expectations. | Module declaration semantic defaults and execution-domain requirements. |
| `openhcs/interop/cellprofiler/execution_validation.py` | L30-L35, L76-L86, L150-L180 | Infrastructure export validation knows `ExportToSpreadsheet` and `SaveImages`; image exports query declarations later in the same flow. | Module declaration export features and table/image materialization semantics. |
| `openhcs/interop/cellprofiler/source_schema.py` | L74, L557-L567, L642-L770, L908-L1065 | Setup/input module parsers keyed by `Images`, `LoadImages`, `Metadata`, `NamesAndTypes`, `Groups`; embeds `LoadImages` setting constants/prefixes. | Infrastructure module declarations for source-schema ingestion settings and parsing semantics. |
| `openhcs/interop/cellprofiler/grid_settings.py` | L22-L39, L42-L60, L60-L94, L101-L193 | Grid function variant resolver and setting lowering for `DefineGrid` / `IdentifyObjectsInGrid`. | Module declaration function variant and grid setting bindings. |

## Settings-Module Cluster

These files are still production Python and should be treated as semantic mirrors unless they are only enum/type definitions shared by declarations. The short-term target is not to rename them; it is to pull the declaration-owned facts into the `CellProfilerModule` classes and leave only reusable value types behind.

| File | Current role | Declaration-owned fact |
|---|---|---|
| `openhcs/interop/cellprofiler/calculate_math_settings.py` | Typed lowering for CalculateMath settings. | Operand setting bindings and measurement/object/image operand ABI. |
| `openhcs/interop/cellprofiler/expand_or_shrink_settings.py` | Typed CP settings for expand/shrink behavior. | Module setting domains and function variant/default behavior. |
| `openhcs/interop/cellprofiler/grid_settings.py` | DefineGrid / IdentifyObjectsInGrid settings and variants. | Grid module setting bindings and function variant selection. |
| `openhcs/interop/cellprofiler/illumination_settings.py` | CorrectIllumination setting choices and lowering. | Illumination module setting bindings and execution/runtime semantics. |
| `openhcs/interop/cellprofiler/image_math_settings.py` | ImageMath operand settings. | Variadic image operand ABI and setting bindings. |
| `openhcs/interop/cellprofiler/image_module_settings.py` | Shared image-module setting helpers. | Common inherited image-module setting families. |
| `openhcs/interop/cellprofiler/intensity_distribution_settings.py` | Intensity distribution setting rows. | Measurement row/setting domain facts for the module declaration. |
| `openhcs/interop/cellprofiler/mask_objects_settings.py` | MaskObjects setting choices. | Object-label primary domain and mask binding facts. |
| `openhcs/interop/cellprofiler/relate_objects_settings.py` | RelateObjects endpoint/distance settings. | Parent/child endpoint and distance-output facts. |
| `openhcs/interop/cellprofiler/resize_settings.py` | Resize / ResizeObjects settings. | Resize variant and volumetric setting facts. |
| `openhcs/interop/cellprofiler/save_images_settings.py` | SaveImages output settings. | Infrastructure retained-artifact/export behavior. |
| `openhcs/interop/cellprofiler/structuring_element_settings.py` | Structuring-element setting names/defaults. | Morphology volumetric execution and footprint setting facts. |
| `openhcs/interop/cellprofiler/watershed_settings.py` | Watershed setting choices and special inputs. | Watershed method-specific special input roles and runtime-family semantics. |

Previously deleted `_settings.py` files in the current worktree are also part of this cleanup family: `align_settings.py`, `area_occupied_settings.py`, `classify_objects_settings.py`, `color_to_gray_settings.py`, `crop_settings.py`, `display_data_settings.py`, `enhance_edges_settings.py`, `export_to_database_settings.py`, `filter_objects_settings.py`, `gray_to_color_settings.py`, `overlay_outlines_settings.py`, `resize_objects_settings.py`, `smooth_settings.py`, `straighten_worms_settings.py`, `unmix_colors_settings.py`, and `untangle_worms_settings.py`.

## Residual Runtime Boilerplate

This section covers runtime files not already detailed in `cellprofiler_runtime_declaration_semantics_inventory_20260626.md`, or additional surfaces found after that pass.

| File | Evidence | Branch / policy | Semantic mirror | Declaration-owned fact |
|---|---|---|---|---|
| `openhcs/interop/cellprofiler/runtime/policy_registry.py` | L28-L78, L127-L180, L237-L248 | `CellProfilerModulePolicyAutoRegisterMeta` / `CellProfilerModulePolicyLeafSpec` | Runtime owns module-name keyed policy lookup and generated leaf declarations. | Module declaration should own lookup key and module-specific policy facts. |
| `openhcs/interop/cellprofiler/runtime/output_recording.py` | L94-L146 | `DefaultMeasurementRecordBuilder` | Generic measurement ownership inferred from emitted object-name fields, declared object inputs, and primary image inputs. | Measurement row ownership/source qualification rule. |
| `openhcs/interop/cellprofiler/runtime/output_recording.py` | L149-L172 | `MeasureColocalizationMeasurementRecordBuilder` | Source-pair measurements preserve table-level source identity. | Source-pair measurement row ownership. |
| `openhcs/interop/cellprofiler/runtime/output_recording.py` | L175-L193 | `MeasureObjectNeighborsMeasurementRecordBuilder` | Object-topology measurements are object-owned and unqualified by image source. | Object topology measurement ownership/source rule. |
| `openhcs/interop/cellprofiler/runtime/output_recording.py` | L196-L254 | `ProducedImageMeasurementRecordBuilder` / `Crop` leaf | Diagnostic rows are owned by exactly one produced image artifact; multi-output ownership resolved by retained image payload type. | Produced-image measurement owner and output selection rule. |
| `openhcs/interop/cellprofiler/runtime/output_recording.py` | L257-L279 | `ThresholdMeasurementRecordBuilder` | Threshold diagnostics describe produced binary image, with unqualified row source name and source payload from produced artifact. | Threshold diagnostic feature owner/provenance. |
| `openhcs/interop/cellprofiler/runtime/output_recording.py` | L282-L300 | `AlignMeasurementRecordBuilder` | Align shift rows are image-scoped per declared image output. | Align output-index to image-source measurement mapping. |
| `openhcs/interop/cellprofiler/runtime/output_recording.py` | L303-L338 | `RelateObjectsMeasurementRecordBuilder` | RelateObjects merges emitted table rows with relationship rows and keeps parent-scoped source context. | RelateObjects mixed relationship/table measurement recording. |
| `openhcs/interop/cellprofiler/runtime/output_recording.py` | L341-L397 | `IdentifyObjectRelationshipsMeasurementRecordBuilder` / `ClassifyObjects*` leaves | IdentifySecondary adds threshold plus relationship facts; ClassifyObjects emits image/object classification-bin rows. | Object-creation relationship diagnostics and classification row schema. |
| `openhcs/interop/cellprofiler/runtime/output_recording.py` | L401-L442 | `CalculateMathMeasurementRecordBuilder`, `MeasureObjectSizeShapeMeasurementRecordBuilder` | These modules suppress inherited image-source qualification and use object/input ownership. | Per-module measurement source qualification override. |
| `openhcs/interop/cellprofiler/runtime/output_recording.py` | L445-L504 | `IdentifyObjects*MeasurementRecordBuilder` leaves | IdentifyPrimary emits threshold diagnostics; IdentifySecondary emits relationship facts. | Segmentation diagnostic row family per identify module. |
| `openhcs/interop/cellprofiler/runtime/output_recording.py` | L508-L552 | `IdentifyTertiaryObjectsMeasurementRecordBuilder`, `TrackObjectsMeasurementRecordBuilder` | Tertiary emits relationship rows; TrackObjects annotates long-form rows with object/source ownership. | Tertiary relationship row policy and TrackObjects mixed row ownership. |
| `openhcs/interop/cellprofiler/runtime/measurement_rows.py` | L75-L111, L197-L305 | `ClassifyObjectsMeasurementRows` | Absorbed classifier stats become CP feature names `Classify_*`, with dense object labels from `1..total_objects`. | Classification feature template and object-domain completion. |
| `openhcs/interop/cellprofiler/runtime/measurement_rows.py` | L114-L137, L309-L346 | `AlignMeasurementRows` | `output_index` selects declared output image; emits `Align_Xshift` / `Align_Yshift`. | Align result-field schema and output-image ownership. |
| `openhcs/interop/cellprofiler/runtime/measurement_rows.py` | L349-L465 | `ThresholdMeasurementRows` | Threshold stat fields map to `FinalThreshold_*`, `OrigThreshold_*`, variance, entropy; final threshold falls back across known field names. | Threshold stat schema and CP feature-name template. |
| `openhcs/interop/cellprofiler/runtime/measurement_rows.py` | L492-L663 | `ObjectLocationMeasurementRows` | Object-label output gets `Center_X` / `Center_Y` rows over declared or present label domains, including empty declared labels. | Object-location diagnostic row schema and domain inclusion rule. |
| `openhcs/interop/cellprofiler/runtime/relationship_measurement_rows.py` | L100-L149, L668-L689 | `RelationshipMeasurementRows` / `RelateObjectsRelationshipMeasurementRows` | Module-name registry specializes `RelateObjects`; default emits child counts and parent IDs, RelateObjects adds configured distances. | Relationship row family and RelateObjects distance extension. |
| `openhcs/interop/cellprofiler/runtime/relationship_measurement_rows.py` | L239-L367 | `child_count_rows_for_ids`, `parent_rows_for_pairs` | Parent rows are keyed by object numbers, not raw label IDs; missing parent defaults to `0`. | Parent/child feature semantics and object-number projection. |
| `openhcs/interop/cellprofiler/runtime/relationship_measurement_rows.py` | L493-L584 | relationship slice pairing fallbacks | Relationship pairs are slice-qualified from measurement slice, child-label stack slices, or payload slice indices. | Relationship slice-axis ownership/source rule. |
| `openhcs/interop/cellprofiler/runtime/relationship_measurement_rows.py` | L732-L886 | RelateObjects distance rows | Distance features emit child centroid/minimum distances and optional parent mean distance rows via `calculate_per_parent_means`. | RelateObjects distance feature set and option-gated parent summaries. |
| `openhcs/interop/cellprofiler/runtime/relationship_endpoints.py` | L62-L165 | `RelationshipEndpointResolver` | Relationship endpoints inferred from artifact names, two-input fallback, or declaration mixins with primary indexes. | Parent/child endpoint contract and fallback priority. |
| `openhcs/interop/cellprofiler/runtime/relationship_endpoints.py` | L184-L230 | distance relationship type probe | Distance rows apply only to the indexed relationship output of `PrimaryObjectInputRelationshipDistanceModule`. | Distance-output index and relationship endpoint declaration. |
| `openhcs/interop/cellprofiler/runtime/measurement_image_sources.py` | L19-L85 | `CellProfilerImageMeasurementSource` strategies | Measurement rows are either owned by produced image artifact or unqualified runtime image payload. | Image-measurement source ownership strategy. |
| `openhcs/interop/cellprofiler/runtime/measurement_image_resolver.py` | L50-L92 | per-object measurement-image resolver | Per-object measurement modules use composed image request unless policy says images are independent; no image input uses object-label carrier. | Measurement-image composition/cardinality mode. |
| `openhcs/interop/cellprofiler/runtime/measurement_source_names.py` | L23-L39 | source-name helpers | Composed measurement source names join input image names with `__`; multiple image sources require row source names. | Measurement source-name composition and qualification requirement. |
| `openhcs/interop/cellprofiler/runtime/runtime_artifact_cache_invalidation.py` | L19-L92 | artifact-kind invalidation policies | Image/object/measurement/relationship writes invalidate different runtime caches; relationships invalidate none. | Artifact-kind cache dependency semantics. |
| `openhcs/interop/cellprofiler/runtime/pure2d_output_aggregation.py` | L48-L74, L133-L180 | `CellProfilerPure2DOutputAggregator` and relationship aggregator | Per-slice outputs aggregate by runtime output type; parent-child relationship payloads concatenate ids across slices. | Output-type aggregation strategy and relationship slice aggregation rule. |

## Backend Declaration And Wrapper Surfaces

| File | Evidence | Classification | Declaration-owned fact |
|---|---|---|---|
| `openhcs/processing/backends/cellprofiler/module_classes.py` | L1-L5, L55-L80 | Source-of-truth declaration file, not a mirror. It defines base metadata fields for module declarations. | Owns `module_name`, `function_name`, `aliases`, `function_variants`, setting bindings, ignored settings, artifact input settings, contract/category/confidence/validated. |
| `openhcs/processing/backends/cellprofiler/module_classes.py` | L82-L132, L134-L153 | Registry boilerplate around declaration facts. | Validates/normalizes declared module facts and resolves canonical module names/aliases from `AutoRegisterMeta`. |
| `openhcs/processing/backends/cellprofiler/module_classes.py` | L163-L220, L222-L347 | Binding/component/export projection code. | Uses declaration-owned setting aliases, declared setting bindings, artifact input settings, enforced variable components, retained artifacts, and image export specs. |
| `openhcs/processing/backends/cellprofiler/module_classes.py` | L643-L655 | `Align` declaration block. | Owns `Align -> align`, `contract='flexible'`, `category='channel_operation'`, confidence/validated, and CP setting labels. |
| `openhcs/processing/backends/cellprofiler/alignment.py` | L679-L696, L718 | Wrapper has callable ABI, `@numpy(contract=ProcessingContract.FLEXIBLE)`, special output schema, and export boilerplate mirroring declared contract/function identity. | Module identity and declared contract belong to `AlignModule`; wrapper owns executable parameters and alignment implementation. |
| `openhcs/processing/backends/cellprofiler/module_classes.py` | L3175-L3190 | Area/volume occupied declaration variants. | Owns canonical module, alias `MeasureImageAreaOccupied`, variants, `contract='flexible'`, and mode setting family. |
| `openhcs/processing/backends/cellprofiler/area_occupied.py` | L140-L154, L200-L230 | Wrapper duplicates callable variants, decorators, special output fields, and ABI names for area measurements. | Declaration owns function-to-module/variant mapping; wrapper owns measurement algorithms and CSV output payload fields. |
| `openhcs/processing/backends/cellprofiler/module_classes.py` | L4259-L4263 | Minimal `Threshold` declaration block. | Owns `Threshold -> threshold`, validated/confidence. |
| `openhcs/processing/backends/cellprofiler/thresholding.py` | L89-L97, L2802-L2835 | CP constants plus wrapper contract/output/parameter ABI. | Declaration owns module mapping; thresholding file owns algorithm constants, runtime image mode, special output fields, and callable defaults. |
| `openhcs/processing/backends/cellprofiler/module_classes.py` | L4664-L4685 | `Watershed` declaration block. | Owns module mapping, declared `contract='unknown'`, setting-to-parameter aliases, artifact input settings, ignored CP settings. |
| `openhcs/processing/backends/cellprofiler/watershed.py` | L46-L48, L1248-L1284 | Runtime constants/registry key plus wrapper ABI, decorators, special inputs/outputs. | Declaration owns CP setting and artifact facts; wrapper owns executable watershed strategy surface and output payload schema. |
| `openhcs/processing/backends/cellprofiler/library.py` | L48-L118, L342-L367 | Projection mirror of `module_classes.py` into `AbsorbedFunctionMetadata` and default contracts. | Declaration-owned facts are loaded from registered module classes, not authored here. |
| `openhcs/processing/backends/cellprofiler/library.py` | L370-L438, L504-L506 | Function-location registry boilerplate derived from declared function names and AST export discovery. | Function-to-module facts originate in declarations; this file owns discovery/indexing mechanics. |
| `openhcs/processing/backends/cellprofiler/function_documentation.py` | L16-L24, L27-L74, L287-L302, L404-L428 | Documentation/setting-name mirror from CP source AST and parameter aliases. | CP source docs/settings are external declaration facts; this file owns rendered docs and parameter-setting mapping. |
| `openhcs/processing/backends/cellprofiler/__init__.py` | L42-L45, L47-L112, L121-L135 | Package export/catalog boilerplate around discovered functions. | Function list comes from `function_inventory()`; module contract comes from `get_contract()`. |
| `openhcs/processing/backends/cellprofiler/_backend.py` | L21-L40, L78-L180 | True backend selection production code. | Backend provider/strategy facts are owned here and are separate from CellProfiler module declarations. |
| Wrapper modules broadly | e.g. `alignment.py` L679-L696; `area_occupied.py` L140-L154; `thresholding.py` L2802-L2835; `watershed.py` L1248-L1284 | Repeated decorator/ABI mirrors: `@numpy(...)`, `@special_inputs`, `@special_outputs`, `runtime_image_execution_mode`, function signatures. | Module declarations own module name/function mapping and CP setting metadata; wrappers own executable algorithms and runtime payload schemas. |

## NRA DSL Automation Shape

The inventory should be normalized into operation-ready rows before generating recipes. Prose-only rows are not enough.

### Useful Selector And Operation Families

| NRA feature | Use here |
|---|---|
| `source_index_target` | Select classes, methods, assignments, or call sites by file path, qualname, name, or regex. |
| `class_family_target` | Select a module declaration class and inherited/descendant declaration families. |
| `inheritance_edge_target` | Batch add/remove declaration mixins once a fact family is promoted to a parent class. |
| `call_site_target` | Find callers of old policy registries, builders, and helper functions such as `for_module(...)`. |
| `target_set_expression` | Compose include/require/exclude selectors for safe batch edits. |
| `delete_class_assignment` | Remove duplicated class attrs after moving them into inherited declaration families. |
| `add_class_base` / `remove_class_base` | Migrate module declaration classes to nominal policy mixins. |
| `ensure_import` / `remove_import_names` | Move imports from old helper modules to declaration/mixin modules. |
| `apply_selected_targets` | Apply repeated target-local `replace_text` or attr operations with `selection_count` gates. |
| `delete_selected_targets` | Delete generated policy leaves only after a selector proves the exact expected targets. |

### Inventory Row Schema

Use this shape for rows intended to become codemod payloads:

```yaml
- file_path: openhcs/interop/cellprofiler/module_function_resolution.py
  target_qualname: ResizeFunctionResolutionStrategy
  target_kind: class
  current_owner: module_function_resolution_registry
  declaration_owner: ResizeModule
  declaration_fact: function_variant_resolution
  operation: add_class_base
  operation_args:
    base_name: VolumetricResizeFunctionVariantMixin
  followup_operations:
    - operation: delete_class_assignment
      attribute_name: module_name
    - operation: delete_selected_targets
      selector: source_index_target
  selection_count:
    exact: 1
  rationale: function-resolution fact should be inherited by the module declaration, not stored in a parallel registry
```

### Mechanical Candidate Batches

| Batch | Selector idea | Operation family | Safety gate |
|---|---|---|---|
| Move module-name keyed runtime leaves | `source_index_target` for classes under `openhcs/interop/cellprofiler/runtime` with class attr `module_name` | `add_class_base`, declaration attr additions, then `delete_selected_targets` for empty leaves | `selection_count.exact` from inventory rows |
| Remove duplicated class attr constants | `class_family_target` under `CellProfilerModule` declarations with now-inherited attrs | `delete_class_assignment` | Compare declaration-derived catalog before/after |
| Replace `CellProfilerModulePolicyLeafSpec(...)` generated leaves | `call_site_target` with callee `CellProfilerModulePolicyLeafSpec` | declaration mixin additions and import cleanup | Exact count from grep/inventory |
| Replace symbol-table setting imports | `source_index_target` for imports from `symbol_table` with `*_SETTING` names | `ensure_import` from declaration/setting family owner, `remove_import_names` | Import parse validation and targeted tests |
| Replace function-resolution registry leaves | `class_family_target` under `_ModuleFunctionResolutionStrategy` | move fact to declaration, update generic caller to query declaration | Targeted generator import tests |
| Replace settings-binding registry leaves | `class_family_target` under `_ModuleSettingsBindingStrategy` | move bindings to declarations or inherited binding mixins | Setting coverage fixture parity |

### Validation Gates

- Every generated recipe should run NRA simulation with `parse_valid`, `validated_file_paths`, and nested `parse_validation` checked before applying.
- Every batch should include `selection_count` where inventory gives a known expected count.
- After applying a batch, rerun the inventory selector for the old owner and require that only known generic executor code remains.
- Then run focused OpenHCS tests for the moved fact family before deleting the old layer.

## Work Order Across Inventories

1. Establish `CellProfilerModule` as the single AutoRegisterMeta-backed semantic root for CP module facts. AutoRegisterMeta registries are acceptable only when they are the one registry for that semantic family; they are not acceptable when they mirror module facts already declared on `CellProfilerModule`.
2. Make function resolution automatic from declarations: `function_name`, `function_variants`, aliases, and optional declaration override hooks select the backend callable. `_ModuleFunctionResolutionStrategy` can remain only as generic execution/helper code during migration; it must not remain a parallel module-name registry.
3. Make settings binding automatic from declarations: callable signatures, `setting_bindings`, `setting_parameter_aliases`, `ignored_settings`, and module-local postprocess hooks live on the module declaration or inherited declaration mixins. `SettingsBinder` parses and executes binding mechanics; it does not own module-specific binding semantics.
4. Move symbol-table artifact semantics into declaration query hooks and build the generic artifact-flow compiler described in `cellprofiler_symbol_table_usage_inventory_20260626.md`.
5. Rewire generator/compiler code to query the selected module declaration and generic compiler result. The generator lowers already-declared facts; it does not select module-specific semantics itself.
6. Rewire runtime policy lookup to query module declarations for execution mode, primary domain, special inputs, measurement ownership, output provenance, and row materialization.
7. Convert residual module-name keyed registries into declaration mixins and generic executors.
8. Delete or collapse empty policy files only after callers no longer depend on them and selection-count/parse-validation gates pass.

## Scan Commands Used

```bash
rg --files openhcs/interop/cellprofiler openhcs/processing/backends/cellprofiler
git status --short openhcs/interop/cellprofiler openhcs/processing/backends/cellprofiler
rg -n "module_name\\s*=|module_names\\s*=|CellProfilerModulePolicyLeafSpec\\(|for_module\\(|__registry__|class .*Policy|class .*Strategy|class .*Builder|class .*Binding|_SETTING\\b|_SETTINGS\\b|source_bindings|semantic_contract|artifact_inputs|ignored_settings_for|runtime_bound_parameters|CellProfilerModule\\.for_module|module_type\\.|canonical_module_name" openhcs/interop/cellprofiler openhcs/processing/backends/cellprofiler -g '*.py'
rg -n "CellProfilerModulePolicyLeafSpec\\(|module_name\\s*=|module_names\\s*=|for_module\\(|__registry__|class .*Policy|class .*Binding|class .*Resolver|class .*Strategy" openhcs/interop/cellprofiler/runtime -g '*.py'
```
