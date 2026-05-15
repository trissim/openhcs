# CellProfiler Source Bindings in the GUI

## Current state

CellProfiler setup modules are lowered into OpenHCS source semantics before pipeline generation. The important core types are:

- `PipelineImageSchema`: pipeline-level source schema lowered from CellProfiler Images, Metadata, NamesAndTypes, and Groups.
- `StepSourceBindingsConfig`: editable `FunctionStep` field that declares which named source artifacts a step needs.
- `NamedSourceBinding`: one CellProfiler-style alias such as `DNA`, `OrigStain1`, or `Objects`.
- `SourceSelector`: typed selector for file/component/metadata filtering.
- `MetadataExtractionRule`: regex-backed metadata extraction from source filenames or folders.
- `SourceBindingMatchPlan`: typed rule for matching aliases into image sets by order or metadata.

The PyQt step editor does not currently have a CellProfiler-specific source-binding widget. `FunctionStep.source_bindings` is rendered by the generic `ParameterFormManager` because it is just another constructor parameter. The editor detects dataclass-backed step parameters, adds them to the left hierarchy tree, and renders their nested fields in the scrollable form. That is structurally correct, but it exposes implementation shape rather than the way a CellProfiler user thinks about image loading.

## CellProfiler comparison

CellProfiler splits this user experience across the setup modules:

- Images: define the file universe and filter out irrelevant files.
- Metadata: extract metadata fields from filenames, folders, file headers, or tables.
- NamesAndTypes: assign meaningful image/object names and match channels into image sets.
- Groups: split image sets into independent groups for group-aware processing.

OpenHCS currently stores the same semantics, but not in the same UI shape:

- `PipelineImageSchema.images_rule` maps to the CellProfiler Images file universe/filtering.
- `PipelineImageSchema.metadata_rules` maps to CellProfiler Metadata extraction.
- `PipelineImageSchema.assignments_by_alias` and `source_artifacts_by_alias` map to NamesAndTypes names.
- `PipelineImageSchema.match_plan` maps to NamesAndTypes image-set matching.
- `PipelineImageSchema.grouping` maps to CellProfiler Groups.
- Per-step `StepSourceBindingsConfig` is the local projection of that schema onto one `FunctionStep`.

This is stronger than CellProfiler internally because the source contract is typed and attached to the actual OpenHCS step. It is weaker as a wet-lab UI because a user sees nested dataclasses instead of an image list, image-set table, and named channel/object columns.

## Desired GUI shape

Keep the generic form system as the authority. The source-binding UI should be a registered inline widget for the structural type `StepSourceBindingsConfig`, not a CellProfiler-specific branch in the step editor.

The PyQt form stack already routes parameters through a generic type-to-widget pipeline:

- `ParameterFormManager` owns form construction.
- `ParameterFormService` classifies parameters as regular, nested, or optional nested.
- `widget_creation_config.create_widget_parametric()` creates the concrete row/container/widget from that classification.
- `MagicGuiWidgetFactory` already has a replacement registry for type-specific regular widgets.

The correct extension is therefore a generic structural-widget hook:

- Add a structural dataclass editor hook, then register `StepSourceBindingsConfig -> SourceBindingsEditorWidget`.
- Let the editor implement the normal value-get/value-set/change-signal expectations used by the form system.
- Render `StepSourceBindingsConfig` inline wherever that type appears, including non-CellProfiler pipelines.
- Optionally detect CellProfiler provenance only to choose labels/help text and preview defaults, not to choose a separate code path.

Keep the existing nested dataclass rendering as the fallback/raw mode inside that widget.

### Fact-check: current registry gap

`StepSourceBindingsConfig` is a dataclass, so it is currently classified as `DirectDataclassInfo` before regular widget creation. That means the existing `MagicGuiWidgetFactory.WIDGET_REPLACEMENT_REGISTRY` is not sufficient by itself; that registry is reached for regular fields after dataclass fields have already been routed to the nested-form path.

The missing generic hook should be one of:

- A new `StructuralDataclassInfo` or `InlineDataclassWidgetInfo` registered before `DirectDataclassInfo`, selected by a type-to-inline-widget registry.
- A new branch in `create_widget_parametric()` for `DirectDataclassInfo` whose type is present in a structural dataclass widget registry.
- A generalized `DirectDataclassInfo` field such as `inline_widget_factory`, populated by the parameter-info factory when the dataclass type is registered.

The first option is the cleanest because it keeps the discriminated-union model intact and does not make `DirectDataclassInfo` carry special cases.

The custom editor should present:

- Source aliases table: alias, artifact kind, origin, required, selector summary, matched count.
- Filter builder: subject, operator, value, with preview count after each rule.
- Metadata rules table: source, regex pattern, extracted fields, filtered subset.
- Match plan editor: order vs metadata, aliases, metadata fields, and validation status.
- Image-set preview: rows are image sets; columns are aliases; cells show matched files or source artifacts.
- Group preview: optional grouping fields, group values, image-set counts.

For imported CellProfiler pipelines, the first implementation can be read-only plus "Edit as code" for exact control. A later pass can make the table editor bidirectional and regenerate the typed `StepSourceBindingsConfig`.

## Integration points

- The generic form system should route `StepSourceBindingsConfig` to a specialized `SourceBindingsEditorWidget`; `StepParameterEditorWidget` should not contain a CellProfiler-specific conditional.
- `PipelineEditorWidget` should expose a pipeline-level "Sources" or "Images" panel that deduplicates source aliases across steps and displays the full image-set table.
- CellProfiler `.cppipe` imports should attach source provenance to the pipeline-level schema, not duplicate source UI state in every step.
- Debug mode should reuse the image-set preview: selecting a row defines the image set/well/site used for step-by-step execution.
- Napari/Fiji buttons should render the selected source image or intermediate artifact, but the source-binding inspector owns the semantic table.

## Implementation notes

The key rule is that the GUI should not invent CellProfiler-specific string tables. It should render the existing typed schema and bindings, using a CP-like layout as presentation only.

### Progress

Implemented the first load-bearing seam as `openhcs.core.source_bindings_view`. It provides a pure, PyQt-free `SourceBindingsViewModel` over `PipelineImageSchema` and `StepSourceBindingsConfig`, including typed rows for:

- pipeline file-universe filters, explicit image-plane sources, and imported metadata joins;
- pipeline image/source-artifact assignments;
- step-local grouped `NamedSourceBinding` declarations;
- metadata extraction rules with named capture fields;
- order/metadata match plans;
- grouping metadata fields.

The same module now also includes `SourceInventory` and `SourceBindingsPreview`, which apply the typed selectors to concrete source candidates. Preview matching reuses `openhcs.core.source_matching` for filters, metadata extraction, component/metadata comparison, and `ImageSetAssembler` for image-set rows. This keeps the preview aligned with runtime source-schema semantics instead of adding a second matcher.

This deliberately stops before a widget implementation. The current useful boundary is a reusable presentation/preview model that can feed GUI, CLI diagnostics, LLM context, and benchmark reporting without duplicating source-binding semantics or adding CellProfiler-specific branches in the step editor.

The right abstraction is therefore a typed source-schema view model:

- input: `PipelineImageSchema`, `StepSourceBindingsConfig`, current plate/source inventory;
- output: alias rows, filter rows, metadata rows, image-set preview rows, group rows;
- no dependency on generated `.cppipe` Python code;
- usable by GUI, CLI diagnostics, LLM context, and benchmark reporting.

The base view model and preview are pure and testable. They do not import PyQt. Matched counts and image-set preview rows require actual filesystem/VFS/plate inventory, so they are represented by `SourceInventory` and `SourceBindingsPreview` instead of being inferred from `StepSourceBindingsConfig` alone.

Suggested layers:

- `SourceBindingsViewModel`: pure data rows. Implemented.
- `SourceInventory`: resolves candidate source files/artifacts from explicit paths plus a schema/bindings context. Implemented for explicit paths, local directories, embedded `ImagePlaneSource` URI lists, and `FileManagerLike`/VFS-backed file listings.
- `SourceBindingsPreview`: applies filters, metadata rules, matching, and grouping to produce preview rows. Implemented.
- `SourceBindingsEditorWidget`: PyQt renderer/editor for the view model.

This keeps CellProfiler support integrated with OpenHCS rather than creating a separate CellProfiler wizard.

Status update: `openhcs.pyqt_gui.widgets.source_bindings_editor.SourceBindingsEditorWidget` now renders the typed `SourceBindingsViewModel` in a CP-like table layout and consumes the same pure model used by CLI/tests.

The generic `pyqt-reactive` structural dataclass hook is now in place. `InlineDataclassWidgetInfo` is selected before `DirectDataclassInfo` when a dataclass type has a registered inline editor, and OpenHCS registers `StepSourceBindingsConfig -> SourceBindingsEditorWidget`. That means the standard `ParameterFormManager` route owns this UI; the step editor does not need a CellProfiler-specific conditional.

The inline widget now participates in the generic form value protocol. It exposes `get_value`, `set_value`, and a `value` property, emits `changed`, and is registered virtually as a `ValueGettable`/`ValueSettable` to avoid Qt/ABC metaclass conflicts. That makes the source-bindings editor a real form widget instead of a passive preview.

The widget also accepts schema-aware preview context. `set_preview_context(...)` takes the active `PipelineImageSchema` and an optional `SourceInventory`, then renders matched source counts and assembled image-set rows using `SourceBindingsPreview`. This keeps GUI preview behavior on the same typed matcher used by tests and CLI diagnostics.

The first bidirectional editing slice is implemented. `SourceBindingsEditorWidget` now exposes an editable step-binding table backed by a typed `EditableSourceBindingRow` and `SourceBindingColumn` enum. Edits add/remove/update `NamedSourceBinding` values and rebuild the owning `StepSourceBindingsConfig`, preserving metadata rules and match plans. Selector semantics are also editable through typed component, metadata, filter, and inherit-scope columns; the table now reads rows from the authoritative `StepSourceBindingsConfig` instead of reconstructing bindings from the presentation-only view model. This keeps the GUI table as a projection of the typed source-binding model instead of making string tables authoritative.

The host editor now supplies imported source-schema context. `PipelineEditorWidget` retains the `CellProfilerPipelineImportResult` from `.cppipe` import, passes its `source_schema` into `DualEditorWindow`, and `StepParameterEditorWidget` applies that schema to inline `SourceBindingsEditorWidget` instances. This means imported CP source assignments are available to the step-local source-binding editor through the normal form/editor path rather than a CP-specific side channel.

The first concrete inventory bridge is now in the pure model layer. `SourceInventory.from_schema_sources(...)` derives preview candidates from explicit `PipelineImageSchema.image_plane_sources`, decodes local `file://` URIs, applies schema/step metadata rules, and feeds the same `SourceBindingsPreview` matcher used by tests. `SourceInventory.from_directory(...)` covers local plate/input-directory discovery, applies `PipelineImageSchema.images_rule` before preview matching, and `SourceInventory.from_schema_context(...)` chooses embedded image-plane sources first with local input-dir discovery as the fallback. `StepParameterEditorWidget` now supplies that inventory to inline source-binding editors when imported schema context is available, so embedded CellProfiler image-plane sources and local plate directories produce GUI match counts and image-set previews without a CellProfiler-specific widget path.

Metadata-rule and match-plan editing are now implemented in the same inline widget. The editor exposes step metadata extraction rules as typed `MetadataExtractionRule` rows and the step match plan as typed `SourceBindingMatchPlan`/`SourceBindingMatchDimension` rows. Row parsing uses the same selector/filter codec as named bindings, and all edits rebuild the authoritative `StepSourceBindingsConfig` rather than storing GUI-only state.

The VFS and enum-control slices are now implemented. `SourceInventory.from_filemanager(...)` builds preview candidates from an OpenHCS `FileManagerLike` backend using the same source-schema matcher as local previews, so non-local/shared storage can feed the editor without a separate CellProfiler code path. `SourceBindingsEditorWidget` now renders closed enum fields as typed combo cells for artifact kind, source origin, metadata source, and match method while preserving the authoritative `StepSourceBindingsConfig` round trip.

The source-binding GUI plan is complete for the intended first implementation: it renders, round-trips, edits named bindings/selectors, edits step metadata rules, edits step match plans, previews embedded/local/VFS inventories through typed source-schema semantics, and integrates through the generic inline dataclass-widget route rather than a CellProfiler-specific branch. Structured cells now open table-backed semantic dialogs for component selectors, metadata selectors, filter clauses, and match dimensions. The dialog columns, hints, closed-domain combo columns, validation hints, parsing, and serialization are projected from one typed `StructuredSelectorEditorSpec` table, so selector UI polish remains inside the existing source-binding abstraction instead of becoming CellProfiler-specific string-table logic.
