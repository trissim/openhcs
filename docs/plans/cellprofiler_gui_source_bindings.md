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

The right abstraction is therefore a typed source-schema view model:

- input: `PipelineImageSchema`, `StepSourceBindingsConfig`, current plate/source inventory;
- output: alias rows, filter rows, metadata rows, image-set preview rows, group rows;
- no dependency on generated `.cppipe` Python code;
- usable by GUI, CLI diagnostics, LLM context, and benchmark reporting.

The view model must be pure and testable. It should not import PyQt. It should accept a source inventory service explicitly, because matched counts and image-set previews require actual filesystem/VFS/plate inventory, not just `StepSourceBindingsConfig`.

Suggested layers:

- `SourceBindingsViewModel`: pure data rows.
- `SourceInventoryResolver`: resolves candidate source files/artifacts from a plate/VFS/source namespace.
- `SourceBindingsPreviewService`: applies filters, metadata rules, matching, and grouping to produce preview rows.
- `SourceBindingsEditorWidget`: PyQt renderer/editor for the view model.

This keeps CellProfiler support integrated with OpenHCS rather than creating a separate CellProfiler wizard.
