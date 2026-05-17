# Source Assignment Model Cleanup Plan

## Goal

Finish the source-assignment cleanup below the GUI layer. The editor is no
longer the main problem; the remaining debt is in source matching and workspace
materialization.

The target is one coherent source-assignment model where selectors, filters,
image-set grouping, virtual workspace filenames, imported metadata joins, and
auxiliary artifact mappings are represented by typed domain records instead of
repeated structural mappings and helper pipelines.

## Verified Current State

Advisor scan run on:

```bash
.venv/bin/python -m nominal_refactor_advisor \
  openhcs/core/source_bindings.py \
  openhcs/core/pipeline_image_schema.py \
  openhcs/core/source_schema_workspace.py \
  openhcs/core/source_matching.py
```

Current findings are concentrated in:

- `openhcs/core/source_matching.py`
- `openhcs/core/source_schema_workspace.py`

Previously completed and currently not the target:

- `openhcs/core/source_bindings.py` was reduced to zero findings in the earlier
  source-binding pass.
- `openhcs/core/pipeline_image_schema.py` was reduced to zero findings in the
  earlier source-role pass.
- `openhcs/pyqt_gui/widgets/source_bindings_editor.py` is already a typed editor
  over `SourceBindingsViewModel`, `EditableSourceBindingRow`,
  `StructuredSelectorCellWidget`, and semantic picker dialogs.

Relevant tests:

- `tests/unit/test_source_matching.py`
- `tests/unit/test_source_schema_workspace.py`
- `tests/unit/test_source_bindings.py`
- `tests/unit/test_pipeline_image_schema.py`
- `tests/unit/test_source_bindings_view.py`
- `tests/unit/test_cellprofiler_source_schema.py`
- `tests/unit/pyqt_gui/test_source_bindings_editor.py`

## Concrete Findings To Address

### Source Filter Matching

`openhcs/core/source_matching.py` has 12 metadata-only matcher leaves:

- value predicates: contains, not contains, regex contains, equals, starts/ends
- path predicates: image and TIFF checks

These should be generated from typed matcher declarations rather than
hand-written class shells. The existing nominal family is good; the leaf
declaration mechanism is the missing layer.

Also remove `_source_filter_target_text`, which is now a trivial forwarding
wrapper over `SourceFilterTargetResolver.for_subject(...).resolve_text(...)`.

### Workspace Source Schema

`openhcs/core/source_schema_workspace.py` has the larger remaining model debt:

- repeated `construct_filename(...)` projections in primary, auxiliary, anchor,
  and collision-free site mappings
- repeated structural annotations:
  - `Mapping[AllComponents, Mapping[str, str | None]]`
  - `Mapping[str, tuple[SourceSchemaCandidate, ...]]`
  - `Mapping[tuple[str, ...], tuple[Mapping[str, str], ...]]`
  - tuple return shapes combining virtual path, metadata, and component values
- unused `_empty_workspace_component_values`
- helper candidates that should become nominal authorities:
  - `_add_mapping`
  - `_merged_image_set_metadata`
  - `_source_metadata_for_virtual_path`

## Target Architecture

Introduce or promote these load-bearing source-workspace domain objects:

- `SourceFilterMatcherDeclaration`
  - generates value/path matcher leaves for `SourceFilterMatcher`
  - owns `match_type`, registry key, and predicate semantics

- `SourceSchemaCandidateGroups`
  - semantic alias or dataclass around `Mapping[str, tuple[SourceSchemaCandidate, ...]]`
  - used by metadata/order image-set assemblers and validation

- `WorkspaceComponentValueMap`
  - semantic alias or dataclass around component-to-value projection
  - used by filename construction and metadata writing

- `ImportedMetadataJoinIndex`
  - semantic alias or dataclass around join-key to imported rows

- `WorkspaceVirtualPathProjection`
  - authoritative builder for `construct_filename(...)` calls
  - owns defaults for well/site/channel/z/timepoint/extension

- `WorkspaceMappingSink`
  - replaces `_add_mapping`
  - owns conflict handling and map mutation semantics, not just argument
    forwarding

- `WorkspaceVirtualPathMetadata`
  - replaces `_source_metadata_for_virtual_path`
  - owns how image-set metadata, candidate metadata, and source assignment
    metadata combine

- `ImageSetMetadataMerge`
  - moves `_merged_image_set_metadata` into the image-set assembler boundary

## Non-Goals

- Do not rewrite `SourceBindingsEditorWidget` in this campaign.
- Do not change serialized dataclass field names for source-binding configs.
- Do not make generic private helpers whose only value is hiding repeated code.
- Do not weaken CellProfiler `Images` / `Metadata` / `NamesAndTypes` import
  compatibility.

## Implementation Passes

### Pass 1: Matcher Declarations

1. Add `SourceFilterMatcherDeclaration`.
2. Materialize the 12 matcher leaves from a declaration tuple.
3. Keep `SourceFilterMatcher.for_match_type(...)` stable.
4. Delete `_source_filter_target_text` and call the resolver directly or via a
   behavior-bearing request object.

Verification:

```bash
.venv/bin/python -m pytest tests/unit/test_source_matching.py -q
.venv/bin/python -m nominal_refactor_advisor openhcs/core/source_matching.py
```

### Pass 2: Source Workspace Type Aliases

1. Add semantic aliases or small frozen dataclasses for the repeated mapping
   shapes.
2. Replace annotations only where the alias names a real domain.
3. Avoid wrappers that merely rename a `dict`.

Verification:

```bash
.venv/bin/python -m py_compile openhcs/core/source_schema_workspace.py
.venv/bin/python -m pytest tests/unit/test_source_schema_workspace.py -q
```

### Pass 3: Virtual Filename Builder

1. Introduce `WorkspaceVirtualPathProjection`.
2. Route every `construct_filename(...)` call in workspace materialization
   through it.
3. Preserve exact generated virtual paths in existing tests.
4. Add a regression test for collision-free site component generation if one is
   missing.

Verification:

```bash
.venv/bin/python -m pytest tests/unit/test_source_schema_workspace.py -q
```

### Pass 4: Mapping And Metadata Authorities

1. Replace `_add_mapping` with `WorkspaceMappingSink`.
2. Replace `_source_metadata_for_virtual_path` with
   `WorkspaceVirtualPathMetadata`.
3. Move `_merged_image_set_metadata` into the image-set assembler boundary.
4. Delete `_empty_workspace_component_values` if still unreferenced.

Verification:

```bash
.venv/bin/python -m pytest \
  tests/unit/test_source_schema_workspace.py \
  tests/unit/test_cellprofiler_source_schema.py \
  -q
```

### Pass 5: Integrated Gate

Run the complete source-assignment gate:

```bash
.venv/bin/python -m pytest \
  tests/unit/test_source_matching.py \
  tests/unit/test_source_bindings.py \
  tests/unit/test_pipeline_image_schema.py \
  tests/unit/test_source_bindings_view.py \
  tests/unit/test_source_schema_workspace.py \
  tests/unit/test_cellprofiler_source_schema.py \
  tests/unit/pyqt_gui/test_source_bindings_editor.py \
  -q
```

Run advisor:

```bash
.venv/bin/python -m nominal_refactor_advisor \
  openhcs/core/source_bindings.py \
  openhcs/core/pipeline_image_schema.py \
  openhcs/core/source_schema_workspace.py \
  openhcs/core/source_matching.py
```

## Completion Criteria

- Source matching and source workspace advisor findings are resolved or
  explicitly documented as stable analyzer noise.
- Existing source-binding serialization and CP source-schema import behavior are
  unchanged.
- Focused source tests pass.
- Full `tests/unit` passes before merging this campaign.
