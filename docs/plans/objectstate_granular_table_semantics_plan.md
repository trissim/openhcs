# ObjectState Granular Table Semantics Plan

Date: 2026-07-01

## Purpose

Show ObjectState semantic markers at the row/cell level inside complex
dataclass editors, starting with the SourceBindings `Source Filters` table.
The user should be able to see whether `subject`, `match_type`, or `value` in
one `SourceFilterClause` row is dirty, differs from the signature default, or is
displaying an inherited/resolved placeholder value.

Today `SourceBindingsEditorWidget` can show `* Source Filters` or underline the
whole section, but it cannot say which cell inside the table caused the marker.

Implementation status: the first SourceFilters slice is implemented. Keep this
stream separate from runtime/debugger work and SourceBindings live robustness
fixes.

Implemented slice:

- ObjectState exposes structural subfield semantics from typed
  `DottedFieldPath` owner fields and `StructuralValuePath` leaves.
- pyqt-reactive refreshes inline dataclass child semantics through nominal
  protocols and can flash structural table-cell targets.
- SourceBindings routes `source_filters` row/cell markers through those generic
  protocols without recomputing dirty/default/inherited state from display text.

Remaining slice:

- Move the remaining SourceBindings-local table mechanics into pyqt-reactive
  once selector-editor dependencies are separated from the generic table
  controller. The generic structural path, semantic index, and cell target types
  are already in the reusable layer.

Updated after backend review: the implementation must also decompose the local
SourceBindings inline-editor infrastructure. The current SourceBindings editor
has started carrying nominal `DottedFieldPath` child identities, but that code
belongs in reusable pyqt-reactive/ObjectState infrastructure so future
SourceBindings-like configs get the same chrome, reset, provenance, time-travel,
flash, and granular table semantics automatically.

## Non-Negotiables

- Do not add manual semantic mirrors: no separate hardcoded lists of
  SourceBindings fields, SourceFilter columns, or per-widget semantic registries
  that duplicate domain declarations.
- Derive semantic identity from existing authorities:
  - ObjectState field paths and dirty/signature/default/inheritance state.
  - Dataclass fields and sequence element positions.
  - Existing pyqt-reactive widget/value-editor metadata.
  - Existing SourceBindings table column/row metadata.
- Keep any new abstraction generic and nominal. `SourceFilterClause` is the
  worked example, not a new special case.
- Preserve value ownership: SourceBindings table editors continue to produce
  typed dataclass values, and ObjectState remains the semantic-state authority.
- Do not use table display text as state. Dirty/default markers must not corrupt
  the logical values read back by table controllers.
- Do not use enum member names or column title strings as semantic authority.
  Either derive an isomorphic dataclass row from dataclass field order or let the
  existing column/editor declaration own its structural path.

## Backend Review Findings

### Current Implementation State

This checkout already moved part of the generic inline-dataclass work out of
`SourceBindingsEditorWidget`:

- `external/pyqt-reactive/src/pyqt_reactive/forms/inline_dataclass_context.py`
  owns `InlineDataclassFormContext` and
  `InlineDataclassChildFieldIdentity`.
- `external/pyqt-reactive/src/pyqt_reactive/forms/inline_dataclass_chrome.py`
  owns `InlineDataclassChildChrome` for section labels, reset controls,
  dimming, and direct child navigation.
- `openhcs/pyqt_gui/widgets/source_bindings_editor.py` consumes those generic
  classes for SourceBindings section chrome.

That extraction is only the section-level substrate. It does not yet provide
row/cell semantics because ObjectState still exposes dirty/default/inherited
state at field paths, not structural leaves inside tuple/dataclass values. The
next implementation slice must add the UI-neutral structural projection first;
otherwise SourceBindings tables would have to compare display rows locally,
which would be another semantic mirror.

### Generic Code Already Extracted

The section-level inline dataclass substrate has already moved out of
`openhcs/pyqt_gui/widgets/source_bindings_editor.py`:

- `InlineDataclassFormContext`
  - Owns manager/ObjectState path construction, child raw/resolved value lookup,
    reset behavior, child descriptions, child types, and inherited-preview state.
  - It is built from `ParameterFormManager`, `InlineDataclassWidgetInfo`, and
    `DottedFieldPath`.
- `InlineDataclassChildFieldIdentity`
  - Owns reusable nominal child-field identity keyed by ObjectState and manager
    `DottedFieldPath` values.
- `InlineDataclassChildChrome`
  - Owns section labels, reset controls, provenance/dimming refresh, and direct
    child navigation targets.

The remaining generic extraction is table-level:

- `EditableTableColumn`, `EditableTableController`, and `EditableTableLayout`
  - These are generic typed table editor pieces. SourceBindings supplies column
    declarations and row conversion; pyqt-reactive should own row replacement,
    logical/display value separation, semantic chrome, and cell navigation.
- SourceBindings enableable-title handling
  - This mirrors regular nested dataclass enableable title behavior.
  - Reuse `EnabledTitleWidgetMoveAuthority` through an inline-dataclass adapter
    instead of maintaining separate enableable title logic in SourceBindings.

### Current Generic Authorities To Reuse

- `objectstate.DottedFieldPath`
  - Nominal authority for dotted ObjectState paths.
  - Every new child/subfield identity must hold this type internally and only emit
    strings at existing API boundaries that still require them.
- `DataclassFieldAccess`
  - Raw dataclass reads and dotted dataclass traversal authority.
- `ParameterFormChromeSync`
  - Existing refresh authority for labels, groupboxes, provenance controls, and
    widget state.
- `FieldChangeDispatcher`
  - Existing mutation dispatch path. Granular table updates must still enter via
    the owning container field update, not a parallel subfield write path.
- `InlineDataclassGroupBox`, `ChildFieldChromeRefreshable`,
  `ChildFieldNavigationTargetProvider`, and `ResolvedValuePreviewSettable`
  - Existing inline dataclass extension seam. Extend these nominal protocols
    instead of adding SourceBindings-only callbacks.
- `LabelWithHelp`, `GroupBoxWithHelp`, `DirtyLabelState`, and
  `ResetButtonStyler`
  - Existing marker/chrome semantics for `*`, underline, reset styling, and
    provenance. Table-cell chrome should reuse the same semantic meanings.
- `ProvenanceLabel`
  - Existing click-to-source affordance. Cell-level provenance should route to
    the same ObjectState owner path plus a typed structural suffix; it should
    not open windows through SourceBindings-specific callbacks.

### Marker Semantics To Preserve

The table plan must not invent new meanings for existing UI markers:

- `*` means the live resolved value differs from the saved resolved value.
  This is currently implemented by `DirtyLabelState` for labels.
- `_` means the raw value differs from the signature default. This is currently
  rendered as font underline for labels and reset buttons.
- Placeholder/dimmed styling means the displayed value is inherited/resolved
  while the raw editable value is missing or `None`.

For table cells, these meanings come from `ObjectStateSubfieldSemantic`, not
from the visible cell text, enum display names, or table column titles. A cell
may render a visible `*`, underline, border, or tooltip, but readback must use
the logical value stored by the table controller.

## Current Authorities

### Semantic State

- `external/ObjectState/src/objectstate/object_state.py`
  - `ObjectState.parameters` owns raw editable values.
  - `ObjectState.get_resolved_value(field_path)` owns live resolved values.
  - `ObjectState.get_saved_resolved_value(field_path)` owns saved resolved
    values.
  - `ObjectState.signature_default(field_path)` owns signature defaults.
  - `ObjectState.dirty_fields` means live resolved value differs from saved
    resolved value.
  - `ObjectState.signature_diff_fields` means raw value differs from the
    signature default.
  - `_live_provenance` and `project_ui_visible_field_path(...)` own inherited
    source remapping.
- `openhcs/pyqt_gui/services/ui_bridge_object_state.py`
  - `ObjectStateFieldSemanticProjection` is the existing projection of one
    ObjectState field into `dirty`, `signature_diff`, `inherited_value`, raw and
    resolved previews, and provenance.
- `openhcs/agent/services/object_state_field_projection.py`
  - `ObjectStateFieldFilterDeclaration` is the agent-facing filter authority for
    semantic fields.
- `openhcs/agent/dto/ui_bridge.py`
  - `UiObjectStateFieldSummary` and `UiObjectStateFieldProjection` are the
    existing transport DTOs for field-level semantics.

### Value Shape

- `external/ObjectState/src/objectstate/field_access.py`
  - `DataclassFieldAccess` is the raw dataclass field access authority.
- `openhcs/core/source_bindings.py`
  - `SourceBindingsConfig.source_filters` declares the tuple shape.
  - `StepSourceBindingsConfig.source_filters` declares lazy/inherited
    step-level source filters.
  - `SourceFilterClause` declares `subject`, `match_type`, and `value`.
- `openhcs/pyqt_gui/widgets/source_bindings_editor.py`
  - `SourceBindingsEditorValue` reads raw source-binding child fields without
    triggering lazy resolution.
  - `EditableSourceFilterRow` is the nominal row model for one
    `SourceFilterClause`.
  - `SourceFilterColumn` is existing table editor metadata for displayed
    columns.
  - `EditableTableController` owns the `QTableWidget` mechanics and the
    conversion between typed row models and visible cells.
- `external/pyqt-reactive/src/pyqt_reactive/forms/inline_dataclass_context.py`
  - `InlineDataclassFormContext` connects inline dataclass editors to
    ObjectState field paths, reset behavior, raw values, resolved preview
    values, child descriptions, and child types.

### Form and Widget Propagation

- `external/pyqt-reactive/src/pyqt_reactive/forms/parameter_form_manager.py`
  - Owns ObjectState to widget updates, flash queuing, and change dispatch.
  - `_queue_inline_dataclass_flash_for_path(...)` already delegates child
    navigation to inline dataclass widgets.
- `external/pyqt-reactive/src/pyqt_reactive/forms/parameter_form_chrome_sync.py`
  - Owns label/groupbox dirty marker refresh.
  - Calls `ChildFieldChromeRefreshable.refresh_child_field_chrome()` for inline
    dataclass widgets.
- `external/pyqt-reactive/src/pyqt_reactive/protocols/widget_protocols.py`
  - `ChildFieldChromeRefreshable`,
    `ChildFieldNavigationTargetProvider`, and
    `ResolvedValuePreviewSettable` are the current nominal widget extension
    points.
- `external/pyqt-reactive/src/pyqt_reactive/forms/widget_strategies.py`
  - Placeholder state and placeholder styling are already nominal widget
    behavior. Granular table semantics should reuse the same inherited/raw
    distinction, not invent a second placeholder system.

## Target Model

Add a generic structural subfield semantic projection:

```text
ObjectState owner field path
    -> raw / resolved / saved-resolved / signature-default container values
    -> generic structural leaf traversal
       - dataclass field segment
       - tuple/list index segment
       - optional mapping key segment later
    -> ObjectStateSubfieldSemantic rows
    -> pyqt-reactive child-widget semantic refresh
    -> SourceBindings table cell chrome
```

For `source_bindings.source_filters`, the owner field remains the existing
ObjectState field. The granular rows are derived leaves under that field:

```text
source_bindings.source_filters[0].subject
source_bindings.source_filters[0].match_type
source_bindings.source_filters[0].value
```

These display paths are projection paths, not new writable ObjectState
parameters. Mutations still update `source_bindings` or
`source_bindings.source_filters` through the existing form/widget flow.

### Nominal Identity Flow

Every granular table marker follows this chain:

```text
ObjectState owner DottedFieldPath
    -> ObjectStateSubfieldSemanticIndex
    -> InlineDataclassChildFieldIdentity for the direct child section
    -> StructuralValuePath for row/cell descendants
    -> structural table cell target
```

The direct child identity is the bridge from an inline dataclass field such as
`source_bindings.source_filters` to a structural leaf such as
`[0].match_type`. The row/cell target is not an ObjectState writable field; it
is a visual projection target for marker refresh, click-to-provenance, and
flash.

No layer should rebuild this identity from display labels. If an implementation
needs a string, it must be a final rendering of `DottedFieldPath` or
`StructuralValuePath`, not the authority.

## API Drafts

### ObjectState Structural Path

Add a small ObjectState-owned path model. The exact module can be adjusted, but
keep it in ObjectState or a UI-neutral OpenHCS service that only depends on
ObjectState values.

File candidate:
`external/ObjectState/src/objectstate/subfield_semantics.py`

```python
from dataclasses import dataclass
from enum import Enum
from typing import TypeVar


SubfieldValueT = TypeVar("SubfieldValueT")


@dataclass(frozen=True, slots=True)
class MissingValue:
    """Nominal sentinel for an absent structural leaf."""


class StructuralSegmentKind(str, Enum):
    DATACLASS_FIELD = "dataclass_field"
    SEQUENCE_INDEX = "sequence_index"
    MAPPING_KEY = "mapping_key"


@dataclass(frozen=True, slots=True)
class StructuralPathSegment:
    kind: StructuralSegmentKind
    value: str | int


@dataclass(frozen=True, slots=True)
class StructuralValuePath:
    segments: tuple[StructuralPathSegment, ...]

    def child_field(self, name: str) -> "StructuralValuePath": ...
    def child_index(self, index: int) -> "StructuralValuePath": ...
    def display_suffix(self) -> str: ...


MISSING = MissingValue()
```

Use display suffixes only for UI/debug output. Comparisons should use the typed
segments so mapping keys, indexes, and field names are not conflated.

### ObjectState Subfield Projection

```python
@dataclass(frozen=True, slots=True)
class ObjectStateSubfieldSemantic:
    owner_field_path: DottedFieldPath
    relative_path: StructuralValuePath
    display_path: str
    value_type_name: str | None

    raw_value: SubfieldValueT | MissingValue
    resolved_value: SubfieldValueT | MissingValue
    saved_resolved_value: SubfieldValueT | MissingValue
    signature_default_value: SubfieldValueT | MissingValue

    raw_present: bool
    resolved_present: bool
    saved_resolved_present: bool
    signature_default_present: bool

    dirty: bool
    signature_diff: bool
    inherited_value: bool
    semantic_markers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ObjectStateSubfieldSemanticIndex:
    owner_field_path: DottedFieldPath
    owner_dirty: bool
    owner_signature_diff: bool
    owner_inherited_value: bool
    leaves: tuple[ObjectStateSubfieldSemantic, ...]

    def leaf_for(self, relative_path: StructuralValuePath) -> ObjectStateSubfieldSemantic | None: ...
```

Projection rules:

- `dirty`: projected live resolved leaf differs from projected saved resolved
  leaf.
- `signature_diff`: projected raw leaf differs from projected signature-default
  leaf. If the signature default has no such leaf and raw has a leaf, the leaf
  is a default override.
- `inherited_value`: projected raw leaf is missing because the owner raw value is
  `None`, while projected resolved leaf is present.
- `semantic_markers`: reuse existing field marker policy, `*` for dirty and `_`
  for signature/default diff.
- Missing leaves must be explicit with a sentinel; do not collapse missing and
  `None`.
- Provenance starts with the owner field provenance. A follow-up can append the
  structural path once inherited provenance DTOs accept relative subpaths.
- Recent-change/flash state is UI chrome state owned by pyqt-reactive manager
  refresh/flash machinery. Do not store it on `ObjectStateSubfieldSemantic`.

Concrete ObjectState projection draft:

```python
def build_subfield_semantic_index(
    *,
    owner_field_path: DottedFieldPath,
    raw_value,
    resolved_value,
    saved_resolved_value,
    signature_default_value,
    owner_dirty: bool,
    owner_signature_diff: bool,
) -> ObjectStateSubfieldSemanticIndex:
    raw_leaves = _leaf_values(raw_value)
    resolved_leaves = _leaf_values(resolved_value)
    saved_leaves = _leaf_values(saved_resolved_value)
    default_leaves = _leaf_values(signature_default_value)

    ordered_paths = _ordered_paths(raw_leaves, resolved_leaves, saved_leaves, default_leaves)
    if any(path.segments for path in ordered_paths):
        ordered_paths = tuple(path for path in ordered_paths if path.segments)

    return ObjectStateSubfieldSemanticIndex(
        owner_field_path=owner_field_path,
        owner_dirty=owner_dirty,
        owner_signature_diff=owner_signature_diff,
        owner_inherited_value=raw_value is None and resolved_value is not None,
        leaves=tuple(
            _build_leaf(
                owner_field_path=owner_field_path,
                relative_path=relative_path,
                raw_value=raw_leaves.get(relative_path, MISSING),
                resolved_value=resolved_leaves.get(relative_path, MISSING),
                saved_resolved_value=saved_leaves.get(relative_path, MISSING),
                signature_default_value=default_leaves.get(relative_path, MISSING),
            )
            for relative_path in ordered_paths
        ),
    )


def _build_leaf(
    *,
    owner_field_path: DottedFieldPath,
    relative_path: StructuralValuePath,
    raw_value: SubfieldValueT | MissingValue,
    resolved_value: SubfieldValueT | MissingValue,
    saved_resolved_value: SubfieldValueT | MissingValue,
    signature_default_value: SubfieldValueT | MissingValue,
) -> ObjectStateSubfieldSemantic[SubfieldValueT]:
    raw_present = raw_value is not MISSING
    resolved_present = resolved_value is not MISSING
    saved_resolved_present = saved_resolved_value is not MISSING
    signature_default_present = signature_default_value is not MISSING
    dirty = resolved_value != saved_resolved_value
    signature_diff = raw_value != signature_default_value

    semantic_markers: list[str] = []
    if dirty:
        semantic_markers.append("*")
    if signature_diff:
        semantic_markers.append("_")

    value_for_type = resolved_value if resolved_present else raw_value
    return ObjectStateSubfieldSemantic(
        owner_field_path=owner_field_path,
        relative_path=relative_path,
        display_path=f"{owner_field_path.value}{relative_path.display_suffix()}",
        value_type_name=(
            None if value_for_type is MISSING else type(value_for_type).__qualname__
        ),
        raw_value=raw_value,
        resolved_value=resolved_value,
        saved_resolved_value=saved_resolved_value,
        signature_default_value=signature_default_value,
        raw_present=raw_present,
        resolved_present=resolved_present,
        saved_resolved_present=saved_resolved_present,
        signature_default_present=signature_default_present,
        dirty=dirty,
        signature_diff=signature_diff,
        inherited_value=not raw_present and resolved_present,
        semantic_markers=tuple(semantic_markers),
    )
```

`external/ObjectState/src/objectstate/subfield_semantics.py` must keep
`last_changed_field` out of `ObjectStateSubfieldSemantic`. ObjectState emits
stable dirty/default/inherited semantics only; pyqt-reactive owns transient
flash state.

Concrete `ObjectState` method draft:

```python
class ObjectState:
    def subfield_semantics(
        self,
        owner_field_path: DottedFieldPath,
    ) -> ObjectStateSubfieldSemanticIndex:
        self._check_and_sync_delegate()
        self._ensure_live_resolved()
        assert self._live_resolved is not None
        if not self._saved_resolved:
            self._saved_resolved = self._compute_resolved_snapshot(use_saved=True)

        raw_value = self.parameters.get(owner_field_path.value, MISSING)
        resolved_value = self.get_resolved_value(owner_field_path.value)
        saved_resolved_value = self.get_saved_resolved_value(owner_field_path.value)
        signature_default_value = self._signature_defaults.get(
            owner_field_path.value,
            MISSING,
        )

        return build_subfield_semantic_index(
            owner_field_path=owner_field_path,
            raw_value=raw_value,
            resolved_value=resolved_value,
            saved_resolved_value=saved_resolved_value,
            signature_default_value=signature_default_value,
            owner_dirty=self._path_or_descendant_in(
                owner_field_path.value,
                self.dirty_fields,
            ),
            owner_signature_diff=self._path_or_descendant_in(
                owner_field_path.value,
                self.signature_diff_fields,
            ),
        )
```

This keeps the semantic computation next to the raw/resolved/saved/default
authorities and prevents table widgets from comparing snapshots themselves. The
method accepts `DottedFieldPath`, not a path string; callers convert at existing
string API boundaries.

### Generic Traversal Rules

Implement traversal over values, not source-binding names:

- Dataclass instance: iterate `dataclasses.fields(type(value))` and read each
  field with `DataclassFieldAccess.raw_value(...)`.
- Tuple/list: iterate indexes in order. If comparing multiple containers, use
  the union of indexes present in raw, resolved, saved, and default values.
- Primitive/scalar: produce one leaf for the current path.
- Mapping: optional later slice. If added, use actual mapping keys as typed
  `MAPPING_KEY` segments.

Type derivation:

- When owner type annotation is `tuple[SourceFilterClause, ...]`, sequence
  element type is `SourceFilterClause`.
- When runtime values are present, validate that projected dataclass field names
  come from the dataclass type.
- Do not infer semantics from column title text.

Concrete traversal draft:

```python
def _leaf_values(value):
    if value is MISSING:
        return {}
    return dict(_iter_leaf_values(value, StructuralValuePath()))


def _iter_leaf_values(
    value,
    path: StructuralValuePath,
):
    if _is_dataclass_instance(value):
        for dataclass_field in dataclass_fields(type(value)):
            child_path = path.child_field(dataclass_field.name)
            child_value = DataclassFieldAccess.raw_value(value, dataclass_field.name)
            yield from _iter_leaf_values(child_value, child_path)
        return

    if isinstance(value, (tuple, list)):
        for index, child_value in enumerate(value):
            yield from _iter_leaf_values(child_value, path.child_index(index))
        return

    yield path, value


def _ordered_paths(
    *leaf_sets: Mapping,
) -> tuple[StructuralValuePath, ...]:
    ordered: list[StructuralValuePath] = []
    seen: set[StructuralValuePath] = set()
    for leaves in leaf_sets:
        for path in leaves:
            if path in seen:
                continue
            seen.add(path)
            ordered.append(path)
    return tuple(ordered)
```

For this slice, mapping traversal stays out of the implementation. If mapping
cells are needed later, they get a `MAPPING_KEY` segment in this same traversal
module; no table widget adds its own mapping-key semantic registry.

### pyqt-reactive Protocol

Add reusable inline dataclass infrastructure beside the existing
`ChildFieldChromeRefreshable` and `ChildFieldNavigationTargetProvider`
protocols.

File candidates:

- `external/pyqt-reactive/src/pyqt_reactive/forms/inline_dataclass_context.py`
- `external/pyqt-reactive/src/pyqt_reactive/forms/inline_dataclass_chrome.py`
- `external/pyqt-reactive/src/pyqt_reactive/widgets/structural_table.py`
- `external/pyqt-reactive/src/pyqt_reactive/protocols/widget_protocols.py`

#### Inline Dataclass Context

```python
InlineDataclassValueT = TypeVar("InlineDataclassValueT")
InlineDataclassChildValueT = TypeVar("InlineDataclassChildValueT")


@dataclass(frozen=True, slots=True)
class InlineDataclassChildFieldIdentity:
    object_state_path: DottedFieldPath
    manager_path: DottedFieldPath
    owner_type: type

    @property
    def field_name(self) -> str: ...


@dataclass(frozen=True, slots=True)
class InlineDataclassFormContext:
    state: ObjectState
    manager: ParameterFormManager
    owner_path: DottedFieldPath
    local_owner_path: DottedFieldPath
    owner_type: type
    color_scheme: ColorScheme | None
    scope_accent_color: QColor | None

    @classmethod
    def from_inline_widget(
        cls,
        *,
        manager: ParameterFormManager,
        param_info: InlineDataclassWidgetInfo,
        current_value: InlineDataclassValueT,
    ) -> "InlineDataclassFormContext": ...

    def child_identity(self, field_name: str) -> InlineDataclassChildFieldIdentity: ...
    def raw_child_value(self, field_name: str) -> InlineDataclassChildValueT | MissingValue: ...
    def resolved_child_value(self, field_name: str) -> InlineDataclassChildValueT | MissingValue: ...
    def saved_resolved_child_value(self, field_name: str) -> InlineDataclassChildValueT | MissingValue: ...
    def signature_default_child_value(self, field_name: str) -> InlineDataclassChildValueT | MissingValue: ...
    def child_has_inherited_preview(self, field_name: str) -> bool: ...
    def child_help_context(self, field_name: str) -> HelpContext: ...
    def reset_child(self, field_name: str) -> None: ...
    def child_semantic_index(self, field_name: str) -> ObjectStateSubfieldSemanticIndex: ...
```

This is the generic inline dataclass context for SourceBindings and any future
inline dataclass editors. The only OpenHCS-specific input is the dataclass
value/type being edited.

`reset_child(...)` must own the raw container lookup internally:

- read the raw owner container from `state.parameters[owner_path.value]`;
- read the child default from `state.signature_default(child_path.value)`;
- call `replace_raw(container_value, **{field_name: default_value})`;
- pass the replacement through `manager.update_parameter(local_owner_path.value, replacement)`.

Callers must not pass a container snapshot into `reset_child(...)`; that would
split reset ownership between the widget and the generic context.

#### Child Chrome Controller

```python
@dataclass(slots=True)
class InlineDataclassChildChrome:
    context: InlineDataclassFormContext
    labels: dict[InlineDataclassChildFieldIdentity, LabelWithHelp]
    groups: dict[InlineDataclassChildFieldIdentity, QWidget]
    reset_buttons: dict[InlineDataclassChildFieldIdentity, QPushButton]

    def child_identity(self, field_name: str) -> InlineDataclassChildFieldIdentity: ...
    def make_section_title(self, title: str, field_name: str) -> QWidget: ...
    def register_section_group(self, field_name: str, group: QWidget) -> None: ...
    def navigation_target(self, field_name: str) -> QWidget | None: ...
    def refresh_markers(self) -> None: ...
```

`SourceBindingsEditorWidget` should use this controller instead of maintaining
its own `section_labels`, `section_groups`, and `section_reset_buttons` maps.

#### Semantic Refresh Protocol

Add a nominal protocol beside `ChildFieldChromeRefreshable`:

File:
`external/pyqt-reactive/src/pyqt_reactive/protocols/widget_protocols.py`

```python
class ChildFieldSemanticChromeRefreshable(ABC):
    @abstractmethod
    def child_field_semantic_owner_paths(self) -> tuple[DottedFieldPath, ...]:
        """Return ObjectState owner paths for structural child semantics."""
        pass

    @abstractmethod
    def refresh_child_field_semantics(
        self,
        owner_field_path: DottedFieldPath,
        semantic_index: ObjectStateSubfieldSemanticIndex,
    ) -> None:
        """Refresh child/subfield dirty, default, inherited, and flash chrome."""
        pass
```

Import the semantic index type from ObjectState. The key is nominal dispatch,
not duck typing or erased payloads.

Replace erased navigation protocol returns with concrete widget targets:

```python
class ChildFieldNavigationTargetProvider(ABC):
    @abstractmethod
    def child_field_navigation_target(self, field_name: str) -> QWidget | None:
        pass


@dataclass(frozen=True, slots=True)
class StructuralTableCellTarget:
    table: QTableWidget
    row_index: int
    column_index: int
    cell_widget: QWidget | None


class ChildSubfieldNavigationTargetProvider(ABC):
    @abstractmethod
    def child_subfield_navigation_target(
        self,
        child_identity: InlineDataclassChildFieldIdentity,
        relative_path: StructuralValuePath,
    ) -> QWidget | StructuralTableCellTarget | None:
        pass
```

`external/pyqt-reactive/src/pyqt_reactive/protocols/widget_protocols.py`
currently returns `Any` for direct and subfield navigation targets. That is a
current implementation gap; the replacement above is the target signature.

Update:
`external/pyqt-reactive/src/pyqt_reactive/forms/parameter_form_chrome_sync.py`

Concrete chrome sync draft:

```python
def _owner_path(self, param_name: str) -> DottedFieldPath:
    manager = self.manager
    return DottedFieldPath(
        f"{manager.field_id}.{param_name}" if manager.field_id else param_name
    )


def update_label_styling(self, param_name: str) -> None:
    manager = self.manager
    dotted_path = self._owner_path(param_name)
    should_underline = self._path_or_descendant_in(
        dotted_path.value,
        manager.state.signature_diff_fields,
    )
    is_dirty = self._path_or_descendant_in(
        dotted_path.value,
        manager.state.dirty_fields,
    )

    if param_name in manager.labels:
        label = manager.labels[param_name]
        label.set_underline(should_underline)
        label.set_dirty_indicator(is_dirty)

    widget = manager.widgets.get(param_name)
    if isinstance(widget, GroupBoxWithHelp):
        widget.set_dirty_marker(is_dirty, should_underline)
    self._refresh_compound_widget_semantics(dotted_path, widget)


def _refresh_compound_widget_semantics(
    self,
    owner_path: DottedFieldPath,
    widget: QWidget | None,
) -> None:
    if isinstance(widget, ChildFieldChromeRefreshable):
        widget.refresh_child_field_chrome()
    if isinstance(widget, ChildFieldSemanticChromeRefreshable):
        for child_owner_path in widget.child_field_semantic_owner_paths():
            widget.refresh_child_field_semantics(
                child_owner_path,
                self.manager.state.subfield_semantics(child_owner_path),
            )
```

Concrete SourceBindings implementation draft:

```python
def child_field_semantic_owner_paths(self) -> tuple[DottedFieldPath, ...]:
    if self._form_context is None:
        return ()
    return (self._form_context.child_path("source_filters"),)


def refresh_child_field_semantics(
    self,
    owner_field_path: DottedFieldPath,
    semantic_index: ObjectStateSubfieldSemanticIndex,
) -> None:
    if self._form_context is None:
        return
    if owner_field_path != self._form_context.child_path("source_filters"):
        return
    if self.source_filters_controller is None:
        return
    self.source_filters_controller.apply_semantic_index(semantic_index)
```

Delete the current local refresh path:

```python
def refresh_child_field_chrome(self) -> None:
    self.refresh_section_label_markers()
    self.refresh_source_filter_cell_semantics()
```

Replace it with:

```python
def refresh_child_field_chrome(self) -> None:
    self.refresh_section_label_markers()
```

Then remove `refresh_source_filter_cell_semantics(...)`. SourceBindings should
not pull `state.subfield_semantics(...)` during section chrome refresh; generic
`ParameterFormChromeSync` pulls the semantic index through
`child_field_semantic_owner_paths()` and sends it back through
`refresh_child_field_semantics(...)`.

With this shape, pyqt-reactive does not know that SourceBindings has a
`source_filters` child. SourceBindings declares that typed child owner path
through its inline dataclass context, and ObjectState computes the semantics for
that path.

For a `source_filters` semantic owner path, the table cell relative paths are
`[0].subject`, `[0].match_type`, and `[0].value`.

In `refresh_widgets_for_paths(...)`, the owner-container branch should call
`self.update_label_styling(owner_field)` after raw value and resolved-preview
refresh. That single call refreshes the label, section chrome, and subfield
semantic index in order.

Update:
`external/pyqt-reactive/src/pyqt_reactive/forms/parameter_form_manager.py`

Concrete structural flash draft:

```python
def queue_inline_dataclass_subfield_flash(
    self,
    *,
    container_widget: QWidget,
    child_identity: InlineDataclassChildFieldIdentity,
    relative_path: StructuralValuePath,
) -> bool:
    if not isinstance(container_widget, ChildSubfieldNavigationTargetProvider):
        return False
    target = container_widget.child_subfield_navigation_target(
        child_identity,
        relative_path,
    )
    if isinstance(target, QWidget):
        flash_key = (
            f"{child_identity.manager_path.value}"
            f"{relative_path.display_suffix()}"
        )
        self.register_flash_widget_rect(flash_key, target)
        self.queue_flash_local(flash_key)
        return True
    if isinstance(target, StructuralTableCellTarget):
        flash_key = (
            f"{child_identity.manager_path.value}"
            f"{relative_path.display_suffix()}"
        )
        self.register_flash_table_cell_rect(flash_key, target)
        self.queue_flash_local(flash_key)
        return True
    return False
```

Concrete flash element draft for item-backed table cells:

```python
def create_table_cell_element(
    key: str,
    target: StructuralTableCellTarget,
) -> FlashElement:
    def get_rect(window: QWidget) -> QRect | None:
        table = target.table
        if not table.isVisible() or not table.isVisibleTo(window):
            return None
        model_index = table.model().index(target.row_index, target.column_index)
        if not model_index.isValid():
            return None
        cell_rect = table.visualRect(model_index)
        if cell_rect.isNull():
            return None
        global_pos = table.viewport().mapToGlobal(cell_rect.topLeft())
        window_pos = window.mapFromGlobal(global_pos)
        return QRect(window_pos, cell_rect.size())

    return FlashElement(
        key=key,
        get_rect_in_window=get_rect,
        source_id=(
            f"table_cell:{id(target.table)}:"
            f"{target.row_index}:{target.column_index}"
        ),
    )


def register_flash_table_cell_rect(
    self,
    key: str,
    target: StructuralTableCellTarget,
) -> None:
    self._register_flash_element_internal(
        key,
        lambda element_key: create_table_cell_element(element_key, target),
        target.table.viewport(),
    )
```

The existing `_queue_inline_dataclass_flash_for_path(...)` remains the direct
child-section path. Structural cell flashing uses the nominal child identity
plus `StructuralValuePath`; it does not parse a bracketed display string.

Concrete SourceBindings target draft:

```python
def child_subfield_navigation_target(
    self,
    child_identity: InlineDataclassChildFieldIdentity,
    relative_path: StructuralValuePath,
) -> QWidget | StructuralTableCellTarget | None:
    if self._form_context is None:
        return None
    if child_identity != self._form_context.child_identity("source_filters"):
        return None
    if self.source_filters_controller is None:
        return None
    return self.source_filters_controller.cell_target_for_semantic_path(relative_path)
```

### Table Cell Binding

Extract the existing local table controller into pyqt-reactive and extend it
there rather than adding a SourceBindings semantic registry.

File candidate:
`external/pyqt-reactive/src/pyqt_reactive/widgets/structural_table.py`

```python
@dataclass(frozen=True, slots=True)
class IsomorphicDataclassRowPathPolicy:
    row_value_type: type
    column_count: int

    def __post_init__(self) -> None:
        if not is_dataclass(self.row_value_type):
            raise TypeError(
                "IsomorphicDataclassRowPathPolicy requires a dataclass row type; "
                f"got {self.row_value_type!r}."
            )
        field_count = len(dataclass_fields(self.row_value_type))
        if self.column_count != field_count:
            raise ValueError(
                f"{self.row_value_type.__qualname__} has {field_count} fields "
                f"but the table declares {self.column_count} columns."
            )

    def relative_path_for_cell(
        self,
        row_index: int,
        column_index: int,
    ) -> StructuralValuePath:
        field_name = dataclass_fields(self.row_value_type)[column_index].name
        return (
            StructuralValuePath()
            .child_index(row_index)
            .child_field(field_name)
        )


@dataclass(frozen=True, slots=True)
class EditableTableSemanticBinding:
    owner_field_name: str
    row_path_policy: IsomorphicDataclassRowPathPolicy

    def relative_path_for_cell(
        self,
        row_index: int,
        column_index: int,
    ) -> StructuralValuePath:
        return self.row_path_policy.relative_path_for_cell(row_index, column_index)
```

For the first slice:

- `owner_field_name="source_filters"`
- `row_value_type=SourceFilterClause`, derived from the
  `SourceBindingsConfig.source_filters` annotation or from the row model
  declaration.
- The first path policy is isomorphic: column index maps to the dataclass field
  at the same index from `dataclass_fields(SourceFilterClause)`.
  - This derives `subject`, `match_type`, and `value` from the row dataclass,
    not from enum names or titles.
  - The policy validates `column_count=len(SourceFilterColumn)` against
    `len(dataclass_fields(SourceFilterClause))` in `__post_init__`.
- `SourceFilterColumn` may remain the display/editor column declaration, but it
  must not be the semantic authority for `subject`, `match_type`, or `value`
  unless it explicitly carries a typed `StructuralValuePath` declaration for a
  non-isomorphic cell.
- Header labels such as `Match_Type` or `Match Type` are presentation only.
  They must not be parsed, normalized, or compared to choose a semantic path.
- If a future table cell is not isomorphic, put a typed
  `StructuralValuePath` declaration on the existing column/editor declaration
  that already owns that displayed cell.

For future non-isomorphic cells, add path metadata to the existing value-editor
declaration that already owns the displayed column. Do not add a separate
semantic mirror. Example: a complex cell could declare that it represents
`selector.components`, but that declaration must live with the table/editor
column that already creates and parses the cell.

Concrete SourceFilters binding draft:

```python
self.source_filters_controller = EditableTableController(
    table=table,
    columns=tuple(SourceFilterColumn),
    free_form_cell_specs=self._free_form_cell_specs(),
    row_cells=EditableSourceFilterRow.cells,
    row_from_cells=EditableSourceFilterRow.from_cells,
    apply_changes=self._apply_source_filters_table,
    semantic_binding=EditableTableSemanticBinding(
        owner_field_name="source_filters",
        row_path_policy=IsomorphicDataclassRowPathPolicy(
            row_value_type=SourceFilterClause,
            column_count=len(SourceFilterColumn),
        ),
    ),
)
```

Because `ParameterFormChromeSync` passes the semantic index for the child owner
path declared by `child_field_semantic_owner_paths()` (
`source_bindings.source_filters` in the SourceBindings editor), the controller's
cell paths are relative to that owner field: `[0].subject`, `[0].match_type`,
`[0].value`. They must not include a leading `.source_filters` segment.

Add controller methods:

```python
class EditableTableController(Generic[EditableRowT]):
    def semantic_paths(self) -> Mapping[tuple[int, int], StructuralValuePath]: ...
    def apply_semantic_index(
        self,
        semantic_index: ObjectStateSubfieldSemanticIndex,
    ) -> None: ...
    def cell_target_for_semantic_path(
        self,
        relative_path: StructuralValuePath,
    ) -> StructuralTableCellTarget | None: ...
```

Concrete controller draft:

```python
def semantic_paths(self) -> Mapping[tuple[int, int], StructuralValuePath]:
    if self.semantic_binding is None:
        return MappingProxyType({})
    return MappingProxyType(
        {
            (row_index, column_index): self.semantic_binding.relative_path_for_cell(
                row_index,
                column_index,
            )
            for row_index in range(self.table.rowCount())
            for column_index, _column in enumerate(self.columns)
        }
    )


def apply_semantic_index(
    self,
    semantic_index: ObjectStateSubfieldSemanticIndex,
) -> None:
    for (row_index, column_index), relative_path in self.semantic_paths().items():
        semantic = semantic_index.leaf_for(relative_path)
        if semantic is None:
            continue

        widget = self.table.cellWidget(row_index, column_index)
        if widget is not None:
            self._apply_widget_semantic(widget, semantic)
            continue

        item = self.table.item(row_index, column_index)
        if item is not None:
            self._apply_item_semantic(item, semantic)


def cell_target_for_semantic_path(
    self,
    relative_path: StructuralValuePath,
) -> StructuralTableCellTarget | None:
    for (row_index, column_index), path in self.semantic_paths().items():
        if path != relative_path:
            continue
        return StructuralTableCellTarget(
            table=self.table,
            row_index=row_index,
            column_index=column_index,
            cell_widget=self.table.cellWidget(row_index, column_index),
        )
    return None


def _apply_item_semantic(
    self,
    item: QTableWidgetItem,
    semantic: ObjectStateSubfieldSemantic,
) -> None:
    logical_value = item.data(Qt.ItemDataRole.UserRole)
    if not isinstance(logical_value, str):
        logical_value = item.text()
        item.setData(Qt.ItemDataRole.UserRole, logical_value)
    item.setText(f"*{logical_value}" if semantic.dirty else logical_value)
    font = item.font()
    font.setUnderline(semantic.signature_diff)
    item.setFont(font)
    item.setToolTip(self._semantic_tooltip(semantic))


def _apply_widget_semantic(
    self,
    widget: QWidget,
    semantic: ObjectStateSubfieldSemantic,
) -> None:
    widget.setProperty("objectstate_dirty", semantic.dirty)
    widget.setProperty("objectstate_signature_diff", semantic.signature_diff)
    widget.setProperty("objectstate_inherited", semantic.inherited_value)
    widget.setToolTip(self._semantic_tooltip(semantic))
    font = widget.font()
    font.setUnderline(semantic.signature_diff)
    widget.setFont(font)
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()
```

The controller must not set `objectstate_last_changed` from table semantics.
A separate pyqt-reactive flash method may set `pyqt_reactive_recent_flash`
when the flash queue actually targets a cell.

Cell chrome should use dynamic Qt properties on the item or cell widget:

```text
objectstate_dirty=true
objectstate_signature_diff=true
objectstate_inherited=true
pyqt_reactive_recent_flash=true
```

For plain `QTableWidgetItem`, keep logical value and display chrome separate:

- Store the parseable logical cell text in a role such as
  `Qt.ItemDataRole.UserRole`.
- The visible display role may add a dirty marker if that is the chosen style.
  `_cell_text(...)` must read the logical role and must never parse a marker
  prefix.
- Underline/default-diff styling should use font underline, matching existing
  label/groupbox semantics.

For `QComboBox` and `StructuredSelectorCellWidget`, do not mutate the enum or
selector display text to carry semantic markers. Use one of these generic
mechanisms instead:

- dynamic properties on the widget plus stylesheet/delegate rules;
- a small marker label inside a reusable compound cell widget;
- a tooltip-only marker for the first implementation slice if visual styling is
  not yet available.

The read path for combo cells remains `currentData()`/enum value. The read path
for selector cells remains the selector widget's logical text API.

### Flash And Provenance Contract

Section-level navigation already uses
`ChildFieldNavigationTargetProvider.child_field_navigation_target(field_name)`.
Cell-level navigation uses the stricter descendant protocol drafted above:

```python
class ChildSubfieldNavigationTargetProvider(ABC):
    @abstractmethod
    def child_subfield_navigation_target(
        self,
        child_identity: InlineDataclassChildFieldIdentity,
        relative_path: StructuralValuePath,
    ) -> QWidget | StructuralTableCellTarget | None:
        pass
```

Concrete SourceFilters provenance/flash path:

```python
child_identity = self._form_context.child_identity("source_filters")
relative_path = (
    StructuralValuePath()
    .child_index(row_index)
    .child_field(field_name)
)
target = self.child_subfield_navigation_target(child_identity, relative_path)
```

Cell provenance opens the owner `DottedFieldPath`
`child_identity.object_state_path`, then uses `target` only for scroll/flash
chrome. It never writes a subfield ObjectState parameter and never opens an
unrelated step window.

### Marker Refresh Contract

`ParameterFormChromeSync` is the refresh authority. Its concrete behavior is
the `_refresh_compound_widget_semantics(...)` draft above:

- refresh the owner widget value/preview exactly as it does today;
- ask the compound widget for `child_field_semantic_owner_paths()`;
- build `ObjectState.subfield_semantics(child_owner_path)` for each declared
  child owner path;
- pass each index back to the same widget through
  `refresh_child_field_semantics(child_owner_path, semantic_index)`;
- let the table controller apply the semantic index to registered cell targets.

This keeps scalar fields, inline dataclass sections, and table cells on one
ObjectState-driven refresh path.

Tooltips should include the owner field path and the semantic reason, for
example:

```text
source_bindings.source_filters[0].value
* differs from saved value
_ differs from signature default
inherited from pipeline default
```

## SourceBindings Implementation Steps

1. Add or finish ObjectState nominal path support.
   - `DottedFieldPath` is the ObjectState dotted path authority.
   - Replace local string path construction at inline dataclass boundaries with
     `DottedFieldPath.child(...)`.
   - Keep `.value` conversion only at APIs that still require strings.

2. Add ObjectState structural subfield projection.
   - Implement generic dataclass and sequence traversal.
   - Add unit tests with a small nested dataclass containing
     `tuple[Clause, ...]`.
   - Do not import SourceBindings from ObjectState tests.

3. Finish and test generic inline dataclass context/chrome.
   - Keep `InlineDataclassFormContext`,
     `InlineDataclassChildFieldIdentity`, and `InlineDataclassChildChrome` in
     pyqt-reactive as the section-level authority.
   - Keep SourceBindings as a consumer of those classes.
   - Remove any remaining SourceBindings-local section maps that are not direct
     aliases to `InlineDataclassChildChrome`.
   - Do not keep string-keyed migration maps.
   - Before building granular cell semantics, add pyqt-reactive tests proving
     these classes are reusable and do not depend on SourceBindings.

4. Add pyqt-reactive semantic refresh protocol.
   - Extend `ParameterFormChromeSync` to pass semantic indexes to compound
     widgets.
   - Keep existing `ChildFieldChromeRefreshable` behavior for section-level
     labels.
   - Add focused tests with a fake compound widget that records the semantic
     index it receives.

5. Extract and extend `EditableTableController`.
   - Move generic typed table mechanics to pyqt-reactive.
   - Accept optional `semantic_binding`.
   - Derive per-cell `StructuralValuePath` from row index and the row path
     policy owned by the column/table declaration.
   - Apply semantic properties/tooltips after append, replace, and refresh.
   - Return the concrete cell widget/item target for flash/navigation.
   - Preserve logical values separately from visible marker text.

6. Implement granular Source Filters semantics.
   - Pass `EditableTableSemanticBinding(owner_field_name="source_filters", ...)`
     when creating `source_filters_controller`.
   - In `SourceBindingsEditorWidget.refresh_child_field_semantics(...)`, route
     the `source_filters` semantic index to `source_filters_controller`.
   - Keep section label markers by calling `refresh_section_label_markers()`.

7. Generalize to sibling SourceBindings tables only after Source Filters is
   green.
   - `metadata_rules`: row type `MetadataExtractionRule`.
   - `match_plan`: row type `SourceBindingMatchDimension` is not one row per
     field today, so defer until the generic binding can describe non-isomorphic
     row models without a mirror.
   - `bindings`: the modal `StepBindingsTableEditor` has nested selector cells;
     defer until structural path metadata can point through `NamedSourceBinding`
     and `SourceSelector` without duplicating selector semantics.

## Dry Run: SourceFilterClause Cells

Assume:

```python
saved_resolved source_bindings.source_filters =
(
    SourceFilterClause(
        subject=SourceFilterSubject.FILE,
        match_type=SourceFilterMatchType.CONTAINS,
        value="DAPI",
    ),
)

signature default for StepSourceBindingsConfig.source_filters = None
```

### State A: Fully Inherited

Raw `source_bindings.source_filters` is `None`; resolved value is the saved row.

| Cell | Display path | Dirty | Default diff | Inherited |
| --- | --- | --- | --- | --- |
| subject | `source_bindings.source_filters[0].subject` | false | false | true |
| match_type | `source_bindings.source_filters[0].match_type` | false | false | true |
| value | `source_bindings.source_filters[0].value` | false | false | true |

UI result:

- Cells show the resolved values.
- Cells are dimmed/placeholder-styled as inherited.
- Section may still show inherited preview styling, but there is no `*`.

### State B: User Changes Only `value`

Raw/source editor writes:

```python
(
    SourceFilterClause(
        subject=SourceFilterSubject.FILE,
        match_type=SourceFilterMatchType.CONTAINS,
        value="GFP",
    ),
)
```

| Cell | Dirty | Default diff | Inherited | Reason |
| --- | --- | --- | --- | --- |
| subject | false | true | false | Explicit raw cell equals saved resolved cell, but the default owner field had no row. |
| match_type | false | true | false | Explicit raw cell equals saved resolved cell, but the default owner field had no row. |
| value | true | true | false | Live resolved value `GFP` differs from saved resolved value `DAPI`. |

UI result:

- Only `value` gets the dirty `*` marker/flash.
- All three cells may carry default-diff chrome because the row is now a
  concrete override of a `None` lazy default.
- Tooltips distinguish "differs from saved" from "differs from signature
  default" so explicit ownership does not look like an accidental value edit.

### State C: User Changes `match_type` to `is_image`

`SourceFilterClause.__post_init__` normalizes `value` to `None` when
`match_type.requires_value` is false.

| Cell | Dirty | Default diff | Inherited | Reason |
| --- | --- | --- | --- | --- |
| subject | false | true | false | Same value as saved, explicit override row. |
| match_type | true | true | false | `is_image` differs from saved `contains`. |
| value | true | true | false | Resolved `None` differs from saved `DAPI`; missing is distinct from `None`. |

UI result:

- `match_type` and `value` get dirty markers.
- `value` tooltip should say the resolved value is `None`, not that the cell is
  missing.

## Files To Edit

Expected implementation files:

- `external/ObjectState/src/objectstate/subfield_semantics.py`
- `external/ObjectState/src/objectstate/object_state.py`
- `external/ObjectState/src/objectstate/__init__.py`
- `external/pyqt-reactive/src/pyqt_reactive/forms/inline_dataclass_context.py`
- `external/pyqt-reactive/src/pyqt_reactive/forms/inline_dataclass_chrome.py`
- `external/pyqt-reactive/src/pyqt_reactive/widgets/structural_table.py`
- `external/pyqt-reactive/src/pyqt_reactive/protocols/widget_protocols.py`
- `external/pyqt-reactive/src/pyqt_reactive/protocols/__init__.py`
- `external/pyqt-reactive/src/pyqt_reactive/forms/parameter_form_chrome_sync.py`
- `external/pyqt-reactive/src/pyqt_reactive/forms/parameter_form_manager.py`
- `openhcs/pyqt_gui/widgets/source_bindings_editor.py`

Optional agent-facing files, only if MCP/UI bridge surfaces need granular cell
semantics in the same implementation batch:

- `openhcs/agent/dto/ui_bridge.py`
- `openhcs/pyqt_gui/services/ui_bridge_object_state.py`
- `openhcs/agent/services/object_state_field_projection.py`
- `openhcs/mcp/dev_client_renderers/object_state.py`

Do not edit compiler/runtime SourceBindings behavior for this plan.

## Tests To Add

### ObjectState

File:
`external/ObjectState/tests/test_subfield_semantics.py`

Tests:

- `test_tuple_dataclass_leaf_semantics_detect_only_changed_leaf`
  - Dataclass with `clauses: tuple[Clause, ...] | None`.
  - Saved resolved row has `value="DAPI"`.
  - Live resolved row has `value="GFP"`.
  - Assert only `[0].value` is dirty.
- `test_missing_default_leaf_marks_signature_diff_without_confusing_none`
  - Signature default is `None`.
  - Raw row exists.
  - Assert row cells are default diff and `value=None` is present, not missing.
- `test_inherited_tuple_cells_are_marked_inherited`
  - Raw owner field is `None`, resolved owner field has rows.
  - Assert all resolved leaves are inherited and not dirty when saved resolved
    matches.

### pyqt-reactive

File:
`external/pyqt-reactive/tests/test_child_subfield_semantic_chrome.py`

Tests:

- `InlineDataclassFormContext.from_inline_widget(...)` produces owner and child
  `DottedFieldPath` identities without string concatenation in widget code.
- `InlineDataclassChildChrome` builds label/reset/provenance controls and
  refreshes dirty/signature/inherited markers from ObjectState.
- Fake inline dataclass widget implements the new semantic refresh protocol.
- `ParameterFormChromeSync.state_changed()` passes a semantic index for the
  owner field.
- `refresh_widgets_for_paths({"config.clauses"})` refreshes both value preview
  and semantic index.
- Existing `ChildFieldChromeRefreshable` tests still pass without implementing
  the new protocol.
- Structural table controller keeps logical values parseable when visible dirty
  markers are applied.
- Combo-box cells receive semantic properties without changing enum display text
  or current data.

### OpenHCS SourceBindings UI

File:
`tests/unit/pyqt_gui/test_source_bindings_editor.py`

Tests:

- Source Filters inherited row:
  - Construct `SourceBindingsEditorWidget` through the normal inline dataclass
    factory so it receives an `InlineDataclassFormContext` whose raw child value
    is `None` and resolved child value has one `SourceFilterClause`.
  - Assert `subject`, `match_type`, and `value` cells have
    `objectstate_inherited=true` and no dirty marker.
- Source Filters value edit:
  - Saved/resolved row has `value="DAPI"`.
  - Edit only the `value` cell to `GFP`.
  - Assert only the value cell has `objectstate_dirty=true`.
  - Assert the row cells can still carry default-diff chrome independently.
- Source Filters match type edit:
  - Change `match_type` to `IS_IMAGE`.
  - Assert `match_type` and `value` are dirty after normalization.
- Navigation/flash:
  - `child_subfield_navigation_target("source_filters", [0].value)` returns the
    concrete cell widget or table viewport rect target.
- Time travel:
  - Undo/redo of `source_filters[0].match_type` refreshes only the owner
    SourceBindings widget and marks the affected cell path, without opening
    unrelated step windows.

### Agent/UI Bridge, If Exposed

File:
`tests/unit/agent/test_object_state_field_projection.py` or existing
`tests/unit/agent/test_ui_bridge_service.py`

Tests:

- Field list queries can include subfield projections without replacing the
  existing field-level summaries.
- Filtering for semantic subfields uses the same dirty/default/inherited
  predicates as field-level filtering.
- `include_values=False` still returns bounded previews, not full containers.

## Completion Gates

- The Source Filters section can still show section-level markers, but the
  table also marks individual cells.
- Cell semantics come from ObjectState structural projections; the table widget
  does not compare saved/default values itself.
- No new SourceBindings-specific semantic registry exists.
- No enum display text or header label text is used as a semantic path.
- Combo-box and selector-cell readback returns clean logical values after
  visible markers are applied.
- Click-to-provenance on a Source Filters cell opens or focuses the right
  ObjectState owner window and scrolls/flashes the relevant cell or, when no
  cell target exists, the direct child section target.
- Undo/redo of a Source Filters cell change refreshes the visible cell state
  without opening unrelated step windows.
- After implementation,
  `rg -n "source_filters.*subject|SourceFilterColumn.*semantic" openhcs external -g '*.py'`
  should not find a new hardcoded semantic mirror.
- Existing SourceBindings row parsing and `replace_raw(...)` update ownership
  remain unchanged.
- Existing placeholder behavior remains intact for normal form fields.
- The tests above pass without requiring runtime/compiler changes.

## Open Questions

- Should default-diff chrome on explicit rows use the same underline as scalar
  fields, or should tables distinguish "explicit override" from "value changed
  relative to saved" more strongly?
- Should agent-facing ObjectState field-list output include subfields by default
  for semantic fields, or only behind an explicit `include_subfields` option?
- For future `bindings` rows, should `NamedSourceBinding.alias` become an
  optional display key for row labels while sequence index remains the actual
  writable structural identity?
