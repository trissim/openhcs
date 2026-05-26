# Dual Editor / Config Window PyQt-Reactive Refactor Plan - 2026-05-26

## Goal

Make `DualEditorWindow` and `ConfigWindow` clients of a shared, nominal
`pyqt-reactive` form-window architecture instead of two OpenHCS windows that
each partially reimplement form lifecycle, dirty state, save/cancel behavior,
scope styling, and responsive action layout.

The target is not a generic "window superclass" that absorbs OpenHCS domain
logic. The target is a load-bearing `pyqt-reactive` form-session spine:

- explicit form-window contracts instead of `hasattr` / `getattr` probing;
- one save/cancel/dirty lifecycle template;
- reusable responsive action header and tab action-strip widgets;
- OpenHCS-specific step/config behavior kept in OpenHCS session objects.

## Verified Current State

### Shared Behavior Already Present, But Not Nominal

`external/pyqt-reactive/src/pyqt_reactive/widgets/shared/base_form_dialog.py`
already owns pieces of the shared lifecycle:

- singleton-per-scope show behavior through `WindowManager`;
- Shift+Save support through `_setup_save_button`;
- save-without-close through `_mark_saved_and_refresh_all`;
- cancel/close restoration through `state.restore_saved`;
- automatic dirty detection by connecting discovered form managers.

The problem is that the base class discovers its contract structurally:

- `getattr(self, "scope_id", None)` for window identity;
- `hasattr(self, "state")` for ObjectState lifecycle;
- `hasattr(self, "form_manager")`, `hasattr(self, "step_editor")`, and
  `hasattr(self, "config_editor")` for form-manager discovery;
- `hasattr(self, "detect_changes")` before invoking dirty detection.

The advisor flags this as semantic role recovery from attribute probing. That
is correct: the shared layer knows the shape it needs, but does not declare it.

### `DualEditorWindow` Is A Product Of Multiple Roles

`openhcs/pyqt_gui/windows/dual_editor_window.py` currently carries at least
these roles in one class:

- step editing session: clone/new/original step handling;
- form-window lifecycle: dirty detection, save button state, title marker,
  accept/reject/close cleanup;
- tab shell: header row, tab row, active tab action buttons, stacked widget
  extraction;
- scope presentation: border initialization, accent save button, tab bar,
  tree selection, child widget color schemes;
- step form creation: `StepParameterEditorWidget` construction under OpenHCS
  config context;
- function-pattern editor synchronization;
- artifact preview refresh;
- global event-bus subscription for pipeline/config updates;
- time-travel title/function refresh;
- field navigation dispatch across step/function tabs.

The advisor reports a high method-role quotient on `DualEditorWindow` and
attribute probing. This is not a cosmetic issue: the class is acting as facade,
session model, tab host, style presenter, and OpenHCS synchronization service
at once.

### `ConfigWindow` Shares The Same Lifecycle Shape

`openhcs/pyqt_gui/windows/config_window.py` independently implements behavior
that overlaps with `DualEditorWindow`:

- `changes_detected` signal;
- `has_changes` state;
- `detect_changes`;
- save button enablement;
- title/header dirty marker and signature-diff underline;
- default size and resize logging;
- scope-accent save button, header, and tree selection styling;
- save with `close_window=True/False`;
- reject/close ObjectState cleanup and refresh.

It also has OpenHCS-specific behavior that should not move upstream:

- global vs pipeline config semantics;
- saved/live thread-local global config updates;
- code editor round-trip through `pycodify`;
- lazy config constructor patching;
- config hierarchy tree semantics.

### Existing `pyqt-reactive` Widgets Are Close But Incomplete

Useful upstream pieces already exist:

- `BaseFormDialog` / `BaseManagedWindow`;
- `ScopedBorderMixin`;
- `ResponsiveTwoRowWidget`;
- `StagedWrapLayout`;
- `TabbedFormWidget`;
- `ButtonPanel`;
- `ConfigHierarchyTreeHelper`;
- `CollapsibleSplitterHelper`;
- `ParameterFormManager`;
- `FunctionListEditorWidget`.

The missing abstraction is not another form manager. It is a form-window
session contract that composes these widgets and owns the lifecycle.

## Advisor Evidence To Resolve

The focused advisor pass over `DualEditorWindow`, `ConfigWindow`, and the
relevant `pyqt-reactive` shared modules reports these material findings:

- `BaseManagedWindow._discover_form_managers` recovers `form_manager`,
  `step_editor.form_manager`, and `config_editor.form_manager` by attribute
  probing.
- `BaseManagedWindow.accept`, `reject`, `closeEvent`, and
  `_mark_saved_and_refresh_all` probe for `state`.
- `BaseManagedWindow._get_window_scope_key` probes for `scope_id`.
- `BaseManagedWindow._on_parameter_changed_for_change_detection` probes for
  `detect_changes`.
- `DualEditorWindow` has a high method-role quotient and attribute probes.
- `ConfigWindow._StagedButtonWrap` overlaps with `StagedWrapLayout` in
  `pyqt-reactive`.
- `BaseManagedWindow._create_compact_header` appears to be dead private
  residue.

These findings point to one architectural conclusion: fix the upstream
form-window spine first, then shrink OpenHCS windows against that spine.

## Target Architecture

### New Upstream Contract: `ManagedFormWindow`

Add an ABC or mixin contract in `pyqt-reactive`, owned near
`widgets/shared/base_form_dialog.py`.

Conceptual shape:

```python
class ManagedFormWindowABC(ABC):
    @property
    @abstractmethod
    def scope_id(self) -> str | None: ...

    @property
    @abstractmethod
    def state(self) -> ObjectState | None: ...

    @property
    def restore_descendants_on_close(self) -> bool:
        return True

    @abstractmethod
    def form_managers(self) -> tuple[ParameterFormManager, ...]: ...

    @abstractmethod
    def detect_changes(self) -> None: ...
```

`BaseFormDialog` should inherit from or delegate to this contract. It should
call the contract directly. It should not scan for `form_manager`,
`step_editor`, or `config_editor` by string.

The recursive nested-manager traversal can remain in `pyqt-reactive`, but it
must operate on the explicit `form_managers()` return value.

### New Upstream Presenter: Dirty Form Window State

Introduce a small reusable state/presenter boundary:

```python
@dataclass(frozen=True)
class DirtyWindowState:
    base_title: str
    is_dirty: bool
    has_signature_diff: bool
    save_label: str
```

And a presenter:

```python
class DirtyWindowPresenter:
    def apply(
        self,
        *,
        window: QDialog,
        header_label: QLabel,
        save_button: QPushButton,
        state: DirtyWindowState,
    ) -> None: ...
```

This replaces duplicated dirty-title and save-button enablement logic. The
OpenHCS windows still compute domain-specific base titles:

- `ConfigWindow`: `Configure {config_class.__name__}`;
- `DualEditorWindow`: `New/Edit Step: {current step name}`.

The common presenter owns:

- `* ` dirty prefix;
- signature-diff underline;
- save button enabled state;
- optional dirty prefix on save label.

### New Upstream Widget: `FormWindowActionHeader`

Create a reusable responsive action header in `pyqt-reactive`.

Responsibilities:

- title widget;
- ordered action groups;
- save/cancel buttons;
- optional reset/code/help groups;
- right alignment for save/cancel group;
- content-based wrapping.

This should be built on `StagedWrapLayout`, not a new ad hoc layout. After it
exists, delete `ConfigWindow._StagedButtonWrap` if it is still unused, and use
the upstream header in both windows where practical.

### New Upstream Widget: `ActionTabbedWindowBody`

`DualEditorWindow` has a useful generic layout: tabs on the left, currently
active tab's action widgets on the right, content below. This is not OpenHCS
specific.

Add an upstream tab shell that accepts records like:

```python
@dataclass(frozen=True)
class ActionTabSpec:
    label: str
    content: QWidget
    actions: QWidget | None = None
```

Responsibilities:

- own `QTabWidget`;
- render tab bar in a responsive row;
- show only the active tab's action widget;
- expose current-tab change signal;
- avoid parent code fishing the internal `QStackedWidget` out of `QTabWidget`.

`DualEditorWindow` should use this for Step Settings, Function Pattern, and
Artifacts. The artifact tab likely has no action widget.

### OpenHCS Session Objects

Keep OpenHCS semantics outside `pyqt-reactive` in explicit session objects.

#### `DualEditorSession`

Owns:

- new-step creation;
- editing clone;
- original-step reference;
- save target selection;
- validation of step name;
- applying saved values to original;
- stable step scope id calculation inputs.

It should replace the current spread across:

- `__init__`;
- `_clone_step`;
- `_create_new_step`;
- `_apply_changes_to_original`;
- chunks of `save_edit`.

Important: avoid reflecting over arbitrary attributes in
`_apply_changes_to_original`. `FunctionStep` is a dataclass in the current
usage path. If there is a non-dataclass step type, it should have a declared
copy/apply protocol. Do not keep `dir(...)` fallback copying as the abstraction.

#### `DualEditorFunctionSync`

Owns:

- canonicalizing function patterns if needed;
- syncing `FunctionListEditorWidget.current_pattern` into `editing_step.func`
  and `ObjectState`;
- batching refresh through a timer;
- artifact-preview refresh request emission.

The current silent fallbacks around import refresh (`except Exception: pass`) are
not acceptable as a generic abstraction. Either the canonicalizer returns a
typed "unresolved but still usable" result with provenance, or it fails loud.

#### `DualEditorEventBridge`

Owns OpenHCS event bus subscription and filtering:

- resolve event bus from explicit service adapter;
- subscribe/unsubscribe;
- route pipeline changes by scope token;
- route config changes for matching global/pipeline config.

This is OpenHCS-specific and should not move to `pyqt-reactive`.

#### `ConfigEditSession`

Owns:

- current config class/object;
- global vs pipeline config distinction;
- saved/live global config updates;
- code editor update semantics;
- cancel-time global context restoration.

This keeps the upstream save lifecycle generic while preserving OpenHCS
config-framework invariants.

## Phased Implementation Plan

### Phase 0: Characterization

Add or verify focused tests before moving behavior:

- `BaseFormDialog` no longer uses attribute probing for form managers, state,
  scope id, or dirty detection.
- `ConfigWindow` save button enables/disables from `ObjectState.is_raw_dirty`.
- `ConfigWindow` header gets dirty prefix and signature-diff underline.
- `DualEditorWindow` save button/title update from the step ObjectState.
- `DualEditorWindow` cancel restores step ObjectState and increments registry
  token.
- Shift+Save still marks saved and keeps the window open.
- Closing via X still restores saved state and unregisters the window.
- Dual editor tab actions switch when active tab changes.

Use Qt offscreen tests where needed.

### Phase 1: Upstream Nominal Form-Window Contract

In `external/pyqt-reactive`:

1. Add `ManagedFormWindowABC` or equivalent protocol/ABC.
2. Change `BaseFormDialog` to call:
   - `self.scope_id`;
   - `self.state`;
   - `self.restore_descendants_on_close`;
   - `self.form_managers()`;
   - `self.detect_changes()`.
3. Delete `BaseFormDialog._discover_form_managers` structural branches.
4. Keep nested-manager recursion, but make the input explicit.
5. Delete or make public/used `_create_compact_header`.

In OpenHCS:

1. Implement `form_managers()` on `ConfigWindow`.
2. Implement `form_managers()` on `DualEditorWindow`, returning the step editor
   form manager once available.
3. Replace `_restore_descendants_on_close` attribute probing with a direct
   property override on `ConfigWindow`.

Verification:

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/pytest tests/unit/pyqt_gui/test_main_config_propagation.py tests/unit/pyqt_gui/test_dual_editor_window_artifact_refresh.py -q
nominal-refactor-advisor external/pyqt-reactive/src/pyqt_reactive/widgets/shared/base_form_dialog.py openhcs/pyqt_gui/windows/config_window.py openhcs/pyqt_gui/windows/dual_editor_window.py
```

Expected advisor result: `base_form_dialog.py` no longer reports structural
attribute probing for state/scope/form-manager/detect_changes.

### Phase 2: Dirty/Save Presenter

In `pyqt-reactive`:

1. Add `DirtyWindowState`.
2. Add `DirtyWindowPresenter`.
3. Add a small helper on `BaseFormDialog`:

   ```python
   def dirty_window_state(self) -> DirtyWindowState: ...
   def apply_dirty_window_state(self) -> None: ...
   ```

   The base class may define the template, but title construction should remain
   a subclass hook.

In OpenHCS:

1. Replace `ConfigWindow._update_window_title_dirty_marker` with state-building
   logic plus presenter call.
2. Replace `DualEditorWindow._update_window_title` and
   `_update_save_button_text` with state-building logic plus presenter call.
3. Keep OpenHCS-specific title text construction in each window.

Verification:

- Existing dirty marker tests.
- Manual/offscreen smoke for dirty prefix and underline.
- Advisor should no longer report duplicated dirty-title skeletons.

### Phase 3: Responsive Action Header

In `pyqt-reactive`:

1. Add `FormWindowActionHeader`.
2. Use `StagedWrapLayout` internally.
3. Support action groups with stable IDs and returned button references.
4. Centralize Shift+Save button wiring by accepting a save action spec.

In OpenHCS:

1. Replace `ConfigWindow` header assembly with `FormWindowActionHeader`.
2. Replace `DualEditorWindow` title header assembly with
   `FormWindowActionHeader`.
3. Delete `ConfigWindow._StagedButtonWrap` if it remains unused.

Do not push OpenHCS-specific reset/code/help semantics into the upstream
header. The upstream header should render actions; `ConfigWindow` owns what
those actions mean.

Verification:

- Qt offscreen import/smoke for both windows.
- Visual smoke if practical.
- Advisor should no longer flag `_StagedButtonWrap._row_width` overlap.

### Phase 4: Action Tab Shell

In `pyqt-reactive`:

1. Add `ActionTabbedWindowBody`.
2. It should own tab row, active action widget switching, and content area.
3. It should expose:
   - `add_tab(ActionTabSpec)`;
   - `set_current_index`;
   - `current_changed`;
   - `current_widget`;
   - `tab_widget` only if needed for compatibility.

In OpenHCS:

1. Replace `DualEditorWindow` tab-row construction and
   `_setup_tab_button_containers` / `_show_tab_buttons` with this shell.
2. Stop retrieving `QStackedWidget` from `QTabWidget` internals.
3. Leave tab creation in OpenHCS for now: step editor, function editor, and
   artifact preview are domain tabs.

Verification:

- Existing dual editor window tests.
- Add a small upstream widget test proving active action widget switches.
- Advisor should reduce `DualEditorWindow` tab/setup role findings.

### Phase 5: OpenHCS Dual Editor Session Decomposition

Add OpenHCS-side modules, likely under `openhcs/pyqt_gui/widgets/shared/services`
or `openhcs/pyqt_gui/windows/services`:

- `dual_editor_session.py`
- `dual_editor_function_sync.py`
- `dual_editor_event_bridge.py`

Move behavior out of `DualEditorWindow`:

- clone/new/original/save target into `DualEditorSession`;
- function pattern sync and artifact refresh events into
  `DualEditorFunctionSync`;
- service-adapter/event-bus lookup and subscriptions into
  `DualEditorEventBridge`.

`DualEditorWindow` remains responsible for:

- constructing widgets;
- connecting Qt signals to services;
- exposing public navigation methods used by time travel/window manager;
- presenting message boxes for domain validation errors.

Verification:

- Existing dual editor tests.
- New pure tests for `DualEditorSession` without Qt.
- Advisor should reduce `DualEditorWindow` method-role quotient.

### Phase 6: Config Session Decomposition

Add `ConfigEditSession` in OpenHCS:

- construct/reuse `ObjectState`;
- decide `restore_descendants_on_close`;
- track global context dirty state;
- apply saved global config updates;
- restore global edit context on cancel;
- own `state.to_object()` save extraction.

`ConfigWindow` remains responsible for:

- tree widget;
- form manager widget;
- code editor launch;
- message boxes;
- Qt signal emission.

Verification:

- global config propagation tests;
- pipeline config cancel/restore tests;
- code editor round-trip tests if available.

### Phase 7: Final Advisor And Test Gates

Run:

```bash
python -m py_compile \
  external/pyqt-reactive/src/pyqt_reactive/widgets/shared/base_form_dialog.py \
  external/pyqt-reactive/src/pyqt_reactive/widgets/shared/form_window_action_header.py \
  external/pyqt-reactive/src/pyqt_reactive/widgets/shared/action_tabbed_window_body.py \
  openhcs/pyqt_gui/windows/config_window.py \
  openhcs/pyqt_gui/windows/dual_editor_window.py

QT_QPA_PLATFORM=offscreen .venv/bin/pytest tests/unit/pyqt_gui -q
QT_QPA_PLATFORM=offscreen .venv/bin/pytest external/pyqt-reactive/tests -q

nominal-refactor-advisor \
  external/pyqt-reactive/src/pyqt_reactive/widgets/shared/base_form_dialog.py \
  external/pyqt-reactive/src/pyqt_reactive/widgets/shared/responsive_layout_widgets.py \
  openhcs/pyqt_gui/windows/config_window.py \
  openhcs/pyqt_gui/windows/dual_editor_window.py
```

The final state is acceptable only if:

- `BaseFormDialog` has no role-recovery attribute probing findings;
- `DualEditorWindow` no longer reports the same high control-hub quotient;
- `ConfigWindow` has no local staged-layout duplication;
- OpenHCS tests pass under offscreen Qt;
- pyqt-reactive tests pass.

## What Must Not Move Upstream

Do not move these to `pyqt-reactive`:

- `FunctionStep`;
- `PipelineConfig` / `GlobalPipelineConfig`;
- config context manager semantics;
- lazy placeholder services;
- source-binding context;
- artifact contract preview;
- OpenHCS event bus;
- pycodify config-code round trip;
- CellProfiler/source schema behavior;
- orchestrator-specific scope construction.

If an abstraction needs any of those names in its constructor, it belongs in
OpenHCS, not upstream.

## Critical Review Of The Plan

### Architecture Strengths

- The first move fixes the actual upstream smell: `BaseFormDialog` is already
  acting like an abstract form-window lifecycle, but without a nominal
  contract.
- The plan avoids a fake generic `EditorWindow` that would need OpenHCS types.
- The plan extracts UI lifecycle before domain sessions, which prevents
  `DualEditorSession` from becoming a bucket for Qt layout concerns.
- The proposed upstream widgets are reusable outside OpenHCS:
  action headers, dirty presenters, and tab action shells are generic UI
  infrastructure.
- The OpenHCS session objects have crisp domain ownership.

### Architecture Risks

- If `ManagedFormWindowABC` is implemented as a rigid inheritance base, it may
  fight Qt multiple inheritance and existing mixins. Prefer a small ABC plus
  `BaseFormDialog` implementation hooks, or a runtime-checkable protocol where
  direct calls still fail loud.
- `DirtyWindowPresenter` must not become a theme mega-object. It should only
  apply title/header/save dirty state.
- `ActionTabbedWindowBody` must not assume every tab has a `ParameterFormManager`.
  The artifact tab is read-only and should remain valid.
- `DualEditorFunctionSync` is a correctness-sensitive area. The current import
  refresh fallback is smelly; replacing it needs a typed result or fail-loud
  error, not another try/except wrapper.
- `ConfigEditSession` must preserve global live/saved context semantics exactly.
  That needs characterization before extraction.

### Rejected Alternatives

- **Move all of `DualEditorWindow` into `pyqt-reactive`.** Rejected because most
  of the class is OpenHCS-specific step/function/orchestrator behavior.
- **Make `ConfigWindow` and `DualEditorWindow` inherit from a new OpenHCS
  superclass.** Rejected because the duplicated lifecycle belongs to the shared
  library and the advisor finding is already in `pyqt-reactive`.
- **Keep `BaseFormDialog` structural and only extract OpenHCS services.**
  Rejected because it leaves the shared layer recovering semantics from field
  names.
- **Generalize `TabbedFormWidget` until it can host the dual editor.** Rejected
  because the dual editor has non-form tabs and tab-level actions. A separate
  action-tab shell is cleaner.

### Optimality Check

This is the best decomposition because each abstraction pays rent:

- `ManagedFormWindowABC` removes structural probing and gives the lifecycle a
  declared contract.
- `DirtyWindowPresenter` removes duplicated state presentation without knowing
  OpenHCS domain objects.
- `FormWindowActionHeader` replaces local layout code and standardizes save
  affordances.
- `ActionTabbedWindowBody` removes tab/action plumbing without owning tab
  semantics.
- `DualEditorSession` isolates step identity/save semantics.
- `ConfigEditSession` isolates config-framework context semantics.

The only pieces that move upstream are those that could be used by another
PyQt application with ObjectState-backed forms. Everything that needs OpenHCS
domain names stays local.

## Completion Criteria

- `DualEditorWindow` reads as a thin facade over:
  - `DualEditorSession`;
  - `DualEditorFunctionSync`;
  - `DualEditorEventBridge`;
  - upstream form-window widgets/presenters.
- `ConfigWindow` reads as a thin facade over:
  - `ConfigEditSession`;
  - upstream form-window lifecycle/header/presenter;
  - config hierarchy tree and code editor actions.
- `BaseFormDialog` has a declared contract and no stringly self-probing.
- Advisor findings for the touched modules are either resolved or explicitly
  documented as remaining broader work.
- Tests cover behavior before and after each extraction phase.
