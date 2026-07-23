# CellProfiler Binding Context Cleanup Plan

## Goal

Resolve or deliberately retire the remaining advisor pressure around
`ThresholdSettingsBindingRequest` and `ColorToGrayModuleBinding` without
creating fake inheritance.

Current finding:

- `ThresholdSettingsBindingRequest` repeats semantic fields `module` and
  `binder` already present on `ColorToGrayModuleBinding`.

Prior experiment:

- A neutral shared base for the two classes made advisor output worse by adding
  a new metaclass-family finding.
- Direct inheritance from `ColorToGrayModuleBinding` would be semantically wrong:
  threshold binding is not a ColorToGray binding family.

## Verified Current Seams

### ColorToGray Binding Family

`openhcs/interop/cellprofiler/color_to_gray_settings.py` contains:

- `ColorToGrayModuleBinding`
- `ColorToGrayModePlan`
- `ColorToGrayOutputPlan`
- `ColorToGrayContributionPlan`

This family is specific to ColorToGray module semantics and is registered by
`AutoRegisterMeta`.

### Threshold Binding Request

`openhcs/interop/cellprofiler/module_settings_binding.py` contains:

- `ThresholdSettingsBindingRequest`
- `_parse_cellprofiler_threshold_setting(...)`
- `_threshold_scope(...)`
- `_upgrade_legacy_cellprofiler_threshold_kwargs(...)`
- `ThresholdModuleSettingsBindingStrategy`

This request carries mutable kwargs/unmapped-kwargs state in addition to
`module` and `binder`.

### Settings Binder

`SettingsBinder` is the typed parser authority for CP string settings. It is
not itself a module-context record and should not grow module lifecycle state.

## Architectural Rule

Do not make `ThresholdSettingsBindingRequest` inherit from
`ColorToGrayModuleBinding`.

The shared fields are incidental:

- both need a parsed `ModuleBlock`;
- both need a `SettingsBinder`;
- they belong to different module-specific binding domains.

A shared abstraction is valid only if it owns a real CP binding invariant:

- module identity,
- binder access,
- setting lookup,
- coverage/unmapped-setting mutation,
- fail-loud parse context,
- or provenance for error messages.

## Candidate Target Shapes

### Option A: Module Binding Context Value

Create a small non-registered context value:

```python
@dataclass(frozen=True, slots=True)
class ModuleSettingsBindingContext:
    module: ModuleBlock
    binder: SettingsBinder

    def optional_value(self, setting: str | SettingNameFamily) -> str | None:
        ...
```

Use it by composition:

- `ColorToGrayModuleBinding.context`
- `ThresholdSettingsBindingRequest.context`

This avoids false semantic inheritance but may increase field nesting. Use only
if it removes enough repeated behavior to pay rent.

### Option B: Mutable Binding Workspace

Extract the mutation-bearing part from `ThresholdSettingsBindingRequest`:

```python
@dataclass(slots=True)
class MutableModuleBindingWorkspace:
    kwargs: dict[str, Any]
    unmapped_kwargs: dict[str, Any]

    def bind_value(...): ...
    def mark_consumed(...): ...
```

Then threshold binding becomes:

```python
@dataclass(frozen=True, slots=True)
class ThresholdSettingsBindingRequest:
    context: ModuleSettingsBindingContext
    workspace: MutableModuleBindingWorkspace
    include_advanced_setting: bool
```

This is likely the strongest shape if more binding strategies can reuse the
workspace.

### Option C: Wait for Callable Request Binding

If `callable_request(...)` introduces a generic request/context idiom, reuse that
pattern before changing CP setting-binding classes. The cleanup should converge
with core request semantics instead of inventing a CP-only context vocabulary.

## Implementation Sequence

### Phase 1: Inventory Binding Requests

Search for module/binder/context carriers:

```bash
rg "module: ModuleBlock|binder: SettingsBinder|kwargs: dict\\[str, Any\\]|unmapped_kwargs" \
  openhcs/interop/cellprofiler
```

Classify each candidate:

- immutable parse context;
- mutable binding workspace;
- module-specific registered binding plan;
- one-off helper request.

Do not refactor only the two currently flagged classes without checking nearby
binding requests.

### Phase 2: Add Focused Tests Around Threshold Binding

Pin:

- threshold method parsing;
- advanced threshold settings;
- unmapped setting removal;
- legacy threshold defaults;
- generated pipeline behavior for Threshold modules.

Suggested tests:

- existing `tests/unit/test_settings_binder.py`
- add smaller tests if needed instead of growing the monolith further.

### Phase 3: Extract a Load-Bearing Context or Workspace

Prefer extracting behavior, not fields.

Good extracted behavior:

- setting lookup through `SettingsBinder`;
- normalized setting-name consumption;
- typed parse failures with module number/name context;
- updating `kwargs` and `unmapped_kwargs` together.

Bad extracted behavior:

- only `module` and `binder` fields with no methods;
- a registered base used by unrelated module families;
- a generic `Context` suffix without lifecycle ownership.

### Phase 4: Migrate Threshold First

Move threshold code to the new context/workspace.

Preserve:

- error text quality;
- coverage records;
- `include_advanced_setting` semantics;
- generated pipeline output.

### Phase 5: Evaluate ColorToGray Migration

Only migrate `ColorToGrayModuleBinding` if the new context gives real behavior
reuse.

If migration only turns `self.module` into `self.context.module`, leave
ColorToGray unchanged and document the remaining advisor finding as incidental
field overlap.

## Advisor Strategy

Run after each slice:

```bash
.venv/bin/python -m nominal_refactor_advisor \
  openhcs/interop/cellprofiler/color_to_gray_settings.py \
  openhcs/interop/cellprofiler/module_settings_binding.py
```

If advisor count improves but the code loses semantic clarity, revert the slice.
This campaign is about stronger CP binding architecture, not count-chasing.

## Verification Gates

Focused:

```bash
.venv/bin/python -m pytest \
  tests/unit/test_settings_binder.py \
  tests/unit/test_cellprofiler_generated_pipeline_execution.py \
  tests/unit/test_runner_cellprofiler_compatibility.py -q
```

Full:

```bash
.venv/bin/python -m pytest tests/unit -q
```

## Completion Criteria

- Threshold binding has a clearer request/context/workspace boundary.
- No fake inheritance from ColorToGray-specific classes.
- If a shared context exists, it owns behavior or invariants, not just fields.
- Focused and full tests pass.
- The advisor finding is either removed or documented as an accepted incidental
  field-overlap false positive after a failed load-bearing extraction attempt.

## Execution Note

Implemented the load-bearing part without fake inheritance:

- `ModuleSettingsBindingContext` now owns immutable module/binder parsing
  context;
- `MutableModuleSettingsBindingWorkspace` owns kwargs and unmapped-setting
  mutation;
- `ThresholdSettingsBindingRequest` composes both and no longer directly carries
  the mutable workspace fields.

Advisor still suggests routing this generic context through
`ColorToGrayModuleBinding`, but that remains rejected as false semantic
inheritance. The accepted finding is recorded in
`docs/plans/advisor_known_noise.md`.
