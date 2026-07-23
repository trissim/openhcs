# Advisor Known Noise

This ledger records advisor findings that were inspected against the real
OpenHCS code seams and intentionally not refactored because the suggested shape
would weaken the architecture.

## 2026-05-18 CellProfiler Interop

Verification command:

```bash
.venv/bin/python -m nominal_refactor_advisor openhcs/interop/cellprofiler
```

Current accepted findings:

| Stable id | Evidence | Reason |
| --- | --- | --- |
| `4b02975f0d` | `ModuleSettingsBindingContext` and `ColorToGrayModuleBinding` | Both carry `module` and `binder`, but `ColorToGrayModuleBinding` is a ColorToGray-specific registered family. Making generic threshold/settings binding inherit from it would be false semantic inheritance. The extracted `ModuleSettingsBindingContext` and `MutableModuleSettingsBindingWorkspace` own generic settings-binding lifecycle by composition instead. |
| `d2606a5064` | `ModuleInputRolePolicy.preserve_duplicate_inputs` and `CellProfilerObjectMeasurementRowPolicy.row_is_object_scoped` | These are unrelated domain predicates. The former is compile-time symbol-table input role planning; the latter is runtime measurement-row ownership. A shared boolean-policy base would have no stable CellProfiler concept behind it. |

Do not add generic predicate bases or route unrelated settings-binding contexts
through ColorToGray-specific registries to silence these findings.

## 2026-05-18 Deprecated Textual TUI

Verification command:

```bash
timeout 180 .venv/bin/python -m nominal_refactor_advisor openhcs
```

Current accepted category:

| Stable id/category | Evidence | Reason |
| --- | --- | --- |
| Textual TUI readability/layout findings | `openhcs/textual_tui/services/terminal_enhancements.py`, `openhcs/textual_tui/widgets/config_form.py`, `openhcs/textual_tui/widgets/function_list_editor.py` | Textual TUI is deprecated and excluded from active refactor campaigns. Do not spend architecture budget formatting or decomposing it unless the code is deleted, revived, or blocks active imports/tests. |

## 2026-05-18 Side-Effecting Property Alias

Verification command:

```bash
timeout 120 .venv/bin/python -m nominal_refactor_advisor openhcs/core/orchestrator/orchestrator.py
```

Current accepted finding:

| Stable id | Evidence | Reason |
| --- | --- | --- |
| `cd4dc8d0f8` | `PipelineOrchestrator.pipeline_config` | Getter is a direct alias, but the property owns a side-effecting setter that synchronizes `pipeline_config` with `metadata_cache.pipeline_config`. Replacing it with a read-only descriptor alias would erase setter behavior; defer until a typed config-sync descriptor exists. |
