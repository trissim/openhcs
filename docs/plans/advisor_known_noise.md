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
