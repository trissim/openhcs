# CellProfiler Cross-Domain Skeleton Refactor Plan

## Goal

Remove the remaining advisor-reported repeated method skeletons only where they
represent a real shared CellProfiler domain, not just same-shaped code in
unrelated domains.

This plan covers the current broad `openhcs/interop/cellprofiler` advisor
findings in the "Repeated non-orthogonal method skeleton across classes"
category after checkpoint `d8ba6adf`.

## Current Evidence

Current advisor findings include these skeleton groups:

- `CellProfilerObjectMeasurementRowPolicy.row_is_object_scoped` in
  `openhcs/interop/cellprofiler/runtime/module_execution.py`
- `ModuleInputRolePolicy.preserve_duplicate_inputs` in
  `openhcs/interop/cellprofiler/symbol_table.py`
- `RepeatedSettingValues.at` in
  `openhcs/interop/cellprofiler/filter_objects_settings.py`
- `RepeatedOverlaySetting.at` in
  `openhcs/interop/cellprofiler/overlay_outlines_settings.py`
- `ColorToGrayImageTypeSettingsStrategy.for_image_type` in
  `openhcs/interop/cellprofiler/color_to_gray_settings.py`
- `BitDepthConversionStrategy.for_bit_depth` in
  `openhcs/interop/cellprofiler/image_export.py`
- `GrayToColorSchemeBindingStrategy.for_scheme` in
  `openhcs/interop/cellprofiler/module_settings_binding.py`
- enum `__new__` methods in:
  `calculate_math_settings.py`, `image_math_settings.py`,
  `intensity_distribution_settings.py`

The repeated-setting sequence case is already addressed by
`RepeatedSettingSequence`; this plan remains for the remaining cross-domain
skeletons that need deeper classification.

## Boundary Rule

Do not create an `ExtractedBase` just because the AST shape matches.

A shared base is valid only if it names a real CellProfiler concept with:

- a stable domain name,
- at least two coherent production consumers,
- one shared invariant or failure mode,
- focused tests proving behavior did not move into a generic wrapper.

If those conditions do not hold, document the finding as advisor noise or move
the shared primitive into a more specific existing abstraction.

## Refactor Sequence

### 1. Classify Each Skeleton Group

Create a short evidence table in the implementation PR or follow-up note:

| Group | Shared domain? | Candidate owner | Action |
| --- | --- | --- | --- |
| row/object duplicate predicates | likely no | none | document as noise unless a real input-role predicate exists |
| repeated setting fallback | yes | `RepeatedSettingSequence` | complete |
| strategy selection helpers | maybe | strategy registry mixin | evaluate for shared enum-keyed selector API |
| enum `__new__` parsing | maybe | enum payload constructor helper | evaluate package-level helper |

### 2. Strategy Selector Family

Investigate whether these methods are all spelling the same "registered enum
member selector" concept:

- `ColorToGrayImageTypeSettingsStrategy.for_image_type`
- `BitDepthConversionStrategy.for_bit_depth`
- `GrayToColorSchemeBindingStrategy.for_scheme`

Preferred target if verified:

- reuse or extend `EnumKeyedStrategyMixin` in `openhcs/core/registry_strategies.py`
- standardize only the selector mechanics
- keep domain names on each root class

Do not force these classes into one CellProfiler-specific base if the only
shared fact is "look up a registry by enum value".

### 3. Enum Payload Constructors

Inspect enum `__new__` methods in:

- `calculate_math_settings.py`
- `image_math_settings.py`
- `intensity_distribution_settings.py`

If they all encode the same payload pattern, add a package-level constructor
helper or a small typed declaration materializer. If the payloads are genuinely
different, keep them local and document why.

### 4. Predicate Skeletons

Compare:

- `CellProfilerObjectMeasurementRowPolicy.row_is_object_scoped`
- `ModuleInputRolePolicy.preserve_duplicate_inputs`

These names suggest different domains. Treat as false positive unless code
inspection finds a real shared "role predicate" abstraction already used by
both runtime row projection and symbol-table input-role planning.

## Verification

Run after each slice:

```bash
.venv/bin/python -m nominal_refactor_advisor \
  openhcs/interop/cellprofiler/runtime/module_execution.py \
  openhcs/interop/cellprofiler/symbol_table.py \
  openhcs/interop/cellprofiler/filter_objects_settings.py \
  openhcs/interop/cellprofiler/overlay_outlines_settings.py \
  openhcs/interop/cellprofiler/color_to_gray_settings.py \
  openhcs/interop/cellprofiler/image_export.py \
  openhcs/interop/cellprofiler/module_settings_binding.py \
  openhcs/interop/cellprofiler/calculate_math_settings.py \
  openhcs/interop/cellprofiler/image_math_settings.py \
  openhcs/interop/cellprofiler/intensity_distribution_settings.py
```

```bash
.venv/bin/python -m pytest \
  tests/unit/test_cellprofiler_strategy_registries.py \
  tests/unit/test_cellprofiler_module_execution.py \
  tests/unit/test_cellprofiler_symbol_table.py \
  tests/unit/test_runner_cellprofiler_compatibility.py -q
```

Before final commit:

```bash
.venv/bin/python -m pytest tests/unit -q
```

## Completion Criteria

- Every remaining skeleton finding is either removed by a load-bearing
  abstraction or documented as cross-domain analyzer noise.
- No new helper-only wrapper findings are introduced.
- CP focused tests and full unit tests pass.

## Execution Note

Completed the load-bearing parts:

- strategy selector helpers now use the existing `EnumKeyedStrategyMixin`;
- repeated enum payload constructors now use the shared
  `enum_member_with_payload` utility in `registry_strategies.py`;
- repeated setting-sequence cases were already collapsed into
  `RepeatedSettingSequence`.

The remaining predicate skeleton pair is intentionally not collapsed:
`ModuleInputRolePolicy.preserve_duplicate_inputs` plans symbol-table input role
deduplication, while `CellProfilerObjectMeasurementRowPolicy.row_is_object_scoped`
classifies runtime measurement rows. They share only a default boolean body, not
a CellProfiler domain invariant or lifecycle. A common base would be a fake
abstraction.
