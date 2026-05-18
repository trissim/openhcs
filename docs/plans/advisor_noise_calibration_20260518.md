# Advisor Noise Calibration Plan

## Goal

Calibrate or document advisor findings that are structurally correct but
architecturally wrong for OpenHCS, starting with the remaining CellProfiler
cross-domain skeleton false positive.

Current CP-wide remaining finding:

- `ModuleInputRolePolicy.preserve_duplicate_inputs`
- `CellProfilerObjectMeasurementRowPolicy.row_is_object_scoped`

Both default to `False`/`True`-style predicate bodies, but they belong to
different domains.

## Verified Current Seams

### Symbol Table Input Role Policy

File:

- `openhcs/interop/cellprofiler/symbol_table.py`

Classes/methods:

- `ModuleInputRolePolicy`
- `DeduplicatingModuleInputRolePolicy`
- `RolePreservingModuleInputRolePolicy`
- `WatershedInputRolePolicy`
- `CorrectIlluminationApplyInputRolePolicy`
- `preserve_duplicate_inputs(module: ModuleBlock) -> bool`

Domain:

- compile-time symbol-table planning;
- duplicate input-name semantics;
- whether same name/kind inputs should remain distinct because positional roles
  matter.

### Runtime Measurement Row Policy

File:

- `openhcs/interop/cellprofiler/runtime/module_execution.py`

Classes/methods:

- `CellProfilerObjectMeasurementRowPolicy`
- `row_is_object_scoped(row: Any) -> bool`
- `annotate_record_row(...)`
- `row_has_measured_object(...)`
- module-specific row policies.

Domain:

- runtime measurement row ownership;
- object-vs-image measurement scope;
- output row annotation and materialization.

## Why This Is Noise

The methods have the same normalized skeleton because both are small policy
predicates with ignored default arguments.

They do not share:

- lifecycle;
- input type;
- output consumer;
- error behavior;
- module registry;
- data model;
- runtime phase.

A shared base would create a cross-domain "boolean policy" abstraction, which is
weaker architecture than the current separate domain policies.

## Calibration Options

### Option A: Advisor Suppression Metadata in Code

Add a local suppression marker if the advisor supports it.

Requirements:

- suppression reason must name the domain split;
- suppression must be narrow to these two methods or stable finding id;
- suppression must be visible in review.

Do not use broad file-level ignores.

### Option B: Advisor Package Heuristic Fix

Patch `nominal-refactor-advisor` so Pattern 5 skeleton findings discount
single-return predicate defaults when:

- method names differ by domain vocabulary;
- owning classes belong to different registered families;
- argument types are unrelated;
- there is no common call-site fanout;
- the proposed base would have no concrete shared method body beyond
  `return <constant>`.

This is likely the cleanest long-term fix because it prevents future "boolean
policy" false positives across the repo.

### Option C: Repository Known-Noise Ledger

Create or update a known-noise file:

- `docs/plans/advisor_known_noise.md`

Record:

- advisor command;
- finding stable id;
- evidence paths;
- why not refactored;
- date and commit that verified tests.

This is acceptable if the advisor package has no suppression mechanism.

## Implementation Sequence

### Phase 1: Check Advisor Suppression Support

Inspect the installed advisor/package docs or local repo if available:

```bash
python - <<'PY'
import nominal_refactor_advisor, inspect
print(nominal_refactor_advisor.__file__)
PY
```

Search for:

- `suppress`
- `ignore`
- `noqa`
- stable id handling
- config file support

### Phase 2: Build a Minimal Reproduction

Create a tiny sample in a temp path with two unrelated predicate policies and
run the advisor on it.

Purpose:

- confirm this is a generic analyzer issue;
- avoid changing OpenHCS code just to test advisor behavior.

### Phase 3: Choose Suppression vs Package Patch

Prefer package patch if:

- the user wants advisor quality improved globally;
- the advisor repo is available locally;
- tests can be added there.

Prefer known-noise ledger if:

- advisor package changes would be too large for the current OpenHCS branch;
- there is no local editable advisor checkout;
- suppression support is absent.

### Phase 4: Avoid OpenHCS Fake Abstractions

Do not introduce any of the following:

- `BooleanPolicyBase`;
- `PredicatePolicy`;
- shared ABC solely for `return True/False`;
- mixin that exists only to pacify Pattern 5.

Any OpenHCS code change must improve domain clarity independent of advisor
count.

## Verification

OpenHCS:

```bash
.venv/bin/python -m nominal_refactor_advisor \
  openhcs/interop/cellprofiler/symbol_table.py \
  openhcs/interop/cellprofiler/runtime/module_execution.py
```

If patching advisor package:

```bash
python -m pytest
```

Then re-run CP-wide advisor:

```bash
.venv/bin/python -m nominal_refactor_advisor openhcs/interop/cellprofiler
```

OpenHCS tests:

```bash
.venv/bin/python -m pytest \
  tests/unit/test_cellprofiler_symbol_table.py \
  tests/unit/test_cellprofiler_module_execution.py \
  tests/unit/test_runner_cellprofiler_compatibility.py -q
```

## Completion Criteria

- The remaining cross-domain skeleton false positive is either suppressed with a
  precise reason or fixed in advisor heuristics.
- No fake shared predicate abstraction is added to OpenHCS.
- The decision is documented with paths, stable id if available, and verification
  commands.
- CP-wide advisor output is more meaningful after the change.

## Execution Note

Added `docs/plans/advisor_known_noise.md` with current CP-wide advisor stable
ids and rationale:

- `4b02975f0d`: generic settings binding context must not inherit from the
  ColorToGray-specific registered binding family;
- `d2606a5064`: symbol-table input role planning and runtime measurement-row
  ownership are unrelated predicate domains.

No OpenHCS fake predicate base or ColorToGray inheritance was added.
