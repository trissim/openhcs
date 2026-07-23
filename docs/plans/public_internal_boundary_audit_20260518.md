# Public/Internal Boundary Audit Plan

## Goal

Audit OpenHCS and CellProfiler interop modules for accidental public extension
surface, then narrow single-consumer or implementation-only abstractions to
internal names without breaking tests, generated pipelines, or documented APIs.

This continues the successful narrowing from:

- `GrayToColorInputNameResolver` to `_GrayToColorInputNameResolver`
- `ModuleFunctionResolutionStrategy` to `_ModuleFunctionResolutionStrategy`
- `ModuleSettingsBindingStrategy` to `_ModuleSettingsBindingStrategy`

## Why This Matters

The advisor's matcher-fanout findings were not saying "delete the
abstractions." They were saying those roots looked public while having one
production consumer.

OpenHCS should make the distinction explicit:

- public API: stable, documented, multi-consumer or user-facing;
- extension seam: intentionally imported by external code or plugins;
- internal collaborator: nominal and testable, but not exported as API.

## Verified Current Seams

### CP Package Root

File:

- `openhcs/interop/cellprofiler/__init__.py`

This file defines the public import surface for CellProfiler interop records and
helpers. Anything imported here should be treated as stable or intentionally
public.

### Dynamic Public Names

Some modules use:

- `declared_public_names(...)`
- `__all__`

Examples found during scan:

- `filter_objects_settings.py`
- `measurement_lookup.py`
- `thresholding.py`
- `unmix_colors_settings.py`

These modules need explicit review because dynamic public-name export can
accidentally expose implementation collaborators.

### Internal-but-Tested Collaborators

Current internal roots include:

- `_GrayToColorInputNameResolver`
- `_ModuleFunctionResolutionStrategy`
- `_ModuleSettingsBindingStrategy`

White-box tests may import these. That is acceptable if tests are explicitly
testing internal behavior and production users are not expected to import them.

## Audit Classification

For each exported or importable symbol, assign one category:

| Category | Definition | Expected naming/export |
| --- | --- | --- |
| Public API | User-facing or documented import | no leading underscore, exported from package if appropriate |
| Extension seam | Intended subclass/registry/plugin family | no leading underscore, tests for registration/fanout |
| Internal collaborator | Production implementation detail | leading underscore, not in `__all__` |
| Test fixture/helper | Test-only support | local to tests or under test namespace |
| Generated compatibility | Required by generated code but not hand-authored API | documented marker or generator-owned import path |

## Implementation Sequence

### Phase 1: Build Export Map

Commands:

```bash
rg "__all__|declared_public_names|from openhcs\\.interop\\.cellprofiler import|import openhcs\\.interop\\.cellprofiler" \
  openhcs tests docs
```

Also inspect package roots:

```bash
sed -n '1,180p' openhcs/interop/cellprofiler/__init__.py
find openhcs/interop/cellprofiler -maxdepth 1 -name "__init__.py" -print
```

Output artifact:

- short table in the PR or a follow-up markdown section listing public exports
  changed or intentionally preserved.

### Phase 2: Find Public Single-Consumer Strategy Roots

Commands:

```bash
rg "class [A-Za-z0-9]+(Strategy|Policy|Resolver|Provider|Builder|Planner)" \
  openhcs/interop/cellprofiler
```

For each root:

- count production consumers;
- count test-only consumers;
- determine if it is in `__all__` or package root;
- determine if generated code imports it.

Do not rename behavior-bearing public extension seams just because current
consumer count is one. Some registries are intentionally extension points.

### Phase 3: Narrow Internal Collaborators

For symbols classified as internal:

- add leading underscore;
- remove from `__all__` or dynamic public names;
- update production imports;
- update white-box tests deliberately;
- keep error text user-facing, not underscore-heavy.

Avoid broad `perl` renames unless immediately followed by diff review and tests.

### Phase 4: Stabilize Extension Seams

For symbols classified as extension seams:

- add tests proving registration behavior;
- document expected consumers;
- keep no leading underscore;
- ensure advisor fanout findings are either gone or justified.

Examples:

- `ModuleRuntimeSemanticsBinding`
- `CellProfilerModuleSemanticTraits`
- module-role and runtime-semantic registries.

### Phase 5: Review Dynamic Public Name Helpers

If `declared_public_names(...)` exports too much, replace with explicit
`__all__` or add predicate filters.

Rules:

- constants may remain exported if they are used by conversion tables or tests;
- generated declarations should not automatically become public API;
- private classes/functions must not be exported by name-generation helpers.

## Risk Points

- Generated CP pipeline code may import names indirectly from module-level
  imports.
- Tests currently import internals for coverage; update tests without confusing
  public API expectations.
- Package root exports are more stable than direct module imports. Treat root
  removal as a breaking change unless verified unused.
- Dynamic `__all__` helpers can mask accidental public surface.

## Verification

Advisor:

```bash
.venv/bin/python -m nominal_refactor_advisor openhcs/interop/cellprofiler
```

Focused tests:

```bash
.venv/bin/python -m pytest \
  tests/unit/test_cellprofiler_strategy_registries.py \
  tests/unit/test_settings_binder.py \
  tests/unit/test_cellprofiler_generated_pipeline_execution.py \
  tests/unit/test_runner_cellprofiler_compatibility.py -q
```

Import smoke:

```bash
.venv/bin/python - <<'PY'
import openhcs.interop.cellprofiler as cp
print(cp.__name__)
PY
```

Full:

```bash
.venv/bin/python -m pytest tests/unit -q
```

## Completion Criteria

- Public CP interop exports are intentional and documented by category.
- Single-consumer internal collaborators use private names and are not exported.
- Extension seams retain public names and have registration/fanout tests.
- Dynamic public-name helpers do not leak private collaborators.
- Generated pipeline and compatibility tests pass.

## Execution Note

Current audit result:

- package-root exports in `openhcs/interop/cellprofiler/__init__.py` are
  explicit re-exports filtered by `exported_public_names`;
- dynamic `declared_public_names(...)` already rejects leading-underscore names;
- sampled dynamic export modules have no private leaks:
  `gray_to_color_settings`, `filter_objects_settings`, `flag_image`,
  `unmix_colors_settings`, and `watershed_settings`;
- recent matcher roots remain internal:
  `_GrayToColorInputNameResolver`, `_ModuleFunctionResolutionStrategy`, and
  `_ModuleSettingsBindingStrategy`.

No code change is required for this audit pass beyond preserving the current
public/private split and avoiding package-root re-export of internal
collaborators.
