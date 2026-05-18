# Runtime Artifact Query Axis Refactor - 2026-05-18

## Advisor Evidence

Full-repo scan flagged mirrored suffix-axis APIs in
`openhcs/core/runtime_artifact_queries.py`:

- `measurement_table_for_slice`
- `measurement_tables_for_slice`
- `measurement_table_for_image_number`
- `measurement_tables_for_image_number`

Advisor summary: the module repeats suffix-axis APIs for axes `image_number` and
`slice` across `measurement_table` and `measurement_tables` operations.

## Current Problem

The query surface encodes the same operation twice:

- choose an axis field;
- coerce the axis value;
- apply `MeasurementTableAxisProjection`;
- map across a tuple of measurement tables.

The current functions are readable, but they are a symptom of an expanding axis
surface. Any new row axis will add another pair of nearly identical functions.

## Target Shape

Introduce a typed row-axis query primitive:

```python
@dataclass(frozen=True, slots=True)
class MeasurementTableAxisQuery:
    field: MeasurementRowAxisField
    value: int

    def table(self, table: MeasurementTable) -> MeasurementTable: ...
    def tables(self, tables: tuple[MeasurementTable, ...]) -> tuple[MeasurementTable, ...]: ...
```

Then keep compatibility wrappers:

```python
def measurement_table_for_slice(table, slice_index):
    return MeasurementTableAxisQuery.slice(slice_index).table(table)
```

## Phase 1: Add Axis Query Object

Add the query object near `MeasurementTableAxisProjection`.

Required behavior:

- stores `MeasurementRowAxisField`;
- coerces runtime axis values once;
- owns `table(...)` and `tables(...)`;
- exposes named constructors `slice(...)` and `image_number(...)`.

## Phase 2: Migrate Existing Helpers

Rewrite the four mirrored helpers to use the new object.

Do not delete the helpers yet. They are public module-level API and likely used
by downstream/debug code.

## Phase 3: Tests

Add or extend runtime artifact query tests to cover:

- slice table projection;
- image-number table projection;
- tuple projection;
- tables without the relevant field remain valid by current projection rules;
- columnar row path and mapping row path if both are already testable.

## Phase 4: Optional Public API Cleanup

Only after usages are migrated, consider one generic public function:

```python
def measurement_table_for_axis(table, axis_query): ...
```

Do not remove named wrappers until call-site migration is complete.

## Risks

- CellProfiler compatibility depends on `ImageNumber`; keep names and field
  semantics exact.
- Runtime slice semantics differ by subject scope. Preserve
  `projects_runtime_slices` behavior.
- This is a small refactor; avoid creating a broad abstraction that hides the
  simple projection logic.

## Verification Gates

```bash
.venv/bin/python -m pytest \
  tests/unit/test_runtime_artifact*.py \
  tests/unit/test_cellprofiler_generated_pipeline_execution.py -q

python -m nominal_refactor_advisor openhcs/core/runtime_artifact_queries.py
```

## Completion Criteria

- The mirrored axis API finding is removed or reduced to compatibility wrappers.
- The authoritative axis projection behavior lives in one typed query object.
- Existing helper names remain available.

## Execution Note

Implemented `MeasurementTableAxisQuery` in
`openhcs/core/runtime_artifact_queries.py`:

- `MeasurementTableAxisQuery.slice(...)` owns runtime slice-axis projection;
- `MeasurementTableAxisQuery.image_number(...)` owns CellProfiler image-number
  projection;
- `table(...)` and `tables(...)` centralize single-table and tuple projection;
- existing public helper names now delegate to the query object.

Added tests in `tests/unit/test_runtime_artifact_queries.py` for direct query
projection, tuple projection, and wrapper delegation. Also replaced
`DataclassMeasurementColumnarRows.columns` with `AliasProperty`, removing the
same-file direct property alias advisor finding.

The advisor still reports the suffix-axis family because the four public
compatibility wrapper names remain. This is accepted for this campaign because
the implementation authority is centralized and existing public imports are
preserved.
