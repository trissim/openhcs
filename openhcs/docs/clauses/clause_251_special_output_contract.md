# Clause 251: Special Output Contract

## Summary
Steps must explicitly declare non-standard intermediate products using `@special_out` and `@special_in` decorators, enabling compile-time validation of these dependencies.

## Motivation
In OpenHCS, most steps declare their data dependencies via input_fields, output_fields, and the context object. However, certain non-standard intermediate products (like position_array, warp_field, or well_id) are produced by one step and consumed by another, but do not exist as standard inputs or outputs.

Without a formal way to declare these dependencies, hidden contracts between steps increase the risk of:
- Missing dependencies (execution failure)
- Implicit fallback logic (doctrinal violation)
- Planner ambiguity (step routing becomes unclear)

## Rules

### Step Annotation
All special keys must be declared using `@special_out(key)` or `@special_in(key)` at the class level.

### Static Resolution
During pipeline compilation, the planner must:
- Build the full DAG of planned steps
- For each `@special_in(key)`, verify that some upstream step in the plan declares `@special_out(key)`
- Raise a `PlanContractViolation` if missing

### Ordering
- `@special_out(key)` steps must precede `@special_in(key)` steps in execution order.
- Loops or cycles must not be allowed.

### No Fallbacks
- A step must not "check" whether a special input is available.
- If the contract is unsatisfied, the pipeline must not run.

### Key Registration
All special keys must be schema-documented under `schema/special_contracts.py` or similar.
Keys must include:
- Name
- Type (e.g. Tensor[N, 2], List[str], Dict[str, Any])
- Description and units (if applicable)

## Example

```python
@special_out("position_array")
class MISTPositionGeneratorStep(AbstractStep):
    ...

@special_in("position_array")
class ImageAssemblyStep(AbstractStep):
    ...
```

If the planner sees `ImageAssemblyStep` in a plan, it must confirm that a previous step declares `@special_out("position_array")`. If none is found, compilation halts with a clear error:

```
PlanContractViolation: Step 'ImageAssemblyStep' requires special input 'position_array', but no upstream step produces it.
```

## Additional Enforcement
- Allow `@special_in("warp_field", required=False)` to mark optional (non-fatal) dependencies
- Allow `get_declared_special_outputs()` for static planning

## Related Clauses
- Clause 3 — Declarative Primacy
- Clause 66 — Immutability After Construction
- Clause 88 — No Inferred Capabilities
- Clause 245 — Declarative Enforcement
- Clause 246 — Statelessness Mandate
