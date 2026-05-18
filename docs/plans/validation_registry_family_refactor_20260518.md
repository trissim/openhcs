# Validation Registry Family Refactor - 2026-05-18

## Advisor Evidence

Full-repo scan flagged:

- `openhcs/validation/ast_validator.py`
- `ASTValidator`
- concrete leaves:
  - `BackendParameterValidator`
  - `MemoryTypeValidator`
  - `PathTypeValidator`
  - `VFSBoundaryValidator`

Advisor summary: the validator family has concrete semantic leaves and shared
visitor methods but no metaclass membership SSOT.

## Current Problem

The module has a clear nominal family but currently relies on loose subclass
structure and external orchestration:

- validation types are string constants;
- violations are mutable objects;
- validators share traversal state;
- concrete validators are semantic leaves;
- registration/execution order is not represented as data.

This is a good candidate for a small registered family, not a broad framework.

## Target Shape

Introduce:

```python
class ValidationKind(Enum): ...

@dataclass(frozen=True, slots=True)
class ValidationViolation: ...

class ASTValidator(ast.NodeVisitor, metaclass=AutoRegisterMeta):
    __registry_key__ = "validation_kind"
    validation_kind: ClassVar[ValidationKind]
```

Each concrete validator declares:

```python
validation_kind = ValidationKind.PATH_TYPE
```

Expose one execution authority:

```python
def run_ast_validators(source_tree, file_path, kinds=None) -> tuple[ValidationViolation, ...]:
    ...
```

## Phase 1: Freeze Violation Record

Convert `ValidationViolation` to a frozen dataclass while preserving string
rendering.

Keep fields:

- `file_path`
- `line_number`
- `violation_type` or `validation_kind`
- `message`
- `node`

If `node` prevents freezing semantics, keep it as optional opaque context but do
not use it for equality-sensitive behavior.

## Phase 2: Introduce ValidationKind

Replace string constants with `ValidationKind`, then keep compatibility aliases:

```python
PATH_TYPE = ValidationKind.PATH_TYPE.value
```

Only remove aliases after call sites are migrated.

## Phase 3: Register Validator Family

Use `AutoRegisterMeta` if the package dependency is already available in
OpenHCS. Declare:

- registry key;
- skip-if-no-key if the root should not register;
- concrete kind per validator.

Avoid boilerplate if the metaclass registry package now supports automatic
family metadata injection.

## Phase 4: Execution Authority

Find current validator runner call sites, then replace manual lists with:

```python
ASTValidator.validators_for(kinds)
```

or equivalent registry-derived projection.

## Phase 5: Tests

Add tests for:

- each validator still emits expected violations;
- registry returns concrete validators by kind;
- execution order is stable;
- compatibility constants still work during migration.

## Risks

- AST visitors mutate traversal state (`current_function`). Keep instances
  per-run.
- Existing callers may compare string violation types. Preserve compatibility
  until migrated.
- Overusing AutoRegisterMeta for four validators is only justified if execution
  derives from the registry. Do not add a registry that no caller uses.

## Verification Gates

```bash
.venv/bin/python -m pytest tests/unit -q
python -m nominal_refactor_advisor openhcs/validation/ast_validator.py
```

## Completion Criteria

- Validator execution derives from a nominal family registry or a documented
  typed table.
- Violation records are typed/frozen.
- The semantic inheritance family finding is removed or replaced by a smaller
  accepted finding with a clear reason.
