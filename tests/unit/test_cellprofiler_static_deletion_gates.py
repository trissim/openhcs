"""Canonical static gates for the CellProfiler runtime-unification deletions."""

from __future__ import annotations

import ast
from collections.abc import Iterable, Iterator
from fnmatch import fnmatchcase
from functools import cache
from pathlib import Path
import warnings


PROJECT_ROOT = Path(__file__).parents[2]
PLAN_PATH = (
    PROJECT_ROOT
    / "docs/archive/architecture/cellprofiler_openhcs_runtime_unification_plan.md"
)
CONSOLIDATION_PLAN_PATH = (
    PROJECT_ROOT
    / "docs/archive/architecture/cellprofiler_runtime_execution_import_dispatch_consolidation_plan.md"
)
THIS_FILE = Path(__file__).resolve()
SCANNED_ROOTS = (
    PROJECT_ROOT / "openhcs",
    PROJECT_ROOT / "benchmark",
    PROJECT_ROOT / "tests",
)

# Files and package trees explicitly named for deletion by Phases 1 through 3.
REQUIRED_DELETED_PATHS = (
    "benchmark/converter/absorb.py",
    "benchmark/converter/add_parameter_mappings.py",
    "benchmark/converter/backfill_parameter_mappings.py",
    "benchmark/converter/cppipe_module_roles.py",
    "benchmark/converter/fix_registry.py",
    "benchmark/converter/library_absorber.py",
    "benchmark/converter/llm_converter.py",
    "benchmark/converter/source_locator.py",
    "benchmark/converter/system_prompt.py",
    "benchmark/cellprofiler_compat",
    "benchmark/cellprofiler_library",
    "benchmark/cellprofiler_source",
    "openhcs/core/function_reference_rehydration.py",
    "openhcs/core/function_step_invocation_contracts.py",
    "openhcs/core/pipeline_image_schema.py",
    "openhcs/core/runtime_invocation.py",
    "openhcs/core/runtime_object_label_alignment.py",
    "openhcs/core/runtime_semantics.py",
    "openhcs/core/source_schema_workspace.py",
    "openhcs/interop/cellprofiler/artifact_semantics.py",
    "openhcs/interop/cellprofiler/compiler_registry.py",
    "openhcs/interop/cellprofiler/database_export.py",
    "openhcs/interop/cellprofiler/debug_views.py",
    "openhcs/interop/cellprofiler/import_records.py",
    "openhcs/interop/cellprofiler/import_service.py",
    "openhcs/interop/cellprofiler/flag_image.py",
    "openhcs/interop/cellprofiler/image_export.py",
    "openhcs/interop/cellprofiler/module_artifact_inputs.py",
    "openhcs/interop/cellprofiler/module_function_resolution.py",
    "openhcs/interop/cellprofiler/module_processing_config.py",
    "openhcs/interop/cellprofiler/module_processing_components.py",
    "openhcs/interop/cellprofiler/module_roles.py",
    "openhcs/interop/cellprofiler/module_semantics.py",
    "openhcs/interop/cellprofiler/module_setting_policies.py",
    "openhcs/interop/cellprofiler/pipeline_compiler.py",
    "openhcs/interop/cellprofiler/pipeline_generator.py",
    "openhcs/interop/cellprofiler/processing_contract_resolution.py",
    "openhcs/interop/cellprofiler/runtime/generated_pipeline.py",
    "openhcs/interop/cellprofiler/runtime/current_image_context.py",
    "openhcs/interop/cellprofiler/runtime/binding_authorities.py",
    "openhcs/interop/cellprofiler/runtime/dual_scope_measurement_policies.py",
    "openhcs/interop/cellprofiler/runtime/mapping_lookup.py",
    "openhcs/interop/cellprofiler/runtime/relationship_endpoints.py",
    "openhcs/interop/cellprofiler/runtime/policy_registry.py",
    "openhcs/interop/cellprofiler/runtime/payload_types.py",
    "openhcs/interop/cellprofiler/runtime/measurement_image_sources.py",
    "openhcs/interop/cellprofiler/runtime_pipeline.py",
    "openhcs/interop/cellprofiler/semantic_defaults.py",
    "openhcs/interop/cellprofiler/spreadsheet_export.py",
    "openhcs/interop/cellprofiler/symbol_table.py",
    "openhcs/interop/cellprofiler/thresholding.py",
    "openhcs/processing/backends/cellprofiler/library.py",
    "openhcs/processing/backends/cellprofiler/function_documentation.py",
    "openhcs/runtime/zmq_pipeline_transport.py",
    "tests/unit/test_cellprofiler_module_processing_components.py",
    "tests/unit/test_cellprofiler_source_schema.py",
    "tests/unit/test_cellprofiler_source_schema_ingestion.py",
    "tests/unit/test_cellprofiler_save_images_export.py",
    "tests/unit/test_cellprofiler_symbol_table.py",
)


def _delimited_plan_inventory(
    plan_path: Path,
    inventory_name: str,
) -> tuple[str, ...]:
    plan = plan_path.read_text(encoding="utf-8")
    start = f"<!-- {inventory_name}:start -->"
    end = f"<!-- {inventory_name}:end -->"
    delimited = plan.split(start, maxsplit=1)[1].split(end, maxsplit=1)[0]
    fenced = delimited.split("~~~text", maxsplit=1)[1].split("~~~", maxsplit=1)[0]
    values = tuple(line.strip() for line in fenced.splitlines() if line.strip())
    assert values
    assert len(values) == len(set(values))
    return values


def _forbidden_literals_from_plan() -> tuple[str, ...]:
    plan = PLAN_PATH.read_text(encoding="utf-8")
    static_section = plan.split("## Static Deletion Gates", maxsplit=1)[1]
    literal_block = static_section.split("```text", maxsplit=1)[1].split(
        "```", maxsplit=1
    )[0]
    literals = tuple(
        line.strip() for line in literal_block.splitlines() if line.strip()
    )
    assert literals
    assert len(literals) == len(set(literals))
    return literals


COLLAPSED_ARTIFACT_API_NAMES = frozenset(
    {
        "artifact_contract",
        "artifact_input_names",
        "artifact_inputs_dict",
        "artifact_output_names",
        "artifact_output_specs",
        "artifact_outputs_dict",
        "combine_declarations",
        "declared_input_collection",
        "declared_input_specs",
        "declared_output_collection",
        "input_collection",
        "invocation_artifact_contract",
        "main_flow_input_collection",
        "main_flow_input_specs",
        "names_for_partition",
        "output_collection",
        "runtime_artifact_input_collection",
        "require_module_artifact_contract",
        "source_artifact_input_collection",
    }
)
FORBIDDEN_LITERALS = tuple(
    dict.fromkeys(
        (
            *_forbidden_literals_from_plan(),
            *_delimited_plan_inventory(
                CONSOLIDATION_PLAN_PATH,
                "cellprofiler-runtime-consolidation-forbidden-symbols",
            ),
            *sorted(COLLAPSED_ARTIFACT_API_NAMES),
        )
    )
)
REQUIRED_DELETED_PATHS = tuple(
    dict.fromkeys(
        (
            *REQUIRED_DELETED_PATHS,
            *_delimited_plan_inventory(
                CONSOLIDATION_PLAN_PATH,
                "cellprofiler-runtime-consolidation-forbidden-files",
            ),
        )
    )
)
FORBIDDEN_SIMPLE_NAMES = frozenset(
    literal for literal in FORBIDDEN_LITERALS if literal.isidentifier()
)
FORBIDDEN_QUALIFIED_NAMES = frozenset(
    literal
    for literal in FORBIDDEN_LITERALS
    if "." in literal and "*" not in literal and ":" not in literal
)
FORBIDDEN_ANNOTATED_MEMBERS = frozenset(
    literal for literal in FORBIDDEN_LITERALS if ":" in literal
)
FORBIDDEN_OWNER_PATTERNS = frozenset(
    literal for literal in FORBIDDEN_LITERALS if "*" in literal
)


def _python_module_name(relative_path: str) -> str | None:
    path = Path(relative_path)
    if path.suffix != ".py" or path.parts[0] not in {"openhcs", "benchmark"}:
        return None
    return ".".join(path.with_suffix("").parts)


FORBIDDEN_MODULES = frozenset(
    module_name
    for relative_path in REQUIRED_DELETED_PATHS
    if (module_name := _python_module_name(relative_path)) is not None
)
FORBIDDEN_MODULE_PREFIXES = tuple(
    relative_path.replace("/", ".") + "."
    for relative_path in REQUIRED_DELETED_PATHS
    if Path(relative_path).suffix == ""
    and Path(relative_path).parts[0] in {"openhcs", "benchmark"}
)
FORBIDDEN_DATA_LITERALS = frozenset(
    {
        *FORBIDDEN_LITERALS,
        *FORBIDDEN_MODULES,
        *REQUIRED_DELETED_PATHS,
    }
)


def _source_paths() -> Iterator[Path]:
    for root in SCANNED_ROOTS:
        for path in root.rglob("*.py"):
            if path.resolve() != THIS_FILE:
                yield path


@cache
def _parse_source(path: Path) -> ast.Module:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _dotted_name(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        owner = _dotted_name(node.value)
        return f"{owner}.{node.attr}" if owner is not None else None
    return None


def _assigned_names(node: ast.AST) -> Iterator[str]:
    if isinstance(node, ast.Name):
        yield node.id
    elif isinstance(node, (ast.Tuple, ast.List)):
        for element in node.elts:
            yield from _assigned_names(element)


def _class_members(node: ast.ClassDef) -> Iterator[tuple[str, int]]:
    for statement in node.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield statement.name, statement.lineno
        elif isinstance(statement, ast.AnnAssign):
            for name in _assigned_names(statement.target):
                yield name, statement.lineno
        elif isinstance(statement, ast.Assign):
            for target in statement.targets:
                for name in _assigned_names(target):
                    yield name, statement.lineno
                if isinstance(target, ast.Name) and target.id == "__slots__":
                    for value in ast.walk(statement.value):
                        if isinstance(value, ast.Constant) and isinstance(
                            value.value, str
                        ):
                            yield value.value, value.lineno

    for descendant in ast.walk(node):
        if (
            isinstance(descendant, ast.Attribute)
            and isinstance(descendant.value, ast.Name)
            and descendant.value.id in {"self", "cls"}
        ):
            yield descendant.attr, descendant.lineno


def _class_annotation(
    node: ast.ClassDef,
    member_name: str,
) -> tuple[str | None, int] | None:
    for statement in node.body:
        if (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == member_name
        ):
            return _dotted_name(statement.annotation), statement.lineno
    return None


def _top_level_class(path: Path, class_name: str) -> ast.ClassDef:
    tree = _parse_source(path)
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )


def _annotated_field_names(node: ast.ClassDef) -> frozenset[str]:
    return frozenset(
        statement.target.id
        for statement in node.body
        if isinstance(statement, ast.AnnAssign)
        and isinstance(statement.target, ast.Name)
    )


def _method(node: ast.ClassDef, method_name: str) -> ast.FunctionDef:
    return next(
        statement
        for statement in node.body
        if isinstance(statement, ast.FunctionDef) and statement.name == method_name
    )


def _class_records(
    parsed_sources: Iterable[tuple[Path, ast.Module]],
) -> tuple[tuple[Path, ast.ClassDef, frozenset[str]], ...]:
    raw_records = tuple(
        (
            path,
            node,
            frozenset(
                base_name.rsplit(".", maxsplit=1)[-1]
                for base in node.bases
                if (base_name := _dotted_name(base)) is not None
            ),
        )
        for path, tree in parsed_sources
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
    )
    ancestors_by_name: dict[str, set[str]] = {}
    for _path, node, base_names in raw_records:
        ancestors_by_name.setdefault(node.name, set()).update(base_names)
    changed = True
    while changed:
        changed = False
        for ancestors in ancestors_by_name.values():
            expanded = set(ancestors)
            for ancestor in tuple(ancestors):
                expanded.update(ancestors_by_name.get(ancestor, ()))
            if expanded != ancestors:
                ancestors.update(expanded)
                changed = True
    return tuple(
        (
            path,
            node,
            frozenset({node.name, *ancestors_by_name.get(node.name, ())}),
        )
        for path, node, _base_names in raw_records
    )


def _is_forbidden_module(module_name: str) -> bool:
    return module_name in FORBIDDEN_MODULES or module_name.startswith(
        FORBIDDEN_MODULE_PREFIXES
    )


def _node_references(node: ast.AST) -> Iterator[tuple[str, str]]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        yield "definition", node.name
    elif isinstance(node, ast.Name):
        yield "name", node.id
    elif isinstance(node, ast.Attribute):
        yield "attribute", node.attr
        if (qualified_name := _dotted_name(node)) is not None:
            yield "qualified attribute", qualified_name
    elif isinstance(node, ast.arg):
        yield "argument", node.arg
    elif isinstance(node, ast.keyword) and node.arg is not None:
        yield "keyword", node.arg
    elif isinstance(node, ast.alias):
        yield "import", node.name.rsplit(".", maxsplit=1)[-1]
        if node.asname is not None:
            yield "import alias", node.asname
    elif isinstance(node, ast.Constant) and isinstance(node.value, str):
        yield "string literal", node.value.strip()


def _format_violation(path: Path, line: int, kind: str, name: str) -> str:
    return f"{path.relative_to(PROJECT_ROOT)}:{line}:{kind}:{name}"


def test_forbidden_symbols_and_imports_are_absent() -> None:
    parsed_sources = tuple(
        (
            path,
            _parse_source(path),
        )
        for path in _source_paths()
    )
    violations: list[str] = []

    for path, tree in parsed_sources:
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                if _is_forbidden_module(node.module):
                    violations.append(
                        _format_violation(path, node.lineno, "import", node.module)
                    )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if _is_forbidden_module(alias.name):
                        violations.append(
                            _format_violation(
                                path,
                                node.lineno,
                                "import",
                                alias.name,
                            )
                        )

            for kind, referenced_name in _node_references(node):
                if (
                    referenced_name in FORBIDDEN_SIMPLE_NAMES
                    or referenced_name in FORBIDDEN_QUALIFIED_NAMES
                    or referenced_name in FORBIDDEN_DATA_LITERALS
                ):
                    violations.append(
                        _format_violation(path, node.lineno, kind, referenced_name)
                    )

    qualified_members = tuple(
        (owner, member, literal)
        for literal in FORBIDDEN_QUALIFIED_NAMES
        for owner, member in (literal.rsplit(".", maxsplit=1),)
    )
    for path, node, owner_names in _class_records(parsed_sources):
        members = tuple(_class_members(node))
        for owner, member, literal in qualified_members:
            if owner in owner_names:
                for declared_member, line in members:
                    if declared_member == member:
                        violations.append(
                            _format_violation(path, line, "owned member", literal)
                        )

        for pattern in FORBIDDEN_OWNER_PATTERNS:
            owner, member_pattern = pattern.rsplit(".", maxsplit=1)
            if owner not in owner_names:
                continue
            for declared_member, line in members:
                if fnmatchcase(declared_member, member_pattern):
                    violations.append(
                        _format_violation(
                            path,
                            line,
                            "owned member pattern",
                            pattern,
                        )
                    )

        if "ArtifactDeclarationStepContext" in owner_names:
            annotation = _class_annotation(node, "processing_config")
            if annotation is not None and annotation[0] == "Any":
                violations.append(
                    _format_violation(
                        path,
                        annotation[1],
                        "owned annotation",
                        "ArtifactDeclarationStepContext.processing_config: Any",
                    )
                )

    unique_violations = sorted(set(violations))
    assert not unique_violations, "\n" + "\n".join(unique_violations)


def test_deleted_runtime_plan_has_no_attribute_consumers() -> None:
    violations = [
        _format_violation(path, node.lineno, "attribute", node.attr)
        for path in _source_paths()
        for node in ast.walk(_parse_source(path))
        if isinstance(node, ast.Attribute) and node.attr == "runtime_plan"
    ]

    assert not violations, "\n" + "\n".join(violations)


def test_required_deletion_paths_are_absent() -> None:
    existing_paths = tuple(
        relative_path
        for relative_path in REQUIRED_DELETED_PATHS
        if (PROJECT_ROOT / relative_path).exists()
    )

    assert not existing_paths, "\n" + "\n".join(existing_paths)


def test_split_nominal_owners_have_no_facade_imports() -> None:
    """Require consumers to import definitions from their concrete owner module."""

    module_declarations_path = (
        PROJECT_ROOT / "openhcs/interop/cellprofiler/module_declarations.py"
    )
    declaration_tree = _parse_source(module_declarations_path)
    owned_names = frozenset(
        node.name
        for node in declaration_tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    )
    violations: list[str] = []
    for path in _source_paths():
        for node in ast.walk(_parse_source(path)):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module == "openhcs.core.runtime_semantics":
                violations.append(
                    _format_violation(
                        path,
                        node.lineno,
                        "deleted owner import",
                        node.module,
                    )
                )
            if node.module != "openhcs.interop.cellprofiler.module_declarations":
                continue
            for alias in node.names:
                if alias.name not in owned_names:
                    violations.append(
                        _format_violation(
                            path,
                            node.lineno,
                            "non-owner import",
                            alias.name,
                        )
                    )

    assert not violations, "\n" + "\n".join(sorted(set(violations)))


def test_cellprofiler_short_name_lookup_stays_inside_backend_projection() -> None:
    """Prevent generic consumers from reconstructing module ownership by name."""

    permitted_path = (
        PROJECT_ROOT / "openhcs/processing/backends/cellprofiler/__init__.py"
    ).resolve()
    violations = [
        _format_violation(
            path,
            node.lineno,
            "short-name owner lookup",
            node.attr,
        )
        for path in _source_paths()
        for node in ast.walk(_parse_source(path))
        if isinstance(node, ast.Attribute)
        and node.attr == "for_backend_function_name"
        and path.is_relative_to(PROJECT_ROOT / "openhcs")
        and path.resolve() != permitted_path
    ]
    legacy_violations = [
        _format_violation(
            path,
            node.lineno,
            "deleted short-name owner lookup",
            node.attr,
        )
        for path in _source_paths()
        for node in ast.walk(_parse_source(path))
        if isinstance(node, ast.Attribute) and node.attr == "for_function_name"
    ]

    assert not (*violations, *legacy_violations), "\n" + "\n".join(
        sorted({*violations, *legacy_violations})
    )


def test_cellprofiler_measurement_outputs_declare_feature_owners() -> None:
    """Keep measurement vocabulary ownership on the producing declaration."""

    roots = (
        PROJECT_ROOT / "openhcs/interop/cellprofiler",
        PROJECT_ROOT / "openhcs/processing/backends/cellprofiler",
    )
    violations: list[str] = []
    for root in roots:
        for path in root.rglob("*.py"):
            for node in ast.walk(_parse_source(path)):
                if not isinstance(node, ast.Call) or len(node.args) < 2:
                    continue
                callable_name = _dotted_name(node.func)
                artifact_type_name = _dotted_name(node.args[1])
                keyword_names = {keyword.arg for keyword in node.keywords}
                if (
                    callable_name == "ArtifactSpec.output"
                    and artifact_type_name == "MeasurementsArtifactType"
                    and "measurement_feature_owner" not in keyword_names
                ):
                    violations.append(
                        _format_violation(
                            path,
                            node.lineno,
                            "ownerless measurement output",
                            callable_name,
                        )
                    )

    assert not violations, "\n" + "\n".join(sorted(violations))


def test_split_owner_modules_have_unique_nominal_definitions() -> None:
    """Keep split owners free of replacement facades and shadowed definitions."""

    owner_paths = tuple(
        sorted((PROJECT_ROOT / "openhcs/interop/cellprofiler").glob("module_*.py"))
    )
    violations: list[str] = []
    owners_by_name: dict[str, list[Path]] = {}
    for path in owner_paths:
        for node in _parse_source(path).body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                owners_by_name.setdefault(node.name, []).append(path)
    for name, paths in owners_by_name.items():
        if len(paths) == 1:
            continue
        violations.append(
            _format_violation(
                paths[0],
                1,
                "duplicate split-owner definition",
                f"{name}:{tuple(path.name for path in paths)!r}",
            )
        )

    declaration_path = (
        PROJECT_ROOT / "openhcs/interop/cellprofiler/module_declarations.py"
    )
    declaration_classes = tuple(
        node.name
        for node in _parse_source(declaration_path).body
        if isinstance(node, ast.ClassDef)
    )
    assert declaration_classes == ("CellProfilerModule",)
    assert not violations, "\n" + "\n".join(violations)


def test_plan_inventory_uses_supported_exact_gate_forms() -> None:
    classified_literals = (
        FORBIDDEN_SIMPLE_NAMES
        | FORBIDDEN_QUALIFIED_NAMES
        | FORBIDDEN_ANNOTATED_MEMBERS
        | FORBIDDEN_OWNER_PATTERNS
    )

    assert classified_literals == frozenset(FORBIDDEN_LITERALS)
    assert FORBIDDEN_ANNOTATED_MEMBERS == frozenset(
        {"ArtifactDeclarationStepContext.processing_config: Any"}
    )
    assert FORBIDDEN_OWNER_PATTERNS == frozenset(
        {
            "CellProfilerModule.*compile_time_*",
            "StepSourceBindingsConfig.inherits_*",
        }
    )


def test_cellprofiler_artifact_contract_orchestration_is_base_owned() -> None:
    """Keep contract assembly out of module leaves and call sites."""

    violations: list[str] = []
    for path in _source_paths():
        if "cellprofiler" not in path.parts:
            continue
        tree = _parse_source(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for statement in node.body:
                    if not isinstance(
                        statement, (ast.FunctionDef, ast.AsyncFunctionDef)
                    ):
                        continue
                    if statement.name in {
                        "contract_from_specs",
                        "declare_artifact_contract",
                    }:
                        violations.append(
                            _format_violation(
                                path,
                                statement.lineno,
                                "owned contract assembly",
                                f"{node.name}.{statement.name}",
                            )
                        )
                    if statement.name == "artifact_contract" and not (
                        path.name == "module_artifact_contracts.py"
                        and node.name == "CellProfilerModuleArtifactContracts"
                    ):
                        violations.append(
                            _format_violation(
                                path,
                                statement.lineno,
                                "leaf contract orchestration",
                                f"{node.name}.artifact_contract",
                            )
                        )
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr
                in {"contract_from_specs", "declare_artifact_contract"}
            ):
                violations.append(
                    _format_violation(
                        path,
                        node.lineno,
                        "contract assembly call",
                        node.func.attr,
                    )
                )

    assert not violations, "\n" + "\n".join(sorted(set(violations)))


def test_artifact_spec_refs_are_derived_only_by_nominal_owners() -> None:
    """Prevent callers from reconstructing declaration identity by hand."""

    violations: list[str] = []
    for path in _source_paths():
        tree = _parse_source(path)
        allowed_calls: set[int] = set()
        if path == PROJECT_ROOT / "openhcs/core/artifacts.py":
            for class_node in tree.body:
                if not isinstance(class_node, ast.ClassDef) or class_node.name not in {
                    "ArtifactSpec",
                    "ArtifactPlan",
                }:
                    continue
                for statement in class_node.body:
                    if (
                        not isinstance(
                            statement,
                            (ast.FunctionDef, ast.AsyncFunctionDef),
                        )
                        or statement.name != "ref"
                    ):
                        continue
                    allowed_calls.update(
                        id(node)
                        for node in ast.walk(statement)
                        if isinstance(node, ast.Call)
                    )

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "ArtifactSpecRef"
                and id(node) not in allowed_calls
            ):
                violations.append(
                    _format_violation(
                        path,
                        node.lineno,
                        "manual artifact identity",
                        "ArtifactSpecRef",
                    )
                )
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "ArtifactSpecRef"
            ):
                violations.append(
                    _format_violation(
                        path,
                        node.lineno,
                        "manual artifact identity factory",
                        node.func.attr,
                    )
                )

    assert not violations, "\n" + "\n".join(sorted(set(violations)))


def test_runtime_adapter_state_has_one_nominal_owner() -> None:
    """Prevent runtime request, source context, and adapter fields from being cloned."""

    runtime_adapters = _parse_source(PROJECT_ROOT / "openhcs/core/runtime_adapters.py")
    adapter_module = _parse_source(
        PROJECT_ROOT / "openhcs/interop/cellprofiler/runtime/adapter.py"
    )
    source_bindings = _parse_source(PROJECT_ROOT / "openhcs/core/source_bindings.py")
    function_runtime = _parse_source(
        PROJECT_ROOT / "openhcs/core/steps/function_runtime.py"
    )

    classes = {
        node.name: node
        for tree in (
            runtime_adapters,
            adapter_module,
            source_bindings,
            function_runtime,
        )
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }
    request = classes["RuntimeAdapterRequest"]
    adapter = classes["CellProfilerRuntimeAdapter"]
    source_context = classes["SourceBindingRuntimeContext"]

    request_fields = {
        node.target.id
        for node in request.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    adapter_fields = {
        node.target.id
        for node in adapter.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    mirrored_fields = adapter_fields & (
        request_fields
        | {
            "runtime_value_store",
            "processing_context",
            "filename_parser",
            "filemanager",
            "output_identity_cache",
        }
    )

    assert not mirrored_fields
    assert "request" in adapter_fields
    assert all(
        ast.unparse(base) != "SourceBindingRuntimeContext" for base in request.bases
    )
    assert not any(
        isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "source_binding_context"
        for node in source_context.body
    )
    for class_name in ("FunctionRuntimeScope", "PatternGroupData"):
        assert all(
            ast.unparse(base) != "SourceBindingRuntimeContext"
            for base in classes[class_name].bases
        )


def test_consolidated_runtime_owners_have_exact_state_and_factory_boundary() -> None:
    module_execution_path = (
        PROJECT_ROOT / "openhcs/interop/cellprofiler/runtime/module_execution.py"
    )
    adapter_path = PROJECT_ROOT / "openhcs/interop/cellprofiler/runtime/adapter.py"
    executor = _top_level_class(module_execution_path, "CellProfilerModuleExecutor")
    adapter = _top_level_class(adapter_path, "CellProfilerRuntimeAdapter")

    assert _annotated_field_names(executor) == frozenset(
        {"raw_func", "callable_contract"}
    )
    assert any(
        isinstance(statement, ast.FunctionDef) and statement.name == "__call__"
        for statement in executor.body
    )
    assert _annotated_field_names(adapter) == frozenset({"request", "backend"})

    module_tree = _parse_source(module_execution_path)
    factory = next(
        node
        for node in module_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "cellprofiler_runtime_callable_factory"
    )
    returns = tuple(node for node in ast.walk(factory) if isinstance(node, ast.Return))
    assert len(returns) == 1
    returned = returns[0].value
    assert isinstance(returned, ast.Call)
    assert _dotted_name(returned.func) == "CellProfilerModuleExecutor"


def test_runtime_never_reconstructs_compiled_callable_contracts() -> None:
    runtime_root = PROJECT_ROOT / "openhcs/interop/cellprofiler/runtime"
    violations = tuple(
        _format_violation(
            path,
            node.lineno,
            "runtime contract reconstruction",
            "CallableContract.from_callable",
        )
        for path in runtime_root.rglob("*.py")
        for node in ast.walk(_parse_source(path))
        if isinstance(node, ast.Call)
        and _dotted_name(node.func) == "CallableContract.from_callable"
    )

    assert not violations, "\n" + "\n".join(violations)


def test_generic_core_does_not_import_cellprofiler() -> None:
    violations: list[str] = []
    for path in (PROJECT_ROOT / "openhcs/core").rglob("*.py"):
        for node in ast.walk(_parse_source(path)):
            imported_modules: tuple[str, ...] = ()
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                imported_modules = (node.module,)
            elif isinstance(node, ast.Import):
                imported_modules = tuple(alias.name for alias in node.names)
            for module_name in imported_modules:
                if module_name.startswith(
                    (
                        "openhcs.interop.cellprofiler",
                        "openhcs.processing.backends.cellprofiler",
                    )
                ):
                    violations.append(
                        _format_violation(path, node.lineno, "core import", module_name)
                    )

    assert not violations, "\n" + "\n".join(violations)


def test_forward_artifact_state_has_one_nominal_owner() -> None:
    required_fields = frozenset(
        {
            "available_artifacts",
            "main_flow_artifacts",
            "available_artifact_producers",
        }
    )
    owners = tuple(
        (path, node)
        for path in _source_paths()
        for node in ast.walk(_parse_source(path))
        if isinstance(node, ast.ClassDef)
        and required_fields <= _annotated_field_names(node)
    )

    assert tuple(node.name for _path, node in owners) == (
        "ArtifactDeclarationStepContext",
    )

    compiler_tree = _parse_source(
        PROJECT_ROOT / "openhcs/interop/cellprofiler/compile_time_contracts.py"
    )
    factory = next(
        node
        for node in compiler_tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "CellProfilerInvocationContractProviderFactory"
    )
    provider_method = _method(factory, "provider_for_session")
    calls = {
        _dotted_name(node.func)
        for node in ast.walk(provider_method)
        if isinstance(node, ast.Call)
    }
    assert "extract_artifact_declarations" in calls
    assert "artifact_plan_key_selector_for_contract" not in calls


def test_cellprofiler_import_and_provider_do_not_rebuild_owned_semantics() -> None:
    importer = _parse_source(
        PROJECT_ROOT / "openhcs/interop/cellprofiler/pipeline_import.py"
    )
    provider = _parse_source(
        PROJECT_ROOT / "openhcs/interop/cellprofiler/compile_time_contracts.py"
    )
    module_declarations = _parse_source(
        PROJECT_ROOT / "openhcs/interop/cellprofiler/module_declarations.py"
    )

    importer_attributes = {
        node.attr for node in ast.walk(importer) if isinstance(node, ast.Attribute)
    }
    importer_names = {
        node.id for node in ast.walk(importer) if isinstance(node, ast.Name)
    }
    provider_names = {
        node.id for node in ast.walk(provider) if isinstance(node, ast.Name)
    }
    module_methods = {
        node.name
        for node in ast.walk(module_declarations)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert "projected_source_bindings" not in importer_attributes
    assert "for_artifact_specs" not in importer_attributes
    assert "for_artifact_refs" in importer_attributes
    assert "declared_setting_bindings" in importer_attributes
    assert "_sparse_kwargs_for_contract" not in importer_names
    assert "special_input_names_from_callable" not in importer_names
    assert "CompiledSourceBindingPlan" not in provider_names
    assert "partition_bound_public_kwargs" not in module_methods


def test_step_context_callers_do_not_pass_parallel_artifact_state() -> None:
    violations: list[str] = []
    for path in _source_paths():
        for node in ast.walk(_parse_source(path)):
            if not isinstance(node, ast.Call):
                continue
            keyword_names = frozenset(
                keyword.arg for keyword in node.keywords if keyword.arg is not None
            )
            if "step_context" not in keyword_names:
                continue
            parallel_names = keyword_names & {
                "available_artifacts",
                "main_flow_artifacts",
            }
            for name in sorted(parallel_names):
                violations.append(
                    _format_violation(
                        path,
                        node.lineno,
                        "parallel artifact state",
                        name,
                    )
                )

    assert not violations, "\n" + "\n".join(violations)


def test_source_component_tuple_state_is_not_mirrored() -> None:
    source_bindings_path = PROJECT_ROOT / "openhcs/core/source_bindings.py"
    source_tree = _parse_source(source_bindings_path)
    source_component_fields = frozenset(
        statement.target.id
        for node in ast.walk(source_tree)
        if isinstance(node, ast.ClassDef)
        for statement in node.body
        if isinstance(statement, ast.AnnAssign)
        and isinstance(statement.target, ast.Name)
        and "tuple[AllComponents" in ast.unparse(statement.annotation)
    )
    assert source_component_fields

    allowed_paths = {
        source_bindings_path,
        PROJECT_ROOT / "openhcs/core/config.py",
    }
    violations = tuple(
        _format_violation(
            path,
            statement.lineno,
            "mirrored source component field",
            statement.target.id,
        )
        for path in _source_paths()
        if path not in allowed_paths
        for node in ast.walk(_parse_source(path))
        if isinstance(node, ast.ClassDef)
        for statement in node.body
        if isinstance(statement, ast.AnnAssign)
        and isinstance(statement.target, ast.Name)
        and statement.target.id.lstrip("_") in source_component_fields
    )

    assert not violations, "\n" + "\n".join(violations)


def test_runtime_slice_projection_keeps_nominal_strategy_ownership() -> None:
    path = PROJECT_ROOT / "openhcs/core/runtime_slice_projection.py"
    strategy = _top_level_class(path, "RuntimeSliceProjectionStrategy")
    assert "NominalTypeKeyedStrategyMixin" in {
        _dotted_name(base) for base in strategy.bases
    }
    selector = _method(strategy, "strategy_for_value")
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "for_nominal_value"
        for node in ast.walk(selector)
    )


def test_source_spatial_alignment_keeps_nominal_adapter_registry_ownership() -> None:
    path = PROJECT_ROOT / "openhcs/core/source_spatial_domain.py"
    strategy = _top_level_class(path, "SourceSpatialDomainAdapter")
    assert "NominalTypeKeyedStrategyMixin" in {
        _dotted_name(base) for base in strategy.bases
    }
    assert any(
        keyword.arg == "metaclass" and _dotted_name(keyword.value) == "AutoRegisterMeta"
        for keyword in strategy.keywords
    )
    selector = _method(strategy, "for_value")
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "strategy_types_for_nominal_value"
        for node in ast.walk(selector)
    )


def test_cellprofiler_payload_alias_namespace_stays_deleted() -> None:
    retired_names = frozenset(
        {
            "CellProfilerFunction",
            "CellProfilerKwargs",
            "CellProfilerKwargDict",
            "CellProfilerRuntimeValue",
            "CellProfilerRuntimeValues",
            "CellProfilerRuntimeValueSequence",
            "CellProfilerProfileFields",
            "MeasurementRowMapping",
            "ImagePayloadValue",
            "RuntimeArtifactPayloadValue",
            "CellProfilerMeasurementSliceValues",
        }
    )
    violations = tuple(
        _format_violation(path, node.lineno, "retired payload alias", node.id)
        for path in _source_paths()
        for node in ast.walk(_parse_source(path))
        if isinstance(node, ast.Name) and node.id in retired_names
    )
    assert not violations, "\n" + "\n".join(violations)


def test_cellprofiler_projection_enters_only_runtime_slice_projection() -> None:
    roots = (
        PROJECT_ROOT / "openhcs/interop/cellprofiler/runtime",
        PROJECT_ROOT / "openhcs/processing/backends/cellprofiler",
    )
    violations: list[str] = []
    for root in roots:
        for path in root.rglob("*.py"):
            tree = _parse_source(path)
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in {"project", "project_plane_axis"}
                ):
                    violations.append(
                        _format_violation(
                            path,
                            node.lineno,
                            "direct value projection",
                            _dotted_name(node.func) or "",
                        )
                    )
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                invokes_runtime_projection = any(
                    isinstance(descendant, ast.Call)
                    and (_dotted_name(descendant.func) or "").startswith(
                        "RuntimeSliceProjection."
                    )
                    for descendant in ast.walk(node)
                )
                if not invokes_runtime_projection:
                    continue
                branch_tests = (
                    descendant.test
                    for descendant in ast.walk(node)
                    if isinstance(descendant, (ast.If, ast.IfExp, ast.While))
                )
                forbidden_branch_names = {
                    name.id
                    for branch_test in branch_tests
                    for name in ast.walk(branch_test)
                    if isinstance(name, ast.Name)
                    and name.id in {"ObjectLabelValue", "RuntimeSliceAlignedValueSet"}
                }
                for name in forbidden_branch_names:
                    violations.append(
                        _format_violation(
                            path,
                            node.lineno,
                            "parallel projection branch",
                            name,
                        )
                    )

    assert not violations, "\n" + "\n".join(sorted(set(violations)))


def test_cellprofiler_image_execution_mode_dispatch_has_one_owner() -> None:
    """Keep the closed image-mode matrix at the existing executor boundary."""

    runtime_root = PROJECT_ROOT / "openhcs/interop/cellprofiler/runtime"
    dispatches: list[tuple[str, str, str, int]] = []
    execution_method_names = frozenset(
        {
            "execute",
            "execute_pure_3d",
            "_execute_aligned_multi_image_stack",
        }
    )
    for path in runtime_root.rglob("*.py"):
        tree = _parse_source(path)
        for class_node in ast.walk(tree):
            if not isinstance(class_node, ast.ClassDef):
                continue
            for method_node in class_node.body:
                if not isinstance(method_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                for branch in ast.walk(method_node):
                    if not isinstance(branch, (ast.If, ast.Match)):
                        continue
                    dispatched_methods = frozenset(
                        node.func.attr
                        for node in ast.walk(branch)
                        if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr in execution_method_names
                    )
                    if len(dispatched_methods) < 2:
                        continue
                    dispatches.append(
                        (
                            str(path.relative_to(PROJECT_ROOT)),
                            class_node.name,
                            method_node.name,
                            branch.lineno,
                        )
                    )

    assert len(dispatches) == 1, dispatches
    path, class_name, method_name, _line = dispatches[0]
    assert (
        path,
        class_name,
        method_name,
    ) == (
        "openhcs/interop/cellprofiler/runtime/function_contract_execution.py",
        "CellProfilerFunctionContractExecutor",
        "execute",
    )


def test_generic_runtime_does_not_dispatch_on_module_name_literals() -> None:
    roots = (
        PROJECT_ROOT / "openhcs/core",
        PROJECT_ROOT / "openhcs/interop/cellprofiler/runtime",
    )
    violations: list[str] = []
    for root in roots:
        for path in root.rglob("*.py"):
            for node in ast.walk(_parse_source(path)):
                if not isinstance(node, ast.Compare):
                    continue
                operands = (node.left, *node.comparators)
                has_module_name = any(
                    isinstance(operand, ast.Attribute) and operand.attr == "module_name"
                    for operand in operands
                )
                has_string_literal = any(
                    isinstance(operand, ast.Constant) and isinstance(operand.value, str)
                    for operand in operands
                )
                if has_module_name and has_string_literal:
                    violations.append(
                        _format_violation(
                            path,
                            node.lineno,
                            "module-name dispatch",
                            ast.unparse(node),
                        )
                    )

    assert not violations, "\n" + "\n".join(violations)


def test_nominal_payloads_are_not_erased_before_semantic_projection() -> None:
    """Keep payload extraction on the numerical side of semantic boundaries."""

    type_erasing_calls = frozenset(
        {
            "np.array",
            "np.asarray",
            "image_payload_data",
            "object_label_dense_array",
            "runtime_array_operand",
        }
    )
    semantic_calls = frozenset(
        {
            "MeasurementLabelSourceAlignmentStrategy.align",
            "MeasurementLabelSourceAlignmentStrategy.align_request_labels_to_image_source",
            "RuntimeSliceProjection.value_for_slice",
            "SourceSpatialDomainAdapter.aligned_values",
        }
    )
    violations: list[str] = []
    for path in (PROJECT_ROOT / "openhcs").rglob("*.py"):
        for function in ast.walk(_parse_source(path)):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            erased_assignments: dict[str, tuple[int, str]] = {}
            for node in ast.walk(function):
                if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                    continue
                value = node.value
                if not isinstance(value, ast.Call):
                    continue
                call_name = _dotted_name(value.func)
                if call_name not in type_erasing_calls:
                    continue
                targets = (
                    node.targets if isinstance(node, ast.Assign) else (node.target,)
                )
                for target in targets:
                    for name in _assigned_names(target):
                        erased_assignments[name] = (node.lineno, call_name)

            for call in ast.walk(function):
                if not isinstance(call, ast.Call):
                    continue
                call_name = _dotted_name(call.func)
                if call_name not in semantic_calls:
                    continue
                for descendant in ast.walk(call):
                    if isinstance(descendant, ast.Call) and (
                        _dotted_name(descendant.func) in type_erasing_calls
                    ):
                        violations.append(
                            _format_violation(
                                path,
                                call.lineno,
                                "nested pre-projection type erasure",
                                ast.unparse(descendant),
                            )
                        )
                for name in (
                    descendant.id
                    for descendant in ast.walk(call)
                    if isinstance(descendant, ast.Name)
                ):
                    assignment = erased_assignments.get(name)
                    if assignment is None or assignment[0] >= call.lineno:
                        continue
                    violations.append(
                        _format_violation(
                            path,
                            call.lineno,
                            "pre-projection type erasure",
                            f"{name} from {assignment[1]} at line {assignment[0]}",
                        )
                    )

    assert not violations, "\n" + "\n".join(sorted(set(violations)))


def test_module_declaration_collapse_surface_stays_deleted() -> None:
    """Keep redundant declaration records and pass-through hooks deleted."""

    declaration_path = (
        PROJECT_ROOT / "openhcs/interop/cellprofiler/module_declarations.py"
    )
    tree = _parse_source(declaration_path)
    retired_top_level_names = frozenset(
        {
            "UnmappedModuleSetting",
            "structuring_element_bound_kwargs",
            "_normalize_setting_name",
            "_setting_name_matches",
            "_setting_values",
            "_optional_setting_value",
            "_normalized_symbol_name",
            "_parse_cellprofiler_float",
            "_parse_cellprofiler_int",
            "_parse_cellprofiler_bool",
            "_cellprofiler_setting_token",
        }
    )
    retired_member_names = frozenset(
        {
            "declared_image_input_names",
            "declared_image_output_names",
            "contribute_pipeline_source_bindings",
            "require_authority_type",
            "measurement_feature_family_parts",
            "runtime_kwargs",
            "declared_setting_name",
            "declared_setting_value",
            "artifact_output_specs_from_bindings",
            "available_artifact_input",
            "structuring_element_setting",
            "invocation_setting",
            "image_output_artifact_spec_kwargs",
            "measurement_object_output_specs",
            "spatial_grid_input_bindings_for",
            "spatial_grid_output_bindings_for",
        }
    )

    top_level_names = frozenset(
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    )
    member_names = frozenset(
        member.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        for member in node.body
        if isinstance(member, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    )

    assert retired_top_level_names.isdisjoint(top_level_names)
    assert retired_member_names.isdisjoint(member_names)


def test_cellprofiler_module_mro_has_no_reflective_semantic_dispatch() -> None:
    """Keep module behavior on nominal MRO owners instead of reflection."""

    roots = (
        PROJECT_ROOT / "openhcs/interop/cellprofiler",
        PROJECT_ROOT / "openhcs/processing/backends/cellprofiler",
    )
    definitions: dict[str, list[tuple[Path, ast.ClassDef]]] = {}
    for path in (
        path
        for root in roots
        for path in root.rglob("*.py")
        if path.resolve() != THIS_FILE
    ):
        for node in ast.walk(_parse_source(path)):
            if isinstance(node, ast.ClassDef):
                definitions.setdefault(node.name, []).append((path, node))

    def base_name(node: ast.expr) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        if isinstance(node, ast.Subscript):
            return base_name(node.value)
        return None

    module_mro = {"CellProfilerModule"}
    while True:
        discovered = {
            name
            for name, class_definitions in definitions.items()
            if any(
                base_name(base) in module_mro
                for _path, class_node in class_definitions
                for base in class_node.bases
            )
        }
        if discovered <= module_mro:
            break
        module_mro.update(discovered)

    forbidden_calls = {"getattr", "setattr", "hasattr", "issubclass"}
    violations = []
    for class_name in module_mro:
        for path, class_node in definitions.get(class_name, ()):
            for method in class_node.body:
                if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                for call in (
                    node for node in ast.walk(method) if isinstance(node, ast.Call)
                ):
                    if not isinstance(call.func, ast.Name):
                        continue
                    if call.func.id not in forbidden_calls:
                        continue
                    violations.append(
                        _format_violation(
                            path,
                            call.lineno,
                            "reflective module dispatch",
                            f"{class_name}.{method.name}:{call.func.id}",
                        )
                    )

    assert not violations, "\n" + "\n".join(sorted(violations))


def test_artifact_input_bindings_do_not_conflate_public_and_runtime_names() -> None:
    """Keep setting identity separate from special-input injection identity."""

    violations: list[str] = []
    for path in (
        PROJECT_ROOT / "openhcs/interop/cellprofiler",
        PROJECT_ROOT / "openhcs/processing/backends/cellprofiler",
    ):
        for source_path in path.rglob("*.py"):
            for node in ast.walk(_parse_source(source_path)):
                if not isinstance(node, ast.Call):
                    continue
                if _dotted_name(node.func) != "SettingToKeywordBinding.input":
                    continue
                if len(node.args) > 2:
                    violations.append(
                        _format_violation(
                            source_path,
                            node.lineno,
                            "positional runtime parameter",
                            ast.unparse(node),
                        )
                    )
                if any(keyword.arg == "parameter_name" for keyword in node.keywords):
                    violations.append(
                        _format_violation(
                            source_path,
                            node.lineno,
                            "conflated artifact parameter",
                            ast.unparse(node),
                        )
                    )

    assert not violations, "\n" + "\n".join(sorted(violations))
