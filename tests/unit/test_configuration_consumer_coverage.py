"""Generated receiver-specific coverage for Configure OpenHCS fields."""

from __future__ import annotations

import ast
import inspect
import re
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import get_args, get_type_hints

from openhcs.core.config import GlobalPipelineConfig
from openhcs.pyqt_gui.config import ShortcutConfig, get_default_ui_config
from openhcs.pyqt_gui.main import OpenHCSMainWindow
from openhcs.pyqt_gui.services.main_window_workflows import (
    MainWindowShortcutLifecycle,
)


@dataclass(frozen=True, slots=True)
class _VisibleConfigLeaf:
    path: str
    owner: type[object]
    field_name: str


@dataclass(frozen=True, slots=True)
class _SyntaxUnit:
    path: Path
    module_name: str
    tree: ast.Module


@dataclass(frozen=True, slots=True)
class _ClassScope:
    source_path: Path
    qualname: str


_TypeReference = type[object] | _ClassScope
_TypeDomain = frozenset[_TypeReference]
_OwnerIdentity = tuple[Path, str]
_ConsumedField = tuple[_OwnerIdentity, str]


def _owner_identity(owner: type[object]) -> _OwnerIdentity:
    source_path = inspect.getsourcefile(owner)
    if source_path is None:
        raise AssertionError(f"{owner!r} has no source identity.")
    return Path(source_path).resolve(), owner.__qualname__


def _declaring_owner(runtime_type: type[object], field_name: str) -> type[object]:
    """Return the oldest MRO declaration that owns ``field_name``."""

    for candidate in reversed(runtime_type.__mro__):
        if field_name in getattr(candidate, "__annotations__", {}):
            return candidate
    raise AssertionError(
        f"{runtime_type.__qualname__}.{field_name} has no nominal declaration."
    )


def _config_graph() -> tuple[
    tuple[_VisibleConfigLeaf, ...],
    frozenset[type[object]],
    dict[tuple[type[object], str], type[object]],
]:
    """Derive leaves and nested nominal edges from the actual config roots."""

    leaves: list[_VisibleConfigLeaf] = []
    config_types: set[type[object]] = set()
    nested_fields: dict[tuple[type[object], str], type[object]] = {}

    def visit(value: object, prefix: str = "") -> None:
        runtime_type = type(value)
        config_types.add(runtime_type)
        for declaration in fields(value):
            if declaration.metadata.get("ui_hidden"):
                continue
            field_value = getattr(value, declaration.name)
            path = f"{prefix}.{declaration.name}" if prefix else declaration.name
            if is_dataclass(field_value) and not isinstance(field_value, type):
                nested_fields[(runtime_type, declaration.name)] = type(field_value)
                visit(field_value, path)
                continue
            leaves.append(
                _VisibleConfigLeaf(
                    path=path,
                    owner=_declaring_owner(runtime_type, declaration.name),
                    field_name=declaration.name,
                )
            )

    visit(get_default_ui_config())
    visit(GlobalPipelineConfig())
    config_types.update(leaf.owner for leaf in leaves)
    while True:
        discovered_subclasses = {
            subclass
            for config_type in config_types
            for subclass in config_type.__subclasses__()
            if is_dataclass(subclass)
        }
        if discovered_subclasses <= config_types:
            break
        config_types.update(discovered_subclasses)
    for config_type in tuple(config_types):
        config_types.update(
            candidate
            for candidate in config_type.__mro__
            if candidate is not object
        )

    for config_type in tuple(config_types):
        try:
            hints = get_type_hints(config_type)
        except (NameError, TypeError):
            continue
        for field_name, hint in hints.items():
            hinted_types = {
                candidate
                for candidate in (hint, *get_args(hint))
                if isinstance(candidate, type) and candidate in config_types
            }
            if len(hinted_types) == 1:
                nested_fields.setdefault(
                    (config_type, field_name),
                    hinted_types.pop(),
                )

    return tuple(leaves), frozenset(config_types), nested_fields


def _production_syntax_units() -> tuple[_SyntaxUnit, ...]:
    repository_root = Path(__file__).resolve().parents[2]
    roots = (
        repository_root / "openhcs",
        repository_root / "external" / "pyqt-reactive" / "src",
        repository_root / "external" / "zmqruntime" / "src",
        repository_root / "external" / "arraybridge" / "src",
        repository_root / "external" / "PolyStore" / "src",
    )
    units: list[_SyntaxUnit] = []
    for root in roots:
        for source_path in root.rglob("*.py"):
            relative_module_path = source_path.relative_to(root).with_suffix("")
            module_parts = list(relative_module_path.parts)
            if root.name == "openhcs":
                module_parts.insert(0, "openhcs")
            if module_parts[-1] == "__init__":
                module_parts.pop()
            units.append(
                _SyntaxUnit(
                    path=source_path,
                    module_name=".".join(module_parts),
                    tree=ast.parse(
                        source_path.read_text(encoding="utf-8"),
                        filename=str(source_path),
                    ),
                )
            )
    return tuple(units)


def _functions_with_owner(
    unit: _SyntaxUnit,
) -> tuple[
    tuple[ast.FunctionDef | ast.AsyncFunctionDef, _ClassScope | None],
    ...,
]:
    functions: list[
        tuple[ast.FunctionDef | ast.AsyncFunctionDef, _ClassScope | None]
    ] = []

    class _NestedFunctionVisitor(ast.NodeVisitor):
        """Collect nested callables without assigning enclosing class ownership."""

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            functions.append((node, None))
            for statement in node.body:
                self.visit(statement)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            functions.append((node, None))
            for statement in node.body:
                self.visit(statement)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            return

    def visit(
        statements: list[ast.stmt],
        owner_qualname: str | None = None,
    ) -> None:
        for statement in statements:
            if isinstance(statement, ast.ClassDef):
                class_qualname = (
                    f"{owner_qualname}.{statement.name}"
                    if owner_qualname
                    else statement.name
                )
                visit(statement.body, class_qualname)
            elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                owner = (
                    _ClassScope(unit.path, owner_qualname)
                    if owner_qualname
                    else None
                )
                functions.append((statement, owner))
                nested_visitor = _NestedFunctionVisitor()
                for nested_statement in statement.body:
                    nested_visitor.visit(nested_statement)

    visit(unit.tree.body)
    return tuple(functions)


def _module_imports(unit: _SyntaxUnit) -> tuple[ast.ImportFrom, ...]:
    """Return module imports plus direct ``TYPE_CHECKING`` imports only."""

    imports: list[ast.ImportFrom] = []
    for statement in unit.tree.body:
        if isinstance(statement, ast.ImportFrom):
            imports.append(statement)
            continue
        if not (
            isinstance(statement, ast.If)
            and (
                (
                    isinstance(statement.test, ast.Name)
                    and statement.test.id == "TYPE_CHECKING"
                )
                or (
                    isinstance(statement.test, ast.Attribute)
                    and isinstance(statement.test.value, ast.Name)
                    and statement.test.value.id == "typing"
                    and statement.test.attr == "TYPE_CHECKING"
                )
            )
        ):
            continue
        imports.extend(
            nested
            for nested in statement.body
            if isinstance(nested, ast.ImportFrom)
        )
    return tuple(imports)


def _scope_nodes(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[ast.AST, ...]:
    """Return nodes owned by one function without nested-scope leakage."""

    nodes: list[ast.AST] = []

    def visit(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(
                child,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda),
            ):
                continue
            nodes.append(child)
            visit(child)

    visit(function)
    return tuple(nodes)


def _read_scope_nodes(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[tuple[ast.AST, frozenset[str]], ...]:
    """Return reads in a function and its lambdas with shadowed names tracked."""

    nodes: list[tuple[ast.AST, frozenset[str]]] = []

    def visit(node: ast.AST, shadowed: frozenset[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(child, ast.Lambda):
                bound_names = frozenset(
                    argument.arg
                    for argument in (
                        *child.args.posonlyargs,
                        *child.args.args,
                        *child.args.kwonlyargs,
                    )
                )
                if child.args.vararg is not None:
                    bound_names |= frozenset((child.args.vararg.arg,))
                if child.args.kwarg is not None:
                    bound_names |= frozenset((child.args.kwarg.arg,))
                visit(child.body, shadowed | bound_names)
                continue
            nodes.append((child, shadowed))
            visit(child, shadowed)

    visit(function, frozenset())
    return tuple(nodes)


class _TypedAttributeFlow:
    """Infer config receiver types through annotations and assignments."""

    def __init__(
        self,
        config_types: frozenset[type[object]],
        nested_fields: dict[tuple[type[object], str], type[object]],
        units: tuple[_SyntaxUnit, ...],
    ) -> None:
        self._config_types = config_types
        self._nested_fields = nested_fields
        self._units = units
        types_by_name: dict[str, set[type[object]]] = {}
        for config_type in config_types:
            types_by_name.setdefault(config_type.__name__, set()).add(config_type)
        self._types_by_name = {
            name: frozenset(types)
            for name, types in types_by_name.items()
        }
        types_by_source_qualname: dict[
            tuple[Path, str],
            set[type[object]],
        ] = {}
        for config_type in config_types:
            source_path = inspect.getsourcefile(config_type)
            if source_path is None:
                continue
            types_by_source_qualname.setdefault(
                (
                    Path(source_path).resolve(),
                    config_type.__qualname__,
                ),
                set(),
            ).add(config_type)
        self._types_by_source_qualname = {
            identity: frozenset(types)
            for identity, types in types_by_source_qualname.items()
        }
        self._class_scopes_by_module_name: dict[
            tuple[str, str],
            _ClassScope,
        ] = {}
        for unit in units:
            for statement in unit.tree.body:
                if isinstance(statement, ast.ClassDef):
                    self._class_scopes_by_module_name[
                        (unit.module_name, statement.name)
                    ] = _ClassScope(unit.path.resolve(), statement.name)
        self._class_attribute_types: dict[
            tuple[_ClassScope, str],
            _TypeDomain,
        ] = {}
        self._type_aliases_by_source = {
            unit.path.resolve(): self._type_aliases(unit)
            for unit in units
        }
        self._function_return_aliases_by_source: dict[
            Path,
            dict[str, _TypeDomain],
        ] = {}
        self._method_return_types: dict[
            tuple[_ClassScope, str],
            _TypeDomain,
        ] = {}
        self._function_scopes = tuple(
            (unit, function, owner)
            for unit in units
            for function, owner in _functions_with_owner(unit)
        )
        self._scope_nodes_by_function = {
            function: _scope_nodes(function)
            for _unit, function, _owner in self._function_scopes
        }
        self._read_scope_nodes_by_function = {
            function: _read_scope_nodes(function)
            for _unit, function, _owner in self._function_scopes
        }
        self._index_declared_class_attributes()
        self._index_callable_return_types()

    def _type_aliases(self, unit: _SyntaxUnit) -> dict[str, _TypeDomain]:
        aliases: dict[str, set[_TypeReference]] = {}
        source_path = unit.path.resolve()
        for (
            candidate_source,
            candidate_qualname,
        ), config_types in self._types_by_source_qualname.items():
            if candidate_source == source_path and "." not in candidate_qualname:
                aliases.setdefault(candidate_qualname, set()).update(config_types)
        for (
            candidate_module,
            candidate_name,
        ), candidate_scope in self._class_scopes_by_module_name.items():
            if candidate_module != unit.module_name:
                continue
            if (
                candidate_scope.source_path,
                candidate_scope.qualname,
            ) not in self._types_by_source_qualname:
                aliases.setdefault(candidate_name, set()).add(candidate_scope)
        for statement in _module_imports(unit):
            if statement.module is None:
                continue
            imported_module = self._resolved_import_module(unit, statement)
            for imported in statement.names:
                exact_types = {
                    candidate
                    for candidate in self._types_by_name.get(
                        imported.name,
                        frozenset(),
                    )
                    if candidate.__module__ == imported_module
                }
                if exact_types:
                    aliases.setdefault(
                        imported.asname or imported.name,
                        set(),
                    ).update(exact_types)
                    continue
                candidate_scope = self._class_scopes_by_module_name.get(
                    (imported_module, imported.name)
                )
                if candidate_scope is not None:
                    aliases.setdefault(
                        imported.asname or imported.name,
                        set(),
                    ).add(candidate_scope)
        return {
            name: frozenset(candidates)
            for name, candidates in aliases.items()
        }

    @staticmethod
    def _resolved_import_module(
        unit: _SyntaxUnit,
        statement: ast.ImportFrom,
    ) -> str:
        imported_module = statement.module or ""
        if not statement.level:
            return imported_module
        package_parts = unit.module_name.split(".")[:-statement.level]
        return ".".join(
            part
            for part in (*package_parts, imported_module)
            if part
        )

    def _annotation_types(
        self,
        annotation: ast.expr | None,
        source_path: Path,
    ) -> _TypeDomain:
        if annotation is None:
            return frozenset()
        annotation_source = ast.unparse(annotation)
        return frozenset(
            config_type
            for name, config_types in self._type_aliases_by_source[
                source_path.resolve()
            ].items()
            if re.search(rf"\b{re.escape(name)}\b", annotation_source)
            for config_type in config_types
        )

    def _index_declared_class_attributes(self) -> None:
        """Index exact class-body annotations without importing source modules."""

        def visit(
            statements: list[ast.stmt],
            source_path: Path,
            owner_qualname: str | None = None,
        ) -> None:
            for statement in statements:
                if not isinstance(statement, ast.ClassDef):
                    continue
                qualname = (
                    f"{owner_qualname}.{statement.name}"
                    if owner_qualname
                    else statement.name
                )
                owner = _ClassScope(source_path.resolve(), qualname)
                for member in statement.body:
                    if (
                        isinstance(member, ast.AnnAssign)
                        and isinstance(member.target, ast.Name)
                    ):
                        declared_types = self._annotation_types(
                            member.annotation,
                            source_path,
                        )
                        if declared_types:
                            self._class_attribute_types[
                                (owner, member.target.id)
                            ] = declared_types
                visit(statement.body, source_path, qualname)

        for unit in self._units:
            visit(unit.tree.body, unit.path)

    def _index_callable_return_types(self) -> None:
        """Index exact local/imported callables and method return annotations."""

        module_returns: dict[tuple[str, str], _TypeDomain] = {}
        for unit in self._units:
            for statement in unit.tree.body:
                if not isinstance(
                    statement,
                    (ast.FunctionDef, ast.AsyncFunctionDef),
                ):
                    continue
                return_types = self._annotation_types(
                    statement.returns,
                    unit.path,
                )
                if return_types:
                    module_returns[(unit.module_name, statement.name)] = return_types

        for unit, function, owner in self._function_scopes:
            return_types = self._annotation_types(function.returns, unit.path)
            if return_types and owner is not None:
                key = (owner, function.name)
                self._method_return_types[key] = (
                    self._method_return_types.get(key, frozenset())
                    | return_types
                )

        for unit in self._units:
            aliases: dict[str, _TypeDomain] = {
                name: return_types
                for (module_name, name), return_types in module_returns.items()
                if module_name == unit.module_name
            }
            for statement in _module_imports(unit):
                imported_module = self._resolved_import_module(unit, statement)
                for imported in statement.names:
                    return_types = module_returns.get(
                        (imported_module, imported.name)
                    )
                    if return_types:
                        aliases[imported.asname or imported.name] = return_types
            self._function_return_aliases_by_source[
                unit.path.resolve()
            ] = aliases

    @staticmethod
    def _runtime_class_scope(
        runtime_type: type[object],
    ) -> _ClassScope | None:
        try:
            source_path = inspect.getsourcefile(runtime_type)
        except TypeError:
            return None
        if source_path is None:
            return None
        return _ClassScope(
            Path(source_path).resolve(),
            runtime_type.__qualname__,
        )

    def _method_return_type(
        self,
        receiver: _TypeReference,
        method_name: str,
    ) -> _TypeDomain:
        if isinstance(receiver, _ClassScope):
            scopes = (receiver,)
        else:
            scopes = tuple(
                scope
                for candidate in receiver.__mro__
                if (scope := self._runtime_class_scope(candidate)) is not None
            )
        return frozenset().union(
            *(
                self._method_return_types.get(
                    (scope, method_name),
                    frozenset(),
                )
                for scope in scopes
            )
        )

    def _nested_type(
        self,
        config_type: type[object],
        field_name: str,
    ) -> type[object] | None:
        for candidate in config_type.__mro__:
            nested_type = self._nested_fields.get((candidate, field_name))
            if nested_type is not None:
                return nested_type
        return None

    def _attribute_type(
        self,
        receiver: _TypeReference,
        attribute_name: str,
    ) -> _TypeDomain:
        if isinstance(receiver, _ClassScope):
            return self._class_attribute_types.get(
                (receiver, attribute_name),
                frozenset(),
            )
        nested_type = self._nested_type(receiver, attribute_name)
        return (
            frozenset((nested_type,))
            if nested_type is not None
            else frozenset()
        )

    def _infer(
        self,
        expression: ast.expr | None,
        environment: dict[str, _TypeDomain],
        owner: _ClassScope | None,
        source_path: Path,
    ) -> _TypeDomain:
        if expression is None:
            return frozenset()
        if isinstance(expression, ast.Name):
            return environment.get(expression.id, frozenset())
        if isinstance(expression, ast.Call):
            if isinstance(expression.func, ast.Name):
                return (
                    self._type_aliases_by_source[source_path.resolve()].get(
                        expression.func.id,
                        frozenset(),
                    )
                    | self._function_return_aliases_by_source[
                        source_path.resolve()
                    ].get(
                        expression.func.id,
                        frozenset(),
                    )
                )
            if isinstance(expression.func, ast.Attribute):
                receivers = self._infer(
                    expression.func.value,
                    environment,
                    owner,
                    source_path,
                )
                if (
                    isinstance(expression.func.value, ast.Name)
                    and expression.func.value.id == "self"
                    and owner is not None
                ):
                    receivers |= frozenset((owner,))
                return frozenset().union(
                    *(
                        self._method_return_type(
                            receiver,
                            expression.func.attr,
                        )
                        for receiver in receivers
                    )
                )
            return frozenset()
        if isinstance(expression, ast.Attribute):
            if (
                isinstance(expression.value, ast.Name)
                and expression.value.id == "self"
                and owner is not None
            ):
                owned_type = self._class_attribute_types.get(
                    (owner, expression.attr)
                )
                if owned_type:
                    return owned_type
            base_types = self._infer(
                expression.value,
                environment,
                owner,
                source_path,
            )
            return frozenset().union(
                *(
                    self._attribute_type(
                        base_type,
                        expression.attr,
                    )
                    for base_type in base_types
                )
            )
        if isinstance(expression, ast.IfExp):
            return self._infer(
                expression.body,
                environment,
                owner,
                source_path,
            ) | self._infer(
                expression.orelse,
                environment,
                owner,
                source_path,
            )
        if isinstance(expression, ast.BoolOp):
            return frozenset().union(
                *(
                    self._infer(value, environment, owner, source_path)
                    for value in expression.values
                )
            )
        return frozenset()

    def _function_environment(
        self,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        owner: _ClassScope | None,
        source_path: Path,
    ) -> dict[str, _TypeDomain]:
        environment = {
            argument.arg: annotation_types
            for argument in (
                *function.args.posonlyargs,
                *function.args.args,
                *function.args.kwonlyargs,
            )
            if (
                annotation_types := self._annotation_types(
                    argument.annotation,
                    source_path,
                )
            )
        }
        owner_types = (
            self._types_by_source_qualname.get(
                (owner.source_path.resolve(), owner.qualname),
                frozenset(),
            )
            if owner is not None
            else frozenset()
        )
        if owner is not None:
            environment["self"] = frozenset((owner,)) | frozenset(
                config_type
                for config_type in self._config_types
                if any(
                    owner_type in config_type.__mro__
                    for owner_type in owner_types
                )
            )
        return environment

    def _assignment_type(
        self,
        assignment: ast.Assign | ast.AnnAssign,
        environment: dict[str, _TypeDomain],
        owner: _ClassScope | None,
        source_path: Path,
    ) -> tuple[tuple[ast.expr, ...], _TypeDomain]:
        if isinstance(assignment, ast.AnnAssign):
            return (
                (assignment.target,),
                self._annotation_types(assignment.annotation, source_path)
                or self._infer(
                    assignment.value,
                    environment,
                    owner,
                    source_path,
                ),
            )
        return (
            tuple(assignment.targets),
            self._infer(
                assignment.value,
                environment,
                owner,
                source_path,
            ),
        )

    def _apply_local_assignments(
        self,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        environment: dict[str, _TypeDomain],
        owner: _ClassScope | None,
        source_path: Path,
    ) -> None:
        while True:
            changed = False
            for node in self._scope_nodes_by_function[function]:
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "isinstance"
                    and len(node.args) == 2
                    and isinstance(node.args[0], ast.Name)
                    and isinstance(node.args[1], ast.Name)
                ):
                    narrowed_types = self._type_aliases_by_source[
                        source_path.resolve()
                    ].get(node.args[1].id, frozenset())
                    if narrowed_types:
                        variable_name = node.args[0].id
                        previous = environment.get(variable_name, frozenset())
                        combined = previous | narrowed_types
                        if combined != previous:
                            environment[variable_name] = combined
                            changed = True
                if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                    continue
                targets, assigned_types = self._assignment_type(
                    node,
                    environment,
                    owner,
                    source_path,
                )
                if not assigned_types:
                    continue
                for target in targets:
                    if (
                        isinstance(target, ast.Name)
                    ):
                        previous = environment.get(target.id, frozenset())
                        combined = previous | assigned_types
                        if combined != previous:
                            environment[target.id] = combined
                            changed = True
            if not changed:
                break

    def _collect_class_attribute_types(self) -> None:
        while True:
            changed = False
            for unit, function, owner in self._function_scopes:
                if owner is None:
                    continue
                environment = self._function_environment(
                    function,
                    owner,
                    unit.path,
                )
                self._apply_local_assignments(
                    function,
                    environment,
                    owner,
                    unit.path,
                )
                for node in self._scope_nodes_by_function[function]:
                    if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                        continue
                    targets, assigned_types = self._assignment_type(
                        node,
                        environment,
                        owner,
                        unit.path,
                    )
                    if not assigned_types:
                        continue
                    for target in targets:
                        if (
                            isinstance(target, ast.Attribute)
                            and isinstance(target.value, ast.Name)
                            and target.value.id == "self"
                        ):
                            key = (owner, target.attr)
                            previous = self._class_attribute_types.get(
                                key,
                                frozenset(),
                            )
                            combined = previous | assigned_types
                            if combined != previous:
                                self._class_attribute_types[key] = combined
                                changed = True
            if not changed:
                break

    @staticmethod
    def _declares_field(config_type: type[object], field_name: str) -> bool:
        return any(
            field_name in getattr(candidate, "__annotations__", {})
            for candidate in config_type.__mro__
        )

    def _field_owners_for_receiver(
        self,
        receiver_type: type[object],
        field_name: str,
    ) -> frozenset[type[object]]:
        """Project a base-typed receiver to every concrete config override."""

        return frozenset(
            _declaring_owner(config_type, field_name)
            for config_type in self._config_types
            if receiver_type in config_type.__mro__
            and self._declares_field(config_type, field_name)
        )

    @staticmethod
    def _shortcut_projection_fields() -> frozenset[
        _ConsumedField
    ]:
        """Resolve shortcut lambdas through their exact typed lifecycle owner."""

        binding_methods: set[str] = set()
        for method_name, method in inspect.getmembers(
            MainWindowShortcutLifecycle,
            predicate=inspect.isfunction,
        ):
            try:
                hints = get_type_hints(
                    method,
                    localns={"ShortcutConfig": ShortcutConfig},
                )
            except (NameError, TypeError):
                continue
            callable_signature = get_args(
                hints.get("key_from_config")
            )
            if callable_signature == ([ShortcutConfig], str):
                binding_methods.add(method_name)
        assert binding_methods

        main_window_tree = ast.parse(inspect.getsource(OpenHCSMainWindow))
        owns_lifecycle = any(
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and target.attr == "shortcut_lifecycle"
                for target in node.targets
            )
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == MainWindowShortcutLifecycle.__name__
            for node in ast.walk(main_window_tree)
        )
        assert owns_lifecycle

        consumed: set[_ConsumedField] = set()
        for node in ast.walk(main_window_tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in binding_methods
                and isinstance(node.func.value, ast.Attribute)
                and isinstance(node.func.value.value, ast.Name)
                and node.func.value.value.id == "self"
                and node.func.value.attr == "shortcut_lifecycle"
                and node.args
                and isinstance(node.args[0], ast.Lambda)
                and node.args[0].args.args
            ):
                continue
            projection = node.args[0]
            parameter_name = projection.args.args[0].arg
            for nested in ast.walk(projection.body):
                if (
                    isinstance(nested, ast.Attribute)
                    and isinstance(nested.ctx, ast.Load)
                    and isinstance(nested.value, ast.Name)
                    and nested.value.id == parameter_name
                    and _TypedAttributeFlow._declares_field(
                        ShortcutConfig,
                        nested.attr,
                    )
                ):
                    consumed.add(
                        (
                            _owner_identity(
                                _declaring_owner(ShortcutConfig, nested.attr)
                            ),
                            nested.attr,
                        )
                    )
        return frozenset(consumed)

    def consumed_fields(
        self,
    ) -> frozenset[_ConsumedField]:
        self._collect_class_attribute_types()
        consumed = set(self._shortcut_projection_fields())
        for unit, function, owner in self._function_scopes:
            environment = self._function_environment(
                function,
                owner,
                unit.path,
            )
            self._apply_local_assignments(
                function,
                environment,
                owner,
                unit.path,
            )
            for node, shadowed_names in self._read_scope_nodes_by_function[
                function
            ]:
                if (
                    isinstance(node, ast.Attribute)
                    and isinstance(node.ctx, ast.Load)
                ):
                    read_environment = {
                        name: types
                        for name, types in environment.items()
                        if name not in shadowed_names
                    }
                    for base_type in self._infer(
                        node.value,
                        read_environment,
                        owner,
                        unit.path,
                    ):
                        if not isinstance(base_type, type):
                            continue
                        for field_owner in self._field_owners_for_receiver(
                            base_type,
                            node.attr,
                        ):
                            consumed.add(
                                (
                                    _owner_identity(field_owner),
                                    node.attr,
                                )
                            )
        return frozenset(consumed)


def test_every_visible_config_leaf_has_typed_production_consumer() -> None:
    """Reject rendered options without a receiver-specific production read."""

    leaves, config_types, nested_fields = _config_graph()
    syntax_units = _production_syntax_units()
    consumed_fields = _TypedAttributeFlow(
        config_types,
        nested_fields,
        syntax_units,
    ).consumed_fields()
    missing = {
        leaf.path: f"{leaf.owner.__qualname__}.{leaf.field_name}"
        for leaf in leaves
        if (_owner_identity(leaf.owner), leaf.field_name) not in consumed_fields
    }

    assert leaves
    assert not missing, "\n".join(
        f"{path}: {owner_field}"
        for path, owner_field in sorted(missing.items())
    )
