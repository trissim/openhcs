from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, fields
from enum import Enum
from pathlib import Path

from openhcs.core.callable_contract import CallableContract, CallableImportIdentity
from openhcs.core.function_reference import (
    FunctionReference,
    FunctionReferenceTransportAuthority,
)
from openhcs.core.steps.function_step import FunctionStep
from objectstate.field_access import DataclassFieldAccess, DottedFieldPath
from python_introspect import parameter_exclusions
from pyqt_reactive.pattern_metadata import PatternScopeToken
from objectstate.lazy_factory import LazyDataclass

from pycodify import FormatContext, SourceFormatter, SourceFragment, to_source


@dataclass(frozen=True)
class CallableExportIdentity:
    import_identity: CallableImportIdentity | None

    @classmethod
    def from_callable(cls, func) -> "CallableExportIdentity":
        if isinstance(func, type):
            return cls(
                import_identity=CallableImportIdentity(
                    module_name=func.__module__,
                    function_name=func.__name__,
                ),
            )
        if not (inspect.isfunction(func) or inspect.isbuiltin(func)):
            return cls(import_identity=None)
        try:
            reference = FunctionReferenceTransportAuthority.function_reference(func)
        except RuntimeError:
            return cls(import_identity=None)
        return cls(import_identity=reference.import_identity)

    @property
    def is_importable(self) -> bool:
        return self.import_identity is not None

    @property
    def import_module(self) -> str:
        if self.import_identity is None:
            raise ValueError("Callable identity has no importable module.")
        return self.import_identity.module_name

    @property
    def import_name(self) -> str:
        if self.import_identity is None:
            raise ValueError("Callable identity has no importable name.")
        return self.import_identity.function_name


class NameMappingLookup:
    @staticmethod
    def resolve(
        context: FormatContext,
        import_pair: tuple[str, str],
        default_name: str,
    ) -> str:
        if import_pair in context.name_mappings:
            return context.name_mappings[import_pair]
        return default_name


class OpenHCSCallableFormatter(SourceFormatter):
    priority = 75

    def can_format(self, value) -> bool:
        return callable(value)

    def format(self, value, context: FormatContext) -> SourceFragment:
        if inspect.ismethod(value):
            return SourceFragment(repr(value), frozenset())

        identity = CallableExportIdentity.from_callable(value)
        if not identity.is_importable:
            return SourceFragment(repr(value), frozenset())

        import_pair = (identity.import_module, identity.import_name)
        mapped = NameMappingLookup.resolve(context, import_pair, identity.import_name)
        imports = (
            frozenset()
            if identity.import_module == "builtins"
            else frozenset([import_pair])
        )
        return SourceFragment(mapped, imports)


class FunctionReferenceFormatter(SourceFormatter):
    """Render compiler references from their declared import identity.

    Formatting is a declaration operation. It must not resolve the callable or
    initialize the execution process's registry catalog.
    """

    priority = 76

    def can_format(self, value) -> bool:
        return isinstance(value, FunctionReference)

    def format(
        self,
        value: FunctionReference,
        context: FormatContext,
    ) -> SourceFragment:
        import_pair = (value.original_module, value.function_name)
        mapped = NameMappingLookup.resolve(
            context,
            import_pair,
            value.function_name,
        )
        return SourceFragment(mapped, frozenset((import_pair,)))


class PythonSourceLiteralFormatter(SourceFormatter):
    priority = 120

    def can_format(self, value) -> bool:
        from openhcs.core.python_source_literal import PythonSourceLiteral

        return isinstance(value, PythonSourceLiteral)

    def format(self, value, context: FormatContext) -> SourceFragment:
        from openhcs.core.python_source_literal import PythonSourceLiteral

        if not isinstance(value, PythonSourceLiteral):
            raise TypeError(
                "PythonSourceLiteralFormatter requires PythonSourceLiteral, "
                f"got {type(value).__name__}."
            )
        return SourceFragment(value.source_literal(), value.source_literal_imports())


class OpenHCSPathFormatter(SourceFormatter):
    """Collect and substitute document-scoped path expressions."""

    priority = 118

    def can_format(self, value) -> bool:
        return isinstance(value, Path)

    def format(self, value: Path, context: FormatContext) -> SourceFragment:
        from openhcs.serialization.source_path_factoring import (
            SourcePathFactoringPlan,
            SourcePathOccurrenceCollector,
        )

        collector = context.extension(SourcePathOccurrenceCollector)
        if collector is not None:
            collector.record(value)
        plan = context.extension(SourcePathFactoringPlan)
        if plan is not None:
            expression = plan.expression_for(value)
            if expression is not None:
                return to_source(expression, context)

        import_pair = ("pathlib", "Path")
        name = NameMappingLookup.resolve(context, import_pair, "Path")
        return SourceFragment(
            f"{name}({str(value)!r})",
            frozenset((import_pair,)),
        )


class EnumMemberFormatter(SourceFormatter):
    priority = 115

    def can_format(self, value) -> bool:
        return isinstance(value, Enum)

    def format(self, value, context: FormatContext) -> SourceFragment:
        enum_type = type(value)
        if "<locals>" in enum_type.__qualname__:
            return SourceFragment(repr(value), frozenset())

        root_name, _, nested_path = enum_type.__qualname__.partition(".")
        import_pair = (enum_type.__module__, root_name)
        mapped_root = NameMappingLookup.resolve(context, import_pair, root_name)
        enum_reference = f"{mapped_root}.{nested_path}" if nested_path else mapped_root
        return SourceFragment(
            f"{enum_reference}.{value.name}",
            frozenset([import_pair]),
        )


class MaterializationSpecFormatter(SourceFormatter):
    priority = 110

    def can_format(self, value) -> bool:
        from openhcs.processing.materialization.core import MaterializationSpec

        return isinstance(value, MaterializationSpec)

    def format(self, value, context: FormatContext) -> SourceFragment:
        import_pair = (
            "openhcs.processing.materialization.core",
            "MaterializationSpec",
        )
        class_name = NameMappingLookup.resolve(
            context, import_pair, "MaterializationSpec"
        )
        item_ctx = context.indented()
        output_frags = [to_source(output, item_ctx) for output in value.outputs]
        imports = {import_pair}
        for frag in output_frags:
            imports |= set(frag.imports)

        args = [frag.code for frag in output_frags]
        if value.allowed_backends is not None:
            allowed_backends_frag = to_source(value.allowed_backends, item_ctx)
            imports |= set(allowed_backends_frag.imports)
            args.append(f"allowed_backends={allowed_backends_frag.code}")
        if value.primary != 0:
            args.append(f"primary={value.primary}")
        from openhcs.processing.materialization.constants import WriteMode

        if value.write_mode is not WriteMode.OVERWRITE:
            write_mode_frag = to_source(value.write_mode, item_ctx)
            imports |= set(write_mode_frag.imports)
            args.append(f"write_mode={write_mode_frag.code}")

        inner = f",\n{item_ctx.indent_str}".join(args)
        return SourceFragment(
            f"{class_name}(\n{item_ctx.indent_str}{inner}\n{context.indent_str})",
            frozenset(imports),
        )


def _public_pattern_callable(value: object) -> Callable | None:
    if isinstance(value, FunctionReference):
        return value.resolve()
    return value if callable(value) else None


def _is_pattern_tuple(value) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and _public_pattern_callable(value[0]) is not None
        and isinstance(value[1], dict)
    )


def _is_pattern_item(value) -> bool:
    return _public_pattern_callable(value) is not None or _is_pattern_tuple(value)


def _strip_internal_pattern_metadata(args):
    """Remove UI-only metadata keys from function-pattern kwargs."""
    if not isinstance(args, dict):
        return {}
    return PatternScopeToken.without_token(args)


def _exported_callable_parameter_exclusions(func) -> set[str]:
    """Return parameters that must not be emitted for a callable wrapper."""

    func = _public_pattern_callable(func)
    if func is None:
        raise TypeError("Function-pattern leaves must resolve to public callables.")
    contract = CallableContract.from_callable(func)
    excluded = set(parameter_exclusions(func))
    excluded.update(contract.runtime_owned_parameter_names)
    raw_func = contract.raw_processing_function
    if callable(raw_func):
        excluded.update(parameter_exclusions(raw_func))
    return excluded


class FunctionPatternTupleFormatter(SourceFormatter):
    priority = 85

    def can_format(self, value) -> bool:
        return _is_pattern_tuple(value)

    def format(self, value, context: FormatContext) -> SourceFragment:
        func, args = value
        public_func = _public_pattern_callable(func)
        if public_func is None:
            raise TypeError("Function-pattern leaves must resolve to public callables.")
        args = _strip_internal_pattern_metadata(args)
        hidden_parameters = _exported_callable_parameter_exclusions(public_func)
        if hidden_parameters:
            args = {
                name: arg_value
                for name, arg_value in args.items()
                if name not in hidden_parameters
            }

        if not args and context.clean_mode:
            return to_source(public_func, context)

        if context.clean_mode:
            try:
                defaults = {
                    k: v.default
                    for k, v in inspect.signature(public_func).parameters.items()
                    if v.default is not inspect.Parameter.empty
                }
            except (ValueError, TypeError):
                defaults = {}

            final_args = {
                k: v for k, v in args.items() if k not in defaults or v != defaults[k]
            }
        else:
            final_args = args

        if not final_args and context.clean_mode:
            return to_source(public_func, context)

        declared_paths = CallableContract.from_callable(
            public_func
        ).declared_path_parameters
        final_args = {
            name: (
                Path(arg_value)
                if name in declared_paths and isinstance(arg_value, str)
                else arg_value
            )
            for name, arg_value in final_args.items()
        }

        func_frag = to_source(public_func, context)
        args_frag = to_source(final_args, context.indented())
        code = f"({func_frag.code}, {args_frag.code})"
        imports = func_frag.imports | args_frag.imports
        return SourceFragment(code, imports)


class FunctionPatternListFormatter(SourceFormatter):
    priority = 84

    def can_format(self, value) -> bool:
        return (
            isinstance(value, list)
            and value
            and all(_is_pattern_item(item) for item in value)
        )

    def format(self, value: list, context: FormatContext) -> SourceFragment:
        if context.clean_mode and len(value) == 1:
            return to_source(value[0], context)

        item_ctx = context.indented()
        item_frags = [to_source(item, item_ctx) for item in value]
        imports = frozenset().union(*(frag.imports for frag in item_frags))
        inner = f",\n{item_ctx.indent_str}".join(frag.code for frag in item_frags)
        code = f"[\n{item_ctx.indent_str}{inner}\n{context.indent_str}]"
        return SourceFragment(code, imports)


@dataclass(frozen=True)
class LazyDataclassFieldEmissionState:
    explicit_field_names: frozenset[str]
    has_concrete_field_values: bool

    @classmethod
    def from_instance(cls, value: LazyDataclass) -> "LazyDataclassFieldEmissionState":
        raw_values = value.__dict__
        explicit_field_names = frozenset(raw_values["_explicitly_set_fields"])
        has_concrete_field_values = any(
            raw_values[field.name] is not None for field in fields(value)
        )
        return cls(
            explicit_field_names=explicit_field_names,
            has_concrete_field_values=has_concrete_field_values,
        )

    @property
    def requires_serialization(self) -> bool:
        return bool(self.explicit_field_names) or self.has_concrete_field_values

    @staticmethod
    def value_requires_clean_serialization(value) -> bool:
        if value is None:
            return False
        if isinstance(value, LazyDataclass):
            return LazyDataclassFieldEmissionState.from_instance(
                value
            ).requires_serialization
        return True

    @staticmethod
    def raw_field_values_match(left: LazyDataclass, right: LazyDataclass) -> bool:
        if type(left) is not type(right):
            return False
        left_values = left.__dict__
        right_values = right.__dict__
        for field in fields(left):
            left_value = left_values[field.name]
            right_value = right_values[field.name]
            if isinstance(left_value, LazyDataclass) and isinstance(
                right_value,
                LazyDataclass,
            ):
                if not LazyDataclassFieldEmissionState.raw_field_values_match(
                    left_value,
                    right_value,
                ):
                    return False
            elif left_value != right_value:
                return False
        return True

    @classmethod
    def retain_only_authored_paths(
        cls,
        value: LazyDataclass,
        authored_field_paths: frozenset[str] | set[str],
    ) -> LazyDataclass:
        """Restore constructor intent after a flattened ObjectState rebuild.

        ObjectState reconstructs lazy dataclasses from every flattened field so
        their constructors can preserve raw ``None`` inheritance markers. That
        necessarily marks every constructor keyword as explicit. Clean source
        generation instead needs the raw-vs-signature paths already owned by
        ObjectState, so apply those paths to the existing lazy metadata without
        changing field values.
        """

        paths = tuple(DottedFieldPath(path) for path in authored_field_paths)
        return cls._copy_with_only_authored_paths(
            value,
            prefix=DottedFieldPath(""),
            authored_field_paths=paths,
        )

    @classmethod
    def _copy_with_only_authored_paths(
        cls,
        value: LazyDataclass,
        *,
        prefix: DottedFieldPath,
        authored_field_paths: tuple[DottedFieldPath, ...],
    ) -> LazyDataclass:
        field_values = {}
        for field in fields(value):
            field_value = DataclassFieldAccess.raw_value(value, field.name)
            if isinstance(field_value, LazyDataclass):
                field_value = cls._copy_with_only_authored_paths(
                    field_value,
                    prefix=prefix.child(field.name),
                    authored_field_paths=authored_field_paths,
                )
            field_values[field.name] = field_value

        projected = type(value)(**field_values)
        explicit_field_names = {
            field.name
            for field in fields(projected)
            if prefix.child(field.name).contains_any(authored_field_paths)
        }
        object.__setattr__(projected, "_explicitly_set_fields", explicit_field_names)
        return projected


@dataclass(frozen=True)
class LazyDataclassSerializedField:
    name: str
    fragment: SourceFragment


@dataclass(frozen=True)
class LazyDataclassSerializationPlan:
    class_name: str
    import_pair: tuple[str, str]
    fields: tuple[LazyDataclassSerializedField, ...]

    @classmethod
    def from_instance(
        cls,
        value: LazyDataclass,
        context: FormatContext,
    ) -> "LazyDataclassSerializationPlan":
        lazy_class = type(value)
        import_pair = (lazy_class.__module__, lazy_class.__name__)
        if import_pair in context.name_mappings:
            class_name = context.name_mappings[import_pair]
        else:
            class_name = lazy_class.__name__

        raw_values = value.__dict__
        explicit_field_names = raw_values["_explicitly_set_fields"]
        field_ctx = context.indented()
        serialized_fields = []

        for field in fields(value):
            current_value = raw_values[field.name]
            if context.clean_mode and current_value is None:
                continue
            if context.clean_mode and field.name not in explicit_field_names:
                if not LazyDataclassFieldEmissionState.value_requires_clean_serialization(
                    current_value
                ):
                    continue

            serialized_fields.append(
                LazyDataclassSerializedField(
                    name=field.name,
                    fragment=to_source(current_value, field_ctx),
                )
            )

        return cls(
            class_name=class_name,
            import_pair=import_pair,
            fields=tuple(serialized_fields),
        )


class LazyDataclassFormatEligibility:
    @staticmethod
    def accepts(candidate) -> bool:
        return isinstance(candidate, LazyDataclass)


class FunctionStepCleanModeFieldPolicy:
    def should_emit(
        self,
        field_name,
        current_value,
        default_value,
        context: FormatContext,
    ) -> bool:
        if not context.clean_mode:
            return True

        if isinstance(current_value, LazyDataclass):
            emission_state = LazyDataclassFieldEmissionState.from_instance(
                current_value
            )
            if not emission_state.requires_serialization:
                return False
            if isinstance(
                default_value, LazyDataclass
            ) and LazyDataclassFieldEmissionState.raw_field_values_match(
                current_value,
                default_value,
            ):
                return False
            return True

        return current_value != default_value


class LazyDataclassFormatter(SourceFormatter):
    priority = 36

    def can_format(self, value) -> bool:
        return LazyDataclassFormatEligibility.accepts(value)

    def format(self, value: LazyDataclass, context: FormatContext) -> SourceFragment:
        plan = LazyDataclassSerializationPlan.from_instance(value, context)
        imports = {plan.import_pair}
        field_ctx = context.indented()
        lines = []

        for field in plan.fields:
            imports |= field.fragment.imports
            lines.append(f"{field.name}={field.fragment.code}")

        if not lines:
            return SourceFragment(f"{plan.class_name}()", frozenset(imports))

        inner = f",\n{field_ctx.indent_str}".join(lines)
        code = (
            f"{plan.class_name}(\n"
            f"{field_ctx.indent_str}{inner}\n"
            f"{context.indent_str})"
        )
        return SourceFragment(code, frozenset(imports))


class FunctionStepFormatter(SourceFormatter):
    priority = 80

    def can_format(self, value) -> bool:
        return isinstance(value, FunctionStep)

    def format(self, value: FunctionStep, context: FormatContext) -> SourceFragment:
        from openhcs.core.steps.abstract import AbstractStep

        signatures = [
            (name, param)
            for name, param in inspect.signature(
                FunctionStep.__init__
            ).parameters.items()
            if name != "self" and param.kind != inspect.Parameter.VAR_KEYWORD
        ]
        seen = {name for name, _param in signatures}
        signatures.extend(
            (name, param)
            for name, param in inspect.signature(
                AbstractStep.__init__
            ).parameters.items()
            if name != "self" and name not in seen
        )

        default_step = FunctionStep(func=lambda: None)
        step_values = vars(value)
        default_values = vars(default_step)
        field_ctx = context.indented()
        params = []
        imports = set()
        field_policy = FunctionStepCleanModeFieldPolicy()

        for name, param in signatures:
            if name in step_values:
                current_value = step_values[name]
            else:
                current_value = param.default

            if name in default_values:
                default_value = default_values[name]
            else:
                default_value = param.default

            if not field_policy.should_emit(
                name,
                current_value,
                default_value,
                context,
            ):
                continue

            frag = to_source(current_value, field_ctx)
            imports |= frag.imports
            params.append(f"{name}={frag.code}")

        import_pair = (FunctionStep.__module__, FunctionStep.__name__)
        class_name = NameMappingLookup.resolve(
            context, import_pair, FunctionStep.__name__
        )
        imports.add(import_pair)

        if not params:
            return SourceFragment(f"{class_name}()", frozenset(imports))

        inner = f",\n{field_ctx.indent_str}".join(params)
        code = f"{class_name}(\n{field_ctx.indent_str}{inner}\n{context.indent_str})"
        return SourceFragment(code, frozenset(imports))
