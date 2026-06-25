from __future__ import annotations

import inspect
from dataclasses import dataclass, fields

from openhcs.core.steps.function_step import FunctionStep
from pyqt_reactive.pattern_metadata import SCOPE_TOKEN_KEY
from openhcs.config_framework.lazy_factory import LazyDataclass

from pycodify import FormatContext, SourceFormatter, SourceFragment, to_source


SLICE_BY_SLICE_ATTR = "slice_by_slice"
PROCESSING_CONTRACT_ATTR = "__processing_contract__"


def _module_contract_imports(contract) -> set[tuple[str, str]]:
    from openhcs.core.artifact_materialization_policy import (
        NO_ARTIFACT_MATERIALIZATION,
    )

    imports: set[tuple[str, str]] = set()
    specs = (
        *contract.inputs,
        *contract.runtime_artifact_inputs,
        *contract.outputs,
        *contract.declared_outputs,
    )
    if any(spec.materialization is NO_ARTIFACT_MATERIALIZATION for spec in specs):
        imports.add(
            (
                "openhcs.core.artifact_materialization_policy",
                "NO_ARTIFACT_MATERIALIZATION",
            )
        )
    return imports


@dataclass(frozen=True)
class CallableDecoratorMetadata:
    attribute_names: frozenset[str]

    @classmethod
    def from_callable(cls, func) -> "CallableDecoratorMetadata":
        if inspect.isfunction(func):
            return cls(attribute_names=frozenset(func.__dict__))
        return cls(attribute_names=frozenset())

    def contains(self, attribute_name: str) -> bool:
        return attribute_name in self.attribute_names


@dataclass(frozen=True)
class CallableExportIdentity:
    module: str | None
    name: str | None
    has_slice_by_slice: bool
    has_processing_contract: bool

    @classmethod
    def from_callable(cls, func) -> "CallableExportIdentity":
        if not (inspect.isfunction(func) or inspect.isbuiltin(func)):
            return cls(
                module=None,
                name=None,
                has_slice_by_slice=False,
                has_processing_contract=False,
            )

        metadata = CallableDecoratorMetadata.from_callable(func)

        return cls(
            module=func.__module__,
            name=func.__name__,
            has_slice_by_slice=metadata.contains(SLICE_BY_SLICE_ATTR),
            has_processing_contract=metadata.contains(PROCESSING_CONTRACT_ATTR),
        )

    @property
    def is_importable(self) -> bool:
        return bool(self.module and self.name)

    @property
    def is_external_registered(self) -> bool:
        if self.module is None:
            return False
        return (
            self.has_slice_by_slice
            and not self.has_processing_contract
            and not self.module.startswith("openhcs.")
        )

    @property
    def import_module(self) -> str:
        if self.module is None:
            raise ValueError("Callable identity has no importable module.")
        if self.is_external_registered:
            return f"openhcs.{self.module}"
        return self.module

    @property
    def import_name(self) -> str:
        if self.name is None:
            raise ValueError("Callable identity has no importable name.")
        return self.name


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
        return callable(value) and not isinstance(value, type)

    def format(self, value, context: FormatContext) -> SourceFragment:
        if inspect.ismethod(value):
            return SourceFragment(repr(value), frozenset())

        identity = CallableExportIdentity.from_callable(value)
        if not identity.is_importable:
            return SourceFragment(repr(value), frozenset())

        import_pair = (identity.import_module, identity.import_name)
        mapped = NameMappingLookup.resolve(context, import_pair, identity.import_name)
        return SourceFragment(mapped, frozenset([import_pair]))


class CellProfilerRuntimeCallableFormatter(SourceFormatter):
    priority = 90

    def can_format(self, value) -> bool:
        from openhcs.interop.cellprofiler.runtime.module_execution import (
            CellProfilerRuntimeCallable,
        )

        return isinstance(value, CellProfilerRuntimeCallable)

    def format(self, value, context: FormatContext) -> SourceFragment:
        raw_func_frag = to_source(value.raw_func, context)
        contract_frag = to_source(value.contract, context.indented())
        import_pair = (
            "openhcs.interop.cellprofiler.runtime.module_execution",
            "cellprofiler_module_callable",
        )
        factory_name = NameMappingLookup.resolve(
            context, import_pair, "cellprofiler_module_callable"
        )
        imports = set(raw_func_frag.imports | contract_frag.imports)
        imports |= _module_contract_imports(value.contract)
        imports.add(import_pair)
        args = [
            raw_func_frag.code,
            contract_frag.code,
        ]
        if value.declared_processing_contract is not None:
            args.append(
                f"declared_processing_contract={value.declared_processing_contract!r}"
            )
        if value.processing_contract is not None:
            processing_contract_frag = to_source(value.processing_contract, context)
            imports |= processing_contract_frag.imports
            args.append(f"processing_contract={processing_contract_frag.code}")

        field_ctx = context.indented()
        inner = f",\n{field_ctx.indent_str}".join(args)
        return SourceFragment(
            f"{factory_name}(\n{field_ctx.indent_str}{inner}\n{context.indent_str})",
            frozenset(imports),
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

        inner = f",\n{item_ctx.indent_str}".join(args)
        return SourceFragment(
            f"{class_name}(\n{item_ctx.indent_str}{inner}\n{context.indent_str})",
            frozenset(imports),
        )


def _is_pattern_tuple(value) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and callable(value[0])
        and isinstance(value[1], dict)
    )


def _is_pattern_item(value) -> bool:
    return callable(value) or _is_pattern_tuple(value)


def _strip_internal_pattern_metadata(args):
    """Remove UI-only metadata keys from function-pattern kwargs."""
    if not isinstance(args, dict):
        return {}
    return {k: v for k, v in args.items() if k != SCOPE_TOKEN_KEY}


class FunctionPatternTupleFormatter(SourceFormatter):
    priority = 85

    def can_format(self, value) -> bool:
        return _is_pattern_tuple(value)

    def format(self, value, context: FormatContext) -> SourceFragment:
        func, args = value
        args = _strip_internal_pattern_metadata(args)

        if not args and context.clean_mode:
            return to_source(func, context)

        try:
            defaults = {
                k: v.default
                for k, v in inspect.signature(func).parameters.items()
                if v.default is not inspect.Parameter.empty
            }
        except (ValueError, TypeError):
            defaults = {}

        if context.clean_mode:
            final_args = {
                k: v for k, v in args.items() if k not in defaults or v != defaults[k]
            }
        else:
            final_args = {**defaults, **args}

        if not final_args and context.clean_mode:
            return to_source(func, context)

        func_frag = to_source(func, context)
        args_frag = to_source(final_args, context.indented())
        code = f"({func_frag.code}, {args_frag.code})"
        imports = func_frag.imports | args_frag.imports
        return SourceFragment(code, imports)


class FunctionPatternListFormatter(SourceFormatter):
    priority = 84

    def can_format(self, value) -> bool:
        return isinstance(value, list) and value and all(_is_pattern_item(item) for item in value)

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
            if (
                context.clean_mode
                and field.name not in explicit_field_names
                and current_value is None
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
        current_value,
        default_value,
        context: FormatContext,
    ) -> bool:
        if not context.clean_mode:
            return True

        if isinstance(current_value, LazyDataclass):
            return LazyDataclassFieldEmissionState.from_instance(
                current_value
            ).requires_serialization

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
            for name, param in inspect.signature(FunctionStep.__init__).parameters.items()
            if name != "self" and param.kind != inspect.Parameter.VAR_KEYWORD
        ]
        seen = {name for name, _param in signatures}
        signatures.extend(
            (name, param)
            for name, param in inspect.signature(AbstractStep.__init__).parameters.items()
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

            if not field_policy.should_emit(current_value, default_value, context):
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
