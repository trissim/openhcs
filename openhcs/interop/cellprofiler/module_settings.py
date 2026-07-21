"""Nominal settings authorities for CellProfiler modules."""

from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import UnionType
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)
from openhcs.interop.cellprofiler.settings_binder import (
    SettingToKeywordBinding,
    coerce_cellprofiler_enum,
)
from openhcs.interop.cellprofiler_setting_normalization import (
    normalize_cellprofiler_setting_name,
)
from openhcs.interop.cellprofiler.setting_names import (
    SettingNameFamily,
    setting_name_matches,
    setting_names,
    setting_values,
    split_symbol_names,
)

if TYPE_CHECKING:
    from openhcs.interop.cellprofiler.parser import ModuleBlock
    from openhcs.interop.cellprofiler.settings_binder import SettingsBinder


@dataclass(frozen=True, slots=True)
class ModuleSettingRowRecord:
    """Concrete CellProfiler setting row identity and value."""

    module_name: str
    module_num: int
    setting_name: str
    normalized_setting_name: str
    value: Any


@dataclass(frozen=True, slots=True)
class ModuleSettingCoverageRecord(ModuleSettingRowRecord):
    """Coverage status for one concrete CellProfiler setting row."""

    status: "ModuleSettingCoverageStatus"


class ModuleSettingCoverageStatus(str, Enum):
    """How one CellProfiler setting row was accounted for by import binding."""

    BOUND = "bound"
    IGNORED = "ignored"
    UNMAPPED = "unmapped"

    @property
    def is_covered(self) -> bool:
        """Return whether the declaration accounted for this setting row."""
        return self is not ModuleSettingCoverageStatus.UNMAPPED

    @classmethod
    def for_setting(
        cls,
        normalized_name: str,
        *,
        unmapped_kwargs: Mapping[str, Any],
        ignored_setting_names: frozenset[str],
    ) -> "ModuleSettingCoverageStatus":
        """Return the coverage status owned by this status enum."""
        if normalized_name not in unmapped_kwargs:
            return cls.BOUND
        if normalized_name in ignored_setting_names:
            return cls.IGNORED
        return cls.UNMAPPED


@dataclass(frozen=True, slots=True)
class BoundModuleSettings:
    """Typed module-setting translation result."""

    kwargs: Mapping[str, Any]
    unmapped_kwargs: Mapping[str, Any] = field(default_factory=dict)
    setting_coverage: tuple[ModuleSettingCoverageRecord, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "kwargs", dict(self.kwargs))
        object.__setattr__(self, "unmapped_kwargs", dict(self.unmapped_kwargs))
        object.__setattr__(self, "setting_coverage", tuple(self.setting_coverage))

    def with_kwargs(self, kwargs: Mapping[str, Any]) -> "BoundModuleSettings":
        """Return this binding with additional generated function kwargs."""
        return BoundModuleSettings(
            {**self.kwargs, **kwargs},
            self.unmapped_kwargs,
            self.setting_coverage,
        )

    def with_consumed_settings(
        self,
        *settings: str | "SettingNameFamily",
    ) -> "BoundModuleSettings":
        """Remove rows explicitly consumed by an owning compound parser."""

        consumed_names = {
            normalize_cellprofiler_setting_name(concrete_name)
            for setting in settings
            for concrete_name in setting_names(setting)
        }
        return BoundModuleSettings(
            self.kwargs,
            {
                name: value
                for name, value in self.unmapped_kwargs.items()
                if name not in consumed_names
            },
            self.setting_coverage,
        )

    def with_replaced_kwargs(self, kwargs: Mapping[str, Any]) -> "BoundModuleSettings":
        """Return this binding with the function kwargs replaced."""
        return BoundModuleSettings(
            kwargs,
            self.unmapped_kwargs,
            self.setting_coverage,
        )


class UnmappedModuleSettingsError(ValueError):
    """Raised when enabled module settings are not mapped or explicitly ignored."""

    def __init__(self, settings: tuple[ModuleSettingRowRecord, ...]) -> None:
        self.settings = settings
        rendered = "; ".join(
            (
                f"{setting.module_name}({setting.module_num}).{setting.setting_name}={setting.value!r}"
                for setting in settings
            )
        )
        super().__init__(
            f"Enabled CellProfiler modules have unmapped settings. Add a module settings binding hook or an explicit typed ignore: {rendered}"
        )


def _enum_type_from_annotation(annotation: Any) -> type[Enum] | None:
    """Return the callable-owned Enum type declared by an annotation."""
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return annotation
    origin = get_origin(annotation)
    if origin in (Union, UnionType, tuple):
        for arg in get_args(annotation):
            enum_type = _enum_type_from_annotation(arg)
            if enum_type is not None:
                return enum_type
    return None


def _coerce_callable_enum_kwarg(value: Any, enum_type: type[Enum]) -> Any:
    """Coerce one bound kwarg value to the callable-owned Enum type."""
    if value is None:
        return None
    if isinstance(value, tuple):
        return tuple(_coerce_callable_enum_kwarg(item, enum_type) for item in value)
    if isinstance(value, enum_type):
        return value
    return coerce_cellprofiler_enum(enum_type, value)


class CellProfilerModuleSettings:
    setting_bindings: ClassVar[tuple["SettingToKeywordBinding", ...]] = ()
    ignored_settings: ClassVar[tuple[str | "SettingNameFamily", ...]] = ()

    @classmethod
    def declared_setting_bindings(cls) -> tuple[SettingToKeywordBinding, ...]:
        """Return explicit setting bindings composed through the module MRO."""

        bindings: list[SettingToKeywordBinding] = []
        for owner_type in reversed(cls.__mro__):
            for binding in owner_type.__dict__.get("setting_bindings", ()):
                if not isinstance(binding, SettingToKeywordBinding):
                    raise TypeError(
                        f"{owner_type.__name__}.setting_bindings must contain "
                        "SettingToKeywordBinding values."
                    )
                bindings.append(binding)
        return tuple(bindings)

    @classmethod
    def normalize_setting_name(cls, setting_name: str) -> str:
        """Return the canonical lookup key for CellProfiler setting labels."""
        del cls
        return normalize_cellprofiler_setting_name(setting_name)

    @classmethod
    def _bind_declared_settings(
        cls,
        module: "ModuleBlock",
        *,
        binder: "SettingsBinder",
    ) -> "BoundModuleSettings":
        """Bind rows using setting declarations inherited by this module class."""
        setting_bindings = cls.declared_setting_bindings()
        ignored_settings = tuple(cls.ignored_settings_for(module))
        bound_details = binder.bind_with_details(module)
        kwargs = binder.bind_declared(module, setting_bindings)
        for binding in setting_bindings:
            if not binding.repeated:
                continue
            values = tuple(
                name
                for value in setting_values(module, binding.setting_name)
                for name in split_symbol_names(value)
            )
            if values:
                kwargs[binding.require_parameter_name()] = (
                    values[0] if len(values) == 1 else values
                )
        mapped_settings = {
            normalize_cellprofiler_setting_name(setting_name)
            for binding in setting_bindings
            for setting_name in setting_names(binding.setting_name)
        }
        mapped_settings.update(
            (
                normalize_cellprofiler_setting_name(concrete_setting_name)
                for setting_name in ignored_settings
                for concrete_setting_name in setting_names(setting_name)
            )
        )
        unmapped_kwargs = {
            detail.name: detail.original_value
            for detail in bound_details
            if detail.name not in mapped_settings
        }
        return BoundModuleSettings(kwargs, unmapped_kwargs)

    @classmethod
    def _finalize_bound_settings(
        cls,
        module: "ModuleBlock",
        *,
        binder: "SettingsBinder",
        bound: "BoundModuleSettings",
    ) -> "BoundModuleSettings":
        """Validate and annotate a binding result produced by this declaration."""
        del binder
        bound = bound.with_replaced_kwargs(
            cls._coerce_kwargs_to_callable_signature(bound.kwargs)
        )
        ignored_setting_names = frozenset(
            (
                normalize_cellprofiler_setting_name(concrete_name)
                for setting_name in cls.ignored_settings_for(module)
                for concrete_name in setting_names(setting_name)
            )
        )
        unmapped_kwargs = {
            setting_name: value
            for setting_name, value in bound.unmapped_kwargs.items()
            if setting_name not in ignored_setting_names
        }
        setting_coverage: list[ModuleSettingCoverageRecord] = []
        for setting in module.iter_settings():
            normalized_name = normalize_cellprofiler_setting_name(setting.name)
            setting_coverage.append(
                ModuleSettingCoverageRecord(
                    module_name=module.name,
                    module_num=module.module_num,
                    setting_name=setting.name,
                    normalized_setting_name=normalized_name,
                    value=setting.value,
                    status=ModuleSettingCoverageStatus.for_setting(
                        normalized_name,
                        unmapped_kwargs=bound.unmapped_kwargs,
                        ignored_setting_names=ignored_setting_names,
                    ),
                )
            )
        if unmapped_kwargs:
            raise UnmappedModuleSettingsError(
                tuple(
                    (
                        ModuleSettingRowRecord(
                            module_name=module.name,
                            module_num=module.module_num,
                            setting_name=setting_name,
                            normalized_setting_name=setting_name,
                            value=value,
                        )
                        for setting_name, value in sorted(unmapped_kwargs.items())
                    )
                )
            )
        return BoundModuleSettings(
            bound.kwargs,
            unmapped_kwargs,
            tuple(setting_coverage),
        )

    @classmethod
    def _coerce_kwargs_to_callable_signature(
        cls,
        kwargs: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Return kwargs using enum classes declared by the runtime callable."""
        from inspect import Parameter, signature

        if not kwargs:
            return kwargs
        absorbed_function = cls.require_callable(str(cls.function_name))
        annotations = get_type_hints(absorbed_function)
        parameters = signature(absorbed_function).parameters
        coerced: dict[str, Any] = {}
        for parameter_name, value in kwargs.items():
            parameter = parameters.get(parameter_name)
            if parameter is None:
                coerced[parameter_name] = value
                continue
            annotation = annotations.get(parameter_name)
            enum_type = (
                _enum_type_from_annotation(annotation)
                if annotation is not None
                else None
            )
            if (
                enum_type is None
                and parameter.default is not Parameter.empty
                and isinstance(parameter.default, Enum)
            ):
                enum_type = type(parameter.default)
            if enum_type is None:
                coerced[parameter_name] = value
                continue
            coerced[parameter_name] = _coerce_callable_enum_kwarg(value, enum_type)
        return coerced

    @classmethod
    def bind_settings(
        cls,
        module: "ModuleBlock",
        *,
        binder: "SettingsBinder",
    ) -> "BoundModuleSettings":
        """Bind parsed module settings through this module declaration."""
        bound = cls._bind_declared_settings(module, binder=binder)
        return cls._finalize_bound_settings(
            module,
            binder=binder,
            bound=cls.postprocess_bound_settings(module, bound),
        )

    @classmethod
    def postprocess_bound_settings(
        cls, module: "ModuleBlock", bound: "BoundModuleSettings"
    ) -> "BoundModuleSettings":
        """Apply module-local binding semantics after declared settings bind."""
        del module
        return bound

    @classmethod
    def ignored_settings_for(
        cls, module: "ModuleBlock"
    ) -> tuple[str | "SettingNameFamily", ...]:
        """Return settings consumed outside direct runtime kwargs."""
        del module
        return cls.ignored_settings

    @classmethod
    def setting_value(
        cls,
        module: "ModuleBlock",
        setting_name: str | "SettingNameFamily",
        *,
        include_blank: bool = False,
    ) -> str | None:
        """Return a module setting value through the module declaration boundary."""
        if not include_blank:
            values = setting_values(module, setting_name)
            return values[-1] if values else None
        for setting in module.iter_settings():
            if setting_name_matches(setting.name, setting_name):
                return setting.value.strip()
        for candidate_name, value in module.settings.items():
            if setting_name_matches(candidate_name, setting_name):
                return value.strip()
        return None
