"""Registry contracts shared by CellProfiler runtime policy families."""

from __future__ import annotations
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import ClassVar, Generic, TypeVar
from metaclass_registry import AutoRegisterMeta
from openhcs.core.registry_strategies import (
    EnumKeyedStrategyMixin,
    StrategyLabelRegistryMixin,
)
from openhcs.interop.cellprofiler.runtime.payload_types import (
    CellProfilerKwargDict,
    CellProfilerRuntimeValue,
)
from openhcs.processing.backends.cellprofiler.library import canonical_module_name

MODULE_NAME_REGISTRY_KEY = "module_name"
EnumStrategyT = TypeVar("EnumStrategyT", bound=Enum)


class CellProfilerModulePolicyRegistryKey(str, Enum):
    """Reserved registry keys for CellProfiler module policy families."""

    DEFAULT = "__default__"


@dataclass(frozen=True, slots=True)
class CellProfilerModulePolicyRegistryDefaults:
    """Registry defaults shared by CellProfiler module-name policy roots."""

    registry_key_attr: str = "registry_key"
    module_name_attr: str = MODULE_NAME_REGISTRY_KEY
    fallback_registry_key_attr: str = "fallback_registry_key"

    def applies_to_root_bases(self, bases: tuple[type, ...]) -> bool:
        """Return whether a class declaration starts a new policy registry."""
        return not any((self.mro_declares_registry(base) for base in bases))

    def mro_declares_registry(self, cls: type) -> bool:
        """Return whether a class already belongs to an AutoRegisterMeta family."""
        return any(("__registry__" in vars(mro_type) for mro_type in cls.__mro__))

    def registry_key_for_class(self, name: str, cls: type) -> str | None:
        """Derive the declared registry key for one policy class."""
        del name
        registry_key = vars(cls).get(self.registry_key_attr)
        if registry_key is not None:
            return str(registry_key)
        module_name = cls.module_name
        if module_name is None:
            return f"{cls.__module__}.{cls.__qualname__}"
        return str(module_name)

    def apply_to(self, attrs: CellProfilerKwargDict) -> None:
        """Install AutoRegisterMeta attributes for one policy root."""
        attrs.setdefault("__registry_key__", self.registry_key_attr)
        attrs.setdefault("__skip_if_no_key__", True)
        attrs.setdefault("__key_extractor__", staticmethod(self.registry_key_for_class))
        attrs.setdefault(self.registry_key_attr, None)
        attrs.setdefault(self.module_name_attr, None)
        attrs.setdefault(
            self.fallback_registry_key_attr,
            CellProfilerModulePolicyRegistryKey.DEFAULT.value,
        )

    def clear_inherited_fallback_key(
        self, bases: tuple[type, ...], attrs: CellProfilerKwargDict
    ) -> None:
        """Keep fallback registry keys on the class that declares them."""
        if self.registry_key_attr in attrs:
            return
        inherited_fallback = any(
            (
                vars(base).get(self.registry_key_attr)
                == CellProfilerModulePolicyRegistryKey.DEFAULT.value
                for base in bases
            )
        )
        if inherited_fallback:
            attrs[self.registry_key_attr] = None


CELLPROFILER_MODULE_POLICY_REGISTRY_DEFAULTS = (
    CellProfilerModulePolicyRegistryDefaults()
)


@dataclass(frozen=True, slots=True)
class CellProfilerModulePolicyRegistryConfigContext:
    """Metaclass registry-config context for CellProfiler module policies."""

    raw_registry_config: CellProfilerRuntimeValue
    defaults: CellProfilerModulePolicyRegistryDefaults

    def apply_root_defaults(
        self, bases: tuple[type, ...], attrs: CellProfilerKwargDict
    ) -> None:
        """Install implicit root defaults when this declaration starts a registry."""
        if self.defaults.applies_to_root_bases(bases):
            self.defaults.apply_to(attrs)
            return
        self.defaults.clear_inherited_fallback_key(bases, attrs)


CELLPROFILER_MODULE_POLICY_IMPLICIT_REGISTRY_CONTEXT = (
    CellProfilerModulePolicyRegistryConfigContext(
        raw_registry_config=None, defaults=CELLPROFILER_MODULE_POLICY_REGISTRY_DEFAULTS
    )
)


@dataclass(frozen=True, slots=True)
class CellProfilerModulePolicyRegistryLookup:
    """Resolve a module policy type through primary and fallback registry keys."""

    registry: dict[str, type]
    module_name: str
    fallback_registry_key: str | None

    def candidate_keys(self) -> tuple[str, ...]:
        primary_key = canonical_module_name(self.module_name)
        fallback_key = self.fallback_registry_key
        if fallback_key is None:
            return (primary_key,)
        return (primary_key, fallback_key)

    def policy_type_or_none(self) -> type | None:
        for registry_key in self.candidate_keys():
            if registry_key in self.registry:
                return self.registry[registry_key]
        return None


class CellProfilerModulePolicyAutoRegisterMeta(AutoRegisterMeta):
    """AutoRegisterMeta variant for CellProfiler module-name policy families."""

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        attrs: CellProfilerKwargDict,
        registry_config: CellProfilerModulePolicyRegistryConfigContext = CELLPROFILER_MODULE_POLICY_IMPLICIT_REGISTRY_CONTEXT,
    ):
        registry_config.apply_root_defaults(bases, attrs)
        return super().__new__(
            mcs, name, bases, attrs, registry_config.raw_registry_config
        )

    @lru_cache(maxsize=None)
    def for_module(cls, module_name: str) -> CellProfilerRuntimeValue:
        """Return the policy registered for a module, or the root's fallback."""
        policy_type = CellProfilerModulePolicyRegistryLookup(
            cls.__registry__, module_name, cls.fallback_registry_key
        ).policy_type_or_none()
        if policy_type is None:
            return None
        return policy_type()


class CellProfilerModulePolicyLookupMixin:
    """Shared module-name lookup for explicit CellProfiler policy registries."""

    stable_key_axis: ClassVar[str] = "registry_key"
    declaration_policy_bases: ClassVar[tuple[type, ...]] = ()
    registry_key: ClassVar[str | None]
    module_name: ClassVar[str | None]
    fallback_registry_key: ClassVar[str | None]

    @classmethod
    @lru_cache(maxsize=None)
    def for_module(cls, module_name: str) -> CellProfilerRuntimeValue:
        """Return the policy registered for a module, or the root's fallback."""
        from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule

        module_type = CellProfilerModule.for_module(module_name)
        if module_type is not None:
            declaration_policy_type = cls.declaration_policy_type_or_none(module_type)
            if declaration_policy_type is not None:
                return cls.declaration_policy_instance(
                    module_type, declaration_policy_type
                )
        policy_type = CellProfilerModulePolicyRegistryLookup(
            cls.__registry__, module_name, cls.fallback_registry_key
        ).policy_type_or_none()
        if policy_type is None:
            return None
        return policy_type()

    @classmethod
    def declaration_policy_type_or_none(cls, module_type: type) -> type | None:
        """Return the most-derived policy inherited by a module declaration."""
        from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule

        matching_policy_types = tuple(
            (
                policy_type
                for policy_type in module_type.__mro__[1:]
                if policy_type is not cls
                and isinstance(policy_type, type)
                and (not issubclass(policy_type, CellProfilerModule))
                and any(
                    (
                        issubclass(policy_type, policy_base)
                        for policy_base in cls.declaration_policy_bases
                    )
                )
            )
        )
        owning_policy_types = tuple(
            (
                candidate_type
                for candidate_type in matching_policy_types
                if not any(
                    (
                        other_type is not candidate_type
                        and issubclass(other_type, candidate_type)
                        for other_type in matching_policy_types
                    )
                )
            )
        )
        if len(owning_policy_types) > 1:
            names = tuple(
                (policy_type.__qualname__ for policy_type in owning_policy_types)
            )
            raise ValueError(
                f"{module_type.__name__} inherits multiple {cls.__name__} policies: {names!r}."
            )
        if owning_policy_types:
            return owning_policy_types[0]
        return None

    @classmethod
    def declaration_policy_instance(
        cls, module_type: type, policy_type: type
    ) -> object:
        """Return a policy view carrying module declaration class attributes."""
        return cls.declaration_policy_instance_type(module_type, policy_type)()

    @classmethod
    @lru_cache(maxsize=None)
    def declaration_policy_instance_type(
        cls, module_type: type, policy_type: type
    ) -> type:
        """Create a policy type whose methods come from the owning policy base."""
        attrs = {
            "__module__": module_type.__module__,
            "__qualname__": f"{module_type.__qualname__}.{policy_type.__name__}View",
        }
        for name, value in vars(module_type).items():
            if name.startswith("__"):
                continue
            if name in vars(policy_type):
                continue
            if callable(value) or isinstance(
                value, (staticmethod, classmethod, property)
            ):
                continue
            attrs[name] = value
        return type(
            f"{module_type.__name__}{policy_type.__name__}View", (policy_type,), attrs
        )


class EnumStrategyLabelRegistryMixin(
    EnumKeyedStrategyMixin[EnumStrategyT],
    StrategyLabelRegistryMixin,
    ABC,
    Generic[EnumStrategyT],
):
    """Named MRO bundle for enum-keyed strategy registries."""

    stable_key_axis: ClassVar[str] = "strategy_label"


class NoSourceImageNameMixin:
    """Policy mixin for objects that intentionally do not qualify by source image."""

    def source_image_name(self, request: CellProfilerRuntimeValue) -> str | None:
        del request
        return None
