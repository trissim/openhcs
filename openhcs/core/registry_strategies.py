"""Reusable helpers for AutoRegisterMeta-backed strategy families."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field
from enum import Enum, EnumMeta
from functools import lru_cache
from typing import Any, ClassVar, Generic, TypeVar, cast

from metaclass_registry import AutoRegisterMeta


_EnumT = TypeVar("_EnumT", bound=Enum)
_ContextT = TypeVar("_ContextT")
_StrategyT = TypeVar("_StrategyT", bound="EnumKeyedStrategyMixin[Any]")
_TypeStrategyT = TypeVar("_TypeStrategyT", bound="NominalTypeKeyedStrategyMixin")
_ContextStrategyT = TypeVar(
    "_ContextStrategyT",
    bound="MostDerivedContextStrategyMixin[Any]",
)


class RegisteredEnumMeta(AutoRegisterMeta, EnumMeta):
    """Metaclass for enum families that also need AutoRegisterMeta membership."""


class EnumKeyedStrategyMixin(Generic[_EnumT]):
    """Mixin for AutoRegisterMeta ABC families keyed by enum member values.

    Concrete strategies declare the enum member they implement. The mixin
    derives the JSON-safe registry key before AutoRegisterMeta registers the
    concrete class, avoiding repeated ``label = enum.value`` boilerplate.
    """

    __enum_member_attr__: ClassVar[str] = "strategy_key"
    __enum_label_attr__: ClassVar[str] = "strategy_label"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        member = getattr(cls, cls.__enum_member_attr__, None)
        if not isinstance(member, Enum):
            return
        label_attr = cls._enum_label_attr()
        if getattr(cls, label_attr, None) is None:
            setattr(cls, label_attr, member.value)

    @classmethod
    def _enum_label_attr(cls) -> str:
        """Return the class attribute AutoRegisterMeta should read as a key."""
        label_attr = cls.__enum_label_attr__
        registry_key = getattr(cls, "__registry_key__", None)
        if (
            label_attr == EnumKeyedStrategyMixin.__enum_label_attr__
            and isinstance(registry_key, str)
        ):
            return registry_key
        return label_attr

    @classmethod
    def for_enum_member(cls: type[_StrategyT], member: _EnumT) -> _StrategyT:
        """Instantiate the registered strategy for an enum member."""
        strategy_type = cls.strategy_type_for_enum_member(member)
        return cast(_StrategyT, strategy_type())

    @classmethod
    @lru_cache(maxsize=None)
    def strategy_type_for_enum_member(
        cls: type[_StrategyT],
        member: _EnumT,
    ) -> type[_StrategyT]:
        """Return the registered strategy class for an enum member."""
        return cast(type[_StrategyT], cls.__registry__[member.value])

    @classmethod
    def registered_strategy_types(
        cls: type[_StrategyT],
    ) -> tuple[type[_StrategyT], ...]:
        """Return one registered strategy class per declared enum member.

        Registry cache backends may preserve JSON-safe string forms of non-string
        enum keys alongside the original key. The enum member declared on the
        strategy is the semantic identity, so collapse aliases there.
        """
        strategy_types: dict[tuple[type[Enum], _EnumT] | int, type[_StrategyT]] = {}
        for strategy_type in cls.__registry__.values():
            member = getattr(strategy_type, cls.__enum_member_attr__, None)
            key: tuple[type[Enum], _EnumT] | int
            if isinstance(member, Enum):
                key = (type(member), member)
            else:
                key = id(strategy_type)
            strategy_types.setdefault(key, cast(type[_StrategyT], strategy_type))
        return tuple(strategy_types.values())


class NominalTypeKeyedStrategyMixin:
    """Mixin for strategies selected by nominal runtime value types.

    Concrete strategies declare their owning Python type in ``value_type``.
    The mixin derives a JSON-safe registry key from that nominal type, so
    ``AutoRegisterMeta`` never stores raw type objects as registry/cache keys.
    """

    value_type: ClassVar[type[object] | tuple[type[object], ...] | None] = None
    value_type_label: ClassVar[str | None] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        member = cls.value_type
        if not _is_nominal_type_member(member):
            return
        if cls.__dict__.get("value_type_label") is None:
            cls.value_type_label = _nominal_type_key(member)

    @classmethod
    def registered_strategy_types(
        cls: type[_TypeStrategyT],
    ) -> tuple[type[_TypeStrategyT], ...]:
        """Return registered concrete strategy classes."""
        return tuple(cast(type[_TypeStrategyT], item) for item in cls.__registry__.values())

    @classmethod
    def for_nominal_value(
        cls: type[_TypeStrategyT],
        value: object,
    ) -> _TypeStrategyT | None:
        """Instantiate the most specific registered strategy owning ``value``."""
        value_mro = type(value).mro()
        best_match: tuple[int, type[_TypeStrategyT]] | None = None
        for strategy_type in cls.registered_strategy_types():
            member = strategy_type.value_type
            if _is_nominal_type_member(member) and isinstance(value, member):
                distance = cls.nominal_type_distance(value_mro, member)
                if best_match is None or distance < best_match[0]:
                    best_match = (distance, strategy_type)
        return None if best_match is None else best_match[1]()

    @staticmethod
    def nominal_type_distance(
        value_mro: list[type[object]],
        member: type[object] | tuple[type[object], ...],
    ) -> int:
        """Return the MRO distance from a runtime value to a registered member."""
        if isinstance(member, tuple):
            return min(
                NominalTypeKeyedStrategyMixin.nominal_type_distance(value_mro, item)
                for item in member
            )
        try:
            return value_mro.index(member)
        except ValueError:
            return len(value_mro)


def _is_nominal_type_member(value: object) -> bool:
    if isinstance(value, type):
        return True
    return isinstance(value, tuple) and bool(value) and all(
        isinstance(item, type) for item in value
    )


def _nominal_type_key(value: type[object] | tuple[type[object], ...]) -> str:
    if isinstance(value, tuple):
        return "|".join(_nominal_type_key(item) for item in value)
    return f"{value.__module__}.{value.__qualname__}"


class MostDerivedContextStrategyMixin(Generic[_ContextT], ABC):
    """Mixin for registry families selected by context and strategy inheritance.

    Concrete strategies implement ``matches`` and express precedence through
    normal Python inheritance. Selection returns the single most-derived
    matching implementation, so callers do not need local if/elif chains,
    priority numbers, or repeated registry scans.
    """

    strategy_key_attr: ClassVar[str] = "strategy_key"

    @classmethod
    def registered_strategy_types(
        cls: type[_ContextStrategyT],
    ) -> tuple[type[_ContextStrategyT], ...]:
        """Return registered concrete strategy classes."""
        return tuple(
            cast(type[_ContextStrategyT], item) for item in cls.__registry__.values()
        )

    @classmethod
    def for_context(
        cls: type[_ContextStrategyT],
        context: _ContextT,
        *,
        required: bool = True,
        error_subject: str | None = None,
    ) -> _ContextStrategyT | None:
        """Instantiate the single most-derived registered strategy for context."""
        owning_strategy_types = cls.owning_strategy_types(context)
        if not owning_strategy_types:
            if not required:
                return None
            raise ValueError(
                f"{cls._error_subject(error_subject)} requires a matching strategy."
            )
        if len(owning_strategy_types) != 1:
            names = tuple(
                cls._strategy_name(strategy_type)
                for strategy_type in owning_strategy_types
            )
            raise ValueError(
                f"{cls._error_subject(error_subject)} requires exactly one "
                f"most-derived strategy, got {names!r}."
            )
        return owning_strategy_types[0]()

    @classmethod
    def owning_strategy_types(
        cls: type[_ContextStrategyT],
        context: _ContextT,
    ) -> tuple[type[_ContextStrategyT], ...]:
        """Return most-derived registered strategies matching ``context``."""
        matching_strategy_types = tuple(
            strategy_type
            for strategy_type in cls.registered_strategy_types()
            if strategy_type().matches(context)
        )
        return tuple(
            candidate_type
            for candidate_type in matching_strategy_types
            if not any(
                other_type is not candidate_type
                and issubclass(other_type, candidate_type)
                for other_type in matching_strategy_types
            )
        )

    @classmethod
    def _error_subject(cls, explicit_subject: str | None) -> str:
        return explicit_subject or cls.__name__

    @classmethod
    def _strategy_name(cls, strategy_type: type[_ContextStrategyT]) -> object:
        return getattr(strategy_type, cls.strategy_key_attr, strategy_type.__name__)

    @abstractmethod
    def matches(self, context: _ContextT) -> bool:
        """Return whether this registered strategy owns ``context``."""


class RegisteredLeafClassSpec(ABC):
    """Nominal declaration for generated AutoRegisterMeta leaf classes."""

    class_name: str
    base_type: type[object]

    @abstractmethod
    def class_attributes(self) -> Mapping[str, object]:
        """Return semantic class attributes for the generated leaf."""

    def declare_in(self, namespace: MutableMapping[str, object]) -> type[object]:
        """Materialize the generated leaf in ``namespace`` and return it."""
        class_name = self.class_name
        declared_type = type(
            class_name,
            (self.base_type,),
            {
                "__module__": namespace.get("__name__", self.base_type.__module__),
                **dict(self.class_attributes()),
            },
        )
        namespace[class_name] = declared_type
        return declared_type


@dataclass(frozen=True, slots=True)
class GeneratedLeafClassSpec(RegisteredLeafClassSpec):
    """Concrete generated-leaf declaration with explicit class and base identity."""

    class_name: str
    base_type: type[object]
    attributes: Mapping[str, object] = field(default_factory=dict, kw_only=True)

    def class_attributes(self) -> Mapping[str, object]:
        """Return the declared class attributes for this generated leaf."""
        return self.attributes


@dataclass(frozen=True, slots=True)
class GeneratedEnumClassSpec:
    """Nominal declaration for generated enum classes."""

    class_name: str
    base_type: type[Enum]
    members: Mapping[str, object]

    def declare_in(self, namespace: MutableMapping[str, object]) -> type[Enum]:
        """Materialize the generated enum in ``namespace`` and return it."""
        enum_meta = type(self.base_type)
        enum_namespace = enum_meta.__prepare__(self.class_name, (self.base_type,))
        enum_namespace["__module__"] = str(
            namespace.get("__name__", self.base_type.__module__)
        )
        for name, value in self.members.items():
            enum_namespace[name] = value
        enum_type = enum_meta(self.class_name, (self.base_type,), enum_namespace)
        namespace[self.class_name] = enum_type
        return enum_type
