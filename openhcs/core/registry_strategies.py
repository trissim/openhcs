"""Reusable helpers for AutoRegisterMeta-backed strategy families."""

from __future__ import annotations

from enum import Enum
from typing import Any, ClassVar, Generic, TypeVar, cast


_EnumT = TypeVar("_EnumT", bound=Enum)
_StrategyT = TypeVar("_StrategyT", bound="EnumKeyedStrategyMixin[Any]")
_TypeStrategyT = TypeVar("_TypeStrategyT", bound="NominalTypeKeyedStrategyMixin")


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
        if getattr(cls, cls.__enum_label_attr__, None) is None:
            setattr(cls, cls.__enum_label_attr__, member.value)

    @classmethod
    def for_enum_member(cls: type[_StrategyT], member: _EnumT) -> _StrategyT:
        """Instantiate the registered strategy for an enum member."""
        strategy_type = cls.__registry__[member.value]
        return cast(_StrategyT, strategy_type())

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
        """Instantiate the first registered strategy owning ``value``."""
        for strategy_type in cls.registered_strategy_types():
            member = strategy_type.value_type
            if _is_nominal_type_member(member) and isinstance(value, member):
                return strategy_type()
        return None


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
