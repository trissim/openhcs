from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from openhcs.core.aligned_image_payload import (
    AlignedImageStackKwargResolutionStrategy,
    AlignedImageStackKwargResolver,
    ImagePayloadAlignedKwargResolutionStrategy,
    NestedAlignedImageStackKwargResolutionStrategy,
    ObjectLabelAlignedKwargResolutionStrategy,
    PassThroughAlignedKwargResolutionStrategy,
    RuntimeSliceAlignedValueKwargResolutionStrategy,
    RuntimeSliceProjectableAlignedKwargResolutionStrategy,
    TupleAlignedKwargResolutionStrategy,
)


@pytest.fixture
def isolated_strategy_registry() -> Iterator[
    dict[str, type[AlignedImageStackKwargResolutionStrategy]]
]:
    registry = AlignedImageStackKwargResolutionStrategy.__registry__
    snapshot = registry.copy()
    AlignedImageStackKwargResolutionStrategy.registered_strategy_types.cache_clear()
    AlignedImageStackKwargResolutionStrategy.strategy_types_for_nominal_type.cache_clear()
    try:
        yield registry
    finally:
        registry.clear()
        registry.update(snapshot)
        AlignedImageStackKwargResolutionStrategy.registered_strategy_types.cache_clear()
        AlignedImageStackKwargResolutionStrategy.strategy_types_for_nominal_type.cache_clear()


def test_existing_aligned_kwarg_strategies_register_automatically() -> None:
    expected = {
        TupleAlignedKwargResolutionStrategy,
        ImagePayloadAlignedKwargResolutionStrategy,
        ObjectLabelAlignedKwargResolutionStrategy,
        RuntimeSliceAlignedValueKwargResolutionStrategy,
        RuntimeSliceProjectableAlignedKwargResolutionStrategy,
        PassThroughAlignedKwargResolutionStrategy,
        NestedAlignedImageStackKwargResolutionStrategy,
    }

    assert type(AlignedImageStackKwargResolutionStrategy.__registry__) is dict
    assert set(AlignedImageStackKwargResolutionStrategy.__registry__.values()) == expected
    assert len(AlignedImageStackKwargResolutionStrategy.__registry__) == len(expected)


def test_dynamic_aligned_kwarg_strategy_registers_at_class_definition(
    isolated_strategy_registry: dict[
        str, type[AlignedImageStackKwargResolutionStrategy]
    ],
) -> None:
    class DynamicKwarg:
        pass

    class DynamicKwargResolutionStrategy(
        AlignedImageStackKwargResolutionStrategy
    ):
        value_type = DynamicKwarg

        def resolve(
            self,
            value: Any,
            resolver: AlignedImageStackKwargResolver,
        ) -> Any:
            del resolver
            return value

    assert DynamicKwargResolutionStrategy.value_type_label is not None
    assert (
        isolated_strategy_registry[DynamicKwargResolutionStrategy.value_type_label]
        is DynamicKwargResolutionStrategy
    )
    assert type(
        AlignedImageStackKwargResolutionStrategy.require_nominal_value(
            DynamicKwarg(),
            context="dynamic aligned kwarg",
        )
    ) is DynamicKwargResolutionStrategy


def test_aligned_kwarg_strategy_resolution_follows_exact_value_mro(
    isolated_strategy_registry: dict[
        str, type[AlignedImageStackKwargResolutionStrategy]
    ],
) -> None:
    del isolated_strategy_registry

    class BaseKwarg:
        pass

    class DerivedKwarg(BaseKwarg):
        pass

    class MostDerivedKwarg(DerivedKwarg):
        pass

    class BaseKwargResolutionStrategy(AlignedImageStackKwargResolutionStrategy):
        value_type = BaseKwarg

        def resolve(
            self,
            value: Any,
            resolver: AlignedImageStackKwargResolver,
        ) -> Any:
            del resolver
            return value

    class DerivedKwargResolutionStrategy(AlignedImageStackKwargResolutionStrategy):
        value_type = DerivedKwarg

        def resolve(
            self,
            value: Any,
            resolver: AlignedImageStackKwargResolver,
        ) -> Any:
            del resolver
            return value

    assert AlignedImageStackKwargResolutionStrategy.strategy_types_for_nominal_type(
        MostDerivedKwarg
    ) == (
        DerivedKwargResolutionStrategy,
        BaseKwargResolutionStrategy,
        PassThroughAlignedKwargResolutionStrategy,
    )
    assert type(
        AlignedImageStackKwargResolutionStrategy.require_nominal_value(
            MostDerivedKwarg(),
            context="MRO-aligned kwarg",
        )
    ) is DerivedKwargResolutionStrategy
