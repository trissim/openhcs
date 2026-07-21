"""Nominal runtime plane projection semantics."""

from __future__ import annotations
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from metaclass_registry import AutoRegisterMeta
from openhcs.core.registry_strategies import EnumKeyedStrategyMixin
from collections.abc import Sequence
from typing import ClassVar
from typing import Self


class RuntimeSliceProjectableValue(ABC):
    """Nominal contract for values that own runtime-slice row projection."""

    @abstractmethod
    def project_runtime_slice(self, slice_index: int) -> object:
        """Return the value represented by one runtime-slice index."""


class RuntimeSliceInvariantValue(RuntimeSliceProjectableValue):
    """Nominal contract for values unchanged by runtime-slice projection."""

    def project_runtime_slice(self, slice_index: int) -> Self:
        """Preserve this value for every runtime slice."""
        del slice_index
        return self


class RuntimeSliceIdentityProjectableValue(ABC):
    """Nominal contract for values that can be stamped with execution-slice identity."""

    @abstractmethod
    def with_runtime_slice_identity(
        self, *, slice_index: int, slice_count: int
    ) -> Self:
        """Return the value with execution-slice identity applied."""


class RuntimePlaneAxis(str, Enum):
    """Semantic meaning of the leading plane axis on runtime array stacks."""

    RUNTIME_SLICE = "runtime_slice"
    SOURCE_BINDING = "source_binding"


class RuntimePlaneAxisStrategy(
    EnumKeyedStrategyMixin[RuntimePlaneAxis], ABC, metaclass=AutoRegisterMeta
):
    """Own all behavior for one nominal runtime plane axis."""

    __enum_member_attr__ = "axis"
    axis: ClassVar[RuntimePlaneAxis]
    strategy_label: ClassVar[str | None] = None

    @abstractmethod
    def plane_index(
        self,
        projector: "RuntimePlaneAxisProjector",
        source_aliases: tuple[str, ...],
    ) -> int | None:
        """Resolve this axis against an execution-local projector."""

    @abstractmethod
    def axis_size(
        self,
        projector: "RuntimePlaneAxisProjector",
        source_aliases: tuple[str, ...],
    ) -> int | None:
        """Resolve this axis cardinality against an execution-local projector."""

    @abstractmethod
    def projected_axis(self, projected_plane_count: int) -> RuntimePlaneAxis | None:
        """Return the axis retained after an exact runtime projection."""


class RuntimeSlicePlaneAxisStrategy(RuntimePlaneAxisStrategy):
    """Runtime-slice axis behavior."""

    axis = RuntimePlaneAxis.RUNTIME_SLICE

    def plane_index(
        self,
        projector: "RuntimePlaneAxisProjector",
        source_aliases: tuple[str, ...],
    ) -> int | None:
        del source_aliases
        return projector.runtime_slice_plane_index()

    def axis_size(
        self,
        projector: "RuntimePlaneAxisProjector",
        source_aliases: tuple[str, ...],
    ) -> int | None:
        del source_aliases
        return projector.runtime_slice_axis_size()

    def projected_axis(self, projected_plane_count: int) -> RuntimePlaneAxis | None:
        if projected_plane_count != 1:
            raise ValueError(
                "Runtime-slice object-label projection must select exactly one "
                f"plane, got {projected_plane_count}."
            )
        return None


class SourceBindingPlaneAxisStrategy(RuntimePlaneAxisStrategy):
    """Source-binding axis behavior."""

    axis = RuntimePlaneAxis.SOURCE_BINDING

    def plane_index(
        self,
        projector: "RuntimePlaneAxisProjector",
        source_aliases: tuple[str, ...],
    ) -> int | None:
        return projector.source_binding_axis_plane_index(source_aliases)

    def axis_size(
        self,
        projector: "RuntimePlaneAxisProjector",
        source_aliases: tuple[str, ...],
    ) -> int | None:
        return projector.source_binding_axis_size(source_aliases)

    def projected_axis(self, projected_plane_count: int) -> RuntimePlaneAxis | None:
        if projected_plane_count <= 0:
            raise ValueError(
                "Source-binding object-label projection must select at least one plane."
            )
        if projected_plane_count == 1:
            return None
        return RuntimePlaneAxis.SOURCE_BINDING


class RuntimePlaneAxisProjector(ABC):
    """Nominal provider for execution-local runtime plane selection."""

    @abstractmethod
    def runtime_slice_plane_index(self) -> int | None:
        """Return the execution-local runtime-slice plane index."""

    def runtime_slice_axis_size(self) -> int | None:
        """Return the runtime-slice axis size for the current execution scope."""
        return None

    def source_binding_axis_plane_index(
        self, source_aliases: tuple[str, ...]
    ) -> int | None:
        """Return the execution-local source-binding plane index."""
        raise NotImplementedError(
            f"{type(self).__name__} does not provide source-binding plane projection."
        )

    def source_binding_axis_size(self, source_aliases: tuple[str, ...]) -> int | None:
        """Return the source-binding axis size for this execution scope."""
        return None


@dataclass(frozen=True, slots=True)
class RuntimePlaneProjection(RuntimePlaneAxisProjector):
    """Preserve a runtime stack or select one explicitly proven plane."""

    plane_index: int | None = None
    plane_count: int | None = None

    def __post_init__(self) -> None:
        if self.plane_count is None:
            plane_count = None
        else:
            plane_count = int(self.plane_count)
            if plane_count <= 0:
                raise ValueError(
                    "Runtime-plane projection plane_count must be positive."
                )
            object.__setattr__(self, "plane_count", plane_count)
        if self.plane_index is None:
            return
        plane_index = int(self.plane_index)
        if plane_index < 0:
            raise ValueError("Runtime-plane projection plane_index cannot be negative.")
        if plane_count is not None and plane_index >= plane_count:
            raise ValueError(
                "Runtime-plane projection plane_index must be within plane_count: "
                f"index {plane_index}, count {plane_count}."
            )
        object.__setattr__(self, "plane_index", plane_index)

    @classmethod
    def stack(cls, plane_count: int | None = None) -> "RuntimePlaneProjection":
        """Preserve runtime-slice stacks for stack-scoped execution."""
        return cls(plane_count=plane_count)

    @classmethod
    def selected(
        cls, plane_index: int, plane_count: int | None = None
    ) -> "RuntimePlaneProjection":
        """Select one runtime-slice plane from an explicit projection proof."""
        return cls(plane_index=plane_index, plane_count=plane_count)

    def runtime_slice_plane_index(self) -> int | None:
        """Return selected runtime-slice plane, or None when stacks are preserved."""
        return self.plane_index

    def runtime_slice_axis_size(self) -> int | None:
        """Return the grouped runtime-slice axis size when known."""
        return self.plane_count


@dataclass(frozen=True, slots=True)
class RuntimePlaneAxisValueProjection(RuntimeSliceProjectableValue):
    """Projection of values that explicitly carry a declared runtime plane axis."""

    axis: RuntimePlaneAxis
    source_aliases: tuple[str, ...]
    plane_index: int | None
    axis_size: int

    def __post_init__(self) -> None:
        axis = RuntimePlaneAxis(
            self.axis,
        )
        source_aliases = tuple(self.source_aliases)
        if any(not isinstance(alias, str) or not alias for alias in source_aliases):
            raise ValueError(
                "RuntimePlaneAxisValueProjection.source_aliases must contain "
                "non-empty strings."
            )
        axis_size = int(self.axis_size)
        if axis_size <= 0:
            raise ValueError(
                "RuntimePlaneAxisValueProjection.axis_size must be positive."
            )
        plane_index = self.plane_index
        if plane_index is not None:
            plane_index = int(plane_index)
            if plane_index < 0 or plane_index >= axis_size:
                raise ValueError(
                    "RuntimePlaneAxisValueProjection.plane_index must be within "
                    f"axis_size: index {plane_index}, size {axis_size}."
                )
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "source_aliases", source_aliases)
        object.__setattr__(self, "axis_size", axis_size)
        object.__setattr__(self, "plane_index", plane_index)

    @classmethod
    def from_projector(
        cls,
        projector: RuntimePlaneAxisProjector | None,
        axis: RuntimePlaneAxis,
        source_aliases: tuple[str, ...],
    ) -> "RuntimePlaneAxisValueProjection | None":
        """Return the runtime-axis projection declared by a runtime projector."""
        if projector is None:
            return None
        if not isinstance(projector, RuntimePlaneAxisProjector):
            raise TypeError(
                f"Runtime plane-axis projection requires RuntimePlaneAxisProjector, got {type(projector).__name__}."
            )
        axis = RuntimePlaneAxis(axis)
        source_aliases = tuple(source_aliases)
        strategy = RuntimePlaneAxisStrategy.for_enum_member(axis)
        axis_size = strategy.axis_size(projector, source_aliases)
        if axis_size is None:
            return None
        return cls(
            axis=axis,
            source_aliases=source_aliases,
            plane_index=strategy.plane_index(projector, source_aliases),
            axis_size=axis_size,
        )

    @classmethod
    def require_from_projector(
        cls,
        projector: RuntimePlaneAxisProjector,
        axis: RuntimePlaneAxis,
        source_aliases: tuple[str, ...] = (),
    ) -> "RuntimePlaneAxisValueProjection":
        """Return the complete projection declared for one runtime image axis."""

        projection = cls.from_projector(projector, axis, source_aliases)
        if projection is None:
            raise ValueError(
                f"Declared {axis.value!r} image axis has no runtime cardinality."
            )
        return projection

    @classmethod
    def from_selected_plane(
        cls,
        *,
        axis: RuntimePlaneAxis,
        plane_index: int,
        axis_size: int,
        source_aliases: tuple[str, ...] = (),
    ) -> "RuntimePlaneAxisValueProjection":
        """Return a projection whose explicit plane proof is already resolved."""
        return cls(
            axis=axis,
            source_aliases=tuple(source_aliases),
            plane_index=plane_index,
            axis_size=axis_size,
        )

    @classmethod
    def preserve(
        cls,
        *,
        axis: RuntimePlaneAxis,
        axis_size: int,
        source_aliases: tuple[str, ...] = (),
    ) -> "RuntimePlaneAxisValueProjection":
        """Declare a complete runtime axis without selecting one plane."""

        return cls(
            axis=axis,
            source_aliases=tuple(source_aliases),
            plane_index=None,
            axis_size=axis_size,
        )

    def selected_plane(self, plane_index: int) -> "RuntimePlaneAxisValueProjection":
        """Select one plane while preserving this declaration's exact axis."""

        return type(self).from_selected_plane(
            axis=self.axis,
            source_aliases=self.source_aliases,
            plane_index=plane_index,
            axis_size=self.axis_size,
        )

    def project_runtime_slice(
        self, slice_index: int
    ) -> "RuntimePlaneAxisValueProjection":
        """Select the projection carried into one runtime-slice invocation."""
        return self.selected_plane(slice_index)

    def require_plane_index(self) -> int:
        """Return the selected plane index required by a projected invocation."""

        if self.plane_index is None:
            raise ValueError(
                "Runtime plane projection requires an explicitly selected plane."
            )
        return self.plane_index

    def dense_shape_carries_axis(self, shape: Sequence[int]) -> bool:
        """Return whether a dense shape carries this declared leading axis."""

        shape = tuple(int(size) for size in shape)
        return len(shape) >= 3 and shape[0] == self.axis_size

    def validate_shape(self, shape: Sequence[int], *, value_name: str) -> None:
        """Validate a dense shape against this declared runtime axis."""
        shape = tuple(int(size) for size in shape)
        if not self.dense_shape_carries_axis(shape):
            raise ValueError(
                f"{value_name} does not match its declared {self.axis.value!r} "
                f"axis of size {self.axis_size}: shape {shape!r}."
            )

    @staticmethod
    def validate_plane_index(plane_index: int, shape: tuple[int, ...]) -> None:
        """Validate a selected source-binding plane against dense data shape."""
        if plane_index < 0 or plane_index >= shape[0]:
            raise RuntimeError(
                f"Runtime plane-axis projection produced an out-of-range plane index {plane_index} for shape {shape!r}."
            )
