"""Nominal runtime slice-alignment payloads."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar


SliceValueT = TypeVar("SliceValueT")


class RuntimeSliceAlignedValueSet(ABC, Generic[SliceValueT]):
    """Nominal base for non-image values aligned to runtime slices."""

    @property
    @abstractmethod
    def slice_count(self) -> int:
        """Return the number of runtime slices carried by this value."""

    @abstractmethod
    def value_for_slice(self, slice_index: int) -> SliceValueT:
        """Return the value for one runtime slice."""

    def value_for_aligned_slice(
        self,
        slice_index: int,
        slice_count: int | None,
    ) -> SliceValueT:
        """Return the value for an explicitly declared, exactly aligned slice."""
        if slice_count is None:
            raise ValueError(
                "Runtime-slice-aligned value requires a declared outer slice count."
            )
        if self.slice_count != slice_count:
            raise ValueError(
                "Runtime-slice-aligned value count must exactly match the declared "
                f"outer slice count: {self.slice_count} != {slice_count}."
            )
        if slice_index < 0 or slice_index >= slice_count:
            raise ValueError(
                "Runtime-slice-aligned value index is outside the declared outer "
                f"slice count: index {slice_index}, count {slice_count}."
            )
        return self.value_for_slice(slice_index)


@dataclass(frozen=True, slots=True)
class RuntimeSliceAlignedValues(RuntimeSliceAlignedValueSet[SliceValueT]):
    """Non-image payload with one backend-native value per runtime slice."""

    slices: tuple[SliceValueT, ...]

    def __post_init__(self) -> None:
        slices = tuple(self.slices)
        if not slices:
            raise ValueError("RuntimeSliceAlignedValues.slices cannot be empty.")
        object.__setattr__(self, "slices", slices)

    @property
    def slice_count(self) -> int:
        return len(self.slices)

    def value_for_slice(self, slice_index: int) -> SliceValueT:
        return self.slices[slice_index]
