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
        slice_count: int,
    ) -> SliceValueT:
        """Return the value in an outer aligned slice context.

        A single carried slice is broadcast across the outer alignment; otherwise
        the carried slice count must match the outer count exactly.
        """
        if self.slice_count == slice_count:
            return self.value_for_slice(slice_index)
        if self.slice_count == 1:
            return self.value_for_slice(0)
        if slice_count % self.slice_count == 0:
            return self.value_for_slice(slice_index % self.slice_count)
        raise ValueError(
            "Runtime-slice-aligned value has incompatible slice count "
            f"{self.slice_count}; expected a divisor of {slice_count}."
        )


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
