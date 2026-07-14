"""Typed payloads consumed by generic materialization writers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence


class MaterializationPayload(ABC):
    """Nominal protocol for special-output payload lifecycle operations."""

    @abstractmethod
    def bind_source_paths(
        self, source_paths: Sequence[str]
    ) -> "MaterializationPayload":
        """Return a payload bound to the images used for this function call."""

    @abstractmethod
    def merge(self, other: "MaterializationPayload") -> "MaterializationPayload":
        """Combine payloads emitted by repeated 2D pattern calls."""


@dataclass(frozen=True)
class AlignedROIMask:
    """One 2D labeled mask aligned to an image in the function input stack.

    ``source_index`` is the mask's image index in the exact stack passed to the
    processing function. ``role`` distinguishes the semantic mask type without
    requiring the generic ROI writer to know analysis-specific names.
    """

    mask: Any
    source_index: int
    role: str
    label_metadata: Mapping[int, Mapping[str, Any]] = field(default_factory=dict)
    source_path: str | None = None

    def __post_init__(self) -> None:
        if self.source_index < 0:
            raise ValueError("source_index must be >= 0")
        if not self.role or not self.role.strip():
            raise ValueError("role must be a non-empty string")
        if getattr(self.mask, "ndim", None) != 2:
            raise ValueError(
                f"AlignedROIMask requires a 2D mask, got "
                f"shape {getattr(self.mask, 'shape', None)}"
            )


@dataclass(frozen=True)
class AlignedROIMasks(MaterializationPayload):
    """A collection of independently aligned 2D ROI masks."""

    masks: tuple[AlignedROIMask, ...]

    def __post_init__(self) -> None:
        if not self.masks:
            raise ValueError("AlignedROIMasks requires at least one mask")

    def bind_source_paths(self, source_paths: Sequence[str]) -> "AlignedROIMasks":
        bound_masks = []
        for aligned_mask in self.masks:
            if aligned_mask.source_index >= len(source_paths):
                raise IndexError(
                    f"Aligned ROI source_index {aligned_mask.source_index} is outside "
                    f"the {len(source_paths)} source images"
                )
            bound_masks.append(
                replace(
                    aligned_mask,
                    source_path=str(source_paths[aligned_mask.source_index]),
                )
            )
        return AlignedROIMasks(tuple(bound_masks))

    def merge(self, other: MaterializationPayload) -> "AlignedROIMasks":
        if not isinstance(other, AlignedROIMasks):
            raise TypeError(f"Cannot merge AlignedROIMasks with {type(other).__name__}")
        return AlignedROIMasks(self.masks + other.masks)
