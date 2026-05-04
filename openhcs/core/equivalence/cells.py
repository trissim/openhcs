"""Scalar cell signatures for runtime equivalence."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from openhcs.core.equivalence.policy import RuntimeEquivalencePolicy


class RuntimeCellValueKind(str, Enum):
    """Canonical scalar families used for exported table comparison."""

    EMPTY = "empty"
    NUMBER = "number"
    TEXT = "text"


@dataclass(frozen=True, slots=True)
class RuntimeCellSignature:
    """Canonical scalar value for exported table comparison."""

    kind: RuntimeCellValueKind
    value: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "kind",
            (
                self.kind
                if isinstance(self.kind, RuntimeCellValueKind)
                else RuntimeCellValueKind(self.kind)
            ),
        )

    @property
    def sort_key(self) -> tuple[str, str]:
        """Return a stable ordering key for mixed scalar families."""
        return (self.kind.value, self.value)

    def to_cache_payload(self) -> tuple[str, str]:
        """Return a pickle/JSON-stable semantic cache payload."""
        return (self.kind.value, self.value)

    @classmethod
    def from_cache_payload(cls, payload: object) -> "RuntimeCellSignature":
        """Rebuild a cell signature from a semantic cache payload."""
        kind, value = payload  # type: ignore[misc]
        return cls(RuntimeCellValueKind(str(kind)), str(value))


def runtime_cell_signature(
    value: str,
    policy: RuntimeEquivalencePolicy,
) -> RuntimeCellSignature:
    """Return a canonical scalar signature for runtime table comparison."""
    text = value.strip()
    if not text:
        return RuntimeCellSignature(RuntimeCellValueKind.EMPTY, "")
    try:
        numeric = float(text)
    except ValueError:
        return RuntimeCellSignature(RuntimeCellValueKind.TEXT, text)
    if math.isnan(numeric):
        canonical = "nan"
    elif math.isinf(numeric):
        canonical = "inf" if numeric > 0 else "-inf"
    else:
        canonical = repr(round(numeric, policy.numeric_decimal_places))
    return RuntimeCellSignature(RuntimeCellValueKind.NUMBER, canonical)
