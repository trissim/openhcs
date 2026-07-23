"""Nominal source-literal hooks for objects with no structural constructor."""

from __future__ import annotations

from abc import ABC, abstractmethod


class PythonSourceLiteral(ABC):
    """Object that owns its canonical importable Python source expression."""

    @abstractmethod
    def source_literal(self) -> str:
        """Return the Python expression that reconstructs this value."""

    def source_literal_imports(self) -> frozenset[tuple[str, str]]:
        """Return imports required by :meth:`source_literal`."""
        return frozenset()
