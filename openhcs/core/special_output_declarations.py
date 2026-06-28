"""Pure callable special-output declaration types."""

from __future__ import annotations

from typing import TypeAlias

from openhcs.processing.materialization import MaterializationSpec

SpecialOutputDeclaration: TypeAlias = str | tuple[str, MaterializationSpec | None]
SpecialOutputDeclarations: TypeAlias = tuple[SpecialOutputDeclaration, ...]
