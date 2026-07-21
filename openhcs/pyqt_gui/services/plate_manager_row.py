"""Typed row contract for PlateManager batch and list workflows."""

from __future__ import annotations

from dataclasses import dataclass

from openhcs.ui.shared.plate_scope_identity import PlateScopeIdentity


@dataclass(frozen=True, slots=True)
class PlateManagerRow:
    """Typed visible row in the PlateManager list."""

    identity: PlateScopeIdentity
    cppipe_path_override: str | None = None

    @classmethod
    def from_scope(
        cls,
        scope_id: str,
        *,
        cppipe_path: str | None = None,
    ) -> "PlateManagerRow":
        return cls(
            identity=PlateScopeIdentity.from_scope_id(scope_id),
            cppipe_path_override=cppipe_path,
        )

    @property
    def scope_id(self) -> str:
        return self.identity.scope_id

    @property
    def name(self) -> str:
        return self.identity.display_name

    @property
    def plate_root(self) -> str:
        return str(self.identity.plate_root)

    @property
    def cppipe_path(self) -> str | None:
        if self.cppipe_path_override is not None:
            return str(self.cppipe_path_override)
        if self.identity.cppipe_path is None:
            return None
        return str(self.identity.cppipe_path)
