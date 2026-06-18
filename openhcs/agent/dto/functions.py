"""Function catalog DTOs for agent-facing OpenHCS discovery."""

from __future__ import annotations

from dataclasses import dataclass

from openhcs.agent.dto.common import SCHEMA_VERSION


@dataclass(frozen=True)
class FunctionIdentity:
    function_id: str


@dataclass(frozen=True)
class ImportPathRef:
    import_path: str


@dataclass(frozen=True, slots=True)
class FunctionCatalogEntry(FunctionIdentity, ImportPathRef):
    name: str
    module: str
    library: str
    signature: str
    summary: str | None
    backend_tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FunctionParameterSpec:
    name: str
    annotation: str | None
    default_repr: str | None
    required: bool
    description: str | None = None


@dataclass(frozen=True, slots=True)
class FunctionDetail:
    schema_version: str
    entry: FunctionCatalogEntry
    parameters: tuple[FunctionParameterSpec, ...]
    doc: str | None


@dataclass(frozen=True, slots=True)
class FunctionCatalogPage:
    schema_version: str
    items: tuple[FunctionCatalogEntry, ...]
    total: int
    limit: int
    query: str | None = None
    library: str | None = None


def catalog_page(
    *,
    items: tuple[FunctionCatalogEntry, ...],
    total: int,
    limit: int,
    query: str | None,
    library: str | None,
) -> FunctionCatalogPage:
    return FunctionCatalogPage(
        schema_version=SCHEMA_VERSION,
        items=items,
        total=total,
        limit=limit,
        query=query,
        library=library,
    )
