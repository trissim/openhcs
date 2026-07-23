"""Headless ObjectState field filtering and list projection policies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta

from openhcs.agent.dto.common import JsonValue, SCHEMA_VERSION
from openhcs.agent.dto.ui_bridge import (
    UiObjectStateFieldFilter,
    UiObjectStateFieldListQuery,
    UiObjectStateFieldListResult,
    UiObjectStateFieldPathIndex,
    UiObjectStateFieldProjection,
    UiObjectStateFieldScopeProjection,
    UiObjectStateFieldSummary,
    UiObjectStateScopeCatalog,
)


class ObjectStateFieldFilterDeclaration(ABC, metaclass=AutoRegisterMeta):
    """Nominal semantic predicate for one agent-facing ObjectState filter."""

    __registry_key__ = "field_filter"
    __skip_if_no_key__ = True

    field_filter: ClassVar[UiObjectStateFieldFilter | None] = None

    @classmethod
    @abstractmethod
    def matches(cls, field: UiObjectStateFieldSummary) -> bool:
        raise NotImplementedError

    @classmethod
    def declaration_for(
        cls,
        field_filter: UiObjectStateFieldFilter,
    ) -> type["ObjectStateFieldFilterDeclaration"]:
        try:
            return cls.__registry__[field_filter]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported ObjectState field filter: {field_filter.value}"
            ) from exc

    @classmethod
    def matches_filter(
        cls,
        field_filter: UiObjectStateFieldFilter,
        field: UiObjectStateFieldSummary,
    ) -> bool:
        return cls.declaration_for(field_filter).matches(field)


class AllObjectStateFieldFilter(ObjectStateFieldFilterDeclaration):
    field_filter = UiObjectStateFieldFilter.ALL

    @classmethod
    def matches(cls, field: UiObjectStateFieldSummary) -> bool:
        del field
        return True


class DirtyObjectStateFieldFilter(ObjectStateFieldFilterDeclaration):
    field_filter = UiObjectStateFieldFilter.DIRTY

    @classmethod
    def matches(cls, field: UiObjectStateFieldSummary) -> bool:
        return field.dirty


class DefaultDiffObjectStateFieldFilter(ObjectStateFieldFilterDeclaration):
    field_filter = UiObjectStateFieldFilter.DEFAULT_DIFF

    @classmethod
    def matches(cls, field: UiObjectStateFieldSummary) -> bool:
        return field.signature_diff


class InheritedObjectStateFieldFilter(ObjectStateFieldFilterDeclaration):
    field_filter = UiObjectStateFieldFilter.INHERITED

    @classmethod
    def matches(cls, field: UiObjectStateFieldSummary) -> bool:
        return field.inherited_value


class RawResolvedObjectStateFieldFilter(ObjectStateFieldFilterDeclaration):
    field_filter = UiObjectStateFieldFilter.RAW_RESOLVED

    @classmethod
    def matches(cls, field: UiObjectStateFieldSummary) -> bool:
        return field.raw_value_is_none != field.resolved_value_is_none


class SemanticObjectStateFieldFilter(ObjectStateFieldFilterDeclaration):
    field_filter = UiObjectStateFieldFilter.SEMANTIC

    @classmethod
    def matches(cls, field: UiObjectStateFieldSummary) -> bool:
        return (
            field.dirty
            or field.signature_diff
            or field.inherited_value
            or RawResolvedObjectStateFieldFilter.matches(field)
        )


class ObjectStateFieldListProjector:
    """Project scope catalogs through an ObjectState field query."""

    @classmethod
    def project_catalog(
        cls,
        query: UiObjectStateFieldListQuery,
        catalog: UiObjectStateScopeCatalog,
    ) -> UiObjectStateFieldListResult:
        matched_scopes: list[UiObjectStateFieldScopeProjection] = []
        matched_field_count = 0
        returned_field_count = 0
        source_truncated = False
        remaining_offset = query.field_offset
        requested_field_paths = set(query.field_paths)
        requested_scope_ids = set(query.scope_ids)
        contains_terms = query.contains_terms

        for scope in catalog.scopes:
            scope_id = scope.identity.object_state_scope_id
            if requested_scope_ids and scope_id not in requested_scope_ids:
                continue
            if scope.field_page is not None and scope.field_page.truncated:
                source_truncated = True
            field_index = UiObjectStateFieldPathIndex(tuple(scope.fields))
            container_field_paths = field_index.container_field_paths
            fields: list[UiObjectStateFieldProjection] = []
            for field in scope.fields:
                field_path = field.address.field_path
                exact_field_match = field_path in requested_field_paths
                if requested_field_paths and not exact_field_match:
                    continue
                if contains_terms and not any(
                    term in field_path.lower() for term in contains_terms
                ):
                    continue
                if (
                    not query.include_container_fields
                    and not exact_field_match
                    and field_path in container_field_paths
                ):
                    continue
                if not query.include_clean_fields and not (
                    ObjectStateFieldFilterDeclaration.matches_filter(
                        UiObjectStateFieldFilter.SEMANTIC,
                        field,
                    )
                ):
                    continue
                if not ObjectStateFieldFilterDeclaration.matches_filter(
                    query.field_filter,
                    field,
                ):
                    continue
                matched_field_count += 1
                if remaining_offset:
                    remaining_offset -= 1
                    continue
                if query.return_limit and returned_field_count >= query.return_limit:
                    continue
                fields.append(cls.project_field(query, field))
                returned_field_count += 1
            if fields:
                matched_scopes.append(
                    UiObjectStateFieldScopeProjection(
                        scope_id=scope_id,
                        object_type=scope.object_type,
                        dirty_field_count=scope.dirty_field_count,
                        signature_diff_field_count=(
                            scope.signature_diff_field_count
                        ),
                        has_unsaved_changes=scope.has_unsaved_changes,
                        has_default_overrides=scope.has_default_overrides,
                        fields=tuple(fields),
                    )
                )

        next_offset = None
        if matched_field_count > query.field_offset + returned_field_count:
            next_offset = query.field_offset + returned_field_count

        return UiObjectStateFieldListResult(
            schema_version=SCHEMA_VERSION,
            object_state_token=catalog.object_state_token,
            current_branch=catalog.current_branch,
            current_snapshot_index=catalog.current_snapshot_index,
            requested_scope_ids=query.scope_ids,
            field_paths=query.field_paths,
            field_path_contains=query.field_path_contains,
            field_filter=query.field_filter.value,
            include_container_fields=query.include_container_fields,
            matched_scope_count=len(matched_scopes),
            matched_field_count=matched_field_count,
            returned_field_count=returned_field_count,
            field_limit=query.field_limit,
            field_offset=query.field_offset,
            next_offset=next_offset,
            truncated=(
                source_truncated
                or matched_field_count > query.field_offset + returned_field_count
            ),
            scopes=tuple(matched_scopes),
            errors=catalog.errors,
            warnings=catalog.warnings,
        )

    @classmethod
    def project_field(
        cls,
        query: UiObjectStateFieldListQuery,
        field: UiObjectStateFieldSummary,
    ) -> UiObjectStateFieldProjection:
        return UiObjectStateFieldProjection(
            field_path=field.address.field_path,
            field_name=field.field_name,
            container_path=field.container_path,
            object_state_path_type=field.object_state_path_type,
            dirty=field.dirty,
            signature_diff=field.signature_diff,
            last_changed=field.last_changed,
            semantic_markers=field.semantic_markers,
            raw_value_type=field.raw_value_type,
            resolved_value_type=field.resolved_value_type,
            raw_value_preview=field.raw_value_preview,
            resolved_value_preview=field.resolved_value_preview,
            raw_value=cls.compact_value(query, field.raw_value),
            resolved_value=cls.compact_value(query, field.resolved_value),
            raw_value_is_none=field.raw_value_is_none,
            resolved_value_is_none=field.resolved_value_is_none,
            inherited_value=field.inherited_value,
            provenance=field.provenance,
        )

    @staticmethod
    def compact_value(
        query: UiObjectStateFieldListQuery,
        value: JsonValue | None,
    ) -> JsonValue | None:
        if isinstance(value, str) and len(value) > query.max_value_chars:
            return {
                "type": "str",
                "length": len(value),
                "truncated": True,
                "preview": value[: query.max_value_chars],
            }
        if isinstance(value, (list, tuple)) and len(value) > query.max_value_items:
            return {
                "type": "sequence",
                "length": len(value),
                "truncated": True,
                "preview": tuple(value[: query.max_value_items]),
            }
        if isinstance(value, Mapping) and len(value) > query.max_value_items:
            preview = tuple(value.items())[: query.max_value_items]
            return {
                "type": "mapping",
                "length": len(value),
                "truncated": True,
                "preview": {str(key): item for key, item in preview},
            }
        return value
