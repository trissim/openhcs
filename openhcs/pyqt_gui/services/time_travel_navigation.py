"""Typed helpers for OpenHCS time-travel navigation decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional

from pyqt_reactive.services.function_navigation import (
    FUNCTION_FIELD_ROOT,
    build_function_token_field_path,
    is_function_field_path,
)

from objectstate.object_state import ObjectState

FUNCTION_STEP_SCOPE_PREFIX = "functionstep_"
FIELD_PATH_TARGET_KIND: Literal["field_path"] = "field_path"
FUNCTION_TOKEN_TARGET_KIND: Literal["function_token"] = "function_token"


@dataclass(frozen=True)
class FunctionScopeRef:
    """Typed parse result for function ObjectState scopes."""

    step_scope_id: str
    function_token: str


@dataclass(frozen=True)
class TimeTravelNavigationTarget:
    """Typed time-travel navigation target."""

    kind: Literal["field_path", "function_token"]
    value: str

    @property
    def is_function_target(self) -> bool:
        return self.kind == FUNCTION_TOKEN_TARGET_KIND or is_function_field_path(self.value)

    def to_field_path(self) -> str:
        if self.kind == FUNCTION_TOKEN_TARGET_KIND:
            return build_function_token_field_path(self.value)
        return self.value


@dataclass(frozen=True)
class TimeTravelWindowRequest:
    """Window request derived from one or more time-travel state changes."""

    scope_id: str
    object_state: ObjectState
    target: TimeTravelNavigationTarget | None

    def replace_target(
        self,
        target: TimeTravelNavigationTarget | None,
    ) -> "TimeTravelWindowRequest":
        return TimeTravelWindowRequest(
            scope_id=self.scope_id,
            object_state=self.object_state,
            target=target,
        )


@dataclass(frozen=True)
class TimeTravelSourceScope:
    """Changed scope plus the semantic scope that triggered time-travel."""

    changed_scope_id: str
    triggering_scope: str | None


def parse_function_scope_ref(scope_id: str) -> FunctionScopeRef | None:
    """Parse function ObjectState scope into typed reference."""
    parts = scope_id.rsplit("::", 2)
    if len(parts) < 3:
        return None
    parent_scope, step_token, function_token = parts
    if not step_token.startswith(FUNCTION_STEP_SCOPE_PREFIX):
        return None
    if not function_token:
        return None
    step_scope_id = f"{parent_scope}::{step_token}"
    return FunctionScopeRef(
        step_scope_id=step_scope_id,
        function_token=function_token,
    )


def make_function_token_target(function_token: str) -> TimeTravelNavigationTarget:
    """Create typed function-token navigation target."""
    return TimeTravelNavigationTarget(
        kind=FUNCTION_TOKEN_TARGET_KIND,
        value=function_token,
    )


def make_field_path_target(field_path: str) -> TimeTravelNavigationTarget:
    """Create typed plain field-path navigation target."""
    return TimeTravelNavigationTarget(
        kind=FIELD_PATH_TARGET_KIND,
        value=field_path,
    )


def resolve_fallback_field_path(
    last_changed_field: str | None,
    dirty_fields: Iterable[str],
) -> Optional[str]:
    """Resolve best-effort field path when no explicit function scope mapping exists."""
    if last_changed_field:
        return last_changed_field

    candidates = [field for field in dirty_fields if isinstance(field, str)]
    if not candidates:
        return None
    sorted_fields = sorted(candidates, key=DirtyFieldSortKeyAuthority.sort_key)
    return sorted_fields[0]


def should_replace_navigation_target(
    existing_target: TimeTravelNavigationTarget | None,
    candidate_target: TimeTravelNavigationTarget | None,
) -> bool:
    """Return True when candidate should replace existing target."""
    if existing_target is None:
        return candidate_target is not None
    if candidate_target is None:
        return False
    if existing_target.is_function_target and not candidate_target.is_function_target:
        return True
    return False


def should_include_time_travel_scope(scope: TimeTravelSourceScope) -> bool:
    """Return whether a changed scope belongs to the active time-travel edit.

    ObjectState snapshots capture the whole registry, so old dirty states can
    still be present when the user undoes one edit. UI navigation should follow
    the semantic mutation owner when history provides one instead of reopening
    every dirty ObjectState in the registry.
    """
    if scope.triggering_scope is None:
        return True

    triggering_function_scope = parse_function_scope_ref(scope.triggering_scope)
    if triggering_function_scope is not None:
        if scope.changed_scope_id == scope.triggering_scope:
            return True
        if scope.changed_scope_id.startswith(f"{scope.triggering_scope}::"):
            return True
        return scope.changed_scope_id == triggering_function_scope.step_scope_id

    if scope.changed_scope_id == scope.triggering_scope:
        return True

    if (
        parse_function_scope_ref(scope.changed_scope_id) is not None
        and scope.changed_scope_id.startswith(f"{scope.triggering_scope}::")
    ):
        return False

    return False


class DirtyFieldSortKeyAuthority:
    """Classify dirty fields for deterministic time-travel navigation."""

    FUNCTION_ROOT_RANK = 0
    FIELD_PATH_RANK = 1

    @classmethod
    def sort_key(cls, field: str) -> tuple[int, int, str]:
        function_rank = cls.function_field_rank(field)
        return (function_rank, -field.count("."), field)

    @classmethod
    def function_field_rank(cls, field: str) -> int:
        if field == FUNCTION_FIELD_ROOT:
            return cls.FUNCTION_ROOT_RANK
        return cls.FIELD_PATH_RANK
