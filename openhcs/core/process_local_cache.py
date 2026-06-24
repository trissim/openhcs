"""Small process-local cache substrates for immutable runtime projections."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from collections.abc import Sequence
from typing import Any, ClassVar, Generic, TypeVar

from metaclass_registry import AutoRegisterMeta


CacheKey = TypeVar("CacheKey")
CachedValue = TypeVar("CachedValue")


def identity_owner_tuples_match(
    left: Sequence[object],
    right: Sequence[object],
) -> bool:
    """Return whether identity-keyed cache owners still reference the same objects."""
    return (
        len(left) == len(right)
        and all(
            left_owner is right_owner
            for left_owner, right_owner in zip(left, right, strict=True)
        )
    )


def named_identity_owner_tuples_match(
    left: Sequence[tuple[str, object]],
    right: Sequence[tuple[str, object]],
) -> bool:
    """Return whether named identity-keyed cache owners still match."""
    return (
        len(left) == len(right)
        and all(
            left_name == right_name and left_owner is right_owner
            for (left_name, left_owner), (right_name, right_owner) in zip(
                left,
                right,
                strict=True,
            )
        )
    )


@dataclass(slots=True)
class ProcessLocalBoundedCache(Generic[CacheKey, CachedValue]):
    """Bounded LRU cache with one singleton instance per concrete subclass."""

    max_entries: int = 4096
    entries: OrderedDict[CacheKey, CachedValue] = field(default_factory=OrderedDict)

    def cached_value(self, key: CacheKey) -> CachedValue | None:
        if key not in self.entries:
            return None
        value = self.entries[key]
        self.entries.move_to_end(key)
        return value

    def store_value(self, key: CacheKey, value: CachedValue) -> CachedValue:
        self.entries[key] = value
        self.entries.move_to_end(key)
        while len(self.entries) > self.max_entries:
            self.entries.popitem(last=False)
        return value

    @classmethod
    def process_cache(cls) -> "ProcessLocalBoundedCache[CacheKey, CachedValue]":
        cache = cls._process_cache
        if cache is None:
            cache = cls()
            cls._process_cache = cache
        return cache

    _process_cache: ClassVar["ProcessLocalBoundedCache[object, object] | None"] = None


class IdentityBoundProcessCache(
    ProcessLocalBoundedCache[int, tuple[object, Any]],
    metaclass=AutoRegisterMeta,
):
    """Process-local cache whose keys are protected against id reuse."""

    __registry_key__ = "registry_key"
    __skip_if_no_key__ = True

    max_entries = 4096
    registry_key: ClassVar[str | None] = None

    def get_bound(
        self,
        owner: object,
    ) -> Any | None:
        cache_key = id(owner)
        cached = self.cached_value(cache_key)
        if cached is None:
            return None
        cached_owner, value = cached
        if cached_owner is not owner:
            del self.entries[cache_key]
            return None
        return value

    def put_bound(
        self,
        owner: object,
        value: Any,
    ) -> Any:
        return self.store_value(id(owner), (owner, value))[1]
