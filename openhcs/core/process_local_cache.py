"""Small process-local cache substrates for immutable runtime projections."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import ClassVar, Generic, TypeVar


CacheKey = TypeVar("CacheKey")
CachedValue = TypeVar("CachedValue")


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
