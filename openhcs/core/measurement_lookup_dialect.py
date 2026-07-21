"""Nominal measurement-feature lookup dialects for runtime artifact queries."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Iterator

from openhcs.core.public_api import declared_public_names
from openhcs.core.process_local_cache import RegisteredProcessLocalBoundedCache
from openhcs.core.runtime_identifier import normalize_runtime_identifier


RuntimeMeasurementFeatureParts = tuple[str, ...]
RuntimeMeasurementFeaturePartAliases = Mapping[
    RuntimeMeasurementFeatureParts,
    RuntimeMeasurementFeatureParts,
]
RuntimeMeasurementAlternativeFeaturePartAliases = Mapping[
    RuntimeMeasurementFeatureParts,
    tuple[RuntimeMeasurementFeatureParts, ...],
]

_EMPTY_FEATURE_ALIASES: RuntimeMeasurementFeaturePartAliases = MappingProxyType({})
_EMPTY_ALTERNATIVE_FEATURE_ALIASES: (
    RuntimeMeasurementAlternativeFeaturePartAliases
) = MappingProxyType({})
_EMPTY_FEATURE_FAMILIES: tuple[tuple[str, ...], ...] = ()


def _compact_identifier(value: str) -> str:
    return value.replace("_", "")


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementObjectDomainPolicy:
    """Object-domain policy for feature lookups in a measurement dialect."""

    def query_object_name(
        self,
        lookup: "RuntimeMeasurementFeatureLookup",
        object_name: str | None,
    ) -> str | None:
        """Return the row object constraint for this feature lookup."""
        return object_name


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementLookupDialect:
    """Dialect used to resolve external measurement names to runtime fields."""

    category_prefixes: tuple[RuntimeMeasurementFeatureParts, ...] = ()
    category_prefixes_provider: Callable[
        [],
        Iterable[RuntimeMeasurementFeatureParts],
    ] | None = None
    feature_part_aliases: RuntimeMeasurementFeaturePartAliases = field(
        default_factory=lambda: _EMPTY_FEATURE_ALIASES
    )
    feature_part_aliases_provider: Callable[
        [],
        RuntimeMeasurementFeaturePartAliases,
    ] | None = None
    alternative_feature_part_aliases: (
        RuntimeMeasurementAlternativeFeaturePartAliases
    ) = field(default_factory=lambda: _EMPTY_ALTERNATIVE_FEATURE_ALIASES)
    alternative_feature_part_aliases_provider: Callable[
        [],
        RuntimeMeasurementAlternativeFeaturePartAliases,
    ] | None = None
    source_qualified_feature_families: tuple[RuntimeMeasurementFeatureParts, ...] = (
        _EMPTY_FEATURE_FAMILIES
    )
    source_qualified_feature_families_provider: Callable[
        [],
        Iterable[RuntimeMeasurementFeatureParts],
    ] | None = None
    object_domain_policy: RuntimeMeasurementObjectDomainPolicy = field(
        default_factory=RuntimeMeasurementObjectDomainPolicy
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "category_prefixes",
            tuple(
                tuple(part for part in prefix if part)
                for prefix in self.category_prefixes
            ),
        )
        object.__setattr__(
            self,
            "feature_part_aliases",
            MappingProxyType(
                {
                    tuple(part for part in parts if part): tuple(
                        part for part in alias if part
                    )
                    for parts, alias in self.feature_part_aliases.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "alternative_feature_part_aliases",
            MappingProxyType(
                {
                    tuple(part for part in parts if part): tuple(
                        tuple(alias_part for alias_part in alias if alias_part)
                        for alias in aliases
                        if tuple(alias)
                    )
                    for parts, aliases in self.alternative_feature_part_aliases.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "source_qualified_feature_families",
            self._normalized_feature_families(self.source_qualified_feature_families),
        )
        if (
            self.category_prefixes_provider is not None
            and not callable(self.category_prefixes_provider)
        ):
            raise TypeError(
                "RuntimeMeasurementLookupDialect.category_prefixes_provider "
                "must be callable."
            )
        if (
            self.feature_part_aliases_provider is not None
            and not callable(self.feature_part_aliases_provider)
        ):
            raise TypeError(
                "RuntimeMeasurementLookupDialect.feature_part_aliases_provider "
                "must be callable."
            )
        if (
            self.source_qualified_feature_families_provider is not None
            and not callable(self.source_qualified_feature_families_provider)
        ):
            raise TypeError(
                "RuntimeMeasurementLookupDialect."
                "source_qualified_feature_families_provider must be callable."
            )
        if (
            self.alternative_feature_part_aliases_provider is not None
            and not callable(self.alternative_feature_part_aliases_provider)
        ):
            raise TypeError(
                "RuntimeMeasurementLookupDialect."
                "alternative_feature_part_aliases_provider must be callable."
            )
        if not isinstance(self.object_domain_policy, RuntimeMeasurementObjectDomainPolicy):
            raise TypeError(
                "RuntimeMeasurementLookupDialect.object_domain_policy must be "
                "RuntimeMeasurementObjectDomainPolicy, got "
                f"{type(self.object_domain_policy).__name__}."
            )

    @staticmethod
    def _normalized_feature_families(
        families: Iterable[RuntimeMeasurementFeatureParts],
    ) -> tuple[RuntimeMeasurementFeatureParts, ...]:
        return tuple(
            tuple(
                part
                for part in normalize_runtime_identifier("_".join(family)).split("_")
                if part
            )
            for family in families
            if tuple(family)
        )

    def resolved_source_qualified_feature_families(
        self,
    ) -> tuple[RuntimeMeasurementFeatureParts, ...]:
        """Return static and provider-supplied source-qualified families."""
        provider = self.source_qualified_feature_families_provider
        provided = () if provider is None else provider()
        return tuple(
            dict.fromkeys(
                (
                    *self.source_qualified_feature_families,
                    *self._normalized_feature_families(provided),
                )
            )
        )

    def resolved_category_prefixes(self) -> tuple[RuntimeMeasurementFeatureParts, ...]:
        """Return static and provider-supplied measurement category prefixes."""
        provider = self.category_prefixes_provider
        provided = () if provider is None else provider()
        return tuple(
            dict.fromkeys(
                (
                    *self.category_prefixes,
                    *(
                        tuple(part for part in prefix if part)
                        for prefix in provided
                    ),
                )
            )
        )

    def resolved_feature_part_aliases(self) -> RuntimeMeasurementFeaturePartAliases:
        """Return static and provider-supplied direct feature-part aliases."""
        provider = self.feature_part_aliases_provider
        provided = {} if provider is None else provider()
        return MappingProxyType(
            {
                **self.feature_part_aliases,
                **{
                    tuple(part for part in parts if part): tuple(
                        part for part in alias if part
                    )
                    for parts, alias in provided.items()
                },
            }
        )

    def feature_parts(self, parts: tuple[str, ...]) -> tuple[str, ...]:
        """Return dialect-normalized feature parts for one lookup token."""
        resolved_parts = parts
        for prefix in self.resolved_category_prefixes():
            if len(resolved_parts) > len(prefix) and resolved_parts[: len(prefix)] == prefix:
                resolved_parts = resolved_parts[len(prefix) :]
                break
        return self.resolved_feature_part_aliases().get(resolved_parts, resolved_parts)

    def alternative_feature_parts(
        self,
        parts: tuple[str, ...],
    ) -> tuple[tuple[str, ...], ...]:
        """Return ordered semantic fallback fields for one lookup token."""
        feature_parts = self.feature_parts(parts)
        provider = self.alternative_feature_part_aliases_provider
        provided_aliases = {} if provider is None else provider()
        aliases = {
            **self.alternative_feature_part_aliases,
            **{
                tuple(part for part in alias_parts if part): tuple(
                    tuple(alias_part for alias_part in alias if alias_part)
                    for alias in alternatives
                    if tuple(alias)
                )
                for alias_parts, alternatives in provided_aliases.items()
            },
        }
        return aliases.get(feature_parts, ())

    def feature_lookup(self, feature_name: str) -> "RuntimeMeasurementFeatureLookup":
        """Return the nominal lookup identity for one external feature name."""
        return RuntimeMeasurementFeatureLookup(feature_name, self)


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementFeatureAliasSpan:
    """One nominal field-name span accepted for measurement lookup."""

    parts: tuple[str, ...]

    @property
    def name(self) -> str:
        return "_".join(self.parts)

    @property
    def field_aliases(self) -> tuple[str, ...]:
        name = self.name
        if not name:
            return ()
        compact = _compact_identifier(name)
        if compact == name:
            return (name,)
        return (name, compact)


class RuntimeMeasurementLookupAliasCache(
    RegisteredProcessLocalBoundedCache[tuple[str, int, str], tuple[str, ...]]
):
    """Process-local cache for dialect-resolved measurement lookup aliases."""


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementFeatureLookup:
    """Dialect-resolved aliases for one runtime measurement feature."""

    feature_name: str
    dialect: RuntimeMeasurementLookupDialect

    @property
    def normalized_name(self) -> str:
        return normalize_runtime_identifier(self.feature_name)

    @property
    def normalized_parts(self) -> tuple[str, ...]:
        return tuple(part for part in self.normalized_name.split("_") if part)

    @property
    def normalized_segments(self) -> tuple[tuple[str, ...], ...]:
        return tuple(
            tuple(part for part in normalize_runtime_identifier(segment).split("_") if part)
            for segment in str(self.feature_name).split("_")
            if segment
        )

    @property
    def field_aliases(self) -> tuple[str, ...]:
        """Return schema-safe feature field aliases."""
        cache = RuntimeMeasurementLookupAliasCache.process_cache()
        cache_key = ("field", id(self.dialect), self.feature_name)
        cached = cache.cached_value(cache_key)
        if cached is not None:
            return cached
        aliases: list[str] = []
        for span in self.field_alias_spans:
            for alias in span.field_aliases:
                if alias and alias not in aliases:
                    aliases.append(alias)
        return cache.store_value(cache_key, tuple(aliases))

    @property
    def source_aliases(self) -> tuple[str, ...]:
        """Return source-image aliases encoded by a source-qualified feature."""
        cache = RuntimeMeasurementLookupAliasCache.process_cache()
        cache_key = ("source", id(self.dialect), self.feature_name)
        cached = cache.cached_value(cache_key)
        if cached is not None:
            return cached
        return cache.store_value(
            cache_key,
            self.compact_identifier_aliases(self.source_names),
        )

    @property
    def dialect_feature_name(self) -> str:
        return "_".join(self.dialect.feature_parts(self.normalized_parts))

    @property
    def source_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for feature_family in self.source_qualified_feature_families:
            source_name = "_".join(
                self.dialect_feature_parts[len(feature_family) :]
            )
            if source_name and source_name not in names:
                names.append(source_name)
        return tuple(names)

    @property
    def source_qualified_field_names(self) -> tuple[str, ...]:
        return self.compact_identifier_aliases(
            "_".join(feature_family)
            for feature_family in self.source_qualified_feature_families
        )

    def compact_identifier_aliases(self, names: Iterable[str]) -> tuple[str, ...]:
        """Return ordered normalized and compact aliases for non-empty names."""
        aliases: list[str] = []
        for name in names:
            for alias in (name, _compact_identifier(name)):
                if alias and alias not in aliases:
                    aliases.append(alias)
        return tuple(aliases)

    @property
    def source_qualified_feature_families(self) -> tuple[tuple[str, ...], ...]:
        dialect_feature_parts = self.dialect_feature_parts
        families: list[tuple[str, ...]] = []
        for family in self.dialect.resolved_source_qualified_feature_families():
            if len(dialect_feature_parts) <= len(family):
                continue
            if dialect_feature_parts[: len(family)] != family:
                continue
            families.append(family)
        return tuple(families)

    @property
    def dialect_feature_parts(self) -> tuple[str, ...]:
        return self.dialect.feature_parts(self.normalized_parts)

    @property
    def alternative_feature_parts(self) -> tuple[tuple[str, ...], ...]:
        return self.dialect.alternative_feature_parts(self.normalized_parts)

    @property
    def field_alias_spans(self) -> tuple[RuntimeMeasurementFeatureAliasSpan, ...]:
        """Return nominal field-name spans from most specific to broadest."""
        spans: list[RuntimeMeasurementFeatureAliasSpan] = []
        for parts in (
            self.normalized_parts,
            self.dialect_feature_parts,
            *self.alternative_feature_parts,
            *self.source_qualified_feature_families,
            self.unqualified_feature_parts,
            self.metric_family_parts,
        ):
            if not parts:
                continue
            span = RuntimeMeasurementFeatureAliasSpan(parts)
            if span not in spans:
                spans.append(span)
        return tuple(spans)

    @property
    def unqualified_feature_parts(self) -> tuple[str, ...]:
        """Return the feature with an external measurement category removed."""
        segments = self.normalized_segments
        if len(segments) < 2:
            return ()
        return tuple(part for segment in segments[1:] for part in segment)

    @property
    def metric_family_parts(self) -> tuple[str, ...]:
        """Return the feature metric without category or terminal qualifier."""
        segments = self.normalized_segments
        if len(segments) < 3:
            return ()
        return segments[1]

    def query_object_name(self, object_name: str | None) -> str | None:
        """Return the effective row object constraint for this feature."""
        return self.dialect.object_domain_policy.query_object_name(self, object_name)


DEFAULT_RUNTIME_MEASUREMENT_LOOKUP_DIALECT = RuntimeMeasurementLookupDialect()
_CURRENT_RUNTIME_MEASUREMENT_LOOKUP_DIALECT: ContextVar[RuntimeMeasurementLookupDialect] = (
    ContextVar(
        "current_runtime_measurement_lookup_dialect",
        default=DEFAULT_RUNTIME_MEASUREMENT_LOOKUP_DIALECT,
    )
)


class RuntimeMeasurementLookupDialectReference(ABC):
    """Nominal reference to a runtime measurement lookup dialect."""

    @abstractmethod
    def resolve(self) -> RuntimeMeasurementLookupDialect:
        """Return the concrete lookup dialect for the active runtime scope."""


@dataclass(frozen=True, slots=True)
class CurrentRuntimeMeasurementLookupDialect(RuntimeMeasurementLookupDialectReference):
    """Reference that resolves through the active runtime lookup dialect context."""

    def resolve(self) -> RuntimeMeasurementLookupDialect:
        """Return the current runtime measurement lookup dialect."""
        return _CURRENT_RUNTIME_MEASUREMENT_LOOKUP_DIALECT.get()


CURRENT_RUNTIME_MEASUREMENT_LOOKUP_DIALECT = CurrentRuntimeMeasurementLookupDialect()
RuntimeMeasurementLookupDialectLike = (
    RuntimeMeasurementLookupDialect | RuntimeMeasurementLookupDialectReference
)


def resolve_runtime_measurement_lookup_dialect(
    dialect: RuntimeMeasurementLookupDialectLike,
) -> RuntimeMeasurementLookupDialect:
    """Resolve a nominal lookup dialect or dialect reference."""
    if isinstance(dialect, RuntimeMeasurementLookupDialect):
        return dialect
    if isinstance(dialect, RuntimeMeasurementLookupDialectReference):
        return dialect.resolve()
    raise TypeError(
        "Expected RuntimeMeasurementLookupDialect or "
        "RuntimeMeasurementLookupDialectReference, "
        f"got {type(dialect).__name__}."
    )


@contextmanager
def runtime_measurement_lookup_dialect(
    dialect: RuntimeMeasurementLookupDialect,
) -> Iterator[None]:
    """Temporarily bind the runtime measurement lookup dialect."""
    if not isinstance(dialect, RuntimeMeasurementLookupDialect):
        raise TypeError(
            "runtime_measurement_lookup_dialect requires "
            f"RuntimeMeasurementLookupDialect, got {type(dialect).__name__}."
        )
    token = _CURRENT_RUNTIME_MEASUREMENT_LOOKUP_DIALECT.set(dialect)
    try:
        yield
    finally:
        _CURRENT_RUNTIME_MEASUREMENT_LOOKUP_DIALECT.reset(token)


__all__ = declared_public_names(
    globals(),
    constant_prefixes=(
        "CURRENT_RUNTIME_MEASUREMENT_LOOKUP_DIALECT",
        "DEFAULT_RUNTIME_MEASUREMENT_LOOKUP_DIALECT",
    ),
)
