"""Source-plane identity alignment for runtime payload sequences."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from openhcs.core.runtime_values import (
    ImagePayloadMetadata,
    image_payload_metadata,
)
from openhcs.core.source_image_provenance import SourceImageProvenanceIdentity
from openhcs.core.source_matching import (
    SourceImageSetIdentity,
    SourceImageSetIdentityCompatibility,
    SourceImageSetIdentityPairPredicate,
    SourceImageSetIdentityPolicy,
)


SourcePlaneIdentitySequence = tuple[frozenset[SourceImageSetIdentity], ...]
SourcePayloadPlaneIdentitySequenceCacheValue = SourcePlaneIdentitySequence
_source_payload_plane_identity_sequence_cache_size = 65536


@dataclass(frozen=True, slots=True)
class SourcePayloadPlaneIdentitySequenceCacheKey:
    """Stable identity key for cached per-plane source identity projection."""

    source_provenance_identity: SourceImageProvenanceIdentity
    policy: SourceImageSetIdentityPolicy


_source_payload_plane_identity_sequence_cache: OrderedDict[
    SourcePayloadPlaneIdentitySequenceCacheKey,
    SourcePayloadPlaneIdentitySequenceCacheValue,
] = OrderedDict()


def _cached_source_payload_plane_identity_sequence(
    key: SourcePayloadPlaneIdentitySequenceCacheKey,
) -> SourcePayloadPlaneIdentitySequenceCacheValue | None:
    try:
        value = _source_payload_plane_identity_sequence_cache[key]
    except KeyError:
        return None
    _source_payload_plane_identity_sequence_cache.move_to_end(key)
    return value


def _store_source_payload_plane_identity_sequence(
    key: SourcePayloadPlaneIdentitySequenceCacheKey,
    value: SourcePayloadPlaneIdentitySequenceCacheValue,
) -> SourcePayloadPlaneIdentitySequenceCacheValue:
    _source_payload_plane_identity_sequence_cache[key] = value
    _source_payload_plane_identity_sequence_cache.move_to_end(key)
    if (
        len(_source_payload_plane_identity_sequence_cache)
        > _source_payload_plane_identity_sequence_cache_size
    ):
        _source_payload_plane_identity_sequence_cache.popitem(last=False)
    return value


@dataclass(frozen=True, slots=True)
class SourcePayloadPlaneIdentity:
    """Source image-set identities represented by one runtime payload plane."""

    metadata: ImagePayloadMetadata
    policy: SourceImageSetIdentityPolicy

    @classmethod
    def from_payload(
        cls,
        payload: Any,
        policy: SourceImageSetIdentityPolicy,
    ) -> "SourcePayloadPlaneIdentity":
        return cls(image_payload_metadata(payload), policy)

    def identities(self) -> frozenset[SourceImageSetIdentity]:
        if not self.metadata.has_values:
            return frozenset()
        source_component_metadata = self.metadata.source_component_metadata
        if source_component_metadata is None:
            source_component_metadata = {}
        fallback_source_path = self.metadata.source_path
        if fallback_source_path is None:
            fallback_source_path = ""
        identity = SourceImageSetIdentity.from_metadata(
            source_component_metadata,
            fallback_source_path=fallback_source_path,
            policy=self.policy,
        )
        if identity.components == (("source_path", ""),):
            return frozenset()
        return frozenset((identity,))


@dataclass(frozen=True, slots=True)
class SourcePayloadPlaneIdentitySequence:
    """Per-plane source identities for stack-like or already-sliced payloads."""

    payload: Any
    policy: SourceImageSetIdentityPolicy

    def identities(self) -> SourcePlaneIdentitySequence:
        metadata = image_payload_metadata(self.payload)
        if metadata.source_image_provenance_planes.count == 0:
            return ()
        cache_key = SourcePayloadPlaneIdentitySequenceCacheKey(
            metadata.source_provenance.equality_identity,
            self.policy,
        )
        cached = _cached_source_payload_plane_identity_sequence(cache_key)
        if cached is not None:
            return cached
        return _store_source_payload_plane_identity_sequence(
            cache_key,
            tuple(
                SourcePayloadPlaneIdentity(
                    metadata.for_source_plane(index),
                    self.policy,
                ).identities()
                for index in range(metadata.source_image_provenance_planes.count)
            ),
        )

    @property
    def has_identity(self) -> bool:
        return any(self.identities())

    @classmethod
    def from_payloads(
        cls,
        payloads: Sequence[Any],
        policy: SourceImageSetIdentityPolicy,
    ) -> SourcePlaneIdentitySequence:
        return tuple(
            SourcePayloadPlaneIdentity.from_payload(payload, policy).identities()
            for payload in payloads
        )


@dataclass(frozen=True, slots=True)
class SourcePlaneIdentitySequenceAlignment:
    """Align target source planes to an image source-plane sequence."""

    image_identities: SourcePlaneIdentitySequence
    target_identities: SourcePlaneIdentitySequence
    identity_predicate_type: ClassVar[type[SourceImageSetIdentityPairPredicate]] = (
        SourceImageSetIdentityCompatibility
    )

    def target_index_for_image_plane(
        self,
        image_identity: frozenset[SourceImageSetIdentity],
        *,
        used: frozenset[int] = frozenset(),
    ) -> int | None:
        if not image_identity:
            return None
        matches = tuple(
            index
            for index, target_identity in enumerate(self.target_identities)
            if (
                index not in used
                and self.identity_predicate_type.any_match(
                    target_identity,
                    image_identity,
                )
            )
        )
        if len(matches) != 1:
            return None
        return matches[0]

    def target_indexes_for_image_planes(self) -> tuple[int, ...] | None:
        indexes: list[int] = []
        used: set[int] = set()
        for image_identity in self.image_identities:
            match = self.target_index_for_image_plane(
                image_identity,
                used=frozenset(used),
            )
            if match is None:
                return None
            used.add(match)
            indexes.append(match)
        return tuple(indexes)
