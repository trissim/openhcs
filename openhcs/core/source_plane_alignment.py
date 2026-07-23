"""Source-plane identity alignment for runtime payload sequences."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from openhcs.core.runtime_image_values import (
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
        return self.metadata.source_provenance.image_set_identities(self.policy)


@dataclass(frozen=True, slots=True)
class SourcePayloadPlaneIdentitySequence:
    """Per-plane source identities for stack-like or already-sliced payloads."""

    payload: Any
    policy: SourceImageSetIdentityPolicy

    def identities(self) -> SourcePlaneIdentitySequence:
        metadata = image_payload_metadata(self.payload)
        cache_key = SourcePayloadPlaneIdentitySequenceCacheKey(
            metadata.source_provenance.equality_identity,
            self.policy,
        )
        cached = _cached_source_payload_plane_identity_sequence(cache_key)
        if cached is not None:
            return cached
        return _store_source_payload_plane_identity_sequence(
            cache_key,
            metadata.source_provenance.image_set_plane_identities(self.policy),
        )

    def runtime_axis_identities(self) -> SourcePlaneIdentitySequence:
        """Return the image-set identity axis carried by this payload."""
        metadata = image_payload_metadata(self.payload)
        return metadata.source_provenance.image_set_axis(self.policy)

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

    def target_indexes_for_exact_axis(self) -> tuple[int, ...] | None:
        """Return a bijection only when both sequences describe one exact axis."""
        if (
            not self.image_identities
            or len(self.image_identities) != len(self.target_identities)
        ):
            return None
        return self.target_indexes_for_image_planes()

    @classmethod
    def unaligned_axis_indexes(
        cls,
        axes: tuple[SourcePlaneIdentitySequence, ...],
    ) -> tuple[int, ...]:
        """Return indexes that do not share one known image-set axis."""
        known = tuple(bool(axis) for axis in axes)
        if not any(known):
            return ()
        if not all(known):
            return tuple(index for index, present in enumerate(known) if not present)
        reference = axes[0]
        return tuple(
            index
            for index, target in enumerate(axes[1:], start=1)
            if cls(reference, target).target_indexes_for_exact_axis() is None
        )
