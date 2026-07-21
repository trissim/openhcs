"""Generic runtime artifact addresses carried by progress events."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from openhcs.core.runtime_stores import RuntimeArtifactAddress, StoredRuntimeValue


RUNTIME_ARTIFACTS_CONTEXT_KEY = "runtime_artifacts"


class RuntimeArtifactPayloadError(ValueError):
    """Raised when runtime artifact progress context is malformed."""


@dataclass(frozen=True, slots=True)
class RuntimeArtifactProgressPayload:
    """Transport addresses for runtime values written by one step event."""

    addresses: tuple[RuntimeArtifactAddress, ...]

    @classmethod
    def from_records(
        cls,
        records: Sequence[StoredRuntimeValue],
    ) -> "RuntimeArtifactProgressPayload | None":
        addresses = tuple(RuntimeArtifactAddress.from_record(record) for record in records)
        return None if not addresses else cls(addresses=addresses)

    @classmethod
    def from_context(
        cls,
        context: Mapping[str, Any] | None,
    ) -> "RuntimeArtifactProgressPayload | None":
        if not context or RUNTIME_ARTIFACTS_CONTEXT_KEY not in context:
            return None
        data = context[RUNTIME_ARTIFACTS_CONTEXT_KEY]
        if not isinstance(data, Mapping):
            raise RuntimeArtifactPayloadError(
                f"{RUNTIME_ARTIFACTS_CONTEXT_KEY!r} must be a mapping."
            )
        raw_addresses = data.get("addresses")
        if not isinstance(raw_addresses, Sequence) or isinstance(
            raw_addresses,
            (str, bytes, bytearray),
        ):
            raise RuntimeArtifactPayloadError(
                "Runtime artifact payload field 'addresses' must be a sequence."
            )
        try:
            return cls(
                addresses=tuple(
                    RuntimeArtifactAddress.from_dict(address)
                    for address in raw_addresses
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeArtifactPayloadError(
                f"Malformed runtime artifact address: {error}"
            ) from error

    def to_context(self) -> dict[str, Any]:
        return {
            RUNTIME_ARTIFACTS_CONTEXT_KEY: {
                "addresses": [address.to_dict() for address in self.addresses],
            }
        }


def runtime_artifact_context_for_records(
    records: Sequence[StoredRuntimeValue],
) -> dict[str, Any] | None:
    """Return a progress context for every runtime artifact observation."""

    payload = RuntimeArtifactProgressPayload.from_records(records)
    return None if payload is None else payload.to_context()
