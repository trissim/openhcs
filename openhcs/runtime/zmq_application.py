"""OpenHCS application identity at the generic ZMQ endpoint boundary."""

from __future__ import annotations

from dataclasses import dataclass

from zmqruntime import EndpointApplication, PongResponse

from openhcs import __version__ as OPENHCS_VERSION


OPENHCS_ENDPOINT_APPLICATION = EndpointApplication(
    identifier="openhcs",
    version=OPENHCS_VERSION,
)


@dataclass(frozen=True, slots=True)
class OpenHCSEndpointCompatibility:
    """Comparison derived from the local declaration and one live handshake."""

    expected: EndpointApplication
    observed: EndpointApplication | None

    @classmethod
    def from_handshake(cls, handshake: PongResponse) -> "OpenHCSEndpointCompatibility":
        return cls(
            expected=OPENHCS_ENDPOINT_APPLICATION,
            observed=handshake.application,
        )

    @property
    def matches(self) -> bool:
        return self.observed == self.expected

    @property
    def observed_version_label(self) -> str:
        if self.observed is None:
            return "not reported"
        return self.observed.version
