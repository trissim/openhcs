"""OpenHCS application identity at generic endpoint boundaries."""

from zmqruntime import EndpointApplication

from openhcs import __version__ as OPENHCS_VERSION

OPENHCS_ENDPOINT_APPLICATION = EndpointApplication(
    identifier="openhcs",
    version=OPENHCS_VERSION,
)
