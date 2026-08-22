"""OpenHCS application identity projected through the generic ZMQ handshake."""

from openhcs import __version__ as OPENHCS_VERSION
from openhcs.runtime.zmq_application import (
    OPENHCS_ENDPOINT_APPLICATION,
)
from openhcs.runtime.zmq_execution_server import ZMQExecutionServer


def test_execution_server_advertises_the_openhcs_version_declaration() -> None:
    server = ZMQExecutionServer()

    assert server._create_pong_response().application == OPENHCS_ENDPOINT_APPLICATION
    assert OPENHCS_ENDPOINT_APPLICATION.identifier == "openhcs"
    assert OPENHCS_ENDPOINT_APPLICATION.version == OPENHCS_VERSION


def test_missing_application_identity_is_not_a_match() -> None:
    compatibility = OPENHCS_ENDPOINT_APPLICATION.compatibility_with(None)

    assert not compatibility.matches
    assert compatibility.observed_version_label == "not reported"
