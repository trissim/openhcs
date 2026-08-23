from contextlib import contextmanager

import pytest

from openhcs.agent.services.ui_bridge_service import (
    UiBridgeDescriptorProcessGoneError,
    UiBridgeDescriptorProcessIdentityError,
    UiBridgeGatewayResponseError,
    UiBridgeGatewayTimeoutError,
    UiBridgeGatewayUnavailableError,
)
from openhcs.core.streaming_config_declarations import ViewerType
from openhcs.pyqt_gui.services.ui_agent_bridge import UiCodeDocumentValidationError
from openhcs.pyqt_gui.services.ui_bridge_server import (
    UiBridgeUnsupportedOperationError,
)
from openhcs.runtime.viewer_protocol import ViewerGraphicalSessionUnavailableError


@contextmanager
def _propagate():
    yield


@pytest.mark.parametrize(
    "error",
    (
        UiBridgeGatewayUnavailableError(),
        UiBridgeGatewayResponseError(()),
        UiBridgeGatewayTimeoutError("operation", 1),
        UiBridgeDescriptorProcessGoneError(1),
        UiBridgeDescriptorProcessIdentityError(1, 1.0, 2.0),
        UiCodeDocumentValidationError(()),
        UiBridgeUnsupportedOperationError("operation"),
        ViewerGraphicalSessionUnavailableError(ViewerType.NAPARI, 1),
    ),
)
def test_typed_errors_support_context_manager_traceback_propagation(
    error: Exception,
) -> None:
    with pytest.raises(type(error)):
        with _propagate():
            raise error
