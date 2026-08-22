import asyncio
import time

from zmqruntime import (
    EndpointApplication,
    EndpointApplicationCompatibility,
    EndpointShutdownResult,
)

from openhcs import __version__ as OPENHCS_VERSION
from openhcs.pyqt_gui.widgets.shared.services.zmq_client_service import ZMQClientService
from openhcs.runtime.zmq_application import (
    OPENHCS_ENDPOINT_APPLICATION,
)
from openhcs.runtime.zmq_config import OpenHCSZMQConfig


class SlowFakeExecutionClient:
    instances = []
    existing_endpoint_available = False

    def __init__(
        self,
        *,
        config,
        persistent: bool,
        progress_callback,
        connection_status_callback=None,
    ):
        self.config = config
        self.port = config.default_port
        self.persistent = persistent
        self.progress_callback = progress_callback
        self.connection_status_callback = connection_status_callback
        self.connected = False
        self.connect_existing_calls = 0
        self.disconnect_calls = 0
        self.endpoint_application = OPENHCS_ENDPOINT_APPLICATION
        type(self).instances.append(self)

    def connect(self, timeout: float) -> bool:
        del timeout
        time.sleep(0.02)
        self.connected = True
        return True

    def is_connected(self) -> bool:
        return self.connected

    def connect_existing(self, timeout: float) -> bool:
        del timeout
        self.connect_existing_calls += 1
        self.connected = type(self).existing_endpoint_available
        return self.connected

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self.connected = False

    def endpoint_compatibility(self) -> EndpointApplicationCompatibility:
        return OPENHCS_ENDPOINT_APPLICATION.compatibility_with(
            self.endpoint_application
        )


def test_zmq_client_service_concurrent_connects_reuse_same_client(monkeypatch):
    import openhcs.runtime.zmq_execution_client as client_module

    SlowFakeExecutionClient.instances = []
    monkeypatch.setattr(
        client_module,
        "ZMQExecutionClient",
        SlowFakeExecutionClient,
    )
    config = OpenHCSZMQConfig(default_port=7777)
    service = ZMQClientService(config=config)

    def progress_callback(_event):
        return None

    async def run_case():
        first, second = await asyncio.gather(
            service.connect(progress_callback=progress_callback),
            service.connect(progress_callback=progress_callback),
        )
        return first, second

    first, second = asyncio.run(run_case())

    assert first is second
    assert first.is_connected()
    assert SlowFakeExecutionClient.instances == [first]


def test_zmq_client_service_publishes_derived_endpoint_compatibility(monkeypatch):
    import openhcs.runtime.zmq_execution_client as client_module

    SlowFakeExecutionClient.instances = []
    monkeypatch.setattr(
        client_module,
        "ZMQExecutionClient",
        SlowFakeExecutionClient,
    )
    observations = []
    service = ZMQClientService(
        config=OpenHCSZMQConfig(default_port=7777),
        compatibility_callback=observations.append,
    )

    client = asyncio.run(service.connect())

    assert observations == [client.endpoint_compatibility()]
    assert observations[0].matches
    assert observations[0].expected.version == OPENHCS_VERSION


def test_endpoint_compatibility_rejects_a_foreign_application() -> None:
    compatibility = OPENHCS_ENDPOINT_APPLICATION.compatibility_with(
        EndpointApplication(identifier="other", version=OPENHCS_VERSION)
    )

    assert not compatibility.matches


def test_zmq_client_service_restarts_the_endpoint_under_one_lifecycle_lock(
    monkeypatch,
) -> None:
    import openhcs.runtime.zmq_execution_client as client_module

    SlowFakeExecutionClient.instances = []
    shutdown_ports = []
    monkeypatch.setattr(
        client_module,
        "ZMQExecutionClient",
        SlowFakeExecutionClient,
    )
    monkeypatch.setattr(
        SlowFakeExecutionClient,
        "shutdown_endpoint_on_port",
        lambda *, port, **_kwargs: (
            shutdown_ports.append(port)
            or EndpointShutdownResult(succeeded=True, endpoint_terminated=True)
        ),
        raising=False,
    )
    service = ZMQClientService(config=OpenHCSZMQConfig(default_port=7777))
    original = asyncio.run(service.connect(persistent=True))

    replacement = asyncio.run(service.restart_endpoint())

    assert original.disconnect_calls == 1
    assert replacement is not original
    assert replacement.is_connected()
    assert replacement.persistent is True
    assert shutdown_ports == [7777]


def test_zmq_client_service_reuses_equivalent_bound_progress_callback(monkeypatch):
    import openhcs.runtime.zmq_execution_client as client_module

    class ProgressReceiver:
        def on_progress(self, _event):
            return None

    SlowFakeExecutionClient.instances = []
    monkeypatch.setattr(
        client_module,
        "ZMQExecutionClient",
        SlowFakeExecutionClient,
    )
    service = ZMQClientService(config=OpenHCSZMQConfig(default_port=7777))
    receiver = ProgressReceiver()

    async def run_case():
        first = await service.connect(progress_callback=receiver.on_progress)
        second = await service.connect(progress_callback=receiver.on_progress)
        return first, second

    first, second = asyncio.run(run_case())

    assert first is second
    assert first.disconnect_calls == 0
    assert SlowFakeExecutionClient.instances == [first]


def test_zmq_client_service_attaches_to_an_existing_endpoint(monkeypatch):
    import openhcs.runtime.zmq_execution_client as client_module

    SlowFakeExecutionClient.instances = []
    SlowFakeExecutionClient.existing_endpoint_available = True
    monkeypatch.setattr(
        client_module,
        "ZMQExecutionClient",
        SlowFakeExecutionClient,
    )
    service = ZMQClientService(config=OpenHCSZMQConfig(default_port=7777))

    client = asyncio.run(service.connect_existing())

    assert client is SlowFakeExecutionClient.instances[0]
    assert client.connect_existing_calls == 1
    assert client.is_connected()


def test_zmq_client_service_leaves_an_absent_endpoint_disconnected(monkeypatch):
    import openhcs.runtime.zmq_execution_client as client_module

    SlowFakeExecutionClient.instances = []
    SlowFakeExecutionClient.existing_endpoint_available = False
    monkeypatch.setattr(
        client_module,
        "ZMQExecutionClient",
        SlowFakeExecutionClient,
    )
    service = ZMQClientService(config=OpenHCSZMQConfig(default_port=7777))

    client = asyncio.run(service.connect_existing())

    assert client is None
    assert service.zmq_client is None
    assert SlowFakeExecutionClient.instances[0].connect_existing_calls == 1


def test_endpoint_termination_disconnects_only_its_exact_client(monkeypatch):
    import openhcs.runtime.zmq_execution_client as client_module

    SlowFakeExecutionClient.instances = []
    monkeypatch.setattr(
        client_module,
        "ZMQExecutionClient",
        SlowFakeExecutionClient,
    )
    service = ZMQClientService(config=OpenHCSZMQConfig(default_port=7777))
    client = asyncio.run(service.connect())

    assert service.endpoint_terminated(7888) is False
    assert service.zmq_client is client
    assert client.disconnect_calls == 0

    assert service.endpoint_terminated(7777) is True
    assert service.zmq_client is None
    assert client.disconnect_calls == 1


def test_endpoint_snapshot_absence_invalidates_the_exact_client(monkeypatch):
    import openhcs.runtime.zmq_execution_client as client_module

    SlowFakeExecutionClient.instances = []
    monkeypatch.setattr(
        client_module,
        "ZMQExecutionClient",
        SlowFakeExecutionClient,
    )
    service = ZMQClientService(config=OpenHCSZMQConfig(default_port=7777))
    client = asyncio.run(service.connect())

    assert service.reconcile_endpoint_presence(7777, present=True) is False
    assert service.zmq_client is client
    assert client.disconnect_calls == 0

    assert service.reconcile_endpoint_presence(7777, present=False) is True
    assert service.zmq_client is None
    assert client.disconnect_calls == 1
