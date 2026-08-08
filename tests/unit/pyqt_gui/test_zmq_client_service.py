import asyncio
import time

from openhcs.pyqt_gui.widgets.shared.services.zmq_client_service import ZMQClientService
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
