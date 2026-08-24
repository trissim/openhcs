import asyncio
import threading
import time

import pytest
from zmqruntime import (
    EndpointApplication,
    EndpointApplicationCompatibility,
    EndpointApplicationCompatibilityError,
    EndpointConnectionCancelledError,
    EndpointShutdownResult,
    EndpointStartupPhase,
    EndpointStartupStatus,
    TransportEndpoint,
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
        self.endpoint = TransportEndpoint(
            host=config.client_host,
            port=config.default_port,
            transport_mode=config.transport_mode,
        )
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
        self.connected = True
        return True

    def is_connected(self) -> bool:
        return self.connected

    def connect_existing(self, timeout: float) -> bool:
        del timeout
        self.connect_existing_calls += 1
        self.connected = type(self).existing_endpoint_available
        return self.connected

    def new_connection_attempt(self):
        client = self

        class FakeConnectionAttempt:
            def __init__(self) -> None:
                self.cancelled = threading.Event()

            def cancel(self) -> None:
                self.cancelled.set()

            def connect(self, policy, timeout: float) -> bool:
                time.sleep(0.02)
                if self.cancelled.is_set():
                    return False
                return policy.connect(client, timeout)

        return FakeConnectionAttempt()

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


def test_cancelled_lock_waiter_releases_late_acquisition() -> None:
    service = ZMQClientService(config=OpenHCSZMQConfig(default_port=7777))

    async def run_case() -> None:
        service._client_lock.acquire()
        waiter = asyncio.create_task(service.connect())
        await asyncio.sleep(0.01)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        service._client_lock.release()
        await asyncio.sleep(0.01)
        assert service._client_lock.acquire(timeout=0.1)
        service._client_lock.release()

    asyncio.run(run_case())


def test_disconnect_sync_cancels_an_in_progress_connection_attempt(
    monkeypatch,
) -> None:
    import openhcs.runtime.zmq_execution_client as client_module

    class BlockingExecutionClient(SlowFakeExecutionClient):
        attempt_started = threading.Event()
        attempt_cancelled = threading.Event()

        def new_connection_attempt(self):
            class BlockingConnectionAttempt:
                def cancel(self) -> None:
                    BlockingExecutionClient.attempt_cancelled.set()

                def connect(self, policy, timeout: float) -> bool:
                    del policy, timeout
                    BlockingExecutionClient.attempt_started.set()
                    if not BlockingExecutionClient.attempt_cancelled.wait(timeout=1):
                        raise AssertionError(
                            "Connection attempt was not cancelled by client teardown"
                        )
                    raise EndpointConnectionCancelledError("cancelled")

            return BlockingConnectionAttempt()

    BlockingExecutionClient.instances = []
    BlockingExecutionClient.attempt_started.clear()
    BlockingExecutionClient.attempt_cancelled.clear()
    monkeypatch.setattr(
        client_module,
        "ZMQExecutionClient",
        BlockingExecutionClient,
    )
    service = ZMQClientService(config=OpenHCSZMQConfig(default_port=7777))
    connection_errors = []

    def connect() -> None:
        try:
            asyncio.run(service.connect())
        except Exception as error:
            connection_errors.append(error)

    connection_thread = threading.Thread(target=connect)
    connection_thread.start()
    assert BlockingExecutionClient.attempt_started.wait(timeout=1)

    service.disconnect_sync()
    connection_thread.join(timeout=1)

    assert not connection_thread.is_alive()
    assert len(connection_errors) == 1
    assert isinstance(connection_errors[0], EndpointConnectionCancelledError)
    assert not service.has_client()
    assert BlockingExecutionClient.instances[-1].disconnect_calls == 1


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


def test_foreign_endpoint_is_reported_but_not_admitted(monkeypatch) -> None:
    import openhcs.runtime.zmq_execution_client as client_module

    class ForeignExecutionClient(SlowFakeExecutionClient):
        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            self.endpoint_application = EndpointApplication(
                identifier="foreign",
                version=OPENHCS_VERSION,
            )

    monkeypatch.setattr(client_module, "ZMQExecutionClient", ForeignExecutionClient)
    observations = []
    service = ZMQClientService(
        config=OpenHCSZMQConfig(default_port=7777),
        compatibility_callback=observations.append,
    )

    with pytest.raises(EndpointApplicationCompatibilityError):
        asyncio.run(service.connect())

    client = ForeignExecutionClient.instances[-1]
    assert service.zmq_client is None
    assert not service.has_client()
    with pytest.raises(EndpointApplicationCompatibilityError):
        service.require_client()
    assert client.is_connected()
    assert observations == [client.endpoint_compatibility()]


def test_retired_client_cannot_publish_stale_startup_status(monkeypatch) -> None:
    import openhcs.runtime.zmq_execution_client as client_module

    monkeypatch.setattr(client_module, "ZMQExecutionClient", SlowFakeExecutionClient)
    observed = []
    service = ZMQClientService(
        config=OpenHCSZMQConfig(default_port=7777),
        status_callback=observed.append,
    )
    client = asyncio.run(service.connect())
    callback = client.connection_status_callback
    service.disconnect_sync()

    callback(
        EndpointStartupStatus(
            phase=EndpointStartupPhase.CONNECTED,
            message="late",
        )
    )

    assert observed == []


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


def test_version_restart_rejects_changed_compatibility_before_shutdown(
    monkeypatch,
) -> None:
    import openhcs.runtime.zmq_execution_client as client_module

    SlowFakeExecutionClient.instances = []
    shutdown_ports: list[int] = []
    monkeypatch.setattr(
        client_module,
        "ZMQExecutionClient",
        SlowFakeExecutionClient,
    )
    monkeypatch.setattr(
        SlowFakeExecutionClient,
        "shutdown_endpoint_on_port",
        lambda *, port, **_kwargs: shutdown_ports.append(port),
        raising=False,
    )
    service = ZMQClientService(config=OpenHCSZMQConfig(default_port=7777))
    client = asyncio.run(service.connect(persistent=True))
    expected = client.endpoint_compatibility()
    client.endpoint_application = EndpointApplication(
        identifier=OPENHCS_ENDPOINT_APPLICATION.identifier,
        version="changed-before-confirmation",
    )

    with pytest.raises(RuntimeError, match="compatibility changed"):
        asyncio.run(service.restart_endpoint(expected_compatibility=expected))

    assert service.zmq_client is client
    assert client.disconnect_calls == 0
    assert shutdown_ports == []


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

    other_endpoint = TransportEndpoint(
        host=client.endpoint.host,
        port=7888,
        transport_mode=client.endpoint.transport_mode,
    )
    assert service.endpoint_terminated(other_endpoint) is False
    assert service.zmq_client is client
    assert client.disconnect_calls == 0

    assert service.endpoint_terminated(client.endpoint) is True
    assert service.zmq_client is None
    assert client.disconnect_calls == 1
