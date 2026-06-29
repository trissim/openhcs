"""Read-only runtime server discovery for OpenHCS agent integrations."""

from __future__ import annotations

from abc import ABC, abstractmethod

from openhcs.agent.dto.common import AgentError, JsonObject, SCHEMA_VERSION
from openhcs.agent.dto.execution import (
    DEFAULT_EXECUTION_STATUS_TIMEOUT_MS,
    DEFAULT_RUNTIME_SERVER_INFO_TIMEOUT_MS,
    DEFAULT_RUNTIME_SERVER_SCAN_TIMEOUT_MS,
    ExecutionConnectionSpec,
    RuntimeServerExecutionStatusRequest,
    RuntimeServerInfoRequest,
    RuntimeExecutionStatus,
    RuntimeServerPayload,
    RuntimeServerInfo,
    RuntimeServerScanRequest,
    RuntimeServerScanResult,
    runtime_execution_status_error,
    runtime_execution_status_from_response,
    unreachable_runtime_server_info,
)
from openhcs.runtime.zmq_config import OPENHCS_ZMQ_CONFIG
from openhcs.runtime.zmq_execution_client import ZMQExecutionClient
from zmqruntime.execution.server import ExecutionServer
from zmqruntime.messages import MessageFields


RUNTIME_SERVER_KIND_HINT = (
    "Use openhcs_scan_runtime_servers to discover endpoints. Viewer ports should "
    "be queried with viewer tools, and UI bridge ports should be queried with UI "
    "bridge tools."
)


class RuntimeServerWrongKindError(ValueError):
    """Raised when a pong endpoint is not an execution runtime server."""

    def __init__(
        self,
        *,
        server_name: str | None,
        server_type: str | None,
        port: int | None,
    ) -> None:
        self.server_name = server_name
        self.server_type = server_type
        self.port = port
        super().__init__(
            f"Port {port} responded as {server_name or 'unknown server'} "
            f"with server_type={server_type or 'unknown'}, not a ZMQ execution "
            "runtime server."
        )


class RuntimeServerPongClassifier:
    """Classify control-pong payloads returned by OpenHCS ZMQ endpoints."""

    @classmethod
    def require_execution_server(
        cls,
        response: JsonObject,
        *,
        port: int | None,
    ) -> None:
        server_name = cls.server_name(response)
        server_type = cls.server_type(response)
        if server_type != ExecutionServer.server_type():
            raise RuntimeServerWrongKindError(
                server_name=server_name,
                server_type=server_type,
                port=port,
            )

    @staticmethod
    def server_name(response: JsonObject) -> str | None:
        value = response.get(MessageFields.SERVER)
        if value is None:
            return None
        return str(value)

    @staticmethod
    def server_type(response: JsonObject) -> str | None:
        value = response.get(MessageFields.SERVER_TYPE)
        if value is None:
            return None
        return str(value)


class RuntimeServerGatewayABC(ABC):
    """Transport boundary for querying OpenHCS runtime servers."""

    @abstractmethod
    def server_info(
        self,
        connection: ExecutionConnectionSpec,
        *,
        timeout_ms: int,
    ) -> JsonObject:
        raise NotImplementedError

    @abstractmethod
    def execution_status(
        self,
        connection: ExecutionConnectionSpec,
        execution_id: str | None = None,
        *,
        timeout_ms: int,
    ) -> JsonObject:
        raise NotImplementedError

    @abstractmethod
    def scan(
        self,
        *,
        host: str,
        ports: tuple[int, ...],
        transport_mode: str | None,
        timeout_ms: int,
    ) -> tuple[JsonObject, ...]:
        raise NotImplementedError


class ZMQRuntimeServerGateway(RuntimeServerGatewayABC):
    """Runtime server gateway backed by the existing ZMQ execution client."""

    def server_info(
        self,
        connection: ExecutionConnectionSpec,
        *,
        timeout_ms: int,
    ) -> JsonObject:
        port = connection.require_port("Runtime server info")
        responses = self.scan(
            host=connection.host,
            ports=(port,),
            transport_mode=connection.transport_mode,
            timeout_ms=timeout_ms,
        )
        if not responses:
            raise TimeoutError(
                f"No OpenHCS ZMQ endpoint responded on port {port} within {timeout_ms}ms."
            )
        return responses[0]

    def execution_status(
        self,
        connection: ExecutionConnectionSpec,
        execution_id: str | None = None,
        *,
        timeout_ms: int,
    ) -> JsonObject:
        return dict(
            self._client(connection).get_status(
                execution_id,
                timeout_ms=timeout_ms,
            )
        )

    def scan(
        self,
        *,
        host: str,
        ports: tuple[int, ...],
        transport_mode: str | None,
        timeout_ms: int,
    ) -> tuple[JsonObject, ...]:
        return tuple(
            dict(server)
            for server in ZMQExecutionClient.scan_servers(
                ports,
                host=host,
                timeout_ms=timeout_ms,
                transport_mode=transport_mode,
                config=OPENHCS_ZMQ_CONFIG,
            )
        )

    @staticmethod
    def _client(connection: ExecutionConnectionSpec) -> ZMQExecutionClient:
        return ZMQExecutionClient(**connection.zmq_client_kwargs())


class RuntimeServerService:
    """Expose read-only state from running OpenHCS ZMQ execution servers."""

    def __init__(
        self,
        gateway: RuntimeServerGatewayABC | None = None,
    ) -> None:
        self._gateway = gateway or ZMQRuntimeServerGateway()

    def server_info(
        self,
        *,
        host: str = "localhost",
        port: int | None = None,
        transport_mode: str | None = None,
        persistent: bool = True,
        timeout_ms: int = DEFAULT_RUNTIME_SERVER_INFO_TIMEOUT_MS,
    ) -> RuntimeServerInfo:
        connection = ExecutionConnectionSpec(host, port, transport_mode, persistent)
        try:
            response = self._gateway.server_info(
                connection,
                timeout_ms=timeout_ms,
            )
            RuntimeServerPongClassifier.require_execution_server(
                response,
                port=RuntimeServerPayload(response).optional_int(
                    "port",
                    protocol_default=port,
                ),
            )
        except RuntimeServerWrongKindError as exc:
            return unreachable_runtime_server_info(
                connection=connection,
                error=AgentError.from_exception(
                    "runtime_server_wrong_type",
                    exc,
                    hint=RUNTIME_SERVER_KIND_HINT,
                ),
            )
        except Exception as exc:
            return unreachable_runtime_server_info(
                connection=connection,
                error=AgentError.from_exception(
                    "runtime_server_unreachable",
                    exc,
                    hint=RUNTIME_SERVER_KIND_HINT,
                ),
            )
        return RuntimeServerInfo.from_response(
            connection=connection,
            response=response,
        )

    def server_info_from_request(
        self,
        request: RuntimeServerInfoRequest,
    ) -> RuntimeServerInfo:
        return self.server_info(
            host=request.connection.host,
            port=request.connection.port,
            transport_mode=request.connection.transport_mode,
            persistent=request.connection.persistent,
            timeout_ms=request.timeout_ms,
        )

    def scan(
        self,
        *,
        ports: tuple[int, ...] | None = None,
        host: str = "localhost",
        transport_mode: str | None = None,
        timeout_ms: int = DEFAULT_RUNTIME_SERVER_SCAN_TIMEOUT_MS,
    ) -> RuntimeServerScanResult:
        scanned_ports = self._scan_ports(ports)
        servers = tuple(
            RuntimeServerInfo.from_response(
                connection=ExecutionConnectionSpec(
                    host=host,
                    transport_mode=transport_mode,
                ),
                response=response,
            )
            for response in self._gateway.scan(
                host=host,
                ports=scanned_ports,
                transport_mode=transport_mode,
                timeout_ms=timeout_ms,
            )
        )
        return RuntimeServerScanResult(
            schema_version=SCHEMA_VERSION,
            connection=ExecutionConnectionSpec(
                host=host,
                transport_mode=transport_mode,
            ),
            ports=scanned_ports,
            timeout_ms=timeout_ms,
            servers=servers,
        )

    def scan_from_request(
        self,
        request: RuntimeServerScanRequest,
    ) -> RuntimeServerScanResult:
        return self.scan(
            ports=request.ports,
            host=request.host,
            transport_mode=request.transport_mode,
            timeout_ms=request.timeout_ms,
        )

    def execution_status(
        self,
        *,
        execution_id: str | None = None,
        host: str = "localhost",
        port: int | None = None,
        transport_mode: str | None = None,
        persistent: bool = True,
        timeout_ms: int = DEFAULT_EXECUTION_STATUS_TIMEOUT_MS,
    ) -> RuntimeExecutionStatus:
        connection = ExecutionConnectionSpec(host, port, transport_mode, persistent)
        try:
            response = self._gateway.execution_status(
                connection,
                execution_id,
                timeout_ms=timeout_ms,
            )
        except Exception as exc:
            return runtime_execution_status_error(
                connection=connection,
                execution_id=execution_id,
                error=AgentError.from_exception(
                    "runtime_execution_status_error",
                    exc,
                    hint=(
                        "Check that the port is an OpenHCS execution server and "
                        "increase timeout_ms only if the server is known to be busy."
                    ),
                ),
            )
        return runtime_execution_status_from_response(
            connection=connection,
            execution_id=execution_id,
            response=response,
        )

    def execution_status_from_request(
        self,
        request: RuntimeServerExecutionStatusRequest,
    ) -> RuntimeExecutionStatus:
        return self.execution_status(
            execution_id=request.execution_id,
            host=request.connection.host,
            port=request.connection.port,
            transport_mode=request.connection.transport_mode,
            persistent=request.connection.persistent,
            timeout_ms=request.timeout_ms,
        )

    @staticmethod
    def _scan_ports(ports: tuple[int, ...] | None) -> tuple[int, ...]:
        if ports is None:
            return (OPENHCS_ZMQ_CONFIG.default_port,)
        return tuple(dict.fromkeys(int(port) for port in ports))
