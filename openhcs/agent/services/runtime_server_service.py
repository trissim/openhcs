"""Read-only runtime server discovery for OpenHCS agent integrations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from openhcs.agent.dto.common import AgentError, JsonObject, SCHEMA_VERSION
from openhcs.agent.dto.execution import (
    ExecutionConnectionSpec,
    RuntimeExecutionStatus,
    RuntimeServerInfo,
    RuntimeServerScanResult,
    runtime_execution_status_error,
    runtime_execution_status_from_response,
    unreachable_runtime_server_info,
)
from openhcs.runtime.zmq_config import OPENHCS_ZMQ_CONFIG
from openhcs.runtime.zmq_execution_client import ZMQExecutionClient


class RuntimeServerGatewayABC(ABC):
    """Transport boundary for querying OpenHCS runtime servers."""

    @abstractmethod
    def server_info(self, connection: ExecutionConnectionSpec) -> JsonObject:
        raise NotImplementedError

    @abstractmethod
    def execution_status(
        self,
        connection: ExecutionConnectionSpec,
        execution_id: str | None = None,
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

    def server_info(self, connection: ExecutionConnectionSpec) -> JsonObject:
        return dict(self._client(connection).get_server_info())

    def execution_status(
        self,
        connection: ExecutionConnectionSpec,
        execution_id: str | None = None,
    ) -> JsonObject:
        return dict(self._client(connection).get_status(execution_id))

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
    ) -> RuntimeServerInfo:
        connection = ExecutionConnectionSpec(host, port, transport_mode, persistent)
        try:
            response = self._gateway.server_info(connection)
        except Exception as exc:
            return unreachable_runtime_server_info(
                connection=connection,
                error=AgentError.from_exception("runtime_server_unreachable", exc),
            )
        return RuntimeServerInfo.from_response(
            connection=connection,
            response=response,
        )

    def scan(
        self,
        *,
        ports: tuple[int, ...] | None = None,
        host: str = "localhost",
        transport_mode: str | None = None,
        timeout_ms: int = 200,
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

    def execution_status(
        self,
        *,
        execution_id: str | None = None,
        host: str = "localhost",
        port: int | None = None,
        transport_mode: str | None = None,
        persistent: bool = True,
    ) -> RuntimeExecutionStatus:
        connection = ExecutionConnectionSpec(host, port, transport_mode, persistent)
        try:
            response = self._gateway.execution_status(
                connection,
                execution_id,
            )
        except Exception as exc:
            return runtime_execution_status_error(
                connection=connection,
                execution_id=execution_id,
                error=AgentError.from_exception("runtime_execution_status_error", exc),
            )
        return runtime_execution_status_from_response(
            connection=connection,
            execution_id=execution_id,
            response=response,
        )

    @staticmethod
    def _scan_ports(ports: tuple[int, ...] | None) -> tuple[int, ...]:
        if ports is None:
            return (OPENHCS_ZMQ_CONFIG.default_port,)
        return tuple(dict.fromkeys(int(port) for port in ports))
