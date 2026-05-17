#!/usr/bin/env python3
"""
Remote Orchestrator - Client for OpenHCS Execution Server

DEPRECATED: This class is now a thin wrapper around ZMQExecutionClient.
For new code, use ZMQExecutionClient directly from openhcs.runtime.zmq_execution_client.

This wrapper is maintained for backward compatibility with existing code.
"""

import logging
import warnings
from typing import Any, Dict, List, Optional
from openhcs.runtime.zmq_execution_client import (
    OpenHCSExecutionSubmission,
    ZMQExecutionClient,
)

logger = logging.getLogger(__name__)


class RemoteOrchestrator:
    """
    Client-side orchestrator for remote pipeline execution.

    DEPRECATED: This class is now a thin wrapper around ZMQExecutionClient.
    For new code, use ZMQExecutionClient directly:

    ```python
    from openhcs.runtime.zmq_execution_client import OpenHCSExecutionSubmission, ZMQExecutionClient

    client = ZMQExecutionClient(host='remote-server', port=7777, persistent=True)
    client.connect()
    response = client.execute_pipeline(
        OpenHCSExecutionSubmission(
            plate_id=plate_id,
            pipeline_steps=pipeline_steps,
            global_config=global_config,
            pipeline_config=pipeline_config,
        )
    )
    ```

    This wrapper maintains backward compatibility but delegates all work to
    ZMQExecutionClient, which provides additional features:
    - Progress callbacks
    - Graceful cancellation
    - Multi-instance support
    - Automatic server spawning
    """

    def __init__(self, server_host: str, server_port: int = 7777):
        """
        Initialize remote orchestrator.

        Args:
            server_host: Execution server hostname/IP
            server_port: Execution server port (control port will be port + 1000)
        """
        warnings.warn(
            "RemoteOrchestrator is deprecated. Use ZMQExecutionClient directly for new code. "
            "See openhcs.runtime.zmq_execution_client.ZMQExecutionClient",
            DeprecationWarning,
            stacklevel=2
        )

        self.server_host = server_host
        self.server_port = server_port

        # Create ZMQExecutionClient (persistent mode for remote servers)
        self.client = ZMQExecutionClient(
            host=server_host,
            port=server_port,
            persistent=True  # Remote servers should be persistent
        )

        logger.info(f"RemoteOrchestrator configured for {server_host}:{server_port}")
        logger.info("  (Using ZMQExecutionClient internally)")

    def _connect(self):
        """
        Establish connection to server.

        DEPRECATED: Use client.connect() directly.
        """
        if not self.client.is_connected():
            self.client.connect()

    def _disconnect(self):
        """
        Close connection.

        DEPRECATED: Use client.disconnect() directly.
        """
        if self.client.is_connected():
            self.client.disconnect()

    def run_remote_pipeline(
        self,
        plate_id: int,
        pipeline_steps: List[Any],
        global_config: Any,
        pipeline_config: Any = None,
        viewer_host: str = 'localhost',
        viewer_port: int = 5555
    ) -> Dict[str, Any]:
        """
        Execute pipeline on remote server.

        Args:
            plate_id: OMERO plate ID or plate path
            pipeline_steps: List of FunctionStep objects
            global_config: GlobalPipelineConfig instance
            pipeline_config: Optional PipelineConfig instance
            viewer_host: Host for result streaming (this machine) - DEPRECATED, not used
            viewer_port: Port for result streaming - DEPRECATED, not used

        Returns:
            Response from server with execution_id and results
        """
        # Ensure connected
        self._connect()

        # Delegate to ZMQExecutionClient
        logger.info(f"Executing pipeline on remote server {self.server_host}:{self.server_port}...")

        response = self.client.execute_pipeline(
            OpenHCSExecutionSubmission(
                plate_id=str(plate_id),
                pipeline_steps=pipeline_steps,
                global_config=global_config,
                pipeline_config=pipeline_config,
            )
        )

        # Convert response format for backward compatibility
        # ZMQExecutionClient returns {'status': 'complete', 'execution_id': ..., 'results': ...}
        # Old format expected {'status': 'ok', 'message': ..., 'execution_id': ...}
        if response.get('status') == 'complete':
            return {
                'status': 'ok',
                'message': 'Pipeline execution completed',
                'execution_id': response.get('execution_id'),
                'results': response.get('results')
            }
        elif response.get('status') == 'error':
            return {
                'status': 'error',
                'message': response.get('message', 'Unknown error'),
                'execution_id': response.get('execution_id')
            }
        else:
            return response

    def get_status(self, execution_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Query execution status.

        Args:
            execution_id: Specific execution ID, or None for server status

        Returns:
            Status response
        """
        self._connect()
        return self.client.get_status(execution_id)

    def ping(self) -> bool:
        """
        Ping server to check if alive.

        Returns:
            True if server responds
        """
        try:
            # Don't auto-connect for ping (just check if server is responsive)
            return self.client.ping()
        except Exception as e:
            logger.error(f"Ping failed: {e}")
            return False

    def get_server_info(self) -> Dict[str, Any]:
        """
        Get detailed server information including worker processes.

        Returns:
            Server info with workers, active executions, uptime, etc.
        """
        try:
            return self.client.get_server_info()
        except Exception as e:
            logger.error(f"Get server info failed: {e}")
            return {'status': 'error', 'message': str(e)}

    def __enter__(self):
        """Context manager entry."""
        self._connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self._disconnect()
