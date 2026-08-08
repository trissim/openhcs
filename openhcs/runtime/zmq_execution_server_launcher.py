#!/usr/bin/env python3
"""
ZMQ Execution Server Launcher

Standalone script for spawning ZMQ execution server processes.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from zmqruntime.config import TransportMode
from zmqruntime.startup import (
    EndpointStartupPhase,
    EndpointStartupStatusWriter,
)

logger = logging.getLogger(__name__)

def main(*, execution_server_type=None, server_runner=None):
    """Main entry point for server launcher."""
    parser = argparse.ArgumentParser(description="ZMQ Execution Server Launcher")
    parser.add_argument("--port", type=int, default=None, help="Override data port")
    parser.add_argument("--host", type=str, default=None, help="Override bind host")
    parser.add_argument(
        "--persistent",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override persistent server mode",
    )
    parser.add_argument("--log-file-path", type=str, default=None, help="Path to server log file (for client discovery)")
    parser.add_argument(
        "--transport-mode",
        type=TransportMode,
        default=None,
        choices=tuple(TransportMode),
        help="Override the configured transport mode",
    )
    log_levels = logging.getLevelNamesMapping()
    parser.add_argument(
        "--log-level",
        type=str.upper,
        default="INFO",
        choices=tuple(log_levels),
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--config-source",
        type=str,
        default=None,
        help="Pycodified OpenHCSZMQConfig supplied by the spawning process",
    )
    parser.add_argument(
        "--prepare-capabilities",
        action="store_true",
        help="Prepare endpoint-owned capability caches without binding sockets",
    )
    parser.add_argument(
        "--startup-status-path",
        type=str,
        default=None,
        help="JSONL status channel owned by the spawning client",
    )

    args = parser.parse_args()
    status_reporter = EndpointStartupStatusWriter(
        None if args.startup_status_path is None else Path(args.startup_status_path)
    )
    status_reporter.emit(
        EndpointStartupPhase.STARTING_PROCESS,
        "Execution server process started",
    )

    # Configure logging with the specified level
    # CRITICAL: Must force reconfigure root logger - basicConfig() does nothing if already configured
    log_level = log_levels[args.log_level]
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Also set level on all existing handlers (in case they were configured with a different level)
    for handler in root_logger.handlers:
        handler.setLevel(log_level)

    # If no handlers exist, add a basic console handler
    if not root_logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        root_logger.addHandler(console_handler)

    try:
        status_reporter.emit(
            EndpointStartupPhase.LOADING_CONFIG,
            "Loading execution server configuration",
        )
        from dataclasses import replace

        from openhcs.runtime.zmq_config import OPENHCS_ZMQ_CONFIG, OpenHCSZMQConfig

        config = OPENHCS_ZMQ_CONFIG
        if args.config_source is not None:
            namespace: dict[str, object] = {}
            exec(args.config_source, namespace)
            supplied_config = namespace.get("config")
            if not isinstance(supplied_config, OpenHCSZMQConfig):
                raise TypeError("--config-source must define config as OpenHCSZMQConfig")
            config = supplied_config
        if args.port is not None:
            config = replace(config, default_port=args.port)
        if args.host is not None:
            config = replace(config, server_host=args.host)
        if args.persistent is not None:
            config = replace(config, persistent=args.persistent)
        if args.transport_mode is not None:
            config = replace(config, transport_mode=args.transport_mode)

        status_reporter.emit(
            EndpointStartupPhase.IMPORTING_RUNTIME,
            "Importing execution runtime",
        )
        if execution_server_type is None:
            from openhcs.runtime.zmq_execution_server import (
                ZMQExecutionServer as execution_server_type,
            )
        if server_runner is None:
            from zmqruntime.runner import serve_forever as server_runner

        logger.info("=" * 60)
        logger.info("ZMQ Execution Server")
        logger.info("=" * 60)
        logger.info("Log level: %s (from --log-level=%s)", logging.getLevelName(log_level), args.log_level)
        logger.info(
            "Port: %s (control: %s)",
            config.default_port,
            config.default_port + config.control_port_offset,
        )
        logger.info("Host: %s", config.server_host)
        logger.info("Transport mode: %s", config.transport_mode.value)
        logger.info("Persistent: %s", config.persistent)
        if args.log_file_path:
            logger.info("Log file: %s", args.log_file_path)
        logger.info("=" * 60)

        status_reporter.emit(
            EndpointStartupPhase.CREATING_SERVER,
            "Creating execution server",
        )
        server = execution_server_type(
            log_file_path=args.log_file_path,
            config=config,
        )

        if args.prepare_capabilities:
            status_reporter.emit(
                EndpointStartupPhase.PREPARING_CAPABILITIES,
                "Preparing endpoint function catalog",
            )
            server.prepare_capabilities()
            logger.info("Execution-server capabilities prepared.")
            return

        status_reporter.emit(
            EndpointStartupPhase.BINDING_ENDPOINT,
            "Binding execution server endpoint",
        )
        server.start()
        status_reporter.emit(
            EndpointStartupPhase.SERVER_READY,
            "Execution server is ready",
        )
        logger.info("Server ready - waiting for requests...")
        server_runner(
            server,
            poll_interval=config.server_poll_interval_seconds,
            handle_signals=True,
        )
        logger.info("Server stopped")
    except BaseException as error:
        status_reporter.emit(
            EndpointStartupPhase.FAILED,
            f"Execution server startup failed: {error}",
        )
        raise


if __name__ == "__main__":
    main()
