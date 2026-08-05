#!/usr/bin/env python3
"""
ZMQ Execution Server Launcher

Standalone script for spawning ZMQ execution server processes.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from zmqruntime.config import TransportMode
from zmqruntime.runner import serve_forever

from openhcs.runtime.zmq_config import OPENHCS_ZMQ_CONFIG, OpenHCSZMQConfig
from openhcs.runtime.zmq_execution_server import ZMQExecutionServer

logger = logging.getLogger(__name__)

LOG_LEVEL_BY_NAME = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def main():
    """Main entry point for server launcher."""
    default_mode = OPENHCS_ZMQ_CONFIG.transport_mode
    default_mode_str = "tcp" if default_mode == TransportMode.TCP else "ipc"

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
        type=str,
        default=None,
        choices=["ipc", "tcp"],
        help=f"Transport mode (default: {default_mode_str} for this platform)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=tuple(LOG_LEVEL_BY_NAME),
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--config-source",
        type=str,
        default=None,
        help="Pycodified OpenHCSZMQConfig supplied by the spawning process",
    )

    args = parser.parse_args()

    # Configure logging with the specified level
    # CRITICAL: Must force reconfigure root logger - basicConfig() does nothing if already configured
    log_level = LOG_LEVEL_BY_NAME[args.log_level.upper()]
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

    config = OPENHCS_ZMQ_CONFIG
    if args.config_source is not None:
        namespace: dict[str, object] = {}
        exec(args.config_source, namespace)
        supplied_config = namespace.get("config")
        if not isinstance(supplied_config, OpenHCSZMQConfig):
            raise TypeError("--config-source must define config as OpenHCSZMQConfig")
        config = supplied_config
    overrides = {}
    if args.port is not None:
        overrides["default_port"] = args.port
    if args.host is not None:
        overrides["server_host"] = args.host
    if args.persistent is not None:
        overrides["persistent"] = args.persistent
    if args.transport_mode is not None:
        overrides["transport_mode"] = (
            TransportMode.IPC
            if args.transport_mode == "ipc"
            else TransportMode.TCP
        )
    if overrides:
        config = replace(config, **overrides)

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

    server = ZMQExecutionServer(
        log_file_path=args.log_file_path,
        config=config,
    )

    server.start()
    logger.info("Server ready - waiting for requests...")
    serve_forever(
        server,
        poll_interval=config.server_poll_interval_seconds,
        handle_signals=True,
    )
    logger.info("Server stopped")


if __name__ == "__main__":
    main()
