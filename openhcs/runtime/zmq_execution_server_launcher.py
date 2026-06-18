#!/usr/bin/env python3
"""
ZMQ Execution Server Launcher

Standalone script for spawning ZMQ execution server processes.
"""
from __future__ import annotations

import argparse
import logging
from zmqruntime.config import TransportMode
from zmqruntime.transport import get_default_transport_mode
from zmqruntime.runner import serve_forever

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
    from openhcs.constants.constants import DEFAULT_EXECUTION_SERVER_PORT, CONTROL_PORT_OFFSET

    default_mode = get_default_transport_mode()
    default_mode_str = "tcp" if default_mode == TransportMode.TCP else "ipc"

    parser = argparse.ArgumentParser(description="ZMQ Execution Server Launcher")
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_EXECUTION_SERVER_PORT,
        help=f"Data port (control port will be port + {CONTROL_PORT_OFFSET})",
    )
    parser.add_argument("--host", type=str, default="*", help="Host to bind to (default: * for all interfaces)")
    parser.add_argument("--persistent", action="store_true", help="Run as persistent server (detached)")
    parser.add_argument("--log-file-path", type=str, default=None, help="Path to server log file (for client discovery)")
    parser.add_argument(
        "--transport-mode",
        type=str,
        default=default_mode_str,
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

    transport_mode = TransportMode.IPC if args.transport_mode == "ipc" else TransportMode.TCP

    logger.info("=" * 60)
    logger.info("ZMQ Execution Server")
    logger.info("=" * 60)
    logger.info("Log level: %s (from --log-level=%s)", logging.getLevelName(log_level), args.log_level)
    logger.info("Port: %s (control: %s)", args.port, args.port + 1000)
    logger.info("Host: %s", args.host)
    logger.info("Transport mode: %s", transport_mode.value)
    logger.info("Persistent: %s", args.persistent)
    if args.log_file_path:
        logger.info("Log file: %s", args.log_file_path)
    logger.info("=" * 60)

    server = ZMQExecutionServer(
        port=args.port,
        host=args.host,
        log_file_path=args.log_file_path,
        transport_mode=transport_mode,
    )

    logger.info("Server ready - waiting for requests...")
    serve_forever(server, poll_interval=0.01, handle_signals=True)
    logger.info("Server stopped")


if __name__ == "__main__":
    main()
