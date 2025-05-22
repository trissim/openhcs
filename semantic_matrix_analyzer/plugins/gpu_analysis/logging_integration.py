"""
Logging Integration Module for GPU Analysis Plugin.

This module provides integration with SMA's logging system, ensuring
consistent logging throughout the application and proper logging for
the implementation of unimplemented SMA methods.
"""

from typing import Optional, Any, Dict
import logging

# Configure fallback logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class SMALoggerAdapter:
    """
    Adapter for SMA's logging system.

    This class provides an adapter between Python's standard logging module
    and SMA's logging system, allowing GPU Analysis Plugin components to
    use SMA's logging system through a familiar interface.
    """

    def __init__(self, context: Optional['PluginContext'] = None, name: str = "gpu_analysis"):
        """
        Initialize the logger adapter.

        Args:
            context: SMA plugin context
            name: Logger name
        """
        self.context = context
        self.name = name

        # Set up fallback logger
        self.fallback_logger = logging.getLogger(name)

    def debug(self, message: str) -> None:
        """
        Log a debug message.

        Args:
            message: Message to log
        """
        if self.context:
            self.context.log("debug", f"[{self.name}] {message}")
        else:
            self.fallback_logger.debug(message)

    def info(self, message: str) -> None:
        """
        Log an info message.

        Args:
            message: Message to log
        """
        if self.context:
            self.context.log("info", f"[{self.name}] {message}")
        else:
            self.fallback_logger.info(message)

    def warning(self, message: str) -> None:
        """
        Log a warning message.

        Args:
            message: Message to log
        """
        if self.context:
            self.context.log("warning", f"[{self.name}] {message}")
        else:
            self.fallback_logger.warning(message)

    def error(self, message: str) -> None:
        """
        Log an error message.

        Args:
            message: Message to log
        """
        if self.context:
            self.context.log("error", f"[{self.name}] {message}")
        else:
            self.fallback_logger.error(message)

    def critical(self, message: str) -> None:
        """
        Log a critical message.

        Args:
            message: Message to log
        """
        if self.context:
            self.context.log("critical", f"[{self.name}] {message}")
        else:
            self.fallback_logger.critical(message)

    def exception(self, message: str) -> None:
        """
        Log an exception message.

        Args:
            message: Message to log
        """
        if self.context:
            self.context.log("error", f"[{self.name}] {message}")
        else:
            self.fallback_logger.exception(message)


# Global logger instance
_logger = SMALoggerAdapter()


def set_context(context: 'PluginContext') -> None:
    """
    Set the SMA plugin context for logging.

    Args:
        context: SMA plugin context
    """
    global _logger
    _logger = SMALoggerAdapter(context)


def get_logger(name: str = "gpu_analysis") -> SMALoggerAdapter:
    """
    Get a logger adapter for the given name.

    Args:
        name: Logger name

    Returns:
        Logger adapter
    """
    return SMALoggerAdapter(_logger.context, name)
