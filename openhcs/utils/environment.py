"""
Environment detection utilities for OpenHCS.

Provides functions for detecting runtime environment characteristics
like headless mode, CI environments, and other context-specific settings.
"""

import os
from collections.abc import Mapping


class OpenHCSProcessEnvironment:
    """Own process-mode environment variables that child processes inherit."""

    cpu_only_key = "OPENHCS_CPU_ONLY"
    headless_key = "OPENHCS_HEADLESS"
    use_threading_key = "OPENHCS_USE_THREADING"

    @classmethod
    def child_process_environment_keys(cls) -> tuple[str, ...]:
        """Return mode selectors required for semantic parity in child processes."""

        return (
            cls.cpu_only_key,
            cls.headless_key,
            cls.use_threading_key,
        )

    @staticmethod
    def flag_enabled(
        key: str,
        environment: Mapping[str, str] | None = None,
    ) -> bool:
        """Return whether one owned boolean process flag is enabled."""

        values = os.environ if environment is None else environment
        return values.get(key, "").strip().lower() in {"1", "true", "yes", "on"}

    @classmethod
    def cpu_only_mode(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> bool:
        """Return whether OpenHCS should exclude GPU-backed declarations."""

        return cls.flag_enabled(cls.cpu_only_key, environment)

    @classmethod
    def headless_mode(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> bool:
        """Return whether OpenHCS should avoid interactive display dependencies."""

        return cls.flag_enabled(cls.headless_key, environment)

    @classmethod
    def use_threading_mode(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> bool:
        """Return whether OpenHCS should use thread-backed worker execution."""

        return cls.flag_enabled(cls.use_threading_key, environment)


def is_headless_mode() -> bool:
    """
    Detect headless/CI contexts where viz deps should not be required at import time.

    CPU-only mode does NOT imply headless - you can run CPU mode with napari.
    Only CI or explicit OPENHCS_HEADLESS flag triggers headless mode.

    Returns:
        True if running in headless mode (CI or explicitly set), False otherwise
    """
    try:
        if os.getenv("CI", "").lower() == "true":
            return True
        if OpenHCSProcessEnvironment.headless_mode():
            return True
    except Exception:
        pass
    return False
