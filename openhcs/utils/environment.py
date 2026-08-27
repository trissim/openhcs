"""
Environment detection utilities for OpenHCS.

Provides functions for detecting runtime environment characteristics
like headless mode, CI environments, and other context-specific settings.
"""

import os
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import TYPE_CHECKING

from openhcs.resources.brand import BRAND_PRODUCT_NAME

if TYPE_CHECKING:
    from openhcs.agent.runtime_platform import AgentRuntimePlatformAuthority


class OpenHCSProcessEnvironment:
    """Own process-mode environment variables that child processes inherit."""

    cpu_only_key = "OPENHCS_CPU_ONLY"
    headless_key = "OPENHCS_HEADLESS"
    numba_cache_key = "NUMBA_CACHE_DIR"
    subprocess_no_gpu_key = "OPENHCS_SUBPROCESS_NO_GPU"
    polystore_subprocess_no_gpu_key = "POLYSTORE_SUBPROCESS_NO_GPU"
    use_threading_key = "OPENHCS_USE_THREADING"

    @staticmethod
    def numba_cache_path(
        platform_authority: "AgentRuntimePlatformAuthority",
    ) -> Path:
        """Return the user-local cache path for compiled Numba artifacts."""

        return (
            platform_authority.application_data_root(BRAND_PRODUCT_NAME) / "numba"
        ).resolve(strict=False)

    @classmethod
    def current_numba_cache_path(cls) -> Path:
        """Return the compiled-code cache for the current host platform."""

        from openhcs.agent.runtime_platform import AgentRuntimePlatformAuthority

        return cls.numba_cache_path(AgentRuntimePlatformAuthority.current())

    @classmethod
    def child_process_environment_keys(cls) -> tuple[str, ...]:
        """Return mode selectors required for semantic parity in child processes."""

        return (
            cls.cpu_only_key,
            cls.headless_key,
            cls.numba_cache_key,
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
    def gpu_imports_disabled(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> bool:
        """Return whether this process must avoid GPU-library imports."""

        return cls.cpu_only_mode(environment) or cls.flag_enabled(
            cls.subprocess_no_gpu_key,
            environment,
        )

    @classmethod
    def enable_cpu_only_mode(
        cls,
        environment: MutableMapping[str, str] | None = None,
    ) -> None:
        """Enable CPU-only execution and project it to dependency imports."""

        values = os.environ if environment is None else environment
        values[cls.cpu_only_key] = "true"
        cls.project_dependency_gpu_import_policy(values)

    @classmethod
    def project_dependency_gpu_import_policy(
        cls,
        environment: MutableMapping[str, str] | None = None,
    ) -> None:
        """Project OpenHCS GPU-import policy to import-time consumers."""

        values = os.environ if environment is None else environment
        if cls.gpu_imports_disabled(values):
            values[cls.subprocess_no_gpu_key] = "1"
            values[cls.polystore_subprocess_no_gpu_key] = "1"

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
