# pragma: no cover – test helper only
# __doctrinal_owner__: OpenHCS/core-architecture
# __doctrinal_clauses__: Clause 65, Clause 244
"""
Defines UnsafeRegistry for development and testing scenarios ONLY.
Its usage in production code is a critical doctrine violation.
"""
import inspect
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Type

from openhcs.core.exceptions import DoctrineViolationError, DoctrineWarning
from openhcs.registries.registry_base import (BackendNotFoundError,
                                                 ImmutabilityError,
                                                 RegistryBase)

# --- Doctrinal Constants ---
# Clause 65: No Fallback Logic / Fail Loudly - Misuse of UnsafeRegistry is a hard failure.
# Clause 244: Rot Intolerance - UnsafeRegistry in production is rot.

BackendClass = Type[Any] # Generic type for backend classes

class UnsafeRegistry(RegistryBase):
    """
    A mutable registry intended ONLY for development and testing purposes.
    This class deliberately bypasses the immutability constraints of
    standard OpenHCS registries.

    IMPORTING OR USING THIS CLASS IN PRODUCTION CODE IS A SEVERE
    DOCTRINAL VIOLATION AND WILL FAIL CI CHECKS.

    It should only be used within `tests/` directories or when the
    `DEBUG_SANDBOX=True` environment variable is explicitly set.
    """
    # No _initialized_registry_state override needed from ImmutableRegistry,
    # as we will disable finalization.
    # We need a place to store backends, similar to how a normal registry might.
    _backends: Dict[str, BackendClass]

    def __init__(self):
        # Call super().__init__() to set up _initialized_registry_state
        super().__init__()

        # Override the finalization flag from ImmutableRegistry.
        # This effectively keeps the registry mutable indefinitely.
        object.__setattr__(self, "_initialized_registry_state", False) # Keep it mutable

        if not self._is_allowed_context():
            # Clause 65: Fail Loudly
            raise DoctrineViolationError(
                "CRITICAL DOCTRINE VIOLATION: UnsafeRegistry instantiated outside of an "
                "allowed context (tests/ directory or DEBUG_SANDBOX=True). "
                "This class is strictly for development and testing."
            )

        warnings.warn(
            "UnsafeRegistry is active. This is for development/testing ONLY and "
            "violates standard OpenHCS doctrinal constraints on registry immutability. "
            "Ensure this is not used in production code.",
            DoctrineWarning,
            stacklevel=2
        )
        # Initialize _backends after super init and checks
        object.__setattr__(self, "_backends", {})


    def _is_allowed_context(self) -> bool:
        """
        Checks if the current execution context allows the use of UnsafeRegistry.
        Allowed contexts:
        1. The instantiation occurs within a file located in a `tests/` directory.
        2. The environment variable `DEBUG_SANDBOX` is set to "True".
        """
        # Walk up the stack to find the caller's context outside of this file.
        # Max depth to prevent infinite loops in weird scenarios.
        allowed = False
        for i in range(1, 10): # Check a few frames up
            try:
                frame = inspect.currentframe()
                if frame is None: break
                for _ in range(i): # Go up i frames
                    if frame.f_back:
                        frame = frame.f_back
                    else:
                        frame = None # No more frames
                        break
                if frame is None: break

                filename = inspect.getframeinfo(frame).filename
                # Normalize path for consistent checking
                normalized_filename = Path(filename).resolve().as_posix()

                # Check 1: Is the file in a 'tests/' directory?
                # A more robust check might involve finding project root and checking relative path.
                if "/tests/" in normalized_filename or normalized_filename.startswith("tests/"):
                    allowed = True
                    break
            except Exception:
                # If inspecting frames fails, err on the side of caution
                pass # Continue to DEBUG_SANDBOX check
            finally:
                del frame # Important to break reference cycles

        # Check 2: Is the DEBUG_SANDBOX environment variable set?
        if os.environ.get("DEBUG_SANDBOX") == "True":
            allowed = True

        return allowed

    def add_backend(self, name: str, backend_class: BackendClass) -> None:
        """
        Adds a backend to this unsafe registry.
        This method is a deliberate violation of immutability for testing.
        """
        # Check if somehow finalized by mistake
        if getattr(self, "_initialized_registry_state", True):
            # This should not happen given __init__ logic, but as a safeguard.
            raise ImmutabilityError("UnsafeRegistry was unexpectedly finalized.")

        if not isinstance(name, str) or not name:
            raise ValueError("Backend name must be a non-empty string.")
        if not inspect.isclass(backend_class):
            raise TypeError("backend_class must be a class.")

        self._backends[name] = backend_class
        print(f"UnsafeRegistry: Added backend '{name}' -> {backend_class.__name__}")


    def get_backend(self, name: str) -> BackendClass:
        """Retrieves a backend by name."""
        if name not in self._backends:
            available_backends = list(self._backends.keys())
            raise BackendNotFoundError(
                f"Backend '{name}' not found in UnsafeRegistry. Available: {available_backends}"
            )
        return self._backends[name]

    def clear_backends(self) -> None:
        """Clears all backends. For testing convenience."""
        if getattr(self, "_initialized_registry_state", True):
            raise ImmutabilityError("UnsafeRegistry was unexpectedly finalized.")
        self._backends.clear()
        print("UnsafeRegistry: All backends cleared.")

    # Override _finalize_registry_initialization to do nothing, ensuring it remains mutable.
    def _finalize_registry_initialization(self) -> None:
        """Overrides base method to prevent finalization for UnsafeRegistry."""
        # Do not call object.__setattr__(self, "_initialized_registry_state", True)
        # This keeps the registry mutable
