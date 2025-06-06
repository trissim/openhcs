# plan_02a3_plate_validation.md
## Component: Plate Validation Service

### Objective
Implement a dedicated service for plate validation and ID generation, maintaining clear separation of concerns from both the Plate Manager pane and Dialog Manager components.

### Plan

1. **Validation Service Class**
   - Create a separate PlateValidationService class
   - Implement thread-safe validation operations
   - Provide deterministic plate ID generation
   - Maintain clear boundaries with other components

2. **Plate Directory Validation**
   - Implement asynchronous plate directory validation
   - Support proper backend handling with positional parameters
   - Handle validation errors gracefully
   - Provide detailed validation results

3. **Plate ID Generation**
   - Implement deterministic plate ID generation
   - Ensure IDs are based on path and backend
   - Normalize paths to handle case sensitivity
   - Prevent ID collisions

4. **Error Handling**
   - Implement proper error handling
   - Provide detailed error information
   - Support logging of validation errors
   - Ensure exceptions are properly propagated

### Findings

#### Key Considerations for Plate Validation

1. **🔒 Component Boundaries (Clause 295)**
   - Maintain clear boundaries between components
   - Use composition over inheritance
   - Define explicit interfaces for inter-component communication
   - Avoid implicit dependencies between components

2. **🔒 Backend Propagation (Clause 306, Clause 310)**
   - All I/O operations must explicitly declare backend parameter
   - Backend must be passed positionally to all FileManager methods
   - Implementation: `fm.get_path(path, backend)` (not `fm.get_path(path, backend=backend)`)
   - Any function accepting file_manager must also accept backend

3. **🔒 Deterministic Plate IDs**
   - Plate IDs must be deterministic based on path and backend
   - No order-dependent ID generation
   - Implementation: `plate_id = generate_plate_id(path, backend)`

4. **🔒 VirtualPath Encapsulation (Clause 319)**
   - TUI must display only filesystem paths to users
   - VirtualPath objects must never be exposed to the UI
   - Implementation: `path_str = str(fm.get_path(plate["path"], backend))`

5. **🔒 Avoiding Excessive Modularization (Clause 9)**
   - Methods should encapsulate meaningful logical units
   - Avoid creating helper methods for trivial operations
   - Inline simple operations rather than creating dedicated methods

### Implementation Draft

```python
"""
Plate Validation Service for OpenHCS TUI.

This module implements a dedicated service for plate validation and ID generation,
maintaining clear separation of concerns from both the Plate Manager pane and
Dialog Manager components.

🔒 Clause 295: Component Boundaries
Maintain clear boundaries between components with explicit interfaces.

🔒 Clause 306: Backend Positional Parameters
All backend parameters must be passed positionally, not as keywords.

🔒 Clause 310: Function Backend Propagation
Any function accepting file_manager must also accept backend.

🔒 Clause 319: TUI_NO_VIRTUALPATH_EXPOSURE
TUI must work only with plain strings, not VirtualPath objects.
"""
import os
import asyncio
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Dict, Any, Protocol, Callable

from ezstitcher.core.context.processing_context import ProcessingContext


# Define callback protocols for clean interfaces
class ValidationResultCallback(Protocol):
    """Protocol for validation result callback."""
    async def __call__(self, result: Dict[str, Any]) -> None: ...

class ErrorCallback(Protocol):
    """Protocol for error callback."""
    async def __call__(self, message: str, details: Optional[str] = None) -> None: ...


class PlateValidationService:
    """
    Service for plate validation and ID generation.

    This class handles all validation operations for plates, including
    directory validation and ID generation. It uses callbacks to communicate
    with the parent component, maintaining clear boundaries.
    """
    def __init__(
        self,
        context: ProcessingContext,
        on_validation_result: ValidationResultCallback,
        on_error: ErrorCallback,
        file_manager=None,
        io_executor: Optional[ThreadPoolExecutor] = None
    ):
        """
        Initialize the validation service.

        Args:
            context: The OpenHCS ProcessingContext
            on_validation_result: Callback for validation results
            on_error: Callback for error handling
            file_manager: FileManager instance (defaults to context.file_manager)
            io_executor: ThreadPoolExecutor for I/O operations
        """
        self.context = context
        self.on_validation_result = on_validation_result
        self.on_error = on_error

        # Get file_manager from context if not provided
        self.file_manager = file_manager
        if self.file_manager is None and hasattr(context, 'file_manager'):
            self.file_manager = context.file_manager

        # Create dedicated executor for I/O operations if not provided
        if io_executor is None:
            self.io_executor = ThreadPoolExecutor(
                max_workers=3,  # Limit concurrent I/O operations
                thread_name_prefix="plate-validation-"
            )
            self._owns_executor = True
        else:
            self.io_executor = io_executor
            self._owns_executor = False

    async def validate_plate(self, path: str, backend: str, /) -> Dict[str, Any]:
        """
        Validate a plate directory with proper backend handling.

        Args:
            path: The path to validate
            backend: The storage backend to use (positional-only)

        Returns:
            A dictionary with validation results

        Raises:
            ValueError: If backend is empty or invalid
            Exception: If validation fails with a specific error
        """
        # 🔒 Clause 316: Strict validation for backend
        # Must explicitly check for empty string to prevent implicit defaults
        if not isinstance(backend, str) or backend.strip() == '':
            raise ValueError("A valid storage backend must be explicitly selected.")
        try:
            # Generate plate name from path
            plate_name = os.path.basename(path)

            # Generate deterministic plate ID
            plate_id = await self.generate_plate_id(path, backend)

            # Create initial plate entry with validating status
            plate = {
                'id': plate_id,
                'name': plate_name,
                'path': path,
                'backend': backend,
                'status': 'validating'
            }

            # Notify parent component that validation has started
            await self.on_validation_result(plate)

            # Validate plate directory
            is_valid = await self._validate_plate_directory(path, backend)

            # Update plate status based on validation result
            plate['status'] = 'ready' if is_valid else 'error'

            # Notify parent component of validation result
            await self.on_validation_result(plate)

            return plate
        except Exception as e:
            # 🔒 Clause 221: Service Boundary Error Handling
            # Catch at service boundary and convert to 'error' plate status
            # instead of re-raising to prevent caller pipeline crash
            
            # Get detailed error information
            import traceback
            error_details = traceback.format_exc()

            # Notify parent component of error
            await self.on_error(
                f"Error validating plate directory: {path}\n{str(e)}",
                error_details
            )
            
            # Update plate status to error instead of re-raising
            plate['status'] = 'error'
            plate['error_message'] = str(e)
            
            # Notify parent component of validation result with error status
            await self.on_validation_result(plate)
            
            # Return plate with error status instead of re-raising
            return plate

    async def _validate_plate_directory(self, path: str, backend: str, /) -> bool:
        """
        Validate a plate directory asynchronously with proper backend handling.

        Args:
            path: The path to validate
            backend: The storage backend to use (positional-only)

        Returns:
            True if valid, False otherwise

        Raises:
            Exception: If validation fails with a specific error
        """
        if not self.file_manager:
            raise ValueError("File manager not available")

        try:
            # Get standardized path from file_manager
            standardized_path = self.file_manager.get_path(path, backend)

            # Use dedicated executor for I/O operations
            loop = asyncio.get_event_loop()

            # 🔒 Clause 24: API Compatibility
            # Call context.validate_plate_directory with correct signature
            # OpenHCS context signature is validate_plate_directory(virtual_path)
            is_valid = await loop.run_in_executor(
                self.io_executor,
                lambda: self.context.validate_plate_directory(standardized_path)
            )

            return is_valid
        except Exception as e:
            # 🔒 Clause 221: Service Boundary Error Handling
            # Log the error but don't re-raise
            await self.on_error(
                f"Error validating plate directory: {path}",
                str(e)
            )
            
            # Return False to indicate validation failure
            return False

    async def generate_plate_id(self, path: str, backend: str, /) -> str:
        """
        Generate a deterministic plate ID based on path and backend.

        Args:
            path: The plate path
            backend: The storage backend (positional-only)

        Returns:
            A deterministic plate ID
        """
        # 🔒 Clause 12: Explicit Error Handling
        # Use context if it provides a method for this
        try:
            if hasattr(self.context, 'generate_plate_id'):
                # Check if the method is async
                import inspect
                if inspect.iscoroutinefunction(self.context.generate_plate_id):
                    return await self.context.generate_plate_id(path, backend)
                else:
                    return self.context.generate_plate_id(path, backend)
        except Exception as e:
            # Log error but continue with fallback implementation
            await self.on_error(
                f"Error using context.generate_plate_id: {str(e)}",
                "Falling back to local implementation"
            )
            # Continue with fallback implementation

        # 🔒 Clause 101: Preserve Path Case
        # Use resolved absolute path without lower-casing to prevent ID collisions
        # on case-sensitive filesystems (e.g., /Data/Plate vs /data/plate)
        resolved_path = str(Path(path).resolve().as_posix())

        # Combine resolved path and backend for uniqueness
        combined = f"{resolved_path}:{backend}"

        # Create a deterministic hash
        hash_obj = hashlib.md5(combined.encode())
        return f"plate_{hash_obj.hexdigest()[:8]}"

    async def close(self):
        """
        Explicitly close and clean up resources.
        
        🔒 Clause 241: Resource Lifecycle Management
        Provides deterministic resource cleanup instead of relying on __del__.
        """
        if hasattr(self, 'io_executor') and hasattr(self, '_owns_executor') and self._owns_executor:
            # Use asyncio to ensure thread pool shutdown is non-blocking
            await asyncio.get_event_loop().run_in_executor(
                None,  # Use default executor for shutdown
                lambda: self.io_executor.shutdown(wait=True)
            )
            # Clear reference to prevent double-shutdown
            self.io_executor = None
            
    def __del__(self):
        """
        Fallback cleanup for resources if close() wasn't called.
        
        Note: This is non-deterministic and should not be relied upon.
        Always call close() explicitly from the parent component.
        """
        if hasattr(self, 'io_executor') and hasattr(self, '_owns_executor') and self._owns_executor and self.io_executor is not None:
            import warnings
            warnings.warn("PlateValidationService.__del__ called without explicit close(). This is non-deterministic and may leak resources.")
            self.io_executor.shutdown(wait=False)
```
