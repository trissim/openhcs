===================
Storage Adapter
===================

Overview
--------

The ``StorageAdapter`` is an abstraction layer for storing and retrieving pipeline artifacts (primarily NumPy arrays).
It provides a key-value storage interface that can be implemented by different backends, such as in-memory dictionaries
or Zarr stores on disk.

Storage Modes
------------

EZStitcher supports three storage modes:

1. ``legacy``: Uses the existing in-memory dictionary within Pipeline (default)
2. ``memory``: Uses MemoryStorageAdapter (persists .npy files on completion)
3. ``zarr``: Uses ZarrStorageAdapter (persists to disk immediately)

You can specify the storage mode when creating a PipelineOrchestrator:

.. code-block:: python

    from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator

    # Use memory storage
    orchestrator = PipelineOrchestrator(
        plate_path="path/to/plate",
        storage_mode="memory"
    )

    # Use Zarr storage with a specific root directory
    orchestrator = PipelineOrchestrator(
        plate_path="path/to/plate",
        storage_mode="zarr",
        storage_root=Path("/path/to/zarr/root")
    )

Storage Adapter Interface
------------------------

The ``StorageAdapter`` interface defines the following methods:

.. code-block:: python

    class StorageAdapter(ABC):
        @abstractmethod
        def write(self, key: str, data: np.ndarray) -> None:
            """Store data associated with a key."""
            pass

        @abstractmethod
        def read(self, key: str) -> np.ndarray:
            """Retrieve data associated with a key."""
            pass

        @abstractmethod
        def exists(self, key: str) -> bool:
            """Check if a key exists in the storage."""
            pass

        @abstractmethod
        def delete(self, key: str) -> None:
            """Delete the data associated with a key."""
            pass

        @abstractmethod
        def list_keys(self) -> List[str]:
            """List all keys currently stored."""
            pass

        @abstractmethod
        def persist(self, output_dir: Path) -> None:
            """
            Persist the contents of the storage to a specified directory.
            Behavior depends on the implementation (e.g., write files, no-op).
            """
            pass

Using StorageAdapter in Steps
----------------------------

Pipeline steps automatically use the StorageAdapter when available. The Step._save_images method checks for a
StorageAdapter in the context and uses it if available, falling back to FileManager only when necessary.

You can also use the StorageAdapter directly in your custom steps:

.. code-block:: python

    def process(self, context: 'ProcessingContext') -> 'ProcessingContext':
        # Process images...
        
        # Store results using the storage adapter if available
        if context.orchestrator and context.orchestrator.storage_adapter:
            # Generate a key for the data
            from ezstitcher.io.storage_adapter import generate_storage_key
            key = generate_storage_key(self.name, well, component)
            
            # Store the data
            context.orchestrator.storage_adapter.write(key, data)
        else:
            # Fall back to FileManager
            file_manager = context.orchestrator.file_manager
            file_manager.save_image(data, output_path)
            
        return context

Helper Method in ProcessingContext
---------------------------------

For convenience, ProcessingContext provides a helper method for storing arrays:

.. code-block:: python

    def process(self, context: 'ProcessingContext') -> 'ProcessingContext':
        # Process images...
        
        # Store results using the context helper method
        key = generate_storage_key(self.name, well, component)
        context.store_array(key, data)
            
        return context

Key Generation
------------

To ensure consistent key naming across all steps, use the generate_storage_key utility function:

.. code-block:: python

    from ezstitcher.io.storage_adapter import generate_storage_key
    
    # Generate a key for a step's output
    key = generate_storage_key(step_name, well, component)
    
    # Examples:
    # generate_storage_key("Z-Stack Flattening", "A01", "channel_1")
    # -> "z-stack_flattening_A01_channel_1"
    # 
    # generate_storage_key("Normalization", "B02")
    # -> "normalization_B02"
