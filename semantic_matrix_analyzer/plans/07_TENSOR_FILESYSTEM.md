# Tensor Filesystem Interface

## Overview

This plan outlines the implementation of a CLI interface for the SMA that leverages the existing FileManager to seamlessly load text files from disk into tensors stored in memory. The interface will provide a consistent API for accessing both disk and memory storage, with the backend specified as the last positional argument in all operations.

## Motivation

The Semantic Matrix Analyzer needs to efficiently process code files as tensors for GPU-accelerated analysis. By leveraging the existing FileManager and GPU plugin, we can create a seamless interface that allows agents to interact with the filesystem and process code files as tensors using a familiar terminal-like interface.

## Design Principles

1. **Backend Agnosticism**: The filepath never changes; only the backend enum (DISK, MEMORY, ZARR) changes as the last positional argument.
2. **Seamless Abstraction**: Complete abstraction of the backend to a single positional argument enum.
3. **Terminal-like Interface**: Simplify interaction for agents by providing a familiar terminal-like interface.
4. **Separation of Responsibilities**: Use the existing FileManager for file operations and the GPU plugin for tensorization.

## Implementation Plan

### Phase 1: Core Components

#### 1. Direct Use of FileManager

We'll use the existing FileManager directly with its exact interface, ensuring a 1:1 mapping for all operations:

```python
from openhcs.io.base import StorageBackend
from openhcs.io.filemanager import FileManager
from pathlib import Path
import torch

# Create a registry with standard backends
registry = {
    "disk": DiskBackend,
    "memory": MemoryBackend,
    "zarr": ZarrBackend  # If available
}

# Create a file manager
file_manager = FileManager(registry)

# Example usage:
# Load a file from disk
text_content = file_manager.load("myfile.py", "disk")

# Convert to tensor using GPU plugin
tensor = gpu_plugin.text_to_tensor(text_content)

# Save tensor to memory
file_manager.save("myfile.tensor", tensor, "memory")

# Load tensor from memory
tensor = file_manager.load("myfile.tensor", "memory")

# Convert tensor back to text
text = gpu_plugin.tensor_to_text(tensor)

# List files
files = file_manager.list_files("/path/to/dir", "disk")

# Check if file exists
exists = file_manager.exists("myfile.py", "disk")

# Ensure directory exists
file_manager.ensure_directory("/path/to/dir", "disk")

# All operations maintain the exact same interface as FileManager
# with backend always as the last positional argument
```

#### 2. GPU Plugin Tensorization

The GPU plugin will handle the conversion between text and tensors:

```python
# Add to the GPUAnalysisPlugin class:

def text_to_tensor(self, text, device=None):
    """
    Convert text to a tensor.

    Args:
        text: Text to convert
        device: Device to create the tensor on (defaults to plugin's device)

    Returns:
        PyTorch tensor
    """
    if device is None:
        device = self.device

    # Convert each character to its ASCII/Unicode value
    char_values = [ord(c) for c in text]

    # Create tensor
    return torch.tensor(char_values, dtype=torch.int32, device=device)

def tensor_to_text(self, tensor):
    """
    Convert a tensor back to text.

    Args:
        tensor: PyTorch tensor containing character codes

    Returns:
        Text representation
    """
    # Move to CPU if on GPU
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()

    # Convert each integer back to a character and join
    return ''.join(chr(int(code)) for code in tensor.numpy())
```

#### 3. CLI Interface

A command-line interface for interacting with the tensor filesystem that maintains the exact same interface as FileManager:

```python
class TensorFSCLI:
    """
    Command-line interface for tensor filesystem operations.

    This class provides a terminal-like interface for interacting with
    the filesystem and tensor operations, making it easy for agents to use.
    All commands maintain the exact same interface as FileManager with
    backend always as the last positional argument.
    """

    def __init__(self, file_manager, gpu_plugin):
        """
        Initialize the CLI.

        Args:
            file_manager: FileManager instance
            gpu_plugin: GPUAnalysisPlugin instance
        """
        self.file_manager = file_manager
        self.gpu_plugin = gpu_plugin
        self.current_backend = "disk"  # Default backend

    def execute(self, command):
        """
        Execute a command.

        Args:
            command: Command string to execute

        Returns:
            Command result
        """
        parts = command.split()
        if not parts:
            return ""

        cmd = parts[0].lower()
        args = parts[1:]

        # Command handlers - all maintain the exact FileManager interface
        if cmd == "load":
            return self._handle_load(args)
        elif cmd == "save":
            return self._handle_save(args)
        elif cmd == "list_files":
            return self._handle_list_files(args)
        elif cmd == "exists":
            return self._handle_exists(args)
        elif cmd == "ensure_directory":
            return self._handle_ensure_directory(args)
        elif cmd == "delete":
            return self._handle_delete(args)
        elif cmd == "copy":
            return self._handle_copy(args)
        elif cmd == "move":
            return self._handle_move(args)
        elif cmd == "tensorize":
            return self._handle_tensorize(args)
        elif cmd == "detensorize":
            return self._handle_detensorize(args)
        elif cmd == "backend":
            return self._handle_backend(args)
        elif cmd == "help":
            return self._handle_help(args)
        else:
            return f"Unknown command: {cmd}. Type 'help' for available commands."
```

### Phase 2: SMA Integration

#### 1. SMA CLI Extension

Extend the SMA CLI to include tensor filesystem commands that maintain the exact same interface as FileManager:

```python
def add_tensor_fs_commands(subparsers):
    """Add tensor filesystem commands to the SMA CLI."""
    # Tensor filesystem commands
    tensor_fs_parser = subparsers.add_parser(
        "tensor-fs",
        help="Tensor filesystem operations with exact FileManager interface"
    )
    tensor_fs_subparsers = tensor_fs_parser.add_subparsers(
        dest="tensor_fs_command",
        help="Tensor filesystem command"
    )

    # Load file
    load_parser = tensor_fs_subparsers.add_parser(
        "load",
        help="Load a file"
    )
    load_parser.add_argument(
        "path",
        help="Path to file"
    )
    load_parser.add_argument(
        "backend",
        choices=["disk", "memory", "zarr"],
        help="Backend to use"
    )

    # Save file
    save_parser = tensor_fs_subparsers.add_parser(
        "save",
        help="Save content to a file"
    )
    save_parser.add_argument(
        "path",
        help="Path to save to"
    )
    save_parser.add_argument(
        "content",
        help="Content to save (or '-' to read from stdin)"
    )
    save_parser.add_argument(
        "backend",
        choices=["disk", "memory", "zarr"],
        help="Backend to use"
    )

    # List files
    list_files_parser = tensor_fs_subparsers.add_parser(
        "list_files",
        help="List files in a directory"
    )
    list_files_parser.add_argument(
        "directory",
        help="Directory to list files from"
    )
    list_files_parser.add_argument(
        "backend",
        choices=["disk", "memory", "zarr"],
        help="Backend to use"
    )
    list_files_parser.add_argument(
        "--pattern",
        help="Pattern to filter files"
    )
    list_files_parser.add_argument(
        "--recursive",
        action="store_true",
        help="List files recursively"
    )

    # Check if file exists
    exists_parser = tensor_fs_subparsers.add_parser(
        "exists",
        help="Check if a file exists"
    )
    exists_parser.add_argument(
        "path",
        help="Path to check"
    )
    exists_parser.add_argument(
        "backend",
        choices=["disk", "memory", "zarr"],
        help="Backend to use"
    )

    # Ensure directory exists
    ensure_directory_parser = tensor_fs_subparsers.add_parser(
        "ensure_directory",
        help="Ensure a directory exists"
    )
    ensure_directory_parser.add_argument(
        "directory",
        help="Directory to ensure exists"
    )
    ensure_directory_parser.add_argument(
        "backend",
        choices=["disk", "memory", "zarr"],
        help="Backend to use"
    )

    # Delete file
    delete_parser = tensor_fs_subparsers.add_parser(
        "delete",
        help="Delete a file"
    )
    delete_parser.add_argument(
        "path",
        help="Path to delete"
    )
    delete_parser.add_argument(
        "backend",
        choices=["disk", "memory", "zarr"],
        help="Backend to use"
    )

    # Tensorize file (GPU plugin extension)
    tensorize_parser = tensor_fs_subparsers.add_parser(
        "tensorize",
        help="Convert text file to tensor"
    )
    tensorize_parser.add_argument(
        "source",
        help="Source file path"
    )
    tensorize_parser.add_argument(
        "destination",
        help="Destination file path"
    )
    tensorize_parser.add_argument(
        "backend",
        choices=["memory", "zarr"],
        help="Backend to save tensor to"
    )
    tensorize_parser.add_argument(
        "--device",
        default=None,
        help="Device to create tensor on"
    )

    # Detensorize file (GPU plugin extension)
    detensorize_parser = tensor_fs_subparsers.add_parser(
        "detensorize",
        help="Convert tensor to text file"
    )
    detensorize_parser.add_argument(
        "source",
        help="Source file path"
    )
    detensorize_parser.add_argument(
        "destination",
        help="Destination file path"
    )
    detensorize_parser.add_argument(
        "backend",
        choices=["disk"],
        help="Backend to save text to"
    )

    # Interactive shell
    shell_parser = tensor_fs_subparsers.add_parser(
        "shell",
        help="Start interactive tensor filesystem shell"
    )
```

#### 2. Command Handlers

Implement handlers for the tensor filesystem commands that maintain the exact same interface as FileManager:

```python
def handle_tensor_fs_command(args):
    """Handle tensor filesystem commands."""
    # Create registry and file manager
    from openhcs.io.filemanager import FileManager
    from openhcs.io.disk import DiskBackend
    from openhcs.io.memory import MemoryBackend

    # Create registry with standard backends
    registry = {
        "disk": DiskBackend,
        "memory": MemoryBackend
    }

    # Try to add ZARR backend if available
    try:
        from openhcs.io.zarr import ZarrBackend
        registry["zarr"] = ZarrBackend
    except ImportError:
        pass

    # Create file manager
    file_manager = FileManager(registry)

    # Get GPU plugin for tensorization
    gpu_plugin = get_gpu_plugin()

    # Handle commands - all maintain the exact FileManager interface
    if args.tensor_fs_command == "load":
        handle_tensor_fs_load(args, file_manager, gpu_plugin)
    elif args.tensor_fs_command == "save":
        handle_tensor_fs_save(args, file_manager)
    elif args.tensor_fs_command == "list_files":
        handle_tensor_fs_list_files(args, file_manager)
    elif args.tensor_fs_command == "exists":
        handle_tensor_fs_exists(args, file_manager)
    elif args.tensor_fs_command == "ensure_directory":
        handle_tensor_fs_ensure_directory(args, file_manager)
    elif args.tensor_fs_command == "delete":
        handle_tensor_fs_delete(args, file_manager)
    elif args.tensor_fs_command == "tensorize":
        handle_tensor_fs_tensorize(args, file_manager, gpu_plugin)
    elif args.tensor_fs_command == "detensorize":
        handle_tensor_fs_detensorize(args, file_manager, gpu_plugin)
    elif args.tensor_fs_command == "shell":
        handle_tensor_fs_shell(args, file_manager, gpu_plugin)
    else:
        print(f"Unknown tensor filesystem command: {args.tensor_fs_command}")
```

### Phase 3: Integration with GPU Analysis

#### 1. SMA Core Integration

Integrate the tensor filesystem with SMA's core functionality, maintaining the exact same interface as FileManager:

```python
class SMAWithTensorFS:
    """
    SMA with integrated tensor filesystem.

    This class extends SMA with tensor filesystem capabilities,
    allowing seamless loading and processing of code files as tensors.
    All file operations maintain the exact same interface as FileManager
    with backend always as the last positional argument.
    """

    def __init__(self):
        """Initialize SMA with tensor filesystem."""
        # Initialize SMA
        self.sma = SMA()

        # Initialize file manager
        from openhcs.io.filemanager import FileManager
        from openhcs.io.disk import DiskBackend
        from openhcs.io.memory import MemoryBackend

        # Create registry with standard backends
        registry = {
            "disk": DiskBackend,
            "memory": MemoryBackend
        }

        # Try to add ZARR backend if available
        try:
            from openhcs.io.zarr import ZarrBackend
            registry["zarr"] = ZarrBackend
        except ImportError:
            pass

        # Create file manager - direct use without subclassing
        self.file_manager = FileManager(registry)

        # Get GPU plugin for tensorization
        self.gpu_plugin = self.sma.plugin_manager.get_plugin("gpu_analysis")

    def analyze_code_as_tensor(self, file_path, device=None):
        """
        Analyze code as a tensor.

        This method loads a code file, converts it to a tensor,
        and analyzes it using SMA's analysis capabilities.

        Args:
            file_path: Path to the code file
            device: Device to load the tensor to

        Returns:
            Analysis results
        """
        # Load from disk - using FileManager's exact interface
        text = self.file_manager.load(file_path, "disk")

        # Convert to tensor using GPU plugin
        tensor = self.gpu_plugin.text_to_tensor(text, device)

        # Save tensor to memory - using FileManager's exact interface
        memory_path = f"analysis/{Path(file_path).name}"
        self.file_manager.save(memory_path, tensor, "memory")

        # Analyze the code
        results = self.sma.analyze_tensor(tensor)

        return results

    # Example of other methods that maintain FileManager's exact interface

    def list_tensor_files(self, directory, backend="memory", pattern=None, recursive=False):
        """
        List tensor files in a directory.

        Args:
            directory: Directory to list files from
            backend: Backend to use (last positional argument)
            pattern: Optional pattern to filter files
            recursive: Whether to list files recursively

        Returns:
            List of tensor files
        """
        # Direct use of FileManager's list_files method with exact interface
        return self.file_manager.list_files(directory, backend, pattern, recursive)

    def ensure_tensor_directory(self, directory, backend="memory"):
        """
        Ensure a directory exists for storing tensors.

        Args:
            directory: Directory to ensure exists
            backend: Backend to use (last positional argument)

        Returns:
            Path to the directory
        """
        # Direct use of FileManager's ensure_directory method with exact interface
        return self.file_manager.ensure_directory(directory, backend)
```

## Benefits

1. **Clean Separation of Responsibilities**:
   - FileManager handles file operations
   - GPU plugin handles tensorization
   - SMA handles analysis

2. **Simplified Interface**:
   - Backend is always the last positional argument
   - Filepath never changes, only the backend enum changes
   - Terminal-like interface for agents

3. **Reuse of Existing Components**:
   - Uses the existing FileManager without subclassing
   - Leverages the GPU plugin for tensorization
   - Integrates with SMA's plugin system

4. **Consistent API**:
   - Same interface pattern across all components
   - Backend is always specified as the last positional argument
   - Seamless abstraction of storage backends

## Success Criteria

1. Agents can interact with the filesystem using a familiar terminal-like interface
2. Code files can be seamlessly loaded as tensors for GPU-accelerated analysis
3. The interface maintains a consistent API across all backends
4. The implementation properly separates responsibilities between components

## Timeline

- Phase 1 (Core Components): 1 week
- Phase 2 (SMA Integration): 1 week
- Phase 3 (Integration with GPU Analysis): 1 week

## Conclusion

This plan provides a clean and efficient way to leverage the existing FileManager and GPU plugin for tensor operations. By maintaining a clear separation of responsibilities, we ensure that each component focuses on its core functionality, resulting in a more maintainable and extensible system.
