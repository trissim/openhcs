# Tensor Filesystem Interface

## Overview

This plan outlines the implementation of a CLI interface for the SMA that leverages the existing FileManager to seamlessly load text files from disk into tensors stored in memory. The interface will provide a consistent API for accessing both disk and memory storage, with the backend specified as the last positional argument in all operations.

## Design Principles

1. **Backend Agnosticism**: The filepath never changes; only the backend enum (DISK, MEMORY, ZARR) changes as the last positional argument.
2. **Seamless Abstraction**: Complete abstraction of the backend to a single positional argument enum.
3. **Terminal-like Interface**: Simplify interaction for agents by providing a familiar terminal-like interface.
4. **Separation of Responsibilities**: Use the existing FileManager for file operations and the GPU plugin for tensorization.

## Core Components

### 1. Direct Use of FileManager

Instead of creating a specialized TensorFileManager, we'll use the existing FileManager directly:

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
tensor = gpu_plugin.text_to_tensor(text_content, device="cuda")

# Save tensor to memory
file_manager.save("myfile.tensor", tensor, "memory")

# Load tensor from memory
tensor = file_manager.load("myfile.tensor", "memory")

# Convert tensor back to text
text = gpu_plugin.tensor_to_text(tensor)
```

### 2. GPU Plugin Tensorization

The GPU plugin will handle the conversion between text and tensors:

```python
class GPUAnalysisPlugin:
    """GPU-accelerated analysis plugin."""

    def __init__(self, device="cuda"):
        """Initialize the plugin."""
        self.device = device if torch.cuda.is_available() else "cpu"

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

### 3. CLI Interface

A command-line interface for interacting with the tensor filesystem:

```python
class TensorFSCLI:
    """
    Command-line interface for tensor filesystem operations.

    This class provides a terminal-like interface for interacting with
    the filesystem and tensor operations, making it easy for agents to use.
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

        # Command handlers
        if cmd == "ls":
            return self._handle_ls(args)
        elif cmd == "cat":
            return self._handle_cat(args)
        elif cmd == "tensorize":
            return self._handle_tensorize(args)
        elif cmd == "detensorize":
            return self._handle_detensorize(args)
        elif cmd == "backend":
            return self._handle_backend(args)
        elif cmd == "mkdir":
            return self._handle_mkdir(args)
        elif cmd == "help":
            return self._handle_help(args)
        else:
            return f"Unknown command: {cmd}. Type 'help' for available commands."

    def _handle_ls(self, args):
        """Handle 'ls' command."""
        path = args[0] if args else "."
        recursive = "-r" in args

        try:
            files = self.file_manager.list_files(path, self.current_backend, recursive=recursive)
            return "\n".join(files)
        except Exception as e:
            return f"Error listing files: {e}"

    def _handle_cat(self, args):
        """Handle 'cat' command."""
        if not args:
            return "Usage: cat <file>"

        path = args[0]

        try:
            content = self.file_manager.load(path, self.current_backend)

            # If content is a tensor, convert it to text
            if isinstance(content, torch.Tensor):
                content = self.gpu_plugin.tensor_to_text(content)

            return content
        except Exception as e:
            return f"Error reading file: {e}"

    def _handle_tensorize(self, args):
        """Handle 'tensorize' command."""
        if len(args) < 2:
            return "Usage: tensorize <source_file> <target_file> [device]"

        source_file = args[0]
        target_file = args[1]
        device = args[2] if len(args) > 2 else None

        try:
            # Load from disk
            text = self.file_manager.load(source_file, "disk")

            # Convert to tensor
            tensor = self.gpu_plugin.text_to_tensor(text, device)

            # Save to memory
            self.file_manager.save(target_file, tensor, "memory")

            return f"Tensorized {source_file} to {target_file} on {tensor.device}"
        except Exception as e:
            return f"Error tensorizing file: {e}"

    def _handle_detensorize(self, args):
        """Handle 'detensorize' command."""
        if len(args) < 2:
            return "Usage: detensorize <source_file> <target_file>"

        source_file = args[0]
        target_file = args[1]

        try:
            # Load tensor from memory
            tensor = self.file_manager.load(source_file, "memory")

            # Convert to text
            text = self.gpu_plugin.tensor_to_text(tensor)

            # Save to disk
            self.file_manager.save(target_file, text, "disk")

            return f"Detensorized {source_file} to {target_file}"
        except Exception as e:
            return f"Error detensorizing file: {e}"

    def _handle_backend(self, args):
        """Handle 'backend' command."""
        if not args:
            return f"Current backend: {self.current_backend}"

        backend = args[0].lower()
        if backend in ["disk", "memory", "zarr"]:
            self.current_backend = backend
            return f"Backend set to {backend}"
        else:
            return f"Invalid backend: {backend}. Valid backends: disk, memory, zarr"

    def _handle_mkdir(self, args):
        """Handle 'mkdir' command."""
        if not args:
            return "Usage: mkdir <directory>"

        directory = args[0]

        try:
            self.file_manager.ensure_directory(directory, self.current_backend)
            return f"Created directory {directory}"
        except Exception as e:
            return f"Error creating directory: {e}"

    def _handle_help(self, args):
        """Handle 'help' command."""
        return """
Available commands:
  ls [path] [-r]        List files in directory (use -r for recursive)
  cat <file>            Display file content
  tensorize <src> <dst> [device]  Convert text file to tensor
  detensorize <src> <dst>         Convert tensor to text file
  backend [name]        Get or set current backend (disk, memory, zarr)
  mkdir <directory>     Create directory
  help                  Show this help
"""
```

### 4. SMA Integration

Integration with SMA's CLI:

```python
def add_tensor_fs_commands(subparsers, file_manager, gpu_plugin):
    """Add tensor filesystem commands to the SMA CLI."""
    # Tensor filesystem commands
    tensor_fs_parser = subparsers.add_parser(
        "tensor-fs",
        help="Tensor filesystem operations"
    )
    tensor_fs_subparsers = tensor_fs_parser.add_subparsers(
        dest="tensor_fs_command",
        help="Tensor filesystem command"
    )

    # List files
    list_parser = tensor_fs_subparsers.add_parser(
        "ls",
        help="List files"
    )
    list_parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Path to list files from"
    )
    list_parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="List files recursively"
    )
    list_parser.add_argument(
        "backend",
        choices=["disk", "memory", "zarr"],
        help="Backend to use"
    )

    # Cat file
    cat_parser = tensor_fs_subparsers.add_parser(
        "cat",
        help="Display file content"
    )
    cat_parser.add_argument(
        "path",
        help="Path to file"
    )
    cat_parser.add_argument(
        "backend",
        choices=["disk", "memory", "zarr"],
        help="Backend to use"
    )

    # Tensorize file
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
        "-d", "--device",
        default=None,
        help="Device to create tensor on"
    )

    # Detensorize file
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

    # Interactive shell
    shell_parser = tensor_fs_subparsers.add_parser(
        "shell",
        help="Start interactive tensor filesystem shell"
    )
```
```

## Implementation Plan

1. **Create the Storage Registry**:
   - Implement the `StorageRegistry` class
   - Add support for standard backends (DISK, MEMORY, ZARR)

2. **Implement the TensorFileManager**:
   - Extend the FileManager concept with tensor-specific operations
   - Maintain the same interface pattern (backend as last positional arg)
   - Add methods for converting between text and tensors

3. **Build the CLI Interface**:
   - Create a terminal-like interface for interacting with the tensor filesystem
   - Implement commands for common operations (ls, cat, tensorize, etc.)
   - Add help and documentation

4. **Integration with SMA**:
   - Add the tensor filesystem interface to the SMA CLI
   - Create convenience methods for common SMA operations

## Usage Examples

### Basic Usage

```python
# Create a registry with standard backends
registry = StorageRegistry.create_standard()

# Create a tensor file manager
tensor_fs = TensorFileManager(registry, base_path="/home/ts/code/projects")

# Load a file from disk and convert to tensor
tensor = tensor_fs.load_as_tensor("myfile.py", "DISK")

# Save the tensor to memory
tensor_fs.save("myfile.py", tensor, "MEMORY")

# Load the tensor from memory
tensor = tensor_fs.load("myfile.py", "MEMORY")

# Convert tensor back to text
text = tensor_fs.tensor_to_text(tensor)
```

### CLI Usage

```python
# Create the CLI
cli = TensorFSCLI(tensor_fs)

# List files in the current directory
cli.execute("ls")

# Change backend to MEMORY
cli.execute("backend MEMORY")

# List files in memory
cli.execute("ls")

# Tensorize a file
cli.execute("tensorize myfile.py myfile.tensor cuda")

# View tensor as text
cli.execute("cat myfile.tensor")
```

## Benefits

1. **Simplified Interface**: The interface is simple and consistent, with the backend always specified as the last positional argument.

2. **Seamless Abstraction**: The filepath never changes, only the backend enum changes, providing a complete abstraction of the storage backend.

3. **Terminal-like Interface**: The CLI provides a familiar terminal-like interface for agents to interact with.

4. **GPU Integration**: Tensors can be loaded directly to GPU memory for efficient processing.

## Conclusion

This tensor filesystem interface provides a seamless way to load text files from disk into tensors in memory, with a consistent API for accessing both disk and memory storage. The terminal-like interface makes it easy for agents to interact with the filesystem, simplifying the use of the tool.
