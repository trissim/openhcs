# Tensor Filesystem Implementation Plan

## Overview

This document provides a detailed implementation plan for the Tensor Filesystem Interface, which leverages the existing FileManager to seamlessly load text files from disk into tensors stored in memory. The implementation will follow the design principles outlined in the main plan document, maintaining the exact same interface as FileManager with backend always as the last positional argument.

## Implementation Details

### 1. GPU Plugin Extensions

First, we need to extend the GPU Analysis Plugin to handle text-to-tensor conversion:

```python
# File: brain/gpu_analysis/plugin.py

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

### 2. CLI Interface Implementation

Next, we'll implement the CLI interface for interacting with the tensor filesystem, maintaining the exact same interface as FileManager:

```python
# File: semantic_matrix_analyzer/semantic_matrix_analyzer/tensor_fs_cli.py

import torch
from pathlib import Path

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

    def _handle_load(self, args):
        """Handle 'load' command with exact FileManager interface."""
        if len(args) < 2:
            return "Usage: load <path> <backend>"

        path = args[0]
        backend = args[1]

        try:
            content = self.file_manager.load(path, backend)

            # If content is a tensor, convert it to text for display
            if isinstance(content, torch.Tensor):
                content = self.gpu_plugin.tensor_to_text(content)

            return content
        except Exception as e:
            return f"Error loading file: {e}"

    def _handle_save(self, args):
        """Handle 'save' command with exact FileManager interface."""
        if len(args) < 3:
            return "Usage: save <path> <content> <backend>"

        path = args[0]
        content = args[1]
        backend = args[2]

        try:
            self.file_manager.save(path, content, backend)
            return f"Saved to {path} using {backend} backend"
        except Exception as e:
            return f"Error saving file: {e}"

    def _handle_list_files(self, args):
        """Handle 'list_files' command with exact FileManager interface."""
        if len(args) < 2:
            return "Usage: list_files <directory> <backend> [pattern] [recursive]"

        directory = args[0]
        backend = args[1]
        pattern = args[2] if len(args) > 2 else None
        recursive = args[3].lower() == "true" if len(args) > 3 else False

        try:
            files = self.file_manager.list_files(directory, backend, pattern=pattern, recursive=recursive)
            return "\n".join(files)
        except Exception as e:
            return f"Error listing files: {e}"

    def _handle_exists(self, args):
        """Handle 'exists' command with exact FileManager interface."""
        if len(args) < 2:
            return "Usage: exists <path> <backend>"

        path = args[0]
        backend = args[1]

        try:
            exists = self.file_manager.exists(path, backend)
            return f"Path {path} {'exists' if exists else 'does not exist'}"
        except Exception as e:
            return f"Error checking if path exists: {e}"

    def _handle_ensure_directory(self, args):
        """Handle 'ensure_directory' command with exact FileManager interface."""
        if len(args) < 2:
            return "Usage: ensure_directory <directory> <backend>"

        directory = args[0]
        backend = args[1]

        try:
            path = self.file_manager.ensure_directory(directory, backend)
            return f"Directory {path} ensured"
        except Exception as e:
            return f"Error ensuring directory: {e}"

    def _handle_delete(self, args):
        """Handle 'delete' command with exact FileManager interface."""
        if len(args) < 2:
            return "Usage: delete <path> <backend>"

        path = args[0]
        backend = args[1]

        try:
            self.file_manager.delete(path, backend)
            return f"Deleted {path}"
        except Exception as e:
            return f"Error deleting path: {e}"

    def _handle_copy(self, args):
        """Handle 'copy' command with exact FileManager interface."""
        if len(args) < 3:
            return "Usage: copy <source_path> <dest_path> <backend>"

        source_path = args[0]
        dest_path = args[1]
        backend = args[2]

        try:
            self.file_manager.copy(source_path, dest_path, backend)
            return f"Copied {source_path} to {dest_path}"
        except Exception as e:
            return f"Error copying file: {e}"

    def _handle_move(self, args):
        """Handle 'move' command with exact FileManager interface."""
        if len(args) < 3:
            return "Usage: move <source_path> <dest_path> <backend>"

        source_path = args[0]
        dest_path = args[1]
        backend = args[2]

        try:
            self.file_manager.move(source_path, dest_path, backend)
            return f"Moved {source_path} to {dest_path}"
        except Exception as e:
            return f"Error moving file: {e}"

    def _handle_tensorize(self, args):
        """Handle 'tensorize' command while maintaining FileManager interface."""
        if len(args) < 3:
            return "Usage: tensorize <source_file> <destination_file> <backend> [device]"

        source_file = args[0]
        destination_file = args[1]
        backend = args[2]
        device = args[3] if len(args) > 3 else None

        try:
            # Load from disk
            text = self.file_manager.load(source_file, "disk")

            # Convert to tensor
            tensor = self.gpu_plugin.text_to_tensor(text, device)

            # Save to specified backend
            self.file_manager.save(destination_file, tensor, backend)

            return f"Tensorized {source_file} to {destination_file} on {tensor.device}"
        except Exception as e:
            return f"Error tensorizing file: {e}"

    def _handle_detensorize(self, args):
        """Handle 'detensorize' command while maintaining FileManager interface."""
        if len(args) < 3:
            return "Usage: detensorize <source_file> <destination_file> <backend>"

        source_file = args[0]
        destination_file = args[1]
        backend = args[2]

        try:
            # Load tensor from memory
            tensor = self.file_manager.load(source_file, "memory")

            # Convert to text
            text = self.gpu_plugin.tensor_to_text(tensor)

            # Save to specified backend
            self.file_manager.save(destination_file, text, backend)

            return f"Detensorized {source_file} to {destination_file}"
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

    def _handle_help(self, args):
        """Handle 'help' command."""
        return """
Available commands (all maintain the exact FileManager interface):
  load <path> <backend>                    Load a file
  save <path> <content> <backend>          Save content to a file
  list_files <dir> <backend> [pattern] [recursive]  List files in directory
  exists <path> <backend>                  Check if a path exists
  ensure_directory <dir> <backend>         Ensure a directory exists
  delete <path> <backend>                  Delete a file or directory
  copy <src> <dst> <backend>               Copy a file or directory
  move <src> <dst> <backend>               Move a file or directory
  tensorize <src> <dst> <backend> [device] Convert text file to tensor
  detensorize <src> <dst> <backend>        Convert tensor to text file
  backend [name]                           Get or set current backend
  help                                     Show this help

Note: Backend is always the last positional argument in all commands.
"""
```

### 3. SMA CLI Integration

Next, we'll integrate the tensor filesystem with SMA's CLI, maintaining the exact same interface as FileManager:

```python
# File: semantic_matrix_analyzer/sma_cli.py

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
        help="Backend to use (last positional argument)"
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
        help="Backend to use (last positional argument)"
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
        help="Backend to use (last positional argument)"
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
        help="Backend to use (last positional argument)"
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
        help="Backend to use (last positional argument)"
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
        help="Backend to use (last positional argument)"
    )

    # Copy file
    copy_parser = tensor_fs_subparsers.add_parser(
        "copy",
        help="Copy a file"
    )
    copy_parser.add_argument(
        "source_path",
        help="Source path"
    )
    copy_parser.add_argument(
        "dest_path",
        help="Destination path"
    )
    copy_parser.add_argument(
        "backend",
        choices=["disk", "memory", "zarr"],
        help="Backend to use (last positional argument)"
    )

    # Move file
    move_parser = tensor_fs_subparsers.add_parser(
        "move",
        help="Move a file"
    )
    move_parser.add_argument(
        "source_path",
        help="Source path"
    )
    move_parser.add_argument(
        "dest_path",
        help="Destination path"
    )
    move_parser.add_argument(
        "backend",
        choices=["disk", "memory", "zarr"],
        help="Backend to use (last positional argument)"
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
        help="Backend to save tensor to (last positional argument)"
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
        help="Backend to save text to (last positional argument)"
    )

    # Interactive shell
    shell_parser = tensor_fs_subparsers.add_parser(
        "shell",
        help="Start interactive tensor filesystem shell"
    )
    shell_parser.add_argument(
        "--note",
        default="All commands maintain the exact same interface as FileManager with backend always as the last positional argument",
        help=argparse.SUPPRESS
    )
```

### 4. Command Handlers Implementation

Now, we'll implement the command handlers for the tensor filesystem commands, maintaining the exact same interface as FileManager:

```python
# File: semantic_matrix_analyzer/semantic_matrix_analyzer/tensor_fs_handlers.py

import torch
import argparse
from pathlib import Path

def get_gpu_plugin():
    """Get the GPU plugin for tensorization."""
    from semantic_matrix_analyzer.plugins import plugin_manager
    return plugin_manager.get_plugin("gpu_analysis")

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
    elif args.tensor_fs_command == "copy":
        handle_tensor_fs_copy(args, file_manager)
    elif args.tensor_fs_command == "move":
        handle_tensor_fs_move(args, file_manager)
    elif args.tensor_fs_command == "tensorize":
        handle_tensor_fs_tensorize(args, file_manager, gpu_plugin)
    elif args.tensor_fs_command == "detensorize":
        handle_tensor_fs_detensorize(args, file_manager, gpu_plugin)
    elif args.tensor_fs_command == "shell":
        handle_tensor_fs_shell(args, file_manager, gpu_plugin)
    else:
        print(f"Unknown tensor filesystem command: {args.tensor_fs_command}")

def handle_tensor_fs_load(args, file_manager, gpu_plugin):
    """Handle 'load' command with exact FileManager interface."""
    try:
        content = file_manager.load(args.path, args.backend)

        # If content is a tensor and we're displaying to console, convert to text
        if isinstance(content, torch.Tensor):
            content = gpu_plugin.tensor_to_text(content)

        print(content)
    except Exception as e:
        print(f"Error loading file: {e}")

def handle_tensor_fs_save(args, file_manager):
    """Handle 'save' command with exact FileManager interface."""
    try:
        content = args.content
        if content == '-':
            # Read from stdin
            import sys
            content = sys.stdin.read()

        file_manager.save(args.path, content, args.backend)
        print(f"Saved to {args.path} using {args.backend} backend")
    except Exception as e:
        print(f"Error saving file: {e}")

def handle_tensor_fs_list_files(args, file_manager):
    """Handle 'list_files' command with exact FileManager interface."""
    try:
        pattern = args.pattern if hasattr(args, 'pattern') else None
        recursive = args.recursive if hasattr(args, 'recursive') else False

        files = file_manager.list_files(args.directory, args.backend, pattern=pattern, recursive=recursive)
        for file in files:
            print(file)
    except Exception as e:
        print(f"Error listing files: {e}")

def handle_tensor_fs_exists(args, file_manager):
    """Handle 'exists' command with exact FileManager interface."""
    try:
        exists = file_manager.exists(args.path, args.backend)
        print(f"Path {args.path} {'exists' if exists else 'does not exist'}")
    except Exception as e:
        print(f"Error checking if path exists: {e}")

def handle_tensor_fs_ensure_directory(args, file_manager):
    """Handle 'ensure_directory' command with exact FileManager interface."""
    try:
        path = file_manager.ensure_directory(args.directory, args.backend)
        print(f"Directory {path} ensured")
    except Exception as e:
        print(f"Error ensuring directory: {e}")

def handle_tensor_fs_delete(args, file_manager):
    """Handle 'delete' command with exact FileManager interface."""
    try:
        file_manager.delete(args.path, args.backend)
        print(f"Deleted {args.path}")
    except Exception as e:
        print(f"Error deleting path: {e}")

def handle_tensor_fs_copy(args, file_manager):
    """Handle 'copy' command with exact FileManager interface."""
    try:
        file_manager.copy(args.source_path, args.dest_path, args.backend)
        print(f"Copied {args.source_path} to {args.dest_path}")
    except Exception as e:
        print(f"Error copying file: {e}")

def handle_tensor_fs_move(args, file_manager):
    """Handle 'move' command with exact FileManager interface."""
    try:
        file_manager.move(args.source_path, args.dest_path, args.backend)
        print(f"Moved {args.source_path} to {args.dest_path}")
    except Exception as e:
        print(f"Error moving file: {e}")

def handle_tensor_fs_tensorize(args, file_manager, gpu_plugin):
    """Handle 'tensorize' command while maintaining FileManager interface."""
    try:
        # Load from disk
        text = file_manager.load(args.source, "disk")

        # Convert to tensor
        tensor = gpu_plugin.text_to_tensor(text, args.device)

        # Save to specified backend
        file_manager.save(args.destination, tensor, args.backend)

        print(f"Tensorized {args.source} to {args.destination} on {tensor.device}")
    except Exception as e:
        print(f"Error tensorizing file: {e}")

def handle_tensor_fs_detensorize(args, file_manager, gpu_plugin):
    """Handle 'detensorize' command while maintaining FileManager interface."""
    try:
        # Load tensor from memory
        tensor = file_manager.load(args.source, "memory")

        # Convert to text
        text = gpu_plugin.tensor_to_text(tensor)

        # Save to specified backend
        file_manager.save(args.destination, text, args.backend)

        print(f"Detensorized {args.source} to {args.destination}")
    except Exception as e:
        print(f"Error detensorizing file: {e}")

def handle_tensor_fs_shell(args, file_manager, gpu_plugin):
    """Handle 'shell' command."""
    from semantic_matrix_analyzer.tensor_fs_cli import TensorFSCLI

    cli = TensorFSCLI(file_manager, gpu_plugin)

    print("Tensor Filesystem Shell")
    print("Type 'help' for available commands, 'exit' to quit")
    print("All commands maintain the exact same interface as FileManager")
    print("with backend always as the last positional argument")

    while True:
        try:
            command = input("tensor-fs> ")
            if command.lower() in ["exit", "quit"]:
                break

            result = cli.execute(command)
            print(result)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
```

## Integration Steps

1. Add the GPU plugin extensions to the GPU Analysis Plugin
2. Create the TensorFSCLI class in the SMA codebase
3. Add the tensor filesystem commands to the SMA CLI
4. Implement the command handlers for the tensor filesystem commands
5. Update the SMA main module to register the tensor filesystem commands

## Testing Plan

1. **Unit Tests**:
   - Test text_to_tensor and tensor_to_text methods
   - Test TensorFSCLI command handlers
   - Test command-line argument parsing

2. **Integration Tests**:
   - Test end-to-end workflow from disk to memory and back
   - Test GPU acceleration with different devices
   - Test error handling and edge cases

3. **Manual Testing**:
   - Test interactive shell functionality
   - Test with different file types and sizes
   - Test with different backends

## Dependencies

- openhcs.io.filemanager
- openhcs.io.disk
- openhcs.io.memory
- openhcs.io.zarr (optional)
- torch
- semantic_matrix_analyzer.plugins

## Timeline

- Week 1: Implement GPU plugin extensions and TensorFSCLI
- Week 2: Implement SMA CLI integration and command handlers
- Week 3: Testing and refinement

## Conclusion

This implementation plan provides a detailed roadmap for implementing the Tensor Filesystem Interface. By following this plan, we can create a seamless interface that allows agents to interact with the filesystem and process code files as tensors using a familiar terminal-like interface.
