# Tensor Filesystem Interface - CLI Integration

## Overview

This plan outlines the CLI integration for the tensor filesystem interface, which leverages the existing FileManager to seamlessly load text files from disk into tensors stored in memory. The CLI will provide a terminal-like interface for interacting with the filesystem, maintaining the exact same paths regardless of the backend.

## CLI Components

### 1. Command-Line Interface

A simple CLI that directly maps to FileManager operations:

```python
# File: semantic_matrix_analyzer/semantic_matrix_analyzer/tensor_fs_cli.py

import torch
from pathlib import Path

class TensorFSCLI:
    """
    Command-line interface for tensor filesystem operations.
    
    This class provides a terminal-like interface for interacting with
    the filesystem and tensor operations, making it easy for agents to use.
    All commands map directly to FileManager operations with backend
    always as the last positional argument.
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
        
        # Command handlers - direct mapping to FileManager operations
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
        """Handle 'load' command - direct mapping to FileManager.load."""
        if len(args) < 2:
            return "Usage: load <path> <backend>"
        
        path = args[0]
        backend = args[1]
        
        try:
            content = self.file_manager.load(path, backend)
            
            # If content is a tensor and we're displaying to console, convert to text
            if isinstance(content, torch.Tensor):
                content = self.gpu_plugin.tensor_to_text(content)
            
            return content
        except Exception as e:
            return f"Error loading file: {e}"
    
    def _handle_save(self, args):
        """Handle 'save' command - direct mapping to FileManager.save."""
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
    
    def _handle_tensorize(self, args):
        """Handle 'tensorize' command - load from disk, convert to tensor, save to memory."""
        if len(args) < 1:
            return "Usage: tensorize <path> [device]"
        
        path = args[0]
        device = args[1] if len(args) > 1 else None
        
        try:
            # Load from disk
            text = self.file_manager.load(path, "disk")
            
            # Convert to tensor
            tensor = self.gpu_plugin.text_to_tensor(text, device)
            
            # Save to memory - same path
            self.file_manager.save(path, tensor, "memory")
            
            return f"Tensorized {path} on {tensor.device}"
        except Exception as e:
            return f"Error tensorizing file: {e}"
    
    def _handle_detensorize(self, args):
        """Handle 'detensorize' command - load from memory, convert to text, save to disk."""
        if len(args) < 1:
            return "Usage: detensorize <path>"
        
        path = args[0]
        
        try:
            # Load tensor from memory
            tensor = self.file_manager.load(path, "memory")
            
            # Convert to text
            text = self.gpu_plugin.tensor_to_text(tensor)
            
            # Save to disk - same path
            self.file_manager.save(path, text, "disk")
            
            return f"Detensorized {path}"
        except Exception as e:
            return f"Error detensorizing file: {e}"
    
    def _handle_backend(self, args):
        """Handle 'backend' command - get or set current backend."""
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
Available commands (direct mapping to FileManager operations):
  load <path> <backend>              Load a file
  save <path> <content> <backend>    Save content to a file
  list_files <dir> <backend>         List files in directory
  exists <path> <backend>            Check if a path exists
  ensure_directory <dir> <backend>   Ensure a directory exists
  delete <path> <backend>            Delete a file or directory
  copy <src> <dst> <backend>         Copy a file or directory
  move <src> <dst> <backend>         Move a file or directory
  
Special commands:
  tensorize <path> [device]          Load from disk, convert to tensor, save to memory
  detensorize <path>                 Load from memory, convert to text, save to disk
  backend [name]                     Get or set current backend
  help                               Show this help
  
Note: Backend is always the last positional argument in all commands.
      The same path is used regardless of the backend.
"""
```

### 2. SMA CLI Integration

Integration with SMA's CLI:

```python
# File: semantic_matrix_analyzer/sma_cli.py

def add_tensor_fs_commands(subparsers):
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
    
    # FileManager operations
    
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
    
    # Special commands
    
    # Tensorize file
    tensorize_parser = tensor_fs_subparsers.add_parser(
        "tensorize",
        help="Load from disk, convert to tensor, save to memory"
    )
    tensorize_parser.add_argument(
        "path",
        help="File path"
    )
    tensorize_parser.add_argument(
        "--device",
        default=None,
        help="Device to create tensor on"
    )
    
    # Detensorize file
    detensorize_parser = tensor_fs_subparsers.add_parser(
        "detensorize",
        help="Load from memory, convert to text, save to disk"
    )
    detensorize_parser.add_argument(
        "path",
        help="File path"
    )
    
    # Interactive shell
    shell_parser = tensor_fs_subparsers.add_parser(
        "shell",
        help="Start interactive tensor filesystem shell"
    )
```

### 3. Command Handlers

Handlers for the tensor filesystem commands:

```python
# File: semantic_matrix_analyzer/semantic_matrix_analyzer/tensor_fs_handlers.py

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
    
    # Handle commands
    if args.tensor_fs_command == "load":
        handle_tensor_fs_load(args, file_manager, gpu_plugin)
    elif args.tensor_fs_command == "save":
        handle_tensor_fs_save(args, file_manager)
    elif args.tensor_fs_command == "tensorize":
        handle_tensor_fs_tensorize(args, file_manager, gpu_plugin)
    elif args.tensor_fs_command == "detensorize":
        handle_tensor_fs_detensorize(args, file_manager, gpu_plugin)
    elif args.tensor_fs_command == "shell":
        handle_tensor_fs_shell(args, file_manager, gpu_plugin)
    else:
        print(f"Unknown tensor filesystem command: {args.tensor_fs_command}")

def handle_tensor_fs_tensorize(args, file_manager, gpu_plugin):
    """Handle 'tensorize' command."""
    try:
        # Load from disk
        text = file_manager.load(args.path, "disk")
        
        # Convert to tensor
        tensor = gpu_plugin.text_to_tensor(text, args.device)
        
        # Save to memory - same path
        file_manager.save(args.path, tensor, "memory")
        
        print(f"Tensorized {args.path} on {tensor.device}")
    except Exception as e:
        print(f"Error tensorizing file: {e}")

def handle_tensor_fs_detensorize(args, file_manager, gpu_plugin):
    """Handle 'detensorize' command."""
    try:
        # Load tensor from memory
        tensor = file_manager.load(args.path, "memory")
        
        # Convert to text
        text = gpu_plugin.tensor_to_text(tensor)
        
        # Save to disk - same path
        file_manager.save(args.path, text, "disk")
        
        print(f"Detensorized {args.path}")
    except Exception as e:
        print(f"Error detensorizing file: {e}")
```

## Key Concepts

1. **Direct Mapping to FileManager Operations**:
   - CLI commands map directly to FileManager operations
   - Backend is always the last positional argument
   - Same paths are used regardless of the backend

2. **Special Commands for Tensorization**:
   - `tensorize` command loads from disk, converts to tensor, saves to memory
   - `detensorize` command loads from memory, converts to text, saves to disk
   - Both commands use the same path for both backends

3. **Interactive Shell**:
   - Provides a terminal-like interface for interacting with the filesystem
   - All commands map directly to FileManager operations
   - Backend can be set as the default for the session

## Benefits

1. **Simplicity**:
   - CLI commands map directly to FileManager operations
   - No need for specialized commands for different backends
   - Same paths are used regardless of the backend

2. **Consistency**:
   - Backend is always the last positional argument
   - Same interface for all operations
   - Same paths across backends

3. **Ease of Use**:
   - Terminal-like interface for interacting with the filesystem
   - Special commands for common operations like tensorization
   - Interactive shell for exploratory work

## Conclusion

This CLI integration provides a simple and consistent way to interact with the tensor filesystem interface. By mapping directly to FileManager operations and using the same paths regardless of the backend, it creates an intuitive and easy-to-use interface for agents and humans alike.
