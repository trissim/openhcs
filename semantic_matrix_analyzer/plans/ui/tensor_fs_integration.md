# Tensor Filesystem Integration with SMA

## Overview

This plan outlines how to integrate the existing FileManager with the Semantic Matrix Analyzer (SMA) and GPU plugin to enable seamless loading and processing of code files as tensors. This integration will facilitate the use of the tool by agents interacting with humans by simplifying the interface to a terminal-like experience.

## Integration Components

### 1. SMA CLI Extension

Extend the SMA CLI to include tensor filesystem commands:

```python
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

### 2. Command Handlers

Implement handlers for the tensor filesystem commands:

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

    # Handle commands
    if args.tensor_fs_command == "ls":
        handle_tensor_fs_ls(args, file_manager)
    elif args.tensor_fs_command == "cat":
        handle_tensor_fs_cat(args, file_manager, gpu_plugin)
    elif args.tensor_fs_command == "tensorize":
        handle_tensor_fs_tensorize(args, file_manager, gpu_plugin)
    elif args.tensor_fs_command == "detensorize":
        handle_tensor_fs_detensorize(args, file_manager, gpu_plugin)
    elif args.tensor_fs_command == "shell":
        handle_tensor_fs_shell(args, file_manager, gpu_plugin)
    else:
        print(f"Unknown tensor filesystem command: {args.tensor_fs_command}")

def get_gpu_plugin():
    """Get the GPU plugin for tensorization."""
    # This is a placeholder for getting the GPU plugin
    # The actual implementation would depend on how the GPU plugin is registered
    from semantic_matrix_analyzer.plugins import plugin_manager
    return plugin_manager.get_plugin("gpu_analysis")

def handle_tensor_fs_ls(args, file_manager):
    """Handle 'ls' command."""
    try:
        files = file_manager.list_files(args.path, args.backend, recursive=args.recursive)
        for file in files:
            print(file)
    except Exception as e:
        print(f"Error listing files: {e}")

def handle_tensor_fs_cat(args, file_manager, gpu_plugin):
    """Handle 'cat' command."""
    try:
        content = file_manager.load(args.path, args.backend)

        # If content is a tensor, convert it to text
        if isinstance(content, torch.Tensor):
            content = gpu_plugin.tensor_to_text(content)

        print(content)
    except Exception as e:
        print(f"Error reading file: {e}")

def handle_tensor_fs_tensorize(args, file_manager, gpu_plugin):
    """Handle 'tensorize' command."""
    try:
        # Load from disk
        text = file_manager.load(args.source, "disk")

        # Convert to tensor
        tensor = gpu_plugin.text_to_tensor(text, args.device)

        # Save to memory
        file_manager.save(args.destination, tensor, "memory")

        print(f"Tensorized {args.source} to {args.destination} on {tensor.device}")
    except Exception as e:
        print(f"Error tensorizing file: {e}")

def handle_tensor_fs_detensorize(args, file_manager, gpu_plugin):
    """Handle 'detensorize' command."""
    try:
        # Load tensor from memory
        tensor = file_manager.load(args.source, "memory")

        # Convert to text
        text = gpu_plugin.tensor_to_text(tensor)

        # Save to disk
        file_manager.save(args.destination, text, "disk")

        print(f"Detensorized {args.source} to {args.destination}")
    except Exception as e:
        print(f"Error detensorizing file: {e}")

def handle_tensor_fs_shell(args, file_manager, gpu_plugin):
    """Handle 'shell' command."""
    from tensor_fs_interface import TensorFSCLI

    cli = TensorFSCLI(file_manager, gpu_plugin)

    print("Tensor Filesystem Shell")
    print("Type 'help' for available commands, 'exit' to quit")

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

### 3. Integration with SMA Core

Integrate the tensor filesystem with SMA's core functionality:

```python
class SMAWithTensorFS:
    """
    SMA with integrated tensor filesystem.

    This class extends SMA with tensor filesystem capabilities,
    allowing seamless loading and processing of code files as tensors.
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

        # Create file manager
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
        # Load from disk
        text = self.file_manager.load(file_path, "disk")

        # Convert to tensor
        tensor = self.gpu_plugin.text_to_tensor(text, device)

        # Save tensor to memory for analysis
        memory_path = f"analysis/{Path(file_path).name}"
        self.file_manager.save(memory_path, tensor, "memory")

        # Analyze the code
        # This is a placeholder for SMA's analysis capabilities
        # The actual implementation would depend on SMA's API
        results = self.sma.analyze_tensor(tensor)

        return results

    def batch_analyze_directory(self, directory, recursive=False, device=None):
        """
        Batch analyze all code files in a directory.

        Args:
            directory: Directory containing code files
            recursive: Whether to analyze files recursively
            device: Device to load tensors to

        Returns:
            Dictionary of analysis results by file path
        """
        # List all files
        files = self.file_manager.list_files(directory, "disk", recursive=recursive)

        # Filter for code files
        code_files = [f for f in files if Path(f).suffix in [".py", ".js", ".java", ".cpp", ".c", ".h", ".hpp"]]

        # Analyze each file
        results = {}
        for file_path in code_files:
            results[file_path] = self.analyze_code_as_tensor(file_path, device)

        return results
```

### 4. GPU Plugin Extension

Extend the GPU plugin to handle text-to-tensor conversion:

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

## Usage Examples

### Command-Line Usage

```bash
# List files in a directory
sma tensor-fs ls /path/to/directory disk

# Tensorize a file
sma tensor-fs tensorize /path/to/file.py /file.tensor -d cuda

# Start interactive shell
sma tensor-fs shell
```

### Programmatic Usage

```python
# Create SMA with tensor filesystem
sma = SMAWithTensorFS()

# Load a file from disk
text = sma.file_manager.load("/path/to/file.py", "disk")

# Convert to tensor
tensor = sma.gpu_plugin.text_to_tensor(text, device="cuda")

# Save tensor to memory
sma.file_manager.save("/memory/file.tensor", tensor, "memory")

# Analyze a single file
results = sma.analyze_code_as_tensor("/path/to/file.py", device="cuda")

# Batch analyze a directory
results = sma.batch_analyze_directory("/path/to/directory", recursive=True, device="cuda")
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

## Conclusion

This integration provides a clean and efficient way to leverage the existing FileManager and GPU plugin for tensor operations. By maintaining a clear separation of responsibilities, we ensure that each component focuses on its core functionality:

- FileManager handles file operations with different backends
- GPU plugin handles tensorization and analysis
- SMA provides the overall framework and CLI

This approach simplifies the interface for agents while maintaining the architectural integrity of the system.
