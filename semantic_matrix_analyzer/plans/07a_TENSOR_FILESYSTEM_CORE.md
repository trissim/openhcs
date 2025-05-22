# Tensor Filesystem Interface - Core Concept

## Overview

This plan outlines the core concept of using the existing FileManager directly to seamlessly load text files from disk into tensors stored in memory. The interface will maintain the exact same paths regardless of the backend, with the backend specified as the last positional argument in all operations.

## Design Principles

1. **Backend Agnosticism**: The filepath never changes; only the backend enum (disk, memory, zarr) changes as the last positional argument.
2. **Seamless Abstraction**: Complete abstraction of the backend to a single positional argument enum.
3. **Caller Responsibility**: It is the responsibility of whoever calls load to know what to do with the data they've been given.
4. **Separation of Responsibilities**: Use the existing FileManager for file operations and the GPU plugin for tensorization.

## Core Components

### 1. Direct Use of FileManager

We'll use the existing FileManager directly with its exact interface:

```python
from openhcs.io.filemanager import FileManager
from openhcs.io.disk import DiskBackend
from openhcs.io.memory import MemoryBackend
import torch

# Create a registry with standard backends
registry = {
    "disk": DiskBackend,
    "memory": MemoryBackend
}

# Create a file manager
file_manager = FileManager(registry)

# Example usage:
# Load a file from disk
text_content = file_manager.load("myfile.py", "disk")

# Convert to tensor using GPU plugin
tensor = gpu_plugin.text_to_tensor(text_content)

# Save tensor to memory - same path as the original file
file_manager.save("myfile.py", tensor, "memory")

# Load tensor from memory - same path as the original file
tensor = file_manager.load("myfile.py", "memory")

# Convert tensor back to text
text = gpu_plugin.tensor_to_text(tensor)

# List files - works the same regardless of backend
files = file_manager.list_files("/path/to/dir", "disk")
files = file_manager.list_files("/path/to/dir", "memory")  # Same paths, different backend

# Check if file exists - works the same regardless of backend
exists = file_manager.exists("myfile.py", "disk")
exists = file_manager.exists("myfile.py", "memory")  # Same path, different backend

# All operations maintain the exact same interface as FileManager
# with backend always as the last positional argument
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

### 3. SMA Integration

Integration with SMA's core functionality:

```python
class SMA:
    """Semantic Matrix Analyzer."""
    
    def __init__(self):
        """Initialize SMA."""
        # Initialize file manager
        from openhcs.io.filemanager import FileManager
        from openhcs.io.disk import DiskBackend
        from openhcs.io.memory import MemoryBackend
        
        # Create registry with standard backends
        registry = {
            "disk": DiskBackend,
            "memory": MemoryBackend
        }
        
        # Create file manager - direct use without subclassing
        self.file_manager = FileManager(registry)
        
        # Get GPU plugin for tensorization
        self.gpu_plugin = self.plugin_manager.get_plugin("gpu_analysis")
    
    def analyze_file(self, file_path, backend="disk"):
        """
        Analyze a file.
        
        Args:
            file_path: Path to the file
            backend: Backend to use (last positional argument)
            
        Returns:
            Analysis results
        """
        # Load file
        content = self.file_manager.load(file_path, backend)
        
        # If content is already a tensor (from memory backend), use it directly
        if isinstance(content, torch.Tensor):
            tensor = content
        else:
            # Convert to tensor
            tensor = self.gpu_plugin.text_to_tensor(content)
            
            # Cache the tensor in memory for future use - same path
            self.file_manager.save(file_path, tensor, "memory")
        
        # Analyze the tensor
        results = self._analyze_tensor(tensor)
        
        return results
```

## Key Concepts

1. **Same Paths Across Backends**:
   - The filepath never changes regardless of the backend
   - A file can be loaded from disk as text and saved to memory as a tensor using the same path
   - This simplifies the interface and makes it more intuitive

2. **Direct Use of FileManager**:
   - No need to subclass or extend FileManager
   - Use the existing FileManager directly with its exact interface
   - Backend is always the last positional argument

3. **Caller Responsibility**:
   - It is the responsibility of whoever calls load to know what to do with the data they receive
   - If loading from disk, expect text
   - If loading from memory, expect a tensor
   - The caller can check the type and handle accordingly

4. **GPU Plugin for Tensorization**:
   - The GPU plugin handles the conversion between text and tensors
   - This maintains a clean separation of responsibilities
   - FileManager handles file operations, GPU plugin handles tensorization

## Benefits

1. **Simplicity**:
   - No need for specialized methods like load_as_tensor or save_tensor
   - No need for a TensorFileManager class
   - Just use the existing FileManager directly

2. **Consistency**:
   - Same paths across backends
   - Same interface for all operations
   - Backend always as the last positional argument

3. **Separation of Responsibilities**:
   - FileManager handles file operations
   - GPU plugin handles tensorization
   - SMA handles analysis

4. **Flexibility**:
   - Can easily switch between backends
   - Can easily add new backends
   - Can easily add new operations

## Conclusion

This approach provides a clean and simple way to use the existing FileManager for tensor operations. By maintaining the same paths across backends and letting the caller handle the data they receive, we create a more intuitive and flexible interface.
