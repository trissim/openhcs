# Visualizing Tensorized Text Data

This document explains how to visualize text data that has been tensorized and stored in memory using the file manager system.

## Overview

When working with the GPU Analysis module, text data (such as source code) is often converted to tensor representations for efficient processing on the GPU. This document describes methods to render these tensors back to human-readable text for visualization and debugging purposes.

## Basic Tensor-to-Text Conversion

The most straightforward approach is to convert the tensor values back to characters:

```python
def tensor_to_text(tensor):
    """
    Convert a tensor containing character codes back to text.
    
    Args:
        tensor: PyTorch tensor containing character codes (e.g., ASCII/Unicode)
        
    Returns:
        String representation of the tensor
    """
    # Move to CPU if on GPU
    if tensor.device != "cpu":
        tensor = tensor.cpu()
        
    # Convert each integer back to a character and join
    return ''.join(chr(int(code)) for code in tensor.numpy())
```

## Zero-Copy Approach with DLPack

For more efficient memory usage, you can use DLPack to share tensor data between PyTorch and NumPy without copying:

```python
def render_tensor_dlpack(tensor):
    """
    Render a tensor to text using DLPack for zero-copy data sharing.
    
    Args:
        tensor: PyTorch tensor containing character codes
        
    Returns:
        String representation of the tensor
    """
    import torch
    from torch.utils.dlpack import to_dlpack
    import numpy as np
    
    # Get DLPack capsule
    dlpack = to_dlpack(tensor)
    
    # Convert to NumPy array without copying data
    numpy_array = np.from_dlpack(dlpack)
    
    # Convert to characters
    text = ''.join(chr(int(code)) for code in numpy_array)
    return text
```

## Integration with File Manager

To render a tensorized file stored in the file manager:

```python
def render_tensorized_file(file_path):
    """
    Render a tensorized file stored in the file manager back to text.
    
    Args:
        file_path: Path to the tensorized file (e.g., /home/ts/myfile.py)
        
    Returns:
        The rendered text content
    """
    from openhcs.io.base import FileManager
    import torch
    
    # Get file manager instance
    file_manager = FileManager.get_instance()
    
    # Load the tensorized file
    tensor = file_manager.load(file_path)
    
    # Check if it's actually a tensor
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"File at {file_path} is not a tensor")
    
    # Render the tensor to text
    return tensor_to_text(tensor)
```

## Streaming Approach for Large Files

For large files, a streaming approach can be more memory-efficient:

```python
def render_tensorized_file_streaming(file_path, chunk_size=10000):
    """
    Stream-render a large tensorized file with minimal memory overhead.
    
    Args:
        file_path: Path to the tensorized file
        chunk_size: Number of characters to process at once
        
    Returns:
        Generator yielding chunks of rendered text
    """
    from openhcs.io.base import FileManager
    import torch
    
    file_manager = FileManager.get_instance()
    tensor = file_manager.load(file_path)
    
    # Process in chunks to avoid large memory allocations
    total_size = tensor.size(0)
    for i in range(0, total_size, chunk_size):
        end = min(i + chunk_size, total_size)
        chunk = tensor[i:end]
        
        # Convert chunk to characters
        if chunk.device != "cpu":
            chunk = chunk.cpu()
        text_chunk = ''.join(chr(int(code)) for code in chunk.numpy())
        
        yield text_chunk
```

## Display Options

### Terminal Display

```python
def display_tensorized_file(file_path):
    """Display a tensorized file in the terminal."""
    text = render_tensorized_file(file_path)
    print(text)
```

### GUI Display

```python
def display_in_gui(file_path):
    """Display a tensorized file in a GUI text widget."""
    import tkinter as tk
    from tkinter import scrolledtext
    
    root = tk.Tk()
    root.title(f"Tensor Viewer: {file_path}")
    
    text_widget = scrolledtext.ScrolledText(root, wrap=tk.WORD)
    text_widget.pack(expand=True, fill='both')
    
    # For large files, use streaming to avoid memory issues
    for chunk in render_tensorized_file_streaming(file_path):
        text_widget.insert(tk.END, chunk)
    
    root.mainloop()
```

## Advanced Visualization

For more advanced visualization of tensorized text, consider:

### Syntax Highlighting

```python
def display_with_syntax_highlighting(file_path, language="python"):
    """
    Display tensorized code with syntax highlighting.
    
    Args:
        file_path: Path to the tensorized file
        language: Programming language for syntax highlighting
    """
    from pygments import highlight
    from pygments.lexers import get_lexer_by_name
    from pygments.formatters import TerminalFormatter
    
    text = render_tensorized_file(file_path)
    lexer = get_lexer_by_name(language)
    formatter = TerminalFormatter()
    
    highlighted_text = highlight(text, lexer, formatter)
    print(highlighted_text)
```

### Diff Visualization

```python
def display_tensor_diff(file_path1, file_path2):
    """
    Display diff between two tensorized files.
    
    Args:
        file_path1: Path to the first tensorized file
        file_path2: Path to the second tensorized file
    """
    import difflib
    
    text1 = render_tensorized_file(file_path1)
    text2 = render_tensorized_file(file_path2)
    
    lines1 = text1.splitlines()
    lines2 = text2.splitlines()
    
    diff = difflib.unified_diff(lines1, lines2, 
                               fromfile=file_path1,
                               tofile=file_path2)
    
    print('\n'.join(diff))
```

## Performance Considerations

1. **Memory Usage**: For large files, the streaming approach is recommended to avoid loading the entire file into memory at once.

2. **GPU Memory**: If the tensor is on the GPU, moving it to the CPU for rendering can be expensive. Consider processing in chunks to minimize GPU-CPU transfers.

3. **Zero-Copy**: While DLPack provides zero-copy tensor sharing between frameworks, the conversion from numeric values to characters will still require some memory allocation.

4. **Custom CUDA Kernels**: For extremely large files or real-time visualization, consider implementing custom CUDA kernels for direct rendering on the GPU.

## Integration with SMA

To integrate tensor visualization with the Semantic Matrix Analyzer:

```python
def visualize_sma_tensor(component_name, tensor_name):
    """
    Visualize a tensor from the SMA's memory.
    
    Args:
        component_name: Name of the SMA component
        tensor_name: Name of the tensor to visualize
    """
    from semantic_matrix_analyzer.core import get_component
    
    component = get_component(component_name)
    tensor = component.get_tensor(tensor_name)
    
    # Render and display the tensor
    text = tensor_to_text(tensor)
    print(f"Tensor: {tensor_name} from {component_name}")
    print("-" * 40)
    print(text)
```

## Conclusion

Visualizing tensorized text data is essential for debugging and understanding the GPU Analysis module's operations. The methods described in this document provide efficient ways to render tensors back to human-readable text, with options for both simple rendering and more advanced visualization techniques.
