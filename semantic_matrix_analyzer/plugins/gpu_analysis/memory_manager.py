"""
GPU Memory Management Module

This module provides classes for managing GPU memory allocation and deallocation,
ensuring that data is kept in GPU memory as much as possible while avoiding
out-of-memory errors.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, Set

import torch
import numpy as np

# Set up logging
logger = logging.getLogger(__name__)

class GPUMemoryManager:
    """
    GPU memory manager.
    
    This class manages GPU memory allocation and deallocation, ensuring that
    data is kept in GPU memory as much as possible while avoiding out-of-memory errors.
    
    Attributes:
        device: Device to place tensors on ("cuda" or "cpu")
        max_memory_fraction: Maximum fraction of GPU memory to use
        cache_size: Maximum number of items to cache in GPU memory
    """
    
    def __init__(self, device: str = "cuda", max_memory_fraction: float = 0.9, cache_size: int = 100):
        """
        Initialize the GPU memory manager.
        
        Args:
            device: Device to place tensors on ("cuda" or "cpu")
            max_memory_fraction: Maximum fraction of GPU memory to use
            cache_size: Maximum number of items to cache in GPU memory
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.max_memory_fraction = max_memory_fraction
        self.cache_size = cache_size
        self._cache = {}  # Dict[str, torch.Tensor]
        self._usage = {}  # Dict[str, int]  # Usage count for each tensor
        self._lru = []  # List[str]  # Least recently used keys
        
    def allocate(self, key: str, data: Any) -> Any:
        """
        Allocate GPU memory for data.
        
        Args:
            key: Unique key for the data
            data: Data to allocate memory for
            
        Returns:
            Data in GPU memory
        """
        # Check if data is already in cache
        if key in self._cache:
            # Update usage count and LRU
            self._usage[key] += 1
            self._lru.remove(key)
            self._lru.append(key)
            return self._cache[key]
        
        # Check if we need to free memory
        if len(self._cache) >= self.cache_size:
            self._free_memory()
        
        # Convert data to tensor if it's not already
        if not isinstance(data, torch.Tensor) and not isinstance(data, dict):
            data = self._to_tensor(data)
        elif isinstance(data, dict):
            # Handle dictionary of tensors
            data = {k: self._to_tensor(v) if not isinstance(v, torch.Tensor) else v for k, v in data.items()}
        
        # Move tensor to GPU
        if isinstance(data, torch.Tensor):
            data = data.to(self.device)
        elif isinstance(data, dict):
            data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        
        # Store data in cache
        self._cache[key] = data
        self._usage[key] = 1
        self._lru.append(key)
        
        return data
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get data from GPU memory.
        
        Args:
            key: Unique key for the data
            
        Returns:
            Data from GPU memory, or None if not found
        """
        if key not in self._cache:
            return None
        
        # Update usage count and LRU
        self._usage[key] += 1
        self._lru.remove(key)
        self._lru.append(key)
        
        return self._cache[key]
    
    def release(self, key: str) -> None:
        """
        Release GPU memory for data.
        
        Args:
            key: Unique key for the data
        """
        if key not in self._cache:
            return
        
        # Decrement usage count
        self._usage[key] -= 1
        
        # If usage count is 0, remove from cache
        if self._usage[key] <= 0:
            del self._cache[key]
            del self._usage[key]
            self._lru.remove(key)
    
    def clear(self) -> None:
        """Clear all GPU memory."""
        self._cache.clear()
        self._usage.clear()
        self._lru.clear()
        
        # Force CUDA to release memory
        if self.device == "cuda":
            torch.cuda.empty_cache()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get GPU memory usage statistics.
        
        Returns:
            Dictionary of memory usage statistics
        """
        if self.device != "cuda":
            return {"allocated": 0, "reserved": 0, "max_reserved": 0}
            
        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        max_reserved = torch.cuda.max_memory_reserved(self.device)
        
        return {
            "allocated": allocated / (1024 ** 3),  # GB
            "reserved": reserved / (1024 ** 3),  # GB
            "max_reserved": max_reserved / (1024 ** 3),  # GB
        }
    
    def _free_memory(self) -> None:
        """Free memory by removing least recently used items."""
        # Remove items until we're under the cache size limit
        while len(self._cache) >= self.cache_size and self._lru:
            # Get the least recently used key
            lru_key = self._lru.pop(0)
            
            # Remove from cache
            del self._cache[lru_key]
            del self._usage[lru_key]
    
    def _to_tensor(self, data: Any) -> torch.Tensor:
        """
        Convert data to a PyTorch tensor.
        
        Args:
            data: Data to convert
            
        Returns:
            PyTorch tensor
        """
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(self.device)
        elif isinstance(data, list):
            if all(isinstance(x, (int, float)) for x in data):
                return torch.tensor(data, device=self.device)
            else:
                return [self._to_tensor(x) for x in data]
        elif isinstance(data, dict):
            return {k: self._to_tensor(v) for k, v in data.items()}
        elif isinstance(data, (int, float)):
            return torch.tensor([data], device=self.device)
        elif isinstance(data, str):
            # Convert string to tensor of character codes
            return torch.tensor([ord(c) for c in data], dtype=torch.int32, device=self.device)
        else:
            # Try to convert to tensor, or return as is if not possible
            try:
                return torch.tensor(data, device=self.device)
            except:
                return data

class GPUMemoryContext:
    """
    Context manager for GPU memory operations.
    
    This class provides a context manager for GPU memory operations,
    ensuring that data is kept in GPU memory within the context and
    properly released when the context exits.
    
    Attributes:
        manager: GPU memory manager to use
    """
    
    def __init__(self, manager: GPUMemoryManager):
        """
        Initialize the GPU memory context.
        
        Args:
            manager: GPU memory manager to use
        """
        self.manager = manager
        self.keys = []
        
    def __enter__(self):
        """Enter the context."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context."""
        # Release all allocated memory
        for key in self.keys:
            self.manager.release(key)
        
    def allocate(self, key: str, data: Any) -> Any:
        """
        Allocate GPU memory for data within the context.
        
        Args:
            key: Unique key for the data
            data: Data to allocate memory for
            
        Returns:
            Data in GPU memory
        """
        result = self.manager.allocate(key, data)
        self.keys.append(key)
        return result
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get data from GPU memory within the context.
        
        Args:
            key: Unique key for the data
            
        Returns:
            Data from GPU memory, or None if not found
        """
        return self.manager.get(key)

class GPUStorageBackend:
    """
    GPU storage backend.
    
    This class provides a storage backend that stores data directly in GPU memory.
    It implements the same interface as the storage backends in OpenHCS.
    
    Attributes:
        device: Device to place tensors on ("cuda" or "cpu")
        memory_manager: GPU memory manager to use
    """
    
    def __init__(self, device: str = "cuda", memory_manager: Optional[GPUMemoryManager] = None):
        """
        Initialize the GPU storage backend.
        
        Args:
            device: Device to place tensors on ("cuda" or "cpu")
            memory_manager: GPU memory manager to use
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.memory_manager = memory_manager or GPUMemoryManager(device=self.device)
        self._prefixes = set()  # Declared directory-like namespaces
    
    def _normalize(self, path: Union[str, Path]) -> str:
        """
        Normalize a path to a string key.
        
        Args:
            path: Path to normalize
            
        Returns:
            Normalized path as a string
        """
        return str(Path(path).as_posix())
    
    def load(self, file_path: Union[str, Path], **kwargs) -> Any:
        """
        Load data from GPU memory.
        
        Args:
            file_path: Path to the file
            **kwargs: Additional arguments
            
        Returns:
            Data from GPU memory
        """
        key = self._normalize(file_path)
        
        data = self.memory_manager.get(key)
        if data is None:
            raise FileNotFoundError(f"GPU key '{key}' not found.")
        
        return data
    
    def save(self, data: Any, output_path: Union[str, Path], **kwargs) -> None:
        """
        Save data to GPU memory.
        
        Args:
            data: Data to save
            output_path: Path to save to
            **kwargs: Additional arguments
        """
        key = self._normalize(output_path)
        
        # Ensure parent directories exist
        parent = Path(output_path).parent
        parent_key = self._normalize(parent)
        self._prefixes.add(parent_key)
        
        # Save data to GPU memory
        self.memory_manager.allocate(key, data)
    
    def exists(self, path: Union[str, Path]) -> bool:
        """
        Check if a path exists in GPU memory.
        
        Args:
            path: Path to check
            
        Returns:
            True if the path exists, False otherwise
        """
        key = self._normalize(path)
        
        # Check if the key exists in the cache
        if self.memory_manager.get(key) is not None:
            return True
        
        # Check if the key is a prefix
        if key in self._prefixes:
            return True
        
        # Check if the key is a parent of any existing key
        for prefix in self._prefixes:
            if prefix.startswith(key):
                return True
        
        return False
    
    def is_file(self, path: Union[str, Path]) -> bool:
        """
        Check if a path is a file in GPU memory.
        
        Args:
            path: Path to check
            
        Returns:
            True if the path is a file, False otherwise
        """
        key = self._normalize(path)
        
        # A path is a file if it exists in the cache
        return self.memory_manager.get(key) is not None
    
    def is_dir(self, path: Union[str, Path]) -> bool:
        """
        Check if a path is a directory in GPU memory.
        
        Args:
            path: Path to check
            
        Returns:
            True if the path is a directory, False otherwise
        """
        key = self._normalize(path)
        
        # A path is a directory if it's a prefix
        if key in self._prefixes:
            return True
        
        # A path is a directory if it's a parent of any existing key
        for prefix in self._prefixes:
            if prefix.startswith(key):
                return True
        
        return False
