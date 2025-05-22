"""
Batch Processing Module

This module provides batch processing capabilities for GPU-accelerated semantic analysis.
It allows processing multiple files simultaneously to optimize GPU utilization.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, Set

import torch
import numpy as np

from gpu_analysis.ast_tensor import GPUASTTensorizer
from gpu_analysis.analyzers.semantic_analyzer import SemanticAnalyzer

# Set up logging
logger = logging.getLogger(__name__)

class BatchProcessingManager:
    """
    Batch processing manager for GPU-accelerated analysis.

    This class manages batch processing of multiple files to optimize GPU utilization.

    Attributes:
        device: Device to place tensors on ("cuda" or "cpu")
        batch_size: Number of files to process in a batch
        semantic_analyzer: Semantic analyzer to use
    """

    def __init__(self, device: str = "cuda", batch_size: int = 10,
                config: Optional[Dict[str, Any]] = None):
        """
        Initialize the batch processing manager.

        Args:
            device: Device to place tensors on ("cuda" or "cpu")
            batch_size: Number of files to process in a batch
            config: Configuration dictionary
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.batch_size = batch_size
        self.config = config or {}

        # Initialize analyzers
        self.semantic_analyzer = SemanticAnalyzer(
            device=self.device,
            config=self.config)

    def process_batch(self, file_paths: List[Union[str, Path]],
                     analysis_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Process a batch of files using GPU acceleration.

        Args:
            file_paths: List of file paths to process
            analysis_types: Types of analysis to perform (if None, perform all)

        Returns:
            Dictionary mapping file paths to analysis results
        """
        # Split files into batches
        batches = self._split_into_batches(file_paths)

        # Process each batch
        results = {}
        for batch in batches:
            batch_results = self._process_single_batch(batch, analysis_types)
            results.update(batch_results)

        return results

    def _split_into_batches(self, file_paths: List[Union[str, Path]]) -> List[List[Union[str, Path]]]:
        """
        Split file paths into batches of appropriate size.

        Args:
            file_paths: List of file paths to split

        Returns:
            List of batches of file paths
        """
        return [file_paths[i:i+self.batch_size]
                for i in range(0, len(file_paths), self.batch_size)]

    def _process_single_batch(self, file_paths: List[Union[str, Path]],
                             analysis_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Process a single batch of files.

        Args:
            file_paths: List of file paths to process
            analysis_types: Types of analysis to perform (if None, perform all)

        Returns:
            Dictionary mapping file paths to analysis results
        """
        # Load all files in the batch
        files = {}
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    files[str(file_path)] = f.read()
            except Exception as e:
                logger.error(f"Error loading file {file_path}: {e}")
                files[str(file_path)] = ""

        # Perform analyses on the batch
        results = {}

        # Analyze each file
        for file_path, content in files.items():
            try:
                # Use the semantic analyzer to analyze the file
                file_results = self.semantic_analyzer.analyze(
                    content, file_path, analysis_types)
                results[file_path] = file_results
            except Exception as e:
                logger.error(f"Error analyzing {file_path}: {e}")
                results[file_path] = {"error": str(e)}

        return results

class DynamicBatchSizeManager:
    """
    Dynamic batch size manager.

    This class dynamically adjusts the batch size based on GPU memory usage.

    Attributes:
        device: Device to place tensors on ("cuda" or "cpu")
        batch_size: Current batch size
        min_batch_size: Minimum batch size
        max_batch_size: Maximum batch size
        target_memory_usage: Target memory usage as a fraction (0.0 to 1.0)
    """

    def __init__(self, device: str = "cuda", initial_batch_size: int = 10,
                min_batch_size: int = 1, max_batch_size: int = 100,
                target_memory_usage: float = 0.8):
        """
        Initialize the dynamic batch size manager.

        Args:
            device: Device to place tensors on ("cuda" or "cpu")
            initial_batch_size: Initial batch size
            min_batch_size: Minimum batch size
            max_batch_size: Maximum batch size
            target_memory_usage: Target memory usage as a fraction (0.0 to 1.0)
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.target_memory_usage = target_memory_usage

    def get_batch_size(self) -> int:
        """
        Get the current batch size.

        Returns:
            Current batch size
        """
        return self.batch_size

    def update_batch_size(self, memory_usage: float) -> int:
        """
        Update the batch size based on GPU memory usage.

        Args:
            memory_usage: Current GPU memory usage as a fraction (0.0 to 1.0)

        Returns:
            New batch size
        """
        if memory_usage > self.target_memory_usage * 1.1:
            # Memory usage is too high, decrease batch size
            self.batch_size = max(self.min_batch_size,
                                 int(self.batch_size * 0.8))
        elif memory_usage < self.target_memory_usage * 0.9:
            # Memory usage is low, increase batch size
            self.batch_size = min(self.max_batch_size,
                                 int(self.batch_size * 1.2))

        return self.batch_size

    def get_memory_usage(self) -> float:
        """
        Get current GPU memory usage as a fraction.

        Returns:
            Current GPU memory usage as a fraction (0.0 to 1.0)
        """
        if self.device != "cuda" or not torch.cuda.is_available():
            return 0.0

        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)

        if reserved == 0:
            return 0.0

        return allocated / reserved

class GPUAcceleratedAnalysis:
    """
    Main integration module for GPU-accelerated analysis.

    This class provides a unified interface for GPU-accelerated analysis
    that integrates with SMA's existing systems.

    Attributes:
        config: Configuration dictionary
        batch_manager: Batch processing manager
        batch_size_manager: Dynamic batch size manager
        semantic_analyzer: Semantic analyzer
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the GPU-accelerated analysis module.

        Args:
            config: Configuration dictionary
        """
        # Create GPU configuration manager
        self.config = config or {}

        # Get device from config
        self.device = self.config.get("gpu.device", "cuda")
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"

        # Create batch processing manager
        self.batch_manager = BatchProcessingManager(
            device=self.device,
            batch_size=self.config.get("gpu.batch.initial_size", 10),
            config=self.config
        )

        # Create dynamic batch size manager
        self.batch_size_manager = DynamicBatchSizeManager(
            device=self.device,
            initial_batch_size=self.config.get("gpu.batch.initial_size", 10),
            min_batch_size=self.config.get("gpu.batch.min_size", 1),
            max_batch_size=self.config.get("gpu.batch.max_size", 100),
            target_memory_usage=self.config.get("gpu.memory.max_fraction", 0.9) * 0.9
        )

        # Create semantic analyzer
        self.semantic_analyzer = SemanticAnalyzer(
            device=self.device,
            config=self.config
        )

    def analyze(self, code: str, file_path: Optional[Union[str, Path]] = None,
               analysis_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze a single file using GPU acceleration.

        Args:
            code: Python code to analyze
            file_path: Optional path to the file
            analysis_types: Types of analysis to perform (if None, perform all)

        Returns:
            Dictionary of analysis results
        """
        return self.semantic_analyzer.analyze(code, file_path, analysis_types)

    def analyze_batch(self, file_paths: List[Union[str, Path]],
                     analysis_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze a batch of files using GPU acceleration.

        Args:
            file_paths: List of file paths to analyze
            analysis_types: Types of analysis to perform (if None, perform all)

        Returns:
            Dictionary mapping file paths to analysis results
        """
        # Update batch size based on current memory usage
        if self.config.get("gpu.batch.dynamic_sizing", True):
            memory_usage = self.batch_size_manager.get_memory_usage()
            new_batch_size = self.batch_size_manager.update_batch_size(memory_usage)
            self.batch_manager.batch_size = new_batch_size

        # Process batch
        return self.batch_manager.process_batch(file_paths, analysis_types)

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get GPU memory usage statistics.

        Returns:
            Dictionary of memory usage statistics
        """
        if self.device != "cuda" or not torch.cuda.is_available():
            return {"allocated": 0, "reserved": 0, "fraction": 0}

        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)

        return {
            "allocated": allocated,
            "reserved": reserved,
            "fraction": allocated / reserved if reserved > 0 else 0
        }
