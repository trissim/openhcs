from __future__ import annotations 

import logging
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

class FocusAnalyzer:
    """
    Provides focus metrics and best focus selection.

    This class implements various focus measure algorithms and methods to find
    the best focused image in a Z-stack. All methods are static and do not require
    an instance.

    Usage:
        The class provides several focus metrics and methods to select the best focused
        image from a Z-stack. The main methods are:

        - find_best_focus: Returns the index of the best focused image in a stack
        - select_best_focus: Returns the best focused image from a stack
        - compute_focus_metrics: Computes focus metrics for all images in a stack

        For most use cases, it's recommended to use the select_best_focus method
        with an explicit metric parameter:

        ```python
        best_image, best_idx, scores = FocusAnalyzer.select_best_focus(
            image_stack,
            metric="laplacian"  # Explicitly specify the metric
        )
        ```

        Available metrics:
        - "combined": Uses a weighted combination of all metrics (default)
        - "normalized_variance" or "nvar": Uses normalized variance
        - "laplacian" or "lap": Uses Laplacian energy
        - "tenengrad" or "ten": Uses Tenengrad variance
        - "fft": Uses adaptive FFT focus measure

        You can also provide a custom weights dictionary for the combined metric:

        ```python
        custom_weights = {'nvar': 0.4, 'lap': 0.4, 'ten': 0.1, 'fft': 0.1}
        best_image, best_idx, scores = FocusAnalyzer.select_best_focus(
            image_stack,
            metric=custom_weights
        )
        ```
    """

    # Default weights for the combined focus measure.
    # These weights are used when no custom weights are provided to the
    # combined_focus_measure method or when using the "combined" metric
    # with find_best_focus, select_best_focus, or compute_focus_metrics.
    # The weights determine the contribution of each focus metric to the final score:
    DEFAULT_WEIGHTS = {
        'nvar': 0.3,  # Normalized variance (robust to illumination changes)
        'lap': 0.3,   # Laplacian energy (sensitive to edges)
        'ten': 0.2,   # Tenengrad variance (based on gradient magnitude)
        'fft': 0.2    # FFT-based focus (frequency domain analysis)
    }

    @staticmethod
    def normalized_variance(img: np.ndarray) -> float:
        """
        Normalized variance focus measure.
        Robust to illumination changes.

        Args:
            img: Input grayscale image

        Returns:
            Focus quality score
        """
        mean_val = np.mean(img)
        if mean_val == 0:  # Avoid division by zero
            return 0

        return np.var(img) / mean_val

    @staticmethod
    def laplacian_energy(img: np.ndarray, ksize: int = 3) -> float:
        """
        Laplacian energy focus measure.
        Sensitive to edges and high-frequency content.

        Args:
            img: Input grayscale image
            ksize: Kernel size for Laplacian

        Returns:
            Focus quality score
        """
        lap = cv2.Laplacian(img, cv2.CV_64F, ksize=ksize)
        return np.mean(np.square(lap))

    @staticmethod
    def tenengrad_variance(img: np.ndarray, ksize: int = 3, threshold: float = 0) -> float:
        """
        Tenengrad variance focus measure.
        Based on gradient magnitude.

        Args:
            img: Input grayscale image
            ksize: Kernel size for Sobel operator
            threshold: Threshold for gradient magnitude

        Returns:
            Focus quality score
        """
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
        fm = gx**2 + gy**2
        fm[fm < threshold] = 0  # Thresholding to reduce noise impact

        return np.mean(fm)

    @staticmethod
    def adaptive_fft_focus(img: np.ndarray) -> float:
        """
        Adaptive FFT focus measure optimized for low-contrast microscopy images.
        Uses image statistics to set threshold adaptively.

        Args:
            img: Input grayscale image

        Returns:
            Focus quality score
        """
        # Apply FFT
        fft = np.fft.fft2(img)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)

        # Calculate image statistics for adaptive thresholding
        # Only img_std is used for thresholding
        img_std = np.std(img)

        # Adaptive threshold based on image statistics
        threshold_factor = max(0.1, min(1.0, img_std / 50.0))
        threshold = np.max(magnitude) * threshold_factor

        # Count frequency components above threshold
        high_freq_count = np.sum(magnitude > threshold)

        # Normalize by image size
        score = high_freq_count / (img.shape[0] * img.shape[1])

        return score

    @staticmethod
    def combined_focus_measure(
        img: np.ndarray,
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Combined focus measure using multiple metrics.
        Optimized for microscopy images, especially low-contrast specimens.

        This method combines multiple focus metrics (normalized variance, Laplacian energy,
        Tenengrad variance, and FFT-based focus) using weighted averaging. The weights
        determine the contribution of each metric to the final score.

        Args:
            img: Input grayscale image
            weights: Weights for each metric as a dictionary with keys 'nvar', 'lap', 'ten', 'fft'.
                     If None, uses DEFAULT_WEIGHTS (nvar=0.3, lap=0.3, ten=0.2, fft=0.2).
                     Provide custom weights when you want to emphasize specific focus characteristics:
                     - Increase 'nvar' weight for better performance with illumination variations
                     - Increase 'lap' weight for better edge detection
                     - Increase 'ten' weight for better gradient-based focus
                     - Increase 'fft' weight for better frequency domain analysis

        Returns:
            Combined focus quality score (higher values indicate better focus)

        Example:
            ```python
            # Use default weights
            score = FocusAnalyzer.combined_focus_measure(image)

            # Use custom weights to emphasize edges
            custom_weights = {'nvar': 0.2, 'lap': 0.5, 'ten': 0.2, 'fft': 0.1}
            score = FocusAnalyzer.combined_focus_measure(image, weights=custom_weights)
            ```
        """
        # Use provided weights or defaults
        if weights is None:
            weights = FocusAnalyzer.DEFAULT_WEIGHTS

        # Calculate individual metrics
        nvar = FocusAnalyzer.normalized_variance(img)
        lap = FocusAnalyzer.laplacian_energy(img)
        ten = FocusAnalyzer.tenengrad_variance(img)
        fft = FocusAnalyzer.adaptive_fft_focus(img)

        # Weighted combination
        score = (
            weights.get('nvar', 0.3) * nvar +
            weights.get('lap', 0.3) * lap +
            weights.get('ten', 0.2) * ten +
            weights.get('fft', 0.2) * fft
        )

        return score

    @staticmethod
    def _get_focus_function(metric: Union[str, Dict[str, float]]):
        """
        Get the appropriate focus measure function based on metric.

        Args:
            metric: Focus detection method name or weights dictionary
                   If string: "combined", "normalized_variance", "laplacian", "tenengrad", "fft"
                   If dict: Weights for combined focus measure

        Returns:
            callable: The focus measure function and any additional arguments

        Raises:
            ValueError: If the method is unknown
        """
        # If metric is a dictionary, use it as weights for combined focus measure
        if isinstance(metric, dict):
            return lambda img: FocusAnalyzer.combined_focus_measure(img, metric)

        # Otherwise, treat it as a string method name
        if metric == 'combined':
            return FocusAnalyzer.combined_focus_measure
        if metric in ('nvar', 'normalized_variance'):
            return FocusAnalyzer.normalized_variance
        if metric in ('lap', 'laplacian'):
            return FocusAnalyzer.laplacian_energy
        if metric in ('ten', 'tenengrad'):
            return FocusAnalyzer.tenengrad_variance
        if metric == 'fft':
            return FocusAnalyzer.adaptive_fft_focus

        # If we get here, the metric is unknown
        raise ValueError(f"Unknown focus method: {metric}")

    @staticmethod
    def find_best_focus(
        image_stack: np.ndarray, # Changed from List[np.ndarray] to np.ndarray (Z, H, W)
        metric: Union[str, Dict[str, float]] = "combined"
    ) -> Tuple[int, List[Tuple[int, float]]]:
        """
        Find the best focused image in a 3D stack using specified method.

        Args:
            image_stack: 3D NumPy array of shape (Z, H, W).
            metric: Focus detection method or weights dictionary
                   If string: "combined", "normalized_variance", "laplacian", "tenengrad", "fft"
                   If dict: Weights for combined focus measure

        Returns:
            Tuple of (best_focus_index, focus_scores)
        """
        if not isinstance(image_stack, np.ndarray) or image_stack.ndim != 3:
            raise TypeError("image_stack must be a 3D NumPy ndarray of shape (Z, H, W).")
        
        focus_scores = []
        focus_func = FocusAnalyzer._get_focus_function(metric)

        for i in range(image_stack.shape[0]): # Iterate over Z dimension
            img_slice = image_stack[i, :, :]
            score = focus_func(img_slice)
            focus_scores.append((i, score))
        
        if not focus_scores: # Should not happen if image_stack is not empty
             raise ValueError("Could not compute focus scores, image_stack might be empty or invalid.")

        best_focus_idx = max(focus_scores, key=lambda x: x[1])[0]
        return best_focus_idx, focus_scores

    @staticmethod
    def select_best_focus(
        image_stack: np.ndarray, # Changed from List[np.ndarray] to np.ndarray (Z, H, W)
        metric: Union[str, Dict[str, float]] = "combined"
    ) -> Tuple[np.ndarray, int, List[Tuple[int, float]]]: # Return best image as (1,H,W)
        """
        Select the best focus plane from a 3D stack of images.

        Args:
            image_stack: 3D NumPy array of shape (Z, H, W).
            metric: Focus detection method or weights dictionary
                   If string: "combined", "normalized_variance", "laplacian", "tenengrad", "fft"
                   If dict: Weights for combined focus measure

        Returns:
            Tuple of (best_focus_image (1,H,W), best_focus_index, focus_scores)
        """
        best_idx, scores = FocusAnalyzer.find_best_focus(image_stack, metric)
        best_image_slice = image_stack[best_idx, :, :]
        # Return as a 3D array with a single Z-slice
        return best_image_slice.reshape(1, best_image_slice.shape[0], best_image_slice.shape[1]), best_idx, scores

    @staticmethod
    def compute_focus_metrics(image_stack: np.ndarray, # Changed from List[np.ndarray]
                             metric: Union[str, Dict[str, float]] = "combined") -> List[float]:
        """
        Compute focus metrics for a 3D stack of images.

        Args:
            image_stack: 3D NumPy array of shape (Z, H, W).
            metric: Focus detection method or weights dictionary
                   If string: "combined", "normalized_variance", "laplacian", "tenengrad", "fft"
                   If dict: Weights for combined focus measure

        Returns:
            List of focus scores for each image slice
        """
        if not isinstance(image_stack, np.ndarray) or image_stack.ndim != 3:
            raise TypeError("image_stack must be a 3D NumPy ndarray of shape (Z, H, W).")

        focus_scores = []
        focus_func = FocusAnalyzer._get_focus_function(metric)

        for i in range(image_stack.shape[0]): # Iterate over Z dimension
            img_slice = image_stack[i, :, :]
            score = focus_func(img_slice)
            focus_scores.append(score)

        return focus_scores
