Focus Analyzer
==============

.. module:: ezstitcher.core.focus_analyzer

This module contains the FocusAnalyzer class for analyzing focus quality in microscopy images.

FocusAnalyzer
------------

.. py:class:: FocusAnalyzer

   Provides focus metrics and best focus selection.

   This class implements various focus measure algorithms and methods to find
   the best focused image in a Z-stack. All methods are static and do not require
   an instance.

   .. py:attribute:: DEFAULT_WEIGHTS
      :type: dict
      :value: {'nvar': 0.3, 'lap': 0.3, 'ten': 0.2, 'fft': 0.2}

      Default weights for combined focus measure.

   .. py:staticmethod:: normalized_variance(img)

      Normalized variance focus measure.
      Robust to illumination changes.

      :param img: Input grayscale image
      :type img: numpy.ndarray
      :return: Focus quality score
      :rtype: float

   .. py:staticmethod:: laplacian_energy(img, ksize=3)

      Laplacian energy focus measure.
      Sensitive to edges and high-frequency content.

      :param img: Input grayscale image
      :type img: numpy.ndarray
      :param ksize: Kernel size for Laplacian
      :type ksize: int
      :return: Focus quality score
      :rtype: float

   .. py:staticmethod:: tenengrad_variance(img, ksize=3, threshold=0)

      Tenengrad variance focus measure.
      Based on gradient magnitude.

      :param img: Input grayscale image
      :type img: numpy.ndarray
      :param ksize: Kernel size for Sobel operator
      :type ksize: int
      :param threshold: Threshold for gradient magnitude
      :type threshold: float
      :return: Focus quality score
      :rtype: float

   .. py:staticmethod:: adaptive_fft_focus(img)

      Adaptive FFT focus measure optimized for low-contrast microscopy images.
      Uses image statistics to set threshold adaptively.

      :param img: Input grayscale image
      :type img: numpy.ndarray
      :return: Focus quality score
      :rtype: float

   .. py:staticmethod:: combined_focus_measure(img, weights=None)

      Combined focus measure using multiple metrics.
      Optimized for microscopy images, especially low-contrast specimens.

      :param img: Input grayscale image
      :type img: numpy.ndarray
      :param weights: Weights for each metric. If None, uses DEFAULT_WEIGHTS.
      :type weights: dict, optional
      :return: Combined focus quality score
      :rtype: float

   .. py:staticmethod:: find_best_focus(image_stack, metric="combined")

      Find the best focused image in a stack using specified method.

      :param image_stack: List of images
      :type image_stack: list
      :param metric: Focus detection method or weights dictionary. Options: "combined", "normalized_variance", "laplacian", "tenengrad", "fft" or a dictionary of weights.
      :type metric: str or dict
      :return: Tuple of (best_focus_index, focus_scores)
      :rtype: tuple

   .. py:staticmethod:: select_best_focus(image_stack, metric="combined")

      Select the best focus plane from a stack of images.

      :param image_stack: List of images
      :type image_stack: list
      :param metric: Focus detection method or weights dictionary. Options: "combined", "normalized_variance", "laplacian", "tenengrad", "fft" or a dictionary of weights.
      :type metric: str or dict
      :return: Tuple of (best_focus_image, best_focus_index, focus_scores)
      :rtype: tuple

   .. py:staticmethod:: compute_focus_metrics(image_stack, metric="combined")

      Compute focus metrics for a stack of images.

      :param image_stack: List of images
      :type image_stack: list
      :param metric: Focus detection method or weights dictionary. Options: "combined", "normalized_variance", "laplacian", "tenengrad", "fft" or a dictionary of weights.
      :type metric: str or dict
      :return: List of focus scores for each image
      :rtype: list

   .. py:staticmethod:: _get_focus_function(metric)

      Get the appropriate focus measure function based on metric.

      :param metric: Focus detection method or weights dictionary. Options: "combined", "normalized_variance", "laplacian", "tenengrad", "fft" or a dictionary of weights.
      :type metric: str or dict
      :return: The focus measure function
      :rtype: callable
      :raises: ValueError: If the method is unknown
