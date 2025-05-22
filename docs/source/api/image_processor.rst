Image Processor
==============

.. module:: ezstitcher.core.image_processor

This module contains the ImageProcessor class for handling image normalization, filtering, and compositing.

.. important::
   For a comprehensive guide to all image processing operations, including:

   * Detailed explanations of each operation
   * Examples of how to use operations in pipelines
   * Common use cases and best practices
   * Function handling patterns

   See :doc:`image_processing_operations`.

   The documentation below provides the API reference for the ImageProcessor class, while
   :doc:`image_processing_operations` provides more user-friendly documentation focused on
   practical usage.

ImageProcessor
------------

.. py:class:: ImageProcessor

   Handles image normalization, filtering, and compositing.
   All methods are static and do not require an instance.

   .. py:staticmethod:: preprocess(image, channel, preprocessing_funcs=None)

      Apply preprocessing to a single image for a given channel.

      :param image: Input image
      :type image: numpy.ndarray
      :param channel: Channel identifier
      :type channel: str
      :param preprocessing_funcs: Dictionary mapping channels to preprocessing functions
      :type preprocessing_funcs: dict, optional
      :return: Processed image
      :rtype: numpy.ndarray

   .. py:staticmethod:: blur(image, sigma=1)

      Apply Gaussian blur to an image.

      :param image: Input image
      :type image: numpy.ndarray
      :param sigma: Standard deviation for Gaussian kernel
      :type sigma: float
      :return: Blurred image
      :rtype: numpy.ndarray

   .. py:staticmethod:: sharpen(image, radius=1, amount=1.0)

      Sharpen an image using unsharp masking.

      :param image: Input image
      :type image: numpy.ndarray
      :param radius: Radius of Gaussian blur
      :type radius: float
      :param amount: Sharpening strength
      :type amount: float
      :return: Sharpened image
      :rtype: numpy.ndarray

   .. py:staticmethod:: normalize(image, target_min=0, target_max=65535)

      Normalize image to specified range.

      :param image: Input image
      :type image: numpy.ndarray
      :param target_min: Target minimum value
      :type target_min: int
      :param target_max: Target maximum value
      :type target_max: int
      :return: Normalized image
      :rtype: numpy.ndarray

   .. py:staticmethod:: percentile_normalize(image, low_percentile=1, high_percentile=99, target_min=0, target_max=65535)

      Normalize image using percentile-based contrast stretching.

      :param image: Input image
      :type image: numpy.ndarray
      :param low_percentile: Lower percentile (0-100)
      :type low_percentile: float
      :param high_percentile: Upper percentile (0-100)
      :type high_percentile: float
      :param target_min: Target minimum value
      :type target_min: int
      :param target_max: Target maximum value
      :type target_max: int
      :return: Normalized image
      :rtype: numpy.ndarray

   .. py:staticmethod:: stack_percentile_normalize(stack, low_percentile=1, high_percentile=99, target_min=0, target_max=65535)

      Normalize a stack of images using global percentile-based contrast stretching.
      This ensures consistent normalization across all images in the stack.

      For examples and common use cases, see :ref:`operation-normalize` in :doc:`image_processing_operations`.

      :param stack: Stack of images
      :type stack: list or numpy.ndarray
      :param low_percentile: Lower percentile (0-100)
      :type low_percentile: float
      :param high_percentile: Upper percentile (0-100)
      :type high_percentile: float
      :param target_min: Target minimum value
      :type target_min: int
      :param target_max: Target maximum value
      :type target_max: int
      :return: Normalized stack of images
      :rtype: numpy.ndarray

   .. py:staticmethod:: create_composite(images, weights=None)

      Create a grayscale composite image from multiple channels.

      :param images: Dictionary mapping channel names to images or list of images
      :type images: dict or list
      :param weights: Optional dictionary with weights for each channel or list of weights
      :type weights: dict or list, optional
      :return: Grayscale composite image (16-bit)
      :rtype: numpy.ndarray

   .. py:staticmethod:: apply_mask(image, mask)

      Apply a mask to an image.

      :param image: Input image
      :type image: numpy.ndarray
      :param mask: Mask image (same shape as input)
      :type mask: numpy.ndarray
      :return: Masked image
      :rtype: numpy.ndarray

   .. py:staticmethod:: create_weight_mask(shape, margin_ratio=0.1)

      Create a weight mask for blending images.

      :param shape: Shape of the mask (height, width)
      :type shape: tuple
      :param margin_ratio: Ratio of image size to use as margin
      :type margin_ratio: float
      :return: Weight mask
      :rtype: numpy.ndarray

   .. py:staticmethod:: max_projection(stack)

      Create a maximum intensity projection from a Z-stack.

      :param stack: Stack of images
      :type stack: list or numpy.ndarray
      :return: Maximum intensity projection
      :rtype: numpy.ndarray

   .. py:staticmethod:: mean_projection(stack)

      Create a mean intensity projection from a Z-stack.

      :param stack: Stack of images
      :type stack: list or numpy.ndarray
      :return: Mean intensity projection
      :rtype: numpy.ndarray

   .. py:staticmethod:: stack_equalize_histogram(stack, bins=65536, range_min=0, range_max=65535)

      Apply true histogram equalization to an entire stack of images.
      This ensures consistent contrast enhancement across all images in the stack.

      :param stack: Stack of images
      :type stack: list or numpy.ndarray
      :param bins: Number of bins for histogram computation
      :type bins: int
      :param range_min: Minimum value for histogram range
      :type range_min: int
      :param range_max: Maximum value for histogram range
      :type range_max: int
      :return: Histogram-equalized stack of images
      :rtype: numpy.ndarray

   .. py:staticmethod:: create_projection(stack, method="max_projection", focus_analyzer=None)

      Create a projection from a stack using the specified method.

      :param stack: List of images
      :type stack: list
      :param method: Projection method (max_projection, mean_projection, best_focus)
      :type method: str
      :param focus_analyzer: Focus analyzer for best_focus method
      :type focus_analyzer: FocusAnalyzer, optional
      :return: Projected image
      :rtype: numpy.ndarray

   .. py:staticmethod:: tophat(image, selem_radius=50, downsample_factor=4)

      Apply white top-hat transform to an image.

      :param image: Input image
      :type image: numpy.ndarray
      :param selem_radius: Radius of structuring element
      :type selem_radius: int
      :param downsample_factor: Factor to downsample image for faster processing
      :type downsample_factor: int
      :return: Top-hat transformed image
      :rtype: numpy.ndarray

