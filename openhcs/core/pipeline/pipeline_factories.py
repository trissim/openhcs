"""
Pipeline factory system for the OpenHCS pipeline architecture.

This module contains the AutoPipelineFactory class that creates pre-configured pipelines
for all common workflows, leveraging specialized steps to reduce boilerplate code.

The AutoPipelineFactory uses a unified approach that handles 2D multichannel, z-stack per plane stitch,
and z-stack projection stitch with a single implementation, simplifying the pipeline architecture.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Import all required modules with absolute imports
from openhcs.core.pipeline import Pipeline
# Import step classes from the appropriate modules
# This is a cleaner approach that avoids circular imports
# Import steps from the core steps module
from openhcs.core.steps import CompositeStep, FocusStep
from openhcs.core.steps import ImageAssemblyStep as ImageStitchingStep
from openhcs.core.steps import NormStep, PositionGenerationStep, ZFlatStep
from openhcs.io.virtual_path_factory import VirtualPathFactory


class AutoPipelineFactory:
    """
    Unified factory for creating pipelines for all common use cases.

    This factory handles all types of stitching workflows with a single implementation:
    - 2D multichannel stitching
    - Z-stack per plane stitching
    - Z-stack projection stitching

    It automatically configures the appropriate steps based on the input parameters,
    with no need to differentiate between different types of pipelines.
    """

    def __init__(
        self,
        input_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        normalize: bool = True,
        normalization_params: Optional[Dict[str, Any]] = None,
        flatten_z: bool = False,
        z_method: str = "max",
        channel_weights: Optional[Union[List[float], Dict[str, float]]] = None,
        well_filter: Optional[List[str]] = None,
    ):
        """
        Initialize with pipeline parameters.

        Args:
            input_dir: Input directory containing images
            output_dir: Output directory for stitched images
            normalize: Whether to include normalization
            normalization_params: Parameters for normalization
            well_filter: Wells to process
            flatten_z: Whether to flatten Z-stacks (if Z-stacks are present)
            z_method: Z-stack processing method:
                      - "max", "mean", "median" for projection methods
                      - "combined", "laplacian", "tenengrad" for focus detection methods
            channel_weights: Weights for channel compositing (for reference image only).
                           Should be a list with length equal to the number of channels.
        """
        # Convert input_dir to VirtualPath
        self.input_dir = VirtualPathFactory.from_path(input_dir)
        # Only use output_dir if explicitly provided, otherwise let the pipeline handle it
        self.output_dir = VirtualPathFactory.from_path(output_dir) if output_dir else None

        self.normalize = normalize
        # Default normalization parameters
        self.normalization_params = normalization_params or {
            'low_percentile': 1.0,
            'high_percentile': 99.0
        }
        self.well_filter = well_filter
        self.flatten_z = flatten_z
        self.z_method = z_method
        self.channel_weights = channel_weights

        # Determine if z_method is a focus method or projection method
        self.focus_methods = ["combined", "laplacian", "tenengrad", "normalized_variance", "fft"]
        self.is_focus_method = self.z_method in self.focus_methods

    def create_pipelines(self) -> List[Pipeline]:
        """
        Create pipeline configuration based on parameters.

        This method creates two pipelines:
        1. Position generation pipeline - Creates position files for stitching
        2. Image assembly pipeline - Stitches images using the position files
        """
        # Create position generation pipeline
        pos_pipeline = Pipeline(
            steps=[
                # Always include Z-flattening for position generation
                FunctionStep(
                    func=(create_projection, {'method': 'max_projection'}),
                    variable_components=['z_index'],
                    name="Z-Stack Flattening"
                ),

                # Include normalization if enabled
                FunctionStep(
                    func=(stack_percentile_normalize, self.normalization_params),
                    variable_components=['site'],
                    name="Image Normalization"
                ) if self.normalize else None,

                # Always include channel compositing for reference image
                FunctionStep(
                    func=(create_composite, {'weights': self.channel_weights}),
                    variable_components=['channel'],
                    name="Channel Compositing"
                ),

                # TODO: Replace with actual position generation function
                # PositionGenerationStep()
            ],
            name="Position Generation Pipeline"
        )

        # In the new architecture, steps don't have output_dir attributes
        # Instead, we need to define a positions directory for the ImageStitchingStep
        # Use VirtualPathFactory to create a VirtualPath for positions_dir
        input_dir_vp = VirtualPathFactory.from_path(self.input_dir)
        positions_dir = input_dir_vp.parent.joinpath(f"{input_dir_vp.name}_positions")

        # Create assembly pipeline
        assembly_pipeline = Pipeline(
            steps=[
                # Include normalization if enabled (create new instance)
                FunctionStep(
                    func=(stack_percentile_normalize, self.normalization_params),
                    variable_components=['site'],
                    name="Image Normalization"
                ) if self.normalize else None,

                # Include Z-flattening for assembly if enabled
                (FunctionStep(
                    func=deep_focus,
                    variable_components=['z_index'],
                    name="Focus Selection"
                ) if self.is_focus_method else
                FunctionStep(
                    func=(create_projection, {'method': f'{self.z_method}_projection'}),
                    variable_components=['z_index'],
                    name="Z-Stack Flattening"
                )) if self.flatten_z else None,

                # TODO: Replace with actual image stitching function
                # ImageStitchingStep(positions_dir=positions_dir)
            ],
            name="Image Assembly Pipeline"
        )

        # Return both pipelines
        return [pos_pipeline, assembly_pipeline]