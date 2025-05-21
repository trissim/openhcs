"""Core module for openhcs."""

from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.steps.abstract import AbstractStep as Step

# Import configuration classes
#from openhcs.core.config import (
#    StitcherConfig,
#    PipelineConfig
#)

# Import pipeline factory class
#from openhcs.pipeline.factories.pipeline_factories import AutoPipelineFactory

__all__ = [
    'ProcessingContext',
    #'ImageProcessor',  # Pure utility class for image processing
    #'FocusAnalyzer',
    # 'Stitcher' removed from exports
    # 'PipelineOrchestrator' removed from exports
    #4'StitcherConfig',
    #'PipelineConfig',
    #'AutoPipelineFactory',
    'Step',

    # 'Pipeline' removed from exports as it's been moved
]
