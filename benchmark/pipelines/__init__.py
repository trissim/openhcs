"""Pipeline registry."""

from openhcs.core.public_api import public_names_from_objects

from benchmark.contracts.pipeline import PipelineSpec
from benchmark.pipelines.registry import (
    NUCLEI_SEGMENTATION as NUCLEI_SEGMENTATION,
    PIPELINE_REGISTRY as PIPELINE_REGISTRY,
    get_pipeline_spec,
)

__all__ = public_names_from_objects(
    PipelineSpec,
    get_pipeline_spec,
    extra_names=("NUCLEI_SEGMENTATION", "PIPELINE_REGISTRY"),
)
