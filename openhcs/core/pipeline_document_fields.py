"""Public field names shared by complete and step-fragment pipeline documents."""

from enum import Enum


class PipelineDocumentField(str, Enum):
    """Exact public assignments owned by the pipeline document contract."""

    PIPELINE_CONFIG = "pipeline_config"
    PIPELINE_STEPS = "pipeline_steps"
