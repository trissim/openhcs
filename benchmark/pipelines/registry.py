"""Registry of benchmark pipelines."""

from dataclasses import dataclass


@dataclass
class PipelineSpec:
    name: str
    description: str
    parameters: dict


NUCLEI_SEGMENTATION = PipelineSpec(
    name="nuclei_segmentation",
    description="BBBC021 nuclei segmentation (CellProfiler-equivalent)",
    parameters={
        "cppipe_reference_index": 0,
    },
)

# Extension point: CELL_PAINTING = PipelineSpec(...)

PIPELINE_REGISTRY: dict[str, PipelineSpec] = {
    NUCLEI_SEGMENTATION.name: NUCLEI_SEGMENTATION,
}


def get_pipeline_spec(name: str) -> PipelineSpec:
    """
    Retrieve pipeline specification by name.

    Raises:
        KeyError: if pipeline name is unknown.
    """
    try:
        return PIPELINE_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown pipeline '{name}'. "
                       f"Available: {list(PIPELINE_REGISTRY.keys())}") from exc
