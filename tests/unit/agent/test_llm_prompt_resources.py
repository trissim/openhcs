"""Nominal-registry coverage for LLM authoring prompt resources."""

from openhcs.agent.services.llm_prompt_resources import LLMPromptResourceCatalog
from openhcs.core.artifacts import ArtifactType


def test_dynamic_imports_project_registered_artifact_types() -> None:
    class TemporaryPromptArtifactType(ArtifactType):
        value = "temporary_prompt_artifact"

    try:
        imports = LLMPromptResourceCatalog().dynamic_imports_section()
        assert "TemporaryPromptArtifactType" in imports
        assert "SpatialGraphArtifactType" in imports
        assert "from openhcs.core.runtime_spatial_graph import SpatialGraph" in imports
        assert "SWCOptions" in imports
        assert "SpatialGraphROIOptions" in imports
    finally:
        ArtifactType.__registry__.pop(TemporaryPromptArtifactType.value)


def test_materialization_prompt_explains_one_graph_with_two_projections() -> None:
    materializers = LLMPromptResourceCatalog().dynamic_materializers_section()

    assert "SpatialGraphArtifactType" in materializers
    assert "SWCOptions()" in materializers
    assert "SpatialGraphROIOptions()" in materializers
    assert ".graph.roi.zip" in materializers
    assert "without recomputing topology" in materializers
    assert "OpenHCS Napari reader" in materializers
    assert "Standard SWC retains sample/type/radius/parent fields" in materializers
