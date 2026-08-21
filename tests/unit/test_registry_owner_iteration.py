"""Generic consumers iterate authoritative registries, not subclass trees."""

from arraybridge import SliceBySliceRuntimeParameter

from openhcs.agent.services.architecture_projection_service import (
    ArchitectureTopicProjection,
)
from openhcs.microscopes.microscope_interfaces import MetadataArtifactProvider
from openhcs.processing.backends.lib_registry.unified_registry import (
    ProcessingContract,
)


def test_architecture_topics_follow_the_owner_registry() -> None:
    assert ArchitectureTopicProjection.projection_types() == tuple(
        ArchitectureTopicProjection.__registry__.values()
    )


def test_metadata_artifact_providers_follow_the_owner_registry() -> None:
    assert MetadataArtifactProvider.registered_provider_types() == tuple(
        MetadataArtifactProvider.__registry__.values()
    )


def test_contract_runtime_parameters_follow_processing_contract_declarations() -> None:
    assert ProcessingContract.semantic_control_parameter_types() == (
        SliceBySliceRuntimeParameter,
    )
