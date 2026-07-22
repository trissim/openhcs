"""Generic consumers iterate authoritative registries, not subclass trees."""

from openhcs.agent.services.architecture_projection_service import (
    ArchitectureTopicProjection,
)
from openhcs.microscopes.microscope_interfaces import MetadataArtifactProvider
from openhcs.processing.backends.lib_registry.unified_registry import (
    ContractRuntimeParameter,
)


def test_architecture_topics_follow_the_owner_registry() -> None:
    assert ArchitectureTopicProjection.projection_types() == tuple(
        ArchitectureTopicProjection.__registry__.values()
    )


def test_metadata_artifact_providers_follow_the_owner_registry() -> None:
    assert MetadataArtifactProvider.registered_provider_types() == tuple(
        MetadataArtifactProvider.__registry__.values()
    )


def test_contract_runtime_parameters_follow_the_owner_registry() -> None:
    assert ContractRuntimeParameter.registered_parameter_types() == tuple(
        ContractRuntimeParameter.__registry__.values()
    )
