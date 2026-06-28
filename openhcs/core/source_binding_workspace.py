"""Project source-binding declarations into OpenHCS source workspaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

from openhcs.constants.constants import AllComponents, Backend
from openhcs.core.pipeline_image_schema import (
    GrayscaleImageTypeSourceRole,
    ImageAssignment,
    ImagesRule,
    PipelineImageSchema,
    SourceArtifactAssignment,
)
from openhcs.core.source_bindings import (
    ComponentSelector,
    NamedSourceBinding,
    SourceBindingsConfig,
)
from openhcs.core.source_matching import (
    source_component_metadata_values,
    source_filters_match,
    source_metadata_value,
    source_metadata_values_equal,
)
from openhcs.core.source_projection import (
    OpenHCSPlaneAddress,
    SourcePixelRef,
    SourcePlaneProjection,
    SourceProjectionSet,
)
from openhcs.core.source_schema_workspace import (
    ComponentProjection,
    ImageSetAssembler,
    ImageSetRecord,
    SourceSchemaCandidate,
    SourceSchemaCandidateDiscovery,
    SourceSchemaCandidateDiscoveryMode,
    SourceSchemaCandidateDiscoveryRequest,
)
from openhcs.microscopes.source_schema import SourceSchemaFilenameParser


@dataclass(frozen=True, slots=True)
class SourceBindingWorkspaceProjector:
    """Project arbitrary source-bound image files into an OpenHCS workspace."""

    source_bindings: SourceBindingsConfig
    parser: SourceSchemaFilenameParser = field(default_factory=SourceSchemaFilenameParser)

    def __post_init__(self) -> None:
        if not isinstance(self.source_bindings, SourceBindingsConfig):
            raise TypeError(
                "SourceBindingWorkspaceProjector.source_bindings must be "
                f"SourceBindingsConfig, got {type(self.source_bindings).__name__}."
            )

    def projection_set(
        self,
        plate_path: Path,
        image_files: Sequence[str | Path],
    ) -> SourceProjectionSet:
        """Return a validated source projection set for the selected files."""

        return SourceProjectionSet(self.projections(plate_path, image_files))

    def source_schema(self) -> PipelineImageSchema:
        """Project source bindings into the generic source-schema declaration."""

        return PipelineImageSchema(
            images_rule=ImagesRule(filters=self.source_bindings.source_filters),
            metadata_rules=self.source_bindings.metadata_rules,
            assignments_by_alias=MappingProxyType(
                {
                    binding.alias: ImageAssignment(
                        alias=binding.alias,
                        image_type=GrayscaleImageTypeSourceRole.image_type(),
                        selector=binding.selector,
                        origin=binding.origin,
                    )
                    for binding in self.source_bindings.bindings
                    if binding.participates_in_execution_anchoring
                }
            ),
            source_artifacts_by_alias=MappingProxyType(
                {
                    binding.alias: SourceArtifactAssignment(
                        alias=binding.alias,
                        artifact_kind=binding.artifact_kind,
                        selector=binding.selector,
                        origin=binding.origin,
                    )
                    for binding in self.source_bindings.bindings
                    if not binding.participates_in_execution_anchoring
                }
            ),
            match_plan=self.source_bindings.match_plan,
        )

    def projections(
        self,
        plate_path: Path,
        image_files: Sequence[str | Path],
    ) -> tuple[SourcePlaneProjection, ...]:
        """Project source files into canonical OpenHCS source-plane records."""

        plate_path = Path(plate_path)
        candidates = self._source_candidates(plate_path, image_files)
        bindings = tuple(
            binding
            for binding in self.source_bindings.bindings
            if binding.participates_in_execution_anchoring
        )
        if not bindings:
            return self._ungrouped_projections(candidates)
        return self._bound_projections(bindings, candidates)

    def _source_candidates(
        self,
        plate_path: Path,
        image_files: Sequence[str | Path],
    ) -> tuple[SourceSchemaCandidate, ...]:
        schema = PipelineImageSchema(
            images_rule=ImagesRule(filters=self.source_bindings.source_filters),
            metadata_rules=self.source_bindings.metadata_rules,
        )
        source_files = tuple(Path(path) for path in image_files)
        return SourceSchemaCandidateDiscovery(
            SourceSchemaCandidateDiscoveryRequest(
                source_root=plate_path,
                source_files=source_files,
                schema=schema,
                discovery_mode=SourceSchemaCandidateDiscoveryMode.LOCAL_FILES,
            )
        ).candidates()

    def _bound_projections(
        self,
        bindings: tuple[NamedSourceBinding, ...],
        candidates: tuple[SourceSchemaCandidate, ...],
    ) -> tuple[SourcePlaneProjection, ...]:
        candidates_by_alias = {
            binding.alias: self._candidates_for_binding(binding, candidates)
            for binding in bindings
        }
        missing = tuple(
            alias for alias, alias_candidates in candidates_by_alias.items()
            if not alias_candidates
        )
        if missing:
            raise ValueError(
                "Source-binding projection matched no image files for aliases: "
                + ", ".join(missing)
            )
        schema = PipelineImageSchema(
            images_rule=ImagesRule(filters=self.source_bindings.source_filters),
            metadata_rules=self.source_bindings.metadata_rules,
            assignments_by_alias=MappingProxyType(
                {
                    binding.alias: ImageAssignment(
                        alias=binding.alias,
                        image_type=GrayscaleImageTypeSourceRole.image_type(),
                    )
                    for binding in bindings
                }
            ),
            match_plan=self.source_bindings.match_plan,
        )
        image_sets = ImageSetAssembler.for_schema(schema).image_sets(
            schema,
            candidates_by_alias,
        )
        return self._image_set_projections(bindings, image_sets)

    def _candidates_for_binding(
        self,
        binding: NamedSourceBinding,
        candidates: tuple[SourceSchemaCandidate, ...],
    ) -> tuple[SourceSchemaCandidate, ...]:
        selected: list[SourceSchemaCandidate] = []
        for candidate in candidates:
            matcher = SourceBindingProjectionMatcher(candidate.metadata)
            if not matcher.matches(candidate, binding):
                continue
            selected.append(
                SourceBindingCandidateProjection(
                    candidate,
                    binding,
                ).projected_candidate()
            )
        return tuple(selected)

    def _image_set_projections(
        self,
        bindings: tuple[NamedSourceBinding, ...],
        image_sets: tuple[ImageSetRecord, ...],
    ) -> tuple[SourcePlaneProjection, ...]:
        projections: list[SourcePlaneProjection] = []
        site_indexes_by_well: dict[str, int] = {}
        for image_set in image_sets:
            well = ComponentProjection.resolve(
                AllComponents.WELL,
                image_set.metadata,
                image_set.index,
            )
            site_index = site_indexes_by_well.get(well, 0)
            address_base = SourceBindingImageSetAddressBase(
                metadata=image_set.metadata,
                image_index=site_index,
            )
            site_indexes_by_well[well] = site_index + 1
            for channel_index, binding in enumerate(bindings, start=1):
                candidate = image_set.candidates_by_alias[binding.alias]
                address = OpenHCSPlaneAddress(
                    well=well,
                    site=address_base.component(AllComponents.SITE),
                    channel=SourceBindingChannelProjection(
                        candidate,
                        channel_index,
                    ).channel(),
                    z_index=address_base.component(AllComponents.Z_INDEX),
                    timepoint=address_base.component(AllComponents.TIMEPOINT),
                )
                projections.append(
                    SourceBindingCandidateProjection(
                        candidate,
                        binding,
                    ).projection(address)
                )
        return tuple(projections)

    def _ungrouped_projections(
        self,
        candidates: tuple[SourceSchemaCandidate, ...],
    ) -> tuple[SourcePlaneProjection, ...]:
        projections: list[SourcePlaneProjection] = []
        for image_index, candidate in enumerate(candidates):
            address = OpenHCSPlaneAddress(
                well=ComponentProjection.resolve(
                    AllComponents.WELL,
                    candidate.metadata,
                    image_index,
                ),
                site=ComponentProjection.resolve(
                    AllComponents.SITE,
                    candidate.metadata,
                    image_index,
                ),
                channel=ComponentProjection.resolve(
                    AllComponents.CHANNEL,
                    candidate.metadata,
                    image_index,
                ),
                z_index=ComponentProjection.resolve(
                    AllComponents.Z_INDEX,
                    candidate.metadata,
                    image_index,
                ),
                timepoint=ComponentProjection.resolve(
                    AllComponents.TIMEPOINT,
                    candidate.metadata,
                    image_index,
                ),
            )
            projections.append(
                SourceBindingCandidateProjection(candidate).projection(address)
            )
        return tuple(projections)


@dataclass(frozen=True, slots=True)
class SourceBindingProjectionMatcher:
    """Select source files and project binding component assignments at init time."""

    metadata: Mapping[str, str]

    def matches(
        self,
        candidate: SourceSchemaCandidate,
        binding: NamedSourceBinding,
    ) -> bool:
        selector = binding.selector
        return (
            any(
                source_filters_match(path, selector.filters)
                for path in candidate.source_filter_path_identities()
            )
            and all(
                (value := source_metadata_value(self.metadata, metadata.field)) is not None
                and source_metadata_values_equal(value, metadata.value)
                for metadata in selector.metadata
            )
            and all(
                self.component_is_compatible(component)
                for component in selector.components
            )
        )

    def component_assignments(
        self,
        binding: NamedSourceBinding,
    ) -> Mapping[AllComponents, str]:
        assignments: dict[AllComponents, str] = {}
        for selector in binding.selector.components:
            existing = assignments.get(selector.component)
            if existing is not None and not source_metadata_values_equal(
                existing,
                selector.value,
            ):
                raise ValueError(
                    f"Binding {binding.alias!r} assigns conflicting "
                    f"{selector.component.value!r} values {existing!r} and "
                    f"{selector.value!r}."
                )
            assignments[selector.component] = selector.value
        return MappingProxyType(assignments)

    def component_is_compatible(self, selector: ComponentSelector) -> bool:
        values = tuple(source_component_metadata_values(self.metadata, selector.component))
        return not values or any(
            source_metadata_values_equal(value, selector.value)
            for value in values
        )


@dataclass(frozen=True, slots=True)
class SourceBindingCandidateProjection:
    """Project binding-selected candidates into workspace projection payloads."""

    candidate: SourceSchemaCandidate
    binding: NamedSourceBinding | None = None

    def candidate_with_binding_components(self) -> SourceSchemaCandidate:
        if self.binding is None:
            return self.candidate
        metadata = dict(self.candidate.metadata)
        assignments = SourceBindingProjectionMatcher(
            self.candidate.metadata
        ).component_assignments(self.binding)
        for component, value in assignments.items():
            current = ComponentProjection.resolve_from_metadata(
                component,
                metadata,
            )
            if current is not None and not source_metadata_values_equal(current, value):
                raise ValueError(
                    f"Source candidate {self.candidate.relative_path!r} has "
                    f"conflicting {component.value!r} values {current!r} and "
                    f"{value!r}."
                )
            metadata[component.value] = value
        return SourceSchemaCandidate(
            path=self.candidate.path,
            relative_path=self.candidate.relative_path,
            metadata=MappingProxyType(metadata),
            source_plane_index=self.candidate.source_plane_index,
            source_plane_count=self.candidate.source_plane_count,
            source_filter_paths=self.candidate.source_filter_paths,
        )

    def projected_candidate(self) -> SourceSchemaCandidate:
        """Return this candidate after binding component projection."""

        return self.candidate_with_binding_components()

    def projection(
        self,
        address: OpenHCSPlaneAddress,
    ) -> SourcePlaneProjection:
        source_alias = None if self.binding is None else self.binding.alias
        return SourcePlaneProjection(
            address=address,
            ref=SourceBindingCandidateSourceRef(self.candidate).ref(),
            source_alias=source_alias,
            source_metadata=self.candidate.metadata,
        )


@dataclass(frozen=True, slots=True)
class SourceBindingCandidateSourceRef:
    """Build a source pixel reference for a candidate."""

    candidate: SourceSchemaCandidate

    def ref(self) -> SourcePixelRef:
        return SourcePixelRef(
            backend=Backend.DISK.value,
            source_path=self.candidate.relative_path,
            plane_index=self.candidate.source_plane_index,
            source_z_index=self.source_z_index(),
        )

    def source_z_index(self) -> int | None:
        value = source_metadata_value(
            self.candidate.metadata,
            AllComponents.Z_INDEX.value,
        )
        if value is None or not value.isdecimal():
            return None
        return int(value)


@dataclass(frozen=True, slots=True)
class SourceBindingImageSetAddressBase:
    """Project shared image-set metadata into common address components."""

    metadata: Mapping[str, str]
    image_index: int

    def component(self, component: AllComponents) -> str:
        return ComponentProjection.resolve(component, self.metadata, self.image_index)


@dataclass(frozen=True, slots=True)
class SourceBindingChannelProjection:
    """Resolve one image-set member's OpenHCS channel component."""

    candidate: SourceSchemaCandidate
    alias_index: int

    def channel(self) -> str:
        value = ComponentProjection.resolve_from_metadata(
            AllComponents.CHANNEL,
            self.candidate.metadata,
        )
        if value is not None:
            return value
        return str(self.alias_index)
