"""Nominal runtime spatial-graph values."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import numpy as np

from openhcs.core.artifacts import NamedArtifactPayload
from openhcs.core.source_image_provenance import (
    SourceImageProvenance,
    SourceImageProvenanceFields,
)

SpatialGraphFeatureValue = str | int | float | bool | None
SpatialCoordinate = tuple[float, ...]


def _validated_coordinates(
    value: Sequence[Sequence[float]] | np.ndarray,
    *,
    subject: str,
) -> np.ndarray:
    """Return an immutable finite N-dimensional coordinate matrix."""

    coordinates = np.array(value, dtype=float, copy=True)
    if coordinates.ndim != 2 or coordinates.shape[1] not in (2, 3):
        raise ValueError(
            f"{subject} must be an N x 2 or N x 3 array, got {coordinates.shape!r}."
        )
    if len(coordinates) < 2:
        raise ValueError(f"{subject} must contain at least two points.")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError(f"{subject} must contain only finite values.")
    coordinates.setflags(write=False)
    return coordinates


def _validated_features(
    features: Sequence[tuple[str, SpatialGraphFeatureValue]],
    *,
    subject: str,
) -> tuple[tuple[str, SpatialGraphFeatureValue], ...]:
    normalized = tuple((str(name), value) for name, value in features)
    feature_names = tuple(name for name, _value in normalized)
    if any(not name for name in feature_names):
        raise ValueError(f"{subject} feature names cannot be empty.")
    if len(set(feature_names)) != len(feature_names):
        raise ValueError(f"{subject} feature names must be unique.")
    return normalized


@dataclass(frozen=True, slots=True)
class SpatialGraphNode:
    """One spatial topology node with scalar features and a physical radius."""

    node_id: int
    coordinates: SpatialCoordinate
    radius: float = 1.0
    features: tuple[tuple[str, SpatialGraphFeatureValue], ...] = ()

    def __post_init__(self) -> None:
        if self.node_id <= 0:
            raise ValueError("SpatialGraphNode.node_id must be positive.")
        coordinates = tuple(float(value) for value in self.coordinates)
        if len(coordinates) not in (2, 3):
            raise ValueError(
                "SpatialGraphNode.coordinates must contain two or three values."
            )
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("SpatialGraphNode.coordinates must be finite.")
        radius = float(self.radius)
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("SpatialGraphNode.radius must be finite and positive.")
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "radius", radius)
        object.__setattr__(
            self,
            "features",
            _validated_features(self.features, subject="SpatialGraphNode"),
        )

    @classmethod
    def from_features(
        cls,
        *,
        node_id: int,
        coordinates: Sequence[float],
        radius: float = 1.0,
        features: Mapping[str, SpatialGraphFeatureValue],
    ) -> "SpatialGraphNode":
        """Build a node while preserving declared feature order."""

        return cls(
            node_id=node_id,
            coordinates=tuple(coordinates),
            radius=radius,
            features=tuple(features.items()),
        )

    def feature_mapping(self) -> dict[str, SpatialGraphFeatureValue]:
        """Return scalar node features for interchange projections."""

        return dict(self.features)


@dataclass(frozen=True, slots=True)
class SpatialGraphEdge:
    """One directed edge whose path begins and ends at its referenced nodes."""

    edge_id: int
    source: SpatialGraphNode
    target: SpatialGraphNode
    coordinates: np.ndarray = field(compare=False, repr=False)
    features: tuple[tuple[str, SpatialGraphFeatureValue], ...] = ()

    def __post_init__(self) -> None:
        if self.edge_id <= 0:
            raise ValueError("SpatialGraphEdge.edge_id must be positive.")
        if self.source is self.target:
            raise ValueError("SpatialGraphEdge cannot connect a node to itself.")
        coordinates = _validated_coordinates(
            self.coordinates,
            subject="SpatialGraphEdge.coordinates",
        )
        if (
            len(self.source.coordinates) != coordinates.shape[1]
            or len(self.target.coordinates) != coordinates.shape[1]
        ):
            raise ValueError(
                "SpatialGraphEdge path dimensionality must match both endpoint nodes."
            )
        if not np.allclose(coordinates[0], self.source.coordinates):
            raise ValueError(
                "SpatialGraphEdge.coordinates must begin at source.coordinates."
            )
        if not np.allclose(coordinates[-1], self.target.coordinates):
            raise ValueError(
                "SpatialGraphEdge.coordinates must end at target.coordinates."
            )
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(
            self,
            "features",
            _validated_features(self.features, subject="SpatialGraphEdge"),
        )

    @property
    def source_node_id(self) -> int:
        """Return the referenced source node identity."""

        return self.source.node_id

    @property
    def target_node_id(self) -> int:
        """Return the referenced target node identity."""

        return self.target.node_id

    @classmethod
    def from_features(
        cls,
        *,
        edge_id: int,
        source: SpatialGraphNode,
        target: SpatialGraphNode,
        coordinates: Sequence[Sequence[float]] | np.ndarray,
        features: Mapping[str, SpatialGraphFeatureValue],
    ) -> "SpatialGraphEdge":
        """Build an edge while preserving declared feature order."""

        return cls(
            edge_id=edge_id,
            source=source,
            target=target,
            coordinates=np.asarray(coordinates),
            features=tuple(features.items()),
        )

    def feature_mapping(self) -> dict[str, SpatialGraphFeatureValue]:
        """Return scalar edge features for table and transport projection."""

        return dict(self.features)


@dataclass(slots=True)
class SpatialGraph(SourceImageProvenanceFields, NamedArtifactPayload):
    """Named spatial graph with direct node references and path geometry."""

    name: str
    nodes: tuple[SpatialGraphNode, ...]
    edges: tuple[SpatialGraphEdge, ...]
    coordinate_spacing: SpatialCoordinate = (1.0, 1.0)
    source_plane_index: int | None = None

    def __post_init__(self, *source_provenance_values: object) -> None:
        self.absorb_explicit_source_provenance(
            SourceImageProvenance.from_init_values(source_provenance_values)
        )
        self.normalize_source_provenance_fields()
        self.validate_artifact_name()
        nodes = tuple(self.nodes)
        edges = tuple(self.edges)
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "edges", edges)

        node_ids = tuple(node.node_id for node in nodes)
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("SpatialGraph node IDs must be unique.")
        edge_ids = tuple(edge.edge_id for edge in edges)
        if len(set(edge_ids)) != len(edge_ids):
            raise ValueError("SpatialGraph edge IDs must be unique.")

        spacing = tuple(float(value) for value in self.coordinate_spacing)
        if len(spacing) not in (2, 3):
            raise ValueError(
                "SpatialGraph.coordinate_spacing must contain two or three values."
            )
        if not np.all(np.isfinite(spacing)) or any(value <= 0 for value in spacing):
            raise ValueError(
                "SpatialGraph.coordinate_spacing must contain finite positive values."
            )
        if any(len(node.coordinates) != len(spacing) for node in nodes):
            raise ValueError(
                "Every SpatialGraphNode dimensionality must match coordinate_spacing."
            )

        node_object_ids = {id(node) for node in nodes}
        for edge in edges:
            if (
                id(edge.source) not in node_object_ids
                or id(edge.target) not in node_object_ids
            ):
                raise ValueError(
                    "SpatialGraphEdge endpoints must directly reference nodes in the graph."
                )
            if edge.coordinates.shape[1] != len(spacing):
                raise ValueError(
                    "Every SpatialGraphEdge dimensionality must match coordinate_spacing."
                )
        object.__setattr__(self, "coordinate_spacing", spacing)
        if self.source_plane_index is not None:
            if isinstance(self.source_plane_index, bool) or not isinstance(
                self.source_plane_index,
                (int, np.integer),
            ):
                raise TypeError("SpatialGraph.source_plane_index must be an integer")
            if self.source_plane_index < 0:
                raise ValueError("SpatialGraph.source_plane_index cannot be negative")
            object.__setattr__(self, "source_plane_index", int(self.source_plane_index))

    def contextualized_source_provenance(
        self,
        source_provenance: SourceImageProvenance,
    ) -> SourceImageProvenance:
        """Project invocation provenance through the graph's declared source plane."""

        if self.source_plane_index is None:
            return source_provenance
        return source_provenance.for_source_plane(self.source_plane_index)

    @classmethod
    def from_swc(cls, path: str | Path) -> Self:
        """Read one standard SWC morphology as a physical 3D graph.

        SWC sample IDs, structure types, radii, and parent IDs remain attached
        to the nominal graph. Standard SWC cannot carry arbitrary OpenHCS edge
        feature columns; the feature-bearing 2D viewer projection remains the
        ``.graph.roi.zip`` materialization.
        """

        source_path = Path(path)
        graph_name = source_path.stem
        samples: dict[int, tuple[SpatialGraphNode, int]] = {}
        for line_number, raw_line in enumerate(
            source_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                name_prefix = "# OpenHCS spatial graph:"
                if line.startswith(name_prefix):
                    declared_name = line.removeprefix(name_prefix).strip()
                    if declared_name:
                        graph_name = declared_name
                continue
            fields = line.split()
            if len(fields) != 7:
                raise ValueError(
                    f"SWC line {line_number} requires seven fields, got {len(fields)}."
                )
            try:
                sample_id = int(fields[0])
                sample_type = int(fields[1])
                x, y, z = (float(value) for value in fields[2:5])
                radius = float(fields[5])
                parent_sample_id = int(fields[6])
            except ValueError as exc:
                raise ValueError(
                    f"SWC line {line_number} contains an invalid numeric field."
                ) from exc
            if sample_id in samples:
                raise ValueError(f"SWC sample ID {sample_id} is duplicated.")
            if sample_type < 0:
                raise ValueError(
                    f"SWC sample {sample_id} has a negative structure type."
                )
            samples[sample_id] = (
                SpatialGraphNode.from_features(
                    node_id=sample_id,
                    coordinates=(z, y, x),
                    radius=radius,
                    features={
                        "swc_type": sample_type,
                        "swc_parent_id": parent_sample_id,
                    },
                ),
                parent_sample_id,
            )

        nodes = tuple(node for node, _parent_sample_id in samples.values())
        edges: list[SpatialGraphEdge] = []
        for node, parent_sample_id in samples.values():
            if parent_sample_id == -1:
                continue
            try:
                parent = samples[parent_sample_id][0]
            except KeyError as exc:
                raise ValueError(
                    f"SWC sample {node.node_id} references missing parent "
                    f"{parent_sample_id}."
                ) from exc
            edges.append(
                SpatialGraphEdge.from_features(
                    edge_id=node.node_id,
                    source=parent,
                    target=node,
                    coordinates=(parent.coordinates, node.coordinates),
                    features={
                        "swc_type": node.feature_mapping()["swc_type"],
                        "swc_parent_id": parent_sample_id,
                    },
                )
            )

        graph = cls(
            name=graph_name,
            nodes=nodes,
            edges=tuple(edges),
            coordinate_spacing=(1.0, 1.0, 1.0),
            source_path=str(source_path),
        )
        graph.require_directed_forest()
        return graph

    def roots(self) -> tuple[SpatialGraphNode, ...]:
        """Return deterministic roots after requiring a directed forest."""

        self.require_directed_forest()
        targeted_node_ids = {edge.target.node_id for edge in self.edges}
        return tuple(
            sorted(
                (node for node in self.nodes if node.node_id not in targeted_node_ids),
                key=lambda node: node.node_id,
            )
        )

    def require_directed_forest(self) -> None:
        """Reject cycles and nodes with more than one incoming edge."""

        incoming_counts = defaultdict(int)
        outgoing: dict[int, list[SpatialGraphEdge]] = defaultdict(list)
        for edge in self.edges:
            incoming_counts[edge.target.node_id] += 1
            if incoming_counts[edge.target.node_id] > 1:
                raise ValueError(
                    "SpatialGraph is not a directed forest: node "
                    f"{edge.target.node_id} has multiple incoming edges."
                )
            outgoing[edge.source.node_id].append(edge)

        remaining_incoming = {
            node.node_id: incoming_counts[node.node_id] for node in self.nodes
        }
        ready = deque(
            node.node_id
            for node in sorted(self.nodes, key=lambda item: item.node_id)
            if remaining_incoming[node.node_id] == 0
        )
        visited_count = 0
        while ready:
            node_id = ready.popleft()
            visited_count += 1
            for edge in outgoing[node_id]:
                target_id = edge.target.node_id
                remaining_incoming[target_id] -= 1
                if remaining_incoming[target_id] == 0:
                    ready.append(target_id)
        if visited_count != len(self.nodes):
            raise ValueError("SpatialGraph is not a directed forest: cycle detected.")
