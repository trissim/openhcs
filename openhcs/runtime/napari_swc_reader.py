"""Napari reader contribution for standard SWC morphology files."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from openhcs.core.runtime_spatial_graph import SpatialGraph

LayerData = tuple[Any, dict[str, Any], str]


def napari_get_swc_reader(
    path: str | Sequence[str],
) -> Callable[[str | Sequence[str]], list[LayerData]] | None:
    """Return the SWC reader when every requested path has an SWC suffix."""

    paths = (path,) if isinstance(path, str) else tuple(path)
    if not paths or any(Path(item).suffix.lower() != ".swc" for item in paths):
        return None
    return read_swc_layers


def read_swc_layers(path: str | Sequence[str]) -> list[LayerData]:
    """Read SWC files into physical 3D Points and Shapes layer data."""

    paths = (path,) if isinstance(path, str) else tuple(path)
    layers: list[LayerData] = []
    for item in paths:
        graph = SpatialGraph.from_swc(item)
        node_features = tuple(node.feature_mapping() for node in graph.nodes)
        point_features = {
            "sample_id": np.asarray(
                [node.node_id for node in graph.nodes], dtype=np.int64
            ),
            "sample_type": np.asarray(
                [features["swc_type"] for features in node_features],
                dtype=np.int64,
            ),
            "radius": np.asarray(
                [node.radius for node in graph.nodes], dtype=float
            ),
            "parent_sample_id": np.asarray(
                [features["swc_parent_id"] for features in node_features],
                dtype=np.int64,
            ),
        }
        point_data = (
            np.asarray(
                [node.coordinates for node in graph.nodes],
                dtype=float,
            ).reshape((-1, 3))
        )
        layers.append(
            (
                point_data,
                {
                    "name": f"{graph.name} samples",
                    "ndim": 3,
                    "size": 2.0 * point_features["radius"],
                    "features": point_features,
                    "metadata": {
                        "source_format": "swc",
                        "source_path": str(item),
                    },
                },
                "points",
            )
        )

        edge_features = tuple(edge.feature_mapping() for edge in graph.edges)
        shape_features = {
            "sample_id": np.asarray(
                [edge.target_node_id for edge in graph.edges], dtype=np.int64
            ),
            "sample_type": np.asarray(
                [features["swc_type"] for features in edge_features],
                dtype=np.int64,
            ),
            "radius": np.asarray(
                [edge.target.radius for edge in graph.edges], dtype=float
            ),
            "parent_sample_id": np.asarray(
                [edge.source_node_id for edge in graph.edges], dtype=np.int64
            ),
        }
        layers.append(
            (
                [edge.coordinates for edge in graph.edges],
                {
                    "name": f"{graph.name} morphology",
                    "ndim": 3,
                    "shape_type": "line",
                    "features": shape_features,
                    "metadata": {
                        "source_format": "swc",
                        "source_path": str(item),
                    },
                },
                "shapes",
            )
        )
    return layers
