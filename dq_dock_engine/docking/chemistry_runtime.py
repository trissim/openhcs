from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from typing import ClassVar

import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from dq_dock_engine.docking.scoring import CertifiedOptionalInteractionTerm


class SiteGeometry(Enum):
    POINT = "point"
    DIRECTIONAL = "directional"
    RING = "ring"


class DerivedPytreeRecord:
    _aux_fields: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def _child_fields(cls) -> tuple[str, ...]:
        return tuple(
            field.name for field in fields(cls) if field.name not in cls._aux_fields
        )

    def tree_flatten(self):
        children = tuple(getattr(self, name) for name in type(self)._child_fields())
        aux = tuple(getattr(self, name) for name in type(self)._aux_fields)
        return (children, aux)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        values = dict(zip(cls._child_fields(), children, strict=True))
        values.update(dict(zip(cls._aux_fields, aux_data, strict=True)))
        return cls(**values)


def _subset_mask(
    anchor_indices: jnp.ndarray, retained_indices: jnp.ndarray
) -> jnp.ndarray:
    return jnp.any(
        anchor_indices[:, None] == retained_indices[None, :],
        axis=1,
    )


def _normalize_rows(vectors: jnp.ndarray) -> jnp.ndarray:
    norms = jnp.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / jnp.maximum(norms, 1e-6)


@register_pytree_node_class
@dataclass(frozen=True)
class AnchoredSiteArray(DerivedPytreeRecord):
    geometry: SiteGeometry
    positions: jnp.ndarray
    vectors: jnp.ndarray
    strengths: jnp.ndarray
    anchor_indices: jnp.ndarray

    _aux_fields = ("geometry",)

    @classmethod
    def empty(cls, geometry: SiteGeometry) -> "AnchoredSiteArray":
        return cls(
            geometry=geometry,
            positions=jnp.zeros((0, 3), dtype=jnp.float32),
            vectors=jnp.zeros((0, 3), dtype=jnp.float32),
            strengths=jnp.zeros((0,), dtype=jnp.float32),
            anchor_indices=jnp.zeros((0,), dtype=jnp.int32),
        )

    def validate(self) -> None:
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError("anchored positions must have shape (N, 3)")
        if self.vectors.ndim != 2 or self.vectors.shape[1] != 3:
            raise ValueError("anchored vectors must have shape (N, 3)")
        if self.strengths.ndim != 1:
            raise ValueError("anchored strengths must be 1D")
        if self.anchor_indices.ndim != 1:
            raise ValueError("anchored indices must be 1D")
        if not (
            self.positions.shape[0]
            == self.vectors.shape[0]
            == self.strengths.shape[0]
            == self.anchor_indices.shape[0]
        ):
            raise ValueError(
                "anchored site arrays must have matching leading dimensions"
            )

    def subset(self, retained_indices: jnp.ndarray) -> "AnchoredSiteArray":
        mask = _subset_mask(self.anchor_indices, retained_indices)
        return AnchoredSiteArray(
            geometry=self.geometry,
            positions=self.positions,
            vectors=self.vectors,
            strengths=jnp.where(mask, self.strengths, 0.0),
            anchor_indices=self.anchor_indices,
        )


@register_pytree_node_class
@dataclass(frozen=True)
class IndexedSiteArray(DerivedPytreeRecord):
    geometry: SiteGeometry
    atom_index_rows: jnp.ndarray
    atom_index_mask: jnp.ndarray
    reference_index_rows: jnp.ndarray
    reference_index_mask: jnp.ndarray
    strengths: jnp.ndarray

    _aux_fields = ("geometry",)

    @classmethod
    def empty(cls, geometry: SiteGeometry) -> "IndexedSiteArray":
        return cls(
            geometry=geometry,
            atom_index_rows=jnp.zeros((0, 1), dtype=jnp.int32),
            atom_index_mask=jnp.zeros((0, 1), dtype=bool),
            reference_index_rows=jnp.zeros((0, 1), dtype=jnp.int32),
            reference_index_mask=jnp.zeros((0, 1), dtype=bool),
            strengths=jnp.zeros((0,), dtype=jnp.float32),
        )

    def validate(self) -> None:
        if self.atom_index_rows.ndim != 2 or self.atom_index_mask.ndim != 2:
            raise ValueError("indexed atom rows and masks must be 2D")
        if self.reference_index_rows.ndim != 2 or self.reference_index_mask.ndim != 2:
            raise ValueError("indexed reference rows and masks must be 2D")
        if self.atom_index_rows.shape != self.atom_index_mask.shape:
            raise ValueError("indexed atom rows and mask must match")
        if self.reference_index_rows.shape != self.reference_index_mask.shape:
            raise ValueError("indexed reference rows and mask must match")
        if self.strengths.ndim != 1:
            raise ValueError("indexed strengths must be 1D")
        if self.atom_index_rows.shape[0] != self.strengths.shape[0]:
            raise ValueError("indexed strengths must match site row count")


def _gather_rows(poses_coords: jnp.ndarray, rows: jnp.ndarray) -> jnp.ndarray:
    return jnp.take(poses_coords, jnp.maximum(rows, 0), axis=1)


def _masked_average(points: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    mask4 = mask[None, :, :, None].astype(points.dtype)
    counts = jnp.maximum(jnp.sum(mask4, axis=2), 1.0)
    return jnp.sum(points * mask4, axis=2) / counts


def indexed_site_positions(
    poses_coords: jnp.ndarray, site_array: IndexedSiteArray
) -> jnp.ndarray:
    return _masked_average(
        _gather_rows(poses_coords, site_array.atom_index_rows),
        site_array.atom_index_mask,
    )


def indexed_site_vectors(
    poses_coords: jnp.ndarray, site_array: IndexedSiteArray
) -> jnp.ndarray:
    positions = indexed_site_positions(poses_coords, site_array)
    if site_array.geometry == SiteGeometry.POINT:
        return jnp.zeros_like(positions)
    reference_points = _gather_rows(poses_coords, site_array.reference_index_rows)
    if site_array.geometry == SiteGeometry.DIRECTIONAL:
        reference_mean = _masked_average(
            reference_points, site_array.reference_index_mask
        )
        return _normalize_rows(positions - reference_mean)
    if site_array.reference_index_rows.shape[1] < 3:
        return jnp.zeros_like(positions)
    plane_points = reference_points[:, :, :3, :]
    v1 = plane_points[:, :, 1, :] - plane_points[:, :, 0, :]
    v2 = plane_points[:, :, 2, :] - plane_points[:, :, 0, :]
    return _normalize_rows(jnp.cross(v1, v2))


class DerivedInteractionTerm(CertifiedOptionalInteractionTerm, DerivedPytreeRecord):
    _subset_fields: ClassVar[tuple[str, ...]] = ()
    _validated_fields: ClassVar[tuple[str, ...]] = ()
    _positive_scalar_fields: ClassVar[tuple[str, ...]] = ()
    _activity_field_groups: ClassVar[tuple[tuple[str, ...], ...]] = ()
    _geometry_requirements: ClassVar[dict[str, SiteGeometry]] = {}
    _aux_fields: ClassVar[tuple[str, ...]] = ()

    def validate(self) -> None:
        for field_name in type(self)._validated_fields:
            getattr(self, field_name).validate()
        for field_name, geometry in type(self)._geometry_requirements.items():
            if getattr(self, field_name).geometry is not geometry:
                raise ValueError(f"{field_name} must use geometry {geometry.name}")
        for field_name in type(self)._positive_scalar_fields:
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")

    def receptor_subset(self, indices: jnp.ndarray):
        values = {}
        for field_name in type(self)._child_fields() + type(self)._aux_fields:
            value = getattr(self, field_name)
            values[field_name] = (
                value.subset(indices)
                if field_name in type(self)._subset_fields
                else value
            )
        return type(self)(**values)

    @property
    def cutoff_radius(self) -> float:
        return self.cutoff

    @property
    def is_active(self) -> jnp.ndarray:
        if not type(self)._activity_field_groups:
            return jnp.array(False)
        branch_activities = []
        for group in type(self)._activity_field_groups:
            branch_activities.append(
                jnp.all(
                    jnp.stack(
                        [
                            jnp.any(getattr(self, name).strengths != 0.0)
                            for name in group
                        ],
                        axis=0,
                    )
                )
            )
        return jnp.any(jnp.stack(branch_activities, axis=0))


@register_pytree_node_class
@dataclass(frozen=True)
class PiStackingInteractionTerm(DerivedInteractionTerm):
    receptor_rings: AnchoredSiteArray
    ligand_rings: IndexedSiteArray
    ideal_distance: float = 3.6
    distance_width: float = 0.6
    offset_width: float = 1.2
    cutoff: float = 6.0

    _subset_fields = ("receptor_rings",)
    _validated_fields = ("receptor_rings", "ligand_rings")
    _positive_scalar_fields = (
        "ideal_distance",
        "distance_width",
        "offset_width",
        "cutoff",
    )
    _activity_field_groups = (("receptor_rings", "ligand_rings"),)
    _geometry_requirements = {
        "receptor_rings": SiteGeometry.RING,
        "ligand_rings": SiteGeometry.RING,
    }
    _aux_fields = ("ideal_distance", "distance_width", "offset_width", "cutoff")

    def _pair_scores(
        self, poses_coords: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        ligand_centers = indexed_site_positions(poses_coords, self.ligand_rings)
        ligand_normals = indexed_site_vectors(poses_coords, self.ligand_rings)
        receptor_normals = _normalize_rows(self.receptor_rings.vectors)
        delta = (
            ligand_centers[:, None, :, :]
            - self.receptor_rings.positions[None, :, None, :]
        )
        dists = jnp.linalg.norm(delta, axis=-1)
        radial = jnp.exp(-(((dists - self.ideal_distance) / self.distance_width) ** 2))
        face_alignment = jnp.abs(
            jnp.sum(
                receptor_normals[None, :, None, :] * ligand_normals[:, None, :, :],
                axis=-1,
            )
        )
        receptor_components = (
            jnp.sum(delta * receptor_normals[None, :, None, :], axis=-1)[..., None]
            * receptor_normals[None, :, None, :]
        )
        lateral_offset = jnp.linalg.norm(delta - receptor_components, axis=-1)
        offset_factor = jnp.exp(-((lateral_offset / self.offset_width) ** 2))
        strengths = (
            self.receptor_rings.strengths[None, :, None]
            * self.ligand_rings.strengths[None, None, :]
        )
        return dists, -(strengths * radial * face_alignment * offset_factor)

    def exact_scores(
        self, receptor_coords: jnp.ndarray, poses_coords: jnp.ndarray
    ) -> jnp.ndarray:
        del receptor_coords
        _, pair_scores = self._pair_scores(poses_coords)
        return jnp.sum(pair_scores, axis=(1, 2))

    def cutoff_scores(
        self, receptor_coords: jnp.ndarray, poses_coords: jnp.ndarray
    ) -> jnp.ndarray:
        del receptor_coords
        dists, pair_scores = self._pair_scores(poses_coords)
        return jnp.sum(jnp.where(dists < self.cutoff, pair_scores, 0.0), axis=(1, 2))


@register_pytree_node_class
@dataclass(frozen=True)
class PiCationInteractionTerm(DerivedInteractionTerm):
    receptor_rings: AnchoredSiteArray
    receptor_cations: AnchoredSiteArray
    ligand_rings: IndexedSiteArray
    ligand_cations: IndexedSiteArray
    ideal_distance: float = 4.5
    distance_width: float = 0.8
    cutoff: float = 7.0

    _subset_fields = ("receptor_rings", "receptor_cations")
    _validated_fields = (
        "receptor_rings",
        "receptor_cations",
        "ligand_rings",
        "ligand_cations",
    )
    _positive_scalar_fields = ("ideal_distance", "distance_width", "cutoff")
    _activity_field_groups = (
        ("receptor_rings", "ligand_cations"),
        ("receptor_cations", "ligand_rings"),
    )
    _geometry_requirements = {
        "receptor_rings": SiteGeometry.RING,
        "receptor_cations": SiteGeometry.POINT,
        "ligand_rings": SiteGeometry.RING,
        "ligand_cations": SiteGeometry.POINT,
    }
    _aux_fields = ("ideal_distance", "distance_width", "cutoff")

    def _branch_scores(
        self, delta: jnp.ndarray, ring_normals: jnp.ndarray, strengths: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        dists = jnp.linalg.norm(delta, axis=-1)
        radial = jnp.exp(-(((dists - self.ideal_distance) / self.distance_width) ** 2))
        unit = delta / jnp.maximum(dists[..., None], 1e-6)
        alignment = jnp.abs(jnp.sum(ring_normals * unit, axis=-1))
        return dists, -(strengths * radial * alignment)

    def _scores(self, poses_coords: jnp.ndarray, *, cutoff_only: bool) -> jnp.ndarray:
        total = jnp.zeros((poses_coords.shape[0],), dtype=poses_coords.dtype)
        ligand_ring_centers = indexed_site_positions(poses_coords, self.ligand_rings)
        ligand_ring_normals = indexed_site_vectors(poses_coords, self.ligand_rings)
        ligand_cation_positions = indexed_site_positions(
            poses_coords, self.ligand_cations
        )

        receptor_ring_delta = (
            ligand_cation_positions[:, None, :, :]
            - self.receptor_rings.positions[None, :, None, :]
        )
        receptor_ring_strengths = (
            self.receptor_rings.strengths[None, :, None]
            * self.ligand_cations.strengths[None, None, :]
        )
        receptor_ring_dists, receptor_ring_scores = self._branch_scores(
            receptor_ring_delta,
            _normalize_rows(self.receptor_rings.vectors)[None, :, None, :],
            receptor_ring_strengths,
        )
        total = total + jnp.sum(
            jnp.where(receptor_ring_dists < self.cutoff, receptor_ring_scores, 0.0)
            if cutoff_only
            else receptor_ring_scores,
            axis=(1, 2),
        )

        receptor_cation_delta = (
            self.receptor_cations.positions[None, :, None, :]
            - ligand_ring_centers[:, None, :, :]
        )
        receptor_cation_strengths = (
            self.receptor_cations.strengths[None, :, None]
            * self.ligand_rings.strengths[None, None, :]
        )
        receptor_cation_dists, receptor_cation_scores = self._branch_scores(
            receptor_cation_delta,
            ligand_ring_normals[:, None, :, :],
            receptor_cation_strengths,
        )
        total = total + jnp.sum(
            jnp.where(receptor_cation_dists < self.cutoff, receptor_cation_scores, 0.0)
            if cutoff_only
            else receptor_cation_scores,
            axis=(1, 2),
        )
        return total

    def exact_scores(
        self, receptor_coords: jnp.ndarray, poses_coords: jnp.ndarray
    ) -> jnp.ndarray:
        del receptor_coords
        return self._scores(poses_coords, cutoff_only=False)

    def cutoff_scores(
        self, receptor_coords: jnp.ndarray, poses_coords: jnp.ndarray
    ) -> jnp.ndarray:
        del receptor_coords
        return self._scores(poses_coords, cutoff_only=True)


@register_pytree_node_class
@dataclass(frozen=True)
class HalogenBondInteractionTerm(DerivedInteractionTerm):
    receptor_acceptors: AnchoredSiteArray
    receptor_donors: AnchoredSiteArray
    ligand_acceptors: IndexedSiteArray
    ligand_donors: IndexedSiteArray
    ideal_distance: float = 3.2
    distance_width: float = 0.45
    cutoff: float = 5.0

    _subset_fields = ("receptor_acceptors", "receptor_donors")
    _validated_fields = (
        "receptor_acceptors",
        "receptor_donors",
        "ligand_acceptors",
        "ligand_donors",
    )
    _positive_scalar_fields = ("ideal_distance", "distance_width", "cutoff")
    _activity_field_groups = (
        ("receptor_acceptors", "ligand_donors"),
        ("receptor_donors", "ligand_acceptors"),
    )
    _geometry_requirements = {
        "receptor_acceptors": SiteGeometry.DIRECTIONAL,
        "receptor_donors": SiteGeometry.DIRECTIONAL,
        "ligand_acceptors": SiteGeometry.DIRECTIONAL,
        "ligand_donors": SiteGeometry.DIRECTIONAL,
    }
    _aux_fields = ("ideal_distance", "distance_width", "cutoff")

    def _branch_scores(
        self,
        donor_positions: jnp.ndarray,
        donor_vectors: jnp.ndarray,
        donor_strengths: jnp.ndarray,
        acceptor_positions: jnp.ndarray,
        acceptor_vectors: jnp.ndarray,
        acceptor_strengths: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        vectors = acceptor_positions - donor_positions
        dists = jnp.linalg.norm(vectors, axis=-1)
        unit = vectors / jnp.maximum(dists[..., None], 1e-6)
        radial = jnp.exp(-(((dists - self.ideal_distance) / self.distance_width) ** 2))
        donor_angle = jnp.clip(jnp.sum(donor_vectors * unit, axis=-1), 0.0, 1.0)
        acceptor_angle = jnp.clip(
            jnp.sum(acceptor_vectors * (-unit), axis=-1), 0.0, 1.0
        )
        return dists, -(
            donor_strengths * acceptor_strengths * radial * donor_angle * acceptor_angle
        )

    def _scores(self, poses_coords: jnp.ndarray, *, cutoff_only: bool) -> jnp.ndarray:
        total = jnp.zeros((poses_coords.shape[0],), dtype=poses_coords.dtype)
        ligand_acceptor_positions = indexed_site_positions(
            poses_coords, self.ligand_acceptors
        )
        ligand_acceptor_vectors = indexed_site_vectors(
            poses_coords, self.ligand_acceptors
        )
        ligand_donor_positions = indexed_site_positions(
            poses_coords, self.ligand_donors
        )
        ligand_donor_vectors = indexed_site_vectors(poses_coords, self.ligand_donors)

        branches = (
            self._branch_scores(
                ligand_donor_positions[:, None, :, :],
                ligand_donor_vectors[:, None, :, :],
                self.ligand_donors.strengths[None, None, :],
                self.receptor_acceptors.positions[None, :, None, :],
                _normalize_rows(self.receptor_acceptors.vectors)[None, :, None, :],
                self.receptor_acceptors.strengths[None, :, None],
            ),
            self._branch_scores(
                self.receptor_donors.positions[None, :, None, :],
                _normalize_rows(self.receptor_donors.vectors)[None, :, None, :],
                self.receptor_donors.strengths[None, :, None],
                ligand_acceptor_positions[:, None, :, :],
                ligand_acceptor_vectors[:, None, :, :],
                self.ligand_acceptors.strengths[None, None, :],
            ),
        )
        for dists, branch_scores in branches:
            total = total + jnp.sum(
                jnp.where(dists < self.cutoff, branch_scores, 0.0)
                if cutoff_only
                else branch_scores,
                axis=(1, 2),
            )
        return total

    def exact_scores(
        self, receptor_coords: jnp.ndarray, poses_coords: jnp.ndarray
    ) -> jnp.ndarray:
        del receptor_coords
        return self._scores(poses_coords, cutoff_only=False)

    def cutoff_scores(
        self, receptor_coords: jnp.ndarray, poses_coords: jnp.ndarray
    ) -> jnp.ndarray:
        del receptor_coords
        return self._scores(poses_coords, cutoff_only=True)


@register_pytree_node_class
@dataclass(frozen=True)
class WaterMediatedHBondInteractionTerm(DerivedInteractionTerm):
    receptor_waters: AnchoredSiteArray
    ligand_polar_sites: IndexedSiteArray
    ideal_distance: float = 2.9
    distance_width: float = 0.8
    cutoff: float = 4.5

    _subset_fields = ("receptor_waters",)
    _validated_fields = ("receptor_waters", "ligand_polar_sites")
    _positive_scalar_fields = ("ideal_distance", "distance_width", "cutoff")
    _activity_field_groups = (("receptor_waters", "ligand_polar_sites"),)
    _geometry_requirements = {
        "receptor_waters": SiteGeometry.DIRECTIONAL,
        "ligand_polar_sites": SiteGeometry.DIRECTIONAL,
    }
    _aux_fields = ("ideal_distance", "distance_width", "cutoff")

    def _pair_scores(
        self, poses_coords: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        ligand_positions = indexed_site_positions(poses_coords, self.ligand_polar_sites)
        ligand_vectors = indexed_site_vectors(poses_coords, self.ligand_polar_sites)
        vectors = (
            ligand_positions[:, None, :, :]
            - self.receptor_waters.positions[None, :, None, :]
        )
        dists = jnp.linalg.norm(vectors, axis=-1)
        unit = vectors / jnp.maximum(dists[..., None], 1e-6)
        radial = jnp.exp(-(((dists - self.ideal_distance) / self.distance_width) ** 2))
        water_angle = jnp.clip(
            jnp.sum(
                _normalize_rows(self.receptor_waters.vectors)[None, :, None, :] * unit,
                axis=-1,
            ),
            0.0,
            1.0,
        )
        ligand_angle = jnp.clip(
            jnp.sum(ligand_vectors[:, None, :, :] * (-unit), axis=-1), 0.0, 1.0
        )
        strengths = (
            self.receptor_waters.strengths[None, :, None]
            * self.ligand_polar_sites.strengths[None, None, :]
        )
        return dists, -(strengths * radial * water_angle * ligand_angle)

    def exact_scores(
        self, receptor_coords: jnp.ndarray, poses_coords: jnp.ndarray
    ) -> jnp.ndarray:
        del receptor_coords
        _, pair_scores = self._pair_scores(poses_coords)
        return jnp.sum(pair_scores, axis=(1, 2))

    def cutoff_scores(
        self, receptor_coords: jnp.ndarray, poses_coords: jnp.ndarray
    ) -> jnp.ndarray:
        del receptor_coords
        dists, pair_scores = self._pair_scores(poses_coords)
        return jnp.sum(jnp.where(dists < self.cutoff, pair_scores, 0.0), axis=(1, 2))
