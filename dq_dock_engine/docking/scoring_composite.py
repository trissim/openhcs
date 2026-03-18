"""
Literature-backed composite scoring function.

Combines:
    1. Vinardo steric
    2. AD4 hydrogen bonding
    3. ΔSASA or AD4 desolvation
    4. Coulomb electrostatics

OpenHCS Compliance:
- ABC contract for composite calculators
- Frozen dataclasses for configuration
- Enum-driven behavior selection
- Explicit dependency injection
"""

from __future__ import annotations

import jax
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto

import jax.numpy as jnp

from dq_dock_engine.docking.scoring_vinardo import (
    VinardoConfig,
    VinardoBackend,
    ScoringFamily,
    create_scoring_backend,
)
from dq_dock_engine.docking.hbonds_ad4 import (
    HBondParameters,
    HBondDonor,
    HBondDonorType,
    HBondAcceptor,
    create_hbond_calculator,
    total_hbond_energy,
)
from dq_dock_engine.docking.sasa import (
    SASAConfig,
    _fibonacci_sphere_points,
    create_sasa_config,
    get_solvation_params,
)
from dq_dock_engine.docking.electrostatics import (
    ElectrostaticsConfig,
    create_electrostatics_calculator,
)
from dq_dock_engine.docking.charges import (
    ChargeMethod,
    create_charge_assigner,
)


def _normalize(vector: jnp.ndarray) -> jnp.ndarray:
    norm = jnp.linalg.norm(vector)
    if float(norm) < 1e-8:
        return jnp.array([1.0, 0.0, 0.0])
    return vector / norm


def _normalize_rows(vectors: jnp.ndarray) -> jnp.ndarray:
    norms = jnp.linalg.norm(vectors, axis=-1, keepdims=True)
    safe_norms = jnp.maximum(norms, 1e-8)
    normalized = vectors / safe_norms
    fallback = jnp.broadcast_to(jnp.array([1.0, 0.0, 0.0]), vectors.shape)
    return jnp.where(norms > 1e-8, normalized, fallback)


def _hbond_donor_parameters(
    elements: tuple[str, ...],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    donor_indices = []
    r_o_values = []
    depth_values = []

    for idx, element in enumerate(elements):
        upper = element.upper()
        if upper not in ("N", "O", "S"):
            continue
        params = (
            HBondParameters.for_sulfur()
            if upper == "S"
            else HBondParameters.for_oxygen_nitrogen()
        )
        donor_indices.append(idx)
        r_o_values.append(params.r_o)
        depth_values.append(params.depth)

    return (
        jnp.array(donor_indices, dtype=jnp.int32),
        jnp.array(r_o_values),
        jnp.array(depth_values),
    )


def _hbond_acceptor_indices(elements: tuple[str, ...]) -> jnp.ndarray:
    acceptor_indices = []
    for idx, element in enumerate(elements):
        if element.upper() in ("N", "O", "S"):
            acceptor_indices.append(idx)
    return jnp.array(acceptor_indices, dtype=jnp.int32)


def _hbond_matrix_energy(
    donor_positions: jnp.ndarray,
    donor_directions: jnp.ndarray,
    donor_r_o: jnp.ndarray,
    donor_depth: jnp.ndarray,
    acceptor_positions: jnp.ndarray,
    acceptor_directions: jnp.ndarray,
    angle_power: int,
) -> jnp.ndarray:
    if donor_positions.shape[1] == 0:
        return jnp.zeros(donor_positions.shape[0])

    if acceptor_positions.ndim == 2:
        if acceptor_positions.shape[0] == 0:
            return jnp.zeros(donor_positions.shape[0])
        acceptor_positions_batched = jnp.broadcast_to(
            acceptor_positions[None, :, :],
            (donor_positions.shape[0], acceptor_positions.shape[0], 3),
        )
        acceptor_directions_batched = jnp.broadcast_to(
            acceptor_directions[None, :, :],
            (donor_positions.shape[0], acceptor_directions.shape[0], 3),
        )
    else:
        if acceptor_positions.shape[1] == 0:
            return jnp.zeros(donor_positions.shape[0])
        acceptor_positions_batched = acceptor_positions
        acceptor_directions_batched = acceptor_directions

    hydrogen_positions = donor_positions + donor_directions
    diffs = (
        acceptor_positions_batched[:, None, :, :] - hydrogen_positions[:, :, None, :]
    )
    distances = jnp.linalg.norm(diffs, axis=-1)
    within_cutoff = distances < 4.0
    safe_distances = jnp.maximum(distances, 0.1)

    r_ratio = donor_r_o[None, :, None] / safe_distances
    distance_energy = donor_depth[None, :, None] * (
        5.0 * r_ratio**12 - 6.0 * r_ratio**10
    )

    cos_theta = jnp.sum(
        donor_directions[:, :, None, :] * acceptor_directions_batched[:, None, :, :],
        axis=-1,
    )
    cos_theta = jnp.clip(cos_theta, -1.0, 1.0)
    theta = jnp.arccos(cos_theta)
    angular_energy = jnp.cos(theta - jnp.pi) ** angle_power

    return jnp.sum(
        jnp.where(within_cutoff, distance_energy * angular_energy, 0.0),
        axis=(1, 2),
    )


def _hbond_batch(
    poses_coords: jnp.ndarray,
    receptor_coords: jnp.ndarray,
    ligand_elements: tuple[str, ...],
    receptor_elements: tuple[str, ...],
    parameters: HBondParameters,
) -> jnp.ndarray:
    ligand_donor_indices, ligand_r_o, ligand_depth = _hbond_donor_parameters(
        ligand_elements
    )
    receptor_donor_indices, receptor_r_o, receptor_depth = _hbond_donor_parameters(
        receptor_elements
    )
    ligand_acceptor_indices = _hbond_acceptor_indices(ligand_elements)
    receptor_acceptor_indices = _hbond_acceptor_indices(receptor_elements)

    pose_centroids = jnp.mean(poses_coords, axis=1, keepdims=True)
    pose_directions = _normalize_rows(poses_coords - pose_centroids)
    receptor_centroid = jnp.mean(receptor_coords, axis=0, keepdims=True)
    receptor_directions = _normalize_rows(receptor_coords - receptor_centroid)

    ligand_to_receptor = _hbond_matrix_energy(
        poses_coords[:, ligand_donor_indices, :],
        pose_directions[:, ligand_donor_indices, :],
        ligand_r_o,
        ligand_depth,
        receptor_coords[receptor_acceptor_indices],
        -receptor_directions[receptor_acceptor_indices],
        parameters.angle_power,
    )
    receptor_to_ligand = _hbond_matrix_energy(
        jnp.broadcast_to(
            receptor_coords[receptor_donor_indices][None, :, :],
            (poses_coords.shape[0], receptor_donor_indices.shape[0], 3),
        ),
        jnp.broadcast_to(
            receptor_directions[receptor_donor_indices][None, :, :],
            (poses_coords.shape[0], receptor_donor_indices.shape[0], 3),
        ),
        receptor_r_o,
        receptor_depth,
        poses_coords[:, ligand_acceptor_indices, :],
        -pose_directions[:, ligand_acceptor_indices, :],
        parameters.angle_power,
    )
    return ligand_to_receptor + receptor_to_ligand


def _infer_hbond_groups(
    coords: jnp.ndarray,
    elements: tuple[str, ...],
) -> tuple[tuple[HBondDonor, ...], tuple[HBondAcceptor, ...]]:
    centroid = jnp.mean(coords, axis=0)
    donors: list[HBondDonor] = []
    acceptors: list[HBondAcceptor] = []

    for idx, element in enumerate(elements):
        upper = element.upper()
        direction = _normalize(coords[idx] - centroid)

        if upper in ("N", "O", "S"):
            donor_type = (
                HBondDonorType.S_DONOR if upper == "S" else HBondDonorType.O_DONOR
            )
            donors.append(
                HBondDonor(
                    donor_atom_idx=idx,
                    hydrogen_idx=idx,
                    donor_type=donor_type,
                    position=coords[idx],
                    h_position=coords[idx] + 1.0 * direction,
                    direction=direction,
                )
            )

        if upper in ("N", "O", "S"):
            acceptors.append(
                HBondAcceptor(
                    acceptor_idx=idx,
                    position=coords[idx],
                    lone_pair_direction=-direction,
                )
            )

    return tuple(donors), tuple(acceptors)


def _solvation_parameter_arrays(
    elements: tuple[str, ...],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    sigma_values = []
    radius_values = []
    for element in elements:
        atom_config = get_solvation_params(element)
        sigma_values.append(atom_config.sigma)
        radius_values.append(atom_config.radius)
    return jnp.array(sigma_values), jnp.array(radius_values)


def _environment_burial_delta(
    target_positions: jnp.ndarray,
    target_sigma: jnp.ndarray,
    target_solvation_radii: jnp.ndarray,
    environment_positions: jnp.ndarray,
    environment_solvation_radii: jnp.ndarray,
    surface_directions: jnp.ndarray,
    probe_radius: float,
) -> jnp.ndarray:
    if environment_positions.shape[0] == 0:
        return jnp.array(0.0)

    sphere_radii = target_solvation_radii + probe_radius
    surface_points = (
        target_positions[:, None, :]
        + sphere_radii[:, None, None] * surface_directions[None, :, :]
    )
    deltas = surface_points[:, :, None, :] - environment_positions[None, None, :, :]
    distances = jnp.linalg.norm(deltas, axis=-1)
    burial_thresholds = environment_solvation_radii[None, None, :] + probe_radius
    buried = jnp.any(distances < burial_thresholds, axis=-1)

    sphere_areas = 4.0 * jnp.pi * sphere_radii**2
    area_per_point = sphere_areas / surface_directions.shape[0]
    buried_area = jnp.sum(buried, axis=1) * area_per_point
    return jnp.sum(target_sigma * buried_area / 1000.0)


def _desolvation_batch(
    poses_coords: jnp.ndarray,
    receptor_coords: jnp.ndarray,
    ligand_elements: tuple[str, ...],
    receptor_elements: tuple[str, ...],
    config: SASAConfig,
) -> jnp.ndarray:
    surface_directions = _fibonacci_sphere_points(config.n_points)
    ligand_sigma, ligand_solvation_radii = _solvation_parameter_arrays(ligand_elements)
    receptor_sigma, receptor_solvation_radii = _solvation_parameter_arrays(
        receptor_elements
    )

    def pose_desolvation(pose_coords: jnp.ndarray) -> jnp.ndarray:
        ligand_delta = _environment_burial_delta(
            pose_coords,
            ligand_sigma,
            ligand_solvation_radii,
            receptor_coords,
            receptor_solvation_radii,
            surface_directions,
            config.probe_radius,
        )
        receptor_delta = _environment_burial_delta(
            receptor_coords,
            receptor_sigma,
            receptor_solvation_radii,
            pose_coords,
            ligand_solvation_radii,
            surface_directions,
            config.probe_radius,
        )
        return ligand_delta + receptor_delta

    return jax.vmap(pose_desolvation)(poses_coords)


def _electrostatics_batch(
    poses_coords: jnp.ndarray,
    ligand_charges: jnp.ndarray,
    receptor_coords: jnp.ndarray,
    receptor_charges: jnp.ndarray,
    config: ElectrostaticsConfig,
) -> jnp.ndarray:
    diffs = poses_coords[:, :, None, :] - receptor_coords[None, None, :, :]
    dists = jnp.linalg.norm(diffs, axis=-1)
    within_cutoff = dists < config.cutoff
    dists_safe = jnp.maximum(dists, 0.5)
    q_i_q_j = ligand_charges[None, :, None] * receptor_charges[None, None, :]
    energies = q_i_q_j / (config.dielectric * dists_safe)
    return jnp.sum(jnp.where(within_cutoff, energies, 0.0), axis=(1, 2))


class DesolvationType(Enum):
    """Desolvation calculation method."""

    SASA = auto()
    AD4 = auto()


@dataclass(frozen=True)
class CompositeScoringConfig:
    """
    Composite scoring function configuration.

    OpenHCS Compliance:
    - @dataclass(frozen=True) for immutability
    - Explicit validation
    - Factory defaults
    """

    steric: VinardoConfig = VinardoConfig()
    hbond: HBondParameters = HBondParameters.for_oxygen_nitrogen()
    desolv_type: DesolvationType = DesolvationType.SASA
    sasa: SASAConfig = SASAConfig()
    electrostatics: ElectrostaticsConfig = ElectrostaticsConfig()
    charge_method: ChargeMethod = ChargeMethod.GASTEIGER

    w_steric: float = 0.30
    w_hbond: float = 0.25
    w_desolv: float = 0.20
    w_electrostatic: float = 0.25

    def validate(self) -> None:
        """Fail-loud validation."""
        total = self.w_steric + self.w_hbond + self.w_desolv + self.w_electrostatic
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights sum to {total}, should be 1.0")


class CompositeScoringCalculator(ABC):
    """ABC contract for composite scoring calculators."""

    @abstractmethod
    def compute(
        self,
        pose_coords: jnp.ndarray,
        receptor_coords: jnp.ndarray,
        receptor_radii: jnp.ndarray,
        ligand_radii: jnp.ndarray,
        pose_charges: jnp.ndarray,
        receptor_charges: jnp.ndarray,
        pose_elements: tuple[str, ...],
        receptor_elements: tuple[str, ...],
    ) -> float:
        """Compute total composite score."""

    @property
    @abstractmethod
    def config(self) -> CompositeScoringConfig:
        """Direct access to config."""


class AD4CompositeCalculator(CompositeScoringCalculator):
    """
    AutoDock4-style composite scoring calculator.

    OpenHCS Compliance:
    - Explicit dependency injection at construction
    - Fail-loud on invalid state
    - Pure compute method
    """

    def __init__(self, config: CompositeScoringConfig | None = None) -> None:
        self._config = config if config is not None else CompositeScoringConfig()
        self._config.validate()

        self._steric = create_scoring_backend(
            ScoringFamily.VINARDO, self._config.steric
        )
        self._hbond_on = create_hbond_calculator(
            HBondDonorType.O_DONOR, self._config.hbond
        )
        self._hbond_s = create_hbond_calculator(
            HBondDonorType.S_DONOR, HBondParameters.for_sulfur()
        )
        self._elec = create_electrostatics_calculator(self._config.electrostatics)
        self._charges = create_charge_assigner(self._config.charge_method)
        self._sasa = create_sasa_config(
            self._config.sasa.probe_radius,
            self._config.sasa.n_points,
        )

    @property
    def config(self) -> CompositeScoringConfig:
        return self._config

    def compute(
        self,
        pose_coords: jnp.ndarray,
        receptor_coords: jnp.ndarray,
        receptor_radii: jnp.ndarray,
        ligand_radii: jnp.ndarray,
        pose_charges: jnp.ndarray,
        receptor_charges: jnp.ndarray,
        pose_elements: tuple[str, ...],
        receptor_elements: tuple[str, ...],
    ) -> float:
        """Compute total composite score."""
        E_steric = self._steric.score_single(
            receptor_coords, pose_coords, receptor_radii, ligand_radii
        )

        pose_donors, pose_acceptors = _infer_hbond_groups(pose_coords, pose_elements)
        receptor_donors, receptor_acceptors = _infer_hbond_groups(
            receptor_coords, receptor_elements
        )
        E_hbond = total_hbond_energy(
            pose_donors,
            pose_acceptors,
            receptor_donors,
            receptor_acceptors,
            self._hbond_on,
        )

        E_desolv = float(
            _desolvation_batch(
                pose_coords[None, :, :],
                receptor_coords,
                pose_elements,
                receptor_elements,
                self._sasa,
            )[0]
        )

        E_elec = self._elec.compute(
            pose_coords, pose_charges, receptor_coords, receptor_charges
        )

        return (
            self._config.w_steric * E_steric
            + self._config.w_hbond * E_hbond
            + self._config.w_desolv * E_desolv
            + self._config.w_electrostatic * E_elec
        )


def create_composite_calculator(
    config: CompositeScoringConfig | None = None,
) -> CompositeScoringCalculator:
    """Factory function with explicit dependency injection."""
    cfg = config if config is not None else CompositeScoringConfig()
    cfg.validate()
    return AD4CompositeCalculator(cfg)


def score_composite_batch(
    receptor_coords: jnp.ndarray,
    receptor_radii: jnp.ndarray,
    poses_coords: jnp.ndarray,
    ligand_radii: jnp.ndarray,
    receptor_charges: jnp.ndarray | None = None,
    ligand_charges: jnp.ndarray | None = None,
    receptor_elements: tuple[str, ...] | None = None,
    ligand_elements: tuple[str, ...] | None = None,
    config: CompositeScoringConfig | None = None,
    steric_backend=None,
    chunk_size: int = 128,
) -> jnp.ndarray:
    """Chunked composite scoring to avoid large JAX memory peaks.

    - Runs JAX-accelerated steric backend per chunk
    - Runs host-side electrostatics per pose (small overhead)
    - H-bond and desolvation are computed host-side per pose within each chunk
    """
    cfg = config if config is not None else CompositeScoringConfig()
    cfg.validate()

    # Steric backend
    backend = steric_backend
    if backend is None:
        backend = create_scoring_backend(ScoringFamily.VINARDO, cfg.steric)

    receptor_elements_resolved = (
        receptor_elements if receptor_elements is not None else tuple()
    )
    ligand_elements_resolved = (
        ligand_elements if ligand_elements is not None else tuple()
    )

    n_poses = int(poses_coords.shape[0])
    energies = []

    for i in range(0, n_poses, chunk_size):
        j = min(n_poses, i + chunk_size)
        chunk = poses_coords[i:j]

        # JAX-accelerated steric scoring (batch)
        E_steric_chunk = backend.score_batch(
            receptor_coords, chunk, receptor_radii, ligand_radii
        )
        pose_charges = (
            ligand_charges
            if ligand_charges is not None
            else jnp.zeros(len(ligand_radii))
        )
        rec_charges = (
            receptor_charges
            if receptor_charges is not None
            else jnp.zeros(len(receptor_coords))
        )
        E_elec_chunk = _electrostatics_batch(
            chunk,
            pose_charges,
            receptor_coords,
            rec_charges,
            cfg.electrostatics,
        )
        E_desolv_chunk = _desolvation_batch(
            chunk,
            receptor_coords,
            ligand_elements_resolved,
            receptor_elements_resolved,
            cfg.sasa,
        )
        E_hbond_chunk = _hbond_batch(
            chunk,
            receptor_coords,
            ligand_elements_resolved,
            receptor_elements_resolved,
            cfg.hbond,
        )

        total_chunk = (
            cfg.w_steric * jnp.array(E_steric_chunk)
            + cfg.w_hbond * E_hbond_chunk
            + cfg.w_desolv * E_desolv_chunk
            + cfg.w_electrostatic * jnp.array(E_elec_chunk)
        )
        energies.append(total_chunk)

    if energies:
        return jnp.concatenate(energies, axis=0)
    return jnp.array([])
