"""
Pocket-guided pose sampling using pure JAX.

OpenHCS Compliance:
- Enum-driven sampling strategies
- Frozen dataclasses for results
- ABC contract for samplers
- Factory function with explicit dependency injection
- Pure JAX functions with @jax.jit/@jax.vmap
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto


# =============================================================================
# SAMPLING STRATEGY ENUM (OpenHCS Pattern)
# =============================================================================


class SamplingStrategy(Enum):
    """
    Pocket-guided sampling strategies.

    OpenHCS Compliance:
    - Enum-driven behavior selection
    - No string-based dispatch
    """

    RANDOM = auto()  # Pure random (baseline)
    GUIDED = auto()  # Fully guided by pockets
    HYBRID = auto()  # 50% guided, 50% random

    def keep_ratio(self) -> float:
        """Return guided sampling ratio."""
        match self:
            case SamplingStrategy.RANDOM:
                return 0.0
            case SamplingStrategy.GUIDED:
                return 1.0
            case SamplingStrategy.HYBRID:
                return 0.5

    @staticmethod
    def recommended() -> SamplingStrategy:
        """Literature-recommended default (Fpocket/DOCK6 style)."""
        return SamplingStrategy.HYBRID


# =============================================================================
# POSE TEMPLATE (Frozen Dataclass)
# =============================================================================


@dataclass(frozen=True)
class PoseTemplate:
    """
    A template pose for sampling.

    OpenHCS Compliance:
    - Frozen dataclass (immutable)
    - Explicit types
    """

    translation: jnp.ndarray
    rotation_bias: jnp.ndarray
    uncertainty: tuple[float, float]


# =============================================================================
# SAMPLING RESULT (Frozen Dataclass)
# =============================================================================


@dataclass(frozen=True)
class SamplingResult:
    """
    Immutable result of intelligent pose sampling.

    OpenHCS Compliance:
    - @dataclass(frozen=True) for immutability
    - Explicit types
    - Tuple for collections
    """

    translations: jnp.ndarray
    quaternions: jnp.ndarray
    strategy: SamplingStrategy
    n_guided: int
    n_random: int
    templates_used: int


# =============================================================================
# ABC CONTRACT FOR POCKET-GUIDED SAMPLERS
# =============================================================================


class PocketSampler(ABC):
    """
    ABC contract for pocket-guided samplers.

    OpenHCS Compliance:
    - ABC enforces explicit contract
    """

    @abstractmethod
    def sample(
        self,
        key: jax.Array,
        n_poses: int,
        strategy: SamplingStrategy,
        pocket_coords: jnp.ndarray,
        pocket_elements: tuple[str, ...],
        ligand_com: jnp.ndarray,
        box_center: jnp.ndarray,
        box_size: float,
    ) -> SamplingResult:
        """Sample poses with specified strategy."""


# =============================================================================
# PURE JAX SAMPLING FUNCTIONS
# =============================================================================


@jax.jit
def _sample_gaussian_noise(
    key: jax.Array,
    mean: jnp.ndarray,
    std: float,
) -> jnp.ndarray:
    """
    Sample Gaussian noise around mean.

    OpenHCS Compliance:
    - Pure function
    - jax.jit for GPU acceleration
    """
    noise = jax.random.normal(key, shape=mean.shape)
    return mean + std * noise


def _select_template_indices(
    key: jax.Array,
    weights: jnp.ndarray,
    n_samples: int,
) -> jnp.ndarray:
    """
    Select template indices weighted by subpocket weights.
    """
    return jax.random.choice(
        key, len(weights), shape=(n_samples,), p=weights, replace=True
    )


@jax.jit
def _sample_translations_vmap(
    keys: jnp.ndarray,
    centers: jnp.ndarray,
    stds: jnp.ndarray,
) -> jnp.ndarray:
    """
    Vectorized translation sampling.
    """

    def sample_one(args):
        key, center, std = args
        noise = jax.random.normal(key, shape=(3,))
        return center + std * noise

    return jax.vmap(sample_one)((keys, centers, stds))


@jax.jit
def _sample_rotations_vmap(
    keys: jnp.ndarray,
    rot_biases: jnp.ndarray,
    rot_stds: jnp.ndarray,
) -> jnp.ndarray:
    """
    Vectorized rotation sampling.
    """

    def sample_one(args):
        key, bias, std = args
        noise = jax.random.normal(key, shape=(4,)) * std
        q_noisy = bias + noise
        return q_noisy / jnp.linalg.norm(q_noisy)

    return jax.vmap(sample_one)((keys, rot_biases, rot_stds))


@jax.jit
def _axis_angle_to_quaternion(axis: jnp.ndarray, angle: float) -> jnp.ndarray:
    """
    Convert axis-angle representation to quaternion.
    """
    half_angle = angle / 2.0
    sin_half = jnp.sin(half_angle)
    cos_half = jnp.cos(half_angle)

    return jnp.array(
        [
            cos_half,
            axis[0] * sin_half,
            axis[1] * sin_half,
            axis[2] * sin_half,
        ]
    )


@jax.jit
def _sample_uniform_quaternions_batch(
    keys: jnp.ndarray,
) -> jnp.ndarray:
    """
    Sample batch of uniform quaternions using Shoemake's algorithm.
    """

    def sample_one(key):
        u1 = jax.random.uniform(key, shape=(3,))
        sqrt_1_minus_u2_sq = jnp.sqrt(1.0 - u1[0])
        sin_2pi_u2 = jnp.sin(2.0 * jnp.pi * u1[1])
        cos_2pi_u2 = jnp.cos(2.0 * jnp.pi * u1[1])
        sin_2pi_u3 = jnp.sin(2.0 * jnp.pi * u1[2])
        cos_2pi_u3 = jnp.cos(2.0 * jnp.pi * u1[2])

        return jnp.array(
            [
                sqrt_1_minus_u2_sq * sin_2pi_u2,
                sqrt_1_minus_u2_sq * cos_2pi_u2,
                u1[0] * sin_2pi_u3,
                u1[0] * cos_2pi_u3,
            ]
        )

    return jax.vmap(sample_one)(keys)


def sample_biased_translations(
    key: jax.Array,
    templates: tuple[PoseTemplate, ...],
    weights: jnp.ndarray,
    n_poses: int,
) -> jnp.ndarray:
    """
    Sample translations biased toward sub-pockets.

    OpenHCS Compliance:
    - Pure function (no side effects)
    - Tuple input (immutable)
    - jax.jit for GPU acceleration
    - No Python loops
    """
    key_select, key_noise = jax.random.split(key)

    indices = _select_template_indices(key_select, weights, n_poses)

    centers = jnp.array([templates[i].translation for i in indices])
    stds = jnp.array([templates[i].uncertainty[0] for i in indices])

    keys_noise = jax.random.split(key_noise, n_poses)
    translations = _sample_translations_vmap(keys_noise, centers, stds)

    return translations


def sample_biased_rotations(
    key: jax.Array,
    templates: tuple[PoseTemplate, ...],
    indices: jnp.ndarray,
    n_poses: int,
) -> jnp.ndarray:
    """
    Sample rotations biased toward template orientations.
    """
    keys = jax.random.split(key, n_poses)

    rot_biases = jnp.array([templates[i].rotation_bias for i in indices])
    rot_stds = jnp.array([templates[i].uncertainty[1] for i in indices])

    return _sample_rotations_vmap(keys, rot_biases, rot_stds)


# =============================================================================
# POCKET-GUIDED SAMPLER IMPLEMENTATION
# =============================================================================


class PocketGuidedSampler(PocketSampler):
    """
    Pocket-guided pose sampler.

    OpenHCS Compliance:
    - Explicit dependency injection
    - ABC contract for extensibility
    - Fail-loud validation
    """

    def __init__(
        self,
        n_templates_per_subpocket: int = 10,
        trans_std: float = 1.0,
        rot_std: float = 0.5,
    ) -> None:
        self._n_templates = n_templates_per_subpocket
        self._trans_std = trans_std
        self._rot_std = rot_std

    def sample(
        self,
        key: jax.Array,
        n_poses: int,
        strategy: SamplingStrategy,
        pocket_coords: jnp.ndarray,
        pocket_elements: tuple[str, ...],
        ligand_com: jnp.ndarray,
        box_center: jnp.ndarray,
        box_size: float,
    ) -> SamplingResult:
        """
        Sample poses with specified strategy.
        """
        match strategy:
            case SamplingStrategy.RANDOM:
                return self._sample_random(
                    key, n_poses, ligand_com, box_center, box_size
                )

            case SamplingStrategy.GUIDED:
                return self._sample_guided(
                    key, n_poses, pocket_coords, pocket_elements, ligand_com
                )

            case SamplingStrategy.HYBRID:
                n_guided = int(n_poses * strategy.keep_ratio())
                n_random = n_poses - n_guided

                key_guided, key_random = jax.random.split(key)

                guided_result = self._sample_guided(
                    key_guided, n_guided, pocket_coords, pocket_elements, ligand_com
                )
                random_result = self._sample_random(
                    key_random, n_random, ligand_com, box_center, box_size
                )

                return SamplingResult(
                    translations=jnp.concatenate(
                        [guided_result.translations, random_result.translations]
                    ),
                    quaternions=jnp.concatenate(
                        [guided_result.quaternions, random_result.quaternions]
                    ),
                    strategy=strategy,
                    n_guided=n_guided,
                    n_random=n_random,
                    templates_used=guided_result.templates_used,
                )

            case _:
                raise ValueError(f"Unknown SamplingStrategy: {strategy}")

    def _sample_random(
        self,
        key: jax.Array,
        n_poses: int,
        ligand_com: jnp.ndarray,
        box_center: jnp.ndarray,
        box_size: float,
    ) -> SamplingResult:
        """Pure random sampling."""
        half_size = box_size / 2.0

        key_trans, key_rot = jax.random.split(key)

        translations = jax.random.uniform(
            key_trans,
            shape=(n_poses, 3),
            minval=box_center - half_size,
            maxval=box_center + half_size,
        )

        quaternions = _sample_uniform_quaternions_batch(
            jax.random.split(key_rot, n_poses)
        )

        return SamplingResult(
            translations=translations,
            quaternions=quaternions,
            strategy=SamplingStrategy.RANDOM,
            n_guided=0,
            n_random=n_poses,
            templates_used=0,
        )

    def _sample_guided(
        self,
        key: jax.Array,
        n_poses: int,
        pocket_coords: jnp.ndarray,
        pocket_elements: tuple[str, ...],
        ligand_com: jnp.ndarray,
    ) -> SamplingResult:
        """Pocket-guided sampling."""
        from dq_dock_engine.docking.pocket_analysis import (
            compute_pocket_shape,
            compute_accessibility,
            detect_pharmacophore_features,
            ACCESSIBILITY_CONFIG,
        )

        accessibility = compute_accessibility(
            pocket_coords, ACCESSIBILITY_CONFIG.probe_radius
        )
        features = detect_pharmacophore_features(
            pocket_coords, pocket_elements, accessibility, ACCESSIBILITY_CONFIG
        )

        pocket_shape = compute_pocket_shape(pocket_coords)

        templates, template_weights = self._generate_templates(
            pocket_coords, pocket_shape, features
        )

        key_trans, key_rot = jax.random.split(key)

        indices = _select_template_indices(key_trans, template_weights, n_poses)
        translations = sample_biased_translations(
            key_trans, templates, template_weights, n_poses
        )
        quaternions = sample_biased_rotations(key_rot, templates, indices, n_poses)

        return SamplingResult(
            translations=translations,
            quaternions=quaternions,
            strategy=SamplingStrategy.GUIDED,
            n_guided=n_poses,
            n_random=0,
            templates_used=len(templates),
        )

    def _generate_templates(
        self,
        pocket_coords: jnp.ndarray,
        pocket_shape,
        features,
    ) -> tuple[tuple[PoseTemplate, ...], jnp.ndarray]:
        """Generate pose templates and weights."""
        all_templates = []
        weights = []

        com = pocket_shape.center_of_mass
        eigenvectors = pocket_shape.principal_axes

        for j in range(self._n_templates):
            axis_idx = j % 3
            axis = eigenvectors[:, axis_idx]
            angle = (2.0 * jnp.pi / self._n_templates) * (j // 3)

            rotation = _axis_angle_to_quaternion(axis, angle)

            template = PoseTemplate(
                translation=com,
                rotation_bias=rotation,
                uncertainty=(self._trans_std, self._rot_std),
            )

            all_templates.append(template)
            weights.append(1.0 + 0.5 * len(features))

        total_weight = sum(weights)
        normalized_weights = jnp.array([w / total_weight for w in weights])

        return tuple(all_templates), normalized_weights


# =============================================================================
# FACTORY FUNCTION (Explicit Dependency Injection)
# =============================================================================


def create_pocket_sampler(
    strategy: SamplingStrategy | None = None,
    n_templates_per_subpocket: int = 10,
) -> PocketGuidedSampler:
    """
    Factory function with explicit dependency injection.

    OpenHCS Compliance:
    - Explicit factory, not __init__
    - No hidden object creation
    """
    return PocketGuidedSampler(
        n_templates_per_subpocket=n_templates_per_subpocket,
    )


# =============================================================================
# INTEGRATION WITH DOCKINGBOX
# =============================================================================


def sample_intelligent_poses(
    key: jax.Array,
    box_center: jnp.ndarray,
    box_size: float,
    n_poses: int,
    protein_coords: jnp.ndarray,
    receptor_elements: tuple[str, ...] | None,
    ligand_com: jnp.ndarray,
    strategy: SamplingStrategy = SamplingStrategy.HYBRID,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Intelligent pose sampling using pocket analysis.

    OpenHCS Compliance:
    - Pure function
    - Explicit dependencies
    - Enum-driven strategy selection
    - Factory function for sampler

    Args:
        key: JAX random key
        box_center: Center of docking box
        box_size: Size of docking box
        n_poses: Number of poses to sample
        protein_coords: Protein coordinates for pocket analysis
        receptor_elements: Receptor elements aligned with protein_coords
        ligand_com: Ligand center of mass
        strategy: Sampling strategy (RANDOM, GUIDED, HYBRID)

    Returns:
        (translations, quaternions) tuple
    """
    pocket_radius = box_size / 2.0

    distances = jnp.linalg.norm(protein_coords - box_center, axis=1)
    pocket_mask = distances < pocket_radius
    pocket_coords = protein_coords[pocket_mask]

    if receptor_elements is None:
        pocket_elements = tuple(["C"] * len(pocket_coords))
    else:
        pocket_elements = tuple(
            element for element, keep in zip(receptor_elements, pocket_mask) if keep
        )

    sampler = create_pocket_sampler()

    result = sampler.sample(
        key=key,
        n_poses=n_poses,
        strategy=strategy,
        pocket_coords=pocket_coords,
        pocket_elements=pocket_elements,
        ligand_com=ligand_com,
        box_center=box_center,
        box_size=box_size,
    )

    return result.translations, result.quaternions
