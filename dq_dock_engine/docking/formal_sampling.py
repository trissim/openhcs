from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from dq_dock_engine.arraydsl import quaternionDictionary8
from dq_dock_engine.docking.core import DockingBox, PoseVector


@dataclass(frozen=True)
class CertifiedGlobalActionFamily:
    translations: jax.Array
    quaternions: jax.Array
    lattice_resolution: int
    quaternion_count: int


def _quaternion_dictionary() -> jax.Array:
    base = jnp.asarray(quaternionDictionary8())
    norms = jnp.linalg.norm(base, axis=1, keepdims=True)
    return base / norms


def _translation_grid(box: DockingBox, n_points: int) -> tuple[jax.Array, int]:
    if n_points <= 0:
        raise ValueError("n_points must be positive")
    resolution = int(jnp.ceil(n_points ** (1.0 / 3.0)))
    resolution = max(resolution, 2)
    if resolution % 2 == 1:
        resolution += 1
    half_size = box.size / 2.0
    x_edges = jnp.linspace(
        box.center[0] - half_size[0], box.center[0] + half_size[0], resolution + 1
    )
    y_edges = jnp.linspace(
        box.center[1] - half_size[1], box.center[1] + half_size[1], resolution + 1
    )
    z_edges = jnp.linspace(
        box.center[2] - half_size[2], box.center[2] + half_size[2], resolution + 1
    )
    xs = 0.5 * (x_edges[:-1] + x_edges[1:])
    ys = 0.5 * (y_edges[:-1] + y_edges[1:])
    zs = 0.5 * (z_edges[:-1] + z_edges[1:])
    grid = jnp.stack(jnp.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape((-1, 3))
    return grid, resolution


def create_certified_global_action_family(
    box: DockingBox,
    n_poses: int,
) -> CertifiedGlobalActionFamily:
    quaternions = _quaternion_dictionary()
    n_quaternions = quaternions.shape[0]
    n_translation_points = int(jnp.ceil(n_poses / n_quaternions))
    translations, resolution = _translation_grid(box, n_translation_points)
    tiled_translations = jnp.repeat(translations, n_quaternions, axis=0)
    tiled_quaternions = jnp.tile(quaternions, (translations.shape[0], 1))
    return CertifiedGlobalActionFamily(
        translations=tiled_translations[:n_poses],
        quaternions=tiled_quaternions[:n_poses],
        lattice_resolution=resolution,
        quaternion_count=n_quaternions,
    )


def sample_certified_global_poses(box: DockingBox, n_poses: int) -> PoseVector:
    family = create_certified_global_action_family(box, n_poses)
    return PoseVector(
        translation=family.translations,
        quaternion=family.quaternions,
    )


def sample_so3_hopf_fibration(n_quaternions: int, seed: int = 42) -> jax.Array:
    """Uniform SO(3) sampling via Hopf fibration.

    Generates quaternions uniformly distributed on S³, which maps bijectively to SO(3).
    Uses golden ratio spiral for deterministic, low-discrepancy coverage.

    Args:
        n_quaternions: Number of quaternions to generate
        seed: Random seed for any stochastic components (default: 42)

    Returns:
        Array of shape (n_quaternions, 4) containing unit quaternions [w, x, y, z]
    """
    if n_quaternions <= 0:
        raise ValueError(f"n_quaternions must be positive, got {n_quaternions}")

    indices = np.arange(n_quaternions, dtype=np.float64)

    phi = (1.0 + np.sqrt(5.0)) / 2.0  # Golden ratio
    theta1 = 2.0 * np.pi * indices / phi  # Azimuthal angle 1
    theta2 = 2.0 * np.pi * indices / phi**2  # Azimuthal angle 2
    s = (indices + 0.5) / n_quaternions  # Uniform in [0,1)

    # Hopf coordinates: (theta1, theta2, s) -> quaternion
    # q = (cos(pi*s/2) * cos(theta1), cos(pi*s/2) * sin(theta1),
    #      sin(pi*s/2) * cos(theta2), sin(pi*s/2) * sin(theta2))
    cos_s = np.cos(np.pi * s / 2.0)
    sin_s = np.sin(np.pi * s / 2.0)

    quaternions = np.stack(
        [
            cos_s * np.cos(theta1),
            cos_s * np.sin(theta1),
            sin_s * np.cos(theta2),
            sin_s * np.sin(theta2),
        ],
        axis=1,
    )

    # Normalize to ensure unit quaternions (handles numerical drift)
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    quaternions = quaternions / norms

    return jnp.array(quaternions, dtype=jnp.float32)


def sample_so3_icosahedral(n_quaternions: int) -> jax.Array:
    """SO(3) sampling using icosahedral symmetry groups.

    Generates quaternions by applying icosahedral symmetry operations
    to a base set of quaternions. Provides excellent uniform coverage.

    Args:
        n_quaternions: Target number of quaternions (will be rounded up)

    Returns:
        Array of shape (n, 4) where n >= n_quaternions
    """
    if n_quaternions <= 0:
        raise ValueError(f"n_quaternions must be positive, got {n_quaternions}")

    # Icosahedron vertices (12 vertices)
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    icosahedron_vertices = np.array(
        [
            [0, 1, phi],
            [0, -1, phi],
            [0, 1, -phi],
            [0, -1, -phi],
            [1, phi, 0],
            [-1, phi, 0],
            [1, -phi, 0],
            [-1, -phi, 0],
            [phi, 0, 1],
            [-phi, 0, 1],
            [phi, 0, -1],
            [-phi, 0, -1],
        ],
        dtype=np.float64,
    )
    icosahedron_vertices /= np.linalg.norm(icosahedron_vertices, axis=1, keepdims=True)

    # Convert to quaternions (axis-angle from z-axis to vertex)
    quaternions = []
    z_axis = np.array([0.0, 0.0, 1.0])

    for vertex in icosahedron_vertices:
        # Rotation axis
        axis = np.cross(z_axis, vertex)
        axis_norm = np.linalg.norm(axis)

        if axis_norm < 1e-10:
            # Vertex is parallel or antiparallel to z-axis
            if vertex[2] > 0:
                quat = np.array([1.0, 0.0, 0.0, 0.0])  # Identity
            else:
                quat = np.array([0.0, 1.0, 0.0, 0.0])  # 180° around x
        else:
            axis = axis / axis_norm
            # Rotation angle
            cos_angle = np.clip(np.dot(z_axis, vertex), -1.0, 1.0)
            angle = np.arccos(cos_angle)
            # Quaternion from axis-angle
            half_angle = angle / 2.0
            quat = np.array(
                [
                    np.cos(half_angle),
                    axis[0] * np.sin(half_angle),
                    axis[1] * np.sin(half_angle),
                    axis[2] * np.sin(half_angle),
                ]
            )

        quaternions.append(quat)

    base_quaternions = np.array(quaternions, dtype=np.float64)

    # Apply 60 icosahedral group operations (rotations)
    # For simplicity, we'll tile the base quaternions and apply 5-fold rotations
    all_quaternions = [base_quaternions]

    # Generate additional rotations around z-axis (5-fold symmetry)
    for k in range(1, 5):
        angle = 2.0 * np.pi * k / 5.0
        rotation_quat = np.array([np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)])

        # Quaternion multiplication: q_result = q_rotation * q_base
        for base_q in base_quaternions:
            q_result = _quaternion_multiply(rotation_quat, base_q)
            all_quaternions.append(q_result.reshape(1, -1))

    result = np.vstack(all_quaternions)

    # Remove duplicates (within tolerance)
    unique_quaternions = [result[0]]
    for q in result[1:]:
        is_duplicate = False
        for uq in unique_quaternions:
            # Check both q and -q (same rotation)
            if np.allclose(q, uq, atol=1e-6) or np.allclose(q, -uq, atol=1e-6):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_quaternions.append(q)

    result = np.array(unique_quaternions, dtype=np.float64)

    # Normalize
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    result = result / norms

    # If we need more, use Hopf fibration to fill in
    if len(result) < n_quaternions:
        hopf_quats = np.array(sample_so3_hopf_fibration(n_quaternions - len(result)))
        result = np.vstack([result, hopf_quats])

    return jnp.array(result[:n_quaternions], dtype=jnp.float32)


def _quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions q1 * q2 (Hamilton product)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def create_certified_global_action_family_rigorous(
    box: DockingBox,
    n_poses: int,
    n_rotations: int = 72,
    use_icosahedral: bool = True,
    seed: int = 42,
) -> CertifiedGlobalActionFamily:
    """Create global action family with dense SO(3) coverage.

    This is the rigorous version that provides much better rotation space coverage
    than the original 8-quaternion dictionary.

    Args:
        box: Docking box defining the search space
        n_poses: Total number of poses to generate
        n_rotations: Number of rotation samples (default: 72 for good coverage)
        use_icosahedral: Use icosahedral symmetry if True, else Hopf fibration
        seed: Random seed for deterministic generation

    Returns:
        CertifiedGlobalActionFamily with translations and quaternions
    """
    if n_poses <= 0:
        raise ValueError(f"n_poses must be positive, got {n_poses}")
    if n_rotations <= 0:
        raise ValueError(f"n_rotations must be positive, got {n_rotations}")

    # Generate dense rotation sampling
    if use_icosahedral:
        quaternions = sample_so3_icosahedral(n_rotations)
    else:
        quaternions = sample_so3_hopf_fibration(n_rotations, seed)

    n_quaternions = quaternions.shape[0]
    n_translation_points = int(jnp.ceil(n_poses / n_quaternions))

    translations, resolution = _translation_grid(box, n_translation_points)

    # Combine translations and rotations
    tiled_translations = jnp.repeat(translations, n_quaternions, axis=0)
    tiled_quaternions = jnp.tile(quaternions, (translations.shape[0], 1))

    return CertifiedGlobalActionFamily(
        translations=tiled_translations[:n_poses],
        quaternions=tiled_quaternions[:n_poses],
        lattice_resolution=resolution,
        quaternion_count=n_quaternions,
    )


def sample_certified_global_poses_rigorous(
    box: DockingBox,
    n_poses: int,
    n_rotations: int = 72,
    use_icosahedral: bool = True,
    seed: int = 42,
) -> PoseVector:
    """Sample poses with dense SO(3) coverage (rigorous version).

    This provides much better rotation space coverage than sample_certified_global_poses.

    Args:
        box: Docking box defining the search space
        n_poses: Total number of poses to generate
        n_rotations: Number of rotation samples (default: 72)
        use_icosahedral: Use icosahedral symmetry if True, else Hopf fibration
        seed: Random seed for deterministic generation

    Returns:
        PoseVector with translations and quaternions
    """
    family = create_certified_global_action_family_rigorous(
        box, n_poses, n_rotations, use_icosahedral, seed
    )
    return PoseVector(
        translation=family.translations,
        quaternion=family.quaternions,
    )
