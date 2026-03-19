import jax
import jax.numpy as jnp

from dq_dock_engine.arraydsl import (
    ARRAYDSL_PRIMITIVES,
    PRIMITIVE_REGISTRY,
    available_primitives,
    applyCutoff,
    coulombCutoff,
    elemBinaryAdd,
    ewaldRealSpaceKernel,
    get_registered_primitive,
    lennardJones,
    minimumImagePairwiseDistances,
    normalizeProbabilityVector,
    noopBiasedProbabilityVectorLike,
    norm,
    ambiguityBandMask,
    pairwiseDistances3D,
    reduce_sum,
    rigidTransform3D,
    rowWiseDistance,
    rowWiseNorm,
    stableArgmaxMasked,
    supportConditioning,
    topKWithTiesMask,
    sumPairPotentials3D,
    typedLennardJonesCutoff,
    typedLennardJonesMatrix,
    uniformProbabilityVectorLike,
    upperTriangleMaskedSum,
)
from dq_dock_engine.codegen.arraydsl_codegen import (
    DEFAULT_EXPORT_PATH,
    generate_modules,
)
from dq_dock_engine.physics.kernels import (
    apply_cutoff,
    coulomb_cutoff,
    distance as kernel_distance,
    ewald_real_space_kernel,
    lennard_jones_potential,
    minimum_image_pairwise_distances,
    norm as kernel_norm,
    pairwise_distances,
    rigid_transform_3d,
    typed_lennard_jones_cutoff,
    typed_lennard_jones_matrix,
    upper_triangle_masked_sum,
)
from dq_dock_engine.proof_status import ProofStatus, get_status, get_theorem


def test_arraydsl_public_registry_matches_exported_metadata():
    assert available_primitives() == tuple(
        metadata.name for metadata in ARRAYDSL_PRIMITIVES
    )
    assert set(PRIMITIVE_REGISTRY) == set(available_primitives())


def test_arraydsl_proof_metadata_is_attached_to_generated_callable():
    metadata = get_registered_primitive("norm")

    assert metadata.proof_status is ProofStatus.CERTIFIED
    assert get_status(norm) is ProofStatus.CERTIFIED
    assert (
        get_theorem(norm) == "DecisionQuotient.Computation.ArrayDSL.norm_nonneg_bound"
    )


def test_arraydsl_generated_wrappers_execute_with_jax_arrays():
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([4.0, 5.0, 6.0])
    batched_x = jnp.array([[1.0, 2.0, 2.0], [3.0, 0.0, 4.0]])
    batched_y = jnp.array([[0.0, 2.0, 2.0], [0.0, 0.0, 4.0]])
    coords1 = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    coords2 = jnp.array([[0.0, 1.0, 0.0], [0.0, 0.0, 2.0]])
    quaternion = jnp.array([0.0, 0.0, 0.0, 1.0])
    translation = jnp.array([1.0, -1.0, 0.5])
    box_size = jnp.array([10.0, 10.0, 10.0])
    epsilons = jnp.array([[0.5, 0.25], [0.125, 0.75]])
    sigmas = jnp.array([[1.2, 1.1], [1.0, 1.3]])
    charges1 = jnp.array([1.0, -1.0])
    charges2 = jnp.array([-0.5, 0.5])
    values = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    mask = jnp.array([[False, True], [True, False]])
    support_mask = jnp.array([True, False])
    weights = jnp.array([2.0, 3.0])
    utilities = jnp.array([3.0, 1.0, 2.0])
    template = jnp.zeros((3,), dtype=jnp.float32)

    assert float(reduce_sum(x)) == 6.0
    assert tuple(elemBinaryAdd(x, y).tolist()) == (5.0, 7.0, 9.0)
    assert round(float(norm(x)), 6) == round(14.0**0.5, 6)
    assert tuple(rowWiseNorm(batched_x).tolist()) == (3.0, 5.0)
    assert tuple(rowWiseDistance(batched_x, batched_y).tolist()) == (1.0, 3.0)
    assert jnp.allclose(
        pairwiseDistances3D(coords1, coords2),
        jnp.array([[1.0, 2.0], [2.0**0.5, 5.0**0.5]]),
    )
    assert jnp.allclose(
        rigidTransform3D(coords1, quaternion, translation),
        jnp.array([[1.0, -1.0, 0.5], [0.0, -1.0, 0.5]]),
    )
    assert jnp.allclose(
        minimumImagePairwiseDistances(coords1, coords2, box_size),
        pairwiseDistances3D(coords1, coords2),
    )
    assert jnp.allclose(
        jnp.asarray(
            typedLennardJonesMatrix(
                pairwiseDistances3D(coords1, coords2), epsilons, sigmas
            )
        ),
        4.0
        * epsilons
        * (
            (sigmas / pairwiseDistances3D(coords1, coords2)) ** 12
            - (sigmas / pairwiseDistances3D(coords1, coords2)) ** 6
        ),
    )
    assert jnp.allclose(
        upperTriangleMaskedSum(values, mask),
        2.0,
    )
    assert jnp.allclose(
        ewaldRealSpaceKernel(pairwiseDistances3D(coords1, coords2), 0.5),
        jnp.exp(-((0.5 * pairwiseDistances3D(coords1, coords2)) ** 2))
        / pairwiseDistances3D(coords1, coords2),
    )
    assert jnp.isfinite(
        coulombCutoff(
            charges1, charges2, pairwiseDistances3D(coords1, coords2), 3.0, 1.0
        )
    )
    assert jnp.isfinite(
        typedLennardJonesCutoff(
            pairwiseDistances3D(coords1, coords2), epsilons, sigmas, 3.0
        )
    )
    assert jnp.allclose(
        jnp.asarray(supportConditioning(weights, support_mask)),
        jnp.array([2.0, 0.0]),
    )
    assert jnp.allclose(
        jnp.asarray(normalizeProbabilityVector(weights)),
        jnp.array([0.4, 0.6]),
    )
    assert jnp.allclose(
        jnp.asarray(uniformProbabilityVectorLike(template)),
        jnp.array([1 / 3, 1 / 3, 1 / 3], dtype=jnp.float32),
    )
    assert jnp.allclose(
        jnp.asarray(noopBiasedProbabilityVectorLike(template, 0.4)),
        jnp.array([0.4, 0.3, 0.3], dtype=jnp.float32),
    )
    assert jnp.array_equal(
        topKWithTiesMask(utilities, 1), jnp.array([True, False, False])
    )
    assert jnp.array_equal(
        ambiguityBandMask(utilities, 1, 1.0), jnp.array([True, False, True])
    )
    assert (
        int(
            stableArgmaxMasked(
                jnp.array([0.4, 0.3, 0.3]), jnp.array([True, True, True])
            )
        )
        == 0
    )


def test_arraydsl_molecular_primitives_match_physics_kernels():
    batched_x = jnp.array([[1.0, 2.0, 2.0], [3.0, 0.0, 4.0]])
    batched_y = jnp.array([[0.0, 2.0, 2.0], [0.0, 0.0, 4.0]])
    coords1 = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    coords2 = jnp.array([[0.0, 1.0, 0.0], [0.0, 0.0, 2.0]])
    dists = pairwiseDistances3D(coords1, coords2)
    box_size = jnp.array([10.0, 10.0, 10.0])
    quaternion = jnp.array([0.0, 0.0, 0.0, 1.0])
    translation = jnp.array([1.0, -1.0, 0.5])
    epsilons = jnp.array([[0.5, 0.25], [0.125, 0.75]])
    sigmas = jnp.array([[1.2, 1.1], [1.0, 1.3]])
    charges1 = jnp.array([1.0, -1.0])
    charges2 = jnp.array([-0.5, 0.5])
    mask = jnp.array([[False, True], [True, False]])

    assert jnp.allclose(rowWiseNorm(batched_x), jnp.asarray(kernel_norm(batched_x)))
    assert jnp.allclose(
        rowWiseDistance(batched_x, batched_y),
        jnp.asarray(kernel_distance(batched_x, batched_y)),
    )
    assert jnp.allclose(
        pairwiseDistances3D(coords1, coords2),
        jnp.asarray(pairwise_distances(coords1, coords2)),
    )
    assert jnp.allclose(
        jnp.asarray(applyCutoff(dists, 1.5)),
        jnp.asarray(apply_cutoff(dists, 1.5)),
    )
    assert jnp.allclose(
        jnp.asarray(lennardJones(0.5, 1.2, dists)),
        jnp.asarray(lennard_jones_potential(dists, 0.5, 1.2)),
    )
    assert jnp.allclose(
        sumPairPotentials3D(coords1, coords2, 3.0, 0.5, 1.2),
        jnp.sum(
            jnp.asarray(lennard_jones_potential(apply_cutoff(dists, 3.0), 0.5, 1.2))
        ),
    )
    assert jnp.allclose(
        rigidTransform3D(coords1, quaternion, translation),
        jnp.asarray(rigid_transform_3d(coords1, quaternion, translation)),
    )
    assert jnp.allclose(
        minimumImagePairwiseDistances(coords1, coords2, box_size),
        jnp.asarray(minimum_image_pairwise_distances(coords1, coords2, box_size)),
    )
    assert jnp.allclose(
        jnp.asarray(typedLennardJonesMatrix(dists, epsilons, sigmas)),
        jnp.asarray(typed_lennard_jones_matrix(dists, epsilons, sigmas)),
    )
    assert jnp.allclose(
        typedLennardJonesCutoff(dists, epsilons, sigmas, 3.0),
        jnp.asarray(typed_lennard_jones_cutoff(dists, epsilons, sigmas, 3.0)),
    )
    assert jnp.allclose(
        coulombCutoff(charges1, charges2, dists, 3.0, 1.0),
        jnp.asarray(coulomb_cutoff(charges1, charges2, dists, 3.0, 1.0)),
    )
    assert jnp.allclose(
        upperTriangleMaskedSum(dists, mask),
        jnp.asarray(upper_triangle_masked_sum(dists, mask)),
    )
    assert jnp.allclose(
        ewaldRealSpaceKernel(dists, 0.5),
        jnp.asarray(ewald_real_space_kernel(dists, 0.5)),
    )


def test_arraydsl_molecular_primitives_are_differentiable():
    coords1 = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    coords2 = jnp.array([[0.0, 1.0, 0.0], [0.0, 0.0, 2.0]])

    def energy(c1: jnp.ndarray) -> jnp.ndarray:
        return sumPairPotentials3D(c1, coords2, 3.0, 0.5, 1.2)

    grad = jax.grad(energy)(coords1)
    assert grad.shape == coords1.shape

    def transform_energy(c1: jnp.ndarray) -> jnp.ndarray:
        moved = rigidTransform3D(
            c1, jnp.array([1.0, 0.0, 0.0, 0.0]), jnp.array([0.1, -0.2, 0.3])
        )
        return sumPairPotentials3D(moved, coords2, 3.0, 0.5, 1.2)

    transform_grad = jax.grad(transform_energy)(coords1)
    assert transform_grad.shape == coords1.shape


def test_codegen_can_write_bridge_modules_to_requested_directory(tmp_path):
    output_dir = tmp_path / "generated"

    generated_paths = generate_modules(
        DEFAULT_EXPORT_PATH, output_dir, package_name="dq_dock_engine.generated"
    )

    assert [path.name for path in generated_paths] == [
        "__init__.py",
        "arraydsl_primitives.py",
        "arraydsl_registry.py",
    ]
    assert (output_dir / "arraydsl_primitives.py").exists()
    assert (output_dir / "arraydsl_registry.py").exists()
