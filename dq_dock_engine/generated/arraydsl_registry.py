"""Generated registry for Lean ArrayDSL primitives."""

from __future__ import annotations

from dq_dock_engine.codegen.arraydsl_runtime import PrimitiveMetadata, PRIMITIVE_REGISTRY, register_primitive
from dq_dock_engine.generated.arraydsl_primitives import (
    map,
    reduce_sum,
    elemBinaryAdd,
    elemBinarySub,
    norm,
    distance,
    pairwiseDistances,
    applyCutoff,
    lennardJones,
    sumPairPotentials,
)
from dq_dock_engine.proof_status import ProofStatus


register_primitive(
    PrimitiveMetadata(
        name='map',
        lean_symbol='DecisionQuotient.Computation.ArrayDSL.map',
        jax_module='jax',
        jax_symbol='vmap',
        lowering_kind='vmap',
        supports_grad=True,
        proof_ref=None,
        proof_status=None,
        callable=map,
    )
)

register_primitive(
    PrimitiveMetadata(
        name='reduce_sum',
        lean_symbol='DecisionQuotient.Computation.ArrayDSL.reduce_sum',
        jax_module='jax.numpy',
        jax_symbol='sum',
        lowering_kind='reduce_sum',
        supports_grad=True,
        proof_ref=None,
        proof_status=None,
        callable=reduce_sum,
    )
)

register_primitive(
    PrimitiveMetadata(
        name='elemBinaryAdd',
        lean_symbol='DecisionQuotient.Computation.ArrayDSL.elemBinaryAdd',
        jax_module='jax.numpy',
        jax_symbol='add',
        lowering_kind='elem_binary_add',
        supports_grad=True,
        proof_ref=None,
        proof_status=None,
        callable=elemBinaryAdd,
    )
)

register_primitive(
    PrimitiveMetadata(
        name='elemBinarySub',
        lean_symbol='DecisionQuotient.Computation.ArrayDSL.elemBinarySub',
        jax_module='jax.numpy',
        jax_symbol='subtract',
        lowering_kind='elem_binary_sub',
        supports_grad=True,
        proof_ref=None,
        proof_status=None,
        callable=elemBinarySub,
    )
)

register_primitive(
    PrimitiveMetadata(
        name='norm',
        lean_symbol='DecisionQuotient.Computation.ArrayDSL.norm',
        jax_module='jax.numpy.linalg',
        jax_symbol='norm',
        lowering_kind='norm',
        supports_grad=True,
        proof_ref='DecisionQuotient.Computation.ArrayDSL.norm_nonneg_bound',
        proof_status=ProofStatus.CERTIFIED,
        callable=norm,
    )
)

register_primitive(
    PrimitiveMetadata(
        name='distance',
        lean_symbol='DecisionQuotient.Computation.ArrayDSL.distance',
        jax_module='jax.numpy.linalg',
        jax_symbol='norm',
        lowering_kind='distance',
        supports_grad=True,
        proof_ref='DecisionQuotient.Computation.ArrayDSL.distance_triangle_bound',
        proof_status=ProofStatus.CERTIFIED,
        callable=distance,
    )
)

register_primitive(
    PrimitiveMetadata(
        name='pairwiseDistances',
        lean_symbol='DecisionQuotient.Computation.ArrayDSL.pairwiseDistances',
        jax_module='jax.numpy',
        jax_symbol='abs',
        lowering_kind='pairwise_distances',
        supports_grad=True,
        proof_ref=None,
        proof_status=None,
        callable=pairwiseDistances,
    )
)

register_primitive(
    PrimitiveMetadata(
        name='applyCutoff',
        lean_symbol='DecisionQuotient.Computation.ArrayDSL.applyCutoff',
        jax_module='jax.numpy',
        jax_symbol='where',
        lowering_kind='apply_cutoff',
        supports_grad=True,
        proof_ref=None,
        proof_status=None,
        callable=applyCutoff,
    )
)

register_primitive(
    PrimitiveMetadata(
        name='lennardJones',
        lean_symbol='DecisionQuotient.Computation.ArrayDSL.lennardJones',
        jax_module='jax.numpy',
        jax_symbol='where',
        lowering_kind='lennard_jones',
        supports_grad=True,
        proof_ref='DecisionQuotient.Computation.ArrayDSL.lennardJones',
        proof_status=ProofStatus.CERTIFIED,
        callable=lennardJones,
    )
)

register_primitive(
    PrimitiveMetadata(
        name='sumPairPotentials',
        lean_symbol='DecisionQuotient.Computation.ArrayDSL.sumPairPotentials',
        jax_module='jax.numpy',
        jax_symbol='sum',
        lowering_kind='sum_pair_potentials',
        supports_grad=True,
        proof_ref=None,
        proof_status=None,
        callable=sumPairPotentials,
    )
)


ARRAYDSL_PRIMITIVES = tuple(PRIMITIVE_REGISTRY[name] for name in [
    'map',
    'reduce_sum',
    'elemBinaryAdd',
    'elemBinarySub',
    'norm',
    'distance',
    'pairwiseDistances',
    'applyCutoff',
    'lennardJones',
    'sumPairPotentials'
])
