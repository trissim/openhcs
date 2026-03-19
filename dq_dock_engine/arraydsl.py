"""Public Lean-backed ArrayDSL bridge API."""

from __future__ import annotations

from dq_dock_engine.codegen.arraydsl_runtime import (
    PRIMITIVE_REGISTRY,
    PrimitiveMetadata,
    get_registered_primitive,
)
from dq_dock_engine.generated.arraydsl_primitives import (
    applyCutoff,
    axisAngleQuaternion,
    coulombCutoff,
    distance,
    elemBinaryAdd,
    elemBinarySub,
    ewaldRealSpaceKernel,
    lennardJones,
    localRotationStencil3D,
    localTranslationStencil3D,
    map,
    minimumImagePairwiseDistances,
    normalizeProbabilityVector,
    noopBiasedProbabilityVectorLike,
    norm,
    ambiguityBandMask,
    pairwiseDistances,
    pairwiseDistances3D,
    quaternionDictionary8,
    reduce_sum,
    rigidTransform3D,
    rowWiseDistance,
    rowWiseNorm,
    stableArgmaxMasked,
    supportConditioning,
    topKWithTiesMask,
    sumPairPotentials,
    sumPairPotentials3D,
    sumPairPotentialsMatrix,
    uniformProbabilityVectorLike,
    typedLennardJonesCutoff,
    typedLennardJonesMatrix,
    upperTriangleMaskedSum,
)
from dq_dock_engine.generated.arraydsl_registry import ARRAYDSL_PRIMITIVES


def available_primitives() -> tuple[str, ...]:
    """Return all generated primitive names in stable order."""

    return tuple(metadata.name for metadata in ARRAYDSL_PRIMITIVES)


__all__ = [
    "ARRAYDSL_PRIMITIVES",
    "PRIMITIVE_REGISTRY",
    "PrimitiveMetadata",
    "available_primitives",
    "get_registered_primitive",
    "map",
    "reduce_sum",
    "elemBinaryAdd",
    "elemBinarySub",
    "norm",
    "rowWiseNorm",
    "distance",
    "rowWiseDistance",
    "supportConditioning",
    "normalizeProbabilityVector",
    "uniformProbabilityVectorLike",
    "noopBiasedProbabilityVectorLike",
    "topKWithTiesMask",
    "ambiguityBandMask",
    "stableArgmaxMasked",
    "axisAngleQuaternion",
    "localTranslationStencil3D",
    "localRotationStencil3D",
    "quaternionDictionary8",
    "rigidTransform3D",
    "pairwiseDistances",
    "pairwiseDistances3D",
    "minimumImagePairwiseDistances",
    "applyCutoff",
    "lennardJones",
    "sumPairPotentials",
    "sumPairPotentialsMatrix",
    "sumPairPotentials3D",
    "typedLennardJonesMatrix",
    "typedLennardJonesCutoff",
    "coulombCutoff",
    "upperTriangleMaskedSum",
    "ewaldRealSpaceKernel",
]
