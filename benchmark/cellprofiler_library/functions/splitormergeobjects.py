"""
Converted from CellProfiler: SplitOrMergeObjects.

Compatibility facade for the OpenHCS CellProfiler backend implementation.
"""

from openhcs.processing.backends.cellprofiler.morphology import (
    DistanceSplitOrMergeMergeMethodStrategy,
    MergeObjectsStrategy,
    ParentSplitOrMergeMergeMethodStrategy,
    SplitObjectsStrategy,
    SplitOrMergeConvexHull,
    SplitOrMergeGuideImageFilter,
    SplitOrMergeIntensityMethod as IntensityMethod,
    SplitOrMergeMergeMethod as MergeMethod,
    SplitOrMergeMergeMethodStrategy,
    SplitOrMergeOperation as Operation,
    SplitOrMergeOperationStrategy,
    SplitOrMergeOutputObjectType as OutputObjectType,
    SplitOrMergeRequest,
    SplitOrMergeStats,
    split_or_merge_objects,
)

__all__ = [
    "DistanceSplitOrMergeMergeMethodStrategy",
    "IntensityMethod",
    "MergeMethod",
    "MergeObjectsStrategy",
    "Operation",
    "OutputObjectType",
    "ParentSplitOrMergeMergeMethodStrategy",
    "SplitObjectsStrategy",
    "SplitOrMergeConvexHull",
    "SplitOrMergeGuideImageFilter",
    "SplitOrMergeMergeMethodStrategy",
    "SplitOrMergeOperationStrategy",
    "SplitOrMergeRequest",
    "SplitOrMergeStats",
    "split_or_merge_objects",
]
