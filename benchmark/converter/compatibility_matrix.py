"""CellProfiler compatibility coverage matrix."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from benchmark.cellprofiler_library import (
    canonical_module_name,
    get_contract,
    list_modules,
    require_function,
)
from openhcs.processing.backends.lib_registry.unified_registry import (
    ProcessingContract,
)

from .cppipe_corpus import CPPipeCorpusStatus, in_tree_cppipe_corpus
from .parser import CPPipeParser
from .processing_contract_resolution import (
    ProcessingContractResolutionSource,
    resolve_processing_contract,
)
from .symbol_table import ModuleContractBuilder


class ArtifactContractCoverage(str, Enum):
    """How artifact semantics are known for one CellProfiler module."""

    DECLARED_BUILDER = "declared_builder"
    GENERIC_INFERENCE = "generic_inference"


class ModuleCorpusCoverage(str, Enum):
    """Whether one module appears in accepted in-tree real pipelines."""

    SUPPORTED_CORPUS = "supported_corpus"
    KNOWN_INVALID_CORPUS = "known_invalid_corpus"
    NOT_IN_CORPUS = "not_in_corpus"


@dataclass(frozen=True, slots=True)
class ModuleCompatibilityCoverage:
    """Compatibility coverage for one absorbed CellProfiler module."""

    module_name: str
    function_name: str
    importable: bool
    processing_contract: ProcessingContract | None
    processing_contract_source: ProcessingContractResolutionSource | None
    processing_contract_error: str | None
    artifact_contract_coverage: ArtifactContractCoverage
    corpus_coverage: ModuleCorpusCoverage

    @property
    def has_processing_contract(self) -> bool:
        return self.processing_contract is not None


@dataclass(frozen=True, slots=True)
class CellProfilerCompatibilityReport:
    """Typed compatibility matrix over absorbed modules and real pipelines."""

    modules: tuple[ModuleCompatibilityCoverage, ...]

    @property
    def unresolved_processing_contracts(
        self,
    ) -> tuple[ModuleCompatibilityCoverage, ...]:
        return tuple(
            module for module in self.modules if not module.has_processing_contract
        )

    @property
    def supported_corpus_processing_contract_gaps(
        self,
    ) -> tuple[ModuleCompatibilityCoverage, ...]:
        return tuple(
            module
            for module in self.unresolved_processing_contracts
            if module.corpus_coverage is ModuleCorpusCoverage.SUPPORTED_CORPUS
        )


def build_cellprofiler_compatibility_report(
    *,
    parser: CPPipeParser | None = None,
) -> CellProfilerCompatibilityReport:
    """Build the current CellProfiler compatibility coverage matrix."""
    corpus_coverage = _module_corpus_coverage(parser or CPPipeParser())
    modules = tuple(
        _module_compatibility_coverage(module_name, corpus_coverage)
        for module_name in sorted(list_modules())
    )
    return CellProfilerCompatibilityReport(modules=modules)


def _module_compatibility_coverage(
    module_name: str,
    corpus_coverage: Mapping[str, ModuleCorpusCoverage],
) -> ModuleCompatibilityCoverage:
    contract = get_contract(module_name)
    if contract is None:
        raise KeyError(f"Absorbed module {module_name!r} has no contract metadata.")
    function_name = str(contract["function_name"])

    importable = _module_importable(module_name)
    processing_contract = None
    processing_contract_source = None
    processing_contract_error = None
    if importable:
        try:
            resolved_contract = resolve_processing_contract(
                module_name,
                function_name,
                str(contract["contract"]),
            )
        except ValueError as error:
            processing_contract_error = str(error)
        else:
            processing_contract = resolved_contract.contract
            processing_contract_source = resolved_contract.source

    return ModuleCompatibilityCoverage(
        module_name=module_name,
        function_name=function_name,
        importable=importable,
        processing_contract=processing_contract,
        processing_contract_source=processing_contract_source,
        processing_contract_error=processing_contract_error,
        artifact_contract_coverage=_artifact_contract_coverage(module_name),
        corpus_coverage=corpus_coverage.get(
            module_name,
            ModuleCorpusCoverage.NOT_IN_CORPUS,
        ),
    )


def _module_importable(module_name: str) -> bool:
    try:
        require_function(module_name)
    except Exception:
        return False
    return True


def _artifact_contract_coverage(module_name: str) -> ArtifactContractCoverage:
    if canonical_module_name(module_name) in ModuleContractBuilder.__registry__:
        return ArtifactContractCoverage.DECLARED_BUILDER
    return ArtifactContractCoverage.GENERIC_INFERENCE


def _module_corpus_coverage(
    parser: CPPipeParser,
) -> Mapping[str, ModuleCorpusCoverage]:
    coverage: dict[str, ModuleCorpusCoverage] = {}
    for case in in_tree_cppipe_corpus():
        case_coverage = _case_corpus_coverage(case.status)
        for module_name in _cppipe_module_names(parser, case.cppipe_path):
            coverage[module_name] = _merged_corpus_coverage(
                coverage.get(module_name),
                case_coverage,
            )
    return coverage


def _case_corpus_coverage(status: CPPipeCorpusStatus) -> ModuleCorpusCoverage:
    if status is CPPipeCorpusStatus.SUPPORTED:
        return ModuleCorpusCoverage.SUPPORTED_CORPUS
    return ModuleCorpusCoverage.KNOWN_INVALID_CORPUS


def _cppipe_module_names(
    parser: CPPipeParser,
    cppipe_path: Path,
) -> Sequence[str]:
    return tuple(
        canonical_module_name(module.name)
        for module in parser.parse(cppipe_path)
    )


def _merged_corpus_coverage(
    existing: ModuleCorpusCoverage | None,
    candidate: ModuleCorpusCoverage,
) -> ModuleCorpusCoverage:
    if existing is ModuleCorpusCoverage.SUPPORTED_CORPUS:
        return existing
    if candidate is ModuleCorpusCoverage.SUPPORTED_CORPUS:
        return candidate
    return existing or candidate
