"""CellProfiler compatibility coverage matrix."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from openhcs.processing.backends.cellprofiler.library import (
    canonical_module_name,
    get_contract,
    list_modules,
)
from openhcs.processing.backends.cellprofiler.module_classes import (
    ArtifactContractModule,
    CellProfilerModule,
)
from openhcs.processing.backends.cellprofiler import require_cellprofiler_function
from openhcs.processing.backends.lib_registry.unified_registry import (
    ProcessingContract,
)
from openhcs.core.alias_property import AliasProperty
from openhcs.interop.cellprofiler.parser import CPPipeParser, ModuleBlock, ModuleSetting
from openhcs.processing.backends.cellprofiler.module_classes import (
    ModuleSettingCoverageRecord,
    ModuleSettingRowRecord,
)
from openhcs.interop.cellprofiler.settings_binder import (
    normalize_cellprofiler_setting_name,
)
from openhcs.interop.cellprofiler.module_semantics import (
    CellProfilerModuleCategory,
    CellProfilerModuleDimensionality,
    CellProfilerModuleSemantics,
    cellprofiler_module_semantics_family,
    cellprofiler_module_semantics,
)
from openhcs.interop.cellprofiler.pipeline_generator import PipelineGenerator
from openhcs.interop.cellprofiler.runtime_pipeline import partition_cppipe_modules

from .cppipe_corpus import (
    CPPipeCorpusCase,
    CPPipeCorpusStatus,
    comparison_manifest_cppipe_corpus,
    comparison_manifests_cppipe_corpus,
    default_cppipe_corpus,
)
from .cppipe_module_roles import CPPipeModuleRole, cppipe_module_role
from openhcs.interop.cellprofiler.processing_contract_resolution import (
    ProcessingContractResolutionSource,
    resolve_processing_contract,
)


INFRASTRUCTURE_COVERAGE_VALUE = "infrastructure"


class ArtifactContractCoverage(str, Enum):
    """How artifact semantics are known for one CellProfiler module."""

    DECLARED_BUILDER = "declared_builder"
    GENERIC_INFERENCE = "generic_inference"


class ModuleCorpusCoverage(str, Enum):
    """Whether one module appears in accepted in-tree real pipelines."""

    SUPPORTED_CORPUS = "supported_corpus"
    KNOWN_INVALID_CORPUS = "known_invalid_corpus"
    NOT_IN_CORPUS = "not_in_corpus"


class CPPipeModuleAbsorptionCoverage(str, Enum):
    """How one real-corpus .cppipe module is handled by conversion."""

    ABSORBED_PROCESSING = "absorbed_processing"
    INFRASTRUCTURE = INFRASTRUCTURE_COVERAGE_VALUE
    MISSING_PROCESSING = "missing_processing"


class SourceModuleCoverage(str, Enum):
    """How one checked-in CellProfiler source module is covered."""

    ABSORBED = "absorbed"
    INFRASTRUCTURE = INFRASTRUCTURE_COVERAGE_VALUE
    MISSING = "missing"


class SemanticFamilyCoverageKind(str, Enum):
    """How an absorbed module is covered by direct or family-level evidence."""

    DIRECT_SUPPORTED = "direct_supported"
    SEMANTIC_FAMILY_SUPPORTED = "semantic_family_supported"
    NOT_SUPPORTED = "not_supported"


class CPPipeSettingCoverage(str, Enum):
    """How one concrete .cppipe setting row is covered by OpenHCS import."""

    BOUND = "bound", True
    ARTIFACT_CONTRACT = "artifact_contract", True
    TYPED_IGNORE = "typed_ignore", True
    CALLER_IGNORE = "caller_ignore", True
    INFRASTRUCTURE = INFRASTRUCTURE_COVERAGE_VALUE, True
    UNMAPPED = "unmapped", False
    MODULE_NOT_ABSORBED = "module_not_absorbed", False
    GENERATION_ERROR = "generation_error", False

    def __new__(cls, value: str, covered: bool) -> "CPPipeSettingCoverage":
        member = str.__new__(cls, value)
        member._value_ = value
        member._covered = covered
        return member

    is_covered = AliasProperty[bool]("_covered")


@dataclass(frozen=True, slots=True)
class _CorpusModuleObservations:
    """Observed module coverage from the registered .cppipe benchmark corpus."""

    coverage: Mapping[str, ModuleCorpusCoverage]
    case_names_by_module: Mapping[str, tuple[str, ...]]
    setting_coverage: tuple["CPPipeSettingCompatibilityCoverage", ...]
    cppipe_case_count: int
    supported_cppipe_case_count: int
    known_invalid_cppipe_case_count: int
    module_instance_count: int


@dataclass(frozen=True, slots=True)
class ModuleCompatibilityCoverage:
    """Compatibility coverage for one absorbed CellProfiler module."""

    module_name: str
    semantics: CellProfilerModuleSemantics | None
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
class CPPipeModuleCompatibilityCoverage:
    """Compatibility coverage for one module observed in accepted .cppipe corpus."""

    module_name: str
    semantics: CellProfilerModuleSemantics | None
    corpus_coverage: ModuleCorpusCoverage
    absorption_coverage: CPPipeModuleAbsorptionCoverage
    cppipe_case_names: tuple[str, ...] = ()

    @property
    def is_missing_processing_module(self) -> bool:
        return (
            self.absorption_coverage
            is CPPipeModuleAbsorptionCoverage.MISSING_PROCESSING
        )


@dataclass(frozen=True, slots=True)
class SourceModuleCompatibilityCoverage:
    """Compatibility coverage for one checked-in CellProfiler source module."""

    module_name: str
    semantics: CellProfilerModuleSemantics | None
    coverage: SourceModuleCoverage

    @property
    def is_missing(self) -> bool:
        return self.coverage is SourceModuleCoverage.MISSING


@dataclass(frozen=True, slots=True)
class SemanticFamilyCompatibilityCoverage:
    """Coverage evidence for one absorbed module's semantic family."""

    module_name: str
    family_name: str
    category: CellProfilerModuleCategory | None
    dimensionality: CellProfilerModuleDimensionality | None
    respects_masks: bool
    corpus_coverage: ModuleCorpusCoverage
    family_supported_modules: tuple[str, ...]
    family_absorbed_modules: tuple[str, ...]

    @property
    def family_coverage(self) -> SemanticFamilyCoverageKind:
        if self.corpus_coverage is ModuleCorpusCoverage.SUPPORTED_CORPUS:
            return SemanticFamilyCoverageKind.DIRECT_SUPPORTED
        if self.family_supported_modules:
            return SemanticFamilyCoverageKind.SEMANTIC_FAMILY_SUPPORTED
        return SemanticFamilyCoverageKind.NOT_SUPPORTED


@dataclass(frozen=True, slots=True)
class CPPipeSettingCompatibilityCoverage(ModuleSettingRowRecord):
    """Compatibility coverage for one concrete setting row in a benchmark .cppipe."""

    case_name: str
    canonical_module_name: str
    coverage: CPPipeSettingCoverage


@dataclass(frozen=True, slots=True)
class CPPipeSettingCoverageCollector:
    """Project generated-pipeline binding coverage onto concrete .cppipe settings."""

    generator: PipelineGenerator

    def for_case(
        self,
        case: CPPipeCorpusCase,
        modules: Sequence[ModuleBlock],
    ) -> tuple[CPPipeSettingCompatibilityCoverage, ...]:
        partition = partition_cppipe_modules(modules)
        absorbed_modules = frozenset(list_modules())
        absorbed_processing_modules = tuple(
            module
            for module in partition.processing_modules
            if canonical_module_name(module.name) in absorbed_modules
        )
        missing_processing_rows = tuple(
            self.for_module_setting(
                case,
                module,
                setting,
                CPPipeSettingCoverage.MODULE_NOT_ABSORBED,
            )
            for module in partition.processing_modules
            if canonical_module_name(module.name) not in absorbed_modules
            for setting in module.iter_settings()
        )
        infrastructure_rows = tuple(
            self.for_module_setting(
                case,
                module,
                setting,
                CPPipeSettingCoverage.INFRASTRUCTURE,
            )
            for module in partition.infrastructure_modules
            for setting in module.iter_settings()
        )
        try:
            generated = self.generator.generate_from_registry(
                pipeline_name=case.cppipe_path.stem,
                source_cppipe=case.cppipe_path,
                modules=list(absorbed_processing_modules),
                skipped_modules=list(partition.infrastructure_modules),
            )
        except Exception:
            return (
                *infrastructure_rows,
                *missing_processing_rows,
                *(
                    self.for_module_setting(
                        case,
                        module,
                        setting,
                        CPPipeSettingCoverage.GENERATION_ERROR,
                    )
                    for module in absorbed_processing_modules
                    for setting in module.iter_settings()
                ),
            )
        return (
            *infrastructure_rows,
            *missing_processing_rows,
            *(
                self.for_bound_setting(case, coverage)
                for coverage in generated.setting_coverage
            ),
        )

    def for_bound_setting(
        self,
        case: CPPipeCorpusCase,
        coverage: ModuleSettingCoverageRecord,
    ) -> CPPipeSettingCompatibilityCoverage:
        return CPPipeSettingCompatibilityCoverage(
            case_name=case.name,
            module_name=coverage.module_name,
            canonical_module_name=canonical_module_name(coverage.module_name),
            module_num=coverage.module_num,
            setting_name=coverage.setting_name,
            normalized_setting_name=coverage.normalized_setting_name,
            value=str(coverage.value),
            coverage=CPPipeSettingCoverage(coverage.status.value),
        )

    def for_module_setting(
        self,
        case: CPPipeCorpusCase,
        module: ModuleBlock,
        setting: ModuleSetting,
        coverage: CPPipeSettingCoverage,
    ) -> CPPipeSettingCompatibilityCoverage:
        return CPPipeSettingCompatibilityCoverage(
            case_name=case.name,
            module_name=module.name,
            canonical_module_name=canonical_module_name(module.name),
            module_num=module.module_num,
            setting_name=setting.name,
            normalized_setting_name=normalize_cellprofiler_setting_name(setting.name),
            value=setting.value,
            coverage=coverage,
        )


@dataclass(frozen=True, slots=True)
class CellProfilerBenchmarkCoverageSummary:
    """Benchmark corpus coverage for absorbed and observed CellProfiler modules."""

    cppipe_case_count: int
    supported_cppipe_case_count: int
    known_invalid_cppipe_case_count: int
    module_instance_count: int
    unique_cppipe_module_count: int
    supported_absorbed_processing_modules: tuple[str, ...]
    known_invalid_absorbed_processing_modules: tuple[str, ...]
    untested_absorbed_processing_modules: tuple[str, ...]
    infrastructure_cppipe_modules: tuple[str, ...]
    missing_processing_cppipe_modules: tuple[str, ...]

    @property
    def supported_absorbed_processing_module_count(self) -> int:
        return len(self.supported_absorbed_processing_modules)

    @property
    def known_invalid_absorbed_processing_module_count(self) -> int:
        return len(self.known_invalid_absorbed_processing_modules)

    @property
    def untested_absorbed_processing_module_count(self) -> int:
        return len(self.untested_absorbed_processing_modules)

    @property
    def infrastructure_cppipe_module_count(self) -> int:
        return len(self.infrastructure_cppipe_modules)

    @property
    def missing_processing_cppipe_module_count(self) -> int:
        return len(self.missing_processing_cppipe_modules)


@dataclass(frozen=True, slots=True)
class CellProfilerCompatibilityReport:
    """Typed compatibility matrix over absorbed modules and real pipelines."""

    modules: tuple[ModuleCompatibilityCoverage, ...]
    cppipe_modules: tuple[CPPipeModuleCompatibilityCoverage, ...]
    cppipe_settings: tuple[CPPipeSettingCompatibilityCoverage, ...]
    source_modules: tuple[SourceModuleCompatibilityCoverage, ...]
    semantic_families: tuple[SemanticFamilyCompatibilityCoverage, ...]
    benchmark_coverage: CellProfilerBenchmarkCoverageSummary

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

    @property
    def missing_cppipe_processing_modules(
        self,
    ) -> tuple[CPPipeModuleCompatibilityCoverage, ...]:
        return tuple(
            module
            for module in self.cppipe_modules
            if module.is_missing_processing_module
        )

    @property
    def missing_source_modules(
        self,
    ) -> tuple[SourceModuleCompatibilityCoverage, ...]:
        return tuple(
            module
            for module in self.source_modules
            if module.is_missing
        )


def build_cellprofiler_compatibility_report(
    *,
    parser: CPPipeParser | None = None,
    corpus_cases: Sequence[CPPipeCorpusCase] | None = None,
    source_modules_root: Path | None = None,
) -> CellProfilerCompatibilityReport:
    """Build the current CellProfiler compatibility coverage matrix."""
    absorbed_modules = frozenset(list_modules())
    corpus_observations = _corpus_module_observations(
        parser or CPPipeParser(),
        corpus_cases or default_cppipe_corpus(),
    )
    modules = tuple(
        _module_compatibility_coverage(
            module_name,
            corpus_observations.coverage,
        )
        for module_name in sorted(absorbed_modules)
    )
    cppipe_modules = tuple(
        _cppipe_module_compatibility_coverage(
            module_name,
            coverage,
            corpus_observations.case_names_by_module.get(module_name, ()),
            absorbed_modules,
        )
        for module_name, coverage in sorted(corpus_observations.coverage.items())
    )
    return CellProfilerCompatibilityReport(
        modules=modules,
        cppipe_modules=cppipe_modules,
        cppipe_settings=corpus_observations.setting_coverage,
        source_modules=tuple(
            _source_module_compatibility_coverage(
                module_name,
            )
            for module_name in _cellprofiler_source_module_names(
                source_modules_root,
            )
        ),
        semantic_families=_semantic_family_compatibility_coverage(modules),
        benchmark_coverage=_benchmark_coverage_summary(
            corpus_observations,
            modules,
            cppipe_modules,
        ),
    )


def build_cellprofiler_compatibility_report_for_manifest(
    manifest_path: Path,
    *,
    parser: CPPipeParser | None = None,
    source_modules_root: Path | None = None,
) -> CellProfilerCompatibilityReport:
    """Build compatibility coverage over a benchmark comparison manifest."""

    return build_cellprofiler_compatibility_report(
        parser=parser,
        corpus_cases=comparison_manifest_cppipe_corpus(manifest_path),
        source_modules_root=source_modules_root,
    )


def build_cellprofiler_compatibility_report_for_manifests(
    manifest_paths: Sequence[Path],
    *,
    parser: CPPipeParser | None = None,
    source_modules_root: Path | None = None,
) -> CellProfilerCompatibilityReport:
    """Build compatibility coverage over multiple benchmark comparison manifests."""

    return build_cellprofiler_compatibility_report(
        parser=parser,
        corpus_cases=comparison_manifests_cppipe_corpus(manifest_paths),
        source_modules_root=source_modules_root,
    )


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
        semantics=cellprofiler_module_semantics(module_name),
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


def _cppipe_module_compatibility_coverage(
    module_name: str,
    corpus_coverage: ModuleCorpusCoverage,
    cppipe_case_names: tuple[str, ...],
    absorbed_modules: frozenset[str],
) -> CPPipeModuleCompatibilityCoverage:
    return CPPipeModuleCompatibilityCoverage(
        module_name=module_name,
        semantics=cellprofiler_module_semantics(module_name),
        corpus_coverage=corpus_coverage,
        absorption_coverage=_cppipe_module_absorption_coverage(
            module_name,
            absorbed_modules,
        ),
        cppipe_case_names=cppipe_case_names,
    )


def _cppipe_module_absorption_coverage(
    module_name: str,
    absorbed_modules: frozenset[str],
) -> CPPipeModuleAbsorptionCoverage:
    if cppipe_module_role(module_name).role is CPPipeModuleRole.INFRASTRUCTURE:
        return CPPipeModuleAbsorptionCoverage.INFRASTRUCTURE
    if module_name in absorbed_modules:
        return CPPipeModuleAbsorptionCoverage.ABSORBED_PROCESSING
    return CPPipeModuleAbsorptionCoverage.MISSING_PROCESSING


def _source_module_compatibility_coverage(
    module_name: str,
) -> SourceModuleCompatibilityCoverage:
    if cppipe_module_role(module_name).role is CPPipeModuleRole.INFRASTRUCTURE:
        coverage = SourceModuleCoverage.INFRASTRUCTURE
    elif get_contract(module_name) is not None:
        coverage = SourceModuleCoverage.ABSORBED
    else:
        coverage = SourceModuleCoverage.MISSING
    return SourceModuleCompatibilityCoverage(
        module_name=module_name,
        semantics=cellprofiler_module_semantics(module_name),
        coverage=coverage,
    )


def _semantic_family_compatibility_coverage(
    modules: Sequence[ModuleCompatibilityCoverage],
) -> tuple[SemanticFamilyCompatibilityCoverage, ...]:
    modules_by_name = {module.module_name: module for module in modules}
    rows: list[SemanticFamilyCompatibilityCoverage] = []
    for module in modules:
        family = cellprofiler_module_semantics_family(module.module_name)
        if family is None:
            rows.append(
                SemanticFamilyCompatibilityCoverage(
                    module_name=module.module_name,
                    family_name="",
                    category=None,
                    dimensionality=None,
                    respects_masks=False,
                    corpus_coverage=module.corpus_coverage,
                    family_supported_modules=(),
                    family_absorbed_modules=(module.module_name,),
                )
            )
            continue
        family_absorbed_modules = tuple(
            family_module_name
            for family_module_name in family.module_names
            if family_module_name in modules_by_name
        )
        family_supported_modules = tuple(
            family_module_name
            for family_module_name in family_absorbed_modules
            if (
                modules_by_name[family_module_name].corpus_coverage
                is ModuleCorpusCoverage.SUPPORTED_CORPUS
            )
        )
        rows.append(
            SemanticFamilyCompatibilityCoverage(
                module_name=module.module_name,
                family_name=family.family_name,
                category=family.category,
                dimensionality=family.dimensionality,
                respects_masks=family.respects_masks,
                corpus_coverage=module.corpus_coverage,
                family_supported_modules=family_supported_modules,
                family_absorbed_modules=family_absorbed_modules,
            )
        )
    return tuple(rows)


def _cellprofiler_source_module_names(
    source_modules_root: Path | None,
) -> tuple[str, ...]:
    root = source_modules_root or (
        Path(__file__).resolve().parents[1] / "cellprofiler_source" / "modules"
    )
    if not root.exists():
        return ()
    return tuple(
        path.stem
        for path in sorted(root.glob("*.py"))
        if path.stem != "__init__" and not path.stem.startswith("_")
    )


def _module_importable(module_name: str) -> bool:
    try:
        require_cellprofiler_function(module_name)
    except Exception:
        return False
    return True


def _artifact_contract_coverage(module_name: str) -> ArtifactContractCoverage:
    module_type = CellProfilerModule.for_module(module_name)
    if module_type is not None and issubclass(module_type, ArtifactContractModule):
        return ArtifactContractCoverage.DECLARED_BUILDER
    return ArtifactContractCoverage.GENERIC_INFERENCE


def _corpus_module_observations(
    parser: CPPipeParser,
    corpus_cases: Sequence[CPPipeCorpusCase],
) -> _CorpusModuleObservations:
    coverage: dict[str, ModuleCorpusCoverage] = {}
    case_names_by_module: dict[str, list[str]] = {}
    setting_coverage: list[CPPipeSettingCompatibilityCoverage] = []
    supported_cppipe_case_count = 0
    known_invalid_cppipe_case_count = 0
    module_instance_count = 0
    setting_coverage_collector = CPPipeSettingCoverageCollector(PipelineGenerator())

    for case in corpus_cases:
        case_coverage = _case_corpus_coverage(case.status)
        if case_coverage is ModuleCorpusCoverage.SUPPORTED_CORPUS:
            supported_cppipe_case_count += 1
        else:
            known_invalid_cppipe_case_count += 1

        modules = tuple(parser.parse(case.cppipe_path))
        module_names = tuple(canonical_module_name(module.name) for module in modules)
        module_instance_count += len(module_names)
        for module_name in module_names:
            coverage[module_name] = _merged_corpus_coverage(
                coverage.get(module_name),
                case_coverage,
            )
            module_case_names = case_names_by_module.setdefault(module_name, [])
            if case.name not in module_case_names:
                module_case_names.append(case.name)
        if case.status is CPPipeCorpusStatus.SUPPORTED:
            setting_coverage.extend(
                setting_coverage_collector.for_case(case, modules)
            )

    return _CorpusModuleObservations(
        coverage=coverage,
        case_names_by_module={
            module_name: tuple(case_names)
            for module_name, case_names in case_names_by_module.items()
        },
        setting_coverage=tuple(setting_coverage),
        cppipe_case_count=len(corpus_cases),
        supported_cppipe_case_count=supported_cppipe_case_count,
        known_invalid_cppipe_case_count=known_invalid_cppipe_case_count,
        module_instance_count=module_instance_count,
    )


def _benchmark_coverage_summary(
    corpus_observations: _CorpusModuleObservations,
    modules: Sequence[ModuleCompatibilityCoverage],
    cppipe_modules: Sequence[CPPipeModuleCompatibilityCoverage],
) -> CellProfilerBenchmarkCoverageSummary:
    return CellProfilerBenchmarkCoverageSummary(
        cppipe_case_count=corpus_observations.cppipe_case_count,
        supported_cppipe_case_count=corpus_observations.supported_cppipe_case_count,
        known_invalid_cppipe_case_count=(
            corpus_observations.known_invalid_cppipe_case_count
        ),
        module_instance_count=corpus_observations.module_instance_count,
        unique_cppipe_module_count=len(corpus_observations.coverage),
        supported_absorbed_processing_modules=tuple(
            module.module_name
            for module in modules
            if module.corpus_coverage is ModuleCorpusCoverage.SUPPORTED_CORPUS
        ),
        known_invalid_absorbed_processing_modules=tuple(
            module.module_name
            for module in modules
            if module.corpus_coverage is ModuleCorpusCoverage.KNOWN_INVALID_CORPUS
        ),
        untested_absorbed_processing_modules=tuple(
            module.module_name
            for module in modules
            if module.corpus_coverage is ModuleCorpusCoverage.NOT_IN_CORPUS
        ),
        infrastructure_cppipe_modules=tuple(
            module.module_name
            for module in cppipe_modules
            if (
                module.absorption_coverage
                is CPPipeModuleAbsorptionCoverage.INFRASTRUCTURE
            )
        ),
        missing_processing_cppipe_modules=tuple(
            module.module_name
            for module in cppipe_modules
            if module.is_missing_processing_module
        ),
    )


def _case_corpus_coverage(status: CPPipeCorpusStatus) -> ModuleCorpusCoverage:
    if status is CPPipeCorpusStatus.SUPPORTED:
        return ModuleCorpusCoverage.SUPPORTED_CORPUS
    return ModuleCorpusCoverage.KNOWN_INVALID_CORPUS


def _merged_corpus_coverage(
    existing: ModuleCorpusCoverage | None,
    candidate: ModuleCorpusCoverage,
) -> ModuleCorpusCoverage:
    if existing is ModuleCorpusCoverage.SUPPORTED_CORPUS:
        return existing
    if candidate is ModuleCorpusCoverage.SUPPORTED_CORPUS:
        return candidate
    return existing or candidate
