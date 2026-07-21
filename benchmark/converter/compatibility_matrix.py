"""CellProfiler compatibility coverage derived from nominal declarations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from openhcs.core.callable_contract import (
    CallableContract,
    FunctionStepExecutionScope,
)
from openhcs.interop.cellprofiler.module_declarations import (
    CellProfilerModule,
)
from openhcs.interop.cellprofiler.module_settings import (
    ModuleSettingCoverageRecord,
    ModuleSettingCoverageStatus,
    ModuleSettingRowRecord,
)
from openhcs.interop.cellprofiler.parser import CPPipeParser, ModuleBlock
from openhcs.interop.cellprofiler.pipeline_import import import_cellprofiler_pipeline
from openhcs.interop.cellprofiler.settings_binder import SettingsBinder
from openhcs.processing.backends.lib_registry.unified_registry import (
    ProcessingContract,
)

from .cppipe_corpus import (
    CPPipeCorpusCase,
    CPPipeCorpusStatus,
    comparison_manifest_cppipe_corpus,
    comparison_manifests_cppipe_corpus,
    default_cppipe_corpus,
)


INFRASTRUCTURE_COVERAGE_VALUE = "infrastructure"


class ModuleCorpusCoverage(str, Enum):
    """Whether one module appears in accepted in-tree real pipelines."""

    SUPPORTED_CORPUS = "supported_corpus"
    KNOWN_INVALID_CORPUS = "known_invalid_corpus"
    NOT_IN_CORPUS = "not_in_corpus"


class CPPipeModuleAbsorptionCoverage(str, Enum):
    """How one real-corpus module resolves against nominal declarations."""

    ABSORBED_PROCESSING = "absorbed_processing"
    INFRASTRUCTURE = INFRASTRUCTURE_COVERAGE_VALUE
    MISSING_PROCESSING = "missing_processing"


class SourceModuleCoverage(str, Enum):
    """How one checked-in CellProfiler source module resolves."""

    ABSORBED = "absorbed"
    INFRASTRUCTURE = INFRASTRUCTURE_COVERAGE_VALUE
    MISSING = "missing"


@dataclass(frozen=True, slots=True)
class _CorpusModuleObservations:
    """Observed module coverage from the registered benchmark corpus."""

    coverage: Mapping[str, ModuleCorpusCoverage]
    case_names_by_module: Mapping[str, tuple[str, ...]]
    setting_coverage: tuple["CPPipeSettingCompatibilityCoverage", ...]
    cppipe_case_count: int
    supported_cppipe_case_count: int
    known_invalid_cppipe_case_count: int
    module_instance_count: int


@dataclass(frozen=True, slots=True)
class ModuleCompatibilityCoverage:
    """Compatibility coverage for one registered module declaration."""

    module_type: type[CellProfilerModule]
    callable_contract: CallableContract | None
    corpus_coverage: ModuleCorpusCoverage

    @property
    def module_name(self) -> str:
        return str(self.module_type.module_name)

    @property
    def function_names(self) -> tuple[str, ...]:
        return self.module_type.declared_function_names()

    @property
    def importable(self) -> bool:
        return self.callable_contract is not None

    @property
    def execution_scope(self) -> FunctionStepExecutionScope | None:
        if self.callable_contract is None:
            return None
        return self.callable_contract.execution_scope

    @property
    def processing_contract(self) -> ProcessingContract | None:
        if self.callable_contract is None:
            return None
        contract = self.callable_contract.processing_contract
        return contract if isinstance(contract, ProcessingContract) else None

    @property
    def respects_masks(self) -> bool:
        return self.module_type.respects_masks

    @property
    def emits_function_step(self) -> bool:
        return self.module_type.emits_function_step()

    @property
    def has_processing_contract(self) -> bool:
        return self.processing_contract is not None

    @property
    def is_infrastructure(self) -> bool:
        return not self.emits_function_step

    @property
    def requires_processing_contract(self) -> bool:
        return self.execution_scope is FunctionStepExecutionScope.AXIS


@dataclass(frozen=True, slots=True)
class CPPipeModuleCompatibilityCoverage:
    """Compatibility coverage for one module observed in the corpus."""

    module_name: str
    module_type: type[CellProfilerModule] | None
    corpus_coverage: ModuleCorpusCoverage
    cppipe_case_names: tuple[str, ...] = ()

    @property
    def absorption_coverage(self) -> CPPipeModuleAbsorptionCoverage:
        if self.module_type is None:
            return CPPipeModuleAbsorptionCoverage.MISSING_PROCESSING
        if self.module_type.emits_function_step():
            return CPPipeModuleAbsorptionCoverage.ABSORBED_PROCESSING
        return CPPipeModuleAbsorptionCoverage.INFRASTRUCTURE

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
    module_type: type[CellProfilerModule] | None

    @property
    def coverage(self) -> SourceModuleCoverage:
        if self.module_type is None:
            return SourceModuleCoverage.MISSING
        if self.module_type.emits_function_step():
            return SourceModuleCoverage.ABSORBED
        return SourceModuleCoverage.INFRASTRUCTURE

    @property
    def is_missing(self) -> bool:
        return self.coverage is SourceModuleCoverage.MISSING


@dataclass(frozen=True, slots=True)
class CPPipeSettingCompatibilityCoverage(ModuleSettingRowRecord):
    """Declaration-owned setting coverage for one corpus module row."""

    case_name: str
    canonical_module_name: str
    coverage: ModuleSettingCoverageStatus


@dataclass(frozen=True, slots=True)
class CellProfilerBenchmarkCoverageSummary:
    """Benchmark corpus coverage for registered and observed modules."""

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
    """Typed compatibility matrix over declarations and real pipelines."""

    modules: tuple[ModuleCompatibilityCoverage, ...]
    cppipe_modules: tuple[CPPipeModuleCompatibilityCoverage, ...]
    cppipe_settings: tuple[CPPipeSettingCompatibilityCoverage, ...]
    source_modules: tuple[SourceModuleCompatibilityCoverage, ...]
    benchmark_coverage: CellProfilerBenchmarkCoverageSummary

    @property
    def unresolved_processing_contracts(
        self,
    ) -> tuple[ModuleCompatibilityCoverage, ...]:
        return tuple(
            module
            for module in self.modules
            if module.requires_processing_contract
            and not module.has_processing_contract
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
    def missing_source_modules(self) -> tuple[SourceModuleCompatibilityCoverage, ...]:
        return tuple(module for module in self.source_modules if module.is_missing)


def build_cellprofiler_compatibility_report(
    *,
    parser: CPPipeParser | None = None,
    corpus_cases: Sequence[CPPipeCorpusCase] | None = None,
    source_modules_root: Path | None = None,
) -> CellProfilerCompatibilityReport:
    """Build compatibility coverage from the nominal module registry."""

    selected_cases = default_cppipe_corpus() if corpus_cases is None else corpus_cases
    corpus_observations = _corpus_module_observations(
        CPPipeParser() if parser is None else parser,
        selected_cases,
    )
    module_types = tuple(
        sorted(
            CellProfilerModule.__registry__.values(),
            key=lambda module_type: str(module_type.module_name),
        )
    )
    modules = tuple(
        _module_compatibility_coverage(
            module_type,
            corpus_observations.coverage,
        )
        for module_type in module_types
    )
    cppipe_modules = tuple(
        _cppipe_module_compatibility_coverage(
            module_name,
            coverage,
            corpus_observations.case_names_by_module.get(module_name, ()),
        )
        for module_name, coverage in sorted(corpus_observations.coverage.items())
    )
    return CellProfilerCompatibilityReport(
        modules=modules,
        cppipe_modules=cppipe_modules,
        cppipe_settings=corpus_observations.setting_coverage,
        source_modules=tuple(
            _source_module_compatibility_coverage(module_name)
            for module_name in _cellprofiler_source_module_names(source_modules_root)
        ),
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
    """Build compatibility coverage over one comparison manifest."""

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
    """Build compatibility coverage over multiple comparison manifests."""

    return build_cellprofiler_compatibility_report(
        parser=parser,
        corpus_cases=comparison_manifests_cppipe_corpus(manifest_paths),
        source_modules_root=source_modules_root,
    )


def _module_compatibility_coverage(
    module_type: type[CellProfilerModule],
    corpus_coverage: Mapping[str, ModuleCorpusCoverage],
) -> ModuleCompatibilityCoverage:
    function_names = module_type.declared_function_names()
    callable_contract = (
        CallableContract.from_callable(module_type.require_callable(function_names[0]))
        if function_names
        else None
    )
    return ModuleCompatibilityCoverage(
        module_type=module_type,
        callable_contract=callable_contract,
        corpus_coverage=corpus_coverage.get(
            str(module_type.module_name),
            ModuleCorpusCoverage.NOT_IN_CORPUS,
        ),
    )


def _cppipe_module_compatibility_coverage(
    module_name: str,
    corpus_coverage: ModuleCorpusCoverage,
    cppipe_case_names: tuple[str, ...],
) -> CPPipeModuleCompatibilityCoverage:
    module_type = CellProfilerModule.for_module(module_name)
    return CPPipeModuleCompatibilityCoverage(
        module_name=module_name,
        module_type=module_type,
        corpus_coverage=corpus_coverage,
        cppipe_case_names=cppipe_case_names,
    )


def _source_module_compatibility_coverage(
    module_name: str,
) -> SourceModuleCompatibilityCoverage:
    module_type = CellProfilerModule.for_module(module_name)
    return SourceModuleCompatibilityCoverage(
        module_name=module_name,
        module_type=module_type,
    )


def _cellprofiler_source_module_names(
    source_modules_root: Path | None,
) -> tuple[str, ...]:
    root = (
        source_modules_root
        if source_modules_root is not None
        else Path(__file__).resolve().parents[1] / "cellprofiler_source" / "modules"
    )
    if not root.exists():
        return ()
    return tuple(
        path.stem
        for path in sorted(root.glob("*.py"))
        if path.stem != "__init__" and not path.stem.startswith("_")
    )


def _setting_coverage_for_case(
    case: CPPipeCorpusCase,
    modules: Sequence[ModuleBlock],
) -> tuple[CPPipeSettingCompatibilityCoverage, ...]:
    binder = SettingsBinder(
        source_root=(
            case.cppipe_path.parent if case.source_root is None else case.source_root
        )
    )
    rows: list[CPPipeSettingCompatibilityCoverage] = []
    for module in modules:
        if not module.enabled:
            continue
        module_type = CellProfilerModule.require_module(module.name)
        if not module_type.emits_function_step():
            continue
        bound = module_type.bind_settings(module, binder=binder)
        rows.extend(
            _setting_coverage_record(case, module_type, coverage)
            for coverage in bound.setting_coverage
        )
    return tuple(rows)


def _setting_coverage_record(
    case: CPPipeCorpusCase,
    module_type: type[CellProfilerModule],
    coverage: ModuleSettingCoverageRecord,
) -> CPPipeSettingCompatibilityCoverage:
    return CPPipeSettingCompatibilityCoverage(
        case_name=case.name,
        module_name=coverage.module_name,
        canonical_module_name=str(module_type.module_name),
        module_num=coverage.module_num,
        setting_name=coverage.setting_name,
        normalized_setting_name=coverage.normalized_setting_name,
        value=str(coverage.value),
        coverage=coverage.status,
    )


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
    for case in corpus_cases:
        case_coverage = _case_corpus_coverage(case.status)
        if case_coverage is ModuleCorpusCoverage.SUPPORTED_CORPUS:
            supported_cppipe_case_count += 1
        else:
            known_invalid_cppipe_case_count += 1
        if case.status is CPPipeCorpusStatus.SUPPORTED:
            parsed_modules = tuple(parser.parse(case.cppipe_path))
        else:
            parsed_modules = tuple(parser.parse(case.cppipe_path))
        module_names: list[str] = []
        for module in parsed_modules:
            module_type = CellProfilerModule.for_module(module.name)
            module_names.append(
                module.name if module_type is None else str(module_type.module_name)
            )
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
            import_cellprofiler_pipeline(
                case.cppipe_path,
                source_root=case.source_root,
            )
            setting_coverage.extend(_setting_coverage_for_case(case, parsed_modules))
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
            and module.emits_function_step
        ),
        known_invalid_absorbed_processing_modules=tuple(
            module.module_name
            for module in modules
            if module.corpus_coverage is ModuleCorpusCoverage.KNOWN_INVALID_CORPUS
            and module.emits_function_step
        ),
        untested_absorbed_processing_modules=tuple(
            module.module_name
            for module in modules
            if module.corpus_coverage is ModuleCorpusCoverage.NOT_IN_CORPUS
            and module.emits_function_step
        ),
        infrastructure_cppipe_modules=tuple(
            module.module_name
            for module in cppipe_modules
            if module.absorption_coverage
            is CPPipeModuleAbsorptionCoverage.INFRASTRUCTURE
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
    return candidate if existing is None else existing
