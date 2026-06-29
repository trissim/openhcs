"""Headless function registry projection for OpenHCS agents."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from python_introspect import parameter_exclusions

from openhcs.agent.dto.common import SCHEMA_VERSION
from openhcs.agent.dto.functions import (
    CellProfilerModuleDeclarationSummary,
    CustomFunctionRegistrationRequest,
    CustomFunctionRegistrationResult,
    DEFAULT_FUNCTION_DETAIL_DOC_CHARS,
    FunctionArtifactSpec,
    FunctionCatalogEntry,
    FunctionDetail,
    FunctionParameterSpec,
    FunctionParameterSource,
    FunctionRuntimeContractSummary,
    catalog_page,
)
from openhcs.agent.exceptions import AgentFacingErrorMixin
from openhcs.core.artifacts import ArtifactKind, ArtifactSpec
from openhcs.core.callable_contract import CallableContract
from openhcs.core.module_artifact_contract import ModuleArtifactContract
from openhcs.agent.services.stdio import AgentStdoutRedirect
from openhcs.processing.backends.lib_registry.registry_service import RegistryService
from openhcs.processing.backends.lib_registry.unified_registry import FunctionMetadata
import openhcs.processing.custom_functions.manager as custom_function_manager


MAX_FUNCTION_DETAIL_DOC_CHARS = 50_000


class FunctionCatalogError(AgentFacingErrorMixin, ValueError):
    """Base class for function catalog failures intended for agents."""


class UnknownFunctionIdError(FunctionCatalogError):
    """Raised when an agent references a function id outside the registry."""

    agent_error_code = "unknown_function_id"
    agent_error_hint = (
        "Call openhcs_search_functions with biology or method terms, then pass "
        "one returned function_id exactly."
    )

    def __init__(self, function_id: str) -> None:
        self.function_id = function_id
        super().__init__(f"Unknown OpenHCS function_id: {function_id}")


class CatalogSearchRank(Enum):
    ALL_PASS = 0
    EXACT_NAME = 10
    PREFIX_NAME = 20
    NAME_CONTAINS = 30
    ID_CONTAINS = 40
    IMPORT_CONTAINS = 50
    TAG_CONTAINS = 60
    SUMMARY_CONTAINS = 70
    DOC_CONTAINS = 80
    NO_MATCH = 1_000

    @property
    def matched(self) -> bool:
        return self is not CatalogSearchRank.NO_MATCH


class SignatureView(Enum):
    FULL = 10_000
    COMPACT = 4

    @property
    def parameter_limit(self) -> int:
        return self.value

    @property
    def compact(self) -> bool:
        return self is SignatureView.COMPACT


class AgentFunctionSearchPolicy:
    """Search-only text policy for agent function catalog ranking."""

    stop_words = frozenset(
        {
            "a",
            "an",
            "and",
            "are",
            "can",
            "could",
            "do",
            "does",
            "for",
            "have",
            "how",
            "i",
            "in",
            "is",
            "maybe",
            "my",
            "of",
            "or",
            "please",
            "the",
            "to",
            "use",
            "using",
            "want",
            "what",
            "with",
        }
    )
    domain_token_aliases: ClassVar[dict[str, tuple[str, ...]]] = {
        "nuclei": ("nucleus", "primary"),
        "nucleus": ("nuclei", "primary"),
        "cell": ("cells", "secondary"),
        "cells": ("cell", "secondary"),
        "segment": ("segmentation", "identify"),
        "segmentation": ("segment", "identify"),
    }

    @classmethod
    def accepts_token(cls, token: str) -> bool:
        return token not in cls.stop_words

    @classmethod
    def token_variants(cls, token: str) -> tuple[str, ...]:
        variants = [token]
        if len(token) > 4 and token.endswith("ies"):
            variants.append(f"{token[:-3]}y")
        if len(token) > 3 and token.endswith("s"):
            variants.append(token[:-1])
        variants.extend(cls.domain_token_aliases.get(token, ()))
        return tuple(dict.fromkeys(variants))


@dataclass(frozen=True, slots=True)
class CatalogFilterText:
    text: str = ""
    tokens: tuple[str, ...] = ()

    @classmethod
    def from_request(cls, value: str | None) -> "CatalogFilterText":
        if value is None:
            return cls.all_pass()
        return cls.from_text(value)

    @classmethod
    def all_pass(cls) -> "CatalogFilterText":
        return cls()

    @classmethod
    def from_text(cls, value: str) -> "CatalogFilterText":
        text = _normalized_search_text(value)
        tokens = []
        seen = set()
        for token in text.split():
            if not AgentFunctionSearchPolicy.accepts_token(token) or token in seen:
                continue
            tokens.append(token)
            seen.add(token)
        return cls(text, tuple(tokens))

    def accepts_library(self, library: str) -> bool:
        return not self.text or library.casefold() == self.text

    def search_rank(
        self,
        entry: FunctionCatalogEntry,
        metadata: FunctionMetadata,
    ) -> CatalogSearchRank:
        return self.search_match(entry, metadata).rank

    def search_match(
        self,
        entry: FunctionCatalogEntry,
        metadata: FunctionMetadata,
    ) -> "CatalogTextMatch":
        if not self.text:
            return CatalogTextMatch(CatalogSearchRank.ALL_PASS, 0)
        name_text = _normalized_search_text(entry.name)
        function_id_text = _normalized_search_text(entry.function_id)
        import_path_text = _normalized_search_text(entry.import_path)
        tag_text = _normalized_search_text(" ".join(entry.backend_tags))
        if name_text == self.text:
            return CatalogTextMatch(CatalogSearchRank.EXACT_NAME, 1_000)
        if name_text.startswith(self.text):
            return CatalogTextMatch(CatalogSearchRank.PREFIX_NAME, 900)
        for rank, value in (
            (CatalogSearchRank.NAME_CONTAINS, name_text),
            (CatalogSearchRank.ID_CONTAINS, function_id_text),
            (CatalogSearchRank.IMPORT_CONTAINS, import_path_text),
            (CatalogSearchRank.TAG_CONTAINS, tag_text),
            (CatalogSearchRank.SUMMARY_CONTAINS, entry.summary or ""),
            (CatalogSearchRank.DOC_CONTAINS, metadata.doc or ""),
        ):
            score = self._text_score(value)
            if score:
                return CatalogTextMatch(rank, score)
        return CatalogTextMatch(CatalogSearchRank.NO_MATCH, 0)

    def _matches_text(self, value: str) -> bool:
        return self._text_score(value) > 0

    def _text_score(self, value: str) -> int:
        text = _normalized_search_text(value)
        if self.text in text:
            return 1_000 + len(self.tokens)
        matched_count = sum(
            1
            for token in self.tokens
            if self._token_matches_text(token, text)
        )
        if not matched_count:
            return 0
        score = matched_count * 10
        if matched_count == len(self.tokens):
            score += 250
        return score

    @classmethod
    def _token_matches_text(cls, token: str, text: str) -> bool:
        text_tokens = tuple(text.split())
        for variant in cls._token_variants(token):
            if variant in text_tokens:
                return True
            if len(variant) >= 5 and any(variant in text_token for text_token in text_tokens):
                return True
        return False

    @staticmethod
    def _token_variants(token: str) -> tuple[str, ...]:
        return AgentFunctionSearchPolicy.token_variants(token)


@dataclass(frozen=True, slots=True)
class CatalogTextMatch:
    rank: CatalogSearchRank
    score: int

    @property
    def matched(self) -> bool:
        return self.rank.matched


class SummaryView(Enum):
    FULL = 10_000
    COMPACT = 180

    @property
    def character_limit(self) -> int:
        return self.value

    @property
    def compact(self) -> bool:
        return self is SummaryView.COMPACT


@dataclass(frozen=True, slots=True)
class CatalogSearchCandidate:
    rank: CatalogSearchRank
    score: int
    entry: FunctionCatalogEntry

    @property
    def sort_key(self) -> tuple[int, int, str, str]:
        return (
            self.rank.value,
            -self.score,
            self.entry.name.casefold(),
            self.entry.function_id,
        )


class ParameterDocumentationPolicy:
    """Nominal policy for exposing callable parameters to agents."""

    variadic_kinds = frozenset(
        {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }
    )

    def should_document(
        self,
        name: str,
        parameter: inspect.Parameter,
        *,
        skip_keyword_bag: bool = False,
        hidden_names: frozenset[str] = frozenset(),
    ) -> bool:
        if name in hidden_names:
            return False
        if skip_keyword_bag and name == "kwargs":
            return False
        return parameter.kind not in self.variadic_kinds

    def display_signature(
        self,
        func: Callable,
        display_name: str,
        view: SignatureView,
        contract: CallableContract | None = None,
    ) -> str:
        sig = inspect.signature(func)
        supplied_by = self.supplied_by(func, contract)
        hidden_names = parameter_exclusions(func)
        visible_parameters = tuple(
            parameter.replace(annotation=inspect.Parameter.empty)
            for name, parameter in sig.parameters.items()
            if self.should_document(
                name,
                parameter,
                skip_keyword_bag=True,
                hidden_names=hidden_names,
            )
            and supplied_by.get(name) is FunctionParameterSource.AGENT
        )
        if view.compact and len(visible_parameters) > view.parameter_limit:
            parameter_names = ", ".join(
                parameter.name
                for parameter in visible_parameters[:view.parameter_limit]
            )
            return f"{display_name}({parameter_names}, ...)"
        visible_signature = sig.replace(parameters=visible_parameters)
        return f"{display_name}{visible_signature}"

    def parameter_specs(
        self,
        func: Callable,
        contract: CallableContract | None = None,
    ) -> tuple[FunctionParameterSpec, ...]:
        sig = inspect.signature(func)
        supplied_by = self.supplied_by(func, contract)
        hidden_names = parameter_exclusions(func)
        specs = []
        for name, parameter in sig.parameters.items():
            if not self.should_document(name, parameter, hidden_names=hidden_names):
                continue
            supplier = supplied_by[name]
            specs.append(
                FunctionParameterSpec(
                    name=name,
                    annotation=_format_annotation(parameter.annotation),
                    default_repr=_format_default(parameter.default),
                    required=(
                        supplier is FunctionParameterSource.AGENT
                        and parameter.default is inspect.Parameter.empty
                    ),
                    supplied_by=supplier.value,
                    description=self.parameter_description(name, supplier),
                )
            )
        return tuple(specs)

    def agent_parameter_names(
        self,
        func: Callable,
        contract: CallableContract | None = None,
    ) -> tuple[str, ...]:
        sig = inspect.signature(func)
        supplied_by = self.supplied_by(func, contract)
        hidden_names = parameter_exclusions(func)
        return tuple(
            name
            for name, parameter in sig.parameters.items()
            if self.should_document(name, parameter, hidden_names=hidden_names)
            and supplied_by[name] is FunctionParameterSource.AGENT
        )

    def supplied_by(
        self,
        func: Callable,
        contract: CallableContract | None = None,
    ) -> dict[str, FunctionParameterSource]:
        sig = inspect.signature(func)
        callable_contract = contract or CallableContract.from_callable(func)
        hidden_names = parameter_exclusions(func)
        supplied_by = {
            name: FunctionParameterSource.AGENT
            for name, parameter in sig.parameters.items()
            if self.should_document(name, parameter, hidden_names=hidden_names)
        }
        primary_input_name = callable_contract.primary_input_parameter_name
        if primary_input_name in supplied_by:
            supplied_by[primary_input_name] = FunctionParameterSource.PRIMARY_INPUT
        for name in callable_contract.artifact_input_names:
            if name in supplied_by:
                supplied_by[name] = FunctionParameterSource.ARTIFACT_INPUT
        for name in callable_contract.runtime_bound_parameters:
            if name in supplied_by:
                supplied_by[name] = FunctionParameterSource.RUNTIME_PARAMETER
        runtime_adapter = callable_contract.runtime_adapter
        if runtime_adapter is not None and runtime_adapter.parameter_name in supplied_by:
            supplied_by[runtime_adapter.parameter_name] = (
                FunctionParameterSource.RUNTIME_ADAPTER
            )
        return supplied_by

    def parameter_description(
        self,
        name: str,
        supplier: FunctionParameterSource,
    ) -> str | None:
        if supplier is FunctionParameterSource.PRIMARY_INPUT:
            return (
                "Supplied by OpenHCS from the FunctionStep input image payload; "
                "do not pass this as a function kwarg."
            )
        if supplier is FunctionParameterSource.ARTIFACT_INPUT:
            return (
                "Supplied by OpenHCS from a declared artifact input during "
                "pipeline execution; do not pass this as a function kwarg."
            )
        if supplier is FunctionParameterSource.RUNTIME_PARAMETER:
            return (
                "Supplied by OpenHCS runtime execution infrastructure; do not "
                "pass this as a function kwarg."
            )
        if supplier is FunctionParameterSource.RUNTIME_ADAPTER:
            return (
                "Supplied by OpenHCS as a runtime adapter object; do not pass "
                "this as a function kwarg."
            )
        return None


PARAMETER_DOCUMENTATION_POLICY = ParameterDocumentationPolicy()


class FunctionCatalogService:
    """Expose registered OpenHCS processing callables through stable IDs."""

    def register_custom_function(
        self,
        request: CustomFunctionRegistrationRequest,
    ) -> CustomFunctionRegistrationResult:
        manager = custom_function_manager.CustomFunctionManager()
        registered_functions = manager.register_from_code(
            request.source_code,
            persist=request.persist,
        )
        function_ids = self.function_ids_for_callables(tuple(registered_functions))
        entries = tuple(
            self.get(
                function_id,
                max_doc_chars=0,
                compact_signature=request.compact_signature,
            ).entry
            for function_id in function_ids
        )
        source_file_paths = (
            tuple(
                str(manager.source_path_for_function(registered_function))
                for registered_function in registered_functions
            )
            if request.persist
            else ()
        )
        return CustomFunctionRegistrationResult(
            schema_version=SCHEMA_VERSION,
            registered_count=len(entries),
            persisted=request.persist,
            storage_dir=str(manager.storage_dir),
            source_file_paths=source_file_paths,
            functions=entries,
            next_steps=tuple(
                f"Call openhcs_describe_function(function_id={function_id!r}), then use openhcs_add_function_step or draft-pipeline-step."
                for function_id in function_ids
            ),
        )

    def search(
        self,
        *,
        query: str | None = None,
        library: str | None = None,
        limit: int = 50,
        compact_signatures: bool = False,
    ):
        if limit < 1:
            raise ValueError("limit must be at least 1")

        query_filter = CatalogFilterText.from_request(query)
        library_filter = CatalogFilterText.from_request(library)
        signature_view = SignatureView.COMPACT if compact_signatures else SignatureView.FULL
        summary_view = SummaryView.COMPACT if compact_signatures else SummaryView.FULL
        candidates = []

        for function_id, metadata in sorted(self._all_metadata().items()):
            entry = self._entry(function_id, metadata, signature_view, summary_view)
            if not library_filter.accepts_library(entry.library):
                continue
            match = query_filter.search_match(entry, metadata)
            if not match.matched:
                continue
            candidates.append(CatalogSearchCandidate(match.rank, match.score, entry))

        entries = tuple(
            candidate.entry
            for candidate in sorted(candidates, key=lambda candidate: candidate.sort_key)
        )

        return catalog_page(
            items=tuple(entries[:limit]),
            total=len(entries),
            limit=limit,
            query=query,
            library=library,
        )

    def get(
        self,
        function_id: str,
        *,
        max_doc_chars: int | None = DEFAULT_FUNCTION_DETAIL_DOC_CHARS,
        compact_signature: bool = True,
    ) -> FunctionDetail:
        metadata = self._metadata(function_id)
        func = metadata.func
        contract = CallableContract.from_callable(func)
        doc = _detail_doc(func, metadata.doc)
        bounded_doc, doc_truncated, effective_max_doc_chars = _bounded_detail_doc(
            doc,
            max_doc_chars=max_doc_chars,
        )
        return FunctionDetail(
            schema_version=SCHEMA_VERSION,
            entry=self._entry(
                function_id,
                metadata,
                signature_view=(
                    SignatureView.COMPACT
                    if compact_signature
                    else SignatureView.FULL
                ),
                summary_view=SummaryView.COMPACT,
                contract=contract,
            ),
            parameters=PARAMETER_DOCUMENTATION_POLICY.parameter_specs(func, contract),
            doc=bounded_doc,
            runtime_contract=_runtime_contract_summary(func, metadata, contract),
            doc_truncated=doc_truncated,
            doc_chars=len(doc or ""),
            max_doc_chars=effective_max_doc_chars,
        )

    def get_by_import_path(
        self,
        import_path: str,
        *,
        max_doc_chars: int | None = DEFAULT_FUNCTION_DETAIL_DOC_CHARS,
        compact_signature: bool = True,
    ) -> FunctionDetail | None:
        requested_import_path = import_path.strip()
        if not requested_import_path:
            return None
        for function_id, metadata in sorted(self._all_metadata().items()):
            entry = self._entry(
                function_id,
                metadata,
                signature_view=(
                    SignatureView.COMPACT
                    if compact_signature
                    else SignatureView.FULL
                ),
                summary_view=SummaryView.COMPACT,
            )
            if requested_import_path in _import_path_candidates(entry, metadata):
                return self.get(
                    function_id,
                    max_doc_chars=max_doc_chars,
                    compact_signature=compact_signature,
                )
        return None

    def resolve(self, function_id: str) -> Callable:
        return self._metadata(function_id).func

    def function_ids_for_callables(
        self,
        functions: tuple[Callable, ...],
    ) -> tuple[str, ...]:
        metadata_by_id = self._all_metadata()
        return tuple(
            self._function_id_for_callable(func, metadata_by_id)
            for func in functions
        )

    @classmethod
    def _function_id_for_callable(
        cls,
        func: Callable,
        metadata_by_id: dict[str, FunctionMetadata],
    ) -> str:
        identity_matches = tuple(
            function_id
            for function_id, metadata in metadata_by_id.items()
            if metadata.func is func
        )
        if len(identity_matches) == 1:
            return identity_matches[0]
        semantic_matches = tuple(
            function_id
            for function_id, metadata in metadata_by_id.items()
            if cls._metadata_matches_callable(metadata, func)
        )
        if len(semantic_matches) == 1:
            return semantic_matches[0]
        if not semantic_matches:
            raise UnknownFunctionIdError(func.__name__)
        raise ValueError(
            f"Registered callable {func.__module__}.{func.__name__} matched "
            f"multiple function ids: {semantic_matches!r}"
        )

    @staticmethod
    def _metadata_matches_callable(
        metadata: FunctionMetadata,
        func: Callable,
    ) -> bool:
        if metadata.module != func.__module__:
            return False
        return metadata.name == func.__name__ or metadata.original_name == func.__name__

    def _all_metadata(self) -> dict[str, FunctionMetadata]:
        with AgentStdoutRedirect.to_stderr():
            from openhcs.processing.func_registry import initialize_registry

            initialize_registry()
            return RegistryService.get_all_functions_with_metadata()

    def _metadata(self, function_id: str) -> FunctionMetadata:
        try:
            return self._all_metadata()[function_id]
        except KeyError as exc:
            raise UnknownFunctionIdError(function_id) from exc

    def _entry(
        self,
        function_id: str,
        metadata: FunctionMetadata,
        signature_view: SignatureView = SignatureView.FULL,
        summary_view: SummaryView = SummaryView.FULL,
        contract: CallableContract | None = None,
    ) -> FunctionCatalogEntry:
        name = _metadata_display_name(function_id, metadata)
        module = _metadata_module(metadata)
        library = metadata.get_registry_name()
        callable_contract = contract or CallableContract.from_callable(metadata.func)
        signature = PARAMETER_DOCUMENTATION_POLICY.display_signature(
            metadata.func,
            name,
            signature_view,
            callable_contract,
        )
        summary = _summary(metadata.func, metadata.doc, summary_view)
        tags = metadata.tags
        if tags is None:
            backend_tags = ()
        else:
            backend_tags = tuple(str(tag) for tag in tags)
        return FunctionCatalogEntry(
            function_id=function_id,
            name=name,
            module=module,
            library=library,
            import_path=_import_path(module, name),
            signature=signature,
            summary=summary,
            backend_tags=backend_tags,
        )

def _format_annotation(annotation) -> str | None:
    if annotation is inspect.Parameter.empty:
        return None
    if isinstance(annotation, type):
        return annotation.__name__
    return str(annotation)


def _format_default(default) -> str | None:
    if default is inspect.Parameter.empty:
        return None
    if isinstance(default, Enum):
        return f"{type(default).__name__}.{default.name}"
    return repr(default)


def _summary(
    func: Callable,
    metadata_doc: str | None,
    view: SummaryView,
) -> str | None:
    doc = _detail_doc(func, metadata_doc)
    if doc is None:
        return None
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped:
            return _bounded_summary(stripped, view)
    return None


def _bounded_summary(text: str, view: SummaryView) -> str:
    if not view.compact:
        return text
    if len(text) <= view.character_limit:
        return text
    return f"{text[: view.character_limit - 3].rstrip()}..."


def _import_path(module: str, name: str) -> str:
    if not module:
        return name
    return f"{module}.{name}"


def _import_path_candidates(
    entry: FunctionCatalogEntry,
    metadata: FunctionMetadata,
) -> frozenset[str]:
    func = metadata.func
    return frozenset(
        candidate
        for candidate in (
            entry.import_path,
            _import_path(func.__module__, func.__name__),
            _import_path(func.__module__, func.__qualname__),
            _import_path(entry.module, func.__name__),
            _import_path(entry.module, func.__qualname__),
        )
        if candidate
    )


def _normalized_search_text(value: str) -> str:
    text = value.casefold().strip()
    for separator in ("_", "-", ".", ":"):
        text = text.replace(separator, " ")
    return " ".join(text.split())


def _metadata_display_name(function_id: str, metadata: FunctionMetadata) -> str:
    if metadata.display_name:
        return metadata.display_name
    return function_id.rsplit(":", 1)[-1]


def _metadata_module(metadata: FunctionMetadata) -> str:
    if metadata.module:
        return metadata.module
    return metadata.func.__module__


def _runtime_contract_summary(
    func: Callable,
    metadata: FunctionMetadata,
    contract: CallableContract | None = None,
) -> FunctionRuntimeContractSummary:
    contract = contract or CallableContract.from_callable(func)
    module_contract = contract.module_artifact_contract
    cellprofiler_module = _cellprofiler_module_summary(func, metadata, module_contract)
    callable_kind = "cellprofiler_module" if cellprofiler_module is not None else "regular"
    required_variable_components = tuple(
        dict.fromkeys(
            (
                *(component.name for component in contract.required_variable_components),
                *(
                    ()
                    if cellprofiler_module is None
                    else cellprofiler_module.required_variable_components
                ),
            )
        )
    )
    return FunctionRuntimeContractSummary(
        callable_kind=callable_kind,
        processing_contract=_enum_member_name(contract.processing_contract),
        declared_processing_contract=contract.declared_processing_contract,
        runtime_bound_parameters=contract.runtime_bound_parameters,
        required_variable_components=required_variable_components,
        artifact_inputs=_artifact_specs_from_contract_inputs(contract, module_contract),
        runtime_artifact_inputs=(
            ()
            if module_contract is None
            else _artifact_specs(module_contract.runtime_artifact_inputs)
        ),
        artifact_outputs=_artifact_specs_from_contract_outputs(contract, module_contract),
        declared_artifact_outputs=(
            ()
            if module_contract is None
            else _artifact_specs(module_contract.declared_outputs)
        ),
        cellprofiler_module=cellprofiler_module,
        source_binding_rule=_source_binding_rule(cellprofiler_module, module_contract),
        materialization_rule=_materialization_rule(cellprofiler_module, module_contract),
        measurement_rule=_measurement_rule(cellprofiler_module, module_contract),
        pattern_compatibility_rule=_pattern_compatibility_rule(cellprofiler_module),
    )


def _artifact_specs_from_contract_inputs(
    contract: CallableContract,
    module_contract: ModuleArtifactContract | None,
) -> tuple[FunctionArtifactSpec, ...]:
    if module_contract is not None:
        return _artifact_specs(module_contract.inputs)
    return _artifact_specs(spec for _, spec in contract.artifact_inputs)


def _artifact_specs_from_contract_outputs(
    contract: CallableContract,
    module_contract: ModuleArtifactContract | None,
) -> tuple[FunctionArtifactSpec, ...]:
    if module_contract is not None:
        return _artifact_specs(module_contract.outputs)
    return _artifact_specs(spec for _, spec in contract.artifact_outputs)


def _artifact_specs(specs) -> tuple[FunctionArtifactSpec, ...]:
    return tuple(_artifact_spec(spec) for spec in specs)


def _artifact_spec(spec: ArtifactSpec) -> FunctionArtifactSpec:
    return FunctionArtifactSpec(
        name=spec.name,
        kind=spec.kind.value,
        required=spec.required,
        sidecar_role=None if spec.sidecar_role is None else spec.sidecar_role.value,
        materialization_uses_source_identity_filename=(
            spec.materialization_uses_source_identity_filename()
        ),
    )


def _cellprofiler_module_summary(
    func: Callable,
    metadata: FunctionMetadata,
    module_contract: ModuleArtifactContract | None,
) -> CellProfilerModuleDeclarationSummary | None:
    module_type = _cellprofiler_module_type(func, metadata, module_contract)
    if module_type is None:
        return None
    declared_input_settings = tuple(
        _declared_setting_summary(setting, kind)
        for setting, kind in module_type.declared_artifact_input_settings()
    )
    declared_output_settings = tuple(
        _declared_setting_summary(setting, kind)
        for setting, kind in module_type.declared_artifact_output_settings()
    )
    return CellProfilerModuleDeclarationSummary(
        module_name=str(module_type.module_name),
        declaration_class=f"{module_type.__module__}.{module_type.__qualname__}",
        category=str(module_type.category),
        contract=str(module_type.contract),
        validated=bool(module_type.validated),
        function_names=module_type.declared_function_names(),
        aliases=tuple(module_type.aliases),
        declared_artifact_input_settings=declared_input_settings,
        declared_artifact_output_settings=declared_output_settings,
        required_variable_components=tuple(
            component.name for component in module_type.required_variable_components
        ),
    )


def _cellprofiler_module_type(
    func: Callable,
    metadata: FunctionMetadata,
    module_contract: ModuleArtifactContract | None,
):
    from openhcs.processing.backends.cellprofiler.module_classes import CellProfilerModule

    if module_contract is not None:
        return CellProfilerModule.for_module(module_contract.module_name)
    function_names = tuple(
        dict.fromkeys(
            (
                metadata.original_name,
                metadata.name,
                func.__name__,
            )
        )
    )
    for module_type in CellProfilerModule.__registry__.values():
        declared_names = module_type.declared_function_names()
        if any(function_name in declared_names for function_name in function_names):
            return module_type
    return None


def _declared_setting_summary(setting, kind: ArtifactKind) -> str:
    return f"{_declared_setting_name(setting)}:{kind.value}"


def _declared_setting_name(setting) -> str:
    if isinstance(setting, Enum):
        return setting.value
    return str(setting)


def _source_binding_rule(
    cellprofiler_module: CellProfilerModuleDeclarationSummary | None,
    module_contract: ModuleArtifactContract | None,
) -> str | None:
    if cellprofiler_module is not None:
        return (
            "CellProfiler source bindings are generated from the module "
            "declaration and .cppipe settings, then validated by the compiler."
        )
    if module_contract is not None and module_contract.declared_input_specs():
        return (
            "Source and runtime artifact bindings are resolved from the callable "
            "ModuleArtifactContract during compilation."
        )
    return None


def _materialization_rule(
    cellprofiler_module: CellProfilerModuleDeclarationSummary | None,
    module_contract: ModuleArtifactContract | None,
) -> str | None:
    if cellprofiler_module is not None:
        return (
            "Concrete CellProfiler sidecars and materialized artifacts are derived "
            "from the generated module contract and runtime plan, not chosen by MCP."
        )
    if module_contract is not None and module_contract.outputs:
        return (
            "Artifact output materialization is derived from output artifact kinds "
            "and compile/runtime materialization policy."
        )
    return None


def _measurement_rule(
    cellprofiler_module: CellProfilerModuleDeclarationSummary | None,
    module_contract: ModuleArtifactContract | None,
) -> str | None:
    measurement_kinds = {ArtifactKind.MEASUREMENTS, ArtifactKind.RELATIONSHIPS}
    if module_contract is not None and any(
        spec.kind in measurement_kinds
        for spec in (
            *module_contract.inputs,
            *module_contract.runtime_artifact_inputs,
            *module_contract.outputs,
        )
    ):
        return (
            "Measurement and relationship rows are projected by the selected "
            "runtime plan and materialized as declared artifacts."
        )
    if cellprofiler_module is not None:
        return (
            "If this module emits measurements, row ownership and sharing are "
            "declaration/runtime-plan behavior rather than agent-authored wiring."
        )
    return None


def _pattern_compatibility_rule(
    cellprofiler_module: CellProfilerModuleDeclarationSummary | None,
) -> str | None:
    if cellprofiler_module is None:
        return (
            "Regular OpenHCS callables may participate in standard FunctionStep "
            "callable, tuple, list, or dict patterns subject to compiler validation."
        )
    return (
        "Generated CellProfiler lowering uses one CP module contract per "
        "FunctionStep by default. Do not mix multiple CP module callables in one "
        "generated step unless declarations and runtime binding explicitly allow it."
    )


def _enum_member_name(value: Enum | None) -> str | None:
    if value is None:
        return None
    return value.name


def _detail_doc(func: Callable, metadata_doc: str | None) -> str | None:
    inspect_doc = inspect.getdoc(func)
    if inspect_doc is not None:
        return inspect_doc
    if metadata_doc is not None and metadata_doc.strip():
        return metadata_doc
    return None


def _bounded_detail_doc(
    doc: str | None,
    *,
    max_doc_chars: int | None,
) -> tuple[str | None, bool, int | None]:
    if doc is None:
        return None, False, max_doc_chars
    if max_doc_chars is None:
        return doc, False, None
    effective_max_doc_chars = max(0, min(max_doc_chars, MAX_FUNCTION_DETAIL_DOC_CHARS))
    if len(doc) <= effective_max_doc_chars:
        return doc, False, effective_max_doc_chars
    return doc[:effective_max_doc_chars].rstrip(), True, effective_max_doc_chars
