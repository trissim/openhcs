"""Headless function registry projection for OpenHCS agents."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from openhcs.agent.dto.common import SCHEMA_VERSION
from openhcs.agent.dto.functions import (
    FunctionCatalogEntry,
    FunctionDetail,
    FunctionParameterSpec,
    catalog_page,
)
from openhcs.processing.backends.lib_registry.registry_service import RegistryService
from openhcs.processing.backends.lib_registry.unified_registry import FunctionMetadata


INTERNAL_PARAMS = frozenset({"enabled", "slice_by_slice", "dtype_config"})


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
        return cls(text, tuple(token for token in text.split() if token))

    def accepts_library(self, library: str) -> bool:
        return not self.text or library.casefold() == self.text

    def search_rank(
        self,
        entry: FunctionCatalogEntry,
        metadata: FunctionMetadata,
    ) -> CatalogSearchRank:
        if not self.text:
            return CatalogSearchRank.ALL_PASS
        name_text = _normalized_search_text(entry.name)
        function_id_text = _normalized_search_text(entry.function_id)
        import_path_text = _normalized_search_text(entry.import_path)
        tag_text = _normalized_search_text(" ".join(entry.backend_tags))
        if name_text == self.text:
            return CatalogSearchRank.EXACT_NAME
        if name_text.startswith(self.text):
            return CatalogSearchRank.PREFIX_NAME
        if self._matches_text(name_text):
            return CatalogSearchRank.NAME_CONTAINS
        if self._matches_text(function_id_text):
            return CatalogSearchRank.ID_CONTAINS
        if self._matches_text(import_path_text):
            return CatalogSearchRank.IMPORT_CONTAINS
        if self._matches_text(tag_text):
            return CatalogSearchRank.TAG_CONTAINS
        if entry.summary is not None and self._matches_text(entry.summary):
            return CatalogSearchRank.SUMMARY_CONTAINS
        if metadata.doc is not None and self._matches_text(metadata.doc):
            return CatalogSearchRank.DOC_CONTAINS
        return CatalogSearchRank.NO_MATCH

    def _matches_text(self, value: str) -> bool:
        text = _normalized_search_text(value)
        if self.text in text:
            return True
        return all(token in text for token in self.tokens)


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
    entry: FunctionCatalogEntry

    @property
    def sort_key(self) -> tuple[int, str, str]:
        return (
            self.rank.value,
            self.entry.name.casefold(),
            self.entry.function_id,
        )


class ParameterDocumentationPolicy:
    """Nominal policy for exposing callable parameters to agents."""

    hidden_names = INTERNAL_PARAMS
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
    ) -> bool:
        if name in self.hidden_names:
            return False
        if skip_keyword_bag and name == "kwargs":
            return False
        return parameter.kind not in self.variadic_kinds

    def display_signature(
        self,
        func: Callable,
        display_name: str,
        view: SignatureView,
    ) -> str:
        sig = inspect.signature(func)
        visible_parameters = tuple(
            parameter.replace(annotation=inspect.Parameter.empty)
            for name, parameter in sig.parameters.items()
            if self.should_document(name, parameter, skip_keyword_bag=True)
        )
        if view.compact and len(visible_parameters) > view.parameter_limit:
            parameter_names = ", ".join(
                parameter.name
                for parameter in visible_parameters[:view.parameter_limit]
            )
            return f"{display_name}({parameter_names}, ...)"
        visible_signature = sig.replace(parameters=visible_parameters)
        return f"{display_name}{visible_signature}"

    def parameter_specs(self, func: Callable) -> tuple[FunctionParameterSpec, ...]:
        sig = inspect.signature(func)
        specs = []
        for name, parameter in sig.parameters.items():
            if not self.should_document(name, parameter):
                continue
            specs.append(
                FunctionParameterSpec(
                    name=name,
                    annotation=_format_annotation(parameter.annotation),
                    default_repr=_format_default(parameter.default),
                    required=parameter.default is inspect.Parameter.empty,
                )
            )
        return tuple(specs)


PARAMETER_DOCUMENTATION_POLICY = ParameterDocumentationPolicy()


class FunctionCatalogService:
    """Expose registered OpenHCS processing callables through stable IDs."""

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
            rank = query_filter.search_rank(entry, metadata)
            if not rank.matched:
                continue
            candidates.append(CatalogSearchCandidate(rank, entry))

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

    def get(self, function_id: str) -> FunctionDetail:
        metadata = self._metadata(function_id)
        func = metadata.func
        return FunctionDetail(
            schema_version=SCHEMA_VERSION,
            entry=self._entry(function_id, metadata),
            parameters=PARAMETER_DOCUMENTATION_POLICY.parameter_specs(func),
            doc=_detail_doc(func, metadata.doc),
        )

    def resolve(self, function_id: str) -> Callable:
        return self._metadata(function_id).func

    def _all_metadata(self) -> dict[str, FunctionMetadata]:
        return RegistryService.get_all_functions_with_metadata()

    def _metadata(self, function_id: str) -> FunctionMetadata:
        try:
            return self._all_metadata()[function_id]
        except KeyError as exc:
            raise KeyError(f"Unknown OpenHCS function_id: {function_id}") from exc

    def _entry(
        self,
        function_id: str,
        metadata: FunctionMetadata,
        signature_view: SignatureView = SignatureView.FULL,
        summary_view: SummaryView = SummaryView.FULL,
    ) -> FunctionCatalogEntry:
        name = _metadata_display_name(function_id, metadata)
        module = _metadata_module(metadata)
        library = metadata.get_registry_name()
        signature = PARAMETER_DOCUMENTATION_POLICY.display_signature(
            metadata.func,
            name,
            signature_view,
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


def _detail_doc(func: Callable, metadata_doc: str | None) -> str | None:
    inspect_doc = inspect.getdoc(func)
    if inspect_doc is not None:
        return inspect_doc
    if metadata_doc is not None and metadata_doc.strip():
        return metadata_doc
    return None
