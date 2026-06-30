"""Source-backed documentation knowledge base for OpenHCS agents."""

from __future__ import annotations

import re
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import ClassVar, Protocol, cast

from openhcs.agent.dto.common import (
    AgentError,
    AgentWarning,
    JsonObject,
    JsonValue,
    SCHEMA_VERSION,
)
from openhcs.agent.dto.knowledge import (
    KnowledgeBaseCatalog,
    KnowledgeBaseContentBounds,
    KnowledgeBaseDocument,
    KnowledgeBaseDocumentRequest,
    KnowledgeBaseDocumentSourceProjection,
    KnowledgeBaseDocumentSummary,
    KnowledgeBaseSearchHit,
    KnowledgeBaseSearchRequest,
    KnowledgeBaseSearchResult,
    KnowledgeBaseSectionSummary,
    KnowledgeBaseSourceSpan,
)
from openhcs.agent.path_policy import AgentPathPolicy


DEFAULT_MAX_DOCUMENT_CHARS = 12_000
MAX_DOCUMENT_CHARS = 50_000
MAX_SEARCH_HITS = 25


class KnowledgeBaseIssueCode(str, Enum):
    DOCUMENT_MISSING = "knowledge_document_missing"
    DOCUMENT_UNKNOWN = "knowledge_document_unknown"
    QUERY_EMPTY = "knowledge_query_empty"
    SECTION_UNKNOWN = "knowledge_section_unknown"


@dataclass(frozen=True, slots=True)
class KnowledgeBaseDocumentSpec:
    document: KnowledgeBaseDocumentSummary


class _ComparisonManifestPathResolverLike(Protocol):
    def resolve(self, raw_case: Mapping[str, JsonValue], path_key: str) -> Path:
        """Resolve one case path through benchmark manifest path roots."""


class _ComparisonManifestLike(Protocol):
    path_resolver: _ComparisonManifestPathResolverLike


@dataclass(frozen=True, slots=True)
class _Official30CaseModuleInventory:
    case_name: str
    cppipe_path: Path | None
    modules: tuple[str, ...]

    @property
    def unique_modules(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.modules))


@dataclass(frozen=True, slots=True)
class _ExampleSourceFile:
    relative_path: Path
    source_text: str

    @property
    def line_count(self) -> int:
        return len(self.source_text.splitlines())


DEFAULT_KNOWLEDGE_BASE_MANIFEST_PATH = Path(
    "docs/source/development/mcp_knowledge_base_manifest.json"
)


class KnowledgeBaseManifestField(str, Enum):
    """JSON field names for the source-backed knowledge-base manifest."""

    DOCUMENTS = "documents"
    DOCUMENT_ID = "document_id"
    TITLE = "title"
    SUMMARY = "summary"
    SOURCE_PATH = "source_path"
    TAGS = "tags"
    SECTION_COUNT = "section_count"


def load_document_specs_from_manifest(
    manifest_path: Path,
) -> tuple[KnowledgeBaseDocumentSpec, ...]:
    """Load knowledge-base document specs from the source manifest."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("Knowledge-base manifest root must be a JSON object")
    documents = _manifest_sequence(manifest, KnowledgeBaseManifestField.DOCUMENTS)
    specs = tuple(_manifest_document_spec(document) for document in documents)
    if not specs:
        raise ValueError("Knowledge-base manifest must declare at least one document")
    return specs


@lru_cache(maxsize=1)
def default_document_specs() -> tuple[KnowledgeBaseDocumentSpec, ...]:
    """Return the source-backed default knowledge-base document specs."""
    return load_document_specs_from_manifest(
        _default_repo_root() / DEFAULT_KNOWLEDGE_BASE_MANIFEST_PATH
    )


def _manifest_document_spec(document: JsonObject) -> KnowledgeBaseDocumentSpec:
    return KnowledgeBaseDocumentSpec(
        KnowledgeBaseDocumentSummary(
            document_id=_manifest_string(
                document,
                KnowledgeBaseManifestField.DOCUMENT_ID,
            ),
            title=_manifest_string(document, KnowledgeBaseManifestField.TITLE),
            summary=_manifest_string(document, KnowledgeBaseManifestField.SUMMARY),
            source_path=_manifest_string(
                document,
                KnowledgeBaseManifestField.SOURCE_PATH,
            ),
            tags=_manifest_string_tuple(document, KnowledgeBaseManifestField.TAGS),
            section_count=_manifest_int(
                document,
                KnowledgeBaseManifestField.SECTION_COUNT,
            ),
        )
    )


def _manifest_value(document: JsonObject, field: KnowledgeBaseManifestField) -> JsonValue:
    if field.value not in document:
        raise ValueError(f"Knowledge-base manifest document missing {field.value!r}")
    return document[field.value]


def _manifest_string(document: JsonObject, field: KnowledgeBaseManifestField) -> str:
    value = _manifest_value(document, field)
    if not isinstance(value, str):
        raise ValueError(f"Knowledge-base manifest field {field.value!r} must be a string")
    return value


def _manifest_int(document: JsonObject, field: KnowledgeBaseManifestField) -> int:
    value = _manifest_value(document, field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Knowledge-base manifest field {field.value!r} must be an integer")
    return value


def _manifest_sequence(
    document: JsonObject,
    field: KnowledgeBaseManifestField,
) -> tuple[JsonObject, ...]:
    value = _manifest_value(document, field)
    if not isinstance(value, list):
        raise ValueError(f"Knowledge-base manifest field {field.value!r} must be a list")
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError(
                f"Knowledge-base manifest field {field.value!r} must contain objects"
            )
    return tuple(value)


def _manifest_string_tuple(
    document: JsonObject,
    field: KnowledgeBaseManifestField,
) -> tuple[str, ...]:
    value = _manifest_value(document, field)
    if not isinstance(value, list):
        raise ValueError(f"Knowledge-base manifest field {field.value!r} must be a list")
    for item in value:
        if not isinstance(item, str):
            raise ValueError(
                f"Knowledge-base manifest field {field.value!r} must contain strings"
            )
    return tuple(value)


@dataclass(frozen=True, slots=True)
class _ParsedDocument:
    spec: KnowledgeBaseDocumentSpec
    source_path: Path
    text: str
    lines: tuple[str, ...]
    sections: tuple[KnowledgeBaseSectionSummary, ...]

    def source_projection(self, repo_root: Path) -> KnowledgeBaseDocumentSourceProjection:
        return KnowledgeBaseDocumentSourceProjection(
            source_path=self.source_path.relative_to(repo_root).as_posix(),
            section_count=len(self.sections),
        )

    def search_lines_for_section(self, section_index: int) -> tuple[str, ...]:
        section = self.sections[section_index]
        next_heading_line = (
            self.sections[section_index + 1].start_line
            if section_index + 1 < len(self.sections)
            else section.end_line + 1
        )
        return self.lines[
            section.start_line - 1 : max(section.start_line, next_heading_line - 1)
        ]


@dataclass(frozen=True, slots=True)
class KnowledgeBaseRootPolicy:
    path_policy: AgentPathPolicy
    document_specs: tuple[KnowledgeBaseDocumentSpec, ...]

    def repo_root(self) -> Path:
        candidate_roots = tuple(
            root.resolve()
            for root in self.path_policy.readable_roots.roots
        )
        if not candidate_roots:
            raise ValueError("Knowledge base requires at least one readable root")
        canonical_root = self._canonical_document_root(candidate_roots)
        if canonical_root is not None:
            return canonical_root
        return candidate_roots[0]

    def _canonical_document_root(
        self,
        candidate_roots: tuple[Path, ...],
    ) -> Path | None:
        canonical_document = self.document_specs[0].document
        for candidate_root in candidate_roots:
            try:
                source_path = KnowledgeBaseService.resolve_source_path(
                    candidate_root,
                    canonical_document,
                )
            except ValueError:
                continue
            if source_path.is_file():
                return candidate_root
        return None


@dataclass(frozen=True, slots=True)
class KnowledgeBaseSearchQuery:
    raw: str
    normalized: str
    terms: tuple[str, ...]

    STOP_WORDS: ClassVar[tuple[str, ...]] = (
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
    )

    @classmethod
    def from_text(cls, raw: str) -> "KnowledgeBaseSearchQuery":
        normalized = raw.strip().casefold()
        terms = cls._terms_from_normalized_text(normalized)
        return cls(
            raw=raw,
            normalized=normalized,
            terms=terms,
        )

    @classmethod
    def _terms_from_normalized_text(cls, normalized: str) -> tuple[str, ...]:
        terms: list[str] = []
        seen: set[str] = set()
        for term in re.findall(r"[a-z0-9_]+", normalized):
            if term in cls.STOP_WORDS or term in seen:
                continue
            terms.append(term)
            seen.add(term)
        return tuple(terms)

    @property
    def is_empty(self) -> bool:
        return not self.normalized

    def matches_document(self, document: KnowledgeBaseDocumentSummary) -> bool:
        if document.matches_query(self.normalized):
            return True
        return self.matches_text(
            " ".join(
                (
                    document.document_id,
                    document.title,
                    document.summary,
                    " ".join(document.tags),
                )
            )
        )

    def matches_text(self, text: str) -> bool:
        return self.score_text(text)[0] > 0

    def score_text(self, text: str) -> tuple[int, tuple[str, ...]]:
        normalized_text = text.casefold()
        exact_match = bool(self.normalized and self.normalized in normalized_text)
        matched_terms = self.matched_terms(text)
        if exact_match:
            return 1_000 + len(matched_terms), matched_terms
        if not matched_terms:
            return 0, ()
        score = len(matched_terms) * 10
        if len(matched_terms) == len(self.terms):
            score += 250
        return score, matched_terms

    def matched_terms(self, text: str) -> tuple[str, ...]:
        normalized_text = text.casefold()
        if self.normalized in normalized_text:
            return self.terms
        if not self.terms:
            return ()
        return tuple(
            term
            for term in self.terms
            if any(variant in normalized_text for variant in self._term_variants(term))
        )

    @staticmethod
    def _term_variants(term: str) -> tuple[str, ...]:
        variants = [term]
        if len(term) > 4 and term.endswith("ies"):
            variants.append(f"{term[:-3]}y")
        if len(term) > 3 and term.endswith("s"):
            variants.append(term[:-1])
        return tuple(dict.fromkeys(variants))

    def first_matching_line(
        self,
        lines: tuple[str, ...],
        start_line: int,
    ) -> int | None:
        best_index: int | None = None
        best_score = 0
        for index, line in enumerate(lines):
            score, _ = self.score_text(line)
            if score > best_score:
                best_score = score
                best_index = index
        if best_index is None:
            return None
        return start_line + best_index

    def snippet(self, lines: tuple[str, ...]) -> str:
        scored_lines = tuple(
            (self.score_text(line)[0], line.strip())
            for line in lines
            if line.strip()
        )
        for _, line in sorted(scored_lines, key=lambda item: item[0], reverse=True):
            if line:
                return line[:240]
        for line in lines:
            stripped = line.strip()
            if stripped:
                return stripped[:240]
        return ""


class KnowledgeBaseService:
    """Read-only OpenHCS documentation surface for agents."""

    def __init__(
        self,
        *,
        repo_root: Path | None = None,
        document_specs: tuple[KnowledgeBaseDocumentSpec, ...] | None = None,
    ) -> None:
        self._repo_root = (repo_root or _default_repo_root()).resolve()
        document_specs = document_specs or default_document_specs()
        self._document_specs = document_specs
        self._specs_by_id = {spec.document.document_id: spec for spec in document_specs}
        if len(self._specs_by_id) != len(document_specs):
            raise ValueError("Duplicate knowledge-base document id")

    @classmethod
    def from_path_policy(
        cls,
        path_policy: AgentPathPolicy,
        *,
        document_specs: tuple[KnowledgeBaseDocumentSpec, ...] | None = None,
    ) -> "KnowledgeBaseService":
        document_specs = document_specs or default_document_specs()
        return cls(
            repo_root=KnowledgeBaseRootPolicy(
                path_policy=path_policy,
                document_specs=document_specs,
            ).repo_root(),
            document_specs=document_specs,
        )

    @classmethod
    def default_source_paths(cls) -> tuple[Path, ...]:
        repo_root = _default_repo_root()
        return tuple(
            path
            for path in (
                cls.resolve_source_path(repo_root, spec.document)
                for spec in default_document_specs()
            )
        )

    def list_documents(self) -> KnowledgeBaseCatalog:
        parsed_documents = self._existing_parsed_documents()
        return KnowledgeBaseCatalog(
            schema_version=SCHEMA_VERSION,
            documents=tuple(
                self._document_summary(parsed)
                for parsed in parsed_documents
            ),
            warnings=self._missing_document_warnings(),
        )

    def get_document(
        self,
        request: KnowledgeBaseDocumentRequest,
    ) -> KnowledgeBaseDocument:
        target = request.target
        bounds = request.bounds
        document_id = target.document_id
        spec = self._specs_by_id.get(document_id)
        if spec is None:
            return self._document_error(
                KnowledgeBaseIssueCode.DOCUMENT_UNKNOWN,
                f"Unknown knowledge-base document {document_id!r}.",
                hint=f"Known documents: {', '.join(self._specs_by_id)}",
                bounds=bounds,
            )

        source_path = self._source_path(spec)
        if not source_path.is_file():
            return self._document_error(
                KnowledgeBaseIssueCode.DOCUMENT_MISSING,
                f"Knowledge-base document {document_id!r} is not present on disk.",
                path=spec.document.source_path,
                bounds=bounds,
            )

        parsed = self._parse_document(spec)
        selected_lines = self._document_content_lines(parsed)
        section_id = target.section_id
        if section_id is not None:
            section = target.find_section(parsed.sections)
            if section is None:
                return KnowledgeBaseDocument(
                    schema_version=SCHEMA_VERSION,
                    document=self._document_summary(parsed),
                    sections=parsed.sections,
                    content="",
                    selected_section_id=section_id,
                    truncated=False,
                    max_chars=bounds.effective_max_chars(MAX_DOCUMENT_CHARS),
                    errors=(
                        AgentError(
                            code=KnowledgeBaseIssueCode.SECTION_UNKNOWN.value,
                            message=(
                                f"Unknown section {section_id!r} in knowledge-base "
                                f"document {document_id!r}."
                            ),
                            hint=(
                                "Use openhcs_get_knowledge_document without "
                                "section_id to inspect available sections."
                            ),
                        ),
                    ),
                )
            selected_lines = section.span.line_slice(parsed.lines)

        content, truncated, effective_max_chars = bounds.apply(
            "\n".join(selected_lines),
            MAX_DOCUMENT_CHARS,
        )
        return KnowledgeBaseDocument(
            schema_version=SCHEMA_VERSION,
            document=self._document_summary(parsed),
            sections=parsed.sections,
            content=content,
            selected_section_id=section_id,
            truncated=truncated,
            max_chars=effective_max_chars,
        )

    def _document_content_lines(self, parsed: _ParsedDocument) -> tuple[str, ...]:
        return parsed.lines

    @staticmethod
    def _display_lines(
        spec: KnowledgeBaseDocumentSpec,
        text: str,
        lines: tuple[str, ...],
        *,
        repo_root: Path,
        source_path: Path,
    ) -> tuple[str, ...]:
        manifest = KnowledgeBaseService._official30_recipe_manifest(text)
        if manifest is not None:
            return KnowledgeBaseService._official30_recipe_content_lines(
                spec,
                manifest,
                lines,
                repo_root=repo_root,
                source_path=source_path,
            )
        if spec.document.document_id == "openhcs_example_corpus_map":
            return (
                *lines,
                *KnowledgeBaseService._native_example_source_projection_lines(
                    lines,
                    repo_root,
                ),
            )
        return lines

    @staticmethod
    def _native_example_source_projection_lines(
        lines: tuple[str, ...],
        repo_root: Path,
    ) -> tuple[str, ...]:
        source_files = _native_example_source_files(lines, repo_root)
        if not source_files:
            return ()

        projected_lines = [
            "",
            "Native Example Source Index",
            "---------------------------",
            "",
            "Generated from the Python paths declared in the Native OpenHCS "
            "Examples section. Use the file-path section ids to inspect actual "
            "source without duplicating example code in the documentation.",
            "",
        ]
        for source_file in source_files:
            projected_lines.append(
                f"* ``{source_file.relative_path.as_posix()}`` "
                f"({source_file.line_count} lines)"
            )

        for source_file in source_files:
            title = source_file.relative_path.as_posix()
            projected_lines.extend(("", title, "-" * len(title), ""))
            projected_lines.append(f"Source path: {title}")
            projected_lines.append(f"Lines included: {source_file.line_count}")
            projected_lines.extend(("", ".. code-block:: python", ""))
            projected_lines.extend(
                f"   {line}" if line else ""
                for line in source_file.source_text.splitlines()
            )
        return tuple(projected_lines)

    @staticmethod
    def _official30_recipe_manifest(text: str) -> Mapping[str, JsonValue] | None:
        try:
            manifest = json.loads(text)
        except json.JSONDecodeError:
            return None
        if not isinstance(manifest, Mapping):
            return None
        if not isinstance(manifest.get("cases"), list):
            return None
        return cast(Mapping[str, JsonValue], manifest)

    @staticmethod
    def _official30_recipe_content_lines(
        spec: KnowledgeBaseDocumentSpec,
        manifest: Mapping[str, JsonValue],
        fallback_lines: tuple[str, ...],
        *,
        repo_root: Path,
        source_path: Path,
    ) -> tuple[str, ...]:
        cases = manifest.get("cases")
        if not isinstance(cases, list):
            return fallback_lines

        recipe_lines = [
            "Official30 Benchmark Pipeline Recipes",
            "=" * len("Official30 Benchmark Pipeline Recipes"),
            "",
            f"Source manifest: {spec.document.source_path}",
            f"Manifest version: {manifest.get('manifest_version', '<unknown>')}",
            f"Recipe count: {len(cases)}",
            KnowledgeBaseService._official30_mapping_line(
                "Default pipeline params",
                manifest.get("default_pipeline_params"),
            ),
            KnowledgeBaseService._official30_mapping_line(
                "Path roots",
                manifest.get("path_roots"),
            ),
            "",
            "Use the case section id with ``knowledge-document "
            f"{spec.document.document_id} --section-id <case>`` "
            "to inspect one recipe without loading the full manifest.",
            "",
            "Case Index",
            "----------",
            "",
        ]
        valid_cases: list[Mapping[str, JsonValue]] = [
            case
            for case in cases
            if isinstance(case, dict)
        ]
        inventories = KnowledgeBaseService._official30_module_inventories(
            source_path=source_path,
            repo_root=repo_root,
            cases=tuple(valid_cases),
        )
        for index, case in enumerate(valid_cases, 1):
            recipe_lines.append(
                KnowledgeBaseService._official30_recipe_line(index, case)
            )

        recipe_lines.extend(
            KnowledgeBaseService._official30_module_usage_lines(inventories)
        )

        inventories_by_case = {
            inventory.case_name: inventory
            for inventory in inventories
        }
        for case in valid_cases:
            recipe_lines.extend(
                KnowledgeBaseService._official30_case_section_lines(
                    case,
                    inventories_by_case.get(str(case.get("name") or "<unnamed>")),
                )
            )
        return tuple(recipe_lines)

    @staticmethod
    def _official30_case_section_lines(
        case: Mapping[str, JsonValue],
        inventory: _Official30CaseModuleInventory | None,
    ) -> tuple[str, ...]:
        name = str(case.get("name") or "<unnamed>")
        lines = [
            "",
            name,
            "~" * len(name),
            "",
        ]
        for key in (
            "dataset_id",
            "dataset_path",
            "dataset_path_root",
            "cppipe_path",
            "cppipe_path_root",
            "microscope_type",
            "assay_category",
            "module_category",
            "cellprofiler_timeout_seconds",
            "value_only",
        ):
            if key not in case:
                continue
            lines.append(
                f"{key}: {KnowledgeBaseService._official30_scalar(case[key])}"
            )
        if inventory is not None:
            if inventory.modules:
                lines.append(
                    "modules: "
                    + ", ".join(inventory.unique_modules)
                )
            if inventory.cppipe_path is not None:
                lines.append(
                    f"resolved_cppipe_path: {inventory.cppipe_path}"
                )
        if "pipeline_params" in case:
            lines.append(
                KnowledgeBaseService._official30_mapping_line(
                    "pipeline_params",
                    case.get("pipeline_params"),
                )
            )
        return tuple(lines)

    @staticmethod
    def _official30_mapping_line(label: str, value: JsonValue) -> str:
        if not isinstance(value, dict) or not value:
            return f"{label}: <none>"
        return (
            f"{label}: "
            + ", ".join(
                f"{key}={KnowledgeBaseService._official30_scalar(item)}"
                for key, item in sorted(value.items())
            )
        )

    @staticmethod
    def _official30_scalar(value: JsonValue) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if value is None:
            return "<none>"
        if isinstance(value, (int, float, str)):
            return str(value)
        if isinstance(value, dict):
            items = tuple(sorted(value.items()))
            visible_items = items[:4]
            body = ", ".join(
                f"{key}={KnowledgeBaseService._official30_scalar(item)}"
                for key, item in visible_items
            )
            if len(items) > len(visible_items):
                body += f", +{len(items) - len(visible_items)}"
            return f"{{{body}}}"
        if isinstance(value, (list, tuple)):
            visible_items = value[:4]
            body = ", ".join(
                KnowledgeBaseService._official30_scalar(item)
                for item in visible_items
            )
            if len(value) > len(visible_items):
                body += f", +{len(value) - len(visible_items)}"
            return f"[{body}]"
        return json.dumps(value, sort_keys=True)

    @staticmethod
    def _official30_recipe_line(
        index: int,
        case: Mapping[str, JsonValue],
    ) -> str:
        name = str(case.get("name") or "<unnamed>")
        dataset_id = str(case.get("dataset_id") or "<unknown>")
        cppipe_path = str(case.get("cppipe_path") or "<none>")
        return f"{index}. {name}: dataset={dataset_id} cppipe={cppipe_path}"

    @staticmethod
    def _official30_module_usage_lines(
        inventories: tuple[_Official30CaseModuleInventory, ...],
    ) -> tuple[str, ...]:
        lines = [
            "",
            "Module Usage Index",
            "------------------",
            "",
            "Derived from the official30 manifest's resolved .cppipe paths. "
            "The benchmark manifest acquisition layer can materialize missing "
            "CellProfiler examples and tutorial datasets into the configured "
            "cache roots.",
            "",
        ]
        cases_by_module: dict[str, list[str]] = {}
        for inventory in inventories:
            if inventory.modules:
                for module_name in inventory.unique_modules:
                    cases_by_module.setdefault(module_name, []).append(
                        inventory.case_name
                    )
                lines.append(
                    f"{inventory.case_name}: module_count={len(inventory.modules)} "
                    f"modules={', '.join(inventory.unique_modules)}"
                )
            else:
                lines.append(
                    f"{inventory.case_name}: module inventory unavailable; "
                    "materialize the benchmark manifest roots and retry."
                )

        lines.extend(("", "Module To Case Lookup", "^^^^^^^^^^^^^^^^^^^^", ""))
        for module_name, case_names in sorted(cases_by_module.items()):
            lines.append(f"{module_name}: cases={', '.join(case_names)}")
        return tuple(lines)

    @staticmethod
    def _official30_module_inventories(
        *,
        source_path: Path,
        repo_root: Path,
        cases: tuple[Mapping[str, JsonValue], ...],
    ) -> tuple[_Official30CaseModuleInventory, ...]:
        try:
            return _official30_module_inventories_cached(
                str(source_path),
                source_path.stat().st_mtime_ns,
                str(repo_root),
                os.environ.get("CELLPROFILER_EXAMPLES_ROOT"),
                os.environ.get("OPENHCS_BENCHMARK_DATASET_CACHE_ROOT"),
            )
        except (ImportError, OSError, TypeError, ValueError, UnicodeDecodeError):
            return tuple(
                _Official30CaseModuleInventory(
                    case_name=str(case.get("name") or "<unnamed>"),
                    cppipe_path=None,
                    modules=(),
                )
                for case in cases
            )

    def search(self, request: KnowledgeBaseSearchRequest) -> KnowledgeBaseSearchResult:
        query = KnowledgeBaseSearchQuery.from_text(request.query)
        if query.is_empty:
            return KnowledgeBaseSearchResult(
                schema_version=SCHEMA_VERSION,
                query=request.query,
                hits=(),
                errors=(
                    AgentError(
                        code=KnowledgeBaseIssueCode.QUERY_EMPTY.value,
                        message="Knowledge-base search query must not be empty.",
                    ),
                ),
        )

        hit_limit = max(1, min(request.limit, MAX_SEARCH_HITS))
        ranked_hits: list[tuple[int, int, KnowledgeBaseSearchHit]] = []
        parsed_documents = self._existing_parsed_documents()
        for document_index, parsed in enumerate(parsed_documents):
            spec = parsed.spec
            summary = self._document_summary(parsed)

            document_text = " ".join(
                (
                    spec.document.document_id,
                    spec.document.title,
                    spec.document.summary,
                    " ".join(spec.document.tags),
                )
            )
            score, matched_terms = query.score_text(document_text)
            if score:
                ranked_hits.append(
                    (
                        score,
                        document_index * 1_000,
                        KnowledgeBaseSearchHit(
                            document=summary,
                            section=None,
                            line_number=None,
                            snippet=spec.document.summary,
                            score=score,
                            matched_terms=matched_terms,
                        ),
                    )
                )

        for document_index, parsed in enumerate(parsed_documents):
            spec = parsed.spec
            summary = self._document_summary(parsed)
            for section_index, section in enumerate(parsed.sections):
                if section.level == 1:
                    continue
                section_lines = parsed.search_lines_for_section(section_index)
                section_text = "\n".join(section_lines)
                title_score, title_terms = query.score_text(section.title)
                text_score, text_terms = query.score_text(section_text)
                score = title_score + text_score
                if (
                    self._official30_recipe_manifest(parsed.text) is not None
                    and not self._official30_query_is_specific_to_case(
                        query,
                        section,
                        section_text,
                    )
                ):
                    score = max(0, score - 50)
                if (
                    spec.document.document_id == "openhcs_example_corpus_map"
                    and _native_example_source_section(section)
                    and not _native_example_query_is_specific_to_source(query)
                ):
                    score = max(0, score - 80)
                if not score:
                    continue
                matched_terms = tuple(
                    dict.fromkeys((*title_terms, *text_terms))
                )
                ranked_hits.append(
                    (
                        score,
                        document_index * 1_000 + section_index + 1,
                        KnowledgeBaseSearchHit(
                            document=summary,
                            section=section,
                            line_number=query.first_matching_line(
                                section_lines,
                                section.start_line,
                            ),
                            snippet=query.snippet(section_lines),
                            score=score,
                            matched_terms=matched_terms,
                        ),
                    )
                )

        hits = tuple(
            hit
            for _, _, hit in sorted(
                ranked_hits,
                key=lambda item: (-item[0], item[1]),
            )[:hit_limit]
        )

        return KnowledgeBaseSearchResult(
            schema_version=SCHEMA_VERSION,
            query=request.query,
            hits=hits,
            warnings=self._missing_document_warnings(),
        )

    @staticmethod
    def _official30_query_is_specific_to_case(
        query: KnowledgeBaseSearchQuery,
        section: KnowledgeBaseSectionSummary,
        section_text: str,
    ) -> bool:
        recipe_terms = {
            "cellprofiler",
            "example",
            "examples",
            "official30",
            "module",
            "modules",
            "recipe",
            "recipes",
            "benchmark",
            "cppipe",
            "manifest",
        }
        if any(term in recipe_terms for term in query.terms):
            return True
        if section.section_id == "module-usage-index":
            case_names = frozenset(
                match.group(1)
                for match in re.finditer(
                    r"^([A-Za-z0-9_]+): module_count=",
                    section_text,
                    flags=re.MULTILINE,
                )
            )
            module_terms: set[str] = set()
            for match in re.finditer(
                r"^([A-Za-z][A-Za-z0-9]+): cases=(.+)$",
                section_text,
                flags=re.MULTILINE,
            ):
                module_case_names = frozenset(
                    item.strip()
                    for item in match.group(2).split(",")
                    if item.strip()
                )
                if case_names and module_case_names == case_names:
                    continue
                module_terms.add(match.group(1).casefold())
            return any(term in module_terms for term in query.terms)
        return any(term == section.section_id for term in query.terms)

    def _source_path(self, spec: KnowledgeBaseDocumentSpec) -> Path:
        return self.resolve_source_path(self._repo_root, spec.document)

    @staticmethod
    def resolve_source_path(
        repo_root: Path,
        document: KnowledgeBaseDocumentSummary,
    ) -> Path:
        relative_path = Path(document.source_path)
        if relative_path.is_absolute():
            raise ValueError(
                f"Knowledge-base document path must be relative: {document.source_path}"
            )
        resolved_path = (repo_root / relative_path).resolve()
        if not resolved_path.is_relative_to(repo_root):
            raise ValueError(
                f"Knowledge-base document path escapes repository root: {document.source_path}"
            )
        return resolved_path

    def _parse_document(self, spec: KnowledgeBaseDocumentSpec) -> _ParsedDocument:
        source_path = self._source_path(spec)
        text = source_path.read_text(encoding="utf-8")
        source_lines = tuple(text.splitlines())
        lines = self._display_lines(
            spec,
            text,
            source_lines,
            repo_root=self._repo_root,
            source_path=source_path,
        )
        return _ParsedDocument(
            spec=spec,
            source_path=source_path,
            text=text,
            lines=lines,
            sections=_parse_sections(lines),
        )

    def _existing_parsed_documents(self) -> tuple[_ParsedDocument, ...]:
        parsed_documents: list[_ParsedDocument] = []
        for spec in self._document_specs:
            if self._source_path(spec).is_file():
                parsed_documents.append(self._parse_document(spec))
        return tuple(parsed_documents)

    def _missing_document_warnings(self) -> tuple[AgentWarning, ...]:
        warnings: list[AgentWarning] = []
        for spec in self._document_specs:
            if self._source_path(spec).is_file():
                continue
            warnings.append(
                AgentWarning(
                    code=KnowledgeBaseIssueCode.DOCUMENT_MISSING.value,
                    message=(
                        "Allowlisted knowledge-base document "
                        f"{spec.document.document_id!r} is not present on disk."
                    ),
                    hint=spec.document.source_path,
                )
            )
        return tuple(warnings)

    def _document_summary(
        self,
        parsed: _ParsedDocument,
    ) -> KnowledgeBaseDocumentSummary:
        return parsed.spec.document.with_source_projection(
            parsed.source_projection(self._repo_root)
        )

    def _document_error(
        self,
        code: KnowledgeBaseIssueCode,
        message: str,
        *,
        hint: str | None = None,
        path: str | None = None,
        bounds: KnowledgeBaseContentBounds,
    ) -> KnowledgeBaseDocument:
        return KnowledgeBaseDocument(
            schema_version=SCHEMA_VERSION,
            document=None,
            sections=(),
            content="",
            selected_section_id=None,
            truncated=False,
            max_chars=bounds.effective_max_chars(MAX_DOCUMENT_CHARS),
            errors=(
                AgentError(
                    code=code.value,
                    message=message,
                    hint=hint,
                    path=path,
                ),
            ),
        )


_MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_RST_UNDERLINE_LEVELS = {
    "=": 1,
    "-": 2,
    "~": 3,
    "^": 4,
    '"': 5,
    "'": 6,
}


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@lru_cache(maxsize=8)
def _official30_module_inventories_cached(
    manifest_path: str,
    manifest_mtime_ns: int,
    repo_root: str,
    cellprofiler_examples_root: str | None,
    dataset_cache_root: str | None,
) -> tuple[_Official30CaseModuleInventory, ...]:
    del manifest_mtime_ns, cellprofiler_examples_root, dataset_cache_root

    from benchmark.contracts.comparison_manifest import ComparisonManifest
    from openhcs.interop.cellprofiler.parser import CPPipeParser

    manifest = ComparisonManifest.load(Path(manifest_path), materialize_roots=False)
    raw_cases = manifest.payload.get("cases")
    if not isinstance(raw_cases, list):
        return ()

    inventories: list[_Official30CaseModuleInventory] = []
    for raw_case in raw_cases:
        if not isinstance(raw_case, Mapping):
            continue
        case_name = str(raw_case.get("name") or "<unnamed>")
        cppipe_path = _official30_existing_cppipe_path(
            repo_root=Path(repo_root),
            manifest=manifest,
            raw_case=raw_case,
            case_name=case_name,
        )
        modules: tuple[str, ...] = ()
        if cppipe_path is not None:
            try:
                modules = tuple(
                    module.name
                    for module in CPPipeParser(cppipe_path).parse()
                    if module.enabled
                )
            except (OSError, ValueError, UnicodeDecodeError):
                modules = ()
        inventories.append(
            _Official30CaseModuleInventory(
                case_name=case_name,
                cppipe_path=cppipe_path,
                modules=modules,
            )
        )
    return tuple(inventories)


def _official30_existing_cppipe_path(
    *,
    repo_root: Path,
    manifest: _ComparisonManifestLike,
    raw_case: Mapping[str, JsonValue],
    case_name: str,
) -> Path | None:
    try:
        resolved_path = manifest.path_resolver.resolve(raw_case, "cppipe_path")
    except (KeyError, TypeError, ValueError):
        resolved_path = None
    if isinstance(resolved_path, Path) and resolved_path.is_file():
        return resolved_path

    raw_cppipe_path = raw_case.get("cppipe_path")
    if raw_cppipe_path is None:
        return None
    return _official30_native_ref_cppipe_path(
        repo_root,
        case_name,
        Path(str(raw_cppipe_path)).name,
    )


def _official30_native_ref_cppipe_path(
    repo_root: Path,
    case_name: str,
    cppipe_name: str,
) -> Path | None:
    native_refs_root = repo_root / "benchmark/native_refs/official30_scoped_rows"
    candidates = tuple(
        sorted(native_refs_root.glob(f"*/native_cellprofiler_headless/{cppipe_name}"))
    )
    for candidate in candidates:
        if case_name in candidate.parent.parent.name:
            return candidate
    if len(candidates) == 1:
        return candidates[0]
    return None


def _native_example_source_files(
    lines: tuple[str, ...],
    repo_root: Path,
) -> tuple[_ExampleSourceFile, ...]:
    source_paths = tuple(
        dict.fromkeys(
            _native_example_python_paths(
                _native_example_source_references(lines),
                repo_root,
            )
        )
    )
    source_files: list[_ExampleSourceFile] = []
    for source_path in source_paths:
        try:
            source_text = source_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        source_files.append(
            _ExampleSourceFile(
                relative_path=source_path.relative_to(repo_root),
                source_text=source_text,
            )
        )
    return tuple(source_files)


def _native_example_source_section(section: KnowledgeBaseSectionSummary) -> bool:
    return section.title.endswith(".py")


def _native_example_query_is_specific_to_source(
    query: KnowledgeBaseSearchQuery,
) -> bool:
    source_terms = {
        "benchmark",
        "code",
        "example",
        "examples",
        "functionstep",
        "pipeline_steps",
        "preset",
        "presets",
        "py",
        "python",
        "source",
    }
    return any(term in source_terms or "_" in term for term in query.terms)


def _native_example_source_references(lines: tuple[str, ...]) -> tuple[str, ...]:
    section_lines = _rst_section_body_lines(lines, "Native OpenHCS Examples")
    references: list[str] = []
    for line in section_lines:
        for match in re.finditer(r"``([^`]+)``", line):
            value = match.group(1)
            if value.endswith("/") or value.endswith(".py"):
                references.append(value)
    return tuple(references)


def _native_example_python_paths(
    references: tuple[str, ...],
    repo_root: Path,
) -> tuple[Path, ...]:
    paths: list[Path] = []
    for reference in references:
        source_path = _repo_relative_path(repo_root, reference)
        if source_path is None:
            continue
        if source_path.is_dir():
            paths.extend(
                candidate
                for candidate in sorted(source_path.glob("*.py"))
                if candidate.name != "__init__.py"
            )
        elif source_path.is_file() and source_path.suffix == ".py":
            paths.append(source_path)
    return tuple(paths)


def _repo_relative_path(repo_root: Path, raw_path: str) -> Path | None:
    path = Path(raw_path)
    if path.is_absolute():
        return None
    resolved_path = (repo_root / path).resolve()
    if not resolved_path.is_relative_to(repo_root):
        return None
    return resolved_path


def _rst_section_body_lines(
    lines: tuple[str, ...],
    title: str,
) -> tuple[str, ...]:
    for index in range(len(lines) - 1):
        if lines[index].strip() != title:
            continue
        underline = lines[index + 1].strip()
        if not _is_rst_underline(underline, title):
            continue
        section_level = _RST_UNDERLINE_LEVELS[next(iter(set(underline)))]
        body_start = index + 2
        body_end = len(lines)
        for candidate_index in range(body_start, len(lines) - 1):
            candidate_title = lines[candidate_index].strip()
            candidate_underline = lines[candidate_index + 1].strip()
            if not candidate_title:
                continue
            if not _is_rst_underline(candidate_underline, candidate_title):
                continue
            candidate_level = _RST_UNDERLINE_LEVELS[
                next(iter(set(candidate_underline)))
            ]
            if candidate_level <= section_level:
                body_end = candidate_index
                break
        return lines[body_start:body_end]
    return ()


def _is_rst_underline(underline: str, title: str) -> bool:
    underline_chars = set(underline)
    if len(underline_chars) != 1:
        return False
    marker = next(iter(underline_chars))
    return marker in _RST_UNDERLINE_LEVELS and len(underline) >= len(title)


def _parse_sections(lines: tuple[str, ...]) -> tuple[KnowledgeBaseSectionSummary, ...]:
    sections = _parse_markdown_sections(lines)
    if not sections:
        sections = _parse_rst_sections(lines)
    return KnowledgeBaseSectionHierarchy(
        sections=sections,
        line_count=len(lines),
    ).with_end_lines()


def _parse_markdown_sections(
    lines: tuple[str, ...],
) -> tuple[KnowledgeBaseSectionSummary, ...]:
    sections: list[KnowledgeBaseSectionSummary] = []
    id_authority = UniqueSectionIdAuthority()
    for line_number, line in enumerate(lines, start=1):
        match = _MARKDOWN_HEADING_RE.match(line)
        if match is None:
            continue
        title = match.group(2).strip()
        sections.append(
            KnowledgeBaseSectionSummary(
                section_id=id_authority.next(title, line_number),
                title=title,
                level=len(match.group(1)),
                span=KnowledgeBaseSourceSpan(
                    start_line=line_number,
                    end_line=line_number,
                ),
            )
        )
    return tuple(sections)


def _parse_rst_sections(
    lines: tuple[str, ...],
) -> tuple[KnowledgeBaseSectionSummary, ...]:
    sections: list[KnowledgeBaseSectionSummary] = []
    id_authority = UniqueSectionIdAuthority()
    for index in range(len(lines) - 1):
        title = lines[index].strip()
        underline = lines[index + 1].strip()
        if not title:
            continue
        if not _is_rst_underline(underline, title):
            continue
        marker = next(iter(set(underline)))
        line_number = index + 1
        sections.append(
            KnowledgeBaseSectionSummary(
                section_id=id_authority.next(title, line_number),
                title=title,
                level=_RST_UNDERLINE_LEVELS[marker],
                span=KnowledgeBaseSourceSpan(
                    start_line=line_number,
                    end_line=line_number,
                ),
            )
        )
    return tuple(sections)


@dataclass(frozen=True, slots=True)
class KnowledgeBaseSectionHierarchy:
    sections: tuple[KnowledgeBaseSectionSummary, ...]
    line_count: int

    def with_end_lines(self) -> tuple[KnowledgeBaseSectionSummary, ...]:
        completed: list[KnowledgeBaseSectionSummary] = []
        for index, section in enumerate(self.sections):
            next_start_line = self.line_count + 1
            for candidate in self.sections[index + 1:]:
                if candidate.level <= section.level:
                    next_start_line = candidate.start_line
                    break
            completed.append(
                section.with_span(
                    section.span.close_before(next_start_line)
                )
            )
        return tuple(completed)


@dataclass(slots=True)
class UniqueSectionIdAuthority:
    used_ids: set[str] = field(default_factory=set)

    def next(self, title: str, line_number: int) -> str:
        section_id = re.sub(r"[^a-z0-9]+", "-", title.casefold()).strip("-")
        if not section_id:
            section_id = f"section-{line_number}"
        if section_id in self.used_ids:
            section_id = f"{section_id}-{line_number}"
        self.used_ids.add(section_id)
        return section_id
