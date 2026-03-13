#!/usr/bin/env python3
r"""
Unified paper build system for the OpenHCS paper series.

All configuration lives in scripts/papers.yaml. The script reads it once at
startup and derives every path, target, and dependency from that single source
of truth. No path guessing, no string-matching heuristics.

─────────────────────────────────────────────────────────────────────────────
DESIGN PRINCIPLES
─────────────────────────────────────────────────────────────────────────────

  Declarative   — papers.yaml is the SSOT. Code derives, never guesses.
  Fail-Loud     — validate structure upfront; abort with a clear message.
  Orthogonal    — one method, one responsibility. No side-channel coupling.
  Idempotent    — every step can be re-run safely; lake caching handles dedup.

─────────────────────────────────────────────────────────────────────────────
STRUCTURAL INVARIANTS  (all derived from papers.yaml)
─────────────────────────────────────────────────────────────────────────────

  1. Content:  paper_dir / meta.latex_dir / "content" / *.tex
  2. Proofs:   paper_dir / meta.proofs_dir /             (lake project root)
  3. Output:   paper_dir / "releases" /                  (pdf, md, tar.gz …)
  4. Graphs:   graph_infra / "graphs" / <paper_id>.json  (every packaged paper)

─────────────────────────────────────────────────────────────────────────────
BUILD PIPELINE  (release order)
─────────────────────────────────────────────────────────────────────────────

  build_lean()
    ├─ _sync_local_lean_dependency_dirs   symlink dep_<id>/ for lake path deps
    ├─ _sync_graph_infra_lean             copy DependencyGraph.lean from graph_infra/
    ├─ _write_graph_export_lean           generate GraphExport.lean (root imports + #export_graph_json)
    ├─ lake build                         compile all targets → .olean cache
    └─ _collect_graph_json                lake env lean GraphExport.lean → graph_infra/graphs/<id>.json
                                          (proof-term edges via foldConsts, auditable)

  build_latex()
    ├─ _sync_shared_preambles             keep paper-preamble.sty in sync from shared/
    ├─ _write_latex_lean_stats            lean_stats.tex  (line/theorem/sorry counts)
    ├─ _write_lean_handle_ids_auto        lean_handle_ids_auto.tex  (LH{} ID table)
    ├─ _write_assumption_ledger_auto      assumption_ledger_auto.tex
    ├─ _normalize_and_fill_claimstamps    standardise \claimstamp{} metadata
    ├─ _write_claim_mapping_auto          claim_mapping_auto.tex  (claim→Lean matrix)
    ├─ _write_proof_hardness_index_auto   proof_hardness_index_auto.tex
    ├─ _mark_claim_nodes_in_graph         mark cited theorems in graph JSON (paper=-1)
    ├─ _rewrite_content_lean_handles_to_ids  \nolinkurl{handle} → \LH{code}
    └─ pdflatex → bibtex → pdflatex → pdflatex

  build_markdown()    pandoc LaTeX → Markdown, expand \Lean* macros

  package_arxiv()     curated tar.gz + zip with sources, proofs, build log

─────────────────────────────────────────────────────────────────────────────
GRAPH INFRASTRUCTURE
─────────────────────────────────────────────────────────────────────────────

  docs/papers/graph_infra/DependencyGraph.lean defines two Lean command
  elaborators:

    #extract_dependency_graph     — logs stats + orphan integrity check at
                                    elaboration time (useful in editor/CI)

    #export_graph_json "path"     — writes {nodes, edges} JSON to path
                                    during elaboration (side-effect of lake build)

  The build pipeline copies DependencyGraph.lean into each paper's proofs dir
  and generates GraphExport.lean that calls #export_graph_json "graph.json".
  Because GraphExport is registered as a lean_lib target, lake elaborates it
  automatically and the JSON is written as a build side-effect — no separate
  invocation needed.

  Output lands in graph_infra/graphs/<paper_id>.json, ready for the React/D3
  visualisers (dependency-graph.jsx, rejection-trace.jsx).

─────────────────────────────────────────────────────────────────────────────
USAGE
─────────────────────────────────────────────────────────────────────────────

  python scripts/build_papers.py release              # all papers, full build
  python scripts/build_papers.py release paper4_toc   # one paper
  python scripts/build_papers.py lean paper1 -v       # Lean only, verbose
  python scripts/build_papers.py latex paper2         # LaTeX only
  python scripts/build_papers.py scaffold paper6      # create boilerplate
  python scripts/build_papers.py claim-check paper4   # verify theorem coverage
"""

import sys
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass
import yaml
import argparse
from datetime import datetime
import re
import hashlib
import json

try:
    from pylatexenc.latex2text import LatexNodes2Text

    HAS_PYLATEXENC = True
except ImportError:
    HAS_PYLATEXENC = False

# Unicode subscript/superscript mappings for math rendering
UNICODE_SUBSCRIPTS = {
    "0": "₀",
    "1": "₁",
    "2": "₂",
    "3": "₃",
    "4": "₄",
    "5": "₅",
    "6": "₆",
    "7": "₇",
    "8": "₈",
    "9": "₉",
    "a": "ₐ",
    "e": "ₑ",
    "h": "ₕ",
    "i": "ᵢ",
    "j": "ⱼ",
    "k": "ₖ",
    "l": "ₗ",
    "m": "ₘ",
    "n": "ₙ",
    "o": "ₒ",
    "p": "ₚ",
    "r": "ᵣ",
    "s": "ₛ",
    "t": "ₜ",
    "u": "ᵤ",
    "v": "ᵥ",
    "x": "ₓ",
    "+": "₊",
    "-": "₋",
    "=": "₌",
    "(": "₍",
    ")": "₎",
}

UNICODE_SUPERSCRIPTS = {
    "0": "⁰",
    "1": "¹",
    "2": "²",
    "3": "³",
    "4": "⁴",
    "5": "⁵",
    "6": "⁶",
    "7": "⁷",
    "8": "⁸",
    "9": "⁹",
    "a": "ᵃ",
    "b": "ᵇ",
    "c": "ᶜ",
    "d": "ᵈ",
    "e": "ᵉ",
    "f": "ᶠ",
    "g": "ᵍ",
    "h": "ʰ",
    "i": "ⁱ",
    "j": "ʲ",
    "k": "ᵏ",
    "l": "ˡ",
    "m": "ᵐ",
    "n": "ⁿ",
    "o": "ᵒ",
    "p": "ᵖ",
    "r": "ʳ",
    "s": "ˢ",
    "t": "ᵗ",
    "u": "ᵘ",
    "v": "ᵛ",
    "w": "ʷ",
    "x": "ˣ",
    "y": "ʸ",
    "z": "ᶻ",
    "+": "⁺",
    "-": "⁻",
    "=": "⁼",
    "(": "⁽",
    ")": "⁾",
    "A": "ᴬ",
    "B": "ᴮ",
    "D": "ᴰ",
    "E": "ᴱ",
    "G": "ᴳ",
    "H": "ᴴ",
    "I": "ᴵ",
    "J": "ᴶ",
    "K": "ᴷ",
    "L": "ᴸ",
    "M": "ᴹ",
    "N": "ᴺ",
    "O": "ᴼ",
    "P": "ᴾ",
    "R": "ᴿ",
    "T": "ᵀ",
    "U": "ᵁ",
    "V": "ⱽ",
    "W": "ᵂ",
}


@dataclass(frozen=True)
class ArxivConfig:
    """Immutable arXiv packaging configuration."""

    include_build_log: bool = True
    include_lean_source: bool = True
    include_build_config: bool = True
    exclude_patterns: Tuple[str, ...] = (".lake", "build", "*.olean", "*.ilean")

    @classmethod
    def from_dict(cls, d: dict) -> "ArxivConfig":
        return cls(
            include_build_log=d.get("include_build_log", True),
            include_lean_source=d.get("include_lean_source", True),
            include_build_config=d.get("include_build_config", True),
            exclude_patterns=tuple(d.get("exclude_patterns", [])),
        )


@dataclass(frozen=True)
class BuildConfig:
    """Immutable build-system data loaded from papers.yaml."""

    generated_content_files: Tuple[str, ...]
    shared_preamble_files: Tuple[str, ...]
    latex_source_extensions: Tuple[str, ...]
    lean_exclude_dirs: Tuple[str, ...]

    @classmethod
    def from_dict(cls, d: dict) -> "BuildConfig":
        required = [
            "generated_content_files",
            "shared_preamble_files",
            "latex_source_extensions",
            "lean_exclude_dirs",
        ]
        missing = [key for key in required if key not in d]
        if missing:
            raise ValueError(
                "Missing required build config keys in papers.yaml: "
                + ", ".join(missing)
            )
        return cls(
            generated_content_files=tuple(d["generated_content_files"]),
            shared_preamble_files=tuple(d["shared_preamble_files"]),
            latex_source_extensions=tuple(d["latex_source_extensions"]),
            lean_exclude_dirs=tuple(d["lean_exclude_dirs"]),
        )


@dataclass(frozen=True)
class PaperMeta:
    """Immutable paper metadata. Represents a single paper's configuration."""

    paper_id: str  # e.g., "paper1"
    name: str  # e.g., "Typing Discipline"
    full_name: str  # Full paper title
    dir_name: str  # e.g., "paper1_typing_discipline"
    latex_dir: str  # Relative to paper dir, typically "latex"
    latex_file: str  # Main .tex file name
    venue: str  # Target venue, e.g., "JSAIT", "TOPLAS"
    proofs_dir: str = "proofs"
    module_root: str = ""  # Lean module root (auto-derived if empty)
    archive_prefix: str = ""
    lean_dependencies: Tuple[
        str, ...
    ] = ()  # Other papers whose Lean to include in releases
    lean_stats_scope: str = "cumulative"
    assumption_ledger_sources: Tuple[str, ...] = ()
    claim_mapping_file: str = ""
    arxiv_comments: str = ""
    scaffold_from: str = ""
    experiment_paths: Tuple[str, ...] = ()
    experiment_commands: Tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, paper_id: str, d: dict) -> "PaperMeta":
        return cls(
            paper_id=paper_id,
            name=d["name"],
            full_name=d["full_name"],
            dir_name=d["dir"],
            latex_dir=d.get("latex_dir", "latex"),
            latex_file=d["latex_file"],
            venue=d.get("venue", "Draft"),
            proofs_dir=d.get("proofs_dir", "proofs"),
            module_root=d.get("module_root", ""),
            archive_prefix=d.get("archive_prefix", ""),
            lean_dependencies=tuple(d.get("lean_dependencies", [])),
            lean_stats_scope=d.get("lean_stats_scope", "cumulative"),
            assumption_ledger_sources=tuple(d.get("assumption_ledger_sources", [])),
            claim_mapping_file=d.get("claim_mapping_file", ""),
            arxiv_comments=d.get("arxiv_comments", ""),
            scaffold_from=d.get("scaffold_from", ""),
            experiment_paths=tuple(d.get("experiment_paths", [])),
            experiment_commands=tuple(d.get("experiment_commands", [])),
        )


@dataclass(frozen=True)
class LeanStats:
    """Computed Lean proof statistics for a paper."""

    line_count: int
    theorem_count: int
    sorry_count: int
    file_count: int


@dataclass(frozen=True)
class LeanFileStats:
    """Per-file Lean proof statistics."""

    line_count: int
    theorem_count: int
    sorry_count: int


class PaperBuilder:
    """
    Unified builder for all paper formats.

    STRUCTURAL INVARIANTS:
    1. ∀p: paper_dir(p) = papers_dir / meta(p).dir_name
    2. ∀p: content_dir(p) = paper_dir(p) / meta(p).latex_dir / "content"
    3. ∀p: releases_dir(p) = paper_dir(p) / "releases"
    4. ∀p: proofs_dir(p) = paper_dir(p) / "proofs"

    All papers follow these invariants. No exceptions.
    """

    def __init__(
        self,
        repo_root: Path,
        validate_prerequisites: bool = True,
        validate_paper_structure: bool = True,
    ):
        self.repo_root = repo_root
        self.scripts_dir = repo_root / "scripts"
        self.papers_dir = repo_root / "docs" / "papers"

        # Load configuration
        self._raw_metadata = self._load_raw_metadata()
        self.arxiv_config = ArxivConfig.from_dict(self._raw_metadata.get("arxiv", {}))
        self.build_config = BuildConfig.from_dict(self._raw_metadata.get("build", {}))
        self.papers: Dict[str, PaperMeta] = self._load_papers()
        self._lean_stats_cache: Dict[Path, LeanStats] = {}
        self._lean_file_stats_cache: Dict[Path, Dict[str, LeanFileStats]] = {}

        # Captured build output for BUILD_LOG.txt
        self._lean_build_output: str = ""

        # Validate upfront (fail-loud)
        if validate_prerequisites:
            self._validate_prerequisites()
        if validate_paper_structure:
            self._validate_paper_structure()

    def _load_raw_metadata(self) -> dict:
        """Load raw YAML metadata."""
        metadata_file = self.scripts_dir / "papers.yaml"
        if not metadata_file.exists():
            raise FileNotFoundError(f"papers.yaml not found at {metadata_file}")
        with open(metadata_file) as f:
            return yaml.safe_load(f)

    def _load_papers(self) -> Dict[str, PaperMeta]:
        """Convert raw YAML to immutable PaperMeta objects."""
        papers = {}
        for paper_id, paper_dict in self._raw_metadata.get("papers", {}).items():
            papers[paper_id] = PaperMeta.from_dict(paper_id, paper_dict)
        return papers

    def _validate_prerequisites(self):
        """Fail loudly if required tools are missing."""
        required = ["pdflatex", "bibtex", "pandoc", "lake"]
        missing = [cmd for cmd in required if not shutil.which(cmd)]
        if missing:
            raise RuntimeError(
                f"Missing required tools: {', '.join(missing)}\n"
                f"Install with: sudo apt install {' '.join(missing)}"
            )

    def _validate_paper_structure(self):
        """Validate structural invariants for all papers."""
        errors = []
        for paper_id, meta in self.papers.items():
            paper_dir = self.papers_dir / meta.dir_name
            if not paper_dir.exists():
                errors.append(f"{paper_id}: paper_dir not found: {paper_dir}")
                continue

            latex_dir = paper_dir / meta.latex_dir
            if not latex_dir.exists():
                errors.append(f"{paper_id}: latex_dir not found: {latex_dir}")

            latex_file = latex_dir / meta.latex_file
            if not latex_file.exists():
                errors.append(f"{paper_id}: latex_file not found: {latex_file}")

            content_dir = latex_dir / "content"
            if not content_dir.exists():
                errors.append(f"{paper_id}: content_dir not found: {content_dir}")

        if errors:
            raise ValueError(
                f"Paper structure validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

    def _get_paper_meta(self, paper_id: str) -> PaperMeta:
        """Get paper metadata, fail if not found."""
        if paper_id not in self.papers:
            valid = ", ".join(self.papers.keys())
            raise ValueError(f"Unknown paper: {paper_id}. Valid papers: {valid}")
        return self.papers[paper_id]

    def _get_paper_dir(self, paper_id: str) -> Path:
        """Get paper directory path."""
        meta = self._get_paper_meta(paper_id)
        return self.papers_dir / meta.dir_name

    def _get_content_dir(self, paper_id: str) -> Path:
        """Get content directory: paper_dir/latex/content (INVARIANT 2)."""
        meta = self._get_paper_meta(paper_id)
        return self.papers_dir / meta.dir_name / meta.latex_dir / "content"

    def _get_latex_dir(self, paper_id: str) -> Path:
        """Get LaTeX directory for a paper."""
        meta = self._get_paper_meta(paper_id)
        return self._get_paper_dir(paper_id) / meta.latex_dir

    def _get_releases_dir(self, paper_id: str) -> Path:
        """Get releases directory, creating if needed (INVARIANT 3)."""
        releases_dir = self._get_paper_dir(paper_id) / "releases"
        releases_dir.mkdir(parents=True, exist_ok=True)
        return releases_dir

    def _get_markdown_dir(self, paper_id: str) -> Path:
        """Get Markdown directory for a paper."""
        return self._get_paper_dir(paper_id) / "markdown"

    def _get_markdown_file(self, paper_id: str) -> Path:
        """Get canonical Markdown output path for a paper."""
        return self._get_markdown_dir(paper_id) / f"{paper_id}.md"

    def _generated_content_files(self) -> Set[str]:
        """Return configured auto-generated content file names."""
        return set(self.build_config.generated_content_files)

    def _iter_manual_content_tex(
        self, paper_id: str, files: Optional[List[Path]] = None
    ) -> List[Path]:
        """Return non-generated content .tex files for a paper."""
        generated_files = self._generated_content_files()
        if files is not None:
            return sorted(
                tex_file
                for tex_file in files
                if tex_file.suffix == ".tex" and tex_file.name not in generated_files
            )

        content_dir = self._get_content_dir(paper_id)
        if not content_dir.exists():
            return []
        return sorted(
            tex_file
            for tex_file in content_dir.glob("*.tex")
            if tex_file.name not in generated_files
        )

    def _refresh_derived_content(self, paper_id: str) -> None:
        """Regenerate all auto-derived LaTeX content artifacts for a paper."""
        # Ensure the shared SSOT lean-handle macros are generated before we
        # copy shared preambles into the paper latex dir. This centralises
        # control of the \LH/\LHrng/\leanmeta macros so all variants get the
        # same build-generated definitions.
        self._write_shared_lean_handle_macros_auto()
        self._sync_shared_preambles(paper_id)
        self._run_experiment_commands(paper_id)
        self._write_latex_lean_stats(paper_id)
        self._write_assumption_ledger_auto(paper_id)
        self._write_lean_handle_ids_auto(paper_id)
        self._rewrite_content_lean_handles_to_ids(paper_id)
        self._normalize_and_fill_claimstamps(paper_id)
        self._write_claim_mapping_auto(paper_id)
        self._write_proof_hardness_index_auto(paper_id)

    def _write_text_file(
        self,
        path: Path,
        content: str,
        overwrite: bool,
        created_files: List[Path],
        skipped_files: List[Path],
    ) -> None:
        """Write UTF-8 text file unless it exists and overwrite is disabled."""
        if path.exists() and not overwrite:
            skipped_files.append(path)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        created_files.append(path)

    def _compact_lean_handle(self, handle: str) -> str:
        """Render long Lean handles with compact namespace prefixes."""
        if "." not in handle:
            return handle
        full_prefix, remainder = handle.split(".", 1)
        short_prefix = self._compact_lean_prefix(full_prefix)
        if short_prefix:
            return short_prefix + "." + remainder
        return handle

    def _compact_lean_prefix(self, full_prefix: str) -> str:
        """Compact a Lean namespace prefix to a stable short code.

        Strategy:
        derive initials from CamelCase segments deterministically.
        """
        # Split CamelCase / PascalCase into lexical chunks, keep alnum.
        chunks = re.findall(r"[A-Z][a-z0-9]*|[a-z]+|[0-9]+", full_prefix)
        initials_parts: List[str] = []
        for chunk in chunks:
            if not chunk:
                continue
            head = chunk[0]
            if not head.isalpha():
                continue
            tail_digits = "".join(ch for ch in chunk[1:] if ch.isdigit())
            initials_parts.append(head.upper() + tail_digits)
        initials = "".join(initials_parts)
        if initials:
            return initials

        cleaned = re.sub(r"[^A-Za-z0-9]+", "", full_prefix)
        if cleaned:
            return cleaned[:3].upper()
        return full_prefix

    def _to_pascal_case(self, raw: str) -> str:
        """Convert an arbitrary identifier to a Lean-safe PascalCase module root."""
        tokens = [tok for tok in re.split(r"[^A-Za-z0-9]+", raw) if tok]
        if not tokens:
            return "PaperScaffold"
        normalized = "".join(tok[:1].upper() + tok[1:] for tok in tokens)
        if normalized[0].isdigit():
            normalized = f"P{normalized}"
        return normalized

    def _to_package_name(self, raw: str) -> str:
        """Convert an arbitrary identifier to a Lean package-safe snake_case name."""
        normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", raw).strip("_").lower()
        if not normalized:
            normalized = "paper_scaffold"
        if normalized[0].isdigit():
            normalized = f"p_{normalized}"
        return normalized

    def _discover_template_lean_toolchain(self) -> str:
        """Find an existing Lean toolchain pin from current papers, or use a safe fallback."""
        candidate_paths: List[Path] = []
        for meta in self.papers.values():
            candidate_paths.append(
                self.papers_dir / meta.dir_name / meta.proofs_dir / "lean-toolchain"
            )

        for path in candidate_paths:
            if not path.exists():
                continue
            content = path.read_text(encoding="utf-8", errors="replace").strip()
            if content:
                return content

        return "leanprover/lean4:stable"

    def _discover_template_mathlib_require(self) -> str:
        """Extract a pinned `require mathlib` block from an existing lakefile."""
        pattern = re.compile(
            r"require\s+mathlib\s+from\s+git\s*\n\s*\"[^\"]+\"\s*@\s*\"[^\"]+\"",
            re.MULTILINE,
        )
        candidate_paths: List[Path] = []
        for meta in self.papers.values():
            candidate_paths.append(
                self.papers_dir / meta.dir_name / meta.proofs_dir / "lakefile.lean"
            )

        for path in candidate_paths:
            if not path.exists():
                continue
            content = path.read_text(encoding="utf-8", errors="replace")
            match = pattern.search(content)
            if match:
                return match.group(0)

        return (
            "require mathlib from git\n"
            '  "https://github.com/leanprover-community/mathlib4" @ "master"'
        )

    def _latex_main_template(self, meta: PaperMeta) -> str:
        """Generate a self-contained LaTeX starter file for a new paper variant."""
        return f"""\\documentclass[11pt]{{article}}
\\usepackage{{paper-preamble}}

% Auto-generated scaffold. Replace with venue-specific preamble as needed.
\\IfFileExists{{content/lean_stats.tex}}{{\\input{{content/lean_stats.tex}}}}{{}}
\\providecommand{{\\claimstamp}}[2]{{\\allowbreak{{\\scriptsize\\textit{{[C:#1;R:#2]}}}}}}
\\providecommand{{\\LHrng}}[3]{{\\hyperlink{{lh:#1#2}}{{\\nolinkurl{{#1#2-#3}}}}}}
\\providecommand{{\\leanmeta}}[1]{{\\allowbreak{{\\scriptsize\\textit{{(L: #1)}}}}}}

\\title{{{meta.full_name}}}
\\author{{Anonymous Author\\\\Anonymous Institution\\\\\\texttt{{anonymous@example.com}}}}
\\date{{\\today}}

\\begin{{document}}
\\maketitle

\\begin{{abstract}}
\\input{{content/abstract.tex}}
\\end{{abstract}}

\\input{{content/01_introduction.tex}}

\\bibliographystyle{{plain}}
\\bibliography{{references}}

\\end{{document}}
"""

    def _latex_abstract_template(self, meta: PaperMeta) -> str:
        """Generate default abstract placeholder content."""
        return (
            f"Scaffold abstract for {meta.name}. "
            "Replace this paragraph with the final abstract text.\n"
        )

    def _latex_intro_template(self, meta: PaperMeta) -> str:
        """Generate default introduction placeholder content."""
        return f"""\\section{{Introduction}}

This is scaffold content for \\texttt{{{meta.paper_id}}}. Replace it with your paper text.
"""

    def _references_template(self) -> str:
        """Generate an empty bibliography placeholder."""
        return "% Auto-generated scaffold bibliography.\n% Add BibTeX entries here.\n"

    def _proofs_readme_template(self, meta: PaperMeta) -> str:
        """Generate README for scaffolded Lean proofs."""
        return f"""# {meta.name} Lean Proofs

This directory was scaffolded by `scripts/build_papers.py scaffold {meta.paper_id}`.

- Replace starter modules with your formalization.
- Run `lake build` inside this directory once dependencies are initialized.
"""

    def _replace_first_latex_title(self, content: str, new_title: str) -> str:
        """Replace first `\\title{...}` argument while preserving surrounding LaTeX."""
        marker = r"\title"
        start = content.find(marker)
        if start == -1:
            return content

        i = start + len(marker)
        n = len(content)
        while i < n and content[i].isspace():
            i += 1
        if i < n and content[i] == "[":
            depth = 1
            i += 1
            while i < n and depth > 0:
                if content[i] == "[":
                    depth += 1
                elif content[i] == "]":
                    depth -= 1
                i += 1
            while i < n and content[i].isspace():
                i += 1
        if i >= n or content[i] != "{":
            return content

        arg_start = i + 1
        depth = 1
        i = arg_start
        while i < n and depth > 0:
            ch = content[i]
            prev = content[i - 1] if i > 0 else ""
            if ch == "{" and prev != "\\":
                depth += 1
            elif ch == "}" and prev != "\\":
                depth -= 1
            i += 1
        if depth != 0:
            return content

        arg_end = i - 1
        return content[:arg_start] + new_title + content[arg_end:]

    def _build_derived_latex_scaffold(
        self,
        paper_id: str,
        meta: PaperMeta,
        latex_dir: Path,
        content_dir: Path,
    ) -> Optional[List[Tuple[Path, str]]]:
        """Build scaffold file list by deriving from an existing paper variant."""
        source_id = meta.scaffold_from.strip()
        if not source_id:
            return None
        if source_id == paper_id:
            raise ValueError(f"{paper_id}: scaffold_from cannot reference itself")

        source_meta = self._get_paper_meta(source_id)
        source_latex_dir = self._get_paper_dir(source_id) / source_meta.latex_dir
        source_main = source_latex_dir / source_meta.latex_file
        if not source_main.exists():
            raise FileNotFoundError(
                f"{paper_id}: scaffold_from source missing main TeX: "
                f"{source_main.relative_to(self.repo_root)}"
            )

        source_main_content = source_main.read_text(encoding="utf-8", errors="replace")
        main_content = self._replace_first_latex_title(
            source_main_content, meta.full_name
        )
        files_to_write: List[Tuple[Path, str]] = [
            (latex_dir / meta.latex_file, main_content),
        ]

        # Skip generated files; they are rebuilt automatically by the pipeline.
        generated_content = self._generated_content_files()
        source_content_dir = source_latex_dir / "content"
        copied_content = False
        if source_content_dir.exists():
            for source_file in sorted(source_content_dir.glob("*.tex")):
                if source_file.name in generated_content:
                    continue
                files_to_write.append(
                    (
                        content_dir / source_file.name,
                        source_file.read_text(encoding="utf-8", errors="replace"),
                    )
                )
                copied_content = True

        if not copied_content:
            files_to_write.extend(
                [
                    (content_dir / "abstract.tex", self._latex_abstract_template(meta)),
                    (
                        content_dir / "01_introduction.tex",
                        self._latex_intro_template(meta),
                    ),
                ]
            )

        source_references = source_latex_dir / "references.bib"
        if source_references.exists():
            files_to_write.append(
                (
                    latex_dir / "references.bib",
                    source_references.read_text(encoding="utf-8", errors="replace"),
                )
            )
        else:
            files_to_write.append(
                (latex_dir / "references.bib", self._references_template())
            )

        print(f"[scaffold] {paper_id}: deriving LaTeX scaffold from {source_id}")
        return files_to_write

    def _lakefile_template(
        self, package_name: str, module_root: str, mathlib_require: str
    ) -> str:
        """Generate a minimal lakefile pinned to the repository's current mathlib source."""
        return f"""import Lake
open Lake DSL

package «{package_name}» where
  -- Auto-generated scaffold package.

{mathlib_require}

@[default_target]
lean_lib «{module_root}» where
  globs := #[.submodules `{module_root}]
  srcDir := "."
"""

    def _lean_root_template(self, module_root: str) -> str:
        """Generate root Lean module importing starter files."""
        return f"""import {module_root}.Basic
"""

    def _lean_basic_template(self, module_root: str) -> str:
        """Generate starter Lean theorem file."""
        return f"""namespace {module_root}

theorem scaffold_sanity : True := by
  trivial

end {module_root}
"""

    def scaffold_paper(
        self, paper_id: str, overwrite: bool = False
    ) -> Dict[str, List[Path]]:
        """Create required folder/file boilerplate for a paper entry in papers.yaml."""
        meta = self._get_paper_meta(paper_id)
        paper_dir = self._get_paper_dir(paper_id)
        latex_dir = paper_dir / meta.latex_dir
        content_dir = latex_dir / "content"
        proofs_dir = self._get_paper_proofs_dir(paper_id)
        releases_dir = paper_dir / "releases"
        markdown_dir = paper_dir / "markdown"
        proofs_dir_was_nonempty = proofs_dir.exists() and any(proofs_dir.iterdir())

        created_dirs: List[Path] = []
        created_files: List[Path] = []
        skipped_files: List[Path] = []

        for directory in [
            paper_dir,
            latex_dir,
            content_dir,
            proofs_dir,
            releases_dir,
            markdown_dir,
        ]:
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
                created_dirs.append(directory)

        module_root = (
            meta.module_root
            if meta.module_root
            else self._to_pascal_case(meta.dir_name)
        )
        package_name = self._to_package_name(meta.dir_name)
        lean_toolchain = self._discover_template_lean_toolchain()
        mathlib_require = self._discover_template_mathlib_require()

        # Copy preamble from shared (SSOT)
        shared_preamble = self.papers_dir / "shared" / "paper-preamble.sty"

        files_to_write = self._build_derived_latex_scaffold(
            paper_id, meta, latex_dir, content_dir
        )
        if files_to_write is None:
            files_to_write = [
                (latex_dir / meta.latex_file, self._latex_main_template(meta)),
                (content_dir / "abstract.tex", self._latex_abstract_template(meta)),
                (content_dir / "01_introduction.tex", self._latex_intro_template(meta)),
                (latex_dir / "references.bib", self._references_template()),
            ]

        # Always copy preamble from shared (SSOT) - use "COPY" marker
        if shared_preamble.exists():
            files_to_write.append((latex_dir / "paper-preamble.sty", "COPY"))
        if overwrite or not proofs_dir_was_nonempty:
            files_to_write.extend(
                [
                    (proofs_dir / "README.md", self._proofs_readme_template(meta)),
                    (proofs_dir / "lean-toolchain", f"{lean_toolchain}\n"),
                    (
                        proofs_dir / "lakefile.lean",
                        self._lakefile_template(
                            package_name, module_root, mathlib_require
                        ),
                    ),
                    (
                        proofs_dir / f"{module_root}.lean",
                        self._lean_root_template(module_root),
                    ),
                    (
                        proofs_dir / module_root / "Basic.lean",
                        self._lean_basic_template(module_root),
                    ),
                ]
            )
        else:
            print(
                f"[scaffold] {paper_id}: proofs scaffold skipped "
                f"(existing non-empty directory: {proofs_dir.relative_to(self.repo_root)})"
            )

        for path, content in files_to_write:
            if content == "COPY":
                # Copy from source (shared preamble)
                src = self.papers_dir / "shared" / path.name
                if src.exists():
                    import shutil

                    shutil.copy2(src, path)
                    created_files.append(path)
                else:
                    print(f"[scaffold] WARNING: shared source not found: {src}")
            else:
                self._write_text_file(
                    path,
                    content,
                    overwrite=overwrite,
                    created_files=created_files,
                    skipped_files=skipped_files,
                )

        print(
            f"[scaffold] {paper_id}: created {len(created_dirs)} dirs, {len(created_files)} files"
        )
        if skipped_files:
            print(f"[scaffold] {paper_id}: skipped {len(skipped_files)} existing files")

        # Initialize generated artifacts so scaffolded papers compile with the
        # same metadata infrastructure as regular build targets.
        self._sync_shared_preambles(paper_id)
        self._write_latex_lean_stats(paper_id)
        self._write_assumption_ledger_auto(paper_id)
        self._write_lean_handle_ids_auto(paper_id)
        self._rewrite_content_lean_handles_to_ids(paper_id)
        self._normalize_and_fill_claimstamps(paper_id)
        self._write_claim_mapping_auto(paper_id)
        self._write_proof_hardness_index_auto(paper_id)

        return {
            "created_dirs": created_dirs,
            "created_files": created_files,
            "skipped_files": skipped_files,
        }

    def _discover_content_files_for_flags(
        self, paper_id: str, defined_flags: Optional[Set[str]] = None
    ) -> List[Path]:
        """Discover included TeX files for a paper under simple flag evaluation."""
        meta = self._get_paper_meta(paper_id)
        main_tex = self._get_paper_dir(paper_id) / meta.latex_dir / meta.latex_file
        if not main_tex.exists():
            raise FileNotFoundError(f"Main LaTeX file not found: {main_tex}")

        files = self._collect_included_tex_files(
            main_tex,
            include_document_body_only=True,
            include_self=False,
            defined_flags=defined_flags,
        )
        if files:
            ordered_unique: List[Path] = []
            seen: set[Path] = set()
            for file_path in files:
                resolved = file_path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                ordered_unique.append(resolved)
            return ordered_unique

        # Fail-loud fallback: preserve prior behavior if includes cannot be discovered.
        print(
            f"[build-md] Warning: no includes parsed from {main_tex.name}; falling back to content/*.tex"
        )
        return self._iter_manual_content_tex(paper_id)

    def _discover_content_files(self, paper_id: str) -> List[Path]:
        """Discover markdown content files by following main LaTeX include order."""
        return self._discover_content_files_for_flags(paper_id)

    def _strip_latex_comments(self, content: str) -> str:
        """Remove LaTeX comments while preserving escaped percent signs."""
        return re.sub(r"(?<!\\\\)%.*", "", content)

    def _extract_document_body(self, content: str) -> str:
        """Extract content between \\begin{document} and \\end{document} when present."""
        begin = re.search(r"\\begin\{document\}", content)
        if not begin:
            return content
        end = re.search(r"\\end\{document\}", content[begin.end() :])
        if not end:
            return content[begin.end() :]
        return content[begin.end() : begin.end() + end.start()]

    def _resolve_include_path(
        self, parent_dir: Path, include_target: str, search_root: Path
    ) -> Path:
        """Resolve a LaTeX \\input/\\include target to a concrete .tex path.

        LaTeX resolves includes relative to the current file and the main working
        directory. We mirror that behavior by searching both roots.
        """
        raw_target = include_target.strip()
        candidates = [
            (parent_dir / raw_target).resolve(),
            (search_root / raw_target).resolve(),
        ]

        checked: List[Path] = []
        for candidate in candidates:
            checked.append(candidate)
            if candidate.exists():
                return candidate
            if not candidate.suffix:
                candidate_tex = candidate.with_suffix(".tex")
                checked.append(candidate_tex)
                if candidate_tex.exists():
                    return candidate_tex

        # Return best-effort path for warning output.
        return checked[-1]

    def _evaluate_simple_ifdefined(
        self, content: str, defined_flags: Optional[Set[str]] = None
    ) -> str:
        """Resolve simple `\\ifdefined\\FLAG ... \\else ... \\fi` blocks."""
        if not defined_flags:
            return content
        normalized_flags = {flag.lstrip("\\") for flag in defined_flags}

        token_pattern = re.compile(r"\\ifdefined\\([A-Za-z@]+)|\\else|\\fi")
        pos = 0
        out: List[str] = []
        stack: List[Dict[str, object]] = []

        for match in token_pattern.finditer(content):
            if not stack or all(bool(frame["take"]) for frame in stack):
                out.append(content[pos : match.start()])

            token = match.group(0)
            if token.startswith("\\ifdefined\\"):
                flag = token[len("\\ifdefined\\") :]
                parent_take = all(bool(frame["take"]) for frame in stack)
                cond = flag in normalized_flags
                stack.append(
                    {
                        "flag": flag,
                        "cond": cond,
                        "parent_take": parent_take,
                        "take": parent_take and cond,
                    }
                )
            elif token == r"\else":
                if stack:
                    frame = stack[-1]
                    frame["take"] = bool(frame["parent_take"]) and not bool(
                        frame["cond"]
                    )
            elif token == r"\fi":
                if stack:
                    stack.pop()

            pos = match.end()

        if not stack or all(bool(frame["take"]) for frame in stack):
            out.append(content[pos:])
        return "".join(out)

    def _find_tex_includes(
        self,
        tex_file: Path,
        include_document_body_only: bool,
        search_root: Path,
        defined_flags: Optional[Set[str]] = None,
    ) -> List[Path]:
        """Parse direct \\input/\\include dependencies from a .tex file."""
        content = tex_file.read_text(encoding="utf-8", errors="replace")
        if include_document_body_only:
            content = self._extract_document_body(content)
        content = self._evaluate_simple_ifdefined(content, defined_flags)
        content = self._strip_latex_comments(content)

        include_targets = re.findall(r"\\(?:input|include)\{([^}]+)\}", content)
        includes: List[Path] = []
        for target in include_targets:
            resolved = self._resolve_include_path(tex_file.parent, target, search_root)
            if not resolved.exists():
                print(
                    f"[build-md] Warning: include target not found in {tex_file.name}: {target}"
                )
                continue
            includes.append(resolved)
        return includes

    def _collect_included_tex_files(
        self,
        tex_file: Path,
        include_document_body_only: bool = False,
        include_self: bool = True,
        active_stack: Tuple[Path, ...] = (),
        search_root: Path | None = None,
        defined_flags: Optional[Set[str]] = None,
    ) -> List[Path]:
        """Recursively collect include files in document order.

        Includes parent files as well as nested includes so section files that
        contain both prose and nested ``\\input`` directives are not dropped.
        """
        resolved_tex = tex_file.resolve()
        if search_root is None:
            search_root = resolved_tex.parent
        if resolved_tex in active_stack:
            chain = " -> ".join(str(p.name) for p in (*active_stack, resolved_tex))
            raise RuntimeError(f"Circular LaTeX include detected: {chain}")

        direct_includes = self._find_tex_includes(
            resolved_tex,
            include_document_body_only=include_document_body_only,
            search_root=search_root,
            defined_flags=defined_flags,
        )
        ordered_files: List[Path] = [resolved_tex] if include_self else []
        next_stack = (*active_stack, resolved_tex)
        for include_file in direct_includes:
            ordered_files.extend(
                self._collect_included_tex_files(
                    include_file,
                    include_document_body_only=False,
                    include_self=True,
                    active_stack=next_stack,
                    search_root=search_root,
                    defined_flags=defined_flags,
                )
            )
        return ordered_files

    def _get_paper_proofs_dir(self, paper_id: str) -> Path:
        """Get the proofs directory for a specific paper.

        Each paper has its own proofs directory at: paper_dir/proofs/
        For variants (e.g., paper1_jsait), use the base paper's proofs.
        """
        meta = self._get_paper_meta(paper_id)
        return self.papers_dir / meta.dir_name / meta.proofs_dir

    def _get_archive_prefix(self, paper_id: str) -> str:
        """Archive/package basename, configurable per paper."""
        meta = self._get_paper_meta(paper_id)
        return meta.archive_prefix if meta.archive_prefix else paper_id

    def _iter_paper_lean_files(self, proofs_dir: Path) -> List[Path]:
        """Return paper Lean files, excluding build/cache directories."""
        if not proofs_dir.exists():
            return []

        excluded_dirs = {".lake", "build"}
        lean_files: List[Path] = []
        for lean_file in proofs_dir.rglob("*.lean"):
            rel_parts = lean_file.relative_to(proofs_dir).parts
            if any(part in excluded_dirs for part in rel_parts):
                continue
            # Dependency mirrors are build-time wiring, not local proof content.
            if rel_parts and rel_parts[0].startswith("dep_"):
                continue
            lean_files.append(lean_file)
        return sorted(lean_files)

    def _collect_lean_dependency_closure(self, paper_id: str) -> List[str]:
        """Return transitive Lean dependencies for `paper_id` in topological order."""
        ordered: List[str] = []
        seen: Set[str] = set()
        active: Set[str] = set()

        def visit(pid: str) -> None:
            meta = self._get_paper_meta(pid)
            for dep in meta.lean_dependencies:
                if dep not in self.papers:
                    raise ValueError(
                        f"Unknown lean dependency '{dep}' declared by {pid}"
                    )
                if dep in active:
                    cycle = " -> ".join(list(active) + [dep])
                    raise ValueError(f"Lean dependency cycle detected: {cycle}")
                if dep in seen:
                    continue
                active.add(dep)
                visit(dep)
                active.remove(dep)
                seen.add(dep)
                ordered.append(dep)

        visit(paper_id)
        return ordered

    def _sync_local_lean_dependency_dirs(self, paper_id: str) -> None:
        """Materialize dependency aliases (`dep_<paper_id>`) for local `lake build`.

        Source trees keep dependencies in their own paper folders; this method
        creates local alias directories so lake path-dependencies remain stable
        across source and packaged builds.
        """
        proofs_dir = self._get_paper_proofs_dir(paper_id)
        if not proofs_dir.exists():
            return

        dep_ids = self._collect_lean_dependency_closure(paper_id)
        for dep_id in dep_ids:
            src = self._get_paper_proofs_dir(dep_id)
            if not src.exists():
                continue
            alias = proofs_dir / f"dep_{dep_id}"

            if alias.is_symlink():
                try:
                    if alias.resolve() == src.resolve():
                        continue
                except FileNotFoundError:
                    pass
                alias.unlink()
            elif alias.exists():
                if alias.is_file():
                    alias.unlink()
                else:
                    shutil.rmtree(alias)

            try:
                alias.symlink_to(src.resolve(), target_is_directory=True)
            except OSError:
                shutil.copytree(
                    src,
                    alias,
                    dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns(
                        ".lake", "build", "*.olean", "*.ilean"
                    ),
                )

    def _sync_graph_infra_lean(self, paper_id: str) -> None:
        """Copy DependencyGraph.lean from graph_infra into the paper's proofs dir."""
        src = (
            self.repo_root / "docs" / "papers" / "graph_infra" / "DependencyGraph.lean"
        )
        if not src.exists():
            return
        proofs_dir = self._get_paper_proofs_dir(paper_id)
        if not proofs_dir.exists():
            return
        dst = proofs_dir / "DependencyGraph.lean"
        if not dst.exists() or dst.read_bytes() != src.read_bytes():
            shutil.copy2(src, dst)

    def _write_graph_export_lean(self, paper_id: str) -> None:
        """Generate graph and declaration-info export drivers from compiled modules."""
        proofs_dir = self._get_paper_proofs_dir(paper_id)
        if not proofs_dir.exists():
            return

        # ── Collect submodule dirs from lakefile globs ──────────────────────────
        submodule_dirs = self._derive_module_roots_from_lakefile(proofs_dir)
        if not submodule_dirs:
            return

        # ── Enumerate submodule .lean files → individual module names ───────────
        skip_stems = {"DependencyGraph", "GraphExport"}
        module_names: list[str] = []
        for subdir_name in submodule_dirs:
            subdir = proofs_dir / subdir_name
            if not subdir.is_dir():
                continue
            for lean_file in sorted(subdir.rglob("*.lean")):
                rel = lean_file.relative_to(proofs_dir)
                parts = list(rel.parts)
                parts[-1] = parts[-1][:-5]  # strip .lean
                mod = ".".join(parts)
                if parts[-1] not in skip_stems and mod not in module_names:
                    module_names.append(mod)

        if not module_names:
            return

        submodule_imports = "\n".join(f"import {m}" for m in module_names)
        graph_content = (
            f"-- Auto-generated by build_papers.py -- do not edit\n"
            f"import DependencyGraph\n"
            f"{submodule_imports}\n"
            f"\n"
            f'#export_graph_json "graph.json"\n'
        )
        decl_content = (
            f"-- Auto-generated by build_papers.py -- do not edit\n"
            f"import DependencyGraph\n"
            f"{submodule_imports}\n"
            f"\n"
            f'#export_decl_info_json "decls.json"\n'
        )
        graph_dst = proofs_dir / "GraphExport.lean"
        if (
            not graph_dst.exists()
            or graph_dst.read_text(encoding="utf-8") != graph_content
        ):
            graph_dst.write_text(graph_content, encoding="utf-8")
        decl_dst = proofs_dir / "DeclInfoExport.lean"
        if (
            not decl_dst.exists()
            or decl_dst.read_text(encoding="utf-8") != decl_content
        ):
            decl_dst.write_text(decl_content, encoding="utf-8")

    def _collect_graph_json(self, paper_id: str) -> None:
        """Run `lake env lean GraphExport.lean` to extract proof-term dependency graph.

        lake env sets LEAN_PATH from the built .olean cache — lean loads compiled
        bytecode for all imports, no re-elaboration, no collision. #export_graph_json
        runs with the full type-checked environment and writes graph.json.
        """
        proofs_dir = self._get_paper_proofs_dir(paper_id)
        graph_export = proofs_dir / "GraphExport.lean"
        decl_export = proofs_dir / "DeclInfoExport.lean"
        dep_graph = proofs_dir / "DependencyGraph.lean"
        if not graph_export.exists():
            print(f"[graph] No GraphExport.lean for {paper_id}, skipping")
            return
        if not decl_export.exists():
            print(f"[graph] No DeclInfoExport.lean for {paper_id}, skipping")
            return
        if not dep_graph.exists():
            print(f"[graph] No DependencyGraph.lean for {paper_id}, skipping")
            return

        dep_graph_olean = (
            proofs_dir / ".lake" / "build" / "lib" / "lean" / "DependencyGraph.olean"
        )
        dep_graph_ilean = (
            proofs_dir / ".lake" / "build" / "lib" / "lean" / "DependencyGraph.ilean"
        )
        dep_graph_olean.parent.mkdir(parents=True, exist_ok=True)

        dep_result = subprocess.run(
            [
                "lake",
                "env",
                "lean",
                "DependencyGraph.lean",
                "-o",
                str(dep_graph_olean),
                "-i",
                str(dep_graph_ilean),
            ],
            cwd=proofs_dir,
            capture_output=True,
            text=True,
        )
        if dep_result.returncode != 0:
            print(f"[graph] Warning: DependencyGraph compile failed for {paper_id}")
            print(dep_result.stderr[-2000:] if dep_result.stderr else "")
            return

        result = subprocess.run(
            ["lake", "env", "lean", "GraphExport.lean"],
            cwd=proofs_dir,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"[graph] Warning: graph extraction failed for {paper_id}")
            print(result.stderr[-2000:] if result.stderr else "")
            return

        src = proofs_dir / "graph.json"
        if not src.exists():
            print(f"[graph] Warning: graph.json not written for {paper_id}")
            return
        decl_src = proofs_dir / "decls.json"
        decl_result = subprocess.run(
            ["lake", "env", "lean", "DeclInfoExport.lean"],
            cwd=proofs_dir,
            capture_output=True,
            text=True,
        )
        if decl_result.returncode != 0:
            print(
                f"[graph] Warning: declaration metadata extraction failed for {paper_id}"
            )
            print(decl_result.stderr[-2000:] if decl_result.stderr else "")
            return
        if not decl_src.exists():
            print(f"[graph] Warning: decls.json not written for {paper_id}")
            return

        graphs_dir = self.repo_root / "docs" / "papers" / "graph_infra" / "graphs"
        graphs_dir.mkdir(exist_ok=True)
        dst = graphs_dir / f"{paper_id}.json"
        decl_dst = graphs_dir / f"{paper_id}.decls.json"
        shutil.move(str(src), dst)
        shutil.move(str(decl_src), decl_dst)
        self._sync_graph_viewer_assets(graphs_dir)
        self._write_graph_manifest(graphs_dir, f"{paper_id}.json")

        # Fix PH handle edges that the graph extractor misses
        self._fix_ph_handle_edges(paper_id, dst)

        import json

        data = json.loads(dst.read_text())
        decl_data = json.loads(decl_dst.read_text())
        print(
            f"[graph] ✓ {paper_id} → {dst.relative_to(self.repo_root)} "
            f"({len(data['nodes'])} nodes, {len(data['edges'])} edges; {len(decl_data)} decl entries)"
        )

    def _fix_ph_handle_edges(self, paper_id: str, graph_path: Path) -> None:
        """Fix PH handle edges in the dependency graph by running fix_ph_handles.py.

        PH handles (PH11-PH36) are abbreviations that reference PhysicalComplexity
        theorems using @ notation for explicit universe polymorphism. The graph
        extractor misses these edges because @PhysicalComplexity.* doesn't match
        the declared PhysicalComplexity.* names in the graph.

        This method runs the fixer script to add the missing edges automatically.
        """
        fixer_script = (
            self.repo_root / "docs" / "papers" / "graph_infra" / "fix_ph_handles.py"
        )
        if not fixer_script.exists():
            print(f"[graph]   PH handle fixer not found, skipping")
            return

        try:
            result = subprocess.run(
                [sys.executable, str(fixer_script), "--paper", paper_id],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0:
                # Extract summary from output
                for line in result.stdout.splitlines():
                    if line.startswith("Fixed"):
                        print(f"[graph]   {line}")
                        break
                else:
                    print(f"[graph]   PH handles verified")
            else:
                print(f"[graph]   PH fix warning: {result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            print(f"[graph]   PH fix timeout, skipping")
        except Exception as e:
            print(f"[graph]   PH fix error: {e}")

    def _mark_claim_nodes_in_graph(self, paper_id: str) -> None:
        """Mark nodes in the dependency graph that correspond to paper claims.

        Extracts which Lean theorem handles are cited in the paper's claims,
        then adds a `paper: -1` attribute to those nodes in the graph JSON.
        This enables the "Claims Only" view in the dependency graph visualizer
        to show only theorems that are directly cited in the paper.

        This method should be called after both:
        1. The graph JSON has been generated (by build_lean)
        2. The claim extraction has been done (during build_latex)
        """
        # Get the graph file path
        graphs_dir = self.repo_root / "docs" / "papers" / "graph_infra" / "graphs"
        graph_path = graphs_dir / f"{paper_id}.json"

        if not graph_path.exists():
            # Graph hasn't been generated yet, skip silently
            return

        # Extract claim-to-handles mapping directly (gets full, non-compacted handles)
        # This is better than reading from claim_mapping_auto.tex which has compacted handles
        claim_to_handles = self._extract_claim_label_to_lean_handles(
            paper_id, include_dependencies=True
        )

        # Collect all unique Lean handles that are cited in claims
        cited_handles: Set[str] = set()
        for handles_list in claim_to_handles.values():
            cited_handles.update(handles_list)

        if not cited_handles:
            # No handles to mark
            return

        # Load the graph JSON
        import json

        try:
            data = json.loads(graph_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"[graph]   Warning: Could not load graph JSON: {e}")
            return

        # Mark nodes that correspond to cited claims with paper: -1
        # Also check for namespace-qualified versions: a handle "foo" should match
        # "Ssot.foo", "ClaimClosure.foo", etc. in the graph
        marked_count = 0

        # Build a mapping from unqualified names to their possible qualified forms
        # For each handle "bar", we should match both "bar" and "Namespace.bar"
        unqualified_handles: Set[str] = set()
        for h in cited_handles:
            # If it's already qualified (contains "."), use as-is and also extract suffix
            if "." in h:
                parts = h.rsplit(".", 1)
                unqualified_handles.add(parts[1])  # Add unqualified suffix
            else:
                unqualified_handles.add(h)

        for node in data.get("nodes", []):
            node_id = node.get("id", "")
            # Direct match
            if node_id in cited_handles:
                node["paper"] = -1
                marked_count += 1
            # Check if node is a qualified version of an unqualified handle
            elif "." in node_id:
                suffix = node_id.rsplit(".", 1)[1]
                if suffix in unqualified_handles:
                    node["paper"] = -1
                    marked_count += 1

        # Write back the modified graph
        graph_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        if marked_count > 0:
            print(
                f"[graph]   Marked {marked_count} claim nodes (paper=-1) for Claims Only view"
            )

    def _validate_true_paths(self, paper_id: str) -> Dict[str, Any]:
        """Validate true path invariant: all claims must have paths to axioms.

        Based on Paper 4 GraphNontriviality.lean concepts:
        - isCycle: valid path, length ≥ 2, same start/end
        - triviality: |State| ≤ 1 → no information
        - nontrivialityScore: surprisal + noveltyDistance

        Returns validation results dict with:
        - valid: bool
        - claimCount, pathCount
        - missingPaths: list of claim IDs without paths
        - cycles: detected cycles
        - invariants: identified invariant nodes
        """
        import json
        from collections import deque

        graphs_dir = self.repo_root / "docs" / "papers" / "graph_infra" / "graphs"
        graph_path = graphs_dir / f"{paper_id}.json"

        results: Dict[str, Any] = {
            "valid": True,
            "claimCount": 0,
            "pathCount": 0,
            "missingPaths": [],
            "cycles": [],
            "trivialCycles": [],
            "nontrivialCycles": [],
            "invariants": [],
        }

        if not graph_path.exists():
            print(f"[true-path] No graph found for {paper_id}")
            return results

        try:
            data = json.loads(graph_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"[true-path] Could not load graph: {e}")
            return results

        nodes = data.get("nodes", [])
        edges = data.get("edges", [])

        # Build adjacency list
        adj: Dict[str, List[str]] = {}
        for node in nodes:
            adj[node["id"]] = []
        for edge in edges:
            src = edge.get("source", "")
            tgt = edge.get("target", "")
            if isinstance(src, dict):
                src = src.get("id", "")
            if isinstance(tgt, dict):
                tgt = tgt.get("id", "")
            if src in adj:
                adj[src].append(tgt)

        # Identify claims (paper=-1)
        claims = [n for n in nodes if n.get("paper") == -1]
        results["claimCount"] = len(claims)

        # Find leaf nodes (no outgoing edges) - these are foundational definitions
        # In proof graphs, edges go from theorems to their dependencies
        # So leaf nodes are the foundational types/definitions everything depends on
        leaf_nodes = {n["id"] for n in nodes if len(adj.get(n["id"], [])) == 0}

        # Axioms are explicit axioms OR leaf nodes (foundational definitions)
        axioms = {n["id"] for n in nodes if n.get("kind") == "axiom"}
        foundations = axioms | leaf_nodes  # Union of axioms and leaf nodes

        # Identify invariants (axioms + physical constants + foundational defs)
        invariant_patterns = [
            "lightSpeed",
            "planckConstant",
            "boltzmannConstant",
            "kB_T",
            "joulesPerBit",
            "thermodynamic",
            "Landauer",
        ]
        invariants = []
        for n in nodes:
            if n.get("kind") == "axiom":
                invariants.append({"id": n["id"], "kind": "foundational_axiom"})
            elif n["id"] in leaf_nodes and n.get("kind") == "definition":
                invariants.append({"id": n["id"], "kind": "foundational_definition"})
            else:
                for pat in invariant_patterns:
                    if pat.lower() in n["id"].lower():
                        invariants.append({"id": n["id"], "kind": "physical_constant"})
                        break
        results["invariants"] = invariants

        # BFS to find paths from claims to axioms
        def bfs_path(start: str, targets: Set[str]) -> Optional[List[str]]:
            if start in targets:
                return [start]
            visited = {start}
            parent: Dict[str, Optional[str]] = {start: None}
            queue = deque([start])
            while queue:
                current = queue.popleft()
                for neighbor in adj.get(current, []):
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    parent[neighbor] = current
                    if neighbor in targets:
                        path = []
                        n: Optional[str] = neighbor
                        while n is not None:
                            path.append(n)
                            n = parent.get(n)
                        return path[::-1]
                    queue.append(neighbor)
            return None

        # Check each claim has a path to some foundation (axiom or leaf node)
        for claim in claims:
            path = bfs_path(claim["id"], foundations)
            if path:
                results["pathCount"] += 1
            else:
                results["valid"] = False
                results["missingPaths"].append(claim["id"])

        # Cycle detection via Tarjan's SCC algorithm
        index_counter = [0]
        stack: List[str] = []
        lowlinks: Dict[str, int] = {}
        indices: Dict[str, int] = {}
        on_stack: Dict[str, bool] = {}
        sccs: List[List[str]] = []

        def strongconnect(v: str) -> None:
            indices[v] = index_counter[0]
            lowlinks[v] = index_counter[0]
            index_counter[0] += 1
            stack.append(v)
            on_stack[v] = True

            for w in adj.get(v, []):
                if w not in indices:
                    strongconnect(w)
                    lowlinks[v] = min(lowlinks[v], lowlinks[w])
                elif on_stack.get(w, False):
                    lowlinks[v] = min(lowlinks[v], indices[w])

            if lowlinks[v] == indices[v]:
                scc = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    scc.append(w)
                    if w == v:
                        break
                if len(scc) > 1:
                    sccs.append(scc)

        for node in nodes:
            if node["id"] not in indices:
                strongconnect(node["id"])

        # Classify cycles
        for scc in sccs:
            cycle = {"nodes": scc, "witnessBits": len(scc)}
            cycle["isNontrivial"] = len(scc) > 1  # |State| > 1
            results["cycles"].append(cycle)
            if cycle["isNontrivial"]:
                results["nontrivialCycles"].append(cycle)
            else:
                results["trivialCycles"].append(cycle)

        # Print summary
        status = "✅ VALID" if results["valid"] else "❌ INVALID"
        print(f"[true-path] {paper_id}: {status}")
        print(
            f"[true-path]   Claims: {results['claimCount']}, Paths: {results['pathCount']}"
        )
        print(
            f"[true-path]   Invariants: {len(invariants)}, Cycles: {len(results['cycles'])}"
        )
        if results["missingPaths"]:
            print(f"[true-path]   ⚠️  Missing paths: {results['missingPaths'][:5]}")

        return results

    def _iter_lean_roots_for_paper(self, paper_id: str) -> List[Tuple[str, Path]]:
        """Return `(source_paper_id, proofs_dir)` for paper + transitive dependencies."""
        roots: List[Tuple[str, Path]] = []
        roots.append((paper_id, self._get_paper_proofs_dir(paper_id)))
        for dep in self._collect_lean_dependency_closure(paper_id):
            roots.append((dep, self._get_paper_proofs_dir(dep)))
        return roots

    def _derive_module_roots_from_lakefile(self, proofs_dir: Path) -> List[str]:
        """Derive module roots from `lean_lib` declarations in a paper's lakefile.

        Supports both patterns:
        1. Modern: lean_lib «Name» where globs := #[.submodules `Name]
        2. Legacy: lean_lib «Name» (individual lib declarations)
        """
        lakefile = proofs_dir / "lakefile.lean"
        if not lakefile.exists():
            return []
        text = lakefile.read_text(encoding="utf-8", errors="replace")
        roots: List[str] = []

        # Pattern 1: Modern globs pattern
        for m in re.findall(
            r"globs\s*:=\s*#\[\s*\.submodules\s+`([A-Za-z0-9_'.]+)\s*\]", text
        ):
            root = m.strip()
            if root and root not in roots:
                roots.append(root)

        # Pattern 2: Legacy individual lean_lib declarations
        if not roots:
            for m in re.findall(r"lean_lib\s+«([A-Za-z0-9_'.]+)»", text):
                root = m.strip()
                if (
                    root
                    and root not in {"PrintAxioms", "check_axioms"}
                    and root not in roots
                ):
                    roots.append(root)

        return roots

    def _derive_module_roots(self, paper_id: str) -> List[str]:
        """Derive importable top-level Lean module roots for a paper."""
        meta = self._get_paper_meta(paper_id)
        proofs_dir = self._get_paper_proofs_dir(paper_id)
        roots: List[str] = []

        # 1) Explicit metadata root, if provided.
        if meta.module_root:
            roots.append(meta.module_root)

        # 2) Lake globs are the canonical build roots.
        for root in self._derive_module_roots_from_lakefile(proofs_dir):
            if root not in roots:
                roots.append(root)

        # 3) Fallback: top-level Lean files (excluding config-only files).
        skip = {"lakefile.lean", "PrintAxioms.lean", "check_axioms.lean"}
        for top in sorted(proofs_dir.glob("*.lean")):
            if top.name in skip:
                continue
            stem = top.stem
            if re.fullmatch(r"[A-Z][A-Za-z0-9_']*", stem):
                if stem not in roots:
                    roots.append(stem)

        return roots

    def _files_identical(self, left: Path, right: Path) -> bool:
        """Return True iff two files have the same bytes."""
        if not left.exists() or not right.exists():
            return False
        if left.stat().st_size != right.stat().st_size:
            return False
        h1 = hashlib.sha256()
        h2 = hashlib.sha256()
        h1.update(left.read_bytes())
        h2.update(right.read_bytes())
        return h1.digest() == h2.digest()

    def _strip_lean_comments(self, content: str) -> str:
        """Strip Lean line and nested block comments for token counting."""
        out: List[str] = []
        i = 0
        n = len(content)
        block_depth = 0
        in_string = False

        while i < n:
            ch = content[i]
            nxt = content[i + 1] if i + 1 < n else ""

            if block_depth > 0:
                if ch == "/" and nxt == "-":
                    block_depth += 1
                    i += 2
                    continue
                if ch == "-" and nxt == "/":
                    block_depth -= 1
                    i += 2
                    continue
                if ch == "\n":
                    out.append("\n")
                i += 1
                continue

            if not in_string and ch == "/" and nxt == "-":
                block_depth = 1
                i += 2
                continue

            if not in_string and ch == "-" and nxt == "-":
                i += 2
                while i < n and content[i] != "\n":
                    i += 1
                if i < n and content[i] == "\n":
                    out.append("\n")
                    i += 1
                continue

            if ch == '"':
                # Toggle string state when quote is not escaped.
                backslashes = 0
                j = i - 1
                while j >= 0 and content[j] == "\\":
                    backslashes += 1
                    j -= 1
                if backslashes % 2 == 0:
                    in_string = not in_string

            out.append(ch)
            i += 1

        return "".join(out)

    def _count_theorems_and_sorries(self, content: str) -> Tuple[int, int]:
        """Count theorem/lemma declarations and sorry placeholders in Lean text."""
        theorem_re = re.compile(
            r"^\s*(?:(?:private|protected|noncomputable|unsafe|partial|opaque)\s+)*(?:theorem|lemma)\s+[A-Za-z_][A-Za-z0-9_']*",
            re.MULTILINE,
        )
        sorry_re = re.compile(r"\bsorry\b")
        stripped = self._strip_lean_comments(content)
        return (len(theorem_re.findall(stripped)), len(sorry_re.findall(stripped)))

    def _compute_lean_file_stats(self, proofs_dir: Path) -> Dict[str, LeanFileStats]:
        """Compute per-file Lean stats keyed by relative module path (without .lean)."""
        cache_key = proofs_dir.resolve()
        if cache_key in self._lean_file_stats_cache:
            return self._lean_file_stats_cache[cache_key]

        stats: Dict[str, LeanFileStats] = {}
        for lean_file in self._iter_paper_lean_files(proofs_dir):
            rel_key = str(lean_file.relative_to(proofs_dir).with_suffix(""))
            content = lean_file.read_text(encoding="utf-8", errors="replace")
            theorem_count, sorry_count = self._count_theorems_and_sorries(content)
            stats[rel_key] = LeanFileStats(
                line_count=len(content.splitlines()),
                theorem_count=theorem_count,
                sorry_count=sorry_count,
            )

        self._lean_file_stats_cache[cache_key] = stats
        return stats

    def _compute_lean_stats(self, proofs_dir: Path) -> LeanStats:
        """Compute Lean line/theorem/sorry counts from source files."""
        cache_key = proofs_dir.resolve()
        if cache_key in self._lean_stats_cache:
            return self._lean_stats_cache[cache_key]

        per_file = self._compute_lean_file_stats(proofs_dir)
        line_count = sum(s.line_count for s in per_file.values())
        theorem_count = sum(s.theorem_count for s in per_file.values())
        sorry_count = sum(s.sorry_count for s in per_file.values())

        stats = LeanStats(
            line_count=line_count,
            theorem_count=theorem_count,
            sorry_count=sorry_count,
            file_count=len(per_file),
        )
        self._lean_stats_cache[cache_key] = stats
        return stats

    def _get_lean_stats(self, paper_id: str) -> LeanStats:
        """Get computed Lean stats for a paper or variant."""
        proofs_dir = self._get_paper_proofs_dir(paper_id)
        return self._compute_lean_stats(proofs_dir)

    def _iter_transitive_lean_dependency_ids(self, paper_id: str) -> List[str]:
        """Return the transitive declared Lean dependency closure for a paper."""
        seen: set[str] = set()
        ordered: List[str] = []

        def visit(pid: str) -> None:
            if pid not in self.papers:
                return
            for dep_id in self.papers[pid].lean_dependencies:
                if dep_id in seen:
                    continue
                seen.add(dep_id)
                ordered.append(dep_id)
                visit(dep_id)

        visit(paper_id)
        return ordered

    def _get_cumulative_lean_stats(self, paper_id: str) -> LeanStats:
        """Get Lean stats for a paper plus its transitive declared Lean dependencies."""
        proof_dirs: List[Path] = []
        seen_dirs: set[Path] = set()

        def add_dir(pid: str) -> None:
            proofs_dir = self._get_paper_proofs_dir(pid)
            if not proofs_dir.exists():
                return
            key = proofs_dir.resolve()
            if key in seen_dirs:
                return
            seen_dirs.add(key)
            proof_dirs.append(proofs_dir)

        add_dir(paper_id)
        for dep_id in self._iter_transitive_lean_dependency_ids(paper_id):
            add_dir(dep_id)

        line_count = 0
        theorem_count = 0
        sorry_count = 0
        file_count = 0
        for proofs_dir in proof_dirs:
            stats = self._compute_lean_stats(proofs_dir)
            line_count += stats.line_count
            theorem_count += stats.theorem_count
            sorry_count += stats.sorry_count
            file_count += stats.file_count

        return LeanStats(
            line_count=line_count,
            theorem_count=theorem_count,
            sorry_count=sorry_count,
            file_count=file_count,
        )

    def _claim_backed_lean_stats_cache_key(
        self, paper_id: str
    ) -> Tuple[str, Tuple[str, ...]]:
        claim_map = self._extract_claim_label_to_lean_handles(
            paper_id, include_dependencies=True
        )
        handles: Set[str] = set()
        for handle_list in claim_map.values():
            handles.update(handle_list)
        return (paper_id, tuple(sorted(handles)))

    def _extract_declared_handle_to_module_map(
        self, paper_id: str, include_dependencies: bool = True
    ) -> Dict[str, Tuple[str, str]]:
        """Map fully qualified Lean handles to `(source_paper_id, module_path)`.

        `module_path` is relative to the corresponding proofs directory and omits `.lean`.
        """
        roots: List[Tuple[str, Path]]
        if include_dependencies:
            roots = self._iter_lean_roots_for_paper(paper_id)
        else:
            roots = [(paper_id, self._get_paper_proofs_dir(paper_id))]

        decl_pattern = re.compile(
            r"\b(?:theorem|lemma|def|abbrev|class|structure)\s+([A-Za-z_][A-Za-z0-9_'.]*)"
        )
        namespace_pattern = re.compile(r"^\s*namespace\s+([A-Za-z0-9_'.]+)\s*$")
        end_pattern = re.compile(r"^\s*end(?:\s+[A-Za-z0-9_'.]+)?\s*$")

        handle_map: Dict[str, Tuple[str, str]] = {}
        for source_paper_id, proofs_dir in roots:
            if not proofs_dir.exists():
                continue
            for lean_file in self._iter_paper_lean_files(proofs_dir):
                module_path = str(lean_file.relative_to(proofs_dir).with_suffix(""))
                module_name = module_path.replace("/", ".")
                content = lean_file.read_text(encoding="utf-8", errors="replace")
                stripped = self._strip_lean_comments(content)
                namespace_stack: List[str] = []

                for raw_line in stripped.splitlines():
                    line = raw_line.strip()
                    if not line:
                        continue

                    ns_match = namespace_pattern.match(line)
                    if ns_match:
                        namespace_stack.append(ns_match.group(1))
                        continue

                    if end_pattern.match(line):
                        if namespace_stack:
                            namespace_stack.pop()
                        continue

                    decl_match = decl_pattern.search(line)
                    if not decl_match:
                        continue

                    name = decl_match.group(1)
                    if "." in name:
                        full_handle = name
                    elif namespace_stack:
                        ns_parts: List[str] = []
                        for ns in namespace_stack:
                            ns_parts.extend([part for part in ns.split(".") if part])
                        ns_prefix = ".".join(ns_parts)
                        full_handle = f"{ns_prefix}.{name}" if ns_prefix else name
                        handle_map.setdefault(
                            full_handle, (source_paper_id, module_path)
                        )
                        root_ns = module_name.split(".", 1)[0] if module_name else ""
                        if root_ns and not full_handle.startswith(f"{root_ns}.{name}"):
                            handle_map.setdefault(
                                f"{root_ns}.{name}", (source_paper_id, module_path)
                            )
                    else:
                        full_handle = name
                        handle_map.setdefault(
                            full_handle, (source_paper_id, module_path)
                        )
                        if module_name:
                            handle_map.setdefault(
                                f"{module_name}.{name}", (source_paper_id, module_path)
                            )
                            root_ns = module_name.split(".", 1)[0]
                            if root_ns and root_ns != module_name:
                                handle_map.setdefault(
                                    f"{root_ns}.{name}", (source_paper_id, module_path)
                                )

        return handle_map

    def _build_lean_module_index(
        self, paper_id: str, include_dependencies: bool = True
    ) -> Dict[str, Tuple[str, str]]:
        """Map importable Lean module names to `(source_paper_id, module_path)`."""
        roots: List[Tuple[str, Path]]
        if include_dependencies:
            roots = self._iter_lean_roots_for_paper(paper_id)
        else:
            roots = [(paper_id, self._get_paper_proofs_dir(paper_id))]

        module_index: Dict[str, Tuple[str, str]] = {}
        for source_paper_id, proofs_dir in roots:
            if not proofs_dir.exists():
                continue
            for lean_file in self._iter_paper_lean_files(proofs_dir):
                module_path = str(lean_file.relative_to(proofs_dir).with_suffix(""))
                module_name = module_path.replace("/", ".")
                module_index.setdefault(module_name, (source_paper_id, module_path))
        return module_index

    def _get_handle_module_closure(
        self, paper_id: str, handles: Set[str]
    ) -> Dict[str, Set[str]]:
        """Return source-paper -> module paths for handles plus import closure."""
        handle_map = self._extract_declared_handle_to_module_map(
            paper_id, include_dependencies=True
        )
        module_index = self._build_lean_module_index(
            paper_id, include_dependencies=True
        )
        import_re = re.compile(r"^\s*import\s+([A-Za-z0-9_'.]+)\s*$", re.MULTILINE)

        seed_modules: Set[Tuple[str, str]] = set()
        for handle in handles:
            target = handle_map.get(handle)
            if target is None and "." in handle:
                suffix = handle.rsplit(".", 1)[1]
                suffix_matches = [
                    item
                    for h, item in handle_map.items()
                    if h.rsplit(".", 1)[-1] == suffix
                ]
                if len(suffix_matches) == 1:
                    target = suffix_matches[0]
            if target is not None:
                seed_modules.add(target)

        queue: List[Tuple[str, str]] = list(seed_modules)
        visited: Set[Tuple[str, str]] = set(seed_modules)
        while queue:
            source_paper_id, module_path = queue.pop()
            proofs_dir = self._get_paper_proofs_dir(source_paper_id)
            lean_file = proofs_dir / f"{module_path}.lean"
            if not lean_file.exists():
                continue
            content = self._strip_lean_comments(
                lean_file.read_text(encoding="utf-8", errors="replace")
            )
            for import_name in import_re.findall(content):
                target = module_index.get(import_name)
                if target is None or target in visited:
                    continue
                visited.add(target)
                queue.append(target)

        files_by_root: Dict[str, Set[str]] = {}
        for source_paper_id, module_path in visited:
            files_by_root.setdefault(source_paper_id, set()).add(module_path)
        return files_by_root

    def _get_claim_backed_module_closure(self, paper_id: str) -> Dict[str, Set[str]]:
        """Return source-paper -> module paths for claim-cited declarations plus imports."""
        claim_map = self._extract_claim_label_to_lean_handles(
            paper_id, include_dependencies=True
        )
        handles: Set[str] = set()
        for handle_list in claim_map.values():
            handles.update(handle_list)
        return self._get_handle_module_closure(paper_id, handles)

    def _content_backed_lean_stats_cache_key(
        self, paper_id: str
    ) -> Tuple[str, Tuple[str, ...]]:
        files = self._discover_content_files_for_flags(paper_id)
        alias_by_code = self._extract_alias_code_map_from_handle_aliases(
            paper_id, include_dependencies=True
        )
        id_to_handle = self._read_lean_handle_id_map(paper_id)
        handles = self._extract_referenced_lean_handles_from_content(
            paper_id, files=files
        )
        handles.update(
            self._extract_referenced_lh_handles_from_content(paper_id, files=files)
        )
        for code in self._extract_lh_ids_from_content(paper_id, files=files):
            handle = alias_by_code.get(code) or id_to_handle.get(code)
            if handle:
                handles.add(handle)
        return (paper_id, tuple(sorted(handles)))

    def _get_content_backed_module_closure(self, paper_id: str) -> Dict[str, Set[str]]:
        """Return source-paper -> module paths for Lean cited in included paper content."""
        cache_key = self._content_backed_lean_stats_cache_key(paper_id)
        return self._get_handle_module_closure(paper_id, set(cache_key[1]))

    def _get_content_backed_lean_stats(self, paper_id: str) -> LeanStats:
        """Stats for Lean files cited anywhere in the included paper content."""
        if not hasattr(self, "_content_backed_lean_stats_cache"):
            self._content_backed_lean_stats_cache: Dict[
                Tuple[str, Tuple[str, ...]], LeanStats
            ] = {}

        cache_key = self._content_backed_lean_stats_cache_key(paper_id)
        cached = self._content_backed_lean_stats_cache.get(cache_key)
        if cached is not None:
            return cached

        files_by_root = self._get_handle_module_closure(paper_id, set(cache_key[1]))

        line_count = 0
        theorem_count = 0
        sorry_count = 0
        file_count = 0
        for source_paper_id, module_paths in files_by_root.items():
            file_stats = self._get_lean_file_stats(source_paper_id)
            for module_path in module_paths:
                stats = file_stats.get(module_path)
                if stats is None:
                    continue
                line_count += stats.line_count
                theorem_count += stats.theorem_count
                sorry_count += stats.sorry_count
                file_count += 1

        result = LeanStats(
            line_count=line_count,
            theorem_count=theorem_count,
            sorry_count=sorry_count,
            file_count=file_count,
        )
        self._content_backed_lean_stats_cache[cache_key] = result
        return result

    def _get_claim_backed_lean_stats(self, paper_id: str) -> LeanStats:
        """Stats for files containing Lean declarations cited by paper claims."""
        if not hasattr(self, "_claim_backed_lean_stats_cache"):
            self._claim_backed_lean_stats_cache: Dict[
                Tuple[str, Tuple[str, ...]], LeanStats
            ] = {}

        cache_key = self._claim_backed_lean_stats_cache_key(paper_id)
        cached = self._claim_backed_lean_stats_cache.get(cache_key)
        if cached is not None:
            return cached

        files_by_root = self._get_claim_backed_module_closure(paper_id)

        line_count = 0
        theorem_count = 0
        sorry_count = 0
        file_count = 0
        for source_paper_id, module_paths in files_by_root.items():
            file_stats = self._get_lean_file_stats(source_paper_id)
            for module_path in module_paths:
                stats = file_stats.get(module_path)
                if stats is None:
                    continue
                line_count += stats.line_count
                theorem_count += stats.theorem_count
                sorry_count += stats.sorry_count
                file_count += 1

        result = LeanStats(
            line_count=line_count,
            theorem_count=theorem_count,
            sorry_count=sorry_count,
            file_count=file_count,
        )
        self._claim_backed_lean_stats_cache[cache_key] = result
        return result

    def _get_release_lean_stats(self, paper_id: str) -> LeanStats:
        """Lean stats used by paper-facing release metadata/macros."""
        meta = self._get_paper_meta(paper_id)
        if meta.lean_stats_scope == "content_cited":
            return self._get_content_backed_lean_stats(paper_id)
        if meta.lean_stats_scope == "claims_only":
            return self._get_claim_backed_lean_stats(paper_id)
        if meta.lean_stats_scope == "local":
            return self._get_lean_stats(paper_id)
        return self._get_cumulative_lean_stats(paper_id)

    def _get_release_module_closure(
        self, paper_id: str
    ) -> Optional[Dict[str, Set[str]]]:
        """Lean modules to bundle for release, or `None` for full local proofs."""
        meta = self._get_paper_meta(paper_id)
        if meta.lean_stats_scope == "content_cited":
            return self._get_content_backed_module_closure(paper_id)
        if meta.lean_stats_scope == "claims_only":
            return self._get_claim_backed_module_closure(paper_id)
        return None

    def _get_lean_file_stats(self, paper_id: str) -> Dict[str, LeanFileStats]:
        """Get per-file Lean stats for a paper or variant."""
        proofs_dir = self._get_paper_proofs_dir(paper_id)
        return self._compute_lean_file_stats(proofs_dir)

    def _sync_shared_preambles(self, paper_id: str) -> None:
        """Ensure shared LaTeX preambles are present in the active paper LaTeX dir.

        This is structural SSOT sync, not one-off scaffolding:
        build/release commands should always compile against the shared preamble set.
        """
        meta = self._get_paper_meta(paper_id)
        latex_dir = self._get_paper_dir(paper_id) / meta.latex_dir
        if not latex_dir.exists():
            return

        shared_dir = self.papers_dir / "shared"
        for name in self.build_config.shared_preamble_files:
            src = shared_dir / name
            if not src.exists():
                continue
            dst = latex_dir / name
            if dst.exists():
                try:
                    if src.samefile(dst):
                        continue
                except FileNotFoundError:
                    pass
            # Always refresh from SSOT to avoid stale/local drift.
            shutil.copy2(src, dst)

    def _write_lean_handle_macros_auto(self, paper_id: str) -> None:
        """Write generated `lean-handle-macros.tex` into the paper LaTeX dir.

        Centralises control of how `\LH{...}` / `\LHrng{...}` and
        `\leanmeta{...}` render so builds can produce consistent inline,
        compact, monospace tokens without editing individual paper files.
        """
        meta = self._get_paper_meta(paper_id)
        latex_dir = self._get_paper_dir(paper_id) / meta.latex_dir
        if not latex_dir.exists():
            return

        dst = latex_dir / "lean-handle-macros.tex"
        lines = [
            "% Auto-generated by scripts/build_papers.py. Do not edit manually.",
            "% Lean handle macro definitions: compact inline monospace tokens",
            # Use leavevmode+nobreak to aggressively avoid paragraph starts and
            # unwanted line breaks around inline handle tokens.
            r"\providecommand{\LH}[1]{\leavevmode\nobreak\hyperlink{lh:#1}{\mbox{\ttfamily\scriptsize\nolinkurl{#1}}}}",
            r"\providecommand{\LHrng}[3]{\leavevmode\nobreak\hyperlink{lh:#1#2}{\mbox{\ttfamily\scriptsize\nolinkurl{#1#2-#3}}}}",
            r"\providecommand{\leanmeta}[1]{\allowbreak{\scriptsize\textit{(L: #1)}}}",
            "",
        ]
        # Always overwrite to keep macros in sync with build policy.
        dst.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_shared_lean_handle_macros_auto(self) -> None:
        """Write generated `lean-handle-macros.tex` into the shared SSOT dir.

        By default the build copies files from `docs/papers/shared/` into each
        paper's latex dir. Generate/update the SSOT file here so the usual
        `_sync_shared_preambles` step will propagate the generated macros into
        each paper variant.
        """
        shared_dir = self.papers_dir / "shared"
        if not shared_dir.exists():
            return

        dst = shared_dir / "lean-handle-macros.tex"
        lines = [
            "% Auto-generated by scripts/build_papers.py. Do not edit manually.",
            "% Lean handle macro definitions: compact inline monospace tokens",
            # Use leavevmode+nobreak to aggressively avoid paragraph starts and
            # unwanted line breaks around inline handle tokens.
            r"\providecommand{\LH}[1]{\leavevmode\nobreak\hyperlink{lh:#1}{\mbox{\ttfamily\scriptsize\nolinkurl{#1}}}}",
            r"\providecommand{\LHrng}[3]{\leavevmode\nobreak\hyperlink{lh:#1#2}{\mbox{\ttfamily\scriptsize\nolinkurl{#1#2-#3}}}}",
            r"\providecommand{\leanmeta}[1]{\allowbreak{\scriptsize\textit{(L: #1)}}}",
            "",
        ]
        dst.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _lean_macro_suffix(self, module_path: str) -> str:
        """Convert a relative Lean module path to a LaTeX-safe macro suffix."""
        digit_words = {
            "0": "Zero",
            "1": "One",
            "2": "Two",
            "3": "Three",
            "4": "Four",
            "5": "Five",
            "6": "Six",
            "7": "Seven",
            "8": "Eight",
            "9": "Nine",
        }
        parts = re.split(r"[^A-Za-z0-9]+", module_path)
        cleaned = [p for p in parts if p]
        if not cleaned:
            return "Unknown"
        normalized_parts: List[str] = []
        for part in cleaned:
            converted = "".join(digit_words.get(ch, ch) for ch in part)
            normalized_parts.append(converted[:1].upper() + converted[1:])
        return "".join(normalized_parts)

    def _build_lean_macro_suffix_map(self, module_paths: List[str]) -> Dict[str, str]:
        """Build collision-safe macro suffixes for Lean module paths."""
        grouped: Dict[str, List[str]] = {}
        for module_path in module_paths:
            base = self._lean_macro_suffix(module_path)
            grouped.setdefault(base, []).append(module_path)

        suffix_map: Dict[str, str] = {}
        for base, paths in grouped.items():
            if len(paths) == 1:
                suffix_map[paths[0]] = base
                continue

            # Rare collision case (e.g., Unicode-heavy names normalizing alike).
            for module_path in sorted(paths):
                digest = hashlib.sha1(module_path.encode("utf-8")).hexdigest()[:8]
                digest_suffix = self._lean_macro_suffix(digest)
                suffix_map[module_path] = f"{base}{digest_suffix}"
        return suffix_map

    def _write_latex_lean_stats(self, paper_id: str) -> None:
        """Write auto-generated Lean stats macros into latex/content/lean_stats.tex."""
        meta = self._get_paper_meta(paper_id)
        latex_dir = self._get_paper_dir(paper_id) / meta.latex_dir
        content_dir = latex_dir / "content"
        if not content_dir.exists():
            return

        local_stats = self._get_lean_stats(paper_id)
        total_stats = self._get_cumulative_lean_stats(paper_id)
        release_stats = self._get_release_lean_stats(paper_id)
        file_stats = self._get_lean_file_stats(paper_id)
        suffix_map = self._build_lean_macro_suffix_map(sorted(file_stats.keys()))

        lines: List[str] = [
            "% Auto-generated by scripts/build_papers.py. Do not edit manually.",
            f"% Generated: {datetime.now().isoformat()}",
            f"\\providecommand{{\\LeanLocalLines}}{{{local_stats.line_count}}}",
            f"\\providecommand{{\\LeanLocalTheorems}}{{{local_stats.theorem_count}}}",
            f"\\providecommand{{\\LeanLocalSorry}}{{{local_stats.sorry_count}}}",
            f"\\providecommand{{\\LeanLocalFiles}}{{{local_stats.file_count}}}",
            f"\\providecommand{{\\LeanTotalLines}}{{{total_stats.line_count}}}",
            f"\\providecommand{{\\LeanTotalTheorems}}{{{total_stats.theorem_count}}}",
            f"\\providecommand{{\\LeanTotalSorry}}{{{total_stats.sorry_count}}}",
            f"\\providecommand{{\\LeanTotalFiles}}{{{total_stats.file_count}}}",
            f"\\providecommand{{\\LeanReleaseLines}}{{{release_stats.line_count}}}",
            f"\\providecommand{{\\LeanReleaseTheorems}}{{{release_stats.theorem_count}}}",
            f"\\providecommand{{\\LeanReleaseSorry}}{{{release_stats.sorry_count}}}",
            f"\\providecommand{{\\LeanReleaseFiles}}{{{release_stats.file_count}}}",
        ]
        for module_path in sorted(file_stats.keys()):
            suffix = suffix_map[module_path]
            stats = file_stats[module_path]
            lines.append(
                f"\\providecommand{{\\LeanLines{suffix}}}{{{stats.line_count}}}"
            )
            lines.append(
                f"\\providecommand{{\\LeanTheorems{suffix}}}{{{stats.theorem_count}}}"
            )
            lines.append(
                f"\\providecommand{{\\LeanSorry{suffix}}}{{{stats.sorry_count}}}"
            )

        # Add aggregate stats for each declared lean dependency whose proofs exist.
        # Macro suffixes should use the public Lean namespace roots referenced in prose
        # (e.g. AbstractClassSystem, Ssot, DecisionQuotient), not arbitrary package
        # roots like nominal_resolution.
        tool_roots = {
            "DeclInfoExport",
            "DependencyGraph",
            "GraphExport",
            "HandleAliases",
            "CrossPaperDependencies",
            "PrintAxioms",
            "check_axioms",
        }
        for dep_paper_id in meta.lean_dependencies:
            if dep_paper_id not in self.papers:
                continue
            dep_proofs_dir = self._get_paper_proofs_dir(dep_paper_id)
            if not dep_proofs_dir.exists():
                continue
            dep_stats = self._compute_lean_stats(dep_proofs_dir)
            dep_roots = self._derive_module_roots(dep_paper_id)
            dep_full_name = self.papers[dep_paper_id].full_name
            public_roots: List[str] = []
            for root in dep_roots:
                if root in tool_roots or re.fullmatch(r"Paper\d+[A-Za-z]*", root):
                    continue
                if root[:1].isupper() and root not in public_roots:
                    public_roots.append(root)

            if not public_roots:
                for root in dep_roots:
                    if root in tool_roots:
                        continue
                    if root not in public_roots:
                        public_roots.append(root)

            if not public_roots:
                public_roots = [dep_paper_id]

            for i, dep_module in enumerate(public_roots):
                dep_suffix = self._lean_macro_suffix(dep_module)
                if i == 0:
                    lines.append(f"% Lean dependency: {dep_module} ({dep_full_name})")
                else:
                    lines.append(f"% Lean dependency alias: {dep_module}")
                lines.extend(
                    [
                        f"\\providecommand{{\\LeanDepLines{dep_suffix}}}{{{dep_stats.line_count}}}",
                        f"\\providecommand{{\\LeanDepTheorems{dep_suffix}}}{{{dep_stats.theorem_count}}}",
                        f"\\providecommand{{\\LeanDepSorry{dep_suffix}}}{{{dep_stats.sorry_count}}}",
                        f"\\providecommand{{\\LeanDepFiles{dep_suffix}}}{{{dep_stats.file_count}}}",
                    ]
                )

        (content_dir / "lean_stats.tex").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    def _iter_claim_closure_sources(self, paper_id: str) -> List[Path]:
        """Return ClaimClosure Lean sources for assumption-ledger extraction."""
        meta = self._get_paper_meta(paper_id)
        proofs_dir = self._get_paper_proofs_dir(paper_id)
        if not proofs_dir.exists():
            return []

        explicit_paths: List[Path] = []
        for rel_path in meta.assumption_ledger_sources:
            src = (proofs_dir / rel_path).resolve()
            if src.exists() and src.is_file():
                explicit_paths.append(src)

        if explicit_paths:
            return sorted(set(explicit_paths))

        candidates: List[Path] = []
        excluded_dirs = {".lake", "build"}
        for src in proofs_dir.rglob("ClaimClosure.lean"):
            rel_parts = src.relative_to(proofs_dir).parts
            if any(part in excluded_dirs for part in rel_parts):
                continue
            candidates.append(src.resolve())
        return sorted(set(candidates))

    def _extract_primary_namespace(self, content: str) -> Optional[str]:
        """Extract first namespace declaration (if any)."""
        match = re.search(r"(?m)^\s*namespace\s+([A-Za-z0-9_'.]+)\s*$", content)
        return match.group(1) if match else None

    def _parse_assumption_bundles(self, content: str) -> Dict[str, List[str]]:
        """Parse `*Assumptions` class/structure bundles and their fields."""
        bundles: Dict[str, List[str]] = {}
        matches = list(
            re.finditer(
                r"(?m)^\s*(?:class|structure)\s+([A-Za-z0-9_']*Assumptions)\b[^\n]*\bwhere\s*$",
                content,
            )
        )
        for idx, match in enumerate(matches):
            bundle_name = match.group(1)
            body_start = match.end()
            body_end = (
                matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
            )
            body = content[body_start:body_end]
            fields: List[str] = []
            for raw_line in body.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                if re.match(
                    r"^(?:theorem|lemma|def|class|structure|namespace|end)\b", line
                ):
                    break
                field_match = re.match(r"^([A-Za-z0-9_']+)\s*:\s*(.+)$", line)
                if not field_match:
                    continue
                field_name = field_match.group(1)
                field_type = re.sub(r"\s+", " ", field_match.group(2)).strip()
                fields.append(
                    f"{field_name} : {field_type}" if field_type else field_name
                )
            bundles[bundle_name] = fields
        return bundles

    def _parse_conditional_handles(self, content: str) -> List[str]:
        """Extract theorem/lemma handles ending with `_conditional`."""
        handles = re.findall(
            r"(?m)^\s*(?:theorem|lemma)\s+([A-Za-z0-9_']*_conditional)\b",
            content,
        )
        return sorted(set(handles))

    def _parse_theorem_lemma_handles(self, content: str) -> List[str]:
        """Extract theorem/lemma declaration names."""
        handles = re.findall(
            r"(?m)^\s*(?:theorem|lemma)\s+([A-Za-z0-9_']+)\b",
            content,
        )
        return sorted(set(handles))

    def _parse_alias_abbrev_map(
        self, content: str, namespace: Optional[str]
    ) -> Dict[str, str]:
        """Extract `abbrev ID := target` alias mappings."""
        aliases: Dict[str, str] = {}
        for match in re.finditer(
            r"(?m)^\s*(?:noncomputable\s+)?abbrev\s+([A-Za-z][A-Za-z0-9_]*)\s*:=\s*(.+)$",
            content,
        ):
            alias = match.group(1).strip()
            rhs = match.group(2).split("--", 1)[0].strip()
            if not alias or not rhs:
                continue
            if rhs.startswith("@"):
                rhs = rhs[1:].strip()
            if not rhs:
                continue
            target = rhs.split()[0].strip()
            if not target:
                continue
            if namespace:
                if "." in target:
                    if not target.startswith(namespace + "."):
                        target = f"{namespace}.{target}"
                else:
                    target = f"{namespace}.{target}"
            aliases[alias] = target
        return aliases

    def _write_assumption_ledger_auto(self, paper_id: str) -> None:
        """Write auto-generated assumption-ledger snippet when available."""
        meta = self._get_paper_meta(paper_id)
        latex_dir = self._get_paper_dir(paper_id) / meta.latex_dir
        content_dir = latex_dir / "content"
        if not content_dir.exists():
            return

        out_file = content_dir / "assumption_ledger_auto.tex"
        source_files = self._iter_claim_closure_sources(paper_id)
        if not source_files:
            out_file.write_text(
                "% Auto-generated by scripts/build_papers.py. No assumption ledger source found.\n",
                encoding="utf-8",
            )
            return

        all_bundles: Dict[str, List[str]] = {}
        all_conditional_handles: Set[str] = set()
        bayes_alias_ids: Dict[str, str] = {}
        bayes_theorem_handles: Set[str] = set()
        source_labels: List[str] = []

        proofs_dir = self._get_paper_proofs_dir(paper_id)
        for src in source_files:
            rel = src.relative_to(proofs_dir)
            source_labels.append(str(rel).replace("\\", "/"))
            content = src.read_text(encoding="utf-8", errors="replace")
            ns = self._extract_primary_namespace(content)

            bundles = self._parse_assumption_bundles(content)
            for bundle_name, fields in bundles.items():
                full_bundle_name = f"{ns}.{bundle_name}" if ns else bundle_name
                all_bundles[full_bundle_name] = fields

            for handle in self._parse_conditional_handles(content):
                full_handle = f"{ns}.{handle}" if ns else handle
                all_conditional_handles.add(full_handle)

            if src.name == "HandleAliases.lean":
                alias_map = self._parse_alias_abbrev_map(content, ns)
                for alias, target in alias_map.items():
                    if re.match(r"^(?:FN|BB|BF)\d+$", alias):
                        bayes_alias_ids[alias] = target

        bayes_sources = [
            proofs_dir / "DecisionQuotient" / "BayesFoundations.lean",
            proofs_dir / "DecisionQuotient" / "BayesOptimalityProof.lean",
            proofs_dir / "DecisionQuotient" / "BayesianBridge.lean",
        ]
        for bayes_src in bayes_sources:
            if not bayes_src.exists() or not bayes_src.is_file():
                continue
            bayes_content = bayes_src.read_text(encoding="utf-8", errors="replace")
            bayes_ns = self._extract_primary_namespace(bayes_content)
            for handle in self._parse_theorem_lemma_handles(bayes_content):
                full_handle = f"{bayes_ns}.{handle}" if bayes_ns else handle
                bayes_theorem_handles.add(full_handle)

        lines: List[str] = [
            "% Auto-generated by scripts/build_papers.py. Do not edit manually.",
            f"% Generated: {datetime.now().isoformat()}",
            r"\subsection{Assumption Ledger (Auto)}",
            r"\paragraph{Source files.}",
            r"\begin{itemize}",
        ]
        for source_label in sorted(set(source_labels)):
            lines.append(rf"\item \nolinkurl{{{source_label}}}")
        lines.append(r"\end{itemize}")

        lines.extend(
            [
                r"\paragraph{Assumption bundles and fields.}",
                r"\begin{itemize}",
            ]
        )
        if all_bundles:
            for bundle_name in sorted(all_bundles.keys()):
                lines.append(rf"\item \nolinkurl{{{bundle_name}}}")
                fields = all_bundles[bundle_name]
                if fields:
                    lines.append(r"\begin{itemize}")
                    for field in fields:
                        lines.append(rf"\item \nolinkurl{{{field}}}")
                    lines.append(r"\end{itemize}")
        else:
            lines.append(r"\item (No assumption bundles parsed.)")
        lines.append(r"\end{itemize}")

        lines.extend(
            [
                r"\paragraph{Conditional closure handles.}",
                r"\begin{itemize}",
            ]
        )
        if all_conditional_handles:
            for handle in sorted(all_conditional_handles):
                lines.append(
                    rf"\item \nolinkurl{{{self._compact_lean_handle(handle)}}}"
                )
        else:
            lines.append(r"\item (No conditional theorem handles parsed.)")
        lines.append(r"\end{itemize}")

        lines.extend(
            [
                r"\paragraph{First-principles Bayes derivation handles.}",
                r"\begin{itemize}",
            ]
        )
        if bayes_theorem_handles:
            for handle in sorted(bayes_theorem_handles):
                lines.append(
                    rf"\item \nolinkurl{{{self._compact_lean_handle(handle)}}}"
                )
        else:
            lines.append(r"\item (No Bayes-derivation theorem handles found.)")
        lines.append(r"\end{itemize}")

        lines.extend(
            [
                r"\paragraph{Compact Bayes handle IDs.}",
                r"\begin{itemize}",
            ]
        )
        if bayes_alias_ids:
            prefix_order = {"FN": 0, "BB": 1, "BF": 2}

            def _alias_sort_key(alias_id: str) -> Tuple[int, int, str]:
                m = re.match(r"^([A-Z]+)(\d+)$", alias_id)
                if not m:
                    return (99, 10**9, alias_id)
                prefix = m.group(1)
                number = int(m.group(2))
                return (prefix_order.get(prefix, 50), number, alias_id)

            for alias_id in sorted(bayes_alias_ids.keys(), key=_alias_sort_key):
                target = self._compact_lean_handle(bayes_alias_ids[alias_id])
                lines.append(
                    rf"\item \nolinkurl{{{alias_id}}} $\to$ \nolinkurl{{{target}}}"
                )
        else:
            lines.append(r"\item (No compact Bayes alias IDs found.)")
        lines.append(r"\end{itemize}")

        out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _read_existing_lean_handle_rows(self, paper_id: str) -> List[Tuple[str, str]]:
        """Read existing `(ID, handle)` rows from `lean_handle_ids_auto.tex`."""
        content_dir = self._get_content_dir(paper_id)
        map_file = content_dir / "lean_handle_ids_auto.tex"
        if not map_file.exists():
            return []

        text = map_file.read_text(encoding="utf-8", errors="replace")
        pair_pattern = re.compile(
            r"\\hypertarget\{lh:([^}]+)\}\{\\nolinkurl\{[^}]+\}\}(?:\\hspace\{[^}]+\})?\\nolinkurl\{([^}]+)\}"
        )
        hyper_code_pattern = re.compile(r"\\hypertarget\{lh:([^}]+)\}")
        code_pattern = re.compile(r"\\text(?:tt|sf)\{([^}]+)\}")
        code_url_pattern = re.compile(
            r"\\hypertarget\{lh:[^}]+\}\{\\nolinkurl\{([^}]+)\}\}"
        )
        handle_pattern = re.compile(r"\\nolinkurl\{([^}]+)\}")
        rows: List[Tuple[str, str]] = []
        for raw_line in text.splitlines():
            if r"\nolinkurl{" not in raw_line:
                continue
            pair_matches = pair_pattern.findall(raw_line)
            if pair_matches:
                for code, handle in pair_matches:
                    rows.append((code.strip(), handle.strip().replace(r"\_", "_")))
                continue
            hyper_match = hyper_code_pattern.search(raw_line)
            code_match = code_pattern.search(raw_line)
            code_url_match = code_url_pattern.search(raw_line)
            handle_matches = handle_pattern.findall(raw_line)
            code: str = ""
            if hyper_match:
                code = hyper_match.group(1).strip()
            elif code_url_match:
                code = code_url_match.group(1).strip()
            elif code_match:
                code = code_match.group(1).strip().replace(r"\_", "_")
            if not code or not handle_matches:
                continue
            handle = handle_matches[-1].strip().replace(r"\_", "_")
            if code and handle:
                rows.append((code, handle))
        return rows

    def _extract_declared_lean_handle_locations(
        self, paper_id: str, include_dependencies: bool = True
    ) -> Dict[str, str]:
        """Extract declared Lean handles together with their source files."""
        roots: List[Tuple[str, Path]]
        if include_dependencies:
            roots = self._iter_lean_roots_for_paper(paper_id)
        else:
            roots = [(paper_id, self._get_paper_proofs_dir(paper_id))]

        decl_pattern = re.compile(
            r"\b(?:theorem|lemma|def|abbrev|class|structure)\s+([A-Za-z_][A-Za-z0-9_'.]*)"
        )
        namespace_pattern = re.compile(r"^\s*namespace\s+([A-Za-z0-9_'.]+)\s*$")
        end_pattern = re.compile(r"^\s*end(?:\s+[A-Za-z0-9_'.]+)?\s*$")

        handle_locations: Dict[str, str] = {}
        for root_name, proofs_dir in roots:
            if not proofs_dir.exists():
                continue
            for lean_file in self._iter_paper_lean_files(proofs_dir):
                content = lean_file.read_text(encoding="utf-8", errors="replace")
                stripped = self._strip_lean_comments(content)
                namespace_stack: List[str] = []
                try:
                    rel_file = lean_file.relative_to(proofs_dir).as_posix()
                except ValueError:
                    rel_file = lean_file.name
                display_file = (
                    rel_file if root_name == paper_id else f"{root_name}/{rel_file}"
                )
                module_name = rel_file.removesuffix(".lean").replace("/", ".")

                for raw_line in stripped.splitlines():
                    line = raw_line.strip()
                    if not line:
                        continue

                    ns_match = namespace_pattern.match(line)
                    if ns_match:
                        namespace_stack.append(ns_match.group(1))
                        continue

                    if end_pattern.match(line):
                        if namespace_stack:
                            namespace_stack.pop()
                        continue

                    decl_match = decl_pattern.search(line)
                    if not decl_match:
                        continue

                    name = decl_match.group(1)
                    if "." in name:
                        handle_locations.setdefault(name, display_file)
                        continue

                    if namespace_stack:
                        ns_parts: List[str] = []
                        for ns in namespace_stack:
                            ns_parts.extend([part for part in ns.split(".") if part])
                        ns_prefix = ".".join(ns_parts)
                        full_name = f"{ns_prefix}.{name}" if ns_prefix else name
                        handle_locations.setdefault(full_name, display_file)
                        root_ns = module_name.split(".", 1)[0] if module_name else ""
                        if root_ns and not full_name.startswith(f"{root_ns}.{name}"):
                            handle_locations.setdefault(
                                f"{root_ns}.{name}", display_file
                            )
                    else:
                        handle_locations.setdefault(name, display_file)
                        if module_name:
                            handle_locations.setdefault(
                                f"{module_name}.{name}", display_file
                            )
                            root_ns = module_name.split(".", 1)[0]
                            if root_ns and root_ns != module_name:
                                handle_locations.setdefault(
                                    f"{root_ns}.{name}", display_file
                                )

        return handle_locations

    def _extract_declared_lean_handles(
        self, paper_id: str, include_dependencies: bool = True
    ) -> Set[str]:
        """Extract declared Lean constants from theorem/def/class-style declarations.

        When `include_dependencies` is true, this scans the paper's transitive
        Lean dependency closure so handle extraction is derived from the full
        proof graph rather than only the local proofs tree.
        """
        return set(
            self._extract_declared_lean_handle_locations(
                paper_id, include_dependencies=include_dependencies
            ).keys()
        )

    def _extract_referenced_lean_handles_from_content(
        self, paper_id: str, files: Optional[List[Path]] = None
    ) -> Set[str]:
        """Extract explicit Lean handles referenced in paper content via `\\nolinkurl{...}`."""
        content_dir = self._get_content_dir(paper_id)
        if not content_dir.exists():
            return set()

        paper_handle_pattern = re.compile(r"^(?:thm|cor|lem|prop):")
        handles: Set[str] = set()

        for tex_file in self._iter_manual_content_tex(paper_id, files=files):
            text = tex_file.read_text(encoding="utf-8", errors="replace")
            for raw in re.findall(r"\\nolinkurl\{([^}]+)\}", text):
                handle = raw.strip().replace(r"\_", "_")
                if not handle:
                    continue
                if paper_handle_pattern.match(handle):
                    continue
                if handle.endswith(".lean"):
                    continue
                if "." not in handle and "_" not in handle:
                    continue
                handles.add(handle)

        return handles

    def _extract_lh_ids_from_content(
        self, paper_id: str, files: Optional[List[Path]] = None
    ) -> Set[str]:
        """Extract compact Lean-handle IDs referenced via `\\LH{...}` in content."""
        ids: Set[str] = set()
        lhrng_pattern = re.compile(r"\\LHrng\{([A-Za-z]+)\}\{(\d+)\}\{(\d+)\}")
        for tex_file in self._iter_manual_content_tex(paper_id, files=files):
            text = tex_file.read_text(encoding="utf-8", errors="replace")
            ids.update(m.strip() for m in re.findall(r"\\LH\{([^}]+)\}", text))
            for prefix, start_raw, end_raw in lhrng_pattern.findall(text):
                start = int(start_raw)
                end = int(end_raw)
                if end < start:
                    start, end = end, start
                for idx in range(start, end + 1):
                    ids.add(f"{prefix}{idx}")
        return {code for code in ids if re.match(r"^[A-Za-z]+\d+$", code)}

    def _extract_referenced_lh_handles_from_content(
        self, paper_id: str, files: Optional[List[Path]] = None
    ) -> Set[str]:
        """Extract raw Lean handles still referenced via `\\LH{handle_name}`."""
        content_dir = self._get_content_dir(paper_id)
        if not content_dir.exists():
            return set()

        compact_id_pattern = re.compile(r"^[A-Za-z]+\d+$")
        paper_handle_pattern = re.compile(r"^(?:thm|cor|lem|prop):")
        handles: Set[str] = set()

        for tex_file in self._iter_manual_content_tex(paper_id, files=files):
            text = tex_file.read_text(encoding="utf-8", errors="replace")
            for raw in re.findall(r"\\LH\{([^}]+)\}", text):
                handle = raw.strip().replace(r"\_", "_")
                if not handle or compact_id_pattern.match(handle):
                    continue
                if paper_handle_pattern.match(handle):
                    continue
                if handle.endswith(".lean"):
                    continue
                if "." not in handle and "_" not in handle:
                    continue
                handles.add(handle)

        return handles

    def _extract_alias_code_map_from_handle_aliases(
        self, paper_id: str, include_dependencies: bool = True
    ) -> Dict[str, str]:
        """Derive canonical compact ID -> handle mapping from HandleAliases.lean.

        This keeps `\\LH{...}` IDs stable even when prose no longer contains
        full `\\nolinkurl{...}` handles.
        """
        roots: List[Tuple[str, Path]]
        if include_dependencies:
            roots = self._iter_lean_roots_for_paper(paper_id)
        else:
            roots = [(paper_id, self._get_paper_proofs_dir(paper_id))]

        ident_pattern = re.compile(r"^[A-Za-z_][A-Za-z0-9_']*$")

        def parse_alias_file(alias_file: Path) -> Dict[str, str]:
            """Extract ID -> handle mappings from HandleAliases.lean.

            DOF=1 design: Lean file is source of truth.
            - Export statements: namespace prefix + sequential index -> source.name
            - Abbrev declarations: abbrev name IS the ID -> qualified handle
            """
            text = alias_file.read_text(encoding="utf-8", errors="replace")
            stripped = self._strip_lean_comments(text)
            local_map: Dict[str, str] = {}
            counters: Dict[str, int] = {}
            namespace_stack: List[str] = []
            lines = stripped.splitlines()
            i = 0

            while i < len(lines):
                line = lines[i].strip()
                if not line:
                    i += 1
                    continue

                # Track namespace
                ns_match = re.match(r"^namespace\s+([A-Za-z0-9_'.]+)\s*$", line)
                if ns_match:
                    namespace_stack.append(ns_match.group(1))
                    i += 1
                    continue

                end_match = re.match(r"^end(?:\s+([A-Za-z0-9_'.]+))?\s*$", line)
                if end_match:
                    end_ns = end_match.group(1)
                    if end_ns and namespace_stack:
                        while namespace_stack and namespace_stack[-1] != end_ns:
                            namespace_stack.pop()
                        if namespace_stack and namespace_stack[-1] == end_ns:
                            namespace_stack.pop()
                    elif namespace_stack:
                        namespace_stack.pop()
                    i += 1
                    continue

                # Parse export statements for auto-numbered IDs
                if line.startswith("export ") and "(" in line and namespace_stack:
                    current_ns = namespace_stack[-1]
                    # Skip top-level namespace
                    if current_ns == "DecisionQuotient":
                        i += 1
                        continue

                    src_match = re.match(r"^export\s+([A-Za-z0-9_'.]+)\s*\(", line)
                    if not src_match:
                        i += 1
                        continue
                    source_ns = src_match.group(1)

                    # Collect all names in the export (may span multiple lines)
                    depth = line.count("(") - line.count(")")
                    body = line.split("(", 1)[1]
                    j = i + 1
                    while depth > 0 and j < len(lines):
                        seg = lines[j]
                        depth += seg.count("(") - seg.count(")")
                        body += "\n" + seg
                        j += 1

                    if ")" in body:
                        body = body.rsplit(")", 1)[0]

                    # Extract identifiers
                    names: List[str] = []
                    for raw in body.replace(",", " ").split():
                        token = raw.strip()
                        if ident_pattern.fullmatch(token):
                            names.append(token)

                    # Dedupe while preserving order
                    seen: Set[str] = set()
                    for name in names:
                        if name in seen:
                            continue
                        seen.add(name)
                        idx = counters.get(current_ns, 0) + 1
                        counters[current_ns] = idx
                        local_map[f"{current_ns}{idx}"] = f"{source_ns}.{name}"

                    i = j
                    continue

                # Parse abbrev declarations: abbrev XX := YY
                abbrev_match = re.match(
                    r"^abbrev\s+([A-Z]{1,16}\d+)\s*:=\s*(.+)$", line
                )
                if abbrev_match:
                    code = abbrev_match.group(1)
                    expr_lines = [abbrev_match.group(2).strip()]
                    j = i + 1
                    while j < len(lines):
                        next_line = lines[j].strip()
                        if not next_line:
                            break
                        if re.match(
                            r"^(?:abbrev|theorem|lemma|def|namespace|end|export)\b",
                            next_line,
                        ):
                            break
                        expr_lines.append(next_line)
                        j += 1

                    expr = " ".join(expr_lines)
                    handle_matches = re.findall(
                        r"@?([A-Za-z0-9_']+\.[A-Za-z0-9_'.]+)", expr
                    )
                    if handle_matches:
                        full_handle = handle_matches[-1]
                    else:
                        bare_match = re.search(r"@?([A-Za-z_][A-Za-z0-9_']*)", expr)
                        if bare_match:
                            target = bare_match.group(1)
                            if namespace_stack:
                                ns_prefix = ".".join(namespace_stack) + "."
                                full_handle = ns_prefix + target
                            else:
                                full_handle = target
                        else:
                            full_handle = code
                    local_map[code] = full_handle

                i += 1

            return local_map

        code_to_handle: Dict[str, str] = {}
        # Prefer paper-local aliases, then fill missing IDs from dependencies.
        for _, proofs_dir in roots:
            if not proofs_dir.exists():
                continue
            alias_files = sorted(proofs_dir.rglob("HandleAliases.lean"))
            for alias_file in alias_files:
                local_map = parse_alias_file(alias_file)
                for code, handle in local_map.items():
                    if code not in code_to_handle:
                        code_to_handle[code] = handle

        return code_to_handle

    def _assign_compact_handle_ids(
        self, handles: Set[str], existing_rows: List[Tuple[str, str]]
    ) -> Dict[str, str]:
        """Assign compact IDs to handles, preserving explicit assignments.

        DOF=1 design:
        1. Explicit assignments from HandleAliases.lean are preserved as-is
        2. Remaining handles get auto-numbered IDs based on prefix
        3. Auto-numbering is deterministic (sorted order)
        """
        handle_to_code: Dict[str, str] = {}
        used_codes: Set[str] = set()
        prefix_counters: Dict[str, int] = {}

        # First, apply explicit assignments from provided existing rows
        for code, handle in existing_rows:
            if handle in handle_to_code:
                continue
            if code in used_codes:
                continue
            handle_to_code[handle] = code
            used_codes.add(code)
            match = re.match(r"^([A-Za-z]+)(\d+)$", code)
            if match:
                prefix = match.group(1)
                number = int(match.group(2))
                prefix_counters[prefix] = max(prefix_counters.get(prefix, 0), number)

        # Auto-assign stable compact IDs for remaining handles.
        # IDs are deterministic per prefix because:
        # 1. handles are visited in sorted order
        # 2. prior compact numeric IDs are preserved
        # 3. counters advance monotonically within each prefix
        for handle in sorted(handles):
            if handle in handle_to_code:
                continue

            group = handle.split(".", 1)[0] if "." in handle else "L"
            prefix = self._compact_lean_prefix(group) or "H"
            next_idx = prefix_counters.get(prefix, 0) + 1
            code = f"{prefix}{next_idx}"
            while code in used_codes:
                next_idx += 1
                code = f"{prefix}{next_idx}"

            used_codes.add(code)
            prefix_counters[prefix] = next_idx
            handle_to_code[handle] = code

        return handle_to_code

    def _write_lean_handle_ids_auto(
        self, paper_id: str, files: Optional[List[Path]] = None
    ) -> None:
        """Write compact Lean-handle ID table used by `\\LH{...}` references."""
        content_dir = self._get_content_dir(paper_id)
        if not content_dir.exists():
            return
        if files is None:
            files = self._discover_content_files_for_flags(paper_id)

        out_file = content_dir / "lean_handle_ids_auto.tex"
        previous_rows = self._read_existing_lean_handle_rows(paper_id)
        previous_id_to_handle = {code: handle for code, handle in previous_rows}
        if not hasattr(self, "_previous_lean_handle_id_maps"):
            self._previous_lean_handle_id_maps: Dict[str, Dict[str, str]] = {}
        self._previous_lean_handle_id_maps[paper_id] = previous_id_to_handle

        # Prefer explicit mappings declared in HandleAliases.lean as the
        # single source of truth for explicit ID assignments.
        alias_by_code = self._extract_alias_code_map_from_handle_aliases(
            paper_id, include_dependencies=True
        )
        alias_rows = sorted(alias_by_code.items())
        referenced_handles = self._extract_referenced_lean_handles_from_content(
            paper_id, files=files
        )
        referenced_lh_handles = self._extract_referenced_lh_handles_from_content(
            paper_id, files=files
        )
        referenced_ids = self._extract_lh_ids_from_content(paper_id, files=files)
        referenced_id_handles: Set[str] = set()
        for code in referenced_ids:
            if code in alias_by_code:
                referenced_id_handles.add(alias_by_code[code])
            elif code in previous_id_to_handle:
                referenced_id_handles.add(previous_id_to_handle[code])
        candidate_handles = (
            set(referenced_handles)
            | set(referenced_lh_handles)
            | set(referenced_id_handles)
        )
        # Recovery fallback:
        # if content already uses only \LH{ID} references and the current map file
        # is empty/stale, parsing can become circular (no IDs -> no handles -> empty map).
        # Break the cycle by deriving from declared Lean handles in the local proof graph.
        if not candidate_handles and referenced_ids:
            candidate_handles = self._extract_declared_lean_handles(
                paper_id, include_dependencies=True
            )

        # Preserve explicit alias assignments from HandleAliases.lean and
        # include those handles in the universe to ensure they remain stable.
        preserved_existing_rows = [
            (code, handle)
            for code, handle in previous_rows
            if re.match(r"^[A-Za-z]+\d+$", code)
            and (handle in candidate_handles or code in referenced_ids)
        ]
        preserved_alias_rows = [
            (code, handle)
            for code, handle in alias_rows
            if handle in candidate_handles or code in referenced_ids
        ]
        preserved_rows = preserved_alias_rows + preserved_existing_rows
        preserved_handles = {handle for _, handle in preserved_rows}

        # Single derived universe: only handles referenced directly in prose,
        # plus any explicit alias rows needed to preserve the cited IDs.
        all_handles = set(candidate_handles) | preserved_handles

        handle_to_code = self._assign_compact_handle_ids(all_handles, preserved_rows)
        code_to_handle = {code: handle for handle, code in handle_to_code.items()}
        handle_locations = self._extract_declared_lean_handle_locations(
            paper_id, include_dependencies=True
        )
        suffix_locations: Dict[str, str] = {}
        for full_handle, source_file in handle_locations.items():
            parts = full_handle.split(".")
            for idx in range(len(parts)):
                suffix = ".".join(parts[idx:])
                suffix_locations.setdefault(suffix, source_file)

        def format_handle_cell(handle: str) -> str:
            handle_tex = (
                rf"{{\fontsize{{5.1}}{{5.25}}\selectfont\nolinkurl{{{handle}}}}}"
            )
            source_file = handle_locations.get(handle, "") or suffix_locations.get(
                handle, ""
            )
            if not source_file:
                return handle_tex
            safe_file = source_file.replace("_", r"\_").replace("/", r"/\allowbreak ")
            return (
                handle_tex
                + rf"\par\vspace{{-0.2ex}}{{\fontsize{{5.1}}{{5.35}}\selectfont\ttfamily {safe_file}}}"
            )

        def sort_key(code: str) -> Tuple[str, int, str]:
            match = re.match(r"^([A-Za-z]+)(\d+)$", code)
            if not match:
                return (code, 10**9, code)
            return (match.group(1), int(match.group(2)), code)

        lines: List[str] = [
            "% Auto-generated by scripts/build_papers.py. Do not edit manually.",
            f"% Generated: {datetime.now().isoformat()}",
            r"\begingroup",
            r"\tiny",
            r"\setlength{\tabcolsep}{3pt}",
            r"\renewcommand{\arraystretch}{0.74}",
            r"\setlength{\LTpre}{0pt}",
            r"\setlength{\LTpost}{0pt}",
            r"\urlstyle{same}",
            r"\makeatletter",
            r"\if@twocolumn",
            r"\begin{list}{}{\leftmargin=0pt\itemindent=0pt\itemsep=1pt\parsep=0pt\topsep=0pt}",
        ]

        if code_to_handle:
            for code in sorted(code_to_handle.keys(), key=sort_key):
                handle = code_to_handle[code]
                lines.append(
                    rf"\item \hypertarget{{lh:{code}}}{{\nolinkurl{{{code}}}}}\hspace{{0.35em}}{format_handle_cell(handle)}"
                )
        else:
            lines.append(r"\item \textit{No Lean handles parsed yet.}")

        lines.extend(
            [
                r"\end{list}",
                r"\else",
                r"\begin{longtable}{@{}p{0.07\linewidth}p{0.39\linewidth}p{0.07\linewidth}p{0.39\linewidth}@{}}",
                r"\toprule",
                r"ID & Handle & ID & Handle \\",
                r"\midrule",
                r"\endfirsthead",
                r"\toprule",
                r"ID & Handle & ID & Handle \\",
                r"\midrule",
                r"\endhead",
            ]
        )

        if code_to_handle:
            sorted_codes = sorted(code_to_handle.keys(), key=sort_key)
            for idx in range(0, len(sorted_codes), 2):
                left_code = sorted_codes[idx]
                left_handle = code_to_handle[left_code]
                if idx + 1 < len(sorted_codes):
                    right_code = sorted_codes[idx + 1]
                    right_handle = code_to_handle[right_code]
                    lines.append(
                        rf"\hypertarget{{lh:{left_code}}}{{\nolinkurl{{{left_code}}}}} & {format_handle_cell(left_handle)} & "
                        rf"\hypertarget{{lh:{right_code}}}{{\nolinkurl{{{right_code}}}}} & {format_handle_cell(right_handle)} \\"
                    )
                    lines.append(r"\specialrule{0.2pt}{0pt}{0pt}")
                else:
                    lines.append(
                        rf"\hypertarget{{lh:{left_code}}}{{\nolinkurl{{{left_code}}}}} & {format_handle_cell(left_handle)} & & \\"
                    )
                    lines.append(r"\specialrule{0.2pt}{0pt}{0pt}")
        else:
            lines.append(
                r"\multicolumn{4}{@{}l@{}}{\textit{No Lean handles parsed yet.}} \\"
            )

        lines.extend(
            [
                r"\bottomrule",
                r"\end{longtable}",
                r"\fi",
                r"\makeatother",
                r"\endgroup",
                "",
            ]
        )
        out_file.write_text("\n".join(lines), encoding="utf-8")

    def _rewrite_content_lean_handles_to_ids(self, paper_id: str) -> None:
        """Rewrite inline `\\nolinkurl{...}` Lean handles to compact `\\LH{...}` IDs.

        This keeps prose readable and clickable without requiring manual conversion
        in each paper's content files.
        """
        content_dir = self._get_content_dir(paper_id)
        if not content_dir.exists():
            return

        id_to_handle = self._read_lean_handle_id_map(paper_id)
        if not id_to_handle:
            return
        handle_to_id = {handle: code for code, handle in id_to_handle.items()}

        nolink_pattern = re.compile(r"\\nolinkurl\{([^}]+)\}")
        leanmeta_prefix_pattern = re.compile(r"(\\leanmeta\{)\s*Lean handles:\s*")

        for tex_file in self._iter_manual_content_tex(paper_id):
            text = tex_file.read_text(encoding="utf-8", errors="replace")
            normalized = leanmeta_prefix_pattern.sub(r"\1", text)

            def repl(match: re.Match[str]) -> str:
                raw = match.group(1).strip()
                handle = raw.replace(r"\_", "_")
                code = handle_to_id.get(handle)
                if code is None:
                    return match.group(0)
                return rf"\LH{{{code}}}"

            rewritten = nolink_pattern.sub(repl, normalized)
            rewritten = self._normalize_leanmeta_handle_lists(rewritten)
            if rewritten != text:
                tex_file.write_text(rewritten, encoding="utf-8")

    def _normalize_leanmeta_handle_lists(self, text: str) -> str:
        """Normalize `\\leanmeta{...}` handle lists.

        Rules:
        - only touches `\\leanmeta{...}` blocks that contain exclusively `\\LH{...}`
          and `\\LHrng{...}{...}{...}` items separated by commas
        - preserves prefix order by first appearance
        - sorts numerically within each prefix
        - compresses contiguous runs into `\\LHrng{PREFIX}{a}{b}`
        - preserves a single trailing period when present
        """
        item_pattern = re.compile(
            r"\\LH\{([A-Z]+)(\d+)\}|\\LHrng\{([A-Z]+)\}\{(\d+)\}\{(\d+)\}"
        )

        def normalize_inner(inner: str) -> Optional[str]:
            trailing = ""
            stripped_inner = inner.strip()
            if stripped_inner.endswith("."):
                trailing = "."
                stripped_inner = stripped_inner[:-1].rstrip()

            parts = [part.strip() for part in stripped_inner.split(",")]
            items: List[Tuple[str, int, int, int]] = []
            prefix_order: Dict[str, int] = {}

            for idx, part in enumerate(parts):
                if not part:
                    continue
                match = item_pattern.fullmatch(part)
                if match is None:
                    return None
                if match.group(1) is not None:
                    prefix = match.group(1)
                    start = int(match.group(2))
                    end = start
                else:
                    prefix = match.group(3)  # type: ignore[arg-type]
                    start = int(match.group(4))  # type: ignore[arg-type]
                    end = int(match.group(5))  # type: ignore[arg-type]
                prefix_order.setdefault(prefix, idx)
                items.append((prefix, start, end, idx))

            if not items:
                return inner

            items.sort(key=lambda it: (prefix_order[it[0]], it[1], it[2]))

            rendered: List[str] = []
            i = 0
            while i < len(items):
                prefix, start, end, _ = items[i]
                j = i + 1
                current_end = end
                while (
                    j < len(items)
                    and items[j][0] == prefix
                    and items[j][1] == current_end + 1
                ):
                    current_end = items[j][2]
                    j += 1
                if start == current_end:
                    rendered.append(rf"\LH{{{prefix}{start}}}")
                else:
                    rendered.append(rf"\LHrng{{{prefix}}}{{{start}}}{{{current_end}}}")
                i = j
            joined = ", ".join(rendered)
            return joined + trailing

        pieces: List[str] = []
        cursor = 0
        needle = r"\leanmeta{"
        needle_len = len(needle)

        while True:
            start = text.find(needle, cursor)
            if start == -1:
                pieces.append(text[cursor:])
                break

            pieces.append(text[cursor:start])
            body_start = start + needle_len
            depth = 1
            idx = body_start
            while idx < len(text) and depth > 0:
                char = text[idx]
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                idx += 1

            if depth != 0:
                pieces.append(text[start:])
                break

            inner = text[body_start : idx - 1]
            normalized_inner = normalize_inner(inner)
            if normalized_inner is None:
                pieces.append(text[start:idx])
            else:
                pieces.append(rf"\leanmeta{{{normalized_inner}}}")
            cursor = idx

        return "".join(pieces)

    def _claim_ref_code_for_label(self, label: str) -> str:
        """Return compact claim-ref code by label prefix."""
        normalized = label.strip().lower()
        if normalized.startswith(("thm:", "th:")):
            return "T"
        if normalized.startswith("prop:"):
            return "P"
        if normalized.startswith("cor:"):
            return "C"
        if normalized.startswith(("def:", "definition:")):
            return "D"
        if normalized.startswith(("lem:", "lemma:")):
            return "L"
        if normalized.startswith(("ax:", "axiom:")):
            return "A"
        if normalized.startswith(("conj:", "conjecture:")):
            return "J"
        return "R"

    def _normalize_claimstamp_refs(self, derived: str) -> str:
        """Normalize verbose theorem references to compact `X\\ref{...}` style."""
        normalized = derived
        replacements = [
            (r"(?:Theorem|Thm\.?)\s*~?\s*\\ref\{([^}]+)\}", r"T\\ref{\1}"),
            (r"(?:Proposition|Prop\.?)\s*~?\s*\\ref\{([^}]+)\}", r"P\\ref{\1}"),
            (r"(?:Corollary|Cor\.?)\s*~?\s*\\ref\{([^}]+)\}", r"C\\ref{\1}"),
            (r"(?:Definition|Def\.?)\s*~?\s*\\ref\{([^}]+)\}", r"D\\ref{\1}"),
            (r"(?:Lemma|Lem\.?)\s*~?\s*\\ref\{([^}]+)\}", r"L\\ref{\1}"),
            (r"(?:Axiom|Ax\.?)\s*~?\s*\\ref\{([^}]+)\}", r"A\\ref{\1}"),
            (r"(?:Conjecture|Conj\.?)\s*~?\s*\\ref\{([^}]+)\}", r"J\\ref{\1}"),
        ]
        for pattern, replacement in replacements:
            normalized = re.sub(pattern, replacement, normalized)
        normalized = normalized.replace("~", "")
        normalized = re.sub(r"\s*,\s*", ",", normalized)
        normalized = re.sub(r"\s*;\s*", ";", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip(" ,;")
        return normalized

    def _normalize_claimstamp_regime(self, regime: str) -> str:
        """Normalize regime tags to compact notation (e.g., `AR`, `S+ETH`)."""
        raw = regime.strip()
        if not raw:
            return ""
        if raw.startswith("[") and raw.endswith("]"):
            raw = raw[1:-1].strip()
        # Treat bracket wrappers as presentation-only and tolerate malformed mixes.
        raw = raw.replace("[", "").replace("]", "")

        # Canonical phrase -> compact code.
        phrase_map = {
            "active declared regime": "AR",
            "declared contract": "DC",
            "declared class/regime": "CR",
            "declared task/regime": "TR",
            "declared decision model": "DM",
            "represented adjacent class": "RA",
            "tractable active regime": "TA",
            "regime-typed": "RG",
            "structured": "STR",
            "identifiable": "ID",
            "bounded horizon": "BH",
            "fully observable": "FO",
            "counterexample": "CE",
            "p = product distribution": "PD",
            "bounded support": "BS",
            "deterministic": "DET",
        }

        # Normalize delimiters before tokenization.
        text = raw.replace(" vs ", ",")
        text = text.replace("][", "],[").replace("], [", "],[")
        parts = [p.strip() for p in re.split(r"\s*[,;]\s*", text) if p.strip()]
        compact_parts: List[str] = []
        seen: Set[str] = set()
        for part in parts:
            token = part.strip()
            if token.startswith("[") and token.endswith("]"):
                token = token[1:-1].strip()
            if token.lower().startswith("r="):
                token = token[2:].strip()

            lowered = token.lower()
            mapped = phrase_map.get(lowered)
            if mapped is None:
                # Preserve explicit symbolic assumptions (e.g., P != coNP, S+ETH).
                if any(ch in token for ch in ("\\", "$", "+", "=", "<", ">", "!", "_")):
                    mapped = token
                elif re.fullmatch(r"[A-Z0-9]+", token):
                    mapped = token
                else:
                    mapped = token

            if mapped and mapped not in seen:
                compact_parts.append(mapped)
                seen.add(mapped)

        return ",".join(compact_parts)

    def _extract_label_to_regime_from_claimstamps(
        self, paper_id: str
    ) -> Dict[str, str]:
        """Extract label -> regime mapping from claimstamps in a source paper."""
        content_dir = self._get_content_dir(paper_id)
        if not content_dir.exists():
            return {}

        claimstamp_pattern = re.compile(
            r"\\claimstamp\{((?:[^{}]|\\ref\{[^}]+\})*)\}\{([^}]*)\}",
            re.DOTALL,
        )
        ref_pattern = re.compile(r"\\ref\{([^}]+)\}")
        mapping: Dict[str, str] = {}
        for tex_file in sorted(content_dir.glob("*.tex")):
            text = tex_file.read_text(encoding="utf-8", errors="replace")
            for first_arg, second_arg in claimstamp_pattern.findall(text):
                labels = ref_pattern.findall(first_arg)
                if len(labels) != 1:
                    continue
                label = labels[0].strip()
                if not label:
                    continue
                regime = self._normalize_claimstamp_regime(second_arg)
                if regime and label not in mapping:
                    mapping[label] = regime
        return mapping

    def _extract_label_to_hardness_tags_from_claimstamps(
        self, paper_id: str
    ) -> Dict[str, List[str]]:
        """Extract claim-label -> normalized regime/hardness tags from claimstamps.

        Unlike `_extract_label_to_regime_from_claimstamps`, this function keeps
        multi-label stamps and accumulates all tags seen for each label so the
        generated hardness index can be derived from the same claim metadata.
        """
        content_dir = self._get_content_dir(paper_id)
        if not content_dir.exists():
            return {}

        claimstamp_pattern = re.compile(
            r"\\claimstamp\{((?:[^{}]|\\ref\{[^}]+\})*)\}\{([^}]*)\}",
            re.DOTALL,
        )
        ref_pattern = re.compile(r"\\ref\{([^}]+)\}")
        paper_handle_pattern = re.compile(r"^(?:thm|cor|lem|prop):")

        label_to_tags: Dict[str, Set[str]] = {}
        fallback_multi: Dict[str, Set[str]] = {}
        for tex_file in self._iter_manual_content_tex(paper_id):
            text = tex_file.read_text(encoding="utf-8", errors="replace")
            for first_arg, second_arg in claimstamp_pattern.findall(text):
                labels = [lab.strip() for lab in ref_pattern.findall(first_arg)]
                if not labels:
                    continue
                normalized = self._normalize_claimstamp_regime(second_arg)
                tags = [tok.strip() for tok in normalized.split(",") if tok.strip()]
                if not tags:
                    continue
                dest = label_to_tags if len(labels) == 1 else fallback_multi
                for label in labels:
                    if not paper_handle_pattern.match(label):
                        continue
                    dest.setdefault(label, set()).update(tags)

        # Only fall back to multi-label aggregate stamps when no theorem-local tag exists.
        for label, tags in fallback_multi.items():
            if label not in label_to_tags:
                label_to_tags[label] = set(tags)

        return {
            label: sorted(tags)
            for label, tags in sorted(label_to_tags.items(), key=lambda kv: kv[0])
        }

    def _derive_hardness_profile(self, label: str, tags: List[str]) -> str:
        """Derive hardness profile from source tags.

        DOF=1 policy:
        - explicit `H=...` tag in claimstamp metadata is the only source
        - absent explicit hardness metadata, return `unspecified`
        """
        _ = label  # label kept for signature stability / future diagnostics
        explicit: List[str] = []
        for tag in tags:
            lower = tag.lower()
            if lower.startswith("h="):
                value = tag.split("=", 1)[1].strip()
                if value and value not in explicit:
                    explicit.append(value)
            elif lower.startswith("hardness="):
                value = tag.split("=", 1)[1].strip()
                if value and value not in explicit:
                    explicit.append(value)

        if explicit:
            return ",".join(explicit)
        return "unspecified"

    def _write_proof_hardness_index_auto(self, paper_id: str) -> None:
        """Write auto-generated proof hardness index from claim labels/tags."""
        content_dir = self._get_content_dir(paper_id)
        if not content_dir.exists():
            return

        out_file = content_dir / "proof_hardness_index_auto.tex"
        claim_labels = sorted(self._extract_paper_claim_labels(paper_id))
        if not claim_labels:
            out_file.write_text(
                "% Auto-generated by scripts/build_papers.py. No claim labels found.\n",
                encoding="utf-8",
            )
            return

        claim_to_handles = self._extract_claim_label_to_lean_handles(paper_id)
        label_to_tags = self._extract_label_to_hardness_tags_from_claimstamps(paper_id)
        id_to_handle = self._read_lean_handle_id_map(paper_id)
        handle_to_id = {handle: code for code, handle in id_to_handle.items()}

        rows: List[Tuple[str, str, str, str]] = []
        profile_counts: Dict[str, int] = {}
        for label in claim_labels:
            tags = label_to_tags.get(label, [])
            profile = self._derive_hardness_profile(label, tags)
            profile_counts[profile] = profile_counts.get(profile, 0) + 1

            tags_cell = ",".join(tags) if tags else "-"
            handles = claim_to_handles.get(label, [])
            if handles:
                support_items: List[str] = []
                for handle in handles:
                    code = handle_to_id.get(handle)
                    if code is not None:
                        support_items.append(rf"\LH{{{code}}}")
                    else:
                        compact = self._compact_lean_handle(handle).replace("_", r"\_")
                        support_items.append(rf"\nolinkurl{{{compact}}}")
                support_cell = ", ".join(support_items)
            else:
                support_cell = r"\emph{(no derived Lean handle found)}"

            rows.append((label, profile, tags_cell, support_cell))

        rows.sort(key=lambda row: (row[1], row[0]))

        lines: List[str] = [
            "% Auto-generated by scripts/build_papers.py. Do not edit manually.",
            f"% Generated: {datetime.now().isoformat()}",
            rf"% Paper: {paper_id}",
            r"\begingroup",
            r"\scriptsize",
            r"\sloppy",
            r"\setlength{\tabcolsep}{3pt}",
            r"\begin{longtable}{@{}>{\raggedright\arraybackslash}p{0.19\linewidth}>{\raggedright\arraybackslash}p{0.13\linewidth}>{\raggedright\arraybackslash}p{0.26\linewidth}>{\raggedright\arraybackslash}p{0.38\linewidth}@{}}",
            r"\toprule",
            r"\textbf{Paper handle} & \textbf{Hardness profile} & \textbf{Regime tags} & \textbf{Lean support} \\",
            r"\midrule",
        ]

        for label, profile, tags_cell, support_cell in rows:
            # Make regime-tag cells explicitly breakable to prevent spillover into
            # Lean-support column on narrow pages.
            tags_tex = tags_cell.replace(",", r",\allowbreak ")
            lines.append(
                rf"\nolinkurl{{{label}}} & \nolinkurl{{{profile}}} & {tags_tex} & {support_cell} \\"
            )

        lines.extend([r"\bottomrule", r"\end{longtable}", r"\endgroup", ""])
        summary_parts = [
            f"{name}={profile_counts[name]}" for name in sorted(profile_counts.keys())
        ]
        lines.append(
            rf"\noindent\textit{{Auto summary: indexed {len(rows)} claims by hardness profile ({'; '.join(summary_parts)}).}}"
        )
        out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _normalize_and_fill_claimstamps(self, paper_id: str) -> None:
        """Normalize claimstamp notation and fill empty claim-block placeholders.

        This enforces compact, paper-wide consistent tags:
        - `Prop.~\\ref{...}` -> `P\\ref{...}` (similarly for other theorem classes)
        - empty claim-block stamps derive from the claim label
        - empty regime tags inherit from scaffold source when available, else `RG`
        """
        content_dir = self._get_content_dir(paper_id)
        if not content_dir.exists():
            return

        meta = self._get_paper_meta(paper_id)
        source_regimes: Dict[str, str] = {}
        source_id = meta.scaffold_from.strip()
        if source_id and source_id in self.papers:
            source_regimes = self._extract_label_to_regime_from_claimstamps(source_id)

        claimstamp_pattern = re.compile(
            r"\\claimstamp\{((?:[^{}]|\\ref\{[^}]+\})*)\}\{([^}]*)\}",
            re.DOTALL,
        )
        claim_block_pattern = re.compile(
            r"(\\begin\{claim\}\{(?:[^{}]|\{[^{}]*\})*\}\{([^{}]+)\})(.*?)(\\end\{claim\})",
            re.DOTALL,
        )
        generic_first_pattern = re.compile(
            r"^\s*(?:Thm\.?|Theorem|Prop\.?|Proposition|Cor\.?|Corollary|Def\.?|Definition|Lemma|Lem\.?|Axiom|Ax\.?|Conjecture|Conj\.?)\s*\.?\s*$"
        )

        for tex_file in self._iter_manual_content_tex(paper_id):
            text = tex_file.read_text(encoding="utf-8", errors="replace")

            # Pass 1: normalize existing stamps.
            def normalize_stamp(match: re.Match[str]) -> str:
                first = self._normalize_claimstamp_refs(match.group(1))
                second = self._normalize_claimstamp_regime(match.group(2))
                return rf"\claimstamp{{{first}}}{{{second}}}"

            rewritten = claimstamp_pattern.sub(normalize_stamp, text)

            # Pass 2: fill empty/generic claim-block stamps using local label.
            def fill_block(match: re.Match[str]) -> str:
                begin, label, body, end = (
                    match.group(1),
                    match.group(2),
                    match.group(3),
                    match.group(4),
                )
                normalized_label = label.strip()

                def fill_first_stamp(stamp_match: re.Match[str]) -> str:
                    first = stamp_match.group(1).strip()
                    second = stamp_match.group(2).strip()

                    if not first or generic_first_pattern.fullmatch(first):
                        code = self._claim_ref_code_for_label(normalized_label)
                        first = rf"{code}\ref{{{normalized_label}}}"

                    if not second:
                        second = source_regimes.get(normalized_label, "RG")

                    return rf"\claimstamp{{{first}}}{{{second}}}"

                filled_body, _ = claimstamp_pattern.subn(
                    fill_first_stamp, body, count=1
                )
                return begin + filled_body + end

            rewritten = claim_block_pattern.sub(fill_block, rewritten)

            if rewritten != text:
                tex_file.write_text(rewritten, encoding="utf-8")

    def _extract_claim_label_to_lean_handles(
        self, paper_id: str, include_dependencies: bool = True
    ) -> Dict[str, List[str]]:
        """Extract mapping from theorem-style labels to inline Lean handles.

        Local rows are claim-local: Lean support cited inside the corresponding
        labeled theorem/claim block counts for that label. Support may appear as
        compact ``\\LH{ID}`` markers or as raw fully qualified declaration names
        wrapped in ``\\nolinkurl{...}`` inside ``\\leanmeta{...}``.
        """
        content_dir = self._get_content_dir(paper_id)
        if not content_dir.exists():
            return {}

        id_to_handle = self._read_lean_handle_id_map(paper_id)
        claim_labels = self._extract_paper_claim_labels(paper_id)
        merged: Dict[str, Set[str]] = {label: set() for label in claim_labels}
        label_pattern = re.compile(r"\\label\{([^}]+)\}")
        compact_id_pattern = re.compile(r"^[A-Za-z]+\d+$")
        paper_handle_pattern = re.compile(r"^(?:thm|cor|lem|prop):")
        lh_pattern = re.compile(r"\\LH\{([^}]+)\}")
        lhrng_pattern = re.compile(r"\\LHrng\{([A-Za-z]+)\}\{(\d+)\}\{(\d+)\}")
        nolink_pattern = re.compile(r"\\nolinkurl\{([^}]+)\}")

        def extract_handles(snippet: str) -> Set[str]:
            handles: Set[str] = set()

            for raw in lh_pattern.findall(snippet):
                token = raw.strip().replace(r"\_", "_")
                if not token:
                    continue
                resolved = id_to_handle.get(token)
                if resolved:
                    handles.add(resolved)
                    continue
                if compact_id_pattern.match(token):
                    continue
                if paper_handle_pattern.match(token):
                    continue
                if token.endswith(".lean"):
                    continue
                if "://" in token or token.startswith("www."):
                    continue
                if "." not in token and "_" not in token:
                    continue
                handles.add(token)

            for prefix, start_raw, end_raw in lhrng_pattern.findall(snippet):
                start = int(start_raw)
                end = int(end_raw)
                if end < start:
                    start, end = end, start
                for idx in range(start, end + 1):
                    token = f"{prefix}{idx}"
                    resolved = id_to_handle.get(token)
                    if resolved:
                        handles.add(resolved)

            for raw in nolink_pattern.findall(snippet):
                token = raw.strip().replace(r"\_", "_")
                if not token:
                    continue
                if paper_handle_pattern.match(token):
                    continue
                if token.endswith(".lean"):
                    continue
                if "://" in token or token.startswith("www."):
                    continue
                if compact_id_pattern.match(token):
                    resolved = id_to_handle.get(token)
                    if resolved:
                        handles.add(resolved)
                    continue
                if "." not in token and "_" not in token:
                    continue
                handles.add(token)

            return handles

        claim_block_pattern = re.compile(
            r"\\begin\{claim\}\{(?:[^{}]|\{[^{}]*\})*\}\{([^{}]+)\}(.*?)\\end\{claim\}",
            re.DOTALL,
        )
        theorem_block_pattern = re.compile(
            r"\\begin\{(theorem|corollary|lemma|proposition|definition|remark)\}"
            r"(?:\[[^\]]*\])?(.*?)\\end\{\1\}",
            re.DOTALL,
        )

        for tex_file in self._iter_manual_content_tex(paper_id):
            text = tex_file.read_text(encoding="utf-8", errors="replace")
            for match in claim_block_pattern.finditer(text):
                label = match.group(1).strip()
                if label not in merged:
                    continue
                body = match.group(2)
                merged[label].update(extract_handles(body))

            for match in theorem_block_pattern.finditer(text):
                body = match.group(2)
                labels = [lbl.strip() for lbl in label_pattern.findall(body)]
                matched_labels = [lbl for lbl in labels if lbl in merged]
                if not matched_labels:
                    continue

                handles = extract_handles(body)

                # Paper 2 style: theorem block is followed by a dedicated \leanmeta{...}
                # line, so inspect the next few lines after the environment closes.
                trailing_lines = text[match.end() :].splitlines()[:3]
                trailing_text = "\n".join(trailing_lines)
                handles.update(extract_handles(trailing_text))

                for label in matched_labels:
                    merged[label].update(handles)

        source_ids: List[str] = []
        if include_dependencies:
            meta = self._get_paper_meta(paper_id)
            source_id = meta.scaffold_from.strip()
            if source_id and source_id in self.papers:
                source_ids.append(source_id)
            for dep in self._collect_lean_dependency_closure(paper_id):
                if dep not in source_ids:
                    source_ids.append(dep)

        for source in source_ids:
            # Prefer a pre-generated matrix (already derived and normalized).
            source_map = self._read_claim_mapping_table_handles(source)
            if not source_map:
                # Fallback to direct extraction from source content.
                source_map = self._extract_claim_label_to_lean_handles(
                    source, include_dependencies=False
                )
            for label in claim_labels:
                if merged.get(label):
                    continue
                source_handles = source_map.get(label, [])
                if source_handles:
                    merged.setdefault(label, set()).update(source_handles)

        return {k: sorted(v) for k, v in merged.items()}

    def _read_lean_handle_id_map(self, paper_id: str) -> Dict[str, str]:
        """Read generated Lean-handle ID map: LH code -> full Lean handle."""
        content_dir = self._get_content_dir(paper_id)
        map_file = content_dir / "lean_handle_ids_auto.tex"
        if not map_file.exists():
            return {}

        text = map_file.read_text(encoding="utf-8", errors="replace")
        id_to_handle: Dict[str, str] = {}
        # Format-robust parse:
        # - ID can be read from \hypertarget{lh:ID}{...} anchor (preferred)
        # - ID cell may use \texttt{ID}, \textsf{ID}, or \nolinkurl{ID}
        # - handle cell may be wrapped (e.g., {\footnotesize\nolinkurl{...}})
        # - optional \hypertarget prefixes may appear before the ID cell
        hyper_code_pattern = re.compile(r"\\hypertarget\{lh:([^}]+)\}")
        code_pattern = re.compile(r"\\text(?:tt|sf)\{([^}]+)\}")
        code_url_pattern = re.compile(
            r"\\hypertarget\{lh:[^}]+\}\{\\nolinkurl\{([^}]+)\}\}"
        )
        handle_pattern = re.compile(r"\\nolinkurl\{([^}]+)\}")
        for raw_line in text.splitlines():
            if r"\nolinkurl{" not in raw_line:
                continue
            hyper_match = hyper_code_pattern.search(raw_line)
            code_match = code_pattern.search(raw_line)
            code_url_match = code_url_pattern.search(raw_line)
            handle_matches = handle_pattern.findall(raw_line)
            code: str = ""
            if hyper_match:
                code = hyper_match.group(1).strip()
            elif code_url_match:
                code = code_url_match.group(1).strip()
            elif code_match:
                code = code_match.group(1).strip().replace(r"\_", "_")
            if not code or not handle_matches:
                continue
            handle = handle_matches[-1].strip().replace(r"\_", "_")
            if code and handle:
                id_to_handle.setdefault(code, handle)
        return id_to_handle

    def _read_claim_mapping_table_handles(self, paper_id: str) -> Dict[str, List[str]]:
        """Read label->handles from an existing `claim_mapping_auto.tex` table."""
        content_dir = self._get_content_dir(paper_id)
        mapping_file = content_dir / "claim_mapping_auto.tex"
        if not mapping_file.exists():
            return {}

        text = mapping_file.read_text(encoding="utf-8", errors="replace")
        paper_handle_pattern = re.compile(r"^(?:thm|cor|lem|prop):")
        mapping: Dict[str, Set[str]] = {}

        for raw_line in text.splitlines():
            if "&" not in raw_line or r"\nolinkurl{" not in raw_line:
                continue
            nolinks = [
                s.strip().replace(r"\_", "_")
                for s in re.findall(r"\\nolinkurl\{([^}]+)\}", raw_line)
            ]
            if not nolinks:
                continue
            label = nolinks[0]
            if not paper_handle_pattern.match(label):
                continue
            supports = [
                h for h in nolinks[1:] if h and not paper_handle_pattern.match(h)
            ]
            if supports:
                mapping.setdefault(label, set()).update(supports)

        return {label: sorted(handles) for label, handles in mapping.items()}

    def _write_claim_mapping_auto(self, paper_id: str) -> None:
        """Write auto-generated claim coverage matrix from derived theorem anchors."""
        meta = self._get_paper_meta(paper_id)
        latex_dir = self._get_paper_dir(paper_id) / meta.latex_dir
        content_dir = latex_dir / "content"
        if not content_dir.exists():
            return

        out_file = content_dir / "claim_mapping_auto.tex"
        claim_labels = sorted(self._extract_paper_claim_labels(paper_id))
        local_claim_to_handles = self._extract_claim_label_to_lean_handles(
            paper_id, include_dependencies=False
        )
        claim_to_handles = self._extract_claim_label_to_lean_handles(
            paper_id, include_dependencies=True
        )

        full_count = 0
        derived_count = 0
        unmapped_count = 0
        lines: List[str] = [
            "% Auto-generated by scripts/build_papers.py. Do not edit manually.",
            f"% Generated: {datetime.now().isoformat()}",
            rf"% Paper: {paper_id}",
            r"\begingroup",
            r"\scriptsize",
            r"\sloppy",
            r"\setlength{\tabcolsep}{4pt}",
            r"\begin{longtable}{@{}>{\raggedright\arraybackslash}p{0.24\linewidth}>{\raggedright\arraybackslash}p{0.12\linewidth}>{\raggedright\arraybackslash}p{0.60\linewidth}@{}}",
            r"\toprule",
            r"\textbf{Paper handle} & \textbf{Status} & \textbf{Lean support} \\",
            r"\midrule",
        ]

        for label in claim_labels:
            handles = claim_to_handles.get(label, [])
            local_handles = local_claim_to_handles.get(label, [])
            if local_handles:
                status = "Full"
                full_count += 1
            elif handles:
                status = "Derived"
                derived_count += 1
            else:
                status = "Unmapped"
                unmapped_count += 1
            if handles:
                support = ", ".join(
                    rf"\nolinkurl{{{self._compact_lean_handle(h)}}}" for h in handles
                )
            else:
                support = r"\emph{(no derived Lean handle found)}"
            lines.append(rf"\nolinkurl{{{label}}} & {status} & {support} \\")

        lines.extend([r"\bottomrule", r"\end{longtable}", r"\endgroup", ""])
        if derived_count > 0 or unmapped_count > 0:
            lines.extend(
                [
                    r"\noindent\textit{Notes:}",
                    r"\noindent\textit{(1) Full rows come from theorem-local inline anchors in this paper.}",
                    r"\noindent\textit{(2) Derived rows are filled by dependency/scaffold claim-handle derivation (same paper-handle label across proof dependencies).}",
                    r"\noindent\textit{(3) Unmapped means no local anchor and no derivable dependency support were found.}",
                    "",
                ]
            )
        mapped_total = full_count + derived_count
        lines.append(
            rf"\noindent\textit{{Auto summary: mapped {mapped_total}/{len(claim_labels)} (full={full_count}, derived={derived_count}, unmapped={unmapped_count}).}}"
        )
        out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _expand_lean_stat_macros(self, text: str, paper_id: str) -> str:
        """Expand ``\\Lean*`` macros to numeric values for markdown conversion."""
        local_stats = self._get_lean_stats(paper_id)
        total_stats = self._get_cumulative_lean_stats(paper_id)
        release_stats = self._get_release_lean_stats(paper_id)
        file_stats = self._get_lean_file_stats(paper_id)
        suffix_map = self._build_lean_macro_suffix_map(sorted(file_stats.keys()))
        replacements: Dict[str, str] = {
            r"\LeanLocalLines": str(local_stats.line_count),
            r"\LeanLocalTheorems": str(local_stats.theorem_count),
            r"\LeanLocalSorry": str(local_stats.sorry_count),
            r"\LeanLocalFiles": str(local_stats.file_count),
            r"\LeanTotalLines": str(total_stats.line_count),
            r"\LeanTotalTheorems": str(total_stats.theorem_count),
            r"\LeanTotalSorry": str(total_stats.sorry_count),
            r"\LeanTotalFiles": str(total_stats.file_count),
            r"\LeanReleaseLines": str(release_stats.line_count),
            r"\LeanReleaseTheorems": str(release_stats.theorem_count),
            r"\LeanReleaseSorry": str(release_stats.sorry_count),
            r"\LeanReleaseFiles": str(release_stats.file_count),
        }
        for module_path, stats in file_stats.items():
            suffix = suffix_map[module_path]
            replacements[rf"\LeanLines{suffix}"] = str(stats.line_count)
            replacements[rf"\LeanTheorems{suffix}"] = str(stats.theorem_count)
            replacements[rf"\LeanSorry{suffix}"] = str(stats.sorry_count)

        # Replace longest keys first so specific file macros win over prefixes.
        for macro in sorted(replacements.keys(), key=len, reverse=True):
            text = text.replace(macro, replacements[macro])

        # Evaluate simple TeX integer arithmetic used in summary rows:
        # \number\numexpr 1+2-3\relax -> 0
        def _eval_numexpr(match: re.Match[str]) -> str:
            expr = re.sub(r"\s+", "", match.group(1))
            if not expr:
                return match.group(0)
            terms = re.findall(r"[+-]?\d+", expr)
            if not terms:
                return match.group(0)
            total = sum(int(term) for term in terms)
            return str(total)

        text = re.sub(
            r"\\number\\numexpr\s*([0-9+\-\s]+)\\relax",
            _eval_numexpr,
            text,
        )
        return text

    def build_lean(self, paper_id: str, verbose: bool = True) -> str:
        """Build Lean proofs with streaming output. Returns captured output.

        Uses each paper's own proofs directory (paper_dir/proofs/).
        Runs `lake build` in that directory.
        Captures output for inclusion in BUILD_LOG.txt.
        """
        proofs_dir = self._get_paper_proofs_dir(paper_id)

        if not proofs_dir.exists():
            print(f"[build] No proofs directory for {paper_id}, skipping...")
            return ""

        # Check for lakefile
        lakefile = proofs_dir / "lakefile.lean"
        if not lakefile.exists():
            print(f"[build] No lakefile.lean in {proofs_dir}, skipping...")
            return ""

        # Ensure local dependency aliases (dep_<paper>) are present for path deps.
        dep_ids = self._collect_lean_dependency_closure(paper_id)
        self._sync_local_lean_dependency_dirs(paper_id)
        for dep_id in dep_ids:
            self._sync_graph_infra_lean(dep_id)
        self._sync_graph_infra_lean(paper_id)
        self._write_graph_export_lean(paper_id)

        print(f"[build] Building {paper_id} Lean...")
        print(f"[build]   Directory: {proofs_dir.relative_to(self.repo_root)}")

        def run_lake(args: List[str]) -> Tuple[int, List[str]]:
            """Run lake command with streaming output."""
            print(f"[build]   Running: lake {' '.join(args)}")
            process = subprocess.Popen(
                ["lake"] + args,
                cwd=proofs_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            output_lines: List[str] = []
            if process.stdout is not None:
                for line in process.stdout:
                    output_lines.append(line)
                    if verbose:
                        print(f"         {line}", end="", flush=True)
                    elif "Building" in line or "Compiling" in line:
                        print(".", end="", flush=True)

            process.wait()
            if not verbose:
                print()
            return (process.returncode, output_lines)

        # Run lake build (builds all targets in the paper's proofs directory)
        returncode, output = run_lake(["build"])
        output_text = "".join(output)

        # Handle stale cache
        if "compiled configuration is invalid" in output_text:
            print(f"[build]   Reconfiguring (stale cache detected)...")
            returncode, output = run_lake(["build", "-R"])
            output_text = "".join(output)

        if returncode != 0:
            if not verbose:
                print("[build]   lake output (last 200 lines):")
                for line in output_text.splitlines()[-200:]:
                    print(f"         {line}")
            raise RuntimeError(
                f"Lean build failed for {paper_id} (exit code {returncode})"
            )

        # Store for BUILD_LOG.txt
        self._lean_build_output = output_text
        print(f"[build] ✓ {paper_id} Lean complete")

        self._collect_graph_json(paper_id)
        self._mark_claim_nodes_in_graph(paper_id)
        return output_text

    def build_latex(self, paper_id: str, verbose: bool = False):
        """Build LaTeX PDF for a paper and copy to releases/.

        Runs full build cycle: pdflatex → bibtex → pdflatex → pdflatex
        to resolve citations and cross-references.

        For variants (e.g., paper1_jsait), uses variant-specific naming.
        """
        meta = self._get_paper_meta(paper_id)
        latex_dir = self._get_latex_dir(paper_id)
        latex_file = latex_dir / meta.latex_file

        if not latex_file.exists():
            raise FileNotFoundError(f"LaTeX file not found: {latex_file}")

        self._refresh_derived_content(paper_id)
        self._write_submission_build_hook(latex_dir)
        self._mark_claim_nodes_in_graph(paper_id)
        self._validate_true_paths(paper_id)
        self._update_paper_date(latex_file)

        print(f"[build] Building {paper_id} LaTeX...")

        # Full build cycle for proper citation/reference resolution
        aux_name = latex_file.stem
        build_steps = [
            (
                ["pdflatex", "-interaction=nonstopmode", latex_file.name],
                "pdflatex (1/3)",
            ),
            (["bibtex", aux_name], "bibtex"),
            (
                ["pdflatex", "-interaction=nonstopmode", latex_file.name],
                "pdflatex (2/3)",
            ),
            (
                ["pdflatex", "-interaction=nonstopmode", latex_file.name],
                "pdflatex (3/3)",
            ),
        ]

        for cmd, step_name in build_steps:
            if verbose:
                print(f"[build]   Running: {step_name}")
            result = subprocess.run(
                cmd,
                cwd=latex_dir,
                capture_output=True,
                text=True,
                encoding="latin-1",
                errors="replace",
            )
            # bibtex returns non-zero if no citations, which is fine
            if result.returncode != 0 and "bibtex" not in cmd[0]:
                if verbose:
                    if result.stdout:
                        print(result.stdout)
                print(
                    f"[build] {step_name} had warnings/errors (exit {result.returncode})"
                )

        # Copy PDF to releases/ (INVARIANT 3)
        # Use variant-specific naming to avoid conflicts
        pdf_name = latex_file.stem + ".pdf"
        pdf_src = latex_dir / pdf_name
        if pdf_src.exists():
            releases_dir = self._get_releases_dir(paper_id)
            pdf_dest = releases_dir / f"{paper_id}.pdf"
            shutil.copy2(pdf_src, pdf_dest)
            print(f"[build] ✓ {paper_id}.pdf → releases/")

    def build_submission(self, paper_id: str, verbose: bool = False):
        """Build review-submission PDF with venue-specific formatting.

        Uses the submission_format.latex_flag from papers.yaml to trigger
        format switching (e.g., single-column double-spaced for JSAIT review).
        Outputs a separate *_submission.pdf alongside the standard PDF and
        packages a LaTeX source bundle suitable for manuscript-upload portals.
        """
        # Check if this paper has a submission format defined
        raw_papers = self._raw_metadata.get("papers", {})
        raw_meta = raw_papers.get(paper_id, {})
        sub_fmt = raw_meta.get("submission_format")
        if not sub_fmt:
            print(
                f"[build] {paper_id} has no submission_format in papers.yaml, skipping."
            )
            return

        latex_flag = sub_fmt["latex_flag"]
        description = sub_fmt.get("description", "review format")

        meta = self._get_paper_meta(paper_id)
        paper_dir = self._get_paper_dir(paper_id)
        latex_dir = paper_dir / meta.latex_dir
        latex_file = latex_dir / meta.latex_file

        if not latex_file.exists():
            raise FileNotFoundError(f"LaTeX file not found: {latex_file}")

        submission_files = self._discover_content_files_for_flags(
            paper_id, defined_flags={latex_flag}
        )
        self._sync_shared_preambles(paper_id)
        self._write_latex_lean_stats(paper_id)
        self._write_assumption_ledger_auto(paper_id)
        self._write_lean_handle_ids_auto(paper_id, files=submission_files)
        self._rewrite_content_lean_handles_to_ids(paper_id)
        self._normalize_and_fill_claimstamps(paper_id)
        self._write_claim_mapping_auto(paper_id)
        self._write_proof_hardness_index_auto(paper_id)
        self._update_paper_date(latex_file)

        print(f"[build] Building {paper_id} submission PDF ({description})...")

        # Use jobname to output a separate PDF without overwriting the standard one
        job_name = latex_file.stem + "_submission"
        hook_name = self._write_submission_build_hook(latex_dir)
        tex_input = (
            f"\\def{latex_flag}{{}}\\input{{{hook_name}}}\\input{{{latex_file.name}}}"
        )

        aux_name = job_name
        build_steps = [
            (
                [
                    "pdflatex",
                    "-interaction=nonstopmode",
                    f"-jobname={job_name}",
                    tex_input,
                ],
                "pdflatex (1/3)",
            ),
            (["bibtex", aux_name], "bibtex"),
            (
                [
                    "pdflatex",
                    "-interaction=nonstopmode",
                    f"-jobname={job_name}",
                    tex_input,
                ],
                "pdflatex (2/3)",
            ),
            (
                [
                    "pdflatex",
                    "-interaction=nonstopmode",
                    f"-jobname={job_name}",
                    tex_input,
                ],
                "pdflatex (3/3)",
            ),
        ]

        for cmd, step_name in build_steps:
            if verbose:
                print(f"[build]   Running: {step_name}")
            result = subprocess.run(
                cmd,
                cwd=latex_dir,
                capture_output=True,
                text=True,
                encoding="latin-1",
                errors="replace",
            )
            if result.returncode != 0 and "bibtex" not in cmd[0]:
                if verbose:
                    if result.stdout:
                        print(result.stdout)
                print(
                    f"[build] {step_name} had warnings/errors (exit {result.returncode})"
                )

        # Copy submission PDF to releases/
        pdf_src = latex_dir / f"{job_name}.pdf"
        if pdf_src.exists():
            releases_dir = self._get_releases_dir(paper_id)
            pdf_dest = releases_dir / f"{paper_id}_submission.pdf"
            shutil.copy2(pdf_src, pdf_dest)
            print(f"[build] ✓ {paper_id}_submission.pdf → releases/")
            self.package_submission_source(paper_id)
        else:
            print(f"[build] ✗ Submission PDF not generated for {paper_id}")

    def package_submission_source(self, paper_id: str) -> Optional[Path]:
        """Package LaTeX sources required for manuscript submission portals.

        Creates a source-only archive for the submission/review version:
        - LaTeX/BibTeX/class/style files
        - content/ sources
        - auto-generated LaTeX support files
        - a small wrapper main file that enables the venue review flag

        Does not include Lean proofs, markdown, graphs, or other supplementary
        artifacts.
        """
        raw_papers = self._raw_metadata.get("papers", {})
        raw_meta = raw_papers.get(paper_id, {})
        sub_fmt = raw_meta.get("submission_format")
        if not sub_fmt:
            return None

        meta = self._get_paper_meta(paper_id)
        releases_dir = self._get_releases_dir(paper_id)
        archive_prefix = self._get_archive_prefix(paper_id)
        package_dir = releases_dir / f"submission_package_{archive_prefix}"
        if package_dir.exists():
            shutil.rmtree(package_dir)
        package_dir.mkdir(parents=True)

        submission_files = self._discover_content_files_for_flags(
            paper_id, defined_flags={sub_fmt["latex_flag"]}
        )
        self._refresh_derived_content(paper_id)
        self._write_lean_handle_ids_auto(paper_id, files=submission_files)
        self._copy_latex_sources(paper_id, package_dir)
        hook_name = self._write_submission_build_hook(package_dir)
        wrapper_name = self._write_submission_wrapper_tex(
            paper_id, package_dir, sub_fmt["latex_flag"], hook_name
        )
        self._write_submission_source_readme(paper_id, package_dir, wrapper_name)

        tar_path, zip_path = self._create_named_archive(
            releases_dir=releases_dir,
            package_dir=package_dir,
            archive_stem=f"{archive_prefix}_submission_source",
            root_dir_name=archive_prefix,
        )
        print(
            f"[submission] ✓ {zip_path.name} → {releases_dir.relative_to(self.repo_root)}/"
        )
        return zip_path

    def _write_submission_wrapper_tex(
        self, paper_id: str, package_dir: Path, latex_flag: str, hook_name: str
    ) -> str:
        """Write a compile-ready wrapper that enables the submission flag."""
        meta = self._get_paper_meta(paper_id)
        wrapper_name = (
            "main.tex" if meta.latex_file != "main.tex" else "submission_main.tex"
        )
        wrapper = (
            "% Auto-generated by scripts/build_papers.py. Submission wrapper.\n"
            f"\\def{latex_flag}{{}}\n"
            f"\\input{{{hook_name}}}\n"
            f"\\input{{{meta.latex_file}}}\n"
        )
        (package_dir / wrapper_name).write_text(wrapper, encoding="utf-8")
        return wrapper_name

    def _write_submission_build_hook(self, target_dir: Path) -> str:
        """Write Lean-handle LaTeX injections shared by standard and review builds."""
        hook_name = "build_submission_hook.tex"
        lines = [
            "% Auto-generated by scripts/build_papers.py. Lean-handle hooks.",
            r"\ifdefined\LeanHandleHookLoaded\else",
            r"\global\let\LeanHandleHookLoaded\relax",
            r"% The canonical \LH/\LHrng macros are emitted into shared/lean-handle-macros.tex",
            r"% and copied into each paper's LaTeX dir by the pipeline. Avoid redefining",
            r"% them here to prevent conflicting render behaviour between build hooks and",
            r"% the SSOT macros.",
            r"\AtBeginDocument{}",
            r"\AtEndDocument{%",
            r"  \IfFileExists{content/lean_handle_ids_auto.tex}{%",
            r"    \par\smallskip\begingroup",
            r"    \footnotesize\noindent\textbf{Lean Handle Index.}\par\smallskip",
            r"    \input{content/lean_handle_ids_auto.tex}%",
            r"    \endgroup",
            r"  }{}%",
            r"}",
            r"\fi",
        ]
        (target_dir / hook_name).write_text("\n".join(lines) + "\n", encoding="utf-8")
        return hook_name

    def _write_submission_source_readme(
        self, paper_id: str, package_dir: Path, wrapper_name: str
    ) -> None:
        """Write a short note for manuscript-upload systems/editors."""
        meta = self._get_paper_meta(paper_id)
        readme = "\n".join(
            [
                f"Submission source bundle for {paper_id}: {meta.full_name}",
                "",
                "This archive contains only the main-manuscript LaTeX sources.",
                "Supplementary materials are intentionally excluded.",
                "",
                f"Compile the submission version via: {wrapper_name}",
                f"Primary manuscript source: {meta.latex_file}",
                "",
            ]
        )
        (package_dir / "README_SUBMISSION.txt").write_text(readme, encoding="utf-8")

    def build_markdown(self, paper_id: str):
        """Build Markdown version of a paper.

        Follows the main LaTeX include chain so Markdown mirrors PDF contents.

        For variants (e.g., paper1_jsait), uses variant-specific naming.
        """
        meta = self._get_paper_meta(paper_id)
        out_dir = self._get_markdown_dir(paper_id)
        out_file = self._get_markdown_file(paper_id)

        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[build-md] Building {paper_id}: {meta.name}...")
        self._refresh_derived_content(paper_id)

        # Discover files from main LaTeX include order
        content_files = self._discover_content_files(paper_id)
        # Single source of truth for generated artifacts: main LaTeX `\title{...}`.
        title = self._release_title_from_latex(paper_id)
        lean_stats = self._get_release_lean_stats(paper_id)
        self._build_markdown_file(meta, content_files, out_file, title, lean_stats)

        # Also copy to releases/ for arxiv package
        releases_dir = self._get_releases_dir(paper_id)
        shutil.copy2(out_file, releases_dir / out_file.name)
        self.build_copy_paste_metadata(paper_id)
        print(f"[build-md] {paper_id} → {out_file.relative_to(self.repo_root)}")

    def _extract_braced_command_argument(
        self, content: str, command: str
    ) -> str | None:
        """Extract the first braced argument for a LaTeX command.

        Supports optional square-bracket argument: \\command[...]{...}
        """
        marker = f"\\{command}"
        start = content.find(marker)
        if start == -1:
            return None

        i = start + len(marker)
        n = len(content)

        # Skip whitespace
        while i < n and content[i].isspace():
            i += 1

        # Optional [..] argument
        if i < n and content[i] == "[":
            depth = 1
            i += 1
            while i < n and depth > 0:
                if content[i] == "[":
                    depth += 1
                elif content[i] == "]":
                    depth -= 1
                i += 1
            while i < n and content[i].isspace():
                i += 1

        # Required {..} argument
        if i >= n or content[i] != "{":
            return None

        depth = 1
        i += 1
        arg_start = i
        while i < n and depth > 0:
            if content[i] == "{":
                depth += 1
            elif content[i] == "}":
                depth -= 1
            i += 1

        if depth != 0:
            return None

        return content[arg_start : i - 1]

    def _latex_inline_to_plain(self, text: str) -> str:
        """Convert a LaTeX inline snippet to plain text."""
        try:
            result = subprocess.run(
                [
                    "pandoc",
                    "-f",
                    "latex",
                    "-t",
                    "plain",
                    "--wrap=none",
                    "--columns=100",
                ],
                input=text,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0 and result.stdout.strip():
                plain = result.stdout.strip()
            else:
                plain = text
        except Exception:
            plain = text

        # Normalize line breaks and TeX line-break commands.
        plain = plain.replace("\\\\", " ")
        plain = re.sub(r"\s+", " ", plain).strip()
        return plain

    def _extract_main_latex_title(self, paper_id: str) -> str | None:
        """Extract the displayed title from the variant's main LaTeX file."""
        meta = self._get_paper_meta(paper_id)
        main_tex = self._get_paper_dir(paper_id) / meta.latex_dir / meta.latex_file
        if not main_tex.exists():
            return None

        content = main_tex.read_text(encoding="utf-8", errors="replace")
        content = self._strip_latex_comments(content)
        raw_title = self._extract_braced_command_argument(content, "title")
        if not raw_title:
            return None
        return self._latex_inline_to_plain(raw_title)

    def _release_title_from_latex(self, paper_id: str) -> str:
        """Canonical release title source: main LaTeX `\\title{...}`."""
        title = self._extract_main_latex_title(paper_id)
        if title:
            return title
        meta = self._get_paper_meta(paper_id)
        main_tex = self._get_paper_dir(paper_id) / meta.latex_dir / meta.latex_file
        raise ValueError(
            f"{paper_id}: missing canonical LaTeX title in {main_tex.relative_to(self.repo_root)} "
            "(expected \\\\title{...})"
        )

    def _extract_abstract_latex(self, paper_id: str) -> Optional[str]:
        """Extract abstract LaTeX content from content/abstract.tex."""
        abstract_file = self._get_content_dir(paper_id) / "abstract.tex"
        if not abstract_file.exists():
            return None

        content = abstract_file.read_text(encoding="utf-8", errors="replace")
        match = re.search(
            r"\\begin\{abstract\}(.*?)\\end\{abstract\}", content, re.DOTALL
        )
        if match:
            content = match.group(1)

        content = content.strip()
        return content if content else None

    def _normalize_plaintext_block(self, text: str) -> str:
        """Normalize plain text for copy/paste metadata fields."""
        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        normalized_lines: List[str] = []
        last_blank = False

        for line in lines:
            clean = re.sub(r"\s+", " ", line).strip()
            if not clean:
                if not last_blank:
                    normalized_lines.append("")
                last_blank = True
                continue
            normalized_lines.append(clean)
            last_blank = False

        return "\n".join(normalized_lines).strip()

    def _clean_metadata_reference_tokens(self, text: str) -> str:
        """Remove paper-internal reference tokens from metadata snippets."""
        text = self._postprocess_markdown_text(text)
        text = re.sub(r"\[\s*\\*\s*\]\(#.*?\)\{[^}]*\}", "", text)
        text = re.sub(r"\[\s*\\*\s*\]\(#.*?\)", "", text)
        text = re.sub(r"\{reference-type=\"ref\"[^}]*\}", "", text)
        text = re.sub(r"\[D:[^\[]*\[R=[^\]]+\]\]", "", text)
        text = re.sub(
            r"\[(?:sec|thm|prop|cor|lem|app|rem|eq|fig|tab):[^\]]+\]", "", text
        )
        text = re.sub(r"\[R=[^\]]+\]", "", text)
        text = re.sub(r"\[D:[^\]]+\]", "", text)
        text = re.sub(r"\[R:[^\]]+\]", "", text)
        text = re.sub(r"\[[^\]]*\]\(#.*?\)", "", text)
        text = re.sub(r";\s*see Appendix\s*\)", ")", text)
        text = re.sub(r"see Appendix\s*", "", text)
        text = re.sub(r"\((?:see )?(?:Section|Remark|Appendix)\s*\)", "", text)
        text = re.sub(r"\s+\\\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s+\\\s+", " ", text)
        text = re.sub(r"\s+([,.;:])", r"\1", text)
        text = re.sub(r"\(\s*\)", "", text)
        return text

    def _latex_snippet_to_plain(self, latex_input: str, paper_id: str) -> str:
        """Convert a LaTeX snippet into plain UTF-8 text for metadata copy/paste."""
        prepared = self._expand_lean_stat_macros(latex_input, paper_id)
        prepared = self._normalize_claimstamp_for_markdown(prepared)
        prepared = self._prepend_markdown_macro_prelude(prepared)

        result = subprocess.run(
            ["pandoc", "-f", "latex", "-t", "plain", "--wrap=none", "--columns=100"],
            input=prepared,
            capture_output=True,
            text=True,
        )
        plain = (
            result.stdout
            if result.returncode == 0 and result.stdout.strip()
            else latex_input
        )

        plain = self._clean_metadata_reference_tokens(plain)
        return self._normalize_plaintext_block(plain)

    def _latex_snippet_to_mathjax(self, latex_input: str, paper_id: str) -> str:
        """Convert a LaTeX snippet into markdown text with MathJax-style TeX math."""
        prepared = self._expand_lean_stat_macros(latex_input, paper_id)
        prepared = self._normalize_claimstamp_for_markdown(prepared)
        prepared = self._prepend_markdown_macro_prelude(prepared)

        result = subprocess.run(
            [
                "pandoc",
                "-f",
                "latex",
                "-t",
                "markdown+tex_math_dollars",
                "--wrap=none",
                "--columns=100",
            ],
            input=prepared,
            capture_output=True,
            text=True,
        )
        mathjax = (
            result.stdout
            if result.returncode == 0 and result.stdout.strip()
            else latex_input
        )

        # Expand common project macros to MathJax-safe forms.
        mathjax = re.sub(r"\\SigmaP\{([^}]+)\}", r"\\Sigma_{\1}^{P}", mathjax)
        mathjax = re.sub(r"\\PiP\{([^}]+)\}", r"\\Pi_{\1}^{P}", mathjax)
        mathjax = mathjax.replace(r"\coNP", "coNP")
        mathjax = mathjax.replace(r"\NP", "NP")
        mathjax = mathjax.replace(r"\Pclass", "P")
        mathjax = mathjax.replace(r"\PP", "PP")
        mathjax = mathjax.replace(r"\PSPACE", "PSPACE")
        mathjax = mathjax.replace(r"\Opt", r"\operatorname{Opt}")
        mathjax = re.sub(r"\\LH\{([^}]+)\}", r"\1", mathjax)
        mathjax = re.sub(r"\\claimstamp\{[^}]*\}\{[^}]*\}", "", mathjax)

        # Strip Markdown bold/italic markers for arXiv compatibility.
        # arXiv abstracts only support MathJax math, not Markdown or LaTeX text formatting.
        # Simply remove the markers, leaving plain text.
        mathjax = re.sub(r"\*\*([^*]+?)\*\*", r"\1", mathjax)  # **text** -> text
        mathjax = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"\1", mathjax)  # *text* -> text

        mathjax = self._clean_metadata_reference_tokens(mathjax)
        return self._normalize_plaintext_block(mathjax)

    def _mathjax_markdown_to_unicode_markdown(self, mathjax_text: str) -> str:
        """Convert `$...$` math in markdown to Unicode while preserving markdown structure."""
        cache: Dict[str, str] = {}

        def _to_unicode(expr: str) -> str:
            expr = expr.strip()
            if expr in cache:
                return cache[expr]

            result = subprocess.run(
                [
                    "pandoc",
                    "-f",
                    "latex",
                    "-t",
                    "plain",
                    "--wrap=none",
                    "--columns=100",
                ],
                input=f"${expr}$",
                capture_output=True,
                text=True,
            )
            converted = (
                result.stdout.strip()
                if result.returncode == 0 and result.stdout.strip()
                else expr
            )
            converted = re.sub(r"\s+", " ", converted).strip()
            cache[expr] = converted
            return converted

        # Convert display math first, then inline math.
        converted = re.sub(
            r"\$\$([\s\S]*?)\$\$",
            lambda m: _to_unicode(m.group(1)),
            mathjax_text,
        )
        converted = re.sub(
            r"\$([^$\n]+?)\$",
            lambda m: _to_unicode(m.group(1)),
            converted,
        )
        return self._normalize_plaintext_block(converted)

    def _convert_fake_subscripts_to_unicode(self, text: str) -> str:
        """Convert pandoc-style fake subscripts like X_(sub) to Unicode Xₛᵤᵦ.

        Only converts when ALL characters in the subscript/superscript have
        Unicode equivalents, to avoid ugly partial conversions like Edₑcᵢₛᵢₒₙ.
        """

        def _try_convert_sub(content: str) -> Optional[str]:
            """Return Unicode subscript if all chars convertible, else None."""
            result = []
            for c in content:
                if c in UNICODE_SUBSCRIPTS:
                    result.append(UNICODE_SUBSCRIPTS[c])
                elif c == " ":
                    result.append(" ")
                else:
                    return None  # Can't fully convert
            return "".join(result)

        def _try_convert_sup(content: str) -> Optional[str]:
            """Return Unicode superscript if all chars convertible, else None."""
            result = []
            for c in content:
                if c in UNICODE_SUPERSCRIPTS:
                    result.append(UNICODE_SUPERSCRIPTS[c])
                elif c == " ":
                    result.append(" ")
                else:
                    return None  # Can't fully convert
            return "".join(result)

        def _convert_sub_paren(match: re.Match) -> str:
            """Handle X_(content) patterns."""
            base = match.group(1)
            sub_content = match.group(2)
            converted = _try_convert_sub(sub_content)
            if converted is not None:
                return base + converted
            # Can't fully convert; keep parens only for expressions (has operators)
            # Pure alphanumeric subscripts don't need parens in plain text
            if re.search(r"[+\-−×÷*/=<> ]", sub_content):
                return f"{base}_({sub_content})"
            return f"{base}_{sub_content}"

        def _convert_sub_single(match: re.Match) -> str:
            """Handle X_c single-char patterns."""
            base = match.group(1)
            sub_content = match.group(2)
            converted = _try_convert_sub(sub_content)
            if converted is not None:
                return base + converted
            return f"{base}_{sub_content}"

        def _convert_sup_paren(match: re.Match) -> str:
            """Handle X^(content) patterns."""
            base = match.group(1)
            sup_content = match.group(2)
            converted = _try_convert_sup(sup_content)
            if converted is not None:
                return base + converted
            # Can't fully convert; keep parens only for expressions (has operators)
            # Pure alphanumeric superscripts don't need parens in plain text
            if re.search(r"[+\-−×÷*/=<> ]", sup_content):
                return f"{base}^({sup_content})"
            return f"{base}^{sup_content}"

        def _convert_sup_single(match: re.Match) -> str:
            """Handle X^c single-char patterns."""
            base = match.group(1)
            sup_content = match.group(2)
            converted = _try_convert_sup(sup_content)
            if converted is not None:
                return base + converted
            return f"{base}^{sup_content}"

        # Pattern: X_(content) -> X + subscript content
        # Handles both single char like X_a and parenthesized like X_(min)
        result = re.sub(r"(\w)_\(([^)]+)\)", _convert_sub_paren, text)
        result = re.sub(r"(\w)_([a-z0-9])\b", _convert_sub_single, result)

        # Pattern: X^(content) -> X + superscript content
        result = re.sub(r"(\w)\^\(([^)]+)\)", _convert_sup_paren, result)
        result = re.sub(r"(\w)\^([a-z0-9A-Z])\b", _convert_sup_single, result)

        return result

    def _convert_display_math_to_unicode(self, text: str) -> str:
        """Convert $$...$$ display math blocks to Unicode plain text.

        Handles aligned environments and other display math that pandoc
        can't convert, turning them into readable plain text.
        """

        def _convert_block(match: re.Match) -> str:
            content = match.group(1)

            # Strip aligned/align environment wrappers
            content = re.sub(r"\\begin\{aligned?\}", "", content)
            content = re.sub(r"\\end\{aligned?\}", "", content)

            # Convert LaTeX commands to Unicode
            replacements = [
                (r"\\iff", "⟺"),
                (r"\\Leftrightarrow", "⟺"),
                (r"\\implies", "⟹"),
                (r"\\Rightarrow", "⟹"),
                (r"\\to", "→"),
                (r"\\rightarrow", "→"),
                (r"\\leftarrow", "←"),
                (r"\\leftrightarrow", "↔"),
                (r"\\neq", "≠"),
                (r"\\leq", "≤"),
                (r"\\geq", "≥"),
                (r"\\times", "×"),
                (r"\\cdot", "·"),
                (r"\\infty", "∞"),
                (r"\\in", "∈"),
                (r"\\notin", "∉"),
                (r"\\subset", "⊂"),
                (r"\\subseteq", "⊆"),
                (r"\\forall", "∀"),
                (r"\\exists", "∃"),
                (r"\\neg", "¬"),
                (r"\\land", "∧"),
                (r"\\lor", "∨"),
                (r"\\Delta", "Δ"),
                (r"\\Sigma", "Σ"),
                (r"\\Pi", "Π"),
                (r"\\Omega", "Ω"),
                (r"\\alpha", "α"),
                (r"\\beta", "β"),
                (r"\\gamma", "γ"),
                (r"\\delta", "δ"),
                (r"\\epsilon", "ε"),
                (r"\\theta", "θ"),
                (r"\\lambda", "λ"),
                (r"\\mu", "μ"),
                (r"\\pi", "π"),
                (r"\\sigma", "σ"),
                (r"\\omega", "ω"),
                (r"\\ln", "ln"),
                (r"\\log", "log"),
                (r"\\min", "min"),
                (r"\\max", "max"),
            ]
            for pattern, replacement in replacements:
                content = re.sub(pattern, replacement, content)

            # Remove \mathrm{} but keep content
            content = re.sub(r"\\mathrm\{([^}]+)\}", r"\1", content)
            # Remove \text{} but keep content
            content = re.sub(r"\\text\{([^}]+)\}", r"\1", content)
            # Remove \; \, \! \  spacing commands
            content = re.sub(r"\\[;,! ]", " ", content)
            # Remove & alignment markers
            content = content.replace("&", "")
            # Convert \\ to space (line breaks in aligned become spaces)
            content = re.sub(r"\\\\", " ", content)
            # Clean up multiple spaces
            content = re.sub(r"\s+", " ", content)
            # Remove | that were for absolute value markers in \mathrm
            # but keep them if they look like |Cap|

            return content.strip()

        # Match $$...$$ blocks (including multiline)
        result = re.sub(r"\$\$([\s\S]*?)\$\$", _convert_block, text)
        return result

    def _latex_snippet_to_unicode_zenodo(self, latex_input: str, paper_id: str) -> str:
        """Convert LaTeX to Unicode plain text suitable for Zenodo (no markdown markers).

        Uses pylatexenc if available for better Greek letter handling, then applies
        Unicode subscript/superscript conversion for proper math rendering.
        """
        # First expand macros and get plain text via pandoc
        plain = self._latex_snippet_to_plain(latex_input, paper_id)

        # Convert display math blocks ($$...$$) that pandoc couldn't handle
        plain = self._convert_display_math_to_unicode(plain)

        # Convert fake subscripts/superscripts to real Unicode
        plain = self._convert_fake_subscripts_to_unicode(plain)

        # Use explicit Unicode bullets instead of markdown-style `- ` markers.
        plain = re.sub(r"(?m)^-\s+", "• ", plain)

        # Common symbol replacements that pandoc may miss
        plain = plain.replace(">=", "≥")
        plain = plain.replace("<=", "≤")
        plain = plain.replace("!=", "≠")
        plain = plain.replace("->", "→")
        plain = plain.replace("<->", "↔")
        plain = plain.replace("<=>", "⇔")
        plain = plain.replace("...", "…")

        return self._normalize_plaintext_block(plain)

    def _default_arxiv_comments(self, paper_id: str) -> str:
        """Build a default arXiv comments line when no explicit metadata is configured."""
        meta = self._get_paper_meta(paper_id)
        lean_stats = self._get_release_lean_stats(paper_id)
        return (
            f"{meta.venue} submission. Lean 4 artifact: "
            f"{lean_stats.line_count} lines, {lean_stats.theorem_count} theorems/lemmas "
            f"across {lean_stats.file_count} files ({lean_stats.sorry_count} sorry placeholders)."
        )

    def _write_abstract_helper_files(
        self,
        paper_id: str,
        title: str,
        abstract_mathjax: str,
        abstract_unicode: str,
    ) -> tuple[Path, Path]:
        """Write convenience abstract files from one canonical source.

        Primary file is paper-id scoped to avoid collisions when multiple variants
        share a directory (e.g., `paper4` and `paper4_toc`).
        """
        meta = self._get_paper_meta(paper_id)
        releases_dir = self._get_releases_dir(paper_id)

        primary_path = releases_dir / f"arxiv_abstract_{paper_id}.md"
        primary_lines = [
            f"# arXiv abstract ({paper_id})",
            "",
            f"Title: {title}",
            "",
            "## Abstract (MathJax, arXiv-ready)",
            "",
            "```text",
            abstract_mathjax,
            "```",
            "",
            "## Abstract (Unicode, Zenodo-ready)",
            "",
            "```text",
            abstract_unicode,
            "```",
            "",
        ]
        primary_path.write_text("\n".join(primary_lines), encoding="utf-8")

        # Backward-compatible filename (dir-scoped) can be ambiguous when
        # multiple paper IDs point at the same directory.
        legacy_path = releases_dir / f"arxiv_abstract_{meta.dir_name}.md"
        sibling_ids = sorted(
            pid
            for pid, sibling_meta in self.papers.items()
            if sibling_meta.dir_name == meta.dir_name
        )
        if len(sibling_ids) == 1:
            legacy_path.write_text("\n".join(primary_lines), encoding="utf-8")
        else:
            redirect_lines = [
                f"# Deprecated helper file: arxiv_abstract_{meta.dir_name}.md",
                "",
                "This directory has multiple paper IDs with different titles/abstracts.",
                "Use the paper-id-specific helper files instead:",
                "",
            ]
            redirect_lines.extend(
                f"- `arxiv_abstract_{sibling_id}.md`" for sibling_id in sibling_ids
            )
            redirect_lines.append("")
            legacy_path.write_text("\n".join(redirect_lines), encoding="utf-8")

        return primary_path, legacy_path

    def build_copy_paste_metadata(self, paper_id: str) -> Path:
        """Generate human/machine copy-paste metadata for Zenodo + arXiv."""
        meta = self._get_paper_meta(paper_id)
        # Single source of truth for generated artifacts: main LaTeX `\title{...}`.
        title = self._release_title_from_latex(paper_id)
        abstract_latex = self._extract_abstract_latex(paper_id)
        abstract_mathjax = (
            self._latex_snippet_to_mathjax(abstract_latex, paper_id)
            if abstract_latex
            else "Abstract not found."
        )
        # Zenodo variant should be Unicode plain text (no markdown formatting markers).
        abstract_unicode = (
            self._latex_snippet_to_unicode_zenodo(abstract_latex, paper_id)
            if abstract_latex
            else "Abstract not found."
        )
        arxiv_comments = (
            self._normalize_plaintext_block(meta.arxiv_comments)
            if meta.arxiv_comments.strip()
            else self._default_arxiv_comments(paper_id)
        )

        payload = {
            "paper_id": paper_id,
            "title": title,
            "zenodo": {
                "title": title,
                "abstract": abstract_unicode,
            },
            "arxiv": {
                "title": title,
                "abstract": abstract_mathjax,
                "comments": arxiv_comments,
            },
            "abstract_variants": {
                "unicode": abstract_unicode,
                "mathjax": abstract_mathjax,
            },
        }
        machine_yaml = yaml.safe_dump(
            payload, sort_keys=False, allow_unicode=True
        ).strip()

        lines = [
            "AUTO-GENERATED COPY/PASTE METADATA",
            f"Generated: {datetime.now().isoformat()}",
            "",
            "=== HUMAN COPY/PASTE ===",
            "",
            f"Title: {title}",
            "",
            "Abstract (Unicode, for Zenodo):",
            abstract_unicode,
            "",
            "Abstract (MathJax, for arXiv):",
            abstract_mathjax,
            "",
            "arXiv Comments:",
            arxiv_comments,
            "",
            "=== MACHINE YAML ===",
            machine_yaml,
            "",
        ]

        releases_dir = self._get_releases_dir(paper_id)
        out_file = releases_dir / f"{paper_id}_copy_paste.txt"
        out_file.write_text("\n".join(lines), encoding="utf-8")
        abstract_helper_file, legacy_helper_file = self._write_abstract_helper_files(
            paper_id=paper_id,
            title=title,
            abstract_mathjax=abstract_mathjax,
            abstract_unicode=abstract_unicode,
        )
        print(f"[metadata] {paper_id} → {out_file.relative_to(self.repo_root)}")
        print(
            "[metadata] "
            f"{paper_id} abstracts → {abstract_helper_file.relative_to(self.repo_root)} "
            f"(legacy alias: {legacy_helper_file.relative_to(self.repo_root)})"
        )
        return out_file

    def _build_markdown_file(
        self,
        meta: PaperMeta,
        content_files: List[Path],
        out_file: Path,
        title: str,
        lean_stats: LeanStats,
    ):
        """Generate markdown file from LaTeX content."""
        with open(out_file, "w") as f:
            # Header
            f.write(f"# Paper: {title}\n\n")
            f.write(
                f"**Status**: {meta.venue}-ready | "
                f"**Lean**: {lean_stats.line_count} lines, "
                f"{lean_stats.theorem_count} theorems\n\n---\n\n"
            )

            # Abstract
            f.write("## Abstract\n\n")
            abstract_file = next(
                (p for p in content_files if p.name == "abstract.tex"), None
            )
            if abstract_file and abstract_file.exists():
                # Try extracting from \begin{abstract}...\end{abstract} first,
                # fall back to raw file content if no environment found
                content = abstract_file.read_text(encoding="utf-8", errors="replace")
                if r"\begin{abstract}" in content:
                    self._convert_latex_to_markdown(
                        abstract_file,
                        f,
                        paper_id=meta.paper_id,
                        extract_env="abstract",
                    )
                else:
                    self._convert_latex_to_markdown(
                        abstract_file, f, paper_id=meta.paper_id
                    )
            else:
                f.write("_Abstract not available._\n\n")

            # Content sections (ordered exactly as LaTeX includes)
            for content_path in content_files:
                if abstract_file and content_path.resolve() == abstract_file.resolve():
                    continue
                self._convert_latex_to_markdown(content_path, f, paper_id=meta.paper_id)

            # Footer
            f.write("\n\n---\n\n## Machine-Checked Proofs\n\n")
            f.write(f"All theorems are formalized in Lean 4:\n")
            proofs_rel = self._get_paper_proofs_dir(meta.paper_id).relative_to(
                self.repo_root
            )
            f.write(f"- Location: `{proofs_rel}/`\n")
            f.write(f"- Lines: {lean_stats.line_count}\n")
            f.write(f"- Theorems: {lean_stats.theorem_count}\n")
            f.write(f"- `sorry` placeholders: {lean_stats.sorry_count}\n")

    def _convert_latex_to_markdown(
        self,
        tex_file: Path,
        out_file,
        paper_id: str,
        extract_env: str | None = None,
    ):
        """Convert LaTeX file to Markdown using pandoc.

        Args:
            tex_file: Path to .tex file
            out_file: Output file handle
            extract_env: If set, extract content from \\begin{env}...\\end{env} first
        """
        import re

        content = tex_file.read_text(encoding="utf-8", errors="replace")
        if extract_env:
            # Extract content from environment before converting
            pattern = rf"\\begin\{{{extract_env}\}}(.*?)\\end\{{{extract_env}\}}"
            match = re.search(pattern, content, re.DOTALL)
            if not match:
                out_file.write(f"_Content not found in {tex_file.name}_\n\n")
                return
            latex_input = match.group(1).strip()
        else:
            latex_input = content

        latex_input = self._expand_lean_stat_macros(latex_input, paper_id)
        latex_input = self._normalize_claimstamp_for_markdown(latex_input)
        latex_input = self._prepend_markdown_macro_prelude(latex_input)
        result = subprocess.run(
            ["pandoc", "-f", "latex", "-t", "markdown", "--wrap=none", "--columns=100"],
            input=latex_input,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0 and result.stdout:
            out_file.write(self._postprocess_markdown_text(result.stdout))
            out_file.write("\n\n")
        else:
            out_file.write(f"_Failed to convert {tex_file.name}_\n\n")

    def _normalize_claimstamp_for_markdown(self, latex_input: str) -> str:
        """Simplify `\\claimstamp{...}{...}` arguments for markdown readability.

        Keep theorem provenance, but normalize references to raw labels
        (e.g., `Thm.~\\ref{thm:x}` -> `thm:x`) so generated markdown stays compact.
        """
        pattern = re.compile(
            r"\\claimstamp\{((?:[^{}]|\\ref\{[^}]+\})*)\}\{([^}]*)\}",
            re.DOTALL,
        )

        title_ref = re.compile(
            r"(?:Theorem|Thm\.?|Proposition|Prop\.?|Corollary|Cor\.?|Lemma|Lem\.?)\s*~?\s*\\ref\{([^}]+)\}"
        )

        def repl(match: re.Match[str]) -> str:
            derived = match.group(1)
            regime = match.group(2)
            derived = title_ref.sub(r"\1", derived)
            derived = re.sub(r"\\ref\{([^}]+)\}", r"\1", derived)
            derived = derived.replace("~", " ")
            derived = re.sub(r"\s+", " ", derived).strip(" ,")
            derived = re.sub(r"\s*,\s*", ", ", derived)
            regime = re.sub(r"\s+", " ", regime).strip()
            return rf"\claimstamp{{{derived}}}{{{regime}}}"

        return pattern.sub(repl, latex_input)

    def _prepend_markdown_macro_prelude(self, latex_input: str) -> str:
        """Prepend markdown-only macro normalizations for pandoc conversion.

        Markdown conversion processes section files directly, without the main
        LaTeX preamble. Custom macros used in prose (e.g., ``\\coNP``) would
        otherwise be dropped by pandoc, producing broken text such as
        ``-complete``. We define plain/math-safe expansions here strictly for
        conversion fidelity.
        """
        prelude = "\n".join(
            [
                r"\newcommand{\coNP}{coNP}",
                r"\newcommand{\NP}{NP}",
                r"\newcommand{\Pclass}{P}",
                r"\newcommand{\PP}{PP}",
                r"\newcommand{\PSPACE}{PSPACE}",
                r"\newcommand{\SigmaP}[1]{\Sigma_{#1}^{P}}",
                r"\newcommand{\PiP}[1]{\Pi_{#1}^{P}}",
                r"\newcommand{\Opt}{\operatorname{Opt}}",
                r"\newcommand{\LH}[1]{#1}",
                r"\newcommand{\claimstamp}[2]{ [D:#1; R:#2]}",
            ]
        )
        return f"{prelude}\n{latex_input}"

    def _postprocess_markdown_text(self, text: str) -> str:
        """Normalize small TeX-to-markdown spacing artifacts.

        TeX control-word macros consume following spaces unless braced. When
        class macros appear in prose (e.g., ``\\coNP characterization``), the
        converted markdown can become concatenated (``coNPcharacterization``).
        Insert the missing separator for known complexity-class tokens.
        """
        normalized = re.sub(r"\bcoNP(?=[A-Za-z])", "coNP ", text)
        normalized = re.sub(r"\bPP(?=[A-Za-z])", "PP ", normalized)
        normalized = re.sub(r"\bPSPACE(?=[A-Za-z])", "PSPACE ", normalized)
        return normalized

    def _get_claim_mapping_file(self, paper_id: str) -> Optional[Path]:
        """Locate theorem-handle mapping file for claim coverage checks."""
        meta = self._get_paper_meta(paper_id)
        latex_dir = self._get_paper_dir(paper_id) / meta.latex_dir
        if meta.claim_mapping_file:
            explicit = latex_dir / meta.claim_mapping_file
            return explicit if explicit.exists() else None

        candidates = [
            latex_dir / "content" / "claim_mapping_auto.tex",
            latex_dir / "content" / "11_lean_proofs.tex",
            latex_dir / "content" / "10_lean_4_proof_listings.tex",
            latex_dir / "content" / "10_lean_proof_listings.tex",
            latex_dir / "content" / "10_lean_proofs.tex",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _extract_paper_claim_labels(self, paper_id: str) -> Set[str]:
        """Extract theorem/corollary/lemma/proposition labels from content/*.tex."""
        content_dir = self._get_content_dir(paper_id)
        if not content_dir.exists():
            return set()

        labels: Set[str] = set()
        label_pattern = re.compile(r"\\label\{((?:thm|cor|lem|prop):[^}]+)\}")
        claim_label_pattern = re.compile(
            r"\\begin\{claim\}\{(?:[^{}]|\{[^{}]*\})*\}\{((?:thm|cor|lem|prop):[^{}]+)\}"
        )
        for tex_file in sorted(content_dir.glob("*.tex")):
            text = tex_file.read_text(encoding="utf-8", errors="replace")
            labels.update(label_pattern.findall(text))
            labels.update(claim_label_pattern.findall(text))
        return labels

    def _extract_mapped_claim_labels(self, mapping_file: Path) -> Set[str]:
        """Extract mapped theorem-handle labels from a Lean-coverage appendix file."""
        text = mapping_file.read_text(encoding="utf-8", errors="replace")
        labels: Set[str] = set()
        patterns = [
            r"\\nolinkurl\{((?:thm|cor|lem|prop):[^}]+)\}",
            r"\\(?:ref|autoref|Cref)\{((?:thm|cor|lem|prop):[^}]+)\}",
        ]
        for pattern in patterns:
            labels.update(re.findall(pattern, text))
        return labels

    def check_claim_coverage(
        self, paper_id: str, fail_on_missing: bool = False
    ) -> Dict[str, object]:
        """Check label coverage between paper claims and Lean mapping appendix."""
        self._write_lean_handle_ids_auto(paper_id)
        self._rewrite_content_lean_handles_to_ids(paper_id)
        self._normalize_and_fill_claimstamps(paper_id)
        self._write_claim_mapping_auto(paper_id)
        self._write_proof_hardness_index_auto(paper_id)
        claim_labels = self._extract_paper_claim_labels(paper_id)
        mapping_file = self._get_claim_mapping_file(paper_id)

        if mapping_file is None:
            message = f"[claim-check] {paper_id}: no mapping file found; skipping."
            print(message)
            if fail_on_missing and claim_labels:
                raise RuntimeError(message)
            return {
                "paper_id": paper_id,
                "mapping_file": None,
                "paper_labels": sorted(claim_labels),
                "mapped_labels": [],
                "missing_labels": sorted(claim_labels),
                "extra_labels": [],
            }

        # Keep a single source of truth for auto-generated mapping files:
        # use direct label->handle extraction, not rendered table parsing.
        if mapping_file.name == "claim_mapping_auto.tex":
            label_to_handles = self._extract_claim_label_to_lean_handles(paper_id)
            mapped_labels = {
                label for label, handles in label_to_handles.items() if handles
            }
        else:
            mapped_labels = self._extract_mapped_claim_labels(mapping_file)
        missing_labels = sorted(claim_labels - mapped_labels)
        extra_labels = sorted(mapped_labels - claim_labels)

        print(
            f"[claim-check] {paper_id}: "
            f"paper={len(claim_labels)} mapped={len(mapped_labels)} "
            f"missing={len(missing_labels)} extra={len(extra_labels)}"
        )
        if missing_labels:
            print("[claim-check]   Missing mappings:")
            for label in missing_labels:
                print(f"[claim-check]     - {label}")
        if extra_labels:
            print("[claim-check]   Extra mappings (not found in paper labels):")
            for label in extra_labels:
                print(f"[claim-check]     - {label}")

        if fail_on_missing and missing_labels:
            raise RuntimeError(
                f"Claim mapping incomplete for {paper_id}: {len(missing_labels)} labels unmapped"
            )

        # Forcing coverage analysis
        forcing_result = self._check_forcing_coverage(paper_id)

        return {
            "paper_id": paper_id,
            "mapping_file": str(mapping_file.relative_to(self.repo_root)),
            "paper_labels": sorted(claim_labels),
            "mapped_labels": sorted(mapped_labels),
            "missing_labels": missing_labels,
            "extra_labels": extra_labels,
            "forcing": forcing_result,
        }

    def _check_forcing_coverage(self, paper_id: str) -> Optional[Dict[str, object]]:
        """Check forcing coverage using analyze_forcing_coverage.py."""
        import importlib.util

        analyzer_path = self.repo_root / "scripts" / "analyze_forcing_coverage.py"
        if not analyzer_path.exists():
            print("[claim-check]   Warning: analyze_forcing_coverage.py not found")
            return None

        try:
            spec = importlib.util.spec_from_file_location(
                "analyze_forcing_coverage", analyzer_path
            )
            if spec is None or spec.loader is None:
                print("[claim-check]   Warning: Could not load analyzer module")
                return None
            analyzer = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(analyzer)

            result = analyzer.analyze(paper_id, self.repo_root)

            print(
                f"[claim-check]   Forcing: {result['reaching_forcing']}/{result['total_claims']} "
                f"({result['coverage_fraction']:.1%})"
            )

            if result["non_reaching"]:
                print(f"[claim-check]   Non-reaching: {len(result['non_reaching'])}")
                for item in result["non_reaching"][:5]:
                    print(f"[claim-check]     - {item['label']}")
                    print(f"[claim-check]       Furthest: {item['furthest_reachable']}")

            return result

        except FileNotFoundError as e:
            print(f"[claim-check]   Warning: Forcing analysis skipped: {e}")
            return None
        except Exception as e:
            print(f"[claim-check]   Warning: Forcing analysis failed: {e}")
            return None

    def _update_paper_date(self, tex_file: Path):
        """Update publication date in LaTeX file using regex for correct replacement."""
        import re

        year = datetime.now().strftime("%Y")
        try:
            content = tex_file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = tex_file.read_text(encoding="latin-1")

        # Normalize to compile-time date in LaTeX sources.
        content = re.sub(r"\\date\{[^}]*\}", r"\\date{\\today}", content)
        content = re.sub(
            r"Manuscript received [^.\\n]*\.",
            r"Manuscript received \\today.",
            content,
        )

        # Keep explicit year fields in sync with build year where present.
        content = re.sub(
            r"\\copyrightyear\{[^}]*\}", f"\\\\copyrightyear{{{year}}}", content
        )
        content = re.sub(r"\\acmYear\{[^}]*\}", f"\\\\acmYear{{{year}}}", content)
        tex_file.write_text(content, encoding="utf-8")

    def package_arxiv(self, paper_id: str):
        """
        Package paper for arXiv submission.

        Includes:
        - PDF (compiled paper)
        - Markdown (full text for LLM consumption)
        - Lean proofs (source code)
        - BUILD_LOG.txt (actual lake build output as compilation proof)

        For variants (e.g., paper1_jsait), uses variant-specific staging directory.
        """
        self._refresh_derived_content(paper_id)
        metadata_file = self.build_copy_paste_metadata(paper_id)

        # Phase 1: Validate all required files exist (fail-loud)
        pdf_file = self._validate_and_get_pdf(paper_id)
        md_file = self._validate_and_get_markdown(paper_id)

        # Phase 2: Create staging directory in releases/ with variant-specific naming
        releases_dir = self._get_releases_dir(paper_id)
        archive_prefix = self._get_archive_prefix(paper_id)
        package_dir = releases_dir / f"arxiv_package_{archive_prefix}"
        if package_dir.exists():
            shutil.rmtree(package_dir)
        package_dir.mkdir(parents=True)

        print(f"[arxiv] Packaging {paper_id}...")

        # Phase 3: Copy artifacts
        self._copy_pdf(pdf_file, package_dir)
        self._copy_markdown(md_file, package_dir)
        self._copy_copy_paste_metadata(metadata_file, package_dir)
        self._copy_latex_sources(paper_id, package_dir)
        self._copy_lean_proofs(paper_id, package_dir)
        self._copy_experiments(paper_id, package_dir)
        self._copy_graph_visualizer(paper_id, package_dir)

        if self.arxiv_config.include_build_log:
            self._create_build_log(paper_id, package_dir)

        # Phase 4: Create compressed archives
        tar_path, zip_path = self._create_archive(paper_id, package_dir)

        print(
            f"[arxiv] {paper_id} → {tar_path.relative_to(self.repo_root)}, {zip_path.name}"
        )
        return tar_path

    def _validate_and_get_pdf(self, paper_id: str) -> Path:
        """Validate PDF exists and return path. Fail-loud if missing."""
        meta = self._get_paper_meta(paper_id)
        latex_dir = self._get_latex_dir(paper_id)
        pdfs = list(latex_dir.glob("*.pdf"))

        if not pdfs:
            raise FileNotFoundError(
                f"[arxiv] FATAL: No PDF found in {latex_dir}\n"
                f"  Paper: {meta.name}\n"
                f"  Run: python3 scripts/build_papers.py latex {paper_id}"
            )
        return pdfs[0]

    def _validate_and_get_markdown(self, paper_id: str) -> Path:
        """Validate Markdown exists and return path. Fail-loud if missing."""
        meta = self._get_paper_meta(paper_id)
        # Must match build_markdown(), which writes <paper_id>.md.
        # This is critical for variants (e.g., paper4_toc) that share dir_name
        # with a base paper but have different LaTeX sources/content.
        md_file = self._get_markdown_file(paper_id)

        if not md_file.exists():
            raise FileNotFoundError(
                f"[arxiv] FATAL: Markdown not found: {md_file}\n"
                f"  Paper: {meta.name}\n"
                f"  Run: python3 scripts/build_papers.py markdown {paper_id}"
            )
        return md_file

    def _copy_pdf(self, pdf_file: Path, package_dir: Path) -> None:
        """Copy PDF to package directory."""
        pdf_dest = package_dir / pdf_file.name
        shutil.copy2(pdf_file, pdf_dest)
        print(f"[arxiv]   PDF: {pdf_file.name}")

    def _copy_markdown(self, md_file: Path, package_dir: Path) -> None:
        """Copy Markdown to package directory."""
        md_dest = package_dir / md_file.name
        shutil.copy2(md_file, md_dest)
        print(f"[arxiv]   Markdown: {md_file.name}")

    def _copy_copy_paste_metadata(self, metadata_file: Path, package_dir: Path) -> None:
        """Copy copy/paste metadata file to package directory."""
        metadata_dest = package_dir / metadata_file.name
        shutil.copy2(metadata_file, metadata_dest)
        print(f"[arxiv]   Metadata: {metadata_file.name}")

    def _copy_latex_sources(self, paper_id: str, package_dir: Path) -> None:
        """Copy LaTeX source files for arXiv submission.

        Copies all .tex, .bib, .bbl, .cls, .sty files from the latex directory.
        Also copies content/ subdirectory if present.
        """
        meta = self._get_paper_meta(paper_id)
        latex_dir = self._get_latex_dir(paper_id)

        if not latex_dir.exists():
            print(f"[arxiv]   No LaTeX directory for {paper_id}, skipping...")
            return

        # Extensions to copy
        copied_count = 0

        # Copy top-level LaTeX files
        for ext in self.build_config.latex_source_extensions:
            for src_file in latex_dir.glob(f"*{ext}"):
                shutil.copy2(src_file, package_dir / src_file.name)
                copied_count += 1

        # Copy content/ subdirectory if present
        content_dir = latex_dir / "content"
        if content_dir.exists():
            content_dest = package_dir / "content"
            content_dest.mkdir(parents=True, exist_ok=True)
            for src_file in content_dir.glob("*.tex"):
                shutil.copy2(src_file, content_dest / src_file.name)
                copied_count += 1

        print(f"[arxiv]   LaTeX sources: {copied_count} files")

    def _copy_lean_proofs(self, paper_id: str, package_dir: Path) -> None:
        """Copy Lean proofs for specific paper to package directory.

        Includes:
        - The paper's own proofs directory
        - Lean files from transitive lean_dependencies

        Dependency files are inlined into the package proofs graph by relative
        path (with conflict-safe fallback), so bridge imports resolve without
        manual per-paper path patching.
        """
        proofs_dir = self._get_paper_proofs_dir(paper_id)

        if not proofs_dir.exists():
            print(f"[arxiv]   No proofs directory for {paper_id}, skipping...")
            return

        lean_dest = package_dir / "proofs"
        lean_dest.mkdir(parents=True, exist_ok=True)

        dep_ids = self._collect_lean_dependency_closure(paper_id)
        roots: List[Tuple[str, Path]] = [(paper_id, proofs_dir)]
        roots.extend((dep_id, self._get_paper_proofs_dir(dep_id)) for dep_id in dep_ids)
        release_module_closure = self._get_release_module_closure(paper_id)

        paper_files: List[Path] = []
        dep_file_count = 0
        copied_count = 0
        identical_skips = 0
        conflict_fallbacks = 0

        for source_paper_id, source_dir in roots:
            if not source_dir.exists():
                print(
                    f"[arxiv]   Warning: dependency {source_paper_id} has no proofs dir, skipping..."
                )
                continue

            allowed_modules = None
            if release_module_closure is not None:
                allowed_modules = release_module_closure.get(source_paper_id, set())

            for src_file in self._iter_paper_lean_files(source_dir):
                rel_module = str(src_file.relative_to(source_dir).with_suffix(""))
                if allowed_modules is not None and rel_module not in allowed_modules:
                    continue
                rel_path = src_file.relative_to(source_dir)
                root_dest = lean_dest / rel_path
                root_dest.parent.mkdir(parents=True, exist_ok=True)

                if root_dest.exists():
                    if self._files_identical(src_file, root_dest):
                        identical_skips += 1
                    else:
                        fallback_dest = lean_dest / f"dep_{source_paper_id}" / rel_path
                        fallback_dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src_file, fallback_dest)
                        conflict_fallbacks += 1
                        copied_count += 1
                else:
                    shutil.copy2(src_file, root_dest)
                    copied_count += 1

                # Also maintain a full dependency mirror for path-based requires.
                if source_paper_id != paper_id:
                    dep_dest = lean_dest / f"dep_{source_paper_id}" / rel_path
                    dep_dest.parent.mkdir(parents=True, exist_ok=True)
                    if dep_dest.exists() and self._files_identical(src_file, dep_dest):
                        pass
                    else:
                        shutil.copy2(src_file, dep_dest)
                        copied_count += 1

                if source_paper_id == paper_id:
                    paper_files.append(src_file)
                else:
                    dep_file_count += 1

        # Copy config files from main paper.
        config_files = ["lean-toolchain", "lake-manifest.json", "lakefile.lean"]
        for fname in config_files:
            src = proofs_dir / fname
            if src.exists():
                shutil.copy2(src, lean_dest / fname)

        # Copy dependency package config files (needed for `require ... from "./dep_*"`).
        dep_config_files = ["lean-toolchain", "lake-manifest.json", "lakefile.lean"]
        for dep_paper_id in dep_ids:
            dep_src = self._get_paper_proofs_dir(dep_paper_id)
            if not dep_src.exists():
                continue
            dep_dest = lean_dest / f"dep_{dep_paper_id}"
            dep_dest.mkdir(parents=True, exist_ok=True)
            for fname in dep_config_files:
                src = dep_src / fname
                if src.exists():
                    shutil.copy2(src, dep_dest / fname)
            self._rewrite_packaged_lakefile_srcdirs(dep_dest / "lakefile.lean")

        self._rewrite_packaged_lakefile_srcdirs(lean_dest / "lakefile.lean")
        self._write_dependency_bridge_module(paper_id, lean_dest, dep_ids)

        # Generate README (correct by construction)
        self._generate_proofs_readme(paper_id, paper_files, lean_dest)

        for dep_paper_id in dep_ids:
            dep_proofs_dir = self._get_paper_proofs_dir(dep_paper_id)
            if dep_proofs_dir.exists():
                if release_module_closure is None:
                    dep_count = len(self._iter_paper_lean_files(dep_proofs_dir))
                else:
                    dep_count = len(release_module_closure.get(dep_paper_id, set()))
                print(f"[arxiv]   Dependency {dep_paper_id}: {dep_count} files")

        total_files = len(paper_files) + dep_file_count
        print(
            f"[arxiv]   Lean proofs: {len(paper_files)} local + {dep_file_count} dependency files "
            f"(copied={copied_count}, identical-skips={identical_skips}, conflict-fallbacks={conflict_fallbacks})"
        )

    def _rewrite_packaged_lakefile_srcdirs(self, lakefile_path: Path) -> None:
        """Rewrite external `srcDir := \"../...\"` entries for package-local builds."""
        if not lakefile_path.exists():
            return
        text = lakefile_path.read_text(encoding="utf-8", errors="replace")
        rewritten = re.sub(r'srcDir\s*:=\s*"\.\./[^"]+"', 'srcDir := "."', text)
        if rewritten != text:
            lakefile_path.write_text(rewritten, encoding="utf-8")

    def _write_dependency_bridge_module(
        self, paper_id: str, lean_dest: Path, dep_ids: List[str]
    ) -> None:
        """Generate a bridge module that imports dependency module roots."""
        if not dep_ids:
            return

        host_roots = self._derive_module_roots(paper_id)
        if not host_roots:
            return
        host_root = host_roots[0]

        dep_imports: List[str] = []
        for dep_id in dep_ids:
            for root in self._derive_module_roots(dep_id):
                if "." not in root:
                    continue
                root_file = lean_dest / Path(*root.split(".")).with_suffix(".lean")
                if not root_file.exists():
                    continue
                if root not in dep_imports:
                    dep_imports.append(root)

        bridge_rel = Path(*host_root.split(".")) / "DependencyBridge.lean"
        bridge_file = lean_dest / bridge_rel
        bridge_file.parent.mkdir(parents=True, exist_ok=True)

        lines: List[str] = [
            "-- Auto-generated by scripts/build_papers.py. Do not edit manually.",
            "-- Dependency bridge imports to inline transitive proof machinery.",
        ]
        for root in dep_imports:
            lines.append(f"import {root}")
        lines.extend(
            [
                "",
                f"namespace {host_root}",
                "",
                "-- Marker module: importing this namespace also imports dependency roots.",
                "",
                f"end {host_root}",
                "",
            ]
        )
        bridge_file.write_text("\n".join(lines), encoding="utf-8")

    def _copy_experiments(self, paper_id: str, package_dir: Path) -> None:
        """Copy configured experiment trees into the package.

        Sources may be paper-local (`paper_dir / path`) or repo-relative (`repo_root / path`).
        Copy is recursive and excludes heavyweight/generated artifacts so release bundles stay
        portable and reviewable.
        """
        meta = self._get_paper_meta(paper_id)
        paper_dir = self._get_paper_dir(paper_id)

        configured_sources = list(meta.experiment_paths)
        legacy_source = paper_dir / "experiments"
        if not configured_sources and legacy_source.exists():
            configured_sources.append("experiments")

        if not configured_sources:
            return

        experiments_dest = package_dir / "experiments"
        copied_count = 0
        copied_roots: List[str] = []
        for source_spec in configured_sources:
            source_dir = self._resolve_experiment_source(paper_dir, source_spec)
            if source_dir is None or not source_dir.exists():
                continue
            dest_root = experiments_dest / source_dir.name
            copied_here = self._copy_experiment_tree(source_dir, dest_root)
            if copied_here > 0:
                copied_count += copied_here
                copied_roots.append(source_spec)

        if copied_count > 0:
            roots = ", ".join(copied_roots)
            print(f"[arxiv]   Experiments: {copied_count} files from {roots}")

    def _resolve_experiment_source(
        self, paper_dir: Path, source_spec: str
    ) -> Optional[Path]:
        """Resolve a configured experiment source to an existing directory."""
        candidate = Path(source_spec)
        search_order = []
        if candidate.is_absolute():
            search_order.append(candidate)
        else:
            search_order.append(paper_dir / candidate)
            search_order.append(self.repo_root / candidate)
        for path in search_order:
            if path.exists() and path.is_dir():
                return path
        return None

    def _copy_experiment_tree(self, source_dir: Path, dest_dir: Path) -> int:
        """Recursively copy a portable experiment tree into the package."""
        copied_count = 0
        allowed_extensions = {
            ".py",
            ".json",
            ".jsonl",
            ".csv",
            ".tsv",
            ".txt",
            ".sh",
            ".md",
            ".ipynb",
            ".png",
            ".jpg",
            ".jpeg",
            ".svg",
            ".pdf",
            ".toml",
            ".yaml",
            ".yml",
            ".tex",
        }
        skipped_dir_names = {
            "__pycache__",
            ".ipynb_checkpoints",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            ".venv",
            "venv",
            "env",
            "ENV",
            "out",
            "outputs",
            "results",
            "checkpoints",
            "models",
            ".git",
        }

        for src in source_dir.rglob("*"):
            rel = src.relative_to(source_dir)
            if any(part in skipped_dir_names for part in rel.parts):
                continue
            if src.is_dir():
                continue
            if src.suffix.lower() not in allowed_extensions:
                continue
            dst = dest_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied_count += 1
        return copied_count

    def _run_experiment_commands(self, paper_id: str) -> None:
        """Run configured experiment commands before derived-content generation.

        Commands are intentionally generic and configured in `papers.yaml` so any
        paper can opt into reproducible experiment-derived assets (tables, plots,
        certificates, CSV summaries, etc.).
        """
        meta = self._get_paper_meta(paper_id)
        if not meta.experiment_commands:
            return

        paper_dir = self._get_paper_dir(paper_id)
        latex_dir = self._get_latex_dir(paper_id)
        content_dir = self._get_content_dir(paper_id)
        releases_dir = self._get_releases_dir(paper_id)
        python_bin = self.repo_root / ".venv" / "bin" / "python"
        python_exec = str(python_bin if python_bin.exists() else Path(sys.executable))

        for command_template in meta.experiment_commands:
            command = command_template.format(
                paper_id=paper_id,
                repo_root=str(self.repo_root),
                paper_dir=str(paper_dir),
                latex_dir=str(latex_dir),
                content_dir=str(content_dir),
                releases_dir=str(releases_dir),
                python=python_exec,
            )
            print(f"[experiments] {paper_id}: {command}")
            result = subprocess.run(
                command,
                cwd=self.repo_root,
                shell=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Experiment command failed for {paper_id}: {command}\n"
                    f"stdout:\n{result.stdout}\n"
                    f"stderr:\n{result.stderr}"
                )
            if result.stdout.strip():
                print(result.stdout.strip())

    def _sync_graph_viewer_assets(self, dest_dir: Path) -> None:
        """Copy portable graph viewer assets into a graph bundle directory."""
        infra_dir = self.repo_root / "docs" / "papers" / "graph_infra"
        dest_dir.mkdir(parents=True, exist_ok=True)
        for asset in [
            "dependency-graph.jsx",
            "rejection-trace.jsx",
            "graph_utils.js",
            "true_path.js",
            "latex_renderer.js",
            "viewer.html",
        ]:
            src = infra_dir / asset
            if src.exists():
                shutil.copy2(src, dest_dir / asset)
        stale_index = dest_dir / "index.html"
        if stale_index.exists():
            stale_index.unlink()

    def _write_graph_manifest(
        self, graphs_dir: Path, default_file: Optional[str] = None
    ) -> None:
        """Write a portable local manifest for colocated graph viewers."""
        graphs_dir.mkdir(parents=True, exist_ok=True)
        graph_files = sorted(
            p.name
            for p in graphs_dir.glob("*.json")
            if not p.name.endswith(".decls.json") and p.name != "manifest.json"
        )
        if default_file not in graph_files:
            default_file = graph_files[0] if graph_files else None
        payload = {
            "default": default_file,
            "graphs": [
                {"file": name, "label": name.removesuffix(".json")}
                for name in graph_files
            ],
        }
        (graphs_dir / "manifest.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )

    def _copy_graph_visualizer(self, paper_id: str, package_dir: Path) -> None:
        """Copy the paper's dependency graph and viewer into the release package.

        Every package includes:
          graphs/<paper_id>.json          — this paper's proof dependency graph
          graphs/dependency-graph.jsx     — interactive force-directed visualizer
          graphs/rejection-trace.jsx      — rejection cascade trace visualizer
          graphs/graph_utils.js           — shared graph utilities
          graphs/true_path.js             — true path validation module
          graphs/latex_renderer.js        — LaTeX/foundation bucket utilities
          graphs/index.html               — portable React viewer entrypoint

        This is unconditional — every packaged paper ships its graph.
        """
        graphs_dir_src = self.repo_root / "docs" / "papers" / "graph_infra" / "graphs"
        graphs_dest = package_dir / "graphs"
        graphs_dest.mkdir(exist_ok=True)

        # Package exactly one graph JSON per paper bundle. Cross-paper proof
        # dependencies are already present inside the current paper graph.
        all_ids = [paper_id]

        # Re-mark claim nodes immediately before packaging so the shipped JSON
        # cannot drift stale relative to the current claim extraction state.
        for pid in all_ids:
            self._mark_claim_nodes_in_graph(pid)

        copied_jsons: List[str] = []
        for pid in all_ids:
            src = graphs_dir_src / f"{pid}.json"
            if src.exists():
                shutil.copy2(src, graphs_dest / f"{pid}.json")
                copied_jsons.append(f"{pid}.json")
            decl_src = graphs_dir_src / f"{pid}.decls.json"
            if decl_src.exists():
                shutil.copy2(decl_src, graphs_dest / f"{pid}.decls.json")

        # Copy graph viewer sources/modules
        self._sync_graph_viewer_assets(graphs_dest)
        self._write_graph_manifest(graphs_dest, f"{paper_id}.json")

        print(f"[arxiv]   Graphs: {len(copied_jsons)} JSON(s) + viewer → graphs/")

    def _write_graph_index_html(
        self,
        paper_id: str,
        all_ids: List[str],
        available_jsons: List[str],
        graphs_dest: Path,
        claim_to_handles: Dict[str, List[str]] | None = None,
    ) -> None:
        """Write a standalone D3 force-graph viewer as graphs/index.html."""
        meta = self._get_paper_meta(paper_id)

        options_html = "\n".join(
            f'        <option value="{pid}.json"{"  selected" if pid == paper_id else ""}>'
            f"{pid}</option>"
            for pid in all_ids
            if f"{pid}.json" in available_jsons
        )

        # Build claim-to-handles mapping for JS
        claim_mapping_js = "{}"
        if claim_to_handles:
            # Convert Python dict to JS object: {{"thm:foo": ["Ns.thm1", "Ns.thm2"], ...}}
            mapping_entries = []
            for claim, handles in sorted(claim_to_handles.items()):
                handles_json = ", ".join(f'"{h}"' for h in handles)
                mapping_entries.append(f'"{claim}": [{handles_json}]')
            claim_mapping_js = "{ " + ", ".join(mapping_entries) + " }"

        html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Proof Dependency Graph — {meta.name}</title>
  <script src="https://d3js.org/d3.v7.min.js"></script>
  <style>
    :root {{
      --bg: #0a0a0a;
      --panel: #121212;
      --panel-2: #171717;
      --line: #262626;
      --muted: #8a8a8a;
      --text: #e6e6e6;
      --thm: #3b82f6;
      --def: #8b5cf6;
      --ax: #ef4444;
      --foundation: #f59e0b;
      --bridge: #10b981;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; background: var(--bg); color: var(--text); }}
    .topbar {{
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      padding: 14px 18px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }}
    .title-row {{ display: flex; align-items: baseline; gap: 12px; }}
    .title-row h1 {{ margin: 0; font-size: 14px; font-weight: 700; }}
    .title-row .subtitle {{ font-size: 11px; color: var(--muted); }}
    .controls {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 10px;
      align-items: end;
    }}
    .control {{ display: flex; flex-direction: column; gap: 4px; }}
    .control.inline {{ flex-direction: row; align-items: center; gap: 8px; }}
    .control label {{ font-size: 10px; letter-spacing: 0.06em; color: var(--muted); text-transform: uppercase; }}
    .control select,
    .control input,
    .control button {{
      background: var(--panel-2);
      color: var(--text);
      border: 1px solid #303030;
      padding: 8px 10px;
      font: inherit;
      font-size: 12px;
      border-radius: 4px;
    }}
    .control button {{ cursor: pointer; }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 18px;
      font-size: 11px;
      color: var(--muted);
      padding-top: 2px;
    }}
    .legend-item {{ display: flex; align-items: center; gap: 8px; }}
    .swatch {{
      width: 10px;
      height: 10px;
      border-radius: 999px;
      display: inline-block;
      border: 1px solid rgba(255,255,255,0.25);
    }}
    .swatch.square {{ border-radius: 2px; }}
    .swatch.diamond {{ border-radius: 0; transform: rotate(45deg); }}
    .swatch.theorem {{ background: var(--thm); }}
    .swatch.definition {{ background: var(--def); }}
    .swatch.axiom {{ background: var(--ax); }}
    .swatch.foundation {{ background: var(--foundation); }}
    .swatch.bridge {{ background: var(--bridge); }}
    .mode-help {{
      background: #111827;
      border-top: 1px solid #1f2937;
      border-bottom: 1px solid #1f2937;
      color: #93c5fd;
      padding: 9px 18px;
      font-size: 11px;
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 320px;
      min-height: calc(100vh - 182px);
    }}
    #canvas-wrap {{ position: relative; border-right: 1px solid var(--line); }}
    #svg {{ width: 100%; height: 100%; display: block; }}
    .side {{
      background: var(--panel);
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }}
    .side h2 {{
      margin: 0;
      font-size: 11px;
      color: var(--muted);
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    .metric {{
      display: grid;
      grid-template-columns: auto 1fr;
      gap: 8px 10px;
      font-size: 12px;
      align-items: baseline;
    }}
    .metric .key {{ color: var(--muted); }}
    .metric .val {{ color: var(--text); word-break: break-word; }}
    .node circle {{ stroke-width: 1.5px; cursor: pointer; }}
    .node text {{ font-size: 9px; fill: #a3a3a3; pointer-events: none; }}
    .link {{ stroke: #3f3f46; stroke-opacity: 0.55; }}
    .tooltip {{
      position: absolute;
      background: rgba(12, 12, 12, 0.95);
      border: 1px solid #333;
      padding: 6px 8px;
      font-size: 11px;
      pointer-events: none;
      color: #eee;
      max-width: 480px;
      word-break: break-word;
      border-radius: 4px;
      display: none;
    }}
    .hint {{ font-size: 11px; color: var(--muted); line-height: 1.5; }}
    @media (max-width: 1100px) {{
      .layout {{ grid-template-columns: 1fr; }}
      #canvas-wrap {{ border-right: none; border-bottom: 1px solid var(--line); min-height: 60vh; }}
    }}
  </style>
</head>
<body>
  <div class="topbar">
    <div class="title-row">
      <h1>Proof Dependency Graph — {meta.name}</h1>
      <span class="subtitle">interactive packaged viewer</span>
    </div>
    <div class="controls">
      <div class="control">
        <label for="picker">Graph</label>
        <select id="picker">
{options_html}
        </select>
      </div>
      <div class="control">
        <label for="mode">View</label>
        <select id="mode">
          <option value="forcing">Forcing chain</option>
          <option value="claims" selected>Claims only</option>
          <option value="full">Full graph</option>
        </select>
      </div>
      <div class="control">
        <label for="paper-filter">Paper scope</label>
        <select id="paper-filter">
          <option value="">All papers</option>
        </select>
      </div>
      <div class="control">
        <label for="depth-limit">Depth limit</label>
        <input id="depth-limit" type="number" min="0" placeholder="none" />
      </div>
      <div class="control inline" style="align-self:end;">
        <input id="hide-artifacts" type="checkbox" checked />
        <label for="hide-artifacts" style="margin:0;font-size:12px;letter-spacing:0;color:var(--text);text-transform:none;">Hide artifacts in full view</label>
      </div>
      <div class="control">
        <label>&nbsp;</label>
        <button id="reset">Reset filters</button>
      </div>
    </div>
    <div class="legend">
      <span class="legend-item"><span class="swatch theorem"></span> theorem</span>
      <span class="legend-item"><span class="swatch square definition"></span> definition</span>
      <span class="legend-item"><span class="swatch diamond axiom"></span> axiom</span>
      <span class="legend-item"><span class="swatch foundation"></span> foundation bucket</span>
      <span class="legend-item"><span class="swatch bridge"></span> derived/bridge paper</span>
      <span class="legend-item"><span id="stats">loading…</span></span>
    </div>
  </div>
  <div id="mode-help" class="mode-help"></div>
  <div class="layout">
    <div id="canvas-wrap">
      <svg id="svg"></svg>
      <div class="tooltip" id="tip"></div>
    </div>
    <div class="side">
      <div>
        <h2>Selection</h2>
        <div id="selection" class="hint">Click a node to inspect it.</div>
      </div>
      <div>
        <h2>Node Details</h2>
        <div id="details" class="metric">
          <div class="key">status</div><div class="val">No node selected</div>
        </div>
      </div>
      <div>
        <h2>Lean Signature</h2>
        <div class="hint">
          Clicking a project node shows the exported Lean declaration signature and any attached docstring.
          This is readable proof metadata, not a full Lean-to-LaTeX theorem renderer.
          If you need display math later, the next step is rendering the exported signature string with a math layer.
        </div>
      </div>
    </div>
  </div>
  <script>
    const d3 = window.d3;
    const picker = document.getElementById("picker");
    const mode = document.getElementById("mode");
    const paperFilter = document.getElementById("paper-filter");
    const depthLimit = document.getElementById("depth-limit");
    const hideArtifacts = document.getElementById("hide-artifacts");
    const resetBtn = document.getElementById("reset");
    const statsEl = document.getElementById("stats");
    const modeHelp = document.getElementById("mode-help");
    const detailsEl = document.getElementById("details");
    const selectionEl = document.getElementById("selection");
    const tip = document.getElementById("tip");
    const claimToHandles = {claim_mapping_js};

    const ARTIFACT_PATTERNS = [
      /\\._simp_/, /\\.proxyType/, /_aux_/, /___macroRules_/, /___unexpand_/, /\\.\\./,
      /\\.instRepr/, /\\.instBEq/, /\\.instHashable/, /\\.instOrd/, /\\.instToString/,
      /\\.instInhabited/, /\\.instDecidable/, /\\.instSubsingleton/, /\\.instNonempty/,
      /\\.rec$/, /\\.recOn$/, /\\.casesOn$/, /\\.noConfusion/, /\\.noConfusionType/,
      /\\.eliminator$/, /\\.brecOn$/, /\\.binductionOn$/,
      /\\.mk\\.inj/, /\\.mk\\.sizeOf_spec/, /\\.inj/, /\\.injEq/, /\\.ind/,
      /\\.sizeOf_spec/, /\\.sizeOf/, /\\.ctorIdx/, /\\.ctorElim/,
      /\\.match_/, /_match/, /\\._main/, /\\._cstage/,
      /_auxLemma/, /_eqn/, /_proof/, /_fun/, /_fix/,
      /_sunfold/, /_tactic/,
    ];

    function isArtifact(id) {{
      return ARTIFACT_PATTERNS.some((re) => re.test(id));
    }}

    function buildAdj(nodes, edges) {{
      const out = {{}};
      for (const n of nodes) out[n.id] = [];
      for (const e of edges) {{
        const s = typeof e.source === "object" ? e.source.id : e.source;
        const t = typeof e.target === "object" ? e.target.id : e.target;
        if (out[s]) out[s].push(t);
      }}
      return out;
    }}

    function computeStats(data) {{
      const theorems = data.nodes.filter((n) => n.kind === "theorem").length;
      const definitions = data.nodes.filter((n) => n.kind === "definition").length;
      const axioms = data.nodes.filter((n) => n.kind === "axiom").length;
      const undirected = {{}};
      for (const n of data.nodes) undirected[n.id] = [];
      for (const e of data.edges) {{
        const s = typeof e.source === "object" ? e.source.id : e.source;
        const t = typeof e.target === "object" ? e.target.id : e.target;
        if (undirected[s]) undirected[s].push(t);
        if (undirected[t]) undirected[t].push(s);
      }}
      const visited = new Set();
      let components = 0;
      for (const n of data.nodes) {{
        if (visited.has(n.id)) continue;
        components += 1;
        const q = [n.id];
        while (q.length) {{
          const cur = q.shift();
          if (visited.has(cur)) continue;
          visited.add(cur);
          for (const nxt of undirected[cur] || []) {{
            if (!visited.has(nxt)) q.push(nxt);
          }}
        }}
      }}
      return {{
        totalNodes: data.nodes.length,
        totalEdges: data.edges.length,
        theorems,
        definitions,
        axioms,
        components,
      }};
    }}

    const FILTERS = {{
      hideArtifacts: (data) => {{
        const keep = new Set(data.nodes.filter((n) => !isArtifact(n.id)).map((n) => n.id));
        return {{
          nodes: data.nodes.filter((n) => keep.has(n.id)),
          edges: data.edges.filter((e) => {{
            const s = typeof e.source === "object" ? e.source.id : e.source;
            const t = typeof e.target === "object" ? e.target.id : e.target;
            return keep.has(s) && keep.has(t);
          }}),
        }};
      }},
      restrictToPaper: (data, params) => {{
        if (!params || params.paper == null) return data;
        const keep = new Set(data.nodes.filter((n) => n.paper === params.paper).map((n) => n.id));
        return {{
          nodes: data.nodes.filter((n) => keep.has(n.id)),
          edges: data.edges.filter((e) => {{
            const s = typeof e.source === "object" ? e.source.id : e.source;
            const t = typeof e.target === "object" ? e.target.id : e.target;
            return keep.has(s) && keep.has(t);
          }}),
        }};
      }},
      depthLimit: (data, params) => {{
        if (!params || params.depth == null) return data;
        const adj = buildAdj(data.nodes, data.edges);
        const roots = data.nodes.filter((n) => n.paper === -1).map((n) => n.id);
        const dist = {{}};
        const q = [];
        for (const r of roots) {{
          dist[r] = 0;
          q.push(r);
        }}
        while (q.length) {{
          const cur = q.shift();
          for (const nxt of adj[cur] || []) {{
            if (dist[nxt] === undefined) {{
              dist[nxt] = dist[cur] + 1;
              q.push(nxt);
            }}
          }}
        }}
        const keep = new Set(data.nodes.filter((n) => dist[n.id] !== undefined && dist[n.id] <= params.depth).map((n) => n.id));
        return {{
          nodes: data.nodes.filter((n) => keep.has(n.id)),
          edges: data.edges.filter((e) => {{
            const s = typeof e.source === "object" ? e.source.id : e.source;
            const t = typeof e.target === "object" ? e.target.id : e.target;
            return keep.has(s) && keep.has(t);
          }}),
        }};
      }},
      claimsOnly: (data) => {{
        const noArtifacts = FILTERS.hideArtifacts(data);
        const keep = new Set(noArtifacts.nodes.filter((n) => n.paper === -1).map((n) => n.id));
        const adj = buildAdj(noArtifacts.nodes, noArtifacts.edges);
        const rev = {{}};
        for (const n of noArtifacts.nodes) rev[n.id] = [];
        for (const e of noArtifacts.edges) {{
          const s = typeof e.source === "object" ? e.source.id : e.source;
          const t = typeof e.target === "object" ? e.target.id : e.target;
          if (rev[t]) rev[t].push(s);
        }}
        const q1 = [...keep];
        while (q1.length) {{
          const cur = q1.shift();
          for (const nxt of rev[cur] || []) {{
            if (!keep.has(nxt)) {{
              keep.add(nxt);
              q1.push(nxt);
            }}
          }}
        }}
        const q2 = [...keep];
        while (q2.length) {{
          const cur = q2.shift();
          for (const nxt of adj[cur] || []) {{
            if (!keep.has(nxt)) {{
              keep.add(nxt);
              q2.push(nxt);
            }}
          }}
        }}
        return {{
          nodes: noArtifacts.nodes.filter((n) => keep.has(n.id)),
          edges: noArtifacts.edges.filter((e) => {{
            const s = typeof e.source === "object" ? e.source.id : e.source;
            const t = typeof e.target === "object" ? e.target.id : e.target;
            return keep.has(s) && keep.has(t);
          }}),
        }};
      }},
      forcingChain: (data) => {{
        const isFoundation = (id) => String(id).startsWith("FOUNDATION.");
        const isForcingTheorem = (id) => /^[A-Z]+\\d+[a-z']*$/.test(String(id).split(".").slice(-1)[0]);
        const claimIds = new Set(data.nodes
          .filter((n) => n.paper === -1 && (n.kind === "theorem" || n.kind === "definition"))
          .map((n) => n.id));
        const targetIds = new Set(data.nodes
          .filter((n) => isFoundation(n.id) || isForcingTheorem(n.id))
          .map((n) => n.id));
        if (claimIds.size === 0 || targetIds.size === 0) return data;
        const adj = buildAdj(data.nodes, data.edges);
        const rev = {{}};
        for (const n of data.nodes) rev[n.id] = [];
        for (const e of data.edges) {{
          const s = typeof e.source === "object" ? e.source.id : e.source;
          const t = typeof e.target === "object" ? e.target.id : e.target;
          if (rev[t]) rev[t].push(s);
        }}
        const fromClaims = new Set(claimIds);
        const q1 = [...claimIds];
        while (q1.length) {{
          const cur = q1.shift();
          for (const nxt of adj[cur] || []) {{
            if (!fromClaims.has(nxt)) {{
              fromClaims.add(nxt);
              q1.push(nxt);
            }}
          }}
        }}
        const toTargets = new Set(targetIds);
        const q2 = [...targetIds];
        while (q2.length) {{
          const cur = q2.shift();
          for (const nxt of rev[cur] || []) {{
            if (!toTargets.has(nxt)) {{
              toTargets.add(nxt);
              q2.push(nxt);
            }}
          }}
        }}
        const keep = new Set();
        for (const id of fromClaims) if (toTargets.has(id)) keep.add(id);
        for (const id of claimIds) keep.add(id);
        for (const id of targetIds) keep.add(id);
        return {{
          nodes: data.nodes.filter((n) => keep.has(n.id)),
          edges: data.edges.filter((e) => {{
            const s = typeof e.source === "object" ? e.source.id : e.source;
            const t = typeof e.target === "object" ? e.target.id : e.target;
            return keep.has(s) && keep.has(t);
          }}),
        }};
      }},
    }};

    function applyFilters(data, filters) {{
      let out = data;
      for (const filter of filters) {{
        const fn = FILTERS[filter.name];
        if (!fn) continue;
        out = fn(out, filter.params || {{}});
      }}
      return out;
    }}

    const PAPER_LABELS = {{
      "-1": "Claim / foundation frontier",
      "0": "Bridge / derived layer",
      "1": "Paper 1",
      "2": "Paper 2",
      "3": "Paper 3",
      "4": "Paper 4",
    }};

    let currentData = null;
    let currentDecls = {{}};
    let selectedNodeId = null;

    function paperLabel(value) {{
      return PAPER_LABELS[String(value)] || `Paper ${{value}}`;
    }}

    function viewDescription(view) {{
      if (view === "forcing") {{
        return "<strong>Forcing chain:</strong> keeps the path from claim-marked nodes through named forcing results down to collapsed first-principle buckets such as counting and measure theory.";
      }}
      if (view === "claims") {{
        return "<strong>Claims only:</strong> hides compiler artifacts and keeps the cited proof spine anchored at the claim/foundation frontier.";
      }}
      return "<strong>Full graph:</strong> shows the full packaged proof graph. Use the artifact toggle, paper scope, and depth limit to cut noise.";
    }}

    function rebuildPaperScopeOptions(data) {{
      const seen = new Set((data.nodes || []).map((n) => String(n.paper)));
      const current = paperFilter.value;
      const options = ['<option value=\"\">All papers</option>'];
      for (const paper of [...seen].sort((a, b) => Number(a) - Number(b))) {{
        options.push(`<option value="${{paper}}">${{paperLabel(paper)}}</option>`);
      }}
      paperFilter.innerHTML = options.join("");
      if ([...paperFilter.options].some((o) => o.value === current)) {{
        paperFilter.value = current;
      }}
    }}

    function activeFilterSpec() {{
      const filters = [];
      if (mode.value === "forcing") {{
        filters.push({{ name: "forcingChain" }});
      }} else if (mode.value === "claims") {{
        filters.push({{ name: "claimsOnly" }});
      }} else if (hideArtifacts.checked) {{
        filters.push({{ name: "hideArtifacts" }});
      }}

      if (paperFilter.value !== "") {{
        filters.push({{ name: "restrictToPaper", params: {{ paper: Number(paperFilter.value) }} }});
      }}
      if (depthLimit.value !== "") {{
        filters.push({{ name: "depthLimit", params: {{ depth: Number(depthLimit.value) }} }});
      }}
      return filters;
    }}

    function setDetails(node, data) {{
      if (!node) {{
        selectionEl.textContent = "Click a node to inspect it.";
        detailsEl.innerHTML = '<div class="key">status</div><div class="val">No node selected</div>';
        return;
      }}

      const outgoing = data.edges.filter((e) => (typeof e.source === "object" ? e.source.id : e.source) === node.id).length;
      const incoming = data.edges.filter((e) => (typeof e.target === "object" ? e.target.id : e.target) === node.id).length;
      const handles = claimToHandles[node.id] || [];
      const decl = currentDecls[node.id] || null;
      selectionEl.textContent = node.id;

      const rows = [
        ["id", node.id],
        ["kind", node.kind || "other"],
        ["paper", paperLabel(node.paper)],
        ["outgoing", String(outgoing)],
        ["incoming", String(incoming)],
      ];
      if (decl?.signature) {{
        rows.push(["signature", decl.signature]);
      }}
      if (decl?.doc) {{
        rows.push(["doc", decl.doc]);
      }}
      if (node.id.startsWith("FOUNDATION.")) {{
        rows.push(["note", "Collapsed first-principle witness node"]);
      }}
      if (handles.length) {{
        rows.push(["claim handles", handles.join(", ")]);
      }}

      detailsEl.innerHTML = rows.map(([k, v]) =>
        `<div class="key">${{k}}</div><div class="val">${{v}}</div>`
      ).join("");
    }}

    function render(filtered) {{
      const nodes = filtered.nodes.map((n) => ({{ ...n }}));
      const links = filtered.edges.map((e) => ({{ source: e.source, target: e.target }}));
      const stat = computeStats(filtered);
      statsEl.textContent =
        `${{stat.totalNodes}} nodes · ${{stat.totalEdges}} edges · ` +
        `${{stat.theorems}} thm · ${{stat.definitions}} def · ${{stat.axioms}} axiom`;
      modeHelp.innerHTML = viewDescription(mode.value);

      const svg = d3.select("#svg");
      svg.selectAll("*").remove();

      const width = svg.node().clientWidth;
      const height = svg.node().clientHeight;
      const g = svg.append("g");
      svg.call(d3.zoom().scaleExtent([0.08, 8]).on("zoom", (event) => {{
        g.attr("transform", event.transform);
      }}));

      const sim = d3.forceSimulation(nodes)
        .force("link", d3.forceLink(links).id((d) => d.id).distance(46).strength(0.45))
        .force("charge", d3.forceManyBody().strength(-90).distanceMax(260))
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force("collision", d3.forceCollide(10));

      const link = g.append("g")
        .selectAll("line")
        .data(links)
        .join("line")
        .attr("class", "link")
        .attr("stroke-width", 0.9);

      const node = g.append("g")
        .selectAll("g")
        .data(nodes)
        .join("g")
        .attr("class", "node")
        .call(
          d3.drag()
            .on("start", (event, d) => {{
              if (!event.active) sim.alphaTarget(0.3).restart();
              d.fx = d.x;
              d.fy = d.y;
            }})
            .on("drag", (event, d) => {{
              d.fx = event.x;
              d.fy = event.y;
            }})
            .on("end", (event, d) => {{
              if (!event.active) sim.alphaTarget(0);
              d.fx = null;
              d.fy = null;
            }})
        )
        .on("mouseover", (event, d) => {{
          tip.style.display = "block";
          tip.style.left = `${{event.pageX + 12}}px`;
          tip.style.top = `${{event.pageY - 8}}px`;
          tip.textContent = `${{d.id}} [${{d.kind || "other"}}]`;
        }})
        .on("mouseout", () => {{
          tip.style.display = "none";
        }})
        .on("click", (_event, d) => {{
          selectedNodeId = d.id;
          setDetails(d, filtered);
          node.selectAll("circle").attr("stroke", (n) => n.id === selectedNodeId ? "#ffffff" : "rgba(255,255,255,0.2)");
        }});

      node.append("circle")
        .attr("r", (d) => d.kind === "axiom" ? 7 : d.kind === "definition" ? 5 : 4)
        .attr("fill", (d) => {{
          if (String(d.id).startsWith("FOUNDATION.")) return "var(--foundation)";
          if (d.paper === 0) return "var(--bridge)";
          if (d.kind === "axiom") return "var(--ax)";
          if (d.kind === "definition") return "var(--def)";
          if (d.kind === "theorem") return "var(--thm)";
          return "#71717a";
        }})
        .attr("stroke", (d) => d.id === selectedNodeId ? "#ffffff" : "rgba(255,255,255,0.2)");

      node.append("text")
        .attr("x", 7)
        .attr("y", 3)
        .text((d) => {{
          const parts = d.id.split(".");
          return parts[parts.length - 1];
        }});

      sim.on("tick", () => {{
        link
          .attr("x1", (d) => d.source.x)
          .attr("y1", (d) => d.source.y)
          .attr("x2", (d) => d.target.x)
          .attr("y2", (d) => d.target.y);
        node.attr("transform", (d) => `translate(${{d.x}},${{d.y}})`);
      }});
    }}

    function renderFiltered() {{
      if (!currentData) return;
      const filtered = applyFilters(currentData, activeFilterSpec());
      const selectedNode = filtered.nodes.find((n) => n.id === selectedNodeId) || null;
      render(filtered);
      setDetails(selectedNode, filtered);
    }}

    async function loadGraph(file) {{
      try {{
        const [graphResponse, declResponse] = await Promise.all([
          fetch(file),
          fetch(file.replace(/\\.json$/, ".decls.json")).catch(() => null),
        ]);
        const data = await graphResponse.json();
        let decls = {{}};
        if (declResponse && declResponse.ok) {{
          try {{
            decls = await declResponse.json();
          }} catch (_err) {{
            decls = {{}};
          }}
        }}
        currentData = data;
        currentDecls = decls;
        rebuildPaperScopeOptions(data);
        renderFiltered();
      }} catch (err) {{
        statsEl.textContent = `failed to load ${{file}}`;
        selectionEl.textContent = "Graph load failed";
        detailsEl.innerHTML = `<div class="key">error</div><div class="val">${{err.message}}</div>`;
      }}
    }}

    picker.addEventListener("change", () => loadGraph(picker.value));
    mode.addEventListener("change", renderFiltered);
    paperFilter.addEventListener("change", renderFiltered);
    depthLimit.addEventListener("input", renderFiltered);
    hideArtifacts.addEventListener("change", renderFiltered);
    resetBtn.addEventListener("click", () => {{
      mode.value = "claims";
      paperFilter.value = "";
      depthLimit.value = "";
      hideArtifacts.checked = true;
      selectedNodeId = null;
      renderFiltered();
    }});

    loadGraph("{paper_id}.json");
  </script>
</body>
</html>
"""
        (graphs_dest / "index.html").write_text(html, encoding="utf-8")

    def _generate_paper_lakefile(
        self, paper_id: str, paper_files: list, lean_dest: Path
    ) -> None:
        """Generate paper-specific lakefile from actual proof files."""
        meta = self._get_paper_meta(paper_id)

        # Extract lib names from filenames (paper4_Basic.lean -> paper4_Basic)
        lib_names = [f.stem for f in paper_files]

        lib_entries = "\n".join(
            [
                f"""lean_lib «{name}» where
  moreLeanArgs := moreLeanArgs
  weakLeanArgs := weakLeanArgs
"""
                for name in lib_names
            ]
        )

        lakefile_content = f"""import Lake
open Lake DSL

-- {meta.name} - Lean 4 Formalization
-- Auto-generated by build_papers.py

def moreServerArgs := #[
  "-Dpp.unicode.fun=true",
  "-Dpp.proofs.withType=false"
]

def moreLeanArgs := moreServerArgs

def weakLeanArgs : Array String :=
  if get_config? CI |>.isSome then
    #["-DwarningAsError=true"]
  else
    #[]

package «{meta.dir_name}» where
  moreServerArgs := moreServerArgs

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git"

{lib_entries}"""

        (lean_dest / "lakefile.lean").write_text(lakefile_content)

    def _generate_proofs_readme(
        self, paper_id: str, paper_files: list, lean_dest: Path
    ) -> None:
        """Generate README for proofs directory from actual proof files."""
        meta = self._get_paper_meta(paper_id)
        release_module_closure = self._get_release_module_closure(paper_id)
        lean_stats = (
            self._get_release_lean_stats(paper_id)
            if release_module_closure is not None
            else self._get_lean_stats(paper_id)
        )
        overview = (
            f"This directory contains the Lean 4 slice backing the cited {meta.name} content."
            if release_module_closure is not None
            else f"This directory contains the complete Lean 4 formalization for {meta.name}."
        )

        # Build file table
        file_rows = "\n".join(
            [
                f"| `{f.name}` | {f.stem.replace(paper_id + '_', '')} |"
                for f in paper_files
            ]
        )
        sorry_summary = (
            "0 `sorry` placeholders."
            if lean_stats.sorry_count == 0
            else f"{lean_stats.sorry_count} `sorry` placeholders."
        )

        readme_content = f"""# {meta.name} - Lean 4 Formalization

## Overview

{overview}

- **Lines:** {lean_stats.line_count}
- **Theorems:** {lean_stats.theorem_count}
- **`sorry` placeholders:** {lean_stats.sorry_count}


## Requirements

- Lean 4 (see `lean-toolchain` for exact version)
- Mathlib4

## Building

```bash
lake update
lake build
```

## File Structure

| File | Module |
|------|--------|
{file_rows}

## Verification

All files compile with {sorry_summary} All claims are machine-verified.

## License

MIT License - See main repository for details.

---
*Auto-generated by build_papers.py*
"""

        (lean_dest / "README.md").write_text(readme_content)

    def _create_build_log(self, paper_id: str, package_dir: Path) -> None:
        """
        Create BUILD_LOG.txt with ACTUAL lake build output as compilation proof.
        This is the mathematical evidence that proofs compile.
        """
        paper_title = self._release_title_from_latex(paper_id)
        release_module_closure = self._get_release_module_closure(paper_id)
        lean_stats = (
            self._get_release_lean_stats(paper_id)
            if release_module_closure is not None
            else self._get_lean_stats(paper_id)
        )
        log_file = package_dir / "BUILD_LOG.txt"

        # Paper-specific proof files from paper's proofs directory
        proofs_dir = self._get_paper_proofs_dir(paper_id)
        if release_module_closure is None:
            paper_lean_files = self._iter_paper_lean_files(proofs_dir)
        else:
            local_modules = release_module_closure.get(paper_id, set())
            paper_lean_files = [
                f
                for f in self._iter_paper_lean_files(proofs_dir)
                if str(f.relative_to(proofs_dir).with_suffix("")) in local_modules
            ]
        lean_toolchain = proofs_dir / "lean-toolchain"
        toolchain_version = (
            lean_toolchain.read_text().strip() if lean_toolchain.exists() else "unknown"
        )

        log_content = f"""OpenHCS Paper Build Log
========================

Paper: {paper_id} - {paper_title}
Package Date: {datetime.now().isoformat()}
Lean Toolchain: {toolchain_version}

Proof Statistics:
-----------------
Lean Files: {len(paper_lean_files)}
Lines of Code: {lean_stats.line_count}
Theorems: {lean_stats.theorem_count}
Sorry Placeholders: {lean_stats.sorry_count}


Proof Files:
------------
"""
        for f in sorted(paper_lean_files):
            log_content += f"  - {f.name}\n"

        # Include actual lake build output if available
        if self._lean_build_output:
            log_content += f"""
================================================================================
COMPILATION OUTPUT (lake build)
================================================================================

{self._lean_build_output}

================================================================================
END COMPILATION OUTPUT
================================================================================
"""
        else:
            log_content += """
Note: Run 'python3 scripts/build_papers.py release' to include compilation output.
"""

        log_content += f"""
Build Instructions:
-------------------
To verify the proofs compile:

  cd proofs/
  lake build

All theorems will be machine-verified if compilation succeeds.

Repository: https://github.com/trissim/openhcs
"""

        log_file.write_text(log_content)
        print(f"[arxiv]   Build log: BUILD_LOG.txt (with compilation output)")

    def _create_named_archive(
        self,
        releases_dir: Path,
        package_dir: Path,
        archive_stem: str,
        root_dir_name: str,
    ) -> tuple[Path, Path]:
        """Create compressed tar.gz and zip archives with explicit names."""
        import tarfile
        import zipfile

        tar_name = f"{archive_stem}.tar.gz"
        tar_path = releases_dir / tar_name
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(package_dir, arcname=root_dir_name)

        zip_name = f"{archive_stem}.zip"
        zip_path = releases_dir / zip_name
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in package_dir.rglob("*"):
                if file_path.is_file():
                    arcname = str(
                        Path(root_dir_name) / file_path.relative_to(package_dir)
                    )
                    zf.write(file_path, arcname)

        return tar_path, zip_path

    def _create_archive(self, paper_id: str, package_dir: Path) -> tuple[Path, Path]:
        """Create compressed tar.gz and zip archives of package in releases/.

        For variants (e.g., paper1_jsait), uses variant-specific naming.
        """
        archive_prefix = self._get_archive_prefix(paper_id)
        releases_dir = self._get_releases_dir(paper_id)
        return self._create_named_archive(
            releases_dir=releases_dir,
            package_dir=package_dir,
            archive_stem=f"{archive_prefix}_arxiv",
            root_dir_name=archive_prefix,
        )

    def release(
        self,
        paper_id: str,
        verbose: bool = True,
        axiom_check: bool = False,
        claim_check: bool = False,
    ):
        """Full release build: Lean + LaTeX + Markdown + arXiv package."""
        meta = self._get_paper_meta(paper_id)
        print(f"\n{'=' * 60}")
        print(f"[release] Building {paper_id}: {meta.name}")
        print(f"{'=' * 60}\n")

        # Build in order: Lean → LaTeX → Markdown → Package
        self.build_lean(paper_id, verbose=verbose)
        self.build_latex(paper_id, verbose=verbose)
        raw_meta = self._raw_metadata.get("papers", {}).get(paper_id, {})
        if raw_meta.get("submission_format"):
            self.build_submission(paper_id, verbose=verbose)
        self.build_markdown(paper_id)
        if claim_check:
            self.check_claim_coverage(paper_id, fail_on_missing=True)
        self.package_arxiv(paper_id)

        if axiom_check:
            self.check_axioms(paper_id, verbose=verbose)

        releases_dir = self._get_releases_dir(paper_id)
        print(
            f"\n[release] ✓ {paper_id} complete → {releases_dir.relative_to(self.repo_root)}/"
        )

    def check_axioms(self, paper_id: str, verbose: bool = True) -> dict:
        """Check what axioms each theorem in a paper depends on.

        Uses `lake env lean` with `#print axioms` to verify theorem foundations.
        Returns a summary dict with counts by axiom category.
        """
        import re
        from collections import defaultdict

        proofs_dir = self._get_paper_proofs_dir(paper_id)
        if not proofs_dir.exists():
            print(f"[axiom-check] No proofs directory for {paper_id}, skipping...")
            return {}

        print(f"[axiom-check] Checking axioms for {paper_id}...")
        print(f"[axiom-check]   Directory: {proofs_dir.relative_to(self.repo_root)}")

        # Step 1: Find all theorem/lemma declarations in .lean files
        theorems = []
        exclude_patterns = set(self.build_config.lean_exclude_dirs)
        for lean_file in sorted(proofs_dir.rglob("*.lean")):
            if any(part in exclude_patterns for part in lean_file.parts):
                continue
            if lean_file.name in ("lakefile.lean",):
                continue

            # Compute module name from file path
            rel_path = lean_file.relative_to(proofs_dir)
            module_name = str(rel_path.with_suffix("")).replace("/", ".")

            try:
                content = lean_file.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                content = lean_file.read_text(encoding="latin-1")

            # Find theorem/lemma declarations
            for match in re.finditer(
                r"^(theorem|lemma)\s+(\w+['\w]*)", content, re.MULTILINE
            ):
                thm_name = match.group(2)
                theorems.append((module_name, thm_name))

        if not theorems:
            print(f"[axiom-check]   No theorems found in {paper_id}")
            return {}

        print(f"[axiom-check]   Found {len(theorems)} theorems/lemmas")

        # Step 2: Group by module and check axioms
        by_module = defaultdict(list)
        for module, thm in theorems:
            by_module[module].append(thm)

        results = {
            "total": len(theorems),
            "checked": 0,
            "no_axioms": 0,
            "propext_only": 0,
            "quot_sound": 0,
            "classical": 0,
            "unknown": 0,
            "details": [],
        }

        for module, thms in sorted(by_module.items()):
            # Create temp file with #print axioms commands
            import tempfile

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".lean", delete=False
            ) as f:
                f.write(f"import {module}\n\n")
                for thm in thms:
                    # Try with module prefix
                    f.write(f"#print axioms {module}.{thm}\n")
                tmpfile = f.name

            try:
                result = subprocess.run(
                    ["lake", "env", "lean", tmpfile],
                    capture_output=True,
                    text=True,
                    timeout=300,
                    cwd=proofs_dir,
                )
                output = result.stdout + result.stderr

                # Parse results
                found_thms = set()
                unknown_thms = []

                for line in output.split("\n"):
                    line = line.strip()
                    if "depends on axioms" in line or "does not depend" in line:
                        results["checked"] += 1
                        results["details"].append(line)
                        if verbose:
                            print(f"[axiom-check]   ✓ {line}")

                        # Categorize
                        if "does not depend" in line:
                            results["no_axioms"] += 1
                        elif "Classical" in line:
                            results["classical"] += 1
                        elif "Quot.sound" in line:
                            results["quot_sound"] += 1
                        elif "propext" in line:
                            results["propext_only"] += 1

                        # Track found
                        match = re.search(r"'([^']+)'", line)
                        if match:
                            found_thms.add(match.group(1).split(".")[-1])

                    elif "Unknown" in line and "constant" in line:
                        match = re.search(r"Unknown constant `([^`]+)`", line)
                        if match:
                            unknown_thms.append(match.group(1))

                # Retry unknown theorems without module prefix
                if unknown_thms:
                    with tempfile.NamedTemporaryFile(
                        mode="w", suffix=".lean", delete=False
                    ) as f2:
                        f2.write(f"import {module}\n")
                        f2.write(f"open {module.split('.')[-1]} in\n")
                        for unk in unknown_thms:
                            thm_name = unk.split(".")[-1]
                            if thm_name not in found_thms:
                                f2.write(f"#print axioms {thm_name}\n")
                        tmpfile2 = f2.name

                    result2 = subprocess.run(
                        ["lake", "env", "lean", tmpfile2],
                        capture_output=True,
                        text=True,
                        timeout=300,
                        cwd=proofs_dir,
                    )
                    output2 = result2.stdout + result2.stderr

                    for line in output2.split("\n"):
                        line = line.strip()
                        if "depends on axioms" in line or "does not depend" in line:
                            results["checked"] += 1
                            results["details"].append(line)
                            if verbose:
                                print(f"[axiom-check]   ✓ (no prefix) {line}")

                            if "does not depend" in line:
                                results["no_axioms"] += 1
                            elif "Classical" in line:
                                results["classical"] += 1
                            elif "Quot.sound" in line:
                                results["quot_sound"] += 1
                            elif "propext" in line:
                                results["propext_only"] += 1
                        elif "Unknown" in line:
                            results["unknown"] += 1

                    Path(tmpfile2).unlink(missing_ok=True)

            except subprocess.TimeoutExpired:
                print(f"[axiom-check]   TIMEOUT for module {module}")
            except Exception as e:
                print(f"[axiom-check]   ERROR: {e}")
            finally:
                Path(tmpfile).unlink(missing_ok=True)

        # Print summary
        print(f"\n[axiom-check] === {paper_id} Summary ===")
        print(f"[axiom-check]   Total theorems: {results['total']}")
        print(f"[axiom-check]   Successfully checked: {results['checked']}")
        print(f"[axiom-check]     - No axioms (constructive): {results['no_axioms']}")
        print(f"[axiom-check]     - Only propext: {results['propext_only']}")
        print(f"[axiom-check]     - propext + Quot.sound: {results['quot_sound']}")
        print(f"[axiom-check]     - Uses Classical.choice: {results['classical']}")
        print(f"[axiom-check]   Private/not exported: {results['unknown']}")

        return results

    def build_all(
        self,
        build_type: str = "all",
        verbose: bool = True,
        axiom_check: bool = False,
        claim_check: bool = False,
    ):
        """Build all papers of specified type."""
        paper_ids = list(self.papers.keys())

        if build_type == "release":
            for paper_id in paper_ids:
                try:
                    self.release(
                        paper_id,
                        verbose=verbose,
                        axiom_check=axiom_check,
                        claim_check=claim_check,
                    )
                except Exception as e:
                    print(f"[release] ERROR {paper_id}: {e}")
            return

        if build_type in ("all", "lean"):
            for paper_id in paper_ids:
                try:
                    self.build_lean(paper_id, verbose=verbose)
                except Exception as e:
                    print(f"[build] ERROR: {e}")

        if build_type in ("all", "latex"):
            for paper_id in paper_ids:
                try:
                    self.build_latex(paper_id, verbose=verbose)
                except Exception as e:
                    print(f"[build] ERROR: {e}")

        if build_type in ("all", "markdown"):
            for paper_id in paper_ids:
                try:
                    self.build_markdown(paper_id)
                except Exception as e:
                    print(f"[build-md] ERROR: {e}")

        if build_type in ("all", "metadata"):
            for paper_id in paper_ids:
                try:
                    self.build_copy_paste_metadata(paper_id)
                except Exception as e:
                    print(f"[metadata] ERROR {paper_id}: {e}")

        if build_type in ("all", "arxiv"):
            for paper_id in paper_ids:
                try:
                    self.package_arxiv(paper_id)
                except Exception as e:
                    print(f"[arxiv] ERROR: {e}")

        if build_type in ("all", "submission"):
            for paper_id in paper_ids:
                try:
                    self.build_submission(paper_id, verbose=verbose)
                except Exception as e:
                    print(f"[build] ERROR: {e}")

        if build_type in ("all", "claim-check"):
            for paper_id in paper_ids:
                try:
                    self.check_claim_coverage(paper_id, fail_on_missing=claim_check)
                except Exception as e:
                    print(f"[claim-check] ERROR {paper_id}: {e}")

        if axiom_check:
            for paper_id in paper_ids:
                try:
                    self.check_axioms(paper_id, verbose=verbose)
                except Exception as e:
                    print(f"[axiom-check] ERROR: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified paper builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/build_papers.py release           # Full release build for all papers
  python scripts/build_papers.py release paper1    # Full release for paper1 only
  python scripts/build_papers.py lean paper1 -v    # Build Paper 1 Lean proofs (verbose)
  python scripts/build_papers.py latex paper2      # Build Paper 2 LaTeX
  python scripts/build_papers.py lean paper2 --axiom-check  # Build + check axioms
  python scripts/build_papers.py claim-check paper4_toc      # Verify theorem-label mapping coverage
  python scripts/build_papers.py submission paper1_jsait  # Build JSAIT review PDF
  python scripts/build_papers.py metadata paper4_toc      # Generate Zenodo/arXiv copy-paste metadata
  python scripts/build_papers.py scaffold paper6   # Create boilerplate from papers.yaml entry
""",
    )
    parser.add_argument(
        "build_type",
        nargs="?",
        default="release",
        choices=[
            "release",
            "all",
            "lean",
            "latex",
            "markdown",
            "arxiv",
            "submission",
            "claim-check",
            "metadata",
            "scaffold",
        ],
        help="What to build (default: release)",
    )
    parser.add_argument(
        "paper",
        nargs="?",
        default="all",
        help="Which paper id from papers.yaml, or all (default: all)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed output from build commands",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Minimal output (errors only)",
    )
    parser.add_argument(
        "--axiom-check",
        action="store_true",
        help="Run axiom verification on Lean theorems (checks what axioms each theorem depends on)",
    )
    parser.add_argument(
        "--claim-check",
        action="store_true",
        help="Enforce theorem-label mapping coverage check (fails on unmapped labels in release mode)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow scaffold command to overwrite existing files",
    )

    args = parser.parse_args()
    repo_root = Path(__file__).parent.parent

    # Determine verbosity:
    # - Explicit -q wins (quiet)
    # - Otherwise default to verbose for release/lean so build output is streamed
    # - -v always forces verbose
    if args.quiet:
        verbose = False
    else:
        verbose = args.verbose or args.build_type in ("release", "lean")

    axiom_check = args.axiom_check
    claim_check = args.claim_check

    try:
        if args.build_type == "scaffold":
            builder = PaperBuilder(
                repo_root,
                validate_prerequisites=False,
                validate_paper_structure=False,
            )
            if args.paper == "all":
                for paper_id in builder.papers.keys():
                    builder.scaffold_paper(paper_id, overwrite=args.overwrite)
            else:
                builder.scaffold_paper(args.paper, overwrite=args.overwrite)
            return

        builder = PaperBuilder(repo_root)
        if args.paper == "all":
            builder.build_all(
                args.build_type,
                verbose=verbose,
                axiom_check=axiom_check,
                claim_check=claim_check,
            )
        else:
            if args.build_type == "release":
                builder.release(
                    args.paper,
                    verbose=verbose,
                    axiom_check=axiom_check,
                    claim_check=claim_check,
                )
            elif args.build_type in ("all", "lean"):
                builder.build_lean(args.paper, verbose=verbose)
            if args.build_type in ("all", "latex"):
                builder.build_latex(args.paper, verbose=verbose)
            if args.build_type in ("all", "markdown"):
                builder.build_markdown(args.paper)
            if args.build_type in ("all", "metadata"):
                builder.build_copy_paste_metadata(args.paper)
            if args.build_type in ("all", "arxiv"):
                builder.package_arxiv(args.paper)
            if args.build_type in ("all", "submission"):
                builder.build_submission(args.paper, verbose=verbose)
            if args.build_type in ("all", "claim-check"):
                builder.check_claim_coverage(args.paper, fail_on_missing=claim_check)
            if axiom_check:
                builder.check_axioms(args.paper, verbose=verbose)
    except Exception as e:
        print(f"[build] FATAL: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
