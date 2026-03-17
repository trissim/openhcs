# build_papers.py Refactoring Plan

## THE REAL PROBLEM

**7768 lines in ONE class. 179 methods.**

This is NOT a code organization problem. This is a MATHEMATICAL problem.

---

## DESIGN PRINCIPLES (MUST PRESERVE)

From original docstring - these are INVARIANTS that must survive refactoring:

```
Declarative   — papers.yaml is the SSOT. Code derives, never guesses.
Fail-Loud    — validate structure upfront; abort with a clear message.
Orthogonal   — one method, one responsibility. No side-channel coupling.
Idempotent   — every step can be re-run safely; lake caching handles dedup.
```

---

## STRUCTURAL INVARIANTS (MUST PRESERVE)

All derived from papers.yaml:

```
1. Content:  paper_dir / meta.latex_dir / "content" / *.tex
2. Proofs:   paper_dir / meta.proofs_dir /             (lake project root)
3. Output:   paper_dir / "releases" /                  (pdf, md, tar.gz …)
4. Graphs:   graph_infra / "graphs" / <paper_id>.json  (every packaged paper)
```

---

## MATHEMATICAL ANALYSIS

### The Structure

ALL builds follow the same MONOID structure:

```
validate inputs → gather inputs → process → write outputs
```

---

## AUTO-GENERATED FILES (COMPLETE LIST)

The build pipeline generates these files - EVERY ONE must be preserved:

### Per-paper LaTeX content/ directory:
```
lean_stats.tex              - line/theorem/sorry counts (local + cumulative + release)
assumption_ledger_auto.tex  - assumption bundles (Axiom, Definition, Class)
lean_handle_ids_auto.tex    - LH{} ID table: \newcommand{\LHpaper4Ns1}{Ns.thm1}
claim_mapping_auto.tex      - claim→Lean matrix: theorem labels to handles
proof_hardness_index_auto.tex - hardness profiles per claim
lean-handle-macros.tex     - shared macro definitions (copied from papers/shared/)
```

### Per-paper releases/ directory:
```
<paper_id>.pdf                     - compiled PDF
<paper_id>_submission.pdf          - submission format PDF (if configured)
<paper_id>_copy_paste.txt          - human + machine metadata
<paper_id>_copy_paste.yaml         - machine YAML
arxiv_abstract_<paper_id>.md       - abstract (mathjax + unicode variants)
arxiv_abstract_<dir_name>.md      - legacy alias (backward compat)
arxiv_package_<archive_prefix>.tar.gz
arxiv_package_<archive_prefix>.zip
submission_package_<archive_prefix>.tar.gz
submission_package_<archive_prefix>.zip
```

### Per-paper root directory:
```
<paper_id>.md                      - full markdown conversion
```

### Graph infrastructure (docs/papers/graph_infra/):
```
graphs/<paper_id>.json             - proof dependency graph
graphs/<paper_id>.decls.json       - declaration signatures
graphs/dependency-graph.jsx        - interactive visualizer
graphs/rejection-trace.jsx        - rejection cascade visualizer
graphs/graph_utils.js              - shared utilities
graphs/true_path.js                - true path validation
graphs/latex_renderer.js           - LaTeX rendering
graphs/viewer.html                 - portable viewer entry
graphs/manifest.json               - graph list
```

### Per-paper latex/ directory:
```
theorem_verification.json          - LH annotation verification results
theorem_verification.tex           - LaTeX macro: \theoremverificationbadge
machine_checked_claim.tex          - claim for inclusion in paper
```

---

## BUILD TYPES (EXPLICIT)

All build types from original code (lines 7643-7658):

```python
class BuildType(Enum):
    LEAN = "lean"
    LATEX = "latex"
    MARKDOWN = "markdown"
    SUBMISSION = "submission"
    ARXIV = "arxiv"
    METADATA = "metadata"
    SCAFFOLD = "scaffold"
    PACKAGE_SUBMISSION_SOURCE = "package_submission_source"  # MISSING from original plan!
    RELEASE = "release"  # Orchestrator: lean → latex → submission → markdown → arxiv
    BUILD_ALL = "build_all"  # Orchestrator for batch builds
```

---

## CHECK TYPES (EXPLICIT)

All check types from original code:

```python
class CheckType(Enum):
    AXIOM = "axiom-check"      # Uses lake env lean + #print axioms
    CLAIM = "claim-check"      # Verifies theorem-label→Lean handle mapping
    LH = "lh-check"           # Enforces exactly one LH{} per theorem environment
```

---

## CLI FLAGS (EXPLICIT)

From original code (lines 7667-7698):

```python
# Positionals:
build_type: str = "release"   # Choices: release, all, lean, latex, markdown, arxiv, submission, claim-check, metadata, scaffold
paper: str = "all"            # Paper ID or "all"

# Flags:
-v, --verbose: bool = False   # Show detailed output (default: True for release/lean)
-q, --quiet: bool = False     # Minimal output (errors only)
--axiom-check: bool = False   # Run axiom verification
--claim-check: bool = False   # Enforce theorem-label mapping coverage
--lh-check: bool = False      # Enforce exactly one LH per theorem
--overwrite: bool = False     # Allow scaffold to overwrite existing files
```

---

## BUILD DEPENDENCIES (EXPLICIT)

From original code analysis:

```
LEAN           → []                              (no deps)
LATEX          → [LEAN]                         (needs compiled .olean)
MARKDOWN       → [LATEX]                        (needs .tex files)
SUBMISSION     → [LATEX]                        (needs .tex with submission_format)
ARXIV          → [LEAN, LATEX, MARKDOWN, METADATA]  (needs all artifacts)
METADATA       → [LATEX]                        (needs compiled .tex for abstract)
SCAFFOLD       → []                              (no deps, creates structure)
RELEASE        → [LEAN, LATEX, SUBMISSION, MARKDOWN, ARXIV]  # orchestrator
BUILD_ALL      → [varies by type]               # orchestrator
```

---

## BUILD STEPS (EXPLICIT - NO HANDWAVING)

### build_lean(paper_id, verbose=True) → str
From lines 4157-4238:
```
1. _refresh_derived_content(paper_id)    - run experiments, write auto-files
2. _get_paper_proofs_dir(paper_id)       - validate proofs dir exists
3. _collect_lean_dependency_closure(pid) - get transitive lean deps
4. _sync_local_lean_dependencies()       - copy dep .lake files
5. _sync_graph_infra_to_proofs()        - copy graph extraction deps
6. _write_export_filter_script()         - generate export.lean
7. _run_lake_build(paper_id, verbose)     - execute: lake build
8. _handle_lean_build_output()           - capture stdout for BUILD_LOG
9. _collect_lean_export_json(paper_id)    - parse lean#print graph_output
```

### build_latex(paper_id, verbose=False) → None
From lines 4240-4318:
```
1. _refresh_derived_content(paper_id)         - run experiments, write auto-files
2. _sync_shared_preamble_to_paper(paper_id)  - copy shared/*.sty
3. _write_lean_stats_auto(paper_id)          - generate lean_stats.tex
4. _write_lean_handle_ids_auto(paper_id)     - generate lean_handle_ids_auto.tex
5. _write_assumption_ledger_auto(paper_id)   - generate assumption_ledger_auto.tex
6. _normalize_and_fill_claimstamps(paper_id) - process \claimstamp{} in content
7. _write_claim_mapping_auto(paper_id)       - generate claim_mapping_auto.tex
8. _write_proof_hardness_index_auto(paper_id) - generate proof_hardness_index_auto.tex
9. _mark_claim_nodes_in_graph(paper_id)      - update graph JSON with claims
10. _rewrite_content_lean_handles_to_ids(paper_id) - replace handles with \LH{id}
11. _run_pdflatex_cycle(paper_id, verbose)   - pdflatex × N until stable
```

### build_markdown(paper_id) → None
From lines 4552-4627:
```
1. _refresh_derived_content(paper_id)    - ensure auto-files exist
2. _get_content_tex_files(paper_id)      - collect *.tex from content/
3. _extract_title_from_latex(paper_id)   - parse \title{}
4. _get_lean_stats(paper_id)             - get line/theorem counts
5. _build_markdown_file(meta, content_files, out_file, title, lean_stats)
   - Write header with title, venue, Lean stats
   - Convert abstract (from abstract.tex or \begin{abstract})
   - Convert each content/*.tex via pandoc
   - Write footer with proof location stats
```

### build_submission(paper_id, verbose=False) → None
From lines 4320-4440:
```
1. build_latex(paper_id, verbose)          - reuse latex build
2. If meta.submission_format:
   - Generate submission-specific PDF (e.g., with \JSAITREVIEW)
   - Copy to releases/<paper_id>_submission.pdf
3. package_submission_source(paper_id)    - create submission package
```

### package_submission_source(paper_id) → Optional[Path]
From lines 4442-4550:
```
1. Validate PDF exists
2. Create staging: releases/submission_package_<prefix>/
3. Copy: PDF, LaTeX sources, content/, Lean proofs (if configured)
4. Create submission_package_<prefix>.tar.gz and .zip
5. Return Path to tar.gz
```

### build_copy_paste_metadata(paper_id) → Path
From lines 5126-5207:
```
1. _release_title_from_latex(paper_id)      - parse \title{} from main.tex
2. _extract_abstract_latex(paper_id)        - extract from abstract.tex or main.tex
3. _latex_snippet_to_mathjax(abstract)      - convert for arXiv
4. _latex_snippet_to_unicode_zenodo(abstract) - convert for Zenodo
5. _default_arxiv_comments(paper_id)        - generate default if not set
6. Build payload dict with zenodo/arxiv/abstract_variants
7. Write releases/<paper_id>_copy_paste.txt (human + YAML)
8. Write releases/arxiv_abstract_<paper_id>.md (both variants)
9. Write releases/arxiv_abstract_<dir_name>.md (legacy alias)
```

### package_arxiv(paper_id) → Path
From lines 5726-5789:
```
1. _refresh_derived_content(paper_id)
2. _validate_and_get_pdf(paper_id)          - fail if no PDF
3. _validate_and_get_markdown(paper_id)     - fail if no .md
4. Create staging: releases/arxiv_package_<prefix>/
5. _copy_pdf(pdf_file, package_dir)
6. _copy_markdown(md_file, package_dir)
7. _copy_copy_paste_metadata(metadata_file, package_dir)
8. _copy_latex_sources(paper_id, package_dir)
   - Copy *.tex, *.bib, *.bbl, *.cls, *.sty
   - Copy content/ subdirectory
   - Fix lean-handle-macros paths (../../shared/ → ./)
   - Create main.tex wrapper if needed
   - Copy aux/bbl for citation resolution
9. _copy_lean_proofs(paper_id, package_dir)
   - Copy paper's proofs/
   - Copy transitive lean_dependencies (with dep_ prefix)
   - Rewrite lakefile srcdir paths
   - Write dependency bridge module
   - Copy config: lean-toolchain, lake-manifest.json, lakefile.lean
   - Generate proofs/README.md
10. _copy_experiments(paper_id, package_dir)
    - Copy configured experiment_paths (recursive, filtered by extension)
11. _copy_graph_visualizer(paper_id, package_dir)
    - Copy <paper_id>.json, <paper_id>.decls.json
    - Copy graph viewer JSX files
    - Generate graphs/manifest.json
    - Write graphs/index.html (D3 force-graph viewer)
12. If arxiv_config.include_build_log:
    - _create_build_log(paper_id, package_dir)
13. Create arxiv_package_<prefix>.tar.gz and .zip
14. Return Path to tar.gz
```

### scaffold_paper(paper_id, overwrite=False) → None
From lines 795-893:
```
1. Read meta from papers.yaml
2. Create paper_dir/
3. Create latex_dir/ with latex_file (main.tex skeleton)
4. Create content/ with placeholder sections
5. Create proofs/ with lakefile.lean, lean-toolchain
6. Create releases/ (empty)
7. Output created paths
```

### check_claim_coverage(paper_id, fail_on_missing=False, fail_on_lh_violation=False) → Dict
From lines 5562-5658:
```
1. _write_lean_handle_ids_auto(paper_id)
2. _rewrite_content_lean_handles_to_ids(paper_id)
3. _normalize_and_fill_claimstamps(paper_id)
4. _write_claim_mapping_auto(paper_id)
5. _write_proof_hardness_index_auto(paper_id)
6. _extract_paper_claim_labels(paper_id)   - regex for \label{(thm|cor|lem|prop):...}
7. _get_claim_mapping_file(paper_id)       - find claim_mapping_auto.tex or legacy
8. _extract_claim_label_to_lean_handles(paper_id)  - for auto files
9. _extract_mapped_claim_labels(mapping_file)  - for non-auto files
10. Compare: missing_labels, extra_labels
11. _check_forcing_coverage(paper_id)       - call analyze_forcing_coverage.py
12. _check_theorem_like_lean_annotations(paper_id)
    - Verify exactly one LH{} per theorem/coremma/lemma/proposition
    - Return violations, valid_theorems
13. _write_theorem_verification_badge(paper_id, valid_theorems, violations)
    - Write theorem_verification.json
    - Write theorem_verification.tex (\theoremverificationbadge)
    - Write machine_checked_claim.tex
14. Print summary and optionally raise on violations
15. Return dict with all results
```

### check_axioms(paper_id, verbose=True) → dict
From lines 7365-7544:
```
1. proofs_dir = _derive_path(paper_id)
2. Find all theorem/lemma declarations via regex
3. Group by module
4. For each module:
   a. Create temp file with #print axioms commands
   b. Run: lake env lean tmpfile
   c. Parse output for axiom dependencies
   d. Categorize: no_axioms, propext_only, quot_sound, classical, unknown
   e. Retry unknown with module prefix
5. Return summary dict with counts and details
```

---

## CHECK PROCESSORS (EXPLICIT - NO PLACEHOLDERS)

### run_claim_check(paper_id, fail_on_missing, fail_on_lh_violation) → Dict
Implementation from check_claim_coverage():
```
Input: paper_id, fail_on_missing (bool), fail_on_lh_violation (bool)
Output: {
    "paper_id": str,
    "mapping_file": str | None,
    "paper_labels": List[str],
    "mapped_labels": List[str],
    "missing_labels": List[str],
    "extra_labels": List[str],
    "forcing": Dict | None,
    "lh_check_violations": List[Dict],
    "lh_valid_theorems": List[Dict],
}
Steps:
1. Write all auto-generated files (lean_handle_ids, claim_mapping, etc.)
2. Extract paper labels via regex: \\label\{(thm|cor|lem|prop):[^}]+\}
3. Extract mapped labels from claim_mapping_auto.tex or legacy file
4. Compute missing = paper_labels - mapped_labels
5. Compute extra = mapped_labels - paper_labels
6. If fail_on_missing and missing: raise RuntimeError
7. Run forcing coverage analysis via analyze_forcing_coverage.py
8. Check LH annotations: verify exactly one \LH{} per theorem environment
9. If fail_on_lh_violation and violations: raise RuntimeError
10. Write theorem_verification.json/tex
```

### run_lh_check(paper_id) → Dict
Implementation from _check_theorem_like_lean_annotations():
```
Input: paper_id
Output: {
    "violations": List[Dict],
    "valid_theorems": List[Dict],
}
Steps:
1. Find all theorem/corollary/lemma/proposition environments via regex
2. For each environment:
   a. Extract trailing \leanmeta{} blocks
   b. Count \LH{} or \LHref{} references
   c. If count != 1: add to violations
   d. Else: add to valid_theorems
3. Return results
```

### run_axiom_check(paper_id, verbose) → Dict
Implementation from check_axioms():
```
Input: paper_id, verbose (bool)
Output: {
    "total": int,
    "checked": int,
    "no_axioms": int,
    "propext_only": int,
    "quot_sound": int,
    "classical": int,
    "unknown": int,
    "details": List[str],
}
Steps:
1. Find all theorem/lemma declarations in *.lean files via regex
2. Group by module (filename without .lean)
3. For each module:
   a. Create temp .lean file with #print axioms <thm> commands
   b. Run: lake env lean <tmpfile> (timeout 300s)
   c. Parse stdout/stderr for axiom info
   d. Categorize each theorem
   e. Retry failed with module prefix
4. Return summary
```

---

## PATH RESOLVER (EXPLICIT)

From original code analysis - all path methods needed:

```python
class PathResolver:
    """Single source for ALL path derivations - invariants enforced."""
    
    def __init__(self, repo_root: Path, papers_yaml: Path):
        self.repo_root = repo_root
        self.papers_dir = repo_root / "docs" / "papers"
        self._papers = self._load_papers(papers_yaml)
    
    # Core derivation - ONE invariant
    def derive_path(self, paper_id: str, *parts: str) -> Path:
        """Derive any paper-related path."""
        meta = self._get_paper_meta(paper_id)
        return self.papers_dir / meta.dir_name / Path(*parts)
    
    # Derived accessors (convenience, not new logic)
    def get_paper_dir(self, paper_id: str) -> Path:
        return self.derive_path(paper_id)
    
    def get_latex_dir(self, paper_id: str) -> Path:
        meta = self._get_paper_meta(paper_id)
        return self.derive_path(paper_id, meta.latex_dir)
    
    def get_content_dir(self, paper_id: str) -> Path:
        return self.get_latex_dir(paper_id) / "content"
    
    def get_proofs_dir(self, paper_id: str) -> Path:
        meta = self._get_paper_meta(paper_id)
        return self.derive_path(paper_id, meta.proofs_dir)
    
    def get_releases_dir(self, paper_id: str) -> Path:
        return self.derive_path(paper_id, "releases")
    
    def get_main_tex(self, paper_id: str) -> Path:
        meta = self._get_paper_meta(paper_id)
        return self.get_latex_dir(paper_id) / meta.latex_file
    
    def get_markdown_file(self, paper_id: str) -> Path:
        return self.derive_path(paper_id) / f"{paper_id}.md"
    
    def get_pdf_file(self, paper_id: str) -> Path:
        latex_dir = self.get_latex_dir(paper_id)
        pdfs = list(latex_dir.glob("*.pdf"))
        return pdfs[0] if pdfs else None
    
    def get_archive_prefix(self, paper_id: str) -> str:
        """Get variant-specific archive prefix."""
        meta = self._get_paper_meta(paper_id)
        return f"{meta.dir_name}_{paper_id}" if meta.variant else meta.dir_name
    
    def get_shared_preamble_dir(self) -> Path:
        return self.papers_dir / "shared"
```

---

## FUNCTION COMPOSITION (EXPLICIT)

From original code - all processors needed:

```python
# Type alias
Processor = Callable[[str], str]

# Function composition
def compose(*processors: Processor) -> Processor:
    """Compose: (f ∘ g)(x) = f(g(x))"""
    def composed(text: str) -> str:
        for p in reversed(processors):
            text = p(text)
        return text
    return composed

# Individual processors (each is str → str)
def strip_latex_comments(text: str) -> str: ...
def normalize_claimstamps(text: str) -> str: ...
def convert_display_math(text: str) -> str: ...
def convert_fake_subscripts(text: str) -> str: ...
def convert_to_markdown(text: str) -> str: ...
def convert_to_plain(text: str) -> str: ...
def convert_to_unicode(text: str) -> str: ...
def expand_lean_stat_macros(text: str, paper_id: str) -> str: ...
def prepend_markdown_macro_prelude(text: str) -> str: ...
def postprocess_markdown(text: str) -> str: ...

# Pipelines
MARKDOWN_PIPELINE = compose(
    strip_latex_comments,
    normalize_claimstamps,
    expand_lean_stat_macros,  # needs paper_id - special case
    convert_display_math,
    convert_fake_subscripts,
    prepend_markdown_macro_prelude,
    convert_to_markdown,
    postprocess_markdown,
)

ZENODO_PIPELINE = compose(
    strip_latex_comments,
    normalize_claimstamps,
    convert_display_math,
    convert_fake_subscripts,
    convert_to_unicode,
)

ARXIV_PIPELINE = compose(
    strip_latex_comments,
    normalize_claimstamps,
    convert_display_math,
    convert_to_markdown,  # For abstract
)
```

---

## GRAPH INFRASTRUCTURE (EXPLICIT)

From original code - what needs to be preserved:

### Graph JSON structure:
```python
{
    "nodes": [
        {"id": str, "kind": "theorem"|"definition"|"axiom", "paper": int}
    ],
    "edges": [
        {"source": str, "target": str}
    ]
}
```

### Operations needed:
```
_mark_claim_nodes_in_graph(paper_id)
  - Load graphs/<paper_id>.json
  - For each claim in paper: set node.paper = -1 (claim marker)
  - Write back

_collect_graph_json(paper_id)
  - Parse lake build output for graph data
  - Structure as nodes + edges

_mark_claim_nodes_in_graph() → None
  - Load graph JSON
  - Update nodes with claim markers
  - Write back
```

### Viewer features (from _write_graph_index_html):
- D3 force-directed graph
- Modes: forcing chain, claims only, full
- Filters: hide artifacts, paper scope, depth limit
- Node details: kind, paper, outgoing/incoming edges
- Lean signature display (from .decls.json)
- Claim-to-handles mapping

---

## EXPERIMENT COMMANDS (EXPLICIT)

From original code (_run_experiment_commands, lines 6219-6264):

```python
def run_experiment_commands(paper_id: str) -> None:
    """
    Run configured experiment_commands from papers.yaml.
    
    Templates support placeholders:
    - {paper_id}
    - {repo_root}
    - {paper_dir}
    - {latex_dir}
    - {content_dir}
    - {releases_dir}
    - {python}
    
    Each command:
    1. Format template with paths
    2. subprocess.run in repo_root
    3. Raise on non-zero returncode
    4. Print stdout
    """
```

---

## ORCHESTRATORS (EXPLICIT)

### release(paper_id, verbose, axiom_check, claim_check, lh_check) → None
From lines 7331-7363:
```
1. Print header
2. build_lean(paper_id, verbose)
3. build_latex(paper_id, verbose)
4. If meta.submission_format: build_submission(paper_id, verbose)
5. build_markdown(paper_id)
6. If claim_check: check_claim_coverage(paper_id, fail_on_missing=True, fail_on_lh_violation=True)
7. package_arxiv(paper_id)
8. If axiom_check: check_axioms(paper_id, verbose)
9. Print completion
```

### build_all(build_type, verbose, axiom_check, claim_check) → None
From lines 7546-7623:
```
For each paper_id in papers.keys():
    If build_type == "release": release(...)
    If build_type in ("all", "lean"): build_lean(...)
    If build_type in ("all", "latex"): build_latex(...)
    If build_type in ("all", "markdown"): build_markdown(...)
    If build_type in ("all", "metadata"): build_copy_paste_metadata(...)
    If build_type in ("all", "arxiv"): package_arxiv(...)
    If build_type in ("all", "submission"): build_submission(...)
    If build_type in ("all", "claim-check"): check_claim_coverage(...)
    If axiom_check: check_axioms(...)
```

---

## FILE STRUCTURE (EXPLICIT)

```
scripts/build/
├── __init__.py
├── specs.py              # BuildSpec, CheckSpec, BuildType, CheckType enums
├── builder.py            # Builder class - interprets specs
├── path_resolver.py     # PathResolver - all path derivations
├── processor.py         # Function composition + pipelines
├── stats.py             # LeanStats dataclass, stats collection
├── graph.py             # Graph operations (mark claims, parse JSON)
├── checks/
│   ├── __init__.py
│   ├── axiom.py         # run_axiom_check implementation
│   ├── claim.py         # run_claim_check + run_lh_check
│   └── forcing.py       # run_forcing_check (via analyze_forcing_coverage.py)
├── ops/
│   ├── __init__.py
│   ├── latex.py         # _run_pdflatex_cycle, write auto-files
│   ├── lean.py          # _run_lake_build, _collect_lean_export_json
│   ├── copy.py          # _copy_latex_sources, _copy_lean_proofs, etc.
│   └── experiment.py    # _run_experiment_commands
├── cli.py               # argparse + dispatch to builder
└── main.py              # main() entry point
```

---

## TARGET LINE COUNTS

| Module | Target Lines |
|--------|-------------|
| specs.py | ~80 |
| builder.py | ~60 |
| path_resolver.py | ~50 |
| processor.py | ~100 |
| stats.py | ~40 |
| graph.py | ~50 |
| checks/axiom.py | ~80 |
| checks/claim.py | ~100 |
| checks/forcing.py | ~30 |
| ops/latex.py | ~150 |
| ops/lean.py | ~80 |
| ops/copy.py | ~150 |
| ops/experiment.py | ~30 |
| cli.py | ~100 |
| main.py | ~30 |
| **TOTAL** | **~1030** |

Slightly higher than 600 target due to explicit check implementations, but still ~87% reduction from 7768.

---

## EXECUTION (4 PHASES)

### Phase 1: Define specs.py + path_resolver.py + stats.py
- Define BuildType, CheckType enums
- Define BuildSpec, CheckSpec dataclasses
- Implement PathResolver with ALL path methods
- Implement LeanStats dataclass

### Phase 2: Implement Builder + ops/latex.py + ops/lean.py
- Implement Builder.build() method
- Implement build_lean steps
- Implement build_latex steps (including all auto-file generation)
- Implement _run_pdflatex_cycle

### Phase 3: Implement processor.py + ops/copy.py + graph.py
- Implement function composition
- Implement all pipelines
- Implement copy operations (arxiv, submission, lean proofs)
- Implement graph marking operations

### Phase 4: Implement checks/ + cli.py + wire it all
- Implement run_claim_check
- Implement run_lh_check
- Implement run_axiom_check
- Implement cli.py with ALL flags
- Implement orchestrators (release, build_all)
- Test end-to-end

---

## VERIFICATION CHECKLIST

After implementation, verify ALL original functionality preserved:

- [ ] CLI: release, all, lean, latex, markdown, arxiv, submission, metadata, scaffold
- [ ] CLI: claim-check, axiom-check as standalone commands
- [ ] CLI: -v, -q, --axiom-check, --claim-check, --lh-check, --overwrite flags
- [ ] Auto-files: lean_stats.tex, assumption_ledger_auto.tex, lean_handle_ids_auto.tex, claim_mapping_auto.tex, proof_hardness_index_auto.tex
- [ ] Graph: claim marking, JSON structure, viewer
- [ ] Packaging: arxiv package with all components
- [ ] Check: claim coverage, LH annotation, axiom verification
- [ ] Metadata: copy-paste files for Zenodo/arXiv

---

## WHY THIS IS MATHEMATICAL

1. **Monoid**: `validate → gather → process → write` is a monoid homomorphism
2. **Function Composition**: `f ∘ g` for pipelines
3. **Data-Driven**: Specs are algebraic, not imperative
4. **Category Theory**: Builders are functors mapping Spec → Artifact
5. **No Branching**: No if/else on build types, just lookup

This is not "refactoring". This is simplifying the algebraic structure.
