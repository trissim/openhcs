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

All build types from original code (lines 7647-7658):

```python
class BuildType(Enum):
    LEAN = "lean"
    LATEX = "latex"
    MARKDOWN = "markdown"
    SUBMISSION = "submission"
    ARXIV = "arxiv"
    METADATA = "metadata"
    SCAFFOLD = "scaffold"
    CLAIM_CHECK = "claim-check"  # Standalone CLI command
    RELEASE = "release"  # Orchestrator: lean → latex → submission → markdown → arxiv
    ALL = "all"  # Orchestrator for batch builds (NOT "build_all"!)
```

NOTE: `package_submission_source` is NOT a CLI build type - it's an internal method called by `build_submission`.

---

## MISSING SYSTEMS AUDIT - CRITICAL GAPS TO ADD

Based on audit of current code vs refactor plan:

### CRITICAL (COMPLETELY MISSING - Would Break Features)

1. **SUBMISSION Build Type** - Needs SubmissionBuildOperation
   - Missing: `build_submission()`, `package_submission_source()`
   - Missing: Submission wrapper generation, build hooks
   - Missing: Submission README generation
   - **Impact**: `build_papers.py submission paper1` would fail

2. **LH Check Operation** - Needs LHCheckOperation
   - Missing: `_check_theorem_like_lean_annotations()` - Validate exactly one LH per theorem
   - Missing: `_write_theorem_verification_badge()` - Write badge JSON + LaTeX
   - Missing: `machine_checked_claim.tex` generation
   - **Impact**: `--lh-check` flag would do nothing

3. **True Path Validation** - Needs validate_true_paths() function
   - Missing: BFS path validation
   - Missing: Tarjan's SCC cycle detection
   - Missing: Claim-to-axiom path verification
   - **Impact**: Graph validation wouldn't work

### HIGH (PARTIALLY MISSING - Would Cause Regressions)

4. **Graph Infrastructure** - Incomplete in plan
   - Missing: `_collect_graph_json()` - Run GraphExport.lean
   - Missing: `_fix_ph_handle_edges()` - Fix PH handle edges
   - Missing: `_sync_graph_viewer_assets()` - Copy JSX files
   - Missing: `_write_graph_manifest()` - Graph list
   - Missing: `_write_graph_index_html()` - D3 viewer (700 lines)
   - **Impact**: Graph packaging would be broken

5. **LaTeX Processing** - Incomplete in plan
   - Missing: `_strip_latex_comments()` - Comment stripping
   - Missing: `_extract_document_body()` - Extract body
   - Missing: `_evaluate_simple_ifdefined()` - Resolve \ifdefined\FLAG
   - Missing: `_resolve_include_path()` - Include resolution
   - Missing: `_find_tex_includes()` - Parse \input/\include
   - Missing: `_collect_included_tex_files()` - Recursive include collection
   - Missing: `_discover_content_files_for_flags()` - Flag-aware discovery
   - **Impact**: Markdown conversion would be broken

6. **Text Conversions** - Incomplete in plan
   - Missing: `_extract_braced_command_argument()` - Parse LaTeX args
   - Missing: `_latex_inline_to_plain()` - Inline LaTeX → plain
   - Missing: `_latex_snippet_to_mathjax()` - LaTeX → MathJax for arXiv
   - Missing: `_latex_snippet_to_unicode_zenodo()` - LaTeX → Unicode for Zenodo
   - Missing: `_convert_display_math_to_unicode()` - Display math → text
   - Missing: `_normalize_claimstamp_for_markdown()` - Handle claimstamps in MD
   - Missing: `_postprocess_markdown_text()` - Post-process MD
   - **Impact**: Metadata generation would be broken

7. **Module Index/Closure** - Incomplete in plan
   - Missing: `_extract_declared_handle_to_module_map()` - Handles → modules
   - Missing: `_build_lean_module_index()` - Module → files
   - Missing: `_get_handle_module_closure()` - Handle → transitive modules
   - Missing: `_get_content_backed_module_closure()` - Content-cited modules
   - Missing: `_get_claim_backed_module_closure()` - Claim-cited modules
   - Missing: `_get_release_module_closure()` - Select closure for release
   - **Impact**: Selective Lean packaging wouldn't work

8. **Lean Dependency Sync** - Incomplete in plan
   - Missing: `_sync_local_lean_dependency_dirs()` - Create dep_ symlinks
   - Missing: `_sync_graph_infra_lean()` - Copy DependencyGraph.lean
   - Missing: `_write_graph_export_lean()` - Generate export drivers
   - Missing: `_derive_module_roots_from_lakefile()` - Parse lakefile globs
   - Missing: `_rewrite_packaged_lakefile_srcdirs()` - Rewrite lakefile paths
   - Missing: `_write_dependency_bridge_module()` - Bridge module for deps
   - **Impact**: Lean builds with deps would fail

9. **Scaffold Templates** - Incomplete in plan
   - Missing: 8 template functions (latex_main, latex_abstract, latex_intro, references, etc.)
   - Missing: `_build_derived_latex_scaffold()` - Derive from existing paper
   - Missing: `_replace_first_latex_title()` - Title replacement
   - Missing: `_to_pascal_case()` / `_to_package_name()` - Name conversion
   - Missing: Template discovery (toolchain, mathlib)
   - **Impact**: `build_papers.py scaffold` would be broken

10. **Copy Operations** - Incomplete in plan
    - Missing: `_write_submission_wrapper_tex()` - Generate submission main.tex
    - Missing: `_write_submission_build_hook()` - Submission build script
    - Missing: `_write_submission_source_readme()` - Submission README
    - Missing: `_create_build_log()` - Generate BUILD_LOG.txt
    - Missing: `_copy_experiments()` - Copy experiment dirs
    - Missing: `_resolve_experiment_source()` - Resolve experiment paths
    - Missing: `_copy_experiment_tree()` - Recursive copy with filtering
    - **Impact**: ArXiv/submission packaging would be broken

### MEDIUM (Optimizations / Helpers)

11. **File Comparison** - Missing
    - Missing: `_files_identical()` - SHA-256 based comparison
    - **Impact**: Performance regression in packaging

12. **Claim Processing** - Incomplete
    - Missing: `_get_claim_mapping_file()` - Find mapping file
    - Missing: `_extract_paper_claim_labels()` - Extract theorem labels
    - Missing: `_extract_mapped_claim_labels()` - Extract from legacy
    - Missing: Claimstamp normalization, regime/tag extraction
    - **Impact**: Claim-based features would be broken

13. **Metadata Helpers** - Incomplete
    - Missing: `_default_arxiv_comments()` - Generate default comments
    - Missing: `_write_abstract_helper_files()` - Write abstract variants
    - Missing: `_update_paper_date()` - Update LaTeX date
    - **Impact**: Metadata generation would be incomplete

### LOW (Bug / Edge Case)

14. **Path Resolution Bug** - In Current Code
    - Missing: `_get_releases_dir()` is used but undefined
    - **Impact**: Current code is BROKEN, refactor MUST fix this

---

## NEW SECTIONS TO ADD TO PLAN

To complete the refactor plan, add these sections (in order):

### 1. Submission Build Type (NEW)
- SubmissionBuildOperation implements BuildOperation ABC
- Submission PDF generation (via latex_flag in papers.yaml)
- package_submission_source() implementation
- Submission wrapper, build hook, README generation

### 2. LH Check Operation (NEW)
- LHCheckOperation implements CheckOperation ABC
- _check_theorem_like_lean_annotations() - Validate exactly one LH per theorem
- _write_theorem_verification_badge() - Write badge JSON + LaTeX
- machine_checked_claim.tex generation

### 3. True Path Validation (NEW)
- validate_true_paths() pure function
- BFS path traversal
- Tarjan's SCC cycle detection
- Claim-to-axiom path verification
- Nontriviality marking

### 4. Complete Graph Operations (EXPAND)
- _collect_graph_json() - Run GraphExport.lean
- _fix_ph_handle_edges() - Fix PH handle edges
- _sync_graph_viewer_assets() - Copy JSX files
- _write_graph_manifest() - Graph list
- _write_graph_index_html() - D3 viewer generation

### 5. Complete LaTeX Processing (EXPAND)
- _strip_latex_comments() - Comment stripping
- _extract_document_body() - Extract body
- _evaluate_simple_ifdefined() - Resolve \ifdefined\FLAG
- _resolve_include_path() - Include resolution
- _find_tex_includes() - Parse \input/\include
- _collect_included_tex_files() - Recursive include collection
- _discover_content_files_for_flags() - Flag-aware discovery

### 6. Complete Text Conversions (EXPAND)
- _extract_braced_command_argument() - Parse LaTeX args
- _latex_inline_to_plain() - Inline LaTeX → plain
- _latex_snippet_to_mathjax() - LaTeX → MathJax
- _latex_snippet_to_unicode_zenodo() - LaTeX → Unicode
- _convert_display_math_to_unicode() - Display math → text
- _normalize_claimstamp_for_markdown() - Handle claimstamps
- _postprocess_markdown_text() - Post-process MD

### 7. Complete Module Index/Closure (EXPAND)
- _extract_declared_handle_to_module_map() - Handles → modules
- _build_lean_module_index() - Module → files
- _get_handle_module_closure() - Handle → transitive modules
- _get_content_backed_module_closure() - Content-cited modules
- _get_claim_backed_module_closure() - Claim-cited modules
- _get_release_module_closure() - Select closure

### 8. Complete Lean Dependency Sync (EXPAND)
- _sync_local_lean_dependency_dirs() - Create dep_ symlinks
- _sync_graph_infra_lean() - Copy DependencyGraph.lean
- _write_graph_export_lean() - Generate export drivers
- _derive_module_roots_from_lakefile() - Parse lakefile globs
- _rewrite_packaged_lakefile_srcdirs() - Rewrite paths
- _write_dependency_bridge_module() - Bridge module

### 9. Complete Scaffold Templates (EXPAND)
- 8 template functions (latex, lakefile, lean, readme, etc.)
- _build_derived_latex_scaffold() - Derive from existing
- _replace_first_latex_title() - Title replacement
- _to_pascal_case() / _to_package_name() - Name conversion
- Template discovery (toolchain, mathlib)

### 10. Complete Copy Operations (EXPAND)
- _write_submission_wrapper_tex() - Submission main.tex
- _write_submission_build_hook() - Submission build script
- _write_submission_source_readme() - Submission README
- _create_build_log() - BUILD_LOG.txt
- _copy_experiments() - Copy experiment dirs
- _resolve_experiment_source() - Resolve paths
- _copy_experiment_tree() - Recursive copy with filtering

### 11. Complete Claim Processing (EXPAND)
- _get_claim_mapping_file() - Find mapping file
- _extract_paper_claim_labels() - Extract theorem labels
- _extract_mapped_claim_labels() - Extract from legacy
- Claimstamp normalization
- Regime/tag extraction

### 12. Complete Metadata Helpers (EXPAND)
- _default_arxiv_comments() - Generate default
- _write_abstract_helper_files() - Write abstract variants
- _update_paper_date() - Update LaTeX date

### 13. File Comparison Utility (NEW)
- _files_identical() - SHA-256 based comparison

### 14. Path Resolution Fix (NEW)
- Implement _get_releases_dir() (used but undefined in current code)

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

## CORE INTERFACES (ABCs - NO IF/ELSE DISPATCH)

All build/check operations implement abstract interfaces. Routing is via dict lookup, not if/else chains.

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Optional


# ==================== SPECS ====================

@dataclass
class BuildSpec:
    build_type: str  # "lean", "latex", "markdown", "submission", etc.
    paper_id: str
    verbose: bool = False
    axiom_check: bool = False
    claim_check: bool = False
    lh_check: bool = False
    overwrite: bool = False


@dataclass
class CheckSpec:
    check_type: str  # "axiom", "claim", "forcing", "lh"
    paper_id: str
    verbose: bool = False
    fail_on_missing: bool = False
    fail_on_violation: bool = False


@dataclass
class BuildResult:
    paper_id: str
    build_type: str
    success: bool
    artifacts: List[Path]
    duration_seconds: float
    error: Optional[str] = None


@dataclass
class CheckResult:
    check_type: str
    paper_id: str
    success: bool
    violations: List[Dict[str, Any]]
    passed_count: int
    error: Optional[str] = None


# ==================== ABSTRACT INTERFACES ====================

class BuildOperation(ABC):
    """Abstract interface for all build operations (lean, latex, markdown, submission, arxiv, etc.)"""
    @abstractmethod
    def execute(self, spec: BuildSpec) -> BuildResult:
        """Execute of build operation"""
        pass

    @abstractmethod
    def name(self) -> str:
        """Operation name for logging"""
        pass


class CheckOperation(ABC):
    """Abstract interface for all check operations (axiom, claim, lh, forcing)"""
    @abstractmethod
    def execute(self, spec: CheckSpec) -> CheckResult:
        """Execute of check operation"""
        pass

    @abstractmethod
    def name(self) -> str:
        """Check name for logging"""
        pass


# ==================== DATA-DRIVEN ROUTING ====================

# Build operations registered at module import time
BUILD_OPERATIONS: Dict[str, BuildOperation] = {}
CHECK_OPERATIONS: Dict[str, CheckOperation] = {}


def register_build_operation(op: BuildOperation) -> None:
    """Register a build operation (called by ops/*.py at import)"""
    BUILD_OPERATIONS[op.name()] = op


def register_check_operation(op: CheckOperation) -> None:
    """Register a check operation (called by checks/*.py at import)"""
    CHECK_OPERATIONS[op.name()] = op


# Auto-registration in ops/ modules
# Example: register_build_operation(LatexBuildOperation())
# Example: register_build_operation(SubmissionBuildOperation())
# Auto-registration in checks/ modules
# Example: register_check_operation(ClaimCheckOperation())
# Example: register_check_operation(LHCheckOperation())
```

**Benefits:**
- NO if/else dispatch: `BUILD_OPERATIONS[spec.build_type].execute(spec)`
- Easy to add new builds: Implement BuildOperation, register it
- Type-safe: mypy enforces interface compliance
- Testable: Mock any BuildOperation/CheckOperation

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

## CODE EXAMPLES: Function Composition vs Private Methods

### Example 1: Auto-File Generation

**Current (monolith, private methods):**
```python
def _write_lean_handle_ids_auto(self, paper_id: str) -> None:
    """180 lines calling 10 private methods"""
    handles = self._extract_declared_lean_handle_locations(paper_id)
    aliases = self._extract_alias_code_map_from_handle_aliases(paper_id)
    id_map = self._assign_compact_handle_ids(handles, aliases)
    # ... 7 more private method calls
    out_file.write_text(render_table(id_map))
```

**Refactored (pure functions, composable):**
```python
# ops/latex.py

def extract_declared_handles(paper_id: str) -> List[Handle]:
    """Pure: paper_id → list of handles"""
    lean_files = glob_lean_files(paper_id)
    return parse_handle_locations(lean_files)


def extract_alias_map(paper_id: str) -> Dict[str, str]:
    """Pure: paper_id → alias mapping"""
    handle_aliases = find_handle_aliases(paper_id)
    return parse_alias_content(handle_aliases)


def assign_compact_ids(handles: List[Handle], aliases: Dict[str, str]) -> Dict[str, str]:
    """Pure: handles + aliases → compact ID mapping"""
    code_map = merge_handle_and_alias_handles(handles, aliases)
    return assign_collision_free_ids(code_map)


def build_handle_ids_latex(paper_id: str) -> str:
    """Compose pure functions into pipeline"""
    handles = extract_declared_handles(paper_id)
    aliases = extract_alias_map(paper_id)
    id_map = assign_compact_ids(handles, aliases)
    return render_handle_ids_table(id_map)
```

**Benefits:**
- Each function is unit-testable with simple inputs
- `parse_handle_locations` can be reused for other auto-files
- No hidden state or `self._` calls
- Pipeline is explicit: `extract → merge → assign → render`

### Example 2: Build Pipeline

**Current (imperative, stateful):**
```python
def build_latex(self, paper_id: str) -> None:
    self._refresh_derived_content(paper_id)
    self._sync_shared_preamble_to_paper(paper_id)
    self._write_lean_stats_auto(paper_id)
    # ... 8 more method calls mutating state
    self._run_pdflatex_cycle(paper_id, verbose)
```

**My bad refactor (still imperative, just extracted functions):**
```python
def build_latex(paper_id: str, verbose: bool = False) -> Path:
    latex_source = read_latex_source(paper_id)
    lean_files = glob_lean_files(paper_id)
    lean_stats = compute_lean_stats(lean_files)
    handles = extract_lean_handles(latex_source)
    claim_labels = extract_claim_labels(latex_source)
    handle_ids = build_handle_ids_latex(paper_id)
    claim_mapping = build_claim_mapping_latex(paper_id, claim_labels, handles)
    annotated = add_lean_handle_ids(latex_source, handle_ids)
    annotated = add_claim_mapping(annotated, claim_mapping)
    validated = validate_true_paths(paper_id, annotated)
    return compile_latex(validated, verbose)
```
**Still smells:** Just renamed `self._` to functions, but it's still a long imperative list.

**CORRECT refactor (ABC implementation + dict dispatch):**
```python
# ops/latex.py

from build.interfaces import BuildOperation, BuildSpec, BuildResult, register_build_operation

class LatexBuildOperation(BuildOperation):
    """Implement BuildOperation ABC for LaTeX compilation"""

    def name(self) -> str:
        return "latex"

    def execute(self, spec: BuildSpec) -> BuildResult:
        """Build PDF via declarative pipeline"""
        start = time.time()

        try:
            # Pipeline: read → extract → generate → annotate → validate → compile
            latex_source = read_latex_source(spec.paper_id)
            lean_files = glob_lean_files(spec.paper_id)
            lean_stats = compute_lean_stats(lean_files)
            handles = extract_lean_handles(latex_source)
            claims = extract_claim_labels(latex_source)

            # Generate auto-files (as pure function pipelines)
            handle_ids = build_handle_ids_latex(spec.paper_id)
            claim_mapping = build_claim_mapping_latex(spec.paper_id, claims, handles)

            # Annotate source
            annotated = add_lean_handle_ids(latex_source, handle_ids)
            annotated = add_claim_mapping(annotated, claim_mapping)

            # Validate
            validated = validate_true_paths(spec.paper_id, annotated)

            # Compile (side effect, but isolated)
            pdf_path = compile_latex(validated, spec.verbose)

            return BuildResult(
                paper_id=spec.paper_id,
                build_type="latex",
                success=True,
                artifacts=[pdf_path],
                duration_seconds=time.time() - start,
            )

        except Exception as e:
            return BuildResult(
                paper_id=spec.paper_id,
                build_type="latex",
                success=False,
                artifacts=[],
                duration_seconds=time.time() - start,
                error=str(e),
            )


# Auto-register at import
register_build_operation(LatexBuildOperation())
```

**In builder.py (dispatch via dict lookup, NO if/else):**
```python
# builder.py

from build.interfaces import BuildOperation, BuildSpec, BuildResult, BUILD_OPERATIONS


def build(spec: BuildSpec) -> BuildResult:
    """Dispatch to registered build operation via dict lookup"""
    op = BUILD_OPERATIONS.get(spec.build_type)
    if not op:
        raise ValueError(f"Unknown build type: {spec.build_type}")
    return op.execute(spec)
```

**Benefits:**
- NO if/else dispatch: `BUILD_OPERATIONS[spec.build_type].execute(spec)`
- Each operation implements ABC (enforced by mypy)
- Auto-registration at module import (no manual wiring)
- Easy to add new builds: Implement BuildOperation, register it
- Each execute() is self-contained and testable

### Example 4: Check Operations (ABC Implementation)

**Current (imperative check methods):**
```python
def check_claim_coverage(self, paper_id: str, fail_on_missing: bool = False):
    if fail_on_missing and missing_labels:
        raise RuntimeError(...)
    violations, valid = self._check_theorem_like_lean_annotations(paper_id)
    # ... 80 lines of imperative logic
```

**Refactored (ABC implementation + dict dispatch):**
```python
# checks/claim.py

from build.interfaces import CheckOperation, CheckSpec, CheckResult, register_check_operation


class ClaimCheckOperation(CheckOperation):
    """Implement CheckOperation ABC for claim coverage validation"""

    def name(self) -> str:
        return "claim"

    def execute(self, spec: CheckSpec) -> CheckResult:
        """Check claim-to-Lean-handle mapping coverage"""
        start = time.time()
        violations = []
        passed_count = 0

        try:
            # Extract claim labels
            claim_labels = extract_claim_labels(spec.paper_id)

            # Extract mapped handles
            mapping_file = get_claim_mapping_file(spec.paper_id)
            if mapping_file and mapping_file.name == "claim_mapping_auto.tex":
                label_to_handles = extract_claim_label_to_handles(spec.paper_id)
                mapped_labels = {label for label, handles in label_to_handles.items() if handles}
            elif mapping_file:
                mapped_labels = extract_mapped_claim_labels(mapping_file)
            else:
                mapped_labels = set()

            # Find violations
            missing_labels = sorted(claim_labels - mapped_labels)
            if missing_labels:
                violations.extend([
                    {"type": "missing_mapping", "label": label}
                    for label in missing_labels
                ])

            # Check LH annotations
            lh_violations, lh_valid = check_theorem_like_lean_annotations(spec.paper_id)
            violations.extend(lh_violations)
            passed_count = len(lh_valid)

            # Fail if requested and violations found
            if spec.fail_on_missing and missing_labels:
                raise RuntimeError(
                    f"Claim mapping incomplete for {spec.paper_id}: {len(missing_labels)} labels unmapped"
                )

            if spec.fail_on_violation and lh_violations:
                raise RuntimeError(
                    f"LH annotation violation in {spec.paper_id}: {len(lh_violations)} environments lack exactly one handle"
                )

            return CheckResult(
                check_type="claim",
                paper_id=spec.paper_id,
                success=True,
                violations=violations,
                passed_count=passed_count,
            )

        except Exception as e:
            return CheckResult(
                check_type="claim",
                paper_id=spec.paper_id,
                success=False,
                violations=[],
                passed_count=0,
                error=str(e),
            )


# Auto-register at import
register_check_operation(ClaimCheckOperation())
```

**In cli.py (check dispatch via dict lookup, NO if/else):**
```python
# cli.py

from build.interfaces import CheckSpec, CHECK_OPERATIONS


def run_check(check_type: str, paper_id: str, **kwargs) -> CheckResult:
    """Dispatch to registered check operation via dict lookup"""
    spec = CheckSpec(check_type=check_type, paper_id=paper_id, **kwargs)
    op = CHECK_OPERATIONS.get(check_type)
    if not op:
        raise ValueError(f"Unknown check type: {check_type}")
    return op.execute(spec)
```

**Benefits:**
- NO if/else dispatch: `CHECK_OPERATIONS[check_type].execute(spec)`
- Each check implements ABC (enforced by mypy)
- Auto-registration at module import
- Easy to add new checks: Implement CheckOperation, register it

---

## BUILD STEPS (EXPLICIT - NO HANDWAVING)

### build_lean(paper_id, verbose=True) → str
From lines 4157-4238:
```
1. _refresh_derived_content(paper_id)    - run experiments, write auto-files (orchestration)
2. _get_paper_proofs_dir(paper_id)       - validate proofs dir exists
3. _collect_lean_dependency_closure(pid) - get transitive lean deps
4. _sync_local_lean_dependencies()       - copy dep .lake files
5. _sync_graph_infra_lean(paper_id)       - copy DependencyGraph.lean
6. _write_graph_export_lean(paper_id)       - generate GraphExport.lean + DeclInfoExport.lean
7. _run_lake_build(paper_id, verbose)     - execute: lake build
8. _handle_lean_build_output()           - capture stdout for BUILD_LOG
9. _collect_graph_json(paper_id)        - run lake env lean GraphExport.lean
10. _collect_graph_json(paper_id)        - run lake env lean DeclInfoExport.lean
11. _fix_ph_handle_edges(paper_id)       - fix PH handle edges via fix_ph_handles.py
12. _mark_claim_nodes_in_graph(paper_id) - mark paper=-1 on cited claims
```

### build_latex(paper_id, verbose=False) → None
From lines 4240-4318:
```
1. _refresh_derived_content(paper_id)         - run experiments, write auto-files (orchestration)
2. _sync_shared_preamble_to_paper(paper_id)  - copy shared/*.sty
3. _write_latex_lean_stats(paper_id)          - generate lean_stats.tex (~100 lines)
4. _write_lean_handle_ids_auto(paper_id)     - generate lean_handle_ids_auto.tex (~180 lines)
5. _write_assumption_ledger_auto(paper_id)   - generate assumption_ledger_auto.tex (~140 lines)
6. _normalize_and_fill_claimstamps(paper_id) - process \claimstamp{} in content
7. _write_claim_mapping_auto(paper_id)       - generate claim_mapping_auto.tex (~70 lines)
8. _write_proof_hardness_index_auto(paper_id) - generate proof_hardness_index_auto.tex (~100 lines)
9. _mark_claim_nodes_in_graph(paper_id)      - update graph JSON with claims
10. _rewrite_content_lean_handles_to_ids(paper_id) - replace handles with \LH{id}
11. _run_pdflatex_cycle(paper_id, verbose)   - pdflatex × N until stable
12. _validate_true_paths(paper_id)             - BFS path validation + Tarjan SCC detection
13. _update_paper_date(latex_file)          - update LaTeX \date{} fields
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

### build_markdown(paper_id) → None
From lines 4552-4627:
```
1. _refresh_derived_content(paper_id)    - ensure auto-files exist
2. _discover_content_files_for_flags(paper_id) - follow main LaTeX include chain
3. _extract_title_from_latex(paper_id)   - parse \title{}
4. _get_lean_stats(paper_id)             - get line/theorem counts
5. _build_markdown_file(meta, content_files, out_file, title, lean_stats)
    - Write header with title, venue, Lean stats
    - Convert abstract (from abstract.tex or \begin{abstract})
    - Convert each content/*.tex via pandoc
    - Write footer with proof location stats
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

### release(paper_id, verbose, axiom_check, claim_check, lh_check) → BuildResult
From lines 7331-7363, refactored to use dict dispatch:
```python
def release(paper_id: str, verbose: bool = True, axiom_check: bool = False,
           claim_check: bool = False, lh_check: bool = False) -> BuildResult:
    """Full release build: lean → latex → markdown → arxiv"""
    print(f"\n{'=' * 60}")
    print(f"[release] Building {paper_id}: {load_paper_meta(paper_id).full_name}")
    print(f"{'=' * 60}\n")

    # Build in order via dict dispatch (NO if/else)
    build("lean", paper_id, verbose=verbose)
    build("latex", paper_id, verbose=verbose)

    # Submission if configured
    if load_paper_meta(paper_id).submission_format:
        build("submission", paper_id, verbose=verbose)

    build("markdown", paper_id)

    # Checks if requested
    if claim_check or lh_check:
        run_check("claim", paper_id, fail_on_missing=claim_check, fail_on_violation=lh_check)

    build("arxiv", paper_id)

    # Axiom check if requested
    if axiom_check:
        run_check("axiom", paper_id, verbose=verbose)

    print(f"\n[release] ✓ {paper_id} complete\n")
```

**Benefits:**
- NO if/else dispatch for builds: `build("latex", ...)`
- NO if/else dispatch for checks: `run_check("claim", ...)`
- Orchestrator focuses on sequencing, not routing logic

### build_all(build_type, verbose, axiom_check, claim_check) → None
From lines 7546-7623, refactored to use dict dispatch:
```python
def build_all(build_type: str, paper: str = "all", verbose: bool = True,
             axiom_check: bool = False, claim_check: bool = False) -> None:
    """Batch build all papers or single paper"""
    paper_ids = list_all_papers() if paper == "all" else [paper]

    for paper_id in paper_ids:
        try:
            # Dispatch via dict lookup (NO if/else chains)
            if build_type == "release":
                release(paper_id, verbose, axiom_check, claim_check)
            else:
                build(build_type, paper_id, verbose=verbose)

            # Run checks if requested
            if claim_check:
                run_check("claim", paper_id)
            if axiom_check:
                run_check("axiom", paper_id, verbose=verbose)

        except Exception as e:
            print(f"[{build_type}] ERROR {paper_id}: {e}")
```

---

## FILE STRUCTURE (EXPLICIT)

```
scripts/build/
├── __init__.py
├── specs.py              # BuildSpec, CheckSpec, BuildType, CheckType (dataclasses)
├── interfaces.py        # BuildOperation, CheckOperation ABCs + registration
├── builder.py            # Builder class - dispatches via BUILD_OPERATIONS dict
├── path_resolver.py     # All path derivations as pure functions (no class)
├── processor.py         # Function composition utilities + pipelines
├── stats.py             # LeanStats dataclass + pure stats functions (4 scopes)
├── graph.py             # Graph operations as pure functions
├── checks/
│   ├── __init__.py
│   ├── axiom.py         # AxiomCheckOperation implements CheckOperation ABC
│   ├── claim.py         # ClaimCheckOperation implements CheckOperation ABC
│   ├── forcing.py       # ForcingCheckOperation implements CheckOperation ABC
│   └── lh_check.py      # LHCheckOperation implements CheckOperation ABC (NEW)
├── ops/
│   ├── __init__.py
│   ├── latex.py         # LatexBuildOperation implements BuildOperation ABC
│   ├── lean.py         # LeanBuildOperation implements BuildOperation ABC
│   ├── submission.py     # SubmissionBuildOperation implements BuildOperation ABC (NEW)
│   ├── copy.py         # CopyBuildOperation implements BuildOperation ABC
│   └── experiment.py   # ExperimentBuildOperation implements BuildOperation ABC
└── cli.py              # argparse + dispatch (pure function)
```

**Key point**: Each ops/checks module implements ABC interfaces and auto-registers at import. Builder dispatches via dict lookup, not if/else chains.

**NEW**: Added `submission.py` and `lh_check.py` to cover missing critical features.

---

## TARGET LINE COUNTS (UPDATED FOR MISSING SYSTEMS)

| Module | Target Lines | Notes |
|--------|-------------|-------|
| specs.py | ~80 | Enums + dataclasses |
| interfaces.py | ~60 | BuildOperation, CheckOperation ABCs + registration |
| builder.py | ~50 | Builder class (dict dispatch, NO if/else) |
| path_resolver.py | ~50 | All path derivations (pure functions) + FIX _get_releases_dir bug |
| processor.py | ~370 | Pipeline composition + text conversion + LaTeX processing + module index |
| stats.py | ~60 | Lean stats (4 scopes) as pure functions |
| graph.py | ~280 | Graph operations + viewer + PH fixing + true path validation |
| checks/axiom.py | ~180 | AxiomCheckOperation implements CheckOperation ABC |
| checks/claim.py | ~270 | ClaimCheckOperation + claim processing helpers |
| checks/forcing.py | ~80 | ForcingCheckOperation implements CheckOperation ABC |
| checks/lh_check.py | ~120 | LHCheckOperation implements CheckOperation ABC (NEW) |
| ops/latex.py | ~450 | LatexBuildOperation + LaTeX processing + templates |
| ops/lean.py | ~250 | LeanBuildOperation + dependency sync + graph export |
| ops/submission.py | ~200 | SubmissionBuildOperation implements BuildOperation ABC (NEW) |
| ops/copy.py | ~350 | CopyBuildOperation + submission helpers + experiments + build log |
| ops/experiment.py | ~50 | ExperimentBuildOperation implements BuildOperation ABC |
| scaffold/ | ~250 | Template functions + derivation + name conversion (NEW MODULE) |
| cli.py | ~80 | argparse + dict dispatch (pure) |
| main.py | ~30 | Entry point (pure) |
| utils/ | ~110 | File comparison + metadata helpers (NEW MODULE) |
| **TOTAL** | **~3270** |

~58% reduction from 7768. Increased from ~1740 after audit revealed missing systems.

**NEW MODULES ADDED:**
- `checks/lh_check.py` (~120 lines) - LH validation + badge generation
- `ops/submission.py` (~200 lines) - Submission build + packaging
- `scaffold/` (~250 lines) - Template functions + derivation
- `utils/` (~110 lines) - File comparison + metadata helpers

**Architecture:**
- Each ops/checks module implements ABC interface
- Auto-registers at import: `register_build_operation(LatexBuildOperation())`
- Builder dispatches: `BUILD_OPERATIONS[spec.build_type].execute(spec)`
- NO if/else chains anywhere

**Line counts represent:**
- ABC implementations in ops/checks (NOT private methods)
- Pure functions with clear inputs/outputs
- Complete coverage of all 174 methods from original code
- All missing systems from audit now accounted for

---

## EXECUTION (6 PHASES - EXPANDED)

### Phase 0: Pre-Implementation Review
- **CRITICAL**: Review audit report above
- Verify plan covers ALL 174 methods from original code
- Confirm NO missing critical features before starting implementation

### Phase 1: Implement Core Infrastructure
- Implement specs.py (BuildSpec, CheckSpec, PaperMeta dataclasses)
- Implement interfaces.py (BuildOperation, CheckOperation ABCs + registration)
- Implement builder.py (dict dispatch, NO if/else)
- Implement path_resolver.py (ALL path derivations + FIX _get_releases_dir bug)
- Implement stats.py (LeanStats + 4-scope stats)
- Test: ABC implementations enforce interface
- Test: Registration populates BUILD_OPERATIONS/CHECK_OPERATIONS dicts
- Test: Path resolution for all variants

### Phase 2: Implement Build Operations (lean, latex, lean, submission)
- Implement LeanBuildOperation implements BuildOperation ABC:
  - execute() with dependency collection, lake build, graph export
  - name() returns "lean"
  - Auto-register: `register_build_operation(LeanBuildOperation())`
- Implement LatexBuildOperation implements BuildOperation ABC:
  - execute() with auto-file generation pipelines (7 files)
  - name() returns "latex"
  - Auto-register: `register_build_operation(LatexBuildOperation())`
- Implement SubmissionBuildOperation implements BuildOperation ABC (NEW):
  - execute() with submission PDF + packaging
  - name() returns "submission"
  - Auto-register: `register_build_operation(SubmissionBuildOperation())`
- Implement MarkdownBuildOperation, MetadataBuildOperation, ArxivBuildOperation
- Implement ExperimentBuildOperation
- Test: Each ABC implementation
- Test: Builder dispatches correctly for all build types
- Test: End-to-end builds for each type

### Phase 3: Implement Check Operations (axiom, claim, forcing, lh)
- Implement AxiomCheckOperation implements CheckOperation ABC:
  - execute() with module-grouped theorem discovery + tempfile strategy
  - name() returns "axiom"
  - Auto-register: `register_check_operation(AxiomCheckOperation())`
- Implement ClaimCheckOperation implements CheckOperation ABC:
  - execute() with claim coverage + claim processing helpers
  - name() returns "claim"
  - Auto-register: `register_check_operation(ClaimCheckOperation())`
- Implement ForcingCheckOperation implements CheckOperation ABC (same pattern)
- Implement LHCheckOperation implements CheckOperation ABC (NEW):
  - execute() with LH annotation validation + badge generation
  - name() returns "lh"
  - Auto-register: `register_check_operation(LHCheckOperation())`
- Test: Each check function
- Test: CLI dispatches to all check types
- Test: End-to-end checks

### Phase 4: Implement processor.py + graph.py + scaffold/ + utils/
- Implement function composition utilities (`compose`, `pipe`)
- Implement all text pipelines as **pure functions**:
  - Nested comment stripping
  - Pandoc conversion
  - LaTeX command argument extraction
  - MathJax conversion (arXiv)
  - Unicode conversion (Zenodo)
  - Display math → text
  - Claimstamp normalization
  - Post-process markdown
- Implement all LaTeX processing as **pure functions**:
  - Document body extraction
  - Comment stripping
  - Include resolution
  - Flag evaluation (\ifdefined)
- Implement all graph operations as **pure functions**:
  - Graph collection via lake env lean
  - PH handle edge fixing
  - Graph viewer asset copying
  - Graph manifest writing
  - D3 viewer HTML generation
  - True path validation (BFS + Tarjan SCC)
- Implement scaffold/ module:
  - 8 template functions (latex, lakefile, lean, readme, etc.)
  - Template derivation from existing paper
  - Title replacement
  - Name conversion (PascalCase, package_name)
  - Toolchain/mathlib discovery
- Implement utils/ module:
  - File comparison (SHA-256)
  - Metadata helpers (arXiv comments, abstract variants, date updates)
- Test: Each text pipeline
- Test: Each LaTeX processing function
- Test: Graph operations
- Test: Scaffold templates
- Test: File comparison

### Phase 5: Implement ops/copy.py (arxiv, submission, experiments)
- Implement CopyBuildOperation implements BuildOperation ABC:
  - execute() with arXiv packaging (all components)
  - name() returns "arxiv"
  - Auto-register: `register_build_operation(CopyBuildOperation())`
- Implement all copy operations as **pure functions**:
  - PDF/markdown/metadata copying
  - LaTeX source copying
  - Lean proof copying with module closures
  - Lean dependency syncing (dep_ symlinks, graph infra, lakefile rewriting)
  - Submission packaging (wrapper, build hook, README)
  - Experiment directory copying (path resolution, recursive copy with filtering)
  - Build log generation
- Test: Copy operations for all package types
- Test: Experiment handling

### Phase 6: Implement cli.py + main.py + wire it all
- Implement cli.py with ALL flags (pure function):
  - Positional: build_type, paper
  - Flags: -v, -q, --axiom-check, --claim-check, --lh-check, --overwrite
  - Dict dispatch to BUILD_OPERATIONS/CHECK_OPERATIONS
- Implement main.py entry point
- Test: CLI argument parsing for all build/check types
- Test: End-to-end: release, all build types, all check types
- Test: Full integration with original API compatibility

---

## REFACTOR STRATEGY: Function Composition, NOT Method Splitting

The refactor is NOT about breaking 7768 lines into 100 tiny private methods. That would just move complexity around.

Instead: **Transform imperative `self.do_thing()` chains into composable pure functions.**

### What This Means in Practice

**Before (monolith, stateful):**
```python
def build_latex(self, paper_id):
    self._refresh_derived_content(paper_id)  # 10+ internal state mutations
    self._sync_shared_preamble_to_paper(paper_id)
    self._write_lean_stats_auto(paper_id)  # Calls 5 other private methods
    self._write_lean_handle_ids_auto(paper_id)  # Calls 10 other private methods
    # ... 6 more method calls
    self._run_pdflatex_cycle(paper_id, verbose)
```

**After (pipeline, composable):**
```python
def build_latex(paper_id: str) -> Path:
    """Pipeline: content → handles → annotated → validated → PDF"""
    content = read_latex_source(paper_id)
    handles = extract_lean_handles(content)
    annotated = add_lean_handle_ids(content, handles)
    validated = validate_true_paths(paper_id, annotated)
    return compile_latex(validated)
```

Each step is a **pure function** (no state, predictable output) that can be:
- Tested in isolation
- Composed with other steps
- Reused across different build types

### Auto-File Generation = Function Pipelines

The plan lists 7 auto-files, but the refactor doesn't mean "7 functions with 10 private helpers each."

Instead: **Each auto-file is a composable pipeline of pure functions.**

**Example: lean_handle_ids_auto.tex generation**
```python
# NOT: _write_lean_handle_ids_auto() calling 10 private methods
# INSTEAD: Composable pipeline

def build_handle_ids_latex(paper_id: str) -> str:
    """Generate lean_handle_ids_auto.tex as composable pipeline"""
    lean_source = read_lean_files(paper_id)
    decl_map = extract_declarations(lean_source)  # Pure function
    alias_map = extract_aliases(lean_source)       # Pure function
    compact_ids = assign_compact_ids(decl_map, alias_map)  # Pure function
    return render_handle_ids_table(compact_ids)      # Pure function
```

Each sub-step is a **small pure function** that can be:
- Tested independently
- Reused for other auto-files
- Composed into different pipelines

### Why This Works

1. **No hidden state**: Functions take inputs and return outputs
2. **Testable**: Each function can be unit tested with simple inputs
3. **Composable**: `pipeline = compose(f, g, h, k)`
4. **Reusable**: `extract_declarations` used across multiple auto-files
5. **Parallelizable**: Pure functions can run in parallel where possible

### What We're NOT Doing

❌ Breaking `write_auto_file()` into `_prepare_data()`, `_format_output()`, `_write()` (state mutations)
✅ Making `prepare_data: str → Data`, `format_output: Data → str`, `write: str → Path` (pure functions)

❌ Adding private methods to hold intermediate state
✅ Passing intermediate results through function composition

❌ Object-oriented encapsulation with internal state
✌ Functional composition with explicit data flow

### Line Count Implications

The target ~1570 lines accounts for:
- Pure functions (NOT private methods) that can be tested in isolation
- Each function doing ONE thing (orthogonality)
- Pipeline orchestration in processor.py (compose functions, don't mutate state)
- Public APIs in ops/ modules (NOT private methods)

This is a **vertical cut** (separate concerns), not a **horizontal slice** (split monoliths).

---

## VERIFICATION CHECKLIST (EXPANDED)

After implementation, verify ALL original functionality preserved:

### CLI and Build Types
- [ ] CLI: release, all, lean, latex, markdown, arxiv, **submission**, metadata, scaffold
- [ ] CLI: claim-check, axiom-check, **lh-check** as standalone commands
- [ ] CLI: -v, -q, --axiom-check, --claim-check, --lh-check, --overwrite flags

### Auto-File Generation (7 files)
- [ ] lean_stats.tex
- [ ] lean_handle_ids_auto.tex
- [ ] assumption_ledger_auto.tex
- [ ] claim_mapping_auto.tex
- [ ] proof_hardness_index_auto.tex
- [ ] lean-handle-macros.tex (shared + per-paper)
- [ ] theorem_verification.tex/json/badge

### Graph Infrastructure
- [ ] Graph JSON collection (GraphExport.lean + DeclInfoExport.lean)
- [ ] PH handle edge fixing
- [ ] Claim node marking
- [ ] True path validation (BFS + Tarjan SCC)
- [ ] Graph viewer asset copying (JSX files)
- [ ] Graph manifest writing
- [ ] D3 viewer HTML generation
- [ ] Graph syncing to proofs directory

### Text Conversions
- [ ] LaTeX comment stripping
- [ ] Document body extraction
- [ ] Include resolution (\input/\include)
- [ ] Flag evaluation (\ifdefined)
- [ ] LaTeX command argument extraction
- [ ] Inline LaTeX → plain text
- [ ] LaTeX → MathJax (arXiv)
- [ ] LaTeX → Unicode (Zenodo)
- [ ] Display math → text
- [ ] Claimstamp normalization
- [ ] Markdown post-processing

### Module Index and Closure
- [ ] Handle → module mapping
- [ ] Module → files mapping
- [ ] Handle → transitive module closure
- [ ] Content-backed module closure
- [ ] Claim-backed module closure
- [ ] Release module closure (selective)

### Lean Dependency Sync
- [ ] Dep_ symlink creation
- [ ] Graph infra copying (DependencyGraph.lean)
- [ ] Graph export drivers (GraphExport.lean, DeclInfoExport.lean)
- [ ] Lakefile glob parsing
- [ ] Packaged lakefile path rewriting
- [ ] Dependency bridge module generation

### Packaging Operations
- [ ] ArXiv package with all components (PDF, markdown, metadata, LaTeX, proofs, experiments, graphs)
- [ ] Submission PDF generation (via latex_flag)
- [ ] Submission source packaging
- [ ] Submission wrapper generation
- [ ] Submission build hook generation
- [ ] Submission README generation
- [ ] Experiment directory copying (path resolution, recursive copy)
- [ ] Build log generation (BUILD_LOG.txt)

### Checks
- [ ] Axiom check (module-grouped theorem discovery, tempfile strategy)
- [ ] Claim coverage (label mapping, missing labels)
- [ ] LH annotation (exactly one handle per theorem)
- [ ] Theorem verification badge (JSON + LaTeX)
- [ ] Forcing coverage (via analyze_forcing_coverage.py)

### Scaffold Templates
- [ ] LaTeX main template
- [ ] LaTeX abstract template
- [ ] LaTeX intro template
- [ ] Bibliography template
- [ ] Proofs README template
- [ ] Lakefile template
- [ ] Lean root module template
- [ ] Lean basic template
- [ ] Template derivation from existing paper
- [ ] Title replacement in derived scaffolds
- [ ] Name conversion (PascalCase, package_name)
- [ ] Toolchain discovery
- [ ] Mathlib discovery

### Metadata Helpers
- [ ] Default arXiv comments generation
- [ ] Abstract helper files (MathJax + Unicode variants)
- [ ] LaTeX date updates

### Utilities
- [ ] File comparison (SHA-256 based)
- [ ] Path resolution fix (_get_releases_dir bug)

### Integration
- [ ] Builder dict dispatch for all build types
- [ ] CLI dict dispatch for all check types
- [ ] Auto-registration at import for all ops/checks modules
- [ ] NO if/else chains anywhere
- [ ] End-to-end: release works
- [ ] End-to-end: all build types work
- [ ] End-to-end: all check types work
- [ ] API compatibility: Same commands work as original

---

## WHY THIS IS MATHEMATICAL

1. **Monoid**: `validate → gather → process → write` is a monoid homomorphism
2. **Function Composition**: `f ∘ g` for pipelines
3. **Data-Driven**: Specs are algebraic, not imperative
4. **Category Theory**: Builders are functors mapping Spec → Artifact
5. **No Branching**: No if/else on build types, just lookup

This is not "refactoring". This is simplifying the algebraic structure.
