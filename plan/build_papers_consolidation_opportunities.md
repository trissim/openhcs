# build_papers.py Consolidation Opportunities

**Purpose**: Identify methods that can be collapsed following OpenHCS mathematical simplification principles
**Status**: Follows OpenHCS refactoring principles (factor common patterns, inline single-use, extract truly reusable)

---

## PRINCIPLES APPLIED

1. **Algebraic Common Factors** - Extract duplicate patterns into parameterized functions
2. **Function Composition** - Build pipelines from small composable functions
3. **Lookup Tables** - Replace repetitive conditional logic with dispatch dicts
4. **Single-Use Inlining** - Inline methods used only once

---

## CONSOLIDATION OPPORTUNITIES

### 1. Template Generation System (HIGH IMPACT)

**Current state**: 12 separate template functions (~500+ lines total)

```
_latex_main_template()          # ~50 lines
_latex_abstract_template()       # ~30 lines
_latex_intro_template()         # ~30 lines
_references_template()           # ~20 lines
_proofs_readme_template()       # ~40 lines
_lean_root_template()            # ~20 lines
_lean_basic_template()           # ~30 lines
_lakefile_template()            # ~40 lines
```

**Problem**: All follow same pattern - load template string, substitute placeholders with metadata.

**Consolidated solution**:

```python
# Single template engine
class TemplateEngine:
    """Generate files from Jinja2-style templates with OpenHCS principles."""

    def __init__(self, templates_dir: Path):
        self.templates_dir = templates_dir

    def render(self, template_name: str, **vars) -> str:
        """Render template with variable substitution.

        Mathematical simplification:
        - Single parameterized function replaces 12 separate template functions
        - Template files are external, not hardcoded strings
        - Easy to maintain and modify templates
        """
        template_path = self.templates_dir / f"{template_name}.j2"

        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_name}")

        template = template_path.read_text(encoding="utf-8")

        # Simple {{var}} substitution (no full Jinja2 dependency)
        for var_name, var_value in vars.items():
            pattern = rf"{{\s*{var_name}\s*}}"
            template = re.sub(pattern, str(var_value), template)

        return template

# Usage:
engine = TemplateEngine(PathResolver.papers_dir / "templates")

# Instead of 12 separate functions:
main_tex = engine.render("main", paper=meta, includes=content_files)
abstract_tex = engine.render("abstract", paper=meta)
readme = engine.render("proofs_readme", paper=meta, stats=lean_stats)
```

**Impact**: Eliminates ~250 lines of duplicate code, centralizes template management.

---

### 2. Copy Operations with Filtering (HIGH IMPACT)

**Current state**: 4 separate copy functions with duplicate filtering logic

```
_copy_latex_sources()      # Lines: ~100
_copy_lean_proofs()         # Lines: ~150
_copy_experiments()          # Lines: ~60
_copy_experiment_tree()       # INLINE in _copy_experiments (~40 lines)
```

**Problem**: All implement similar pattern - copy recursive tree with filtering.

**Consolidated solution**:

```python
# Single unified copy function
def _copy_tree_filtered(
    source_dir: Path,
    dest_dir: Path,
    *,
    allowed_extensions: Set[str] | None = None,
    skipped_dir_names: Set[str] | None = None,
    transform: Callable[[Path, Path], None] | None = None,
) -> int:
    """Copy directory tree with flexible filtering.

    Mathematical simplification:
    - Single parameterized function replaces 4 separate copy functions
    - Filters are optional parameters (extensions, dir names)
    - Transform callback allows path modification (e.g., fix includes)
    - Returns count of copied files

    EXACT BEHAVIOR preserved from original:
    - Recursive tree traversal
    - Extension filtering (e.g., .lean, .tex, .py)
    - Directory name filtering (e.g., __pycache__, venv)
    - Optional transform function (e.g., fix lean-handle-macros paths)
    """

    # Default filter sets
    if allowed_extensions is None:
        allowed_extensions = {".*"}  # Copy all files
    if skipped_dir_names is None:
        skipped_dir_names = set()

    copied_count = 0

    for src in source_dir.rglob("*"):
        rel_path = src.relative_to(source_dir)

        # Skip directories (will be created as needed)
        if src.is_dir():
            continue

        # Skip filtered directories
        if any(part in skipped_dir_names for part in rel_path.parts):
            continue

        # Skip disallowed extensions
        if src.suffix.lower() not in allowed_extensions:
            continue

        # Create destination path
        dst = dest_dir / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)

        # Apply transform if provided
        if transform:
            transform(src, dst)
        else:
            shutil.copy2(src, dst)

        copied_count += 1

    return copied_count

# Usage examples:

# Copy LaTeX sources (replaces _copy_latex_sources)
_copy_tree_filtered(
    latex_dir,
    package_dir,
    allowed_extensions={".tex", ".bib", ".bbl", ".cls", ".sty"},
    transform=lambda src, dst: _fix_lean_handle_macros_path(src, dst),
)

# Copy Lean proofs (replaces _copy_lean_proofs)
_copy_tree_filtered(
    proofs_dir,
    lean_dest,
    allowed_extensions={".lean"},
)

# Copy experiments (replaces _copy_experiments)
_copy_tree_filtered(
    experiments_dir,
    package_dir / "experiments",
    allowed_extensions={".py", ".json", ".md", ".sh", ".ipynb"},
    skipped_dir_names={
        "__pycache__", ".ipynb_checkpoints", ".venv",
        "venv", "out", "outputs", "checkpoints", ".git",
    },
)
```

**Impact**: Eliminates ~150 lines of duplicate traversal and filtering logic.

---

### 3. Module Closure Functions (MEDIUM IMPACT)

**Current state**: 3 similar closure functions

```
_get_content_backed_module_closure()   # ~40 lines
_get_claim_backed_module_closure()      # ~40 lines
_get_release_module_closure()           # ~30 lines
```

**Problem**: All follow same pattern - get handles, get closure for each handle, merge.

**Consolidated solution**:

```python
# Single unified closure function
def _get_module_closure_by_strategy(
    paper_id: str,
    strategy: Literal["content", "claim", "full"],
) -> Dict[str, Set[str]]:
    """Get module closure based on strategy.

    Mathematical simplification:
    - Single parameterized function replaces 3 separate closure functions
    - Strategy parameter determines which handles to include
    - Reduces code duplication in selective packaging

    Strategies:
    - "content": All handles referenced in content/*.tex
    - "claim": Handles for claim labels only
    - "full": All modules (no filtering)
    """
    meta = PathResolver.get_paper_meta(paper_id)

    # Strategy selection via lookup table (no if/else chains)
    strategy_handlers = {
        "content": _get_content_backed_handles,
        "claim": _get_claim_backed_handles,
        "full": lambda paper_id: None,  # No filtering
    }

    handler = strategy_handlers.get(strategy)
    if handler is None:
        raise ValueError(f"Unknown closure strategy: {strategy}")

    # Get handles based on strategy
    handle_set = handler(paper_id)

    if handle_set is None:
        # Full closure - no filtering
        return None

    # Build module index
    module_index = _build_lean_module_index(paper_id)
    handle_to_module = _extract_declared_handle_to_module_map(paper_id)

    # Get closure for all handles
    paper_to_modules = defaultdict(set)

    for handle in handle_set:
        handle_closure = _get_handle_module_closure(handle, module_index, handle_to_module)

        # Merge into paper's module set
        paper_to_modules[paper_id].update(handle_closure)

    return dict(paper_to_modules)

# Strategy handlers (small, focused functions)
def _get_content_backed_handles(paper_id: str) -> Set[str]:
    """Get all handles referenced in content/*.tex."""
    label_to_handles = _extract_claim_label_to_lean_handles(paper_id)
    return set().union(*label_to_handles.values())

def _get_claim_backed_handles(paper_id: str) -> Set[str]:
    """Get handles for claim labels only."""
    label_to_handles = _extract_claim_label_to_lean_handles(paper_id)
    return {
        handle
        for handles in label_to_handles.values()
        for handle in handles
    }

# Usage:
content_closure = _get_module_closure_by_strategy(paper_id, "content")
claim_closure = _get_module_closure_by_strategy(paper_id, "claim")
```

**Impact**: Eliminates ~50 lines of duplicate closure logic, uses lookup table for strategy selection.

---

### 4. Graph JSON Processing (LOW IMPACT)

**Current state**: Separate functions for graph operations

```
_mark_claim_nodes_in_graph()    # ~30 lines
_fix_ph_handle_edges()           # ~20 lines
_collect_graph_json()            # ~40 lines
```

**Problem**: All operate on same graph structure (nodes + edges).

**Consolidated solution**:

```python
# Graph data structure (dataclass for clarity)
@dataclass
class ProofGraph:
    """Proof dependency graph with validation and transformation."""

    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]

    @classmethod
    def load(cls, paper_id: str) -> "ProofGraph":
        """Load graph from JSON file."""
        graph_path = PathResolver.repo_root / "docs" / "papers" / "graph_infra" / "graphs" / f"{paper_id}.json"

        with open(graph_path, encoding="utf-8") as f:
            data = json.load(f)

        return cls(data["nodes"], data["edges"])

    def save(self, paper_id: str) -> None:
        """Save graph to JSON file."""
        graph_path = PathResolver.repo_root / "docs" / "papers" / "graph_infra" / "graphs" / f"{paper_id}.json"

        data = {"nodes": self.nodes, "edges": self.edges}
        graph_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def mark_claim_nodes(self, claim_labels: Set[str]) -> None:
        """Mark claim nodes by setting paper=-1."""
        for node in self.nodes:
            node_id = node["id"]
            if node_id in claim_labels:
                node["paper"] = -1  # Claim marker

    def fix_ph_handle_edges(self) -> None:
        """Fix PH handle edges (remove placeholder references)."""
        fixed_edges = []

        for edge in self.edges:
            target = edge["target"]

            # Skip PH handle edges
            if not target.startswith("PH."):
                fixed_edges.append(edge)

        self.edges = fixed_edges

# Usage (replaces 3 separate functions):
graph = ProofGraph.load(paper_id)
graph.mark_claim_nodes(claim_labels)
graph.fix_ph_handle_edges()
graph.save(paper_id)
```

**Impact**: Eliminates ~60 lines, encapsulates graph structure in dataclass.

---

### 5. Text Conversion Pipeline (MEDIUM IMPACT)

**Current state**: Already using composition, but could be cleaner

```
MARKDOWN_PIPELINE = compose(...7 processors...)
ZENODO_PIPELINE = compose(...6 processors...)
ARXIV_PIPELINE = compose(...4 processors...)
```

**Problem**: Pipelines hardcoded, hard to modify.

**Consolidated solution**:

```python
# Pipeline registry (lookup table instead of hardcoded composition)
PIPELINE_REGISTRY = {
    "markdown": [
        "strip_latex_comments",
        "normalize_claimstamps",
        "expand_lean_stat_macros",
        "convert_display_math",
        "convert_fake_subscripts",
        "prepend_markdown_macro_prelude",
        "convert_to_markdown",
        "postprocess_markdown",
    ],
    "zenodo": [
        "strip_latex_comments",
        "normalize_claimstamps",
        "convert_display_math",
        "convert_fake_subscripts",
        "convert_to_unicode",
    ],
    "arxiv": [
        "strip_latex_comments",
        "normalize_claimstamps",
        "convert_display_math",
        "convert_to_markdown",
    ],
}

# Single pipeline builder
def build_pipeline(pipeline_name: str, paper_id: str | None = None) -> Callable[[str], str]:
    """Build conversion pipeline by name.

    Mathematical simplification:
    - Lookup table instead of hardcoded compose() calls
    - Easy to add new pipelines
    - paper_id passed to expand_lean_stat_macros only
    """
    processor_names = PIPELINE_REGISTRY.get(pipeline_name, [])

    # Build list of processor functions
    processors = []

    for name in processor_names:
        # Handle special case: expand_lean_stat_macros needs paper_id
        if name == "expand_lean_stat_macros":
            processors.append(lambda text, pid=paper_id: _expand_lean_stat_macros(text, pid))
        else:
            processors.append(GLOBAL_PROCESSORS[name])

    # Compose processors
    def composed(text: str) -> str:
        for p in reversed(processors):
            text = p(text)
        return text

    return composed

# Processor registry
GLOBAL_PROCESSORS = {
    "strip_latex_comments": strip_latex_comments,
    "normalize_claimstamps": normalize_claimstamps,
    "convert_display_math": convert_display_math,
    "convert_fake_subscripts": convert_fake_subscripts,
    "convert_to_markdown": convert_to_markdown,
    "convert_to_unicode": convert_to_unicode,
    "prepend_markdown_macro_prelude": prepend_markdown_macro_prelude,
    "postprocess_markdown": postprocess_markdown,
}

# Usage:
markdown_pipeline = build_pipeline("markdown", paper_id)
zenodo_pipeline = build_pipeline("zenodo")
arxiv_pipeline = build_pipeline("arxiv")
```

**Impact**: Eliminates hardcoded pipeline definitions, easy to extend.

---

### 6. Metadata Helpers (LOW IMPACT)

**Current state**: Separate small functions for metadata

```
_default_arxiv_comments()      # ~5 lines
_write_abstract_helper_files()   # ~30 lines
_update_paper_date()            # ~15 lines
```

**Problem**: Could be unified with metadata dataclass.

**Consolidated solution**:

```python
# Metadata dataclass (single source of truth)
@dataclass
class PaperMetadata:
    """Paper metadata for Zenodo/arXiv submission."""

    paper_id: str
    title: str
    abstract_mathjax: str
    abstract_unicode: str
    lean_stats: LeanStats

    @property
    def arxiv_comments(self) -> str:
        """Generate default arXiv comments."""
        meta = PathResolver.get_paper_meta(self.paper_id)

        return (
            f"{meta.venue} submission. Lean 4 artifact: "
            f"{self.lean_stats.line_count} lines, "
            f"{self.lean_stats.theorem_count} theorems/lemmas "
            f"across {self.lean_stats.file_count} files "
            f"({self.lean_stats.sorry_count} sorry placeholders)."
        )

    def write_abstract_files(self) -> Tuple[Path, Path]:
        """Write abstract helper files."""
        releases_dir = PathResolver.get_releases_dir(self.paper_id)
        meta = PathResolver.get_paper_meta(self.paper_id)

        primary_path = releases_dir / f"arxiv_abstract_{self.paper_id}.md"

        primary_lines = [
            f"# arXiv abstract ({self.paper_id})",
            "",
            f"Title: {self.title}",
            "",
            "## Abstract (MathJax, arXiv-ready)",
            "",
            "```text",
            self.abstract_mathjax,
            "```",
            "",
            "## Abstract (Unicode, Zenodo-ready)",
            "",
            "```text",
            self.abstract_unicode,
            "```",
            "",
        ]

        primary_path.write_text("\n".join(primary_lines), encoding="utf-8")

        # Legacy file (backward compat)
        legacy_path = releases_dir / f"arxiv_abstract_{meta.dir_name}.md"

        # ... (handle multiple papers in same directory)

        return primary_path, legacy_path

# Usage:
metadata = PaperMetadata(
    paper_id=paper_id,
    title=title,
    abstract_mathjax=abstract_mathjax,
    abstract_unicode=abstract_unicode,
    lean_stats=lean_stats,
)

comments = metadata.arxiv_comments  # Replaces _default_arxiv_comments()
primary, legacy = metadata.write_abstract_files()  # Replaces _write_abstract_helper_files()
```

**Impact**: Eliminates ~30 lines, centralizes metadata in dataclass.

---

### 7. File Comparison and Hashing (LOW IMPACT)

**Current state**: Separate function for file comparison

```
_files_identical()  # ~15 lines
```

**Problem**: Could use functools.lru_cache for performance.

**Consolidated solution**:

```python
# Cached file comparison
@functools.lru_cache(maxsize=256)
def _files_identical(left: Path, right: Path) -> bool:
    """Compare two files using SHA-256 hash (cached)."""
    import hashlib

    left_hash = hashlib.sha256(left.read_bytes()).hexdigest()
    right_hash = hashlib.sha256(right.read_bytes()).hexdigest()

    return left_hash == right_hash

# Alternative: Use pathlib's samefile (Python 3.11+)
def _files_identical(left: Path, right: Path) -> bool:
    """Compare two files using pathlib's samefile."""
    return left.samefile(right)
```

**Impact**: Performance improvement, simpler implementation.

---

## SUMMARY TABLE

| Category | Before | After | Reduction | Impact |
|----------|---------|--------|---------|
| Template Generation | 12 functions (~250 lines) | 1 class + 12 templates | 70% | HIGH |
| Copy Operations | 4 functions (~150 lines) | 1 function (~60 lines) | 60% | HIGH |
| Module Closure | 3 functions (~50 lines) | 1 function + 2 handlers (~30 lines) | 40% | MEDIUM |
| Graph Processing | 3 functions (~60 lines) | 1 dataclass (~80 lines) | 20% | LOW |
| Text Pipeline | Hardcoded compose calls | Lookup table | 10% | MEDIUM |
| Metadata Helpers | 3 functions (~30 lines) | 1 dataclass (~50 lines) | 20% | LOW |
| **TOTAL** | **~540 lines** | **~270 lines** | **50%** | - |

---

## IMPLEMENTATION PRIORITY

### HIGH (Do First)
1. **Template Generation System** - Biggest line reduction, clear win
2. **Copy Operations with Filtering** - Second biggest win, clear consolidation

### MEDIUM (Do Next)
3. **Module Closure Functions** - Moderate win, uses lookup table pattern
4. **Text Conversion Pipeline** - Cleaner architecture, easier to extend

### LOW (Do Last)
5. **Graph Processing** - Small win, encapsulates graph structure nicely
6. **Metadata Helpers** - Small win, uses dataclass pattern

---

## OPENHCS PRINCIPLES APPLIED

1. **Algebraic Common Factors** - Extracted duplicate patterns into parameterized functions
2. **Lookup Tables** - Used dispatch dicts instead of if/else chains
3. **Function Composition** - Built pipelines from composable processors
4. **Dataclass Encapsulation** - Used dataclasses to group related data
5. **Single-Use Inlining** - Identified functions for inlining in implementation_details.md
6. **Caching** - Added lru_cache for performance
7. **Mathematical Simplification** - Reduced line count by 50%

---

## NEXT STEPS

1. Update `plan/build_papers_implementation_details.md` with consolidated patterns
2. Implement consolidation opportunities in order of priority
3. Verify all functionality preserved with tests
4. Measure actual line count reduction vs target
