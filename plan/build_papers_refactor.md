# build_papers.py Refactoring Plan

## THE REAL PROBLEM

**7789 lines in ONE class. 179 methods.**

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

This is isomorphic across ALL build types (must preserve all steps):

## AUTO-GENERATED FILES (must preserve)

The build pipeline generates these files (specs must capture all):

### LaTeX content/ directory:
- `lean_stats.tex` - line/theorem/sorry counts (local + cumulative + release)
- `assumption_ledger_auto.tex` - assumption bundles
- `lean_handle_ids_auto.tex` - LH{} ID table
- `claim_mapping_auto.tex` - claim→Lean matrix
- `proof_hardness_index_auto.tex` - hardness profiles

### Shared (synced from papers/shared/):
- `paper-preamble.sty` - shared preamble

### Build steps:
- build_lean: 
  1. check exists → 2. check lakefile → 3. collect deps → 4. sync local deps → 5. sync graph_infra → 6. write export → 7. run lake → 8. handle cache → 9. collect JSON
  
- build_latex: 
  1. sync shared preambles → 2. write lean_stats → 3. write handle_ids → 4. write assumption_ledger → 5. normalize claimstamps → 6. write claim_mapping → 7. write hardness_index → 8. mark claims in graph → 9. rewrite handles → 10. pdflatex cycle
  
- build_markdown: pandoc LaTeX → Markdown, expand \Lean* macros

- build_submission: similar to latex but with submission_format

- package_arxiv: copy PDF, markdown, metadata, latex sources, lean proofs, experiments, graph visualizer

- build_copy_paste_metadata: extract title, abstract (unicode, mathjax), generate YAML

### The Insight

Instead of 6 builder classes, we have ONE data-driven builder that interprets specs.

Instead of 10 converter methods, we have FUNCTION COMPOSITION.

Instead of 6 normalizer methods, we have a PIPELINE.

**This is not OOP. This is CATEGORY THEORY.**

---

## MATHEMATICAL SOLUTION

### 1. Build Specs as Data (not code)

```python
# scripts/build/specs.py

from dataclasses import dataclass
from enum import Enum
from typing import Callable, List

class BuildType(Enum):
    LEAN = "lean"
    LATEX = "latex"
    MARKDOWN = "markdown"
    SUBMISSION = "submission"
    ARXIV = "arxiv"
    METADATA = "metadata"
    SCAFFOLD = "scaffold"

    # Add CHECK specs for validation
class CheckType(Enum):
    AXIOM = "axiom-check"      # Verify what axioms each theorem depends on
    CLAIM = "claim-check"     # Verify theorem-label mapping coverage (fail_on_missing, fail_on_lh_violation)
    LH = "lh-check"           # Enforce exactly one Lean handle per theorem

@dataclass(frozen=True)
class CheckSpec:
    """Specification for validation checks."""
    name: CheckType
    processor: Callable
    fail_on_missing: bool = False     # For claim-check: fail if labels not mapped
    fail_on_lh_violation: bool = False  # For claim-check: fail if multiple Lean handles

@dataclass(frozen=True)
class BuildSpec:
    """Mathematical specification of a build - data, not code.
    
    Note: Some processors are complex (e.g., build_lean has 9 steps: 
    check_exists → check_lakefile → collect_deps → sync_deps → sync_graph → write_export → run_lake → handle_cache → collect_json) 
    → write_export → run_lake → handle_cache → collect_json).
    These remain as functions, but the SPEC is data.
    """
    name: BuildType
    input_paths: List[Callable]      # Functions that return input paths
    validator: Callable              # Input validation
    processor: Callable              # Main processing function (can be complex)
    output_paths: List[Callable]    # Functions that derive output paths
    dependencies: List[BuildType]   # Other builds required first

# ONE generic builder interprets specs
LEAN_SPEC = BuildSpec(
    name=BuildType.LEAN,
    input_paths=[lambda pr: pr.get_proofs_dir()],
    validator=lambda paths: all(p.exists() for p in paths),
    processor=run_lake_build,
    output_paths=[lambda pr: pr.get_proofs_dir() / "build" / "*.olean"],
    dependencies=[]
)

LATEX_SPEC = BuildSpec(
    name=BuildType.LATEX,
    input_paths=[lambda pr: pr.get_latex_dir(), lambda pr: pr.get_content_dir()],
    validator=lambda paths: all(p.exists() for p in paths),
    processor=run_pdflatex_cycle,
    output_paths=[lambda pr: pr.get_releases_dir() / "*.pdf"],
    dependencies=[BuildType.LEAN]  # Requires Lean compiled first
)

SCAFFOLD_SPEC = BuildSpec(
    name=BuildType.SCAFFOLD,
    input_paths=[],  # No inputs needed
    validator=lambda _: True,
    processor=run_scaffold,
    output_paths=[],  # Creates structure, no outputs
    dependencies=[]
)

# Check specs preserve ALL validation functionality
AXIOM_CHECK_SPEC = CheckSpec(
    name=CheckType.AXIOM,
    processor=run_axiom_check,
)

CLAIM_CHECK_SPEC = CheckSpec(
    name=CheckType.CLAIM,
    processor=run_claim_check,
    fail_on_missing=False,
    fail_on_lh_violation=False,
)

LH_CHECK_SPEC = CheckSpec(
    name=CheckType.LH,
    processor=run_lh_check,
)
```

### 2. ONE Generic Builder

```python
# scripts/build/builder.py

class Builder:
    """ONE builder that interprets specs - no if/else, just data."""
    
    def __init__(self, path_resolver: PathResolver, specs: dict[BuildType, BuildSpec]):
        self.pr = path_resolver
        self.specs = specs
    
    def build(self, build_type: BuildType, paper_id: str, **kwargs) -> Any:
        spec = self.specs[build_type]
        
        # 1. Validate dependencies first
        for dep in spec.dependencies:
            self.build(dep, paper_id, **kwargs)
        
        # 2. Gather inputs (from spec)
        inputs = [fn(self.pr, paper_id) for fn in spec.input_paths]
        
        # 3. Validate
        if not spec.validator(inputs):
            raise BuildError(f"Invalid inputs for {build_type}")
        
        # 4. Process
        return spec.processor(paper_id, inputs, **kwargs)
```

**This collapses 6 build methods → 1 method + 6 specs**

### 3. Function Composition for Processors

Instead of:
- `_convert_fake_subscripts()`
- `_convert_display_math()`
- `_latex_to_plain()`
- `_latex_to_mathjax()`

We have:

```python
# scripts/build/processors.py

# All processors are: str → str
Processor = Callable[[str], str]

# Function composition (category theory)
def compose(*processors: Processor) -> Processor:
    """Compose processors: (f ∘ g)(x) = f(g(x))"""
    def composed(text: str) -> str:
        for p in reversed(processors):
            text = p(text)
        return text
    return composed

# Build pipelines from specs
MARKDOWN_PIPELINE = compose(
    strip_latex_comments,
    normalize_claimstamps,
    convert_display_math,
    convert_fake_subscripts,
    convert_to_markdown,
)

PLAINTEXT_PIPELINE = compose(
    strip_latex_comments,
    normalize_claimstamps,
    convert_to_plain,
)

ZENODO_PIPELINE = compose(
    strip_latex_comments,
    normalize_claimstamps,
    convert_display_math,
    convert_subscripts,
    convert_to_unicode,
)
```

**This collapses 10 converter methods → 3 pipelines (composed of smaller functions)**

### 4. Data-Driven Normalizers

```python
# Normalizers are also: str → str
# Define as data, not methods

NORMALIZERS = {
    "leanmeta_handle_lists": normalize_leanmeta,
    "claimstamp_refs": normalize_claimstamp_refs,
    "claimstamp_regime": normalize_claimstamp_regime,
    "plaintext_block": normalize_plaintext,
}

def normalize(text: str, normalizer_name: str) -> str:
    return NORMALIZERS[normalizer_name](text)

def normalize_pipeline(text: str, *normalizer_names: str) -> str:
    for name in normalizer_names:
        text = normalize(text, name)
    return text
```

**This collapses 6 normalizer methods → 1 function + dict lookup**

### 5. ONE Unified Processor Class

```python
class Processor:
    """All text transformations - unified via function composition."""
    
    PIPELINES = {
        BuildType.MARKDOWN: MARKDOWN_PIPELINE,
        BuildType.PLAINTEXT: PLAINTEXT_PIPELINE,
        BuildType.ZENODO: ZENODO_PIPELINE,
    }
    
    @staticmethod
    def process(text: str, build_type: BuildType) -> str:
        return Processor.PIPELINES[build_type](text)
    
    @staticmethod
    def normalize(text: str, *normalizers: str) -> str:
        return normalize_pipeline(text, *normalizers)
```

---

## REVISED ARCHITECTURE

```
                    ┌─────────────────────────────────────┐
                    │          YAML Build Specs           │  ← DATA, not code
                    │   (LEAN_SPEC, LATEX_SPEC, etc.)    │
                    └──────────────┬──────────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────────────┐
                    │         ONE Generic Builder         │  ← INTERPRETS SPECS
                    │     builder.build(spec, paper_id)   │
                    └──────────────┬──────────────────────┘
                                   │
         ┌──────────────┬──────────┼──────────┬──────────────┐
         ▼              ▼          ▼          ▼              ▼
   ┌──────────┐  ┌──────────┐ ┌──────┐ ┌──────────┐  ┌──────────┐
   │  Path    │  │Processor │ │Stats │ │  Copy    │  │  File   │
   │ Resolver │  │(compose) │ │(spec)│ │ (spec)   │  │ Writer  │
   └──────────┘  └──────────┘ └──────┘ └──────────┘  └──────────┘
```

---

## FILE STRUCTURE (MATHEMATICAL)

```
scripts/build/
├── __init__.py
├── specs.py           # ~100 lines - ALL build specs as DATA
├── builder.py         # ~80 lines  - ONE generic builder
├── path_resolver.py   # ~50 lines  - path derivation
├── processor.py      # ~100 lines - function composition
├── normalizer.py     # ~60 lines  - dict-driven
├── stats.py          # ~80 lines  - data-driven
├── file_ops.py       # ~80 lines  - copy/write as specs
└── cli.py            # ~100 lines - ALL commands and flags

---

## COMPLETE CLI SPEC

```python
# scripts/build/cli.py
"""
Full CLI - preserves ALL original functionality plus adds new:
- release, all, lean, latex, markdown, arxiv, submission, metadata, scaffold
- claim-check, axiom-check, lh-check
- verbose, quiet, overwrite flags
"""
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Unified paper builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/build_papers.py release           # Full release build for all papers
  python scripts/build_papers.py release paper1    # Full release for paper1 only
  python scripts/build_papers.py lean paper1 -v    # Build Paper 1 Lean proofs (verbose)
  python scripts/build_papers.py latex paper2       # Build Paper 2 LaTeX
  python scripts/build_papers.py lean paper2 --axiom-check  # Build + check axioms
  python scripts/build_papers.py claim-check paper4_toc    # Verify theorem-label mapping
  python scripts/build_papers.py submission paper1_jsait     # Build JSAIT review PDF
  python scripts/build_papers.py metadata paper4_toc         # Generate copy-paste metadata
  python scripts/build_papers.py scaffold paper6             # Create boilerplate
        """
    )
    parser.add_argument(
        "build_type",
        nargs="?",
        default="release",
        choices=[
            "release",    # Full release pipeline
            "all",         # Build all of one type
            "lean",        # Lean proofs only
            "latex",       # LaTeX PDF only
            "markdown",    # Markdown only
            "arxiv",       # ArXiv package only
            "submission",  # Submission format only
            "metadata",    # Copy-paste metadata only
            "scaffold",    # Create boilerplate
        ],
        help="What to build (default: release)",
    )
    parser.add_argument(
        "paper",
        nargs="?",
        default="all",
        help="Which paper id from papers.yaml, or all (default: all)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed output")
    parser.add_argument("-q", "--quiet", action="store_true", help="Minimal output (errors only)")
    parser.add_argument("--axiom-check", action="store_true", 
        help="Run axiom verification on Lean theorems (checks what axioms each theorem depends on)")
    parser.add_argument("--claim-check", action="store_true", 
        help="Enforce theorem-label mapping coverage check (fails on unmapped labels)")
    parser.add_argument("--lh-check", action="store_true", 
        help="Enforce exactly one Lean handle per theorem-like environment (fails on violations)")
    parser.add_argument("--overwrite", action="store_true", 
        help="Allow scaffold command to overwrite existing files")
    
    args = parser.parse_args()
    
    # Dispatch to builder
    builder = Builder(path_resolver, specs)
    
    if args.build_type == "release":
        builder.release(args.paper, verbose=not args.quiet, 
                       axiom_check=args.axiom_check, 
                       claim_check=args.claim_check,
                       lh_check=args.lh_check)
    elif args.build_type == "all":
        builder.build_all(args.paper, verbose=not args.quiet)
    # ... other commands
```
```

**TOTAL: ~600 lines (92% reduction from 7789)**

---

## EXECUTION

### Phase 1: Define specs.py
Extract all 6 build types as data structures.

### Phase 2: Implement builder.py
ONE builder that interprets any spec.

### Phase 3: Implement processor.py
Function composition for all conversions.

### Phase 4: Wire CLI to builder
Thin wrapper that calls `builder.build(spec, paper_id)`

---

## WHY THIS IS MATHEMATICAL

1. **Monoid**: `validate → gather → process → write` is a monoid homomorphism
2. **Function Composition**: `f ∘ g` for pipelines
3. **Data-Driven**: Specs are algebraic, not imperative
4. **Category Theory**: Builders are functors mapping Spec → Artifact
5. **No Branching**: No if/else on build types, just lookup

This is not "refactoring". This is simplifying the algebraic structure.

---

## EXECUTION (4 PHASES)

### Phase 1: Define Build Specs (DATA)

Extract build definitions into YAML/specs - no code, just data:

```python
# specs.py
LEAN_SPEC = BuildSpec(...)
LATEX_SPEC = BuildSpec(...)
MARKDOWN_SPEC = BuildSpec(...)
```

### Phase 2: Implement Generic Builder (INTERPRETER)

One builder class that interprets any spec:

```python
# builder.py
class Builder:
    def build(self, spec, paper_id): ...
```

### Phase 3: Implement Processor (FUNCTION COMPOSITION)

Transformations as composing functions:

```python
# processor.py
MARKDOWN_PIPELINE = compose(strip, normalize, convert, ...)
```

### Phase 4: Wire CLI

Thin wrapper that calls `Builder().build(spec, paper_id)`

---

## SUMMARY

| Original | Mathematical |
|----------|-------------|
| 7789 lines | ~600 lines |
| 179 methods | ~20 functions |
| 6 builder classes | 1 generic builder |
| 10 converter methods | 3 pipelines (composed) |

**92% reduction through algebraic simplification**
