# build_papers.py Refactoring Plan

## THE REAL PROBLEM

**7789 lines in ONE class. 179 methods.**

This is NOT a code organization problem. This is a MATHEMATICAL problem.

---

## MATHEMATICAL ANALYSIS

### The Structure

ALL builds follow the same MONOID structure:

```
validate inputs → gather inputs → process → write outputs
```

This is isomorphic across ALL 6 build types:
- build_lean: validate proofs → gather .lean → lake build → write .olean
- build_latex: validate latex → gather .tex → pdflatex → write .pdf
- build_markdown: validate latex → gather .tex → pandoc → write .md
- etc.

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

@dataclass(frozen=True)
class BuildSpec:
    """Mathematical specification of a build - data, not code."""
    name: BuildType
    input_paths: List[Callable]      # Functions that return input paths
    validator: Callable              # Input validation
    processor: Callable              # Main processing function
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
└── cli.py            # ~50 lines  - thin wrapper
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
