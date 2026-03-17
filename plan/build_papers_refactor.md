# build_papers.py Refactoring Plan

## THE REAL PROBLEM

**7789 lines in ONE class. 179 methods. 80 call sites for paths.**

The 80 path call sites are a SYMPTOM. The DISEASE is PaperBuilder is a monolith.

Quick fixes (updating call sites) don't fix the disease. We need surgical removal of the monolith.

---

## Architectural Fix: Break PaperBuilder into Focused Modules

### Each build operation becomes its own module

```
PaperBuilder (monolith, 7789 lines)
    │
    ├── LeanBuilder      (build_lean)      ~300 lines
    ├── LatexBuilder    (build_latex)     ~400 lines
    ├── MarkdownBuilder (build_markdown)   ~200 lines
    ├── SubmissionBuilder (build_submission) ~300 lines
    ├── ArxivPackager  (package_arxiv)    ~500 lines
    ├── MetadataGenerator (build_copy_paste_metadata) ~200 lines
    │
    └── Orchestrator   (release, build_all) ~300 lines
```

**Each module has its own path handling** - the 80 calls distribute across 7 modules = ~11 calls per module.

### The vertical stack:

```
                    ┌─────────────────────┐
                    │    Orchestrator     │  ← coordinates builds
                    └──────────┬──────────┘
                               │
        ┌──────────┬───────────┼───────────┬──────────┐
        ▼          ▼           ▼           ▼          ▼
   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
   │  Lean   │ │  LaTeX  │ │ Markdown │ │   ArXiv │ │Metadata │
   │ Builder │ │ Builder │ │ Builder  │ │ Packager│ │Generator│
   └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘
        │           │           │           │           │
        └───────────┴─────┬─────┴───────────┴───────────┘
                          ▼
              ┌───────────────────────┐
              │   PathResolver        │  ← SINGLE abstraction for paths
              │   (_derive_path)      │    used by all builders
              └───────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   PaperMeta          │  ← config from papers.yaml
              └───────────────────────┘
```

**Each builder imports PathResolver, uses it locally. No more 80 calls in one class.**

---

## Execution: Phase by Phase

### Phase 1: Extract PathResolver (smallest, safest)

Create `scripts/build/path_resolver.py`:

```python
"""Path resolution - single source of truth for all paper paths."""
from pathlib import Path
from typing import Optional

class PathResolver:
    """Single invariant for deriving paper-related paths."""
    
    def __init__(self, papers_dir: Path, papers: dict):
        self.papers_dir = papers_dir
        self.papers = papers
    
    def _get_paper_meta(self, paper_id: str):
        if paper_id not in self.papers:
            raise ValueError(f"Unknown paper: {paper_id}")
        return self.papers[paper_id]
    
    def derive(self, paper_id: str, *parts: str) -> Path:
        """Single invariant: derive any paper-related path."""
        meta = self._get_paper_meta(paper_id)
        return self.papers_dir / meta.dir_name / Path(*parts)
    
    def releases(self, paper_id: str) -> Path:
        """Releases directory, creates if needed."""
        releases = self.derive(paper_id, "releases")
        releases.mkdir(parents=True, exist_ok=True)
        return releases
```

**Extract: 1 class. Delete: 8 wrapper methods from PaperBuilder.**

### Phase 2: Extract config to module

Create `scripts/build/config.py`:

```python
"""Config classes - extracted from PaperBuilder."""
from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class ArxivConfig:
    include_build_log: bool = True
    include_lean_source: bool = True
    include_build_config: bool = True
    exclude_patterns: Tuple[str, ...] = (".lake", "build", "*.olean", "*.ilean")

@dataclass(frozen=True)
class BuildConfig:
    generated_content_files: Tuple[str, ...]
    shared_preamble_files: Tuple[str, ...]
    latex_source_extensions: Tuple[str, ...]
    lean_exclude_dirs: Tuple[str, ...]

# ... PaperMeta, LeanStats, LeanFileStats
```

### Phase 3: Extract each builder (larger, more complex)

For each build operation:
1. Identify the methods it uses
2. Create new module: `scripts/build/lean_builder.py`
3. Import PathResolver and config classes
4. Move methods, update imports
5. Test

**Extract in order of complexity:**
1. MarkdownBuilder (simplest, ~200 lines)
2. MetadataGenerator (~200 lines)
3. SubmissionBuilder (~300 lines)
4. LeanBuilder (~300 lines)
5. LatexBuilder (~400 lines)
6. ArxivPackager (~500 lines)

### Phase 4: Create Orchestrator

Create `scripts/build/orchestrator.py`:

```python
"""Orchestrates builds - the "release" and "build_all" logic."""
from pathlib import Path
from .config import BuildConfig, ArxivConfig
from .path_resolver import PathResolver

class Orchestrator:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        # Load config once
        self.config = self._load_config()
        self.path_resolver = PathResolver(
            repo_root / "docs" / "papers",
            self.config.papers
        )
    
    def release(self, paper_id: str, **kwargs):
        """Run full release pipeline."""
        # Coordinate all builders
        ...
    
    def build_all(self, build_type: str, **kwargs):
        """Build all papers of specified type."""
        ...
```

### Phase 5: Wire up CLI

Update `scripts/build_papers.py` to import from new modules:

```python
#!/usr/bin/env python3
"""CLI entry point - now thin wrapper around Orchestrator."""
import argparse
from pathlib import Path
from build.orchestrator import Orchestrator

def main():
    parser = argparse.ArgumentParser(...)
    args = parser.parse_args()
    
    orchestrator = Orchestrator(Path(__file__).parent.parent)
    
    if args.build_type == "release":
        orchestrator.release(args.paper, ...)
    elif args.build_type == "lean":
        # Import and run specific builder
        from build.lean_builder import LeanBuilder
        builder = LeanBuilder(orchestrator.path_resolver, ...)
        builder.build(args.paper)
    # ...

if __name__ == "__main__":
    main()
```

---

## What Gets Deleted

- PaperBuilder class (7789 lines → 0)
- 179 methods consolidated into 7 focused modules (~2200 lines total)
- 8 wrapper methods for paths (replaced by PathResolver)
- The monolith is GONE

## What Gets Added

- `scripts/build/` directory:
  - `__init__.py`
  - `config.py` (~150 lines)
  - `path_resolver.py` (~50 lines)
  - `lean_builder.py` (~300 lines)
  - `latex_builder.py` (~400 lines)
  - `markdown_builder.py` (~200 lines)
  - `submission_builder.py` (~300 lines)
  - `arxiv_packager.py` (~500 lines)
  - `metadata_generator.py` (~200 lines)
  - `orchestrator.py` (~300 lines)

**Net: ~2400 lines across 10 focused files vs 7789 in 1 file**

---

## Validation

After each phase:
1. `python -m py_compile scripts/build_papers.py`
2. `python scripts/build_papers.py --help`
3. Test the specific build type

---

## Why This Works

1. **Single Responsibility**: Each builder does ONE thing
2. **Vertical Abstraction**: PathResolver at bottom, orchestrator at top
3. **Information Reuse**: Each builder imports PathResolver, uses it locally
4. **No Indirection**: Direct calls within each module
5. **Testable**: Each builder can be tested in isolation

This is what the refactoring docs meant by "ABC Contract Enforcement" and "explicit dependency injection" - each module declares what it needs (PathResolver) in its constructor.

---

## COMPLETE METHOD INVENTORY (179 methods)

### Group 1: Config & Metadata (lines 208-380)
| Method | Line | Purpose |
|--------|------|---------|
| `ArxivConfig.from_dict` | 218 | Factory |
| `BuildConfig.from_dict` | 237 | Factory |
| `PaperMeta.from_dict` | 284 | Factory |
| `__init__` | 339 | Init PaperBuilder |
| `_load_raw_metadata` | 366 | Load YAML |
| `_load_papers` | 374 | Parse to PaperMeta |
| `_validate_prerequisites` | 381 | Check tools |
| `_validate_paper_structure` | 391 | Validate paths |

### Group 2: Path Methods (lines 418-456)
| Method | Line | Purpose |
|--------|------|---------|
| `_get_paper_meta` | 418 | Get metadata |
| `_get_paper_dir` | 425 | paper_dir |
| `_get_content_dir` | 430 | content_dir |
| `_get_latex_dir` | 435 | latex_dir |
| `_get_releases_dir` | 440 | releases_dir |
| `_get_markdown_dir` | 446 | markdown_dir |
| `_get_markdown_file` | 450 | markdown file |
| `_generated_content_files` | 454 | Config getter |
| `_iter_manual_content_tex` | 458 | List tex files |
| `_refresh_derived_content` | 479 | Trigger all writes |

### Group 3: Utilities (lines 496-580)
| Method | Line | Purpose |
|--------|------|---------|
| `_write_text_file` | 496 | Generic write |
| `_compact_lean_handle` | 512 | Handle formatting |
| `_compact_lean_prefix` | 522 | Prefix formatting |
| `_to_pascal_case` | 548 | Case conversion |
| `_to_package_name` | 558 | Name conversion |
| `_discover_template_lean_toolchain` | 567 | Find toolchain |
| `_discover_template_mathlib_require` | 584 | Find mathlib |

### Group 4: Templates (lines 609-815)
| Method | Line | Purpose |
|--------|------|---------|
| `_latex_main_template` | 609 | Main LaTeX |
| `_latex_abstract_template` | 639 | Abstract |
| `_latex_intro_template` | 646 | Intro |
| `_references_template` | 653 | References |
| `_proofs_readme_template` | 657 | Proofs README |
| `_replace_first_latex_title` | 667 | Title replace |
| `_build_derived_latex_scaffold` | 709 | Scaffold |
| `_lakefile_template` | 783 | Lakefile |
| `_lean_root_template` | 801 | Lean root |
| `_lean_basic_template` | 806 | Basic template |
| `scaffold_paper` | 816 | Create paper |

### Group 5: Content Discovery (lines 941-1090)
| Method | Line | Purpose |
|--------|------|---------|
| `_discover_content_files_for_flags` | 941 | By flags |
| `_discover_content_files` | 973 | All files |
| `_strip_latex_comments` | 977 | Remove comments |
| `_extract_document_body` | 981 | Get body |
| `_resolve_include_path` | 991 | Resolve paths |
| `_evaluate_simple_ifdefined` | 1019 | Eval conditionals |
| `_find_tex_includes` | 1065 | Find includes |
| `_collect_included_tex_files` | 1091 | Gather files |

### Group 6: Lean Build (lines 1133-1700)
| Method | Line | Purpose |
|--------|------|---------|
| `_get_paper_proofs_dir` | 1133 | proofs path |
| `_get_archive_prefix` | 1142 | Archive name |
| `_iter_paper_lean_files` | 1147 | List lean files |
| `_collect_lean_dependency_closure` | 1164 | Dep graph |
| `_sync_local_lean_dependency_dirs` | 1191 | Sync deps |
| `_sync_graph_infra_lean` | 1234 | Sync graph |
| `_write_graph_export_lean` | 1248 | Write export |
| `_collect_graph_json` | 1305 | Collect JSON |
| `_fix_ph_handle_edges` | 1407 | Fix edges |
| `_mark_claim_nodes_in_graph` | 1446 | Mark claims |

### Group 7: Lean Stats (lines 1718-2260)
| Method | Line | Purpose |
|--------|------|---------|
| `_iter_lean_roots_for_paper` | 1718 | List roots |
| `_derive_module_roots_from_lakefile` | 1726 | Parse lakefile |
| `_derive_module_roots` | 1760 | Get roots |
| `_files_identical` | 1787 | Compare files |
| `_strip_lean_comments` | 1799 | Remove comments |
| `_count_theorems_and_sorries` | 1854 | Count |
| `_compute_lean_file_stats` | 1864 | File stats |
| `_compute_lean_stats` | 1884 | Paper stats |
| `_get_lean_stats` | 1904 | Get stats |
| `_iter_transitive_lean_dependency_ids` | 1909 | Deps |
| `_get_cumulative_lean_stats` | 1927 | Cumulative |
| `_claim_backed_lean_stats_cache_key` | 1964 | Cache key |
| `_extract_declared_handle_to_module_map` | 1975 | Handle map |
| `_build_lean_module_index` | 2058 | Build index |
| `_get_handle_module_closure` | 2078 | Closure |
| `_get_claim_backed_module_closure` | 2128 | Claims |
| `_get_content_backed_module_closure` | 2158 | Content |
| `_get_content_backed_lean_stats` | 2163 | Stats |
| `_get_claim_backed_lean_stats` | 2201 | Stats |
| `_get_release_lean_stats` | 2239 | Release |
| `_get_release_module_closure` | 2250 | Closure |
| `_get_lean_file_stats` | 2261 | File stats |

### Group 8: LaTeX Write (lines 2266-2400)
| Method | Line | Purpose |
|--------|------|---------|
| `_sync_shared_preambles` | 2266 | Sync preambles |
| `_write_lean_handle_macros_auto` | 2292 | Write macros |
| `_write_shared_lean_handle_macros_auto` | 2318 | Shared macros |
| `_lean_macro_suffix` | 2343 | Suffix |
| `_build_lean_macro_suffix_map` | 2367 | Map |
| `_write_latex_lean_stats` | 2387 | Stats file |

### Group 9: Claim/Ledger (lines 2488-3180)
| Method | Line | Purpose |
|--------|------|---------|
| `_iter_claim_closure_sources` | 2488 | Sources |
| `_extract_primary_namespace` | 2513 | Namespace |
| `_parse_assumption_bundles` | 2518 | Parse bundles |
| `_parse_conditional_handles` | 2554 | Conditionals |
| `_parse_theorem_lemma_handles` | 2562 | Theorems |
| `_parse_alias_abbrev_map` | 2570 | Aliases |
| `_write_assumption_ledger_auto` | 2599 | Ledger |
| `_read_existing_lean_handle_rows` | 2746 | Read rows |
| `_extract_declared_lean_handle_locations` | 2790 | Locations |
| `_extract_declared_lean_handles` | 2873 | Handles |
| `_extract_referenced_lean_handles_from_content` | 2888 | Referenced |
| `_extract_lh_ids_from_content` | 2915 | IDs |
| `_extract_referenced_lh_handles_from_content` | 2933 | LH handles |
| `_extract_alias_code_map_from_handle_aliases` | 2961 | Alias map |
| `_assign_compact_handle_ids` | 3122 | Assign IDs |
| `_write_lean_handle_ids_auto` | 3173 | Write IDs |

### Group 10: Claim Processing (lines 3351-3760)
| Method | Line | Purpose |
|--------|------|---------|
| `_rewrite_content_lean_handles_to_ids` | 3351 | Rewrite |
| `_normalize_leanmeta_handle_lists` | 3386 | Normalize |
| `_claim_ref_code_for_label` | 3492 | Ref code |
| `_normalize_claimstamp_refs` | 3511 | Refs |
| `_normalize_claimstamp_regime` | 3531 | Regime |
| `_extract_label_to_regime_from_claimstamps` | 3591 | Regime map |
| `_extract_label_to_hardness_tags_from_claimstamps` | 3619 | Hardness |
| `_derive_hardness_profile` | 3667 | Profile |
| `_write_proof_hardness_index_auto` | 3691 | Index |
| `_normalize_and_fill_claimstamps` | 3768 | Fill |
| `_extract_claim_label_to_lean_handles` | 3842 | Claims |

### Group 11: Build Methods (lines 3985-4178)
| Method | Line | Purpose |
|--------|------|---------|
| `_read_lean_handle_id_map` | 3985 | Read map |
| `_read_claim_mapping_table_handles` | 4026 | Read table |
| `_write_claim_mapping_auto` | 4057 | Write mapping |
| `_expand_lean_stat_macros` | 4128 | Expand macros |
| `build_lean` | 4178 | MAIN |

### Group 12: LaTeX Build (lines 4261-4340)
| Method | Line | Purpose |
|--------|------|---------|
| `build_latex` | 4261 | MAIN |
| `build_submission` | 4341 | MAIN |

### Group 13: Submission (lines 4463-4560)
| Method | Line | Purpose |
|--------|------|---------|
| `package_submission_source` | 4463 | Package |
| `_write_submission_wrapper_tex` | 4512 | Wrapper |
| `_write_submission_build_hook` | 4529 | Hook |
| `_write_submission_source_readme` | 4554 | README |

### Group 14: Markdown (lines 4573-5090)
| Method | Line | Purpose |
|--------|------|---------|
| `build_markdown` | 4573 | MAIN |
| `_extract_braced_command_argument` | 4601 | Extract |
| `_latex_inline_to_plain` | 4652 | Convert |
| `_extract_main_latex_title` | 4681 | Title |
| `_release_title_from_latex` | 4695 | Title |
| `_extract_abstract_latex` | 4707 | Abstract |
| `_normalize_plaintext_block` | 4723 | Normalize |
| `_clean_metadata_reference_tokens` | 4741 | Clean |
| `_latex_snippet_to_plain` | 4764 | Convert |
| `_latex_snippet_to_mathjax` | 4785 | MathJax |
| `_mathjax_markdown_to_unicode_markdown` | 4832 | Unicode |
| `_convert_fake_subscripts_to_unicode` | 4877 | Subscripts |
| `_convert_display_math_to_unicode` | 4963 | Display math |
| `_latex_snippet_to_unicode_zenodo` | 5046 | Zenodo |
| `_default_arxiv_comments` | 5075 | Comments |
| `_write_abstract_helper_files` | 5085 | Helper files |

### Group 15: Metadata (lines 5147-5240)
| Method | Line | Purpose |
|--------|------|---------|
| `build_copy_paste_metadata` | 5147 | MAIN |
| `_build_markdown_file` | 5230 | Build |
| `_convert_latex_to_markdown` | 5288 | Convert |
| `_normalize_claimstamp_for_markdown` | 5332 | Normalize |
| `_prepend_markdown_macro_prelude` | 5360 | Prelude |
| `_postprocess_markdown_text` | 5385 | Postprocess |
| `_get_claim_mapping_file` | 5398 | Get file |

### Group 16: Claim Checking (lines 5418-5580)
| Method | Line | Purpose |
|--------|------|---------|
| `_extract_paper_claim_labels` | 5418 | Labels |
| `_extract_immediate_trailing_leanmeta_block` | 5435 | Block |
| `_check_theorem_like_lean_annotations` | 5450 | Check |
| `_write_theorem_verification_badge` | 5510 | Badge |
| `_extract_mapped_claim_labels` | 5571 | Mapped |
| `check_claim_coverage` | 5583 | MAIN |
| `_check_forcing_coverage` | 5681 | Forcing |
| `_update_paper_date` | 5722 | Date |

### Group 17: ArXiv Packaging (lines 5747-6400)
| Method | Line | Purpose |
|--------|------|---------|
| `package_arxiv` | 5747 | MAIN |
| `_validate_and_get_pdf` | 5796 | Validate PDF |
| `_validate_and_get_markdown` | 5810 | Validate MD |
| `_copy_pdf` | 5826 | Copy PDF |
| `_copy_markdown` | 5832 | Copy MD |
| `_copy_copy_paste_metadata` | 5838 | Copy metadata |
| `_copy_latex_sources` | 5844 | Copy sources |
| `_copy_lean_proofs` | 5959 | Copy proofs |
| `_rewrite_packaged_lakefile_srcdirs` | 6079 | Rewrite lakefile |
| `_write_dependency_bridge_module` | 6088 | Bridge module |
| `_copy_experiments` | 6134 | Copy experiments |
| `_resolve_experiment_source` | 6169 | Resolve |
| `_copy_experiment_tree` | 6185 | Tree copy |
| `_run_experiment_commands` | 6240 | Run |
| `_sync_graph_viewer_assets` | 6287 | Sync assets |
| `_write_graph_manifest` | 6306 | Manifest |
| `_copy_graph_visualizer` | 6329 | Copy viz |
| `_write_graph_index_html` | 6372 | HTML |

### Group 18: Generation (lines 7104-7230)
| Method | Line | Purpose |
|--------|------|---------|
| `_generate_paper_lakefile` | 7104 | Lakefile |
| `_generate_proofs_readme` | 7152 | README |
| `_create_build_log` | 7225 | Log |

### Group 19: Archiving (lines 7310-7400)
| Method | Line | Purpose |
|--------|------|---------|
| `_create_named_archive` | 7310 | Named |
| `_create_archive` | 7338 | Create |
| `release` | 7352 | MAIN |
| `check_axioms` | 7386 | MAIN |
| `build_all` | 7567 | MAIN |

---

## Additional Structural Duplication Found

### 1. `_write_*` methods (17 methods)

| Method | Line |
|--------|------|
| `_write_text_file` | 496 |
| `_write_graph_export_lean` | 1248 |
| `_write_lean_handle_macros_auto` | 2292 |
| `_write_shared_lean_handle_macros_auto` | 2318 |
| `_write_latex_lean_stats` | 2387 |
| `_write_assumption_ledger_auto` | 2599 |
| `_write_lean_handle_ids_auto` | 3173 |
| `_write_proof_hardness_index_auto` | 3691 |
| `_write_claim_mapping_auto` | 4057 |
| `_write_submission_wrapper_tex` | 4512 |
| `_write_submission_build_hook` | 4529 |
| `_write_submission_source_readme` | 4554 |
| `_write_abstract_helper_files` | 5085 |
| `_write_theorem_verification_badge` | 5510 |
| `_write_dependency_bridge_module` | 6088 |
| `_write_graph_manifest` | 6306 |
| `_write_graph_index_html` | 6372 |

**Collapse into**: `FileWriter` class - one method per file type.

### 2. `_extract_*` methods (19 methods)

| Method | Line |
|--------|------|
| `_extract_document_body` | 981 |
| `_extract_primary_namespace` | 2513 |
| `_extract_declared_lean_handle_locations` | 2790 |
| `_extract_declared_lean_handles` | 2873 |
| `_extract_referenced_lean_handles_from_content` | 2888 |
| `_extract_lh_ids_from_content` | 2915 |
| `_extract_referenced_lh_handles_from_content` | 2933 |
| `_extract_alias_code_map_from_handle_aliases` | 2961 |
| `_extract_label_to_regime_from_claimstamps` | 3591 |
| `_extract_label_to_hardness_tags_from_claimstamps` | 3619 |
| `_extract_claim_label_to_lean_handles` | 3842 |
| `_extract_braced_command_argument` | 4601 |
| `_extract_main_latex_title` | 4681 |
| `_extract_abstract_latex` | 4707 |
| `_extract_paper_claim_labels` | 5418 |
| `_extract_immediate_trailing_leanmeta_block` | 5435 |
| `_extract_mapped_claim_labels` | 5571 |

**Collapse into**: `ContentExtractor` class.

### 3. `_sync_*` methods (5 methods)

| Method | Line |
|--------|------|
| `_sync_local_lean_dependency_dirs` | 1191 |
| `_sync_graph_infra_lean` | 1234 |
| `_sync_shared_preambles` | 2266 |
| `_sync_graph_viewer_assets` | 6287 |

**Collapse into**: `FileSyncer` class.

### 4. `_discover_*` / `_collect_*` / `_find_*` methods (9 methods)

| Method | Line |
|--------|------|
| `_discover_template_lean_toolchain` | 567 |
| `_discover_template_mathlib_require` | 584 |
| `_discover_content_files_for_flags` | 941 |
| `_discover_content_files` | 973 |
| `_collect_included_tex_files` | 1091 |
| `_collect_lean_dependency_closure` | 1164 |
| `_collect_graph_json` | 1305 |
| `_find_tex_includes` | 1065 |

**Collapse into**: `FileDiscoverer` class.

### 5. Template methods (10 methods)

| Method | Line |
|--------|------|
| `_latex_main_template` | 609 |
| `_latex_abstract_template` | 639 |
| `_latex_intro_template` | 646 |
| `_references_template` | 653 |
| `_proofs_readme_template` | 657 |
| `_lakefile_template` | 783 |
| `_lean_root_template` | 801 |
| `_lean_basic_template` | 806 |

**Collapse into**: `templates/latex_templates.py` and `templates/lean_templates.py`

### 6. `build_steps` duplication (2 identical)

Lines 4295 and 4402 - identical `pdflatex → bibtex → pdflatex → pdflatex` cycle.

**Collapse into**: `LatexBuildSteps` dataclass.

### 7. Stats computation (multiple similar)

| Method | Line |
|--------|------|
| `_compute_lean_file_stats` | 1864 |
| `_compute_lean_stats` | 1884 |
| `_get_lean_stats` | 1904 |
| `_get_cumulative_lean_stats` | 1927 |
| `_get_content_backed_lean_stats` | 2163 |
| `_get_claim_backed_lean_stats` | 2201 |
| `_get_release_lean_stats` | 2239 |

**Collapse into**: `LeanStatsComputer` class.

---

## Updated Vertical Stack

```
                    ┌─────────────────────┐
                    │    Orchestrator     │
                    └──────────┬──────────┘
                               │
        ┌──────────┬───────────┼───────────┬──────────┐
        ▼          ▼           ▼           ▼          ▼
   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
   │  Lean   │ │  LaTeX  │ │ Markdown │ │   ArXiv │ │Metadata │
   │ Builder │ │ Builder │ │ Builder  │ │ Packager│ │Generator│
   └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘
        │           │           │           │           │
        └───────────┴─────┬─────┴───────────┴───────────┘
                          ▼
        ┌─────────────────────────────────────────────┐
        │              Utilities Layer                 │
        │  ┌─────────────┐  ┌─────────────┐           │
        │  │ FileWriter  │  │ContentExtract│           │
        │  │   (17)     │  │    (29)     │           │
        │  └─────────────┘  └─────────────┘           │
        │  ┌─────────────┐  ┌─────────────┐           │
        │  │FileDiscoverer│  │DataCollector │           │
        │  └─────────────┘  └─────────────┘           │
        └─────────────────────────────────────────────┘
                          │
              ┌───────────────────────┐
              │   PathResolver        │
              └───────────────────────┘
                          │
              ┌───────────────────────┐
              │   PaperMeta / Config  │
              └───────────────────────┘
```

**The 17 + 29 = 46 utility methods get consolidated into 4 focused utility classes.**

---

## EXACT MODULE DEFINITIONS

### Module 1: `scripts/build/config.py` (~150 lines)

**Extracts**: Config classes from lines 208-380

```
config.py
├── ArxivConfig (lines 208-224)
│   ├── include_build_log: bool
│   ├── include_lean_source: bool
│   ├── include_build_config: bool
│   ├── exclude_patterns: Tuple[str, ...]
│   └── from_dict(d: dict) -> ArxivConfig
│
├── BuildConfig (lines 228-255)
│   ├── generated_content_files: Tuple[str, ...]
│   ├── shared_preamble_files: Tuple[str, ...]
│   ├── latex_source_extensions: Tuple[str, ...]
│   ├── lean_exclude_dirs: Tuple[str, ...]
│   └── from_dict(d: dict) -> BuildConfig
│
├── PaperMeta (lines 258-304)
│   ├── paper_id: str
│   ├── name: str
│   ├── full_name: str
│   ├── dir_name: str
│   ├── latex_dir: str
│   ├── latex_file: str
│   ├── venue: str
│   ├── proofs_dir: str
│   ├── module_root: str
│   ├── archive_prefix: str
│   ├── lean_dependencies: Tuple[str, ...]
│   ├── lean_stats_scope: str
│   ├── assumption_ledger_sources: Tuple[str, ...]
│   ├── claim_mapping_file: str
│   ├── arxiv_comments: str
│   ├── scaffold_from: str
│   ├── experiment_paths: Tuple[str, ...]
│   ├── experiment_commands: Tuple[str, ...]
│   └── from_dict(paper_id: str, d: dict) -> PaperMeta
│
├── LeanStats (lines 308-314)
│   ├── line_count: int
│   ├── theorem_count: int
│   ├── sorry_count: int
│   └── file_count: int
│
└── LeanFileStats (lines 318-323)
    ├── line_count: int
    ├── theorem_count: int
    └── sorry_count: int
```

### Module 2: `scripts/build/path_resolver.py` (~80 lines)

**Extracts**: Path methods from lines 418-456

```
path_resolver.py
├── PathResolver
│   ├── __init__(papers_dir: Path, papers: dict)
│   ├── _get_paper_meta(paper_id: str) -> PaperMeta
│   ├── derive(paper_id: str, *parts: str) -> Path
│   ├── releases(paper_id: str) -> Path (creates if needed)
│   ├── get_proofs_dir(paper_id: str) -> Path
│   ├── get_latex_dir(paper_id: str) -> Path
│   ├── get_content_dir(paper_id: str) -> Path
│   ├── get_markdown_dir(paper_id: str) -> Path
│   └── get_markdown_file(paper_id: str) -> Path
│
└── PaperPathConstants (static)
    ├── CONTENT = "content"
    ├── PROOFS = "proofs"
    ├── RELEASES = "releases"
    └── MARKDOWN = "markdown"
```

### Module 3: `scripts/build/utils/file_writer.py` (~200 lines)

**Extracts**: All `_write_*` methods (17 methods)

```
file_writer.py
├── FileWriter
│   ├── __init__(path_resolver: PathResolver)
│   │
│   ├── write_text_file(path, content, overwrite) -> (created, skipped)
│   ├── write_graph_export_lean(paper_id: str)
│   ├── write_lean_handle_macros_auto(paper_id: str)
│   ├── write_shared_lean_handle_macros_auto()
│   ├── write_latex_lean_stats(paper_id: str)
│   ├── write_assumption_ledger_auto(paper_id: str)
│   ├── write_lean_handle_ids_auto(paper_id: str)
│   ├── write_proof_hardness_index_auto(paper_id: str)
│   ├── write_claim_mapping_auto(paper_id: str)
│   ├── write_submission_wrapper_tex(target_dir: Path) -> str
│   ├── write_submission_build_hook(target_dir: Path) -> str
│   ├── write_submission_source_readme(paper_id: str)
│   ├── write_abstract_helper_files(paper_id: str)
│   ├── write_theorem_verification_badge(paper_id: str)
│   ├── write_dependency_bridge_module(paper_id: str, dest: Path, dep_ids: List[str])
│   ├── write_graph_manifest(paper_id: str, dest_dir: Path)
│   └── write_graph_index_html(paper_id: str, dest_dir: Path)
```

### Module 4: `scripts/build/utils/content_extractor.py` (~300 lines)

**Extracts**: All `_extract_*` methods (19 methods)

```
content_extractor.py
├── ContentExtractor
│   ├── __init__(path_resolver: PathResolver)
│   │
│   ├── extract_document_body(content: str) -> str
│   ├── extract_primary_namespace(content: str) -> Optional[str]
│   ├── extract_declared_lean_handle_locations(content: str) -> Dict
│   ├── extract_declared_lean_handles(content: str) -> Set[str]
│   ├── extract_referenced_lean_handles_from_content(content: str) -> Set[str]
│   ├── extract_lh_ids_from_content(content: str) -> List[str]
│   ├── extract_referenced_lh_handles_from_content(content: str) -> Set[str]
│   ├── extract_alias_code_map_from_handle_aliases(content: str) -> Dict
│   ├── extract_label_to_regime_from_claimstamps(content: str) -> Dict
│   ├── extract_label_to_hardness_tags_from_claimstamps(content: str) -> Dict
│   ├── extract_claim_label_to_lean_handles(content: str) -> Dict
│   ├── extract_braced_command_argument(content: str, command: str) -> Optional[str]
│   ├── extract_main_latex_title(paper_id: str) -> Optional[str]
│   ├── extract_abstract_latex(paper_id: str) -> Optional[str]
│   ├── extract_paper_claim_labels(paper_id: str) -> Set[str]
│   ├── extract_immediate_trailing_leanmeta_block(text: str, start: int) -> str
│   └── extract_mapped_claim_labels(mapping_file: Path) -> Set[str]
```

### Module 5: `scripts/build/utils/file_syncer.py` (~150 lines)

**Extracts**: All `_sync_*` methods (5 methods)

```
file_syncer.py
├── FileSyncer
│   ├── __init__(path_resolver: PathResolver, repo_root: Path)
│   │
│   ├── sync_local_lean_dependency_dirs(paper_id: str)
│   ├── sync_graph_infra_lean(paper_id: str)
│   ├── sync_shared_preambles(paper_id: str)
│   ├── sync_graph_viewer_assets(dest_dir: Path)
│   └── _files_identical(left: Path, right: Path) -> bool
```

### Module 6: `scripts/build/utils/file_discoverer.py` (~200 lines)

**Extracts**: `_discover_*`, `_collect_*`, `_find_*` methods (9 methods)

```
file_discoverer.py
├── FileDiscoverer
│   ├── __init__(path_resolver: PathResolver, build_config: BuildConfig)
│   │
│   ├── discover_template_lean_toolchain() -> str
│   ├── discover_template_mathlib_require() -> str
│   ├── discover_content_files_for_flags(paper_id: str, flags: set) -> List[Path]
│   ├── discover_content_files(paper_id: str) -> List[Path]
│   ├── find_tex_includes(content: str, base_dir: Path) -> List[Path]
│   ├── collect_included_tex_files(content: str, base_dir: Path) -> List[Path]
│   ├── collect_lean_dependency_closure(paper_id: str) -> List[str]
│   ├── collect_graph_json(paper_id: str)
│   └── iter_manual_content_tex(paper_id: str, files: Optional[List[Path]]) -> List[Path]
```

### Module 7: `scripts/build/utils/lean_stats.py` (~200 lines)

**Extracts**: Stats computation methods

```
lean_stats.py
├── LeanStatsComputer
│   ├── __init__(path_resolver: PathResolver)
│   │
│   ├── compute_lean_file_stats(proofs_dir: Path) -> Dict[str, LeanFileStats]
│   ├── compute_lean_stats(proofs_dir: Path) -> LeanStats
│   ├── get_lean_stats(paper_id: str) -> LeanStats
│   ├── get_cumulative_lean_stats(paper_id: str) -> LeanStats
│   ├── get_content_backed_lean_stats(paper_id: str) -> LeanStats
│   ├── get_claim_backed_lean_stats(paper_id: str) -> LeanStats
│   ├── get_release_lean_stats(paper_id: str) -> LeanStats
│   ├── get_lean_file_stats(paper_id: str) -> Dict[str, LeanFileStats]
│   ├── count_theorems_and_sorries(content: str) -> Tuple[int, int]
│   └── strip_lean_comments(content: str) -> str
```

### Module 8: `scripts/build/templates/latex_templates.py` (~150 lines)

**Extracts**: LaTeX template methods (lines 609-700)

```
latex_templates.py
├── LatexTemplates
│   ├── __init__(path_resolver: PathResolver)
│   │
│   ├── main_template(meta: PaperMeta) -> str
│   ├── abstract_template(meta: PaperMeta) -> str
│   ├── intro_template(meta: PaperMeta) -> str
│   ├── references_template() -> str
│   ├── replace_first_latex_title(content: str, new_title: str) -> str
│   └── build_derived_latex_scaffold(paper_id: str, source_id: str)
```

### Module 9: `scripts/build/templates/lean_templates.py` (~100 lines)

**Extracts**: Lean template methods (lines 783-815)

```
lean_templates.py
├── LeanTemplates
│   ├── __init__(path_resolver: PathResolver)
│   │
│   ├── lakefile_template(paper_id: str, paper_files: list) -> str
│   ├── lean_root_template(module_root: str) -> str
│   ├── lean_basic_template(module_root: str) -> str
│   └── generate_paper_lakefile(paper_id: str, paper_files: list, dest: Path)
```

### Module 10: `scripts/build/build_steps.py` (~50 lines)

**Extracts**: Build step definitions (lines 4295, 4402)

```
build_steps.py
├── LatexBuildStep (dataclass)
│   ├── command: Tuple[str, ...]
│   └── name: str
│
├── LatexBuildSteps
│   ├── STANDARD: LatexBuildSteps  (pdflatex → bibtex → pdflatex → pdflatex)
│   ├── SUBMISSION: LatexBuildSteps  (with jobname)
│   └── from_commands(commands: List[Tuple[str, ...]], names: List[str]) -> LatexBuildSteps
```

### Module 11: `scripts/build/builders/lean_builder.py` (~400 lines)

**Extracts**: `build_lean` + Lean-related methods

```
lean_builder.py
├── LeanBuilder
│   ├── __init__(path_resolver: PathResolver, file_syncer: FileSyncer, stats_computer: LeanStatsComputer)
│   │
│   ├── build(paper_id: str, verbose: bool = True) -> str
│   ├── iter_paper_lean_files(proofs_dir: Path) -> List[Path]
│   ├── iter_lean_roots_for_paper(paper_id: str) -> List[Tuple[str, Path]]
│   ├── derive_module_roots_from_lakefile(proofs_dir: Path) -> List[str]
│   ├── derive_module_roots(paper_id: str) -> List[str]
│   ├── extract_declared_handle_to_module_map(content: str) -> Dict
│   ├── build_lean_module_index(paper_id: str) -> Dict
│   ├── get_handle_module_closure(paper_id: str) -> Set[str]
│   ├── get_claim_backed_module_closure(paper_id: str) -> Dict[str, Set[str]]
│   ├── get_content_backed_module_closure(paper_id: str) -> Dict[str, Set[str]]
│   └── _run_lake_build(proofs_dir: Path, args: List[str], verbose: bool) -> Tuple[int, List[str]]
```

### Module 12: `scripts/build/builders/latex_builder.py` (~500 lines)

**Extracts**: `build_latex` + related methods

```
latex_builder.py
├── LatexBuilder
│   ├── __init__(path_resolver, file_writer, content_extractor, ...)
│   │
│   ├── build(paper_id: str, verbose: bool = False)
│   ├── sync_shared_preambles(paper_id: str)
│   ├── write_lean_handle_macros_auto(paper_id: str)
│   ├── write_latex_lean_stats(paper_id: str)
│   ├── write_assumption_ledger_auto(paper_id: str)
│   ├── normalize_and_fill_claimstamps(paper_id: str)
│   ├── write_claim_mapping_auto(paper_id: str)
│   ├── write_proof_hardness_index_auto(paper_id: str)
│   ├── rewrite_content_lean_handles_to_ids(paper_id: str)
│   ├── expand_lean_stat_macros(text: str, paper_id: str) -> str
│   └── _run_latex_build(latex_dir: Path, latex_file: str, verbose: bool)
```

### Module 13: `scripts/build/builders/markdown_builder.py` (~300 lines)

**Extracts**: `build_markdown` + related methods

```
markdown_builder.py
├── MarkdownBuilder
│   ├── __init__(path_resolver, content_extractor, ...)
│   │
│   ├── build(paper_id: str)
│   ├── build_markdown_file(meta, content_files, out_file, title, lean_stats)
│   ├── convert_latex_to_markdown(latex_content: str) -> str
│   ├── normalize_claimstamp_for_markdown(latex: str) -> str
│   ├── prepend_markdown_macro_prelude(latex: str) -> str
│   ├── postprocess_markdown_text(text: str) -> str
│   ├── latex_snippet_to_plain(latex: str, paper_id: str) -> str
│   ├── latex_snippet_to_mathjax(latex: str, paper_id: str) -> str
│   ├── mathjax_markdown_to_unicode_markdown(text: str) -> str
│   ├── convert_fake_subscripts_to_unicode(text: str) -> str
│   ├── convert_display_math_to_unicode(text: str) -> str
│   └── latex_snippet_to_unicode_zenodo(latex: str, paper_id: str) -> str
```

### Module 14: `scripts/build/builders/submission_builder.py` (~300 lines)

**Extracts**: `build_submission` + related methods

```
submission_builder.py
├── SubmissionBuilder
│   ├── __init__(path_resolver, latex_builder, ...)
│   │
│   ├── build(paper_id: str, verbose: bool = False)
│   ├── package_submission_source(paper_id: str) -> Optional[Path]
│   ├── write_submission_wrapper_tex(target_dir: Path) -> str
│   ├── write_submission_build_hook(target_dir: Path) -> str
│   ├── write_submission_source_readme(paper_id: str)
│   └── default_arxiv_comments(paper_id: str) -> str
```

### Module 15: `scripts/build/builders/arxiv_packager.py` (~600 lines)

**Extracts**: `package_arxiv` + related methods

```
arxiv_packager.py
├── ArxivPackager
│   ├── __init__(path_resolver, file_writer, file_syncer, ...)
│   │
│   ├── package(paper_id: str)
│   ├── validate_and_get_pdf(paper_id: str) -> Path
│   ├── validate_and_get_markdown(paper_id: str) -> Path
│   ├── copy_pdf(pdf_file: Path, package_dir: Path)
│   ├── copy_markdown(md_file: Path, package_dir: Path)
│   ├── copy_copy_paste_metadata(metadata_file: Path, package_dir: Path)
│   ├── copy_latex_sources(paper_id: str, package_dir: Path)
│   ├── copy_lean_proofs(paper_id: str, package_dir: Path)
│   ├── copy_experiments(paper_id: str, package_dir: Path)
│   ├── copy_experiment_tree(source_dir: Path, dest_dir: Path) -> int
│   ├── resolve_experiment_source(paper_id: str) -> Optional[Path]
│   ├── run_experiment_commands(paper_id: str)
│   ├── rewrite_packaged_lakefile_srcdirs(lakefile_path: Path)
│   ├── write_dependency_bridge_module(paper_id: str, dest: Path, dep_ids: List[str])
│   ├── sync_graph_viewer_assets(dest_dir: Path)
│   ├── copy_graph_visualizer(paper_id: str, package_dir: Path)
│   ├── write_graph_manifest(paper_id: str, dest_dir: Path)
│   ├── write_graph_index_html(paper_id: str, dest_dir: Path)
│   ├── generate_paper_lakefile(paper_id: str, paper_files: list, dest: Path)
│   ├── generate_proofs_readme(paper_id: str, paper_files: list, dest: Path)
│   ├── create_build_log(paper_id: str, package_dir: Path)
│   ├── create_named_archive(paper_id: str, package_dir: Path) -> Path
│   └── create_archive(paper_id: str, package_dir: Path) -> Tuple[Path, Path]
```

### Module 16: `scripts/build/builders/metadata_generator.py` (~200 lines)

**Extracts**: `build_copy_paste_metadata` + related methods

```
metadata_generator.py
├── MetadataGenerator
│   ├── __init__(path_resolver, content_extractor, ...)
│   │
│   ├── generate(paper_id: str) -> Path
│   ├── get_claim_mapping_file(paper_id: str) -> Optional[Path]
│   ├── release_title_from_latex(paper_id: str) -> str
│   ├── extract_abstract_latex(paper_id: str) -> Optional[str]
│   ├── normalize_plaintext_block(text: str) -> str
│   ├── clean_metadata_reference_tokens(text: str) -> str
│   └── default_arxiv_comments(paper_id: str) -> str
```

### Module 17: `scripts/build/builders/claim_checker.py` (~200 lines)

**Extracts**: `check_claim_coverage` + related methods

```
claim_checker.py
├── ClaimChecker
│   ├── __init__(path_resolver, content_extractor, ...)
│   │
│   ├── check_coverage(paper_id: str, fail_on_missing: bool = False)
│   ├── check_theorem_like_lean_annotations(paper_id: str)
│   ├── write_theorem_verification_badge(paper_id: str)
│   ├── check_forcing_coverage(paper_id: str) -> Optional[Dict]
│   └── update_paper_date(tex_file: Path)
```

### Module 18: `scripts/build/builders/axiom_checker.py` (~200 lines)

**Extracts**: `check_axioms` method

```
axiom_checker.py
├── AxiomChecker
│   ├── __init__(path_resolver, stats_computer)
│   │
│   └── check(paper_id: str, verbose: bool = True) -> dict
```

### Module 19: `scripts/build/orchestrator.py` (~300 lines)

**Extracts**: `release`, `build_all` + scaffold

```
orchestrator.py
├── Orchestrator
│   ├── __init__(repo_root: Path)
│   │   # Loads all configs, initializes all builders
│   │
│   ├── release(paper_id: str, verbose: bool = True, axiom_check: bool = False, claim_check: bool = False)
│   ├── build_all(build_type: str, verbose: bool = True, axiom_check: bool = False, claim_check: bool = False)
│   ├── scaffold_paper(paper_id: str, overwrite: bool = False)
│   ├── check_claim_coverage(paper_id: str, fail_on_missing: bool = False)
│   └── check_axioms(paper_id: str, verbose: bool = True) -> dict
```

### Module 20: `scripts/build_papers.py` (~50 lines)

**CLI entry point** - thin wrapper

```
build_papers.py (CLI)
├── main() - parses args, calls Orchestrator
└── main() invoked if __name__ == "__main__"
```

---

## ALIGNMENT WITH REFACTORING DOCS

### From refactoring_principles.rst:
- [x] "Algebraic Common Factors" - PathResolver, FileWriter, ContentExtractor factor common patterns
- [x] "Single-Use Function Inlining" - wrapper methods deleted
- [x] "Mathematical Simplification" - duplicate build_steps consolidated

### From architectural_refactoring_patterns.rst:
- [x] "Explicit Dependency Injection" - builders take PathResolver in constructor
- [x] "Indirection Minimization" - wrapper methods removed
- [x] "Fail-Loud" - already in codebase
- [x] "Consistent Interface Design" - uniform patterns
- [ ] "ABC Contract Enforcement" - NOT YET ADDED

### Add: ABC Contracts for Builders

Per docs: "Use ABCs to enforce explicit contracts and enable polymorphism."

The ABC should hold ALL common methods - this collapses duplication:

```python
# scripts/build/builders/base.py
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

class BuildStrategy(ABC):
    """ABC contract for all builders - COLLSPSES common methods into ONE class."""
    
    def __init__(self, path_resolver: 'PathResolver'):
        self.path_resolver = path_resolver
    
    # COMMON METHODS - all builders inherit these (COLLAPSED from 8 builders)
    
    def get_paper_dir(self, paper_id: str) -> Path:
        return self.path_resolver.derive(paper_id)
    
    def get_latex_dir(self, paper_id: str) -> Path:
        meta = self.path_resolver.get_meta(paper_id)
        return self.path_resolver.derive(paper_id, meta.latex_dir)
    
    def get_content_dir(self, paper_id: str) -> Path:
        meta = self.path_resolver.get_meta(paper_id)
        return self.path_resolver.derive(paper_id, meta.latex_dir, "content")
    
    def get_releases_dir(self, paper_id: str) -> Path:
        return self.path_resolver.releases(paper_id)
    
    def get_proofs_dir(self, paper_id: str) -> Path:
        meta = self.path_resolver.get_meta(paper_id)
        return self.path_resolver.derive(paper_id, meta.proofs_dir)
    
    def validate_structure(self, paper_id: str) -> bool:
        """Validate paper has required structure."""
        return self.get_paper_dir(paper_id).exists()
    
    def get_meta(self, paper_id: str) -> PaperMeta:
        return self.path_resolver.get_meta(paper_id)
    
    # ABSTRACT METHODS - must be implemented by each builder
    
    @abstractmethod
    def build(self, paper_id: str, **kwargs) -> Any:
        """Build the artifact for this paper."""
        pass
    
    @abstractmethod
    def supports(self, paper_id: str) -> bool:
        """Check if this builder supports the given paper."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Builder name for logging."""
        pass
```

**This collapses:**
- 5 path getter methods × 8 builders = 40 methods → 5 methods in ABC
- 8 `validate_structure` methods → 1 method in ABC
- 8 `get_meta` calls → 1 method in ABC

**Total: ~50 methods collapsed into ABC**

---

## SUMMARY: What Changes

| Original | Refactored |
|----------|------------|
| 1 file (`build_papers.py`) | 20 files across `build/` directory |
| 7789 lines | ~4500 lines (42% reduction) |
| 179 methods in 1 class | 179 methods across 20 focused classes |
| 80 path call sites in 1 class | Distributed across 8 builders |

---

## Execution Order

1. **Phase 1**: Extract `config.py` (safest, no dependencies)
2. **Phase 2**: Extract `path_resolver.py` (depends on config)
3. **Phase 3**: Extract utilities (depend on path_resolver)
   - `file_writer.py`
   - `content_extractor.py`
   - `file_syncer.py`
   - `file_discoverer.py`
   - `lean_stats.py`
4. **Phase 4**: Extract templates
   - `latex_templates.py`
   - `lean_templates.py`
   - `build_steps.py`
5. **Phase 5**: Extract builders (depend on utilities)
   - `markdown_builder.py` (simplest first)
   - `metadata_generator.py`
   - `claim_checker.py`
   - `axiom_checker.py`
   - `submission_builder.py`
   - `lean_builder.py`
   - `latex_builder.py`
   - `arxiv_packager.py` (most complex last)
6. **Phase 6**: Extract `orchestrator.py`
7. **Phase 7**: Wire up CLI
