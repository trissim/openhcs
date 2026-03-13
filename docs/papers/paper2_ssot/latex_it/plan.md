# Paper 2 IT Revision Plan

## Overview
Restructure the IEEE TIT submission based on reviewer feedback. This plan tracks all changes with motivations and status.

## Change Log

---

### Item 1: Move Affine Section After Graph Characterization

**Motivation**: The matroid on coordinates is a *dual* characterization of the same structure that gives the confusability graph. Currently it reads as an "aside" in Section 11. Moving it earlier makes it feel like the natural dual to the state-side graph characterization.

**Location**: Currently Section 11 (after equality). Should be Section 8 (after graph characterization).

**Files affected**:
- `coherence_it.tex` (main file, section order)
- `content/11_affine.tex` (rename to `08_affine.tex`)
- All files that cross-reference Section 11

**Specific edits**:

| Status | File | Change |
|--------|------|--------|
| [x] | `coherence_it.tex` | Change line 88 from `\input{content/11_affine.tex}` to `\input{content/08_affine.tex}` |
| [x] | `coherence_it.tex` | Move `\input{content/08_affine.tex}` from after line 87 to after line 84 (after graph section) |
| [x] | `content/11_affine.tex` | Copy to `content/08_affine.tex` |
| [x] | `content/08_affine.tex` | Update section label from `\section{Affine Fact-Side Structure}` to `\section{Affine Dual: Coordinate Matroid}\label{sec:affine}` |
| [x] | Add bridge paragraph in `content/07_graph.tex` | Added after line 139 |

**Cross-reference updates needed**:
- [x] Verified: cross-references use labels (`\ref{sec:affine}`), not hardcoded numbers - no changes needed

---

### Item 2: Formalize and Move Evaluation to Appendix

**Motivation**: Currently reads as narrative prose ("Pattern A/B/C") which weakens rigor. Should be moved to appendix while keeping a brief reference in main text summarizing surprising findings.

**Location**: 
- Main text: content/05_evaluation.tex (shortened to 1-2 paragraphs)
- Appendix: new content/13_classification.tex

**Specific edits**:

| Status | File | Change |
|--------|------|--------|
| [x] | `content/05_evaluation.tex` | Replaced with 1-2 paragraph summary referencing appendix |
| [x] | `content/12_appendix_classification.tex` | Already existed with full table - no changes needed |

**Proposed new main-text structure (content/05_evaluation.tex)**:
```latex
\section{System Classification}\label{sec:evaluation}

Corollary~\ref{cor:ssot-iff} states that verifiable structural integrity holds iff
both causal propagation and provenance observability hold. We apply this criterion
to representative systems. Complete classification with full justifications appears
in Appendix~\ref{sec:system-classification}.

The key finding is that the majority of common infrastructure falls into patterns
that cannot verify integrity. This is not merely a maintenance inconvenience but
an architectural exposure: the host cannot distinguish valid derived states from
undetected stale states through its own structural interface.
```

**Proposed appendix structure (content/13_classification.tex)**:
```latex
\appendix
\section{System Classification}\label{sec:system-classification}

\begin{table}[h]
\begin{tabular}{|l|c|c|c|}
\hline
System & Causal Propagation & Provenance & Verifiable Integrity \\
\hline
npm/Cargo & No & No & No \\
Rust runtime & No & Yes & No \\
Materialized views & Yes & Yes & Yes \\
\hline
\end{tabular}
\end{table}

Justification for each entry: [bullet list with citations]
```

---

### Item 3: Trim Introduction

**Motivation**: Currently ~130 lines, too long for TIT. TIT intros should be ~2 pages.

**Location**: content/01_introduction.tex

**Specific edits**:

| Status | File | Change |
|--------|------|--------|
| [x] | `content/01_introduction.tex` | Removed roadmap table |
| [x] | `content/01_introduction.tex` | Trimmed "Novelty boundary" paragraph to 1 sentence |
| [x] | `content/01_introduction.tex` | Removed "Terminology and Analogies" subsection (table) |
| [x] | `content/01_introduction.tex` | Removed "Relation to Classical IT" subsection |

**Target**: Reduce intro from 130 lines to ~80 lines

---

### Item 4: Rewrite Abstract

**Motivation**: Currently too dense, reads like compressed table of contents. TIT abstracts should highlight: (a) model, (b) main characterization, (c) capacity result.

**Location**: content/abstract.tex

**Specific edits**:

| Status | File | Change |
|--------|------|--------|
| [x] | `content/abstract.tex` | Rewrote to 6 sentences: model, graph characterization, capacity, equality, affine, instantiation |
| [x] | `content/abstract.tex` | Added punchy last sentence about utility/instantiation |
| [x] | `content/abstract.tex` | Cut: realizability discussion, finite converse details, Lean mention |

---

### Item 5: Update Section Numbers After Affine Move ⏭️ SKIPPED

**Motivation**: Moving affine from Section 11 to Section 8 shifts all subsequent numbers.

**Resolution**: Not needed. LaTeX auto-numbers sections based on order in `coherence_it.tex`. Cross-references use labels (`\ref{sec:affine}`), not hardcoded numbers, so they work correctly regardless of section number.

---

### Item 6: Verify Build ✅ COMPLETE

**After all edits**:

| Status | Action |
|--------|--------|
| [x] | Run `python build_papers.py latex paper2_it` - PASSED |
| [x] | Run `python build_papers.py lean paper2_it` - PASSED (warnings only, no errors) |
| [x] | Check PDF compiles without errors - PASSED |
| [x] | Verify all cross-references resolve - PASSED |

---

## Remaining Work

### Item 7: Commit and Push Changes 🔲 PENDING

| Status | Action |
|--------|--------|
| [ ] | Stage all paper2 changes |
| [ ] | Commit with message: "paper2: restructure for TIT (move affine, trim intro, rewrite abstract)" |
| [ ] | Push to remote |

### Item 8: Optional Polish 🔲 OPTIONAL

| Status | Action |
|--------|--------|
| [ ] | Final language tightening pass |
| [ ] | Check figure/table captions are concise |
| [ ] | Reviewer response letter |

---

## Notes

- Em-dash punctuation: avoid throughout (per user preference)
- When moving sections, preserve all content - do not cut
- After moving affine, verify the mathematical flow still makes sense (graph -> capacity -> theta -> equality -> affine dual)
- **Keep Lean handles** (`\leanmeta{}`): they are discrete in PDF, serve as hyperlinks, and do not obstruct prose
- Item 6 (Lean handle cleanup) is cancelled per user preference
