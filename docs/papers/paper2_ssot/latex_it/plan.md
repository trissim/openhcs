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
| [ ] | `coherence_it.tex` | Change line 88 from `\input{content/11_affine.tex}` to `\input{content/08_affine.tex}` |
| [ ] | `coherence_it.tex` | Move `\input{content/08_affine.tex}` from after line 87 to after line 84 (after graph section) |
| [ ] | `content/11_affine.tex` | Rename to `content/08_affine.tex` |
| [ ] | `content/08_affine.tex` | Update section label from `\section{Affine Fact-Side Structure}` to include cross-ref to graph: `\section{Affine Dual: Coordinate Matroid}\label{sec:affine}` |
| [ ] | Add bridge paragraph in `content/07_graph.tex` | After line 139, before Section 8: "The preceding sections studied distinguishability of latent states. A dual question: which fact coordinates determine others across the realized state family? Under an affine restriction, that determination problem becomes span membership of coordinate functionals, yielding a representable matroid on fact indices that provides tractable upper bounds for the main confusability problem. This matroid is not merely a parallel construction: it is the coordinate-side dual of the same structure that generates the confusability graph." |

**Cross-reference updates needed**:
- [ ] Search for `\ref{sec:affine}` and update section numbers
- [ ] Search for `Section~\ref{sec:affine}` and update
- [ ] Update any `\eqref` references

---

### Item 2: Formalize and Move Evaluation to Appendix

**Motivation**: Currently reads as narrative prose ("Pattern A/B/C") which weakens rigor. Should be moved to appendix while keeping a brief reference in main text summarizing surprising findings.

**Location**: 
- Main text: content/05_evaluation.tex (shortened to 1-2 paragraphs)
- Appendix: new content/13_classification.tex

**Specific edits**:

| Status | File | Change |
|--------|------|--------|
| [ ] | `content/05_evaluation.tex` | Replace with 1-2 paragraph summary: "Corollary X states verifiable integrity requires causal propagation and provenance. We apply this to representative systems. Complete classification and justifications appear in Appendix Y. Key finding: most common infrastructure falls into patterns that cannot verify integrity." |
| [ ] | New `content/13_classification.tex` | Full formal table + theorem-statement-then-apply structure |
| [ ] | `coherence_it.tex` | Add `\input{content/13_classification.tex}` before appendix section |

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
| [ ] | `content/01_introduction.tex` | Move roadmap table (lines 14-27) to after Section 2 (foundations) as summary |
| [ ] | `content/01_introduction.tex` | Cut "Novelty boundary" paragraph (lines 30-32) - keep only one sentence |
| [ ] | `content/01_introduction.tex` | Consolidate "Problem Formulation" subsection |
| [ ] | `content/01_introduction.tex` | Cut "Terminology and Analogies" subsection - move to appendix or remove |
| [ ] | `content/01_introduction.tex` | Cut "Relation to Classical IT" subsection - integrate into Related Work |

**Target**: Reduce intro from 130 lines to ~80 lines

---

### Item 4: Rewrite Abstract

**Motivation**: Currently too dense, reads like compressed table of contents. TIT abstracts should highlight: (a) model, (b) main characterization, (c) capacity result.

**Location**: content/abstract.tex

**Specific edits**:

| Status | File | Change |
|--------|------|--------|
| [ ] | `content/abstract.tex` | Rewrite to 3-4 sentences: model, graph characterization, capacity, equality |
| [ ] | `content/abstract.tex` | Move affine/matroid mention to end: "Under an affine restriction, the coordinate structure becomes a representable matroid..." |
| [ ] | `content/abstract.tex` | Cut: realizability discussion, finite converse details, Lean mention |

---

### Item 5: Update Section Numbers After Affine Move

**Motivation**: Moving affine from Section 11 to Section 8 shifts all subsequent numbers.

**Files affected**: All files with section cross-references

| Status | Old Number | New Number |
|--------|------------|------------|
| [ ] | Section 11 (Affine) | Section 8 |
| [ ] | Section 12 (Rate corollaries) | Section 9 |
| [ ] | Section 3 (SSOT) | Section 10 |
| [ ] | Section 4 (Requirements) | Section 11 |
| [ ] | Section 5 (Evaluation) | Section 12 |
| [ ] | Section 7 (Empirical) | Section 13 |
| [ ] | Section 8 (Related) | Section 14 |
| [ ] | Section 9 (Conclusion) | Section 15 |

**Action**: After Item 1 is complete, run grep to find all `\ref{sec:affine}` and update to new section number, then propagate to all shifted sections.

---

### Item 6: Verify Build

**After all edits**:

| Status | Action |
|--------|--------|
| [ ] | Run `python build_papers.py latex paper2_it` |
| [ ] | Run `python build_papers.py lean paper2_it` |
| [ ] | Check PDF compiles without errors |
| [ ] | Verify all cross-references resolve |

---

## Notes

- Em-dash punctuation: avoid throughout (per user preference)
- When moving sections, preserve all content - do not cut
- After moving affine, verify the mathematical flow still makes sense (graph -> capacity -> theta -> equality -> affine dual)
- **Keep Lean handles** (`\leanmeta{}`): they are discrete in PDF, serve as hyperlinks, and do not obstruct prose
- Item 6 (Lean handle cleanup) is cancelled per user preference
