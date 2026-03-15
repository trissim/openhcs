# Plan: LaTeX Table Column Width Optimizer

## Problem

Table column widths in paper4 (and other papers) are hand-tuned by trial and error.
The goal is to automatically compute optimal `\linewidth`-fraction column widths so that:
1. No word is ever split mid-word in a column (hard minimum: column ≥ widest word)
2. Remaining space is distributed proportionally to how much natural content each column has

## Correct Approach: Use LaTeX to Measure LaTeX

Do not use character counts or font-width heuristics.
LaTeX knows the exact rendered width of every piece of content.
We exploit this by writing a probe document that emits measurements via `\typeout`.

### How the probe works

```latex
% probe.tex
\documentclass[11pt]{article}
\usepackage{paper-preamble}   % same preamble as actual paper
\usepackage{tabularx, ragged2e, calc}
\newlength{\probewidth}
\begin{document}

% For each cell in column i, row j:
\settowidth{\probewidth}{cell content here}
\typeout{PROBE:col=0:row=0:type=cell:\the\probewidth}

% For each word w in column i:
\settowidth{\probewidth}{word}
\typeout{PROBE:col=0:word=0:type=word:\the\probewidth}

\stop
\end{document}
```

Run `pdflatex probe.tex` and parse stdout for `PROBE:` lines → exact widths in `pt`.

### Why this is correct, not heuristic

- `\settowidth` uses the actual font metrics (Computer Modern / Latin Modern via T1 encoding)
- Same preamble as the paper → same fonts, same math setup
- Widths are in `pt`, convertible to `\linewidth` fractions using the paper's known `textwidth`
  - geometry: `margin=1in`, `\documentclass[11pt]{article}` → textwidth = `8.5in - 2in = 6.5in = 468.0pt`
  - We can also measure `\textwidth` itself in the probe to be robust

---

## Algorithm

### Inputs
- One `.tex` file with one or more `tabularx` environments
- Paper preamble path (for the probe)

### Step 1: Parse — extract tables

For each `\begin{tabularx}{...}{colspec}` ... `\end{tabularx}`:
- Extract the current `colspec` (second `{...}` argument)
- Parse `colspec` to enumerate column indices and their current formatting decorators
  - e.g., `>{\RaggedRight\arraybackslash\hyphenpenalty=10000}m{0.15\linewidth}` → col 0, decorator = `>{\RaggedRight...}`, type = `m`
  - e.g., `>{\RaggedRight\arraybackslash}X` → col 2, decorator, type = `X`
- Extract data rows (skip `\toprule`, `\midrule`, `\bottomrule`, header rows with `\textbf`)
- Split rows into cells using brace-depth-aware `&` tokenizer (don't split inside `{...}` or `$...$`)
- For each cell, also split into words (same tokenizer but on whitespace outside commands/math)

### Step 2: Write probe document

For each table, for each (col, cell) pair:
- Emit: `\settowidth{\probewidth}{CELL_CONTENT}\typeout{PROBE:t=T:c=C:r=R:type=cell:\the\probewidth}`

For each table, for each (col, word) pair:
- Emit: `\settowidth{\probewidth}{WORD}\typeout{PROBE:t=T:c=C:w=W:type=word:\the\probewidth}`

Also emit: `\typeout{PROBE:textwidth:\the\textwidth}` to get the actual linewidth.

### Step 3: Run probe

`pdflatex -interaction=nonstopmode probe.tex`

Parse stdout for `PROBE:` lines → dict of widths keyed by (table, col, row/word).

Errors in the probe (invalid LaTeX in a cell) → log warning, skip that table, leave original spec untouched.

### Step 4: Compute optimal widths

For each column `i` in table `T`:
```
min_width[i]  = max over all words in col i of word_width_pt
nat_width[i]  = max over all cells in col i of cell_width_pt
```

Available linewidth:
```
textwidth_pt = measured from probe (or fallback: 468.0pt for 11pt article, 1in margins)
available_pt = textwidth_pt  (tabularx fills full width)
```

Allocation (two-phase):
```
phase1[i] = min_width[i]
if sum(phase1) >= available_pt:
    WARN: content is too wide even for minimum widths, skip this table

remaining_pt = available_pt - sum(phase1)
extra[i]     = nat_width[i] - min_width[i]   # how much extra each col wants
total_extra  = sum(extra)
if total_extra == 0:
    # all columns already at natural width; distribute remaining evenly
    phase2[i] = remaining_pt / n_cols
else:
    phase2[i] = remaining_pt * extra[i] / total_extra

final_width_pt[i]   = phase1[i] + phase2[i]
final_width_frac[i] = final_width_pt[i] / textwidth_pt
```

Round fractions to 2 decimal places. Sum may be slightly off 1.0 due to rounding; assign remainder to the largest column.

### Step 5: Rewrite column spec

For each column, reconstruct the column spec token:
- Preserve existing decorator (`>{\RaggedRight\arraybackslash...}`)
- Replace `m{OLD_FRAC\linewidth}` or `X` with `m{NEW_FRAC\linewidth}`
- Columns that were `X` get a computed fraction (no longer `X` unless it's truly dominant)

Replace the column spec argument in the source file. Use a regex that matches the second `{...}` of `\begin{tabularx}{\linewidth}{...}`.

The rewrite is idempotent: same table content → same probe output → same widths → same spec.

---

## Key implementation details

### Brace-depth-aware tokenizer
```python
def split_cells(row: str) -> list[str]:
    """Split a LaTeX table row on & outside braces and math."""
    cells, current, depth, in_math = [], [], 0, False
    for ch in row:
        if ch == '$': in_math = not in_math
        if not in_math:
            if ch == '{': depth += 1
            elif ch == '}': depth -= 1
            elif ch == '&' and depth == 0:
                cells.append(''.join(current).strip())
                current = []
                continue
        current.append(ch)
    cells.append(''.join(current).strip())
    return cells
```

### Word splitting in LaTeX content
A "word" for minimum-width purposes is any maximal token with no breakable whitespace:
- Split on ` ` (space) and `~` (non-breaking space counts as breakable for our purposes)
- LaTeX commands (`\cmd{...}`) at the start/end of a word may extend it but don't split it
- Math `$...$` is one atomic token
- Hyphenated words like `coNP-complete` — the hyphen IS a break point, so split there too
  - This means `coNP` and `complete` are separate words → min width is `\the\widthof{coNP-complete}` vs max of parts
  - Actually: for min_width we want the longest unbreakable run. LaTeX will break at hyphens by default (unless `\hyphenpenalty=10000`).
  - If the column already has `\hyphenpenalty=10000\exhyphenpenalty=10000`, then the whole hyphenated token is one word.
  - Parse the decorator to detect this: if decorator contains `hyphenpenalty=10000`, words don't break at hyphens.

### Probe preamble
The probe must be run from the paper's latex directory so that `\usepackage{pentalogy-preamble}` resolves.
Use `cwd=latex_dir` when running pdflatex.

### Handling `@{}` suppression tokens and column separators
The `@{}` at start/end of colspec suppresses inter-column padding. This doesn't affect measurements but must be preserved in the rewrite (it's outside the per-column tokens).

---

## Integration into build_papers.py

### New method
```python
def _optimize_table_column_widths(self, paper_id: str) -> None:
    """Rewrite tabularx column specs with LaTeX-measured optimal widths."""
    ...
```

### Wire into pipeline
In `_refresh_derived_content()`, call AFTER `_sync_shared_preambles()` (preamble must exist)
but BEFORE pdflatex runs:
```python
self._optimize_table_column_widths(paper_id)
```

### Standalone script
`scripts/optimize_table_widths.py` — can be run standalone:
```
python scripts/optimize_table_widths.py paper4_decision_quotient
python scripts/optimize_table_widths.py paper4_decision_quotient --file content/06_regime_hierarchy.tex
python scripts/optimize_table_widths.py paper4_decision_quotient --dry-run   # print proposed specs, no writes
```

---

## File structure

```
scripts/
  optimize_table_widths.py     # standalone script + library
  PLAN_table_optimizer.md      # this file
```

The library functions are importable by build_papers.py.

---

## What we are NOT doing

- Not writing a new auto-generated file (we rewrite in-place, since the table content is authored)
- Not touching `longtable` environments (different pagination model)
- Not optimizing tables with `\multicolumn` (defer: would need to solve a linear system)
- Not changing `\toprule`/`\midrule`/`\bottomrule` or row content — only the colspec

---

## Open question: `X` column preservation

Current tables mix `m{frac}` fixed columns with one `X` expanding column.
With our optimizer, all columns get computed fractions that sum to 1.0 — no `X` needed.
This is actually fine: `tabularx` with all `m{}` columns whose fractions sum to ≈1.0 is equivalent.
But if we want to keep the `X` semantic (let one column absorb slack from rounding), we can:
- Assign the column with largest `extra[i]` as `X`
- Assign other columns `m{frac\linewidth}` fractions that sum to < 1.0
This is a config option: `--keep-x-column` (default: True).

---

## Testing plan

1. Run optimizer on paper4 `06_regime_hierarchy.tex` in dry-run mode — inspect proposed fractions
2. Run `pdflatex decision_quotient_arxiv.tex` and compare PDF visually
3. Verify idempotency: run optimizer twice, check file is unchanged second time
4. Run existing build pipeline: `python scripts/build_papers.py latex paper4_decision_quotient`
