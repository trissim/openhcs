# Lean Tag Mapping Methodology for Paper Auditing

## Overview

This document describes the systematic method for mapping Lean proof handles (`\LH{...}`, `\LHrng{...}`) to their relevant claims in the paper text. This enables parallel auditing where multiple subagents can work on different sections simultaneously.

## The Mapping Problem

The paper contains Lean proof handles that reference formal proofs in the artifact. To audit whether:
1. Each claim has proper Lean support
2. The Lean proofs actually prove what's claimed
3. No gaps exist between formal and informal arguments

...we need a reliable way to map tags → claims.

## Three Mapping Strategies

### 1. Geometric Mapping (Position-Based)

Each Lean tag can be located by its **position** in the document:

```
Paper Structure:
├── Section 1 (Introduction)
│   └── Tags typically reference setup/definitions
├── Section 2 (Formal Setup)  
│   └── Tags reference core definitions
├── Section 3 (Complexity)
│   └── Tags reference hardness proofs
├── Section 4 (Tractable Cases)
│   └── Tags reference tractability proofs ← DQ44-DQ90 live here
├── Section 5 (Engineering)
│   └── Tags reference applications
└── Conclusion
```

**How to use geometric mapping:**
1. Note the section number where the tag appears
2. Cross-reference with the Lean file organization
3. Files in `proofs/DecisionQuotient/Tractability/` → Section 4 claims

### 2. Structural Mapping (Hierarchical-Based)

The Lean handle encoding provides **structural hints**:

```
Handle Format: 
  LH{DQ##}  → DecisionQuotient main file
  LH{DN##}  → Decision-noise file  
  LH{DC##}  → Dichotomy file
  LH{CC##}  → Core classification file
  LH{OU##}  → Optimizer quotient file
  LH{EH##}  → Explicit hardness file

Range Format:
  LHrng{DQ}{start}{end} → multiple consecutive handles
```

**Known mappings (Paper 4):**
| Handle Prefix | Lean Module | Paper Section |
|--------------|-------------|--------------|
| DQ | DecisionQuotient.lean | Throughout |
| DN | DecisionNoise | Intro/Setup |
| DC | Dichotomy | Section 3 |
| CC | CoreClassification | Section 3 |
| OU | OptimizerQuotient | Section 2 |
| QT | QuotientTheory | Section 2 |
| EH | ExplicitHardness | Section 3 |

### 3. Proximal Mapping (Nearby-Text Based)

The **nearest text** around a tag identifies the claim:

```latex
% Example 1: Tag immediately follows the theorem
\begin theorem}[Name]\label{...}
  Claim statement...
\end theorem}
\leanmeta{\LH{DQ51}.}  ← DQ51 proves this theorem

% Example 2: Tag embedded in proof
\begin proof
  ...proof steps...
  By Lemma 3.2, we have... \leanmeta{\LH{DQ44}.}
\end proof}  ← DQ44 proves this lemma

% Example 3: Tag after equation
\[
  s_I = s'_I \implies \Opt(s) = \Opt(s')
\]
\leanmeta{\LH{QT1}.}  ← QT1 defines this equivalence
```

## Practical Audit Workflow

### Step 1: Generate Tag Inventory

```bash
# Extract all Lean tags from LaTeX
grep -r "\\LH{" papers/paper4_decision_quotient/latex/content/ | \
  sed 's/.*\\LHrng{\([^}]*\)}{\([^}]*\)}{\([^}]*\)}.*/\1 \2 \3/' | \
  sort -u > tags_inventory.txt
```

### Step 2: Categorize by Type

```
# Count by prefix
grep -o "\\LH{[A-Z]*" file.tex | sort | uniq -c | sort -rn

# Example output:
#   45 LH{DQ
#   12 LH{CC
#    8 LH{DC
#    5 LH{EH
```

### Step 3: Create Audit Batches

Group tags by **section** for parallel processing:

```markdown
## Batch 1: Section 2 (Formal Setup)
- QT1, QT7, OU1-OU12
- Focus: Core definitions and quotient theory

## Batch 2: Section 3 (Static Complexity)  
- DC91-DC99, EH1-EH10
- Focus: Hardness proofs and dichotomy

## Batch 3: Section 4 (Tractability)
- DQ44, DQ51, DQ57, DQ63, DQ77, DQ81, DQ85
- DQ86-DQ90 (new cases)
- Focus: Polynomial-time cases

## Batch 4: Sections 5-6 (Stochastic/Sequential)
- SS* handles
- Focus: Regime-specific results
```

### Step 4: Parallel Audit Instructions

For each subagent, provide:

```markdown
## Subagent Audit Task

**Your section:** Section 4 (Tractability)
**Handles to verify:** DQ44, DQ51, DQ57, DQ63, DQ77, DQ81, DQ85, DQ86-DQ90

**Your job:**
1. Find each handle in the LaTeX source
2. Read the claim it supports
3. Check the corresponding Lean file
4. Verify: Does the Lean theorem actually prove the claim?
5. Report: GAP / MATCH / PARTIAL

**Report format:**
| Handle | Claim Location | Lean Theorem | Status |
|-------|---------------|--------------|--------|
| DQ51 | Line ~58, bounded-actions | bounded_actions_tractable | MATCH |
```

## Example: Auditing Section 4 (Tractability)

### Manual Example

**Handle DQ51** appears in:
```latex
\begin theorem}[Bounded Actions Tractability]
If $|A| \leq k$ for constant $k$, then SUFFICIENCY-CHECK 
runs in $O(|S|^2 \cdot k^2)$ time.
\end theorem}
\leanmeta{\LH{DQ51}, \LH{DQ72}.}
```

**Audit steps:**
1. Look up `DQ51` in Lean → `bounded_actions_tractable` in BoundedActions.lean
2. Read theorem statement in Lean
3. Compare: Does Lean prove O(|S|²·k²) complexity? YES
4. **Status: MATCH**

**Handle DQ86** appears in:
```latex
\item \textbf{Single action:} $|A| = 1$. Any coordinate set is sufficient. 
Complexity: $O(1)$. \leanmeta{\LH{DQ86}.}
```

**Audit steps:**
1. Look up `DQ86` → `single_action_all_sufficient` in SingleAction.lean
2. Read theorem: "if |A| = 1, any I is sufficient"
3. Compare with claim: "Any coordinate set is sufficient. Complexity: O(1)"
4. **Status: MATCH**

## Quick Reference Card

| If handle starts with... | It proves... | Located in section... |
|-------------------------|--------------|----------------------|
| QT | Quotient definitions | 2 |
| OU | Optimizer quotient | 2 |
| DC | Dichotomy theorem | 3 |
| CC | Core classification | 3 |
| EH | Explicit hardness | 3 |
| DQ44 | Structural rank | 4 |
| DQ51 | Bounded actions | 4 |
| DQ57 | Separable utility | 4 |
| DQ63 | Tree structure | 4 |
| DQ77 | Low tensor rank | 4 |
| DQ81 | Bounded treewidth | 4 |
| DQ85 | Coordinate symmetry | 4 |
| DQ86-89 | New trivial cases | 4 |
| DQ90 | FPT | 4 |
| SS | Stochastic/Sequential | 5-6 |

## Spawning Parallel Audits

To audit in parallel, create separate task prompts:

```markdown
**Task 1 (Static Regime):**
- Handles: DC91-99, EH1-10, CC*
- Files: docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Dichotomy.lean
- Verify hardness proofs

**Task 2 (Tractability):**  
- Handles: DQ44, DQ51, DQ57, DQ63, DQ77, DQ81, DQ85, DQ86-90
- Files: docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/*.lean
- Verify tractability proofs

**Task 3 (Stochastic):**
- Handles: SS*
- Files: docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Stochastic*.lean
- Verify regime results

**Task 4 (Lean Coverage):**
- Check for sorry-free modules
- Verify no gaps in proof chain
- Check claim_mapping_auto.tex alignment
```

---

*This methodology enables systematic, auditable verification of the paper's formal foundations.*
