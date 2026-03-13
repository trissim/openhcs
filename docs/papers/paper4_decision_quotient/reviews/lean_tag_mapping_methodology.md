# Lean Tag → Claim Mapping by Proximity

## The Method

**Simple rule:** The Lean tag `\LH{...}` always supports the **nearest preceding** theorem/definition/claim. Read outward from the tag until you hit the nearest claim above it.

## How Tags Appear in the Text

### Pattern 1: Tag right after theorem

```latex
\begin theorem}[Bounded Actions]
If $|A| \leq k$, then SUFFICIENCY-CHECK runs in $O(|S|^2 \cdot k^2)$.
\end theorem}
\leanmeta{\LH{DQ51}.}
```

→ DQ51 proves the Bounded Actions theorem (line ~58)

### Pattern 2: Tag in proof

```latex
\begin proof
...proof steps...
By Lemma 3.2, we have X. \leanmeta{\LH{DQ44}.}
\end proof
```

→ DQ44 proves the lemma/step at this location

### Pattern 3: Tag after equation/definition

```latex
\[
\Opt(s) = \arg\max_a U(a,s)
\]
\leanmeta{\LH{QT1}.}
```

→ QT1 defines this optimizer mapping

## Finding Any Tag

1. **Search for the tag** in the LaTeX: `grep -n "\\LH{DQ51}" *.tex`
2. **Read the 10-20 lines above it** - that's the claim it supports
3. **Done** - no need to understand file structure or prefixes

## Quick Lookup for Paper 4 Tractability Section

| Tag | Location (approx) | Claim |
|-----|------------------|-------|
| DQ86 | Line ~31 | Single action tractability |
| DQ87 | Line ~32 | Bounded state space |
| DQ88 | Line ~33-34 | Strict dominance / Constant optimal set |
| DQ89 | Line ~35 | Multiplicative separability |
| DQ51 | Line ~36 | Bounded actions |
| DQ57 | Line ~37 | Separable utility |
| DQ77 | Line ~38 | Low tensor rank |
| DQ63 | Line ~39 | Tree structure |
| DQ81 | Line ~40 | Bounded treewidth |
| DQ85 | Line ~41 | Coordinate symmetry |
| DQ90 | Line ~42 | FPT |

## Spawning Parallel Audits

Just tell subagent: "Find tag X in the LaTeX, read the claim 10 lines above it, verify against Lean file for that handle."

---

*That's it - proximity only.*
