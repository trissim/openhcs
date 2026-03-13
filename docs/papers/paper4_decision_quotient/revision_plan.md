# Revision Plan for Paper 4: Exact Relevance Certification and the Optimizer Quotient

**Status:** Targeting JACM submission deadline (March 31)

---

## Priority 1 (Required for Acceptance)

### 1. Foreground the structural divergence earlier

**Location:** Section 8.2 ("Rough Sets, Reducts, and Attribute Reduction") currently contains the structural insight about unique minimal sufficient sets vs. multiple incomparable reducts. This should appear earlier in the narrative.

**Change:**
- **Option A (Recommended):** Add 2-3 sentences to Section 1 ("Contributions") bullet point 2, after mentioning the static collapse
- **Option B:** Add a brief paragraph after Section 3.3 ("Why You Might Expect Σ₂ᵖ") explaining why the collapse is surprising

**Draft addition (for Section 1):**
> This collapse is structurally non-trivial. In classical rough-set theory, general decision tables can admit multiple incomparable minimal reducts, which drives the minimum-reduct problem into combinatorial search territory. Under the exact optimal-action preservation predicate studied here, Proposition 2.9 proves there is exactly one minimal sufficient set: the relevant-coordinate set itself. This uniqueness is the structural reason for the collapse to coNP relevance containment. To our knowledge, this specific structural collapse—from Σ₂ᵖ search to coNP driven by optimizer-quotient uniqueness—is a novel classification not captured by general rough-set bounds.

**Rationale:** Helps readers understand why the static collapse is surprising, rather than treating it as a routine corollary. Connects the theoretical contribution to the rough-set literature more explicitly early in the narrative.

---

### 2. Add a reduction sketch for Conjecture 11.1

**Location:** In Section 4.2 ("Stochastic Regime Complexity"), immediately after Conjecture 11.1 and its three-part justification (currently ends at line 110).

**Change:** Add a 3-4 sentence paragraph or footnote sketching a candidate reduction approach.

**Draft addition:**
> *Reduction intuition.* A natural hardness reduction would encode static counterexample pairs into zero-probability witness states of the distribution. Concretely, from a TAUTOLOGY instance $\varphi$, one could construct a succinct stochastic decision problem where the distribution assigns zero mass to a carefully chosen set of states that encode the $\varphi$-violating assignments. Preservation must check the universal condition over ALL states (including these zero-probability witnesses), so a coNP-style verification can detect the hidden TAUTOLOGY counterexample even though the distribution never observes it. This intuition aligns with the support-sensitive obstruction lemmas (Proposition 4.14) but requires overcoming the technical challenge of succinctly encoding witness-state support.

**Rationale:** Gives the reader a concrete sense of what a reduction might look like, strengthening the conjecture from a bare structural claim to a plausible technical direction. Does not require a full proof.

---

### 3. Clarify mechanization scope for oracle classes

**Location:** Section 4.2 ("Stochastic Regime Complexity"), in the mechanization note around line 226 (or Section 3 if applicable to static oracle claims).

**Current state:** Paper states "The oracle-class placements are proved in the paper text, while the artifact independently verifies the finite combinatorial core used by the paper's oracle-class arguments."

**Change:** Add 1-2 sentences specifying EXACTLY what is mechanized vs. what is argued informally.

**Draft addition:**
> Concretely, the artifact verifies: (i) the existential-universal quantifier structure of the predicate (e.g., $\exists$ anchor action $\forall$ fiber state), (ii) the witness/checking schema (guess anchor, verify conditional-uniqueness using PP comparisons), and (iii) the finite-step-counted search wrappers (handles OU1–OU2, OU6–OU7) that bound the nondeterministic guessing phase. The standard oracle-machine reduction proving NP^PP membership is argued in the paper text and not mechanized.

**Rationale:** Makes the division between mechanized finite core and informal oracle-machine reasoning precise, addressing reviewer concern about "what exactly is mechanized vs. what is argued informally."

---

## Priority 2 (Strongly Recommended)

### 4. Tighten Section 4 (Tractable Subcases)

**Location:** Section 4.1 ("Tractable Subcases") is currently comprehensive but includes several straightforward cases (separable utility, constant optimal set, single action) that could be merged or abbreviated.

**Change:**
- Merge trivial cases (separable utility, constant optimal set, single action) into a single "simple structural cases" bullet
- Focus the section's main exposition on structurally interesting subcases (low tensor rank, tree structure, bounded treewidth, coordinate symmetry)
- Keep formal definitions for all cases in subsections (as they are used by the artifact), but streamline the narrative table and opening discussion

**Draft restructuring:**
> **Table 3 (streamlined).** Group trivial cases:
> - *Simple structural cases:* single action ($|A|=1$), bounded state space ($|S|\le k$), strict global dominance, constant optimal set, multiplicative separability—all $O(1)$
> - *Nontrivial structural restrictions:* low tensor rank, tree-structured dependencies, bounded treewidth, coordinate symmetry

**Rationale:** Section is already long; tightening without losing technical content improves readability. The artifact still certifies all cases, so no formal coverage is lost.

---

### 5. Add early intuition for stochastic optimizer notation

**Location:** Section 4.2 ("Stochastic Regime Complexity"), immediately after Definition 4.2 (stochastic sufficiency) or after Definition 4.3 (stochastic decisiveness).

**Change:** Add 1-2 sentence paragraph explaining the intuitive difference between $\Opt^{\text{stoch}}_I(\alpha)$ and the full-information optimizer $\Opt(s)$.

**Draft addition:**
> **Intuition.** The conditional optimizer $\Opt^{\text{stoch}}_I(\alpha)$ aggregates expected utilities over the entire fiber $\{s : s_I = \alpha\}$ using the distribution $P$. By contrast, the full-information optimizer $\Opt(s)$ conditions on the exact state $s$ itself. Preservation asks whether this averaging over a fiber ever changes which action is optimal; decisiveness asks whether, after averaging, the optimal action on each fiber is uniquely determined.

**Rationale:** Helps readers new to stochastic decision theory connect the formal notation to the underlying probabilistic intuition. The notation is clear but a brief intuitive bridge aids comprehension.

---

### 6. Move tangential related work to appendix or compress

**Location:** Section 8.5 ("Abduction, Diagnosis, and Exact Explanatory Cores") and Section 8.6 ("Causality, Responsibility, and Structural Explanation").

**Change:** 
- Compress these subsections into 1-2 paragraphs each, or move them to an appendix
- Keep the key connection points (e.g., "The kinship with anchor sufficiency is especially relevant for the paper's anchor query") but remove detailed surveys
- If moving to appendix, add a brief forward-reference in Section 8.1 or 8.2

**Draft compression (for Section 8.5):**
> **Abduction and diagnosis.** The paper's minimum and anchor queries share structural kinship with abduction and diagnosis frameworks, which study explanatory hypotheses sufficient for a target consequence. The anchor query retains an outer existential choice (analogous to explanatory discovery) while the inner verification condition preserves the optimizer structure. This logical-complexity intuition motivates the Σ₂ᵖ classification of static anchor sufficiency.

**Rationale:** These subsections are tangential to the main contribution; compressing them streamlines the narrative without losing the key connection points. The paper is already long.

---

## Priority 3 (Optional)

### 7. Add a concluding sentence to Section 6 (Regime Hierarchy)

**Location:** End of Section 6 ("Regime Hierarchy").

**Change:** Add 1-2 sentences suggesting how the hierarchy might extend to other regimes.

**Draft addition:**
> **Forward-looking perspective.** The regime hierarchy presented here naturally extends to richer models: partially observable settings, multi-agent decision problems, and game-theoretic regimes would introduce additional layers of complexity (e.g., knowledge reasoning, equilibrium computation). The optimizer-quotient framework provides a unifying language for tracking exact decision-relevance across these extended regimes, though their detailed classification remains open.

**Rationale:** Provides forward-looking perspective without overcommitting to new technical results. Helps position the paper as a foundation for broader research program.

---

### 8. Add a worked example for stochastic regime

**Location:** Section 4.2 ("Stochastic Regime Complexity"), after the stochastic problem definitions (around line 50).

**Change:** Add a small 2-3 state numeric example illustrating preservation vs. decisiveness, similar to the worked POMDP example in Section 5.1.

**Draft addition:**
> **Example.** Consider a stochastic decision problem with $S=\{s_1,s_2\}$, $A=\{a,b\}$, uniform distribution $P(s_1)=P(s_2)=1/2$, and utilities $U(a,s_1)=2, U(b,s_1)=1$, $U(a,s_2)=0, U(b,s_2)=3$. For $I=\emptyset$: (i) **Preservation:** $\mathbb{E}[U(a,S)]=1$, $\mathbb{E}[U(b,S)]=2$, so $\Opt^{\text{stoch}}_\emptyset=\{b\}$; but $\Opt(s_1)=\{a\}$, so preservation fails. (ii) **Decisiveness:** The empty fiber itself has a unique conditional optimum $\{b\}$, so $I=\emptyset$ IS decisive. This shows decisiveness is strictly stronger than preservation.

**Rationale:** Grounds the formal definitions in a concrete setting; helps readers see why preservation and decisiveness are different predicates.

---

## Summary

**Total changes:** 8 items across 3 priority levels

**Estimated effort:**
- Priority 1: ~2-3 hours (substantive changes requiring careful integration)
- Priority 2: ~1-2 hours (mostly editorial tightening)
- Priority 3: ~1 hour (optional enhancements)

**Critical path for acceptance:** Items 1-3 (Priority 1) are required based on JACM reviewer feedback. Items 4-6 (Priority 2) are strongly recommended to improve readability and impact. Items 7-8 (Priority 3) are optional polish.

**Files affected:**
- `01_introduction.tex` (item 1)
- `04_stochastic_regime_complexity.tex` (items 2, 3, 5, 8)
- `04_tractable_special_cases.tex` (item 4)
- `08_related_work.tex` (item 6)
- `06_regime_hierarchy.tex` (item 7)

**Next steps:**
1. Implement Priority 1 changes first
2. Run LaTeX build to ensure no new compilation errors
3. Implement Priority 2 changes
4. Review overall paper flow
5. Optionally implement Priority 3 changes
6. Final proofread and submission preparation
