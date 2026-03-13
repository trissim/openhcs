# JACM Reviewer Assessment: Exact Relevance Certification and the Optimizer Quotient

## Overall Verdict

**Recommendation: Accept with Minor Revisions**

This is a substantial and well-executed theoretical paper that makes a coherent technical contribution across multiple computational regimes. The work is rigorous, well-organized, and addresses a genuinely interesting question at the intersection of decision theory, complexity theory, and mechanized verification. The optimizer quotient provides a clean unifying abstraction, and the regime-sensitive complexity classification is both technically nontrivial and well-motivated by applications in configuration simplification, POMDP abstraction, and hyperparameter pruning.

The paper is particularly strong in its:
- **Structural clarity:** The optimizer quotient as a canonical abstraction is introduced early and used consistently throughout
- **Technical depth:** The static/stochastic/sequential complexity results are complete and correctly situated relative to known literature
- **Mechanization:** 8,000 lines of Lean 4 verification provide unusual empirical weight to the theoretical claims
- **Application connections:** The engineering corollaries (configuration minimization, POMDP abstraction, hyperparameter redundancy) are concrete and well-explained

The main weaknesses are:
- **Presentation pacing:** Some sections feel slightly over-developed (especially tractable subcases and related work survey)
- **Open issue framing:** The stochastic preservation gap is now properly marked, but could be further sharpened with more explicit reduction intuition (conjecture 11.1 is a step in the right direction)
- **Related work integration:** The rough-set comparison is now more explicit about the structural divergence, but this could have been foregrounded earlier

Overall, the contribution is publishable at JACM level. The minor issues below are mostly about tightening exposition and a few technical clarifications.

---

## Strengths

### 1. Clean Unifying Abstraction

The optimizer quotient ($Q_\mathcal{D} = S/{\sim_\Opt}$) is introduced as the coarsest abstraction that preserves optimal-action distinctions. The universal property (Theorem 2.10) is elegantly proved and used repeatedly to connect static sufficiency, stochastic preservation, and the induced decision table. This gives the paper a strong conceptual backbone rather than a laundry list of unrelated complexity results.

The formalization is clean:
- Definitions 2.6–2.8 set up sufficiency, relevance, and the quotient precisely
- Proposition 2.9 (minimal sufficient sets = relevant-coordinate sets) is the linchpin of the static regime
- The quotient as coimage/image factorization (Proposition 2.11) connects to universal-algebraic intuition

### 2. Regime-Sensitive Complexity Landscape

The static/stochastic/sequential split is not arbitrary—it is motivated by how probability and temporal structure change the certification question:

- **Static:** Sufficiency reduces to counterexample exclusion, collapses to relevance containment (coNP)
- **Stochastic:** Separates into preservation (optimizer-preserving) and decisiveness (uniqueness of conditional optimum), with different complexity classes (P vs. PP/$\mathsf{NP}^{\mathsf{PP}}$)
- **Sequential:** Temporal contingency lifts to PSPACE; the TQBF reduction is well-motivated and cleanly explained

The hierarchical summary (Section 6) and Table 2 provide a clear reference point. The preservation/decisiveness split in the stochastic regime is particularly insightful—it explains why the same decision problem admits two very different complexity-theoretic flavors.

### 3. Encoding-Sensitive Contrast and ETH Lower Bounds

Section 3 (dichotomy) provides an important complement to the class-level results. The contrast between explicit-state tractability (when relevant support is $O(\log N)$) and ETH-conditioned exponential lower bounds (when relevant support is $\Omega(n)$) is both technically sound and practically relevant. This explains why exact certification can sometimes be easy despite coNP hardness—because the instance representation matters.

The ETH transfer chain (3-SAT → TAUTOLOGY → SUFFICIENCY-CHECK) is standard but cleanly executed.

### 4. Mechanized Verification at Serious Scale

8,000 lines of Lean 4 with \LeanReleaseSorry = 0 across the cited modules is an impressive artifact. The mechanization is not cosmetic—it verifies:

- Optimizer quotient universal property and uniqueness (handles QT1, QT7)
- Main reduction-correctness lemmas for all regimes
- Finite deciders and counted search procedures
- Bridge theorems (full-support equivalence, quotient equivalence)
- Witness/checking duality

The paper wisely does not claim full oracle-machine formalization (which would be unrealistic), but instead positions this as "problem-specific certified reduction infrastructure." That framing is honest and appropriate.

### 5. Structural Tractability Coverage

Twelve tractable subcases (Section 4) is comprehensive and not merely a list of exceptions. Each one is justified by identifying a specific hardness source that it removes:

- Bounded actions → removes unrestricted action comparison
- Separable/low-tensor-rank → removes cross-coordinate interaction
- Tree structure/bounded treewidth → removes high-width dependence
- Coordinate symmetry → removes redundant state comparisons

This shows that the theory has explanatory power for when exact certification is feasible, not just when it is hard.

### 6. Engineering Corollaries

Section 5 provides concrete motivation for why the theory matters:

- Configuration simplification (Corollary 5.3) shows that no general-purpose exact minimizer exists unless $\P = \coNP$
- Over-specification rationality (Proposition 5.6) explains when carrying extra parameters is cost-justified
- POMDP and hyperparameter reduction examples (Propositions 5.4, 5.6) connect to control and ML practice

The "simplicity tax" framing (Section 5.4) is a nice conceptual addition—it captures the intuitive idea that heuristic simplification doesn't remove decision-relevant burden, it just displaces it.

---

## Weaknesses / Areas for Improvement

### 1. Related Work Integration (Partially Addressed)

The rough-set comparison (Section 8.2) now explicitly states:
- Known NP-hard/$\Sigma_2^P$ bounds for general reduct computation
- Structural divergence: general decision tables admit multiple incomparable minimal reducts, whereas your framework forces a unique minimal sufficient set (the relevant-coordinate set)
- Novelty claim: "this specific structural collapse, from $\Sigma_2^P$ search to coNP relevance containment driven by optimizer-quotient uniqueness, is a novel classification not captured by general rough-set bounds"

This is now much clearer and should satisfy the reviewer feedback. However, this structural insight could have been foregrounded earlier in the paper—perhaps even in Section 1 or Section 2—to help the reader understand why the static collapse is surprising.

### 2. Stochastic Preservation Open Problem

Conjecture 11.1 and Open Problem 11.2 now properly frame the gap:
- Explicit claim that stochastic preservation under general distributions (succinct encoding) is coNP-hard
- Structured justification: (i) known results under full support, (ii) structural obstacle (zero-probability fibers), (iii) why coNP rather than PP (universal quantifier shape matches static regime)
- References to mechanized support (handles OU12, DC96, DC98)

This is a substantial improvement over the previous version. However, the conjecture would be even stronger if it included a brief reduction sketch—even an informal one—showing how one might attempt to encode a static counterexample into the distribution support to preserve the coNP-hardness of the static regime under the stochastic view. Currently, the structural justification is solid but does not give the reader a concrete sense of what a reduction might look like.

Recommendation: Add a 3–4 sentence paragraph or footnote sketching a candidate reduction idea (e.g., "a reduction from TAUTOLOGY would embed static counterexample pairs into zero-probability witness states of the distribution, exploiting that the preservation query quantifies universally over all states"). This does not need to be a full proof, but would help the reader see why the conjecture is plausible.

### 3. Pacing and Length

Some sections feel slightly over-developed:
- Section 4 (tractable subcases) is comprehensive but could be tightened—some subcases are straightforward (separable utility, constant optimal set) and could be merged or referenced more briefly
- Section 8 (related work) is thorough but covers a lot of ground; some subsections (e.g., abduction/diagnosis, causality) are tangential and could be moved to an appendix or referenced in a sentence

This is not a fatal flaw, but the paper is already long, and tightening could improve readability without sacrificing technical content.

### 4. Oracle-Class Membership Proofs

The oracle-class membership proofs ($\textsf{NP}^{\mathsf{PP}}$ for stochastic anchor/minimum decisiveness) are given in the text but not mechanized. The paper acknowledges this explicitly and correctly positions the artifact as verifying the "finite combinatorial core." This is a defensible position, but a JACM reviewer might ask for a bit more precision on what exactly is mechanized vs. what is argued informally.

Recommendation: In the mechanization note (Section 4, lines around 226–233), add one sentence clarifying that the artifact verifies:
- The existential/universal quantifier structure of the predicate
- The witness-checking schema (e.g., "exists anchor action + PP oracle verifies uniqueness on fiber")
- The finite-step-counted search wrappers (handles OU1–OU2, OU6–OU7)

This makes the division between mechanized and informal clearer.

### 5. Minor Technical / Notational Issues

- In the stochastic regime definition (Section 4.1), the distinction between $\Opt^\text{stoch}_I(\alpha)$ and the full-information optimizer $\Opt(s)$ could be sharpened in the surrounding exposition. The notation is clear, but a brief paragraph after Definition 4.2 explaining the intuition ("$\Opt^\text{stoch}_I$ averages over the fiber, while $\Opt(s)$ conditions on the exact state") would help readers new to stochastic decision theory.
- In the sequential regime (Section 5.1), the relationship between the "state-based certification predicate" and the "policy-level interpretation" is explained in paragraph 23–26, but could be moved earlier or given more emphasis. The paper uses the state-based predicate throughout, which is defensible, but the policy intuition is important for motivation.

---

## Technical Correctness

I have checked the main technical claims and they appear sound:
- Theorem 3.3 (dichotomy): The ETH transfer is standard
- Theorems 3.4–3.7 (static regime): coNP and $\Sigma_2^P$ classifications are standard given the relevance characterization
- Theorem 4.19 (stochastic PP-hardness): MAJSAT reduction is correct
- Theorem 5.3 (sequential PSPACE-completeness): TQBF reduction is correct and well-motivated
- Tractable subcases (Section 4): Reductions and complexity bounds are standard

The mechanized artifact (8,000 lines, 0 \texttt{sorry}) gives substantial empirical confidence that the core finite combinatorial arguments are correct.

---

## Suggestions for Minor Revisions

### Priority 1 (Required for Acceptance)

1. **Foreground the structural divergence earlier** (Section 8.2): Move or summarize the key insight about unique minimal sufficient sets vs. multiple incomparable reducts into Section 1 (Contributions) or Section 3 (Why You Might Expect $\Sigma_2^P$). This helps the reader understand the surprise of the static collapse earlier in the narrative.

2. **Add a reduction sketch for Conjecture 11.1**: A 3–4 sentence paragraph or footnote sketching how a TAUTOLOGY reduction might embed static counterexample pairs into zero-probability witness states. This strengthens the conjecture and gives the reader a concrete intuition.

3. **Clarify mechanization scope for oracle classes** (Section 4): Add one sentence specifying exactly what the artifact verifies about the $\textsf{NP}^{\mathsf{PP}}$ membership (quantifier structure + witness/checking schema + finite-step-counted wrappers).

### Priority 2 (Strongly Recommended)

4. **Tighten Section 4**: Merge or abbreviate the most straightforward tractable subcases (separable utility, constant optimal set, single action) and focus on the structurally interesting ones (low tensor rank, tree structure, bounded treewidth, coordinate symmetry).

5. **Add early intuition for stochastic optimizer notation** (Section 4.1): A brief paragraph after Definition 4.2 explaining the difference between $\Opt^\text{stoch}_I(\alpha)$ and $\Opt(s)$.

6. **Move tangential related work**: Some subsections of Section 8 (abduction/diagnosis, causality) could be compressed or moved to an appendix to streamline the main narrative.

### Priority 3 (Optional)

7. **Add a concluding sentence to Section 6 (Regime Hierarchy)**: A brief statement about how the hierarchy might extend to other regimes (e.g., partially observable, multi-agent, or game-theoretic settings) could give forward-looking perspective without overcommitting.

8. **Add a worked example** for the stochastic regime (similar to the worked numeric example in Section 5.1): A small 2–3 state stochastic instance showing preservation vs. decisiveness would help ground the formal definitions.

---

## Conclusion

This is a strong theoretical paper with a clean conceptual framework, comprehensive complexity classification, and substantial mechanization. The main contribution—unifying exact relevance certification around the optimizer quotient and tracking its cost across static, stochastic, and sequential regimes—is publishable at JACM level.

The rough-set structural divergence is now explicitly claimed, and the stochastic preservation open problem is framed as a conjecture with structured justification. With the minor revisions suggested above (especially Priority 1 items 1–3), the paper should meet JACM standards for acceptance.

The mechanized artifact adds unusual empirical weight to the theoretical claims, and the engineering corollaries provide concrete motivation for why the theory matters beyond pure complexity theory. This is a well-executed piece of theoretical computer science that should be of interest to the JACM readership in complexity theory, decision theory, and formal verification.

---

**Summary:** Accept with Minor Revisions. Address Priority 1 items (1–3) for acceptance; Priority 2 items (4–6) are strongly recommended to improve readability and impact. Priority 3 items (7–8) are optional.
