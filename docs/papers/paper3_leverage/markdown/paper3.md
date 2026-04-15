# Paper: Structural Rank, Decision Entropy, and Thermodynamic Selection in Finite Information-Processing Systems

**Status**: Draft-ready | **Lean**: 75987 lines, 3303 theorems

---

## Abstract

Exact resolution in finite bounded physical systems carries irreducible thermodynamic cost. Under Landauer calibration, any exact-resolution cycle satisfies $$E \geq k_B T\, H_{\mathrm{nats}}(D),$$ where $H_{\mathrm{nats}}(D)$ is the natural-log entropy of the decision quotient. In the canonical binary encoding studied here, this bound sharpens to $$E \geq \mathrm{DOF}(A)\, k_B T \ln 2,$$ and the rank-$1$ regime is the unique thermodynamic ground state: every system with more than one degree of freedom lies strictly above the minimum per-cycle resolution cost.

The England replication inequality, $$\Delta S_{\min}(k) - \Delta S_{\min}(1) \geq k_B \ln k,$$ is also obtained in the same framework. The proof reduces the entropy premium of replication to finite counting: a $k$-coordinate system has $2^k$ states, and the elementary inequality $k \leq 2^{k-1}$ yields the gap.

These thermodynamic statements are derived from a finite structural theorem. Exact resolution in a bounded system occurs through finitely many discrete acquisition events. The associated canonical decision problem records those events as Boolean coordinate reads, and the number of independent coordinates is identified exactly with the structural rank of the encoded decision problem: $$\mathrm{DOF}(A) = \mathrm{srank}(\mathrm{canonicalDP}(A)).$$ The decision quotient therefore has at most $2^{\mathrm{DOF}(A)}$ optimal-action classes, so its entropy is controlled by the same coordinate count that governs exact physical resolution.

All theorems are machine-checked in Lean 4 with no `sorry` placeholders. The mechanized artifact records the proof provenance in full.

Keywords: thermodynamics, Landauer principle, decision entropy, structural rank, bounded physical systems, formal verification


_Failed to convert lean_stats.tex_

# Introduction

Finite bounded physical systems resolve decisions through finitely many discrete acquisition events on bounded horizons. Under Landauer calibration [@landauer1961irreversibility; @bennett1982thermodynamics], exact resolution therefore carries irreducible thermodynamic cost. The formal object is a bounded decision system, represented in Lean by `Architecture`, carrying a positive degree-of-freedom count together with a canonical binary decision encoding `canonicalDP`. This counting parameter is exactly the structural rank of the encoded decision problem, it bounds the entropy of the decision quotient, and it determines the minimum per-cycle resolution cost. The quotient object is closer to zero-error and confusability-based information than to average-case coding [@shannon1956zero; @korner1973graphs; @lovasz1979shannon]. All claims are verified in Lean 4 [@moura2021lean4] with Mathlib support [@mathlib2020] and zero `sorry` placeholders.

## Central Result

The convergence theorem proved in Section [\[five-way-equivalence\]](#five-way-equivalence){reference-type="ref" reference="five-way-equivalence"} is:

$$\begin{aligned}
&\underbrace{\mathrm{DOF}(A) = 1}_{\text{one-coordinate regime}}
\;\iff\;
\underbrace{\mathrm{srank}(\mathrm{canonicalDP}(A)) = 1}_{\text{structural rank}} \\
&\iff\;
\underbrace{\text{tractable sufficiency for } \mathrm{canonicalDP}(A)}_{\text{complexity}}
\;\iff\;
\underbrace{\text{minimum per-cycle thermodynamic cost}}_{\text{physics}}
\end{aligned}$$

An imported coherence theorem gives a separate single-source reading of the same rank-$1$ point: $$\mathrm{SSOT}(A) \iff \mathrm{DOF}(A)=1,$$ where $\mathrm{SSOT}(A)$ denotes the coherent single-source condition that one locus is authoritative, every remaining encoding is a derived view, and all reachable states remain coherent.

The structural-rank and thermodynamic clauses are central. The imported coherence statement is a companion interpretation of the same rank-$1$ regime, not a premise of the mathematical-physics chain.

## Contributions

1.  **Finite Physical Acquisition:** bounded regions admit finitely many acquisition events, those events are discrete state transitions, and exact resolution requires a sufficient coordinate set.

2.  **DOF-Structural-Rank Identification:** the canonical decision problem attached to an $n$-degree-of-freedom system has structural rank $n$.

3.  **Entropy-Rank Control:** the number of decision classes is at most $2^{\mathrm{DOF}(A)}$, so the decision entropy is bounded by the degree-of-freedom count.

4.  **Landauer-Linear Resolution Cost:** under Landauer calibration, exact-resolution cost is bounded below by $\mathrm{DOF}(A)\,k_B T \ln 2$.

5.  **Energy-Information Duality:** the same system satisfies $E \geq k_B T\,H_{\mathrm{nats}}(D)$, linking thermodynamic cost directly to the entropy of the decision quotient.

6.  **Finite-Budget No-Collapse:** bounded budget, positive per-bit cost, and exponential lower-bound growth jointly preclude physical collapse of the higher-rank regime.

7.  **England Replication Inequality (Theorem [\[thm:england\]](#thm:england){reference-type="ref" reference="thm:england"}):** $\Delta S_{\min}(k) - \Delta S_{\min}(1) \geq k_B \ln k$. The proof reduces the entropy gap to counting via $k \leq 2^{k-1}$.

8.  **Rank-1 Convergence (Theorem [\[thm:five-way\]](#thm:five-way){reference-type="ref" reference="thm:five-way"}):** the rank-$1$ regime is simultaneously the one-coordinate regime, the tractable sufficiency regime, and the thermodynamic ground state.

#### Physical significance.

Once finite acquisition is fixed as the physical interface, the degree-of-freedom count becomes simultaneously an interaction dimension, an entropy bound, and a Landauer-calibrated cost coordinate. The result concerns exact resolution structure in matter.

## Scope

The mathematical structure links structural rank, quotient entropy, complexity, and thermodynamic cost. Theorems are stated for bounded decision systems represented by the Lean object `Architecture` and their canonical decision encoding, not for arbitrary physical systems without mediation. Architectural and programming interpretations are downstream readings of the same formal core.

## Organization

Section [\[foundations\]](#foundations){reference-type="ref" reference="foundations"} defines the structural model, the finite-acquisition interface, and the canonical decision encoding. Section [\[probability-model\]](#probability-model){reference-type="ref" reference="probability-model"} derives the structural-rank and entropy consequences of that encoding. Section [\[main-theorems\]](#main-theorems){reference-type="ref" reference="main-theorems"} derives the thermodynamic consequences. Section [\[five-way-equivalence\]](#five-way-equivalence){reference-type="ref" reference="five-way-equivalence"} proves the convergence theorem and the England inequality. Section [\[related-work\]](#related-work){reference-type="ref" reference="related-work"} situates the paper relative to thermodynamics, information theory, and formalized complexity. Section [\[appendix-lean\]](#appendix-lean){reference-type="ref" reference="appendix-lean"} describes the Lean mechanization.


# Foundations

The formal objects that carry the mathematical-physics content are a positive degree-of-freedom count, a canonical binary decision encoding, structural rank, and decision entropy.

## Formal Object

::: definition
[]{#def:architecture label="def:architecture"} A *bounded decision system* is a finite bounded physical system equipped with a positive integer $\mathrm{DOF}(A)$. In the mechanized artifact the corresponding Lean object is named `Architecture`, but the results used below depend only on the degree-of-freedom count and on the canonical decision encoding.
:::

**Interpretation.** $\mathrm{DOF}(A)$ counts independent coordinates that can vary separately. The rest of the paper studies what that count forces once one asks for exact resolution.

## Degrees of Freedom

::: definition
[]{#def:dof label="def:dof"} The quantity $\mathrm{DOF}(A) \in \mathbb{N}$ counts independent coordinates of variation in a bounded decision system $A$. In the mechanized development it is the local structural parameter attached to `Architecture`; later sections identify it exactly with the structural rank of a canonical decision problem.
:::

**Operational meaning.** If $\mathrm{DOF}(A)=n$, the system has $n$ independent coordinates that must be resolved in the worst case by any exact resolver.

::: proposition
[]{#prop:dof-additive label="prop:dof-additive"} For disjoint bounded decision systems $A_1$ and $A_2$: $$\mathrm{DOF}(A_1 \oplus A_2) = \mathrm{DOF}(A_1) + \mathrm{DOF}(A_2)$$
:::

::: proof
*Proof.* Independent coordinate sets combine by disjoint union, so the coordinate count is additive under composition. ◻
:::

## Finite Physical Acquisition

::: theorem
[]{#thm:counting-gap label="thm:counting-gap"} Let $\varepsilon, C \in \mathbb{N}$ with $\varepsilon>0$ and $C>0$. If each information-acquisition event consumes $\varepsilon$ discrete cost units, then $$\varepsilon \cdot N \le C \implies N \le C.$$ Equivalently, any bounded system with positive per-event cost admits only finitely many acquisition events.
:::

::: proof
*Proof.* In $\mathbb{N}$ every positive integer is at least one, so $N = 1\cdot N \le \varepsilon N \le C$. ◻
:::

Bounded capacity plus positive per-event cost already forbids infinite checking. Physics enters when the abstract cost unit is calibrated. Under Landauer-type calibration, exact irreversible acquisition carries positive cost, so bounded material systems inherit the counting gap directly.

::: proposition
[]{#prop:bounded-region label="prop:bounded-region"} A bounded physical region is characterized by diameter $d>0$ and signal speed $c>0$. Its maximum information-acquisition rate is $c/d$ events per unit time.
:::

::: theorem
[]{#thm:bounded-acquisition label="thm:bounded-acquisition"} For a bounded region with diameter $d$, signal speed $c$, and operating time $T$, $$\mathrm{acquisitions}(T) \le \frac{cT}{d}.$$ In particular, acquisition count is finite on finite horizons.
:::

::: proof
*Proof.* Signals require at least $d/c$ time to traverse the region, so no more than $c/d$ acquisition events can occur per unit time. ◻
:::

::: theorem
[]{#thm:discrete-acquisition label="thm:discrete-acquisition"} Information acquisition is realized by discrete state transitions. A bounded physical decision process is therefore a finite transition system whose acquisition events are countable transition points.
:::

::: theorem
[]{#thm:one-transition-one-bit label="thm:one-transition-one-bit"} Each elementary acquisition transition carries one boolean bit. Boolean coordinates are therefore the primitive units of exact physical information exchange in the canonical model.
:::

Together, Theorems [\[thm:bounded-acquisition\]](#thm:bounded-acquisition){reference-type="ref" reference="thm:bounded-acquisition"}, [\[thm:discrete-acquisition\]](#thm:discrete-acquisition){reference-type="ref" reference="thm:discrete-acquisition"}, and [\[thm:one-transition-one-bit\]](#thm:one-transition-one-bit){reference-type="ref" reference="thm:one-transition-one-bit"} identify the later binary decision encoding as the natural finite acquisition model. A bounded resolver acquires information through finitely many discrete events, and each such event contributes one elementary boolean distinction.

::: theorem
[]{#thm:resolution-sufficient label="thm:resolution-sufficient"} Any exact physical resolver for a decision problem must read a sufficient coordinate set. If fewer coordinates are read, there exist states indistinguishable to the resolver but requiring different optimal actions.
:::

::: proof
*Proof.* If the accessed coordinates are not sufficient, two states agree on every read coordinate while disagreeing on the optimal action. Any resolver limited to those coordinates must therefore err on at least one of the two states. ◻
:::

## Canonical Decision Encoding

::: definition
[]{#def:canonical-dp label="def:canonical-dp"} For a bounded decision system $A$ with $\mathrm{DOF}(A)=n$, the canonical decision problem $$\mathrm{canonicalDP}(A)$$ has state space $\mathrm{Fin}\;n \to \mathrm{Bool}$ and action space $\mathrm{Fin}\;n \oplus \mathrm{Unit}$. Action $\mathrm{inl}(i)$ queries coordinate $i$; the fallback action $\mathrm{inr}(\star)$ receives constant utility $1$. Query action $i$ receives utility $2$ exactly when coordinate $i$ is true and $0$ otherwise.
:::

The encoding is the exact Lean object `canonicalDP` in `Leverage/BridgeToDQ.lean`. Every coordinate is relevant by construction, so the encoding exposes the full interaction dimension of the source system.

The encoding records, in the smallest exact-resolution object, the coordinate structure that any bounded physical resolver must already confront. The Boolean coordinates represent primitive acquisition events.

## Structural Rank

::: definition
[]{#def:srank label="def:srank"} The *structural rank* of a finite decision problem is the cardinality of its relevant coordinate set, equivalently the size of any minimal sufficient set. It is the minimum interaction dimension that must be read to determine the optimal action exactly.
:::

For the canonical decision problem attached to $A$, the relevant coordinate set is all of $\mathrm{Fin}\;\mathrm{DOF}(A)$, so the structural-rank problem is exactly matched to the local degree-of-freedom count.

## Decision Quotient and Entropy

::: definition
[]{#def:decision-quotient label="def:decision-quotient"} For a decision problem $D$, states are identified when they induce the same optimal-action set. The quotient space of these equivalence classes is the *decision quotient* of $D$.
:::

::: proposition
[]{#prop:optimizer-quotient label="prop:optimizer-quotient"} Let $\operatorname{Opt}: S \to \mathcal P(A)$ be the optimizer map of a finite decision problem. In **Set**, the decision quotient is the coimage of $\operatorname{Opt}$, canonically equivalent to $\operatorname{im}(\operatorname{Opt})$ [@maclane1998categories]. Any surjective decision-preserving summary factors through this quotient.
:::

The quotient is the coarsest exact abstraction of the decision problem: it forgets only decision-irrelevant distinctions and preserves every distinction needed for exact action selection. The entropy and thermodynamic bounds below are stated for this canonical exact abstraction.

::: definition
[]{#def:decision-entropy label="def:decision-entropy"} Let $\mathrm{numOptClasses}(D)$ be the number of equivalence classes in the decision quotient. Two entropy normalizations are used: $$H_{\mathrm{bits}}(D) = \log_2 \mathrm{numOptClasses}(D),
\qquad
H_{\mathrm{nats}}(D) = \log \mathrm{numOptClasses}(D).$$
:::

The physics results are naturally stated in nats because Landauer calibration contributes the factor $k_B T$ per nat of resolved decision information [@landauer1961irreversibility; @bennett1982thermodynamics].

## Formalization in Lean

The local degree-of-freedom object lives in `Leverage/Foundations.lean`, while the canonical decision encoding and its rank-identification theorems live in `Leverage/BridgeToDQ.lean`. Structural rank and decision entropy are formalized in the decision-quotient development.


# Structural Rank and Decision Entropy {#probability-model}

The canonical decision encoding has two immediate consequences. The degree-of-freedom count is exactly the interaction dimension of the encoded decision problem, and exact physical resolution must pay for that interaction dimension in discrete bit events. Structural rank is the physical count of irreducible coordinate reads required by exact resolution.

## Degree of Freedom Equals Structural Rank

::: theorem
[]{#thm:dof-srank label="thm:dof-srank"} For every bounded decision system $A$, $$\mathrm{srank}(\mathrm{canonicalDP}(A)) = \mathrm{DOF}(A).$$
:::

::: proof
*Proof.* Write $n = \mathrm{DOF}(A)$. By Definition [\[def:canonical-dp\]](#def:canonical-dp){reference-type="ref" reference="def:canonical-dp"}, the state space is $\mathrm{Fin}\;n \to \mathrm{Bool}$, query action $\mathrm{inl}(i)$ has utility $2$ exactly when coordinate $i$ is true and utility $0$ otherwise, and the fallback action has utility $1$. Fix any coordinate $i$ and choose two states that agree everywhere except at $i$, with one state setting $i$ to true and the other setting $i$ to false; then $\mathrm{inl}(i)$ is optimal in the first state and not optimal in the second, so erasing coordinate $i$ changes the optimizer. Thus every coordinate in $\mathrm{Fin}\;n$ is relevant, the relevant-coordinate set has cardinality $n$, and the structural rank is $n$. Substituting $n = \mathrm{DOF}(A)$ gives the claim. ◻
:::

::: corollary
[]{#cor:rank-one label="cor:rank-one"} For every bounded decision system $A$, $$\mathrm{DOF}(A)=1 \iff \mathrm{srank}(\mathrm{canonicalDP}(A))=1.$$
:::

::: corollary
[]{#cor:rank-above-one label="cor:rank-above-one"} For every bounded decision system $A$, $$\mathrm{DOF}(A)>1 \implies \mathrm{srank}(\mathrm{canonicalDP}(A))>1.$$
:::

::: theorem
[]{#thm:min-bit-operations label="thm:min-bit-operations"} Any exact resolver for $\mathrm{canonicalDP}(A)$ requires at least $\mathrm{DOF}(A)$ elementary bit-acquisition events.
:::

::: proof
*Proof.* By Theorem [\[thm:resolution-sufficient\]](#thm:resolution-sufficient){reference-type="ref" reference="thm:resolution-sufficient"}, exact resolution requires reading a sufficient coordinate set. The structural-rank theorem implies every sufficient set has cardinality at least $\mathrm{srank}(\mathrm{canonicalDP}(A))$. Theorem [\[thm:dof-srank\]](#thm:dof-srank){reference-type="ref" reference="thm:dof-srank"} identifies that rank with $\mathrm{DOF}(A)$, so at least $\mathrm{DOF}(A)$ coordinate reads are required. By Theorem [\[thm:one-transition-one-bit\]](#thm:one-transition-one-bit){reference-type="ref" reference="thm:one-transition-one-bit"}, each read contributes one elementary bit-acquisition event. ◻
:::

## Decision-Quotient Size

::: theorem
[]{#thm:numopt-bound label="thm:numopt-bound"} For the canonical binary decision problem attached to a bounded decision system $A$, $$\mathrm{numOptClasses}(\mathrm{canonicalDP}(A)) \le 2^{\mathrm{DOF}(A)}.$$
:::

::: proof
*Proof.* For binary coordinate spaces, the number of distinct optimal-action classes is at most $2^{\mathrm{srank}}$. Apply that theorem to the canonical encoding and substitute [\[thm:dof-srank\]](#thm:dof-srank){reference-type="ref" reference="thm:dof-srank"}. ◻
:::

::: theorem
[]{#thm:entropy-bound label="thm:entropy-bound"} For the canonical binary decision problem attached to a bounded decision system $A$, $$H_{\mathrm{bits}}(\mathrm{canonicalDP}(A)) \le \mathrm{DOF}(A),
\qquad
H_{\mathrm{nats}}(\mathrm{canonicalDP}(A)) \le \mathrm{DOF}(A)\,\ln 2.$$
:::

::: proof
*Proof.* The bit-entropy statement is the entropy-rank inequality for binary coordinate spaces, again composed with Theorem [\[thm:dof-srank\]](#thm:dof-srank){reference-type="ref" reference="thm:dof-srank"}. The nat-entropy statement is obtained by multiplying by $\ln 2$. ◻
:::

## Formalization

The structural-rank bridge is formalized in `Leverage/BridgeToDQ.lean`; the minimum-bit and entropy bounds are formalized in the decision-quotient physics and information development. These are the objects used directly by the thermodynamic theorems of the next section.


# Thermodynamic Consequences {#main-theorems}

The previous section identified degree of freedom with structural rank and bounded the entropy of the decision quotient. Exact resolution requires irreducible bit events. Matter pays for them.

## Landauer-Linear Resolution Cost

::: theorem
[]{#thm:energy-rank label="thm:energy-rank"} Let $A$ be a bounded decision system and let $M$ be a thermodynamic model with positive per-bit conversion constant. Then $$M.\mathrm{joulesPerBit} \cdot \mathrm{DOF}(A)
\le
\mathrm{energyLowerBound}(M, \mathrm{DOF}(A)).$$ In particular, exact-resolution cost is at least linear in the degree-of-freedom count.
:::

::: proof
*Proof.* Theorem [\[thm:min-bit-operations\]](#thm:min-bit-operations){reference-type="ref" reference="thm:min-bit-operations"} gives a lower bound of $\mathrm{DOF}(A)$ elementary bit-acquisition events for exact resolution. The bounded-acquisition energy theorem then converts that bit lower bound into the displayed energy lower bound. ◻
:::

::: theorem
[]{#thm:rank-one-ground label="thm:rank-one-ground"} If $\mathrm{DOF}(A)=1$, then every exact-resolution cycle for the canonical problem has energy at least one Landauer unit. If $\mathrm{DOF}(A)>1$, then the system lies strictly above that ground state.
:::

::: proof
*Proof.* The rank-one statement is exactly the imported ground-state theorem for structural rank $1$ (BA8). For the higher-rank regime, Corollary [\[cor:rank-above-one\]](#cor:rank-above-one){reference-type="ref" reference="cor:rank-above-one"} gives $\mathrm{srank}>1$, and Theorem [\[thm:energy-rank\]](#thm:energy-rank){reference-type="ref" reference="thm:energy-rank"} then places the resulting exact-resolution cycle strictly above the one-Landauer-unit floor. ◻
:::

## Energy--Information Duality

::: theorem
[]{#thm:energy-entropy label="thm:energy-entropy"} Let $D = \mathrm{canonicalDP}(A)$, and let $E$ be the realized energy for one exact-resolution cycle. If Landauer calibration holds at positive Boltzmann constant and temperature, then $$E \ge k_B T\, H_{\mathrm{nats}}(D).$$ Equivalently, the minimum exact-resolution cost is at least $k_B T$ times the natural-log entropy of the decision quotient.
:::

::: proof
*Proof.* The entropy-rank inequality gives $$H_{\mathrm{nats}}(D) \le \mathrm{DOF}(A)\ln 2$$ by Theorem [\[thm:entropy-bound\]](#thm:entropy-bound){reference-type="ref" reference="thm:entropy-bound"}. Theorem [\[thm:energy-rank\]](#thm:energy-rank){reference-type="ref" reference="thm:energy-rank"} gives the complementary lower bound $$E \ge \mathrm{DOF}(A) k_B T \ln 2.$$ Comparing the two right-hand sides yields the announced inequality. In the mechanized artifact this is the energy-information theorem (EI1). ◻
:::

::: corollary
[]{#cor:minimum-cost-regime label="cor:minimum-cost-regime"} Among bounded decision systems in the canonical binary encoding, the unique minimum-cost regime is $\mathrm{DOF}(A)=1$.
:::

::: proof
*Proof.* Theorem [\[thm:rank-one-ground\]](#thm:rank-one-ground){reference-type="ref" reference="thm:rank-one-ground"} identifies $\mathrm{DOF}(A)=1$ as the one-Landauer-unit ground state, while every bounded decision system with more than one degree of freedom lies strictly above it. ◻
:::

## Worked Examples

Two toy canonical systems fix the scale of the bound.

#### One relevant coordinate.

Take states $\{0,1\}$ and actions $\{a_0,a_1\}$, with $\operatorname{Opt}(0)=\{a_0\}$ and $\operatorname{Opt}(1)=\{a_1\}$. Then $\mathrm{DOF}(A)=1$, the decision quotient has two classes, $H_{\mathrm{nats}}(D)=\ln 2$, and Theorem [\[thm:energy-entropy\]](#thm:energy-entropy){reference-type="ref" reference="thm:energy-entropy"} gives $$E \ge k_B T \ln 2.$$ This is the rank-$1$ ground regime.

#### Two independent coordinates.

Take states $\{00,01,10,11\}$ and actions $\{a_{00},a_{01},a_{10},a_{11}\}$, with each state having its matching optimal action. Then $\mathrm{DOF}(A)=2$, the decision quotient has four classes, $H_{\mathrm{nats}}(D)=\ln 4 = 2\ln 2$, and Theorem [\[thm:energy-entropy\]](#thm:energy-entropy){reference-type="ref" reference="thm:energy-entropy"} gives $$E \ge 2 k_B T \ln 2.$$ Relative to the one-coordinate case, the minimum exact-resolution cost doubles.

## Optimal-Transport Witness

The Landauer route lower-bounds exact resolution through irreversible bit acquisition. A complementary witness measures separation between future distributions on the integrity space $\{\mathrm{intact},\mathrm{compromised}\}$. It does not replace the main proof chain, but it supplies an independent transport-theoretic signal that multiple distinguishable futures have nonzero cost.

::: remark
[]{#rem:wasserstein-bridge label="rem:wasserstein-bridge"} The same separation admits an independent transport-cost witness on the two-state integrity space $\{\mathrm{intact},\mathrm{compromised}\}$. The diagonal coupling has zero transport cost in the single-future regime (W1). Any coupling with off-diagonal mass has positive transport cost (W2). If the intact future mass dominates the compromised future mass, the intact state minimizes total transport to the future distribution (W3). If both future states carry positive mass, then transport from either pure state is strictly positive (W4). Multiple distinguishable futures therefore force positive transport cost independently of the Landauer route.
:::

A transport witness and the Landauer witness emphasize different structures: one counts irreducible coordinate reads, while the other measures geometric separation of future mass. In the two-state integrity model they point in the same direction, since a single future has zero transport cost whereas genuinely split futures force strictly positive transport cost.

## Interpretation

If degree of freedom is read as the number of independent physical coordinates that can vary separately, then lower DOF means lower exact-resolution cost because fewer independent coordinates must be resolved.

## Formalization

The local bridge from degree of freedom to structural rank is formalized in `Leverage/BridgeToDQ.lean`. The physical acquisition and Landauer theorems are imported from the decision-quotient physics stack, in particular `Physics/BoundedAcquisition.lean` and `ThermodynamicLift.lean`. The role of the local `Architecture` object in this section is to provide the coordinate count transported into those theorems.


# Convergence of Rank, Tractability, and Cost {#five-way-equivalence}

Degree of freedom equals structural rank, and structural rank bounds decision entropy. The same rank-$1$ regime also appears as tractable sufficiency and minimum thermodynamic cost. An imported coherence theorem gives a separate single-source interpretation of that same point.

## Imported Coherence Reading

::: theorem
[]{#thm:coherent-single-source label="thm:coherent-single-source"} A bounded decision system lies in the coherent unit-independent-rate regime if and only if $\mathrm{DOF}(A)=1$.
:::

::: informal
In the imported coherence development, rank $1$ means exactly one locus is authoritative, every remaining encoding is a derived view, and all reachable states remain coherent.
:::

## Structural Rank

::: theorem
[]{#thm:rank-identification label="thm:rank-identification"} For every bounded decision system $A$, $$\mathrm{srank}(\mathrm{canonicalDP}(A)) = \mathrm{DOF}(A).$$
:::

::: informal
The degree-of-freedom count is not separate from the decision-theoretic object; it is exactly the interaction dimension of the canonical decision problem.
:::

## Tractability Boundary

::: theorem
[]{#thm:tractable-rank-one label="thm:tractable-rank-one"} In the canonical decision problem family, structural rank $1$ is the tractable sufficiency regime, while higher structural rank enters the hard regime.
:::

::: informal
When exactly one relevant coordinate survives, exact sufficiency certification is tractable. Once more than one relevant coordinate survives, the canonical family crosses into the imported hardness regime.
:::

## Thermodynamic Selection

::: theorem
[]{#thm:thermodynamic-selection label="thm:thermodynamic-selection"} In the canonical decision encoding, every bounded decision system with $\mathrm{DOF}(A)>1$ lies strictly above the rank-$1$ Landauer ground state in per-cycle resolution cost.
:::

::: informal
The theorem uses only the rank identity together with Landauer calibration. Stronger hardness consequences require additional imported hypotheses.
:::

## Convergence Theorem

::: theorem
[]{#thm:five-way label="thm:five-way"} For every bounded decision system $A$, the following conditions are equivalent: $$\mathrm{DOF}(A)=1
\iff \mathrm{srank}(\mathrm{canonicalDP}(A))=1
\iff \text{tractable sufficiency for } \mathrm{canonicalDP}(A)
\iff \text{minimum per-cycle thermodynamic cost}.$$
:::

::: informal
The same rank-$1$ regime is simultaneously the exact one-coordinate regime, the tractable exact-certification regime, and the thermodynamic ground state. The imported coherence theorem gives an additional single-source interpretation of the same point.
:::

**Proof.**

1.  Theorem [\[thm:rank-identification\]](#thm:rank-identification){reference-type="ref" reference="thm:rank-identification"} identifies $\mathrm{DOF}(A)=1$ with structural rank $1$.

2.  Theorem [\[thm:tractable-rank-one\]](#thm:tractable-rank-one){reference-type="ref" reference="thm:tractable-rank-one"} identifies structural rank $1$ with the tractable sufficiency regime for the canonical family.

3.  Theorem [\[thm:thermodynamic-selection\]](#thm:thermodynamic-selection){reference-type="ref" reference="thm:thermodynamic-selection"} identifies rank $1$ as the unique minimum-cost thermodynamic regime.

Transitivity of logical equivalence completes the proof.

Theorem [\[thm:coherent-single-source\]](#thm:coherent-single-source){reference-type="ref" reference="thm:coherent-single-source"} supplies an imported single-source interpretation of the same rank-$1$ regime.

## Formalization

The proof chain is explicit in the mechanized artifact. The local bridge theorems live in `Leverage/BridgeToDQ.lean`; the coherence theorem is imported from `Ssot`; and the tractability and Landauer-cost theorems are imported from the decision-quotient development. The proof provenance remains fully auditable.

## England Replication Inequality

The England Replication Inequality is mechanized in `Leverage/BridgeToDQ.lean` (L45).

::: theorem
[]{#thm:england label="thm:england"} Let $\Delta S_{\min}(r) = r \cdot k_B \ln 2$ be the rank-indexed minimal entropy production under Landauer calibration. For the rank-$1$ ground regime and any replicated rank-$k$ regime: $$\Delta S_{\min}(1) + k_B \ln k \leq \Delta S_{\min}(k)$$ equivalently, $\Delta S_{\min}(k) - \Delta S_{\min}(1) \geq k_B \ln k$.
:::

::: proof
*Proof.* The gap is $(k-1) \cdot k_B \ln 2$. Since $k \leq 2^{k-1}$ (L52), taking logs gives $\ln k \leq (k-1) \ln 2$, so the gap is $\geq k_B \ln k$. ◻
:::

**Modeling note.** $\Delta S_{\min}$ is a definition within the model: the exact Landauer entropy cost of the canonical exact-resolution cycle. The "min" refers to physical optimality inside the calibrated decision model. In England's 2013 stochastic-thermodynamic framework [@england2013statistical], the comparison point is the entropy premium associated with replication above a single-copy baseline. The present reformulation does not reproduce England's full path-space dynamics or detailed-balance setting. It isolates the same $k_B \ln k$ multiplicity penalty as a finite-counting consequence of rank-indexed Landauer cost together with $k \le 2^{k-1}$, making the selection penalty explicit in a finite exact-resolution model.

## Finite-Budget No-Collapse

::: theorem
[]{#thm:finite-budget-no-collapse label="thm:finite-budget-no-collapse"} Let $B : \mathbb{N} \to \mathbb{N}$ be a budget profile, let $\mathrm{ops} : \mathbb{N} \to \mathbb{N}$ be a required-operation profile, and let $\mathrm{bitCost} > 0$ be the per-bit physical cost. If

1.  $B$ is globally bounded,

2.  $\mathrm{ops}$ has an exponential lower bound, and

3.  collapse means: for every input size $n$, some feasible bit budget realizes at least $\mathrm{ops}(n)$ operations within budget $B(n)$,

then no such physical collapse profile exists.
:::

::: proof
*Proof.* This is exactly the bounded-budget physical no-collapse theorem in the physical-hardness layer. Exponential growth eventually exceeds every fixed finite budget, and positive per-bit cost lifts that growth into an energy contradiction. ◻
:::

::: remark
Finite budget, positive event cost, and exponential exact-certification demand are jointly incompatible with a physical collapse model. Any stronger complexity-collapse conclusion requires an additional bridge from the chosen complexity claim to such a collapse profile.
:::

::: corollary
[]{#cor:pnp-nogo label="cor:pnp-nogo"} Assume a polynomial-collapse claim supplies a bridge to the finite-budget collapse profile of Theorem [\[thm:finite-budget-no-collapse\]](#thm:finite-budget-no-collapse){reference-type="ref" reference="thm:finite-budget-no-collapse"}. Then that claim is physically impossible in the same model. In particular, any bridge from a $P = NP$ collapse claim to such a feasible physical-collapse profile yields a contradiction.
:::

::: proof
*Proof.* Theorem [\[thm:finite-budget-no-collapse\]](#thm:finite-budget-no-collapse){reference-type="ref" reference="thm:finite-budget-no-collapse"} rules out the collapse profile itself. The transfer theorem states that if a $P = NP$ claim implies that profile, then the claim is false in the model. ◻
:::


# Related Work

## Landauer, Non-Equilibrium Thermodynamics, and Selection

Landauer's principle gives the standard calibration from logically irreversible discrimination to minimum heat production and energy cost [@landauer1961irreversibility; @bennett1982thermodynamics]. Stochastic thermodynamics extends that floor to trajectory-level entropy production, work identities, and fluctuation relations [@seifert2012stochastic; @vandenbroeck2015ensemble; @wolpert2019stochastic; @jarzynski1997nonequilibrium; @crooks1999entropy]. Finite-time erasure and mismatch corrections sharpen the same theme for controlled nonequilibrium protocols [@diana2013finite; @proesmans2020finite; @manzano2024absolute]. The theorem chain above isolates a different object: a finite exact-resolution lower bound indexed by the number of independent coordinates that must be resolved to preserve the optimizer.

Relative to the Seifert and Van den Broeck--Esposito framework, the present model does not attempt a full trajectory description of a driven Markov process, housekeeping heat, or protocol-dependent dissipation. It gives instead a mechanized non-asymptotic lower bound in terms of structural rank and decision-quotient entropy. Conversely, stochastic-thermodynamic frameworks resolve time-dependent nonequilibrium refinements that are outside the current finite exact-resolution model.

Non-equilibrium selection and replication arguments lie in the same neighborhood [@england2013statistical]. The entropy premium used above is deliberately discrete: multiplicity is reduced to counting over a finite quotient family and the elementary inequality $k \le 2^{k-1}$, rather than to a separate stochastic-process ansatz.

## Zero-Error, Functional, and Quotient Information

The information object is closer to zero-error and confusability-based information theory than to average-case source coding [@shannon1956zero; @korner1973graphs; @lovasz1979shannon; @csiszar2011information]. The central quantity is not full state entropy but the entropy of the decision quotient: how many distinct optimal-action classes survive after irrelevant coordinates are erased.

Function-relative information in physics and origins-of-life work also conditions information on successful function or selection [@szostak2003functional; @wong2023roles]. The object studied here is narrower and exact: coordinate erasure is admissible precisely when optimal-action correspondence is preserved. The rank-$1$ regime is therefore the one-coordinate exact-decision regime, the tractable sufficiency regime, and the minimum calibrated-cost regime.

## Categorical Quotients and Exact Abstraction

Quotienting states by equality of $\operatorname{Opt}$ is the standard coimage construction for the decision quotient of the optimizer map $\operatorname{Opt}: S \to \mathcal{P}(A)$ in **Set**, canonically equivalent to its image [@maclane1998categories]. The novelty is not the existence of that quotient, but the theorem chain tying it to coordinate sufficiency, structural rank, decision entropy, and thermodynamic cost in one proof object.

## Formal Verification and Machine-Checked Theory

The proofs are machine-checked in Lean 4 [@moura2021lean4] against the Mathlib library [@mathlib2020]. Related mechanized precedents include verified computability and semantics developments in Coq and Isabelle [@forster2019verified; @nipkow2002isabelle; @nipkow2014concrete] and certificate-carrying proof artifacts [@necula1997proof]. Mechanization matters because the main identifications are exact rather than rhetorical: degree of freedom, structural rank, quotient entropy, tractable sufficiency, and calibrated thermodynamic cost are kept distinct and then linked by explicit theorems.


# Conclusion

## Summary

The central result is the convergence theorem (Theorem [\[thm:five-way\]](#thm:five-way){reference-type="ref" reference="thm:five-way"}): the rank-$1$ regime of the canonical decision encoding is simultaneously the one-coordinate regime, the tractable sufficiency regime, and the thermodynamic ground state. An imported coherence theorem gives a separate single-source reading of the same point.

::: center
  **Framework**              **Rank-$1$ means**                     **Formal source**
  -------------------------- -------------------------------------- -------------------------------------------
  Local system parameter     $\mathrm{DOF}(A)=1$                    `Leverage/Foundations`
  Structural information     Structural rank $= 1$                  `Leverage/BridgeToDQ`
  Computational complexity   Tractable sufficiency checking         `DecisionQuotient`
  Statistical physics        Minimum thermodynamic cost per cycle   `BoundedAcquisition`, `ThermodynamicLift`
:::

The theorem package is a mathematical-physics statement: DOF is identified with structural rank, structural rank bounds decision entropy, and Landauer calibration turns that rank bound into an energy bound. All of these statements are machine-checked in Lean 4 with an auditable proof trail.

**Main consequences:**

-   The exact identification of degree of freedom with structural rank in the canonical decision encoding.

-   The energy--information theorem $E \ge k_B T H_{\mathrm{nats}}(D)$ for exact-resolution cost.

-   The unconditional thermodynamic selection statement that every higher-rank regime lies above the rank-$1$ Landauer ground state.

-   The finite-budget no-collapse theorem: bounded budget, positive per-bit cost, and exponential lower-bound growth cannot coexist with physical collapse.

-   The England replication inequality (L45): $\Delta S_{\min}(k) - \Delta S_{\min}(1) \geq k_B \ln k$, proved by finite counting rather than by an informal thermodynamic analogy.

## Mechanized Status

The central thermodynamic and convergence statements are all theorem-level results in the mechanized artifact:

-   L55: unconditional energy separation above the rank-$1$ ground state.

-   L49: quantitative Landauer-linear energy bound.

-   PH26: bounded budget plus positive bit-cost plus exponential lower bound implies no physical collapse.

-   L45: $\Delta S_{\min}(k) - \Delta S_{\min}(1) \geq k_B \ln k$ (Theorem [\[thm:england\]](#thm:england){reference-type="ref" reference="thm:england"}).

## Limitations

**1. Canonical-encoding scope:** the main theorems are exact for the canonical binary decision encoding attached to the bounded decision system. Extending the same conclusions to more general physical encodings requires an explicit transport argument.

**2. Imported hardness regime:** the thermodynamic cost theorems are unconditional, but the stronger no-polynomial-certification claims rely on imported hardness results for the canonical family.

**3. Calibration choice:** Landauer calibration supplies the physical conversion constant. Stronger substrate-dependent lower bounds are possible, but they belong to a different modeling layer than the one studied here.

**4. Finite-budget model class:** the no-collapse theorem is a statement about globally bounded budget profiles with positive per-bit cost and exponential lower-bound growth. Different collapse claims require explicit bridges into that profile language.

## Impact

**For mathematical physics:** the England Replication Inequality (Theorem [\[thm:england\]](#thm:england){reference-type="ref" reference="thm:england"}) is reduced to finite counting plus Landauer calibration. The result gives a mechanized, non-asymptotic entropy gap for replication without appealing to an informal thermodynamic analogy.

**For information theory and complexity:** structural rank appears as the information coordinate of DOF, and the minimum-cost thermodynamic regime coincides with the tractable sufficiency regime.

**For formalized theory building:** the argument shows that physically meaningful theorem packages can be built from exact finite objects and kept fully auditable at the proof-artifact level.

## Final Remarks

Finite thermodynamic and structural consequences of exact resolution follow from first principles. In the formal bounded-system-to-`canonicalDP` map, the regime DOF $=1$ is the unique point at which structural rank, tractable sufficiency, and minimum calibrated thermodynamic cost coincide. The imported coherence theorem supplies a separate single-source interpretation of that same point.

All theorems are machine-checked. The thermodynamic lower bound is unconditional, and the replication inequality gives the entropy premium for multiplicity as $k_B \ln k$ per cycle. A companion manuscript develops the software-engineering and case-study implications. The mathematical-physics core is isolated here.


# Lean Proof Artifacts {#appendix-lean}

This appendix reports proof traceability directly from source and generated mapping artifacts.

## Claim Coverage Matrix

## Lean Handle Map

## Proof Hardness Index


  ----------------------------------------------------------------------------------------------
  **Paper claim**                                                  **Lean handle**
  ---------------------------------------------------------------- -----------------------------
  Corollary 4.4: Unique Minimum-Cost Regime                        BA8, L54, L55

  Corollary 5.9: $P = NP$ No-Go Transfer                           PH26, PH15, PH14

  Corollary 3.3: Higher-Rank Regime                                L46, L51

  Corollary 3.2: Rank-One Regime                                   L51

  Proposition 2.5: Bounded Region                                  BA1

  Definition 2.3: Bounded Decision System                          L17, L19

  Definition 2.13: Canonical Decision Problem                      QT2, QT7, QT1, QT3

  Theorem 2.6: Bounded Acquisition Rate                            BA1, BA2

  Theorem 5.1: Coherent Single-Source Regime                       ORA1

  Theorem 2.4: Counting Gap                                        BA10

  Theorem 2.7: Discrete Acquisition                                BA3

  Theorem 3.1: DOF--Structural-Rank Identity                       L43

  Theorem 4.3: Energy--Information Duality                         IT3, EI1, L43

  Theorem 4.1: Rank Controls Exact-Resolution Cost                 BA7, BA6, L43

  Theorem 5.6: England Replication Inequality                      L45

  Theorem 3.6: Decision-Entropy Bound                              IT3, L43

  Theorem 5.7: Finite-Budget No-Collapse                           PH26

  Theorem 5.5: Convergence                                         L43, L44, L47, L55

  Theorem 3.4: Minimum Physical Bit Operations                     BA5, BA6, L43, L46

  Theorem 3.5: Decision-Class Bound                                IT4, L43

  Theorem 2.8: One Transition, One Bit                             BA3, BA4

  Theorem 5.2: Rank Identification                                 L43, L46, L51

  Theorem 4.2: Rank-One Ground State                               BA8, L54, L55

  Theorem 2.9: Resolution Requires a Sufficient Coordinate Set     BA5

  Theorem 5.4: Thermodynamic Selection                             BA8, L49, L54, L55

  Theorem 5.3: Tractable Sufficiency at Rank One                   L47, L53, L56
  ----------------------------------------------------------------------------------------------

*Auto summary: mapped 26/26 (full=26, derived=0, unmapped=0).*


::: list
**`BA1`**[]{#lh:BA1} paper4/DecisionQuotient/Physics/BoundedAcquisition.lean

**`BA2`**[]{#lh:BA2} paper4/DecisionQuotient/Physics/BoundedAcquisition.lean

**`BA3`**[]{#lh:BA3} paper4/DecisionQuotient/Physics/BoundedAcquisition.lean

**`BA4`**[]{#lh:BA4} paper4/DecisionQuotient/Physics/BoundedAcquisition.lean

**`BA5`**[]{#lh:BA5} paper4/DecisionQuotient/Physics/BoundedAcquisition.lean

**`BA6`**[]{#lh:BA6} paper4/DecisionQuotient/Physics/BoundedAcquisition.lean

**`BA7`**[]{#lh:BA7} paper4/DecisionQuotient/Physics/BoundedAcquisition.lean

**`BA8`**[]{#lh:BA8} paper4/DecisionQuotient/Physics/BoundedAcquisition.lean

**`BA10`**[]{#lh:BA10} paper4/DecisionQuotient/Physics/BoundedAcquisition.lean

**`EI1`**[]{#lh:EI1} paper4/DecisionQuotient/ThermodynamicLift.lean

**`IT3`**[]{#lh:IT3} paper4/DecisionQuotient/Information.lean

**`IT4`**[]{#lh:IT4} paper4/DecisionQuotient/Information.lean

**`L17`**[]{#lh:L17} Leverage/Foundations.lean

**`L19`**[]{#lh:L19} Leverage/Theorems.lean

**`L43`**[]{#lh:L43} Leverage/BridgeToDQ.lean

**`L44`**[]{#lh:L44} Leverage/Foundations.lean

**`L45`**[]{#lh:L45} Leverage/BridgeToDQ.lean

**`L46`**[]{#lh:L46} Leverage/BridgeToDQ.lean

**`L47`**[]{#lh:L47} Leverage/BridgeToDQ.lean

**`L49`**[]{#lh:L49} Leverage/BridgeToDQ.lean

**`L51`**[]{#lh:L51} Leverage/BridgeToDQ.lean

**`L52`**[]{#lh:L52} Leverage/BridgeToDQ.lean

**`L53`**[]{#lh:L53}

**`L54`**[]{#lh:L54} Leverage/BridgeToDQ.lean

**`L55`**[]{#lh:L55} Leverage/BridgeToDQ.lean

**`L56`**[]{#lh:L56} paper4/DecisionQuotient/ClaimClosure.lean

**`ORA1`**[]{#lh:ORA1} paper2/Ssot/Coherence.lean

**`PH14`**[]{#lh:PH14} paper4/DecisionQuotient/Physics/PhysicalHardness.lean

**`PH15`**[]{#lh:PH15} paper4/DecisionQuotient/Physics/PhysicalHardness.lean

**`PH26`**[]{#lh:PH26} paper4/DecisionQuotient/Physics/PhysicalHardness.lean

**`QT1`**[]{#lh:QT1} paper4/DecisionQuotient/Quotient.lean

**`QT2`**[]{#lh:QT2} paper4/DecisionQuotient/Quotient.lean

**`QT3`**[]{#lh:QT3} paper4/DecisionQuotient/Quotient.lean

**`QT7`**[]{#lh:QT7} paper4/DecisionQuotient/Quotient.lean

**`W1`**[]{#lh:W1} paper4/DecisionQuotient/Physics/WassersteinIntegrity.lean

**`W2`**[]{#lh:W2} paper4/DecisionQuotient/Physics/WassersteinIntegrity.lean

**`W3`**[]{#lh:W3} paper4/DecisionQuotient/Physics/WassersteinIntegrity.lean

**`W4`**[]{#lh:W4} paper4/DecisionQuotient/Physics/WassersteinIntegrity.lean
:::

::: longtable
\@p0.05p0.42p0.05p0.42@ **ID** & **Lean Handle / Source** & **ID** & **Lean Handle / Source**\
**ID** & **Lean Handle / Source** & **ID** & **Lean Handle / Source**\
\
[**`BA1`**]{#lh:BA1} & `Physics.BoundedAcquisition.BoundedRegion`

& [**`BA2`**]{#lh:BA2} & `Physics.BoundedAcquisition.acquisition_rate_bound`

\
[**`BA3`**]{#lh:BA3} & `Physics.BoundedAcquisition.acquisitions_are_transitions`

& [**`BA4`**]{#lh:BA4} & `Physics.BoundedAcquisition.one_bit_per_transition`

\
[**`BA5`**]{#lh:BA5} & `Physics.BoundedAcquisition.resolution_reads_sufficient`

& [**`BA6`**]{#lh:BA6} & `Physics.BoundedAcquisition.srank_le_resolution_bits`

\
[**`BA7`**]{#lh:BA7} & `Physics.BoundedAcquisition.energy_ge_srank_cost`

& [**`BA8`**]{#lh:BA8} & `Physics.BoundedAcquisition.srank_one_energy_minimum`

\
[**`BA10`**]{#lh:BA10} & `Physics.BoundedAcquisition.counting_gap_theorem`

& [**`EI1`**]{#lh:EI1} & `ThermodynamicLift.energy_ge_kbt_nat_entropy`

\
[**`IT3`**]{#lh:IT3} & `DecisionQuotient.quotientEntropy_le_srank_binary`

& [**`IT4`**]{#lh:IT4} & `DecisionQuotient.numOptClasses_le_pow_srank_binary`

\
[**`L17`**]{#lh:L17} & `Leverage.compose_dof`

& [**`L19`**]{#lh:L19} & `Leverage.composition_dof_additive`

\
[**`L43`**]{#lh:L43} & `dof_eq_srank`

& [**`L44`**]{#lh:L44} & `dof_one_iff_max_leverage`

\
[**`L45`**]{#lh:L45} & `england_replication_inequality`

& [**`L46`**]{#lh:L46} & `incoherent_srank_gt_one`

\
[**`L47`**]{#lh:L47} & `max_coherence_forces_tractability`

& [**`L49`**]{#lh:L49} & `srank_energy_lower_bound`

\
[**`L51`**]{#lh:L51} & `ssot_srank_one`

& [**`L52`**]{#lh:L52} & `succ_le_two_pow`

\
[**`L53`**]{#lh:L53} & `sufficiency_conp_hard` & [**`L54`**]{#lh:L54} & `thermodynamic_selection`

\
[**`L55`**]{#lh:L55} & `thermodynamic_selection_unconditional`

& [**`L56`**]{#lh:L56} & `tractable_bounded_core`

\
[**`ORA1`**]{#lh:ORA1} & `oracle_arbitrary`

& [**`PH14`**]{#lh:PH14} & `PhysicalComplexity.p_eq_np_physically_impossible_of_collapse_map`

\
[**`PH15`**]{#lh:PH15} & `PhysicalComplexity.p_eq_np_physically_impossible_canonical`

& [**`PH26`**]{#lh:PH26} & `PhysicalComplexity.no_collapse_of_bounded_budget_pos_cost_exp_lb`

\
[**`QT1`**]{#lh:QT1} & `DecisionProblem.quotient_is_coarsest`

& [**`QT2`**]{#lh:QT2} & `DecisionProblem.quotientMap_preservesOpt`

\
[**`QT3`**]{#lh:QT3} & `DecisionProblem.quotient_represents_opt_equiv`

& [**`QT7`**]{#lh:QT7} & `DecisionProblem.quotient_has_unique_factorization`

\
[**`W1`**]{#lh:W1} & `Physics.single_future_zero_cost`

& [**`W2`**]{#lh:W2} & `Physics.transportCost_pos_of_offDiag`

\
[**`W3`**]{#lh:W3} & `Physics.integrity_is_centroid`

& [**`W4`**]{#lh:W4} & `Physics.wasserstein_bridge`

\
:::


  ------------------------------------------------------------------------------------------------------------------------
  **Paper handle**                  **Hardness profile**   **Regime tags**           **Lean support**
  --------------------------------- ---------------------- ------------------------- -------------------------------------
  `cor:minimum-cost-regime`         `unspecified`          \-                        BA8, L54, L55

  `cor:pnp-nogo`                    `unspecified`          \-                        PH26, PH15, PH14

  `cor:rank-above-one`              `unspecified`          \-                        L46, L51

  `cor:rank-one`                    `unspecified`          \-                        L51

  `prop:bounded-region`             `unspecified`          \-                        BA1

  `prop:dof-additive`               `unspecified`          \-                        L17, L19

  `prop:optimizer-quotient`         `unspecified`          \-                        QT2, QT7, QT1, QT3

  `thm:bounded-acquisition`         `unspecified`          \-                        BA1, BA2

  `thm:coherent-single-source`      `unspecified`          \-                        ORA1

  `thm:counting-gap`                `unspecified`          \-                        BA10

  `thm:discrete-acquisition`        `unspecified`          \-                        BA3

  `thm:dof-srank`                   `unspecified`          \-                        L43

  `thm:energy-entropy`              `unspecified`          \-                        IT3, EI1, L43

  `thm:energy-rank`                 `unspecified`          \-                        BA7, BA6, L43

  `thm:england`                     `unspecified`          \-                        L45

  `thm:entropy-bound`               `unspecified`          \-                        IT3, L43

  `thm:finite-budget-no-collapse`   `unspecified`          \-                        PH26

  `thm:five-way`                    `unspecified`          \-                        L43, L44, L47, L55

  `thm:min-bit-operations`          `unspecified`          \-                        BA5, BA6, L43, L46

  `thm:numopt-bound`                `unspecified`          \-                        IT4, L43

  `thm:one-transition-one-bit`      `unspecified`          \-                        BA3, BA4

  `thm:rank-identification`         `unspecified`          \-                        L43, L46, L51

  `thm:rank-one-ground`             `unspecified`          \-                        BA8, L54, L55

  `thm:resolution-sufficient`       `unspecified`          \-                        BA5

  `thm:thermodynamic-selection`     `unspecified`          \-                        BA8, L49, L54, L55

  `thm:tractable-rank-one`          `unspecified`          \-                        L47, L53, L56
  ------------------------------------------------------------------------------------------------------------------------

*Auto summary: indexed 26 claims by hardness profile (unspecified=26).*


# Notes on assumptions and extensions {#appendix-assumptions}

This appendix lists the principal modeling assumptions and common extensions relevant for the finite decision-thermodynamic framework of the paper:

-   **Canonical encoding:** The main theorems are exact for the canonical binary decision problem attached to the bounded decision system. Other physical encodings require an explicit transport theorem.

-   **Landauer calibration:** Thermodynamic cost is calibrated by a per-bit Landauer floor. Stronger substrate-dependent lower bounds may exist, but they are additional assumptions, not part of the theorem package under discussion.

-   **Exact decision setting:** The results concern exact sufficiency and exact-resolution cost. Approximate, stochastic, or bounded-confidence regimes require separate analysis.

-   **Finite state family:** The entropy and replication theorems are finite counting results. Continuum models must first be reduced to a finite decision quotient before these arguments apply.

-   **Future work:** Natural extensions include richer transport theorems from physical encodings to canonical decision problems, continuous-information analogues, and stronger substrate-specific dissipation bounds.


# Complete Theorem Index {#appendix-theorems}

Paper-level labeled claims:

**Foundations (Section 2):**

-   Proposition [\[prop:dof-additive\]](#prop:dof-additive){reference-type="ref" reference="prop:dof-additive"} (DOF Additivity)

-   Theorem [\[thm:counting-gap\]](#thm:counting-gap){reference-type="ref" reference="thm:counting-gap"}

-   Proposition [\[prop:bounded-region\]](#prop:bounded-region){reference-type="ref" reference="prop:bounded-region"}

-   Theorem [\[thm:bounded-acquisition\]](#thm:bounded-acquisition){reference-type="ref" reference="thm:bounded-acquisition"}

-   Theorem [\[thm:discrete-acquisition\]](#thm:discrete-acquisition){reference-type="ref" reference="thm:discrete-acquisition"}

-   Theorem [\[thm:one-transition-one-bit\]](#thm:one-transition-one-bit){reference-type="ref" reference="thm:one-transition-one-bit"}

-   Theorem [\[thm:resolution-sufficient\]](#thm:resolution-sufficient){reference-type="ref" reference="thm:resolution-sufficient"}

**Structural Rank and Decision Entropy (Section 3):**

-   Theorem [\[thm:dof-srank\]](#thm:dof-srank){reference-type="ref" reference="thm:dof-srank"}

-   Corollary [\[cor:rank-one\]](#cor:rank-one){reference-type="ref" reference="cor:rank-one"}

-   Corollary [\[cor:rank-above-one\]](#cor:rank-above-one){reference-type="ref" reference="cor:rank-above-one"}

-   Theorem [\[thm:min-bit-operations\]](#thm:min-bit-operations){reference-type="ref" reference="thm:min-bit-operations"}

-   Theorem [\[thm:numopt-bound\]](#thm:numopt-bound){reference-type="ref" reference="thm:numopt-bound"}

-   Theorem [\[thm:entropy-bound\]](#thm:entropy-bound){reference-type="ref" reference="thm:entropy-bound"}

**Thermodynamic Consequences (Section 4):**

-   Theorem [\[thm:energy-rank\]](#thm:energy-rank){reference-type="ref" reference="thm:energy-rank"}

-   Theorem [\[thm:rank-one-ground\]](#thm:rank-one-ground){reference-type="ref" reference="thm:rank-one-ground"}

-   Theorem [\[thm:energy-entropy\]](#thm:energy-entropy){reference-type="ref" reference="thm:energy-entropy"}

-   Corollary [\[cor:minimum-cost-regime\]](#cor:minimum-cost-regime){reference-type="ref" reference="cor:minimum-cost-regime"}

**Convergence (Section 5):**

-   Theorem [\[thm:five-way\]](#thm:five-way){reference-type="ref" reference="thm:five-way"}

-   Theorem [\[thm:england\]](#thm:england){reference-type="ref" reference="thm:england"}

-   Theorem [\[thm:finite-budget-no-collapse\]](#thm:finite-budget-no-collapse){reference-type="ref" reference="thm:finite-budget-no-collapse"}

-   Corollary [\[cor:pnp-nogo\]](#cor:pnp-nogo){reference-type="ref" reference="cor:pnp-nogo"}

**Primary Lean sources:**

-   `Leverage/Foundations.lean`

-   `Leverage/BridgeToDQ.lean`

-   `LambdaDR.lean`

-   `Leverage.lean`




---

## Machine-Checked Proofs

All theorems are formalized in Lean 4:
- Location: `docs/papers/paper3_leverage/proofs/`
- Lines: 75987
- Theorems: 3303
- `sorry` placeholders: 0
