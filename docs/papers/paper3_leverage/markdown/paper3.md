# Paper: Molecular Docking: A Machine-Checked Theory of Exact Resolution, Complexity, and Thermodynamic Cost

**Status**: Draft-ready | **Lean**: 79763 lines, 3498 theorems

---

## Abstract

Molecular docking lacks a rigorous theory of the exact object its methods target. Exact configuration resolution under constraints supplies that object. The abstract layer is a bounded decision system equipped with sufficient coordinate sets, a decision quotient, structural rank, and decision-quotient entropy. The molecular layer instantiates the same framework through holonomic constraints, cutoff-local interaction structure, sampled action families, and concrete scorer families.

The theorem package identifies one exact-resolution spine. Surjective abstractions either factor through the decision quotient or erase a decision-relevant distinction, and any physically feasible surjective collapse must therefore factor through the quotient. Structural rank is simultaneously the irreducible coordinate count, the quotient-entropy controller, and the exact Fisher-information dimension. General exact sufficiency certification contains a hardness core, and any sound checker for that core requires witness budget at least $2^{n-1}$. Cutoff locality bounds docking structural rank by active-site and ligand coordinates and therefore isolates a theorem-backed low-rank regime. Continuous-to-grid resolution control yields uniform exact/coarse score approximation and strict-gap winner preservation, while large-cutoff lattice-tail bounds yield direct structural-rank bounds for exact Lennard-Jones, Coulomb, and real-space Ewald docking families. Finite sampled docking admits a canonical uniform discrepancy radius, and bounded action families admit polynomial-time exact sufficiency checking. For sampled docking, exact and coarse winner sets agree under an explicit half-gap hypothesis, an action-side symmetry-broken flat refinement canonically selects a strict sampled winner and yields a uniform positive strict gap whenever at least two sampled actions remain, inside-cutoff coordinates are sufficient under the stated compatibility assumptions, exact top-$k$ survivors are preserved under a certified boundary gap, and near-tie regimes admit a certified ambiguity band. Strict dominance and constant-sign multiplicative separability collapse conformer dependence completely by making every coordinate set sufficient. Finite sampled Lennard-Jones and Coulomb scorer families satisfy theorem-backed exact/coarse invariance criteria under explicit cutoff error and half-gap conditions. The geometric execution layer includes exact phase-space volume preservation for Velocity-Verlet, a closed-form Lennard-Jones gradient, and exponentially decaying Ewald real-space control for long-range Coulomb splitting.

The same exact-resolution spine carries irreducible thermodynamic cost once a positive per-bit lower bound is fixed. Under Landauer calibration, any exact-resolution cycle satisfies $$E \geq k_B T\, H_{\mathrm{nats}}(D),$$ where $H_{\mathrm{nats}}(D)$ is the natural-log entropy of the decision quotient. In the canonical binary encoding, this bound sharpens to $$E \geq \mathrm{DOF}(A)\, k_B T \ln 2,$$ and the rank-$1$ regime is the unique thermodynamic ground state: every system with more than one degree of freedom lies strictly above the minimum per-cycle resolution cost.

The structural part of the argument is finite. Bounded systems admit only finitely many acquisition events. Exact resolution requires a sufficient coordinate set. The associated canonical decision problem records one binary acquisition channel for each degree of freedom, and the number of independent coordinates is identified exactly with the structural rank of the encoded decision problem: $$\mathrm{DOF}(A) = \mathrm{srank}(\mathrm{canonicalDP}(A)).$$ The decision quotient therefore has at most $2^{\mathrm{DOF}(A)}$ optimal-action classes, so its entropy is controlled by the same coordinate count that governs exact physical resolution.

For constrained molecular systems with $N$ atoms and $k$ independent holonomic constraints, the exact-resolution floor scales with $3N-k$. Thresholded one-bit readout channels and two-level atomic transition systems instantiate the same binary interface.

For bounded regions this yields the bounded-acquisition inequality $$\mathrm{DOF}(A) \le \frac{c\tau}{d},$$ and corresponding bounds on decision-class count from spacetime and energy budget.

Theorem-level mismatch and residual witnesses place nonideal implementations strictly above the Landauer floor. An explicit binary mismatch witness and an explicit two-state residual witness each yield at least one additional per-bit lower-bound unit above the Landauer floor and raise the energy--information coefficient above $k_B T$. Finite-capacity substrates therefore have bounded lifetime and bounded cumulative entropy throughput.

The empirical input is the per-bit conversion constant. Landauer furnishes the universal floor. The same calibrated model yields theorem-level mismatch and residual overhead above Landauer, bounded substrate lifetime and cumulative entropy throughput, the finite replication entropy gap, $$\Delta S_{\min}(k) - \Delta S_{\min}(1) \geq k_B \ln k.$$ and the finite-budget no-collapse theorem.

Keywords: molecular docking, exact resolution, structural rank, decision entropy, Landauer principle


_Failed to convert lean_stats.tex_

# Introduction

Molecular docking lacks a rigorous theory of the exact object its methods target. Exact configuration resolution under constraints supplies that object. The structural statements are finite counting, coordinate sufficiency, structural rank, decision-quotient entropy, certification hardness, and thermodynamic lower bounds. The abstract layer is a bounded decision system, represented in Lean by `Architecture` together with the canonical binary encoding `canonicalDP`. The molecular layer instantiates the same framework through holonomic constraint topology, cutoff-local interaction structure, sampled docking families, and concrete scorer families. The quotient object is closer to zero-error and confusability-based information than to average-case coding [@shannon1956zero; @korner1973graphs; @lovasz1979shannon].

One exact-resolution spine controls the theorem package. The decision quotient fixes the coarsest exact abstraction. Structural rank fixes the irreducible dimension of that quotient. The same rank controls quotient entropy, Fisher-information dimension, certification burden, and Landauer cost. For constrained molecular systems with $N$ atoms and $k$ independent holonomic constraints, the transported degree-of-freedom count is $3N-k$, and the exact-resolution floor scales with that remaining unconstrained dimension. Every docking pipeline therefore either solves this object, approximates it, or replaces it with a surrogate objective.

The theorem package extends the rank-entropy-cost chain. Surjective abstractions either factor through the decision quotient or erase a decision-relevant distinction, and any physically feasible surjective collapse must therefore factor through the quotient. Any sound checker for the empty-set hardness core requires witness budget at least $2^{n-1}$. The same structural rank is also the exact Fisher-information dimension of the decision problem. Continuous-to-grid resolution control yields uniform exact/coarse approximation and strict-gap winner preservation. Lattice-tail bounds discharge the docking cutoff hypothesis and compose directly into structural-rank bounds for exact Lennard-Jones, Coulomb, and real-space Ewald docking families. Finite sampled domains admit canonical discrepancy radii, bounded action families admit polynomial-time exact checking, action-side symmetry breaking yields a canonical strict sampled winner with a uniform positive strict gap and no structural-rank increase whenever at least two sampled actions remain, cutoff locality, sampled exact/coarse invariance, top-$k$ survivor control, ambiguity-band containment, strict-dominance and multiplicative-separability collapse regimes, and concrete Lennard-Jones, Coulomb, and Ewald theorems transport the abstract object into a molecular docking theory with explicit scorer families, approximation regimes, certificates, and geometric execution structure.

## Theorem Package

The endpoint of the theory is the convergence theorem in Section [\[five-way-equivalence\]](#five-way-equivalence){reference-type="ref" reference="five-way-equivalence"}:

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

The convergence theorem is the endpoint of the theorem chain. Earlier sections establish four ingredients that make the equivalence nontrivial: the quotient boundary for exact abstractions, the identification of structural rank with both irreducible coordinate count and Fisher-information dimension, the qualitative and quantitative certification lower bounds, and the molecular locality and stability theorems that make the docking specialization concrete.

Exact molecular resolution requires a sufficient coordinate set. Cutoff locality bounds the number of decision-relevant protein coordinates by the active site together with the ligand coordinates. Sampled docking preserves exact winners under explicit half-gap control, and an action-side symmetry-broken flat refinement canonically chooses a strict sampled winner with a uniform positive strict gap over the finite grid whenever the sampled action family has size at least two. The same sampled layer retains exact top-$k$ survivors under a certified boundary gap and admits a near-tie ambiguity band when strict separation fails. The resulting structural-rank bounds identify low-rank exact docking regimes. Finite flat grid encodings also carry a direct structural-rank entropy bound, and Section [\[main-theorems\]](#main-theorems){reference-type="ref" reference="main-theorems"} now applies the thermodynamic floor directly to the exact docking decision problem, to an explicit fixed-length binary encoding of the finite sampled grid problem, and to the bounded-region sampled inside-cutoff process witness under explicit sufficient-set and budget hypotheses. Together these give the concrete molecular instantiation of the abstract theory.

## Structural and Empirical Inputs

Theorem [\[thm:counting-gap\]](#thm:counting-gap){reference-type="ref" reference="thm:counting-gap"} fixes the finite-event statement of the model. Proposition [\[prop:bounded-region\]](#prop:bounded-region){reference-type="ref" reference="prop:bounded-region"} and Theorem [\[thm:bounded-acquisition\]](#thm:bounded-acquisition){reference-type="ref" reference="thm:bounded-acquisition"} fix the traversal-rate statement for bounded regions. Theorems [\[thm:discrete-acquisition\]](#thm:discrete-acquisition){reference-type="ref" reference="thm:discrete-acquisition"} and [\[thm:one-transition-one-bit\]](#thm:one-transition-one-bit){reference-type="ref" reference="thm:one-transition-one-bit"} fix the acquisition-event interface used by the canonical encoding. Landauer calibration enters only in Section [\[main-theorems\]](#main-theorems){reference-type="ref" reference="main-theorems"}, where the per-bit lower bound is converted into energy.

## Contributions

1.  **Reference Semantics for Molecular Docking:** exact molecular docking is exact configuration resolution under constraints through sufficient coordinate sets, the decision quotient, and the canonical exact-resolution encoding. Surjective exact summaries factor through the quotient, while any extra collapse erases a decision-relevant distinction. Approximate and heuristic docking methods are thereby located relative to a single exact target object.

2.  **Complexity Boundary:** general exact sufficiency certification contains a hardness core, any sound checker for that core requires witness budget at least $2^{n-1}$, bounded action families admit polynomial-time exact checking, cutoff locality bounds docking structural rank by active-site and ligand coordinates, sampled docking admits theorem-backed exact/coarse winner preservation and inside-cutoff sufficiency under explicit gap and compatibility hypotheses, action-side symmetry breaking yields a canonical strict sampled winner with a uniform positive strict gap and no structural-rank increase, top-$k$ as well as near-tie regimes remain under certified control, and strict-dominance or multiplicative-separable scorer families collapse conformer dependence by making every coordinate set sufficient.

3.  **Thermodynamic and Statistical Dimension:** the canonical exact-resolution problem assigns one binary acquisition channel to each degree of freedom, the resulting structural rank equals that channel count, the decision entropy is bounded by the same count, the Fisher-information dimension is the same structural rank, and under Landauer calibration exact-resolution cost is bounded below by both $\mathrm{DOF}(A)\,k_B T \ln 2$ and $k_B T H_{\mathrm{nats}}(D)$.

4.  **Approximation, Geometry, and Concrete Scorer Families:** constrained molecular systems with $N$ atoms and $k$ independent holonomic constraints transport directly into the framework with effective dimension $3N-k$, binary constraint-status interface, and a Landauer-linear floor scaling with the remaining unconstrained coordinates. Continuous-to-grid Lipschitz control yields uniform exact/coarse approximation radii, large-cutoff lattice-tail bounds discharge the cutoff-locality hypothesis, finite sampled docking admits a canonical discrepancy radius and theorem-backed pruning certificates, finite sampled Lennard-Jones and Coulomb scorer families satisfy exact/coarse invariance criteria under explicit cutoff error and half-gap conditions, and the geometric execution layer proves exact phase-space volume preservation for Velocity-Verlet, a closed-form Lennard-Jones gradient, and exponential Ewald decay for long-range Coulomb splitting.

5.  **Nonideal Exact Resolution:** theorem-level mismatch and residual witnesses force effective per-bit floors strictly above Landauer, explicit binary mismatch and two-state residual witnesses each yield an additive one-unit overhead above the Landauer floor, and finite-capacity substrates therefore have bounded lifetime and bounded entropy throughput.

6.  **Convergence and Universal Consequences:** the rank-$1$ regime is simultaneously the one-coordinate regime, the tractable sufficiency regime, the coherent single-source regime, and the thermodynamic ground state; the same model yields the finite replication entropy gap and the finite-budget no-collapse theorem.

7.  **Finite Physical Acquisition:** bounded systems admit finitely many acquisition events, bounded regions admit finite acquisition rates, acquisition is represented by discrete transition events, and exact resolution requires a sufficient coordinate set.

8.  **Concrete Substrate Instantiation:** thresholded one-bit readouts and two-level atomic transition systems instantiate the canonical binary interface used by the theorem chain.

#### Physical significance.

Once finite acquisition is fixed as the physical interface, the degree-of-freedom count becomes simultaneously an interaction dimension, an entropy bound, and a Landauer-calibrated cost coordinate. The result concerns exact resolution structure in matter.

Informally: exact resolution must be paid for.

## Scope

The mathematical structure links structural rank, quotient entropy, Fisher-information dimension, certification hardness, thermodynamic cost, and geometric execution structure. Theorems are stated abstractly for bounded decision systems represented by the Lean object `Architecture` and are instantiated concretely for constrained molecular systems, cutoff-local docking problems, sampled docking families, finite scorer approximations, and phase-space-preserving execution maps.

## Organization

Section [\[foundations\]](#foundations){reference-type="ref" reference="foundations"} defines the structural model, the finite-acquisition interface, and the exact docking semantics. Section [\[probability-model\]](#probability-model){reference-type="ref" reference="probability-model"} derives the quotient boundary, structural-rank identities, Fisher-dimension theorems, and the finite compression bridge. Section [\[complexity-boundary\]](#complexity-boundary){reference-type="ref" reference="complexity-boundary"} states the hardness core, checker lower bounds, bounded-action checking regime, cutoff-local low-rank regime, continuous-to-grid and large-cutoff approximation bridges, conformer-search collapse regimes, sampled docking preservation theorems, top-$k$ and near-tie control, pruning certificates, concrete scorer-family invariance results, and the geometric and electrostatic support theorems. Section [\[main-theorems\]](#main-theorems){reference-type="ref" reference="main-theorems"} derives the thermodynamic cost consequences. Section [\[five-way-equivalence\]](#five-way-equivalence){reference-type="ref" reference="five-way-equivalence"} states the convergence theorem and the remaining universal consequences. Section [\[related-work\]](#related-work){reference-type="ref" reference="related-work"} situates the results relative to thermodynamics, information theory, and molecular computation. Appendix [\[appendix-lean\]](#appendix-lean){reference-type="ref" reference="appendix-lean"} records proof provenance.


# Exact-Resolution Model {#foundations}

The formal objects that carry the mathematical-physics content are a positive degree-of-freedom count, a canonical binary decision encoding, structural rank, and decision entropy.

## Formal Object

::: definition
[]{#def:architecture label="def:architecture"} A *bounded decision system* is a finite bounded physical system equipped with a positive integer $\mathrm{DOF}(A)$. The corresponding Lean object is named `Architecture`. The results used below depend only on the degree-of-freedom count and on the canonical decision encoding.
:::

**Interpretation.** $\mathrm{DOF}(A)$ counts independent coordinates that can vary separately. Subsequent sections study what that count forces once one asks for exact resolution.

**Molecular instantiation.** In the docking setting, the bounded decision system is a constrained molecular configuration space together with the binding decision problem induced by the chosen interaction model. The abstract degree-of-freedom count is later instantiated by holonomic constraint topology and local interaction structure.

## Degrees of Freedom

::: definition
[]{#def:dof label="def:dof"} The quantity $\mathrm{DOF}(A) \in \mathbb{N}$ counts independent coordinates of variation in a bounded decision system $A$. In the mechanized development it is the structural parameter attached to `Architecture`; later sections identify it exactly with the structural rank of a canonical decision problem.
:::

**Operational meaning.** If $\mathrm{DOF}(A)=n$, the system has $n$ independent coordinates that must be resolved in the worst case by any exact resolver.

**Molecular instantiation.** For a constrained molecular system with $N$ atoms and $k$ independent holonomic constraints, the transported degree-of-freedom count is $3N-k$. Later sections combine this finite topological count with cutoff-local docking structure, yielding structural-rank bounds derived entirely from molecular topology and interaction geometry.

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

Theorem [\[thm:counting-gap\]](#thm:counting-gap){reference-type="ref" reference="thm:counting-gap"} fixes the finite-event statement of the model. Proposition [\[prop:bounded-region\]](#prop:bounded-region){reference-type="ref" reference="prop:bounded-region"} and Theorem [\[thm:bounded-acquisition\]](#thm:bounded-acquisition){reference-type="ref" reference="thm:bounded-acquisition"} fix the geometric acquisition bound. Theorems [\[thm:discrete-acquisition\]](#thm:discrete-acquisition){reference-type="ref" reference="thm:discrete-acquisition"} and [\[thm:one-transition-one-bit\]](#thm:one-transition-one-bit){reference-type="ref" reference="thm:one-transition-one-bit"} fix the acquisition-event interface. Landauer calibration is applied only after that interface is fixed.

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
[]{#thm:discrete-acquisition label="thm:discrete-acquisition"} In the imported bounded-acquisition model, information acquisition is counted by transition points of a finite discrete system. Acquisition counts are therefore discrete event counts.
:::

::: proof
*Proof.* The imported model represents a bounded physical decision process by a finite `DiscreteSystem`. Its acquisition count is `bitOperations`, which counts transition points along a run. ◻
:::

::: theorem
[]{#thm:one-transition-one-bit label="thm:one-transition-one-bit"} In the imported discrete acquisition model, each transition point contributes one unit to the bit-operation count. The canonical binary encoding therefore uses one Boolean coordinate per elementary acquisition event.
:::

::: proof
*Proof.* The imported theorem states that a transition point at time $t$ contributes at least one unit to the acquisition count up to time $t+1$. The model therefore treats each elementary transition as one Boolean acquisition event. ◻
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

The encoding is the exact Lean object `canonicalDP` in `Leverage/BridgeToDQ.lean`. It serves as the exact finite-resolution object attached to the declared degree-of-freedom count. It assigns one binary acquisition channel to each degree of freedom and one query action to each channel. The next section identifies the structural rank of this object with that coordinate count.

**Docking reading.** The canonical exact-resolution problem records the distinctions that any exact docking resolver must preserve. The quotient of this problem is therefore the exact abstraction of docking correctness, not an auxiliary coding artifact.

## Structural Rank

::: definition
[]{#def:srank label="def:srank"} The *structural rank* of a finite decision problem is the cardinality of its relevant coordinate set, equivalently the size of any minimal sufficient set. It is the minimum interaction dimension that must be read to determine the optimal action exactly.
:::

For the canonical decision problem attached to $A$, the relevant coordinate set is all of $\mathrm{Fin}\;\mathrm{DOF}(A)$, so the structural-rank problem is exactly matched to the declared degree-of-freedom count.

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

The degree-of-freedom object lives in `Leverage/Foundations.lean`, while the canonical decision encoding and its rank-identification theorems live in `Leverage/BridgeToDQ.lean`. Structural rank and decision entropy are formalized in the decision-quotient development.


# Exact Resolution, Quotient Structure, and Compression {#probability-model}

The theorems of this section identify the exact object before any complexity or thermodynamic lower bound is applied. The decision quotient is the coarsest exact abstraction of the optimizer. Structural rank is the irreducible dimension of that exact object. Fisher information gives the same dimension a statistical reading. The canonical exact-resolution encoding then transports that abstract structure into degree-of-freedom counts, entropy bounds, and physical bit requirements.

## Abstraction Boundary

::: theorem
[]{#thm:abstraction-factors-or-erases label="thm:abstraction-factors-or-erases"} Let $\phi : S \to T$ be a surjective abstraction of states for a decision problem $D$. Then exactly one of the following structural possibilities occurs: $$\text{$\phi$ factors through the decision quotient of $D$}
\qquad\text{or}\qquad
\text{$\phi$ erases a decision-relevant distinction.}$$
:::

::: proof
*Proof.* If $\phi$ preserves the optimal-action correspondence, the quotient is the coarsest such abstraction and $\phi$ factors through it. If $\phi$ fails to preserve the optimal-action correspondence, then by definition it identifies two states with different optimal-action sets and therefore erases a decision-relevant distinction. ◻
:::

::: theorem
[]{#thm:feasible-collapse-factors label="thm:feasible-collapse-factors"} Let $\phi : S \to T$ be a surjective abstraction of states for a decision problem $D$. If every decision-relevant distinction erased by $\phi$ were mapped to a physically feasible collapse at the canonical requirement profile, then $\phi$ must factor through the decision quotient of $D$.
:::

::: proof
*Proof.* The physical no-collapse layer rules out any physically feasible realization of an abstraction that erases a decision-relevant distinction at the canonical requirement profile. The only remaining possibility is that the abstraction preserves the optimal-action correspondence and therefore factors through the decision quotient. ◻
:::

The quotient is the coarsest surjective exact abstraction that remains available once decision-relevant erasure is excluded.

## Fisher Dimension

::: theorem
[]{#thm:fisher-sum-srank label="thm:fisher-sum-srank"} For every finite decision problem $D$, $$\sum_i \mathrm{FisherScore}_D(i) = \mathrm{srank}(D).$$
:::

::: proof
*Proof.* Each coordinate contributes Fisher score $1$ exactly when it is structurally relevant and score $0$ otherwise. Summing those indicator values therefore counts the relevant coordinates, which is exactly the structural rank. ◻
:::

::: theorem
[]{#thm:fisher-rank-srank label="thm:fisher-rank-srank"} For every finite decision problem $D$, $$\operatorname{rank}(I_D) = \mathrm{srank}(D),$$ where $I_D$ is the diagonal Fisher information matrix induced by the relevance profile of $D$.
:::

::: proof
*Proof.* The Fisher matrix is diagonal with a $1$ on each structurally relevant coordinate and a $0$ on each irrelevant coordinate. Its rank is therefore the number of nonzero diagonal entries, which is exactly the structural rank. ◻
:::

Structural rank therefore has three exact readings in the present development: combinatorial irreducible-coordinate count, quotient entropy controller, and Fisher-information dimension.

## Canonical Exact-Resolution Encoding

::: theorem
[]{#thm:dof-srank label="thm:dof-srank"} For every bounded decision system $A$, $$\mathrm{srank}(\mathrm{canonicalDP}(A)) = \mathrm{DOF}(A).$$
:::

::: proof
*Proof.* Write $n = \mathrm{DOF}(A)$. By Definition [\[def:canonical-dp\]](#def:canonical-dp){reference-type="ref" reference="def:canonical-dp"}, the state space is $\mathrm{Fin}\;n \to \mathrm{Bool}$, query action $\mathrm{inl}(i)$ has utility $2$ exactly when coordinate $i$ is true and utility $0$ otherwise, and the fallback action has utility $1$. Fix any coordinate $i$ and choose two states that agree everywhere except at $i$, with one state setting $i$ to true and the other setting $i$ to false; then $\mathrm{inl}(i)$ is optimal in the first state and not optimal in the second, so erasing coordinate $i$ changes the optimizer. Thus every coordinate in $\mathrm{Fin}\;n$ is relevant, the relevant-coordinate set has cardinality $n$, and the structural rank is $n$. Substituting $n = \mathrm{DOF}(A)$ gives the claim. ◻
:::

The canonical encoding provides the bridge from the abstract exact object to a degree-of-freedom count carried by a declared architecture.

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

## Finite Compression Bridge

The next proposition packages the finite bridge in direct compression language: a finite Hamiltonian induces a deterministic tie-broken compression relation, and the paper1 fiber moment becomes the exact collision moment of that relation.

::: proposition
[]{#prop:finite-compression-bridge label="prop:finite-compression-bridge"} Let $H(c,\bar c)$ be a finite compression Hamiltonian. Write $$R_H^{\min}(c,\bar c) \iff \bar c \in \operatorname*{arg\,min}_{\bar c'} H(c,\bar c'),
\qquad
R_H^{\mathrm{tb}}(c,\bar c) \iff \bar c \text{ is the least minimizer.}$$ For $$M_H(\bar c) := \left|\left\{c : R_H^{\mathrm{tb}}(c,\bar c)\right\}\right|,$$ and every $s \in \mathbb{N}$, $$\left|\left\{(\bar c,(c_i)_{i < s}) : \forall i,\; R_H^{\mathrm{tb}}(c_i,\bar c)\right\}\right|
=
\sum_{\bar c} M_H(\bar c)^s.$$ If moreover $$\left|\left\{c : R_H^{\min}(c,\bar c)\right\}\right| \le 2^b
\qquad\text{for every } \bar c,$$ then the induced tie-broken encoder has zero identity debt at budget $b$.
:::

::: proof
*Proof.* The first identity is the exact finite shared-codeword/fiber-moment theorem for the least-minimizer relation induced by $H$. The second statement uses that each tie-broken fiber sits inside the corresponding raw argmin fiber, so a uniform raw argmin bound transfers to the deterministic tie-broken encoder. ◻
:::

The compression bridge is the point of contact with the Landauer chain: in the exact-resolution reading of the canonical model, thermodynamic cost is the combinatorial cost of avoiding encoder collisions, because zero identity debt reduces to a uniform argmin-fiber bound, and that finite fiber-size condition is exactly what the energy--information theorem charges.

In docking language, the same finite fiber structure records when distinct molecular configurations remain exactly distinguishable under the binding decision relation.

Informally: to avoid collisions is to pay for distinctions.

## Formalization

The structural-rank bridge is formalized in `Leverage/BridgeToDQ.lean`; the abstraction-collapse, Fisher-rank, and exact-sufficiency bridge theorems are assembled in `Leverage/DockingTheoryBridge.lean`; the finite compression bridge is formalized in `Leverage/ColumnComplexityBridge.lean`; and the minimum-bit and entropy bounds are formalized in the decision-quotient physics and information development. These are the objects used directly by the complexity and thermodynamic theorems of the next sections.


# Complexity Boundary of Exact Molecular Docking {#complexity-boundary}

Exact molecular docking has a genuine tractability boundary because the exact object already carries both qualitative and quantitative certification lower bounds. General exact sufficiency certification contains a hardness core. Sound checking requires witness budget. Molecular locality, sampling hypotheses, and concrete scorer approximations then carve out theorem-backed low-rank and stability regimes inside that harder ambient problem class. The claims in this section isolate that boundary before the thermodynamic lower bounds are applied.

## General Hardness Core

::: theorem
[]{#thm:exact-sufficiency-hardness-core label="thm:exact-sufficiency-hardness-core"} For every Boolean formula $\phi$, the empty coordinate set is sufficient for the reduction problem induced by $\phi$ if and only if $\phi$ is a tautology.
:::

::: proof
*Proof.* The reduction theorem is exact: tautology is encoded as sufficiency of the empty coordinate set for the induced decision problem. The equivalence therefore supplies the formal hardness core for exact sufficiency certification. ◻
:::

::: theorem
[]{#thm:hard-family-srank label="thm:hard-family-srank"} Let $n>0$, and let $\phi$ be a non-tautology over $n$ Boolean variables. Then the many-coordinate reduction family has structural rank exactly $n$.
:::

::: proof
*Proof.* Every coordinate is relevant in the non-tautology branch of the strengthened reduction family. Structural rank is the cardinality of the relevant-coordinate set, so the rank is exactly $n$. ◻
:::

The hard family therefore witnesses full interaction dimensionality: exact sufficiency can force the decision boundary to depend on every available coordinate.

## Quantitative Certification Lower Bounds

::: theorem
[]{#thm:checker-budget-lower-bound label="thm:checker-budget-lower-bound"} For the empty-set sufficiency core on $n \ge 1$ coordinates, any sound finite checker must inspect at least $$2^{n-1}$$ pair witnesses.
:::

::: proof
*Proof.* The witness budget for the empty-set core is $2^{n-1}$. Any sound checker must inspect enough witness pairs to refute every false empty-set sufficiency claim, so the checking budget is bounded below by that witness budget. ◻
:::

::: corollary
[]{#cor:no-sound-checker-below-budget label="cor:no-sound-checker-below-budget"} For the same empty-set core, any checker operating strictly below the witness budget fails to be sound.
:::

::: proof
*Proof.* This is the contrapositive form of Theorem [\[thm:checker-budget-lower-bound\]](#thm:checker-budget-lower-bound){reference-type="ref" reference="thm:checker-budget-lower-bound"}. ◻
:::

::: corollary
[]{#cor:checking-time-lower-bound label="cor:checking-time-lower-bound"} If runtime is bounded below by the number of checked witness pairs, then any sound checker for the empty-set core requires runtime at least $$2^{n-1}.$$
:::

::: proof
*Proof.* The checking budget lower bound transfers directly to runtime once runtime dominates the number of checked pairs. ◻
:::

The hardness core is therefore quantitative as well as qualitative: exact certification carries an unavoidable witness budget in addition to the reduction-theoretic hardness statement.

## Cutoff-Local Docking Regime

::: theorem
[]{#thm:molecular-docking-srank-bound label="thm:molecular-docking-srank-bound"} Let $P_{\mathrm{rel}}$ be the number of protein atoms within the cutoff radius of the binding site, and let $L$ be the number of ligand atoms. Under the strict-optimum and outside-cutoff boundedness hypotheses, $$\mathrm{srank}(D_{\mathrm{dock}}) \le 3P_{\mathrm{rel}} + 3L.$$
:::

::: proof
*Proof.* Any protein coordinate that remains decision-relevant must come from an atom within the cutoff radius, while every ligand coordinate remains available. Structural rank is therefore bounded by three coordinates for each cutoff-local protein atom together with three coordinates for each ligand atom. ◻
:::

::: corollary
[]{#cor:bounded-pocket-regime label="cor:bounded-pocket-regime"} If at most $K$ protein atoms lie within the cutoff radius of the binding site and the ligand has at most $L$ atoms, then $$\mathrm{srank}(D_{\mathrm{dock}}) \le 3K + 3L.$$
:::

::: proof
*Proof.* Substitute the pocket-size and ligand-size bounds into Theorem [\[thm:molecular-docking-srank-bound\]](#thm:molecular-docking-srank-bound){reference-type="ref" reference="thm:molecular-docking-srank-bound"}. ◻
:::

Informally: bounded pockets bound exact difficulty.

## Conformer-Search Collapse Regimes

::: theorem
[]{#thm:strict-dominance-empty-sufficient label="thm:strict-dominance-empty-sufficient"} Let $D$ be a finite exact docking problem. If one action strictly dominates every other action at every state, then every coordinate set is sufficient for $D$.
:::

::: proof
*Proof.* Strict global dominance makes the dominant action uniquely optimal at every state. The optimal-action set is therefore constant, so no coordinate distinction can change exact resolution. ◻
:::

::: theorem
[]{#thm:multiplicative-separable-empty-sufficient label="thm:multiplicative-separable-empty-sufficient"} Let $D$ be a finite exact docking problem with utility of the form $$U(a,s)=f(a)g(s).$$ If the state factor $g$ has constant sign, then every coordinate set is sufficient for $D$.
:::

::: proof
*Proof.* With constant-sign state factor, the optimal-action set depends only on the ordering of the action factor $f$. The optimizer is therefore independent of state and exact sufficiency collapses to the empty-information regime. ◻
:::

These theorems give exact collapse regimes for conformer search: under strict dominance or constant-sign multiplicative separability, conformational variation does not change the exact optimizer.

## Sampled Docking

::: theorem
[]{#thm:sampled-docking-gap label="thm:sampled-docking-gap"} For a finite sampled docking problem, let $a_\ast$ be a strict exact winner at a sampled state $s$. If the coarse score differs from the exact score by at most $\delta$ on every sampled action at $s$, and $$\delta < \frac{1}{2}\,\mathrm{StrictUtilityGap}(a_\ast,s),$$ then the exact and coarse optimal-action sets agree at $s$.
:::

::: proof
*Proof.* The theorem is the sampled docking specialization of the strict half-gap invariance principle: a perturbation smaller than half the exact winner's strict margin cannot change the optimal set. ◻
:::

::: theorem
[]{#thm:action-pinned-uniform-gap label="thm:action-pinned-uniform-gap"} Let $$D^{\mathrm{flat}}_{\mathrm{dock,pin}}$$ be the action-side symmetry-broken refinement of the flat exact sampled docking problem, and let $a^\dagger(s)$ denote the canonical max-code representative of the original exact optimal set at flat sampled grid state $s$. If the sampled action family has more than one retained action, then there exists $\gamma>0$ such that for every flat sampled grid state $s$, $$\gamma \le \mathrm{StrictUtilityGap}_{D^{\mathrm{flat}}_{\mathrm{dock,pin}}}(a^\dagger(s),s).$$ Moreover, $$\mathrm{srank}\!\bigl(D^{\mathrm{flat}}_{\mathrm{dock,pin}}\bigr)
\le
\mathrm{srank}\!\bigl(D^{\mathrm{flat}}_{\mathrm{dock,exact}}\bigr).$$
:::

::: proof
*Proof.* Action-side pinning canonically chooses a strict sampled winner from each exact tie class. The resulting strict utility gap is positive at every flat sampled grid state, and finiteness of the sampled grid state space yields a positive global minimum $\gamma$. Because the symmetry breaking acts only on the action side, it does not increase the number of state coordinates needed for exact resolution. ◻
:::

This action-pinned refinement is the theorem-backed replacement for a generic no-ties assumption. It preserves the state-side exact-resolution problem, does not increase structural rank, and converts every finite sampled tie class into a canonical strict winner with a uniform positive margin. The later half-gap invariance theorems can therefore be applied to the pinned sampled family with an explicit witness $\gamma$ rather than an ad hoc generic-position premise.

::: theorem
[]{#thm:sampled-inside-cutoff-sufficient label="thm:sampled-inside-cutoff-sufficient"} Under the cutoff boundedness, sampled-optimum capture, coordinate-compatibility, and injectivity hypotheses, the retained coordinate set consisting of inside-cutoff protein coordinates together with all ligand coordinates is sufficient for the sampled restricted docking problem.
:::

::: proof
*Proof.* Cutoff locality forces every relevant sampled coordinate into the retained set. The compatibility and injectivity hypotheses then lift the retained-set relevance bound into a sufficiency theorem for the sampled restricted problem. ◻
:::

## Top-k and Near-Tie Control

::: theorem
[]{#thm:topk-boundary-gap label="thm:topk-boundary-gap"} Let $u_{\mathrm{exact}}$ and $u_{\mathrm{coarse}}$ be finite score functions on a docking action family. If the coarse score differs from the exact score by at most $\delta$ on every action, and if $\delta$ is no larger than the boundary gap at threshold $\tau$, then every exact top-$k$ action survives the coarse threshold filter at $\tau$.
:::

::: proof
*Proof.* The boundary-gap condition places every exact top-$k$ action at least $\delta$ above the threshold. Uniform score error bounded by $\delta$ therefore keeps every exact top-$k$ action above the coarse threshold as well. ◻
:::

::: theorem
[]{#thm:topk-ambiguity-band label="thm:topk-ambiguity-band"} For every nonnegative slack parameter $\varepsilon$, every exact top-$k$ action lies inside the certified ambiguity band of width $\varepsilon$ around the exact $k$th boundary.
:::

::: proof
*Proof.* The ambiguity band is defined by lowering the exact $k$th threshold by $\varepsilon$. Every exact top-$k$ action remains above that relaxed threshold and is therefore retained. ◻
:::

These top-$k$ theorems give a conservative exact-screening regime even when strict single-winner separation is unavailable.

## Continuous-to-Discrete Approximation Bridges

::: theorem
[]{#thm:lipschitz-grid-approx label="thm:lipschitz-grid-approx"} Let $u_{\mathrm{cont}}$ be a continuous-state score family, let $u_{\mathrm{grid}}$ be its grid approximation, and let each grid state carry a state-discretization error bounded by $\mathrm{res}$. If the score discrepancy is Lipschitz with constant $L$, then the exact-versus-grid score error is uniformly bounded by $L\,\mathrm{res}$.
:::

::: proof
*Proof.* The Lipschitz hypothesis bounds score discrepancy at each grid state by $L$ times its state error. A uniform state-error bound by $\mathrm{res}$ therefore yields a uniform score radius $L\,\mathrm{res}$. ◻
:::

::: theorem
[]{#thm:resolution-controlled-uniform label="thm:resolution-controlled-uniform"} If the exact-versus-grid score discrepancy is bounded by a resolution-dependent envelope $\varepsilon(\mathrm{res})$ on every lifted grid state, then the lifted grid problem is a uniform utility approximation with radius $\varepsilon(\mathrm{res})$.
:::

::: proof
*Proof.* The resolution-controlled hypothesis already states the required pointwise uniform score bound on the lifted grid state space. ◻
:::

::: theorem
[]{#thm:resolution-controlled-gap-invariance label="thm:resolution-controlled-gap-invariance"} Let $u_{\mathrm{cont}}$ be a continuous score family, $u_{\mathrm{grid}}$ its discretized score family, and $\mathrm{lift}$ the map from grid states to continuous states. If the exact-versus-grid score discrepancy is bounded by $\varepsilon(\mathrm{res})$ on every lifted grid state, and if $$\varepsilon(\mathrm{res}) < \frac{1}{2}\,\mathrm{StrictUtilityGap},$$ then the lifted exact optimizer and the grid optimizer agree at that grid state.
:::

::: proof
*Proof.* Resolution-controlled approximation yields a uniform utility approximation with radius $\varepsilon(\mathrm{res})$. The standard strict half-gap criterion then gives exact/coarse winner preservation. ◻
:::

::: theorem
[]{#thm:large-cutoff-bounded label="thm:large-cutoff-bounded"} Let a binding problem admit a lattice-tail perturbation bound with coefficient $c$ and a positive strict minimum decision gap $\gamma$. If the chosen cutoff radius satisfies $$c\,\mathrm{TailSum}_6(R_{\mathrm{cut}}) < \gamma/2,$$ then the exact docking problem satisfies the cutoff-boundedness hypothesis required by the structural-rank and sampled-docking locality theorems.
:::

::: proof
*Proof.* The lattice-tail bound supplies a uniform upper bound on every outside-cutoff utility perturbation. The strict minimum gap condition then shows that the perturbation remains below half the exact decision gap, which is exactly the cutoff-boundedness criterion. ◻
:::

These theorems provide a direct route from continuous regularity and physical tail control to the finite exact/coarse approximation theorems used later in the docking development.

::: theorem
[]{#thm:lj-shell-derivative-envelope label="thm:lj-shell-derivative-envelope"} Fix a positive lower shell radius $r_{\min}>0$. Then every radius $r \ge r_{\min}$ satisfies $$\left|\frac{d}{dr}U_{\mathrm{LJ}}(r)\right|
\le
\left(24|\varepsilon|r_{\min}^{-1}\right)
\left((|\sigma|r_{\min}^{-1})^6 + 2(|\sigma|r_{\min}^{-1})^{12}\right).$$
:::

::: proof
*Proof.* The closed-form Lennard-Jones derivative is bounded by replacing every inverse radius factor with the lower-shell inverse $r_{\min}^{-1}$ and then bounding the polynomial terms by monotonicity on the positive shell. ◻
:::

::: theorem
[]{#thm:lj-shell-second-derivative-envelope label="thm:lj-shell-second-derivative-envelope"} Fix a positive lower shell radius $r_{\min}>0$. Then every radius $r \ge r_{\min}$ satisfies $$\left|\frac{d^2}{dr^2}U_{\mathrm{LJ}}(r)\right|
\le
\left(24|\varepsilon|r_{\min}^{-2}\right)
\left(26(|\sigma|r_{\min}^{-1})^{12} + 7(|\sigma|r_{\min}^{-1})^6\right).$$
:::

::: proof
*Proof.* Differentiate the closed-form gradient once more and bound every inverse radius factor by the lower-shell inverse. The polynomial terms are again controlled by monotonicity on the positive shell. ◻
:::

::: theorem
[]{#thm:lj-shell-hessian-lipschitz label="thm:lj-shell-hessian-lipschitz"} On a positive shell $[r_{\min},r_{\max}]$, any uniform bound on the Lennard-Jones second derivative yields a Lipschitz bound on the Lennard-Jones gradient: $$|U'_{\mathrm{LJ}}(y)-U'_{\mathrm{LJ}}(x)| \le K|y-x|.$$
:::

::: proof
*Proof.* The second derivative is the derivative of the gradient. A uniform shell bound on that derivative therefore makes the gradient Lipschitz by the one-dimensional mean value theorem. ◻
:::

::: theorem
[]{#thm:lj-shell-quadratic-remainder label="thm:lj-shell-quadratic-remainder"} Fix a positive shell $[r_{\min}, r_{\max}]$. Then for every $x,y \in [r_{\min}, r_{\max}]$, $$|U_{\mathrm{LJ}}(y)-U_{\mathrm{LJ}}(x)-U'_{\mathrm{LJ}}(x)(y-x)|
\le
\frac{1}{2}
\left(24|\varepsilon|r_{\min}^{-2}\right)
\left(26(|\sigma|r_{\min}^{-1})^{12} + 7(|\sigma|r_{\min}^{-1})^6\right)
|y-x|^2.$$
:::

::: proof
*Proof.* The shell second-derivative envelope gives a uniform Hessian bound on the positive shell. The one-dimensional second-order Taylor remainder is therefore bounded by one half of that shellwise Hessian envelope times the squared radial displacement. ◻
:::

::: theorem
[]{#thm:lj-shell-grad-stability label="thm:lj-shell-grad-stability"} Fix a positive radial shell $[r_{\min}, r_{\max}]$ and assume the Lennard-Jones gradient magnitude is bounded by $C$ on that shell. Then for any two radii $x,y \in [r_{\min}, r_{\max}]$, $$|U_{\mathrm{LJ}}(y)-U_{\mathrm{LJ}}(x)| \le C|y-x|.$$
:::

::: proof
*Proof.* The closed-form Lennard-Jones derivative is bounded by hypothesis on the positive shell. The one-dimensional mean value theorem then bounds variation of the potential by the derivative envelope times the radial displacement. ◻
:::

::: theorem
[]{#thm:lj-shell-uniform-approx label="thm:lj-shell-uniform-approx"} Let $d_{\mathrm{exact}}$ and $d_{\mathrm{grid}}$ be exact and discretized distance maps whose values remain inside a positive shell $[r_{\min}, r_{\max}]$. If the Lennard-Jones gradient magnitude is bounded by $C$ on that shell and $$|d_{\mathrm{exact}}-d_{\mathrm{grid}}| \le \delta$$ pointwise, then the exact and grid Lennard-Jones score families form a uniform utility approximation with radius $C\delta$.
:::

::: proof
*Proof.* The shell gradient theorem bounds score variation by $C$ times the radial displacement. Applying that bound pointwise to the exact-versus-grid distance error yields a uniform score discrepancy radius $C\delta$. ◻
:::

::: theorem
[]{#thm:lj-shell-quadratic-discretization label="thm:lj-shell-quadratic-discretization"} Let $d_{\mathrm{exact}}$ and $d_{\mathrm{grid}}$ be exact and discretized distance maps whose values remain inside a positive shell $[r_{\min}, r_{\max}]$, and assume $$|d_{\mathrm{exact}}-d_{\mathrm{grid}}| \le \delta$$ pointwise. Then the first-order-corrected exact/grid Lennard-Jones discrepancy satisfies the quadratic bound $$\left|
U_{\mathrm{LJ}}(d_{\mathrm{exact}})-
\left(U_{\mathrm{LJ}}(d_{\mathrm{grid}})+U'_{\mathrm{LJ}}(d_{\mathrm{grid}})(d_{\mathrm{exact}}-d_{\mathrm{grid}})\right)
\right|
\le
\frac{1}{2}
\left(24|\varepsilon|r_{\min}^{-2}\right)
\left(26(|\sigma|r_{\min}^{-1})^{12} + 7(|\sigma|r_{\min}^{-1})^6\right)
\delta^2.$$
:::

::: proof
*Proof.* Apply the quadratic Taylor remainder theorem to the shellwise exact-versus-grid radial displacement and then substitute the uniform distance-discretization bound $|d_{\mathrm{exact}}-d_{\mathrm{grid}}| \le \delta$. ◻
:::

::: theorem
[]{#thm:lj-shell-gap-invariance label="thm:lj-shell-gap-invariance"} In the setting of Theorem [\[thm:lj-shell-uniform-approx\]](#thm:lj-shell-uniform-approx){reference-type="ref" reference="thm:lj-shell-uniform-approx"}, if the radius $C\delta$ is smaller than half the strict exact utility gap at state $s$, then the exact and grid Lennard-Jones optimal-action sets agree at $s$.
:::

::: proof
*Proof.* The shell gradient theorem gives a uniform exact/grid score approximation radius $C\delta$. The strict half-gap criterion then yields exact/coarse winner preservation. ◻
:::

::: theorem
[]{#thm:bounded-potential-large-cutoff-srank label="thm:bounded-potential-large-cutoff-srank"} If a molecular docking problem satisfies a uniform outside-cutoff perturbation bound with coefficient $c$ and minimum exact decision gap $\gamma$, and if the chosen cutoff satisfies $$c\,\mathrm{TailSum}_6(R_{\mathrm{cut}}) < \gamma/2,$$ then the structural rank of the exact docking problem is bounded by $$\mathrm{srank}(D_{\mathrm{dock}}) \le 3P_{\mathrm{rel}} + 3L.$$
:::

::: proof
*Proof.* The large-cutoff condition converts the bounded-potential witness into the cutoff-boundedness hypothesis. The molecular locality theorem then bounds structural rank by the inside-cutoff protein coordinates together with all ligand coordinates. ◻
:::

::: theorem
[]{#thm:bounded-potential-large-cutoff-sampled-srank label="thm:bounded-potential-large-cutoff-sampled-srank"} Under the same bounded-potential and large-cutoff hypotheses, the sampled restricted docking problem also satisfies $$\mathrm{srank}(D_{\mathrm{dock}}^{\mathrm{sampled}}) \le 3P_{\mathrm{rel}} + 3L.$$
:::

::: proof
*Proof.* The large-cutoff condition again supplies the cutoff-boundedness hypothesis. The sampled cutoff locality theorem then bounds the structural rank of the restricted sampled problem by the same inside-cutoff protein coordinates together with all ligand coordinates. ◻
:::

Exact Lennard-Jones, exact Coulomb, and exact real-space Ewald scorer families each instantiate this sampled structural-rank bridge through their corresponding tail-control theorems in the mechanized development.

## Certified Pruning and Finite Algorithms

::: theorem
[]{#thm:finite-uniform-error-radius label="thm:finite-uniform-error-radius"} Every finite sampled docking problem admits a canonical exact/coarse discrepancy radius that witnesses uniform approximation between its exact and coarse score families on the sampled domain.
:::

::: proof
*Proof.* The finite sampled domain contains only finitely many exact-versus-coarse score differences. Their maximum gives the required uniform discrepancy radius. ◻
:::

::: theorem
[]{#thm:grid-docking-entropy label="thm:grid-docking-entropy"} Let $D^{\mathrm{flat}}_{\mathrm{dock,exact}}$ be the flat exact sampled docking decision problem obtained by reconstructing each discretized molecular grid state from its finite coordinate function on a grid of side length $2N+1$. Then $$H_{\mathrm{nats}}\!\bigl(D^{\mathrm{flat}}_{\mathrm{dock,exact}}\bigr)
\le
\mathrm{srank}\!\bigl(D^{\mathrm{flat}}_{\mathrm{dock,exact}}\bigr)\,\ln(2N+1).$$
:::

::: proof
*Proof.* The flat grid state family has a finite alphabet of size $2N+1$ at each coordinate. The number of exact optimizer classes is therefore bounded by $(2N+1)^{\mathrm{srank}}$, so taking natural logarithms yields the displayed entropy bound. ◻
:::

::: theorem
[]{#thm:topk-certificate-sound label="thm:topk-certificate-sound"} Under a certified top-$k$ margin condition, the theorem-backed pruning certificate retains every exact top-$k$ action inside its survivor set.
:::

::: proof
*Proof.* The certificate packages the top-$k$ survivor containment theorem into a finite survivor set together with its soundness proof. The soundness clause is exactly the stated inclusion. ◻
:::

::: theorem
[]{#thm:grid-erase-irrelevant label="thm:grid-erase-irrelevant"} For discretized molecular grid states, any coordinate already known to be irrelevant may be erased from a sufficient retained set without destroying exact sufficiency.
:::

::: proof
*Proof.* Irrelevant coordinates do not change the optimizer when toggled. Erasing one from a retained sufficient set therefore preserves the exact optimal-action correspondence. ◻
:::

::: theorem
[]{#thm:bounded-actions-poly label="thm:bounded-actions-poly"} If the action family has size at most $k$, then exact sufficiency checking is decidable in polynomial time. More precisely, the total checking cost is bounded by $$|S|^2 (1 + k^2).$$
:::

::: proof
*Proof.* With at most $k$ actions, each optimal-action comparison costs at most quadratic time in $k$, and sufficiency checking examines all state pairs. The resulting total cost is therefore bounded by $|S|^2(1+k^2)$. ◻
:::

These theorems convert exact-resolution stability into explicit finite certificates and polynomial-time checking regimes.

## Concrete Scorer Families

::: theorem
[]{#thm:lj-cutoff-invariance label="thm:lj-cutoff-invariance"} For a finite sampled Lennard-Jones docking family, let $a_\ast$ be a strict exact winner at state $s$. If the finite cutoff error radius is smaller than half the strict exact utility gap at $s$, then the exact and cutoff Lennard-Jones optimal-action sets agree at $s$.
:::

::: proof
*Proof.* The finite cutoff radius gives a uniform approximation theorem for exact and cutoff Lennard-Jones scores on the sampled domain. A strict half-gap bound then forces winner preservation. ◻
:::

::: theorem
[]{#thm:lj-tail-srank label="thm:lj-tail-srank"} If an exact Lennard-Jones docking score admits a physical distance-decay tail bound, if the finite exact decision gap is positive, and if the cutoff radius is large enough that the tail term is below half that gap, then $$\mathrm{srank}(D_{\mathrm{dock}}) \le 3P_{\mathrm{rel}} + 3L.$$
:::

::: proof
*Proof.* The physical distance-decay theorem packages the exact Lennard-Jones score into a bounded-potential witness. The large-cutoff theorem converts that witness into the cutoff-boundedness hypothesis, and the molecular locality theorem then gives the structural-rank bound. ◻
:::

::: theorem
[]{#thm:coulomb-cutoff-uniform-approx label="thm:coulomb-cutoff-uniform-approx"} For a finite sampled Coulomb docking family, the exact and cutoff Coulomb score families differ by at most the finite cutoff error radius uniformly over the sampled action-state domain.
:::

::: proof
*Proof.* The cutoff error radius is defined as the maximum exact-versus-cutoff discrepancy over the finite sampled domain. Uniform approximation follows immediately from that extremal definition. ◻
:::

::: theorem
[]{#thm:coulomb-cutoff-invariance label="thm:coulomb-cutoff-invariance"} For a finite sampled Coulomb docking family, let $a_\ast$ be a strict exact winner at state $s$. If the finite cutoff error radius is smaller than half the strict exact utility gap at $s$, then the exact and cutoff Coulomb optimal-action sets agree at $s$.
:::

::: proof
*Proof.* The uniform cutoff-error bound from Theorem [\[thm:coulomb-cutoff-uniform-approx\]](#thm:coulomb-cutoff-uniform-approx){reference-type="ref" reference="thm:coulomb-cutoff-uniform-approx"} combines with the strict half-gap criterion to force winner preservation. ◻
:::

::: theorem
[]{#thm:coulomb-tail-srank label="thm:coulomb-tail-srank"} If an exact Coulomb docking score admits a tail perturbation bound, if the finite exact decision gap is positive, and if the cutoff radius is large enough that the tail term is below half that gap, then $$\mathrm{srank}(D_{\mathrm{dock}}) \le 3P_{\mathrm{rel}} + 3L.$$
:::

::: proof
*Proof.* The Coulomb tail theorem packages the exact score into a bounded-potential witness. The large-cutoff theorem converts that witness into the cutoff-boundedness hypothesis, and the molecular locality theorem then gives the structural-rank bound. ◻
:::

::: theorem
[]{#thm:ewald-tail-srank label="thm:ewald-tail-srank"} If an exact real-space Ewald docking score admits a tail perturbation bound, if the finite exact decision gap is positive, and if the cutoff radius is large enough that the tail term is below half that gap, then $$\mathrm{srank}(D_{\mathrm{dock}}) \le 3P_{\mathrm{rel}} + 3L.$$
:::

::: proof
*Proof.* The real-space Ewald tail theorem packages the exact score into a bounded-potential witness. The large-cutoff theorem converts that witness into the cutoff-boundedness hypothesis, and the molecular locality theorem then gives the structural-rank bound. ◻
:::

::: theorem
[]{#thm:lj-gradient label="thm:lj-gradient"} For nonzero separation radius $r$, the Lennard-Jones potential admits the exact derivative $$\frac{d}{dr}U_{\mathrm{LJ}}(r)
=
\left(24\varepsilon r^{-1}\right)
\left((\sigma r^{-1})^6 - 2(\sigma r^{-1})^{12}\right).$$
:::

::: proof
*Proof.* Differentiate the closed-form Lennard-Jones expression directly by the chain rule through the inverse-power representation and simplify algebraically. ◻
:::

## Geometric and Electrostatic Structure

::: theorem
[]{#thm:velocity-verlet-volume label="thm:velocity-verlet-volume"} For every differentiable potential, timestep, mass, and phase-space state, the Jacobian determinant of the Velocity-Verlet map is exactly $1$.
:::

::: proof
*Proof.* The Jacobian factors into three block-triangular steps with unit diagonal determinants. Multiplicativity of determinant then gives exact phase-space volume preservation. ◻
:::

::: theorem
[]{#thm:ewald-real-space-decay label="thm:ewald-real-space-decay"} For positive radius $r$ and splitting parameter $\alpha$, the Ewald real-space core satisfies $$\mathrm{Ewald}_{\mathrm{real}}(r,\alpha)
\le
\frac{e^{-\alpha^2 r^2}}{r}.$$
:::

::: proof
*Proof.* The formal real-space core is defined by the Gaussian upper bound on the complementary-error screening factor. Expanding the square gives the displayed exponential-decay form exactly. ◻
:::

::: theorem
[]{#thm:ewald-fourier-positive label="thm:ewald-fourier-positive"} For positive wavevector magnitude $k$ and splitting parameter $\alpha$, the Ewald reciprocal-space core is strictly positive.
:::

::: proof
*Proof.* The reciprocal-space core is the product of $k^{-2}$ and a positive Gaussian factor. Both factors are positive under the stated hypotheses. ◻
:::

These geometric and electrostatic theorems show that the docking layer already contains exact statements about phase-space evolution, force-law differentiation, and long-range Coulomb splitting together with the rank and approximation bounds.

## Formalization

The docking bridge theorems are assembled in `Leverage/DockingTheoryBridge.lean`. They expose the abstraction-collapse boundary, the Fisher-rank identities, the general exact-sufficiency hardness core, the quantitative witness/checking lower bounds, the cutoff-local structural-rank bounds for molecular docking, the conformer-search collapse regimes, the sampled exact/coarse preservation and sufficiency theorems, the top-$k$ and ambiguity-band control theorems, the concrete Lennard-Jones and Coulomb cutoff invariance statements, and the geometric and electrostatic structure theorems used in the molecular development.


# Thermodynamic Cost of Exact Molecular Docking {#main-theorems}

The preceding sections fixed the exact object, its unavoidable quotient boundary, its structural and Fisher dimensions, and its certification burden. The thermodynamic theorems of this section convert that same exact-resolution spine into cost. The abstract statements hold for bounded decision systems, and the constrained-molecular corollaries transport them to holonomic topologies and binding-resolution problems. Landauer furnishes the universal floor for the conversion constant.

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

Informally: rank one is the ground state.

## Energy--Information Duality

::: theorem
[]{#thm:energy-entropy label="thm:energy-entropy"} Let $D = \mathrm{canonicalDP}(A)$, and let $E$ be the realized energy for one exact-resolution cycle. If Landauer calibration holds at positive Boltzmann constant and temperature, then $$E \ge k_B T\, H_{\mathrm{nats}}(D).$$ Equivalently, the minimum exact-resolution cost is at least $k_B T$ times the natural-log entropy of the decision quotient.
:::

::: proof
*Proof.* The entropy-rank inequality gives $$H_{\mathrm{nats}}(D) \le \mathrm{DOF}(A)\ln 2$$ by Theorem [\[thm:entropy-bound\]](#thm:entropy-bound){reference-type="ref" reference="thm:entropy-bound"}. Theorem [\[thm:energy-rank\]](#thm:energy-rank){reference-type="ref" reference="thm:energy-rank"} gives the complementary lower bound $$E \ge \mathrm{DOF}(A) k_B T \ln 2.$$ Comparing the two right-hand sides yields the announced inequality. ◻
:::

Theorem [\[thm:energy-entropy\]](#thm:energy-entropy){reference-type="ref" reference="thm:energy-entropy"} composes the entropy-rank inequality with the per-bit lower bound from Theorem [\[thm:energy-rank\]](#thm:energy-rank){reference-type="ref" reference="thm:energy-rank"}.

Informally: the quotient fixing correctness also fixes cost.

::: corollary
[]{#cor:minimum-cost-regime label="cor:minimum-cost-regime"} Among bounded decision systems in the canonical binary encoding, the unique minimum-cost regime is $\mathrm{DOF}(A)=1$.
:::

::: proof
*Proof.* Theorem [\[thm:rank-one-ground\]](#thm:rank-one-ground){reference-type="ref" reference="thm:rank-one-ground"} identifies $\mathrm{DOF}(A)=1$ as the one-Landauer-unit ground state, while every bounded decision system with more than one degree of freedom lies strictly above it. ◻
:::

## Exact Docking Decision Problems

::: theorem
[]{#thm:md-energy-rank label="thm:md-energy-rank"} Let $D_{\mathrm{dock}}$ be the exact docking decision problem induced by a binding problem, let $I$ be a sufficient coordinate set for $D_{\mathrm{dock}}$, and let $M$ be a thermodynamic model with positive per-bit conversion constant. Then $$M.\mathrm{joulesPerBit}\cdot \mathrm{srank}(D_{\mathrm{dock}})
\le
\mathrm{energyLowerBound}(M,|I|).$$
:::

::: proof
*Proof.* This is the generic structural-rank energy lower bound applied directly to the exact docking decision problem. Exact docking therefore enters the thermodynamic layer through its own sufficient-coordinate structure, not only through the canonical binary encoding. ◻
:::

::: theorem
[]{#thm:md-energy-entropy label="thm:md-energy-entropy"} Let $D^{\mathrm{flat}}_{\mathrm{dock,exact}}$ be the flat finite-grid exact docking problem induced by a sampled docking family, and let $$D^{\mathrm{bits}}_{\mathrm{dock,exact}}$$ be its fixed-length binary encoding. Let $I$ be a sufficient bit-register set for $D^{\mathrm{bits}}_{\mathrm{dock,exact}}$, and let $M$ satisfy Landauer calibration at positive Boltzmann constant and temperature. Then $$\mathrm{energyLowerBound}(M,|I|)
\ge
k_B T\,H_{\mathrm{nats}}\!\bigl(D^{\mathrm{flat}}_{\mathrm{dock,exact}}\bigr).$$ Equivalently, once the finite grid state is lifted to an exact fixed-length binary register, the Landauer floor applies directly to the entropy of the original finite docking quotient without any extra entropy-versus-rank hypothesis.
:::

::: proof
*Proof.* The binary-encoded problem has binary state space, so its optimizer entropy is bounded by its structural rank through the native binary entropy theorem. The fixed-length decoder preserves the optimizer classes of the original flat finite-grid docking problem, so both problems have the same quotient entropy. The binary structural-rank energy theorem therefore yields the displayed lower bound for the finite docking quotient itself. ◻
:::

::: theorem
[]{#thm:md-above-ground label="thm:md-above-ground"} Let $D_{\mathrm{dock}}$ be an exact docking decision problem induced by a binding problem, let $I$ be a sufficient coordinate set for $D_{\mathrm{dock}}$, and let $M$ be a thermodynamic model with positive per-bit conversion constant. If $$\mathrm{srank}(D_{\mathrm{dock}}) > 1,$$ then $$M.\mathrm{joulesPerBit} < \mathrm{energyLowerBound}(M,|I|).$$
:::

::: proof
*Proof.* Structural rank above one places the exact docking problem strictly above the one-bit ground state. The structural-rank energy lower bound then transfers that strict separation to every sufficient-set exact-resolution cycle. ◻
:::

The thermodynamic bridge for docking is therefore split into two honest layers. Theorem [\[thm:md-energy-rank\]](#thm:md-energy-rank){reference-type="ref" reference="thm:md-energy-rank"} is the direct exact-docking structural-rank cost floor once a sufficient coordinate set is fixed. Theorem [\[thm:md-energy-entropy\]](#thm:md-energy-entropy){reference-type="ref" reference="thm:md-energy-entropy"} is the entropy-calibrated Landauer statement for the explicit finite binary encoding of the sampled flat grid problem. Theorem [\[thm:grid-docking-entropy\]](#thm:grid-docking-entropy){reference-type="ref" reference="thm:grid-docking-entropy"} separately bounds the same finite-grid quotient entropy in the original $(2N+1)$-ary coordinate presentation. The cutoff-local theorems from Section [\[complexity-boundary\]](#complexity-boundary){reference-type="ref" reference="complexity-boundary"} still provide only upper bounds on $\mathrm{srank}(D_{\mathrm{dock}})$; those bounds identify low-rank exact regimes, but they do not by themselves produce a coordinate-explicit lower bound.

::: theorem
[]{#thm:sampled-inside-cutoff-budget-energy label="thm:sampled-inside-cutoff-budget-energy"} Let $D^{\mathrm{sampled}}_{\mathrm{dock,exact}}$ be the sampled restricted exact docking problem. Assume the inside-cutoff retained coordinate set is sufficient for $D^{\mathrm{sampled}}_{\mathrm{dock,exact}}$, and suppose that retained set has cardinality at most the bounded-region acquisition budget $\mathrm{MaxAcquisitions}(R,T)$. Then for every thermodynamic model with positive per-bit conversion constant, $$M.\mathrm{joulesPerBit}\cdot \mathrm{srank}\!\bigl(D^{\mathrm{sampled}}_{\mathrm{dock,exact}}\bigr)
\le
\mathrm{energyLowerBound}\!\bigl(M,\mathrm{MaxAcquisitions}(R,T)\bigr).$$
:::

::: proof
*Proof.* The inside-cutoff theorem supplies a concrete sufficient retained set. The budget hypothesis states that this retained set fits inside the bounded-region acquisition budget. The generic structural-rank energy theorem then applies to that exact-resolution witness and lifts the witness cardinality bound into the displayed bounded-region energy floor. ◻
:::

## Finite-Time and Budget Bounds

::: theorem
[]{#thm:time-lower-bound label="thm:time-lower-bound"} Let $A$ be a bounded decision system, and let $I$ be a sufficient coordinate set for $\mathrm{canonicalDP}(A)$. Suppose $A$ is resolved inside a bounded region of diameter $d$ and signal speed $c$ over operating horizon $\tau$, and suppose $$|I| \le \frac{c\tau}{d}.$$ Then $$\mathrm{DOF}(A) \le \frac{c\tau}{d}.$$
:::

::: proof
*Proof.* Theorem [\[thm:min-bit-operations\]](#thm:min-bit-operations){reference-type="ref" reference="thm:min-bit-operations"} gives a lower bound of $\mathrm{DOF}(A)$ elementary acquisition events for exact resolution. Theorem [\[thm:bounded-acquisition\]](#thm:bounded-acquisition){reference-type="ref" reference="thm:bounded-acquisition"} bounds the total number of acquisition events on horizon $\tau$ by $c\tau/d$. Therefore exact resolution on that horizon requires $\mathrm{DOF}(A) \le c\tau/d$. ◻
:::

::: theorem
[]{#thm:budget-class-bound label="thm:budget-class-bound"} Let $D = \mathrm{canonicalDP}(A)$, and let $I$ be a sufficient coordinate set for $D$. Suppose $$|I| \le \frac{c\tau}{d}$$ inside a bounded region of diameter $d$ and signal speed $c$ over operating horizon $\tau$, and let $E$ satisfy $$E \ge \mathrm{DOF}(A)\,k_B \Theta \ln 2
\qquad (\Theta > 0).$$ Then $$\mathrm{numOptClasses}(D) \le 2^{c\tau/d}
\qquad\text{and}\qquad
\mathrm{numOptClasses}(D) \le \exp\!\left(\frac{E}{k_B \Theta}\right).$$ Consequently, $$\mathrm{numOptClasses}(D) \le
\min\!\left(2^{c\tau/d},\ \exp\!\left(\frac{E}{k_B \Theta}\right)\right).$$
:::

::: proof
*Proof.* By Theorem [\[thm:time-lower-bound\]](#thm:time-lower-bound){reference-type="ref" reference="thm:time-lower-bound"}, exact resolution on horizon $\tau$ requires $\mathrm{DOF}(A) \le c\tau/d$. Theorem [\[thm:numopt-bound\]](#thm:numopt-bound){reference-type="ref" reference="thm:numopt-bound"} gives $$\mathrm{numOptClasses}(D) \le 2^{\mathrm{DOF}(A)} \le 2^{c\tau/d}.$$ Theorem [\[thm:energy-entropy\]](#thm:energy-entropy){reference-type="ref" reference="thm:energy-entropy"} gives $$E \ge k_B \Theta\, H_{\mathrm{nats}}(D) = k_B \Theta \ln(\mathrm{numOptClasses}(D)).$$ Hence $$\ln(\mathrm{numOptClasses}(D)) \le \frac{E}{k_B \Theta},$$ which is equivalent to $$\mathrm{numOptClasses}(D) \le \exp\!\left(\frac{E}{k_B \Theta}\right).$$ Taking the smaller of the two upper bounds gives the final statement. ◻
:::

::: corollary
[]{#cor:budget-entropy-bound label="cor:budget-entropy-bound"} Let $D = \mathrm{canonicalDP}(A)$, and let $I$ be a sufficient coordinate set for $D$. Suppose $$|I| \le \frac{c\tau}{d}$$ inside a bounded region of diameter $d$ and signal speed $c$ over operating horizon $\tau$, and let $E$ satisfy $$E \ge \mathrm{DOF}(A)\,k_B \Theta \ln 2
\qquad (\Theta > 0).$$ Then $$H_{\mathrm{bits}}(D) \le \frac{c\tau}{d}
\qquad\text{and}\qquad
H_{\mathrm{nats}}(D) \le
\min\!\left(\frac{c\tau}{d}\ln 2,\ \frac{E}{k_B \Theta}\right).$$
:::

::: proof
*Proof.* Theorem [\[thm:time-lower-bound\]](#thm:time-lower-bound){reference-type="ref" reference="thm:time-lower-bound"} gives $\mathrm{DOF}(A) \le c\tau/d$. Theorem [\[thm:entropy-bound\]](#thm:entropy-bound){reference-type="ref" reference="thm:entropy-bound"} gives $$H_{\mathrm{bits}}(D) \le \mathrm{DOF}(A),
\qquad
H_{\mathrm{nats}}(D) \le \mathrm{DOF}(A)\ln 2.$$ Hence $$H_{\mathrm{bits}}(D) \le \frac{c\tau}{d},
\qquad
H_{\mathrm{nats}}(D) \le \frac{c\tau}{d}\ln 2.$$ Theorem [\[thm:energy-entropy\]](#thm:energy-entropy){reference-type="ref" reference="thm:energy-entropy"} also gives $$H_{\mathrm{nats}}(D) \le \frac{E}{k_B \Theta}.$$ Taking the smaller of the two nat-valued upper bounds gives the final statement. ◻
:::

::: corollary
[]{#cor:composition-budget-law label="cor:composition-budget-law"} Let $A_1$ and $A_2$ be disjoint bounded decision systems. Suppose the composite system $A_1 \oplus A_2$ is resolved inside a bounded region of diameter $d$ and signal speed $c$ over operating horizon $\tau$, and suppose some sufficient coordinate set for $\mathrm{canonicalDP}(A_1 \oplus A_2)$ has cardinality at most $c\tau/d$. Then $$\mathrm{DOF}(A_1)+\mathrm{DOF}(A_2) \le \frac{c\tau}{d}$$ and for any thermodynamic model with positive per-bit conversion constant, $$\mathrm{joulesPerBit}\cdot\bigl(\mathrm{DOF}(A_1)+\mathrm{DOF}(A_2)\bigr)
\le
\mathrm{energyLowerBound}\!\left(M,\frac{c\tau}{d}\right).$$
:::

::: proof
*Proof.* Proposition [\[prop:dof-additive\]](#prop:dof-additive){reference-type="ref" reference="prop:dof-additive"} gives $$\mathrm{DOF}(A_1 \oplus A_2) = \mathrm{DOF}(A_1)+\mathrm{DOF}(A_2).$$ Apply Theorem [\[thm:time-lower-bound\]](#thm:time-lower-bound){reference-type="ref" reference="thm:time-lower-bound"} and Theorem [\[thm:energy-rank\]](#thm:energy-rank){reference-type="ref" reference="thm:energy-rank"} to the composite system and substitute the additive degree-of-freedom identity. ◻
:::

## Worked Examples

Two toy canonical systems fix the scale of the bound.

#### One coordinate in the canonical encoding.

Let $A$ satisfy $\mathrm{DOF}(A)=1$, and write $D=\mathrm{canonicalDP}(A)$. Then the state space of $D$ has two states. The false state has optimal set $\{\mathrm{inr}(\star)\}$, and the true state has optimal set $\{\mathrm{inl}(0)\}$. The decision quotient therefore has two classes, $H_{\mathrm{nats}}(D)=\ln 2$, and Theorem [\[thm:energy-entropy\]](#thm:energy-entropy){reference-type="ref" reference="thm:energy-entropy"} gives $$E \ge k_B T \ln 2.$$ This is the rank-$1$ ground regime.

#### Two coordinates in the canonical encoding.

Let $A$ satisfy $\mathrm{DOF}(A)=2$, and write $D=\mathrm{canonicalDP}(A)$. Then the four states of $D$ have optimal sets $\{\mathrm{inr}(\star)\}$, $\{\mathrm{inl}(0)\}$, $\{\mathrm{inl}(1)\}$, and $\{\mathrm{inl}(0),\mathrm{inl}(1)\}$. The decision quotient therefore has four classes, $H_{\mathrm{nats}}(D)=\ln 4 = 2\ln 2$, and Theorem [\[thm:energy-entropy\]](#thm:energy-entropy){reference-type="ref" reference="thm:energy-entropy"} gives $$E \ge 2 k_B T \ln 2.$$ Relative to the one-coordinate case, the minimum exact-resolution cost doubles.

## Concrete Substrate Instantiation

::: proposition
[]{#prop:threshold-channel label="prop:threshold-channel"} Fix a threshold $\tau$ and a sampled substrate observable $x_t \in \mathbb{R}$. The induced readout bit $$b_t = \mathbf{1}[x_t \ge \tau]$$ has binary state space $\{0,1\}$. A readout flip $b_{t+1} \ne b_t$ is equivalent to a positive one-bit lower bound, and under a positive per-bit conversion constant it implies a positive energy lower bound.
:::

::: proposition
[]{#prop:atomic-realization label="prop:atomic-realization"} Let $c_0$ and $c_1$ be atomic configurations with distinct orbital occupancies or distinct energies. Then $c_0 \ne c_1$. Upward transitions from $c_0$ to $c_1$ require positive energy input, and downward transitions release positive energy. A choice of labels $c_0 \mapsto 0$ and $c_1 \mapsto 1$ therefore gives a physical binary readout layer.
:::

Thresholded one-bit readouts and two-level atomic transitions instantiate the same binary interface [@berut2012experimental; @planck1901distribution; @dirac1930principles; @sakurai2017modern]. A $k$-channel substrate has joint readout state in $\{0,1\}^k$, and the canonical state space $\mathrm{Fin}\;k \to \mathrm{Bool}$ is the same object written in indexed form.

## Substrate Time Law

::: proposition
[]{#prop:substrate-time-law label="prop:substrate-time-law"} For any substrate model whose observed interface obeys decision ticks, every one-step substrate evolution realizes a decision event and advances interface time by one unit. The tick law is independent of substrate tag.
:::

## Strict Overhead Above Landauer

::: proposition
[]{#prop:strict-overhead label="prop:strict-overhead"} Let $W$ be a decomposed process model. If the mismatch term is instantiated by a theorem-level distribution-mismatch witness, then the effective per-bit lower bound of $W$ is strictly above the Landauer floor. If the residual term is instantiated by a theorem-level finite discrete residual witness, the same strict inequality holds. For any sufficient coordinate set $I$ of $\mathrm{canonicalDP}(A)$, either branch therefore yields an exact-resolution energy lower bound strictly above $\mathrm{DOF}(A)\,k_B T \ln 2$.
:::

::: proposition
[]{#prop:finite-discrete-residual label="prop:finite-discrete-residual"} Let a finite computational-state process admit a positive forward edge together with decision-relevant asymmetry. Then the theorem-level discrete residual lower bound is positive. If this witness is used as the residual term of a decomposed process model, the effective per-bit lower bound is strictly above the Landauer floor.
:::

::: proposition
[]{#prop:binary-residual-example label="prop:binary-residual-example"} There exists a two-state irreversible residual witness with one positive forward edge and zero reverse edge. The induced residual lower-bound term is exactly one nat-valued overhead unit. Any decomposed process model that uses this witness as its residual term therefore satisfies $$\mathrm{landauerJoulesPerBit}(k_B,T) + 1
\le
W.\mathrm{effectiveModel}.\mathrm{joulesPerBit},$$ and for any sufficient coordinate set $I$ of $\mathrm{canonicalDP}(A)$, $$\mathrm{DOF}(A)\,(k_B T \ln 2 + 1)
\le
\mathrm{energyLowerBound}(W.\mathrm{effectiveModel},|I|).$$ The same example yields the strengthened energy--information inequality $$\frac{k_B T \ln 2 + 1}{\ln 2}
\, H_{\mathrm{nats}}(\mathrm{canonicalDP}(A))
\le
\mathrm{energyLowerBound}(W.\mathrm{effectiveModel},|I|).$$
:::

::: proposition
[]{#prop:binary-residual-cumulative-work label="prop:binary-residual-cumulative-work"} For any $m \in \mathbb{N}$, repeated exact-resolution cycles under the same explicit two-state residual witness satisfy $$m\,\frac{k_B T \ln 2 + 1}{\ln 2}
\, H_{\mathrm{nats}}(\mathrm{canonicalDP}(A))
\le
m\,\mathrm{energyLowerBound}(W.\mathrm{effectiveModel},|I|).$$ The required cumulative work therefore grows linearly with cycle count.
:::

::: proposition
[]{#prop:ei-hierarchy label="prop:ei-hierarchy"} Let $I$ be a sufficient coordinate set for $\mathrm{canonicalDP}(A)$. Then the ideal Landauer-calibrated floor satisfies $$k_B T\,H_{\mathrm{nats}}(\mathrm{canonicalDP}(A))
\le
\mathrm{energyLowerBound}(M,|I|)$$ whenever the declared model $M$ dominates the Landauer floor. Under the explicit binary mismatch example and the explicit two-state residual example, the strengthened coefficient $$\frac{k_B T \ln 2 + 1}{\ln 2}$$ replaces $k_B T$: $$\frac{k_B T \ln 2 + 1}{\ln 2}
\,H_{\mathrm{nats}}(\mathrm{canonicalDP}(A))
\le
\mathrm{energyLowerBound}(W_{\mathrm{mm}}.\mathrm{effectiveModel},|I|),$$ $$\frac{k_B T \ln 2 + 1}{\ln 2}
\,H_{\mathrm{nats}}(\mathrm{canonicalDP}(A))
\le
\mathrm{energyLowerBound}(W_{\mathrm{res}}.\mathrm{effectiveModel},|I|).$$
:::

::: proposition
[]{#prop:structural-resource-overhead label="prop:structural-resource-overhead"} Let $W$ be a decomposed process model and let $r$ be a declared structural resource. If $r$ is lower-bounded by the mismatch term, then the effective per-bit lower bound dominates the Landauer floor plus $r$. For any sufficient coordinate set $I$ of $\mathrm{canonicalDP}(A)$, $$\mathrm{energyLowerBound}(W.\mathrm{base},|I|) + r\,\mathrm{DOF}(A)
\le
\mathrm{energyLowerBound}(W.\mathrm{effectiveModel},|I|).$$
:::

::: proposition
[]{#prop:canonical-wolpert-bundle label="prop:canonical-wolpert-bundle"} Let $I$ be a nonempty sufficient coordinate set for $\mathrm{canonicalDP}(A)$, and let $W$ be a decomposed process model whose base lower bound dominates the Landauer floor. Then $$\mathrm{DOF}(A) \le |I|,$$ $$W.\mathrm{effectiveModel}.\mathrm{joulesPerBit}\cdot\mathrm{DOF}(A)
\le
\mathrm{energyLowerBound}(W.\mathrm{effectiveModel},|I|),$$ and $$0 < \mathrm{energyLowerBound}(W.\mathrm{effectiveModel},|I|).$$
:::

::: proposition
[]{#prop:strict-canonical-energy label="prop:strict-canonical-energy"} Let $I$ be a sufficient coordinate set for $\mathrm{canonicalDP}(A)$, and let $W$ be a decomposed process model whose base lower bound dominates the Landauer floor. If either theorem-level Wolpert branch applies to $W$, then $$\mathrm{DOF}(A)\,k_B T \ln 2
<
\mathrm{energyLowerBound}(W.\mathrm{effectiveModel},|I|).$$
:::

::: proposition
[]{#prop:binary-mismatch-example label="prop:binary-mismatch-example"} Fix the actual input distribution $$p(1)=\tfrac34,
\qquad
p(0)=\tfrac14,$$ and the designed distribution $$q(1)=\tfrac14,
\qquad
q(0)=\tfrac34.$$ The induced mismatch lower-bound term is at least one nat-valued overhead unit. Any decomposed process model that uses this witness as its mismatch term therefore satisfies $$\mathrm{landauerJoulesPerBit}(k_B,T) + 1
\le
W.\mathrm{effectiveModel}.\mathrm{joulesPerBit},$$ and for any sufficient coordinate set $I$ of $\mathrm{canonicalDP}(A)$, $$\mathrm{DOF}(A)\,(k_B T \ln 2 + 1)
\le
\mathrm{energyLowerBound}(W.\mathrm{effectiveModel},|I|).$$
:::

::: proposition
[]{#prop:binary-mismatch-energy-information label="prop:binary-mismatch-energy-information"} Under the same explicit binary mismatch witness, $$\frac{k_B T \ln 2 + 1}{\ln 2}
\, H_{\mathrm{nats}}(\mathrm{canonicalDP}(A))
\le
\mathrm{energyLowerBound}(W.\mathrm{effectiveModel},|I|).$$ The coefficient is strictly larger than $k_B T$.
:::

::: proposition
[]{#prop:binary-mismatch-cumulative-work label="prop:binary-mismatch-cumulative-work"} For any $m \in \mathbb{N}$, repeated exact-resolution cycles under the same explicit binary mismatch witness satisfy $$m\,\frac{k_B T \ln 2 + 1}{\ln 2}
\, H_{\mathrm{nats}}(\mathrm{canonicalDP}(A))
\le
m\,\mathrm{energyLowerBound}(W.\mathrm{effectiveModel},|I|).$$ The required cumulative work therefore grows linearly with cycle count.
:::

## Cyclewise Heat and Lifetime

::: proposition
[]{#prop:finite-lifetime label="prop:finite-lifetime"} In the substrate heat-capacity model, every computational cycle generates positive heat, cumulative heat grows linearly with cycle count, heat above capacity causes degradation, and finite integrity together with finite heat capacity yields bounded lifetime.
:::

::: proposition
[]{#prop:lifetime-throughput label="prop:lifetime-throughput"} Let $s$ be a finite substrate with lifetime ceiling $\mathrm{maxCycles}(s)$. For any run of $m$ exact-resolution cycles with $$m \le \mathrm{maxCycles}(s),$$ the cumulative nat-valued decision entropy processed by $\mathrm{canonicalDP}(A)$ satisfies $$m\,H_{\mathrm{nats}}(\mathrm{canonicalDP}(A))
\le
\mathrm{maxCycles}(s)\,\mathrm{DOF}(A)\ln 2.$$
:::

::: proposition
[]{#prop:speed-heat-tradeoff label="prop:speed-heat-tradeoff"} In the same substrate model, faster computation yields a larger heat rate. Once heat rate exceeds substrate capacity, faster computation yields faster degradation.
:::

## Constrained Molecular Application

::: corollary
[]{#cor:holonomic-landauer-floor label="cor:holonomic-landauer-floor"} Let $X$ be a finite RATTLE holonomic topology with $N$ atoms and $k$ independent constraints, where each constraint check is recorded as a binary satisfied/violated status. Then the full constraint-status observation space has cardinality $$2^k.$$ Let $A_X$ be the transported bounded decision system with $$\mathrm{DOF}(A_X) = 3N-k.$$ Then the canonical exact-resolution problem satisfies $$\mathrm{srank}(\mathrm{canonicalDP}(A_X)) = 3N-k.$$ Moreover, for any sufficient coordinate set $I$ for $\mathrm{canonicalDP}(A_X)$ and any thermodynamic model with positive per-bit conversion constant, $$M.\mathrm{joulesPerBit}\cdot(3N-k)
\le
\mathrm{energyLowerBound}(M,|I|).$$ In particular, the per-cycle exact-resolution floor scales linearly with the unconstrained molecular dimension.
:::

::: proof
*Proof.* The RATTLE holonomic status register is a $k$-bit binary interface by the finite cardinality theorem. The transported architecture has degree of freedom exactly $3N-k$ by construction. The bridge theorem identifies the structural rank of the canonical exact-resolution problem with that same count, and the energy lower bound then gives the displayed Landauer-linear floor. ◻
:::

Informally: matter pays for what its topology requires it to know.

## Optimal-Transport Witness

The Landauer route lower-bounds exact resolution through irreversible bit acquisition. A complementary witness measures separation between future distributions on the integrity space $\{\mathrm{intact},\mathrm{compromised}\}$. It supplies an independent transport-theoretic signal that multiple distinguishable futures have nonzero cost.

::: remark
[]{#rem:wasserstein-bridge label="rem:wasserstein-bridge"} The same separation admits an independent transport-cost witness on the two-state integrity space $\{\mathrm{intact},\mathrm{compromised}\}$. The diagonal coupling has zero transport cost in the single-future regime (W1). Any coupling with off-diagonal mass has positive transport cost (W2). If the intact future mass dominates the compromised future mass, the intact state minimizes total transport to the future distribution (W3). If both future states carry positive mass, then transport from either pure state is strictly positive (W4). Multiple distinguishable futures therefore force positive transport cost independently of the Landauer route.
:::

A transport witness and the Landauer witness emphasize different structures: one counts irreducible coordinate reads, while the other measures geometric separation of future mass. In the two-state integrity model they point in the same direction, since a single future has zero transport cost whereas genuinely split futures force strictly positive transport cost.

## Interpretation

If degree of freedom is read as the number of independent physical coordinates that can vary separately, then lower DOF means lower exact-resolution cost because fewer independent coordinates must be resolved. The constrained-molecular corollary makes that transport explicit for the finite count $3N-k$.

## Formalization

The bridge from degree of freedom to structural rank is formalized in `Leverage/BridgeToDQ.lean`, including the direct finite RATTLE transport with effective dimension $3N-k$. The finite holonomic-constraint counting layer lives in `Computation/GeometricConstraints.lean`. The physical acquisition and Landauer theorems are imported from the decision-quotient physics stack, in particular `Physics/BoundedAcquisition.lean` and `ThermodynamicLift.lean`. The `Architecture` object in this section provides the coordinate count transported into those theorems.


# Convergence and Universal Consequences {#five-way-equivalence}

Degree of freedom equals structural rank, structural rank fixes quotient entropy and Fisher dimension, and the same rank controls exact-certification burden and thermodynamic cost. The molecular instantiations above make that chain concrete for constrained molecular systems. The remaining theorems in this section record the universal consequences of that chain once the docking specialization has already been made explicit.

## Imported Coherence Reading

::: theorem
[]{#thm:coherent-single-source label="thm:coherent-single-source"} A bounded decision system lies in the coherent unit-independent-rate regime if and only if $\mathrm{DOF}(A)=1$.
:::

::: remark
In the imported coherence development, rank $1$ means exactly one locus is authoritative, every remaining encoding is a derived view, and all reachable states remain coherent.
:::

## Structural Rank

::: theorem
[]{#thm:rank-identification label="thm:rank-identification"} For every bounded decision system $A$, $$\mathrm{srank}(\mathrm{canonicalDP}(A)) = \mathrm{DOF}(A).$$
:::

## Tractability Boundary

::: theorem
[]{#thm:tractable-rank-one label="thm:tractable-rank-one"} In the canonical decision problem family, structural rank $1$ is the tractable sufficiency regime, while higher structural rank enters the hard regime.
:::

## Thermodynamic Selection

::: theorem
[]{#thm:thermodynamic-selection label="thm:thermodynamic-selection"} In the canonical decision encoding, every bounded decision system with $\mathrm{DOF}(A)>1$ lies strictly above the rank-$1$ Landauer ground state in per-cycle resolution cost.
:::

::: remark
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

The bridge theorems live in `Leverage/BridgeToDQ.lean`; the coherence theorem is imported from `Ssot`; and the tractability and Landauer-cost theorems are imported from the decision-quotient development. Appendix [\[appendix-lean\]](#appendix-lean){reference-type="ref" reference="appendix-lean"} records source provenance.

## Finite Replication Entropy Gap

The theorem below uses the rank-indexed entropy cost $\Delta S_{\min}(r)=r k_B \ln 2$ from the calibrated exact-resolution model.

::: theorem
[]{#thm:england label="thm:england"} Let $\Delta S_{\min}(r) = r \cdot k_B \ln 2$ be the rank-indexed minimal entropy production under Landauer calibration. For the rank-$1$ ground regime and any replicated rank-$k$ regime: $$\Delta S_{\min}(1) + k_B \ln k \leq \Delta S_{\min}(k)$$ equivalently, $\Delta S_{\min}(k) - \Delta S_{\min}(1) \geq k_B \ln k$.
:::

::: proof
*Proof.* The gap is $(k-1) \cdot k_B \ln 2$. Since $k \leq 2^{k-1}$ (L52), taking logs gives $\ln k \leq (k-1) \ln 2$, so the gap is $\geq k_B \ln k$. ◻
:::

**Model class.** $\Delta S_{\min}$ is the rank-indexed Landauer entropy cost in the calibrated exact-resolution model. The theorem uses the finite inequality $k \le 2^{k-1}$. England's 2013 result [@england2013statistical] is a stochastic-thermodynamic path-space theorem with detailed balance and far-from-equilibrium dynamics. The common term is the multiplicity penalty $k_B \ln k$.

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


# Related Work

## Landauer, Non-Equilibrium Thermodynamics, and Selection

Landauer's principle gives the standard calibration from logically irreversible discrimination to minimum heat production and energy cost [@landauer1961irreversibility; @bennett1982thermodynamics]. Stochastic thermodynamics extends that floor to trajectory-level entropy production, work identities, and fluctuation relations [@seifert2012stochastic; @vandenbroeck2015ensemble; @wolpert2019stochastic; @jarzynski1997nonequilibrium; @crooks1999entropy]. Finite-time erasure and mismatch corrections sharpen the same theme for controlled nonequilibrium protocols [@diana2013finite; @proesmans2020finite; @manzano2024absolute]. The theorem chain above isolates a different object: a finite exact-resolution lower bound indexed by the number of independent coordinates that must be resolved to preserve the optimizer.

Relative to the Seifert and Van den Broeck--Esposito framework, the present model gives a non-asymptotic lower bound in terms of structural rank and decision-quotient entropy. Stochastic-thermodynamic frameworks resolve time-dependent nonequilibrium refinements that are outside the current finite exact-resolution model.

England's 2013 result is a stochastic-thermodynamic path-space theorem with detailed balance and far-from-equilibrium dynamics [@england2013statistical]. The corresponding replication theorem in the calibrated exact-resolution model is a finite Landauer-counting statement. The common term is the multiplicity penalty $k_B \ln k$.

## Zero-Error, Functional, and Quotient Information

The information object is closer to zero-error and confusability-based information theory than to average-case source coding [@shannon1956zero; @korner1973graphs; @lovasz1979shannon; @csiszar2011information]. The central quantity is the entropy of the decision quotient: the number of distinct optimal-action classes that survive after irrelevant coordinates are erased.

Function-relative information in physics and origins-of-life work also conditions information on successful function or selection [@szostak2003functional; @wong2023roles]. The exact-resolution object is narrower: coordinate erasure is admissible precisely when optimal-action correspondence is preserved. The rank-$1$ regime is therefore the one-coordinate exact-resolution regime, the tractable sufficiency regime, and the minimum calibrated-cost regime.

The molecular docking specialization adds a different layer from score benchmarking or heuristic search comparison: the claims are theorem-level statements about exact sufficiency, structural rank, and thermodynamic floor prior to algorithm choice.

The same molecular layer also contains geometric execution and electrostatic control statements: exact phase-space volume preservation for Velocity-Verlet, closed-form force differentiation for Lennard-Jones, and exponential Ewald decay bounds for long-range Coulomb splitting.

## Categorical Quotients and Exact Abstraction

Quotienting states by equality of $\operatorname{Opt}$ is the standard coimage construction for the decision quotient of the optimizer map $\operatorname{Opt}: S \to \mathcal{P}(A)$ in **Set**, canonically equivalent to its image [@maclane1998categories]. The theorem chain ties that quotient to coordinate sufficiency, structural rank, decision entropy, and thermodynamic cost in one proof object.

## Formal Source Provenance

Appendix [\[appendix-lean\]](#appendix-lean){reference-type="ref" reference="appendix-lean"} records source provenance for the stated claims. A Lean 4 proof file accompanies the archived artifact [@moura2021lean4; @mathlib2020]. Related precedents include verified computability and semantics developments in Coq and Isabelle [@forster2019verified; @nipkow2002isabelle; @nipkow2014concrete] and certificate-carrying proof artifacts [@necula1997proof].


# Conclusion

## Summary

The central result is a complete abstract-plus-molecular theory of exact resolution. The quotient theorems identify the coarsest exact abstraction. The rank theorems identify the irreducible coordinate count and Fisher-information dimension of that object. The complexity theorems identify both the hardness core and the witness budget required for sound checking. The thermodynamic theorems identify the corresponding cost floor. Theorem [\[thm:five-way\]](#thm:five-way){reference-type="ref" reference="thm:five-way"} then identifies the rank-$1$ regime of the canonical decision encoding as simultaneously the one-coordinate regime, the tractable sufficiency regime, and the thermodynamic ground state. The molecular sections instantiate the same framework for constrained molecular systems, cutoff-local docking structure, sampled docking, top-$k$ screening, and concrete scorer families.

The theorem package separates a structural part from an empirical calibration. The structural part is the finite acquisition chain, the canonical exact-resolution encoding, the quotient-factorization boundary, the exact-sufficiency hardness core together with its witness/checking lower bounds, the identity $\mathrm{DOF}(A)=\mathrm{srank}(\mathrm{canonicalDP}(A))$, the Fisher-rank identities, the cutoff-local docking rank bounds, the top-$k$ and ambiguity-band control theorems, the concrete Lennard-Jones and Coulomb cutoff invariance theorems, and the decision-entropy bound. The empirical inputs are bounded signal speed, the discrete transition interface used for acquisition, cutoff-local approximation control, and a positive per-bit lower bound. Landauer furnishes the universal floor.

**Main consequences:**

-   The quotient-factorization boundary for exact abstractions and the physical exclusion of extra surjective collapse beyond the decision quotient.

-   The exact identification of degree of freedom with structural rank in the canonical decision encoding and the exact identification of structural rank with Fisher-information dimension.

-   The general hardness core for exact sufficiency certification, the maximal-rank hard family, and the quantitative witness/checking lower bounds for sound exact certification.

-   The cutoff-local structural-rank bound for molecular docking, the bounded-pocket low-rank regime, strict-dominance and multiplicative-separable collapse regimes for conformer search, sampled exact/coarse winner preservation under a half-gap hypothesis, an action-side symmetry-broken sampled refinement with a canonical strict winner, a uniform positive strict gap, and no structural-rank increase, inside-cutoff sufficiency for sampled docking under the stated compatibility assumptions, top-$k$ survivor preservation under a certified boundary gap, and ambiguity-band containment in near-tie regimes.

-   The exact/coarse Lennard-Jones and Coulomb invariance theorems for finite sampled scorer families under explicit cutoff error and half-gap conditions, the closed-form Lennard-Jones gradient, exact phase-space volume preservation for Velocity-Verlet, and exponential Ewald control for long-range Coulomb splitting.

-   The energy--information theorem $E \ge k_B T H_{\mathrm{nats}}(D)$ for exact-resolution cost.

-   The bounded-acquisition inequalities $\mathrm{DOF}(A) \le c\tau/d$, the induced decision-class and decision-entropy bounds from spacetime and energy budget, and the linear budget law for independent composition.

-   The theorem-level strict-overhead branches above the Landauer floor, the finite discrete residual witness, the canonical Wolpert grounding bundle, the unified ideal/mismatch/residual energy--information hierarchy, the induced strict canonical energy separation above the Landauer-linear floor, explicit binary mismatch and two-state residual examples with additive one-unit overhead, strengthened energy--information coefficients, cumulative work laws, and the substrate step time law.

-   The bounded-lifetime consequences of positive cyclewise heat in finite-capacity substrates and the resulting finite entropy-throughput ceiling.

-   The unconditional thermodynamic selection statement that every higher-rank regime lies above the rank-$1$ Landauer ground state.

-   The finite-budget no-collapse theorem: bounded budget, positive per-bit cost, and exponential lower-bound growth cannot coexist with physical collapse.

-   The finite replication entropy gap: $\Delta S_{\min}(k) - \Delta S_{\min}(1) \geq k_B \ln k$ in the calibrated exact-resolution model.

## Scope

**1. Canonical-encoding scope:** the main theorems are exact for the canonical binary decision encoding attached to the bounded decision system. Extending the same conclusions to more general physical encodings requires an explicit transport argument.

**2. Calibration choice:** Landauer calibration supplies the physical conversion constant. Stronger substrate-dependent lower bounds belong to a different modeling layer.

**3. Replication theorem:** the finite replication entropy gap is a theorem of the calibrated exact-resolution model. England's 2013 theorem belongs to a stochastic-thermodynamic path-space model.

**4. Finite-budget model class:** the no-collapse theorem is a statement about globally bounded budget profiles with positive per-bit cost and exponential lower-bound growth. Different collapse claims require explicit bridges into that profile language.

## Final Remarks

Molecular docking now sits inside one formal decision-theoretic framework linking semantic, statistical, complexity, geometric, and thermodynamic statements. Abstractly, exact resolution is governed by sufficient coordinate sets, the decision quotient, structural rank, Fisher dimension, certification burden, and calibrated thermodynamic floor. Concretely, constrained molecular systems instantiate the same framework through holonomic constraint topology, cutoff-local interaction structure, conformer-search collapse regimes, sampled exact/coarse stability, top-$k$ survivor control, finite grid entropy bounds, explicit binary lifts of finite grid docking, concrete scorer-family invariance, exact phase-space volume preservation, long-range electrostatic splitting, and direct thermodynamic theorems for the exact docking decision problem itself.

The abstract theorems state what exact resolution costs for any bounded decision system. The molecular instantiation now states that cost directly for the exact docking decision problem once a sufficient coordinate set is fixed, states the entropy-calibrated Landauer theorem for the explicit finite binary encoding of the sampled grid docking problem, and states the bounded-region sampled inside-cutoff consequence once the retained witness fits within a bounded-region acquisition budget. The sampled finite layer now also contains a theorem-backed action-side symmetry-breaking step: ties can be converted into a canonical strict winner with a uniform positive strict gap over the finite grid state space without increasing structural rank. The cutoff-local docking theorems separately bound the docking structural rank from above and therefore identify low-rank exact regimes rather than standalone coordinate-explicit lower bounds. Approximate and heuristic docking procedures lie in the same scope through approximation, sampling, or surrogate replacement. Corollary [\[cor:holonomic-landauer-floor\]](#cor:holonomic-landauer-floor){reference-type="ref" reference="cor:holonomic-landauer-floor"} gives the direct RATTLE finite derivation: the constraint-status interface is a $k$-bit binary register, the effective coordinate count is $3N-k$, and the canonical Landauer floor scales linearly with that remaining unconstrained dimension. Remark [\[rem:molecular-independence-scope\]](#rem:molecular-independence-scope){reference-type="ref" reference="rem:molecular-independence-scope"} states the remaining scope condition precisely: the finite transport is proved once independence is specified, while derivation of that independence hypothesis from a concrete geometric constraint family remains additional work.

Appendix [\[appendix-lean\]](#appendix-lean){reference-type="ref" reference="appendix-lean"} records proof provenance.


# Proof Provenance {#appendix-lean}

This appendix reports claim traceability directly from source and generated mapping artifacts.

## Claim Coverage Matrix

## Lean Handle Map

## Proof Hardness Index


  ---------------------------------------------------------------------------------------------------------------------------------------------
  **Paper claim**                                                                                    **Lean handle**
  -------------------------------------------------------------------------------------------------- ------------------------------------------
  Corollary 4.7: Bounded-Pocket Low-Rank Regime                                                      L65

  Corollary 5.11: Decision-Entropy Bound from Spacetime and Energy Budget                            IT3, BA1, BA2, BA5, BA6, EI1, L43

  Corollary 4.5: Checking Time Lower Bound                                                           L70

  Corollary 5.12: Independent Composition Budget Law                                                 L17, L19, BA1, BA2, BA7, BA5, BA6, L43

  Corollary 5.30: RATTLE Holonomic-Constraint Landauer Floor                                         L60, L61, L62

  Corollary 5.4: Unique Minimum-Cost Regime                                                          BA8, L54, L55

  Corollary 4.4: No Sound Checker Below Witness Budget                                               L71

  Corollary 3.7: Higher-Rank Regime                                                                  L46

  Corollary 3.6: Rank-One Regime                                                                     L51

  Proposition 5.14: Two-Level Atomic Realization                                                     AC1, AC3, AC4

  Proposition 5.26: Repeated Binary Mismatch Work Law                                                IT3, BA5, BA6, WP2, WM4, L43

  Proposition 5.25: Binary Mismatch Strengthens the Energy--Information Coefficient                  IT3, BA5, BA6, WP2, WM4, L43

  Proposition 5.24: Explicit Binary Mismatch Example                                                 BA5, BA6, WP2, WM4, L43

  Proposition 5.19: Repeated Residual-Example Work Law                                               IT3, BA5, BA6, WR12, WR11, L43

  Proposition 5.18: Explicit Two-State Residual Example                                              IT3, BA5, BA6, WR12, WR11, L43

  Proposition 2.5: Bounded Region                                                                    BA1

  Proposition 5.22: Canonical Wolpert Grounding Bundle                                               BA5, BA6, WP9, L43

  Definition 2.3: Bounded Decision System                                                            L17, L19

  Proposition 5.20: Unified Energy--Information Hierarchy                                            IT3, BA5, BA6, WR12, WP2, WM4, WR11, L43

  Proposition 3.11: Finite Compression-Relation Bridge                                               L57, L58

  Proposition 5.17: Finite Discrete Residual Witness                                                 WR10, WR7, WR6

  Proposition 5.27: Positive Heat and Bounded Lifetime                                               SE1, SE2, SE3, SE4, SE5

  Proposition 5.28: Finite Lifetime Throughput Bound                                                 SE5, IT3, L43

  Definition 2.13: Canonical Decision Problem                                                        QT2, QT7, QT1, QT3

  Proposition 5.29: Speed-Heat Tradeoff                                                              SE6

  Proposition 5.23: Strict Canonical Energy Above the Landauer Floor                                 BA5, BA6, WP6, L43

  Proposition 5.16: Theorem-Level Strict Overhead Branches                                           BA5, BA6, WM6, WP6, WR10, L43

  Proposition 5.21: Structural-Resource Overhead                                                     BA5, BA6, WP8, WP7, L43

  Proposition 5.15: Substrate Step is Unit Interface Time                                            DT23, DT22, DT24

  Proposition 5.13: Threshold Channel Realization                                                    CV8, CV9, CV7

  Theorem 3.1: Surjective Abstractions Either Factor or Erase                                        L77

  Theorem 4.11: Action-Pinned Sampled Docking Has a Uniform Positive Strict Gap                      L115, L116, L117

  Theorem 2.6: Bounded Acquisition Rate                                                              BA1, BA2

  Theorem 4.33: Bounded-Action Exact Checking Is Polynomial-Time Decidable                           L87, L93

  Theorem 4.28: Bounded Potential Plus Large Cutoff Gives Sampled Structural-Rank Bound              L104

  Theorem 4.27: Bounded Potential Plus Large Cutoff Gives Structural-Rank Bound                      L95

  Theorem 5.10: Decision-Class Bound from Spacetime and Energy Budget                                IT4, IT3, BA1, BA2, BA5, BA6, EI1, L43

  Theorem 4.3: Checker Budget Lower Bound                                                            L69

  Theorem 6.1: Coherent Single-Source Regime                                                         ORA1

  Theorem 4.37: Cutoff Coulomb Winner Preservation                                                   L73

  Theorem 4.36: Cutoff Coulomb Uniform Approximation                                                 L74

  Theorem 4.38: Exact Coulomb Tail Control Gives Structural-Rank Bound                               L96

  Theorem 2.4: Counting Gap                                                                          BA10

  Theorem 2.7: Discrete Acquisition                                                                  BA3

  Theorem 3.5: DOF--Structural-Rank Identity                                                         L43

  Theorem 5.3: Energy--Information Duality                                                           IT3, EI1, L43

  Theorem 5.1: Rank Controls Exact-Resolution Cost                                                   BA7, BA6, L43

  Theorem 6.8: Finite Replication Entropy Gap                                                        L45

  Theorem 3.10: Decision-Entropy Bound                                                               IT3, L43

  Theorem 4.43: Ewald Reciprocal Core Is Positive                                                    L81

  Theorem 4.42: Ewald Real-Space Exponential Decay                                                   L82

  Theorem 4.39: Exact Real-Space Ewald Tail Control Gives Structural-Rank Bound                      L98

  Theorem 4.1: General Hardness Core for Exact Sufficiency                                           L63

  Theorem 3.2: Feasible Collapse Maps Force Quotient Factorization                                   L78

  Theorem 6.9: Finite-Budget No-Collapse                                                             PH26

  Theorem 4.29: Finite Sampled Docking Has a Canonical Uniform Error Radius                          L92

  Theorem 3.4: Fisher-Matrix Rank Equals Structural Rank                                             L76

  Theorem 3.3: Total Fisher Information Equals Structural Rank                                       L80

  Theorem 6.7: Convergence                                                                           L43, L44, L47, L55

  Theorem 4.30: Finite Grid Exact-Docking Entropy Is Controlled by Structural Rank                   L113

  Theorem 4.32: Grid Irrelevance Erasure Is Sound                                                    L88

  Theorem 4.2: Maximal Structural Rank in the Hard Family                                            L64

  Theorem 4.18: Large Cutoff Implies Docking Cutoff Boundedness                                      L89

  Theorem 4.15: Lipschitz Grid Error Gives Resolution-Controlled Approximation                       L90

  Theorem 4.34: Cutoff Lennard-Jones Winner Preservation                                             L75

  Theorem 4.40: Closed-Form Lennard-Jones Gradient                                                   L83

  Theorem 4.19: Explicit Lennard-Jones Shell Derivative Envelope                                     L103

  Theorem 4.26: Lennard-Jones Shell Gradient Bound Preserves Exact Winners                           L100

  Theorem 4.23: Lennard-Jones Shell Gradient Bound Gives Score Stability                             L102

  Theorem 4.21: Lennard-Jones Hessian Envelope Makes the Gradient Lipschitz                          L106

  Theorem 4.25: First-Order-Corrected Lennard-Jones Discretization Error Is Quadratic                L107

  Theorem 4.22: Lennard-Jones Hessian Envelope Gives a Quadratic Taylor Remainder                    L108

  Theorem 4.20: Explicit Lennard-Jones Shell Second-Derivative Envelope                              L105

  Theorem 4.24: Lennard-Jones Shell Gradient Bound Gives Uniform Exact/Grid Approximation            L101

  Theorem 4.35: Exact Lennard-Jones Tail Control Gives Structural-Rank Bound                         L97

  Theorem 5.7: Higher-Rank Exact Docking Lies Above the Ground State                                 L109

  Theorem 5.6: Binary-Encoded Finite Grid Docking Carries Landauer Cost                              L114

  Theorem 5.5: Exact Docking Structural Rank Controls Resolution Cost                                L111

  Theorem 3.8: Minimum Physical Bit Operations                                                       BA5, BA6, L43

  Theorem 4.6: Cutoff-Local Structural-Rank Bound for Exact Docking                                  L66

  Theorem 4.9: Multiplicative Separability Eliminates Conformer Dependence                           L84

  Theorem 3.9: Decision-Class Bound                                                                  IT4, L43

  Theorem 2.8: One Transition, One Bit                                                               BA4

  Theorem 6.3: Rank Identification                                                                   L43, L46, L51

  Theorem 5.2: Rank-One Ground State                                                                 BA8, L54, L55

  Theorem 4.17: Resolution-Controlled Approximation Preserves Exact Winners                          L99

  Theorem 4.16: Resolution-Controlled Approximation Gives Uniform Utility Approximation              L91

  Theorem 2.9: Resolution Requires a Sufficient Coordinate Set                                       BA5

  Theorem 4.10: Sampled Exact-Coarse Winner Preservation                                             L67

  Theorem 5.8: Inside-Cutoff Sampled Exact Resolution Gives a Concrete Bounded-Region Energy Floor   L112

  Theorem 4.12: Inside-Cutoff Sufficiency for Sampled Docking                                        L68

  Theorem 4.8: Strict Dominance Makes Coordinates Irrelevant                                         L85

  Theorem 6.5: Thermodynamic Selection                                                               BA8, L49, L54, L55

  Theorem 5.9: Exact-Resolution Time Lower Bound                                                     BA1, BA2, BA5, BA6, L43

  Theorem 4.14: Exact Top-k Ambiguity-Band Containment                                               L72

  Theorem 4.13: Top-k Preservation Under Boundary Gap                                                L79

  Theorem 4.31: Top-k Margin Certificates Are Sound                                                  L94

  Theorem 6.4: Tractable Sufficiency at Rank One                                                     L47, L53, L56

  Theorem 4.41: Velocity-Verlet Preserves Phase-Space Volume                                         L86
  ---------------------------------------------------------------------------------------------------------------------------------------------

*Auto summary: mapped 99/99 (full=99, derived=0, unmapped=0).*


::: list
**`AC1`**[]{#lh:AC1}

**`AC3`**[]{#lh:AC3}

**`AC4`**[]{#lh:AC4}

**`BA1`**[]{#lh:BA1} paper4/DecisionQuotient/Physics/BoundedAcquisition.lean

**`BA2`**[]{#lh:BA2} paper4/DecisionQuotient/Physics/BoundedAcquisition.lean

**`BA3`**[]{#lh:BA3} paper4/DecisionQuotient/Physics/BoundedAcquisition.lean

**`BA4`**[]{#lh:BA4} paper4/DecisionQuotient/Physics/BoundedAcquisition.lean

**`BA5`**[]{#lh:BA5} paper4/DecisionQuotient/Physics/BoundedAcquisition.lean

**`BA6`**[]{#lh:BA6} paper4/DecisionQuotient/Physics/BoundedAcquisition.lean

**`BA7`**[]{#lh:BA7} paper4/DecisionQuotient/Physics/BoundedAcquisition.lean

**`BA8`**[]{#lh:BA8} paper4/DecisionQuotient/Physics/BoundedAcquisition.lean

**`BA10`**[]{#lh:BA10} paper4/DecisionQuotient/Physics/BoundedAcquisition.lean

**`CV7`**[]{#lh:CV7} paper4/DecisionQuotient/Physics/Conversation.lean

**`CV8`**[]{#lh:CV8} paper4/DecisionQuotient/Physics/Conversation.lean

**`CV9`**[]{#lh:CV9} paper4/DecisionQuotient/Physics/Conversation.lean

**`DT22`**[]{#lh:DT22} paper4/DecisionQuotient/Physics/DecisionTime.lean

**`DT23`**[]{#lh:DT23} paper4/DecisionQuotient/Physics/DecisionTime.lean

**`DT24`**[]{#lh:DT24} paper4/DecisionQuotient/Physics/DecisionTime.lean

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

**`L57`**[]{#lh:L57}

**`L58`**[]{#lh:L58}

**`L60`**[]{#lh:L60} Leverage/BridgeToDQ.lean

**`L61`**[]{#lh:L61} Leverage/BridgeToDQ.lean

**`L62`**[]{#lh:L62} Leverage/BridgeToDQ.lean

**`L63`**[]{#lh:L63} Leverage/DockingTheoryBridge.lean

**`L64`**[]{#lh:L64} Leverage/DockingTheoryBridge.lean

**`L65`**[]{#lh:L65} Leverage/DockingTheoryBridge.lean

**`L66`**[]{#lh:L66} Leverage/DockingTheoryBridge.lean

**`L67`**[]{#lh:L67} Leverage/DockingTheoryBridge.lean

**`L68`**[]{#lh:L68} Leverage/DockingTheoryBridge.lean

**`L69`**[]{#lh:L69} Leverage/DockingTheoryBridge.lean

**`L70`**[]{#lh:L70} Leverage/DockingTheoryBridge.lean

**`L71`**[]{#lh:L71} Leverage/DockingTheoryBridge.lean

**`L72`**[]{#lh:L72} Leverage/DockingTheoryBridge.lean

**`L73`**[]{#lh:L73} Leverage/DockingTheoryBridge.lean

**`L74`**[]{#lh:L74} Leverage/DockingTheoryBridge.lean

**`L75`**[]{#lh:L75} Leverage/DockingTheoryBridge.lean

**`L76`**[]{#lh:L76} Leverage/DockingTheoryBridge.lean

**`L77`**[]{#lh:L77} Leverage/DockingTheoryBridge.lean

**`L78`**[]{#lh:L78} Leverage/DockingTheoryBridge.lean

**`L79`**[]{#lh:L79} Leverage/DockingTheoryBridge.lean

**`L80`**[]{#lh:L80} Leverage/DockingTheoryBridge.lean

**`L81`**[]{#lh:L81} Leverage/DockingTheoryBridge.lean

**`L82`**[]{#lh:L82} Leverage/DockingTheoryBridge.lean

**`L83`**[]{#lh:L83} Leverage/DockingTheoryBridge.lean

**`L84`**[]{#lh:L84} Leverage/DockingTheoryBridge.lean

**`L85`**[]{#lh:L85} Leverage/DockingTheoryBridge.lean

**`L86`**[]{#lh:L86} Leverage/DockingTheoryBridge.lean

**`L87`**[]{#lh:L87} Leverage/DockingTheoryBridge.lean

**`L88`**[]{#lh:L88} Leverage/DockingTheoryBridge.lean

**`L89`**[]{#lh:L89} Leverage/DockingTheoryBridge.lean

**`L90`**[]{#lh:L90} Leverage/DockingTheoryBridge.lean

**`L91`**[]{#lh:L91} Leverage/DockingTheoryBridge.lean

**`L92`**[]{#lh:L92} Leverage/DockingTheoryBridge.lean

**`L93`**[]{#lh:L93} Leverage/DockingTheoryBridge.lean

**`L94`**[]{#lh:L94} Leverage/DockingTheoryBridge.lean

**`L95`**[]{#lh:L95} Leverage/DockingTheoryBridge.lean

**`L96`**[]{#lh:L96} Leverage/DockingTheoryBridge.lean

**`L97`**[]{#lh:L97} Leverage/DockingTheoryBridge.lean

**`L98`**[]{#lh:L98} Leverage/DockingTheoryBridge.lean

**`L99`**[]{#lh:L99} Leverage/DockingTheoryBridge.lean

**`L100`**[]{#lh:L100} Leverage/DockingTheoryBridge.lean

**`L101`**[]{#lh:L101} Leverage/DockingTheoryBridge.lean

**`L102`**[]{#lh:L102} Leverage/DockingTheoryBridge.lean

**`L103`**[]{#lh:L103} Leverage/DockingTheoryBridge.lean

**`L104`**[]{#lh:L104} Leverage/DockingTheoryBridge.lean

**`L105`**[]{#lh:L105} Leverage/DockingTheoryBridge.lean

**`L106`**[]{#lh:L106} Leverage/DockingTheoryBridge.lean

**`L107`**[]{#lh:L107} Leverage/DockingTheoryBridge.lean

**`L108`**[]{#lh:L108} Leverage/DockingTheoryBridge.lean

**`L109`**[]{#lh:L109} Leverage/DockingTheoryBridge.lean

**`L110`**[]{#lh:L110} Leverage/DockingTheoryBridge.lean

**`L111`**[]{#lh:L111} Leverage/DockingTheoryBridge.lean

**`L112`**[]{#lh:L112} Leverage/DockingTheoryBridge.lean

**`L113`**[]{#lh:L113} Leverage/DockingTheoryBridge.lean

**`L114`**[]{#lh:L114} Leverage/DockingTheoryBridge.lean

**`L115`**[]{#lh:L115} Leverage/DockingTheoryBridge.lean

**`L116`**[]{#lh:L116} Leverage/DockingTheoryBridge.lean

**`L117`**[]{#lh:L117} Leverage/DockingTheoryBridge.lean

**`ORA1`**[]{#lh:ORA1} paper2/Ssot/Coherence.lean

**`PH26`**[]{#lh:PH26} paper4/DecisionQuotient/Physics/PhysicalHardness.lean

**`QT1`**[]{#lh:QT1} paper4/DecisionQuotient/Quotient.lean

**`QT2`**[]{#lh:QT2} paper4/DecisionQuotient/Quotient.lean

**`QT3`**[]{#lh:QT3} paper4/DecisionQuotient/Quotient.lean

**`QT7`**[]{#lh:QT7} paper4/DecisionQuotient/Quotient.lean

**`SE1`**[]{#lh:SE1} paper4/DecisionQuotient/ClaimClosure.lean

**`SE2`**[]{#lh:SE2} paper4/DecisionQuotient/ClaimClosure.lean

**`SE3`**[]{#lh:SE3} paper4/DecisionQuotient/ClaimClosure.lean

**`SE4`**[]{#lh:SE4} paper4/DecisionQuotient/ClaimClosure.lean

**`SE5`**[]{#lh:SE5} paper4/DecisionQuotient/ClaimClosure.lean

**`SE6`**[]{#lh:SE6} paper4/DecisionQuotient/ClaimClosure.lean

**`W1`**[]{#lh:W1} paper4/DecisionQuotient/Physics/WassersteinIntegrity.lean

**`W2`**[]{#lh:W2} paper4/DecisionQuotient/Physics/WassersteinIntegrity.lean

**`W3`**[]{#lh:W3} paper4/DecisionQuotient/Physics/WassersteinIntegrity.lean

**`W4`**[]{#lh:W4} paper4/DecisionQuotient/Physics/WassersteinIntegrity.lean

**`WM4`**[]{#lh:WM4} paper4/DecisionQuotient/Physics/WolpertMismatch.lean

**`WM6`**[]{#lh:WM6} paper4/DecisionQuotient/Physics/WolpertDecomposition.lean

**`WP2`**[]{#lh:WP2} paper4/DecisionQuotient/Physics/WolpertDecomposition.lean

**`WP6`**[]{#lh:WP6} paper4/DecisionQuotient/Physics/WolpertDecomposition.lean

**`WP7`**[]{#lh:WP7} paper4/DecisionQuotient/Physics/WolpertDecomposition.lean

**`WP8`**[]{#lh:WP8} paper4/DecisionQuotient/Physics/WolpertDecomposition.lean

**`WP9`**[]{#lh:WP9} paper4/DecisionQuotient/Physics/WolpertDecomposition.lean

**`WR6`**[]{#lh:WR6} paper4/DecisionQuotient/Physics/WolpertResidual.lean

**`WR7`**[]{#lh:WR7} paper4/DecisionQuotient/Physics/WolpertDecomposition.lean

**`WR10`**[]{#lh:WR10} paper4/DecisionQuotient/Physics/WolpertDecomposition.lean

**`WR11`**[]{#lh:WR11} paper4/DecisionQuotient/Physics/WolpertResidual.lean

**`WR12`**[]{#lh:WR12} paper4/DecisionQuotient/Physics/WolpertDecomposition.lean
:::

::: longtable
\@p0.05p0.42p0.05p0.42@ **ID** & **Lean Handle / Source** & **ID** & **Lean Handle / Source**\
**ID** & **Lean Handle / Source** & **ID** & **Lean Handle / Source**\
\
[**`AC1`**]{#lh:AC1} & `ClaimClosure.AtomicCircuitExports.AC1`

& [**`AC3`**]{#lh:AC3} & `ClaimClosure.AtomicCircuitExports.AC3`

\
[**`AC4`**]{#lh:AC4} & `ClaimClosure.AtomicCircuitExports.AC4`

& [**`BA1`**]{#lh:BA1} & `Physics.BoundedAcquisition.BoundedRegion`

\
[**`BA2`**]{#lh:BA2} & `Physics.BoundedAcquisition.acquisition_rate_bound`

& [**`BA3`**]{#lh:BA3} & `Physics.BoundedAcquisition.acquisitions_are_transitions`

\
[**`BA4`**]{#lh:BA4} & `Physics.BoundedAcquisition.one_bit_per_transition`

& [**`BA5`**]{#lh:BA5} & `Physics.BoundedAcquisition.resolution_reads_sufficient`

\
[**`BA6`**]{#lh:BA6} & `Physics.BoundedAcquisition.srank_le_resolution_bits`

& [**`BA7`**]{#lh:BA7} & `Physics.BoundedAcquisition.energy_ge_srank_cost`

\
[**`BA8`**]{#lh:BA8} & `Physics.BoundedAcquisition.srank_one_energy_minimum`

& [**`BA10`**]{#lh:BA10} & `Physics.BoundedAcquisition.counting_gap_theorem`

\
[**`CV7`**]{#lh:CV7} & `Physics.Conversation.clamp_projection_eq_iff_same_clamped_bit`

& [**`CV8`**]{#lh:CV8} & `Physics.Conversation.clampDecisionEvent_iff_bitOps_pos`

\
[**`CV9`**]{#lh:CV9} & `Physics.Conversation.clamp_event_implies_positive_energy`

& [**`DT22`**]{#lh:DT22} & `Physics.DecisionTime.substrate_step_realizes_decision_event`

\
[**`DT23`**]{#lh:DT23} & `Physics.DecisionTime.substrate_step_is_time_unit`

& [**`DT24`**]{#lh:DT24} & `Physics.DecisionTime.time_unit_law_substrate_invariant`

\
[**`EI1`**]{#lh:EI1} & `ThermodynamicLift.energy_ge_kbt_nat_entropy`

& [**`IT3`**]{#lh:IT3} & `DecisionQuotient.quotientEntropy_le_srank_binary`

\
[**`IT4`**]{#lh:IT4} & `DecisionQuotient.numOptClasses_le_pow_srank_binary`

& [**`L17`**]{#lh:L17} & `Leverage.compose_dof`

\
[**`L19`**]{#lh:L19} & `Leverage.composition_dof_additive`

& [**`L43`**]{#lh:L43} & `dof_eq_srank`

\
[**`L44`**]{#lh:L44} & `dof_one_iff_max_leverage`

& [**`L45`**]{#lh:L45} & `england_replication_inequality`

\
[**`L46`**]{#lh:L46} & `incoherent_srank_gt_one`

& [**`L47`**]{#lh:L47} & `max_coherence_forces_tractability`

\
[**`L49`**]{#lh:L49} & `srank_energy_lower_bound`

& [**`L51`**]{#lh:L51} & `ssot_srank_one`

\
[**`L52`**]{#lh:L52} & `succ_le_two_pow`

& [**`L53`**]{#lh:L53} & `sufficiency_conp_hard`\
[**`L54`**]{#lh:L54} & `thermodynamic_selection`

& [**`L55`**]{#lh:L55} & `thermodynamic_selection_unconditional`

\
[**`L56`**]{#lh:L56} & `tractable_bounded_core`

& [**`L57`**]{#lh:L57} & `Leverage.ColumnComplexityBridge.SharedCodewordCount_eq_TieBrokenRelationMoment`

\
[**`L58`**]{#lh:L58} & `Leverage.ColumnComplexityBridge.zeroIdentityDebt_tieBrokenArgmin_of_uniform_argmin_relation_bound`

& [**`L60`**]{#lh:L60} & `Leverage.rattle_constraintObservations_card`

\
[**`L61`**]{#lh:L61} & `Leverage.rattle_energy_lower_bound`

& [**`L62`**]{#lh:L62} & `Leverage.rattle_srank_eq_effectiveDOF`

\
[**`L63`**]{#lh:L63} & `Leverage.exactSufficiency_conp_core`

& [**`L64`**]{#lh:L64} & `Leverage.exactSufficiency_hardFamily_srank_eq_n`

\
[**`L65`**]{#lh:L65} & `Leverage.molecularDocking_boundedPocket_srank_bound`

& [**`L66`**]{#lh:L66} & `Leverage.molecularDocking_srank_bound`

\
[**`L67`**]{#lh:L67} & `Leverage.sampledDocking_exactCoarse_opt_agree_of_gap`

& [**`L68`**]{#lh:L68} & `Leverage.sampledDocking_insideCutoff_sufficient`

\
[**`L69`**]{#lh:L69} & `Leverage.exactSufficiency_checkerBudget_ge_witnessBudget`

& [**`L70`**]{#lh:L70} & `Leverage.exactSufficiency_checkingTime_ge_witnessBudget`

\
[**`L71`**]{#lh:L71} & `Leverage.exactSufficiency_noSoundChecker_below_witnessBudget`

& [**`L72`**]{#lh:L72} & `Leverage.exactTopK_subset_ambiguityBand`

\
[**`L73`**]{#lh:L73} & `Leverage.exactVsCutoffCoulomb_opt_invariance`

& [**`L74`**]{#lh:L74} & `Leverage.exactVsCutoffCoulomb_uniformApprox`

\
[**`L75`**]{#lh:L75} & `Leverage.exactVsCutoffLJ_opt_invariance`

& [**`L76`**]{#lh:L76} & `Leverage.fisherMatrix_rank_eq_srank`

\
[**`L77`**]{#lh:L77} & `Leverage.surjectiveAbstraction_factors_or_erases`

& [**`L78`**]{#lh:L78} & `Leverage.surjectiveAbstraction_withFeasibleCollapseMap_factors`

\
[**`L79`**]{#lh:L79} & `Leverage.topKPreserved_of_boundaryGap`

& [**`L80`**]{#lh:L80} & `Leverage.totalFisher_eq_srank`

\
[**`L81`**]{#lh:L81} & `Leverage.ewaldFourier_positive`

& [**`L82`**]{#lh:L82} & `Leverage.ewaldRealSpace_exponentialDecay`

\
[**`L83`**]{#lh:L83} & `Leverage.lennardJones_gradient`

& [**`L84`**]{#lh:L84} & `Leverage.multiplicativeSeparable_emptySufficient`

\
[**`L85`**]{#lh:L85} & `Leverage.strictGlobalDominance_emptySufficient`

& [**`L86`**]{#lh:L86} & `Leverage.velocityVerlet_preservesVolume`

\
[**`L87`**]{#lh:L87} & `Leverage.boundedActions_polynomialTime`

& [**`L88`**]{#lh:L88} & `Leverage.gridMDState_sufficient_erase_irrelevant`

\
[**`L89`**]{#lh:L89} & `Leverage.largeCutoff_implies_bounded`

& [**`L90`**]{#lh:L90} & `Leverage.lipschitzUtilityApprox_implies_resolutionControlled`

\
[**`L91`**]{#lh:L91} & `Leverage.resolutionControlledApprox_implies_uniformApprox`

& [**`L92`**]{#lh:L92} & `Leverage.sampledDocking_finiteUniformErrorRadius_witnesses`

\
[**`L93`**]{#lh:L93} & `Leverage.sufficiency_poly_bounded_actions`

& [**`L94`**]{#lh:L94} & `Leverage.topKMargin_certificate_sound`

\
[**`L95`**]{#lh:L95} & `Leverage.boundedPotential_largeCutoff_srank_bound`

& [**`L96`**]{#lh:L96} & `Leverage.exactCoulomb_largeCutoff_srank_bound`

\
[**`L97`**]{#lh:L97} & `Leverage.exactLJ_largeCutoff_srank_bound`

& [**`L98`**]{#lh:L98} & `Leverage.exactRealEwald_largeCutoff_srank_bound`

\
[**`L99`**]{#lh:L99} & `Leverage.resolutionControlled_gap_implies_opt_invariance`

& [**`L100`**]{#lh:L100} & `Leverage.exactLJ_distanceError_gap_implies_opt_invariance`

\
[**`L101`**]{#lh:L101} & `Leverage.exactLJ_uniformApprox_of_distanceError_and_shellGradBound`

& [**`L102`**]{#lh:L102} & `Leverage.lennardJones_value_diff_le_of_shellGradBound`

\
[**`L103`**]{#lh:L103} & `Leverage.abs_ljGradExpr_le_shellBound`

& [**`L104`**]{#lh:L104} & `Leverage.boundedPotential_largeCutoff_sampledSrank_bound`

\
[**`L105`**]{#lh:L105} & `Leverage.abs_ljSecondDerivExpr_le_shellBound`

& [**`L106`**]{#lh:L106} & `Leverage.lennardJones_grad_diff_le_of_shellSecondDerivBound`

\
[**`L107`**]{#lh:L107} & `Leverage.exactLJ_quadraticDiscretizationError_of_distanceError_and_shellSecondDerivShellBound`

& [**`L108`**]{#lh:L108} & `Leverage.lennardJones_taylor_remainder_le_of_shellSecondDerivShellBound`

\
[**`L109`**]{#lh:L109} & `Leverage.molecularDocking_above_ground_of_srank_gt_one`

& [**`L110`**]{#lh:L110} & `Leverage.molecularDocking_landauer_entropy_lower_bound`

\
[**`L111`**]{#lh:L111} & `Leverage.molecularDocking_srank_energy_lower_bound`

& [**`L112`**]{#lh:L112} & `Leverage.sampledDocking_energy_le_boundedAcquisitions_of_insideCutoff_and_budget`

\
[**`L113`**]{#lh:L113} & `Leverage.sampledDocking_flatExact_natEntropy_le_srank_log_gridArity`

& [**`L114`**]{#lh:L114} & `Leverage.sampledDocking_flatBinaryEncoded_landauer_entropy_lower_bound`

\
[**`L115`**]{#lh:L115} & `Leverage.sampledDocking_flatActionPinned_srank_le_srank`

& [**`L116`**]{#lh:L116} & `Leverage.sampledDocking_flatActionPinned_strictOpt`

\
[**`L117`**]{#lh:L117} & `Leverage.sampledDocking_flatActionPinned_uniformStrictUtilityGap_of_card_gt_one`

& [**`ORA1`**]{#lh:ORA1} & `oracle_arbitrary`

\
[**`PH26`**]{#lh:PH26} & `PhysicalComplexity.no_collapse_of_bounded_budget_pos_cost_exp_lb`

& [**`QT1`**]{#lh:QT1} & `DecisionProblem.quotient_is_coarsest`

\
[**`QT2`**]{#lh:QT2} & `DecisionProblem.quotientMap_preservesOpt`

& [**`QT3`**]{#lh:QT3} & `DecisionProblem.quotient_represents_opt_equiv`

\
[**`QT7`**]{#lh:QT7} & `DecisionProblem.quotient_has_unique_factorization`

& [**`SE1`**]{#lh:SE1} & `ClaimClosure.SE1`

\
[**`SE2`**]{#lh:SE2} & `ClaimClosure.SE2`

& [**`SE3`**]{#lh:SE3} & `ClaimClosure.SE3`

\
[**`SE4`**]{#lh:SE4} & `ClaimClosure.SE4`

& [**`SE5`**]{#lh:SE5} & `ClaimClosure.SE5`

\
[**`SE6`**]{#lh:SE6} & `ClaimClosure.SE6`

& [**`W1`**]{#lh:W1} & `Physics.single_future_zero_cost`

\
[**`W2`**]{#lh:W2} & `Physics.transportCost_pos_of_offDiag`

& [**`W3`**]{#lh:W3} & `Physics.integrity_is_centroid`

\
[**`W4`**]{#lh:W4} & `Physics.wasserstein_bridge`

& [**`WM4`**]{#lh:WM4} & `Physics.WolpertMismatch.mismatchNatLowerBound_pos_of_exists_ne`

\
[**`WM6`**]{#lh:WM6} & `Physics.WolpertDecomposition.effective_model_strictly_exceeds_landauer_of_distribution_mismatch`

& [**`WP2`**]{#lh:WP2} & `Physics.WolpertDecomposition.landauer_floor_plus_decomposition_lower_bound`

\
[**`WP6`**]{#lh:WP6} & `Physics.WolpertDecomposition.effective_model_strictly_exceeds_landauer_of_either_cited_component`

& [**`WP7`**]{#lh:WP7} & `Physics.WolpertDecomposition.landauer_floor_plus_structural_resource_lower_bound`

\
[**`WP8`**]{#lh:WP8} & `Physics.WolpertDecomposition.energy_lower_bound_increases_by_structural_resource`

& [**`WP9`**]{#lh:WP9} & `Physics.WolpertDecomposition.physical_grounding_bundle_with_wolpert_decomposition`

\
[**`WR6`**]{#lh:WR6} & `Physics.WolpertResidual.discreteResidualNatLowerBound_pos_of_asymmetry_or_oneway`

& [**`WR7`**]{#lh:WR7} & `Physics.WolpertDecomposition.stopping_time_residual_of_discrete_edge_split`

\
[**`WR10`**]{#lh:WR10} & `Physics.WolpertDecomposition.effective_model_strictly_exceeds_landauer_of_finite_discrete_witness`

& [**`WR11`**]{#lh:WR11} & `Physics.WolpertResidual.binaryEncodedResidualNatLowerBound_eq_one`

\
[**`WR12`**]{#lh:WR12} & `Physics.WolpertDecomposition.effective_model_ge_landauer_plus_one_of_binary_encoded_residual_example`

& &\
:::


  ------------------------------------------------------------------------------------------------------------------------------------------------
  **Paper handle**                                     **Hardness profile**   **Regime tags**           **Lean support**
  ---------------------------------------------------- ---------------------- ------------------------- ------------------------------------------
  `cor:bounded-pocket-regime`                          `unspecified`          \-                        L65

  `cor:budget-entropy-bound`                           `unspecified`          \-                        IT3, BA1, BA2, BA5, BA6, EI1, L43

  `cor:checking-time-lower-bound`                      `unspecified`          \-                        L70

  `cor:composition-budget-law`                         `unspecified`          \-                        L17, L19, BA1, BA2, BA7, BA5, BA6, L43

  `cor:holonomic-landauer-floor`                       `unspecified`          \-                        L60, L61, L62

  `cor:minimum-cost-regime`                            `unspecified`          \-                        BA8, L54, L55

  `cor:no-sound-checker-below-budget`                  `unspecified`          \-                        L71

  `cor:rank-above-one`                                 `unspecified`          \-                        L46

  `cor:rank-one`                                       `unspecified`          \-                        L51

  `prop:atomic-realization`                            `unspecified`          \-                        AC1, AC3, AC4

  `prop:binary-mismatch-cumulative-work`               `unspecified`          \-                        IT3, BA5, BA6, WP2, WM4, L43

  `prop:binary-mismatch-energy-information`            `unspecified`          \-                        IT3, BA5, BA6, WP2, WM4, L43

  `prop:binary-mismatch-example`                       `unspecified`          \-                        BA5, BA6, WP2, WM4, L43

  `prop:binary-residual-cumulative-work`               `unspecified`          \-                        IT3, BA5, BA6, WR12, WR11, L43

  `prop:binary-residual-example`                       `unspecified`          \-                        IT3, BA5, BA6, WR12, WR11, L43

  `prop:bounded-region`                                `unspecified`          \-                        BA1

  `prop:canonical-wolpert-bundle`                      `unspecified`          \-                        BA5, BA6, WP9, L43

  `prop:dof-additive`                                  `unspecified`          \-                        L17, L19

  `prop:ei-hierarchy`                                  `unspecified`          \-                        IT3, BA5, BA6, WR12, WP2, WM4, WR11, L43

  `prop:finite-compression-bridge`                     `unspecified`          \-                        L57, L58

  `prop:finite-discrete-residual`                      `unspecified`          \-                        WR10, WR7, WR6

  `prop:finite-lifetime`                               `unspecified`          \-                        SE1, SE2, SE3, SE4, SE5

  `prop:lifetime-throughput`                           `unspecified`          \-                        SE5, IT3, L43

  `prop:optimizer-quotient`                            `unspecified`          \-                        QT2, QT7, QT1, QT3

  `prop:speed-heat-tradeoff`                           `unspecified`          \-                        SE6

  `prop:strict-canonical-energy`                       `unspecified`          \-                        BA5, BA6, WP6, L43

  `prop:strict-overhead`                               `unspecified`          \-                        BA5, BA6, WM6, WP6, WR10, L43

  `prop:structural-resource-overhead`                  `unspecified`          \-                        BA5, BA6, WP8, WP7, L43

  `prop:substrate-time-law`                            `unspecified`          \-                        DT23, DT22, DT24

  `prop:threshold-channel`                             `unspecified`          \-                        CV8, CV9, CV7

  `thm:abstraction-factors-or-erases`                  `unspecified`          \-                        L77

  `thm:action-pinned-uniform-gap`                      `unspecified`          \-                        L115, L116, L117

  `thm:bounded-acquisition`                            `unspecified`          \-                        BA1, BA2

  `thm:bounded-actions-poly`                           `unspecified`          \-                        L87, L93

  `thm:bounded-potential-large-cutoff-sampled-srank`   `unspecified`          \-                        L104

  `thm:bounded-potential-large-cutoff-srank`           `unspecified`          \-                        L95

  `thm:budget-class-bound`                             `unspecified`          \-                        IT4, IT3, BA1, BA2, BA5, BA6, EI1, L43

  `thm:checker-budget-lower-bound`                     `unspecified`          \-                        L69

  `thm:coherent-single-source`                         `unspecified`          \-                        ORA1

  `thm:coulomb-cutoff-invariance`                      `unspecified`          \-                        L73

  `thm:coulomb-cutoff-uniform-approx`                  `unspecified`          \-                        L74

  `thm:coulomb-tail-srank`                             `unspecified`          \-                        L96

  `thm:counting-gap`                                   `unspecified`          \-                        BA10

  `thm:discrete-acquisition`                           `unspecified`          \-                        BA3

  `thm:dof-srank`                                      `unspecified`          \-                        L43

  `thm:energy-entropy`                                 `unspecified`          \-                        IT3, EI1, L43

  `thm:energy-rank`                                    `unspecified`          \-                        BA7, BA6, L43

  `thm:england`                                        `unspecified`          \-                        L45

  `thm:entropy-bound`                                  `unspecified`          \-                        IT3, L43

  `thm:ewald-fourier-positive`                         `unspecified`          \-                        L81

  `thm:ewald-real-space-decay`                         `unspecified`          \-                        L82

  `thm:ewald-tail-srank`                               `unspecified`          \-                        L98

  `thm:exact-sufficiency-hardness-core`                `unspecified`          \-                        L63

  `thm:feasible-collapse-factors`                      `unspecified`          \-                        L78

  `thm:finite-budget-no-collapse`                      `unspecified`          \-                        PH26

  `thm:finite-uniform-error-radius`                    `unspecified`          \-                        L92

  `thm:fisher-rank-srank`                              `unspecified`          \-                        L76

  `thm:fisher-sum-srank`                               `unspecified`          \-                        L80

  `thm:five-way`                                       `unspecified`          \-                        L43, L44, L47, L55

  `thm:grid-docking-entropy`                           `unspecified`          \-                        L113

  `thm:grid-erase-irrelevant`                          `unspecified`          \-                        L88

  `thm:hard-family-srank`                              `unspecified`          \-                        L64

  `thm:large-cutoff-bounded`                           `unspecified`          \-                        L89

  `thm:lipschitz-grid-approx`                          `unspecified`          \-                        L90

  `thm:lj-cutoff-invariance`                           `unspecified`          \-                        L75

  `thm:lj-gradient`                                    `unspecified`          \-                        L83

  `thm:lj-shell-derivative-envelope`                   `unspecified`          \-                        L103

  `thm:lj-shell-gap-invariance`                        `unspecified`          \-                        L100

  `thm:lj-shell-grad-stability`                        `unspecified`          \-                        L102

  `thm:lj-shell-hessian-lipschitz`                     `unspecified`          \-                        L106

  `thm:lj-shell-quadratic-discretization`              `unspecified`          \-                        L107

  `thm:lj-shell-quadratic-remainder`                   `unspecified`          \-                        L108

  `thm:lj-shell-second-derivative-envelope`            `unspecified`          \-                        L105

  `thm:lj-shell-uniform-approx`                        `unspecified`          \-                        L101

  `thm:lj-tail-srank`                                  `unspecified`          \-                        L97

  `thm:md-above-ground`                                `unspecified`          \-                        L109

  `thm:md-energy-entropy`                              `unspecified`          \-                        L114

  `thm:md-energy-rank`                                 `unspecified`          \-                        L111

  `thm:min-bit-operations`                             `unspecified`          \-                        BA5, BA6, L43

  `thm:molecular-docking-srank-bound`                  `unspecified`          \-                        L66

  `thm:multiplicative-separable-empty-sufficient`      `unspecified`          \-                        L84

  `thm:numopt-bound`                                   `unspecified`          \-                        IT4, L43

  `thm:one-transition-one-bit`                         `unspecified`          \-                        BA4

  `thm:rank-identification`                            `unspecified`          \-                        L43, L46, L51

  `thm:rank-one-ground`                                `unspecified`          \-                        BA8, L54, L55

  `thm:resolution-controlled-gap-invariance`           `unspecified`          \-                        L99

  `thm:resolution-controlled-uniform`                  `unspecified`          \-                        L91

  `thm:resolution-sufficient`                          `unspecified`          \-                        BA5

  `thm:sampled-docking-gap`                            `unspecified`          \-                        L67

  `thm:sampled-inside-cutoff-budget-energy`            `unspecified`          \-                        L112

  `thm:sampled-inside-cutoff-sufficient`               `unspecified`          \-                        L68

  `thm:strict-dominance-empty-sufficient`              `unspecified`          \-                        L85

  `thm:thermodynamic-selection`                        `unspecified`          \-                        BA8, L49, L54, L55

  `thm:time-lower-bound`                               `unspecified`          \-                        BA1, BA2, BA5, BA6, L43

  `thm:topk-ambiguity-band`                            `unspecified`          \-                        L72

  `thm:topk-boundary-gap`                              `unspecified`          \-                        L79

  `thm:topk-certificate-sound`                         `unspecified`          \-                        L94

  `thm:tractable-rank-one`                             `unspecified`          \-                        L47, L53, L56

  `thm:velocity-verlet-volume`                         `unspecified`          \-                        L86
  ------------------------------------------------------------------------------------------------------------------------------------------------

*Auto summary: indexed 99 claims by hardness profile (unspecified=99).*


# Scope Statements {#appendix-assumptions}

This appendix lists the principal scope statements for the finite decision-thermodynamic framework:

-   **Canonical encoding:** The main theorems are exact for the canonical binary decision problem attached to the bounded decision system. Other physical encodings require an explicit transport theorem.

-   **Structural chain:** Counting Gap fixes the finite-event statement. Bounded Acquisition fixes the traversal-rate statement. Discrete Acquisition and One Transition, One Bit fix the acquisition-event interface.

-   **Landauer calibration:** Thermodynamic cost is calibrated by a per-bit Landauer floor. Stronger substrate-dependent lower bounds may exist, but they are additional assumptions, not part of the theorem package under discussion.

-   **Exact-resolution setting:** The results concern exact sufficiency and exact-resolution cost. Approximate, stochastic, or bounded-confidence regimes require separate analysis.

-   **Finite state family:** The entropy and replication theorems are finite counting results. Continuum models must first be reduced to a finite decision quotient before these arguments apply.

-   **Replication theorem:** The finite replication entropy gap is a theorem of the calibrated exact-resolution model. England's 2013 theorem belongs to a stochastic-thermodynamic path-space model.

-   **Finite-budget theorem:** Finite-Budget No-Collapse is a theorem about bounded budgets, positive per-bit cost, and exponential lower-bound growth.

::: remark
[]{#rem:molecular-independence-scope label="rem:molecular-independence-scope"} Corollary [\[cor:holonomic-landauer-floor\]](#cor:holonomic-landauer-floor){reference-type="ref" reference="cor:holonomic-landauer-floor"} proves the finite constrained-molecular transport once the RATTLE holonomic topology supplies $k$ independent constraints and the corresponding binary status interface. It does not yet derive that independence hypothesis from a concrete geometric constraint family and molecular topology object. Establishing that derivation is additional work beyond the present theorem package, not a hidden premise of the corollary.
:::


# Complete Theorem Index {#appendix-theorems}

Paper-level labeled claims:

**Exact-Resolution Model (Section 2):**

-   Proposition [\[prop:dof-additive\]](#prop:dof-additive){reference-type="ref" reference="prop:dof-additive"} (DOF Additivity)

-   Theorem [\[thm:counting-gap\]](#thm:counting-gap){reference-type="ref" reference="thm:counting-gap"}

-   Proposition [\[prop:bounded-region\]](#prop:bounded-region){reference-type="ref" reference="prop:bounded-region"}

-   Theorem [\[thm:bounded-acquisition\]](#thm:bounded-acquisition){reference-type="ref" reference="thm:bounded-acquisition"}

-   Theorem [\[thm:discrete-acquisition\]](#thm:discrete-acquisition){reference-type="ref" reference="thm:discrete-acquisition"}

-   Theorem [\[thm:one-transition-one-bit\]](#thm:one-transition-one-bit){reference-type="ref" reference="thm:one-transition-one-bit"}

-   Theorem [\[thm:resolution-sufficient\]](#thm:resolution-sufficient){reference-type="ref" reference="thm:resolution-sufficient"}

**Exact Resolution, Quotient Structure, and Compression (Section 3):**

-   Theorem [\[thm:dof-srank\]](#thm:dof-srank){reference-type="ref" reference="thm:dof-srank"}

-   Corollary [\[cor:rank-one\]](#cor:rank-one){reference-type="ref" reference="cor:rank-one"}

-   Corollary [\[cor:rank-above-one\]](#cor:rank-above-one){reference-type="ref" reference="cor:rank-above-one"}

-   Theorem [\[thm:min-bit-operations\]](#thm:min-bit-operations){reference-type="ref" reference="thm:min-bit-operations"}

-   Theorem [\[thm:numopt-bound\]](#thm:numopt-bound){reference-type="ref" reference="thm:numopt-bound"}

-   Theorem [\[thm:entropy-bound\]](#thm:entropy-bound){reference-type="ref" reference="thm:entropy-bound"}

-   Theorem [\[thm:abstraction-factors-or-erases\]](#thm:abstraction-factors-or-erases){reference-type="ref" reference="thm:abstraction-factors-or-erases"}

-   Theorem [\[thm:feasible-collapse-factors\]](#thm:feasible-collapse-factors){reference-type="ref" reference="thm:feasible-collapse-factors"}

-   Theorem [\[thm:fisher-sum-srank\]](#thm:fisher-sum-srank){reference-type="ref" reference="thm:fisher-sum-srank"}

-   Theorem [\[thm:fisher-rank-srank\]](#thm:fisher-rank-srank){reference-type="ref" reference="thm:fisher-rank-srank"}

-   Proposition [\[prop:finite-compression-bridge\]](#prop:finite-compression-bridge){reference-type="ref" reference="prop:finite-compression-bridge"}

**Complexity Boundary (Section 4):**

-   Theorem [\[thm:exact-sufficiency-hardness-core\]](#thm:exact-sufficiency-hardness-core){reference-type="ref" reference="thm:exact-sufficiency-hardness-core"}

-   Theorem [\[thm:hard-family-srank\]](#thm:hard-family-srank){reference-type="ref" reference="thm:hard-family-srank"}

-   Theorem [\[thm:checker-budget-lower-bound\]](#thm:checker-budget-lower-bound){reference-type="ref" reference="thm:checker-budget-lower-bound"}

-   Corollary [\[cor:no-sound-checker-below-budget\]](#cor:no-sound-checker-below-budget){reference-type="ref" reference="cor:no-sound-checker-below-budget"}

-   Corollary [\[cor:checking-time-lower-bound\]](#cor:checking-time-lower-bound){reference-type="ref" reference="cor:checking-time-lower-bound"}

-   Theorem [\[thm:molecular-docking-srank-bound\]](#thm:molecular-docking-srank-bound){reference-type="ref" reference="thm:molecular-docking-srank-bound"}

-   Corollary [\[cor:bounded-pocket-regime\]](#cor:bounded-pocket-regime){reference-type="ref" reference="cor:bounded-pocket-regime"}

-   Theorem [\[thm:lipschitz-grid-approx\]](#thm:lipschitz-grid-approx){reference-type="ref" reference="thm:lipschitz-grid-approx"}

-   Theorem [\[thm:resolution-controlled-uniform\]](#thm:resolution-controlled-uniform){reference-type="ref" reference="thm:resolution-controlled-uniform"}

-   Theorem [\[thm:lj-shell-derivative-envelope\]](#thm:lj-shell-derivative-envelope){reference-type="ref" reference="thm:lj-shell-derivative-envelope"}

-   Theorem [\[thm:lj-shell-second-derivative-envelope\]](#thm:lj-shell-second-derivative-envelope){reference-type="ref" reference="thm:lj-shell-second-derivative-envelope"}

-   Theorem [\[thm:lj-shell-hessian-lipschitz\]](#thm:lj-shell-hessian-lipschitz){reference-type="ref" reference="thm:lj-shell-hessian-lipschitz"}

-   Theorem [\[thm:lj-shell-quadratic-remainder\]](#thm:lj-shell-quadratic-remainder){reference-type="ref" reference="thm:lj-shell-quadratic-remainder"}

-   Theorem [\[thm:lj-shell-grad-stability\]](#thm:lj-shell-grad-stability){reference-type="ref" reference="thm:lj-shell-grad-stability"}

-   Theorem [\[thm:lj-shell-uniform-approx\]](#thm:lj-shell-uniform-approx){reference-type="ref" reference="thm:lj-shell-uniform-approx"}

-   Theorem [\[thm:lj-shell-quadratic-discretization\]](#thm:lj-shell-quadratic-discretization){reference-type="ref" reference="thm:lj-shell-quadratic-discretization"}

-   Theorem [\[thm:lj-shell-gap-invariance\]](#thm:lj-shell-gap-invariance){reference-type="ref" reference="thm:lj-shell-gap-invariance"}

-   Theorem [\[thm:resolution-controlled-gap-invariance\]](#thm:resolution-controlled-gap-invariance){reference-type="ref" reference="thm:resolution-controlled-gap-invariance"}

-   Theorem [\[thm:large-cutoff-bounded\]](#thm:large-cutoff-bounded){reference-type="ref" reference="thm:large-cutoff-bounded"}

-   Theorem [\[thm:bounded-potential-large-cutoff-srank\]](#thm:bounded-potential-large-cutoff-srank){reference-type="ref" reference="thm:bounded-potential-large-cutoff-srank"}

-   Theorem [\[thm:bounded-potential-large-cutoff-sampled-srank\]](#thm:bounded-potential-large-cutoff-sampled-srank){reference-type="ref" reference="thm:bounded-potential-large-cutoff-sampled-srank"}

-   Theorem [\[thm:finite-uniform-error-radius\]](#thm:finite-uniform-error-radius){reference-type="ref" reference="thm:finite-uniform-error-radius"}

-   Theorem [\[thm:grid-docking-entropy\]](#thm:grid-docking-entropy){reference-type="ref" reference="thm:grid-docking-entropy"}

-   Theorem [\[thm:topk-certificate-sound\]](#thm:topk-certificate-sound){reference-type="ref" reference="thm:topk-certificate-sound"}

-   Theorem [\[thm:grid-erase-irrelevant\]](#thm:grid-erase-irrelevant){reference-type="ref" reference="thm:grid-erase-irrelevant"}

-   Theorem [\[thm:bounded-actions-poly\]](#thm:bounded-actions-poly){reference-type="ref" reference="thm:bounded-actions-poly"}

-   Theorem [\[thm:strict-dominance-empty-sufficient\]](#thm:strict-dominance-empty-sufficient){reference-type="ref" reference="thm:strict-dominance-empty-sufficient"}

-   Theorem [\[thm:multiplicative-separable-empty-sufficient\]](#thm:multiplicative-separable-empty-sufficient){reference-type="ref" reference="thm:multiplicative-separable-empty-sufficient"}

-   Theorem [\[thm:sampled-docking-gap\]](#thm:sampled-docking-gap){reference-type="ref" reference="thm:sampled-docking-gap"}

-   Theorem [\[thm:action-pinned-uniform-gap\]](#thm:action-pinned-uniform-gap){reference-type="ref" reference="thm:action-pinned-uniform-gap"}

-   Theorem [\[thm:sampled-inside-cutoff-sufficient\]](#thm:sampled-inside-cutoff-sufficient){reference-type="ref" reference="thm:sampled-inside-cutoff-sufficient"}

-   Theorem [\[thm:topk-boundary-gap\]](#thm:topk-boundary-gap){reference-type="ref" reference="thm:topk-boundary-gap"}

-   Theorem [\[thm:topk-ambiguity-band\]](#thm:topk-ambiguity-band){reference-type="ref" reference="thm:topk-ambiguity-band"}

-   Theorem [\[thm:lj-cutoff-invariance\]](#thm:lj-cutoff-invariance){reference-type="ref" reference="thm:lj-cutoff-invariance"}

-   Theorem [\[thm:lj-tail-srank\]](#thm:lj-tail-srank){reference-type="ref" reference="thm:lj-tail-srank"}

-   Theorem [\[thm:coulomb-cutoff-uniform-approx\]](#thm:coulomb-cutoff-uniform-approx){reference-type="ref" reference="thm:coulomb-cutoff-uniform-approx"}

-   Theorem [\[thm:coulomb-cutoff-invariance\]](#thm:coulomb-cutoff-invariance){reference-type="ref" reference="thm:coulomb-cutoff-invariance"}

-   Theorem [\[thm:coulomb-tail-srank\]](#thm:coulomb-tail-srank){reference-type="ref" reference="thm:coulomb-tail-srank"}

-   Theorem [\[thm:ewald-tail-srank\]](#thm:ewald-tail-srank){reference-type="ref" reference="thm:ewald-tail-srank"}

-   Theorem [\[thm:lj-gradient\]](#thm:lj-gradient){reference-type="ref" reference="thm:lj-gradient"}

-   Theorem [\[thm:velocity-verlet-volume\]](#thm:velocity-verlet-volume){reference-type="ref" reference="thm:velocity-verlet-volume"}

-   Theorem [\[thm:ewald-real-space-decay\]](#thm:ewald-real-space-decay){reference-type="ref" reference="thm:ewald-real-space-decay"}

-   Theorem [\[thm:ewald-fourier-positive\]](#thm:ewald-fourier-positive){reference-type="ref" reference="thm:ewald-fourier-positive"}

**Thermodynamic Cost (Section 5):**

-   Theorem [\[thm:energy-rank\]](#thm:energy-rank){reference-type="ref" reference="thm:energy-rank"}

-   Theorem [\[thm:rank-one-ground\]](#thm:rank-one-ground){reference-type="ref" reference="thm:rank-one-ground"}

-   Theorem [\[thm:energy-entropy\]](#thm:energy-entropy){reference-type="ref" reference="thm:energy-entropy"}

-   Corollary [\[cor:minimum-cost-regime\]](#cor:minimum-cost-regime){reference-type="ref" reference="cor:minimum-cost-regime"}

-   Theorem [\[thm:md-energy-rank\]](#thm:md-energy-rank){reference-type="ref" reference="thm:md-energy-rank"}

-   Theorem [\[thm:md-energy-entropy\]](#thm:md-energy-entropy){reference-type="ref" reference="thm:md-energy-entropy"}

-   Theorem [\[thm:md-above-ground\]](#thm:md-above-ground){reference-type="ref" reference="thm:md-above-ground"}

-   Theorem [\[thm:sampled-inside-cutoff-budget-energy\]](#thm:sampled-inside-cutoff-budget-energy){reference-type="ref" reference="thm:sampled-inside-cutoff-budget-energy"}

-   Theorem [\[thm:time-lower-bound\]](#thm:time-lower-bound){reference-type="ref" reference="thm:time-lower-bound"}

-   Theorem [\[thm:budget-class-bound\]](#thm:budget-class-bound){reference-type="ref" reference="thm:budget-class-bound"}

-   Corollary [\[cor:budget-entropy-bound\]](#cor:budget-entropy-bound){reference-type="ref" reference="cor:budget-entropy-bound"}

-   Corollary [\[cor:composition-budget-law\]](#cor:composition-budget-law){reference-type="ref" reference="cor:composition-budget-law"}

-   Proposition [\[prop:threshold-channel\]](#prop:threshold-channel){reference-type="ref" reference="prop:threshold-channel"}

-   Proposition [\[prop:atomic-realization\]](#prop:atomic-realization){reference-type="ref" reference="prop:atomic-realization"}

-   Proposition [\[prop:substrate-time-law\]](#prop:substrate-time-law){reference-type="ref" reference="prop:substrate-time-law"}

-   Proposition [\[prop:strict-overhead\]](#prop:strict-overhead){reference-type="ref" reference="prop:strict-overhead"}

-   Proposition [\[prop:finite-discrete-residual\]](#prop:finite-discrete-residual){reference-type="ref" reference="prop:finite-discrete-residual"}

-   Proposition [\[prop:binary-residual-example\]](#prop:binary-residual-example){reference-type="ref" reference="prop:binary-residual-example"}

-   Proposition [\[prop:binary-residual-cumulative-work\]](#prop:binary-residual-cumulative-work){reference-type="ref" reference="prop:binary-residual-cumulative-work"}

-   Proposition [\[prop:ei-hierarchy\]](#prop:ei-hierarchy){reference-type="ref" reference="prop:ei-hierarchy"}

-   Proposition [\[prop:structural-resource-overhead\]](#prop:structural-resource-overhead){reference-type="ref" reference="prop:structural-resource-overhead"}

-   Proposition [\[prop:canonical-wolpert-bundle\]](#prop:canonical-wolpert-bundle){reference-type="ref" reference="prop:canonical-wolpert-bundle"}

-   Proposition [\[prop:strict-canonical-energy\]](#prop:strict-canonical-energy){reference-type="ref" reference="prop:strict-canonical-energy"}

-   Proposition [\[prop:binary-mismatch-example\]](#prop:binary-mismatch-example){reference-type="ref" reference="prop:binary-mismatch-example"}

-   Proposition [\[prop:binary-mismatch-energy-information\]](#prop:binary-mismatch-energy-information){reference-type="ref" reference="prop:binary-mismatch-energy-information"}

-   Proposition [\[prop:binary-mismatch-cumulative-work\]](#prop:binary-mismatch-cumulative-work){reference-type="ref" reference="prop:binary-mismatch-cumulative-work"}

-   Proposition [\[prop:finite-lifetime\]](#prop:finite-lifetime){reference-type="ref" reference="prop:finite-lifetime"}

-   Proposition [\[prop:lifetime-throughput\]](#prop:lifetime-throughput){reference-type="ref" reference="prop:lifetime-throughput"}

-   Proposition [\[prop:speed-heat-tradeoff\]](#prop:speed-heat-tradeoff){reference-type="ref" reference="prop:speed-heat-tradeoff"}

-   Corollary [\[cor:holonomic-landauer-floor\]](#cor:holonomic-landauer-floor){reference-type="ref" reference="cor:holonomic-landauer-floor"}

**Convergence and Universal Consequences (Section 6):**

-   Theorem [\[thm:five-way\]](#thm:five-way){reference-type="ref" reference="thm:five-way"}

-   Theorem [\[thm:england\]](#thm:england){reference-type="ref" reference="thm:england"}

-   Theorem [\[thm:finite-budget-no-collapse\]](#thm:finite-budget-no-collapse){reference-type="ref" reference="thm:finite-budget-no-collapse"}

**Primary Lean sources:**

-   `Leverage/Foundations.lean`

-   `Leverage/BridgeToDQ.lean`

-   `Leverage/ColumnComplexityBridge.lean`

-   `Leverage/DockingTheoryBridge.lean`

-   `LambdaDR.lean`

-   `Leverage.lean`




---

## Machine-Checked Proofs

All theorems are formalized in Lean 4:
- Location: `docs/papers/paper3_leverage/proofs/`
- Lines: 79763
- Theorems: 3498
- `sorry` placeholders: 0
