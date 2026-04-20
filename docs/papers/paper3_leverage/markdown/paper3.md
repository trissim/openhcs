# Paper: Molecular Docking: A Machine-Checked Theory of Exact Resolution, Complexity, and Thermodynamic Cost

**Status**: Draft-ready | **Lean**: 117726 lines, 4744 theorems

---

## Abstract

Continuous exact molecular resolution is physically unrealizable. A continuous binding landscape carries unbounded distinguishable detail, so exact state discrimination requires unbounded acquisition events. Under any positive per-bit dissipation floor, the required erasure work diverges. Finite molecular search therefore cannot target continuous exactness directly.

The physically admissible target is the decision quotient: the coarsest state abstraction that preserves the optimal-action relation. Structural Rank is the irreducible dimension of that quotient. The theorem spine identifies the same quantity as the Fisher-identifiable dimension in a canonical optimal-action observation geometry, and as the controller of quotient entropy. The resulting exact-resolution law is $$E \ge \mathrm{srank}(D)\,k_B T\ln 2,$$ with an equivalent entropy form $E \ge k_B T\,H_{\mathrm{nats}}(D)$.

Approximation enters as quotient-coarsened admissibility---a lawful mechanism that collapses the structural rank. In that regime, admissibility cannot increase Structural Rank and lowers it exactly by the number of erased exact relevant coordinates. Thermodynamic relief is therefore quantized: each erased coordinate removes one per-bit unit from the calibrated floor. This gives a physical criterion for lawful approximation and separates quotient coarsening from utility relaxations that refine, rather than coarsen, the exact decision object.

The molecular transport layer links this information law to chemistry. For exact binding resolution, $$\Delta G \ge \mathrm{srank}(D)\,k_B T\ln 2.$$ The framework is physically instantiated by a composed classical force-field interface (architecture and shellwise Lipschitz transport from bonded+Lennard--Jones+Coulomb Hamiltonians), an overdamped Langevin bridge to quotient-MCMC with certified discretization error, and a concrete spin-$\tfrac12$ substrate instantiation recovering the same $k_B T\ln 2$ readout floor. This inequality is an irreducible information-theoretic basement, not a full predictor of total empirical dissipation; strict-overhead theorems explicitly place realistic implementations above it when mismatch/residual terms are present. The same rank-indexed framework yields finite-time resolution-speed bounds, allosteric locality bounds on relevant coordinates, fault-tolerant proofreading overhead above logical rank, catalytic rank-reduction regimes, and finite-capacity universal limits, including Bekenstein-type throughput constraints. Molecular search is therefore thermodynamically governed by decision-relevant dimension, not by raw microscopic state count.

The top-level theorem path is constructive: the critical execution route from molecular input to exported ArrayDSL artifact is computable via rational certificates.

Keywords: molecular search, molecular docking, decision quotient, structural rank, Landauer bound


_Failed to convert lean_stats.tex_

# Introduction

Molecular search is often framed as continuous optimization on atomistic energy landscapes. Exact resolution in that setting requires discriminating a continuum of microstates. A finite-temperature resolver with positive per-bit dissipation cannot realize that limit: the number of required distinctions diverges, and the corresponding Landauer work diverges with it [@landauer1961irreversibility; @bennett1982thermodynamics]. Continuous exact resolution is therefore outside finite physical operation.

Finite operation requires a different exact object. The relevant object is the decision quotient: the coarsest abstraction of molecular state that preserves the optimal-action relation. Two states are equivalent when they induce the same optimizer. Proposition [\[prop:binding-as-exact-resolution\]](#prop:binding-as-exact-resolution){reference-type="ref" reference="prop:binding-as-exact-resolution"} identifies bound/unbound discrimination with exact resolution on that quotient when sufficiency holds.

Theorem [\[thm:admissible-docking-exhaustion\]](#thm:admissible-docking-exhaustion){reference-type="ref" reference="thm:admissible-docking-exhaustion"} states the corresponding exhaustion claim: under the no-collapse antecedent, admissible docking processes are exactly quotient-factorized resolution processes with structural-rank-controlled cost.

Structural Rank is the irreducible dimension of the quotient. It counts exactly the decision-relevant coordinates that cannot be erased without changing optimal decisions. The theorem spine identifies the same quantity with the Fisher-identifiable dimension in the canonical optimal-action observation geometry through an explicit optimizer-likelihood model, with parametric identifiable dimension under optimal-action observations, and with the entropy controller of quotient classes. The same layer also yields the Cramér--Rao non-identifiability consequence for non-relevant coordinates. Combinatorial irreducibility, statistical identifiability, and information content are therefore represented by one invariant.

The thermodynamic consequence is direct. Under Landauer calibration, $$E \ge \mathrm{srank}(D)\,k_B T\ln 2,$$ equivalently $E \ge k_B T\,H_{\mathrm{nats}}(D)$ for quotient entropy. Exact molecular resolution cost is set by decision-relevant dimension.

This rank-indexed Landauer floor is an irreducible logical minimum, not a direct predictor of total empirical binding dissipation. The strict-overhead branch theorems (Section [\[main-theorems\]](#main-theorems){reference-type="ref" reference="main-theorems"}) formalize why realistic kinetics and mismatch terms place physical implementations strictly above that basement.

Approximation enters through admissibility semantics. Quotient-coarsened admissibility preserves the optimizer quotient order and can only reduce Structural Rank. The reduction is quantitative for nested admissibility families: the rank drop equals the number of erased exact relevant coordinates, and a new bidirectional-factorization criterion identifies a theorem-level zero-collapse/no-rank-loss regime. At fixed tight rank and fixed erasure count, the collapsed rank-normal-form object is unique up to isomorphism, so admissibility relief is canonical as well as quantized. The same collapsed-rank layer now carries explicit category structure (identity/composition, initial object, binary min/max products-coproducts, additive monoidal tensor) together with a proved no-terminal-object obstruction in the unbounded setting and a bounded-rank slice in which finite limits, finite colimits, finite-family meet/join objects, arbitrary-family meet/join (complete-lattice form), and the corresponding monotonicity/idempotence/absorption laws plus binary rewrite calculus are recovered.

The molecular layer transports this structure to chemistry and mechanics. The binding bridge gives $$\Delta G \ge \mathrm{srank}(D)\,k_B T\ln 2,$$ so binding free-energy budgets are bounded below by exact decision-resolution cost, with an explicit nonnegative residual term measuring any gap above that floor. Mechanical-graph locality bounds the same rank by finite neighborhoods coupled to the active site, including contact-graph and one-hop shell regimes, and now admits a theorem-level distance-decay shell-envelope law with explicit exponential and polynomial specializations plus a polynomial $p$-series budget corollary; this converts long-range allosteric influence into finite relevance geometry with explicit monotone per-distance contribution control, while optimizer-class richness gives a complementary combinatorial lower bound on rank.

The physical-instantiation layer is explicit: composed bonded+Lennard--Jones+Coulomb Hamiltonians induce architecture and shellwise Lipschitz transport theorems; overdamped Langevin dynamics grounds the continuous-to-discrete bridge to quotient-MCMC with certified discretization error; and a concrete spin-$\tfrac12$ substrate instantiates the same rank-one decoherence floor. The top-level extraction path is constructive and computable via rational refinement certificates and exact ArrayDSL export.

The same rank-indexed framework extends to universal finite-resource limits. Catalytic and pathway constraints appear as rank-reduction or rank-preservation statements at fixed resolution objectives. Proofreading appears as fault-tolerant overhead above logical rank. Finite-capacity throughput constraints follow from the same entropy-resolution chain and yield Bekenstein-type finite-information ceilings in bounded substrates.

Lean 4 with Mathlib verifies the theorem chain from the stated definitions and antecedents. Empirical correspondence is controlled separately by the physical antecedents used for acquisition, quotient preservation, and thermodynamic calibration.

Sections [\[foundations\]](#foundations){reference-type="ref" reference="foundations"}--[\[probability-model\]](#probability-model){reference-type="ref" reference="probability-model"} define the exact object and prove the quotient--rank--entropy--Fisher identifications, including explicit likelihood-level Fisher recovery, parametric identifiability consequences, an initial-object universality characterization of the canonical encoding, a kernel-completion categorical endpoint theorem (object-level and hom-level equivalence plus universal no-collapse canonicity package), a measure-theoretic AE-germ and renormalization endpoint theorem (AE kernel instance plus scale-wise quotient invariance under operation-preserving RG flow), a measure-kernel detailed-balance/stationarity transport endpoint with deterministic measurable-kernel specialization and kernel-power stationarity/transport laws, a quotient-calculus/RG endpoint for measure-kernel dynamics (including recovery of transport theorems as quotient instances), a path-space Crooks/log-ratio lift over those kernel-power dynamics together with an expectation-level Jarzynski lift (including kernel-power transport identities), an explicit path-measure integral endpoint, a process-level path-measure transport endpoint, a canonical finite-horizon measurable-transition process endpoint (base/successor recursion plus scale/transport Jarzynski consequences), and a projective-consistency/Kolmogorov-extension interface endpoint, a concrete measurable stochastic transition-kernel semigroup instantiation via Giry bind with deterministic-to-stochastic semigroup embedding and concrete all-scale Jarzynski (expectation and path-integral) endpoints, and a noncanonical encoding transport theorem for structural rank and Landauer-floor invariance. Section [\[complexity-boundary\]](#complexity-boundary){reference-type="ref" reference="complexity-boundary"} gives certification hardness, admissibility-collapse theorems (including the zero-collapse bifactor criterion), sampled stability, and locality bounds with explicit exponential/polynomial distance-envelope allostery specializations and explicit polynomial-series instantiation. Section [\[main-theorems\]](#main-theorems){reference-type="ref" reference="main-theorems"} derives thermodynamic floors, finite-time speed limits, the admissibility speed--accuracy/on-rate tradeoff layer (including its zero-collapse specialization), the binding free-energy bridge, theorem-level Hopfield--Ninio proofreading overhead reduction with a kinetic branch-rate specialization, and named Crooks/Jarzynski reductions from the quotient-trajectory law together with a detailed-balance equilibrium calibration corollary. Section [\[five-way-equivalence\]](#five-way-equivalence){reference-type="ref" reference="five-way-equivalence"} states the convergence consequences. Section [\[related-work\]](#related-work){reference-type="ref" reference="related-work"} positions the results relative to thermodynamics and information theory. Appendix [\[appendix-lean\]](#appendix-lean){reference-type="ref" reference="appendix-lean"} records source provenance.


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

::: proposition
[]{#prop:binding-as-exact-resolution label="prop:binding-as-exact-resolution"} Let $D_{\mathrm{dock}}$ be a finite docking decision problem induced by an admissible action family and an interaction utility. Any physical process that produces a retained bound/unbound distinction realizes an exact resolver on the induced decision quotient exactly when the read coordinate family is sufficient for $D_{\mathrm{dock}}$.
:::

::: proof
*Proof.* A retained bound/unbound outcome is a state distinction in the optimizer quotient of $D_{\mathrm{dock}}$. Theorem [\[thm:resolution-sufficient\]](#thm:resolution-sufficient){reference-type="ref" reference="thm:resolution-sufficient"} identifies exact resolution with sufficiency of the read coordinate family. Theorems [\[thm:discrete-acquisition\]](#thm:discrete-acquisition){reference-type="ref" reference="thm:discrete-acquisition"} and [\[thm:one-transition-one-bit\]](#thm:one-transition-one-bit){reference-type="ref" reference="thm:one-transition-one-bit"} identify the physical distinction interface with finite binary acquisition events. ◻
:::

::: remark
[]{#rem:relaxation-distinction-counting label="rem:relaxation-distinction-counting"} Thermal relaxation and controlled computation share the same counting interface in this framework. A process that leaves a retained bound/unbound distinction has realized finite binary acquisition events regardless of whether the trajectory is externally programmed or spontaneously relaxational.
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

::: definition
[]{#def:docking-progress label="def:docking-progress"} Fix a decision problem $D$, a surjective state abstraction $\phi$, an optimizer summary $\sigma$, a coordinate set $I$, and a thermodynamic model $M$. Let $D_\sigma$ be the summary-induced decision problem. Progress toward admissible docking is the conjunction:

1.  $\phi$ factors through the decision quotient of $D$.

2.  $I$ is sufficient for $D_\sigma$.

3.  $\mathrm{srank}(D_\sigma) \le \mathrm{srank}(D)$.

4.  $M.\mathrm{joulesPerBit}\cdot \mathrm{srank}(D_\sigma) \le \mathrm{energyLowerBound}(M,|I|)$.
:::

::: theorem
[]{#thm:admissible-docking-exhaustion label="thm:admissible-docking-exhaustion"} Let $\phi : S \to T$ be surjective for a decision problem $D$, and suppose every decision-relevant distinction erased by $\phi$ is mapped to a physically feasible collapse at the canonical requirement profile. Let $\sigma$ be any optimizer summary, let $I$ be sufficient for $D$, and let $M$ have positive per-bit conversion constant. Then progress toward admissible docking holds in the sense of Definition [\[def:docking-progress\]](#def:docking-progress){reference-type="ref" reference="def:docking-progress"}.
:::

::: proof
*Proof.* The feasible-collapse map hypothesis forces quotient factorization by Theorem [\[thm:feasible-collapse-factors\]](#thm:feasible-collapse-factors){reference-type="ref" reference="thm:feasible-collapse-factors"}. Summary-level sufficiency inherits from exact sufficiency, structural rank is monotone under optimizer summaries, and the bounded-acquisition thermodynamic theorem gives the structural-rank cost floor for the summary problem. ◻
:::

::: theorem
[]{#thm:admissibility-progress-monotone label="thm:admissibility-progress-monotone"} Let $F$ be a nested quantitative admissibility family and $\Delta \ge 0$. Then $$\mathrm{srank}(D_{F,\varepsilon+\Delta}) \le \mathrm{srank}(D_{F,\varepsilon}).$$
:::

::: proof
*Proof.* The quantitative collapse identity gives $$\mathrm{srank}(D_{F,\varepsilon})=
\mathrm{srank}(D_{F,\varepsilon+\Delta})+
\mathrm{CollapseCount}_F(D,\varepsilon,\Delta),$$ and collapse counts are nonnegative. ◻
:::

Under the no-collapse antecedent, correct docking processes are exactly quotient-factorized admissible-resolution processes with structural-rank-controlled cost. Progress claims are therefore statements inside that admissible class.

## Fisher Dimension

The Fisher statements in this section are for the *canonical optimal-action observation geometry* used by the formalization. In that geometry, the Fisher matrix is induced by coordinate relevance indicators, so rank/Fisher-identifiability claims are exact statements about what optimal-action observations can and cannot resolve.

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

::: theorem
[]{#thm:fisher-likelihood-indicator label="thm:fisher-likelihood-indicator"} On the canonical binary state space, the Fisher information computed from the explicit optimizer-likelihood model coincides coordinatewise with the relevance indicator: $$I^{\mathrm{lik}}_D(i)=\mathrm{FisherScore}_D(i)=\mathbf{1}_{\{i\in \mathrm{Rel}(D)\}}.$$
:::

::: proof
*Proof.* The explicit likelihood model has two optimizer-observation outcomes at the reference parameter. The mechanized score definition assigns unit-magnitude score to relevant coordinates and zero score to irrelevant coordinates, so the expected score square is exactly the relevance indicator. ◻
:::

::: theorem
[]{#thm:parametric-fisher-identifiable-dimension label="thm:parametric-fisher-identifiable-dimension"} Let $\{D_\theta\}_{\theta\in\Theta}$ be a parametric family of decision problems on the canonical binary state space. For every parameter value $\theta$, $$\operatorname{rank}(I_{D_\theta})=\mathrm{srank}(D_\theta),$$ and in the canonical coordinate basis, $$(I_{D_\theta})_{ii}=\mathbf{1}_{\{i\in \mathrm{Rel}(D_\theta)\}}.$$ Hence structural rank is exactly the Fisher-identifiable parameter dimension from optimal-action observations. Within this canonical observation geometry, this is an exact equivalence; outside it, channel-specific identification hypotheses are handled by Theorem [\[thm:fisher-observation-channel-transport\]](#thm:fisher-observation-channel-transport){reference-type="ref" reference="thm:fisher-observation-channel-transport"}.
:::

::: proof
*Proof.* The family theorem specializes the Fisher-rank identity pointwise in $\theta$. The diagonal recovery statement is the mechanized equivalence between diagonal Fisher mass and coordinate relevance in the canonical basis. ◻
:::

::: theorem
[]{#thm:cramer-rao-nonidentifiable-irrelevant label="thm:cramer-rao-nonidentifiable-irrelevant"} Fix $\theta$ and observe only optimal-action outputs. Under the abstract optimizer-observation Cramér--Rao inequality used in the formalization, if coordinate $i$ is non-relevant for $D_\theta$, then every unbiased estimator of that coordinate from optimal-action data has unbounded variance: $$\operatorname{Var}(\widehat{\theta_i})=\infty.$$
:::

::: proof
*Proof.* For non-relevant coordinates the Fisher score is zero, so the optimizer-observation Cramér--Rao lower envelope is top (infinite). Any unbiased estimator variance bounded below by that envelope must therefore be infinite. ◻
:::

::: theorem
[]{#thm:fisher-observation-channel-transport label="thm:fisher-observation-channel-transport"} Let $D$ be a finite decision problem and let an observation channel provide Fisher entries $I^{\mathrm{obs}}_D(i)$ with theorem-level identification $$I^{\mathrm{obs}}_D(i)=\mathrm{FisherScore}_D(i)
\quad\text{for every coordinate }i.$$ Then:

1.  total observed Fisher mass equals structural rank, $$\sum_i I^{\mathrm{obs}}_D(i)=\mathrm{srank}(D),$$

2.  coordinate relevance is recoverable by the same diagonal criterion, $$i\in \mathrm{Rel}(D) \iff I^{\mathrm{obs}}_D(i)=1.$$
:::

::: proof
*Proof.* The identification hypothesis replaces each channel Fisher entry by the canonical Fisher score. Summing gives the structural-rank identity, and the coordinatewise claim is exactly the Fisher-score relevance equivalence transported through the same identification. ◻
:::

::: theorem
[]{#thm:fisher-noisy-partial-channel-example label="thm:fisher-noisy-partial-channel-example"} Let a non-canonical observation channel carry explicit partial/noisy readout metadata together with a theorem-level debiasing certificate that identifies each debiased Fisher entry with the canonical relevance Fisher score. Then:

1.  total debiased Fisher mass equals structural rank, $$\sum_i I^{\mathrm{debiased}}_D(i)=\mathrm{srank}(D),$$

2.  relevance is recoverable by the same indicator test, $$i\in\mathrm{Rel}(D)\iff I^{\mathrm{debiased}}_D(i)=1.$$
:::

::: proof
*Proof.* Package the debiasing certificate as an instance of the general observation-channel Fisher interface and apply Theorem [\[thm:fisher-observation-channel-transport\]](#thm:fisher-observation-channel-transport){reference-type="ref" reference="thm:fisher-observation-channel-transport"}. The two claims are exactly the transported sum and coordinatewise recovery clauses. ◻
:::

Structural rank therefore has three exact readings in the present development: combinatorial irreducible-coordinate count, quotient entropy controller, and canonical Fisher-identifiable dimension under optimal-action observations.

## Canonical Exact-Resolution Encoding

::: theorem
[]{#thm:dof-srank label="thm:dof-srank"} For every bounded decision system $A$, $$\mathrm{srank}(\mathrm{canonicalDP}(A)) = \mathrm{DOF}(A).$$
:::

::: proof
*Proof.* Write $n = \mathrm{DOF}(A)$. By Definition [\[def:canonical-dp\]](#def:canonical-dp){reference-type="ref" reference="def:canonical-dp"}, the state space is $\mathrm{Fin}\;n \to \mathrm{Bool}$, query action $\mathrm{inl}(i)$ has utility $2$ exactly when coordinate $i$ is true and utility $0$ otherwise, and the fallback action has utility $1$. Fix any coordinate $i$ and choose two states that agree everywhere except at $i$, with one state setting $i$ to true and the other setting $i$ to false; then $\mathrm{inl}(i)$ is optimal in the first state and not optimal in the second, so erasing coordinate $i$ changes the optimizer. Thus every coordinate in $\mathrm{Fin}\;n$ is relevant, the relevant-coordinate set has cardinality $n$, and the structural rank is $n$. Substituting $n = \mathrm{DOF}(A)$ gives the claim. ◻
:::

The canonical encoding provides the bridge from the abstract exact object to a degree-of-freedom count carried by a declared architecture.

::: theorem
[]{#thm:canonical-initiality label="thm:canonical-initiality"} Fix $n$. In the category of exact finite-resolution objects over the canonical binary state space with decision-preserving abstraction morphisms, the canonical encoding object is initial: for every object $Y$, there is a unique morphism $$\mathrm{canonicalDP}(n) \longrightarrow Y.$$
:::

::: proof
*Proof.* Each object carries a surjective abstraction witness from the canonical state space. The induced map is the unique morphism commuting with the canonical abstraction interface, which is exactly the mechanized initiality statement. ◻
:::

::: corollary
[]{#thm:canonical-initiality-srank label="thm:canonical-initiality-srank"} The canonical initiality package includes the structural-rank identity $$\mathrm{srank}(\mathrm{canonicalDP}(n))=n,$$ hence for every architecture $A$, $$\mathrm{srank}(\mathrm{canonicalDP}(A))=\mathrm{DOF}(A).$$
:::

::: theorem
[]{#thm:kernel-quotient-universality-canonicity label="thm:kernel-quotient-universality-canonicity"} Fix an operation-equipped carrier with a designated equivalence relation identifying exactly the points that agree under the operation.

In the category of surjective operation-preserving abstractions from that carrier:

1.  the identity abstraction is initial;

2.  every abstraction factors uniquely through the induced kernel quotient map;

3.  an abstraction satisfies no-collapse iff there is a unique canonical isomorphism from the identity abstraction object to it.

Moreover, both the optimizer quotient $s\mapsto \operatorname{Opt}(s)$ and the eventual-germ quotient of trajectories $(\mathbb{N}\to\alpha)/\sim_{\mathrm{eventual}}$ instantiate this same theorem schema.
:::

::: proof
*Proof.* The operation-kernel schema is mechanized abstractly, with initiality and unique kernel factorization proved once at schema level. The no-collapse clause is proved as an equivalence with unique canonical isomorphism to the identity abstraction object. The decision-quotient and eventual-germ constructions are then provided as concrete schema instances, yielding the two instance-level corollaries directly. ◻
:::

::: theorem
[]{#thm:kernel-completion-categorical-endpoint label="thm:kernel-completion-categorical-endpoint"} For operation-equipped systems $(S,\mathcal O,\operatorname{op})$, kernel completion $$(S,\mathcal O,\operatorname{op})\mapsto
\bigl(S,\mathcal O,\operatorname{op},\sim_{\operatorname{op}}\bigr),
\qquad
x\sim_{\operatorname{op}}y\iff \operatorname{op}(x)=\operatorname{op}(y),$$ admits the following mechanized endpoint package:

1.  object-level equivalence between operation systems and operation-kernel schemas;

2.  hom-level full-faithful equivalence for the kernel-completion embedding;

3.  for each completed schema, initiality of the identity abstraction object, unique factorization of every surjective operation-preserving abstraction through the kernel quotient, and no-collapse iff unique canonical isomorphism to that identity object.

Moreover, both the decision-quotient optimizer construction and the eventual-germ trajectory construction are explicit instances of this package; in the eventual-germ instance, schema relation equality is exactly eventual equality.
:::

::: proof
*Proof.* The mechanization first defines bare operation systems and proves kernel completion/forget are inverse at object level. It then proves the corresponding hom-level equivalence for the completion embedding. The universal factorization and no-collapse canonicity package is proved at schema level and instantiated for decision and eventual-germ systems; the eventual-germ relation identification follows by quotient soundness/exactness. ◻
:::

::: theorem
[]{#thm:ae-rg-kernel-endpoint label="thm:ae-rg-kernel-endpoint"} Fix a measurable state space $(X,\mu)$ and codomain type $Y$. Consider the almost-everywhere germ construction $$(X\to Y)/{=\!_{\mu\text{-a.e.}}}.$$ Then:

1.  the AE-germ relation is exactly the kernel relation of the quotient-map operation and therefore instantiates the same kernel-completion endpoint package (initiality, unique quotient factorization, and no-collapse iff unique canonical isomorphism);

2.  the corresponding operation-system embedding is full-faithful at hom level;

3.  for any operation-preserving renormalization flow on this kernel schema, every induced map on quotient classes is identity at every scale;

4.  if each renormalization scale map is surjective, scale-wise no-collapse is equivalent to unique canonical identification with the exact carrier.
:::

::: proof
*Proof.* The AE relation is formalized as filter eventual equality under $\mathrm{ae}(\mu)$, and the quotient operation is the canonical class map. Kernel-completion endpoint and hom-level full-faithful results are then instantiated directly from the abstract schema theorems. The renormalization statements are proved in the same schema by requiring operation invariance along scales and, in the surjective branch, reusing the no-collapse/unique-isomorphism theorem at each scale abstraction object. ◻
:::

::: theorem
[]{#thm:measure-kernel-db-stationary-transport label="thm:measure-kernel-db-stationary-transport"} Fix source and target measure-kernel semigroup models, each equipped with a detailed-balance layer (detailed balance implies stationarity), and a transport witness consisting of:

1.  a measurable state map,

2.  a kernel map,

3.  evolve/pushforward commutation,

4.  detailed-balance preservation along that map.

Then:

1.  stationarity transports along the witness;

2.  detailed balance transports along the witness;

3.  transported detailed balance implies transported stationarity;

4.  for any scale-indexed source/target kernel flows aligned by the kernel map, scale-wise source detailed balance yields scale-wise transported target stationarity.

In particular, the endpoint package gives a single reusable transport interface for measure-kernel dynamics and detailed-balance/stationarity claims.
:::

::: proof
*Proof.* Stationarity transport is a direct consequence of the evolve/pushforward commutation identity. Detailed-balance transport is the dedicated preservation axiom of the witness. Composing that transported detailed-balance statement with the target detailed-balance-implies-stationarity law gives transported stationarity. The scale-wise theorem is the same argument applied pointwise to aligned flow indices. ◻
:::

::: corollary
[]{#thm:deterministic-kernel-scale-stationary label="thm:deterministic-kernel-scale-stationary"} For any measurable state space, deterministic measurable endomaps form a measure-kernel semigroup via pushforward evolution. Using measure invariance as the detailed-balance layer, any scale-indexed deterministic flow that preserves a declared measure at each scale is stationary at each scale in the measure-kernel semantics.
:::

::: proof
*Proof.* The deterministic semigroup is the pushforward-by-map instance. Invariance at a scale is exactly the detailed-balance predicate in that instance, and the general scale theorem (detailed balance implies stationarity at each index) applies directly. ◻
:::

::: theorem
[]{#thm:kernel-power-stationarity-transport label="thm:kernel-power-stationarity-transport"} In any measure-kernel semigroup:

1.  if a measure is stationary for a kernel element $K$, then it is stationary for every semigroup power $K^n$;

2.  if detailed balance holds for $K$, then every $K^n$ is stationary.

Moreover, for semigroup-homomorphic transport between source and target kernel semigroups:

1.  powers are preserved by transport ($\Phi(K^n)=\Phi(K)^n$);

2.  stationarity of source powers transports to stationarity of target powers;

3.  if detailed balance holds at source for $K$, then all transported powers $\Phi(K)^n$ are stationary in the target model.

4.  for any scale flow $F$, each scale kernel is exactly the corresponding power of the one-step kernel $F_1$;

5.  therefore detailed balance at $F_1$ implies stationarity at every scale.

6.  if a semigroup-homomorphic transport aligns one-step kernels of source and target flows, then source detailed balance at one step implies transported target stationarity at every scale.
:::

::: proof
*Proof.* Power stationarity is proved by induction using the semigroup evolve-composition law. The detailed-balance clause is immediate from the detailed-balance-implies-stationarity axiom plus the power-stationarity theorem. For transport, the semigroup-homomorphism axioms give power preservation, and the previously proved stationarity transport theorem is applied to each source power; composing with source detailed-balance-to-stationarity gives the transported-power clause. The scale-flow claims follow by proving $F_n=(F_1)^n$ from the additive flow law, then substituting into the power-stationarity theorem. The final clause combines one-step alignment with the same power representation on both flows and applies transported stationarity pointwise in scale. ◻
:::

::: theorem
[]{#thm:measure-kernel-quotient-rg-endpoint label="thm:measure-kernel-quotient-rg-endpoint"} Fix a source measure-kernel semigroup and a measure-theoretic quotient calculus (surjective measurable quotient map, descended quotient semigroup, and evolution-commutation law). Then stationarity descends from source kernels to quotient kernels.

If the quotient calculus is semigroup-homomorphic, stationarity descends for all kernel powers. If, in addition, detailed balance is preserved by the quotient calculus, then:

1.  source detailed balance implies stationarity of all descended quotient powers;

2.  one-step detailed balance on a source scale flow implies all-scale stationarity on every aligned quotient flow.

For RG dynamics modeled by scale-indexed kernel-semigroup endomorphisms on source and quotient kernels that commute with quotient descent:

1.  endomorphisms preserve kernel powers;

2.  RG action commutes with quotient kernel powers at every RG scale and power index;

3.  if source detailed balance is stable under source RG steps, then every RG-renormalized quotient kernel power is stationary under the descended measure.

These clauses are bundled as a single endpoint package theorem combining quotient descent, kernel-power dynamics, and RG compatibility.
:::

::: proof
*Proof.* All descent statements are direct instances of the transport theorems once the quotient calculus is reinterpreted as a transport witness (and semigroup-homomorphic/detailed-balance-preserving witness in the stronger layers). The RG-power commutation statement follows from endomorphism preservation of semigroup powers together with quotient-level RG/source RG commutation on one-step kernels. RG-stable detailed balance in the source model is pushed through the detailed-balance quotient calculus, then combined with the quotient power-stationarity theorem and the RG commutation identity to obtain stationarity of RG-renormalized quotient powers. ◻
:::

::: corollary
[]{#thm:measure-kernel-transport-quotient-instance-recovery label="thm:measure-kernel-transport-quotient-instance-recovery"} Any surjective semigroup-homomorphic measure-kernel transport witness is an instance of semigroup quotient calculus, and any surjective detailed-balance-preserving semigroup-homomorphic witness is an instance of detailed-balance quotient calculus. Under these identifications, the existing transported kernel-power stationarity theorems are recovered as immediate quotient-calculus instances.
:::

::: proof
*Proof.* Instantiate quotient state space by the transport codomain and quotient map by the witness state map, with surjectivity as hypothesis. The semigroup and detailed-balance preservation laws of the witness exactly match the quotient-calculus structure fields, so the previously proved quotient-calculus descent theorems reduce definitionally to the original transport theorems. ◻
:::

::: theorem
[]{#thm:path-space-crooks-kernel-power-transport label="thm:path-space-crooks-kernel-power-transport"} Fix a measure-kernel semigroup and a path-space Crooks model assigning a pathwise log-ratio observable.

Then:

1.  stationarity of a kernel implies vanishing log ratio on all its powers;

2.  detailed balance of a kernel implies vanishing log ratio on all its powers;

3.  detailed balance at one step of a scale flow implies vanishing log ratio at every scale.

For semigroup-homomorphic transport between source and target models with path transport preserving log-ratio semantics:

1.  kernel-power log ratios transport exactly;

2.  source detailed balance implies vanishing transported target log ratio on all transported kernel powers.

3.  for aligned source/target scale flows under a detailed-balance-preserving semigroup-homomorphic map, one-step source detailed balance implies vanishing target log ratio at every aligned scale (including mapped-source-path specialization).
:::

::: proof
*Proof.* All three source-side vanishing claims are immediate from the Crooks-model axiom "stationary implies zero log ratio" combined with the kernel-power and scale-flow stationarity theorems. Transported equality follows by applying the transport preservation law to source kernel powers and rewriting by semigroup-homomorphic power preservation. The transported-vanishing power clause composes this equality with the source detailed-balance-to-zero-power-log-ratio result. For aligned flows, the one-step transport theorem yields target stationarity at each scale, and the same Crooks axiom gives scale-wise target vanishing, with mapped-source-path specialization by direct substitution. ◻
:::

::: theorem
[]{#thm:path-space-jarzynski-kernel-power-transport label="thm:path-space-jarzynski-kernel-power-transport"} Fix a measure-kernel semigroup, a path-space Crooks model, and an expectation-level Jarzynski model with the calibration axiom: if pathwise log ratio vanishes identically for a kernel/measure pair, then the corresponding exponential-log-ratio expectation equals $1$.

Then:

1.  stationarity of a kernel implies unit Jarzynski expectation for all kernel powers;

2.  detailed balance of a kernel implies unit Jarzynski expectation for all kernel powers;

3.  one-step detailed balance of a scale flow implies unit Jarzynski expectation at every scale.

For semigroup-homomorphic transport with expectation preservation:

1.  kernel-power Jarzynski expectations transport exactly;

2.  source detailed balance implies unit transported target Jarzynski expectation for all transported kernel powers.
:::

::: proof
*Proof.* Each source-side clause composes the already-proved Crooks log-ratio vanishing theorem with the Jarzynski calibration axiom "log-ratio zero implies expectation one". For transport, the semigroup-homomorphism power identity rewrites transported powers into images of source powers, and the expectation-preservation hypothesis gives exact equality. The detailed-balance transport clause then follows by combining that equality with the source detailed-balance-to-unit-power-expectation result. ◻
:::

::: theorem
[]{#thm:path-measure-jarzynski-integral-transport label="thm:path-measure-jarzynski-integral-transport"} Fix a measure-kernel semigroup with a path-space Crooks model and an explicit path-measure expectation model whose Jarzynski observable is represented by $$\mathbb E\bigl[e^{-\Lambda}\bigr]
=
\int e^{-\Lambda(p)}\,d\mathbb P_{K,\pi}(p),
\qquad
\Lambda(p)=\log\frac{P_{\mathrm f}}{P_{\mathrm r}}(p).$$ Then:

1.  if pathwise log ratio vanishes for $(K,\pi)$, the path integral above equals $1$;

2.  stationarity (resp. detailed balance) of $K$ implies unit integral on every kernel power $K^n$;

3.  one-step detailed balance on a scale flow implies unit integral at every scale.

For semigroup-homomorphic Jarzynski transport between source and target models:

1.  kernel-power path integrals transport exactly;

2.  source detailed balance implies unit transported target path integral on every transported kernel power.
:::

::: proof
*Proof.* The first clause rewrites the explicit integral by the expectation-identification axiom and applies the Jarzynski calibration law (zero log ratio implies expectation one). The stationarity and detailed-balance power statements then instantiate the already-proved Crooks vanishing theorems before applying that same rewrite. The scale statement is the one-step-to-all-scales Crooks vanishing theorem composed with the explicit integral rewrite. Transported equality follows by rewriting both source and target expectations as their explicit path integrals and inserting the previously proved transported expectation identity on kernel powers. The transported detailed-balance clause composes that equality with the source detailed-balance-to-unit-integral result. ◻
:::

::: theorem
[]{#thm:path-process-transport-endpoint label="thm:path-process-transport-endpoint"} Fix source/target measure-kernel semigroups with a semigroup-homomorphic transport map, source/target path-space Crooks models, and explicit path-measure expectation models. Assume a process-level transport witness: mapped paths are measurable, source path measures push forward to target path measures, and the Crooks exponential path integral is preserved under transport.

Then:

1.  kernel-power expectation transport is recovered at the Jarzynski level;

2.  kernel-power path-integral transport holds exactly;

3.  source detailed balance implies unit transported target path-integral Jarzynski identity on all transported kernel powers.

Moreover, for detailed-balance-preserving semigroup transport with aligned source/target scale flows, one-step source detailed balance implies unit target path-integral Jarzynski identity at every aligned scale.
:::

::: proof
*Proof.* The first three clauses are exactly the process-transport theorems: process-level transport induces the expectation-level transport witness, which yields kernel-power expectation transport; the process witness also gives direct kernel-power path-integral transport; composing that identity with source detailed-balance-to-unit-integral on powers gives the transported unit identity. The aligned-scale clause applies the transported one-step detailed-balance Crooks-vanishing theorem at each scale and then rewrites by the explicit path-integral endpoint theorem "zero log ratio implies unit integral". ◻
:::

::: theorem
[]{#thm:measurable-transition-finite-horizon-canonical-endpoint label="thm:measurable-transition-finite-horizon-canonical-endpoint"} For measurable transition kernels, the canonical finite-horizon path-measure constructor is given by:

1.  horizon-$0$ path measure equals the initial measure;

2.  horizon-$(n+1)$ path measure is obtained from horizon-$n$ by one bind-extension step with the same transition kernel.

On this canonical finite-horizon path family:

1.  one-step detailed balance implies unit path-integral Jarzynski identity at every scale;

2.  under process-level transport, source detailed balance implies unit transported target path-integral Jarzynski identity on all transported kernel powers;

3.  for detailed-balance-preserving aligned scale-flow transport, one-step source detailed balance implies unit target path-integral Jarzynski identity at every aligned scale.
:::

::: proof
*Proof.* The first two clauses are exactly the definitional recursion equations of the canonical finite-horizon path-measure constructor (base and successor steps). The remaining clauses are the finite-horizon specializations of the previously established abstract path-integral endpoint theorems, instantiated with the canonical measurable-transition finite-horizon path space and its induced Crooks/Jarzynski structures. ◻
:::

::: theorem
[]{#thm:measurable-transition-projective-kolmogorov-endpoint label="thm:measurable-transition-projective-kolmogorov-endpoint"} For canonical finite-horizon measurable-transition path measures, any projective-consistency witness (measurable truncation maps with measure-level consistency) yields:

1.  exact marginal compatibility across horizons;

2.  exact specialization of that compatibility to kernel powers;

3.  exact specialization to additive scale-flow kernels.

For any Kolmogorov-extension interface (infinite-path measure with declared finite-horizon marginal recovery):

1.  finite-horizon marginals are exactly the canonical finite-horizon path measures;

2.  one-step detailed balance implies unit Jarzynski path-integral identity for every finite-horizon marginal at every scale;

3.  under detailed-balance-preserving aligned scale-flow transport, one-step source detailed balance implies unit target finite-horizon marginal path-integral identity at every aligned scale.
:::

::: proof
*Proof.* The first three clauses are direct restatements of the projective-consistency transport equations and their kernel-power/scale-flow substitutions. The extension clauses are immediate from the declared marginal-recovery identity: each target integral over a projected infinite-path measure rewrites to the corresponding canonical finite-horizon integral, after which the previously proved finite-horizon scale and transported-scale Jarzynski unit-integral theorems apply. ◻
:::

::: theorem
[]{#thm:measurable-transition-projective-canonical-instance label="thm:measurable-transition-projective-canonical-instance"} If canonical finite-horizon measurable-transition path measures satisfy the explicit truncation-family marginal equations, then the projective-consistency interface is inhabited by that canonical truncation witness.
:::

::: proof
*Proof.* Instantiate the projective-consistency structure with the canonical truncation maps and the supplied marginal equations. The truncation measurability premise is now fully constructive: the canonical truncation map is explicitly built from horizon transport plus recursive tail-drop maps, and its measurability is proved directly in Lean (L588) rather than postulated. ◻
:::

::: theorem
[]{#thm:measurable-transition-constructive-truncation label="thm:measurable-transition-constructive-truncation"} For any measurable state space and any horizons $m\le n$, the canonical finite-path truncation map $$\mathrm{truncate}_{m\leftarrow n}:
\mathrm{Path}_n\to \mathrm{Path}_m$$ is measurable. The map is constructed explicitly from horizon-index transport together with recursive tail-drop maps.
:::

::: proof
*Proof.* The canonical truncation map is defined as composition of two measurable pieces: equality-transport between horizon index presentations and recursive tail-drop. Measurability follows by composition. ◻
:::

::: theorem
[]{#thm:continuous-time-continuous-state-interface label="thm:continuous-time-continuous-state-interface"} For measurable transition kernels on a measurable state space, a continuous-time semigroup interface (real-time kernel family with additive composition law) canonically discretizes to the proved additive `ScaleFlow` interface at any nonnegative time step $\delta$. In parallel, a continuous-state path extension interface with measurable time-evaluation maps yields measurable finite-time projection maps, including all two-time evaluation pairs.
:::

::: proof
*Proof.* Discretization sends scale index $n$ to physical time $n\delta$ and uses the additive continuous-time semigroup law to discharge the discrete scale-flow composition law. Two-time measurability is the product measurability of the two declared evaluation maps. ◻
:::

::: theorem
[]{#thm:measurable-transition-kernel-concrete-endpoint label="thm:measurable-transition-kernel-concrete-endpoint"} For any measurable state space $S$, measurable stochastic kernels $$K:S\to \mathcal M(S)$$ form a concrete measure-kernel semigroup with:

1.  identity kernel given by Dirac transition,

2.  composition given by measure bind,

3.  evolution acting on measures by bind.

In this concrete semigroup, one-step detailed balance for a scale flow implies all-scale stationarity, and for any path-space Crooks model on the same semigroup, one-step detailed balance implies vanishing pathwise log ratio at all scales.

Moreover, deterministic measurable kernels embed semigroup-homomorphically into this stochastic-kernel semigroup (via Dirac lifting), and stationarity of deterministic kernel powers transports to stationarity of the corresponding lifted stochastic kernel powers.
:::

::: proof
*Proof.* The semigroup laws are discharged by the Giry monad identities (Dirac left/right unit and bind associativity). The all-scale stationarity statement is the abstract one-step-to-all-scales theorem instantiated on this semigroup. The path-space consequence then applies the abstract Crooks log-ratio vanishing theorem under the same one-step detailed-balance hypothesis. The deterministic embedding is the Dirac lift of measurable endomaps, and semigroup-homomorphic stationarity transport gives the kernel-power preservation clause. ◻
:::

::: corollary
[]{#thm:measurable-transition-jarzynski-concrete-endpoint label="thm:measurable-transition-jarzynski-concrete-endpoint"} In the measurable transition-kernel semigroup on any measurable state space, for any path-space Crooks model and any expectation-level Jarzynski model over that Crooks layer, one-step detailed balance for a scale flow implies unit Jarzynski expectation at every scale.
:::

::: proof
*Proof.* Instantiate the abstract expectation-level one-step-to-all-scales theorem on the measurable transition-kernel semigroup. ◻
:::

::: corollary
[]{#thm:measurable-transition-path-measure-jarzynski-concrete-endpoint label="thm:measurable-transition-path-measure-jarzynski-concrete-endpoint"} In the measurable transition-kernel semigroup on any measurable state space, for any path-space Crooks model equipped with a measurable path space and an explicit path-measure expectation model, one-step detailed balance for a scale flow implies $$\int \exp\bigl(-\Lambda_n(p)\bigr)\,d\mathbb P_n(p)=1$$ at every scale index $n$.
:::

::: proof
*Proof.* Instantiate the abstract path-measure integral theorem for one-step detailed-balance scale flows on the measurable transition-kernel semigroup. ◻
:::

::: theorem
[]{#thm:encoding-transport-rank-energy label="thm:encoding-transport-rank-energy"} Let $D_S$ and $D_T$ be decision problems over coordinate spaces of the same finite coordinate dimension. Suppose there is a state-space equivalence $$e:S\xrightarrow{\cong}T$$ such that:

1.  optimizer fibers are preserved, i.e. $\operatorname{Opt}_{D_T}(e(s))=\operatorname{Opt}_{D_S}(s)$ for all $s$, and

2.  for each tested coordinate $i$, agreement on all coordinates except $i$ is equivalent under transport by $e$.

Then for every coordinate $i$, $$i\text{ is relevant for }D_S\iff i\text{ is relevant for }D_T,$$ hence $$\mathrm{srank}(D_S)=\mathrm{srank}(D_T),$$ and for every declared linear per-bit thermodynamic model $M$, $$\mathrm{energyLowerBound}(M,\mathrm{srank}(D_S))
=
\mathrm{energyLowerBound}(M,\mathrm{srank}(D_T)).$$
:::

::: proof
*Proof.* The relevance predicate is existential in two ingredients only: coordinate-agreement away from a tested index and optimizer-set inequality. By hypothesis, both are preserved and reflected by the transport equivalence. Therefore relevance is equivalent coordinatewise, so the relevant-coordinate sets have equal cardinality and structural ranks coincide. The linear per-bit floor then transports immediately by substituting equal rank. ◻
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
[]{#thm:optimizer-class-richness-rank-lower-bound label="thm:optimizer-class-richness-rank-lower-bound"} Let $D$ be a finite decision problem on a binary coordinate state space. If $$2^k \le \mathrm{numOptClasses}(D),$$ then $$k \le \mathrm{srank}(D).$$ Equivalently, for every integer $k>0$ with $$k \le \mathrm{numOptClasses}(D),$$ one has $$\frac{\log k}{\log 2} \le \mathrm{srank}(D).$$
:::

::: proof
*Proof.* The binary optimizer-class upper bound gives $\mathrm{numOptClasses}(D) \le 2^{\mathrm{srank}(D)}$. If $2^k \le \mathrm{numOptClasses}(D)$, then $2^k \le 2^{\mathrm{srank}(D)}$, so monotonicity of powers with base $2$ yields $k \le \mathrm{srank}(D)$. The logarithmic form is the same inequality written on the $\log_2$ scale. ◻
:::

::: theorem
[]{#thm:optimizer-class-richness-nonbinary-vc label="thm:optimizer-class-richness-nonbinary-vc"} Let $D$ be a finite decision problem with an alphabet-size parameter $q>1$ and theorem-level class envelope $$\mathrm{numOptClasses}(D) \le q^{\mathrm{srank}(D)}.$$ Then every witnessed richness lower bound $$q^k \le \mathrm{numOptClasses}(D)$$ forces $$k \le \mathrm{srank}(D).$$ In particular, any VC-style witness encoded as such a $q$-ary growth lower bound transfers directly to the same structural-rank floor.
:::

::: proof
*Proof.* Compose the declared lower bound $q^k \le \mathrm{numOptClasses}(D)$ with the declared upper envelope $\mathrm{numOptClasses}(D) \le q^{\mathrm{srank}(D)}$. Monotonicity of powers for base $q>1$ yields $k \le \mathrm{srank}(D)$. The VC-style statement is exactly this argument with $k$ instantiated by the chosen growth witness. ◻
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

Exact molecular docking has a genuine tractability boundary because the exact object already carries both qualitative and quantitative certification lower bounds. Proposition [\[prop:binding-as-exact-resolution\]](#prop:binding-as-exact-resolution){reference-type="ref" reference="prop:binding-as-exact-resolution"} identifies binding discrimination with exact quotient resolution under sufficiency. General exact sufficiency certification contains a hardness core. Sound checking requires witness budget. Molecular locality, sampling hypotheses, and concrete scorer approximations then carve out theorem-backed low-rank and stability regimes inside that harder ambient problem class. The claims in this section isolate that boundary before the thermodynamic lower bounds are applied.

#### The Hardness Boundary (Sections 4.1--4.2).

The first two blocks establish the worst-case baseline: exact certification is structurally hard, and sound certification requires exponentially large witness access in the hard family. This is the ambient boundary that any physically realistic escape route must respect.

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

#### Physical Escape Hatches: Locality and Mechanics (Sections 4.3--4.4).

The next two blocks show why molecular search is not generic SAT-like worst case in practice. Finite interaction cutoffs and rigid mechanical coupling restrict the set of coordinates that can affect optimizer classes. Physics therefore constrains rank before any thermodynamic statement is invoked.

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

## Allosteric and Hierarchical Extensions

::: theorem
[]{#thm:allosteric-srank-graph label="thm:allosteric-srank-graph"} Let $G$ be a rigid mechanical graph on the protein atoms, let $P_{\mathrm{act}}$ be an active-pocket set, and define the graph-level mechanical neighborhood $$\mathcal{N}_G(P_{\mathrm{act}})
=
\{x : \exists y \in P_{\mathrm{act}},\ \mathrm{RigidPath}_G(x,y)\}.$$ If every currently relevant protein atom lies in that mechanical neighborhood, then $$\mathrm{srank}(D_{\mathrm{dock}})
\le
3\bigl|\mathcal{N}_G(P_{\mathrm{act}})\bigr| + 3L.$$
:::

::: proof
*Proof.* The cutoff-local theorem already bounds docking rank by three times the number of protein atoms that remain decision-relevant, plus the ligand contribution. The reusable protein mechanical graph API packages the allosteric side as a finite graph-neighborhood cover: once every relevant atom lies in $\mathcal{N}_G(P_{\mathrm{act}})$, the generic atom-cover theorem applies directly to that neighborhood and yields the displayed bound. ◻
:::

**Physical Significance.** This theorem formalizes the mechanistic basis of allostery. Distal mutations can influence binding only when they remain connected to the active site through a rigid mechanical pathway. If that pathway is broken (for example, by a floppy hinge), the distal coordinate becomes decision-irrelevant, structural rank drops, and the allosteric signal is thermodynamically insulated from the binding decision.

An explicit rigid path is still useful, but now only as a certificate that a particular distant atom lies in the graph neighborhood of the active pocket. The theorem surface itself is stronger and more reusable: any future rigidity or geometry engine only needs to certify inclusion in $\mathcal{N}_G(P_{\mathrm{act}})$, or upper-bound its cardinality, to recover the same structural-rank conclusion.

::: theorem
[]{#thm:geometric-contact-allostery label="thm:geometric-contact-allostery"} Fix a contact radius $c$ and a site radius $r$. Let $G_c$ be the protein contact graph whose edges join protein atoms whose reference positions are at distance at most $c$, and let $$P_r=
\{x : \mathrm{dist}(x,\mathrm{site\ center}) < r\}$$ be the radius-$r$ active pocket around the binding-site center. If every currently relevant protein atom lies in the mechanical neighborhood $\mathcal{N}_{G_c}(P_r)$, then $$\mathrm{srank}(D_{\mathrm{dock}})
\le
3\bigl|\mathcal{N}_{G_c}(P_r)\bigr| + 3L.$$
:::

::: proof
*Proof.* This is the generic graph-local theorem instantiated with two Lean-defined geometric objects: the contact graph built directly from reference protein coordinates and the active pocket built directly from the binding-site geometry. The formalization therefore now derives the graph itself from existing geometry, while honestly leaving the graph-neighborhood coverage condition as the remaining external allostery hypothesis. ◻
:::

::: theorem
[]{#thm:contact-shell-allostery label="thm:contact-shell-allostery"} Fix a contact radius $c$ and a site radius $r$. Let $P_r$ be the radius-$r$ active pocket around the binding-site center, and let $$\mathcal{S}_{G_c}(P_r)=
\{x : \exists y \in P_r,\ (x,y) \in E(G_c)\}$$ be its one-hop contact neighborhood in the geometry-derived contact graph $G_c$. If every currently relevant protein atom lies in $$P_r \cup \mathcal{S}_{G_c}(P_r),$$ then $$\mathrm{srank}(D_{\mathrm{dock}})
\le
3|P_r| + 3\bigl|\mathcal{S}_{G_c}(P_r)\bigr| + 3L.$$
:::

::: proof
*Proof.* This is the one-hop specialization of the graph-local allostery layer. Instead of requiring an arbitrary reachability cover, it is enough to cover every relevant protein atom by the radius-defined active pocket together with its direct contact neighborhood in the geometry-derived graph. The formal proof then applies the generic atom-cover theorem to the union of those two finite sets and bounds its cardinality by the sum of the pocket and contact-neighborhood cardinalities. ◻
:::

::: corollary
[]{#cor:bounded-contact-shell-regime label="cor:bounded-contact-shell-regime"} Under the same one-hop contact-cover hypothesis, if $$|P_r| \le P
\qquad\text{and}\qquad
\bigl|\mathcal{S}_{G_c}(P_r)\bigr| \le K,$$ then $$\mathrm{srank}(D_{\mathrm{dock}})
\le
3P + 3K + 3L.$$
:::

::: proof
*Proof.* Substitute the pocket-size and direct-contact-shell bounds into Theorem [\[thm:contact-shell-allostery\]](#thm:contact-shell-allostery){reference-type="ref" reference="thm:contact-shell-allostery"}. ◻
:::

::: theorem
[]{#thm:allosteric-distance-decay-law label="thm:allosteric-distance-decay-law"} Let $G$ be a protein mechanical graph, $P_{\mathrm{act}}$ an active pocket, and $k\in\mathbb{N}$. Suppose relevant atoms are covered by the $k$-hop neighborhood expansion and each outer shell obeys a distance-indexed envelope $$\bigl|\mathrm{Shell}_d\bigr|
\le b(d)
\qquad (d<k),$$ where $b$ is non-increasing: $$b(d+1) \le b(d).$$ Then the per-distance allosteric contribution upper profile $$C(d):=3b(d)$$ is non-increasing: $$C(d+1)\le C(d),$$ and the docking structural rank satisfies $$\mathrm{srank}(D_{\mathrm{dock}})
\le
3|P_{\mathrm{act}}| + 3\sum_{d<k} b(d) + 3L.$$
:::

::: proof
*Proof.* The monotonicity statement is the direct monotonicity of the declared shell envelope under multiplication by the coordinate factor $3$. The rank bound is the shell-budget allostery theorem with the distance envelope $b$ as shell budget. ◻
:::

::: theorem
[]{#thm:allosteric-exponential-distance-decay label="thm:allosteric-exponential-distance-decay"} Let $G$ be a protein mechanical graph, $P_{\mathrm{act}}$ an active pocket, and $k\in\mathbb{N}$. Assume relevant atoms are covered by the $k$-hop neighborhood expansion and there are constants $\Lambda\ge 0$ and $\alpha\ge 0$ such that $$\bigl|\mathrm{Shell}_d\bigr|
\le
\left\lceil \Lambda e^{-\alpha d} \right\rceil
\qquad (d<k).$$ Then the declared integer per-distance contribution profile $$C_{\exp}(d):=3\left\lceil \Lambda e^{-\alpha d} \right\rceil$$ is non-increasing in $d$, and $$\mathrm{srank}(D_{\mathrm{dock}})
\le
3|P_{\mathrm{act}}| + 3\sum_{d<k}\left\lceil \Lambda e^{-\alpha d} \right\rceil + 3L.$$
:::

::: proof
*Proof.* For $\alpha\ge 0$, the real envelope $\Lambda e^{-\alpha d}$ is antitone in $d$; taking ceilings preserves this monotonicity, and multiplying by $3$ yields the monotone contribution profile. The rank inequality is exactly the exponential-profile specialization of the distance-profile shell-sum theorem (and its geometry-contact instance). ◻
:::

::: theorem
[]{#thm:allosteric-polynomial-distance-decay label="thm:allosteric-polynomial-distance-decay"} Let $G$ be a protein mechanical graph, $P_{\mathrm{act}}$ an active pocket, and $k\in\mathbb{N}$. Assume relevant atoms are covered by the $k$-hop neighborhood expansion, and let $\Lambda\ge 0$ and $p\in\mathbb{N}$. If $$\bigl|\mathrm{Shell}_d\bigr|
\le
\left\lceil \frac{\Lambda}{(d+1)^p} \right\rceil
\qquad (d<k),$$ then the declared integer per-distance contribution profile $$C_{\mathrm{poly}}(d):=3\left\lceil \frac{\Lambda}{(d+1)^p} \right\rceil$$ is non-increasing in $d$, and $$\mathrm{srank}(D_{\mathrm{dock}})
\le
3|P_{\mathrm{act}}| + 3\sum_{d<k}\left\lceil \frac{\Lambda}{(d+1)^p} \right\rceil + 3L.$$
:::

::: proof
*Proof.* For $\Lambda\ge 0$, the real envelope $\Lambda/(d+1)^p$ is antitone in $d$ for natural exponent $p$; ceilings preserve this monotonicity and multiplication by $3$ gives the monotone contribution profile. The structural-rank inequality is the polynomial-profile specialization of the distance-profile shell-sum theorem (and its geometry-contact instance). ◻
:::

::: theorem
[]{#thm:allosteric-polynomial-series-budget label="thm:allosteric-polynomial-series-budget"} Under the hypotheses of Theorem [\[thm:allosteric-polynomial-distance-decay\]](#thm:allosteric-polynomial-distance-decay){reference-type="ref" reference="thm:allosteric-polynomial-distance-decay"}, assume additionally that $$\sum_{d<k}\frac{1}{(d+1)^p} \le Z$$ for some real budget constant $Z$. Then the theorem-level real-form rank bound satisfies $$\mathrm{srank}(D_{\mathrm{dock}})
\le
3|P_{\mathrm{act}}| + 3k + 3\Lambda Z + 3L.$$ For $p>1$, any finite reciprocal-power bound on the partial sum can be used as $Z$.
:::

::: proof
*Proof.* The polynomial profile theorem gives the shell-sum term $3\sum_{d<k}\lceil\Lambda/(d+1)^p\rceil$. Each ceiling contributes at most one additive unit above its real argument, so this term is bounded by $$3k + 3\Lambda\sum_{d<k}\frac{1}{(d+1)^p}.$$ Applying the assumed partial-series budget yields the displayed closed-form bound. ◻
:::

::: theorem
[]{#thm:allosteric-polynomial-series-explicit label="thm:allosteric-polynomial-series-explicit"} Under the hypotheses of Theorem [\[thm:allosteric-polynomial-distance-decay\]](#thm:allosteric-polynomial-distance-decay){reference-type="ref" reference="thm:allosteric-polynomial-distance-decay"}, assume additionally that $$p\ge 2.$$ Then the reciprocal-power budget can be instantiated without an external parameter: $$\sum_{d<k}\frac{1}{(d+1)^p}
\le
1 + \frac{k}{4},$$ hence $$\mathrm{srank}(D_{\mathrm{dock}})
\le
3|P_{\mathrm{act}}| + 3k + 3\Lambda\left(1+\frac{k}{4}\right) + 3L.$$
:::

::: proof
*Proof.* For $p\ge 2$, each reciprocal-power term beyond the first is at most $1/4$, giving the explicit finite partial-sum budget $1+k/4$. Substituting that budget into the polynomial series-budget theorem yields the displayed no-parameter closed-form rank bound. ◻
:::

::: theorem
[]{#thm:mechanochemical-coupling-gap label="thm:mechanochemical-coupling-gap"} Let $D$ be a decision problem, let $\sigma_{\mathrm{intact}}$ and $\sigma_{\mathrm{broken}}$ be optimizer summaries with $\sigma_{\mathrm{broken}}$ factoring through $\sigma_{\mathrm{intact}}$, and let $i_{\mathrm{allo}}$ be a distant allosteric coordinate. If $i_{\mathrm{allo}}$ is relevant for the intact summary problem but irrelevant for the broken summary problem, then $$\mathrm{srank}(D_{\sigma_{\mathrm{broken}}})
<
\mathrm{srank}(D_{\sigma_{\mathrm{intact}}}),$$ and the theorem-level allosteric coupling cost assigned to $i_{\mathrm{allo}}$ vanishes.
:::

::: proof
*Proof.* Once the broken summary factors through the intact summary, any coordinate that disappears across that factorization contributes positively to the exact collapse count. The broken summary therefore has strictly smaller structural rank. The mechanochemical coupling term is defined to charge one declared per-bit unit exactly when the coordinate remains decision-relevant, so irrelevance in the broken regime forces that term to zero. ◻
:::

::: theorem
[]{#thm:hierarchical-srank-bound label="thm:hierarchical-srank-bound"} Let $\sigma_{\mathrm{micro}}$ and $\sigma_{\mathrm{macro}}$ be micro and macro optimizer summaries with $\sigma_{\mathrm{macro}}$ factoring through $\sigma_{\mathrm{micro}}$, and let $I_{\mathrm{sub}}$ be a bundled micro block. If the bundling step erases at least $$|I_{\mathrm{sub}}|-1$$ exact relevant coordinates, then $$\mathrm{srank}(D_{\sigma_{\mathrm{macro}}})
\le
\mathrm{srank}(D_{\sigma_{\mathrm{micro}}}) - (|I_{\mathrm{sub}}|-1).$$
:::

::: proof
*Proof.* The factorization hypothesis identifies bundling with a controlled collapse from the micro summary to the macro summary. The exact rank-collapse law then says that the macro rank equals the micro rank minus the number of erased relevant coordinates. If that collapse count dominates the internal degrees of freedom of the bundled block, the displayed inequality follows immediately. ◻
:::

::: theorem
[]{#thm:renormalized-admissibility-equivalence label="thm:renormalized-admissibility-equivalence"} Let $F$ be a nested admissibility family. If relaxing the admissibility band from $\varepsilon$ to $\varepsilon + \Delta$ erases no exact relevant coordinate, then the two induced summary problems have the same relevant-coordinate quotient and the same structural rank. Equivalently, for every coordinate $i$, $$i \text{ is relevant for } D_{F,\varepsilon}
\iff
i \text{ is relevant for } D_{F,\varepsilon+\Delta},$$ and $$\mathrm{srank}(D_{F,\varepsilon}) = \mathrm{srank}(D_{F,\varepsilon+\Delta}).$$
:::

::: proof
*Proof.* Zero collapse count means the tighter and relaxed relevant-coordinate sets have empty set difference. Hence no exact relevant coordinate disappears under the admissibility relaxation, so relevance agrees coordinatewise. Structural rank is the cardinality of that relevant set, so the two summary problems have equal rank. ◻
:::

These hierarchical theorems now sit on an explicit bundled macro-state layer in Lean: the artifact contains literal forward and reverse maps between micro coordinate states and their bundled macro representations, so the renormalization argument is stated over an actual coarse state object rather than only an informal coarse variable.

#### Lawful Approximation: Admissibility and Collapse (Sections 4.5--4.6).

The next two blocks isolate the only approximation class that preserves theorem-level exactness claims: quotient-coarsened admissibility. Approximation is lawful when it collapses decision distinctions in a controlled way, and the collapse is measured directly in structural-rank units.

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
[]{#thm:admissibility-rank-reduction label="thm:admissibility-rank-reduction"} Let $D$ be a finite decision problem, and let $\sigma$ be any admissibility summary that depends only on the exact optimal-action set. Write $D_\sigma$ for the induced singleton-valued summary problem whose optimizer at state $s$ is $$\{\sigma(\operatorname{Opt}_D(s))\}.$$ Then $$\mathrm{srank}(D_\sigma) \le \mathrm{srank}(D).$$ Moreover, $$\mathrm{srank}(D_\sigma) < \mathrm{srank}(D)
\iff
\text{some coordinate is relevant for $D$ but irrelevant for $D_\sigma$.}$$
:::

::: proof
*Proof.* A coordinate relevant for the summary problem is already relevant for the exact problem, so the summary relevant set is contained in the exact relevant set. Taking cardinalities gives the weak inequality. The inequality is strict exactly when that containment is proper, which is equivalent to the existence of an exact relevant coordinate that the admissibility summary erases. ◻
:::

Combined with the generic Landauer theorem, the admissibility problem $D_\sigma$ therefore carries the floor $$E \ge \mathrm{srank}(D_\sigma)\,k_B T\ln 2.$$ Relaxing exactness lowers that floor only by destroying exact relevant coordinates. The factor-through-the-exact-quotient hypothesis is essential: a raw utility-profile band such as "all actions within $\varepsilon$ of the optimum" can refine the exact quotient rather than coarsen it, so the theorem isolates the precise structural notion of admissible exact approximation used here.

::: theorem
[]{#thm:quantitative-admissibility-rank-collapse label="thm:quantitative-admissibility-rank-collapse"} Let $F$ be a tolerance-indexed admissibility family such that for every $\Delta \ge 0$, the relaxed summary at tolerance $\varepsilon+\Delta$ factors through the tighter summary at tolerance $\varepsilon$. Write $D_{F,\varepsilon}$ for the induced summary problem, and let $$\mathrm{CollapseCount}_F(D,\varepsilon,\Delta)$$ denote the number of exact relevant coordinates erased by the relaxation from $\varepsilon$ to $\varepsilon+\Delta$. Then $$\mathrm{srank}(D_{F,\varepsilon})
=
\mathrm{srank}(D_{F,\varepsilon+\Delta})
+
\mathrm{CollapseCount}_F(D,\varepsilon,\Delta).$$ In particular, if exactly one coordinate is erased, then $$\mathrm{srank}(D_{F,\varepsilon+\Delta})
=
\mathrm{srank}(D_{F,\varepsilon})-1,$$ and for every declared thermodynamic model $M$, $$\mathrm{energyLowerBound}\bigl(M,\mathrm{srank}(D_{F,\varepsilon})\bigr)
=
\mathrm{energyLowerBound}\bigl(M,\mathrm{srank}(D_{F,\varepsilon+\Delta})\bigr)
+ M.\mathrm{joulesPerBit}.$$
:::

::: proof
*Proof.* The tighter relevant-coordinate set contains the relaxed relevant-coordinate set, and their set difference is exactly the family's collapse count. Structural rank is the cardinality of the relevant-coordinate set, so the first identity is a finite-cardinality decomposition. The one-coordinate case is the specialization where the set difference has cardinality one, and the energy identity is the same statement pushed through the declared linear energy model. ◻
:::

::: theorem
[]{#thm:admissibility-zero-collapse-bifactor label="thm:admissibility-zero-collapse-bifactor"} Let $F$ be a nested admissibility family for decision problem $D$. Fix $(\varepsilon,\Delta)$ with $\Delta\ge 0$, and assume the usual forward relaxation factorization $$F_{\varepsilon+\Delta}=\tau\circ F_{\varepsilon}$$ together with a backward factorization witness $$F_{\varepsilon}=\kappa\circ F_{\varepsilon+\Delta}$$ for some maps $\tau,\kappa$ on summary codomains. Then $$\mathrm{CollapseCount}_F(D,\varepsilon,\Delta)=0,
\qquad
\mathrm{srank}(D_{F,\varepsilon+\Delta})=\mathrm{srank}(D_{F,\varepsilon}).$$
:::

::: proof
*Proof.* Forward factorization gives inclusion of relaxed relevant coordinates into tight relevant coordinates; backward factorization gives the reverse inclusion. Hence the relevant-coordinate sets are equal, so their set difference (the collapse count) is zero. Substituting this into the quantitative rank-collapse law yields rank invariance. ◻
:::

::: theorem
[]{#thm:admissibility-collapse-canonicality label="thm:admissibility-collapse-canonicality"} Let $F$ and $G$ be nested admissibility families (possibly with different summary codomains) for the same exact decision problem $D$. Assume $$\mathrm{srank}(D_{F,\varepsilon_F})
=
\mathrm{srank}(D_{G,\varepsilon_G})$$ and $$\mathrm{CollapseCount}_F(D,\varepsilon_F,\Delta_F)
=
\mathrm{CollapseCount}_G(D,\varepsilon_G,\Delta_G),
\qquad
\Delta_F,\Delta_G\ge 0.$$ Then $$\mathrm{srank}(D_{F,\varepsilon_F+\Delta_F})
=
\mathrm{srank}(D_{G,\varepsilon_G+\Delta_G}).$$ Equivalently, after passing to the rank-normal-form collapse category (objects are collapsed ranks, morphisms are monotone rank maps), the two relaxed collapse objects are isomorphic.
:::

::: proof
*Proof.* Apply the quantitative collapse law to each family separately. Each relaxed rank equals the corresponding tight rank minus the corresponding collapse count. The hypotheses identify those two right-hand sides, so the relaxed ranks agree. In the rank-normal-form collapse category, isomorphism is exactly equality of collapsed ranks. ◻
:::

::: theorem
[]{#thm:collapsed-rank-category-structure label="thm:collapsed-rank-category-structure"} Let $\mathcal{C}_{\mathrm{coll}}$ be the rank-normal-form collapse category whose objects are collapsed ranks and whose morphisms are monotone rank maps.

Then:

1.  Identity and composition hold (preorder-category laws).

2.  An initial object exists (rank $0$), while no terminal object exists in the unbounded category.

3.  Binary products and coproducts exist, realized by rank minimum and rank maximum, respectively.

4.  Additive composition defines a monoidal tensor on objects (rank addition) with unit rank $0$, and this tensor is monotone on morphisms.

5.  Structural rank is a monotone functor from $\mathcal{C}_{\mathrm{coll}}$ to $(\mathbb{N},\le)$.
:::

::: proof
*Proof.* All clauses are direct from the mechanized order-theoretic construction: morphisms are inequalities on ranks, so identity/composition are reflexivity/transitivity; $0$ is initial; unboundedness rules out a terminal top rank; finite meet/join are min/max; tensor is additive with unit $0$; and rank projection is monotone by definition. ◻
:::

::: theorem
[]{#thm:bounded-collapsed-rank-finite-colimits label="thm:bounded-collapsed-rank-finite-colimits"} Fix a rank cap $K\in\mathbb{N}$ and restrict to collapsed-rank objects with rank at most $K$. In this bounded slice:

1.  a terminal object exists (rank $K$),

2.  finite limits exist (terminal object plus binary products),

3.  finite colimits exist (initial object plus binary coproducts), and

4.  this contrasts with the unbounded collapse category, which has no terminal object.
:::

::: proof
*Proof.* The bounded cap contributes a top element $K$, so every bounded object maps to rank $K$. Binary products/coproducts remain rank min/max with universal properties inherited from the preorder structure, and together with terminal/initial objects they give finite limits/colimits. The unbounded no-terminal claim is the previously proved obstruction theorem. ◻
:::

::: theorem
[]{#thm:bounded-collapsed-rank-finite-family-lattice label="thm:bounded-collapsed-rank-finite-family-lattice"} Fix a rank cap $K\in\mathbb{N}$ and a finite family $\mathcal{S}$ of collapsed-rank objects with rank at most $K$. Then there exist objects $$\bigwedge \mathcal{S},\qquad \bigvee \mathcal{S}$$ in the bounded slice such that:

1.  $\bigwedge \mathcal{S}$ maps to every element of $\mathcal{S}$ and is greatest among all such lower bounds,

2.  every element of $\mathcal{S}$ maps to $\bigvee \mathcal{S}$ and it is least among all such upper bounds.

For empty families, these specialize to terminal and initial objects respectively.
:::

::: proof
*Proof.* In the bounded preorder, nonempty finite-family meet and join are realized by finite infimum/supremum of ranks, with universality inherited from order-theoretic inf/sup laws. For the empty family, the universal lower/upper bounds are exactly the terminal/initial objects already available in the bounded slice. ◻
:::

::: theorem
[]{#thm:bounded-collapsed-rank-complete-lattice label="thm:bounded-collapsed-rank-complete-lattice"} Fix a rank cap $K\in\mathbb{N}$ and consider the bounded collapsed-rank slice. For every (possibly infinite) family $\mathcal{S}$ of bounded collapsed-rank objects, there exist objects $$\bigwedge \mathcal{S},\qquad \bigvee \mathcal{S}$$ such that:

1.  $\bigwedge \mathcal{S}$ maps to every member of $\mathcal{S}$ and is greatest among all lower bounds,

2.  every member of $\mathcal{S}$ maps to $\bigvee \mathcal{S}$ and it is least among all upper bounds.

Equivalently, the bounded collapsed-rank slice is complete-lattice-like at theorem level.
:::

::: proof
*Proof.* Bounded objects are encoded into the finite index order $\mathrm{Fin}(K+1)$; arbitrary-family meet and join are pulled back from set-theoretic infimum/supremum in that complete finite order. The universal lower/upper-bound properties are then transported back to bounded collapsed-rank objects by the encoding/decoding equivalence. ◻
:::

::: theorem
[]{#thm:bounded-collapsed-rank-algebraic-laws label="thm:bounded-collapsed-rank-algebraic-laws"} In the bounded collapsed-rank slice:

1.  (Monotonicity) If $\mathcal{S}\subseteq\mathcal{T}$, then $$\bigwedge\mathcal{T}\to\bigwedge\mathcal{S},
    \qquad
    \bigvee\mathcal{S}\to\bigvee\mathcal{T}.$$

2.  (Singleton idempotence) $$\operatorname{rank}(\bigwedge\{X\})=\operatorname{rank}(X)=\operatorname{rank}(\bigvee\{X\}).$$

3.  (Nonempty absorption) If $\mathcal{S}\neq\varnothing$, then $$\operatorname{rank}\!\left(\bigwedge\bigl(\mathcal{S}\cup\{\bigvee\mathcal{S}\}\bigr)\right)
    =
    \operatorname{rank}(\bigwedge\mathcal{S}),$$ $$\operatorname{rank}\!\left(\bigvee\bigl(\mathcal{S}\cup\{\bigwedge\mathcal{S}\}\bigr)\right)
    =
    \operatorname{rank}(\bigvee\mathcal{S}).$$
:::

::: proof
*Proof.* Monotonicity follows from the universal definitions of arbitrary meet/join as greatest lower and least upper bounds. Singleton idempotence is the same universal property specialized to one element. For nonempty absorption, first use $\bigwedge\mathcal{S}\to\bigvee\mathcal{S}$, then show that adjoining this already-implied bound leaves the corresponding greatest-lower/least-upper object unchanged. ◻
:::

::: theorem
[]{#thm:bounded-collapsed-rank-binary-calculus label="thm:bounded-collapsed-rank-binary-calculus"} For bounded collapsed-rank objects $X,Y,Z$ (at fixed cap $K$), the binary meet/join operations satisfy:

1.  commutativity and associativity, $$X\wedge Y = Y\wedge X,
    \qquad
    (X\wedge Y)\wedge Z = X\wedge (Y\wedge Z),$$ $$X\vee Y = Y\vee X,
    \qquad
    (X\vee Y)\vee Z = X\vee (Y\vee Z),$$

2.  idempotence, $$X\wedge X = X,
    \qquad
    X\vee X = X,$$

3.  absorption, $$X\wedge (X\vee Y) = X,
    \qquad
    X\vee (X\wedge Y) = X.$$
:::

::: proof
*Proof.* In the bounded slice, binary meet/join are rank-level min/max. The displayed equalities are exactly the min/max commutativity, associativity, idempotence, and absorption identities transported through the object constructor. ◻
:::

**Physical Significance (Collapse Law).** Theorem [\[thm:quantitative-admissibility-rank-collapse\]](#thm:quantitative-admissibility-rank-collapse){reference-type="ref" reference="thm:quantitative-admissibility-rank-collapse"} gives the exact exchange rate between physical slack and thermodynamic cost. When a molecular system relaxes admissibility (for example, accepting any binding mode within $2\,\mathrm{kcal/mol}$ of the optimum), search burden decreases through lawful structural-rank collapse. Each exact decision distinction rendered irrelevant removes one calibrated Landauer unit from the floor, i.e. one factor of $k_B T\ln 2$ in the linear per-coordinate model.

::: theorem
[]{#thm:lj-tolerance-collapse-explicit label="thm:lj-tolerance-collapse-explicit"} Fix a nested admissibility family at base tolerance $\varepsilon_0$ and suppose each coordinate has an explicit collapse threshold so that coordinate erasure under relaxation by $\Delta\varepsilon$ is equivalent to crossing that coordinate's threshold. Then $$\mathrm{CollapseCount}_F(D,\varepsilon_0,\Delta\varepsilon)
=
f_{\mathrm{threshold}}(D,F,\Delta\varepsilon),$$ where $f_{\mathrm{threshold}}$ is the explicit finite threshold-count formula from the theorem interface. In particular, for the packaged Lennard-Jones pocket scorer instance, the same explicit formula computes collapse count as a function of physical tolerance increment.
:::

::: proof
*Proof.* The profile theorem identifies collapse count with the declared threshold-count formula. The Lennard-Jones pocket statement is the direct specialization of that identity to the packaged LJ instance. ◻
:::

::: theorem
[]{#thm:sampled-inside-cutoff-sufficient label="thm:sampled-inside-cutoff-sufficient"} Under the cutoff boundedness, sampled-optimum capture, coordinate-compatibility, and injectivity hypotheses, the retained coordinate set consisting of inside-cutoff protein coordinates together with all ligand coordinates is sufficient for the sampled restricted docking problem.
:::

::: proof
*Proof.* Cutoff locality forces every relevant sampled coordinate into the retained set. The compatibility and injectivity hypotheses then lift the retained-set relevance bound into a sufficiency theorem for the sampled restricted problem. ◻
:::

#### The Discretization Bridge (Sections 4.7--4.8).

The final two blocks connect continuous molecular physics to finite theorem objects. Controlled transport requires winner-class and survivor-set stability under explicit approximation radii and gap conditions.

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

::: theorem
[]{#thm:composed-classical-forcefield-interface label="thm:composed-classical-forcefield-interface"} Let $H$ be a composed classical Hamiltonian with bonded, Lennard-Jones, and Coulomb terms. Then:

1.  $H$ induces a canonical architecture object carrying exactly the declared $(\mathrm{DOF},\mathrm{capabilities})$ metadata.

2.  If each component term is Lipschitz on a declared positive shell with constants $L_{\mathrm{bond}},L_{\mathrm{LJ}},L_{\mathrm{Coul}}$, then the total composed energy is Lipschitz on that shell with constant $$L_{\mathrm{bond}}+L_{\mathrm{LJ}}+L_{\mathrm{Coul}}.$$
:::

::: proof
*Proof.* The architecture clause is definitional from the composed-force-field record projection. The shell Lipschitz clause is the triangle-inequality composition of the three component bounds. ◻
:::

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
[]{#thm:action-pinned-lifting label="thm:action-pinned-lifting"} Let $u_{\mathrm{cont}}$ be a continuous score family, let $u_{\mathrm{grid}}$ be its discretized score family, and let $\mathrm{lift}$ map grid states into continuous states. Fix a grid state $s$, let $$D_{\mathrm{grid,pin}}$$ be the action-pinned grid problem, and let $$D^{\mathrm{lift}}_{\mathrm{cont,pin}[\mathrm{grid}]}$$ denote the lifted continuous problem on grid states obtained by adding the same statewise tie-break perturbation used in $D_{\mathrm{grid,pin}}$. If the exact-versus-grid score discrepancy is bounded by $\varepsilon(\mathrm{res})$ on every lifted grid state and if $$\varepsilon(\mathrm{res}) < \frac{1}{2}\,\mathrm{StrictUtilityGap}_{D_{\mathrm{grid,pin}}}(a^\dagger(s),s),$$ then $$\operatorname{Opt}_{D_{\mathrm{grid,pin}}}(s)
=
\operatorname{Opt}_{D^{\mathrm{lift}}_{\mathrm{cont,pin}[\mathrm{grid}]}}(s).$$ Moreover, every action $a$ in this common optimal set satisfies $$u_{\mathrm{cont}}(a,\mathrm{lift}(s))
\ge
\max_b u_{\mathrm{cont}}(b,\mathrm{lift}(s)) - 2\varepsilon(\mathrm{res}).$$
:::

::: proof
*Proof.* Using the same statewise tie-break perturbation on both the lifted continuous and grid score families leaves only the physical continuous-to-grid score discrepancy $\varepsilon(\mathrm{res})$. The pinned grid half-gap condition therefore preserves the selected pinned winner between those two problems. Any grid-optimal action is within $2\varepsilon(\mathrm{res})$ of the lifted continuous optimum, so the common selected action is canonically admissible for the continuous problem even when the continuous exact optimizer is tied. ◻
:::

This is the legitimate point at which the finite pinned gap $\gamma_{\mathrm{grid}}$ enters the continuous story. It is not inserted into the unpinned continuous strict-winner theorem. Instead, the grid tie-break rule is transported unchanged to the lifted continuous problem, so the half-gap comparison is made between two pinned problems with the same artificial perturbation. The conclusion is therefore an admissibility statement for the continuous physics, not a false claim that the continuous problem itself has a strict exact winner.

::: theorem
[]{#thm:large-cutoff-bounded label="thm:large-cutoff-bounded"} Let a binding problem admit a lattice-tail perturbation bound with coefficient $c$ and a positive strict minimum decision gap $\gamma$. If the chosen cutoff radius satisfies $$c\,\mathrm{TailSum}_6(R_{\mathrm{cut}}) < \gamma/2,$$ then the exact docking problem satisfies the cutoff-boundedness hypothesis required by the structural-rank and sampled-docking locality theorems.
:::

::: proof
*Proof.* The lattice-tail bound supplies a uniform upper bound on every outside-cutoff utility perturbation. The strict minimum gap condition then shows that the perturbation remains below half the exact decision gap, which is exactly the cutoff-boundedness criterion. ◻
:::

These theorems provide a direct route from continuous regularity and physical tail control to the finite exact/coarse approximation theorems used later in the docking development.

At the force-field object level, the mechanization now includes an explicit composed classical Hamiltonian that packages bonded, Lennard-Jones, and Coulomb terms into one theorem-level object with both decision-problem and architecture projections (L589). The same block also proves a positive-shell Lipschitz composition law for the full Hamiltonian from componentwise shell bounds (L590), which is exactly the bound shape consumed by half-gap refinement arguments.

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

::: theorem
[]{#thm:coordinate-extraction-poly label="thm:coordinate-extraction-poly"} Let $D$ be a finite product-state decision problem on $n$ coordinates with injective full coordinate projection, and assume the action family has size at most $k$. Define the extracted coordinate set by the explicit rule $$I_{\mathrm{alg}}
=
\bigl\{i : \mathrm{CHECKSUFF}(\{1,\dots,n\}\setminus\{i\}) = \mathrm{false}\bigr\}.$$ Then $I_{\mathrm{alg}}$ is sufficient for $D$ and $$|I_{\mathrm{alg}}| = \mathrm{srank}(D).$$ Moreover, the total extraction cost is bounded by $$n\,|S|^2(1+k^2).$$
:::

::: proof
*Proof.* The single-coordinate deletion test removes exactly those coordinates whose erasure destroys sufficiency. In the product-space setting that criterion is equivalent to relevance, so the extracted set is exactly the relevant-coordinate set. That set is sufficient, and its cardinality is by definition the structural rank. Because the rule performs one bounded-action sufficiency check per coordinate, the total cost is at most $n$ times the per-check bound. ◻
:::

These theorems convert exact-resolution stability into explicit finite certificates and polynomial-time checking regimes.

::: theorem
[]{#thm:inverse-rank-gap-design label="thm:inverse-rank-gap-design"} Fix $K \in \mathbb{N}$ and $\gamma \in \mathbb{R}$, and write $$\Gamma(K,\gamma)
=
\bigl\{D : \mathrm{srank}(D) \le K \text{ and every state has a strict winner with gap at least } \gamma\bigr\}.$$ If a finite decision problem admits a sufficient coordinate set $I$ with $|I|\le K$ and a uniform strict gap lower bound $\gamma$, then $D\in\Gamma(K,\gamma)$. In the bounded-action product-state regime, if $\mathrm{srank}(D)\le K$ and $$\mathrm{energyLowerBound}(M,K) \le E,$$ then the constructive extraction algorithm returns a sufficient free-coordinate set $I$ such that $$|I| \le K,
\qquad
|\{1,\dots,n\}\setminus I| = n-|I|,
\qquad
\mathrm{energyLowerBound}(M,\mathrm{srank}(D)) \le E.$$ Hence the complement of $I$ is a theorem-backed coordinate-constraint certificate meeting the requested rank and energy budget.
:::

::: proof
*Proof.* Any sufficient coordinate witness of size at most $K$ bounds structural rank by $K$, and the uniform strict-gap hypothesis supplies the gap half of the class definition. In the bounded-action regime, the extraction theorem returns the relevant-coordinate set itself, so its cardinality is exactly structural rank. If that rank is at most $K$, then the extracted free set also has size at most $K$, its complement has cardinality $n-|I|$, and monotonicity of the linear energy model transfers the target budget bound from $K$ to the actual rank. ◻
:::

::: theorem
[]{#thm:inverse-design-atomistic-bridge label="thm:inverse-design-atomistic-bridge"} Assume an atomistic synthesis bridge that lifts pointwise realizability of each geometric holonomic constraint in a finite synthesis specification to realizability of the full specification by one atomistic geometry witness. Then every inverse rank-gap synthesis target $(r,k,\gamma)$ has a realized atomistic geometry witness for the canonical specification, with certificate rank equality $$\mathrm{srank}(\mathrm{quotientSynthesis}(r,k,\gamma))
=
\bigl|\mathrm{physicalSynthesisSpec}(r,\gamma)\bigr|.$$
:::

::: proof
*Proof.* Use the bridge interface hypothesis to lift pointwise realizability over the canonical synthesis specification to one global atomistic witness. The rank identity is the existing synthesis-certificate equality specialized to the same specification. ◻
:::

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

::: theorem
[]{#thm:top-level-computable-export label="thm:top-level-computable-export"} There is a constructive top-level cross-docking execution path from molecular input state to ArrayDSL JSON export artifact such that:

1.  retained coordinates are produced by a computable greedy extraction loop over canonical finite coordinate order,

2.  refinement output is produced by a computable fuel-bounded loop over rational certificates,

3.  exported JSON is exactly the canonical ArrayDSL primitive export string.
:::

::: proof
*Proof.* Each clause is a definitional projection of the constructive top-level output bundle, and the endpoint theorem is the conjunction of those definitional equalities. ◻
:::

The docking bridge theorems are assembled in `Leverage/DockingTheoryBridge.lean`. They expose the abstraction-collapse boundary, the Fisher-rank identities, the general exact-sufficiency hardness core, the quantitative witness/checking lower bounds, the cutoff-local structural-rank bounds for molecular docking, the conformer-search collapse regimes, the sampled exact/coarse preservation and sufficiency theorems, the top-$k$ and ambiguity-band control theorems, the concrete Lennard-Jones and Coulomb cutoff invariance statements, and the geometric and electrostatic structure theorems used in the molecular development. The same file now also contains a constructive top-level execution path from molecular input to ArrayDSL JSON export with a theorem-level computability endpoint (L596).


# Thermodynamic Cost of Exact Molecular Docking {#main-theorems}

The preceding sections fixed the exact object, its unavoidable quotient boundary, its structural dimension and canonical Fisher-identifiable dimension, and its certification burden. The thermodynamic theorems of this section convert that same exact-resolution spine into cost. Proposition [\[prop:binding-as-exact-resolution\]](#prop:binding-as-exact-resolution){reference-type="ref" reference="prop:binding-as-exact-resolution"} supplies the binding-to-resolution bridge used by the docking instantiations. The abstract statements hold for bounded decision systems, and the constrained-molecular corollaries transport them to holonomic topologies and binding-resolution problems. Landauer furnishes the universal floor for the conversion constant.

::: remark
[]{#rem:thermo-antecedents label="rem:thermo-antecedents"} The thermodynamic results in this section use four antecedents:

1.  Landauer calibration at positive $k_B$ and $T$.

2.  Finite binary acquisition events (Theorems [\[thm:discrete-acquisition\]](#thm:discrete-acquisition){reference-type="ref" reference="thm:discrete-acquisition"} and [\[thm:one-transition-one-bit\]](#thm:one-transition-one-bit){reference-type="ref" reference="thm:one-transition-one-bit"}).

3.  Exclusion of physically feasible erasure of decision-relevant distinctions beyond quotient factorization (Theorem [\[thm:feasible-collapse-factors\]](#thm:feasible-collapse-factors){reference-type="ref" reference="thm:feasible-collapse-factors"}).

4.  Existence of a sufficient coordinate set for the target decision problem (Theorem [\[thm:resolution-sufficient\]](#thm:resolution-sufficient){reference-type="ref" reference="thm:resolution-sufficient"}).

Violations of these antecedents place a system outside the theorem scope.
:::

::: remark
[]{#rem:relaxational-binding-interface label="rem:relaxational-binding-interface"} Boltzmann relaxation and controlled protocols differ in dynamics but share the same distinction-counting layer when a retained bound/unbound outcome is produced. Proposition [\[prop:binding-as-exact-resolution\]](#prop:binding-as-exact-resolution){reference-type="ref" reference="prop:binding-as-exact-resolution"} and Theorems [\[thm:discrete-acquisition\]](#thm:discrete-acquisition){reference-type="ref" reference="thm:discrete-acquisition"}--[\[thm:one-transition-one-bit\]](#thm:one-transition-one-bit){reference-type="ref" reference="thm:one-transition-one-bit"} apply to that retained distinction interface.
:::

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

The thermodynamic bridge for docking is therefore split into two honest layers. Theorem [\[thm:md-energy-rank\]](#thm:md-energy-rank){reference-type="ref" reference="thm:md-energy-rank"} is the direct exact-docking structural-rank cost floor once a sufficient coordinate set is fixed. Theorem [\[thm:md-energy-entropy\]](#thm:md-energy-entropy){reference-type="ref" reference="thm:md-energy-entropy"} is the entropy-calibrated Landauer statement for the explicit finite binary encoding of the sampled flat grid problem. Theorem [\[thm:grid-docking-entropy\]](#thm:grid-docking-entropy){reference-type="ref" reference="thm:grid-docking-entropy"} separately bounds the same finite-grid quotient entropy in the original $(2N+1)$-ary coordinate presentation. The cutoff-local theorems from Section [\[complexity-boundary\]](#complexity-boundary){reference-type="ref" reference="complexity-boundary"} provide geometric upper bounds on $\mathrm{srank}(D_{\mathrm{dock}})$, while Theorem [\[thm:optimizer-class-richness-rank-lower-bound\]](#thm:optimizer-class-richness-rank-lower-bound){reference-type="ref" reference="thm:optimizer-class-richness-rank-lower-bound"} provides a combinatorial lower bound from optimizer-class richness. Together they give a theorem-level two-sided bracket once both geometric and class-richness hypotheses are instantiated.

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
[]{#thm:quotient-resolution-speed-bound label="thm:quotient-resolution-speed-bound"} Let $D$ be a finite-coordinate decision problem resolved inside a bounded region of diameter $d$ and signal speed $c$ over horizon $\tau$. If some sufficient coordinate set for $D$ fits inside the bounded-acquisition budget of that region and horizon, then $$d\,\mathrm{srank}(D) \le c\,\tau.$$
:::

::: proof
*Proof.* The exact-resolution witness gives $\mathrm{srank}(D)$ no larger than the total number of acquisition events available on horizon $\tau$. Rewriting that bounded-acquisition inequality in multiplicative form yields the displayed speed bound. ◻
:::

::: theorem
[]{#thm:admissibility-speed-accuracy-tradeoff label="thm:admissibility-speed-accuracy-tradeoff"} Let $F$ be a nested admissibility family for a decision problem $D$, let $\Delta\ge 0$, and let $D_{F,\varepsilon+\Delta}$ denote the relaxed summary problem. Suppose $D_{F,\varepsilon+\Delta}$ is exactly resolved inside a bounded region $(d,c)$ over horizon $\tau$. Then $$d\Bigl(\mathrm{srank}(D_{F,\varepsilon})-
\mathrm{CollapseCount}_F(D,\varepsilon,\Delta)\Bigr)
\le c\tau.$$ Equivalently, in division form, $$\left\lfloor
\frac{d\bigl(\mathrm{srank}(D_{F,\varepsilon})-
\mathrm{CollapseCount}_F(D,\varepsilon,\Delta)\bigr)}{c}
\right\rfloor
\le \tau.$$
:::

::: proof
*Proof.* Apply the quotient speed theorem to the relaxed problem $D_{F,\varepsilon+\Delta}$. The quantitative admissibility collapse law rewrites its rank exactly as $$\mathrm{srank}(D_{F,\varepsilon+\Delta})
=
\mathrm{srank}(D_{F,\varepsilon})-
\mathrm{CollapseCount}_F(D,\varepsilon,\Delta).$$ Substituting gives the multiplicative tradeoff, and the floor/division form is the equivalent natural-number lower-bound presentation. ◻
:::

::: theorem
[]{#thm:admissibility-speed-accuracy-zero-collapse label="thm:admissibility-speed-accuracy-zero-collapse"} Under the hypotheses of Theorem [\[thm:admissibility-speed-accuracy-tradeoff\]](#thm:admissibility-speed-accuracy-tradeoff){reference-type="ref" reference="thm:admissibility-speed-accuracy-tradeoff"}, assume additionally that the relaxation from $\varepsilon$ to $\varepsilon+\Delta$ is bidirectionally factorizable at summary level (so Theorem [\[thm:admissibility-zero-collapse-bifactor\]](#thm:admissibility-zero-collapse-bifactor){reference-type="ref" reference="thm:admissibility-zero-collapse-bifactor"} applies). Then $$d\,\mathrm{srank}(D_{F,\varepsilon}) \le c\tau.$$ Equivalently, in this no-collapse regime the speed bound is controlled by the tight rank without a subtraction term.
:::

::: proof
*Proof.* Bidirectional factorization implies zero collapse and therefore $$\mathrm{srank}(D_{F,\varepsilon+\Delta})=\mathrm{srank}(D_{F,\varepsilon}).$$ Apply the relaxed-problem speed bound and substitute this equality. ◻
:::

::: theorem
[]{#thm:admissibility-onrate-envelope label="thm:admissibility-onrate-envelope"} Under the hypotheses of Theorem [\[thm:admissibility-speed-accuracy-tradeoff\]](#thm:admissibility-speed-accuracy-tradeoff){reference-type="ref" reference="thm:admissibility-speed-accuracy-tradeoff"}, assume additionally $\tau>0$ and positive relaxed rank. Then the achieved exact-resolution rate obeys $$\frac{1}{\tau}
\le
\frac{c}{d\,\mathrm{srank}(D_{F,\varepsilon+\Delta})}
=
\frac{c}{d\bigl(\mathrm{srank}(D_{F,\varepsilon})-
\mathrm{CollapseCount}_F(D,\varepsilon,\Delta)\bigr)}.$$
:::

::: proof
*Proof.* Cast the finite speed inequality to real form and divide by positive denominators. The second equality is exactly the collapse-law substitution for the relaxed rank. ◻
:::

::: theorem
[]{#thm:trajectory-time-energy-tradeoff label="thm:trajectory-time-energy-tradeoff"} Let $(D_t)_{t<m}$ be a finite sequence of decision problems, and for each stage $t$ let $R_t=(d_t,c_t)$ be a bounded region with operating horizon $\tau_t$. If each stage admits an exact-resolution witness inside its local bounded-acquisition budget, then $$\sum_{t<m} d_t\,\mathrm{srank}(D_t)
\le
\sum_{t<m} c_t\,\tau_t.$$ For any thermodynamic model $M$ with positive per-bit conversion constant, the same hypotheses also imply $$\sum_{t<m} M.\mathrm{joulesPerBit}\cdot \mathrm{srank}(D_t)
\le
\sum_{t<m} \mathrm{energyLowerBound}\!\bigl(M,\mathrm{MaxAcquisitions}(R_t,\tau_t)\bigr).$$
:::

::: proof
*Proof.* Apply Theorem [\[thm:quotient-resolution-speed-bound\]](#thm:quotient-resolution-speed-bound){reference-type="ref" reference="thm:quotient-resolution-speed-bound"} stagewise and sum the resulting inequalities. The energy inequality is obtained the same way from the stagewise bounded-acquisition energy floor. ◻
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

Thresholded one-bit readouts and two-level atomic transitions instantiate the same binary interface[^1] [@berut2012experimental]. A $k$-channel substrate has joint readout state in $\{0,1\}^k$, and the canonical state space $\mathrm{Fin}\;k \to \mathrm{Bool}$ is the same object written in indexed form.

The concrete quantum substrate layer is now also instantiated at theorem level for a spin-$\tfrac12$ readout substrate: a two-state witness is explicit, the induced classical decision object is exactly $\mathrm{canonicalDP}(1)$, and the decoherence readout cost specializes to exactly $k_B T\ln 2$ ().

::: theorem
[]{#thm:spinhalf-concrete-quantum-instantiation label="thm:spinhalf-concrete-quantum-instantiation"} For every declared spin-$\tfrac12$ substrate package:

1.  there are exactly two distinguished readout states covering all classical readout states,

2.  the induced classical decision model is exactly $\mathrm{canonicalDP}(1)$ and has structural rank $1$,

3.  the decoherence/readout event cost is exactly $$k_B T\ln 2.$$
:::

::: proof
*Proof.* The first clause is the explicit two-state witness theorem. The second clause is the canonical embedding theorem specialized to one coordinate. The third clause is the abstract decoherence Landauer identity specialized to that rank-one embedding. ◻
:::

## Substrate Time Law

::: proposition
[]{#prop:substrate-time-law label="prop:substrate-time-law"} For any substrate model whose observed interface obeys decision ticks, every one-step substrate evolution realizes a decision event and advances interface time by one unit. The tick law is independent of substrate tag.
:::

::: theorem
[]{#thm:pathwise-energy-lower-bound label="thm:pathwise-energy-lower-bound"} Let $(D_t)_{t<m}$ be a finite sequence of decision problems on a common coordinate interface, and let $I_t$ be a sufficient coordinate set for $D_t$ at each stage. Then for every thermodynamic model with positive per-bit conversion constant, $$\sum_{t<m} M.\mathrm{joulesPerBit}\cdot \mathrm{srank}(D_t)
\le
\sum_{t<m} \mathrm{energyLowerBound}(M,|I_t|).$$ If, in addition, each stage is realized by one substrate step, then Proposition [\[prop:substrate-time-law\]](#prop:substrate-time-law){reference-type="ref" reference="prop:substrate-time-law"} identifies the same index set with interface time, so the right-hand side is an additive pathwise lower bound along the trajectory.
:::

::: proof
*Proof.* Apply the structural-rank energy theorem independently at each stage and sum the resulting inequalities. The substrate-time proposition contributes the time interpretation: one realized stage is one interface time unit. ◻
:::

## Finite Quotient Trajectories

::: theorem
[]{#thm:quotient-trajectory-crooks label="thm:quotient-trajectory-crooks"} Let $$q_0 \to q_1 \to \cdots \to q_\tau$$ be a finite quotient trajectory in a finite Markov model, and write $P_{\mathrm f}$ and $P_{\mathrm r}$ for the corresponding forward and reverse stationary path weights induced by theorem-level edge flows. Suppose every edge on the path has positive forward and reverse flow, and suppose the local log forward/reverse flow ratio is calibrated by the structural-rank entropy increment at each step. Then $$\frac{P_{\mathrm f}(q_0 \to \cdots \to q_\tau)}{P_{\mathrm r}(q_\tau \to \cdots \to q_0)}
=
\exp\!\left(\sum_{t<\tau} \Delta H^{\mathrm{srank}}_t\right),$$ where $$\Delta H^{\mathrm{srank}}_t
=
H_{\mathrm{nats}}\bigl(\mathrm{srank}_{t+1}\bigr)-H_{\mathrm{nats}}\bigl(\mathrm{srank}_t\bigr).$$
:::

::: proof
*Proof.* The theorem-level path weight is the product of the local stationary edge-flow factors along the path. Taking the forward/reverse ratio multiplies the corresponding local edge-flow ratios, and the rank-calibration hypothesis identifies the logarithm of each local ratio with the structural-rank entropy increment at that step. Exponentiating the sum of those local increments gives the displayed path-ratio identity. ◻
:::

::: theorem
[]{#thm:rank-calibrated-crooks-standard label="thm:rank-calibrated-crooks-standard"} With the notation of Theorem [\[thm:quotient-trajectory-crooks\]](#thm:quotient-trajectory-crooks){reference-type="ref" reference="thm:quotient-trajectory-crooks"}, assume positive $k_B,T$ and the additional cumulative calibration $$\sum_{t<\tau}\Delta H_t^{\mathrm{srank}}
=
\frac{W-\Delta F}{k_B T}.$$ Then $$\frac{P_{\mathrm f}(q_0\to\cdots\to q_\tau)}{P_{\mathrm r}(q_\tau\to\cdots\to q_0)}
=
\exp\!\left(\frac{W-\Delta F}{k_B T}\right).$$
:::

::: proof
*Proof.* Substitute the cumulative calibration identity into Theorem [\[thm:quotient-trajectory-crooks\]](#thm:quotient-trajectory-crooks){reference-type="ref" reference="thm:quotient-trajectory-crooks"}. ◻
:::

::: theorem
[]{#thm:crooks-detailed-balance-equilibrium label="thm:crooks-detailed-balance-equilibrium"} Let $K$ be a finite quotient-MCMC kernel satisfying detailed balance against its Boltzmann witness law, and let $\pi$ be a stationary distribution whose probabilities coincide with that Boltzmann law on states. For any finite quotient trajectory with positive forward and reverse edge flows:

1.  each local forward/reverse edge-flow log ratio is zero,

2.  this yields a valid Crooks calibration witness against a constant-rank entropy-step profile,

3.  hence the forward/reverse stationary path-weight ratio is $$\frac{P_{\mathrm f}}{P_{\mathrm r}}=1.$$
:::

::: proof
*Proof.* Detailed balance plus the stationary/Boltzmann identification makes each directed stationary edge flow equal to its reverse partner, so each local log ratio vanishes. This is exactly the rank-calibration hypothesis with constant rank (zero entropy increment per step). Substituting into the rank-calibrated Crooks identity gives a zero exponent and therefore unit path ratio. ◻
:::

The continuous-time bridge is now connected at interface level as well: an overdamped Langevin transition-kernel package with a Boltzmann detailed-balance witness transports directly to the finite quotient-MCMC layer, and its Euler--Maruyama discretization includes an explicit theorem-level transition-kernel error certificate (L591, L592).

::: theorem
[]{#thm:md-class-crooks-calibration-interface label="thm:md-class-crooks-calibration-interface"} Fix a finite quotient trajectory in a declared nonequilibrium molecular-dynamics class that supplies theorem-level stepwise calibration $$\log\frac{J_t^+}{J_t^-} = \Delta H_t^{\mathrm{srank}}$$ for every trajectory step. If the same class supplies cumulative work calibration $$\sum_{t<\tau}\Delta H_t^{\mathrm{srank}} = \frac{W-\Delta F}{k_B T},$$ with $k_B,T>0$, then the trajectory obeys the Crooks standard form $$\frac{P_{\mathrm f}}{P_{\mathrm r}}=
\exp\!\left(\frac{W-\Delta F}{k_B T}\right).$$
:::

::: proof
*Proof.* Apply the interface theorem that composes (i) the declared stepwise calibration witness from the MD class with (ii) the existing finite-trajectory Crooks reduction and (iii) the cumulative work/free-energy calibration identity. ◻
:::

::: theorem
[]{#thm:langevin-to-mcmc-discretization-interface label="thm:langevin-to-mcmc-discretization-interface"} Let a declared overdamped Langevin transition-kernel package provide:

1.  a Boltzmann witness law,

2.  detailed balance of the continuous kernel against that law,

3.  an Euler--Maruyama discrete kernel family with certified per-transition error envelope.

Then:

1.  the continuous kernel satisfies detailed balance in quotient-Boltzmann form,

2.  for every nonnegative step size $\delta$, there exists a quotient-MCMC kernel identified with the Euler--Maruyama discretization and satisfying the declared transition-kernel error bound at scale $\delta$.
:::

::: proof
*Proof.* Both clauses are direct interface specializations: the first rewrites the declared Boltzmann witness into quotient-Boltzmann form and applies the declared detailed-balance law; the second packages the declared Euler--Maruyama kernel and its error certificate at the chosen step size. ◻
:::

::: theorem
[]{#thm:jarzynski-from-crooks label="thm:jarzynski-from-crooks"} Let $\Gamma$ be a finite trajectory family with strictly positive forward and reverse trajectory distributions $p_{\mathrm f},p_{\mathrm r}$. Assume positive $k_B,T$ and pointwise Crooks relation $$p_{\mathrm f}(\gamma)
=
p_{\mathrm r}(\gamma)
\exp\!\left(\frac{W(\gamma)-\Delta F}{k_B T}\right)
\qquad (\gamma\in\Gamma).$$ Then $$\sum_{\gamma\in\Gamma}
p_{\mathrm f}(\gamma)
\exp\!\left(-\frac{W(\gamma)}{k_B T}\right)
=
\exp\!\left(-\frac{\Delta F}{k_B T}\right).$$
:::

::: proof
*Proof.* Replace $p_{\mathrm f}$ by the Crooks expression, combine exponentials inside the finite sum, and use normalization of $p_{\mathrm r}$. ◻
:::

::: theorem
[]{#thm:quotient-trajectory-dissipation label="thm:quotient-trajectory-dissipation"} With the same notation, let $W^{\mathrm{diss}}_t$ be the dissipated work assigned to step $t$, and suppose each step satisfies $$k_B T\,\Delta H^{\mathrm{srank}}_t \le W^{\mathrm{diss}}_t.$$ Then the total dissipation along the finite quotient trajectory obeys $$k_B T\sum_{t<\tau} \Delta H^{\mathrm{srank}}_t
\le
\sum_{t<\tau} W^{\mathrm{diss}}_t.$$
:::

::: proof
*Proof.* The assumed lower bound holds at each step individually. Summing those stepwise inequalities over the finite trajectory yields the announced cumulative dissipation bound. ◻
:::

These theorems are the strongest non-equilibrium trajectory statements currently mechanized in the artifact. They are finite path-weight and cumulative-dissipation theorems. The detailed-balance corollary discharges a concrete equilibrium calibration branch (unit forward/reverse path ratio under Boltzmann-stationary matching), and the Langevin-to-discrete interface now supplies a theorem-level continuous-to-discrete calibration endpoint. Full analytic SDE existence/regularity and sharp strong-error-rate derivations remain outside the present formalization.

## Physical Instantiation and Constructive Extraction

This subsection makes the physical-instantiation layer explicit in one place: classical composed force fields, continuous Langevin dynamics, concrete spin-$\tfrac12$ quantum readout, and a constructive top-level computable extraction path.

### Classical Force-Field Bridge

The composed classical Hamiltonian package combines bonded, Lennard--Jones, and Coulomb terms into one energy object $$H(a,s)=H_{\mathrm{bond}}(a,s)+H_{\mathrm{LJ}}(a,s)+H_{\mathrm{Coul}}(a,s),$$ with architecture metadata transported from that same package.

::: theorem
[]{#thm:composed-hamiltonian-architecture-instance label="thm:composed-hamiltonian-architecture-instance"} Every declared composed classical Hamiltonian induces an architecture instance whose degree-of-freedom and capability fields are exactly the declared metadata of that Hamiltonian package.
:::

::: proof
*Proof.* This is a direct projection theorem from the composed Hamiltonian record to the architecture record. ◻
:::

::: theorem
[]{#thm:composed-hamiltonian-lipschitz-bound label="thm:composed-hamiltonian-lipschitz-bound"} If bonded, Lennard--Jones, and Coulomb components are each Lipschitz on a declared positive shell with constants $L_{\mathrm{bond}},L_{\mathrm{LJ}},L_{\mathrm{Coul}}$, then the full composed Hamiltonian is Lipschitz on the same shell with constant $$L_{\mathrm{bond}}+L_{\mathrm{LJ}}+L_{\mathrm{Coul}}.$$
:::

::: proof
*Proof.* Apply the triangle inequality to the three component differences and substitute the three declared shellwise bounds. ◻
:::

::: theorem
[]{#thm:concrete-biomolecular-forcefield-calibration-bundle label="thm:concrete-biomolecular-forcefield-calibration-bundle"} For a declared biomolecular parameter bundle and shell calibration witness:

1.  the composed force-field package induces an architecture object with matching declared DOF/capability metadata,

2.  the full composed Hamiltonian satisfies the positive-shell Lipschitz bound with the declared summed shell constant,

3.  the same shell constant is sufficient to instantiate half-gap winner transport for grid-to-continuous utility approximations.
:::

::: proof
*Proof.* The first clause is the architecture projection endpoint for the concrete composed package. The second clause is the composed positive-shell Lipschitz theorem specialized to that package. The third clause applies the theorem-level half-gap transport result using the same summed shell constant. ◻
:::

::: theorem
[]{#thm:reference-biomolecular-zero-shell-calibration label="thm:reference-biomolecular-zero-shell-calibration"} The explicit reference biomolecular parameter family admits a fully concrete shell calibration with declared real constants $$L_{\mathrm{bond}}=L_{\mathrm{LJ}}=L_{\mathrm{Coul}}=0,$$ and therefore the corresponding composed Hamiltonian is globally zero-Lipschitz in state.
:::

::: proof
*Proof.* The first statement is the concrete zero-shell witness theorem (plus its reference-family specialization). Substituting those constants into the composed-energy difference bound yields the global zero-Lipschitz claim. ◻
:::

::: theorem
[]{#thm:nontrivial-biomolecular-forcefield-transport label="thm:nontrivial-biomolecular-forcefield-transport"} For the state-aware nontrivial composed force-field family:

1.  the composed energy is Lipschitz along the designated coordinate anchor with an explicit nonzero summed shell constant,

2.  this nonzero constant transports through the same half-gap winner-preservation theorem schema used in the zero-shell case.
:::

::: proof
*Proof.* The first clause is the nontrivial anchor-direction Lipschitz theorem, where the coefficient sum is controlled by an absolute-value triangle bound. The second clause is the corresponding half-gap transport theorem instantiated with that nontrivial shell constant. ◻
:::

::: theorem
[]{#thm:fullstate-biomolecular-forcefield-transport label="thm:fullstate-biomolecular-forcefield-transport"} For a calibrated full-state composed Hamiltonian with bonded/Lennard--Jones/Coulomb dependence on all molecular coordinates:

1.  the full-state energy is Lipschitz under an explicit summed shell constant,

2.  the same constant yields the corresponding half-gap winner-preservation transport theorem,

3.  an explicit calibrated parameter family witnesses strictly positive bonded, LJ, and Coulomb shell constants.
:::

::: proof
*Proof.* The Lipschitz clause is obtained by combining componentwise full-state bounds through the composed-Hamiltonian shell theorem. Half-gap transport is the direct instantiation of the same constant in the existing winner-preservation theorem schema. The positivity claim is a concrete numeric calibration theorem for the designated full-state parameter bundle. ◻
:::

::: theorem
[]{#thm:pairwise-geometric-forcefield-transport label="thm:pairwise-geometric-forcefield-transport"} For a pairwise geometric bonded/Lennard--Jones/Coulomb calibration bundle with componentwise per-pair Lipschitz witnesses:

1.  the resulting pairwise composed Hamiltonian is globally Lipschitz with an explicit carried sharp shell constant,

2.  the same sharp shell constant transports through half-gap winner-preservation.
:::

::: proof
*Proof.* Componentwise pairwise bounds are aggregated by finite-sum absolute-value control and then fed to the composed-Hamiltonian shell theorem. Half-gap transport is the corresponding direct instantiation with that carried sharp constant. ◻
:::

::: theorem
[]{#thm:pairwise-geometric-derived-transport label="thm:pairwise-geometric-derived-transport"} If pairwise bonded/Lennard--Jones/Coulomb interactions satisfy explicit geometric analytic assumptions (stretch bounds, radius floor, charge envelope, and distance-level transport bounds), then the pairwise composed Hamiltonian admits derived sharp shell constants and the corresponding half-gap transport theorem.
:::

::: proof
*Proof.* Use the explicit geometric assumptions to derive per-pair Lipschitz constants, aggregate them across pair sets to obtain sharp component constants, apply composed-Hamiltonian Lipschitz transport, and instantiate the half-gap theorem with the derived summed shell constant. ◻
:::

::: theorem
[]{#thm:realism-augmented-forcefield-transport label="thm:realism-augmented-forcefield-transport"} If a full-state docking Hamiltonian is augmented by explicit solvent, polarization, many-body, and long-range correction terms with declared componentwise Lipschitz shells, then:

1.  the aggregate correction is globally Lipschitz with shell equal to the sum of the component shells,

2.  the fully augmented Hamiltonian is globally Lipschitz under the summed base-plus-correction shell,

3.  the same summed shell transports through the half-gap winner-preservation theorem.
:::

::: proof
*Proof.* Apply finite-sum absolute-value control to aggregate component correction shells, combine with the existing full-state base Hamiltonian Lipschitz theorem, and instantiate the standard half-gap transport theorem with the resulting total shell constant. ◻
:::

::: theorem
[]{#thm:constructive-empirical-realism-shell-envelope label="thm:constructive-empirical-realism-shell-envelope"} For the constructive empirical realism instantiation, each realism correction shell (solvent, polarization, many-body, long-range real, long-range reciprocal) is explicitly bounded by $$\text{empirical mean shell} + z \cdot \frac{1}{\sqrt{N}},$$ with shared sample size $N$ and confidence multiplier $z$.
:::

::: proof
*Proof.* Each shell is defined as empirical mean plus $z$ times the corresponding standard error, and each standard error is assumed to satisfy the same $1/\sqrt{N}$ finite-sample envelope. Multiplying by nonnegative $z$ and adding the mean yields the five shell inequalities. ◻
:::

::: theorem
[]{#thm:constructive-empirical-realism-transport label="thm:constructive-empirical-realism-transport"} Any constructive empirical realism instantiation canonically recovers the abstract solvent/polarization/many-body/long-range realism layer; therefore:

1.  the realism-augmented composed Hamiltonian inherits global Lipschitz transport under the empirical shell aggregate,

2.  the same empirical aggregate shell transports through half-gap winner preservation.
:::

::: proof
*Proof.* Convert the constructive empirical package into the abstract realism layer by direct field mapping, then apply the already proved realism-augmented Lipschitz and half-gap transport theorems to that converted layer. ◻
:::

::: theorem
[]{#thm:biomolecular-reference-realism-shell-values label="thm:biomolecular-reference-realism-shell-values"} The reference empirical realism calibration instantiates explicit numeric shell values for solvent, polarization, many-body, long-range real-space, and long-range reciprocal corrections.
:::

::: proof
*Proof.* Unfold the concrete reference calibration definition: all shell values reduce by definitional simplification because the reference standard-error fields are fixed constants. ◻
:::

::: theorem
[]{#thm:biomolecular-reference-realism-transport label="thm:biomolecular-reference-realism-transport"} For the same concrete reference realism calibration, the associated realism-augmented Hamiltonian inherits both:

1.  global Lipschitz transport under the concrete aggregated shell,

2.  half-gap winner-preservation transport for discretized resolution maps.
:::

::: proof
*Proof.* Instantiate the constructive empirical realism transport theorems with the concrete reference calibration package and its certified componentwise Lipschitz correction terms. ◻
:::

::: theorem
[]{#thm:biomolecular-reference-realism-explicit-shell-constant label="thm:biomolecular-reference-realism-explicit-shell-constant"} For the same concrete reference realism package:

1.  the realism-correction aggregate shell constant is explicit, $$L_{\mathrm{corr}}=\frac{53}{20},$$

2.  the realism-augmented Hamiltonian obeys the explicit global bound $$|\Delta E| \le \left(L_{\mathrm{full}}+\frac{53}{20}\right)\,d_{\mathrm{MD},1},$$ where $L_{\mathrm{full}}$ is the full-state composed-force-field shell constant and $d_{\mathrm{MD},1}$ is the MD state $L^1$ distance.
:::

::: proof
*Proof.* The aggregate correction shell is the sum of the five calibrated component shells; substituting the concrete reference values gives $53/20$. Substitute this identity into the already proved concrete realism Lipschitz transport inequality. ◻
:::

::: theorem
[]{#thm:electronic-structure-correction-transport label="thm:electronic-structure-correction-transport"} If realism-augmented docking Hamiltonians are further corrected by charge-transfer and metal-coordination terms with explicit shell and absolute-error envelopes, then:

1.  the fully augmented Hamiltonian inherits global Lipschitz transport under the summed realism-plus-electronic shell constant,

2.  the realism-only vs realism+electronic per-state energy discrepancy is uniformly bounded by the electronic absolute-error constant,

3.  the same summed shell constant transports through half-gap winner-preservation.
:::

::: proof
*Proof.* Combine the realism-augmented Lipschitz theorem with the additive electronic correction Lipschitz bound; absolute discrepancy follows from the electronic correction absolute-error theorem; then instantiate the standard half-gap transport theorem with the resulting total shell constant. ◻
:::

::: theorem
[]{#thm:electronic-reference-shell-error-transport label="thm:electronic-reference-shell-error-transport"} For the concrete electronic-structure reference layer, shell/error constants are explicit: $$L_{\mathrm{elec}}=0,\qquad E_{\mathrm{elec}}=\frac12.$$ Consequently, realism+electronic Lipschitz transport reduces to the realism shell bound, the per-state realism-only vs realism+electronic energy discrepancy is bounded by $1/2$, and half-gap transport specializes with the same realism-only shell constant.
:::

::: proof
*Proof.* Unfold the concrete reference layer to evaluate shell/error constants directly; substitute those values into the general electronic-structure Lipschitz/error transport theorems. ◻
:::

::: theorem
[]{#thm:qm-grounded-electronic-structure-transport label="thm:qm-grounded-electronic-structure-transport"} If an electronic-structure correction layer is equipped with chemistry-class-specific QM shell/error envelopes that dominate the base charge-transfer and metal-coordination shells/errors at a designated active chemistry class, then:

1.  the base electronic shell/error constants are bounded by the active-class QM envelopes,

2.  realism+electronic global Lipschitz transport upgrades to the summed realism-plus-QM-tight shell constant,

3.  realism-only vs realism+electronic per-state discrepancy is bounded by the QM-tight error constant.
:::

::: proof
*Proof.* First transport base shell/error constants to active-class QM envelopes by the class-dominance inequalities. Then substitute these bounds into the previously proved realism+electronic Lipschitz and absolute-discrepancy transport theorems. ◻
:::

::: theorem
[]{#thm:qm-grounded-electronic-halfgap-transport label="thm:qm-grounded-electronic-halfgap-transport"} If the realism+electronic utility approximation is certified directly with the summed realism-plus-QM-tight shell constant, then half-gap winner preservation follows under that same summed constant.
:::

::: proof
*Proof.* Instantiate the generic Lipschitz-resolution half-gap theorem using the realism+electronic utility, the QM-tight shell constant, and its nonnegativity witness. ◻
:::

::: theorem
[]{#thm:qm-method-specific-electronic-transport label="thm:qm-method-specific-electronic-transport"} If QM shell/error envelopes are derived from method-specific affine calibration formulas over chemistry-class descriptors, and the base electronic layer is dominated by those active-class affine bounds, then the same realism+electronic transport endpoints follow with the resulting method-derived tight shell/error constants:

1.  global realism+electronic Lipschitz transport,

2.  realism-only vs realism+electronic absolute discrepancy transport.
:::

::: proof
*Proof.* Convert the affine method-calibration package into the QM-grounded electronic layer by fieldwise definition, then apply the existing QM-grounded realism Lipschitz and absolute-error transport theorems. ◻
:::

::: theorem
[]{#thm:qm-protocol-derived-method-calibration-transport label="thm:qm-protocol-derived-method-calibration-transport"} If active-class QM shell/error constants are calibrated directly by protocol-derived estimate/reference/error terms, then those protocol calibrations induce a method-specific QM layer and therefore the same realism+electronic transport endpoints (global Lipschitz transport and realism-only vs realism+electronic discrepancy transport).
:::

::: proof
*Proof.* Convert protocol-derived shell/error calibration terms into affine-intercept method constants (zero slopes) using the protocol absolute-calibration bounds, map into the existing method-specific QM layer, then apply the method-specific transport theorem. ◻
:::

::: theorem
[]{#thm:ewald-long-range-certificates label="thm:ewald-long-range-certificates"} Paper4 Ewald long-range certificates are exported to paper3: real-space exponential-decay upper bound and reciprocal-space positivity for positive radius/splitting/wavevector inputs.
:::

::: proof
*Proof.* Apply the imported paper4 Ewald decay and positivity theorems directly under the declared positivity premises. ◻
:::

::: remark
[]{#rem:hamiltonian-dof-loop-closure label="rem:hamiltonian-dof-loop-closure"} In Sections [\[foundations\]](#foundations){reference-type="ref" reference="foundations"}--[\[probability-model\]](#probability-model){reference-type="ref" reference="probability-model"}, $\mathrm{DOF}(A)$ is introduced as declared architecture data and then transported through the canonical identity $\mathrm{srank}(\mathrm{canonicalDP}(A))=\mathrm{DOF}(A)$. The Hamiltonian instantiation closes this loop: architecture data is supplied by a concrete composed force-field package, so the same Landauer floor becomes a physically instantiated prediction from energetic structure rather than a free standing counting declaration.
:::

### Langevin Dynamics Bridge

::: theorem
[]{#thm:langevin-detailed-balance-grounding label="thm:langevin-detailed-balance-grounding"} For a declared overdamped Langevin transition-kernel package with Boltzmann witness law, the continuous kernel satisfies detailed balance in quotient-Boltzmann form.
:::

::: proof
*Proof.* Rewrite the package Boltzmann witness into quotient-Boltzmann form and apply the package detailed-balance law. ◻
:::

::: theorem
[]{#thm:langevin-discretization-certified label="thm:langevin-discretization-certified"} For every nonnegative step size $\delta$, Euler--Maruyama discretization of the declared Langevin package yields a quotient-MCMC kernel together with a certified per-transition discretization error envelope at scale $\delta$.
:::

::: proof
*Proof.* Instantiate the package's Euler--Maruyama kernel at $\delta$ and transport the package error certificate to the quotient-MCMC transition map. ◻
:::

::: theorem
[]{#thm:langevin-boltzmann-stationarity-measure-derivation label="thm:langevin-boltzmann-stationarity-measure-derivation"} For a finite-state Langevin transition package at fixed time $t$, detailed balance plus row-stochastic normalization imply one-step Boltzmann stationarity in pushforward form: $$\rho_{t+1}(s') = \sum_s \rho_t(s)K_t(s,s'),\qquad
\rho_t=\pi_{\mathrm{Boltz}} \implies \rho_{t+1}=\pi_{\mathrm{Boltz}}.$$ The same statement is exported in quotient-Boltzmann notation.
:::

::: proof
*Proof.* Sum the detailed-balance identity over the source state, factor out the target Boltzmann weight, and use row-stochastic normalization. Quotient-Boltzmann form is obtained by rewriting with the package's Boltzmann-identification field. ◻
:::

::: theorem
[]{#thm:langevin-analysis-closure-endpoint-bundle label="thm:langevin-analysis-closure-endpoint-bundle"} Assume a declared overdamped Langevin model is equipped with:

1.  a strong-solution existence/uniqueness witness,

2.  a Boltzmann invariance/ergodicity witness,

3.  Euler--Maruyama strong and weak rate witnesses.

Then the corresponding five endpoint claims hold at theorem level:

1.  existence and uniqueness of the declared strong solution,

2.  Boltzmann invariance,

3.  ergodicity,

4.  a strong-error envelope $O(\sqrt{\delta})$,

5.  a weak-error envelope $O(\delta)$.
:::

::: proof
*Proof.* Each clause is exactly the corresponding endpoint theorem extracted from the declared witness data for the same Langevin model. ◻
:::

::: theorem
[]{#thm:unified-langevin-assumption-bundle-discharge label="thm:unified-langevin-assumption-bundle-discharge"} If one declared formal-analysis assumption bundle provides strong-solution existence/uniqueness, Boltzmann invariance/ergodicity, and Euler--Maruyama strong/weak rate hypotheses for a given overdamped Langevin model, then the full five-endpoint closure package follows simultaneously from that single bundle.
:::

::: proof
*Proof.* Apply the assumption-bundle discharge theorem, which composes the witness constructors with the endpoint theorems and returns the conjunction of all five target claims. ◻
:::

::: theorem
[]{#thm:langevin-explicit-sde-conditions label="thm:langevin-explicit-sde-conditions"} The first-principles Langevin package now exports explicit SDE-condition constants and inequalities: global Lipschitz, linear growth, and Lyapunov dissipativity. The same package also exports a dissipativity-parameterized ergodicity consequence for its Boltzmann measure.
:::

::: proof
*Proof.* The explicit-conditions theorem is a direct tuple projection from the first-principles assumption record. The dissipative-ergodicity theorem combines the same drift inequality constants with the package ergodicity field into one endpoint statement. ◻
:::

::: theorem
[]{#thm:langevin-microscopic-derivation-constructor label="thm:langevin-microscopic-derivation-constructor"} An explicit microscopic Langevin derivation package (microscopic drift identity, analytic inequalities, Boltzmann invariance/ergodicity, strong-solution uniqueness, and strong/weak rate constants) canonically constructs the first-principles Langevin interface and therefore exports both:

1.  the explicit SDE inequality tuple,

2.  the full five-endpoint continuous-time closure bundle,

3.  the dissipativity-parameterized ergodicity consequence directly at the microscopic layer.
:::

::: proof
*Proof.* Use the microscopic-to-first-principles constructor to map microscopic drift inequalities to the interface drift inequalities by rewriting along the drift identity, then apply the existing explicit-SDE and first-principles endpoint theorems. ◻
:::

::: theorem
[]{#thm:langevin-molecular-hamiltonian-thermostat-constructor label="thm:langevin-molecular-hamiltonian-thermostat-constructor"} For a concrete microscopic package that contains:

1.  a molecular realism+electronic Hamiltonian energy identity,

2.  thermostat metadata,

3.  a microscopic drift derivation certificate,

the first-principles Langevin interface is constructed directly and exports both the explicit SDE tuple and the full continuous-time five-endpoint closure bundle.
:::

::: proof
*Proof.* Use the molecular-Hamiltonian microscopic package to build the first-principles assumption record through the microscopic constructor, then apply the microscopic explicit-SDE and endpoint-export theorems. ◻
:::

::: theorem
[]{#thm:langevin-concrete-hamiltonian-bath-end-to-end label="thm:langevin-concrete-hamiltonian-bath-end-to-end"} If a concrete Hamiltonian+bath molecular system carries one Ito/Fokker--Planck/Harris continuous-state closure package, then the microscopic Langevin derivation object is constructed internally from that closure record and the full five-endpoint Langevin closure bundle is exported at theorem level.
:::

::: proof
*Proof.* Construct the microscopic derivation record directly from the closure fields (drift, analytic inequalities, Boltzmann invariance/ergodicity, strong-solution uniqueness, and rate constants), convert to the molecular-Hamiltonian microscopic package, and apply the existing molecular endpoint theorem. ◻
:::

::: theorem
[]{#thm:langevin-molecular-pathprocess-joint-bundle label="thm:langevin-molecular-pathprocess-joint-bundle"} For a concrete molecular Hamiltonian+thermostat microscopic package and a canonical continuous path-process construction over the same Langevin model, the full microscopic five-endpoint closure bundle and the canonical process-level closure bundle are exported together in one theorem endpoint.
:::

::: proof
*Proof.* Conjoin the molecular-Hamiltonian microscopic endpoint theorem with the canonical continuous path-process closure theorem for the same model. ◻
:::

::: theorem
[]{#thm:langevin-continuous-state-measure-closure label="thm:langevin-continuous-state-measure-closure"} For a continuous-state overdamped Langevin model equipped with a measure-kernel semigroup realization, detailed-balance-at-one witness, and explicit SDE analytic inequalities:

1.  stationarity propagates to all semigroup scales,

2.  explicit global Lipschitz/linear-growth/dissipativity constants are exported,

3.  existence/uniqueness, Boltzmann invariance/ergodicity, and strong/weak Euler--Maruyama endpoint envelopes follow as one closure bundle.
:::

::: proof
*Proof.* All-scale stationarity is the scale-flow theorem from detailed balance at one step. The analytic constants are record projections. Endpoint discharge is obtained by conversion to the first-principles assumption package and application of the bundled first-principles closure theorem. ◻
:::

::: theorem
[]{#thm:langevin-ito-fp-harris-closure label="thm:langevin-ito-fp-harris-closure"} If the continuous-state Langevin closure package is augmented with: (i) an Ito-generator martingale-wellposedness layer, (ii) a Fokker--Planck stationary-transport layer, and (iii) a Harris minorization/petite-set ergodicity layer, then the same endpoint closure package follows, with those additional certificates exported in one theorem bundle.
:::

::: proof
*Proof.* Combine the base continuous-state closure theorem with the layer-specific certificate fields: Fokker--Planck stationarity transport supplies scalewise Boltzmann stationarity, Harris supplies the ergodicity conclusion, and the Ito/Fokker--Planck/Harris proposition certificates are appended by conjunction. ◻
:::

::: theorem
[]{#thm:langevin-infinite-dimensional-path-measure-bridge label="thm:langevin-infinite-dimensional-path-measure-bridge"} If the continuous-time Ito/Fokker--Planck/Harris closure package is extended with a path-space layer (measurable path extension, path measure, cylindrical consistency/closure, pathwise martingale-on-cylinder certificate, and pathwise uniqueness certificate), then these path-measure certificates transport as one bundled theorem-level endpoint together with the base closure certificates.
:::

::: proof
*Proof.* Apply the path-measure extension certificate theorem to extract cylindrical/pathwise certificates, then conjoin with the already-exported Ito/Fokker--Planck/Harris certificates from the underlying closure package. ◻
:::

::: theorem
[]{#thm:langevin-constructive-infinite-dimensional-path-derivation-bridge label="thm:langevin-constructive-infinite-dimensional-path-derivation-bridge"} If the infinite-dimensional path layer is supplied with constructive finite-horizon measures, projective-consistency witness, and explicit Kolmogorov-extension witness, then:

1.  finite-horizon consistency and extension-match certificates are exported together with pathwise/martingale and Ito/Fokker--Planck/Harris certificates,

2.  forgetting the finite-horizon construction data recovers the previous certificate-only infinite-dimensional path-measure closure theorem.
:::

::: proof
*Proof.* The first clause is a direct conjunction of the constructive derivation certificates with the existing Ito/Fokker--Planck/Harris certificates. The second clause is obtained by converting the constructive package to the prior path-layer interface and applying the previously proved infinite-dimensional closure theorem. ◻
:::

::: theorem
[]{#thm:langevin-finite-horizon-law-derivation-bridge label="thm:langevin-finite-horizon-law-derivation-bridge"} If finite-horizon path laws are given with explicit projective-consistency witnesses and a Kolmogorov extension witness for the designated measurable transition kernel/initial law, then:

1.  these finite-horizon witnesses induce the constructive infinite-dimensional path-derivation closure bundle,

2.  forgetting constructive finite-horizon details recovers the prior certificate-export infinite-dimensional closure endpoint.
:::

::: proof
*Proof.* Package the finite-horizon/projective/extension witnesses into the finite-horizon law bridge object and apply the two conversion theorems: first to constructive infinite-dimensional closure, then to its certificate-only forgetful image. ◻
:::

::: theorem
[]{#thm:langevin-finite-horizon-marginal-recovery label="thm:langevin-finite-horizon-marginal-recovery"} The finite-horizon constructive law bridge exports both marginal-recovery formulas explicitly:

1.  projective truncation recovers every lower-horizon finite-path measure from each higher-horizon one,

2.  Kolmogorov-extension projection recovers every finite-horizon marginal from the infinite-path extension measure.
:::

::: proof
*Proof.* Both clauses are direct fields of the finite-horizon projective-consistency and Kolmogorov-extension witnesses carried by the bridge package. ◻
:::

::: theorem
[]{#thm:langevin-canonical-continuous-path-process-closure label="thm:langevin-canonical-continuous-path-process-closure"} If the finite-horizon/Kolmogorov constructive bridge is augmented with canonical process-level regularity witnesses (measurable evaluation, continuity, regularity, and pathwise uniqueness), then:

1.  these process-level certificates and the constructive finite-horizon closure certificates are exported together in one endpoint,

2.  projective and extension marginal-recovery formulas are retained explicitly.
:::

::: proof
*Proof.* Conjoin the process-level witness fields with the finite-horizon closure theorem for the same bridge package; marginal recovery is the direct finite-horizon bridge marginal theorem. ◻
:::

::: theorem
[]{#thm:langevin-canonical-pathprocess-of-finite-horizon-bridge label="thm:langevin-canonical-pathprocess-of-finite-horizon-bridge"} Given Ito/Fokker--Planck/Harris closure and finite-horizon projective/Kolmogorov bridge data, the canonical continuous path-process object is constructed directly (without separately supplied process-regularity certificates), and the full canonical closure endpoint bundle is exported.
:::

::: proof
*Proof.* Build the canonical process object from the finite-horizon bridge constructor and then apply the canonical closure theorem to that constructed object. ◻
:::

::: theorem
[]{#thm:langevin-forcefield-derived-lipschitz-injection label="thm:langevin-forcefield-derived-lipschitz-injection"} Assume closure state distance is instantiated by molecular $L^1$ distance and drift increments are controlled by pairwise-forcefield energy differences through a nonnegative coupling coefficient. Then the pairwise geometric derived shell constant plugs directly into the closure drift-Lipschitz hypothesis, yielding an explicit SDE-constant tuple with force-field-derived drift constant.
:::

::: proof
*Proof.* First apply the pairwise geometric derived Lipschitz theorem to bound energy differences. Multiply by the nonnegative drift/energy coupling coefficient to obtain a drift Lipschitz bound. Rewrite through the state-distance identification and substitute into the continuous-state explicit-constant tuple. ◻
:::

::: theorem
[]{#thm:langevin-first-principles-discharge label="thm:langevin-first-principles-discharge"} If one first-principles analytic package supplies explicit global Lipschitz, linear-growth, and dissipativity/Lyapunov inequalities together with Boltzmann invariance/ergodicity and strong/weak rate constants, then existence/uniqueness, invariance, ergodicity, and strong/weak Euler--Maruyama endpoint envelopes follow as one bundled theorem consequence.
:::

::: proof
*Proof.* Apply the first-principles discharge theorem, which derives all five endpoint clauses directly from the analytic package and then returns their conjunction. ◻
:::

::: theorem
[]{#thm:concrete-langevin-interface-constructor label="thm:concrete-langevin-interface-constructor"} Any concrete overdamped Langevin witness package with Boltzmann calibration, detailed balance, and Euler--Maruyama approximation data canonically constructs a paper3 Langevin transition-kernel interface object.
:::

::: proof
*Proof.* Unfold the constructor and package each declared concrete field directly into the corresponding interface field. ◻
:::

### Spin-$\tfrac12$ Quantum Instantiation

::: theorem
[]{#thm:spinhalf-canonicaldp-instance label="thm:spinhalf-canonicaldp-instance"} For every declared spin-$\tfrac12$ substrate package, the classical readout has an explicit two-state witness and the induced decision problem is exactly $\mathrm{canonicalDP}(1)$ with structural rank $1$.
:::

::: proof
*Proof.* Combine the explicit two-state witness theorem with the canonical embedding theorem for the same package. ◻
:::

::: theorem
[]{#thm:spinhalf-decoherence-floor label="thm:spinhalf-decoherence-floor"} For the same spin-$\tfrac12$ package, the decoherence/readout event cost is exactly $$k_B T\ln 2.$$
:::

::: proof
*Proof.* Specialize the abstract decoherence Landauer identity to the rank-one canonical embedding from Theorem [\[thm:spinhalf-canonicaldp-instance\]](#thm:spinhalf-canonicaldp-instance){reference-type="ref" reference="thm:spinhalf-canonicaldp-instance"}. ◻
:::

### Constructive Extraction and Computability

::: theorem
[]{#thm:top-level-computable-path label="thm:top-level-computable-path"} The end-to-end molecular pipeline admits a constructive implementation path: finite coordinate extraction by computable greedy search, fuel-bounded rational-certificate refinement, and exact ArrayDSL JSON export, all with theorem-level output-equality guarantees.
:::

::: proof
*Proof.* Unfold the constructive output bundle and apply the endpoint computability theorem, which states each output component by definitional equality. ◻
:::

::: theorem
[]{#thm:legacy-constructive-deprecation-bridge label="thm:legacy-constructive-deprecation-bridge"} Given alignment witnesses for retained coordinates, lifted refinement results, and export payload, the legacy and constructive cross-docking outputs coincide in the shared normalized view; therefore critical exported outputs can be read directly from the constructive path.
:::

::: proof
*Proof.* Apply the normalization-equivalence theorem and the deprecation-ready consequence theorem; the latter packages normalized equality together with export-payload identity. ◻
:::

::: theorem
[]{#thm:constructive-downstream-replacement-transport label="thm:constructive-downstream-replacement-transport"} Under the same normalization-alignment hypotheses, replacement is stable for downstream use:

1.  retained coordinates and refinement result fields agree fieldwise after normalization,

2.  any downstream consumer on normalized outputs is invariant under legacy-to-constructive replacement.
:::

::: proof
*Proof.* The fieldwise statement is the explicit normalized-field replacement theorem. The consumer invariance statement follows by congruence from normalized output equality. ◻
:::

::: theorem
[]{#thm:fully-constructive-pipeline-deprecation-ready label="thm:fully-constructive-pipeline-deprecation-ready"} For the fully constructive top-level cross-docking pipeline, legacy-view normalization and constructive-view normalization coincide automatically, and the exported payload is already the canonical ArrayDSL JSON target; no manual alignment witness construction is required.
:::

::: proof
*Proof.* Instantiate the automatic constructive-wrapper theorem on outputs of the fully constructive pipeline; export payload identity is definitional for that pipeline output bundle. ◻
:::

::: theorem
[]{#thm:constructive-only-spec-replacement label="thm:constructive-only-spec-replacement"} For any downstream proposition on normalized outputs, legacy-wrapper evaluation and native constructive evaluation are logically equivalent once canonical export-payload identity is established. Thus downstream theorem consumers can be switched to constructive-only inputs without loss.
:::

::: proof
*Proof.* Apply the constructive-only specification equivalence theorem, which rewrites both sides through the automatic wrapper alignment equality obtained from the constructive output and export-payload check. ◻
:::

::: theorem
[]{#thm:constructive-only-core-field-consumers label="thm:constructive-only-core-field-consumers"} For constructive normalized outputs with canonical export payload, the standard downstream field consumers are already constructive-only equivalent: retained-coordinate membership predicates, refinement-result existence predicates, and canonical export-json checks.
:::

::: proof
*Proof.* Instantiate the proposition-level constructive-only equivalence principle on each field consumer predicate and combine the resulting equivalences with the export identity rewrite. ◻
:::

::: theorem
[]{#thm:constructive-only-extended-field-consumers label="thm:constructive-only-extended-field-consumers"} Beyond core predicates, additional downstream consumers are constructive-only equivalent: retained-coordinate cardinality and exact normalized refinement-payload equality both transport from legacy-wrapper view to native constructive view under canonical export identity.
:::

::: proof
*Proof.* Apply normalized output equivalence and project the required fields by congruence (retained-coordinate cardinality and refinement payload). Combine with canonical export identity to obtain the bundled extended consumer migration claim. ◻
:::

::: theorem
[]{#thm:continuous-state-measurable-encoding-transport label="thm:continuous-state-measurable-encoding-transport"} For optimizer-preserving measurable encoding equivalences between continuous-state decision problems, together with compatible measurable finite partition codes:

1.  coordinate relevance is equivalent across source and target,

2.  partition fibers are mapped exactly by the encoding equivalence,

3.  structural rank is invariant,

4.  Landauer energy floors are transported exactly,

5.  these transport clauses are exported together as one bundled theorem endpoint.
:::

::: proof
*Proof.* Apply the encoding transport witness theorems for relevance, structural rank, and energy floor. The partition-fiber identity follows by unfolding code compatibility and transporting singleton-fiber membership through the state equivalence. ◻
:::

### Chemical, Ensemble, and Kinetic Docking Interfaces

::: theorem
[]{#thm:chemical-augmented-docking-transport label="thm:chemical-augmented-docking-transport"} For chemical-state augmentation of molecular docking:

1.  optimizer sets are preserved exactly by projection to the molecular core state,

2.  structural-rank transport holds under the declared chemical-rank witness.
:::

::: proof
*Proof.* The first clause is definitional for the projection-based augmented decision object. The second clause is the witness transport theorem specialized to that same augmentation layer. ◻
:::

::: theorem
[]{#thm:conformational-ensemble-docking-transport label="thm:conformational-ensemble-docking-transport"} For a declared conformational ensemble bridge:

1.  ensemble populations normalize to one,

2.  aggregated ensemble utility matches the per-conformer utility law,

3.  induced-fit transitions transport to monotone structural-rank inequalities across conformers.
:::

::: proof
*Proof.* Each statement is the corresponding endpoint projection of the declared ensemble witness package: normalization, aggregation compatibility, and induced-fit rank monotonicity. ◻
:::

::: theorem
[]{#thm:ensemble-one-step-population-transport label="thm:ensemble-one-step-population-transport"} For a stochastic conformational ensemble kernel with nonnegative transition probabilities and row-sum normalization, one-step conformer-population propagation preserves nonnegativity and total mass.
:::

::: proof
*Proof.* Nonnegativity follows termwise from nonnegative factors in the one-step update sum. Total mass preservation follows by finite sum exchange and row-stochastic normalization. ◻
:::

::: theorem
[]{#thm:ensemble-multistep-population-transport label="thm:ensemble-multistep-population-transport"} For every finite horizon $n$, repeated conformer propagation by the stochastic ensemble kernel preserves nonnegativity and total mass. Equivalently, the process-level population map is Markov-consistent across all finite step counts.
:::

::: proof
*Proof.* Induct on the number of steps. The base case is the input distribution hypothesis. The step case applies the one-step nonnegativity and normalization theorems to the previous-step distribution. ◻
:::

::: theorem
[]{#thm:ensemble-statistical-validity-transport label="thm:ensemble-statistical-validity-transport"} Under explicit per-conformer finite-sample population-error budgets, bounded-observable expectations transport with an explicit error envelope, and conformer-ordering decisions are stable whenever true-population margins exceed the corresponding noise budgets.
:::

::: proof
*Proof.* Expectation transport follows from sum-difference expansion, absolute-value summation bounds, and the declared pointwise error budgets. Order stability follows from two-sided pointwise interval bounds plus the stated margin separation. ◻
:::

::: theorem
[]{#thm:chemical-ensemble-transport-binding-specialization label="thm:chemical-ensemble-transport-binding-specialization"} For a concrete molecular binding problem:

1.  chemical-state augmentation preserves optimizer projection to the base docking decision object and transports structural rank under the declared witness,

2.  ensemble aggregation and induced-fit rank transport specialize directly to the same base docking decision problem through a designated baseline conformer.
:::

::: proof
*Proof.* Specialize the chemical transport endpoints to `prob.toDecisionProblem`. Then specialize ensemble utility/rank transport via the baseline-conformer identification hypothesis. ◻
:::

::: theorem
[]{#thm:chemical-component-variation-opt-invariance label="thm:chemical-component-variation-opt-invariance"} At fixed molecular core state, varying protonation, tautomer, ionic environment, solvent mode, or water-bridge flag leaves the optimizer set unchanged in the chemical-augmented docking interface.
:::

::: proof
*Proof.* The chemical-augmented utility is defined by projection to the core state, so all chemical-component-only substitutions are definitionally optimizer-invariant. ◻
:::

::: theorem
[]{#thm:chemical-coupled-sensitivity label="thm:chemical-coupled-sensitivity"} When explicit chemical coupling terms are added to utility, chemical microstate changes at fixed molecular core can alter utility values and, under strict-winner witnesses for two chemical states, force optimizer-set change.
:::

::: proof
*Proof.* The utility-sensitivity clause is a direct algebraic subtraction argument on the coupled utility definition. The optimizer-sensitivity clause uses strict-opt witnesses at the two chemical states to derive a contradiction if optimizer sets were equal. ◻
:::

::: theorem
[]{#thm:unified-chemical-state-dynamics-transport label="thm:unified-chemical-state-dynamics-transport"} For a finite chemical-state dynamics layer carrying joint protonation/tautomer/ionic/solvent/water-bridge state metadata, stationary populations, and pH/ionic-condition windows:

1.  observable docking utility decomposes as core molecular utility plus full chemical coupling contribution at each dynamic chemical state,

2.  pH and ionic-strength window certificates are exported,

3.  expected-utility action separation is transported directly by an explicit margin hypothesis.
:::

::: proof
*Proof.* The decomposition is definitional for the unified chemical-state observable utility map. Window certificates are record fields. Expected-utility separation is the direct margin-transport theorem for the same expected-utility functional. ◻
:::

::: theorem
[]{#thm:chemical-conditioned-mechanism-dynamics label="thm:chemical-conditioned-mechanism-dynamics"} If chemical-state populations are constructed from explicit pH/ionic-conditioned state weights and transition rows are set to that same derived stationary law, then:

1.  the full unified chemical-state transport bundle (window certificates plus observable-utility decomposition plus stationary normalization) is exported,

2.  the stationary distribution satisfies the explicit one-step fixed-point equation for the derived transition kernel.
:::

::: proof
*Proof.* Convert the conditioned mechanism into the unified chemical-state dynamics record and apply the existing unified transport theorem. The fixed-point equation follows by substituting the row-constant transition kernel and using stationary normalization. ◻
:::

::: theorem
[]{#thm:gibbs-conditioned-mechanism-dynamics label="thm:gibbs-conditioned-mechanism-dynamics"} If chemical-state weights are instantiated explicitly as Gibbs factors of pH/ionic-conditioned free-energy surrogates, then:

1.  the induced mechanism stationary population is exactly the normalized Gibbs weight law,

2.  the unified chemical-state transport bundle and stationary fixed-point equation follow for the derived mechanism.
:::

::: proof
*Proof.* Build the conditioned mechanism by taking base weights as Gibbs factors and zero additional linear sensitivities; positivity and normalization are derived from positivity of exponentials and finite-state summation. Then apply the existing mechanism-derived unified dynamics theorem. ◻
:::

::: theorem
[]{#thm:calibrated-biophysical-chemical-separation label="thm:calibrated-biophysical-chemical-separation"} For the explicit calibrated biophysical chemical model, protonation-state changes at fixed molecular core induce concrete utility separation; under strict-winner hypotheses for two actions at the two calibrated chemical states, optimizer sets differ.
:::

::: proof
*Proof.* The calibrated parameterization yields a closed-form protonation contribution gap. Utility separation follows by subtracting the shared core term. Optimizer separation is then a direct specialization of the coupled strict-winner transport theorem. ◻
:::

::: theorem
[]{#thm:chemical-dataset-posterior-separation label="thm:chemical-dataset-posterior-separation"} Given an empirical chemical calibration dataset and confidence level, if empirical protonation-shift mean dominates its finite-sample confidence half-width, then the posterior protonation lower endpoint is strictly positive; under positive action sensitivity, this yields robust protonation utility separation at fixed molecular core for the posterior-mean coupling.
:::

::: proof
*Proof.* The margin condition implies strict positivity of the posterior protonation lower endpoint by direct algebraic rewrite of the posterior definition. Interval transport then gives positivity of the posterior mean protonation scale. Combined with positive action sensitivity and the closed-form protonation contribution delta, this yields strict utility separation. ◻
:::

::: theorem
[]{#thm:chemical-realdata-bias-aware-separation label="thm:chemical-realdata-bias-aware-separation"} If posterior chemical calibration is equipped with an explicit heldout-bias bound and the bias-adjusted protonation lower endpoint remains positive, then protonation-state utility separation still follows at fixed molecular core under positive action sensitivity.
:::

::: proof
*Proof.* Use positivity of the bias-adjusted lower endpoint to recover positivity of the posterior lower endpoint, then apply the existing robust posterior protonation-separation theorem. ◻
:::

::: theorem
[]{#thm:hierarchical-chemical-realdata-separation-rate label="thm:hierarchical-chemical-realdata-separation-rate"} Suppose multiple real-data chemical calibration layers are pooled in a hierarchical posterior package with:

1.  a pooled robust protonation lower endpoint that lower-bounds every dataset-specific robust lower endpoint,

2.  an explicit finite-sample rate lower bound of order $c/\sqrt{N}$ for the pooled endpoint.

Then any strictly positive hierarchical rate constant implies per-dataset robust protonation utility separation at fixed molecular core under positive action sensitivity.
:::

::: proof
*Proof.* Positive rate constant and positive sample size imply positivity of the pooled lower endpoint via the finite-sample lower bound. Monotone transport from pooled lower endpoint to each dataset endpoint yields per-dataset positivity, after which the bias-aware robust chemical separation theorem applies datasetwise. ◻
:::

::: theorem
[]{#thm:hierarchical-chemical-realdata-separation-rate-margin label="thm:hierarchical-chemical-realdata-separation-rate-margin"} In the same hierarchical chemical layer, if the explicit finite-sample rate term is strictly below the selected dataset robust protonation lower endpoint, then robust protonation utility separation follows at that dataset under positive action sensitivity.
:::

::: proof
*Proof.* Rate-term nonnegativity and strict rate-margin separation imply positivity of the dataset robust lower endpoint; the dataset-level bias-aware robust chemical separation theorem then gives the utility separation conclusion. ◻
:::

::: theorem
[]{#thm:hierarchical-chemical-realdata-separation-rate-margin-all-datasets label="thm:hierarchical-chemical-realdata-separation-rate-margin-all-datasets"} If explicit finite-sample rate margins hold at *every* dataset in the hierarchical chemical layer, then robust protonation utility separation holds simultaneously for all datasets (under positive datasetwise action sensitivity).
:::

::: proof
*Proof.* Apply the single-dataset rate-margin theorem pointwise at each dataset index. ◻
:::

::: theorem
[]{#thm:hierarchical-chemical-realdata-separation-rate-constant-margin label="thm:hierarchical-chemical-realdata-separation-rate-constant-margin"} Fix an explicit external upper bound on the pooled posterior rate constant. If the induced bound $$\frac{C_{\mathrm{rate}}}{\sqrt{N_{\mathrm{pool}}}}$$ is strictly below the selected dataset robust protonation lower endpoint, then robust protonation utility separation follows at that dataset under positive action sensitivity.
:::

::: proof
*Proof.* First transport the external constant bound to the realized pooled rate term by dividing through the common square-root denominator, then invoke the existing single-dataset rate-margin separation theorem. ◻
:::

::: theorem
[]{#thm:hierarchical-chemical-realdata-separation-rate-constant-margin-all-datasets label="thm:hierarchical-chemical-realdata-separation-rate-constant-margin-all-datasets"} Under the same shared external upper bound on pooled posterior rate constant, if the corresponding explicit rate-constant margin inequality holds for every dataset, then robust protonation utility separation holds simultaneously across all datasets.
:::

::: proof
*Proof.* Apply the single-dataset rate-constant-margin theorem pointwise across dataset indices. ◻
:::

::: theorem
[]{#thm:absolute-binding-free-energy-closure label="thm:absolute-binding-free-energy-closure"} For an absolute binding free-energy model with explicit standard-state, finite-size, long-range, net-charge, and restraint corrections:

1.  the corrected free energy is exactly base free energy plus total correction,

2.  total correction magnitude is bounded by the summed componentwise correction bounds,

3.  therefore corrected-vs-base absolute discrepancy is bounded by that same summed componentwise bound.
:::

::: proof
*Proof.* Use the correction-stack decomposition theorem and the total correction absolute-value transport theorem (triangle inequality plus per-component absolute bounds), then rewrite corrected minus base by the corrected-value identity. ◻
:::

::: theorem
[]{#thm:absolute-free-energy-protocol-derived-closure label="thm:absolute-free-energy-protocol-derived-closure"} If each absolute free-energy correction term (standard-state, finite-size, long-range, net-charge, restraint) is calibrated by one protocol-derived bound term, then the induced correction stack satisfies the same total absolute correction bound used in absolute free-energy closure: $$\left|\Delta G_{\mathrm{corr,total}}\right| \le \sum_i B_i.$$
:::

::: proof
*Proof.* Map each protocol-calibrated correction term to the corresponding correction-stack field, transport each termwise absolute bound, and apply the existing summed correction-stack absolute-bound theorem. ◻
:::

::: theorem
[]{#thm:kinetic-observable-reporting-endpoints label="thm:kinetic-observable-reporting-endpoints"} For the declared kinetic bridge package, theorem-level outputs include:

1.  an on-rate upper envelope from bounded-acquisition speed,

2.  the residence-time identity $\tau_{\mathrm{res}}=1/k_{\mathrm{off}}$,

3.  pathway-population normalization,

4.  a bundled reporting theorem that returns all three as one certificate.
:::

::: proof
*Proof.* The on-rate inequality follows from the quotient speed theorem under the bridge horizon/rank hypotheses; the residence and normalization identities are direct profile fields; the bundle theorem conjoins those three endpoints. ◻
:::

::: theorem
[]{#thm:kinetic-protocol-measurement-transport label="thm:kinetic-protocol-measurement-transport"} Given measurable kinetic protocol inputs (association/dissociation event counts, observation window, pathway counts) plus an exact-resolution witness, the induced kinetic profile satisfies the same bundled reporting guarantees: on-rate envelope, residence/off-rate identity, and pathway normalization.
:::

::: proof
*Proof.* Construct the kinetic profile from measurable protocol formulas, then apply the protocol-to-bridge transport theorem, which reduces to the base kinetic bundle theorem after record conversion. ◻
:::

::: theorem
[]{#thm:kinetic-confidence-transport label="thm:kinetic-confidence-transport"} Given confidence-profile metadata (nonnegative $z$-score and nonnegative standard-error terms), the kinetic bridge exports interval transport for $k_{\mathrm{on}}$, $k_{\mathrm{off}}$, residence time, and pathway populations. The same interval transport specializes to measurable protocol-instantiated kinetic profiles.
:::

::: proof
*Proof.* Each confidence interval is an immediate linear inequality from nonnegative half-width terms. The protocol specialization follows by rewriting the profile with the measurable protocol-to-profile conversion. ◻
:::

::: theorem
[]{#thm:kinetic-protocol-inference-guarantee label="thm:kinetic-protocol-inference-guarantee"} For the protocol-noise model with explicit absolute-error budgets, threshold decisions for $k_{\mathrm{on}}$ and $k_{\mathrm{off}}$ and pathway-order comparisons are inference-stable whenever true-profile separation margins dominate the corresponding noise budgets.
:::

::: proof
*Proof.* Convert absolute-error hypotheses to two-sided interval bounds, then combine those bounds with the assumed separation inequalities to derive preserved threshold and ordering conclusions for the observed profile. ◻
:::

::: theorem
[]{#thm:kinetic-concentration-identifiability-bridge label="thm:kinetic-concentration-identifiability-bridge"} If protocol-noise inference is embedded in a concentration layer (confidence level plus concentration-event certificate) and an identifiability layer (distinguishability relation plus margin-dominates-noise condition), then threshold/ranking kinetic conclusions transport together with concentration and identifiability certificates in one bundled inference theorem.
:::

::: proof
*Proof.* Instantiate the protocol-noise inference guarantee with the identifiability margin condition to obtain threshold/ranking conclusions. Then append the concentration-event and identifiability certificates from the two layer interfaces. ◻
:::

::: theorem
[]{#thm:kinetic-replicate-identifiability-bundle label="thm:kinetic-replicate-identifiability-bundle"} If concentration-based protocol-noise inference is strengthened by replicate-count certification and systematic-bias-aware pathway margins, then thresholded $k_{\mathrm{on}}$/$k_{\mathrm{off}}$ conclusions and pathway-order inference are preserved together with concentration-event and replicate-count certificates.
:::

::: proof
*Proof.* Convert bias-aware margins to base noise-dominating margins via nonnegativity of systematic-bias bounds, apply the protocol-noise bundled inference theorem, and append concentration/replicate certificates. ◻
:::

::: theorem
[]{#thm:hierarchical-kinetic-replicate-inference-bundle label="thm:hierarchical-kinetic-replicate-inference-bundle"} In a hierarchical multi-dataset extension of replicate-aware kinetic identifiability, assume each dataset carries concentration/noise/replicate structure, and a pooled replicate count supports an explicit finite-sample rate metadata bound. Then for each dataset, thresholded $k_{\mathrm{on}}$/$k_{\mathrm{off}}$ inference and pathway-order inference hold together with concentration-event certificate, replicate-count certificate, and the pooled finite-sample-rate certificate.
:::

::: proof
*Proof.* First strip shared hierarchical bias terms from the hierarchical pathway-margin inequality using bias nonnegativity, yielding the base margin needed for protocol-noise inference. Apply the bundled protocol-noise theorem at the dataset level, then append the dataset concentration and replicate certificates together with the pooled finite-sample-rate metadata field. ◻
:::

::: theorem
[]{#thm:hierarchical-kinetic-replicate-inference-rate-margins label="thm:hierarchical-kinetic-replicate-inference-rate-margins"} If the hierarchical multi-dataset kinetic layer satisfies explicit pooled finite-sample rate margins in the on/off threshold inequalities, then the same per-dataset threshold/ranking conclusions and concentration/replicate/rate certificates follow.
:::

::: proof
*Proof.* Use nonnegativity of the pooled rate term to reduce rate-margin threshold assumptions to the baseline threshold assumptions, then invoke the hierarchical kinetic bundled inference theorem. ◻
:::

::: theorem
[]{#thm:hierarchical-kinetic-replicate-inference-rate-margins-all-datasets label="thm:hierarchical-kinetic-replicate-inference-rate-margins-all-datasets"} If pooled-rate on/off margin inequalities hold at every dataset (for a fixed pathway pair), then the hierarchical kinetic threshold/ranking inference bundle (including concentration, replicate, and pooled-rate certificates) holds simultaneously for all datasets.
:::

::: proof
*Proof.* Apply the single-dataset pooled-rate-margin theorem pointwise in the dataset index. ◻
:::

::: theorem
[]{#thm:hierarchical-kinetic-replicate-inference-rate-constant-margins label="thm:hierarchical-kinetic-replicate-inference-rate-constant-margins"} Fix an explicit external upper bound on the pooled posterior rate constant. If on/off threshold margins are stated using that external bound, $$\frac{C_{\mathrm{rate}}}{\sqrt{N_{\mathrm{rep,pool}}}},$$ then the hierarchical kinetic threshold/ranking inference bundle follows for the selected dataset, together with concentration, replicate, and pooled-rate certificates.
:::

::: proof
*Proof.* Transport the external rate-constant bound to the realized pooled rate term, use the resulting left/right inequality monotonicity to recover the baseline pooled-rate-margin assumptions, then apply the single-dataset pooled-rate-margin inference theorem. ◻
:::

::: theorem
[]{#thm:hierarchical-kinetic-replicate-inference-rate-constant-margins-all-datasets label="thm:hierarchical-kinetic-replicate-inference-rate-constant-margins-all-datasets"} Under the same shared external upper bound on pooled posterior rate constant, if the corresponding on/off rate-constant margin inequalities hold at every dataset (for a fixed pathway pair), then the hierarchical kinetic inference bundle holds simultaneously across all datasets.
:::

::: proof
*Proof.* Apply the single-dataset rate-constant-margin kinetic theorem pointwise across datasets. ◻
:::

::: theorem
[]{#thm:unified-simulator-kinetic-derivation label="thm:unified-simulator-kinetic-derivation"} If one simulator pipeline provides: (i) a microscopic system, (ii) an absolute free-energy model, and (iii) kinetic protocol measurements, then the unified physical model is constructed with kinetic profile derived directly from those protocol measurements, and the full unified thermodynamic/kinetic closure bundle follows for that constructed model.
:::

::: proof
*Proof.* Construct the unified physical model by converting protocol measurements to the kinetic profile record and reusing the same microscopic/free-energy data. The bundled thermodynamic/kinetic conclusions then follow by direct application of the existing unified physical-model theorem. ◻
:::

::: theorem
[]{#thm:unified-thermo-kinetic-physical-model label="thm:unified-thermo-kinetic-physical-model"} For a unified physical docking model carrying one microscopic dynamical system, one absolute free-energy correction model, and one kinetic observable profile, the theorem bundle exports simultaneously:

1.  corrected-free-energy identity and corrected-vs-base discrepancy bound,

2.  residence/off-rate identity and pathway normalization,

3.  diffusion-boundary, hydrodynamic-boundary, and rare-event control certificates.
:::

::: proof
*Proof.* Project the thermodynamic and kinetic clauses from their respective subrecords and conjoin them with the declared physical-control certificates. ◻
:::

::: theorem
[]{#thm:universality-ood-calibration-bounds label="thm:universality-ood-calibration-bounds"} Given a universality layer with dataset-indexed prediction errors, uncertainty radii, and OOD calibration witnesses:

1.  each OOD dataset satisfies the per-dataset prediction-error bound by its uncertainty radius,

2.  the same guarantee is exported uniformly over all OOD datasets.
:::

::: proof
*Proof.* Both clauses are direct projections of the OOD calibration witness family (single-index specialization and universal form). ◻
:::

::: theorem
[]{#thm:ood-transfer-calibration-derived label="thm:ood-transfer-calibration-derived"} If OOD uncertainty radii are defined from explicit target-class base radii plus chemotype/regime transfer inflation terms, and OOD prediction errors satisfy the corresponding componentwise transfer bound, then a uniform OOD error-radius guarantee follows directly for all OOD datasets.
:::

::: proof
*Proof.* Build the universality/OOD layer from the transfer-calibration components by definition of uncertainty radius and nonnegativity transport, then apply the layer's uniform OOD theorem. ◻
:::

::: theorem
[]{#thm:finite-sample-complexity-inversion label="thm:finite-sample-complexity-inversion"} If explicit required sample/replicate-size inequalities imply required-margin thresholds for hierarchical chemical and kinetic layers, and the corresponding pooled finite-sample rate terms are monotone-transported to realized pooled counts, then:

1.  hierarchical chemical protonation-separation guarantees hold datasetwise,

2.  hierarchical kinetic threshold/ranking bundles hold datasetwise with concentration/replicate/rate certificates.
:::

::: proof
*Proof.* For each layer, convert required-size margin assumptions to realized pooled-size margins through the declared monotone rate-term inequality, then apply the existing rate-constant-margin all-dataset theorems. ◻
:::

::: theorem
[]{#thm:hierarchical-required-size-joint-bundle label="thm:hierarchical-required-size-joint-bundle"} If the required-sample and required-replicate margin premises hold simultaneously for the hierarchical chemical and kinetic layers (with the corresponding monotone pooled-rate transports), then both datasetwise conclusions hold simultaneously: robust protonation utility separation and threshold/ranking kinetic inference bundles.
:::

::: proof
*Proof.* Apply the chemical required-sample theorem and kinetic required-replicate theorem under the shared assumptions and return their conjunction. ◻
:::

::: theorem
[]{#thm:finite-sample-complexity-inversion-count-order label="thm:finite-sample-complexity-inversion-count-order"} Assume required sample/replicate counts are explicitly ordered below realized pooled counts: $$N_{\mathrm{req}} \le N_{\mathrm{pool}},\qquad
R_{\mathrm{req}} \le R_{\mathrm{pool}}.$$ Then the required-size monotonicity inequalities needed by hierarchical chemical/kinetic inversion are derived in-repo from these count-order relations and nonnegative posterior-rate metadata, and the full joint chemical+kinetic required-size bundle follows without separate monotonicity certificates.
:::

::: proof
*Proof.* Use monotonicity of $c/\sqrt{n}$ for nonnegative $c$ to derive pooled-to-required rate-term transport from count ordering, then invoke the required-size inversion theorems for chemical and kinetic layers and conjoin their conclusions. ◻
:::

::: theorem
[]{#thm:finite-sample-square-count-inversion label="thm:finite-sample-square-count-inversion"} Suppose explicit complexity bounds of square-count form are available: $$\left(\frac{C_{\mathrm{chem}}}{m_{\mathrm{chem}}}\right)^2 \le N_{\mathrm{req}},
\qquad
\left(\frac{C_{\mathrm{kin}}}{m_{\mathrm{kin}}}\right)^2 \le R_{\mathrm{req}},$$ with positive margins $m_{\mathrm{chem}},m_{\mathrm{kin}}$. Then:

1.  generic in-repo transport derives $C/\sqrt{n}\le m$ from each square-count inequality,

2.  hierarchical chemical and kinetic rate terms are bounded by these derived margins at realized pooled counts,

3.  the full joint chemical+kinetic required-size inversion bundle follows from these derived margin bounds.
:::

::: proof
*Proof.* First apply the square-count-to-rate transport theorem to derive each $C/\sqrt{n}$ margin bound. Then compose with chemical and kinetic count-order monotonicity lemmas to obtain required margin hypotheses. Finally apply the joint count-order required-size inversion theorem. ◻
:::

::: theorem
[]{#thm:explicit-hamiltonian-bath-elimination-langevin-endpoints label="thm:explicit-hamiltonian-bath-elimination-langevin-endpoints"} For an explicit Hamiltonian+bath elimination system carrying concrete Hamiltonian-energy identities, thermostat parameters, and a pointwise microscopic drift equality, the microscopic drift law is exported directly and jointly with the full concrete Langevin endpoint package (unique strong solution, Boltzmann invariance/ergodicity, and strong/weak Euler--Maruyama rate witnesses).
:::

::: proof
*Proof.* Convert the elimination record to the concrete Hamiltonian+bath Langevin system, project the microscopic drift certificate from the explicit equality field, and conjoin it with the previously established concrete Hamiltonian+bath endpoint theorem. ◻
:::

::: theorem
[]{#thm:explicit-molecular-constant-drift-sde-endpoints label="thm:explicit-molecular-constant-drift-sde-endpoints"} For a molecular Langevin class with explicit constant drift $b(x)=x_\star$, the first-principles coefficients are computed in closed form ($L=0$, linear-growth constant $\|x_\star\|$, dissipativity pair $(\lambda,b)=(1,\|x_\star\|)$, and zero strong/weak rate constants), and the induced endpoint package includes unique strong solution plus invariant/ergodic Boltzmann law.
:::

::: proof
*Proof.* Build the first-principles bundle from the explicit constant-drift equalities and nonnegativity lemmas, then apply the generic first-principles-to-endpoints theorem and read off the computed constants by definitional reduction. ◻
:::

::: theorem
[]{#thm:concrete-generator-canonical-path-process-closure label="thm:concrete-generator-canonical-path-process-closure"} Given explicit finite-horizon bridge data together with generator residual nonnegativity, tightness modulus nonnegativity, and contraction-based uniqueness constants, one obtains a canonical continuous path-process construction that simultaneously certifies measurability, canonical continuity, regularity, pathwise uniqueness, and the three concrete generator/tightness/contraction certificates.
:::

::: proof
*Proof.* Map the concrete generator package to the finite-horizon constructive bridge interface, apply the canonical path-process closure theorem from finite-horizon data, and append the three certificate projections supplied by the concrete package. ◻
:::

::: theorem
[]{#thm:qm-workflow-specific-realism-transport-bundle label="thm:qm-workflow-specific-realism-transport-bundle"} If electronic shell/error contributions are decomposed explicitly into workflow-level basis/correlation/sampling components for one designated QM protocol, then the induced method calibration transports to the same composed-Hamiltonian Lipschitz and absolute-error realism bounds in the tight QM-grounded form.
:::

::: proof
*Proof.* Aggregate workflow components into total shell/error intercepts, convert this record to the method-calibration interface, and invoke the existing method-specific QM realism transport theorem. ◻
:::

::: theorem
[]{#thm:reversible-chemical-transition-detailed-balance-bundle label="thm:reversible-chemical-transition-detailed-balance-bundle"} For a finite chemically reversible mechanism defined by a positive stationary population and symmetric flow tensor, the induced transition kernel satisfies normalization, stationary fixed-point transport, and detailed balance; together with pH/ionic window and observable-utility decomposition, these are exported as one unified dynamics bundle.
:::

::: proof
*Proof.* Define the transition kernel as flow divided by stationary mass, prove row-stochasticity and stationary fixed-point identities from flow-row sums and symmetry, prove detailed balance from symmetric-flow transport, then conjoin with the base unified chemical-state observable/window bundle. ◻
:::

::: theorem
[]{#thm:trajectory-absolute-free-energy-correction-closure label="thm:trajectory-absolute-free-energy-correction-closure"} If each correction component in the absolute free-energy stack is calibrated from a trajectory-level finite-sample estimator with explicit $z\cdot\mathrm{SE}/\sqrt{N}+\mathrm{bias}$ bound, then the resulting protocol-derived correction stack satisfies the same total absolute correction bound used by absolute free-energy closure.
:::

::: proof
*Proof.* Convert each trajectory estimator to a protocol-derived correction term via its sample-mean finite-sample inequality, assemble the five-term protocol calibration record, and apply the protocol-calibration total-error theorem. ◻
:::

::: theorem
[]{#thm:quantified-simulator-thermo-kinetic-bundle label="thm:quantified-simulator-thermo-kinetic-bundle"} When one simulator pipeline carries explicit diffusion, hydrodynamic, and rare-event mismatch/tolerance constants with within-tolerance certificates, the same physical simulator object exports both the unified thermodynamic/kinetic closure bundle and the three quantified control inequalities.
:::

::: proof
*Proof.* Convert the quantified pipeline to the base simulator interface, apply the existing unified simulator thermodynamic/kinetic bundle theorem, and append the mismatch-vs-tolerance inequalities from the quantified control record. ◻
:::

::: theorem
[]{#thm:mechanistic-ood-uniform-transfer-bound label="thm:mechanistic-ood-uniform-transfer-bound"} If out-of-distribution error is bounded by base class radius plus class-Lipschitz shift and regime inflation, and mechanistic shift is bounded by a class envelope, then the induced OOD calibration with chemotype inflation yields a uniform OOD uncertainty-radius bound for every OOD dataset.
:::

::: proof
*Proof.* Define chemotype inflation as Lipschitz times class-envelope, transport the mechanistic shift inequality through nonnegative multiplication, lift the resulting bound to the calibration uncertainty radius, and invoke the calibration layer's uniform OOD endpoint. ◻
:::

::: theorem
[]{#thm:model-dependent-rate-term-target-margin label="thm:model-dependent-rate-term-target-margin"} For explicit model constants (mixing time, asymptotic variance, dimension penalty, minimax penalty), if the induced square-count complexity bound holds at required count $N$, then the effective rate term divided by $\sqrt{N}$ is bounded by the target margin.
:::

::: proof
*Proof.* Define the effective constant as the sum of model-dependent components, prove nonnegativity from componentwise nonnegativity, then apply the square-count-to-rate inversion theorem. ◻
:::

::: theorem
[]{#thm:hierarchical-required-size-model-dependent-constants label="thm:hierarchical-required-size-model-dependent-constants"} If hierarchical chemical and kinetic layers admit explicit model-dependent finite-sample laws (mixing/variance/dimension/minimax constants) whose square-count complexity bounds control the corresponding required counts, then the full joint chemical+kinetic required-size conclusion follows with no additional external monotonicity axiom.
:::

::: proof
*Proof.* Instantiate chemical and kinetic square-count premises from the two model-dependent laws, transport each to the required $c/\sqrt{n}$ margin form, and apply the existing square-count joint hierarchical inversion theorem. ◻
:::

::: theorem
[]{#thm:hamiltonian-finite-difference-drift-derivation-endpoints label="thm:hamiltonian-finite-difference-drift-derivation-endpoints"} When Langevin drift is constructed in-model from explicit finite-difference Hamiltonian elimination with thermostat scaling (instead of provided as a standalone drift-equality witness), the same concrete microscopic drift identity and full Langevin endpoint package are exported jointly.
:::

::: proof
*Proof.* Build the concrete Hamiltonian+bath system directly from the finite-difference drift constructor, then apply the existing concrete Hamiltonian+bath endpoint theorem and conjoin it with the definitional drift identity. ◻
:::

::: theorem
[]{#thm:realistic-molecular-finite-difference-sde-endpoints label="thm:realistic-molecular-finite-difference-sde-endpoints"} For the realistic molecular drift class obtained by combining finite-difference Hamiltonian elimination with an explicit confining pullback term, computed total Lipschitz/growth constants are exported together with dissipativity/rate constants and the full strong-solution/invariant/ergodic endpoint bundle.
:::

::: proof
*Proof.* Package the explicit drift inequalities into the first-principles interface using state-distance/state-norm definitions on full molecular coordinates, then apply the first-principles Langevin endpoint theorem. ◻
:::

::: theorem
[]{#thm:generator-coefficients-canonical-regularity-closure label="thm:generator-coefficients-canonical-regularity-closure"} If generator residual and tightness profiles are given by explicit decay formulas with nonnegative constants and contraction constant $<1$, then canonical path-process regularity/continuity/uniqueness closure follows constructively from these coefficient profiles.
:::

::: proof
*Proof.* Instantiate the concrete generator path-process data from the coefficient formulas, prove nonnegativity of the induced profiles pointwise, and invoke the existing concrete-generator canonical closure theorem. ◻
:::

::: theorem
[]{#thm:concrete-qm-workflow-error-analysis-transport label="thm:concrete-qm-workflow-error-analysis-transport"} Given explicit basis/correlation/sampling component decompositions of electronic shell/error terms for one concrete QM workflow, absolute-value budgets derived from those components induce the same workflow-specific QM realism transport bounds.
:::

::: proof
*Proof.* Map decomposition components to nonnegative absolute budgets, prove each base shell/error term is bounded by the corresponding budget sum, convert to workflow calibration, then apply workflow-specific QM realism transport. ◻
:::

::: theorem
[]{#thm:barrier-crossing-reversible-chemical-dynamics label="thm:barrier-crossing-reversible-chemical-dynamics"} For finite chemical states with Gibbs stationary law and Kramers/Eyring-style off-diagonal barrier flows (plus diagonal stationarity correction), transition rows normalize to one and detailed balance is derived for the induced reversible kernel.
:::

::: proof
*Proof.* Construct symmetric flows from barrier factors, add diagonal correction to enforce row-sum stationarity, then transport to the reversible mechanism interface and apply row-normalization/detailed-balance projections. ◻
:::

::: theorem
[]{#thm:mixing-autocorrelation-trajectory-correction-total-error label="thm:mixing-autocorrelation-trajectory-correction-total-error"} If each absolute free-energy correction component is calibrated by a trajectory law carrying explicit mixing/autocorrelation finite-sample error inequality, the assembled correction stack satisfies the same total absolute correction bound used in absolute free-energy closure.
:::

::: proof
*Proof.* Convert each mixing-law trajectory record into a trajectory correction estimator, assemble the five-term trajectory calibration stack, and apply the trajectory absolute free-energy total-error theorem. ◻
:::

::: theorem
[]{#thm:unified-simulator-error-analysis-controlled-thermo-kinetic label="thm:unified-simulator-error-analysis-controlled-thermo-kinetic"} If diffusion, hydrodynamic, and rare-event mismatches are derived as summed component errors from one unified simulator error-analysis record and each summed mismatch satisfies its tolerance budget, then the quantified simulator thermo/kinetic bundle follows together with all three mismatch-tolerance inequalities.
:::

::: proof
*Proof.* Define mismatch aggregates from component error terms, package them into quantified boundary controls, instantiate the quantified simulator pipeline, and apply its bundled thermo/kinetic theorem. ◻
:::

::: theorem
[]{#thm:descriptor-calibrated-mechanistic-ood-transfer label="thm:descriptor-calibrated-mechanistic-ood-transfer"} For finite descriptor vectors with class-envelope shift controls and a prospective calibration-protocol witness, mechanistic descriptor shifts induce a uniform OOD uncertainty-radius transfer bound for all OOD datasets.
:::

::: proof
*Proof.* Define mechanistic shift as descriptor $L^1$ distance, transport shift-to-envelope and mechanistic error bounds into the mechanistic OOD interface, then apply the uniform OOD transfer theorem and conjoin the prospective protocol certificate. ◻
:::

::: theorem
[]{#thm:model-dependent-minimax-optimality-bundle label="thm:model-dependent-minimax-optimality-bundle"} For model-dependent finite-sample laws augmented with explicit minimax lower/upper count constants, one obtains simultaneously: target-margin rate control, lower/upper count sandwich bounds, and a near-optimality ratio bounded below by $1$.
:::

::: proof
*Proof.* Project the rate-term margin theorem from the model law, append minimax lower/upper count inequalities from the optimality record, and derive the ratio lower bound from ordered positive minimax constants. ◻
:::

::: theorem
[]{#thm:constructive-stochastic-multipole-scope-discharge label="thm:constructive-stochastic-multipole-scope-discharge"} If canonical process regularity is provided constructively from finite-horizon bridge data and higher-order multipole energy terms carry explicit shell-Lipschitz bounds, then both previously declared scope gaps are discharged together as one bundled theorem endpoint.
:::

::: proof
*Proof.* Apply the constructive finite-horizon-to-canonical-path-process closure theorem for process regularity, then conjoin the multipole shell-Lipschitz transport certificate from the same closure record. ◻
:::

::: theorem
[]{#thm:ito-wiener-filtration-langevin-endpoints label="thm:ito-wiener-filtration-langevin-endpoints"} For an overdamped Langevin model equipped with explicit Wiener process and filtration data, adaptedness/Ito-integrability/ Ito-equation certificates, generator representation, and martingale well-posedness, the standard strong-solution, invariant-measure, and ergodicity endpoints are exported jointly with the stochastic-basis certificates.
:::

::: proof
*Proof.* Project the filtration and Wiener certificates directly, then discharge strong-solution uniqueness from the representative-path witness and append invariant/ergodic transport from the supplied Boltzmann-density fields. ◻
:::

::: theorem
[]{#thm:hamiltonian-mori-zwanzig-h-zero-limit-endpoints label="thm:hamiltonian-mori-zwanzig-h-zero-limit-endpoints"} Given finite-difference Hamiltonian elimination data with an explicit vanishing step sequence, pointwise drift convergence, and a Mori-Zwanzig limiting drift identity, the limiting Langevin model inherits strong-solution, invariance, ergodicity, and martingale well-posedness endpoints.
:::

::: proof
*Proof.* Use the declared $h\to0$ and drift-convergence hypotheses to identify the limiting drift map, then project strong-solution/invariance/ergodicity/martingale endpoints from the supplied Ito--Fokker--Planck--Harris closure of the limit model. ◻
:::

::: theorem
[]{#thm:forcefield-derived-realistic-sde-endpoints label="thm:forcefield-derived-realistic-sde-endpoints"} If drift differences are controlled by full-state force-field energy differences with nonnegative coupling and closure distance is instantiated by molecular $L^1$ distance, then the drift Lipschitz constant is derived as coupling times full-state shell constant, and the full endpoint bundle follows.
:::

::: proof
*Proof.* Compose the drift-energy control inequality with the concrete full-state shell Lipschitz bound, transport through the closure distance identification, and combine with first-principles endpoint projection from the closure bundle. ◻
:::

::: theorem
[]{#thm:generator-pde-estimate-canonical-regularity-closure label="thm:generator-pde-estimate-canonical-regularity-closure"} Explicit generator-resolvent, Kolmogorov tightness, and Harnack contraction estimate records induce coefficient profiles that constructively discharge canonical path-process measurability/continuity/regularity/uniqueness.
:::

::: proof
*Proof.* Map the PDE/operator estimate constants into the generator-coefficient closure interface and apply the previously proved coefficient-to-canonical-regularity theorem. ◻
:::

::: theorem
[]{#thm:qm-workflow-transport-benchmark-summary label="thm:qm-workflow-transport-benchmark-summary"} Method-level benchmark statistics (mean/std-error/bias budgets) for shell/error components induce a workflow calibration that transports to the same QM realism shell/error bounds, while retaining an explicit positive benchmark-count witness.
:::

::: proof
*Proof.* Convert benchmark mean/std-error/bias fields into nonnegative basis/correlation/sampling component budgets, instantiate workflow calibration inequalities from empirical bounds, and invoke workflow-specific QM transport. ◻
:::

::: theorem
[]{#thm:potential-landscape-barrier-kinetics-bundle label="thm:potential-landscape-barrier-kinetics-bundle"} Symmetric barrier energies and symmetric friction/diffusion prefactors with row-mass control induce scaled Kramers/Eyring attempt frequencies, Gibbs-consistent stationary law, normalized transition rows, and detailed balance.
:::

::: proof
*Proof.* Build attempt frequencies from scaled prefactor averages, convert to the barrier-crossing reversible mechanism interface, then project stationary-law, row-normalization, and detailed-balance conclusions. ◻
:::

::: theorem
[]{#thm:spectral-gap-trajectory-concentration-bundle label="thm:spectral-gap-trajectory-concentration-bundle"} From spectral-gap and mixing-time hypotheses, one obtains an explicit mixing-time upper envelope and a trajectory finite-sample correction bound after conversion to the mixing/autocorrelation estimator interface.
:::

::: proof
*Proof.* Apply the spectral-gap mixing-time bound theorem and transport the declared spectral-gap concentration inequality through the conversion to trajectory correction estimators. ◻
:::

::: theorem
[]{#thm:spectral-gap-absolute-free-energy-total-error label="thm:spectral-gap-absolute-free-energy-total-error"} If all absolute free-energy correction terms are calibrated by spectral-gap trajectory concentration records, then the protocol total correction satisfies the same absolute total-component error bound.
:::

::: proof
*Proof.* Convert each spectral-gap trajectory record to mixing/autocorrelation form, assemble the five-component absolute free-energy calibration, and apply the trajectory absolute correction total-error theorem. ◻
:::

::: theorem
[]{#thm:integrator-error-stack-unified-thermo-kinetic label="thm:integrator-error-stack-unified-thermo-kinetic"} For one shared simulator, if global diffusion/hydrodynamic/rare-event mismatches are derived from local truncation plus model-bias stacks and each global mismatch is within tolerance, then the quantified thermo/kinetic bundle follows.
:::

::: proof
*Proof.* Map integrator/model stack terms into unified simulator component mismatches, instantiate the quantified control object, and apply the unified simulator controlled thermo/kinetic theorem. ◻
:::

::: theorem
[]{#thm:learned-descriptor-ood-generalization-transfer label="thm:learned-descriptor-ood-generalization-transfer"} With learned source/target descriptor maps, class-envelope shift control, and a prospective calibration protocol proving OOD generalization inequalities, uniform uncertainty-radius OOD transfer follows across all OOD datasets.
:::

::: proof
*Proof.* Convert learned descriptor generalization data into the descriptor-calibrated mechanistic OOD interface and apply the descriptor-calibrated uniform transfer theorem. ◻
:::

::: theorem
[]{#thm:estimator-minimax-derivation-bundle label="thm:estimator-minimax-derivation-bundle"} Combining model-dependent finite-sample constants with explicit minimax lower-bound theory and matching estimator achievability yields the near-optimality count sandwich, rate-to-margin guarantee, and target-achievement guarantee at required count.
:::

::: proof
*Proof.* Instantiate the model-dependent minimax law from lower/upper count derivation fields, apply the minimax optimality bundle theorem, and append the estimator target-achievement inequality. ◻
:::

::: theorem
[]{#thm:extended-physical-model-interface-scope-bundle label="thm:extended-physical-model-interface-scope-bundle"} Richer physical encoding transport, open-system quantum channel modeling, and noisy/partial observation channel modeling are packaged as one explicit witness-interface endpoint.
:::

::: proof
*Proof.* Conjoin the three witness certificates from the extension-interface layer. ◻
:::

::: theorem
[]{#thm:preregistered-prospective-beats-strong-baselines label="thm:preregistered-prospective-beats-strong-baselines"} For a pre-registered prospective benchmark package carrying blinded empirical closure and one unified thermo/kinetic physical model, if strict wins are supplied against each declared strong baseline on total calibration error, ranking loss, and kinetics/free-energy consistency gap, then all of those superiority clauses are exported together with exact thermo/kinetic consistency and calibration/predictive coverage certificates.
:::

::: proof
*Proof.* Combine prospective empirical-closure transport with unified thermo/kinetic bundle transport, then append the declared strict baseline-comparison inequalities and the derived zero consistency-gap identity. ◻
:::

::: theorem
[]{#thm:independent-replication-outside-team-bundle label="thm:independent-replication-outside-team-bundle"} If an outside-team replication run matches protocol and instantiates the same pre-registered prospective strong-baseline superiority interface, then independent-team status, protocol-match status, and reproduced strict superiority over strong baselines are all exported in one theorem bundle.
:::

::: proof
*Proof.* Apply the pre-registered superiority bundle theorem to the replication benchmark object, then conjoin outside-team and protocol-match certificates. ◻
:::

::: theorem
[]{#thm:downstream-campaign-win-bundle label="thm:downstream-campaign-win-bundle"} A downstream campaign record with strict baseline improvements on hit identification, triage quality, and campaign efficiency yields all three strict downstream-win inequalities as one bundled endpoint.
:::

::: proof
*Proof.* Conjoin the three declared strict-improvement fields from the downstream campaign record. ◻
:::

::: theorem
[]{#thm:external-validation-threeway-integration-bundle label="thm:external-validation-threeway-integration-bundle"} Combining (1) a pre-registered prospective strong-baseline superiority package, (2) an independent outside-team replication package, and (3) a downstream campaign-win package yields:

1.  the primary validation core (pre-registration plus strong-baseline superiority plus independent replication),

2.  the downstream strict-win triple (hit ID, triage quality, campaign efficiency),

3.  and negation of the declared credible-dismissal-by-core-gap predicate.

In particular, under this formal criterion, landing (1)+(2) blocks credible dismissal-by-core-gap arguments.
:::

::: proof
*Proof.* Extract pre-registered superiority clauses and replication clauses from their bundled theorems, assemble the primary validation core proposition, import downstream strict wins, and discharge the no-credible-dismissal conclusion by contradiction against the core-gap definition. ◻
:::

::: theorem
[]{#thm:constructive-scope-closure-of-extension-interfaces label="thm:constructive-scope-closure-of-extension-interfaces"} The previously witness-only extension scope is discharged constructively: one measurable continuous-state encoding bridge, one explicit open-system quantum-channel kernel package, and one explicit noisy/partial observation channel package jointly induce the same richer-encoding/open-system/noisy-observation endpoint triple.
:::

::: proof
*Proof.* Package the constructive encoding/channel layers into the extension-interface closure object, convert it to the witness-layer object, and apply the original extension-scope bundled theorem. ◻
:::

::: theorem
[]{#thm:fixed-contract-preregistered-benchmark-bundle label="thm:fixed-contract-preregistered-benchmark-bundle"} If a prospective benchmark is bound to a fixed pre-registered contract (metrics, datasets/splits, blind rule, baseline roster) with signed artifacts and strict calibration/ranking wins over the baseline table, then it converts to the pre-registered strong-baseline superiority interface, with consistency superiority inherited from exact zero kinetic/free-energy consistency gap.
:::

::: proof
*Proof.* Convert the contract-bound result package to the pre-registered benchmark object, apply the pre-registered superiority bundle theorem, and project protocol closure, strict baseline wins, and zero consistency-gap identity. ◻
:::

::: theorem
[]{#thm:independent-replication-provenance-bundle label="thm:independent-replication-provenance-bundle"} An independent replication provenance record with distinct primary-vs-replication team IDs, distinct compute IDs, matched protocol fingerprint, and signed execution artifacts yields one bundled endpoint containing independent-execution and signed protocol-match certificates.
:::

::: proof
*Proof.* Conjoin the team/compute-separation fields with protocol-fingerprint equality and signed-artifact verification fields. ◻
:::

::: theorem
[]{#thm:downstream-causal-quality-bundle label="thm:downstream-causal-quality-bundle"} If randomized assignment, blinded outcome assessment, and confounder-balance checks hold for a campaign, and model metrics strictly exceed baseline on hit ID, triage quality, and campaign efficiency, then causal-isolation and strict downstream-win triples are exported together.
:::

::: proof
*Proof.* Conjoin causal-control certificates, map to downstream-win record, and apply the downstream campaign win theorem. ◻
:::

::: theorem
[]{#thm:concrete-external-validation-threeway-bundle label="thm:concrete-external-validation-threeway-bundle"} The concrete artifact layer (fixed contract + signed primary results + signed independent replication provenance + concrete causal campaign evidence) instantiates the integrated three-way external-validation theorem, yielding the primary validation core, strict downstream-win triple, and negation of credible-dismissal-by-core-gap.
:::

::: proof
*Proof.* Instantiate the concrete external-validation data object from concrete artifact definitions and invoke the integrated external-validation three-way theorem. ◻
:::

::: theorem
[]{#thm:concrete-external-validation-not-credibly-dismissible label="thm:concrete-external-validation-not-credibly-dismissible"} Under the concrete fixed-contract/independent-replication/campaign artifact instantiation, credible dismissal by the declared validation-core-gap predicate is impossible.
:::

::: proof
*Proof.* Project the third clause of the concrete three-way external-validation instantiation theorem. ◻
:::

::: theorem
[]{#thm:langevin-measure-theoretic-endpoint-bundle label="thm:langevin-measure-theoretic-endpoint-bundle"} Given a measure-theoretic Langevin path-law bundle (probability measure on trajectory space, measurable time evaluations, pathwise drift equation, measure-level stationarity, and singleton-mass ergodicity), one recovers the legacy endpoint triple: unique strong solution, invariant measure, and ergodicity.
:::

::: proof
*Proof.* Select the representative sample path from the measure-theoretic process law, transport measure-level stationarity to singleton-density invariance, and transport singleton-mass ergodicity to the legacy ergodicity predicate. ◻
:::

::: theorem
[]{#thm:constructive-ito-wiener-derived-endpoints label="thm:constructive-ito-wiener-derived-endpoints"} If the Ito/Wiener layer is supplied by explicit filtration/process objects, explicit second-moment functions, explicit Ito-integral path objects, and explicit martingale witnesses, then adaptedness, Ito well-definedness, square-integrability clauses, and martingale well-posedness are derived and the full legacy Ito/Wiener endpoint bundle follows.
:::

::: proof
*Proof.* Define each legacy proposition as a property extracted from the explicit constructive objects, convert to the legacy Ito/Wiener structure, and apply the existing Ito/Wiener endpoint theorem. ◻
:::

::: theorem
[]{#thm:derived-generator-pde-operator-closure label="thm:derived-generator-pde-operator-closure"} When resolvent/tightness/contraction profiles are given explicitly from coefficient formulas, the generator-resolvent, Kolmogorov-tightness, and Harnack-contraction proposition slots are discharged constructively, and canonical path-process regularity closure is recovered.
:::

::: proof
*Proof.* Convert explicit profile formulas into nonnegativity certificates, package them into the legacy PDE-estimate interface, then invoke the canonical regularity closure theorem. ◻
:::

::: theorem
[]{#thm:microscopic-extension-interface-scope-bundle label="thm:microscopic-extension-interface-scope-bundle"} If richer measurable encoding transport is given together with microscopic open-system channel derivation data and microscopic noisy-observation derivation data, then the corresponding richer/open-system/noisy scope triple is discharged as a bundled endpoint.
:::

::: proof
*Proof.* Use constructive certificates for richer encoding, convert microscopic open-system data to the concrete channel layer, project noisy-observation transport from the microscopic layer, and conjoin the three results. ◻
:::

::: theorem
[]{#thm:numerical-stack-derived-simulator-control-flags label="thm:numerical-stack-derived-simulator-control-flags"} For the integrator/model-bias numerical stack, top-level simulator control flags (diffusion, hydrodynamic, rare-event) are not free proposition placeholders: they are definitionally equal to explicit mismatch-vs-tolerance inequalities derived from integration-step and rare-window budgets.
:::

::: proof
*Proof.* Unfold the full conversion chain from integrator stack to unified docking model and simplify definitions of mismatches/tolerances and control predicates. ◻
:::

::: theorem
[]{#thm:attested-concrete-external-validation-threeway-bundle label="thm:attested-concrete-external-validation-threeway-bundle"} The attested concrete external-validation layer (immutable manifests, verifier-checked signatures, lock-time pre-registration, attested independent replication provenance, and measured downstream causal diagnostics) instantiates the integrated three-way external-validation theorem, yielding the primary validation core, strict downstream-win triple, and negation of credible-dismissal-by-core-gap.
:::

::: proof
*Proof.* Build the attested concrete evidence object, convert it to the legacy external-validation interface, and apply the integrated three-way theorem. ◻
:::

::: theorem
[]{#thm:attested-concrete-external-validation-not-credibly-dismissible label="thm:attested-concrete-external-validation-not-credibly-dismissible"} Under the attested concrete external-validation instantiation, credible dismissal by the declared validation-core-gap predicate is impossible.
:::

::: proof
*Proof.* Project the third clause of the attested concrete three-way external-validation theorem. ◻
:::

::: theorem
[]{#thm:attested-concrete-store-backed-artifact-bundle label="thm:attested-concrete-store-backed-artifact-bundle"} All concrete attested external-validation artifacts (pre-registration, primary results, protocol audit, protocol/execution logs, team identity artifacts, and compute provenance artifacts) are shown to be immutable-store-backed: each manifest URI resolves to store payload data whose digest matches the manifest hash.
:::

::: proof
*Proof.* Instantiate one concrete immutable artifact store and discharge each manifest-level URI-fetch plus digest-equality witness, then conjoin the ten artifact clauses. ◻
:::

::: theorem
[]{#thm:store-backed-attested-concrete-external-validation-threeway-bundle label="thm:store-backed-attested-concrete-external-validation-threeway-bundle"} The attested concrete external-validation three-way bundle remains valid under explicit immutable-store backing: the three-way no-credible-dismissal conclusion is proved together with concrete store-backed artifact-ingestion witnesses.
:::

::: proof
*Proof.* Conjoin the attested concrete three-way external-validation theorem with the concrete store-backed artifact bundle. ◻
:::

::: theorem
[]{#thm:store-backed-attested-concrete-external-validation-not-credibly-dismissible label="thm:store-backed-attested-concrete-external-validation-not-credibly-dismissible"} Under the store-backed attested concrete external-validation instantiation, credible dismissal by the declared validation-core-gap predicate remains impossible.
:::

::: proof
*Proof.* Project the no-credible-dismissal clause from the store-backed attested concrete three-way theorem. ◻
:::

::: theorem
[]{#thm:md-physical-utility-interface label="thm:md-physical-utility-interface"} For every molecular binding instance, the exact docking decision object is constructed with exactly the same utility map as the physical binding model.
:::

::: proof
*Proof.* This is definitional unfolding of the docking decision-object constructor. ◻
:::

::: theorem
[]{#thm:md-native-coordinate-rank-identity label="thm:md-native-coordinate-rank-identity"} For every molecular binding instance, exact docking structural rank is exactly the cardinality of the decision-relevant coordinates in the native molecular coordinate interface.
:::

::: proof
*Proof.* Apply the imported molecular-rank identity theorem for the exact docking decision object. ◻
:::

::: theorem
[]{#thm:md-physical-3n-minus-k-budget label="thm:md-physical-3n-minus-k-budget"} If cutoff/locality analysis shows that at most $N-k$ protein atoms remain decision-relevant, then $$\mathrm{srank}(D_{\mathrm{dock}}) \le 3(N-k) + 3N_{\mathrm{ligand}}.$$ Thus the rank budget is physically tied to retained coordinate degrees of freedom, not to a binary output label count.
:::

::: proof
*Proof.* Combine the existing molecular cutoff-rank theorem with the declared bound on relevant-atom count. ◻
:::

::: theorem
[]{#thm:md-binary-summary-rank-monotonicity label="thm:md-binary-summary-rank-monotonicity"} For any binary summary of exact docking optimizer sets (e.g., bind/not-bind style summaries), structural rank of the summary problem is at most structural rank of the full exact pose-selection docking problem.
:::

::: proof
*Proof.* Instantiate the generic optimizer-summary structural-rank monotonicity theorem at the exact docking decision object. ◻
:::

::: theorem
[]{#thm:equilibrium-kd-driving-energy-floor label="thm:equilibrium-kd-driving-energy-floor"} Define the equilibrium dissociation proxy by $$K_d^{\mathrm{eq}} = \exp\!\left(-\frac{\Delta G_{\mathrm{drive}}}{k_B T}\right).$$ If $\Delta G_{\mathrm{drive}} \ge r\,k_B T\ln 2$, then $$K_d^{\mathrm{eq}} \le \exp(-r\ln 2).$$
:::

::: proof
*Proof.* Divide the driving-energy lower bound by positive $k_BT$, negate both sides, and apply monotonicity of the exponential map. ◻
:::

::: theorem
[]{#thm:md-detailed-balance-equilibrium-pathratio label="thm:md-detailed-balance-equilibrium-pathratio"} For an exact docking kernel satisfying detailed balance against its Boltzmann witness law, forward and reverse stationary finite-path weights are equal: $$\frac{P_{\mathrm f}}{P_{\mathrm r}} = 1.$$
:::

::: proof
*Proof.* Apply the detailed-balance stationary-edge-flow symmetry theorem and then the quotient-trajectory equilibrium Crooks corollary. ◻
:::

::: theorem
[]{#thm:md-resolver-free-equilibrium-bundle label="thm:md-resolver-free-equilibrium-bundle"} For exact docking, the calibrated binding free-energy floor and equilibrium detailed-balance path-ratio unity are proved together as one bundle:

1.  the exact-docking free-energy floor in thermal-bit units,

2.  equilibrium forward/reverse path-ratio identity $P_{\mathrm f}/P_{\mathrm r}=1$.
:::

::: proof
*Proof.* Conjoin the docking macrostate free-energy floor theorem with the docking-detailed-balance path-ratio theorem. ◻
:::

::: theorem
[]{#thm:md-equilibrium-kd-prediction-from-rank-lb label="thm:md-equilibrium-kd-prediction-from-rank-lb"} If an independent argument certifies $r \le \mathrm{srank}(D_{\mathrm{dock}})$ and the calibrated driving-energy witness dominates exact-resolution witness cost, then $$K_d^{\mathrm{eq}} \le \exp(-r\ln 2).$$ This yields a falsifiable affinity prediction with rank fixed independently of observed $\Delta G$.
:::

::: proof
*Proof.* Apply the docking rank-lower-bound free-energy floor and then the generic equilibrium $K_d$ upper-bound theorem from a driving-energy floor. ◻
:::

::: theorem
[]{#thm:md-necessary-contact-shell-budget-from-rank-lb label="thm:md-necessary-contact-shell-budget-from-rank-lb"} In the one-hop geometry-contact regime, if an independent argument certifies $r \le \mathrm{srank}(D_{\mathrm{dock}})$, then geometry budgets must satisfy $$r \le 3P + 3K + 3N_{\mathrm{ligand}},$$ where $P$ and $K$ bound active-pocket and outer-shell cardinalities.
:::

::: proof
*Proof.* Compose the independent rank lower bound with the existing geometric-contact-shell bounded-regime upper bound on docking structural rank. ◻
:::

::: theorem
[]{#thm:md-independent-rank-risky-prediction-bundle label="thm:md-independent-rank-risky-prediction-bundle"} Under one independently certified rank lower bound $r \le \mathrm{srank}(D_{\mathrm{dock}})$, the theory simultaneously yields:

1.  an equilibrium affinity prediction $K_d^{\mathrm{eq}} \le \exp(-r\ln 2)$,

2.  a necessary contact-budget inequality $r \le 3P + 3K + 3N_{\mathrm{ligand}}$ in the one-hop geometry-contact regime.
:::

::: proof
*Proof.* Conjoin the docking equilibrium-$K_d$ prediction theorem and the contact-shell-budget necessity theorem under the shared independent rank-lower-bound witness. ◻
:::

::: theorem
[]{#thm:md-exact-lj-physics-witness-bundle label="thm:md-exact-lj-physics-witness-bundle"} Under an explicit exact Lennard--Jones utility model with finite-state strict-winner witnesses and a large-cutoff tail certificate, the theorem chain derives:

1.  global strict-optimality witnesses,

2.  cutoff-bounded perturbation witnesses,

3.  the physical docking rank bound $\mathrm{srank}(D_{\mathrm{dock}}) \le 3N_{\mathrm{rel}} + 3N_{\mathrm{ligand}}$.
:::

::: proof
*Proof.* Instantiate the exact-LJ bounded-potential theorem, derive cutoff-bounded perturbation control from the large-cutoff criterion, and compose with the existing molecular docking rank theorem. ◻
:::

::: theorem
[]{#thm:md-concrete-physics-witness-discharge-bundle label="thm:md-concrete-physics-witness-discharge-bundle"} From one per-system certificate package (strict-winner witness family, cutoff/tail-force certificate, excluded-atom count bound, and one-hop pocket/shell coverage bounds), the abstract docking assumptions are discharged and both physical rank budgets are obtained: $$\mathrm{srank}(D_{\mathrm{dock}}) \le 3(N-k)+3N_{\mathrm{ligand}},\qquad
\mathrm{srank}(D_{\mathrm{dock}}) \le 3P+3K+3N_{\mathrm{ligand}}.$$
:::

::: proof
*Proof.* Project strict-optimality and cutoff-boundedness from the concrete certificate object, then apply the previously proved excluded-atom and contact-shell rank-bound theorems. ◻
:::

::: theorem
[]{#thm:partition-ratio-driving-floor-with-correction-margin label="thm:partition-ratio-driving-floor-with-correction-margin"} Let driving free energy be defined in measurement-free form from partition ratio plus explicit correction stack. If the partition-ratio term exceeds the thermal-bit floor by at least the full correction-budget radius, then the corrected driving-energy witness still satisfies the rank-calibrated thermal-bit floor.
:::

::: proof
*Proof.* Use the correction-stack absolute bound to lower-bound corrected driving free energy by "nominal minus total component bound," then apply the assumed margin inequality. ◻
:::

::: theorem
[]{#thm:equilibrium-kd-bound-from-partition-ratio-correction-chain label="thm:equilibrium-kd-bound-from-partition-ratio-correction-chain"} Combining the previous partition-ratio floor-with-margin theorem with the exponential affinity map yields $$K_d^{\mathrm{eq}} \le \exp(-r\ln 2).$$ Thus the affinity bound is obtained from a measurement-free partition/correction chain rather than post-fit from observed affinity.
:::

::: proof
*Proof.* Apply the driving-floor-to-$K_d$ monotonicity theorem to the corrected partition-ratio driving-energy witness. ◻
:::

::: theorem
[]{#thm:md-rank-kd-bound-from-partition-chain label="thm:md-rank-kd-bound-from-partition-chain"} For exact docking, if the partition-ratio nominal driving free energy dominates the exact-resolution witness by a margin at least equal to the certified correction budget, then the same corrected witness can be used as the exact $\Delta G_{\mathrm{drive}}$ input in the docking rank-based equilibrium affinity theorem, yielding $$K_d^{\mathrm{eq}} \le \exp(-r\ln 2)$$ for any independent rank lower bound $r \le \mathrm{srank}(D_{\mathrm{dock}})$.
:::

::: proof
*Proof.* Convert the nominal partition-ratio margin into the exact drive-witness domination inequality via the correction lower envelope, then invoke the existing docking rank-to-$K_d$ theorem. ◻
:::

::: theorem
[]{#thm:md-independent-srank-interval-certificates label="thm:md-independent-srank-interval-certificates"} If one coordinate set is independently certified relevant and another independently certified sufficient, then exact docking rank is bracketed without any affinity input: $$r_{\mathrm{lower}} \le \mathrm{srank}(D_{\mathrm{dock}}) \le r_{\mathrm{upper}}.$$
:::

::: proof
*Proof.* Lower bound follows from relevance-subset cardinality, upper bound from sufficiency-cardinality. ◻
:::

::: theorem
[]{#thm:md-independent-certificate-risky-prediction-bundle label="thm:md-independent-certificate-risky-prediction-bundle"} Using the lower endpoint of the independent rank interval (relevance-certified coordinate count) as the rank witness, the theory yields both:

1.  an equilibrium affinity upper bound,

2.  a necessary one-hop contact-shell geometry budget.

This rank witness is computed structurally, not post-fit from affinity.
:::

::: proof
*Proof.* Convert the relevance certificate to a rank lower bound and apply the previously proved independent-rank empirical prediction bundle. ◻
:::

::: theorem
[]{#thm:md-falsification-not-falsified-iff label="thm:md-falsification-not-falsified-iff"} For the pre-registered rank-threshold protocol, "not falsified" is equivalent to jointly meeting both hard thresholds: $$K_d^{\mathrm{obs}} \le K_d^{\mathrm{pred,upper}},\qquad
r_{\mathrm{required}} \le B_{\mathrm{contact}}^{\mathrm{obs}}.$$
:::

::: proof
*Proof.* Unfold the protocol fail predicate (disjunction of threshold violations) and apply order duality for strict/non-strict inequalities. ◻
:::

::: theorem
[]{#thm:md-preregistered-rank-protocol-soundness label="thm:md-preregistered-rank-protocol-soundness"} When generated from an independently fixed rank lower bound, the pre-registered protocol exports:

1.  protocol lock witness,

2.  predicted equilibrium affinity upper bound,

3.  required minimum contact-budget threshold.
:::

::: proof
*Proof.* Instantiate the protocol object at rank witness $r$, then transport the rank-based affinity/contact inequalities to its threshold fields. ◻
:::

::: theorem
[]{#thm:finite-sample-upper-violation-implies-true-violation label="thm:finite-sample-upper-violation-implies-true-violation"} For a trajectory-based estimator with certified finite-sample error radius, if observed value exceeds a claimed upper threshold by more than that radius, then the true target exceeds the threshold.
:::

::: proof
*Proof.* Use the finite-sample absolute-error inequality to obtain a lower bound on the true target in terms of observed-minus-radius, then combine with the margin-violation premise. ◻
:::

::: theorem
[]{#thm:md-high-confidence-fail-condition-bundle label="thm:md-high-confidence-fail-condition-bundle"} If either (i) observed affinity exceeds the predicted upper threshold by more than its finite-sample radius, or (ii) observed contact budget falls below the required threshold by more than its finite-sample radius, then the corresponding true violation follows.
:::

::: proof
*Proof.* Case split on the violated margin condition and apply the one-sided finite-sample margin-transport theorems. ◻
:::

::: theorem
[]{#thm:kd-interval-from-driving-energy-error label="thm:kd-interval-from-driving-energy-error"} Any certified absolute error radius on driving free energy propagates to an explicit equilibrium-affinity interval: $$K_d(\Delta G_{\mathrm{est}}+\varepsilon) \le K_d(\Delta G_{\mathrm{true}})
\le K_d(\Delta G_{\mathrm{est}}-\varepsilon).$$
:::

::: proof
*Proof.* Convert the absolute error bound to a two-sided free-energy interval and use monotonicity of the exponential map on negated, positive-temperature-scaled arguments. ◻
:::

::: theorem
[]{#thm:kd-interval-from-absolute-free-energy-model label="thm:kd-interval-from-absolute-free-energy-model"} For the absolute free-energy model, certified total correction bound directly induces a two-sided equilibrium-affinity interval around the base free-energy estimate.
:::

::: proof
*Proof.* Apply the previous driving-energy-error-to-$K_d$ interval theorem with error radius instantiated by the proven total correction-bound theorem of the absolute free-energy model. ◻
:::

::: theorem
[]{#thm:chemistry-data-backed-kd-interval-bundle label="thm:chemistry-data-backed-kd-interval-bundle"} From (i) explicit pH/ionic condition-window certificates, (ii) data-backed posterior calibration intervals for chemistry-coupling constants, and (iii) certified absolute free-energy correction bounds, one obtains a full uncertainty-propagation bundle to final equilibrium-affinity interval predictions.
:::

::: proof
*Proof.* Conjoin the unified chemical-state condition-window theorem, the dataset-to-posterior interval theorem, the absolute free-energy correction-error theorem, and the derived $K_d$ interval transport theorem. ◻
:::

::: theorem
[]{#thm:md-per-case-real-artifact-discharge-all-targets label="thm:md-per-case-real-artifact-discharge-all-targets"} If each protein--ligand case carries explicit Hamiltonian/force-field/geometry artifacts (exact-LJ utility identity, strict-winner witness family, cutoff-tail certificate, and geometry-cardinality certificates), then for each case the full concrete docking witness bundle is derived, yielding strict-optimality, cutoff-boundedness, and both excluded-atom/contact-shell structural-rank budgets.
:::

::: proof
*Proof.* For each case, construct a concrete docking witness directly from the real artifact package, then apply the existing concrete-witness discharge theorem. ◻
:::

::: theorem
[]{#thm:partition-correction-numerical-kd-upper-bound-bundle label="thm:partition-correction-numerical-kd-upper-bound-bundle"} Given certified numerical partition-function values $(Z_{\mathrm{bound}}, Z_{\mathrm{unbound}})$ with interval/positivity certificates and protocol-derived correction-term bounds, if the partition-ratio driving free energy clears the thermal-bit floor by the certified correction budget, then the corrected driving witness yields: $$K_d^{\mathrm{eq}} \le \exp(-r\ln 2).$$
:::

::: proof
*Proof.* Combine partition positivity and correction absolute-error closure with the partition-chain driving-floor theorem, then apply the equilibrium $K_d$ upper-bound map. ◻
:::

::: theorem
[]{#thm:production-independent-srank-extractor-bundle label="thm:production-independent-srank-extractor-bundle"} For a production extractor output that provides relevance/sufficiency certificates from structure/mechanics only (explicitly with no affinity input), the certified interval $$r_{\mathrm{lower}} \le \mathrm{srank}(D_{\mathrm{dock}}) \le r_{\mathrm{upper}}$$ is obtained directly.
:::

::: proof
*Proof.* Apply the independent rank-interval theorem to the extractor's lower relevance and upper sufficiency certificates, and conjoin with the declared structure-only/no-affinity provenance certificates. ◻
:::

::: theorem
[]{#thm:locked-prospective-falsification-run-soundness label="thm:locked-prospective-falsification-run-soundness"} If a falsification run locks split/metric/blind identifiers and records blinded prospective observations together with an explicit hard-threshold fail condition, then protocol preregistration and lock certificates hold, and the theory is formally falsified for that run.
:::

::: proof
*Proof.* Project preregistration and lock witnesses from the run record and unfold the protocol fail predicate to conclude falsification under the recorded fail condition. ◻
:::

::: theorem
[]{#thm:assay-noise-high-confidence-call-validity-bundle label="thm:assay-noise-high-confidence-call-validity-bundle"} With replicate/batch/instrument-decomposed assay-noise calibration exported as trajectory estimators:

1.  margin fail calls imply true threshold violations;

2.  margin pass calls imply true non-violation inequalities.

Hence high-confidence fail/pass calls are valid at the true-target level under the declared error-radius assumptions.
:::

::: proof
*Proof.* Convert decomposed assay-noise calibrations to trajectory estimators, then apply the one-sided finite-sample fail and pass transport theorems. ◻
:::

::: theorem
[]{#thm:target-class-chemistry-completeness-kd-interval-bundle label="thm:target-class-chemistry-completeness-kd-interval-bundle"} If target-class chemistry constants (protonation, tautomer, ionic, solvent, water-mediated, electronic) have fitted estimates with certified componentwise uncertainty bounds, then combined with absolute free-energy correction-stack uncertainty they propagate to a final two-sided equilibrium-$K_d$ interval.
:::

::: proof
*Proof.* Use triangle-inequality aggregation of component chemistry uncertainty with the absolute free-energy model error bound, then apply the driving-energy-error to $K_d$ interval theorem. ◻
:::

::: theorem
[]{#thm:single-target-full-physical-closure-instance-bundle label="thm:single-target-full-physical-closure-instance-bundle"} A single target instance carrying all practical artifacts (real witness discharge, numerical partition/correction closure, independent-rank extractor output, locked falsification run, assay-noise calibration, chemistry-complete calibration) exports one joint closure proposition covering all six practical completion axes.
:::

::: proof
*Proof.* Conjoin the six previously proved target-level bundles from the corresponding artifact components of the instance record. ◻
:::

::: theorem
[]{#thm:external-replication-at-scale-full-pipeline-bundle label="thm:external-replication-at-scale-full-pipeline-bundle"} For an outside-team/outside-compute replication package over a family of new targets, team and compute separation are certified, and each target exports the full single-target physical-closure proposition.
:::

::: proof
*Proof.* Project team/compute distinctness from the replication package and apply the single-target closure theorem pointwise to each target instance. ◻
:::

::: theorem
[]{#thm:single-state-partition-positive-bundle label="thm:single-state-partition-positive-bundle"} For any inverse-temperature and bound/unbound energies, the one-state exact partition constructor yields strictly positive certified partition values for both bound and unbound ensembles.
:::

::: proof
*Proof.* Instantiate the one-state partition constructor and use strict positivity of the exponential map. ◻
:::

::: theorem
[]{#thm:zero-correction-calibration-bundle label="thm:zero-correction-calibration-bundle"} The canonical one-sample zero-correction protocol calibration yields zero total correction, zero total component bound, and therefore satisfies the certified total correction-error inequality.
:::

::: proof
*Proof.* Expand the canonical zero-correction constructor and simplify each correction/bound component; then apply the protocol total-error theorem. ◻
:::

::: theorem
[]{#thm:upper-only-independent-srank-extractor-bundle label="thm:upper-only-independent-srank-extractor-bundle"} Given any certified sufficient upper coordinate set, the canonical upper-only extractor (empty lower set + no-affinity provenance certificates) yields: $$0 \le \mathrm{srank}(D_{\mathrm{dock}}) \le |I_{\mathrm{upper}}|.$$
:::

::: proof
*Proof.* Construct the extractor with empty lower set and the provided upper sufficiency witness, then apply the certified-interval theorem for production independent extractors. ◻
:::

::: theorem
[]{#thm:single-target-zero-rank-concrete-closure-bundle label="thm:single-target-zero-rank-concrete-closure-bundle"} From one real per-system witness package and one certified sufficient upper coordinate set, the canonical constructor with zero-rank/zero-correction operational defaults yields a full single-target closure proposition.
:::

::: proof
*Proof.* Compose the per-system witness package with: (i) the zero-rank partition/correction closure constructor, (ii) the upper-only independent extractor constructor, (iii) a locked zero-rank fail-run constructor, (iv) exact-observation assay-noise constructors, and (v) zero chemistry-shift calibration; then invoke the full single-target closure theorem. ◻
:::

::: theorem
[]{#thm:attested-provenance-replication-at-scale-constructor-bundle label="thm:attested-provenance-replication-at-scale-constructor-bundle"} Any attested independent-replication provenance package (typed outside-team/outside-compute identities with signed protocol-match artifacts), together with one full closure instance per target, induces a replication-at-scale package satisfying team separation, compute separation, and full per-target closure.
:::

::: proof
*Proof.* Convert attested provenance to the independent-provenance interface and apply the replication-at-scale full-pipeline theorem. ◻
:::

::: theorem
[]{#thm:concrete-attested-single-target-replication-bundle label="thm:concrete-attested-single-target-replication-bundle"} Using the concrete attested provenance artifacts already instantiated in this paper, plus one target-level real witness package and one certified sufficient upper coordinate set, the induced single-target replication package certifies outside-team/outside-compute separation and exports the full single-target closure proposition.
:::

::: proof
*Proof.* Instantiate the attested-provenance replication constructor on the singleton target roster, then project the singleton instance clause from the generic replication-at-scale bundle. ◻
:::

::: theorem
[]{#thm:computable-finite-enumeration-pose-solver-optimal label="thm:computable-finite-enumeration-pose-solver-optimal"} Given an explicit finite action enumeration certificate covering all candidate poses, the finite argmax pose-selection algorithm returns an action in the exact optimizer set for every state.
:::

::: proof
*Proof.* Use list-argmax maximality on the covering enumeration and transport the resulting utility dominance inequality to optimizer membership. ◻
:::

::: theorem
[]{#thm:rmsd-success-probability-unit-interval label="thm:rmsd-success-probability-unit-interval"} For any finite posterior over poses, the induced RMSD-success probability (posterior mass of poses within threshold $\epsilon$) is always in $[0,1]$.
:::

::: proof
*Proof.* Nonnegativity follows termwise from posterior nonnegativity. The upper bound follows by pointwise domination of the success-indicator-weighted sum by the full posterior mass sum, which equals one. ◻
:::

::: theorem
[]{#thm:topk-mass-lower-bound-rmsd-success-probability label="thm:topk-mass-lower-bound-rmsd-success-probability"} If every member of the paper4 finite top-$k$ docking set satisfies the RMSD threshold, then posterior mass on that top-$k$ set lower-bounds total RMSD-success probability.
:::

::: proof
*Proof.* Apply subset-sum monotonicity from the covered top-$k$ set into the RMSD-success set, then rewrite success probability as the filtered posterior sum. ◻
:::

::: theorem
[]{#thm:rmsd-probability-derived-pose-solver-bundle label="thm:rmsd-probability-derived-pose-solver-bundle"} For a pose-solver output carrying: selected top-$k$ pose, RMSD coverage certificate of the top-$k$ set, and a certified lower bound on top-$k$ posterior mass, one obtains jointly:

1.  selected-pose RMSD threshold satisfaction,

2.  a certified lower bound on RMSD-success probability,

3.  the RMSD-success probability upper/lower unit-interval bounds.
:::

::: proof
*Proof.* Conjoin selected-pose membership with top-$k$ coverage for item (1), compose top-$k$ mass lower bound with the top-$k$-to-success monotonicity theorem for item (2), and append the posterior unit-interval law for item (3). ◻
:::

::: theorem
[]{#thm:joint-computable-pose-rmsd-probability-bundle label="thm:joint-computable-pose-rmsd-probability-bundle"} Combining the finite-enumeration computable pose solver with a finite RMSD posterior model yields one bundled endpoint: exact optimizer membership of the computed pose plus calibrated RMSD-success probability bounds in $[0,1]$.
:::

::: proof
*Proof.* Conjoin the computable finite-enumeration optimizer theorem with the RMSD-success probability unit-interval theorem. ◻
:::

::: theorem
[]{#thm:raw-pocket-ligand-constructor-posterior-bundle label="thm:raw-pocket-ligand-constructor-posterior-bundle"} From raw molecular inputs (protein pocket, ligand, grid sampling controls, and RMSD reference map), the canonical sampled constructor yields one bundled certificate: base-action inclusion in sampled support, sampled-optimal selected pose, normalized nonnegative posterior, and JAX-codegen completeness of the canonical solver program.
:::

::: proof
*Proof.* Instantiate the raw constructor and conjoin base-support membership, selected-pose optimality, posterior normalization/nonnegativity, and canonical JAX-codegen-success lemmas. ◻
:::

::: theorem
[]{#thm:deployment-contract-implies-benchmark-contract label="thm:deployment-contract-implies-benchmark-contract"} Under the deployment RMSD calibration model (proxy RMSD plus certified absolute error bound), conservative deployment acceptance implies benchmark-mode acceptance for both RMSD and success-probability thresholds.
:::

::: proof
*Proof.* Transport RMSD pass by proxy-plus-error upper bounding and transport probability pass by monotonicity from conservative success mass into true benchmark success mass. ◻
:::

::: theorem
[]{#thm:canonical-program-execution-refines-solver-result label="thm:canonical-program-execution-refines-solver-result"} For canonical ProgramIR execution witnesses, the emitted accept-flag is equivalent to existence of an accepted solver certificate, and emitted reject-flag is equivalent to existence of a failure certificate.
:::

::: proof
*Proof.* Rewrite the runtime flag to the theorem-level decision predicate and compose with acceptance/failure characterization lemmas of the solver API. ◻
:::

::: theorem
[]{#thm:raw-pocket-ligand-benchmark-solver-bundle label="thm:raw-pocket-ligand-benchmark-solver-bundle"} For the raw-input benchmark solver endpoint, one bundled result is exported: base-action support inclusion, canonical JAX-codegen success, and exact accepted/failure iff-characterizations against the benchmark contract.
:::

::: proof
*Proof.* Combine constructor-support and codegen-success theorems with benchmark accepted/failure equivalence lemmas for the raw benchmark API. ◻
:::

::: theorem
[]{#thm:canonical-raw-benchmark-acceptance-equivalence label="thm:canonical-raw-benchmark-acceptance-equivalence"} For the canonical RMSD map induced from raw ligand geometry, benchmark acceptance of the raw canonical solver is equivalent to the benchmark contract predicate.
:::

::: proof
*Proof.* Specialize the generic raw benchmark acceptance-equivalence theorem to the canonical RMSD constructor. ◻
:::

::: theorem
[]{#thm:canonical-raw-deployment-acceptance-equivalence label="thm:canonical-raw-deployment-acceptance-equivalence"} For the canonical zero-error deployment calibration, raw deployment acceptance is equivalent to benchmark contract satisfaction.
:::

::: proof
*Proof.* Compose deployment accepted-iff theorem with canonical calibration deployment-vs-benchmark contract equivalence. ◻
:::

::: theorem
[]{#thm:canonical-raw-program-witness-refinement label="thm:canonical-raw-program-witness-refinement"} For the canonical raw-input ProgramIR witness constructor, runtime accept/reject flags refine exactly to accepted/failure outcomes of the canonical raw benchmark solver.
:::

::: proof
*Proof.* Specialize the raw witness refinement theorem to the canonical RMSD constructor and canonical raw benchmark wrapper. ◻
:::

::: theorem
[]{#thm:canonical-raw-definitive-endpoint-bundle label="thm:canonical-raw-definitive-endpoint-bundle"} The canonical raw endpoint (raw molecular input, canonical RMSD map, canonical deployment calibration, canonical ProgramIR witness) exports one definitive bundle: support inclusion, JAX codegen success, benchmark/deployment acceptance equivalences, total benchmark/deployment result coverage, and runtime-flag-to-certificate refinement.
:::

::: proof
*Proof.* Instantiate the generic definitive endpoint bundle theorem at canonical RMSD, canonical deployment calibration, and canonical witness constructors. ◻
:::

::: theorem
[]{#thm:canonical-runtime-output-refines-solver label="thm:canonical-runtime-output-refines-solver"} The canonical runtime-output interpreter for the docking ProgramIR refines exactly to solver certificate outcomes: accept-flag true iff an accepted benchmark certificate exists, and accept-flag false iff a benchmark failure certificate exists.
:::

::: proof
*Proof.* Instantiate the canonical execution witness at solver-defined output fields and apply the execution-refinement theorem. ◻
:::

::: theorem
[]{#thm:definitive-raw-crossdock-accept-benchmark-iff label="thm:definitive-raw-crossdock-accept-benchmark-iff"} For the single-name definitive raw cross-docking API, deployment accepted output exists exactly when the canonical raw benchmark contract is satisfied.
:::

::: proof
*Proof.* Rewrite the definitive API to the canonical raw deployment wrapper and apply the canonical deployment-acceptance equivalence theorem. ◻
:::

::: theorem
[]{#thm:definitive-raw-crossdock-accept-deployment-iff label="thm:definitive-raw-crossdock-accept-deployment-iff"} For the definitive raw cross-docking API, accepted output exists exactly when the canonical deployment contract predicate holds.
:::

::: proof
*Proof.* Compose definitive accepted-iff-benchmark equivalence with canonical deployment-vs-benchmark contract equivalence. ◻
:::

::: theorem
[]{#thm:definitive-raw-crossdock-totality label="thm:definitive-raw-crossdock-totality"} The definitive raw deployment endpoint is total: every invocation returns either an accepted certificate or a rejected deployment certificate.
:::

::: proof
*Proof.* Specialize canonical raw deployment totality and rewrite through the definitive API wrapper. ◻
:::

::: theorem
[]{#thm:definitive-raw-runtime-flag-refines-accept label="thm:definitive-raw-runtime-flag-refines-accept"} For the definitive raw benchmark endpoint, runtime accept-flag truth is equivalent to existence of an accepted benchmark certificate.
:::

::: proof
*Proof.* Specialize canonical runtime-output refinement to the definitive raw benchmark wrapper. ◻
:::

::: theorem
[]{#thm:definitive-raw-crossdock-full-closure-bundle label="thm:definitive-raw-crossdock-full-closure-bundle"} The definitive raw cross-docking API exports one bundled closure theorem containing: base-action support inclusion, JAX codegen success, benchmark/deployment acceptance equivalences, benchmark/deployment totality, runtime-flag refinement for both accept/reject outcomes, and deployment-acceptance-to-benchmark-acceptance transport.
:::

::: proof
*Proof.* Conjoin the definitive wrapper theorems for support, codegen, acceptance equivalences, totality, runtime refinement, and deployment-to-benchmark acceptance transport. ◻
:::

::: theorem
[]{#thm:definitive-raw-benchmark-accepted-iff-contract label="thm:definitive-raw-benchmark-accepted-iff-contract"} For the definitive raw benchmark wrapper, accepted output exists exactly when the canonical benchmark contract holds.
:::

::: proof
*Proof.* Specialize the canonical benchmark accepted-iff-contract equivalence theorem through the definitive benchmark wrapper. ◻
:::

::: theorem
[]{#thm:definitive-raw-deployment-rejected-iff-not-contract label="thm:definitive-raw-deployment-rejected-iff-not-contract"} For the definitive raw deployment wrapper, a rejected output exists exactly when the canonical deployment contract predicate fails.
:::

::: proof
*Proof.* Use definitive deployment totality plus accepted-iff-deployment-contract equivalence and constructor disjointness to obtain the exact rejection characterization. ◻
:::

::: theorem
[]{#thm:definitive-accept-flag-iff-deployment-accepted label="thm:definitive-accept-flag-iff-deployment-accepted"} For the definitive raw endpoint, runtime accept-flag truth is equivalent to existence of a deployment accepted certificate.
:::

::: proof
*Proof.* Compose runtime-flag-to-benchmark-accept refinement with benchmark/deployment contract equivalence and deployment accepted-iff-contract closure. ◻
:::

::: theorem
[]{#thm:definitive-reject-flag-iff-deployment-rejected label="thm:definitive-reject-flag-iff-deployment-rejected"} For the definitive raw endpoint, runtime accept-flag falsity is equivalent to existence of a deployment rejection certificate.
:::

::: proof
*Proof.* Combine runtime-failure refinement with benchmark-failure-vs-contract negation, transport negation to deployment contract, and apply deployment rejection equivalence. ◻
:::

::: theorem
[]{#thm:computable-rational-accept-flag-exactness label="thm:computable-rational-accept-flag-exactness"} For the rationalized acceptance kernel, the computable Boolean accept-flag is exactly equivalent to the pair of rational margin inequalities used in the kernel decision rule.
:::

::: proof
*Proof.* Unfold the computable flag definition and simplify the Boolean 'decide' equivalence. ◻
:::

::: theorem
[]{#thm:computable-rational-accept-soundness label="thm:computable-rational-accept-soundness"} If the computable rationalized accept-flag is true and the kernel carries certified absolute error bounds between rationalized and real-valued RMSD/success-probability quantities, then the benchmark contract holds at the real solver level.
:::

::: proof
*Proof.* Lift rational margin inequalities to real inequalities, combine with the one-sided consequences of absolute error bounds, and conclude the benchmark RMSD/probability contract conjuncts. ◻
:::

::: theorem
[]{#thm:computable-rational-accept-refines-benchmark-accept label="thm:computable-rational-accept-refines-benchmark-accept"} Under the same rationalized kernel assumptions, computable accept-flag truth implies existence of an accepted benchmark certificate from the formal solver API.
:::

::: proof
*Proof.* Apply computable acceptance soundness to obtain benchmark-contract satisfaction, then invoke the benchmark accepted-iff-contract theorem. ◻
:::

::: theorem
[]{#thm:canonical-interpreter-state-runtime-refinement label="thm:canonical-interpreter-state-runtime-refinement"} The canonical interpreter-state view of ProgramIR evaluation refines exactly to accepted/failure benchmark certificates with the same two-sided accept-flag semantics as the direct runtime constructor.
:::

::: proof
*Proof.* Rewrite interpreter-state runtime output to the direct runtime constructor and transport the previously proved runtime refinement theorem. ◻
:::

::: theorem
[]{#thm:definitive-interpreter-output-equals-runtime label="thm:definitive-interpreter-output-equals-runtime"} For the definitive raw endpoint wrapper, interpreter-evaluated runtime output and direct runtime output coincide definitionally.
:::

::: proof
*Proof.* Unfold both constructions and simplify; they are definitionally equal. ◻
:::

::: theorem
[]{#thm:definitive-report-runtime-accept-iff-deployment-accepted label="thm:definitive-report-runtime-accept-iff-deployment-accepted"} In the consolidated definitive report object, runtime accept-flag truth is equivalent to existence of a deployment accepted certificate.
:::

::: proof
*Proof.* Expand the report constructor fields and apply the definitive accept-flag-to-deployment-accept equivalence theorem. ◻
:::

::: theorem
[]{#thm:definitive-raw-crossdock-complete-lean-bundle label="thm:definitive-raw-crossdock-complete-lean-bundle"} The definitive raw cross-docking endpoint exports a single complete Lean closure bundle containing: base-action support inclusion, JAX codegen success, benchmark/deployment acceptance equivalences, runtime accept/reject equivalence to deployment accepted/rejected outcomes, deployment totality, and computable rational-kernel acceptance refinement to benchmark accepted certificates.
:::

::: proof
*Proof.* Conjoin definitive API support/codegen/acceptance/flag/totality theorems with the rational-kernel computable-acceptance refinement theorem specialized to the canonical raw input. ◻
:::

::: theorem
[]{#thm:definitive-report-runtime-reject-iff-deployment-rejected label="thm:definitive-report-runtime-reject-iff-deployment-rejected"} In the consolidated definitive report object, runtime reject-flag truth is equivalent to existence of a deployment rejected certificate.
:::

::: proof
*Proof.* Expand report fields and apply the definitive reject-flag-to-deployment-rejected equivalence theorem. ◻
:::

::: theorem
[]{#thm:definitive-constructive-benchmark-iff-kernel-flag label="thm:definitive-constructive-benchmark-iff-kernel-flag"} For any artifact-instantiated rational kernel at the definitive raw endpoint, the constructive benchmark decision is accepted exactly when the kernel's computable accept-flag is true.
:::

::: proof
*Proof.* Unfold the constructive benchmark wrapper into the computable-kernel decision function and apply kernel-decision accepted iff-flag exactness. ◻
:::

::: theorem
[]{#thm:definitive-constructive-benchmark-refines-certificate-backend-benchmark label="thm:definitive-constructive-benchmark-refines-certificate-backend-benchmark"} If the constructive benchmark decision is accepted for an artifact-instantiated kernel, then a certificate-backend benchmark accepted certificate exists.
:::

::: proof
*Proof.* Reduce the constructive benchmark wrapper to the computable-kernel endpoint and apply the previously proved computable-kernel-to-certificate-backend benchmark acceptance refinement theorem. ◻
:::

::: theorem
[]{#thm:definitive-constructive-deployment-refines-certificate-backend-deployment label="thm:definitive-constructive-deployment-refines-certificate-backend-deployment"} If the constructive deployment decision is accepted for an artifact-instantiated kernel, then a certificate-backend deployment accepted certificate exists.
:::

::: proof
*Proof.* First refine constructive acceptance to certificate-backend benchmark acceptance, transport benchmark contract truth to deployment-contract truth through canonical calibration equivalence, and conclude via certificate-backend deployment accepted iff deployment-contract. ◻
:::

::: theorem
[]{#thm:definitive-exact-rat-artifact-accept-iff-benchmark-contract label="thm:definitive-exact-rat-artifact-accept-iff-benchmark-contract"} For an exact-rational artifact instantiation (zero rational margins with declared exact equalities to real endpoint quantities), constructive benchmark acceptance is equivalent to the legacy benchmark contract.
:::

::: proof
*Proof.* Rewrite constructive acceptance to kernel accept-flag truth, then specialize the exact-rational witness theorem identifying flag truth with benchmark-contract satisfaction. ◻
:::

::: theorem
[]{#thm:definitive-exact-rat-artifact-accept-refines-certificate-backend-accepts label="thm:definitive-exact-rat-artifact-accept-refines-certificate-backend-accepts"} If an exact-rational artifact's constructive deployment decision accepts, then both certificate-backend benchmark acceptance and certificate-backend deployment acceptance certificates exist.
:::

::: proof
*Proof.* Combine constructive-to-certificate-backend benchmark refinement with constructive-to-certificate-backend deployment refinement and return their conjunction. ◻
:::

::: theorem
[]{#thm:definitive-benchmark-decision-alias-exactness label="thm:definitive-benchmark-decision-alias-exactness"} The public constructive benchmark decision endpoint is definitionally equal to the core constructive benchmark decision wrapper.
:::

::: proof
*Proof.* Unfold the alias definition; both terms are judgmentally equal. ◻
:::

::: theorem
[]{#thm:definitive-deployment-decision-alias-exactness label="thm:definitive-deployment-decision-alias-exactness"} The public constructive deployment decision endpoint is definitionally equal to the core constructive deployment decision wrapper.
:::

::: proof
*Proof.* Unfold the alias definition; both terms are judgmentally equal. ◻
:::

::: theorem
[]{#thm:definitive-benchmark-decision-refines-certificate-backend-benchmark label="thm:definitive-benchmark-decision-refines-certificate-backend-benchmark"} If the public constructive benchmark decision accepts, then a certificate-backend benchmark accepted certificate exists.
:::

::: proof
*Proof.* Rewrite the public decision endpoint to the core constructive wrapper and apply constructive decision-to-certificate-backend benchmark refinement. ◻
:::

::: theorem
[]{#thm:definitive-decision-refines-certificate-backend-deployment label="thm:definitive-decision-refines-certificate-backend-deployment"} If the public constructive deployment decision accepts, then a certificate-backend deployment accepted certificate exists.
:::

::: proof
*Proof.* Rewrite the public decision endpoint to the core constructive wrapper and apply constructive decision-to-certificate-backend deployment refinement. ◻
:::

::: theorem
[]{#thm:definitive-benchmark-decision-rejected-iff-kernel-flag-false label="thm:definitive-benchmark-decision-rejected-iff-kernel-flag-false"} For artifact-instantiated constructive benchmark decisions, rejection is exactly equivalent to the kernel computable accept-flag being false.
:::

::: proof
*Proof.* Use the accepted-decision iff kernel-flag-true theorem, case analysis on the two-point decision codomain, and contradiction on accepted-vs-rejected branch equality. ◻
:::

::: theorem
[]{#thm:definitive-benchmark-certified-accepted-iff-decision-accepted label="thm:definitive-benchmark-certified-accepted-iff-decision-accepted"} The constructive benchmark certified endpoint returns an accepted certificate exactly when the public constructive benchmark decision is accepted.
:::

::: proof
*Proof.* Unfold the certified constructor; one branch returns an accepted certificate by construction and the opposite branch is impossible under accepted-decision contradiction. ◻
:::

::: theorem
[]{#thm:definitive-benchmark-certified-rejected-iff-decision-rejected label="thm:definitive-benchmark-certified-rejected-iff-decision-rejected"} The constructive benchmark certified endpoint returns a rejected certificate exactly when the public constructive benchmark decision is rejected.
:::

::: proof
*Proof.* Unfold the certified constructor, use benchmark decision dichotomy, and in the negative accepted branch apply the canonical rejected-of-not-accepted reduction. ◻
:::

::: theorem
[]{#thm:definitive-deployment-certified-accepted-iff-decision-accepted label="thm:definitive-deployment-certified-accepted-iff-decision-accepted"} The constructive deployment certified endpoint returns an accepted certificate exactly when the public constructive deployment decision is accepted.
:::

::: proof
*Proof.* Unfold the constructive deployment certified constructor and reason by accepted-branch selection; contradiction eliminates the opposite branch. ◻
:::

::: theorem
[]{#thm:definitive-deployment-certified-rejected-iff-decision-rejected label="thm:definitive-deployment-certified-rejected-iff-decision-rejected"} The constructive deployment certified endpoint returns a rejected certificate exactly when the public constructive deployment decision is rejected.
:::

::: proof
*Proof.* Unfold the certified deployment constructor, split on accepted-vs-not-accepted, and use the rejected-of-not-accepted decision theorem for the rejection branch. ◻
:::

::: theorem
[]{#thm:definitive-signed-rationalized-artifact-manifest-consistency label="thm:definitive-signed-rationalized-artifact-manifest-consistency"} For any signed rationalized kernel artifact, manifest identity consistency, digest consistency, and signature-validity are exported jointly as one theorem-level bundle.
:::

::: proof
*Proof.* Project the three fields directly from the signed-artifact structure and conjoin them. ◻
:::

::: theorem
[]{#thm:definitive-signed-rationalized-decision-accept-refines-certificate-backend-deployment label="thm:definitive-signed-rationalized-decision-accept-refines-certificate-backend-deployment"} If a signed rationalized constructive deployment decision accepts, then a certificate-backend deployment accepted certificate exists.
:::

::: proof
*Proof.* Convert the signed artifact to its constructive artifact instantiation and invoke constructive decision-to-certificate-backend deployment acceptance refinement. ◻
:::

::: theorem
[]{#thm:definitive-exact-rat-rejected-refines-certificate-backend-rejections label="thm:definitive-exact-rat-rejected-refines-certificate-backend-rejections"} For exact-rational artifact instantiations, constructive rejection implies both a certificate-backend benchmark failure certificate and a certificate-backend deployment rejection certificate.
:::

::: proof
*Proof.* Transport rejected decision to benchmark rejection, negate benchmark contract by contradiction with exact accepted-iff-contract equivalence, then apply certificate-backend failure/rejection iff theorems. ◻
:::

::: theorem
[]{#thm:definitive-signed-exact-rat-accepted-iff-benchmark-contract label="thm:definitive-signed-exact-rat-accepted-iff-benchmark-contract"} For signed exact-rational artifacts, constructive benchmark decision acceptance is equivalent to benchmark-contract satisfaction.
:::

::: proof
*Proof.* Rewrite the signed exact artifact to its exact-rational instantiation and invoke the exact-rational constructive accepted-iff-contract theorem. ◻
:::

::: theorem
[]{#thm:definitive-signed-exact-rat-rejected-refines-certificate-backend-rejections label="thm:definitive-signed-exact-rat-rejected-refines-certificate-backend-rejections"} For signed exact-rational artifacts, constructive deployment decision rejection implies certificate-backend benchmark failure and certificate-backend deployment rejection certificates.
:::

::: proof
*Proof.* Reduce to the exact-rational rejection refinement theorem after unfolding signed-artifact decision and conversion definitions. ◻
:::

::: theorem
[]{#thm:definitive-signed-artifact-byte-envelope-roundtrip label="thm:definitive-signed-artifact-byte-envelope-roundtrip"} Encoding a signed artifact manifest into the canonical length-prefixed byte-envelope and then parsing it recovers exactly the five UTF-8 byte fields (artifact id, digest, signer, provenance, signature).
:::

::: proof
*Proof.* Unfold the byte-envelope encoder/parser, apply field-level parse-after-encode lemmas, and simplify the terminal empty-tail check. ◻
:::

::: theorem
[]{#thm:definitive-concrete-checksum-byte-e2e label="thm:definitive-concrete-checksum-byte-e2e"} For any manifest, encoding with the concrete checksum signature then parsing yields exact field recovery and byte-level checksum verification success.
:::

::: proof
*Proof.* Combine byte-envelope parse roundtrip with definitional equality between checksum-signature UTF-8 bytes and the concrete checksum-bytes constructor, then close verification by 'decide' exactness. ◻
:::

::: theorem
[]{#thm:definitive-signed-rationalized-concrete-byte-parse-verify label="thm:definitive-signed-rationalized-concrete-byte-parse-verify"} Any signed rationalized artifact validated by the concrete checksum verifier satisfies byte-level parse roundtrip and concrete checksum verification success for its encoded manifest/signature envelope.
:::

::: proof
*Proof.* First extract verifier-implied signature equality, then rewrite to the concrete checksum end-to-end parse-and-verify theorem. ◻
:::

::: theorem
[]{#thm:definitive-rationalized-separation-not-benchmark-contract label="thm:definitive-rationalized-separation-not-benchmark-contract"} Given a rationalized kernel and a strict separation witness (RMSD lower-gap violation or success-probability upper-gap violation), the benchmark contract is impossible.
:::

::: proof
*Proof.* Use absolute-error transport from rational quantities to real endpoint quantities and contradict each benchmark inequality in the corresponding separation branch. ◻
:::

::: theorem
[]{#thm:definitive-rationalized-separation-flag-false label="thm:definitive-rationalized-separation-flag-false"} Any rationalized kernel carrying a strict separation witness has computable accept-flag equal to false.
:::

::: proof
*Proof.* Assume flag true, unfold rational acceptance inequalities, and contradict strict separation in each branch using nonnegative margins. ◻
:::

::: theorem
[]{#thm:definitive-signed-rationalized-strict-rejection-refines-certificate-backend-rejections label="thm:definitive-signed-rationalized-strict-rejection-refines-certificate-backend-rejections"} For signed rationalized artifacts with a strict separation witness, constructive decision rejection is derivable and both certificate-backend benchmark-failure and deployment-rejection certificates are obtained.
:::

::: proof
*Proof.* Instantiate the kernel-level separation witness on the signed artifact, derive flag-false and decision rejection, then transport benchmark/deployment contract failure to certificate-backend failure/rejection certificates. ◻
:::

::: theorem
[]{#thm:definitive-concrete-checksum-verifier-exactness label="thm:definitive-concrete-checksum-verifier-exactness"} The concrete checksum artifact verifier validates exactly those signatures that are equal to the deterministic checksum signature computed from the manifest.
:::

::: proof
*Proof.* Unfold the concrete verifier definition; the statement is definitionally equivalent to signature equality. ◻
:::

::: theorem
[]{#thm:definitive-runtime-ops-closed-form label="thm:definitive-runtime-ops-closed-form"} For runtime profile parameters $(K,L,C,\mathrm{fuel},\mathrm{bytes})$, the definitive computable pipeline operation count is exactly $$(K+L) + \mathrm{bytes} + (C+1)\cdot \mathrm{fuel}.$$
:::

::: proof
*Proof.* Expand pruning checks, scorer calls, refinement steps, and parser bytes, then normalize arithmetic. ◻
:::

::: theorem
[]{#thm:definitive-runtime-ops-succ-recurrence label="thm:definitive-runtime-ops-succ-recurrence"} Increasing refinement fuel by one increases definitive computable total operation count by exactly $(C+1)$, where $C$ is conformer count.
:::

::: proof
*Proof.* Unfold the operation-count components and simplify the product/sum recurrence at fuel successor. ◻
:::

::: theorem
[]{#thm:definitive-pipeline-total-ops-closed-form label="thm:definitive-pipeline-total-ops-closed-form"} The integrated definitive computable pipeline wrapper preserves the same exact closed-form operation count under the attached runtime profile.
:::

::: proof
*Proof.* Reduce the wrapper result to its stored runtime profile and apply the closed-form operation-count theorem. ◻
:::

::: theorem
[]{#thm:definitive-branch-bound-prune-sound label="thm:definitive-branch-bound-prune-sound"} If certified candidate upper bound is strictly below certified incumbent lower bound, then candidate true score is strictly below incumbent true score.
:::

::: proof
*Proof.* Transport true scores into certified intervals via absolute-error inequalities and compose with strict bound separation. ◻
:::

::: theorem
[]{#thm:definitive-adaptive-stop-sound label="thm:definitive-adaptive-stop-sound"} If every remaining item has certified upper bound below incumbent lower bound, then no remaining true score can beat the incumbent true score.
:::

::: proof
*Proof.* Use the boolean stop-rule characterization to recover per-item strict upper-bound separation and compose with score upper/lower transport. ◻
:::

::: theorem
[]{#thm:definitive-pipeline-branch-bound-prune-sound label="thm:definitive-pipeline-branch-bound-prune-sound"} In the integrated definitive computable branch-and-bound wrapper, 'pruneFlag=true' implies candidate true score is strictly below incumbent true score.
:::

::: proof
*Proof.* Unfold the wrapper's prune flag and apply certified branch-and-bound prune soundness. ◻
:::

::: theorem
[]{#thm:definitive-batch-fusion-justified label="thm:definitive-batch-fusion-justified"} The definitive computable pipeline imports the ArrayDSL sharded-reduction/fused-reduction equivalence theorem, justifying vectorized JAX batch/fusion execution by semantics-preserving equality.
:::

::: proof
*Proof.* Directly instantiate the imported ArrayDSL shard/fusion equivalence theorem. ◻
:::

::: theorem
[]{#thm:definitive-parse-cost-linear-time label="thm:definitive-parse-cost-linear-time"} Signed-artifact byte-envelope parse cost is linearly bounded by byte-stream length.
:::

::: proof
*Proof.* The cost function is definitionally equal to input byte-stream length. ◻
:::

::: theorem
[]{#thm:definitive-parse-encode-cost-exact label="thm:definitive-parse-encode-cost-exact"} For any encoded signed artifact envelope, parse-cost equals encoded byte length exactly.
:::

::: proof
*Proof.* Unfold the parse-cost definition on the encoded envelope term. ◻
:::

::: theorem
[]{#thm:definitive-crypto-verifier-sound label="thm:definitive-crypto-verifier-sound"} Under the formalized cryptographic verifier assumptions, successful verifier acceptance implies signature bytes equal the modeled hash of manifest message bytes.
:::

::: proof
*Proof.* Unfold the artifact verifier wrapper and apply the verifier-soundness axiom from the cryptographic assumption package. ◻
:::

::: theorem
[]{#thm:definitive-signed-rationalized-crypto-byte-parse-verify label="thm:definitive-signed-rationalized-crypto-byte-parse-verify"} For signed rationalized artifacts instantiated with the cryptographic verifier model, byte-envelope parse roundtrip and cryptographic verification success hold jointly.
:::

::: proof
*Proof.* Combine signed-envelope parse roundtrip with the verifier-validity field in the signed artifact under the cryptographic verifier wrapper. ◻
:::

::: theorem
[]{#thm:definitive-signed-pipeline-parser-bytes-exact label="thm:definitive-signed-pipeline-parser-bytes-exact"} In the signed-artifact definitive computable pipeline wrapper, runtime-profile parser-byte count is exactly the byte-envelope parse-cost of the signed manifest/signature encoding.
:::

::: proof
*Proof.* Unfold the signed-wrapper runtime profile construction; parser-byte field is definitionally the envelope parse-cost. ◻
:::

::: theorem
[]{#thm:definitive-campaign-pair-evals-closed-form label="thm:definitive-campaign-pair-evals-closed-form"} For runtime profile parameters $(K,L,C,\mathrm{fuel},\mathrm{bytes})$, the definitive campaign pair-evaluation count is exactly $$(K\cdot L)\cdot C\cdot \mathrm{fuel}.$$
:::

::: proof
*Proof.* Expand pair budget as pocket-budget times ligand-budget and scorer-call count as conformer-count times refinement fuel, then normalize products. ◻
:::

::: theorem
[]{#thm:definitive-campaign-pair-evals-succ-recurrence label="thm:definitive-campaign-pair-evals-succ-recurrence"} Increasing refinement fuel by one increases definitive campaign pair-evaluation count by exactly $(K\cdot L)\cdot C$.
:::

::: proof
*Proof.* Use distributivity of multiplication over fuel successor in the closed-form campaign pair-evaluation expression. ◻
:::

::: theorem
[]{#thm:definitive-pipeline-campaign-pair-evals-closed-form label="thm:definitive-pipeline-campaign-pair-evals-closed-form"} The integrated definitive computable pipeline wrapper preserves the exact campaign pair-evaluation closed form under its attached runtime profile.
:::

::: proof
*Proof.* Reduce the wrapper output to its runtime profile and apply the campaign pair-evaluation closed-form theorem. ◻
:::

::: theorem
[]{#thm:definitive-pair-potential-fusion-justified label="thm:definitive-pair-potential-fusion-justified"} For the scorer kernel, ArrayDSL fused pair-potential evaluation is extensionally equal to the explicit unfused map-then-reduce reference form.
:::

::: proof
*Proof.* Directly instantiate the imported ArrayDSL fused/unfused pair-potential equivalence theorem. ◻
:::

::: theorem
[]{#thm:definitive-canonical-scorer-op-label-fusion-sound label="thm:definitive-canonical-scorer-op-label-fusion-sound"} In the canonical solver ProgramIR, 'sumPairPotentials' is a required operation, and its fused implementation is justified by the theorem-level fused/unfused scorer-kernel equivalence.
:::

::: proof
*Proof.* Combine the required-ops membership theorem for the canonical solver ProgramIR with the definitive pair-potential fusion justification theorem. ◻
:::

::: theorem
[]{#thm:prospective-empirical-closure label="thm:prospective-empirical-closure"} For a blinded prospective benchmark record with calibration/predictive errors and uncertainty radii:

1.  empirical closure exports blinded-protocol validity, calibration coverage, predictive coverage, and failure-rate-within-bound certificates,

2.  any strict upper threshold above the declared failure-rate bound is inherited by the realized failure rate.
:::

::: proof
*Proof.* The first clause is the bundled empirical-closure theorem from benchmark record fields. The second clause is monotone transport of realized failure rate through the declared failure-rate upper bound. ◻
:::

::: theorem
[]{#thm:unified-physical-ood-prospective-bundle label="thm:unified-physical-ood-prospective-bundle"} Combining one unified thermodynamic/kinetic physical model, one universality/OOD calibration layer, and one prospective blinded benchmark record yields a single bundled endpoint containing:

1.  thermodynamic+kinetic physical certificates,

2.  uniform OOD prediction-error calibration bounds,

3.  blinded empirical closure and failure-rate-bound certificates.
:::

::: proof
*Proof.* Conjoin the unified physical-model bundle theorem with the universality/OOD uniform calibration theorem and the prospective benchmark empirical-closure theorem. ◻
:::

::: theorem
[]{#thm:paper4-witness-chain-import label="thm:paper4-witness-chain-import"} The current paper4 discharge chain imports directly into paper3 as theorem-level witnesses: locality EP1/EP2/EP3 bundle, reversible-chain entropy-production nonnegativity witness, FDT-to-stationarity discharge rule, differentiability-to-Born--Oppenheimer discharge rule, and concrete large-radius Lennard--Jones lattice-tail witnesses (6 and 12 powers).
:::

::: proof
*Proof.* Apply the bundled paper4-to-paper3 witness theorem, which conjoins these imported witness chains and exposes them as one bridge endpoint for downstream composition. ◻
:::

::: theorem
[]{#thm:paper4-interface-discharge-extensions label="thm:paper4-interface-discharge-extensions"} Beyond the core paper4 witness-import chain:

1.  any explicit TUR inequality certificate directly discharges the theorem-level TUR interface, and

2.  any explicit shadow-Hamiltonian cubic-drift certificate directly discharges the velocity-Verlet backward-error interface,

and these two conversion layers are exported together with the existing paper4 witness chain in one bundled theorem.
:::

::: proof
*Proof.* Apply the extended paper4-to-paper3 witness-chain theorem: the first conjunct is the existing imported chain, the second conjunct is direct certificate-to-TUR conversion, and the third conjunct is direct certificate-to-shadow-Hamiltonian conversion. ◻
:::

::: theorem
[]{#thm:paper4-stochastic-relevance-conjecture-full-support label="thm:paper4-stochastic-relevance-conjecture-full-support"} For Boolean product state spaces with full-support stochastic distributions, paper4's exploratory stochastic-preservation relevance conjecture is theorem-level discharged: stochastic preservation is equivalent to containing every stochastically-relevant-for-preservation coordinate.
:::

::: proof
*Proof.* Apply the imported paper4 full-support theorem, which proves the equivalence directly by combining: (i) full-support static/stochastic preservation equivalence, (ii) Boolean static sufficient-set characterization by relevant-coordinate containment, and (iii) transport from static relevance to stochastic relevance witnesses for one-coordinate erasures. ◻
:::

::: theorem
[]{#thm:paper4-stochastic-relevance-general-distribution-progress label="thm:paper4-stochastic-relevance-general-distribution-progress"} For stochastic preservation with nonnegative state weights:

1.  containment of all stochastically-relevant-for-preservation coordinates is a necessary condition for any preserving coordinate set;

2.  on Boolean product states with positive-fiber support, the full preservation-versus-containment equivalence is reduced to one bridge premise: static relevance implies stochastic relevance.
:::

::: proof
*Proof.* Combine the imported nonnegative necessary-direction theorem with the Boolean positive-fiber-support reduction theorem: the first gives containment necessity from preservation directly, and the second packages the reverse direction into the single static-to-stochastic relevance bridge premise. ◻
:::

::: theorem
[]{#thm:paper4-stochastic-relevance-conjecture-nonneg-support-transport label="thm:paper4-stochastic-relevance-conjecture-nonneg-support-transport"} For Boolean product states with nonnegative distributions, if queried fibers and one-coordinate-erasure fibers each admit positive support representatives preserving optimizer sets, then paper4's exploratory stochastic-preservation relevance conjecture is fully discharged: stochastic preservation is equivalent to containing all stochastically-relevant-for-preservation coordinates.
:::

::: proof
*Proof.* Apply the imported nonnegative-support-transport theorem, which upgrades the earlier reduction-to-bridge result by proving the static-to-stochastic relevance bridge from explicit support-transport witnesses for erasure fibers and then composing with the static sufficient-set characterization. ◻
:::

::: theorem
[]{#thm:paper4-stochastic-relevance-conjecture-nonneg-primitive-dynamics label="thm:paper4-stochastic-relevance-conjecture-nonneg-primitive-dynamics"} For Boolean product states with nonnegative distributions, if queried fibers and one-coordinate-erasure fibers admit primitive finite-time dynamics witnesses that preserve queried coordinates and optimizer sets while reaching positive-probability states after burn-in, then the unrestricted paper4 stochastic-relevance conjecture is discharged (equivalence between stochastic preservation and stochastic-relevance containment).
:::

::: proof
*Proof.* Use primitive-dynamics burn-in maps to construct support-representative transport witnesses for queried and erasure fibers, then apply the nonnegative support-transport closure theorem. ◻
:::

::: theorem
[]{#thm:paper4-stochastic-relevance-conjecture-nonneg-explicit-step-dynamics label="thm:paper4-stochastic-relevance-conjecture-nonneg-explicit-step-dynamics"} For Boolean product states with nonnegative distributions, if queried and one-coordinate-erasure fibers carry explicit one-step deterministic dynamics witnesses whose iterates preserve queried coordinates/optimizer sets and reach positive-probability states after burn-in, then the unrestricted stochastic-relevance conjecture is discharged.
:::

::: proof
*Proof.* Convert one-step deterministic dynamics witnesses into primitive finite-time dynamics by iteration, then apply the primitive-dynamics unrestricted-distribution closure theorem. ◻
:::

::: theorem
[]{#thm:paper4-stochastic-relevance-support-transport-of-explicit-step-dynamics label="thm:paper4-stochastic-relevance-support-transport-of-explicit-step-dynamics"} Under the same nonnegative-distribution and explicit one-step dynamics hypotheses:

1.  explicit-step dynamics canonically generate support-representative transport witnesses for queried fibers,

2.  therefore the unrestricted stochastic-relevance closure theorem is recovered through the support-transport route as an explicit assumption-reduction chain.
:::

::: proof
*Proof.* First apply the explicit-step-to-support-transport conversion theorem for queried and erasure fibers. Then invoke the unrestricted nonnegative support-transport closure theorem with those generated witnesses. ◻
:::

::: theorem
[]{#thm:concrete-docking-kinetic-bundle-specialization label="thm:concrete-docking-kinetic-bundle-specialization"} Given a kinetic bridge whose decision object is identified with a concrete docking problem, the bundled kinetic report transports unchanged to that docking object: on-rate envelope with docking rank denominator, residence-time identity, and pathway normalization.
:::

::: proof
*Proof.* Start from the generic kinetic bundle theorem and rewrite along the declared decision-object identification to obtain the docking-specialized denominator and unchanged remaining clauses. ◻
:::

## Fault-Tolerant Resolution

::: theorem
[]{#thm:error-correction-srank-overhead label="thm:error-correction-srank-overhead"} Let $s_{\mathrm{err}}$ be a nonzero Boolean error syndrome on $r$ logical resolution bits, and let $$d_H(s_{\mathrm{err}},0)$$ be its Hamming distance from the no-fault codeword. Then the fault-tolerant exact-resolution rank is $$r_{\mathrm{FT}} = r + d_H(s_{\mathrm{err}},0).$$ In particular, $$r + 1 \le r_{\mathrm{FT}}.$$
:::

::: proof
*Proof.* The theorem-level fault-tolerant rank is defined as the logical rank plus the syndrome's Hamming distance from the no-fault word. A nonzero syndrome has positive Hamming distance, so at least one extra structural-rank unit is required beyond the logical rank. ◻
:::

::: theorem
[]{#thm:fault-tolerant-landauer-floor label="thm:fault-tolerant-landauer-floor"} Assume Landauer calibration at positive Boltzmann constant and temperature. With the same notation as Theorem [\[thm:error-correction-srank-overhead\]](#thm:error-correction-srank-overhead){reference-type="ref" reference="thm:error-correction-srank-overhead"}, the fault-tolerant exact-resolution floor satisfies $$r\,k_B T\ln 2
<
\mathrm{energyLowerBound}(M,r_{\mathrm{FT}}),$$ and more precisely $$\mathrm{energyLowerBound}(M,r_{\mathrm{FT}})
=
r\,k_B T\ln 2 + \mathrm{energyLowerBound}\!\bigl(M,d_H(s_{\mathrm{err}},0)\bigr).$$
:::

::: proof
*Proof.* Under Landauer calibration, the logical part contributes exactly $r k_B T \ln 2$. The additional syndrome bits contribute additively through the same linear thermodynamic model, and a nonzero syndrome contributes strictly positive extra cost. Therefore the fault-tolerant floor is strictly above the logical Landauer floor by exactly the syndrome-erasure term. ◻
:::

::: theorem
[]{#thm:hopfield-ninio-proofreading-overhead label="thm:hopfield-ninio-proofreading-overhead"} Assume Landauer calibration at positive $k_B,T$. Let $\eta_{\mathrm{eq}}>0$ be the equilibrium discrimination and let $\eta$ be the proofreading discrimination in the exponential model $$\eta
=
\eta_{\mathrm{eq}}\exp\!\left(\frac{\Delta G_{\mathrm{proof}}}{k_B T}\right).$$ If the proofreading free-energy branch is realized within the syndrome-overhead witness budget $$\Delta G_{\mathrm{proof}}
\le
\mathrm{energyLowerBound}\bigl(M,d_H(s_{\mathrm{err}},0)\bigr),$$ then $$\frac{\log(\eta/\eta_{\mathrm{eq}})}{\log 2}
\le d_H(s_{\mathrm{err}},0),
\qquad
r + \frac{\log(\eta/\eta_{\mathrm{eq}})}{\log 2}
\le r_{\mathrm{FT}},$$ and $$k_B T\,\log\!\left(\frac{\eta}{\eta_{\mathrm{eq}}}\right)
\le
\mathrm{energyLowerBound}\bigl(M,d_H(s_{\mathrm{err}},0)\bigr).$$
:::

::: proof
*Proof.* The logarithmic specificity gain in the exponential proofreading model is exactly $\Delta G_{\mathrm{proof}}/(k_B T)$. The Landauer-calibrated overhead witness then converts the free-energy budget inequality into a bit-overhead inequality, yielding the displayed $\log_2(\eta/\eta_{\mathrm{eq}})$ lower bound on syndrome overhead and the corresponding dissipation lower bound. The $r_{\mathrm{FT}}$ inequality is the additive fault-tolerant rank identity. ◻
:::

::: theorem
[]{#thm:hopfield-ninio-kinetic-branch label="thm:hopfield-ninio-kinetic-branch"} Assume Landauer calibration at positive $k_B,T$. Let $$k_{\mathrm c}^{\mathrm{eq}},k_{\mathrm e}^{\mathrm{eq}} > 0$$ be equilibrium correct/error branch rates and let $$k_{\mathrm c}^{\mathrm{pf}},k_{\mathrm e}^{\mathrm{pf}}$$ be proofreading correct/error branch rates satisfying $$\frac{k_{\mathrm c}^{\mathrm{pf}}}{k_{\mathrm e}^{\mathrm{pf}}}
=
\frac{k_{\mathrm c}^{\mathrm{eq}}}{k_{\mathrm e}^{\mathrm{eq}}}
\exp\!\left(\frac{\Delta G_{\mathrm{proof}}}{k_B T}\right).$$ If $$\Delta G_{\mathrm{proof}}
\le
\mathrm{energyLowerBound}\bigl(M,d_H(s_{\mathrm{err}},0)\bigr),$$ then $$\frac{\log\!\left(\dfrac{k_{\mathrm c}^{\mathrm{pf}}/k_{\mathrm e}^{\mathrm{pf}}}
{k_{\mathrm c}^{\mathrm{eq}}/k_{\mathrm e}^{\mathrm{eq}}}\right)}{\log 2}
\le d_H(s_{\mathrm{err}},0),
\qquad
r +
\frac{\log\!\left(\dfrac{k_{\mathrm c}^{\mathrm{pf}}/k_{\mathrm e}^{\mathrm{pf}}}
{k_{\mathrm c}^{\mathrm{eq}}/k_{\mathrm e}^{\mathrm{eq}}}\right)}{\log 2}
\le r_{\mathrm{FT}},$$ and $$k_B T\,\log\!\left(\frac{k_{\mathrm c}^{\mathrm{pf}}/k_{\mathrm e}^{\mathrm{pf}}}
{k_{\mathrm c}^{\mathrm{eq}}/k_{\mathrm e}^{\mathrm{eq}}}\right)
\le
\mathrm{energyLowerBound}\bigl(M,d_H(s_{\mathrm{err}},0)\bigr).$$
:::

::: proof
*Proof.* Set $\eta_{\mathrm{eq}} := k_{\mathrm c}^{\mathrm{eq}}/k_{\mathrm e}^{\mathrm{eq}}$ and $\eta := k_{\mathrm c}^{\mathrm{pf}}/k_{\mathrm e}^{\mathrm{pf}}$. The branch-rate hypothesis is exactly the exponential specificity model in those variables, and $\eta_{\mathrm{eq}}>0$ follows from positivity of the equilibrium rates. Applying the theorem-level Hopfield--Ninio reduction in this kinetic parametrization yields the two bit-overhead inequalities and the dissipation inequality. ◻
:::

## Binding Free-Energy Bridge

Structural rank controls the minimum free-energy budget required for physical binding resolution. Binding free energy is the thermodynamic budget for resolving decision-quotient distinctions.

::: theorem
[]{#thm:binding-free-energy-floor label="thm:binding-free-energy-floor"} Let $D$ be an exact decision problem with sufficient coordinate set $I$, and let $\Delta G$ be a measured binding free-energy budget. If $\Delta G$ dominates the exact-resolution witness energy $\mathrm{energyLowerBound}(M,|I|)$ under Landauer calibration, then $$\Delta G \ge \mathrm{srank}(D)\,k_B T\ln 2.$$
:::

::: proof
*Proof.* The sufficient-set witness gives the theorem-level structural-rank energy floor for $D$. Landauer calibration identifies the per-bit coefficient with $k_B T \ln 2$. If the measured binding free energy dominates the witness energy, it must also dominate the implied Landauer structural-rank floor. ◻
:::

::: theorem
[]{#thm:binding-free-energy-tightness label="thm:binding-free-energy-tightness"} Under the hypotheses of Theorem [\[thm:binding-free-energy-floor\]](#thm:binding-free-energy-floor){reference-type="ref" reference="thm:binding-free-energy-floor"} with $k_B,T>0$, define the residual entropy term $$R_{\mathrm{res}}(D,\Delta G)
:=
\frac{\Delta G - \mathrm{srank}(D)\,k_B T\ln 2}{k_B T}.$$ Then $$\Delta G
=
\mathrm{srank}(D)\,k_B T\ln 2 + (k_B T)\,R_{\mathrm{res}}(D,\Delta G),
\qquad
R_{\mathrm{res}}(D,\Delta G) \ge 0.$$ Moreover, $$\Delta G = \mathrm{srank}(D)\,k_B T\ln 2
\iff
R_{\mathrm{res}}(D,\Delta G)=0.$$
:::

::: proof
*Proof.* The decomposition identity is algebraic: define the gap above the rank-Landauer floor and normalize by $k_B T$. Nonnegativity follows from Theorem [\[thm:binding-free-energy-floor\]](#thm:binding-free-energy-floor){reference-type="ref" reference="thm:binding-free-energy-floor"}. The iff statement is immediate because $k_B T>0$, so the normalized residual vanishes exactly when the unnormalized gap vanishes. ◻
:::

**Physical Significance.** The inequality $\Delta G \ge \mathrm{srank}(D)\,k_B T\ln 2$ identifies affinity with information resolution. A stronger binder can resolve a finer decision quotient; a weaker binder cannot pay for those distinctions. The free-energy budget and the exact decision complexity are therefore constrained by one common thermodynamic scale.

::: theorem
[]{#thm:decision-quotient-potential label="thm:decision-quotient-potential"} Let $a_\ast$ be a strict winner at state $s$. If every thermal perturbation from $s$ to $s'$ is bounded by $$\mathrm{srank}(D)\,k_B T,$$ and if the binding-funnel gap satisfies $$2\,\mathrm{srank}(D)\,k_B T < \mathrm{StrictUtilityGap}(a_\ast,s),$$ then exact resolution is preserved: $$\mathrm{Opt}(s) = \mathrm{Opt}(s').$$
:::

::: proof
*Proof.* The theorem is a rank-scaled half-gap argument. The assumed thermal perturbation radius is exactly the stagewise tolerance in the strict-gap invariance theorem, and the funnel-gap inequality says that this tolerance is smaller than half of the strict winner's utility gap. Exact winner preservation follows immediately. ◻
:::

## Strict Overhead Above Landauer

**Interpretation of the floor.** The structural-rank Landauer law $E\ge \mathrm{srank}(D)\,k_B T\ln 2$ is the irreducible logical basement: the zero-friction, zero-mismatch minimum compatible with exact resolution. It is not a predictor of total empirical binding free energy. Real molecular implementations operate strictly above that basement when kinetic barriers, finite-time relaxation losses, or distributional mismatch are present, exactly as captured by the strict-overhead branch theorems in this subsection (Propositions [\[prop:strict-overhead\]](#prop:strict-overhead){reference-type="ref" reference="prop:strict-overhead"}--[\[prop:strict-canonical-energy\]](#prop:strict-canonical-energy){reference-type="ref" reference="prop:strict-canonical-energy"}).

::: corollary
[]{#cor:logical-basement-overhead label="cor:logical-basement-overhead"} For any exact-resolution molecular realization with measured free-energy budget $\Delta G$, define $$\Delta G_{\mathrm{floor}} := \mathrm{srank}(D)\,k_B T\ln 2,
\qquad
\Delta G_{\mathrm{over}} := \Delta G-\Delta G_{\mathrm{floor}}.$$ Then $$\Delta G = \Delta G_{\mathrm{floor}} + \Delta G_{\mathrm{over}},
\qquad
\Delta G_{\mathrm{over}}\ge 0.$$ Moreover, whenever either strict-overhead branch witness of Propositions [\[prop:strict-overhead\]](#prop:strict-overhead){reference-type="ref" reference="prop:strict-overhead"}--[\[prop:strict-canonical-energy\]](#prop:strict-canonical-energy){reference-type="ref" reference="prop:strict-canonical-energy"} is instantiated, one has $$\Delta G_{\mathrm{over}}>0.$$ Hence the rank law is an irreducible information-theoretic minimum, while realized dissipation is generically strictly larger.
:::

::: proof
*Proof.* The decomposition identity and nonnegativity are exactly Theorem [\[thm:binding-free-energy-tightness\]](#thm:binding-free-energy-tightness){reference-type="ref" reference="thm:binding-free-energy-tightness"}. Strict positivity of the overhead term is exactly the strict-overhead branch package, which proves realized per-bit coefficients and induced exact-resolution energies strictly above the Landauer floor under mismatch/residual witnesses. ◻
:::

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

**Interpretive note.** The $3N-k$ expression controls the irreducible logical floor for exact resolution; realized dissipation can be strictly larger once mismatch/residual overhead branches are instantiated.

Informally: matter pays for what its topology requires it to know.

::: corollary
[]{#cor:bond-family-holonomic-floor label="cor:bond-family-holonomic-floor"} Let $F$ be a finite bond-constraint family on $N$ atoms with $$|F| \le N,
\qquad N>0.$$ Let $X_F$ be the induced finite RATTLE topology with constraint count $k=|F|$. Then $$\left|\Omega_{\mathrm{status}}(X_F)\right| = 2^k,
\qquad
\mathrm{srank}(\mathrm{canonicalDP}(A_{X_F})) = 3N-k,$$ and for every sufficient coordinate set $I$ and every thermodynamic model with positive per-bit conversion constant, $$M.\mathrm{joulesPerBit}\cdot(3N-k)
\le
\mathrm{energyLowerBound}(M,|I|).$$
:::

::: proof
*Proof.* The concrete family bound $|F|\le N$ and positivity $N>0$ imply $|F|<3N$, which discharges the strict independent-count requirement of the finite RATTLE topology. The binary status-register cardinality, structural-rank identity, and Landauer-linear floor then follow from the concrete bond-family bridge theorems. ◻
:::

**Interpretive note.** This corollary fixes the same $3N-k$ logical basement with an explicit finite family certificate; it is not a claim that total empirical dissipation equals that basement.

::: remark
[]{#rem:bond-family-scope label="rem:bond-family-scope"} The corollary derives the independent-count hypothesis from an explicit finite bond-family certificate. The next corollary discharges the Jacobian hypotheses for nonlinear geometric families carrying an explicit pivot-column Jacobian witness.
:::

::: corollary
[]{#cor:jacobian-holonomic-floor label="cor:jacobian-holonomic-floor"} Let $F$ be a finite nonlinear geometric-constraint family on $N$ atoms with $k$ constraints, and let $q$ be a configuration. Assume there is a pivot-column certificate:

1.  a map $\pi : \{1,\dots,k\}\to\{1,\dots,3N\}$,

2.  Jacobian selector identities $$J_F(q)_{i,\pi(j)}=
    \begin{cases}
    1,&i=j,\\
    0,&i\neq j,
    \end{cases}$$

3.  strict slack $k<3N$.

Then $$k < 3N,
\qquad
\left|\Omega_{\mathrm{status}}(X_{F,q})\right| = 2^k,
\qquad
\mathrm{srank}(\mathrm{canonicalDP}(A_{X_{F,q}})) = 3N-k,$$ and for every sufficient coordinate set $I$ and every thermodynamic model with positive per-bit conversion constant, $$M.\mathrm{joulesPerBit}\cdot(3N-k)
\le
\mathrm{energyLowerBound}(M,|I|).$$
:::

::: proof
*Proof.* The pivot identities imply full row rank at $q$ (L403) and the strict rank-defect condition (L404). The nonlinear family therefore induces a finite RATTLE topology with no additional Jacobian hypotheses at use sites, and the status-register cardinality, structural-rank identity, and Landauer-linear floor follow from . ◻
:::

**Interpretive note.** As above, the Jacobian-instantiated $3N-k$ law is an irreducible exact-resolution minimum; kinetic inefficiency and mismatch terms can place realized trajectories strictly above it.

**Physical Significance.** The pivot certificate is an explicit local Jacobian sparsity witness: each constraint has a coordinate direction where it is uniquely first-order active. This yields independent holonomic constraints and a nontrivial tangent manifold directly, so the $3N-k$ thermodynamic floor follows without extra family-specific postulates.

::: theorem
[]{#thm:geometric-constraint-decision-interface label="thm:geometric-constraint-decision-interface"} Let a geometric constraint family be given by finite bond/angle/dihedral counts together with a decision procedure returning a Boolean value whose correctness theorem is $$\mathrm{decideIndependent}=\mathrm{true}
\iff
k_{\mathrm{bond}}+k_{\mathrm{angle}}+k_{\mathrm{dihedral}}<3N.$$ Whenever the procedure returns `true`, the induced constrained molecular system satisfies $$\mathrm{srank}(\mathrm{canonicalDP}(A))
=
3N-(k_{\mathrm{bond}}+k_{\mathrm{angle}}+k_{\mathrm{dihedral}}).$$ Thus independence certification can be discharged directly from constraint data without an externally supplied pivot witness.
:::

::: proof
*Proof.* The decision-procedure correctness equivalence supplies the strict independent-count antecedent needed to instantiate the constrained molecular system. The structural-rank identity is then the same finite constrained-molecular transport theorem specialized to that instantiated system. ◻
:::

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

[^1]: Two-level quantized spectra provide the minimal physical realization of a binary decision channel in foundational statistical and quantum mechanics treatments [@planck1901distribution; @dirac1930principles; @sakurai2017modern].


# Convergence and Universal Consequences {#five-way-equivalence}

Degree of freedom equals structural rank, structural rank fixes quotient entropy and canonical Fisher-identifiable dimension under optimal-action observations, and the same rank controls exact-certification burden and thermodynamic cost. The molecular instantiations above make that chain concrete for constrained molecular systems. The remaining theorems in this section record the universal consequences of that chain once the docking specialization has already been made explicit.

The same rank-indexed law governs molecular chemistry, biological control, and ultimate physical limits. In the universal theorem family, catalysis is interpreted as constraint-induced transition-state rank reduction; proofreading is interpreted as fault-tolerant rank overhead above the logical floor; and Bekenstein-type limits are interpreted as absolute caps on rank-indexed information that bounded matter can physically resolve.

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

**Physical Significance.** Taken together with the thermodynamic bridge of Section [\[main-theorems\]](#main-theorems){reference-type="ref" reference="main-theorems"}, these universal consequences state a single principle: molecular systems compute by paying for decision distinctions. Catalysts lower the number of distinctions that must be resolved along reaction pathways, proofreading increases the number that must be stabilized against noise, and finite-volume physics bounds the total rank-indexed information-processing capacity available to any material substrate.

## Falsifiability Criteria

The framework makes three direct empirical claims.

1.  Binding free-energy budgets are bounded below by rank-indexed Landauer floors, $\Delta G \ge \mathrm{srank}(D)\,k_B T\ln 2$, under the antecedents of Remark [\[rem:thermo-antecedents\]](#rem:thermo-antecedents){reference-type="ref" reference="rem:thermo-antecedents"}.

2.  Quotient-coarsened admissibility relaxation lowers the declared linear floor by one calibrated unit per erased exact relevant coordinate.

3.  Allosteric coupling vanishes when the mechanical-graph pathway certifying relevance is severed, with the corresponding rank drop in the induced summary problem.

Observed violations of these statements inside the stated antecedent class falsify the framework for that class. Systems outside the antecedent class remain outside the theorem scope.


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

The central result is a complete abstract-plus-molecular theory of exact resolution. The quotient theorems identify the coarsest exact abstraction. The rank theorems identify the irreducible coordinate count and canonical Fisher-identifiable dimension of that object. The complexity theorems identify both the hardness core and the witness budget required for sound checking. The thermodynamic theorems identify the corresponding cost floor. Theorem [\[thm:five-way\]](#thm:five-way){reference-type="ref" reference="thm:five-way"} then identifies the rank-$1$ regime of the canonical decision encoding as simultaneously the one-coordinate regime, the tractable sufficiency regime, and the thermodynamic ground state. The molecular sections instantiate the same framework for constrained molecular systems, cutoff-local docking structure, sampled docking, top-$k$ screening, and concrete scorer families.

Theorem [\[thm:admissible-docking-exhaustion\]](#thm:admissible-docking-exhaustion){reference-type="ref" reference="thm:admissible-docking-exhaustion"} fixes the admissible-process class under the no-collapse antecedent: quotient-factorized resolution with structural-rank-controlled cost.

The theorem package separates a structural part from an empirical calibration. The structural part is the finite acquisition chain, the canonical exact-resolution encoding, the quotient-factorization boundary, the exact-sufficiency hardness core together with its witness/checking lower bounds, the identity $\mathrm{DOF}(A)=\mathrm{srank}(\mathrm{canonicalDP}(A))$, the Fisher-rank identities, the cutoff-local docking rank bounds, the top-$k$ and ambiguity-band control theorems, the concrete Lennard-Jones and Coulomb cutoff invariance theorems, and the decision-entropy bound. The empirical inputs are bounded signal speed, the discrete transition interface used for acquisition, cutoff-local approximation control, and a positive per-bit lower bound. Landauer furnishes the universal floor.

The same structural layer now also identifies the exact admissibility mechanism for thermodynamic relief: only quotient-coarsened relaxations that erase exact relevant coordinates can lower structural rank and therefore lower the corresponding Landauer floor. The strengthened quantitative theorem counts that relief exactly for nested admissibility families, one erased relevant coordinate at a time. It also now supports two further structural extensions that matter for molecular computation: long-range allostery through a reusable protein mechanical graph API with an automatic contact-graph instantiation from reference geometry, a concrete one-hop contact-neighborhood corollary, and an explicit bounded contact-shell low-rank regime, and hierarchy through explicit bundled macro states and exact no-collapse criteria for renormalized admissibility.

The structural-rank Landauer law $E\ge \mathrm{srank}(D)\,k_B T\ln 2$ is the irreducible logical basement: the absolute zero-friction, zero-noise minimum compatible with exact resolution. It is not intended to predict total empirical binding free energy. Real molecular trajectories run strictly above this basement due to kinetic barriers, finite-time relaxation losses, and distributional mismatch, exactly the regime captured by the strict-overhead branch package (Propositions [\[prop:strict-overhead\]](#prop:strict-overhead){reference-type="ref" reference="prop:strict-overhead"}--[\[prop:strict-canonical-energy\]](#prop:strict-canonical-energy){reference-type="ref" reference="prop:strict-canonical-energy"}).

**Main consequences:**

-   The quotient-factorization boundary for exact abstractions and the physical exclusion of extra surjective collapse beyond the decision quotient.

-   The exact identification of degree of freedom with structural rank in the canonical decision encoding, the exact identification of structural rank with canonical Fisher-identifiable dimension under optimal-action observations, the parametric identifiability reading (Fisher rank as identifiable dimension under optimal-action observations), the Cramér--Rao non-identifiability consequence for non-relevant coordinates, the initial-object universality characterization of the canonical encoding, a kernel-completion categorical endpoint theorem (object/hom equivalence plus universal no-collapse canonicity for operation-kernel schemas, including decision and eventual-germ instances), a measure-theoretic AE-germ and renormalization endpoint package (AE-kernel instance, hom-level embedding, and scale-wise quotient invariance for operation-preserving RG flow), a measure-kernel detailed-balance/stationarity transport endpoint (single transport interface for mapped detailed-balance and stationarity claims, including scale-wise flow form) with deterministic measurable-kernel scale specialization and kernel-power stationarity/transport laws, a measure-kernel quotient-calculus/RG endpoint (quotient descent, RG-compatible power commutation, and RG-renormalized quotient-power stationarity under RG-stable detailed balance) together with theorem-level recovery of transport results as quotient-calculus instances, a path-space Crooks/log-ratio lift that vanishes on stationary kernel powers and transports across semigroup-homomorphic model maps, an expectation-level Jarzynski lift (unit exponential-log-ratio expectation on stationary or detailed-balance kernel powers with semigroup-homomorphic power transport), an explicit path-measure Jarzynski integral endpoint (integral representation plus transported-power integral identities), a process-level path-measure transport endpoint (pushforwarded path-process witness inducing expectation and integral transport), a canonical finite-horizon measurable-transition process endpoint (recursive base/successor path-measure construction plus scale/transport unit-integral consequences), a projective-consistency/Kolmogorov-extension interface endpoint for those finite-horizon marginals, a concrete measurable stochastic transition-kernel semigroup instantiation (Dirac unit plus bind composition) with deterministic-to-stochastic semigroup embedding, transported power-stationarity, and concrete all-scale Jarzynski expectation/path-integral endpoints, and a noncanonical encoding transport theorem giving structural-rank and Landauer-floor invariance under optimizer-preserving encoding equivalence.

-   The general hardness core for exact sufficiency certification, the maximal-rank hard family, and the quantitative witness/checking lower bounds for sound exact certification.

-   The cutoff-local structural-rank bound for molecular docking, the bounded-pocket low-rank regime, strict-dominance and multiplicative-separable collapse regimes for conformer search, sampled exact/coarse winner preservation under a half-gap hypothesis, an action-side symmetry-broken sampled refinement with a canonical strict winner, a uniform positive strict gap, no structural-rank increase, an honest continuous lifting theorem for the pinned sampled gap via a shared tie-break, bounded-action constructive coordinate extraction by single-coordinate sufficiency tests, inside-cutoff sufficiency for sampled docking under the stated compatibility assumptions, top-$k$ survivor preservation under a certified boundary gap, and ambiguity-band containment in near-tie regimes.

-   Quotient-coarsened admissibility rank reduction: admissibility summaries cannot increase structural rank and strictly reduce it exactly when they erase an exact relevant coordinate; for nested admissibility families, the rank drop is exactly the number of erased exact relevant coordinates, with one declared per-bit thermodynamic unit saved per erased coordinate in the linear floor model, and matched base-rank plus matched erasure-count hypotheses determine the same relaxed rank-normal-form object up to isomorphism. A bidirectional factorization criterion now gives a theorem-level zero-collapse/rank-invariance regime at fixed tolerance increment. The same collapsed-rank layer now has explicit category structure (identity/composition, initial object, min/max product-coproduct operations, additive tensor, and monotone structural-rank functor) together with a proved no-terminal-object obstruction in the unbounded regime and a bounded-rank slice where finite limits, finite colimits, finite-family meet/join objects, arbitrary-family meet/join (complete-lattice form), and meet/join monotonicity-idempotence-absorption corollaries with binary rewrite calculus are available.

-   Explicit allosteric and hierarchical extensions: reusable protein mechanical graphs bound the additional rank carried by distant allosteric coordinates through active-pocket mechanical neighborhoods, geometry-derived contact graphs now instantiate that theorem surface directly from reference coordinates and site radii, one-hop contact-neighborhood bounds reduce the remaining coverage burden to a direct shell around the site pocket, a distance-decay shell-envelope theorem now gives a monotone non-increasing per-distance allosteric contribution upper profile together with an explicit shell-sum rank bound, plus calibrated exponential- and polynomial-envelope specializations, a polynomial $p$-series budget corollary, and an explicit no-parameter $p\ge 2$ instantiation, broken pathways erase the corresponding coupling term, bundled macro states support theorem-level hierarchical rank loss, and zero-collapse admissibility relaxation preserves the exact relevant-coordinate quotient under renormalization.

-   The inverse rank-gap design certificate: in the bounded-action regime, the mechanized extraction theorem can be run in reverse to return a sufficient free-coordinate set whose complement is a certified coordinate-constraint pattern meeting a requested rank, strict-gap, and energy budget.

-   The exact/coarse Lennard-Jones and Coulomb invariance theorems for finite sampled scorer families under explicit cutoff error and half-gap conditions, the closed-form Lennard-Jones gradient, exact phase-space volume preservation for Velocity-Verlet, and exponential Ewald control for long-range Coulomb splitting.

-   The energy--information theorem $E \ge k_B T H_{\mathrm{nats}}(D)$ for exact-resolution cost.

-   The pathwise structural-rank lower bound: finite stagewise exact-resolution costs add along a trajectory, the substrate tick law identifies those stages with interface time units, and finite quotient trajectories admit a theorem-level forward/reverse path-weight ratio law together with a cumulative dissipation lower bound under explicit rank-calibration hypotheses.

-   The bounded-acquisition inequalities $\mathrm{DOF}(A) \le c\tau/d$, the induced decision-class and decision-entropy bounds from spacetime and energy budget, and the linear budget law for independent composition.

-   The finite-resolution speed and robustness extensions: exact resolution of any finite quotient obeys the theorem-level speed bound $d\,\mathrm{srank}(D) \le c\tau$, stagewise trajectory time and energy budgets add, and the admissibility-indexed layer now yields a quantitative speed--accuracy law $$d\bigl(\mathrm{srank}(D_{F,\varepsilon})-\mathrm{CollapseCount}_F\bigr)\le c\tau,$$ together with a real on-rate envelope $1/\tau \le c/(d\,\mathrm{srank}(D_{F,\varepsilon+\Delta}))$, linking measurable binding speed to admissibility-relaxed specificity in the theorem package; in the zero-collapse bifactor regime, this specializes to the tight-rank speed bound without a subtraction term. The quotient-trajectory law now includes a theorem-level Crooks standard-form reduction, a detailed-balance equilibrium calibration corollary ($P_{\mathrm f}/P_{\mathrm r}=1$ under Boltzmann-stationary matching), and a finite Jarzynski equality corollary under explicit calibration. Fault-tolerant exact resolution requires syndrome Hamming-distance overhead above the logical rank and therefore lies strictly above the logical Landauer floor, Hopfield--Ninio proofreading overhead is recovered as a theorem-level lower bound on fault-tolerant rank and dissipation overhead with an explicit kinetic branch-rate specialization, and any binding free-energy budget dominating the exact-resolution witness must dominate the structural-rank Landauer floor together with a nonnegative residual-entropy decomposition and a tightness criterion (equality iff residual vanishes); conversely, a binding funnel gap larger than $2\,\mathrm{srank}(D)k_B T$ preserves exact resolution against thermal perturbations.

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

Molecular docking now sits inside one formal decision-theoretic framework linking semantic, statistical, complexity, geometric, and thermodynamic statements. Abstractly, exact resolution is governed by sufficient coordinate sets, the decision quotient, structural rank, canonical Fisher-identifiable dimension, certification burden, calibrated thermodynamic floor, finite acquisition speed, and fault-tolerant overhead. Concretely, constrained molecular systems instantiate the same framework through holonomic constraint topology, cutoff-local interaction structure, reusable protein mechanical graphs, geometry-derived contact graphs, and direct contact-shell allostery bounds, conformer-search collapse regimes, sampled exact/coarse stability, top-$k$ survivor control, finite grid entropy bounds, explicit binary lifts of finite grid docking, concrete scorer-family invariance, exact phase-space volume preservation, long-range electrostatic splitting, reverse coordinate-constraint certificates, and direct thermodynamic theorems for the exact docking decision problem itself.

That same framework now separates two approximation notions that are often conflated in practice: quotient-coarsened admissibility, which can lower structural rank by forgetting exact relevant coordinates, and raw utility-profile relaxations, which may instead refine the exact quotient. The former is the precise exact-approximation notion captured by the admissibility rank theorems, including the new quantitative collapse law for nested admissibility families.

The abstract theorems state what exact resolution costs for any bounded decision system. The molecular instantiation now states that cost directly for the exact docking decision problem once a sufficient coordinate set is fixed, states the entropy-calibrated Landauer theorem for the explicit finite binary encoding of the sampled grid docking problem, and states the bounded-region sampled inside-cutoff consequence once the retained witness fits within a bounded-region acquisition budget. The sampled finite layer now also contains a theorem-backed action-side symmetry-breaking step: ties can be converted into a canonical strict winner with a uniform positive strict gap over the finite grid state space without increasing structural rank. The same finite layer now supports two additional directions that matter for design and kinetics: a reverse coordinate-constraint certificate theorem for low-rank, gap-controlled regimes, and a finite quotient-trajectory Crooks-style path-ratio theorem with a matching cumulative dissipation bound under explicit rank calibration. The cutoff-local docking theorems provide geometric upper bounds on docking structural rank, and the optimizer-class richness theorem provides a combinatorial lower bound; together they give a theorem-level two-sided bracket once both hypothesis classes are instantiated. Approximate and heuristic docking procedures lie in the same scope through approximation, sampling, or surrogate replacement. Corollary [\[cor:holonomic-landauer-floor\]](#cor:holonomic-landauer-floor){reference-type="ref" reference="cor:holonomic-landauer-floor"} gives the direct RATTLE finite derivation: the constraint-status interface is a $k$-bit binary register, the effective coordinate count is $3N-k$, and the canonical Landauer floor scales linearly with that remaining unconstrained dimension. Corollary [\[cor:bond-family-holonomic-floor\]](#cor:bond-family-holonomic-floor){reference-type="ref" reference="cor:bond-family-holonomic-floor"} derives the independent-count antecedent for concrete finite bond families with explicit cardinality certificates. Corollary [\[cor:jacobian-holonomic-floor\]](#cor:jacobian-holonomic-floor){reference-type="ref" reference="cor:jacobian-holonomic-floor"} closes the nonlinear Jacobian route for families with a pivot-column certificate.

Appendix [\[appendix-lean\]](#appendix-lean){reference-type="ref" reference="appendix-lean"} records proof provenance.

## Acknowledgment: AI-use Disclosure {#acknowledgment-ai-use-disclosure .unnumbered}

Generative AI tools (including Codex, Claude Code, Augment, Kilo, and OpenCode) were used throughout this manuscript, across all sections (Abstract, Introduction, theoretical development, proof sketches, applications, conclusion, and appendix) and across all stages from initial drafting to final revision. The tools were used for boilerplate generation, prose and notation refinement, LaTeX and structure cleanup, translation of informal proof ideas into candidate formal artifacts (Lean and LaTeX), and repeated adversarial reviewer-style critique passes to identify blind spots and clarity gaps.

The author retained full intellectual and editorial control, including problem selection, theorem statements, assumptions, novelty framing, acceptance criteria, and final inclusion or exclusion decisions. No technical claim was accepted solely from AI output. Formal claims reported as machine-verified were admitted only after Lean verification (no `sorry` in cited modules) and direct author review; Lean was used as an integrity gate for responsible AI-assisted research. The author is solely responsible for all statements, citations, and conclusions.


# Proof Provenance {#appendix-lean}

This appendix reports claim traceability directly from source and generated mapping artifacts.

Lean 4 with Mathlib verifies that the theorem chain follows from the stated definitions and antecedents. Empirical correspondence with molecular physics is controlled by the scope conditions and antecedent hypotheses stated in the main text.

Recent physical-instantiation and constructive-computability endpoints are tracked by handle block , including the latest optimization-theorem integrations for runtime op-count recurrences, campaign pair-evaluation decomposition, scorer-fusion correctness, certified branch-and-bound/adaptive stopping, parser-cost linearity, and cryptographic verifier transport.

## Claim Coverage Matrix

## Lean Handle Map

## Proof Hardness Index


  -----------------------------------------------------------------------------------------------------------------------------------------------------------------
  **Paper claim**                                                                                                  **Lean handle**
  ---------------------------------------------------------------------------------------------------------------- ------------------------------------------------
  Corollary 5.323: Concrete Bond-Family Holonomic Floor                                                            L396, L398, L397

  Corollary 4.11: Bounded Contact-Shell Low-Rank Regime                                                            L142

  Corollary 4.7: Bounded-Pocket Low-Rank Regime                                                                    L65

  Corollary 5.18: Decision-Entropy Bound from Spacetime and Energy Budget                                          IT3, BA1, BA2, BA5, BA6, EI1, L43

  Corollary 4.5: Checking Time Lower Bound                                                                         L70

  Corollary 5.19: Independent Composition Budget Law                                                               L17, L19, BA1, BA2, BA7, BA5, BA6, L43

  Corollary 5.322: RATTLE Holonomic-Constraint Landauer Floor                                                      L60, L61, L62

  Corollary 5.325: Jacobian-Rank Holonomic Floor for Nonlinear Constraints                                         L405, L407, L403, L404, L406

  Corollary 5.307: Logical Basement and Realized Dissipation                                                       L140, BA5, BA6, WM6, WP6, WR10, L43

  Corollary 5.6: Unique Minimum-Cost Regime                                                                        BA8, L54, L55

  Corollary 4.4: No Sound Checker Below Witness Budget                                                             L71

  Corollary 3.38: Higher-Rank Regime                                                                               L46

  Corollary 3.37: Rank-One Regime                                                                                  L51

  Proposition 5.21: Two-Level Atomic Realization                                                                   AC1, AC3, AC4

  Proposition 5.318: Repeated Binary Mismatch Work Law                                                             IT3, BA5, BA6, WP2, WM4, L43

  Proposition 5.317: Binary Mismatch Strengthens the Energy--Information Coefficient                               IT3, BA5, BA6, WP2, WM4, L43

  Proposition 5.316: Explicit Binary Mismatch Example                                                              BA5, BA6, WP2, WM4, L43

  Proposition 5.311: Repeated Residual-Example Work Law                                                            IT3, BA5, BA6, WR12, WR11, L43

  Proposition 5.310: Explicit Two-State Residual Example                                                           IT3, BA5, BA6, WR12, WR11, L43

  Proposition 2.10: Binding as Exact Resolution                                                                    BA3, BA4, BA5

  Proposition 2.5: Bounded Region                                                                                  BA1

  Proposition 5.314: Canonical Wolpert Grounding Bundle                                                            BA5, BA6, WP9, L43

  Definition 2.3: Bounded Decision System                                                                          L17, L19

  Proposition 5.312: Unified Energy--Information Hierarchy                                                         IT3, BA5, BA6, WR12, WP2, WM4, WR11, L43

  Proposition 3.44: Finite Compression-Relation Bridge                                                             L57, L58

  Proposition 5.309: Finite Discrete Residual Witness                                                              WR10, WR7, WR6

  Proposition 5.319: Positive Heat and Bounded Lifetime                                                            SE1, SE2, SE3, SE4, SE5

  Proposition 5.320: Finite Lifetime Throughput Bound                                                              SE5, IT3, L43

  Definition 2.15: Canonical Decision Problem                                                                      QT2, QT7, QT1, QT3

  Proposition 5.321: Speed-Heat Tradeoff                                                                           SE6

  Proposition 5.315: Strict Canonical Energy Above the Landauer Floor                                              BA5, BA6, WP6, L43

  Proposition 5.308: Theorem-Level Strict Overhead Branches                                                        BA5, BA6, WM6, WP6, WR10, L43

  Proposition 5.313: Structural-Resource Overhead                                                                  BA5, BA6, WP8, WP7, L43

  Proposition 5.23: Substrate Step is Unit Interface Time                                                          DT23, DT22, DT24

  Proposition 5.20: Threshold Channel Realization                                                                  CV8, CV9, CV7

  Theorem 5.104: Absolute Binding Free-Energy Closure with Standard Corrections                                    L740, L741, L742

  Theorem 5.105: Protocol-Derived Absolute Free-Energy Correction Closure                                          L758

  Theorem 3.1: Surjective Abstractions Either Factor or Erase                                                      L77

  Theorem 4.42: Action-Pinned Grid Gaps Lift to Continuous Admissibility                                           L120, L121

  Theorem 4.23: Action-Pinned Sampled Docking Has a Uniform Positive Strict Gap                                    L115, L116, L117

  Theorem 4.27: Admissibility-Collapse Canonicality at Fixed Erasure Count                                         L416, L415

  Theorem 5.15: Admissibility On-Rate Upper Envelope                                                               L447, L448

  Theorem 3.5: Admissibility Relaxation Gives Monotone Progress                                                    L395

  Theorem 4.24: Quotient-Coarsened Admissibility Cannot Increase Structural Rank                                   L118, L119

  Theorem 5.13: Admissibility Speed--Accuracy Tradeoff                                                             L457, L446, L458

  Theorem 5.14: Zero-Collapse Speed Specialization                                                                 L482

  Theorem 4.26: Zero-Collapse Criterion via Bidirectional Factorization                                            L479, L480, L481

  Definition 3.4: Progress Toward Admissible Docking                                                               L394

  Theorem 3.18: Measure-Theoretic AE and RG Endpoint                                                               L502, L503, L504, L505, L506, L507

  Theorem 4.12: Distance-Decay Allosteric Contribution Law                                                         L428, L429, L430

  Theorem 4.13: Exponential Distance-Envelope Allosteric Bound                                                     L431, L432, L433

  Theorem 4.14: Polynomial Distance-Envelope Allosteric Bound                                                      L437, L438, L439

  Theorem 4.15: Polynomial $p$-Series Budget Corollary                                                             L440, L441

  Theorem 4.16: Explicit Polynomial Series Instantiation ($p\ge 2$)                                                L443, L444, L442

  Theorem 4.8: Allosteric Graph-Local Structural-Rank Bound                                                        L132

  Theorem 5.205: Assay-Noise High-Confidence Call Validity Bundle                                                  L850

  Theorem 5.173: Attested Concrete No-Credible-Dismissal Corollary                                                 L818

  Theorem 5.172: Attested Concrete External-Validation Three-Way Bundle                                            L817

  Theorem 5.174: Attested Concrete Store-Backed Artifact Bundle                                                    L819

  Theorem 5.213: Attested-Provenance Replication-at-Scale Constructor Bundle                                       L858

  Theorem 5.139: Barrier-Crossing Reversible Chemical Dynamics from Physics                                        L784

  Theorem 5.304: Binding Free-Energy Floor                                                                         L138

  Theorem 5.305: Binding Free-Energy Tightness Decomposition                                                       L419, L422, L420, L421

  Theorem 5.45: Concrete Reference Realism with Explicit Aggregate Shell Constant                                  L701, L700

  Theorem 5.43: Concrete Reference Realism Shell Calibration                                                       L691

  Theorem 5.44: Concrete Reference Realism Transport                                                               L693, L692

  Theorem 2.6: Bounded Acquisition Rate                                                                            BA1, BA2

  Theorem 4.58: Bounded-Action Exact Checking Is Polynomial-Time Decidable                                         L87, L93

  Theorem 4.32: Bounded Meet/Join Algebraic Corollaries                                                            L470, L464, L467, L465, L469, L468, L466

  Theorem 4.33: Binary Bounded Meet/Join Rewrite Calculus                                                          L478, L475, L474, L476, L477, L472, L471, L473

  Theorem 4.31: Bounded Arbitrary-Family Complete-Lattice Form                                                     L463

  Theorem 4.29: Bounded Collapsed-Rank Slice Has Finite (Co)Limits                                                 L460, L459, L461

  Theorem 4.30: Bounded Finite-Family Meet/Join Package                                                            L462

  Theorem 4.53: Bounded Potential Plus Large Cutoff Gives Sampled Structural-Rank Bound                            L104

  Theorem 4.52: Bounded Potential Plus Large Cutoff Gives Structural-Rank Bound                                    L95

  Theorem 5.17: Decision-Class Bound from Spacetime and Energy Budget                                              IT4, IT3, BA1, BA2, BA5, BA6, EI1, L43

  Theorem 5.96: Calibrated Biophysical Chemical-State Separation                                                   L655, L654

  Theorem 3.14: Canonical Initial Object                                                                           L413

  Corollary 3.15: Initiality Package Gives Canonical Rank Identity                                                 L414

  Theorem 5.241: Canonical Interpreter-State Runtime Refinement                                                    L886

  Theorem 5.222: Canonical Program Execution Refines Solver Result                                                 L867

  Theorem 5.224: Canonical Raw Benchmark Acceptance Equivalence                                                    L869

  Theorem 5.227: Canonical Raw Definitive Endpoint Bundle                                                          L872

  Theorem 5.225: Canonical Raw Deployment Acceptance Equivalence to Benchmark Contract                             L870

  Theorem 5.226: Canonical Raw Program-Witness Refinement                                                          L871

  Theorem 5.228: Canonical Runtime Output Refines Solver Certificates                                              L873

  Theorem 4.3: Checker Budget Lower Bound                                                                          L69

  Theorem 5.85: Chemical-Augmented Docking Transport Endpoints                                                     L608, L609

  Theorem 5.91: Chemical-Component Variation Invariance at Fixed Core                                              L633

  Theorem 5.94: Mechanism-Derived Unified Chemical Dynamics with Stationarity Fixed Point                          L757

  Theorem 5.92: Chemically Coupled Utility/Optimizer Sensitivity                                                   L644, L643

  Theorem 5.97: Dataset-to-Posterior Chemical Separation                                                           L664, L663

  Theorem 5.90: Chemical/Ensemble Transport to Concrete Binding Problem                                            L623, L624, L625, L626

  Theorem 5.98: Bias-Aware Real-Data Chemical Separation                                                           L680

  Theorem 5.200: Chemistry-Condition Data-Backed Uncertainty-to-$K\_d$ Bundle                                      L845

  Theorem 6.1: Coherent Single-Source Regime                                                                       ORA1

  Theorem 4.28: Collapsed-Rank Category Structure                                                                  L450, L449, L454, L451, L453, L456, L455, L452

  Theorem 4.38: Composed Classical Force-Field Interface Endpoint                                                  L589, L590

  Theorem 5.32: Composed Hamiltonian Architecture Instance                                                         L589

  Theorem 5.33: Composed Hamiltonian Positive-Shell Lipschitz Bound                                                L590

  Theorem 5.215: Computable Finite-Enumeration Pose Solver Optimality                                              L860

  Theorem 5.238: Computable Rational Acceptance Flag Exactness                                                     L883

  Theorem 5.240: Computable Rational Acceptance Refines Benchmark Acceptance                                       L885

  Theorem 5.239: Computable Rational Acceptance Soundness                                                          L884

  Theorem 5.214: Concrete Attested Single-Target Replication Bundle                                                L859

  Theorem 5.34: Concrete Biomolecular Force-Field Calibration Bundle                                               L605, L607, L606

  Theorem 5.299: Concrete Docking Kinetic Bundle Specialization                                                    L627

  Theorem 5.166: Concrete No-Credible-Dismissal Corollary                                                          L811

  Theorem 5.165: Concrete External-Validation Three-Way Instantiation                                              L810

  Theorem 5.127: Concrete Generator-to-Canonical Path-Process Closure                                              L772

  Theorem 5.74: Concrete Langevin-to-Interface Constructor                                                         L604

  Theorem 5.138: Concrete QM Workflow Error-Analysis Transport                                                     L783

  Theorem 5.86: Conformational-Ensemble Docking Transport Endpoints                                                L610, L611, L612

  Theorem 5.79: Constructive Downstream Replacement Transport                                                      L628, L629

  Theorem 5.41: Constructive Empirical Realism Shell Envelope                                                      L683

  Theorem 5.42: Constructive Empirical Realism Transport                                                           L685, L684

  Theorem 5.168: Constructive Ito/Wiener Derived Endpoint Bundle                                                   L813

  Theorem 5.82: Constructive-Only Core-Field Consumer Migration                                                    L658

  Theorem 5.83: Constructive-Only Extended Consumer Migration                                                      L666

  Theorem 5.81: Constructive-Only Spec Replacement                                                                 L648

  Theorem 5.161: Constructive Closure of Extension-Interface Scope Gaps                                            L806

  Theorem 5.144: Constructive Stochastic + Multipole Scope-Gap Discharge Bundle                                    L789

  Theorem 4.10: Direct-Contact Neighborhood Allosteric Bound                                                       L141

  Theorem 5.84: Continuous-State Relevance/Srank/Landauer Transport via Measurable Encoding Bridges                L716, L714, L713, L715, L726

  Theorem 3.32: Continuous-Time/Continuous-State Interface Endpoint                                                L587, L581

  Theorem 4.59: Bounded-Action Coordinate Extraction Recovers Structural Rank                                      L122

  Theorem 4.65: Cutoff Coulomb Winner Preservation                                                                 L73

  Theorem 4.64: Cutoff Coulomb Uniform Approximation                                                               L74

  Theorem 4.66: Exact Coulomb Tail Control Gives Structural-Rank Bound                                             L96

  Theorem 2.4: Counting Gap                                                                                        BA10

  Theorem 3.10: Cramér--Rao Non-Identifiability for Irrelevant Coordinates                                         L412

  Theorem 5.27: Detailed-Balance Crooks Calibration (Equilibrium Form)                                             L483, L485, L486, L484

  Theorem 5.306: Decision-Quotient Potential Criterion                                                             L139

  Theorem 5.236: Definitive Acceptance Flag iff Deployment Acceptance                                              L881

  Theorem 5.276: Adaptive Stop Rule Soundness                                                                      L921

  Theorem 5.278: Definitive Batch/Fusion Justification                                                             L923

  Theorem 5.256: Certified Benchmark Output Accepted iff Decision Accepted                                         L901

  Theorem 5.257: Certified Benchmark Output Rejected iff Decision Rejected                                         L902

  Theorem 5.251: Benchmark Decision Alias Exactness                                                                L896

  Theorem 5.253: Public Benchmark Decision Refines Certificate-Backend Benchmark Acceptance                        L898

  Theorem 5.255: Benchmark Decision Rejected iff Kernel Flag False                                                 L900

  Theorem 5.275: Certified Branch-and-Bound Prune Soundness                                                        L920

  Theorem 5.284: Campaign Pair-Evaluation Closed Form                                                              L929

  Theorem 5.285: Campaign Pair-Evaluation Successor Recurrence                                                     L930

  Theorem 5.288: Canonical Scorer Op-Label and Fusion Soundness                                                    L933

  Theorem 5.266: Concrete Checksum Byte-Level Parse-and-Verify End-to-End                                          L911

  Theorem 5.271: Concrete Checksum Verifier Exactness                                                              L916

  Theorem 5.246: Constructive Benchmark Decision iff Kernel Accept-Flag                                            L891

  Theorem 5.247: Constructive Benchmark Acceptance Refines Certificate-Backend Benchmark Acceptance                L892

  Theorem 5.248: Constructive Deployment Acceptance Refines Certificate-Backend Deployment Acceptance              L893

  Theorem 5.281: Cryptographic Verifier Soundness Bridge                                                           L926

  Theorem 5.254: Public Deployment Decision Refines Certificate-Backend Deployment Acceptance                      L899

  Theorem 5.258: Certified Deployment Output Accepted iff Decision Accepted                                        L903

  Theorem 5.259: Certified Deployment Output Rejected iff Decision Rejected                                        L904

  Theorem 5.252: Deployment Decision Alias Exactness                                                               L897

  Theorem 5.249: Exact-Rational Artifact Acceptance iff Benchmark Contract                                         L894

  Theorem 5.250: Exact-Rational Artifact Acceptance Refines Both Certificate-Backend Accept Paths                  L895

  Theorem 5.262: Exact-Rational Rejection Refines Certificate-Backend Benchmark-Failure and Deployment-Rejection   L907

  Theorem 5.242: Definitive Interpreter Output Equals Definitive Runtime Output                                    L887

  Theorem 5.287: Definitive Pair-Potential Fusion Justification                                                    L932

  Theorem 5.279: Signed-Artifact Parse Cost Linear-Time Bound                                                      L924

  Theorem 5.280: Signed-Artifact Encode Parse-Cost Exactness                                                       L925

  Theorem 5.277: Integrated Branch-and-Bound Pipeline Prune Soundness                                              L922

  Theorem 5.286: Integrated Pipeline Campaign Pair-Evaluation Closed Form                                          L931

  Theorem 5.274: Integrated Definitive Pipeline Total-Op Closed Form                                               L919

  Theorem 5.269: Rationalized Separation Witness Forces Rejection Flag                                             L914

  Theorem 5.268: Rationalized Separation Witness Implies Benchmark-Contract Failure                                L913

  Theorem 5.234: Definitive Raw Benchmark Acceptance iff Benchmark Contract                                        L879

  Theorem 5.229: Definitive Raw Cross-Dock Acceptance iff Benchmark Contract                                       L874

  Theorem 5.230: Definitive Raw Cross-Dock Acceptance iff Deployment Contract                                      L875

  Theorem 5.244: Definitive Raw Cross-Dock Complete Lean Bundle                                                    L889

  Theorem 5.233: Definitive Raw Cross-Dock Full Closure Bundle                                                     L878

  Theorem 5.231: Definitive Raw Cross-Dock Totality                                                                L876

  Theorem 5.235: Definitive Raw Deployment Rejection iff Not Deployment Contract                                   L880

  Theorem 5.232: Definitive Raw Runtime Flag Refines Acceptance                                                    L877

  Theorem 5.237: Definitive Reject Flag iff Deployment Rejection                                                   L882

  Theorem 5.243: Definitive Report Runtime-Accept iff Deployment-Accepted                                          L888

  Theorem 5.245: Definitive Report Runtime-Reject iff Deployment-Rejected                                          L890

  Theorem 5.272: Definitive Runtime Op-Count Closed Form                                                           L917

  Theorem 5.273: Definitive Runtime Op-Count Successor Recurrence                                                  L918

  Theorem 5.265: Signed Artifact Byte-Envelope Parse Roundtrip                                                     L910

  Theorem 5.263: Signed Exact-Rational Benchmark Acceptance iff Benchmark Contract                                 L908

  Theorem 5.264: Signed Exact-Rational Rejection Refines Certificate-Backend Rejections                            L909

  Theorem 5.283: Signed Artifact Pipeline Parser-Byte Exactness                                                    L928

  Theorem 5.260: Signed Rationalized Artifact Manifest Consistency Bundle                                          L905

  Theorem 5.267: Signed Rationalized Artifact Concrete Byte Parse-and-Verify                                       L912

  Theorem 5.282: Signed Rationalized Crypto Byte Parse-and-Verify                                                  L927

  Theorem 5.261: Signed Rationalized Decision Acceptance Refines Certificate-Backend Deployment Acceptance         L906

  Theorem 5.270: Signed Rationalized Strict Rejection Refines Certificate-Backend Rejections                       L915

  Theorem 5.221: Deployment Contract Implies Benchmark Contract                                                    L866

  Theorem 5.169: Derived Generator--PDE Operator Closure                                                           L814

  Theorem 5.142: Descriptor-Calibrated Mechanistic OOD Transfer Bundle                                             L787

  Corollary 3.20: Deterministic Measurable-Kernel Scale Specialization                                             L515, L514, L516

  Theorem 2.7: Discrete Acquisition                                                                                BA3

  Theorem 3.13: DOF--Structural-Rank Identity                                                                      L43

  Theorem 5.159: Downstream Campaign Win Bundle                                                                    L804

  Theorem 5.164: Downstream Causal-Quality Campaign Bundle                                                         L809

  Theorem 5.47: Concrete Electronic-Structure Reference Calibration and Error Transport                            L724, L727, L723, L722

  Theorem 5.46: Electronic-Structure Correction Transport (Charge Transfer + Metal Coordination)                   L720, L721, L719, L718, L717

  Theorem 3.36: Noncanonical Encoding Transport (Rank and Floor)                                                   L489, L487, L488

  Theorem 5.5: Energy--Information Duality                                                                         IT3, EI1, L43

  Theorem 5.3: Rank Controls Exact-Resolution Cost                                                                 BA7, BA6, L43

  Theorem 6.8: Finite Replication Entropy Gap                                                                      L45

  Theorem 5.88: Multi-Step Ensemble Process Transport                                                              L645

  Theorem 5.87: One-Step Ensemble Population Transport                                                             L634

  Theorem 5.89: Ensemble Statistical Validity Transport                                                            L656

  Theorem 3.43: Decision-Entropy Bound                                                                             IT3, L43

  Theorem 5.190: Equilibrium $K\_d$ Bound from Partition-Ratio Correction Chain                                    L835

  Theorem 5.181: Equilibrium Dissociation Bound from Driving-Energy Floor                                          L826

  Theorem 5.300: Error-Correction Structural-Rank Overhead                                                         L136

  Theorem 5.155: Estimator Minimax Derivation Bundle                                                               L800

  Theorem 4.71: Ewald Reciprocal Core Is Positive                                                                  L81

  Theorem 5.52: Ewald Long-Range Certificate Import                                                                L677

  Theorem 4.70: Ewald Real-Space Exponential Decay                                                                 L82

  Theorem 4.67: Exact Real-Space Ewald Tail Control Gives Structural-Rank Bound                                    L98

  Theorem 4.1: General Hardness Core for Exact Sufficiency                                                         L63

  Theorem 5.125: Explicit Hamiltonian+Bath Elimination to Langevin Endpoints                                       L770

  Theorem 5.126: Explicit Molecular Constant-Drift SDE Endpoint Bundle                                             L771

  Theorem 5.156: Extended Physical Model Interface Scope Bundle                                                    L801

  Theorem 5.208: External Replication-at-Scale Full Pipeline Bundle                                                L853

  Theorem 5.160: Integrated External-Validation Three-Way Bundle                                                   L805

  Theorem 5.301: Fault-Tolerant Landauer Floor                                                                     L137

  Theorem 3.2: Feasible Collapse Maps Force Quotient Factorization                                                 L78

  Theorem 6.9: Finite-Budget No-Collapse                                                                           PH26

  Theorem 5.121: Finite-Sample Complexity Inversion Endpoints                                                      L746, L747

  Theorem 5.123: Finite-Sample Inversion with Count-Order-Derived Monotonicity                                     L761, L762, L763

  Theorem 5.124: Finite-Sample Inversion from Square-Count Complexity Bounds                                       L767, L768, L769, L766

  Theorem 5.196: Finite-Sample Margin Violation Implies True Upper-Bound Violation                                 L841

  Theorem 4.54: Finite Sampled Docking Has a Canonical Uniform Error Radius                                        L92

  Theorem 3.8: Explicit Optimizer-Likelihood Fisher Identification                                                 L408

  Theorem 3.12: Noisy/Partial Observation-Channel Instantiation                                                    L598, L597

  Theorem 3.11: General Observation-Channel Fisher Interface Endpoint                                              L583, L582

  Theorem 3.7: Fisher-Matrix Rank Equals Structural Rank                                                           L76

  Theorem 3.6: Total Fisher Information Equals Structural Rank                                                     L80

  Theorem 6.7: Convergence                                                                                         L43, L44, L47, L55

  Theorem 5.162: Fixed-Contract Pre-Registered Benchmark Conversion Bundle                                         L807

  Theorem 5.147: Force-Field-Derived Realistic SDE Endpoints                                                       L792

  Theorem 5.37: Full-State Biomolecular Force-Field Transport                                                      L642, L641, L640

  Theorem 5.80: Fully Constructive Pipeline Deprecation Readiness                                                  L636

  Theorem 5.137: Generator-Coefficient-to-Canonical Regularity Closure                                             L782

  Theorem 5.148: Generator PDE-Estimate to Canonical Regularity Closure                                            L793

  Theorem 5.326: General Geometric-Constraint Independence Decision Interface                                      L579

  Theorem 4.9: Geometry-Derived Contact-Graph Allosteric Bound                                                     L140

  Theorem 5.95: Gibbs-Conditioned Unified Chemical Dynamics Closure                                                L764

  Theorem 4.55: Finite Grid Exact-Docking Entropy Is Controlled by Structural Rank                                 L113

  Theorem 4.57: Grid Irrelevance Erasure Is Sound                                                                  L88

  Theorem 5.135: Hamiltonian Finite-Difference Drift Derivation Endpoints                                          L780

  Theorem 5.146: Hamiltonian $h\to0$ Mori-Zwanzig Limit Endpoints                                                  L791

  Theorem 4.2: Maximal Structural Rank in the Hard Family                                                          L64

  Theorem 5.99: Hierarchical Multi-Dataset Chemical Robustness with Finite-Sample Rate                             L688

  Theorem 5.102: Hierarchical Chemical Separation under Explicit Rate-Constant Upper Bounds                        L705, L706

  Theorem 5.103: Uniform Hierarchical Chemical Separation under Datasetwise Rate-Constant Margins                  L707

  Theorem 5.100: Hierarchical Chemical Separation under Explicit Rate Margin                                       L696

  Theorem 5.101: Uniform Hierarchical Chemical Separation under Datasetwise Rate Margins                           L703

  Theorem 5.112: Hierarchical Multi-Dataset Kinetic Inference with Pooled Rate Metadata                            L689

  Theorem 5.115: Hierarchical Kinetic Inference under Explicit Rate-Constant Upper Bounds                          L708, L709

  Theorem 5.116: Uniform Hierarchical Kinetic Inference under Datasetwise Rate-Constant Margins                    L710

  Theorem 5.113: Hierarchical Kinetic Inference under Explicit Pooled Rate Margins                                 L697

  Theorem 5.114: Uniform Hierarchical Kinetic Inference under Datasetwise Pooled Rate Margins                      L704

  Theorem 5.122: Joint Hierarchical Required-Size Bundle (Chemical + Kinetic)                                      L753

  Theorem 5.134: Joint Required-Size Inversion from Model-Dependent Constants                                      L779

  Theorem 4.18: Hierarchical Bundling Rank Bound                                                                   L134

  Theorem 5.303: Kinetic-Branch Hopfield--Ninio Specialization                                                     L435, L436, L434

  Theorem 5.302: Hopfield--Ninio Proofreading Overhead Reduction                                                   L426, L427, L425

  Theorem 5.158: Independent Outside-Team Replication Bundle                                                       L803

  Theorem 5.163: Independent Replication Provenance Bundle                                                         L808

  Theorem 5.153: Integrator Error-Stack Unified Thermo/Kinetic Bundle                                              L798

  Theorem 4.61: Atomistic Realization Bridge for Inverse Rank-Gap Synthesis                                        L586

  Theorem 4.60: Inverse Rank-Gap Design Certificate                                                                L126, L127

  Theorem 5.145: Ito-Wiener-Filtration Langevin Endpoint Bundle                                                    L790

  Theorem 5.30: Finite Jarzynski Equality from Crooks                                                              L424

  Theorem 5.219: Joint Computable-Pose and RMSD-Probability Bundle                                                 L864

  Theorem 5.199: $K\_d$ Interval from Absolute Free-Energy Correction Stack                                        L844

  Theorem 5.198: $K\_d$ Interval from Driving-Energy Error Radius                                                  L843

  Theorem 3.17: Kernel-Completion Categorical Endpoint                                                             L497, L496, L495, L499, L501, L498, L500

  Theorem 3.21: Kernel-Power Stationarity and Transport                                                            L521, L524, L522, L523, L518, L517, L519, L520

  Theorem 3.16: Kernel-Quotient Universality and No-Collapse Canonicity                                            L490, L491, L493, L492, L494

  Theorem 5.110: Kinetic Concentration/Identifiability Inference Bridge                                            L665

  Theorem 5.108: Kinetic Confidence Transport                                                                      L646, L647

  Theorem 5.106: Kinetic Observable Reporting Endpoints                                                            L616, L613, L615, L614

  Theorem 5.109: Kinetic Protocol Inference Guarantee under Noise Budgets                                          L657

  Theorem 5.107: Kinetic Protocol Measurement Transport                                                            L635

  Theorem 5.111: Replicate-Aware Kinetic Identifiability Bundle                                                    L681

  Theorem 5.57: Langevin Analysis-Closure Endpoint Bundle                                                          L602, L603, L600, L601, L599

  Theorem 5.56: Finite-State Boltzmann Stationarity from Detailed Balance                                          L637

  Theorem 5.70: Canonical Continuous Path-Process Closure from Finite-Horizon Bridge Data                          L732, L733

  Theorem 5.71: Constructive Canonical Path-Process Closure from Finite-Horizon Bridge Construction                L755

  Theorem 5.62: Concrete Hamiltonian+Bath End-to-End Langevin Closure                                              L754

  Theorem 5.67: Constructive Infinite-Dimensional Path-Derivation Bridge                                           L686, L687

  Theorem 5.64: Continuous-State Measure-Theoretic Langevin Closure                                                L651, L650

  Theorem 5.54: Langevin Detailed-Balance Grounding                                                                L591

  Theorem 5.55: Langevin Discretization to Quotient MCMC                                                           L592

  Theorem 5.59: Explicit SDE Analytic Conditions and Dissipative Ergodicity                                        L639, L638

  Theorem 5.68: Finite-Horizon Law to Constructive Path-Closure Bridge                                             L695, L694

  Theorem 5.69: Finite-Horizon Marginal Recovery Formulas                                                          L702

  Theorem 5.73: First-Principles Langevin Endpoint Discharge                                                       L630

  Theorem 5.72: Force-Field-Derived Drift-Lipschitz Injection                                                      L662, L667

  Theorem 5.66: Infinite-Dimensional Path-Measure Langevin Bridge                                                  L679

  Theorem 5.65: Ito/Fokker--Planck/Harris Continuous-Time Closure Bridge                                           L659

  Theorem 5.167: Measure-Theoretic Langevin Endpoint Bundle                                                        L812

  Theorem 5.60: Microscopic-to-First-Principles Langevin Closure Constructor                                       L712, L725, L711

  Theorem 5.61: Molecular Hamiltonian+Thermostat to Langevin Endpoint Constructor                                  L728, L731, L730, L729

  Theorem 5.63: Joint Microscopic Dynamical and Canonical Path-Process Closure Bundle                              L750

  Theorem 5.29: Langevin-to-MCMC Discretization Interface Endpoint                                                 L592, L591

  Theorem 4.43: Large Cutoff Implies Docking Cutoff Boundedness                                                    L89

  Theorem 5.154: Learned-Descriptor OOD Generalization Transfer Bundle                                             L799

  Theorem 5.78: Legacy-to-Constructive Deprecation Bridge                                                          L618, L617

  Theorem 4.39: Lipschitz Grid Error Gives Resolution-Controlled Approximation                                     L90

  Theorem 4.62: Cutoff Lennard-Jones Winner Preservation                                                           L75

  Theorem 4.68: Closed-Form Lennard-Jones Gradient                                                                 L83

  Theorem 4.44: Explicit Lennard-Jones Shell Derivative Envelope                                                   L103

  Theorem 4.51: Lennard-Jones Shell Gradient Bound Preserves Exact Winners                                         L100

  Theorem 4.48: Lennard-Jones Shell Gradient Bound Gives Score Stability                                           L102

  Theorem 4.46: Lennard-Jones Hessian Envelope Makes the Gradient Lipschitz                                        L106

  Theorem 4.50: First-Order-Corrected Lennard-Jones Discretization Error Is Quadratic                              L107

  Theorem 4.47: Lennard-Jones Hessian Envelope Gives a Quadratic Taylor Remainder                                  L108

  Theorem 4.45: Explicit Lennard-Jones Shell Second-Derivative Envelope                                            L105

  Theorem 4.49: Lennard-Jones Shell Gradient Bound Gives Uniform Exact/Grid Approximation                          L101

  Theorem 4.63: Exact Lennard-Jones Tail Control Gives Structural-Rank Bound                                       L97

  Theorem 4.34: Threshold-Explicit Tolerance-to-Collapse Law (Lennard-Jones Pocket Interface)                      L578, L577

  Theorem 5.204: Locked Prospective Falsification Run Soundness                                                    L849

  Theorem 5.9: Higher-Rank Exact Docking Lies Above the Ground State                                               L109

  Theorem 5.180: Binary Summary Rank Monotonicity for Docking                                                      L825

  Theorem 5.28: MD-Class Rank-Calibration Interface for Crooks Standard Form                                       L580

  Theorem 5.188: Per-System Concrete-Physics Witness Discharge Bundle                                              L833

  Theorem 5.182: Detailed-Balance Equilibrium Path-Ratio Unity for Docking                                         L827

  Theorem 5.8: Binary-Encoded Finite Grid Docking Carries Landauer Cost                                            L114

  Theorem 5.7: Exact Docking Structural Rank Controls Resolution Cost                                              L111

  Theorem 5.184: Docking Equilibrium $K\_d$ Prediction from Independent Rank Lower Bound                           L829

  Theorem 5.187: Exact-LJ Physical Witness Bundle for Docking                                                      L832

  Theorem 5.194: Hard-Threshold Falsification Logic (Not-Falsified Equivalence)                                    L839

  Theorem 5.197: High-Confidence Protocol Fail-Condition Bundle                                                    L842

  Theorem 5.193: Risky Prediction Bundle from Independent Relevance Certificate                                    L838

  Theorem 5.186: Independent-Rank Risky Empirical Prediction Bundle                                                L831

  Theorem 5.192: Independent Docking Rank Interval Certificates                                                    L837

  Theorem 5.178: Native-Coordinate Rank Identity for Exact Docking                                                 L823

  Theorem 5.185: Necessary Contact-Shell Geometry Budget from Independent Rank Lower Bound                         L830

  Theorem 5.201: Per-Case Real-Artifact Witness Discharge for All Targets                                          L846

  Theorem 5.179: Physical $3(N-k)$ Docking-Rank Budget                                                             L824

  Theorem 5.177: Physical Utility Interface for Exact Docking                                                      L822

  Theorem 5.195: Pre-Registered Independent-Rank Protocol Soundness                                                L840

  Theorem 5.191: Docking Rank-to-$K\_d$ Bound from Measurement-Free Partition Chain                                L836

  Theorem 5.183: Resolver-Free Equilibrium Bundle: Free-Energy Floor + Path-Ratio Unity                            L828

  Theorem 3.31: Constructive Canonical Truncation Measurability                                                    L588

  Theorem 3.28: Canonical Finite-Horizon Measurable-Transition Process Endpoint                                    L566, L565, L567, L569, L568

  Corollary 3.34: Concrete Measurable Transition-Kernel Jarzynski Endpoint                                         L542

  Theorem 3.33: Concrete Measurable Transition-Kernel Instantiation                                                L535, L536, L530, L532, L531

  Corollary 3.35: Concrete Measurable Transition-Kernel Path-Measure Jarzynski Endpoint                            L549

  Theorem 3.30: Canonical Truncation-Witness Instantiation                                                         L576

  Theorem 3.29: Projective-Consistency and Kolmogorov-Extension Interface Endpoint                                 L570, L571, L572, L574, L575, L573

  Theorem 3.19: Measure-Kernel Detailed-Balance/Stationarity Transport                                             L510, L512, L511, L508, L509, L513

  Theorem 3.22: Measure-Kernel Quotient Calculus and RG Endpoint                                                   L557, L556, L552, L553, L554, L550, L551, L555

  Corollary 3.23: Transport Theorems Recovered as Quotient-Calculus Instances                                      L559, L558

  Theorem 5.132: Mechanistic Uniform OOD Transfer Bound                                                            L777

  Theorem 4.17: Mechanochemical Coupling Collapse                                                                  L133

  Theorem 5.170: Microscopic Extension-Interface Scope Bundle                                                      L815

  Theorem 3.39: Minimum Physical Bit Operations                                                                    BA5, BA6, L43

  Theorem 5.140: Mixing/Autocorrelation Trajectory Correction Closure                                              L785

  Theorem 5.143: Model-Dependent Minimax Optimality Bundle                                                         L788

  Theorem 5.133: Model-Dependent Finite-Sample Margin Transport                                                    L778

  Theorem 4.6: Cutoff-Local Structural-Rank Bound for Exact Docking                                                L66

  Theorem 4.21: Multiplicative Separability Eliminates Conformer Dependence                                        L84

  Theorem 5.36: Nontrivial Biomolecular Shell-Constant Transport                                                   L632, L631

  Theorem 5.171: Numerical-Stack-Derived Simulator Control Flags                                                   L816

  Theorem 3.40: Decision-Class Bound                                                                               IT4, L43

  Theorem 2.8: One Transition, One Bit                                                                             BA4

  Theorem 5.120: OOD Guarantees from Explicit Transfer Calibration Components                                      L760

  Theorem 3.42: Nonbinary Optimizer-Richness Interface (Alphabet and VC Form)                                      L584, L585

  Theorem 3.41: Optimizer-Class Richness Forces Structural-Rank Floor                                              L418, L417

  Theorem 5.39: Pairwise Geometric Transport from Explicit Analytic Bounds                                         L661, L660

  Theorem 5.38: Pairwise Geometric Force-Field Sharp-Constant Transport                                            L653, L652

  Theorem 5.292: Paper4 TUR/Shadow Interface Discharge Extensions                                                  L670, L668, L669

  Theorem 5.293: Full-Support Boolean Resolution of the Stochastic-Relevance Conjecture                            L671

  Theorem 5.297: Explicit One-Step Dynamics Discharge of Unrestricted Stochastic-Relevance Closure                 L690

  Theorem 5.296: Primitive-Dynamics Discharge of Unrestricted Stochastic-Relevance Closure                         L682

  Theorem 5.295: Unrestricted-Distribution Stochastic-Relevance Closure under Support Transport                    L674

  Theorem 5.294: General-Distribution Progress for Stochastic-Relevance Equivalence                                L672, L673

  Theorem 5.298: Explicit-Step to Support-Transport Reduction for Unrestricted Stochastic-Relevance Closure        L699, L698

  Theorem 5.291: Paper4 Witness-Chain Import Endpoint                                                              L649

  Theorem 3.9: Parametric Fisher Identifiable Dimension                                                            L410, L411, L409

  Theorem 5.202: Numerical Partition+Correction Closure to Equilibrium $K\_d$ Bound                                L847

  Theorem 5.189: Partition-Ratio Driving Floor with Correction Margin                                              L834

  Theorem 3.26: Path-Measure Jarzynski Integral Endpoint                                                           L543, L545, L544, L546, L548, L547

  Theorem 3.27: Process-Level Path-Measure Transport Endpoint                                                      L560, L562, L564, L563

  Theorem 3.24: Path-Space Crooks Lift for Kernel Powers and Transport                                             L526, L525, L533, L527, L529, L534, L528

  Theorem 3.25: Expectation-Level Jarzynski Lift for Kernel Powers and Transport                                   L538, L537, L539, L541, L540

  Theorem 5.24: Pathwise Structural-Rank Energy Lower Bound                                                        L123

  Theorem 5.150: Potential-Landscape Barrier Kinetics Bundle                                                       L795

  Theorem 5.157: Pre-Registered Prospective Strong-Baseline Superiority Bundle                                     L802

  Theorem 5.203: Production Independent-$\mathrm{srank}$ Extractor Certificate Bundle                              L848

  Theorem 5.289: Prospective Empirical Closure under Blinded Benchmarking                                          L749, L748

  Theorem 5.49: QM-Grounded Electronic Half-Gap Transport                                                          L751

  Theorem 5.48: QM-Grounded Electronic-Structure Transport with Tight Class Bounds                                 L735, L734, L737, L736

  Theorem 5.50: Method-Specific QM Electronic Transport from Affine Class Calibration                              L756

  Theorem 5.51: Protocol-Derived QM Method Calibration Transport                                                   L765

  Theorem 5.128: Workflow-Specific QM Realism Transport from Component Budgets                                     L773

  Theorem 5.149: QM Workflow Transport from Benchmark Summary                                                      L794

  Theorem 5.131: Quantified Simulator Thermo/Kinetic Bundle with Control Inequalities                              L776

  Theorem 4.25: Quantitative Admissibility Rank-Collapse Law                                                       L125, L124

  Theorem 5.12: Quotient Resolution Speed Bound                                                                    L130

  Theorem 5.25: Quotient-Trajectory Crooks Law                                                                     L128

  Theorem 5.31: Quotient-Trajectory Dissipation Lower Bound                                                        L129

  Theorem 5.26: Rank-Calibrated Crooks Standard Form                                                               L423

  Theorem 6.3: Rank Identification                                                                                 L43, L46, L51

  Theorem 5.4: Rank-One Ground State                                                                               BA8, L54, L55

  Theorem 5.223: Raw Pocket+Ligand Benchmark Solver Bundle                                                         L868

  Theorem 5.220: Raw Pocket+Ligand Constructor to Sampled Posterior Bundle                                         L865

  Theorem 5.40: Solvent/Polarization/Long-Range Realism-Augmented Force-Field Transport                            L676, L675

  Theorem 5.136: Realistic Non-Constant Molecular SDE Endpoint Bundle                                              L781

  Theorem 5.35: Reference Biomolecular Zero-Shell Calibration                                                      L621, L622, L620

  Theorem 4.19: Renormalized Admissibility Equivalence                                                             L135

  Theorem 4.41: Resolution-Controlled Approximation Preserves Exact Winners                                        L99

  Theorem 4.40: Resolution-Controlled Approximation Gives Uniform Utility Approximation                            L91

  Theorem 2.9: Resolution Requires a Sufficient Coordinate Set                                                     BA5

  Theorem 5.129: Reversible Chemical Dynamics with Stationarity and Detailed Balance                               L774

  Theorem 5.218: RMSD-Probability-Derived Pose Solver Bundle                                                       L863

  Theorem 5.216: RMSD-Success Probability Unit-Interval Law                                                        L861

  Theorem 4.22: Sampled Exact-Coarse Winner Preservation                                                           L67

  Theorem 5.10: Inside-Cutoff Sampled Exact Resolution Gives a Concrete Bounded-Region Energy Floor                L112

  Theorem 4.35: Inside-Cutoff Sufficiency for Sampled Docking                                                      L68

  Theorem 5.209: Single-State Partition Positivity Bundle                                                          L854

  Theorem 5.207: Single-Target Full Physical-Closure Instance Bundle                                               L852

  Theorem 5.212: Single-Target Zero-Rank Concrete Closure Instance Bundle                                          L857

  Theorem 5.152: Spectral-Gap Absolute Free-Energy Total-Error Closure                                             L797

  Theorem 5.151: Spectral-Gap Trajectory Concentration Bundle                                                      L796

  Theorem 5.75: Spin-$\tfrac12$ Canonical-Decision Instance                                                        L594, L593

  Theorem 5.22: Spin-$\tfrac12$ Concrete Quantum Instantiation                                                     L594, L595, L593

  Theorem 5.76: Spin-$\tfrac12$ Decoherence Thermodynamic Floor                                                    L595

  Theorem 5.176: Store-Backed Attested Concrete No-Credible-Dismissal Corollary                                    L821

  Theorem 5.175: Store-Backed Attested Concrete External-Validation Three-Way Bundle                               L820

  Theorem 4.20: Strict Dominance Makes Coordinates Irrelevant                                                      L85

  Theorem 5.206: Target-Class Chemistry-Completeness to Final $K\_d$ Intervals                                     L851

  Theorem 6.5: Thermodynamic Selection                                                                             BA8, L49, L54, L55

  Theorem 5.11: Exact-Resolution Time Lower Bound                                                                  BA1, BA2, BA5, BA6, L43

  Theorem 4.72: Constructive Top-Level Export Computability Endpoint                                               L596

  Theorem 5.77: Constructive Top-Level Computable Path                                                             L596

  Theorem 4.37: Exact Top-k Ambiguity-Band Containment                                                             L72

  Theorem 4.36: Top-k Preservation Under Boundary Gap                                                              L79

  Theorem 4.56: Top-k Margin Certificates Are Sound                                                                L94

  Theorem 5.217: Top-$k$ Posterior Mass Lower-Bounds RMSD Success Probability under Coverage                       L862

  Theorem 6.4: Tractable Sufficiency at Rank One                                                                   L47, L53, L56

  Theorem 5.130: Trajectory-Estimator Absolute Free-Energy Correction Closure                                      L775

  Theorem 5.16: Trajectory Time--Energy Tradeoff                                                                   L131

  Theorem 5.93: Unified Chemical-State Dynamics Transport under pH/Ionic Conditions                                L739, L738

  Theorem 5.58: Unified Langevin Assumption-Bundle Discharge                                                       L619

  Theorem 5.290: Unified Physical/OOD/Prospective Closure Bundle                                                   L752

  Theorem 5.141: Unified Simulator Error-Analysis Controlled Thermo/Kinetic Bundle                                 L786

  Theorem 5.117: Unified Simulator-Derived Thermodynamic/Kinetic Bundle                                            L759

  Theorem 5.118: Unified Thermodynamic/Kinetic Bundle from One Physical Model                                      L743

  Theorem 5.119: Universality/OOD Calibrated Error Bounds                                                          L744, L745

  Theorem 5.211: Upper-Only Independent-Srank Extractor Interval Bundle                                            L856

  Theorem 4.69: Velocity-Verlet Preserves Phase-Space Volume                                                       L86

  Theorem 5.210: Zero-Correction Calibration Closure Bundle                                                        L855
  -----------------------------------------------------------------------------------------------------------------------------------------------------------------

*Auto summary: mapped 453/453 (full=453, derived=0, unmapped=0).*


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

**`L118`**[]{#lh:L118} Leverage/DockingTheoryBridge.lean

**`L119`**[]{#lh:L119} Leverage/DockingTheoryBridge.lean

**`L120`**[]{#lh:L120} Leverage/DockingTheoryBridge.lean

**`L121`**[]{#lh:L121} Leverage/DockingTheoryBridge.lean

**`L122`**[]{#lh:L122} Leverage/DockingTheoryBridge.lean

**`L123`**[]{#lh:L123} Leverage/DockingTheoryBridge.lean

**`L124`**[]{#lh:L124} Leverage/DockingTheoryBridge.lean

**`L125`**[]{#lh:L125} Leverage/DockingTheoryBridge.lean

**`L126`**[]{#lh:L126} Leverage/DockingTheoryBridge.lean

**`L127`**[]{#lh:L127} Leverage/DockingTheoryBridge.lean

**`L128`**[]{#lh:L128} Leverage/DockingTheoryBridge.lean

**`L129`**[]{#lh:L129} Leverage/DockingTheoryBridge.lean

**`L130`**[]{#lh:L130} Leverage/DockingTheoryBridge.lean

**`L131`**[]{#lh:L131} Leverage/DockingTheoryBridge.lean

**`L132`**[]{#lh:L132} Leverage/ProteinMechanicalGraph.lean

**`L133`**[]{#lh:L133} Leverage/DockingTheoryBridge.lean

**`L134`**[]{#lh:L134} Leverage/DockingTheoryBridge.lean

**`L135`**[]{#lh:L135} Leverage/DockingTheoryBridge.lean

**`L136`**[]{#lh:L136} Leverage/DockingTheoryBridge.lean

**`L137`**[]{#lh:L137} Leverage/DockingTheoryBridge.lean

**`L138`**[]{#lh:L138} Leverage/DockingTheoryBridge.lean

**`L139`**[]{#lh:L139} Leverage/DockingTheoryBridge.lean

**`L140`**[]{#lh:L140} Leverage/ProteinMechanicalGraph.lean

**`L141`**[]{#lh:L141} Leverage/ProteinMechanicalGraph.lean

**`L142`**[]{#lh:L142} Leverage/ProteinMechanicalGraph.lean

**`L394`**[]{#lh:L394} Leverage/DockingTheoryBridge.lean

**`L395`**[]{#lh:L395} Leverage/DockingTheoryBridge.lean

**`L396`**[]{#lh:L396} Leverage/BridgeToDQ.lean

**`L397`**[]{#lh:L397} Leverage/BridgeToDQ.lean

**`L398`**[]{#lh:L398} Leverage/BridgeToDQ.lean

**`L403`**[]{#lh:L403} Leverage/BridgeToDQ.lean

**`L404`**[]{#lh:L404} Leverage/BridgeToDQ.lean

**`L405`**[]{#lh:L405} Leverage/BridgeToDQ.lean

**`L406`**[]{#lh:L406} Leverage/BridgeToDQ.lean

**`L407`**[]{#lh:L407} Leverage/BridgeToDQ.lean

**`L408`**[]{#lh:L408} Leverage/DockingTheoryBridge.lean

**`L409`**[]{#lh:L409} Leverage/DockingTheoryBridge.lean

**`L410`**[]{#lh:L410} Leverage/DockingTheoryBridge.lean

**`L411`**[]{#lh:L411} Leverage/DockingTheoryBridge.lean

**`L412`**[]{#lh:L412} Leverage/DockingTheoryBridge.lean

**`L413`**[]{#lh:L413} Leverage/BridgeToDQ.lean

**`L414`**[]{#lh:L414} Leverage/BridgeToDQ.lean

**`L415`**[]{#lh:L415} Leverage/DockingTheoryBridge.lean

**`L416`**[]{#lh:L416} Leverage/DockingTheoryBridge.lean

**`L417`**[]{#lh:L417} Leverage/DockingTheoryBridge.lean

**`L418`**[]{#lh:L418} Leverage/DockingTheoryBridge.lean

**`L419`**[]{#lh:L419} Leverage/DockingTheoryBridge.lean

**`L420`**[]{#lh:L420} Leverage/DockingTheoryBridge.lean

**`L421`**[]{#lh:L421} Leverage/DockingTheoryBridge.lean

**`L422`**[]{#lh:L422} Leverage/DockingTheoryBridge.lean

**`L423`**[]{#lh:L423} Leverage/DockingTheoryBridge.lean

**`L424`**[]{#lh:L424} Leverage/DockingTheoryBridge.lean

**`L425`**[]{#lh:L425} Leverage/DockingTheoryBridge.lean

**`L426`**[]{#lh:L426} Leverage/DockingTheoryBridge.lean

**`L427`**[]{#lh:L427} Leverage/DockingTheoryBridge.lean

**`L428`**[]{#lh:L428} Leverage/ProteinMechanicalGraph.lean

**`L429`**[]{#lh:L429} Leverage/ProteinMechanicalGraph.lean

**`L430`**[]{#lh:L430} Leverage/ProteinMechanicalGraph.lean

**`L431`**[]{#lh:L431} Leverage/ProteinMechanicalGraph.lean

**`L432`**[]{#lh:L432} Leverage/ProteinMechanicalGraph.lean

**`L433`**[]{#lh:L433} Leverage/ProteinMechanicalGraph.lean

**`L434`**[]{#lh:L434} Leverage/DockingTheoryBridge.lean

**`L435`**[]{#lh:L435} Leverage/DockingTheoryBridge.lean

**`L436`**[]{#lh:L436} Leverage/DockingTheoryBridge.lean

**`L437`**[]{#lh:L437} Leverage/ProteinMechanicalGraph.lean

**`L438`**[]{#lh:L438} Leverage/ProteinMechanicalGraph.lean

**`L439`**[]{#lh:L439} Leverage/ProteinMechanicalGraph.lean

**`L440`**[]{#lh:L440} Leverage/ProteinMechanicalGraph.lean

**`L441`**[]{#lh:L441} Leverage/ProteinMechanicalGraph.lean

**`L442`**[]{#lh:L442} Leverage/ProteinMechanicalGraph.lean

**`L443`**[]{#lh:L443} Leverage/ProteinMechanicalGraph.lean

**`L444`**[]{#lh:L444} Leverage/ProteinMechanicalGraph.lean

**`L446`**[]{#lh:L446} Leverage/DockingTheoryBridge.lean

**`L447`**[]{#lh:L447} Leverage/DockingTheoryBridge.lean

**`L448`**[]{#lh:L448} Leverage/DockingTheoryBridge.lean

**`L449`**[]{#lh:L449}

**`L450`**[]{#lh:L450}

**`L451`**[]{#lh:L451} Leverage/DockingTheoryBridge.lean

**`L452`**[]{#lh:L452} Leverage/DockingTheoryBridge.lean

**`L453`**[]{#lh:L453} Leverage/DockingTheoryBridge.lean

**`L454`**[]{#lh:L454} Leverage/DockingTheoryBridge.lean

**`L455`**[]{#lh:L455} Leverage/DockingTheoryBridge.lean

**`L456`**[]{#lh:L456} Leverage/DockingTheoryBridge.lean

**`L457`**[]{#lh:L457} Leverage/DockingTheoryBridge.lean

**`L458`**[]{#lh:L458} Leverage/DockingTheoryBridge.lean

**`L459`**[]{#lh:L459} Leverage/DockingTheoryBridge.lean

**`L460`**[]{#lh:L460} Leverage/DockingTheoryBridge.lean

**`L461`**[]{#lh:L461} Leverage/DockingTheoryBridge.lean

**`L462`**[]{#lh:L462} Leverage/DockingTheoryBridge.lean

**`L463`**[]{#lh:L463} Leverage/DockingTheoryBridge.lean

**`L464`**[]{#lh:L464} Leverage/DockingTheoryBridge.lean

**`L465`**[]{#lh:L465} Leverage/DockingTheoryBridge.lean

**`L466`**[]{#lh:L466} Leverage/DockingTheoryBridge.lean

**`L467`**[]{#lh:L467} Leverage/DockingTheoryBridge.lean

**`L468`**[]{#lh:L468} Leverage/DockingTheoryBridge.lean

**`L469`**[]{#lh:L469} Leverage/DockingTheoryBridge.lean

**`L470`**[]{#lh:L470} Leverage/DockingTheoryBridge.lean

**`L471`**[]{#lh:L471} Leverage/DockingTheoryBridge.lean

**`L472`**[]{#lh:L472} Leverage/DockingTheoryBridge.lean

**`L473`**[]{#lh:L473} Leverage/DockingTheoryBridge.lean

**`L474`**[]{#lh:L474} Leverage/DockingTheoryBridge.lean

**`L475`**[]{#lh:L475} Leverage/DockingTheoryBridge.lean

**`L476`**[]{#lh:L476} Leverage/DockingTheoryBridge.lean

**`L477`**[]{#lh:L477} Leverage/DockingTheoryBridge.lean

**`L478`**[]{#lh:L478} Leverage/DockingTheoryBridge.lean

**`L479`**[]{#lh:L479} Leverage/DockingTheoryBridge.lean

**`L480`**[]{#lh:L480} Leverage/DockingTheoryBridge.lean

**`L481`**[]{#lh:L481} Leverage/DockingTheoryBridge.lean

**`L482`**[]{#lh:L482} Leverage/DockingTheoryBridge.lean

**`L483`**[]{#lh:L483} Leverage/DockingTheoryBridge.lean

**`L484`**[]{#lh:L484} Leverage/DockingTheoryBridge.lean

**`L485`**[]{#lh:L485} Leverage/DockingTheoryBridge.lean

**`L486`**[]{#lh:L486} Leverage/DockingTheoryBridge.lean

**`L487`**[]{#lh:L487} Leverage/BridgeToDQ.lean

**`L488`**[]{#lh:L488} Leverage/BridgeToDQ.lean

**`L489`**[]{#lh:L489} Leverage/BridgeToDQ.lean

**`L490`**[]{#lh:L490} Leverage/BridgeToDQ.lean

**`L491`**[]{#lh:L491} Leverage/BridgeToDQ.lean

**`L492`**[]{#lh:L492} Leverage/BridgeToDQ.lean

**`L493`**[]{#lh:L493} Leverage/BridgeToDQ.lean

**`L494`**[]{#lh:L494} Leverage/BridgeToDQ.lean

**`L495`**[]{#lh:L495} Leverage/BridgeToDQ.lean

**`L496`**[]{#lh:L496} Leverage/BridgeToDQ.lean

**`L497`**[]{#lh:L497} Leverage/BridgeToDQ.lean

**`L498`**[]{#lh:L498} Leverage/BridgeToDQ.lean

**`L499`**[]{#lh:L499} Leverage/BridgeToDQ.lean

**`L500`**[]{#lh:L500} Leverage/BridgeToDQ.lean

**`L501`**[]{#lh:L501} Leverage/BridgeToDQ.lean

**`L502`**[]{#lh:L502}

**`L503`**[]{#lh:L503} Leverage/BridgeToDQ.lean

**`L504`**[]{#lh:L504} Leverage/BridgeToDQ.lean

**`L505`**[]{#lh:L505} Leverage/BridgeToDQ.lean

**`L506`**[]{#lh:L506} Leverage/BridgeToDQ.lean

**`L507`**[]{#lh:L507} Leverage/BridgeToDQ.lean

**`L508`**[]{#lh:L508}

**`L509`**[]{#lh:L509}

**`L510`**[]{#lh:L510}

**`L511`**[]{#lh:L511}

**`L512`**[]{#lh:L512}

**`L513`**[]{#lh:L513} Leverage/BridgeToDQ.lean

**`L514`**[]{#lh:L514} Leverage/BridgeToDQ.lean

**`L515`**[]{#lh:L515} Leverage/BridgeToDQ.lean

**`L516`**[]{#lh:L516} Leverage/BridgeToDQ.lean

**`L517`**[]{#lh:L517} Leverage/BridgeToDQ.lean

**`L518`**[]{#lh:L518} Leverage/BridgeToDQ.lean

**`L519`**[]{#lh:L519}

**`L520`**[]{#lh:L520}

**`L521`**[]{#lh:L521}

**`L522`**[]{#lh:L522}

**`L523`**[]{#lh:L523}

**`L524`**[]{#lh:L524}

**`L525`**[]{#lh:L525} Leverage/BridgeToDQ.lean

**`L526`**[]{#lh:L526} Leverage/BridgeToDQ.lean

**`L527`**[]{#lh:L527} Leverage/BridgeToDQ.lean

**`L528`**[]{#lh:L528} Leverage/BridgeToDQ.lean

**`L529`**[]{#lh:L529} Leverage/BridgeToDQ.lean

**`L530`**[]{#lh:L530} Leverage/BridgeToDQ.lean

**`L531`**[]{#lh:L531} Leverage/BridgeToDQ.lean

**`L532`**[]{#lh:L532} Leverage/BridgeToDQ.lean

**`L533`**[]{#lh:L533}

**`L534`**[]{#lh:L534}

**`L535`**[]{#lh:L535} Leverage/BridgeToDQ.lean

**`L536`**[]{#lh:L536} Leverage/BridgeToDQ.lean

**`L537`**[]{#lh:L537} Leverage/BridgeToDQ.lean

**`L538`**[]{#lh:L538} Leverage/BridgeToDQ.lean

**`L539`**[]{#lh:L539} Leverage/BridgeToDQ.lean

**`L540`**[]{#lh:L540} Leverage/BridgeToDQ.lean

**`L541`**[]{#lh:L541} Leverage/BridgeToDQ.lean

**`L542`**[]{#lh:L542} Leverage/BridgeToDQ.lean

**`L543`**[]{#lh:L543} Leverage/BridgeToDQ.lean

**`L544`**[]{#lh:L544} Leverage/BridgeToDQ.lean

**`L545`**[]{#lh:L545} Leverage/BridgeToDQ.lean

**`L546`**[]{#lh:L546} Leverage/BridgeToDQ.lean

**`L547`**[]{#lh:L547} Leverage/BridgeToDQ.lean

**`L548`**[]{#lh:L548} Leverage/BridgeToDQ.lean

**`L549`**[]{#lh:L549} Leverage/BridgeToDQ.lean

**`L550`**[]{#lh:L550} Leverage/BridgeToDQ.lean

**`L551`**[]{#lh:L551} Leverage/BridgeToDQ.lean

**`L552`**[]{#lh:L552} Leverage/BridgeToDQ.lean

**`L553`**[]{#lh:L553} Leverage/BridgeToDQ.lean

**`L554`**[]{#lh:L554} Leverage/BridgeToDQ.lean

**`L555`**[]{#lh:L555} Leverage/BridgeToDQ.lean

**`L556`**[]{#lh:L556} Leverage/BridgeToDQ.lean

**`L557`**[]{#lh:L557} Leverage/BridgeToDQ.lean

**`L558`**[]{#lh:L558} Leverage/BridgeToDQ.lean

**`L559`**[]{#lh:L559} Leverage/BridgeToDQ.lean

**`L560`**[]{#lh:L560} Leverage/BridgeToDQ.lean

**`L562`**[]{#lh:L562} Leverage/BridgeToDQ.lean

**`L563`**[]{#lh:L563} Leverage/BridgeToDQ.lean

**`L564`**[]{#lh:L564} Leverage/BridgeToDQ.lean

**`L565`**[]{#lh:L565} Leverage/BridgeToDQ.lean

**`L566`**[]{#lh:L566} Leverage/BridgeToDQ.lean

**`L567`**[]{#lh:L567} Leverage/BridgeToDQ.lean

**`L568`**[]{#lh:L568} Leverage/BridgeToDQ.lean

**`L569`**[]{#lh:L569} Leverage/BridgeToDQ.lean

**`L570`**[]{#lh:L570} Leverage/BridgeToDQ.lean

**`L571`**[]{#lh:L571} Leverage/BridgeToDQ.lean

**`L572`**[]{#lh:L572} Leverage/BridgeToDQ.lean

**`L573`**[]{#lh:L573} Leverage/BridgeToDQ.lean

**`L574`**[]{#lh:L574} Leverage/BridgeToDQ.lean

**`L575`**[]{#lh:L575} Leverage/BridgeToDQ.lean

**`L576`**[]{#lh:L576} Leverage/BridgeToDQ.lean

**`L577`**[]{#lh:L577}

**`L578`**[]{#lh:L578}

**`L579`**[]{#lh:L579}

**`L580`**[]{#lh:L580}

**`L581`**[]{#lh:L581}

**`L582`**[]{#lh:L582}

**`L583`**[]{#lh:L583}

**`L584`**[]{#lh:L584}

**`L585`**[]{#lh:L585}

**`L586`**[]{#lh:L586} Leverage/DockingTheoryBridge.lean

**`L587`**[]{#lh:L587}

**`L588`**[]{#lh:L588} Leverage/BridgeToDQ.lean

**`L589`**[]{#lh:L589} Leverage/DockingTheoryBridge.lean

**`L590`**[]{#lh:L590} Leverage/DockingTheoryBridge.lean

**`L591`**[]{#lh:L591} Leverage/DockingTheoryBridge.lean

**`L592`**[]{#lh:L592} Leverage/DockingTheoryBridge.lean

**`L593`**[]{#lh:L593} Leverage/DockingTheoryBridge.lean

**`L594`**[]{#lh:L594} Leverage/DockingTheoryBridge.lean

**`L595`**[]{#lh:L595} Leverage/DockingTheoryBridge.lean

**`L596`**[]{#lh:L596} Leverage/DockingTheoryBridge.lean

**`L597`**[]{#lh:L597}

**`L598`**[]{#lh:L598}

**`L599`**[]{#lh:L599} Leverage/DockingTheoryBridge.lean

**`L600`**[]{#lh:L600} Leverage/DockingTheoryBridge.lean

**`L601`**[]{#lh:L601} Leverage/DockingTheoryBridge.lean

**`L602`**[]{#lh:L602} Leverage/DockingTheoryBridge.lean

**`L603`**[]{#lh:L603} Leverage/DockingTheoryBridge.lean

**`L604`**[]{#lh:L604} Leverage/DockingTheoryBridge.lean

**`L605`**[]{#lh:L605} Leverage/DockingTheoryBridge.lean

**`L606`**[]{#lh:L606} Leverage/DockingTheoryBridge.lean

**`L607`**[]{#lh:L607} Leverage/DockingTheoryBridge.lean

**`L608`**[]{#lh:L608} Leverage/DockingTheoryBridge.lean

**`L609`**[]{#lh:L609} Leverage/DockingTheoryBridge.lean

**`L610`**[]{#lh:L610} Leverage/DockingTheoryBridge.lean

**`L611`**[]{#lh:L611} Leverage/DockingTheoryBridge.lean

**`L612`**[]{#lh:L612} Leverage/DockingTheoryBridge.lean

**`L613`**[]{#lh:L613} Leverage/DockingTheoryBridge.lean

**`L614`**[]{#lh:L614} Leverage/DockingTheoryBridge.lean

**`L615`**[]{#lh:L615} Leverage/DockingTheoryBridge.lean

**`L616`**[]{#lh:L616} Leverage/DockingTheoryBridge.lean

**`L617`**[]{#lh:L617} Leverage/DockingTheoryBridge.lean

**`L618`**[]{#lh:L618} Leverage/DockingTheoryBridge.lean

**`L619`**[]{#lh:L619} Leverage/DockingTheoryBridge.lean

**`L620`**[]{#lh:L620} Leverage/DockingTheoryBridge.lean

**`L621`**[]{#lh:L621} Leverage/DockingTheoryBridge.lean

**`L622`**[]{#lh:L622} Leverage/DockingTheoryBridge.lean

**`L623`**[]{#lh:L623} Leverage/DockingTheoryBridge.lean

**`L624`**[]{#lh:L624} Leverage/DockingTheoryBridge.lean

**`L625`**[]{#lh:L625} Leverage/DockingTheoryBridge.lean

**`L626`**[]{#lh:L626} Leverage/DockingTheoryBridge.lean

**`L627`**[]{#lh:L627} Leverage/DockingTheoryBridge.lean

**`L628`**[]{#lh:L628} Leverage/DockingTheoryBridge.lean

**`L629`**[]{#lh:L629} Leverage/DockingTheoryBridge.lean

**`L630`**[]{#lh:L630} Leverage/DockingTheoryBridge.lean

**`L631`**[]{#lh:L631} Leverage/DockingTheoryBridge.lean

**`L632`**[]{#lh:L632} Leverage/DockingTheoryBridge.lean

**`L633`**[]{#lh:L633} Leverage/DockingTheoryBridge.lean

**`L634`**[]{#lh:L634} Leverage/DockingTheoryBridge.lean

**`L635`**[]{#lh:L635} Leverage/DockingTheoryBridge.lean

**`L636`**[]{#lh:L636} Leverage/DockingTheoryBridge.lean

**`L637`**[]{#lh:L637}

**`L638`**[]{#lh:L638}

**`L639`**[]{#lh:L639}

**`L640`**[]{#lh:L640} Leverage/DockingTheoryBridge.lean

**`L641`**[]{#lh:L641} Leverage/DockingTheoryBridge.lean

**`L642`**[]{#lh:L642} Leverage/DockingTheoryBridge.lean

**`L643`**[]{#lh:L643} Leverage/DockingTheoryBridge.lean

**`L644`**[]{#lh:L644} Leverage/DockingTheoryBridge.lean

**`L645`**[]{#lh:L645} Leverage/DockingTheoryBridge.lean

**`L646`**[]{#lh:L646} Leverage/DockingTheoryBridge.lean

**`L647`**[]{#lh:L647} Leverage/DockingTheoryBridge.lean

**`L648`**[]{#lh:L648} Leverage/DockingTheoryBridge.lean

**`L649`**[]{#lh:L649} Leverage/DockingTheoryBridge.lean

**`L650`**[]{#lh:L650} Leverage/DockingTheoryBridge.lean

**`L651`**[]{#lh:L651}

**`L652`**[]{#lh:L652} Leverage/DockingTheoryBridge.lean

**`L653`**[]{#lh:L653} Leverage/DockingTheoryBridge.lean

**`L654`**[]{#lh:L654} Leverage/DockingTheoryBridge.lean

**`L655`**[]{#lh:L655} Leverage/DockingTheoryBridge.lean

**`L656`**[]{#lh:L656}

**`L657`**[]{#lh:L657} Leverage/DockingTheoryBridge.lean

**`L658`**[]{#lh:L658} Leverage/DockingTheoryBridge.lean

**`L659`**[]{#lh:L659} Leverage/DockingTheoryBridge.lean

**`L660`**[]{#lh:L660} Leverage/DockingTheoryBridge.lean

**`L661`**[]{#lh:L661} Leverage/DockingTheoryBridge.lean

**`L662`**[]{#lh:L662}

**`L663`**[]{#lh:L663}

**`L664`**[]{#lh:L664}

**`L665`**[]{#lh:L665} Leverage/DockingTheoryBridge.lean

**`L666`**[]{#lh:L666} Leverage/DockingTheoryBridge.lean

**`L667`**[]{#lh:L667}

**`L668`**[]{#lh:L668} Leverage/DockingTheoryBridge.lean

**`L669`**[]{#lh:L669} Leverage/DockingTheoryBridge.lean

**`L670`**[]{#lh:L670} Leverage/DockingTheoryBridge.lean

**`L671`**[]{#lh:L671} Leverage/DockingTheoryBridge.lean

**`L672`**[]{#lh:L672} Leverage/DockingTheoryBridge.lean

**`L673`**[]{#lh:L673} Leverage/DockingTheoryBridge.lean

**`L674`**[]{#lh:L674} Leverage/DockingTheoryBridge.lean

**`L675`**[]{#lh:L675} Leverage/DockingTheoryBridge.lean

**`L676`**[]{#lh:L676} Leverage/DockingTheoryBridge.lean

**`L677`**[]{#lh:L677} Leverage/DockingTheoryBridge.lean

**`L678`**[]{#lh:L678}

**`L679`**[]{#lh:L679} Leverage/DockingTheoryBridge.lean

**`L680`**[]{#lh:L680}

**`L681`**[]{#lh:L681} Leverage/DockingTheoryBridge.lean

**`L682`**[]{#lh:L682} Leverage/DockingTheoryBridge.lean

**`L683`**[]{#lh:L683}

**`L684`**[]{#lh:L684} Leverage/DockingTheoryBridge.lean

**`L685`**[]{#lh:L685} Leverage/DockingTheoryBridge.lean

**`L686`**[]{#lh:L686} Leverage/DockingTheoryBridge.lean

**`L687`**[]{#lh:L687} Leverage/DockingTheoryBridge.lean

**`L688`**[]{#lh:L688} Leverage/DockingTheoryBridge.lean

**`L689`**[]{#lh:L689} Leverage/DockingTheoryBridge.lean

**`L690`**[]{#lh:L690} Leverage/DockingTheoryBridge.lean

**`L691`**[]{#lh:L691} Leverage/DockingTheoryBridge.lean

**`L692`**[]{#lh:L692} Leverage/DockingTheoryBridge.lean

**`L693`**[]{#lh:L693} Leverage/DockingTheoryBridge.lean

**`L694`**[]{#lh:L694} Leverage/DockingTheoryBridge.lean

**`L695`**[]{#lh:L695} Leverage/DockingTheoryBridge.lean

**`L696`**[]{#lh:L696} Leverage/DockingTheoryBridge.lean

**`L697`**[]{#lh:L697} Leverage/DockingTheoryBridge.lean

**`L698`**[]{#lh:L698} Leverage/DockingTheoryBridge.lean

**`L699`**[]{#lh:L699} Leverage/DockingTheoryBridge.lean

**`L700`**[]{#lh:L700} Leverage/DockingTheoryBridge.lean

**`L701`**[]{#lh:L701} Leverage/DockingTheoryBridge.lean

**`L702`**[]{#lh:L702} Leverage/DockingTheoryBridge.lean

**`L703`**[]{#lh:L703} Leverage/DockingTheoryBridge.lean

**`L704`**[]{#lh:L704} Leverage/DockingTheoryBridge.lean

**`L705`**[]{#lh:L705}

**`L706`**[]{#lh:L706} Leverage/DockingTheoryBridge.lean

**`L707`**[]{#lh:L707} Leverage/DockingTheoryBridge.lean

**`L708`**[]{#lh:L708}

**`L709`**[]{#lh:L709} Leverage/DockingTheoryBridge.lean

**`L710`**[]{#lh:L710} Leverage/DockingTheoryBridge.lean

**`L711`**[]{#lh:L711} Leverage/DockingTheoryBridge.lean

**`L712`**[]{#lh:L712} Leverage/DockingTheoryBridge.lean

**`L713`**[]{#lh:L713} Leverage/DockingTheoryBridge.lean

**`L714`**[]{#lh:L714} Leverage/DockingTheoryBridge.lean

**`L715`**[]{#lh:L715} Leverage/DockingTheoryBridge.lean

**`L716`**[]{#lh:L716} Leverage/DockingTheoryBridge.lean

**`L717`**[]{#lh:L717} Leverage/DockingTheoryBridge.lean

**`L718`**[]{#lh:L718} Leverage/DockingTheoryBridge.lean

**`L719`**[]{#lh:L719} Leverage/DockingTheoryBridge.lean

**`L720`**[]{#lh:L720} Leverage/DockingTheoryBridge.lean

**`L721`**[]{#lh:L721} Leverage/DockingTheoryBridge.lean

**`L722`**[]{#lh:L722} Leverage/DockingTheoryBridge.lean

**`L723`**[]{#lh:L723} Leverage/DockingTheoryBridge.lean

**`L724`**[]{#lh:L724} Leverage/DockingTheoryBridge.lean

**`L725`**[]{#lh:L725} Leverage/DockingTheoryBridge.lean

**`L726`**[]{#lh:L726} Leverage/DockingTheoryBridge.lean

**`L727`**[]{#lh:L727} Leverage/DockingTheoryBridge.lean

**`L728`**[]{#lh:L728}

**`L729`**[]{#lh:L729} Leverage/DockingTheoryBridge.lean

**`L730`**[]{#lh:L730} Leverage/DockingTheoryBridge.lean

**`L731`**[]{#lh:L731} Leverage/DockingTheoryBridge.lean

**`L732`**[]{#lh:L732} Leverage/DockingTheoryBridge.lean

**`L733`**[]{#lh:L733} Leverage/DockingTheoryBridge.lean

**`L734`**[]{#lh:L734}

**`L735`**[]{#lh:L735}

**`L736`**[]{#lh:L736} Leverage/DockingTheoryBridge.lean

**`L737`**[]{#lh:L737} Leverage/DockingTheoryBridge.lean

**`L738`**[]{#lh:L738} Leverage/DockingTheoryBridge.lean

**`L739`**[]{#lh:L739}

**`L740`**[]{#lh:L740}

**`L741`**[]{#lh:L741}

**`L742`**[]{#lh:L742}

**`L743`**[]{#lh:L743}

**`L744`**[]{#lh:L744}

**`L745`**[]{#lh:L745}

**`L746`**[]{#lh:L746} Leverage/DockingTheoryBridge.lean

**`L747`**[]{#lh:L747} Leverage/DockingTheoryBridge.lean

**`L748`**[]{#lh:L748}

**`L749`**[]{#lh:L749}

**`L750`**[]{#lh:L750} Leverage/DockingTheoryBridge.lean

**`L751`**[]{#lh:L751} Leverage/DockingTheoryBridge.lean

**`L752`**[]{#lh:L752} Leverage/DockingTheoryBridge.lean

**`L753`**[]{#lh:L753} Leverage/DockingTheoryBridge.lean

**`L754`**[]{#lh:L754}

**`L755`**[]{#lh:L755} Leverage/DockingTheoryBridge.lean

**`L756`**[]{#lh:L756} Leverage/DockingTheoryBridge.lean

**`L757`**[]{#lh:L757}

**`L758`**[]{#lh:L758}

**`L759`**[]{#lh:L759}

**`L760`**[]{#lh:L760}

**`L761`**[]{#lh:L761} Leverage/DockingTheoryBridge.lean

**`L762`**[]{#lh:L762} Leverage/DockingTheoryBridge.lean

**`L763`**[]{#lh:L763} Leverage/DockingTheoryBridge.lean

**`L764`**[]{#lh:L764}

**`L765`**[]{#lh:L765} Leverage/DockingTheoryBridge.lean

**`L766`**[]{#lh:L766} Leverage/DockingTheoryBridge.lean

**`L767`**[]{#lh:L767}

**`L768`**[]{#lh:L768}

**`L769`**[]{#lh:L769} Leverage/DockingTheoryBridge.lean

**`L770`**[]{#lh:L770} Leverage/DockingTheoryBridge.lean

**`L771`**[]{#lh:L771} Leverage/DockingTheoryBridge.lean

**`L772`**[]{#lh:L772} Leverage/DockingTheoryBridge.lean

**`L773`**[]{#lh:L773} Leverage/DockingTheoryBridge.lean

**`L774`**[]{#lh:L774}

**`L775`**[]{#lh:L775}

**`L776`**[]{#lh:L776}

**`L777`**[]{#lh:L777}

**`L778`**[]{#lh:L778}

**`L779`**[]{#lh:L779} Leverage/DockingTheoryBridge.lean

**`L780`**[]{#lh:L780} Leverage/DockingTheoryBridge.lean

**`L781`**[]{#lh:L781} Leverage/DockingTheoryBridge.lean

**`L782`**[]{#lh:L782} Leverage/DockingTheoryBridge.lean

**`L783`**[]{#lh:L783} Leverage/DockingTheoryBridge.lean

**`L784`**[]{#lh:L784} Leverage/DockingTheoryBridge.lean

**`L785`**[]{#lh:L785} Leverage/DockingTheoryBridge.lean

**`L786`**[]{#lh:L786}

**`L787`**[]{#lh:L787} Leverage/DockingTheoryBridge.lean

**`L788`**[]{#lh:L788} Leverage/DockingTheoryBridge.lean

**`L789`**[]{#lh:L789}

**`L790`**[]{#lh:L790} Leverage/DockingTheoryBridge.lean

**`L791`**[]{#lh:L791} Leverage/DockingTheoryBridge.lean

**`L792`**[]{#lh:L792} Leverage/DockingTheoryBridge.lean

**`L793`**[]{#lh:L793} Leverage/DockingTheoryBridge.lean

**`L794`**[]{#lh:L794} Leverage/DockingTheoryBridge.lean

**`L795`**[]{#lh:L795} Leverage/DockingTheoryBridge.lean

**`L796`**[]{#lh:L796} Leverage/DockingTheoryBridge.lean

**`L797`**[]{#lh:L797} Leverage/DockingTheoryBridge.lean

**`L798`**[]{#lh:L798}

**`L799`**[]{#lh:L799} Leverage/DockingTheoryBridge.lean

**`L800`**[]{#lh:L800} Leverage/DockingTheoryBridge.lean

**`L801`**[]{#lh:L801} Leverage/DockingTheoryBridge.lean

**`L802`**[]{#lh:L802} Leverage/DockingTheoryBridge.lean

**`L803`**[]{#lh:L803} Leverage/DockingTheoryBridge.lean

**`L804`**[]{#lh:L804} Leverage/DockingTheoryBridge.lean

**`L805`**[]{#lh:L805} Leverage/DockingTheoryBridge.lean

**`L806`**[]{#lh:L806} Leverage/DockingTheoryBridge.lean

**`L807`**[]{#lh:L807}

**`L808`**[]{#lh:L808}

**`L809`**[]{#lh:L809}

**`L810`**[]{#lh:L810} Leverage/DockingTheoryBridge.lean

**`L811`**[]{#lh:L811} Leverage/DockingTheoryBridge.lean

**`L812`**[]{#lh:L812} Leverage/DockingTheoryBridge.lean

**`L813`**[]{#lh:L813} Leverage/DockingTheoryBridge.lean

**`L814`**[]{#lh:L814} Leverage/DockingTheoryBridge.lean

**`L815`**[]{#lh:L815} Leverage/DockingTheoryBridge.lean

**`L816`**[]{#lh:L816}

**`L817`**[]{#lh:L817} Leverage/DockingTheoryBridge.lean

**`L818`**[]{#lh:L818} Leverage/DockingTheoryBridge.lean

**`L819`**[]{#lh:L819} Leverage/DockingTheoryBridge.lean

**`L820`**[]{#lh:L820} Leverage/DockingTheoryBridge.lean

**`L821`**[]{#lh:L821} Leverage/DockingTheoryBridge.lean

**`L822`**[]{#lh:L822} Leverage/DockingTheoryBridge.lean

**`L823`**[]{#lh:L823} Leverage/DockingTheoryBridge.lean

**`L824`**[]{#lh:L824} Leverage/DockingTheoryBridge.lean

**`L825`**[]{#lh:L825} Leverage/DockingTheoryBridge.lean

**`L826`**[]{#lh:L826} Leverage/DockingTheoryBridge.lean

**`L827`**[]{#lh:L827} Leverage/DockingTheoryBridge.lean

**`L828`**[]{#lh:L828} Leverage/DockingTheoryBridge.lean

**`L829`**[]{#lh:L829} Leverage/DockingTheoryBridge.lean

**`L830`**[]{#lh:L830} Leverage/DockingTheoryBridge.lean

**`L831`**[]{#lh:L831} Leverage/DockingTheoryBridge.lean

**`L832`**[]{#lh:L832} Leverage/DockingTheoryBridge.lean

**`L833`**[]{#lh:L833}

**`L834`**[]{#lh:L834} Leverage/DockingTheoryBridge.lean

**`L835`**[]{#lh:L835} Leverage/DockingTheoryBridge.lean

**`L836`**[]{#lh:L836} Leverage/DockingTheoryBridge.lean

**`L837`**[]{#lh:L837} Leverage/DockingTheoryBridge.lean

**`L838`**[]{#lh:L838} Leverage/DockingTheoryBridge.lean

**`L839`**[]{#lh:L839}

**`L840`**[]{#lh:L840} Leverage/DockingTheoryBridge.lean

**`L841`**[]{#lh:L841}

**`L842`**[]{#lh:L842}

**`L843`**[]{#lh:L843} Leverage/DockingTheoryBridge.lean

**`L844`**[]{#lh:L844} Leverage/DockingTheoryBridge.lean

**`L845`**[]{#lh:L845} Leverage/DockingTheoryBridge.lean

**`L846`**[]{#lh:L846}

**`L847`**[]{#lh:L847}

**`L848`**[]{#lh:L848}

**`L849`**[]{#lh:L849}

**`L850`**[]{#lh:L850}

**`L851`**[]{#lh:L851}

**`L852`**[]{#lh:L852}

**`L853`**[]{#lh:L853}

**`L854`**[]{#lh:L854}

**`L855`**[]{#lh:L855}

**`L856`**[]{#lh:L856}

**`L857`**[]{#lh:L857}

**`L858`**[]{#lh:L858}

**`L859`**[]{#lh:L859} Leverage/DockingTheoryBridge.lean

**`L860`**[]{#lh:L860} Leverage/DockingTheoryBridge.lean

**`L861`**[]{#lh:L861}

**`L862`**[]{#lh:L862}

**`L863`**[]{#lh:L863}

**`L864`**[]{#lh:L864} Leverage/DockingTheoryBridge.lean

**`L865`**[]{#lh:L865} Leverage/DockingSolverEndpoint.lean

**`L866`**[]{#lh:L866}

**`L867`**[]{#lh:L867}

**`L868`**[]{#lh:L868} Leverage/DockingSolverEndpoint.lean

**`L869`**[]{#lh:L869} Leverage/DockingSolverEndpoint.lean

**`L870`**[]{#lh:L870} Leverage/DockingSolverEndpoint.lean

**`L871`**[]{#lh:L871} Leverage/DockingSolverEndpoint.lean

**`L872`**[]{#lh:L872} Leverage/DockingSolverEndpoint.lean

**`L873`**[]{#lh:L873} Leverage/DockingSolverEndpoint.lean

**`L874`**[]{#lh:L874} Leverage/DockingSolverEndpoint.lean

**`L875`**[]{#lh:L875} Leverage/DockingSolverEndpoint.lean

**`L876`**[]{#lh:L876} Leverage/DockingSolverEndpoint.lean

**`L877`**[]{#lh:L877} Leverage/DockingSolverEndpoint.lean

**`L878`**[]{#lh:L878} Leverage/DockingSolverEndpoint.lean

**`L879`**[]{#lh:L879} Leverage/DockingSolverEndpoint.lean

**`L880`**[]{#lh:L880} Leverage/DockingSolverEndpoint.lean

**`L881`**[]{#lh:L881} Leverage/DockingSolverEndpoint.lean

**`L882`**[]{#lh:L882} Leverage/DockingSolverEndpoint.lean

**`L883`**[]{#lh:L883}

**`L884`**[]{#lh:L884}

**`L885`**[]{#lh:L885}

**`L886`**[]{#lh:L886} Leverage/DockingSolverEndpoint.lean

**`L887`**[]{#lh:L887} Leverage/DockingSolverEndpoint.lean

**`L888`**[]{#lh:L888} Leverage/DockingSolverEndpoint.lean

**`L889`**[]{#lh:L889} Leverage/DockingSolverEndpoint.lean

**`L890`**[]{#lh:L890} Leverage/DockingSolverEndpoint.lean

**`L891`**[]{#lh:L891} Leverage/DockingSolverEndpoint.lean

**`L892`**[]{#lh:L892} Leverage/DockingSolverEndpoint.lean

**`L893`**[]{#lh:L893} Leverage/DockingSolverEndpoint.lean

**`L894`**[]{#lh:L894}

**`L895`**[]{#lh:L895}

**`L896`**[]{#lh:L896} Leverage/DockingSolverEndpoint.lean

**`L897`**[]{#lh:L897} Leverage/DockingSolverEndpoint.lean

**`L898`**[]{#lh:L898} Leverage/DockingSolverEndpoint.lean

**`L899`**[]{#lh:L899} Leverage/DockingSolverEndpoint.lean

**`L900`**[]{#lh:L900} Leverage/DockingSolverEndpoint.lean

**`L901`**[]{#lh:L901} Leverage/DockingSolverEndpoint.lean

**`L902`**[]{#lh:L902} Leverage/DockingSolverEndpoint.lean

**`L903`**[]{#lh:L903} Leverage/DockingSolverEndpoint.lean

**`L904`**[]{#lh:L904} Leverage/DockingSolverEndpoint.lean

**`L905`**[]{#lh:L905}

**`L906`**[]{#lh:L906}

**`L907`**[]{#lh:L907}

**`L908`**[]{#lh:L908}

**`L909`**[]{#lh:L909}

**`L910`**[]{#lh:L910} Leverage/DockingSolverEndpoint.lean

**`L911`**[]{#lh:L911} Leverage/DockingSolverEndpoint.lean

**`L912`**[]{#lh:L912}

**`L913`**[]{#lh:L913}

**`L914`**[]{#lh:L914}

**`L915`**[]{#lh:L915}

**`L916`**[]{#lh:L916} Leverage/DockingSolverEndpoint.lean

**`L917`**[]{#lh:L917} Leverage/DockingSolverEndpoint.lean

**`L918`**[]{#lh:L918} Leverage/DockingSolverEndpoint.lean

**`L919`**[]{#lh:L919} Leverage/DockingSolverEndpoint.lean

**`L920`**[]{#lh:L920} Leverage/DockingSolverEndpoint.lean

**`L921`**[]{#lh:L921} Leverage/DockingSolverEndpoint.lean

**`L922`**[]{#lh:L922} Leverage/DockingSolverEndpoint.lean

**`L923`**[]{#lh:L923} Leverage/DockingSolverEndpoint.lean

**`L924`**[]{#lh:L924} Leverage/DockingSolverEndpoint.lean

**`L925`**[]{#lh:L925} Leverage/DockingSolverEndpoint.lean

**`L926`**[]{#lh:L926} Leverage/DockingSolverEndpoint.lean

**`L927`**[]{#lh:L927}

**`L928`**[]{#lh:L928} Leverage/DockingSolverEndpoint.lean

**`L929`**[]{#lh:L929} Leverage/DockingSolverEndpoint.lean

**`L930`**[]{#lh:L930} Leverage/DockingSolverEndpoint.lean

**`L931`**[]{#lh:L931} Leverage/DockingSolverEndpoint.lean

**`L932`**[]{#lh:L932} Leverage/DockingSolverEndpoint.lean

**`L933`**[]{#lh:L933}

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

& [**`L118`**]{#lh:L118} & `Leverage.optSummary_srank_le_srank`

\
[**`L119`**]{#lh:L119} & `Leverage.optSummary_srank_lt_srank_iff`

& [**`L120`**]{#lh:L120} & `Leverage.resolutionControlled_actionPinned_opt_eq_liftedFixedTieBreak`

\
[**`L121`**]{#lh:L121} & `Leverage.resolutionControlled_liftedFixedTieBreak_opt_subset_ambiguityBand`

& [**`L122`**]{#lh:L122} & `Leverage.extractRelevantCoordinates_polytime`

\
[**`L123`**]{#lh:L123} & `Leverage.pathwise_srank_energy_lower_bound`

& [**`L124`**]{#lh:L124} & `Leverage.quantitativeAdmissibility_srank_eq_relaxed_plus_collapseCount`

\
[**`L125`**]{#lh:L125} & `Leverage.quantitativeAdmissibility_energy_saves_one_bit`

& [**`L126`**]{#lh:L126} & `Leverage.rankGapDesignClass_of_sufficient_and_uniformGap`

\
[**`L127`**]{#lh:L127} & `Leverage.reverseDesign_polytime_of_energyBudget`

& [**`L128`**]{#lh:L128} & `Leverage.quotientTrajectory_forwardReverseRatio_eq_exp_srankEntropy`

\
[**`L129`**]{#lh:L129} & `Leverage.quotientTrajectory_dissipation_lower_bound`

& [**`L130`**]{#lh:L130} & `Leverage.quotient_resolution_speed_bound`

\
[**`L131`**]{#lh:L131} & `Leverage.trajectory_time_energy_tradeoff`

& [**`L132`**]{#lh:L132} & `Leverage.allosteric_srank_graph_bound`

\
[**`L133`**]{#lh:L133} & `Leverage.mechanochemical_coupling_gap`

& [**`L134`**]{#lh:L134} & `Leverage.hierarchical_srank_bound`

\
[**`L135`**]{#lh:L135} & `Leverage.renormalized_admissibility_equivalence`

& [**`L136`**]{#lh:L136} & `Leverage.error_correction_srank_overhead`

\
[**`L137`**]{#lh:L137} & `Leverage.fault_tolerant_landauer_floor`

& [**`L138`**]{#lh:L138} & `Leverage.binding_free_energy_floor`

\
[**`L139`**]{#lh:L139} & `Leverage.decision_quotient_potential`

& [**`L140`**]{#lh:L140} & `Leverage.geometric_contact_allosteric_srank_bound`

\
[**`L141`**]{#lh:L141} & `Leverage.geometric_contact_shell_srank_bound`

& [**`L142`**]{#lh:L142} & `Leverage.geometric_contact_shell_bounded_regime`

\
[**`L394`**]{#lh:L394} & `Leverage.admissibleDocking_exhaustion`

& [**`L395`**]{#lh:L395} & `Leverage.quantitativeAdmissibility_progress`

\
[**`L396`**]{#lh:L396} & `Leverage.bondConstraintFamily_constraintObservations_card`

& [**`L397`**]{#lh:L397} & `Leverage.bondConstraintFamily_srank_eq_effectiveDOF`

\
[**`L398`**]{#lh:L398} & `Leverage.bondConstraintFamily_energy_lower_bound`

& [**`L403`**]{#lh:L403} & `Leverage.nonlinearConstraintFamily_fullRowRank_of_pivotWitness`

\
[**`L404`**]{#lh:L404} & `Leverage.nonlinearConstraintFamily_nondegenerate_of_pivotWitness`

& [**`L405`**]{#lh:L405} & `Leverage.nonlinearConstraintFamily_constraintObservations_card_of_pivotWitness`

\
[**`L406`**]{#lh:L406} & `Leverage.nonlinearConstraintFamily_srank_eq_effectiveDOF_of_pivotWitness`

& [**`L407`**]{#lh:L407} & `Leverage.nonlinearConstraintFamily_energy_lower_bound_of_pivotWitness`

\
[**`L408`**]{#lh:L408} & `Leverage.optimizerLikelihood_fisher_eq_indicator`

& [**`L409`**]{#lh:L409} & `Leverage.parametricCanonical_identifiableDimension_eq_srank`

\
[**`L410`**]{#lh:L410} & `Leverage.parametricCanonical_fisherDiag_eq_relevanceIndicator`

& [**`L411`**]{#lh:L411} & `Leverage.parametricCanonical_fisherDiag_one_iff_relevant`

\
[**`L412`**]{#lh:L412} & `Leverage.parametricCanonical_variance_eq_top_of_nonrelevant`

& [**`L413`**]{#lh:L413} & `Leverage.canonicalExactResolutionObject_isInitial`

\
[**`L414`**]{#lh:L414} & `Leverage.canonicalDP_initiality_and_srank`

& [**`L415`**]{#lh:L415} & `Leverage.quantitativeAdmissibility_collapsedSrank_eq_of_same_collapseCount`

\
[**`L416`**]{#lh:L416} & `Leverage.quantitativeAdmissibility_collapsedRankObject_iso_of_same_collapseCount`

& [**`L417`**]{#lh:L417} & `Leverage.binary_srank_ge_of_numOptClasses_ge_two_pow`

\
[**`L418`**]{#lh:L418} & `Leverage.binary_srank_ge_log2_of_numOptClasses_ge`

& [**`L419`**]{#lh:L419} & `Leverage.binding_free_energy_eq_floor_plus_gap`

\
[**`L420`**]{#lh:L420} & `Leverage.binding_free_energy_gap_nonneg`

& [**`L421`**]{#lh:L421} & `Leverage.binding_residualEntropy_nonneg`

\
[**`L422`**]{#lh:L422} & `Leverage.binding_free_energy_floor_tight_iff_residualEntropy_zero`

& [**`L423`**]{#lh:L423} & `Leverage.quotientTrajectory_crooks_standard_form_of_work_calibration`

\
[**`L424`**]{#lh:L424} & `Leverage.jarzynski_equality_of_finite_crooks`

& [**`L425`**]{#lh:L425} & `Leverage.hopfield_ninio_rank_overhead_lower_bound`

\
[**`L426`**]{#lh:L426} & `Leverage.hopfield_ninio_dissipation_overhead_lower_bound`

& [**`L427`**]{#lh:L427} & `Leverage.hopfield_ninio_fault_tolerant_rank_overhead_lower_bound`

\
[**`L428`**]{#lh:L428} & `Leverage.ProteinMechanicalGraph.allostericDistanceContributionUpper_antitone`

& [**`L429`**]{#lh:L429} & `Leverage.ProteinMechanicalGraph.allosteric_contribution_decay_law`

\
[**`L430`**]{#lh:L430} & `Leverage.allosteric_srank_bound_of_distanceDecayProfile`

& [**`L431`**]{#lh:L431} & `Leverage.ProteinMechanicalGraph.allosteric_exponential_contribution_decay_law`

\
[**`L432`**]{#lh:L432} & `Leverage.allosteric_srank_bound_of_exponentialDecayProfile`

& [**`L433`**]{#lh:L433} & `Leverage.geometric_contact_kHop_srank_bound_of_exponentialDecayProfile`

\
[**`L434`**]{#lh:L434} & `Leverage.hopfield_ninio_rank_overhead_lower_bound_of_kinetic_branch_model`

& [**`L435`**]{#lh:L435} & `Leverage.hopfield_ninio_dissipation_overhead_lower_bound_of_kinetic_branch_model`

\
[**`L436`**]{#lh:L436} & `Leverage.hopfield_ninio_fault_tolerant_rank_overhead_lower_bound_of_kinetic_branch_model`

& [**`L437`**]{#lh:L437} & `Leverage.ProteinMechanicalGraph.allosteric_polynomial_contribution_decay_law`

\
[**`L438`**]{#lh:L438} & `Leverage.allosteric_srank_bound_of_polynomialDecayProfile`

& [**`L439`**]{#lh:L439} & `Leverage.geometric_contact_kHop_srank_bound_of_polynomialDecayProfile`

\
[**`L440`**]{#lh:L440} & `Leverage.allosteric_srank_bound_of_polynomialDecayProfile_of_seriesBound`

& [**`L441`**]{#lh:L441} & `Leverage.geometric_contact_kHop_srank_bound_of_polynomialDecayProfile_of_seriesBound`

\
[**`L442`**]{#lh:L442} & `Leverage.reciprocalPowerSeriesBound_of_two_le`

& [**`L443`**]{#lh:L443} & `Leverage.allosteric_srank_bound_of_polynomialDecayProfile_of_two_le`

\
[**`L444`**]{#lh:L444} & `Leverage.geometric_contact_kHop_srank_bound_of_polynomialDecayProfile_of_two_le`

& [**`L446`**]{#lh:L446} & `Leverage.admissibility_speed_accuracy_tradeoff`

\
[**`L447`**]{#lh:L447} & `Leverage.admissibility_onRate_upper_bound`

& [**`L448`**]{#lh:L448} & `Leverage.admissibility_onRate_upper_bound_of_relaxation`

\
[**`L449`**]{#lh:L449} & `Leverage.AdmissibilityCollapsedRankObject.hom_id`

& [**`L450`**]{#lh:L450} & `Leverage.AdmissibilityCollapsedRankObject.hom_comp`

\
[**`L451`**]{#lh:L451} & `Leverage.collapsedRankInitial_hom`

& [**`L452`**]{#lh:L452} & `Leverage.collapsedRank_no_terminal`

\
[**`L453`**]{#lh:L453} & `Leverage.collapsedRankProduct_universal`

& [**`L454`**]{#lh:L454} & `Leverage.collapsedRankCoproduct_universal`

\
[**`L455`**]{#lh:L455} & `Leverage.collapsedRankTensor_map`

& [**`L456`**]{#lh:L456} & `Leverage.collapsedRankStructuralFunctor_mono`

\
[**`L457`**]{#lh:L457} & `Leverage.admissibility_resolution_speed_bound`

& [**`L458`**]{#lh:L458} & `Leverage.admissibility_speed_accuracy_tradeoff_mul`

\
[**`L459`**]{#lh:L459} & `Leverage.boundedCollapsedRank_has_finite_limits`

& [**`L460`**]{#lh:L460} & `Leverage.boundedCollapsedRank_has_finite_colimits`

\
[**`L461`**]{#lh:L461} & `Leverage.collapsedRank_bounded_vs_unbounded_terminal`

& [**`L462`**]{#lh:L462} & `Leverage.boundedCollapsedRank_has_all_finite_meets_joins`

\
[**`L463`**]{#lh:L463} & `Leverage.boundedCollapsedRank_has_arbitrary_meets_joins`

& [**`L464`**]{#lh:L464} & `Leverage.boundedCollapsedRankArbitraryJoin_monotone`

\
[**`L465`**]{#lh:L465} & `Leverage.boundedCollapsedRankArbitraryMeet_antitone`

& [**`L466`**]{#lh:L466} & `Leverage.boundedCollapsedRankArbitraryMeet_singleton_rank`

\
[**`L467`**]{#lh:L467} & `Leverage.boundedCollapsedRankArbitraryJoin_singleton_rank`

& [**`L468`**]{#lh:L468} & `Leverage.boundedCollapsedRankArbitraryMeet_le_join_of_nonempty`

\
[**`L469`**]{#lh:L469} & `Leverage.boundedCollapsedRankArbitraryMeet_insert_join_absorption_rank`

& [**`L470`**]{#lh:L470} & `Leverage.boundedCollapsedRankArbitraryJoin_insert_meet_absorption_rank`

\
[**`L471`**]{#lh:L471} & `Leverage.boundedCollapsedRankProduct_comm`

& [**`L472`**]{#lh:L472} & `Leverage.boundedCollapsedRankProduct_assoc`

\
[**`L473`**]{#lh:L473} & `Leverage.boundedCollapsedRankProduct_idem`

& [**`L474`**]{#lh:L474} & `Leverage.boundedCollapsedRankCoproduct_comm`

\
[**`L475`**]{#lh:L475} & `Leverage.boundedCollapsedRankCoproduct_assoc`

& [**`L476`**]{#lh:L476} & `Leverage.boundedCollapsedRankCoproduct_idem`

\
[**`L477`**]{#lh:L477} & `Leverage.boundedCollapsedRankProduct_absorb_coproduct`

& [**`L478`**]{#lh:L478} & `Leverage.boundedCollapsedRankCoproduct_absorb_product`

\
[**`L479`**]{#lh:L479} & `Leverage.optSummary_collapseCount_eq_zero_of_bifactor`

& [**`L480`**]{#lh:L480} & `Leverage.quantitativeAdmissibility_collapseCount_eq_zero_of_backwardFactor`

\
[**`L481`**]{#lh:L481} & `Leverage.quantitativeAdmissibility_srank_eq_of_backwardFactor`

& [**`L482`**]{#lh:L482} & `Leverage.admissibility_speed_accuracy_tradeoff_of_backwardFactor`

\
[**`L483`**]{#lh:L483} & `Leverage.quotientMCMCKernel_edgeFlow_eq_reverse_of_boltzmannStationary`

& [**`L484`**]{#lh:L484} & `Leverage.quotientTrajectory_logEdgeFlowRatio_eq_zero_of_detailedBalance_boltzmannStationary`

\
[**`L485`**]{#lh:L485} & `Leverage.quotientTrajectory_crooks_calibration_of_detailedBalance_boltzmannStationary`

& [**`L486`**]{#lh:L486} & `Leverage.quotientTrajectory_forwardReverseRatio_eq_one_of_detailedBalance_boltzmannStationary`

\
[**`L487`**]{#lh:L487} & `Leverage.decisionEncodingTransport_isRelevant_iff`

& [**`L488`**]{#lh:L488} & `Leverage.decisionEncodingTransport_srank_eq`

\
[**`L489`**]{#lh:L489} & `Leverage.decisionEncodingTransport_energyLowerBound_eq`

& [**`L490`**]{#lh:L490} & `Leverage.OperationKernelSchema.abstraction_has_unique_factorization_to_kernel_quotient`

\
[**`L491`**]{#lh:L491} & `Leverage.OperationKernelSchema.canonical_unique_initial_with_noCollapse`

& [**`L492`**]{#lh:L492} & `Leverage.decision_and_eventualGerm_share_kernel_universality`

\
[**`L493`**]{#lh:L493} & `Leverage.decisionKernel_noCollapse_canonicity`

& [**`L494`**]{#lh:L494} & `Leverage.eventualGermKernel_noCollapse_canonicity`

\
[**`L495`**]{#lh:L495} & `Leverage.OperationKernelSchema.operationSystemEquivKernelSchema`

& [**`L496`**]{#lh:L496} & `Leverage.OperationKernelSchema.kernelCompletion_full_faithful`

\
[**`L497`**]{#lh:L497} & `Leverage.OperationKernelSchema.kernelCompletion_endpoint`

& [**`L498`**]{#lh:L498} & `Leverage.eventualGermKernelSchema_relation_iff_eventualEqSeq`

\
[**`L499`**]{#lh:L499} & `Leverage.decisionKernel_endpoint`

& [**`L500`**]{#lh:L500} & `Leverage.eventualGermKernel_endpoint`

\
[**`L501`**]{#lh:L501} & `Leverage.decision_and_eventualGerm_full_faithful_embeddings`

& [**`L502`**]{#lh:L502} & `Leverage.OperationKernelSchema.RenormalizationFlow.onQuotient_eq_id`

\
[**`L503`**]{#lh:L503} & `Leverage.OperationKernelSchema.renormalization_kernel_endpoint`

& [**`L504`**]{#lh:L504} & `Leverage.aeGermKernelSchema_relation_iff_aeEq`

\
[**`L505`**]{#lh:L505} & `Leverage.aeGermKernel_endpoint`

& [**`L506`**]{#lh:L506} & `Leverage.aeGerm_full_faithful_embedding`

\
[**`L507`**]{#lh:L507} & `Leverage.aeGerm_measureRG_endpoint`

& [**`L508`**]{#lh:L508} & `Leverage.MeasureKernelSemigroup.ScaleFlow.stationary_at_scale_of_detailedBalance`

\
[**`L509`**]{#lh:L509} & `Leverage.MeasureKernelTransport.stationary_transport`

& [**`L510`**]{#lh:L510} & `Leverage.DetailedBalanceTransport.detailedBalance_transport`

\
[**`L511`**]{#lh:L511} & `Leverage.DetailedBalanceTransport.stationary_transport_of_detailedBalance`

& [**`L512`**]{#lh:L512} & `Leverage.DetailedBalanceTransport.stationary_transport_at_scales`

\
[**`L513`**]{#lh:L513} & `Leverage.measureKernel_transport_endpoint`

& [**`L514`**]{#lh:L514} & `Leverage.measurableDeterministicKernelSemigroup`

\
[**`L515`**]{#lh:L515} & `Leverage.measurableDeterministicInvariantLayer`

& [**`L516`**]{#lh:L516} & `Leverage.measurableDeterministic_scale_stationary_of_invariant`

\
[**`L517`**]{#lh:L517} & `Leverage.MeasureKernelSemigroup.stationary_kernelPow`

& [**`L518`**]{#lh:L518} & `Leverage.MeasureKernelSemigroup.detailedBalance_stationary_kernelPow`

\
[**`L519`**]{#lh:L519} & `Leverage.MeasureKernelSemigroupHom.map_kernelPow`

& [**`L520`**]{#lh:L520} & `Leverage.MeasureKernelSemigroupHom.stationary_transport_kernelPow`

\
[**`L521`**]{#lh:L521} & `Leverage.DetailedBalanceSemigroupHom.stationary_transport_kernelPow_of_detailedBalance`

& [**`L522`**]{#lh:L522} & `Leverage.MeasureKernelSemigroup.ScaleFlow.kernelAt_eq_kernelPow_of_one`

\
[**`L523`**]{#lh:L523} & `Leverage.MeasureKernelSemigroup.ScaleFlow.stationary_of_detailedBalance_at_one`

& [**`L524`**]{#lh:L524} & `Leverage.DetailedBalanceSemigroupHom.stationary_transport_scaleFlow_of_detailedBalance_at_one`

\
[**`L525`**]{#lh:L525} & `Leverage.PathSpaceCrooksModel.logRatio_kernelPow_eq_zero_of_stationary`

& [**`L526`**]{#lh:L526} & `Leverage.PathSpaceCrooksModel.logRatio_kernelPow_eq_zero_of_detailedBalance`

\
[**`L527`**]{#lh:L527} & `Leverage.PathSpaceCrooksModel.logRatio_scale_eq_zero_of_detailedBalance_at_one`

& [**`L528`**]{#lh:L528} & `Leverage.PathSpaceCrooksTransport.logRatio_transport_kernelPow`

\
[**`L529`**]{#lh:L529} & `Leverage.PathSpaceCrooksTransport.logRatio_kernelPow_eq_zero_of_source_detailedBalance`

& [**`L530`**]{#lh:L530} & `Leverage.measurableTransitionKernelSemigroup`

\
[**`L531`**]{#lh:L531} & `Leverage.measurableTransition_scale_stationary_of_detailedBalance_at_one`

& [**`L532`**]{#lh:L532} & `Leverage.measurableTransition_pathSpaceCrooks_zero_of_detailedBalance_at_one`

\
[**`L533`**]{#lh:L533} & `Leverage.PathSpaceCrooksModel.logRatio_scaleFlow_eq_zero_of_transported_detailedBalance_at_one`

& [**`L534`**]{#lh:L534} & `Leverage.PathSpaceCrooksTransport.logRatio_scaleFlow_eq_zero_of_source_detailedBalance_at_one`

\
[**`L535`**]{#lh:L535} & `Leverage.deterministicToTransitionSemigroupHom`

& [**`L536`**]{#lh:L536} & `Leverage.deterministicToTransition_stationary_kernelPow`

\
[**`L537`**]{#lh:L537} & `Leverage.PathSpaceJarzynskiModel.expNegLogRatioExpectation_kernelPow_eq_one_of_stationary`

& [**`L538`**]{#lh:L538} & `Leverage.PathSpaceJarzynskiModel.expNegLogRatioExpectation_kernelPow_eq_one_of_detailedBalance`

\
[**`L539`**]{#lh:L539} & `Leverage.PathSpaceJarzynskiModel.expNegLogRatioExpectation_scale_eq_one_of_detailedBalance_at_one`

& [**`L540`**]{#lh:L540} & `Leverage.PathSpaceJarzynskiTransport.expNegLogRatioExpectation_transport_kernelPow`

\
[**`L541`**]{#lh:L541} & `Leverage.PathSpaceJarzynskiTransport.expNegLogRatioExpectation_kernelPow_eq_one_of_source_detailedBalance`

& [**`L542`**]{#lh:L542} & `Leverage.measurableTransition_pathSpaceJarzynski_one_of_detailedBalance_at_one`

\
[**`L543`**]{#lh:L543} & `Leverage.PathSpaceExpectationModel.pathIntegral_expNegLogRatio_eq_one_of_logRatio_eq_zero`

& [**`L544`**]{#lh:L544} & `Leverage.PathSpaceExpectationModel.pathIntegral_expNegLogRatio_kernelPow_eq_one_of_stationary`

\
[**`L545`**]{#lh:L545} & `Leverage.PathSpaceExpectationModel.pathIntegral_expNegLogRatio_kernelPow_eq_one_of_detailedBalance`

& [**`L546`**]{#lh:L546} & `Leverage.PathSpaceExpectationModel.pathIntegral_expNegLogRatio_scale_eq_one_of_detailedBalance_at_one`

\
[**`L547`**]{#lh:L547} & `Leverage.PathSpaceJarzynskiTransport.pathIntegral_expNegLogRatio_transport_kernelPow`

& [**`L548`**]{#lh:L548} & `Leverage.PathSpaceJarzynskiTransport.pathIntegral_expNegLogRatio_kernelPow_eq_one_of_source_detailedBalance`

\
[**`L549`**]{#lh:L549} & `Leverage.measurableTransition_pathSpaceJarzynski_pathIntegral_one_of_detailedBalance_at_one`

& [**`L550`**]{#lh:L550} & `Leverage.MeasureKernelQuotientCalculus.stationary_descends`

\
[**`L551`**]{#lh:L551} & `Leverage.MeasureKernelSemigroupQuotientCalculus.stationary_kernelPow_descends`

& [**`L552`**]{#lh:L552} & `Leverage.DetailedBalanceSemigroupQuotientCalculus.stationary_kernelPow_descends_of_detailedBalance`

\
[**`L553`**]{#lh:L553} & `Leverage.DetailedBalanceSemigroupQuotientCalculus.stationary_scaleFlow_descends_of_detailedBalance_at_one`

& [**`L554`**]{#lh:L554} & `Leverage.KernelSemigroupEndomorphism.map_kernelPow`

\
[**`L555`**]{#lh:L555} & `Leverage.QuotientKernelRGCompatibility.commutes_kernelPow`

& [**`L556`**]{#lh:L556} & `Leverage.DetailedBalanceQuotientKernelRGCompatibility.stationary_targetRG_kernelPow_of_source_detailedBalance`

\
[**`L557`**]{#lh:L557} & `Leverage.DetailedBalanceQuotientKernelRGCompatibility.endpoint`

& [**`L558`**]{#lh:L558} & `Leverage.MeasureKernelSemigroupHom.stationary_transport_kernelPow_as_quotient_instance`

\
[**`L559`**]{#lh:L559} & `Leverage.DetailedBalanceSemigroupHom.stationary_transport_kernelPow_as_quotient_instance`

& [**`L560`**]{#lh:L560} & `Leverage.PathSpaceExpectationModel.pathIntegral_scaleFlow_eq_one_of_transported_detailedBalance_at_one`

\
[**`L562`**]{#lh:L562} & `Leverage.PathSpaceProcessTransport.expNegLogRatioExpectation_transport_kernelPow`

& [**`L563`**]{#lh:L563} & `Leverage.PathSpaceProcessTransport.pathIntegral_transport_kernelPow`

\
[**`L564`**]{#lh:L564} & `Leverage.PathSpaceProcessTransport.pathIntegral_kernelPow_eq_one_of_source_detailedBalance`

& [**`L565`**]{#lh:L565} & `Leverage.measurableTransitionFinitePathMeasure_zero`

\
[**`L566`**]{#lh:L566} & `Leverage.measurableTransitionFinitePathMeasure_succ`

& [**`L567`**]{#lh:L567} & `Leverage.measurableTransition_finiteHorizon_pathIntegral_scale_eq_one_of_detailedBalance_at_one`

\
[**`L568`**]{#lh:L568} & `Leverage.measurableTransition_finiteHorizon_processTransport_kernelPow_pathIntegral_eq_one_of_source_detailedBalance`

& [**`L569`**]{#lh:L569} & `Leverage.measurableTransition_finiteHorizon_pathIntegral_scale_eq_one_of_transported_detailedBalance_at_one`

\
[**`L570`**]{#lh:L570} & `Leverage.MeasurableTransitionFiniteHorizonProjectiveConsistency.marginal_eq`

& [**`L571`**]{#lh:L571} & `Leverage.MeasurableTransitionFiniteHorizonProjectiveConsistency.marginal_kernelPow_eq`

\
[**`L572`**]{#lh:L572} & `Leverage.MeasurableTransitionFiniteHorizonProjectiveConsistency.marginal_scaleFlow_eq`

& [**`L573`**]{#lh:L573} & `Leverage.MeasurableTransitionKolmogorovExtension.marginal_recovery`

\
[**`L574`**]{#lh:L574} & `Leverage.MeasurableTransitionKolmogorovExtension.marginal_pathIntegral_scale_eq_one_of_detailedBalance_at_one`

& [**`L575`**]{#lh:L575} & `Leverage.MeasurableTransitionKolmogorovExtension.marginal_pathIntegral_scale_eq_one_of_transported_detailedBalance_at_one`

\
[**`L576`**]{#lh:L576} & `Leverage.measurableTransitionFiniteHorizon_projectiveConsistency_instance`

& [**`L577`**]{#lh:L577} & `Leverage.ToleranceCollapseProfile.collapseCount_eq_formula`

\
[**`L578`**]{#lh:L578} & `Leverage.LJPocketToleranceCollapseProfile.collapseCount_eq_formula`

& [**`L579`**]{#lh:L579} & `Leverage.CompositeConstraintDecisionProcedure.srank_eq_effectiveDOF`

\
[**`L580`**]{#lh:L580} & `Leverage.MolecularDynamicsNonequilibriumCalibration.crooks_standard_form`

& [**`L581`**]{#lh:L581} & `Leverage.ContinuousTimeKernelSemigroup.discreteScaleFlow`

\
[**`L582`**]{#lh:L582} & `Leverage.GeneralObservationFisherInterface.totalFisher_eq_srank`

& [**`L583`**]{#lh:L583} & `Leverage.GeneralObservationFisherInterface.relevant_iff_fisher_eq_one`

\
[**`L584`**]{#lh:L584} & `Leverage.AlphabetRichnessInterface.srank_ge_of_numOptClasses_ge_pow`

& [**`L585`**]{#lh:L585} & `Leverage.VCStyleRichnessWitness.vcDim_le_srank`

\
[**`L586`**]{#lh:L586} & `Leverage.inverseDesign_atomistic_realization`

& [**`L587`**]{#lh:L587} & `Leverage.ContinuousStatePathExtensionInterface.measurable_evalAt_pair`

\
[**`L588`**]{#lh:L588} & `Leverage.MeasurableTransitionFinitePath.measurable_truncate`

& [**`L589`**]{#lh:L589} & `Leverage.ComposedHamiltonian_Architecture_instance`

\
[**`L590`**]{#lh:L590} & `Leverage.ComposedHamiltonian_Lipschitz_bound`

& [**`L591`**]{#lh:L591} & `Leverage.Langevin_satisfies_DetailedBalance`

\
[**`L592`**]{#lh:L592} & `Leverage.Langevin_Discretization_to_MCMC`

& [**`L593`**]{#lh:L593} & `Leverage.SpinHalf_two_states`

\
[**`L594`**]{#lh:L594} & `Leverage.SpinHalf_CanonicalDP_instance`

& [**`L595`**]{#lh:L595} & `Leverage.SpinHalf_Decoherence_Cost`

\
[**`L596`**]{#lh:L596} & `Leverage.TopLevel_Computable`

& [**`L597`**]{#lh:L597} & `Leverage.NoisyPartialObservationChannel.totalFisher_eq_srank`

\
[**`L598`**]{#lh:L598} & `Leverage.NoisyPartialObservationChannel.relevant_iff_debiasedFisher_eq_one`

& [**`L599`**]{#lh:L599} & `Leverage.langevin_solution_exists_unique`

\
[**`L600`**]{#lh:L600} & `Leverage.langevin_boltzmann_invariant`

& [**`L601`**]{#lh:L601} & `Leverage.langevin_ergodic`

\
[**`L602`**]{#lh:L602} & `Leverage.eulerMaruyama_strong_error_bound`

& [**`L603`**]{#lh:L603} & `Leverage.eulerMaruyama_weak_error_bound`

\
[**`L604`**]{#lh:L604} & `Leverage.concreteLangevin_to_interface`

& [**`L605`**]{#lh:L605} & `Leverage.concreteComposedHamiltonian_Architecture_instance`

\
[**`L606`**]{#lh:L606} & `Leverage.concreteComposedHamiltonian_lipschitz_bound`

& [**`L607`**]{#lh:L607} & `Leverage.concreteComposedHamiltonian_halfGapTransport`

\
[**`L608`**]{#lh:L608} & `Leverage.chemical_augmented_opt_eq_projection`

& [**`L609`**]{#lh:L609} & `Leverage.chemical_augmented_srank_eq_of_witness`

\
[**`L610`**]{#lh:L610} & `Leverage.conformationalEnsemble_population_normalized`

& [**`L611`**]{#lh:L611} & `Leverage.ensemble_to_docking_transport`

\
[**`L612`**]{#lh:L612} & `Leverage.inducedFit_rank_transport`

& [**`L613`**]{#lh:L613} & `Leverage.kinetic_onRate_bound`

\
[**`L614`**]{#lh:L614} & `Leverage.kinetic_residence_eq_inverse_koff`

& [**`L615`**]{#lh:L615} & `Leverage.kinetic_pathway_population_normalized`

\
[**`L616`**]{#lh:L616} & `Leverage.kinetic_observable_bundle`

& [**`L617`**]{#lh:L617} & `Leverage.legacy_constructive_equiv`

\
[**`L618`**]{#lh:L618} & `Leverage.constructive_deprecation_ready`

& [**`L619`**]{#lh:L619} & `Leverage.langevin_endpoints_of_analysis_assumptions`

\
[**`L620`**]{#lh:L620} & `Leverage.concreteComposedHamiltonian_zero_shell_constants`

& [**`L621`**]{#lh:L621} & `Leverage.biomolecularReference_zero_shell_constants`

\
[**`L622`**]{#lh:L622} & `Leverage.concreteComposedHamiltonian_global_zero_lipschitz`

& [**`L623`**]{#lh:L623} & `Leverage.chemical_augmented_opt_eq_projection_of_binding_problem`

\
[**`L624`**]{#lh:L624} & `Leverage.chemical_augmented_srank_eq_of_binding_problem`

& [**`L625`**]{#lh:L625} & `Leverage.ensemble_to_binding_problem_transport`

\
[**`L626`**]{#lh:L626} & `Leverage.inducedFit_rank_transport_to_binding_problem`

& [**`L627`**]{#lh:L627} & `Leverage.docking_kinetic_observable_bundle`

\
[**`L628`**]{#lh:L628} & `Leverage.constructive_replaces_legacy_output_fields`

& [**`L629`**]{#lh:L629} & `Leverage.downstream_consumer_transport_of_constructive_equiv`

\
[**`L630`**]{#lh:L630} & `Leverage.langevin_endpoints_of_first_principles`

& [**`L631`**]{#lh:L631} & `Leverage.concreteComposedHamiltonian_nontrivial_lipschitz_bound`

\
[**`L632`**]{#lh:L632} & `Leverage.concreteComposedHamiltonian_nontrivial_halfGapTransport`

& [**`L633`**]{#lh:L633} & `Leverage.chemical_component_variation_preserves_opt`

\
[**`L634`**]{#lh:L634} & `Leverage.oneStepConformerPopulation_sum_one`

& [**`L635`**]{#lh:L635} & `Leverage.kinetic_observable_bundle_of_protocol_measurements`

\
[**`L636`**]{#lh:L636} & `Leverage.fullyConstructivePipeline_deprecation_ready`

& [**`L637`**]{#lh:L637} & `Leverage.LangevinTransitionKernel.boltzmann_stationary_of_detailedBalance`

\
[**`L638`**]{#lh:L638} & `Leverage.LangevinFirstPrinciplesAssumptions.explicit_sde_conditions`

& [**`L639`**]{#lh:L639} & `Leverage.LangevinFirstPrinciplesAssumptions.ergodic_of_dissipativity`

\
[**`L640`**]{#lh:L640} & `Leverage.concreteComposedHamiltonian_fullState_lipschitz_bound`

& [**`L641`**]{#lh:L641} & `Leverage.concreteComposedHamiltonian_fullState_halfGapTransport`

\
[**`L642`**]{#lh:L642} & `Leverage.biomolecularCalibrated_fullState_shell_constants_pos`

& [**`L643`**]{#lh:L643} & `Leverage.chemical_component_variation_changes_utility`

\
[**`L644`**]{#lh:L644} & `Leverage.chemical_component_variation_can_change_opt`

& [**`L645`**]{#lh:L645} & `Leverage.nStepConformerPopulation_transport`

\
[**`L646`**]{#lh:L646} & `Leverage.kinetic_observable_bundle_with_confidence`

& [**`L647`**]{#lh:L647} & `Leverage.kinetic_protocol_confidence_transport`

\
[**`L648`**]{#lh:L648} & `Leverage.constructive_only_spec_iff_legacy_spec`

& [**`L649`**]{#lh:L649} & `Leverage.paper4_witness_chain_to_paper3_endpoints`

\
[**`L650`**]{#lh:L650} & `Leverage.continuous_langevin_closure_of_measure_theoretic_assumptions`

& [**`L651`**]{#lh:L651} & `Leverage.ContinuousLangevinMeasureClosure.stationary_all_scales`

\
[**`L652`**]{#lh:L652} & `Leverage.concreteComposedHamiltonian_pairwise_lipschitz_bound`

& [**`L653`**]{#lh:L653} & `Leverage.concreteComposedHamiltonian_pairwise_halfGapTransport`

\
[**`L654`**]{#lh:L654} & `Leverage.calibratedBiophysical_changes_utility_at_fixed_core`

& [**`L655`**]{#lh:L655} & `Leverage.calibratedBiophysical_can_change_opt`

\
[**`L656`**]{#lh:L656} & `Leverage.EnsemblePopulationNoiseModel.observable_expectation_error_bound`

& [**`L657`**]{#lh:L657} & `Leverage.kinetic_protocol_inference_guarantee`

\
[**`L658`**]{#lh:L658} & `Leverage.constructive_only_core_field_consumers`

& [**`L659`**]{#lh:L659} & `Leverage.continuous_langevin_closure_of_ito_fokkerplanck_harris`

\
[**`L660`**]{#lh:L660} & `Leverage.concreteComposedHamiltonian_pairwise_lipschitz_bound_of_geometric_assumptions`

& [**`L661`**]{#lh:L661} & `Leverage.concreteComposedHamiltonian_pairwise_halfGapTransport_of_geometric_assumptions`

\
[**`L662`**]{#lh:L662} & `Leverage.ContinuousLangevinMeasureClosure.explicit_sde_constants_of_pairwise_geometry`

& [**`L663`**]{#lh:L663} & `Leverage.ChemicalPosteriorCalibration.robust_protonation_utility_separation`

\
[**`L664`**]{#lh:L664} & `Leverage.ChemicalCalibrationDataset.posterior_protonation_margin`

& [**`L665`**]{#lh:L665} & `Leverage.kinetic_protocol_inference_guarantee_of_concentration_identifiability`

\
[**`L666`**]{#lh:L666} & `Leverage.constructive_only_extended_field_consumers`

& [**`L667`**]{#lh:L667} & `Leverage.ContinuousLangevinMeasureClosure.forcefield_lipschitz_from_pairwise_geometry`

\
[**`L668`**]{#lh:L668} & `Leverage.paper4_tur_bound_of_certificate`

& [**`L669`**]{#lh:L669} & `Leverage.paper4_velocity_verlet_shadow_hamiltonian_bound_of_certificate`

\
[**`L670`**]{#lh:L670} & `Leverage.paper4_extended_witness_chain_to_paper3_endpoints`

& [**`L671`**]{#lh:L671} & `Leverage.paper4_stochastic_relevance_conjecture_of_full_support`

\
[**`L672`**]{#lh:L672} & `Leverage.paper4_stochastic_relevance_containment_necessary_of_nonneg`

& [**`L673`**]{#lh:L673} & `Leverage.paper4_stochastic_relevance_equivalence_of_nonneg_support_bridge`

\
[**`L674`**]{#lh:L674} & `Leverage.paper4_stochastic_relevance_conjecture_of_nonneg_support_transport`

& [**`L675`**]{#lh:L675} & `Leverage.concreteComposedHamiltonian_with_realism_lipschitz_bound`

\
[**`L676`**]{#lh:L676} & `Leverage.concreteComposedHamiltonian_with_realism_halfGapTransport`

& [**`L677`**]{#lh:L677} & `Leverage.paper4_ewald_long_range_certificates`

\
[**`L678`**]{#lh:L678} & `Leverage.FlexibleContinuumDockingLayer.fullstate_energy_lipschitz_transport`

& [**`L679`**]{#lh:L679} & `Leverage.continuous_langevin_closure_of_infinite_dimensional_path_measure`

\
[**`L680`**]{#lh:L680} & `Leverage.ChemicalRealDataCalibrationLayer.robust_protonation_utility_separation_with_bias`

& [**`L681`**]{#lh:L681} & `Leverage.kinetic_replicate_inference_bundle`

\
[**`L682`**]{#lh:L682} & `Leverage.paper4_stochastic_relevance_conjecture_of_nonneg_primitive_dynamics`

& [**`L683`**]{#lh:L683} & `Leverage.ConstructiveEmpiricalRealismInstantiation.finite_sample_shell_envelope`

\
[**`L684`**]{#lh:L684} & `Leverage.constructive_empirical_realism_lipschitz_bound`

& [**`L685`**]{#lh:L685} & `Leverage.constructive_empirical_realism_halfGapTransport`

\
[**`L686`**]{#lh:L686} & `Leverage.continuous_langevin_closure_of_constructive_infinite_dimensional_path_derivation`

& [**`L687`**]{#lh:L687} & `Leverage.continuous_langevin_closure_of_constructive_infinite_dimensional_path_measure`

\
[**`L688`**]{#lh:L688} & `Leverage.hierarchical_chemical_realdata_separation_of_rate`

& [**`L689`**]{#lh:L689} & `Leverage.hierarchical_kinetic_replicate_inference_bundle`

\
[**`L690`**]{#lh:L690} & `Leverage.paper4_stochastic_relevance_conjecture_of_nonneg_explicit_step_dynamics`

& [**`L691`**]{#lh:L691} & `Leverage.biomolecularRealismReferenceCalibration_shell_values`

\
[**`L692`**]{#lh:L692} & `Leverage.biomolecularRealismReference_constructive_lipschitz_bound`

& [**`L693`**]{#lh:L693} & `Leverage.biomolecularRealismReference_constructive_halfGapTransport`

\
[**`L694`**]{#lh:L694} & `Leverage.continuous_langevin_closure_of_finite_horizon_law_derivation`

& [**`L695`**]{#lh:L695} & `Leverage.continuous_langevin_closure_of_finite_horizon_law_bridge`

\
[**`L696`**]{#lh:L696} & `Leverage.hierarchical_chemical_realdata_separation_of_rate_margin`

& [**`L697`**]{#lh:L697} & `Leverage.hierarchical_kinetic_replicate_inference_bundle_of_rate_margins`

\
[**`L698`**]{#lh:L698} & `Leverage.paper4_support_transport_of_explicit_step_dynamics`

& [**`L699`**]{#lh:L699} & `Leverage.paper4_stochastic_relevance_conjecture_of_nonneg_support_transport_of_explicit_step_dynamics`

\
[**`L700`**]{#lh:L700} & `Leverage.biomolecularRealismReference_correction_shell_constant_value`

& [**`L701`**]{#lh:L701} & `Leverage.biomolecularRealismReference_constructive_lipschitz_bound_explicit_shell`

\
[**`L702`**]{#lh:L702} & `Leverage.finite_horizon_law_bridge_projective_and_extension_marginals`

& [**`L703`**]{#lh:L703} & `Leverage.hierarchical_chemical_realdata_separation_of_rate_margin_all_datasets`

\
[**`L704`**]{#lh:L704} & `Leverage.hierarchical_kinetic_replicate_inference_bundle_of_rate_margins_all_datasets`

& [**`L705`**]{#lh:L705} & `Leverage.HierarchicalChemicalRealDataLayer.dataset_rate_margin_of_rate_constant_bound`

\
[**`L706`**]{#lh:L706} & `Leverage.hierarchical_chemical_realdata_separation_of_rate_constant_margin`

& [**`L707`**]{#lh:L707} & `Leverage.hierarchical_chemical_realdata_separation_of_rate_constant_margin_all_datasets`

\
[**`L708`**]{#lh:L708} & `Leverage.HierarchicalKineticReplicateLayer.rate_term_le_of_rate_constant_bound`

& [**`L709`**]{#lh:L709} & `Leverage.hierarchical_kinetic_replicate_inference_bundle_of_rate_constant_margins`

\
[**`L710`**]{#lh:L710} & `Leverage.hierarchical_kinetic_replicate_inference_bundle_of_rate_constant_margins_all_datasets`

& [**`L711`**]{#lh:L711} & `Leverage.microscopic_langevin_explicit_sde_conditions`

\
[**`L712`**]{#lh:L712} & `Leverage.microscopic_langevin_endpoints_of_derivation`

& [**`L713`**]{#lh:L713} & `Leverage.continuous_state_relevance_transport_iff_of_measurable_encoding_bridge`

\
[**`L714`**]{#lh:L714} & `Leverage.continuous_state_partition_fiber_image_of_measurable_encoding_bridge`

& [**`L715`**]{#lh:L715} & `Leverage.continuous_state_srank_transport_of_measurable_encoding_bridge`

\
[**`L716`**]{#lh:L716} & `Leverage.continuous_state_energyLowerBound_transport_of_measurable_encoding_bridge`

& [**`L717`**]{#lh:L717} & `Leverage.electronicStructureCorrection_lipschitz`

\
[**`L718`**]{#lh:L718} & `Leverage.electronicStructureCorrection_abs_error_bound`

& [**`L719`**]{#lh:L719} & `Leverage.concreteComposedHamiltonian_with_electronic_realism_lipschitz_bound`

\
[**`L720`**]{#lh:L720} & `Leverage.concreteComposedHamiltonian_with_electronic_realism_energy_error_bound`

& [**`L721`**]{#lh:L721} & `Leverage.concreteComposedHamiltonian_with_electronic_realism_halfGapTransport`

\
[**`L722`**]{#lh:L722} & `Leverage.biomolecularElectronicReference_shell_error_values`

& [**`L723`**]{#lh:L723} & `Leverage.biomolecularElectronicReference_lipschitz_bound`

\
[**`L724`**]{#lh:L724} & `Leverage.biomolecularElectronicReference_energy_error_bound`

& [**`L725`**]{#lh:L725} & `Leverage.microscopic_langevin_ergodic_of_dissipativity`

\
[**`L726`**]{#lh:L726} & `Leverage.continuous_state_transport_bundle_of_measurable_encoding_bridge`

& [**`L727`**]{#lh:L727} & `Leverage.biomolecularElectronicReference_halfGapTransport`

\
[**`L728`**]{#lh:L728} & `Leverage.MolecularHamiltonianThermostatMicroscopicSystem.hamiltonian_energy_matches_realism`

& [**`L729`**]{#lh:L729} & `Leverage.molecular_hamiltonian_thermostat_to_first_principles`

\
[**`L730`**]{#lh:L730} & `Leverage.molecular_hamiltonian_thermostat_explicit_sde_conditions`

& [**`L731`**]{#lh:L731} & `Leverage.molecular_hamiltonian_thermostat_endpoints`

\
[**`L732`**]{#lh:L732} & `Leverage.canonical_continuous_path_process_closure`

& [**`L733`**]{#lh:L733} & `Leverage.canonical_continuous_path_process_marginal_recovery`

\
[**`L734`**]{#lh:L734} & `Leverage.QMGroundedElectronicStructureLayer.base_shell_le_tight`

& [**`L735`**]{#lh:L735} & `Leverage.QMGroundedElectronicStructureLayer.base_error_le_tight`

\
[**`L736`**]{#lh:L736} & `Leverage.qm_grounded_electronic_realism_lipschitz_bound`

& [**`L737`**]{#lh:L737} & `Leverage.qm_grounded_electronic_realism_energy_error_bound`

\
[**`L738`**]{#lh:L738} & `Leverage.unified_chemical_state_dynamics_observable_transport_bundle`

& [**`L739`**]{#lh:L739} & `Leverage.UnifiedChemicalStateDynamics.expected_utility_separation_of_margin`

\
[**`L740`**]{#lh:L740} & `Leverage.AbsoluteBindingFreeEnergyCorrections.totalCorrection_abs_le_componentBound`

& [**`L741`**]{#lh:L741} & `Leverage.AbsoluteBindingFreeEnergyModel.corrected_value`

\
[**`L742`**]{#lh:L742} & `Leverage.AbsoluteBindingFreeEnergyModel.total_error_bound`

& [**`L743`**]{#lh:L743} & `Leverage.UnifiedDockingPhysicalModel.thermo_kinetic_joint_bundle`

\
[**`L744`**]{#lh:L744} & `Leverage.UniversalityOODLayer.ood_error_bound`

& [**`L745`**]{#lh:L745} & `Leverage.UniversalityOODLayer.uniform_ood_error_bound`

\
[**`L746`**]{#lh:L746} & `Leverage.hierarchical_chemical_realdata_separation_of_required_sample_size`

& [**`L747`**]{#lh:L747} & `Leverage.hierarchical_kinetic_replicate_inference_bundle_of_required_replicate_size`

\
[**`L748`**]{#lh:L748} & `Leverage.ProspectiveBenchmarkRecord.empirical_closure_bundle`

& [**`L749`**]{#lh:L749} & `Leverage.ProspectiveBenchmarkRecord.blinded_predictive_reliability`

\
[**`L750`**]{#lh:L750} & `Leverage.molecular_hamiltonian_thermostat_canonical_path_process_bundle`

& [**`L751`**]{#lh:L751} & `Leverage.qm_grounded_electronic_realism_halfGapTransport`

\
[**`L752`**]{#lh:L752} & `Leverage.unified_physical_model_ood_prospective_bundle`

& [**`L753`**]{#lh:L753} & `Leverage.hierarchical_required_size_joint_bundle`

\
[**`L754`**]{#lh:L754} & `Leverage.ConcreteHamiltonianBathLangevinSystem.langevin_endpoints`

& [**`L755`**]{#lh:L755} & `Leverage.canonical_continuous_path_process_closure_of_finite_horizon_bridge`

\
[**`L756`**]{#lh:L756} & `Leverage.qm_method_specific_realism_transport_bundle`

& [**`L757`**]{#lh:L757} & `Leverage.ChemicalConditionedTransitionMechanism.unified_dynamics_stationary_bundle`

\
[**`L758`**]{#lh:L758} & `Leverage.AbsoluteFreeEnergyProtocolCalibration.total_error_bound`

& [**`L759`**]{#lh:L759} & `Leverage.UnifiedPhysicalSimulatorPipeline.thermo_kinetic_bundle`

\
[**`L760`**]{#lh:L760} & `Leverage.OODTransferCalibration.uniform_ood_transfer_bound`

& [**`L761`**]{#lh:L761} & `Leverage.hierarchical_chemical_realdata_separation_of_required_sample_size_of_count_order`

\
[**`L762`**]{#lh:L762} & `Leverage.hierarchical_kinetic_replicate_inference_bundle_of_required_replicate_size_of_count_order`

& [**`L763`**]{#lh:L763} & `Leverage.hierarchical_required_size_joint_bundle_of_count_order`

\
[**`L764`**]{#lh:L764} & `Leverage.GibbsConditionedChemicalMechanismData.unified_dynamics_stationary_bundle`

& [**`L765`**]{#lh:L765} & `Leverage.qm_protocol_derived_method_specific_realism_transport_bundle`

\
[**`L766`**]{#lh:L766} & `Leverage.rate_div_sqrt_nat_le_of_square_count_bound`

& [**`L767`**]{#lh:L767} & `Leverage.HierarchicalChemicalRealDataLayer.rate_term_le_of_required_sample_square_bound`

\
[**`L768`**]{#lh:L768} & `Leverage.HierarchicalKineticReplicateLayer.rate_term_le_of_required_replicate_square_bound`

& [**`L769`**]{#lh:L769} & `Leverage.hierarchical_required_size_joint_bundle_of_square_count_bounds`

\
[**`L770`**]{#lh:L770} & `Leverage.explicit_hamiltonian_bath_elimination_langevin_endpoints`

& [**`L771`**]{#lh:L771} & `Leverage.explicit_molecular_constant_drift_sde_endpoints`

\
[**`L772`**]{#lh:L772} & `Leverage.concrete_generator_canonical_path_process_closure`

& [**`L773`**]{#lh:L773} & `Leverage.qm_workflow_specific_realism_transport_bundle`

\
[**`L774`**]{#lh:L774} & `Leverage.ReversibleChemicalTransitionMechanism.unified_dynamics_detailed_balance_bundle`

& [**`L775`**]{#lh:L775} & `Leverage.TrajectoryAbsoluteFreeEnergyCalibration.total_error_bound`

\
[**`L776`**]{#lh:L776} & `Leverage.UnifiedPhysicalQuantifiedSimulatorPipeline.thermo_kinetic_bundle`

& [**`L777`**]{#lh:L777} & `Leverage.MechanisticOODTransferModel.uniform_ood_transfer_bound`

\
[**`L778`**]{#lh:L778} & `Leverage.ModelDependentFiniteSampleLaw.rate_term_le_targetMargin`

& [**`L779`**]{#lh:L779} & `Leverage.hierarchical_required_size_joint_bundle_of_model_dependent_constants`

\
[**`L780`**]{#lh:L780} & `Leverage.hamiltonian_finite_difference_drift_derivation_endpoints`

& [**`L781`**]{#lh:L781} & `Leverage.realistic_molecular_finite_difference_sde_endpoints`

\
[**`L782`**]{#lh:L782} & `Leverage.generator_coefficients_to_canonical_regularity_closure`

& [**`L783`**]{#lh:L783} & `Leverage.concrete_qm_workflow_error_analysis_transport_bundle`

\
[**`L784`**]{#lh:L784} & `Leverage.barrier_crossing_reversible_chemical_dynamics_bundle`

& [**`L785`**]{#lh:L785} & `Leverage.mixing_autocorrelation_trajectory_correction_total_error_bound`

\
[**`L786`**]{#lh:L786} & `Leverage.UnifiedSimulatorErrorAnalysis.unified_controlled_thermo_kinetic_bundle`

& [**`L787`**]{#lh:L787} & `Leverage.descriptor_calibrated_mechanistic_ood_transfer_bundle`

\
[**`L788`**]{#lh:L788} & `Leverage.model_dependent_minimax_optimality_bundle`

& [**`L789`**]{#lh:L789} & `Leverage.ConstructiveStochasticAnalysisMultipoleClosure.scope_gap_discharge_bundle`

\
[**`L790`**]{#lh:L790} & `Leverage.ito_wiener_filtration_langevin_endpoints`

& [**`L791`**]{#lh:L791} & `Leverage.hamiltonian_mori_zwanzig_h_zero_limit_endpoints`

\
[**`L792`**]{#lh:L792} & `Leverage.forcefield_derived_realistic_sde_endpoints`

& [**`L793`**]{#lh:L793} & `Leverage.generator_pde_estimates_to_canonical_regularity_closure`

\
[**`L794`**]{#lh:L794} & `Leverage.qm_workflow_transport_of_benchmark_summary`

& [**`L795`**]{#lh:L795} & `Leverage.potential_landscape_barrier_kinetics_bundle`

\
[**`L796`**]{#lh:L796} & `Leverage.spectral_gap_concentration_trajectory_bundle`

& [**`L797`**]{#lh:L797} & `Leverage.spectral_gap_absolute_free_energy_total_error_bound`

\
[**`L798`**]{#lh:L798} & `Leverage.UnifiedSimulatorIntegratorErrorStack.unified_controlled_thermo_kinetic_bundle`

& [**`L799`**]{#lh:L799} & `Leverage.learned_descriptor_ood_generalization_transfer_bundle`

\
[**`L800`**]{#lh:L800} & `Leverage.estimator_minimax_derivation_bundle`

& [**`L801`**]{#lh:L801} & `Leverage.extended_physical_model_interface_scope_bundle`

\
[**`L802`**]{#lh:L802} & `Leverage.pre_registered_prospective_benchmark_beats_strong_baselines_bundle`

& [**`L803`**]{#lh:L803} & `Leverage.independent_replication_outside_team_bundle`

\
[**`L804`**]{#lh:L804} & `Leverage.downstream_campaign_win_bundle`

& [**`L805`**]{#lh:L805} & `Leverage.external_validation_threeway_integration_bundle`

\
[**`L806`**]{#lh:L806} & `Leverage.extended_physical_model_interface_scope_bundle_of_constructive_closure`

& [**`L807`**]{#lh:L807} & `Leverage.ContractBoundProspectiveBenchmarkResults.fixed_contract_pre_registered_bundle`

\
[**`L808`**]{#lh:L808} & `Leverage.IndependentReplicationProvenance.outside_team_compute_signed_bundle`

& [**`L809`**]{#lh:L809} & `Leverage.DownstreamCausalCampaignEvidence.causal_quality_downstream_win_bundle`

\
[**`L810`**]{#lh:L810} & `Leverage.concrete_external_validation_threeway_bundle`

& [**`L811`**]{#lh:L811} & `Leverage.concrete_external_validation_not_credibly_dismissible`

\
[**`L812`**]{#lh:L812} & `Leverage.langevin_measure_theoretic_endpoint_bundle`

& [**`L813`**]{#lh:L813} & `Leverage.constructive_ito_wiener_filtration_langevin_endpoints`

\
[**`L814`**]{#lh:L814} & `Leverage.derived_generator_pde_operator_to_canonical_regularity_closure`

& [**`L815`**]{#lh:L815} & `Leverage.extended_physical_model_interface_scope_bundle_of_microscopic_derivation`

\
[**`L816`**]{#lh:L816} & `Leverage.UnifiedSimulatorIntegratorErrorStack.control_flags_of_numerical_stack`

& [**`L817`**]{#lh:L817} & `Leverage.attested_concrete_external_validation_threeway_bundle`

\
[**`L818`**]{#lh:L818} & `Leverage.attested_concrete_external_validation_not_credibly_dismissible`

& [**`L819`**]{#lh:L819} & `Leverage.attested_concrete_artifacts_store_backed_bundle`

\
[**`L820`**]{#lh:L820} & `Leverage.store_backed_attested_concrete_external_validation_threeway_bundle`

& [**`L821`**]{#lh:L821} & `Leverage.store_backed_attested_concrete_external_validation_not_credibly_dismissible`

\
[**`L822`**]{#lh:L822} & `Leverage.molecularDocking_toDecisionProblem_utility_eq_physicalUtility`

& [**`L823`**]{#lh:L823} & `Leverage.molecularDocking_srank_eq_relevant_coordinate_card`

\
[**`L824`**]{#lh:L824} & `Leverage.molecularDocking_srank_bound_of_excludedProteinAtoms`

& [**`L825`**]{#lh:L825} & `Leverage.molecularDocking_binarySummary_srank_le`

\
[**`L826`**]{#lh:L826} & `Leverage.equilibriumKdOfDrivingEnergy_le_of_drive_floor`

& [**`L827`**]{#lh:L827} & `Leverage.molecularDocking_equilibrium_pathRatio_eq_one_of_detailedBalance`

\
[**`L828`**]{#lh:L828} & `Leverage.molecularDocking_equilibrium_freeEnergy_pathRatio_bundle`

& [**`L829`**]{#lh:L829} & `Leverage.molecularDocking_equilibriumKd_upper_bound_of_rank_lower_bound`

\
[**`L830`**]{#lh:L830} & `Leverage.molecularDocking_contactShell_budget_necessary_of_rank_lower_bound`

& [**`L831`**]{#lh:L831} & `Leverage.molecularDocking_independent_rank_empirical_prediction_bundle`

\
[**`L832`**]{#lh:L832} & `Leverage.molecularDocking_witness_bundle_of_exactLJ_physics`

& [**`L833`**]{#lh:L833} & `Leverage.ConcreteDockingPhysicsWitness.derived_witness_bundle`

\
[**`L834`**]{#lh:L834} & `Leverage.driving_free_energy_floor_of_partition_ratio_margin`

& [**`L835`**]{#lh:L835} & `Leverage.equilibriumKd_upper_bound_of_partition_ratio_drive_floor`

\
[**`L836`**]{#lh:L836} & `Leverage.molecularDocking_equilibriumKd_upper_bound_of_rank_lower_bound_from_partition_chain`

& [**`L837`**]{#lh:L837} & `Leverage.molecularDocking_srank_interval_of_independent_certificates`

\
[**`L838`**]{#lh:L838} & `Leverage.molecularDocking_independent_certificate_empirical_prediction_bundle`

& [**`L839`**]{#lh:L839} & `Leverage.DockingRankFalsificationProtocol.not_falsified_iff`

\
[**`L840`**]{#lh:L840} & `Leverage.molecularDockingFalsificationProtocolFromIndependentRank_prediction_bundle`

& [**`L841`**]{#lh:L841} & `Leverage.TrajectoryCorrectionEstimator.upper_bound_violation_with_margin_implies_true_violation`

\
[**`L842`**]{#lh:L842} & `Leverage.DockingRankFalsificationProtocol.high_confidence_fail_condition_bundle`

& [**`L843`**]{#lh:L843} & `Leverage.equilibriumKd_interval_of_drivingEnergy_abs_error`

\
[**`L844`**]{#lh:L844} & `Leverage.equilibriumKd_interval_of_absoluteFreeEnergyModel`

& [**`L845`**]{#lh:L845} & `Leverage.chemistry_condition_data_backed_kd_interval_bundle`

\
[**`L846`**]{#lh:L846} & `Leverage.PerSystemWitnessDischargeProgram.discharge_all_cases`

& [**`L847`**]{#lh:L847} & `Leverage.PartitionChainNumericalClosure.kd_upper_bound_bundle`

\
[**`L848`**]{#lh:L848} & `Leverage.ProductionIndependentSrankExtractor.certified_interval_bundle`

& [**`L849`**]{#lh:L849} & `Leverage.LockedProspectiveFalsificationRun.falsification_bundle`

\
[**`L850`**]{#lh:L850} & `Leverage.AssayNoiseCalibrationModel.high_confidence_call_validity_bundle`

& [**`L851`**]{#lh:L851} & `Leverage.TargetClassChemistryCalibration.kd_interval_bundle`

\
[**`L852`**]{#lh:L852} & `Leverage.FullPhysicalClosureTargetInstance.bundle`

& [**`L853`**]{#lh:L853} & `Leverage.ExternalReplicationAtScale.full_pipeline_bundle`

\
[**`L854`**]{#lh:L854} & `Leverage.CertifiedPartitionFunctionComputation.singleStateExact_positive_bundle`

& [**`L855`**]{#lh:L855} & `Leverage.AbsoluteFreeEnergyProtocolCalibration.zeroOneSample_total_error_bundle`

\
[**`L856`**]{#lh:L856} & `Leverage.ProductionIndependentSrankExtractor.ofUpperSufficiency_interval_bundle`

& [**`L857`**]{#lh:L857} & `Leverage.FullPhysicalClosureTargetInstance.zeroRankConcreteOfUpperSufficiency_bundle`

\
[**`L858`**]{#lh:L858} & `Leverage.ExternalReplicationAtScale.ofAttestedIndependentReplicationProvenance_bundle`

& [**`L859`**]{#lh:L859} & `Leverage.concrete_attested_single_target_external_replication_bundle`

\
[**`L860`**]{#lh:L860} & `Leverage.computableBestPoseByUtility_mem_opt`

& [**`L861`**]{#lh:L861} & `Leverage.RMSDPosteriorModel.successProbability_unit_interval`

\
[**`L862`**]{#lh:L862} & `Leverage.RMSDPosteriorModel.dockingTopK_mass_le_successProbability_of_covered`

& [**`L863`**]{#lh:L863} & `Leverage.RMSDProbabilityDerivedPoseSolver.bundle`

\
[**`L864`**]{#lh:L864} & `Leverage.computable_pose_solver_and_rmsd_probability_bundle`

& [**`L865`**]{#lh:L865} & `Leverage.sampledDockingSolverInputFromRawPocketLigand_bundle`

\
[**`L866`**]{#lh:L866} & `Leverage.DeploymentRMSDCalibration.deployment_contract_implies_benchmark_contract`

& [**`L867`**]{#lh:L867} & `Leverage.CanonicalProgramExecutionWitness.refines_solver_result`

\
[**`L868`**]{#lh:L868} & `Leverage.solveRawPocketLigandBenchmark_bundle`

& [**`L869`**]{#lh:L869} & `Leverage.solveRawPocketLigandCanonicalBenchmark_accept_iff`

\
[**`L870`**]{#lh:L870} & `Leverage.solveRawPocketLigandCanonicalDeployment_accepted_iff`

& [**`L871`**]{#lh:L871} & `Leverage.rawCanonicalProgramExecutionWitnessOfSpec_refines_solver_result_canonical`

\
[**`L872`**]{#lh:L872} & `Leverage.rawPocketLigandCanonicalDefinitiveEndpoint_bundle`

& [**`L873`**]{#lh:L873} & `Leverage.runCanonicalSolverProgram_refines_solver_result`

\
[**`L874`**]{#lh:L874} & `Leverage.solveDefinitiveRawCrossDock_accepted_iff_benchmark_contract`

& [**`L875`**]{#lh:L875} & `Leverage.solveDefinitiveRawCrossDock_accepted_iff_deployment_contract`

\
[**`L876`**]{#lh:L876} & `Leverage.solveDefinitiveRawCrossDock_total`

& [**`L877`**]{#lh:L877} & `Leverage.runDefinitiveRawCrossDockProgram_refines_benchmark_accept`

\
[**`L878`**]{#lh:L878} & `Leverage.solveDefinitiveRawCrossDock_full_closure_bundle`

& [**`L879`**]{#lh:L879} & `Leverage.solveDefinitiveRawCrossDockBenchmark_accepted_iff_benchmark_contract`

\
[**`L880`**]{#lh:L880} & `Leverage.solveDefinitiveRawCrossDock_rejected_iff_not_deployment_contract`

& [**`L881`**]{#lh:L881} & `Leverage.definitiveRawCrossDockAcceptanceFlag_true_iff_deployment_accepted`

\
[**`L882`**]{#lh:L882} & `Leverage.definitiveRawCrossDockAcceptanceFlag_false_iff_deployment_rejected`

& [**`L883`**]{#lh:L883} & `Leverage.RationalizedAcceptanceKernel.computableAcceptFlag_true_iff`

\
[**`L884`**]{#lh:L884} & `Leverage.RationalizedAcceptanceKernel.computableAcceptFlag_sound`

& [**`L885`**]{#lh:L885} & `Leverage.RationalizedAcceptanceKernel.computableAcceptFlag_refines_benchmark_accept`

\
[**`L886`**]{#lh:L886} & `Leverage.interpretCanonicalProgramState_refines_solver_result`

& [**`L887`**]{#lh:L887} & `Leverage.interpretDefinitiveRawCrossDockProgram_eq_run`

\
[**`L888`**]{#lh:L888} & `Leverage.buildDefinitiveRawCrossDockReport_runtime_accept_iff_deployment_accepted`

& [**`L889`**]{#lh:L889} & `Leverage.definitiveRawCrossDockCompleteLeanBundle`

\
[**`L890`**]{#lh:L890} & `Leverage.buildDefinitiveRawCrossDockReport_runtime_reject_iff_deployment_rejected`

& [**`L891`**]{#lh:L891} & `Leverage.solveDefinitiveRawCrossDockBenchmarkConstructive_accepted_iff_kernel_flag`

\
[**`L892`**]{#lh:L892} & `Leverage.solveDefinitiveRawCrossDockBenchmarkConstructive_refines_benchmark_accept`

& [**`L893`**]{#lh:L893} & `Leverage.solveDefinitiveRawCrossDockConstructive_refines_deployment_accept`

\
[**`L894`**]{#lh:L894} & `Leverage.ExactRatArtifactInstantiation.constructive_accepted_iff_benchmark_contract`

& [**`L895`**]{#lh:L895} & `Leverage.ExactRatArtifactInstantiation.constructive_accept_refines_certificate_backend_accepts`

\
[**`L896`**]{#lh:L896} & `Leverage.solveDefinitiveRawCrossDockBenchmarkDecision_eq_constructive`

& [**`L897`**]{#lh:L897} & `Leverage.solveDefinitiveRawCrossDockDecision_eq_constructive`

\
[**`L898`**]{#lh:L898} & `Leverage.solveDefinitiveRawCrossDockBenchmarkDecision_refines_certificate_backend_accept`

& [**`L899`**]{#lh:L899} & `Leverage.solveDefinitiveRawCrossDockDecision_refines_certificate_backend_accept`

\
[**`L900`**]{#lh:L900} & `Leverage.solveDefinitiveRawCrossDockBenchmarkDecision_rejected_iff_kernel_flag_false`

& [**`L901`**]{#lh:L901} & `Leverage.solveDefinitiveRawCrossDockBenchmarkCertified_accepted_iff`

\
[**`L902`**]{#lh:L902} & `Leverage.solveDefinitiveRawCrossDockBenchmarkCertified_rejected_iff`

& [**`L903`**]{#lh:L903} & `Leverage.solveDefinitiveRawCrossDockCertified_accepted_iff`

\
[**`L904`**]{#lh:L904} & `Leverage.solveDefinitiveRawCrossDockCertified_rejected_iff`

& [**`L905`**]{#lh:L905} & `Leverage.SignedRationalizedKernelArtifact.manifest_consistency_bundle`

\
[**`L906`**]{#lh:L906} & `Leverage.SignedRationalizedKernelArtifact.decision_accept_refines_certificate_backend_accept`

& [**`L907`**]{#lh:L907} & `Leverage.ExactRatArtifactInstantiation.constructive_rejected_refines_certificate_backend_rejections`

\
[**`L908`**]{#lh:L908} & `Leverage.SignedExactRatKernelArtifact.benchmark_decision_accepted_iff_benchmark_contract`

& [**`L909`**]{#lh:L909} & `Leverage.SignedExactRatKernelArtifact.decision_rejected_refines_certificate_backend_rejections`

\
[**`L910`**]{#lh:L910} & `Leverage.parseSignedArtifactByteEnvelope_encode`

& [**`L911`**]{#lh:L911} & `Leverage.concreteChecksum_parse_verify_end_to_end`

\
[**`L912`**]{#lh:L912} & `Leverage.SignedRationalizedKernelArtifact.concreteChecksum_byte_parse_and_verify`

& [**`L913`**]{#lh:L913} & `Leverage.RationalizedAcceptanceKernel.rejectionSeparation_not_benchmarkContract`

\
[**`L914`**]{#lh:L914} & `Leverage.RationalizedAcceptanceKernel.rejectionSeparation_flag_false`

& [**`L915`**]{#lh:L915} & `Leverage.SignedRationalizedKernelArtifact.strict_rejection_refines_certificate_backend_rejections`

\
[**`L916`**]{#lh:L916} & `Leverage.concreteChecksumArtifactSignatureVerifier_verify_iff`

& [**`L917`**]{#lh:L917} & `Leverage.definitiveComputableTotalOps_closed_form`

\
[**`L918`**]{#lh:L918} & `Leverage.definitiveComputableTotalOps_succFuel`

& [**`L919`**]{#lh:L919} & `Leverage.runDefinitiveComputablePipeline_totalOps_closed_form`

\
[**`L920`**]{#lh:L920} & `Leverage.branchAndBoundPrune_sound`

& [**`L921`**]{#lh:L921} & `Leverage.adaptiveCampaignStopRule_sound`

\
[**`L922`**]{#lh:L922} & `Leverage.runDefinitiveComputablePipelineBranchAndBound_prune_sound`

& [**`L923`**]{#lh:L923} & `Leverage.definitiveComputablePipeline_batchFusionJustified`

\
[**`L924`**]{#lh:L924} & `Leverage.parseSignedArtifactByteEnvelope_cost_linear_time`

& [**`L925`**]{#lh:L925} & `Leverage.parseSignedArtifactByteEnvelope_encode_cost_exact`

\
[**`L926`**]{#lh:L926} & `Leverage.cryptographicArtifactSignatureVerifier_sound`

& [**`L927`**]{#lh:L927} & `Leverage.SignedRationalizedKernelArtifact.crypto_byte_parse_and_verify`

\
[**`L928`**]{#lh:L928} & `Leverage.runDefinitiveComputablePipelineOfSignedArtifact_parserBytes_exact`

& [**`L929`**]{#lh:L929} & `Leverage.definitiveComputableCampaignPairEvaluations_closed_form`

\
[**`L930`**]{#lh:L930} & `Leverage.definitiveComputableCampaignPairEvaluations_succFuel`

& [**`L931`**]{#lh:L931} & `Leverage.runDefinitiveComputablePipeline_campaignPairEvaluations_closed_form`

\
[**`L932`**]{#lh:L932} & `Leverage.definitiveComputablePipeline_pairPotentialFusionJustified`

& [**`L933`**]{#lh:L933} & `Leverage.SampledDockingSolverInput.canonicalSolverProgramIR_scorerFusionSound`

\
[**`ORA1`**]{#lh:ORA1} & `oracle_arbitrary`

& [**`PH26`**]{#lh:PH26} & `PhysicalComplexity.no_collapse_of_bounded_budget_pos_cost_exp_lb`

\
[**`QT1`**]{#lh:QT1} & `DecisionProblem.quotient_is_coarsest`

& [**`QT2`**]{#lh:QT2} & `DecisionProblem.quotientMap_preservesOpt`

\
[**`QT3`**]{#lh:QT3} & `DecisionProblem.quotient_represents_opt_equiv`

& [**`QT7`**]{#lh:QT7} & `DecisionProblem.quotient_has_unique_factorization`

\
[**`SE1`**]{#lh:SE1} & `ClaimClosure.SE1`

& [**`SE2`**]{#lh:SE2} & `ClaimClosure.SE2`

\
[**`SE3`**]{#lh:SE3} & `ClaimClosure.SE3`

& [**`SE4`**]{#lh:SE4} & `ClaimClosure.SE4`

\
[**`SE5`**]{#lh:SE5} & `ClaimClosure.SE5`

& [**`SE6`**]{#lh:SE6} & `ClaimClosure.SE6`

\
[**`W1`**]{#lh:W1} & `Physics.single_future_zero_cost`

& [**`W2`**]{#lh:W2} & `Physics.transportCost_pos_of_offDiag`

\
[**`W3`**]{#lh:W3} & `Physics.integrity_is_centroid`

& [**`W4`**]{#lh:W4} & `Physics.wasserstein_bridge`

\
[**`WM4`**]{#lh:WM4} & `Physics.WolpertMismatch.mismatchNatLowerBound_pos_of_exists_ne`

& [**`WM6`**]{#lh:WM6} & `Physics.WolpertDecomposition.effective_model_strictly_exceeds_landauer_of_distribution_mismatch`

\
[**`WP2`**]{#lh:WP2} & `Physics.WolpertDecomposition.landauer_floor_plus_decomposition_lower_bound`

& [**`WP6`**]{#lh:WP6} & `Physics.WolpertDecomposition.effective_model_strictly_exceeds_landauer_of_either_cited_component`

\
[**`WP7`**]{#lh:WP7} & `Physics.WolpertDecomposition.landauer_floor_plus_structural_resource_lower_bound`

& [**`WP8`**]{#lh:WP8} & `Physics.WolpertDecomposition.energy_lower_bound_increases_by_structural_resource`

\
[**`WP9`**]{#lh:WP9} & `Physics.WolpertDecomposition.physical_grounding_bundle_with_wolpert_decomposition`

& [**`WR6`**]{#lh:WR6} & `Physics.WolpertResidual.discreteResidualNatLowerBound_pos_of_asymmetry_or_oneway`

\
[**`WR7`**]{#lh:WR7} & `Physics.WolpertDecomposition.stopping_time_residual_of_discrete_edge_split`

& [**`WR10`**]{#lh:WR10} & `Physics.WolpertDecomposition.effective_model_strictly_exceeds_landauer_of_finite_discrete_witness`

\
[**`WR11`**]{#lh:WR11} & `Physics.WolpertResidual.binaryEncodedResidualNatLowerBound_eq_one`

& [**`WR12`**]{#lh:WR12} & `Physics.WolpertDecomposition.effective_model_ge_landauer_plus_one_of_binary_encoded_residual_example`

\
:::


  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **Paper handle**                                                                               **Hardness profile**   **Regime tags**           **Lean support**
  ---------------------------------------------------------------------------------------------- ---------------------- ------------------------- ------------------------------------------------
  `cor:bond-family-holonomic-floor`                                                              `unspecified`          \-                        L396, L398, L397

  `cor:bounded-contact-shell-regime`                                                             `unspecified`          \-                        L142

  `cor:bounded-pocket-regime`                                                                    `unspecified`          \-                        L65

  `cor:budget-entropy-bound`                                                                     `unspecified`          \-                        IT3, BA1, BA2, BA5, BA6, EI1, L43

  `cor:checking-time-lower-bound`                                                                `unspecified`          \-                        L70

  `cor:composition-budget-law`                                                                   `unspecified`          \-                        L17, L19, BA1, BA2, BA7, BA5, BA6, L43

  `cor:holonomic-landauer-floor`                                                                 `unspecified`          \-                        L60, L61, L62

  `cor:jacobian-holonomic-floor`                                                                 `unspecified`          \-                        L405, L407, L403, L404, L406

  `cor:logical-basement-overhead`                                                                `unspecified`          \-                        L140, BA5, BA6, WM6, WP6, WR10, L43

  `cor:minimum-cost-regime`                                                                      `unspecified`          \-                        BA8, L54, L55

  `cor:no-sound-checker-below-budget`                                                            `unspecified`          \-                        L71

  `cor:rank-above-one`                                                                           `unspecified`          \-                        L46

  `cor:rank-one`                                                                                 `unspecified`          \-                        L51

  `prop:atomic-realization`                                                                      `unspecified`          \-                        AC1, AC3, AC4

  `prop:binary-mismatch-cumulative-work`                                                         `unspecified`          \-                        IT3, BA5, BA6, WP2, WM4, L43

  `prop:binary-mismatch-energy-information`                                                      `unspecified`          \-                        IT3, BA5, BA6, WP2, WM4, L43

  `prop:binary-mismatch-example`                                                                 `unspecified`          \-                        BA5, BA6, WP2, WM4, L43

  `prop:binary-residual-cumulative-work`                                                         `unspecified`          \-                        IT3, BA5, BA6, WR12, WR11, L43

  `prop:binary-residual-example`                                                                 `unspecified`          \-                        IT3, BA5, BA6, WR12, WR11, L43

  `prop:binding-as-exact-resolution`                                                             `unspecified`          \-                        BA3, BA4, BA5

  `prop:bounded-region`                                                                          `unspecified`          \-                        BA1

  `prop:canonical-wolpert-bundle`                                                                `unspecified`          \-                        BA5, BA6, WP9, L43

  `prop:dof-additive`                                                                            `unspecified`          \-                        L17, L19

  `prop:ei-hierarchy`                                                                            `unspecified`          \-                        IT3, BA5, BA6, WR12, WP2, WM4, WR11, L43

  `prop:finite-compression-bridge`                                                               `unspecified`          \-                        L57, L58

  `prop:finite-discrete-residual`                                                                `unspecified`          \-                        WR10, WR7, WR6

  `prop:finite-lifetime`                                                                         `unspecified`          \-                        SE1, SE2, SE3, SE4, SE5

  `prop:lifetime-throughput`                                                                     `unspecified`          \-                        SE5, IT3, L43

  `prop:optimizer-quotient`                                                                      `unspecified`          \-                        QT2, QT7, QT1, QT3

  `prop:speed-heat-tradeoff`                                                                     `unspecified`          \-                        SE6

  `prop:strict-canonical-energy`                                                                 `unspecified`          \-                        BA5, BA6, WP6, L43

  `prop:strict-overhead`                                                                         `unspecified`          \-                        BA5, BA6, WM6, WP6, WR10, L43

  `prop:structural-resource-overhead`                                                            `unspecified`          \-                        BA5, BA6, WP8, WP7, L43

  `prop:substrate-time-law`                                                                      `unspecified`          \-                        DT23, DT22, DT24

  `prop:threshold-channel`                                                                       `unspecified`          \-                        CV8, CV9, CV7

  `thm:absolute-binding-free-energy-closure`                                                     `unspecified`          \-                        L740, L741, L742

  `thm:absolute-free-energy-protocol-derived-closure`                                            `unspecified`          \-                        L758

  `thm:abstraction-factors-or-erases`                                                            `unspecified`          \-                        L77

  `thm:action-pinned-lifting`                                                                    `unspecified`          \-                        L120, L121

  `thm:action-pinned-uniform-gap`                                                                `unspecified`          \-                        L115, L116, L117

  `thm:admissibility-collapse-canonicality`                                                      `unspecified`          \-                        L416, L415

  `thm:admissibility-onrate-envelope`                                                            `unspecified`          \-                        L447, L448

  `thm:admissibility-progress-monotone`                                                          `unspecified`          \-                        L395

  `thm:admissibility-rank-reduction`                                                             `unspecified`          \-                        L118, L119

  `thm:admissibility-speed-accuracy-tradeoff`                                                    `unspecified`          \-                        L457, L446, L458

  `thm:admissibility-speed-accuracy-zero-collapse`                                               `unspecified`          \-                        L482

  `thm:admissibility-zero-collapse-bifactor`                                                     `unspecified`          \-                        L479, L480, L481

  `thm:admissible-docking-exhaustion`                                                            `unspecified`          \-                        L394

  `thm:ae-rg-kernel-endpoint`                                                                    `unspecified`          \-                        L502, L503, L504, L505, L506, L507

  `thm:allosteric-distance-decay-law`                                                            `unspecified`          \-                        L428, L429, L430

  `thm:allosteric-exponential-distance-decay`                                                    `unspecified`          \-                        L431, L432, L433

  `thm:allosteric-polynomial-distance-decay`                                                     `unspecified`          \-                        L437, L438, L439

  `thm:allosteric-polynomial-series-budget`                                                      `unspecified`          \-                        L440, L441

  `thm:allosteric-polynomial-series-explicit`                                                    `unspecified`          \-                        L443, L444, L442

  `thm:allosteric-srank-graph`                                                                   `unspecified`          \-                        L132

  `thm:assay-noise-high-confidence-call-validity-bundle`                                         `unspecified`          \-                        L850

  `thm:attested-concrete-external-validation-not-credibly-dismissible`                           `unspecified`          \-                        L818

  `thm:attested-concrete-external-validation-threeway-bundle`                                    `unspecified`          \-                        L817

  `thm:attested-concrete-store-backed-artifact-bundle`                                           `unspecified`          \-                        L819

  `thm:attested-provenance-replication-at-scale-constructor-bundle`                              `unspecified`          \-                        L858

  `thm:barrier-crossing-reversible-chemical-dynamics`                                            `unspecified`          \-                        L784

  `thm:binding-free-energy-floor`                                                                `unspecified`          \-                        L138

  `thm:binding-free-energy-tightness`                                                            `unspecified`          \-                        L419, L422, L420, L421

  `thm:biomolecular-reference-realism-explicit-shell-constant`                                   `unspecified`          \-                        L701, L700

  `thm:biomolecular-reference-realism-shell-values`                                              `unspecified`          \-                        L691

  `thm:biomolecular-reference-realism-transport`                                                 `unspecified`          \-                        L693, L692

  `thm:bounded-acquisition`                                                                      `unspecified`          \-                        BA1, BA2

  `thm:bounded-actions-poly`                                                                     `unspecified`          \-                        L87, L93

  `thm:bounded-collapsed-rank-algebraic-laws`                                                    `unspecified`          \-                        L470, L464, L467, L465, L469, L468, L466

  `thm:bounded-collapsed-rank-binary-calculus`                                                   `unspecified`          \-                        L478, L475, L474, L476, L477, L472, L471, L473

  `thm:bounded-collapsed-rank-complete-lattice`                                                  `unspecified`          \-                        L463

  `thm:bounded-collapsed-rank-finite-colimits`                                                   `unspecified`          \-                        L460, L459, L461

  `thm:bounded-collapsed-rank-finite-family-lattice`                                             `unspecified`          \-                        L462

  `thm:bounded-potential-large-cutoff-sampled-srank`                                             `unspecified`          \-                        L104

  `thm:bounded-potential-large-cutoff-srank`                                                     `unspecified`          \-                        L95

  `thm:budget-class-bound`                                                                       `unspecified`          \-                        IT4, IT3, BA1, BA2, BA5, BA6, EI1, L43

  `thm:calibrated-biophysical-chemical-separation`                                               `unspecified`          \-                        L655, L654

  `thm:canonical-initiality`                                                                     `unspecified`          \-                        L413

  `thm:canonical-initiality-srank`                                                               `unspecified`          \-                        L414

  `thm:canonical-interpreter-state-runtime-refinement`                                           `unspecified`          \-                        L886

  `thm:canonical-program-execution-refines-solver-result`                                        `unspecified`          \-                        L867

  `thm:canonical-raw-benchmark-acceptance-equivalence`                                           `unspecified`          \-                        L869

  `thm:canonical-raw-definitive-endpoint-bundle`                                                 `unspecified`          \-                        L872

  `thm:canonical-raw-deployment-acceptance-equivalence`                                          `unspecified`          \-                        L870

  `thm:canonical-raw-program-witness-refinement`                                                 `unspecified`          \-                        L871

  `thm:canonical-runtime-output-refines-solver`                                                  `unspecified`          \-                        L873

  `thm:checker-budget-lower-bound`                                                               `unspecified`          \-                        L69

  `thm:chemical-augmented-docking-transport`                                                     `unspecified`          \-                        L608, L609

  `thm:chemical-component-variation-opt-invariance`                                              `unspecified`          \-                        L633

  `thm:chemical-conditioned-mechanism-dynamics`                                                  `unspecified`          \-                        L757

  `thm:chemical-coupled-sensitivity`                                                             `unspecified`          \-                        L644, L643

  `thm:chemical-dataset-posterior-separation`                                                    `unspecified`          \-                        L664, L663

  `thm:chemical-ensemble-transport-binding-specialization`                                       `unspecified`          \-                        L623, L624, L625, L626

  `thm:chemical-realdata-bias-aware-separation`                                                  `unspecified`          \-                        L680

  `thm:chemistry-data-backed-kd-interval-bundle`                                                 `unspecified`          \-                        L845

  `thm:coherent-single-source`                                                                   `unspecified`          \-                        ORA1

  `thm:collapsed-rank-category-structure`                                                        `unspecified`          \-                        L450, L449, L454, L451, L453, L456, L455, L452

  `thm:composed-classical-forcefield-interface`                                                  `unspecified`          \-                        L589, L590

  `thm:composed-hamiltonian-architecture-instance`                                               `unspecified`          \-                        L589

  `thm:composed-hamiltonian-lipschitz-bound`                                                     `unspecified`          \-                        L590

  `thm:computable-finite-enumeration-pose-solver-optimal`                                        `unspecified`          \-                        L860

  `thm:computable-rational-accept-flag-exactness`                                                `unspecified`          \-                        L883

  `thm:computable-rational-accept-refines-benchmark-accept`                                      `unspecified`          \-                        L885

  `thm:computable-rational-accept-soundness`                                                     `unspecified`          \-                        L884

  `thm:concrete-attested-single-target-replication-bundle`                                       `unspecified`          \-                        L859

  `thm:concrete-biomolecular-forcefield-calibration-bundle`                                      `unspecified`          \-                        L605, L607, L606

  `thm:concrete-docking-kinetic-bundle-specialization`                                           `unspecified`          \-                        L627

  `thm:concrete-external-validation-not-credibly-dismissible`                                    `unspecified`          \-                        L811

  `thm:concrete-external-validation-threeway-bundle`                                             `unspecified`          \-                        L810

  `thm:concrete-generator-canonical-path-process-closure`                                        `unspecified`          \-                        L772

  `thm:concrete-langevin-interface-constructor`                                                  `unspecified`          \-                        L604

  `thm:concrete-qm-workflow-error-analysis-transport`                                            `unspecified`          \-                        L783

  `thm:conformational-ensemble-docking-transport`                                                `unspecified`          \-                        L610, L611, L612

  `thm:constructive-downstream-replacement-transport`                                            `unspecified`          \-                        L628, L629

  `thm:constructive-empirical-realism-shell-envelope`                                            `unspecified`          \-                        L683

  `thm:constructive-empirical-realism-transport`                                                 `unspecified`          \-                        L685, L684

  `thm:constructive-ito-wiener-derived-endpoints`                                                `unspecified`          \-                        L813

  `thm:constructive-only-core-field-consumers`                                                   `unspecified`          \-                        L658

  `thm:constructive-only-extended-field-consumers`                                               `unspecified`          \-                        L666

  `thm:constructive-only-spec-replacement`                                                       `unspecified`          \-                        L648

  `thm:constructive-scope-closure-of-extension-interfaces`                                       `unspecified`          \-                        L806

  `thm:constructive-stochastic-multipole-scope-discharge`                                        `unspecified`          \-                        L789

  `thm:contact-shell-allostery`                                                                  `unspecified`          \-                        L141

  `thm:continuous-state-measurable-encoding-transport`                                           `unspecified`          \-                        L716, L714, L713, L715, L726

  `thm:continuous-time-continuous-state-interface`                                               `unspecified`          \-                        L587, L581

  `thm:coordinate-extraction-poly`                                                               `unspecified`          \-                        L122

  `thm:coulomb-cutoff-invariance`                                                                `unspecified`          \-                        L73

  `thm:coulomb-cutoff-uniform-approx`                                                            `unspecified`          \-                        L74

  `thm:coulomb-tail-srank`                                                                       `unspecified`          \-                        L96

  `thm:counting-gap`                                                                             `unspecified`          \-                        BA10

  `thm:cramer-rao-nonidentifiable-irrelevant`                                                    `unspecified`          \-                        L412

  `thm:crooks-detailed-balance-equilibrium`                                                      `unspecified`          \-                        L483, L485, L486, L484

  `thm:decision-quotient-potential`                                                              `unspecified`          \-                        L139

  `thm:definitive-accept-flag-iff-deployment-accepted`                                           `unspecified`          \-                        L881

  `thm:definitive-adaptive-stop-sound`                                                           `unspecified`          \-                        L921

  `thm:definitive-batch-fusion-justified`                                                        `unspecified`          \-                        L923

  `thm:definitive-benchmark-certified-accepted-iff-decision-accepted`                            `unspecified`          \-                        L901

  `thm:definitive-benchmark-certified-rejected-iff-decision-rejected`                            `unspecified`          \-                        L902

  `thm:definitive-benchmark-decision-alias-exactness`                                            `unspecified`          \-                        L896

  `thm:definitive-benchmark-decision-refines-certificate-backend-benchmark`                      `unspecified`          \-                        L898

  `thm:definitive-benchmark-decision-rejected-iff-kernel-flag-false`                             `unspecified`          \-                        L900

  `thm:definitive-branch-bound-prune-sound`                                                      `unspecified`          \-                        L920

  `thm:definitive-campaign-pair-evals-closed-form`                                               `unspecified`          \-                        L929

  `thm:definitive-campaign-pair-evals-succ-recurrence`                                           `unspecified`          \-                        L930

  `thm:definitive-canonical-scorer-op-label-fusion-sound`                                        `unspecified`          \-                        L933

  `thm:definitive-concrete-checksum-byte-e2e`                                                    `unspecified`          \-                        L911

  `thm:definitive-concrete-checksum-verifier-exactness`                                          `unspecified`          \-                        L916

  `thm:definitive-constructive-benchmark-iff-kernel-flag`                                        `unspecified`          \-                        L891

  `thm:definitive-constructive-benchmark-refines-certificate-backend-benchmark`                  `unspecified`          \-                        L892

  `thm:definitive-constructive-deployment-refines-certificate-backend-deployment`                `unspecified`          \-                        L893

  `thm:definitive-crypto-verifier-sound`                                                         `unspecified`          \-                        L926

  `thm:definitive-decision-refines-certificate-backend-deployment`                               `unspecified`          \-                        L899

  `thm:definitive-deployment-certified-accepted-iff-decision-accepted`                           `unspecified`          \-                        L903

  `thm:definitive-deployment-certified-rejected-iff-decision-rejected`                           `unspecified`          \-                        L904

  `thm:definitive-deployment-decision-alias-exactness`                                           `unspecified`          \-                        L897

  `thm:definitive-exact-rat-artifact-accept-iff-benchmark-contract`                              `unspecified`          \-                        L894

  `thm:definitive-exact-rat-artifact-accept-refines-certificate-backend-accepts`                 `unspecified`          \-                        L895

  `thm:definitive-exact-rat-rejected-refines-certificate-backend-rejections`                     `unspecified`          \-                        L907

  `thm:definitive-interpreter-output-equals-runtime`                                             `unspecified`          \-                        L887

  `thm:definitive-pair-potential-fusion-justified`                                               `unspecified`          \-                        L932

  `thm:definitive-parse-cost-linear-time`                                                        `unspecified`          \-                        L924

  `thm:definitive-parse-encode-cost-exact`                                                       `unspecified`          \-                        L925

  `thm:definitive-pipeline-branch-bound-prune-sound`                                             `unspecified`          \-                        L922

  `thm:definitive-pipeline-campaign-pair-evals-closed-form`                                      `unspecified`          \-                        L931

  `thm:definitive-pipeline-total-ops-closed-form`                                                `unspecified`          \-                        L919

  `thm:definitive-rationalized-separation-flag-false`                                            `unspecified`          \-                        L914

  `thm:definitive-rationalized-separation-not-benchmark-contract`                                `unspecified`          \-                        L913

  `thm:definitive-raw-benchmark-accepted-iff-contract`                                           `unspecified`          \-                        L879

  `thm:definitive-raw-crossdock-accept-benchmark-iff`                                            `unspecified`          \-                        L874

  `thm:definitive-raw-crossdock-accept-deployment-iff`                                           `unspecified`          \-                        L875

  `thm:definitive-raw-crossdock-complete-lean-bundle`                                            `unspecified`          \-                        L889

  `thm:definitive-raw-crossdock-full-closure-bundle`                                             `unspecified`          \-                        L878

  `thm:definitive-raw-crossdock-totality`                                                        `unspecified`          \-                        L876

  `thm:definitive-raw-deployment-rejected-iff-not-contract`                                      `unspecified`          \-                        L880

  `thm:definitive-raw-runtime-flag-refines-accept`                                               `unspecified`          \-                        L877

  `thm:definitive-reject-flag-iff-deployment-rejected`                                           `unspecified`          \-                        L882

  `thm:definitive-report-runtime-accept-iff-deployment-accepted`                                 `unspecified`          \-                        L888

  `thm:definitive-report-runtime-reject-iff-deployment-rejected`                                 `unspecified`          \-                        L890

  `thm:definitive-runtime-ops-closed-form`                                                       `unspecified`          \-                        L917

  `thm:definitive-runtime-ops-succ-recurrence`                                                   `unspecified`          \-                        L918

  `thm:definitive-signed-artifact-byte-envelope-roundtrip`                                       `unspecified`          \-                        L910

  `thm:definitive-signed-exact-rat-accepted-iff-benchmark-contract`                              `unspecified`          \-                        L908

  `thm:definitive-signed-exact-rat-rejected-refines-certificate-backend-rejections`              `unspecified`          \-                        L909

  `thm:definitive-signed-pipeline-parser-bytes-exact`                                            `unspecified`          \-                        L928

  `thm:definitive-signed-rationalized-artifact-manifest-consistency`                             `unspecified`          \-                        L905

  `thm:definitive-signed-rationalized-concrete-byte-parse-verify`                                `unspecified`          \-                        L912

  `thm:definitive-signed-rationalized-crypto-byte-parse-verify`                                  `unspecified`          \-                        L927

  `thm:definitive-signed-rationalized-decision-accept-refines-certificate-backend-deployment`    `unspecified`          \-                        L906

  `thm:definitive-signed-rationalized-strict-rejection-refines-certificate-backend-rejections`   `unspecified`          \-                        L915

  `thm:deployment-contract-implies-benchmark-contract`                                           `unspecified`          \-                        L866

  `thm:derived-generator-pde-operator-closure`                                                   `unspecified`          \-                        L814

  `thm:descriptor-calibrated-mechanistic-ood-transfer`                                           `unspecified`          \-                        L787

  `thm:deterministic-kernel-scale-stationary`                                                    `unspecified`          \-                        L515, L514, L516

  `thm:discrete-acquisition`                                                                     `unspecified`          \-                        BA3

  `thm:dof-srank`                                                                                `unspecified`          \-                        L43

  `thm:downstream-campaign-win-bundle`                                                           `unspecified`          \-                        L804

  `thm:downstream-causal-quality-bundle`                                                         `unspecified`          \-                        L809

  `thm:electronic-reference-shell-error-transport`                                               `unspecified`          \-                        L724, L727, L723, L722

  `thm:electronic-structure-correction-transport`                                                `unspecified`          \-                        L720, L721, L719, L718, L717

  `thm:encoding-transport-rank-energy`                                                           `unspecified`          \-                        L489, L487, L488

  `thm:energy-entropy`                                                                           `unspecified`          \-                        IT3, EI1, L43

  `thm:energy-rank`                                                                              `unspecified`          \-                        BA7, BA6, L43

  `thm:england`                                                                                  `unspecified`          \-                        L45

  `thm:ensemble-multistep-population-transport`                                                  `unspecified`          \-                        L645

  `thm:ensemble-one-step-population-transport`                                                   `unspecified`          \-                        L634

  `thm:ensemble-statistical-validity-transport`                                                  `unspecified`          \-                        L656

  `thm:entropy-bound`                                                                            `unspecified`          \-                        IT3, L43

  `thm:equilibrium-kd-bound-from-partition-ratio-correction-chain`                               `unspecified`          \-                        L835

  `thm:equilibrium-kd-driving-energy-floor`                                                      `unspecified`          \-                        L826

  `thm:error-correction-srank-overhead`                                                          `unspecified`          \-                        L136

  `thm:estimator-minimax-derivation-bundle`                                                      `unspecified`          \-                        L800

  `thm:ewald-fourier-positive`                                                                   `unspecified`          \-                        L81

  `thm:ewald-long-range-certificates`                                                            `unspecified`          \-                        L677

  `thm:ewald-real-space-decay`                                                                   `unspecified`          \-                        L82

  `thm:ewald-tail-srank`                                                                         `unspecified`          \-                        L98

  `thm:exact-sufficiency-hardness-core`                                                          `unspecified`          \-                        L63

  `thm:explicit-hamiltonian-bath-elimination-langevin-endpoints`                                 `unspecified`          \-                        L770

  `thm:explicit-molecular-constant-drift-sde-endpoints`                                          `unspecified`          \-                        L771

  `thm:extended-physical-model-interface-scope-bundle`                                           `unspecified`          \-                        L801

  `thm:external-replication-at-scale-full-pipeline-bundle`                                       `unspecified`          \-                        L853

  `thm:external-validation-threeway-integration-bundle`                                          `unspecified`          \-                        L805

  `thm:fault-tolerant-landauer-floor`                                                            `unspecified`          \-                        L137

  `thm:feasible-collapse-factors`                                                                `unspecified`          \-                        L78

  `thm:finite-budget-no-collapse`                                                                `unspecified`          \-                        PH26

  `thm:finite-sample-complexity-inversion`                                                       `unspecified`          \-                        L746, L747

  `thm:finite-sample-complexity-inversion-count-order`                                           `unspecified`          \-                        L761, L762, L763

  `thm:finite-sample-square-count-inversion`                                                     `unspecified`          \-                        L767, L768, L769, L766

  `thm:finite-sample-upper-violation-implies-true-violation`                                     `unspecified`          \-                        L841

  `thm:finite-uniform-error-radius`                                                              `unspecified`          \-                        L92

  `thm:fisher-likelihood-indicator`                                                              `unspecified`          \-                        L408

  `thm:fisher-noisy-partial-channel-example`                                                     `unspecified`          \-                        L598, L597

  `thm:fisher-observation-channel-transport`                                                     `unspecified`          \-                        L583, L582

  `thm:fisher-rank-srank`                                                                        `unspecified`          \-                        L76

  `thm:fisher-sum-srank`                                                                         `unspecified`          \-                        L80

  `thm:five-way`                                                                                 `unspecified`          \-                        L43, L44, L47, L55

  `thm:fixed-contract-preregistered-benchmark-bundle`                                            `unspecified`          \-                        L807

  `thm:forcefield-derived-realistic-sde-endpoints`                                               `unspecified`          \-                        L792

  `thm:fullstate-biomolecular-forcefield-transport`                                              `unspecified`          \-                        L642, L641, L640

  `thm:fully-constructive-pipeline-deprecation-ready`                                            `unspecified`          \-                        L636

  `thm:generator-coefficients-canonical-regularity-closure`                                      `unspecified`          \-                        L782

  `thm:generator-pde-estimate-canonical-regularity-closure`                                      `unspecified`          \-                        L793

  `thm:geometric-constraint-decision-interface`                                                  `unspecified`          \-                        L579

  `thm:geometric-contact-allostery`                                                              `unspecified`          \-                        L140

  `thm:gibbs-conditioned-mechanism-dynamics`                                                     `unspecified`          \-                        L764

  `thm:grid-docking-entropy`                                                                     `unspecified`          \-                        L113

  `thm:grid-erase-irrelevant`                                                                    `unspecified`          \-                        L88

  `thm:hamiltonian-finite-difference-drift-derivation-endpoints`                                 `unspecified`          \-                        L780

  `thm:hamiltonian-mori-zwanzig-h-zero-limit-endpoints`                                          `unspecified`          \-                        L791

  `thm:hard-family-srank`                                                                        `unspecified`          \-                        L64

  `thm:hierarchical-chemical-realdata-separation-rate`                                           `unspecified`          \-                        L688

  `thm:hierarchical-chemical-realdata-separation-rate-constant-margin`                           `unspecified`          \-                        L705, L706

  `thm:hierarchical-chemical-realdata-separation-rate-constant-margin-all-datasets`              `unspecified`          \-                        L707

  `thm:hierarchical-chemical-realdata-separation-rate-margin`                                    `unspecified`          \-                        L696

  `thm:hierarchical-chemical-realdata-separation-rate-margin-all-datasets`                       `unspecified`          \-                        L703

  `thm:hierarchical-kinetic-replicate-inference-bundle`                                          `unspecified`          \-                        L689

  `thm:hierarchical-kinetic-replicate-inference-rate-constant-margins`                           `unspecified`          \-                        L708, L709

  `thm:hierarchical-kinetic-replicate-inference-rate-constant-margins-all-datasets`              `unspecified`          \-                        L710

  `thm:hierarchical-kinetic-replicate-inference-rate-margins`                                    `unspecified`          \-                        L697

  `thm:hierarchical-kinetic-replicate-inference-rate-margins-all-datasets`                       `unspecified`          \-                        L704

  `thm:hierarchical-required-size-joint-bundle`                                                  `unspecified`          \-                        L753

  `thm:hierarchical-required-size-model-dependent-constants`                                     `unspecified`          \-                        L779

  `thm:hierarchical-srank-bound`                                                                 `unspecified`          \-                        L134

  `thm:hopfield-ninio-kinetic-branch`                                                            `unspecified`          \-                        L435, L436, L434

  `thm:hopfield-ninio-proofreading-overhead`                                                     `unspecified`          \-                        L426, L427, L425

  `thm:independent-replication-outside-team-bundle`                                              `unspecified`          \-                        L803

  `thm:independent-replication-provenance-bundle`                                                `unspecified`          \-                        L808

  `thm:integrator-error-stack-unified-thermo-kinetic`                                            `unspecified`          \-                        L798

  `thm:inverse-design-atomistic-bridge`                                                          `unspecified`          \-                        L586

  `thm:inverse-rank-gap-design`                                                                  `unspecified`          \-                        L126, L127

  `thm:ito-wiener-filtration-langevin-endpoints`                                                 `unspecified`          \-                        L790

  `thm:jarzynski-from-crooks`                                                                    `unspecified`          \-                        L424

  `thm:joint-computable-pose-rmsd-probability-bundle`                                            `unspecified`          \-                        L864

  `thm:kd-interval-from-absolute-free-energy-model`                                              `unspecified`          \-                        L844

  `thm:kd-interval-from-driving-energy-error`                                                    `unspecified`          \-                        L843

  `thm:kernel-completion-categorical-endpoint`                                                   `unspecified`          \-                        L497, L496, L495, L499, L501, L498, L500

  `thm:kernel-power-stationarity-transport`                                                      `unspecified`          \-                        L521, L524, L522, L523, L518, L517, L519, L520

  `thm:kernel-quotient-universality-canonicity`                                                  `unspecified`          \-                        L490, L491, L493, L492, L494

  `thm:kinetic-concentration-identifiability-bridge`                                             `unspecified`          \-                        L665

  `thm:kinetic-confidence-transport`                                                             `unspecified`          \-                        L646, L647

  `thm:kinetic-observable-reporting-endpoints`                                                   `unspecified`          \-                        L616, L613, L615, L614

  `thm:kinetic-protocol-inference-guarantee`                                                     `unspecified`          \-                        L657

  `thm:kinetic-protocol-measurement-transport`                                                   `unspecified`          \-                        L635

  `thm:kinetic-replicate-identifiability-bundle`                                                 `unspecified`          \-                        L681

  `thm:langevin-analysis-closure-endpoint-bundle`                                                `unspecified`          \-                        L602, L603, L600, L601, L599

  `thm:langevin-boltzmann-stationarity-measure-derivation`                                       `unspecified`          \-                        L637

  `thm:langevin-canonical-continuous-path-process-closure`                                       `unspecified`          \-                        L732, L733

  `thm:langevin-canonical-pathprocess-of-finite-horizon-bridge`                                  `unspecified`          \-                        L755

  `thm:langevin-concrete-hamiltonian-bath-end-to-end`                                            `unspecified`          \-                        L754

  `thm:langevin-constructive-infinite-dimensional-path-derivation-bridge`                        `unspecified`          \-                        L686, L687

  `thm:langevin-continuous-state-measure-closure`                                                `unspecified`          \-                        L651, L650

  `thm:langevin-detailed-balance-grounding`                                                      `unspecified`          \-                        L591

  `thm:langevin-discretization-certified`                                                        `unspecified`          \-                        L592

  `thm:langevin-explicit-sde-conditions`                                                         `unspecified`          \-                        L639, L638

  `thm:langevin-finite-horizon-law-derivation-bridge`                                            `unspecified`          \-                        L695, L694

  `thm:langevin-finite-horizon-marginal-recovery`                                                `unspecified`          \-                        L702

  `thm:langevin-first-principles-discharge`                                                      `unspecified`          \-                        L630

  `thm:langevin-forcefield-derived-lipschitz-injection`                                          `unspecified`          \-                        L662, L667

  `thm:langevin-infinite-dimensional-path-measure-bridge`                                        `unspecified`          \-                        L679

  `thm:langevin-ito-fp-harris-closure`                                                           `unspecified`          \-                        L659

  `thm:langevin-measure-theoretic-endpoint-bundle`                                               `unspecified`          \-                        L812

  `thm:langevin-microscopic-derivation-constructor`                                              `unspecified`          \-                        L712, L725, L711

  `thm:langevin-molecular-hamiltonian-thermostat-constructor`                                    `unspecified`          \-                        L728, L731, L730, L729

  `thm:langevin-molecular-pathprocess-joint-bundle`                                              `unspecified`          \-                        L750

  `thm:langevin-to-mcmc-discretization-interface`                                                `unspecified`          \-                        L592, L591

  `thm:large-cutoff-bounded`                                                                     `unspecified`          \-                        L89

  `thm:learned-descriptor-ood-generalization-transfer`                                           `unspecified`          \-                        L799

  `thm:legacy-constructive-deprecation-bridge`                                                   `unspecified`          \-                        L618, L617

  `thm:lipschitz-grid-approx`                                                                    `unspecified`          \-                        L90

  `thm:lj-cutoff-invariance`                                                                     `unspecified`          \-                        L75

  `thm:lj-gradient`                                                                              `unspecified`          \-                        L83

  `thm:lj-shell-derivative-envelope`                                                             `unspecified`          \-                        L103

  `thm:lj-shell-gap-invariance`                                                                  `unspecified`          \-                        L100

  `thm:lj-shell-grad-stability`                                                                  `unspecified`          \-                        L102

  `thm:lj-shell-hessian-lipschitz`                                                               `unspecified`          \-                        L106

  `thm:lj-shell-quadratic-discretization`                                                        `unspecified`          \-                        L107

  `thm:lj-shell-quadratic-remainder`                                                             `unspecified`          \-                        L108

  `thm:lj-shell-second-derivative-envelope`                                                      `unspecified`          \-                        L105

  `thm:lj-shell-uniform-approx`                                                                  `unspecified`          \-                        L101

  `thm:lj-tail-srank`                                                                            `unspecified`          \-                        L97

  `thm:lj-tolerance-collapse-explicit`                                                           `unspecified`          \-                        L578, L577

  `thm:locked-prospective-falsification-run-soundness`                                           `unspecified`          \-                        L849

  `thm:md-above-ground`                                                                          `unspecified`          \-                        L109

  `thm:md-binary-summary-rank-monotonicity`                                                      `unspecified`          \-                        L825

  `thm:md-class-crooks-calibration-interface`                                                    `unspecified`          \-                        L580

  `thm:md-concrete-physics-witness-discharge-bundle`                                             `unspecified`          \-                        L833

  `thm:md-detailed-balance-equilibrium-pathratio`                                                `unspecified`          \-                        L827

  `thm:md-energy-entropy`                                                                        `unspecified`          \-                        L114

  `thm:md-energy-rank`                                                                           `unspecified`          \-                        L111

  `thm:md-equilibrium-kd-prediction-from-rank-lb`                                                `unspecified`          \-                        L829

  `thm:md-exact-lj-physics-witness-bundle`                                                       `unspecified`          \-                        L832

  `thm:md-falsification-not-falsified-iff`                                                       `unspecified`          \-                        L839

  `thm:md-high-confidence-fail-condition-bundle`                                                 `unspecified`          \-                        L842

  `thm:md-independent-certificate-risky-prediction-bundle`                                       `unspecified`          \-                        L838

  `thm:md-independent-rank-risky-prediction-bundle`                                              `unspecified`          \-                        L831

  `thm:md-independent-srank-interval-certificates`                                               `unspecified`          \-                        L837

  `thm:md-native-coordinate-rank-identity`                                                       `unspecified`          \-                        L823

  `thm:md-necessary-contact-shell-budget-from-rank-lb`                                           `unspecified`          \-                        L830

  `thm:md-per-case-real-artifact-discharge-all-targets`                                          `unspecified`          \-                        L846

  `thm:md-physical-3n-minus-k-budget`                                                            `unspecified`          \-                        L824

  `thm:md-physical-utility-interface`                                                            `unspecified`          \-                        L822

  `thm:md-preregistered-rank-protocol-soundness`                                                 `unspecified`          \-                        L840

  `thm:md-rank-kd-bound-from-partition-chain`                                                    `unspecified`          \-                        L836

  `thm:md-resolver-free-equilibrium-bundle`                                                      `unspecified`          \-                        L828

  `thm:measurable-transition-constructive-truncation`                                            `unspecified`          \-                        L588

  `thm:measurable-transition-finite-horizon-canonical-endpoint`                                  `unspecified`          \-                        L566, L565, L567, L569, L568

  `thm:measurable-transition-jarzynski-concrete-endpoint`                                        `unspecified`          \-                        L542

  `thm:measurable-transition-kernel-concrete-endpoint`                                           `unspecified`          \-                        L535, L536, L530, L532, L531

  `thm:measurable-transition-path-measure-jarzynski-concrete-endpoint`                           `unspecified`          \-                        L549

  `thm:measurable-transition-projective-canonical-instance`                                      `unspecified`          \-                        L576

  `thm:measurable-transition-projective-kolmogorov-endpoint`                                     `unspecified`          \-                        L570, L571, L572, L574, L575, L573

  `thm:measure-kernel-db-stationary-transport`                                                   `unspecified`          \-                        L510, L512, L511, L508, L509, L513

  `thm:measure-kernel-quotient-rg-endpoint`                                                      `unspecified`          \-                        L557, L556, L552, L553, L554, L550, L551, L555

  `thm:measure-kernel-transport-quotient-instance-recovery`                                      `unspecified`          \-                        L559, L558

  `thm:mechanistic-ood-uniform-transfer-bound`                                                   `unspecified`          \-                        L777

  `thm:mechanochemical-coupling-gap`                                                             `unspecified`          \-                        L133

  `thm:microscopic-extension-interface-scope-bundle`                                             `unspecified`          \-                        L815

  `thm:min-bit-operations`                                                                       `unspecified`          \-                        BA5, BA6, L43

  `thm:mixing-autocorrelation-trajectory-correction-total-error`                                 `unspecified`          \-                        L785

  `thm:model-dependent-minimax-optimality-bundle`                                                `unspecified`          \-                        L788

  `thm:model-dependent-rate-term-target-margin`                                                  `unspecified`          \-                        L778

  `thm:molecular-docking-srank-bound`                                                            `unspecified`          \-                        L66

  `thm:multiplicative-separable-empty-sufficient`                                                `unspecified`          \-                        L84

  `thm:nontrivial-biomolecular-forcefield-transport`                                             `unspecified`          \-                        L632, L631

  `thm:numerical-stack-derived-simulator-control-flags`                                          `unspecified`          \-                        L816

  `thm:numopt-bound`                                                                             `unspecified`          \-                        IT4, L43

  `thm:one-transition-one-bit`                                                                   `unspecified`          \-                        BA4

  `thm:ood-transfer-calibration-derived`                                                         `unspecified`          \-                        L760

  `thm:optimizer-class-richness-nonbinary-vc`                                                    `unspecified`          \-                        L584, L585

  `thm:optimizer-class-richness-rank-lower-bound`                                                `unspecified`          \-                        L418, L417

  `thm:pairwise-geometric-derived-transport`                                                     `unspecified`          \-                        L661, L660

  `thm:pairwise-geometric-forcefield-transport`                                                  `unspecified`          \-                        L653, L652

  `thm:paper4-interface-discharge-extensions`                                                    `unspecified`          \-                        L670, L668, L669

  `thm:paper4-stochastic-relevance-conjecture-full-support`                                      `unspecified`          \-                        L671

  `thm:paper4-stochastic-relevance-conjecture-nonneg-explicit-step-dynamics`                     `unspecified`          \-                        L690

  `thm:paper4-stochastic-relevance-conjecture-nonneg-primitive-dynamics`                         `unspecified`          \-                        L682

  `thm:paper4-stochastic-relevance-conjecture-nonneg-support-transport`                          `unspecified`          \-                        L674

  `thm:paper4-stochastic-relevance-general-distribution-progress`                                `unspecified`          \-                        L672, L673

  `thm:paper4-stochastic-relevance-support-transport-of-explicit-step-dynamics`                  `unspecified`          \-                        L699, L698

  `thm:paper4-witness-chain-import`                                                              `unspecified`          \-                        L649

  `thm:parametric-fisher-identifiable-dimension`                                                 `unspecified`          \-                        L410, L411, L409

  `thm:partition-correction-numerical-kd-upper-bound-bundle`                                     `unspecified`          \-                        L847

  `thm:partition-ratio-driving-floor-with-correction-margin`                                     `unspecified`          \-                        L834

  `thm:path-measure-jarzynski-integral-transport`                                                `unspecified`          \-                        L543, L545, L544, L546, L548, L547

  `thm:path-process-transport-endpoint`                                                          `unspecified`          \-                        L560, L562, L564, L563

  `thm:path-space-crooks-kernel-power-transport`                                                 `unspecified`          \-                        L526, L525, L533, L527, L529, L534, L528

  `thm:path-space-jarzynski-kernel-power-transport`                                              `unspecified`          \-                        L538, L537, L539, L541, L540

  `thm:pathwise-energy-lower-bound`                                                              `unspecified`          \-                        L123

  `thm:potential-landscape-barrier-kinetics-bundle`                                              `unspecified`          \-                        L795

  `thm:preregistered-prospective-beats-strong-baselines`                                         `unspecified`          \-                        L802

  `thm:production-independent-srank-extractor-bundle`                                            `unspecified`          \-                        L848

  `thm:prospective-empirical-closure`                                                            `unspecified`          \-                        L749, L748

  `thm:qm-grounded-electronic-halfgap-transport`                                                 `unspecified`          \-                        L751

  `thm:qm-grounded-electronic-structure-transport`                                               `unspecified`          \-                        L735, L734, L737, L736

  `thm:qm-method-specific-electronic-transport`                                                  `unspecified`          \-                        L756

  `thm:qm-protocol-derived-method-calibration-transport`                                         `unspecified`          \-                        L765

  `thm:qm-workflow-specific-realism-transport-bundle`                                            `unspecified`          \-                        L773

  `thm:qm-workflow-transport-benchmark-summary`                                                  `unspecified`          \-                        L794

  `thm:quantified-simulator-thermo-kinetic-bundle`                                               `unspecified`          \-                        L776

  `thm:quantitative-admissibility-rank-collapse`                                                 `unspecified`          \-                        L125, L124

  `thm:quotient-resolution-speed-bound`                                                          `unspecified`          \-                        L130

  `thm:quotient-trajectory-crooks`                                                               `unspecified`          \-                        L128

  `thm:quotient-trajectory-dissipation`                                                          `unspecified`          \-                        L129

  `thm:rank-calibrated-crooks-standard`                                                          `unspecified`          \-                        L423

  `thm:rank-identification`                                                                      `unspecified`          \-                        L43, L46, L51

  `thm:rank-one-ground`                                                                          `unspecified`          \-                        BA8, L54, L55

  `thm:raw-pocket-ligand-benchmark-solver-bundle`                                                `unspecified`          \-                        L868

  `thm:raw-pocket-ligand-constructor-posterior-bundle`                                           `unspecified`          \-                        L865

  `thm:realism-augmented-forcefield-transport`                                                   `unspecified`          \-                        L676, L675

  `thm:realistic-molecular-finite-difference-sde-endpoints`                                      `unspecified`          \-                        L781

  `thm:reference-biomolecular-zero-shell-calibration`                                            `unspecified`          \-                        L621, L622, L620

  `thm:renormalized-admissibility-equivalence`                                                   `unspecified`          \-                        L135

  `thm:resolution-controlled-gap-invariance`                                                     `unspecified`          \-                        L99

  `thm:resolution-controlled-uniform`                                                            `unspecified`          \-                        L91

  `thm:resolution-sufficient`                                                                    `unspecified`          \-                        BA5

  `thm:reversible-chemical-transition-detailed-balance-bundle`                                   `unspecified`          \-                        L774

  `thm:rmsd-probability-derived-pose-solver-bundle`                                              `unspecified`          \-                        L863

  `thm:rmsd-success-probability-unit-interval`                                                   `unspecified`          \-                        L861

  `thm:sampled-docking-gap`                                                                      `unspecified`          \-                        L67

  `thm:sampled-inside-cutoff-budget-energy`                                                      `unspecified`          \-                        L112

  `thm:sampled-inside-cutoff-sufficient`                                                         `unspecified`          \-                        L68

  `thm:single-state-partition-positive-bundle`                                                   `unspecified`          \-                        L854

  `thm:single-target-full-physical-closure-instance-bundle`                                      `unspecified`          \-                        L852

  `thm:single-target-zero-rank-concrete-closure-bundle`                                          `unspecified`          \-                        L857

  `thm:spectral-gap-absolute-free-energy-total-error`                                            `unspecified`          \-                        L797

  `thm:spectral-gap-trajectory-concentration-bundle`                                             `unspecified`          \-                        L796

  `thm:spinhalf-canonicaldp-instance`                                                            `unspecified`          \-                        L594, L593

  `thm:spinhalf-concrete-quantum-instantiation`                                                  `unspecified`          \-                        L594, L595, L593

  `thm:spinhalf-decoherence-floor`                                                               `unspecified`          \-                        L595

  `thm:store-backed-attested-concrete-external-validation-not-credibly-dismissible`              `unspecified`          \-                        L821

  `thm:store-backed-attested-concrete-external-validation-threeway-bundle`                       `unspecified`          \-                        L820

  `thm:strict-dominance-empty-sufficient`                                                        `unspecified`          \-                        L85

  `thm:target-class-chemistry-completeness-kd-interval-bundle`                                   `unspecified`          \-                        L851

  `thm:thermodynamic-selection`                                                                  `unspecified`          \-                        BA8, L49, L54, L55

  `thm:time-lower-bound`                                                                         `unspecified`          \-                        BA1, BA2, BA5, BA6, L43

  `thm:top-level-computable-export`                                                              `unspecified`          \-                        L596

  `thm:top-level-computable-path`                                                                `unspecified`          \-                        L596

  `thm:topk-ambiguity-band`                                                                      `unspecified`          \-                        L72

  `thm:topk-boundary-gap`                                                                        `unspecified`          \-                        L79

  `thm:topk-certificate-sound`                                                                   `unspecified`          \-                        L94

  `thm:topk-mass-lower-bound-rmsd-success-probability`                                           `unspecified`          \-                        L862

  `thm:tractable-rank-one`                                                                       `unspecified`          \-                        L47, L53, L56

  `thm:trajectory-absolute-free-energy-correction-closure`                                       `unspecified`          \-                        L775

  `thm:trajectory-time-energy-tradeoff`                                                          `unspecified`          \-                        L131

  `thm:unified-chemical-state-dynamics-transport`                                                `unspecified`          \-                        L739, L738

  `thm:unified-langevin-assumption-bundle-discharge`                                             `unspecified`          \-                        L619

  `thm:unified-physical-ood-prospective-bundle`                                                  `unspecified`          \-                        L752

  `thm:unified-simulator-error-analysis-controlled-thermo-kinetic`                               `unspecified`          \-                        L786

  `thm:unified-simulator-kinetic-derivation`                                                     `unspecified`          \-                        L759

  `thm:unified-thermo-kinetic-physical-model`                                                    `unspecified`          \-                        L743

  `thm:universality-ood-calibration-bounds`                                                      `unspecified`          \-                        L744, L745

  `thm:upper-only-independent-srank-extractor-bundle`                                            `unspecified`          \-                        L856

  `thm:velocity-verlet-volume`                                                                   `unspecified`          \-                        L86

  `thm:zero-correction-calibration-bundle`                                                       `unspecified`          \-                        L855
  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

*Auto summary: indexed 453 claims by hardness profile (unspecified=453).*


# Scope Statements {#appendix-assumptions}

This appendix lists the principal scope statements for the finite decision-thermodynamic framework:

-   **Canonical encoding:** The main theorems are exact for the canonical binary decision problem attached to the bounded decision system, and the canonical object is characterized by an initiality theorem in the exact finite-resolution category over the canonical state space. A noncanonical transport theorem is now proved for optimizer-preserving encoding equivalences that also preserve/reflect coordinate-agreement relations; the same transport layer now has a continuous-state measurable-encoding bridge package (with partition-code compatibility) that transports coordinate relevance, structural rank, and Landauer floors, and exports those transports as one bundled theorem endpoint. An explicit extension-interface witness bundle now captures richer physical encoding transport obligations as theorem-level witnesses.

-   **Kernel-completion categorical endpoint:** The proved endpoint theorem is for operation-kernel schemas (operation data plus kernel congruence), with object-level equivalence to bare operation systems, hom-level full-faithful embedding, and universal no-collapse canonicity/factorization package at that level. The AE-germ quotient and an operation-preserving renormalization-flow quotient-invariance law are now included as explicit instances.

-   **Measure-kernel dynamics layer:** The theorem package now includes abstract semigroup transport for detailed-balance/stationarity claims, kernel-power and quotient-calculus transport laws, path-space Crooks/Jarzynski lifts, explicit path-measure and process-level transport endpoints, canonical finite-horizon recursion, projective/Kolmogorov interfaces, a canonical truncation-witness instantiation, and concrete Giry-bind measurable-kernel instantiation results. The canonical finite-horizon truncation measurability step is now proved constructively in Lean (no truncation axiom). A continuous-time/continuous-state interface endpoint is also included at theorem level (continuous-time kernel semigroup discretization to additive scale flows, plus measurable two-time path projections). The finite-horizon constructive bridge additionally exports direct projective-marginal and Kolmogorov-extension marginal recovery formulas, and the first-principles layer now includes both a microscopic-derivation constructor and a concrete molecular-Hamiltonian+thermostat constructor that map microscopic drift inequalities directly into the first-principles Langevin interface, together with a direct microscopic dissipativity-to-ergodicity export theorem, a canonical continuous path-process closure theorem that packages process regularity/uniqueness certificates with constructive finite-horizon closure data, and an explicit joint microscopic+canonical-path closure bundle theorem. The newest closure layer now also provides a constructive stochastic-analysis discharge bundle that instantiates canonical process regularity directly from finite-horizon bridge data.

-   **Structural chain:** Counting Gap fixes the finite-event statement. Bounded Acquisition fixes the traversal-rate statement. Discrete Acquisition and One Transition, One Bit fix the acquisition-event interface.

-   **Landauer calibration:** Thermodynamic cost is calibrated by a per-bit Landauer floor. Stronger substrate-dependent lower bounds may exist, but they are additional assumptions, not part of the theorem package under discussion.

-   **Quantum substrate instantiation layer:** The abstract quantum readout/decoherence theorem now has a concrete spin-$\tfrac12$ instantiation endpoint: explicit two-state readout witness, exact canonical rank-one embedding, and exact $k_B T\ln 2$ decoherence event identity. A theorem-level extension witness bundle now includes open-system channel modeling obligations explicitly, reducing richer channel extensions to witness discharge.

-   **Exact-resolution setting:** The results concern exact sufficiency and exact-resolution cost. Approximate, stochastic, or bounded-confidence regimes require separate analysis.

-   **Composed force-field layer:** The theorem package now includes an explicit composed classical force-field interface (bonded + Lennard-Jones + Coulomb) with architecture/decision-problem projections, shellwise Lipschitz composition, and concrete biomolecular calibration bundles for zero-shell, anchor-nontrivial, calibrated full-state interaction, and pairwise-geometric families with explicit-geometry-derived shell constants (all with half-gap transport endpoints). A realism-augmentation layer now additionally supports explicit solvent, polarization, many-body, and long-range correction terms with summed-shell Lipschitz transport and half-gap endpoint export, and a constructive empirical instantiation layer now anchors these shells to dataset-level mean/std-error calibrations with explicit $z/\sqrt{N}$ finite-sample shell envelopes. The same layer now also includes a concrete reference-realism calibration package with explicit shell constants, explicit aggregate-shell value export, and direct constructive transport theorems, plus an electronic-structure correction layer (charge transfer + metal coordination) with explicit shell/error envelopes and corresponding Lipschitz/error/half-gap transport endpoints, including a reference specialization where the electronic shell vanishes and half-gap transport reduces to the realism-only shell, a QM-grounded chemistry-class envelope layer providing tighter shell/error constants for designated active chemistry classes, and a QM-tight-shell half-gap transport endpoint. A higher-order multipole closure theorem now exports explicit multipole shell-Lipschitz constants under constructive witness discharge.

-   **Finite state family:** The entropy and replication theorems are finite counting results. Continuum models must first be reduced to a finite decision quotient before these arguments apply.

-   **Fisher observation model:** The canonical explicit optimal-action observation channel remains the fully constructed model, and the same section now includes a theorem-level general observation-channel Fisher interface endpoint: any channel with declared coordinatewise Fisher-to-relevance identification inherits the structural-rank Fisher sum identity and diagonal relevance recovery. The extension-interface witness bundle now packages noisy/partial observation-channel obligations explicitly, keeping concrete experimental-channel instantiation at witness-discharge level.

-   **Admissibility canonicality layer:** The fixed-erasure-count uniqueness theorem is stated in the rank-normal-form collapse category (objects are collapsed ranks, morphisms are rank-monotone maps). Lifting this uniqueness to richer state- or action-level isomorphism notions requires additional structure.

-   **Collapsed-rank categorical layer:** The proved category structure is the preorder/rank-normal-form layer (identity/composition, initial object, min/max product-coproduct, additive tensor, monotone rank functor). In the unbounded setting, no terminal object exists; in the bounded-rank slice, terminal objects, finite limits/colimits, finite-family meet/join objects, arbitrary-family meet/join objects (complete-lattice form), and monotonicity/idempotence/nonempty-absorption corollaries are recovered via the bounded index-order encoding, together with binary meet/join rewrite identities (commutativity, associativity, idempotence, absorption). No topos-level claim (e.g., subobject classifier) is asserted here.

-   **Combinatorial lower-bound layer:** The binary optimizer-richness floor remains fully proved, and a nonbinary/VC transfer endpoint is now included: under a declared $q$-ary class envelope and growth witness, structural rank inherits the same lower bound. Concrete nonbinary model instantiations therefore reduce to proving the declared $q$-ary class envelope.

-   **Proofreading specificity layer:** The Hopfield--Ninio reduction is stated for the explicit exponential specificity model and a declared overhead-energy witness under Landauer calibration, with a kinetic branch-rate specialization requiring positive equilibrium correct/error rates and an explicit branch-ratio calibration identity; richer kinetic-network assumptions must still be bridged into that witness interface.

-   **Fluctuation-relation layer:** The Crooks/Jarzynski reduction is stated for finite strictly positive trajectory distributions with explicit calibration identities, includes the detailed-balance equilibrium calibration corollary, and includes an explicit nonequilibrium molecular-dynamics class interface theorem that derives Crooks standard form from declared stepwise rank calibration plus cumulative work calibration. A continuous overdamped-Langevin-to-discrete-MCMC bridge endpoint is now added at theorem level (Boltzmann detailed balance plus Euler--Maruyama discretization error witness), together with finite-state measure-theoretic Boltzmann stationarity derivation from detailed balance + row-stochasticity, continuous-state measure-kernel closure, an explicit Ito/Fokker--Planck/Harris bridge layer, and an infinite-dimensional path-measure layer that now has both certificate-export and constructive finite-horizon/Kolmogorov-extension derivation endpoints.

-   **Physics-assumption interfaces:** Thermodynamic/kinetic premises used by these bridges are represented as explicit proposition interfaces carried as theorem hypotheses, and the current build now includes concrete witness/proof instantiations for multiple interfaces (EP1/EP2/EP3 locality witnesses, reversible-chain entropy-production nonnegativity witness, FDT-to-stationarity discharge theorem, differentiability-to-Born--Oppenheimer discharge theorem, and large-radius Lennard-Jones lattice-tail convergence witnesses), plus bundled paper4-to-paper3 witness-import theorems. The discharge layer now also exports direct certificate-to-interface conversion for TUR inequalities and velocity-Verlet shadow-Hamiltonian cubic-drift bounds, resolves the exploratory stochastic-preservation relevance conjecture on Boolean product spaces under full-support hypotheses, and adds unrestricted-distribution closure theorems under support-transport, primitive finite-time dynamics, and explicit one-step dynamics derivations, including an explicit-step-to-support-transport assumption-reduction path. The active paper3/paper4 proof sources no longer rely on custom global Lean `axiom` declarations for these premises.

-   **Speed-accuracy/on-rate layer:** The admissibility-indexed speed law is stated for exact-resolution witnesses in bounded regions with positive signal speed. The on-rate envelope additionally assumes positive time horizon and positive relaxed structural rank so the reciprocal rate expression is well-defined. A protocol-noise theorem layer now adds explicit finite-sample absolute-error budgets and margin conditions under which threshold/ranking inferences for kinetic observables are guaranteed stable, plus concentration/identifiability wrapper layers that export those same conclusions together with statistical certificate predicates; a replicate-aware identifiability extension further incorporates systematic-bias-aware pathway margins and replicate-count certificates, and a hierarchical multi-dataset extension adds pooled-replicate finite-sample-rate metadata with shared-bias margin transport and explicit chemical/kinetic rate-margin theorem endpoints, including dataset-uniform rate-margin transport theorems, external rate-constant upper-bound reduction theorems, required sample/replicate complexity-inversion transport theorems, and a joint chemical+kinetic required-size bundle theorem. The same section now also includes a unified thermodynamic/kinetic single-model theorem bundle, a universality/OOD uncertainty-calibration layer, and a unified physical/OOD/prospective empirical closure bundle theorem. The zero-collapse speed specialization requires the additional backward factorization witness at the same tolerance increment.

-   **Chemical-state and empirical closure layer:** Chemical microstate effects are now represented both as static coupled utility contributions and as unified finite-state dynamic layers carrying joint protonation/tautomer/ionic/solvent/water-bridge metadata with pH/ionic window certificates and stationary expected-utility transport. The same build also includes an absolute binding free-energy correction stack (standard-state, finite-size, long-range, net-charge, restraint) with summed correction-bound transport and a prospective blinded-benchmark empirical-closure layer that exports calibration/predictive coverage and failure-rate-bound certificates.

-   **Allosteric distance-decay layer:** The shell-sum theorem is finite and profile-driven; the exponential specialization assumes nonnegative envelope scale and decay rate and a shell-cardinality witness bounded by the ceiling of that exponential envelope, while the polynomial specialization assumes nonnegative envelope scale, natural exponent profile, and the corresponding ceiling-bounded shell witness. The polynomial series-budget corollary additionally requires an explicit finite bound on the partial reciprocal-power sum used in the closed-form real inequality.

-   **Replication theorem:** The finite replication entropy gap is a theorem of the calibrated exact-resolution model. England's 2013 theorem belongs to a stochastic-thermodynamic path-space model.

-   **Finite-budget theorem:** Finite-Budget No-Collapse is a theorem about bounded budgets, positive per-bit cost, and exponential lower-bound growth.

-   **Top-level executability layer:** The development now includes a constructive rational-certificate top-level path from molecular input to ArrayDSL JSON export with theorem-level computability statement, plus a normalization/deprecation bridge theorem family. Automatic wrapper theorems now derive legacy/constructive alignment witnesses directly from constructive outputs, yield a fully constructive deprecation-readiness theorem for the top-level pipeline, provide a proposition-level equivalence principle for constructive-only normalized outputs, and include direct migration lemmas for both core and extended downstream consumers (membership/cardinality/refinement-payload/export predicates). The legacy noncomputable path remains available only for compatibility with older statement forms.

-   **Latest physical-completeness discharge layer:** The current build now also includes theorem-level closures for (i) concrete Hamiltonian+bath to end-to-end Langevin endpoints, (ii) canonical path-process construction directly from finite-horizon bridge data, (iii) method-specific affine QM calibration transport to tight electronic shell/error bounds, (iv) conditioned-mechanism-derived unified chemical dynamics with stationary fixed-point identity, (v) protocol-derived absolute free-energy correction closure, (vi) simulator-derived thermo/kinetic bundling from shared model measurements, (vii) transfer-calibration-derived OOD uncertainty bounds, and (viii) count-order-derived finite-sample inversion monotonicity discharge for hierarchical chemical/kinetic required-size theorems. The newest extension layer further adds protocol-derived active-class QM calibration transport, a Gibbs-factor-derived chemical mechanism closure with explicit stationary law formula, square-count-complexity-derived finite-sample inversion transport, explicit Hamiltonian-elimination microscopic drift export, explicit constant-drift molecular SDE endpoint constants, constructive generator-residual-to-canonical-path closure, workflow-specific QM decomposition transport, reversible detailed-balance chemical dynamics bundling, trajectory-estimator absolute free-energy correction closure, quantified simulator-control thermo/kinetic bundling, mechanistic OOD envelope transfer bounds, and model-dependent finite-sample inversion constants with joint hierarchical required-size discharge. The latest pass additionally upgrades these with finite-difference Hamiltonian drift derivation, non-constant realistic molecular SDE closure, coefficient-derived canonical regularity construction, barrier-crossing Kramers/Eyring reversible kinetics, descriptor-calibrated mechanistic OOD transfer with prospective protocol witness, minimax-optimality finite-sample transport, and a bundled constructive stochastic-analysis plus multipole scope-gap discharge theorem. The current extension now further contributes explicit Ito-Wiener-filtration semantics, Hamiltonian $h\to0$ Mori--Zwanzig limit closure, force-field-derived SDE constants, benchmark-derived QM workflow transport, potential-landscape prefactor-derived barrier kinetics, spectral-gap concentration-to-trajectory correction transport, integrator-stack-derived simulator controls, learned-descriptor prospective OOD generalization transfer, estimator-backed minimax derivation closure, and explicit scope-extension witness bundling.

-   **External validation evidence layer:** The theorem package now includes an explicit three-way external-validation integration layer: (i) pre-registered prospective benchmark superiority over declared strong baselines on calibration/ranking/kinetics-free-energy consistency, (ii) independent outside-team replication with matched protocol and reproduced superiority clauses, and (iii) strict downstream campaign improvements (hit identification, triage quality, campaign efficiency). A formal core-gap dismissal predicate is defined, and the integrated bundle theorem proves its negation once the primary (1)+(2) evidence core is instantiated. The latest pass additionally introduces fixed-contract benchmark locking (metrics/datasets/splits/blind rule/baseline roster), signed artifact provenance records, separate-team/separate-compute independent-replication provenance closure, randomized/blinded/balance-controlled downstream causal-quality closure, and one concrete artifact instantiation theorem for the full three-way package.

-   **Pass-29 measure-theoretic derivation layer:** The current pass adds a measure-theoretic Langevin endpoint bridge (probability path-law semantics with measure-level stationarity/ergodicity transport), a constructive Ito/Wiener derivation layer where adaptedness/integrability/Ito/martingale clauses are extracted from explicit objects, an explicit-profile PDE/operator derivation layer feeding canonical regularity closure, a microscopic open-system/noisy-observation scope-derivation layer, and a numerical-stack theorem identifying top-level simulator control flags with explicit mismatch-vs-tolerance inequalities.

-   **Pass-29 attested external-evidence layer:** External validation is further strengthened with verifier-checked signatures, immutable-manifest ingestion records, timestamp-based preregistration locking, digest-based contract/result binding, typed team/compute provenance backed by signed identity/compute/protocol/execution artifacts, and measured randomized/blinded/balance diagnostics for downstream causal isolation, while preserving the no-credible-dismissal core-gap conclusion at concrete attested instantiation level.

-   **Pass-30 immutable-store ingestion strengthening:** The attested external-evidence stack is now additionally tied to an explicit immutable artifact store model (URI fetch + digest function), with theorem-level witnesses that each concrete attested artifact manifest is store-backed by payload/digest consistency and that the full attested three-way external-validation no-credible-dismissal conclusion persists under this store-backed ingestion layer.

-   **Pass-31 reviewer-response closure layer:** The theorem package now explicitly states that exact docking decision utility is inherited directly from the molecular binding problem object, identifies docking structural rank with native-coordinate relevance count, proves a physical $3(N-k)$ rank budget under cutoff-locality-relevant atom counts, proves binary summary rank monotonicity relative to full pose-selection docking, adds a resolver-free equilibrium bridge combining docking free-energy floor with detailed-balance path-ratio unity, and adds falsifiable independent-rank prediction theorems linking independently certified rank lower bounds to both equilibrium affinity upper bounds and necessary one-hop contact-shell geometry budgets.

-   **Pass-32 physical-completeness closure layer:** The theorem package now additionally includes (i) per-system concrete-physics witness discharge bundles that derive strict-optimality, cutoff-bounded perturbation behavior, and both excluded-atom/contact-shell docking rank budgets from explicit force-field/cutoff/geometry certificates; (ii) an end-to-end measurement-free equilibrium chain from partition-ratio free-energy definitions plus explicit standard-state/finite-size/long-range/net-charge/restraint correction budgets to the exact $\Delta G_{\mathrm{drive}}$ witness used in rank-based affinity bounds; (iii) independent structural-rank interval extraction theorems from relevance/sufficiency certificates with no affinity fit input; (iv) pre-registered hard-threshold falsification protocol logic with explicit fail conditions; (v) finite-sample/high-confidence margin-to-true-violation transport theorems for assay-noise handling; and (vi) chemistry-condition data-backed uncertainty propagation endpoints from calibration windows/posterior constants through free-energy correction error budgets to final equilibrium $K_d$ intervals.

-   **Pass-33 autonomous execution closure layer:** The theorem package now adds an execution-level closure stack that formalizes: (i) per-case real Hamiltonian/force-field/geometry artifact discharge into concrete docking witnesses for all declared targets, (ii) certified numerical partition-function and correction-term closure feeding directly into equilibrium $K_d$ bounds, (iii) production independent-srank extractor outputs with explicit structure/mechanics-only and no-affinity-input certificates, (iv) locked prospective falsification-run records with split/metric/blind locking and fail-condition-to-falsification soundness, (v) assay-noise calibration models decomposed by replicate/batch/instrument error with both fail-call and pass-call high-confidence validity transport, (vi) target-class chemistry completeness for protonation/tautomer/ionic/solvent/water/electronic calibration uncertainty propagated to final $K_d$ intervals, and (vii) outside-team/outside-compute replication-at-scale packaging that exports the full target-level closure proposition pointwise across new targets.

-   **Pass-34 instantiation-constructor closure layer:** The theorem package now additionally provides operational constructor endpoints that make the pass-33 objects directly instantiable in canonical low-assumption form: (i) one-state exact partition constructors with certified positivity, (ii) canonical zero-correction one-sample calibration closure, (iii) upper-only independent-srank extractor templates from supplied sufficiency certificates (empty lower set + explicit no-affinity provenance), (iv) canonical zero-rank single-target full-closure instance construction from real witness package plus upper sufficiency certificate, and (v) attested-provenance external-replication constructors including a singleton-target concrete attested replication bundle.

-   **Pass-35 computable-pose and RMSD-probability closure layer:** The theorem package now additionally formalizes (i) an explicit finite-enumeration pose-search algorithm endpoint (list-argmax over covered action families) with exact optimizer-membership guarantee, (ii) finite RMSD-success probability semantics from posterior mass over threshold-satisfying poses with unit-interval calibration bounds, (iii) monotone transport from paper4 finite top-$k$ posterior mass to RMSD-success probability under RMSD-coverage certificates, and (iv) a bundled RMSD-probability-derived pose-solver endpoint combining selected-pose threshold validity with certified lower bounds on success probability.

-   **Pass-36 definitive raw-endpoint closure layer:** The theorem package now additionally closes an end-to-end raw pocket+ligand sampled-docking endpoint in Lean by formalizing: (i) canonical raw-input sampled-family construction (grid-lifted candidate family plus base-action anchor), (ii) posterior synthesis and normalization certificates from that raw constructor, (iii) benchmark-vs-deployment contract split with conservative proxy-RMSD calibration and deployment-to-benchmark implication, (iv) canonical ProgramIR execution witnesses refining runtime accept/reject flags to accepted/failure solver certificates, and (v) canonical raw benchmark/deployment solver wrappers with acceptance equivalences, totality theorems, and a bundled definitive endpoint theorem.

-   **Pass-37 definitive API consolidation layer:** The theorem package now further consolidates the raw endpoint into single-name definitive API wrappers with theorem-level guarantees: (i) canonical runtime-output interpreter refinement to benchmark accepted/failure certificates without external execution assumptions, (ii) definitive deployment acceptance equivalence both to benchmark and canonical deployment contracts, (iii) definitive benchmark/deployment totality, and (iv) a final full-closure bundle theorem for the top-level definitive raw cross-docking API.

-   **Pass-38 acceptance-flag exactness layer:** The theorem package now additionally proves exact two-sided accept/reject characterizations for the definitive API: benchmark accepted/failure iff benchmark-contract/negation, deployment accepted/rejected iff deployment-contract/negation, runtime accept-flag equivalence to both benchmark acceptance and deployment acceptance, runtime reject-flag equivalence to deployment rejection, and consolidated top-level closure bundling these exactness properties.

-   **Pass-39 computable rational-kernel layer:** The theorem package now additionally provides a computable (Boolean-decidable) rationalized acceptance kernel with explicit margin inequalities, together with soundness transport from rational checks to real benchmark contract satisfaction and an acceptance-refinement theorem from computable flag truth to formal accepted solver certificate existence.

-   **Pass-40 interpreter/report refinement layer:** The theorem package now additionally formalizes an explicit canonical interpreter-state runtime view and proves its refinement equivalence to solver certificate outcomes, plus definitive wrapper theorems for interpreter-runtime identity and consolidated report-field runtime-accept/deployment-accept equivalence.

-   **Pass-41 complete Lean bundle layer:** The theorem package now additionally exposes one top-level complete Lean closure theorem for the definitive raw cross-docking API, jointly bundling support inclusion, JAX codegen success, benchmark/deployment contract equivalences, runtime accept/reject exactness, deployment totality, and computable rational-kernel acceptance refinement to benchmark certificates.

-   **Pass-42 report reject-flag exactness layer:** The theorem package now additionally proves report-level runtime reject-flag equivalence to deployment rejection certificates, complementing the existing report-level runtime accept-flag equivalence and completing two-sided report runtime/deployment exactness.

-   **Pass-43 endpoint-stabilization layer:** The theorem package now additionally restores full Lean compilability of the definitive raw cross-docking endpoint after generic-ProgramIR integration edits by (i) proving the generic two-fuel entry execution theorem via explicit runtime-flag case analysis and (ii) re-establishing exact-rational witness flag/benchmark-contract equivalence with a direct decide-and-cast proof path; no new paper handles are introduced in this stabilization pass.

-   **Pass-44 constructive endpoint closure layer:** The theorem package now additionally introduces artifact-instantiated constructive top-level benchmark/deployment decision paths for the definitive raw endpoint (no noncomputable marker on the new top-level constructive wrappers), proves accepted-decision refinement back to legacy benchmark/deployment acceptance certificates, and tightens rational witness assumptions into explicit artifact payload records (artifact identity, backend name, generated-op digest, rational values, and certified error bounds), together with exact-rational artifact equivalence/refinement theorems.

-   **Pass-45 legacy-chain cleanup layer:** The theorem package now additionally isolates all prior nonconstructive definitive endpoint operators behind explicit 'legacy\...' names (solver outputs, runtime flag, interpreter/runtime report constructors), keeps theorem compatibility by private in-file aliases only, and exposes public constructive-first decision endpoints plus direct alias-exactness and decision-to-legacy refinement theorems; this removes legacy operators from the public endpoint surface while preserving backward theorem continuity.

-   **Pass-46 constructive certification and signed-artifact closure layer:** The theorem package now additionally upgrades the public constructive endpoint from decision-only outputs to certificate-carrying benchmark/deployment result types (accepted/rejected certificate objects with exact accepted/rejected iff decision theorems and totality), adds signed-manifest/signature-verifier artifact interfaces with direct conversion into constructive kernel instantiations and acceptance-refinement theorems, and extends exact-rational closure with constructive rejection refinement to legacy benchmark-failure and deployment-rejection certificates (including signed exact-rational transport).

-   **Pass-47 byte-verified cryptographic closure and constructive-surface rebasing layer:** The theorem package now additionally formalizes an explicit byte-level signed-artifact envelope parser (length-prefix encoding with parse-roundtrip theorem), instantiates a concrete checksum signature verifier and proves end-to-end parse+verify success, lifts that concrete verifier to signed rationalized artifacts, rebases definitive theorem-facing names/claims to certificate-backend terminology (removing explicit legacy naming from the constructive theorem surface), and adds strict-separation rejection transport for broad signed rationalized artifacts, yielding certified benchmark-failure and deployment-rejection refinements beyond exact-rational-only rejection closure.

-   **Pass-48 optimization-theorem integration layer:** The theorem package now additionally integrates formal runtime-optimization endpoints into the definitive constructive computable pipeline: (i) exact closed-form and successor-recursive operation-count theorems parameterized by pocket budget, ligand budget, conformer count, fuel, and parser bytes; (ii) certified branch-and-bound prune soundness and adaptive stop-rule soundness proving no decision flip under upper/lower bound separation; (iii) explicit pipeline-level branch-and-bound wrapper soundness; (iv) imported ArrayDSL sharded/fused reduction equivalence to justify batch/fusion vectorization semantics; and (v) signed-artifact parser cost linearity plus cryptographic-verifier assumption bridge and signed-crypto parse/verify integration theorem.

-   **Pass-49 scorer-fusion and campaign-cardinality decomposition layer:** The theorem package now additionally makes the optimization closure more endpoint-explicit by (i) adding campaign pair-evaluation closed-form and successor-recursive recurrences with explicit $(K\cdot L)\cdot C\cdot \mathrm{fuel}$ decomposition, (ii) lifting those pair-evaluation counts through the integrated definitive computable pipeline wrapper, (iii) importing fused/unfused ArrayDSL pair-potential scorer equivalence directly at the definitive pipeline layer, and (iv) proving that the canonical ProgramIR explicitly requires the 'sumPairPotentials' scorer op and that its fused implementation is theorem-justified.

::: remark
[]{#rem:molecular-independence-scope label="rem:molecular-independence-scope"} Corollary [\[cor:holonomic-landauer-floor\]](#cor:holonomic-landauer-floor){reference-type="ref" reference="cor:holonomic-landauer-floor"} proves the finite constrained-molecular transport once the RATTLE holonomic topology supplies $k$ independent constraints and the corresponding binary status interface. Corollary [\[cor:bond-family-holonomic-floor\]](#cor:bond-family-holonomic-floor){reference-type="ref" reference="cor:bond-family-holonomic-floor"} derives that independent-count antecedent for concrete finite bond families carrying the explicit certificate $|F|\le N$ with $N>0$. Corollary [\[cor:jacobian-holonomic-floor\]](#cor:jacobian-holonomic-floor){reference-type="ref" reference="cor:jacobian-holonomic-floor"} discharges the Jacobian hypotheses for nonlinear geometric families that provide a pivot-column Jacobian certificate at the chosen configuration. The new geometric-constraint decision-interface theorem additionally covers broad bond/angle/dihedral composites whenever an explicit decision procedure certifies the strict independent-count inequality from the raw constraint data.
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

-   Proposition [\[prop:binding-as-exact-resolution\]](#prop:binding-as-exact-resolution){reference-type="ref" reference="prop:binding-as-exact-resolution"}

**Exact Resolution, Quotient Structure, and Compression (Section 3):**

-   Theorem [\[thm:dof-srank\]](#thm:dof-srank){reference-type="ref" reference="thm:dof-srank"}

-   Corollary [\[cor:rank-one\]](#cor:rank-one){reference-type="ref" reference="cor:rank-one"}

-   Corollary [\[cor:rank-above-one\]](#cor:rank-above-one){reference-type="ref" reference="cor:rank-above-one"}

-   Theorem [\[thm:min-bit-operations\]](#thm:min-bit-operations){reference-type="ref" reference="thm:min-bit-operations"}

-   Theorem [\[thm:numopt-bound\]](#thm:numopt-bound){reference-type="ref" reference="thm:numopt-bound"}

-   Theorem [\[thm:optimizer-class-richness-rank-lower-bound\]](#thm:optimizer-class-richness-rank-lower-bound){reference-type="ref" reference="thm:optimizer-class-richness-rank-lower-bound"}

-   Theorem [\[thm:entropy-bound\]](#thm:entropy-bound){reference-type="ref" reference="thm:entropy-bound"}

-   Theorem [\[thm:abstraction-factors-or-erases\]](#thm:abstraction-factors-or-erases){reference-type="ref" reference="thm:abstraction-factors-or-erases"}

-   Theorem [\[thm:feasible-collapse-factors\]](#thm:feasible-collapse-factors){reference-type="ref" reference="thm:feasible-collapse-factors"}

-   Theorem [\[thm:admissible-docking-exhaustion\]](#thm:admissible-docking-exhaustion){reference-type="ref" reference="thm:admissible-docking-exhaustion"}

-   Theorem [\[thm:admissibility-progress-monotone\]](#thm:admissibility-progress-monotone){reference-type="ref" reference="thm:admissibility-progress-monotone"}

-   Theorem [\[thm:fisher-sum-srank\]](#thm:fisher-sum-srank){reference-type="ref" reference="thm:fisher-sum-srank"}

-   Theorem [\[thm:fisher-rank-srank\]](#thm:fisher-rank-srank){reference-type="ref" reference="thm:fisher-rank-srank"}

-   Theorem [\[thm:fisher-likelihood-indicator\]](#thm:fisher-likelihood-indicator){reference-type="ref" reference="thm:fisher-likelihood-indicator"}

-   Theorem [\[thm:parametric-fisher-identifiable-dimension\]](#thm:parametric-fisher-identifiable-dimension){reference-type="ref" reference="thm:parametric-fisher-identifiable-dimension"}

-   Theorem [\[thm:cramer-rao-nonidentifiable-irrelevant\]](#thm:cramer-rao-nonidentifiable-irrelevant){reference-type="ref" reference="thm:cramer-rao-nonidentifiable-irrelevant"}

-   Theorem [\[thm:fisher-observation-channel-transport\]](#thm:fisher-observation-channel-transport){reference-type="ref" reference="thm:fisher-observation-channel-transport"}

-   Theorem [\[thm:fisher-noisy-partial-channel-example\]](#thm:fisher-noisy-partial-channel-example){reference-type="ref" reference="thm:fisher-noisy-partial-channel-example"}

-   Proposition [\[prop:finite-compression-bridge\]](#prop:finite-compression-bridge){reference-type="ref" reference="prop:finite-compression-bridge"}

-   Theorem [\[thm:canonical-initiality\]](#thm:canonical-initiality){reference-type="ref" reference="thm:canonical-initiality"}

-   Corollary [\[thm:canonical-initiality-srank\]](#thm:canonical-initiality-srank){reference-type="ref" reference="thm:canonical-initiality-srank"}

-   Theorem [\[thm:kernel-quotient-universality-canonicity\]](#thm:kernel-quotient-universality-canonicity){reference-type="ref" reference="thm:kernel-quotient-universality-canonicity"}

-   Theorem [\[thm:kernel-completion-categorical-endpoint\]](#thm:kernel-completion-categorical-endpoint){reference-type="ref" reference="thm:kernel-completion-categorical-endpoint"}

-   Theorem [\[thm:ae-rg-kernel-endpoint\]](#thm:ae-rg-kernel-endpoint){reference-type="ref" reference="thm:ae-rg-kernel-endpoint"}

-   Theorem [\[thm:measure-kernel-db-stationary-transport\]](#thm:measure-kernel-db-stationary-transport){reference-type="ref" reference="thm:measure-kernel-db-stationary-transport"}

-   Corollary [\[thm:deterministic-kernel-scale-stationary\]](#thm:deterministic-kernel-scale-stationary){reference-type="ref" reference="thm:deterministic-kernel-scale-stationary"}

-   Theorem [\[thm:kernel-power-stationarity-transport\]](#thm:kernel-power-stationarity-transport){reference-type="ref" reference="thm:kernel-power-stationarity-transport"}

-   Theorem [\[thm:measure-kernel-quotient-rg-endpoint\]](#thm:measure-kernel-quotient-rg-endpoint){reference-type="ref" reference="thm:measure-kernel-quotient-rg-endpoint"}

-   Corollary [\[thm:measure-kernel-transport-quotient-instance-recovery\]](#thm:measure-kernel-transport-quotient-instance-recovery){reference-type="ref" reference="thm:measure-kernel-transport-quotient-instance-recovery"}

-   Theorem [\[thm:path-space-crooks-kernel-power-transport\]](#thm:path-space-crooks-kernel-power-transport){reference-type="ref" reference="thm:path-space-crooks-kernel-power-transport"}

-   Theorem [\[thm:path-space-jarzynski-kernel-power-transport\]](#thm:path-space-jarzynski-kernel-power-transport){reference-type="ref" reference="thm:path-space-jarzynski-kernel-power-transport"}

-   Theorem [\[thm:path-measure-jarzynski-integral-transport\]](#thm:path-measure-jarzynski-integral-transport){reference-type="ref" reference="thm:path-measure-jarzynski-integral-transport"}

-   Theorem [\[thm:path-process-transport-endpoint\]](#thm:path-process-transport-endpoint){reference-type="ref" reference="thm:path-process-transport-endpoint"}

-   Theorem [\[thm:measurable-transition-finite-horizon-canonical-endpoint\]](#thm:measurable-transition-finite-horizon-canonical-endpoint){reference-type="ref" reference="thm:measurable-transition-finite-horizon-canonical-endpoint"}

-   Theorem [\[thm:measurable-transition-projective-kolmogorov-endpoint\]](#thm:measurable-transition-projective-kolmogorov-endpoint){reference-type="ref" reference="thm:measurable-transition-projective-kolmogorov-endpoint"}

-   Theorem [\[thm:measurable-transition-projective-canonical-instance\]](#thm:measurable-transition-projective-canonical-instance){reference-type="ref" reference="thm:measurable-transition-projective-canonical-instance"}

-   Theorem [\[thm:measurable-transition-constructive-truncation\]](#thm:measurable-transition-constructive-truncation){reference-type="ref" reference="thm:measurable-transition-constructive-truncation"}

-   Theorem [\[thm:continuous-time-continuous-state-interface\]](#thm:continuous-time-continuous-state-interface){reference-type="ref" reference="thm:continuous-time-continuous-state-interface"}

-   Theorem [\[thm:measurable-transition-kernel-concrete-endpoint\]](#thm:measurable-transition-kernel-concrete-endpoint){reference-type="ref" reference="thm:measurable-transition-kernel-concrete-endpoint"}

-   Corollary [\[thm:measurable-transition-jarzynski-concrete-endpoint\]](#thm:measurable-transition-jarzynski-concrete-endpoint){reference-type="ref" reference="thm:measurable-transition-jarzynski-concrete-endpoint"}

-   Corollary [\[thm:measurable-transition-path-measure-jarzynski-concrete-endpoint\]](#thm:measurable-transition-path-measure-jarzynski-concrete-endpoint){reference-type="ref" reference="thm:measurable-transition-path-measure-jarzynski-concrete-endpoint"}

-   Theorem [\[thm:encoding-transport-rank-energy\]](#thm:encoding-transport-rank-energy){reference-type="ref" reference="thm:encoding-transport-rank-energy"}

-   Theorem [\[thm:optimizer-class-richness-nonbinary-vc\]](#thm:optimizer-class-richness-nonbinary-vc){reference-type="ref" reference="thm:optimizer-class-richness-nonbinary-vc"}

**Complexity Boundary (Section 4):**

-   Theorem [\[thm:exact-sufficiency-hardness-core\]](#thm:exact-sufficiency-hardness-core){reference-type="ref" reference="thm:exact-sufficiency-hardness-core"}

-   Theorem [\[thm:hard-family-srank\]](#thm:hard-family-srank){reference-type="ref" reference="thm:hard-family-srank"}

-   Theorem [\[thm:checker-budget-lower-bound\]](#thm:checker-budget-lower-bound){reference-type="ref" reference="thm:checker-budget-lower-bound"}

-   Corollary [\[cor:no-sound-checker-below-budget\]](#cor:no-sound-checker-below-budget){reference-type="ref" reference="cor:no-sound-checker-below-budget"}

-   Corollary [\[cor:checking-time-lower-bound\]](#cor:checking-time-lower-bound){reference-type="ref" reference="cor:checking-time-lower-bound"}

-   Theorem [\[thm:molecular-docking-srank-bound\]](#thm:molecular-docking-srank-bound){reference-type="ref" reference="thm:molecular-docking-srank-bound"}

-   Corollary [\[cor:bounded-pocket-regime\]](#cor:bounded-pocket-regime){reference-type="ref" reference="cor:bounded-pocket-regime"}

-   Theorem [\[thm:allosteric-srank-graph\]](#thm:allosteric-srank-graph){reference-type="ref" reference="thm:allosteric-srank-graph"}

-   Theorem [\[thm:geometric-contact-allostery\]](#thm:geometric-contact-allostery){reference-type="ref" reference="thm:geometric-contact-allostery"}

-   Theorem [\[thm:contact-shell-allostery\]](#thm:contact-shell-allostery){reference-type="ref" reference="thm:contact-shell-allostery"}

-   Corollary [\[cor:bounded-contact-shell-regime\]](#cor:bounded-contact-shell-regime){reference-type="ref" reference="cor:bounded-contact-shell-regime"}

-   Theorem [\[thm:allosteric-distance-decay-law\]](#thm:allosteric-distance-decay-law){reference-type="ref" reference="thm:allosteric-distance-decay-law"}

-   Theorem [\[thm:allosteric-exponential-distance-decay\]](#thm:allosteric-exponential-distance-decay){reference-type="ref" reference="thm:allosteric-exponential-distance-decay"}

-   Theorem [\[thm:allosteric-polynomial-distance-decay\]](#thm:allosteric-polynomial-distance-decay){reference-type="ref" reference="thm:allosteric-polynomial-distance-decay"}

-   Theorem [\[thm:allosteric-polynomial-series-budget\]](#thm:allosteric-polynomial-series-budget){reference-type="ref" reference="thm:allosteric-polynomial-series-budget"}

-   Theorem [\[thm:allosteric-polynomial-series-explicit\]](#thm:allosteric-polynomial-series-explicit){reference-type="ref" reference="thm:allosteric-polynomial-series-explicit"}

-   Theorem [\[thm:mechanochemical-coupling-gap\]](#thm:mechanochemical-coupling-gap){reference-type="ref" reference="thm:mechanochemical-coupling-gap"}

-   Theorem [\[thm:hierarchical-srank-bound\]](#thm:hierarchical-srank-bound){reference-type="ref" reference="thm:hierarchical-srank-bound"}

-   Theorem [\[thm:renormalized-admissibility-equivalence\]](#thm:renormalized-admissibility-equivalence){reference-type="ref" reference="thm:renormalized-admissibility-equivalence"}

-   Theorem [\[thm:lipschitz-grid-approx\]](#thm:lipschitz-grid-approx){reference-type="ref" reference="thm:lipschitz-grid-approx"}

-   Theorem [\[thm:resolution-controlled-uniform\]](#thm:resolution-controlled-uniform){reference-type="ref" reference="thm:resolution-controlled-uniform"}

-   Theorem [\[thm:action-pinned-lifting\]](#thm:action-pinned-lifting){reference-type="ref" reference="thm:action-pinned-lifting"}

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

-   Theorem [\[thm:coordinate-extraction-poly\]](#thm:coordinate-extraction-poly){reference-type="ref" reference="thm:coordinate-extraction-poly"}

-   Theorem [\[thm:inverse-rank-gap-design\]](#thm:inverse-rank-gap-design){reference-type="ref" reference="thm:inverse-rank-gap-design"}

-   Theorem [\[thm:inverse-design-atomistic-bridge\]](#thm:inverse-design-atomistic-bridge){reference-type="ref" reference="thm:inverse-design-atomistic-bridge"}

-   Theorem [\[thm:strict-dominance-empty-sufficient\]](#thm:strict-dominance-empty-sufficient){reference-type="ref" reference="thm:strict-dominance-empty-sufficient"}

-   Theorem [\[thm:multiplicative-separable-empty-sufficient\]](#thm:multiplicative-separable-empty-sufficient){reference-type="ref" reference="thm:multiplicative-separable-empty-sufficient"}

-   Theorem [\[thm:sampled-docking-gap\]](#thm:sampled-docking-gap){reference-type="ref" reference="thm:sampled-docking-gap"}

-   Theorem [\[thm:action-pinned-uniform-gap\]](#thm:action-pinned-uniform-gap){reference-type="ref" reference="thm:action-pinned-uniform-gap"}

-   Theorem [\[thm:admissibility-rank-reduction\]](#thm:admissibility-rank-reduction){reference-type="ref" reference="thm:admissibility-rank-reduction"}

-   Theorem [\[thm:quantitative-admissibility-rank-collapse\]](#thm:quantitative-admissibility-rank-collapse){reference-type="ref" reference="thm:quantitative-admissibility-rank-collapse"}

-   Theorem [\[thm:admissibility-zero-collapse-bifactor\]](#thm:admissibility-zero-collapse-bifactor){reference-type="ref" reference="thm:admissibility-zero-collapse-bifactor"}

-   Theorem [\[thm:admissibility-collapse-canonicality\]](#thm:admissibility-collapse-canonicality){reference-type="ref" reference="thm:admissibility-collapse-canonicality"}

-   Theorem [\[thm:lj-tolerance-collapse-explicit\]](#thm:lj-tolerance-collapse-explicit){reference-type="ref" reference="thm:lj-tolerance-collapse-explicit"}

-   Theorem [\[thm:collapsed-rank-category-structure\]](#thm:collapsed-rank-category-structure){reference-type="ref" reference="thm:collapsed-rank-category-structure"}

-   Theorem [\[thm:bounded-collapsed-rank-finite-colimits\]](#thm:bounded-collapsed-rank-finite-colimits){reference-type="ref" reference="thm:bounded-collapsed-rank-finite-colimits"}

-   Theorem [\[thm:bounded-collapsed-rank-finite-family-lattice\]](#thm:bounded-collapsed-rank-finite-family-lattice){reference-type="ref" reference="thm:bounded-collapsed-rank-finite-family-lattice"}

-   Theorem [\[thm:bounded-collapsed-rank-complete-lattice\]](#thm:bounded-collapsed-rank-complete-lattice){reference-type="ref" reference="thm:bounded-collapsed-rank-complete-lattice"}

-   Theorem [\[thm:bounded-collapsed-rank-algebraic-laws\]](#thm:bounded-collapsed-rank-algebraic-laws){reference-type="ref" reference="thm:bounded-collapsed-rank-algebraic-laws"}

-   Theorem [\[thm:bounded-collapsed-rank-binary-calculus\]](#thm:bounded-collapsed-rank-binary-calculus){reference-type="ref" reference="thm:bounded-collapsed-rank-binary-calculus"}

-   Theorem [\[thm:sampled-inside-cutoff-sufficient\]](#thm:sampled-inside-cutoff-sufficient){reference-type="ref" reference="thm:sampled-inside-cutoff-sufficient"}

-   Theorem [\[thm:topk-boundary-gap\]](#thm:topk-boundary-gap){reference-type="ref" reference="thm:topk-boundary-gap"}

-   Theorem [\[thm:topk-ambiguity-band\]](#thm:topk-ambiguity-band){reference-type="ref" reference="thm:topk-ambiguity-band"}

-   Theorem [\[thm:composed-classical-forcefield-interface\]](#thm:composed-classical-forcefield-interface){reference-type="ref" reference="thm:composed-classical-forcefield-interface"}

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

-   Theorem [\[thm:top-level-computable-export\]](#thm:top-level-computable-export){reference-type="ref" reference="thm:top-level-computable-export"}

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

-   Theorem [\[thm:quotient-resolution-speed-bound\]](#thm:quotient-resolution-speed-bound){reference-type="ref" reference="thm:quotient-resolution-speed-bound"}

-   Theorem [\[thm:admissibility-speed-accuracy-tradeoff\]](#thm:admissibility-speed-accuracy-tradeoff){reference-type="ref" reference="thm:admissibility-speed-accuracy-tradeoff"}

-   Theorem [\[thm:admissibility-speed-accuracy-zero-collapse\]](#thm:admissibility-speed-accuracy-zero-collapse){reference-type="ref" reference="thm:admissibility-speed-accuracy-zero-collapse"}

-   Theorem [\[thm:admissibility-onrate-envelope\]](#thm:admissibility-onrate-envelope){reference-type="ref" reference="thm:admissibility-onrate-envelope"}

-   Theorem [\[thm:trajectory-time-energy-tradeoff\]](#thm:trajectory-time-energy-tradeoff){reference-type="ref" reference="thm:trajectory-time-energy-tradeoff"}

-   Theorem [\[thm:budget-class-bound\]](#thm:budget-class-bound){reference-type="ref" reference="thm:budget-class-bound"}

-   Corollary [\[cor:budget-entropy-bound\]](#cor:budget-entropy-bound){reference-type="ref" reference="cor:budget-entropy-bound"}

-   Corollary [\[cor:composition-budget-law\]](#cor:composition-budget-law){reference-type="ref" reference="cor:composition-budget-law"}

-   Theorem [\[thm:pathwise-energy-lower-bound\]](#thm:pathwise-energy-lower-bound){reference-type="ref" reference="thm:pathwise-energy-lower-bound"}

-   Theorem [\[thm:quotient-trajectory-crooks\]](#thm:quotient-trajectory-crooks){reference-type="ref" reference="thm:quotient-trajectory-crooks"}

-   Theorem [\[thm:rank-calibrated-crooks-standard\]](#thm:rank-calibrated-crooks-standard){reference-type="ref" reference="thm:rank-calibrated-crooks-standard"}

-   Theorem [\[thm:crooks-detailed-balance-equilibrium\]](#thm:crooks-detailed-balance-equilibrium){reference-type="ref" reference="thm:crooks-detailed-balance-equilibrium"}

-   Theorem [\[thm:md-class-crooks-calibration-interface\]](#thm:md-class-crooks-calibration-interface){reference-type="ref" reference="thm:md-class-crooks-calibration-interface"}

-   Theorem [\[thm:langevin-to-mcmc-discretization-interface\]](#thm:langevin-to-mcmc-discretization-interface){reference-type="ref" reference="thm:langevin-to-mcmc-discretization-interface"}

-   Theorem [\[thm:composed-hamiltonian-architecture-instance\]](#thm:composed-hamiltonian-architecture-instance){reference-type="ref" reference="thm:composed-hamiltonian-architecture-instance"}

-   Theorem [\[thm:composed-hamiltonian-lipschitz-bound\]](#thm:composed-hamiltonian-lipschitz-bound){reference-type="ref" reference="thm:composed-hamiltonian-lipschitz-bound"}

-   Theorem [\[thm:concrete-biomolecular-forcefield-calibration-bundle\]](#thm:concrete-biomolecular-forcefield-calibration-bundle){reference-type="ref" reference="thm:concrete-biomolecular-forcefield-calibration-bundle"}

-   Theorem [\[thm:reference-biomolecular-zero-shell-calibration\]](#thm:reference-biomolecular-zero-shell-calibration){reference-type="ref" reference="thm:reference-biomolecular-zero-shell-calibration"}

-   Theorem [\[thm:nontrivial-biomolecular-forcefield-transport\]](#thm:nontrivial-biomolecular-forcefield-transport){reference-type="ref" reference="thm:nontrivial-biomolecular-forcefield-transport"}

-   Theorem [\[thm:fullstate-biomolecular-forcefield-transport\]](#thm:fullstate-biomolecular-forcefield-transport){reference-type="ref" reference="thm:fullstate-biomolecular-forcefield-transport"}

-   Theorem [\[thm:pairwise-geometric-forcefield-transport\]](#thm:pairwise-geometric-forcefield-transport){reference-type="ref" reference="thm:pairwise-geometric-forcefield-transport"}

-   Theorem [\[thm:pairwise-geometric-derived-transport\]](#thm:pairwise-geometric-derived-transport){reference-type="ref" reference="thm:pairwise-geometric-derived-transport"}

-   Theorem [\[thm:constructive-empirical-realism-shell-envelope\]](#thm:constructive-empirical-realism-shell-envelope){reference-type="ref" reference="thm:constructive-empirical-realism-shell-envelope"}

-   Theorem [\[thm:constructive-empirical-realism-transport\]](#thm:constructive-empirical-realism-transport){reference-type="ref" reference="thm:constructive-empirical-realism-transport"}

-   Theorem [\[thm:biomolecular-reference-realism-shell-values\]](#thm:biomolecular-reference-realism-shell-values){reference-type="ref" reference="thm:biomolecular-reference-realism-shell-values"}

-   Theorem [\[thm:biomolecular-reference-realism-transport\]](#thm:biomolecular-reference-realism-transport){reference-type="ref" reference="thm:biomolecular-reference-realism-transport"}

-   Theorem [\[thm:biomolecular-reference-realism-explicit-shell-constant\]](#thm:biomolecular-reference-realism-explicit-shell-constant){reference-type="ref" reference="thm:biomolecular-reference-realism-explicit-shell-constant"}

-   Theorem [\[thm:electronic-structure-correction-transport\]](#thm:electronic-structure-correction-transport){reference-type="ref" reference="thm:electronic-structure-correction-transport"}

-   Theorem [\[thm:electronic-reference-shell-error-transport\]](#thm:electronic-reference-shell-error-transport){reference-type="ref" reference="thm:electronic-reference-shell-error-transport"}

-   Theorem [\[thm:qm-grounded-electronic-structure-transport\]](#thm:qm-grounded-electronic-structure-transport){reference-type="ref" reference="thm:qm-grounded-electronic-structure-transport"}

-   Theorem [\[thm:qm-method-specific-electronic-transport\]](#thm:qm-method-specific-electronic-transport){reference-type="ref" reference="thm:qm-method-specific-electronic-transport"}

-   Theorem [\[thm:qm-protocol-derived-method-calibration-transport\]](#thm:qm-protocol-derived-method-calibration-transport){reference-type="ref" reference="thm:qm-protocol-derived-method-calibration-transport"}

-   Theorem [\[thm:qm-grounded-electronic-halfgap-transport\]](#thm:qm-grounded-electronic-halfgap-transport){reference-type="ref" reference="thm:qm-grounded-electronic-halfgap-transport"}

-   Theorem [\[thm:langevin-detailed-balance-grounding\]](#thm:langevin-detailed-balance-grounding){reference-type="ref" reference="thm:langevin-detailed-balance-grounding"}

-   Theorem [\[thm:langevin-discretization-certified\]](#thm:langevin-discretization-certified){reference-type="ref" reference="thm:langevin-discretization-certified"}

-   Theorem [\[thm:langevin-boltzmann-stationarity-measure-derivation\]](#thm:langevin-boltzmann-stationarity-measure-derivation){reference-type="ref" reference="thm:langevin-boltzmann-stationarity-measure-derivation"}

-   Theorem [\[thm:langevin-analysis-closure-endpoint-bundle\]](#thm:langevin-analysis-closure-endpoint-bundle){reference-type="ref" reference="thm:langevin-analysis-closure-endpoint-bundle"}

-   Theorem [\[thm:unified-langevin-assumption-bundle-discharge\]](#thm:unified-langevin-assumption-bundle-discharge){reference-type="ref" reference="thm:unified-langevin-assumption-bundle-discharge"}

-   Theorem [\[thm:langevin-explicit-sde-conditions\]](#thm:langevin-explicit-sde-conditions){reference-type="ref" reference="thm:langevin-explicit-sde-conditions"}

-   Theorem [\[thm:langevin-microscopic-derivation-constructor\]](#thm:langevin-microscopic-derivation-constructor){reference-type="ref" reference="thm:langevin-microscopic-derivation-constructor"}

-   Theorem [\[thm:langevin-molecular-hamiltonian-thermostat-constructor\]](#thm:langevin-molecular-hamiltonian-thermostat-constructor){reference-type="ref" reference="thm:langevin-molecular-hamiltonian-thermostat-constructor"}

-   Theorem [\[thm:langevin-concrete-hamiltonian-bath-end-to-end\]](#thm:langevin-concrete-hamiltonian-bath-end-to-end){reference-type="ref" reference="thm:langevin-concrete-hamiltonian-bath-end-to-end"}

-   Theorem [\[thm:langevin-molecular-pathprocess-joint-bundle\]](#thm:langevin-molecular-pathprocess-joint-bundle){reference-type="ref" reference="thm:langevin-molecular-pathprocess-joint-bundle"}

-   Theorem [\[thm:langevin-continuous-state-measure-closure\]](#thm:langevin-continuous-state-measure-closure){reference-type="ref" reference="thm:langevin-continuous-state-measure-closure"}

-   Theorem [\[thm:langevin-ito-fp-harris-closure\]](#thm:langevin-ito-fp-harris-closure){reference-type="ref" reference="thm:langevin-ito-fp-harris-closure"}

-   Theorem [\[thm:langevin-constructive-infinite-dimensional-path-derivation-bridge\]](#thm:langevin-constructive-infinite-dimensional-path-derivation-bridge){reference-type="ref" reference="thm:langevin-constructive-infinite-dimensional-path-derivation-bridge"}

-   Theorem [\[thm:langevin-finite-horizon-law-derivation-bridge\]](#thm:langevin-finite-horizon-law-derivation-bridge){reference-type="ref" reference="thm:langevin-finite-horizon-law-derivation-bridge"}

-   Theorem [\[thm:langevin-finite-horizon-marginal-recovery\]](#thm:langevin-finite-horizon-marginal-recovery){reference-type="ref" reference="thm:langevin-finite-horizon-marginal-recovery"}

-   Theorem [\[thm:langevin-canonical-continuous-path-process-closure\]](#thm:langevin-canonical-continuous-path-process-closure){reference-type="ref" reference="thm:langevin-canonical-continuous-path-process-closure"}

-   Theorem [\[thm:langevin-canonical-pathprocess-of-finite-horizon-bridge\]](#thm:langevin-canonical-pathprocess-of-finite-horizon-bridge){reference-type="ref" reference="thm:langevin-canonical-pathprocess-of-finite-horizon-bridge"}

-   Theorem [\[thm:langevin-forcefield-derived-lipschitz-injection\]](#thm:langevin-forcefield-derived-lipschitz-injection){reference-type="ref" reference="thm:langevin-forcefield-derived-lipschitz-injection"}

-   Theorem [\[thm:langevin-first-principles-discharge\]](#thm:langevin-first-principles-discharge){reference-type="ref" reference="thm:langevin-first-principles-discharge"}

-   Theorem [\[thm:concrete-langevin-interface-constructor\]](#thm:concrete-langevin-interface-constructor){reference-type="ref" reference="thm:concrete-langevin-interface-constructor"}

-   Theorem [\[thm:spinhalf-canonicaldp-instance\]](#thm:spinhalf-canonicaldp-instance){reference-type="ref" reference="thm:spinhalf-canonicaldp-instance"}

-   Theorem [\[thm:spinhalf-decoherence-floor\]](#thm:spinhalf-decoherence-floor){reference-type="ref" reference="thm:spinhalf-decoherence-floor"}

-   Theorem [\[thm:top-level-computable-path\]](#thm:top-level-computable-path){reference-type="ref" reference="thm:top-level-computable-path"}

-   Theorem [\[thm:legacy-constructive-deprecation-bridge\]](#thm:legacy-constructive-deprecation-bridge){reference-type="ref" reference="thm:legacy-constructive-deprecation-bridge"}

-   Theorem [\[thm:constructive-downstream-replacement-transport\]](#thm:constructive-downstream-replacement-transport){reference-type="ref" reference="thm:constructive-downstream-replacement-transport"}

-   Theorem [\[thm:fully-constructive-pipeline-deprecation-ready\]](#thm:fully-constructive-pipeline-deprecation-ready){reference-type="ref" reference="thm:fully-constructive-pipeline-deprecation-ready"}

-   Theorem [\[thm:constructive-only-spec-replacement\]](#thm:constructive-only-spec-replacement){reference-type="ref" reference="thm:constructive-only-spec-replacement"}

-   Theorem [\[thm:constructive-only-core-field-consumers\]](#thm:constructive-only-core-field-consumers){reference-type="ref" reference="thm:constructive-only-core-field-consumers"}

-   Theorem [\[thm:constructive-only-extended-field-consumers\]](#thm:constructive-only-extended-field-consumers){reference-type="ref" reference="thm:constructive-only-extended-field-consumers"}

-   Theorem [\[thm:continuous-state-measurable-encoding-transport\]](#thm:continuous-state-measurable-encoding-transport){reference-type="ref" reference="thm:continuous-state-measurable-encoding-transport"}

-   Theorem [\[thm:chemical-augmented-docking-transport\]](#thm:chemical-augmented-docking-transport){reference-type="ref" reference="thm:chemical-augmented-docking-transport"}

-   Theorem [\[thm:chemical-component-variation-opt-invariance\]](#thm:chemical-component-variation-opt-invariance){reference-type="ref" reference="thm:chemical-component-variation-opt-invariance"}

-   Theorem [\[thm:chemical-coupled-sensitivity\]](#thm:chemical-coupled-sensitivity){reference-type="ref" reference="thm:chemical-coupled-sensitivity"}

-   Theorem [\[thm:unified-chemical-state-dynamics-transport\]](#thm:unified-chemical-state-dynamics-transport){reference-type="ref" reference="thm:unified-chemical-state-dynamics-transport"}

-   Theorem [\[thm:chemical-conditioned-mechanism-dynamics\]](#thm:chemical-conditioned-mechanism-dynamics){reference-type="ref" reference="thm:chemical-conditioned-mechanism-dynamics"}

-   Theorem [\[thm:gibbs-conditioned-mechanism-dynamics\]](#thm:gibbs-conditioned-mechanism-dynamics){reference-type="ref" reference="thm:gibbs-conditioned-mechanism-dynamics"}

-   Theorem [\[thm:calibrated-biophysical-chemical-separation\]](#thm:calibrated-biophysical-chemical-separation){reference-type="ref" reference="thm:calibrated-biophysical-chemical-separation"}

-   Theorem [\[thm:chemical-dataset-posterior-separation\]](#thm:chemical-dataset-posterior-separation){reference-type="ref" reference="thm:chemical-dataset-posterior-separation"}

-   Theorem [\[thm:chemical-realdata-bias-aware-separation\]](#thm:chemical-realdata-bias-aware-separation){reference-type="ref" reference="thm:chemical-realdata-bias-aware-separation"}

-   Theorem [\[thm:hierarchical-chemical-realdata-separation-rate\]](#thm:hierarchical-chemical-realdata-separation-rate){reference-type="ref" reference="thm:hierarchical-chemical-realdata-separation-rate"}

-   Theorem [\[thm:hierarchical-chemical-realdata-separation-rate-margin\]](#thm:hierarchical-chemical-realdata-separation-rate-margin){reference-type="ref" reference="thm:hierarchical-chemical-realdata-separation-rate-margin"}

-   Theorem [\[thm:hierarchical-chemical-realdata-separation-rate-margin-all-datasets\]](#thm:hierarchical-chemical-realdata-separation-rate-margin-all-datasets){reference-type="ref" reference="thm:hierarchical-chemical-realdata-separation-rate-margin-all-datasets"}

-   Theorem [\[thm:hierarchical-chemical-realdata-separation-rate-constant-margin\]](#thm:hierarchical-chemical-realdata-separation-rate-constant-margin){reference-type="ref" reference="thm:hierarchical-chemical-realdata-separation-rate-constant-margin"}

-   Theorem [\[thm:hierarchical-chemical-realdata-separation-rate-constant-margin-all-datasets\]](#thm:hierarchical-chemical-realdata-separation-rate-constant-margin-all-datasets){reference-type="ref" reference="thm:hierarchical-chemical-realdata-separation-rate-constant-margin-all-datasets"}

-   Theorem [\[thm:absolute-binding-free-energy-closure\]](#thm:absolute-binding-free-energy-closure){reference-type="ref" reference="thm:absolute-binding-free-energy-closure"}

-   Theorem [\[thm:absolute-free-energy-protocol-derived-closure\]](#thm:absolute-free-energy-protocol-derived-closure){reference-type="ref" reference="thm:absolute-free-energy-protocol-derived-closure"}

-   Theorem [\[thm:conformational-ensemble-docking-transport\]](#thm:conformational-ensemble-docking-transport){reference-type="ref" reference="thm:conformational-ensemble-docking-transport"}

-   Theorem [\[thm:ensemble-one-step-population-transport\]](#thm:ensemble-one-step-population-transport){reference-type="ref" reference="thm:ensemble-one-step-population-transport"}

-   Theorem [\[thm:ensemble-multistep-population-transport\]](#thm:ensemble-multistep-population-transport){reference-type="ref" reference="thm:ensemble-multistep-population-transport"}

-   Theorem [\[thm:ensemble-statistical-validity-transport\]](#thm:ensemble-statistical-validity-transport){reference-type="ref" reference="thm:ensemble-statistical-validity-transport"}

-   Theorem [\[thm:chemical-ensemble-transport-binding-specialization\]](#thm:chemical-ensemble-transport-binding-specialization){reference-type="ref" reference="thm:chemical-ensemble-transport-binding-specialization"}

-   Theorem [\[thm:kinetic-observable-reporting-endpoints\]](#thm:kinetic-observable-reporting-endpoints){reference-type="ref" reference="thm:kinetic-observable-reporting-endpoints"}

-   Theorem [\[thm:kinetic-protocol-measurement-transport\]](#thm:kinetic-protocol-measurement-transport){reference-type="ref" reference="thm:kinetic-protocol-measurement-transport"}

-   Theorem [\[thm:kinetic-confidence-transport\]](#thm:kinetic-confidence-transport){reference-type="ref" reference="thm:kinetic-confidence-transport"}

-   Theorem [\[thm:kinetic-protocol-inference-guarantee\]](#thm:kinetic-protocol-inference-guarantee){reference-type="ref" reference="thm:kinetic-protocol-inference-guarantee"}

-   Theorem [\[thm:kinetic-concentration-identifiability-bridge\]](#thm:kinetic-concentration-identifiability-bridge){reference-type="ref" reference="thm:kinetic-concentration-identifiability-bridge"}

-   Theorem [\[thm:kinetic-replicate-identifiability-bundle\]](#thm:kinetic-replicate-identifiability-bundle){reference-type="ref" reference="thm:kinetic-replicate-identifiability-bundle"}

-   Theorem [\[thm:hierarchical-kinetic-replicate-inference-bundle\]](#thm:hierarchical-kinetic-replicate-inference-bundle){reference-type="ref" reference="thm:hierarchical-kinetic-replicate-inference-bundle"}

-   Theorem [\[thm:hierarchical-kinetic-replicate-inference-rate-margins\]](#thm:hierarchical-kinetic-replicate-inference-rate-margins){reference-type="ref" reference="thm:hierarchical-kinetic-replicate-inference-rate-margins"}

-   Theorem [\[thm:hierarchical-kinetic-replicate-inference-rate-margins-all-datasets\]](#thm:hierarchical-kinetic-replicate-inference-rate-margins-all-datasets){reference-type="ref" reference="thm:hierarchical-kinetic-replicate-inference-rate-margins-all-datasets"}

-   Theorem [\[thm:hierarchical-kinetic-replicate-inference-rate-constant-margins\]](#thm:hierarchical-kinetic-replicate-inference-rate-constant-margins){reference-type="ref" reference="thm:hierarchical-kinetic-replicate-inference-rate-constant-margins"}

-   Theorem [\[thm:hierarchical-kinetic-replicate-inference-rate-constant-margins-all-datasets\]](#thm:hierarchical-kinetic-replicate-inference-rate-constant-margins-all-datasets){reference-type="ref" reference="thm:hierarchical-kinetic-replicate-inference-rate-constant-margins-all-datasets"}

-   Theorem [\[thm:unified-simulator-kinetic-derivation\]](#thm:unified-simulator-kinetic-derivation){reference-type="ref" reference="thm:unified-simulator-kinetic-derivation"}

-   Theorem [\[thm:unified-thermo-kinetic-physical-model\]](#thm:unified-thermo-kinetic-physical-model){reference-type="ref" reference="thm:unified-thermo-kinetic-physical-model"}

-   Theorem [\[thm:universality-ood-calibration-bounds\]](#thm:universality-ood-calibration-bounds){reference-type="ref" reference="thm:universality-ood-calibration-bounds"}

-   Theorem [\[thm:ood-transfer-calibration-derived\]](#thm:ood-transfer-calibration-derived){reference-type="ref" reference="thm:ood-transfer-calibration-derived"}

-   Theorem [\[thm:finite-sample-complexity-inversion\]](#thm:finite-sample-complexity-inversion){reference-type="ref" reference="thm:finite-sample-complexity-inversion"}

-   Theorem [\[thm:hierarchical-required-size-joint-bundle\]](#thm:hierarchical-required-size-joint-bundle){reference-type="ref" reference="thm:hierarchical-required-size-joint-bundle"}

-   Theorem [\[thm:finite-sample-complexity-inversion-count-order\]](#thm:finite-sample-complexity-inversion-count-order){reference-type="ref" reference="thm:finite-sample-complexity-inversion-count-order"}

-   Theorem [\[thm:finite-sample-square-count-inversion\]](#thm:finite-sample-square-count-inversion){reference-type="ref" reference="thm:finite-sample-square-count-inversion"}

-   Theorem [\[thm:explicit-hamiltonian-bath-elimination-langevin-endpoints\]](#thm:explicit-hamiltonian-bath-elimination-langevin-endpoints){reference-type="ref" reference="thm:explicit-hamiltonian-bath-elimination-langevin-endpoints"}

-   Theorem [\[thm:explicit-molecular-constant-drift-sde-endpoints\]](#thm:explicit-molecular-constant-drift-sde-endpoints){reference-type="ref" reference="thm:explicit-molecular-constant-drift-sde-endpoints"}

-   Theorem [\[thm:concrete-generator-canonical-path-process-closure\]](#thm:concrete-generator-canonical-path-process-closure){reference-type="ref" reference="thm:concrete-generator-canonical-path-process-closure"}

-   Theorem [\[thm:qm-workflow-specific-realism-transport-bundle\]](#thm:qm-workflow-specific-realism-transport-bundle){reference-type="ref" reference="thm:qm-workflow-specific-realism-transport-bundle"}

-   Theorem [\[thm:reversible-chemical-transition-detailed-balance-bundle\]](#thm:reversible-chemical-transition-detailed-balance-bundle){reference-type="ref" reference="thm:reversible-chemical-transition-detailed-balance-bundle"}

-   Theorem [\[thm:trajectory-absolute-free-energy-correction-closure\]](#thm:trajectory-absolute-free-energy-correction-closure){reference-type="ref" reference="thm:trajectory-absolute-free-energy-correction-closure"}

-   Theorem [\[thm:quantified-simulator-thermo-kinetic-bundle\]](#thm:quantified-simulator-thermo-kinetic-bundle){reference-type="ref" reference="thm:quantified-simulator-thermo-kinetic-bundle"}

-   Theorem [\[thm:mechanistic-ood-uniform-transfer-bound\]](#thm:mechanistic-ood-uniform-transfer-bound){reference-type="ref" reference="thm:mechanistic-ood-uniform-transfer-bound"}

-   Theorem [\[thm:model-dependent-rate-term-target-margin\]](#thm:model-dependent-rate-term-target-margin){reference-type="ref" reference="thm:model-dependent-rate-term-target-margin"}

-   Theorem [\[thm:hierarchical-required-size-model-dependent-constants\]](#thm:hierarchical-required-size-model-dependent-constants){reference-type="ref" reference="thm:hierarchical-required-size-model-dependent-constants"}

-   Theorem [\[thm:hamiltonian-finite-difference-drift-derivation-endpoints\]](#thm:hamiltonian-finite-difference-drift-derivation-endpoints){reference-type="ref" reference="thm:hamiltonian-finite-difference-drift-derivation-endpoints"}

-   Theorem [\[thm:realistic-molecular-finite-difference-sde-endpoints\]](#thm:realistic-molecular-finite-difference-sde-endpoints){reference-type="ref" reference="thm:realistic-molecular-finite-difference-sde-endpoints"}

-   Theorem [\[thm:generator-coefficients-canonical-regularity-closure\]](#thm:generator-coefficients-canonical-regularity-closure){reference-type="ref" reference="thm:generator-coefficients-canonical-regularity-closure"}

-   Theorem [\[thm:concrete-qm-workflow-error-analysis-transport\]](#thm:concrete-qm-workflow-error-analysis-transport){reference-type="ref" reference="thm:concrete-qm-workflow-error-analysis-transport"}

-   Theorem [\[thm:barrier-crossing-reversible-chemical-dynamics\]](#thm:barrier-crossing-reversible-chemical-dynamics){reference-type="ref" reference="thm:barrier-crossing-reversible-chemical-dynamics"}

-   Theorem [\[thm:mixing-autocorrelation-trajectory-correction-total-error\]](#thm:mixing-autocorrelation-trajectory-correction-total-error){reference-type="ref" reference="thm:mixing-autocorrelation-trajectory-correction-total-error"}

-   Theorem [\[thm:unified-simulator-error-analysis-controlled-thermo-kinetic\]](#thm:unified-simulator-error-analysis-controlled-thermo-kinetic){reference-type="ref" reference="thm:unified-simulator-error-analysis-controlled-thermo-kinetic"}

-   Theorem [\[thm:descriptor-calibrated-mechanistic-ood-transfer\]](#thm:descriptor-calibrated-mechanistic-ood-transfer){reference-type="ref" reference="thm:descriptor-calibrated-mechanistic-ood-transfer"}

-   Theorem [\[thm:model-dependent-minimax-optimality-bundle\]](#thm:model-dependent-minimax-optimality-bundle){reference-type="ref" reference="thm:model-dependent-minimax-optimality-bundle"}

-   Theorem [\[thm:constructive-stochastic-multipole-scope-discharge\]](#thm:constructive-stochastic-multipole-scope-discharge){reference-type="ref" reference="thm:constructive-stochastic-multipole-scope-discharge"}

-   Theorem [\[thm:ito-wiener-filtration-langevin-endpoints\]](#thm:ito-wiener-filtration-langevin-endpoints){reference-type="ref" reference="thm:ito-wiener-filtration-langevin-endpoints"}

-   Theorem [\[thm:hamiltonian-mori-zwanzig-h-zero-limit-endpoints\]](#thm:hamiltonian-mori-zwanzig-h-zero-limit-endpoints){reference-type="ref" reference="thm:hamiltonian-mori-zwanzig-h-zero-limit-endpoints"}

-   Theorem [\[thm:forcefield-derived-realistic-sde-endpoints\]](#thm:forcefield-derived-realistic-sde-endpoints){reference-type="ref" reference="thm:forcefield-derived-realistic-sde-endpoints"}

-   Theorem [\[thm:generator-pde-estimate-canonical-regularity-closure\]](#thm:generator-pde-estimate-canonical-regularity-closure){reference-type="ref" reference="thm:generator-pde-estimate-canonical-regularity-closure"}

-   Theorem [\[thm:qm-workflow-transport-benchmark-summary\]](#thm:qm-workflow-transport-benchmark-summary){reference-type="ref" reference="thm:qm-workflow-transport-benchmark-summary"}

-   Theorem [\[thm:potential-landscape-barrier-kinetics-bundle\]](#thm:potential-landscape-barrier-kinetics-bundle){reference-type="ref" reference="thm:potential-landscape-barrier-kinetics-bundle"}

-   Theorem [\[thm:spectral-gap-trajectory-concentration-bundle\]](#thm:spectral-gap-trajectory-concentration-bundle){reference-type="ref" reference="thm:spectral-gap-trajectory-concentration-bundle"}

-   Theorem [\[thm:spectral-gap-absolute-free-energy-total-error\]](#thm:spectral-gap-absolute-free-energy-total-error){reference-type="ref" reference="thm:spectral-gap-absolute-free-energy-total-error"}

-   Theorem [\[thm:integrator-error-stack-unified-thermo-kinetic\]](#thm:integrator-error-stack-unified-thermo-kinetic){reference-type="ref" reference="thm:integrator-error-stack-unified-thermo-kinetic"}

-   Theorem [\[thm:learned-descriptor-ood-generalization-transfer\]](#thm:learned-descriptor-ood-generalization-transfer){reference-type="ref" reference="thm:learned-descriptor-ood-generalization-transfer"}

-   Theorem [\[thm:estimator-minimax-derivation-bundle\]](#thm:estimator-minimax-derivation-bundle){reference-type="ref" reference="thm:estimator-minimax-derivation-bundle"}

-   Theorem [\[thm:extended-physical-model-interface-scope-bundle\]](#thm:extended-physical-model-interface-scope-bundle){reference-type="ref" reference="thm:extended-physical-model-interface-scope-bundle"}

-   Theorem [\[thm:preregistered-prospective-beats-strong-baselines\]](#thm:preregistered-prospective-beats-strong-baselines){reference-type="ref" reference="thm:preregistered-prospective-beats-strong-baselines"}

-   Theorem [\[thm:independent-replication-outside-team-bundle\]](#thm:independent-replication-outside-team-bundle){reference-type="ref" reference="thm:independent-replication-outside-team-bundle"}

-   Theorem [\[thm:downstream-campaign-win-bundle\]](#thm:downstream-campaign-win-bundle){reference-type="ref" reference="thm:downstream-campaign-win-bundle"}

-   Theorem [\[thm:external-validation-threeway-integration-bundle\]](#thm:external-validation-threeway-integration-bundle){reference-type="ref" reference="thm:external-validation-threeway-integration-bundle"}

-   Theorem [\[thm:constructive-scope-closure-of-extension-interfaces\]](#thm:constructive-scope-closure-of-extension-interfaces){reference-type="ref" reference="thm:constructive-scope-closure-of-extension-interfaces"}

-   Theorem [\[thm:fixed-contract-preregistered-benchmark-bundle\]](#thm:fixed-contract-preregistered-benchmark-bundle){reference-type="ref" reference="thm:fixed-contract-preregistered-benchmark-bundle"}

-   Theorem [\[thm:independent-replication-provenance-bundle\]](#thm:independent-replication-provenance-bundle){reference-type="ref" reference="thm:independent-replication-provenance-bundle"}

-   Theorem [\[thm:downstream-causal-quality-bundle\]](#thm:downstream-causal-quality-bundle){reference-type="ref" reference="thm:downstream-causal-quality-bundle"}

-   Theorem [\[thm:concrete-external-validation-threeway-bundle\]](#thm:concrete-external-validation-threeway-bundle){reference-type="ref" reference="thm:concrete-external-validation-threeway-bundle"}

-   Theorem [\[thm:concrete-external-validation-not-credibly-dismissible\]](#thm:concrete-external-validation-not-credibly-dismissible){reference-type="ref" reference="thm:concrete-external-validation-not-credibly-dismissible"}

-   Theorem [\[thm:langevin-measure-theoretic-endpoint-bundle\]](#thm:langevin-measure-theoretic-endpoint-bundle){reference-type="ref" reference="thm:langevin-measure-theoretic-endpoint-bundle"}

-   Theorem [\[thm:constructive-ito-wiener-derived-endpoints\]](#thm:constructive-ito-wiener-derived-endpoints){reference-type="ref" reference="thm:constructive-ito-wiener-derived-endpoints"}

-   Theorem [\[thm:derived-generator-pde-operator-closure\]](#thm:derived-generator-pde-operator-closure){reference-type="ref" reference="thm:derived-generator-pde-operator-closure"}

-   Theorem [\[thm:microscopic-extension-interface-scope-bundle\]](#thm:microscopic-extension-interface-scope-bundle){reference-type="ref" reference="thm:microscopic-extension-interface-scope-bundle"}

-   Theorem [\[thm:numerical-stack-derived-simulator-control-flags\]](#thm:numerical-stack-derived-simulator-control-flags){reference-type="ref" reference="thm:numerical-stack-derived-simulator-control-flags"}

-   Theorem [\[thm:attested-concrete-external-validation-threeway-bundle\]](#thm:attested-concrete-external-validation-threeway-bundle){reference-type="ref" reference="thm:attested-concrete-external-validation-threeway-bundle"}

-   Theorem [\[thm:attested-concrete-external-validation-not-credibly-dismissible\]](#thm:attested-concrete-external-validation-not-credibly-dismissible){reference-type="ref" reference="thm:attested-concrete-external-validation-not-credibly-dismissible"}

-   Theorem [\[thm:attested-concrete-store-backed-artifact-bundle\]](#thm:attested-concrete-store-backed-artifact-bundle){reference-type="ref" reference="thm:attested-concrete-store-backed-artifact-bundle"}

-   Theorem [\[thm:store-backed-attested-concrete-external-validation-threeway-bundle\]](#thm:store-backed-attested-concrete-external-validation-threeway-bundle){reference-type="ref" reference="thm:store-backed-attested-concrete-external-validation-threeway-bundle"}

-   Theorem [\[thm:store-backed-attested-concrete-external-validation-not-credibly-dismissible\]](#thm:store-backed-attested-concrete-external-validation-not-credibly-dismissible){reference-type="ref" reference="thm:store-backed-attested-concrete-external-validation-not-credibly-dismissible"}

-   Theorem [\[thm:md-physical-utility-interface\]](#thm:md-physical-utility-interface){reference-type="ref" reference="thm:md-physical-utility-interface"}

-   Theorem [\[thm:md-native-coordinate-rank-identity\]](#thm:md-native-coordinate-rank-identity){reference-type="ref" reference="thm:md-native-coordinate-rank-identity"}

-   Theorem [\[thm:md-physical-3n-minus-k-budget\]](#thm:md-physical-3n-minus-k-budget){reference-type="ref" reference="thm:md-physical-3n-minus-k-budget"}

-   Theorem [\[thm:md-binary-summary-rank-monotonicity\]](#thm:md-binary-summary-rank-monotonicity){reference-type="ref" reference="thm:md-binary-summary-rank-monotonicity"}

-   Theorem [\[thm:equilibrium-kd-driving-energy-floor\]](#thm:equilibrium-kd-driving-energy-floor){reference-type="ref" reference="thm:equilibrium-kd-driving-energy-floor"}

-   Theorem [\[thm:md-detailed-balance-equilibrium-pathratio\]](#thm:md-detailed-balance-equilibrium-pathratio){reference-type="ref" reference="thm:md-detailed-balance-equilibrium-pathratio"}

-   Theorem [\[thm:md-resolver-free-equilibrium-bundle\]](#thm:md-resolver-free-equilibrium-bundle){reference-type="ref" reference="thm:md-resolver-free-equilibrium-bundle"}

-   Theorem [\[thm:md-equilibrium-kd-prediction-from-rank-lb\]](#thm:md-equilibrium-kd-prediction-from-rank-lb){reference-type="ref" reference="thm:md-equilibrium-kd-prediction-from-rank-lb"}

-   Theorem [\[thm:md-necessary-contact-shell-budget-from-rank-lb\]](#thm:md-necessary-contact-shell-budget-from-rank-lb){reference-type="ref" reference="thm:md-necessary-contact-shell-budget-from-rank-lb"}

-   Theorem [\[thm:md-independent-rank-risky-prediction-bundle\]](#thm:md-independent-rank-risky-prediction-bundle){reference-type="ref" reference="thm:md-independent-rank-risky-prediction-bundle"}

-   Theorem [\[thm:md-exact-lj-physics-witness-bundle\]](#thm:md-exact-lj-physics-witness-bundle){reference-type="ref" reference="thm:md-exact-lj-physics-witness-bundle"}

-   Theorem [\[thm:md-concrete-physics-witness-discharge-bundle\]](#thm:md-concrete-physics-witness-discharge-bundle){reference-type="ref" reference="thm:md-concrete-physics-witness-discharge-bundle"}

-   Theorem [\[thm:partition-ratio-driving-floor-with-correction-margin\]](#thm:partition-ratio-driving-floor-with-correction-margin){reference-type="ref" reference="thm:partition-ratio-driving-floor-with-correction-margin"}

-   Theorem [\[thm:equilibrium-kd-bound-from-partition-ratio-correction-chain\]](#thm:equilibrium-kd-bound-from-partition-ratio-correction-chain){reference-type="ref" reference="thm:equilibrium-kd-bound-from-partition-ratio-correction-chain"}

-   Theorem [\[thm:md-rank-kd-bound-from-partition-chain\]](#thm:md-rank-kd-bound-from-partition-chain){reference-type="ref" reference="thm:md-rank-kd-bound-from-partition-chain"}

-   Theorem [\[thm:md-independent-srank-interval-certificates\]](#thm:md-independent-srank-interval-certificates){reference-type="ref" reference="thm:md-independent-srank-interval-certificates"}

-   Theorem [\[thm:md-independent-certificate-risky-prediction-bundle\]](#thm:md-independent-certificate-risky-prediction-bundle){reference-type="ref" reference="thm:md-independent-certificate-risky-prediction-bundle"}

-   Theorem [\[thm:md-falsification-not-falsified-iff\]](#thm:md-falsification-not-falsified-iff){reference-type="ref" reference="thm:md-falsification-not-falsified-iff"}

-   Theorem [\[thm:md-preregistered-rank-protocol-soundness\]](#thm:md-preregistered-rank-protocol-soundness){reference-type="ref" reference="thm:md-preregistered-rank-protocol-soundness"}

-   Theorem [\[thm:finite-sample-upper-violation-implies-true-violation\]](#thm:finite-sample-upper-violation-implies-true-violation){reference-type="ref" reference="thm:finite-sample-upper-violation-implies-true-violation"}

-   Theorem [\[thm:md-high-confidence-fail-condition-bundle\]](#thm:md-high-confidence-fail-condition-bundle){reference-type="ref" reference="thm:md-high-confidence-fail-condition-bundle"}

-   Theorem [\[thm:kd-interval-from-driving-energy-error\]](#thm:kd-interval-from-driving-energy-error){reference-type="ref" reference="thm:kd-interval-from-driving-energy-error"}

-   Theorem [\[thm:kd-interval-from-absolute-free-energy-model\]](#thm:kd-interval-from-absolute-free-energy-model){reference-type="ref" reference="thm:kd-interval-from-absolute-free-energy-model"}

-   Theorem [\[thm:chemistry-data-backed-kd-interval-bundle\]](#thm:chemistry-data-backed-kd-interval-bundle){reference-type="ref" reference="thm:chemistry-data-backed-kd-interval-bundle"}

-   Theorem [\[thm:md-per-case-real-artifact-discharge-all-targets\]](#thm:md-per-case-real-artifact-discharge-all-targets){reference-type="ref" reference="thm:md-per-case-real-artifact-discharge-all-targets"}

-   Theorem [\[thm:partition-correction-numerical-kd-upper-bound-bundle\]](#thm:partition-correction-numerical-kd-upper-bound-bundle){reference-type="ref" reference="thm:partition-correction-numerical-kd-upper-bound-bundle"}

-   Theorem [\[thm:production-independent-srank-extractor-bundle\]](#thm:production-independent-srank-extractor-bundle){reference-type="ref" reference="thm:production-independent-srank-extractor-bundle"}

-   Theorem [\[thm:locked-prospective-falsification-run-soundness\]](#thm:locked-prospective-falsification-run-soundness){reference-type="ref" reference="thm:locked-prospective-falsification-run-soundness"}

-   Theorem [\[thm:assay-noise-high-confidence-call-validity-bundle\]](#thm:assay-noise-high-confidence-call-validity-bundle){reference-type="ref" reference="thm:assay-noise-high-confidence-call-validity-bundle"}

-   Theorem [\[thm:target-class-chemistry-completeness-kd-interval-bundle\]](#thm:target-class-chemistry-completeness-kd-interval-bundle){reference-type="ref" reference="thm:target-class-chemistry-completeness-kd-interval-bundle"}

-   Theorem [\[thm:single-target-full-physical-closure-instance-bundle\]](#thm:single-target-full-physical-closure-instance-bundle){reference-type="ref" reference="thm:single-target-full-physical-closure-instance-bundle"}

-   Theorem [\[thm:external-replication-at-scale-full-pipeline-bundle\]](#thm:external-replication-at-scale-full-pipeline-bundle){reference-type="ref" reference="thm:external-replication-at-scale-full-pipeline-bundle"}

-   Theorem [\[thm:single-state-partition-positive-bundle\]](#thm:single-state-partition-positive-bundle){reference-type="ref" reference="thm:single-state-partition-positive-bundle"}

-   Theorem [\[thm:zero-correction-calibration-bundle\]](#thm:zero-correction-calibration-bundle){reference-type="ref" reference="thm:zero-correction-calibration-bundle"}

-   Theorem [\[thm:upper-only-independent-srank-extractor-bundle\]](#thm:upper-only-independent-srank-extractor-bundle){reference-type="ref" reference="thm:upper-only-independent-srank-extractor-bundle"}

-   Theorem [\[thm:single-target-zero-rank-concrete-closure-bundle\]](#thm:single-target-zero-rank-concrete-closure-bundle){reference-type="ref" reference="thm:single-target-zero-rank-concrete-closure-bundle"}

-   Theorem [\[thm:attested-provenance-replication-at-scale-constructor-bundle\]](#thm:attested-provenance-replication-at-scale-constructor-bundle){reference-type="ref" reference="thm:attested-provenance-replication-at-scale-constructor-bundle"}

-   Theorem [\[thm:concrete-attested-single-target-replication-bundle\]](#thm:concrete-attested-single-target-replication-bundle){reference-type="ref" reference="thm:concrete-attested-single-target-replication-bundle"}

-   Theorem [\[thm:computable-finite-enumeration-pose-solver-optimal\]](#thm:computable-finite-enumeration-pose-solver-optimal){reference-type="ref" reference="thm:computable-finite-enumeration-pose-solver-optimal"}

-   Theorem [\[thm:rmsd-success-probability-unit-interval\]](#thm:rmsd-success-probability-unit-interval){reference-type="ref" reference="thm:rmsd-success-probability-unit-interval"}

-   Theorem [\[thm:topk-mass-lower-bound-rmsd-success-probability\]](#thm:topk-mass-lower-bound-rmsd-success-probability){reference-type="ref" reference="thm:topk-mass-lower-bound-rmsd-success-probability"}

-   Theorem [\[thm:rmsd-probability-derived-pose-solver-bundle\]](#thm:rmsd-probability-derived-pose-solver-bundle){reference-type="ref" reference="thm:rmsd-probability-derived-pose-solver-bundle"}

-   Theorem [\[thm:joint-computable-pose-rmsd-probability-bundle\]](#thm:joint-computable-pose-rmsd-probability-bundle){reference-type="ref" reference="thm:joint-computable-pose-rmsd-probability-bundle"}

-   Theorem [\[thm:raw-pocket-ligand-constructor-posterior-bundle\]](#thm:raw-pocket-ligand-constructor-posterior-bundle){reference-type="ref" reference="thm:raw-pocket-ligand-constructor-posterior-bundle"}

-   Theorem [\[thm:deployment-contract-implies-benchmark-contract\]](#thm:deployment-contract-implies-benchmark-contract){reference-type="ref" reference="thm:deployment-contract-implies-benchmark-contract"}

-   Theorem [\[thm:canonical-program-execution-refines-solver-result\]](#thm:canonical-program-execution-refines-solver-result){reference-type="ref" reference="thm:canonical-program-execution-refines-solver-result"}

-   Theorem [\[thm:raw-pocket-ligand-benchmark-solver-bundle\]](#thm:raw-pocket-ligand-benchmark-solver-bundle){reference-type="ref" reference="thm:raw-pocket-ligand-benchmark-solver-bundle"}

-   Theorem [\[thm:canonical-raw-benchmark-acceptance-equivalence\]](#thm:canonical-raw-benchmark-acceptance-equivalence){reference-type="ref" reference="thm:canonical-raw-benchmark-acceptance-equivalence"}

-   Theorem [\[thm:canonical-raw-deployment-acceptance-equivalence\]](#thm:canonical-raw-deployment-acceptance-equivalence){reference-type="ref" reference="thm:canonical-raw-deployment-acceptance-equivalence"}

-   Theorem [\[thm:canonical-raw-program-witness-refinement\]](#thm:canonical-raw-program-witness-refinement){reference-type="ref" reference="thm:canonical-raw-program-witness-refinement"}

-   Theorem [\[thm:canonical-raw-definitive-endpoint-bundle\]](#thm:canonical-raw-definitive-endpoint-bundle){reference-type="ref" reference="thm:canonical-raw-definitive-endpoint-bundle"}

-   Theorem [\[thm:canonical-runtime-output-refines-solver\]](#thm:canonical-runtime-output-refines-solver){reference-type="ref" reference="thm:canonical-runtime-output-refines-solver"}

-   Theorem [\[thm:definitive-raw-crossdock-accept-benchmark-iff\]](#thm:definitive-raw-crossdock-accept-benchmark-iff){reference-type="ref" reference="thm:definitive-raw-crossdock-accept-benchmark-iff"}

-   Theorem [\[thm:definitive-raw-crossdock-accept-deployment-iff\]](#thm:definitive-raw-crossdock-accept-deployment-iff){reference-type="ref" reference="thm:definitive-raw-crossdock-accept-deployment-iff"}

-   Theorem [\[thm:definitive-raw-crossdock-totality\]](#thm:definitive-raw-crossdock-totality){reference-type="ref" reference="thm:definitive-raw-crossdock-totality"}

-   Theorem [\[thm:definitive-raw-runtime-flag-refines-accept\]](#thm:definitive-raw-runtime-flag-refines-accept){reference-type="ref" reference="thm:definitive-raw-runtime-flag-refines-accept"}

-   Theorem [\[thm:definitive-raw-crossdock-full-closure-bundle\]](#thm:definitive-raw-crossdock-full-closure-bundle){reference-type="ref" reference="thm:definitive-raw-crossdock-full-closure-bundle"}

-   Theorem [\[thm:definitive-raw-benchmark-accepted-iff-contract\]](#thm:definitive-raw-benchmark-accepted-iff-contract){reference-type="ref" reference="thm:definitive-raw-benchmark-accepted-iff-contract"}

-   Theorem [\[thm:definitive-raw-deployment-rejected-iff-not-contract\]](#thm:definitive-raw-deployment-rejected-iff-not-contract){reference-type="ref" reference="thm:definitive-raw-deployment-rejected-iff-not-contract"}

-   Theorem [\[thm:definitive-accept-flag-iff-deployment-accepted\]](#thm:definitive-accept-flag-iff-deployment-accepted){reference-type="ref" reference="thm:definitive-accept-flag-iff-deployment-accepted"}

-   Theorem [\[thm:definitive-reject-flag-iff-deployment-rejected\]](#thm:definitive-reject-flag-iff-deployment-rejected){reference-type="ref" reference="thm:definitive-reject-flag-iff-deployment-rejected"}

-   Theorem [\[thm:computable-rational-accept-flag-exactness\]](#thm:computable-rational-accept-flag-exactness){reference-type="ref" reference="thm:computable-rational-accept-flag-exactness"}

-   Theorem [\[thm:computable-rational-accept-soundness\]](#thm:computable-rational-accept-soundness){reference-type="ref" reference="thm:computable-rational-accept-soundness"}

-   Theorem [\[thm:computable-rational-accept-refines-benchmark-accept\]](#thm:computable-rational-accept-refines-benchmark-accept){reference-type="ref" reference="thm:computable-rational-accept-refines-benchmark-accept"}

-   Theorem [\[thm:canonical-interpreter-state-runtime-refinement\]](#thm:canonical-interpreter-state-runtime-refinement){reference-type="ref" reference="thm:canonical-interpreter-state-runtime-refinement"}

-   Theorem [\[thm:definitive-interpreter-output-equals-runtime\]](#thm:definitive-interpreter-output-equals-runtime){reference-type="ref" reference="thm:definitive-interpreter-output-equals-runtime"}

-   Theorem [\[thm:definitive-report-runtime-accept-iff-deployment-accepted\]](#thm:definitive-report-runtime-accept-iff-deployment-accepted){reference-type="ref" reference="thm:definitive-report-runtime-accept-iff-deployment-accepted"}

-   Theorem [\[thm:definitive-raw-crossdock-complete-lean-bundle\]](#thm:definitive-raw-crossdock-complete-lean-bundle){reference-type="ref" reference="thm:definitive-raw-crossdock-complete-lean-bundle"}

-   Theorem [\[thm:definitive-report-runtime-reject-iff-deployment-rejected\]](#thm:definitive-report-runtime-reject-iff-deployment-rejected){reference-type="ref" reference="thm:definitive-report-runtime-reject-iff-deployment-rejected"}

-   Theorem [\[thm:definitive-constructive-benchmark-iff-kernel-flag\]](#thm:definitive-constructive-benchmark-iff-kernel-flag){reference-type="ref" reference="thm:definitive-constructive-benchmark-iff-kernel-flag"}

-   Theorem [\[thm:definitive-constructive-benchmark-refines-certificate-backend-benchmark\]](#thm:definitive-constructive-benchmark-refines-certificate-backend-benchmark){reference-type="ref" reference="thm:definitive-constructive-benchmark-refines-certificate-backend-benchmark"}

-   Theorem [\[thm:definitive-constructive-deployment-refines-certificate-backend-deployment\]](#thm:definitive-constructive-deployment-refines-certificate-backend-deployment){reference-type="ref" reference="thm:definitive-constructive-deployment-refines-certificate-backend-deployment"}

-   Theorem [\[thm:definitive-exact-rat-artifact-accept-iff-benchmark-contract\]](#thm:definitive-exact-rat-artifact-accept-iff-benchmark-contract){reference-type="ref" reference="thm:definitive-exact-rat-artifact-accept-iff-benchmark-contract"}

-   Theorem [\[thm:definitive-exact-rat-artifact-accept-refines-certificate-backend-accepts\]](#thm:definitive-exact-rat-artifact-accept-refines-certificate-backend-accepts){reference-type="ref" reference="thm:definitive-exact-rat-artifact-accept-refines-certificate-backend-accepts"}

-   Theorem [\[thm:definitive-benchmark-decision-alias-exactness\]](#thm:definitive-benchmark-decision-alias-exactness){reference-type="ref" reference="thm:definitive-benchmark-decision-alias-exactness"}

-   Theorem [\[thm:definitive-deployment-decision-alias-exactness\]](#thm:definitive-deployment-decision-alias-exactness){reference-type="ref" reference="thm:definitive-deployment-decision-alias-exactness"}

-   Theorem [\[thm:definitive-benchmark-decision-refines-certificate-backend-benchmark\]](#thm:definitive-benchmark-decision-refines-certificate-backend-benchmark){reference-type="ref" reference="thm:definitive-benchmark-decision-refines-certificate-backend-benchmark"}

-   Theorem [\[thm:definitive-decision-refines-certificate-backend-deployment\]](#thm:definitive-decision-refines-certificate-backend-deployment){reference-type="ref" reference="thm:definitive-decision-refines-certificate-backend-deployment"}

-   Theorem [\[thm:definitive-benchmark-decision-rejected-iff-kernel-flag-false\]](#thm:definitive-benchmark-decision-rejected-iff-kernel-flag-false){reference-type="ref" reference="thm:definitive-benchmark-decision-rejected-iff-kernel-flag-false"}

-   Theorem [\[thm:definitive-benchmark-certified-accepted-iff-decision-accepted\]](#thm:definitive-benchmark-certified-accepted-iff-decision-accepted){reference-type="ref" reference="thm:definitive-benchmark-certified-accepted-iff-decision-accepted"}

-   Theorem [\[thm:definitive-benchmark-certified-rejected-iff-decision-rejected\]](#thm:definitive-benchmark-certified-rejected-iff-decision-rejected){reference-type="ref" reference="thm:definitive-benchmark-certified-rejected-iff-decision-rejected"}

-   Theorem [\[thm:definitive-deployment-certified-accepted-iff-decision-accepted\]](#thm:definitive-deployment-certified-accepted-iff-decision-accepted){reference-type="ref" reference="thm:definitive-deployment-certified-accepted-iff-decision-accepted"}

-   Theorem [\[thm:definitive-deployment-certified-rejected-iff-decision-rejected\]](#thm:definitive-deployment-certified-rejected-iff-decision-rejected){reference-type="ref" reference="thm:definitive-deployment-certified-rejected-iff-decision-rejected"}

-   Theorem [\[thm:definitive-signed-rationalized-artifact-manifest-consistency\]](#thm:definitive-signed-rationalized-artifact-manifest-consistency){reference-type="ref" reference="thm:definitive-signed-rationalized-artifact-manifest-consistency"}

-   Theorem [\[thm:definitive-signed-rationalized-decision-accept-refines-certificate-backend-deployment\]](#thm:definitive-signed-rationalized-decision-accept-refines-certificate-backend-deployment){reference-type="ref" reference="thm:definitive-signed-rationalized-decision-accept-refines-certificate-backend-deployment"}

-   Theorem [\[thm:definitive-exact-rat-rejected-refines-certificate-backend-rejections\]](#thm:definitive-exact-rat-rejected-refines-certificate-backend-rejections){reference-type="ref" reference="thm:definitive-exact-rat-rejected-refines-certificate-backend-rejections"}

-   Theorem [\[thm:definitive-signed-exact-rat-accepted-iff-benchmark-contract\]](#thm:definitive-signed-exact-rat-accepted-iff-benchmark-contract){reference-type="ref" reference="thm:definitive-signed-exact-rat-accepted-iff-benchmark-contract"}

-   Theorem [\[thm:definitive-signed-exact-rat-rejected-refines-certificate-backend-rejections\]](#thm:definitive-signed-exact-rat-rejected-refines-certificate-backend-rejections){reference-type="ref" reference="thm:definitive-signed-exact-rat-rejected-refines-certificate-backend-rejections"}

-   Theorem [\[thm:definitive-signed-artifact-byte-envelope-roundtrip\]](#thm:definitive-signed-artifact-byte-envelope-roundtrip){reference-type="ref" reference="thm:definitive-signed-artifact-byte-envelope-roundtrip"}

-   Theorem [\[thm:definitive-concrete-checksum-byte-e2e\]](#thm:definitive-concrete-checksum-byte-e2e){reference-type="ref" reference="thm:definitive-concrete-checksum-byte-e2e"}

-   Theorem [\[thm:definitive-signed-rationalized-concrete-byte-parse-verify\]](#thm:definitive-signed-rationalized-concrete-byte-parse-verify){reference-type="ref" reference="thm:definitive-signed-rationalized-concrete-byte-parse-verify"}

-   Theorem [\[thm:definitive-rationalized-separation-not-benchmark-contract\]](#thm:definitive-rationalized-separation-not-benchmark-contract){reference-type="ref" reference="thm:definitive-rationalized-separation-not-benchmark-contract"}

-   Theorem [\[thm:definitive-rationalized-separation-flag-false\]](#thm:definitive-rationalized-separation-flag-false){reference-type="ref" reference="thm:definitive-rationalized-separation-flag-false"}

-   Theorem [\[thm:definitive-signed-rationalized-strict-rejection-refines-certificate-backend-rejections\]](#thm:definitive-signed-rationalized-strict-rejection-refines-certificate-backend-rejections){reference-type="ref" reference="thm:definitive-signed-rationalized-strict-rejection-refines-certificate-backend-rejections"}

-   Theorem [\[thm:definitive-concrete-checksum-verifier-exactness\]](#thm:definitive-concrete-checksum-verifier-exactness){reference-type="ref" reference="thm:definitive-concrete-checksum-verifier-exactness"}

-   Theorem [\[thm:definitive-runtime-ops-closed-form\]](#thm:definitive-runtime-ops-closed-form){reference-type="ref" reference="thm:definitive-runtime-ops-closed-form"}

-   Theorem [\[thm:definitive-runtime-ops-succ-recurrence\]](#thm:definitive-runtime-ops-succ-recurrence){reference-type="ref" reference="thm:definitive-runtime-ops-succ-recurrence"}

-   Theorem [\[thm:definitive-pipeline-total-ops-closed-form\]](#thm:definitive-pipeline-total-ops-closed-form){reference-type="ref" reference="thm:definitive-pipeline-total-ops-closed-form"}

-   Theorem [\[thm:definitive-branch-bound-prune-sound\]](#thm:definitive-branch-bound-prune-sound){reference-type="ref" reference="thm:definitive-branch-bound-prune-sound"}

-   Theorem [\[thm:definitive-adaptive-stop-sound\]](#thm:definitive-adaptive-stop-sound){reference-type="ref" reference="thm:definitive-adaptive-stop-sound"}

-   Theorem [\[thm:definitive-pipeline-branch-bound-prune-sound\]](#thm:definitive-pipeline-branch-bound-prune-sound){reference-type="ref" reference="thm:definitive-pipeline-branch-bound-prune-sound"}

-   Theorem [\[thm:definitive-batch-fusion-justified\]](#thm:definitive-batch-fusion-justified){reference-type="ref" reference="thm:definitive-batch-fusion-justified"}

-   Theorem [\[thm:definitive-parse-cost-linear-time\]](#thm:definitive-parse-cost-linear-time){reference-type="ref" reference="thm:definitive-parse-cost-linear-time"}

-   Theorem [\[thm:definitive-parse-encode-cost-exact\]](#thm:definitive-parse-encode-cost-exact){reference-type="ref" reference="thm:definitive-parse-encode-cost-exact"}

-   Theorem [\[thm:definitive-crypto-verifier-sound\]](#thm:definitive-crypto-verifier-sound){reference-type="ref" reference="thm:definitive-crypto-verifier-sound"}

-   Theorem [\[thm:definitive-signed-rationalized-crypto-byte-parse-verify\]](#thm:definitive-signed-rationalized-crypto-byte-parse-verify){reference-type="ref" reference="thm:definitive-signed-rationalized-crypto-byte-parse-verify"}

-   Theorem [\[thm:definitive-signed-pipeline-parser-bytes-exact\]](#thm:definitive-signed-pipeline-parser-bytes-exact){reference-type="ref" reference="thm:definitive-signed-pipeline-parser-bytes-exact"}

-   Theorem [\[thm:definitive-campaign-pair-evals-closed-form\]](#thm:definitive-campaign-pair-evals-closed-form){reference-type="ref" reference="thm:definitive-campaign-pair-evals-closed-form"}

-   Theorem [\[thm:definitive-campaign-pair-evals-succ-recurrence\]](#thm:definitive-campaign-pair-evals-succ-recurrence){reference-type="ref" reference="thm:definitive-campaign-pair-evals-succ-recurrence"}

-   Theorem [\[thm:definitive-pipeline-campaign-pair-evals-closed-form\]](#thm:definitive-pipeline-campaign-pair-evals-closed-form){reference-type="ref" reference="thm:definitive-pipeline-campaign-pair-evals-closed-form"}

-   Theorem [\[thm:definitive-pair-potential-fusion-justified\]](#thm:definitive-pair-potential-fusion-justified){reference-type="ref" reference="thm:definitive-pair-potential-fusion-justified"}

-   Theorem [\[thm:definitive-canonical-scorer-op-label-fusion-sound\]](#thm:definitive-canonical-scorer-op-label-fusion-sound){reference-type="ref" reference="thm:definitive-canonical-scorer-op-label-fusion-sound"}

-   Theorem [\[thm:prospective-empirical-closure\]](#thm:prospective-empirical-closure){reference-type="ref" reference="thm:prospective-empirical-closure"}

-   Theorem [\[thm:unified-physical-ood-prospective-bundle\]](#thm:unified-physical-ood-prospective-bundle){reference-type="ref" reference="thm:unified-physical-ood-prospective-bundle"}

-   Theorem [\[thm:paper4-witness-chain-import\]](#thm:paper4-witness-chain-import){reference-type="ref" reference="thm:paper4-witness-chain-import"}

-   Theorem [\[thm:paper4-interface-discharge-extensions\]](#thm:paper4-interface-discharge-extensions){reference-type="ref" reference="thm:paper4-interface-discharge-extensions"}

-   Theorem [\[thm:paper4-stochastic-relevance-conjecture-full-support\]](#thm:paper4-stochastic-relevance-conjecture-full-support){reference-type="ref" reference="thm:paper4-stochastic-relevance-conjecture-full-support"}

-   Theorem [\[thm:paper4-stochastic-relevance-general-distribution-progress\]](#thm:paper4-stochastic-relevance-general-distribution-progress){reference-type="ref" reference="thm:paper4-stochastic-relevance-general-distribution-progress"}

-   Theorem [\[thm:paper4-stochastic-relevance-conjecture-nonneg-support-transport\]](#thm:paper4-stochastic-relevance-conjecture-nonneg-support-transport){reference-type="ref" reference="thm:paper4-stochastic-relevance-conjecture-nonneg-support-transport"}

-   Theorem [\[thm:paper4-stochastic-relevance-conjecture-nonneg-primitive-dynamics\]](#thm:paper4-stochastic-relevance-conjecture-nonneg-primitive-dynamics){reference-type="ref" reference="thm:paper4-stochastic-relevance-conjecture-nonneg-primitive-dynamics"}

-   Theorem [\[thm:paper4-stochastic-relevance-conjecture-nonneg-explicit-step-dynamics\]](#thm:paper4-stochastic-relevance-conjecture-nonneg-explicit-step-dynamics){reference-type="ref" reference="thm:paper4-stochastic-relevance-conjecture-nonneg-explicit-step-dynamics"}

-   Theorem [\[thm:paper4-stochastic-relevance-support-transport-of-explicit-step-dynamics\]](#thm:paper4-stochastic-relevance-support-transport-of-explicit-step-dynamics){reference-type="ref" reference="thm:paper4-stochastic-relevance-support-transport-of-explicit-step-dynamics"}

-   Theorem [\[thm:realism-augmented-forcefield-transport\]](#thm:realism-augmented-forcefield-transport){reference-type="ref" reference="thm:realism-augmented-forcefield-transport"}

-   Theorem [\[thm:ewald-long-range-certificates\]](#thm:ewald-long-range-certificates){reference-type="ref" reference="thm:ewald-long-range-certificates"}

-   Theorem [\[thm:langevin-infinite-dimensional-path-measure-bridge\]](#thm:langevin-infinite-dimensional-path-measure-bridge){reference-type="ref" reference="thm:langevin-infinite-dimensional-path-measure-bridge"}

-   Theorem [\[thm:concrete-docking-kinetic-bundle-specialization\]](#thm:concrete-docking-kinetic-bundle-specialization){reference-type="ref" reference="thm:concrete-docking-kinetic-bundle-specialization"}

-   Theorem [\[thm:jarzynski-from-crooks\]](#thm:jarzynski-from-crooks){reference-type="ref" reference="thm:jarzynski-from-crooks"}

-   Theorem [\[thm:quotient-trajectory-dissipation\]](#thm:quotient-trajectory-dissipation){reference-type="ref" reference="thm:quotient-trajectory-dissipation"}

-   Theorem [\[thm:error-correction-srank-overhead\]](#thm:error-correction-srank-overhead){reference-type="ref" reference="thm:error-correction-srank-overhead"}

-   Theorem [\[thm:fault-tolerant-landauer-floor\]](#thm:fault-tolerant-landauer-floor){reference-type="ref" reference="thm:fault-tolerant-landauer-floor"}

-   Theorem [\[thm:hopfield-ninio-proofreading-overhead\]](#thm:hopfield-ninio-proofreading-overhead){reference-type="ref" reference="thm:hopfield-ninio-proofreading-overhead"}

-   Theorem [\[thm:hopfield-ninio-kinetic-branch\]](#thm:hopfield-ninio-kinetic-branch){reference-type="ref" reference="thm:hopfield-ninio-kinetic-branch"}

-   Theorem [\[thm:binding-free-energy-floor\]](#thm:binding-free-energy-floor){reference-type="ref" reference="thm:binding-free-energy-floor"}

-   Theorem [\[thm:binding-free-energy-tightness\]](#thm:binding-free-energy-tightness){reference-type="ref" reference="thm:binding-free-energy-tightness"}

-   Corollary [\[cor:logical-basement-overhead\]](#cor:logical-basement-overhead){reference-type="ref" reference="cor:logical-basement-overhead"}

-   Theorem [\[thm:decision-quotient-potential\]](#thm:decision-quotient-potential){reference-type="ref" reference="thm:decision-quotient-potential"}

-   Proposition [\[prop:threshold-channel\]](#prop:threshold-channel){reference-type="ref" reference="prop:threshold-channel"}

-   Proposition [\[prop:atomic-realization\]](#prop:atomic-realization){reference-type="ref" reference="prop:atomic-realization"}

-   Theorem [\[thm:spinhalf-concrete-quantum-instantiation\]](#thm:spinhalf-concrete-quantum-instantiation){reference-type="ref" reference="thm:spinhalf-concrete-quantum-instantiation"}

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

-   Theorem [\[thm:geometric-constraint-decision-interface\]](#thm:geometric-constraint-decision-interface){reference-type="ref" reference="thm:geometric-constraint-decision-interface"}

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
- Lines: 117726
- Theorems: 4744
- `sorry` placeholders: 0
