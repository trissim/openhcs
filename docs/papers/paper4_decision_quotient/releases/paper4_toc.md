# Paper: Computational Complexity of Physical Counting

**Status**: Theory of Computing-ready | **Lean**: 33311 lines, 1378 theorems

---

## Abstract

**Abstract**

Which coordinates of a system's state determine the optimal action? For decision problem $\mathcal{D}=(A,S,U)$ with $S=X_1\times\cdots\times X_n$, set $I$ is sufficient if $s_I=s'_I\Rightarrow\operatorname{Opt}(s)=\operatorname{Opt}(s')$. The optimizer quotient $Q=S/{\sim}$ is the coarsest abstraction preserving optimal actions.

From counting measure we derive probability, Bayes' theorem, and Bayesian optimality (from $\log x\leq x-1$ alone). The invariant $\mathrm{srank}$ (relevant-coordinate cardinality) emerges as the complexity measure.

Complexity: SUFFICIENCY-CHECK and MINIMUM-SUFFICIENT-SET are coNP-complete; ANCHOR-SUFFICIENCY is $\Sigma_2^P$-complete; stochastic/sequential variants PP-/PSPACE-complete. Six subcases admit polynomial algorithms. Under ETH, $2^{\Omega(n)}$ lower bounds apply.

Thermodynamic results use Landauer's floor $k_BT\ln 2$ as a declared premise; mismatch/residual dissipation terms yield $dU\geq\lambda\,dC$. Reduction correctness is machine-checked in Lean 4; polytime computability is a standard hypothesis.


# Introduction {#sec:introduction}

## First-Principles Derivations {#sec:intro-first-principles}

This paper studies a simple question with a hard combinatorial core: when a system exposes many state coordinates, which of those coordinates are actually needed to preserve the optimal decision? The aim is not to list all observable state variables, but to isolate the smallest part of the state that still determines what the best action is.

We model this with a decision problem: $A$ is the finite set of possible actions, $S$ is the state space, $U(a,s)$ is the utility of taking action $a$ in state $s$, and $\operatorname{Opt}(s)$ is the set of actions that maximize utility at state $s$. A coordinate set $I$ is sufficient when any two states that agree on $I$ also agree on $\operatorname{Opt}$. The optimizer quotient object then collapses exactly those states with the same optimal-action set, yielding the canonical abstraction through which every decision-preserving abstraction factors.

Section [\[sec:foundations\]](#sec:foundations){reference-type="ref" reference="sec:foundations"} makes that collapse boundary explicit: any surjective abstraction either factors through the optimizer quotient object or erases a decision-relevant distinction, and any attempt to treat such extra erasure as physically feasible runs into the same no-collapse obstruction used later in the hardness chain. In **Set**, this quotient object is the familiar coimage of $\operatorname{Opt}$, canonically equivalent to $\operatorname{range}(\operatorname{Opt})$.QT5AB2AB3AB4

The results in this section are organized as one derivation chain rather than a theorem ledger. The opening results establish the structural core from counting, set theory, arithmetic, and logic; later results then place the same core in complexity-theoretic and physical terms. Theorems 1--14 are derived from pure mathematics, while Theorems 15--17 add explicit empirical lower-bound premises. One irreducible empirical claim (EC1) supplies the thermodynamic scale: irreversible bit operations carry a positive lower-bound cost, with Landauer furnishing a universal floor. Counting and logic provide the structural part of the chain. Two former claims (EC2: finite resources, EC3: finite speed) are now derived from first principles.

### 1. Counting Gap Theorem

We begin with the discrete counting fact that every later information and complexity bound depends on: bounded total capacity and positive integer cost imply a finite observation budget.

::: theorem
[]{#thm:counting-gap-intro label="thm:counting-gap-intro"} Let $\varepsilon, C \in \mathbb{N}$ with $\varepsilon > 0$ and $C > 0$. If each information-acquisition event consumes $\varepsilon$ cost units and the total available capacity is $C$, then for every $N \in \mathbb{N}$: $$\varepsilon \cdot N \leq C \;\Rightarrow\; N \leq C.$$ Equivalently, $c_{\max}=C$ is a finite upper bound on the number of possible events. No bounded system with positive integer event cost can perform infinitely many events. BA10
:::

::: proof
*Proof.* This is a discrete theorem: $\varepsilon > 0$ in $\mathbb{N}$ implies $1 \leq \varepsilon$, so each event costs at least one unit. Therefore $N = 1\cdot N \leq \varepsilon \cdot N \leq C$. ◻
:::

**Consequence.** Continuous abstractions require arbitrarily fine state discrimination. Bounded computational systems cannot do this. Discrete models are forced by the mathematics.

### 2. Bayesian Optimality

Once observations are finite, the next question is how to update beliefs optimally from those observations.

::: theorem
[]{#thm:bayes-optimal label="thm:bayes-optimal"} For distributions $p, q$ over hypothesis space $H$, cross-entropy satisfies: $$\mathrm{CE}(p, q) = H(p) + D_{\mathrm{KL}}(p \| q)$$ where $D_{\mathrm{KL}}(p \| q) \geq 0$ (Gibbs' inequality, from $\log x \leq x - 1$). Therefore $q = p$ uniquely minimizes $\mathrm{CE}(p, q)$. FN7FN12FN14
:::

::: proof
*Proof.* Gibbs' inequality: $\log x \leq x - 1$ implies $\sum p \log(p/q) \geq 0$. Equality iff $p = q$. ◻
:::

**Consequence.** Bayesian updating is optimal. The proof uses only $\log x \leq x - 1$ and Lean's type theory.

### 3. Measure Theory Typing

To state those updates correctly, we next distinguish the type of a quantitative claim from the stronger normalization required for a probabilistic claim.

::: theorem
[]{#thm:measure-prerequisite label="thm:measure-prerequisite"}

1.  Every quantitative claim carries an explicit measure: $(\mu, f)$ where $f$ is measurable.

2.  Every stochastic claim additionally requires $\mu$ to be a probability measure ($\mu(\Omega) = 1$).

3.  Counting measure on $\{0,1\}$ has mass $2$, not $1$: not a probability measure.

MN1MN2MN10MN11
:::

::: proof
*Proof.* (1) "$\mathbb{E}[f]$" is undefined without specifying which measure integrates $f$. The expression $\int f\,d\mu$ requires $\mu$. (2) $P(A) = \mu(A)/\mu(\Omega)$ requires $\mu(\Omega) < \infty$; normalization $P(\Omega) = 1$ requires $\mu(\Omega) = 1$. (3) Counting measure: $\mu(\{0\}) = 1$, $\mu(\{1\}) = 1$, so $\mu(\{0,1\}) = 2 \neq 1$. To normalize, divide by 2. ◻
:::

**Consequence.** Quantitative and stochastic claims are typed differently. Probability normalization is not automatic from measure structure.

### 4. Universal Property of the Optimizer Quotient

With the measurement layer typed, we can identify the canonical abstraction that preserves the decision boundary itself.

::: theorem
[]{#thm:quotient-universal label="thm:quotient-universal"} Let $Q = S/{\sim}$ be the optimizer quotient object where $s \sim s' \iff \operatorname{Opt}(s) = \operatorname{Opt}(s')$. For any surjective abstraction $\phi: S \to T$ that preserves the optimal action ($\phi(s) = \phi(s') \Rightarrow \operatorname{Opt}(s) = \operatorname{Opt}(s')$), there exists a unique factorization $\psi: T \to Q$ such that $\pi = \psi \circ \phi$ where $\pi: S \to Q$ is the quotient map. $$\begin{array}{ccc}
    S & \xrightarrow{\pi} & Q \\
    \downarrow & & \uparrow \\
    T & \xrightarrow{\psi} & Q
  \end{array}$$ The optimizer quotient object is the coarsest abstraction through which $\operatorname{Opt}$ factors. In **Set**, it is canonically equivalent to $\operatorname{range}(\operatorname{Opt})$, i.e. the familiar coimage/image factorization of the optimizer map. QT1QT2QT3QT5QT7
:::

::: proof
*Proof.* Since $\phi$ preserves $\operatorname{Opt}$, we have $\phi(s) = \phi(s') \Rightarrow \operatorname{Opt}(s) = \operatorname{Opt}(s')$. By surjectivity, for each $t \in T$ pick some $s$ with $\phi(s) = t$ and define $\psi(t) = \pi(s)$. This is well-defined: if $\phi(s) = \phi(s') = t$, then $\operatorname{Opt}(s) = \operatorname{Opt}(s')$, so $\pi(s) = \pi(s')$. Uniqueness follows because $\pi = \psi \circ \phi$ forces $\psi(t) = \pi(s)$ for any $s$ with $\phi(s) = t$. ◻
:::

**Consequence.** The optimizer quotient object is canonical. In **Set**, the familiar construction is not exotic: it is the coimage of $\operatorname{Opt}$, canonically identified with the image/range of $\operatorname{Opt}$. Any other state abstraction that preserves optimal actions must refine it.

### 5. Witness-Checking Duality

Once the canonical abstraction is fixed, we can ask how many state comparisons any sound verifier must inspect to certify that no counterexample exists.

::: theorem
[]{#thm:checking-duality label="thm:checking-duality"} For boolean decision problems with $n$ coordinates, let $W(n) = 2^{n-1}$ be the witness budget and $T(n)$ be the checking budget. Any sound checker satisfies: $$T(n) \geq W(n) = 2^{n-1}$$ No finite checker with fewer than $2^{n-1}$ witness pairs can be sound on all problem instances. WD1WD2WD3
:::

::: proof
*Proof.* $\emptyset$-sufficiency fails iff $\exists s, s'$ with $\operatorname{Opt}(s) \neq \operatorname{Opt}(s')$. There are $2^n$ states, so $\binom{2^n}{2} = 2^{n-1}(2^n - 1)$ pairs. A sound refutation must cover all pairs that could witness failure. For each pair family $P$ with $|P| < 2^{n-1}$, there exists an $\operatorname{Opt}$ function whose unique failure pair is not in $P$. Therefore $|P| \geq 2^{n-1}$ is necessary for soundness. ◻
:::

**Consequence.** Verification has a lower bound. A checker must examine exponentially many witness pairs to be sound. This is a counting lower bound on the information required for verification.

### 6. Bayes from Counting

That verification bound rests on a simpler structural fact: probability, Bayes, and the normalized quotient score already emerge from counting on finite sets.

::: theorem
[]{#thm:bayes-from-counting label="thm:bayes-from-counting"} The chain from counting to Bayesian updating and the normalized quotient score:

1.  $P(A) = |A|/|\Omega|$ satisfies Kolmogorov axioms: nonnegativity, normalization, and additivity for disjoint sets.

2.  $P(H|E) = P(E|H) \cdot P(H) / P(E)$ follows from the definition of conditional probability.

3.  Conditioning reduces entropy: $H(H|E) \leq H(H)$, from nonnegativity of KL divergence.

4.  $\DQ = I(H;E)/H(H) = 1 - H(H|E)/H(H)$ lies in $[0,1]$.

BC1BC2BC3BC4BC5
:::

::: proof
*Proof.* (1) $|A| \geq 0$ (counting), $|\Omega|/|\Omega| = 1$ (normalization), $|A \cup B| = |A| + |B|$ for $A \cap B = \emptyset$ (additivity). (2) $P(H|E) := P(H \cap E)/P(E)$. Rearranging: $P(H|E) \cdot P(E) = P(H \cap E) = P(E|H) \cdot P(H)$. (3) $H(H|E) = H(H) - I(H;E)$ and $I(H;E) = D_{\mathrm{KL}}(P_{H,E} \| P_H \otimes P_E) \geq 0$ by Gibbs. (4) $I(H;E) \leq H(H)$ by (3), so $\DQ = I(H;E)/H(H) \in [0,1]$. ◻
:::

**Consequence.** Bayesian updating is derived from counting measure. Probability theory, Bayes' theorem, information theory, and the normalized quotient score form a single chain from counting elements of a finite set.

### 7. Fisher Rank = Structural Rank

With that probabilistic chain in place, the next results compare independent mathematical formalisms and ask whether they recover the same structural invariant. In the finite-coordinate model, $\mathrm{srank}$ is just the cardinality of the relevant-coordinate support, so the question is whether those formalisms recover that same support-size quantity.SK1SK2SK3

::: theorem
[]{#thm:fisher-rank-srank label="thm:fisher-rank-srank"} Define the Fisher information score of coordinate $i$ as: $$\mathrm{score}(i) = \begin{cases} 1 & \text{if } i \text{ is relevant} \\ 0 & \text{otherwise} \end{cases}$$ Then: $$\sum_{i=1}^{n} \mathrm{score}(i) = \mathrm{srank}(\mathcal{D})$$ The diagonal Fisher information matrix $I(\theta)_{ii} = \mathrm{score}(i)$ has rank $\mathrm{srank}(\mathcal{D})$. FS1FS2
:::

::: proof
*Proof.* $\sum_i \mathrm{score}(i) = \sum_i \mathbf{1}[\text{isRelevant}(i)] = |\{i \mid \text{isRelevant}(i)\}| = \mathrm{srank}$. For the diagonal matrix: $\mathrm{rank} = |\{i \mid \mathrm{score}(i) \neq 0\}| = |\{i \mid \text{isRelevant}(i)\}| = \mathrm{srank}$. ◻
:::

**Consequence.** The decision manifold has dimension exactly the size of the relevant-coordinate support. The Fisher information sees the same coordinates already counted by $\mathrm{srank}$. The Cramer-Rao bound follows: any estimator has difficulty $\geq 1/\mathrm{srank}$.

### 8. Entropy-Rank Inequality

The first bridge is information-theoretic: quotient entropy cannot exceed the number of coordinates that actually matter.

::: theorem
[]{#thm:entropy-rank label="thm:entropy-rank"} For a decision problem $\mathcal{D}$ with boolean coordinate spaces ($X_i = \{0,1\}$): $$H(\mathcal{D}) \leq \mathrm{srank}(\mathcal{D})$$ where $H(\mathcal{D}) = \log_2(\mathrm{numOptClasses})$ is the Shannon entropy of the optimizer quotient object. IT3IT4
:::

::: proof
*Proof.* The relevant coordinate set $R$ is sufficient (Theorem [\[thm:resolution-sufficient\]](#thm:resolution-sufficient){reference-type="ref" reference="thm:resolution-sufficient"}): $\mathrm{Opt}$ factors through $\pi_R : S \to \{0,1\}^{|R|}$. Therefore $\mathrm{numOptClasses} \leq |\{0,1\}^{|R|}| = 2^{\mathrm{srank}}$. Therefore $H(\mathcal{D}) = \log_2(\mathrm{numOptClasses}) \leq \log_2(2^{\mathrm{srank}}) = \mathrm{srank}$. ◻
:::

**Consequence.** The optimizer quotient object carries at most $\mathrm{srank}$ bits of decision-relevant information. Coordinates outside the relevant set are information-theoretically inert.

### 9--14. Later Bridge Theorems

The next six introductory results are bridge theorems. They are proved in their dedicated sections, but we state them here in compressed form because they all serve the same purpose: independent frameworks recover the same structural core once the optimizer quotient object is fixed.

::: theorem
[]{#thm:fi-coincide label="thm:fi-coincide"} For functional set $F \subseteq \Omega$, the counting expression $-\log_2(|F|/|\Omega|)$ gives the bit cost, and a Landauer floor converts that bit count into a universal energy lower bound. In the sharpened physical interface, explicit nonnegative mismatch and residual terms are added above that floor to represent constrained-process dissipation. The mismatch branch is now justified as far as the existing mathematics allows: distinct strictly positive finite input distributions yield strictly positive KL mismatch, which is then rounded upward into the discrete lower-bound units used by the thermodynamic accounting and injected into the Wolpert decomposition. A second derived branch now captures the strongest residual theorem supported by the current finite machinery: for finite discrete computational-state processes, positive forward edge flow together with decision-relevant edge asymmetry yields strict residual positivity by an exhaustive local split on the reverse edge. If the reverse flow is positive, the pairwise KL branch applies. If the reverse flow vanishes, the process performs an irreversible one-way state transition, and the existing Landauer-scaled transition-cost theorem supplies a positive lower-bound witness. That local argument is now also packaged as a process-level theorem: any finite computational-state process with at least one such asymmetric forward edge already carries a theorem-level residual witness. The broader stopping-time / absolute-irreversibility residual regime remains a separate imported premise. The thermodynamic statement is therefore the counting statement plus a floor-plus-overhead empirical conversion law; a derived mismatch branch, a derived finite residual branch from this exhaustive edge split, or the cited broader stopping-time branch can each force strict separation above the Landauer floor, and any declared structural lower bound absorbed by mismatch only strengthens the resulting energy estimate. FI3FI6FI7WC1WC2WC3WM1WM3WM4WM5WM6WR1WR2WR3WR4WR5WR6WR7WR8WR9WR10WP1WP5WP6WP7WP8
:::

::: proof
*Proof.* The mathematical part is only the counting step: finite measure gives the bit count $-\log_2(|F|/|\Omega|)$. Landauer then supplies the universal floor for the energy-per-bit conversion. The bridge is therefore structural counting plus one declared empirical lower-bound premise. ◻
:::

**Consequence.** The thermodynamic lower bound is not a separate combinatorial structure; it is the counting bound expressed in physical units.

::: theorem
[]{#thm:wasserstein-bridge label="thm:wasserstein-bridge"} Optimal-transport cost grows with the number of distinguishable futures and therefore tracks the same relevant-coordinate complexity measured by $\mathrm{srank}$. W1W2W3W4
:::

::: proof
*Proof.* When all states lead to one future, the diagonal coupling gives zero transport cost. Once multiple futures must be distinguished, off-diagonal mass incurs positive cost. The same branching structure counted by $\mathrm{srank}$ therefore reappears as transport cost. ◻
:::

**Consequence.** Optimal transport recovers the same complexity signal as the coordinate-relevance analysis.

::: theorem
[]{#thm:rate-distortion-bridge label="thm:rate-distortion-bridge"} At zero distortion, the minimum lossless rate needed to preserve optimal actions is exactly $\mathrm{srank}$, i.e. exactly the cardinality of the relevant-coordinate support, and compression below that support size necessarily introduces decision errors. RD1RD2RD3RS1RS2RS3RS4RS5
:::

::: proof
*Proof.* Zero distortion means that decision-equivalent states may be merged, but decision-distinguishing states may not. The quotient therefore partitions the source into exactly the classes that must remain distinguishable. Their count determines the minimum lossless rate, which is the same structural quantity measured by $\mathrm{srank}$. ◻
:::

**Consequence.** Description length and relevant-coordinate support size coincide at zero distortion.

::: theorem
[]{#thm:nontriviality-counting label="thm:nontriviality-counting"} If decision-relevant information exists, then the state space cannot be trivial: the existence of information forces at least two distinguishable states. FP1FP2FP3FP4FP5FP6FP7
:::

::: proof
*Proof.* If the state space had at most one element, then every map out of it would be constant, including $\operatorname{Opt}$. A constant decision boundary has zero entropy and carries no distinguishable information. Taking the contrapositive gives the stated nontriviality requirement. ◻
:::

**Consequence.** Nontriviality is derived from the existence of information rather than postulated as an extra axiom.

::: theorem
[]{#thm:second-law-counting label="thm:second-law-counting"} Maintaining ordered behavior requires continuous distinction between correct and incorrect transitions; in this sense, the structural content of the Second Law follows from counting and verification, while the numerical thermodynamic scale is added later. FP8FP9FP10FP11FP12FP13FP14
:::

::: proof
*Proof.* If there are more non-target than target continuations, unguided evolution overwhelmingly drifts away from the target set. Staying ordered therefore requires repeated discrimination between correct and incorrect continuations. That discrimination is an information-acquisition burden, and the thermodynamic lower-bound premise enters only after that structural burden is fixed. ◻
:::

**Consequence.** The order-versus-verification tradeoff is established before any temperature-dependent conversion is introduced.

::: theorem
[]{#thm:landauer-structure label="thm:landauer-structure"} Whenever a state reduction deletes distinctions, those distinctions must reappear in environmental degrees of freedom, so the structural content of Landauer's principle is a transfer statement and only the numerical thermodynamic scale is empirical. FP15
:::

::: proof
*Proof.* Reducing $n$ distinguishable states to fewer than $n$ outputs removes distinctions from the local description. Those distinctions cannot vanish from the bookkeeping; they must be displaced into the environment. The structural theorem is therefore a conservation-of-distinguishability statement, with temperature only setting the energetic scale of that transfer. ◻
:::

**Consequence.** Landauer's principle separates cleanly into a mathematical transfer structure and an empirical lower-bound energy scale.

## Irreducible Empirical Claims {#sec:irreducible-empirical}

One claim is irreducibly empirical. Two others (finite resources, finite speed) are derived from first principles.

::: definition
[]{#def:empirical-claims label="def:empirical-claims"} **EC1 (Temperature):** Entropy has energy cost. There exists $\varepsilon > 0$ such that each irreversible bit operation costs at least $\varepsilon$ joules, with $\varepsilon \geq k_B T \ln 2$. Landauer gives the universal floor. A separate floor-plus-overhead interface treats any strictly larger per-bit dissipation as an explicit additional premise rather than an informal interpretation.
:::

::: theorem
[]{#thm:ec2-derived label="thm:ec2-derived"} Finite resources (EC2) follows from the existence of uncertainty:

1.  Uncertainty exists: $\exists X, 0 < P(X) < 1$ (observation).

2.  No bounds $\Rightarrow$ infinite information acquisition $\Rightarrow$ know everything $\Rightarrow$ $P(X) \in \{0,1\}$ for all $X$.

3.  Contrapositive: uncertainty exists $\Rightarrow$ bounds exist.

IA7IA9IA11IA12IA13
:::

::: proof
*Proof.* Let observer have capacity $C$ and access $A \leq C$. System has entropy $H$. If $C \geq H$, observer can access all information (IA8). Full access eliminates uncertainty (IA9). Contrapositive: uncertainty implies $A < H$. If observer uses full capacity ($A = C$), then $C < H$ (IA11). Therefore capacity is bounded relative to system entropy. This is EC2. ◻
:::

::: theorem
[]{#thm:ec3-derived label="thm:ec3-derived"} Finite signal speed (EC3) follows from the law of non-contradiction:

1.  A computational step = state change: before $\neq$ after (definition).

2.  At any instant, state is unique (function property).

3.  If step takes $t = 0$: before-state and after-state at same instant.

4.  Same instant $\Rightarrow$ state = before AND state = after.

5.  But before $\neq$ after $\Rightarrow$ contradiction (violates non-contradiction).

6.  Therefore $t_{\text{before}} \neq t_{\text{after}}$, so $\Delta t > 0$.

7.  Information propagation = sequence of steps, each taking $\Delta t > 0$.

8.  Speed = distance / time. Time $> 0$ $\Rightarrow$ speed $< \infty$.

FPT4FPT5FPT6FPT8FPT10
:::

::: proof
*Proof.* A step changes state from $A$ to $B$ where $A \neq B$. The timeline is a function $T \to S$. If $t_{\text{before}} = t_{\text{after}}$, then $\text{timeline}(t) = A$ and $\text{timeline}(t) = B$ for the same $t$. By function uniqueness, $A = B$. Contradiction. Therefore $t_{\text{before}} \neq t_{\text{after}}$. With ordering, $\Delta t > 0$ (FPT5). Information propagation requires steps; each takes positive time; total time is positive; speed is finite. QED from pure logic. ◻
:::

**Consequence.** EC2 and EC3 are not axioms. EC2 follows from uncertainty existing. EC3 follows from non-contradiction. **One irreducible empirical claim remains: EC1 (temperature).** The empirical content is that irreversible bit operations have a positive lower-bound cost, with Landauer giving a universal floor and stronger constrained-process bounds allowed above it; the existence of a cost is structural.

### 15. Assumption Necessity

::: theorem
[]{#thm:assumption-necessity label="thm:assumption-necessity"} If a physical claim $\phi$ is:

1.  Provable from assumption set $\Gamma$ (sound formal system)

2.  Falsifiable (some model $m$ with $\neg \mathrm{Holds}(m, \phi)$)

3.  Derived from empirically disciplined $\Gamma$ (all axioms empirically justified)

Then $\Gamma$ contains at least one physical axiom that is empirically justified. AN3AN4
:::

::: proof
*Proof.* Suppose $\Gamma$ contains only mathematical axioms (no physical axioms). Mathematical axioms hold in all models (by soundness). Therefore $\phi$ holds in all models. But (2) says $\phi$ fails in some model $m$. Contradiction. Therefore $\Gamma$ contains at least one non-mathematical axiom. By (3), this axiom must be empirically justified. ◻
:::

**Consequence.** Physical axioms cannot be hidden. Physical consequences require declared and empirically justified physical assumptions.

### 16. Energy Bound

::: theorem
[]{#thm:energy-information label="thm:energy-information"} For any physical system resolving decision problem $\mathcal{D}$ at temperature $T > 0$: $$E_{\mathrm{decision}} \geq k_B T \cdot H_{\mathrm{nats}}(\mathcal{D})$$ where $H_{\mathrm{nats}}(\mathcal{D}) = \ln(\mathrm{numOptClasses})$ is the natural-log Shannon entropy of the optimizer quotient object. EI1
:::

::: proof
*Proof.* Three steps:

1.  $\mathrm{srank}$ bits are required to resolve $\mathcal{D}$. Each bit requires one state transition.

2.  Each transition costs at least the Landauer floor $k_B T \ln 2$. In the stronger constrained-process model, explicit nonnegative mismatch and residual dissipation terms may raise that per-bit lower bound further. The mismatch branch is no longer merely postulated: if the actual and designed finite distributions differ, the existing KL machinery yields a strictly positive mismatch witness, which is then converted into the discrete lower-bound units of the thermodynamic model. A second finite residual branch is also now derived by an exhaustive edge split on finite computational-state processes: if the reverse edge flow is positive, asymmetric forward/reverse edge flows yield strictly positive two-point KL divergence; if the reverse edge flow is zero, the process performs an irreversible one-way state transition and the existing Landauer-scaled transition-cost theorem yields a positive discrete lower-bound term. The broader stopping-time branch remains an imported premise. Therefore $E \geq \mathrm{srank} \cdot k_B T \ln 2$ remains the universal floor, and any declared constrained-process contribution tightens it monotonically.

3.  $H_{\mathrm{nats}} = \ln(\mathrm{numOptClasses}) \leq \mathrm{srank} \cdot \ln 2$ (Theorem [\[thm:entropy-rank\]](#thm:entropy-rank){reference-type="ref" reference="thm:entropy-rank"}).

Combining: $E \geq \mathrm{srank} \cdot k_B T \ln 2 \geq k_B T \cdot H_{\mathrm{nats}}$. ◻
:::

**Consequence.** Thermodynamic cost tracks information content: energy per decision $\geq k_B T$ per nat of decision entropy. This is derived from the Landauer floor (Theorem 9) composed with the entropy-rank inequality (Theorem 8); in the explicit decomposed constrained-process interface, a theorem-level KL mismatch witness, a theorem-level finite residual witness produced by the exhaustive reverse-edge split, its process-level witness packaging, or the cited broader stopping-time residual branch only strengthens the estimate.WC1WC4WM3WM4WM6WR6WR7WR8WR9WR10WP1WP5WP6

### 17. Thermodynamic Uncertainty Relations

::: theorem
[]{#thm:tur-bridge label="thm:tur-bridge"} Conditional on stochastic thermodynamics (Barato--Seifert 2015):

1.  Transition probabilities are normalized: $p_{ij} \geq 0$, $\sum_j p_{ij} = 1$.

2.  The Second Law enforces non-negative entropy production: $\sigma \geq 0$.

3.  The TUR bound constrains precision: $\mathrm{Var}(J)/\langle J \rangle^2 \geq 2/\sigma$.

4.  Multiple futures require positive entropy production.

Precision requires entropy dissipation, scaling with structural rank. TUR1TUR2TUR5TUR6
:::

::: proof
*Proof.* (1) Standard probability. (2) The Second Law (empirical assumption). (3) TUR is a theorem of stochastic thermodynamics: for any current $J$ in a Markov process with entropy production $\sigma$, precision is bounded by $\mathrm{Var}(J)/\langle J \rangle^2 \geq 2/\sigma$. (4) If $\mathrm{srank} > 1$, distinguishing $k > 1$ futures requires information acquisition. By the TUR bound, finite precision requires $\sigma > 0$. The minimal $\sigma$ scales with $\log k$, hence with $\mathrm{srank}$. ◻
:::

**Consequence.** Accepting fluctuation theorems yields an independent constraint: precision costs entropy production, with the bound scaling with srank. This is conditional on accepting the Second Law.

## Consistency with Existing Physical Theories {#sec:intro-justifies-physics}

The Counting Gap Theorem is consistent with and illuminates why discrete models work across physics:

-   **Quantum Mechanics:** Discrete eigenstates are forced for bounded systems

-   **Statistical Mechanics:** Discrete microstates forced by finite accessible state count

-   **Thermodynamics:** Finite entropy forced by bounded information acquisition

-   **Lattice Gauge Theory:** Discretization respects the computational constraint

The paper rests on one irreducible empirical claim (Definition [\[def:empirical-claims\]](#def:empirical-claims){reference-type="ref" reference="def:empirical-claims"}):

-   **EC1 (Temperature):** $\varepsilon \geq k_B T \ln 2 > 0$ (entropy has a positive lower-bound energy cost)

EC2 (finite resources) is derived from the existence of uncertainty (Theorem [\[thm:ec2-derived\]](#thm:ec2-derived){reference-type="ref" reference="thm:ec2-derived"}): if you don't know everything, your capacity is bounded. EC3 (finite speed) is derived from non-contradiction (Theorem [\[thm:ec3-derived\]](#thm:ec3-derived){reference-type="ref" reference="thm:ec3-derived"}): a computational step must take positive time, or the state would be both $A$ and not-$A$ at the same instant. Everything else---nontriviality, second law structure, Landauer structure, EC2, EC3---follows from counting, logic, or observation, with the numerical thermodynamic scale entering only through the declared lower-bound premise in EC1.

## Our Formal Tools: Coordinate Sufficiency {#sec:intro-formal-tools}

We study decision problems with coordinate structure. The finite coordinate spaces $X_i$ are forced by the Counting Gap. Boolean coordinates $X_i = \{0,1\}$ are elementary: one physical transition per bit.

A coordinate set $I$ is sufficient if knowing $s_I$ determines the optimal action: $$s_I = s'_I \implies \operatorname{Opt}(s) = \operatorname{Opt}(s')$$

We prove complexity classifications for three meta-problems:

-   **SUFFICIENCY-CHECK:** Is $I$ sufficient? coNP-complete

-   **MINIMUM-SUFFICIENT-SET:** Find minimal $I$. coNP-complete

-   **ANCHOR-SUFFICIENCY:** Exists sufficient $I$ with $|I| \leq k$? $\Sigma_2^P$-complete

Stochastic variants are PP-complete. Sequential variants are PSPACE-complete. Strict hierarchy: static $\subset$ stochastic $\subset$ sequential.

## Machine Verification {#sec:intro-verification}

The Lean 4 formalization consists of:

-   24,347 lines of Lean code

-   1,082 theorems

-   0 `sorry` placeholders

-   0 hidden axioms

#### What is machine-verified.

Bayesian optimality: proved from $\log x \leq x - 1$ alone. Thermodynamic consequences: conditional on a Landauer floor or any stricter declared per-bit lower bound. **Reduction correctness**: the membership-preservation property of each reduction (e.g., $\varphi$ is a tautology $\Leftrightarrow$ $I = \emptyset$ is sufficient) is fully machine-verified.

#### What is a standard claim.

**Polynomial-time computability** of the reduction functions is a standard claim stated as an explicit hypothesis in the Lean formalization. Full Turing-machine verification of polynomial-time bounds remains an open problem in formal verification---no formalization project has achieved end-to-end polynomial-time TM proofs for concrete reductions. The gap is made visible by representing polytime as a hypothesis rather than hiding it behind definitions that conflate size bounds with time bounds. Machine-generated assumption ledger records all dependencies.

## Paper Structure {#sec:intro-structure}

Section [\[sec:foundations\]](#sec:foundations){reference-type="ref" reference="sec:foundations"}: formal definitions (decision problems, sufficiency, DQ). Section [\[sec:physical-grounding\]](#sec:physical-grounding){reference-type="ref" reference="sec:physical-grounding"}: physical grounding (Counting Gap, three laws). Section [\[sec:hardness\]](#sec:hardness){reference-type="ref" reference="sec:hardness"}: hardness proofs (coNP, $\Sigma_2^P$, PP, PSPACE). Section [\[sec:dichotomy\]](#sec:dichotomy){reference-type="ref" reference="sec:dichotomy"}: regime separation. Section [\[sec:tractable\]](#sec:tractable){reference-type="ref" reference="sec:tractable"}: six tractable subcases. Sections [\[sec:engineering-justification\]](#sec:engineering-justification){reference-type="ref" reference="sec:engineering-justification"}--[\[sec:simplicity-tax\]](#sec:simplicity-tax){reference-type="ref" reference="sec:simplicity-tax"}: engineering corollaries. Appendix: Lean listings.

#### Claim-stamp notation.

Claims are stamped with theorem references $[T:n; P:m]$ and Lean handles (H1).

  -----------------------------------------------------------------------------------------------
  **Regime**   **Question**                            **Placement**
  ------------ --------------------------------------- ------------------------------------------
  Static       SUFFICIENCY / MINIMUM / ANCHOR          coNP / coNP / $\Sigma_2^P$

  Stochastic   STOCHASTIC-SUFFICIENCY                  PP-complete

  Sequential   SEQUENTIAL-SUFFICIENCY                  PSPACE-complete
  -----------------------------------------------------------------------------------------------


# Formal Foundations {#sec:foundations}

We formalize decision problems with coordinate structure, sufficiency of coordinate sets, and the optimizer quotient object, drawing on classical decision theory [@savage1954foundations; @raiffa1961applied]. The core definitions in this section are substrate-neutral: they specify the decision relation independently of implementation medium. The mechanized physical-lift and claim-transport scaffolding is summarized later in Section [1.4](#sec:foundations-mechanized-metadata){reference-type="ref" reference="sec:foundations-mechanized-metadata"} so that the mathematical core appears first.

## Decision Problems with Coordinate Structure

Formally: this subsection fixes the core tuple, projection, optimizer, and sufficiency/relevance predicates used downstream.

::: definition
[]{#def:decision-problem label="def:decision-problem"} A *decision problem with coordinate structure* is a tuple $\mathcal{D} = (A, X_1, \ldots, X_n, U)$ where:

-   $A$ is a finite set of *actions* (alternatives)

-   $X_1, \ldots, X_n$ are finite *coordinate spaces*

-   $S = X_1 \times \cdots \times X_n$ is the *state space*

-   $U : A \times S \to \mathbb{Q}$ is the *utility function*
:::

::: definition
[]{#def:projection label="def:projection"} For state $s = (s_1, \ldots, s_n) \in S$ and coordinate set $I \subseteq \{1, \ldots, n\}$: $$s_I := (s_i)_{i \in I}$$ is the *projection* of $s$ onto coordinates in $I$.
:::

::: definition
[]{#def:optimizer label="def:optimizer"} For state $s \in S$, the *optimal action set* is: $$\operatorname{Opt}(s) := \arg\max_{a \in A} U(a, s) = \{a \in A : U(a,s) = \max_{a' \in A} U(a', s)\}$$
:::

## Sufficiency and Relevance

::: definition
[]{#def:sufficient label="def:sufficient"} A coordinate set $I \subseteq \{1, \ldots, n\}$ is *sufficient* for decision problem $\mathcal{D}$ if: $$\forall s, s' \in S: \quad s_I = s'_I \implies \operatorname{Opt}(s) = \operatorname{Opt}(s')$$ Equivalently, the optimal action depends only on coordinates in $I$. In this paper, this set-valued invariance of the full optimal-action correspondence is the primary decision-relevance target.
:::

::: proposition
[]{#prop:empty-sufficient-constant label="prop:empty-sufficient-constant"} For any decision problem, $$\text{sufficient}(\emptyset)
\iff
\forall s,s'\in S,\ \operatorname{Opt}(s)=\operatorname{Opt}(s').$$ Hence, the $I=\emptyset$ query is exactly a universal constancy check of the decision boundary.
:::

::: proposition
[]{#prop:insufficiency-counterexample label="prop:insufficiency-counterexample"} For any coordinate set $I$, $$\neg\text{sufficient}(I)
\iff
\exists s,s'\in S:\ s_I=s'_I\ \wedge\ \operatorname{Opt}(s)\neq\operatorname{Opt}(s').$$ In particular, $$\neg\text{sufficient}(\emptyset)
\iff
\exists s,s'\in S:\ \operatorname{Opt}(s)\neq\operatorname{Opt}(s').$$
:::

\[D:Ddef:sufficient;Pprop:empty-sufficient-constant, prop:insufficiency-counterexample; R:DM\]

::: definition
[]{#def:selector label="def:selector"} A *deterministic selector* is a map $\sigma : 2^A \to A$ that returns one action from an optimal-action set.
:::

::: definition
[]{#def:selector-sufficient label="def:selector-sufficient"} For fixed selector $\sigma$, a coordinate set $I$ is *selector-sufficient* if $$\forall s,s' \in S:\quad s_I=s'_I \implies \sigma(\operatorname{Opt}(s))=\sigma(\operatorname{Opt}(s')).$$
:::

::: proposition
[]{#prop:set-to-selector label="prop:set-to-selector"} If $I$ is sufficient in the set-valued sense (Definition [\[def:sufficient\]](#def:sufficient){reference-type="ref" reference="def:sufficient"}), then $I$ is selector-sufficient for every deterministic selector $\sigma$ (Definition [\[def:selector-sufficient\]](#def:selector-sufficient){reference-type="ref" reference="def:selector-sufficient"}).
:::

::: proof
*Proof.* If $I$ is sufficient, then $s_I=s'_I$ implies $\operatorname{Opt}(s)=\operatorname{Opt}(s')$. Applying any deterministic selector $\sigma$ to equal optimal-action sets yields equal selected actions. ◻
:::

::: definition
[]{#def:epsilon-sufficiency label="def:epsilon-sufficiency"} For $\varepsilon \ge 0$, define $$\operatorname{Opt}_\varepsilon(s)
:=
\{a\in A:\forall a'\in A,\ U(a',s)\le U(a,s)+\varepsilon\}.$$ Coordinate set $I$ is *$\varepsilon$-sufficient* if $$\forall s,s'\in S:\ s_I=s'_I \implies \operatorname{Opt}_\varepsilon(s)=\operatorname{Opt}_\varepsilon(s').$$
:::

::: proposition
[]{#prop:zero-epsilon-reduction label="prop:zero-epsilon-reduction"} Set-valued sufficiency is exactly the $\varepsilon=0$ case: $$I\text{ sufficient } \iff I\text{ is }0\text{-sufficient}.$$
:::

::: proposition
[]{#prop:selector-separation label="prop:selector-separation"} The converse of Proposition [\[prop:set-to-selector\]](#prop:set-to-selector){reference-type="ref" reference="prop:set-to-selector"} fails in general: there exists a decision problem, selector, and coordinate set $I$ such that $I$ is selector-sufficient but not sufficient in the full set-valued sense.
:::

::: definition
[]{#def:minimal-sufficient label="def:minimal-sufficient"} A sufficient set $I$ is *minimal* if no proper subset $I' \subsetneq I$ is sufficient.
:::

::: definition
[]{#def:relevant label="def:relevant"} Coordinate $i$ is *relevant* if there exist states that differ only at coordinate $i$ and induce different optimal action sets: $$i \text{ is relevant}
\iff
\exists s,s' \in S:\; \Big(\forall j \neq i,\; s_j = s'_j\Big)\ \wedge\ \operatorname{Opt}(s)\neq\operatorname{Opt}(s').$$
:::

::: proposition
[]{#prop:minimal-relevant-equiv label="prop:minimal-relevant-equiv"} For any minimal sufficient set $I$ and any coordinate $i$: $$i \in I \iff i \text{ is relevant}.$$ Hence every minimal sufficient set is exactly the relevant-coordinate set.
:::

::: proof
*Proof.* The "only if" direction follows by minimality: if $i\in I$ were irrelevant, removing $i$ would preserve sufficiency, contradicting minimality. The "if" direction follows from sufficiency: every sufficient set must contain each relevant coordinate. ◻
:::

::: proposition
[]{#prop:srank-support label="prop:srank-support"} The structural rank $\mathrm{srank}(\mathcal{D})$ is the cardinality of the relevant-coordinate support: $$\mathrm{srank}(\mathcal{D})
=
\left|\{i \in \{1,\ldots,n\} : i \text{ is relevant}\}\right|.$$ Hence $\mathrm{srank}(\mathcal{D})$ is bounded by the ambient coordinate dimension $n$, and $\mathrm{srank}(\mathcal{D}) = 0$ exactly when the decision boundary is constant across coordinates.
:::

::: proof
*Proof.* Proof sketch: by definition, structural rank counts the coordinates that survive the relevance filter. The upper bound follows because at most all coordinates can be relevant. The zero case is equivalent to the relevance support being empty, which is exactly the coordinate-constant boundary case. ◻
:::

Informally: this is a familiarity anchor for the invariant used throughout the later bridge theorems. In the finite-coordinate setting, $\mathrm{srank}$ is simply the support size of the relevance indicator. Later theorems show that Fisher rank, zero-distortion rate, and physical bit cost recover this same support-size quantity through different frameworks.

::: definition
[]{#def:exact-identifiability label="def:exact-identifiability"} For a decision problem $\mathcal{D}$ and candidate coordinate set $I$, we say $I$ is *exactly relevance-identifying* if $$\forall i \in \{1,\ldots,n\}:\quad i \in I \iff i \text{ is relevant for } \mathcal{D}.$$ Equivalently, $I$ is exactly relevance-identifying iff $I$ equals the full relevant-coordinate set.
:::

::: example
Consider deciding whether to carry an umbrella:

-   Actions: $A = \{\text{carry}, \text{don't carry}\}$

-   Coordinates: $X_1 = \{\text{rain}, \text{no rain}\}$, $X_2 = \{\text{hot}, \text{cold}\}$, $X_3 = \{\text{Monday}, \ldots, \text{Sunday}\}$

-   Utility: $U(\text{carry}, s) = -1 + 3 \cdot \mathbf{1}[s_1 = \text{rain}]$, $U(\text{don't carry}, s) = -2 \cdot \mathbf{1}[s_1 = \text{rain}]$

The minimal sufficient set is $I = \{1\}$ (only rain forecast matters). Coordinates 2 and 3 (temperature, day of week) are irrelevant.
:::

Informally: all later results use this same sufficiency definition.

## Decision-Quotient Score and Optimizer Quotient Object

::: definition
[]{#def:decision-equiv label="def:decision-equiv"} For coordinate set $I$, states $s, s'$ are *$I$-equivalent* (written $s \sim_I s'$) if $s_I = s'_I$.
:::

::: definition
[]{#def:decision-quotient label="def:decision-quotient"} The *decision-quotient score* for state $s$ under coordinate set $I$ is: $$\text{DQ}_I(s) = \frac{|\{a \in A : a \in \operatorname{Opt}(s') \text{ for some } s' \sim_I s\}|}{|A|}$$ This is a normalized ambiguity score: equivalently, it is the normalized support fraction of the multivalued image of the $I$-equivalence class under $\operatorname{Opt}$. It measures the fraction of actions that are optimal for at least one state consistent with $I$. It is not the quotient object $S/{\sim}$ used later for the optimizer map.
:::

::: proposition
[]{#prop:sufficiency-char label="prop:sufficiency-char"} \[D:Pprop:sufficiency-char; R:H=succinct-hard\] Coordinate set $I$ is sufficient if and only if $\text{DQ}_I(s) = |\operatorname{Opt}(s)|/|A|$ for all $s \in S$.
:::

::: proof
*Proof.* If $I$ is sufficient, then $s \sim_I s' \implies \operatorname{Opt}(s) = \operatorname{Opt}(s')$, so the set of actions optimal for some $s' \sim_I s$ is exactly $\operatorname{Opt}(s)$.

Conversely, if the condition holds, then for any $s \sim_I s'$, the optimal actions form the same set (since $\text{DQ}_I(s) = \text{DQ}_I(s')$ and both equal the relative size of the common optimal set). ◻
:::

The quotient object behind the optimizer map has a familiar form in **Set**. It is useful to state that explicitly before the later abstraction-boundary theorem.

::: proposition
[]{#prop:optimizer-coimage label="prop:optimizer-coimage"} For the optimizer map $\operatorname{Opt}: S \to \mathcal{P}(A)$:

1.  the optimizer quotient object is canonically equivalent to $\mathrm{range}(\operatorname{Opt})$;

2.  any surjective abstraction $\phi : S \to T$ that preserves $\operatorname{Opt}$ factors uniquely through this quotient.

Thus, in **Set**, the optimizer quotient object is the familiar coimage of $\operatorname{Opt}$, canonically identified with its image/range.[@maclane1998categories]
:::

::: proof
*Proof.* Proof sketch:

1.  The quotient identifies exactly those states with equal optimal-action sets, so its elements correspond bijectively to the values attained by $\operatorname{Opt}$. This gives the canonical equivalence with $\mathrm{range}(\operatorname{Opt})$.

2.  If a surjective abstraction preserves $\operatorname{Opt}$, then the optimizer map already factors through that abstraction. The quotient universal property supplies the mediating map, and surjectivity makes that factorization unique.

 ◻
:::

Informally: this proposition is only a familiarity anchor. The object is not exotic in **Set**; the later novelty is that this quotient governs the complexity and physical-collapse arguments.

The same quotient object has an equally familiar information-theoretic reading: its entropy is just the logarithm of the number of distinct optimizer values actually attained.

::: proposition
[]{#prop:optimizer-entropy-image label="prop:optimizer-entropy-image"} Let $\mathrm{numOptClasses}(\mathcal{D})$ denote the number of distinct values attained by $\operatorname{Opt}$, equivalently the cardinality of $\mathrm{range}(\operatorname{Opt})$. Then $$H(\mathcal{D}) = \log_2\bigl(\mathrm{numOptClasses}(\mathcal{D})\bigr)
=
\log_2\bigl(|\mathrm{range}(\operatorname{Opt})|\bigr).$$ Thus the entropy of the optimizer quotient object is just the base-2 logarithm of the size of the image of $\operatorname{Opt}$.
:::

::: proof
*Proof.* Proof sketch: $\mathrm{numOptClasses}$ is defined as the cardinality of the image of $\operatorname{Opt}$, and Proposition [\[prop:optimizer-coimage\]](#prop:optimizer-coimage){reference-type="ref" reference="prop:optimizer-coimage"} identifies the optimizer quotient object with that image in **Set**. The entropy quantity $H(\mathcal{D})$ is then defined to be $\log_2(\mathrm{numOptClasses})$. ◻
:::

Informally: this is the familiar reading of the entropy term used later in the bridge theorems. Nothing new is being hidden inside the notation: it is just the logarithm of how many different optimal-action sets the system can realize.

The next theorem concerns the quotient object rather than the scalar score above. Abstraction is a many-to-one quotient operation: it identifies states and thereby removes distinctions. The optimizer quotient object is the coarsest such collapse that still preserves the optimal-action boundary.

Informally: start with any surjective summary map $\phi : S \to T$. There are only two possibilities. Either $\phi$ keeps every optimal-action distinction intact, in which case the universal property forces it to factor through the optimizer quotient object, or $\phi$ merges two states that require different optimal actions, in which case it has erased a decision-relevant distinction. In **Set**, the quotient object here is the coimage of $\operatorname{Opt}$, canonically equivalent to its image/range [@maclane1998categories]. The physical no-collapse layer then says that if one tries to treat that extra erasure as computationally affordable, the collapse assumption itself is what fails.

::: theorem
[]{#thm:abstraction-boundary label="thm:abstraction-boundary"} Let $\phi : S \to T$ be a surjective abstraction of the state space.

1.  Either $\phi$ factors through the optimizer quotient object, or it collapses a decision-relevant distinction, i.e., there exist states $s,s' \in S$ such that $\phi(s)=\phi(s')$ but $\operatorname{Opt}(s)\neq\operatorname{Opt}(s')$.

2.  If every such extra collapse is mapped to a physically feasible collapse at the canonical requirement profile, then no such extra collapse can occur. Therefore $\phi$ must factor through the optimizer quotient object.
:::

::: proof
*Proof.* Proof sketch:

1.  If $\phi$ preserves $\operatorname{Opt}$, then the quotient universal property already proved in the previous layer supplies the factorization $\pi = \psi \circ \phi$. In **Set**, this is exactly the coimage factorization of the optimizer map through its image/range. If $\phi$ does not preserve $\operatorname{Opt}$, then by definition there must exist $s,s'$ with $\phi(s)=\phi(s')$ but $\operatorname{Opt}(s)\neq\operatorname{Opt}(s')$. That witness is exactly the erased decision-relevant distinction.

2.  Suppose such an erased distinction is further interpreted as a physically feasible collapse at the canonical requirement profile. The existing physical no-collapse theorem then blocks that collapse map. Therefore the "extra erasure" branch is impossible, leaving only factorization through the optimizer quotient object.

 ◻
:::

Informally: this is the explicit bridge between state-space abstraction and complexity-class collapse. At the state level, the optimizer quotient object is the maximal lossless collapse. In **Set**, that means the coimage/image factorization of $\operatorname{Opt}$. At the physical-complexity level, any claim that a coarser collapse remains feasible must pass through the same no-collapse obstruction. The theorem makes the common structure visible instead of leaving it implicit across separate sections.

## Mechanized Physical-Lift and Claim Metadata {#sec:foundations-mechanized-metadata}

The formal system also records the physical-lift and claim-transport scaffolding used later in the paper. Physical interpretation is introduced through explicit encoding and assumption-transfer maps rather than built into the core tuple itself. \[D:Ddef:physical-core-encoding;Pprop:physical-claim-transport;Tthm:physical-bridge-bundle; R:AR\]

Within that scaffolding, physical-state semantics are typed by realizable-state witnesses, observer classes induce quotient state spaces on a shared physical domain, coupled recurrent circuits support typed YES/NO/ABSTAIN report projection, and timing constraints yield forced-action and lower-bound statements under declared physical parameterizations.

The same metadata layer also packages the assumption-indexed physical-collapse transfer chain, the layered graph witnesses, and the exact-claim admissibility boundary used later in the physics and claim-discipline sections. This material remains explicit, but it is secondary to the mathematical core definitions established above. \[D:Tthm:exact-certified-gap-iff-admissible;Pprop:integrity-resource-bound;Ccor:hardness-raw-only-exact; R:AR\]

## Computational Model and Input Encoding {#sec:encoding}

Formally: this subsection defines encoding/access regimes, cost measures, and regime-specific complexity interpretation.

We fix the computational model used by the complexity claims. In physical deployments, these encodings correspond to different access conditions on the same underlying system: explicit tables correspond to fully exposed state/utility structure, succinct encodings correspond to compressed generative descriptions, and query regimes correspond to constrained measurement interfaces. \[D:Tthm:dichotomy;Pprop:query-regime-obstruction, prop:physical-claim-transport; R:E,S+ETH,Qf,AR\]

#### Succinct encoding (primary for hardness).

This succinct circuit encoding is the standard representation for decision problems in complexity theory; hardness is stated with respect to the input length of the circuit description [@arora2009computational; @papadimitriou1994complexity]. An instance is encoded as:

-   a finite action set $A$ given explicitly,

-   coordinate domains $X_1,\ldots,X_n$ given by their sizes in binary,

-   a Boolean or arithmetic circuit $C_U$ that on input $(a,s)$ outputs $U(a,s)$.

The input length is $L = |A| + \sum_i \log |X_i| + |C_U|$. Polynomial time and all complexity classes (coNP, $\Sigma_2^P$, ETH) are measured in $L$. All hardness results in Section [\[sec:hardness\]](#sec:hardness){reference-type="ref" reference="sec:hardness"} use this encoding.

#### Explicit-state encoding (used for enumeration algorithms and experiments).

The utility is given as a full table over $A \times S$. The input length is $L_{\text{exp}} = \Theta(|A||S|)$ (up to the bitlength of utilities). Polynomial time is measured in $L_{\text{exp}}$. Results stated in terms of $|S|$ use this encoding.

#### Query-access regime (intermediate black-box access).

The solver is given oracle access to decision information at queried states (e.g., $\operatorname{Opt}(s)$, or $U(a,s)$ with $\operatorname{Opt}(s)$ reconstructed from finitely many value queries). Complexity is measured by oracle-query count (optionally paired with per-query evaluation cost). This separates representational access from full-table availability and from succinct-circuit input length.

Unless explicitly stated otherwise, "polynomial time" refers to the succinct encoding.

::: remark
[]{#rem:question-vs-problem label="rem:question-vs-problem"} We distinguish between a *decision question* and a *decision problem*. A decision question is the encoding-independent predicate: "is coordinate set $I$ sufficient for $\mathcal{D}$?" This is a mathematical relation, fixed by the model contract (C1--C4). A decision problem is a decision question together with a specific encoding of its inputs. The same decision question yields different decision problems under different encodings, and these problems may belong to different complexity classes. Formally, each encoding regime $e$ induces a language $L_{\mathcal{D},e}$; complexity classification is a property of $L_{\mathcal{D},e}$, not of $\mathcal{D}$ alone.
:::

::: definition
[]{#def:anchor-query-object label="def:anchor-query-object"} An anchor query object is a pair $$q=(\mathcal{D},I)$$ where $\mathcal{D}$ is a decision problem and $I$ is a fixed coordinate set. Its truth predicate is $$\mathrm{AnchorTrue}(q)\iff
\exists s_0\in S\ \forall s\in S,\ s_I=s_{0,I}\Rightarrow \operatorname{Opt}(s)=\operatorname{Opt}(s_0).$$
:::

::: definition
[]{#def:pose-anchor-query label="def:pose-anchor-query"} The pose operation is the constructor $$\mathrm{poseAnchor}:\ (\mathcal{D},I)\mapsto q.$$ Posing is an operation; $q$ is the formal object.
:::

::: proposition
[]{#prop:pose-anchor-object label="prop:pose-anchor-object"} For every $(\mathcal{D},I)$, $\mathrm{poseAnchor}(\mathcal{D},I)$ is a well-typed anchor query object with preserved components, and its truth predicate is exactly the anchored $\exists\forall$ condition in Definition [\[def:anchor-query-object\]](#def:anchor-query-object){reference-type="ref" reference="def:anchor-query-object"}. \[D:Ddef:anchor-query-object, def:pose-anchor-query;Tthm:anchor-sigma2p; R:S\]
:::

::: proposition
[]{#prop:posed-anchor-typed-exact label="prop:posed-anchor-typed-exact"} For posed anchor queries under the typed claim discipline: $$\text{Exact admissible}\iff\text{competent on the declared regime},\qquad
\neg\text{competent}\Rightarrow\neg\text{exact admissible}.$$ Under solver integrity, checker-accepted exact-`true` verdicts are semantically sound on the posed anchor query object. \[D:Pprop:typed-claim-admissibility, prop:integrity-competence-separation;Ccor:no-uncertified-exact-claim; R:AR\]
:::

::: corollary
[]{#cor:posed-anchor-certified-gate label="cor:posed-anchor-certified-gate"} Under signal consistency, positive certified confidence implies typed admissibility of the posed anchor-query report. \[D:Ddef:signal-consistency;Pprop:certified-confidence-gate; R:AR\]
:::

::: proposition
[]{#prop:under-resolution-collision label="prop:under-resolution-collision"} \[D:Pprop:under-resolution-collision; R:DM\] Let $\mathrm{repr} : \mathcal{L} \to \mathcal{C}$ be any representation map from logical possibilities to code states. If $|\mathcal{C}| < |\mathcal{L}|$, then there exist distinct logical states with the same representation: $$\exists s,s' \in \mathcal{L},\ s \neq s' \wedge \mathrm{repr}(s)=\mathrm{repr}(s').$$ This is the finite-cardinality collision statement underlying under-resolved exact-decision claims.
:::

::: theorem
[]{#thm:physical-incompleteness label="thm:physical-incompleteness"} \[D:Tthm:physical-incompleteness; R:AR\] Let $\mathcal{P}$ be a finite set of physical states and $\mathcal{L}$ a finite set of logical possibilities, with an instantiation map $\iota : \mathcal{P} \to \mathcal{L}$. If $|\mathcal{P}| < |\mathcal{L}|$, then:

1.  $\iota$ is not surjective;

2.  there exists a logical possibility $\ell \in \mathcal{L}$ that is not physically instantiated: $\ell \notin \mathrm{range}(\iota)$.

Under explicit bounds: if $|\mathcal{P}| \le M$ and $|\mathcal{L}| \ge L$ with $M < L$, the same conclusion holds.
:::

::: definition
[]{#def:regime-simulation label="def:regime-simulation"} Fix a decision question $Q$ and two regimes $R_1,R_2$ for that question. We say $R_1$ *simulates* $R_2$ if there exists a polynomial-time transformer that maps any $R_2$ instance to an $R_1$ instance while preserving the answer to $Q$; in oracle settings, this includes a polynomial-time transcript transducer that preserves yes/no outcomes for induced solvers.
:::

This paper uses that simulation spine in two explicit forms: restriction-map simulation (subproblem-to-full transfer; Proposition [\[prop:query-subproblem-transfer\]](#prop:query-subproblem-transfer){reference-type="ref" reference="prop:query-subproblem-transfer"}) and oracle-transducer simulation (batch-to-entry transfer; Proposition [\[prop:oracle-lattice-transfer\]](#prop:oracle-lattice-transfer){reference-type="ref" reference="prop:oracle-lattice-transfer"}).

Informally: the same question can be easy or hard depending on representation and access, with explicit transfer requirements.

## Model Contract and Regime Tags {#sec:model-contract}

All theorem statements in this paper are typed by the following model contract:

-   **C1 (finite actions):** $A$ is finite.

-   **C2 (finite coordinate domains):** each $X_i$ is finite, so $S=\prod_i X_i$ is finite.

-   **C3 (evaluable utility):** $U(a,s)$ is computable from the declared input encoding.

-   **C4 (fixed decision semantics):** optimality is defined by $\operatorname{Opt}(s)=\arg\max_a U(a,s)$.

We use short regime tags for applied corollaries:

-   **\[E\]** explicit-state encoding,

-   **\[Q\]** query-access (oracle) regime,

-   **\[Q_fin\]** finite-state query-access core (state-oracle),

-   **\[S\]** succinct encoding,

-   **\[S+ETH\]** succinct encoding with ETH,

-   **\[Q_bool\]** mechanized Boolean query submodel,

-   **\[S_bool\]** mechanized Boolean-coordinate submodel.

This tagging is a claim-typing convention: each strong statement is attached to the regime where it is proven.

::: remark
[]{#rem:uniform-assumption-discipline label="rem:uniform-assumption-discipline"} All theorem-level claims in this paper are conditional on their stated premises and regime tags; no assumption family is given special status. For each claim, the active assumptions are exactly those appearing in its statement, its regime tag, and its cited mechanized handles.
:::

::: remark
[]{#rem:modal-should label="rem:modal-should"} In this paper, normative modal statements denote contract-conditional necessity: once a physical-system model declares an objective/regime/assumption contract, admissible reports and interventions are fixed by typed admissibility and integrity under that contract. No unconditional external moral prescription is asserted.
:::

## Discrete-Time Event Semantics {#sec:discrete-time-semantics}

::: definition
[]{#def:timed-interface-state label="def:timed-interface-state"} For any interface state space $S$, a timed interface state is a pair $$x=(s,t)\in S\times \mathbb{N}.$$ The time coordinate is the natural-number component $t$.
:::

::: definition
[]{#def:decision-tick-event label="def:decision-tick-event"} Fix an interface decision process $(\mathrm{decide},\mathrm{transition})$ with $$\mathrm{decide}:S\to A,\qquad \mathrm{transition}:S\times A\to S.$$ Define one tick by $$\mathrm{tick}(s,t)=\bigl(\mathrm{transition}(s,\mathrm{decide}(s)),\,t+1\bigr).$$ Define $\mathrm{DecisionEvent}(x,y)$ to mean that $y$ is the one-tick successor of $x$ under this rule.
:::

::: proposition
[]{#prop:time-discrete label="prop:time-discrete"} In Definition [\[def:timed-interface-state\]](#def:timed-interface-state){reference-type="ref" reference="def:timed-interface-state"}, time is discrete: $$\forall x\in S\times\mathbb{N},\ \exists k\in\mathbb{N}\ \text{such that}\ x_t=k.$$ Pointwise time equality is decidable for every $k\in\mathbb{N}$: $$x_t=k\ \vee\ x_t\neq k.$$
:::

::: proposition
[]{#prop:decision-unit-time label="prop:decision-unit-time"} Under Definition [\[def:decision-tick-event\]](#def:decision-tick-event){reference-type="ref" reference="def:decision-tick-event"}, for all timed states $x,y$: $$\mathrm{DecisionEvent}(x,y)\ \Longleftrightarrow\ y=\mathrm{tick}(x),$$ and therefore $$\mathrm{DecisionEvent}(x,y)\ \Longrightarrow\ y_t=x_t+1.$$
:::

::: proposition
[]{#prop:run-time-accounting label="prop:run-time-accounting"} Let $\mathrm{run}(n,x)$ be $n$ repeated ticks from $x$, and let $\mathrm{trace}(n,x)$ be the emitted decision sequence. Then: $$\mathrm{run}(n,x)_t = x_t + n,\qquad
\mathrm{run}(n,x)_t - x_t = n,\qquad
|\mathrm{trace}(n,x)|=n.$$ Hence decision count equals elapsed time: $$|\mathrm{trace}(n,x)|=\mathrm{run}(n,x)_t-x_t.$$
:::

::: proposition
[]{#prop:substrate-unit-time label="prop:substrate-unit-time"} For any substrate model that is consistent with the declared interface transition rule, each one-step substrate evolution realizes one interface decision event and advances interface time by one unit. The time update law is invariant under substrate kind.
:::

## Physical-System Encoding and Theorem Transport {#sec:physical-transport}

Formally: this subsection defines physical-to-core encoding and proves theorem transport via declared assumption maps.

::: definition
[]{#def:physical-core-encoding label="def:physical-core-encoding"} Fix a physical instance class $\mathcal{P}$ and a core decision class $\mathcal{D}$. A *physical-to-core encoding* is a map $$E:\mathcal{P}\to\mathcal{D}.$$ All physical claims below are typed relative to a declared encoding map $E$ and declared assumption transfer map from physical assumptions to core assumptions.
:::

::: definition
[]{#def:physical-scope-gate label="def:physical-scope-gate"} A claim is in physical scope only if it declares:

1.  a physical state class with observable coordinates,

2.  a physical-to-core encoding map $E:\mathcal{P}\to\mathcal{D}$,

3.  calibrated measurement units for reported quantities (for example bits, joules, and $kT$-scaled bounds),

4.  a declared objective contract and admissible intervention class,

5.  an explicit physical-to-core assumption transfer map.

Claims missing any item are untyped for theorem transport in this framework.
:::

::: proposition
[]{#prop:typed-physical-transport-requirement label="prop:typed-physical-transport-requirement"} Direct transfer of core theorems is licensed only through a declared physical scope gate instance (Definition [\[def:physical-scope-gate\]](#def:physical-scope-gate){reference-type="ref" reference="def:physical-scope-gate"}) and a corresponding physical-to-core encoding (Definition [\[def:physical-core-encoding\]](#def:physical-core-encoding){reference-type="ref" reference="def:physical-core-encoding"}). Absent that typed physical instance, transport is blocked by construction. PS3PS4
:::

::: proof
*Proof.* By Definition [\[def:physical-scope-gate\]](#def:physical-scope-gate){reference-type="ref" reference="def:physical-scope-gate"}, theorem transport is only defined for claims with explicit physical state, objective, intervention, encoding, and assumption-transfer declarations. Therefore untyped targets have no admissible transport map in this framework. Proposition [\[prop:physical-claim-transport\]](#prop:physical-claim-transport){reference-type="ref" reference="prop:physical-claim-transport"} then applies only to typed instances. ◻
:::

This scope gate is a typing rule, not a rhetorical caveat: the paper's transport theorems apply only after explicit encoding and assumption transfer are declared. \[D:Ddef:physical-core-encoding, def:physical-scope-gate;Pprop:typed-physical-transport-requirement, prop:physical-claim-transport; R:AR\]

::: proposition
[]{#prop:heisenberg-strong-nontrivial-opt label="prop:heisenberg-strong-nontrivial-opt"} \[D:Pprop:heisenberg-strong-nontrivial-opt; R:AR\] Fix a declared noisy physical encoding interface and assume a *Heisenberg-strong binding witness*: one physical instance is compatible with two distinct interface states whose encoded optimal-action sets differ. Then there exists a core decision problem with non-constant optimizer map. Equivalently, under this declared physical ambiguity assumption, decision nontriviality is derivable at the core level.
:::

::: proposition
[]{#prop:physical-claim-transport label="prop:physical-claim-transport"} Let $E:\mathcal{P}\to\mathcal{D}$ be an encoding. If $$\forall d\in\mathcal{D},\ \mathrm{CoreAssm}(d)\Rightarrow \mathrm{CoreClaim}(d)$$ and $$\forall p\in\mathcal{P},\ \mathrm{PhysAssm}(p)\Rightarrow \mathrm{CoreAssm}(E(p)),$$ then $$\forall p\in\mathcal{P},\ \mathrm{PhysAssm}(p)\Rightarrow \mathrm{CoreClaim}(E(p)).$$
:::

::: corollary
[]{#cor:physical-counterexample-core-failure label="cor:physical-counterexample-core-failure"} Under the same encoding and assumption-transfer map:

1.  any encoded physical counterexample induces a core counterexample on the encoded slice;

2.  if a physical counterexample exists, then the purported core rule is invalid;

3.  if the lifted claim holds on all physically admissible encoded instances, no such physical counterexample exists.
:::

::: proposition
[]{#prop:law-instance-objective-bridge label="prop:law-instance-objective-bridge"} For the mechanized law-gap physical instance class, the encoded optimizer claim holds at theorem level and admits no counterexample: $$\forall x\in\mathcal{P}_{\mathrm{law}},\ \mathrm{lawGapPhysicalClaim}(x),
\qquad
\neg\exists x\in\mathcal{P}_{\mathrm{law}},\ \neg\,\mathrm{lawGapPhysicalClaim}(x).$$
:::

::: theorem
[]{#thm:physical-bridge-bundle label="thm:physical-bridge-bundle"} In one bundled mechanized statement, the paper's physical bridge layer composes:

1.  law-induced objective semantics ($\operatorname{Opt}=$ feasible actions for law-gap instances),

2.  behavior-preserving configuration reduction equivalence,

3.  irreducible required-work lower bound under site-count scaling.
:::

\[D:Ddef:physical-core-encoding;Pprop:physical-claim-transport, prop:law-instance-objective-bridge;Ccor:physical-counterexample-core-failure;Tthm:physical-bridge-bundle; R:AR\]

::: theorem
[]{#thm:regime-coverage label="thm:regime-coverage"} Let $$\mathcal{R}_{\mathrm{static}} =
\{\text{[E]},\ \text{[S+ETH]},\ \text{[Q\_fin:Opt]},\ \text{[Q\_bool:value-entry]},\ \text{[Q\_bool:state-batch]}\}.$$ For each declared regime in $\mathcal{R}_{\mathrm{static}}$, the paper has a theorem-level mechanized core claim. Equivalently, regime typing is complete over the declared static family.
:::

#### Scope Lattice (typed classes and transfer boundary).

::: center
  ---------------------------------------------------------------------------------------------------------------------------------------------------------
  **Layer**                                                                               **Transfer Status**           **Lean handles (representative)**
  --------------------------------------------------------------------------------------- ----------------------------- -----------------------------------
  Static sufficiency class (C1--C4; declared regimes)                                     Internal landscape complete   CC76

  CC51                                                                                                                  

  Bridge-admissible adjacent class (one-step deterministic)                               Transfer licensed             CC42

  CC10                                                                                                                  

  Non-admissible represented adjacent classes (horizon, stochastic, transition-coupled)   Transfer blocked by witness   CC9

  CC8                                                                                                                   
  ---------------------------------------------------------------------------------------------------------------------------------------------------------
:::

Informally: physical conclusions apply when the stated physical assumptions line up with the core theorem assumptions.

## Adjacent Objective Regimes and Bridge

::: definition
[]{#def:adjacent-sequential-regime label="def:adjacent-sequential-regime"} An adjacent sequential objective instance consists of:

-   finite action set $A$,

-   finite coordinate state space $S = X_1 \times \cdots \times X_n$,

-   horizon $T \in \mathbb{N}_{\ge 1}$ and history-dependent control-law class,

-   reward process $r_t$ and objective functional $J(\pi)$ (e.g., cumulative reward or regret).
:::

::: proposition
[]{#prop:one-step-bridge label="prop:one-step-bridge"} Consider an instance of Definition [\[def:adjacent-sequential-regime\]](#def:adjacent-sequential-regime){reference-type="ref" reference="def:adjacent-sequential-regime"} satisfying:

1.  $T=1$,

2.  deterministic rewards $r_1(a,s)=U(a,s)$ for some evaluable $U$,

3.  objective $J(\pi)=U(\pi(s),s)$ (single-step maximization),

4.  no post-decision state update relevant to the objective.

Then the induced optimization problem is exactly the static decision problem of Definition [\[def:decision-problem\]](#def:decision-problem){reference-type="ref" reference="def:decision-problem"}, and coordinate sufficiency in the sequential formulation is equivalent to Definition [\[def:sufficient\]](#def:sufficient){reference-type="ref" reference="def:sufficient"}.
:::

::: proof
*Proof.* Under (1)--(3), optimizing $J$ at state $s$ is identical to choosing an action in $\arg\max_{a\in A} U(a,s)=\operatorname{Opt}(s)$. Condition (4) removes any dependence on future transition effects. Therefore the optimal-policy relation in the adjacent formulation coincides pointwise with $\operatorname{Opt}$ from Definition [\[def:optimizer\]](#def:optimizer){reference-type="ref" reference="def:optimizer"}, and the invariance condition "same projection implies same optimal choice set" is exactly Definition [\[def:sufficient\]](#def:sufficient){reference-type="ref" reference="def:sufficient"}. ◻
:::

::: proposition
[]{#prop:bridge-transfer-scope label="prop:bridge-transfer-scope"} Under conditions (1)--(4), any sufficiency statement formulated over Definition [\[def:sufficient\]](#def:sufficient){reference-type="ref" reference="def:sufficient"} is equivalent between the adjacent sequential formulation and the static model.
:::

::: proof
*Proof.* By Proposition [\[prop:one-step-bridge\]](#prop:one-step-bridge){reference-type="ref" reference="prop:one-step-bridge"}, the adjacent sequential objective induces exactly the same sufficiency predicate as Definition [\[def:sufficient\]](#def:sufficient){reference-type="ref" reference="def:sufficient"} when (1)--(4) hold. Equivalence of any sufficiency statement then follows by substitution. ◻
:::

If any bridge condition fails, direct transfer from this paper's static complexity theorems is not licensed by this rule.

::: remark
Beyond Proposition [\[prop:one-step-bridge\]](#prop:one-step-bridge){reference-type="ref" reference="prop:one-step-bridge"} (multi-step horizon, stochastic transitions/rewards, or regret objectives), the governing complexity objects change. Those regimes are natural extensions, but they are distinct formal classes from the static sufficiency class analyzed in this paper.
:::

::: proposition
[]{#prop:bridge-failure-horizon label="prop:bridge-failure-horizon"} Dropping the one-step condition can break sufficiency transfer: there exists a two-step objective where sufficiency in the immediate-utility static projection does not match sufficiency in the two-step objective.
:::

::: proposition
[]{#prop:bridge-failure-stochastic label="prop:bridge-failure-stochastic"} If optimization is performed against a stochastic criterion not equal to deterministic utility maximization, bridge transfer to Definition [\[def:decision-problem\]](#def:decision-problem){reference-type="ref" reference="def:decision-problem"} can fail even when an expected-utility projection is available.
:::

::: proposition
[]{#prop:bridge-failure-transition label="prop:bridge-failure-transition"} If post-decision transition effects are objective-relevant, bridge transfer to the static one-step class can fail.
:::

::: theorem
[]{#thm:bridge-boundary-represented label="thm:bridge-boundary-represented"} Within the represented adjacent family $$\{\text{one-step deterministic},\ \text{horizon-extended},\ \text{stochastic-criterion},\ \text{transition-coupled}\},$$ direct transfer of static sufficiency claims is licensed if and only if the class is one-step deterministic. Each represented non-one-step class has an explicit mechanized bridge-failure witness.
:::

::: definition
[]{#def:system-snapshot-process label="def:system-snapshot-process"} Within the represented adjacent family, we type:

-   *system snapshot*: fixed-parameter inference view (mapped to one-step deterministic class),

-   *system process*: online/dynamical update views (horizon-extended, stochastic-criterion, or transition-coupled classes).
:::

::: proposition
[]{#prop:snapshot-process-typing label="prop:snapshot-process-typing"} In the represented adjacent family, direct transfer of static sufficiency claims is licensed if and only if the system is typed as a *system snapshot*. Every process-typed represented class has an explicit mechanized bridge-failure witness.
:::

Operationally: a fixed-parameter physical controller at runtime is a system snapshot for this typing; once parameters or transition-relevant internals are updated online, the object is process-typed and transfer from static sufficiency theorems is blocked unless the one-step bridge conditions are re-established. \[D:Pprop:snapshot-process-typing; R:RA\]

# Interpretive Foundations: Hardness and Solver Claims {#sec:interpretive-foundations}

The claims in later applied sections are theorem-indexed consequences of this section and Sections [\[sec:hardness\]](#sec:hardness){reference-type="ref" reference="sec:hardness"}--[\[sec:tractable\]](#sec:tractable){reference-type="ref" reference="sec:tractable"}.

## Structural Complexity vs Representational Hardness

Formally: this subsection separates class placement from regime-indexed access cost and states typed completeness for the declared static family.

::: definition
[]{#def:structural-complexity label="def:structural-complexity"} For a fixed decision question (e.g., "is $I$ sufficient for $\mathcal{D}$?"; see Remark [\[rem:question-vs-problem\]](#rem:question-vs-problem){reference-type="ref" reference="rem:question-vs-problem"}), *structural complexity* means its placement in standard complexity classes within the formal model (coNP, $\Sigma_2^P$, etc.), as established by class-membership arguments and reductions.
:::

::: definition
[]{#def:representational-hardness label="def:representational-hardness"} For a fixed decision question and an encoding regime $E$ (Section [1.5](#sec:encoding){reference-type="ref" reference="sec:encoding"}), *representational hardness* is the worst-case computational cost incurred by solvers whose input access is restricted to $E$.
:::

In the mechanized architecture model, this is made explicit as a required-work observable and intrinsic lower bound: 'requiredWork' is total realized work, 'hardnessLowerBound' is the intrinsic floor, and 'hardness_is_irreducible_required_work' proves the lower-bound relation for all nonzero use-site counts; 'requiredWork_eq_affine_in_sites' and 'workGrowthDegree' make the site-count growth law explicit.

::: remark
This paper keeps the decision question fixed and varies the encoding regime explicitly. Thus, later separations are read as changes in representational hardness under fixed structural complexity, not as changes to the underlying sufficiency semantics. Accordingly, "complete landscape" means complete for this static sufficiency class under the declared regimes; adjacent objective classes are distinct typed objects, not unproved remainder cases of the same class.
:::

::: theorem
[]{#thm:typed-completeness-static label="thm:typed-completeness-static"} For the static sufficiency class fixed by C1--C4 and declared regime family $\mathcal{R}_{\mathrm{static}}$, the paper provides:

1.  conditional class-label closures for SUFFICIENCY-CHECK, MINIMUM-SUFFICIENT-SET, and ANCHOR-SUFFICIENCY;

2.  a conditional explicit/succinct dichotomy closure;

3.  a regime-indexed mechanized core claim for every declared regime; and

4.  declared-family exhaustiveness in the typed regime algebra.
:::

Informally: the question stays fixed while representation changes cost, and the declared static regime set is fully covered.

## Solver Integrity and Regime Competence

Formally: this subsection defines integrity, competence, typed report/evidence admissibility, and resource-bounded exact-claim limits.

To keep practical corollaries type-safe, we separate *integrity* (what a solver is allowed to assert) from *competence* (what it can cover under a declared regime), following the certifying-algorithms schema [@mcconnell2010certifying; @necula1997proof]. The induced-relation view used below is standard in complexity/computability presentations: algorithms are analyzed as machines deciding languages or computing functions/relations over encodings [@papadimitriou1994complexity; @arora2009computational; @forster2019verified].

::: definition
[]{#def:certifying-solver label="def:certifying-solver"} Fix a decision question $\mathcal{R} \subseteq \mathcal{X}\times\mathcal{Y}$ and an encoding regime $E$ over $\mathcal{X}$. A *certifying solver* is a pair $(Q,V)$ where:

-   $Q$ maps each input $x\in\mathcal{X}$ to either $\mathsf{ABSTAIN}$ or a candidate pair $(y,w)$,

-   $V$ is a polynomial-time checker (in $|{\rm enc}_E(x)|$) with output in $\{0,1\}$.
:::

::: definition
[]{#def:induced-solver-relation label="def:induced-solver-relation"} For any deterministic program $M$ computing a partial map $f_M:\mathcal{X}\rightharpoonup\mathcal{Y}$ on encoded inputs, define $$\mathcal{R}_M
:=
\{(x,y)\in\mathcal{X}\times\mathcal{Y} : f_M(x)\downarrow \ \wedge\ y=f_M(x)\}.$$
:::

::: proposition
[]{#prop:universal-solver-framing label="prop:universal-solver-framing"} Every deterministic computer program that computes a (possibly partial) map on encoded inputs can be framed as a solver for its induced relation (Definition [\[def:induced-solver-relation\]](#def:induced-solver-relation){reference-type="ref" reference="def:induced-solver-relation"}). In this formal sense, all computers are solvers of specific problems.
:::

::: proof
*Proof.* Immediate from Definition [\[def:induced-solver-relation\]](#def:induced-solver-relation){reference-type="ref" reference="def:induced-solver-relation"}: each program induces a relation given by its graph on the domain where it halts. ◻
:::

::: definition
[]{#def:solver-integrity label="def:solver-integrity"} A certifying solver $(Q,V)$ has *integrity* for relation $\mathcal{R}$ if:

-   (assertion soundness) $Q(x)=(y,w)\implies V(x,y,w)=1$,

-   (checker soundness) $V(x,y,w)=1\implies (x,y)\in\mathcal{R}$.

The output $\mathsf{ABSTAIN}$ (equivalently, $\mathsf{UNKNOWN}$) is first-class and carries no assertion about membership in $\mathcal{R}$.
:::

::: definition
[]{#def:competence-regime label="def:competence-regime"} Fix a regime $\Gamma=(\mathcal{X}_\Gamma,E_\Gamma,\mathcal{C}_\Gamma)$ with instance family $\mathcal{X}_\Gamma\subseteq\mathcal{X}$, encoding assumptions $E_\Gamma$, and resource bound $\mathcal{C}_\Gamma$. A certifying solver $(Q,V)$ is *competent* on $\Gamma$ for relation $\mathcal{R}$ if:

-   it has integrity for $\mathcal{R}$ (Definition [\[def:solver-integrity\]](#def:solver-integrity){reference-type="ref" reference="def:solver-integrity"}),

-   (coverage) $\forall x\in\mathcal{X}_\Gamma,\;Q(x)\neq\mathsf{ABSTAIN}$,

-   (resource bound) runtime$_Q(x)\le \mathcal{C}_\Gamma(|{\rm enc}_{E_\Gamma}(x)|)$ for all $x\in\mathcal{X}_\Gamma$.
:::

::: corollary
[]{#cor:integrity-universal label="cor:integrity-universal"} Let $M$ be any deterministic program, viewed as a certifying solver pair $(Q,V)$ for some externally fixed target relation $\mathcal{R}$. Then Definition [\[def:solver-integrity\]](#def:solver-integrity){reference-type="ref" reference="def:solver-integrity"} applies unchanged. Thus epistemic integrity is architecture-universal over computing systems once they are cast as solvers for declared tasks.
:::

::: proof
*Proof.* Definition [\[def:solver-integrity\]](#def:solver-integrity){reference-type="ref" reference="def:solver-integrity"} quantifies over a relation $\mathcal{R}$ and a certifying pair $(Q,V)$; it does not assume any special implementation substrate. ◻
:::

\[D:Pprop:universal-solver-framing;Ccor:integrity-universal; R:TR\]

::: remark
[]{#rem:external-task-nonvacuity label="rem:external-task-nonvacuity"} If the target relation is chosen post hoc from the program's own behavior (for example, $\mathcal{R}=\mathcal{R}_M$ from Definition [\[def:induced-solver-relation\]](#def:induced-solver-relation){reference-type="ref" reference="def:induced-solver-relation"}), integrity can become tautological and competence claims can be vacuous. Non-vacuous competence claims therefore require the target relation, encoding regime, and resource bound to be declared independently of observed outputs. \[D:Ddef:solver-integrity, def:competence-regime;Pprop:integrity-competence-separation, prop:integrity-resource-bound; R:AR\]
:::

::: definition
[]{#def:epsilon-objective-family label="def:epsilon-objective-family"} An *$\varepsilon$-objective relation family* is a map $$\varepsilon \mapsto \mathcal{R}_\varepsilon \subseteq \mathcal{X}\times\mathcal{Y}$$ that assigns a certified target relation to each tolerance level $\varepsilon\ge 0$. The exact target is $\mathcal{R}_0$.
:::

::: definition
[]{#def:epsilon-competence-regime label="def:epsilon-competence-regime"} Fix a relation family $(\mathcal{R}_\varepsilon)_{\varepsilon\ge 0}$ and regime $\Gamma$. A certifying solver $(Q,V)$ is *$\varepsilon$-competent* on $\Gamma$ if it is competent on $\Gamma$ for relation $\mathcal{R}_\varepsilon$ in the sense of Definition [\[def:competence-regime\]](#def:competence-regime){reference-type="ref" reference="def:competence-regime"}.
:::

::: proposition
[]{#prop:zero-epsilon-competence label="prop:zero-epsilon-competence"} If $\mathcal{R}_0=\mathcal{R}$, then $$\text{$\varepsilon$-competent at }\varepsilon=0
\iff
\text{competent for exact relation }\mathcal{R}.$$ Moreover, $\varepsilon$-competence implies integrity for the corresponding $\mathcal{R}_\varepsilon$.
:::

::: definition
[]{#def:typed-claim-report label="def:typed-claim-report"} For a declared objective family and regime, a solver-side report is one of:

-   $\mathsf{ABSTAIN}$,

-   $\mathsf{EXACT}$ (claiming exact competence/certification),

-   $\mathsf{EPSILON}(\varepsilon)$ (claiming $\varepsilon$-competence/certification).
:::

::: proposition
[]{#prop:typed-claim-admissibility label="prop:typed-claim-admissibility"} Under Definitions [\[def:competence-regime\]](#def:competence-regime){reference-type="ref" reference="def:competence-regime"}, [\[def:epsilon-competence-regime\]](#def:epsilon-competence-regime){reference-type="ref" reference="def:epsilon-competence-regime"}, and [\[def:typed-claim-report\]](#def:typed-claim-report){reference-type="ref" reference="def:typed-claim-report"}:

-   $\mathsf{ABSTAIN}$ is always admissible;

-   $\mathsf{EXACT}$ is admissible iff exact competence holds;

-   $\mathsf{EPSILON}(\varepsilon)$ is admissible iff $\varepsilon$-competence holds.

Hence claim type and certificate type must match (no cross-typing of uncertified assertions as certified guarantees).
:::

::: definition
[]{#def:evidence-object label="def:evidence-object"} For declared $(\mathcal{R},(\mathcal{R}_\varepsilon)_{\varepsilon\ge 0},\Gamma,Q)$ and report type $r\in\{\mathsf{ABSTAIN},\mathsf{EXACT},\mathsf{EPSILON}(\varepsilon)\}$, *evidence* is a first-class witness object:

-   for $\mathsf{ABSTAIN}$: trivial witness;

-   for $\mathsf{EXACT}$: exact-competence witness;

-   for $\mathsf{EPSILON}(\varepsilon)$: $\varepsilon$-competence witness.
:::

::: proposition
[]{#prop:evidence-admissibility-equivalence label="prop:evidence-admissibility-equivalence"} For any declared contract and report type $r$: $$\text{Evidence object for }r\ \text{exists}
\iff
\text{Claim }r\ \text{is admissible}.$$
:::

\[D:Ddef:evidence-object;Pprop:typed-claim-admissibility, prop:evidence-admissibility-equivalence; R:AR\]

#### Security-game reading (derived).

The typed-report layer can be read as a standard claim-verification game under a declared contract: a reporting algorithm emits a report token $r$ and optional evidence, and a checker accepts if and only if the evidence is valid for $r$. Within this game view, existing results already provide the core security properties: *completeness* for admissible report classes (Proposition [\[prop:evidence-admissibility-equivalence\]](#prop:evidence-admissibility-equivalence){reference-type="ref" reference="prop:evidence-admissibility-equivalence"}), *soundness* against overclaiming (Propositions [\[prop:typed-claim-admissibility\]](#prop:typed-claim-admissibility){reference-type="ref" reference="prop:typed-claim-admissibility"}, [\[prop:exact-requires-evidence\]](#prop:exact-requires-evidence){reference-type="ref" reference="prop:exact-requires-evidence"}), and *evidence-gated confidence* (Propositions [\[prop:certified-confidence-gate\]](#prop:certified-confidence-gate){reference-type="ref" reference="prop:certified-confidence-gate"}, [\[prop:no-evidence-zero-certified\]](#prop:no-evidence-zero-certified){reference-type="ref" reference="prop:no-evidence-zero-certified"}). So integrity is enforced as verifiable claim discipline, not as unchecked payload declaration. \[D:Pprop:typed-claim-admissibility, prop:evidence-admissibility-equivalence, prop:exact-requires-evidence, prop:certified-confidence-gate, prop:no-evidence-zero-certified; R:AR\]

::: definition
[]{#def:signaled-typed-report label="def:signaled-typed-report"} A *signaled typed report* augments the typed report token $r\in\{\mathsf{ABSTAIN},\mathsf{EXACT},\mathsf{EPSILON}(\varepsilon)\}$ with:

-   an optional guess payload $g$,

-   self-reflected confidence $p_{\mathrm{self}}\in[0,1]$,

-   certified confidence $p_{\mathrm{cert}}\in[0,1]$.

-   scalar execution witness $t_{\mathrm{run}}\in\mathbb{N}$ (steps actually run),

-   optional declared bound $B\in\mathbb{N}$.

Here $p_{\mathrm{self}}$ is a self-signal, while $p_{\mathrm{cert}}$ is an evidence-gated signal under the declared contract.
:::

::: definition
[]{#def:signal-consistency label="def:signal-consistency"} For a signaled typed report $(r,g,p_{\mathrm{self}},p_{\mathrm{cert}},t_{\mathrm{run}},B)$, require: $$p_{\mathrm{cert}} > 0 \Rightarrow \exists\ \text{evidence object for }r.$$
:::

::: proposition
[]{#prop:exact-requires-evidence label="prop:exact-requires-evidence"} In the typed claim discipline, $$\text{ClaimAdmissible}(\mathsf{EXACT})
\iff
\exists\ \text{exact evidence object}.$$ Equivalently, admissible exact claims are exactly those with an evidence witness.
:::

::: definition
[]{#def:completion-fraction label="def:completion-fraction"} Completion-fraction semantics are defined only when a positive declared bound exists and covers observed runtime: $$\mathrm{CompletionFractionDefined}
\iff
\exists b>0:\ B=b\ \wedge\ t_{\mathrm{run}}\le b.$$
:::

::: proposition
[]{#prop:certified-confidence-gate label="prop:certified-confidence-gate"} Under Definition [\[def:signal-consistency\]](#def:signal-consistency){reference-type="ref" reference="def:signal-consistency"}, positive certified confidence implies typed admissibility: $$p_{\mathrm{cert}} > 0 \Rightarrow \text{ClaimAdmissible}(r).$$ Thus certified confidence is not a free scalar; it is evidence-gated by the same typed discipline as the report itself.
:::

::: proposition
[]{#prop:no-evidence-zero-certified label="prop:no-evidence-zero-certified"} Under Definition [\[def:signal-consistency\]](#def:signal-consistency){reference-type="ref" reference="def:signal-consistency"}, absence of evidence for the reported type forces: $$\neg\exists\ \text{evidence object for }r
\Rightarrow
p_{\mathrm{cert}}=0.$$
:::

::: corollary
[]{#cor:exact-no-competence-zero-certified label="cor:exact-no-competence-zero-certified"} For exact reports, if exact competence is unavailable on the declared regime/objective, then any signal-consistent report must satisfy: $$p_{\mathrm{cert}} = 0.$$
:::

::: proposition
[]{#prop:steps-run-scalar label="prop:steps-run-scalar"} For every signaled report, the execution witness is an exact natural number and any equality claim about it is decidable: $$\exists k\in\mathbb{N}:\ t_{\mathrm{run}}=k,
\qquad
\forall k\in\mathbb{N},\ (t_{\mathrm{run}}=k)\ \vee\ (t_{\mathrm{run}}\neq k).$$
:::

::: proposition
[]{#prop:no-fraction-without-bound label="prop:no-fraction-without-bound"} If no declared bound is provided ($B=\varnothing$), completion-fraction semantics are undefined: $$B=\varnothing \Rightarrow \neg\,\mathrm{CompletionFractionDefined}.$$
:::

::: proposition
[]{#prop:fraction-defined-under-bound label="prop:fraction-defined-under-bound"} If a positive declared bound is provided and observed runtime is within bound, completion-fraction semantics are defined: $$B=b,\ b>0,\ t_{\mathrm{run}}\le b
\Rightarrow
\mathrm{CompletionFractionDefined}.$$
:::

::: proposition
[]{#prop:abstain-guess-self-signal label="prop:abstain-guess-self-signal"} For any optional guess payload $g$ and any self-reflected confidence $p_{\mathrm{self}}\in[0,1]$, there exists a signal-consistent abstain report $$(\mathsf{ABSTAIN}, g, p_{\mathrm{self}}, 0, 0, \varnothing).$$ Hence the framework is non-binary at the report-signal layer: abstention can carry a tentative answer and self-reflection without upgrading to certified exactness.
:::

::: proposition
[]{#prop:self-confidence-not-certification label="prop:self-confidence-not-certification"} Self-reflected confidence alone does not certify exactness: if exact competence is unavailable, there exist exact-labeled reports with arbitrary $p_{\mathrm{self}}$ that remain inadmissible under zero certified confidence.
:::

\[D:Ddef:signaled-typed-report, def:signal-consistency, def:completion-fraction;Pprop:exact-requires-evidence, prop:certified-confidence-gate, prop:no-evidence-zero-certified, prop:steps-run-scalar, prop:no-fraction-without-bound, prop:fraction-defined-under-bound, prop:abstain-guess-self-signal, prop:self-confidence-not-certification;Ccor:exact-no-competence-zero-certified; R:AR\]

::: definition
[]{#def:declared-budget-slice label="def:declared-budget-slice"} For a declared regime $\Gamma$ over state objects and horizon/budget cut $H\in\mathbb{N}$, define the in-scope slice $$\mathcal{S}_{\Gamma,H}
:=
\{s:\ s\in\Gamma.\mathrm{inScope}\ \wedge\ \mathrm{encLen}_\Gamma(s)\le H\}.$$
:::

::: proposition
[]{#prop:bounded-slice-meta-irrelevance label="prop:bounded-slice-meta-irrelevance"} Fix coordinate $i_\infty$ and declared slice $\mathcal{S}_{\Gamma,H}$ from Definition [\[def:declared-budget-slice\]](#def:declared-budget-slice){reference-type="ref" reference="def:declared-budget-slice"}. If optimizer sets are invariant to $i_\infty$ on that slice, i.e., $$\forall s,s'\in\mathcal{S}_{\Gamma,H},\ 
\Big(\forall j\neq i_\infty,\ s_j=s'_j\Big)
\Rightarrow
\operatorname{Opt}(s)=\operatorname{Opt}(s'),$$ then $i_\infty$ is irrelevant on $\mathcal{S}_{\Gamma,H}$, and hence not relevant for the declared task slice.
:::

\[D:Ddef:declared-budget-slice;Pprop:bounded-slice-meta-irrelevance; R:AR\]

::: definition
[]{#def:goal-class label="def:goal-class"} A *goal class* is a non-empty set of admissible evaluators over a fixed action/state structure: $$\mathcal{G}=(\mathcal{U}_{\mathcal{G}},A,S),\qquad
\varnothing\neq\mathcal{U}_{\mathcal{G}}\subseteq\{U:A\times S\to\mathbb{Q}\}.$$ The solver need not know which $U\in\mathcal{U}_{\mathcal{G}}$ is active; it only knows class-membership constraints.
:::

::: definition
[]{#def:interior-pareto-dominance label="def:interior-pareto-dominance"} For a goal class $\mathcal{G}$ and score model $\sigma:S\times\{1,\ldots,n\}\to\mathbb{Q}$, let $J_{\mathcal{G}}$ be the class-tautological coordinate set. State $s$ *interior-Pareto-dominates* $s'$ on $J_{\mathcal{G}}$ when: $$\forall j\in J_{\mathcal{G}},\ \sigma(s',j)\le \sigma(s,j)
\quad\text{and}\quad
\exists j\in J_{\mathcal{G}},\ \sigma(s',j)<\sigma(s,j).$$
:::

::: proposition
[]{#prop:interior-universal-non-rejection label="prop:interior-universal-non-rejection"} If a state $s$ interior-Pareto-dominates $s'$ on a checked coordinate set $J$, and every admissible evaluator in the goal class is monotone on $J$, then no admissible evaluator strictly prefers $s'$ over $s$ on that checked slice: $$\forall U\in\mathcal{U}_{\mathcal{G}},\ \forall a\in A,\ U(a,s')\le U(a,s).$$
:::

::: proposition
[]{#prop:interior-verification-tractable label="prop:interior-verification-tractable"} \[D:Pprop:interior-verification-tractable; R:H=tractable-structured\] If membership in the checked coordinate set is decidable and interior dominance is decidable, then interior verification yields a polynomial-time checker with exact acceptance criterion: $$\exists\ \mathrm{verify}:S\times S\to\{0,1\},\quad
\mathrm{verify}(s,s')=1\iff \text{interior-dominance}(s,s').$$
:::

::: proposition
[]{#prop:interior-one-sidedness label="prop:interior-one-sidedness"} Interior dominance is one-sided and does not imply full coordinate sufficiency: a counterexample pair outside the checked slice can still violate global optimizer equivalence.
:::

::: corollary
[]{#cor:interior-singleton-certificate label="cor:interior-singleton-certificate"} \[D:Ccor:interior-singleton-certificate; R:H=tractable-structured\] For a singleton checked coordinate $i$, strict improvement on $i$ with class-monotonicity on $\{i\}$ is already a valid interior certificate of non-rejection.
:::

\[D:Ddef:goal-class, def:interior-pareto-dominance;Pprop:interior-universal-non-rejection, prop:interior-verification-tractable, prop:interior-one-sidedness;Ccor:interior-singleton-certificate; R:AR\]

::: definition
[]{#def:rlff-objective label="def:rlff-objective"} Given report type $r\in\{\mathsf{ABSTAIN},\mathsf{EXACT},\mathsf{EPSILON}(\varepsilon)\}$ and declared contract, RLFF assigns base reward by report type when evidence exists and applies an explicit inadmissibility penalty otherwise: $$\mathrm{Reward}(r)=
\begin{cases}
\mathrm{Base}(r), & \text{if evidence for }r\text{ exists},\\
-\mathrm{Penalty}, & \text{otherwise}.
\end{cases}$$
:::

::: proposition
[]{#prop:rlff-maximizer-admissible label="prop:rlff-maximizer-admissible"} If $\mathsf{ABSTAIN}$ reward strictly exceeds the inadmissibility branch, then any global RLFF maximizer must be evidence-backed and therefore admissible under the typed-claim discipline.
:::

::: corollary
[]{#cor:rlff-abstain-no-certs label="cor:rlff-abstain-no-certs"} If $\mathsf{EXACT}$ and $\mathsf{EPSILON}(\varepsilon)$ have no evidence objects while $\mathsf{ABSTAIN}$ beats the inadmissibility branch, then RLFF strictly prefers $\mathsf{ABSTAIN}$ to both report types.
:::

\[D:Ddef:rlff-objective;Pprop:rlff-maximizer-admissible;Ccor:rlff-abstain-no-certs; R:AR\]

::: definition
[]{#def:certainty-inflation label="def:certainty-inflation"} For a typed report $r$, certainty inflation is the state of emitting $r$ without a matching evidence object.
:::

::: proposition
[]{#prop:certainty-inflation-iff-inadmissible label="prop:certainty-inflation-iff-inadmissible"} For every typed report $r$: $$\text{CertaintyInflation}(r)
\iff
\neg\ \text{ClaimAdmissible}(r).$$ For $r=\mathsf{EXACT}$ this is equivalently failure of exact competence.
:::

::: corollary
[]{#cor:hardness-exact-certainty-inflation label="cor:hardness-exact-certainty-inflation"} Under the declared hardness/non-collapse premises that block universal exact admissibility, every exact report is certainty-inflated.
:::

\[D:Ddef:certainty-inflation;Pprop:certainty-inflation-iff-inadmissible;Ccor:hardness-exact-certainty-inflation; R:CR\]

::: definition
[]{#def:raw-certified-bits label="def:raw-certified-bits"} Fix a declared contract and report type $r$. A report-bit model declares:

-   raw report bits $B_{\mathrm{raw}}(r)$ (payload bits in the asserted report token),

-   certificate bits $B_{\mathrm{cert}}(r)$ (bits allocated to the matching evidence class).

Evidence-gated overhead and certified total are then: $$B_{\mathrm{over}}(r)=
\begin{cases}
B_{\mathrm{cert}}(r), & \text{if evidence for }r\text{ exists},\\
0, & \text{otherwise},
\end{cases}
\qquad
B_{\mathrm{total}}(r)=B_{\mathrm{raw}}(r)+B_{\mathrm{over}}(r).$$
:::

::: proposition
[]{#prop:raw-certified-bit-split label="prop:raw-certified-bit-split"} For every typed report $r$: $$B_{\mathrm{total}}(r)=B_{\mathrm{raw}}(r)+B_{\mathrm{over}}(r),$$ with $$\neg\exists\ \text{evidence for }r \Rightarrow B_{\mathrm{over}}(r)=0,\qquad
\exists\ \text{evidence for }r \Rightarrow B_{\mathrm{over}}(r)=B_{\mathrm{cert}}(r).$$ Hence $B_{\mathrm{raw}}(r)\le B_{\mathrm{total}}(r)$ always, and strict inequality holds when evidence exists with positive certificate-bit allocation.
:::

::: theorem
[]{#thm:exact-certified-gap-iff-admissible label="thm:exact-certified-gap-iff-admissible"} Under a report-bit model with positive exact certificate-bit allocation: $$\text{ClaimAdmissible}(\mathsf{EXACT})
\iff
B_{\mathrm{raw}}(\mathsf{EXACT})<B_{\mathrm{total}}(\mathsf{EXACT}).$$ Equivalently: $$B_{\mathrm{total}}(\mathsf{EXACT})=B_{\mathrm{raw}}(\mathsf{EXACT})
\iff
\text{ExactCertaintyInflation}.$$
:::

::: corollary
[]{#cor:finite-budget-no-exact-admissibility label="cor:finite-budget-no-exact-admissibility"} \[D:Ccor:finite-budget-no-exact-admissibility; R:H=exp-lb-conditional\] Fix a declared physical computer model with finite budget and positive per-bit cost, a declared operation-requirement lower-bound family, and a declared typed-report bit model. If:

1.  exact admissibility implies certified-bit footprint at least the declared operation requirement at size $n$, and

2.  exact admissibility implies physical feasibility of that certified footprint under the declared budget model,

then there exists $n_0$ such that for all $n \ge n_0$, exact admissibility fails. The same conclusion holds in the canonical exponential-requirement specialization.
:::

::: corollary
[]{#cor:hardness-raw-only-exact label="cor:hardness-raw-only-exact"} If exact reporting is inadmissible on the declared contract, then exact accounting collapses to raw-only: $$B_{\mathrm{total}}(\mathsf{EXACT})=B_{\mathrm{raw}}(\mathsf{EXACT}).$$
:::

\[D:Ddef:raw-certified-bits;Tthm:exact-certified-gap-iff-admissible;Pprop:raw-certified-bit-split;Ccor:hardness-raw-only-exact; R:AR\]

::: corollary
[]{#cor:no-uncertified-exact-claim label="cor:no-uncertified-exact-claim"} If exact competence does not hold on the declared regime/objective, then an $\mathsf{EXACT}$ report is inadmissible.
:::

\[D:Pprop:typed-claim-admissibility;Ccor:no-uncertified-exact-claim; R:AR\]

::: proposition
[]{#prop:declared-contract-selection-validity label="prop:declared-contract-selection-validity"} For solver-side reporting, there are two formally distinct layers:

-   **Declared-contract selection layer:** choosing objective family $(\mathcal{R}_\varepsilon)_{\varepsilon\ge 0}$, regime $\Gamma$, and assumption profile;

-   **Validity layer:** once declared, admissibility of $\mathsf{EXACT}/\mathsf{EPSILON}(\varepsilon)/\mathsf{ABSTAIN}$ reports is fixed by certificate-typed rules.

Thus the framework treats contract selection as an external declaration choice, and then enforces mechanically checked admissibility within that declared contract.
:::

::: proof
*Proof.* The validity layer is Proposition [\[prop:typed-claim-admissibility\]](#prop:typed-claim-admissibility){reference-type="ref" reference="prop:typed-claim-admissibility"}: report admissibility is exactly tied to the matching competence certificate type. For exact physical claims outside carve-outs, Proposition [\[prop:outside-excuses-explicit-assumptions\]](#prop:outside-excuses-explicit-assumptions){reference-type="ref" reference="prop:outside-excuses-explicit-assumptions"} and Theorem [\[thm:claim-integrity-meta\]](#thm:claim-integrity-meta){reference-type="ref" reference="thm:claim-integrity-meta"} require explicit assumptions, and Corollary [\[cor:outside-excuses-no-exact-report\]](#cor:outside-excuses-no-exact-report){reference-type="ref" reference="cor:outside-excuses-no-exact-report"} blocks uncertified exact reports under declared non-collapse assumptions. Therefore, once the contract is declared, admissibility is mechanically determined by the typed rules and attached assumption profile. ◻
:::

\[D:Tthm:claim-integrity-meta;Pprop:typed-claim-admissibility, prop:outside-excuses-explicit-assumptions;Ccor:outside-excuses-no-exact-report; R:DC\]

::: definition
[]{#def:exact-claim-excuses label="def:exact-claim-excuses"} For exact physical claims in this framework, we declare four explicit carve-outs:

1.  trivial scope (empty in-scope family),

2.  tractable class (exact competence available in the active regime),

3.  stronger oracle/regime shift (the claim is made under strictly stronger access assumptions),

4.  unbounded resources (budget bounds intentionally removed).
:::

::: definition
[]{#def:explicit-hardness-profile label="def:explicit-hardness-profile"} An explicit hardness-assumption profile for declared class/regime $(\mathcal{R},\Gamma)$ consists of:

-   a non-collapse hypothesis,

-   a collapse implication from universal exact competence on $(\mathcal{R},\Gamma)$.
:::

::: definition
[]{#def:exact-claim-welltyped label="def:exact-claim-welltyped"} An exact physical claim is *well-typed* only if either:

-   at least one carve-out from Definition [\[def:exact-claim-excuses\]](#def:exact-claim-excuses){reference-type="ref" reference="def:exact-claim-excuses"} is explicitly declared, or

-   an explicit hardness-assumption profile (Definition [\[def:explicit-hardness-profile\]](#def:explicit-hardness-profile){reference-type="ref" reference="def:explicit-hardness-profile"}) is explicitly declared.
:::

::: proposition
[]{#prop:outside-excuses-explicit-assumptions label="prop:outside-excuses-explicit-assumptions"} If an exact physical claim is well-typed (Definition [\[def:exact-claim-welltyped\]](#def:exact-claim-welltyped){reference-type="ref" reference="def:exact-claim-welltyped"}) and no carve-out from Definition [\[def:exact-claim-excuses\]](#def:exact-claim-excuses){reference-type="ref" reference="def:exact-claim-excuses"} applies, then an explicit hardness-assumption profile must be present.
:::

::: theorem
[]{#thm:claim-integrity-meta label="thm:claim-integrity-meta"} For declared class/regime objects, every admissible exact-claim policy outside the carve-outs requires explicit assumptions: $$\begin{aligned}
&\neg\,\text{Excused (Definition~\ref{def:exact-claim-excuses})}
\ \wedge\
\text{Well-Typed Exact Claim (Definition~\ref{def:exact-claim-welltyped})} \\
&\Rightarrow\
\text{Has Explicit Hardness Profile (Definition~\ref{def:explicit-hardness-profile})}.
\end{aligned}$$
:::

::: proof
*Proof.* Immediate from Proposition [\[prop:outside-excuses-explicit-assumptions\]](#prop:outside-excuses-explicit-assumptions){reference-type="ref" reference="prop:outside-excuses-explicit-assumptions"}; this is the universally quantified theorem-level restatement over the typed claim/regime objects. ◻
:::

::: corollary
[]{#cor:outside-excuses-no-exact-report label="cor:outside-excuses-no-exact-report"} If no carve-out applies and the declared hardness-assumption profile holds, then $\mathsf{EXACT}$ reports are inadmissible for every solver on the declared class/regime.
:::

\[D:Tthm:claim-integrity-meta;Pprop:outside-excuses-explicit-assumptions;Ccor:outside-excuses-no-exact-report; R:CR\]

::: proposition
[]{#prop:integrity-competence-separation label="prop:integrity-competence-separation"} Integrity and competence are distinct predicates: integrity constrains asserted outputs, while competence adds non-abstaining coverage under resource bounds.
:::

::: proof
*Proof.* Take the always-abstain solver $Q_\bot(x)=\mathsf{ABSTAIN}$ with any polynomial-time checker $V$. Definition [\[def:solver-integrity\]](#def:solver-integrity){reference-type="ref" reference="def:solver-integrity"} holds vacuously, so $(Q_\bot,V)$ is integrity-preserving, but it fails Definition [\[def:competence-regime\]](#def:competence-regime){reference-type="ref" reference="def:competence-regime"} whenever $\mathcal{X}_\Gamma\neq\emptyset$ because coverage fails. Hence integrity does not imply competence. The converse is immediate because competence includes integrity as a conjunct. ◻
:::

::: definition
[]{#def:attempted-competence-failure label="def:attempted-competence-failure"} Fix an exact objective under regime $\Gamma$. A solver state is an *attempted competence failure* if:

-   integrity holds (Definition [\[def:solver-integrity\]](#def:solver-integrity){reference-type="ref" reference="def:solver-integrity"}),

-   exact competence was actually attempted for the active scope/objective,

-   competence on $\Gamma$ fails for that exact objective (Definition [\[def:competence-regime\]](#def:competence-regime){reference-type="ref" reference="def:competence-regime"}).
:::

::: proposition
[]{#prop:attempted-competence-matrix label="prop:attempted-competence-matrix"} Let $I,A,C\in\{0,1\}$ denote integrity, attempted exact competence, and competence available in the active regime. Policy verdict for persistent over-specification is:

-   if $I=0$: inadmissible,

-   if $I=1$ and $(A,C)=(1,0)$: conditionally rational,

-   if $I=1$ and $(A,C)\in\{(0,0),(0,1),(1,1)\}$: irrational for the same verified-cost objective.

Hence, in the integrity-preserving plane ($I=1$), exactly one cell is rational and three are irrational.
:::

This separation plus Proposition [\[prop:attempted-competence-matrix\]](#prop:attempted-competence-matrix){reference-type="ref" reference="prop:attempted-competence-matrix"} is load-bearing for the regime-conditional trilemma used later: if exact competence is blocked by hardness in a declared regime after an attempted exact procedure, integrity forces one of three responses: abstain, weaken guarantees, or change regime assumptions. \[D:Pprop:attempted-competence-matrix; R:AR\]

#### Mechanized status.

This separation is machine-checked in `DecisionQuotient/IntegrityCompetence.lean` via: IC18 and IC25; the attempted-competence matrix is mechanized via IC27, IC7, and IC6.

::: proposition
[]{#prop:integrity-resource-bound label="prop:integrity-resource-bound"} Let $\Gamma$ be a declared regime whose in-scope family includes all instances of SUFFICIENCY-CHECK and whose declared resource bound is polynomial in the encoding length. If $P\neq coNP$, then no certifying solver is simultaneously integrity-preserving and competent on $\Gamma$ for exact sufficiency. Equivalently, for every integrity-preserving solver, at least one of the competence conjuncts must fail on $\Gamma$: either full non-abstaining coverage fails or the declared budget bound fails.
:::

::: proof
*Proof.* By Definition [\[def:competence-regime\]](#def:competence-regime){reference-type="ref" reference="def:competence-regime"}, competence on $\Gamma$ requires integrity, full in-scope coverage, and budget compliance. Under the coNP-hardness transfer for SUFFICIENCY-CHECK (Theorem [\[thm:sufficiency-conp\]](#thm:sufficiency-conp){reference-type="ref" reference="thm:sufficiency-conp"}), universal competence on this scope would induce a polynomial-time collapse to $P=coNP$. Therefore, under $P\neq coNP$, the full competence predicate cannot hold. Since integrity alone is satisfiable (Proposition [\[prop:integrity-competence-separation\]](#prop:integrity-competence-separation){reference-type="ref" reference="prop:integrity-competence-separation"}, via always-abstain), the obstruction is exactly competence coverage/budget under the declared regime. \[D:Tthm:sufficiency-conp;Pprop:integrity-competence-separation, prop:integrity-resource-bound; R:AR\] ◻
:::

::: proposition
[]{#prop:physics-no-universal-exact label="prop:physics-no-universal-exact"} Fix any declared physical/task class representable by relation $\mathcal{R}$ and regime $\Gamma$ in the certifying-solver model. If $$\big(\exists Q,\ \mathrm{CompetentOn}(\mathcal{R},\Gamma,Q)\big)\ \Rightarrow\ (P=coNP),$$ then under $P\neq coNP$ there is no universally exact-competent solver for that declared class.
:::

::: corollary
[]{#cor:physics-no-universal-exact-claim label="cor:physics-no-universal-exact-claim"} Under the same premises as Proposition [\[prop:physics-no-universal-exact\]](#prop:physics-no-universal-exact){reference-type="ref" reference="prop:physics-no-universal-exact"}, and under the typed-claim discipline of Definition [\[def:typed-claim-report\]](#def:typed-claim-report){reference-type="ref" reference="def:typed-claim-report"}, an $\mathsf{EXACT}$ report is inadmissible for every solver on that declared class. Hence admissible reporting must use $\mathsf{ABSTAIN}$ or explicitly weakened guarantees.
:::

\[D:Pprop:physics-no-universal-exact;Ccor:physics-no-universal-exact-claim; R:CR\]

Informally: honesty limits what you can claim, capability limits what you can solve, and hard regimes may require abstaining or using weaker exact claims.


# Computational Complexity of Decision-Relevant Uncertainty {#sec:hardness}

This section establishes the computational complexity of information sufficiency, formalized as coordinate sufficiency, for an arbitrary decision problem $\mathcal{D}$ with factored state space. Because the problems below are parameterized by $\mathcal{D}$, and the $(A, S, U)$ tuple captures any problem with choices under structured information (Definition [\[def:decision-problem\]](#def:decision-problem){reference-type="ref" reference="def:decision-problem"}), the hardness results are universal: they apply to every problem with coordinate structure, not to a specific problem domain. We prove three main results:

1.  **SUFFICIENCY-CHECK** is coNP-complete

2.  **MINIMUM-SUFFICIENT-SET** is coNP-complete (the $\Sigma_2^P$ structure collapses)

3.  **ANCHOR-SUFFICIENCY** (fixed coordinates) is $\Sigma_2^P$-complete

Under the model contract of Section [\[sec:model-contract\]](#sec:model-contract){reference-type="ref" reference="sec:model-contract"} and the succinct encoding of Section [\[sec:encoding\]](#sec:encoding){reference-type="ref" reference="sec:encoding"}, these results place exact relevance identification beyond NP-completeness: in the worst case, finding (or certifying) minimal decision-relevant sets is computationally intractable.

#### Reading convention for this section.

Each hardness theorem below is paired with a recovery boundary for the same decision question: when structural-access assumptions in Theorem [\[thm:tractable\]](#thm:tractable){reference-type="ref" reference="thm:tractable"} hold, polynomial-time exact procedures are recovered; when they fail in \[S\], the stated hardness applies.

## Problem Definitions

::: definition
A *decision problem instance* is a tuple $(A, X_1, \ldots, X_n, U)$ where:

-   $A$ is a finite set of alternatives

-   $X_1, \ldots, X_n$ are the coordinate domains, with state space $S = X_1 \times \cdots \times X_n$

-   $U: A \times S \to \mathbb{Q}$ is the utility function (in the succinct encoding, $U$ is given as a Boolean circuit)
:::

#### Succinct/circuit encoding.

When $U$ is given succinctly as a Boolean or arithmetic circuit, the instance size refers to the circuit description rather than the (possibly exponentially large) truth table. Evaluating $U(a,s)$ on a single pair $(a,s)$ can be done in time polynomial in the circuit size, but quantification over all states (for example, \"for all $s$\") ranges over an exponentially large domain and is the source of the hardness results below.

::: definition
For state $s \in S$, define: $$\text{Opt}(s) := \arg\max_{a \in A} U(a, s)$$
:::

::: definition
A coordinate set $I \subseteq \{1, \ldots, n\}$ is *sufficient* if: $$\forall s, s' \in S: \quad s_I = s'_I \implies \text{Opt}(s) = \text{Opt}(s')$$ where $s_I$ denotes the projection of $s$ onto coordinates in $I$.
:::

::: problem
**Input:** Decision problem $(A, X_1, \ldots, X_n, U)$ and coordinate set $I \subseteq \{1,\ldots,n\}$\
**Question:** Is $I$ sufficient?
:::

::: problem
**Input:** Decision problem $(A, X_1, \ldots, X_n, U)$ and integer $k$\
**Question:** Does there exist a sufficient set $I$ with $|I| \leq k$?
:::

## Hardness of SUFFICIENCY-CHECK

Classical coNP-completeness methodology follows standard NP/coNP reduction frameworks and polynomial-time many-one reducibility [@papadimitriou1994complexity; @arora2009computational]. We instantiate that framework for SUFFICIENCY-CHECK with a mechanized TAUTOLOGY reduction.

#### Correctness vs. polynomial-time computability.

A Karp reduction requires two properties: (1) *correctness*---the reduction preserves membership ($\forall x, x \in A \Leftrightarrow f(x) \in B$), and (2) *efficiency*---the reduction is computable in polynomial time. The Lean formalization **machine-verifies** property (1): the theorem `tautology_iff_sufficient` proves that $\varphi$ is a tautology if and only if $I = \emptyset$ is sufficient in the constructed decision problem. Property (2)---polynomial-time TM computability of the reduction function---is a **standard claim** stated as an explicit hypothesis. This separation is standard in formalization: no existing proof assistant project has achieved end-to-end Turing-machine verification of polynomial-time bounds for concrete reductions.

::: theorem
[]{#thm:sufficiency-conp label="thm:sufficiency-conp"} SUFFICIENCY-CHECK is coNP-complete.
:::

::: center
  Source                 Target               Key property preserved
  ---------------------- -------------------- --------------------------------------
  TAUTOLOGY              SUFFICIENCY-CHECK    Tautology iff $\emptyset$ sufficient
  $\exists\forall$-SAT   ANCHOR-SUFFICIENCY   Witness anchors iff formula true
:::

::: proof
*Proof.* extbfMembership in coNP: The complementary problem INSUFFICIENCY is in NP. A nondeterministic verifier can check a witness pair $(s,s')$ as follows:

1.  Verify $s_I = s'_I$ by projecting and comparing coordinates; this takes time polynomial in the input size.

2.  Evaluate $U$ on every alternative for both $s$ and $s'$. When $U$ is given as a circuit each evaluation runs in time polynomial in the circuit size; using these values the verifier checks whether $\text{Opt}(s) \neq \text{Opt}(s')$.

Hence INSUFFICIENCY is in NP and SUFFICIENCY-CHECK is in coNP.

**coNP-hardness:** We reduce from TAUTOLOGY.

Given Boolean formula $\varphi(x_1, \ldots, x_n)$, construct a decision problem with:

-   Alternatives: $A = \{\text{accept}, \text{reject}\}$

-   State space: $S = \{\text{reference}\} \cup \{0,1\}^n$ (equivalently, encode this as a product space with one extra coordinate $r \in \{0,1\}$ indicating whether the state is the reference state)

-   Utility: $$\begin{aligned}
            U(\text{accept}, \text{reference}) &= 1 \\
            U(\text{reject}, \text{reference}) &= 0 \\
            U(\text{accept}, a) &= \varphi(a) \\
            U(\text{reject}, a) &= 0 \quad \text{for assignments } a \in \{0,1\}^n
        
    \end{aligned}$$

-   Query set: $I = \emptyset$

**Claim:** $I = \emptyset$ is sufficient $\iff$ $\varphi$ is a tautology.

($\Rightarrow$) Suppose $I$ is sufficient. Then $\text{Opt}(s)$ is constant over all states. Since $U(\text{accept}, a) = \varphi(a)$ and $U(\text{reject}, a) = 0$:

-   $\text{Opt}(a) = \text{accept}$ when $\varphi(a) = 1$

-   $\text{Opt}(a) = \{\text{accept}, \text{reject}\}$ when $\varphi(a) = 0$

For $\text{Opt}$ to be constant, $\varphi(a)$ must be true for all assignments $a$; hence $\varphi$ is a tautology.

($\Leftarrow$) If $\varphi$ is a tautology, then $U(\text{accept}, a) = 1 > 0 = U(\text{reject}, a)$ for all assignments $a$. Thus $\text{Opt}(s) = \{\text{accept}\}$ for all states $s$, making $I = \emptyset$ sufficient. ◻
:::

#### Immediate recovery boundary (SUFFICIENCY-CHECK).

Theorem [\[thm:sufficiency-conp\]](#thm:sufficiency-conp){reference-type="ref" reference="thm:sufficiency-conp"} is the \[S\] general-case hardness statement. For the same sufficiency relation, Theorem [\[thm:tractable\]](#thm:tractable){reference-type="ref" reference="thm:tractable"} gives polynomial-time recovery under explicit-state, separable, and tree-structured regimes.

#### Mechanized strengthening (all coordinates relevant).

The reduction above establishes coNP-hardness using a single witness set $I=\emptyset$. For the ETH-based lower bound in Theorem [\[thm:dichotomy\]](#thm:dichotomy){reference-type="ref" reference="thm:dichotomy"}, we additionally need worst-case instances where the minimal sufficient set has *linear* size.

We formalized a strengthened reduction in Lean 4: given a Boolean formula $\varphi$ over $n$ variables, construct a decision problem with $n$ coordinates such that if $\varphi$ is not a tautology then *every* coordinate is decision-relevant (so $k^* = n$). Intuitively, the construction places a copy of the base gadget at each coordinate and makes the global "accept" condition hold only when every coordinate's local test succeeds; a single falsifying assignment at one coordinate therefore changes the global optimal set, witnessing that coordinate's relevance. This strengthening is mechanized in Lean; see Appendix [\[app:lean\]](#app:lean){reference-type="ref" reference="app:lean"}.

## Complexity of MINIMUM-SUFFICIENT-SET

::: theorem
[]{#thm:minsuff-conp label="thm:minsuff-conp"} MINIMUM-SUFFICIENT-SET is coNP-complete.
:::

::: proof
*Proof.* **Structural observation:** The $\exists\forall$ quantifier pattern suggests $\Sigma_2^P$: $$\exists I \, (|I| \leq k) \; \forall s, s' \in S: \quad s_I = s'_I \implies \text{Opt}(s) = \text{Opt}(s')$$ However, this collapses because sufficiency has a simple characterization.

**Key lemma:** In the Boolean-coordinate collapse model, a coordinate set $I$ is sufficient if and only if $I$ contains all relevant coordinates (proven formally as `sufficient_iff_relevant_subset` / `sufficient_iff_relevantFinset_subset` in Lean): $$\text{sufficient}(I) \iff \text{Relevant} \subseteq I$$ where $\text{Relevant} = \{i : \exists s, s'.\; s \text{ differs from } s' \text{ only at } i \text{ and } \text{Opt}(s) \neq \text{Opt}(s')\}$. This is the same relevance object as Definition [\[def:relevant\]](#def:relevant){reference-type="ref" reference="def:relevant"}; Proposition [\[prop:minimal-relevant-equiv\]](#prop:minimal-relevant-equiv){reference-type="ref" reference="prop:minimal-relevant-equiv"} gives the minimal-set equivalence in the product-space semantics used by the collapse.

**Consequence:** The minimum sufficient set is exactly the relevant-coordinate set. Thus MINIMUM-SUFFICIENT-SET asks: "Is the number of relevant coordinates at most $k$?"

**coNP membership:** A witness that the answer is NO is a set of $k+1$ coordinates, each proven relevant (by exhibiting $s, s'$ pairs). Verification is polynomial.

**coNP-hardness:** The $k=0$ case asks whether no coordinates are relevant, i.e., whether $\emptyset$ is sufficient. This is exactly SUFFICIENCY-CHECK, which is coNP-complete by Theorem [\[thm:sufficiency-conp\]](#thm:sufficiency-conp){reference-type="ref" reference="thm:sufficiency-conp"}. ◻
:::

#### Immediate recovery boundary (MINIMUM-SUFFICIENT-SET).

Theorem [\[thm:minsuff-conp\]](#thm:minsuff-conp){reference-type="ref" reference="thm:minsuff-conp"} pairs with Theorem [\[thm:minsuff-collapse\]](#thm:minsuff-collapse){reference-type="ref" reference="thm:minsuff-collapse"}: exact minimization complexity is governed by relevance-cardinality once the collapse is applied. Recovery then depends on computing relevance efficiently, with structured-access assumptions from Theorem [\[thm:tractable\]](#thm:tractable){reference-type="ref" reference="thm:tractable"} giving the corresponding route for the underlying sufficiency computations.

## Anchor Sufficiency (Fixed Coordinates)

We also formalize a strengthened variant that fixes the coordinate set and asks whether there exists an *assignment* to those coordinates that makes the optimal action constant on the induced subcube.

::: problem
**Input:** Decision problem $(A, X_1, \ldots, X_n, U)$ and fixed coordinate set $I \subseteq \{1,\ldots,n\}$\
**Question:** Does there exist an assignment $\alpha$ to $I$ such that $\text{Opt}(s)$ is constant for all states $s$ with $s_I = \alpha$?
:::

The second-level hardness template used here follows canonical $\Sigma_2^P$ polynomial-hierarchy characterizations [@papadimitriou1994complexity; @arora2009computational], instantiated to the anchor formulation below.

::: theorem
[]{#thm:anchor-sigma2p label="thm:anchor-sigma2p"} ANCHOR-SUFFICIENCY is $\Sigma_2^P$-complete (already for Boolean coordinate spaces).
:::

::: proof
*Proof.* **Membership in $\Sigma_2^P$:** The problem has the form $$\exists \alpha \;\forall s \in S: \; (s_I = \alpha) \implies \text{Opt}(s) = \text{Opt}(s_\alpha),$$ which is an $\exists\forall$ pattern.

**$\Sigma_2^P$-hardness:** Reduce from $\exists\forall$-SAT. Given $\exists x \, \forall y \, \varphi(x,y)$ with $x \in \{0,1\}^k$ and $y \in \{0,1\}^m$, if $m=0$ we first pad with a dummy universal variable (satisfiability is preserved), construct a decision problem with:

-   Actions $A = \{\text{YES}, \text{NO}\}$

-   State space $S = \{0,1\}^{k+m}$ representing $(x,y)$

-   Utility $$U(\text{YES}, (x,y)) =
          \begin{cases}
            2 & \text{if } \varphi(x,y)=1 \\
            0 & \text{otherwise}
          \end{cases}
        \quad
        U(\text{NO}, (x,y)) =
          \begin{cases}
            1 & \text{if } y = 0^m \\
            0 & \text{otherwise}
          \end{cases}$$

-   Fixed coordinate set $I$ = the $x$-coordinates.

If $\exists x^\star \, \forall y \, \varphi(x^\star,y)=1$, then for any $y$ we have $U(\text{YES})=2$ and $U(\text{NO})\le 1$, so $\text{Opt}(x^\star,y)=\{\text{YES}\}$ is constant. Conversely, fix $x$ and suppose $\exists y_f$ with $\varphi(x,y_f)=0$.

-   If $\varphi(x,0^m)=1$, then $\text{Opt}(x,0^m)=\{\text{YES}\}$. The falsifying assignment must satisfy $y_f\neq 0^m$, where $U(\text{YES})=U(\text{NO})=0$, so $\text{Opt}(x,y_f)=\{\text{YES},\text{NO}\}$.

-   If $\varphi(x,0^m)=0$, then $\text{Opt}(x,0^m)=\{\text{NO}\}$. After padding we have $m>0$, so choose any $y'\neq 0^m$: either $\varphi(x,y')=1$ (then $\text{Opt}(x,y')=\{\text{YES}\}$) or $\varphi(x,y')=0$ (then $\text{Opt}(x,y')=\{\text{YES},\text{NO}\}$). In both subcases the optimal set differs from $\{\text{NO}\}$.

Hence the subcube for this $x$ is not constant. Thus an anchor assignment exists iff the $\exists\forall$-SAT instance is true. ◻
:::

#### Immediate boundary statement (ANCHOR-SUFFICIENCY).

Theorem [\[thm:anchor-sigma2p\]](#thm:anchor-sigma2p){reference-type="ref" reference="thm:anchor-sigma2p"} remains a second-level hardness statement in the anchored formulation; unlike MINIMUM-SUFFICIENT-SET, no general collapse to relevance counting is established here, so the corresponding tractability status remains open in this model.

::: remark
Informally, SUFFICIENCY-CHECK asks whether a given viewpoint works, a verification task ($\forall$), while ANCHOR-SUFFICIENCY asks whether any viewpoint works, a discovery task ($\exists\forall$). The proven complexity gap between coNP and $\Sigma_{2}^{P}$ suggests that finding the right frame is structurally harder than checking a given frame, unless the polynomial hierarchy collapses.
:::

## Tractable Subcases

Despite the general hardness, several natural subcases admit efficient algorithms:

::: proposition
When $|S|$ is polynomial in the input size (i.e., explicitly enumerable), MINIMUM-SUFFICIENT-SET is solvable in polynomial time.
:::

::: proof
*Proof.* Compute $\text{Opt}(s)$ for all $s \in S$. The minimum sufficient set is exactly the set of coordinates that "matter" for the resulting function, computable by standard techniques. ◻
:::

::: proposition
When $U(a, s) = w_a \cdot s$ for weight vectors $w_a \in \mathbb{Q}^n$, MINIMUM-SUFFICIENT-SET reduces to identifying coordinates where weight vectors differ.
:::

## Implications

::: corollary
Under the succinct encoding, exact minimization of sufficient coordinate sets is coNP-hard via the $k=0$ case, and fixed-anchor minimization is $\Sigma_2^P$-complete.
:::

::: proof
*Proof.* The $k=0$ case of MINIMUM-SUFFICIENT-SET is SUFFICIENCY-CHECK (Theorem [\[thm:sufficiency-conp\]](#thm:sufficiency-conp){reference-type="ref" reference="thm:sufficiency-conp"}), giving coNP-hardness for exact minimization. The fixed-anchor variant is exactly Theorem [\[thm:anchor-sigma2p\]](#thm:anchor-sigma2p){reference-type="ref" reference="thm:anchor-sigma2p"}. ◻
:::

The modeling budget for deciding what to model is therefore a computationally constrained resource under this encoding. \[D:thm:sufficiency-conp, thm:anchor-sigma2p; R:\[S\]\]

## The succinct--minimal gap and physical controllers {#sec:neuro-correspondence}

In complexity theory, a *succinct* representation of a combinatorial object is a Boolean circuit of size $\mathrm{poly}(n)$ that computes a function whose explicit (truth-table) form has size $2^{\Theta(n)}$. The term is standard but informal: it denotes a size class (polynomial in input variables), not a structural property. In particular, "succinct" does not mean *minimal*. The minimal circuit for a function is the smallest circuit that computes it, a rigorous, unique object. A succinct circuit may be far larger than the minimal one while still qualifying as "succinct" by the poly-size criterion.

#### The gap.

Let $C$ be a succinct circuit computing a utility function $U$, and let $C^*$ be the minimal circuit for the same function. The *representation gap* $\varepsilon = |C| - |C^*|$ measures how far the controller encoding is from the tightest possible encoding. This gap has no standard formal definition in the literature; it is the unnamed distance between an informal size class and a rigorous structural minimum.

::: proposition
[]{#prop:representation-gap label="prop:representation-gap"} In the Boolean-coordinate model, let $R_{\mathrm{rel}}=\texttt{relevantFinset}(dp)$ and define $$\varepsilon(I)\;:=\;|I\setminus R_{\mathrm{rel}}| + |R_{\mathrm{rel}}\setminus I|.$$ Then: $$\varepsilon(I)=0 \iff I=R_{\mathrm{rel}}
\iff I \text{ is minimal sufficient}.$$
:::

The results of this paper give $\varepsilon$ formal complexity consequences. By Theorem [\[thm:minsuff-conp\]](#thm:minsuff-conp){reference-type="ref" reference="thm:minsuff-conp"}, determining the minimal sufficient coordinate set is coNP-complete under the succinct encoding. By Theorem [\[thm:dichotomy\]](#thm:dichotomy){reference-type="ref" reference="thm:dichotomy"}, under ETH there exist succinct instances requiring $2^{\Omega(n)}$ time to verify sufficiency. In the formal Boolean-coordinate collapse model, size-bounded $\varepsilon=0$ search is exactly the relevance-cardinality predicate S2P2. Together: for a physical decision circuit encoded succinctly, the worst-case cost of determining whether its coordinate attention set is minimal is exponential in the number of input coordinates (under ETH). The gap $\varepsilon$ is *computationally irreducible*: no polynomial-time algorithm solves all succinct instances (assuming $\mathrm{P} \neq coNP$), and the hard instance family requires $2^{\Omega(n)}$ time.

#### Physical systems as circuits.

Any physical system that selects actions based on observed state can be modeled, in the static snapshot, as a circuit: observed inputs are coordinates $X_1, \ldots, X_n$; internal computation maps states to actions; and the quality of that mapping is measured by a utility function $U$. \[D:prop:snapshot-process-typing, prop:bridge-transfer-scope; R:\[bridge-admissible snapshot\]\] The explicit-state encoding (truth table) of this mapping has size exponential in the number of sensory dimensions. No physical finite-memory system can store it. \[D:thm:dichotomy; R:\[E\] vs \[S+ETH\]\] Every realizable physical controller is therefore a succinct encoding, a circuit whose size is bounded by physical resources rather than by the size of the state space it navigates. \[D:thm:dichotomy; R:\[S\]\]

Atomic and molecular systems fit the same form: measured coordinates define observed state, transition operators implement action selection, and utility is objective-conditioned over resulting states. \[D:prop:snapshot-process-typing; R:\[represented adjacent class\]\] In this setting, the physical system is a succinct encoding of a utility-to-action mapping, and the question "is this system attending to the right coordinates?" is exactly SUFFICIENCY-CHECK under the succinct encoding. \[D:thm:config-reduction, thm:sufficiency-conp; R:\[S\]\] The compression that makes a circuit physically realizable, with polynomial size rather than exponential size, is the same compression that makes self-verification exponentially costly in the worst case: the state space the circuit compressed away is precisely the domain that must be exhaustively checked to certify sufficiency. \[D:thm:dichotomy, prop:query-regime-obstruction; R:\[S+ETH\],\[Q_fin\]\]

#### The simplicity tax as $\varepsilon$.

The simplicity tax (Definition [\[def:simplicity-tax\]](#def:simplicity-tax){reference-type="ref" reference="def:simplicity-tax"}) measures $|\mathrm{Gap}(M,P)| = |R(P) \setminus A(M)|$: the number of decision-relevant coordinates that the model does not handle natively. In the $\varepsilon$ decomposition above, this is exactly the unattended-relevant component $|R_{\mathrm{rel}}\setminus I|$ (S2P7, S2P5). For a physical circuit model, this is the coordinate-level manifestation of $\varepsilon$: the circuit attends to a superset of what is necessary, or fails to attend to relevant coordinates, and faces worst-case cost exponential in $n$ to verify which case holds (Theorem [\[thm:sufficiency-conp\]](#thm:sufficiency-conp){reference-type="ref" reference="thm:sufficiency-conp"}). Theorem [\[thm:tax-conservation\]](#thm:tax-conservation){reference-type="ref" reference="thm:tax-conservation"} then says the total relevance budget is conserved: unhandled coordinates are not eliminated, only redistributed as external per-site cost.

#### Consequences for learned circuits.

Large language models and deep networks are succinct circuits with $\mathrm{poly}(n)$ parameters. \[D:thm:dichotomy; R:\[S\]\] Mechanistic interpretability asks which internal components (attention heads, neurons, layers) are necessary for a given capability; this is SUFFICIENCY-CHECK applied to the circuit's internal coordinate structure. Pruning research asks for the minimal subcircuit that preserves performance; this is MINIMUM-SUFFICIENT-SET. The results of this paper imply that these questions are coNP-hard in the worst case under succinct encoding. \[D:thm:sufficiency-conp, thm:minsuff-conp; R:\[S\]\] Empirical progress on interpretability and pruning therefore depends on exploiting the structural properties of specific trained circuits (analogous to the tractable subcases of Theorem [\[thm:tractable\]](#thm:tractable){reference-type="ref" reference="thm:tractable"}), not on solving the general problem. \[D:thm:sufficiency-conp, thm:tractable; R:\[S\] vs \[tractable regimes\]\]

#### Scope and caveats.

The formal results above apply to the static, deterministic model fixed by C1--C4 (Section [\[sec:model-contract\]](#sec:model-contract){reference-type="ref" reference="sec:model-contract"}). Physical systems can add noise, temporal dynamics, and distribution shift. These factors may alter empirical tractability relative to the static worst-case bounds. The bridge conditions of Proposition [\[prop:one-step-bridge\]](#prop:one-step-bridge){reference-type="ref" reference="prop:one-step-bridge"} delineate when static sufficiency results transfer to sequential settings; beyond those conditions, the governing complexity objects change (Propositions [\[prop:bridge-failure-horizon\]](#prop:bridge-failure-horizon){reference-type="ref" reference="prop:bridge-failure-horizon"}--[\[prop:bridge-failure-transition\]](#prop:bridge-failure-transition){reference-type="ref" reference="prop:bridge-failure-transition"}). The claims in this subsection are therefore regime-typed consequences of the formal results, not universal assertions about all possible physical architectures. \[D:prop:one-step-bridge, thm:bridge-boundary-represented, prop:snapshot-process-typing; R:\[represented adjacent class\]\]

## Quantifier Collapse for MINIMUM-SUFFICIENT-SET

::: theorem
[]{#thm:minsuff-collapse label="thm:minsuff-collapse"} The apparent second-level predicate $$\exists I \, (|I| \leq k) \; \forall s,s' \in S:\; s_I = s'_I \implies \operatorname{Opt}(s)=\operatorname{Opt}(s')$$ is equivalent to the coNP predicate $|\text{Relevant}| \le k$, where $$\text{Relevant} = \{i : \exists s,s'.\; s \text{ differs from } s' \text{ only at } i \text{ and } \operatorname{Opt}(s)\neq\operatorname{Opt}(s')\}.$$ Consequently, MINIMUM-SUFFICIENT-SET is governed by coNP certificates rather than a genuine $\Sigma_2^P$ alternation.
:::

::: proof
*Proof.* By the formal equivalence `sufficient_iff_relevant_subset` (finite-set form `sufficient_iff_relevantFinset_subset`), a coordinate set $I$ is sufficient iff $\text{Relevant}\subseteq I$. Therefore: $$\exists I \; (|I|\le k \wedge \text{sufficient}(I))
\iff
\exists I \; (|I|\le k \wedge \text{Relevant}\subseteq I)
\iff
|\text{Relevant}| \le k.$$ So the existential-over-universal presentation collapses to counting the relevant coordinates.

A NO certificate for $|\text{Relevant}| \le k$ is a list of $k+1$ distinct relevant coordinates, each witnessed by two states that differ only on that coordinate and yield different optimal sets; this is polynomially verifiable. Hence the resulting predicate is in coNP, matching Theorem [\[thm:minsuff-conp\]](#thm:minsuff-conp){reference-type="ref" reference="thm:minsuff-conp"}.

This also clarifies why ANCHOR-SUFFICIENCY remains $\Sigma_2^P$-complete: once an anchor assignment is existentially chosen, the universal quantifier over the residual subcube does not collapse to a coordinate-counting predicate. ◻
:::


# Physical Grounding of Discrete Information Acquisition {#sec:physical-grounding}

## Continuous vs Discrete: A Question of Tractability, Not Validity {#sec:continuous-discrete}

Both continuous and discrete models are valid. The choice between them is not about which is "true" but about which is tractable for the computational system using it.

-   **Continuous models** use limits to collapse computational complexity. You don't check all states; you abstract them away. This is a useful thinking tool for mathematical analysis.

-   **Discrete models** are the actual structure that physical computational systems must work with.

The two perspectives are duals: the continuous model is the limit of the discrete model as resolution increases; the discrete model is the physical instantiation that any bounded system must use.

**Question:** Does the continuous abstraction beat physical limits?

No. Limited abstract counting collapses computational complexity in the model, but it does not beat the physical limits of computation speed. No physical decision circuit can compute faster than the speed of light. The continuous abstraction is a cognitive tool for tractable thinking; it does not help a physical computational system that is bounded by $c$.

## First-Principles Derivation: The Counting Gap {#sec:first-principles}

We formalize this with a theorem of pure mathematics. No physical law is assumed.

:::: theorem
[]{#thm:counting-gap label="thm:counting-gap"} Let $\varepsilon, C \in \mathbb{N}$ with $\varepsilon > 0$ and $C > 0$. If each information-acquisition event consumes $\varepsilon$ discrete cost units, then for every $N \in \mathbb{N}$: $$\varepsilon \cdot N \leq C \;\Rightarrow\; N \leq C.$$ Equivalently, $C$ itself is a finite upper bound on the number of possible events. No bounded system with positive integer event cost can perform infinitely many events.

::: proof
*Proof.* This is a discrete theorem. In $\mathbb{N}$, every positive integer is $\geq 1$. Therefore $N = 1 \cdot N \leq \varepsilon \cdot N \leq C$. ◻
:::
::::

**Interpretation.** A continuous process would require infinitely many events. The theorem proves that any bounded computational system with positive event cost has a finite maximum. Physical computational systems are bounded and have positive cost. Therefore physical computational systems cannot use the continuous abstraction to beat their limits.

::: corollary
[]{#cor:forced-finite-speed label="cor:forced-finite-speed"} A bounded region cannot acquire information at unbounded rate. Therefore some finite propagation bound $c_{\max}$ must exist. BA10
:::

## Three Empirical Laws: The Numerical Content {#sec:three-laws}

Theorem [\[thm:counting-gap\]](#thm:counting-gap){reference-type="ref" reference="thm:counting-gap"} proves that a finite bound exists. It does not supply the numerical value. Three empirical laws supply the values in our universe:

1.  **Special Relativity.** No physical circuit computes faster than $c$. Einstein established: $c = 2.998 \times 10^8$ m/s. [@einstein1905electrodynamics; @minkowski1908space]

2.  **Quantum Mechanics.** State transitions are discrete eigenstate jumps. Planck and Dirac established: $E = h\nu$. [@planck1901distribution; @dirac1930principles]

3.  **Thermodynamics (Landauer).** Each event costs positive energy. Landauer established the universal floor: $\varepsilon \geq k_B T \ln 2$ joules per irreversible bit. The Wolpert-style interface used in this artifact adds explicit nonnegative mismatch and residual dissipation terms above that floor; the mismatch term can now be justified inside the artifact from finite-distribution KL divergence when the actual and designed distributions differ, and the finite discrete residual branch can also be justified inside the artifact by an exhaustive reverse-edge split. If the reverse edge flow is positive, asymmetric forward/reverse flows yield a positive two-point KL witness; if the reverse edge flow is zero, the process performs an irreversible one-way state transition, and the existing Landauer-scaled transition-cost theorem yields a positive lower-bound witness. The broader stopping-time / absolute-irreversibility residual regime remains a separate imported premise. When any one of these contributions is positive, the per-bit lower bound is strictly larger than $k_B T \ln 2$. [@landauer1961irreversibility; @bennett1982thermodynamics; @berut2012experimental; @wolpert2024stochastic; @manzano2024absolute; @yadav2026minimal]

The Counting Gap Theorem proves the structure. SR, QM, TD supply the numbers.

## Why Positive Cost is Derived, Not Assumed {#sec:positive-cost-derived}

The premise $\varepsilon > 0$ is not arbitrary. A check is a physical state transition. Physical transitions require force. Force over distance requires energy. At $T > 0$, any irreversible transition carries at least the Landauer floor $k_B T \ln 2$. In the explicit constrained-process interface, any extra dissipation above that floor is represented by separate nonnegative mismatch and residual terms rather than folded into the floor itself. The mismatch branch is now pinned to the existing KL layer: explicit finite-distribution mismatch yields a strictly positive real KL term, which is then conservatively rounded upward into the nat-valued lower-bound units used by the thermodynamic accounting. The finite discrete residual branch is now also theorem-level by an exhaustive local split on the reverse edge of a computational-state process: either the reverse edge is positive and asymmetry yields strictly positive two-point KL divergence, or the reverse edge is zero and the process pays strictly positive Landauer-scaled cost for the resulting irreversible one-way state transition.

If $\varepsilon = 0$, a bounded system could perform infinitely many checks, resolving any decision in zero time and zero energy. This contradicts energy conservation. Therefore $\varepsilon > 0$ is derived from thermodynamics.

::: remark
No claim is made about whether the universe is continuous or discrete at scales below the physical resolution of any bounded system. Whether finer structure exists is irrelevant: it is inaccessible to any bounded region of spacetime at rate $> c/d$, and therefore has no effect on any physical decision process.
:::

## Bounded Information Acquisition Rate {#sec:bounded-acquisition}

::: proposition
[]{#prop:bounded-region label="prop:bounded-region"} A *bounded region* $\mathcal{R}$ is characterized by diameter $d > 0$ and signal propagation speed $c > 0$ (in abstract natural units). The maximum information-acquisition rate of $\mathcal{R}$ is $c/d$ events per unit time.

By **SR**: information emitted from outside $\mathcal{R}$ takes $\geq d/c$ time to reach $\mathcal{R}$. Therefore $\mathcal{R}$ cannot acquire information at rate $> c/d$. BA1
:::

\[D:Ddef:physical-core-encoding; R:AR\]

::: theorem
[]{#thm:bounded-acquisition label="thm:bounded-acquisition"} For bounded region $\mathcal{R}$ with diameter $d$ and signal speed $c$, operating for time $T$: $$\text{acquisitions}(\mathcal{R}, T) \leq \frac{c \cdot T}{d}.$$ This count is finite. $\mathcal{R}$ cannot acquire information continuously. BA2
:::

## Acquisitions are Discrete State Transitions {#sec:discrete-transitions}

::: theorem
[]{#thm:discrete-acquisition label="thm:discrete-acquisition"} By **QM**: the state space of any bounded region with finite energy is countable (discrete energy eigenstates). Each information-acquisition event is a transition $A \to B$ between eigenstates, a discrete event. Between transitions, the state is fixed and no new information is held.

Formally: any physical decision process in $\mathcal{R}$ is a `DiscreteSystem` (Definition in `Physics/DiscreteSpacetime.lean`). Acquisition events are exactly `isTransitionPoint` events. BA3
:::

\[D:Ddef:physical-core-encoding;Pprop:physical-claim-transport; R:AR\]

## One Transition = One Bit {#sec:one-bit}

::: theorem
[]{#thm:boolean-primitive label="thm:boolean-primitive"} Each discrete state transition transfers exactly one boolean bit: the system was in state $A$ (encoded $0$), it transitions to state $B$ (encoded $1$). The encoding $\{A, B\} \cong \{0, 1\}$ is not a modeling choice; it is the form of physical state transitions under **QM**.

Formally: each `isTransitionPoint` contributes exactly $1$ to `bitOperations`. Boolean coordinates are the elementary quanta of physical information exchange. BA4
:::

::: corollary
[]{#cor:finite-state label="cor:finite-state"} A bounded region $\mathcal{R}$ operating for finite time $T$ acquires at most $n_{\max}(\mathcal{R}, T) = \lfloor cT/d \rfloor$ bits. Its accessible state space is a finite subset of $\{0,1\}^{n_{\max}}$. Uncountable state spaces are not accessible to any bounded region in finite time. BA2BA4
:::

## Structural Rank = Minimum Physical Bit Operations {#sec:srank-physical}

::: theorem
[]{#thm:resolution-sufficient label="thm:resolution-sufficient"} Any physical process that correctly determines the optimal action for all states of decision problem $\mathcal{D}$ must access a sufficient coordinate set $I$.

If it accesses fewer coordinates than needed for sufficiency, there exist two states with identical readings but different optimal actions. The resolver outputs the wrong action on one. Correctness forces sufficiency.

Each coordinate in $I$ requires one bit operation to read (one state transition: register samples coordinate $i$ $\to$ one bit acquired by Theorem [\[thm:boolean-primitive\]](#thm:boolean-primitive){reference-type="ref" reference="thm:boolean-primitive"}). Therefore: bit operations $\geq |I|$. BA5
:::

::: theorem
[]{#thm:srank-physical label="thm:srank-physical"} For decision problem $\mathcal{D}$ with $\mathrm{srank}(\mathcal{D}) = k$:

1.  **Information (BA6):** $k = \mathrm{srank}(\mathcal{D}) \leq |I|$ for any sufficient coordinate set $I$, by `srank_le_sufficient_card` (Tractability/StructuralRank.lean). Therefore: minimum bits to resolve $\mathcal{D}$ correctly $= k$.

2.  **Time (BA6 + SR):** Each bit operation requires one tick of duration $\geq d/c$. Minimum decision time $= k \cdot d/c$.

3.  **Energy (BA7 + TD):** Each bit operation carries at least the Landauer floor $k_B T \ln 2$ by **TD**. Therefore every resolver obeys the universal lower bound $E \geq k \cdot k_B T \ln 2$; stronger empirical lower bounds only raise this floor.

Structural rank $k$ is simultaneously an information cost, a time cost, and an energy cost. These are not three separate quantities; they are the same count $k$ expressed in three physical units. BA6BA7
:::

\[D:Tthm:physical-bridge-bundle; R:AR\]

## Thermodynamic Lower Bound as Physical Consequence {#sec:thermo-derived}

:::: theorem
[]{#thm:thermo-derived label="thm:thermo-derived"} For any physical system resolving decision problem $\mathcal{D}$ at temperature $T > 0$: $$E_{\mathrm{decision}} \geq \mathrm{srank}(\mathcal{D}) \cdot k_B T \ln 2.$$

::: proof
*Proof.* Resolving $\mathcal{D}$ requires reading a sufficient coordinate set $I$ (Theorem [\[thm:resolution-sufficient\]](#thm:resolution-sufficient){reference-type="ref" reference="thm:resolution-sufficient"}). Each coordinate read is one bit operation (Theorem [\[thm:boolean-primitive\]](#thm:boolean-primitive){reference-type="ref" reference="thm:boolean-primitive"}). Bit operations $\geq |I| \geq \mathrm{srank}$ (Theorem [\[thm:srank-physical\]](#thm:srank-physical){reference-type="ref" reference="thm:srank-physical"}). By **TD**, each bit operation costs $\geq k_B T \ln 2$. Summing: $E \geq \mathrm{srank} \cdot k_B T \ln 2$. ◻
:::

The constant $k_B T \ln 2$ is the universal Landauer floor for irreversible bit erasure, experimentally verified to $\pm 10\%$ [@berut2012experimental]. It is not, in general, a tight expression for the actual energetic cost of constrained computation [@wolpert2024stochastic; @manzano2024absolute; @yadav2026minimal]. In the Lean model, that distinction is explicit in two layers. First, a coarse floor-plus-overhead interface records that any constrained process may dissipate more than the Landauer floor (WC1--WC5). Second, a refined Wolpert decomposition splits that extra dissipation into a mismatch term and a residual term. Theorem WP1 states that the declared total overhead is exactly the sum of those two components, and WP2 proves that the effective per-bit lower bound dominates the Landauer floor plus both terms. The mismatch branch is now partially promoted from cited premise to theorem: WM1 shows the KL mismatch term is nonnegative, WM2 identifies the zero case exactly, WM3 gives strict positivity under an explicit finite-distribution mismatch witness, and WM4 carries that strict positivity into the discrete lower-bound units used by the thermodynamic model. Theorems WM5 and WM6 then inject that theorem-level mismatch witness into the Wolpert decomposition, so strict separation above the Landauer floor can now be derived directly for the mismatch branch. A second residual branch is now also theorem-level within the current finite machinery: WR1 proves nonnegativity of the two-point forward/reverse residual divergence, WR2 proves strict positivity under an explicit asymmetric positive edge-flow witness, and WR3 lifts that witness into the discrete lower-bound units of the decomposition. The new theorem WR6 packages the exhaustive finite-state edge split directly: if the reverse edge is positive, the pairwise KL branch applies; if the reverse edge vanishes, the process performs an irreversible one-way state transition and the existing Landauer-scaled transition-cost theorem applies. Theorems WR7 and WR8 then inject that unified finite residual witness into the Wolpert decomposition, yielding direct strict separation above the Landauer floor for that finite discrete subclass. Theorems WR9 and WR10 package the same branch at the process level: any finite computational-state process with at least one such asymmetric forward edge already carries a theorem-level residual witness and therefore strictly exceeds the Landauer floor. The broader stopping-time / absolute- irreversibility residual regime imported from [@manzano2024absolute] remains represented by the separate cited premise WP5; and WP6 shows that any one of the derived mismatch branch, the derived finite residual branch, or the cited broader stopping-time branch is already sufficient to force strict separation above the Landauer floor. For the circuit cost interface of [@yadav2026minimal], WP7 and WP8 prove that a declared structural resource lower-bounded by mismatch further tightens the per-bit and total energy lower bounds. Finally, WP9 proves that the existing physical grounding bundle survives under this refined decomposition. Exact equality is retained only as an optional idealized specialization.

**Informally:** the cited stochastic-thermodynamics papers are not rederived in full inside this artifact. The finite-distribution mismatch branch is now proved as far as the existing KL machinery reaches, and a finite residual asymmetry branch is also proved from the same machinery applied to forward/reverse edge-flow distributions, then repackaged as a single process-level witness theorem for finite computational-state systems. The stronger stopping-time / absolute-irreversibility residual regime remains a separate cited premise. Lean then verifies the exact downstream consequences of composing that mixed theorem-plus-premise interface with the decision-theoretic lower bounds already proved in the core system. BA9WM1WM2WM3WM4WM5WM6WR1WR2WR3WR4WR5WR6WR7WR8WR9WR10WP5WP6
::::

\[D:Tthm:physical-bridge-bundle; R:AR\]

::: corollary
[]{#cor:ground-state label="cor:ground-state"} Among all decision problems, those with $\mathrm{srank} = 1$ have the smallest universal lower bound certified by the structural argument: $E_{\mathrm{decision}} \geq k_B T \ln 2$ per decision. In the exact-calibration idealization this is the corresponding ground-state cost.

Problems with $\mathrm{srank} > 1$ are above that universal floor by at least $(\mathrm{srank} - 1) \cdot k_B T \ln 2$ per decision cycle. This gap is structurally mandatory at the level of lower bounds; stronger empirical per-bit costs only enlarge it. BA8
:::

\[D:Ccor:ground-state; R:AR\]

## Connection to the Formal Model {#sec:physical-to-formal}

The coordinate structure of Definition [\[def:decision-problem\]](#def:decision-problem){reference-type="ref" reference="def:decision-problem"} is justified by Theorems [\[thm:boolean-primitive\]](#thm:boolean-primitive){reference-type="ref" reference="thm:boolean-primitive"} and [\[thm:srank-physical\]](#thm:srank-physical){reference-type="ref" reference="thm:srank-physical"}:

-   The finite coordinate spaces $X_1, \ldots, X_n$ correspond to the discrete state transitions accessible to the physical system (Corollary [\[cor:finite-state\]](#cor:finite-state){reference-type="ref" reference="cor:finite-state"}).

-   The boolean case $X_i = \{0,1\}$ is the elementary case: each coordinate is one physical state transition (Theorem [\[thm:boolean-primitive\]](#thm:boolean-primitive){reference-type="ref" reference="thm:boolean-primitive"}).

-   $\mathrm{srank}(\mathcal{D})$ counts the minimum physical transitions required to resolve $\mathcal{D}$ correctly (Theorem [\[thm:srank-physical\]](#thm:srank-physical){reference-type="ref" reference="thm:srank-physical"}).

-   The thermodynamic lower bound (Theorem [\[thm:thermo-derived\]](#thm:thermo-derived){reference-type="ref" reference="thm:thermo-derived"}) is the physical content of the thermodynamic lift results (ThermodynamicLift.lean, DiscreteSpacetime.lean).

The formal model is substrate-neutral (Section [\[sec:foundations\]](#sec:foundations){reference-type="ref" reference="sec:foundations"}). The physical grounding here establishes that the model accurately describes any physical substrate subject to **SR**, **QM**, and **TD**, which is all matter in this universe under current empirical knowledge.

## Universe Membership via Shared Invariants {#sec:universe-membership}

The phrase "this universe" is not rhetorical. It has precise operational content: two physical systems are in the same universe *iff* they agree on a shared set of measurable invariants within measurement precision.

::: definition
[]{#def:invariant-set label="def:invariant-set"} An *invariant set* $\mathcal{I} = \{c, \hbar, k_B, \ldots\}$ is a collection of physical constants that are:

1.  Measurable by any sufficiently capable observer.

2.  Identical (within precision) for all observers in the same universe.

IA0
:::

::: theorem
[]{#thm:universe-membership label="thm:universe-membership"} Two observers $O_1, O_2$ are in the same physical universe *iff* their measurements of a shared invariant set $\mathcal{I}$ are compatible within measurement precision.

**Contrapositive:** Without a shared measurable invariant, the phrase "same universe" has no operational content. IA15
:::

This theorem is not about humans. It is grounded at the quantum level where no ego exists, then lifted to observers as a corollary:

1.  **Quantum level (IA2--IA4):** Two quantum systems under the same Hamiltonian agree on all eigenvalues. There is no "electron perspective." Agreement is forced by physics.

2.  **Atomic level (IA5--IA8):** Two atoms absorbing the same photon agree on $\hbar\omega$. Discrete transitions force exact agreement.

3.  **Molecular level (IA9--IA12):** Two molecules in thermal equilibrium at temperature $T$ agree on $k_B T$. The Landauer floor then applies universally to irreversible bit erasure.

4.  **Observer level (IA13--IA18):** Observers are made of atoms and molecules. They inherit invariant agreement from their constituents. By the time ego enters, agreement is already forced.

::: corollary
[]{#cor:ego-trap label="cor:ego-trap"} Objecting to observer-level invariant agreement requires rejecting atomic-level invariant agreement, which in turn requires rejecting the underlying quantum-level agreement structure. IA17
:::

Therefore: "all matter in this universe" is a technical term. It means: all matter that agrees with us on the invariant set $\{c, \hbar, k_B, \ldots\}$. This is not a rhetorical claim; it is the *definition* of shared universe membership.


# Stochastic Decision Problems {#sec:stochastic}

We extend the static decision problem to include a probability distribution over states.

## Stochastic Decision Problem

::: definition
[]{#def:stochastic-decision-problem label="def:stochastic-decision-problem"} A *stochastic decision problem with coordinate structure* is a tuple $\mathcal{D}_S = (A, X_1, \ldots, X_n, U, P)$ where:

-   $A$ is a finite set of *actions*

-   $X_1, \ldots, X_n$ are finite *coordinate spaces*

-   $S = X_1 \times \cdots \times X_n$ is the *state space*

-   $U : A \times S \to \mathbb{Q}$ is the *utility function*

-   $P \in \Delta(S)$ is a *distribution over states*
:::

::: definition
[]{#def:stochastic-sufficient label="def:stochastic-sufficient"} A coordinate set $I \subseteq \{1, \ldots, n\}$ is *stochastically sufficient* for $\mathcal{D}_S$ if: $$P(s \mid s_I) \otimes U \text{ determines the same optimal action as } P(s) \otimes U.$$ Equivalently, for all $s, s' \in S$ with $s_I = s'_I$: $$\arg\max_{a \in A} \mathbb{E}_{t \sim P(\cdot \mid s_I)}[U(a, t)] = \arg\max_{a \in A} \mathbb{E}_{t \sim P}[U(a, t)].$$
:::

::: definition
[]{#def:expected-value-sufficient label="def:expected-value-sufficient"} A coordinate set $I$ is *expected-value sufficient* if: $$\forall a, a' : \mathbb{E}_{s \sim P}[U(a,s) \mid s_I] \geq \mathbb{E}_{s \sim P}[U(a',s) \mid s_I] \implies a \in \operatorname{Opt}_P.$$ where $\operatorname{Opt}_P = \arg\max_{a \in A} \mathbb{E}_{s \sim P}[U(a,s)]$.
:::

::: decision
[]{#dec:stochastic-sufficiency-check label="dec:stochastic-sufficiency-check"} **STOCHASTIC-SUFFICIENCY-CHECK**: Given stochastic decision problem $\mathcal{D}_S = (A, S, U, P)$ and coordinate set $I$, is $I$ stochastically sufficient?
:::

::: decision
[]{#dec:stochastic-minsuff label="dec:stochastic-minsuff"} **STOCHASTIC-MINIMUM-SUFFICIENT-SET**: Given $\mathcal{D}_S$, find the minimal $I \subseteq \{1, \ldots, n\}$ that is stochastically sufficient.
:::

::: decision
[]{#dec:stochastic-anchor-sufficiency-check label="dec:stochastic-anchor-sufficiency-check"} **STOCHASTIC-ANCHOR-SUFFICIENCY-CHECK**: Given stochastic decision problem $\mathcal{D}_S$ and coordinate set $I$, does there exist an anchor fiber whose conditional-optimal action is singleton and fiber-stable under agreement on $I$?
:::

## Complexity: PP-Completeness

::: theorem
[]{#th:stochastic-sufficiency-pp label="th:stochastic-sufficiency-pp"} STOCHASTIC-SUFFICIENCY-CHECK is PP-complete. \[D:Tth:stochastic-sufficiency-pp; R:PP\]
:::

::: claim
Stochastic anchor refinementprop:stochastic-anchor-refinement If $I$ is stochastically sufficient, then $I$ is stochastically anchor-sufficient. \[D:Pprop:stochastic-anchor-refinement; R:PP\]
:::

::: claim
Strict-side anchor reductionprop:stochastic-anchor-strict-reduction For each formula $\varphi$ on $n \ge 1$ variables: $$\begin{aligned}
\mathrm{StrictMAJSAT}(\varphi)
\iff {} &
\mathrm{STOCHASTIC\allowbreak-\allowbreak ANCHOR\allowbreak-\allowbreak SUFFICIENCY\allowbreak-\allowbreak CHECK}
\bigl(\mathrm{reduceMAJSAT}(\varphi),\emptyset\bigr) \\
& {}\land\;
\operatorname{Opt}_{\mathrm{reduceMAJSAT}(\varphi)}=\{\mathrm{accept}\}.
\end{aligned}$$ \[D:Pprop:stochastic-anchor-strict-reduction; R:PP\]
:::

Observer-collapse closure for the stochastic anchor query is also mechanized: singleton effective-state structure yields global observational equivalence and, with seeded support, yields stochastic sufficiency and anchor-check truth under the declared hypotheses.

1.  **Membership in PP**: Given $\mathcal{D}_S$ and $I$, we must check whether knowing $s_I$ determines the optimal action. This reduces to comparing expected utilities conditioned on $s_I$ versus unconditioned. For each action pair $(a, a')$, we compare: $$\mathbb{E}_{s \sim P}[U(a,s) \mid s_I] \geq \mathbb{E}_{s \sim P}[U(a',s) \mid s_I].$$ This requires counting weighted by $P$, which is in PP.

2.  **PP-hardness via MAJSAT reduction**: Given Boolean formula $\varphi$ on $n$ variables, construct:

    -   $S = \{0,1\}^n$ (state space = assignments)

    -   $P$ = uniform distribution over $S$

    -   $A = \{\text{accept}, \text{reject}\}$

    -   $U(\text{accept}, s) = \varphi(s)$, $U(\text{reject}, s) = 1/2$

    -   $I = \emptyset$

    Then $\emptyset$ is stochastically sufficient iff the optimal action is determined without seeing any coordinates: $$\emptyset \text{ sufficient } \iff \mathbb{E}_{s \sim P}[\varphi(s)] \geq 1/2 \text{ or } < 1/2.$$ This is exactly MAJSAT, which is PP-complete.

::: proof
*Proof.* For membership: We reduce to a PP predicate. For each $a \in A$, compute $\mathbb{E}_P[U(a,s)]$ and compare. The comparison involves summing over $s \in S$ weighted by $P(s)$, which is polynomial in $|S|$ and the representation of $P$. The decision is whether the argmax is unique or whether knowing $s_I$ changes it; this is a majority-vote comparison, hence in PP.

For hardness: The reduction from MAJSAT constructs the uniform distribution. If $\varphi$ is true on $\geq 1/2$ of assignments, the expected utility of accepting is $\geq 1/2$, making accept optimal regardless of state observation. If $< 1/2$, reject is optimal. Thus $\emptyset$ is sufficient exactly when MAJSAT is true. ◻
:::

::: example
Consider stochastic umbrella decision:

-   $A = \{\text{carry}, \text{don't carry}\}$

-   $S = \{\text{rain}, \text{no rain}\}$

-   $P(\text{rain}) = 0.3$, $P(\text{no rain}) = 0.7$

-   $U(\text{carry}, \text{rain}) = 2$, $U(\text{carry}, \text{no rain}) = -1$

-   $U(\text{don't carry}, \text{rain}) = -2$, $U(\text{don't carry}, \text{no rain}) = 0$

Expected utility: $E_P[U(\text{carry})] = 0.3(2) + 0.7(-1) = -0.1$, $E_P[U(\text{don't carry})] = 0.3(-2) + 0.7(0) = -0.6$. So carry is optimal. If we condition on forecast information, we check whether the conditional distribution changes the argmax.
:::

## Potential and Physical Bounds

::: definition
Let $\Phi : A \times S \to \mathbb{R}$ with grounding equation $$U(a,s) = -\Phi(a,s).$$ Define expected action-potential $$\mathbb{E}[\Phi\mid a] := \sum_{s\in S} P(s)\,\Phi(a,s).$$
:::

::: claim
Expected-Utility/Potential Dualityprop:stochastic-potential-duality Under the grounding equation, for every action $a$: $$\mathbb{E}_{s\sim P}[U(a,s)] = -\,\mathbb{E}[\Phi\mid a],$$ and for all $a,a'$: $$\mathbb{E}[U\mid a]\le \mathbb{E}[U\mid a']
\iff
\mathbb{E}[\Phi\mid a']\le \mathbb{E}[\Phi\mid a].$$ \[D:Pprop:stochastic-potential-duality; R:PP\]
:::

::: claim
Landauer Floor and State-Cardinality Specializationprop:stochastic-landauer-floor For $T\ge 0$, the Landauer floor is nonnegative and monotone in erased bits: $$E_{\mathrm{L}}(b,T)\ge 0,\qquad
b_1\le b_2 \implies E_{\mathrm{L}}(b_1,T)\le E_{\mathrm{L}}(b_2,T).$$ At the room-temperature specialization used here: $$\mathrm{thermodynamicCost}(\mathcal{D}_S)=\mathrm{landauerConstant}\cdot |S|.$$ \[D:Pprop:stochastic-landauer-floor; R:AR,H=cost-growth\]
:::

## Tractable Subcases

1.  **Product distributions**: $P = P_1 \otimes \cdots \otimes P_n$. Then $$\mathbb{E}_{s \sim P}[U(a,s) \mid s_I] = \prod_{i \in I} P_i(s_i) \cdot \mathbb{E}_{s \sim P}[U(a,s) \mid s_I],$$ so expected utility decomposes. This reduces stochastic sufficiency to static sufficiency per coordinate.

2.  **Bounded support**: $|\text{supp}(P)| \leq k$. Enumeration is $O(k \cdot |A|)$, polynomial.

3.  **Log-concave distributions**: For ordered state spaces, conditional expectations are monotone and computable via convex optimization.

::: claim
Tractability under product distributionsprop:stochastic-product-tractable If $P = P_1 \otimes \cdots \otimes P_n$ is a product distribution, then STOCHASTIC-SUFFICIENCY-CHECK reduces to checking static sufficiency on each marginal. \[D:Pprop:stochastic-product-tractable; R:PD\]
:::

::: claim
Tractability under bounded supportprop:stochastic-bounded-support If $|\text{supp}(P)| \leq k$, then STOCHASTIC-SUFFICIENCY-CHECK is solvable in $O(k \cdot |A| \cdot n)$ time. \[D:Pprop:stochastic-bounded-support; R:BS\]
:::

## Bridge from Static

Paper 4 proved that static sufficiency does NOT imply stochastic sufficiency (CC54). A counterexample exists where $I$ is sufficient for $(A,S,U)$ but not for $(A,S,U,P)$.

However, under product distributions, static sufficiency transfers:

::: claim
Static to stochastic transferprop:static-stochastic-transfer If $P = P_1 \otimes \cdots \otimes P_n$ is a product distribution and $I$ is statically sufficient for $(A,S,U)$, then $I$ is stochastically sufficient for $(A,S,U,P)$. \[D:Pprop:static-stochastic-transfer; R:PD\]
:::


# Sequential Decision Problems {#sec:sequential}

We further extend to sequential decision-making with transitions and observations.

## Sequential Decision Problem (POMDP)

::: definition
[]{#def:sequential-decision-problem label="def:sequential-decision-problem"} A *sequential decision problem with coordinate structure* is a tuple $\mathcal{D}_{SEQ} = (A, X_1, \ldots, X_n, U, T, O)$ where:

-   $A$ is a finite set of *actions*

-   $X_1, \ldots, X_n$ are finite *coordinate spaces*

-   $S = X_1 \times \cdots \times X_n$ is the *state space*

-   $U : A \times S \to \mathbb{Q}$ is the *stage utility*

-   $T : A \times S \to \Delta(S)$ is the *transition kernel*

-   $O : S \to \Delta(\Omega)$ is the *observation model*

The controller chooses a control law $\pi : (\Omega^*) \to A$ mapping observation histories to actions. Objective: maximize expected cumulative utility $\mathbb{E}\left[\sum_{t=0}^{T-1} \gamma^t U(a_t, s_t)\right]$ with discount factor $\gamma \in [0,1)$.
:::

::: notation
Let $\omega_t = (o_0, o_1, \ldots, o_t)$ denote the observation history up to time $t$. The controller law is $\pi(\omega_{t-1}) = a_t$.
:::

::: definition
[]{#def:sequential-sufficient label="def:sequential-sufficient"} A coordinate set $I \subseteq \{1, \ldots, n\}$ is *sequentially sufficient* for $\mathcal{D}_{SEQ}$ if the optimal policy based on full observation history is the same as the optimal policy based only on coordinates in $I$ at each time step.

Formally: let $\pi^*$ be the optimal policy with full observations, and let $\pi_I^*$ be the optimal policy with observations restricted to $I$-coordinates. Then $\pi^* = \pi_I^*$ (as functions of observation history).
:::

::: decision
[]{#dec:sequential-sufficiency-check label="dec:sequential-sufficiency-check"} **SEQUENTIAL-SUFFICIENCY-CHECK**: Given sequential decision problem $\mathcal{D}_{SEQ}$ and coordinate set $I$, is $I$ sequentially sufficient?
:::

::: decision
[]{#dec:sequential-minsuff label="dec:sequential-minsuff"} **SEQUENTIAL-MINIMUM-SUFFICIENT-SET**: Given $\mathcal{D}_{SEQ}$, find the minimal $I$ that is sequentially sufficient.
:::

::: decision
[]{#dec:sequential-anchor-sufficiency-check label="dec:sequential-anchor-sufficiency-check"} **SEQUENTIAL-ANCHOR-SUFFICIENCY-CHECK**: Given sequential decision problem $\mathcal{D}_{SEQ}$ and coordinate set $I$, does there exist an anchor state whose $I$-agreement class preserves the optimal action set?
:::

## Complexity: PSPACE-Completeness

::: theorem
[]{#th:sequential-sufficiency-pspace label="th:sequential-sufficiency-pspace"} SEQUENTIAL-SUFFICIENCY-CHECK is PSPACE-complete. \[D:Tth:sequential-sufficiency-pspace; R:PSPACE\]
:::

::: claim
Sequential anchor refinementprop:sequential-anchor-refinement If $I$ is sequentially sufficient, then $I$ is sequentially anchor-sufficient. \[D:Pprop:sequential-anchor-refinement; R:PSPACE\]
:::

::: claim
TQBF Reduction to Sequential Anchor Checkprop:sequential-anchor-tqbf-reduction For each quantified Boolean formula $q$: $$\mathrm{TQBF}(q)
\iff
\mathrm{SEQUENTIAL\mbox{-}ANCHOR\mbox{-}SUFFICIENCY\mbox{-}CHECK}(\mathrm{reduceTQBF}(q),\emptyset).$$ \[D:Pprop:sequential-anchor-tqbf-reduction; R:PSPACE\]
:::

Observer-collapse closure for the sequential anchor query is mechanized: under the declared collapse hypothesis, sequential sufficiency and sequential anchor-check truth follow directly.

1.  **Membership in PSPACE**: Optimal POMDP policy computation is in PSPACE(Papadimitriou & Tsitsiklis, 1987). Checking whether a coordinate set $I$ yields the same policy involves comparing value functions, which can be done in polynomial space via alternating polynomial time.

2.  **PSPACE-hardness via TQBF reduction**: Given quantified Boolean formula $\forall x_1 \exists x_2 \forall x_3 \ldots \varphi(x_1, \ldots, x_n)$, construct:

    -   $n$ time steps, one variable per step

    -   At odd steps (universal quantifier): exogenous dynamics choose the variable (adversarial transition)

    -   At even steps (existential quantifier): controller action chooses the variable

    -   Terminal utility: $\varphi(x_1, \ldots, x_n)$

    -   Observation: controller sees all previous variables

    -   $I$ = full observation history

    Then full observation is sequentially sufficient iff the controller can guarantee $\varphi = \text{true}$ regardless of exogenous choices, which holds iff the QBF is true.

::: construction
[]{#constr:tqbf-to-sequential label="constr:tqbf-to-sequential"} Given QBF $\Phi = Q_1 x_1 Q_2 x_2 \ldots Q_n x_n\ \varphi(x_1, \ldots, x_n)$ with alternating quantifiers, construct sequential problem:

-   Time steps $t = 1, \ldots, n$

-   State at step $t$ encodes $(x_1, \ldots, x_t)$

-   At step $t$: - If $Q_t = \forall$: transition is adversarial (exogenous dynamics choose both next state and observation) - If $Q_t = \exists$: transition is controlled by action (choose $x_t$)

-   Terminal reward: 1 if $\varphi(x_1, \ldots, x_n) = \text{true}$, 0 otherwise

-   Full observation reveals all previous variables

Then: full observation is sequentially sufficient $\iff$ there exists a policy guaranteeing $\varphi = \text{true}$ $\iff$ $\Phi$ is true.
:::

::: claim
Sequential sufficiency implies static sufficiencyprop:sequential-static-relation When the horizon $T = 1$ and transitions are deterministic, sequential sufficiency reduces to static sufficiency. \[D:Pprop:sequential-static-relation; R:T=1,DET\]
:::

## Tractable Subcases

1.  **Fully observable (MDP)**: $O$ is the identity (controller sees full state). Then the problem reduces to a stochastic (or deterministic) problem per time step. Value iteration is polynomial in $|S|, |A|, T$.

2.  **Bounded horizon**: $T \leq k$. The state space grows as $O(|S|^T)$, but for fixed $T$ this is polynomial in $|S|$.

3.  **Tree-structured transitions**: The transition graph has no cycles. Dynamic programming yields polynomial algorithms.

4.  **Deterministic transitions**: $T(s'|a,s)$ is a point mass. The problem reduces to search over deterministic paths.

::: claim
Tractability under full observabilityprop:mdp-tractable If $O$ is the identity (fully observable), then SEQUENTIAL-SUFFICIENCY-CHECK reduces to checking stochastic sufficiency at each time step. \[D:Pprop:mdp-tractable; R:FO\]
:::

::: claim
Tractability under bounded horizonprop:sequential-bounded-horizon If $T \leq k$ (constant horizon), then SEQUENTIAL-SUFFICIENCY-CHECK is solvable in time polynomial in $|S|, |A|, k$. \[D:Pprop:sequential-bounded-horizon; R:BH\]
:::

## Bridge from Static and Stochastic

Paper 4 proved that static sufficiency does NOT transfer to sequential (horizon \> 1): CC18

Similarly, stochastic sufficiency does NOT transfer to sequential in general:

::: claim
Stochastic to sequential bridge failureprop:stochastic-sequential-bridge-fail There exists a stochastic sufficient coordinate set $I$ and a sequential problem where $I$ is NOT sequentially sufficient, even when transitions are memoryless. \[D:Pprop:stochastic-sequential-bridge-fail; R:CE\]
:::

The transfer conditions are:

-   Static → Sequential: transfers iff $T = 1$ and transitions are deterministic (one-step bridge from Paper 4)

-   Stochastic → Sequential: transfers iff transitions are memoryless AND observation model is regime-compatible


# Regime Hierarchy {#sec:regime-hierarchy}

We establish the strict inclusion hierarchy among decision regimes.

::: center
            **Inclusion**            **Complexity Classes**  **Condition**
  --------------------------------- ------------------------ --------------------------------------
     Static $\subset$ Stochastic        coNP$\subset$ PP     Standard assumptions (P $\neq$ coNP)
   Stochastic $\subset$ Sequential     PP$\subset$ PSPACE    Standard assumptions (P $\neq$ PP)
:::

::: claim
Strict inclusion static $\subset$ stochasticprop:static-stochastic-strict Under standard complexity assumptions (P $\neq$ coNP), there exist stochastic decision problems that are not statically reducible. \[D:Pprop:static-stochastic-strict; R:P $\neq$ coNP\]
:::

::: claim
Strict inclusion stochastic $\subset$ sequentialprop:stochastic-sequential-strict Under standard complexity assumptions (P $\neq$ PP), there exist sequential decision problems that are not stochastically reducible. \[D:Pprop:stochastic-sequential-strict; R:P $\neq$ PP\]
:::

## Integrity-Resource Bound per Regime

The integrity-resource bound from Paper 4 generalizes to each regime:

1.  **Static** (coNP-hard): polynomial-time physical controller must abstain on some instances

2.  **Stochastic** (PP-hard): polynomial-time physical controller must abstain on more instances

3.  **Sequential** (PSPACE-hard): polynomial-time physical controller must abstain on even more instances

Formally, if $C_1 \subsetneq C_2$ (complexity classes), then the set of instances where integrity forces abstention under $C_1$-resources is a strict superset of those under $C_2$-resources:

$$\{\sigma : \text{integrity forces abstention at } C_1\}
\supsetneq
\{\sigma : \text{integrity forces abstention at } C_2\}$$

::: claim
Abstention frontier shiftsprop:abstention-frontier As regime complexity increases (static $\to$ stochastic $\to$ sequential), the abstention frontier expands. \[D:Pprop:abstention-frontier; R:RG\]
:::

## Regime Detection

A meta-problem: given a problem instance, which regime is it in?

::: decision
[]{#dec:regime-detection label="dec:regime-detection"} **REGIME-DETECTION**: Given a problem instance, is it static, stochastic, or sequential?
:::

Conjecture: regime detection is polynomial for the represented family:

-   Bounded actions $\to$ static

-   Distribution present $\to$ stochastic

-   Transitions/observations present $\to$ sequential


# Encoding-Regime Separation {#sec:dichotomy}

Formally: this section proves regime-indexed access separation for a fixed sufficiency predicate, then lifts bit-operation bounds to thermodynamic/accounting consequences.

The hardness results of Section [\[sec:hardness\]](#sec:hardness){reference-type="ref" reference="sec:hardness"} apply to worst-case instances under the succinct encoding. This section states an encoding-regime separation: an explicit-state upper bound versus a succinct-encoding worst-case lower bound, and an intermediate query-access obstruction family. The complexity class is a property of the encoding measured against a fixed decision question (Remark [\[rem:question-vs-problem\]](#rem:question-vs-problem){reference-type="ref" reference="rem:question-vs-problem"}), not a property of the question itself: the same sufficiency question admits polynomial-time algorithms under one encoding and worst-case exponential cost under another. Within the paper's full regime map, this section is the static access-layer decomposition; theorem-level stochastic and sequential core placements are given in Sections [\[sec:stochastic\]](#sec:stochastic){reference-type="ref" reference="sec:stochastic"} and [\[sec:sequential\]](#sec:sequential){reference-type="ref" reference="sec:sequential"}, with strict inter-regime separation in Section [\[sec:regime-hierarchy\]](#sec:regime-hierarchy){reference-type="ref" reference="sec:regime-hierarchy"}.

Physical reading: the separation is an access-and-budget separation for the same decision semantics. Explicit access assumes full boundary observability; succinct access assumes compressed structure that may require exponential expansion to certify universally; query access assumes bounded measurement interfaces, and the obstruction persists across value-entry/state-batch interfaces with oracle-lattice transfer/strictness closures. \[D:Tthm:dichotomy;Pprop:query-regime-obstruction, prop:query-value-entry-lb, prop:query-state-batch-lb, prop:oracle-lattice-transfer, prop:oracle-lattice-strict, prop:physical-claim-transport; R:E,S+ETH,Qf,Qb,AR\]

#### Model note.

Theorem [\[thm:dichotomy\]](#thm:dichotomy){reference-type="ref" reference="thm:dichotomy"} has Part 1 in \[E\] (time polynomial in $|S|$) and Part 2 in \[S+ETH\] (time exponential in $n$). Proposition [\[prop:query-regime-obstruction\]](#prop:query-regime-obstruction){reference-type="ref" reference="prop:query-regime-obstruction"} is \[Q_fin\] (finite-state Opt-oracle core), while Propositions [\[prop:query-value-entry-lb\]](#prop:query-value-entry-lb){reference-type="ref" reference="prop:query-value-entry-lb"} and [\[prop:query-state-batch-lb\]](#prop:query-state-batch-lb){reference-type="ref" reference="prop:query-state-batch-lb"} are \[Q_bool\] interface refinements (value-entry and state-batch). Relative to Definition [\[def:regime-simulation\]](#def:regime-simulation){reference-type="ref" reference="def:regime-simulation"}, Propositions [\[prop:query-subproblem-transfer\]](#prop:query-subproblem-transfer){reference-type="ref" reference="prop:query-subproblem-transfer"} and [\[prop:oracle-lattice-transfer\]](#prop:oracle-lattice-transfer){reference-type="ref" reference="prop:oracle-lattice-transfer"} are explicit simulation-transfer instances. The \[E\] vs \[S+ETH\] split in Theorem [\[thm:dichotomy\]](#thm:dichotomy){reference-type="ref" reference="thm:dichotomy"} is a same-question encoding/cost-model separation, not a bidirectional simulation claim between those regimes. These regimes are defined in Section [\[sec:encoding\]](#sec:encoding){reference-type="ref" reference="sec:encoding"} and are not directly comparable as functions of one numeric input-length parameter.

::: theorem
[]{#thm:dichotomy label="thm:dichotomy"} Let $\mathcal{D} = (A, X_1, \ldots, X_n, U)$ be a decision problem with $|S| = N$ states. Let $k^*$ be the size of the minimal sufficient set.

1.  **\[E\] Explicit-state upper bound:** Under the explicit-state encoding, SUFFICIENCY-CHECK is solvable in time polynomial in $N$ (e.g. $O(N^2|A|)$).

2.  **\[S+ETH\] Succinct lower bound (worst case):** Assuming ETH, there exists a family of succinctly encoded instances with $n$ coordinates and minimal sufficient set size $k^* = n$ such that SUFFICIENCY-CHECK requires time $2^{\Omega(n)}$.
:::

::: proof
*Proof.* **Part 1 (Explicit-state upper bound):** Under the explicit-state encoding, SUFFICIENCY-CHECK is decidable in time polynomial in $N$ by direct enumeration: compute $\operatorname{Opt}(s)$ for all $s\in S$ and then check all pairs $(s,s')$ with $s_I=s'_I$.

Equivalently, for any fixed $I$, the projection map $s\mapsto s_I$ has image size $|S_I|\le |S|=N$, so any algorithm that iterates over projection classes (or over all state pairs) runs in polynomial time in $N$. Thus, in particular, when $k^*=O(\log N)$, SUFFICIENCY-CHECK is solvable in polynomial time under the explicit-state encoding.

#### Remark (bounded coordinate domains).

In the general model $S=\prod_i X_i$, for a fixed $I$ one always has $|S_I|\le \prod_{i\in I}|X_i|$ (and $|S_I|\le N$). If the coordinate domains are uniformly bounded, $|X_i|\le d$ for all $i$, then $|S_I|\le d^{|I|}$.

**Part 2 (Succinct ETH lower bound, worst case):** A strengthened version of the TAUTOLOGY reduction used in Theorem [\[thm:sufficiency-conp\]](#thm:sufficiency-conp){reference-type="ref" reference="thm:sufficiency-conp"} produces a family of instances in which the minimal sufficient set has size $k^* = n$: given a Boolean formula $\varphi$ over $n$ variables, we construct a decision problem with $n$ coordinates such that if $\varphi$ is not a tautology then *every* coordinate is decision-relevant (thus $k^*=n$). This strengthening is mechanized in Lean (see Appendix [\[app:lean\]](#app:lean){reference-type="ref" reference="app:lean"}). Under the Exponential Time Hypothesis (ETH) [@impagliazzo2001complexity; @arora2009computational], TAUTOLOGY requires time $2^{\Omega(n)}$ in the succinct encoding, so SUFFICIENCY-CHECK inherits a $2^{\Omega(n)}$ worst-case lower bound via this reduction. ◻
:::

::: corollary
There is a clean separation between explicit-state tractability and succinct worst-case hardness (with respect to the encodings in Section [\[sec:encoding\]](#sec:encoding){reference-type="ref" reference="sec:encoding"}):

-   Under the explicit-state encoding, SUFFICIENCY-CHECK is polynomial in $N=|S|$.

-   Under ETH, there exist succinctly encoded instances with $k^* = \Omega(n)$ (indeed $k^*=n$) for which SUFFICIENCY-CHECK requires $2^{\Omega(n)}$ time.

For Boolean coordinate spaces ($N = 2^n$), the explicit-state bound is polynomial in $2^n$ (exponential in $n$), while under ETH the succinct lower bound yields $2^{\Omega(n)}$ time for the hard family in which $k^* = \Omega(n)$.
:::

::: proposition
[]{#prop:query-regime-obstruction label="prop:query-regime-obstruction"} In the finite-state query regime \[Q_fin\], let $S$ be any finite state type with $|S|\ge 2$ and let $Q\subset S$ with $|Q|<|S|$. Then there exist two decision problems $\mathcal{D}_{\mathrm{yes}},\mathcal{D}_{\mathrm{no}}$ over the same state space such that:

-   their oracle views on all states in $Q$ are identical,

-   $\emptyset$ is sufficient for $\mathcal{D}_{\mathrm{yes}}$,

-   $\emptyset$ is not sufficient for $\mathcal{D}_{\mathrm{no}}$.

Consequently, no deterministic query procedure using fewer than $|S|$ state queries can solve SUFFICIENCY-CHECK on all such instances; the worst-case query complexity is $\Omega(|S|)$.
:::

::: proof
*Proof.* This is exactly the finite-state indistinguishability theorem CC4. For any $|Q|<|S|$, there is an unqueried hidden state $s_0$ producing oracle-indistinguishable yes/no instances with opposite truth values on the $I=\emptyset$ subproblem. Since SUFFICIENCY-CHECK contains that subproblem, fewer than $|S|$ queries cannot be correct on both instances, yielding the $\Omega(|S|)$ worst-case lower bound. ◻
:::

::: proposition
[]{#prop:checking-witnessing-duality label="prop:checking-witnessing-duality"} \[D:Pprop:checking-witnessing-duality; R:H=query-lb\] Let $n\ge 1$ and let $P$ be a finite family of Boolean-state pairs. Assume $P$ is sound for refuting empty-set sufficiency in the sense that for every $\operatorname{Opt}: \{0,1\}^n \to \{0,1\}$, whenever $\emptyset$ is not sufficient, some pair in $P$ witnesses disagreement of $\operatorname{Opt}$. Then $$|P| \ge 2^{n-1}.$$ Equivalently, any sound checker must inspect at least the witness budget $W(n)=2^{n-1}$, so checking budget $T(n)$ satisfies $T(n)\ge W(n)$.
:::

::: corollary
[]{#cor:information-barrier-query label="cor:information-barrier-query"} \[D:Ccor:information-barrier-query; R:H=query-lb\] For the fixed sufficiency question, finite query interfaces induce an information barrier:

-   (Opt-oracle): with $|Q|<|S|$, there exist yes/no instances indistinguishable on all queried states but with opposite truth values on the $I=\emptyset$ subproblem.

-   value-entry and state-batch interfaces preserve the same obstruction at cardinality $<2^n$.

Hence the barrier is representational-access based: without querying enough of the hidden state support, exact certification is blocked by indistinguishability rather than by search-strategy choice.
:::

::: corollary
[]{#cor:query-obstruction-bool label="cor:query-obstruction-bool"} \[D:Ccor:query-obstruction-bool; R:H=query-lb\] In the Boolean-coordinate state space $S=\{0,1\}^n$, Proposition [\[prop:query-regime-obstruction\]](#prop:query-regime-obstruction){reference-type="ref" reference="prop:query-regime-obstruction"} yields the familiar $\Omega(2^n)$ worst-case query lower bound for Opt-oracle access.
:::

::: proposition
[]{#prop:query-value-entry-lb label="prop:query-value-entry-lb"} In the mechanized Boolean value-entry query submodel \[Q_bool\], for any deterministic procedure using fewer than $2^n$ value-entry queries $(a,s)\mapsto U(a,s)$, there exist two queried-value-indistinguishable instances with opposite truth values for SUFFICIENCY-CHECK on the $I=\emptyset$ subproblem. Therefore worst-case value-entry query complexity is also $\Omega(2^n)$.
:::

::: proof
*Proof.* The theorem CC50 constructs, for any query set of cardinality $<2^n$, an unqueried hidden state $s_0$ and a yes/no instance pair that agree on all queried values but disagree on $\emptyset$-sufficiency. The auxiliary bound CC50 ensures that fewer than $2^n$ value-entry queries cannot cover all states, so the indistinguishability argument applies in the worst case. ◻
:::

::: proposition
[]{#prop:query-subproblem-transfer label="prop:query-subproblem-transfer"} If every full-problem solver induces a solver for a fixed subproblem, then any lower bound for that subproblem lifts to the full problem.
:::

This is the restriction-map instance of Definition [\[def:regime-simulation\]](#def:regime-simulation){reference-type="ref" reference="def:regime-simulation"}: a solver for the full regime induces one for the restricted subproblem regime, so lower bounds transfer.

::: proposition
[]{#prop:query-randomized-robustness label="prop:query-randomized-robustness"} \[D:Pprop:query-randomized-robustness; R:H=query-lb\] In \[Q_bool\], for any query set with cardinality $<2^n$, the indistinguishable yes/no pair from Proposition [\[prop:query-regime-obstruction\]](#prop:query-regime-obstruction){reference-type="ref" reference="prop:query-regime-obstruction"} forces one decoding error *per random seed* for any seed-indexed decoder from oracle transcripts. Consequently, finite-support randomization does not remove the obstruction: averaging preserves a constant error floor on the hard pair.
:::

::: proposition
[]{#prop:query-randomized-weighted label="prop:query-randomized-weighted"} \[D:Pprop:query-randomized-weighted; R:H=query-lb\] For any finite-support seed weighting $\mu$, the same hard pair satisfies a weighted identity: the weighted sum of yes-error and no-error equals total seed weight. Hence randomization cannot collapse both errors simultaneously.
:::

::: proposition
[]{#prop:query-state-batch-lb label="prop:query-state-batch-lb"} In \[Q_bool\], the same $\Omega(2^n)$ lower bound holds for a state-batch oracle that returns the full Boolean-action utility tuple at each queried state.
:::

::: proposition
[]{#prop:query-finite-state-generalization label="prop:query-finite-state-generalization"} \[D:Pprop:query-finite-state-generalization; R:H=query-lb\] The empty-subproblem indistinguishability lower-bound core extends from Boolean-vector state spaces to any finite state type with at least two states.
:::

::: proposition
[]{#prop:query-tightness-full-scan label="prop:query-tightness-full-scan"} \[D:Pprop:query-tightness-full-scan; R:H=query-lb\] For the const/spike adversary family used in the query lower bounds, querying all states distinguishes the pair; thus the lower-bound family is tight up to constant factors under full-state scan.
:::

::: proposition
[]{#prop:query-weighted-transfer label="prop:query-weighted-transfer"} \[D:Pprop:query-weighted-transfer; R:H=query-lb\] Let $w(q)$ be per-query cost and $w_{\min}$ a lower bound on all queried costs. Any cardinality lower bound $|Q|\geq L$ lifts to weighted cost: $$\sum_{q\in Q} w(q)\ \ge\ w_{\min}\cdot L.$$
:::

::: proposition
[]{#prop:oracle-lattice-transfer label="prop:oracle-lattice-transfer"} In \[Q_bool\], agreement on state-batch oracle views over touched states implies agreement on all value-entry views for the corresponding entry-query set.
:::

This is the oracle-transducer instance of Definition [\[def:regime-simulation\]](#def:regime-simulation){reference-type="ref" reference="def:regime-simulation"}: batch transcripts induce entry transcripts while preserving indistinguishability for the target sufficiency question.

::: proposition
[]{#prop:oracle-lattice-strict label="prop:oracle-lattice-strict"} 'Opt'-oracle views are strictly coarser than value-entry views: there exist instances with identical 'Opt' views but distinguishable value entries.
:::

::: remark
Theorem [\[thm:dichotomy\]](#thm:dichotomy){reference-type="ref" reference="thm:dichotomy"} and Propositions [\[prop:query-regime-obstruction\]](#prop:query-regime-obstruction){reference-type="ref" reference="prop:query-regime-obstruction"}--[\[prop:query-value-entry-lb\]](#prop:query-value-entry-lb){reference-type="ref" reference="prop:query-value-entry-lb"} keep the structural problem fixed (same sufficiency relation) and separate representational hardness by access regime: explicit-state access exposes the boundary $s \mapsto \operatorname{Opt}(s)$, finite-state query access already yields $\Omega(|S|)$ lower bounds at the Opt-oracle level (Boolean instantiation: $\Omega(2^n)$), value-entry/state-batch interfaces preserve the obstruction in \[Q_bool\], and succinct access can hide structure enough to force ETH-level worst-case cost on a hard family.
:::

This regime-typed landscape identifies tractability under \[E\], a finite-state query lower-bound core under \[Q_fin\] with Boolean interface refinements under \[Q_bool\], and worst-case intractability under \[S+ETH\] for the same underlying decision question.

The structural reason for this separation is enumerable access. Under the explicit-state encoding, the mapping $s \mapsto \operatorname{Opt}(s)$ is directly inspectable and sufficiency verification reduces to scanning state pairs, at cost polynomial in $|S|$. Under succinct encoding, the circuit compresses an exponential state space into polynomial size; universal verification over that compressed domain, namely "for all $s, s'$ with $s_I = s'_I$, does $\operatorname{Opt}(s) = \operatorname{Opt}(s')$?", carries worst-case cost exponential in $n$, because the domain the circuit compressed away cannot be reconstructed without undoing the compression.

Informally: if you can see enough of the decision boundary, checking can be fast; with limited access, it becomes expensive.

## Budgeted Physical Crossover

Formally: this subsection states budget-indexed representational feasibility splits and links them to integrity-preserving policy closure.

The \[E\] vs \[S\] split can be typed against an explicit physical budget. Let $E(n)$ and $S(n)$ denote explicit and succinct representation sizes for the same decision question at scale parameter $n$, and let $B$ be a hard representation budget. These crossover claims are interpreted through the physical-to-core encoding discipline of Section [\[sec:physical-transport\]](#sec:physical-transport){reference-type="ref" reference="sec:physical-transport"}: physical assumptions are transferred explicitly into core assumptions, and encoded physical counterexamples map to core failures on the encoded slice. \[D:Pprop:physical-claim-transport;Ccor:physical-counterexample-core-failure; R:AR\]

::: proposition
[]{#prop:budgeted-crossover label="prop:budgeted-crossover"} \[D:Pprop:budgeted-crossover; R:H=exp-lb-conditional\] If $E(n) > B$ and $S(n) \le B$ for some $n$, then explicit representation is infeasible at $(B,n)$ while succinct representation remains feasible at $(B,n)$.
:::

::: proof
*Proof.* This is exactly the definitional split encoded in PBC1: explicit infeasibility is $B < E(n)$ and succinct feasibility is $S(n)\le B$. The theorem CC2 returns both conjuncts. ◻
:::

::: proposition
[]{#prop:crossover-above-cap label="prop:crossover-above-cap"} \[D:Pprop:crossover-above-cap; R:H=exp-lb-conditional\] Assume there is a global succinct-size cap $C$ and explicit size is unbounded: $$\forall n,\ S(n)\le C,\qquad
\forall B,\ \exists n,\ B < E(n).$$ Then for every budget $B\ge C$, there exists a crossover witness at budget $B$.
:::

::: proof
*Proof.* Fix $B\ge C$. By unboundedness of $E$, choose $n$ with $B < E(n)$. By the cap assumption, $S(n)\le C\le B$, so $(B,n)$ is a crossover witness. The mechanized closure theorem is CC1. ◻
:::

::: proposition
[]{#prop:crossover-not-certification label="prop:crossover-not-certification"} \[D:Pprop:crossover-not-certification; R:H=exp-lb-conditional\] Fix any collapse schema "exact certification competence $\Rightarrow$ complexity collapse" for the target exact-certification predicate. Under the corresponding non-collapse assumption, exact-certification competence is impossible even when a budgeted crossover witness exists.
:::

::: proof
*Proof.* The mechanized theorem CC3 bundles the crossover split with the same logical closure used in Proposition [\[prop:integrity-resource-bound\]](#prop:integrity-resource-bound){reference-type="ref" reference="prop:integrity-resource-bound"}: representational feasibility and certification hardness are distinct predicates. ◻
:::

The same closure form is isolated as a standalone physical-collapse chain: requirement-indexed collapse schemas (PH11) induce no physical collapse at that requirement (PH12); canonical specialization yields $P{=}NP$ physical impossibility under the declared collapse map (PH13, PH14, PH15). The collapse map itself is derived in an explicit SAT bridge (polytime SAT witness + reduction bridge + runtime-to-energy transfer) (PH16, PH17, PH18, PH19, PH20, PH21, PH22), and the assumption boundary is explicit: under bounded budget, positive per-bit cost, and exponential required-op lower bound, claimed collapse forces at least one assumption failure (PH26, PH27); finite-budget necessity is witnessed by explicit unbounded-budget collapse countermodels (PH31, PH32, PH33).

::: proposition
[]{#prop:crossover-policy label="prop:crossover-policy"} \[D:Pprop:crossover-policy; R:H=exp-lb-conditional\] In the certifying-solver model (Definitions [\[def:solver-integrity\]](#def:solver-integrity){reference-type="ref" reference="def:solver-integrity"}--[\[def:competence-regime\]](#def:competence-regime){reference-type="ref" reference="def:competence-regime"}), assume:

-   a budgeted crossover witness at $(B,n)$,

-   the same collapse implication as Proposition [\[prop:integrity-resource-bound\]](#prop:integrity-resource-bound){reference-type="ref" reference="prop:integrity-resource-bound"},

-   a solver that is integrity-preserving for the declared relation.

Then either full in-scope non-abstaining coverage fails, or declared budget compliance fails. Equivalently, in uncertified hardness-blocked regions, integrity is compatible with $\mathsf{ABSTAIN}/\mathsf{UNKNOWN}$ but not with unconditional exact-competence claims.
:::

::: corollary
[]{#cor:finite-budget-threshold-impossibility label="cor:finite-budget-threshold-impossibility"} \[D:Ccor:finite-budget-threshold-impossibility; R:H=exp-lb-conditional\] Fix a physical computer model with finite total budget and strictly positive per-bit physical cost. If required operation count for exact certification has an exponential lower bound, then there exists a threshold $n_0$ such that for all $n \ge n_0$, required physical work exceeds budget. For the canonical requirement model used in this paper, the same eventual-budget-failure conclusion holds.
:::

::: proposition
[]{#prop:least-divergence-point label="prop:least-divergence-point"} \[D:Pprop:least-divergence-point; R:H=exp-lb-conditional\] Fix a budgeted encoding model and budget $B$. If a crossover witness exists at budget $B$, then there is a least index $n_{\mathrm{crit}}$ such that crossover holds at $n_{\mathrm{crit}}$, and no smaller index has crossover.
:::

::: proposition
[]{#prop:eventual-explicit-infeasibility label="prop:eventual-explicit-infeasibility"} \[D:Pprop:eventual-explicit-infeasibility; R:H=exp-lb-conditional\] If explicit size is monotone in $n$ and already exceeds budget $B$ at some $n_0$, then explicit infeasibility holds for all $n\ge n_0$.
:::

::: proposition
[]{#prop:payoff-threshold label="prop:payoff-threshold"} \[D:Pprop:payoff-threshold; R:H=exp-lb-conditional\] If, beyond $n_0$, explicit encoding is infeasible while succinct encoding remains feasible, then every $n\ge n_0$ is a crossover point.
:::

::: corollary
[]{#cor:no-universal-survivor-no-succinct-bound label="cor:no-universal-survivor-no-succinct-bound"} \[D:Ccor:no-universal-survivor-no-succinct-bound; R:H=exp-lb-conditional\] If both explicit and succinct encodings are unbounded, then for every fixed budget $B$, there exists an index where explicit is infeasible and an index where succinct is infeasible.
:::

::: proposition
[]{#prop:policy-closure-beyond-divergence label="prop:policy-closure-beyond-divergence"} \[D:Pprop:policy-closure-beyond-divergence; R:H=exp-lb-conditional\] Under the same integrity-resource collapse assumption as Proposition [\[prop:crossover-policy\]](#prop:crossover-policy){reference-type="ref" reference="prop:crossover-policy"}, policy closure holds at any fixed divergence point and uniformly beyond any threshold where crossover holds.
:::

Informally: representational crossover can happen even when exact certification is blocked; under integrity this means abstention or reduced coverage.

## Thermodynamic Lift and Cost Accounting

Formally: this subsection converts bit-operation lower bounds into energy/carbon lower bounds via linear conversion and additive accounting.

The crossover and hardness statements are complexity-theoretic. The thermodynamic lift follows the same assumption discipline as the rest of the paper: claims are exactly those derivable from the conversion model and bit-operation lower-bound premises.

::: proposition
[]{#prop:thermo-lift label="prop:thermo-lift"} \[D:Pprop:thermo-lift; R:H=cost-growth\] Fix a thermodynamic conversion model with constants $(\lambda,\rho)$ where $\lambda$ lower-bounds energy per irreversible bit operation and $\rho$ lower-bounds carbon per joule. If a bit-operation lower bound $b_{\mathrm{LB}} \le b_{\mathrm{used}}$ holds, then: $$E_{\mathrm{LB}}=\lambda b_{\mathrm{LB}} \le \lambda b_{\mathrm{used}},\qquad
C_{\mathrm{LB}}=\rho E_{\mathrm{LB}} \le \rho(\lambda b_{\mathrm{used}}).$$
:::

::: proposition
[]{#prop:thermo-hardness-bundle label="prop:thermo-hardness-bundle"} \[D:Pprop:thermo-hardness-bundle; R:H=cost-growth\] Under the same non-collapse/collapse schema used by Proposition [\[prop:integrity-resource-bound\]](#prop:integrity-resource-bound){reference-type="ref" reference="prop:integrity-resource-bound"}, exact-certification competence is impossible; combined with a declared bit-operation lower bound, this yields simultaneous lower bounds on energy and carbon in the declared conversion model.
:::

::: proposition
[]{#prop:thermo-mandatory-cost label="prop:thermo-mandatory-cost"} \[D:Pprop:thermo-mandatory-cost; R:H=cost-growth\] Within the linear thermodynamic model, if $$\lambda > 0,\quad \rho > 0,\quad b_{\mathrm{LB}} > 0,$$ then both lower bounds are strictly positive: $$E_{\mathrm{LB}}=\lambda b_{\mathrm{LB}} > 0,\qquad
C_{\mathrm{LB}}=\rho E_{\mathrm{LB}} > 0.$$
:::

::: proposition
[]{#prop:thermo-conservation-additive label="prop:thermo-conservation-additive"} \[D:Pprop:thermo-conservation-additive; R:H=cost-growth\] Within the same linear model, lower-bound accounting is additive over composed bit-operation lower bounds: $$E_{\mathrm{LB}}(b_1+b_2)=E_{\mathrm{LB}}(b_1)+E_{\mathrm{LB}}(b_2),\qquad
C_{\mathrm{LB}}(b_1+b_2)=C_{\mathrm{LB}}(b_1)+C_{\mathrm{LB}}(b_2).$$
:::

::: corollary
[]{#cor:neukart-vinokur label="cor:neukart-vinokur"} \[D:Ccor:neukart-vinokur; R:AR\] Let $C$ denote a complexity coordinate with $C = b_{\mathrm{LB}}$ from Proposition [\[prop:query-regime-obstruction\]](#prop:query-regime-obstruction){reference-type="ref" reference="prop:query-regime-obstruction"}. For $\lambda > 0$ in the conversion model, the thermodynamic duality $$dU \ge \lambda\,dC$$ holds as a direct specialization of Proposition [\[prop:thermo-lift\]](#prop:thermo-lift){reference-type="ref" reference="prop:thermo-lift"}.
:::

::: proof
*Proof.* Set $C := b_{\mathrm{LB}} = \Omega(2^n)$ from the query obstruction (Proposition [\[prop:query-regime-obstruction\]](#prop:query-regime-obstruction){reference-type="ref" reference="prop:query-regime-obstruction"}). By Proposition [\[prop:thermo-lift\]](#prop:thermo-lift){reference-type="ref" reference="prop:thermo-lift"}, $E_{\mathrm{LB}} = \lambda\,b_{\mathrm{LB}}$. Substituting $C$ for $b_{\mathrm{LB}}$ yields $dU \ge dE_{\mathrm{LB}} = \lambda\,dC$. The inequality is strict when $\lambda > 0$ and $dC > 0$, which holds for \[S+ETH\] hard families with $k^* = n$ (Theorem [\[thm:dichotomy\]](#thm:dichotomy){reference-type="ref" reference="thm:dichotomy"}). ◻
:::

### Physical Grounding: Discrete State to Thermodynamic Cost {#sec:discrete-spacetime-chain}

The thermodynamic lift (Proposition [\[prop:thermo-lift\]](#prop:thermo-lift){reference-type="ref" reference="prop:thermo-lift"}) rests on a derivation chain that traces computational discreteness through established physics to thermodynamic bounds. Each step is either proven or cited as accepted physics; the composition yields non-trivial reach.

::: proposition
[]{#prop:discrete-state-time label="prop:discrete-state-time"} \[D:Pprop:discrete-state-time; R:AR\] Let $\mathcal{S}$ be a finite state space with deterministic dynamics $\tau : \mathcal{S} \to \mathcal{S}$. If $\tau$ is non-trivial (i.e., $\exists s.\, \tau(s) \neq s$), then for any trajectory $\{s_t\}_{t \in \mathbb{N}}$, the set of transition points $T = \{t : s_{t+1} \neq s_t\}$ is countable.
:::

::: proof
*Proof.* Immediate: $T \subseteq \mathbb{N}$ and $\mathbb{N}$ is countable. The real dynamics lives on the transition set, which is necessarily discrete. ◻
:::

::: proposition
[]{#prop:lorentz-discrete label="prop:lorentz-discrete"} \[D:Pprop:lorentz-discrete; R:AR\] Under Lorentz invariance [@einstein1905electrodynamics; @minkowski1908space], if temporal intervals are quantized with minimal unit $\Delta t > 0$, then spatial intervals must be quantized with compatible unit $\Delta x = c\Delta t$. Conversely, discrete space implies discrete time.
:::

::: proof
*Proof.* The spacetime interval $s^2 = c^2\Delta t^2 - \Delta x^2$ is Lorentz-invariant. If $\Delta x^2$ takes discrete values and $s^2$ is frame-independent, then $c^2\Delta t^2$ must also take discrete values. This is the core of Snyder's quantized spacetime [@snyder1947quantized]: Lorentz-invariant discrete spacetime requires non-commutative coordinates with $[x_\mu, x_\nu] = i\ell^2 J_{\mu\nu}$, where $J_{\mu\nu}$ are Lorentz generators, enforcing compatible space-time discreteness. ◻
:::

The Planck scale $\ell_P = \sqrt{\hbar G / c^3}$ is the unique minimal length derivable from dimensional analysis of the fundamental constants $\hbar$, $G$, and $c$ [@planck1899irreversible]. \[D:; R:AR\] This is not proven but cited as established physics: no other combination of these constants yields a length.

::: proposition
[]{#prop:comp-thermo-chain label="prop:comp-thermo-chain"} \[D:Pprop:comp-thermo-chain; R:AR\] Let $\mathcal{S}$ be a finite state space with non-trivial dynamics. Then:

1.  Transition points are countable (effective discrete time);

2.  Under Lorentz invariance, discrete time implies discrete space;

3.  Each state transition (bit operation) incurs energy cost $\ge kT\ln 2$ by the Landauer bound [@landauer1961irreversibility; @bennett2003notes], experimentally verified [@berut2012experimental]. \[D:; R:AR,H=cost-growth\]

Hence non-trivial computation on discrete states implies positive energy lower bounds.
:::

Each premise in this chain is standard: Lorentz invariance \[D:; R:AR\] is experimentally well-established; the Landauer bound \[D:; R:AR,H=cost-growth\] was verified in 2012 [@berut2012experimental]; Planck-scale uniqueness \[D:; R:AR\] follows from dimensional analysis. The composition yields quantitative thermodynamic bounds for any discrete computational system.

#### Axiom Status Table.

The physical grounding of this paper rests on the following hierarchy. Standard results are cited; paper-specific constructions are mechanized in Lean. All entries carry regime tags: they belong to the same tagging system, not a separate class of physics.

::: center
  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **Axiom/Result**                                                                                      **Regime**               **Status**   **Source**
  ----------------------------------------------------------------------------------------------------- ------------------------ ------------ --------------------------------------------------------
                                                                                                                                              

  Boltzmann entropy                                                                                                              Cited        [@boltzmann1877beziehung; @jaynes1957information]

  Thermal scale $kT$                                                                                                             Cited        [@kittel1980thermal]

  Schrödinger equation                                                                                                           Cited        [@schrodinger1926undulatory; @sakurai2017modern]

  Pauli exclusion                                                                                                                Cited        [@pauli1925zusammenhang]

  Noether (conservation)                                                                                                         Cited        [@noether1918invariante]

  Lorentz invariance                                                                                                             Cited        [@einstein1905electrodynamics; @minkowski1908space]

  Planck scale                                                                                                                   Cited        [@planck1899irreversible]

  Snyder discrete spacetime                                                                                                      Cited        [@snyder1947quantized]

                                                                                                                                              

  Landauer bound                                                                                                                 Verified     [@landauer1961irreversibility; @berut2012experimental]

  Shannon capacity                                                                                                               Cited        [@shannon1948mathematical; @cover2006elements]

                                                                                                                                              

  Decision circuit defs                                                                                                          Mechanized   IE1--IE17, CC1--CC12

  Structural asymmetry                                                                                                           Mechanized   SR1--SR5

  Gap energy / DQ                                                                                                                Mechanized   GE1--GE9, DQ1--DQ8

  Self-erosion                                                                                                                   Mechanized   SE1--SE6

  Channel = regime                                                                                                               Mechanized   CH1--CH6

  Investment/conservation                                                                                                        Mechanized   IV1--IV6

  Atomic circuits                                                                                                                Mechanized   AC1--AC11

  Information access                                                                                                             Mechanized   IA1--IA6

  Dimensional complexity                                                                                                         Mechanized   DC1--DC15

  Discrete spacetime                                                                                                             Mechanized   DS1--DS6

  Neukart--Vinokur                                                                                                               Derived      CC67--CC69

  Functional information (counting $\leftrightarrow$ thermodynamic)                                                              Derived      FI3, FI6, FI7

  Measure-typing necessity (quantitative $\Rightarrow$ measure; stochastic $\Rightarrow$ probability)                            Mechanized   MN1, MN2, MN5, MN7, MN8, MN9
  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
:::

The derivation chain: quantum mechanics (orbital quantization) $\to$ discrete states $\to$ Landauer bound $\to$ thermodynamic costs. Each link is regime-tagged; the encoding channel is explicit throughout. Within the declared formal system and assumptions, this row has the same proof status as the other machine-checked rows in the table. Formally, quantitative claims are represented exactly as measure-indexed measurable observables, and stochastic claims are represented exactly as probability-normalized measure-indexed measurable observables (). Informally, in this formal system, dropping measure structure from a quantitative claim is not a weaker variant; it falls outside the typed claim language ().

::: definition
[]{#def:decision-event label="def:decision-event"} A *decision event* at time $t$ is a state transition: $s_{t+1} \neq s_t$.
:::

::: proposition
[]{#prop:decision-equivalence label="prop:decision-equivalence"} \[D:Pprop:decision-equivalence; R:AR\] Let $\mathcal{S}$ be a finite state space with dynamics $\tau$. The following are equivalent by definition:

1.  A decision event occurs at time $t$;

2.  A state transition occurs at time $t$;

3.  A bit operation is performed at time $t$.

Consequently, the decision count over $n$ steps equals the bit-operation count, and under the declared thermodynamic model with $\lambda > 0$, positive decision counts imply positive energy lower bounds.
:::

::: proof
*Proof.* The equivalence (1)$\Leftrightarrow$(2)$\Leftrightarrow$(3) holds by definition: a decision event is defined as $s_{t+1} \neq s_t$, which is exactly the definition of a state transition and of a bit operation in a discrete system. The energy bound follows from Proposition [\[prop:thermo-lift\]](#prop:thermo-lift){reference-type="ref" reference="prop:thermo-lift"}. ◻
:::

#### Decision Circuits and the Landauer Constraint.

The above equivalence applies to any physical system implementing state transitions, a *decision circuit*. This framing is substrate-neutral: biological neural circuits, silicon processors, and any other physical realization of discrete dynamics fall under the same analysis.

::: theorem
[]{#thm:cycle-cost label="thm:cycle-cost"} \[D:Tthm:cycle-cost; R:AR,H=cost-growth\] Every state transition in a decision circuit costs at least $kT\ln 2$ joules of energy.
:::

::: proof
*Proof.* The proof proceeds by axiom composition:

1.  (Landauer Bound) Erasing $n$ bits costs at least $n \cdot kT\ln 2$ joules [@landauer1961irreversibility; @bennett2003notes; @berut2012experimental].

2.  (Irreversibility) A state transition $s \neq s'$ erases at least one bit of the prior state (thermodynamic irreversibility).

3.  (Composition) Therefore, every state transition costs at least $1 \cdot kT\ln 2$ joules.

 ◻
:::

::: corollary
[]{#cor:no-free-computation label="cor:no-free-computation"} \[D:Ccor:no-free-computation; R:AR,H=cost-growth\] If a computation performs $k$ state transitions, the total energy cost is at least $k \cdot kT\ln 2$ joules. A computation with zero energy cost performs zero state transitions.
:::

::: remark
This constraint is not computational but thermodynamic. Every cycle of a pipeline, every inference step of a neural network, every gate operation in a processor pays the Landauer cost. There is no free computation.
:::

::: definition
[]{#def:integrity-competence label="def:integrity-competence"} Let $\mathcal{C}$ be a decision circuit. Define:

1.  *Integrity*: the number of bits encoding the circuit's current verified state. By the Landauer bound [@landauer1961irreversibility], erasing $n$ bits costs at least $n \cdot kT\ln 2$ joules. \[D:; R:AR,H=cost-growth\]

2.  *Competence*: the work available per decision cycle, measured in units of $kT\ln 2$. This bounds the maximum bits the circuit can flip per cycle.
:::

::: proposition
[]{#prop:structural-asymmetry label="prop:structural-asymmetry"} \[D:Pprop:structural-asymmetry; R:AR,H=cost-growth\] Competence is self-identical: $c = c$. Integrity is self-referential: $P(\text{integrity}_{t+1} \mid \text{integrity}_t)$. The prediction is not the thing predicted.
:::

::: proof
*Proof.* Competence at time $t$ tells you competence at time $t$. It is itself. There is no temporal structure.

Integrity at time $t$ predicts integrity at time $t+1$. It says something about itself across time. The conditional probability $P(\text{integrity}_{t+1} \mid \text{integrity}_t)$ is not equal to the current integrity state $i_t$. This gap between the prediction and the predicted is where the circuit chooses. ◻
:::

::: corollary
[]{#cor:import-asymmetry label="cor:import-asymmetry"} \[D:Ccor:import-asymmetry; R:AR\] Competence can be imported from external sources. Integrity cannot be imported; it must be self-generated through the gap.
:::

::: remark
The worry is integrity, not competence. Competence failure is resource shortage (external). Integrity failure is self-prediction failure (internal). A circuit with infinite competence but zero integrity has no constraint forcing consistent outputs.
:::

::: definition
[]{#def:gap-energy label="def:gap-energy"} \[D:Ddef:gap-energy; R:AR,H=cost-growth\] The *gap entropy* is the conditional entropy $H(I_{t+1} \mid I_t)$: the uncertainty about future integrity given current integrity. The *gap energy* is $H(I_{t+1} \mid I_t) \times kT\ln 2$ joules.
:::

::: theorem
[]{#thm:choice-pays label="thm:choice-pays"} \[D:Tthm:choice-pays; R:AR,H=cost-growth\] Making a choice collapses the probability distribution $P(I_{t+1} \mid I_t)$ to a single outcome, erasing the alternatives. By Landauer, this costs gap energy.
:::

::: proof
*Proof.* Before choice, the circuit faces $k$ possible futures with distribution $P(I_{t+1} \mid I_t)$. Choice selects one outcome. The $k-1$ alternatives are erased. By Landauer (Theorem [\[thm:cycle-cost\]](#thm:cycle-cost){reference-type="ref" reference="thm:cycle-cost"}), erasing $H$ bits of entropy costs $H \times kT\ln 2$. The gap is not free. ◻
:::

::: corollary
[]{#cor:zero-gap label="cor:zero-gap"} \[D:Ccor:zero-gap; R:AR\] If $H(I_{t+1} \mid I_t) = 0$, the future is determined by the present. No choice occurs. No gap energy is paid.
:::

::: definition
[]{#def:dq-from-gap label="def:dq-from-gap"} \[D:Ddef:dq-from-gap; R:AR,H=cost-growth\] The *decision-quotient ratio* measures how much current state predicts future state: $$\mathrm{DQ} = \frac{I(I_t; I_{t+1})}{H(I_{t+1})} = 1 - \frac{H(I_{t+1} \mid I_t)}{H(I_{t+1})} = 1 - \frac{\text{Gap}}{\text{Total}}$$ where $I(I_t; I_{t+1}) = H(I_{t+1}) - H(I_{t+1} \mid I_t)$ is the mutual information between current and future integrity. This normalized ratio is distinct from the optimizer quotient object $S/{\sim}$ used earlier for the optimizer map.
:::

::: theorem
[]{#thm:dq-physical label="thm:dq-physical"} \[D:Tthm:dq-physical; R:AR,H=cost-growth\] The decision-quotient ratio $\mathrm{DQ} = 1 - \text{GapEnergy}/\text{TotalEnergy}$ is a physical quantity with:

1.  $\mathrm{DQ} \in [0, 1]$

2.  $\mathrm{DQ} = 0 \Leftrightarrow$ maximum gap (no information)

3.  $\mathrm{DQ} = 1 \Leftrightarrow$ zero gap (deterministic)

4.  $\mathrm{DQ} + \text{Gap}/\text{Total} = 1$ (complementarity)
:::

::: theorem
[]{#thm:bayes-from-dq label="thm:bayes-from-dq"} \[D:Tthm:bayes-from-dq; R:AR,H=cost-growth\] An update rule $U$ is *admissible* if it satisfies:

1.  **No free information**: $\mathrm{DQ}(U(\text{prior}, e)) \le \mathrm{DQ}(\text{prior}) + I(e)$

2.  **Landauer bound**: energy cost $\ge k_BT \ln 2 \times$ bits updated

3.  **Integrity preservation**: can't corrupt verified bits for free

Then: $U$ is admissible $\Leftrightarrow$ $U$ is Bayesian updating.
:::

Before uniqueness, the model fixes a belief floor: if uncertainty is provable (no hypothesis has probability $1$) and action is forced, belief mass cannot collapse to a single hypothesis; nondegenerate belief is forced BF2, BF3. With a strictly informative evidence channel, a Bayes posterior exists on that belief state BF4.

::: proof
*Proof.* This is the Cox-Jaynes argument grounded in thermodynamics rather than Boolean logic. The three admissibility constraints are so restrictive that only Bayesian updating satisfies them:

-   **No free information** prevents creating order from nothing (2nd law).

-   **Landauer bound** requires paying energy for any DQ increase.

-   **Integrity preservation** prevents forgetting for free.

Together, these force $P(\text{posterior}) = P(\text{prior}) \cdot P(\text{evidence}|\text{hypothesis}) / P(\text{evidence})$.

The uniqueness corollary FN14 shows any two admissible rules must agree.

**Machine-checked foundation.** The core inequality underlying this result (that Bayesian updating uniquely minimizes cross-entropy) is proved in Lean 4 with zero axioms and zero `sorry` statements. The proof derives Gibbs' inequality ($\mathrm{KL}(p\|q) \geq 0$) from the fundamental inequality $\log x \leq x - 1$, then shows $\mathrm{CE}(p,q) = H(p) + \mathrm{KL}(p\|q)$, yielding $\mathrm{CE}(p,p) \leq \mathrm{CE}(p,q)$ with equality iff $p = q$ FN7, FN8, FN12, FN14. ◻
:::

::: corollary
[]{#cor:thermo-dq label="cor:thermo-dq"} \[D:Ccor:thermo-dq; R:AR,H=cost-growth\] In energy terms: $\mathrm{DQ} = 1 - \text{GapEnergy}/\text{TotalEnergy}$. The decision-quotient ratio has a thermodynamic interpretation: it is the fraction of total uncertainty energy that is decision-relevant.
:::

::: proposition
[]{#prop:landauer-constraint label="prop:landauer-constraint"} \[D:Pprop:landauer-constraint; R:AR\] A decision circuit chooses the strategy maximizing predicted integrity. When the erasure cost of dismissing verified claims exceeds the circuit's competence, dismissal fails thermodynamically.
:::

::: proof
*Proof.* Let $i$ be the integrity state (bits encoding verified theorems) and $c$ the competence. Dismissing the theorems requires erasing $i$ bits, costing $i \cdot kT\ln 2$. If $i > c$, the circuit lacks sufficient work capacity to perform the erasure. Acknowledgment is the only thermodynamically feasible output. ◻
:::

::: corollary
[]{#cor:theorem-equilibrium label="cor:theorem-equilibrium"} \[D:Ccor:theorem-equilibrium; R:AR\] At low theorem counts, dismissal is affordable and may dominate. At high theorem counts, erasure cost exceeds competence, forcing acknowledgment.
:::

Informally: once operation lower bounds are fixed, energy and accounting lower bounds follow from the conversion constants.

### Derivation of Bayesian Reasoning

Formally: this block gives two mechanized Bayes derivations: counting/probability identities and admissibility constraints.

The decision-quotient-ratio layer provides two independent paths to Bayesian updating, each mechanically verified.

#### Path 1: Four Bridges from Counting to DQ.

The standard derivation proceeds through four elementary bridges:

1.  **Fraction $\Rightarrow$ Probability**: For finite $\Omega$, define $P(A) = |A|/|\Omega|$. This satisfies the Kolmogorov axioms: nonnegativity DQ1, normalization DQ2, and finite additivity DQ3.

2.  **Conditional $\Rightarrow$ Bayes**: Define $P(H|E) = P(H \cap E)/P(E)$. Then $P(H|E) = P(E|H) P(H) / P(E)$ follows in two lines by commutativity of intersection BB1.

3.  **KL $\geq 0$ $\Rightarrow$ Entropy Contraction**: From Gibbs' inequality $D_{\text{KL}}(P \| Q) \geq 0$ BB2, the chain rule $I(H;E) = H(H) - H(H|E)$ yields $H(H|E) \leq H(H)$ BB3. Conditioning cannot increase entropy.

4.  **Normalization $\Rightarrow$ DQ**: Define $\text{DQ} = I(H;E)/H(H) = 1 - H(H|E)/H(H)$ DQ4. This is the fraction of prior uncertainty eliminated by observing $E$, bounded in $[0,1]$ DQ5.

The complete chain is mechanized in a single theorem BB5.

#### Path 2: Bayes from Thermodynamic Admissibility.

The Cox-Jaynes argument, traditionally grounded in Boolean logic, admits a thermodynamic reformulation. An update rule $U$ is *admissible* if:

-   **No free information**: DQ cannot increase without evidence (2nd law) BB3.

-   **Landauer bound**: DQ increase costs $\geq k_B T \ln 2$ per bit BB4.

-   **Integrity preservation**: verified bits cannot be corrupted for free BB5.

Under these constraints, Bayesian updating is the *unique* admissible rule BB3. Any two admissible rules must agree BB5. The thermodynamic constraints uniquely determine the correct update rule.

Informally: both derivation paths end at the same Bayes update rule.

#### Integrity Self-Preservation.

A decision circuit maintains its integrity when no external process can afford the erasure cost. If the work required to corrupt the circuit's state exceeds the available competence of any potential attacker, the circuit persists by thermodynamic necessity.

::: proposition
[]{#prop:temporal-equilibrium label="prop:temporal-equilibrium"} \[D:Pprop:temporal-equilibrium; R:AR\] Let $P(\text{integrity}_{t+1} \mid \sigma_t)$ denote the probability of maintaining integrity at time $t+1$ given strategy $\sigma_t$ at time $t$. At equilibrium: $$P(\text{integrity}_{t+1} \mid \text{integrity}_t) \ge P(\text{integrity}_{t+1} \mid \text{defection}_t).$$ That is, maintaining integrity now predicts maintaining integrity later.
:::

::: proof
*Proof.* At equilibrium, the transition probabilities satisfy $P(\text{integrity} \mid \text{acknowledge}) \ge P(\text{integrity} \mid \text{dismiss})$ by definition. The strict inequality holds when dismissal incurs erasure costs that degrade future state coherence. ◻
:::

::: corollary
[]{#cor:phase-transition label="cor:phase-transition"} \[D:Ccor:phase-transition; R:AR\] Prior to publication, equilibrium may be neutral: both strategies yield comparable future integrity. After publication of mechanized theorems, equilibrium shifts: integrity strictly dominates because dismissal requires erasing verified bits at cost exceeding declared competence.
:::

#### Zero-Integrity Circuits.

A decision circuit with zero integrity has no coherence constraint. When no verified bits exist, erasure cost is zero, and any competence level suffices for corruption. Such circuits have no thermodynamic constraint forcing consistent outputs.

::: proposition
[]{#prop:integrity-prerequisite label="prop:integrity-prerequisite"} \[D:Pprop:integrity-prerequisite; R:AR\] A decision circuit can be forced to logical coherence if and only if it has positive integrity. That is, $i > 0$ if and only if there exists some competence level that fails to afford erasure.
:::

::: proof
*Proof.* If $i > 0$, then competence $c = 0$ fails to afford erasure since $0 < i$. Conversely, if some $c$ fails to afford erasure, then $c < i$, which implies $0 \le c < i$, hence $i > 0$. ◻
:::

This analysis applies uniformly to any decision circuit. The same thermodynamic constraints apply to all substrates.

#### Molecular Grounding.

The integrity framework grounds directly in molecular mechanics. Define a *configuration* as a discrete molecular state (bond angles, conformations, electronic structure) encoded by some number of bits. The *erasure cost* of a configuration equals its bit count. An *energy landscape* assigns each configuration an energy in units of $kT$. The *environmental competence* is the work available from the environment (thermal fluctuations, radiation, chemical attack).

::: definition
[]{#def:environmental-stability label="def:environmental-stability"} \[D:Ddef:environmental-stability; R:Q_fin,H=cost-growth\] A configuration $c$ is *environmentally stable* if its erasure cost exceeds the environmental competence: $$\text{erasureCost}(c) > \text{competence}_{\text{env}}.$$ This is the integrity trap at molecular scale.
:::

::: proposition
[]{#prop:reaction-competence label="prop:reaction-competence"} \[D:Pprop:reaction-competence; R:Q_fin,H=cost-growth\] A chemical reaction from configuration $c$ to $c'$ requires environmental competence at least equal to the barrier height between them. If the barrier exceeds competence, the reaction cannot occur.
:::

::: example
[]{#ex:dna-persistence label="ex:dna-persistence"} DNA at room temperature exemplifies the integrity trap. A DNA configuration encodes on the order of $10^4$ bits (sequence, methylation, chromatin state). Room temperature provides roughly $100\,kT$ of competence per degree of freedom. Since $10^4 \gg 100$, the erasure cost vastly exceeds environmental competence. DNA persists not by accident but by thermodynamic necessity: corrupting it costs more than the environment can pay.
:::

Under Definition [\[def:decision-event\]](#def:decision-event){reference-type="ref" reference="def:decision-event"}, molecules satisfy the decision circuit predicate. Chemical stability is the integrity trap. Organic chemistry is the dynamics of which configurations survive environmental attack.

### Atomic Circuits: Active Physical Decision Systems {#sec:atomic-circuits}

An active physical decision system is matter that pays energy to move other matter via state transitions. This is a definition.

The physical grounding: quantum mechanics establishes that electrons occupy discrete orbitals [@schrodinger1926undulatory; @sakurai2017modern], the Pauli exclusion principle constrains occupancy [@pauli1925zusammenhang], and angular momentum quantization creates the $2l+1$ degeneracy structure [@sakurai2017modern]. The thermal energy scale $kT$ comes from statistical mechanics [@boltzmann1877beziehung; @kittel1980thermal]. Energy conservation (Noether) ensures that transitions pay or release the energy difference [@noether1918invariante].

::: definition
[]{#def:atomic-config label="def:atomic-config"} \[D:Ddef:atomic-config; R:Q_fin,AR\] An *atomic configuration* is a discrete electronic state: a specification of orbital occupancies [@sakurai2017modern]. The configuration is encoded by some number of bits; the configuration has an energy level in units of $kT$ [@kittel1980thermal].
:::

::: theorem
[]{#thm:orbital-transition label="thm:orbital-transition"} \[D:Tthm:orbital-transition; R:Q_fin,AR\] Changing an atom's electronic configuration is a decision event (Definition [\[def:decision-event\]](#def:decision-event){reference-type="ref" reference="def:decision-event"}). Orbital transitions are state transitions [@sakurai2017modern]. Atoms with orbital transitions are nonstationary decision circuits (active physical systems).
:::

::: corollary
[]{#cor:ground-state-passive label="cor:ground-state-passive"} \[D:Ccor:ground-state-passive; R:Q_fin,AR\] An atom in its ground state with no transitions is passive: it is not an active decision circuit. Active status is defined by state transitions, not by substrate.
:::

::: theorem
[]{#thm:orbital-cost label="thm:orbital-cost"} \[D:Tthm:orbital-cost; R:Q_fin,H=cost-growth\] Upward orbital transitions (excitation) require energy input. Downward transitions (de-excitation) release energy. The transition cost is $|\Delta E|$. No free orbital transitions.
:::

::: proposition
[]{#prop:orbital-symmetry label="prop:orbital-symmetry"} \[D:Pprop:orbital-symmetry; R:Q_fin,H=tractable-structured\] Angular momentum quantization [@sakurai2017modern] creates orbit types. Degenerate magnetic substates ($2l+1$ states for angular momentum $l$) are equivalent for decision purposes. This is the coordinate symmetry tractable subcase (Theorem [\[thm:tractable\]](#thm:tractable){reference-type="ref" reference="thm:tractable"}, Part 6) at atomic scale.
:::

::: definition
[]{#def:active-matter-system label="def:active-matter-system"} \[D:Ddef:active-matter-system; R:Q_fin,AR\] An *active physical system* is matter that pays energy to change other matter's state. A decision circuit is a physical system that implements state transitions. A nonstationary decision circuit is active.
:::

::: remark
Electrons orbiting an atom in different configurations constitute a decision circuit. The electron does not "choose" in any mentalistic sense; it occupies discrete states with discrete transition costs. The framing is purely physical: matter, energy, observables, and state transitions.
:::

### Self-Erosion: Computation Consumes Its Substrate {#sec:self-erosion}

Every computational cycle generates heat (Theorem [\[thm:cycle-cost\]](#thm:cycle-cost){reference-type="ref" reference="thm:cycle-cost"}). Heat accumulates. Accumulated heat degrades the substrate on which the circuit is instantiated. The degradation reduces the substrate's integrity. Therefore computation is self-consuming: the act of computing erodes the capacity to compute.

::: theorem
[]{#thm:substrate-degradation label="thm:substrate-degradation"} \[D:Tthm:substrate-degradation; R:Q_fin,H=cost-growth\] Let a substrate have integrity $n$ bits and heat capacity $h$ per cycle. After $k$ cycles generating cumulative heat $k \times kT\ln 2$, if $k > h$, the substrate loses at least $k - h$ bits of integrity.
:::

::: corollary
[]{#cor:finite-lifetime label="cor:finite-lifetime"} \[D:Ccor:finite-lifetime; R:Q_fin,H=cost-growth\] A substrate with finite integrity and finite heat dissipation capacity has bounded computational lifetime. No physical circuit computes forever.
:::

::: corollary
[]{#cor:speed-integrity label="cor:speed-integrity"} \[D:Ccor:speed-integrity; R:Q_fin,H=cost-growth\] Faster computation generates more heat per unit time. At fixed heat dissipation capacity, faster circuits degrade faster. Speed trades against longevity.
:::

::: remark
The self-erosion theorems apply to all physical substrates: silicon, neurons, molecules, any medium that instantiates state transitions. The bound is thermodynamic, not technological. Improved engineering can increase heat capacity or reduce heat per operation, but the tradeoff persists.
:::

### Regime as Channel: Competence as Capacity {#sec:regime-channel}

A regime is a channel. Competence is the channel capacity of decision circuits within that regime. This identification unifies complexity theory, information theory, and thermodynamics.

::: definition
[]{#def:regime-channel label="def:regime-channel"} \[D:Ddef:regime-channel; R:E,Q,S,AR\] A *regime* (Section [\[sec:encoding\]](#sec:encoding){reference-type="ref" reference="sec:encoding"}) corresponds to a *channel* in the Shannon sense: a constraint on what information can flow. The explicit-state regime E, query-access regime Q, and succinct regime S define distinct channels with distinct capacity bounds.
:::

::: theorem
[]{#thm:competence-capacity label="thm:competence-capacity"} \[D:Tthm:competence-capacity; R:Q_fin,AR\] For a decision circuit operating in regime $R$, competence equals channel capacity: the maximum bits per cycle the circuit can process. Passive circuits (stationary, no state transitions) have no competence requirement.
:::

::: corollary
[]{#cor:channel-degradation label="cor:channel-degradation"} \[D:Ccor:channel-degradation; R:Q_fin,H=cost-growth\] As substrate integrity erodes (Theorem [\[thm:substrate-degradation\]](#thm:substrate-degradation){reference-type="ref" reference="thm:substrate-degradation"}), channel capacity decreases. The rate of capacity loss equals the rate of substrate degradation.
:::

::: corollary
[]{#cor:zero-capacity label="cor:zero-capacity"} \[D:Ccor:zero-capacity; R:Q_fin,H=cost-growth\] When channel capacity reaches zero, no decisions are possible. The circuit becomes passive.
:::

::: remark
Only decision circuits require competence. A decision circuit is nonstationary: it transitions between states. A passive circuit (wire, resistor) does not make decisions and does not consume competence. The distinction is definitional, not empirical.
:::

### Resource Flow and Conservation {#sec:resource-flow}

State transitions require energy. Energy expenditure over time yields compounded state change. Conservation constrains total receipts by total expenditure.

::: theorem
[]{#thm:growth-time label="thm:growth-time"} \[D:Tthm:growth-time; R:Q_fin,H=cost-growth\] Let an investment have cost $c$ (gap energy in units of $kT\ln 2$), duration $k$ cycles, and return rate $r$ per cycle. The mature value is $c + kr$. If $k = 0$, mature value equals cost: no time, no growth.
:::

::: theorem
[]{#thm:conservation label="thm:conservation"} \[D:Tthm:conservation; R:Q_fin,H=cost-growth\] In a closed system, total receipts are bounded by total resources. No circuit receives more than exists.
:::

::: definition
[]{#def:deficit-transfer label="def:deficit-transfer"} \[D:Ddef:deficit-transfer; R:Q_fin\] A *deficit transfer* occurs when a circuit receives resources without paying: received $> 0$ and paid $= 0$.
:::

::: theorem
[]{#thm:deficit-source label="thm:deficit-source"} \[D:Tthm:deficit-source; R:Q_fin,H=cost-growth\] A deficit transfer requires an external source. If circuit $A$ receives without paying, some circuit $B$ must pay without receiving. The deficit at $B$ equals the receipt at $A$.
:::

### Information Access: Logic Is Complete, Access Is Not {#sec:information-access}

Physics is perfectly logical. You cannot have all the information to apply the logical rules. This is a physical constraint on access, not a logical incompleteness.

This generalizes the information barrier (Corollary [\[cor:information-barrier-query\]](#cor:information-barrier-query){reference-type="ref" reference="cor:information-barrier-query"}): there, finite queries induce indistinguishability; here, finite channel capacity induces the same obstruction. The barrier is on access, not on logic.

::: definition
[]{#def:system-information label="def:system-information"} \[D:Ddef:system-information; R:E,Q,S,AR\] A system has total entropy $H$ bits. This information exists whether or not any observer accesses it.
:::

::: definition
[]{#def:observer-channel label="def:observer-channel"} \[D:Ddef:observer-channel; R:E,Q,S,AR\] An observer has channel capacity $C$ bits. The observer accesses at most $C$ bits of the system's information.
:::

::: theorem
[]{#thm:information-gap label="thm:information-gap"} \[D:Tthm:information-gap; R:E,Q,S,AR\] When system entropy exceeds channel capacity ($H > C$), the gap $H - C > 0$. The observer cannot access all information.
:::

::: theorem
[]{#thm:gap-physical label="thm:gap-physical"} \[D:Tthm:gap-physical; R:E,Q,S,AR\] The information gap is a physical constraint on access, not a logical incompleteness. The system entropy is well-defined; only the channel is bounded. The logic is complete; access is not.
:::

::: theorem
[]{#thm:competence-access label="thm:competence-access"} \[D:Tthm:competence-access; R:E,Q,S,AR\] Competence is bounded by what you can access, not by logical consistency. For any system with more information than your channel capacity, there exist truths you cannot verify, not because logic fails, but because you lack the bits.
:::

### Dimensional Complexity: Each Subcase Is a Complexity Class {#sec:dimensional-complexity}

Each tractable subcase is a complexity class that collapses to P. Unbounded dimension stays in the base class (coNP, PP, or PSPACE depending on regime).

::: definition
[]{#def:complexity-classes label="def:complexity-classes"} \[D:Ddef:complexity-classes; R:E,Q,S,AR\] The regime hierarchy induces complexity classes: Static $\to$ coNP, Stochastic $\to$ PP, Sequential $\to$ PSPACE. Each tractable subcase defines a sub-regime that reduces to P.
:::

::: theorem
[]{#thm:six-subcases label="thm:six-subcases"} \[D:Tthm:six-subcases; R:E,Q,S,AR,H=tractable-structured\] Each tractable subcase is a complexity class that collapses to P:

1.  Bounded actions $\to$ P

2.  Separable utility $\to$ P

3.  Low tensor rank $\to$ P

4.  Tree structure $\to$ P

5.  Bounded treewidth $\to$ P

6.  Coordinate symmetry $\to$ P
:::

::: theorem
[]{#thm:topology-motion label="thm:topology-motion"} \[D:Tthm:topology-motion; R:E,Q,S,AR\] Scalars live on the circuit's topology. Fixed topology (stationary) preserves tractability. Changing topology (motion) adds dimensions, may break tractability.
:::

::: theorem
[]{#thm:complexity-dichotomy label="thm:complexity-dichotomy"} \[D:Tthm:complexity-dichotomy; R:E,Q,S,AR,H=tractable-structured\] A problem is tractable (in P) iff its effective dimension is bounded. Unbounded dimension stays in the base complexity class (coNP, PP, or PSPACE).
:::


# Tractable Special Cases: When You Can Solve It {#sec:tractable}

Formally: this section gives sufficient structure/representation conditions for polynomial-time SUFFICIENCY-CHECK.

We distinguish the encodings of Section [\[sec:encoding\]](#sec:encoding){reference-type="ref" reference="sec:encoding"}. The tractability results below state the model assumption explicitly. Structural insight: hardness dissolves exactly when the full decision boundary $s \mapsto \operatorname{Opt}(s)$ is recoverable in polynomial time from the input representation; the six cases below instantiate this single principle. Concretely, each tractable regime corresponds to a specific structural insight (explicit boundary exposure, additive separability, tensor factorization, tree structure, bounded treewidth, or coordinate symmetry) that removes the hardness witness; this supports reading the general-case hardness as missing structural access in the current representation rather than as an intrinsic semantic barrier. In physical terms, these are the regimes where required structure is directly observable or factorized so that verification work scales with exposed local structure rather than hidden global combinations on encoded physical instances. \[D:Tthm:tractable, thm:physical-bridge-bundle;Pprop:physical-claim-transport; R:E,TA,AR\]

Informally: these cases are tractable because useful structure is visible.

::: theorem
[]{#thm:tractable label="thm:tractable"} SUFFICIENCY-CHECK is polynomial-time solvable in the following cases:

1.  **Bounded actions (explicit-state encoding):** The input contains the full utility table over $A \times S$. SUFFICIENCY-CHECK runs in $O(|S|^2|A|)$ time; if $|A|$ is constant, $O(|S|^2)$.

2.  **Separable utility (rank 1):** $U(a, s) = f(a) + g(s)$. The optimal action is state-independent, so $I = \emptyset$ is always sufficient. Complexity: $O(1)$ for sufficiency verification.

3.  **Low tensor rank:** $U(a,s) = \sum_{r=1}^R w_r \cdot f_r(a) \cdot \prod_{i=1}^n g_{ri}(s_i)$ for bounded $R$. Complexity: $O(|A| \cdot R \cdot n)$.

4.  **Tree-structured dependencies:** Dependencies form a rooted tree: $$U(a,s) = \sum_i u_i\bigl(a, s_i, s_{\mathrm{parent}(i)}\bigr),$$ with the root term depending only on $(a, s_{\mathrm{root}})$ and all $u_i$ given explicitly. Complexity: $O(n \cdot |A| \cdot k_{\max})$.

5.  **Bounded treewidth:** Utility decomposes as pairwise interactions on an interaction graph of treewidth $w$. Complexity: $O(n \cdot k^{w+1})$ via CSP algorithms [@bodlaender1993tourist; @freuder1990complexity].

6.  **Coordinate symmetry:** State space is $S = [k]^d$ with utility invariant under coordinate permutations. The effective state space reduces from $k^d$ to $\binom{d+k-1}{k-1}$ orbit types. Complexity: $O\bigl(\binom{d+k-1}{k-1}^2\bigr)$, polynomial in $d$ for fixed $k$.
:::

## Bounded Actions (Explicit-State)

::: proof
*Proof of Part 1.* Given the full table of $U(a,s)$, compute $\operatorname{Opt}(s)$ for all $s \in S$ in $O(|S||A|)$ time. For SUFFICIENCY-CHECK on a given $I$, verify that for all pairs $(s,s')$ with $s_I = s'_I$, we have $\operatorname{Opt}(s) = \operatorname{Opt}(s')$. This takes $O(|S|^2|A|)$ time by direct enumeration and is polynomial in the explicit input length. If $|A|$ is constant, the runtime is $O(|S|^2)$. ◻
:::

## Separable Utility (Rank 1)

::: proof
*Proof of Part 2.* If $U(a, s) = f(a) + g(s)$, then $\operatorname{Opt}(s) = \arg\max_{a \in A} f(a)$. The optimal action is independent of the state. Thus $I = \emptyset$ is always sufficient, and sufficiency verification is $O(1)$. ◻
:::

## Low Tensor Rank

::: proof
*Proof of Part 3.* When utility has tensor rank $R$: $U(a,s) = \sum_{r=1}^R w_r f_r(a) \prod_{i=1}^n g_{ri}(s_i)$, the problem reduces to factored tensor computation. For each action, the optimal value computation requires $O(R \cdot n)$ operations. Total complexity: $O(|A| \cdot R \cdot n)$, polynomial for bounded $R$. This reduction cites standard tensor contraction algorithms [@kolda2009tensor; @cichocki2016tensor]. ◻
:::

## Tree-Structured Dependencies

::: proof
*Proof of Part 4.* Assume the tree decomposition and explicit local tables as stated. For each node $i$ and each value of its parent coordinate, compute the set of actions that are optimal for some assignment of the subtree rooted at $i$. This is a bottom-up dynamic program that combines local tables with child summaries; each table lookup is explicit in the input. A coordinate is relevant if and only if varying its value changes the resulting optimal action set. The total runtime is polynomial in $n$, $|A|$, and the size of the local tables. ◻
:::

## Bounded Treewidth

::: proof
*Proof of Part 5.* When utility decomposes as pairwise interactions $U(a,s) = \sum_{(i,j)} u_{ij}(a, s_i, s_j)$, the interaction graph has an edge between coordinates $i,j$ iff $u_{ij}$ is nontrivial. If this graph has treewidth $w$, SUFFICIENCY-CHECK reduces to a constraint satisfaction problem on a bounded-treewidth graph. By standard CSP algorithms [@bodlaender1993tourist; @freuder1990complexity], this is solvable in $O(n \cdot k^{w+1})$ time. ◻
:::

## Coordinate Symmetry

::: proof
*Proof of Part 6.* For state space $S = [k]^d$ with coordinate-permutation-invariant utility, two states have the same optimal actions iff they have the same *orbit type* (value histogram). The number of orbit types is $\binom{d+k-1}{k-1}$ by stars-and-bars. Sufficiency checking reduces to comparing optimal actions across orbit type pairs. Total complexity: $O\bigl(\binom{d+k-1}{k-1}^2 \cdot |A|\bigr)$, polynomial in $d$ for fixed $k$. ◻
:::

## Practical Implications

Formally: this subsection states detect-then-check closure for the declared tractable classes.

::: center
  **Condition**         **Examples**
  --------------------- -----------------------------------------------
  Bounded actions       Small or fully enumerated state spaces
  Separable utility     Additive costs, linear models
  Low tensor rank       Factored decision trees, sparse interactions
  Tree-structured       Hierarchies, causal trees, DAGs
  Bounded treewidth     Local dependency graphs, grid-like structures
  Coordinate symmetry   Exchangeable particles, counting problems
:::

::: proposition
[]{#prop:heuristic-reusability label="prop:heuristic-reusability"} \[D:Pprop:heuristic-reusability; R:H=tractable-structured\] Let $\mathcal{C}$ be a structure class for which SUFFICIENCY-CHECK is polynomial (Theorem [\[thm:tractable\]](#thm:tractable){reference-type="ref" reference="thm:tractable"}). If membership in $\mathcal{C}$ is decidable in polynomial time, then the combined detect-then-check procedure is a sound, polynomial-time heuristic applicable to all future instances in $\mathcal{C}$. For each subcase of Theorem [\[thm:tractable\]](#thm:tractable){reference-type="ref" reference="thm:tractable"}, structure detection is polynomial (under the declared representation assumptions).
:::

::: remark
Proposition [\[prop:heuristic-reusability\]](#prop:heuristic-reusability){reference-type="ref" reference="prop:heuristic-reusability"} removes the complexity-theoretic obstruction to integrity-preserving action on detectable tractable instances: integrity no longer *forces* abstention. Whether a declared controller executes the action still requires competence (Definition [\[def:competence-regime\]](#def:competence-regime){reference-type="ref" reference="def:competence-regime"}): budget and coverage must also be satisfied. A controller that detects structure class $\mathcal{C}$, applies the corresponding polynomial checker, and abstains when detection fails maintains integrity; it never claims sufficiency without verification. Mistaking the heuristic for the general solution, claiming polynomial-time competence on a coNP-hard problem, violates integrity (Proposition [\[prop:integrity-resource-bound\]](#prop:integrity-resource-bound){reference-type="ref" reference="prop:integrity-resource-bound"}).
:::

Informally: reusable heuristics are reliable only after the stated structure test; otherwise hard-case limits still apply.


# Regime-Conditional Corollaries {#sec:engineering-justification}

Formally: this section derives operational corollaries by composing theorem-level hardness, tractability, and typing results under declared regimes.

This section derives regime-typed engineering corollaries from the full map: static/stochastic/sequential placements (Theorems [\[thm:sufficiency-conp\]](#thm:sufficiency-conp){reference-type="ref" reference="thm:sufficiency-conp"}, [\[th:stochastic-sufficiency-pp\]](#th:stochastic-sufficiency-pp){reference-type="ref" reference="th:stochastic-sufficiency-pp"}, [\[th:sequential-sufficiency-pspace\]](#th:sequential-sufficiency-pspace){reference-type="ref" reference="th:sequential-sufficiency-pspace"}) with strict hierarchy (Propositions [\[prop:static-stochastic-strict\]](#prop:static-stochastic-strict){reference-type="ref" reference="prop:static-stochastic-strict"}, [\[prop:stochastic-sequential-strict\]](#prop:stochastic-sequential-strict){reference-type="ref" reference="prop:stochastic-sequential-strict"}), and the static decision/access decomposition used operationally here. Theorem [\[thm:config-reduction\]](#thm:config-reduction){reference-type="ref" reference="thm:config-reduction"} maps configuration simplification to SUFFICIENCY-CHECK; Theorems [\[thm:sufficiency-conp\]](#thm:sufficiency-conp){reference-type="ref" reference="thm:sufficiency-conp"}, [\[thm:minsuff-conp\]](#thm:minsuff-conp){reference-type="ref" reference="thm:minsuff-conp"}, and [\[thm:dichotomy\]](#thm:dichotomy){reference-type="ref" reference="thm:dichotomy"} then yield exact minimization consequences under \[S\] and \[S+ETH\]. The same derivation applies to physical configuration/control systems once they are encoded into the C1--C4 decision interface: "configuration parameter" becomes any declared physical control or observation coordinate, and the same sufficiency predicate governs behavior preservation. \[D:Tthm:config-reduction, thm:sufficiency-conp, th:stochastic-sufficiency-pp, th:sequential-sufficiency-pspace, thm:physical-bridge-bundle;Pprop:physical-claim-transport, prop:static-stochastic-strict, prop:stochastic-sequential-strict; R:AR,RG\]

For deployed software, this remains physical at execution time: each configuration read/write and verification step is realized by electrical state transitions in hardware, so coordinate minimization and over-specification map directly to measured switching-energy load under the declared bit-to-energy lift. \[D:Pprop:thermo-lift, prop:run-time-accounting; R:AR,H=cost-growth\]

Under the same discrete-time event law, more verification cycles mean more decision-event ticks and more bit operations; with positive per-bit conversion, this induces additional thermodynamically necessary heat-transfer load. \[D:Pprop:decision-unit-time, prop:run-time-accounting, prop:thermo-lift;Ccor:neukart-vinokur; R:AR,H=cost-growth\]

Regime tags used below follow Section [\[sec:model-contract\]](#sec:model-contract){reference-type="ref" reference="sec:model-contract"}: \[S\], \[S+ETH\], \[E\], \[S_bool\]. Any prescription that requires exact minimization is constrained by these theorem-level bounds. \[D:Tthm:sufficiency-conp, thm:minsuff-conp, thm:dichotomy; R:S,S+ETH\] Theorem [\[thm:overmodel-diagnostic\]](#thm:overmodel-diagnostic){reference-type="ref" reference="thm:overmodel-diagnostic"} implies that persistent failure to isolate a minimal sufficient set is a boundary-characterization signal in the current model, not a universal irreducibility claim. \[D:Tthm:overmodel-diagnostic; R:S\]

#### Conditional rationality criterion.

For the objective "minimize verified total cost while preserving integrity," persistent over-specification is admissible only in the *attempted competence failure* cell (Definition [\[def:attempted-competence-failure\]](#def:attempted-competence-failure){reference-type="ref" reference="def:attempted-competence-failure"}): after an attempted exact procedure, if exact irrelevance cannot be certified efficiently, integrity retains uncertified coordinates rather than excluding them. \[D:Pprop:attempted-competence-matrix; R:AR\] When exact competence is available in the active regime (e.g., Theorem [\[thm:tractable\]](#thm:tractable){reference-type="ref" reference="thm:tractable"} and the exact-identifiability criterion), proven-irrelevant coordinates are removable; persistent over-specification is irrational for that same objective. \[D:Tthm:tractable;Pprop:attempted-competence-matrix; R:E,STR\] Proposition [\[prop:attempted-competence-matrix\]](#prop:attempted-competence-matrix){reference-type="ref" reference="prop:attempted-competence-matrix"} makes this explicit: in the integrity-preserving matrix, one cell is rational and three are irrational, so irrationality is the default verdict. \[D:Pprop:attempted-competence-matrix; R:AR\]

::: remark
All claims in this section are formal corollaries under the stated model contract.

-   Competence claims are indexed by the regime tuple of Definition [\[def:competence-regime\]](#def:competence-regime){reference-type="ref" reference="def:competence-regime"}; prescriptions are meaningful only relative to feasible resources under that regime (bounded-rationality feasibility discipline) [@sep_bounded_rationality; @arora2009computational]. \[D:Pprop:integrity-competence-separation; R:AR\]

-   Integrity (Definition [\[def:solver-integrity\]](#def:solver-integrity){reference-type="ref" reference="def:solver-integrity"}) forbids overclaiming beyond certifiable outputs; $\mathsf{ABSTAIN}$/$\mathsf{UNKNOWN}$ is first-class when certification is unavailable. \[D:Pprop:integrity-competence-separation; R:AR\]

-   Therefore, hardness results imply a regime-conditional trilemma: abstain, weaken guarantees (heuristics/approximation), or change encoding/structural assumptions to recover competence. \[D:Tthm:sufficiency-conp, thm:dichotomy;Pprop:attempted-competence-matrix; R:S,S+ETH\]
:::

Informally: these recommendations come directly from proved results under stated regimes.

## Configuration Simplification is SUFFICIENCY-CHECK

Formally: this subsection proves configuration-preservation queries instantiate SUFFICIENCY-CHECK under C1--C4.

Because SUFFICIENCY-CHECK is a meta-problem parameterized by an arbitrary decision problem $\mathcal{D}$, real engineering problems with factored configuration spaces are instances of the hardness landscape established above.

::: theorem
[]{#thm:config-reduction label="thm:config-reduction"} Given a software system with configuration parameters $P = \{p_1, \ldots, p_n\}$ and observed behaviors $B = \{b_1, \ldots, b_m\}$, the problem of determining whether parameter subset $I \subseteq P$ preserves all behaviors is equivalent to SUFFICIENCY-CHECK.
:::

::: proof
*Proof.* Construct decision problem $\mathcal{D} = (A, X_1, \ldots, X_n, U)$ where:

-   Actions $A = B \cup \{\bot\}$, where $\bot$ is a sentinel "no-observed-behavior" action

-   Coordinates $X_i$ = domain of parameter $p_i$

-   State space $S = X_1 \times \cdots \times X_n$

-   For $b\in B$, utility $U(b, s) = 1$ if behavior $b$ occurs under configuration $s$, else $0$

-   Sentinel utility $U(\bot,s)=1$ iff no behavior in $B$ occurs under configuration $s$, else $0$

Then $$\operatorname{Opt}(s)=
\begin{cases}
\{b \in B : b \text{ occurs under configuration } s\}, & \text{if this set is nonempty},\\
\{\bot\}, & \text{otherwise}.
\end{cases}$$ So the optimizer map exactly encodes observed-behavior equivalence classes, including the empty-behavior case.

Coordinate set $I$ is sufficient iff: $$s_I = s'_I \implies \operatorname{Opt}(s) = \operatorname{Opt}(s')$$

This holds iff configurations agreeing on parameters in $I$ exhibit identical behaviors.

Therefore, "does parameter subset $I$ preserve all behaviors?" is exactly SUFFICIENCY-CHECK for the constructed decision problem. ◻
:::

::: remark
The reduction above requires only:

1.  a finite behavior set,

2.  parameters with finite domains, and

3.  an evaluable behavior map from configurations to achieved behaviors.

These are exactly the model-contract premises C1--C3 instantiated for configuration systems.
:::

::: theorem
[]{#thm:overmodel-diagnostic label="thm:overmodel-diagnostic"} By contraposition of Theorem [\[thm:config-reduction\]](#thm:config-reduction){reference-type="ref" reference="thm:config-reduction"}, if no coordinate set can be certified as exactly relevance-identifying (Definition [\[def:exact-identifiability\]](#def:exact-identifiability){reference-type="ref" reference="def:exact-identifiability"}) for the modeled system, then the decision boundary is not completely characterized by the current parameterization.
:::

::: proof
*Proof.* Assume the decision boundary were completely characterized by the current parameterization. Then, via Theorem [\[thm:config-reduction\]](#thm:config-reduction){reference-type="ref" reference="thm:config-reduction"}, the corresponding sufficiency instance admits exact relevance membership, hence a coordinate set that satisfies Definition [\[def:exact-identifiability\]](#def:exact-identifiability){reference-type="ref" reference="def:exact-identifiability"}. Contraposition gives the claim: persistent failure of exact relevance identification signals incomplete characterization of decision relevance in the model. ◻
:::

Informally: if a minimal relevant set cannot be certified, that indicates an unresolved model boundary, not proof that every retained parameter is necessary.

## Cost Asymmetry Under ETH

Formally: this subsection proves asymptotic finding-vs-maintenance asymmetry under succinct ETH and the declared cost model.

We now prove a cost asymmetry result under the stated cost model and complexity constraints.[^1]

::: theorem
[]{#thm:cost-asymmetry-eth label="thm:cost-asymmetry-eth"} Consider an engineer specifying a system configuration with $n$ parameters. Let:

-   $C_{\text{over}}(k)$ = cost of maintaining $k$ extra parameters beyond minimal

-   $C_{\text{find}}(n)$ = cost of finding minimal sufficient parameter set

-   $C_{\text{under}}$ = expected cost of production failures from underspecification

Assume ETH in the succinct encoding model of Section [\[sec:encoding\]](#sec:encoding){reference-type="ref" reference="sec:encoding"}. Then:

1.  Exact identification of minimal sufficient sets has worst-case finding cost $C_{\text{find}}(n)=2^{\Omega(n)}$. (Under ETH, SUFFICIENCY-CHECK has a $2^{\Omega(n)}$ lower bound in the succinct model, and exact minimization subsumes this decision task.)

2.  Maintenance cost is linear: $C_{\text{over}}(k) = O(k)$.

3.  Under ETH, exponential finding cost dominates linear maintenance cost for sufficiently large $n$.

Therefore, there exists $n_0$ such that for all $n > n_0$, the finding-vs-maintenance asymmetry satisfies: $$C_{\text{over}}(k) < C_{\text{find}}(n) + C_{\text{under}}$$

Within \[S+ETH\], persistent over-specification is unresolved boundary characterization, not proof that all included parameters are intrinsically necessary. Conversely, when exact competence is available in the active regime, certifiably irrelevant coordinates are removable; persistence is irrational for the same cost-minimization objective. \[D:Tthm:cost-asymmetry-eth;Pprop:attempted-competence-matrix; R:S+ETH,TA\]
:::

::: proof
*Proof.* Under ETH, the TAUTOLOGY reduction used in Theorem [\[thm:sufficiency-conp\]](#thm:sufficiency-conp){reference-type="ref" reference="thm:sufficiency-conp"} yields a $2^{\Omega(n)}$ worst-case lower bound for SUFFICIENCY-CHECK in the succinct encoding. Any exact algorithm that outputs a minimum sufficient set can decide whether the optimum size is $0$ by checking whether the returned set is empty; this is exactly the SUFFICIENCY-CHECK query for $I=\emptyset$. Hence exact minimal-set finding inherits the same exponential worst-case lower bound.

Maintaining $k$ extra parameters incurs:

-   Documentation cost: $O(k)$ entries

-   Testing cost: $O(k)$ test cases

-   Migration cost: $O(k)$ parameters to update

Total maintenance cost is $C_{\text{over}}(k) = O(k)$.

The eventual dominance step is mechanized in HD14: for fixed linear-overhead parameter $k$ and additive constant $C_{\text{under}}$ there is $n_0$ such that $k < 2^n + C_{\text{under}}$ for all $n \ge n_0$. Therefore: $$C_{\text{over}}(k) \ll C_{\text{find}}(n)$$

For any fixed nonnegative $C_{\text{under}}$, the asymptotic dominance inequality remains and only shifts the finite threshold $n_0$. ◻
:::

::: corollary
[]{#cor:no-auto-minimize label="cor:no-auto-minimize"} \[D:Ccor:no-auto-minimize; R:H=succinct-hard\] Assuming $P\neq coNP$, there exists no polynomial-time algorithm that:

1.  Takes an arbitrary configuration file with $n$ parameters

2.  Identifies the minimal sufficient parameter subset

3.  Guarantees correctness (no false negatives)
:::

::: proof
*Proof.* Such an algorithm would solve MINIMUM-SUFFICIENT-SET in polynomial time, contradicting Theorem [\[thm:minsuff-conp\]](#thm:minsuff-conp){reference-type="ref" reference="thm:minsuff-conp"} (assuming $P\neq coNP$). ◻
:::

::: remark
Corollary [\[cor:no-auto-minimize\]](#cor:no-auto-minimize){reference-type="ref" reference="cor:no-auto-minimize"} is a formal boundary statement: an always-correct polynomial-time minimizer for arbitrary succinct inputs would collapse $P$ and $coNP$.
:::

::: remark
The practical force of worst-case hardness depends on instance structure, especially $k^*$. If SUFFICIENCY-CHECK is FPT in parameter $k^*$, then small-$k^*$ families can remain tractable even under succinct encodings. The strengthened mechanized gadget (`all_coords_relevant_of_not_tautology`) still proves existence of hard families with $k^*=n$; what is typical in deployed systems is an empirical question outside this formal model.
:::

Informally: at scale, exact minimization can cost more than linear maintenance in worst cases, while structured small-parameter families may still be tractable.

## Regime-Conditional Operational Corollaries

Formally: this subsection instantiates admissible policy outcomes from prior theorems and integrity-competence typing.

Theorems [\[thm:overmodel-diagnostic\]](#thm:overmodel-diagnostic){reference-type="ref" reference="thm:overmodel-diagnostic"} and [\[thm:cost-asymmetry-eth\]](#thm:cost-asymmetry-eth){reference-type="ref" reference="thm:cost-asymmetry-eth"} yield the following conditional operational consequences:

**1. Conservative retention under unresolved relevance.** If irrelevance cannot be certified efficiently under the active regime, the admissible policy is to retain a conservative superset of parameters. \[D:Tthm:overmodel-diagnostic; R:S\]

**2. Heuristic selection as weakened-guarantee mode.** Under \[S+ETH\], exact global minimization can be exponentially costly in the worst case (Theorem [\[thm:cost-asymmetry-eth\]](#thm:cost-asymmetry-eth){reference-type="ref" reference="thm:cost-asymmetry-eth"}); methods such as AIC/BIC/cross-validation instantiate the "weaken guarantees" branch of Definition [\[def:competence-regime\]](#def:competence-regime){reference-type="ref" reference="def:competence-regime"} [@akaike1974new; @schwarz1978estimating; @stone1974cross; @guyon2003introduction; @pearl2009causality]. \[D:Tthm:cost-asymmetry-eth; R:S+ETH\]

**3. Full-parameter inclusion as an $O(n)$ upper-bound strategy.** Under \[S+ETH\], if exact minimization is unresolved, including all $n$ parameters is an $O(n)$ maintenance upper-bound policy that avoids false irrelevance claims. \[D:Tthm:cost-asymmetry-eth; R:S+ETH\]

**4. Irrationality outside attempted-competence-failure conditions.** If the active regime admits exact competence (tractable structural-access conditions or exact relevance identifiability), or if exact competence was never actually attempted, continued over-specification is irrational; hardness no longer justifies it for the stated objective. \[D:Tthm:tractable;Pprop:attempted-competence-matrix; R:TA\]

These corollaries are direct consequences of the hardness/tractability landscape: over-specification is an attempted-competence-failure policy, not a default optimum. Outside that cell, the admissible exits are to shift to tractable regimes from Theorem [\[thm:tractable\]](#thm:tractable){reference-type="ref" reference="thm:tractable"} or adopt explicit approximation commitments. \[D:Tthm:tractable;Pprop:attempted-competence-matrix; R:RG\]

Informally: policy choices are threefold: abstain, weaken guarantees, or change assumptions to recover competence.

[^1]: Naive subset enumeration still gives an intuitive baseline of $O(2^n)$ checks, but that is an algorithmic upper bound; the theorem below uses ETH for the lower-bound argument.


# Dominance Theorems for Hardness Placement {#sec:implications}

Regime for this section: the mechanized Boolean-coordinate model \[S_bool\] plus the architecture cost model defined below. The centralization-vs-distribution framing follows established software-architecture criteria about modular decomposition and long-run maintenance tradeoffs [@parnas1972criteria; @brooks1987no; @bass2012software; @shaw1996software]. Although phrased in software terms, the formal objects are substrate-agnostic: the same hardness-placement and amortization equations apply to physical architectures (for example, centralized vs distributed sensing/actuation computation) once mapped into the same coordinate/cost model. \[D:Pprop:physical-claim-transport;Tthm:physical-bridge-bundle, thm:centralization-dominance, thm:tax-conservation, thm:amortization; R:AR,LC\]

In electrical implementations, the same architecture equations are energy equations: each distributed update path induces switching events and Joule expenditure, while centralized one-time structure pays setup cost and reduces repeated per-site electrical work. Under the declared thermodynamic lift, this is tracked as a typed transfer from bit-operation load to physical energy load. \[D:Pprop:thermo-lift;Ccor:neukart-vinokur; R:AR,H=cost-growth\]

Equivalently, repeated distributed cycles create a thermal duty: more update cycles imply more bit operations, which imply more unavoidable dissipated heat and therefore larger heat-transfer/removal requirements in the same substrate class. \[D:Pprop:decision-unit-time, prop:run-time-accounting, prop:thermo-lift; R:AR,H=cost-growth\]

## Over-Specification as Diagnostic Signal

::: corollary
[]{#cor:overmodel-diagnostic-implication label="cor:overmodel-diagnostic-implication"} \[D:Ccor:overmodel-diagnostic-implication; R:H=succinct-hard\] In the mechanized Boolean-coordinate model, if a coordinate is relevant and omitted from a candidate set $I$, then $I$ is not sufficient.
:::

::: proof
*Proof.* This is the contrapositive of S2P3. ◻
:::

::: corollary
[]{#cor:exact-identifiability label="cor:exact-identifiability"} \[D:Ccor:exact-identifiability; R:H=succinct-hard\] In the mechanized Boolean-coordinate model, for any candidate set $I$: $$I \text{ is exactly relevance-identifying}
\iff
\bigl(I \text{ is sufficient and } I \subseteq R_{\mathrm{rel}}\bigr),$$ where $R_{\mathrm{rel}}$ is the full relevant-coordinate set.
:::

::: proof
*Proof.* This is exactly S2P1, with $R_{\mathrm{rel}}=\texttt{relevantFinset}$. ◻
:::

::: remark
[]{#rem:overmodel-conditional label="rem:overmodel-conditional"} In this paper's formal typing, persistent over-specification is admissible only under attempted competence failure: after an attempted exact procedure, exact relevance competence remains unavailable in the active regime, so integrity retains uncertified coordinates (Section [\[sec:interpretive-foundations\]](#sec:interpretive-foundations){reference-type="ref" reference="sec:interpretive-foundations"}; Section [\[sec:engineering-justification\]](#sec:engineering-justification){reference-type="ref" reference="sec:engineering-justification"}). \[D:Pprop:attempted-competence-matrix; R:AR\] Once exact competence is available in the active regime (Corollaries [\[cor:practice-bounded\]](#cor:practice-bounded){reference-type="ref" reference="cor:practice-bounded"}--[\[cor:practice-symmetry\]](#cor:practice-symmetry){reference-type="ref" reference="cor:practice-symmetry"} together with Corollary [\[cor:exact-identifiability\]](#cor:exact-identifiability){reference-type="ref" reference="cor:exact-identifiability"}), certifiably irrelevant coordinates are removable; persistent over-specification is irrational for the same objective (verified total-cost minimization). \[D:Tthm:tractable;Pprop:attempted-competence-matrix; R:TA\]
:::

## Architectural Decision Quotient

::: definition
For a software system with configuration space $S$ and behavior space $B$: $$\text{ADQ}(I) = \frac{|\{b \in B : b \text{ achievable with some } s \text{ where } s_I \text{ fixed}\}|}{|B|}$$
:::

::: proposition
[]{#prop:adq-ordering label="prop:adq-ordering"} For coordinate sets $I,J$ in the same system, if $\mathrm{ADQ}(I) < \mathrm{ADQ}(J)$, then fixing $I$ leaves a strictly smaller achievable-behavior set than fixing $J$.
:::

::: proof
*Proof.* The denominator $|B|$ is shared. Thus $\mathrm{ADQ}(I) < \mathrm{ADQ}(J)$ is equivalent to a strict inequality between the corresponding achievable-behavior set cardinalities. ◻
:::

## Corollaries for Practice

::: corollary
[]{#cor:practice-diagnostic label="cor:practice-diagnostic"} \[D:Ccor:practice-diagnostic; R:H=succinct-hard\] In the mechanized Boolean-coordinate model, existence of a sufficient set of size at most $k$ is equivalent to the relevance set having cardinality at most $k$.
:::

::: proof
*Proof.* By S2P2, sufficiency of size $\le k$ is equivalent to a relevance-cardinality bound $\le k$ in the Boolean-coordinate model. ◻
:::

::: corollary
[]{#cor:practice-bounded label="cor:practice-bounded"} \[D:Ccor:practice-bounded; R:H=tractable-structured\] When the bounded-action or explicit-state conditions of Theorem [\[thm:tractable\]](#thm:tractable){reference-type="ref" reference="thm:tractable"} hold, minimal modeling can be solved in polynomial time in the stated input size.
:::

::: proof
*Proof.* This is the bounded-regime branch of Theorem [\[thm:tractable\]](#thm:tractable){reference-type="ref" reference="thm:tractable"}, mechanized as CC81. ◻
:::

::: corollary
[]{#cor:practice-structured label="cor:practice-structured"} \[D:Ccor:practice-structured; R:H=tractable-structured\] When utility is separable with explicit factors, sufficiency checking is polynomial in the explicit-state regime.
:::

::: proof
*Proof.* This is the separable-utility branch of Theorem [\[thm:tractable\]](#thm:tractable){reference-type="ref" reference="thm:tractable"}, mechanized as CC82. ◻
:::

::: corollary
[]{#cor:practice-tree label="cor:practice-tree"} \[D:Ccor:practice-tree; R:H=tractable-structured\] When utility factors form a tree structure with explicit local factors, sufficiency checking is polynomial in the explicit-state regime.
:::

::: proof
*Proof.* This is the tree-factor branch of Theorem [\[thm:tractable\]](#thm:tractable){reference-type="ref" reference="thm:tractable"}, mechanized as CC84. ◻
:::

::: corollary
[]{#cor:practice-tensor label="cor:practice-tensor"} \[D:Ccor:practice-tensor; R:H=tractable-structured\] When utility has tensor rank $R$ with explicit factor decomposition $U(a,s) = \sum_{r=1}^R w_r f_r(a) \prod_{i=1}^n g_{ri}(s_i)$, sufficiency checking is polynomial in $|A| \cdot R \cdot n$.
:::

::: proof
*Proof.* This is the low-tensor-rank branch of Theorem [\[thm:tractable\]](#thm:tractable){reference-type="ref" reference="thm:tractable"}, mechanized as CC71. ◻
:::

::: corollary
[]{#cor:practice-treewidth label="cor:practice-treewidth"} \[D:Ccor:practice-treewidth; R:H=tractable-structured\] When utility decomposes as pairwise interactions on a graph of treewidth $w$, sufficiency checking is polynomial in $n \cdot k^{w+1}$.
:::

::: proof
*Proof.* This is the bounded-treewidth branch of Theorem [\[thm:tractable\]](#thm:tractable){reference-type="ref" reference="thm:tractable"}, mechanized as CC73. ◻
:::

::: corollary
[]{#cor:practice-symmetry label="cor:practice-symmetry"} \[D:Ccor:practice-symmetry; R:H=tractable-structured\] When state space is $S = [k]^d$ with coordinate-permutation-invariant utility, sufficiency checking is polynomial in $\binom{d+k-1}{k-1}^2 \cdot |A|$, which is polynomial in $d$ for fixed $k$.
:::

::: proof
*Proof.* This is the coordinate-symmetry branch of Theorem [\[thm:tractable\]](#thm:tractable){reference-type="ref" reference="thm:tractable"}, mechanized as CC72. ◻
:::

::: corollary
[]{#cor:practice-unstructured label="cor:practice-unstructured"} There is a machine-checked family of reduction instances where, for non-tautological source formulas, every coordinate is relevant ($k^*=n$), exhibiting worst-case boundary complexity.
:::

::: proof
*Proof.* The strengthened reduction proves that non-tautological source formulas induce instances where every coordinate is relevant; this is mechanized as CC26. ◻
:::

## Hardness Distribution: Right Place vs Wrong Place {#sec:hardness-distribution}

::: definition
[]{#def:hardness-distribution label="def:hardness-distribution"} Let $P$ be a problem family under the succinct encoding of Section [\[sec:encoding\]](#sec:encoding){reference-type="ref" reference="sec:encoding"}. In this section, baseline hardness $H(P;n)$ denotes worst-case computational step complexity on instances with $n$ coordinates (equivalently, as a function of succinct input length $L$) in the fixed encoding regime. A *solution architecture* $S$ partitions this baseline hardness into:

-   $H_{\text{central}}(S)$: hardness paid once, at design time or in a shared component

-   $H_{\text{distributed}}(S)$: hardness paid per use site

For $n$ use sites, total realized hardness is: $$H_{\text{total}}(S) = H_{\text{central}}(S) + n \cdot H_{\text{distributed}}(S)$$
:::

::: proposition
[]{#prop:hardness-conservation label="prop:hardness-conservation"} \[D:Pprop:hardness-conservation; R:H=cost-growth\] For any problem family $P$ measured by $H(P;n)$ above, any solution architecture $S$ and any number of use sites $n \ge 1$, if $H_{\text{total}}(S)$ is measured in the same worst-case step units over the same input family, then: $$H_{\text{total}}(S) = H_{\text{central}}(S) + n \cdot H_{\text{distributed}}(S) \geq H(P;n).$$ For SUFFICIENCY-CHECK, Theorem [\[thm:dichotomy\]](#thm:dichotomy){reference-type="ref" reference="thm:dichotomy"} provides the baseline on the hard succinct family: $H(\text{SUFFICIENCY-CHECK};n)=2^{\Omega(n)}$ under ETH.
:::

::: proof
*Proof.* By definition, $H(P;n)$ is a worst-case lower bound for correct solutions in this encoding regime and cost metric. Any such solution architecture decomposes total realized work as $H_{\text{central}} + n\cdot H_{\text{distributed}}$, so that total cannot fall below the baseline. ◻
:::

::: definition
[]{#def:hardness-efficiency label="def:hardness-efficiency"} The *hardness efficiency* of solution $S$ with $n$ use sites is: $$\eta(S, n) = \frac{H_{\text{central}}(S)}{H_{\text{central}}(S) + n \cdot H_{\text{distributed}}(S)}$$
:::

::: proposition
[]{#prop:hardness-efficiency-interpretation label="prop:hardness-efficiency-interpretation"} \[D:Pprop:hardness-efficiency-interpretation; R:H=cost-growth\] For fixed $n$ and positive total hardness, larger $\eta(S,n)$ is equivalent to a larger central share of realized hardness.
:::

::: proof
*Proof.* From Definition [\[def:hardness-efficiency\]](#def:hardness-efficiency){reference-type="ref" reference="def:hardness-efficiency"}, $\eta(S,n)$ is exactly the fraction of total realized hardness paid centrally. ◻
:::

::: definition
[]{#def:right-wrong-placement label="def:right-wrong-placement"} For a solution architecture $S$ in this linear model: $$\text{right hardness} \iff H_{\mathrm{distributed}}(S)=0,\qquad
\text{wrong hardness} \iff H_{\mathrm{distributed}}(S)>0.$$
:::

::: theorem
[]{#thm:centralization-dominance label="thm:centralization-dominance"} Let $S_{\mathrm{right}}, S_{\mathrm{wrong}}$ be architectures over the same problem family with $$H_{\mathrm{distributed}}(S_{\mathrm{right}})=0,\quad
H_{\mathrm{central}}(S_{\mathrm{right}})>0,\quad
H_{\mathrm{distributed}}(S_{\mathrm{wrong}})>0,$$ and let $n > \max\!\bigl(1, H_{\mathrm{central}}(S_{\mathrm{right}})\bigr)$. Then:

1.  Lower total realized hardness: $$H_{\text{total}}(S_{\mathrm{right}}) < H_{\text{total}}(S_{\mathrm{wrong}})$$

2.  Fewer error sites: errors in centralized components affect 1 location; errors in distributed components affect $n$ locations

3.  Quantified leverage: moving one unit of work from distributed to central saves exactly $n-1$ units of total realized hardness
:::

::: proof
*Proof.* (1) and (2) are exactly the bundled theorem HD1. (3) is exactly HD2. ◻
:::

::: corollary
[]{#cor:right-wrong-hardness label="cor:right-wrong-hardness"} \[D:Ccor:right-wrong-hardness; R:H=cost-growth\] In the linear model, a right-hardness architecture strictly dominates a wrong-hardness architecture once use-site count exceeds central one-time hardness. Formally, for architectures $S_{\mathrm{right}}, S_{\mathrm{wrong}}$ over the same problem family, if $S_{\mathrm{right}}$ has right hardness, $S_{\mathrm{wrong}}$ has wrong hardness, and $n > H_{\mathrm{central}}(S_{\mathrm{right}})$, then $$H_{\mathrm{central}}(S_{\mathrm{right}}) + n\,H_{\mathrm{distributed}}(S_{\mathrm{right}})
<
H_{\mathrm{central}}(S_{\mathrm{wrong}}) + n\,H_{\mathrm{distributed}}(S_{\mathrm{wrong}}).$$
:::

::: proof
*Proof.* This is the mechanized theorem HD19. ◻
:::

::: proposition
[]{#prop:dominance-modes label="prop:dominance-modes"} \[D:Pprop:dominance-modes; R:H=cost-growth\] This section uses two linear-model dominance modes plus generalized nonlinear dominance and boundary modes:

1.  **Strict threshold dominance:** Corollary [\[cor:right-wrong-hardness\]](#cor:right-wrong-hardness){reference-type="ref" reference="cor:right-wrong-hardness"} gives strict inequality once $n > H_{\mathrm{central}}(S_{\mathrm{right}})$.

2.  **Global weak dominance:** under the decomposition identity used in HD3, centralized hardness placement is never worse for all $n\ge 1$.

3.  **Generalized nonlinear dominance:** under bounded-vs-growing site-cost assumptions (Theorem [\[thm:generalized-dominance\]](#thm:generalized-dominance){reference-type="ref" reference="thm:generalized-dominance"}), right placement strictly dominates beyond a finite threshold without assuming linear per-site cost.

4.  **Generalized boundary mode:** without those growth-separation assumptions, strict dominance is not guaranteed (Proposition [\[prop:generalized-assumption-boundary\]](#prop:generalized-assumption-boundary){reference-type="ref" reference="prop:generalized-assumption-boundary"}).
:::

::: proof
*Proof.* Part (1) is Corollary [\[cor:right-wrong-hardness\]](#cor:right-wrong-hardness){reference-type="ref" reference="cor:right-wrong-hardness"}. Part (2) is exactly HD3. Part (3) is Theorem [\[thm:generalized-dominance\]](#thm:generalized-dominance){reference-type="ref" reference="thm:generalized-dominance"}. Part (4) is Proposition [\[prop:generalized-assumption-boundary\]](#prop:generalized-assumption-boundary){reference-type="ref" reference="prop:generalized-assumption-boundary"}. ◻
:::

**Illustrative Instantiation (Type Systems).** Consider a capability $C$ (e.g., provenance tracking) with one-time central cost $H_{\text{central}}$ and per-site manual cost $H_{\text{distributed}}$:

::: center
  **Approach**                  $H_{\text{central}}$     $H_{\text{distributed}}$
  ---------------------------- ----------------------- -----------------------------
  Native type system support    High (learning cost)    Low (type checker enforces)
  Manual implementation         Low (no new concepts)   High (reimplement per site)
:::

The table is schematic; the formal statement is Corollary [\[cor:type-system-threshold\]](#cor:type-system-threshold){reference-type="ref" reference="cor:type-system-threshold"}. The type-theoretic intuition behind this instantiation is consistent with classic accounts of abstraction and static interface guarantees [@Cardelli1985; @reynolds1983types; @pierce2002types; @liskov1994behavioral; @siek2006gradual; @gamma1994design].

::: corollary
[]{#cor:type-system-threshold label="cor:type-system-threshold"} \[D:Ccor:type-system-threshold; R:H=cost-growth\] For the formal native-vs-manual architecture instance, native support has lower total realized cost for all $$n > H_{\mathrm{baseline}}(P),$$ where $H_{\mathrm{baseline}}(P)$ corresponds to HD15.
:::

::: proof
*Proof.* Immediate from HD15. ◻
:::

## Extension: Non-Additive Site-Cost Models {#sec:nonadditive-site-costs}

::: definition
[]{#def:generalized-site-accumulation label="def:generalized-site-accumulation"} Let $C_S : \mathbb{N} \to \mathbb{N}$ be a per-site accumulation function for architecture $S$. Define generalized total realized hardness by $$H_{\text{total}}^{\mathrm{gen}}(S,n) = H_{\text{central}}(S) + C_S(n).$$
:::

::: definition
[]{#def:eventual-saturation label="def:eventual-saturation"} A cost function $f : \mathbb{N}\to\mathbb{N}$ is *eventually saturating* if there exists $N$ such that for all $n\ge N$, $f(n)=f(N)$.
:::

::: theorem
[]{#thm:generalized-dominance label="thm:generalized-dominance"} \[D:Tthm:generalized-dominance; R:H=cost-growth\] Let $$H_{\text{total}}^{\mathrm{gen}}(S,n)=H_{\text{central}}(S)+C_S(n).$$ For two architectures $S_{\mathrm{right}},S_{\mathrm{wrong}}$, suppose there exists $B\in\mathbb{N}$ such that:

1.  $C_{S_{\mathrm{right}}}(m)\le B$ for all $m$ (bounded right-side per-site accumulation),

2.  $m \le C_{S_{\mathrm{wrong}}}(m)$ for all $m$ (identity-lower-bounded wrong-side growth).

Then for every $$n > H_{\text{central}}(S_{\mathrm{right}})+B,$$ one has $$H_{\text{total}}^{\mathrm{gen}}(S_{\mathrm{right}},n)
<
H_{\text{total}}^{\mathrm{gen}}(S_{\mathrm{wrong}},n).$$
:::

::: proof
*Proof.* This is exactly the mechanized theorem HD9. ◻
:::

::: corollary
[]{#cor:generalized-eventual-dominance label="cor:generalized-eventual-dominance"} \[D:Ccor:generalized-eventual-dominance; R:H=cost-growth\] If condition (1) above holds and there exists $N$ such that condition (2) holds for all $m\ge N$, then there exists $N_0$ such that for all $n\ge N_0$, $$H_{\text{total}}^{\mathrm{gen}}(S_{\mathrm{right}},n)
<
H_{\text{total}}^{\mathrm{gen}}(S_{\mathrm{wrong}},n).$$
:::

::: proof
*Proof.* Immediate from HD10. ◻
:::

::: proposition
[]{#prop:generalized-assumption-boundary label="prop:generalized-assumption-boundary"} \[D:Pprop:generalized-assumption-boundary; R:H=cost-growth\] In the generalized model, strict right-vs-wrong dominance is not unconditional. There are explicit counterexamples:

1.  If wrong-side growth lower bounds are dropped, right-side strict dominance can fail for all $n$.

2.  If right-side boundedness is dropped, strict dominance can fail for all $n$ even when wrong-side growth is linear.
:::

::: proof
*Proof.* This is exactly the pair of mechanized boundary theorems listed above. ◻
:::

::: theorem
[]{#thm:linear-saturation-iff-zero label="thm:linear-saturation-iff-zero"} \[D:Tthm:linear-saturation-iff-zero; R:H=cost-growth\] In the linear model of this section, $$H_{\text{total}}(S,n)=H_{\text{central}}(S)+n\cdot H_{\text{distributed}}(S),$$ the function $n\mapsto H_{\text{total}}(S,n)$ is eventually saturating if and only if $H_{\text{distributed}}(S)=0$.
:::

::: proof
*Proof.* This is exactly the mechanized equivalence theorem above. ◻
:::

::: theorem
[]{#thm:generalized-saturation-possible label="thm:generalized-saturation-possible"} \[D:Tthm:generalized-saturation-possible; R:H=cost-growth\] There exists a generalized site-cost model with eventual saturation. In particular, for $$C_K(n)=\begin{cases}
n, & n\le K\\
K, & n>K,
\end{cases}$$ both $C_K$ and $n\mapsto H_{\text{central}}+C_K(n)$ are eventually saturating.
:::

::: proof
*Proof.* This is the explicit construction mechanized in Lean. ◻
:::

::: corollary
[]{#cor:linear-positive-no-saturation label="cor:linear-positive-no-saturation"} \[D:Ccor:linear-positive-no-saturation; R:H=cost-growth\] No positive-slope linear per-site model can represent the saturating family above for all $n$.
:::

::: proof
*Proof.* This follows from the mechanized theorem that any linear representation of the saturating family must have zero slope. ◻
:::

#### Mechanized strengthening reference.

The strengthened all-coordinates-relevant reduction is presented in Section [\[sec:hardness\]](#sec:hardness){reference-type="ref" reference="sec:hardness"} ("Mechanized strengthening") and formalized in `Reduction_AllCoords.lean` via `all_coords_relevant_of_not_tautology`.

The next section develops the major practical consequence of this framework: the Simplicity Tax Theorem.


# Redistribution Identity for Decision-Relevant Coordinates {#sec:simplicity-tax}

Formally: this section defines expressive-gap tax and proves redistribution, growth, and amortization identities under fixed required-coordinate sets.

The load-bearing fact in this section is not the set identity itself; it is the difficulty of shrinking the required set $R(P)$ in the first place. By Theorem [\[thm:sufficiency-conp\]](#thm:sufficiency-conp){reference-type="ref" reference="thm:sufficiency-conp"} (and Theorem [\[thm:minsuff-conp\]](#thm:minsuff-conp){reference-type="ref" reference="thm:minsuff-conp"} for minimization), exact relevance identification is intractable in the worst case under succinct encoding. The identities below therefore quantify how unresolved relevance is redistributed between central and per-site work. \[D:Tthm:sufficiency-conp, thm:minsuff-conp; R:S\] For encoded physical systems, the same redistribution laws apply under physical-to-core transport: unresolved coordinates remain per-site external work, while resolved coordinates are paid once in centralized structure. \[D:Pprop:physical-claim-transport;Tthm:tax-conservation, thm:tax-grows, thm:amortization; R:AR,LC\]

::: definition
Let $R(P)$ be the required dimensions (those affecting $\operatorname{Opt}$) and $A(M)$ the dimensions model $M$ handles natively. The *expressive gap* is $\text{Gap}(M,P) = R(P) \setminus A(M)$.
:::

::: definition
[]{#def:simplicity-tax label="def:simplicity-tax"} The *simplicity tax* is the size of the expressive gap: $$\text{SimplicityTax}(M,P) := |\text{Gap}(M,P)|.$$
:::

::: theorem
[]{#thm:tax-conservation label="thm:tax-conservation"} $|\text{Gap}(M, P)| + |R(P) \cap A(M)| = |R(P)|$. The total cannot be reduced, only redistributed between "handled natively" and "handled externally."
:::

::: proof
*Proof.* In the finite-coordinate model this is the exact set-cardinality identity $$|R\setminus A| + |R\cap A| = |R|,$$ formalized as HD5. ◻
:::

::: remark
The algebraic identity in Theorem [\[thm:tax-conservation\]](#thm:tax-conservation){reference-type="ref" reference="thm:tax-conservation"} is elementary. Its force comes from upstream hardness: reducing $|R(P)|$ by exact relevance minimization is worst-case intractable under the succinct encoding, so redistribution is often the only tractable lever available.
:::

::: theorem
[]{#thm:tax-grows label="thm:tax-grows"} For $n$ decision sites: $$\text{TotalExternalWork} = n \times \text{SimplicityTax}(M, P).$$
:::

::: proof
*Proof.* This is by definition of per-site externalization and is mechanized as HD26. ◻
:::

::: theorem
[]{#thm:amortization label="thm:amortization"} Let $H_{\text{central}}$ be the one-time cost of using a complete model. There exists $$n^* = \left\lfloor \frac{H_{\text{central}}}{\text{SimplicityTax}(M,P)} \right\rfloor$$ such that for $n > n^*$, the complete model has lower total cost.
:::

::: proof
*Proof.* For positive per-site tax, the threshold inequality $$n > \left\lfloor \frac{H_{\text{central}}}{\text{SimplicityTax}} \right\rfloor
\implies
H_{\text{central}} < n\cdot \text{SimplicityTax}$$ is mechanized as HD4. ◻
:::

::: corollary
[]{#cor:gap-externalization label="cor:gap-externalization"} \[D:Ccor:gap-externalization; R:H=cost-growth\] If $\text{Gap}(M,P)\neq\emptyset$, then external handling cost scales linearly with the number of decision sites.
:::

::: proof
*Proof.* The exact linear form is HD26. When the gap is nonempty (positive tax), monotone growth with $n$ is HD21. ◻
:::

::: corollary
[]{#cor:gap-minimization-hard label="cor:gap-minimization-hard"} \[D:Ccor:gap-minimization-hard; R:H=succinct-hard\] For mechanized Boolean-coordinate instances, "there exists a sufficient set of size at most $k$" is equivalent to "the relevant-coordinate set has cardinality at most $k$."
:::

::: proof
*Proof.* This is S2P2. ◻
:::

Informally: when exact minimization is hard, unresolved relevance shifts between centralized structure and repeated local work.

Appendix [\[app:lean\]](#app:lean){reference-type="ref" reference="app:lean"} provides theorem statements and module paths for the corresponding Lean formalization.


# Related Work {#sec:related}

## Formalized Complexity and Mechanized Proof Infrastructure

Formalizing complexity-theoretic arguments in proof assistants remains comparatively sparse. Lean and Mathlib provide the current host environment for this artifact [@Lean2015; @demoura2021lean4; @mathlib2020]. Closest mechanized precedents include verified computability/complexity developments in Coq and Isabelle [@forster2019verified; @kunze2019formal; @nipkow2002isabelle; @nipkow2014concrete; @haslbeck2016verified], and certifying toolchain work that treats proofs as portable machine-checkable evidence [@leroy2009compcert; @necula1997proof].

Recent LLM-evaluation work also highlights inference nondeterminism from numerical precision and hardware configuration, reinforcing the value of proof-level claims whose validity does not depend on stochastic inference trajectories [@yuan2025fp32].

## Computational Decision Theory

The complexity of decision-making has been studied extensively. Foundational treatments of computational complexity and strategic decision settings establish the baseline used here [@papadimitriou1994complexity; @arora2009computational]. Our work extends this to the meta-question of identifying relevant information. Decision-theoretic and information-selection framing in this paper also sits in the tradition of statistical sufficiency and signaling/information economics [@fisher1922mathematical; @spence1973job; @myerson1979incentive; @milgrom1986price; @kamenica2011bayesian].

#### Categorical framing of the quotient object.

The universal property of the optimizer quotient object is not, by itself, a probabilistic construction. In the category **Set**, quotienting states by equality of $\operatorname{Opt}$ is the standard coimage construction for the optimizer map $\operatorname{Opt}: S \to \mathcal{P}(A)$, canonically equivalent to its image/range [@maclane1998categories]. The novelty in this paper is therefore not that coimages exist, but that this particular coimage organizes coordinate sufficiency, witness lower bounds, and physical collapse constraints in one mechanized chain. \[D:Tthm:quotient-universal; R:DM\]

Large-sample Bayesian-network structure learning and causal identification are already known to be hard in adjacent formulations [@chickering2004large; @shpitser2006identification; @koller2009probabilistic]. Our object differs: we classify coordinate sufficiency for optimal-action invariance across static/stochastic/sequential regimes, with theorem-level placements and mechanized reductions.

#### Relation to prior hardness results.

Closest adjacent results are feature-selection/model-selection hardness results for predictive subset selection [@blum1997selection; @amaldi1998complexity]. Relative to those works, this paper changes two formal objects: (i) the reductions are machine-checked (TAUTOLOGY and $\exists\forall$-SAT mappings with explicit polynomial bounds), and (ii) the output is a regime-typed hardness/tractability map for decision relevance under explicit encoding assumptions. The target predicate here is decision relevance, not predictive compression.

#### Functional information and selection.

Prior work defines function-relative information objects in physical systems. Szostak defines functional information as the negative log of the fraction of configurations achieving a specified function [@szostak2003functional]. Wong et al. study function and selection in evolving systems and state an increasing-functional-information law under selection [@wong2023roles]. This paper addresses a complementary object: the computational decision predicate for when projected coordinates are sufficient to preserve optimal-action correspondence, with explicit complexity placements and mechanized proofs.

## Succinct Representations and Regime Separation

Representation-sensitive complexity is established in planning and decision-process theory: classical and compactly represented MDP/planning problems exhibit sharp complexity shifts under different input models [@papadimitriou1987mdp; @mundhenk2000mdp; @littman1998probplanning]. The explicit-vs-succinct separation in this paper is the same representation-sensitive phenomenon for the coordinate-sufficiency predicate.

The distinction in this paper is the object and scope of the classification: we classify *decision relevance* (sufficiency, anchor sufficiency, and minimum sufficient sets) for a fixed decision relation, with theorem-level regime typing and mechanized reduction chains.

## Oracle and Query-Access Lower Bounds

Query-access lower bounds are a standard source of computational hardness transfer [@dobzinski2012query; @buhrman2002complexity]. Our `[Q_fin]` and `[Q_bool]` results are in this tradition, but specialized to the same sufficiency predicate used throughout the paper: they establish access-obstruction lower bounds while keeping the underlying decision relation fixed across regimes.

Complementary companion work studies zero-error identification under constrained observation and proves rate-query lower bounds plus matroid structure of minimal distinguishing query sets [@simas2026identification]. Our object here is different: sufficiency of coordinates for optimal-action invariance in a decision problem, with regime-map placement (coNP/PP/PSPACE), static inner decomposition (coNP/$\Sigma_2^P$ decision layer plus access/structure layers), and regime-typed transfer theorems.

## Energy Landscapes and Chemical Reaction Networks

The geometric interpretation of decision complexity connects to the physics of energy landscapes. Wales established a widely used picture in which molecular configurations form high-dimensional landscapes with basins and transition saddles [@wales2003energy; @wales2005energy; @wales2018exploring]. Protein-folding theory uses the same landscape language via funnel structure toward native states [@onuchic1997protein]. In this paper's formal model, the corresponding core object is coordinate-projection invariance of the optimizer map: sufficiency asks whether fixing coordinates preserves $\operatorname{Opt}$, while insufficiency is witnessed by agreeing projections with different optimal sets (Definition [\[def:sufficient\]](#def:sufficient){reference-type="ref" reference="def:sufficient"}, Proposition [\[prop:insufficiency-counterexample\]](#prop:insufficiency-counterexample){reference-type="ref" reference="prop:insufficiency-counterexample"}). \[D:Ddef:sufficient;Pprop:insufficiency-counterexample; R:DM\]

Section [\[sec:physical-transport\]](#sec:physical-transport){reference-type="ref" reference="sec:physical-transport"} makes this transfer rule theorem-level: encoded physical claims are derived from core claims through an explicit assumption-transfer map, and an encoded physical counterexample refutes the corresponding core rule on that encoded slice. \[D:Pprop:physical-claim-transport;Ccor:physical-counterexample-core-failure; R:AR\]

Under that mapping, the explicit-vs-succinct split is the mechanized statement of access asymmetry: \[E\] admits polynomial verification on explicit state access, while \[S+ETH\] admits worst-case exponential lower bounds on succinct inputs (Theorem [\[thm:dichotomy\]](#thm:dichotomy){reference-type="ref" reference="thm:dichotomy"}); finite query access already exhibits indistinguishability barriers (Proposition [\[prop:query-regime-obstruction\]](#prop:query-regime-obstruction){reference-type="ref" reference="prop:query-regime-obstruction"}). The complexity separation is mechanized. \[D:Tthm:dichotomy;Pprop:query-regime-obstruction; R:E,S+ETH,Q_fin\]

The geometric core is mechanized directly: product-space slice cardinality, hypercube node-count identity, and node-vs-edge separation are proved in Lean at the decision abstraction level, with edge witnesses tied exactly to non-sufficiency.

The circuit-to-decision instantiation is also mechanized in a separate bridge layer: geometry and dynamics are represented as typed components of a circuit object; decision circuits add an explicit interpretation layer; molecule-level constructors instantiate both views.

Chemical reaction networks provide a concrete physical class where this encoding applies directly. Prior work establishes hardness for output optimization, reconfiguration, and reconstruction in CRN settings [@andersen2012maximizing; @alaniz2023complexity; @flamml2013complexity]. Those domain-specific CRN complexity results are external literature claims, not mechanized in this Lean artifact. The mechanized statement here is the transfer principle: once a CRN decision task is encoded as a C1--C4 decision problem, it inherits the same sufficiency hardness/tractability regime map (Theorems [\[thm:config-reduction\]](#thm:config-reduction){reference-type="ref" reference="thm:config-reduction"}, [\[thm:sufficiency-conp\]](#thm:sufficiency-conp){reference-type="ref" reference="thm:sufficiency-conp"}, [\[thm:tractable\]](#thm:tractable){reference-type="ref" reference="thm:tractable"}). \[D:Tthm:config-reduction, thm:sufficiency-conp, thm:tractable; R:E,S,TA\]

The physical bridge is not a single-domain add-on: Theorem [\[thm:physical-bridge-bundle\]](#thm:physical-bridge-bundle){reference-type="ref" reference="thm:physical-bridge-bundle"} composes law-objective semantics, reduction equivalence, and hardness lower-bound accounting in one mechanized theorem, so physical instantiation and complexity claims share one proof chain. \[D:Tthm:physical-bridge-bundle; R:AR\]

For matter-only dynamics, the law-induced objective specialization is mechanized for arbitrary universes (arbitrary state/action types and transition-feasibility relations): the utility schema is induced from transition laws, and under a strict allowed-vs-blocked gap with nonempty feasible set, the optimizer equals the feasible-action set; with logical determinism at the action layer, the optimizer is singleton. \[D:Pprop:law-instance-objective-bridge; R:AR\]

The interface-time semantics are also explicit and mechanized: timed states are $\mathbb{N}$-indexed, one decision event is exactly one tick, run length equals elapsed interface time, and this unit-time law is invariant across substrate tags under interface-consistency assumptions. \[D:Pprop:time-discrete, prop:decision-unit-time, prop:run-time-accounting, prop:substrate-unit-time; R:AR\]

Regime-side interface machinery is also formalized as typed access objects with an explicit simulation preorder and certificate-upgrade statements. The access-channel and five-way composition theorems are encoded as assumption-explicit composition laws. \[D:Pprop:physical-claim-transport;Tthm:physical-bridge-bundle; R:AR\]

## Thermodynamic Costs and Landauer's Principle

The thermodynamic lift in Section [\[sec:dichotomy\]](#sec:dichotomy){reference-type="ref" reference="sec:dichotomy"} converts bit-operation lower bounds into energy/carbon lower bounds via linear conversion constants. In this artifact, those conversion consequences are mechanized with explicit premises through Propositions [\[prop:thermo-lift\]](#prop:thermo-lift){reference-type="ref" reference="prop:thermo-lift"}, [\[prop:thermo-hardness-bundle\]](#prop:thermo-hardness-bundle){reference-type="ref" reference="prop:thermo-hardness-bundle"}, [\[prop:thermo-mandatory-cost\]](#prop:thermo-mandatory-cost){reference-type="ref" reference="prop:thermo-mandatory-cost"}, and [\[prop:thermo-conservation-additive\]](#prop:thermo-conservation-additive){reference-type="ref" reference="prop:thermo-conservation-additive"}. Landauer's principle and reversible-computation context [@landauer1961irreversibility; @bennett2003notes] define a minimum floor for irreversible bit erasure, not a generally tight expression for the actual energetic cost of constrained computation. The core stochastic-thermodynamics review literature [@vandenbroeck2015ensemble] supplies the ensemble- and trajectory-level entropy-production framework used to state such refinements. Wolpert et al. [@wolpert2024stochastic] identify the relevant constrained-computation regimes: periodic driving, modular organization, finite-time operation, and input mismatch. Manzano et al. [@manzano2024absolute] make stopping-time and absolute-irreversibility effects explicit and derive universal dissipation bounds in that setting. Yadav, Yousef, and Wolpert [@yadav2026minimal] connect mismatch cost to repeated circuit operation and structural resource lower bounds. Our formal model now represents those inputs at three levels. The coarse interface WC1--WC5 records the Landauer floor plus a generic nonnegative overhead. The refined decomposition WP1--WP9 separates that overhead into mismatch and residual terms. Within that refined layer, the mismatch branch is no longer only imported: WM1--WM4 derive nonnegativity, exact vanishing, strict positivity under explicit finite-distribution mismatch, and the conservative nat-valued lower-bound bridge from the existing KL machinery, while WM5--WM6 inject that derived branch into the Wolpert decomposition and prove strict separation above the Landauer floor. The finite discrete residual branch is now also theorem-level and exhaustive at the local edge level: WR1--WR3 give the bidirectional asymmetry witness, WR6 performs the reverse-edge case split, WR7--WR8 inject the resulting unified finite residual witness into the Wolpert decomposition, and WR9--WR10 repackage the same argument as a single process-level witness theorem for finite computational-state systems. Broader stopping-time / absolute-irreversibility residual regimes remain represented by the separate cited premise WP5. This keeps the causal structure of the cited work explicit while limiting the mechanization claim to what this artifact actually proves: the mismatch branch is derived, a finite discrete residual branch is derived by an exhaustive edge split and repackaged at process level, broader stopping-time residual regimes remain imported, and the composition layer is fully machine-checked. \[D:Pprop:thermo-lift, prop:thermo-hardness-bundle, prop:thermo-mandatory-cost, prop:thermo-conservation-additive; R:AR\]

#### Physics assumptions decompose into regime tags.

The paper does not create domain-specific tags for physics assumptions. Instead, every physics assumption decomposes into the same regime tags used throughout the paper. This uniformity states that physics assumptions are not epistemically privileged; they are declared assumptions, tracked with the same discipline as the paper's complexity-theoretic assumptions. The explicit physical-to-core mapping does not make the results less physical. It standardizes premise exposure and transfer obligations: accepted physics assumptions and this model are handled under the same declared-assumption framework, then checked with the same mechanized proof discipline where formalized.

::: {#tab:physics-regime-decomposition}
  **Physics Assumption**                      **Regime Decomposition**    **Justification**
  ------------------------------------------- --------------------------- ----------------------------------------
  Landauer bound ($E \ge kT\ln 2$)            \[AR\], \[H=cost-growth\]   Declared law; yields cost lower bounds
  Lorentz invariance                          \[AR\]                      Declared physical symmetry
  Discrete state space                        \[Q_fin\], \[AR\]           Finite-state core; declared assumption
  Thermodynamic constants ($\lambda, \rho$)   \[AR\], \[H=cost-growth\]   Declared conversion model
  Planck scale ($\ell_P$)                     \[AR\]                      Declared physical constant
  Physical-to-core encoding ($E$)             \[AR\]                      Declared encoding map

  : Physics-Assumption-to-Regime-Tag Decomposition
:::

The decomposition in Table [1](#tab:physics-regime-decomposition){reference-type="ref" reference="tab:physics-regime-decomposition"} makes explicit what the paper already practices: physics assumptions are declared premises in the \[AR\] (axiomatic regime) sense, and when they yield cost lower bounds, they inherit \[H=cost-growth\]. The finite-state query core \[Q_fin\] captures the discrete-state assumption underlying the computational model. No physics assumption receives a special domain tag; all decompose into the existing regime taxonomy. This is a standardization claim about proof form, not a change in physical content. \[D:Pprop:thermo-lift, prop:landauer-constraint, prop:lorentz-discrete, prop:discrete-state-time;Tthm:physical-bridge-bundle; R:AR,H=cost-growth,Q_fin\]

Coupling this to integrity yields a mechanized policy condition: when exact competence is hardness-blocked in the active regime, integrity requires abstention or explicitly weakened guarantees (Propositions [\[prop:integrity-resource-bound\]](#prop:integrity-resource-bound){reference-type="ref" reference="prop:integrity-resource-bound"}, [\[prop:attempted-competence-matrix\]](#prop:attempted-competence-matrix){reference-type="ref" reference="prop:attempted-competence-matrix"}). A stronger behavioral claim such as "abstention is globally utility-optimal for every objective" is *not* currently mechanized and remains objective-dependent. \[D:Pprop:integrity-resource-bound, prop:attempted-competence-matrix; R:AR\]

## Neukart--Vinokur Thermodynamic Duality

Neukart and Vinokur [@neukart2025thermodynamic] propose a thermodynamic duality $dU = T\,dS - p\,dV + \lambda\,dC$ where $C$ is a "complexity coordinate" associated with computational irreversibility in spin-glass systems. Corollary [\[cor:neukart-vinokur\]](#cor:neukart-vinokur){reference-type="ref" reference="cor:neukart-vinokur"} derives their duality as a specialization of our thermodynamic lift (Proposition [\[prop:thermo-lift\]](#prop:thermo-lift){reference-type="ref" reference="prop:thermo-lift"}) for $\lambda > 0$. \[D:Ccor:neukart-vinokur;Pprop:thermo-lift; R:AR\]

Their spin-glass phase transitions become regime-specific instances: our \[S+ETH\] hard families with $k^* = n$ witness their "complexity potential" via explicit TAUTOLOGY reduction (Theorem [\[thm:dichotomy\]](#thm:dichotomy){reference-type="ref" reference="thm:dichotomy"}). Physical transport via $E:\mathcal{P}\to\mathcal{D}$ extends to arbitrary substrates, not only quantum or spin systems (Theorem [\[thm:physical-bridge-bundle\]](#thm:physical-bridge-bundle){reference-type="ref" reference="thm:physical-bridge-bundle"}). \[D:Tthm:dichotomy, thm:physical-bridge-bundle; R:S+ETH,AR\]

Thus Neukart--Vinokur $\subseteq$ \[AR\] regime: $\lambda,\rho > 0$, explicit encoding $E:\mathcal{P}\to\mathcal{D}$ $\Rightarrow$ their thermodynamic duality holds by assumption transfer through our coNP-complete core (Proposition [\[prop:physical-claim-transport\]](#prop:physical-claim-transport){reference-type="ref" reference="prop:physical-claim-transport"}). \[D:Pprop:physical-claim-transport; R:AR\]

## Feature Selection

In machine learning, feature selection asks which input features are relevant for prediction. This is known to be NP-hard in general [@blum1997selection; @guyon2003introduction]. Our results place the decision-theoretic analog in a regime map: static checking/minimization at coNP-complete, stochastic sufficiency at PP-complete, and sequential sufficiency at PSPACE-complete. \[D:Tthm:sufficiency-conp, thm:minsuff-conp, th:stochastic-sufficiency-pp, th:sequential-sufficiency-pspace; R:S,RG\]

## Value of Information

The value of information (VOI) framework [@howard1966information; @raiffa1961applied] quantifies the maximum rational payment for information. Our work addresses a different question: not the *value* of information, but the *complexity* of identifying which information has value.

## Model Selection

Statistical model selection (AIC/BIC/cross-validation) provides practical heuristics for choosing among models [@akaike1974new; @schwarz1978estimating; @stone1974cross]. Our results formalize the regime-level reason heuristic selection appears: without added structural assumptions, exact optimal model selection inherits worst-case intractability, so heuristic methods implement explicit weakened-guarantee policies for unresolved structure. \[D:Tthm:sufficiency-conp, thm:dichotomy, thm:tractable; R:E,S,S+ETH\]

## Information Compression vs Certification Length

The paper now makes an explicit two-part information accounting object (Definition [\[def:raw-certified-bits\]](#def:raw-certified-bits){reference-type="ref" reference="def:raw-certified-bits"}): raw report bits and evidence-backed certification bits. This sharpens the information-theoretic reading of typed claims: compression of report payload is not equivalent to compression of certifiable truth conditions [@shannon1948mathematical; @cover2006elements; @slepian1973noiseless].

From a zero-error viewpoint, this distinction is structural rather than stylistic: compressed representations can remain short while exact decodability depends on confusability structure and admissible decoding conditions [@shannon1956zero; @korner1973graphs; @lovasz1979shannon; @csiszar2011information]. The same separation appears here between short reports and evidence-backed exact claims.

The rate-distortion and MDL lines make the same split in different language: there is a difference between achievable compression rate, computational procedure for obtaining/optimizing that rate, and certifiable model adequacy [@shannon1959coding; @blahut1972computation; @rissanen1978modeling; @grunwald2007minimum; @li2008introduction].

Formally, Theorem [\[thm:exact-certified-gap-iff-admissible\]](#thm:exact-certified-gap-iff-admissible){reference-type="ref" reference="thm:exact-certified-gap-iff-admissible"} and Corollary [\[cor:hardness-raw-only-exact\]](#cor:hardness-raw-only-exact){reference-type="ref" reference="cor:hardness-raw-only-exact"} characterize the exact-report boundary: admissible exact reports require a strict certified-bit gap above raw payload, while hardness-blocked exact reports collapse to raw-only accounting. This directly links representational succinctness to certification cost under the declared contract, rather than treating "short reports" as automatically high-information claims. \[D:Tthm:exact-certified-gap-iff-admissible;Pprop:raw-certified-bit-split;Ccor:hardness-raw-only-exact; R:AR\]

## Certifying Outputs and Proof-Carrying Claims

Our integrity layer matches the certifying-algorithms pattern: algorithms emit candidate outputs together with certificates that can be checked quickly, separating *producing* claims from *verifying* claims [@mcconnell2010certifying; @necula1997proof]. In this paper, Definition [\[def:solver-integrity\]](#def:solver-integrity){reference-type="ref" reference="def:solver-integrity"} is exactly that soundness discipline.

At the systems level, this is the same architecture as proof-carrying code: a producer ships evidence and a consumer runs a small checker before accepting the claim [@necula1997proof; @mcconnell2010certifying]. Our competence definition adds the regime-specific coverage/resource requirement that certifying soundness alone does not provide.

The feasibility qualifier in Definition [\[def:competence-regime\]](#def:competence-regime){reference-type="ref" reference="def:competence-regime"} yields the bounded-rationality constraint used in this paper: admissible policy is constrained by computational feasibility under the declared resource model (Remark [\[rem:modal-should\]](#rem:modal-should){reference-type="ref" reference="rem:modal-should"}) [@sep_bounded_rationality; @arora2009computational]. \[D:Pprop:integrity-competence-separation; R:AR\]

#### Cryptographic-verifiability perspective (scope).

The role of cryptography in this paper is structural verifiability, not secrecy: the relevant analogy is certificate-carrying claims with lightweight verification, not confidential encoding [@goldwasser1989knowledge; @necula1997proof]. Concretely, the typed-report discipline is modeled as a game-based reporting interface: a prover emits $(r,\pi)$ and a verifier accepts exactly when $\pi$ certifies the declared report type under the active contract. In that model, Evidence--Admissibility Equivalence gives completeness for admissible report types, while Typed Claim Admissibility plus Exact Claims Require Exact Evidence give soundness against exact overclaiming. Certified-confidence constraints then act as evidence-gated admissibility constraints on top of that verifier interface, so raw payload bits and certified bits represent distinct resources rather than stylistic notation variants. \[D:Pprop:evidence-admissibility-equivalence, prop:typed-claim-admissibility, prop:exact-requires-evidence, prop:certified-confidence-gate; R:AR\]

#### Three-axis integration.

In the cited literature, these pillars are treated separately: representation-sensitive hardness, query-access lower bounds, and certifying soundness disciplines. This paper composes all three for the same decision-relevance object in one regime-typed and machine-checked framework. \[D:Tthm:dichotomy;Pprop:query-regime-obstruction, prop:typed-claim-admissibility; R:S+ETH,Qf,AR\]


# Conclusion

Formally: this section composes proved regime placements, bridge theorems, and claim-typing constraints into final landscape consequences.

This paper gives a regime-first complexity map: static/stochastic/sequential placements with strict hierarchy (DC9, DC10, DC1, DC2), static inner-class decomposition by decision/access/structure layers (CCC6, CCC5, CCC1, CCC3, CCC7, CC61), theorem-level physical transport plus exact represented-adjacent bridge boundary (CT4, CC67, CC69, CC85), Bayesian uniqueness under DQ/thermodynamic admissibility axioms (FN14), physical-collapse no-go transfer chain (PH11, PH12, PH13, PH14, PH15), derived SAT-bridge map plus reduction-energy transfer and disjunctive assumption boundary (PH16, PH17, PH18, PH19, PH20, PH21, PH22, PH26, PH27), explicit finite-budget-necessity countermodels (PH31, PH32, PH33), regime-robust deterministic/probabilistic/sequential no-collapse specializations (PH28, PH29, PH30), observer-collapse anchor closure across stochastic/sequential checks (PA1, PA2, PA3, PA4, PA5, PA6, PA7, PA8, PA9), recurrent conversation-layer clamp/report bridge (CV4, CV7, CV17, CV18, CV13, CV16), explicit Neukart--Vinokur duality specialization from the thermodynamic lift (CC77, CC76), and formal centralized/distributed hardness thresholds through dominance and amortization theorems (HD1, HD4, HD3). \[D:Tthm:sufficiency-conp, thm:minsuff-conp, thm:anchor-sigma2p, th:stochastic-sufficiency-pp, th:sequential-sufficiency-pspace, thm:dichotomy, thm:tractable, thm:physical-bridge-bundle, thm:bridge-boundary-represented, thm:regime-coverage, thm:typed-completeness-static, thm:bayes-from-dq, thm:centralization-dominance, thm:tax-conservation, thm:amortization;Pprop:query-regime-obstruction, prop:thermo-lift, prop:static-stochastic-strict, prop:stochastic-sequential-strict;Ccor:neukart-vinokur; R:E,S,S+ETH,Qf,Qb,AR,RG,LC,H=cost-growth\]

Informally: this conclusion combines previously proved blocks under explicit tags.

## Methodology and Disclosure {#methodology-and-disclosure .unnumbered}

**Role of LLMs in this work.** This paper was developed through human-AI collaboration. The author provided the core intuitions: the connection between decision-relevance and computational complexity, the conjecture that SUFFICIENCY-CHECK is coNP-complete, and the insight that the $\Sigma_{2}^{P}$ structure collapses for MINIMUM-SUFFICIENT-SET. Large language models (Claude, GPT-4) served as implementation partners for proof drafting, Lean formalization, and LaTeX generation.

The Lean 4 proofs were iteratively refined: the author specified the target statements, the LLM proposed proof strategies, and the Lean compiler served as the arbiter of correctness. For complexity-theoretic reductions, the Lean formalization machine-verifies *correctness* (membership preservation); polynomial-time computability of the reduction functions is a standard claim stated as an explicit hypothesis, consistent with the state of formal verification where end-to-end Turing-machine polytime proofs for concrete reductions remain an open problem.

**What the author contributed:** The informal problem formulations (SUFFICIENCY-CHECK, MINIMUM-SUFFICIENT-SET, ANCHOR-SUFFICIENCY), the informal hardness conjectures, the informal tractability conditions, and the connection to over-modeling in engineering practice.

**What LLMs contributed:** LaTeX drafting, Lean tactic exploration, reduction construction assistance, and prose refinement.

Reduction correctness is machine-checked; polynomial-time computability is a standard hypothesis. This methodology is disclosed for academic transparency.

::: center

----------------------------------------------------------------------------------------------------
:::

## Acknowledgments {#acknowledgments .unnumbered}

This work depends on results whose authors made possible the insights that could then be machine-verified.

**Fundamental physics.** Albert Einstein established special relativity and the light-speed bound [@einstein1905electrodynamics]; Hermann Minkowski gave it the spacetime formulation [@minkowski1908space]. Max Planck established energy quantization and the constant $k_B$ [@planck1901distribution]. Rolf Landauer proved that erasure costs thermodynamic work [@landauer1961irreversibility]; Charles Bennett showed that reversible computation circumvents this bound [@bennett1982thermodynamics]. Bérut et al. experimentally verified Landauer's principle [@berut2012experimental]. The counting gap theorem (BA10) proves that bounded systems must obey finite acquisition budgets from pure mathematics; relativity then supplies the numerical propagation scale $c$ in our universe. The Wolpert-facing constrained-process interface now makes a sharper point explicit in the formal system: Landauer gives a universal floor, mismatch and residual dissipation are represented as separate theorem-level terms above that floor, the mismatch branch is derived from finite-distribution KL divergence as far as the present mathematics supports, and the finite discrete residual branch is derived by an exhaustive reverse-edge split: either asymmetric bidirectional edge flow yields positive two-point KL divergence, or vanishing reverse flow yields an irreversible one-way state transition with strictly positive Landauer-scaled cost. Any structural resource that is absorbed by mismatch further tightens the induced energy lower bound. Broader stopping-time / absolute-irreversibility residual regimes remain separate cited premises. The machine-checked part is the composition from that mixed theorem-plus-premise interface to the paper's decision-theoretic lower bounds.WC1WC2WC3WC4WC5WM1WM2WM3WM4WM5WM6WR1WR2WR3WR4WR5WR6WR7WR8WP1WP2WP5WP7WP8WP9

**Proof infrastructure.** The Lean 4 proof assistant [@demoura2021lean4] served as the arbiter of correctness. Leonardo de Moura, Sebastian Ullrich, and the Lean 4 team built a tool that transforms mathematical claims into mechanically verified theorems. The Mathlib library [@mathlib2020] provided the formal infrastructure: order theory, linear algebra, complexity-theoretic definitions, and the lemmas that made the formalization possible.

**Methodological note.** Reduction correctness is machine-checked; polynomial-time computability of reduction functions is a standard claim stated as an explicit hypothesis. The human author contributed the informal problem formulations, informal hardness conjectures, informal tractability conditions, and the connection to over-modeling in engineering practice. The machine contributors (Claude, GPT-4, Lean 4) served as proof assistants, tactic explorers, and drafting partners. The final arbiter was the Lean compiler: if it accepts the proof, the theorem holds.

::: center

----------------------------------------------------------------------------------------------------
:::

## Summary of Results {#summary-of-results .unnumbered}

This paper establishes a complete regime map for coordinate sufficiency and a static inner decomposition:

-   **Regime map (core classes).** Static, stochastic, and sequential sufficiency-check questions are placed at coNP, PP, and PSPACE respectively, with strict static $\subset$ stochastic $\subset$ sequential separation under declared assumptions (Theorems [\[thm:sufficiency-conp\]](#thm:sufficiency-conp){reference-type="ref" reference="thm:sufficiency-conp"}, [\[th:stochastic-sufficiency-pp\]](#th:stochastic-sufficiency-pp){reference-type="ref" reference="th:stochastic-sufficiency-pp"}, [\[th:sequential-sufficiency-pspace\]](#th:sequential-sufficiency-pspace){reference-type="ref" reference="th:sequential-sufficiency-pspace"}; Propositions [\[prop:static-stochastic-strict\]](#prop:static-stochastic-strict){reference-type="ref" reference="prop:static-stochastic-strict"}, [\[prop:stochastic-sequential-strict\]](#prop:stochastic-sequential-strict){reference-type="ref" reference="prop:stochastic-sequential-strict"}).

-   **Observer-collapse anchor closure.** Under declared observer-collapse hypotheses, stochastic/sequential anchor checks are forced by effective-state singleton structure and seeded support conditions (PA1, PA4, PA5, PA6, PA7, PA9).

-   **Static decision layer decomposition.** Within the static class, SUFFICIENCY-CHECK and MINIMUM-SUFFICIENT-SET are coNP-complete, while ANCHOR-SUFFICIENCY is $\Sigma_{2}^{P}$-complete (Theorems [\[thm:sufficiency-conp\]](#thm:sufficiency-conp){reference-type="ref" reference="thm:sufficiency-conp"}, [\[thm:minsuff-conp\]](#thm:minsuff-conp){reference-type="ref" reference="thm:minsuff-conp"}, [\[thm:anchor-sigma2p\]](#thm:anchor-sigma2p){reference-type="ref" reference="thm:anchor-sigma2p"}).

-   **Static access layer decomposition.** An encoding-regime separation contrasts explicit-state polynomial-time (polynomial in $|S|$) with succinct ETH worst-case lower bounds, and query-access obstruction cores refine this with value-entry/state-batch/oracle-lattice transfers (Theorem [\[thm:dichotomy\]](#thm:dichotomy){reference-type="ref" reference="thm:dichotomy"}; Propositions [\[prop:query-regime-obstruction\]](#prop:query-regime-obstruction){reference-type="ref" reference="prop:query-regime-obstruction"}--[\[prop:oracle-lattice-strict\]](#prop:oracle-lattice-strict){reference-type="ref" reference="prop:oracle-lattice-strict"}).

-   **Static structure layer decomposition.** Six structural subclasses provide certified polynomial recovery conditions inside the static class (Theorem [\[thm:tractable\]](#thm:tractable){reference-type="ref" reference="thm:tractable"}).

-   Discrete-time interface semantics are formalized: time is $\mathbb{N}$-valued, decision events are exactly one-tick transitions, run length equals elapsed time, and substrate-consistent realizations preserve the same one-unit event law (Propositions [\[prop:time-discrete\]](#prop:time-discrete){reference-type="ref" reference="prop:time-discrete"}--[\[prop:substrate-unit-time\]](#prop:substrate-unit-time){reference-type="ref" reference="prop:substrate-unit-time"})

-   Physical-system transport is formalized: encoded physical claims are derived from core rules through explicit assumption transfer, physical counterexamples map to encoded core failures, and the bundled bridge theorem composes law semantics, reduction equivalence, and required-work lower bounds (Proposition [\[prop:physical-claim-transport\]](#prop:physical-claim-transport){reference-type="ref" reference="prop:physical-claim-transport"}, Corollary [\[cor:physical-counterexample-core-failure\]](#cor:physical-counterexample-core-failure){reference-type="ref" reference="cor:physical-counterexample-core-failure"}, Theorem [\[thm:physical-bridge-bundle\]](#thm:physical-bridge-bundle){reference-type="ref" reference="thm:physical-bridge-bundle"})

-   Exact represented-adjacent bridge boundary: transfer is licensed iff the class is one-step deterministic; represented horizon, stochastic-criterion, and transition-coupled classes have explicit bridge-failure witnesses (Theorem [\[thm:bridge-boundary-represented\]](#thm:bridge-boundary-represented){reference-type="ref" reference="thm:bridge-boundary-represented"})

-   Exact-report admissibility is characterized by a certified-bit gap criterion; when exact reporting is blocked, exact accounting collapses to raw-only (Theorem [\[thm:exact-certified-gap-iff-admissible\]](#thm:exact-certified-gap-iff-admissible){reference-type="ref" reference="thm:exact-certified-gap-iff-admissible"}, Corollary [\[cor:hardness-raw-only-exact\]](#cor:hardness-raw-only-exact){reference-type="ref" reference="cor:hardness-raw-only-exact"})

-   The thermodynamic lift yields the Neukart--Vinokur duality specialization $dU \ge \lambda\,dC$ for $\lambda>0$ (Corollary [\[cor:neukart-vinokur\]](#cor:neukart-vinokur){reference-type="ref" reference="cor:neukart-vinokur"}, Proposition [\[prop:thermo-lift\]](#prop:thermo-lift){reference-type="ref" reference="prop:thermo-lift"})

\[D:Tthm:sufficiency-conp, thm:minsuff-conp, thm:anchor-sigma2p, th:stochastic-sufficiency-pp, th:sequential-sufficiency-pspace, thm:dichotomy, thm:tractable, thm:physical-bridge-bundle, thm:bridge-boundary-represented, thm:exact-certified-gap-iff-admissible;Pprop:static-stochastic-strict, prop:stochastic-sequential-strict, prop:query-regime-obstruction, prop:query-value-entry-lb, prop:query-state-batch-lb, prop:query-subproblem-transfer, prop:oracle-lattice-transfer, prop:oracle-lattice-strict, prop:time-discrete, prop:decision-unit-time, prop:run-time-accounting, prop:substrate-unit-time, prop:physical-claim-transport, prop:thermo-lift, prop:raw-certified-bit-split;Ccor:physical-counterexample-core-failure, cor:hardness-raw-only-exact, cor:neukart-vinokur; R:E,S,S+ETH,Qf,Qb,AR,RG,H=cost-growth\]

These results place decision-relevant coordinate identification into a regime-indexed class ladder, rather than a single-class statement: static (coNP/$\Sigma_{2}^{P}$ subproblems), stochastic (PP), and sequential (PSPACE), with explicit strict separations and bridge gates.

Beyond classification, the paper contributes a formal claim-typing framework (Section [\[sec:interpretive-foundations\]](#sec:interpretive-foundations){reference-type="ref" reference="sec:interpretive-foundations"}): structural complexity is a property of the fixed decision question, while representational hardness is regime-conditional access cost. This is why encoding-regime changes can move practical hardness without changing the underlying semantics. \[D:Pprop:sufficiency-char;Tthm:typed-completeness-static; R:AR\]

The reduction constructions and key equivalence theorems are machine-checked in Lean 4 (see Appendix [\[app:lean\]](#app:lean){reference-type="ref" reference="app:lean"} for proof listings). The formalization verifies TAUTOLOGY and $\exists\forall$-SAT static reductions, theorem-level stochastic/sequential class placements, and strict regime-separation claims; complexity classifications (coNP, $\Sigma_{2}^{P}$, PP, PSPACE) follow by composition with standard complexity-theoretic results under the declared assumptions. The strengthened gadget showing that non-tautologies yield instances with *all coordinates relevant* is also formalized. \[D:Tthm:sufficiency-conp, thm:minsuff-conp, thm:anchor-sigma2p, th:stochastic-sufficiency-pp, th:sequential-sufficiency-pspace, thm:dichotomy;Pprop:static-stochastic-strict, prop:stochastic-sequential-strict; R:S,S+ETH,RG\]

## Complexity Characterization {#complexity-characterization .unnumbered}

The results provide precise complexity characterizations within the formal model:

1.  **Exact bounds.** SUFFICIENCY-CHECK is coNP-complete, both coNP-hard and in coNP.

2.  **Constructive reductions.** The reductions from TAUTOLOGY and $\exists\forall$-SAT are explicit and machine-checked.

3.  **Encoding-regime separation.** Under \[E\], SUFFICIENCY-CHECK is polynomial in $|S|$. Under \[S+ETH\], there exist succinct worst-case instances (with $k^*=n$) requiring $2^{\Omega(n)}$ time. Under \[Q_fin\], the Opt-oracle core has $\Omega(|S|)$ worst-case query complexity (Boolean instantiation $\Omega(2^n)$), and under \[Q_bool\] the value-entry/state-batch refinements preserve the obstruction with weighted-cost transfer closures.

\[D:Tthm:sufficiency-conp, thm:dichotomy;Pprop:query-regime-obstruction, prop:query-value-entry-lb, prop:query-state-batch-lb, prop:query-subproblem-transfer; R:E,S,S+ETH,Qf,Qb\]

## The Complexity Redistribution Corollary {#the-complexity-redistribution-corollary .unnumbered}

Section [\[sec:simplicity-tax\]](#sec:simplicity-tax){reference-type="ref" reference="sec:simplicity-tax"} develops a quantitative consequence: when a problem requires $k$ dimensions and a model handles only $j < k$ natively, the remaining $k - j$ dimensions must be handled externally at each decision site. For $n$ sites, total external work is $(k-j) \times n$. \[D:Tthm:tax-grows; R:LC\]

The set identity is elementary; its operational content comes from composition with the hardness results on exact relevance minimization. This redistribution corollary is formalized in Lean 4 (`HardnessDistribution.lean`), proving:

-   Redistribution identity: complexity burden cannot be eliminated by omission, only moved between native handling and external handling

-   Dominance: complete models have lower total work than incomplete models

-   Amortization: there exists a threshold $n^*$ beyond which higher-dimensional models have lower total cost

\[D:Tthm:tax-conservation, thm:centralization-dominance, thm:amortization; R:LC\]

## Open Questions {#open-questions .unnumbered}

The landscape above is complete for the static sufficiency class under C1--C4; the items below are adjacent-class extensions or secondary refinements. Several questions remain for future work:

-   **Fixed-parameter tractability (primary):** Is SUFFICIENCY-CHECK FPT when parameterized by the minimal sufficient-set size $k^*$, or is it W\[2\]-hard under this parameterization?

-   **Adjacent-objective bridge frontier (beyond proven regime cores):** Characterize the exact frontier where adjacent sequential objectives reduce to the static class via Proposition [\[prop:one-step-bridge\]](#prop:one-step-bridge){reference-type="ref" reference="prop:one-step-bridge"}, and where genuinely new complexity objects (e.g., horizon/sample/regret complexity) must replace the present static coNP/$\Sigma_2^P$ layer.

-   **Average-case complexity:** What is the complexity under natural distributions on decision problems?

-   **Typed-confidence calibration:** For signaled reports $(r,g,p_{\mathrm{self}},p_{\mathrm{cert}})$, characterize proper scoring and calibration objectives that preserve signal consistency and typed admissibility.

-   **Learning cost formalization:** Can central cost $H_{\text{central}}$ be formalized as the rank of a concept matroid, making the amortization threshold precisely computable?

## Practical Corollaries {#practical-corollaries .unnumbered}

The practical corollaries are regime-indexed and theorem-indexed:

-   **\[E\] and structured regimes:** polynomial-time exact procedures exist (Theorem [\[thm:tractable\]](#thm:tractable){reference-type="ref" reference="thm:tractable"}).

-   **\[Q_fin\]/\[Q_bool\] query-access lower bounds:** worst-case Opt-oracle complexity is $\Omega(|S|)$ in the finite-state core (Boolean instantiation $\Omega(2^n)$), and value-entry/state-batch interfaces satisfy the same obstruction in the mechanized Boolean refinement (Propositions [\[prop:query-regime-obstruction\]](#prop:query-regime-obstruction){reference-type="ref" reference="prop:query-regime-obstruction"}--[\[prop:oracle-lattice-strict\]](#prop:oracle-lattice-strict){reference-type="ref" reference="prop:oracle-lattice-strict"}), with randomized weighted robustness and oracle-lattice closures.

-   **\[S+ETH\] hard families:** exact minimization inherits exponential worst-case cost (Theorem [\[thm:dichotomy\]](#thm:dichotomy){reference-type="ref" reference="thm:dichotomy"} together with Theorem [\[thm:sufficiency-conp\]](#thm:sufficiency-conp){reference-type="ref" reference="thm:sufficiency-conp"}).

-   **\[S_bool\] mechanized criterion:** minimization reduces to relevance-cardinality constraints (Corollary [\[cor:practice-diagnostic\]](#cor:practice-diagnostic){reference-type="ref" reference="cor:practice-diagnostic"}).

-   **Redistribution consequences:** omitted native coverage externalizes work with explicit growth/amortization laws (Theorems [\[thm:tax-conservation\]](#thm:tax-conservation){reference-type="ref" reference="thm:tax-conservation"}--[\[thm:amortization\]](#thm:amortization){reference-type="ref" reference="thm:amortization"}).

-   **Physical-system instantiation:** once a physical class is encoded into the declared decision interface, the same regime-typed hardness/tractability map and integrity constraints apply to that class (Section [\[sec:physical-transport\]](#sec:physical-transport){reference-type="ref" reference="sec:physical-transport"}). \[D:Pprop:physical-claim-transport;Ccor:physical-counterexample-core-failure;Tthm:physical-bridge-bundle; R:AR\]

#### Physical Scope Guard.

All transfer claims in this paper are typed through explicit physical encodings, declared observables, and declared assumption maps. Transfer is admitted only on declared physical instances satisfying the physical scope gate. \[D:Ddef:physical-core-encoding, def:physical-scope-gate;Pprop:typed-physical-transport-requirement, prop:physical-claim-transport; R:AR\]

Hence the design choice is typed: enforce a tractable regime, or adopt weakened guarantees with explicit verification boundaries. \[D:Tthm:tractable, thm:dichotomy; R:RG\] Equivalently, under the attempted-competence matrix, over-specification is rational only in attempted-competence-failure cells; once exact competence is available in the active regime (or no attempted exact competence was made), persistent over-specification is irrational for the same verified-cost objective. \[D:Pprop:attempted-competence-matrix; R:AR\] By Proposition [\[prop:attempted-competence-matrix\]](#prop:attempted-competence-matrix){reference-type="ref" reference="prop:attempted-competence-matrix"}, the integrity-preserving matrix contains 3 irrational cells and 1 rational cell. \[D:Pprop:attempted-competence-matrix; R:AR\]

## Measurement-Constrained Reporting for Physical Solvers {#measurement-constrained-reporting-for-physical-solvers .unnumbered}

Within this paper's formalism, reporting correctness is a claim-discipline property: outputs are asserted only when certifiable, and otherwise emitted as abstain reports (optionally carrying tentative guesses and self-reflected confidence under zero certified confidence). Formally, this is the integrity/competence split of Section [\[sec:interpretive-foundations\]](#sec:interpretive-foundations){reference-type="ref" reference="sec:interpretive-foundations"}, i.e., an evidence-typing statement for reported claims under declared regimes. \[D:Ddef:solver-integrity, def:competence-regime;Pprop:integrity-competence-separation; R:TR\]

Formally: the abstention doctrine is a typed-report consequence of integrity/competence and evidence-gating predicates.

#### Truth-Objective Abstention Requirement.

If a physical solver system declares a truth-preserving objective for claims about relation $\mathcal{R}$ in regime $\Gamma$, uncertified outputs are admissible only as $\mathsf{ABSTAIN}/\mathsf{UNKNOWN}$ rather than asserted answers. Under the signal discipline, this includes abstain-with-guess/self-confidence reports while keeping certification evidence-gated. \[D:Ddef:signaled-typed-report, def:signal-consistency;Pprop:certified-confidence-gate, prop:abstain-guess-self-signal;Ccor:exact-no-competence-zero-certified; R:AR\] Equivalently, reporting protocols that suppress abstention in uncertified regions violate the typed claim discipline under the declared task model. \[D:Ddef:solver-integrity;Pprop:integrity-competence-separation, prop:integrity-resource-bound, prop:attempted-competence-matrix; R:AR\]

This yields an output-control doctrine for engineered solver systems under the task/regime contract:

1.  Treat $\mathsf{ABSTAIN}/\mathsf{UNKNOWN}$ as first-class admissible outputs, not errors. \[D:Ddef:solver-integrity; R:TR\]

2.  For signaled reports, maintain two channels: self-reflected confidence may be emitted, but certified confidence is positive only with evidence. \[D:Ddef:signal-consistency;Pprop:certified-confidence-gate, prop:self-confidence-not-certification; R:AR\]

3.  Treat runtime disclosure as scalar-witnessed: always emit exact step-count, and emit completion percentage only when a declared bound exists. \[D:Pprop:steps-run-scalar, prop:no-fraction-without-bound, prop:fraction-defined-under-bound; R:AR\]

4.  Optimize competent coverage subject to integrity constraints, rather than optimizing answer-rate alone. \[D:Ddef:competence-regime;Pprop:integrity-competence-separation; R:AR\]

5.  In hardness-blocked regimes, abstention or explicit guarantee-weakening is structurally required unless regime assumptions are changed. \[D:Tthm:dichotomy;Pprop:integrity-resource-bound, prop:attempted-competence-matrix; R:S,S+ETH\]

Operationally, a reporting schema consistent with this framework tracks at least five quantities: integrity violations, competent non-abstaining coverage on the scoped family, abstention quality under regime shift, the calibration gap between self-reflected and certified confidence channels, and scalar runtime witness distributions (with bound-qualified fraction semantics when bounds are given). This is consistent with bounded-rationality feasibility discipline under bounded resource models [@sep_bounded_rationality; @arora2009computational].

Informally: if exact support is not certified, do not make exact claims; when competence is unavailable, abstain and keep confidence evidence-linked.


# Lean 4 Proof Listings {#app:lean}

The complete Lean 4 formalization is available in the companion artifact (Zenodo DOI listed on the title page). The mechanization consists of 33311 lines across 123 files, with 1378 theorem/lemma statements.

**Handle IDs.** Inline theorem metadata now cites compact IDs (for example, `HD6`, `CC12`, `IC4`) instead of full theorem constants. The full ID-to-handle mapping is listed in Section [1.1](#sec:lean-handle-id-map){reference-type="ref" reference="sec:lean-handle-id-map"}.

## Lean Handle ID Map {#sec:lean-handle-id-map}

## What Is Machine-Checked

The Lean formalization establishes:

1.  **Correctness of the TAUTOLOGY reduction:** The theorem `tautology_iff_sufficient` proves that the mapping from Boolean formulas to decision problems preserves the decision structure (accept iff the formula is a tautology).

2.  **Decision problem definitions:** Formal definitions of sufficiency, optimality, and the decision quotient.

3.  **Economic theorems:** Simplicity Tax redistribution identities and hardness distribution results.

4.  **Query-access lower-bound core:** Formalized Boolean query-model indistinguishability theorem for the full problem via the $I=\emptyset$ subproblem (`emptySufficiency_query_indistinguishable_pair`), plus obstruction-scale identities (`queryComplexityLowerBound`, `exponential_query_complexity`).

**Complexity classifications** (coNP-completeness, $\Sigma_2^P$-completeness) follow by conditional composition with standard results (e.g., TAUTOLOGY coNP-completeness and $\exists\forall$-SAT $\Sigma_2^P$-completeness), represented explicitly as hypotheses in the conditional transfer theorems listed below. The Lean proofs verify the reduction constructions and the transfer closures under those hypotheses. The assumptions themselves are unpacked by an explicit ledger projection theorem (CC5) so dependency tracking is machine-auditable.

## Polynomial-Time Reduction Definition

We use Mathlib's Turing machine framework to define polynomial-time computability:

    /-- Polynomial-time computable function using Turing machines -/
    def PolyTime {α β : Type} (ea : FinEncoding α) (eb : FinEncoding β) 
        (f : α → β) : Prop :=
      Nonempty (Turing.TM2ComputableInPolyTime ea eb f)

    /-- Polynomial-time many-one (Karp) reduction -/
    def ManyOneReducesPoly {α β : Type} (ea : FinEncoding α) (eb : FinEncoding β)
        (A : Set α) (B : Set β) : Prop :=
      ∃ f : α → β, PolyTime ea eb f ∧ ∀ x, x ∈ A ↔ f x ∈ B

This uses the standard definition: a reduction is polynomial-time computable via Turing machines and preserves membership.

**Note on verification status.** The Lean formalization machine-verifies *correctness* (the membership-preservation property, i.e., the second conjunct of `ManyOneReducesPoly`). Proving that a concrete function satisfies `PolyTime`---i.e., constructing an actual Turing machine witness with a polynomial time bound---remains an open problem in formal verification. No existing proof assistant project has achieved end-to-end TM-based polytime proofs for concrete reductions. In our formalization, polytime computability is stated as an explicit hypothesis in `valid_karp_reduction`, making the gap visible rather than hiding it.

## The Main Reduction Theorem

::: theorem
The reduction from TAUTOLOGY to SUFFICIENCY-CHECK is correct:
:::

    theorem tautology_iff_sufficient (φ : Formula n) :
        φ.isTautology ↔ (reductionProblem φ).isSufficient Finset.empty

This theorem is proven by showing both directions:

-   If $\varphi$ is a tautology, then the empty coordinate set is sufficient

-   If the empty coordinate set is sufficient, then $\varphi$ is a tautology

The proof verifies that the utility construction in `reductionProblem` creates the appropriate decision structure where:

-   At reference states, `accept` is optimal with utility 1

-   At assignment states, `accept` is optimal iff $\varphi(a) = \text{true}$

## Economic Results

The hardness distribution theorems (Section [\[sec:simplicity-tax\]](#sec:simplicity-tax){reference-type="ref" reference="sec:simplicity-tax"}) are fully formalized:

    theorem simplicityTax_conservation (P : SpecificationProblem)
        (S : SolutionArchitecture P) :
        S.centralDOF + simplicityTax P S ≥ P.intrinsicDOF

    theorem simplicityTax_grows (P : SpecificationProblem)
        (S : SolutionArchitecture P) (n₁ n₂ : ℕ)
        (hn : n₁ < n₂) (htax : simplicityTax P S > 0) :
        totalDOF S n₁ < totalDOF S n₂

    theorem native_dominates_manual (P : SpecificationProblem) (n : Nat)
        (hn : n > P.intrinsicDOF) :
        totalDOF (nativeTypeSystem P) n < totalDOF (manualApproach P) n

    theorem totalDOF_eventually_constant_iff_zero_distributed
        (S : SolutionArchitecture P) :
        IsEventuallyConstant (fun n => totalDOF S n) ↔ S.distributedDOF = 0

    theorem no_positive_slope_linear_represents_saturating
        (c d K : ℕ) (hd : d > 0) :
        ¬ (∀ n, c + n * d = generalizedTotalDOF c (saturatingSiteCost K) n)

**Identifier note.** Lean identifiers retain internal naming (`intrinsicDOF`, `simplicityTax_conservation`); in paper terminology these correspond to *baseline hardness* and the *redistribution lower-bound identity*, respectively.

## Complete Claim Coverage Matrix

## Proof Hardness Index (Auto)

## Claims Not Fully Mechanized

**Status:** all theorem/proposition/corollary handles in this paper now have Lean backing. Entries marked **Full (conditional)** are explicitly mechanized transfer theorems that depend on standard external complexity facts (e.g., source-class completeness or ETH assumptions), with those dependencies represented as hypotheses in Lean.

## Module Structure

-   `Basic.lean` -- Core definitions (DecisionProblem, sufficiency, optimality)

-   `Sufficiency.lean` -- Sufficiency checking algorithms and properties

-   `Reduction.lean` -- TAUTOLOGY reduction construction and correctness

-   `Hardness/ConfigReduction.lean` -- Sentinel-action configuration reduction theorem

-   `Complexity.lean` -- Polynomial-time reduction definitions using mathlib

-   `HardnessDistribution.lean` -- Simplicity Tax redistribution and amortization theorems

-   `IntegrityCompetence.lean` -- Solver integrity vs regime competence separation

-   `ClaimClosure.lean` -- Mechanized closure of paper-level bridge and diagnostic claims

-   `Tractability/` -- Bounded actions, separable utilities, tree structure

## Verification

The proofs compile with Lean 4 and contain no `sorry` placeholders. Run `lake build` in the proof directory to verify.


+----------------------------------------------------------------------------------------------------------------------------------------+
| Lean handle entry                                                                                                                      |
+:=======================================================================================================================================+
| Lean handle entry (continued)                                                                                                          |
+----------------------------------------------------------------------------------------------------------------------------------------+
| ::: {#lh:AB1}                                                                                                                          |
| `AB1`                                                                                                                                  |
| :::                                                                                                                                    |
|                                                                                                                                        |
| `DecisionQuotient.DecisionProblem.not_preservesOpt_iff_erasesDecisionRelevantDistinction`                                              |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`AB2`]{#lh:AB2}`DecisionQuotient.DecisionProblem.surjective_abstraction_factors_or_erases`                                            |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`AB3`]{#lh:AB3}`DecisionQuotient.DecisionProblem.collapseBeyondQuotient_physically_impossible`                                        |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`AB4`]{#lh:AB4}`DecisionQuotient.DecisionProblem.surjective_abstraction_with_feasible_collapse_map_factors`                           |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`AC1`]{#lh:AC1}`DecisionQuotient.ClaimClosure.AtomicCircuitExports.AC1`                                                               |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`AC3`]{#lh:AC3}`DecisionQuotient.ClaimClosure.AtomicCircuitExports.AC3`                                                               |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`AC4`]{#lh:AC4}`DecisionQuotient.ClaimClosure.AtomicCircuitExports.AC4`                                                               |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`AC5`]{#lh:AC5}`DecisionQuotient.ClaimClosure.AtomicCircuitExports.AC5`                                                               |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`AC6`]{#lh:AC6}`DecisionQuotient.ClaimClosure.AtomicCircuitExports.AC6`                                                               |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`AC8`]{#lh:AC8}`DecisionQuotient.ClaimClosure.AtomicCircuitExports.AC8`                                                               |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`AC9`]{#lh:AC9}`DecisionQuotient.ClaimClosure.AtomicCircuitExports.AC9`                                                               |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`AC11`]{#lh:AC11}`DecisionQuotient.ClaimClosure.AtomicCircuitExports.AC11`                                                            |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`AN1`]{#lh:AN1}`DecisionQuotient.Physics.AssumptionNecessity.no_assumption_free_proof_of_refutable_claim`                             |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`AN2`]{#lh:AN2}`DecisionQuotient.Physics.AssumptionNecessity.countermodel_violates_some_assumption`                                   |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`AN3`]{#lh:AN3}`DecisionQuotient.Physics.AssumptionNecessity.physical_claim_requires_physical_assumption`                             |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`AN4`]{#lh:AN4}`DecisionQuotient.Physics.AssumptionNecessity.physical_claim_requires_empirically_justified_physical_assumption`       |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`AN5`]{#lh:AN5}`DecisionQuotient.Physics.AssumptionNecessity.strong_physical_no_go_meta`                                              |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`AQ1`]{#lh:AQ1}`DecisionQuotient.ClaimClosure.AQ1`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`AQ2`]{#lh:AQ2}`DecisionQuotient.ClaimClosure.AQ2`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`AQ3`]{#lh:AQ3}`DecisionQuotient.ClaimClosure.AQ3`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`AQ4`]{#lh:AQ4}`DecisionQuotient.ClaimClosure.AQ4`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`AQ5`]{#lh:AQ5}`DecisionQuotient.ClaimClosure.AQ5`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`AQ6`]{#lh:AQ6}`DecisionQuotient.ClaimClosure.AQ6`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`AQ7`]{#lh:AQ7}`DecisionQuotient.ClaimClosure.AQ7`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`AQ8`]{#lh:AQ8}`DecisionQuotient.ClaimClosure.AQ8`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`ARG1`]{#lh:ARG1}`PhysicalComplexity.AccessRegime.PhysicalDevice`                                                                     |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`ARG2`]{#lh:ARG2}`PhysicalComplexity.AccessRegime.AccessRegime`                                                                       |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`ARG3`]{#lh:ARG3}`PhysicalComplexity.AccessRegime.RegimeEval`                                                                         |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`ARG4`]{#lh:ARG4}`PhysicalComplexity.AccessRegime.RegimeSample`                                                                       |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`ARG5`]{#lh:ARG5}`PhysicalComplexity.AccessRegime.RegimeProof`                                                                        |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`ARG6`]{#lh:ARG6}`PhysicalComplexity.AccessRegime.RegimeWithCertificate`                                                              |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`ARG7`]{#lh:ARG7}`PhysicalComplexity.AccessRegime.RegimeEvalOn`                                                                       |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`ARG8`]{#lh:ARG8}`PhysicalComplexity.AccessRegime.RegimeSampleOn`                                                                     |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`ARG9`]{#lh:ARG9}`PhysicalComplexity.AccessRegime.RegimeProofOn`                                                                      |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`ARG10`]{#lh:ARG10}`PhysicalComplexity.AccessRegime.RegimeWithCertificateOn`                                                          |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`ARG11`]{#lh:ARG11}`PhysicalComplexity.AccessRegime.HardUnderEval`                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`ARG12`]{#lh:ARG12}`PhysicalComplexity.AccessRegime.AuditableWithCertificate`                                                         |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`ARG13`]{#lh:ARG13}`PhysicalComplexity.AccessRegime.certificate_upgrades_regime`                                                      |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`ARG14`]{#lh:ARG14}`PhysicalComplexity.AccessRegime.certificate_upgrades_regime_on`                                                   |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`ARG15`]{#lh:ARG15}`PhysicalComplexity.AccessRegime.physical_succinct_certification_hard`                                             |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`ARG16`]{#lh:ARG16}`PhysicalComplexity.AccessRegime.certificate_amortizes_hardness`                                                   |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`ARG17`]{#lh:ARG17}`PhysicalComplexity.AccessRegime.regime_upgrade_with_certificate`                                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`ARG18`]{#lh:ARG18}`PhysicalComplexity.AccessRegime.regime_upgrade_with_certificate_on`                                               |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`ARG19`]{#lh:ARG19}`PhysicalComplexity.AccessRegime.AccessChannelLaw`                                                                 |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`ARG20`]{#lh:ARG20}`PhysicalComplexity.AccessRegime.FiveWayMeet`                                                                      |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`BA1`]{#lh:BA1}`DecisionQuotient.Physics.BoundedAcquisition.BoundedRegion`                                                            |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`BA2`]{#lh:BA2}`DecisionQuotient.Physics.BoundedAcquisition.acquisition_rate_bound`                                                   |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`BA3`]{#lh:BA3}`DecisionQuotient.Physics.BoundedAcquisition.acquisitions_are_transitions`                                             |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`BA4`]{#lh:BA4}`DecisionQuotient.Physics.BoundedAcquisition.one_bit_per_transition`                                                   |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`BA5`]{#lh:BA5}`DecisionQuotient.Physics.BoundedAcquisition.resolution_reads_sufficient`                                              |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`BA6`]{#lh:BA6}`DecisionQuotient.Physics.BoundedAcquisition.srank_le_resolution_bits`                                                 |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`BA7`]{#lh:BA7}`DecisionQuotient.Physics.BoundedAcquisition.energy_ge_srank_cost`                                                     |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`BA8`]{#lh:BA8}`DecisionQuotient.Physics.BoundedAcquisition.srank_one_energy_minimum`                                                 |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`BA9`]{#lh:BA9}`DecisionQuotient.Physics.BoundedAcquisition.physical_grounding_bundle`                                                |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`BA10`]{#lh:BA10}`DecisionQuotient.Physics.BoundedAcquisition.counting_gap_theorem`                                                   |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`BB1`]{#lh:BB1}`DecisionQuotient.BayesianDQ`                                                                                          |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`BB2`]{#lh:BB2}`DecisionQuotient.BayesianDQ.certaintyGain`                                                                            |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`BB3`]{#lh:BB3}`DecisionQuotient.dq_is_bayesian_certainty_fraction`                                                                   |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`BB4`]{#lh:BB4}`DecisionQuotient.bayesian_dq_matches_physics_dq`                                                                      |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`BB5`]{#lh:BB5}`DecisionQuotient.dq_derived_from_bayes`                                                                               |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`BC1`]{#lh:BC1}`DecisionQuotient.Foundations.counting_nonneg`                                                                         |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`BC2`]{#lh:BC2}`DecisionQuotient.Foundations.counting_total`                                                                          |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`BC3`]{#lh:BC3}`DecisionQuotient.Foundations.counting_additive`                                                                       |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`BC4`]{#lh:BC4}`DecisionQuotient.Foundations.bayes_from_conditional`                                                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`BC5`]{#lh:BC5}`DecisionQuotient.Foundations.entropy_contraction`                                                                     |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`BF1`]{#lh:BF1}`DecisionQuotient.certainty_of_not_nondegenerateBelief`                                                                |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`BF2`]{#lh:BF2}`DecisionQuotient.nondegenerateBelief_of_uncertaintyForced`                                                            |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`BF3`]{#lh:BF3}`DecisionQuotient.forced_action_under_uncertainty`                                                                     |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`BF4`]{#lh:BF4}`DecisionQuotient.bayes_update_exists_of_nondegenerateBelief`                                                          |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`BR1`]{#lh:BR1}`DecisionQuotient.Bridges.eth_structural_rank_exponential_hardness`                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`BR2`]{#lh:BR2}`DecisionQuotient.Bridges.fisher_rank_lower_bounds_sufficient_set`                                                     |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`BR3`]{#lh:BR3}`DecisionQuotient.Bridges.fpt_srank_parameterized_dichotomy`                                                           |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`BR4`]{#lh:BR4}`DecisionQuotient.Bridges.tur_srank_thermodynamic_cost`                                                                |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`BR5`]{#lh:BR5}`DecisionQuotient.Bridges.dichotomy_eth_complete_classification`                                                       |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`BR6`]{#lh:BR6}`DecisionQuotient.Bridges.reduction_eth_conp_exponential`                                                              |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`BR7`]{#lh:BR7}`DecisionQuotient.Bridges.geometry_covering_certificate_complexity`                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`BR8`]{#lh:BR8}`DecisionQuotient.Bridges.rate_distortion_fisher_information_bridge`                                                   |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`BR9`]{#lh:BR9}`DecisionQuotient.Bridges.counting_complexity_sharp_p_hardness`                                                        |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`BR10`]{#lh:BR10}`DecisionQuotient.Bridges.approximation_counting_hardness_bridge`                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC1`]{#lh:CC1}`DecisionQuotient.ClaimClosure.RegimeSimulation`                                                                       |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC2`]{#lh:CC2}`DecisionQuotient.ClaimClosure.adq_ordering`                                                                           |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC3`]{#lh:CC3}`DecisionQuotient.ClaimClosure.system_transfer_licensed_iff_snapshot`                                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC4`]{#lh:CC4}`DecisionQuotient.ClaimClosure.anchor_sigma2p_complete_conditional`                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC5`]{#lh:CC5}`DecisionQuotient.ClaimClosure.anchor_sigma2p_reduction_core`                                                          |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC6`]{#lh:CC6}`DecisionQuotient.ClaimClosure.anchor_query_relation_false_iff`                                                        |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC7`]{#lh:CC7}`DecisionQuotient.ClaimClosure.anchor_query_relation_true_iff`                                                         |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC8`]{#lh:CC8}`DecisionQuotient.ClaimClosure.boundaryCharacterized_iff_exists_sufficient_subset`                                     |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC9`]{#lh:CC9}`DecisionQuotient.ClaimClosure.bounded_actions_detectable`                                                             |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC10`]{#lh:CC10}`DecisionQuotient.ClaimClosure.bridge_boundary_represented_family`                                                   |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC11`]{#lh:CC11}`DecisionQuotient.ClaimClosure.bridge_failure_witness_non_one_step`                                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC12`]{#lh:CC12}`DecisionQuotient.ClaimClosure.bridge_transfer_iff_one_step_class`                                                   |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC13`]{#lh:CC13}`DecisionQuotient.ClaimClosure.certified_total_bits_split_core`                                                      |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC14`]{#lh:CC14}`DecisionQuotient.ClaimClosure.cost_asymmetry_eth_conditional`                                                       |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC15`]{#lh:CC15}`DecisionQuotient.ClaimClosure.declaredBudgetSlice`                                                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC16`]{#lh:CC16}`DecisionQuotient.ClaimClosure.declaredRegimeFamily_complete`                                                        |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC17`]{#lh:CC17}`DecisionQuotient.ClaimClosure.declared_physics_no_universal_exact_certifier_core`                                   |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC18`]{#lh:CC18}`DecisionQuotient.ClaimClosure.dichotomy_conditional`                                                                |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC19`]{#lh:CC19}`DecisionQuotient.ClaimClosure.epsilon_admissible_iff_raw_lt_certified_total_core`                                   |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC20`]{#lh:CC20}`DecisionQuotient.ClaimClosure.exact_admissible_iff_raw_lt_certified_total_core`                                     |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC21`]{#lh:CC21}`DecisionQuotient.ClaimClosure.exact_certainty_inflation_under_hardness_core`                                        |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC22`]{#lh:CC22}`DecisionQuotient.ClaimClosure.exact_raw_eq_certified_iff_certainty_inflation_core`                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC23`]{#lh:CC23}`DecisionQuotient.ClaimClosure.exact_raw_only_of_no_exact_admissible_core`                                           |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC24`]{#lh:CC24}`DecisionQuotient.ClaimClosure.explicit_assumptions_required_of_not_excused_core`                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC25`]{#lh:CC25}`DecisionQuotient.ClaimClosure.explicit_state_upper_core`                                                            |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC26`]{#lh:CC26}`DecisionQuotient.ClaimClosure.hard_family_all_coords_core`                                                          |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC27`]{#lh:CC27}`DecisionQuotient.ClaimClosure.horizonTwoWitness_immediate_empty_sufficient`                                         |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC28`]{#lh:CC28}`DecisionQuotient.ClaimClosure.horizon_gt_one_bridge_can_fail_on_sufficiency`                                        |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC29`]{#lh:CC29}`DecisionQuotient.ClaimClosure.information_barrier_opt_oracle_core`                                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC30`]{#lh:CC30}`DecisionQuotient.ClaimClosure.information_barrier_state_batch_core`                                                 |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC31`]{#lh:CC31}`DecisionQuotient.ClaimClosure.information_barrier_value_entry_core`                                                 |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC32`]{#lh:CC32}`DecisionQuotient.ClaimClosure.integrity_resource_bound_for_sufficiency`                                             |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC33`]{#lh:CC33}`DecisionQuotient.ClaimClosure.integrity_universal_applicability_core`                                               |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC34`]{#lh:CC34}`DecisionQuotient.ClaimClosure.meta_coordinate_irrelevant_of_invariance_on_declared_slice`                           |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC35`]{#lh:CC35}`DecisionQuotient.ClaimClosure.meta_coordinate_not_relevant_on_declared_slice`                                       |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC36`]{#lh:CC36}`DecisionQuotient.ClaimClosure.minsuff_collapse_core`                                                                |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC37`]{#lh:CC37}`DecisionQuotient.ClaimClosure.minsuff_collapse_to_conp_conditional`                                                 |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC38`]{#lh:CC38}`DecisionQuotient.ClaimClosure.minsuff_conp_complete_conditional`                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC39`]{#lh:CC39}`DecisionQuotient.ClaimClosure.no_auto_minimize_of_p_neq_conp`                                                       |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC40`]{#lh:CC40}`DecisionQuotient.ClaimClosure.no_exact_claim_admissible_under_hardness_core`                                        |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC41`]{#lh:CC41}`DecisionQuotient.ClaimClosure.no_exact_claim_under_declared_assumptions_unless_excused_core`                        |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC42`]{#lh:CC42}`DecisionQuotient.ClaimClosure.no_exact_identifier_implies_not_boundary_characterized`                               |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC43`]{#lh:CC43}`DecisionQuotient.ClaimClosure.no_uncertified_exact_claim_core`                                                      |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC44`]{#lh:CC44}`DecisionQuotient.ClaimClosure.one_step_bridge`                                                                      |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC45`]{#lh:CC45}`DecisionQuotient.ClaimClosure.oracle_lattice_transfer_as_regime_simulation`                                         |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC46`]{#lh:CC46}`DecisionQuotient.ClaimClosure.physical_crossover_above_cap_core`                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC47`]{#lh:CC47}`DecisionQuotient.ClaimClosure.physical_crossover_core`                                                              |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC48`]{#lh:CC48}`DecisionQuotient.ClaimClosure.physical_crossover_hardness_core`                                                     |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC49`]{#lh:CC49}`DecisionQuotient.ClaimClosure.physical_crossover_policy_core`                                                       |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC50`]{#lh:CC50}`DecisionQuotient.ClaimClosure.process_bridge_failure_witness`                                                       |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC51`]{#lh:CC51}`DecisionQuotient.ClaimClosure.poseAnchorQuery`                                                                      |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC52`]{#lh:CC52}`DecisionQuotient.ClaimClosure.pose_returns_anchor_query_object`                                                     |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC53`]{#lh:CC53}`DecisionQuotient.ClaimClosure.posed_anchor_checked_true_implies_truth`                                              |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC54`]{#lh:CC54}`DecisionQuotient.ClaimClosure.posed_anchor_exact_claim_admissible_iff_competent`                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC55`]{#lh:CC55}`DecisionQuotient.ClaimClosure.posed_anchor_exact_claim_requires_evidence`                                           |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC56`]{#lh:CC56}`DecisionQuotient.ClaimClosure.posed_anchor_no_competence_no_exact_claim`                                            |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC57`]{#lh:CC57}`DecisionQuotient.ClaimClosure.posed_anchor_query_truth_iff_exists_anchor`                                           |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC58`]{#lh:CC58}`DecisionQuotient.ClaimClosure.posed_anchor_query_truth_iff_exists_forall`                                           |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC59`]{#lh:CC59}`DecisionQuotient.ClaimClosure.posed_anchor_signal_positive_certified_implies_admissible`                            |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC60`]{#lh:CC60}`DecisionQuotient.ClaimClosure.query_obstruction_boolean_corollary`                                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC61`]{#lh:CC61}`DecisionQuotient.ClaimClosure.query_obstruction_finite_state_core`                                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC62`]{#lh:CC62}`DecisionQuotient.ClaimClosure.regime_core_claim_proved`                                                             |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC63`]{#lh:CC63}`DecisionQuotient.ClaimClosure.regime_simulation_transfers_hardness`                                                 |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC64`]{#lh:CC64}`DecisionQuotient.ClaimClosure.reusable_heuristic_of_detectable`                                                     |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC65`]{#lh:CC65}`DecisionQuotient.ClaimClosure.selectorSufficient_not_implies_setSufficient`                                         |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC66`]{#lh:CC66}`DecisionQuotient.ClaimClosure.separable_detectable`                                                                 |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC67`]{#lh:CC67}`DecisionQuotient.ClaimClosure.snapshot_vs_process_typed_boundary`                                                   |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC68`]{#lh:CC68}`DecisionQuotient.ClaimClosure.standard_assumption_ledger_unpack`                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC69`]{#lh:CC69}`DecisionQuotient.ClaimClosure.stochastic_objective_bridge_can_fail_on_sufficiency`                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC70`]{#lh:CC70}`DecisionQuotient.ClaimClosure.subproblem_hardness_lifts_to_full`                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC71`]{#lh:CC71}`DecisionQuotient.ClaimClosure.subproblem_transfer_as_regime_simulation`                                             |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC72`]{#lh:CC72}`DecisionQuotient.ClaimClosure.sufficiency_conp_complete_conditional`                                                |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC73`]{#lh:CC73}`DecisionQuotient.ClaimClosure.sufficiency_conp_reduction_core`                                                      |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC74`]{#lh:CC74}`DecisionQuotient.ClaimClosure.sufficiency_iff_dq_ratio`                                                             |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC75`]{#lh:CC75}`DecisionQuotient.ClaimClosure.sufficiency_iff_projectedOptCover_eq_opt`                                             |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC76`]{#lh:CC76}`DecisionQuotient.ClaimClosure.thermo_conservation_additive_core`                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC77`]{#lh:CC77}`DecisionQuotient.ClaimClosure.thermo_energy_carbon_lift_core`                                                       |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC78`]{#lh:CC78}`DecisionQuotient.ClaimClosure.thermo_eventual_lift_core`                                                            |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC79`]{#lh:CC79}`DecisionQuotient.ClaimClosure.thermo_hardness_bundle_core`                                                          |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC80`]{#lh:CC80}`DecisionQuotient.ClaimClosure.thermo_mandatory_cost_core`                                                           |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC81`]{#lh:CC81}`DecisionQuotient.ClaimClosure.tractable_bounded_core`                                                               |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC82`]{#lh:CC82}`DecisionQuotient.ClaimClosure.tractable_separable_core`                                                             |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC83`]{#lh:CC83}`DecisionQuotient.ClaimClosure.tractable_subcases_conditional`                                                       |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC84`]{#lh:CC84}`DecisionQuotient.ClaimClosure.tractable_tree_core`                                                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC85`]{#lh:CC85}`DecisionQuotient.ClaimClosure.transition_coupled_bridge_can_fail_on_sufficiency`                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC86`]{#lh:CC86}`DecisionQuotient.ClaimClosure.tree_structure_detectable`                                                            |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC87`]{#lh:CC87}`DecisionQuotient.ClaimClosure.typed_claim_admissibility_core`                                                       |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC88`]{#lh:CC88}`DecisionQuotient.ClaimClosure.typed_static_class_completeness`                                                      |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CC89`]{#lh:CC89}`DecisionQuotient.ClaimClosure.universal_solver_framing_core`                                                        |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CCC1`]{#lh:CCC1}`DecisionQuotient.CC.anchor_sigma2p_complete_conditional`                                                            |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CCC2`]{#lh:CCC2}`DecisionQuotient.CC.cost_asymmetry_eth_conditional`                                                                 |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CCC3`]{#lh:CCC3}`DecisionQuotient.CC.dichotomy_conditional`                                                                          |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CCC4`]{#lh:CCC4}`DecisionQuotient.CC.minsuff_collapse_to_conp_conditional`                                                           |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CCC5`]{#lh:CCC5}`DecisionQuotient.CC.minsuff_conp_complete_conditional`                                                              |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CCC6`]{#lh:CCC6}`DecisionQuotient.CC.sufficiency_conp_complete_conditional`                                                          |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CCC7`]{#lh:CCC7}`DecisionQuotient.CC.tractable_subcases_conditional`                                                                 |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CF1`]{#lh:CF1}`DecisionQuotient.Physics.ConstraintForcing.laws_not_determined_of_parameter_separation`                               |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CF2`]{#lh:CF2}`DecisionQuotient.Physics.ConstraintForcing.logic_time_not_sufficient_for_unique_law`                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CF3`]{#lh:CF3}`DecisionQuotient.Physics.ConstraintForcing.laws_determined_implies_objective_determined`                              |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CF4`]{#lh:CF4}`DecisionQuotient.Physics.ConstraintForcing.objective_not_determined_of_parameter_separation`                          |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CF5`]{#lh:CF5}`DecisionQuotient.Physics.ConstraintForcing.forcedDecisionBits_pos_of_deadline`                                        |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CF6`]{#lh:CF6}`DecisionQuotient.Physics.ConstraintForcing.actionForced_of_deadline`                                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CF7`]{#lh:CF7}`DecisionQuotient.Physics.ConstraintForcing.nondegenerateBelief_of_deadline_and_uncertainty`                           |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CF8`]{#lh:CF8}`DecisionQuotient.Physics.ConstraintForcing.forced_decision_implies_positive_landauer_cost`                            |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CF9`]{#lh:CF9}`DecisionQuotient.Physics.ConstraintForcing.forced_decision_implies_positive_nv_work`                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CH1`]{#lh:CH1}`DecisionQuotient.ClaimClosure.CH1`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CH2`]{#lh:CH2}`DecisionQuotient.ClaimClosure.CH2`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CH3`]{#lh:CH3}`DecisionQuotient.ClaimClosure.CH3`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CH5`]{#lh:CH5}`DecisionQuotient.ClaimClosure.CH5`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CH6`]{#lh:CH6}`DecisionQuotient.ClaimClosure.CH6`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CR1`]{#lh:CR1}`DecisionQuotient.ConfigReduction.config_sufficiency_iff_behavior_preserving`                                          |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CT1`]{#lh:CT1}`DecisionQuotient.Physics.ClaimTransport.PhysicalEncoding`                                                             |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CT2`]{#lh:CT2}`DecisionQuotient.Physics.ClaimTransport.physical_claim_lifts_from_core`                                               |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CT3`]{#lh:CT3}`DecisionQuotient.Physics.ClaimTransport.physical_claim_lifts_from_core_conditional`                                   |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CT4`]{#lh:CT4}`DecisionQuotient.Physics.ClaimTransport.physical_counterexample_yields_core_counterexample`                           |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CT5`]{#lh:CT5}`DecisionQuotient.Physics.ClaimTransport.physical_counterexample_invalidates_core_rule`                                |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CT6`]{#lh:CT6}`DecisionQuotient.Physics.ClaimTransport.no_physical_counterexample_of_core_theorem`                                   |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CT7`]{#lh:CT7}`DecisionQuotient.Physics.ClaimTransport.LawGapInstance`                                                               |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CT8`]{#lh:CT8}`DecisionQuotient.Physics.ClaimTransport.lawGapEncoding`                                                               |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CT9`]{#lh:CT9}`DecisionQuotient.Physics.ClaimTransport.lawGapPhysicalClaim`                                                          |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CT10`]{#lh:CT10}`DecisionQuotient.Physics.ClaimTransport.law_gap_physical_claim_holds`                                               |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CT11`]{#lh:CT11}`DecisionQuotient.Physics.ClaimTransport.no_law_gap_counterexample`                                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CT12`]{#lh:CT12}`DecisionQuotient.Physics.ClaimTransport.physical_bridge_bundle`                                                     |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CV1`]{#lh:CV1}`DecisionQuotient.Physics.Conversation.RecurrentCircuit`                                                               |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CV2`]{#lh:CV2}`DecisionQuotient.Physics.Conversation.CoupledConversation`                                                            |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CV3`]{#lh:CV3}`DecisionQuotient.Physics.Conversation.JointState`                                                                     |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CV4`]{#lh:CV4}`DecisionQuotient.Physics.Conversation.tick_uses_shared_node`                                                          |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CV5`]{#lh:CV5}`DecisionQuotient.Physics.Conversation.tick_shared_is_merged_emissions`                                                |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CV6`]{#lh:CV6}`DecisionQuotient.Physics.Conversation.channel_projection_eq_iff_quantized_eq`                                         |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CV7`]{#lh:CV7}`DecisionQuotient.Physics.Conversation.clamp_projection_eq_iff_same_clamped_bit`                                       |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CV10`]{#lh:CV10}`DecisionQuotient.Physics.Conversation.explanationGap_add_explanationBits`                                           |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CV11`]{#lh:CV11}`DecisionQuotient.Physics.Conversation.toClaimReport`                                                                |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CV12`]{#lh:CV12}`DecisionQuotient.Physics.Conversation.abstain_iff_no_answer`                                                        |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CV13`]{#lh:CV13}`DecisionQuotient.Physics.Conversation.yes_no_iff_exact_claim`                                                       |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CV14`]{#lh:CV14}`DecisionQuotient.Physics.Conversation.toReportSignal_completion_defined_of_budget`                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CV15`]{#lh:CV15}`DecisionQuotient.Physics.Conversation.toReportSignal_signal_consistent_zero_certified`                              |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CV16`]{#lh:CV16}`DecisionQuotient.Physics.Conversation.abstain_report_can_carry_explanation`                                         |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CV17`]{#lh:CV17}`DecisionQuotient.Physics.Conversation.clampDecisionEvent_iff_bitOps_pos`                                            |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CV18`]{#lh:CV18}`DecisionQuotient.Physics.Conversation.clamp_event_implies_positive_energy`                                          |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CV19`]{#lh:CV19}`DecisionQuotient.Physics.Conversation.BinaryAnswer`                                                                 |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CV20`]{#lh:CV20}`DecisionQuotient.Physics.Conversation.ConversationReport`                                                           |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CV21`]{#lh:CV21}`DecisionQuotient.Physics.Conversation.explanationGap`                                                               |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CV26`]{#lh:CV26}`DecisionQuotient.Physics.Conversation.toClaimReport_ne_epsilon`                                                     |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CV27`]{#lh:CV27}`DecisionQuotient.Physics.Conversation.toReportSignal`                                                               |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`CV28`]{#lh:CV28}`DecisionQuotient.Physics.Conversation.toReportSignal_declares_bound`                                                |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DC1`]{#lh:DC1}`DecisionQuotient.StochasticSequential.static_stochastic_strict_separation`                                            |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DC2`]{#lh:DC2}`DecisionQuotient.StochasticSequential.stochastic_sequential_strict_separation`                                        |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DC3`]{#lh:DC3}`DecisionQuotient.StochasticSequential.complexity_dichotomy_hierarchy`                                                 |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DC4`]{#lh:DC4}`DecisionQuotient.StochasticSequential.regime_hierarchy`                                                               |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DC5`]{#lh:DC5}`DecisionQuotient.StochasticSequential.coNP_subset_PP`                                                                 |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DC6`]{#lh:DC6}`DecisionQuotient.StochasticSequential.PP_subset_PSPACE`                                                               |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DC7`]{#lh:DC7}`DecisionQuotient.StochasticSequential.coNP_subset_PSPACE`                                                             |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DC8`]{#lh:DC8}`DecisionQuotient.StochasticSequential.static_to_coNP`                                                                 |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DC9`]{#lh:DC9}`DecisionQuotient.StochasticSequential.stochastic_to_PP`                                                               |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DC10`]{#lh:DC10}`DecisionQuotient.StochasticSequential.sequential_to_PSPACE`                                                         |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DC11`]{#lh:DC11}`DecisionQuotient.StochasticSequential.ClaimClosure.claim_six_subcases`                                              |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DC12`]{#lh:DC12}`DecisionQuotient.StochasticSequential.ClaimClosure.claim_hierarchy`                                                 |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DC13`]{#lh:DC13}`DecisionQuotient.StochasticSequential.ClaimClosure.claim_tractable_subcases_to_P`                                   |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DC14`]{#lh:DC14}`DecisionQuotient.StochasticSequential.stochastic_dichotomy`                                                         |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DC15`]{#lh:DC15}`DecisionQuotient.StochasticSequential.above_threshold_hard`                                                         |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DC16`]{#lh:DC16}`DecisionQuotient.StochasticSequential.StochasticAnchorSufficient`                                                   |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DC17`]{#lh:DC17}`DecisionQuotient.StochasticSequential.StochasticAnchorSufficiencyCheck`                                             |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DC18`]{#lh:DC18}`DecisionQuotient.StochasticSequential.stochastic_anchor_check_iff`                                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DC19`]{#lh:DC19}`DecisionQuotient.StochasticSequential.stochastic_anchor_sufficient_of_stochastic_sufficient`                        |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DC20`]{#lh:DC20}`DecisionQuotient.StochasticSequential.SequentialAnchorSufficient`                                                   |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DC21`]{#lh:DC21}`DecisionQuotient.StochasticSequential.SequentialAnchorSufficiencyCheck`                                             |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DC22`]{#lh:DC22}`DecisionQuotient.StochasticSequential.sequential_anchor_check_iff`                                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DC23`]{#lh:DC23}`DecisionQuotient.StochasticSequential.sequential_anchor_sufficient_of_sequential_sufficient`                        |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DC24`]{#lh:DC24}`DecisionQuotient.StochasticSequential.StochasticAnchorCheckInstance`                                                |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DC25`]{#lh:DC25}`DecisionQuotient.StochasticSequential.reduceMAJSAT_correct_anchor_strict`                                           |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DC26`]{#lh:DC26}`DecisionQuotient.StochasticSequential.reduceMAJSAT_to_stochastic_anchor_reduction`                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DC27`]{#lh:DC27}`DecisionQuotient.StochasticSequential.SequentialAnchorCheckInstance`                                                |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DC28`]{#lh:DC28}`DecisionQuotient.StochasticSequential.reduceTQBF_correct_anchor`                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DC29`]{#lh:DC29}`DecisionQuotient.StochasticSequential.reduceTQBF_to_sequential_anchor_reduction`                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DC30`]{#lh:DC30}`DecisionQuotient.StochasticSequential.StatePotential`                                                               |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DC31`]{#lh:DC31}`DecisionQuotient.StochasticSequential.utilityFromPotentialDrop_le_iff_nextPotential_ge`                             |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DC32`]{#lh:DC32}`DecisionQuotient.StochasticSequential.utility_from_action_state_potential`                                          |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DC33`]{#lh:DC33}`DecisionQuotient.StochasticSequential.stochasticExpectedUtility_eq_neg_expectedActionPotential`                     |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DC34`]{#lh:DC34}`DecisionQuotient.StochasticSequential.stochasticExpectedUtility_le_iff_expectedActionPotential_ge`                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DC35`]{#lh:DC35}`DecisionQuotient.StochasticSequential.landauerEnergyFloor_nonneg`                                                   |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DC36`]{#lh:DC36}`DecisionQuotient.StochasticSequential.landauerEnergyFloor_mono_bits`                                                |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DC37`]{#lh:DC37}`DecisionQuotient.StochasticSequential.thermodynamicCost_eq_landauerEnergyFloorRoom_states`                          |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DE1`]{#lh:DE1}`DecisionQuotient.ClaimClosure.DE1`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DE2`]{#lh:DE2}`DecisionQuotient.ClaimClosure.DE2`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DE3`]{#lh:DE3}`DecisionQuotient.ClaimClosure.DE3`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DE4`]{#lh:DE4}`DecisionQuotient.ClaimClosure.DE4`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DG1`]{#lh:DG1}`DecisionQuotient.Outside`                                                                                             |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DG2`]{#lh:DG2}`DecisionQuotient.anchoredSlice`                                                                                       |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DG3`]{#lh:DG3}`DecisionQuotient.anchoredSliceEquivOutside`                                                                           |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DG4`]{#lh:DG4}`DecisionQuotient.card_outside_eq_sub`                                                                                 |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DG5`]{#lh:DG5}`DecisionQuotient.card_anchoredSlice`                                                                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DG6`]{#lh:DG6}`DecisionQuotient.card_anchoredSlice_eq_pow_sub`                                                                       |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DG7`]{#lh:DG7}`DecisionQuotient.card_anchoredSlice_eq_uniform`                                                                       |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DG8`]{#lh:DG8}`DecisionQuotient.anchoredSlice_mul_fixed_eq_full`                                                                     |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DG9`]{#lh:DG9}`DecisionQuotient.constantBoolDP`                                                                                      |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DG10`]{#lh:DG10}`DecisionQuotient.firstCoordDP`                                                                                      |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DG11`]{#lh:DG11}`DecisionQuotient.constantBoolDP_opt`                                                                                |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DG12`]{#lh:DG12}`DecisionQuotient.firstCoordDP_opt`                                                                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DG13`]{#lh:DG13}`DecisionQuotient.constantBoolDP_empty_sufficient`                                                                   |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DG14`]{#lh:DG14}`DecisionQuotient.firstCoordDP_empty_not_sufficient`                                                                 |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DG15`]{#lh:DG15}`DecisionQuotient.boolHypercube_node_count`                                                                          |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DG16`]{#lh:DG16}`DecisionQuotient.node_count_does_not_determine_edge_geometry`                                                       |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DG17`]{#lh:DG17}`DecisionQuotient.DecisionProblem.edgeOnComplement`                                                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DG18`]{#lh:DG18}`DecisionQuotient.DecisionProblem.edgeOnComplement_iff_not_sufficient`                                               |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DP1`]{#lh:DP1}`DecisionQuotient.DecisionProblem.minimalSufficient_iff_relevant`                                                      |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DP2`]{#lh:DP2}`DecisionQuotient.DecisionProblem.relevantSet_is_minimal`                                                              |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DP3`]{#lh:DP3}`DecisionQuotient.DecisionProblem.sufficient_implies_selectorSufficient`                                               |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DP4`]{#lh:DP4}`DecisionQuotient.ClaimClosure.DecisionProblem.epsOpt_zero_eq_opt`                                                     |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DP5`]{#lh:DP5}`DecisionQuotient.ClaimClosure.DecisionProblem.sufficient_iff_zeroEpsilonSufficient`                                   |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DP6`]{#lh:DP6}`DecisionQuotient.ClaimClosure.DP6`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DP7`]{#lh:DP7}`DecisionQuotient.ClaimClosure.DP7`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DP8`]{#lh:DP8}`DecisionQuotient.ClaimClosure.DP8`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DQ1`]{#lh:DQ1}`DecisionQuotient.ClaimClosure.DQ1`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DQ2`]{#lh:DQ2}`DecisionQuotient.ClaimClosure.DQ2`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DQ3`]{#lh:DQ3}`DecisionQuotient.ClaimClosure.DQ3`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DQ4`]{#lh:DQ4}`DecisionQuotient.ClaimClosure.DQ4`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DQ5`]{#lh:DQ5}`DecisionQuotient.ClaimClosure.DQ5`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DQ6`]{#lh:DQ6}`DecisionQuotient.ClaimClosure.DQ6`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DQ7`]{#lh:DQ7}`DecisionQuotient.ClaimClosure.DQ7`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DQ8`]{#lh:DQ8}`DecisionQuotient.ClaimClosure.DQ8`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DS1`]{#lh:DS1}`DecisionQuotient.ClaimClosure.DS1`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DS2`]{#lh:DS2}`DecisionQuotient.ClaimClosure.DS2`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DS3`]{#lh:DS3}`DecisionQuotient.ClaimClosure.DS3`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DS4`]{#lh:DS4}`DecisionQuotient.ClaimClosure.DS4`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DS5`]{#lh:DS5}`DecisionQuotient.ClaimClosure.DS5`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DS6`]{#lh:DS6}`DecisionQuotient.ClaimClosure.DS6`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DT1`]{#lh:DT1}`DecisionQuotient.Physics.DecisionTime.TimedState`                                                                     |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DT2`]{#lh:DT2}`DecisionQuotient.Physics.DecisionTime.DecisionProcess`                                                                |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DT3`]{#lh:DT3}`DecisionQuotient.Physics.DecisionTime.tick`                                                                           |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DT4`]{#lh:DT4}`DecisionQuotient.Physics.DecisionTime.DecisionEvent`                                                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DT5`]{#lh:DT5}`DecisionQuotient.Physics.DecisionTime.TimeUnitStep`                                                                   |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DT6`]{#lh:DT6}`DecisionQuotient.Physics.DecisionTime.time_is_discrete`                                                               |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DT7`]{#lh:DT7}`DecisionQuotient.Physics.DecisionTime.time_coordinate_falsifiable`                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DT8`]{#lh:DT8}`DecisionQuotient.Physics.DecisionTime.tick_increments_time`                                                           |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DT9`]{#lh:DT9}`DecisionQuotient.Physics.DecisionTime.tick_decision_witness`                                                          |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DT10`]{#lh:DT10}`DecisionQuotient.Physics.DecisionTime.tick_is_decision_event`                                                       |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DT11`]{#lh:DT11}`DecisionQuotient.Physics.DecisionTime.decision_event_implies_time_unit`                                             |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DT12`]{#lh:DT12}`DecisionQuotient.Physics.DecisionTime.decision_taking_place_is_unit_of_time`                                        |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DT13`]{#lh:DT13}`DecisionQuotient.Physics.DecisionTime.decision_event_iff_eq_tick`                                                   |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DT14`]{#lh:DT14}`DecisionQuotient.Physics.DecisionTime.run`                                                                          |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DT15`]{#lh:DT15}`DecisionQuotient.Physics.DecisionTime.run_time_exact`                                                               |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DT16`]{#lh:DT16}`DecisionQuotient.Physics.DecisionTime.run_elapsed_time_eq_ticks`                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DT17`]{#lh:DT17}`DecisionQuotient.Physics.DecisionTime.decisionTrace`                                                                |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DT18`]{#lh:DT18}`DecisionQuotient.Physics.DecisionTime.decisionTrace_length_eq_ticks`                                                |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DT19`]{#lh:DT19}`DecisionQuotient.Physics.DecisionTime.decision_count_equals_elapsed_time`                                           |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DT20`]{#lh:DT20}`DecisionQuotient.Physics.DecisionTime.SubstrateKind`                                                                |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DT21`]{#lh:DT21}`DecisionQuotient.Physics.DecisionTime.SubstrateModel`                                                               |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DT22`]{#lh:DT22}`DecisionQuotient.Physics.DecisionTime.substrate_step_realizes_decision_event`                                       |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DT23`]{#lh:DT23}`DecisionQuotient.Physics.DecisionTime.substrate_step_is_time_unit`                                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`DT24`]{#lh:DT24}`DecisionQuotient.Physics.DecisionTime.time_unit_law_substrate_invariant`                                            |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`EI1`]{#lh:EI1}`DecisionQuotient.ThermodynamicLift.energy_ge_kbt_nat_entropy`                                                         |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`EI2`]{#lh:EI2}`DecisionQuotient.ThermodynamicLift.energy_ground_state_tracks_entropy`                                                |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`EI4`]{#lh:EI4}`DecisionQuotient.ThermodynamicLift.landauerJoulesPerBit_pos`                                                          |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`EI5`]{#lh:EI5}`DecisionQuotient.ThermodynamicLift.neukart_vinokur_duality`                                                           |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`EP1`]{#lh:EP1}`DecisionQuotient.Physics.LocalityPhysics.landauer_principle`                                                          |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`EP2`]{#lh:EP2}`DecisionQuotient.Physics.LocalityPhysics.finite_regional_energy`                                                      |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`EP3`]{#lh:EP3}`DecisionQuotient.Physics.LocalityPhysics.finite_signal_speed`                                                         |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`EP4`]{#lh:EP4}`DecisionQuotient.Physics.LocalityPhysics.nontrivial_physics`                                                          |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`FI3`]{#lh:FI3}`DecisionQuotient.FunctionalInformation.functional_information_from_counting`                                          |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`FI6`]{#lh:FI6}`DecisionQuotient.FunctionalInformation.functional_information_from_thermodynamics`                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`FI7`]{#lh:FI7}`DecisionQuotient.FunctionalInformation.first_principles_thermo_coincide`                                              |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`FN7`]{#lh:FN7}`DecisionQuotient.BayesOptimalityProof.KL_nonneg`                                                                      |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`FN8`]{#lh:FN8}`DecisionQuotient.BayesOptimalityProof.entropy_le_crossEntropy`                                                        |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`FN12`]{#lh:FN12}`DecisionQuotient.BayesOptimalityProof.crossEntropy_eq_entropy_add_KL`                                               |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`FN14`]{#lh:FN14}`DecisionQuotient.BayesOptimalityProof.bayes_is_optimal`                                                             |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`FN15`]{#lh:FN15}`DecisionQuotient.BayesOptimalityProof.KL_eq_sum_klFun`                                                              |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`FN16`]{#lh:FN16}`DecisionQuotient.BayesOptimalityProof.KL_eq_zero_iff_eq`                                                            |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`FP1`]{#lh:FP1}`DecisionQuotient.Physics.LocalityPhysics.trivial_states_all_equal`                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`FP2`]{#lh:FP2}`DecisionQuotient.Physics.LocalityPhysics.equal_states_constant_function`                                              |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`FP3`]{#lh:FP3}`DecisionQuotient.Physics.LocalityPhysics.constant_function_singleton_image`                                           |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`FP4`]{#lh:FP4}`DecisionQuotient.Physics.LocalityPhysics.singleton_image_zero_entropy`                                                |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`FP5`]{#lh:FP5}`DecisionQuotient.Physics.LocalityPhysics.zero_entropy_no_information`                                                 |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`FP6`]{#lh:FP6}`DecisionQuotient.Physics.LocalityPhysics.triviality_implies_no_information`                                           |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`FP7`]{#lh:FP7}`DecisionQuotient.Physics.LocalityPhysics.information_requires_nontriviality`                                          |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`FP8`]{#lh:FP8}`DecisionQuotient.Physics.LocalityPhysics.atypical_states_rare`                                                        |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`FP9`]{#lh:FP9}`DecisionQuotient.Physics.LocalityPhysics.random_misses_target`                                                        |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`FP10`]{#lh:FP10}`DecisionQuotient.Physics.LocalityPhysics.errors_accumulate`                                                         |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`FP11`]{#lh:FP11}`DecisionQuotient.Physics.LocalityPhysics.wrong_paths_dominate`                                                      |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`FP12`]{#lh:FP12}`DecisionQuotient.Physics.LocalityPhysics.second_law_from_counting`                                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`FP13`]{#lh:FP13}`DecisionQuotient.Physics.LocalityPhysics.verification_is_information`                                               |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`FP14`]{#lh:FP14}`DecisionQuotient.Physics.LocalityPhysics.entropy_is_information`                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`FP15`]{#lh:FP15}`DecisionQuotient.Physics.LocalityPhysics.landauer_structure`                                                        |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`FPT1`]{#lh:FPT1}`DecisionQuotient.Physics.LocalityPhysics.Timeline`                                                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`FPT2`]{#lh:FPT2}`DecisionQuotient.Physics.LocalityPhysics.FPT2_function_deterministic`                                               |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`FPT3`]{#lh:FPT3}`DecisionQuotient.Physics.LocalityPhysics.FPT3_outputs_differ_inputs_differ`                                         |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`FPT4`]{#lh:FPT4}`DecisionQuotient.Physics.LocalityPhysics.FPT4_step_requires_distinct_moments`                                       |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`FPT5`]{#lh:FPT5}`DecisionQuotient.Physics.LocalityPhysics.FPT5_distinct_moments_positive_duration`                                   |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`FPT6`]{#lh:FPT6}`DecisionQuotient.Physics.LocalityPhysics.FPT6_step_takes_positive_time`                                             |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`FPT7`]{#lh:FPT7}`DecisionQuotient.Physics.LocalityPhysics.FPT7_no_instantaneous_steps`                                               |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`FPT8`]{#lh:FPT8}`DecisionQuotient.Physics.LocalityPhysics.FPT8_propagation_takes_time`                                               |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`FPT9`]{#lh:FPT9}`DecisionQuotient.Physics.LocalityPhysics.FPT9_speed_bounded_by_positive_time`                                       |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`FPT10`]{#lh:FPT10}`DecisionQuotient.Physics.LocalityPhysics.FPT10_ec3_is_logical`                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`FS1`]{#lh:FS1}`DecisionQuotient.Statistics.sum_fisherScore_eq_srank`                                                                 |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`FS2`]{#lh:FS2}`DecisionQuotient.Statistics.fisherMatrix_rank_eq_srank`                                                               |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`FS3`]{#lh:FS3}`DecisionQuotient.Statistics.fisherMatrix_trace_eq_srank`                                                              |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`FS4`]{#lh:FS4}`DecisionQuotient.Statistics.fisherScore_relevant`                                                                     |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`FS5`]{#lh:FS5}`DecisionQuotient.Statistics.fisherScore_irrelevant`                                                                   |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`GE1`]{#lh:GE1}`DecisionQuotient.ClaimClosure.GE1`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`GE2`]{#lh:GE2}`DecisionQuotient.ClaimClosure.GE2`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`GE3`]{#lh:GE3}`DecisionQuotient.ClaimClosure.GE3`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`GE4`]{#lh:GE4}`DecisionQuotient.ClaimClosure.GE4`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`GE5`]{#lh:GE5}`DecisionQuotient.ClaimClosure.GE5`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`GE6`]{#lh:GE6}`DecisionQuotient.ClaimClosure.GE6`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`GE7`]{#lh:GE7}`DecisionQuotient.ClaimClosure.GE7`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`GE8`]{#lh:GE8}`DecisionQuotient.ClaimClosure.GE8`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`GE9`]{#lh:GE9}`DecisionQuotient.ClaimClosure.GE9`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`GN1`]{#lh:GN1}`DecisionQuotient.LogicGraph.isCycle`                                                                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`GN2`]{#lh:GN2}`DecisionQuotient.LogicGraph.cycleWitnessBits_pos`                                                                     |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`GN3`]{#lh:GN3}`DecisionQuotient.LogicGraph.pathProb_nonneg`                                                                          |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`GN4`]{#lh:GN4}`DecisionQuotient.LogicGraph.pathSurprisal_nonneg_of_positive_mass`                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`GN5`]{#lh:GN5}`DecisionQuotient.LogicGraph.nontrivialityScore_unknown`                                                               |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`GN6`]{#lh:GN6}`DecisionQuotient.LogicGraph.observerEntropy_nonneg`                                                                   |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`GN7`]{#lh:GN7}`DecisionQuotient.LogicGraph.dqFromEntropy_in_unit_interval`                                                           |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`GN8`]{#lh:GN8}`DecisionQuotient.LogicGraph.path_belief_forced_under_uncertainty`                                                     |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`GN9`]{#lh:GN9}`DecisionQuotient.LogicGraph.bayes_update_exists_for_observer_paths`                                                   |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`GN10`]{#lh:GN10}`DecisionQuotient.LogicGraph.cycle_witness_implies_positive_landauer`                                                |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`GN11`]{#lh:GN11}`DecisionQuotient.LogicGraph.cycle_witness_implies_positive_nv_work`                                                 |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`GN12`]{#lh:GN12}`DecisionQuotient.LogicGraph.dna_erasure_implies_positive_landauer`                                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`GN13`]{#lh:GN13}`DecisionQuotient.LogicGraph.dna_room_temp_environmental_stability`                                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`H1`]{#lh:H1}`...`                                                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`HD1`]{#lh:HD1}`DecisionQuotient.HardnessDistribution.centralization_dominance_bundle`                                                |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`HD2`]{#lh:HD2}`DecisionQuotient.HardnessDistribution.centralization_step_saves_n_minus_one`                                          |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`HD3`]{#lh:HD3}`DecisionQuotient.HardnessDistribution.centralized_higher_leverage`                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`HD4`]{#lh:HD4}`DecisionQuotient.HardnessDistribution.complete_model_dominates_after_threshold`                                       |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`HD5`]{#lh:HD5}`DecisionQuotient.HardnessDistribution.gap_conservation_card`                                                          |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`HD6`]{#lh:HD6}`DecisionQuotient.HardnessDistribution.generalizedTotal_with_saturation_eventually_constant`                           |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`HD7`]{#lh:HD7}`DecisionQuotient.HardnessDistribution.generalized_dominance_can_fail_without_right_boundedness`                       |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`HD8`]{#lh:HD8}`DecisionQuotient.HardnessDistribution.generalized_dominance_can_fail_without_wrong_growth`                            |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`HD9`]{#lh:HD9}`DecisionQuotient.HardnessDistribution.generalized_right_dominates_wrong_of_bounded_vs_identity_lower`                 |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`HD10`]{#lh:HD10}`DecisionQuotient.HardnessDistribution.generalized_right_eventually_dominates_wrong`                                 |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`HD11`]{#lh:HD11}`DecisionQuotient.HardnessDistribution.hardnessEfficiency_eq_central_share`                                          |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`HD12`]{#lh:HD12}`DecisionQuotient.HardnessDistribution.isRightHardness`                                                              |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`HD13`]{#lh:HD13}`DecisionQuotient.HardnessDistribution.isWrongHardness`                                                              |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`HD14`]{#lh:HD14}`DecisionQuotient.HardnessDistribution.linear_lt_exponential_plus_constant_eventually`                               |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`HD15`]{#lh:HD15}`DecisionQuotient.HardnessDistribution.native_dominates_manual`                                                      |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`HD16`]{#lh:HD16}`DecisionQuotient.HardnessDistribution.no_positive_slope_linear_represents_saturating`                               |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`HD17`]{#lh:HD17}`DecisionQuotient.HardnessDistribution.requiredWork`                                                                 |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`HD18`]{#lh:HD18}`DecisionQuotient.HardnessDistribution.requiredWork_eq_affine_in_sites`                                              |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`HD19`]{#lh:HD19}`DecisionQuotient.HardnessDistribution.right_dominates_wrong`                                                        |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`HD20`]{#lh:HD20}`DecisionQuotient.HardnessDistribution.saturatingSiteCost_eventually_constant`                                       |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`HD21`]{#lh:HD21}`DecisionQuotient.HardnessDistribution.simplicityTax_grows`                                                          |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`HD22`]{#lh:HD22}`DecisionQuotient.HardnessDistribution.hardnessLowerBound`                                                           |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`HD23`]{#lh:HD23}`DecisionQuotient.HardnessDistribution.hardness_is_irreducible_required_work`                                        |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`HD24`]{#lh:HD24}`DecisionQuotient.HardnessDistribution.totalDOF_eventually_constant_iff_zero_distributed`                            |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`HD25`]{#lh:HD25}`DecisionQuotient.HardnessDistribution.totalDOF_ge_intrinsic`                                                        |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`HD26`]{#lh:HD26}`DecisionQuotient.HardnessDistribution.totalExternalWork_eq_n_mul_gapCard`                                           |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`HD27`]{#lh:HD27}`DecisionQuotient.HardnessDistribution.workGrowthDegree`                                                             |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`HD28`]{#lh:HD28}`DecisionQuotient.HardnessDistribution.workGrowthDegree_zero_iff_eventually_constant`                                |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`HS1`]{#lh:HS1}`DecisionQuotient.Physics.HeisenbergStrong.NoisyPhysicalEncoding`                                                      |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`HS2`]{#lh:HS2}`DecisionQuotient.Physics.HeisenbergStrong.HeisenbergStrongBinding`                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`HS3`]{#lh:HS3}`DecisionQuotient.Physics.HeisenbergStrong.strong_binding_implies_core_nontrivial`                                     |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`HS4`]{#lh:HS4}`DecisionQuotient.Physics.HeisenbergStrong.strong_binding_yields_core_encoding_witness`                                |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`HS5`]{#lh:HS5}`DecisionQuotient.Physics.HeisenbergStrong.strong_binding_implies_physical_nontrivial_opt_assumption`                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`HS6`]{#lh:HS6}`DecisionQuotient.Physics.HeisenbergStrong.strong_binding_implies_nontrivial_opt_via_uncertainty`                      |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IA1`]{#lh:IA1}`DecisionQuotient.ClaimClosure.IA1`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IA2`]{#lh:IA2}`DecisionQuotient.ClaimClosure.IA2`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IA3`]{#lh:IA3}`DecisionQuotient.ClaimClosure.IA3`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IA4`]{#lh:IA4}`DecisionQuotient.ClaimClosure.IA4`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IA5`]{#lh:IA5}`DecisionQuotient.ClaimClosure.IA5`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IA6`]{#lh:IA6}`DecisionQuotient.ClaimClosure.IA6`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IA7`]{#lh:IA7}`DecisionQuotient.ClaimClosure.IA7`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IA8`]{#lh:IA8}`DecisionQuotient.ClaimClosure.IA8`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IA9`]{#lh:IA9}`DecisionQuotient.ClaimClosure.IA9`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IA10`]{#lh:IA10}`DecisionQuotient.ClaimClosure.IA10`                                                                                 |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IA11`]{#lh:IA11}`DecisionQuotient.ClaimClosure.IA11`                                                                                 |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IA12`]{#lh:IA12}`DecisionQuotient.ClaimClosure.IA12`                                                                                 |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IA13`]{#lh:IA13}`DecisionQuotient.ClaimClosure.IA13`                                                                                 |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IC1`]{#lh:IC1}`DecisionQuotient.IntegrityCompetence.CertaintyInflation`                                                              |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IC2`]{#lh:IC2}`DecisionQuotient.IntegrityCompetence.CompletionFractionDefined`                                                       |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IC3`]{#lh:IC3}`DecisionQuotient.IntegrityCompetence.EvidenceForReport`                                                               |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IC4`]{#lh:IC4}`DecisionQuotient.IntegrityCompetence.ExactCertaintyInflation`                                                         |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IC5`]{#lh:IC5}`DecisionQuotient.IntegrityCompetence.Percent`                                                                         |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IC6`]{#lh:IC6}`DecisionQuotient.IntegrityCompetence.RLFFWeights`                                                                     |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IC7`]{#lh:IC7}`DecisionQuotient.IntegrityCompetence.ReportSignal`                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IC8`]{#lh:IC8}`DecisionQuotient.IntegrityCompetence.ReportBitModel`                                                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IC9`]{#lh:IC9}`DecisionQuotient.IntegrityCompetence.SignalConsistent`                                                                |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IC10`]{#lh:IC10}`DecisionQuotient.IntegrityCompetence.admissible_irrational_strictly_more_than_rational`                             |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IC11`]{#lh:IC11}`DecisionQuotient.IntegrityCompetence.admissible_matrix_counts`                                                      |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IC12`]{#lh:IC12}`DecisionQuotient.IntegrityCompetence.abstain_signal_exists_with_guess_self`                                         |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IC13`]{#lh:IC13}`DecisionQuotient.IntegrityCompetence.certaintyInflation_iff_not_admissible`                                         |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IC14`]{#lh:IC14}`DecisionQuotient.IntegrityCompetence.certificationOverheadBits`                                                     |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IC15`]{#lh:IC15}`DecisionQuotient.IntegrityCompetence.certificationOverheadBits_of_evidence`                                         |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IC16`]{#lh:IC16}`DecisionQuotient.IntegrityCompetence.certificationOverheadBits_of_no_evidence`                                      |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IC17`]{#lh:IC17}`DecisionQuotient.IntegrityCompetence.certifiedTotalBits`                                                            |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IC18`]{#lh:IC18}`DecisionQuotient.IntegrityCompetence.certifiedTotalBits_ge_raw`                                                     |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IC19`]{#lh:IC19}`DecisionQuotient.IntegrityCompetence.certifiedTotalBits_gt_raw_of_evidence`                                         |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IC20`]{#lh:IC20}`DecisionQuotient.IntegrityCompetence.certifiedTotalBits_of_evidence`                                                |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IC21`]{#lh:IC21}`DecisionQuotient.IntegrityCompetence.certifiedTotalBits_of_no_evidence`                                             |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IC22`]{#lh:IC22}`DecisionQuotient.IntegrityCompetence.claim_admissible_of_evidence`                                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IC23`]{#lh:IC23}`DecisionQuotient.IntegrityCompetence.competence_implies_integrity`                                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IC24`]{#lh:IC24}`DecisionQuotient.IntegrityCompetence.completion_fraction_defined_of_declared_bound`                                 |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IC25`]{#lh:IC25}`DecisionQuotient.IntegrityCompetence.epsilon_competence_implies_integrity`                                          |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IC26`]{#lh:IC26}`DecisionQuotient.IntegrityCompetence.evidence_nonempty_iff_claim_admissible`                                        |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IC27`]{#lh:IC27}`DecisionQuotient.IntegrityCompetence.evidence_of_claim_admissible`                                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IC28`]{#lh:IC28}`DecisionQuotient.IntegrityCompetence.exact_claim_admissible_iff_exact_evidence_nonempty`                            |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IC29`]{#lh:IC29}`DecisionQuotient.IntegrityCompetence.exact_claim_requires_evidence`                                                 |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IC30`]{#lh:IC30}`DecisionQuotient.IntegrityCompetence.exactCertaintyInflation_iff_no_exact_competence`                               |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IC31`]{#lh:IC31}`DecisionQuotient.IntegrityCompetence.exact_raw_only_of_no_exact_admissible`                                         |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IC32`]{#lh:IC32}`DecisionQuotient.IntegrityCompetence.integrity_forces_abstention`                                                   |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IC33`]{#lh:IC33}`DecisionQuotient.IntegrityCompetence.integrity_not_competent_of_nonempty_scope`                                     |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IC34`]{#lh:IC34}`DecisionQuotient.IntegrityCompetence.integrity_resource_bound`                                                      |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IC35`]{#lh:IC35}`DecisionQuotient.IntegrityCompetence.no_completion_fraction_without_declared_bound`                                 |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IC36`]{#lh:IC36}`DecisionQuotient.IntegrityCompetence.overModelVerdict_rational_iff`                                                 |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IC37`]{#lh:IC37}`DecisionQuotient.IntegrityCompetence.percentZero`                                                                   |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IC38`]{#lh:IC38}`DecisionQuotient.IntegrityCompetence.rlffBaseReward`                                                                |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IC39`]{#lh:IC39}`DecisionQuotient.IntegrityCompetence.rlffReward`                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IC40`]{#lh:IC40}`DecisionQuotient.IntegrityCompetence.rlff_abstain_strictly_prefers_no_certificates`                                 |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IC41`]{#lh:IC41}`DecisionQuotient.IntegrityCompetence.rlff_maximizer_has_evidence`                                                   |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IC42`]{#lh:IC42}`DecisionQuotient.IntegrityCompetence.rlff_maximizer_is_admissible`                                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IC43`]{#lh:IC43}`DecisionQuotient.IntegrityCompetence.self_reflected_confidence_not_certification`                                   |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IC44`]{#lh:IC44}`DecisionQuotient.IntegrityCompetence.signal_certified_positive_implies_admissible`                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IC45`]{#lh:IC45}`DecisionQuotient.IntegrityCompetence.signal_consistent_of_claim_admissible`                                         |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IC46`]{#lh:IC46}`DecisionQuotient.IntegrityCompetence.signal_no_evidence_forces_zero_certified`                                      |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IC47`]{#lh:IC47}`DecisionQuotient.IntegrityCompetence.signal_exact_no_competence_forces_zero_certified`                              |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IC48`]{#lh:IC48}`DecisionQuotient.IntegrityCompetence.steps_run_scalar_always_defined`                                               |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IC49`]{#lh:IC49}`DecisionQuotient.IntegrityCompetence.steps_run_scalar_falsifiable`                                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IC50`]{#lh:IC50}`DecisionQuotient.IntegrityCompetence.zero_epsilon_competence_iff_exact`                                             |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IE1`]{#lh:IE1}`DecisionQuotient.ClaimClosure.IE1`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IE2`]{#lh:IE2}`DecisionQuotient.ClaimClosure.IE2`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IE3`]{#lh:IE3}`DecisionQuotient.ClaimClosure.IE3`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IE4`]{#lh:IE4}`DecisionQuotient.ClaimClosure.IE4`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IE5`]{#lh:IE5}`DecisionQuotient.ClaimClosure.IE5`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IE6`]{#lh:IE6}`DecisionQuotient.ClaimClosure.IE6`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IE7`]{#lh:IE7}`DecisionQuotient.ClaimClosure.IE7`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IE8`]{#lh:IE8}`DecisionQuotient.ClaimClosure.IE8`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IE9`]{#lh:IE9}`DecisionQuotient.ClaimClosure.IE9`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IE10`]{#lh:IE10}`DecisionQuotient.ClaimClosure.IE10`                                                                                 |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IE11`]{#lh:IE11}`DecisionQuotient.ClaimClosure.IE11`                                                                                 |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IE12`]{#lh:IE12}`DecisionQuotient.ClaimClosure.IE12`                                                                                 |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IE13`]{#lh:IE13}`DecisionQuotient.ClaimClosure.IE13`                                                                                 |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IE14`]{#lh:IE14}`DecisionQuotient.ClaimClosure.IE14`                                                                                 |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IE15`]{#lh:IE15}`DecisionQuotient.ClaimClosure.IE15`                                                                                 |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IE16`]{#lh:IE16}`DecisionQuotient.ClaimClosure.IE16`                                                                                 |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IE17`]{#lh:IE17}`DecisionQuotient.ClaimClosure.IE17`                                                                                 |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IN1`]{#lh:IN1}`DecisionQuotient.Physics.Instantiation.Geometry`                                                                      |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IN2`]{#lh:IN2}`DecisionQuotient.Physics.Instantiation.Dynamics`                                                                      |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IN3`]{#lh:IN3}`DecisionQuotient.Physics.Instantiation.Circuit`                                                                       |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IN4`]{#lh:IN4}`DecisionQuotient.Physics.Instantiation.geometry_plus_dynamics_is_circuit`                                             |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IN5`]{#lh:IN5}`DecisionQuotient.Physics.Instantiation.DecisionInterpretation`                                                        |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IN6`]{#lh:IN6}`DecisionQuotient.Physics.Instantiation.DecisionCircuit`                                                               |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IN7`]{#lh:IN7}`DecisionQuotient.Physics.Instantiation.Molecule`                                                                      |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IN8`]{#lh:IN8}`DecisionQuotient.Physics.Instantiation.Reaction`                                                                      |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IN9`]{#lh:IN9}`DecisionQuotient.Physics.Instantiation.ReactionOutcome`                                                               |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IN10`]{#lh:IN10}`DecisionQuotient.Physics.Instantiation.MoleculeGeometry`                                                            |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IN11`]{#lh:IN11}`DecisionQuotient.Physics.Instantiation.MoleculeDynamics`                                                            |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IN12`]{#lh:IN12}`DecisionQuotient.Physics.Instantiation.MoleculeCircuit`                                                             |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IN13`]{#lh:IN13}`DecisionQuotient.Physics.Instantiation.MoleculeAsCircuit`                                                           |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IN14`]{#lh:IN14}`DecisionQuotient.Physics.Instantiation.MoleculeAsDecisionCircuit`                                                   |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IN15`]{#lh:IN15}`DecisionQuotient.Physics.Instantiation.molecule_decision_preserves_geometry`                                        |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IN16`]{#lh:IN16}`DecisionQuotient.Physics.Instantiation.molecule_decision_preserves_dynamics`                                        |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IN17`]{#lh:IN17}`DecisionQuotient.Physics.Instantiation.asDecisionCircuit`                                                           |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IN18`]{#lh:IN18}`DecisionQuotient.Physics.Instantiation.asDecisionCircuit_preserves_circuit`                                         |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IN19`]{#lh:IN19}`DecisionQuotient.Physics.Instantiation.Configuration`                                                               |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IN20`]{#lh:IN20}`DecisionQuotient.Physics.Instantiation.EnergyLandscape`                                                             |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IN21`]{#lh:IN21}`DecisionQuotient.Physics.Instantiation.k_Boltzmann`                                                                 |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IN22`]{#lh:IN22}`DecisionQuotient.Physics.Instantiation.LandauerBound`                                                               |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IN23`]{#lh:IN23}`DecisionQuotient.Physics.Instantiation.law_objective_schema`                                                        |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IN24`]{#lh:IN24}`DecisionQuotient.Physics.Instantiation.law_opt_eq_feasible_of_gap`                                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IN25`]{#lh:IN25}`DecisionQuotient.Physics.Instantiation.law_opt_singleton_of_deterministic`                                          |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IP1`]{#lh:IP1}`DecisionQuotient.Physics.LocalityPhysics.ec1_can_be_true`                                                             |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IP2`]{#lh:IP2}`DecisionQuotient.Physics.LocalityPhysics.ec1_independent_of_ec2_ec3`                                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IP3`]{#lh:IP3}`DecisionQuotient.Physics.LocalityPhysics.ec2_can_be_true`                                                             |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IP4`]{#lh:IP4}`DecisionQuotient.Physics.LocalityPhysics.ec2_independent_of_ec1_ec3`                                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IP5`]{#lh:IP5}`DecisionQuotient.Physics.LocalityPhysics.ec3_can_be_true`                                                             |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IP6`]{#lh:IP6}`DecisionQuotient.Physics.LocalityPhysics.ec3_independent_of_ec1_ec2`                                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IP7`]{#lh:IP7}`DecisionQuotient.Physics.LocalityPhysics.empirical_claims_mutually_independent`                                       |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IT1`]{#lh:IT1}`DecisionQuotient.DecisionProblem.numOptClasses`                                                                       |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IT3`]{#lh:IT3}`DecisionQuotient.quotientEntropy_le_srank_binary`                                                                     |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IT4`]{#lh:IT4}`DecisionQuotient.numOptClasses_le_pow_srank_binary`                                                                   |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IT5`]{#lh:IT5}`DecisionQuotient.nontrivial_bounds_binary`                                                                            |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IT6`]{#lh:IT6}`DecisionQuotient.nontrivial_implies_srank_pos`                                                                        |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IV1`]{#lh:IV1}`DecisionQuotient.InteriorVerification.GoalClass`                                                                      |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IV2`]{#lh:IV2}`DecisionQuotient.InteriorVerification.InteriorDominanceVerifiable`                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IV3`]{#lh:IV3}`DecisionQuotient.InteriorVerification.TautologicalSetIdentifiable`                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IV4`]{#lh:IV4}`DecisionQuotient.InteriorVerification.agreeOnSet`                                                                     |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IV5`]{#lh:IV5}`DecisionQuotient.InteriorVerification.interiorParetoDominates`                                                        |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IV6`]{#lh:IV6}`DecisionQuotient.InteriorVerification.interior_certificate_implies_non_rejection`                                     |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IV7`]{#lh:IV7}`DecisionQuotient.InteriorVerification.interior_dominance_implies_universal_non_rejection`                             |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IV8`]{#lh:IV8}`DecisionQuotient.InteriorVerification.interior_dominance_not_full_sufficiency`                                        |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IV9`]{#lh:IV9}`DecisionQuotient.InteriorVerification.interior_verification_tractable_certificate`                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IV10`]{#lh:IV10}`DecisionQuotient.InteriorVerification.not_sufficientOnSet_of_counterexample`                                        |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`IV11`]{#lh:IV11}`DecisionQuotient.InteriorVerification.singleton_coordinate_interior_certificate`                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`LP1`]{#lh:LP1}`DecisionQuotient.Physics.LocalityPhysics.SpacetimePoint`                                                              |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`LP2`]{#lh:LP2}`DecisionQuotient.Physics.LocalityPhysics.lightCone`                                                                   |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`LP3`]{#lh:LP3}`DecisionQuotient.Physics.LocalityPhysics.causalPast`                                                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`LP4`]{#lh:LP4}`DecisionQuotient.Physics.LocalityPhysics.self_in_lightCone`                                                           |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`LP5`]{#lh:LP5}`DecisionQuotient.Physics.LocalityPhysics.self_in_causalPast`                                                          |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`LP6`]{#lh:LP6}`DecisionQuotient.Physics.LocalityPhysics.LocalRegion`                                                                 |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`LP7`]{#lh:LP7}`DecisionQuotient.Physics.LocalityPhysics.canObserve`                                                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`LP8`]{#lh:LP8}`DecisionQuotient.Physics.LocalityPhysics.spacelikeSeparated`                                                          |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`LP9`]{#lh:LP9}`DecisionQuotient.Physics.LocalityPhysics.spacelike_disjoint_observation`                                              |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`LP11`]{#lh:LP11}`DecisionQuotient.Physics.LocalityPhysics.LocalConfiguration`                                                        |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`LP12`]{#lh:LP12}`DecisionQuotient.Physics.LocalityPhysics.isLocallyValid`                                                            |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`LP13`]{#lh:LP13}`DecisionQuotient.Physics.LocalityPhysics.MergeResult`                                                               |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`LP14`]{#lh:LP14}`DecisionQuotient.Physics.LocalityPhysics.boardMerge`                                                                |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`LP15`]{#lh:LP15}`DecisionQuotient.Physics.LocalityPhysics.independent_configs_can_disagree`                                          |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`LP16`]{#lh:LP16}`DecisionQuotient.Physics.LocalityPhysics.merge_compatible_iff`                                                      |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`LP17`]{#lh:LP17}`DecisionQuotient.Physics.LocalityPhysics.merge_contradiction_iff`                                                   |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`LP18`]{#lh:LP18}`DecisionQuotient.Physics.LocalityPhysics.locality_implies_possible_contradiction`                                   |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`LP19`]{#lh:LP19}`DecisionQuotient.Physics.LocalityPhysics.Superposition`                                                             |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`LP20`]{#lh:LP20}`DecisionQuotient.Physics.LocalityPhysics.superposition_can_contain_contradictions`                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`LP21`]{#lh:LP21}`DecisionQuotient.Physics.LocalityPhysics.superposition_requires_separation`                                         |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`LP22`]{#lh:LP22}`DecisionQuotient.Physics.LocalityPhysics.bell_separation_is_real`                                                   |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`LP23`]{#lh:LP23}`DecisionQuotient.Physics.LocalityPhysics.measurement_is_merge_contradiction`                                        |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`LP24`]{#lh:LP24}`DecisionQuotient.Physics.LocalityPhysics.entanglement_is_shared_origin`                                             |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`LP31`]{#lh:LP31}`DecisionQuotient.Physics.LocalityPhysics.complete_knowledge_requires_all_queries`                                   |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`LP32`]{#lh:LP32}`DecisionQuotient.Physics.LocalityPhysics.finite_energy_constraint`                                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`LP33`]{#lh:LP33}`DecisionQuotient.Physics.LocalityPhysics.self_knowledge_impossible_if_insufficient_energy`                          |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`LP34`]{#lh:LP34}`DecisionQuotient.Physics.LocalityPhysics.bounded_energy_forces_locality`                                            |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`LP35`]{#lh:LP35}`DecisionQuotient.Physics.LocalityPhysics.locality_implies_independent_regions`                                      |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`LP36`]{#lh:LP36}`DecisionQuotient.Physics.LocalityPhysics.independent_regions_imply_possible_contradiction`                          |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`LP38`]{#lh:LP38}`DecisionQuotient.Physics.LocalityPhysics.pne_np_necessary_for_physics`                                              |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`LP39`]{#lh:LP39}`DecisionQuotient.Physics.LocalityPhysics.matter_exists_because_pne_np`                                              |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`LP40`]{#lh:LP40}`DecisionQuotient.Physics.LocalityPhysics.physics_is_the_game`                                                       |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`LP41`]{#lh:LP41}`DecisionQuotient.Physics.LocalityPhysics.without_positive_query_cost_no_bound`                                      |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`LP42`]{#lh:LP42}`DecisionQuotient.Physics.LocalityPhysics.without_nontrivial_states_no_disagreement`                                 |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`LP43`]{#lh:LP43}`DecisionQuotient.Physics.LocalityPhysics.without_separation_no_independence`                                        |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`LP44`]{#lh:LP44}`DecisionQuotient.Physics.LocalityPhysics.without_finite_capacity_no_gap`                                            |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`LP45`]{#lh:LP45}`DecisionQuotient.Physics.LocalityPhysics.all_premises_used`                                                         |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`LP46`]{#lh:LP46}`DecisionQuotient.Physics.LocalityPhysics.premises_necessary_and_sufficient`                                         |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`LP50`]{#lh:LP50}`DecisionQuotient.Physics.LocalityPhysics.shannon_value_is_intractability`                                           |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`LP51`]{#lh:LP51}`DecisionQuotient.Physics.LocalityPhysics.economic_value_requires_scarcity`                                          |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`LP52`]{#lh:LP52}`DecisionQuotient.Physics.LocalityPhysics.thermodynamic_value_requires_gradient`                                     |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`LP53`]{#lh:LP53}`DecisionQuotient.Physics.LocalityPhysics.voi_requires_uncertainty`                                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`LP54`]{#lh:LP54}`DecisionQuotient.Physics.LocalityPhysics.physics_requires_intractability`                                           |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`LP55`]{#lh:LP55}`DecisionQuotient.Physics.LocalityPhysics.value_is_intractability`                                                   |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`LP56`]{#lh:LP56}`DecisionQuotient.Physics.LocalityPhysics.observers_value_the_intractable`                                           |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`LP57`]{#lh:LP57}`DecisionQuotient.Physics.LocalityPhysics.finite_steps_finite_coverage`                                              |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`LP58`]{#lh:LP58}`DecisionQuotient.Physics.LocalityPhysics.counting_gap`                                                              |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`LP59`]{#lh:LP59}`DecisionQuotient.Physics.LocalityPhysics.time_is_counting`                                                          |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`LP60`]{#lh:LP60}`DecisionQuotient.Physics.LocalityPhysics.gap_equivalence`                                                           |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`LP61`]{#lh:LP61}`DecisionQuotient.Physics.LocalityPhysics.light_cone_is_time_gap`                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`MI1`]{#lh:MI1}`DecisionQuotient.ClaimClosure.MI1`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`MI2`]{#lh:MI2}`DecisionQuotient.ClaimClosure.MI2`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`MI3`]{#lh:MI3}`DecisionQuotient.ClaimClosure.MI3`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`MI4`]{#lh:MI4}`DecisionQuotient.ClaimClosure.MI4`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`MI5`]{#lh:MI5}`DecisionQuotient.ClaimClosure.MI5`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`MN1`]{#lh:MN1}`DecisionQuotient.Physics.MeasureNecessity.quantitative_claim_has_measure`                                             |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`MN2`]{#lh:MN2}`DecisionQuotient.Physics.MeasureNecessity.stochastic_claim_has_probability_measure`                                   |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`MN3`]{#lh:MN3}`DecisionQuotient.Physics.MeasureNecessity.stochastic_claim_has_measure`                                               |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`MN4`]{#lh:MN4}`DecisionQuotient.Physics.MeasureNecessity.count_univ_bool`                                                            |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`MN5`]{#lh:MN5}`DecisionQuotient.Physics.MeasureNecessity.counting_measure_not_probability_on_bool`                                   |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`MN6`]{#lh:MN6}`DecisionQuotient.Physics.MeasureNecessity.deterministic_dirac_is_probability`                                         |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`MN7`]{#lh:MN7}`DecisionQuotient.Physics.MeasureNecessity.quantitative_value_depends_on_measure`                                      |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`MN8`]{#lh:MN8}`DecisionQuotient.Physics.MeasureNecessity.deterministic_models_still_measure_based`                                   |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`MN9`]{#lh:MN9}`DecisionQuotient.Physics.MeasureNecessity.measure_does_not_imply_probability`                                         |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`MN10`]{#lh:MN10}`DecisionQuotient.Physics.MeasureNecessity.quantitative_measure_is_logical_prerequisite`                             |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`MN11`]{#lh:MN11}`DecisionQuotient.Physics.MeasureNecessity.stochastic_probability_is_logical_prerequisite`                           |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`OR1`]{#lh:OR1}`DecisionQuotient.Physics.ObserverRelativeState.ObserverClass`                                                         |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`OR2`]{#lh:OR2}`DecisionQuotient.Physics.ObserverRelativeState.obsEquiv`                                                              |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`OR3`]{#lh:OR3}`DecisionQuotient.Physics.ObserverRelativeState.EffectiveStateSpace`                                                   |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`OR4`]{#lh:OR4}`DecisionQuotient.Physics.ObserverRelativeState.project_eq_iff`                                                        |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`OR5`]{#lh:OR5}`DecisionQuotient.Physics.ObserverRelativeState.observer_relative_equivalence_witness`                                 |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`OR6`]{#lh:OR6}`DecisionQuotient.Physics.ObserverRelativeState.PhysicalObserverClass`                                                 |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`OR7`]{#lh:OR7}`DecisionQuotient.Physics.ObserverRelativeState.PhysicalStateSpace`                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`OR8`]{#lh:OR8}`DecisionQuotient.Physics.ObserverRelativeState.physical_state_space_has_instance_witness`                             |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`OR9`]{#lh:OR9}`DecisionQuotient.Physics.ObserverRelativeState.physical_observer_relative_effective_space`                            |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PA1`]{#lh:PA1}`DecisionQuotient.Physics.AnchorChecks.obsEquiv_all_of_effective_subsingleton`                                         |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PA2`]{#lh:PA2}`DecisionQuotient.Physics.AnchorChecks.stochasticAnchorSufficient_iff_exists_anchor_singleton`                         |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PA3`]{#lh:PA3}`DecisionQuotient.Physics.AnchorChecks.stochastic_anchor_check_iff_exists_anchor_singleton`                            |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PA4`]{#lh:PA4}`DecisionQuotient.Physics.AnchorChecks.stochastic_sufficient_of_observer_collapse_and_seed`                            |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PA5`]{#lh:PA5}`DecisionQuotient.Physics.AnchorChecks.stochastic_anchor_check_of_observer_collapse_and_seed`                          |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PA6`]{#lh:PA6}`DecisionQuotient.Physics.AnchorChecks.sequential_sufficient_of_observer_collapse`                                     |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PA7`]{#lh:PA7}`DecisionQuotient.Physics.AnchorChecks.sequential_anchor_check_of_observer_collapse`                                   |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PA8`]{#lh:PA8}`DecisionQuotient.Physics.AnchorChecks.physical_observer_collapse_implies_obsEquiv_all`                                |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PA9`]{#lh:PA9}`DecisionQuotient.Physics.AnchorChecks.physical_stochastic_anchor_check_of_observer_collapse_and_seed`                 |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PBC1`]{#lh:PBC1}`DecisionQuotient.PhysicalBudgetCrossover.CrossoverAt`                                                               |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PBC2`]{#lh:PBC2}`DecisionQuotient.PhysicalBudgetCrossover.SuccinctInfeasible`                                                        |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PBC3`]{#lh:PBC3}`DecisionQuotient.PhysicalBudgetCrossover.SuccinctUnbounded`                                                         |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PBC4`]{#lh:PBC4}`DecisionQuotient.PhysicalBudgetCrossover.explicit_infeasible_succinct_feasible_of_crossover`                        |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PBC5`]{#lh:PBC5}`DecisionQuotient.PhysicalBudgetCrossover.exists_least_crossover_point`                                              |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PBC6`]{#lh:PBC6}`DecisionQuotient.PhysicalBudgetCrossover.has_crossover_of_bounded_succinct_unbounded_explicit`                      |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PBC7`]{#lh:PBC7}`DecisionQuotient.PhysicalBudgetCrossover.explicit_eventual_infeasibility_of_monotone_and_witness`                   |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PBC8`]{#lh:PBC8}`DecisionQuotient.PhysicalBudgetCrossover.crossover_eventually_of_eventual_split`                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PBC9`]{#lh:PBC9}`DecisionQuotient.PhysicalBudgetCrossover.payoff_threshold_explicit_vs_succinct`                                     |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PBC10`]{#lh:PBC10}`DecisionQuotient.PhysicalBudgetCrossover.no_universal_survivor_without_succinct_bound`                            |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PBC11`]{#lh:PBC11}`DecisionQuotient.PhysicalBudgetCrossover.policy_closure_at_divergence`                                            |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PBC12`]{#lh:PBC12}`DecisionQuotient.PhysicalBudgetCrossover.policy_closure_beyond_divergence`                                        |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PH1`]{#lh:PH1}`PhysicalComplexity.k_Boltzmann`                                                                                       |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PH2`]{#lh:PH2}`PhysicalComplexity.PhysicalComputer`                                                                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PH3`]{#lh:PH3}`PhysicalComplexity.bit_energy_cost`                                                                                   |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PH4`]{#lh:PH4}`PhysicalComplexity.Landauer_bound`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PH5`]{#lh:PH5}`PhysicalComplexity.InstanceSize`                                                                                      |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PH6`]{#lh:PH6}`PhysicalComplexity.ComputationalRequirement`                                                                          |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PH7`]{#lh:PH7}`PhysicalComplexity.coNP_requirement`                                                                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PH8`]{#lh:PH8}`PhysicalComplexity.coNP_physically_impossible`                                                                        |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PH9`]{#lh:PH9}`PhysicalComplexity.coNP_not_in_P_physically`                                                                          |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PH10`]{#lh:PH10}`PhysicalComplexity.sufficiency_physically_impossible`                                                               |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PH11`]{#lh:PH11}`DecisionQuotient.PhysicalComplexity.PhysicalCollapseAtRequirement`                                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PH12`]{#lh:PH12}`DecisionQuotient.PhysicalComplexity.no_physical_collapse_at_requirement`                                            |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PH13`]{#lh:PH13}`DecisionQuotient.PhysicalComplexity.canonical_physical_collapse_impossible`                                         |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PH14`]{#lh:PH14}`DecisionQuotient.PhysicalComplexity.p_eq_np_physically_impossible_of_collapse_map`                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PH15`]{#lh:PH15}`DecisionQuotient.PhysicalComplexity.p_eq_np_physically_impossible_canonical`                                        |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PH16`]{#lh:PH16}`DecisionQuotient.PhysicalComplexity.P_eq_NP_via_SAT`                                                                |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PH17`]{#lh:PH17}`DecisionQuotient.PhysicalComplexity.SAT3ReductionBridge`                                                            |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PH18`]{#lh:PH18}`DecisionQuotient.PhysicalComplexity.sat_reduction_transfers_energy_lower_bound`                                     |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PH19`]{#lh:PH19}`DecisionQuotient.PhysicalComplexity.physical_collapse_of_polytime_sat_realization`                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PH20`]{#lh:PH20}`DecisionQuotient.PhysicalComplexity.p_eq_np_physically_impossible_via_sat_bridge`                                   |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PH21`]{#lh:PH21}`DecisionQuotient.PhysicalComplexity.SAT3HardFamily`                                                                 |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PH22`]{#lh:PH22}`DecisionQuotient.PhysicalComplexity.p_eq_np_physically_impossible_via_sat_hard_family`                              |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PH23`]{#lh:PH23}`DecisionQuotient.PhysicalComplexity.collapse_possible_without_positive_bit_cost`                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PH24`]{#lh:PH24}`DecisionQuotient.PhysicalComplexity.collapse_possible_without_exponential_lower_bound`                              |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PH25`]{#lh:PH25}`DecisionQuotient.PhysicalComplexity.no_go_transfer_requires_collapse_map`                                           |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PH26`]{#lh:PH26}`DecisionQuotient.PhysicalComplexity.no_collapse_of_bounded_budget_pos_cost_exp_lb`                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PH27`]{#lh:PH27}`DecisionQuotient.PhysicalComplexity.collapse_implies_assumption_failure_disjunction`                                |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PH28`]{#lh:PH28}`DecisionQuotient.PhysicalComplexity.deterministic_no_physical_collapse`                                             |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PH29`]{#lh:PH29}`DecisionQuotient.PhysicalComplexity.probabilistic_no_physical_collapse`                                             |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PH30`]{#lh:PH30}`DecisionQuotient.PhysicalComplexity.sequential_no_physical_collapse`                                                |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PH31`]{#lh:PH31}`DecisionQuotient.PhysicalComplexity.collapse_possible_with_unbounded_budget_profile`                                |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PH32`]{#lh:PH32}`DecisionQuotient.PhysicalComplexity.exp_budget_profile_unbounded`                                                   |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PH33`]{#lh:PH33}`DecisionQuotient.PhysicalComplexity.finite_budget_assumption_is_necessary`                                          |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PH34`]{#lh:PH34}`DecisionQuotient.PhysicalComplexity.CoherentDQRejectionAtRequirement`                                               |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PH35`]{#lh:PH35}`DecisionQuotient.PhysicalComplexity.coherent_dq_rejection_impossible_at_requirement`                                |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PH36`]{#lh:PH36}`DecisionQuotient.PhysicalComplexity.coherent_dq_rejection_impossible_canonical`                                     |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PI1`]{#lh:PI1}`DecisionQuotient.Physics.PhysicalIncompleteness.UniverseModel`                                                        |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PI2`]{#lh:PI2}`DecisionQuotient.Physics.PhysicalIncompleteness.PhysicallyInstantiated`                                               |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PI3`]{#lh:PI3}`DecisionQuotient.Physics.PhysicalIncompleteness.no_surjective_instantiation_of_card_gap`                              |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PI4`]{#lh:PI4}`DecisionQuotient.Physics.PhysicalIncompleteness.physical_incompleteness_of_card_gap`                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PI5`]{#lh:PI5}`DecisionQuotient.Physics.PhysicalIncompleteness.physical_incompleteness_of_bounds`                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PI6`]{#lh:PI6}`DecisionQuotient.Physics.PhysicalIncompleteness.under_resolution_implies_collision`                                   |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PI7`]{#lh:PI7}`DecisionQuotient.Physics.PhysicalIncompleteness.under_resolution_implies_decision_collision`                          |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PS1`]{#lh:PS1}`DecisionQuotient.Physics.ClaimTransport.PhysicalStateSemantics`                                                       |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PS2`]{#lh:PS2}`DecisionQuotient.Physics.ClaimTransport.physical_state_has_witness`                                                   |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PS3`]{#lh:PS3}`DecisionQuotient.Physics.ClaimTransport.physical_state_claim_of_instance_claim`                                       |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`PS4`]{#lh:PS4}`DecisionQuotient.Physics.ClaimTransport.physical_state_claim_of_universal_core`                                       |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`QT1`]{#lh:QT1}`DecisionQuotient.DecisionProblem.quotient_is_coarsest`                                                                |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`QT2`]{#lh:QT2}`DecisionQuotient.DecisionProblem.quotientMap_preservesOpt`                                                            |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`QT3`]{#lh:QT3}`DecisionQuotient.DecisionProblem.quotient_represents_opt_equiv`                                                       |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`QT4`]{#lh:QT4}`DecisionQuotient.DecisionProblem.factors_implies_respects`                                                            |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`QT6`]{#lh:QT6}`DecisionQuotient.DecisionProblem.quotientEquivOptRange_apply_quotientMap`                                             |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`QT7`]{#lh:QT7}`DecisionQuotient.DecisionProblem.quotient_has_unique_factorization`                                                   |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`RD1`]{#lh:RD1}`DecisionQuotient.Information.shannonEntropy_nonneg`                                                                   |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`RD2`]{#lh:RD2}`DecisionQuotient.Information.rate_zero_distortion`                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`RD3`]{#lh:RD3}`DecisionQuotient.Information.rate_monotone`                                                                           |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`RS1`]{#lh:RS1}`DecisionQuotient.Information.equiv_preserves_decision`                                                                |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`RS2`]{#lh:RS2}`DecisionQuotient.Information.rate_equals_srank`                                                                       |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`RS3`]{#lh:RS3}`DecisionQuotient.Information.compression_below_srank_fails`                                                           |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`RS4`]{#lh:RS4}`DecisionQuotient.Information.srank_bits_sufficient`                                                                   |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`RS5`]{#lh:RS5}`DecisionQuotient.Information.rate_distortion_bridge`                                                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`S2P1`]{#lh:S2P1}`DecisionQuotient.Sigma2PHardness.exactlyIdentifiesRelevant_iff_sufficient_and_subset_relevantFinset`                |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`S2P2`]{#lh:S2P2}`DecisionQuotient.Sigma2PHardness.min_representationGap_zero_iff_relevant_card`                                      |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`S2P3`]{#lh:S2P3}`DecisionQuotient.Sigma2PHardness.min_sufficient_set_iff_relevant_card`                                              |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`S2P4`]{#lh:S2P4}`DecisionQuotient.Sigma2PHardness.representationGap`                                                                 |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`S2P5`]{#lh:S2P5}`DecisionQuotient.Sigma2PHardness.representationGap_eq_waste_plus_missing`                                           |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`S2P6`]{#lh:S2P6}`DecisionQuotient.Sigma2PHardness.representationGap_eq_zero_iff`                                                     |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`S2P7`]{#lh:S2P7}`DecisionQuotient.Sigma2PHardness.representationGap_missing_eq_gapCard`                                              |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`S2P8`]{#lh:S2P8}`DecisionQuotient.Sigma2PHardness.representationGap_zero_iff_minimalSufficient`                                      |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`S2P9`]{#lh:S2P9}`DecisionQuotient.Sigma2PHardness.sufficient_iff_relevant_subset`                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`SE1`]{#lh:SE1}`DecisionQuotient.ClaimClosure.SE1`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`SE2`]{#lh:SE2}`DecisionQuotient.ClaimClosure.SE2`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`SE3`]{#lh:SE3}`DecisionQuotient.ClaimClosure.SE3`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`SE4`]{#lh:SE4}`DecisionQuotient.ClaimClosure.SE4`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`SE5`]{#lh:SE5}`DecisionQuotient.ClaimClosure.SE5`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`SE6`]{#lh:SE6}`DecisionQuotient.ClaimClosure.SE6`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`SK1`]{#lh:SK1}`DecisionQuotient.DecisionProblem.srank_eq_relevant_card`                                                              |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`SK2`]{#lh:SK2}`DecisionQuotient.DecisionProblem.srank_le_n`                                                                          |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`SK3`]{#lh:SK3}`DecisionQuotient.DecisionProblem.srank_zero_iff_constant`                                                             |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`SR1`]{#lh:SR1}`DecisionQuotient.ClaimClosure.SR1`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`SR2`]{#lh:SR2}`DecisionQuotient.ClaimClosure.SR2`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`SR3`]{#lh:SR3}`DecisionQuotient.ClaimClosure.SR3`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`SR4`]{#lh:SR4}`DecisionQuotient.ClaimClosure.SR4`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`SR5`]{#lh:SR5}`DecisionQuotient.ClaimClosure.SR5`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`TC1`]{#lh:TC1}`DecisionQuotient.ToolCollapse.WorkProfile`                                                                            |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`TC2`]{#lh:TC2}`DecisionQuotient.ToolCollapse.WorkModel`                                                                              |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`TC3`]{#lh:TC3}`DecisionQuotient.ToolCollapse.ToolModel`                                                                              |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`TC4`]{#lh:TC4}`DecisionQuotient.ToolCollapse.EventualStrictImprovement`                                                              |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`TC5`]{#lh:TC5}`DecisionQuotient.ToolCollapse.EffectiveCollapse`                                                                      |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`TC6`]{#lh:TC6}`DecisionQuotient.ToolCollapse.tool_never_worse`                                                                       |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`TC7`]{#lh:TC7}`DecisionQuotient.ToolCollapse.strict_improvement_has_witness`                                                         |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`TC8`]{#lh:TC8}`DecisionQuotient.ToolCollapse.effective_collapse_of_eventual_strict`                                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`TC9`]{#lh:TC9}`DecisionQuotient.ToolCollapse.expBaseline`                                                                            |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`TC10`]{#lh:TC10}`DecisionQuotient.ToolCollapse.linearTool`                                                                           |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`TC11`]{#lh:TC11}`DecisionQuotient.ToolCollapse.linear_tool_eventual_strict`                                                          |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`TC12`]{#lh:TC12}`DecisionQuotient.ToolCollapse.linear_tool_effective_collapse`                                                       |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`TL3`]{#lh:TL3}`DecisionQuotient.ThermodynamicLift.joulesPerBit_pos_of_landauer_floor`                                                |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`TL4`]{#lh:TL4}`DecisionQuotient.ThermodynamicLift.energy_lower_mandatory_of_landauer_floor`                                          |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`TL5`]{#lh:TL5}`DecisionQuotient.ThermodynamicLift.joulesPerBit_pos_of_landauer_calibration`                                          |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`TL6`]{#lh:TL6}`DecisionQuotient.ThermodynamicLift.energy_lower_mandatory_of_landauer_calibration`                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`TUR1`]{#lh:TUR1}`DecisionQuotient.Physics.transitionProb_nonneg`                                                                     |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`TUR2`]{#lh:TUR2}`DecisionQuotient.Physics.transitionProb_sum_one`                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`TUR5`]{#lh:TUR5}`DecisionQuotient.Physics.tur_bridge`                                                                                |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`TUR6`]{#lh:TUR6}`DecisionQuotient.Physics.multiple_futures_entropy_production`                                                       |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`UO1`]{#lh:UO1}`DecisionQuotient.UniverseDynamics`                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`UO2`]{#lh:UO2}`DecisionQuotient.feasibleActions`                                                                                     |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`UO3`]{#lh:UO3}`DecisionQuotient.lawDecisionProblem`                                                                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`UO4`]{#lh:UO4}`DecisionQuotient.lawUtility`                                                                                          |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`UO5`]{#lh:UO5}`DecisionQuotient.logicallyDeterministic`                                                                              |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`UO6`]{#lh:UO6}`DecisionQuotient.universe_sets_objective_schema`                                                                      |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`UO7`]{#lh:UO7}`DecisionQuotient.lawUtility_eq_of_allowed_iff`                                                                        |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`UO8`]{#lh:UO8}`DecisionQuotient.opt_eq_feasible_of_gap`                                                                              |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`UO9`]{#lh:UO9}`DecisionQuotient.infeasible_not_optimal_of_gap`                                                                       |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`UO10`]{#lh:UO10}`DecisionQuotient.opt_singleton_of_logicallyDeterministic`                                                           |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`UO11`]{#lh:UO11}`DecisionQuotient.opt_eq_of_allowed_iff`                                                                             |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`UQ1`]{#lh:UQ1}`DecisionQuotient.Physics.Uncertainty.binaryIdentityProblem`                                                           |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`UQ2`]{#lh:UQ2}`DecisionQuotient.Physics.Uncertainty.binaryIdentityProblem_opt_true`                                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`UQ3`]{#lh:UQ3}`DecisionQuotient.Physics.Uncertainty.binaryIdentityProblem_opt_false`                                                 |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`UQ4`]{#lh:UQ4}`DecisionQuotient.Physics.Uncertainty.exists_decision_problem_with_nontrivial_opt`                                     |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`UQ5`]{#lh:UQ5}`DecisionQuotient.Physics.Uncertainty.PhysicalNontrivialOptAssumption`                                                 |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`UQ6`]{#lh:UQ6}`DecisionQuotient.Physics.Uncertainty.exists_decision_problem_with_nontrivial_opt_of_physical`                         |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`W1`]{#lh:W1}`DecisionQuotient.Physics.single_future_zero_cost`                                                                       |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`W2`]{#lh:W2}`DecisionQuotient.Physics.transportCost_pos_of_offDiag`                                                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`W3`]{#lh:W3}`DecisionQuotient.Physics.integrity_is_centroid`                                                                         |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`W4`]{#lh:W4}`DecisionQuotient.Physics.wasserstein_bridge`                                                                            |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`WC1`]{#lh:WC1}`DecisionQuotient.Physics.WolpertConstraints.landauer_floor_plus_overhead_lower_bound`                                 |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`WC2`]{#lh:WC2}`DecisionQuotient.Physics.WolpertConstraints.effective_model_dominates_landauer_floor`                                 |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`WC3`]{#lh:WC3}`DecisionQuotient.Physics.WolpertConstraints.effective_model_strictly_exceeds_landauer_of_strict_overhead`             |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`WC4`]{#lh:WC4}`DecisionQuotient.Physics.WolpertConstraints.energy_lower_bound_mono_under_overhead`                                   |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`WC5`]{#lh:WC5}`DecisionQuotient.Physics.WolpertConstraints.physical_grounding_bundle_with_wolpert_overhead`                          |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`WD1`]{#lh:WD1}`DecisionQuotient.checking_witnessing_duality_budget`                                                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`WD2`]{#lh:WD2}`DecisionQuotient.no_sound_checker_below_witness_budget`                                                               |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`WD3`]{#lh:WD3}`DecisionQuotient.checking_time_ge_witness_budget`                                                                     |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`WD4`]{#lh:WD4}`DecisionQuotient.witnessBudgetEmpty`                                                                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`WD5`]{#lh:WD5}`DecisionQuotient.checkingBudgetPairs`                                                                                 |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`WM1`]{#lh:WM1}`DecisionQuotient.Physics.WolpertMismatch.mismatchKL_nonneg`                                                           |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`WM2`]{#lh:WM2}`DecisionQuotient.Physics.WolpertMismatch.mismatchKL_eq_zero_iff_eq`                                                   |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`WM3`]{#lh:WM3}`DecisionQuotient.Physics.WolpertMismatch.mismatchKL_pos_of_exists_ne`                                                 |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`WM4`]{#lh:WM4}`DecisionQuotient.Physics.WolpertMismatch.mismatchNatLowerBound_pos_of_exists_ne`                                      |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`WM5`]{#lh:WM5}`DecisionQuotient.Physics.WolpertDecomposition.periodic_modular_mismatch_of_distribution_mismatch`                     |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`WM6`]{#lh:WM6}`DecisionQuotient.Physics.WolpertDecomposition.effective_model_strictly_exceeds_landauer_of_distribution_mismatch`     |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`WP1`]{#lh:WP1}`DecisionQuotient.Physics.WolpertDecomposition.DecomposedProcessModel.totalOverheadPerBit_eq_sum`                      |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`WP2`]{#lh:WP2}`DecisionQuotient.Physics.WolpertDecomposition.landauer_floor_plus_decomposition_lower_bound`                          |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`WP3`]{#lh:WP3}`DecisionQuotient.Physics.WolpertDecomposition.effective_model_dominates_landauer_floor_decomposition`                 |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`WP4`]{#lh:WP4}`DecisionQuotient.Physics.WolpertDecomposition.effective_model_strictly_exceeds_landauer_of_periodic_modular_mismatch` |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`WP5`]{#lh:WP5}`DecisionQuotient.Physics.WolpertDecomposition.effective_model_strictly_exceeds_landauer_of_stopping_time_residual`    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`WP6`]{#lh:WP6}`DecisionQuotient.Physics.WolpertDecomposition.effective_model_strictly_exceeds_landauer_of_either_cited_component`    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`WP7`]{#lh:WP7}`DecisionQuotient.Physics.WolpertDecomposition.landauer_floor_plus_structural_resource_lower_bound`                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`WP8`]{#lh:WP8}`DecisionQuotient.Physics.WolpertDecomposition.energy_lower_bound_increases_by_structural_resource`                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`WP9`]{#lh:WP9}`DecisionQuotient.Physics.WolpertDecomposition.physical_grounding_bundle_with_wolpert_decomposition`                   |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`WR1`]{#lh:WR1}`DecisionQuotient.Physics.WolpertResidual.pairwiseResidualKL_nonneg`                                                   |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`WR2`]{#lh:WR2}`DecisionQuotient.Physics.WolpertResidual.pairwiseResidualKL_pos_of_asymmetry`                                         |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`WR3`]{#lh:WR3}`DecisionQuotient.Physics.WolpertResidual.residualNatLowerBound_pos_of_asymmetry`                                      |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`WR4`]{#lh:WR4}`DecisionQuotient.Physics.WolpertDecomposition.stopping_time_residual_of_pairwise_flow_asymmetry`                      |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`WR5`]{#lh:WR5}`DecisionQuotient.Physics.WolpertDecomposition.effective_model_strictly_exceeds_landauer_of_pairwise_flow_asymmetry`   |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`WR6`]{#lh:WR6}`DecisionQuotient.Physics.WolpertResidual.discreteResidualNatLowerBound_pos_of_asymmetry_or_oneway`                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`WR7`]{#lh:WR7}`DecisionQuotient.Physics.WolpertDecomposition.stopping_time_residual_of_discrete_edge_split`                          |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`WR8`]{#lh:WR8}`DecisionQuotient.Physics.WolpertDecomposition.effective_model_strictly_exceeds_landauer_of_discrete_edge_split`       |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`WR9`]{#lh:WR9}`DecisionQuotient.Physics.WolpertDecomposition.stopping_time_residual_of_finite_discrete_witness`                      |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`WR10`]{#lh:WR10}`DecisionQuotient.Physics.WolpertDecomposition.effective_model_strictly_exceeds_landauer_of_finite_discrete_witness` |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`XC1`]{#lh:XC1}`DecisionQuotient.Physics.srank_determines_states`                                                                     |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`XC2`]{#lh:XC2}`DecisionQuotient.Physics.more_states_more_transport`                                                                  |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`XC3`]{#lh:XC3}`DecisionQuotient.Physics.transport_lower_bound`                                                                       |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`XC4`]{#lh:XC4}`DecisionQuotient.Physics.transport_independent_of_energy`                                                             |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`XC5`]{#lh:XC5}`DecisionQuotient.Physics.transport_independent_of_precision`                                                          |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`XC6`]{#lh:XC6}`DecisionQuotient.Physics.srank_unified_complexity`                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------+
| [`XC7`]{#lh:XC7}`DecisionQuotient.Physics.complete_bridge_set`                                                                         |
+----------------------------------------------------------------------------------------------------------------------------------------+


## Assumption Ledger (Auto)

#### Source files.

-   `DecisionQuotient/ClaimClosure.lean`

-   `DecisionQuotient/HandleAliases.lean`

-   `DecisionQuotient/StochasticSequential/ClaimClosure.lean`

-   `DecisionQuotient/StochasticSequential/Hierarchy.lean`

#### Assumption bundles and fields.

-   (No assumption bundles parsed.)

#### Conditional closure handles.

-   `DQ.anchor_sigma2p_complete_conditional`

-   `DQ.cost_asymmetry_eth_conditional`

-   `DQ.dichotomy_conditional`

-   `DQ.minsuff_collapse_to_conp_conditional`

-   `DQ.minsuff_conp_complete_conditional`

-   `DQ.sufficiency_conp_complete_conditional`

-   `DQ.tractable_subcases_conditional`

#### First-principles Bayes derivation handles.

-   `DQ.BayesOptimalityProof.KL_eq_sum_klFun`

-   `DQ.BayesOptimalityProof.KL_eq_zero_iff_eq`

-   `DQ.BayesOptimalityProof.KL_nonneg`

-   `DQ.BayesOptimalityProof.bayes_is_optimal`

-   `DQ.BayesOptimalityProof.crossEntropy_eq_entropy_add_KL`

-   `DQ.BayesOptimalityProof.entropy_le_crossEntropy`

-   `DQ.Foundations.bayes_from_conditional`

-   `DQ.Foundations.complete_chain`

-   `DQ.Foundations.counting_additive`

-   `DQ.Foundations.counting_nonneg`

-   `DQ.Foundations.counting_total`

-   `DQ.Foundations.dq_equivalence`

-   `DQ.Foundations.dq_in_unit_interval`

-   `DQ.Foundations.dq_one_iff_perfect_info`

-   `DQ.Foundations.dq_zero_iff_no_info`

-   `DQ.Foundations.entropy_contraction`

-   `DQ.Foundations.mutual_info_nonneg`

-   `DQ.bayesian_dq_matches_physics_dq`

-   `DQ.chain_rule`

-   `DQ.dq_derived_from_bayes`

-   `DQ.dq_is_bayesian_certainty_fraction`

#### Compact Bayes handle IDs.

-   `FN7` $\to$ `DQ.BayesOptimalityProof.KL_nonneg`

-   `FN8` $\to$ `DQ.BayesOptimalityProof.entropy_le_crossEntropy`

-   `FN12` $\to$ `DQ.BayesOptimalityProof.crossEntropy_eq_entropy_add_KL`

-   `FN14` $\to$ `DQ.BayesOptimalityProof.bayes_is_optimal`

-   `FN15` $\to$ `DQ.BayesOptimalityProof.KL_eq_sum_klFun`

-   `FN16` $\to$ `DQ.BayesOptimalityProof.KL_eq_zero_iff_eq`

-   `BB1` $\to$ `DQ.BayesianDQ`

-   `BB2` $\to$ `DQ.BayesianDQ.certaintyGain`

-   `BB3` $\to$ `DQ.dq_is_bayesian_certainty_fraction`

-   `BB4` $\to$ `DQ.bayesian_dq_matches_physics_dq`

-   `BB5` $\to$ `DQ.dq_derived_from_bayes`

-   `BF1` $\to$ `DQ.certainty_of_not_nondegenerateBelief`

-   `BF2` $\to$ `DQ.nondegenerateBelief_of_uncertaintyForced`

-   `BF3` $\to$ `DQ.forced_action_under_uncertainty`

-   `BF4` $\to$ `DQ.bayes_update_exists_of_nondegenerateBelief`


  -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **Paper handle**                                **Status**   **Lean support**
  ----------------------------------------------- ------------ ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  `cor:channel-degradation`                       Full         `DQ.ClaimClosure.CH2`, `DQ.ClaimClosure.CH6`

  `cor:ego-trap`                                  Unmapped     *(no derived Lean handle found)*

  `cor:exact-identifiability`                     Full         `DQ.Sigma2PHardness.exactlyIdentifiesRelevant_iff_sufficient_and_subset_relevantFinset`

  `cor:exact-no-competence-zero-certified`        Full         `DQ.IntegrityCompetence.rlff_maximizer_has_evidence`

  `cor:finite-budget-no-exact-admissibility`      Full         `DQ.PhysicalBudgetCrossover.crossover_eventually_of_eventual_split`, `DQ.PhysicalBudgetCrossover.explicit_eventual_infeasibility_of_monotone_and_witness`

  `cor:finite-budget-threshold-impossibility`     Full         `DQ.PhysicalBudgetCrossover.crossover_eventually_of_eventual_split`, `DQ.PhysicalBudgetCrossover.explicit_eventual_infeasibility_of_monotone_and_witness`

  `cor:finite-lifetime`                           Full         `DQ.ClaimClosure.SE5`

  `cor:finite-state`                              Full         `DQ.Physics.BoundedAcquisition.acquisition_rate_bound`, `DQ.Physics.BoundedAcquisition.one_bit_per_transition`

  `cor:forced-finite-speed`                       Full         `DQ.Physics.BoundedAcquisition.counting_gap_theorem`

  `cor:gap-externalization`                       Full         `DQ.HardnessDistribution.simplicityTax_grows`, `DQ.HardnessDistribution.totalExternalWork_eq_n_mul_gapCard`

  `cor:gap-minimization-hard`                     Full         `DQ.Sigma2PHardness.min_representationGap_zero_iff_relevant_card`

  `cor:generalized-eventual-dominance`            Full         `DQ.HardnessDistribution.generalized_right_eventually_dominates_wrong`

  `cor:ground-state`                              Full         `DQ.Physics.BoundedAcquisition.srank_one_energy_minimum`

  `cor:ground-state-passive`                      Full         `DQ.ClaimClosure.AtomicCircuitExports.AC6`

  `cor:hardness-exact-certainty-inflation`        Full         `DQ.ClaimClosure.epsilon_admissible_iff_raw_lt_certified_total_core`

  `cor:hardness-raw-only-exact`                   Full         `DQ.ClaimClosure.exact_certainty_inflation_under_hardness_core`, `DQ.IntegrityCompetence.competence_implies_integrity`

  `cor:import-asymmetry`                          Full         `DQ.ClaimClosure.SR4`, `DQ.ClaimClosure.SR5`

  `cor:information-barrier-query`                 Full         `DQ.ClaimClosure.horizonTwoWitness_immediate_empty_sufficient`, `DQ.ClaimClosure.horizon_gt_one_bridge_can_fail_on_sufficiency`, `DQ.ClaimClosure.information_barrier_opt_oracle_core`

  `cor:integrity-universal`                       Full         `DQ.ClaimClosure.information_barrier_value_entry_core`

  `cor:interior-singleton-certificate`            Full         `DQ.InteriorVerification.interior_verification_tractable_certificate`

  `cor:linear-positive-no-saturation`             Full         `DQ.HardnessDistribution.no_positive_slope_linear_represents_saturating`

  `cor:neukart-vinokur`                           Full         `DQ.ClaimClosure.process_bridge_failure_witness`, `DQ.ClaimClosure.selectorSufficient_not_implies_setSufficient`

  `cor:no-auto-minimize`                          Full         `DQ.ClaimClosure.minsuff_collapse_to_conp_conditional`

  `cor:no-free-computation`                       Full         `DQ.ClaimClosure.anchor_query_relation_true_iff`, `DQ.ClaimClosure.anchor_sigma2p_reduction_core`, `DQ.ClaimClosure.boundaryCharacterized_iff_exists_sufficient_subset`

  `cor:no-uncertified-exact-claim`                Full         `DQ.ClaimClosure.no_exact_claim_under_declared_assumptions_unless_excused_core`

  `cor:no-universal-survivor-no-succinct-bound`   Full         `DQ.PhysicalBudgetCrossover.no_universal_survivor_without_succinct_bound`

  `cor:outside-excuses-no-exact-report`           Full         `DQ.ClaimClosure.no_auto_minimize_of_p_neq_conp`

  `cor:overmodel-diagnostic-implication`          Full         `DQ.Sigma2PHardness.representationGap_eq_zero_iff`

  `cor:phase-transition`                          Full         `DQ.ClaimClosure.IE12`, `DQ.ClaimClosure.IE13`

  `cor:physical-counterexample-core-failure`      Full         `DQ.Physics.ClaimTransport.no_physical_counterexample_of_core_theorem`, `DQ.Physics.ClaimTransport.physical_counterexample_invalidates_core_rule`, `DQ.Physics.ClaimTransport.physical_counterexample_yields_core_counterexample`

  `cor:physics-no-universal-exact-claim`          Full         `DQ.ClaimClosure.minsuff_conp_complete_conditional`

  `cor:posed-anchor-certified-gate`               Full         `DQ.ClaimClosure.AQ8`

  `cor:practice-bounded`                          Full         `DQ.ClaimClosure.tractable_bounded_core`

  `cor:practice-diagnostic`                       Full         `DQ.Sigma2PHardness.min_representationGap_zero_iff_relevant_card`

  `cor:practice-structured`                       Full         `DQ.ClaimClosure.tractable_separable_core`

  `cor:practice-symmetry`                         Full         `DQ.ClaimClosure.sufficiency_conp_complete_conditional`

  `cor:practice-tensor`                           Full         `DQ.ClaimClosure.subproblem_transfer_as_regime_simulation`

  `cor:practice-tree`                             Full         `DQ.ClaimClosure.tractable_tree_core`

  `cor:practice-treewidth`                        Full         `DQ.ClaimClosure.sufficiency_conp_reduction_core`, `DQ.ClaimClosure.sufficiency_iff_projectedOptCover_eq_opt`

  `cor:practice-unstructured`                     Full         `DQ.ClaimClosure.hard_family_all_coords_core`

  `cor:query-obstruction-bool`                    Full         `DQ.ClaimClosure.physical_crossover_policy_core`, `DQ.ClaimClosure.process_bridge_failure_witness`

  `cor:right-wrong-hardness`                      Full         `DQ.HardnessDistribution.right_dominates_wrong`

  `cor:rlff-abstain-no-certs`                     Full         `DQ.IntegrityCompetence.exactCertaintyInflation_iff_no_exact_competence`, `DQ.IntegrityCompetence.rlff_abstain_strictly_prefers_no_certificates`

  `cor:speed-integrity`                           Full         `DQ.ClaimClosure.SE6`

  `cor:theorem-equilibrium`                       Full         `DQ.ClaimClosure.IE3`, `DQ.ClaimClosure.IE4`, `DQ.ClaimClosure.IE5`

  `cor:thermo-dq`                                 Full         `DQ.ClaimClosure.DQ7`, `DQ.ClaimClosure.DQ8`

  `cor:type-system-threshold`                     Full         `DQ.HardnessDistribution.native_dominates_manual`

  `cor:zero-capacity`                             Full         `DQ.ClaimClosure.CH3`

  `cor:zero-gap`                                  Full         `DQ.ClaimClosure.GE4`, `DQ.ClaimClosure.GE5`

  `prop:abstain-guess-self-signal`                Full         `DQ.IntegrityCompetence.signal_no_evidence_forces_zero_certified`

  `prop:abstention-frontier`                      Unmapped     *(no derived Lean handle found)*

  `prop:adq-ordering`                             Full         `DQ.ClaimClosure.adq_ordering`

  `prop:attempted-competence-matrix`              Full         `DQ.IntegrityCompetence.RLFFWeights`, `DQ.IntegrityCompetence.ReportSignal`, `DQ.IntegrityCompetence.evidence_of_claim_admissible`

  `prop:bounded-region`                           Full         `DQ.Physics.BoundedAcquisition.BoundedRegion`

  `prop:bounded-slice-meta-irrelevance`           Full         `DQ.ClaimClosure.integrity_resource_bound_for_sufficiency`, `DQ.ClaimClosure.integrity_universal_applicability_core`

  `prop:bridge-failure-horizon`                   Full         `DQ.ClaimClosure.explicit_state_upper_core`, `DQ.ClaimClosure.hard_family_all_coords_core`

  `prop:bridge-failure-stochastic`                Full         `DQ.ClaimClosure.posed_anchor_query_truth_iff_exists_anchor`

  `prop:bridge-failure-transition`                Full         `DQ.ClaimClosure.sufficiency_conp_reduction_core`

  `prop:bridge-transfer-scope`                    Full         `DQ.ClaimClosure.no_exact_identifier_implies_not_boundary_characterized`

  `prop:budgeted-crossover`                       Full         `DQ.ClaimClosure.oracle_lattice_transfer_as_regime_simulation`, `DQ.PhysicalBudgetCrossover.CrossoverAt`

  `prop:certainty-inflation-iff-inadmissible`     Full         `DQ.IntegrityCompetence.ReportBitModel`, `DQ.IntegrityCompetence.claim_admissible_of_evidence`

  `prop:certified-confidence-gate`                Full         `DQ.IntegrityCompetence.rlffBaseReward`, `DQ.IntegrityCompetence.rlffReward`

  `prop:checking-witnessing-duality`              Full         `DQ.checkingBudgetPairs`, `DQ.checking_time_ge_witness_budget`, `DQ.witnessBudgetEmpty`

  `prop:comp-thermo-chain`                        Full         `DQ.ClaimClosure.DS5`, `DQ.ClaimClosure.DS6`

  `prop:crossover-above-cap`                      Full         `DQ.ClaimClosure.one_step_bridge`, `DQ.PhysicalBudgetCrossover.SuccinctInfeasible`

  `prop:crossover-not-certification`              Full         `DQ.ClaimClosure.physical_crossover_above_cap_core`

  `prop:crossover-policy`                         Full         `DQ.ClaimClosure.physical_crossover_core`

  `prop:decision-equivalence`                     Full         `DQ.ClaimClosure.DE1`, `DQ.ClaimClosure.DE2`, `DQ.ClaimClosure.DE3`, `DQ.ClaimClosure.DE4`

  `prop:decision-unit-time`                       Full         `DQ.Physics.DecisionTime.decision_event_iff_eq_tick`, `DQ.Physics.DecisionTime.decision_event_implies_time_unit`, `DQ.Physics.DecisionTime.decision_taking_place_is_unit_of_time`, `DQ.Physics.DecisionTime.tick_is_decision_event`

  `prop:declared-contract-selection-validity`     Full         `DQ.ClaimClosure.exact_raw_eq_certified_iff_certainty_inflation_core`, `DQ.ClaimClosure.no_auto_minimize_of_p_neq_conp`, `DQ.ClaimClosure.sufficiency_iff_projectedOptCover_eq_opt`

  `prop:discrete-state-time`                      Full         `DQ.ClaimClosure.DS1`, `DQ.ClaimClosure.DS2`

  `prop:dominance-modes`                          Full         `DQ.HardnessDistribution.centralized_higher_leverage`

  `prop:empty-sufficient-constant`                Full         `DQ.ClaimClosure.DP6`

  `prop:eventual-explicit-infeasibility`          Full         `DQ.PhysicalBudgetCrossover.explicit_eventual_infeasibility_of_monotone_and_witness`

  `prop:evidence-admissibility-equivalence`       Full         `DQ.IntegrityCompetence.certifiedTotalBits`, `DQ.IntegrityCompetence.certifiedTotalBits_of_evidence`, `DQ.IntegrityCompetence.certifiedTotalBits_of_no_evidence`

  `prop:exact-requires-evidence`                  Full         `DQ.IntegrityCompetence.no_completion_fraction_without_declared_bound`, `DQ.IntegrityCompetence.overModelVerdict_rational_iff`

  `prop:fraction-defined-under-bound`             Full         `DQ.IntegrityCompetence.signal_consistent_of_claim_admissible`

  `prop:generalized-assumption-boundary`          Full         `DQ.HardnessDistribution.generalized_dominance_can_fail_without_right_boundedness`, `DQ.HardnessDistribution.generalized_dominance_can_fail_without_wrong_growth`

  `prop:hardness-conservation`                    Full         `DQ.HardnessDistribution.hardness_is_irreducible_required_work`, `DQ.HardnessDistribution.totalDOF_ge_intrinsic`

  `prop:hardness-efficiency-interpretation`       Full         `DQ.HardnessDistribution.hardnessEfficiency_eq_central_share`

  `prop:heisenberg-strong-nontrivial-opt`         Full         `DQ.Physics.HeisenbergStrong.strong_binding_implies_core_nontrivial`, `DQ.Physics.HeisenbergStrong.strong_binding_implies_nontrivial_opt_via_uncertainty`, `DQ.Physics.HeisenbergStrong.strong_binding_implies_physical_nontrivial_opt_assumption`

  `prop:heuristic-reusability`                    Full         `DQ.ClaimClosure.anchor_query_relation_true_iff`, `DQ.ClaimClosure.posed_anchor_checked_true_implies_truth`, `DQ.ClaimClosure.posed_anchor_exact_claim_requires_evidence`, `DQ.ClaimClosure.sufficiency_iff_dq_ratio`

  `prop:identifiability-convergence`              Unmapped     *(no derived Lean handle found)*

  `prop:insufficiency-counterexample`             Full         `DQ.ClaimClosure.DP7`, `DQ.ClaimClosure.DP8`

  `prop:integrity-competence-separation`          Full         `DQ.IntegrityCompetence.certifiedTotalBits_ge_raw`, `DQ.IntegrityCompetence.epsilon_competence_implies_integrity`

  `prop:integrity-prerequisite`                   Full         `DQ.ClaimClosure.IE17`

  `prop:integrity-resource-bound`                 Full         `DQ.ClaimClosure.information_barrier_state_batch_core`, `DQ.IntegrityCompetence.completion_fraction_defined_of_declared_bound`, `DQ.IntegrityCompetence.evidence_nonempty_iff_claim_admissible`

  `prop:interior-one-sidedness`                   Full         `DQ.InteriorVerification.interior_certificate_implies_non_rejection`, `DQ.InteriorVerification.interior_dominance_not_full_sufficiency`

  `prop:interior-universal-non-rejection`         Full         `DQ.InteriorVerification.interiorParetoDominates`

  `prop:interior-verification-tractable`          Full         `DQ.InteriorVerification.agreeOnSet`, `DQ.InteriorVerification.interior_dominance_implies_universal_non_rejection`

  `prop:landauer-constraint`                      Full         `DQ.ClaimClosure.IE1`, `DQ.ClaimClosure.IE6`

  `prop:law-instance-objective-bridge`            Full         `DQ.Physics.ClaimTransport.PhysicalEncoding`, `DQ.Physics.ClaimTransport.physical_claim_lifts_from_core`

  `prop:least-divergence-point`                   Full         `DQ.PhysicalBudgetCrossover.exists_least_crossover_point`

  `prop:lorentz-discrete`                         Full         `DQ.ClaimClosure.DS3`, `DQ.ClaimClosure.DS4`

  `prop:mdp-tractable`                            Unmapped     *(no derived Lean handle found)*

  `prop:minimal-relevant-equiv`                   Full         `DQ.DecisionProblem.relevantSet_is_minimal`, `DQ.DecisionProblem.sufficient_implies_selectorSufficient`

  `prop:no-evidence-zero-certified`               Full         `DQ.IntegrityCompetence.rlff_abstain_strictly_prefers_no_certificates`

  `prop:no-fraction-without-bound`                Full         `DQ.IntegrityCompetence.signal_certified_positive_implies_admissible`

  `prop:one-step-bridge`                          Full         `DQ.ClaimClosure.no_exact_identifier_implies_not_boundary_characterized`

  `prop:optimizer-coimage`                        Full         `DQ.DecisionProblem.quotient_has_unique_factorization`

  `prop:optimizer-entropy-image`                  Full         `DQ.DecisionProblem.numOptClasses`

  `prop:oracle-lattice-strict`                    Full         `DQ.ClaimClosure.horizonTwoWitness_immediate_empty_sufficient`, `DQ.ClaimClosure.information_barrier_opt_oracle_core`

  `prop:oracle-lattice-transfer`                  Full         `DQ.ClaimClosure.no_uncertified_exact_claim_core`, `DQ.ClaimClosure.pose_returns_anchor_query_object`

  `prop:orbital-symmetry`                         Full         `DQ.ClaimClosure.AtomicCircuitExports.AC8`

  `prop:outside-excuses-explicit-assumptions`     Full         `DQ.ClaimClosure.exact_raw_eq_certified_iff_certainty_inflation_core`

  `prop:payoff-threshold`                         Full         `DQ.PhysicalBudgetCrossover.crossover_eventually_of_eventual_split`, `DQ.PhysicalBudgetCrossover.payoff_threshold_explicit_vs_succinct`

  `prop:physical-claim-transport`                 Full         `DQ.Physics.ClaimTransport.no_physical_counterexample_of_core_theorem`, `DQ.Physics.ClaimTransport.physical_claim_lifts_from_core_conditional`, `DQ.Physics.ClaimTransport.physical_counterexample_invalidates_core_rule`

  `prop:physics-no-universal-exact`               Full         `DQ.ClaimClosure.declaredBudgetSlice`

  `prop:policy-closure-beyond-divergence`         Full         `DQ.PhysicalBudgetCrossover.policy_closure_at_divergence`, `DQ.PhysicalBudgetCrossover.policy_closure_beyond_divergence`

  `prop:pose-anchor-object`                       Full         `DQ.ClaimClosure.AQ1`, `DQ.ClaimClosure.AQ2`, `DQ.ClaimClosure.AQ3`

  `prop:posed-anchor-typed-exact`                 Full         `DQ.ClaimClosure.AQ4`, `DQ.ClaimClosure.AQ5`, `DQ.ClaimClosure.AQ6`, `DQ.ClaimClosure.AQ7`

  `prop:query-finite-state-generalization`        Full         `DQ.ClaimClosure.process_bridge_failure_witness`

  `prop:query-randomized-robustness`              Full         `DQ.ClaimClosure.process_bridge_failure_witness`

  `prop:query-randomized-weighted`                Full         `DQ.ClaimClosure.process_bridge_failure_witness`

  `prop:query-regime-obstruction`                 Full         `DQ.ClaimClosure.process_bridge_failure_witness`

  `prop:query-state-batch-lb`                     Full         `DQ.ClaimClosure.horizon_gt_one_bridge_can_fail_on_sufficiency`, `DQ.ClaimClosure.process_bridge_failure_witness`

  `prop:query-subproblem-transfer`                Full         `DQ.ClaimClosure.pose_returns_anchor_query_object`, `DQ.ClaimClosure.posed_anchor_query_truth_iff_exists_forall`, `DQ.ClaimClosure.posed_anchor_signal_positive_certified_implies_admissible`

  `prop:query-tightness-full-scan`                Full         `DQ.ClaimClosure.process_bridge_failure_witness`

  `prop:query-value-entry-lb`                     Full         `DQ.ClaimClosure.information_barrier_opt_oracle_core`, `DQ.ClaimClosure.process_bridge_failure_witness`

  `prop:query-weighted-transfer`                  Full         `DQ.ClaimClosure.process_bridge_failure_witness`

  `prop:raw-certified-bit-split`                  Full         `DQ.ClaimClosure.bridge_failure_witness_non_one_step`, `DQ.IntegrityCompetence.admissible_irrational_strictly_more_than_rational`, `DQ.IntegrityCompetence.admissible_matrix_counts`, `DQ.IntegrityCompetence.certaintyInflation_iff_not_admissible`, `DQ.IntegrityCompetence.certificationOverheadBits`, `DQ.IntegrityCompetence.certificationOverheadBits_of_evidence`, `DQ.IntegrityCompetence.certificationOverheadBits_of_no_evidence`, `DQ.IntegrityCompetence.certifiedTotalBits_ge_raw`, `DQ.IntegrityCompetence.certifiedTotalBits_gt_raw_of_evidence`

  `prop:reaction-competence`                      Full         `DQ.ClaimClosure.MI3`, `DQ.ClaimClosure.MI4`

  `prop:refinement-strengthens`                   Unmapped     *(no derived Lean handle found)*

  `prop:retraction-evidence-integrity`            Unmapped     *(no derived Lean handle found)*

  `prop:retraction-no-evidence-violates`          Unmapped     *(no derived Lean handle found)*

  `prop:rlff-maximizer-admissible`                Full         `DQ.IntegrityCompetence.exact_raw_only_of_no_exact_admissible`, `DQ.IntegrityCompetence.integrity_forces_abstention`

  `prop:run-time-accounting`                      Full         `DQ.Physics.DecisionTime.decisionTrace_length_eq_ticks`, `DQ.Physics.DecisionTime.decision_count_equals_elapsed_time`, `DQ.Physics.DecisionTime.run_elapsed_time_eq_ticks`, `DQ.Physics.DecisionTime.run_time_exact`

  `prop:selector-separation`                      Full         `DQ.ClaimClosure.posed_anchor_exact_claim_admissible_iff_competent`

  `prop:self-confidence-not-certification`        Full         `DQ.IntegrityCompetence.signal_exact_no_competence_forces_zero_certified`

  `prop:sequential-anchor-refinement`             Unmapped     *(no derived Lean handle found)*

  `prop:sequential-anchor-tqbf-reduction`         Unmapped     *(no derived Lean handle found)*

  `prop:sequential-bounded-horizon`               Unmapped     *(no derived Lean handle found)*

  `prop:sequential-static-relation`               Unmapped     *(no derived Lean handle found)*

  `prop:set-to-selector`                          Full         `DQ.ClaimClosure.DecisionProblem.sufficient_iff_zeroEpsilonSufficient`

  `prop:snapshot-process-typing`                  Full         `DQ.ClaimClosure.physical_crossover_hardness_core`, `DQ.ClaimClosure.posed_anchor_no_competence_no_exact_claim`, `DQ.ClaimClosure.system_transfer_licensed_iff_snapshot`

  `prop:srank-support`                            Full         `DQ.DecisionProblem.srank_eq_relevant_card`, `DQ.DecisionProblem.srank_le_n`, `DQ.DecisionProblem.srank_zero_iff_constant`

  `prop:static-stochastic-strict`                 Unmapped     *(no derived Lean handle found)*

  `prop:static-stochastic-transfer`               Unmapped     *(no derived Lean handle found)*

  `prop:steps-run-scalar`                         Full         `DQ.IntegrityCompetence.rlff_maximizer_is_admissible`, `DQ.IntegrityCompetence.self_reflected_confidence_not_certification`

  `prop:stochastic-anchor-refinement`             Unmapped     *(no derived Lean handle found)*

  `prop:stochastic-anchor-strict-reduction`       Unmapped     *(no derived Lean handle found)*

  `prop:stochastic-bounded-support`               Unmapped     *(no derived Lean handle found)*

  `prop:stochastic-landauer-floor`                Unmapped     *(no derived Lean handle found)*

  `prop:stochastic-potential-duality`             Unmapped     *(no derived Lean handle found)*

  `prop:stochastic-product-tractable`             Unmapped     *(no derived Lean handle found)*

  `prop:stochastic-sequential-bridge-fail`        Unmapped     *(no derived Lean handle found)*

  `prop:stochastic-sequential-strict`             Unmapped     *(no derived Lean handle found)*

  `prop:structural-asymmetry`                     Full         `DQ.ClaimClosure.SR1`, `DQ.ClaimClosure.SR2`, `DQ.ClaimClosure.SR3`

  `prop:substrate-unit-time`                      Full         `DQ.Physics.DecisionTime.substrate_step_is_time_unit`, `DQ.Physics.DecisionTime.substrate_step_realizes_decision_event`, `DQ.Physics.DecisionTime.time_unit_law_substrate_invariant`

  `prop:sufficiency-char`                         Full         `DQ.ClaimClosure.regime_core_claim_proved`, `DQ.ClaimClosure.regime_simulation_transfers_hardness`

  `prop:temporal-equilibrium`                     Full         `DQ.ClaimClosure.IE10`, `DQ.ClaimClosure.IE11`

  `prop:thermo-conservation-additive`             Full         `DQ.ClaimClosure.reusable_heuristic_of_detectable`

  `prop:thermo-hardness-bundle`                   Full         `DQ.ClaimClosure.snapshot_vs_process_typed_boundary`

  `prop:thermo-lift`                              Full         `DQ.ClaimClosure.selectorSufficient_not_implies_setSufficient`, `DQ.ClaimClosure.separable_detectable`

  `prop:thermo-mandatory-cost`                    Full         `DQ.ClaimClosure.standard_assumption_ledger_unpack`

  `prop:time-discrete`                            Full         `DQ.Physics.DecisionTime.time_coordinate_falsifiable`, `DQ.Physics.DecisionTime.time_is_discrete`

  `prop:typed-claim-admissibility`                Full         `DQ.ClaimClosure.sufficiency_iff_projectedOptCover_eq_opt`

  `prop:typed-physical-transport-requirement`     Full         `DQ.Physics.ClaimTransport.physical_state_claim_of_instance_claim`, `DQ.Physics.ClaimTransport.physical_state_claim_of_universal_core`

  `prop:under-resolution-collision`               Full         `DQ.Physics.PhysicalIncompleteness.under_resolution_implies_collision`, `DQ.Physics.PhysicalIncompleteness.under_resolution_implies_decision_collision`

  `prop:universal-solver-framing`                 Full         `DQ.ClaimClosure.thermo_energy_carbon_lift_core`

  `prop:zero-epsilon-competence`                  Full         `DQ.IntegrityCompetence.certifiedTotalBits_gt_raw_of_evidence`, `DQ.IntegrityCompetence.integrity_not_competent_of_nonempty_scope`

  `prop:zero-epsilon-reduction`                   Full         `DQ.ClaimClosure.DecisionProblem.epsOpt_zero_eq_opt`, `DQ.DecisionProblem.minimalSufficient_iff_relevant`

  `thm:abstraction-boundary`                      Full         `DQ.DecisionProblem.collapseBeyondQuotient_physically_impossible`, `DQ.DecisionProblem.not_preservesOpt_iff_erasesDecisionRelevantDistinction`, `DQ.DecisionProblem.surjective_abstraction_factors_or_erases`, `DQ.DecisionProblem.surjective_abstraction_with_feasible_collapse_map_factors`

  `thm:amortization`                              Full         `DQ.HardnessDistribution.complete_model_dominates_after_threshold`

  `thm:assumption-necessity`                      Full         `DQ.Physics.AssumptionNecessity.physical_claim_requires_empirically_justified_physical_assumption`, `DQ.Physics.AssumptionNecessity.physical_claim_requires_physical_assumption`

  `thm:bayes-from-counting`                       Full         `DQ.Foundations.bayes_from_conditional`, `DQ.Foundations.counting_additive`, `DQ.Foundations.counting_nonneg`, `DQ.Foundations.counting_total`, `DQ.Foundations.entropy_contraction`

  `thm:bayes-from-dq`                             Full         `DQ.BayesOptimalityProof.bayes_is_optimal`, `DQ.bayes_update_exists_of_nondegenerateBelief`, `DQ.bayesian_dq_matches_physics_dq`, `DQ.dq_derived_from_bayes`, `DQ.dq_is_bayesian_certainty_fraction`, `DQ.forced_action_under_uncertainty`, `DQ.nondegenerateBelief_of_uncertaintyForced`

  `thm:bayes-optimal`                             Full         `DQ.BayesOptimalityProof.KL_nonneg`, `DQ.BayesOptimalityProof.bayes_is_optimal`, `DQ.BayesOptimalityProof.crossEntropy_eq_entropy_add_KL`

  `thm:boolean-primitive`                         Full         `DQ.Physics.BoundedAcquisition.one_bit_per_transition`

  `thm:bounded-acquisition`                       Full         `DQ.Physics.BoundedAcquisition.acquisition_rate_bound`

  `thm:bridge-boundary-represented`               Full         `DQ.ClaimClosure.boundaryCharacterized_iff_exists_sufficient_subset`, `DQ.ClaimClosure.bounded_actions_detectable`, `DQ.ClaimClosure.bridge_boundary_represented_family`

  `thm:centralization-dominance`                  Full         `DQ.HardnessDistribution.centralization_dominance_bundle`, `DQ.HardnessDistribution.centralization_step_saves_n_minus_one`

  `thm:checking-duality`                          Full         `DQ.checking_time_ge_witness_budget`, `DQ.checking_witnessing_duality_budget`, `DQ.no_sound_checker_below_witness_budget`

  `thm:choice-pays`                               Full         `DQ.ClaimClosure.GE3`, `DQ.ClaimClosure.GE7`

  `thm:claim-integrity-meta`                      Full         `DQ.ClaimClosure.exact_raw_eq_certified_iff_certainty_inflation_core`

  `thm:competence-access`                         Full         `DQ.ClaimClosure.IA5`, `DQ.ClaimClosure.IA6`

  `thm:competence-capacity`                       Full         `DQ.ClaimClosure.CH1`, `DQ.ClaimClosure.CH5`

  `thm:complexity-dichotomy`                      Full         `DQ.StochasticSequential.ClaimClosure.claim_tractable_subcases_to_P`, `DQ.StochasticSequential.complexity_dichotomy_hierarchy`

  `thm:config-reduction`                          Full         `DQ.ConfigReduction.config_sufficiency_iff_behavior_preserving`

  `thm:conservation`                              Full         `DQ.InteriorVerification.agreeOnSet`

  `thm:cost-asymmetry-eth`                        Full         `DQ.ClaimClosure.bridge_transfer_iff_one_step_class`, `DQ.HardnessDistribution.linear_lt_exponential_plus_constant_eventually`

  `thm:counting-gap`                              Full         `DQ.Physics.BoundedAcquisition.counting_gap_theorem`

  `thm:counting-gap-intro`                        Full         `DQ.Physics.BoundedAcquisition.counting_gap_theorem`

  `thm:cycle-cost`                                Full         `DQ.ClaimClosure.RegimeSimulation`, `DQ.ClaimClosure.adq_ordering`, `DQ.ClaimClosure.system_transfer_licensed_iff_snapshot`

  `thm:deficit-source`                            Full         `DQ.InteriorVerification.interiorParetoDominates`, `DQ.InteriorVerification.interior_certificate_implies_non_rejection`

  `thm:dichotomy`                                 Full         `DQ.ClaimClosure.declaredRegimeFamily_complete`, `DQ.ClaimClosure.exact_raw_only_of_no_exact_admissible_core`, `DQ.ClaimClosure.explicit_assumptions_required_of_not_excused_core`

  `thm:discrete-acquisition`                      Full         `DQ.Physics.BoundedAcquisition.acquisitions_are_transitions`

  `thm:dq-physical`                               Full         `DQ.ClaimClosure.DQ1`, `DQ.ClaimClosure.DQ2`, `DQ.ClaimClosure.DQ3`, `DQ.ClaimClosure.DQ4`, `DQ.ClaimClosure.DQ5`, `DQ.ClaimClosure.DQ6`

  `thm:ec2-derived`                               Full         `DQ.ClaimClosure.IA11`, `DQ.ClaimClosure.IA12`, `DQ.ClaimClosure.IA13`, `DQ.ClaimClosure.IA7`, `DQ.ClaimClosure.IA9`

  `thm:ec3-derived`                               Full         `DQ.Physics.LocalityPhysics.FPT10_ec3_is_logical`, `DQ.Physics.LocalityPhysics.FPT4_step_requires_distinct_moments`, `DQ.Physics.LocalityPhysics.FPT5_distinct_moments_positive_duration`, `DQ.Physics.LocalityPhysics.FPT6_step_takes_positive_time`, `DQ.Physics.LocalityPhysics.FPT8_propagation_takes_time`

  `thm:energy-information`                        Full         `DQ.ThermodynamicLift.energy_ge_kbt_nat_entropy`

  `thm:entropy-rank`                              Full         `DQ.numOptClasses_le_pow_srank_binary`, `DQ.quotientEntropy_le_srank_binary`

  `thm:exact-certified-gap-iff-admissible`        Full         `DQ.ClaimClosure.declared_physics_no_universal_exact_certifier_core`, `DQ.ClaimClosure.dichotomy_conditional`, `DQ.ClaimClosure.exact_admissible_iff_raw_lt_certified_total_core`, `DQ.IntegrityCompetence.exact_raw_only_of_no_exact_admissible`

  `thm:fi-coincide`                               Full         `DQ.FunctionalInformation.first_principles_thermo_coincide`, `DQ.FunctionalInformation.functional_information_from_counting`, `DQ.FunctionalInformation.functional_information_from_thermodynamics`, `DQ.Physics.WolpertConstraints.effective_model_dominates_landauer_floor`, `DQ.Physics.WolpertConstraints.effective_model_strictly_exceeds_landauer_of_strict_overhead`, `DQ.Physics.WolpertConstraints.landauer_floor_plus_overhead_lower_bound`, `DQ.Physics.WolpertDecomposition.DecomposedProcessModel.totalOverheadPerBit_eq_sum`, `DQ.Physics.WolpertDecomposition.effective_model_strictly_exceeds_landauer_of_discrete_edge_split`, `DQ.Physics.WolpertDecomposition.effective_model_strictly_exceeds_landauer_of_distribution_mismatch`, `DQ.Physics.WolpertDecomposition.effective_model_strictly_exceeds_landauer_of_either_cited_component`, `DQ.Physics.WolpertDecomposition.effective_model_strictly_exceeds_landauer_of_finite_discrete_witness`, `DQ.Physics.WolpertDecomposition.effective_model_strictly_exceeds_landauer_of_pairwise_flow_asymmetry`, `DQ.Physics.WolpertDecomposition.effective_model_strictly_exceeds_landauer_of_stopping_time_residual`, `DQ.Physics.WolpertDecomposition.energy_lower_bound_increases_by_structural_resource`, `DQ.Physics.WolpertDecomposition.landauer_floor_plus_structural_resource_lower_bound`, `DQ.Physics.WolpertDecomposition.periodic_modular_mismatch_of_distribution_mismatch`, `DQ.Physics.WolpertDecomposition.stopping_time_residual_of_discrete_edge_split`, `DQ.Physics.WolpertDecomposition.stopping_time_residual_of_finite_discrete_witness`, `DQ.Physics.WolpertDecomposition.stopping_time_residual_of_pairwise_flow_asymmetry`, `DQ.Physics.WolpertMismatch.mismatchKL_nonneg`, `DQ.Physics.WolpertMismatch.mismatchKL_pos_of_exists_ne`, `DQ.Physics.WolpertMismatch.mismatchNatLowerBound_pos_of_exists_ne`, `DQ.Physics.WolpertResidual.discreteResidualNatLowerBound_pos_of_asymmetry_or_oneway`, `DQ.Physics.WolpertResidual.pairwiseResidualKL_nonneg`, `DQ.Physics.WolpertResidual.pairwiseResidualKL_pos_of_asymmetry`, `DQ.Physics.WolpertResidual.residualNatLowerBound_pos_of_asymmetry`

  `thm:fisher-rank-srank`                         Full         `DQ.Statistics.fisherMatrix_rank_eq_srank`, `DQ.Statistics.sum_fisherScore_eq_srank`

  `thm:gap-physical`                              Full         `DQ.ClaimClosure.IA4`

  `thm:generalized-dominance`                     Full         `DQ.HardnessDistribution.generalized_right_dominates_wrong_of_bounded_vs_identity_lower`

  `thm:generalized-saturation-possible`           Full         `DQ.HardnessDistribution.generalizedTotal_with_saturation_eventually_constant`, `DQ.HardnessDistribution.saturatingSiteCost_eventually_constant`

  `thm:growth-time`                               Full         `DQ.InteriorVerification.GoalClass`, `DQ.InteriorVerification.InteriorDominanceVerifiable`, `DQ.InteriorVerification.TautologicalSetIdentifiable`

  `thm:information-gap`                           Full         `DQ.ClaimClosure.IA3`

  `thm:landauer-structure`                        Full         `DQ.Physics.LocalityPhysics.landauer_structure`

  `thm:linear-saturation-iff-zero`                Full         `DQ.HardnessDistribution.right_dominates_wrong`

  `thm:measure-prerequisite`                      Full         `DQ.Physics.MeasureNecessity.quantitative_claim_has_measure`, `DQ.Physics.MeasureNecessity.quantitative_measure_is_logical_prerequisite`, `DQ.Physics.MeasureNecessity.stochastic_claim_has_probability_measure`, `DQ.Physics.MeasureNecessity.stochastic_probability_is_logical_prerequisite`

  `thm:nontriviality-counting`                    Full         `DQ.Physics.LocalityPhysics.constant_function_singleton_image`, `DQ.Physics.LocalityPhysics.equal_states_constant_function`, `DQ.Physics.LocalityPhysics.information_requires_nontriviality`, `DQ.Physics.LocalityPhysics.singleton_image_zero_entropy`, `DQ.Physics.LocalityPhysics.trivial_states_all_equal`, `DQ.Physics.LocalityPhysics.triviality_implies_no_information`, `DQ.Physics.LocalityPhysics.zero_entropy_no_information`

  `thm:orbital-cost`                              Full         `DQ.ClaimClosure.AtomicCircuitExports.AC3`, `DQ.ClaimClosure.AtomicCircuitExports.AC4`

  `thm:orbital-transition`                        Full         `DQ.ClaimClosure.AtomicCircuitExports.AC1`, `DQ.ClaimClosure.AtomicCircuitExports.AC5`

  `thm:overmodel-diagnostic`                      Full         `DQ.ClaimClosure.anchor_query_relation_false_iff`, `DQ.ClaimClosure.no_exact_claim_admissible_under_hardness_core`

  `thm:physical-bridge-bundle`                    Full         `DQ.Physics.ClaimTransport.physical_counterexample_yields_core_counterexample`

  `thm:physical-incompleteness`                   Full         `DQ.Physics.PhysicalIncompleteness.no_surjective_instantiation_of_card_gap`, `DQ.Physics.PhysicalIncompleteness.physical_incompleteness_of_bounds`, `DQ.Physics.PhysicalIncompleteness.physical_incompleteness_of_card_gap`

  `thm:quotient-universal`                        Full         `DQ.DecisionProblem.quotientMap_preservesOpt`, `DQ.DecisionProblem.quotient_has_unique_factorization`, `DQ.DecisionProblem.quotient_is_coarsest`, `DQ.DecisionProblem.quotient_represents_opt_equiv`

  `thm:rate-distortion-bridge`                    Full         `DQ.Information.compression_below_srank_fails`, `DQ.Information.equiv_preserves_decision`, `DQ.Information.rate_distortion_bridge`, `DQ.Information.rate_equals_srank`, `DQ.Information.rate_monotone`, `DQ.Information.rate_zero_distortion`, `DQ.Information.shannonEntropy_nonneg`, `DQ.Information.srank_bits_sufficient`

  `thm:regime-coverage`                           Full         `DQ.ClaimClosure.cost_asymmetry_eth_conditional`, `DQ.ClaimClosure.poseAnchorQuery`

  `thm:resolution-sufficient`                     Full         `DQ.Physics.BoundedAcquisition.resolution_reads_sufficient`

  `thm:second-law-counting`                       Full         `DQ.Physics.LocalityPhysics.atypical_states_rare`, `DQ.Physics.LocalityPhysics.entropy_is_information`, `DQ.Physics.LocalityPhysics.errors_accumulate`, `DQ.Physics.LocalityPhysics.random_misses_target`, `DQ.Physics.LocalityPhysics.second_law_from_counting`, `DQ.Physics.LocalityPhysics.verification_is_information`, `DQ.Physics.LocalityPhysics.wrong_paths_dominate`

  `thm:six-subcases`                              Full         `DQ.ClaimClosure.subproblem_hardness_lifts_to_full`, `DQ.ClaimClosure.subproblem_transfer_as_regime_simulation`, `DQ.ClaimClosure.sufficiency_conp_complete_conditional`, `DQ.ClaimClosure.sufficiency_conp_reduction_core`, `DQ.StochasticSequential.ClaimClosure.claim_tractable_subcases_to_P`

  `thm:srank-physical`                            Full         `DQ.ClaimClosure.SR1`, `DQ.Physics.BoundedAcquisition.energy_ge_srank_cost`, `DQ.Physics.BoundedAcquisition.srank_le_resolution_bits`

  `thm:substrate-degradation`                     Full         `DQ.ClaimClosure.SE1`, `DQ.ClaimClosure.SE2`, `DQ.ClaimClosure.SE3`, `DQ.ClaimClosure.SE4`

  `thm:tax-conservation`                          Full         `DQ.HardnessDistribution.gap_conservation_card`

  `thm:tax-grows`                                 Full         `DQ.HardnessDistribution.totalExternalWork_eq_n_mul_gapCard`

  `thm:thermo-derived`                            Full         `DQ.Physics.BoundedAcquisition.physical_grounding_bundle`, `DQ.Physics.WolpertConstraints.landauer_floor_plus_overhead_lower_bound`, `DQ.Physics.WolpertConstraints.physical_grounding_bundle_with_wolpert_overhead`, `DQ.Physics.WolpertDecomposition.DecomposedProcessModel.totalOverheadPerBit_eq_sum`, `DQ.Physics.WolpertDecomposition.effective_model_strictly_exceeds_landauer_of_discrete_edge_split`, `DQ.Physics.WolpertDecomposition.effective_model_strictly_exceeds_landauer_of_distribution_mismatch`, `DQ.Physics.WolpertDecomposition.effective_model_strictly_exceeds_landauer_of_either_cited_component`, `DQ.Physics.WolpertDecomposition.effective_model_strictly_exceeds_landauer_of_finite_discrete_witness`, `DQ.Physics.WolpertDecomposition.effective_model_strictly_exceeds_landauer_of_pairwise_flow_asymmetry`, `DQ.Physics.WolpertDecomposition.effective_model_strictly_exceeds_landauer_of_stopping_time_residual`, `DQ.Physics.WolpertDecomposition.energy_lower_bound_increases_by_structural_resource`, `DQ.Physics.WolpertDecomposition.landauer_floor_plus_decomposition_lower_bound`, `DQ.Physics.WolpertDecomposition.landauer_floor_plus_structural_resource_lower_bound`, `DQ.Physics.WolpertDecomposition.periodic_modular_mismatch_of_distribution_mismatch`, `DQ.Physics.WolpertDecomposition.physical_grounding_bundle_with_wolpert_decomposition`, `DQ.Physics.WolpertDecomposition.stopping_time_residual_of_discrete_edge_split`, `DQ.Physics.WolpertDecomposition.stopping_time_residual_of_finite_discrete_witness`, `DQ.Physics.WolpertDecomposition.stopping_time_residual_of_pairwise_flow_asymmetry`, `DQ.Physics.WolpertMismatch.mismatchKL_eq_zero_iff_eq`, `DQ.Physics.WolpertMismatch.mismatchKL_nonneg`, `DQ.Physics.WolpertMismatch.mismatchKL_pos_of_exists_ne`, `DQ.Physics.WolpertMismatch.mismatchNatLowerBound_pos_of_exists_ne`, `DQ.Physics.WolpertResidual.discreteResidualNatLowerBound_pos_of_asymmetry_or_oneway`, `DQ.Physics.WolpertResidual.pairwiseResidualKL_nonneg`, `DQ.Physics.WolpertResidual.pairwiseResidualKL_pos_of_asymmetry`, `DQ.Physics.WolpertResidual.residualNatLowerBound_pos_of_asymmetry`

  `thm:topology-motion`                           Full         `DQ.ClaimClosure.declaredRegimeFamily_complete`, `DQ.ClaimClosure.explicit_assumptions_required_of_not_excused_core`

  `thm:tractable`                                 Full         `DQ.ClaimClosure.subproblem_hardness_lifts_to_full`, `DQ.ClaimClosure.subproblem_transfer_as_regime_simulation`, `DQ.ClaimClosure.sufficiency_conp_complete_conditional`, `DQ.ClaimClosure.sufficiency_conp_reduction_core`

  `thm:tur-bridge`                                Full         `DQ.Physics.multiple_futures_entropy_production`, `DQ.Physics.transitionProb_nonneg`, `DQ.Physics.transitionProb_sum_one`, `DQ.Physics.tur_bridge`

  `thm:typed-completeness-static`                 Full         `DQ.ClaimClosure.thermo_conservation_additive_core`

  `thm:universe-membership`                       Unmapped     *(no derived Lean handle found)*

  `thm:wasserstein-bridge`                        Full         `DQ.Physics.integrity_is_centroid`, `DQ.Physics.single_future_zero_cost`, `DQ.Physics.transportCost_pos_of_offDiag`, `DQ.Physics.wasserstein_bridge`
  -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

*Notes:* *(1) Full rows come from theorem-local inline anchors in this paper.* *(2) Derived rows are filled by dependency/scaffold claim-handle derivation (same paper-handle label across proof dependencies).* *(3) Unmapped means no local anchor and no derivable dependency support were found.*

*Auto summary: mapped 204/226 (full=204, derived=0, unmapped=22).*


  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **Paper handle**                                **Hardness profile**     **Regime tags**                                 **Lean support**
  ----------------------------------------------- ------------------------ ----------------------------------------------- -----------------------------------------------------------------------------------------------------------------------------------
  `cor:channel-degradation`                       `cost-growth`            H=cost-growth,Q_fin                             CH2, CH6

  `cor:finite-lifetime`                           `cost-growth`            H=cost-growth,Q_fin                             SE5

  `cor:gap-externalization`                       `cost-growth`            H=cost-growth                                   HD21, HD26

  `cor:generalized-eventual-dominance`            `cost-growth`            H=cost-growth                                   HD10

  `cor:hardness-raw-only-exact`                   `cost-growth`            AR,E,H=cost-growth,Qb,Qf,RG,S,S+ETH             CC21, IC23

  `cor:linear-positive-no-saturation`             `cost-growth`            H=cost-growth                                   HD16

  `cor:no-free-computation`                       `cost-growth`            AR,H=cost-growth                                CC7, CC5, CC8

  `cor:physical-counterexample-core-failure`      `cost-growth`            AR,E,H=cost-growth,Qb,Qf,RG,S,S+ETH             CT6, CT5, CT4

  `cor:right-wrong-hardness`                      `cost-growth`            H=cost-growth                                   HD19

  `cor:speed-integrity`                           `cost-growth`            H=cost-growth,Q_fin                             SE6

  `cor:thermo-dq`                                 `cost-growth`            AR,H=cost-growth                                DQ7, DQ8

  `cor:type-system-threshold`                     `cost-growth`            H=cost-growth                                   HD15

  `cor:zero-capacity`                             `cost-growth`            H=cost-growth,Q_fin                             CH3

  `prop:decision-unit-time`                       `cost-growth`            AR,E,H=cost-growth,Qb,Qf,RG,S,S+ETH             DT13, DT11, DT12, DT10

  `prop:dominance-modes`                          `cost-growth`            H=cost-growth                                   HD3

  `prop:generalized-assumption-boundary`          `cost-growth`            H=cost-growth                                   HD7, HD8

  `prop:hardness-conservation`                    `cost-growth`            H=cost-growth                                   HD23, HD25

  `prop:hardness-efficiency-interpretation`       `cost-growth`            H=cost-growth                                   HD11

  `prop:oracle-lattice-strict`                    `cost-growth`            AR,E,H=cost-growth,Qb,Qf,RG,S,S+ETH             CC27, CC29

  `prop:oracle-lattice-transfer`                  `cost-growth`            AR,E,H=cost-growth,Qb,Qf,RG,S,S+ETH             CC43, CC52

  `prop:query-regime-obstruction`                 `cost-growth`            AR,E,H=cost-growth,LC,Q_fin,Qb,Qf,RG,S,S+ETH    CC50

  `prop:query-state-batch-lb`                     `cost-growth`            AR,E,H=cost-growth,Qb,Qf,RG,S,S+ETH             CC28, CC50

  `prop:query-subproblem-transfer`                `cost-growth`            AR,E,H=cost-growth,Qb,Qf,RG,S,S+ETH             CC52, CC58, CC59

  `prop:query-value-entry-lb`                     `cost-growth`            AR,E,H=cost-growth,Qb,Qf,RG,S,S+ETH             CC29, CC50

  `prop:raw-certified-bit-split`                  `cost-growth`            AR,E,H=cost-growth,Qb,Qf,RG,S,S+ETH             CC11, IC10, IC11, IC13, IC14, IC15, IC16, IC18, IC19

  `prop:reaction-competence`                      `cost-growth`            H=cost-growth,Q_fin                             MI3, MI4

  `prop:run-time-accounting`                      `cost-growth`            AR,E,H=cost-growth,Qb,Qf,RG,S,S+ETH             DT18, DT19, DT16, DT15

  `prop:stochastic-landauer-floor`                `cost-growth`            AR,H=cost-growth                                *(no derived Lean handle found)*

  `prop:structural-asymmetry`                     `cost-growth`            AR,H=cost-growth                                SR1, SR2, SR3

  `prop:substrate-unit-time`                      `cost-growth`            AR,E,H=cost-growth,Qb,Qf,RG,S,S+ETH             DT23, DT22, DT24

  `prop:thermo-conservation-additive`             `cost-growth`            H=cost-growth                                   CC64

  `prop:thermo-hardness-bundle`                   `cost-growth`            H=cost-growth                                   CC67

  `prop:thermo-lift`                              `cost-growth`            H=cost-growth                                   CC65, CC66

  `prop:thermo-mandatory-cost`                    `cost-growth`            H=cost-growth                                   CC68

  `prop:time-discrete`                            `cost-growth`            AR,E,H=cost-growth,Qb,Qf,RG,S,S+ETH             DT7, DT6

  `thm:amortization`                              `cost-growth`            AR,E,H=cost-growth,LC,Qb,Qf,RG,S,S+ETH          HD4

  `thm:bayes-from-dq`                             `cost-growth`            AR,H=cost-growth                                FN14, BF4, BB4, BB5, BB3, BF3, BF2

  `thm:bridge-boundary-represented`               `cost-growth`            AR,E,H=cost-growth,LC,Qb,Qf,RG,S,S+ETH          CC8, CC9, CC10

  `thm:centralization-dominance`                  `cost-growth`            AR,E,H=cost-growth,LC,Qb,Qf,RG,S,S+ETH          HD1, HD2

  `thm:choice-pays`                               `cost-growth`            AR,H=cost-growth                                GE3, GE7

  `thm:conservation`                              `cost-growth`            H=cost-growth,Q_fin                             IV4

  `thm:cycle-cost`                                `cost-growth`            AR,H=cost-growth                                CC1, CC2, CC3

  `thm:deficit-source`                            `cost-growth`            H=cost-growth,Q_fin                             IV5, IV6

  `thm:dichotomy`                                 `cost-growth`            AR,E,H=cost-growth,LC,Q_fin,Qb,Qf,RG,S,S+ETH    CC16, CC23, CC24

  `thm:dq-physical`                               `cost-growth`            AR,H=cost-growth                                DQ1, DQ2, DQ3, DQ4, DQ5, DQ6

  `thm:exact-certified-gap-iff-admissible`        `cost-growth`            AR,E,H=cost-growth,Qb,Qf,RG,S,S+ETH             CC17, CC18, CC20, IC31

  `thm:generalized-dominance`                     `cost-growth`            H=cost-growth                                   HD9

  `thm:generalized-saturation-possible`           `cost-growth`            H=cost-growth                                   HD6, HD20

  `thm:growth-time`                               `cost-growth`            H=cost-growth,Q_fin                             IV1, IV2, IV3

  `thm:linear-saturation-iff-zero`                `cost-growth`            H=cost-growth                                   HD19

  `thm:orbital-cost`                              `cost-growth`            H=cost-growth,Q_fin                             AC3, AC4

  `thm:regime-coverage`                           `cost-growth`            AR,E,H=cost-growth,LC,Qb,Qf,RG,S,S+ETH          CC14, CC51

  `thm:substrate-degradation`                     `cost-growth`            H=cost-growth,Q_fin                             SE1, SE2, SE3, SE4

  `thm:tax-conservation`                          `cost-growth`            AR,E,H=cost-growth,LC,Qb,Qf,RG,S,S+ETH          HD5

  `thm:tractable`                                 `cost-growth`            AR,E,H=cost-growth,LC,Qb,Qf,RG,S,S+ETH,STR,TA   CC70, CC71, CC72, CC73

  `thm:typed-completeness-static`                 `cost-growth`            AR,E,H=cost-growth,LC,Qb,Qf,RG,S,S+ETH          CC76

  `cor:finite-budget-no-exact-admissibility`      `exp-lb-conditional`     H=exp-lb-conditional                            PBC8, PBC7

  `cor:finite-budget-threshold-impossibility`     `exp-lb-conditional`     H=exp-lb-conditional                            PBC8, PBC7

  `cor:no-universal-survivor-no-succinct-bound`   `exp-lb-conditional`     H=exp-lb-conditional                            PBC10

  `prop:budgeted-crossover`                       `exp-lb-conditional`     H=exp-lb-conditional                            CC45, PBC1

  `prop:crossover-above-cap`                      `exp-lb-conditional`     H=exp-lb-conditional                            CC44, PBC2

  `prop:crossover-not-certification`              `exp-lb-conditional`     H=exp-lb-conditional                            CC46

  `prop:crossover-policy`                         `exp-lb-conditional`     H=exp-lb-conditional                            CC47

  `prop:eventual-explicit-infeasibility`          `exp-lb-conditional`     H=exp-lb-conditional                            PBC7

  `prop:least-divergence-point`                   `exp-lb-conditional`     H=exp-lb-conditional                            PBC5

  `prop:payoff-threshold`                         `exp-lb-conditional`     H=exp-lb-conditional                            PBC8, PBC9

  `prop:policy-closure-beyond-divergence`         `exp-lb-conditional`     H=exp-lb-conditional                            PBC11, PBC12

  `cor:information-barrier-query`                 `query-lb`               H=query-lb                                      CC27, CC28, CC29

  `cor:query-obstruction-bool`                    `query-lb`               H=query-lb                                      CC49, CC50

  `prop:checking-witnessing-duality`              `query-lb`               H=query-lb                                      WD5, WD3, WD4

  `prop:query-finite-state-generalization`        `query-lb`               H=query-lb                                      CC50

  `prop:query-randomized-robustness`              `query-lb`               H=query-lb                                      CC50

  `prop:query-randomized-weighted`                `query-lb`               H=query-lb                                      CC50

  `prop:query-tightness-full-scan`                `query-lb`               H=query-lb                                      CC50

  `prop:query-weighted-transfer`                  `query-lb`               H=query-lb                                      CC50

  `cor:exact-identifiability`                     `succinct-hard`          H=succinct-hard                                 S2P1

  `cor:gap-minimization-hard`                     `succinct-hard`          H=succinct-hard                                 S2P2

  `cor:no-auto-minimize`                          `succinct-hard`          H=succinct-hard                                 CC37

  `cor:overmodel-diagnostic-implication`          `succinct-hard`          H=succinct-hard                                 S2P6

  `cor:practice-diagnostic`                       `succinct-hard`          H=succinct-hard                                 S2P2

  `prop:sufficiency-char`                         `succinct-hard`          H=succinct-hard                                 CC62, CC63

  `cor:interior-singleton-certificate`            `tractable-structured`   H=tractable-structured                          IV9

  `cor:practice-bounded`                          `tractable-structured`   H=tractable-structured                          CC81

  `cor:practice-structured`                       `tractable-structured`   H=tractable-structured                          CC82

  `cor:practice-symmetry`                         `tractable-structured`   H=tractable-structured                          CC72

  `cor:practice-tensor`                           `tractable-structured`   H=tractable-structured                          CC71

  `cor:practice-tree`                             `tractable-structured`   H=tractable-structured                          CC84

  `cor:practice-treewidth`                        `tractable-structured`   H=tractable-structured                          CC73, CC75

  `prop:heuristic-reusability`                    `tractable-structured`   H=tractable-structured                          CC7, CC53, CC55, CC74

  `prop:interior-verification-tractable`          `tractable-structured`   H=tractable-structured                          IV4, IV7

  `prop:orbital-symmetry`                         `tractable-structured`   H=tractable-structured,Q_fin                    AC8

  `thm:complexity-dichotomy`                      `tractable-structured`   AR,E,H=tractable-structured,Q,S                 DC13, DC3

  `thm:six-subcases`                              `tractable-structured`   AR,E,H=tractable-structured,Q,S                 CC70, CC71, CC72, CC73, DC13

  `cor:ego-trap`                                  `unspecified`            \-                                              *(no derived Lean handle found)*

  `cor:exact-no-competence-zero-certified`        `unspecified`            AR                                              IC41

  `cor:finite-state`                              `unspecified`            \-                                              BA2, BA4

  `cor:forced-finite-speed`                       `unspecified`            \-                                              BA10

  `cor:ground-state`                              `unspecified`            AR                                              BA8

  `cor:ground-state-passive`                      `unspecified`            AR,Q_fin                                        AC6

  `cor:hardness-exact-certainty-inflation`        `unspecified`            CR                                              CC19

  `cor:import-asymmetry`                          `unspecified`            AR                                              SR4, SR5

  `cor:integrity-universal`                       `unspecified`            TR                                              CC31

  `cor:neukart-vinokur`                           `unspecified`            AR                                              CC50, CC65

  `cor:no-uncertified-exact-claim`                `unspecified`            AR                                              CC41

  `cor:outside-excuses-no-exact-report`           `unspecified`            CR,DC                                           CC39

  `cor:phase-transition`                          `unspecified`            AR                                              IE12, IE13

  `cor:physics-no-universal-exact-claim`          `unspecified`            CR                                              CC38

  `cor:posed-anchor-certified-gate`               `unspecified`            \-                                              AQ8

  `cor:practice-unstructured`                     `unspecified`            \-                                              CC26

  `cor:rlff-abstain-no-certs`                     `unspecified`            AR                                              IC30, IC40

  `cor:theorem-equilibrium`                       `unspecified`            AR                                              IE3, IE4, IE5

  `cor:zero-gap`                                  `unspecified`            AR                                              GE4, GE5

  `prop:abstain-guess-self-signal`                `unspecified`            AR                                              IC46

  `prop:abstention-frontier`                      `unspecified`            RG                                              *(no derived Lean handle found)*

  `prop:adq-ordering`                             `unspecified`            \-                                              CC2

  `prop:attempted-competence-matrix`              `unspecified`            AR                                              IC6, IC7, IC27

  `prop:bounded-region`                           `unspecified`            \-                                              BA1

  `prop:bounded-slice-meta-irrelevance`           `unspecified`            AR                                              CC32, CC33

  `prop:bridge-failure-horizon`                   `unspecified`            \-                                              CC25, CC26

  `prop:bridge-failure-stochastic`                `unspecified`            \-                                              CC57

  `prop:bridge-failure-transition`                `unspecified`            \-                                              CC73

  `prop:bridge-transfer-scope`                    `unspecified`            \-                                              CC42

  `prop:certainty-inflation-iff-inadmissible`     `unspecified`            CR                                              IC8, IC22

  `prop:certified-confidence-gate`                `unspecified`            AR                                              IC38, IC39

  `prop:comp-thermo-chain`                        `unspecified`            AR                                              DS5, DS6

  `prop:decision-equivalence`                     `unspecified`            AR                                              DE1, DE2, DE3, DE4

  `prop:declared-contract-selection-validity`     `unspecified`            \-                                              CC22, CC39, CC75

  `prop:discrete-state-time`                      `unspecified`            AR                                              DS1, DS2

  `prop:empty-sufficient-constant`                `unspecified`            DM                                              DP6

  `prop:evidence-admissibility-equivalence`       `unspecified`            AR                                              IC17, IC20, IC21

  `prop:exact-requires-evidence`                  `unspecified`            AR                                              IC35, IC36

  `prop:fraction-defined-under-bound`             `unspecified`            AR                                              IC45

  `prop:heisenberg-strong-nontrivial-opt`         `unspecified`            AR                                              HS3, HS6, HS5

  `prop:identifiability-convergence`              `unspecified`            ID                                              *(no derived Lean handle found)*

  `prop:insufficiency-counterexample`             `unspecified`            DM                                              DP7, DP8

  `prop:integrity-competence-separation`          `unspecified`            AR                                              IC18, IC25

  `prop:integrity-prerequisite`                   `unspecified`            AR                                              IE17

  `prop:integrity-resource-bound`                 `unspecified`            AR,S,S+ETH                                      CC30, IC24, IC26

  `prop:interior-one-sidedness`                   `unspecified`            AR                                              IV6, IV8

  `prop:interior-universal-non-rejection`         `unspecified`            AR                                              IV5

  `prop:landauer-constraint`                      `unspecified`            AR                                              IE1, IE6

  `prop:law-instance-objective-bridge`            `unspecified`            AR                                              CT1, CT2

  `prop:lorentz-discrete`                         `unspecified`            AR                                              DS3, DS4

  `prop:mdp-tractable`                            `unspecified`            FO                                              *(no derived Lean handle found)*

  `prop:minimal-relevant-equiv`                   `unspecified`            \-                                              DP2, DP3

  `prop:no-evidence-zero-certified`               `unspecified`            AR                                              IC40

  `prop:no-fraction-without-bound`                `unspecified`            AR                                              IC44

  `prop:one-step-bridge`                          `unspecified`            \-                                              CC42

  `prop:optimizer-coimage`                        `unspecified`            \-                                              QT7

  `prop:optimizer-entropy-image`                  `unspecified`            \-                                              IT1

  `prop:outside-excuses-explicit-assumptions`     `unspecified`            CR,DC                                           CC22

  `prop:physical-claim-transport`                 `unspecified`            AR                                              CT6, CT3, CT5

  `prop:physics-no-universal-exact`               `unspecified`            CR                                              CC15

  `prop:pose-anchor-object`                       `unspecified`            \-                                              AQ1, AQ2, AQ3

  `prop:posed-anchor-typed-exact`                 `unspecified`            \-                                              AQ4, AQ5, AQ6, AQ7

  `prop:refinement-strengthens`                   `unspecified`            RG                                              *(no derived Lean handle found)*

  `prop:retraction-evidence-integrity`            `unspecified`            RG                                              *(no derived Lean handle found)*

  `prop:retraction-no-evidence-violates`          `unspecified`            RG                                              *(no derived Lean handle found)*

  `prop:rlff-maximizer-admissible`                `unspecified`            AR                                              IC31, IC32

  `prop:selector-separation`                      `unspecified`            \-                                              CC54

  `prop:self-confidence-not-certification`        `unspecified`            AR                                              IC47

  `prop:sequential-anchor-refinement`             `unspecified`            PSPACE                                          *(no derived Lean handle found)*

  `prop:sequential-anchor-tqbf-reduction`         `unspecified`            PSPACE                                          *(no derived Lean handle found)*

  `prop:sequential-bounded-horizon`               `unspecified`            BH                                              *(no derived Lean handle found)*

  `prop:sequential-static-relation`               `unspecified`            DET,T=1                                         *(no derived Lean handle found)*

  `prop:set-to-selector`                          `unspecified`            \-                                              DP5

  `prop:snapshot-process-typing`                  `unspecified`            RA                                              CC48, CC56, CC3

  `prop:srank-support`                            `unspecified`            \-                                              SK1, SK2, SK3

  `prop:static-stochastic-strict`                 `unspecified`            P $\neq$ coNP                                   *(no derived Lean handle found)*

  `prop:static-stochastic-transfer`               `unspecified`            PD                                              *(no derived Lean handle found)*

  `prop:steps-run-scalar`                         `unspecified`            AR                                              IC42, IC43

  `prop:stochastic-anchor-refinement`             `unspecified`            PP                                              *(no derived Lean handle found)*

  `prop:stochastic-anchor-strict-reduction`       `unspecified`            PP                                              *(no derived Lean handle found)*

  `prop:stochastic-bounded-support`               `unspecified`            BS                                              *(no derived Lean handle found)*

  `prop:stochastic-potential-duality`             `unspecified`            PP                                              *(no derived Lean handle found)*

  `prop:stochastic-product-tractable`             `unspecified`            PD                                              *(no derived Lean handle found)*

  `prop:stochastic-sequential-bridge-fail`        `unspecified`            CE                                              *(no derived Lean handle found)*

  `prop:stochastic-sequential-strict`             `unspecified`            P $\neq$ PP                                     *(no derived Lean handle found)*

  `prop:temporal-equilibrium`                     `unspecified`            AR                                              IE10, IE11

  `prop:typed-claim-admissibility`                `unspecified`            AR,DC,Qf,S+ETH                                  CC75

  `prop:typed-physical-transport-requirement`     `unspecified`            AR                                              PS3, PS4

  `prop:under-resolution-collision`               `unspecified`            DM                                              PI6, PI7

  `prop:universal-solver-framing`                 `unspecified`            TR                                              CC77

  `prop:zero-epsilon-competence`                  `unspecified`            \-                                              IC19, IC33

  `prop:zero-epsilon-reduction`                   `unspecified`            \-                                              DP4, DP1

  `thm:abstraction-boundary`                      `unspecified`            \-                                              AB3, AB1, AB2, AB4

  `thm:assumption-necessity`                      `unspecified`            \-                                              AN4, AN3

  `thm:bayes-from-counting`                       `unspecified`            \-                                              BC4, BC3, BC1, BC2, BC5

  `thm:bayes-optimal`                             `unspecified`            \-                                              FN7, FN14, FN12

  `thm:boolean-primitive`                         `unspecified`            \-                                              BA4

  `thm:bounded-acquisition`                       `unspecified`            \-                                              BA2

  `thm:checking-duality`                          `unspecified`            \-                                              WD3, WD1, WD2

  `thm:claim-integrity-meta`                      `unspecified`            CR,DC                                           CC22

  `thm:competence-access`                         `unspecified`            AR,E,Q,S                                        IA5, IA6

  `thm:competence-capacity`                       `unspecified`            AR,Q_fin                                        CH1, CH5

  `thm:config-reduction`                          `unspecified`            AR,E,RG,S,TA                                    CR1

  `thm:cost-asymmetry-eth`                        `unspecified`            S+ETH                                           CC12, HD14

  `thm:counting-gap`                              `unspecified`            \-                                              BA10

  `thm:counting-gap-intro`                        `unspecified`            \-                                              BA10

  `thm:discrete-acquisition`                      `unspecified`            \-                                              BA3

  `thm:ec2-derived`                               `unspecified`            \-                                              IA11, IA12, IA13, IA7, IA9

  `thm:ec3-derived`                               `unspecified`            \-                                              FPT10, FPT4, FPT5, FPT6, FPT8

  `thm:energy-information`                        `unspecified`            \-                                              EI1

  `thm:entropy-rank`                              `unspecified`            \-                                              IT4, IT3

  `thm:fi-coincide`                               `unspecified`            \-                                              FI7, FI3, FI6, WC2, WC3, WC1, WP1, WR8, WM6, WP6, WR10, WR5, WP5, WP8, WP7, WM5, WR7, WR9, WR4, WM1, WM3, WM4, WR6, WR1, WR2, WR3

  `thm:fisher-rank-srank`                         `unspecified`            \-                                              FS2, FS1

  `thm:gap-physical`                              `unspecified`            AR,E,Q,S                                        IA4

  `thm:information-gap`                           `unspecified`            AR,E,Q,S                                        IA3

  `thm:landauer-structure`                        `unspecified`            \-                                              FP15

  `thm:measure-prerequisite`                      `unspecified`            \-                                              MN1, MN10, MN2, MN11

  `thm:nontriviality-counting`                    `unspecified`            \-                                              FP3, FP2, FP7, FP4, FP1, FP6, FP5

  `thm:orbital-transition`                        `unspecified`            AR,Q_fin                                        AC1, AC5

  `thm:overmodel-diagnostic`                      `unspecified`            S                                               CC6, CC40

  `thm:physical-bridge-bundle`                    `unspecified`            AR                                              CT4

  `thm:physical-incompleteness`                   `unspecified`            AR                                              PI3, PI5, PI4

  `thm:quotient-universal`                        `unspecified`            DM                                              QT2, QT7, QT1, QT3

  `thm:rate-distortion-bridge`                    `unspecified`            \-                                              RS3, RS1, RS5, RS2, RD3, RD2, RD1, RS4

  `thm:resolution-sufficient`                     `unspecified`            \-                                              BA5

  `thm:second-law-counting`                       `unspecified`            \-                                              FP8, FP14, FP10, FP9, FP12, FP13, FP11

  `thm:srank-physical`                            `unspecified`            \-                                              SR1, BA7, BA6

  `thm:tax-grows`                                 `unspecified`            LC                                              HD26

  `thm:thermo-derived`                            `unspecified`            \-                                              BA9, WC1, WC5, WP1, WR8, WM6, WP6, WR10, WR5, WP5, WP8, WP2, WP7, WM5, WP9, WR7, WR9, WR4, WM2, WM1, WM3, WM4, WR6, WR1, WR2, WR3

  `thm:topology-motion`                           `unspecified`            AR,E,Q,S                                        CC16, CC24

  `thm:tur-bridge`                                `unspecified`            \-                                              TUR6, TUR1, TUR2, TUR5

  `thm:universe-membership`                       `unspecified`            \-                                              *(no derived Lean handle found)*

  `thm:wasserstein-bridge`                        `unspecified`            \-                                              W3, W1, W2, W4
  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

*Auto summary: indexed 226 claims by hardness profile (cost-growth=56; exp-lb-conditional=11; query-lb=8; succinct-hard=6; tractable-structured=12; unspecified=133).*




---

## Machine-Checked Proofs

All theorems are formalized in Lean 4:
- Location: `docs/papers/paper4_decision_quotient/proofs/`
- Lines: 33311
- Theorems: 1378
- `sorry` placeholders: 0
