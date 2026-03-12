# Paper: Decision Quotient: Complexity of Exact Relevance Certification

**Status**: Computational Complexity-ready | **Lean**: 49744 lines, 2152 theorems

---

## Abstract

_Abstract not available._

# Introduction {#sec:introduction}

Which coordinates of a system's state can be safely discarded without changing the optimal action, and how hard is it to certify that claim exactly? For a decision problem $\mathcal{D}=(A,S,U)$ with $S = X_1 \times \cdots \times X_n$, a coordinate set $I$ is sufficient when agreement on $I$ forces agreement on the optimal-action set: $$s_I = s'_I \implies \operatorname{Opt}(s) = \operatorname{Opt}(s').$$ This is the basic problem of exact relevance certification: how much of the state must remain visible in order to preserve the decision boundary, and what it takes to certify that the rest is irrelevant.

The question is easy to state but not algorithmically benign. Evaluating $U(a,s)$ on a given state is one task; certifying that an entire family of coordinates may be ignored is another. The latter carries universal quantification over state pairs and leads naturally to coNP and $\Sigma_2^P$ phenomena. The optimizer quotient packages the same issue structurally: it is the coarsest abstraction of the state space that preserves optimal-action distinctions.

This paper isolates that abstract problem. We work entirely with finite coordinate-structured decision problems and explicit encoding regimes.

Section [\[sec:encoding\]](#sec:encoding){reference-type="ref" reference="sec:encoding"} fixes the computational model and the encoding regimes used by every complexity claim.

## Contributions

1.  **Shared setup for coordinate sufficiency.** We formalize decision problems with coordinate structure, relevant-coordinate witnesses, exact relevance identification, and the optimizer quotient as the canonical decision-preserving abstraction.

2.  **Exact complexity classifications.** We prove that [Sufficiency-Check]{.smallcaps} is coNP-complete, [Minimum-Sufficient-Set]{.smallcaps} is coNP-complete, and [Anchor-Sufficiency]{.smallcaps} is $\Sigma_2^P$-complete.

3.  **Encoding-sensitive hardness and tractability.** We prove an explicit-state versus succinct dichotomy, derive ETH-conditioned lower bounds, and isolate natural tractable subcases under structural restrictions.

4.  **Reliability and approximation impossibility results.** We prove a witness-checking lower bound for exact empty-set certification, approximation-hardness results in which any uniform factor guarantee on the shifted hard family yields tautology detection by thresholding and an exact set-cover equivalence on a separate explicit gadget family yields approximation-ratio transfer, and an integrity-competence impossibility theorem showing that under $P\neq coNP$ no polynomial-budget solver is both integrity-preserving and fully competent on the hard exact regime.

5.  **Hardness transfers for exact simplification.** We show that exact behavior-preserving minimization inherits the hardness of [Minimum-Sufficient-Set]{.smallcaps}, while conservative over-specification can be rational in the hard succinct regime under an explicit cost model. The simplicity-tax and hardness-conservation section pins a chosen architecture against the canonical relevant-coordinate invariant and derives the resulting externalization and amortization consequences.

6.  **Machine-checked reductions and proof artifacts.** The main reductions and supporting lemmas are mechanized in Lean 4, yielding a reproducible proof artifact for the core complexity claims.

## Why the Problem Is Different

The slogan of the paper is simple:

> **Determining what you need to know can be harder than evaluating the full objective once everything is known.**

The complexity gap comes from the structure of the question. Insufficiency has short witnesses: two states that agree on the proposed coordinates but induce different optimal-action sets. Sufficiency, by contrast, asks for the absence of every such witness. So the object-level task is forward evaluation on a fully specified state, while the meta-level task is universal verification over counterexample pairs. This difference drives the coNP classification and explains why relevance identification is not just another forward evaluation problem.

## Formalization Snapshot

::: center
  **Metric**          **Value**
  ----------------- -----------
  Lines of Lean 4         49744
  Theorems/lemmas          2152
  Proof files               189
  `sorry` count               0
:::

The artifact builds with `lake build`. The purpose of the mechanization here is not to replace the mathematical narrative, but to guarantee the correctness of the reduction constructions and the formal statements attached to them.

## Paper Structure

Section [\[sec:formal-setup\]](#sec:formal-setup){reference-type="ref" reference="sec:formal-setup"} introduces the abstract setup and the encoding contract. The hardness theorems are stated in Section [\[sec:hardness\]](#sec:hardness){reference-type="ref" reference="sec:hardness"}. Section [\[sec:dichotomy\]](#sec:dichotomy){reference-type="ref" reference="sec:dichotomy"} gives the encoding-sensitive dichotomy and ETH chain, Section [\[sec:tractable\]](#sec:tractable){reference-type="ref" reference="sec:tractable"} records tractable subcases, Section [\[sec:engineering-corollaries\]](#sec:engineering-corollaries){reference-type="ref" reference="sec:engineering-corollaries"} gives hardness transfers and exact-simplification consequences, Section [\[sec:integrity-competence\]](#sec:integrity-competence){reference-type="ref" reference="sec:integrity-competence"} gives the integrity-competence impossibility theorem, Section [\[sec:witness-duality\]](#sec:witness-duality){reference-type="ref" reference="sec:witness-duality"} gives the witness and approximation consequences, and Section [\[sec:simplicity-tax\]](#sec:simplicity-tax){reference-type="ref" reference="sec:simplicity-tax"} gives the simplicity-tax and hardness-conservation consequences of omitted exact relevance. Section [\[sec:related\]](#sec:related){reference-type="ref" reference="sec:related"} situates the work within formalized complexity theory and decision-theoretic relevance. Section [\[app:lean\]](#app:lean){reference-type="ref" reference="app:lean"} summarizes the Lean artifact.

## Artifact Availability

The complete Lean 4 formalization is available at:

::: center
<https://doi.org/10.5281/zenodo.18140965>
:::

The proofs build with the Lean toolchain specified in `lean-toolchain`.


# Formal Setup {#sec:formal-setup}

This section fixes the abstract, non-physical vocabulary used throughout the paper. We work with finite decision problems whose state spaces factor into coordinates, and we isolate the minimal terminology needed to state the later complexity results.

## Decision Problems with Coordinate Structure

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
[]{#def:sufficient label="def:sufficient"} A coordinate set $I \subseteq \{1, \ldots, n\}$ is *sufficient* for decision problem $\mathcal{D}$ if: $$\forall s, s' \in S: \quad s_I = s'_I \implies \operatorname{Opt}(s) = \operatorname{Opt}(s')$$ Equivalently, the optimal action depends only on coordinates in $I$.
:::

::: proposition
[]{#prop:empty-sufficient-constant label="prop:empty-sufficient-constant"} The empty set $\emptyset$ is sufficient if and only if $\operatorname{Opt}(s)$ is constant over the entire state space.
:::

::: proof
*Proof.* If $\emptyset$ is sufficient, then every two states agree on their empty projection, so sufficiency forces $\operatorname{Opt}(s)=\operatorname{Opt}(s')$ for all $s,s' \in S$. The converse is immediate. ◻
:::

::: proposition
[]{#prop:insufficiency-counterexample label="prop:insufficiency-counterexample"} Coordinate set $I$ is not sufficient if and only if there exist states $s,s' \in S$ such that $s_I=s'_I$ but $\operatorname{Opt}(s)\neq\operatorname{Opt}(s')$.
:::

::: proof
*Proof.* This is the direct negation of Definition [\[def:sufficient\]](#def:sufficient){reference-type="ref" reference="def:sufficient"}. ◻
:::

::: definition
[]{#def:minimal-sufficient label="def:minimal-sufficient"} A sufficient set $I$ is *minimal* if no proper subset $I' \subsetneq I$ is sufficient.
:::

::: definition
[]{#def:relevant label="def:relevant"} Coordinate $i$ is *relevant* if there exist states that differ only at coordinate $i$ and induce different optimal-action sets: $$i \text{ is relevant}
\iff
\exists s,s' \in S:\;
\Big(\forall j \neq i,\; s_j=s'_j\Big)
\ \wedge\
\operatorname{Opt}(s)\neq\operatorname{Opt}(s').$$
:::

::: proposition
[]{#prop:minimal-relevant-equiv label="prop:minimal-relevant-equiv"} For any minimal sufficient set $I$ and any coordinate $i$, $$i \in I \iff i \text{ is relevant}.$$ Hence every minimal sufficient set is exactly the relevant-coordinate set.
:::

::: proof
*Proof.* If $i \in I$ were not relevant, then changing coordinate $i$ while keeping all others fixed would never alter the optimal-action set, so removing $i$ would preserve sufficiency, contradicting minimality. Conversely, every sufficient set must contain each relevant coordinate, because omitting a relevant coordinate leaves a witness pair with identical $I$-projection but different optimal-action sets. ◻
:::

::: definition
[]{#def:exact-identifiability label="def:exact-identifiability"} For a decision problem $\mathcal{D}$ and candidate set $I$, we say that $I$ is *exactly relevance-identifying* if $$\forall i \in \{1,\ldots,n\}:\quad i \in I \iff i \text{ is relevant for } \mathcal{D}.$$
:::

::: definition
[]{#def:srank label="def:srank"} The *structural rank* of a decision problem is the number of relevant coordinates: $$\mathrm{srank}(\mathcal{D}) := \left|\{i \in \{1,\ldots,n\}: i \text{ is relevant}\}\right|.$$
:::

## Optimizer Quotient

::: definition
[]{#def:decision-equiv label="def:decision-equiv"} States $s,s' \in S$ are *decision-equivalent* if they induce the same optimal-action set: $$s \sim_{\operatorname{Opt}} s' \iff \operatorname{Opt}(s)=\operatorname{Opt}(s').$$
:::

::: definition
[]{#def:decision-quotient label="def:decision-quotient"} The *decision quotient* (equivalently, optimizer quotient) of $\mathcal{D}$ is the quotient object $$Q_{\mathcal{D}} := S / {\sim_{\operatorname{Opt}}}.$$ Its equivalence classes are exactly the maximal subsets of states on which the optimal-action set is constant.
:::

::: proposition
[]{#prop:optimizer-coimage label="prop:optimizer-coimage"} The decision quotient $Q_{\mathcal{D}}$ is canonically in bijection with $\operatorname{range}(\operatorname{Opt})$. Equivalently, in **Set**, it is the coimage/image factorization of the optimizer map.
:::

::: proof
*Proof.* Two states are identified by $\sim_{\operatorname{Opt}}$ exactly when they have the same optimizer value. Hence quotient classes are in one-to-one correspondence with attained values of $\operatorname{Opt}$. ◻
:::

::: proposition
[]{#prop:sufficiency-char label="prop:sufficiency-char"} Coordinate set $I$ is sufficient if and only if the optimizer map factors through the projection $\pi_I : S \to S_I$. Equivalently, there exists a function $\overline{\operatorname{Opt}}_I : S_I \to \mathcal{P}(A)$ such that $$\operatorname{Opt}= \overline{\operatorname{Opt}}_I \circ \pi_I.$$
:::

::: proof
*Proof.* If $I$ is sufficient, define $\overline{\operatorname{Opt}}_I(x)$ to be $\operatorname{Opt}(s)$ for any state $s$ with $s_I = x$; this is well defined exactly because sufficiency makes $\operatorname{Opt}$ constant on each projection fiber. Conversely, any such factorization forces states with equal $I$-projection to have equal optimal-action sets. ◻
:::

## Computational Model and Input Encoding {#sec:encoding}

The same decision question yields different computational problems under different input models. We therefore fix the encoding regime before stating any complexity classification.

#### Succinct encoding (primary for hardness).

An instance is encoded by:

-   a finite action set $A$ given explicitly,

-   coordinate domains $X_1,\ldots,X_n$ given by their sizes in binary,

-   a Boolean or arithmetic circuit $C_U$ that on input $(a,s)$ outputs $U(a,s)$.

The input length is $$L = |A| + \sum_i \log |X_i| + |C_U|.$$ Unless stated otherwise, all class-theoretic hardness results in this paper are measured in $L$.

#### Explicit-state encoding.

The utility function is given as a full table over $A \times S$. The input length is $$L_{\mathrm{exp}} = \Theta(|A||S|)$$ up to the bitlength of the utility entries. Polynomial-time claims stated in terms of $|S|$ use this explicit-state regime.

#### Query-access regime.

The solver is given oracle access to $\operatorname{Opt}(s)$ or to the utility values needed to reconstruct it. Complexity is then measured by query count, optionally paired with per-query evaluation cost. This regime separates access obstruction from full-table availability and from succinct circuit input length.

::: remark
[]{#rem:question-vs-problem label="rem:question-vs-problem"} The predicate "is coordinate set $I$ sufficient for $\mathcal{D}$?" is an encoding-independent mathematical question. A *decision problem* arises only after fixing how $\mathcal{D}$ and $I$ are represented. Complexity classes are properties of these encoded problems, not of the abstract predicate in isolation.
:::


# Encoding Dichotomy and ETH Lower Bounds {#sec:dichotomy}

The hardness results of Section [\[sec:hardness\]](#sec:hardness){reference-type="ref" reference="sec:hardness"} apply to worst-case instances under the succinct encoding. This section states a dichotomy that separates an explicit-state upper bound from a succinct-encoding lower bound. The dichotomy and ETH reduction chain are machine-verified in Lean 4.

#### Model note.

Part 1 is an explicit-state upper bound (time polynomial in $|S|$). Part 2 is a succinct-encoding lower bound under ETH (time exponential in $n$). The encodings are defined in Section [\[sec:encoding\]](#sec:encoding){reference-type="ref" reference="sec:encoding"}.

::: theorem
[]{#thm:dichotomy label="thm:dichotomy"} Let $\mathcal{D} = (A, X_1, \ldots, X_n, U)$ be a decision problem with $|S| = N$ states. Let $k^*$ be the size of the minimal sufficient set.

1.  **Logarithmic case (explicit-state upper bound):** If $k^* = O(\log N)$, then SUFFICIENCY-CHECK is solvable in polynomial time in $N$ under the explicit-state encoding.

2.  **Linear case (succinct lower bound under ETH):** If $k^* = \Omega(n)$, then SUFFICIENCY-CHECK requires time $\Omega(2^{n/c})$ for some constant $c > 0$ under the succinct encoding (assuming ETH).
:::

::: proof
*Proof.* **Part 1 (Logarithmic case):** If $k^* = O(\log N)$, then the number of distinct projections $|S_{I^*}|$ is at most $2^{k^*} = O(N^c)$ for some constant $c$. Under the explicit-state encoding, we enumerate all projections and verify sufficiency in polynomial time.

**Part 2 (Linear case):** We establish this via an explicit reduction chain from the Exponential Time Hypothesis under the succinct encoding. ◻
:::

## The ETH Reduction Chain {#sec:eth-chain}

The lower bound in Part 2 of Theorem [\[thm:dichotomy\]](#thm:dichotomy){reference-type="ref" reference="thm:dichotomy"} follows from a chain of reductions originating in the Exponential Time Hypothesis. We make this chain explicit.

::: definition
There exists a constant $\delta > 0$ such that 3-SAT on $n$ variables cannot be solved in time $O(2^{\delta n})$ [@impagliazzo2001complexity].
:::

The chain proceeds as follows:

1.  **ETH $\Rightarrow$ 3-SAT requires $2^{\Omega(n)}$:** This is the definition of ETH.

2.  **3-SAT $\leq_p$ TAUTOLOGY:** Given 3-SAT formula $\varphi(x_1, \ldots, x_n)$, define $\psi = \neg\varphi$. Then $\varphi$ is satisfiable iff $\psi$ is not a tautology. This is a linear-time reduction preserving the number of variables.

3.  **TAUTOLOGY requires $2^{\Omega(n)}$ (under ETH):** By the contrapositive of step 2, if TAUTOLOGY is solvable in $o(2^{\delta n})$ time, then 3-SAT is solvable in $o(2^{\delta n})$ time, contradicting ETH.

4.  **TAUTOLOGY $\leq_p$ SUFFICIENCY-CHECK:** This is Theorem [\[thm:sufficiency-conp\]](#thm:sufficiency-conp){reference-type="ref" reference="thm:sufficiency-conp"}. Given formula $\varphi(x_1, \ldots, x_n)$, we construct a decision problem where:

    -   The empty set $I = \emptyset$ is sufficient iff $\varphi$ is a tautology

    -   When $\varphi$ is not a tautology, all $n$ coordinates are relevant

    The reduction is polynomial-time and preserves the number of coordinates.

5.  **SUFFICIENCY-CHECK requires $2^{\Omega(n)}$ (under ETH):** Combining steps 3 and 4: if SUFFICIENCY-CHECK is solvable in $o(2^{\delta n/c})$ time for some constant $c$, then TAUTOLOGY (and hence 3-SAT) is solvable in subexponential time, contradicting ETH.

::: proposition
[]{#prop:eth-constant label="prop:eth-constant"} The reduction in Theorem [\[thm:sufficiency-conp\]](#thm:sufficiency-conp){reference-type="ref" reference="thm:sufficiency-conp"} preserves the number of variables up to an additive constant: an $n$-variable formula yields an $(n+1)$-coordinate decision problem. Therefore, the constant $c$ in the $2^{n/c}$ lower bound is asymptotically 1: $$\text{SUFFICIENCY-CHECK requires time } \Omega(2^{\delta (n-1)}) = 2^{\Omega(n)} \text{ under ETH}$$ where $\delta$ is the ETH constant for 3-SAT.
:::

::: proof
*Proof.* The TAUTOLOGY reduction (Theorem [\[thm:sufficiency-conp\]](#thm:sufficiency-conp){reference-type="ref" reference="thm:sufficiency-conp"}) constructs:

-   State space $S = \{\text{ref}\} \cup \{0,1\}^n$ with $n+1$ coordinates (one extra for the reference state)

-   Query set $I = \emptyset$

When $\varphi$ has $n$ variables, the constructed problem has $n+1$ coordinates. The asymptotic lower bound is $2^{\Omega(n)}$ with the same constant $\delta$ from ETH. ◻
:::

## Phase Transition

::: corollary
[]{#cor:phase-transition label="cor:phase-transition"} There is a sharp transition between tractable and intractable regimes at the logarithmic scale (with respect to the encodings in Section [\[sec:encoding\]](#sec:encoding){reference-type="ref" reference="sec:encoding"}):

-   If $k^* = O(\log N)$, SUFFICIENCY-CHECK is polynomial in $N$ under the explicit-state encoding

-   If $k^* = \Omega(n)$, SUFFICIENCY-CHECK is exponential in $n$ under ETH in the succinct encoding

For Boolean coordinate spaces ($N = 2^n$), the transition is between $k^* = O(\log n)$ (explicit-state tractable) and $k^* = \Omega(n)$ (succinct-encoding intractable).
:::

::: proof
*Proof.* The logarithmic case (Part 1 of Theorem [\[thm:dichotomy\]](#thm:dichotomy){reference-type="ref" reference="thm:dichotomy"}) gives polynomial time when $k^* = O(\log N)$. More precisely, when $k^* \leq c \log N$ for constant $c$, the algorithm runs in time $O(N^c \cdot \text{poly}(n))$.

The linear case (Part 2) gives exponential time when $k^* = \Omega(n)$.

The transition point is where $2^{k^*} = N^{k^*/\log N}$ stops being polynomial in $N$, i.e., when $k^* = \omega(\log N)$. For Boolean coordinate spaces ($N = 2^n$), this corresponds to the gap between $k^* = O(\log n)$ and $k^* = \Omega(n)$. ◻
:::

::: remark
Under ETH, the lower bound is asymptotically tight in the succinct encoding. The explicit-state upper bound is tight in the sense that it matches enumeration complexity in $N$.
:::

This dichotomy explains why some domains admit tractable model selection (few relevant variables) while others require heuristics (many relevant variables). In the succinct regime, the reduction chain gives a class-level coNP statement together with a worst-case $2^{\Omega(n)}$ lower bound under the declared ETH assumption.

#### Bridge theorems.

The ETH chain composes with structural rank characterizations to yield the complete complexity picture. Specifically: BR1 combines ETH hardness with structural rank to show that problems with maximal srank (all coordinates relevant) inherit the exponential lower bound; CC1 composes the dichotomy theorem with ETH to give the full classification; DQ5 combines the TAUTOLOGY reduction with ETH to establish coNP-completeness plus exponential hardness. These bridge theorems connect the complexity cluster (ETH, reductions) to the geometric cluster (srank, sufficiency) in the proof graph.

::: remark
The ETH lower bound is stated in the succinct circuit encoding of Section [\[sec:encoding\]](#sec:encoding){reference-type="ref" reference="sec:encoding"}, where the utility function $U: A \times S \to \mathbb{R}$ is represented by a Boolean circuit computing $\mathbf{1}[U(a,s) > \theta]$ for threshold comparisons. In this model:

-   The input size is the circuit size $m$, not the state space size $|S| = 2^n$

-   A 3-SAT formula with $n$ variables and $c$ clauses yields a circuit of size $O(n + c)$

-   The reduction preserves instance size up to constant factors: $m_{\text{out}} \leq 3 \cdot m_{\text{in}}$

This linear size preservation is essential for ETH transfer. In the explicit enumeration model (where $S$ is given as a list), the reduction would blow up the instance size exponentially, precluding ETH-based lower bounds. The circuit model is standard in fine-grained complexity and matches practical representations of decision problems.
:::


# Tractable Special Cases {#sec:tractable}

We distinguish the encodings of Section [\[sec:encoding\]](#sec:encoding){reference-type="ref" reference="sec:encoding"}. The tractability results below state the model assumption explicitly. All results are formalized in `DecisionQuotient/Tractability/` and verified in Lean 4 with no `sorry` placeholders.

::: theorem
[]{#thm:tractable label="thm:tractable"} SUFFICIENCY-CHECK is polynomial-time solvable in the following cases:

1.  **Bounded actions (explicit-state encoding):** The input contains the full utility table over $A \times S$ with $|A| \leq k$ for constant $k$. Complexity: $O(|S|^2 \cdot k^2)$.

2.  **Separable utility (rank 1):** $U(a, s) = f(a) + g(s)$. Complexity: $O(1)$ (trivially sufficient).

3.  **Low tensor rank:** The utility admits a rank-$R$ decomposition $U(a,s) = \sum_{r=1}^R w_r \cdot f_r(a) \cdot \prod_i g_{ri}(s_i)$. Complexity: $O(|A| \cdot R \cdot n)$.

4.  **Tree-structured dependencies:** Coordinates are ordered such that $s_i$ depends only on $(s_1, \ldots, s_{i-1})$. Complexity: $O(n \cdot |A| \cdot k_{\max})$ where $k_{\max} = \max_i |X_i|$.

5.  **Bounded treewidth:** The interaction graph has treewidth $\leq w$ and utility factors over edges. Complexity: $O(n \cdot k^{w+1})$ via standard CSP algorithms.

6.  **Coordinate symmetry:** Utility is invariant under coordinate permutations. Complexity: $O\bigl(\binom{d+k-1}{k-1}^2\bigr)$ orbit-type pairs.
:::

The following subsections provide formal definitions, reduction theorems, and prose descriptions for each tractable subcase. For each result, we cite standard algorithms where applicable and prove only the paper-specific reductions in Lean.

## Bounded Actions {#sec:tract-bounded}

When the action set is bounded, brute-force enumeration becomes polynomial.

::: definition
A decision problem has *bounded actions* with parameter $k$ if $|A| \leq k$ for some constant $k$ independent of the state space size.
:::

::: theorem
[]{#thm:bounded-actions label="thm:bounded-actions"} If $|A| \leq k$ for constant $k$, then SUFFICIENCY-CHECK runs in $O(|S|^2 \cdot k^2)$ time.
:::

::: proof
*Proof.* Given the full table of $U(a,s)$:

1.  Compute $\operatorname{Opt}(s)$ for all $s \in S$ in $O(|S| \cdot k)$ time.

2.  For each pair $(s, s')$ with $s_I = s'_I$, compare $\operatorname{Opt}(s)$ and $\operatorname{Opt}(s')$ in $O(k^2)$ time.

3.  Total: $O(|S|^2)$ pairs $\times$ $O(k^2)$ comparisons = $O(|S|^2 \cdot k^2)$.

When $k$ is constant, this is polynomial in $|S|$. ◻
:::

**Lean formalization:** `DecisionQuotient/Tractability/BoundedActions.lean`

-   L10: Correctness of the algorithm.

-   L2: Complexity bound $|S|^2 \cdot (1 + k^2)$.

## Separable Utility (Rank 1) {#sec:tract-separable}

Separable utility decouples actions from states, making sufficiency trivial.

::: definition
[]{#def:separable label="def:separable"} A utility function is *separable* if there exist functions $f : A \to \mathbb{R}$ and $g : S \to \mathbb{R}$ such that $U(a,s) = f(a) + g(s)$ for all $(a,s) \in A \times S$.
:::

::: theorem
[]{#thm:separable label="thm:separable"} If utility is separable, then $I = \emptyset$ is always sufficient.
:::

::: proof
*Proof.* If $U(a, s) = f(a) + g(s)$, then: $$\operatorname{Opt}(s) = \arg\max_{a \in A} [f(a) + g(s)] = \arg\max_{a \in A} f(a)$$ The optimal action depends only on $f$, not on $s$. Hence $\operatorname{Opt}(s) = \operatorname{Opt}(s')$ for all $s, s'$, making any $I$ (including $\emptyset$) sufficient. ◻
:::

**Lean formalization:** `DecisionQuotient/Tractability/SeparableUtility.lean`

-   SeparableUtility: Structure encoding the separability condition.

-   L8: Optimal actions are state-independent.

-   L8: The empty set is sufficient.

## Low Tensor Rank {#sec:tract-tensor}

Utility functions with low tensor rank admit efficient factored computation, generalizing separable utility.

::: definition
[]{#def:tensor-rank label="def:tensor-rank"} A utility function $U : A \times ((i : \mathsf{Fin}\ n) \to X_i) \to \mathbb{R}$ has *tensor rank* $R$ if there exist weights $w_1, \ldots, w_R \in \mathbb{R}$, action factors $f_r : A \to \mathbb{R}$, and coordinate factors $g_{ri} : X_i \to \mathbb{R}$ such that: $$U(a, s) = \sum_{r=1}^R w_r \cdot f_r(a) \cdot \prod_{i=1}^n g_{ri}(s_i)$$
:::

::: remark
Separable utility (Definition [\[def:separable\]](#def:separable){reference-type="ref" reference="def:separable"}) is the special case $R = 1$ with $w_1 = 1$, $f_1(a) = f(a)$, and $\prod_i g_{1i}(s_i) = g(s)$.
:::

::: theorem
[]{#thm:tensor-rank label="thm:tensor-rank"} If utility has tensor rank $R$, then SUFFICIENCY-CHECK runs in $O(|A| \cdot R \cdot n)$ time per state.
:::

::: proof
*Proof.* The factored representation enables efficient computation:

1.  For each rank component $r$, the contribution $w_r \cdot f_r(a) \cdot \prod_i g_{ri}(s_i)$ factors over coordinates.

2.  Two states $s, s'$ agreeing on $I$ have $g_{ri}(s_i) = g_{ri}(s'_i)$ for $i \in I$.

3.  The partial product $\prod_{i \in I} g_{ri}(s_i)$ is identical for agreeing states.

4.  This reduces sufficiency checking to comparing factored sums over orbit representatives.

 ◻
:::

**Lean formalization:** `DecisionQuotient/Tractability/SeparableUtility.lean`

-   TensorRankDecomposition: Structure encoding rank-$R$ decomposition.

-   L14: Axiom citing tensor network algorithms [@kolda2009tensor; @cichocki2016tensor].

-   L4: Reduction theorem for factored computation.

-   L6: Tractability via factored sufficiency check.

## Tree-Structured Dependencies {#sec:tract-tree}

When coordinate dependencies form a tree, dynamic programming yields polynomial-time sufficiency checking.

::: definition
[]{#def:tree-struct label="def:tree-struct"} A decision problem has *tree-structured dependencies* if there exists an ordering of coordinates $(1, \ldots, n)$ such that the utility decomposes as: $$U(a, s) = \sum_{i=1}^n u_i(a, s_i, s_{\mathrm{parent}(i)})$$ where $\mathrm{parent}(i) < i$ for all $i > 1$, and the root term depends only on $(a, s_1)$.
:::

::: theorem
[]{#thm:tree-struct label="thm:tree-struct"} If dependencies are tree-structured with explicit local tables, SUFFICIENCY-CHECK runs in $O(n \cdot |A| \cdot k_{\max})$ time where $k_{\max} = \max_i |X_i|$.
:::

::: proof
*Proof.* Dynamic programming on the tree order:

1.  Process coordinates in order $i = n, n-1, \ldots, 1$.

2.  For each node $i$ and each value of its parent coordinate, compute the set of actions optimal for some subtree assignment.

3.  Combine child summaries with local tables in $O(|A| \cdot k_{\max})$ per node.

4.  Total: $O(n \cdot |A| \cdot k_{\max})$.

A coordinate is relevant iff varying its value changes the optimal action set. ◻
:::

**Lean formalization:** `DecisionQuotient/Tractability/TreeStructure.lean`

-   TreeStructured: Predicate encoding tree dependency structure.

-   L16: Axiom citing standard DP on trees.

## Bounded Treewidth {#sec:tract-treewidth}

The tree-structured case generalizes to bounded treewidth interaction graphs via CSP reduction.

::: definition
[]{#def:interaction-graph label="def:interaction-graph"} Given a pairwise utility decomposition, the *interaction graph* $G = (V, E)$ has:

-   Vertices $V = \{1, \ldots, n\}$ (one per coordinate)

-   Edges $E = \{(i, j) : i \text{ and } j \text{ interact in the utility}\}$
:::

::: definition
[]{#def:pairwise label="def:pairwise"} A utility function is *pairwise* if it decomposes as: $$U(a, s) = \sum_{i=1}^n u_i(a, s_i) + \sum_{(i,j) \in E} u_{ij}(a, s_i, s_j)$$ with unary terms $u_i$ and binary terms $u_{ij}$ given explicitly.
:::

::: theorem
[]{#thm:treewidth label="thm:treewidth"} If the interaction graph has treewidth $\leq w$ and utility is pairwise with explicit factors, then SUFFICIENCY-CHECK runs in $O(n \cdot k^{w+1})$ time where $k = \max_i |X_i|$.
:::

::: proof
*Proof.* We reduce SUFFICIENCY-CHECK to constraint satisfaction on the interaction graph:

1.  **Reduction:** For each action $a \in A$, checking whether $\operatorname{Opt}(s) = \operatorname{Opt}(s')$ for states agreeing on $I$ reduces to a CSP on $G$ with variables $\{s_i : i \notin I\}$.

2.  **Standard algorithm:** CSP on graphs of treewidth $w$ is solvable in $O(n \cdot k^{w+1})$ via tree decomposition [@bodlaender1993tourist; @freuder1990complexity].

The reduction is polynomial; the algorithm is standard. ◻
:::

**Lean formalization:** `DecisionQuotient/Tractability/TreeStructure.lean`

-   InteractionGraph: Definition of the interaction graph.

-   PairwiseUtility: Structure encoding pairwise decomposition.

-   L16: Axiom for treewidth (citing Bodlaender [@bodlaender1993tourist]).

-   L4: Axiom citing CSP algorithms.

-   L12: Paper-specific reduction theorem.

-   L2: Tractability statement combining reduction and algorithm.

## Coordinate Symmetry {#sec:tract-symmetry}

When utility is invariant under coordinate permutations, the effective state space collapses to orbit types.

::: definition
[]{#def:dimensional label="def:dimensional"} A *dimensional state space* with alphabet size $k$ and dimension $d$ is $S = \{0, \ldots, k-1\}^d$, the set of $d$-tuples over a $k$-element alphabet.
:::

::: definition
[]{#def:symmetric label="def:symmetric"} A utility function on a dimensional state space is *symmetric* if it is invariant under coordinate permutations: for all permutations $\sigma \in \mathfrak{S}_d$ and states $s \in S$, $$U(a, s) = U(a, \sigma \cdot s) \quad \text{where } (\sigma \cdot s)_i = s_{\sigma^{-1}(i)}$$
:::

::: definition
[]{#def:orbit-type label="def:orbit-type"} The *orbit type* of a state $s \in \{0, \ldots, k-1\}^d$ is the multiset of its coordinate values: $$\text{orbitType}(s) = \{\!\{s_1, s_2, \ldots, s_d\}\!\}$$ Two states have the same orbit type iff they lie in the same orbit under coordinate permutation.
:::

::: theorem
[]{#thm:orbit-type label="thm:orbit-type"} For dimensional state spaces, two states $s, s'$ have equal orbit types if and only if there exists a coordinate permutation $\sigma$ such that $\sigma \cdot s = s'$.
:::

::: proof
*Proof.* **If:** Permuting coordinates preserves the multiset of values. **Only if:** If $s$ and $s'$ have the same multiset of values, we can construct a permutation mapping each occurrence in $s$ to the corresponding occurrence in $s'$. The Lean proof constructs this permutation explicitly via a fiber bundle argument. ◻
:::

::: theorem
[]{#thm:symmetric-reduction label="thm:symmetric-reduction"} Under symmetric utility, optimal actions depend only on orbit type: $$\text{orbitType}(s) = \text{orbitType}(s') \implies \operatorname{Opt}(s) = \operatorname{Opt}(s')$$ Thus SUFFICIENCY-CHECK reduces to checking pairs with *different* orbit types.
:::

::: proof
*Proof.* If $\text{orbitType}(s) = \text{orbitType}(s')$, then by Theorem [\[thm:orbit-type\]](#thm:orbit-type){reference-type="ref" reference="thm:orbit-type"} there exists $\sigma$ with $\sigma \cdot s = s'$. By symmetry, $U(a, s) = U(a, s')$ for all $a$, so $\operatorname{Opt}(s) = \operatorname{Opt}(s')$.

For sufficiency, we need only check pairs $(s, s')$ agreeing on $I$ with *different* orbit types; same-orbit pairs are automatically equal. ◻
:::

::: theorem
[]{#thm:symmetric-complexity label="thm:symmetric-complexity"} The number of orbit types is bounded by the stars-and-bars count: $$|\text{OrbitTypes}| = \binom{d + k - 1}{k - 1}$$ Thus SUFFICIENCY-CHECK requires at most $\binom{d + k - 1}{k - 1}^2$ orbit-type pair checks, which is polynomial in $d$ for fixed $k$.
:::

::: proof
*Proof.* An orbit type is a multiset of $d$ values from $\{0, \ldots, k-1\}$, equivalently a $k$-tuple of non-negative integers summing to $d$. By stars-and-bars, the count is $\binom{d + k - 1}{k - 1}$.

For fixed $k$, this is $O(d^{k-1})$, polynomial in $d$. ◻
:::

**Lean formalization:** `DecisionQuotient/Tractability/Dimensional.lean`

-   DimensionalStateSpace: Definition of $k$-ary $d$-dimensional state space.

-   SymmetricUtility: Predicate for coordinate-permutation invariance.

-   orbitType: Function computing the orbit type (histogram).

-   L6: **(Core result)** Orbit types equal iff permutation exists.

-   L12: Optimal actions constant on orbits.

-   L10: Reduction to cross-orbit pairs.

-   L14: $O\bigl(\binom{d+k-1}{k-1}^2\bigr)$ bound.

## Practical Implications {#sec:tract-practical}

The six tractable cases correspond to common modeling scenarios:

-   **Bounded actions:** Small action sets (e.g., binary decisions, few alternatives).

-   **Separable utility:** Additive cost models, linear utility functions.

-   **Low tensor rank:** Factored preference models, low-rank approximations.

-   **Tree structure:** Hierarchical decision processes, causal models with tree structure.

-   **Bounded treewidth:** Spatial models, graphical models with limited interaction.

-   **Coordinate symmetry:** Exchangeable features, order-invariant utilities.

For problems given in the succinct encoding without these structural restrictions, the hardness results of Section [\[sec:hardness\]](#sec:hardness){reference-type="ref" reference="sec:hardness"} apply, justifying heuristic approaches.

## Verification Philosophy {#sec:tract-verification}

The Lean formalization follows a consistent pattern for each tractable case:

1.  **Define the structural restriction** as a Lean structure or predicate.

2.  **State axioms for standard algorithms** (DP on trees, CSP on bounded treewidth, tensor contraction) with citations to the literature.

3.  **Prove the paper-specific reduction** showing SUFFICIENCY-CHECK satisfies the preconditions of the standard algorithm.

4.  **Combine** to obtain the tractability statement.

This separation ensures that:

-   Standard results are cited, not re-proved (avoiding redundant mechanization).

-   Paper-specific claims are fully verified (the reductions are where errors hide).

-   The formalization is maintainable (algorithm updates don't require re-verification).

The philosophy: *mechanize what requires mechanical verification for rational belief; cite what is standard*.


# Hardness Transfers and Exact-Simplification Consequences {#sec:engineering-corollaries}

The complexity results of Sections [\[sec:hardness\]](#sec:hardness){reference-type="ref" reference="sec:hardness"} through [\[sec:tractable\]](#sec:tractable){reference-type="ref" reference="sec:tractable"} transfer directly to exact simplification tasks. This section records hardness transfers and exact-simplification consequences of the preceding theorems when the coordinate model is used to represent configuration, feature, or interface choices.

Every claim in this section is model-conditional. The conclusions apply when a design problem is faithfully represented as a decision problem with coordinate structure and when the relevant computational cost is the cost of certifying sufficiency or finding a minimal sufficient set.

## Hardness Transfer to Configuration Simplification

::: proposition
[]{#prop:config-sufficiency label="prop:config-sufficiency"} Let $P=\{p_1,\ldots,p_n\}$ be configuration parameters and let $B$ be a finite set of observable behaviors. Suppose each configuration state determines the subset of behaviors that remain feasible or optimal. Then the question $$\text{``Does parameter subset } I \subseteq P \text{ preserve all decision-relevant behavior?''}$$ is an instance of [Sufficiency-Check]{.smallcaps}.
:::

::: proof
*Proof.* Construct a decision problem $\mathcal{D}=(A,X_1,\ldots,X_n,U)$ by taking actions $A=B$, coordinate domains $X_i$ equal to the domains of the parameters $p_i$, and defining $U(b,s)=1$ exactly when behavior $b$ is realized or admissible under configuration state $s$. Then $\operatorname{Opt}(s)$ is the behavior set induced by $s$, and coordinate subset $I$ is sufficient exactly when agreement on the parameters in $I$ forces agreement on the induced optimal-behavior set. ◻
:::

## Exact-Minimization Impossibility

::: corollary
[]{#cor:no-general-minimizer label="cor:no-general-minimizer"} Assuming $P\neq coNP$, there is no polynomial-time general-purpose procedure that takes an arbitrary succinctly encoded configuration problem and always returns a smallest behavior-preserving parameter subset.
:::

::: proof
*Proof.* Such a procedure would solve [Minimum-Sufficient-Set]{.smallcaps} in polynomial time for arbitrary succinctly encoded instances, contradicting Theorem [\[thm:minsuff-conp\]](#thm:minsuff-conp){reference-type="ref" reference="thm:minsuff-conp"} under the assumption $P\neq coNP$. ◻
:::

::: remark
This does not rule out useful tools. It rules out a fully general exact minimizer with worst-case polynomial guarantees. Domain restrictions of the kind isolated in Section [\[sec:tractable\]](#sec:tractable){reference-type="ref" reference="sec:tractable"} remain viable.
:::

## Over-Specification Under Hardness

::: proposition
[]{#prop:rational-overspecification label="prop:rational-overspecification"} Consider a design process in which:

1.  removing a parameter requires certifying that it is irrelevant,

2.  exact irrelevance certification is charged according to the computational model of Section [\[sec:encoding\]](#sec:encoding){reference-type="ref" reference="sec:encoding"}, and

3.  carrying an extra parameter incurs only linear or otherwise low-order maintenance overhead.

Then, for sufficiently expressive succinctly encoded instances, retaining extra parameters can be an economically rational response to the hardness of exact minimization.
:::

::: proof
*Proof.* By Theorems [\[thm:sufficiency-conp\]](#thm:sufficiency-conp){reference-type="ref" reference="thm:sufficiency-conp"} and [\[thm:minsuff-conp\]](#thm:minsuff-conp){reference-type="ref" reference="thm:minsuff-conp"}, exact sufficiency certification and exact minimization are coNP-complete in the general succinct regime. By Section [\[sec:dichotomy\]](#sec:dichotomy){reference-type="ref" reference="sec:dichotomy"}, this hardness is accompanied by exponential lower bounds under ETH in the linear-support regime. If the cost of carrying extra parameters grows only linearly while exact minimization inherits worst-case exponential cost, then beyond a problem-dependent threshold the latter dominates the former. Under that cost model, over-specification is a rational hedge against intractable exact pruning. ◻
:::

::: remark
The conclusion is not that over-specification is always best. It is that, in the absence of tractable structure, exact minimality need not be a computationally reasonable target.
:::

## Tractability Boundary for Simplification

The tractable regimes of Section [\[sec:tractable\]](#sec:tractable){reference-type="ref" reference="sec:tractable"} explain when exact simplification is computationally realistic. When bounded action sets, separable utility, low tensor rank, tree structure, bounded treewidth, or coordinate symmetry are present, exact certification becomes feasible. Outside such regimes, approximate selection, conservative over-inclusion, and domain-specific simplification strategies are often better aligned with the complexity landscape than unconditional demands for exact minimality.

## Exact-Certification Competence by Regime

::: proposition
[]{#prop:competence-by-regime label="prop:competence-by-regime"} Within the model of this paper, exact certification competence is regime-dependent. In the general succinct regime, exact relevance certification and exact minimization inherit the hardness results of Sections [\[sec:hardness\]](#sec:hardness){reference-type="ref" reference="sec:hardness"} and [\[sec:dichotomy\]](#sec:dichotomy){reference-type="ref" reference="sec:dichotomy"}. In the structured regimes of Section [\[sec:tractable\]](#sec:tractable){reference-type="ref" reference="sec:tractable"}, exact certification becomes available in polynomial time.
:::

::: proof
*Proof.* The negative side is given by Theorems [\[thm:sufficiency-conp\]](#thm:sufficiency-conp){reference-type="ref" reference="thm:sufficiency-conp"} and [\[thm:minsuff-conp\]](#thm:minsuff-conp){reference-type="ref" reference="thm:minsuff-conp"}, together with the ETH-conditioned lower bounds summarized in Section [\[sec:dichotomy\]](#sec:dichotomy){reference-type="ref" reference="sec:dichotomy"}. The positive side is given by the tractability results of Section [\[sec:tractable\]](#sec:tractable){reference-type="ref" reference="sec:tractable"}, which provide explicit polynomial-time certification procedures under structural restrictions. ◻
:::

::: remark
This is the only competence distinction needed in Paper A: not a general theory of reporting or abstention, but a theorem-driven separation between regimes in which exact certification is computationally available and regimes in which it is not.
:::


# Integrity-Competence Impossibility {#sec:integrity-competence}

This section gives an impossibility theorem for reliable exact certification under polynomial budgets in the hard regime.

::: definition
For exact sufficiency on a declared regime $\Gamma$, a certifying solver is a pair $(Q,V)$ where:

-   $Q$ maps each instance to either $\mathsf{ABSTAIN}$ or a candidate verdict with witness,

-   $V$ verifies witnesses in polynomial time (in the declared encoding length).
:::

::: definition
A certifying solver is *integrity-preserving* if every non-abstaining output is accepted by $V$, and every accepted output is correct for the declared exact relation.
:::

::: definition
A certifying solver is *competent* on $\Gamma$ if it is integrity-preserving, non-abstaining on all in-scope instances, and polynomial-budget compliant on all in-scope instances.
:::

::: definition
An *exact reliability claim* on $\Gamma$ is the conjunction of universal non-abstaining coverage, correctness, and polynomial-budget compliance for exact sufficiency.
:::

::: proposition
Integrity and competence are distinct: integrity constrains what can be asserted, while competence adds full coverage under resource bounds.
:::

::: proof
*Proof.* The always-abstain solver is integrity-preserving (it makes no uncertified assertion) but fails competence whenever the in-scope set is non-empty. ◻
:::

::: theorem
Fix a regime $\Gamma$ whose exact-sufficiency decision problem is coNP-complete under the declared encoding. Under $P\neq coNP$, no solver is simultaneously integrity-preserving and competent on $\Gamma$ for exact sufficiency.
:::

::: proof
*Proof.* Assume such a solver $(Q,V)$ exists. We build a polynomial-time decider for exact sufficiency on $\Gamma$.

Given an input instance $x\in \Gamma$, run $Q(x)$. Competence implies that $Q$ never abstains on in-scope instances and always halts within polynomial budget. Let $y$ be the returned verdict with witness. Now run the polynomial-time verifier $V(x,y)$.

Integrity implies two things: every non-abstaining output produced by $Q$ is accepted by $V$, and every accepted output is correct for the declared exact-sufficiency relation. Therefore the combined procedure returns the correct exact verdict on every instance in $\Gamma$ and runs in polynomial time.

This yields a polynomial-time algorithm for a coNP-complete problem on $\Gamma$, hence coNP $\subseteq$ P. Under $P\neq coNP$, this is impossible. Therefore no solver is simultaneously integrity-preserving and competent on $\Gamma$. ◻
:::

::: corollary
Under the assumptions of the theorem, exact reliability claims are impossible on the full hard regime.
:::

::: corollary
In the hard regime, a solver cannot simultaneously maintain integrity, full exact coverage, and polynomial budgets. Operationally: abstain, weaken guarantees, or overclaim.
:::


# Witness-Checking and Approximation Consequences {#sec:witness-duality}

This section isolates the verification bottleneck behind exact sufficiency certification and states the approximation-hardness consequences. Two different theorems are needed. The first is a gap theorem on the shifted hard family: it shows that any uniform factor guarantee on that explicit family already has enough power to decide tautology by thresholding output size. The second is an exact equivalence theorem on a separate set-cover gadget family: it shows that minimum sufficiency and set cover coincide there, so approximation guarantees and impossibility results transfer without loss. Keeping these apart matters. The gap theorem gives an internal obstruction family. The set-cover theorem gives the exact transfer bridge.

## Verification Lower Bound

::: theorem
[]{#thm:witness-lower-bound-4 label="thm:witness-lower-bound-4"} For Boolean decision problems with $n$ coordinates, any sound checker for the empty-set sufficiency core must inspect at least $2^{n-1}$ witness pairs in the worst case.
:::

::: proof
*Proof.* Let $S=\{0,1\}^n$. Empty-set sufficiency is exactly the claim that $\operatorname{Opt}$ is constant on all states.

Partition $S$ into $2^{n-1}$ disjoint witness slots $$\{(0,z),(1,z)\}\quad\text{for }z\in\{0,1\}^{n-1}.$$ Each slot can independently carry a disagreement witness (different optimizer values on its two states).

Consider two instance families:

-   *YES instance*: $\operatorname{Opt}$ constant on all of $S$.

-   *NO$_z$ instance*: identical to YES except slot $z$ has a disagreement.

If a checker inspects fewer than $2^{n-1}$ slots, at least one slot $z^\star$ is uninspected. On all inspected slots, the YES instance and the tailored NO$_{z^\star}$ instance are identical. The checker therefore sees exactly the same transcript on these two inputs and must return the same answer on both.

But YES is an empty-set-sufficient instance, whereas NO$_{z^\star}$ is not. So a common answer cannot be sound on both inputs. Hence any sound worst-case checker must inspect every slot, i.e., at least $2^{n-1}$ witness pairs. ◻
:::

## Approximation Gap on the Shifted Hard Family

For the shifted reduction family used in the mechanization, the optimum support size exhibits a sharp gap: $$\mathrm{OPT}(\varphi)=1 \quad \text{if $\varphi$ is a tautology,}
\qquad
\mathrm{OPT}(\varphi)=n+1 \quad \text{if $\varphi$ is not.}$$ One coordinate acts as a gate: toggling it changes the optimizer even before the formula variables are considered, so optimum size is never $0$. If $\varphi$ is a tautology, every branch behind that gate behaves identically, so all non-gate coordinates become irrelevant and the singleton gate coordinate is sufficient. If $\varphi$ is not a tautology, choose a falsifying assignment $a$. For each formula coordinate $i$, compare the open reference state with the state that inserts $a$ at coordinate $i$: these states agree on all other coordinates but induce different optimal-action sets, so coordinate $i$ is relevant. Together with gate relevance, this forces every coordinate to be present in every sufficient set. That is what creates the exact $1$ versus $n+1$ gap.

::: theorem
Fix $\rho\in\mathbb{N}$. Let $\mathcal{A}$ be any solver that, on every shifted-family instance, returns a sufficient set whose cardinality is within factor $\rho$ of optimum. Then for every formula on $n$ coordinates with $\rho<n+1$, $$\varphi \text{ is a tautology}
\quad\Longleftrightarrow\quad
|\mathcal{A}(\varphi)|\le \rho.$$
:::

::: proof
*Proof.* **Tautology $\Rightarrow$ threshold passes.** If $\varphi$ is a tautology, the optimum support size on the shifted family is $1$, realized by the singleton gate coordinate. A factor-$\rho$ solver therefore returns a sufficient set of size at most $\rho\cdot 1=\rho$, so $|\mathcal{A}(\varphi)|\le \rho$.

**Threshold passes $\Rightarrow$ tautology.** Assume $|\mathcal{A}(\varphi)|\le \rho$ with $\rho<n+1$. If $\varphi$ were not a tautology, then every coordinate would be necessary on the shifted family, so every sufficient set would have size exactly $n+1$. Since $\mathcal{A}(\varphi)$ is sufficient by assumption on $\mathcal{A}$, this would force $|\mathcal{A}(\varphi)|=n+1>\rho$, contradiction.

Therefore $|\mathcal{A}(\varphi)|\le \rho$ holds exactly in the tautology case. ◻
:::

## Counted-Runtime Threshold Decider

::: theorem
Fix $\rho\in\mathbb{N}$. Let $\mathcal{A}$ be any counted polynomial-time factor-$\rho$ solver for the shifted minimum-sufficient-set family. Then the derived procedure $$\varphi \longmapsto \mathbf{1}\!\left\{\,|\mathcal{A}(\varphi)|\le \rho\,\right\}$$ is a counted polynomial-time tautology decider on the gap regime $\rho<n+1$.
:::

::: proof
*Proof.* Correctness is exactly the gap-threshold theorem above: on the regime $\rho<n+1$, the predicate $|\mathcal{A}(\varphi)|\le \rho$ is equivalent to tautology.

For runtime, the derived decider performs two operations: one call to the counted polynomial-time solver $\mathcal{A}$, followed by one cardinality comparison against the fixed threshold $\rho$. The second step contributes only constant additive overhead relative to the solver run. Hence the derived decision procedure remains counted polynomial-time. ◻
:::

## Set-Cover Transfer Boundary

::: theorem
On the mechanized gadget family, a coordinate set is sufficient if and only if the corresponding set family is a cover. In particular, feasible solutions are in bijection and optimum cardinalities coincide exactly.
:::

::: proof
*Proof.* The gadget is constructed so that each universe element $u$ gives two states, $(\mathsf{false},u)$ and $(\mathsf{true},u)$, with different optimal-action sets. A coordinate $i$ distinguishes this pair exactly when the corresponding set covers $u$.

**If $I$ is not a cover**, choose an uncovered universe element $u$. Then every coordinate in $I$ takes the same value on $(\mathsf{false},u)$ and $(\mathsf{true},u)$, but their optimal-action sets differ. So $I$ is not sufficient.

**If $I$ is a cover**, then for every universe element there exists some coordinate in $I$ that separates the two tagged states attached to that element. Hence two states that agree on all coordinates in $I$ cannot disagree in the tag component in a way that changes the optimizer. The optimizer therefore factors through the $I$-projection, so $I$ is sufficient.

This yields an exact bijection between feasible sufficient sets and feasible covers, preserving cardinality. Therefore optimum values coincide on the gadget family. ◻
:::

::: corollary
Any factor guarantee or instance-dependent ratio guarantee for minimum sufficiency on the gadget family induces the same guarantee for set cover on that family. Consequently, any impossibility result proved for set cover on that family transfers immediately to minimum sufficiency.
:::

::: proof
*Proof.* Compose the candidate minimum-sufficiency solver with the gadget translation and then use the exact equivalence theorem above. Because feasible sets correspond bijectively and preserve cardinality, both approximation factors and instance-dependent ratios are unchanged. ◻
:::

The point is therefore two exact reductions with different roles. The shifted family gives a direct gap obstruction: factor approximation already decides tautology there. The set-cover family gives an exact optimization equivalence: sufficiency is cover there. That is why future logarithmic inapproximability theorems for set cover can be imported cleanly once they are available. The current boundary is only that the external set-cover hardness theorem itself is not yet part of this library.


# Simplicity Tax and Hardness Conservation {#sec:simplicity-tax}

This section formalizes the cost of ignoring decision-relevant coordinates. The core object is not an arbitrary partition chosen by the analyst, but the canonical exact-support invariant $R(\mathcal{D})$ identified earlier in the paper. By Proposition [\[prop:minimal-relevant-equiv\]](#prop:minimal-relevant-equiv){reference-type="ref" reference="prop:minimal-relevant-equiv"}, every exact behavior-preserving representation must account for that same set. The point of the present section is to make one consequence explicit: choosing a smaller architectural interface $A_M$ does not destroy omitted exact relevance. It only changes where that relevance must be paid for.

::: definition
For decision problem $\mathcal{D}$, let $$R(\mathcal{D}) := \{i \in \{1,\ldots,n\}: i \text{ is relevant}\}$$ as in Definition [\[def:relevant\]](#def:relevant){reference-type="ref" reference="def:relevant"}.
:::

::: definition
For a model $M$ with native coordinate set $A_M \subseteq \{1,\ldots,n\}$: $$\mathrm{intrinsicDOF}(\mathcal{D}) := |R(\mathcal{D})|,
\qquad
\mathrm{centralDOF}(M,\mathcal{D}) := |R(\mathcal{D}) \cap A_M|.$$
:::

::: definition
$$\mathrm{simplicityTax}(M,\mathcal{D}) := |R(\mathcal{D}) \setminus A_M|.$$
:::

::: theorem
For every finite-coordinate pair $(M,\mathcal{D})$: $$\mathrm{centralDOF}(M,\mathcal{D}) + \mathrm{simplicityTax}(M,\mathcal{D})
=
\mathrm{intrinsicDOF}(\mathcal{D}),$$ hence in particular $$\mathrm{centralDOF}(M,\mathcal{D}) + \mathrm{simplicityTax}(M,\mathcal{D})
\ge
\mathrm{intrinsicDOF}(\mathcal{D}).$$
:::

::: proof
*Proof.* Partition $R(\mathcal{D})$ into handled and unhandled parts: $$R(\mathcal{D}) = (R(\mathcal{D}) \cap A_M)\ \dot\cup\ (R(\mathcal{D}) \setminus A_M),$$ where the union is disjoint by construction. Taking cardinalities gives $$|R(\mathcal{D})|
=
|R(\mathcal{D}) \cap A_M|
+
|R(\mathcal{D}) \setminus A_M|.$$ Substituting the three definitions yields $$\mathrm{intrinsicDOF}(\mathcal{D})
=
\mathrm{centralDOF}(M,\mathcal{D})
+
\mathrm{simplicityTax}(M,\mathcal{D}),$$ which is the claimed equality. The displayed inequality is then immediate. ◻
:::

::: proposition
If each unresolved relevant coordinate induces one unit of per-site external work, then for $N$ decision sites: $$\mathrm{ExternalWork}(N) = N \cdot \mathrm{simplicityTax}(M,\mathcal{D}).$$
:::

::: theorem
Let $H_{\mathrm{central}}>0$ be one-time centralization cost and $\lambda>0$ the per-site cost coefficient. If $\mathrm{simplicityTax}(M,\mathcal{D})>0$, then for $$N > \frac{H_{\mathrm{central}}}{\lambda \cdot \mathrm{simplicityTax}(M,\mathcal{D})},$$ the complete model is cheaper than repeated external handling: $$H_{\mathrm{central}} < \lambda N\cdot \mathrm{simplicityTax}(M,\mathcal{D}).$$
:::

::: proof
*Proof.* Repeated external handling costs $\lambda N\cdot \mathrm{simplicityTax}(M,\mathcal{D})$, while the complete model costs $H_{\mathrm{central}}$ once. The complete model is cheaper exactly when $$H_{\mathrm{central}} < \lambda N\cdot \mathrm{simplicityTax}(M,\mathcal{D}).$$ Because $\lambda>0$ and $\mathrm{simplicityTax}(M,\mathcal{D})>0$, dividing both sides by their product gives $$N > \frac{H_{\mathrm{central}}}{\lambda \cdot \mathrm{simplicityTax}(M,\mathcal{D})}.$$ This is precisely the stated threshold. ◻
:::

::: corollary
Fix a model $M$ and decision problem $\mathcal{D}$. Coordinates in $R(\mathcal{D})\setminus A_M$ are still decision-relevant for exact behavior preservation. Hence any exact deployment must handle them somewhere: by enlarging the central model, by performing local resolution, by demanding extra queries, or by abstaining on some cases. In the hard exact regime, Theorem [\[thm:witness-lower-bound-4\]](#thm:witness-lower-bound-4){reference-type="ref" reference="thm:witness-lower-bound-4"} and Theorem [\[thm:integrity-resource\]](#thm:integrity-resource){reference-type="ref" reference="thm:integrity-resource"} imply that this burden cannot be discharged for free by a polynomial-budget exact certifier.
:::

::: proof
*Proof.* Take any coordinate $i\in R(\mathcal{D})\setminus A_M$. Because $i$ is relevant, there exist witness states whose optimal-action sets differ in a way that cannot be ignored in any exact representation. If a deployment neither models $i$ centrally nor resolves its effect elsewhere, then its exact behavior would factor through the smaller interface $A_M$, making every coordinate outside $A_M$ irrelevant. That contradicts $i\in R(\mathcal{D})$.

So omitted relevant coordinates must be handled somewhere outside the central interface. In the hard exact regime, the witness lower bound shows that exact certification can require exponentially many witness checks, and the integrity-resource theorem shows that polynomial-budget exact competence is unavailable under $P\neq coNP$. Therefore this external burden cannot, in general, be eliminated for free by a polynomial-budget exact certifier. ◻
:::

::: corollary
If $\mathrm{simplicityTax}(M,\mathcal{D})>0$, externalized cost is unbounded in deployment scale $N$.
:::


# Related Work {#sec:related}

## Formalized Complexity Theory

Machine verification of complexity-theoretic results remains sparse compared to other areas of mathematics. We survey existing work and position our contribution.

#### Coq formalizations.

Forster et al. [@forster2019verified] developed a Coq library for computability theory, including undecidability proofs. Their work focuses on computability rather than complexity classes. Kunze et al. [@kunze2019formal] formalized the Cook-Levin theorem in Coq, proving SAT is NP-complete. Our work extends this methodology to coNP-completeness and approximation hardness.

#### Isabelle/HOL.

Nipkow and colleagues formalized substantial algorithm verification in Isabelle [@nipkow2002isabelle], but complexity-theoretic reductions are less developed. Recent work on algorithm complexity [@haslbeck2021verified] provides time bounds for specific algorithms rather than hardness reductions.

#### Lean and Mathlib.

Mathlib's computability library [@mathlib2020] provides primitive recursive functions and basic computability results. Our work extends this to polynomial-time reductions and complexity classes. To our knowledge, this is the first Lean 4 formalization of coNP-completeness proofs and approximation hardness.

#### The verification gap.

Published complexity proofs occasionally contain errors [@lipton2009np]. Machine verification eliminates this uncertainty. Our contribution demonstrates that complexity reductions are amenable to formalization with reasonable effort (49744 lines for the full reduction suite).

## Computational Decision Theory

The complexity of decision-making has been studied extensively. Papadimitriou [@papadimitriou1994complexity] established core results on the complexity of game-theoretic solution concepts. Work on succinctly represented MDP/POMDP settings shows sharp complexity escalation in stochastic and sequential models [@papadimitriou1987mdp; @mundhenk2000mdp]. Our regime results align with that pattern, but focus on a different predicate: coordinate sufficiency of optimal decisions. For a modern treatment of complexity classes, see Arora and Barak [@arora2009computational].

#### Closest prior work and novelty.

Closest to our contribution is the literature on feature selection and model selection hardness, which proves NP-hardness of selecting informative feature subsets and inapproximability for minimum feature sets [@blum1997selection; @amaldi1998complexity]. Those results analyze predictive relevance or compression objectives. We study decision relevance and show coNP-completeness for sufficiency checking, a different quantifier structure (universal verification) with distinct proof techniques and a full hardness/tractability landscape under explicit encoding assumptions, mechanized in Lean 4. The formalization aspect is also novel: prior work establishes hardness on paper, while we provide machine-checked reductions with explicit polynomial bounds.

## Succinct Representations and Regime Separation

Representation-sensitive complexity is established in planning and decision-process theory: classical and compactly represented MDP/planning problems exhibit sharp complexity shifts under different input models [@papadimitriou1987mdp; @mundhenk2000mdp; @littman1998probplanning]. Our explicit-vs-succinct separation aligns with this broader principle.

The distinction in this paper is the object and scope of the classification: we classify *decision relevance* (sufficiency, anchor sufficiency, and minimum sufficient sets) for a fixed decision relation, with theorem-level regime typing and mechanized reduction chains.

## Oracle and Query-Access Lower Bounds

Query-access lower bounds are a standard source of computational hardness transfer [@dobzinski2012query]. Our `[Q_fin]` and `[Q_bool]` results are in this tradition, but specialized to the same sufficiency predicate used throughout the paper: they establish access-obstruction lower bounds while keeping the underlying decision relation fixed across regimes.

## Feature Selection Complexity

In machine learning, feature selection asks which input features are relevant for prediction. Blum and Langley [@blum1997selection] survey the field, noting hardness in general settings. Amaldi and Kann [@amaldi1998complexity] proved that finding minimum feature sets for linear classifiers is NP-hard, and established inapproximability bounds: no polynomial-time algorithm approximates the minimum feature set within factor $2^{\log^{1-\epsilon} n}$ unless NP $\subseteq$ DTIME$(n^{\text{polylog } n})$.

Our results extend this line: the decision-theoretic analog (SUFFICIENCY-CHECK) is coNP-complete, and MINIMUM-SUFFICIENT-SET inherits this hardness. The key insight is that sufficiency checking is "dual" to feature selection; rather than asking which features predict a label, we ask which coordinates determine optimal action. The coNP (rather than NP) classification reflects this duality: insufficiency has short certificates (counterexample state pairs), while sufficiency requires universal verification.

#### Backdoors to tractable classes.

Bessiere et al. [@bessiere2013detecting] show how to exploit CSP instances that are "almost" tractable by computing backdoors: the smallest set of variables whose instantiation yields a tractable subproblem. Computing minimal backdoors is fixed-parameter tractable (FPT) when the tractable subset of relations is known, and W\[2\]-complete otherwise. This complements our dichotomy: when a decision problem has high DOF (many relevant coordinates), their backdoor method identifies the minimal intractable core. The two results together characterize the tractability boundary from different angles.

## Sufficient Statistics

Fisher [@fisher1922mathematical] introduced sufficient statistics: a statistic $T(X)$ is *sufficient* for parameter $\theta$ if the conditional distribution of $X$ given $T(X)$ does not depend on $\theta$. Lehmann and Scheffé [@lehmann1950completeness] characterized minimal sufficient statistics and their uniqueness properties.

Our coordinate sufficiency is the decision-theoretic analog: a coordinate set $I$ is sufficient if knowing $s_I$ determines optimal action, regardless of the remaining coordinates. The parallel is precise:

-   **Statistics:** $T$ is sufficient $\iff$ $P(X | T(X), \theta) = P(X | T(X))$

-   **Decisions:** $I$ is sufficient $\iff$ $\operatorname{Opt}(s) = \operatorname{Opt}(s')$ whenever $s_I = s'_I$

Fisher's factorization theorem provides a characterization; our Theorem [\[thm:minsuff-conp\]](#thm:minsuff-conp){reference-type="ref" reference="thm:minsuff-conp"} shows that *finding* minimal sufficient statistics (in the decision-theoretic sense) is computationally hard.

## Causal Inference and Adjustment Sets

Pearl [@pearl2009causality] and Spirtes et al. [@spirtes2000causation] developed frameworks for identifying causal effects from observational data. A central question is: which variables must be adjusted for to identify a causal effect? The *adjustment criterion* and *back-door criterion* characterize sufficient adjustment sets.

Our sufficiency problem is analogous: which coordinates must be observed to determine optimal action? We conjecture that optimal adjustment set selection is intractable; recent work on the complexity of causal discovery supports this [@chickering2004large].

The connection runs deeper: Shpitser and Pearl [@shpitser2006identification] showed that identifying causal effects is NP-hard in general graphs. Our coNP-completeness result for SUFFICIENCY-CHECK is the decision-theoretic counterpart.

#### Probability of necessity and sufficiency.

Cai et al. [@cai2025pns] formalize the Probability of Necessity and Sufficiency (PNS) for explaining Graph Neural Networks. They argue that a convincing explanation must be *both* necessary and sufficient: necessary explanations alone may be incomplete, while sufficient explanations alone may be non-concise. They model GNNs as Structural Causal Models and optimize a lower bound of PNS. This provides independent confirmation from the neural network explainability literature that necessity and sufficiency must be jointly considered, exactly the perspective underlying our sufficiency-checking framework.

## Minimum Description Length and Kolmogorov Complexity

The Minimum Description Length (MDL) principle [@rissanen1978modeling; @grunwald2007minimum] formalizes model selection as compression: the best model minimizes description length of data plus model. Kolmogorov complexity [@li2008introduction] provides the theoretical background (the shortest program that generates the data).

Our decision quotient connects to this perspective: a coordinate set $I$ is sufficient if it compresses the decision problem without loss. Knowing $s_I$ is as good as knowing $s$ for decision purposes. The minimal sufficient set is the MDL-optimal compression of the decision problem.

The complexity results explain why MDL-based model selection uses heuristics: finding the true minimum description length is uncomputable (Kolmogorov complexity) or intractable (MDL approximations). Our results show the decision-theoretic analog is coNP-complete, intractable but decidable.

## Value of Information

The value of information (VOI) framework [@howard1966information] quantifies the maximum rational payment for information. Our work addresses a different question: not the *value* of information, but the *complexity* of identifying which information has value.

In explicit-state representations, VOI is polynomial to compute in the input size, while identifying which information *to value* (our problem) is coNP-complete. This separation explains why VOI is practical while optimal sensor placement remains heuristic.

## Sensitivity Analysis

Sensitivity analysis asks how outputs change with inputs. Local sensitivity (derivatives) is polynomial; global sensitivity (Sobol indices [@sobol2001global]) requires sampling. Identifying which inputs *matter* for decision-making is our sufficiency problem, which we show is coNP-complete.

This explains why practitioners use sampling-based sensitivity analysis rather than exact methods: exact identification of decision-relevant inputs is intractable. The dichotomy theorem (Theorem [\[thm:dichotomy\]](#thm:dichotomy){reference-type="ref" reference="thm:dichotomy"}) characterizes when sensitivity analysis becomes tractable (logarithmic relevant inputs) versus intractable (linear relevant inputs).

## Model Selection

Statistical model selection (AIC [@akaike1974new], BIC [@schwarz1978estimating], cross-validation [@stone1974cross]) provides practical heuristics for choosing among models. Our results provide theoretical justification: optimal model selection is intractable, so heuristics are necessary. At the level of decision relevance, the lesson is structural rather than heuristic: exact identification of the smallest sufficient representation is hard in general, so restricted model classes and structural assumptions are not optional conveniences but complexity-theoretic necessities.

#### Three-axis integration.

To our knowledge, prior work treats these pillars separately: representation-sensitive hardness, query-access lower bounds, and certifying soundness disciplines. This paper integrates all three for the same decision-relevance object in one regime-typed and machine-checked framework.


# Conclusion

This paper isolates the abstract complexity of decision-relevant information. Once the state space is factored into coordinates, the central question is not merely how to evaluate the objective on a given state, but how to certify which coordinates determine the optimal action.

## Main Results {#main-results .unnumbered}

Within the encoding model of Section [\[sec:encoding\]](#sec:encoding){reference-type="ref" reference="sec:encoding"}, we establish:

-   **Exact certification hardness:** [Sufficiency-Check]{.smallcaps} and [Minimum-Sufficient-Set]{.smallcaps} are coNP-complete, while [Anchor-Sufficiency]{.smallcaps} is $\Sigma_{2}^{P}$-complete

-   **Witness and approximation impossibility:** exact empty-set certification has an exponential witness-checking lower bound; on the shifted hard family, any uniform factor guarantee yields a tautology threshold test; and on a separate explicit gadget family, minimum sufficiency is exactly set cover, so approximation guarantees transfer without loss

-   **Encoding-sensitive boundary:** logarithmic-size sufficient sets admit polynomial-time algorithms in the explicit regime, while linear-size sufficient sets inherit ETH-conditioned exponential lower bounds in the succinct regime

-   **Structured tractability:** bounded actions, separable utility, low tensor rank, tree structure, bounded treewidth, and coordinate symmetry yield polynomial-time certification procedures

Taken together, these results identify a precise source of difficulty: certifying that information may be discarded can be harder than evaluating the full decision rule once all information is already present.

## Reliability and Simplification {#reliability-and-simplification .unnumbered}

The simplification consequences are direct hardness transfers. Configuration simplification is an instance of sufficiency checking, so exact behavior-preserving minimization has no general-purpose polynomial-time worst-case solution unless $P=coNP$. Reliable exact certification is likewise regime-limited: under $P\neq coNP$, no polynomial-budget solver can simultaneously maintain integrity and full exact competence on the hard regime. In the hard succinct regime, conservative over-specification can therefore be a rational response when maintenance overhead is low-order and exact pruning cost is exponential.

## Simplicity Tax {#simplicity-tax .unnumbered}

The simplicity-tax section is not an isolated counting exercise. It is the accounting layer induced by the earlier exact-support theory. Once the canonical relevant-coordinate set $R(\mathcal{D})$ is fixed, any architecture $A_M$ partitions that set into centrally handled and omitted coordinates: $$\mathrm{centralDOF}+\mathrm{simplicityTax}=\mathrm{intrinsicDOF}.$$ The substantive claim is the hardness-conservation consequence: omitted coordinates remain part of the exact decision problem, so they must reappear as local resolution, extra queries, abstentions, or other externalized work. Under the per-site cost model, that externalized cost grows linearly with deployment scale, and under the hard exact regime it cannot be eliminated for free by a polynomial-budget exact certifier.

## Mechanization {#mechanization .unnumbered}

The Lean 4 artifact machine-checks the main reduction constructions, hardness transfers, and tractability statements. The mechanization does not replace the mathematical argument; it certifies that the formal definitions and reductions used by the argument are internally correct and reproducible.

## Outlook {#outlook .unnumbered}

Two immediate directions remain. First, the reduction infrastructure can be further integrated with general-purpose formalized complexity libraries. Second, the abstract theorem stack developed here can serve as a clean upstream citation target for downstream convergence and physical papers without forcing those themes into the structure-and-complexity presentation.


# Lean 4 Proof Listings {#app:lean}

The complete Lean 4 formalization is available at:

::: center
<https://doi.org/10.5281/zenodo.18140966>
:::

The formalization is not a transcription of the paper text. It exposes reusable definitions for reductions, decision problems, hardness transfers, and tractable subcases, so the main complexity claims can be checked mechanically.

## What the Artifact Covers

The appendix supports the non-physical theorem stack of the paper:

-   core decision-problem and sufficiency definitions

-   polynomial-time reduction infrastructure

-   coNP, PP, PSPACE, and $\Sigma_2^P$ hardness reductions

-   shifted-family approximation-gap and counted-runtime threshold constructions

-   ETH-conditioned dichotomy statements

-   cross-regime bridge lemmas (static/stochastic/sequential)

-   tractable subcases under explicit structural assumptions

The point of the artifact is precision. Once the definitions are fixed, Lean verifies that the reductions preserve the intended predicates and that the later theorems invoke the correct formal statements.

## Module Structure

The formalization consists of 189 files organized as follows:

-   `Basic.lean` and `Sufficiency.lean`: core definitions for decision problems, projections, optimizer maps, and sufficiency

-   `AlgorithmComplexity.lean` and `PolynomialReduction.lean`: polynomial-time reduction infrastructure and composition

-   `Reduction.lean` and `Hardness/`: the TAUTOLOGY, SET-COVER, ETH, approximation-gap, and $\Sigma_2^P$ reduction stack

-   `StochasticSequential*.lean`: PP/PSPACE regime definitions, hardness, and hierarchy bridges

-   `QueryComplexity.lean`, `Dichotomy.lean`, and `ComplexityMain.lean`: query lower bounds and summary complexity statements

-   `Tractability/`: bounded actions, separable utility, tensor-rank, tree-structured, treewidth, and symmetry arguments

## Key Theorems

::: theorem
[]{#thm:poly-compose label="thm:poly-compose"} Polynomial-time reductions compose to polynomial-time reductions.
:::

    theorem PolyReduction.comp_exists
        (f : PolyReduction A B) (g : PolyReduction B C) :
        exists h : PolyReduction A C,
          forall a, h.reduce a = g.reduce (f.reduce a)

::: theorem
The canonical reduction used in Theorem [\[thm:sufficiency-conp\]](#thm:sufficiency-conp){reference-type="ref" reference="thm:sufficiency-conp"} is mechanized and proves that the empty set is sufficient exactly when the source formula is a tautology.
:::

    theorem reduction_correct
        (φ : QBF) :
        empty_sufficient (reduce φ) ↔ QBF.eval φ

## Verification Status

-   Total lines: 49744

-   Theorems/lemmas: 2152

-   Files: 189

-   Status: All proofs compile with 0 `sorry`




---

## Machine-Checked Proofs

All theorems are formalized in Lean 4:
- Location: `docs/papers/paper4_decision_quotient/proofs/`
- Lines: 49744
- Theorems: 2152
- `sorry` placeholders: 0
