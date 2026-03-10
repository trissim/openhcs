# Paper: Decision Quotient: A Regime-Sensitive Complexity Theory of Exact Relevance Certification

**Status**: Computational Complexity-ready | **Lean**: 21142 lines, 985 theorems

---

## Abstract

_Abstract not available._

# Introduction {#sec:introduction}

Which coordinates of a decision problem can be hidden without changing the optimal action, and how hard is it to certify that claim exactly? We expected one answer. We found three fundamentally different ones. We study this exact-certification question across static, stochastic, and sequential regimes. For a decision problem $\mathcal{D}=(A,S,U)$ with $S = X_1 \times \cdots \times X_n$, a coordinate set $I$ is sufficient when agreement on $I$ forces agreement on the optimal-action set: $$s_I = s'_I \implies \operatorname{Opt}(s) = \operatorname{Opt}(s').$$ This is the basic problem of exact relevance certification: how much of the state must remain visible to preserve the decision boundary, and what it takes to certify that the rest is irrelevant. The question resembles subset selection, but exact certification has a different quantifier structure and, as the paper shows, a different complexity landscape.

This is a certification problem rather than a forward-evaluation problem. Evaluating $U(a,s)$ or $\operatorname{Opt}(s)$ on a fully specified state is one task; proving that a family of coordinates is irrelevant requires ruling out every counterexample pair of states. That universal structure drives the static regime. A first structural point is worth stating up front: [Minimum-Sufficient-Set]{.smallcaps} looks existential-universal on its face, but in the static regime it collapses because a set is sufficient exactly when it contains every relevant coordinate. So the minimum sufficient set is just the relevant-coordinate set. At the same time, the optimizer quotient is not an arbitrary definition: it is the canonical coarsest abstraction that preserves optimal-action distinctions. [Anchor-Sufficiency]{.smallcaps} does not collapse in the same way, because the anchor assignment is itself part of the existential choice and must then satisfy a universal fiberwise condition.

This distinction also explains why exactness matters. When a system replaces exact relevance certification with a heuristic simplification, the omitted relevance does not disappear; it is shifted into residual uncertainty that must be handled elsewhere in the system. We refer to that externalized burden as the *simplicity tax*. The results of this paper show that this tax is not merely rhetorical: exact simplification can be computationally expensive, but inexact simplification does not erase decision-relevant structure; it relocates it.

#### Informal interpretive guide.

The formal objects studied here admit a compact intuitive reading. A relevant coordinate is a point of *leverage*: changing it can change the decision. The optimizer quotient isolates the decision *signal*: it is the coarsest abstraction that still preserves optimal-action distinctions. The static collapse expresses *inevitability*: in the static regime, the optimal simplification is forced by the structure of relevance rather than discovered by search. The simplicity tax is *liability*: omitted exact relevance does not disappear, but must be handled elsewhere in the system. At the task level, sufficiency is *observation*, anchor sufficiency is *discovery*, and sequential certification is *exposure*: omitted information may appear harmless now while still creating future liability elsewhere in the system.

The paper's central claim is that this is not a collection of isolated hardness facts. Once the same exact-certification question is tracked across richer input models, the difficulty escalates in a systematic way. Static certification is governed by counterexample exclusion, stochastic certification by majority-style conditional comparisons, and sequential certification by policy preservation across evolving information states. Read as a complexity matrix, adding probability lifts exact certification from coNP to PP, and adding time lifts it again to PSPACE. The optimizer quotient packages the same issue structurally: it is the coarsest abstraction of the state space that preserves optimal-action distinctions.

Throughout, we work with finite coordinate-structured decision problems under explicit encoding regimes. Section [\[sec:encoding\]](#sec:encoding){reference-type="ref" reference="sec:encoding"} fixes the computational model used by every complexity claim.

## Contributions

The paper is best read as one theorem package with four linked takeaways: a structural static collapse, a regime-sensitive complexity matrix, a sharp phase transition between tractable and hard exact certification, and a trilemma for exact certifiers in the hard regime.

1.  **The Static Collapse Theorem.** In the static regime, the apparent quantifier complexity of exact minimization simplifies sharply: sufficient sets are exactly the supersets of the relevant-coordinate set. This collapses [Minimum-Sufficient-Set]{.smallcaps} to a coNP problem rather than a genuine $\Sigma_2^P$ search problem, while [Anchor-Sufficiency]{.smallcaps} remains $\Sigma_2^P$-complete because its existential anchor choice does not collapse.

2.  **Unified regime-sensitive complexity matrix.** Inside one common model of finite coordinate-structured decision problems, we formalize coordinate sufficiency, relevance, minimal sufficient sets, anchor sufficiency, and the optimizer quotient, and we show that the resulting exact certification problems organize into a single regime hierarchy: static [Sufficiency-Check]{.smallcaps} and [Minimum-Sufficient-Set]{.smallcaps} are coNP-complete, static [Anchor-Sufficiency]{.smallcaps} is $\Sigma_2^P$-complete, stochastic sufficiency is PP-complete, and sequential sufficiency is PSPACE-complete, with PP-hard and PSPACE-hard stochastic/sequential minimum and anchor queries.

3.  **Sharp phase transition between tractable and hard exact certification.** We prove an explicit-state versus succinct dichotomy: logarithmic-size sufficient sets admit polynomial-time algorithms in the explicit regime, while linear-size sufficient sets in the succinct regime inherit ETH-conditioned exponential lower bounds. We also isolate six structured tractable subcases under standard restrictions.

4.  **The trilemma of exact certification.** We transfer the hierarchy to exact simplification, isolate a witness-checking lower bound and approximation obstructions, and prove an integrity-competence impossibility theorem: no exact certifier can be simultaneously sound, complete, and polynomial-budgeted on the hard regime. This trilemma is a fundamental reliability constraint, and we show that omitted exact relevance is externalized rather than eliminated.

5.  **Lean-certified reduction and finite-decision core.** The main reductions, size-bounded hardness packages, exact finite boolean deciders, counted finite-search procedures, and explicit-state step-counting wrappers for the regime-typed query predicates are mechanized in Lean 4. For the stochastic and sequential completeness claims, the artifact fully internalizes the reduction side together with the finite decision procedures; the standard PP/PSPACE membership arguments remain part of the paper text rather than end-to-end Turing-machine witness proofs in the current repository.

## Quantifier Structure

The complexity gap comes from the structure of the question. Insufficiency has short witnesses: two states that agree on the proposed coordinates but induce different optimal-action sets. Sufficiency, by contrast, asks for the absence of every such witness. So the object-level task is forward evaluation on a fully specified state, while the meta-level task is universal verification over counterexample pairs. This difference drives the coNP classification and explains why relevance identification is not just another forward-evaluation problem.

The static regime also contains the paper's main structural simplification. Although minimum sufficiency can be written with an outer existential quantifier over coordinate sets, the existential layer collapses once sufficiency is characterized as containment of all relevant coordinates. The stochastic and sequential regimes do not simplify in the same way: conditioning introduces majority-style comparisons over fibers, while temporal evolution introduces policy-level dependence on hidden information.

## Formalization Snapshot

::: center
  **Metric**                                 **Value**
  ---------------------------------------- -----------
  Lines of Lean 4 backing cited content          21142
  Theorems/lemmas in cited-content files           985
  Proof files backing cited content                 62
  `sorry` count in cited-content files               0
:::

The artifact builds with `lake build`. It certifies the reduction-correctness lemmas, size-bounded hardness packages, exact finite deciders, counted-search witnesses, tractability statements, and explicit-state step-counting wrappers cited in the main text. The PP/PSPACE membership directions remain paper proofs rather than full Turing-machine formalizations.

## Paper Structure

Section [\[sec:formal-setup\]](#sec:formal-setup){reference-type="ref" reference="sec:formal-setup"} introduces the abstract setup and the encoding contract. Sections [\[sec:hardness\]](#sec:hardness){reference-type="ref" reference="sec:hardness"}, [\[sec:stochastic\]](#sec:stochastic){reference-type="ref" reference="sec:stochastic"}, [\[sec:sequential\]](#sec:sequential){reference-type="ref" reference="sec:sequential"}, and [\[sec:regime-hierarchy\]](#sec:regime-hierarchy){reference-type="ref" reference="sec:regime-hierarchy"} develop the paper's core theorem package: the Static Collapse Theorem, the regime-sensitive complexity matrix, and the resulting hierarchy from static to stochastic to sequential certification. Section [\[sec:dichotomy\]](#sec:dichotomy){reference-type="ref" reference="sec:dichotomy"} gives the encoding-sensitive dichotomy and the sharp phase transition for exact certification, and Section [\[sec:tractable\]](#sec:tractable){reference-type="ref" reference="sec:tractable"} records tractable subcases. Section [\[sec:engineering-corollaries\]](#sec:engineering-corollaries){reference-type="ref" reference="sec:engineering-corollaries"} then collects the main implications for exact simplification, the trilemma of exact certification, witness checking, approximation, and externalized relevance. Section [\[sec:related\]](#sec:related){reference-type="ref" reference="sec:related"} situates the work within formalized complexity theory and decision-theoretic relevance.

## Artifact Availability

The standalone Lean proof artifact and submission snapshot are archived at:

::: center
<https://doi.org/10.5281/zenodo.18140966>
:::

The proofs build with the Lean toolchain specified in `lean-toolchain`.


# Formal Setup {#sec:formal-setup}

This section fixes the abstract vocabulary used throughout the paper. We work with finite decision problems whose state spaces factor into coordinates, and we isolate the minimal terminology needed to state the later complexity results. The same setup will support the paper's three core regime classifications, the encoding-sensitive boundary, and the later consequence sections.

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

::: theorem
[]{#thm:quotient-universal label="thm:quotient-universal"} Let $Q_{\mathcal{D}}=S/{\sim_{\operatorname{Opt}}}$ be the decision quotient with quotient map $\pi:S\to Q_{\mathcal{D}}$. For any surjective abstraction $\phi:S\to T$ such that $$\phi(s)=\phi(s') \implies \operatorname{Opt}(s)=\operatorname{Opt}(s'),$$ there exists a unique map $\psi:T\to Q_{\mathcal{D}}$ such that $$\pi = \psi \circ \phi.$$
:::

::: proof
*Proof.* Because $\phi$ preserves optimizer values on its fibers, any two states identified by $\phi$ are decision-equivalent. So each fiber of $\phi$ is contained in a single $\sim_{\operatorname{Opt}}$-class.

For each $t \in T$, choose any $s \in S$ with $\phi(s)=t$ and define $\psi(t):=\pi(s)$. This is well defined because any two such choices lie in the same $\sim_{\operatorname{Opt}}$-class. By construction, $(\psi\circ\phi)(s)=\pi(s)$ for every $s\in S$, so $\pi=\psi\circ\phi$. Uniqueness follows from surjectivity of $\phi$: if $\psi'$ also satisfies $\pi=\psi'\circ\phi$, then every $t\in T$ has the form $t=\phi(s)$ and therefore $$\psi(t)=\psi(\phi(s))=\pi(s)=\psi'(\phi(s))=\psi'(t).$$ Hence $\psi=\psi'$. ◻
:::

::: remark
[]{#rem:quotient-canonical label="rem:quotient-canonical"} The optimizer quotient is therefore the coarsest abstraction of the state space that preserves optimal-action distinctions. Any surjective abstraction that is strictly coarser must erase some decision-relevant distinction.
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


# Static Regime Complexity {#sec:hardness}

This section studies the static regime, where the input is a finite coordinate-structured decision problem under the succinct encoding of Section [\[sec:encoding\]](#sec:encoding){reference-type="ref" reference="sec:encoding"}. This is the baseline certification setting: no stochastic averaging, no temporal dynamics, only the exact question of whether a candidate coordinate set preserves the optimizer map. The main structural fact appears already here: [Minimum-Sufficient-Set]{.smallcaps} looks like an existential-universal search problem, but it collapses once sufficiency is characterized by relevance containment. The only genuinely higher-level static query is the anchor variant, where an existentially chosen fiber must satisfy a universal constancy condition.

## Problem Definitions

::: problem
**Input:** Decision problem $\mathcal{D}=(A,X_1,\ldots,X_n,U)$ and coordinate set $I \subseteq \{1,\ldots,n\}$.\
**Question:** Is $I$ sufficient for $\mathcal{D}$?
:::

::: problem
**Input:** Decision problem $\mathcal{D}=(A,X_1,\ldots,X_n,U)$ and integer $k$.\
**Question:** Does there exist a sufficient set $I$ with $|I| \le k$?
:::

::: problem
**Input:** Decision problem $\mathcal{D}=(A,X_1,\ldots,X_n,U)$ and fixed coordinate set $I$.\
**Question:** Does there exist an assignment $\alpha$ to $I$ such that $\operatorname{Opt}(s)$ is constant on the fiber $\{s : s_I=\alpha\}$?
:::

## The Static Collapse

At first glance, [Minimum-Sufficient-Set]{.smallcaps} seems to have the form $$\exists I\; \forall s,s'\in S:
\quad s_I=s'_I \implies \operatorname{Opt}(s)=\operatorname{Opt}(s').$$ If that syntax reflected the true structure of the problem, one might expect a $\Sigma_2^P$ classification. But Proposition [\[prop:minimal-relevant-equiv\]](#prop:minimal-relevant-equiv){reference-type="ref" reference="prop:minimal-relevant-equiv"} removes the outer search layer: a set is sufficient exactly when it contains every relevant coordinate, so the minimum sufficient set is simply the relevant-coordinate set. We refer to this phenomenon as the *static collapse*. Static exact minimization is therefore a relevance-counting problem, not a genuinely alternating search problem.

That collapse is specific to minimum sufficiency. [Anchor-Sufficiency]{.smallcaps} still asks the algorithm to choose an assignment $\alpha$ and then certify a universal condition on the induced fiber, so its existential layer cannot be absorbed into a structural characterization of the instance.

## coNP-Completeness Results

The first two results show that exact certification is already harder than pointwise evaluation. A counterexample to sufficiency is short, but certifying that no such counterexample exists is a universal statement over state pairs. The collapse discussed above explains why the same coNP behavior governs both direct sufficiency checking and exact minimization.

::: theorem
[]{#thm:sufficiency-conp label="thm:sufficiency-conp"} SUFFICIENCY-CHECK is coNP-complete.
:::

::: proof
*Proof.* For membership, Proposition [\[prop:insufficiency-counterexample\]](#prop:insufficiency-counterexample){reference-type="ref" reference="prop:insufficiency-counterexample"} gives a polynomial witness $(s,s')$ with $s_I=s'_I$ and $\operatorname{Opt}(s)\ne\operatorname{Opt}(s')$ whenever $I$ is not sufficient, so the complement is in NP.

For hardness, reduce TAUTOLOGY to the empty-set instance. Given formula $\varphi(x_1,\ldots,x_n)$, build a two-action decision problem where one action tracks $\varphi$ and the other is a baseline action. Then $I=\emptyset$ is sufficient iff $\varphi$ is a tautology, because empty-set sufficiency is exactly constancy of $\operatorname{Opt}$ (Proposition [\[prop:empty-sufficient-constant\]](#prop:empty-sufficient-constant){reference-type="ref" reference="prop:empty-sufficient-constant"}). The reduction is polynomial in formula size. ◻
:::

::: theorem
[]{#thm:minsuff-conp label="thm:minsuff-conp"} In the static regime, the apparent existential layer in [Minimum-Sufficient-Set]{.smallcaps} collapses to relevance containment. Consequently, [Minimum-Sufficient-Set]{.smallcaps} is coNP-complete.
:::

::: proof
*Proof.* Proposition [\[prop:minimal-relevant-equiv\]](#prop:minimal-relevant-equiv){reference-type="ref" reference="prop:minimal-relevant-equiv"} identifies the minimum sufficient set with the relevant-coordinate set. The question "$|I^*|\le k$?" therefore becomes the question whether at most $k$ coordinates are relevant, which is a coNP check over relevance witnesses. CoNP-hardness follows from the $k=0$ slice, which is exactly SUFFICIENCY-CHECK with $I=\emptyset$. ◻
:::

## Sigma-2-P Completeness

Anchor sufficiency adds one more layer of choice that the previous collapse does not remove: the algorithm may select a fiber, but must then certify that the chosen fiber is uniformly decision-preserving. That existential-then-universal pattern is exactly what raises the complexity class.

::: theorem
[]{#thm:anchor-sigma2p label="thm:anchor-sigma2p"} ANCHOR-SUFFICIENCY is $\Sigma_2^P$-complete.
:::

::: proof
*Proof.* Membership follows directly from the quantifier pattern: $$\exists\alpha\;\forall s\in S:\ (s_I=\alpha)\Rightarrow \operatorname{Opt}(s)=\operatorname{Opt}(s_\alpha).$$ For hardness, reduce from $\exists\forall$-SAT: existential variables encode the anchor assignment, and universal variables range over the residual coordinates. Utility values are set so that fiberwise constancy holds exactly when the source formula is true. ◻
:::

## Certificate-Size Lower Bound

The next theorem sharpens the static picture. It is not only that exact certification is hard in the complexity-class sense; even a checker restricted to the empty-set core may be forced to inspect exponentially many witness locations in the worst case.

::: theorem
[]{#thm:witness-checking-duality label="thm:witness-checking-duality"} For Boolean state spaces $S=\{0,1\}^n$, any sound checker for the empty-set sufficiency core must inspect at least $2^{n-1}$ witness pairs in the worst case.
:::

::: proof
*Proof.* Empty-set sufficiency asks whether $\operatorname{Opt}$ is constant on all $2^n$ states. A sound refutation-complete checker must be able to separate constant maps from maps that differ on some hidden antipodal partition. An adversary can place the first disagreement in any of $2^{n-1}$ independent pair slots; if fewer than $2^{n-1}$ slots are inspected, two instances remain indistinguishable to the checker but have opposite truth values. Therefore at least $2^{n-1}$ pair checks are necessary. ◻
:::

#### Mechanization note.

The TAUTOLOGY reduction stack, the coNP and $\Sigma_2^P$ classifications, and the witness-budget lower-bound core are indexed by their inline handles.

#### Explicit-state search upper bounds.

Under the explicit-state step-counting model, static sufficiency and static anchor sufficiency are also wrapped as abstract [P]{.smallcaps} predicates on inputs carrying a certified state budget. The artifact proves these upper bounds via counted searches with quadratic and linear state-space dependence respectively, and it also packages the static sufficiency search as an explicit correctness-plus-step-bound witness and as part of the unified finite-search summary theorem.

This completes the baseline regime. The next section keeps the same exact certification question but replaces pairwise counterexample exclusion by conditional expected-utility comparison, which is why the complexity lifts from the static coNP/$\Sigma_2^P$ picture to PP.


# Stochastic Regime Complexity {#sec:stochastic}

We now add a distribution over states. This does not weaken the certification problem; it changes its form. Sufficiency is no longer a pointwise optimizer-preservation condition but a statement about conditional optimal decisions under averaging, and that shift is enough to move the problem to PP. The underlying reason is structural: once utilities are compared through conditional expectations, certification becomes a weighted counting problem rather than a pure counterexample-exclusion problem. Static hardness comes from ruling out bad pairs; stochastic hardness comes from deciding which side of a majority-style threshold a conditional expectation lies on.

## Stochastic Decision Problems

::: definition
[]{#def:stochastic-decision-problem label="def:stochastic-decision-problem"} A stochastic decision problem is a tuple $$\mathcal{D}_S=(A,X_1,\ldots,X_n,U,P),$$ where $P\in\Delta(S)$ is a distribution on $S=X_1\times\cdots\times X_n$.
:::

::: definition
[]{#def:stochastic-sufficient label="def:stochastic-sufficient"} A coordinate set $I$ is *stochastically sufficient* if conditioning on $s_I$ never changes the induced optimal-action set: $$\arg\max_{a\in A}\,\mathbb{E}[U(a,s)\mid s_I]
=
\arg\max_{a\in A}\,\mathbb{E}[U(a,s)]$$ for all admissible $I$-fibers.
:::

::: problem
[]{#prob:stochastic-sufficiency label="prob:stochastic-sufficiency"} **Input:** $\mathcal{D}_S=(A,X_1,\ldots,X_n,U,P)$ and $I\subseteq\{1,\ldots,n\}$.\
**Question:** Is $I$ stochastically sufficient?
:::

::: definition
[]{#def:stochastic-anchor-sufficient label="def:stochastic-anchor-sufficient"} A coordinate set $I$ is *stochastically anchor-sufficient* if there exists an anchor fiber such that the conditional optimal-action set on that fiber is a singleton and every state agreeing with the anchor on $I$ induces that same singleton conditional optimum.
:::

::: problem
[]{#prob:stochastic-anchor-sufficiency label="prob:stochastic-anchor-sufficiency"} **Input:** $\mathcal{D}_S=(A,X_1,\ldots,X_n,U,P)$ and $I\subseteq\{1,\ldots,n\}$.\
**Question:** Is $I$ stochastically anchor-sufficient?
:::

::: problem
[]{#prob:stochastic-minimum-sufficient label="prob:stochastic-minimum-sufficient"} **Input:** $\mathcal{D}_S=(A,X_1,\ldots,X_n,U,P)$ and $k\in\mathbb{N}$.\
**Question:** Does there exist a stochastically sufficient coordinate set $I$ with $|I|\le k$?
:::

## PP-Completeness

The stochastic regime preserves the same underlying question, which coordinates can be ignored without changing the decision, but asks it after conditioning and expectation have been introduced. The resulting majority-style comparisons are exactly where PP enters. For the empty-information case, one is already comparing aggregate mass of satisfying versus nonsatisfying states. More generally, every fiberwise exact-certification check asks whether one conditional expected utility dominates another across all admissible fibers, which is the natural counting analogue of the static universal check.

::: theorem
[]{#thm:stochastic-pp label="thm:stochastic-pp"} STOCHASTIC-SUFFICIENCY-CHECK is PP-complete.
:::

::: proof
*Proof.* For membership, deciding whether one conditional expected utility dominates another is a majority-style weighted counting comparison, hence in PP.

For hardness, reduce MAJSAT. Given Boolean formula $\varphi$, take $S=\{0,1\}^n$, uniform $P$, actions $\{\mathrm{accept},\mathrm{reject}\}$, and utilities $$U(\mathrm{accept},s)=\mathbf{1}[\varphi(s)],\qquad U(\mathrm{reject},s)=1/2,$$ with $I=\emptyset$. Then empty-set stochastic sufficiency is equivalent to deciding whether $\mathbb{E}[\varphi]\ge 1/2$, i.e., MAJSAT. ◻
:::

::: proposition
[]{#prop:stochastic-anchor-refinement label="prop:stochastic-anchor-refinement"} If $I$ is stochastically sufficient, then $I$ is stochastically anchor-sufficient.
:::

::: proof
*Proof.* If every admissible $I$-fiber has a unique conditional optimum, then any chosen anchor fiber has a unique conditional optimum, and every state agreeing with that anchor on $I$ induces the same singleton conditional optimum. ◻
:::

::: theorem
[]{#thm:stochastic-anchor-hard label="thm:stochastic-anchor-hard"} STOCHASTIC-ANCHOR-SUFFICIENCY-CHECK is PP-hard.
:::

::: proof
*Proof.* Reduce MAJSAT. Given Boolean formula $\varphi$ on $n\geq 1$ variables, take $S=\{0,1\}^n$, uniform $P$, $I=\emptyset$, and three actions $\{\mathrm{accept},\mathrm{hold}_L,\mathrm{hold}_R\}$. Set $$U(\mathrm{accept},s)=\mathbf{1}[\varphi(s)],\qquad
U(\mathrm{hold}_L,s)=U(\mathrm{hold}_R,s)=\frac12-2^{-(n+1)}.$$ If at least half of all assignments satisfy $\varphi$, then $\mathrm{accept}$ is the unique optimal action on the empty-information fiber, so the instance is anchor-sufficient. If fewer than half satisfy $\varphi$, then $\mathrm{hold}_L$ and $\mathrm{hold}_R$ tie above $\mathrm{accept}$, so no singleton anchor optimum exists. Thus MAJSAT many-one reduces to the stochastic anchor query. ◻
:::

::: theorem
[]{#thm:stochastic-minimum-hard label="thm:stochastic-minimum-hard"} The stochastic minimum-sufficient-set query is PP-hard.
:::

::: proof
*Proof.* Reduce MAJSAT to the $k=0$ slice. In the same three-action gadget used above, there exists a stochastically sufficient set of size at most $0$ iff the empty set itself is stochastically sufficient. This empty-set equivalence for the gadget is certified in the artifact, so MAJSAT many-one reduces to STOCHASTIC-MINIMUM-SUFFICIENT-SET. ◻
:::

::: proposition
[]{#prop:stochastic-explicit-search label="prop:stochastic-explicit-search"} For finite instances, the artifact contains counted exhaustive-search procedures for stochastic sufficiency, stochastic anchor sufficiency, and stochastic minimum sufficiency. Their certified step bounds are respectively $O(|S|)$, $O(|S||A|)$, and $O(2^n)$.
:::

::: proof
*Proof.* The sufficiency search scans for a violating state whose fiber-optimal set is non-singleton; the anchor search scans over candidate anchor-state/action pairs; and the minimum query scans the subset lattice. Each procedure admits a counted boolean search formulation with both a correctness theorem and an explicit step bound. ◻
:::

#### Relevance boundary.

The stochastic query is fiber-indexed: fixing a candidate information set $I$ induces a derived decision problem whose optimizer is the conditional fiber optimizer. In the current formalization this induced problem is always $I$-sufficient , but its relevance notion is therefore also indexed by $I$. For that reason, the paper does not assert a single global relevance-set characterization for stochastic minimal sufficiency analogous to the static and sequential regimes.

## Tractable Subcases

::: proposition
[]{#prop:stochastic-tractable label="prop:stochastic-tractable"} The stochastic sufficiency problem is polynomial-time solvable under each of the following restrictions:

1.  product distributions $P=\bigotimes_i P_i$,

2.  bounded support $|\mathrm{supp}(P)|\le k$,

3.  structured families where conditional expectations are computable in polynomial time (e.g. log-concave models with oracle access).
:::

::: proof
*Proof.* In each case, conditional expected utilities can either be computed exactly or reduced to a polynomial number of marginal comparisons. ◻
:::

## Bridge from Static

These final propositions show that moving from static to stochastic reasoning is not merely a reformulation. Distributional conditioning can genuinely destroy or preserve sufficiency depending on the structure of the input model.

::: proposition
[]{#prop:static-not-stochastic label="prop:static-not-stochastic"} There exist instances where $I$ is sufficient in the static sense but not stochastically sufficient once a distribution $P$ is fixed.
:::

::: proof
*Proof.* Choose utilities where optimizer ties are harmless pointwise but are broken in expectation after conditioning. Then static fibers preserve $\operatorname{Opt}$, while conditional expectations change the optimal action. ◻
:::

::: proposition
[]{#prop:static-to-stochastic-product label="prop:static-to-stochastic-product"} If $P$ factorizes as a product distribution and $I$ is statically sufficient, then $I$ is stochastically sufficient.
:::

::: proof
*Proof.* Under product factorization, conditioning on $I$ does not introduce cross-coordinate dependence outside the sufficient projection, so optimizer preservation transfers from pointwise to expectation form. ◻
:::

The stochastic regime therefore gives the middle layer of the paper's hierarchy: conditioning already raises exact certification beyond the static setting by turning exact certification into a counting comparison problem, but the full temporal escalation is deferred until sequential dynamics are added in the next section.


# Sequential Regime Complexity {#sec:sequential}

We now move to sequential settings with transitions and observations. Here the same certification question is asked at the policy level: can one suppress some coordinates of the evolving information state without changing the optimal policy? This is where the complexity reaches PSPACE. The structural reason is that hidden information can matter later even when it appears irrelevant now. Certification must therefore compare policies across temporally unfolding contingencies rather than compare one-shot optimizers or conditional expectations. In succinct models, that policy-preservation task naturally supports the same kind of alternating contingency that drives PSPACE hardness in planning and quantified-logic reductions.

## Sequential Decision Problems

::: definition
[]{#def:sequential-decision-problem label="def:sequential-decision-problem"} A sequential decision problem is a tuple $$\mathcal{D}_{\mathrm{seq}}=(A,X_1,\ldots,X_n,U,T,O),$$ with state space $S=X_1\times\cdots\times X_n$, transition kernel $T:A\times S\to\Delta(S)$, and observation model $O:S\to\Delta(\Omega)$.
:::

::: definition
[]{#def:sequential-sufficient label="def:sequential-sufficient"} Coordinate set $I$ is *sequentially sufficient* if the optimal policy using full observation history equals the optimal policy obtained when histories are restricted to $I$-coordinates.
:::

::: problem
[]{#prob:sequential-sufficiency label="prob:sequential-sufficiency"} **Input:** $\mathcal{D}_{\mathrm{seq}}$ and $I\subseteq\{1,\ldots,n\}$.\
**Question:** Is $I$ sequentially sufficient?
:::

::: definition
[]{#def:sequential-anchor-sufficient label="def:sequential-anchor-sufficient"} A coordinate set $I$ is *sequentially anchor-sufficient* if there exists an anchor state whose $I$-agreement class preserves the optimal action set throughout that class.
:::

::: problem
[]{#prob:sequential-anchor-sufficiency label="prob:sequential-anchor-sufficiency"} **Input:** $\mathcal{D}_{\mathrm{seq}}$ and $I\subseteq\{1,\ldots,n\}$.\
**Question:** Is $I$ sequentially anchor-sufficient?
:::

::: problem
[]{#prob:sequential-minimum-sufficient label="prob:sequential-minimum-sufficient"} **Input:** $\mathcal{D}_{\mathrm{seq}}$ and $k\in\mathbb{N}$.\
**Question:** Does there exist a sequentially sufficient coordinate set $I$ with $|I|\le k$?
:::

::: proposition
[]{#prop:sequential-minimal-relevant label="prop:sequential-minimal-relevant"} For any minimal sequentially sufficient set $I$, a coordinate lies in $I$ if and only if it is relevant for the underlying decision boundary.
:::

::: proof
*Proof.* In the current formalization, sequential sufficiency is defined as sufficiency of the underlying one-step optimizer map on the state space. The static minimal-set/relevance equivalence therefore transports directly to the sequential setting. ◻
:::

## PSPACE-Completeness

The sequential regime is the strongest of the three because hidden information can matter not only for the current optimizer but for the future evolution of the decision problem. The same relevance question therefore becomes a policy-comparison problem over dynamic information states. Static certification asks whether bad state pairs exist; stochastic certification asks whether conditional averages cross a threshold; sequential certification asks whether omitted coordinates can change the value of future contingent choices. That added temporal contingency is exactly what the TQBF reduction exploits.

::: theorem
[]{#thm:sequential-pspace label="thm:sequential-pspace"} SEQUENTIAL-SUFFICIENCY-CHECK is PSPACE-complete.
:::

::: proof
*Proof.* For membership, finite-horizon policy comparison for POMDP-style models is decidable in polynomial space by dynamic programming over belief and state summaries.

For hardness, reduce TQBF. Encode alternating quantifiers as alternating controlled and adversarial transitions, and let the terminal reward evaluate the quantified formula. Then full-history sufficiency holds exactly when the quantified instance is true. ◻
:::

::: proposition
[]{#prop:sequential-anchor-refinement label="prop:sequential-anchor-refinement"} If $I$ is sequentially sufficient, then $I$ is sequentially anchor-sufficient.
:::

::: proof
*Proof.* If optimal-action sets are preserved for every pair of states agreeing on $I$, then fixing any anchor state immediately yields preservation across its entire $I$-agreement class. ◻
:::

::: theorem
[]{#thm:sequential-anchor-hard label="thm:sequential-anchor-hard"} The sequential anchor query is PSPACE-hard.
:::

::: proof
*Proof.* Reuse the TQBF reduction for sequential sufficiency with $I=\emptyset$. At empty information, sequential anchor sufficiency is equivalent to global preservation of the optimal-action set, so the same construction yields $$\mathrm{TQBF}(q)
\iff
\mathrm{SEQUENTIAL\mbox{-}ANCHOR\mbox{-}SUFFICIENCY\mbox{-}CHECK}(\mathrm{reduceTQBF}(q),\emptyset).$$ Hence TQBF many-one reduces to the sequential anchor query. ◻
:::

::: theorem
[]{#thm:sequential-minimum-hard label="thm:sequential-minimum-hard"} The sequential minimum-sufficient-set query is PSPACE-hard.
:::

::: proof
*Proof.* Reduce TQBF to the $k=0$ slice. There exists a sequentially sufficient set of size at most $0$ iff the empty set itself is sequentially sufficient. The same construction yields PSPACE-hardness for SEQUENTIAL-MINIMUM-SUFFICIENT-SET. ◻
:::

::: proposition
[]{#prop:sequential-explicit-search label="prop:sequential-explicit-search"} For finite instances, the development contains counted exhaustive-search procedures for sequential sufficiency, sequential anchor sufficiency, and sequential minimum sufficiency. Their certified step bounds are respectively $O(|S|^2)$, $O(|S|)$, and $O(2^n)$.
:::

::: proof
*Proof.* The sufficiency search scans for an agreeing pair of states with different optimal sets; the anchor search scans over candidate anchor states; and the minimum query scans the subset lattice. Each procedure admits a counted boolean search formulation with both a correctness theorem and an explicit step bound. ◻
:::

## Tractable Subcases

::: proposition
[]{#prop:sequential-tractable label="prop:sequential-tractable"} The sequential sufficiency problem is polynomial-time solvable under each of the following restrictions:

1.  full observability (MDP case),

2.  bounded horizon,

3.  tree-structured transition systems,

4.  deterministic transitions.
:::

::: proof
*Proof.* Each restriction reduces policy search to dynamic programming over a polynomially bounded representation. ◻
:::

## Bridge from Stochastic

The bridge from the stochastic regime is strict. A one-shot distribution can hide no temporal contingency; once dynamics are introduced, latent information can become decision-relevant later even when it appears irrelevant at time zero.

::: proposition
[]{#prop:stochastic-not-sequential label="prop:stochastic-not-sequential"} There exist instances where $I$ is sufficient in the stochastic one-shot sense but fails to be sufficient once temporal dynamics are introduced.
:::

::: proof
*Proof.* Construct a process in which the same one-step expected optimizer is preserved under $I$, but hidden-state memory affects future information states and therefore optimal policies. The temporal dependence blocks one-shot sufficiency transfer. ◻
:::

With the static, stochastic, and sequential classifications in place, the next section consolidates them into a single regime hierarchy and makes the escalation explicit: counterexample exclusion, then counting comparison, then policy preservation over time.


# Regime Hierarchy {#sec:regime-hierarchy}

The static, stochastic, and sequential formulations are the paper's central organizing object. The same exact certification question is posed in three increasingly expressive models, and the resulting complexity classes separate accordingly. Read as a complexity matrix, the addition of probability lifts exact certification from coNP to PP, and the addition of time lifts it again to PSPACE.

## Complexity Matrix

::: center
  Regime          Sufficiency        Minimum             Anchor
  ------------ ----------------- --------------- -----------------------
  Static         coNP-complete    coNP-complete   $\Sigma_2^P$-complete
  Stochastic      PP-complete        PP-hard             PP-hard
  Sequential    PSPACE-complete    PSPACE-hard         PSPACE-hard
:::

This table is the paper's compact summary. Adding probability lifts exact certification from coNP to PP, and adding time lifts it again to PSPACE.

The stochastic and sequential anchor and minimum variants are classified with hardness lower bounds only; the explicit-state finite-search deciders in the artifact provide upper bounds under the explicit regime but do not constitute general PP or PSPACE membership proofs under the succinct encoding.

::: theorem
[]{#thm:regime-main label="thm:regime-main"} Under the encoding model of Section [\[sec:encoding\]](#sec:encoding){reference-type="ref" reference="sec:encoding"}, exact relevance certification admits a strict regime-sensitive hierarchy. In the static regime, sufficiency and minimum sufficiency are coNP-complete and anchor sufficiency is $\Sigma_2^P$-complete. In the stochastic regime, sufficiency is PP-complete and the minimum and anchor queries are PP-hard. In the sequential regime, sufficiency is PSPACE-complete and the minimum and anchor queries are PSPACE-hard. Under standard complexity assumptions, this yields strict inclusions $$\text{static} \subset \text{stochastic} \subset \text{sequential}.$$
:::

::: proof
*Proof.* The matrix entries are exactly Theorems [\[thm:sufficiency-conp\]](#thm:sufficiency-conp){reference-type="ref" reference="thm:sufficiency-conp"}, [\[thm:minsuff-conp\]](#thm:minsuff-conp){reference-type="ref" reference="thm:minsuff-conp"}, [\[thm:anchor-sigma2p\]](#thm:anchor-sigma2p){reference-type="ref" reference="thm:anchor-sigma2p"}, [\[thm:stochastic-pp\]](#thm:stochastic-pp){reference-type="ref" reference="thm:stochastic-pp"}, [\[thm:stochastic-anchor-hard\]](#thm:stochastic-anchor-hard){reference-type="ref" reference="thm:stochastic-anchor-hard"}, [\[thm:stochastic-minimum-hard\]](#thm:stochastic-minimum-hard){reference-type="ref" reference="thm:stochastic-minimum-hard"}, [\[thm:sequential-pspace\]](#thm:sequential-pspace){reference-type="ref" reference="thm:sequential-pspace"}, [\[thm:sequential-anchor-hard\]](#thm:sequential-anchor-hard){reference-type="ref" reference="thm:sequential-anchor-hard"}, and [\[thm:sequential-minimum-hard\]](#thm:sequential-minimum-hard){reference-type="ref" reference="thm:sequential-minimum-hard"}. The strict inclusions then follow from the standard class separations invoked in Theorems [\[thm:static-stochastic-strict\]](#thm:static-stochastic-strict){reference-type="ref" reference="thm:static-stochastic-strict"} and [\[thm:stochastic-sequential-strict\]](#thm:stochastic-sequential-strict){reference-type="ref" reference="thm:stochastic-sequential-strict"}. ◻
:::

## Strict Inclusions

::: theorem
[]{#thm:static-stochastic-strict label="thm:static-stochastic-strict"} Under standard assumptions (in particular $P\ne coNP$), the stochastic regime strictly extends the static regime in computational complexity.
:::

::: proof
*Proof.* Static sufficiency checking is coNP-complete (Theorem [\[thm:sufficiency-conp\]](#thm:sufficiency-conp){reference-type="ref" reference="thm:sufficiency-conp"}), while stochastic sufficiency checking is PP-complete (Theorem [\[thm:stochastic-pp\]](#thm:stochastic-pp){reference-type="ref" reference="thm:stochastic-pp"}). Under the stated assumptions, this yields strict separation in regime difficulty. ◻
:::

::: theorem
[]{#thm:stochastic-sequential-strict label="thm:stochastic-sequential-strict"} Under standard assumptions (in particular $P\ne PP$), the sequential regime strictly extends the stochastic regime in computational complexity.
:::

::: proof
*Proof.* Stochastic sufficiency is PP-complete (Theorem [\[thm:stochastic-pp\]](#thm:stochastic-pp){reference-type="ref" reference="thm:stochastic-pp"}), while sequential sufficiency is PSPACE-complete (Theorem [\[thm:sequential-pspace\]](#thm:sequential-pspace){reference-type="ref" reference="thm:sequential-pspace"}). The class gap yields strict regime inclusion under the stated assumptions. ◻
:::

## Integrity-Resource Bounds

The hierarchy has a methodological consequence as well: as one moves upward in the hierarchy, exact certification becomes less compatible with fixed polynomial resource budgets.

::: proposition
[]{#prop:abstention-frontier label="prop:abstention-frontier"} For polynomial-time exact certifiers that abstain whenever they cannot certify correctness within budget, the set of instances requiring abstention expands along $$\text{static} \to \text{stochastic} \to \text{sequential}.$$
:::

::: proof
*Proof.* As the decision predicate moves from coNP to PP to PSPACE hardness, exact worst-case certification requires strictly stronger resources. Fixed polynomial resources therefore force abstention on a larger instance family in higher regimes. ◻
:::


# Encoding Dichotomy and ETH Lower Bounds {#sec:dichotomy}

The regime hierarchy identifies where exact relevance certification sits in the standard complexity landscape. This section extracts the complementary algorithmic message: under the encodings of Section [\[sec:encoding\]](#sec:encoding){reference-type="ref" reference="sec:encoding"}, there is a genuine dichotomy between explicit-state instances whose true relevant support is small and succinct instances whose relevant support is extensive. The former admit direct algorithms; the latter inherit exponential lower bounds under ETH. The practical takeaway is a sharp phase transition: logarithmic relevant support can still be certified exactly in polynomial time under explicit encodings, while linear relevant support under succinct encodings already forces ETH-conditioned exponential cost.

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
[]{#cor:phase-transition label="cor:phase-transition"} The tractable-hard boundary for exact certification is sharp at the logarithmic scale (with respect to the encodings in Section [\[sec:encoding\]](#sec:encoding){reference-type="ref" reference="sec:encoding"}):

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

This dichotomy explains why exact simplification can be straightforward on some instance families and computationally prohibitive on others. When the true sufficient support is tiny and the state space is explicit, exhaustive certification can be polynomial. When the true support is extensive and the problem is given succinctly, the same certification task inherits a worst-case $2^{\Omega(n)}$ lower bound under ETH in addition to the class-level coNP hardness statement.

#### Interpretation.

The ETH chain composes with the structural rank viewpoint introduced earlier. When relevance is extensive, the same objects that drive the regime hierarchy also force the exponential obstruction in the succinct model. In that sense, the dichotomy theorem complements the hierarchy rather than standing apart from it: the hierarchy locates exact certification in the standard class landscape, while the dichotomy explains when explicit structure can still make exact certification feasible.

::: remark
The ETH lower bound is stated in the succinct circuit encoding of Section [\[sec:encoding\]](#sec:encoding){reference-type="ref" reference="sec:encoding"}, where the utility function $U: A \times S \to \mathbb{R}$ is represented by a Boolean circuit computing $\mathbf{1}[U(a,s) > \theta]$ for threshold comparisons. In this model:

-   The input size is the circuit size $m$, not the state space size $|S| = 2^n$

-   A 3-SAT formula with $n$ variables and $c$ clauses yields a circuit of size $O(n + c)$

-   The reduction preserves instance size up to constant factors: $m_{\text{out}} \leq 3 \cdot m_{\text{in}}$

This linear size preservation is essential for ETH transfer. In the explicit enumeration model (where $S$ is given as a list), the reduction would blow up the instance size exponentially, precluding ETH-based lower bounds. The circuit model is standard in fine-grained complexity and matches practical representations of decision problems.
:::


# Tractable Special Cases {#sec:tractable}

We distinguish the encodings of Section [\[sec:encoding\]](#sec:encoding){reference-type="ref" reference="sec:encoding"}. The tractability results below state the model assumption explicitly.

::: theorem
[]{#thm:tractable label="thm:tractable"} SUFFICIENCY-CHECK is polynomial-time solvable in the following cases:

1.  **Bounded actions (explicit-state encoding):** The input contains the full utility table over $A \times S$ with $|A| \leq k$ for constant $k$. Complexity: $O(|S|^2 \cdot k^2)$.

2.  **Separable utility (rank 1):** $U(a, s) = f(a) + g(s)$. Complexity: $O(1)$ (trivially sufficient).

3.  **Low tensor rank:** The utility admits a rank-$R$ decomposition $U(a,s) = \sum_{r=1}^R w_r \cdot f_r(a) \cdot \prod_i g_{ri}(s_i)$. Complexity: $O(|A| \cdot R \cdot n)$.

4.  **Tree-structured dependencies:** Coordinates are ordered such that $s_i$ depends only on $(s_1, \ldots, s_{i-1})$. Complexity: $O(n \cdot |A| \cdot k_{\max})$ where $k_{\max} = \max_i |X_i|$.

5.  **Bounded treewidth:** The interaction graph has treewidth $\leq w$ and utility factors over edges. Complexity: $O(n \cdot k^{w+1})$ via standard CSP algorithms.

6.  **Coordinate symmetry:** Utility is invariant under coordinate permutations. Complexity: $O\bigl(\binom{d+k-1}{k-1}^2\bigr)$ orbit-type pairs.
:::

The following subsections provide the formal definitions and reduction theorems for each tractable subcase.

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
*Proof.* **If:** Permuting coordinates preserves the multiset of values. **Only if:** If $s$ and $s'$ have the same multiset of values, we can construct a permutation mapping each occurrence in $s$ to the corresponding occurrence in $s'$. ◻
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

## Practical Implications {#sec:tract-practical}

The six tractable cases correspond to common modeling scenarios:

-   **Bounded actions:** Small action sets (e.g., binary decisions, few alternatives).

-   **Separable utility:** Additive cost models, linear utility functions.

-   **Low tensor rank:** Factored preference models, low-rank approximations.

-   **Tree structure:** Hierarchical decision processes, causal models with tree structure.

-   **Bounded treewidth:** Spatial models, graphical models with limited interaction.

-   **Coordinate symmetry:** Exchangeable features, order-invariant utilities.

For problems given in the succinct encoding without these structural restrictions, the hardness results of Section [\[sec:hardness\]](#sec:hardness){reference-type="ref" reference="sec:hardness"} apply, justifying heuristic approaches.


# Implications for Exact Certification {#sec:engineering-corollaries}

The regime hierarchy matters for more than class labels. The same exact question, which coordinates can be omitted without changing the decision boundary, reappears when one tries to simplify models, build reliable certifiers, check witnesses, approximate exact minimizers, or compress a central interface. This section collects those consequences in one place. The common lesson is that exact relevance is expensive to certify, subject to a trilemma in the hard regime, and easy to displace rather than remove.

## Exact Simplification and Over-Specification

::: proposition
[]{#prop:config-sufficiency label="prop:config-sufficiency"} Let $P=\{p_1,\ldots,p_n\}$ be configuration parameters and let $B$ be a finite set of observable behaviors. Suppose each configuration state determines the subset of behaviors that remain feasible or optimal. Then the question $$\text{``Does parameter subset } I \subseteq P \text{ preserve all decision-relevant behavior?''}$$ is an instance of [[Sufficiency-Check]{.smallcaps}]{.nodecor}.
:::

::: proof
*Proof.* Construct a decision problem $\mathcal{D}=(A,X_1,\ldots,X_n,U)$ by taking actions $A=B$, coordinate domains $X_i$ equal to the domains of the parameters $p_i$, and defining $U(b,s)=1$ exactly when behavior $b$ is realized or admissible under configuration state $s$. Then $\operatorname{Opt}(s)$ is the behavior set induced by $s$, and coordinate subset $I$ is sufficient exactly when agreement on the parameters in $I$ forces agreement on the induced optimal-behavior set. ◻
:::

::: corollary
[]{#cor:no-general-minimizer label="cor:no-general-minimizer"} Assuming $P\neq coNP$, there is no polynomial-time general-purpose procedure that takes an arbitrary succinctly encoded configuration problem and always returns a smallest behavior-preserving parameter subset.
:::

::: proof
*Proof.* Such a procedure would solve [Minimum-Sufficient-Set]{.smallcaps} in polynomial time for arbitrary succinctly encoded instances, contradicting Theorem [\[thm:minsuff-conp\]](#thm:minsuff-conp){reference-type="ref" reference="thm:minsuff-conp"} under the assumption $P\neq coNP$. ◻
:::

::: proposition
[]{#prop:rational-overspecification label="prop:rational-overspecification"} Consider a design process in which:

1.  removing a parameter requires certifying that it is irrelevant,

2.  exact irrelevance certification is charged according to the computational model of Section [\[sec:encoding\]](#sec:encoding){reference-type="ref" reference="sec:encoding"}, and

3.  carrying an extra parameter incurs only linear or otherwise low-order maintenance overhead.

Then, for sufficiently expressive succinctly encoded instances, retaining extra parameters can minimize total cost under the stated cost model.
:::

::: proof
*Proof.* By Theorems [\[thm:sufficiency-conp\]](#thm:sufficiency-conp){reference-type="ref" reference="thm:sufficiency-conp"} and [\[thm:minsuff-conp\]](#thm:minsuff-conp){reference-type="ref" reference="thm:minsuff-conp"}, exact sufficiency certification and exact minimization are coNP-complete in the general succinct regime. By Section [\[sec:dichotomy\]](#sec:dichotomy){reference-type="ref" reference="sec:dichotomy"}, this hardness is accompanied by exponential lower bounds under ETH in the linear-support regime. If the cost of carrying extra parameters grows only linearly while exact minimization inherits worst-case exponential cost, then beyond a problem-dependent threshold the latter dominates the former. Under that cost model, retaining extra parameters minimizes total cost. ◻
:::

::: remark
This does not rule out useful simplification tools. It rules out a fully general exact minimizer with worst-case polynomial guarantees. The structured regimes isolated in Section [\[sec:tractable\]](#sec:tractable){reference-type="ref" reference="sec:tractable"} remain viable precisely because they restrict the source of hardness.
:::

## Regime-Limited Exact Certification

::: proposition
[]{#prop:competence-by-regime label="prop:competence-by-regime"} Within the model of this paper, exact certification competence is regime-dependent. In the general succinct regime, exact relevance certification and exact minimization inherit the hardness results of Sections [\[sec:hardness\]](#sec:hardness){reference-type="ref" reference="sec:hardness"} and [\[sec:dichotomy\]](#sec:dichotomy){reference-type="ref" reference="sec:dichotomy"}. In the structured regimes of Section [\[sec:tractable\]](#sec:tractable){reference-type="ref" reference="sec:tractable"}, exact certification becomes available in polynomial time.
:::

::: proof
*Proof.* The negative side is given by Theorems [\[thm:sufficiency-conp\]](#thm:sufficiency-conp){reference-type="ref" reference="thm:sufficiency-conp"} and [\[thm:minsuff-conp\]](#thm:minsuff-conp){reference-type="ref" reference="thm:minsuff-conp"}, together with the ETH-conditioned lower bounds summarized in Section [\[sec:dichotomy\]](#sec:dichotomy){reference-type="ref" reference="sec:dichotomy"}. The positive side is given by the tractability results of Section [\[sec:tractable\]](#sec:tractable){reference-type="ref" reference="sec:tractable"}, which provide explicit polynomial-time certification procedures under structural restrictions. ◻
:::

The next three subsections sharpen that basic regime distinction in different directions: exact reliability under polynomial budgets, witness-checking and approximation limits, and the cost of compressing a central interface while leaving exact relevance unresolved.


## Integrity, Competence, and the Trilemma of Exact Certification {#sec:integrity-competence}

Once exact certification is hard, the natural systems question is not only whether the predicate can be computed, but whether a solver can stay exact, non-abstaining, and polynomial-budgeted on the whole hard regime. The theorem below shows that those three goals are incompatible. We refer to this incompatibility as the *trilemma of exact certification*.

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
[]{#thm:integrity-resource label="thm:integrity-resource"} Fix a regime $\Gamma$ whose exact-sufficiency decision problem is coNP-complete under the declared encoding. Under $P\neq coNP$, no solver is simultaneously integrity-preserving and competent on $\Gamma$ for exact sufficiency.
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
In the hard regime, no exact certifier can simultaneously be sound, complete on all in-scope instances, and polynomial-budgeted. Operationally, one of three concessions is unavoidable: abstain on some instances, weaken the guarantee, or risk overclaiming.
:::


## Witness and Approximation Limits {#sec:witness-duality}

Hardness does not disappear when one weakens exact minimization to witness checking or approximation. The first theorem shows that even the empty-set core can force exponential witness inspection. The remaining results then show two complementary approximation obstructions: a direct gap obstruction on the shifted hard family and an exact optimization transfer on the set-cover gadget family.

### Witness-Checking Lower Bound

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

### Approximation Gap on the Shifted Family

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

::: theorem
Fix $\rho\in\mathbb{N}$. Let $\mathcal{A}$ be any counted polynomial-time factor-$\rho$ solver for the shifted minimum-sufficient-set family. Then the derived procedure $$\varphi \longmapsto \mathbf{1}\!\left\{\,|\mathcal{A}(\varphi)|\le \rho\,\right\}$$ is a counted polynomial-time tautology decider on the gap regime $\rho<n+1$.
:::

::: proof
*Proof.* Correctness is exactly the gap-threshold theorem above: on the regime $\rho<n+1$, the predicate $|\mathcal{A}(\varphi)|\le \rho$ is equivalent to tautology.

For runtime, the derived decider performs two operations: one call to the counted polynomial-time solver $\mathcal{A}$, followed by one cardinality comparison against the fixed threshold $\rho$. The second step contributes only constant additive overhead relative to the solver run. Hence the derived decision procedure remains counted polynomial-time. ◻
:::

### Set-Cover Transfer Boundary

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

The two reductions serve different purposes. The shifted family gives a direct gap obstruction: factor approximation already decides tautology there. The set-cover family gives an exact optimization equivalence: sufficiency is cover there. Together they show that weakening exact minimization to witness checking or approximation does not dissolve the core obstruction.


## Externalized Relevance and Simplicity Tax {#sec:simplicity-tax}

The final implication concerns architectural compression. The main point is interpretive rather than algebraic: shrinking a central interface does not remove exact relevance; it relocates the unresolved part of that relevance to some other part of the system. The simplicity tax is the name for that externalized burden.

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

::: proposition
For every finite-coordinate pair $(M,\mathcal{D})$, $$\mathrm{centralDOF}(M,\mathcal{D}) + \mathrm{simplicityTax}(M,\mathcal{D})
=
\mathrm{intrinsicDOF}(\mathcal{D}).$$
:::

::: proof
*Proof.* Immediate from the partition $R(\mathcal{D}) = (R(\mathcal{D}) \cap A_M) \;\dot\cup\; (R(\mathcal{D}) \setminus A_M)$. ◻
:::

In other words, central simplification changes where exact relevance is handled, not how much exact relevance the deployment must ultimately account for.

::: proposition
If each unresolved relevant coordinate induces one unit of per-site external work, then for $N$ decision sites: $$\mathrm{ExternalWork}(N) = N \cdot \mathrm{simplicityTax}(M,\mathcal{D}).$$
:::

::: proof
*Proof.* Each site pays one unit for each unresolved relevant coordinate, so the total cost is the number of sites times the number of unresolved relevant coordinates. ◻
:::

::: theorem
Let $H_{\mathrm{central}}>0$ be one-time centralization cost and $\lambda>0$ the per-site cost coefficient. If $\mathrm{simplicityTax}(M,\mathcal{D})>0$, then for $$N > \frac{H_{\mathrm{central}}}{\lambda \cdot \mathrm{simplicityTax}(M,\mathcal{D})},$$ the complete model is cheaper than repeated external handling: $$H_{\mathrm{central}} < \lambda N\cdot \mathrm{simplicityTax}(M,\mathcal{D}).$$
:::

::: proof
*Proof.* Repeated external handling costs $\lambda N\cdot \mathrm{simplicityTax}(M,\mathcal{D})$, while the complete model costs $H_{\mathrm{central}}$ once. Solving the strict inequality $H_{\mathrm{central}} < \lambda N\cdot \mathrm{simplicityTax}(M,\mathcal{D})$ for $N$ gives the stated threshold. ◻
:::

::: corollary
Fix a model $M$ and decision problem $\mathcal{D}$. Coordinates in $R(\mathcal{D})\setminus A_M$ are still decision-relevant for exact behavior preservation. Hence any exact deployment must handle them somewhere: by enlarging the central model, by performing local resolution, by demanding extra queries, or by abstaining on some cases. In the hard exact regime, Theorem [\[thm:witness-lower-bound-4\]](#thm:witness-lower-bound-4){reference-type="ref" reference="thm:witness-lower-bound-4"} and Theorem [\[thm:integrity-resource\]](#thm:integrity-resource){reference-type="ref" reference="thm:integrity-resource"} imply that this burden cannot be discharged for free by a polynomial-budget exact certifier.
:::

::: proof
*Proof.* Take any coordinate $i\in R(\mathcal{D})\setminus A_M$. Because $i$ is relevant, there exist witness states whose optimal-action sets differ in a way that cannot be ignored in any exact representation. If a deployment neither models $i$ centrally nor resolves its effect elsewhere, then its exact behavior would factor through the smaller interface $A_M$, making every coordinate outside $A_M$ irrelevant. That contradicts $i\in R(\mathcal{D})$.

So omitted relevant coordinates must be handled somewhere outside the central interface. In the hard exact regime, the witness lower bound shows that exact certification can require exponentially many witness checks, and the integrity-resource theorem shows that polynomial-budget exact competence is unavailable under $P\neq coNP$. Therefore this external burden cannot, in general, be eliminated for free by a polynomial-budget exact certifier. ◻
:::


# Related Work {#sec:related}

## Formalized Complexity Theory and Mechanized Reductions

Machine verification of complexity-theoretic arguments remains substantially less mature than formal verification in algebra, analysis, or standard algorithmics. Forster et al. [@forster2019verified] developed certified machine models and computability infrastructure in Coq, and Kunze et al. [@kunze2019formal] formalized major proof-theoretic components in Coq as well. In Isabelle/HOL, much of the mature work has centered on verified algorithms and resource analysis for concrete procedures rather than on families of hardness reductions [@nipkow2002isabelle; @haslbeck2021verified]. Lean 4 and Mathlib provide increasingly capable foundations for mechanized finite mathematics and computation [@mathlib2020; @moura2021lean4], but reusable reduction suites for coNP/$\Sigma_2^P$/PP/PSPACE-style arguments remain relatively rare.

The present artifact is intended as a problem-specific certification layer within that broader program. It does not claim to settle formal complexity theory in full generality; instead, it internalizes the reduction-correctness lemmas, size-bounded hardness packages, exact finite deciders, counted-search procedures, tractability wrappers, and explicit-state upper-bound interfaces needed for the decision-relevance predicates studied here. In that sense the contribution is closest to a certified reduction-and-decision core for one theorem family rather than to a general-purpose complexity library.

## Rough Sets, Reducts, and Attribute Reduction

The closest classical neighbor is rough-set theory and its large literature on reducts and attribute reduction [@pawlak1982rough; @jensen2004semantics; @jensen2008rough]. That literature asks when a smaller attribute set preserves discernibility or decision-table structure, and it has developed both exact and heuristic methods for reduct computation. There is an obvious family resemblance to the static regime of the present paper: both settings ask which coordinates can be removed without losing decision distinctions.

The difference is not merely terminological. Rough-set reducts are typically formulated relative to indiscernibility relations or decision tables, whereas the object here is the optimizer map of a decision problem and the induced exact preservation of optimal-action sets. That shift matters computationally. In the present formulation, insufficiency has short counterexample witnesses, sufficiency is universal over such witnesses, and minimum sufficiency collapses to relevance containment in the static regime. The paper's static-collapse theorem, regime hierarchy, and exact-certification trilemma are therefore not direct corollaries of the reduct literature, even though that literature is an important intellectual precursor and deserves explicit credit as a nearby tradition.

The same generosity is due to the broader feature-selection and subset-selection literature in machine learning and combinatorial optimization [@blum1997selection; @amaldi1998complexity]. Those works study sparse predictive representations, informative feature subsets, and the computational cost of choosing them. Our setting differs in targeting exact preservation of the optimal-action correspondence rather than prediction loss, classifier complexity, or approximate empirical utility. Still, the shared concern is clear: identifying what information matters and what can be safely omitted.

## Decision-Theoretic Sufficiency, Informativeness, and Value of Information

There is also a strong conceptual connection to statistical decision theory and the theory of informativeness. Classical sufficient statistics, value-of-information analysis, and comparison of experiments ask when one information structure is at least as useful for decision making as another [@blackwell1953equivalent; @savage1954foundations; @raiffa1961applied; @howard1966information]. That tradition studies when information can be compressed, compared, or discarded without harming decisions. The optimizer quotient and the exact sufficiency predicate in this paper should be read against that background.

At the same time, the present work studies a different layer of the problem. The focus here is not statistical estimation, asymptotic identifiability, or expected utility under a fixed experiment class. It is the computational certification problem of proving that a coordinate projection preserves the optimal-action correspondence exactly. The connection to classical decision theory is therefore one of ancestry and motivation rather than direct theorem transfer. The paper owes that literature recognition for posing, in older language, the same high-level question: what information is genuinely needed for a decision?

## Abstraction, Bisimulation, and Homomorphisms in Planning and Reinforcement Learning

The stochastic and sequential parts of the paper sit near a second major literature: exact abstraction in Markov decision processes, POMDPs, bisimulation-based aggregation, and MDP homomorphisms [@papadimitriou1987mdp; @littman1998probplanning; @mundhenk2000mdp; @givan2003equivalence; @ravindran2004algebraic]. This body of work studies when state spaces can be aggregated while preserving value or policy structure, and it has established many of the representation-sensitive complexity phenomena that make richer stochastic and sequential models computationally difficult.

That literature is structurally very close to the present paper and should be credited as such. The main difference is that our predicate is coordinate sufficiency for preserving optimal-action sets under an explicit coordinate-hiding operation. We do not study arbitrary abstract state maps, approximate bisimulation metrics, or value-function approximation guarantees. Instead, we ask for exact certification that a chosen coordinate interface is enough. The contribution here is therefore not abstraction theory in general, but a regime-sensitive complexity theory for one exact decision-relevance predicate tracked coherently across static, stochastic, and sequential settings.

## Explanations, Anchors, and Minimal Decision Rules

The paper should also acknowledge adjacent work in explainable AI and local rule-based explanations, especially anchor-style explanations [@ribeiro2018anchors]. Readers will naturally notice the word "anchor" and map it to that literature. The resemblance is real: both settings isolate a restricted condition under which a local piece of information suffices to support a decision or prediction. More broadly, explanation methods based on local rules, counterfactuals, or minimal sufficient subsets share the same informal ambition of identifying a small decision-preserving core.

The difference is again exactness and scope. Anchor explanations in XAI are typically local, model-agnostic, approximate, or precision-calibrated. Our anchor-sufficiency problem is an exact certification problem inside a fully specified decision model, and its complexity is governed by an existential choice followed by a universal fiberwise verification condition. So the paper is not an explanation method, but it is in direct conversation with explanation work that asks what small local structure is enough to preserve a decision.

There is an even closer logical neighbor in recent work on reasons behind decisions, prime-implicant style explanations, and exact decision justifications [@darwiche2023reasons]. That literature is concerned with exact structural reasons for a decision outcome and therefore lies much nearer to the present paper than generic post-hoc explanation methods do. The overlap is substantial at the conceptual level: both ask for exact small structures that preserve a decision. The difference is that our objects are coordinate sufficiency, minimum sufficiency, and anchor sufficiency inside coordinate-structured decision problems, and the paper's main contribution is a regime-sensitive complexity classification of these predicates rather than a logical semantics of reasons alone.

## Abduction, Diagnosis, and Exact Explanatory Cores

The minimum and anchor queries should also be read against the classical literature on abduction and diagnosis [@reiter1987diagnosis; @eiter1995abduction]. Reiter-style diagnosis studies which hidden fault sets explain an observation, while logic-based abduction studies the complexity of finding explanatory hypotheses sufficient for a target consequence. Those frameworks share an important structural feature with the present work: a move from verification to explanatory search typically raises complexity because one is no longer merely checking a candidate structure but searching for one.

That kinship is especially relevant for the paper's anchor query. The existential anchor choice is one reason the static anchor problem remains $\Sigma_2^P$-complete even when minimum sufficiency collapses in the static regime. In that sense the anchor problem behaves more like explanatory discovery than like plain verification. The exact predicate is different---we certify preservation of optimal-action sets under coordinate hiding rather than explanatory entailment of a formula---but the logical-complexity intuition is nearby enough that this literature deserves explicit acknowledgment.

## Causality, Responsibility, and Structural Explanation

The paper also belongs in conversation with structural accounts of causality, explanation, and responsibility [@pearl2009causality; @spirtes2000causation; @halpern2001explanations; @chockler2004responsibility]. Those works ask which variables or interventions matter for an outcome, how explanatory structure should be represented, and how responsibility can be assigned to particular factors. This is not the same problem studied here, but it is nearby in spirit: a relevant coordinate is precisely a coordinate whose perturbation can change the decision, and the optimizer quotient isolates exactly the distinctions that matter for optimal-action structure.

The difference, again, is that the present paper focuses on exact decision preservation under coordinate projection and on the computational cost of certifying that preservation. Causality and responsibility frameworks typically emphasize semantic characterization, counterfactual dependence, or intervention-level explanation. The present work instead asks when hidden coordinates can be ignored *without changing the optimizer map*, and how hard it is to certify that claim exactly across static, stochastic, and sequential regimes.

## Oracle, Query, and Access-Based Lower Bounds

Query-access lower bounds provide another nearby technique family [@dobzinski2012query]. Our witness-checking duality and access-obstruction results belong to that tradition in spirit. The difference is that the lower bounds are developed around the same fixed sufficiency predicate used throughout the paper, which allows direct comparison between forward evaluation, certification, exact minimization, and restricted-access models without changing the underlying decision relation.

## Scope and Novelty

The paper is therefore intentionally synthetic in the good sense. Rough sets and reducts study decision-preserving attribute reduction; feature selection studies informative subsets; statistical decision theory studies informativeness and sufficiency; abstraction and bisimulation literatures study exact aggregation of stochastic and sequential decision processes; explanation methods study local decision-preserving cores; formalized complexity work studies certified reductions and machine-checked decision procedures. The contribution here is to treat exact relevance certification itself as the common object and to develop one coherent complexity-theoretic account around it.

What is new is not merely that these literatures are mentioned together. It is that one predicate---exact preservation of optimal-action distinctions under coordinate hiding---is formalized once and then followed across static, stochastic, and sequential regimes; that the static regime exhibits a non-obvious collapse of minimum sufficiency to relevance containment; that the encoding-sensitive dichotomy isolates a sharp tractable--intractable boundary; that the downstream consequences are pushed through to exact simplification, witness checking, approximation, and the simplicity tax; and that a substantial Lean-backed certification layer is provided for the reduction and finite-decision core. Prior work contains many nearby pieces. The present paper aims to assemble them into a single precise theory of exact relevance certification.


# Conclusion

This paper develops a regime-sensitive complexity theory of exact relevance certification for coordinate-structured decision problems. Once the state space is factored into coordinates, evaluating the objective on one fully specified state and certifying which coordinates determine the optimal action become distinct computational tasks, and the latter organizes into a sharp hierarchy across static, stochastic, and sequential models.

## Main Results {#main-results .unnumbered}

Within the encoding model of Section [\[sec:encoding\]](#sec:encoding){reference-type="ref" reference="sec:encoding"}, we establish:

-   **Static Collapse Theorem plus regime-sensitive complexity matrix:** in the static regime, [Minimum-Sufficient-Set]{.smallcaps} collapses to relevance containment, so exact minimization is coNP-complete rather than a genuine $\Sigma_{2}^{P}$ search problem, while [Anchor-Sufficiency]{.smallcaps} remains $\Sigma_{2}^{P}$-complete. Once conditioning and temporal dynamics are added, the same exact-certification question escalates to PP-complete stochastic sufficiency and PSPACE-complete sequential sufficiency, with PP-hard and PSPACE-hard minimum and anchor queries. Under standard assumptions, these results form a strict hierarchy from static to stochastic to sequential exact certification.

-   **Sharp phase transition for exact certification:** logarithmic-size sufficient sets admit polynomial-time algorithms in the explicit regime, while linear-size sufficient sets inherit ETH-conditioned exponential lower bounds in the succinct regime.

-   **Structural and downstream consequences:** the paper derives witness and approximation obstructions, exact-simplification hardness transfers, and the trilemma of exact certification: no exact certifier can be simultaneously sound, complete, and polynomial-budgeted on the hard regime. This is a fundamental reliability constraint, and a simplicity-tax framing shows that omitted exact relevance is externalized rather than eliminated.

-   **Structured tractability:** bounded actions, separable utility, low tensor rank, tree structure, bounded treewidth, and coordinate symmetry yield polynomial-time certification procedures.

Taken together, these results show that exact relevance certification is a distinct complexity-theoretic object in its own right: computing $\operatorname{Opt}(s)$ on one state can be easy even when proving coordinate irrelevance is hard, and the difficulty of that certification problem depends sharply on whether one is ruling out counterexamples, comparing conditional averages, or preserving policies over time.

## Implications {#implications .unnumbered}

The implications are direct hardness transfers. Configuration simplification is an instance of sufficiency checking, so exact behavior-preserving minimization has no general-purpose polynomial-time worst-case solution unless $P=coNP$. Reliable exact certification is likewise regime-limited: under $P\neq coNP$, the trilemma of exact certification rules out any solver that is simultaneously sound, complete on the full hard regime, and polynomial-budgeted. And once the canonical relevant-coordinate set $R(\mathcal{D})$ is fixed, any compressed interface merely partitions that relevance into centrally handled and externalized parts. Compression therefore relocates exact relevance rather than erasing it; under the per-site cost model, the unresolved part grows linearly with deployment scale.

This is also the practical meaning of the simplicity tax. In the hard exact regime, a cheap simplification procedure cannot be assumed to have removed decision-relevant structure merely because it has compressed the visible interface. Any relevance omitted from the certified core must still be paid for somewhere else in the system: by local resolution, extra queries, abstention, or weaker guarantees. Exact simplification is therefore costly, but inexact simplification is not free; it externalizes the unresolved burden.

## Mechanization {#mechanization .unnumbered}

The Lean 4 development certifies the main reduction constructions, size-bounded hardness packages, exact finite deciders, counted finite-search procedures, tractability statements, and explicit-state step-counting wrappers used by the paper. This gives a mechanically checked core for the reduction and finite-decision layers of the argument. Standard PP/PSPACE membership arguments remain stated in the paper rather than packaged as end-to-end Turing-machine witness proofs.

## Outlook {#outlook .unnumbered}

Two immediate directions remain. First, the reduction infrastructure can be integrated more tightly with general-purpose formalized complexity libraries. Second, the artifact boundary can be made even cleaner by continuing to package the existing finite-decision results into more uniform summary interfaces without changing the paper's complexity claims. On the complexity side, the remaining gap is also conceptually interesting: sequential anchor and minimum variants may plausibly admit PSPACE upper bounds by a guess-and-verify argument, whereas stochastic anchor and minimum membership is less immediate because the added existential layer does not obviously remain inside PP.

## Acknowledgments {#acknowledgments .unnumbered}

Generative AI tools (including Codex, Claude Code, Augment, Kilo, and OpenCode) were used throughout this manuscript, across all sections (Abstract, Introduction, theoretical development, proof sketches, applications, conclusion, and appendix) and across all stages from initial drafting to final revision. The tools were used for boilerplate generation, prose and notation refinement, LaTeX/structure cleanup, translation of informal proof ideas into candidate formal artifacts (Lean/LaTeX), and repeated adversarial reviewer-style critique passes to identify blind spots and clarity gaps.

The author retained full intellectual and editorial control, including problem selection, theorem statements, assumptions, novelty framing, acceptance criteria, and final inclusion/exclusion decisions. No technical claim was accepted solely from AI output. Formal claims reported as machine-verified were admitted only after Lean verification (no `sorry` in cited modules) and direct author review; Lean was used as an integrity gate for responsible AI-assisted research. The author is solely responsible for all statements, citations, and conclusions.




---

## Machine-Checked Proofs

All theorems are formalized in Lean 4:
- Location: `docs/papers/paper4_decision_quotient/proofs/`
- Lines: 21142
- Theorems: 985
- `sorry` placeholders: 0
