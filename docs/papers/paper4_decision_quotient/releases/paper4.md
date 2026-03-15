# Paper: Decision Quotient: A Regime-Sensitive Complexity Theory of Exact Relevance Certification

**Status**: Computational Complexity-ready | **Lean**: 22068 lines, 1015 theorems

---

## Abstract

Which coordinates of a decision problem can be hidden without changing the decision, and what is the coarsest exact abstraction that preserves all decision-relevant distinctions? We study this as an exact relevance-certification problem organized around the optimizer quotient. We classify how hard it is to certify this structure across three settings: static (counterexample exclusion), stochastic (conditioning and expectation), and sequential (temporal structure). In the static regime, sufficiency collapses to relevance containment, so minimum sufficiency is coNP-complete. In the stochastic regime, preservation and decisiveness separate: preservation is polynomial-time under explicit-state encoding with bridge theorems to static sufficiency and the optimizer quotient, while decisiveness is PP-hard under succinct encoding with anchor and minimum variants in $\textsf{NP}^{\textsf{PP}}$. In the sequential regime, all queries are PSPACE-complete. We also prove an encoding-sensitive contrast between explicit-state tractability and succinct-encoding hardness, derive an integrity-competence trilemma, and isolate twelve tractable subcases. A Lean 4 artifact mechanically verifies the optimizer-quotient universal property, main reductions, and finite decider core.


# Introduction {#sec:introduction}

Which coordinates of a decision problem can be hidden without changing the optimal action, and what is the canonical exact abstraction that records all and only the distinctions that matter for that choice? We study this question for a decision problem $\mathcal{D}=(A,S,U)$ with $S = X_1 \times \cdots \times X_n$. A coordinate set $I$ is sufficient when agreement on $I$ forces agreement on the optimal-action set: $$s_I = s'_I \implies \operatorname{Opt}(s) = \operatorname{Opt}(s').$$ The associated optimizer quotient identifies states exactly when they induce the same optimal-action set. In **Set**, it is the coarsest abstraction through which $\operatorname{Opt}$ factors, so every surjective abstraction that preserves optimal-action distinctions factors uniquely through it. This canonical status is theorem-level rather than heuristic.

This paper's main claim is that once that canonical object is fixed, exact relevance certification becomes a regime-sensitive complexity problem. The structural target is the same throughout, but the cost of certifying it changes with the ambient model. In the static regime, certification reduces to counterexample exclusion and collapses to relevance containment. In the stochastic regime, conditioning splits the exact question into preservation and decisiveness. In the sequential regime, temporal contingency lifts the same certification task to planning-style reasoning. The paper therefore has a simple backbone: the optimizer quotient supplies the canonical notion of exact decision-preserving compression, and the regime analyses classify when that structure can be certified exactly.

#### Informal interpretive guide.

The formal objects studied here admit a compact intuitive reading. The optimizer quotient isolates the decision signal: it collapses exactly those state distinctions that do not change the optimal-action set. In that precise sense, it removes exactly decision noise and nothing decision-relevant. Relevant coordinates are points of leverage: changing them can change the decision. In the static regime, the resulting simplification is forced by structure rather than discovered by search. In the later consequence sections, the simplicity tax names the residual burden created when a system uses heuristic simplification in place of exact certification.

This is a certification problem rather than a forward-evaluation problem. Evaluating $U(a,s)$ or $\operatorname{Opt}(s)$ on a fully specified state is one task; proving that a family of coordinates is irrelevant requires ruling out every counterexample pair of states. That universal structure is exactly what makes the optimizer quotient central rather than cosmetic: it records the maximal lossless collapse of the state space, and every exact decision-preserving abstraction must refine it. In the static regime, this structural picture sharpens further. A coordinate set is sufficient exactly when it exposes all relevant coordinates, so the minimum sufficient set is not discovered by a separate alternating search. It is the relevant-coordinate set itself. By contrast, [Anchor-Sufficiency]{.smallcaps} retains a genuine existential choice because the anchor assignment must still satisfy a universal fiberwise condition.

#### Probability splits the certification problem.

In stochastic decision problems, exact analysis separates into a preservation question and a decisiveness question. Preservation asks whether coarse information reproduces the full-information optimizer, so it is the direct stochastic analogue of exact quotient preservation. Decisiveness asks whether each observable fiber already determines a unique Bayes-optimal action, so it is a stronger decision-completeness demand. Formally, these are *stochastic sufficiency* and *stochastic decisiveness*. They are distinct, and the paper treats them separately on purpose. Preservation gives the bridge back to static sufficiency and, under full support, back to the same quotient structure as the static regime, while beyond full support the paper isolates support-sensitive bridge and obstruction lemmas rather than claiming a full general-distribution classification. Decisiveness yields the paper's strongest succinct-encoding hardness classification. The stochastic row is therefore not one theorem about one predicate, but a split exact theory with two different complexity behaviors.

This is also what the complexity classification buys beyond nearby literatures on reducts, feature selection, or abstraction heuristics. Those traditions often ask whether some smaller representation is informative enough, predictive enough, or value-preserving enough. The present paper asks a sharper question: when can one certify exactly that hiding coordinates leaves the optimal-action correspondence unchanged? That exactness requirement is what later yields a no-general exact minimizer theorem, a regime-sensitive tractability boundary, and a simple but important reliability corollary about exact certifiers in the hard regime. The point is not merely that some small interface may work in practice, but whether there is a certificate that no omitted coordinate can change the decision.

This distinction also explains why exactness matters. When a system replaces exact relevance certification with a heuristic simplification, the omitted relevance does not disappear; it is shifted into residual uncertainty that must be handled elsewhere in the system. We refer to that externalized burden as the *simplicity tax*. The results of this paper show that this tax is not merely rhetorical: exact simplification can be computationally expensive, but inexact simplification does not erase decision-relevant structure; it relocates it.

This systematic development reveals a regime-sensitive complexity landscape because the same structural target is being certified under richer models. Static certification is governed by counterexample exclusion. In the stochastic regime, preservation retains the quotient-preservation target while decisiveness introduces conditional-comparison hardness. In the sequential regime, omitted information can matter only through future contingencies, so exact certification becomes planning-hard. Read this way, the complexity matrix is not a list of unrelated classifications. It is the cost profile of certifying the exact decision-relevant distinctions encoded by the optimizer quotient.

Throughout, we work with finite coordinate-structured decision problems under the encoding conventions fixed in Section [\[sec:encoding\]](#sec:encoding){reference-type="ref" reference="sec:encoding"}.

## Contributions

The paper is best read as a theory of exact decision-relevant information organized around one canonical mathematical object: the optimizer quotient. The quotient gives the coarsest exact abstraction that preserves optimal-action distinctions, and this canonical status is theorem-level in the artifact. The paper then classifies how the cost of certifying that structure changes across static, stochastic, and sequential regimes, identifies tractable structural subcases, and supports the finite combinatorial core with a Lean artifact.

1.  **Optimizer quotient as canonical structure.** For a finite decision problem, the optimizer quotient identifies states exactly when they induce the same optimal-action set. In **Set**, it is the coarsest exact decision-preserving abstraction, equivalently the coimage/image factorization of the optimizer map. This gives the paper a common structural object through which the later static, stochastic, and sequential results can be compared.

2.  **Regime-sensitive certification landscape.** The paper studies how hard it is to certify exact preservation of decision-relevant structure across three regimes. In the static regime, sufficiency collapses to relevance containment, so [Minimum-Sufficient-Set]{.smallcaps} is coNP-complete rather than the expected $\Sigma_2^P$ search problem, while [Anchor-Sufficiency]{.smallcaps} remains $\Sigma_2^P$-complete. This collapse is structurally nontrivial. In classical rough-set theory, general decision tables can admit multiple incomparable minimal reducts, which drives the minimum-reduct problem into combinatorial search territory. Under the exact optimal-action preservation predicate studied here, Proposition [\[prop:minimal-relevant-equiv\]](#prop:minimal-relevant-equiv){reference-type="ref" reference="prop:minimal-relevant-equiv"} proves there is exactly one minimal sufficient set, the relevant-coordinate set itself. This uniqueness is the structural reason for the collapse to coNP relevance containment. To our knowledge, this specific structural collapse, from $\Sigma_2^P$ search to coNP driven by optimizer-quotient uniqueness, is a novel classification not captured by general rough-set bounds. In the stochastic regime, exact analysis splits into preservation and decisiveness. Preservation is polynomial-time under explicit-state encoding, bridges back to static sufficiency and the optimizer quotient, and under full support yields inherited coNP and $\Sigma_2^P$ classifications for the minimum and anchor variants. Decisiveness is polynomial-time in explicit state and PP-hard under succinct encoding, with the anchor and minimum variants PP-hard and in $\textsf{NP}^{\textsf{PP}}$ at paper level. In the sequential regime, sufficiency, minimum, and anchor queries are PSPACE-complete.

3.  **Tractability islands and encoding-sensitive contrast.** The paper proves an explicit-state versus succinct-encoding contrast and isolates tractable subcases across two tiers: simple edge cases (single action, bounded state space, strict dominance, constant optimal set) that reduce to immediate checks, and structurally nontrivial cases (bounded actions, separable utility, low tensor rank, tree structure, bounded treewidth, coordinate symmetry) each removing a distinct source of hardness. These are not ad hoc exceptions. They are regimes in which the decision boundary admits enough structural alignment to make exact certification feasible.

4.  **Consequences for exact certifiers.** In the hard exact regime, certifiers cannot in general be simultaneously sound, complete, and polynomial-budgeted. The same theory yields witness-checking lower bounds, approximation obstructions, and the practical consequence called the simplicity tax: when exact relevance is not certified, the unresolved burden is not removed but shifted elsewhere in the system.

5.  **Mechanized finite-decision core.** A Lean 4 artifact mechanically checks the optimizer-quotient universal-property core, the main finite deciders, search procedures, reduction-correctness lemmas, witness/checking schemas, and bridge results The oracle-class placements are proved in the paper text, while the artifact independently verifies the finite combinatorial core on which those arguments rely. This places the mechanization in the emerging category of problem-specific certified reduction infrastructure.

## Quantifier Structure

The complexity gap comes from the structure of the question. Insufficiency has short witnesses: two states that agree on the proposed coordinates but induce different optimal-action sets. Sufficiency, by contrast, asks for the absence of every such witness. So the object-level task is forward evaluation on a fully specified state, while the meta-level task is universal verification over counterexample pairs. This difference drives the coNP classification and explains why relevance identification is not just another forward-evaluation problem.

The static regime also contains the paper's main structural simplification. Although minimum sufficiency can be written with an outer existential quantifier over coordinate sets, the existential layer collapses once sufficiency is characterized as containment of all relevant coordinates. The stochastic and sequential regimes do not simplify in the same way: the stochastic regime separates into preservation and decisiveness, with full-support inheritance on the preservation side and counting-style hardness on the decisiveness side, while temporal evolution introduces dependence on temporally induced optimizer structure.

## Paper Structure

Section [\[sec:formal-setup\]](#sec:formal-setup){reference-type="ref" reference="sec:formal-setup"} introduces the formal setup and encoding conventions. Sections [\[sec:hardness\]](#sec:hardness){reference-type="ref" reference="sec:hardness"}, [\[sec:stochastic\]](#sec:stochastic){reference-type="ref" reference="sec:stochastic"}, [\[sec:sequential\]](#sec:sequential){reference-type="ref" reference="sec:sequential"}, and [\[sec:regime-hierarchy\]](#sec:regime-hierarchy){reference-type="ref" reference="sec:regime-hierarchy"} develop the core theorem package. Sections [\[sec:dichotomy\]](#sec:dichotomy){reference-type="ref" reference="sec:dichotomy"} and [\[sec:tractable\]](#sec:tractable){reference-type="ref" reference="sec:tractable"} give the encoding-sensitive contrast and tractable subcases. Section [\[sec:engineering-corollaries\]](#sec:engineering-corollaries){reference-type="ref" reference="sec:engineering-corollaries"} collects implications for exact simplification and reliability consequences for exact certifiers. Section [\[sec:related\]](#sec:related){reference-type="ref" reference="sec:related"} situates the work relative to formalized complexity and decision theory.

## Artifact Availability

The proof artifact is archived at <https://doi.org/10.5281/zenodo.18998870>. The cited-content snapshot contains 22068 lines across 67 files with 0 occurrences of `sorry`.


# Formal Setup {#sec:formal-setup}

This section fixes the abstract vocabulary used throughout the paper. We work with finite decision problems whose state spaces factor into coordinates, and we isolate the minimal terminology needed to state the later complexity results. The same setup will support the paper's three core regime classifications, the encoding-sensitive boundary, and the later consequence sections.

  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **Object**               **Quantifier shape**                           **Certifies**                              **Role**
  ------------------------ ---------------------------------------------- ------------------------------------------ ---------------------------------------------------------
  SUFFICIENCY-CHECK        universal over state pairs                     exact coordinate sufficiency               baseline certification predicate

  MINIMUM-SUFFICIENT-SET   apparent $\exists I\,\forall (s,s')$ check     smallest sufficient support                collapses to relevance containment in the static regime

  ANCHOR-SUFFICIENCY       $\exists$ anchor, then universal fiber check   exact preservation on one chosen fiber     retains the outer existential layer

  Optimizer quotient       universal factorization property               coarsest decision-preserving abstraction   canonical structural object
  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Setup map.** The four central objects differ mainly in quantifier shape: minimum sufficiency collapses in the static regime, while anchor sufficiency keeps its existential fiber choice.

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

The decision quotient is the canonical mathematical object associated with exact decision preservation. It quotients the state space by optimizer-equivalence and thereby isolates precisely the distinctions that any decision-preserving abstraction must respect.

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

::: remark
[]{#rem:decision-noise label="rem:decision-noise"} The quotient viewpoint identifies decision noise exactly. At the state level, decision noise is the kernel of the optimizer map: two states differ only by decision noise if and only if they determine the same class in $Q_{\mathcal{D}}$. At the coordinate level, a coordinate is pure decision noise if and only if varying it alone never changes the decision, equivalently if and only if it is irrelevant. In the finite probabilistic reading used later, this is also equivalent to conditional independence of the decision from that coordinate given the remaining coordinates.
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

::: remark
In the stochastic row, the preservation predicate compares the optimizer seen under coarse conditioning on $I$ with the fully conditioned optimizer. Under full support, the later bridge package reconnects this stochastic notion to the same quotient picture as the static regime.
:::


# Static Regime Complexity {#sec:hardness}

This section studies the static regime, where the input is a finite coordinate-structured decision problem under the succinct encoding of Section [\[sec:encoding\]](#sec:encoding){reference-type="ref" reference="sec:encoding"}. This is the baseline certification setting: no stochastic averaging, no temporal dynamics, only the exact question of whether a candidate coordinate set preserves the optimizer map. The main contribution here is structural: sufficiency is exactly relevance containment. Once that characterization is in place, the complexity classification for [Minimum-Sufficient-Set]{.smallcaps} follows immediately. The only genuinely higher-level static query is the anchor variant, where an existentially chosen fiber must satisfy a universal constancy condition.

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

## Why You Might Expect $\Sigma_2^P$

At first glance, [Minimum-Sufficient-Set]{.smallcaps} seems to have the form $$\exists I\; \forall s,s'\in S:
\quad s_I=s'_I \implies \operatorname{Opt}(s)=\operatorname{Opt}(s').$$ If that syntax reflected the true structure of the problem, one might expect a $\Sigma_2^P$ classification. The natural algorithm is: guess a coordinate set $I$, then universally verify that every agreeing state pair induces the same optimal-action set. On its face, static exact minimization therefore looks like a genuine existential--universal search problem.

## Structural Characterization of Minimum Sufficiency

Proposition [\[prop:minimal-relevant-equiv\]](#prop:minimal-relevant-equiv){reference-type="ref" reference="prop:minimal-relevant-equiv"} removes the outer search layer: a set is sufficient exactly when it contains every relevant coordinate, so the minimum sufficient set is simply the relevant-coordinate set. This is the key structural theorem of the static regime. Static exact minimization is therefore a relevance-counting problem, not a genuinely alternating search problem.

That collapse is specific to minimum sufficiency. [Anchor-Sufficiency]{.smallcaps} still asks the algorithm to choose an assignment $\alpha$ and then certify a universal condition on the induced fiber, so its existential layer cannot be absorbed into a structural characterization of the instance.

## Complexity Consequence: coNP-Completeness

The first two results show that exact certification is already harder than pointwise evaluation. A counterexample to sufficiency is short, but certifying that no such counterexample exists is a universal statement over state pairs. The collapse discussed above explains why the same coNP behavior governs both direct sufficiency checking and exact minimization.

#### Proof idea.

Membership is immediate from short counterexamples: a "no" witness is just a pair of states that agree on $I$ but induce different optimal-action sets. Hardness comes from reducing TAUTOLOGY to an empty-information instance whose optimizer map is constant exactly in the tautology case.

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

#### Proof idea.

Once sufficiency is equivalent to containing all relevant coordinates, the question "is there a sufficient set of size at most $k$?" becomes "are there at most $k$ relevant coordinates?" The search component disappears, and the problem inherits the same coNP character as direct sufficiency checking.

::: proof
*Proof.* Proposition [\[prop:minimal-relevant-equiv\]](#prop:minimal-relevant-equiv){reference-type="ref" reference="prop:minimal-relevant-equiv"} identifies the minimum sufficient set with the relevant-coordinate set. The question "$|I^*|\le k$?" therefore becomes the question whether at most $k$ coordinates are relevant, which is a coNP check over relevance witnesses. CoNP-hardness follows from the $k=0$ slice, which is exactly SUFFICIENCY-CHECK with $I=\emptyset$. ◻
:::

## Sigma-2-P Completeness

Anchor sufficiency adds one more layer of choice that the previous collapse does not remove: the algorithm may select a fiber, but must then certify that the chosen fiber is uniformly decision-preserving. That existential-then-universal pattern is exactly what raises the complexity class.

#### Proof idea.

Membership follows the visible quantifier pattern: guess an anchor assignment and then verify constancy on its fiber. Hardness is obtained by encoding the existential variables of an $\exists\forall$ formula into the anchor choice and the universal variables into the residual coordinates that must be checked.

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

The TAUTOLOGY reduction stack, the coNP and $\Sigma_2^P$ classifications, and the witness-budget lower-bound core are indexed by their inline Lean handles.

#### Explicit-state search upper bounds.

Under the explicit-state step-counting model, static sufficiency and static anchor sufficiency are also wrapped as abstract [P]{.smallcaps} predicates on inputs carrying a certified state budget. The artifact proves these upper bounds via counted searches with quadratic and linear state-space dependence respectively, and it also packages the static sufficiency search as an explicit correctness-plus-step-bound witness and as part of the unified finite-search summary theorem.

This completes the baseline regime. The next section keeps the same exact certification question but replaces pairwise counterexample exclusion by conditional expected-utility comparison, which is why the complexity lifts from the static coNP/$\Sigma_2^P$ picture to PP.


# Stochastic Regime Complexity {#sec:stochastic}

We now add a distribution over states. Once utilities are compared through conditional expectations, the stochastic regime has two exact predicates rather than one. The first is a preservation predicate: does the optimizer seen after hiding coordinates agree with the optimizer seen when the full state is known? The second is a decisiveness predicate: does each observed fiber admit a unique Bayes-optimal action? Formally, these are stochastic sufficiency and stochastic decisiveness. Stochastic sufficiency gives the direct stochastic analogue of static sufficiency and the bridge back to the optimizer quotient; under full support, its minimum and anchor variants inherit the static coNP and $\Sigma_2^P$ classifications. Stochastic decisiveness yields the strongest succinct-encoding complexity classification established in the paper.

## Preservation and Decisiveness

In static decision problems, the natural exact question is whether hiding coordinates preserves the optimizer. In stochastic decision problems, preservation is still natural, but it is no longer the only exact question. One can also ask whether the retained signal is itself a complete decision interface, meaning that every observable fiber has a unique conditional optimum. That is the decisiveness predicate. The two predicates are not equivalent: a coordinate set may preserve the full-information optimizer without being decisive, and it may be decisive without preserving the full-information optimizer. Probability therefore produces two parallel exact questions rather than one uniform stochastic analogue.

## Stochastic Decision Problems

::: definition
[]{#def:stochastic-decision-problem label="def:stochastic-decision-problem"} A stochastic decision problem is a tuple $$\mathcal{D}_S=(A,X_1,\ldots,X_n,U,P),$$ where $P\in\Delta(S)$ is a distribution on $S=X_1\times\cdots\times X_n$.
:::

::: definition
[]{#def:stochastic-sufficient label="def:stochastic-sufficient"} For each admissible fiber value $\alpha \in \mathrm{supp}(S_I)$, define the conditional optimal-action set $$\operatorname{Opt}^{\mathrm{stoch}}_I(\alpha)
:=
\arg\max_{a\in A}\,\mathbb{E}[U(a,S)\mid S_I = \alpha].$$ A coordinate set $I$ is *stochastically sufficient* if conditioning on $I$ preserves the fully conditioned optimizer: $$\forall s \in S:\quad \operatorname{Opt}^{\mathrm{stoch}}_I(s_I)=\operatorname{Opt}(s).$$ Here $S \sim P$ is the random state drawn from the instance distribution.
:::

#### Intuition.

The conditional optimizer $\operatorname{Opt}^{\mathrm{stoch}}_I(\alpha)$ aggregates expected utilities over the entire fiber $\{s : s_I = \alpha\}$ using the distribution $P$. By contrast, the full-information optimizer $\operatorname{Opt}(s)$ conditions on the exact state $s$ itself. Preservation asks whether this averaging over a fiber ever changes which action is optimal; decisiveness asks whether, after averaging, the optimal action on each fiber is uniquely determined.

::: definition
[]{#def:stochastic-singleton-sufficient label="def:stochastic-singleton-sufficient"} Using the same conditional optimizer map $\operatorname{Opt}^{\mathrm{stoch}}_I$, a coordinate set $I$ is *stochastically decisive* if every admissible fiber has a uniquely determined conditional optimal action: $$\forall \alpha \in \mathrm{supp}(S_I)\; \exists a \in A:\quad \operatorname{Opt}^{\mathrm{stoch}}_I(\alpha)=\{a\}.$$ In the paper, we refer to this as *stochastic decisiveness*. In the current artifact, this is the stochastic predicate with the strongest succinct-encoding complexity package.
:::

#### From witness exclusion to counting.

Static sufficiency asks whether any counterexample pair exists. Stochastic sufficiency asks whether the conditional optimizer obtained from the retained coordinates still matches the fully conditioned optimizer, while stochastic decisiveness asks whether every admissible fiber has a uniquely determined conditional optimum. The first question is about ruling out disagreeing state pairs; the second and third are about conditional expected-utility comparisons on each fiber. That shift from witness exclusion to weighted counting comparison is exactly what brings stochastic hardness into the picture.

#### Example.

Consider a stochastic decision problem with $S=\{s_1,s_2\}$, $A=\{a,b\}$, uniform distribution $P(s_1)=P(s_2)=1/2$, and utilities $U(a,s_1)=2$, $U(b,s_1)=1$, $U(a,s_2)=0$, $U(b,s_2)=3$. For $I=\emptyset$: (i) Preservation: $\mathbb{E}[U(a,S)]=1$, $\mathbb{E}[U(b,S)]=2$, so $\operatorname{Opt}^{\mathrm{stoch}}_\emptyset=\{b\}$; but $\operatorname{Opt}(s_1)=\{a\}$, so preservation fails. (ii) Decisiveness: The empty fiber itself has a unique conditional optimum $\{b\}$, so $I=\emptyset$ is decisive. This shows decisiveness is strictly stronger than preservation.

## Stochastic Status at a Glance

#### Table 2.

Regime classification summary for the stochastic row. Open entries mark explicit scope boundaries. Under full support, stochastic preservation is completely classified in the natural positive-support base case.

  --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **Predicate/** **query**                **What is proved here**                                                                                                               **Mechanized core**                                                             **Open / not formalized**
  --------------------------------------- ------------------------------------------------------------------------------------------------------------------------------------- ------------------------------------------------------------------------------- ------------------------------------------
  Stochastic sufficiency (preservation)   Polynomial-time under explicit-state encoding; bridges to static sufficiency and the optimizer quotient                               Explicit-state decider; full-support bridge package                             No succinct-encoding hardness claimed

  Minimum / anchor preservation           Polynomial-time in explicit state; under full support, inherit coNP-completeness / $\Sigma_2^P$-completeness from the static regime   Full-support inheritance theorems; finite searches; obstruction/bridge lemmas   Open; conjecture coNP-hard

  Stochastic decisiveness                 Polynomial-time under explicit-state encoding; PP-hard under succinct encoding                                                        Reduction core; finite witness/checking schemata                                No oracle-machine formalization

  Minimum / anchor decisiveness           PP-hard; in $\textsf{NP}^{\textsf{PP}}$ at paper level                                                                                Existential witness/checking schemata; bounded explicit-state searches          No oracle-class membership formalization
  --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

The rest of this section unpacks the four rows of this summary. The preservation side is complete in explicit state and under full-support inheritance, and now also has support-sensitive partial results beyond full support. The decisiveness side carries the strongest succinct-encoding classification currently established in the paper.

::: problem
[]{#prob:stochastic-sufficiency label="prob:stochastic-sufficiency"} **Input:** $\mathcal{D}_S=(A,X_1,\ldots,X_n,U,P)$ and $I\subseteq\{1,\ldots,n\}$.\
**Question:** Is $I$ stochastically sufficient?
:::

## Stochastic Sufficiency

::: theorem
[]{#thm:stochastic-preservation-explicit label="thm:stochastic-preservation-explicit"} Under the explicit-state encoding, [Stochastic-Sufficiency-Check]{.smallcaps} is decidable in polynomial time.
:::

::: proof
*Proof.* For a fixed information set $I$, the verifier scans the finite state space and checks whether the conditional fiber optimizer $\operatorname{Opt}^{\mathrm{stoch}}_I(s_I)$ agrees with the fully conditioned optimizer $\operatorname{Opt}(s)$ at every state $s$. The artifact contains a counted explicit-state search witnessing exactly this predicate together with a linear-in-$|S|$ step bound Hence stochastic sufficiency is polynomial-time decidable in the explicit-state model. ◻
:::

#### Conjecture 4.1 (Support-Sensitive Preservation Complexity).

Under the succinct encoding of Section [\[sec:encoding\]](#sec:encoding){reference-type="ref" reference="sec:encoding"}, [Stochastic-Preservation-Check]{.smallcaps} for general distributions (without the full-support assumption) is coNP-hard.

The structured justification is anchored to existing theorems and mechanized lemmas. First, what is known: under full support, preservation inherits the static coNP classification via Proposition [\[prop:stochastic-full-support-equiv\]](#prop:stochastic-full-support-equiv){reference-type="ref" reference="prop:stochastic-full-support-equiv"} and Theorem [\[thm:stochastic-preservation-full-support-variants\]](#thm:stochastic-preservation-full-support-variants){reference-type="ref" reference="thm:stochastic-preservation-full-support-variants"}; the explicit-state base query is polynomial for all distributions by Theorem [\[thm:stochastic-preservation-explicit\]](#thm:stochastic-preservation-explicit){reference-type="ref" reference="thm:stochastic-preservation-explicit"}; and decisiveness under the same encoding is PP-hard by Theorem [\[thm:stochastic-pp\]](#thm:stochastic-pp){reference-type="ref" reference="thm:stochastic-pp"}.

Second, the structural obstacle: the support-sensitive obstruction lemmas in the artifact (Proposition [\[prop:stochastic-preservation-partial-general\]](#prop:stochastic-preservation-partial-general){reference-type="ref" reference="prop:stochastic-preservation-partial-general"}) isolate the exact mechanism. Preservation quantifies over all states while stochastic conditioning weights by probability. Zero-probability states can therefore witness static violations that the distribution never observes; full-support transfer fails exactly when such states are permitted. This is a structural obstruction, and the mechanized artifact isolates the finite combinatorial core any reduction must respect .

Third, why coNP rather than PP: preservation is syntactically universal, because it asks whether the coarse optimizer matches the full optimizer at every state, so its quantifier shape matches the static coNP classification. Decisiveness is PP-hard because it requires majority-style conditional expected-utility comparisons. Preservation compares optimizer sets rather than aggregating utilities, so the natural hardness source is counterexample exclusion (coNP) rather than majority counting (PP); support-sensitivity complicates reduction engineering but not the high-level quantifier structure.

#### Reduction intuition.

A natural hardness reduction would encode static counterexample pairs into zero-probability witness states of the distribution. Concretely, from a TAUTOLOGY instance $\varphi$, one could construct a succinct stochastic decision problem where the distribution assigns zero mass to a carefully chosen set of states that encode the $\varphi$-violating assignments. Preservation must check the universal condition over ALL states, including these zero-probability witnesses, so a coNP-style verification can detect the hidden TAUTOLOGY counterexample even though the distribution never observes it. This intuition aligns with the support-sensitive obstruction lemmas (Proposition [\[prop:stochastic-preservation-partial-general\]](#prop:stochastic-preservation-partial-general){reference-type="ref" reference="prop:stochastic-preservation-partial-general"}) but requires overcoming the technical challenge of succinctly encoding witness-state support.

Open Problem 4.2. Determine the exact succinct-encoding complexity of [Stochastic-Minimum-Preservation]{.smallcaps} and [Stochastic-Anchor-Preservation]{.smallcaps} for general distributions. Under Conjecture 4.1, these inherit coNP-completeness and $\Sigma_2^P$-completeness respectively, matching the static regime; resolving the conjecture would complete the final open cell in the regime matrix (Table 2).

::: proposition
[]{#prop:stochastic-to-static-sufficiency label="prop:stochastic-to-static-sufficiency"} If $I$ is stochastically sufficient, then $I$ is sufficient for the underlying static decision problem.
:::

::: proof
*Proof.* If the coarse conditional optimizer agrees with the fully conditioned optimizer at every state, then any two states that agree on $I$ also have equal full-information optimal-action sets. So the original static optimizer already factors through the projection to $I$. ◻
:::

::: proposition
[]{#prop:stochastic-full-support-equiv label="prop:stochastic-full-support-equiv"} Assume the distribution $P$ has full support and the action set is nonempty. Then $I$ is statically sufficient if and only if it is stochastically sufficient.
:::

::: proof
*Proof.* The forward direction uses full support to show that when the static optimizer is constant on each $I$-fiber, conditional averaging cannot change the optimizer on that fiber. The reverse direction is Proposition [\[prop:stochastic-to-static-sufficiency\]](#prop:stochastic-to-static-sufficiency){reference-type="ref" reference="prop:stochastic-to-static-sufficiency"}. Both directions are mechanized. ◻
:::

::: proposition
[]{#prop:stochastic-quotient-equiv label="prop:stochastic-quotient-equiv"} Under the hypotheses of Proposition [\[prop:stochastic-full-support-equiv\]](#prop:stochastic-full-support-equiv){reference-type="ref" reference="prop:stochastic-full-support-equiv"}, the stochastic fiber quotient induced by $I$ coincides with the original decision quotient.
:::

::: proof
*Proof.* Under stochastic sufficiency, the conditional fiber optimizer and the fully conditioned optimizer agree statewise. Hence the induced stochastic equivalence relation matches the original decision-equivalence relation, so the quotient setoids coincide. Under full support, Proposition [\[prop:stochastic-full-support-equiv\]](#prop:stochastic-full-support-equiv){reference-type="ref" reference="prop:stochastic-full-support-equiv"} supplies the needed preservation hypothesis. ◻
:::

::: problem
[]{#prob:stochastic-minimum-preservation label="prob:stochastic-minimum-preservation"} **Input:** $\mathcal{D}_S=(A,X_1,\ldots,X_n,U,P)$ and $k\in\mathbb{N}$.\
**Question:** Does there exist a stochastically sufficient coordinate set $I$ with $|I|\le k$?
:::

::: problem
[]{#prob:stochastic-anchor-preservation label="prob:stochastic-anchor-preservation"} **Input:** $\mathcal{D}_S=(A,X_1,\ldots,X_n,U,P)$ and $I\subseteq\{1,\ldots,n\}$.\
**Question:** Does there exist an anchor state $s_0$ such that for every state $s$ agreeing with $s_0$ on $I$, one has $\operatorname{Opt}^{\mathrm{stoch}}_I(s_I)=\operatorname{Opt}(s)$?
:::

::: theorem
[]{#thm:stochastic-preservation-full-support-variants label="thm:stochastic-preservation-full-support-variants"} Assume the distribution $P$ has full support and the action set is nonempty. Then:

1.  [Stochastic-Minimum-Preservation]{.smallcaps} is equivalent to the underlying static [Minimum-Sufficient-Set]{.smallcaps} query.

2.  For each fixed $I$, [Stochastic-Anchor-Preservation]{.smallcaps} is equivalent to the underlying static [Anchor-Sufficiency]{.smallcaps} query on $I$.

Consequently, under full support, stochastic minimum preservation is coNP-complete and stochastic anchor preservation is $\Sigma_2^P$-complete.
:::

::: proof
*Proof.* For minimum preservation, Proposition [\[prop:stochastic-full-support-equiv\]](#prop:stochastic-full-support-equiv){reference-type="ref" reference="prop:stochastic-full-support-equiv"} applies pointwise to each candidate coordinate set $I$, so the existential minimization problem is unchanged under full support. For anchor preservation, if one anchor fiber preserves the full-information optimizer, then the optimizer is already constant on that fiber, giving static anchor sufficiency. Conversely, if the static optimizer is constant on one $I$-fiber, then under full support the same conditional-expectation argument used in Proposition [\[prop:stochastic-full-support-equiv\]](#prop:stochastic-full-support-equiv){reference-type="ref" reference="prop:stochastic-full-support-equiv"} shows that the coarse conditional optimizer agrees with the full-information optimizer throughout that fiber. The complexity claims therefore inherit directly from Theorems [\[thm:minsuff-conp\]](#thm:minsuff-conp){reference-type="ref" reference="thm:minsuff-conp"} and [\[thm:anchor-sigma2p\]](#thm:anchor-sigma2p){reference-type="ref" reference="thm:anchor-sigma2p"}. ◻
:::

::: proposition
[]{#prop:stochastic-preservation-variant-search label="prop:stochastic-preservation-variant-search"} For finite instances, the artifact contains counted exhaustive-search procedures for stochastic minimum preservation and stochastic anchor preservation. Their certified step bounds are $O(2^n)$ and $O(|S|)$, respectively.
:::

::: proof
*Proof.* The minimum-preservation search scans the subset lattice and invokes the explicit-state preservation checker on each candidate set. The anchor-preservation search scans candidate anchor states and checks preservation on the induced fiber. Both procedures are packaged with correctness theorems and explicit step bounds in the artifact. ◻
:::

::: theorem
[]{#thm:stochastic-preservation-variants-explicit label="thm:stochastic-preservation-variants-explicit"} Under the explicit-state encoding, [Stochastic-Minimum-Preservation]{.smallcaps} and [Stochastic-Anchor-Preservation]{.smallcaps} are both decidable in polynomial time.
:::

::: proof
*Proof.* For minimum preservation, the explicit-state search enumerates coordinate subsets and invokes the explicit preservation checker on each candidate set; the resulting counted procedure is polynomial in the explicit input size for fixed state and coordinate budgets. For anchor preservation, the explicit-state search enumerates anchor states and checks preservation on the induced fiber, yielding a polynomial-time verifier. Both wrappers are mechanized in the artifact. ◻
:::

::: proposition
[]{#prop:stochastic-preservation-partial-general label="prop:stochastic-preservation-partial-general"} Let $I$ be a coordinate set.

1.  If $I$ is stochastically sufficient, then $I$ contains every relevant coordinate of the underlying static decision problem.

2.  Consequently, in the Boolean-cube setting, if [Stochastic-Minimum-Preservation]{.smallcaps} has a yes-instance with budget $k$, then the static relevant-coordinate set has cardinality at most $k$.

3.  If the underlying static decision problem is sufficient on $I$ and every $I$-fiber contains at least one positive-probability state, then $I$ is stochastically sufficient.

4.  If the underlying static decision problem is anchor-sufficient on $I$ and the anchor fiber contains a positive-probability state, then $I$ is stochastically anchor-preserving.
:::

::: proof
*Proof.* The first statement composes the unconditional bridge from stochastic preservation to static sufficiency with the static theorem that every sufficient set contains all relevant coordinates. The second is the resulting cardinality bound for any witness set in the minimum query. For the positive-support statements, the mechanized proof refines the full-support argument: it uses positivity only on one representative state per relevant fiber, together with fiberwise invariance of the conditional optimizer, to propagate static sufficiency or anchor sufficiency into stochastic preservation on the whole fiber. ◻
:::

#### What this paper establishes for preservation.

For stochastic sufficiency in the preservation sense, this paper proves explicit-state tractability for the base, minimum, and anchor queries (Theorem [\[thm:stochastic-preservation-explicit\]](#thm:stochastic-preservation-explicit){reference-type="ref" reference="thm:stochastic-preservation-explicit"}, Proposition [\[prop:stochastic-preservation-variant-search\]](#prop:stochastic-preservation-variant-search){reference-type="ref" reference="prop:stochastic-preservation-variant-search"}, and Theorem [\[thm:stochastic-preservation-variants-explicit\]](#thm:stochastic-preservation-variants-explicit){reference-type="ref" reference="thm:stochastic-preservation-variants-explicit"}), static-to-stochastic bridge theorems (Propositions [\[prop:stochastic-to-static-sufficiency\]](#prop:stochastic-to-static-sufficiency){reference-type="ref" reference="prop:stochastic-to-static-sufficiency"} and [\[prop:stochastic-full-support-equiv\]](#prop:stochastic-full-support-equiv){reference-type="ref" reference="prop:stochastic-full-support-equiv"}), quotient equivalence under full support (Proposition [\[prop:stochastic-quotient-equiv\]](#prop:stochastic-quotient-equiv){reference-type="ref" reference="prop:stochastic-quotient-equiv"}), and full-support inheritance for the minimum and anchor variants (Theorem [\[thm:stochastic-preservation-full-support-variants\]](#thm:stochastic-preservation-full-support-variants){reference-type="ref" reference="thm:stochastic-preservation-full-support-variants"}). Beyond full support, the paper now also isolates unconditional obstruction lemmas and positive-support bridge results (Proposition [\[prop:stochastic-preservation-partial-general\]](#prop:stochastic-preservation-partial-general){reference-type="ref" reference="prop:stochastic-preservation-partial-general"}). This yields a complete classification for stochastic preservation under full support, the natural base case in which every state has nonzero probability, together with support-sensitive partial theory beyond that base case. The general-distribution case still introduces technical complications from zero-probability fibers, and the paper does not claim a full general-distribution classification for the minimum and anchor preservation variants or a succinct-encoding hardness classification for the preservation predicate itself. The PP-hardness results in the stochastic row concern decisiveness and its anchor/minimum variants, not preservation itself.

::: problem
[]{#prob:stochastic-singleton-sufficiency label="prob:stochastic-singleton-sufficiency"} **Input:** $\mathcal{D}_S=(A,X_1,\ldots,X_n,U,P)$ and $I\subseteq\{1,\ldots,n\}$.\
**Question:** Is $I$ stochastically decisive?
:::

#### Naming convention.

From this point onward, the historical names "stochastic anchor sufficiency" and "stochastic minimum sufficiency" refer to the decisiveness family, not the preservation family introduced above.

::: definition
[]{#def:stochastic-anchor-sufficient label="def:stochastic-anchor-sufficient"} A coordinate set $I$ is *stochastically anchor-sufficient* if there exists an admissible fiber value $\alpha \in \mathrm{supp}(S_I)$ and an action $a \in A$ such that $$\arg\max_{a'\in A}\,\mathbb{E}[U(a',S)\mid S_I = \alpha] = \{a\}.$$ Equivalently, some $I$-fiber has a unique conditional optimal action.
:::

::: problem
[]{#prob:stochastic-anchor-sufficiency label="prob:stochastic-anchor-sufficiency"} **Input:** $\mathcal{D}_S=(A,X_1,\ldots,X_n,U,P)$ and $I\subseteq\{1,\ldots,n\}$.\
**Question:** Is $I$ stochastically anchor-sufficient?
:::

::: problem
[]{#prob:stochastic-minimum-sufficient label="prob:stochastic-minimum-sufficient"} **Input:** $\mathcal{D}_S=(A,X_1,\ldots,X_n,U,P)$ and $k\in\mathbb{N}$.\
**Question:** Does there exist a stochastically decisive coordinate set $I$ with $|I|\le k$?
:::

## Encoding-Sensitive Complexity of Stochastic Decisiveness

The stochastic regime preserves the same underlying certification template, namely whether the retained coordinates determine the optimizer, but asks it after conditioning and expectation have been introduced. The current complexity package concerns stochastic decisiveness because it admits a clean mechanized reduction core and directly supports the existential anchor/minimum queries. For the empty-information case, one is already comparing aggregate mass of satisfying versus nonsatisfying states. More generally, stochastic decisiveness asks whether one action strictly dominates all competitors in conditional expected utility on each admissible fiber.

::: theorem
[]{#thm:stochastic-pp label="thm:stochastic-pp"} For the stochastic decisiveness predicate above, [Stochastic-Decisiveness-Check]{.smallcaps} is decidable in polynomial time under the explicit-state encoding. Under the succinct encoding it is PP-hard. The accompanying upper-bound argument is proved in the paper via an explicit bad-fiber witness characterization whose finite witness/checking core is mechanized in the artifact.
:::

::: proof
*Proof.* For the explicit-state encoding, the number of admissible fibers is at most $|S|$, so one can enumerate the fibers and, for each fiber value $\alpha$, compute the conditional expected utilities of all actions directly from the table. Checking whether exactly one action attains the maximum on each fiber is therefore polynomial in the explicit input size.

For succinct-encoding hardness, reduce MAJSAT by the three-action gadget used in the mechanized reduction. Given Boolean formula $\varphi$, take $S=\{0,1\}^n$, uniform $P$, actions $\{\mathrm{accept},\mathrm{hold}_L,\mathrm{hold}_R\}$, and utilities $$U(\mathrm{accept},s)=\mathbf{1}[\varphi(s)],\qquad
U(\mathrm{hold}_L,s)=U(\mathrm{hold}_R,s)=\frac12-2^{-(n+1)},$$ with $I=\emptyset$. There is only one admissible fiber, so stochastic decisiveness asks whether the prior-optimal action is unique. By construction, this happens exactly when $\mathbb{E}[\varphi]\ge 1/2$, i.e., in the MAJSAT case. Thus MAJSAT many-one reduces to stochastic decisiveness.

For the upper bound, failure of decisiveness is witnessed by one bad state whose observed fiber does not have a singleton conditional optimum. The current artifact packages exactly this no-witness characterization together with the corresponding existential witness schema, and the two directions are bundled in a summary theorem These mechanized witness/checking packages verify the finite combinatorial core used by the paper's oracle-class arguments. Concretely, the artifact verifies the existential-universal quantifier structure of the predicate, the witness-checking schema that guesses an anchor and verifies conditional-uniqueness using PP comparisons, and the finite-step-counted search wrappers (handles OU1 to OU2, OU6 to OU7) that bound the nondeterministic guessing phase. The standard oracle-machine reduction proving $\textsf{NP}^{\textsf{PP}}$ membership is argued in the paper text and not mechanized. The corresponding $\textsf{coNP}^{\textsf{PP}}$-style and $\textsf{NP}^{\textsf{PP}}$-style memberships are then proved in the paper text by standard complexity-theoretic reasoning. ◻
:::

::: proposition
[]{#prop:stochastic-anchor-refinement label="prop:stochastic-anchor-refinement"} If $I$ is stochastically decisive, then $I$ is stochastically anchor-sufficient.
:::

::: proof
*Proof.* Choose any admissible fiber value $\alpha$. By stochastic decisiveness, there exists an action $a$ such that $$\operatorname{Opt}^{\mathrm{stoch}}_I(\alpha)=\{a\}.$$ That fiber therefore witnesses stochastic anchor sufficiency. ◻
:::

::: theorem
[]{#thm:stochastic-anchor-hard label="thm:stochastic-anchor-hard"} [Stochastic-Anchor-Sufficiency-Check]{.smallcaps} is PP-hard.
:::

::: proof
*Proof.* Reduce MAJSAT. Given Boolean formula $\varphi$ on $n\geq 1$ variables, take $S=\{0,1\}^n$, uniform $P$, $I=\emptyset$, and three actions $\{\mathrm{accept},\mathrm{hold}_L,\mathrm{hold}_R\}$. Set $$U(\mathrm{accept},s)=\mathbf{1}[\varphi(s)],\qquad
U(\mathrm{hold}_L,s)=U(\mathrm{hold}_R,s)=\frac12-2^{-(n+1)}.$$ If at least half of all assignments satisfy $\varphi$, then $\mathrm{accept}$ is the unique optimal action on the empty-information fiber, so the instance is anchor-sufficient. If fewer than half satisfy $\varphi$, then $\mathrm{hold}_L$ and $\mathrm{hold}_R$ tie above $\mathrm{accept}$, so no singleton anchor optimum exists. Thus MAJSAT many-one reduces to the stochastic anchor query. ◻
:::

::: theorem
[]{#thm:stochastic-minimum-hard label="thm:stochastic-minimum-hard"} [Stochastic-Minimum-Sufficient-Set]{.smallcaps} is PP-hard.
:::

::: proof
*Proof.* Reduce MAJSAT to the $k=0$ slice. In the same three-action gadget used above, there exists a stochastically decisive set of size at most $0$ iff the empty set itself is stochastically decisive. This empty-set equivalence for the gadget is certified in the artifact, so MAJSAT many-one reduces to [Stochastic-Minimum-Sufficient-Set]{.smallcaps}. ◻
:::

::: proposition
[]{#prop:stochastic-explicit-search label="prop:stochastic-explicit-search"} For finite instances, the artifact contains counted exhaustive-search procedures for stochastic sufficiency, stochastic decisiveness, stochastic anchor sufficiency, and stochastic minimum sufficiency. Their certified step bounds are respectively $O(|S|)$, $O(|S|)$, $O(|S||A|)$, and $O(2^n)$.
:::

::: proof
*Proof.* The stochastic-sufficiency search scans for a state where the coarse conditional optimizer disagrees with the fully conditioned optimizer; the decisiveness search scans for a violating state whose fiber-optimal set is non-singleton; the anchor search scans over candidate anchor-state/action pairs; and the minimum query scans the subset lattice. Each procedure admits a counted boolean search formulation with both a correctness theorem and an explicit step bound. ◻
:::

::: theorem
[]{#thm:stochastic-nppp-upper label="thm:stochastic-nppp-upper"} Under the present encoding, [Stochastic-Anchor-Sufficiency-Check]{.smallcaps} and [Stochastic-Minimum-Sufficient-Set]{.smallcaps} are both in $\textsf{NP}^{\textsf{PP}}$.
:::

::: proof
*Proof.* For anchor sufficiency, the mechanized collapse $$\mathrm{STOCHASTIC\mbox{-}ANCHOR\mbox{-}SUFFICIENCY\mbox{-}CHECK}(P,I)
\iff
\exists s_0\,\exists a\; \bigl(\mathrm{fiberOpt}(P,I,s_0)=\{a\}\bigr)$$ is already available in the artifact PA3. A nondeterministic polynomial-time machine may therefore guess $(s_0,a)$ and use a PP oracle to verify that $a$ is the unique conditional optimum on the anchor fiber. Concretely, uniqueness can be checked by comparing the conditional expected utility of $a$ against every competing action on that fiber, which is exactly the same majority-style conditional comparison used in the decisiveness verification.

For the minimum query, the machine guesses a coordinate set $I$ with $|I|\le k$ and then asks whether $I$ is stochastically decisive. The same witness/checking template used for the anchor case yields the required PP-style verifier for the guessed set, so the minimum query is also in $\textsf{NP}^{\textsf{PP}}$. ◻
:::

Theorem [\[thm:stochastic-nppp-upper\]](#thm:stochastic-nppp-upper){reference-type="ref" reference="thm:stochastic-nppp-upper"} completes the complexity picture for stochastic decisiveness: decisiveness is polynomial-time in the explicit-state model and PP-hard in the succinct model, while the anchor and minimum decisiveness queries are PP-hard and lie in $\textsf{NP}^{\textsf{PP}}$. Those oracle-class memberships are proved in the paper text. The artifact independently verifies the finite core behind those proofs: the bad-fiber counter-witness schema for decisiveness, the anchor-collapse equivalence, the generic existential-witness schema, bounded-witness recoveries for the explicit-state stochastic anchor and minimum searches, and existential-majority hardness/completeness packages for the anchor and decisiveness query families

#### Scope note.

For stochastic preservation, this paper establishes explicit-state tractability for the preservation family, full-support inheritance, quotient equivalence, and support-sensitive obstruction/bridge lemmas beyond full support. The general-distribution minimum and anchor variants still remain open as full classification problems and are not claimed. For stochastic decisiveness, the paper establishes PP-hardness and $\textsf{NP}^{\textsf{PP}}$ membership under the stated encoding. The open entries in Table 2 are explicit scope boundaries rather than unresolved subcases.

The remaining gap is not a bookkeeping omission but a real structural issue. Stochastic decisiveness is a pure verification problem: once $I$ is fixed, one must check a family of majority-style conditional comparisons, which yields the stated encoding-sensitive complexity. By contrast, stochastic anchor and minimum sufficiency add an outer existential choice, namely to guess an anchor witness or a candidate coordinate set, before performing the same conditional verification. The current collapse theorem PA3 shows that anchor sufficiency reduces to existence of a fiber with a singleton conditional optimum, and the new existential-majority reduction shows that this extra witness layer is not vacuous. So the open question is genuinely whether that existential layer can be absorbed into PP under the present encoding, not whether the paper has simply stopped one step short of an obvious closure argument.

For that reason, the paper's claim is intentionally calibrated. It does not present the stochastic row as one undifferentiated theorem about one predicate. Instead, it treats preservation and decisiveness as parallel exact questions. Preservation gives the direct stochastic analogue of the static question and the bridge back to the optimizer quotient. Decisiveness yields the strongest succinct-encoding complexity classification established in the paper. The split is part of the proved theory, not a temporary patch.

#### Relevance boundary.

The stochastic decisiveness query is fiber-indexed: fixing a candidate information set $I$ induces a derived decision problem whose optimizer is the conditional fiber optimizer. In the current formalization this induced problem is always $I$-sufficient , but its relevance notion is therefore also indexed by $I$. For that reason, the paper does not assert a single global relevance-set characterization for stochastic minimum sufficiency analogous to the static and sequential regimes.

## Tractable Subcases

::: proposition
[]{#prop:stochastic-tractable label="prop:stochastic-tractable"} Stochastic sufficiency is polynomial-time solvable under each of the following restrictions:

1.  product distributions $P=\bigotimes_i P_i$,

2.  bounded support $|\mathrm{supp}(P)|\le k$,

3.  structured families where conditional expectations are computable in polynomial time (e.g. log-concave models with oracle access).
:::

::: proof
*Proof.* In each case, conditional expected utilities can either be computed exactly or reduced to a polynomial number of marginal comparisons. ◻
:::

## Bridge from Static

These final propositions show that moving from static to stochastic reasoning is not merely a reformulation. Distributional conditioning can genuinely destroy or preserve stochastic sufficiency depending on the structure of the input model.

::: proposition
[]{#prop:static-not-stochastic label="prop:static-not-stochastic"} There exist instances where $I$ is sufficient in the static sense but not stochastically sufficient once a distribution $P$ is fixed.
:::

::: proof
*Proof.* Choose utilities where optimizer ties are harmless pointwise but are broken in expectation after conditioning. Then static fibers preserve $\operatorname{Opt}$, while the conditional optimizer obtained from $I$ fails to match the fully conditioned optimizer. ◻
:::

::: proposition
[]{#prop:static-to-stochastic-product label="prop:static-to-stochastic-product"} Suppose that on every $I$-fiber the static optimal-action set is a singleton, and that this singleton is constant on that fiber. Then $I$ is stochastically sufficient for every distribution $P$.
:::

::: proof
*Proof.* Fix an admissible fiber value $\alpha$ and let $\{a_\alpha\}$ be the common pointwise optimal-action set on that fiber. For every competing action $b \ne a_\alpha$ and every state $s$ with $s_I=\alpha$, one has $U(a_\alpha,s) > U(b,s)$. Taking conditional expectations over the fiber preserves this strict inequality, so $$\operatorname{Opt}^{\mathrm{stoch}}_I(\alpha)=\{a_\alpha\}.$$ Since this holds for every admissible fiber, the conditional optimizer on each fiber agrees with the fully conditioned optimizer at every state in that fiber. Hence $I$ is stochastically sufficient. ◻
:::

The stochastic regime therefore gives the middle layer of the paper's hierarchy: conditioning already raises exact certification beyond the static setting by turning exact certification into a counting comparison problem. Adding time will lift us further still: the next section shows how temporal contingency presents, alongside stochastic sufficiency, a decisiveness complexity family and then pushes the same underlying question from PP-style comparisons to PSPACE-level temporal certification.


# Sequential Regime Complexity {#sec:sequential}

We now move to sequential settings with transitions and observations. Here the same certification question is asked for a sequential decision model carrying transitions, observations, and a planning horizon. In the current formalization, the certified predicate is state-based: after packaging the sequential instance as its induced one-step decision map, can one suppress some coordinates without changing the resulting optimal-action set? This is where the complexity reaches PSPACE. The structural reason is that hidden information can matter later even when it appears irrelevant now. Certification must therefore account for temporally unfolding contingencies rather than compare only one-shot optimizers or conditional expectations. This complexity jump is consonant with the exact abstraction literature for temporally structured decision models: once hidden information can affect future contingent choices, exact aggregation and exact relevance certification both require reasoning about temporally mediated distinctions rather than one-shot optimizer comparisons. In succinct models, that temporal contingency naturally supports the same kind of alternation that drives PSPACE hardness in planning and quantified-logic reductions.

## Sequential Decision Problems

::: definition
[]{#def:sequential-decision-problem label="def:sequential-decision-problem"} A sequential decision problem is a tuple $$\mathcal{D}_{\mathrm{seq}}=(A,X_1,\ldots,X_n,U,T,O),$$ with state space $S=X_1\times\cdots\times X_n$, transition kernel $T:A\times S\to\Delta(S)$, and observation model $O:S\to\Delta(\Omega)$.
:::

::: definition
[]{#def:sequential-sufficient label="def:sequential-sufficient"} For a sequential instance $\mathcal D_{\mathrm{seq}}$, let $\operatorname{Opt}_{\mathrm{seq}}(s)$ denote the optimal-action set of its induced decision map on state $s$. A coordinate set $I$ is *sequentially sufficient* if $$\forall s,s'\in S:\quad s_I=s'_I \implies \operatorname{Opt}_{\mathrm{seq}}(s)=\operatorname{Opt}_{\mathrm{seq}}(s').$$ This is the sequential state-based certification predicate used throughout the paper and in the artifact.
:::

#### Two interpretations.

There are two closely related readings of this definition. Formally, the sequential row studies sufficiency for the optimizer induced by the sequential model. Its intended interpretation is policy-level: hidden information may be irrelevant at time zero and still become relevant later because it changes future contingent choices.

One can also read the same construction heuristically as a coordinate-restricted POMDP abstraction question. A full controller conditions on evolving observations and therefore on induced beliefs over latent states. Hiding coordinates does not merely erase information about the present state; it changes the informational interface through which future decisions are made. That heuristic is useful for intuition and for understanding the TQBF gadget, but the formal theorems in this paper are stated for the state-based predicate above rather than for a fully developed history-dependent policy class.

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

The sequential regime is the strongest of the three because hidden information can matter not only for the current optimizer but for the future evolution of the decision problem. Static certification asks whether bad state pairs exist; stochastic certification asks whether conditional averages cross a threshold; sequential certification asks whether omitted coordinates can change the induced optimizer after temporal structure is taken into account. That added temporal contingency is exactly what the TQBF reduction exploits.

The reduction is natural for the same reason that quantified formulas and temporally structured control problems are natural companions elsewhere in the planning literature. Existential quantifier blocks correspond to controller-side choices, universal blocks correspond to adversarial or environment-controlled resolutions, and the truth value of the quantified formula is encoded into the induced optimizer of the reduced sequential instance. A hidden coordinate can look locally irrelevant after one transition yet still encode which branch of the quantified game has been entered. In that sense the reduction is not merely a correctness gadget: it exposes the source of PSPACE hardness, namely that exact relevance certification must reason about information whose effect is mediated by temporal contingency rather than by a single one-shot comparison.

::: theorem
[]{#thm:sequential-pspace label="thm:sequential-pspace"} SEQUENTIAL-SUFFICIENCY-CHECK is PSPACE-complete.
:::

::: proof
*Proof.* For membership, the verifier ranges over state pairs or candidate witnesses while evaluating the induced sequential optimizer of the input instance. Under the present finite-horizon encoding, that induced optimizer can be evaluated using only polynomial space in the succinct input description of the reduction gadgets, and the outer search can likewise be carried out in polynomial space. This is the verifier pattern abstracted by the explicit finite-search layer formalized in the artifact.

For hardness, reduce TQBF. Encode alternating quantifiers into the reduced sequential instance so that the induced optimizer is state-independent exactly when the quantified instance is true. Then empty-set sequential sufficiency holds exactly when the source TQBF instance is true. ◻
:::

The proof can be read operationally as follows. A sequential certificate must rule out the possibility that some hidden bit becomes decision-relevant only after several temporally mediated updates. That means the verifier cannot stop at a single static comparison; it has to account for contingent future structure in the induced optimizer. TQBF is therefore the correct reduction target: the same alternation between "there exists a choice" and "for every resolution" is already built into the reduced sequential instance.

::: proposition
[]{#prop:sequential-anchor-refinement label="prop:sequential-anchor-refinement"} If $I$ is sequentially sufficient, then $I$ is sequentially anchor-sufficient.
:::

::: proof
*Proof.* If optimal-action sets are preserved for every pair of states agreeing on $I$, then fixing any anchor state immediately yields preservation across its entire $I$-agreement class. ◻
:::

::: theorem
[]{#thm:sequential-anchor-hard label="thm:sequential-anchor-hard"} The sequential anchor query is PSPACE-complete.
:::

::: proof
*Proof.* Membership is in PSPACE. By definition, the query asks whether there exists an anchor state $s_0$ such that every state agreeing with $s_0$ on $I$ preserves the same optimal-action set. A nondeterministic polynomial-space procedure can guess $s_0$ and then scan candidate states one at a time, verifying the agreement condition and comparing the corresponding optimal-action sets. Since NPSPACE = PSPACE, the query lies in PSPACE.

Reuse the TQBF reduction for sequential sufficiency with $I=\emptyset$. At empty information, sequential anchor sufficiency is equivalent to global preservation of the optimal-action set, so the same construction yields $$\mathrm{TQBF}(q)
\iff
\mathrm{SEQUENTIAL\mbox{-}ANCHOR\mbox{-}SUFFICIENCY\mbox{-}CHECK}(\mathrm{reduceTQBF}(q),\emptyset).$$ Hence TQBF many-one reduces to the sequential anchor query. ◻
:::

::: theorem
[]{#thm:sequential-minimum-hard label="thm:sequential-minimum-hard"} The sequential minimum-sufficient-set query is PSPACE-complete.
:::

::: proof
*Proof.* Membership is in PSPACE. The query asks whether there exists a coordinate set $I$ of size at most $k$ such that $I$ is sequentially sufficient. A nondeterministic polynomial-space procedure can guess $I$ and then run the same polynomial-space verifier used for sequential sufficiency. Again, NPSPACE = PSPACE.

Reduce TQBF to the $k=0$ slice. There exists a sequentially sufficient set of size at most $0$ iff the empty set itself is sequentially sufficient. The same construction yields PSPACE-hardness for SEQUENTIAL-MINIMUM-SUFFICIENT-SET. ◻
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

With the static, stochastic, and sequential classifications in place, the regime hierarchy is now complete. The next section consolidates that hierarchy into a single matrix, and the following section refines it with the encoding-sensitive ETH lower bounds that explain when exact certification remains feasible despite the worst-case class barriers.


# Regime Matrix {#sec:regime-hierarchy}

The static, stochastic, and sequential formulations are the paper's central organizing object. They ask the same underlying question about which coordinates matter for the decision, but they answer it in increasingly expressive models. In the stochastic regime, preservation and decisiveness become parallel exact predicates. Adding time then introduces temporal contingency. Read as a complexity matrix, adding probability introduces conditional-comparison hardness, and adding time lifts the core queries further to PSPACE.

## Complexity Matrix

::: center
  -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **Regime /** **predicate**   **Base query**                     **Minimum**                                           **Anchor**
  ---------------------------- ---------------------------------- ----------------------------------------------------- -------------------------------------------------------------
  Static                       coNP-complete                      coNP-complete                                         $\Sigma_2^P$-complete

  Stochastic preservation      P (explicit), bridge theorems      P (explicit); coNP-complete (full) / open (general)   P (explicit); $\Sigma_2^P$-complete (full) / open (general)

  Stochastic decisiveness      P (explicit), PP-hard (succinct)   PP-hard, in $\textsf{NP}^{\textsf{PP}}$               PP-hard, in $\textsf{NP}^{\textsf{PP}}$

  Sequential                   PSPACE-complete                    PSPACE-complete                                       PSPACE-complete
  -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
:::

This table is the paper's compact summary. The stochastic regime contains two parallel predicates with distinct complexity profiles: preservation and decisiveness. In the static regime, sufficiency and minimum sufficiency are coNP-complete and anchor sufficiency is $\Sigma_2^P$-complete. In the stochastic regime, preservation (stochastic sufficiency) has explicit-state tractability and bridge theorems back to static sufficiency and the optimizer quotient; the minimum and anchor preservation variants are also polynomial-time in explicit state, inherit coNP-completeness and $\Sigma_2^P$-completeness under full support, and admit support-sensitive obstruction and bridge lemmas beyond full support, while the full general-distribution classification remains open. Decisiveness (stochastic decisiveness) has the sharper succinct-encoding classification: polynomial-time in the explicit-state model, PP-hardness in the succinct model, and paper-level $\textsf{NP}^{\textsf{PP}}$ membership for the anchor and minimum variants. The artifact mechanizes the finite witness/checking schemas and reduction cores underlying those upper bounds. In the sequential regime, sufficiency, minimum, and anchor queries are PSPACE-complete.

#### Preservation Boundary.

For stochastic sufficiency in the preservation sense, this paper proves explicit-state tractability for the preservation family (Theorem [\[thm:stochastic-preservation-explicit\]](#thm:stochastic-preservation-explicit){reference-type="ref" reference="thm:stochastic-preservation-explicit"} and Theorem [\[thm:stochastic-preservation-variants-explicit\]](#thm:stochastic-preservation-variants-explicit){reference-type="ref" reference="thm:stochastic-preservation-variants-explicit"}), static-to-stochastic bridge theorems (Propositions [\[prop:stochastic-to-static-sufficiency\]](#prop:stochastic-to-static-sufficiency){reference-type="ref" reference="prop:stochastic-to-static-sufficiency"} and [\[prop:stochastic-full-support-equiv\]](#prop:stochastic-full-support-equiv){reference-type="ref" reference="prop:stochastic-full-support-equiv"}), quotient equivalence under full support (Proposition [\[prop:stochastic-quotient-equiv\]](#prop:stochastic-quotient-equiv){reference-type="ref" reference="prop:stochastic-quotient-equiv"}), full-support inheritance for the minimum and anchor variants (Theorem [\[thm:stochastic-preservation-full-support-variants\]](#thm:stochastic-preservation-full-support-variants){reference-type="ref" reference="thm:stochastic-preservation-full-support-variants"}), and support-sensitive obstruction/bridge lemmas beyond full support (Proposition [\[prop:stochastic-preservation-partial-general\]](#prop:stochastic-preservation-partial-general){reference-type="ref" reference="prop:stochastic-preservation-partial-general"}). Outside the full-support setting, this paper does *not* claim a succinct-encoding hardness classification for the preservation predicate or a full general-distribution classification for its minimum and anchor variants. The PP-hardness results in the stochastic row concern decisiveness and its variants, not preservation itself.

The decisiveness results combine standard hardness claims with mechanized upper-bound packages. The artifact verifies the finite witness/checking schemata and reduction cores used by the stochastic oracle-class arguments, while the corresponding oracle-class memberships are proved in the paper text by standard complexity-theoretic reasoning.

The stochastic anchor and minimum decisiveness queries are not presently collapsed to PP-completeness: Theorem [\[thm:stochastic-nppp-upper\]](#thm:stochastic-nppp-upper){reference-type="ref" reference="thm:stochastic-nppp-upper"} places them in $\textsf{NP}^{\textsf{PP}}$, while Theorems [\[thm:stochastic-anchor-hard\]](#thm:stochastic-anchor-hard){reference-type="ref" reference="thm:stochastic-anchor-hard"} and [\[thm:stochastic-minimum-hard\]](#thm:stochastic-minimum-hard){reference-type="ref" reference="thm:stochastic-minimum-hard"} give PP-hardness. The artifact verifies the finite witness/checking structure and explicit-state bounded-witness recoveries behind that upper-bound story, and it also packages existential-majority hardness/completeness surrogates for the anchor and decisiveness query families The oracle-class memberships themselves remain paper-level proofs under the succinct encoding. By contrast, the sequential anchor and minimum variants admit paper-level PSPACE membership via guess-and-verify over anchors or coordinate subsets together with the sequential sufficiency verifier.

::: theorem
[]{#thm:regime-main label="thm:regime-main"} Under the encoding conventions of Section [\[sec:encoding\]](#sec:encoding){reference-type="ref" reference="sec:encoding"}, the classification is as follows. In the static regime, sufficiency and minimum sufficiency are coNP-complete and anchor sufficiency is $\Sigma_2^P$-complete. In the stochastic regime, preservation (stochastic sufficiency) has a polynomial-time base query under explicit-state encoding with proved bridges back to static sufficiency and the optimizer quotient; the minimum and anchor preservation variants are also polynomial-time under explicit-state encoding, are coNP-complete and $\Sigma_2^P$-complete under full support, and satisfy the support-sensitive partial results of Proposition [\[prop:stochastic-preservation-partial-general\]](#prop:stochastic-preservation-partial-general){reference-type="ref" reference="prop:stochastic-preservation-partial-general"}. Decisiveness (stochastic decisiveness) has a polynomial-time base query under explicit-state encoding and is PP-hard under the succinct encoding; the anchor and minimum decisiveness queries are PP-hard and lie in $\textsf{NP}^{\textsf{PP}}$, with those oracle-class memberships proved in the paper text and their finite witness/checking cores mechanized. In the sequential regime, sufficiency, minimum, and anchor queries are PSPACE-complete.
:::

::: proof
*Proof.* The matrix entries are exactly Theorems [\[thm:sufficiency-conp\]](#thm:sufficiency-conp){reference-type="ref" reference="thm:sufficiency-conp"}, [\[thm:minsuff-conp\]](#thm:minsuff-conp){reference-type="ref" reference="thm:minsuff-conp"}, [\[thm:anchor-sigma2p\]](#thm:anchor-sigma2p){reference-type="ref" reference="thm:anchor-sigma2p"}, [\[thm:stochastic-preservation-explicit\]](#thm:stochastic-preservation-explicit){reference-type="ref" reference="thm:stochastic-preservation-explicit"}, [\[thm:stochastic-preservation-full-support-variants\]](#thm:stochastic-preservation-full-support-variants){reference-type="ref" reference="thm:stochastic-preservation-full-support-variants"}, [\[thm:stochastic-preservation-variants-explicit\]](#thm:stochastic-preservation-variants-explicit){reference-type="ref" reference="thm:stochastic-preservation-variants-explicit"}, Proposition [\[prop:stochastic-preservation-partial-general\]](#prop:stochastic-preservation-partial-general){reference-type="ref" reference="prop:stochastic-preservation-partial-general"}, [\[thm:stochastic-pp\]](#thm:stochastic-pp){reference-type="ref" reference="thm:stochastic-pp"}, [\[thm:stochastic-anchor-hard\]](#thm:stochastic-anchor-hard){reference-type="ref" reference="thm:stochastic-anchor-hard"}, [\[thm:stochastic-minimum-hard\]](#thm:stochastic-minimum-hard){reference-type="ref" reference="thm:stochastic-minimum-hard"}, [\[thm:stochastic-nppp-upper\]](#thm:stochastic-nppp-upper){reference-type="ref" reference="thm:stochastic-nppp-upper"}, [\[thm:sequential-pspace\]](#thm:sequential-pspace){reference-type="ref" reference="thm:sequential-pspace"}, [\[thm:sequential-anchor-hard\]](#thm:sequential-anchor-hard){reference-type="ref" reference="thm:sequential-anchor-hard"}, and [\[thm:sequential-minimum-hard\]](#thm:sequential-minimum-hard){reference-type="ref" reference="thm:sequential-minimum-hard"}, together with the stochastic sufficiency bridge package. The strict regime comparisons are isolated separately in Theorems [\[thm:static-stochastic-strict\]](#thm:static-stochastic-strict){reference-type="ref" reference="thm:static-stochastic-strict"} and [\[thm:stochastic-sequential-strict\]](#thm:stochastic-sequential-strict){reference-type="ref" reference="thm:stochastic-sequential-strict"}. ◻
:::

## Strict Regime Gaps

::: theorem
[]{#thm:static-stochastic-strict label="thm:static-stochastic-strict"} Assuming $coNP\ne PP$, the succinct stochastic decisiveness classification is strictly harder in worst-case complexity than the static sufficiency classification.
:::

::: proof
*Proof.* Static sufficiency checking is coNP-complete (Theorem [\[thm:sufficiency-conp\]](#thm:sufficiency-conp){reference-type="ref" reference="thm:sufficiency-conp"}), while stochastic decisiveness is PP-hard under the succinct encoding (Theorem [\[thm:stochastic-pp\]](#thm:stochastic-pp){reference-type="ref" reference="thm:stochastic-pp"}). Under the assumption $coNP\ne PP$, this yields a strict complexity gap between the two classifications. ◻
:::

::: theorem
[]{#thm:stochastic-sequential-strict label="thm:stochastic-sequential-strict"} Assuming $PP\ne PSPACE$, the sequential sufficiency classification is strictly harder in worst-case complexity than the stochastic decisiveness classification.
:::

::: proof
*Proof.* Stochastic decisiveness is PP-hard under the succinct encoding (Theorem [\[thm:stochastic-pp\]](#thm:stochastic-pp){reference-type="ref" reference="thm:stochastic-pp"}), while sequential sufficiency is PSPACE-complete (Theorem [\[thm:sequential-pspace\]](#thm:sequential-pspace){reference-type="ref" reference="thm:sequential-pspace"}). The class gap yields a strict complexity separation between the two classifications under the assumption $PP\ne PSPACE$. ◻
:::

## Integrity-Resource Bounds

The regime matrix has a methodological consequence as well: as one moves upward in the hierarchy, exact certification becomes less compatible with fixed polynomial resource budgets.

::: proposition
[]{#prop:abstention-frontier label="prop:abstention-frontier"} For polynomial-time exact certifiers that abstain whenever they cannot certify correctness within budget, the set of instances requiring abstention expands along $$\text{static} \to \text{stochastic} \to \text{sequential}.$$
:::

::: proof
*Proof.* As the decision predicate moves from coNP to PP to PSPACE hardness, exact worst-case certification requires strictly stronger resources. Fixed polynomial resources therefore force abstention on a larger instance family in higher regimes. ◻
:::

#### Forward-looking perspective.

The regime hierarchy presented here naturally extends to richer models: partially observable settings, multi-agent decision problems, and game-theoretic regimes would introduce additional layers of complexity such as knowledge reasoning or equilibrium computation. The optimizer-quotient framework provides a unifying language for tracking exact decision-relevance across these extended regimes, though their detailed classification remains open.


# Encoding Dichotomy and ETH Lower Bounds {#sec:dichotomy}

The regime hierarchy identifies where exact relevance certification sits in the standard complexity landscape. This section extracts the complementary algorithmic message: under the encodings of Section [\[sec:encoding\]](#sec:encoding){reference-type="ref" reference="sec:encoding"}, there is a strong contrast between explicit-state instances whose true relevant support is small and succinct instances whose relevant support is extensive. The former admit direct algorithms; the latter inherit exponential lower bounds under ETH. The practical takeaway is cross-model rather than single-model: logarithmic relevant support can still be certified exactly in polynomial time under explicit encodings, while linear relevant support under succinct encodings already forces ETH-conditioned exponential cost.

#### Model note.

Part 1 is an explicit-state upper bound (time polynomial in $|S|$). Part 2 is a succinct-encoding lower bound under ETH (time exponential in $n$). The encodings are defined in Section [\[sec:encoding\]](#sec:encoding){reference-type="ref" reference="sec:encoding"}.

  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **Encoding**       **Support regime**   **Complexity consequence**                                                    **Source**
  ------------------ -------------------- ----------------------------------------------------------------------------- ------------------------------------------------------------------------------------------------
  Explicit-state     $k^* = O(\log N)$    exact certification is polynomial in $N$ by projection enumeration            Theorem [\[thm:dichotomy\]](#thm:dichotomy){reference-type="ref" reference="thm:dichotomy"}(1)

  Succinct           $k^* = \Omega(n)$    exact certification inherits an ETH-conditioned $2^{\Omega(n)}$ lower bound   Theorem [\[thm:dichotomy\]](#thm:dichotomy){reference-type="ref" reference="thm:dichotomy"}(2)
  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Boundary summary.** The tractable-hard contrast is controlled jointly by encoding and by the size of the true sufficient support.

::: theorem
[]{#thm:dichotomy label="thm:dichotomy"} Let $\mathcal{D} = (A, X_1, \ldots, X_n, U)$ be a decision problem with $|S| = N$ states. Let $k^*$ be the size of the minimal sufficient set.

1.  **Logarithmic case (explicit-state upper bound):** If $k^* = O(\log N)$, then SUFFICIENCY-CHECK is solvable in polynomial time in $N$ under the explicit-state encoding.

2.  **Linear case (succinct lower bound under ETH):** If $k^* = \Omega(n)$, then SUFFICIENCY-CHECK requires time $\Omega(2^{n/c})$ for some constant $c > 0$ under the succinct encoding (assuming ETH).
:::

::: proof
*Proof.* **Part 1 (Logarithmic case):** If $k^* = O(\log N)$, then the number of distinct projections $|S_{I^*}|$ is at most $2^{k^*} = O(N^c)$ for some constant $c$. Under the explicit-state encoding, we enumerate all projections and verify sufficiency in polynomial time.

**Part 2 (Linear case):** We establish this by a standard ETH transfer under the succinct encoding. ◻
:::

## The ETH Reduction Chain {#sec:eth-chain}

The lower bound in Part 2 of Theorem [\[thm:dichotomy\]](#thm:dichotomy){reference-type="ref" reference="thm:dichotomy"} follows from a short standard ETH transfer. Under ETH, 3-SAT requires $2^{\Omega(n)}$ time. Negation gives a linear-time reduction from 3-SAT to TAUTOLOGY. The static reduction of Theorem [\[thm:sufficiency-conp\]](#thm:sufficiency-conp){reference-type="ref" reference="thm:sufficiency-conp"} then maps an $n$-variable tautology instance to a sufficiency instance with the same asymptotic coordinate count. Therefore, if [Sufficiency-Check]{.smallcaps} were solvable in $2^{o(n)}$ time under the succinct encoding, then so would TAUTOLOGY and hence 3-SAT, contradicting ETH.

::: definition
There exists a constant $\delta > 0$ such that 3-SAT on $n$ variables cannot be solved in time $O(2^{\delta n})$ [@impagliazzo2001complexity].
:::

::: proposition
[]{#prop:eth-constant label="prop:eth-constant"} The reduction in Theorem [\[thm:sufficiency-conp\]](#thm:sufficiency-conp){reference-type="ref" reference="thm:sufficiency-conp"} preserves the number of variables up to an additive constant: an $n$-variable formula yields an $(n+1)$-coordinate decision problem. Therefore, the constant $c$ in the $2^{n/c}$ lower bound is asymptotically 1: $$\text{SUFFICIENCY-CHECK requires time } \Omega(2^{\delta (n-1)}) = 2^{\Omega(n)} \text{ under ETH}$$ where $\delta$ is the ETH constant for 3-SAT.
:::

::: proof
*Proof.* The TAUTOLOGY reduction (Theorem [\[thm:sufficiency-conp\]](#thm:sufficiency-conp){reference-type="ref" reference="thm:sufficiency-conp"}) constructs:

-   State space $S = \{\text{ref}\} \cup \{0,1\}^n$ with $n+1$ coordinates (one extra for the reference state)

-   Query set $I = \emptyset$

When $\varphi$ has $n$ variables, the constructed problem has $n+1$ coordinates. The asymptotic lower bound is $2^{\Omega(n)}$ with the same constant $\delta$ from ETH. ◻
:::

## Cross-Model Contrast

::: corollary
[]{#cor:phase-transition label="cor:phase-transition"} Across the explicit-state and succinct encodings of Section [\[sec:encoding\]](#sec:encoding){reference-type="ref" reference="sec:encoding"}, exact certification exhibits the following contrast:

-   If $k^* = O(\log N)$, SUFFICIENCY-CHECK is polynomial in $N$ under the explicit-state encoding

-   If $k^* = \Omega(n)$, SUFFICIENCY-CHECK is exponential in $n$ under ETH in the succinct encoding

For Boolean coordinate spaces ($N = 2^n$), one has $\log N = n$. So the explicit-state upper bound should be read as polynomial in the explicit table size when $k^* = O(n)$, while the succinct lower bound is exponential in the succinct variable count when $k^* = \Omega(n)$. The comparison is therefore between two encodings of the same Boolean family, not between $O(\log n)$ and $\Omega(n)$ inside a single model.
:::

::: proof
*Proof.* The logarithmic case (Part 1 of Theorem [\[thm:dichotomy\]](#thm:dichotomy){reference-type="ref" reference="thm:dichotomy"}) gives polynomial time when $k^* = O(\log N)$. More precisely, when $k^* \leq c \log N$ for constant $c$, the algorithm runs in time $O(N^c \cdot \text{poly}(n))$.

The linear case (Part 2) gives exponential time when $k^* = \Omega(n)$.

The transition point inside the explicit-state bound is where $2^{k^*} = N^{k^*/\log N}$ stops being polynomial in $N$, i.e., when $k^* = \omega(\log N)$. For Boolean coordinate spaces, however, the present theorem should be read only as a cross-encoding contrast, because $\log N = n$ and the succinct lower bound is parameterized directly by the succinct variable count. ◻
:::

::: remark
Under ETH, the lower bound is asymptotically tight in the succinct encoding. The explicit-state upper bound is tight in the sense that it matches enumeration complexity in $N$.
:::

This encoding-sensitive contrast explains why exact simplification can be straightforward on some instance families and computationally prohibitive on others. When the true sufficient support is tiny and the state space is explicit, exhaustive certification can still be polynomial. When the true support is extensive and the problem is given succinctly, the same certification task inherits a worst-case $2^{\Omega(n)}$ lower bound under ETH in addition to the class-level coNP hardness statement.

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

Our theory provides comprehensive coverage of tractable subcases. The artifact certifies explicit decision procedures for several families, and also certifies paper-specific reductions and complexity transfers connecting structurally interesting restrictions to standard polynomial-time algorithmic backbones. These subcases are not merely isolated exceptions. Each removes one of the structural sources of hardness in exact certification: unrestricted action comparison, cross-coordinate interaction, high-width dependency structure, or unnecessary state-space multiplicity.

  -----------------------------------------------------------------------------------------------------------------
  **Restriction**       **Why it helps**                                    **Complexity**
  --------------------- --------------------------------------------------- ---------------------------------------
  Low tensor rank       factored evaluation avoids full state enumeration   $O(|A| \cdot R \cdot n)$

  Tree structure        dynamic programming over dependency tree suffices   $O(n \cdot |A| \cdot k_{\max})$

  Bounded treewidth     interaction checking reduces to low-width CSP       $O(n \cdot k^{w+1})$

  Coordinate symmetry   orbit types replace raw state pairs                 $O\!\bigl(\binom{d+k-1}{k-1}^2\bigr)$
  -----------------------------------------------------------------------------------------------------------------

**Tractability map.** Each nontrivial restriction above removes a structural hardness source. Additional simple cases include bounded actions ($O(|S|^2 \cdot k^2)$ when $|A|$ is fixed, separable utility ($O(1)$), and constant or dominated optimal sets ($O(1)$). Formal definitions for all cases are retained in the subsections below.

::: theorem
[]{#thm:tractable label="thm:tractable"} The following structurally interesting tractable subcases are all mechanized in artifact. The artifact certifies paper-specific reductions and complexity transfers connecting each to standard polynomial-time algorithmic backbones. Formal definitions and trivial cases (single action, bounded state space, separable utility, dominance) are retained in the subsections below.

1.  **Low tensor rank:** The utility admits a rank-$R$ decomposition $U(a,s) = \sum_{r=1}^R w_r \cdot f_r(a) \cdot \prod_i g_{ri}(s_i)$. The artifact certifies reduction to factored computation with bound $O(|A| \cdot R \cdot n)$.

2.  **Tree-structured dependencies:** Coordinates are ordered such that $s_i$ depends only on $(s_1,\ldots, s_{i-1})$. The artifact certifies an explicit sufficiency checker for this class.

3.  **Bounded treewidth:** The interaction graph has treewidth $\le w$ and utility factors over edges. The artifact certifies reduction to bounded-treewidth CSP together with bound $O(n \cdot k^{w+1})$ inherited from standard CSP algorithm.

4.  **Coordinate symmetry:** Utility is invariant under coordinate permutations. The artifact certifies reduction to orbit-type representatives together with resulting orbit-count bound $O\bigl(\binom{d+k-1}{k-1}^2\bigr)$.
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
[]{#thm:tensor-rank label="thm:tensor-rank"} If utility has tensor rank $R$, the artifact certifies a reduction of the relevant optimization step to factored computation with bound $O(|A| \cdot R \cdot n)$.
:::

::: proof
*Proof.* The artifact certifies three ingredients. First, bounded-rank tensor contraction admits a polynomial bound Second, the low-rank utility representation reduces the relevant optimization step to factored computation in $O(|A| \cdot R \cdot n)$ steps Third, these are exactly the paper-specific ingredients needed to invoke the standard tensor-network algorithms cited in the text. Thus the formal support here is a certified reduction-and-transfer statement rather than an end-to-end formalization of the tensor-network backbone itself. ◻
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
[]{#thm:treewidth label="thm:treewidth"} If the interaction graph has treewidth $\leq w$ and utility is pairwise with explicit factors, the artifact certifies a reduction to bounded-treewidth CSP together with the standard complexity bound $O(n \cdot k^{w+1})$, where $k = \max_i |X_i|$.
:::

::: proof
*Proof.* The artifact certifies the paper-specific reduction from pairwise sufficiency checking to a CSP on the interaction graph It also certifies the transfer of the standard bounded-treewidth complexity bound Thus the formal support here is a certified reduction-and-transfer theorem: the bounded-treewidth algorithm itself is standard, while the paper-specific bridge to that algorithm is mechanized. ◻
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
[]{#thm:symmetric-complexity label="thm:symmetric-complexity"} Under symmetric utility, the artifact certifies that the number of orbit types is bounded by the stars-and-bars count: $$|\text{OrbitTypes}| = \binom{d + k - 1}{k - 1}$$ and that sufficiency checking reduces to cross-orbit comparisons. Thus the effective number of representative pair checks is at most $\binom{d + k - 1}{k - 1}^2$, which is polynomial in $d$ for fixed $k$.
:::

::: proof
*Proof.* An orbit type is a multiset of $d$ values from $\{0, \ldots, k-1\}$, equivalently a $k$-tuple of non-negative integers summing to $d$. By stars-and-bars, the count is $\binom{d + k - 1}{k - 1}$. The mechanized content here is the orbit-type reduction and the resulting representative-count bound; the resulting polynomial-time conclusion then follows from this certified compression of the comparison space.

For fixed $k$, this is $O(d^{k-1})$, polynomial in $d$. ◻
:::

## Practical Implications {#sec:tract-practical}

The formally developed tractable subcases correspond to common modeling scenarios. The structurally nontrivial cases each remove a distinct source of hardness:

-   **Bounded actions:** Small action sets (e.g., binary decisions, few alternatives). Brute-force enumeration is polynomial when $|A|$ is constant.

-   **Separable utility:** Additive cost models, linear utility functions. The optimizer is independent of state, making any coordinate set sufficient.

-   **Low tensor rank:** Factored preference models, low-rank approximations in recommendation systems or multi-attribute utility theory.

-   **Tree structure:** Hierarchical decision processes, causal models with chain or tree dependency. Dynamic programming replaces exhaustive search.

-   **Bounded treewidth:** Spatial models, graphical models with limited interaction width. CSP algorithms on low-treewidth graphs handle the sufficiency check.

-   **Coordinate symmetry:** Exchangeable features, order-invariant utilities (e.g., set-valued inputs). Orbit-type representatives collapse the comparison space.

Simple edge cases also yield immediate tractability without structural argument: single action ($|\mathrm{Opt}| = |A|$ trivially), bounded state space (exhaustive search is polynomial in $|S|$), strict dominance (one action always weakly optimal), and constant optimal set ($\mathrm{Opt}(s)$ independent of $s$).

For problems given in the succinct encoding without these structural restrictions, the hardness results of Section [\[sec:hardness\]](#sec:hardness){reference-type="ref" reference="sec:hardness"} apply, justifying heuristic approaches.


# Implications for Exact Certification {#sec:engineering-corollaries}

The regime hierarchy matters for more than class labels. The same exact question, which coordinates can be omitted without changing the decision boundary, reappears when one tries to simplify models, build reliable certifiers, check witnesses, approximate exact minimizers, or compress a central interface. This section collects those consequences in one place. The common lesson is that exact relevance is expensive to certify and easy to displace rather than remove. One downstream corollary is a simple trilemma for exact certifiers in the hard regime: once exact certification is hard, every interface-compression story must be read under the possibility of abstention, weakened guarantees, or overclaiming.

## Exact Simplification and Over-Specification

::: proposition
[]{#prop:config-sufficiency label="prop:config-sufficiency"} Let $P=\{p_1,\ldots,p_n\}$ be configuration parameters and let $B$ be a finite set of observable behaviors. Suppose each configuration state determines the subset of behaviors that remain feasible or optimal. Then the question $$\text{``Does parameter subset } I \subseteq P \text{ preserve all decision-relevant behavior?''}$$ is an instance of [[Sufficiency-Check]{.smallcaps}]{.nodecor}.
:::

::: proof
*Proof.* Construct a decision problem $\mathcal{D}=(A,X_1,\ldots,X_n,U)$ by taking actions $A=B$, coordinate domains $X_i$ equal to the domains of the parameters $p_i$, and defining $U(b,s)=1$ exactly when behavior $b$ is realized or admissible under configuration state $s$. Then $\operatorname{Opt}(s)$ is the behavior set induced by $s$, and coordinate subset $I$ is sufficient exactly when agreement on the parameters in $I$ forces agreement on the induced optimal-behavior set. ◻
:::

#### Concrete case study.

Consider a simplified service configuration with three parameters: cache mode $p_1\in\{\mathrm{off},\mathrm{on}\}$, retry policy $p_2\in\{1,3\}$, and replica count $p_3\in\{1,2\}$. Let the observable behavior set be $B=\{\mathrm{latency\text{-}ok},\mathrm{write\text{-}safe},\mathrm{cost\text{-}ok}\}$. Suppose enabling the cache affects only latency, increasing retries affects only write safety, and replica count affects write safety and cost. Then Theorem [\[prop:config-sufficiency\]](#prop:config-sufficiency){reference-type="ref" reference="prop:config-sufficiency"} turns the simplification question into an exact sufficiency question on the induced decision problem: which parameter subsets preserve the optimal-behavior set? In this toy instance, if the target simplification must preserve write safety and cost but is indifferent to latency, the exact behavior-preserving core is $\{p_2,p_3\}$, while $p_1$ is irrelevant to that target behavior set. The example is small enough to inspect by hand, but it is already an instance of the general Lean-backed reduction from behavior preservation to sufficiency.

### Formalized toy POMDP example

::: definition
[]{#def:one-step-pomdp label="def:one-step-pomdp"} A one-step POMDP is a tuple $(S,A,O,p,o,r)$ where $S,A,O$ are finite sets of states, actions, and observations, $p\in\Delta(S)$ is a prior distribution, $o:S\to O$ is the observation map, and $r:A\times S\to\mathbb{R}$ is the immediate reward function. The horizon is one, so the utility of taking action $a$ in true state $s$ is $r(a,s)$.
:::

Write $\operatorname{Opt}_{\mathrm{full}}(s):=\arg\max_{a\in A} r(a,s)$ for the full-information optimal-action set and, for an observation value $o\in O$, set $$\operatorname{Opt}_{\mathrm{obs}}(o):=\arg\max_{a\in A} \mathbb{E}_{S\sim p}[r(a,S)\mid o(S)=o].$$

Let $g:O\to O'$ be any coarsening of observations and write $\phi:=g\circ o:S\to O'$ for the induced coarse signal.

::: proposition
[]{#prop:pomdp-to-preservation label="prop:pomdp-to-preservation"} Let $(S,A,O,p,o,r)$ be a one-step POMDP and let $\phi:S\to O'$ be as above. Define the decision problem $\mathcal{D}=(A,S,U)$ with $U(a,s):=r(a,s)$ and state distribution $p$. Then $\phi$ is stochastically preserving for $\mathcal{D}$ in the sense of Section [\[sec:stochastic\]](#sec:stochastic){reference-type="ref" reference="sec:stochastic"} if and only if for every $s\in S$ we have $$\operatorname{Opt}_{\mathrm{full}}(s)=\operatorname{Opt}_{\mathrm{obs}}(\phi(s)).$$ In particular, checking whether the coarse observation $\phi$ preserves Bayes-optimal actions is exactly an instance of our stochastic preservation predicate for $\mathcal{D}$.
:::

::: proof
*Proof.* By construction $U(a,s)=r(a,s)$ so the full-information optimal-action set in $\mathcal{D}$ equals $\operatorname{Opt}_{\mathrm{full}}(s)$. The coarse-induced optimizer for $\phi$ is, by definition, the set of actions maximizing the conditional expectation of $U$ given the coarse signal. That conditional expectation is exactly the Bayes expectation used to define $\operatorname{Opt}_{\mathrm{obs}}$. Therefore the preservation condition from Section [\[sec:stochastic\]](#sec:stochastic){reference-type="ref" reference="sec:stochastic"} (equality of full-information and coarse-induced optimizers on every state) is equivalent to the displayed identity. ◻
:::

#### Worked numeric instance.

Consider the one-step POMDP with states $S=\{s_1,s_2\}$, actions $A=\{a,b\}$, prior $p(s_1)=p(s_2)=1/2$, observation map $o(s_1)=o_1,o(s_2)=o_2$, and reward table $$r(a,s_1)=2,\quad r(b,s_1)=1,\qquad r(a,s_2)=0,\quad r(b,s_2)=3.$$ Let $g:O\to O'$ be the constant map that coarsens both observations to a single symbol. Then for the coarse signal $\phi=g\circ o$ the conditional expected returns are $$\mathbb{E}[r(a,S)\mid\phi]=\tfrac12(2+0)=1,\qquad \mathbb{E}[r(b,S)\mid\phi]=\tfrac12(1+3)=2.$$ Thus the coarse-induced optimizer is $\{b\}$ while the full-information optimizer at $s_1$ is $\{a\}$, so $\phi$ does not preserve Bayes-optimal actions for this instance. This tiny numeric example illustrates how coarse observations can change the optimal-action correspondence and therefore fail the stochastic preservation predicate.

This gives a direct, elementary encoding of one-step planning questions as stochastic preservation instances. The explicit-state encoding of the POMDP yields an explicit-state preservation instance, while a succinct encoding of transition or observation structure yields a succinct encoding of the corresponding decision problem, explaining how the same modelling choices generate the different complexity behaviors reported in Section [\[sec:stochastic\]](#sec:stochastic){reference-type="ref" reference="sec:stochastic"}.

### Formalized reinforcement-learning hyperparameter example

Fix a finite environment set $E$ and finite hyperparameter domains $X_{\alpha},X_{\gamma},X_{\epsilon}$. Let $$H:=X_{\alpha}\times X_{\gamma}\times X_{\epsilon}$$ be the set of hyperparameter configurations. For each environment $e\in E$ write $f_e:H\to\mathbb{R}$ for the expected return of configuration $h\in H$.

Define the derived decision problem $$\mathcal{D}_{\mathrm{hp}}=(A,H,U),\qquad A:=E,\quad U(e,h):=f_e(h).$$ Here states are hyperparameter configurations and actions are environments; the utility records return when configuration $h$ is evaluated under environment $e$.

Let $\pi:H\to H':=X_{\alpha}\times X_{\epsilon}$ be the projection that drops the $\gamma$ coordinate. Call $\gamma$ redundant for the tuning task if for every environment $e$ the set of maximizers $$\operatorname{Opt}_e:=\arg\max_{h\in H} f_e(h)$$ projects under $\pi$ to the set of maximizers of the collapsed objective $$g_e(h'):=\max_{\gamma\in X_{\gamma}} f_e(h',\gamma),\qquad h'\in H'.$$

::: proposition
[]{#prop:hp-to-static label="prop:hp-to-static"} With notation as above, the statement \"$\gamma$ is redundant for tuning\" is equivalent to a static preservation property for the decision problem $\mathcal{D}_{\mathrm{hp}}$ under the action-space projection $\pi$. Concretely, $\gamma$ is redundant if and only if for every $e\in E$ and every $h\in H$ we have $$h\in\operatorname{Opt}_e \quad\Longleftrightarrow\quad \exists h_0\in\operatorname{Opt}_e\text{ with }\pi(h_0)=\pi(h),$$ which is exactly the condition that the projection $\pi$ preserves the optimal-action correspondence when the decision problem is read with environments as the existential choices.
:::

::: proof
*Proof.* By definition $\gamma$ is redundant when every full maximizer projects to a maximizer of the collapsed objective and conversely every maximizer of the collapsed objective has a witness in the full space; the displayed equivalence is a restatement of that property. The left-to-right implication is immediate from the definition of $g_e$, and the right-to-left implication follows because if some projected maximizer has a preimage that attains the collapsed optimum, then there exists a full configuration achieving the same maximum in the original objective. Rewriting these statements as equalities of optimal-action sets for the derived decision problem $\mathcal{D}_{\mathrm{hp}}$ yields the claimed reduction to a static-style preservation check. ◻
:::

The construction above is explicit and elementary. It shows that hyperparameter redundancy questions can be phrased exactly as decision-preservation predicates within our static regime, and that succinct encodings of the environment family or return functions translate into succinct encodings of the corresponding static instance. In practice this lets one import the static-regime hardness and tractability statements directly into the hyperparameter setting.

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

Once exact certification is hard, the natural systems question is not only whether the predicate can be computed, but whether a solver can stay exact, non-abstaining, and polynomial-budgeted on the whole hard regime. The result below is a direct complexity-theoretic corollary: those three goals are incompatible. We refer to this incompatibility as the *trilemma of exact certification*.

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

::: definition
For the empty-set sufficiency core on Boolean state space $S=\{0,1\}^n$, a *slot-inspection checker* is a deterministic algorithm that adaptively queries witness slots of the form $$\{(0,z),(1,z)\}\qquad z\in\{0,1\}^{n-1},$$ and, for each queried slot, learns whether the optimizer agrees or disagrees on the two states in that slot. The checker is *sound* if it never accepts a non-sufficient instance and never rejects a sufficient one.
:::

The lower bound below is for deterministic checkers in exactly this access model.

::: theorem
[]{#thm:witness-lower-bound-4 label="thm:witness-lower-bound-4"} For Boolean decision problems with $n$ coordinates, any sound slot-inspection checker for the empty-set sufficiency core must inspect at least $2^{n-1}$ witness pairs in the worst case.
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

The identity below is elementary, but its consequence is not: any exact simplification that removes relevant coordinates from the central interface must discharge their effect somewhere else in the system.

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

The present artifact is intended as a problem-specific certification layer within that broader program. It does not claim to settle formal complexity theory in full generality; instead, it internalizes the reduction-correctness lemmas, size-bounded hardness packages, exact finite deciders, counted-search procedures, tractability wrappers, explicit-state upper-bound interfaces, and the finite witness/checking machinery now used for the stochastic $\textsf{NP}^{\textsf{PP}}$ upper-bound story. In that sense the contribution is closest to a certified reduction-and-decision core for one theorem family rather than to a general-purpose complexity library.

## Rough Sets, Reducts, and Attribute Reduction

The closest classical neighbor is rough-set theory and its large literature on reducts and attribute reduction [@pawlak1982rough; @jensen2004semantics; @jensen2008rough]. That literature asks when a smaller attribute set preserves discernibility or decision-table structure, and it has developed both exact and heuristic methods for reduct computation.

The complexity of computing minimal reducts has been studied in the rough-set literature. Finding a minimal attribute set that preserves decision-distinctions is known to be NP-hard in general [@slowinski1995decision], and more refined analysis places certain reduct variants in $\Sigma_2^P$ [@multiplier]. These results establish that the static regime questions studied here, whether a coordinate set preserves decision distinctions, and whether a minimal such set exists, fall within a known complexity landscape for attribute reduction.

The key structural difference is this: general decision tables can admit multiple incomparable minimal reducts, necessitating a combinatorial search through the reduct lattice. Under the exact optimal-action preservation predicate studied here, Proposition [\[prop:minimal-relevant-equiv\]](#prop:minimal-relevant-equiv){reference-type="ref" reference="prop:minimal-relevant-equiv"} proves there is exactly one minimal sufficient set: the relevant-coordinate set itself. This uniqueness is the structural reason for the complexity collapse. The search layer vanishes, reducing exact minimization to parallel coNP relevance checks. To our knowledge, this specific structural collapse, from $\Sigma_2^P$ search to coNP relevance containment driven by optimizer-quotient uniqueness, is a novel classification not captured by general rough-set bounds.

There is an obvious family resemblance to the static regime of the present paper: both settings ask which coordinates can be removed without losing decision distinctions.

::: proposition
[]{#prop:static-rough-set-bridge label="prop:static-rough-set-bridge"} Given a finite decision problem $\mathcal{D}=(A,X_1,\ldots,X_n,U)$, form the induced decision table whose condition attributes are the coordinates $X_i$ and whose decision label at state $s$ is the set-valued class label $\operatorname{Opt}(s) \subseteq A$. Then a coordinate set $I$ is sufficient in the sense of this paper if and only if the projection to $I$ preserves the decision label of the induced table. Consequently, the minimal sufficient sets of $\mathcal{D}$ are exactly the reducts of this induced decision table.
:::

::: proof
*Proof.* By definition, $I$ is sufficient exactly when any two states agreeing on $I$ have the same optimal-action set. In the induced decision table, the decision label of state $s$ is precisely $\operatorname{Opt}(s)$, so the same condition says that projection to $I$ preserves the decision label. Minimality on the present side is therefore exactly reduct minimality for the induced table. ◻
:::

The difference is therefore not merely terminological. Rough-set reducts are typically formulated relative to indiscernibility relations or decision tables, whereas the object here is the optimizer map of a decision problem and the induced exact preservation of optimal-action sets. Proposition [\[prop:static-rough-set-bridge\]](#prop:static-rough-set-bridge){reference-type="ref" reference="prop:static-rough-set-bridge"} shows that the static regime can be recast as reduct preservation for the induced decision table. What the present paper adds is: (i) the collapse theorem showing minimum sufficiency collapses to relevance containment, bypassing the general NP-hardness barrier for exact minimization; and (ii) the systematic extension to stochastic and sequential regimes, which takes the reduct question beyond the classical rough-set setting. The stochastic and sequential layers introduce conditional-comparison and temporal-structure phenomena that are outside the usual rough-set formulation.

The same generosity is due to the broader feature-selection and subset-selection literature in machine learning and combinatorial optimization [@blum1997selection; @amaldi1998complexity]. Those works study sparse predictive representations, informative feature subsets, and the computational cost of choosing them. Our setting differs in targeting exact preservation of the optimal-action correspondence rather than prediction loss, classifier complexity, or approximate empirical utility. Still, the shared concern is clear: identifying what information matters and what can be safely omitted.

## Decision-Theoretic Sufficiency, Informativeness, and Value of Information

There is also a strong conceptual connection to statistical decision theory and the theory of informativeness. Classical sufficient statistics, value-of-information analysis, and comparison of experiments ask when one information structure is at least as useful for decision making as another [@blackwell1953equivalent; @savage1954foundations; @raiffa1961applied; @howard1966information]. That tradition studies when information can be compressed, compared, or discarded without harming decisions. The optimizer quotient and the exact sufficiency predicate in this paper should be read against that background.

At the same time, the present work studies a different layer of the problem. The focus here is not statistical estimation, asymptotic identifiability, or expected utility under a fixed experiment class. It is the computational certification problem of proving that a coordinate projection preserves the optimal-action correspondence exactly. The connection to classical decision theory is therefore one of ancestry and motivation rather than direct theorem transfer. The paper owes that literature recognition for posing, in older language, the same high-level question: what information is genuinely needed for a decision?

## Abstraction, Bisimulation, and Homomorphisms in Planning and Reinforcement Learning

The stochastic and sequential parts of the paper sit near a second major literature: exact abstraction in Markov decision processes, POMDPs, bisimulation-based aggregation, and MDP homomorphisms [@papadimitriou1987mdp; @littman1998probplanning; @mundhenk2000mdp; @givan2003equivalence; @ravindran2004algebraic]. This body of work studies when state spaces can be aggregated while preserving value or policy structure, and it has established many of the representation-sensitive complexity phenomena that make richer stochastic and sequential models computationally difficult.

That literature is structurally very close to the present paper and should be credited as such. The main difference is that our predicate is coordinate sufficiency for preserving optimal-action sets under an explicit coordinate-hiding operation. We do not study arbitrary abstract state maps, approximate bisimulation metrics, or value-function approximation guarantees. Instead, we ask for exact certification that a chosen coordinate interface is enough. This exact-certification emphasis is what the paper adds beyond prior abstraction results: the question is not only whether a smaller representation can support good control, but whether one can certify that no omitted coordinate changes the optimal-action correspondence. The contribution here is therefore not abstraction theory in general, but a unified exact-certification account in which static, stochastic, and sequential regimes are analyzed as variants of one decision-relevance question.

## Explanations, Anchors, and Minimal Decision Rules

The paper should also acknowledge adjacent work in explainable AI and local rule-based explanations, especially anchor-style explanations [@ribeiro2018anchors]. Readers will naturally notice the word "anchor" and map it to that literature. The resemblance is real: both settings isolate a restricted condition under which a local piece of information suffices to support a decision or prediction. More broadly, explanation methods based on local rules, counterfactuals, or minimal sufficient subsets share the same informal ambition of identifying a small decision-preserving core.

The difference is again exactness and scope. Anchor explanations in XAI are typically local, model-agnostic, approximate, or precision-calibrated. Our anchor-sufficiency problem is an exact certification problem inside a fully specified decision model, and its complexity is governed by an existential choice followed by a universal fiberwise verification condition. So the paper is not an explanation method, but it is in direct conversation with explanation work that asks what small local structure is enough to preserve a decision.

There is an even closer logical neighbor in recent work on reasons behind decisions, prime-implicant style explanations, and exact decision justifications [@darwiche2023reasons]. This is especially worth noting because it shows that exact small decision-preserving structures are a live contemporary topic, not only a classical one. That literature is concerned with exact structural reasons for a decision outcome and therefore lies much nearer to the present paper than generic post-hoc explanation methods do. The overlap is substantial at the conceptual level: both ask for exact small structures that preserve a decision. The difference is that our objects are coordinate sufficiency, minimum sufficiency, and anchor sufficiency inside coordinate-structured decision problems, and the paper's main contribution is a regime-sensitive complexity classification of these predicates rather than a logical semantics of reasons alone.

## Abduction, Diagnosis, and Exact Explanatory Cores

The minimum and anchor queries share structural kinship with abduction and diagnosis frameworks, which study explanatory hypotheses sufficient for a target consequence. The anchor query retains an outer existential choice analogous to explanatory discovery, while the inner verification condition preserves optimizer structure. This logical-complexity intuition motivates the $\Sigma_2^P$-completeness of static anchor sufficiency, which remains hard even when minimum sufficiency collapses to coNP.

## Causality, Responsibility, and Structural Explanation

The paper also belongs in conversation with structural accounts of causality, explanation, and responsibility [@pearl2009causality; @spirtes2000causation; @halpern2001explanations; @chockler2004responsibility]. A relevant coordinate is precisely a coordinate whose perturbation can change the decision, and the optimizer quotient isolates exactly the distinctions that matter for optimal-action structure. The paper focuses on exact decision preservation under coordinate projection and on the computational cost of certifying that preservation, whereas causality and responsibility frameworks typically emphasize semantic characterization or intervention-level explanation.

## Oracle, Query, and Access-Based Lower Bounds

Query-access lower bounds provide another nearby technique family [@dobzinski2012query]. Our witness-checking duality and access-obstruction results belong to that tradition in spirit. The difference is that the lower bounds are developed around the same fixed sufficiency predicate used throughout the paper, which allows direct comparison between forward evaluation, certification, exact minimization, and restricted-access models without changing the underlying decision relation.

## Scope and Novelty

The paper is therefore intentionally synthetic in the good sense. Rough sets and reducts study decision-preserving attribute reduction; feature selection studies informative subsets; statistical decision theory studies informativeness and sufficiency; abstraction and bisimulation literatures study exact aggregation of stochastic and sequential decision processes; explanation methods study local decision-preserving cores; formalized complexity work studies certified reductions and machine-checked decision procedures. The contribution here is to treat exact relevance certification itself as the common object and to develop one coherent complexity-theoretic account around it.

What is distinctive in the present paper is not an isolated static or sequential hardness theorem taken on its own. The static regime can be recast as reduct preservation for the induced decision table, and the sequential row belongs near existing exact abstraction phenomena in richer control models. The contribution here is to formalize one exact decision-preservation predicate in optimizer language, track it coherently across static, stochastic, and sequential regimes, expose the preservation/decisiveness split in the stochastic case, identify the tractable and intractable boundaries of exact certification, and support the resulting finite combinatorial core with mechanized verification. The paper's novelty is therefore best understood as a unified exact-certification theory rather than as a single isolated theorem.


# Conclusion

This paper develops a regime-sensitive theory of exact relevance certification for coordinate-structured decision problems. Its aim is to characterize what information a decision actually depends on, how that information is represented by the optimizer quotient, and when exact certification of that dependence is tractable, split, or fundamentally limited across static, stochastic, and sequential regimes.

## Main Results {#main-results .unnumbered}

Within the encoding model of Section [\[sec:encoding\]](#sec:encoding){reference-type="ref" reference="sec:encoding"}, we establish:

-   **Static structure and regime matrix:** in the static regime, the exact information content of the decision is the relevant-coordinate set itself: sufficient sets are exactly its supersets, so exact minimization reduces to relevance containment and is coNP-complete rather than a genuine $\Sigma_{2}^{P}$ search problem, while [Anchor-Sufficiency]{.smallcaps} remains $\Sigma_{2}^{P}$-complete. In the stochastic regime, preservation (stochastic sufficiency) has a polynomial-time base query in the explicit-state model with proved bridges to static sufficiency and the optimizer quotient; the minimum and anchor preservation variants are likewise polynomial-time in explicit state, inherit coNP-completeness and $\Sigma_{2}^{P}$-completeness under full support, and admit support-sensitive obstruction and bridge lemmas beyond full support. The obstruction to general-distribution classification is structural: preservation quantifies over all states, while stochastic analysis is driven by probability-weighted fibers. Zero-probability states can witness violations that the distribution itself never observes, so full-support results do not transfer directly. This is not merely an unproven gap. The support-sensitive obstruction lemmas in the artifact isolate the exact combinatorial mechanism. Decisiveness (stochastic decisiveness) has a polynomial-time base query in the explicit-state model and is PP-hard under succinct encoding; the anchor and minimum decisiveness queries are PP-hard and lie in $\textsf{NP}^{\textsf{PP}}$, with those oracle-class memberships proved in the paper text and their finite witness/checking cores mechanized. In the sequential regime, sufficiency, minimum, and anchor queries are PSPACE-complete.

-   **Encoding-sensitive contrast for exact certification:** logarithmic-size sufficient sets admit polynomial-time algorithms in the explicit regime, while linear-size sufficient sets inherit ETH-conditioned exponential lower bounds in the succinct regime.

-   **Structural and downstream consequences:** the paper derives witness and approximation obstructions, exact-simplification hardness transfers, and a direct reliability corollary for exact certifiers in the hard regime: no exact certifier can simultaneously be sound, complete, and polynomial-budgeted on that regime. A simplicity-tax framing shows that omitted exact relevance is externalized rather than eliminated.

-   **Structured tractability:** tractable subcases across two tiers: simple edge cases (single action, bounded state space, strict dominance, constant optimal set) admitting immediate checks, and structurally nontrivial cases (bounded actions, separable utility, low tensor rank, tree structure, bounded treewidth, coordinate symmetry) each removing a distinct hardness source with a mechanized reduction or decision procedure.

Taken together, these results show that exact relevance certification is a distinct complexity-theoretic object in its own right: computing $\operatorname{Opt}(s)$ on one state can be easy even when proving coordinate irrelevance is hard, and the difficulty of that certification problem depends sharply on whether one is ruling out counterexamples, preserving optimizers under coarser conditioning, checking decisiveness of conditional fibers, or reasoning about temporally structured optimizer behavior.

## Implications {#implications .unnumbered}

The implications are direct hardness transfers. Configuration simplification is an instance of sufficiency checking, so exact behavior-preserving minimization has no general-purpose polynomial-time worst-case solution unless $P=coNP$. The same issue appears in real pipeline orchestration settings, where a nested configuration object controls well selection, output layout, and downstream materialization. There too the practical question is whether one can hide part of the configuration interface without changing the induced compilation or execution decisions. Our results say that this is not merely a UI-cleanup problem; it is an exact certification problem whose worst-case difficulty depends on whether one is in the static, stochastic, or sequential regime. Reliable exact certification is likewise regime-limited: under $P\neq coNP$, the trilemma of exact certification rules out any solver that is simultaneously sound, complete on the full hard regime, and polynomial-budgeted. And once the canonical relevant-coordinate set $R(\mathcal{D})$ is fixed, any compressed interface merely partitions that relevance into centrally handled and externalized parts. Compression therefore relocates exact relevance rather than erasing it; under the per-site cost model, the unresolved part grows linearly with deployment scale.

This is also the practical meaning of the simplicity tax. In the hard exact regime, a cheap simplification procedure cannot be assumed to have removed decision-relevant structure merely because it has compressed the visible interface. Any relevance omitted from the certified core must still be paid for somewhere else in the system: by local resolution, extra queries, abstention, or weaker guarantees. Exact simplification is therefore costly, but inexact simplification is not free; it externalizes the unresolved burden.

## Mechanization {#mechanization .unnumbered}

The proof artifact mechanically checks the main reduction constructions, hardness packages, finite deciders, search procedures, tractability statements, step-counting wrappers, the stochastic sufficiency bridge package, the explicit-state decision procedures for the preservation family, the full-support inheritance results, the new support-sensitive obstruction and bridge lemmas for preservation, the decisiveness upper bounds, and the finite witness/checking schemata for the stochastic existential classification Full oracle-machine formalization in Lean remains outside the current state of formalized complexity theory; the paper proves the oracle-class memberships in the text following standard conventions, while the artifact provides independent verification of the finite combinatorial core. This places the mechanization in the emerging category of problem-specific certified reduction infrastructure.

## Outlook {#outlook .unnumbered}

Two immediate directions remain. First, the reduction infrastructure can be integrated more tightly with general-purpose formalized complexity libraries. Second, the artifact boundary can be made even cleaner by continuing to package the existing finite-decision results into more uniform summary interfaces without changing the paper's complexity claims. On the complexity side, the remaining gap is now sharply localized: stochastic decisiveness already has succinct hardness and a verified witness/checking core for the oracle-class memberships proved in the paper text, but a fully standard oracle-machine formalization of the resulting oracle-class placement and of the $\textsf{NP}^{\textsf{PP}}$ boundary for the anchor and minimum decisiveness queries remains outside the current repository; on the preservation side, Conjecture 4.1 and Open Problem 4.2 sharpen the remaining open terrain to the full general-distribution and succinct-encoding complexity of the minimum and anchor preservation variants. The mechanized artifact already exposes the finite combinatorial core and support-sensitive lemmas that isolate this obstruction; see `lean_handle_ids_auto.tex` and `claim_mapping_auto.tex` for handles such as OU12, DC96, and DC98, which are appropriate citations when framing the open problem.

## Acknowledgments {#acknowledgments .unnumbered}

Generative AI tools (including Codex, Claude Code, Augment, Kilo, and OpenCode) were used throughout this manuscript, across all sections (Abstract, Introduction, theoretical development, proof sketches, applications, conclusion, and appendix) and across all stages from initial drafting to final revision. The tools were used for boilerplate generation, prose and notation refinement, LaTeX/structure cleanup, translation of informal proof ideas into candidate formal artifacts (Lean/LaTeX), and repeated adversarial reviewer-style critique passes to identify blind spots and clarity gaps.

The author retained full intellectual and editorial control, including problem selection, theorem statements, assumptions, novelty framing, acceptance criteria, and final inclusion/exclusion decisions. No technical claim was accepted solely from AI output. Formal claims reported as machine-verified were admitted only after Lean verification (no `sorry` in cited modules) and direct author review; Lean was used as an integrity gate for responsible AI-assisted research. The author is solely responsible for all statements, citations, and conclusions.


# Lean Handle Index {#app:lean}

The table below lists the auto-generated Lean handle identifiers cited inline in the paper.

## Mechanized Witness Schemas and Oracle-Class Arguments {#mechanized-witness-schemas-and-oracle-class-arguments .unnumbered}

The paper uses standard oracle-class language for several stochastic upper bounds, especially the $\textsf{coNP}^{\textsf{PP}}$-style complement packaging for decisiveness and the $\textsf{NP}^{\textsf{PP}}$ upper bounds for the anchor and minimum decisiveness queries. The corresponding oracle-class memberships are proved in the paper text, following standard complexity-theoretic proof conventions. The proof artifact complements those proofs by mechanizing the finite witness/checking schemata and reduction cores that the oracle arguments require.

Concretely, the mechanized layer certifies ingredients of the following form: a bad fiber witnesses failure of decisiveness; an existentially guessed coordinate set or anchor witness reduces the anchor/minimum queries to the corresponding decisiveness-style verifier; and the bounded explicit-state procedures agree with the abstract predicates they are intended to decide. The paper then translates these certified witness/checking packages into the stated oracle-class membership arguments by the usual guess-and-query reasoning in complexity theory. Thus the artifact provides independent verification of the finite combinatorial core, while the oracle-class interpretation is established in the manuscript.


::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: list
::: {#lh:AB2}
`AB2`
:::

::: {#lh:CC1}
`CC1`
:::

::: {#lh:CR1}
`CR1`
:::

::: {#lh:CT12}
`CT12`
:::

::: {#lh:DC19}
`DC19`
:::

::: {#lh:DC23}
`DC23`
:::

::: {#lh:DC28}
`DC28`
:::

::: {#lh:DC29}
`DC29`
:::

::: {#lh:DC40}
`DC40`
:::

::: {#lh:DC41}
`DC41`
:::

::: {#lh:DC42}
`DC42`
:::

::: {#lh:DC44}
`DC44`
:::

::: {#lh:DC45}
`DC45`
:::

::: {#lh:DC46}
`DC46`
:::

::: {#lh:DC47}
`DC47`
:::

::: {#lh:DC48}
`DC48`
:::

::: {#lh:DC49}
`DC49`
:::

::: {#lh:DC50}
`DC50`
:::

::: {#lh:DC57}
`DC57`
:::

::: {#lh:DC62}
`DC62`
:::

::: {#lh:DC63}
`DC63`
:::

::: {#lh:DC64}
`DC64`
:::

::: {#lh:DC65}
`DC65`
:::

::: {#lh:DC66}
`DC66`
:::

::: {#lh:DC67}
`DC67`
:::

::: {#lh:DC68}
`DC68`
:::

::: {#lh:DC69}
`DC69`
:::

::: {#lh:DC70}
`DC70`
:::

::: {#lh:DC71}
`DC71`
:::

::: {#lh:DC72}
`DC72`
:::

::: {#lh:DC73}
`DC73`
:::

::: {#lh:DC74}
`DC74`
:::

::: {#lh:DC75}
`DC75`
:::

::: {#lh:DC76}
`DC76`
:::

::: {#lh:DC80}
`DC80`
:::

::: {#lh:DC81}
`DC81`
:::

::: {#lh:DC82}
`DC82`
:::

::: {#lh:DC91}
`DC91`
:::

::: {#lh:DC92}
`DC92`
:::

::: {#lh:DC93}
`DC93`
:::

::: {#lh:DC94}
`DC94`
:::

::: {#lh:DC95}
`DC95`
:::

::: {#lh:DC96}
`DC96`
:::

::: {#lh:DC97}
`DC97`
:::

::: {#lh:DC98}
`DC98`
:::

::: {#lh:DC99}
`DC99`
:::

::: {#lh:DC100}
`DC100`
:::

::: {#lh:DN1}
`DN1`
:::

::: {#lh:DN2}
`DN2`
:::

::: {#lh:DN5}
`DN5`
:::

::: {#lh:DP1}
`DP1`
:::

::: {#lh:DP2}
`DP2`
:::

::: {#lh:DP6}
`DP6`
:::

::: {#lh:DP7}
`DP7`
:::

::: {#lh:DQ1}
`DQ1`
:::

::: {#lh:DQ2}
`DQ2`
:::

::: {#lh:DQ3}
`DQ3`
:::

::: {#lh:DQ6}
`DQ6`
:::

::: {#lh:DQ7}
`DQ7`
:::

::: {#lh:DQ9}
`DQ9`
:::

::: {#lh:DQ10}
`DQ10`
:::

::: {#lh:DQ11}
`DQ11`
:::

::: {#lh:DQ12}
`DQ12`
:::

::: {#lh:DQ13}
`DQ13`
:::

::: {#lh:DQ14}
`DQ14`
:::

::: {#lh:DQ15}
`DQ15`
:::

::: {#lh:DQ16}
`DQ16`
:::

::: {#lh:DQ18}
`DQ18`
:::

::: {#lh:DQ19}
`DQ19`
:::

::: {#lh:DQ20}
`DQ20`
:::

::: {#lh:DQ25}
`DQ25`
:::

::: {#lh:DQ26}
`DQ26`
:::

::: {#lh:DQ27}
`DQ27`
:::

::: {#lh:DQ28}
`DQ28`
:::

::: {#lh:DQ30}
`DQ30`
:::

::: {#lh:DQ31}
`DQ31`
:::

::: {#lh:DQ32}
`DQ32`
:::

::: {#lh:DQ34}
`DQ34`
:::

::: {#lh:DQ36}
`DQ36`
:::

::: {#lh:DQ38}
`DQ38`
:::

::: {#lh:DQ39}
`DQ39`
:::

::: {#lh:DQ40}
`DQ40`
:::

::: {#lh:DQ42}
`DQ42`
:::

::: {#lh:DQ43}
`DQ43`
:::

::: {#lh:DQ44}
`DQ44`
:::

::: {#lh:DQ45}
`DQ45`
:::

::: {#lh:DQ51}
`DQ51`
:::

::: {#lh:DQ56}
`DQ56`
:::

::: {#lh:DQ57}
`DQ57`
:::

::: {#lh:DQ58}
`DQ58`
:::

::: {#lh:DQ62}
`DQ62`
:::

::: {#lh:DQ63}
`DQ63`
:::

::: {#lh:DQ72}
`DQ72`
:::

::: {#lh:DQ74}
`DQ74`
:::

::: {#lh:DQ75}
`DQ75`
:::

::: {#lh:DQ76}
`DQ76`
:::

::: {#lh:DQ77}
`DQ77`
:::

::: {#lh:DQ78}
`DQ78`
:::

::: {#lh:DQ79}
`DQ79`
:::

::: {#lh:DQ80}
`DQ80`
:::

::: {#lh:DQ81}
`DQ81`
:::

::: {#lh:DQ82}
`DQ82`
:::

::: {#lh:DQ83}
`DQ83`
:::

::: {#lh:DQ84}
`DQ84`
:::

::: {#lh:DQ85}
`DQ85`
:::

::: {#lh:EH1}
`EH1`
:::

::: {#lh:EH2}
`EH2`
:::

::: {#lh:EH3}
`EH3`
:::

::: {#lh:EH4}
`EH4`
:::

::: {#lh:EH5}
`EH5`
:::

::: {#lh:EH6}
`EH6`
:::

::: {#lh:EH7}
`EH7`
:::

::: {#lh:EH8}
`EH8`
:::

::: {#lh:EH9}
`EH9`
:::

::: {#lh:EH10}
`EH10`
:::

::: {#lh:EX1}
`EX1`
:::

::: {#lh:EX2}
`EX2`
:::

::: {#lh:EX3}
`EX3`
:::

::: {#lh:HD4}
`HD4`
:::

::: {#lh:HD5}
`HD5`
:::

::: {#lh:HD14}
`HD14`
:::

::: {#lh:HD23}
`HD23`
:::

::: {#lh:HD26}
`HD26`
:::

::: {#lh:IC30}
`IC30`
:::

::: {#lh:IC32}
`IC32`
:::

::: {#lh:IC33}
`IC33`
:::

::: {#lh:IC34}
`IC34`
:::

::: {#lh:OU1}
`OU1`
:::

::: {#lh:OU2}
`OU2`
:::

::: {#lh:OU6}
`OU6`
:::

::: {#lh:OU7}
`OU7`
:::

::: {#lh:OU8}
`OU8`
:::

::: {#lh:OU9}
`OU9`
:::

::: {#lh:OU10}
`OU10`
:::

::: {#lh:OU11}
`OU11`
:::

::: {#lh:OU12}
`OU12`
:::

::: {#lh:PA3}
`PA3`
:::

::: {#lh:QT1}
`QT1`
:::

::: {#lh:QT7}
`QT7`
:::

::: {#lh:WD1}
`WD1`
:::

::: {#lh:WD2}
`WD2`
:::

::: {#lh:WD3}
`WD3`
:::
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ID              | Handle                                                                                       | ID             | Handle                                                                                    |
+:================+:=============================================================================================+:===============+:==========================================================================================+
| ID              | Handle                                                                                       | ID             | Handle                                                                                    |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:AB2}   | `DecisionProblem.surjective_abstraction_factors_or_erases`                                   | ::: {#lh:CC1}  | `ClaimClosure.DQ8`                                                                        |
| `AB2`           |                                                                                              | `CC1`          |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:CR1}   | `DecisionQuotient.ConfigReduction.config_sufficiency_iff_behavior_preserving`                | ::: {#lh:CT12} | `DecisionQuotient.Physics.ClaimTransport.physical_bridge_bundle`                          |
| `CR1`           |                                                                                              | `CT12`         |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:DC19}  | `StochasticSequential.stochastic_anchor_sufficient_of_stochastic_sufficient`                 | ::: {#lh:DC23} | `StochasticSequential.sequential_anchor_sufficient_of_sequential_sufficient`              |
| `DC19`          |                                                                                              | `DC23`         |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:DC28}  | `StochasticSequential.reduceTQBF_correct_anchor`                                             | ::: {#lh:DC29} | `StochasticSequential.reduceTQBF_to_sequential_anchor_reduction`                          |
| `DC28`          |                                                                                              | `DC29`         |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:DC40}  | `StochasticSequential.reduceMAJSATPureAnchor_correct`                                        | ::: {#lh:DC41} | `StochasticSequential.reduceMAJSAT_to_pure_stochastic_anchor_reduction`                   |
| `DC40`          |                                                                                              | `DC41`         |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:DC42}  | `StochasticSequential.stochastic_anchor_check_pp_hard`                                       | ::: {#lh:DC44} | `StochasticSequential.sequential_anchor_check_pspace_hard`                                |
| `DC42`          |                                                                                              | `DC44`         |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:DC45}  | `StochasticSequential.stochastic_sufficiency_pp_hard`                                        | ::: {#lh:DC46} | `StochasticSequential.sequential_sufficiency_pspace_hard`                                 |
| `DC45`          |                                                                                              | `DC46`         |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:DC47}  | `StochasticSequential.stochastic_minimum_sufficiency_pp_hard`                                | ::: {#lh:DC48} | `StochasticSequential.sequential_minimum_sufficiency_pspace_hard`                         |
| `DC47`          |                                                                                              | `DC48`         |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:DC49}  | `StochasticSequential.sequentialMinimalSufficient_iff_relevant`                              | ::: {#lh:DC50} | `StochasticSequential.sequentialRelevantSet_is_minimal`                                   |
| `DC49`          |                                                                                              | `DC50`         |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:DC57}  | `StochasticSequential.fiberDecisionProblem_sufficient`                                       | ::: {#lh:DC62} | `StochasticSequential.countedStochasticSufficiencySearch_spec`                            |
| `DC57`          |                                                                                              | `DC62`         |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:DC63}  | `StochasticSequential.countedStochasticSufficiencySearch_steps`                              | ::: {#lh:DC64} | `StochasticSequential.countedStochasticAnchorSearch_spec`                                 |
| `DC63`          |                                                                                              | `DC64`         |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:DC65}  | `StochasticSequential.countedStochasticAnchorSearch_steps`                                   | ::: {#lh:DC66} | `StochasticSequential.countedSequentialSufficiencySearch_spec`                            |
| `DC65`          |                                                                                              | `DC66`         |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:DC67}  | `StochasticSequential.countedSequentialSufficiencySearch_steps`                              | ::: {#lh:DC68} | `StochasticSequential.countedSequentialAnchorSearch_spec`                                 |
| `DC67`          |                                                                                              | `DC68`         |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:DC69}  | `StochasticSequential.countedSequentialAnchorSearch_steps`                                   | ::: {#lh:DC70} | `DecisionQuotient.static_sufficiency_inP_explicit`                                        |
| `DC69`          |                                                                                              | `DC70`         |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:DC71}  | `DecisionQuotient.static_anchor_inP_explicit`                                                | ::: {#lh:DC72} | `StochasticSequential.stochastic_sufficiency_inP_explicit`                                |
| `DC71`          |                                                                                              | `DC72`         |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:DC73}  | `StochasticSequential.stochastic_anchor_inP_explicit`                                        | ::: {#lh:DC74} | `StochasticSequential.sequential_sufficiency_inP_explicit`                                |
| `DC73`          |                                                                                              | `DC74`         |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:DC75}  | `StochasticSequential.sequential_anchor_inP_explicit`                                        | ::: {#lh:DC76} | `DecisionQuotient.explicit_state_inP_summary`                                             |
| `DC75`          |                                                                                              | `DC76`         |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:DC80}  | `DecisionQuotient.staticSufficiency_counted_search_witness`                                  | ::: {#lh:DC81} | `DecisionQuotient.static_query_search_matrix`                                             |
| `DC80`          |                                                                                              | `DC81`         |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:DC82}  | `DecisionQuotient.finite_search_summary`                                                     | ::: {#lh:DC91} | `StochasticSequential.countedStochasticPreservationSearch_spec`                           |
| `DC82`          |                                                                                              | `DC91`         |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:DC92}  | `StochasticSequential.countedStochasticPreservationSearch_steps`                             | ::: {#lh:DC93} | `StochasticSequential.stochastic_preservation_inP_explicit`                               |
| `DC92`          |                                                                                              | `DC93`         |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:DC94}  | `StochasticSequential.stochastic_preservation_implies_static_sufficiency`                    | ::: {#lh:DC95} | `StochasticSequential.static_sufficiency_implies_stochastic_preservation_of_full_support` |
| `DC94`          |                                                                                              | `DC95`         |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:DC96}  | `StochasticSequential.static_sufficiency_iff_stochastic_preservation_of_full_support`        | ::: {#lh:DC97} | `StochasticSequential.stochasticDecisionEquiv_iff_decisionEquiv_of_preservation`          |
| `DC96`          |                                                                                              | `DC97`         |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:DC98}  | `StochasticSequential.stochasticDecisionEquiv_iff_decisionEquiv_of_full_support`             | ::: {#lh:DC99} | `StochasticSequential.stochasticEquivSetoid_eq_decisionSetoid_of_full_support`            |
| `DC98`          |                                                                                              | `DC99`         |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:DC100} | `StochasticSequential.benchmark_escalation_summary`                                          | ::: {#lh:DN1}  | `DecisionProblem.stateDecisionNoise_iff_same_quotient`                                    |
| `DC100`         |                                                                                              | `DN1`          |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:DN2}   | `DecisionProblem.decisionNoise_iff_not_relevant`                                             | ::: {#lh:DN5}  | `DecisionProblem.decisionNoise_iff_condIndep`                                             |
| `DN2`           |                                                                                              | `DN5`          |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:DP1}   | `DecisionQuotient.DecisionProblem.minimalSufficient_iff_relevant`                            | ::: {#lh:DP2}  | `DecisionQuotient.DecisionProblem.relevantSet_is_minimal`                                 |
| `DP1`           |                                                                                              | `DP2`          |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:DP6}   | `ClaimClosure.DP6`                                                                           | ::: {#lh:DP7}  | `ClaimClosure.DP7`                                                                        |
| `DP6`           |                                                                                              | `DP7`          |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:DQ1}   | `ClaimClosure.DQ1`                                                                           | ::: {#lh:DQ2}  | `ClaimClosure.DQ2`                                                                        |
| `DQ1`           |                                                                                              | `DQ2`          |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:DQ3}   | `ClaimClosure.DQ3`                                                                           | ::: {#lh:DQ6}  | `ClaimClosure.DQ6`                                                                        |
| `DQ3`           |                                                                                              | `DQ6`          |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:DQ7}   | `ClaimClosure.DQ7`                                                                           | ::: {#lh:DQ9}  | `DecisionQuotient.HardnessDistribution.simplicityTax_conservation`                        |
| `DQ7`           |                                                                                              | `DQ9`          |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:DQ10}  | `DecisionQuotient.IntegrityCompetence.alwaysAbstain_integrity`                               | ::: {#lh:DQ11} | `DecisionQuotient.SetCoverInstance.min_sufficient_iff_set_cover`                          |
| `DQ10`          |                                                                                              | `DQ11`         |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:DQ12}  | `DecisionQuotient.StochasticSequential.bounded_horizon_tractable`                            | ::: {#lh:DQ13} | `DecisionQuotient.StochasticSequential.bounded_support_tractable`                         |
| `DQ12`          |                                                                                              | `DQ13`         |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:DQ14}  | `DecisionQuotient.StochasticSequential.fully_observable_tractable`                           | ::: {#lh:DQ15} | `DecisionQuotient.StochasticSequential.product_distribution_tractable`                    |
| `DQ14`          |                                                                                              | `DQ15`         |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:DQ16}  | `DecisionQuotient.StochasticSequential.product_enables_transfer`                             | ::: {#lh:DQ18} | `DecisionQuotient.StochasticSequential.static_simpler_than_stochastic`                    |
| `DQ16`          |                                                                                              | `DQ18`         |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:DQ19}  | `DecisionQuotient.StochasticSequential.stochastic_simpler_than_sequential`                   | ::: {#lh:DQ20} | `DecisionQuotient.all_coordinates_necessary_of_not_tautology`                             |
| `DQ19`          |                                                                                              | `DQ20`         |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:DQ25}  | `DecisionQuotient.counted_min_sufficient_inapproximability_conditional`                      | ::: {#lh:DQ26} | `DecisionQuotient.dichotomy_conditional`                                                  |
| `DQ25`          |                                                                                              | `DQ26`         |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:DQ27}  | `DecisionQuotient.eth_lower_bound_informal`                                                  | ::: {#lh:DQ28} | `DecisionQuotient.exact_certainty_inflation_under_hardness_core`                          |
| `DQ27`          |                                                                                              | `DQ28`         |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:DQ30}  | `DecisionQuotient.hard_family_all_coords_core`                                               | ::: {#lh:DQ31} | `DecisionQuotient.integrity_resource_bound_for_sufficiency`                               |
| `DQ30`          |                                                                                              | `DQ31`         |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:DQ32}  | `DecisionQuotient.min_sufficient_factor_inapprox_of_set_cover_factor_inapprox`               | ::: {#lh:DQ34} | `DecisionQuotient.min_sufficient_set_cover_equiv`                                         |
| `DQ32`          |                                                                                              | `DQ34`         |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:DQ36}  | `DecisionQuotient.minsuff_conp_complete_conditional`                                         | ::: {#lh:DQ38} | `DecisionQuotient.no_exact_claim_admissible_under_hardness_core`                          |
| `DQ36`          |                                                                                              | `DQ38`         |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:DQ39}  | `DecisionQuotient.singleton_gate_sufficient_of_tautology`                                    | ::: {#lh:DQ40} | `DecisionQuotient.sufficiency_conp_complete_conditional`                                  |
| `DQ39`          |                                                                                              | `DQ40`         |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:DQ42}  | `DecisionQuotient.sufficient_means_factorizable`                                             | ::: {#lh:DQ43} | `DecisionQuotient.tautology_decidable_from_factor_approx`                                 |
| `DQ42`          |                                                                                              | `DQ43`         |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:DQ44}  | `DecisionQuotient.tractable_subcases_conditional`                                            | ::: {#lh:DQ45} | `DecisionQuotient.tractable_tree_core`                                                    |
| `DQ44`          |                                                                                              | `DQ45`         |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:DQ51}  | `DecisionQuotient.bounded_actions_tractable`                                                 | ::: {#lh:DQ56} | `DecisionQuotient.orbitType_count_bound`                                                  |
| `DQ51`          |                                                                                              | `DQ56`         |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:DQ57}  | `DecisionQuotient.separable_utility_tractable`                                               | ::: {#lh:DQ58} | `DecisionQuotient.stochastic_objective_bridge_can_fail_on_sufficiency`                    |
| `DQ57`          |                                                                                              | `DQ58`         |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:DQ62}  | `DecisionQuotient.transition_coupled_bridge_can_fail_on_sufficiency`                         | ::: {#lh:DQ63} | `DecisionQuotient.tree_structure_tractable`                                               |
| `DQ62`          |                                                                                              | `DQ63`         |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:DQ72}  | `DecisionQuotient.bounded_actions_complexity`                                                | ::: {#lh:DQ74} | `DecisionQuotient.sufficiency_poly_separable`                                             |
| `DQ72`          |                                                                                              | `DQ74`         |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:DQ75}  | `DecisionQuotient.tensor_contraction_tractable`                                              | ::: {#lh:DQ76} | `DecisionQuotient.low_rank_utility_admits_factored_computation`                           |
| `DQ75`          |                                                                                              | `DQ76`         |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:DQ77}  | `DecisionQuotient.low_rank_tractability`                                                     | ::: {#lh:DQ78} | `DecisionQuotient.sufficiency_poly_tree_structured`                                       |
| `DQ77`          |                                                                                              | `DQ78`         |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:DQ79}  | `DecisionQuotient.csp_treewidth_tractable`                                                   | ::: {#lh:DQ80} | `DecisionQuotient.sufficiency_reduces_to_interaction_csp`                                 |
| `DQ79`          |                                                                                              | `DQ80`         |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:DQ81}  | `DecisionQuotient.bounded_treewidth_tractability`                                            | ::: {#lh:DQ82} | `DecisionQuotient.orbitType_eq_iff`                                                       |
| `DQ81`          |                                                                                              | `DQ82`         |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:DQ83}  | `DecisionQuotient.symmetric_optimalActions_orbit_invariant`                                  | ::: {#lh:DQ84} | `DecisionQuotient.sufficiency_reduces_to_cross_orbit_check`                               |
| `DQ83`          |                                                                                              | `DQ84`         |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:DQ85}  | `DecisionQuotient.symmetric_sufficiency_complexity_bound`                                    | ::: {#lh:EH1}  | `StochasticSequential.existential_anchor_source_fits_np_over_ppstyle_honest`              |
| `DQ85`          |                                                                                              | `EH1`          |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:EH2}   | `StochasticSequential.existential_anchor_hard_honest`                                        | ::: {#lh:EH3}  | `StochasticSequential.existential_anchor_query_family_hard_honest`                        |
| `EH2`           |                                                                                              | `EH3`          |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:EH4}   | `StochasticSequential.existential_anchor_np_over_ppstyle_hard_honest`                        | ::: {#lh:EH5}  | `StochasticSequential.existential_anchor_query_family_np_over_ppstyle_hard_honest`        |
| `EH4`           |                                                                                              | `EH5`          |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:EH6}   | `StochasticSequential.existential_anchor_query_family_np_over_ppstyle_complete_honest`       | ::: {#lh:EH7}  | `StochasticSequential.existential_decisiveness_complement_np_over_ppstyle_hard_honest`    |
| `EH6`           |                                                                                              | `EH7`          |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:EH8}   | `StochasticSequential.existential_decisiveness_complement_np_over_ppstyle_complete_honest`   | ::: {#lh:EH9}  | `StochasticSequential.existential_decisiveness_query_family_np_over_ppstyle_hard_honest`  |
| `EH8`           |                                                                                              | `EH9`          |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:EH10}  | `StochasticSequential.existential_decisiveness_query_family_np_over_ppstyle_complete_honest` | ::: {#lh:EX1}  | `Examples.pomdp_reduction_to_preservation`                                                |
| `EH10`          |                                                                                              | `EX1`          |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:EX2}   | `Examples.hyperparam_reduction_to_static`                                                    | ::: {#lh:EX3}  | `Examples.toy_full_opt_s1_contains_a`                                                     |
| `EX2`           |                                                                                              | `EX3`          |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:HD4}   | `DecisionQuotient.HardnessDistribution.complete_model_dominates_after_threshold`             | ::: {#lh:HD5}  | `DecisionQuotient.HardnessDistribution.gap_conservation_card`                             |
| `HD4`           |                                                                                              | `HD5`          |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:HD14}  | `DecisionQuotient.HardnessDistribution.linear_lt_exponential_plus_constant_eventually`       | ::: {#lh:HD23} | `DecisionQuotient.HardnessDistribution.hardness_is_irreducible_required_work`             |
| `HD14`          |                                                                                              | `HD23`         |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:HD26}  | `DecisionQuotient.HardnessDistribution.totalExternalWork_eq_n_mul_gapCard`                   | ::: {#lh:IC30} | `DecisionQuotient.IntegrityCompetence.exactCertaintyInflation_iff_no_exact_competence`    |
| `HD26`          |                                                                                              | `IC30`         |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:IC32}  | `DecisionQuotient.IntegrityCompetence.integrity_forces_abstention`                           | ::: {#lh:IC33} | `DecisionQuotient.IntegrityCompetence.integrity_not_competent_of_nonempty_scope`          |
| `IC32`          |                                                                                              | `IC33`         |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:IC34}  | `DecisionQuotient.IntegrityCompetence.integrity_resource_bound`                              | ::: {#lh:OU1}  | `StochasticSequential.stochastic_anchor_query_fits_np_over_ppstyle`                       |
| `IC34`          |                                                                                              | `OU1`          |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:OU2}   | `StochasticSequential.stochastic_minimum_query_fits_np_over_ppstyle`                         | ::: {#lh:OU6}  | `StochasticSequential.stochastic_anchor_inP_explicit_via_witness_schema`                  |
| `OU2`           |                                                                                              | `OU6`          |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:OU7}   | `StochasticSequential.stochastic_minimum_inP_explicit_via_witness_schema`                    | ::: {#lh:OU8}  | `StochasticSequential.fitsCoNPOverPPStyle_of_no_witness`                                  |
| `OU7`           |                                                                                              | `OU8`          |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:OU9}   | `StochasticSequential.stochastic_decisiveness_query_fits_conp_over_ppstyle`                  | ::: {#lh:OU10} | `StochasticSequential.stochastic_decisiveness_complement_fits_np_over_ppstyle`            |
| `OU9`           |                                                                                              | `OU10`         |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:OU11}  | `StochasticSequential.stochastic_decisiveness_scoped_oracle_bounds`                          | ::: {#lh:OU12} | `StochasticSequential.stochastic_preservation_explicit_summary`                           |
| `OU11`          |                                                                                              | `OU12`         |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:PA3}   | `Physics.AnchorChecks.stochastic_anchor_check_iff_exists_anchor_singleton`                   | ::: {#lh:QT1}  | `DecisionProblem.quotient_is_coarsest`                                                    |
| `PA3`           |                                                                                              | `QT1`          |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:QT7}   | `DecisionProblem.quotient_has_unique_factorization`                                          | ::: {#lh:WD1}  | `DecisionQuotient.checking_witnessing_duality_budget`                                     |
| `QT7`           |                                                                                              | `WD1`          |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
| ::: {#lh:WD2}   | `DecisionQuotient.no_sound_checker_below_witness_budget`                                     | ::: {#lh:WD3}  | `DecisionQuotient.checking_time_ge_witness_budget`                                        |
| `WD2`           |                                                                                              | `WD3`          |                                                                                           |
| :::             |                                                                                              | :::            |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+
|                 |                                                                                              |                |                                                                                           |
+-----------------+----------------------------------------------------------------------------------------------+----------------+-------------------------------------------------------------------------------------------+




---

## Machine-Checked Proofs

All theorems are formalized in Lean 4:
- Location: `docs/papers/paper4_decision_quotient/proofs/`
- Lines: 22068
- Theorems: 1015
- `sorry` placeholders: 0
