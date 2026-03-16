# Paper: Decision Quotient Structure and Convergence: Optimizer Quotients, Structural Rank, and Bayesian Optimality

**Status**: IEEE Transactions on Information Theory-ready | **Lean**: 66580 lines, 2945 theorems

---

## Abstract

This paper studies convergence in decision-relevant information. For a decision problem $\mathcal{D}=(A,S,U)$ with $S=X_1 \times \cdots \times X_n$, a coordinate set is sufficient when agreement on those coordinates determines the optimal-action set. The associated optimizer quotient is the canonical decision-preserving abstraction.

The main result is that several standard mathematical viewpoints can be compared against the same structural invariant once that quotient is fixed. The invariant is structural rank, the cardinality of the relevant-coordinate support. We show that the optimizer quotient has the coimage/image universal property in **Set**, that quotient entropy is bounded by structural rank, that Fisher-style support counting recovers the same support size, and that zero-distortion decision-preserving compression is constrained by the same support-size core.

A parallel Bayesian thread starts from finite counting and the inequality $\log x \le x - 1$ and derives probability normalization, Bayes' rule, cross-entropy decomposition, Bayes optimality under log loss, and the normalized decision-quotient score.

The Lean 4 development records the quotient, Bayesian, and information-theoretic lemmas used in the paper.


# Introduction {#sec:introduction}

The decision-relevance problem asks which parts of a state description matter for optimal action. This raises two questions: how hard relevance is to certify, and what structure emerges once the relevant-coordinate support has been identified. This paper addresses the second.

The unifying object is the optimizer quotient. States are identified when they induce the same optimal-action set, and the resulting quotient is the coarsest abstraction that preserves the decision boundary. In **Set**, this is the coimage/image factorization of the optimizer map. The question is then which quantitative summaries of that quotient recover the same structural content.

Several mathematical frameworks can be compared against the same invariant. The support size of the relevant coordinates, written $\mathrm{srank}$, appears directly in Fisher-style support counting, bounds quotient entropy, and constrains zero-distortion summaries through the optimizer quotient. The Bayesian chain is parallel rather than another recovery theorem: it shows that the same finite counting setup also supports the standard Bayes-optimality identities. Convergence therefore means that these frameworks are organized around the same notion of decision-relevant structure, not that they all yield the same formula.

The novelty is not in any individual ingredient. Entropy, Bayes, quotient objects, and Fisher information are classical. The contribution is the theorem-level synthesis: once the optimizer quotient is fixed as the canonical decision-preserving abstraction, these frameworks can be compared in a single formal setting and related to the same structural notion of decision-relevant dimension.

The argument is built from shared definitions, quotient structure, and information-theoretic inequalities.

## Contributions

1.  **Counting-gap preliminaries.** We isolate a discrete counting bound and a measure-before-probability theorem that place normalization and Bayesian identities on explicit finite setup.

2.  **Optimizer quotient as canonical structure.** We restate the shared setup needed for independent readability, including an explicit structural-rank support proposition, and identify the optimizer quotient as the canonical decision-preserving abstraction.

3.  **Structural-rank comparison across frameworks.** We show that Fisher-style support counting recovers $\mathrm{srank}$ exactly, quotient entropy is bounded by $\mathrm{srank}$, lossless zero-distortion summaries are constrained by the same optimizer-quotient structure, and the transport bridge tracks the same quotient branching regimes.

4.  **Bayesian optimality from minimal assumptions.** We derive probability normalization, Bayes' rule, cross-entropy decomposition, and Bayes optimality under log loss from finite counting and the inequality $\log x \le x - 1$.

5.  **Machine-checked support.** The Lean development records the quotient, Bayesian, and information-theoretic lemmas used in the paper.

## Paper Structure

Section [\[sec:counting-preliminaries\]](#sec:counting-preliminaries){reference-type="ref" reference="sec:counting-preliminaries"} states the counting and measure preliminaries. Section [\[sec:formal-setup\]](#sec:formal-setup){reference-type="ref" reference="sec:formal-setup"} restates the shared quotient setup needed for independence. Section [\[sec:convergence-frameworks\]](#sec:convergence-frameworks){reference-type="ref" reference="sec:convergence-frameworks"} develops the structural-rank convergence story. Section [\[sec:bayesian-optimality\]](#sec:bayesian-optimality){reference-type="ref" reference="sec:bayesian-optimality"} gives the Bayesian optimality chain. Section [\[sec:related\]](#sec:related){reference-type="ref" reference="sec:related"} situates the paper relative to information theory, sufficient statistics, and formalized mathematics. Appendix [\[app:lean\]](#app:lean){reference-type="ref" reference="app:lean"} summarizes the corresponding Lean support.

## Proof Strategy at a Glance

Each theorem in the body is proved by a short explicit chain:

1.  **Counting layer (Section [\[sec:counting-preliminaries\]](#sec:counting-preliminaries){reference-type="ref" reference="sec:counting-preliminaries"}):** finite counting bounds and normalization identities.

2.  **Quotient layer (Section [\[sec:formal-setup\]](#sec:formal-setup){reference-type="ref" reference="sec:formal-setup"}):** sufficiency/relevance definitions and universal property of the optimizer quotient.

3.  **Bridge layer (Section [\[sec:convergence-frameworks\]](#sec:convergence-frameworks){reference-type="ref" reference="sec:convergence-frameworks"}):** entropy bounds, support counting, zero-distortion summaries, and transport regimes all tied back to the same optimizer-quotient core.

4.  **Statistical consequences (Section [\[sec:bayesian-optimality\]](#sec:bayesian-optimality){reference-type="ref" reference="sec:bayesian-optimality"}):** Bayes optimality on the normalized finite-probability side.

This removes ambiguity about where each claim comes from: no hidden assumptions, no skipped quantifier steps, and no dependence on informal analogies.


# Counting Preliminaries {#sec:counting-preliminaries}

Before probabilities, entropies, and Bayesian updates, we fix the discrete counting layer used throughout the paper.

## The Counting Gap Theorem

::: theorem
[]{#thm:counting-gap label="thm:counting-gap"} Let $\varepsilon, C \in \mathbb{N}$ with $\varepsilon>0$. If each information-acquisition event consumes at least $\varepsilon$ cost units and total capacity is $C$, then the number of admissible events satisfies $$N \le \left\lfloor\frac{C}{\varepsilon}\right\rfloor.$$ In the unit-cost normalization $\varepsilon=1$, this is $N\le C$.
:::

::: proof
*Proof.* Let $N$ be the number of admitted events. By hypothesis, each event consumes at least $\varepsilon$, so total consumption is at least $N\varepsilon$. Feasibility under budget $C$ therefore requires $$N\varepsilon \le C.$$ Because $\varepsilon>0$, division gives $N \le C/\varepsilon$. Since $N\in\mathbb{N}$, this is equivalent to $$N \le \left\lfloor C/\varepsilon \right\rfloor.$$ Setting $\varepsilon=1$ yields $N\le C$. ◻
:::

## Measure Before Probability

::: theorem
[]{#thm:measure-prerequisite label="thm:measure-prerequisite"} Quantitative claims require a specified measure, while probabilistic claims require a normalized measure. Counting measure on a finite set is therefore a precursor to probability, not probability itself, until normalization is performed.
:::

::: proof
*Proof.* Any quantitative aggregate (e.g., expectation, average cost, entropy with respect to weights) is defined relative to a measure. On a finite space, counting gives a canonical measure $\mu(E)=|E|$, but this has total mass $|\Omega|$, not $1$ in general. A probability model is obtained only after normalization: $$P(E)=\frac{\mu(E)}{\mu(\Omega)}.$$ Hence the logical order is measure first, probability second. ◻
:::

## Finite Probability from Counting

::: theorem
[]{#thm:bayes-from-counting label="thm:bayes-from-counting"} Let $\Omega$ be a finite nonempty set and define $P(E)=|E|/|\Omega|$. Then:

1.  $P$ satisfies the finite Kolmogorov axioms;

2.  conditional probability and Bayes' rule follow algebraically wherever denominators are nonzero;

3.  mutual information is well-defined after normalization;

4.  the normalized decision-quotient score $\DQ=I(H;E)/H(H)$ lies in $[0,1]$ whenever $H(H)>0$.
:::

::: proof
*Proof.* **(1) Kolmogorov axioms.** Nonnegativity is immediate from cardinality. Normalization holds because $$P(\Omega)=\frac{|\Omega|}{|\Omega|}=1.$$ For disjoint $E,F\subseteq \Omega$, $$P(E\cup F)=\frac{|E\cup F|}{|\Omega|}=\frac{|E|+|F|}{|\Omega|}=P(E)+P(F).$$

**(2) Conditional probability and Bayes.** For $P(E)>0$, define $P(H\mid E)=P(H\cap E)/P(E)$. Then $$P(H\cap E)=P(H\mid E)P(E)=P(E\mid H)P(H),$$ and rearranging gives Bayes' rule wherever denominators are nonzero.

**(3) Mutual information.** After normalization, $(H,E)$ is an ordinary finite pair of random variables, so $I(H;E)$ is well-defined by its standard finite-sum formula.

**(4) DQ bound.** For $H(H)>0$, standard finite information inequalities give $$0 \le I(H;E) \le H(H),$$ thus $$0 \le \DQ = \frac{I(H;E)}{H(H)} \le 1.$$ ◻
:::


# Formal Setup {#sec:formal-setup}

This section restates the definitions needed for the paper to stand on its own. The emphasis is on sufficiency, relevance, and the quotient structure of the optimizer map.

## Decision Problems and Sufficiency

::: definition
[]{#def:decision-problem label="def:decision-problem"} A decision problem with coordinate structure is a tuple $\mathcal{D}=(A,X_1,\ldots,X_n,U)$ where $A$ is a finite action set, $S=X_1 \times \cdots \times X_n$ is the state space, and $U : A \times S \to \mathbb{Q}$ is the utility function.
:::

::: definition
[]{#def:projection label="def:projection"} For state $s=(s_1,\ldots,s_n)\in S$ and coordinate set $I \subseteq \{1,\ldots,n\}$, write $s_I$ for the projection of $s$ onto the coordinates in $I$.
:::

::: definition
[]{#def:optimizer label="def:optimizer"} The optimizer map is $$\operatorname{Opt}(s) := \arg\max_{a \in A} U(a,s).$$
:::

::: definition
[]{#def:sufficient label="def:sufficient"} Coordinate set $I$ is sufficient if $$\forall s,s' \in S:\quad s_I=s'_I \implies \operatorname{Opt}(s)=\operatorname{Opt}(s').$$
:::

::: definition
[]{#def:minimal-sufficient label="def:minimal-sufficient"} A sufficient set is minimal if no proper subset is sufficient.
:::

::: definition
[]{#def:relevant label="def:relevant"} Coordinate $i$ is relevant if there exist states that differ only at coordinate $i$ and induce different optimal-action sets: $$\exists s,s' \in S:\;
\Big(\forall j \neq i,\; s_j=s'_j\Big)
\ \wedge\
\operatorname{Opt}(s)\neq\operatorname{Opt}(s').$$
:::

::: proposition
[]{#prop:minimal-relevant-equiv label="prop:minimal-relevant-equiv"} Every minimal sufficient set is exactly the relevant-coordinate set.
:::

::: proof
*Proof.* **Minimal sufficient sets contain only relevant coordinates.** Let $I$ be minimal sufficient and let $i\in I$. Suppose toward contradiction that $i$ is not relevant. Then whenever two states agree on all coordinates except possibly $i$, they still have the same optimizer value. In particular, once the coordinates in $I\setminus\{i\}$ are fixed, changing coordinate $i$ cannot change $\operatorname{Opt}$.

Now take any two states $s,s'$ with $s_{I\setminus\{i\}}=s'_{I\setminus\{i\}}$. Keeping the other coordinates fixed, we can vary coordinate $i$ without affecting $\operatorname{Opt}$, so the value of $\operatorname{Opt}$ is already determined by the coordinates in $I\setminus\{i\}$. Thus $I\setminus\{i\}$ is still sufficient, contradicting minimality of $I$. Therefore every coordinate in a minimal sufficient set is relevant.

**Every relevant coordinate belongs to every sufficient set.** Let $i$ be relevant. By definition, there exist states $s,s'$ differing only at coordinate $i$ such that $\operatorname{Opt}(s)\neq\operatorname{Opt}(s')$. If a coordinate set $I$ omitted $i$, then these two states would agree on all coordinates of $I$, since they differ nowhere else. Sufficiency of $I$ would then force $\operatorname{Opt}(s)=\operatorname{Opt}(s')$, contradiction. So every sufficient set must contain $i$.

Combining the two parts, every minimal sufficient set is contained in the relevant-coordinate set, and the relevant-coordinate set is contained in every sufficient set, hence in every minimal sufficient set. Therefore every minimal sufficient set is exactly the relevant-coordinate set. ◻
:::

::: definition
[]{#def:srank label="def:srank"} The structural rank of $\mathcal{D}$ is the cardinality of the relevant-coordinate support: $$\mathrm{srank}(\mathcal{D}) :=
\left|\{i \in \{1,\ldots,n\} : i \text{ is relevant}\}\right|.$$
:::

::: proposition
[]{#prop:srank-support label="prop:srank-support"} $$\mathrm{srank}(\mathcal{D})
=
\left|\{i \in \{1,\ldots,n\} : i \text{ is relevant}\}\right|.$$ Hence $\mathrm{srank}(\mathcal{D}) \le n$, and $\mathrm{srank}(\mathcal{D})=0$ exactly when no coordinate is relevant.
:::

::: proof
*Proof.* The equality is the definition of structural rank. The upper bound is immediate because the relevance support is a subset of $\{1,\dots,n\}$. The zero case is equivalent to the relevance support being empty. ◻
:::

This quantity is the central invariant of the paper. The remaining sections ask which different mathematical summaries of a decision problem are in fact measuring this same support-size core.

## Optimizer Quotient and Coimage Structure

::: definition
[]{#def:decision-equiv label="def:decision-equiv"} States $s,s' \in S$ are decision-equivalent if $\operatorname{Opt}(s)=\operatorname{Opt}(s')$.
:::

::: definition
[]{#def:decision-quotient label="def:decision-quotient"} For coordinate set $I$ and state $s$, define $$\DQ_I(s) =
\frac{|\{a \in A : a \in \operatorname{Opt}(s') \text{ for some } s' \text{ with } s'_I=s_I\}|}{|A|}.$$
:::

::: proposition
[]{#prop:sufficiency-char label="prop:sufficiency-char"} Coordinate set $I$ is sufficient if and only if $\DQ_I(s)=|\operatorname{Opt}(s)|/|A|$ for every state $s$.
:::

::: proof
*Proof.* **If $I$ is sufficient.** Fix $s$. For any $s'$ with $s'_I=s_I$, sufficiency gives $\operatorname{Opt}(s')=\operatorname{Opt}(s)$. Hence $$\{a : a\in \operatorname{Opt}(s') \text{ for some } s' \text{ with } s'_I=s_I\}=\operatorname{Opt}(s),$$ so $\DQ_I(s)=|\operatorname{Opt}(s)|/|A|$.

**If the displayed equality holds for all $s$.** Suppose $I$ were not sufficient. Then there exist $s,s'$ with $s_I=s'_I$ but $\operatorname{Opt}(s)\neq\operatorname{Opt}(s')$. Pick $a\in \operatorname{Opt}(s')\setminus \operatorname{Opt}(s)$ (or symmetrically). This action belongs to the union in the numerator of $\DQ_I(s)$ but not to $\operatorname{Opt}(s)$, so the numerator is strictly larger than $|\operatorname{Opt}(s)|$, contradiction. ◻
:::

::: proposition
[]{#prop:optimizer-coimage label="prop:optimizer-coimage"} The quotient of $S$ by decision equivalence is canonically equivalent to $\operatorname{range}(\operatorname{Opt})$. Equivalently, the optimizer quotient is the coimage of the optimizer map, identified with its image in **Set**.
:::

::: proof
*Proof.* Define $$\Phi: S/{\sim}\ \to\ \operatorname{range}(\operatorname{Opt}),\qquad \Phi([s])=\operatorname{Opt}(s).$$ This is well-defined because $s\sim s'$ implies $\operatorname{Opt}(s)=\operatorname{Opt}(s')$. It is surjective by definition of range. It is injective because $\Phi([s])=\Phi([s'])$ means $\operatorname{Opt}(s)=\operatorname{Opt}(s')$, hence $[s]=[s']$. Therefore $\Phi$ is a bijection. ◻
:::

::: theorem
[]{#thm:quotient-universal label="thm:quotient-universal"} Let $Q=S/{\sim}$ be the optimizer quotient, where $s \sim s'$ if and only if $\operatorname{Opt}(s)=\operatorname{Opt}(s')$, and let $\pi:S\to Q$ be the quotient map. For any surjective abstraction $\phi:S\to T$ that preserves the optimizer map in the sense that $$\phi(s)=\phi(s') \implies \operatorname{Opt}(s)=\operatorname{Opt}(s'),$$ there exists a unique map $\psi:T\to Q$ such that $\pi=\psi\circ\phi$.
:::

::: proof
*Proof.* Because $\phi$ preserves optimizer values on fibers, we have $$\phi(s)=\phi(s') \implies s\sim s'.$$ So each $\phi$-fiber is contained in a $\sim$-class.

For each $t\in T$, choose any $s$ with $\phi(s)=t$ (possible since $\phi$ is surjective), and define $$\psi(t):=\pi(s)\in Q.$$ This is well-defined: if $s_1,s_2$ are two preimages of $t$, then $\phi(s_1)=\phi(s_2)$, hence $s_1\sim s_2$, hence $\pi(s_1)=\pi(s_2)$.

Now for any $s\in S$, $$(\psi\circ\phi)(s)=\psi(\phi(s))=\pi(s),$$ so $\pi=\psi\circ\phi$.

For uniqueness, let $\psi'$ also satisfy $\pi=\psi'\circ\phi$. For any $t\in T$, choose $s$ with $\phi(s)=t$. Then $$\psi'(t)=\psi'(\phi(s))=\pi(s)=\psi(\phi(s))=\psi(t),$$ so $\psi'=\psi$. ◻
:::

::: proposition
[]{#prop:optimizer-entropy-image label="prop:optimizer-entropy-image"} If $\mathrm{numOptClasses}(\mathcal{D})$ denotes the number of distinct values attained by $\operatorname{Opt}$, then $$H(\mathcal{D}) = \log_2(\mathrm{numOptClasses}(\mathcal{D}))
=
\log_2(|\operatorname{range}(\operatorname{Opt})|).$$
:::

::: proof
*Proof.* By Proposition [\[prop:optimizer-coimage\]](#prop:optimizer-coimage){reference-type="ref" reference="prop:optimizer-coimage"}, quotient classes are in bijection with $\operatorname{range}(\operatorname{Opt})$, so their cardinalities agree: $$\mathrm{numOptClasses}(\mathcal{D})=|\operatorname{range}(\operatorname{Opt})|.$$ By definition, $H(\mathcal{D})$ is the base-$2$ entropy of the counting-normalized quotient distribution; on a finite uniform support of size $m$, entropy is $\log_2 m$. Substituting $m=\mathrm{numOptClasses}(\mathcal{D})$ yields the claim. ◻
:::

::: remark
[]{#rem:question-vs-problem label="rem:question-vs-problem"} The predicate "is $I$ sufficient for $\mathcal{D}$?" is encoding-independent. Complexity classes apply only after choosing how $\mathcal{D}$ and $I$ are represented. This paper uses the same shared definitions, but studies their quotient and convergence consequences rather than their class-theoretic complexity.
:::

The point of Theorem [\[thm:quotient-universal\]](#thm:quotient-universal){reference-type="ref" reference="thm:quotient-universal"} is organizational rather than terminological. In **Set**, the optimizer quotient is a familiar coimage/image construction. Its role here is to provide a common comparison object for entropy, support counting, Bayesian updating, and lossless summaries.


# Convergence Frameworks {#sec:convergence-frameworks}

This section shows how several mathematical frameworks relate to the same structural invariant once the optimizer quotient has been fixed. That invariant is structural rank. The bridge theorems below do not claim that all frameworks recover $\mathrm{srank}$ in the same formal sense: some give exact recovery, some give bounds, and some identify the same quotient branching regimes under exact decision preservation.

## Entropy and the Quotient

The finite counting-to-probability bridge is established in Section [\[sec:counting-preliminaries\]](#sec:counting-preliminaries){reference-type="ref" reference="sec:counting-preliminaries"} (Theorem [\[thm:bayes-from-counting\]](#thm:bayes-from-counting){reference-type="ref" reference="thm:bayes-from-counting"}). Here we use that normalized layer to relate quotient entropy to structural rank.

::: theorem
[]{#thm:entropy-rank label="thm:entropy-rank"} For Boolean coordinate spaces, $$H(\mathcal{D}) \le \mathrm{srank}(\mathcal{D}),$$ where $H(\mathcal{D})$ is the base-$2$ entropy of the optimizer quotient.
:::

::: proof
*Proof.* Let $r=\mathrm{srank}(\mathcal{D})$. By definition, exactly $r$ coordinates are relevant, and this relevant set is sufficient (Section [\[sec:formal-setup\]](#sec:formal-setup){reference-type="ref" reference="sec:formal-setup"}). Therefore $\operatorname{Opt}$ factors through the projection onto those $r$ coordinates: $$S \xrightarrow{\pi_R} \{0,1\}^r \xrightarrow{\bar{\operatorname{Opt}}} \mathcal{P}(A).$$ Every optimizer class is therefore determined by one of the $2^r$ possible assignments to the relevant coordinates. Hence the optimizer quotient has at most $|\{0,1\}^r|=2^r$ classes. Under counting-normalized quotient entropy, $$H(\mathcal{D}) \le \log_2(2^r)=r=\mathrm{srank}(\mathcal{D}).$$ ◻
:::

## Structural Rank Recovery

::: theorem
[]{#thm:fisher-rank-srank label="thm:fisher-rank-srank"} If one assigns score $1$ to relevant coordinates and $0$ to irrelevant ones, the resulting diagonal Fisher-style support matrix has rank $\mathrm{srank}(\mathcal{D})$.
:::

::: proof
*Proof.* In the stated encoding, the diagonal entry for coordinate $i$ is nonzero exactly when $i$ is relevant. For any diagonal matrix, rank equals the number of nonzero diagonal entries. Therefore the rank is exactly $$\big|\{i:\ i \text{ relevant}\}\big|=\mathrm{srank}(\mathcal{D}).$$ ◻
:::

::: theorem
[]{#thm:rate-distortion-bridge label="thm:rate-distortion-bridge"} At zero distortion, any lossless summary that preserves optimal actions must assign distinct codewords to distinct optimizer-quotient classes. Thus the admissible summary size is constrained by the same quotient structure determined by the relevant-coordinate support.
:::

::: proof
*Proof.* At zero distortion, any two states mapped to the same summary must induce the same optimal-action set; otherwise optimal actions are not preserved exactly. So admissible summaries are precisely coarsenings that refine decision equivalence, i.e., they cannot identify states across distinct optimizer-quotient classes.

Therefore any lossless summary alphabet must contain at least one distinct codeword for each optimizer-quotient class. If the alphabet were smaller than the number of decision classes, two different classes would share a codeword, violating zero distortion. Conversely, assigning distinct codewords to the classes yields an exact lossless summary.

By Section [\[sec:formal-setup\]](#sec:formal-setup){reference-type="ref" reference="sec:formal-setup"}, the quotient class structure is determined by the relevant-coordinate support, hence by $\mathrm{srank}(\mathcal{D})$. So the zero-distortion summary size is constrained by the same support-size core. ◻
:::

::: theorem
[]{#thm:wasserstein-bridge label="thm:wasserstein-bridge"} When transport is computed on the optimizer-quotient classes, transport cost is zero in the singleton-class regime and strictly positive once multiple optimizer classes must be distinguished. Thus transport complexity tracks the same quotient branching regimes determined by decision relevance.
:::

::: proof
*Proof.* If the quotient has one class, the identity coupling is diagonal and incurs zero transport cost. If the quotient has at least two classes with nontrivial mass split, any coupling that matches the distinct targets carries off-diagonal mass and therefore positive transport cost. By Section [\[sec:formal-setup\]](#sec:formal-setup){reference-type="ref" reference="sec:formal-setup"}, the same class multiplicity is controlled by relevant-coordinate support, so the transport regimes are determined by the same structural core. ◻
:::

::: theorem
[]{#thm:nontriviality-counting label="thm:nontriviality-counting"} If a decision problem carries nonzero decision-relevant information, then its optimizer quotient is nontrivial. Equivalently, a singleton quotient cannot support positive decision-relevant entropy.
:::

::: proof
*Proof.* If the optimizer quotient is singleton, then all states are decision-equivalent and $\operatorname{Opt}$ is constant on $S$. Hence there is only one attained optimizer value and the quotient entropy is $$H(\mathcal{D})=\log_2(1)=0.$$ So positive decision-relevant information is impossible in the singleton case. Taking contrapositive yields the theorem. ◻
:::

## Framework Convergence

::: theorem
[]{#thm:six-framework-recovery label="thm:six-framework-recovery"} Within the finite setting of this paper, the following objects are linked by the optimizer quotient and the relevant-coordinate support:

1.  relevant-coordinate support,

2.  structural rank,

3.  optimizer-quotient image size,

4.  quotient entropy,

5.  Fisher-style support count,

6.  zero-distortion lossless decision summary size.
:::

::: proof
*Proof.* The chain is explicit:

1.  Section [\[sec:formal-setup\]](#sec:formal-setup){reference-type="ref" reference="sec:formal-setup"} identifies the support core: relevant coordinates and structural rank (Definition [\[def:srank\]](#def:srank){reference-type="ref" reference="def:srank"}; Proposition [\[prop:minimal-relevant-equiv\]](#prop:minimal-relevant-equiv){reference-type="ref" reference="prop:minimal-relevant-equiv"}).

2.  Proposition [\[prop:optimizer-coimage\]](#prop:optimizer-coimage){reference-type="ref" reference="prop:optimizer-coimage"} and Theorem [\[thm:quotient-universal\]](#thm:quotient-universal){reference-type="ref" reference="thm:quotient-universal"} identify optimizer-quotient classes as the canonical decision abstraction and relate them to the image of $\operatorname{Opt}$.

3.  Theorem [\[thm:entropy-rank\]](#thm:entropy-rank){reference-type="ref" reference="thm:entropy-rank"} bounds quotient entropy in terms of this support size.

4.  Theorem [\[thm:fisher-rank-srank\]](#thm:fisher-rank-srank){reference-type="ref" reference="thm:fisher-rank-srank"} shows Fisher-style support counting recovers exactly the same cardinality.

5.  Theorem [\[thm:rate-distortion-bridge\]](#thm:rate-distortion-bridge){reference-type="ref" reference="thm:rate-distortion-bridge"} shows zero-distortion summary size is constrained by the same quotient class structure.

6.  Theorem [\[thm:nontriviality-counting\]](#thm:nontriviality-counting){reference-type="ref" reference="thm:nontriviality-counting"} excludes degenerate singleton quotients when decision-relevant information is positive.

Thus the six listed objects are organized around the same support/quotient core, with exact recovery in some cases and structural bounds in others. ◻
:::

The point is not that these frameworks become identical in every technical sense. Rather, once the optimizer quotient is fixed, they can all be compared against the same support/quotient core for decision relevance. The paper's backbone is therefore simple: shared definitions, the quotient universal property, and a family of bridge theorems centered on $\mathrm{srank}$.


# Bayesian Optimality {#sec:bayesian-optimality}

This section isolates the Bayesian chain in a stripped-down form. The key input is the inequality $\log x \le x - 1$, which yields Gibbs' inequality and therefore the standard optimality of Bayes under log loss. The substantive point is not merely that Bayes is optimal, but that the Bayes chain fits the same first-principles pattern as the quotient and entropy bridges: finite counting first, normalization second, optimization consequences third.

## Cross-Entropy Decomposition

::: theorem
[]{#thm:bayes-optimal label="thm:bayes-optimal"} For distributions $p,q$ on a finite hypothesis space, $$\mathrm{CE}(p,q)=H(p)+D_{\mathrm{KL}}(p\|q),$$ and therefore $q=p$ uniquely minimizes expected log loss.
:::

::: proof
*Proof.* Write $$\mathrm{CE}(p,q):=-\sum_h p(h)\log q(h).$$ Add and subtract $-\sum_h p(h)\log p(h)$: $$\mathrm{CE}(p,q)
=-\sum_h p(h)\log p(h)+\sum_h p(h)\log\frac{p(h)}{q(h)}
=H(p)+D_{\mathrm{KL}}(p\|q).$$ Gibbs' inequality gives $D_{\mathrm{KL}}(p\|q)\ge 0$, with equality iff $p=q$ (on the support of $p$). Hence $\mathrm{CE}(p,q)$ is minimized uniquely at $q=p$. ◻
:::

## Decision-Quotient Ratio

::: proposition
[]{#prop:dq-range label="prop:dq-range"} The normalized decision-quotient score $$\DQ = \frac{I(H;E)}{H(H)}$$ lies in $[0,1]$ whenever $H(H)>0$.
:::

::: proof
*Proof.* Nonnegativity is standard: $$I(H;E)\ge 0.$$ Upper bound follows from $$I(H;E)=H(H)-H(H\mid E)\le H(H),$$ since conditional entropy is nonnegative. Dividing by $H(H)>0$ gives $$0\le \frac{I(H;E)}{H(H)}\le 1.$$ ◻
:::

::: theorem
[]{#thm:bayesian-optimality-chain label="thm:bayesian-optimality-chain"} Within the finite setting of this paper, the following derivation chain holds:

1.  counting induces a finite measure;

2.  normalization turns that measure into probability;

3.  Bayes' rule follows from conditional probability;

4.  $\log x \le x-1$ yields Gibbs' inequality and KL nonnegativity;

5.  cross-entropy decomposes as entropy plus KL divergence;

6.  Bayes is therefore the unique minimizer of expected log loss.
:::

::: proof
*Proof.* **Step 1--2: counting to probability.** Theorem [\[thm:bayes-from-counting\]](#thm:bayes-from-counting){reference-type="ref" reference="thm:bayes-from-counting"} gives a finite measure from counting and then a normalized probability law.

**Step 3: Bayes rule.** From conditional probability, $$P(H\cap E)=P(H\mid E)P(E)=P(E\mid H)P(H),$$ so Bayes follows by rearrangement.

**Step 4: Gibbs from $\log x\le x-1$.** Apply $\log x\le x-1$ pointwise to $x=q(h)/p(h)$ and average under $p$ to obtain $$D_{\mathrm{KL}}(p\|q)\ge 0.$$

**Step 5: cross-entropy decomposition.** As in Theorem [\[thm:bayes-optimal\]](#thm:bayes-optimal){reference-type="ref" reference="thm:bayes-optimal"}, $$\mathrm{CE}(p,q)=H(p)+D_{\mathrm{KL}}(p\|q).$$

**Step 6: optimality.** Because $H(p)$ is fixed in $q$ and KL is nonnegative with equality only at $q=p$, expected log loss is uniquely minimized at $q=p$. ◻
:::

## Interpretation

The convergence claim is therefore twofold. Structurally, multiple frameworks recover $\mathrm{srank}$. Statistically, multiple inferential identities collapse to the same Bayesian optimum once the finite counting model is fixed.

The manuscript does not rely on a broad philosophical analogy between decision relevance and other frameworks. Instead it proves a sequence of concrete equivalences and bounds inside ordinary finite mathematics, with the optimizer quotient serving as the comparison object throughout.


# Related Work {#sec:related}

## Sufficient Statistics and Information Reduction

The closest classical analogue to coordinate sufficiency is the theory of sufficient statistics [@cover2006elements; @csiszar2011information; @savage1954foundations]. In both settings the goal is to replace a large object by a smaller one without losing decision-relevant structure. The difference here is one of object and emphasis: the retained information is indexed by optimal-action preservation rather than parameter inference, and the central summary is the optimizer quotient rather than a statistic defined relative to a parametric family.

## Information Geometry and Bayesian Decision Theory

The Bayesian chain used here is standard: cross-entropy decomposes into entropy plus KL divergence, and Gibbs' inequality yields Bayes optimality under log loss [@cover2006elements; @csiszar2011information; @raiffa1961applied]. What is distinctive here is the way those classical identities are attached to the optimizer quotient and to structural rank, so that Bayesian optimality appears as one part of a larger convergence story rather than as an isolated fact.

## Rate-Distortion and Lossless Compression

The link between lossless compression and decision structure is conceptually close to the minimum-description-length viewpoint [@shannon1959coding; @blahut1972computation; @rissanen1978modeling; @grunwald2007minimum]. We focus on the zero-distortion boundary relevant to exact decision preservation: once only decision-equivalent states may be merged, the quotient structure fixes the admissible compression target. This places rate-distortion language in direct contact with the relevant-coordinate support.

## Categorical and Quotient Viewpoints

The optimizer quotient is a familiar object in **Set**: it is the coimage/image factorization of the optimizer map [@maclane1998categories]. The contribution here is not novelty of construction but novelty of use. The quotient is the organizing object through which the Bayesian, entropy, and structural-rank viewpoints can be compared within a single framework.

## Formalized Mathematics and Verified Complexity

Formal verification has made major progress in algebra, analysis, and basic computability [@forster2019verified; @kunze2019formal; @nipkow2002isabelle; @mathlib2020; @demoura2021lean4]. The present work belongs to the growing body of research that uses mechanization to stabilize arguments spanning more than one mathematical viewpoint. Here the mechanized layer supports quotient-theoretic, information-theoretic, and Bayesian statements within a single development.


# Conclusion

This paper studies convergence in decision-relevant information. Starting from shared definitions of sufficiency and relevance, it identifies the optimizer quotient as the canonical decision-preserving abstraction and shows that multiple mathematical frameworks can be related to the same structural core.

## What Converges {#what-converges .unnumbered}

The common invariant is structural rank: the size of the relevant-coordinate support. Fisher-style support counting recovers that quantity exactly; quotient entropy is bounded by it; lossless zero-distortion summaries are constrained by the same quotient structure; and transport positivity/zero-cost regimes follow the same quotient branching. The point is not that these frameworks become identical, but that they can be compared against the same decision-relevant core.

## Counting Foundation {#counting-foundation .unnumbered}

The counting gap and measure-before-probability results provide the finite foundation for the Bayesian thread.

## Bayesian Consequence {#bayesian-consequence .unnumbered}

The Bayesian argument is direct. Counting on finite sets yields probability after normalization; $\log x \le x - 1$ yields Gibbs' inequality; Gibbs' inequality yields cross-entropy minimization; and Bayes emerges as the unique optimizer under log loss. This places the Bayesian conclusions on the same footing as the quotient and entropy results.

## Contribution {#contribution .unnumbered}

The contribution is to place quotient structure, structural-rank comparisons, and Bayesian optimality in one framework.

## Outlook {#outlook .unnumbered}

Natural next questions are refinement questions rather than boundary questions: which additional frameworks can be related to the same invariant and how much of the quotient story can be internalized categorically.


# Lean 4 Proof Listings {#app:lean}

The Lean development contains the shared definitions and the quotient-, Bayesian-, and information-theoretic lemmas used by this paper. This appendix lists the support most relevant to the main results.

## What Is Mechanized

The Lean development records:

-   core definitions of decision problems, sufficiency, relevance, and optimizer quotients,

-   quotient/image-factorization lemmas for the optimizer map,

-   finite probability and Bayesian optimality lemmas,

-   structural-rank support lemmas and related information-theoretic bridge statements.

## Lean Handle ID Map {#sec:lean-handle-id-map}

This appendix lists the handle families most relevant to the results of the paper:

-   `QT1`, `QT2`, `QT3`, `QT7` for quotient/coimage factorization;

-   `SK1`, `SK2`, `SK3` for structural-rank support facts;

-   `IT3`, `IT4` for quotient-entropy bounds;

-   `DQ1`, `GN7` for normalized decision-quotient range support;

-   `FS1`, `FS2` for Fisher-style structural-rank recovery;

-   `W1`, `W2`, `W3`, `W4` for the Wasserstein transport bridge;

-   `RS1`, `RS2`, `RS3`, `RS4`, `RD1`, `RD2`, `RD3` for rate-distortion and lossless-summary statements;

-   `BC1`--`BC5`, `FN7`, `FN8`, `FN12`, `FN14` for the finite Bayesian chain.

These are the handles most relevant to the claims developed in the manuscript. They provide a focused map from the text to the corresponding mechanized support.

## Representative Modules

-   `DecisionQuotient/Quotient.lean` -- quotient/coimage structure for the optimizer map

-   `DecisionQuotient/BayesBridge.lean` -- finite probability and cross-entropy identities

-   `DecisionQuotient/BayesOptimalityProof.lean` -- Bayes optimality under log loss

-   `DecisionQuotient/Information.lean` and `DecisionQuotient/Information/RateDistortion.lean` -- quotient entropy and compression-facing lemmas

-   `DecisionQuotient/Statistics/FisherInformation.lean` -- support-count style structural-rank lemmas

## Scope

For readability, this appendix provides a focused manual summary rather than repository-level omnibus tables. A paper-specific handle table should include only the statements proved or cited here, together with the Lean handles that support them.




---

## Machine-Checked Proofs

All theorems are formalized in Lean 4:
- Location: `docs/papers/paper4_decision_quotient/proofs/`
- Lines: 66580
- Theorems: 2945
- `sorry` placeholders: 0
