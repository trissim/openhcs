# Paper: Identification Capacity and Rate-Query Tradeoffs in Classification Systems

**Status**: JSAIT-ready | **Lean**: 8092 lines, 352 theorems

---

## Abstract

_Abstract not available._

## The Identification Problem

Consider an encoder-decoder pair communicating about entities from a large universe $\mathcal{V}$. The decoder must *identify* each entity, determining which of $k$ classes it belongs to, using only:

-   A *tag* of $L$ bits stored with the entity, and/or

-   *Queries* to a binary oracle: "does entity $v$ satisfy attribute $I$?"

This is not reconstruction (the decoder need not recover $v$), but *identification* in the sense of Ahlswede and Dueck [@ahlswede1989identification]: the decoder must answer "which class?" with zero or bounded error. Our work extends this framework to consider the tradeoff between tag storage, query complexity, and identification accuracy.

We prove three core theorem families, and then extend them with finite-block, open-world, and mechanization results:

1.  **Information barrier (identifiability limit).** When the attribute profile $\pi: \mathcal{V} \to \{0,1\}^n$ is not injective on classes, zero-error identification via queries alone is impossible: any decoder produces identical output on colliding classes, so cannot be correct for both.

2.  **Optimal tagging (achievability).** A tag of $L = \lceil \log_2 k \rceil$ bits achieves zero-error identification with $W = O(1)$ query cost. For maximal-barrier domains ($A_\pi = k$), this is the unique Pareto-optimal point in the $(L, W, D)$ tradeoff space at $D=0$; in general domains, the converse depends on $A_\pi := \max_u |\{c : \pi(c)=u\}|$.

3.  **Matroid structure (query complexity).** Minimal sufficient query sets form the bases of a matroid. The *distinguishing dimension* (the common cardinality of all minimal sets) is well-defined and lower-bounds the query cost $W$ for any tag-free scheme.

These results are domain-independent within the stated observation model: the same lower bounds appear across databases, biological taxonomy, knowledge graphs, and software runtimes. Once classes collide under observable attributes, constant-cost zero-error identification requires auxiliary naming information. In maximal-barrier domains this yields a unique $D=0$ Pareto solution, so the issue is resource feasibility, not domain-specific style conventions.

## The Observation Model

We formalize the observational constraint as a family of binary predicates. The terminology is deliberately abstract; concrete instantiations follow in Section [\[sec:applications\]](#sec:applications){reference-type="ref" reference="sec:applications"}.

::: definition
Let $\mathcal{V}$ be a set of entities (program objects, database records, biological specimens, library items). Let $\mathcal{I}$ be a finite set of binary *attributes*. Each $I \in \mathcal{I}$ induces a bipartition of $\mathcal{V}$ via $q_I$, and the full family induces the observational equivalence partition.
:::

::: remark
We use "attribute" for the abstract concept. Depending on domain, attributes may be interfaces/signatures, database columns, phenotypic characters, or catalog facets. The mathematics is identical.
:::

::: definition
For each $I \in \mathcal{I}$, define the attribute-membership observation $q_I: \mathcal{V} \to \{0,1\}$: $$q_I(v) = \begin{cases} 1 & \text{if } v \text{ satisfies attribute } I \\ 0 & \text{otherwise} \end{cases}$$ Let $\Phi_{\mathcal{I}} = \{q_I : I \in \mathcal{I}\}$ denote the attribute observation family.
:::

::: remark
We write $n := |\mathcal{I}|$ for the ambient number of available attributes. We write $d$ for the distinguishing dimension (the common size of all minimal distinguishing query sets; Definition [\[def:distinguishing-dimension\]](#def:distinguishing-dimension){reference-type="ref" reference="def:distinguishing-dimension"}), so $d \le n$ and there exist worst-case families with $d = n$. We write $m$ for the number of *query sites* (call sites) that perform attribute checks in a program or protocol (used only in the complexity-of-maintenance discussion). When discussing a particular identification/verification task, we may write $s$ for the number of attributes actually queried/traversed by the procedure (e.g., interface or schema checks in software/data systems, phenotypic checks in taxonomy), with $s \le n$. The maintenance-only parameter $m$ appears only in Section [\[sec:complexity\]](#sec:complexity){reference-type="ref" reference="sec:complexity"}.
:::

::: definition
The attribute profile function $\pi: \mathcal{V} \to \{0,1\}^{|\mathcal{I}|}$ maps each value to its complete attribute signature: $$\pi(v) = (q_I(v))_{I \in \mathcal{I}}$$
:::

::: definition
Values $v, w \in \mathcal{V}$ are *attribute-indistinguishable*, written $v \sim w$, iff $\pi(v) = \pi(w)$.
:::

The relation $\sim$ is an equivalence relation. We write $[v]_\sim$ for the equivalence class of $v$.

::: definition
An *attribute-only observer* is any procedure whose interaction with a value $v \in \mathcal{V}$ is limited to queries in $\Phi_{\mathcal{I}}$. Formally, the observer interacts with $v$ only via primitive attribute queries $q_I \in \Phi_{\mathcal{I}}$; hence any transcript (and output) factors through $\pi(v)$.
:::

## The Central Question

The central question is: **what semantic properties can an attribute-only observer compute?**

A semantic property is a function $P: \mathcal{V} \to \{0,1\}$ (or more generally, $P: \mathcal{V} \to Y$ for some codomain $Y$). We say $P$ is *attribute-computable* if there exists a function $f: \{0,1\}^{|\mathcal{I}|} \to Y$ such that $P(v) = f(\pi(v))$ for all $v$.

## The Information Barrier

::: theorem
[]{#thm:information-barrier label="thm:information-barrier"} Let $P: \mathcal{V} \to Y$ be any function. If $P$ is attribute-computable, then $P$ is constant on $\sim$-equivalence classes: $$v \sim w \implies P(v) = P(w)$$ Equivalently: no attribute-only observer can compute any property that varies within an equivalence class.
:::

::: proof
*Proof.* Suppose $P$ is attribute-computable via $f$, i.e., $P(v) = f(\pi(v))$ for all $v$. Let $v \sim w$, so $\pi(v) = \pi(w)$. Then: $$P(v) = f(\pi(v)) = f(\pi(w)) = P(w)$$ ◻
:::

::: remark
The barrier is *informational*, not computational. Given unlimited time, memory, and computational power, an attribute-only observer still cannot distinguish $v$ from $w$ when $\pi(v) = \pi(w)$. The constraint is on the evidence itself.
:::

::: remark
Theorem [\[thm:information-barrier\]](#thm:information-barrier){reference-type="ref" reference="thm:information-barrier"} is the base invariance statement. The technical contribution is the downstream structure built on top of it: the ambiguity-based converse (Theorem [\[thm:converse\]](#thm:converse){reference-type="ref" reference="thm:converse"}), the Pareto characterization (Theorem [\[thm:lwd-optimal\]](#thm:lwd-optimal){reference-type="ref" reference="thm:lwd-optimal"}), and the matroid/equicardinality results (Section [\[sec:matroid\]](#sec:matroid){reference-type="ref" reference="sec:matroid"}).
:::

::: corollary
[]{#cor:provenance-barrier label="cor:provenance-barrier"} Let $C: \mathcal{V} \to \{1,\ldots,k\}$ be the class assignment function. If there exist values $v, w$ with $\pi(v) = \pi(w)$ but $C(v) \neq C(w)$, then class identity is not attribute-computable.
:::

::: proof
*Proof.* Direct application of Theorem [\[thm:information-barrier\]](#thm:information-barrier){reference-type="ref" reference="thm:information-barrier"} to $P = C$. ◻
:::

## The Positive Result: Nominal Tagging

We now show that augmenting attribute observations with a single primitive, nominal-tag access, achieves constant witness cost.

::: definition
A *nominal tag* is a value $\tau(v) \in \mathcal{T}$ associated with each $v \in \mathcal{V}$, representing the class identity of $v$. The *nominal-tag access* operation returns $\tau(v)$ in $O(1)$ time.
:::

::: definition
The extended primitive query set is $\Phi_{\mathcal{I}}^+ = \Phi_{\mathcal{I}} \cup \{\tau\}$, where $\tau$ denotes nominal-tag access.
:::

::: definition
Let $W(P)$ denote the minimum number of primitive queries from $\Phi_{\mathcal{I}}^+$ required to compute property $P$. We distinguish two tasks:

-   $W_{\text{id}}$: Cost to identify the class of a single entity.

-   $W_{\text{eq}}$: Cost to determine if two entities have the same class.

Unless specified, $W$ refers to $W_{\text{eq}}$.
:::

::: theorem
[]{#thm:constant-witness label="thm:constant-witness"} Under nominal-tag access, class identity checking has constant witness cost: $$W(\text{class-identity}) = O(1)$$ Specifically, the witness procedure is: return $\tau(v_1) = \tau(v_2)$.
:::

::: proof
*Proof.* The procedure makes exactly 2 primitive queries (one $\tau$ access per value) and one comparison. This is $O(1)$ regardless of the number of available attributes $|\mathcal{I}|$. ◻
:::

::: theorem
[]{#thm:interface-lower-bound label="thm:interface-lower-bound"} For attribute-only observers, class identity checking requires: $$W(\text{class-identity}) = \Omega(d)$$ in the worst case, where $d$ is the distinguishing dimension (Definition [\[def:distinguishing-dimension\]](#def:distinguishing-dimension){reference-type="ref" reference="def:distinguishing-dimension"}).
:::

::: proof
*Proof.* Assume a zero-error attribute-only procedure halts after fewer than $d$ queries on every execution path. Fix any execution path and let $Q \subseteq \mathcal{I}$ be the set of queried attributes on that path, so $|Q|<d$. Since $d$ is the cardinality of every minimal distinguishing set, no set of size $<d$ is distinguishing; hence there exist values $v,w$ from different classes with identical answers on all attributes in $Q$.

An adversary can answer the procedure's queries consistently with both $v$ and $w$ along this path. Therefore the resulting transcript (and output) is identical on $v$ and $w$, contradicting zero-error class identification. So some execution path must use at least $d$ queries, giving worst-case cost $\Omega(d)$. ◻
:::

## Main Contributions

This paper establishes the following results:

1.  **Information Barrier Theorem** (Theorem [\[thm:information-barrier\]](#thm:information-barrier){reference-type="ref" reference="thm:information-barrier"}): Attribute-only observers cannot compute any property that varies within $\sim$-equivalence classes. This is an observational impossibility, not a computational limitation, and it is the source of the downstream rate-query tradeoffs.

2.  **Constant-Witness Theorem** (Theorem [\[thm:constant-witness\]](#thm:constant-witness){reference-type="ref" reference="thm:constant-witness"}): Nominal-tag access achieves constant witness cost for class identity. Attribute-only observers have matching lower bound $\Omega(d)$ (Theorem [\[thm:interface-lower-bound\]](#thm:interface-lower-bound){reference-type="ref" reference="thm:interface-lower-bound"}), where $d$ is the distinguishing dimension (Definition [\[def:distinguishing-dimension\]](#def:distinguishing-dimension){reference-type="ref" reference="def:distinguishing-dimension"}).

3.  **Complexity Separation** (Section [\[sec:complexity\]](#sec:complexity){reference-type="ref" reference="sec:complexity"}): We establish O(1) vs O(k) vs $\Omega(d)$ complexity bounds for error localization under different observation regimes (where $d$ is the distinguishing dimension).

4.  **Matroid Structure** (Section [\[sec:matroid\]](#sec:matroid){reference-type="ref" reference="sec:matroid"}): Minimal distinguishing query sets form the bases of a matroid. All such sets have equal cardinality, so the distinguishing dimension is an invariant of the observation model rather than an artifact of a particular proof or query order.

5.  **$(L, W, D)$ Optimality** (Section [\[sec:lwd\]](#sec:lwd){reference-type="ref" reference="sec:lwd"}): We characterize the zero-error converse via collision multiplicity $A_\pi$ and prove uniqueness of the nominal point in the maximal-barrier regime ($A_\pi=k$), turning an informal heuristic into an explicit resource law.

6.  **Finite-Block Confusability Law** (Section [\[sec:framework\]](#sec:framework){reference-type="ref" reference="sec:framework"}): We prove exact block thresholds for zero-error feasibility in the confusability formulation: at blocklength $t$, feasibility is equivalent to an alphabet threshold $A_\pi^t$, with exact linear log-bit scaling and per-coordinate stabilization.

7.  **Open-World Robustness + Decidability Boundary**: We formalize that barrier-freedom is not extension-stable in open worlds (any finite barrier-free world can be extended to force a collision), and we prove a Rice-style impossibility theorem: no computable certifier can decide barrier existence/freedom for arbitrary generators.

8.  **Machine-Checked Proofs**: All results formalized in Lean 4 (8092 lines, 352 theorem/lemma statements, 0 `sorry` placeholders).

Without the equicardinality result behind the matroid structure, the lower bound would depend on which distinguishing set or query strategy one chose. The matroid theorem is what makes $d$ strategy-independent.

## Related Work and Positioning

This paper does not develop asymptotic coding theorems or claim new zero-error capacity results. Its contribution is to make explicit the bit/query/error tradeoffs governing zero-error classification under constrained observations.

**Identification via channels.** Our work is adjacent to the identification paradigm introduced by Ahlswede and Dueck [@ahlswede1989identification; @ahlswede1989identification2]. In their framework, a decoder need not reconstruct a message but only answer "is the message $m$?" for a given hypothesis. Our setting is different in three ways: (1) we consider one-shot zero-error class identification rather than asymptotic vanishing-error identification, (2) queries are adaptive rather than block codes, and (3) we allow auxiliary tagging (rate $L$) to reduce query cost. The connection is motivational and terminological rather than a direct reuse of the original coding theorem.

**Rate-distortion theory.** The $(L, W, D)$ framework is analogous to rate-distortion theory [@shannon1959coding; @berger1971rate]: the "distortion" $D$ is semantic (class misidentification), and there is a second resource $W$ (query cost) alongside rate $L$. Classical rate-distortion asks for the minimum rate needed to achieve distortion $D$; here the main question is which combinations of tag bits and queries permit $D=0$. Theorem [\[thm:lwd-optimal\]](#thm:lwd-optimal){reference-type="ref" reference="thm:lwd-optimal"} gives the resulting converse in terms of collision multiplicity and identifies the unique nominal point in the maximal-barrier regime.

**Rate-distortion-perception tradeoffs.** Blau and Michaeli [@blau2019rethinking] extended rate-distortion theory by adding a perception constraint, creating a three-way tradeoff. Our query cost $W$ plays an analogous role: it measures the interactive cost of achieving low distortion rather than a distributional constraint. We cite this as a structural analogy, not as a claim that the present model inherits the full geometry of that setting.

**Zero-error information theory.** The matroid structure (Section [\[sec:matroid\]](#sec:matroid){reference-type="ref" reference="sec:matroid"}) places the model near zero-error information theory. Körner [@korner1973coding] and Witsenhausen [@witsenhausen1976zero] studied zero-error source coding where confusable symbols must be distinguished. Here the direct theorem is simpler: when $L = 0$, the distinguishing dimension (Definition [\[def:distinguishing-dimension\]](#def:distinguishing-dimension){reference-type="ref" reference="def:distinguishing-dimension"}) is the minimum number of binary queries needed for zero-error class separation. In addition, the paper proves exact finite-block confusability laws: block feasibility threshold $A_\pi^k$, exact linear log-bit scaling, and per-coordinate stabilization in the worst-case regime. These are formalized in Lean and strengthen the operational converse while remaining distinct from full distribution-optimized graph-entropy/capacity theorems.

**Query complexity and communication complexity.** The $\Omega(d)$ lower bound for attribute-only identification relates to decision tree complexity [@buhrman2002complexity] and interactive communication [@orlitsky1991worst]. The key distinction is that our queries are constrained to a fixed observable family $\mathcal{I}$, not arbitrary predicates.

**Compression in classification systems.** The same ambiguity bound appears across databases, knowledge graphs, taxonomy, and software-runtime typing systems: for zero-error identification, ambiguity induces a minimum metadata requirement $L \ge \log_2 A_\pi$ (Theorem [\[thm:converse\]](#thm:converse){reference-type="ref" reference="thm:converse"}), with maximal-barrier specialization $L \ge \log_2 k$.

**Runtime/schema instantiation (secondary).** In nominal-vs-structural interface settings [@Cardelli1985; @cook1990inheritance], the model yields a concrete cost statement: under attribute collisions, purely profile-based identification has worst-case $\Omega(d)$ witness cost, while explicit tags achieve $O(1)$ identification using $O(\log A_\pi)$ bits. This is one instantiation of the general bounds.

## Paper Organization

Section [\[sec:framework\]](#sec:framework){reference-type="ref" reference="sec:framework"} formalizes the compression framework and defines the $(L, W, D)$ tradeoff. Section [\[sec:complexity\]](#sec:complexity){reference-type="ref" reference="sec:complexity"} establishes complexity bounds for error localization. Section [\[sec:matroid\]](#sec:matroid){reference-type="ref" reference="sec:matroid"} proves the matroid structure of distinguishing query families. Section [\[sec:witness\]](#sec:witness){reference-type="ref" reference="sec:witness"} analyzes witness cost in detail. Section [\[sec:lwd\]](#sec:lwd){reference-type="ref" reference="sec:lwd"} proves the ambiguity-based converse and Pareto characterization. Section [\[sec:conclusion\]](#sec:conclusion){reference-type="ref" reference="sec:conclusion"} concludes. Extended instantiations and the Lean formalization are provided in the supplementary material.


## Semantic Compression: The Problem

The fundamental problem of *semantic compression* is: given a value $v$ from a large space $\mathcal{V}$, how can we represent $v$ compactly while preserving the ability to answer semantic queries about $v$? This differs from classical source coding in that the goal is not reconstruction but *identification*: determining which equivalence class $v$ belongs to.

Classical rate-distortion theory [@shannon1959coding] studies the tradeoff between representation size and reconstruction fidelity. We extend this to a discrete classification setting with three dimensions: *tag length* $L$ (bits of storage), *witness cost* $W$ (queries or bits of communication required to determine class membership), and *distortion* $D$ (misclassification probability).

This work treats contemporary classification problems (databases, knowledge graphs, model registries) with explicit resource accounting. The point is not to claim a new Shannon-style asymptotic coding theorem, but to make explicit the bit/query/error tradeoffs that already constrain zero-error identification in practice.

## Universe of Discourse

::: definition
A *classification scheme* is any procedure (deterministic or randomized), with arbitrary time and memory, whose only access to a value $v \in \mathcal{V}$ is via:

1.  The *observation family* $\Phi = \{q_I : I \in \mathcal{I}\}$, where $q_I(v) = 1$ iff $v$ satisfies attribute $I$; and optionally

2.  A *nominal-tag primitive* $\tau: \mathcal{V} \to \mathcal{T}$ returning an opaque class identifier.

All theorems in this paper quantify over all such schemes.
:::

This definition is intentionally broad: schemes may be adaptive, randomized, or computationally unbounded. The constraint is *observational*, not computational.

::: theorem
[]{#thm:info-barrier label="thm:info-barrier"} For all classification schemes with access only to $\Phi$ (no nominal tag), the output is constant on $\sim_\Phi$-equivalence classes. Therefore, no such scheme can compute any property that varies within a $\sim_\Phi$-class.
:::

::: proof
*Proof.* Let $v \sim_\Phi w$, meaning $q_I(v) = q_I(w)$ for all $I \in \mathcal{I}$. Any scheme's execution trace depends only on query responses. Since all queries return identical values for $v$ and $w$, the scheme cannot distinguish them. Any output must therefore be identical. ◻
:::

::: proposition
[]{#prop:model-capture label="prop:model-capture"} Any real-world classification protocol whose evidence consists solely of attribute-membership queries is representable as a scheme in the above model. Conversely, any additional capability corresponds to adding new observations to $\Phi$.
:::

This proposition forces any objection into a precise form: to claim the theorem does not apply, one must name the additional observation capability not in $\Phi$. "Different universe" is not a coherent objection; it must reduce to "I have access to oracle $X \notin \Phi$."

## Two-Axis Instantiation (Representative Domain Example)

The core results require only $(\mathcal{V}, C, \pi)$ and the observation family $\Phi$. The two-axis decomposition below is one representative instantiation and is not an additional axiom for the general theorems.

In that instantiation, each value is characterized by:

-   **Provenance axis ($B$)**: source/lineage information that can distinguish entities with identical observable profiles[^1]

-   **Profile axis ($S$)**: the observable attribute profile induced by $\Phi$

::: definition
A value $v \in \mathcal{V}$ has representation $(B(v), S(v))$ where: $$\begin{aligned}
B(v) &= \text{provenance}(v) \\
S(v) &= \pi(v) = (q_I(v))_{I \in \mathcal{I}}
\end{aligned}$$ The provenance axis captures *identity-bearing origin information*; the profile axis captures *observable behavior/attributes*.

In concrete deployments, $B$ may be carried by lineage metadata, registry identifiers, or equivalent provenance carriers. Any implementation-specific normalization or lookup machinery is auxiliary.
:::

::: theorem
[]{#thm:model-completeness label="thm:model-completeness"} Let a fixed-axis domain be specified by an axis map $\alpha: \mathcal{V} \to \mathcal{A}$ and an observation family $\Phi$ such that each primitive query $q \in \Phi$ factors through $\alpha$. Then every in-scope semantic property (i.e., any property computable by an admissible $\Phi$-only strategy) factors through $\alpha$: there exists $\tilde{P}$ with $$P(v) = \tilde{P}(\alpha(v)) \quad \text{for all } v \in \mathcal{V}.$$
:::

In this two-axis instantiation, $\alpha(v) = (B(v), S(v))$, so in-scope semantic properties are functions of $(B,S)$.

::: proof
*Proof.* An admissible $\Phi$-only strategy observes $v$ solely through responses to primitive queries $q_I \in \Phi$. By hypothesis each such response is a function of $\alpha(v)$. Therefore every query transcript, and hence any strategy's output, depends only on $\alpha(v)$, so the computed property factors through $\alpha$. ◻
:::

::: corollary
[]{#cor:fixed-axis-incompleteness label="cor:fixed-axis-incompleteness"} Any fixed-axis classification system is complete only for properties measurable on the fixed axis map $\alpha$, and incomplete for any property that varies within an $\alpha$-fiber. Equivalently, if $\alpha(v)=\alpha(w)$ but $P(v)\neq P(w)$, then no admissible $\Phi$-only strategy can compute $P$ with zero error.
:::

## Attribute Equivalence and Observational Limits

Recall from Section 1 the attribute equivalence relation:

::: definition
Values $v, w \in \mathcal{V}$ are attribute-equivalent, written $v \sim w$, iff $\pi(v) = \pi(w)$, i.e., they induce exactly the same attribute responses.
:::

::: proposition
[]{#prop:equivalence-class-structure label="prop:equivalence-class-structure"} The relation $\sim$ partitions $\mathcal{V}$ into equivalence classes. Let $\mathcal{V}/{\sim}$ denote the quotient space. An attribute-only observer effectively operates on $\mathcal{V}/{\sim}$, not $\mathcal{V}$.
:::

::: corollary
[]{#cor:information-loss-quantification label="cor:information-loss-quantification"} The information lost by attribute-only observation is: $$H(\mathcal{V}) - H(\mathcal{V}/{\sim}) = H(\mathcal{V} | \pi)$$ where $H$ denotes entropy. This quantity is positive whenever multiple classes share the same attribute profile.
:::

## One-Shot Zero-Error Feasibility {#sec:identification-capacity}

We now formalize the identification problem in channel-theoretic terms. Let $C: \mathcal{V} \to \{1, \ldots, k\}$ denote the class assignment function, and let $\pi: \mathcal{V} \to \{0,1\}^n$ denote the attribute profile.

::: definition
The *observation channel* induced by observation family $\Phi$ is the mapping $C \to \pi(V)$ for a random entity $V$ drawn from distribution $P_V$ over $\mathcal{V}$. The channel output is the attribute profile; the channel input is implicitly the class $C(V)$.
:::

::: theorem
[]{#thm:identification-capacity label="thm:identification-capacity"} Let $\mathcal{C} = \{1, \ldots, k\}$ be the class space. The *zero-error identification capacity* of the observation channel is: $$C_{\text{id}} = \begin{cases}
\log_2 k & \text{if } \pi \text{ is injective on classes} \\
0 & \text{otherwise}
\end{cases}$$ That is, zero-error identification of all $k$ classes is achievable if and only if every class has a distinct attribute profile. When $\pi$ is not class-injective, no rate of identification is achievable with zero error.
:::

::: proof
*Proof.* *Achievability*: If $\pi$ is injective on classes, then observing $\pi(v)$ determines $C(v)$ uniquely. The decoder simply inverts the class-to-profile mapping.

*Converse* (deterministic): Suppose two distinct classes $c_1 \neq c_2$ share a profile: $\exists v_1 \in c_1, v_2 \in c_2$ with $\pi(v_1) = \pi(v_2)$. Then any decoder $g(\pi(v))$ outputs the same class label on both $v_1$ and $v_2$, so it cannot be correct for both. Hence zero-error identification of all classes is impossible. ◻
:::

::: remark
Under any distribution with positive mass on both colliding classes, $I(C; \pi(V)) < H(C)$. This is an average-case consequence of the deterministic barrier above.
:::

::: remark
In the identification paradigm of [@ahlswede1989identification], the decoder asks "is the message $m$?" rather than "what is the message?" This yields double-exponential codebook sizes. Our setting is different: we require zero-error identification of the *class*, not hypothesis testing. The one-shot zero-error identification feasibility threshold ($\pi$ must be class-injective) is binary rather than a rate.
:::

::: remark
Theorem [\[thm:identification-capacity\]](#thm:identification-capacity){reference-type="ref" reference="thm:identification-capacity"} is a one-shot feasibility statement, not a Shannon asymptotic coding theorem. The notation is retained only as a compact summary of the zero-error/non-zero-error dichotomy under the stated observation model.
:::

The key insight is that tagging provides a *side channel* that restores identifiability when the attribute channel fails:

::: theorem
[]{#thm:tag-restored label="thm:tag-restored"} A class-injective nominal tag of length $L \geq \lceil \log_2 k \rceil$ bits suffices for zero-error identification, regardless of whether $\pi$ is class-injective.
:::

::: proof
*Proof.* A nominal tag $\tau: \mathcal{V} \to \{1, \ldots, k\}$ assigns a unique identifier to each class. Reading $\tau(v)$ determines $C(v)$ in $O(1)$ queries, independent of the attribute channel. ◻
:::

## Witness Cost: Query Complexity for Semantic Properties

::: definition
A *witness procedure* for property $P: \mathcal{V} \to Y$ is an algorithm $A$ that:

1.  Takes as input a value $v \in \mathcal{V}$ (via query access only)

2.  Makes queries to the primitive set $\Phi_{\mathcal{I}}^+$

3.  Outputs $P(v)$
:::

::: definition
The *witness cost* of property $P$ is: $$W(P) = \min_{A \text{ computes } P} c(A)$$ where $c(A)$ is the worst-case number of primitive queries made by $A$.
:::

::: remark
Witness cost is a form of query complexity [@buhrman2002complexity] specialized to semantic properties. Unlike Kolmogorov complexity, $W$ is computable and depends on the primitive set, not a universal machine.
:::

::: lemma
[]{#lem:witness-cost-lower-bounds label="lem:witness-cost-lower-bounds"} For any property $P$:

1.  If $P$ is attribute-computable: $W(P) \leq |\mathcal{I}|$

2.  If $P$ varies within some $\sim$-class: $W(P) = \infty$ for attribute-only observers

3.  With nominal-tag access: $W(\text{class-identity}) = O(1)$
:::

## The $(L, W, D)$ Tradeoff

We now define the three-dimensional tradeoff space that characterizes observation strategies, using information-theoretic units.

::: definition
For a set of class identifiers (tags) $\mathcal{T}$ with $|\mathcal{T}| = k$, the *tag rate* $L$ is the minimum number of bits required to encode a class identifier: $$L \geq \log_2 k \quad \text{bits per value}$$ For nominal-tag observers, $L = \lceil \log_2 k \rceil$ (optimal prefix-free encoding). For attribute-only observers, $L = 0$ (no explicit tag stored). Under a distribution $P$ over classes, the expected tag length is $\mathbb{E}[L] \geq H(P)$ by Shannon's source coding theorem [@shannon1959coding].
:::

::: definition
The *witness cost* $W$ is the minimum number of primitive queries (or bits of interactive communication) required for class identification: $$W = \min_{A \text{ decides class}} c(A)$$ where $c(A)$ is the worst-case query count. This is a form of query complexity [@buhrman2002complexity] or interactive identification cost.
:::

::: definition
Fix class map $C : \mathcal{V}\to\{1,\ldots,k\}$. An observation strategy $g$ induces an estimate $$\hat{C}_g(v;\omega) \in \{1,\ldots,k\}$$ from the available evidence (tag bits, query transcript, and optional internal randomness $\omega$).
:::

::: definition
For strategy $g$, define $$d_g(v;\omega) := \mathbf{1}\!\left[\hat{C}_g(v;\omega)\neq C(v)\right].$$ Under data distribution $P_V$ and strategy randomness, expected distortion is $$D(g)=\Pr_{v\sim P_V,\omega}\!\big[\hat{C}_g(v;\omega)\neq C(v)\big].$$ The zero-error regime is $D(g)=0$.
:::

::: remark
In this paper, $D$ is strictly class-misidentification probability. Additional semantic notions (e.g., hierarchical or behavior-weighted penalties) are treated as extensions in the supplementary material.
:::

## The $(L, W, D)$ Tradeoff Space

**Admissible schemes.** To make the Pareto-optimality claim precise, we specify the class of admissible observation strategies:

-   **Deterministic or randomized**: Schemes may use randomness; $W$ is worst-case query count.

-   **Computationally unbounded**: No time/space restrictions; the constraint is observational.

-   **No preprocessing over class universe**: The scheme cannot precompute a global lookup table indexed by all possible classes.

-   **Tags are injective on classes**: A nominal tag $\tau(v)$ uniquely identifies the class of $v$. Variable-length or compressed tags are permitted; $L$ counts bits.

-   **No amortization across queries**: $W$ is per-identification cost, not amortized over a sequence.

**Justification.** The "no preprocessing" and "no amortization" constraints exclude trivializations:

-   *Preprocessing*: With unbounded preprocessing over the class universe $\mathcal{T}$, one could build a lookup table mapping attribute profiles to classes. This reduces identification to $O(1)$ table lookup, but the table has size $O(|\mathcal{T}|)$, hiding the complexity in space rather than eliminating it. The constraint models systems that cannot afford $O(|\mathcal{T}|)$ storage per observer.

-   *Amortization*: If $W$ were amortized over a sequence of identifications, one could cache earlier results. This again hides complexity in state. The per-identification model captures stateless observers (typical in database queries, taxonomy lookup, and protocol/classification services).

Dropping these constraints changes the achievable region but not the qualitative separation: nominal tags still dominate for $D = 0$ because they provide $O(1)$ worst-case identification without requiring $O(|\mathcal{T}|)$ preprocessing.

Under these rules, "dominance" means strict improvement on at least one of $(L, W, D)$ with no regression on others.

::: definition
A point $(L, W, D)$ is *achievable* if there exists an admissible observation strategy realizing those values. Let $\mathcal{R} \subseteq \mathbb{R}_{\geq 0} \times \mathbb{R}_{\geq 0} \times [0,1]$ denote the achievable region.
:::

::: definition
A point $(L^*, W^*, D^*)$ is *Pareto-optimal* if there is no achievable $(L, W, D)$ with $L \leq L^*$, $W \leq W^*$, $D \leq D^*$, and at least one strict inequality.
:::

The main result of Section [\[sec:lwd\]](#sec:lwd){reference-type="ref" reference="sec:lwd"} is: (i) a converse in terms of collision multiplicity $A_\pi$, and (ii) uniqueness of the nominal $D=0$ Pareto point in the maximal-barrier regime.

::: definition
[]{#def:info-barrier-domain label="def:info-barrier-domain"} A classification domain has an *information barrier* (relative to $\Phi$) if there exist distinct classes $c_1 \neq c_2$ with identical $\Phi$-profiles. Equivalently, $\pi$ is not injective on classes.
:::

::: definition
[]{#def:collision-multiplicity label="def:collision-multiplicity"} Let $\mathcal{C}=\{1,\ldots,k\}$ and let $\pi_{\mathcal{C}}:\mathcal{C}\to\{0,1\}^n$ be the class-level profile map. Define $$A_\pi := \max_{u \in \operatorname{Im}(\pi_{\mathcal{C}})} \left|\{c \in \mathcal{C} : \pi_{\mathcal{C}}(c)=u\}\right|.$$ Thus $A_\pi$ is the size of the largest class-collision block under observable profiles.
:::

::: definition
[]{#def:max-barrier label="def:max-barrier"} The domain is *maximal-barrier* if $A_\pi = k$, i.e., all classes collide under the observation map.
:::

## Open-World Robustness and Decidability Boundary {#sec:open-world-boundary}

::: definition
Let $\mathcal{C}$ be the class universe. A *finite realized world* is a finite subset $W \subseteq \mathcal{C}$ of classes currently present in a deployed system snapshot. We call $W$ *barrier-free* iff $$\forall c_1,c_2 \in W,\ \pi_{\mathcal C}(c_1)=\pi_{\mathcal C}(c_2) \Rightarrow c_1=c_2.$$
:::

::: definition
An *open-world extension* is a super-world $W' \supseteq W$ obtained by adding classes while keeping the observation family $\Phi$ fixed.
:::

::: theorem
[]{#thm:open-world-extension-instability label="thm:open-world-extension-instability"} In the open-world class model formalized in Lean, it is not true that barrier-freedom is preserved under all extensions: $$\neg\Big(\forall W \subseteq W',\ \text{BarrierFree}(W)\Rightarrow \text{BarrierFree}(W')\Big).$$
:::

::: proof
*Proof.* Take the empty world $W_0=\varnothing$, which is barrier-free. In the Lean model, two concrete classes (`ConfigType` and `StepConfigType`) are shape-equivalent (same observable profile) but nominally distinct (different lineage), so adding them yields an extension $W_1\supseteq W_0$ that is not barrier-free. Therefore barrier-freedom is not extension-stable. ◻
:::

::: definition
For a partial observer generator $f:\mathbb{N}\rightharpoonup\mathbb{N}$, define $$\mathrm{HasBarrier}(f)\;:\Leftrightarrow\;\exists x\neq y,\exists z,\ z\in f(x)\land z\in f(y).$$ Define $\mathrm{BarrierFree}(f):\Leftrightarrow \neg\mathrm{HasBarrier}(f)$.
:::

::: theorem
[]{#thm:rice-barrier label="thm:rice-barrier"} There is no computable predicate on program codes deciding whether the generated observer has a barrier: $$\neg\mathrm{ComputablePred}\!\left(c\mapsto \mathrm{HasBarrier}(\mathrm{eval}(c))\right).$$ Equivalently, barrier-freedom certification is also non-computable: $$\neg\mathrm{ComputablePred}\!\left(c\mapsto \mathrm{BarrierFree}(\mathrm{eval}(c))\right).$$
:::

::: proof
*Proof.* Assume there is a computable certifier for $P(c):=\mathrm{HasBarrier}(\mathrm{eval}(c))$. Set $$\mathcal{C}_{\mathrm{bar}}:=\{f:\mathbb{N}\rightharpoonup\mathbb{N}\mid \mathrm{HasBarrier}(f)\}.$$ Then $P(c)$ is exactly membership of $\mathrm{eval}(c)$ in $\mathcal{C}_{\mathrm{bar}}$. The property is non-trivial: $$\mathrm{HasBarrier}(\lambda x.\,0)\quad\text{and}\quad \neg\mathrm{HasBarrier}(\lambda x.\,x+1).$$ Rice's theorem (formalized in Lean and instantiated with these two witnesses) implies that if membership in $\mathcal{C}_{\mathrm{bar}}$ were computable from code, then every partial recursive function would have to lie in $\mathcal{C}_{\mathrm{bar}}$, contradicting the successor witness. Hence $P$ is not computable. The barrier-free statement is equivalent by $$\mathrm{BarrierFree}(f)\iff \neg\mathrm{HasBarrier}(f),$$ so computability of one would imply computability of the other. ◻
:::

::: corollary
For unrestricted open-world generators, one cannot computably certify that attribute-only identification remains barrier-free under all future extensions. Consequently, persistent $D=0$ guarantees cannot rely on "currently collision-free" attribute structure alone.
:::

## Converse: Tag Rate Lower Bound

::: theorem
[]{#thm:converse label="thm:converse"} For any classification domain, any scheme achieving $D=0$ requires $$2^L \ge A_\pi
\quad\text{equivalently}\quad
L \ge \log_2 A_\pi,$$ where $A_\pi$ is the collision multiplicity from Definition [\[def:collision-multiplicity\]](#def:collision-multiplicity){reference-type="ref" reference="def:collision-multiplicity"}.
:::

::: proof
*Proof.* Fix a collision block $G \subseteq \mathcal{C}$ with $|G|=A_\pi$ and identical observable profile. For classes in $G$, query transcripts are identical, so zero-error decoding must separate those classes using tag outcomes. With $L$ tag bits there are at most $2^L$ outcomes, hence $2^L \ge |G| = A_\pi$. ◻
:::

::: corollary
[]{#cor:max-barrier-converse label="cor:max-barrier-converse"} If the domain is maximal-barrier ($A_\pi = k$), any zero-error scheme satisfies $L \ge \log_2 k$.
:::

## Confusability Graph Thresholds and Block Scaling

Let $\pi_{\mathcal{C}} : \mathcal{C} \to \{0,1\}^n$ be the class-profile map from Definition [\[def:collision-multiplicity\]](#def:collision-multiplicity){reference-type="ref" reference="def:collision-multiplicity"}. The induced confusability graph connects two distinct classes when they share the same observable profile.

::: theorem
[]{#thm:graph-one-shot-threshold label="thm:graph-one-shot-threshold"} For an alphabet of $n$ nominal tags, zero-error class tagging is feasible if and only if $$A_\pi \le n.$$
:::

::: proof
*Proof.* Classes in the same profile fiber are pairwise confusable, so each such fiber is a clique in the confusability graph. Zero-error tagging is equivalent to a proper graph coloring; therefore feasibility is equivalent to having at least as many colors as the largest clique size, which is exactly $A_\pi$. ◻
:::

::: theorem
[]{#thm:graph-block-threshold label="thm:graph-block-threshold"} For block length $t\ge 1$ with coordinatewise observation on $\mathcal{C}^t$, zero-error tagging with alphabet size $n_t$ is feasible if and only if $$A_\pi^t \le n_t.$$ Equivalently, the minimal feasible block alphabet is exactly $A_\pi^t$.
:::

::: corollary
[]{#cor:graph-logbit-scaling label="cor:graph-logbit-scaling"} Let $L_t^\star := \log_2(A_\pi^t)$ be the minimum block tag budget in bits. Then $$L_t^\star = t\,\log_2 A_\pi,
\qquad
\frac{L_t^\star}{t} = \log_2 A_\pi.$$ So per-entity tag rate is exactly stable across block length.
:::

::: remark
These are worst-case finite-block confusability laws. They strengthen the zero-error converse with exact block thresholds, while remaining distinct from full distribution-optimized Körner graph-entropy theory.
:::

## Lossy Regime: Deterministic vs Noisy Models

The zero-error corner ($D=0$) is governed by Theorem [\[thm:converse\]](#thm:converse){reference-type="ref" reference="thm:converse"}. For $D>0$, the model matters:

-   **Deterministic queries (this section):** there is no universal law of the form $W=O(\log(1/\epsilon)\cdot d)$. If classes collide on all deterministic observations, repeating those same observations does not reduce error.

-   **Noisy queries:** repeated independent observations can reduce error exponentially, yielding logarithmic-in-$1/\epsilon$ sample complexity.

Thus, in the deterministic model, distortion is controlled by collision geometry and decision rules; in noisy models, repetition-based concentration bounds become relevant.

## Concrete Example

Consider a classification system with $k = 1000$ classes, each characterized by a subset of $n = 50$ binary attributes. Table [1](#tab:strategies){reference-type="ref" reference="tab:strategies"} compares the strategies.

::: {#tab:strategies}
  **Strategy**                                **Tag $L$**                 **Witness $W$**
  ----------------------------- --------------------------------------- -------------------
  Explicit-tag identification    $\lceil \log_2 1000 \rceil = 10$ bits        $O(1)$
  Attribute-only (exhaustive)                     $0$                    $\leq 50$ queries
  Attribute-only (adaptive)                       $0$                    $\geq d$ queries

  : Identification strategies for 1000 classes with 50 attributes.
:::

Here $d$ is the distinguishing dimension, the size of any minimal distinguishing query set. For many practical systems, $d$ is well below $n$ but nonzero. The gap between 10 bits of storage versus 5--50 queries per identification is the cost of forgoing explicit tags.

[^1]: In the Lean witness model, this axis is represented as `Bases`.


## The Error Localization Theorem

#### Scope of this section.

This section studies *maintenance/localization* complexity, measured by locations to inspect ($E(\cdot)$), not per-instance identification complexity ($W$) from Sections [\[sec:framework\]](#sec:framework){reference-type="ref" reference="sec:framework"} and [\[sec:witness\]](#sec:witness){reference-type="ref" reference="sec:witness"}. The metrics are related but distinct: $W$ quantifies online class-identification effort, while $E$ quantifies where constraint logic is distributed in implementations.

::: definition
Let $E(\mathcal{O})$ be the number of locations that must be inspected to find all potential violations of a constraint under observation family $\mathcal{O}$.
:::

::: theorem
[]{#thm:nominal-localization label="thm:nominal-localization"} $E(\text{nominal-tag}) = O(1)$.
:::

::: proof
*Proof.* Under nominal-tag observation, the constraint "$v$ must be of class $A$" is satisfied iff $\tau(v) \in \text{subtypes}(A)$. This is determined at a single location: the definition of $\tau(v)$'s class. One location. ◻
:::

::: theorem
[]{#thm:declared-localization label="thm:declared-localization"} $E(\text{attribute-only, declared}) = O(k)$ where $k$ = number of entity classes.
:::

::: proof
*Proof.* With declared attribute sets, the constraint "$v$ must satisfy attribute $I$" requires verifying that each class satisfies all attributes in $I$. For $k$ classes, $O(k)$ locations. ◻
:::

::: theorem
[]{#thm:attribute-localization label="thm:attribute-localization"} $E(\text{attribute-only}) = \Omega(m)$ where $m$ = number of query sites.
:::

::: proof
*Proof.* Under attribute-only observation, each query site independently checks "does $v$ have attribute $a$?" with no centralized declaration. For $m$ query sites, each must be inspected. Lower bound is $\Omega(m)$. ◻
:::

::: corollary
[]{#cor:strict-dominance label="cor:strict-dominance"} Nominal-tag observation strictly dominates attribute-only: $E(\text{nominal-tag}) = O(1) < \Omega(m) = E(\text{attribute-only})$ for all $m > 1$.
:::

## The Information Scattering Theorem

::: definition
Let $I(\mathcal{O}, c)$ be the set of locations where constraint $c$ is encoded under observation family $\mathcal{O}$.
:::

::: theorem
[]{#thm:attribute-scattering label="thm:attribute-scattering"} For attribute-only observation, $|I(\text{attribute-only}, c)| = O(m)$ where $m$ = query sites using constraint $c$.
:::

::: proof
*Proof.* Each attribute query independently encodes the constraint. No shared reference exists. Constraint encodings scale with query sites. ◻
:::

::: theorem
[]{#thm:nominal-centralization label="thm:nominal-centralization"} For nominal-tag observation, $|I(\text{nominal-tag}, c)| = O(1)$.
:::

::: proof
*Proof.* The constraint "must be of class $A$" is encoded once in the definition of $A$. All tag checks reference this single definition. ◻
:::

::: corollary
[]{#cor:maintenance-entropy label="cor:maintenance-entropy"} Attribute-only observation maximizes maintenance entropy; nominal-tag observation minimizes it.
:::


## Model Contract (Fixed-Axis Domains)

Model contract (fixed-axis domain). A domain is specified by a fixed observation family $\Phi$ derived from a fixed axis map $\alpha: \mathcal{V} \to \mathcal{A}$ (e.g., $\alpha(v) = (B(v), S(v))$). An observer is permitted to interact with $v$ only through primitive queries in $\Phi$, and each primitive query factors through $\alpha$: for every $q \in \Phi$, there exists $\tilde{q}$ such that $q(v) = \tilde{q}(\alpha(v))$. A property is in-scope semantic iff it is computable by an admissible strategy that uses only responses to queries in $\Phi$ (under our admissibility constraints: no global preprocessing tables, no amortized caching, etc.).

We adopt $\Phi$ as the complete observation universe for this paper: to claim applicability to a concrete runtime one must either (i) exhibit mappings from each runtime observable into $\Phi$, or (ii) enforce the admissibility constraints (no external registries, no reflection, no preprocessing/amortization). Under either condition the theorems apply without qualification.

::: proposition
[]{#prop:observational-quotient label="prop:observational-quotient"} For any admissible strategy using only $\Phi$, the entire interaction transcript (and hence the output) depends only on $\alpha(v)$. Equivalently, any in-scope semantic property $P$ factors through $\alpha$: there exists $\tilde{P}$ with $P(v) = \tilde{P}(\alpha(v))$ for all $v$.
:::

::: corollary
[]{#cor:ad-hoc-add-axis label="cor:ad-hoc-add-axis"} If two values $v, w$ satisfy $\alpha(v) = \alpha(w)$, then no admissible $\Phi$-only strategy can distinguish them with zero error. Any mechanism that does distinguish such pairs must introduce additional information not present in $\alpha$ (equivalently, refine the axis map by adding a new axis/tag).
:::

## Query Families and Distinguishing Sets

The classification problem is: given a set of queries, which subsets suffice to distinguish all entities?

::: definition
Let $\mathcal{Q}$ be the set of all primitive queries available to an observer. For a classification system with attribute set $\mathcal{I}$, we have $\mathcal{Q} = \{q_I : I \in \mathcal{I}\}$ where $q_I(v) = 1$ iff $v$ satisfies attribute $I$.
:::

In this section, "queries" are the primitive attribute predicates $q \in \Phi$ (equivalently, each $q$ factors through the axis map: $q = \tilde{q} \circ \alpha$). See the Convention above where $\Phi := \mathcal{Q}$.

**Convention:** $\Phi := \mathcal{Q}$. All universal quantification over "queries" ranges over $q \in \Phi$ only.

::: definition
A subset $S \subseteq \mathcal{Q}$ is *distinguishing* if, for all values $v, w$ with $\text{class}(v) \neq \text{class}(w)$, there exists $q \in S$ such that $q(v) \neq q(w)$.
:::

::: definition
A distinguishing set $S$ is *minimal* if no proper subset of $S$ is distinguishing.
:::

## Matroid Structure of Query Families

**Scope and assumptions.** The matroid theorem below is unconditional within the fixed-axis observational theory defined above. In this section, "query" always means a primitive predicate $q \in \Phi$ (equivalently, $q$ factors through $\alpha$ as in the Model Contract). It depends only on:

-   $E = \Phi$ is the ground set of primitive queries (attribute predicates).

-   "Distinguishing": for all values $v,w$ with $\text{class}(v) \neq \text{class}(w)$, there exists $q \in S$ such that $q(v) \neq q(w)$ (Def. above).

-   "Minimal" means inclusion-minimal: no proper subset suffices.

No further assumptions are required within this theory beyond the fixed observation family $\Phi$. In Lean, the mechanization is explicit end-to-end. Closure-operator lemmas (extensive, monotone, idempotent) are proved in `proofs/abstract_class_system.lean` via `AxisClosure`. Exchange and equicardinality lemmas are proved in `proofs/axis_framework.lean` via L4 and L1. The composition point is formalized by `closureInducedAxisMatroid` and L1, with closure-induced nodup/subset/exchange hypotheses made explicit.

::: definition
Let $E = \Phi\;(=\mathcal{Q})$ be the ground set of primitive queries (attribute predicates). Let $\mathcal{B} \subseteq 2^E$ be the family of minimal distinguishing sets.
:::

::: lemma
[]{#lem:basis-exchange label="lem:basis-exchange"} For any $B_1, B_2 \in \mathcal{B}$ and any $q \in B_1 \setminus B_2$, there exists $q' \in B_2 \setminus B_1$ such that $(B_1 \setminus \{q\}) \cup \{q'\} \in \mathcal{B}$.
:::

::: proof
*Proof.* Define the closure operator $\text{cl}(X) = \{q : X\text{-equivalence implies }q\text{-equivalence}\}$. We verify the matroid axioms:

1.  **Closure axioms**: $\text{cl}$ is extensive, monotone, and idempotent. These follow directly from the definition of logical implication.

2.  **Exchange property**: If $q \in \text{cl}(X \cup \{q'\}) \setminus \text{cl}(X)$, then $q' \in \text{cl}(X \cup \{q\})$.

For exchange, take $q \in \text{cl}(X \cup \{q'\}) \setminus \text{cl}(X)$. Since $q \notin \text{cl}(X)$, there exist $v,w$ that are $X$-equivalent but disagree on $q$. Because $q \in \text{cl}(X \cup \{q'\})$, any pair that is $(X \cup \{q'\})$-equivalent must agree on $q$; therefore this witness pair cannot be $(X \cup \{q'\})$-equivalent, so it must disagree on $q'$. Now fix any pair $v',w'$ that are $(X \cup \{q\})$-equivalent. They are in particular $X$-equivalent and agree on $q$. If they disagreed on $q'$, then by the previous implication we could derive disagreement on $q$, contradiction. Hence $v',w'$ agree on $q'$, proving $q' \in \text{cl}(X \cup \{q\})$.

Minimal distinguishing sets are exactly the bases of the matroid defined by this closure operator. The closure component is machine-checked in `AxisClosure`; the exchange/equicardinality component and the closure-to-matroid bridge used for distinguishing-dimension well-definedness are machine-checked in `axis_framework.lean`. ◻
:::

::: theorem
[]{#thm:matroid-bases label="thm:matroid-bases"} $\mathcal{B}$ is the set of bases of a matroid on ground set $E$.
:::

::: proof
*Proof.* By the basis-exchange lemma and the standard characterization of matroid bases [@Welsh1976]. ◻
:::

::: definition
[]{#def:distinguishing-dimension label="def:distinguishing-dimension"} The *distinguishing dimension* of a classification system is the common cardinality of all minimal distinguishing sets.
:::

::: remark
Let $n := |\mathcal{I}|$ be the ambient number of available attributes. Clearly $d \le n$, and there exist worst-case families with $d = n$.
:::

::: corollary
[]{#cor:well-defined-distinguishing-dimension label="cor:well-defined-distinguishing-dimension"} All minimal distinguishing sets have equal cardinality. Thus the distinguishing dimension (Definition [\[def:distinguishing-dimension\]](#def:distinguishing-dimension){reference-type="ref" reference="def:distinguishing-dimension"}) is well-defined.
:::

## Implications for Witness Cost

::: corollary
[]{#cor:attribute-only-witness-lb label="cor:attribute-only-witness-lb"} For any attribute-only observer, $W(\text{class-identity}) \geq d$ where $d$ is the distinguishing dimension.
:::

::: proof
*Proof.* If a procedure queried fewer than $d$ attributes on every execution path, each such queried set would be non-distinguishing by definition of $d$. For that path, there would exist two different classes with identical answers on all queried attributes, yielding identical transcripts and forcing the same output on both values. This contradicts zero-error class identification. Hence some path requires at least $d$ queries. ◻
:::

The key insight: the distinguishing dimension is invariant across all minimal query strategies. The difference between nominal-tag and attribute-only observers lies in *witness cost*: a nominal tag achieves $W = O(1)$ by storing the identity directly, bypassing query enumeration.


## Witness Cost for Class Identity

Recall from Section 2 that the witness cost $W(P)$ is the minimum number of primitive queries required to compute property $P$. For class identity, we ask: what is the minimum number of queries to determine if two values have the same class?

::: theorem
[]{#thm:nominal-minimum-witness-cost label="thm:nominal-minimum-witness-cost"} Nominal-tag observers achieve the minimum witness cost for class identity: $$W_{\text{eq}} = O(1)$$

Specifically, the witness is a single tag read: compare $\text{tag}(v_1) = \text{tag}(v_2)$.

Attribute-only observers require $W_{\text{eq}} = \Omega(d)$ where $d$ is the distinguishing dimension (and $d \le n$, with worst-case $d = n$).
:::

::: proof
*Proof.* See Lean formalization: `proofs/nominal_resolution.lean`. The proof shows:

1.  Nominal-tag access is a single primitive query

2.  Attribute-only observers must query at least $d$ attributes in the worst case (a generic strategy queries all $n$)

3.  No shorter witness exists for attribute-only observers (by the information barrier)

 ◻
:::

## Witness Cost Comparison

  **Observer Class**      **Witness Procedure**      **Witness Cost $W$**
  -------------------- ---------------------------- ----------------------
  Nominal-tag                Single tag read                $O(1)$
  Attribute-only        Query a distinguishing set       $\Omega(d)$

  : Witness cost for class identity by observer class.

The Lean 4 formalization in the supplementary artifact provides a machine-checked proof that nominal-tag access minimizes witness cost for class identity.


## Three-Dimensional Tradeoff: Tag Length, Witness Cost, Distortion

Recall from Section 2 that observer strategies are characterized by three dimensions:

-   **Tag length** $L$: bits required to encode a class identifier ($L \geq \log_2 k$ for $k$ classes under full class tagging)

-   **Witness cost** $W$: minimum number of primitive queries for class identification

-   **Distortion** $D$: probability of misclassification, $D = \Pr[\hat{C} \neq C]$.

We compare two observer classes:

::: definition
An observer that queries only attribute membership ($q_I \in \Phi_{\mathcal{I}}$), with no access to explicit class tags.
:::

::: definition
An observer that may read a single class identifier (nominal tag) per value, in addition to attribute queries.
:::

::: theorem
[]{#thm:lwd-optimal label="thm:lwd-optimal"} Let $A_\pi$ be the collision multiplicity (Definition [\[def:collision-multiplicity\]](#def:collision-multiplicity){reference-type="ref" reference="def:collision-multiplicity"}). Then:

1.  Any $D=0$ scheme must satisfy $L \ge \log_2 A_\pi$ (Theorem [\[thm:converse\]](#thm:converse){reference-type="ref" reference="thm:converse"}).

2.  In maximal-barrier domains ($A_\pi = k$), nominal-tag observers achieve the unique Pareto-optimal $D=0$ point:

    -   **Tag length**: $L = \lceil \log_2 k \rceil$ bits for $k$ classes

    -   **Witness cost**: $W = O(1)$ queries (one tag read)

    -   **Distortion**: $D = 0$ (zero misclassification probability)

3.  In general (non-maximal) domains, nominal tagging remains Pareto-optimal at $D=0$ but need not be unique: partial tags can coexist on the frontier.

In information-barrier domains, attribute-only observers (the $L=0$ face) satisfy:

-   **Tag length**: $L = 0$ bits (no explicit tag)

-   **Witness cost**: $W = \Omega(d)$ queries (must query at least one minimal distinguishing set of size $d$, see Definition [\[def:distinguishing-dimension\]](#def:distinguishing-dimension){reference-type="ref" reference="def:distinguishing-dimension"})

-   **Distortion**: $D > 0$ (probability of misclassification is strictly positive due to collisions)
:::

::: proof
*Proof.* Converse item (1) is Theorem [\[thm:converse\]](#thm:converse){reference-type="ref" reference="thm:converse"}. For item (2), maximal barrier means all classes are observationally colliding, so any $D=0$ scheme must carry full class identity in tag bits (Corollary [\[cor:max-barrier-converse\]](#cor:max-barrier-converse){reference-type="ref" reference="cor:max-barrier-converse"}), while nominal tags realize this lower bound with one tag read. Pareto uniqueness follows because any competing $D=0$ point cannot reduce $L$ below $\log_2 k$ nor reduce $W$ below constant-time tag access under the admissibility rules of Section [\[sec:framework\]](#sec:framework){reference-type="ref" reference="sec:framework"}.

The converse proof path is machine-checked in Lean: `proofs/lwd_converse.lean` formalizes (i) constant transcript on a collision block implies tag injectivity under zero-error decoding, and (ii) counting then yields $2^L \ge A_\pi$. The maximal-barrier corollary is formalized in the same module. Runtime cost instantiations (e.g., unbounded gap examples) remain in `proofs/python_instantiation.lean`. ◻
:::

::: remark
When $1 < A_\pi < k$, the $D=0$ frontier can include mixed designs: a partial tag identifies collision blocks and queries resolve within blocks. This does not contradict nominal optimality; it only removes global uniqueness outside maximal-barrier domains.

1.  Maximal barrier ($A_\pi=k$): unique $D=0$ nominal point.

2.  Intermediate barrier ($1<A_\pi<k$): multiple $D=0$ Pareto points may exist.

3.  No barrier ($A_\pi=1$): $L=0$ zero-error identification is feasible.
:::

## Pareto Frontier

The three-dimensional frontier shows:

-   In maximal-barrier domains, the unique $D=0$ Pareto point is nominal tagging at $L=\lceil \log_2 k\rceil$.

-   In general domains, attribute-only observers trade tag length for distortion on the $L=0$ face when collisions are present.

Figure [1](#fig:lwd-tradeoff){reference-type="ref" reference="fig:lwd-tradeoff"} visualizes the $(L, W, D)$ tradeoff space. The key observation is the ambiguity converse: the minimum zero-error tag rate is $\log_2 A_\pi$, with the maximal-barrier special case $\log_2 k$.

<figure id="fig:lwd-tradeoff">

<figcaption>Schematic <span class="math inline">(<em>L</em>, <em>W</em>, <em>D</em>)</span> tradeoff across barrier regimes. Nominal tagging is the unique zero-error Pareto point in maximal-barrier domains; attribute-only strategies lie on the zero-tag face and leave the zero-error frontier when collisions are present.</figcaption>
</figure>

The Lean 4 formalization in the supplementary artifact machine-checks the full ambiguity-based converse chain and maximal-barrier lower bound that anchor this tradeoff analysis.

::: remark
In one software-runtime instantiation: explicit runtime identifiers correspond to nominal-tag observers (e.g., CPython's `isinstance`, Java's `.getClass()`); attribute-probe schemes correspond to attribute-only observers (e.g., repeated `hasattr`-style checks); and interface-conformance checks are an intermediate case with $D=0$ but $W=O(s)$.
:::

::: remark
When conformance checks traverse $s$ members/fields (rather than ranging over the full attribute universe), the natural bound is $W = O(s)$ with $s \le n$.
:::


This section gives brief illustrations showing the same constraint pattern in established systems.

## Compact Cross-Domain Illustrations

**Biology (phenotype vs genotype).** Distinct species can collide under phenotype, so phenotype-only observation cannot guarantee zero-error identity. DNA barcoding supplies additional naming information that resolves collision blocks in constant-time lookup style [@DNABarcoding].

**Databases (columns vs keys).** Rows may collide on non-key columns, so attribute-only identification is insufficient for guaranteed identity. Primary/surrogate keys realize the nominal-tag role and provide constant-cost identity checks [@Codd1990].

**Software runtimes (attribute probes vs runtime identifiers).** Attribute-probe checks recover identity from observed capabilities and pay query cost that scales with inspected structure. Runtime class/type identifiers behave as explicit tags and collapse identity checks to near-constant cost in the model [@CPythonDocs; @JVMSpec; @JavaDocs; @TypeScriptDocs; @RustDocs].

## Takeaway

These illustrations are not historical causality claims. They show a common resource law: when observable profiles collide, zero-error identification requires additional naming information; otherwise the system pays higher query cost and/or nonzero error.


## Noisy Query Model

Throughout this paper, queries are deterministic: $q_I(v) \in \{0,1\}$ is a fixed function of $v$. In practice, observations may be corrupted. We sketch an extension to noisy queries and state the resulting open problems.

::: definition
A *noisy observation channel* with crossover probability $\epsilon \in [0, 1/2)$ returns: $$\tilde{q}_I(v) = \begin{cases}
q_I(v) & \text{with probability } 1 - \epsilon \\
1 - q_I(v) & \text{with probability } \epsilon
\end{cases}$$ Each query response is an independent BSC$(\epsilon)$ corruption of the true value.
:::

::: definition
The *$\epsilon$-noisy identification capacity* is the supremum rate (in bits per entity) at which zero-error identification is achievable when all attribute queries pass through a BSC$(\epsilon)$.
:::

In the noiseless case ($\epsilon = 0$), Theorem [\[thm:identification-capacity\]](#thm:identification-capacity){reference-type="ref" reference="thm:identification-capacity"} shows the capacity is binary: $\log_2 k$ if $\pi$ is class-injective, $0$ otherwise. For $\epsilon > 0$, several questions arise.

**Open problem (noisy identification cost).** For $\epsilon > 0$ and class-injective $\pi$, zero-error identification is impossible with finite queries (since BSC has nonzero error probability). With bounded error $\delta > 0$, we expect the identification cost to scale as $W = \Theta\left(\frac{\log(1/\delta)}{(1 - 2\epsilon)^2}\right)$ queries per entity. A key observation is that a nominal tag of $L \geq \lceil \log_2 k \rceil$ bits (transmitted noiselessly) should restore $O(1)$ identification regardless of query noise.

The third point is the key insight: *nominal tags provide a noise-free side channel*. Even when attribute observations are corrupted, a clean tag enables $O(1)$ identification. This strengthens the case for explicit tagging in noisy environments, precisely the regime where pure attribute-probe schemes would require many repeated queries to achieve confidence.

**Connection to identification via channels.** The noisy model connects more directly to Ahlswede-Dueck identification [@ahlswede1989identification]. In their framework, identification capacity over a noisy channel can exceed Shannon capacity (double-exponential codebook sizes). Our setting differs: we have *adaptive queries* rather than block codes, and the decoder must identify a *class* rather than test a hypothesis. Characterizing the interplay between adaptive query strategies and channel noise is an open problem.

## Rate-Distortion-Query Tradeoff Surface

The $(L, W, D)$ tradeoff admits a natural geometric interpretation. In the maximal-barrier regime we identify a unique Pareto-optimal point at $D = 0$ (Theorem [\[thm:lwd-optimal\]](#thm:lwd-optimal){reference-type="ref" reference="thm:lwd-optimal"}); outside that regime, the full tradeoff surface can contain multiple $D=0$ frontier points.

**Fixed-$W$ slices.** For fixed query budget $W$, what is the minimum tag rate $L$ to achieve distortion $D$? When $W \geq d$ (the distinguishing dimension), zero distortion is achievable with $L = 0$ via exhaustive querying. When $W < d$, the observer cannot distinguish all classes, and either:

-   Accept $D > 0$ (misidentification), or

-   Add tags ($L > 0$) to compensate for insufficient queries.

**Fixed-$L$ slices.** For fixed tag rate $L < \log_2 k$, the tag partitions the $k$ classes into $2^L$ groups. Within each group, the observer must use queries to distinguish. The query cost is determined by the distinguishing dimension *within each group*, which is potentially much smaller than the global dimension.

**Open problem (subadditivity of query cost).** For a tag of rate $L$ partitioning classes into groups $G_1, \ldots, G_{2^L}$, we expect $W(L) \leq \max_i d(G_i)$, where $d(G_i)$ is the distinguishing dimension within group $G_i$. Optimal tag design should minimize this maximum. Characterizing the optimal partition remains open.

## Semantic Distortion Measures

We have treated distortion $D$ as binary (correct identification or not). Richer distortion measures are possible:

-   **Hierarchical distortion**: Misidentifying a class within the same genus (biological) or within the same subsystem (engineered systems) is less severe than cross-group errors.

-   **Weighted distortion**: Some misidentifications have higher cost than others (e.g., safety-critical misrouting vs. benign confusion).

## Privacy and Security

**Privacy-preserving identification.** Nominal tags enable zero-knowledge proofs of class membership without revealing attribute profiles. An entity can prove \"I belong to class $C$\" by revealing $\tau(v) = C$ without exposing $\pi(v)$, preserving attribute privacy. Attribute-only schemes must reveal the complete profile $\pi(v)$ to prove membership, leaking structural information.

**Secure model verification.** In machine learning deployment, compressed model identifiers prevent model substitution attacks. Verifying model identity via nominal tags ($O(1)$ hash comparison) is more efficient and secure than attribute-based verification ($O(s)$ architecture inspection), which is vulnerable to adversarial perturbations that preserve structural fingerprints while altering behavior.

## Connection to Rate-Distortion-Perception Theory

Blau and Michaeli [@blau2019rethinking] extended classical rate-distortion theory by adding a *perception* constraint: the reconstructed distribution must match a target distribution under some divergence measure. This creates a three-way tradeoff between rate, distortion, and perceptual quality.

Our $(L, W, D)$ framework admits a parallel interpretation. The query cost $W$ plays a role analogous to the perception constraint: it measures the *interactive cost* of achieving low distortion, rather than a distributional constraint. Just as rate-distortion-perception theory asks "what is the minimum rate to achieve distortion $D$ while satisfying perception constraint $P$?", we ask "what is the minimum tag rate $L$ to achieve distortion $D$ with query budget $W$?"

The analogy suggests several directions:

-   **Perception as identification fidelity**: In classification systems that must preserve statistical properties (e.g., sampling from a type distribution), a perception constraint would require the observer's class estimates to have the correct marginal distribution, not just low expected error.

-   **Three-resource tradeoffs**: The $(L, W, D)$ Pareto frontier (Theorem [\[thm:lwd-optimal\]](#thm:lwd-optimal){reference-type="ref" reference="thm:lwd-optimal"}) is a discrete analogue of the rate-distortion-perception tradeoff surface. Characterizing this surface for specific classification problems would extend the geometric rate-distortion program to identification settings.

Formalizing these connections would unify identification capacity with the broader rate-distortion-perception literature.


This paper presents a resource-based analysis of classification under observational constraints. We establish three core theorem families and accompanying extensions:

1.  **Information Barrier**: Observers limited to attribute-membership queries cannot compute properties that vary within indistinguishability classes. This is the structural obstruction from which the later lower bounds and tradeoffs follow.

2.  **Witness Optimality**: Nominal-tag observers achieve $W(\text{identity}) = O(1)$, the minimum witness cost. The gap from attribute-only observation ($\Omega(d)$, with a worst-case family where $d = n$) is unbounded.

3.  **Matroid Structure**: Minimal distinguishing query sets form the bases of a matroid. The distinguishing dimension of a classification problem is well-defined and computable.

## A Recurring Design Constraint

Across domains, the same structure recurs:

-   **Biology**: Phenotypic observation cannot distinguish cryptic species. DNA barcoding (nominal tag) resolves them in $O(1)$.

-   **Databases**: Column-value queries cannot distinguish rows with identical attributes. Primary keys (nominal tag) provide $O(1)$ identity.

-   **Software runtimes**: Attribute observation cannot distinguish entities that collide on observed capabilities. Runtime identifiers provide $O(1)$ identity.

The information barrier is not a quirk of any particular domain; it is a mathematical necessity arising from the quotient structure induced by limited observations. What varies by domain is not the existence of the constraint, but which resource the system chooses to spend to escape it.

## Implications

-   **The necessity of nominal information is a theorem, not a preference.** Any zero-error scheme must satisfy the ambiguity converse $L \ge \log_2 A_\pi$ (Theorem [\[thm:converse\]](#thm:converse){reference-type="ref" reference="thm:converse"}), where $A_\pi$ is the largest collision block induced by observable profiles. In maximal-barrier domains ($A_\pi = k$), this becomes $L \ge \log_2 k$ and nominal tagging gives the unique $D=0$ Pareto point with $W=O(1)$ (Theorem [\[thm:lwd-optimal\]](#thm:lwd-optimal){reference-type="ref" reference="thm:lwd-optimal"}).

-   **The barrier is informational, not computational**: even with unbounded resources, attribute-only observers cannot overcome it.

-   **Finite-block confusability laws are exact**: the mechanized graph-confusability results give one-shot and block thresholds ($A_\pi$ and $A_\pi^k$) and exact linear log-bit scaling across block length.

-   **Open-world guarantees require explicit tags**: barrier-freedom is not extension-stable under system growth, and deciding barrier existence/freedom for arbitrary generators is not computable (Rice-style). So a persistent $D=0$ guarantee cannot rely on "currently collision-free" attribute structure alone.

-   **Fixed-axis systems are necessarily incomplete outside their axis**: by Corollary [\[cor:fixed-axis-incompleteness\]](#cor:fixed-axis-incompleteness){reference-type="ref" reference="cor:fixed-axis-incompleteness"}, any fixed-axis classifier is complete only for axis-measurable properties and cannot represent properties that vary within an axis fiber unless a new axis or explicit tag is introduced.

-   **Classification system design is constrained**: the choice of observation family determines which properties are computable.

## Future Work

Natural extensions include other classification domains, witness complexity for properties beyond identity, and hybrid observers that combine tags and attributes under non-uniform query distributions.

## Conclusion

Classification under observational constraints admits a clean quantitative analysis. The zero-error converse is governed by collision multiplicity: any $D=0$ scheme necessarily has $L \ge \log_2 A_\pi$ (Theorem [\[thm:converse\]](#thm:converse){reference-type="ref" reference="thm:converse"}). In maximal-barrier domains ($A_\pi=k$), nominal-tag observation achieves the unique Pareto-optimal $D=0$ point in the $(L, W, D)$ tradeoff (Theorem [\[thm:lwd-optimal\]](#thm:lwd-optimal){reference-type="ref" reference="thm:lwd-optimal"}). The mechanized graph-confusability extension further supplies exact finite-block thresholds and rate scaling. Within the stated observation model, the paper turns a familiar heuristic into explicit lower bounds, explicit dominance claims, and structural invariants, and all proofs are machine-verified in Lean 4.

## Acknowledgment: AI-use Disclosure {#acknowledgment-ai-use-disclosure .unnumbered}

Generative AI tools (including Codex, Claude Code, Augment, Kilo, and OpenCode) were used throughout this manuscript, across all sections (Abstract, Introduction, theoretical development, proof sketches, applications, conclusion, and appendix) and across all stages from initial drafting to final revision. The tools were used for boilerplate generation, prose and notation refinement, LaTeX/structure cleanup, translation of informal proof ideas into candidate formal artifacts (Lean/LaTeX), and repeated adversarial reviewer-style critique passes to identify blind spots and clarity gaps.

The author retained full intellectual and editorial control, including problem selection, theorem statements, assumptions, novelty framing, acceptance criteria, and final inclusion/exclusion decisions. No technical claim was accepted solely from AI output. Formal claims reported as machine-verified were admitted only after Lean verification (no `sorry` in cited modules) and direct author review; Lean was used as an integrity gate for responsible AI-assisted research. The author is solely responsible for all statements, citations, and conclusions.




---

## Machine-Checked Proofs

All theorems are formalized in Lean 4:
- Location: `docs/papers/paper1_typing_discipline/proofs/`
- Lines: 8092
- Theorems: 352
- `sorry` placeholders: 0
