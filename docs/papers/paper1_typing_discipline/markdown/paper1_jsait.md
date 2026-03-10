# Paper: Zero-Error Semantic Compression and Rate-Query Tradeoffs in Classification Systems

**Status**: JSAIT-ready | **Lean**: 8092 lines, 352 theorems

---

## Abstract

_Abstract not available._

## The Identification Problem

Consider an encoder-decoder pair communicating about entities from a large universe $\mathcal{V}$. The paper studies a worst-case semantic compression problem in which the decoder must identify each entity, determining which of $k$ classes it belongs to, using only:

-   a *tag* of $L$ bits stored with the entity, and/or

-   *queries* to a binary oracle: "does entity $v$ satisfy attribute $I$?"

This is not reconstruction (the decoder need not recover $v$), but identification with task-relevant semantic accuracy. The resulting tradeoff is among stored naming information, interactive query effort, and semantic error under constrained observation.

The observational bottleneck is the attribute-profile map $\pi: \mathcal{V} \to \{0,1\}^n$. Once distinct classes collide under $\pi$, attribute-only observation cannot distinguish them with zero error. The paper turns that obstruction into an exact compression law. Let $$A_\pi := \max_u |\{c : \pi(c)=u\}|$$ be the largest collision block. Then every zero-error scheme must satisfy $$L \ge \log_2 A_\pi,$$ and this lower bound is tight. The same law extends exactly to finite blocks: the minimum block tag budget is $L_t^\star = t\log_2 A_\pi$. These are the paper's main compression-facing statements.

We then develop a second theorem arc for the query resource. Without tags ($L=0$), zero-error identification depends on the structure of distinguishing query families. Minimal sufficient query sets form the bases of a matroid, so the distinguishing dimension $d$ is well-defined as an invariant of the observation structure rather than of a particular query strategy. Every tag-free zero-error scheme therefore has worst-case query cost at least $d$.

These laws are domain-independent within the stated model: the same lower bounds appear across databases, biological taxonomy, knowledge graphs, model registries, and software runtimes. Once classes collide under observable attributes, constant-cost zero-error identification requires auxiliary naming information. In maximal-barrier domains this yields a unique $D=0$ Pareto solution, so the issue is resource feasibility rather than domain-specific style conventions.

The present model can be read as a zero-error semantics-aware compression problem, with direct structural connections to rate-distortion-style tradeoffs and natural privacy/security extensions.

## Main Results

We organize the paper around one primary theorem arc and one secondary consequences arc.

1.  **Zero-error ambiguity converse and exact scaling.** Any $D=0$ scheme must satisfy $L \ge \log_2 A_\pi$ (Theorem [\[thm:converse\]](#thm:converse){reference-type="ref" reference="thm:converse"}), and the finite-block confusability law gives exact scaling $L_t^\star = t\log_2 A_\pi$ (Corollary [\[cor:graph-logbit-scaling\]](#cor:graph-logbit-scaling){reference-type="ref" reference="cor:graph-logbit-scaling"}). In maximal-barrier domains ($A_\pi=k$), the nominal point $(\lceil\log_2 k\rceil,O(1),0)$ is the unique Pareto-optimal zero-error solution (Theorem [\[thm:lwd-optimal\]](#thm:lwd-optimal){reference-type="ref" reference="thm:lwd-optimal"}).

2.  **Query lower bound and matroid structure.** Minimal distinguishing query sets form the bases of a matroid (Theorem [\[thm:matroid-bases\]](#thm:matroid-bases){reference-type="ref" reference="thm:matroid-bases"}). Their common cardinality $d$ is the distinguishing dimension (Definition [\[def:distinguishing-dimension\]](#def:distinguishing-dimension){reference-type="ref" reference="def:distinguishing-dimension"}), and any attribute-only zero-error scheme has worst-case query cost at least $d$ (Theorem [\[thm:interface-lower-bound\]](#thm:interface-lower-bound){reference-type="ref" reference="thm:interface-lower-bound"}). The matroid theorem is what makes $d$ strategy-independent rather than a proof artifact.

3.  **Foundational observational impossibility.** The information barrier theorem (Theorem [\[thm:information-barrier\]](#thm:information-barrier){reference-type="ref" reference="thm:information-barrier"}) shows that attribute-only observers cannot compute properties that vary within observational equivalence classes. This is the base invariance statement from which the converse, the Pareto characterization, and the query lower bounds follow.

4.  **Open-world robustness and decidability boundary.** Barrier-freedom is not extension-stable in open worlds (Theorem [\[thm:open-world-extension-instability\]](#thm:open-world-extension-instability){reference-type="ref" reference="thm:open-world-extension-instability"}), and for arbitrary generators it is not computably certifiable in advance (Theorem [\[thm:rice-barrier\]](#thm:rice-barrier){reference-type="ref" reference="thm:rice-barrier"}). A persistent zero-error guarantee therefore cannot rely on present-day collision-freedom alone.

5.  **Implementation-level consequences.** When identity is carried explicitly, error localization and constraint centralization collapse to $O(1)$ sites; when systems rely on distributed attribute probes, localization and information scattering scale with the number of probe sites (Section [\[sec:applications\]](#sec:applications){reference-type="ref" reference="sec:applications"}).

6.  **Machine-checked proofs.** The claims are accompanied by a supplementary Lean 4 artifact (8092 lines, 352 theorem/lemma statements, 0 `sorry`). The mechanization supports the mathematics rather than acting as a separate theorem-family contribution.

Without the equicardinality result behind the matroid structure, the lower bound would depend on which distinguishing set or query strategy one chose. The matroid theorem is what makes $d$ strategy-independent.

## Related Work and Positioning

We study a worst-case zero-error semantic compression problem under a fixed observable family, derive an exact ambiguity converse and finite-block law, and characterize the complementary query resource via a matroid invariant.

**Identification via channels.** Ahlswede and Dueck [@ahlswede1989identification; @ahlswede1989identification2] study identification over channels, where the decoder answers a hypothesis-testing question rather than reconstructing a message. Our setting identifies semantic classes under constrained observations, allows adaptive queries, and trades explicit tag bits against query cost.

**Rate-distortion theory.** The $(L, W, D)$ framework is rate-distortion-adjacent in the following sense: $L$ stores explicit naming information, $W$ measures interactive disambiguation effort, and $D$ measures semantic failure through class misidentification [@shannon1959coding; @berger1971rate]. Theorem [\[thm:lwd-optimal\]](#thm:lwd-optimal){reference-type="ref" reference="thm:lwd-optimal"} identifies the resulting zero-error frontier in terms of collision multiplicity and distinguishing dimension.

**Rate-distortion-perception tradeoffs.** Blau and Michaeli [@blau2019rethinking] extend rate-distortion theory with a third resource. Our query cost $W$ measures interactive effort in the same three-resource spirit.

**Zero-error information theory.** Körner [@korner1973coding] and Witsenhausen [@witsenhausen1976zero] study coding with confusability constraints. Here the central invariant is collision multiplicity $A_\pi$, which yields an explicit ambiguity converse and exact finite-block scaling.

**Query complexity and communication complexity.** The $\Omega(d)$ lower bound for attribute-only identification relates to decision tree complexity [@buhrman2002complexity] and interactive communication [@orlitsky1991worst]. Here the query resource is the amount of interactive evidence needed once explicit naming information is absent.

**Semantic compression settings.** The same ambiguity bound appears across databases, knowledge graphs, taxonomy, and software-runtime typing systems: for zero-error identification, ambiguity induces a minimum metadata requirement $L \ge \log_2 A_\pi$ (Theorem [\[thm:converse\]](#thm:converse){reference-type="ref" reference="thm:converse"}), with maximal-barrier specialization $L \ge \log_2 k$.

**Secondary instantiation.** In nominal-vs-structural interface settings [@Cardelli1985; @cook1990inheritance], the model yields a concrete cost statement: under attribute collisions, purely profile-based identification has worst-case $\Omega(d)$ witness cost, while explicit tags achieve $O(1)$ identification using $O(\log A_\pi)$ bits. This is one instantiation of the general bounds.

## Paper Organization

Section [\[sec:model\]](#sec:model){reference-type="ref" reference="sec:model"} formalizes the observation model and the foundational information barrier. Section [\[sec:zero-error\]](#sec:zero-error){reference-type="ref" reference="sec:zero-error"} develops one-shot zero-error feasibility, nominal tagging, the ambiguity-based converse, and the exact finite-block law. Section [\[sec:matroid\]](#sec:matroid){reference-type="ref" reference="sec:matroid"} proves the matroid structure of distinguishing query families. Section [\[sec:lwd\]](#sec:lwd){reference-type="ref" reference="sec:lwd"} gives the $(L, W, D)$ frontier. Section [\[sec:robustness\]](#sec:robustness){reference-type="ref" reference="sec:robustness"} studies open-world robustness and the decidability boundary. Section [\[sec:applications\]](#sec:applications){reference-type="ref" reference="sec:applications"} collects practical consequences and cross-domain illustrations. Section [\[sec:extensions\]](#sec:extensions){reference-type="ref" reference="sec:extensions"} sketches extensions and open directions. Section [\[sec:conclusion\]](#sec:conclusion){reference-type="ref" reference="sec:conclusion"} concludes.


## Primitive Objects

The introduction has already motivated the semantic-compression viewpoint. This section fixes the observation model once, with only the primitives and assumptions needed for the theorem chain.

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
We write $n := |\mathcal{I}|$ for the ambient number of available attributes. We write $d$ for the distinguishing dimension (the common size of all minimal distinguishing query sets; Definition [\[def:distinguishing-dimension\]](#def:distinguishing-dimension){reference-type="ref" reference="def:distinguishing-dimension"}), so $d \le n$ and there exist worst-case families with $d = n$. We write $m$ for the number of *query sites* (call sites) that perform attribute checks in a program or protocol (used only in the complexity-of-maintenance discussion). When discussing a particular identification or verification task, we may write $s$ for the number of attributes actually queried or traversed by the procedure, with $s \le n$.
:::

::: definition
The attribute profile function $\pi: \mathcal{V} \to \{0,1\}^{|\mathcal{I}|}$ maps each value to its complete attribute signature: $$\pi(v) = (q_I(v))_{I \in \mathcal{I}}.$$
:::

::: definition
Values $v, w \in \mathcal{V}$ are *attribute-indistinguishable*, written $v \sim w$, iff $\pi(v) = \pi(w)$.
:::

The relation $\sim$ is an equivalence relation. We write $[v]_\sim$ for the equivalence class of $v$.

::: definition
A *classification scheme* is any procedure (deterministic or randomized), with arbitrary time and memory, whose only access to a value $v \in \mathcal{V}$ is via:

1.  the observation family $\Phi = \{q_I : I \in \mathcal{I}\}$, where $q_I(v) = 1$ iff $v$ satisfies attribute $I$, and optionally

2.  a nominal-tag primitive $\tau: \mathcal{V} \to \mathcal{T}$ returning an opaque class identifier.

All theorems in this paper quantify over all such schemes.
:::

This definition is intentionally broad: schemes may be adaptive, randomized, or computationally unbounded. The constraint is *observational*, not computational.

::: definition
An *attribute-only observer* is any procedure whose interaction with a value $v \in \mathcal{V}$ is limited to queries in $\Phi_{\mathcal{I}}$. Formally, the observer interacts with $v$ only via primitive attribute queries $q_I \in \Phi_{\mathcal{I}}$; hence any transcript (and output) factors through $\pi(v)$.
:::

::: definition
[]{#def:admissibility-contract label="def:admissibility-contract"} All impossibility and optimality statements in this paper quantify over schemes whose evidence consists only of the declared observation family $\Phi$ together with any explicit nominal tag charged to the tag budget $L$. In particular, no external registry, reflection oracle, whole-universe preprocessing table, or amortized cross-instance cache may supply hidden distinguishing information.
:::

::: proposition
[]{#prop:model-capture label="prop:model-capture"} Any classification protocol whose evidence consists only of the declared observation family and any explicit nominal tag is representable in this model. Any additional capability changes the model by enlarging the observable family or by introducing auxiliary information that must be charged explicitly.
:::

The core results require only $(\mathcal V,C,\pi)$ and the observation family $\Phi$. The fixed-axis language below is a representative specialization rather than an added axiom for the general theory.

## Attribute-Computable Properties

The central question is: **what semantic properties can an attribute-only observer compute?**

A semantic property is a function $P: \mathcal{V} \to \{0,1\}$ (or more generally, $P: \mathcal{V} \to Y$ for some codomain $Y$). We say $P$ is *attribute-computable* if there exists a function $f: \{0,1\}^{|\mathcal{I}|} \to Y$ such that $P(v) = f(\pi(v))$ for all $v$.

## The Information Barrier

::: theorem
[]{#thm:information-barrier label="thm:information-barrier"}[]{#thm:info-barrier label="thm:info-barrier"} Let $P: \mathcal{V} \to Y$ be any function. If $P$ is attribute-computable, then $P$ is constant on $\sim$-equivalence classes: $$v \sim w \implies P(v) = P(w).$$ Equivalently: no attribute-only observer can compute any property that varies within an equivalence class.
:::

::: proof
*Proof.* Suppose $P$ is attribute-computable via $f$, i.e., $P(v) = f(\pi(v))$ for all $v$. Let $v \sim w$, so $\pi(v) = \pi(w)$. Then: $$P(v) = f(\pi(v)) = f(\pi(w)) = P(w).$$ ◻
:::

::: remark
The barrier is *informational*, not computational. Given unlimited time, memory, and computational power, an attribute-only observer still cannot distinguish $v$ from $w$ when $\pi(v) = \pi(w)$. The constraint is on the evidence itself.
:::

::: remark
Theorem [\[thm:information-barrier\]](#thm:information-barrier){reference-type="ref" reference="thm:information-barrier"} is the base invariance statement. The technical contribution is the downstream structure built on top of it: the ambiguity-based converse (Theorem [\[thm:converse\]](#thm:converse){reference-type="ref" reference="thm:converse"}), the Pareto characterization (Theorem [\[thm:lwd-optimal\]](#thm:lwd-optimal){reference-type="ref" reference="thm:lwd-optimal"}), and the matroid or equicardinality results (Section [\[sec:matroid\]](#sec:matroid){reference-type="ref" reference="sec:matroid"}).
:::

::: corollary
[]{#cor:provenance-barrier label="cor:provenance-barrier"} Let $C: \mathcal{V} \to \{1,\ldots,k\}$ be the class assignment function. If there exist values $v, w$ with $\pi(v) = \pi(w)$ but $C(v) \neq C(w)$, then class identity is not attribute-computable.
:::

::: proof
*Proof.* Direct application of Theorem [\[thm:information-barrier\]](#thm:information-barrier){reference-type="ref" reference="thm:information-barrier"} to $P = C$. ◻
:::

::: remark
The relation $\sim$ partitions $\mathcal{V}$ into observational equivalence classes, so an attribute-only observer operates on the quotient $\mathcal{V}/{\sim}$ rather than on raw entities. Any strategy that distinguishes values within one $\sim$-class must import new information beyond the declared observable family.
:::

::: remark
A representative specialization writes the observable state as $\alpha(v)=(B(v),S(v))$, where $S(v)=\pi(v)$ is the declared profile and $B(v)$ is any explicit provenance carrier. In that viewpoint, admissible semantic properties factor through $\alpha$, and properties that vary within an $\alpha$-fiber require adding new information rather than better inference alone.
:::

::: theorem
[]{#thm:model-completeness label="thm:model-completeness"} Let $\alpha:\mathcal V\to\mathcal A$ be an axis map such that every admissible primitive query factors through $\alpha$. Then every in-scope semantic property factors through $\alpha$: there exists $\tilde P$ with $$P(v)=\tilde P(\alpha(v)) \quad \text{for all } v\in\mathcal V.$$
:::

::: proof
*Proof.* An admissible strategy observes $v$ only through responses to primitive queries in $\Phi$. If each primitive query factors through $\alpha$, then the entire transcript factors through $\alpha$, and so does any output computed from that transcript. ◻
:::

::: corollary
[]{#cor:fixed-axis-incompleteness label="cor:fixed-axis-incompleteness"} If $\alpha(v)=\alpha(w)$ but $P(v)\neq P(w)$, then no admissible strategy can compute $P$ with zero error without adding new information beyond the declared observable state.
:::

::: proposition
[]{#prop:observational-quotient label="prop:observational-quotient"} For any admissible strategy using only the declared observable family, the entire interaction transcript and hence the output depend only on the observable state. Equivalently, every in-scope semantic property factors through that state.
:::

::: corollary
[]{#cor:ad-hoc-add-axis label="cor:ad-hoc-add-axis"} If two values agree on the declared observable state, then no admissible strategy can distinguish them with zero error. Any mechanism that does distinguish such pairs introduces additional information that should be represented explicitly in the model.
:::


## Running Example

We use one small example throughout the paper. Let the class set be $$\mathcal C = \{A,B,C,D,E\}$$ with three observable attributes and class profiles

::: center
    class     $A$     $B$     $C$     $D$     $E$
  --------- ------- ------- ------- ------- -------
   profile   $000$   $000$   $100$   $110$   $111$
:::

Then $A$ and $B$ form a nontrivial collision block, so $A_\pi = 2$. One minimal distinguishing family is $\{q_1,q_2,q_3\}$, so the example also witnesses distinguishing dimension $d=3$. We will use the same example to instantiate the ambiguity converse, the exact block law, the query lower bound, and the open-world extension failure.

## One-Shot Zero-Error Feasibility

We now formalize the identification problem in channel-theoretic terms. Let $C: \mathcal{V} \to \{1, \ldots, k\}$ denote the class assignment function, and let $\pi: \mathcal{V} \to \{0,1\}^n$ denote the attribute profile.

::: definition
The *observation channel* induced by observation family $\Phi$ is the mapping $C \to \pi(V)$ for a random entity $V$ drawn from distribution $P_V$ over $\mathcal{V}$. The channel output is the attribute profile; the channel input is implicitly the class $C(V)$.
:::

::: theorem
[]{#thm:identification-capacity label="thm:identification-capacity"} Let $\mathcal{C} = \{1, \ldots, k\}$ be the class space. Zero-error identification of all $k$ classes is achievable if and only if every class has a distinct attribute profile. When $\pi$ is not class-injective, no zero-error identification scheme exists without auxiliary tag information.
:::

::: proof
*Proof.* *Achievability*: If $\pi$ is injective on classes, then observing $\pi(v)$ determines $C(v)$ uniquely. The decoder simply inverts the class-to-profile mapping.

*Converse*: Suppose two distinct classes $c_1 \neq c_2$ share a profile: there exist $v_1 \in c_1, v_2 \in c_2$ with $\pi(v_1) = \pi(v_2)$. Then any decoder $g(\pi(v))$ outputs the same class label on both $v_1$ and $v_2$, so it cannot be correct for both. Hence zero-error identification of all classes is impossible. ◻
:::

::: remark
Under any distribution with positive mass on both colliding classes, $I(C; \pi(V)) < H(C)$. This is an average-case consequence of the deterministic barrier above.
:::

::: remark
In the identification paradigm of [@ahlswede1989identification], the decoder asks "is the message $m$?" rather than "what is the message?" This yields double-exponential codebook sizes. Our setting is different: we require zero-error identification of the *class*, not hypothesis testing. The one-shot zero-error identification feasibility threshold is binary rather than asymptotic.
:::

## The Positive Result: Nominal Tagging

We now show that augmenting attribute observations with a single primitive, nominal-tag access, achieves constant witness cost.

::: definition
A *nominal tag* is a value $\tau(v) \in \mathcal{T}$ associated with each $v \in \mathcal{V}$, representing the class identity of $v$. The nominal-tag access operation returns $\tau(v)$ in $O(1)$ time.
:::

::: definition
The extended primitive query set is $\Phi_{\mathcal{I}}^+ = \Phi_{\mathcal{I}} \cup \{\tau\}$, where $\tau$ denotes nominal-tag access.
:::

::: theorem
[]{#thm:tag-restored-sufficiency label="thm:tag-restored-sufficiency"} A class-injective nominal tag of length $L \geq \lceil \log_2 k \rceil$ bits suffices for zero-error identification, regardless of whether $\pi$ is class-injective.
:::

::: proof
*Proof.* A nominal tag $\tau: \mathcal{V} \to \{1, \ldots, k\}$ assigns a unique identifier to each class. Reading $\tau(v)$ determines $C(v)$ in $O(1)$ queries, independent of the attribute channel. ◻
:::

::: definition
Let $W(P)$ denote the minimum number of primitive queries from $\Phi_{\mathcal{I}}^+$ required to compute property $P$. We distinguish two tasks:

-   $W_{\text{id}}$: Cost to identify the class of a single entity.

-   $W_{\text{eq}}$: Cost to determine if two entities have the same class.

Unless specified, $W$ refers to worst-case identification cost.
:::

::: theorem
[]{#thm:constant-witness label="thm:constant-witness"}[]{#thm:tag-restored label="thm:tag-restored"} Under nominal-tag access, class identity checking has constant witness cost: $$W(\text{class-identity}) = O(1).$$ Specifically, the witness procedure is to compare the relevant tag values.
:::

::: proof
*Proof.* The procedure makes a constant number of primitive queries (one tag access per value) and one comparison. This is $O(1)$ regardless of the number of available attributes $|\mathcal{I}|$. ◻
:::

## Ambiguity Converse

The next theorem is the paper's main zero-error compression statement. It converts observational collision multiplicity directly into a necessary naming budget.

::: definition
[]{#def:collision-multiplicity label="def:collision-multiplicity"} Let $\mathcal{C}=\{1,\ldots,k\}$ and let $\pi_{\mathcal{C}}:\mathcal{C}\to\{0,1\}^n$ be the class-level profile map. Define $$A_\pi := \max_{u \in \operatorname{Im}(\pi_{\mathcal{C}})} \left|\{c \in \mathcal{C} : \pi_{\mathcal{C}}(c)=u\}\right|.$$ Thus $A_\pi$ is the size of the largest class-collision block under observable profiles.
:::

::: remark
The converse is parameterized by $A_\pi$. When the class-profile map is given explicitly as a finite table, $A_\pi$ is obtained by grouping classes by profile and taking the largest fiber size. When $\pi$ is given implicitly by a large program, circuit, or learned model, computing or approximating $A_\pi$ becomes a separate representation-dependent problem. The theorem identifies the exact zero-error tag budget once $A_\pi$ is known; it does not assume that estimating $A_\pi$ is always tractable from an implicit description.
:::

::: definition
[]{#def:max-barrier label="def:max-barrier"} The domain is *maximal-barrier* if $A_\pi = k$, i.e., all classes collide under the observation map.
:::

::: theorem
[]{#thm:converse label="thm:converse"} For any classification domain, any scheme achieving $D=0$ requires $$2^L \ge A_\pi
\quad\text{equivalently}\quad
L \ge \log_2 A_\pi,$$ where $A_\pi$ is the collision multiplicity.
:::

::: proof
*Proof.* Fix a collision block $G \subseteq \mathcal{C}$ with $|G|=A_\pi$ and identical observable profile. For classes in $G$, query transcripts are identical, so zero-error decoding must separate those classes using tag outcomes. With $L$ tag bits there are at most $2^L$ outcomes, hence $2^L \ge |G| = A_\pi$. ◻
:::

::: corollary
[]{#cor:max-barrier-converse label="cor:max-barrier-converse"} If the domain is maximal-barrier ($A_\pi = k$), any zero-error scheme satisfies $L \ge \log_2 k$.
:::

In the running example, the collision block $\{A,B\}$ shows that $A_\pi = 2$, so Theorem [\[thm:converse\]](#thm:converse){reference-type="ref" reference="thm:converse"} already forces $L \ge 1$ bit for zero-error identification.

## Exact Finite-Block Law

The ambiguity converse extends exactly to blocks, so the main compression law is not merely one-shot. The next theorem family gives exact finite-block scaling rather than an asymptotic approximation.

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

In the running example, $A_\pi = 2$, so the exact block law gives $L_t^\star = t$ bits.

## Witness Cost for Class Identity

Recall from the model setup that the witness cost $W(P)$ is the minimum number of primitive queries required to compute property $P$. For class identity, we ask: what is the minimum number of queries to determine if two values have the same class?

::: theorem
[]{#thm:nominal-minimum-witness-cost label="thm:nominal-minimum-witness-cost"} Nominal-tag observers achieve the minimum witness cost for class identity: $$W_{\text{eq}} = O(1).$$

Specifically, the witness is a single tag comparison. Attribute-only observers require $W_{\text{eq}} = \Omega(d)$ where $d$ is the distinguishing dimension (and $d \le n$, with worst-case $d = n$).
:::

::: proof
*Proof.*

1.  Nominal-tag access is a single primitive query.

2.  Attribute-only observers must query at least $d$ attributes in the worst case (a generic strategy queries all $n$).

3.  No shorter witness exists for attribute-only observers by the information barrier and the distinguishing-dimension lower bound.

 ◻
:::

The supplementary artifact provides a machine-checked proof that nominal-tag access minimizes witness cost for class identity.


## Query Families and Distinguishing Sets

We now study the query resource under the fixed observation family $\Phi$. A distinguishing set separates every pair of distinct classes. Minimal distinguishing sets will be shown to form the bases of a matroid.

The running example already illustrates the basic contrast: one explicit bit resolves the only collision block, but tag-free zero-error separation still has to pay for the full distinguishing family. This section identifies the structural invariant that controls that query cost.

::: definition
Let $\mathcal{Q}$ be the set of all primitive queries available to an observer. For a classification system with attribute set $\mathcal{I}$, we have $\mathcal{Q} = \{q_I : I \in \mathcal{I}\}$ where $q_I(v) = 1$ iff $v$ satisfies attribute $I$.
:::

In this section, "queries" are the primitive attribute predicates $q \in \Phi$.

**Convention:** $\Phi := \mathcal{Q}$. All universal quantification over "queries" ranges over $q \in \Phi$ only.

::: definition
A subset $S \subseteq \mathcal{Q}$ is *distinguishing* if, for all values $v, w$ with $\mathrm{class}(v) \neq \mathrm{class}(w)$, there exists $q \in S$ such that $q(v) \neq q(w)$.
:::

::: definition
A distinguishing set $S$ is *minimal* if no proper subset of $S$ is distinguishing.
:::

In the running example from Section [\[sec:zero-error\]](#sec:zero-error){reference-type="ref" reference="sec:zero-error"}, $\{q_1,q_2,q_3\}$ is distinguishing and minimal, so it already witnesses the query resource needed for tag-free separation.

## Matroid Structure of Query Families

::: definition
Let $E = \Phi\;(=\mathcal{Q})$ be the ground set of primitive queries. Let $\mathcal{B} \subseteq 2^E$ be the family of minimal distinguishing sets.
:::

::: lemma
[]{#lem:basis-exchange label="lem:basis-exchange"} For any $B_1, B_2 \in \mathcal{B}$ and any $q \in B_1 \setminus B_2$, there exists $q' \in B_2 \setminus B_1$ such that $(B_1 \setminus \{q\}) \cup \{q'\} \in \mathcal{B}$.
:::

::: proof
*Proof.* Define the closure operator $\mathrm{cl}(X) = \{q : X\text{-equivalence implies }q\text{-equivalence}\}$. We verify the matroid axioms:

1.  **Closure axioms**: $\mathrm{cl}$ is extensive, monotone, and idempotent. These follow directly from the definition of logical implication.

2.  **Exchange property**: If $q \in \mathrm{cl}(X \cup \{q'\}) \setminus \mathrm{cl}(X)$, then $q' \in \mathrm{cl}(X \cup \{q\})$.

For exchange, take $q \in \mathrm{cl}(X \cup \{q'\}) \setminus \mathrm{cl}(X)$. Since $q \notin \mathrm{cl}(X)$, there exist $v,w$ that are $X$-equivalent but disagree on $q$. Because $q \in \mathrm{cl}(X \cup \{q'\})$, any pair that is $(X \cup \{q'\})$-equivalent must agree on $q$; therefore this witness pair cannot be $(X \cup \{q'\})$-equivalent, so it must disagree on $q'$. Now fix any pair $v',w'$ that are $(X \cup \{q\})$-equivalent. They are in particular $X$-equivalent and agree on $q$. If they disagreed on $q'$, then by the previous implication we could derive disagreement on $q$, contradiction. Hence $v',w'$ agree on $q'$, proving $q' \in \mathrm{cl}(X \cup \{q\})$.

Minimal distinguishing sets are exactly the bases of the matroid defined by this closure operator. The machine-checked development verifies the closure laws, the exchange and equicardinality lemmas, and the bridge from closure to matroid structure used to make the distinguishing dimension well-defined. ◻
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
[]{#cor:well-defined-distinguishing-dimension label="cor:well-defined-distinguishing-dimension"} All minimal distinguishing sets have equal cardinality. Thus the distinguishing dimension is well-defined.
:::

::: remark
Despite the name, the distinguishing dimension $d$ is not a VC or shattering dimension in the usual learning-theoretic sense [@vapnik1998statistical]. VC dimension measures the largest set of examples shattered by a concept class. Here $d$ is the rank of a fixed primitive query family under the closure operator induced by class separation. The key structural fact used in this paper is basis exchange: minimal distinguishing families form the bases of a matroid, so $d$ is certified by equicardinality and exchange rather than by shattering alone.
:::

## Implications for Witness Cost

::: theorem
[]{#thm:interface-lower-bound label="thm:interface-lower-bound"} For attribute-only observers, class identity checking requires $$W(\text{class-identity}) = \Omega(d)$$ in the worst case, where $d$ is the distinguishing dimension.
:::

::: proof
*Proof.* Assume a zero-error attribute-only procedure halts after fewer than $d$ queries on every execution path. Fix any execution path and let $Q \subseteq \mathcal{I}$ be the set of queried attributes on that path, so $|Q|<d$. Since $d$ is the cardinality of every minimal distinguishing set, no set of size $<d$ is distinguishing; hence there exist values $v,w$ from different classes with identical answers on all attributes in $Q$.

An adversary can answer the procedure's queries consistently with both $v$ and $w$ along this path. Therefore the resulting transcript (and output) is identical on $v$ and $w$, contradicting zero-error class identification. So some execution path must use at least $d$ queries, giving worst-case cost $\Omega(d)$. ◻
:::

::: corollary
[]{#cor:attribute-only-witness-lb label="cor:attribute-only-witness-lb"} For any attribute-only observer, $W(\text{class-identity}) \ge d$ where $d$ is the distinguishing dimension.
:::

In the running example, this yields $W \ge 3$ for any attribute-only zero-error scheme, even though the ambiguity converse will later show that one explicit tag bit already suffices to resolve the only collision block.

The key insight: the distinguishing dimension is invariant across all minimal query strategies. The difference between nominal-tag and attribute-only observers lies in *witness cost*: a nominal tag achieves $W = O(1)$ by storing the identity directly, bypassing query enumeration.

This is the paper's complementary resource law. The ambiguity converse says how many bits are needed to eliminate collision multiplicity; the query lower bound says how much structure must still be traversed when those bits are absent.

## Witness Cost Comparison

::: {#tab:witness-comparison}
  **Observer Class**      **Witness Procedure**      **Witness Cost $W$**
  -------------------- ---------------------------- ----------------------
  Nominal-tag                Single tag read                $O(1)$
  Attribute-only        Query a distinguishing set       $\Omega(d)$

  : Witness cost for class identity by observer class.
:::


## Three-Dimensional Tradeoff: Tag Length, Witness Cost, Distortion

Recall from the preceding sections that observer strategies are characterized by three dimensions:

-   **Tag length** $L$: bits required to encode class information,

-   **Witness cost** $W$: minimum number of primitive queries for class identification,

-   **Distortion** $D$: probability of misclassification, $D = \Pr[\hat{C} \neq C]$.

We compare two observer classes: attribute-only observers, which query only attribute membership, and nominal-tag observers, which may read a class identifier in addition to attribute queries.

This section gathers the earlier theorem families into one frontier statement. Section [\[sec:zero-error\]](#sec:zero-error){reference-type="ref" reference="sec:zero-error"} establishes the minimum tag resource needed to eliminate ambiguity; Section [\[sec:matroid\]](#sec:matroid){reference-type="ref" reference="sec:matroid"} establishes the minimum query resource needed when tags are absent. The present section places those two laws on the same geometric picture. In the semantics-aware compression language of the paper, $L$ stores explicit naming information, $W$ measures interactive disambiguation effort, and $D$ measures semantic failure.

::: theorem
[]{#thm:lwd-optimal label="thm:lwd-optimal"} Let $A_\pi$ be the collision multiplicity (Definition [\[def:collision-multiplicity\]](#def:collision-multiplicity){reference-type="ref" reference="def:collision-multiplicity"}). Then:

1.  Any $D=0$ scheme must satisfy $L \ge \log_2 A_\pi$ (Theorem [\[thm:converse\]](#thm:converse){reference-type="ref" reference="thm:converse"}).

2.  In maximal-barrier domains ($A_\pi = k$), nominal-tag observers achieve the unique Pareto-optimal $D=0$ point:

    -   **Tag length**: $L = \lceil \log_2 k \rceil$ bits,

    -   **Witness cost**: $W = O(1)$ queries,

    -   **Distortion**: $D = 0$.

3.  In general (non-maximal) domains, nominal tagging remains Pareto-optimal at $D=0$ but need not be unique: partial tags can coexist on the frontier.

In information-barrier domains, attribute-only observers (the $L=0$ face) satisfy:

-   **Tag length**: $L = 0$ bits,

-   **Witness cost**: $W = \Omega(d)$ queries,

-   **Distortion**: $D > 0$ when collisions persist.
:::

::: proof
*Proof.* Converse item (1) is Theorem [\[thm:converse\]](#thm:converse){reference-type="ref" reference="thm:converse"}. For item (2), maximal barrier means all classes are observationally colliding, so any $D=0$ scheme must carry full class identity in tag bits (Corollary [\[cor:max-barrier-converse\]](#cor:max-barrier-converse){reference-type="ref" reference="cor:max-barrier-converse"}), while nominal tags realize this lower bound with one tag read. Pareto uniqueness follows because any competing $D=0$ point cannot reduce $L$ below $\log_2 k$ nor reduce $W$ below constant-time tag access under the admissibility rules of Section [\[sec:framework\]](#sec:framework){reference-type="ref" reference="sec:framework"}.

The converse chain has two steps: first, a constant transcript on a collision block forces any zero-error decoder to separate classes using tag information; second, counting those tag outcomes yields $2^L \ge A_\pi$. In the maximal-barrier regime this becomes $L \ge \log_2 k$, which nominal tagging achieves with one tag read. ◻
:::

::: remark
When $1 < A_\pi < k$, the $D=0$ frontier can include mixed designs: a partial tag identifies collision blocks and queries resolve within blocks. This does not contradict nominal optimality; it only removes global uniqueness outside maximal-barrier domains.

1.  Maximal barrier ($A_\pi=k$): unique $D=0$ nominal point.

2.  Intermediate barrier ($1<A_\pi<k$): multiple $D=0$ Pareto points may exist.

3.  No barrier ($A_\pi=1$): $L=0$ zero-error identification is feasible.
:::

## Pareto Frontier

The three-dimensional frontier shows:

-   in maximal-barrier domains, the unique $D=0$ Pareto point is nominal tagging at $L=\lceil \log_2 k\rceil$,

-   in general domains, attribute-only observers trade tag length for distortion on the $L=0$ face when collisions are present.

Figure [1](#fig:lwd-tradeoff){reference-type="ref" reference="fig:lwd-tradeoff"} visualizes the $(L, W, D)$ tradeoff space. The key observation is the ambiguity converse: the minimum zero-error tag rate is $\log_2 A_\pi$, with maximal-barrier specialization $\log_2 k$.

In the running example from Section [\[sec:zero-error\]](#sec:zero-error){reference-type="ref" reference="sec:zero-error"}, the point $(L,W,D)=(1,O(1),0)$ is feasible by tagging the collided pair $\{A,B\}$. The tag-free point $(0,\cdot,0)$ is infeasible because $A_\pi=2>1$, and the query lower bound from Section [\[sec:matroid\]](#sec:matroid){reference-type="ref" reference="sec:matroid"} still forces $W\ge 3$ whenever one attempts tag-free zero-error separation.

That example makes the frontier concrete. The converse does not merely say that some metadata is helpful; it identifies the exact one-bit shift needed to cross from impossible to feasible zero-error identification in the presence of the observed collision block.

<figure id="fig:lwd-tradeoff">

<figcaption>Schematic <span class="math inline">(<em>L</em>, <em>W</em>, <em>D</em>)</span> tradeoff across barrier regimes. Here <span class="math inline"><em>W</em></span> counts identification operations, including a constant-time tag read, so the nominal zero-error point sits at <span class="math inline"><em>W</em> = <em>O</em>(1)</span> rather than <span class="math inline"><em>W</em> = 0</span>. The left orange point shows the collision-free attribute-only regime (<span class="math inline"><em>A</em><sub><em>π</em></sub> = 1</span>), while the red point shows a colliding attribute-only regime with <span class="math inline"><em>D</em> &gt; 0</span>. Nominal tagging is the unique zero-error Pareto point in maximal-barrier domains.</figcaption>
</figure>

The supplementary artifact machine-checks the full ambiguity-based converse chain and maximal-barrier lower bound that anchor this tradeoff analysis.

::: remark
In one software-runtime instantiation: explicit runtime identifiers correspond to nominal-tag observers (e.g., CPython's `isinstance`, Java's `.getClass()`); attribute-probe schemes correspond to attribute-only observers (e.g., repeated `hasattr`-style checks); and interface-conformance checks are an intermediate case with $D=0$ but $W=O(s)$.
:::

::: remark
When conformance checks traverse $s$ members or fields (rather than ranging over the full attribute universe), the natural bound is $W = O(s)$ with $s \le n$.
:::

## Interpretation of the Zero-Error Frontier

The exact block law established in Section [\[sec:zero-error\]](#sec:zero-error){reference-type="ref" reference="sec:zero-error"} determines the horizontal zero-error threshold. The query lower bound from Section [\[sec:matroid\]](#sec:matroid){reference-type="ref" reference="sec:matroid"} determines the vertical cost of remaining tag-free. The frontier therefore separates the two resources cleanly: tag bits pay for collision multiplicity, while queries pay for distinguishing structure inside the observable family.

That separation is what makes the three-resource view useful. The rate coordinate does not merely summarize stored metadata, and the query coordinate does not merely summarize computation; together they describe two distinct ways of paying for semantic ambiguity.

::: remark
The finite-block laws used here are worst-case confusability results. They sharpen the ambiguity converse with exact thresholds, while remaining distinct from full distribution-optimized Körner graph-entropy theory.
:::


## Open-World Robustness and Decidability Boundary

The main converse addresses whatever collision structure is present in the observable world. The next two results show that a currently collision-free realized world does not escape the theorem in any durable sense: barrier-freedom is not extension-stable, and in full generality it is not computably certifiable in advance.

::: definition
Let $\mathcal{C}$ be the class universe. A *finite realized world* is a finite subset $W \subseteq \mathcal{C}$ of classes currently present in a deployed system snapshot. We call $W$ *barrier-free* iff $$\forall c_1,c_2 \in W,\ \pi_{\mathcal C}(c_1)=\pi_{\mathcal C}(c_2) \Rightarrow c_1=c_2.$$
:::

::: definition
An *open-world extension* is a super-world $W' \supseteq W$ obtained by adding classes while keeping the observation family $\Phi$ fixed.
:::

::: theorem
[]{#thm:open-world-extension-instability label="thm:open-world-extension-instability"} In the open-world class model formalized in Lean, it is not true that barrier-freedom is preserved under all extensions: $$\neg\Big(\forall W \subseteq W',\ \mathrm{BarrierFree}(W)\Rightarrow \mathrm{BarrierFree}(W')\Big).$$
:::

::: proof
*Proof.* Take the empty world $W_0=\varnothing$, which is barrier-free. In the Lean model, two concrete classes are shape-equivalent (same observable profile) but nominally distinct, so adding them yields an extension $W_1\supseteq W_0$ that is not barrier-free. Therefore barrier-freedom is not extension-stable. ◻
:::

The running example offers the same intuition in finite form: the realized world $\{A,C,D,E\}$ is collision-free, but adding class $B$ immediately restores the collision block $\{A,B\}$.

::: definition
For a partial observer generator $f:\mathbb{N}\rightharpoonup\mathbb{N}$, define $$\mathrm{HasBarrier}(f)\;:\Leftrightarrow\;\exists x\neq y,\exists z,\ z\in f(x)\land z\in f(y).$$ Define $\mathrm{BarrierFree}(f):\Leftrightarrow \neg\mathrm{HasBarrier}(f)$.
:::

::: theorem
[]{#thm:rice-barrier label="thm:rice-barrier"} There is no computable predicate on program codes deciding whether the generated observer has a barrier: $$\neg\mathrm{ComputablePred}\!\left(c\mapsto \mathrm{HasBarrier}(\mathrm{eval}(c))\right).$$ Equivalently, barrier-freedom certification is also non-computable.
:::

::: proof
*Proof.* Assume there is a computable certifier for $P(c):=\mathrm{HasBarrier}(\mathrm{eval}(c))$. The property is non-trivial: some generators produce collisions and some do not. Rice's theorem therefore implies that if membership in the corresponding semantic class were computable from code, every partial recursive function would have to satisfy it, contradiction. The barrier-free statement is equivalent by negation. ◻
:::

::: corollary
For unrestricted open-world generators, one cannot computably certify that attribute-only identification remains barrier-free under all future extensions. Consequently, persistent $D=0$ guarantees cannot rely on "currently collision-free" attribute structure alone.
:::


## Worked Example

The running example now supports the whole theorem chain with almost no extra notation. The collision block $\{A,B\}$ gives $A_\pi = 2$, so the ambiguity converse forces one tag bit for zero-error identification. The minimal distinguishing family has size $d=3$, so every tag-free zero-error scheme pays at least three queries. The exact block law gives $L_t^\star = t$, and the open-world section uses the same example to show how adding $B$ destroys present collision-freedom in the realized world $\{A,C,D,E\}$.

The following consequences show how the same collision structure governing the ambiguity converse reappears in maintenance locality, architectural centralization, and concrete data-management settings.

## Maintenance and Localization Consequences

#### Scope of this subsection.

This subsection studies *maintenance or localization* complexity, measured by locations to inspect ($E(\cdot)$), not per-instance identification complexity ($W$) from earlier sections. The metrics are related but distinct: $W$ quantifies online class-identification effort, while $E$ quantifies where constraint logic is distributed in implementations.

::: definition
Let $E(\mathcal{O})$ be the number of locations that must be inspected to find all potential violations of a constraint under observation family $\mathcal{O}$.
:::

::: theorem
[]{#thm:nominal-localization label="thm:nominal-localization"} $E(\text{nominal-tag}) = O(1)$.
:::

::: proof
*Proof.* Under nominal-tag observation, the constraint "$v$ must be of class $A$" is satisfied iff $\tau(v) \in \mathrm{subtypes}(A)$. This is determined at a single location: the definition of $\tau(v)$'s class. ◻
:::

::: theorem
[]{#thm:declared-localization label="thm:declared-localization"} $E(\text{attribute-only, declared}) = O(k)$ where $k$ is the number of entity classes.
:::

::: proof
*Proof.* With declared attribute sets, the constraint "$v$ must satisfy attribute $I$" requires verifying that each class satisfies all attributes in $I$. For $k$ classes, this yields $O(k)$ locations. ◻
:::

::: theorem
[]{#thm:attribute-localization label="thm:attribute-localization"} $E(\text{attribute-only}) = \Omega(m)$ where $m$ is the number of query sites.
:::

::: proof
*Proof.* Under attribute-only observation, each query site independently checks "does $v$ have attribute $a$?" with no centralized declaration. For $m$ query sites, each must be inspected. The lower bound is therefore $\Omega(m)$. ◻
:::

::: corollary
[]{#cor:strict-dominance label="cor:strict-dominance"} Nominal-tag observation strictly dominates attribute-only: $E(\text{nominal-tag}) = O(1) < \Omega(m) = E(\text{attribute-only})$ for all $m > 1$.
:::

## Information Scattering and Cross-Domain Illustrations

::: definition
Let $I(\mathcal{O}, c)$ be the set of locations where constraint $c$ is encoded under observation family $\mathcal{O}$.
:::

::: theorem
[]{#thm:attribute-scattering label="thm:attribute-scattering"} For attribute-only observation, $|I(\text{attribute-only}, c)| = O(m)$ where $m$ is the number of query sites using constraint $c$.
:::

::: proof
*Proof.* Each attribute query independently encodes the constraint. No shared reference exists. Constraint encodings therefore scale with query sites. ◻
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

**Databases (columns vs keys).** Rows may collide on non-key columns, so attribute-only identification is insufficient for guaranteed identity. Primary or surrogate keys realize the nominal-tag role and provide constant-cost identity checks [@Codd1990].

**Model registries and artifact stores.** Structural signatures can collide across distinct artifacts. Registry identifiers and hashes play the same nominal role as tags, while profile inspection alone inherits the ambiguity lower bounds.

**Biology (phenotype vs genotype).** Distinct species can collide under phenotype, so phenotype-only observation cannot guarantee zero-error identity. DNA barcoding supplies additional naming information that resolves collision blocks in constant-time lookup style [@DNABarcoding].

**Software runtimes (attribute probes vs runtime identifiers).** Attribute-probe checks recover identity from observed capabilities and pay query cost that scales with inspected structure. Runtime class or type identifiers behave as explicit tags and collapse identity checks to near-constant cost in the model [@CPythonDocs; @JVMSpec; @JavaDocs; @TypeScriptDocs; @RustDocs].

These illustrations are not historical causality claims. They show a common resource law: when observable profiles collide, zero-error identification requires additional naming information; otherwise the system pays higher query cost, higher maintenance complexity, and/or nonzero error.

The operational metrics in this section complement the main $L$-, $W$-, and $D$-resource laws. The same collision structure that drives the ambiguity converse also reappears in maintenance burden, error localization, and architectural scattering.


The main theorem chain is complete in the deterministic zero-error setting. We record only brief forward-looking directions here.

Each extension changes one part of the model while preserving the paper's central question: once identity is only partially visible, how do stored naming information, interactive evidence, and allowable error trade off?

## Noisy and Lossy Observation

With noisy queries, repeated sampling can reduce error, so the natural problem becomes the tradeoff among tag rate, repeated query cost, and target error level. In this regime, the model changes by replacing deterministic attribute observations with noisy ones, and the main open question is how the ambiguity law interacts with repeated evidence. Explicit tags should continue to act as a noise-free side channel whenever they are transmitted cleanly.

The noisy setting asks how the same ambiguity law interacts with repeated but unreliable evidence.

## Semantic Distortion Measures

The paper treats $D$ as binary misidentification probability. A lossy extension would replace that with hierarchical distortion, weighted misclassification penalties, or other task-specific semantic losses. The natural open problem is then to characterize how the $(L,W,D)$ frontier changes when near misses and catastrophic confusions are not penalized equally.

This matters most in settings where classes carry internal geometry or hierarchy: the current theory distinguishes zero error from nonzero error, while a lossy theory would need to distinguish tolerable confusion from unacceptable confusion.

## Privacy, Security, and Three-Resource Geometry

Nominal tags also suggest privacy-preserving class-membership proofs and stronger verification mechanisms in model-registry settings. Here the model change is that one constrains what information may be revealed while still certifying identity or class membership, and the open question is how much explicit naming information can be exposed safely.

More broadly, the $(L,W,D)$ frontier invites comparison with other three-resource tradeoff geometries, including rate-distortion-perception formulations [@blau2019rethinking]. The question left open here is whether there is a similarly sharp geometric characterization when the third resource is interactive query effort rather than perceptual fidelity.

These directions preserve the paper's central viewpoint: identity can be partially stored, partially queried, and partially approximated. Changing the model changes how those resources interact, but not the need to account for them explicitly.


This paper presents a resource-based analysis of zero-error semantic compression under observational constraints. The central law is the zero-error ambiguity converse: any $D=0$ scheme must satisfy $L \ge \log_2 A_\pi$, where $A_\pi$ is the largest collision block induced by observable profiles. The finite-block confusability results sharpen that law to exact thresholds and exact linear log-bit scaling across block length. In maximal-barrier domains, nominal tagging achieves the unique Pareto-optimal zero-error point in the $(L, W, D)$ tradeoff.

The second main arc characterizes the query resource. Minimal distinguishing query sets form the bases of a matroid, making the distinguishing dimension $d$ well-defined as an invariant of the observation structure. Attribute-only zero-error identification therefore pays at least $d$ queries in the worst case. Open-world and Rice-style boundary results explain why present-day collision-freedom is not enough for persistent guarantees, and the implementation-level results show how the same structure reappears as maintenance localization and information scattering.

Across domains, the same law recurs: when observable profiles collide, zero-error identification requires either explicit naming information or additional structure-specific work. The ambiguity converse, the exact block law, and the query lower bound formalize that tradeoff, while the open-world boundary shows why present collision-freedom does not remove it.

Biology, databases, software runtimes, and model registries instantiate the same constraint in different guises: phenotype collides where barcodes separate, non-key columns collide where primary keys separate, observable capabilities collide where runtime identifiers separate, and structural signatures collide where registry identifiers separate. The same observable collision structure governs all of them.

Zero-error class identification is therefore governed by three structural quantities: collision multiplicity fixes the minimum naming budget, the exact block law fixes its finite-block scaling, and the distinguishing dimension fixes the tag-free query burden. The open-world results show that these costs do not disappear just because a current snapshot happens to be separable.

The practical consequence is equally direct. Systems can centralize identity by carrying explicit names, or they can recover it indirectly by distributing distinguishing work across queries, declarations, and maintenance sites. The theorems identify exactly where that tradeoff comes from and which parts of it are unavoidable.

Collision multiplicity determines the minimum zero-error tag budget, distinguishing dimension determines the tag-free query burden, and both persist beyond any single realized world.

## Future Work

Natural extensions include other classification domains, witness complexity for semantic properties beyond identity, and hybrid observers that combine tags and attributes under non-uniform query distributions.

## Acknowledgment: AI-use Disclosure {#acknowledgment-ai-use-disclosure .unnumbered}

Generative AI tools (including Codex, Claude Code, Augment, Kilo, and OpenCode) were used throughout this manuscript, across all sections (Abstract, Introduction, theoretical development, proof sketches, applications, conclusion, and appendix) and across all stages from initial drafting to final revision. The tools were used for boilerplate generation, prose and notation refinement, LaTeX and structure cleanup, translation of informal proof ideas into candidate formal artifacts (Lean and LaTeX), and repeated adversarial reviewer-style critique passes to identify blind spots and clarity gaps.

The author retained full intellectual and editorial control, including problem selection, theorem statements, assumptions, novelty framing, acceptance criteria, and final inclusion or exclusion decisions. No technical claim was accepted solely from AI output. Formal claims reported as machine-verified were admitted only after Lean verification (no `sorry` in cited modules) and direct author review; Lean was used as an integrity gate for responsible AI-assisted research. The author is solely responsible for all statements, citations, and conclusions.




---

## Machine-Checked Proofs

All theorems are formalized in Lean 4:
- Location: `docs/papers/paper1_typing_discipline/proofs/`
- Lines: 8092
- Theorems: 352
- `sorry` placeholders: 0
