# Paper: Semantic Representation Compression: Exact Zero-Error Laws and Neurosymbolic Necessity

**Status**: JSAIT-ready | **Lean**: 9166 lines, 399 theorems

---

## Abstract

We prove that learned semantic representations alone cannot achieve zero-error identity recovery. When distinct entities map to the same representation, no decoder, regardless of architecture, training, or compute, can distinguish them without additional bits. If $A_\pi$ is the maximum representation-collision multiplicity, then the minimum worst-case auxiliary description is exactly $\log_2 A_\pi$ bits, and this bound is tight.

The coding laws are exact. For repeated independent uses of the same representation map, the minimum block budget is $L_t^\star=t\log_2 A_\pi$. In the adaptive regime, the optimal fiber budget is $\lceil \log_2 |\pi^{-1}(u)| \rceil$, any feasible adaptive scheme has expected length at least the expectation of that quantity and hence at least $H(C\mid U)/\log 2$, and standard fiberwise prefix coding achieves expected length at most $H(C\mid U)/\log 2 + 1$. Under a uniform source on a collision block of size $a$, subcritical fixed-length budgets satisfy the explicit error floor $D \ge 1-2^L/a$.

The same ambiguity-fiber geometry yields exact task-sufficiency, helper-view, and factorized-representation consequences. We instantiate these generic theorems in a concrete formal identity domain, proving that symbolic handles are information-theoretically necessary, not optional, for zero-error open-world identity recovery (NSL2, NSL3). The instantiated domain is a witness, not a restriction: the necessity claim applies to any learned representation with collisions. The pair consisting of a neural encoder and an injective symbolic handle achieves zero-error identity recovery if and only if the handle distinguishes entities within each semantic fiber (NSL1--NSL6). The identity bits must be paid somewhere, either in the encoder or in the symbolic handle. No free lunch.

**Keywords:** semantics-aware compression, zero-error coding, neurosymbolic systems, learned representations, side information


## Problem Statement

Learned semantic representations, embeddings, concept bottlenecks, and neural encoders compress entities into finite intermediate states. A basic question is whether those representations alone can support zero-error identity recovery in an open world.

This paper proves the answer is no whenever the representation has collisions. If two distinct entities map to the same semantic representation, then no decoder, regardless of architecture, training, or compute, can recover their identities without additional information. The identity bits must be paid somewhere, either in an injective encoder or in a symbolic handle.

Formally, let $C$ be a class label and let $U=\pi(C)$ be a finite representation available to the decoder. When $\pi$ is noninjective, the representation alone does not determine the class. The question in this paper is: how much additional description must be transmitted to recover $C$ exactly?

This is a zero-error compression problem with side information at the decoder. The side information is the representation value $U$, and the encoder sends auxiliary description $M$ only to resolve collisions inside the fiber $\pi^{-1}(U)$. The paper studies both fixed-length worst-case descriptions and representation-adaptive descriptions whose length may depend on the observed fiber, and then compares those transmission costs with two complementary modalities: query costs for operational disambiguation and robustness costs under world growth.

The paper should be read in two connected arcs. First, it gives exact coding laws for the residual ambiguity left by a fixed representation. Second, it shows that the same ambiguity-fiber object yields exact representation-sufficiency laws for downstream tasks, helper views, and modular representations.

The basic combinatorial step is simple: once the decoder already sees the representation, exact recovery reduces to resolving the residual ambiguity inside its fibers. The first arc is the exact finite backbone of the theory; the broader payoff is the wider yield of that same primitive across fixed-length, adaptive, entropy, distortion, and downstream representation-sufficiency consequences.

All theorems are relative to one explicit admissibility contract: the decoder may use only the declared representation value and any auxiliary description whose length is charged to the coding budget. No hidden registry, preprocessing table, or external side channel may contribute uncharged distinguishing information.

## Why This Is a Compression Problem

The central object is the fiber geometry induced by $\pi$. Let $$A_\pi := \max_u |\{c : \pi(c)=u\}|$$ be the largest collision fiber. Then the fixed-length zero-error question is exactly the minimum number of extra bits needed to separate the worst fiber. The answer is an exact lossless coding law: $$L \ge \log_2 A_\pi,$$ and this lower bound is tight. Thus the representation itself acts as decoder side information, and the remaining coding burden is precisely the ambiguity left unresolved by the representation.

This viewpoint also yields a finer representation-adaptive question. If the encoder may vary the extra description length with the observed representation value $u$, then the optimal pointwise budget is governed by the size of the individual fiber $\pi^{-1}(u)$. The resulting expected-length theorem is the average-case counterpart of the worst-case max-fiber law. In this sense, worst-case, adaptive, entropy-comparison, and below-threshold distortion statements are all different views of the same residual ambiguity object.

## Relation to Prior Work

The closest classical objects are zero-error coding with confusability constraints [@korner1973coding; @witsenhausen1976zero], source coding with side information, and more recent task-oriented or semantics-aware communication models [@gunduz2023beyond]. Our setting is deterministic and zero-error on the main theorem path: the representation is fixed, the decoder already observes $U=\pi(C)$, and the coding problem is to quantify the additional description required for exact recovery of $C$.

The paper does not claim a full graph-entropy characterization, a full rate-distortion region, or a general noisy semantic communication theory. Instead, it isolates an exact zero-error residual-coding problem induced by a fixed representation, solves it exactly in the fixed-length and pointwise fiberwise senses, places the adaptive regime between matching lower and standard prefix-style upper bounds, and uses the same fiber geometry to organize several semantic identification settings under one side-information viewpoint.

For the fixed-length zero-error theorem, the confusability graph is a special case: classes sharing one representation value form a clique, so the graph is a disjoint union of representation-fiber cliques. In that setting, the classical zero-error coloring law collapses to the statement that the required alphabet size is the largest fiber size. The interest here is therefore not a new graph-theoretic identity in isolation, but the identification of the correct residual coding object and the consequences that follow from it: exact fixed-length laws, fiberwise adaptive budgets, entropy comparisons, distortion floors, and representation-level interpretations from the same fiber geometry.

The relation to Slepian-Wolf coding is similarly directional rather than identical. Slepian-Wolf studies asymptotically optimal coding with correlated side information, whereas here the side information is the deterministic fixed representation $U=\pi(C)$ and the emphasis is on exact finite-block zero-error recovery. The common theme is decoder side information; the difference is that the present paper treats the representation map as fixed and asks for the residual coding burden it leaves behind. On the lossy side-information axis, Wyner-Ziv problems play the analogous role: the present paper isolates the deterministic exact-recovery corner before any noisy or rate-distortion refinement.

The modern hook is neurosymbolic necessity. If $\pi$ is interpreted as a learned embedding, then collision fibers measure the irreducible identity ambiguity left by semantic compression. Injective representations need no auxiliary metadata. Noninjective representations impose an exact-recovery cost that no downstream reasoning layer can remove on its own. The concrete instantiated formal domain in the paper is a witness for that claim, not a limitation on it: the zero-error necessity statement applies to any fixed learned representation with collisions. This is especially natural in concept-bottleneck and neuro-symbolic settings, where a fixed intermediate representation must support exact downstream symbolic or class-level decisions [@koh2020concept; @garcez2020neurosymbolic].

## Contributions

1.  **Fixed-length zero-error side-information rate.** Any exact identification scheme requires at least $\log_2 A_\pi$ extra bits in the worst case (Theorem [\[thm:converse\]](#thm:converse){reference-type="ref" reference="thm:converse"}), and this bound is tight via the exact one-shot feasibility threshold (Theorem [\[thm:graph-one-shot-threshold\]](#thm:graph-one-shot-threshold){reference-type="ref" reference="thm:graph-one-shot-threshold"}). For block length $t$, the minimum fixed-length budget is exactly $L_t^\star = t\log_2 A_\pi$ (Corollary [\[cor:graph-logbit-scaling\]](#cor:graph-logbit-scaling){reference-type="ref" reference="cor:graph-logbit-scaling"}).

2.  **Adaptive side information and average-case rates.** If the auxiliary description length may depend on the observed representation value $u$, then feasibility requires at least $\lceil \log_2 |\pi^{-1}(u)| \rceil$ bits on that fiber. Consequently, any feasible adaptive scheme has expected length at least the expectation of this pointwise optimal budget (Theorem [\[thm:adaptive-expected-rate-lower-bound\]](#thm:adaptive-expected-rate-lower-bound){reference-type="ref" reference="thm:adaptive-expected-rate-lower-bound"}), hence at least $H(C\mid U)/\log 2$ under any source law. Under standard prefix-coding achievability within each fiber, there also exists an adaptive zero-error code with expected length at most $H(C\mid U)/\log 2 + 1$ (Section [\[sec:lwd\]](#sec:lwd){reference-type="ref" reference="sec:lwd"}). The pointwise fiber budget is exact; the entropy comparison is a lower/upper bound pair rather than an exact equality claim.

3.  **Positive-distortion floor below the zero-error threshold.** On a collision block of size $a$ under the uniform source, any fixed-length $L$-bit scheme satisfies $D \ge 1 - 2^L/a$ (Theorem [\[thm:positive-distortion-floor\]](#thm:positive-distortion-floor){reference-type="ref" reference="thm:positive-distortion-floor"}). Thus subcritical budgets incur a concrete error floor rather than merely failing to reach zero error.

4.  **Neurosymbolic necessity theorems.** We instantiate the generic helper-view framework in a concrete identity domain, proving that semantic representation alone cannot achieve zero-error identity when collisions exist (NSL2) and that no decoder can recover collapsed identities without additional identity information (NSL3). The neurosymbolic linking theorems establish that the pair consisting of a semantic encoder and an injective symbolic handle achieves zero-error identity recovery, with the exact characterization that identity is recoverable if and only if the handle distinguishes all entities within each semantic fiber (NSL1, NSL4, NSL5). This helper-view pattern is canonical for open-world semantic systems (NSL6). The mechanization connects the information-theoretic helper-view framework (GPH40) to a concrete formal identity model (ACS8, ACS9, EMB1). The instantiated domain is a witness for the broader claim about learned representations with collisions. The identity bits must be paid somewhere, either in the encoder or in the symbolic handle.

5.  **Representation sufficiency laws.** In the finite explicit setting, one-symbol zero-error feasibility is equivalent to injectivity, and refining the observation channel cannot increase collision multiplicity (GPH27, GPH28). Coarsening a representation therefore cannot reduce either the worst-case fixed-length burden or the pointwise adaptive burden (Corollary [\[cor:concept-bottleneck-mono\]](#cor:concept-bottleneck-mono){reference-type="ref" reference="cor:concept-bottleneck-mono"}). More generally, for any downstream task label $Y=f(C)$, exact recovery from the representation alone is possible if and only if each representation fiber carries at most one task label (Theorem [\[thm:task-sufficiency\]](#thm:task-sufficiency){reference-type="ref" reference="thm:task-sufficiency"}), and coarsening cannot improve this task-level sufficiency (Corollary [\[cor:task-coarsening-mono\]](#cor:task-coarsening-mono){reference-type="ref" reference="cor:task-coarsening-mono"}). Helper views can only reduce that ambiguity (Theorem [\[thm:helper-view-sufficiency\]](#thm:helper-view-sufficiency){reference-type="ref" reference="thm:helper-view-sufficiency"}, Corollary [\[cor:helper-view-mono\]](#cor:helper-view-mono){reference-type="ref" reference="cor:helper-view-mono"}), and in genuinely factorized product settings the residual ambiguity multiplies exactly across modules (Corollary [\[cor:factorized-product-law\]](#cor:factorized-product-law){reference-type="ref" reference="cor:factorized-product-law"}). These results show that representation collisions are not merely a heuristic notion of quality but an exact zero-error residual-coding burden for downstream identification.

6.  **Query and stability costs of residual ambiguity.** Residual ambiguity can be paid for in more than one currency. If one does not transmit the needed side information, then exact recovery incurs interactive/query cost, quantified by the minimum distinguishing number $d$; equivalently, one must build a sufficiently rich helper view out of primitive observables. Even when a current snapshot is collision-free, ambiguity may return under world growth, and the robustness section quantifies that stability cost. These theorem families are complementary to the coding arc rather than separate topics.

## Worked Example

We use one small example throughout the paper. Five classes have profiles $$A=000,\quad B=000,\quad C=100,\quad D=110,\quad E=111.$$ Then $A$ and $B$ form a collision fiber of size two, so the exact worst-case fixed-length budget is one bit. The same example also gives a simple adaptive picture: the fiber over profile $000$ needs one auxiliary bit, while the singleton fibers need none. In the learned-representation reading, this is the exact price of one residual collision in the representation.

## Paper Organization

Section [\[sec:model\]](#sec:model){reference-type="ref" reference="sec:model"} formalizes the representation-side-information model and the basic information barrier. Section [\[sec:zero-error\]](#sec:zero-error){reference-type="ref" reference="sec:zero-error"} gives the fixed-length zero-error theorems. Section [\[sec:lwd\]](#sec:lwd){reference-type="ref" reference="sec:lwd"} develops the adaptive side-information viewpoint, the positive-distortion floor, and the main worked example. Section [\[sec:applications\]](#sec:applications){reference-type="ref" reference="sec:applications"} interprets the results through learned representations, concept bottlenecks, and neuro-symbolic readings. Section [\[sec:matroid\]](#sec:matroid){reference-type="ref" reference="sec:matroid"} studies query costs as an operational alternative to transmitting side information, and Section [\[sec:robustness\]](#sec:robustness){reference-type="ref" reference="sec:robustness"} studies the stability cost of residual ambiguity under system growth. Section [\[sec:conclusion\]](#sec:conclusion){reference-type="ref" reference="sec:conclusion"} closes with extensions and open directions. Section [\[sec:extensions\]](#sec:extensions){reference-type="ref" reference="sec:extensions"} sketches open directions, and Section [\[sec:conclusion\]](#sec:conclusion){reference-type="ref" reference="sec:conclusion"} concludes.


## Primitive Objects

The introduction has already motivated the side-information viewpoint. For the main theorem arc, the primitive object is the pair $(C,U)$ with $U=\pi(C)$. This section fixes one concrete realization of that coding model: a declared attribute family that induces the representation map $\pi$. The attribute-family formalism is retained because it supports the paper's broader semantic-identification interpretations, but the coding statements depend only on the induced representation together with any charged auxiliary description.

::: definition
Let $C$ be a finite class variable, let $U=\pi(C)$ be the representation available to the decoder, and let $M$ be any auxiliary description transmitted by the encoder. The main theorem arc studies the minimum length of $M$ needed for exact recovery of $C$ from $(U,M)$, in both fixed-length and representation-adaptive regimes.
:::

::: remark
The coding statements are written in terms of a class variable $C$ and representation $U=\pi(C)$. When convenient, we also use an underlying entity variable $V$ together with a class assignment map $C(V)$. Thus $U=\pi(V)$ is the representation of the realized entity, and the induced class label is $C(V)$. The two viewpoints are interchangeable; the coding arc should be read through the pair $(C,U)$.
:::

The remaining definitions explain one concrete way of generating the representation map $\pi$ from an attribute family.

::: definition
Let $\mathcal{V}$ be a set of entities (program objects, database records, biological specimens, library items). Let $\mathcal{I}$ be a finite set of binary *attributes*. Each $I \in \mathcal{I}$ induces a bipartition of $\mathcal{V}$ via $q_I$, and the full family induces the observational equivalence partition.
:::

::: definition
For each $I \in \mathcal{I}$, define the attribute-membership observation $q_I: \mathcal{V} \to \{0,1\}$: $$q_I(v) = \begin{cases} 1 & \text{if } v \text{ satisfies attribute } I \\ 0 & \text{otherwise} \end{cases}$$ Let $\Phi_{\mathcal{I}} = \{q_I : I \in \mathcal{I}\}$ denote the attribute observation family.
:::

::: remark
We write $n := |\mathcal{I}|$ for the ambient number of available attributes. We write $d$ for the minimum distinguishing number (Definition [\[def:distinguishing-dimension\]](#def:distinguishing-dimension){reference-type="ref" reference="def:distinguishing-dimension"}), so $d \le n$ and there exist worst-case families with $d = n$. We write $m$ for the number of *query sites* (call sites) that perform attribute checks in a program or protocol (used only in the secondary query-side discussion). When discussing a particular identification or verification task, we may write $s$ for the number of attributes actually queried or traversed by the procedure, with $s \le n$.
:::

::: definition
The attribute profile function $\pi: \mathcal{V} \to \{0,1\}^{|\mathcal{I}|}$ maps each value to its complete attribute signature: $$\pi(v) = (q_I(v))_{I \in \mathcal{I}}.$$
:::

::: definition
Values $v, w \in \mathcal{V}$ are *attribute-indistinguishable*, written $v \sim w$, iff $\pi(v) = \pi(w)$.
:::

The relation $\sim$ is an equivalence relation. We write $[v]_\sim$ for the equivalence class of $v$.

::: remark
The model intentionally forbids hidden side channels. If an external registry, whole-universe preprocessing table, reflection oracle, or cached cross-instance state is allowed to distinguish values that the declared observation family cannot distinguish, then the effective observable family has changed. The point of Definition [\[def:admissibility-contract\]](#def:admissibility-contract){reference-type="ref" reference="def:admissibility-contract"} is to keep the resource accounting honest: extra distinguishing information must either be represented as part of the observation model or charged to the auxiliary-description resource.
:::

::: definition
A *classification scheme* is any procedure (deterministic or randomized), with arbitrary time and memory, whose only access to a value $v \in \mathcal{V}$ is via:

1.  the observation family $\Phi = \{q_I : I \in \mathcal{I}\}$, where $q_I(v) = 1$ iff $v$ satisfies attribute $I$, and optionally

2.  an auxiliary description primitive returning charged side information about the class.

All theorems in this paper quantify over all such schemes.
:::

This definition is intentionally broad: schemes may be adaptive, randomized, or computationally unbounded. The constraint is *observational*, not computational.

::: definition
An *attribute-only observer* is any procedure whose interaction with a value $v \in \mathcal{V}$ is limited to queries in $\Phi_{\mathcal{I}}$. Formally, the observer interacts with $v$ only via primitive attribute queries $q_I \in \Phi_{\mathcal{I}}$; hence any transcript (and output) factors through $\pi(v)$.
:::

::: definition
[]{#def:admissibility-contract label="def:admissibility-contract"} All impossibility and optimality statements in this paper quantify over schemes whose evidence consists only of the declared observation family $\Phi$ together with any explicit auxiliary description charged to the side-information budget $L$. In particular, no external registry, reflection oracle, whole-universe preprocessing table, or amortized cross-instance cache may supply hidden distinguishing information.
:::

The core results require only $(\mathcal V,C,\pi)$ and the observation family $\Phi$. The fixed-axis language below is a representative specialization rather than an added axiom for the general theory.

## Representation-Induced Information Barrier

The basic coding obstruction is immediate: equal representations give the decoder the same side information. The next theorem records that invariant once, in the attribute-generated realization of the model.

::: definition
A property $P:\mathcal V \to Y$ is *attribute-computable* if there exists a function $f:\{0,1\}^{|\mathcal I|}\to Y$ such that $$P(v)=f(\pi(v))
\qquad\text{for all } v\in\mathcal V.$$ Equivalently, $P$ depends only on the representation induced by the declared attribute family.
:::

::: theorem
[]{#thm:information-barrier label="thm:information-barrier"}[]{#thm:info-barrier label="thm:info-barrier"} Let $P: \mathcal{V} \to Y$ be any function. If $P$ is attribute-computable, then $P$ is constant on $\sim$-equivalence classes: $$v \sim w \implies P(v) = P(w).$$ Equivalently: no attribute-only observer can compute any property that varies within an equivalence class.
:::

::: proof
*Proof.* Suppose $P$ is attribute-computable via $f$, i.e., $P(v) = f(\pi(v))$ for all $v$. Let $v \sim w$, so $\pi(v) = \pi(w)$. Then: $$P(v) = f(\pi(v)) = f(\pi(w)) = P(w).$$ ◻
:::

::: remark
The barrier is *informational*, not computational: once the representation is identical, unlimited time or memory does not help. We state it explicitly because the later converse, adaptive coding, query, and robustness results all reuse the same invariance premise.
:::

::: corollary
[]{#cor:provenance-barrier label="cor:provenance-barrier"} Let $C: \mathcal{V} \to \{1,\ldots,k\}$ be the class assignment function. If there exist values $v, w$ with $\pi(v) = \pi(w)$ but $C(v) \neq C(w)$, then class identity is not attribute-computable.
:::

::: proof
*Proof.* Direct application of Theorem [\[thm:information-barrier\]](#thm:information-barrier){reference-type="ref" reference="thm:information-barrier"} to $P = C$. ◻
:::

The relation $\sim$ partitions $\mathcal{V}$ into representation fibers, so an attribute-only observer operates on the quotient $\mathcal{V}/{\sim}$ rather than on raw entities. The remaining statements are axis-specialized restatements of the same barrier and are used only by later secondary results. Read through the paper's second arc, an axis map $\alpha$ is simply a candidate helper view: if all admissible observables factor through $\alpha$, then $\alpha$ is sufficient for every in-scope semantic property; if a task varies inside one $\alpha$-fiber, then that helper view is not sufficient and must be refined.

::: theorem
[]{#thm:model-completeness label="thm:model-completeness"} Let $\alpha:\mathcal V\to\mathcal A$ be an axis map such that every admissible primitive query factors through $\alpha$. Then every in-scope semantic property factors through $\alpha$: there exists $\tilde P$ with $$P(v)=\tilde P(\alpha(v)) \quad \text{for all } v\in\mathcal V.$$
:::

::: proof
*Proof.* An admissible strategy observes $v$ only through responses to primitive queries in $\Phi$. If each primitive query factors through $\alpha$, then the entire transcript factors through $\alpha$, and so does any output computed from that transcript. ◻
:::

::: corollary
[]{#cor:fixed-axis-incompleteness label="cor:fixed-axis-incompleteness"} If $\alpha(v)=\alpha(w)$ but $P(v)\neq P(w)$, then no admissible strategy can compute $P$ with zero error without adding new information beyond the declared observable state.
:::

::: remark
The fixed-axis sufficiency and insufficiency statements are simply the helper-view version of the information barrier: if a semantic task varies inside one declared state fiber, that state is not sufficient and must be refined by additional information.
:::


## Fixed-Length Zero-Error Recovery

We now formalize the exact side-information problem in channel-theoretic terms. Let $C: \mathcal{V} \to \{1, \ldots, k\}$ denote the class assignment function, and let $\pi: \mathcal{V} \to \{0,1\}^n$ denote the representation or attribute profile observed by the decoder.

::: definition
Let $V$ be a random entity drawn from distribution $P_V$ over $\mathcal{V}$, let $C(V)$ be its class label, and let $U=\pi(V)$ be its representation. The decoder sees the representation value $U$ and may additionally receive auxiliary side information from the encoder.
:::

::: theorem
[]{#thm:identification-capacity label="thm:identification-capacity"} Let $\mathcal{C} = \{1, \ldots, k\}$ be the class space. Zero-error identification with no auxiliary side information is achievable if and only if every class has a distinct representation profile. When $\pi$ is not class-injective, no zero-error identification scheme exists without additional side information.
:::

::: proof
*Proof.* *Achievability*: If $\pi$ is injective on classes, then observing $\pi(v)$ determines $C(v)$ uniquely. The decoder simply inverts the class-to-profile mapping.

*Converse*: Suppose two distinct classes $c_1 \neq c_2$ share a profile: there exist $v_1 \in c_1, v_2 \in c_2$ with $\pi(v_1) = \pi(v_2)$. Then any decoder $g(\pi(v))$ outputs the same class label on both $v_1$ and $v_2$, so it cannot be correct for both. Hence zero-error identification without auxiliary side information is impossible. ◻
:::

In the finite explicit setting, the same condition can be read directly from the representation channel: one-symbol zero-error feasibility is equivalent to injectivity (GPH27). Interpreting $\pi$ as a learned embedding, this is the exact threshold for zero-error class recovery from the representation alone.

::: remark
Under any distribution with positive mass on both colliding classes, $I(C; \pi(V)) < H(C)$. This is an average-case consequence of the deterministic barrier above.
:::

::: remark
In the identification paradigm of [@ahlswede1989identification], the decoder asks "is the message $m$?" rather than "what is the message?" This yields double-exponential codebook sizes. Our setting is different: we require zero-error identification of the *class*, not hypothesis testing. The one-shot zero-error identification feasibility threshold is binary rather than asymptotic.
:::

## Fixed-Length Auxiliary Description

We now show that augmenting the representation with an auxiliary codeword restores exact recovery.

::: definition
An *auxiliary description* is a value $M(v)$ associated with each $v \in \mathcal{V}$ and transmitted in addition to the representation value $\pi(v)$. In the fixed-length setting, the auxiliary description alphabet has size $2^L$.
:::

::: theorem
[]{#thm:tag-restored-sufficiency label="thm:tag-restored-sufficiency"} An auxiliary description of length $L \geq \lceil \log_2 k \rceil$ bits suffices for zero-error identification, regardless of whether $\pi$ is class-injective.
:::

::: proof
*Proof.* A class-injective auxiliary description assigns a unique identifier to each class. Reading that description together with the representation determines $C(v)$ exactly. ◻
:::

## Ambiguity Converse

The next theorem is the paper's main zero-error compression statement. It converts representation collision multiplicity directly into a necessary fixed-length side-information budget.

In graph terms, the present fixed-representation setting is a structured zero-error special case: confusable classes are exactly those lying in the same representation fiber, so the confusability graph is a disjoint union of cliques. The point of the section is therefore not a new general graph-entropy theorem, but the identification of the exact primitive governing this deterministic side-information regime and the theoremic consequences that follow from it.

::: definition
[]{#def:collision-multiplicity label="def:collision-multiplicity"} Let $\mathcal{C}=\{1,\ldots,k\}$ and let $\pi_{\mathcal{C}}:\mathcal{C}\to\{0,1\}^n$ be the class-level profile map. Define $$A_\pi := \max_{u \in \operatorname{Im}(\pi_{\mathcal{C}})} \left|\{c \in \mathcal{C} : \pi_{\mathcal{C}}(c)=u\}\right|.$$ Thus $A_\pi$ is the size of the largest class-collision block under the representation.
:::

::: remark
The converse is parameterized by $A_\pi$. When the class-profile map is given explicitly as a finite table, $A_\pi$ is obtained by grouping classes by profile and taking the largest fiber size. When $\pi$ is given implicitly by a large program, circuit, or learned model, computing or approximating $A_\pi$ becomes a separate representation-dependent problem. The theorem identifies the exact zero-error auxiliary description budget once $A_\pi$ is known; it does not assume that estimating $A_\pi$ is always tractable from an implicit description.
:::

If one observation channel factors through a richer one, collision multiplicity cannot increase (GPH28). Thus adding observable structure can only lower the corresponding zero-error tag bound. Under the same representation reading, enriching the embedding cannot worsen zero-error separability.

::: corollary
[]{#cor:concept-bottleneck-mono label="cor:concept-bottleneck-mono"} Let $\pi_{\mathrm{fine}}$ and $\pi_{\mathrm{coarse}}$ be two representations on the same class domain, and assume $$\pi_{\mathrm{coarse}} = f \circ \pi_{\mathrm{fine}}$$ for some map $f$. Then $$A_{\pi_{\mathrm{fine}}} \le A_{\pi_{\mathrm{coarse}}},
\qquad
\log_2 A_{\pi_{\mathrm{fine}}} \le \log_2 A_{\pi_{\mathrm{coarse}}}.$$ Moreover, for each fine representation value $u$, $$\left\lceil \log_2 \left|\pi_{\mathrm{fine}}^{-1}(u)\right| \right\rceil
\le
\left\lceil \log_2 \left|\pi_{\mathrm{coarse}}^{-1}(f(u))\right| \right\rceil.$$
:::

::: proof
*Proof.* The first inequality is the monotonicity of collision multiplicity under factorization. Taking binary logarithms gives the worst-case fixed-length comparison. The pointwise adaptive comparison is the same monotonicity statement applied fiberwise before taking ceiling logarithms. ◻
:::

::: definition
[]{#def:max-barrier label="def:max-barrier"} The domain is *maximal-barrier* if $A_\pi = k$, i.e., all classes collide under the observation map.
:::

::: theorem
[]{#thm:converse label="thm:converse"} For any classification domain, any fixed-length scheme achieving $D=0$ requires $$2^L \ge A_\pi
\quad\text{equivalently}\quad
L \ge \log_2 A_\pi,$$ where $A_\pi$ is the collision multiplicity.
:::

::: proof
*Proof.* Fix a collision block $G \subseteq \mathcal{C}$ with $|G|=A_\pi$ and identical representation profile. For classes in $G$, the decoder's side information is identical, so zero-error decoding must separate those classes using auxiliary description outcomes. With $L$ bits there are at most $2^L$ outcomes, hence $2^L \ge |G| = A_\pi$. ◻
:::

::: remark
The converse is a lossless coding law for semantic identity. The relevant confusable alphabet is the largest representation-collision block, and zero-error separation of that block requires exactly enough auxiliary outcomes to name its members. The theorem therefore has the same fixed-length form as a classical zero-error side-information budget, but the governing alphabet size is induced by the representation rather than by the ambient class set alone.
:::

::: corollary
[]{#cor:max-barrier-converse label="cor:max-barrier-converse"} If the domain is maximal-barrier ($A_\pi = k$), any zero-error scheme satisfies $L \ge \log_2 k$.
:::

In the running example, the collision block $\{A,B\}$ shows that $A_\pi = 2$, so Theorem [\[thm:converse\]](#thm:converse){reference-type="ref" reference="thm:converse"} already forces $L \ge 1$ bit of auxiliary description for zero-error identification.

## Exact Finite-Block Law

The ambiguity converse extends exactly to blocks, so the main compression law is not merely one-shot. The next theorem family gives exact finite-block scaling rather than an asymptotic approximation.

::: theorem
[]{#thm:graph-one-shot-threshold label="thm:graph-one-shot-threshold"} For an alphabet of $n$ auxiliary descriptions, zero-error class recovery is feasible if and only if $$A_\pi \le n.$$
:::

::: proof
*Proof.* By definition, two classes are adjacent in the confusability graph exactly when they have the same representation profile and are distinct. Hence each nonempty profile fiber is a clique, and there are no edges between different fibers. The confusability graph is therefore a disjoint union of profile-fiber cliques. Zero-error auxiliary description is equivalent to a proper coloring of this graph, so an alphabet of size $n$ is feasible exactly when each fiber can be colored injectively, i.e., exactly when every profile fiber has size at most $n$. Taking the maximum over fibers gives the criterion $A_\pi \le n$. ◻
:::

::: theorem
[]{#thm:graph-block-threshold label="thm:graph-block-threshold"} For block length $t\ge 1$ with coordinatewise observation on $\mathcal{C}^t$, zero-error auxiliary description with alphabet size $n_t$ is feasible if and only if $$A_\pi^t \le n_t.$$ Equivalently, the minimal feasible block alphabet is exactly $A_\pi^t$.
:::

::: corollary
[]{#cor:graph-logbit-scaling label="cor:graph-logbit-scaling"} Let $L_t^\star := \log_2(A_\pi^t)$ be the minimum block auxiliary-description budget in bits. Then $$L_t^\star = t\,\log_2 A_\pi,
\qquad
\frac{L_t^\star}{t} = \log_2 A_\pi.$$ So per-entity tag rate is exactly stable across block length.
:::

::: remark
The block construction repeats the same semantic identification task across $t$ entities while keeping the representation map fixed and applying it coordinatewise. It is therefore the direct block analogue of repeated one-shot classification in this model: the collision structure multiplies across coordinates, and the required zero-error side-information budget scales accordingly.
:::

In the running example, $A_\pi = 2$, so the exact block law gives $L_t^\star = t$ bits.


## Representation-Adaptive Side Information

The fixed-length theorem treats all representation fibers through the worst case $A_\pi$. A finer coding problem allows the auxiliary description length to depend on the observed representation value $u$. Then the relevant budget is not the maximum fiber size but the size of the individual fiber $\pi^{-1}(u)$. This section therefore resolves the same residual ambiguity object at a finer scale: first fiberwise, then in source-weighted expectation, and finally through active binary clarification protocols.

::: definition
An *adaptive auxiliary scheme* assigns to each representation value $u$ a binary description budget $\ell(u)$ and a decoder that, given $u$ together with a binary string of length at most $\ell(u)$, recovers the class exactly whenever $\pi(C)=u$.
:::

::: theorem
[]{#thm:adaptive-fiber-budget label="thm:adaptive-fiber-budget"} For a representation fiber of size $|\pi^{-1}(u)|$, zero-error recovery under a binary auxiliary alphabet is feasible if and only if $$|\pi^{-1}(u)| \le 2^{\ell(u)}.$$ Consequently, the pointwise optimal number of auxiliary bits on fiber $u$ is $$\ell^\star(u) = \left\lceil \log_2 |\pi^{-1}(u)| \right\rceil.$$
:::

::: proof
*Proof.* Once $u$ is fixed, the decoder only needs to distinguish classes inside the fiber $\pi^{-1}(u)$. With $\ell(u)$ binary bits there are at most $2^{\ell(u)}$ auxiliary outcomes, so exact recovery is possible exactly when that many outcomes suffice to name the fiber elements injectively. The minimal feasible budget is therefore the least integer whose power of two dominates the fiber size, namely $\lceil \log_2 |\pi^{-1}(u)| \rceil$. ◻
:::

::: definition
Let $U=\pi(C)$ be the induced representation random variable under a source law on classes. For an adaptive budget $\ell$, define its expected auxiliary length by $$\mathbb E[\ell(U)] = \sum_u P_U(u)\,\ell(u).$$
:::

::: theorem
[]{#thm:adaptive-expected-rate-lower-bound label="thm:adaptive-expected-rate-lower-bound"} Any feasible adaptive auxiliary scheme satisfies $$\mathbb E[\ell(U)] \ge \mathbb E\!\left[\left\lceil \log_2 |\pi^{-1}(U)| \right\rceil\right].$$ Moreover, the pointwise optimal budget from Theorem [\[thm:adaptive-fiber-budget\]](#thm:adaptive-fiber-budget){reference-type="ref" reference="thm:adaptive-fiber-budget"} attains equality.
:::

::: proof
*Proof.* By Theorem [\[thm:adaptive-fiber-budget\]](#thm:adaptive-fiber-budget){reference-type="ref" reference="thm:adaptive-fiber-budget"}, feasibility on fiber $u$ requires $$\ell(u) \ge \left\lceil \log_2 |\pi^{-1}(u)| \right\rceil$$ for every representation value $u$. Multiplying pointwise by $P_U(u)$ and summing over $u$ gives the lower bound on expected length. Equality is achieved by choosing the pointwise optimal budget on every fiber. ◻
:::

::: corollary
Let $C$ be drawn from any source distribution and let $U=\pi(C)$. Then any feasible adaptive auxiliary scheme satisfies $$\mathbb E[\ell(U)] \ge \frac{H(C\mid U)}{\log 2},$$ where $H(C\mid U)$ is conditional Shannon entropy in natural-log units.
:::

::: proof
*Proof.* For each representation value $u$, the conditional law of $C$ given $U=u$ is supported on the fiber $\pi^{-1}(u)$, so its entropy is at most $\log |\pi^{-1}(u)|$. Averaging over $u$ gives $$H(C\mid U) \le \mathbb E[\log |\pi^{-1}(U)|].$$ Since $\log |\pi^{-1}(u)| \le \lceil \log_2 |\pi^{-1}(u)| \rceil \log 2$ fiberwise, Theorem [\[thm:adaptive-expected-rate-lower-bound\]](#thm:adaptive-expected-rate-lower-bound){reference-type="ref" reference="thm:adaptive-expected-rate-lower-bound"} yields the stated lower bound. ◻
:::

::: theorem
Assume standard prefix-coding achievability on each finite fiber. Then there exists an adaptive zero-error auxiliary code satisfying $$\mathbb E[\ell(U)] \le \frac{H(C\mid U)}{\log 2} + 1.$$
:::

::: proof
*Proof.* Because the decoder already knows $U=u$, prefix-freeness is required only within each fiber $\pi^{-1}(u)$. Applying a standard prefix-coding theorem separately to the conditional source on each nonempty fiber gives an expected conditional code length at most $H(C\mid U=u)/\log 2 + 1$. Averaging over $u$ yields the stated bound. ◻
:::

::: corollary
If, after observing $U$, the decoder is allowed to ask adaptive binary clarification questions whose leaves identify the class exactly, then the minimum expected number of clarification questions is bounded between $$\frac{H(C\mid U)}{\log 2}
\qquad\text{and}\qquad
\frac{H(C\mid U)}{\log 2}+1.$$
:::

::: proof
*Proof.* Each binary clarification protocol induces a binary code over the conditional source on each fiber, and conversely any fiberwise binary prefix code can be implemented as an adaptive yes/no decision tree. The lower and upper bounds therefore follow from the preceding conditional-entropy comparison and fiberwise prefix-coding theorem. ◻
:::

::: corollary
Let $T$ be any repeated or noisy observation transcript whose realized value factors through the fixed representation, i.e., $T = h(U)$ for some map $h$. Then recovering $C$ from $T$ cannot require less worst-case or pointwise adaptive side information than recovering $C$ from $U$ itself.
:::

::: proof
*Proof.* If the entire transcript factors through $U$, then it is a coarsening of the original representation. Corollary [\[cor:concept-bottleneck-mono\]](#cor:concept-bottleneck-mono){reference-type="ref" reference="cor:concept-bottleneck-mono"} applies directly: coarsening cannot reduce the largest collision fiber and cannot reduce the pointwise adaptive fiber budget. Thus repeated or noisy observations that never break the original fibers cannot remove the residual zero-error coding burden. ◻
:::

::: remark
Theorem [\[thm:adaptive-expected-rate-lower-bound\]](#thm:adaptive-expected-rate-lower-bound){reference-type="ref" reference="thm:adaptive-expected-rate-lower-bound"} is the average-case counterpart of the max-fiber law. The fixed-length theorem pays for the worst collision fiber; the adaptive theorem pays for each fiber separately and averages under the source-induced law of $U$. Thus the fixed-length law, the pointwise fiber law, and the entropy sandwich are three resolutions of the same object: the cost of naming the classes that remain collapsed inside each representation fiber.
:::

::: remark
The conditional-entropy lower and upper bounds place the adaptive theorem inside classical source-coding language: zero-error adaptive side information is sandwiched around $H(C\mid U)$ up to the standard one-bit prefix-coding overhead, but the exact zero-error rate is still controlled fiberwise by finite collision structure rather than by entropy alone.
:::

## Positive Distortion Below the Fixed-Length Threshold

The zero-error threshold does not exhaust the geometry. On a nontrivial collision block carrying a fixed source, subcritical fixed-length budgets force an explicit distortion floor even if the decoder is otherwise unrestricted.

::: theorem
[]{#thm:positive-distortion-floor label="thm:positive-distortion-floor"} Let $G \subseteq \mathcal{C}$ be a collision block of size $a$, so all classes in $G$ have the same representation profile. Assume the source class $C$ is uniform on $G$. Then any fixed-length scheme with auxiliary budget $L$ bits satisfies $$\Pr[\hat C = C] \le \frac{2^L}{a},
\qquad
D = \Pr[\hat C \neq C] \ge 1 - \frac{2^L}{a}.$$ In particular, if $a = A_\pi$ and $L < \log_2 A_\pi$, then every scheme has strictly positive distortion.
:::

::: proof
*Proof.* Fix a realization of all internal randomness used by the scheme. Because every class in $G$ has the same representation profile, the decoder's side information is the same on all classes in $G$; on that block, the decoder can therefore distinguish classes only through the auxiliary outcome. With $L$ bits there are at most $2^L$ outcomes, and for each such outcome the decoder can be correct on at most one class in $G$. Hence, for the fixed randomness, the scheme is correct on at most $2^L$ of the $a$ classes in $G$, so its success probability under the uniform distribution on $G$ is at most $2^L/a$. Averaging over the scheme's randomness preserves the same bound. The distortion lower bound is the complement of the success bound. ◻
:::

## Worked Example

Return to the running example with profiles $$A=000,\quad B=000,\quad C=100,\quad D=110,\quad E=111.$$ The fixed-length theorem gives $A_\pi=2$, so one extra bit is necessary and sufficient for exact recovery. The adaptive theorem says more: the fiber over profile $000$ requires one bit, while the singleton fibers over $100$, $110$, and $111$ require zero bits. Thus the expected auxiliary rate is exactly the probability of landing in the collided fiber.

This is the representation-level interpretation the paper emphasizes. The representation already resolves three of the four fibers completely; the only irreducible metadata cost is the one residual collision between $A$ and $B$.

The next section turns this coding object into a representation-sufficiency theory: once the residual ambiguity of each fiber is explicit, one can ask exactly which downstream tasks, helper views, and modular factors eliminate it and which do not.

## Secondary Query and Frontier View

The paper also studies a query resource $W$ for settings where the decoder is allowed to interrogate primitive attributes rather than receive all auxiliary information up front. That query-side viewpoint is secondary here: the main compression object is the side-information rate needed to resolve representation collisions. We retain the query-side results later because they show one alternative way to pay for the same unresolved ambiguity.


#### Why semantic representation alone cannot distinguish.

The mechanization proves that semantic representation cannot distinguish entities with identical observable profiles (ACS8), cannot track provenance or source identity (ACS9), and cannot embed explicit identity information into the semantic layer (EMB1). These impossibility theorems establish that the information barrier is not a limitation of specific encodings but a fundamental property of representation-level observation. In compression-theoretic terms, semantic representation alone is a *lossy* identity compressor. When distinct entities map to the same semantic representation, no decoder can recover the original identity without additional side information (NSL2, NSL3).

The coding laws above quantify how much ambiguity a fixed representation leaves unresolved. We now turn that same fiber geometry into a representation-sufficiency theory: which downstream tasks factor through the representation, when auxiliary views genuinely help, and when modular latent structure adds exact information rather than architectural redundancy.

The fixed-length law measures the worst unresolved collision, $$L^\star_{\mathrm{fix}} = \log_2 A_\pi,$$ while the adaptive law measures the source-weighted cost of the individual fibers, $$L^\star_{\mathrm{adapt}} = \mathbb E\!\left[\left\lceil \log_2 |\pi^{-1}(U)| \right\rceil\right].$$ Thus representation collisions translate directly into auxiliary metadata cost. Injective representations need no additional description; noninjective ones require either extra side information or nonzero error.

## Semantic, Identity, and Neuro-Symbolic Readings

The same fiber picture clarifies the distinction between semantic description and exact identity. Semantic information is what the representation exposes. Identity information is the exact class label or entity name that may still need to be carried explicitly. Zero-error recovery from semantic information alone is possible exactly when the semantic representation is injective on identity classes. When it is not, the collision fibers quantify the irreducible identity residue that must be restored by explicit metadata.

This transfers directly to neuro-symbolic systems. A neural encoder supplies the semantic side by producing a learned representation; a symbolic layer attempts to recover or reason about exact class-level distinctions. The fiber geometry says precisely when this is possible without extra scaffolding and when it is not: symbolic reasoning cannot recover distinctions already collapsed by the encoder, except by receiving additional identifiers, additional observables, or additional disclosures.

#### Instantiated neurosymbolic linking theorems.

The generic helper-view theorems specialize to a concrete formal identity domain by setting the class space to $\mathsf{Typ}$ and the task to identity $Y = C$. The mechanization (NSL1--NSL6) proves the following instantiated claims.

-   If the symbolic handle $\sigma$ is injective, the neurosymbolic link $(\pi, \sigma)$ achieves zero-error identity recovery (NSL1). This is the identity-task instantiation of the injectivity criterion GPH25.

-   If $\pi$ has collisions, semantic representation alone cannot achieve zero-error identity (NSL2), and no decoder can recover the collapsed identities without identity-resolving side information (NSL3).

-   The exact characterization is: $(\pi, \sigma)$ recovers identity if and only if $\sigma$ distinguishes all entities within each semantic fiber (NSL4). An injective handle therefore suffices (NSL5). This is the identity-task instantiation of the helper-view theorem GPH40.

-   The pair $(\pi, \sigma)$ is the canonical helper view for open-world identity systems (NSL6).

These are not merely prose interpretations. They are instantiated theorems connecting the generic information-theoretic helper-view framework to a concrete formal identity domain. Together with ACS8, ACS9, and EMB1, they show that the symbolic layer pays the identity bit budget that the semantic layer leaves unresolved.

#### What this means for ML systems.

The instantiated theorems have immediate implications for learned representation systems.

-   **Concept bottleneck models.** If the intermediate concept representation has collisions, then no downstream classifier can distinguish the collapsed concepts without extra identity information. Operationally, exact concept-level decisions require an added disambiguation channel, a retrieval key, or an explicit symbolic concept identifier.

-   **Retrieval-augmented systems.** The retrieval index pays the identity bit budget. If two artifacts have the same embedding, the system must store distinguishing metadata elsewhere, such as document identifiers, source pointers, or exact lookup keys. Otherwise exact retrieval is impossible on the collided fiber.

-   **Pure representation schemes.** Systems that attempt to avoid any nominal layer do not escape the theorem. They can still be useful, but they cannot guarantee zero-error open-world identity when the representation is non-injective.

-   **Open-world deployments.** Any system that must identify entities exactly in an open world must either use an injective encoder or carry explicit identity metadata. There is no third option.

The neurosymbolic point is therefore stronger than an architectural recommendation. It is a necessity statement. Semantic representation alone is lossy identity compression. The symbolic handle restores the missing identity bits. In deployed systems, those bits may appear as identifiers, registry entries, retrieval keys, or explicit symbolic names, but they must appear somewhere if zero-error identity is the goal.

## Task Sufficiency, Helper Views, and Factorization

The neurosymbolic reading is one specialization of a broader theorem family. The same fiber geometry characterizes when a representation is sufficient for an arbitrary downstream task, when helper views genuinely help, and when factorized modules contribute real exact information.

::: definition
Let $Y=f(C)$ be a downstream task label induced by the class. For a fixed representation $\pi$, define $$A_{\pi \to Y} := \max_u \left|\{f(c) : \pi(c)=u\}\right|.$$ This is the maximum number of task labels that remain unresolved inside one representation fiber.
:::

::: theorem
[]{#thm:task-sufficiency label="thm:task-sufficiency"} The representation alone supports zero-error recovery of $Y$ if and only if $A_{\pi \to Y} \le 1$.
:::

::: proof
*Proof.* Zero-error recovery of $Y$ from $U$ alone is possible exactly when every representation fiber carries a single task label. That condition is equivalent to the statement that the largest task-fiber ambiguity is at most one. ◻
:::

::: corollary
[]{#cor:task-coarsening-mono label="cor:task-coarsening-mono"} If $\pi_{\mathrm{coarse}} = g \circ \pi_{\mathrm{fine}}$, then $$A_{\pi_{\mathrm{fine}} \to Y} \le A_{\pi_{\mathrm{coarse}} \to Y}.$$ In particular, coarsening a representation cannot improve exact downstream task sufficiency.
:::

::: proof
*Proof.* Every coarse fiber is a union of fine fibers, so collapsing from $\pi_{\mathrm{fine}}$ to $\pi_{\mathrm{coarse}}$ cannot reduce the number of task labels realizable inside the worst fiber. ◻
:::

::: theorem
[]{#thm:helper-view-sufficiency label="thm:helper-view-sufficiency"} Let $S=\sigma(C)$ be an additional deterministic view available to the decoder. Then exact recovery of $Y$ from $(U,S)$ is possible if and only if each joint fiber of $(\pi,\sigma)$ carries at most one task label.
:::

::: proof
*Proof.* The pair $(U,S)$ is just a refined deterministic representation. Exact recovery is therefore possible exactly when the downstream task is constant on the fibers of that refined representation. ◻
:::

::: corollary
[]{#cor:helper-view-mono label="cor:helper-view-mono"} Adding a deterministic helper view cannot worsen task ambiguity. In particular, the task-fiber ambiguity under $(U,S)$ is at most the task-fiber ambiguity under $U$ alone.
:::

::: proof
*Proof.* Passing from $U$ to $(U,S)$ refines the representation fibers, so the largest number of task labels per fiber can only decrease. ◻
:::

::: corollary
[]{#cor:factorized-product-law label="cor:factorized-product-law"} In a genuinely factorized product setting, where the class space decomposes as a product and the representation decomposes coordinatewise, the joint fiber is the product of the component fibers and the worst-case residual ambiguity multiplies exactly across modules.
:::

::: proof
*Proof.* For coordinatewise product representations, each joint fiber is exactly the product of the corresponding component fibers, so their cardinalities multiply. Taking the largest such product yields multiplicativity of the worst-case residual ambiguity. ◻
:::

For learned representations, estimating quantities such as $A_\pi$, $A_{\pi \to Y}$, or the adaptive fiber profile is itself a representation-analysis problem rather than part of the theorem. In a finite explicit registry or index this is just fiber counting; for a large learned embedding one would need either exact collision enumeration after discretization or provable upper and lower bounds on the induced fibers. The point of the present results is to identify what those quantities mean once they are known.

Read concretely, the task-sufficiency theorem is the exact-recovery analogue of a sufficient-statistic condition for the downstream task $Y$: the representation is sufficient precisely when every fiber carries one task label. The helper-view and factorized laws then say when concept bottlenecks, retrieval-augmented pipelines, multimodal encoders, or modular neural architectures contribute genuinely new exact information rather than repackaging ambiguity already present in the base representation.

In the finite explicit setting, one-symbol zero-error feasibility is equivalent to injectivity and refinement cannot increase collision multiplicity (GPH27, GPH28). The same setting also yields a disclosure separation between nominal access and tag-free identity resolution (PRIV3).

## One Additional Semantic-Identification Example

Consider a model registry in which artifacts are indexed by a semantic representation $\pi$ built from declared capabilities or structural signatures. If two distinct artifacts have the same representation value, then the representation alone cannot support exact registry identification. The fixed-length theorem says that the worst-case auxiliary identifier length is determined by the largest collision bucket. The adaptive theorem says that rare collision buckets contribute little to expected metadata cost, while large or frequent buckets dominate it.

This example is close to the intended task-aware compression reading: the representation is already useful for search and organization, but exact identification may still require additional stored metadata. The theorem identifies the exact price of that residual ambiguity.


## Query Families and Distinguishing Sets

We now record the query-side version of the same representation-sufficiency problem. The preceding sections quantify how many auxiliary bits are needed to resolve representation collisions directly. An alternative operational response to the same residual ambiguity is to spend query resources rather than transmission resources: instead of sending side information to separate a fiber, one can interrogate additional primitive attributes until the class is determined. Equivalently, a query family induces an additional deterministic helper view, and the question is how large that helper view must be before exact recovery becomes possible. A distinguishing set separates every pair of distinct classes. In full generality, inclusion-minimal distinguishing sets need not have the same size, so the first invariant we use is the minimum number of primitive queries needed for zero-error separation.

This section is secondary to the main side-information results, but it belongs to the same second arc. Its role is to quantify how much adaptive interrogation is needed when one chooses not to send the required side information up front. In that sense, the minimum distinguishing number $d$ is the minimum size of a sufficiency-restoring helper view built from primitive observables. It is to queries what fiber size is to side-information bits: both measure how much unresolved ambiguity remains after the representation is fixed.

::: definition
Let $\mathcal{Q}$ be the set of all primitive queries available to an observer. For a classification system with attribute set $\mathcal{I}$, we have $\mathcal{Q} = \{q_I : I \in \mathcal{I}\}$ where $q_I(v) = 1$ iff $v$ satisfies attribute $I$.
:::

In this section, "queries" means primitive attribute predicates from $\Phi=\mathcal Q$.

::: definition
A subset $S \subseteq \mathcal{Q}$ is *distinguishing* if, for all values $v, w$ with $\mathrm{class}(v) \neq \mathrm{class}(w)$, there exists $q \in S$ such that $q(v) \neq q(w)$.
:::

::: definition
[]{#def:distinguishing-dimension label="def:distinguishing-dimension"} The *minimum distinguishing number* of a classification system is $$d := \min\{|S| : S \subseteq \mathcal Q \text{ is distinguishing}\}.$$ Equivalently, $d$ is the smallest number of primitive queries that suffices to separate all classes with zero error.
:::

::: remark
These notions differ: an inclusion-minimal distinguishing set may still be larger than a minimum-cardinality one. The lower bound below depends only on the operational quantity $d$.
:::

In the running example from Section [\[sec:zero-error\]](#sec:zero-error){reference-type="ref" reference="sec:zero-error"}, $\{q_1,q_2,q_3\}$ is distinguishing and no pair of primitive queries is distinguishing, so the minimum distinguishing number is $d=3$.

## Implications for Witness Cost

::: theorem
[]{#thm:interface-lower-bound label="thm:interface-lower-bound"} For attribute-only observers, class identity checking requires $$W(\text{class-identity}) = \Omega(d)$$ in the worst case, where $d$ is the minimum distinguishing number.
:::

::: proof
*Proof.* Assume a zero-error attribute-only procedure halts after fewer than $d$ queries on every execution path. Fix any execution path and let $Q \subseteq \mathcal{I}$ be the set of queried attributes on that path, so $|Q|<d$. By definition of $d$, no set of size $<d$ is distinguishing; hence there exist values $v,w$ from different classes with identical answers on all attributes in $Q$.

An adversary can answer the procedure's queries consistently with both $v$ and $w$ along this path. Therefore the resulting transcript (and output) is identical on $v$ and $w$, contradicting zero-error class identification. So some execution path must use at least $d$ queries, giving worst-case cost $\Omega(d)$. ◻
:::

::: corollary
[]{#cor:attribute-only-witness-lb label="cor:attribute-only-witness-lb"} For any attribute-only observer, $W(\text{class-identity}) \ge d$ where $d$ is the minimum distinguishing number.
:::

In the running example, this yields $W \ge 3$ for any attribute-only zero-error scheme, even though the ambiguity converse will later show that one explicit tag bit already suffices to resolve the only collision block.

This is the complementary resource law. The ambiguity converse says how many bits are needed to eliminate collision multiplicity; the query lower bound says how much additional structure must still be traversed when those bits are absent. Read through the second arc, the bit law measures the residual description cost of restoring sufficiency, while the query law measures the minimal observable augmentation needed to restore it operationally. When side information is cheap but interrogation is costly, the fiber-size bound governs; when transmission is expensive but queries are available, the distinguishing-number bound governs.

## Structured Query Families and Matroid Behavior

The general attribute-query model above does not force inclusion-minimal distinguishing sets to be equicardinal. A finite explicit counterexample is available even for four classes and four primitive binary queries (GPH31). A stronger structural theorem becomes valid only after one restricts the query family. In that restricted setting, the question becomes: when primitive observables are treated as irreducible helper-view coordinates, what is the structure of the minimal coordinate families that restore exact recoverability?

::: definition
A *structured axis model* is a query family presented together with an explicit derivability relation between primitive observables, so that one query may semantically subsume another. A query family is *orthogonal* if no primitive observable is derivable from a distinct primitive observable in the family.
:::

::: remark
If the observable family contains redundant or semantically derivable coordinates, then inclusion-minimality depends on representational accidents. The structured axis model removes that artifact by quotienting out derivable observables and restricting attention to orthogonal primitive axes.
:::

::: theorem
[]{#thm:matroid-bases label="thm:matroid-bases"} In the structured axis model, semantically minimal complete axis families form the bases of a matroid. Consequently, all such families have equal cardinality.
:::

::: proof
*Proof.* The formal theorem is proved for the axis-based representation in which semantic minimality implies orthogonality, orthogonality implies exchange, and standard matroid theory then gives equicardinality of maximal independent sets. The key point is that the theorem is about structured primitive axes modulo derivability, not about arbitrary subsets of binary class-indicator queries. ◻
:::

::: remark
The restricted matroid theorem strengthens the general lower-bound story when the primitive queries come from a derivability-aware axis presentation. In the unrestricted attribute-query model, the invariant used in the paper is the minimum distinguishing number $d$; in the structured axis model, the same quantity is certified by matroid basis cardinality.
:::

## Witness Cost Comparison

::: {#tab:witness-comparison}
  **Observer Class**                **Witness Procedure**               **Witness Cost $W$**
  -------------------- ----------------------------------------------- ----------------------
  Nominal-tag                          Single tag read                         $O(1)$
  Attribute-only        Query a distinguishing family of minimum size       $\Omega(d)$

  : Witness cost for class identity by observer class.
:::


## Secondary Robustness and Decidability Boundary

The main side-information theorems address whatever collision structure is present in the realized representation. A complementary question is whether that fiber geometry is stable under system evolution: can a representation that is currently collision-free become noninjective when the class universe grows? The next two results are secondary robustness statements quantifying that fragility.

::: definition
Let $\mathcal{C}$ be the class universe. A *finite realized world* is a finite subset $W \subseteq \mathcal{C}$ of classes currently present in a deployed system snapshot. We call $W$ *barrier-free* iff $$\forall c_1,c_2 \in W,\ \pi_{\mathcal C}(c_1)=\pi_{\mathcal C}(c_2) \Rightarrow c_1=c_2.$$
:::

::: definition
An *open-world extension* is a super-world $W' \supseteq W$ obtained by adding classes while keeping the observation family $\Phi$ fixed.
:::

::: theorem
[]{#thm:open-world-extension-instability label="thm:open-world-extension-instability"} In the open-world class model used here, it is not true that barrier-freedom is preserved under all extensions: $$\neg\Big(\forall W \subseteq W',\ \mathrm{BarrierFree}(W)\Rightarrow \mathrm{BarrierFree}(W')\Big).$$
:::

::: proof
*Proof.* Take the empty world $W_0=\varnothing$, which is barrier-free. In this model, two concrete classes are shape-equivalent (same observable profile) but nominally distinct, so adding them yields an extension $W_1\supseteq W_0$ that is not barrier-free. Therefore barrier-freedom is not extension-stable. ◻
:::

::: remark
The theorem formalizes a basic fact about open-world systems: present collision-freedom does not by itself yield a stable future guarantee. Persistent zero-error claims therefore need an explicit growth model rather than a one-time snapshot argument.
:::

The running example offers the same intuition in finite form: the realized world $\{A,C,D,E\}$ is collision-free, but adding class $B$ immediately restores the collision block $\{A,B\}$.

For compression system design, the point is direct: a representation that currently needs no auxiliary identification bits can require a positive residual coding budget after extension if new classes enter existing fibers. Thus open-world growth can change the optimal exact side-information rate even when nothing about the deployed decoder changes.

::: definition
For a partial observer generator $f:\mathbb{N}\rightharpoonup\mathbb{N}$, define $$\mathrm{HasBarrier}(f)\;:\Leftrightarrow\;\exists x\neq y,\exists z,\ z\in f(x)\land z\in f(y).$$ Define $\mathrm{BarrierFree}(f):\Leftrightarrow \neg\mathrm{HasBarrier}(f)$.
:::

::: theorem
[]{#thm:rice-barrier label="thm:rice-barrier"} There is no computable predicate on program codes deciding whether the generated observer has a barrier: $$\neg\mathrm{ComputablePred}\!\left(c\mapsto \mathrm{HasBarrier}(\mathrm{eval}(c))\right).$$ Equivalently, barrier-freedom certification is also non-computable.
:::

::: proof
*Proof.* Assume there is a computable certifier for $P(c):=\mathrm{HasBarrier}(\mathrm{eval}(c))$. The property is non-trivial: some generators produce collisions and some do not. Rice's theorem therefore implies that if membership in the corresponding semantic class were computable from code, every partial recursive function would have to satisfy it, contradiction. The barrier-free statement is equivalent by negation. ◻
:::

::: remark
The Rice-style theorem is the algorithmic counterpart to extension instability. Even when a current deployment appears collision-free, there is no general computable procedure that certifies this property for arbitrary future-generating code. Formal guarantees therefore require either a restricted extension model or explicit identity-carrying information that remains valid under growth.
:::

::: corollary
For unrestricted open-world generators, one cannot computably certify that attribute-only identification remains barrier-free under all future extensions. Consequently, persistent $D=0$ guarantees cannot rely on "currently collision-free" attribute structure alone.
:::


The main theorem chain is complete for the deterministic finite side-information model studied here. This section records only brief forward-looking directions and is not part of the core theorem arc.

Each extension changes one part of the model while preserving the paper's central question: once the representation is noninjective, how much additional information is needed for exact or approximate recovery? The common theme is that residual ambiguity left by the representation must still be paid for in some currency---side-information bits, repeated evidence, allowable distortion, or additional disclosure.

## Noisy and Lossy Observation

With noisy representations or noisy side channels, repeated sampling can reduce error, so the natural problem becomes the tradeoff among side-information rate, repeated evidence, and target error level. In this regime, the deterministic fibers are replaced by probabilistic confusion neighborhoods, and the main open question is how the exact fiber laws relax into probabilistic coding theorems. Once the deterministic representation assumption is relaxed, the confusability structure is no longer a disjoint union of cliques, and the problem moves closer to the richer graph-entropy regime studied by Körner and Witsenhausen.

## Semantic Distortion Measures

The paper treats $D$ as binary misidentification probability. A lossy extension would replace that with hierarchical distortion, weighted misclassification penalties, or other task-specific semantic losses. The natural open problem is then to characterize how the side-information rate changes when near misses and catastrophic confusions are not penalized equally, so that the fiber geometry interacts with a distortion matrix rather than a binary error indicator.

This matters most in settings where classes carry internal geometry or hierarchy: the current theory distinguishes zero error from nonzero error, while a lossy theory would need to distinguish tolerable confusion from unacceptable confusion.

## Privacy, Security, and Three-Resource Geometry

Nominal tags also suggest privacy-preserving class-membership proofs and stronger verification mechanisms in model-registry settings. The intro and main results already record the finite explicit disclosure separation: nominal access has unit disclosure budget, while tag-free zero-error identity resolution can require $n-1$ primitive disclosures in the adversarial family (PRIV3). This is another facet of the same residual ambiguity: one may pay for exact identification with side-information bits or with additional disclosures. The extension question is what happens after one imposes an explicit leakage constraint and asks how much naming information may be exposed while still certifying identity or class membership.

More broadly, the present side-information viewpoint invites comparison with rate-distortion-perception formulations [@blau2019rethinking] and with conditional source coding more directly. The main missing theorem is a genuinely entropy-sensitive zero-error variable-length characterization.

In explicit finite registries, auditing residual ambiguity means computing fiber statistics exactly. In learned or implicit models, the same quantities become representation-analysis targets: one may estimate collision profiles through discretization bounds, quantized or empirical collision statistics, class-conditional entropy surrogates, or task-conditioned quantities such as $A_{\pi \to Y}$. The theoremic contribution of the present paper is to identify what these quantities mean once they are available, not to solve the computational problem of extracting them from arbitrary models.

The present distortion floor should therefore be read as a deterministic exact-recovery analogue of rate-distortion-perception constraints, not as a full solution of that stochastic theory.

## Learned Representations and Embedding Collapse

In learned classification systems, the observation map $\pi$ can be instantiated by an embedding or representation map whose coordinates are then queried or thresholded. The earlier sections isolate the finite explicit corollaries: one-symbol zero-error feasibility is equivalent to injectivity (GPH27), and refining the observation channel cannot increase collision multiplicity (GPH28). The extension question is how far that representation reading can be pushed once $\pi$ is implicit, high-dimensional, or learned from data. In that setting, the fixed-length and adaptive fiber laws become a formal measure of residual ambiguity under the representation: unresolved embedding collisions must be paid for either by explicit naming information or by accepting nonzero semantic error.


This paper studies exact class recovery from a noninjective semantic representation $U=\pi(C)$ when the encoder may transmit additional identity-resolving side information. The central primitive is residual ambiguity: once the representation is fixed, exact recovery depends only on how many classes remain unresolved inside its fibers. The contribution is intentionally both exact and unifying. The paper contributes exact coding laws for that residual ambiguity and shows that the same ambiguity-fiber object supports a broader representation-sufficiency theory.

On the theorem side, the main fixed-length law is exact: the minimum worst-case auxiliary description is determined by the largest representation collision fiber, giving the zero-error rate $\log_2 A_\pi$. The finite-block theorem shows that this budget scales exactly linearly across repeated uses of the same representation.

The paper also gives an average-case counterpart in a representation-adaptive model. When auxiliary description length may depend on the observed representation value, the pointwise optimal fiber budget is the ceiling logarithm of the fiber size, and the expected auxiliary rate is controlled fiber by fiber. The resulting lower and upper bounds place the adaptive problem around $H(C\mid U)$ up to the standard one-bit prefix-coding overhead. Thus the worst-case and adaptive theorems quantify the same residual ambiguity at different resolutions.

The broader payoff is representation sufficiency. The same ambiguity-fiber geometry governs when a representation is sufficient for exact downstream tasks, when helper views or retrieval-style augmentation genuinely help, and when modular or factorized latents add exact information rather than architectural redundancy. In the learned-representation reading, the interpretation is direct: injective representations support exact downstream identification with no extra metadata, while noninjective representations impose an irreducible side-information cost.

Query costs and robustness do not begin new stories; they record other ways the same unresolved ambiguity reappears when it is not paid for directly in side-information bits. One can transmit side information to resolve it, or instead build a sufficiently rich helper view out of primitive observables and pay for that view through interactive/query resources; in the structured-axis setting, the matroid law describes the geometry of such minimal sufficiency-restoring augmentations. One can also ask whether an apparently collision-free representation remains safe under world growth. These modalities are different ways of accounting for the same ambiguity left by a noninjective representation.

#### The neurosymbolic necessity theorem.

The central applied contribution is this: symbolic identity tags are information-theoretically necessary for zero-error open-world identity recovery. This is not a design preference or an architectural taste. It is a theorem. Semantic representation alone is lossy identity compression. When collisions exist, no decoder, regardless of architecture or training, can recover identities without additional bits. The pair consisting of a neural encoder and an injective symbolic handle achieves zero-error identity recovery if and only if the handle distinguishes entities within each semantic fiber (NSL1--NSL6). The identity bits must be paid somewhere, either in the encoder or in the symbolic handle. No free lunch.

## Acknowledgment: AI-use Disclosure {#acknowledgment-ai-use-disclosure .unnumbered}

Generative AI tools (including Codex, Claude Code, Augment, Kilo, and OpenCode) were used throughout this manuscript, across all sections (Abstract, Introduction, theoretical development, proof sketches, applications, conclusion, and appendix) and across all stages from initial drafting to final revision. The tools were used for boilerplate generation, prose and notation refinement, LaTeX and structure cleanup, translation of informal proof ideas into candidate formal artifacts (Lean and LaTeX), and repeated adversarial reviewer-style critique passes to identify blind spots and clarity gaps.

The author retained full intellectual and editorial control, including problem selection, theorem statements, assumptions, novelty framing, acceptance criteria, and final inclusion or exclusion decisions. No technical claim was accepted solely from AI output. Formal claims reported as machine-verified were admitted only after Lean verification (no `sorry` in cited modules) and direct author review; Lean was used as an integrity gate for responsible AI-assisted research. The author is solely responsible for all statements, citations, and conclusions.




---

## Machine-Checked Proofs

All theorems are formalized in Lean 4:
- Location: `docs/papers/paper1_typing_discipline/proofs/`
- Lines: 9166
- Theorems: 399
- `sorry` placeholders: 0
