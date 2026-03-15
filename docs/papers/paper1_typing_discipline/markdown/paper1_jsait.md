# Paper: Semantic Identity Compression: Exact Zero-Error Laws, Rate-Distortion, and Neurosymbolic Necessity

**Status**: JSAIT-ready | **Lean**: 13838 lines, 677 theorems

---

## Abstract

Symbolic systems operate over exact identities: variables denote specific objects, pointers target precise memory locations, and database keys refer to singular records. Neural embeddings generalize by compressing away semantic detail, but this compression creates collision ambiguity: multiple distinct entities can share the same representation value. We characterize exactly how much additional information must be supplied to recover precise identity from such representations.

The answer is controlled by a single combinatorial object: the collision-fiber geometry of the representation map $\pi$. Let $A_{\pi}=\max_u |\pi^{-1}(u)|$ be the largest collision fiber. We prove a tight fixed-length converse $L \ge \log_2 A_{\pi}$, an exact finite-block scaling law, a pointwise adaptive budget $\lceil \log_2 |\pi^{-1}(u)|\rceil$, and the rate-distortion tradeoff with an explicit distortion floor when identity bits are withheld. The same fiber geometry determines query complexity and canonical structure for distinguishing families.

Because this residual ambiguity is structural rather than representation-specific, symbolic identity mechanisms (handles, keys, pointers, nominal tags) are the necessary system-level complement to any non-injective semantic representation. All main results are machine-checked in Lean 4. **Keywords:** semantics-aware compression, zero-error coding, neurosymbolic systems, learned representations, side information


## Problem Statement

Symbolic systems are exact identity capable. Variables denote specific objects, pointers target precise memory locations, and database keys refer to singular records. This exactness is not incidental; it is fundamental to symbolic computation, which requires that distinct entities remain distinguishable whenever operations depend on their identity.

Neural embeddings operate differently. They generalize by compressing away semantic detail, mapping distinct inputs to the same representation when those inputs are deemed functionally equivalent. The benefit is generalization: the embedding captures similarity structure and supports efficient nearest-neighbor operations. The cost is collision ambiguity: multiple distinct entities can share the same embedding value, and the representation alone no longer identifies which entity is meant.

This paper asks: given a fixed embedding that has already been deployed, what is the exact price of recovering precise identity?

Formally, let $C$ be a finite class label, let $\pi$ be the representation map, and let $U = \pi(C)$ be the embedding value observed by the decoder. Exact identity recovery asks how much additional description is needed to recover $C$ from $U$. Once the problem is written this way, the central combinatorial quantity is immediate: $$A_{\pi} := \max_u |\pi^{-1}(u)|,$$ the size of the largest collision fiber. The worst-case fixed-length law is then exactly the minimum number of extra bits needed to separate that worst fiber: $$L \ge \log_2 A_{\pi}.$$

This viewpoint also yields a finer adaptive law: if the auxiliary description length may depend on the realized representation value $u$, then the optimal pointwise budget is $\lceil \log_2 |\pi^{-1}(u)|\rceil$ per fiber. On the query side, binary interrogation faces the same combinatorial floor, because a family of $d$ binary observables induces at most $2^d$ transcripts on any collision fiber. In this sense, worst-case bits, adaptive bits, query cost, and deterministic distortion are all different currencies for the same residual ambiguity object.

#### Regime clarification.

All results are stated relative to one admissibility contract: the decoder may use only the declared representation value $U$ and any auxiliary description whose length is charged to the budget. No hidden registry, preprocessing table, or external side channel may contribute uncharged distinguishing information. We study the finite, fixed-representation, zero-error regime formalized by this contract. This is the base case: any higher-level regime (stochastic, continuous, or learning-aware models) must, at deployment time, reduce to a fixed representation and finite realized classes in order to execute. Consequently, those higher-level analyses cannot escape the operational implications derived here; they only change the input quantities (for example, collision statistics) that our exact downstream laws take as given.

#### Systems consequence.

Symbolic handles, identifiers, keys, pointers, nominal tags, registry entries, and retrieval indices are the canonical mechanisms by which real systems pay the identity debt left by non-injective semantic abstraction.

## Relation to Prior Work

The closest classical objects are zero-error coding with confusability constraints [@korner1973coding; @witsenhausen1976zero], source coding with side information, and more recent task-oriented or semantics-aware communication models [@gunduz2023beyond]. Our setting isolates a different question: given a fixed deterministic representation already available at the decoder, what residual coding burden does it leave behind for exact identity recovery? That question admits exact finite answers.

The confusability graph in our setting is highly special: classes sharing one representation value form a clique, so the graph is a disjoint union of representation-fiber cliques. This structural collapse explains why the paper obtains exact closed forms that the general graph-entropy problem does not. The interpretation for neurosymbolic systems is direct: if $\pi$ is a learned embedding, then collision fibers measure the irreducible identity ambiguity left by semantic compression. Injective representations need no auxiliary metadata; non-injective representations impose an exact-recovery cost that no downstream reasoning layer can remove on its own [@koh2020concept; @garcez2020neurosymbolic].

Viewed through the semantics-aware communication lens, the fiber geometry of the meaning map $\pi$ is the exact governing object in the finite fixed-representation regime [@gunduz2023beyond]. All coding laws, query results, and distortion bounds follow from this single combinatorial invariant, making them precise building blocks for any semantics-aware analysis that operates under exact identity-recovery assumptions.

## Contributions

Main contribution. The collision fiber geometry induced by a fixed representation $\pi$ is the single structural object that governs exact identity-recovery costs. From this object flow several direct facets:

1.  Exact coding laws: a tight fixed-length converse $L \ge \log_2 A_{\pi}$, an exact finite-block scaling law, and a pointwise optimal adaptive bit budget $\lceil \log_2 |\pi^{-1}(u)|\rceil$ per fiber.

2.  Deterministic distortion currency: in the uniform single-block corner the optimal distortion is $D^\star(L)=\max(0,1-2^L/a)$, giving an explicit error floor when identity bits are withheld.

3.  Canonical query structure: the fiber-geometry framework extends naturally to the query side, where complete derivability-aware families have a canonical form. Every such family reduces to an orthogonal semantically minimal core, meaning non-orthogonal families are redundant overpresentations rather than rival regimes. On the canonical core, minimal distinguishing families form the bases of a matroid, and the query cost obeys the same counting floor $\lceil \log_2 A_{\pi}\rceil \le d$. This result establishes a single structural backbone that connects the three operational currencies.

4.  Systems payoff with mechanized certainty: symbolic handles are the canonical mechanism for paying the identity debt in real systems, and the main theorem chain is machine-checked in Lean 4.

#### Machine-checked proofs.

All principal theorems in this paper have been formalized and verified in Lean 4. The mechanization functions as an integrity gate: informal arguments were admitted into the paper only after the corresponding formal proofs compiled without `sorry`.

#### Worked example.

We use one small example throughout the paper. Five classes have profiles $$A=000,\quad B=000,\quad C=100,\quad D=110,\quad E=111.$$ Then $A$ and $B$ form a collision fiber of size two, so the exact worst-case fixed-length budget is one bit. The adaptive picture is equally simple: the fiber over profile $000$ needs one auxiliary bit, while the singleton fibers need none. In the learned-representation reading, this is the exact price of one residual collision in the representation.

## Paper Organization

The paper has one theoremic arc and several direct payoffs. Sections [\[sec:model\]](#sec:model){reference-type="ref" reference="sec:model"} and [\[sec:bits\]](#sec:bits){reference-type="ref" reference="sec:bits"} establish the exact coding backbone. Section [\[sec:queries\]](#sec:queries){reference-type="ref" reference="sec:queries"} shows that the query-side cost is canonical once redundant observables are collapsed. Section [\[sec:distortion\]](#sec:distortion){reference-type="ref" reference="sec:distortion"} converts the same ambiguity into a deterministic distortion floor and entropy-sensitive converses. Section [\[sec:systems\]](#sec:systems){reference-type="ref" reference="sec:systems"} explains why symbolic identity layers are therefore necessary in semantics-aware systems. Section [\[sec:robustness\]](#sec:robustness){reference-type="ref" reference="sec:robustness"} studies open-world fragility. Section [\[sec:conclusion\]](#sec:conclusion){reference-type="ref" reference="sec:conclusion"} then closes the paper. Section [\[sec:extensions\]](#sec:extensions){reference-type="ref" reference="sec:extensions"} records extensions and open directions, and Section [\[sec:conclusion\]](#sec:conclusion){reference-type="ref" reference="sec:conclusion"} concludes.


## The Abstract Coding Framework

The paper studies a finite class space together with a fixed representation map. The decoder observes the representation value, and the encoder may additionally send auxiliary description to resolve whatever ambiguity the representation leaves behind. The coding laws depend only on this representation map and its fibers, not on the mechanism that produced it.

::: definition
Let $C$ be a finite class variable, let $U=\pi(C)$ be the representation available to the decoder, and let $M$ be any auxiliary description transmitted by the encoder. The main theorem arc studies the minimum length of $M$ needed for exact recovery of $C$ from $(U,M)$, in both fixed-length and representation-adaptive regimes.
:::

::: definition
For a representation value $u$, define the corresponding fiber by $$\pi^{-1}(u) = \{c : \pi(c)=u\}.$$ The paper's central object is the geometry of these fibers.
:::

::: remark
No further structure is assumed. The map $\pi$ may arise from a learned encoder, a database schema projection, a feature-extraction pipeline, or a hand-designed attribute family. The theoremic contribution is generic in the representation map once its fibers are fixed.
:::

::: remark
The coding statements are written in terms of a class variable $C$ and representation $U=\pi(C)$. When convenient, we also use an underlying entity variable $V$ together with a class assignment map $C(V)$. Thus $U=\pi(V)$ is the representation of the realized entity, and the induced class label is $C(V)$. The two viewpoints are interchangeable; the coding arc should be read through the pair $(C,U)$.
:::

::: definition
[]{#def:admissibility-contract label="def:admissibility-contract"} All impossibility and optimality statements in this paper quantify over schemes whose evidence consists only of the declared representation value together with any explicit auxiliary description charged to the side-information budget. In particular, no hidden registry, reflection oracle, whole-universe preprocessing table, or amortized cross-instance cache may supply uncharged distinguishing information.
:::

This admissibility contract is the formal regime anchor for all impossibility and optimality claims in the paper. The contract that the decoder only has the declared representation and any explicitly charged auxiliary description (no hidden registries or undeclared side channels) is encoded in the mechanized development and thus mechanically enforced in the formal proofs.

## Representation-Induced Information Barrier

The basic obstruction is immediate: equal representations give the decoder the same side information. The next theorem records that invariant in generic form.

::: definition
A property $P:\mathcal C \to Y$ is *representation-computable* if there exists a function $f: \mathcal U \to Y$ such that $$P(c) = f(\pi(c))
\qquad\text{for all } c \in \mathcal C.$$ Equivalently, $P$ depends only on the representation value.
:::

::: theorem
[]{#thm:information-barrier label="thm:information-barrier"}[]{#thm:info-barrier label="thm:info-barrier"} Let $P: \mathcal C \to Y$ be any function. If $P$ is representation-computable, then $P$ is constant on every representation fiber: $$\pi(c_1)=\pi(c_2) \implies P(c_1)=P(c_2).$$ Equivalently, no observer restricted to the representation alone can compute a property that varies within one fiber.
:::

::: proof
*Proof.* Suppose $P$ is representation-computable via $f$, so $P(c)=f(\pi(c))$ for all $c$. If $\pi(c_1)=\pi(c_2)$, then $$P(c_1)=f(\pi(c_1))=f(\pi(c_2))=P(c_2).$$ ◻
:::

::: remark
The barrier is informational rather than computational: once the representation is identical, unlimited time or memory does not help. We state it explicitly because the later converse, adaptive coding, query, and robustness results all reuse the same invariance premise.
:::

::: corollary
[]{#cor:provenance-barrier label="cor:provenance-barrier"} If there exist classes $c_1,c_2$ with $\pi(c_1)=\pi(c_2)$ but $c_1 \neq c_2$, then class identity is not representation-computable.
:::

::: proof
*Proof.* Direct application of Theorem [\[thm:information-barrier\]](#thm:information-barrier){reference-type="ref" reference="thm:information-barrier"} to the identity map on classes. ◻
:::

::: theorem
[]{#thm:model-completeness label="thm:model-completeness"} Let $\alpha : \mathcal C \to \mathcal A$ be any declared observable state such that every admissible primitive observation factors through $\alpha$. Then every in-scope semantic property factors through $\alpha$: there exists $\tilde P$ with $$P(c)=\tilde P(\alpha(c)) \quad \text{for all } c\in\mathcal C.$$
:::

::: proof
*Proof.* If every admissible primitive observation factors through $\alpha$, then every admissible transcript factors through $\alpha$, and so does any output computed from that transcript. ◻
:::

::: corollary
[]{#cor:fixed-axis-incompleteness label="cor:fixed-axis-incompleteness"} If $\alpha(c_1)=\alpha(c_2)$ but $P(c_1)\neq P(c_2)$, then no admissible strategy can compute $P$ with zero error without adding new information beyond the declared observable state.
:::

::: remark
The sufficiency and insufficiency statements are the helper-view version of the information barrier: if a task varies inside one declared state fiber, that state is not sufficient and must be refined by additional information.
:::


## Fixed-Length Zero-Error Recovery

We now state the zero-error coding problem in generic representation-map form. Let $C$ be a finite class variable and let $U=\pi(C)$ be the representation observed by the decoder. If the representation is non-injective on classes, then the decoder cannot determine $C$ from $U$ alone; the unresolved ambiguity is exactly the ambiguity inside the fibers of $\pi$.

::: definition
Let $C$ be a random class label, let $U=\pi(C)$ be its representation, and let the decoder observe $U$ together with any charged auxiliary description supplied by the encoder.
:::

::: theorem
[]{#thm:identification-capacity label="thm:identification-capacity"} Let $\mathcal{C}$ be a finite class space and let $\pi : \mathcal{C} \to \mathcal{U}$ be a representation map. Zero-error recovery of $C$ from $U=\pi(C)$ with no auxiliary description is achievable if and only if $\pi$ is injective on classes.
:::

::: proof
*Proof.* *Achievability*: If $\pi$ is injective on classes, then observing $\pi(c)$ determines $c$ uniquely. The decoder simply inverts the class-to-representation map.

*Converse*: Suppose two distinct classes $c_1 \neq c_2$ share the same representation value: $\pi(c_1)=\pi(c_2)$. Then any decoder $g(U)$ outputs the same class label on both inputs, so it cannot be correct for both. Hence zero-error recovery without auxiliary description is impossible. ◻
:::

In the finite explicit setting, the same condition can be read directly from the representation channel: one-symbol zero-error feasibility is equivalent to injectivity (GPH27). Interpreting $\pi$ as a learned embedding, this is the exact threshold for zero-error class recovery from the representation alone.

::: remark
Under any distribution with positive mass on both colliding classes, $I(C;U) < H(C)$. This is an average-case consequence of the deterministic barrier above.
:::

::: remark
In the identification paradigm of [@ahlswede1989identification], the decoder asks "is the message $m$?" rather than "what is the message?" This yields double-exponential codebook sizes. Our setting is different: we require zero-error identification of the *class*, not hypothesis testing. The one-shot zero-error identification feasibility threshold is binary rather than asymptotic.
:::

## Fixed-Length Auxiliary Description

We now show that augmenting the representation with an auxiliary codeword restores exact recovery.

::: definition
An *auxiliary description* is a value $M(c)$ associated with each class $c \in \mathcal C$ and transmitted in addition to the representation value $\pi(c)$. In the fixed-length setting, the auxiliary description alphabet has size $2^L$.
:::

::: theorem
[]{#thm:tag-restored-sufficiency label="thm:tag-restored-sufficiency"} If $|\mathcal C|=k$, then an auxiliary description of length $L \geq \lceil \log_2 k \rceil$ bits suffices for zero-error identification, regardless of whether $\pi$ is class-injective.
:::

::: proof
*Proof.* A class-injective auxiliary description assigns a unique identifier to each class. Reading that description together with the representation determines the class exactly. ◻
:::

## Ambiguity Converse

The next theorem is the paper's main zero-error compression statement. It converts representation collision multiplicity directly into a necessary fixed-length side-information budget.

In graph terms, the fixed-representation setting is a structured zero-error special case: confusable classes are exactly those lying in the same representation fiber, so the confusability graph is a disjoint union of cliques. This special structure yields exact finite laws for the residual coding burden left by a representation.

::: definition
[]{#def:collision-multiplicity label="def:collision-multiplicity"} Let $\mathcal{C}$ be a finite class space and let $\pi : \mathcal{C} \to \mathcal U$ be a representation map. Define $$A_\pi := \max_{u \in \operatorname{Im}(\pi)} \left|\{c \in \mathcal{C} : \pi(c)=u\}\right|.$$ Thus $A_\pi$ is the size of the largest class-collision block under the representation.
:::

::: remark
Every implementable representation operates on finite-precision values. The representation alphabet is therefore finite, and $A_\pi$ is computable by fiber enumeration. If the deployed representation uses $M$ output values, then the pigeonhole principle yields the lower bound $\lceil |\mathcal C|/M \rceil \le A_\pi$. Conversely, to guarantee $A_\pi \le k$ on an entity set of size $n$, at least $\lceil n/k \rceil$ distinct representation values are necessary. The coding theorems identify the operational meaning of $A_\pi$ once the representation is fixed at its deployed precision.
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
The converse is a lossless coding law for semantic identity [@csiszar2011information]. The relevant confusable alphabet is the largest representation-collision block, and zero-error separation of that block requires exactly enough auxiliary outcomes to name its members. The theorem therefore has the same fixed-length form as a classical zero-error side-information budget, but the governing alphabet size is induced by the representation rather than by the ambient class set alone.
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

## Why the General Zero-Error Theory Simplifies Here

The confusability graph induced by a fixed deterministic representation has special structure: two classes are adjacent exactly when they share the same representation value. The next observations explain why several general zero-error graph quantities collapse in this setting.

::: proposition
Let $\pi : \mathcal C \to \mathcal U$. Define $c_1 \sim_\pi c_2$ iff $\pi(c_1)=\pi(c_2)$ and $c_1 \ne c_2$. Then $\sim_\pi$ is transitive on distinct classes inside a fiber.
:::

::: proof
*Proof.* If $\pi(c_1)=\pi(c_2)$ and $\pi(c_2)=\pi(c_3)$, then $\pi(c_1)=\pi(c_3)$ by equality. Thus any two distinct classes linked through one fiber are directly confusable as well. ◻
:::

::: proposition
The confusability graph $G_\pi$, with edges between distinct classes sharing the same representation, is a disjoint union of cliques. Each connected component is exactly one representation fiber.
:::

::: proof
*Proof.* Each fiber is a clique because all of its classes share the same representation value. No edge connects distinct fibers by construction. Hence each connected component is exactly one fiber, and the graph is a disjoint union of cliques. ◻
:::

::: corollary
For a representation-induced cluster graph, the standard one-shot zero-error parameters collapse to fiber counts, so the fixed-length auxiliary-description law is governed exactly by the largest fiber.
:::

::: proof
*Proof.* In a disjoint union of cliques, proper coloring, clique size, and independence counting reduce to choosing or naming one element per component. In this setting, the general graph inequalities collapse to the same fiber-counting law. ◻
:::

In the running example, $A_\pi = 2$, so the exact block law gives $L_t^\star = t$ bits.


## Representation-Adaptive Side Information

The fixed-length theorem pays for the worst collision fiber through $A_\pi$. A finer coding law allows the auxiliary description length to depend on the observed representation value $u$. Then the relevant budget is not the maximum fiber size but the size of the individual fiber $\pi^{-1}(u)$. The same residual ambiguity object is now resolved fiber by fiber and then averaged under the induced law of $U$.

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
[]{#thm:fiberwise-prefix-coding-upper-bound label="thm:fiberwise-prefix-coding-upper-bound"} Assume standard prefix-coding achievability on each finite fiber. Then there exists an adaptive zero-error auxiliary code satisfying $$\mathbb E[\ell(U)] \le \frac{H(C\mid U)}{\log 2} + 1.$$
:::

::: proof
*Proof.* Because the decoder already knows $U=u$, prefix-freeness is required only within each fiber $\pi^{-1}(u)$. Applying a standard prefix-coding theorem separately to the conditional source on each nonempty fiber gives an expected conditional code length at most $H(C\mid U=u)/\log 2 + 1$. Averaging over $u$ yields the stated bound. ◻
:::

::: corollary
[]{#cor:zero-error-conditional-entropy-sandwich label="cor:zero-error-conditional-entropy-sandwich"} The optimal expected zero-error auxiliary rate lies between $$\frac{H(C\mid U)}{\log 2}
\qquad\text{and}\qquad
\frac{H(C\mid U)}{\log 2}+1.$$
:::

::: corollary
If, after observing $U$, the decoder is allowed to ask adaptive binary clarification questions whose leaves identify the class exactly, then the minimum expected number of clarification questions is bounded between $$\frac{H(C\mid U)}{\log 2}
\qquad\text{and}\qquad
\frac{H(C\mid U)}{\log 2}+1.$$
:::

::: corollary
Let $T$ be any repeated or noisy observation transcript whose realized value factors through the fixed representation, that is, $T = h(U)$ for some map $h$. Then recovering $C$ from $T$ cannot require less worst-case or pointwise adaptive side information than recovering $C$ from $U$ itself.
:::

::: remark
The fixed-length law pays for the worst collision fiber; the adaptive law pays for each fiber separately and averages under the source-induced law of $U$. These are two resolutions of the same object: the cost of naming the classes that remain collapsed inside each representation fiber.
:::

::: remark
The conditional-entropy lower and upper bounds place the adaptive theorem inside classical source-coding language. Zero-error adaptive side information is sandwiched around $H(C\mid U)$ up to the standard one-bit prefix-coding overhead, but the exact zero-error law remains fiberwise and combinatorial rather than purely entropic.
:::


## The Query Currency

The fiber geometry that governs fixed-length and adaptive coding costs extends naturally to the query side. Once the collision structure is understood, the canonical form of complete derivability-aware query families follows directly. This section establishes that structural result: every such family reduces to an orthogonal semantically minimal core, and the set of such cores forms a matroid.

Bits are one way to pay the identity debt left by a non-injective representation. Another is to interrogate primitive observables until the class is determined. This section studies that query currency. The same residual ambiguity that forces auxiliary bits can also be paid for through interactive disambiguation.

::: definition
Let $\mathcal Q$ be the family of primitive queries available to an observer. For a classification system with attribute set $\mathcal I$, we write $\mathcal Q = \{q_I : I \in \mathcal I\}$ where $q_I(v)=1$ iff $v$ satisfies attribute $I$.
:::

::: definition
A subset $S \subseteq \mathcal Q$ is *distinguishing* if, for all values $v,w$ with $\mathrm{class}(v) \ne \mathrm{class}(w)$, there exists $q \in S$ such that $q(v) \ne q(w)$.
:::

::: definition
[]{#def:distinguishing-dimension label="def:distinguishing-dimension"} The *minimum distinguishing number* is $$d := \min\{|S| : S \subseteq \mathcal Q \text{ is distinguishing}\}.$$ Equivalently, $d$ is the smallest number of primitive queries that suffices for zero-error separation.
:::

::: remark
These notions differ: an inclusion-minimal distinguishing family may be larger than a minimum-cardinality one. The operational lower bound below depends only on the quantity $d$.
:::

In the running example from Section [\[sec:bits\]](#sec:bits){reference-type="ref" reference="sec:bits"}, $\{q_1,q_2,q_3\}$ is distinguishing and no pair of primitive queries is distinguishing, so the minimum distinguishing number is $d=3$.

::: theorem
[]{#thm:interface-lower-bound label="thm:interface-lower-bound"} For attribute-only observers, zero-error class identity checking requires $$W(\text{class-identity}) = \Omega(d)$$ in the worst case, where $d$ is the minimum distinguishing number.
:::

::: proof
*Proof.* Assume a zero-error attribute-only procedure halts after fewer than $d$ queries on every execution path. Fix one execution path and let $Q \subseteq \mathcal I$ be the set of queried attributes on that path, so $|Q|<d$. By definition of $d$, no set of size less than $d$ is distinguishing. Hence there exist values $v,w$ from different classes with identical answers on all attributes in $Q$. An adversary can answer the procedure's queries consistently with both $v$ and $w$, so the transcript and output coincide on two different classes, contradicting zero-error identification. ◻
:::

::: corollary
[]{#cor:attribute-only-witness-lb label="cor:attribute-only-witness-lb"} For any attribute-only observer, $W(\text{class-identity}) \ge d$.
:::

This is the operational query law in full generality. It says how much interrogation is needed when the representation leaves ambiguity unresolved and no explicit identity bits are transmitted.

## Counting Bridge to Bit Currency

There is also a direct quantitative bridge back to the compression side. A family of $d$ binary queries induces at most $2^d$ distinct answer transcripts on any fixed collision fiber. Therefore no binary query family can distinguish a fiber of size $m$ unless $m \le 2^d$. In particular, if a worst fiber has size $A_{\pi}$, then any binary distinguishing family satisfies $$\left\lceil \log_2 A_{\pi} \right\rceil \le d.$$ This is the query-side analogue of the fixed-length converse: binary interrogation cannot beat the same counting barrier that governs auxiliary bits.

::: proposition
[]{#prop:binary-query-counting-bound label="prop:binary-query-counting-bound"} Let $S$ be a binary query family of size $d$ that distinguishes a finite set $T$ of classes. Then $$|T| \le 2^d.$$ Equivalently, $$\left\lceil \log_2 |T| \right\rceil \le d.$$
:::

::: proof
*Proof.* Each class in $T$ produces one binary transcript in $\{0,1\}^d$. Distinguishability means that the transcript map is injective on $T$. There are only $2^d$ binary transcripts, so injectivity forces $|T| \le 2^d$. The ceiling-log form is the exact converse of this counting bound. ◻
:::

::: corollary
[]{#cor:worst-fiber-query-lower-bound label="cor:worst-fiber-query-lower-bound"} If a binary query family distinguishes every class inside a worst collision fiber, then $$\left\lceil \log_2 A_{\pi} \right\rceil \le d.$$
:::

::: proof
*Proof.* Apply Proposition [\[prop:binary-query-counting-bound\]](#prop:binary-query-counting-bound){reference-type="ref" reference="prop:binary-query-counting-bound"} to a fiber attaining the maximal size $A_{\pi}$. The induced bound is exactly the stated ceiling-log inequality. ◻
:::

## Canonical Reduction to the Orthogonal Core

The right structural invariant is not the size of an arbitrary raw query family, because raw families may contain semantically derivable redundancy. The source Lean formalization proves a stronger and cleaner fact: every semantically complete derivability-aware query family reduces to an orthogonal semantically minimal core. Non-orthogonal families are therefore redundant overpresentations, not a rival regime.

::: definition
A *derivability-aware axis presentation* specifies primitive observables together with a derivability relation recording when one observable can be recovered from another. A family is *orthogonal* if no primitive observable is derivable from a distinct primitive observable in the family.
:::

::: proposition
[]{#prop:query-redundancy label="prop:query-redundancy"} If a semantically complete derivability-aware query family is non-orthogonal, then some primitive observable can be removed without destroying semantic completeness.
:::

::: proof
*Proof.* If the family is non-orthogonal, some observable $a$ is derivable from a distinct observable $b$ already in the family. Any query requirement satisfied via $a$ can therefore be satisfied via $b$ instead. Erasing $a$ preserves semantic completeness, so $a$ was redundant. ◻
:::

::: proposition
[]{#prop:query-minimal-core label="prop:query-minimal-core"} Every semantically complete derivability-aware query family contains a semantically minimal complete subfamily.
:::

::: proof
*Proof.* The family is finite. If it is not already semantically minimal, Proposition [\[prop:query-redundancy\]](#prop:query-redundancy){reference-type="ref" reference="prop:query-redundancy"} removes a redundant observable while preserving completeness. Repeating this elimination process must terminate, producing a semantically minimal complete subfamily. ◻
:::

::: proposition
[]{#prop:query-orthogonal-core label="prop:query-orthogonal-core"} Every semantically complete derivability-aware query family contains an orthogonal semantically minimal complete subfamily.
:::

::: proof
*Proof.* Choose a semantically minimal complete subfamily using Proposition [\[prop:query-minimal-core\]](#prop:query-minimal-core){reference-type="ref" reference="prop:query-minimal-core"}. If that subfamily were still non-orthogonal, Proposition [\[prop:query-redundancy\]](#prop:query-redundancy){reference-type="ref" reference="prop:query-redundancy"} would remove another observable while preserving completeness, contradicting minimality. Hence the minimal core is orthogonal. ◻
:::

These three propositions identify the canonical regime of the query problem. One starts from an arbitrary complete family, removes derivable redundancy until no further reduction is possible, and arrives at an orthogonal minimal core. The clean query invariant lives on that core [@Welsh1976]. The remaining gap above the counting floor, if any, is then a genuine structural limitation of the available primitive observables rather than an artifact of redundant presentation.

::: theorem
[]{#thm:matroid-bases label="thm:matroid-bases"} For the canonical orthogonal core of a complete derivability-aware query system, the semantically minimal distinguishing families are the bases of a matroid. Consequently, all such families have equal cardinality.
:::

::: proof
*Proof.* By Proposition [\[prop:query-orthogonal-core\]](#prop:query-orthogonal-core){reference-type="ref" reference="prop:query-orthogonal-core"}, every complete derivability-aware query system contains an orthogonal semantically minimal core. On an orthogonal core, exchange holds (L4, L5). Standard matroid theory then implies that the minimal complete families are exactly the bases of a matroid and therefore all have the same cardinality (L1). ◻
:::

::: remark
Arbitrary overcomplete binary query families need not themselves be matroidal; a finite counterexample exists (GPH31). That does not refute the canonical query invariant. It shows only that redundant raw presentations can obscure the primitive structure. After derivable redundancy is stripped away, the orthogonal minimal core is matroidal.
:::

::: remark
The lower bound $\lceil \log_2 A_{\pi} \rceil \le d$ is exact whenever the canonical query family realizes the full binary transcript capacity on the relevant worst fibers. In that case the query side meets the same counting floor as the fixed-length bit side. Any strict gap above the floor is an expressivity gap of the observable family, not a redundancy artifact. The corresponding equality and gap statements are mechanized in GPH60 through GPH64.
:::

## Witness Cost Comparison

::: {#tab:witness-comparison}
  -----------------------------------------------------------------------------------------------
  **Observer Class**    **Witness Procedure**                           **Witness Cost $W$**
  --------------------- ----------------------------------------------- -------------------------
  Nominal-tag           Single tag read                                 $O(1)$

  Attribute-only        Query a distinguishing family of minimum size   $\Omega(d)$

  Canonical axis core   Query any matroid basis of the core             exact basis cardinality
  -----------------------------------------------------------------------------------------------

  : Witness cost for class identity by observer class.
:::

::: remark
The bit law measures the residual description cost of restoring identity. The query law measures the residual interrogation cost. In the raw attribute model, that cost is lower-bounded by both the operational quantity $d$ and the counting floor $\lceil \log_2 A_{\pi} \rceil$. In the canonical derivability-aware presentation, the same cost is certified by matroid basis cardinality on the orthogonal core. In both readings, the ambiguity being paid for is the same.
:::


## Distortion as a Currency

#### Currency comparison.

The counting and entropy converses above show that auxiliary bits, queries, and accepted distortion are alternative currencies for the same residual ambiguity. This section makes one exchange explicit: how many bits must be paid to drive distortion to zero on a collision fiber, and what floor remains when those bits are withheld.

The general law is fiberwise. For arbitrary finite sources [@CoverThomas2006; @berger1971rate], the optimal recoverable mass decomposes exactly across representation fibers. The clean closed form arises in the uniform single-block corner, where the collision geometry alone determines the full budget-distortion tradeoff.

## Finite Fiberwise Decomposition {#sec:fiberwise-decomposition}

The adaptive and fixed-length theorems above quantify the residual ambiguity in each representation fiber separately. We now state the structural law that governs the global rate-distortion problem: for any source distribution and any finite set of fibers, the globally optimal recoverable mass decomposes exactly into fiberwise contributions, and this optimum is attained by independent per-fiber selection.

This is the main structural theorem of the rate-distortion arc. It applies to arbitrary finite sources and arbitrary observation maps, not only to uniform sources on single collision blocks.

::: definition
Fix a uniform per-fiber tag alphabet of size $T$. A subset $S \subseteq \mathcal C$ is *fiber-budget feasible* if every representation fiber contains at most $T$ elements of $S$: $$|S \cap \pi^{-1}(u)| \le T \quad \text{for all } u.$$
:::

::: definition
For each representation value $u$, define the *fiberwise top mass* $$M^\star(u, T) := \max\left\{\sum_{c \in S} \mu(c) : S \subseteq \pi^{-1}(u),\ |S| \le T\right\}.$$ This is the maximum source mass recoverable inside fiber $u$ using at most $T$ tag values.
:::

::: theorem
[]{#thm:fiberwise-decomposition label="thm:fiberwise-decomposition"} Let $\mu$ be any finite source distribution and let $T$ be a uniform per-fiber tag budget. Then the global optimum under the fiber-budget constraint is the sum of the fiberwise optima: $$M^\star_{\mathrm{global}}(T) = \sum_u M^\star(u, T).$$ Moreover, any feasible subset $S$ satisfies $$\mu(S) \le \sum_u M^\star(u, T).$$
:::

::: proof
*Proof.* Any feasible subset decomposes into its fiber slices, each bounded by its fiberwise optimum. Summing over fibers gives the upper bound. Conversely, choosing a mass-maximizing subset of size at most $T$ independently in each fiber yields a globally feasible subset attaining equality. ◻
:::

::: corollary
[]{#cor:fiberwise-attainment label="cor:fiberwise-attainment"} There exists a globally feasible subset $S^\star$ that attains the decomposition bound: $$\mu(S^\star) = \sum_u M^\star(u, T).$$
:::

::: proof
*Proof.* Take $S^\star = \bigcup_u S^\star_u$ where $S^\star_u$ is a maximizing subset in fiber $u$. The union is feasible because each fiber contributes at most $T$ elements, and the mass is the sum of the fiberwise maxima. ◻
:::

::: remark
The decomposition theorem applies to any finite source distribution, including uniform, Zipf, power-law, and arbitrary. The uniform single-block formula derived below is the special case where (i) all source mass lies in one fiber and (ii) the mass is uniformly distributed. Neither assumption is required for the decomposition itself. In this sense, the decomposition is the general zero-error rate-distortion law for the fixed-representation setting; the closed-form curve is its cleanest instantiation.
:::

::: remark
The decomposition is analogous to coding independent sub-sources separately: each fiber acts like an independent sub-problem, and the global optimum is the sum of the sub-problem optima. The sub-sources are induced by the representation partition rather than independent a priori.
:::

## The Uniform Single-Block Corner

The fiberwise decomposition theorem reduces the general problem to independent per-fiber optimization. When the source is uniform on a single collision block, this optimization collapses to a simple closed form. The following theorem is therefore a corollary of the structural law above, specialized to the uniform corner.

::: theorem
[]{#thm:semantic-identity-rate-distortion label="thm:semantic-identity-rate-distortion"} Let $G \subseteq \mathcal{C}$ be a collision block of size $a$, so all classes in $G$ have the same representation profile. Assume the source class $C$ is uniform on $G$. Then the optimal distortion under a fixed-length auxiliary budget of $L$ bits is exactly $$D^\star(L)=\max\!\left(0, 1 - \frac{2^L}{a}\right).$$ Equivalently, every fixed-length scheme with auxiliary budget $L$ bits satisfies $$\Pr[\hat C = C] \le \frac{2^L}{a},
\qquad
D = \Pr[\hat C \neq C] \ge 1 - \frac{2^L}{a}.$$ If $2^L \ge a$, then zero distortion is achievable; if $2^L < a$, then the lower bound is tight.
:::

::: proof
*Proof.* Fix a realization of all internal randomness used by the scheme. Because every class in $G$ has the same representation profile, the decoder's side information is the same on all classes in $G$; on that block, the decoder can therefore distinguish classes only through the auxiliary outcome. With $L$ bits there are at most $2^L$ outcomes, and for each such outcome the decoder can be correct on at most one class in $G$. Hence, for the fixed randomness, the scheme is correct on at most $2^L$ of the $a$ classes in $G$, so its success probability under the uniform distribution on $G$ is at most $2^L/a$. Averaging over the scheme's randomness preserves the same bound. The distortion lower bound is the complement of the success bound.

If $2^L \ge a$, assign distinct auxiliary outcomes to the classes in $G$ and decode injectively; this gives zero distortion. If $2^L < a$, choose $2^L$ classes in $G$, assign them distinct auxiliary outcomes, and map all remaining classes to arbitrary auxiliary outcomes. Then the decoder is correct on exactly $2^L$ of the $a$ classes, so the success probability is $2^L/a$ and the distortion is exactly $1-2^L/a$. This matches the converse. ◻
:::

::: corollary
[]{#cor:zero-error-threshold-distortion label="cor:zero-error-threshold-distortion"} Under the hypotheses of Theorem [\[thm:semantic-identity-rate-distortion\]](#thm:semantic-identity-rate-distortion){reference-type="ref" reference="thm:semantic-identity-rate-distortion"}, zero distortion is achievable if and only if $$L \ge \log_2 a.$$
:::

::: proof
*Proof.* By Theorem [\[thm:semantic-identity-rate-distortion\]](#thm:semantic-identity-rate-distortion){reference-type="ref" reference="thm:semantic-identity-rate-distortion"}, zero distortion is achievable exactly when $D^\star(L)=0$, equivalently when $2^L \ge a$, which is the same as $L \ge \log_2 a$. ◻
:::

::: corollary
[]{#cor:zero-budget-error-floor label="cor:zero-budget-error-floor"} If $L=0$ and the source is uniform on a collision block of size $a$, then any scheme relying only on the representation satisfies $$D \ge 1 - \frac{1}{a}.$$ In particular, if the source is uniform on a worst collision block with $a=A_\pi$, then $$D \ge 1 - \frac{1}{A_\pi}.$$
:::

::: proof
*Proof.* Set $L=0$ in Theorem [\[thm:semantic-identity-rate-distortion\]](#thm:semantic-identity-rate-distortion){reference-type="ref" reference="thm:semantic-identity-rate-distortion"}. Since $2^0=1$, the distortion lower bound becomes $1-1/a$. Choosing a worst collision block gives the second statement. ◻
:::

::: remark
This corollary quantifies the price of relying exclusively on semantic attributes. If one chooses not to store the auxiliary identifier required by the fixed-length coding law, the saved $\lceil \log_2 A_\pi \rceil$ bits are exchanged for a nonzero error floor. On a collision fiber of size $2$, omitting the one-bit tag forces an error rate of at least $50\%$. On a collision fiber of size $100$, omitting the seven-bit tag forces an error rate of at least $99\%$. One pays for identity either in stored bits or in distortion. There is no third option in which identity is both free and exact.
:::

::: remark
The steepness is easy to see numerically. On a collision block of size $100$, zero distortion requires at least $\lceil \log_2 100 \rceil = 7$ bits. With only $6$ bits the distortion is at least $36\%$, and with only $5$ bits it is at least $68\%$. Small shortfalls below the exact-recovery threshold can therefore make identity accuracy unusable in practice. If a system declines to store the identity bits needed to separate collided fibers, the penalty is a concrete minimum error rate, not abstract suboptimality.
:::

::: remark
The exact formula above is stated for the uniform source on a single collision block. That restriction is intentional. It isolates a deterministic corner in which collision geometry alone determines the full tradeoff. For non-uniform sources on the same block, or for mixtures across multiple fibers, the optimal distortion curve depends on the source law as well as the fiber sizes; in those settings one expects a different shape, closer in spirit to entropy-weighted side-information bounds than to the uniform worst-block law treated here. The point of the present theorem is exact isolation of the geometry-only effect.
:::

## Finite Entropy Converses {#sec:entropy-converses}

The counting-based converses above are combinatorial: they bound distortion in terms of fiber sizes and tag budgets. A complementary family of converses connects distortion to information-theoretic quantities [@CoverThomas2006; @csiszar2011information], placing the semantic identity problem inside classical Shannon theory.

The following theorems are mechanized in the RDC series, from RDC1 through RDC9. They apply to any finite source, any observation map, and any tag/decoder pair.

::: theorem
[]{#thm:fano-converse label="thm:fano-converse"} Let $\mu$ be a finite source on $K$ classes, let $(O, T)$ be the observation and tag alphabet sizes, and let $D$ be the distortion (error probability) of any deterministic decoder. Then $$H_\mu(C) \le h_b(D) + (1-D) \log(O \cdot T) + D \log(K - 1),$$ where $H_\mu(C)$ is the source entropy and $h_b$ is binary entropy.
:::

::: proof
*Proof.* Partition the source into success and failure sets. The success set has cardinality at most $O \cdot T$ (injection into observation/tag pairs), and the failure set has cardinality at most $K-1$. Applying the entropy bound for each partition and combining via the chain rule yields the stated inequality. The mechanization follows the standard Fano argument adapted to the finite budgeted setting. ◻
:::

::: corollary
[]{#cor:budget-from-error label="cor:budget-from-error"} Under the hypotheses of Theorem [\[thm:fano-converse\]](#thm:fano-converse){reference-type="ref" reference="thm:fano-converse"}, if $D < 1$ then $$H_\mu(C) - h_b(D) - D \log(K-1) \le \log(O \cdot T).$$
:::

::: proof
*Proof.* Rearrange Theorem [\[thm:fano-converse\]](#thm:fano-converse){reference-type="ref" reference="thm:fano-converse"}. The left-hand side is the "entropy gap" that must be absorbed by the observation/tag budget. ◻
:::

::: theorem
[]{#thm:min-entropy-bound label="thm:min-entropy-bound"} For observation-only schemes (no tags, $T=1$), any decoder with error probability at most $\varepsilon < 1$ satisfies $$H_\infty(\mu) \le -\log(1 - \varepsilon),$$ where $H_\infty(\mu)$ is the min-entropy of the source.
:::

::: proof
*Proof.* The success probability is bounded by the maximum atom mass times the budget (unity, for observation-only). Rearranging and taking logarithms gives the min-entropy bound. ◻
:::

::: remark
The counting converse $2^L \ge a$ is sharp in the zero-error regime but silent about source structure. The entropy converses fill this gap: for high-entropy sources the entropy bound may be tighter, while for low-entropy sources the counting bound dominates. Together they provide complementary views of the same fiber geometry.
:::


The formal and coding results lead to a practical corollary: if a representation $\pi$ is non-injective, then no purely semantic pipeline can guarantee zero-error identity recovery. The identity debt must be paid in one of three currencies: an injective encoder, explicit identity metadata of length at least $\log_2 A_\pi$ in the worst case, or a nonzero distortion floor. In deployed systems the usual payment mechanisms are symbolic handles, such as unique identifiers, pointers, retrieval keys, primary keys, and registry entries.

Example. In a retrieval system with $100$ documents sharing the same embedding, omitting the $\lceil \log_2 100 \rceil = 7$ identifier bits leaves a distortion floor of $1-1/100 = 99\%$ on that collision fiber. Even one bit below the exact-recovery threshold is costly: with only $6$ bits the distortion is still at least $36\%$.

## Learned Representations and Formal Instantiations

The abstract coding framework applies to any representation map. In the learned-representation setting, semantic information is what the encoder exposes, while identity information is the exact class label or entity name that may still need to be carried explicitly. Zero-error recovery from semantic information alone is possible exactly when the representation is injective on identity classes. When it is not, the collision fibers quantify the irreducible identity residue that must be restored by explicit metadata. This transfers directly to neurosymbolic systems: a neural encoder supplies the semantic side, and any symbolic layer that needs exact identity must be given the missing distinguishing information explicitly.

#### What this means for ML systems.

The coding laws have immediate consequences for learned-representation systems.

-   **Concept bottleneck models.** If the intermediate concept representation has collisions, then no downstream classifier can distinguish the collapsed concepts without extra identity information. Exact concept-level decisions therefore require an added disambiguation channel, retrieval key, or explicit symbolic concept identifier.

-   **Retrieval-augmented systems.** The retrieval index pays the identity-bit budget. If two artifacts have the same embedding, the system must preserve distinguishing metadata elsewhere, such as document identifiers, source pointers, or exact lookup keys. Otherwise exact retrieval is impossible on the collided fiber.

-   **Pure representation schemes and open-world deployments.** Systems that attempt to avoid any nominal layer do not escape the theorem. Any system that must identify entities exactly in an open world must either use an injective encoder or carry explicit identity metadata. There is no third option.

#### Attribute-only versus tag-augmented design.

Whenever $A_\pi>1$, the coding laws imply a sharp design dichotomy. In an attribute-only strategy, the system stores or transmits only the semantic representation $\pi(v)$. This pays only for semantic description, but it inherits the distortion floor forced by the collision fibers; in the zero-budget case, the worst collided block incurs error at least $1-1/A_\pi$ by Corollary [\[cor:zero-budget-error-floor\]](#cor:zero-budget-error-floor){reference-type="ref" reference="cor:zero-budget-error-floor"}. In a tag-augmented strategy, the system stores $\pi(v)$ together with auxiliary identity information. This increases storage by the exact identity budget, up to $\lceil \log_2 A_\pi \rceil$ bits in the worst case, but restores zero-error identity recovery.

This is why symbolic identifiers are ubiquitous in high-reliability systems. Primary keys, memory addresses, retrieval identifiers, and similar handles are the mechanisms by which systems avoid the distortion floor mandated by the coding laws. In settings where such handles are already available, the design lesson is stronger than mere feasibility.

**Corollary (Pareto domination).** In computational settings where nominal tags are effectively free (e.g., languages with inheritance, identifier systems, key-value stores), tag-augmented design strictly Pareto dominates attribute-only design for exact identity recovery: it achieves zero-error while attribute-only cannot, at no additional cost. See Table [\[tab:witness-comparison\]](#tab:witness-comparison){reference-type="ref" reference="tab:witness-comparison"} for the operational cost comparison. PD1

The mechanization sharpens this operational claim with an admissibility guarantee for the symbolic layer: in any valid execution trace, every registry read or write names a registry that was declared in the system specification. Thus the formal registry semantics is closed under the declared interface; undeclared registries are semantically inert and cannot appear in certified traces. This does not rule out arbitrary hidden state in real implementations, but it does show that the formal model does not smuggle in extra symbolic storage beyond the declared registry surface. NOH1

#### Formal instantiation.

The paper also develops a concrete formal identity domain in Lean. In that model, entities carry an explicit attribute family $\mathcal I$, primitive observables are the associated binary maps $q_I$, and the representation is the induced profile map $\pi(v)=(q_I(v))_{I\in\mathcal I}$. The generic coding laws then specialize to a machine-checked formal instance. In particular, equal observable profiles cannot distinguish identity (ACS8), provenance is not recoverable from the profile alone when collisions exist (ACS9), and nominal identity cannot be embedded back into the semantic layer (EMB1).

#### Instantiated neurosymbolic linking theorems.

Within that formal identity domain, the generic helper-view theorems specialize by setting the task to identity $Y = C$. The mechanization (NSL1 through NSL6) proves the following instantiated claims.

-   If the symbolic handle $\sigma$ is injective, the neurosymbolic link $(\pi, \sigma)$ achieves zero-error identity recovery (NSL1). This is the identity-task instantiation of the injectivity criterion GPH25.

-   If $\pi$ has collisions, semantic representation alone cannot achieve zero-error identity (NSL2), and no decoder can recover the collapsed identities without identity-resolving side information (NSL3).

-   The exact characterization is: $(\pi, \sigma)$ recovers identity if and only if $\sigma$ distinguishes all entities within each semantic fiber (NSL4). An injective handle therefore suffices (NSL5). This is the identity-task instantiation of the helper-view theorem GPH40.

-   The pair $(\pi, \sigma)$ is the canonical helper view for open-world identity systems (NSL6).

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

For learned representations, estimating quantities such as $A_\pi$, $A_{\pi \to Y}$, or the adaptive fiber profile is a representation-analysis problem rather than part of the theorem. In a finite explicit registry or index this is just fiber counting; in a large learned embedding one instead needs explicit collision enumeration after discretization or provable upper and lower bounds. The point of the present results is to identify what those quantities mean once they are known.

Concretely, the task-sufficiency theorem says the representation is sufficient precisely when every fiber carries one task label, and the helper-view and factorized laws say when added views contribute genuinely new exact information.

In the finite explicit setting, one-symbol zero-error feasibility is equivalent to injectivity and refinement cannot increase collision multiplicity (GPH27, GPH28). The same setting also yields a disclosure separation between nominal access and tag-free identity resolution (PRIV3).


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
The Rice-style theorem is the algorithmic counterpart to extension instability. Even when a current deployment appears collision-free, there is no general computable procedure that certifies this property for arbitrary future-generating code. For compression system designers deploying learned representations, this means that any zero-error recovery guarantee derived from a finite training corpus cannot be automatically relied upon to hold for new classes that the encoder will encounter in production; explicit identity metadata or a restrictive growth model becomes necessary to maintain formal guarantees.
:::

::: corollary
For unrestricted open-world generators, one cannot computably certify that attribute-only identification remains barrier-free under all future extensions. Consequently, persistent $D=0$ guarantees cannot rely on "currently collision-free" attribute structure alone.
:::


The main theorem chain identifies the structural laws governing zero-error identity recovery from fixed representations. This section records forward-looking directions that change one model assumption while preserving the central insight: residual ambiguity must be paid for in some currency.

## Scope and Completeness

The paper provides complete zero-error theory for the fixed-representation setting. The fiberwise decomposition theorem applies to any finite source distribution. The entropy converses apply to any observation map and tag budget. The query and stability laws apply to any attribute family. The uniform single-block closed form is the special case that collapses to an explicit formula.

Several natural extensions remain open. Stochastic representations, noisy channels, and asymptotic block analysis would each modify the model while preserving the core insight that fiber geometry governs residual cost.

## Noisy and Lossy Observation

With noisy representations or noisy side channels, repeated sampling can reduce error, so the natural problem becomes the tradeoff among side-information rate, repeated evidence, and target error level. In this regime, the deterministic fibers are replaced by probabilistic confusion neighborhoods, and the main open question is how the exact fiber laws relax into probabilistic coding theorems. Once the deterministic representation assumption is relaxed, the confusability structure is no longer a disjoint union of cliques, and the problem moves closer to the richer graph-entropy regime studied by Körner and Witsenhausen.

## Growth and Budgeting

The deterministic theorems also suggest operational extensions for systems that evolve over time. In a Poissonized growth model for representation cells, the probability that one cell has already developed a collision takes the explicit form $$1 - e^{-\lambda}(1+\lambda),$$ which isolates the onset of identity pressure as occupancy grows (GRC1). In particular, the collision pressure is strictly positive exactly when the arrival rate is positive (GRC3). On the budgeting side, the required exact tag budget becomes positive exactly when occupancy reaches two or more items in a cell (GTB1), and vanishes exactly when occupancy is at most one (GTB2). These formulas illustrate how the static collision laws induce dynamic trigger conditions for when auxiliary identity information becomes necessary.

Natural extensions. Important follow-on work studies how representation design and training shape the fiber geometry that our theorems take as input. Examples include (i) analysis of how training objectives change expected collision counts, (ii) quantization and embedding-dimension effects on empirical collision statistics, and (iii) continuous or asymptotic models that require careful discretization to connect with the finite explicit setting used here. Each such line of work supplies estimates or bounds on the fiber quantities (for example, $A_\pi$) which then determine the exact operational costs described in this paper (see BST1, BST2, BST3).

## Entropy-Sensitive Converses

The entropy-sensitive converses (RDC series) are now presented in the main text as Section [\[sec:entropy-converses\]](#sec:entropy-converses){reference-type="ref" reference="sec:entropy-converses"}. The mechanization supports Fano-style inequalities connecting distortion to source entropy (RDC1, RDC2, RDC5), budget lower bounds from error targets (RDC6, RDC7), and min-entropy bounds for observation-only schemes (RDC8, RDC9). These connect cleanly to the PMF bridge in the proof stack (PMF1, PMF2, PMF3), which identifies finite-source entropy quantities with explicit PMF entropies.

## Semantic Distortion Measures

The paper treats $D$ as binary misidentification probability. A lossy extension would replace that with hierarchical distortion, weighted misclassification penalties, or other task-specific semantic losses. The natural open problem is then to characterize how the side-information rate changes when near misses and catastrophic confusions are not penalized equally, so that the fiber geometry interacts with a distortion matrix rather than a binary error indicator.

This matters most in settings where classes carry internal geometry or hierarchy: the current theory distinguishes zero error from nonzero error, while a lossy theory would need to distinguish tolerable confusion from unacceptable confusion.

## Privacy, Security, and Three-Resource Geometry

Nominal tags also suggest privacy-preserving class-membership proofs and stronger verification mechanisms in model-registry settings. The intro and main results already record the finite explicit disclosure separation: nominal access has unit disclosure budget, while tag-free zero-error identity resolution can require $n-1$ primitive disclosures in the adversarial family (PRIV3). This is another facet of the same residual ambiguity: one may pay for exact identification with side-information bits or with additional disclosures. The natural extension is to study how imposing explicit leakage constraints changes the feasible naming information that can be exposed while still certifying identity or class membership.

More broadly, the present side-information viewpoint invites comparison with rate-distortion-perception formulations [@blau2019rethinking] and with conditional source coding more directly. The main missing theorem is a genuinely entropy-sensitive zero-error variable-length characterization.

In explicit finite registries, auditing residual ambiguity means computing fiber statistics exactly. In learned or implicit models, the same quantities become representation-analysis targets: one may estimate collision profiles through discretization bounds, quantized or empirical collision statistics, class-conditional entropy surrogates, or task-conditioned quantities such as $A_{\pi \to Y}$. The theoretic contribution of the present paper is to identify what these quantities mean once they are available; the computational task of extracting them from arbitrary models is a separate engineering and estimation problem.

The empirical audit path can also be connected back to the mechanization. Once a deployed representation and quantization rule are fixed, the resulting fiber histogram and finite budget curve can be exported as a Lean certificate whose checks reduce to the same finite formulas used in the paper: the reported $A_\pi$, the threshold budget $\lceil \log_2 A_\pi \rceil$, the worst-fiber floor, and the finite exact recoverable mass under each tag budget. This does not mechanize the neural encoder itself, but it does make the post-quantization collision audit machine-checkable.

The present distortion floor should therefore be read as a deterministic exact-recovery analogue of rate-distortion-perception constraints, not as a full solution of that stochastic theory.

## Learned Representations and Embedding Collapse

In learned classification systems, the observation map $\pi$ can be instantiated by an embedding or representation map whose coordinates are then queried or thresholded. The earlier sections isolate the finite explicit corollaries: one-symbol zero-error feasibility is equivalent to injectivity (GPH27), and refining the observation channel cannot increase collision multiplicity (GPH28). The extension question is how far that representation reading can be pushed once $\pi$ is implicit, high-dimensional, or learned from data. In that setting, the fixed-length and adaptive fiber laws become a formal measure of residual ambiguity under the representation: unresolved embedding collisions must be paid for either by explicit naming information or by accepting nonzero semantic error.


This paper develops an exact fixed-representation compression theory for zero-error identity recovery. Its governing object is the collision fiber geometry. From this single object flow exact laws for multiple operational modalities.

#### Compression contribution.

The same fiber geometry that forces $\log_2 A_\pi$ auxiliary bits also determines the adaptive fiberwise budget, the exact finite-block law, the distortion floor $1-2^L/a$, and the stability conditions under growth. These are not separate phenomena but different currencies for the same residual ambiguity. If the representation is injective, identity is free. If the representation is non-injective, the missing information must be paid somewhere: in auxiliary bits, in query effort, in helper views, or in distortion.

The query side ties directly back to the compression side. Any binary distinguishing family of size $d$ can realize at most $2^d$ transcripts on a collision fiber, so $\lceil \log_2 A_\pi \rceil \le d$ on the worst fiber. Complete derivability-aware query families reduce to orthogonal semantically minimal cores, on which exchange holds and matroid basis cardinality becomes the canonical query invariant.

The same ambiguity-fiber geometry governs when a representation is sufficient for exact downstream tasks, when helper views genuinely help, and when modular or factorized latents add exact information rather than architectural redundancy. In the learned-representation reading, injective representations support exact downstream identification with no extra metadata, while non-injective representations impose an irreducible side-information cost and, if that cost is left unpaid, a correspondingly irreducible deterministic distortion floor.

#### Systems payoff.

The central systems-level consequence is a formal justification for symbolic mechanisms in semantic systems. The coding laws prove that semantic abstraction imposes an exact identity cost: a non-injective representation requires auxiliary description to resolve its collision fibers, with worst-case cost $\log_2 A_\pi$ bits and fiberwise cost $\lceil \log_2 |\pi^{-1}(u)| \rceil$. Systems that require both semantic generalization and exact identity must therefore implement a mechanism to pay this cost.

Symbolic handles, identifiers, pointers, keys, and index entries are standard computational instantiations of that payment. Consequently, the integration of symbolic identity layers into neurosymbolic, retrieval, and other semantics-aware systems follows directly from the coding laws. In the instantiated formal domain, the pair consisting of a neural encoder and an injective symbolic handle achieves zero-error identity recovery if and only if the handle distinguishes entities within each semantic fiber (NSL1 through NSL6).

#### Mechanization.

The main theorem chain is machine-checked in Lean 4. The mechanization certifies that the fiber-geometry framework, the canonical query-core reduction, and the neurosymbolic linking results are exact formal statements in the finite explicit setting used throughout this paper.

## Acknowledgment: AI-use Disclosure {#acknowledgment-ai-use-disclosure .unnumbered}

Generative AI tools (including Codex, Claude Code, Augment, Kilo, and OpenCode) were used throughout this manuscript, across all sections (Abstract, Introduction, theoretical development, proof sketches, applications, conclusion, and appendix) and across all stages from initial drafting to final revision. The tools were used for boilerplate generation, prose and notation refinement, LaTeX and structure cleanup, translation of informal proof ideas into candidate formal artifacts (Lean and LaTeX), and repeated adversarial reviewer-style critique passes to identify blind spots and clarity gaps.

The author retained full intellectual and editorial control, including problem selection, theorem statements, assumptions, novelty framing, acceptance criteria, and final inclusion or exclusion decisions. No technical claim was accepted solely from AI output. Formal claims reported as machine-verified were admitted only after Lean verification (no `sorry` in cited modules) and direct author review; Lean was used as an integrity gate for responsible AI-assisted research. The author is solely responsible for all statements, citations, and conclusions.




---

## Machine-Checked Proofs

All theorems are formalized in Lean 4:
- Location: `docs/papers/paper1_typing_discipline/proofs/`
- Lines: 13838
- Theorems: 677
- `sorry` placeholders: 0
