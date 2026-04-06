# Paper: Semantic Identity Compression: Zero-Error Laws, Rate-Distortion, and Neurosymbolic Necessity

**Status**: JSAIT-ready | **Lean**: 13840 lines, 677 theorems

---

## Abstract

Symbolic systems operate over precise identities: variables denote specific objects, pointers target precise memory locations, and database keys refer to singular records. Neural embeddings generalize by compressing away semantic detail, but this compression creates collision ambiguity: multiple distinct entities can share the same representation value. We characterize how much additional information must be supplied to recover precise identity from such representations.

The answer is controlled by a single combinatorial object: the collision-fiber geometry of the representation map $\pi$. Let $A_{\pi}=\max_u |\pi^{-1}(u)|$ be the largest collision fiber. We prove a tight fixed-length converse $L \ge \log_2 A_{\pi}$, an exact finite-block scaling law, a pointwise adaptive budget $\lceil \log_2 |\pi^{-1}(u)|\rceil$, and an exact fiberwise rate-distortion law for arbitrary finite sources via recoverable-mass decomposition across representation fibers. The uniform single-block formula $D^\star(L)=\max(0,1-2^L/a)$ appears as a closed-form special case when all mass lies on one collision block, where $a = A_{\pi}$ is the collision block size. The same fiber geometry determines query complexity and canonical structure for distinguishing families.

Because this residual ambiguity is structural rather than representation-specific, symbolic identity mechanisms (handles, keys, pointers, nominal tags) are the necessary system-level complement to any non-injective semantic representation. All main results are machine-checked in Lean 4. **Keywords:** semantics-aware compression, zero-error coding, neurosymbolic systems, learned representations, side information


## Problem Statement

Symbolic systems are exact identity capable. Variables denote specific objects, pointers target precise memory locations, and database keys refer to singular records. This capability is not incidental; it is fundamental to symbolic computation, which requires that distinct entities remain distinguishable whenever operations depend on their identity.

Neural embeddings operate differently. They generalize by compressing away semantic detail, mapping distinct inputs to the same representation when those inputs are deemed functionally equivalent. The benefit is generalization: the embedding captures similarity structure and supports efficient nearest-neighbor operations. The cost is collision ambiguity: multiple distinct entities can share the same embedding value, and the representation alone no longer identifies which entity is meant [@saunshi2019theoretical; @higgins2017beta; @alemi2017deep; @jing2022understanding; @wang2020understanding].

This paper asks: given a fixed embedding that has already been deployed, what is the minimum price of recovering precise identity?

Formally, let $C$ be a finite class label, let $\pi$ be the representation map, and let $U = \pi(C)$ be the embedding value observed by the decoder. Exact identity recovery asks how much additional description is needed to recover $C$ from $U$. Once the problem is written this way, the central combinatorial quantity is immediate: $$A_{\pi} := \max_u |\pi^{-1}(u)|,$$ the size of the largest collision fiber.

::: remark
A representation $\pi$ is noninjective exactly when at least one fiber $\pi^{-1}(u)$ has size $>1$. The collision multiplicity $A_\pi$ therefore measures the degree of noninjectivity: $A_\pi=1$ iff $\pi$ is injective, and larger $A_\pi$ corresponds to worse collision.
:::

The worst-case fixed-length law is then exactly the minimum number of extra bits needed to separate that worst fiber: $$L \ge \log_2 A_{\pi}.$$

This viewpoint yields an adaptive law $\lceil \log_2 |\pi^{-1}(u)|\rceil$ per fiber, the fiberwise decomposition of recoverable mass for arbitrary finite sources, and a query-side counting floor $\lceil \log_2 A_{\pi}\rceil \le d$ (since $d$ binary observables induce at most $2^d$ transcripts). Worst-case bits, adaptive bits, query cost, and distortion are different currencies for the same residual ambiguity.

We study the finite, fixed-representation, zero-error regime: the decoder uses only the declared representation $U$ and charged auxiliary description. No hidden registry or external side channel may contribute uncharged distinguishing information. Higher-level regimes must reduce to this base case at deployment time, so they cannot escape the operational implications derived here.

#### Systems consequence

Symbolic handles are the standard mechanisms for paying the identity debt left by non-injective semantic abstraction. This debt equals the residual ambiguity quantified by collision fiber geometry (Theorems [\[thm:converse\]](#thm:converse){reference-type="ref" reference="thm:converse"} and [\[thm:adaptive-fiber-budget\]](#thm:adaptive-fiber-budget){reference-type="ref" reference="thm:adaptive-fiber-budget"}), resolved through auxiliary bits, query effort, or accepted distortion.

#### Privacy and security perspective

The fiber geometry directly governs identity disclosure. When distinct entities share the same representation, recovering which specific entity is meant constitutes an *identity disclosure*, and the auxiliary description required to resolve the collision fiber is exactly the information that is protected if privacy is to be maintained. This connects to the Special Issue theme of "data compression and storage under privacy and security constraints": our results characterize the precise disclosure-recovery tradeoff for semantic representations. The formal development establishes a finite explicit disclosure separation between nominal-tag access and attribute-only identification, aligning with formal exactness requirements in logic-based explanation frameworks [@ignatiev2019abduction; @marquessilva2022trustworthy], complementing broader privacy frameworks such as differential privacy [@dwork2006differential].

#### Related frameworks

The problem relates to Wyner--Ziv side information [@slepian1973noiseless; @wyner1976rate; @yamamoto1981source; @oohama1997gaussian; @gastpar2004wyner], zero-error graph-capacity theory [@shannon1956zero; @lovasz1979shannon], and semantics-aware communication [@gunduz2023beyond; @uysal2022semantic; @bao2011towards; @kountouris2021semantics]. In the fixed-representation regime, our confusability graphs are disjoint cliques, yielding exact finite formulas. The fiber geometry also connects to information bottleneck [@Tishby2000] and rate-distortion-perception [@blau2019rethinking; @chen2022rdp; @xie2025output].

#### Graph-theoretic interpretation

The strong product structure of confusability graphs under repeated composition connects this work to the Shannon capacity program [@simonyi1995graph; @alon1996source; @haemers1978upper; @zuiddam2019asymptotic] and zero-error systems theory [@witsenhausen1976zero].

## Contributions

The collision fiber geometry governs exact identity-recovery costs. Three interlocking threads:

1.  **Rate-distortion and exact coding.** Exact fiberwise rate-distortion law for arbitrary finite sources via recoverable-mass decomposition, with closed-form $D^\star(L)=\max(0,1-2^L/a)$ for uniform single-block sources. Supporting: tight converse $L \ge \log_2 A_{\pi}$, exact finite-block scaling in the finite-regime spirit of [@polyanskiy2010channel], and adaptive budget $\lceil \log_2 |\pi^{-1}(u)|\rceil$ per fiber.

2.  **Query currency and canonical structure.** Complete derivability-aware families reduce to an orthogonal semantically minimal core, where minimal distinguishing families form matroid bases and query cost obeys $\lceil \log_2 A_{\pi}\rceil \le d$.

3.  **Open-world robustness.** Collision-freedom does not persist under extension; barrier-freedom certification is uncomputable for arbitrary future-generating code.

Remaining payoffs: symbolic handles are necessary in semantics-aware systems; Lean 4 verification provides mechanized integrity.

#### Machine-checked proofs

All principal theorems are formalized in Lean 4; informal arguments were admitted only after formal proofs compiled without `sorry`. Superscript Lean codes reference specific theorems.

#### Worked example

We use one small example throughout the paper. Five classes have profiles $$A=000,\quad B=000,\quad C=100,\quad D=110,\quad E=111.$$ Then $A$ and $B$ form a collision fiber of size two, so the exact worst-case fixed-length budget is one bit. The adaptive picture is equally simple: the fiber over profile $000$ needs one auxiliary bit, while the singleton fibers need none. Under the adaptive scheme, the encoder sends one bit only when $U=000$ and sends no extra bit for $U\in\{100,110,111\}$.

## Paper Organization

The paper develops three interlocking threads. Sections [\[sec:model\]](#sec:model){reference-type="ref" reference="sec:model"} and [\[sec:bits\]](#sec:bits){reference-type="ref" reference="sec:bits"} establish the bit-currency backbone. Section [\[sec:queries\]](#sec:queries){reference-type="ref" reference="sec:queries"} develops the query-currency thread and its canonical matroid structure. Section [\[sec:distortion\]](#sec:distortion){reference-type="ref" reference="sec:distortion"} gives the distortion currency via fiberwise recoverable-mass decomposition. Section [\[sec:systems\]](#sec:systems){reference-type="ref" reference="sec:systems"} explains why symbolic identity layers are necessary in semantics-aware systems. Section [\[sec:robustness\]](#sec:robustness){reference-type="ref" reference="sec:robustness"} records open-world consequences of the fiber-geometry framework, and Section [\[sec:conclusion\]](#sec:conclusion){reference-type="ref" reference="sec:conclusion"} closes the paper.


## The Abstract Coding Framework

The paper studies a finite class space together with a fixed representation map. The decoder observes the representation value, and the encoder may additionally send auxiliary description to resolve whatever ambiguity the representation leaves behind. The coding laws depend only on this representation map and its fibers, not on the mechanism that produced it.

::: definition
Let $C$ be a finite class variable, let $U=\pi(C)$ be the representation available to the decoder, and let $M$ be any auxiliary description transmitted by the encoder. The main theorem family studies the minimum length of $M$ needed for exact recovery of $C$ from $(U,M)$, in both fixed-length and representation-adaptive regimes.
:::

::: definition
For a representation value $u$, define the corresponding fiber by $$\pi^{-1}(u) = \{c : \pi(c)=u\}.$$ The paper's central object is the geometry of these fibers.
:::

::: remark
No further structure is assumed. The map $\pi$ may arise from a learned encoder, database projection, feature-extraction pipeline, or hand-designed attribute family.
:::

::: remark
The coding statements use a class variable $C$ and representation $U=\pi(C)$. When convenient, we also use an underlying entity variable $V$ with class assignment $C(V)$; thus $U=\pi(V)$. The two viewpoints are interchangeable.
:::

::: definition
[]{#def:admissibility-contract label="def:admissibility-contract"} All impossibility and optimality statements quantify over schemes that use only (1) the declared representation value and (2) explicitly charged auxiliary description. In particular, no hidden registry, reflection oracle, whole-universe preprocessing table, or amortized cross-instance cache may supply uncharged distinguishing information.
:::

This contract is enforced in the mechanization; all impossibility claims assume no hidden registries or undeclared side channels.

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


## Fixed-Length Zero-Error Recovery

We now state the zero-error coding problem in generic representation-map form. Let $C$ be a finite class variable and let $U=\pi(C)$ be the representation observed by the decoder. If the representation is non-injective on classes, then the decoder cannot determine $C$ from $U$ alone; the unresolved ambiguity is exactly the ambiguity inside the fibers of $\pi$.

::: definition
Let $C$ be a random class label, let $U=\pi(C)$ be its representation, and let the decoder observe $U$ together with any charged auxiliary description supplied by the encoder.
:::

::: theorem
[]{#thm:identification-capacity label="thm:identification-capacity"} Let $\mathcal{C}$ be a finite class space and let $\pi : \mathcal{C} \to \mathcal{U}$ be a representation map. Zero-error recovery of $C$ from $U=\pi(C)$ with no auxiliary description is achievable if and only if $\pi$ is injective on classes.
:::

::: proof
*Proof.* *Achievability*: If $\pi$ is injective on classes, then observing $\pi(c)$ determines $c$ uniquely. The decoder inverts the class-to-representation map.

*Converse*: Suppose two distinct classes $c_1 \neq c_2$ share the same representation value: $\pi(c_1)=\pi(c_2)$. Then any decoder $g(U)$ outputs the same class label on both inputs, so it cannot be correct for both. Hence zero-error recovery without auxiliary description is impossible. ◻
:::

In the finite explicit setting, the same condition can be read directly from the representation channel: one-symbol zero-error feasibility is equivalent to injectivity.

::: remark
Under any distribution with positive mass on both colliding classes, $I(C;U) < H(C)$. This is an average-case consequence of the deterministic barrier above.
:::

## Fixed-Length Auxiliary Description

We now show that augmenting the representation with an auxiliary codeword restores exact recovery.

::: definition
An *auxiliary description* is a value $M(c)$ associated with each class $c \in \mathcal C$ and transmitted in addition to the representation value $\pi(c)$. In the fixed-length setting, the auxiliary description alphabet has size $2^L$.
:::

::: theorem
[]{#thm:tag-restored-sufficiency label="thm:tag-restored-sufficiency"} If $|\mathcal C|=k$, then an auxiliary description of length $L \geq \lceil \log_2 k \rceil$ bits suffices for zero-error identification, regardless of whether $\pi$ is class-injective.
:::

This is the ambient worst-case naming bound over the full class set. The representation-sensitive law below is sharper and is the paper's actual governing converse: it depends on collision multiplicity $A_\pi$ rather than on $|\mathcal C|$.

## Ambiguity Converse

The next theorem is the paper's main zero-error compression statement, in the same fixed-length form as a classical zero-error side-information budget [@csiszar2011information] but with the governing alphabet size induced by the representation rather than by the ambient class set alone. It converts representation collision multiplicity directly into a necessary fixed-length side-information budget.

In graph terms, confusable classes are exactly those lying in the same representation fiber, so the confusability graph is a disjoint union of cliques. This structure yields exact finite laws for the residual coding burden left by a representation.

::: definition
[]{#def:collision-multiplicity label="def:collision-multiplicity"} Let $\mathcal{C}$ be a finite class space and let $\pi : \mathcal{C} \to \mathcal U$ be a representation map. Define $$A_\pi := \max_{u \in \operatorname{Im}(\pi)} \left|\{c \in \mathcal{C} : \pi(c)=u\}\right|.$$ Thus $A_\pi$ is the size of the largest class-collision block under the representation.
:::

::: remark
For finite-precision representations, $A_\pi$ is computable by fiber enumeration. Pigeonhole: $\lceil |\mathcal C|/M \rceil \le A_\pi$ for $M$ outputs; conversely, $A_\pi \le k$ requires $\lceil n/k \rceil$ distinct values.
:::

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

::: corollary
[]{#cor:max-barrier-converse label="cor:max-barrier-converse"} If the domain is maximal-barrier ($A_\pi = k$), any zero-error scheme satisfies $L \ge \log_2 k$.
:::

## Exact Finite-Block Law

The ambiguity converse extends exactly to blocks: the next theorem family gives exact finite-block scaling rather than an asymptotic approximation.

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
The construction repeats the same task across $t$ entities with coordinatewise observation, so collision structure and side-information budget multiply accordingly.
:::

## Collision Fibers as Graph Components

The confusability graph induced by a fixed deterministic representation has special structure: two classes are adjacent exactly when they share the same representation value. This section makes that structure explicit.

::: proposition
[]{#prop:cluster-graph label="prop:cluster-graph"} The confusability graph $G_\pi$, with edges between distinct classes sharing the same representation, is a disjoint union of cliques. Each connected component is exactly one representation fiber. Moreover, confusability is transitive on distinct classes inside a fiber.
:::

::: proof
*Proof.* Each fiber is a clique because all of its classes share the same representation value. No edge connects distinct fibers by construction. Transitivity is immediate: if $\pi(c_1)=\pi(c_2)$ and $\pi(c_2)=\pi(c_3)$, then $\pi(c_1)=\pi(c_3)$ by equality. ◻
:::

::: corollary
[]{#cor:capacity-collapse label="cor:capacity-collapse"} For a representation-induced cluster graph, the standard one-shot zero-error parameters collapse to fiber counts, so the fixed-length auxiliary-description law is governed exactly by the largest fiber.
:::

::: proof
*Proof.* In a disjoint union of cliques, proper coloring, clique size, and independence counting reduce to choosing or naming one element per component. The general graph inequalities collapse to the same fiber-counting law. ◻
:::


## Representation-Adaptive Side Information

When auxiliary description length may depend on the observed representation value $u$, the optimal budget is the fiber size $\pi^{-1}(u)$ rather than the maximum $A_\pi$.

::: definition
An *adaptive auxiliary scheme* assigns to each representation value $u$ a binary description budget $\ell(u)$ and a decoder that, given $u$ together with a binary string of length at most $\ell(u)$, recovers the class exactly whenever $\pi(C)=u$.
:::

::: theorem
[]{#thm:adaptive-fiber-budget label="thm:adaptive-fiber-budget"} For a representation fiber of size $|\pi^{-1}(u)|$, zero-error recovery under a binary auxiliary alphabet is feasible if and only if $$|\pi^{-1}(u)| \le 2^{\ell(u)}.$$ Consequently, the pointwise optimal number of auxiliary bits on fiber $u$ is $$\ell^\star(u) = \left\lceil \log_2 |\pi^{-1}(u)| \right\rceil.$$
:::

::: proof
*Proof.* Once $u$ is fixed, the decoder only needs to distinguish classes inside the fiber $\pi^{-1}(u)$. With $\ell(u)$ binary bits there are at most $2^{\ell(u)}$ auxiliary outcomes, so recovery is possible when that many outcomes suffice to name the fiber elements injectively. The minimal feasible budget is therefore the least integer whose power of two dominates the fiber size, namely $\lceil \log_2 |\pi^{-1}(u)| \rceil$. ◻
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

The zero-error conditional-entropy sandwich places the expected adaptive cost between $H(C\mid U)/\log 2$ and $H(C\mid U)/\log 2 + 1$, but the zero-error law remains fiberwise and combinatorial rather than purely entropic.


## The Query Currency

Bits are one way to pay the identity debt left by a non-injective representation; another is to interrogate primitive observables until the class is determined. This section shows that the same fiber geometry governing coding costs also governs query costs, and that every complete derivability-aware query family reduces to an orthogonal semantically minimal core on which the minimal distinguishing families form a matroid.

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
These notions differ; the operational lower bound depends only on $d$.
:::

In the running example from Section [\[sec:bits\]](#sec:bits){reference-type="ref" reference="sec:bits"}, $\{q_1,q_2,q_3\}$ is distinguishing and no pair of primitive queries is distinguishing, so the minimum distinguishing number is $d=3$. The exact operational law is that any attribute-only zero-error witness procedure must pay at least this many queries in the worst case.

::: theorem
[]{#thm:interface-lower-bound label="thm:interface-lower-bound"} For attribute-only observers, zero-error class identity checking requires $$W(\text{class-identity}) \ge d$$ in the worst case, where $d$ is the minimum distinguishing number.
:::

::: proof
*Proof.* Assume a zero-error attribute-only procedure halts after fewer than $d$ queries on every execution path. Fix one execution path and let $Q \subseteq \mathcal I$ be the set of queried attributes on that path, so $|Q|<d$. By definition of $d$, no set of size less than $d$ is distinguishing. Hence there exist values $v,w$ from different classes with identical answers on all attributes in $Q$. An adversary can answer the procedure's queries consistently with both $v$ and $w$, so the transcript and output coincide on two different classes, contradicting zero-error identification. ◻
:::

This exact finite lower bound is the operational query law in full generality.

::: corollary
[]{#cor:attribute-only-witness-lb label="cor:attribute-only-witness-lb"} For any attribute-only observer, $W(\text{class-identity}) = \Omega(d)$.
:::

It says how much interrogation is needed when the representation leaves ambiguity unresolved and no explicit identity bits are transmitted.

## Counting Bridge to Bit Currency

A family of $d$ binary queries induces at most $2^d$ transcripts on any fiber, so $\lceil \log_2 A_{\pi} \rceil \le d$. This is the query-side analogue of the fixed-length converse.

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
*Proof.* By Proposition [\[prop:query-orthogonal-core\]](#prop:query-orthogonal-core){reference-type="ref" reference="prop:query-orthogonal-core"}, every complete derivability-aware query system contains an orthogonal semantically minimal core. On an orthogonal core, exchange holds. Standard matroid theory then implies that the minimal complete families are exactly the bases of a matroid and therefore all have the same cardinality. ◻
:::

::: remark
Arbitrary overcomplete families need not be matroidal (GPH31), but after derivable redundancy is stripped, the orthogonal minimal core is matroidal.
:::

## Witness Cost Comparison

::: {#tab:witness-comparison}
  -------------------------------------------------------------------------------------------------
  **Observer Class**    **Witness Procedure**                           **Witness Cost $W$**
  --------------------- ----------------------------------------------- ---------------------------
  Nominal-tag           Single tag read                                 $O(1)$

  Attribute-only        Query a distinguishing family of minimum size   $\ge d$

  Canonical axis core   Query any matroid basis of the core             exact basis cardinality
  -------------------------------------------------------------------------------------------------

  : Witness cost for class identity by observer class.
:::

::: remark
The bit law measures residual description cost; the query law measures residual interrogation cost. Both lower-bounds equal $\lceil \log_2 A_{\pi} \rceil$ in the canonical derivability-aware presentation.
:::


## Rate-Distortion Law and Distortion Currency

Auxiliary bits, queries, and accepted distortion are alternative currencies for the same residual ambiguity. This section gives the fiberwise rate-distortion law: optimal recoverable mass decomposes exactly across representation fibers for arbitrary finite sources, with closed-form $D^\star(L)=\max(0,1-2^L/a)$ in the uniform single-block corner.

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
The uniform single-block formula below is the special case where all source mass lies in one uniformly distributed fiber. Neither assumption is required for the decomposition itself, which is the general zero-error rate-distortion law for the fixed-representation setting.
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
The uniform single-block formula isolates geometry-only effects; non-uniform or multi-fiber sources introduce source-law dependence beyond this paper's scope.
:::

## Finite Entropy Converses {#sec:entropy-converses}

The counting-based converses above are combinatorial: they bound distortion in terms of fiber sizes and tag budgets. A complementary family of converses connects distortion to information-theoretic quantities [@CoverThomas2006; @csiszar2011information; @barron1998information], placing the semantic identity problem inside classical Shannon theory.

The following theorems are mechanized in the RDC series. They apply to any finite source, any observation map, and any tag/decoder pair.

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
The counting converse $2^L \ge a$ is sharp for zero-error but silent on source structure; entropy converses complement this for high-entropy sources.
:::


Lossy compression and semantic abstraction necessarily create collision fibers by discarding distinguishing information. In closed-world systems, the deployed representation already contains collisions; in open-world systems, collisions cannot be certified away. The identity debt is paid in one of three currencies: an injective encoder, explicit identity metadata of length at least $\log_2 A_\pi$ in the worst case, or a nonzero distortion floor. In deployed systems the usual payment mechanisms are symbolic handles, such as unique identifiers, pointers, retrieval keys, primary keys, and registry entries.

Example. In a retrieval system with $100$ documents sharing the same embedding, omitting the $\lceil \log_2 100 \rceil = 7$ identifier bits leaves a distortion floor of $1-1/100 = 99\%$ on that collision fiber. Even one bit below the exact-recovery threshold is costly: with only $6$ bits the distortion is still at least $36\%$.

## Learned Representations and Formal Instantiations

The abstract coding framework applies to any representation map. Zero-error recovery from semantic information alone is possible exactly when the representation is injective on identity classes; otherwise, collision fibers quantify the irreducible identity residue restored by explicit metadata.

#### What this means for ML systems.

-   **Concept bottleneck models.** Collapsed concepts require extra identity information: disambiguation channels, retrieval keys, or symbolic identifiers.

-   **Retrieval-augmented systems.** The retrieval index pays the identity-bit budget $\lceil \log_2 A_\pi \rceil$; if embeddings collide, distinguishing metadata (document identifiers, lookup keys) must be preserved [@lewis2020retrieval].

-   **Pure representation schemes.** Open-world systems need either an injective encoder or explicit identity metadata.

#### Attribute-only versus tag-augmented design.

When $A_\pi>1$, attribute-only strategies inherit the distortion floor (error at least $1-1/A_\pi$), while tag-augmented strategies pay up to $\lceil \log_2 A_\pi \rceil$ bits for zero-error recovery. This is why symbolic identifiers are ubiquitous in high-reliability systems.

::: corollary
[]{#cor:pareto-domination label="cor:pareto-domination"} In any system where nominal identity is provided for free (databases with primary keys, object stores with stable identifiers, programming languages with inheritance hierarchies), the nominal architecture must be chosen for exact identity recovery: attribute-only alternatives cannot achieve zero error, while the nominal system does at no additional cost. By Theorem [\[thm:converse\]](#thm:converse){reference-type="ref" reference="thm:converse"}, the required bit budget is $\lceil \log_2 A_\pi \rceil$; when this budget is zero (nominal identity is free), the nominal architecture achieves zero error while attribute-only systems cannot. This follows from the information barrier , the impossibility of shape-based distinction and provenance recovery , the non-embeddability of semantic identity , and the fact that shape-only systems are concessions not alternatives .
:::

The mechanization guarantees that every registry read/write in a valid trace names a declared registry; undeclared registries cannot appear in certified traces.

#### Formal instantiation.

The Lean formalization instantiates the generic coding laws with entities carrying attribute family $\mathcal I$ and profile map $\pi(v)=(q_I(v))_{I\in\mathcal I}$; equal profiles cannot distinguish identity, and provenance is unrecoverable when collisions exist.

#### Instantiated neurosymbolic linking theorems.

Setting the task to identity $Y=C$, the mechanization proves: injective $\sigma$ gives zero-error recovery via $(\pi,\sigma)$ ; if $\pi$ has collisions, semantic representation alone cannot achieve zero-error identity ; $(\pi,\sigma)$ recovers identity iff $\sigma$ distinguishes entities within each semantic fiber ; and $(\pi,\sigma)$ is the canonical helper view for open-world identity systems .

## Task Sufficiency, Helper Views, and Factorization

The fiber geometry characterizes task sufficiency, helper views, and factorized modules [@orlitsky2001coding; @doshi2010functional].

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
*Proof.* The pair $(U,S)$ is a refined deterministic representation. Exact recovery is therefore possible exactly when the downstream task is constant on the fibers of that refined representation. ◻
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

For learned representations, estimating quantities such as $A_\pi$, $A_{\pi \to Y}$, or the adaptive fiber profile is a representation-analysis problem rather than part of the theorem. In a finite explicit registry or index this is fiber counting; in a large learned embedding one instead needs explicit collision enumeration after discretization or provable upper and lower bounds. The point of the present results is to identify what those quantities mean once they are known.

Concretely, the task-sufficiency theorem says the representation is sufficient precisely when every fiber carries one task label, and the helper-view and factorized laws say when added views contribute genuinely new exact information.

In the finite explicit setting, one-symbol zero-error feasibility is equivalent to injectivity and refinement cannot increase collision multiplicity. The same setting also yields a disclosure separation between nominal access and tag-free identity resolution.


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
Present collision-freedom does not guarantee stability under extension; persistent zero-error claims need an explicit growth model rather than a one-time snapshot argument.
:::

For compression system design: a representation needing no auxiliary bits today can require positive budget after extension. Open-world growth changes the optimal rate even when the deployed decoder is unchanged.

::: definition
For a partial observer generator $f:\mathbb{N}\rightharpoonup\mathbb{N}$, define $$\mathrm{HasBarrier}(f)\;:\Leftrightarrow\;\exists x\neq y,\exists z,\ z\in f(x)\land z\in f(y).$$ Define $\mathrm{BarrierFree}(f):\Leftrightarrow \neg\mathrm{HasBarrier}(f)$.
:::

::: theorem
[]{#thm:rice-barrier label="thm:rice-barrier"} There is no computable predicate on program codes deciding whether the generated observer has a barrier: $$\neg\mathrm{ComputablePred}\!\left(c\mapsto \mathrm{HasBarrier}(\mathrm{eval}(c))\right).$$ Equivalently, barrier-freedom certification is also non-computable.
:::

::: proof
*Proof.* The property is non-trivial (some generators produce collisions, some do not), so by Rice's theorem no computable predicate can decide it from code. ◻
:::

::: remark
No general computable procedure can certify barrier-freedom for arbitrary future-generating code, even when deployment appears collision-free. Zero-error guarantees from finite training cannot extend to new production classes without explicit identity metadata or restrictive growth models.
:::

::: corollary
For unrestricted open-world generators, one cannot computably certify that attribute-only identification remains barrier-free under all future extensions. Barrier-freedom fails under arbitrary extensions , and both barrier-existence and barrier-freedom certification are uncomputable by Rice's theorem . Consequently, persistent $D=0$ guarantees cannot rely on "currently collision-free" attribute structure alone.
:::


The main theorem chain identifies structural laws for zero-error identity recovery. This section records generalizations that relax specific model assumptions while preserving the core insight: residual ambiguity must be paid for in some resource currency.

The zero-error identification setting differs from the identification paradigm of [@ahlswede1989identification], where the decoder asks "is the message $m$?" yielding double-exponential codebook sizes; here zero-error class identification has a binary one-shot feasibility threshold rather than an asymptotic one.

## Stochastic and Semantic Generalizations

Noisy representations replace deterministic fibers with probabilistic confusion neighborhoods, moving the problem toward the graph-entropy regime [@korner1973coding]. A lossy extension with semantic distortion measures would distinguish tolerable confusion from unacceptable failure along the same fiber geometry.

## Dynamic Growth and Trigger Conditions

The static theorems also suggest operational limits for evolving systems. In a Poissonized growth model for representation cells with arrival rate $\lambda$, the probability of at least one collision in a cell is $1 - e^{-\lambda}(1+\lambda)$, which isolates the onset of identity pressure. The required exact identity budget becomes positive exactly when occupancy reaches two or more items per cell. These formulas identify the dynamic trigger conditions at which auxiliary symbolic handles become necessary to prevent identity collapse.

## Information-Theoretic Vistas: Privacy and Entropy

The entropy converses (Section [\[sec:entropy-converses\]](#sec:entropy-converses){reference-type="ref" reference="sec:entropy-converses"}) also imply privacy-preserving disclosure limits: nominal access has unit disclosure cost, while tag-free identity resolution can require $n-1$ primitive disclosures in adversarial settings. This three-resource geometry invites comparison with rate-distortion-perception formulations [@blau2019rethinking].

## Empirical Audit: Embedding Collapse

For learned representations, the fixed-length and adaptive fiber laws provide a formal measure of residual ambiguity. Once a quantization rule is fixed, the collision audit can be connected back to the mechanization: the resulting fiber histogram and budget curve are exported as Lean certificates whose consistency checks reduce to the same finite formulas proved in the paper. This does not mechanize the neural encoder itself, but it does make the empirical collision audits machine-checkable.


This paper develops a fixed-representation compression theory for zero-error identity recovery, governed by the collision fiber geometry $A_\pi$.

#### Compression contribution.

The same fiber geometry determines the adaptive fiberwise budget, exact finite-block law, fiberwise decomposition of recoverable mass, and query-side counting law. The uniform single-block curve $D^\star(L)=\max(0,1-2^L/a)$ is the sharp closed-form specialization. If $\pi$ is injective, identity is free; if non-injective, the missing information must be paid in bits, queries, helper views, or distortion.

The query side ties directly back to the compression side. Any binary distinguishing family of size $d$ can realize at most $2^d$ transcripts on a collision fiber, so $\lceil \log_2 A_\pi \rceil \le d$ on the worst fiber. Complete derivability-aware query families reduce to orthogonal semantically minimal cores, on which exchange holds and matroid basis cardinality becomes the canonical query invariant.

The same ambiguity-fiber geometry governs when a representation is sufficient for exact downstream tasks, when helper views genuinely help, and when modular or factorized latents add exact information rather than architectural redundancy. In the learned-representation reading, injective representations support exact downstream identification with no extra metadata, while non-injective representations impose an irreducible side-information cost and, if that cost is left unpaid, a correspondingly irreducible deterministic distortion floor.

#### Systems payoff.

Semantic abstraction imposes an exact identity cost: non-injective representations require auxiliary description with worst-case cost $\log_2 A_\pi$ bits and fiberwise cost $\lceil \log_2 |\pi^{-1}(u)| \rceil$. Symbolic handles are the standard payment mechanism. A neural encoder plus injective handle achieves zero-error recovery iff the handle distinguishes entities within each semantic fiber.

#### Secondary robustness consequence.

Present collision-freedom need not persist under extension, and unrestricted future-generating code has no computable barrier-freedom certificate. Persistent zero-error guarantees require explicit growth models or identity metadata.

#### Mechanization.

The main theorem chain is machine-checked in Lean 4, certifying the fiber-geometry framework, canonical query-core reduction, and neurosymbolic linking results in the finite explicit setting.

## Acknowledgment: AI-use Disclosure {#acknowledgment-ai-use-disclosure .unnumbered}

Generative AI tools (including Codex, Claude Code, Augment, Kilo, and OpenCode) were used throughout this manuscript, across all sections (Abstract, Introduction, theoretical development, proof sketches, applications, conclusion, and appendix) and across all stages from initial drafting to final revision. The tools were used for boilerplate generation, prose and notation refinement, LaTeX and structure cleanup, translation of informal proof ideas into candidate formal artifacts (Lean and LaTeX), and repeated adversarial reviewer-style critique passes to identify blind spots and clarity gaps.

The author retained full intellectual and editorial control, including problem selection, theorem statements, assumptions, novelty framing, acceptance criteria, and final inclusion or exclusion decisions. No technical claim was accepted solely from AI output. Formal claims reported as machine-verified were admitted only after Lean verification (no `sorry` in cited modules) and direct author review; Lean was used as an integrity gate for responsible AI-assisted research. The author is solely responsible for all statements, citations, and conclusions.




---

## Machine-Checked Proofs

All theorems are formalized in Lean 4:
- Location: `docs/papers/paper1_typing_discipline/proofs/`
- Lines: 13840
- Theorems: 677
- `sorry` placeholders: 0
