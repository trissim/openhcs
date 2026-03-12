# Paper 1 Reviewer Framing Integration Plan

## Goal

Integrate the latest reviewer comments without rewriting the mathematical core. The objective is to make the paper read as:

1. a deliberately scoped deterministic compression paper,
2. centered on one exact object, residual ambiguity / collision fibers,
3. with an explicit semantic identity rate-distortion result,
4. and with neurosymbolic / symbolic-handle claims grounded in concrete systems practice.

The required changes are almost entirely LaTeX/prose. No new Lean mechanization is required unless we later decide to add dedicated handles for newly named paper corollaries. The current theoremic backbone is already sufficient.

---

## Current Reviewer Requests To Integrate

1. Add a short section after the introduction: `Why This Is Not Obvious`
2. Sharpen the rate-distortion framing around the deterministic corner
3. Ground the neurosymbolic claims in concrete systems failures / design mistakes
4. Add an explicit near-end section: `What This Paper Does Not Do`
5. Recast the abstract so it leads with the design-choice-versus-theorem contrast
6. Reframe the whole paper so the generic coding framework leads and the formal entity-attribute / Lean material appears later as a rigorous instantiation

---

## Preferred Structural Reframing

The current paper is much stronger than before, but it still partly reads as if the formal entity-attribute model is the conceptual front door. The preferred restructuring is to make the front door fully generic and move the PL-flavored machinery into a later witness / instantiation role.

### Current implicit arc

1. entity / attribute machinery,
2. information barrier,
3. zero-error law,
4. adaptive coding,
5. neurosymbolic / ML applications.

### Preferred arc

1. **The Abstract Coding Framework**
   - finite class space `C`
   - representation map `pi : C -> U`
   - fibers `pi^{-1}(u)` and collision multiplicity `A_pi`
   - decoder sees `U = pi(C)` and encoder may send auxiliary `M`

2. **Main Generic Theorems**
   - zero-error threshold / exact fixed-length law
   - finite-block scaling
   - adaptive fiberwise law
   - entropy comparison
   - deterministic-corner rate-distortion law

3. **Generic Consequences**
   - task sufficiency
   - helper views
   - factorization
   - query-side interpretation

4. **Instantiations**
   - learned embeddings / ML systems first
   - formal entity-attribute / Lean witness second

5. **Extensions and Open Directions / What This Paper Does Not Do**

### What this achieves

- The main theorems become obviously information-theoretic and generic.
- The ML story gets equal or earlier billing.
- The Lean/PL material stays important, but as a machine-checked witness rather than the only framing lens.
- A JSAIT reviewer can understand the theoremic contribution without needing to care where `pi` came from.

### Guiding principle for all edits

Whenever a statement depends only on fibers of `pi`, cardinalities, and auxiliary descriptions, state it generically. Only use the entity/attribute language where the paper is deliberately instantiating the abstract framework.

---

## Non-Negotiable Constraints

### Keep the claims scoped correctly

- The exact curve `D^*(L) = max(0, 1 - 2^L / a)` is proved only for the uniform source on a single collision block.
- Do **not** accidentally widen this to a full general rate-distortion theory.
- Do **not** say symbols are the only possible mechanism. The theorem says injective auxiliary information is required; symbols are the standard systems implementation.

### Preserve the current theoremic spine

The paper's spine should remain:

`collision fibers -> exact zero-error law -> exact adaptive law -> deterministic rate-distortion corner -> helper-view / sufficiency consequences -> symbolic payment as systems implementation`

### Generic-first does not mean over-generalized

The theorem statements should be generic only where the current proofs truly justify that level of abstraction. If a statement genuinely depends on the formal entity-attribute model, keep it as an explicit instantiation or corollary.

---

## File-Level Edit Plan

## 1. `latex_jsait/content/abstract.tex`

### Purpose
Replace the current theorem-list opening with the stronger framing:

- people often treat identity-vs-semantics as design taste,
- the paper shows it is governed by exact coding constraints,
- unpaid identity cost becomes an exact distortion floor,
- the deterministic corner and neurosymbolic implication are then stated clearly.

### Current issue
The current abstract is strong, but still reads like a technical result list beginning with `We prove...`. The reviewer wants the opening contrast to be:

`this is often treated as architectural opinion; we show it is a theoremic coding constraint.`

### Exact changes

Replace the abstract with a 3-paragraph structure:

#### Paragraph 1: design choice vs theorem
- Open with: whether semantic representations can capture identity without auxiliary identifiers is often treated as a design choice.
- State the negative result: if two entities collide under the representation, no decoder can recover identity without extra distinguishing information.
- State the exact worst-case cost: `log_2 A_pi` bits.
- State the zero-budget floor: `D >= 1 - 1 / A_pi` on a worst collision block.

#### Paragraph 2: deterministic corner, exact laws, scope
- Explicitly say: the paper characterizes the clean deterministic corner.
- State the exact collision-block law:
  - `D^*(L)=max(0,1-2^L/a)`
- State the adaptive fiberwise law:
  - `ceil(log_2 |pi^{-1}(u)|)`
- Use one sentence to say the underlying mathematics is elementary once the problem is posed correctly.

#### Paragraph 3: instantiated consequence
- Say the paper instantiates the helper-view laws in a formal identity domain.
- State that a neural encoder + symbolic handle achieves zero-error identity iff the handle distinguishes entities within each semantic fiber.
- End with the systems-level line:
  - the identity bits must be paid somewhere,
  - symbolic layers are the standard computational mechanism for paying them.

### Guardrails
- Keep the phrase `uniform collision block` in the abstract.
- Avoid calling the result a full rate-distortion theory.

---

## 2. `latex_jsait/content/01_introduction.tex`

### Purpose
Add a short skeptical-reader section and refine the framing of what the contribution actually is.

This file should also begin the generic-first restructuring. The introduction should present the paper as being about arbitrary fixed representation maps, with the entity-attribute formalization introduced later as one important witness model.

### New subsection to add

Insert **after** `\subsection{Relation to Prior Work}` and **before** `\subsection{Contributions}`:

`\subsection{Why This Is Not Obvious}`

### Content to include

Open with the key honesty sentence:

> The core coding fact is simple once the problem is posed correctly. The contribution is not a difficult converse trick but identifying the correct pose.

Then give 4 short bullets or short paragraphs covering:

1. **Ambient attributes feel like they should suffice**
   - rich semantics tempts one to think identity is included “for free”
   - the paper proves this fails whenever collisions exist

2. **Closed worlds hide the issue**
   - a closed deployment universe can be engineered to avoid collisions
   - the theorems matter for open worlds and fixed representations not designed for identity

3. **Error tolerance masks the floor**
   - many systems tolerate “some error”
   - the deterministic corner shows that omitted identity bits induce an exact floor, not vague degradation

4. **The bill is often paid implicitly**
   - retrieval keys, IDs, pointers, cache keys, memory addresses
   - the paper's role is to expose where those bits are being paid

### Why this section matters

This section directly answers the skeptical reviewer who says:

`isn't this just pigeonhole principle?`

The response becomes:

`yes, once correctly posed; the contribution is that the literature and systems practice often pose it incorrectly or leave the payment implicit.`

### Contribution list changes

Keep the current rate-distortion-first ordering. Only lightly polish wording:

- Contribution 1 stays the deterministic corner tradeoff
- Contribution 2 stays zero-error threshold and semantic identity rate law
- Contribution 3 stays adaptive side information

Do **not** over-reorder further unless page pressure requires it.

### Additional wording tweak in intro

In the existing rate-distortion framing, prefer `deterministic corner` or `clean deterministic corner` over `rate-distortion picture` when possible.

### Additional structural edits to capture the new framing

1. Add one sentence in the introduction explicitly saying that the paper studies any fixed representation map `pi`, whether it comes from attributes, learned embeddings, schema projections, or other feature pipelines.
2. Keep the contribution statements generic before naming the formal witness model.
3. If the introduction still uses too much entity/attribute language early, rewrite those sentences in terms of classes, representations, and fibers.

---

## 3. `latex_jsait/content/02_compression_framework.tex`

### Purpose
This is the key file for the structural reframing. It should become the paper's generic abstract coding framework section rather than an early PL-flavored formal setup.

### Current issue
The file currently opens with attribute-family machinery, observable functions, and entity notation. That material is useful, but it is too domain-specific for the front of a JSAIT paper.

### Target structure

Rework the front of the file so it opens with a generic framework:

1. finite class space `C`
2. representation map `pi : C -> U`
3. decoder side information `U = pi(C)`
4. auxiliary description `M`
5. fibers `pi^{-1}(u)` and collision multiplicity `A_pi`

Only after that should the file say that one concrete formal realization comes from an entity domain with an attribute family.

### Specific edit strategy

- Keep the current generic coding-model definition, but move it to the very top.
- Add a short paragraph saying no further structure is assumed.
- Add a short list of sources of `pi`: learned encoders, database projections, feature extractors, hand-designed attributes.
- Move the attribute-family formalism under a heading like `A Formal Entity-Attribute Instantiation`.

### Why this matters

If this section is generic-first, the rest of the paper naturally reads as an information-theory paper with later witness models. If it stays attribute-first, the later reframing will feel cosmetic.

---

## 4. `latex_jsait/content/04_kolmogorov_witness.tex`

### Purpose
Make the main zero-error theorem arc read generically wherever the proofs only depend on fibers and finite counting.

### Current issue
The opening statements still use the entity variable `V`, class assignment maps, and representation profiles as if those are conceptually primary.

### Target

Rewrite the local framing and theorem statements, when possible, into the language of:

- finite class space `C`
- representation map `pi`
- side information `U = pi(C)`
- auxiliary description alphabet of size `2^L`

### Practical rule

- If the theorem uses only collision multiplicity and fiber cardinalities, state it generically.
- If the theorem uses the formal witness model in an essential way, keep that notation and mark it as an instantiation.

### Why this matters

This is the place where the paper either convincingly becomes generic-first or remains PL-framed under a new introduction.

---

## 5. `latex_jsait/content/05_rate_distortion.tex`

### Purpose
Make the local framing match the reviewer guidance precisely:

- this is the deterministic corner,
- it is intentionally scoped,
- the uniform single-block assumption is a feature for isolation,
- the steepness of the curve is the practical systems point.

### Existing strengths to preserve

Already present:

- theorem title renamed to semantic identity rate-distortion theorem
- exact formula with achievability
- zero-error threshold corollary
- zero-budget floor corollary
- scope remark
- steepness remark

### Required edits

#### Change 3.1: first paragraph of subsection
Current local intro is good, but revise so it says more explicitly:

> We characterize the exact distortion-bits tradeoff in the cleanest deterministic setting: a uniform source on a single collision block.

Then explain why that scope was chosen:

> this isolates collision geometry alone, separate from source statistics and cross-fiber interactions.

#### Change 3.2: strengthen steepness remark with concrete numbers
In `Why this matters`, add one or two concrete examples:

- if `a = 100`, one bit short of threshold implies error at least `1 - 2^6/100 = 36%` when threshold is near 7 bits, or use simpler examples tied to zero budget and low budgets
- cleaner option: use the examples already in the omission remark and add one sentence that even small deficits can be operationally unacceptable

Need to be careful to keep arithmetic correct and easy to scan.

#### Change 3.3: preserve honest scope remark
Keep the current `Scope of the exact curve` remark, but make it slightly stronger:

- say extensions are plausible but would need source-sensitive analysis
- say this paper intentionally isolates the geometry-only corner

### Optional paper-level rename

Consider whether the subsection title should be:

- `The Deterministic Corner: Semantic Identity Rate-Distortion on a Collision Block`

This is likely a net improvement if it does not look too long in JSAIT format.

---

## 6. `latex_jsait/content/06_applications.tex`

### Purpose
Ground the neurosymbolic statements in concrete systems practice and common failure modes.

This file should also carry the explicit instantiation logic: learned-representation instances should appear first, and the formal entity-attribute / Lean-backed instance should read as a witness model rather than the default global frame.

### New subsection / paragraph to add

Insert after `\paragraph{What this means for ML systems.}` and before `Attribute-only versus tag-augmented design`, or fold into that area:

`\paragraph{What systems get wrong.}`

### Exact content to include

Use 3 concrete examples, matching the reviewer request but avoiding overclaim:

1. **Pure embedding retrieval without identifiers**
   - if two documents collide in embedding space, exact retrieval is impossible without extra metadata
   - many systems rely on rarity of collision rather than proving it away

2. **Concept bottleneck models without concept identifiers**
   - if multiple ground-truth concepts collapse into one learned representation, downstream classification cannot restore exact concept identity without additional signals

3. **Semantic caching without key disambiguation**
   - semantic hashes can preserve meaning while still colliding on identity
   - the result is semantically appropriate but identity-wrong retrieval

### Why this addition matters

The current section already states the positive systems reading. This addition gives the negative reading reviewers want:

`here are specific design patterns that fail because they ignore the coding law.`

### Guardrail

Phrase these as:

- `can fail`
- `cannot guarantee exact identity`
- `inherits the error floor when collisions occur`

Avoid universal claims about all deployed systems.

### Additional restructuring goal for this section

Split the section's internal logic clearly into:

1. **ML / learned-representation instantiation**
   - retrieval
   - embeddings
   - concept bottlenecks
   - semantic caching

2. **Formal entity-attribute witness instantiation**
   - `ACS8`, `ACS9`, `EMB1`
   - `NSL1`-`NSL6`
   - explicit note that the Lean development is optional for reading the paper but provides a machine-checked witness of the formal instantiation

### Why this matters

The reviewer suggestion is not merely to add more examples, but to give ML systems equal billing and to stop making the formal witness model feel like the only native habitat of the theorems.

---

## 7. `latex_jsait/content/06b_extensions.tex`

### Purpose
Turn the current forward-looking section into a more explicit scoping defense.

### New subsection to add near the top

Insert after the opening paragraph and before `Noisy and Lossy Observation`:

`\subsection{What This Paper Does Not Do}`

### Content to include

Use 4 short items or paragraphs:

1. **Not a full general rate-distortion theory**
   - exact closed form only in the deterministic uniform single-block corner
   - non-uniform and multi-fiber cases are not solved here

2. **Not a proof that symbols are the only possible solution**
   - injective encoders are an alternative
   - allowing nonzero error is an alternative
   - the theorem only says exact identity from a non-injective representation requires auxiliary distinguishing information

3. **Not an estimator for `A_pi` in learned systems**
   - computing collision structure from an embedding model is separate
   - the paper tells you what the quantity means once known

4. **Not a stochastic / noisy representation theory**
   - repeated observations, probabilistic confusion, and noisy side channels require different techniques

### Why this helps

This section lowers disappointment risk. It also makes the paper read as confident and honest rather than overextended.

### Relationship to existing content

Much of this is already partially present in `06b_extensions.tex`. The task is mainly to reorganize the existing caveats under a clearer heading and tighten them into direct reviewer-facing language.

---

## 8. `latex_jsait/content/07_conclusion.tex`

### Purpose
Close with the scoped theorem and the systems payoff, while preventing overclaim.

### Changes

Revise the first two paragraphs so they emphasize:

- exact zero-error law,
- exact deterministic corner distortion law,
- same residual ambiguity object,
- systems implication from steepness.

Then slightly tighten the neurosymbolic paragraph so it says:

- the theorem is about auxiliary identity bits,
- symbolic handles are the standard implementation,
- not the only abstract possibility,
- but the practical systems mechanism in the settings discussed.

### Add one explicit scoping sentence

Something like:

> The paper does not attempt a full stochastic rate-distortion theory; its contribution is to solve the clean deterministic corner exactly and show why that corner already governs many semantics-versus-identity design choices.

This can go either in the main conclusion body or at the end of the neurosymbolic paragraph.

---

## 9. Optional structural insertion in `type_compression_jsait.tex`

### Only if needed

If `Why This Is Not Obvious` should be a top-level section instead of an introduction subsection, then the file structure would need to change. This is probably unnecessary.

### Recommendation

Keep it inside the introduction as a subsection. That preserves flow and page budget.

### Structural note on top-level sectioning

No top-level section renaming is strictly required if `02_compression_framework.tex` and `06_applications.tex` are rewritten carefully. The paper can keep its current top-level skeleton while still reading as:

- abstract framework,
- generic theorem arc,
- generic consequences,
- instantiations,
- limitations and extensions.

---

## Draft Edit Sketches After Reread

These drafts are based on rereading the current `abstract.tex`, `01_introduction.tex`, `02_compression_framework.tex`, `04_kolmogorov_witness.tex`, `05_rate_distortion.tex`, `06_applications.tex`, `06b_extensions.tex`, and `07_conclusion.tex`.

They are not yet final text, but they are intended to be close enough to paste and revise.

---

### Draft A. Replacement abstract

```latex
\begin{abstract}
The question of whether semantic representations can capture identity without auxiliary identifiers is often treated as a design choice. This paper shows it is not. When distinct entities map to the same representation, no decoder can recover their identities without additional distinguishing information. The minimum worst-case auxiliary description is exactly $\log_2 A_\pi$ bits, where $A_\pi$ is the maximum collision multiplicity. If this cost is left unpaid, the distortion floor on a worst collision block is $D \ge 1 - 1/A_\pi$.

The paper isolates the clean deterministic corner of this problem. On a uniform collision block of size $a$ with auxiliary budget $L$ bits, the optimal distortion is exactly $D^\star(L)=\max(0,1-2^L/a)$. The adaptive fiberwise budget is $\lceil \log_2 |\pi^{-1}(u)| \rceil$ bits. The underlying mathematics is elementary once the problem is posed correctly; the contribution is showing that a question often treated as architectural opinion is governed by exact coding constraints.

We then instantiate these generic laws in a concrete formal identity domain and show that a neural encoder paired with a symbolic handle achieves zero-error identity recovery if and only if the handle distinguishes entities within each semantic fiber (\LH{NSL1}--\LH{NSL6}). The identity bits must be paid somewhere; symbolic layers are the standard computational mechanism for paying them.

\textbf{Keywords:} semantics-aware compression, zero-error coding, learned representations, neurosymbolic systems, side information
\end{abstract}
```

**Reasoning:**
- This keeps the deterministic-corner scope explicit.
- It leads with the design-choice vs theorem contrast reviewers want.
- It makes the distortion floor feel central without pretending to solve full rate-distortion theory.
- It preserves the instantiated formal-domain payoff while not forcing the abstract to open with the PL witness model.

---

### Draft B. New introduction subsection: `Why This Is Not Obvious`

```latex
\subsection{Why This Is Not Obvious}

The core coding fact is simple once the problem is posed correctly. If $A_\pi$ classes share the same representation value, then distinguishing them exactly requires at least $\log_2 A_\pi$ bits. In that sense, the proof is not the main difficulty. The contribution is identifying the correct pose.

Several things make that pose easy to miss. First, rich semantic representations often feel as though they should capture identity as a special case. The paper shows this is false whenever collisions exist: once distinct entities lie in the same representation fiber, no decoder can recover identity from the representation alone. Second, many systems are designed and evaluated in effectively closed worlds, where collisions can be engineered away or ignored. The coding laws matter precisely when the world is open or the representation is fixed and not built for exact identity.

Third, error tolerance can mask the issue. Many applications accept some misidentification, so the failure mode can seem mild. The deterministic-corner distortion theorem shows that the penalty is not merely ``some error'' but a quantitatively sharp lower bound. Finally, the bill is often paid implicitly. Systems that appear to rely on pure representations usually carry hidden identifiers in the background: retrieval keys, document IDs, memory addresses, cache keys, or registry entries. The role of the paper is to make explicit what systems often pay implicitly or reason about incorrectly.
```

**Reasoning:**
- This directly answers the “isn’t this obvious?” reviewer objection.
- It honestly says the proof idea is simple, while defending the real contribution: posing the question correctly and tracing its consequences.
- It provides a bridge from theoremic simplicity to practical importance.

---

### Draft C. New opening for `02_compression_framework.tex`

```latex
\subsection{The Abstract Coding Framework}

We study a finite class space $\mathcal C$, a representation map
\[
\pi : \mathcal C \to \mathcal U,
\]
and the induced side information $U=\pi(C)$ observed by the decoder. The encoder may additionally transmit an auxiliary description $M$. The coding question is: how many auxiliary bits are needed to recover $C$ exactly, or approximately, once the representation $U$ is already known?

For each representation value $u \in \mathcal U$, the corresponding fiber is
\[
\pi^{-1}(u) = \{c \in \mathcal C : \pi(c)=u\}.
\]
The collision geometry of these fibers is the paper's central object. In particular, we write
\[
A_\pi := \max_u |\pi^{-1}(u)|
\]
for the largest collision multiplicity. No further structure is assumed. The map $\pi$ may come from a learned encoder, a schema projection, a feature extractor, or a hand-designed attribute family.
```

Then transition into the current entity/attribute material with:

```latex
\subsection{A Formal Entity-Attribute Instantiation}

One important witness model realizes the abstract representation map through an explicit entity domain and a finite family of declared attributes. This formal instantiation is retained because it supports the paper's mechanized witness theorems and the later neurosymbolic specialization, but the main coding laws depend only on the induced representation map and its fibers.
```

**Reasoning:**
- This is the main architectural fix.
- It lets the entire theorem arc read as generic IT.
- It preserves the formal witness model while removing it from the paper’s conceptual front door.

---

### Draft D. Genericized opening for `04_kolmogorov_witness.tex`

```latex
\subsection{Fixed-Length Zero-Error Recovery}

We now state the zero-error coding problem in generic representation-map form. Let $C$ be a finite class variable and let $U=\pi(C)$ be the representation observed by the decoder. If the representation is non-injective on classes, then the decoder cannot determine $C$ from $U$ alone; the unresolved ambiguity is exactly the ambiguity inside the fibers of $\pi$.
```

Potential generic theorem restatement:

```latex
\begin{theorem}[Zero auxiliary description iff injective representation]\label{thm:identification-capacity}
Let $\mathcal C$ be a finite class space and let $\pi : \mathcal C \to \mathcal U$ be a representation map. Zero-error recovery of $C$ from $U=\pi(C)$ with no auxiliary description is achievable if and only if $\pi$ is injective on classes.
\end{theorem}
```

**Reasoning:**
- The current proof is already generic in spirit.
- The present notation still smells like attribute profiles and programs.
- Rewriting this section generically would do more than any late-stage application prose to signal JSAIT fit.

---

### Draft E. Stronger deterministic-corner framing in `05_rate_distortion.tex`

Replace the local intro with something close to:

```latex
\subsection{The Deterministic Corner: Semantic Identity Rate-Distortion on a Collision Block}

The zero-error threshold does not exhaust the geometry. We now characterize the exact distortion-bits tradeoff in the cleanest deterministic setting: a uniform source on a single collision block. This isolates the role of collision geometry alone, separate from source statistics and cross-fiber interference. In that setting, the full tradeoff admits a closed form.
```

Add a stronger steepness remark after the current `Why this matters` discussion:

```latex
\begin{remark}[Steepness]
The practical importance of the deterministic corner is the steepness of the curve. On a collision block of size $100$, zero distortion requires at least $\lceil \log_2 100 \rceil = 7$ bits. With only $6$ bits the distortion is at least $36\%$, and with only $5$ bits it is at least $68\%$. Small deficits below the identity threshold can therefore cause catastrophic losses in exact identity accuracy.
\end{remark}
```

**Reasoning:**
- The current section already has the theorem, corollaries, omission remark, and scope remark.
- What is still missing is a more vivid explanation of why this restricted result matters.
- The steepness argument gives the systems-design payoff without overstating the theorem's scope.

---

### Draft F. New paragraph in `06_applications.tex`: `What Systems Get Wrong`

```latex
\paragraph{What systems get wrong.}
Several common design patterns implicitly violate the coding laws. Pure embedding retrieval without identifiers cannot guarantee exact retrieval on collided fibers; if two documents share the same embedding, distinguishing metadata must be stored somewhere else. Concept bottleneck systems without concept-level identifiers cannot recover which ground-truth instance generated a collapsed concept representation. Semantic caches keyed only by semantic hashes can preserve approximate meaning while still returning the wrong identity when collisions occur. In each case, the coding laws say exactly what is missing: an auxiliary channel that separates the collided fiber, and, if that channel is omitted, the corresponding distortion floor.
```

**Reasoning:**
- The current applications section already has positive systems implications.
- This adds the negative systems reading reviewers asked for.
- It grounds the claims in specific design mistakes without saying every such system is broken in all settings.

---

### Draft G. New subsection in `06b_extensions.tex`: `What This Paper Does Not Do`

```latex
\subsection{What This Paper Does Not Do}

The paper does not provide a full general rate-distortion theory for non-uniform sources or general distortion measures. The exact closed-form tradeoff derived here belongs to the deterministic corner of a uniform source on a single collision block. Extensions beyond that setting are natural, but they require different techniques and produce source-sensitive expressions.

The paper also does not prove that symbolic handles are the only possible resolution of identity ambiguity. An injective encoder is an alternative, and allowing nonzero error is another alternative. The theoremic claim is narrower and cleaner: if exact identity is required from a non-injective representation, some auxiliary distinguishing information must pay the missing bits.

Nor does the paper solve the computational problem of estimating $A_\pi$ or fiber geometry from an arbitrary learned model. The coding laws identify the exact meaning of those quantities once they are known. Extracting them from a neural representation is a separate representation-analysis problem.

Finally, the paper does not treat stochastic representations, noisy side channels, or repeated probabilistic observations. Those settings raise different questions and belong to a broader stochastic theory beyond the deterministic corner solved here.
```

**Reasoning:**
- This consolidates caveats already scattered through the paper.
- It lowers reviewer disappointment risk.
- It makes the paper sound more precise, not weaker.

---

### Draft H. Conclusion tightening

Suggested replacement for the first two paragraphs:

```latex
This paper studies what remains to be coded once a semantic representation is fixed. The answer is residual ambiguity: the collision fibers of the representation determine the exact cost of recovering identity. In the zero-error regime, that cost is the semantic identity rate $\log_2 A_\pi$. In the adaptive regime, it becomes the fiberwise budget $\lceil \log_2 |\pi^{-1}(u)| \rceil$. In the clean deterministic lossy corner, it becomes an exact distortion curve on a collision block.

The main conceptual point is that these are not different problems but different faces of the same one. If the representation is injective, identity is free. If it is non-injective, the missing information must be paid for somewhere. One may pay for it in auxiliary bits, in helper views, in additional disclosures, in query effort, or in distortion. The paper's contribution is to make that accounting exact.
```

**Reasoning:**
- The current conclusion is already strong, but this version makes the single-spine structure even more explicit.
- It sets up the neurosymbolic paragraph as a systems payoff of the same accounting identity.

---

### Draft I. Optional explicit instantiation bridge in `06_applications.tex`

If the section is refactored into explicit instantiations, use a bridge sentence like:

```latex
The abstract coding framework applies to any representation map. We now develop two instantiations in parallel: first the learned-representation setting that motivates many modern systems, and then a formal entity-attribute model that serves as a machine-checked witness of the same laws.
```

**Reasoning:**
- This sentence cleanly explains why the section contains both ML and Lean/PL material.
- It prevents the formal witness model from feeling like an abrupt genre shift.

---

## Mechanization Impact

### No new Lean proofs required

The requested edits are framing and scope edits, not new theoremic content.

Already sufficient:

- zero-error law
- adaptive law
- semantic identity rate-distortion theorem on uniform collision blocks
- zero-error threshold corollary
- zero-budget error floor corollary
- NSL instantiation theorems

### Optional Lean polish only

Only if desired later:

- create a dedicated alias/handle for the zero-error-threshold corollary
- create a dedicated alias/handle for the zero-budget corollary

This is optional presentation polish, not a proof requirement.

---

## Execution Order

1. Recast `abstract.tex`
2. Add `Why This Is Not Obvious` to `01_introduction.tex`
3. Rewrite the front of `02_compression_framework.tex` so the abstract coding framework comes first
4. Genericize `04_kolmogorov_witness.tex` wherever the theorem statements only depend on fibers and counting
5. Tighten deterministic-corner framing in `05_rate_distortion.tex`
6. Add `What systems get wrong` and sharpen the ML-first instantiation flow in `06_applications.tex`
7. Add `What This Paper Does Not Do` to `06b_extensions.tex`
8. Tighten `07_conclusion.tex`
9. Rebuild paper
10. Verify references, page count, and section flow
11. Commit LaTeX source changes only

---

## Success Criteria

After these edits, a skeptical reviewer should come away with four impressions:

1. the paper knows exactly what it proves and what it does not prove,
2. the deterministic rate-distortion result is intentionally scoped, not accidentally limited,
3. the simplicity of the core proof is acknowledged honestly,
4. the systems-level symbolic claim is grounded in real failure modes rather than abstract rhetoric.

If those four land, the paper will read as sharper, more honest, and more persuasive.

The stronger structural success criterion is this:

- a non-PL reader should be able to read the abstract, introduction, and theorem sections without needing to know anything about attribute families or admissibility contracts;
- a formal-methods reader should then encounter the entity-attribute / Lean model as a serious witness instantiation of the generic coding laws, not as the only way to understand them.
