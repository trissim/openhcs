# Paper 1 JSAIT Revision Plan

This note is a concrete editing plan for revising the JSAIT version of Paper 1. It is not a response letter. The goal is to improve the paper itself by changing the frame, tightening the structure, and surfacing the strongest real contribution already present in the manuscript.

## Main framing change

The paper should stop reading as if its main novelty is the counting theorem alone.

Instead, the revised frame will be:

1. the paper identifies **residual ambiguity** under a fixed representation as the exact primitive governing zero-error recovery;
2. the basic coding laws for that primitive are the clean finite setup layer;
3. the main structural payoff is a **representation-sufficiency theory** for downstream tasks, helper views, retrieval-style augmentation, and modular/factorized representations.

In other words:

- the fixed-length, block, adaptive, and distortion results stay in the paper and stay important;
- but they become the exact finite coding backbone, not the whole sales pitch;
- the learned-representation / neuro-symbolic / downstream-task results become a central theoremic arc rather than a late interpretive afterthought.

## What this plan is trying to fix

The current draft has three real issues.

### 1. Scope mismatch

The paper often sounds like a broad modern compression / ML paper, but the actual theoremic setting is deterministic, finite, and exact. That makes reviewers read the ML connection as motivational only.

### 2. Hidden second contribution

The strongest nontrivial learned-representation content is currently buried in `latex_jsait/content/06_applications.tex` as prose. In particular, the following are real theoremic ingredients, not just interpretation:

- task-level sufficiency via `A_{\pi \to Y}`;
- monotonicity under representation coarsening;
- helper-view / pair-view recovery laws;
- factorized-product laws for modular representations.

These need to be surfaced as formal results.

### 3. Dilution of the compression arc

The core coding story is clean, but it is spread across a heavy model section and then followed by secondary query/robustness sections that currently read as partially separate projects.

The fix is not to remove those sections, but to trim them and tie them more explicitly back to residual ambiguity as the common primitive.

## High-level revision goals

### Goal A: Make the paper read as two connected theoremic arcs

Arc 1:

- exact residual-coding laws for a fixed representation.

Arc 2:

- exact representation-sufficiency laws for downstream tasks, auxiliary views, and modular/factorized representations.

### Goal B: Keep JSAIT fit without overselling

The revised paper should still look like a semantics-aware compression paper, but one that is honest about its setting:

- deterministic;
- finite explicit;
- zero-error;
- representation-side-information driven.

### Goal C: Make the neuro-symbolic / learned-representation contribution visible

The draft already has genuine content here. The job is to promote it from discussion to theoremic structure.

## Existing formal support for Arc 2

This revision is not trying to invent a second arc from scratch. The second arc already exists formally and is only weakly surfaced in the manuscript.

### Existing sources

- `proofs/Paper1IT/GraphEntropy.lean`
  - bottleneck / coarsening monotonicity;
  - exact task recoverability;
  - helper-view / pair-view recovery laws;
  - factorized / product ambiguity laws;
  - active clarification links.

- `proofs/HandleAliases.lean`
  - paper-visible handles for the above claims, especially `GPH38`--`GPH43`.

- `latex_jsait/content/04_kolmogorov_witness.tex`
  - current bottleneck monotonicity corollary.

- `latex_jsait/content/05_rate_distortion.tex`
  - adaptive coding, entropy sandwich, active clarification bridge, and transcript-factorization bridge.

- `latex_jsait/content/06_applications.tex`
  - current prose exposition of task sufficiency, helper views, modular structure, and neuro-symbolic interpretation.

- `latex_jsait/content/appendix_lean.tex`
  - confirms the mechanized scope of these theorem families.

### Consequence for the revision

The revision should therefore be framed as exposing and centralizing an already-existing theorem family, not as merely polishing an applications section.

## Target revised visible section roles

- `02_compression_framework.tex`: concise model specification and basic barrier.
- `04_kolmogorov_witness.tex`: exact fixed-length and block coding backbone.
- `05_rate_distortion.tex`: adaptive, entropy, distortion, and clarification consequences.
- `06_applications.tex`: main representation-sufficiency arc.
- `03_matroid_structure.tex`: secondary query-side consequence of residual ambiguity.
- `04a_complexity_bounds.tex`: secondary robustness consequence under world growth.
- `06b_extensions.tex`: forward-looking learned-representation and modern-compression outlook.

## Planned edits by file

## 1. `latex_jsait/content/abstract.tex`

### Planned change

Rewrite the abstract so that it presents the coding laws as the exact primitive and the representation-sufficiency consequences as the broader payoff.

### Concrete edit direction

- Keep the fixed-length theorem, block law, adaptive law, and distortion floor.
- Add one sentence making clear that the same residual ambiguity object also yields exact task-sufficiency and helper-view consequences for learned / neuro-symbolic representations.
- Avoid making the abstract sound like the paper is claiming a new general graph-entropy theory or a neural compression algorithm.

### Draft edit language

Candidate opening sentence:

> We study a deterministic zero-error side-information problem in which a decoder observes a fixed representation $U=\pi(C)$ and an encoder transmits only the residual information needed to recover the class $C$ exactly.

Candidate closing sentence for the abstract:

> Beyond the coding laws themselves, the same ambiguity-fiber geometry yields exact task-sufficiency, helper-view, and factorized-representation consequences for learned and neuro-symbolic systems.

### Current wording to replace

Current opening emphasis:

> We study exact class identification from a finite representation $U=\pi(C)$ when the representation is not injective and an encoder may transmit additional side information $M$.

Planned replacement emphasis:

> We study a deterministic zero-error side-information problem in which a decoder observes a fixed representation $U=\pi(C)$ and an encoder transmits only the residual information needed to recover the class $C$ exactly.

### Why this helps

The abstract currently makes the paper sound mostly like a simple coding note with an ML interpretation at the end. The revision will make the representation-theoretic contribution visible from the first paragraph.

## 2. `latex_jsait/content/01_introduction.tex`

### Planned change

Rebuild the introduction around the simple `(C,U,M)` model first, then explain the attribute-family realization as one concrete instantiation.

### Concrete edit direction

- Open with the simple model: class `C`, representation `U = \pi(C)`, auxiliary description `M`.
- State early that the paper asks for the exact residual burden left by a noninjective representation.
- Keep the attribute-family language, but reposition it as one explicit realization of the representation map rather than the conceptual front door.
- Rewrite the contributions list into two explicit arcs:
  1. residual coding laws;
  2. representation sufficiency laws for tasks, helper views, and modular representations.
- State once, explicitly, that Arc 1 is the exact finite backbone of the paper rather than the sole novelty claim; the stronger newer payoff is the structural yield developed in Arc 2.
- Tone down novelty language around the fixed-length theorem.
- Tone up the theoremic status of the representation-side results already in the paper.
- Name the Arc 2 theorem families explicitly in the introduction rather than gesturing at them abstractly:
  - task sufficiency;
  - bottleneck monotonicity;
  - helper-view fusion;
  - factorized/modular product laws.

### Draft edit language

Candidate opening transition:

> The paper should be read in two connected arcs. First, it gives exact coding laws for the residual ambiguity left by a fixed representation. Second, it shows that the same ambiguity-fiber object yields exact representation-sufficiency laws for downstream tasks, helper views, and modular representations.

Candidate sentence distinguishing setup from payoff:

> The first arc is the exact finite backbone of the theory; the paper's broader structural payoff is that the same primitive yields nontrivial sufficiency laws for downstream reasoning under learned or fixed representations.

Candidate sentence for the prior-work paragraph:

> In classical terms, the paper is best read as a deterministic, representation-centered specialization of zero-error source coding with decoder side information, where the confusability structure is induced directly by the fibers of $\pi$.

Candidate contribution bullet headings:

1. Exact residual-coding laws.
2. Fiberwise adaptive and entropy-comparison laws.
3. Below-threshold distortion consequences.
4. Representation sufficiency for tasks, helper views, and factorized structures.
5. Query and stability costs of unresolved ambiguity.

### Current wording to replace

Current contribution emphasis:

> The contribution is twofold. First, the paper gives exact zero-error side-information laws for recovering a class from a fixed representation. Second, it identifies representation-induced ambiguity fibers as a reusable invariant that gives a common coding language for semantic identification settings in which a useful representation does not by itself determine exact identity.

Planned replacement emphasis:

> The paper should be read in two connected arcs. First, it gives exact coding laws for the residual ambiguity left by a fixed representation. Second, it shows that the same ambiguity-fiber object yields exact representation-sufficiency laws for downstream tasks, helper views, and modular representations.

Current contribution bullet 4 is too compressed. It currently bundles:

> task sufficiency, helper views, factorized settings, and representation interpretation

into one long paragraph-level bullet. The revision should split these into a visibly theoremic sequence with cross-references.

### Why this helps

This directly addresses the current structural problem: readers reach the paper through too much model machinery and too little immediate sense of what the strongest contribution is.

## 3. `latex_jsait/content/02_compression_framework.tex`

### Planned change

Reshape the section so that it starts from the representation-side-information model and only then introduces the attribute-generated realization.

### Concrete edit direction

- Add a short opening subsection or opening paragraph that explicitly says:
  - the core object is `(C,U)` with `U = \pi(C)`;
  - the theoremic coding arc depends only on the representation and auxiliary description;
  - the attribute-family formalism is retained because it supports the semantic-identification interpretations.
- Trim repeated motivational remarks that restate the information barrier in multiple ways.
- Keep the admissibility contract, because it is actually useful and reviewers liked it.
- Make the section feel like a concise model specification rather than a competing paper.

### Draft edit language

Candidate opening paragraph:

> The theoremic core of the paper depends only on a class variable $C$, a deterministic representation $U=\pi(C)$ available at the decoder, and any charged auxiliary description $M$. The attribute-family formalism introduced below is one explicit realization of this representation-side-information model and is retained because it supports the broader semantic-identification interpretations later in the paper.

Candidate shortened role paragraph after the barrier theorem:

> The barrier theorem is the one invariance statement the later sections need: if two classes remain in the same representation fiber, no admissible scheme can separate them without importing new information.

### Current wording to replace

Current opening sentence:

> The introduction has already motivated the side-information viewpoint. For the main theorem arc, the primitive object is the pair $(C,U)$ with $U=\pi(C)$.

Planned replacement emphasis:

> The theoremic core of the paper depends only on a class variable $C$, a deterministic representation $U=\pi(C)$ available at the decoder, and any charged auxiliary description $M$.

The revision should keep the current coding-first move but make it sharper and more economical.

### Why this helps

This keeps all the formal machinery you want, but stops it from overwhelming the reader before the compression story starts.

## 4. `latex_jsait/content/04_kolmogorov_witness.tex`

### Planned change

Keep the fixed-length and block results, but present them more explicitly as the exact finite coding backbone for the later representation-sufficiency theory.

### Concrete edit direction

- Keep theorems and proofs largely intact.
- Add one concise paragraph acknowledging that in graph terms the confusability structure here is the profile-fiber special case, so the exact law collapses to maximum fiber size.
- Make clear that the point of this section is not to claim a new general zero-error graph theorem, but to isolate the right primitive for the fixed-representation setting.
- Keep `cor:concept-bottleneck-mono`, but point forward to its later role in the learned-representation section.

### Draft edit language

Candidate paragraph after the converse setup:

> In graph terms, the present fixed-representation setting is a structured zero-error special case: confusable classes are exactly those lying in the same representation fiber, so the confusability graph is a disjoint union of cliques. The point of the section is therefore not a new general graph-entropy theorem, but the identification of the exact primitive governing this deterministic side-information regime and the theoremic consequences that follow from it.

### Current wording to replace

Current paragraph already moves in this direction, but it is still too easy to skim as a side remark after the theorem. The revised version should be promoted into a visibly central positioning paragraph immediately around the converse setup, not left as a soft explanatory aside.

### Why this helps

This takes the force out of the "elementary counting argument" criticism without weakening the paper. It also prepares the reader to see the later representation theorems as consequences of the same primitive, not as afterthoughts.

## 5. `latex_jsait/content/05_rate_distortion.tex`

### Planned change

Keep the adaptive and distortion results as part of the core backbone, but make their role in the paper more explicit.

### Concrete edit direction

- Keep the exact pointwise fiber law and the expected-length lower bound.
- Keep the entropy sandwich, but phrase it more carefully as a classical conditional-coding comparison rather than a standalone novelty claim.
- Keep the positive-distortion floor as the finite below-threshold consequence of residual ambiguity.
- Strengthen the transitions at the end so that the section points forward to representation quality and task sufficiency.
- Explicitly position the section as linking residual ambiguity not only to coding cost, but also to adaptive clarification / decision-tree-style resolution.

### Draft edit language

Candidate transition sentence near the end:

> The fixed-length theorem, the pointwise adaptive law, and the entropy sandwich are three resolutions of the same object: the cost of naming the classes that remain collapsed inside each representation fiber.

Candidate sentence before the applications section:

> The next section turns this coding object into a representation-sufficiency theory: once the residual ambiguity of each fiber is explicit, one can ask exactly which downstream tasks, helper views, and modular factors eliminate it and which do not.

### Current wording to replace

Current transition is too mild. The section currently ends with a secondary-frontier sentence. The revision should make the forward pointer to Arc 2 explicit and theoremic, so the reader expects the downstream representation results before reaching `06_applications.tex`.

### Why this helps

This preserves the compression core while avoiding overclaiming on the most standard source-coding parts.

## 6. `latex_jsait/content/06_applications.tex`

### Planned change

This will become the paper's highlighted second theoremic arc, not just an applications section.

### Concrete edit direction

- Rename the section to something closer to:
  - `Representation Sufficiency, Auxiliary Views, and Modular Structure`, or
  - `Representation-Level Consequences of Residual Ambiguity`.
- Formalize and centralize the main claims in this section into an explicit theorem/proposition/corollary sequence. This is mandatory, not optional.
- Make sure each promoted result is explicitly cross-referenced from the introduction contribution list by theorem/corollary number, so the second arc is visible before the reader reaches this section.

At minimum, formalize and foreground the following sequence:

1. **Task-sufficiency criterion**
   - exact recovery of `Y = f(C)` from `U` iff each representation fiber carries at most one task label;
   - define `A_{\pi \to Y}` and make the criterion look central, not incidental.

2. **Monotonicity under coarsening**
   - make clear that coarsening cannot improve task-level sufficiency;
   - tie this directly to concept bottlenecks and learned representations.

3. **Helper-view / pair-view theorem**
   - formalize the law for an additional deterministic view `S = \sigma(C)`;
   - present this as a theoremic model of retrieval augmentation, helper features, or second-modality support.

4. **Helper-view monotonicity**
   - make explicit that adding a helper view can only reduce task ambiguity;
   - tie this directly to deterministic context fusion.

5. **Factorized / modular representation law**
   - formalize the product case rather than leaving it as prose;
   - state clearly when multiplicative ambiguity is real and when architectural modularity is representationally empty.

- After each formal statement, add a short interpretation paragraph for learned representations / neuro-symbolic systems.
- Make explicit in the plan that `GPH38`--`GPH43` should each correspond to a visible labeled result in the LaTeX body.

### Why this helps

This is the single most important revision. It surfaces the paper's strongest currently-hidden contribution and changes the reading from:

- "simple coding theorem plus interpretation"

to:

- "exact residual-coding primitive plus a real downstream representation theory."

### Draft edit language

Candidate section opening:

> The coding laws above quantify how much ambiguity a fixed representation leaves unresolved. We now turn that same fiber geometry into a representation-sufficiency theory: which downstream tasks factor through the representation, when auxiliary views genuinely help, and when modular latent structure adds exact information rather than architectural redundancy.

Candidate theorem-format sentence for task sufficiency:

> For a downstream task label $Y=f(C)$, the representation alone supports zero-error recovery of $Y$ if and only if each representation fiber carries at most one task label.

Candidate theorem-format sentence for helper views:

> If the decoder receives an additional deterministic view $S=\sigma(C)$, then exact task recovery from $(U,S)$ is possible exactly when each joint fiber carries at most one task label; adding such a view can only reduce task ambiguity.

Candidate theorem-format sentence for factorized structure:

> In a genuinely factorized product setting, the joint fibers factor as products of the component fibers, so the worst-case residual ambiguity multiplies exactly across modules.

### Current wording to replace

Current status of this section:

> it already contains the theorem family in prose, but the prose still reads like explanation rather than a sequence of named results.

Planned restructuring:

- introduce the section as Arc 2, not as a narrative applications section;
- give each of the following its own visible theorem/proposition/corollary label:
  1. task sufficiency;
  2. task-level coarsening monotonicity;
  3. helper-view exact recovery;
  4. helper-view monotonicity;
  5. factorized product law.

The plan should be executed as reclassification of existing mathematics, not as stylistic polishing.

## 7. `latex_jsait/content/03_matroid_structure.tex`

### Planned change

Keep the section, but trim and recast it as a secondary operational consequence of residual ambiguity.

### Concrete edit direction

- Tighten the opening so it states plainly that this is a secondary resource law: if one does not transmit the missing information, one may have to pay in queries.
- Keep the minimum distinguishing number result.
- Shorten the matroid discussion if possible, especially explanatory passages that feel like a different paper.
- Reduce emphasis on the unrestricted-vs-structured query-family distinction unless it directly serves the main message.
- Make the final comparison table explicitly tie side-information bits and query cost back to the same residual ambiguity story.
- Target trim level: moderate-to-strong. Keep the theorem statements, one compact intuition paragraph, and the comparison table, but aim to cut roughly 25--35\% of the current expository bulk so the section does not read like a parallel paper.

### Draft edit language

Candidate opening sentence:

> This section should be read as the query-cost analogue of the coding laws above: if one does not transmit the residual information needed to resolve a fiber, one may instead have to pay for it through adaptive interrogation.

### Current wording to replace

Current opening is already close, but still reads as “secondary query-side theorem family.” The revised version should preserve the secondary status while more strongly linking it to the same debt/ambiguity language used everywhere else.

### Why this helps

The section can remain in the paper without competing with the main compression + representation-sufficiency arcs.

## 8. `latex_jsait/content/04a_complexity_bounds.tex`

### Planned change

Keep the robustness section, but shorten it and connect it more directly to representation sufficiency under system growth.

### Concrete edit direction

- Reframe the section as: even a currently sufficient representation may fail to remain sufficient under world growth.
- Keep the open-world instability theorem.
- Keep the Rice-style noncomputability theorem only if it is stated compactly and clearly tied to the practical point: persistent collision-freedom cannot be certified in full generality.
- Trim explanatory material that reads like a separate computability paper.
- Target trim level: moderate. Keep both theorem statements and one short interpretation remark each, but compress proofs/exposition enough that the section reads as a consequence chapter rather than a standalone undecidability note.

### Draft edit language

Candidate opening sentence:

> The coding laws in the main arc characterize residual ambiguity for a fixed realized representation. The question here is whether that ambiguity profile is stable under world growth, or whether a currently sufficient representation can later become ambiguous as new classes appear.

### Current wording to replace

Current section opening is directionally right. The planned edit is mainly a compression of proof/exposition so the section reads like a consequence chapter rather than a separate computability mini-paper.

### Why this helps

This preserves the conceptual value of the robustness section while making it serve the central representation story rather than pulling the paper into a separate undecidability direction.

## 9. `latex_jsait/content/06b_extensions.tex`

### Planned change

Keep this as a forward-looking bridge to learned representations and modern compression, but do not let it carry the burden of the paper's modernity argument.

### Concrete edit direction

Add a substantial subsection on **estimating or auditing residual ambiguity in learned representations**. It should cover:

- exact fiber counting in finite explicit registries;
- collision audits after discretization or quantization of continuous embeddings;
- lower and upper bounds on `A_\pi` when exact enumeration is not feasible;
- task-conditioned surrogates such as `A_{\pi \to Y}`;
- empirical fiber-profile auditing as a practical target even when theoremic guarantees are unavailable;
- why computing or approximating `A_\pi` for implicit models is a separate computational problem.

Also add a tighter paragraph on **rate-distortion-perception**:

- present your distortion floor as a deterministic exact-recovery analogue, not as a solved rate-distortion-perception theorem;
- explain what is missing for a full stochastic theory.

### Draft edit language

Candidate learned-representation paragraph:

> In explicit finite registries, auditing residual ambiguity means computing fiber statistics exactly. In learned or implicit models, the same quantities become representation-analysis targets: one may estimate collision profiles only after discretization, quantization, certified bounding, or task-conditioned surrogates such as $A_{\pi \to Y}$. The theoremic contribution of the paper is to identify what these quantities mean once they are available, not to solve the computational problem of extracting them from arbitrary models.

### Current wording to replace

Current extensions section is too short and generic to carry the “modern advances” burden by itself. The planned replacement should make it a disciplined support section, not a speculative grab-bag, and should explicitly defer algorithmic estimation while still naming concrete audit-style directions.

Candidate rate-distortion-perception sentence:

> The present distortion floor should therefore be presented as a deterministic exact-recovery analogue of rate-distortion-perception constraints, not as a full solution of that stochastic theory.

### Why this helps

This is useful supporting material, but the strongest modern contribution should come from the already-mechanized theorem family in `06_applications.tex`, not from speculative extensions.

## 10. `latex_jsait/content/07_conclusion.tex`

### Planned change

Rewrite the conclusion so that it mirrors the new two-arc structure.

### Concrete edit direction

- First paragraph: residual ambiguity as the exact finite primitive.
- Second paragraph: coding laws.
- Third paragraph: representation sufficiency, helper views, and modular structure as the broader structural consequence.
- Final paragraph: query and robustness as secondary consequences of the same primitive.

### Draft edit language

Candidate first sentence:

> The paper contributes exact coding laws for the residual ambiguity left by a fixed representation and shows that the same ambiguity-fiber object supports a broader representation-sufficiency theory for downstream tasks, helper views, and modular structure.

Candidate final paragraph opening:

> Query costs and robustness do not begin new stories; they record other ways the same unresolved ambiguity reappears when it is not paid for directly in side-information bits.

### Current wording to replace

Current conclusion already has the right direction, but it still needs a cleaner Arc 1 / Arc 2 split:

- paragraph 1: residual ambiguity as primitive;
- paragraph 2: coding backbone;
- paragraph 3: representation sufficiency / helper views / modular structure;
- final paragraph: query and robustness as other costs of the same primitive.

### Why this helps

The paper should end on the strongest theoremic contribution, not only on the coding arc.

## 11. Classical positioning spine

### Planned change

Add one explicit positioning pass that makes the paper's classical lineage impossible to miss.

### Concrete edit direction

- Add a short paragraph in the introduction / related-work section explicitly stating that the paper is best read as a deterministic, representation-centered specialization of zero-error source coding with decoder side information.
- Make the Witsenhausen / Körner lineage more explicit.
- State plainly that the fixed-length theorem is the disjoint-union-of-cliques special case.
- State equally plainly that the real novelty lies in:
  - the representation-centered formulation;
  - the ambiguity-fiber invariant;
  - the exact fixed-length, adaptive, distortion, task-sufficiency, helper-view, and factorized-product consequences.
- Strengthen the distinction from Ahlswede-Dueck identification and from general graph-entropy claims.

### Draft edit language

Candidate positioning paragraph:

> The paper is best read as a deterministic, representation-centered specialization of zero-error source coding with decoder side information. Its fixed-length converse is the disjoint-union-of-cliques special case of the classical confusability picture. The novelty lies not in claiming a new general graph theorem, but in isolating representation fibers as the exact residual ambiguity primitive and deriving from that object a unified law family spanning fixed-length coding, adaptive coding, distortion, task sufficiency, helper-view fusion, and factorized modular structure.

### Current wording to replace

Current related-work prose already gestures toward Witsenhausen / Körner and Slepian-Wolf, but the plan should make this paragraph mandatory and central rather than optional. This is the paragraph that should carry the strongest answer to the “elementary special case” criticism.

### Why this helps

This is now the strongest defensible answer to the "elementary special case" critique, and it lets the paper be both more modest and more forceful at the same time.

## Planned narrative after revision

If the edits succeed, the paper should read roughly like this:

1. A noninjective representation leaves residual ambiguity.
2. That ambiguity has exact finite coding laws.
3. The same ambiguity quantity governs when a representation is sufficient for exact downstream tasks.
4. Extra views, retrieval augmentation, or modular factors help exactly insofar as they refine those ambiguity fibers.
5. Query costs and robustness are secondary ways the same unresolved ambiguity reappears.
6. The whole picture is explicitly situated inside classical zero-error side-information theory, specialized to deterministic representations.

That is a much stronger and more defensible paper than a draft centered mainly on the counting theorem.

## Editing order

Recommended order of implementation:

1. Rewrite `abstract.tex` and `01_introduction.tex`.
2. Reshape `02_compression_framework.tex` so the simple model comes first.
3. Make `06_applications.tex` a formal theorem sequence and centralize the second arc.
4. Tighten framing in `05_rate_distortion.tex`, especially the active-clarification bridge.
5. Add the classical-positioning paragraph to `04_kolmogorov_witness.tex`.
6. Trim and reconnect `03_matroid_structure.tex` and `04a_complexity_bounds.tex`.
7. Strengthen `06b_extensions.tex` with a disciplined learned-representation discussion.
8. Rewrite `07_conclusion.tex`.

## Non-goals for this revision pass

This revision plan does **not** assume we will add:

- new experiments on real neural embeddings;
- a full stochastic noisy-representation theory;
- a new general graph-entropy theorem;
- a full privacy formalism;
- a full hardness theorem for computing `A_\pi`.

Those may be follow-on work. The immediate goal is to make the current paper accurately present its strongest existing contribution.
