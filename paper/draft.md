# Throw It a Banana: Lexical Transfer of Physically-Acquired Competence in RL-Finetuned Language Models

*Do word-trained priors carry skills learned by acting?*

**Status: working draft — Experiment 1 complete (runs 2–3 + diagnostics);
Experiment 2 redesign in progress (see §4).**

## Abstract

A language model that has never caught anything knows a great deal about
falling fruit. We ask whether that knowledge becomes *competence* when a small
LLM is finetuned with outcome-based reinforcement learning in a toy physical
environment — and whether competence acquired through interaction generalizes
along semantic lines laid down by text alone. We finetune Qwen3-0.6B with
hand-rolled multi-turn GRPO (~400 lines of PyTorch) on a text-interfaced
catching game: three training fruits with distinct drift dynamics, plus a
five-word held-out battery — a matched real word (**banana**), a
misleading near-synonym (tangerine), a neutral word (plum), and two nonce
words (blorple, quorf) that cannot know their own dynamics. Outcome RL
works emphatically: by 300 steps every training fruit reaches 0.86–0.98
from base rates as low as 0.08. But the transfer question resolves
*negatively, and diagnostically*: the entire held-out battery — nonce words
included, misleading condition included — converges to the same band
(0.86–0.98), a turn-1 probe shows the policy's name-conditioned action
biases collapsing to zero rather than sharpening, and the chain-of-thought
atrophies to empty strings while catch rates climb. The trained policy is a
name-blind reactive controller. The mechanism is instructive: the
environment's drift is observable and correctable within the horizon, so
reward never priced in the name, and KL-regularized RL found the cheapest
policy consistent with reward — discarding both the reasoning and the
prior's name-linked structure. We derive a design requirement for
lexical-transfer experiments that we believe is general: alongside (1) a
learned binding outside the prior and (2) a transfer cue inside it, (3)
**the reward must be unreachable name-blind**, or the experiment silently
tests reactive control. We release the implementation, derivation
notebooks, and the protocol.

## 1. Motivation

Two claims are often run together in the debate over whether language models
can acquire physical intelligence. The first — that text carries physical
*knowledge* — is settled: frontier models write working simulators on request.
The second — that a language-native system can convert that knowledge into
*competence* through interaction, and that the words themselves mediate how
the competence generalizes — is not, and cannot be tested by prompting,
because prompting never closes the loop between action and outcome.

This project isolates the second claim at the smallest scale that can carry
it. The base model is the text-only control: it demonstrably "knows" that
objects fall and drift (it will tell you so), yet catches poorly. If outcome
RL closes that gap, the interesting question is *what the gain generalizes
through*. The design principle has two ingredients in deliberate tension:
the binding being **learned** must sit outside the prior (our drift
assignments are counterfactual — no corpus says apples drift left; only
experience teaches it), while the **transfer cue** must sit inside the prior
(else language has no work to do and a from-scratch policy would generalize
identically). Experiment 1 (this paper's data) satisfies the first
ingredient and probes a deliberately limited form of the second: banana's
assigned dynamics are semantically arbitrary, so held-out-banana transfer
distinguishes name-conditional lookup from object-general skill — it cannot,
by construction, demonstrate semantic inheritance of dynamics. Experiment 2
(§4) supplies the missing ingredient with counterbalanced semantic clusters:
dynamics stay arbitrary (so the base model cannot already know them), and
the only route from a trained binding to a held-out word is pretrained
similarity.

**Hypothesis.** Outcome RL on top of a text prior produces a property neither
ingredient produces alone: grounded competence that generalizes along semantic
lines laid down by text. Text supplies the map of meanings; outcomes forge the
skill; the detectable signature is skill that travels the map.

**Positioning.** That language priors aid RL generalization is established
(pretrained word embeddings transferring goal-conditioned policies,
shower→bathtub — Hutsebaut-Buysse, Mets & Latré 2020, arXiv:2007.05196;
entity–dynamics grounding from manuals, Hanjie et al. 2021; "Language
Representations for Generalization in RL", PMLR 157, 2021), as is grounding an LLM through
online RL: **GLAM** (Carta et al. 2023) is the closest ancestor — it also
tested unseen nouns, invented words, and pretrained-vs-random ablations —
and TWOSOME and LLaRP extend the recipe. Experiment 1 below is therefore
best read as a GLAM-style pilot on a modern decoder-only model with a
minimal from-scratch GRPO stack. The contribution this project aims at is
narrower and, to our knowledge, untested: a **counterbalanced causal
demonstration that a newly reward-acquired, arbitrary dynamics binding is
routed to unseen objects specifically through pretrained lexical
semantics** — with instruments (misleading names, nonce floors, first-action
probes at diagnostic states) that quantify the routing rather than
illustrate it.

## 2. Setup

**Environment.** A 7-column grid; a named fruit falls six rows over five
decision turns while the agent moves a basket LEFT/RIGHT/STAY via text.
Observations are plain English ("Turn 2 of 5. The apple is at column 3, row
2. Your basket is at column 5."). Reward is binary at episode end: caught or
not. Per-fruit dynamics: strawberry falls straight; apple drifts left on even
turns; orange drifts right on even turns. **Evaluation-only holdouts:**
banana (real word, orange's dynamics), tangerine (real, semantically near
orange, apple's dynamics — the misleading condition), plum (real, no
drift), and blorple/quorf (nonce, matched/opposed — the generic-skill
floor). A scripted greedy-chase policy establishes a ceiling of ≈0.95.

**Policy and training.** Qwen3-0.6B (instruct), full fine-tune in bf16. Each
turn the model reasons briefly in free text and emits `ACTION: X`;
unparseable output falls back to STAY. Multi-turn GRPO with deliberate
simplifications, each load-bearing:

- **Group baseline, leave-one-out mean.** G=8 rollouts share an identical
  initial state (fruit + seed); advantage = reward − mean of the *sibling*
  rollouts' rewards (a rollout's own reward in its baseline makes the
  baseline action-dependent and rescales the gradient by (G−1)/G). No std
  division (Dr.GRPO): with binary rewards, std-scaling manufactures large
  gradients in near-deterministic groups.
- **Strictly on-policy, one update per batch.** The PPO ratio is identically
  1, so the objective is REINFORCE-with-a-group-baseline; clipping would be a
  no-op. Sampling at temperature 1.0 with all sampler warping disabled, so the
  sampled distribution *is* the distribution whose logprobs the update uses.
- **Exact generated-token masking.** The gradient touches only tokens the
  model sampled; observations and chat-template scaffolding contribute
  nothing. (Formally: environment terms in log P(trajectory) carry no θ
  dependence. Practically: a mask-reconstruction test guards the #1 silent
  bug in multi-turn RL.)
- **KL leash** to the frozen base model (k3 estimator, coefficient 0.02) —
  the mechanism that protects the very prior the transfer hypothesis needs.
- G=8 was chosen empirically: with binary rewards, smaller groups produce
  frequent zero-variance (all-fail/all-succeed) groups that contribute zero
  gradient — early in training these consumed ~40% of batches even at G=8.
  This is the practical face of sparse outcome rewards and a first-class
  logged quantity, alongside within-group reward std, a Monte-Carlo entropy
  estimate (mean −logp of sampled tokens), and pre-clip gradient norm as
  early-warning collapse indicators.

**Evaluation protocol.** Greedy decoding, fixed evaluation seeds shared
across all models, n=100 episodes per fruit. The transfer metric is
**delta-from-base** per fruit — fruits differ in base-model catch rate
(the base policy has a marked rightward action bias), so raw-rate comparisons
conflate prior bias with transfer.

## 3. Results

### 3.1 Run 2 (100 steps, 2 groups × G=8)

Greedy catch rates, n=100 per fruit:

| model | strawberry | apple | orange | **banana (held out)** |
|---|---|---|---|---|
| base | 0.21 | 0.08 | 0.36 | 0.42 |
| ckpt-20 | 0.23 | 0.06 | 0.32 | 0.36 |
| ckpt-40 | 0.26 | 0.15 | 0.34 | 0.38 |
| ckpt-60 | 0.23 | 0.16 | 0.34 | 0.44 |
| ckpt-80 | 0.26 | 0.24 | 0.36 | 0.45 |
| ckpt-100 | **0.37** | **0.34** | **0.51** | **0.50** |
| Δ (final − base) | +0.16 | **+0.26** | +0.15 | **+0.08** |

Three observations:

1. **Outcome RL converts knowledge to competence.** All training fruits
   improve. The cleanest effect is apple (+0.26, >4 standard errors): the
   base policy's rightward bias makes the left-drifting fruit nearly
   uncatchable at baseline (0.08), and RL overcomes a prior bias rather than
   merely amplifying one.
2. **No evidence of task-level erosion.** Held-out banana never falls
   meaningfully below its base rate at any checkpoint (transient −0.06 at
   ckpt-20, ~1σ). Two caveats keep this honest: the logged KL is measured on
   policy trajectories, not a generic-language retention probe; and staying
   near base is the *expected* outcome under a KL leash at 100 steps — the
   erosion question becomes interesting at longer horizons, where the
   checkpoint curve can be overlaid against KL drift.
3. **A held-out improvement that is not yet attributable.** Banana improves
   +0.08 without ever appearing in training — but this undershoots the
   train-fruit deltas (+0.15 to +0.26), so the data is equally consistent
   with three stories: (a) transfer of newly acquired skill through the
   name, (b) a generic competence lift (better observation parsing, action
   formatting, reactive tracking) that raises every fruit, and (c)
   concentration of probability mass on an already-favorable disposition —
   banana has the *highest* base rate (0.42), courtesy of the base policy's
   rightward bias, making the sharpening story most available for exactly
   this fruit. The diagnostics in §4 are designed to separate (a) from (b)
   from (c); until they do, +0.08 is a measurement, not a mechanism.
   Training also ended mid-acceleration — most of orange's gain arrived in
   the final 20 steps — so run 2 is an early snapshot of an unconverged
   policy (which licenses no claim about what longer training yields).

### 3.2 Run 3 (300 steps, fresh from base) and the full battery

Greedy catch rates, n=100 per fruit, across the eight-word battery:

| word | drift | role | base | run-2 ckpt-100 | run-3 ckpt-300 | Δ (300 − base) |
|---|---|---|---|---|---|---|
| strawberry | none | train | 0.21 | 0.37 | 0.98 | +0.77 |
| apple | left | train | 0.08 | 0.34 | 0.86 | +0.78 |
| orange | right | train | 0.36 | 0.51 | 0.90 | +0.54 |
| banana | right | held out, matched | 0.42 | 0.50 | 0.86 | +0.44 |
| tangerine | left | held out, misleading | 0.04 | 0.29 | 0.89 | **+0.85** |
| plum | none | held out, neutral | 0.21 | 0.33 | 0.98 | +0.77 |
| blorple | right | held out, nonce | 0.35 | 0.38 | 0.90 | +0.55 |
| quorf | left | held out, nonce | 0.04 | 0.16 | 0.86 | +0.82 |

The three stories of §3.1 are adjudicated, and story (b) — generic
competence — wins outright:

1. **The nonce floor rose to the ceiling.** Blorple and quorf cannot know
   their assigned dynamics; by construction their catch rates measure what
   name-free skill achieves. At ckpt-0300 that floor is 0.86–0.90 —
   statistically indistinguishable from the training fruits. Held-out mean
   0.90 vs train mean 0.91. Whatever the policy learned, the name
   contributes nothing to it.
2. **The misleading condition failed to mislead.** Tangerine — semantically
   adjacent to orange, carrying opposite dynamics, the one word for which
   name-mediated anticipation predicts a *decrement* — posts the largest
   gain on the board (+0.85). Name-mediated anticipation is absent exactly
   where it would have been most visible.
3. **Turn-1 name-conditioning collapsed rather than sharpened.** At aligned
   states (fruit over basket; the reactive-correct action is STAY), the
   base model leans rightward for *every* name (P(RIGHT)−P(LEFT) between
   +0.11 and +0.42, K=72 per word — a generic directional bias with no
   name structure beyond the ±0.08 noise floor). At ckpt-0300 every lean
   is within noise of zero and STAY probability is 0.83–0.99. Training did
   not teach the policy to use names; it erased the name-adjacent bias the
   base model had.
4. **The chain-of-thought atrophied to nothing while performance climbed.**
   Generated tokens per episode fell 122 → 35 (minimum 20), entropy
   0.29 → 0.17, and late-run transcripts contain literally empty `<think>`
   blocks — bare `ACTION:` emissions — at 0.86–0.98 catch rates. Group
   reward-std stayed healthy throughout (0.24–0.33): this is not collapse
   but *compilation*. KL from base rose to ≈0.2 — the policy paid a real
   KL price, and spent it deleting the reasoning and the base model's
   action biases, not building name-keyed structure.

**Why it happened — the finding we take forward.** The environment's drift
is observable turn-to-turn and correctable within the five-turn horizon: a
scripted policy that merely chases the fruit's *observed* trajectory
already catches ≈0.95. The reward therefore never paid anything for
knowing what a tangerine is. Under a KL leash, the cheapest policy
consistent with the reward is a name-blind reactive controller, and that
is what GRPO found — a small, clean instance of reward-shaped Occam:
optimization routes around the prior whenever the prior is not needed to
earn the reward. The negative result is thus not "language priors don't
carry acquired skill"; it is "this reward never asked them to." Which
yields the third design ingredient, stated in §4.

**Relation to documented failure modes.** Neither half of this outcome is
novel in isolation, and it matters to say so. The CoT atrophy is the
phenomenon RAGEN (Wang et al. 2025) reports across simple multi-turn
environments at comparable model scales — "without fine-grained,
reasoning-aware reward signals, agent reasoning hardly emerges" — and
RAGEN-2 (2026) observes reasoning length declining monotonically across
all eight of its environments, with collapse arriving precisely "when
tasks become solvable via template strategies." Our diagnostics place run
3 on RAGEN-2's *healthy* branch rather than in the pathological one:
their echo-trap/template-collapse signature is reward-variance collapse
with degrading performance, whereas our group reward-std stayed at
0.24–0.33 and performance rose to ceiling — task mastery discarding
computation it no longer needs, consistent with the simplicity bias
toward shorter CoT as accuracy improves (Wu et al. 2025). The
name-blindness, meanwhile, is the lexical analogue of input-channel
shortcut learning documented in multimodal RL — policies that learn to
stop reading an input channel (there, the image; here, the noun) when
reward is attainable from another channel. What Experiment 1 adds is the
conjunction: a transfer experiment whose *instrument* (the held-out-word
battery) detected its own reward's failure to price in the channel under
test — which is exactly what the battery was for.

Two incidental positives: no evidence of task-level erosion anywhere in
the battery (every word ends far above base), and the "is the CoT
load-bearing?" question answered itself for this task — at convergence,
the reasoning was decoration.

## 4. Planned analyses

**Remaining analyses on existing checkpoints:**

- **Paired analysis.** Evaluation seeds are shared across all models, so
  base-vs-checkpoint comparisons should be paired per episode (McNemar)
  rather than marginal — substantially tighter at the same n. (Requires
  per-episode eval logging, a small trainer change.)
- **Base-model pass@k** on the task, separating RL creating competence from
  RL concentrating probability mass on already-reachable behavior. Run 3's
  near-ceiling convergence makes this less load-bearing than it was for
  run 2's marginal deltas, but it still calibrates how much of the skill
  pre-existed in the sampling distribution.
- **Checkpoint erosion curve** over run 3's fifteen checkpoints (evals in
  progress), overlaid with the logged KL-from-base — locating *when* the
  turn-1 biases washed out and whether the CoT atrophy and the KL rise are
  the same event.
- **A mutual-information reasoning diagnostic for Experiment 2's
  dashboard.** RAGEN-2 shows entropy alone cannot distinguish healthy
  convergence from template collapse (reasoning can stay diverse within an
  input while going input-*agnostic*); their in-batch cross-scoring proxy —
  score each reasoning trace's likelihood against every prompt in the
  batch, measure whether traces identify their own inputs — needs only the
  rollout data we already have, and would have flagged run 3's CoT
  deletion far earlier than the token-count trend did.

**Second wave:**

- **Experiment 2 — counterbalanced semantic routing (the headline test).**
  Semantic clusters ({rock, anvil, brick}, {feather, leaf, petal}) with
  cluster→dynamics assignments made **arbitrarily and reversed across
  matched training runs** — in half the runs, "feathers" get the fast
  straight fall. Train on some cluster members; evaluate held-out neighbors
  (anvil, leaf), crossed/misleading names, and nonce names, at aligned
  diagnostic states where different dynamics demand different first actions.
  The counterbalancing is what makes the test causal: if held-out anvil
  inherits whatever arbitrary behavior rock was taught *in that run*, and
  the inheritance **reverses when the assignment reverses**, no
  corpus-physics story (anvils-are-heavy is in the text) and no
  generic-competence story survives — the only remaining route is the
  newly-taught binding traveling through pretrained similarity. Controls:
  base model, and a lexical-scrambling run (shuffled name↔cluster
  pairings). A conflict condition (misleading name, turns-of-evidence to
  override the prior's prediction) quantifies prior-vs-observation
  arbitration. Two requirements for the reversal logic to hold: **several
  training seeds per assignment** (a single pair of runs cannot distinguish
  mapping reversal from run-to-run variance), and **replication across ≥2
  independent semantic cluster pairs** (e.g. heavy/light and fast/slow
  clusters), so the result does not rest on one lexical axis.
  **New requirement from run 3 — the third ingredient:** the environment
  must make the reward unreachable name-blind. Concretely: dynamics must be
  decisive but unobservable-in-time — e.g. the drift lands on the final
  fall, after the last action, so only a policy that committed early (on
  the basis of the name) is under the object when it arrives; or drift
  magnitude that outruns the remaining moves. Acceptance test before any
  training run: scripted reactive ceiling ≤0.5 while the scripted
  name-aware ceiling stays ≥0.9. Run 3 is the demonstration of what
  happens otherwise: if reactive play reaches the reward, RL finds it,
  and the experiment silently stops being about language. A first 400-step
  Exp 2 run then surfaced a **fourth ingredient — rare wins must be able
  to consolidate**. The run converged to a perfect name-blind tracker
  (every straight-landing word ≥0.72, nonce included; every shifted word
  ≤0.11, *the trained one included*), and the training log rules out the
  tempting "exploration desert" reading: the shifted word's sampled reward
  held at 0.11–0.15 throughout — roughly one positively-rewarded episode
  per group, ~400 across the run — and the greedy policy still never
  moved. The wins arrived and failed to bind: each lucky catch is a
  different trajectory (incoherent gradient), the straight cluster
  supplies a dense coherent gradient, the KL leash taxes drift, and
  falling entropy shrinks the exploration budget precisely as the tracker
  sharpens. The remedy that preserves the environment's bite is
  training-side: an entropy bonus (exact differentiable entropy at
  sampled positions; the sampler is untouched, so the on-policy invariant
  holds) to sustain exploration pressure while the name accumulates
  credit.
- **No-prior baseline:** a small policy network on raw state, trained with
  the identical loss — it learns the task easily and cannot transfer by
  construction. (The nonce-name and shuffled-pairing controls formerly
  listed here are now constituent conditions of Experiment 2.)
- **The erosion curve, properly:** held-out performance per checkpoint
  overlaid with KL-from-base on generic text; the prediction from RL's Razor
  is lockstep decay if and when forgetting begins.
- **Consolidation vs creation:** base-model pass@k on the task, to
  distinguish RL creating competence from RL concentrating probability mass
  on already-reachable behavior.
- **Mechanistic follow-ups:** fruit-token representational geometry across
  checkpoints (does banana stay glued to orange while dynamics-relevant
  structure is carved?); cross-fruit activation patching (causal test that
  banana rides orange's machinery); crosscoder diffing of base vs final for
  RL-born features shared by {orange, banana} but not strawberry.
- **A verbalization probe:** ask the trained policy to state each fruit's
  dynamics — it only ever observed positions and rewards — testing whether
  implicitly acquired dynamics became linguistically recoverable. Run it on
  banana too: has training made the model *willing to assert* that bananas
  drift right?
- **Multiple training seeds** — larger eval n reduces environment noise but
  not training-run variance; replication is the only cure for the latter.
- **Engineering for exact resumption** (run directories carrying config,
  optimizer/RNG/global-step state, and per-checkpoint evaluations), so the
  erosion curve and seed replications are cheap to extend.

## 5. Limitations

One seed per run, one model size, one toy environment with five-step
horizons and a three-word action space; the environment's dynamics are
simple enough that "anticipate drift" may be learnable as a shallow feature
rather than anything deserving the word *physics*. The turn-1 probe covers
aligned states only — it cannot rule out name effects expressed solely in
states we did not probe, though the battery's convergence makes such
effects behaviorally inert here. And Experiment 1's negative is a negative
about *this reward structure*, not about language priors: a task solvable
without the name cannot test whether skill travels through the name. The
claim under test remains narrow: not that language models understand the
physical world, but that when interaction teaches a language-shaped policy
a sensorimotor regularity, the language prior is the medium through which
that regularity generalizes to novel descriptions. Experiment 1 established
the instruments and the failure mode; Experiment 2 carries the claim.

A stronger title — *Language Is All You Need to Catch* — sits in a drawer,
reserved for the day P1 clears significance. Titles, like claims, should be
earned.

## References

*(to be completed — currently: Williams 1992; Shao et al. 2024 (GRPO); Liu et
al. 2025 (Dr.GRPO); RAGEN (Wang et al. 2025); GiGPO (Feng et al. 2025);
Shenfeld et al. 2025 (RL's Razor); Yue et al. 2025 (pass@k); GLAM (Carta et
al. 2023); TWOSOME (Tan et al. 2024); LLaRP (Szot et al. 2024);
Hutsebaut-Buysse, Mets & Latré 2020 (embedding-mediated goal transfer);
Hanjie et al. 2021 (entity–dynamics grounding); "Language Representations
for Generalization in RL" (PMLR 157, 2021); RT-2 (Brohan et al. 2023);
Schulman 2020 (KL estimators); RAGEN-2 (arXiv:2604.06268, template
collapse & MI diagnostics); GTR (Wei et al. 2025, arXiv:2503.08525,
thought collapse); Wu et al. 2025 (arXiv:2502.07266, CoT simplicity
bias); arXiv:2606.22043 (reward strength and input-channel shortcuts);
Geirhos et al. 2020 (shortcut learning).)*
