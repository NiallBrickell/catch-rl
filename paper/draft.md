# Throw It a Banana: Lexical Transfer of Physically-Acquired Competence in RL-Finetuned Language Models

*Do word-trained priors carry skills learned by acting?*

**Status: working draft — results are preliminary (see §5). Numbers current as
of run 2 (100 GRPO steps); run 3 (300 steps) in progress.**

## Abstract

A language model that has never caught anything knows a great deal about
falling fruit. We ask whether that knowledge becomes *competence* when a small
LLM is finetuned with outcome-based reinforcement learning in a toy physical
environment — and whether competence acquired through interaction generalizes
along semantic lines laid down by text alone. We finetune Qwen3-0.6B with
hand-rolled multi-turn GRPO (~400 lines of PyTorch) on a text-interfaced
catching game: three training fruits with distinct drift dynamics, plus a
held-out evaluation fruit — **banana** — whose name never appears in a
training episode but whose dynamics match orange. After 100 steps, every
training fruit improves (apple +0.26 absolute, against an adverse prior
bias); the held-out banana improves +0.08 over its own base rate (n=100,
~1σ) — directionally positive, but not yet separable from an across-the-board
competence gain (train-fruit deltas ran +0.15 to +0.26); and held-out
performance shows no evidence of task-level degradation across checkpoints,
as expected under a KL leash at this step count. The controls that would
make the transfer claim identifiable — nonce-word holdouts, a
misleading-semantics condition, and a turn-1 name-conditioning diagnostic —
are specified and running on existing checkpoints. We release the
implementation, derivation notebooks, and the protocol.

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

## 3. Results (run 2: 100 steps, 2 groups × G=8)

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

## 4. Planned analyses

**Immediate diagnostics (running on existing checkpoints, ahead of longer
training — these determine what the numbers in §3 mean):**

- **Turn-1 name-conditioning.** On turn 1 the fruit has not moved; drift is
  unobservable, so any action-distribution difference across names at
  identical positions is name-conditioned anticipation. Preemptive rightward
  lean for orange (and possibly banana) and leftward lean for apple is
  direct evidence the policy uses the word; the nonce names should stay
  neutral (they calibrate the diagnostic's noise floor). Identical
  distributions across names would show no detectable conditioning *in
  these states* — reframing a null banana result as expected rather than
  ambiguous, though it cannot rule out name effects expressed only in
  states we did not probe.
- **The holdout battery** (evaluated on existing checkpoints): banana (real,
  matched), plum (real, neutral), blorple/quorf (nonce, matched/opposed),
  and **tangerine** — semantically adjacent to orange but carrying apple's
  dynamics. A *decrement* on tangerine relative to base is the single most
  diagnostic outcome available: neither generic competence nor
  concentration-on-priors predicts it; only name-mediated anticipation does.
  Interpretive note on the nonce words: blorple and quorf *cannot know their
  secret dynamics* — their catch rates measure the generic-skill floor (and
  their first-action distributions should be name-neutral, calibrating the
  diagnostic's noise level), not failed transfer.
- **Paired analysis.** Evaluation seeds are shared across all models, so
  base-vs-checkpoint comparisons should be paired per episode (McNemar)
  rather than marginal — substantially tighter at the same n.
- **Base-model pass@k** on the task, separating RL creating competence from
  RL concentrating probability mass on already-reachable behavior — the
  story most available for banana specifically (§3.3c).

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
- **Longer training** (run 3, 300 steps, in progress) with the collapse
  dashboard active; larger eval n for the banana–orange delta comparison;
  a first-action analysis on banana episodes (leading rightward before any
  drift has been observed would indicate name-based borrowing rather than
  in-episode evidence).
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

One seed, one model size, one toy environment with five-step horizons and a
three-word action space; the headline transfer delta is within noise at
current n; the environment's dynamics are simple enough that "anticipate
drift" may be learnable as a shallow feature rather than anything deserving
the word *physics*. The claim under test is correspondingly narrow: not that
language models understand the physical world, but that when interaction
teaches a language-shaped policy a sensorimotor regularity, the language
prior is the medium through which that regularity generalizes to novel
descriptions. The experiment is designed so that either answer — transfer or
slot-filling — is a result.

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
Schulman 2020 (KL estimators).)*
