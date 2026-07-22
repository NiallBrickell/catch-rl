# Throw It a Banana: Lexical Transfer of Physically-Acquired Competence in RL-Finetuned Language Models

*Do language priors carry embodied skill to words never trained?*

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
~1σ), tracking orange's +0.15; and held-out performance never degrades across
checkpoints — the language prior survives RL intact, consistent with the
KL-minimal character of on-policy updates. The transfer signal is directionally
positive but not yet significant; training ended while performance was still
accelerating. We release the implementation, derivation notebooks, and a
protocol for the checkpoint-level "erosion curve" and mechanistic follow-ups.

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
through*. A policy could learn "the token `orange` means drift right" (lexical
slot-filling) or something the word merely indexes (semantics). A fruit whose
name never appears in training, with dynamics identical to a trained fruit,
separates the two: slot-filling cannot transfer to a never-seen slot value;
semantic indexing can, because *banana* sits near *orange* in representation
space courtesy of pretraining alone.

**Hypothesis.** Outcome RL on top of a text prior produces a property neither
ingredient produces alone: grounded competence that generalizes along semantic
lines laid down by text. Text supplies the map of meanings; outcomes forge the
skill; the detectable signature is skill that travels the map.

## 2. Setup

**Environment.** A 7-column grid; a named fruit falls six rows over five
decision turns while the agent moves a basket LEFT/RIGHT/STAY via text.
Observations are plain English ("Turn 2 of 5. The apple is at column 3, row
2. Your basket is at column 5."). Reward is binary at episode end: caught or
not. Per-fruit dynamics: strawberry falls straight; apple drifts left on even
turns; orange drifts right on even turns. **Banana (drift right, identical to
orange) is evaluation-only.** A scripted greedy-chase policy establishes a
ceiling of ≈0.95.

**Policy and training.** Qwen3-0.6B (instruct), full fine-tune in bf16. Each
turn the model reasons briefly in free text and emits `ACTION: X`;
unparseable output falls back to STAY. Multi-turn GRPO with deliberate
simplifications, each load-bearing:

- **Group baseline, mean-only.** G=8 rollouts share an identical initial
  state (fruit + seed); advantage = reward − group mean. No std division
  (Dr.GRPO): with binary rewards, std-scaling manufactures large gradients in
  near-deterministic groups.
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
2. **No erosion.** Held-out banana never falls meaningfully below its base
   rate at any checkpoint (transient −0.06 at ckpt-20, ~1σ). One hundred
   steps of outcome RL did not strip-mine the prior — consistent with the
   small measured KLs throughout and with the view that on-policy RL finds
   KL-minimal solutions.
3. **A transfer signal, not yet a claim.** Banana improves +0.08 without ever
   appearing in training, directionally tracking orange's +0.15. At n=100
   this is ~1σ: consistent with P1, insufficient to establish it. Notably,
   training ended mid-acceleration — most of orange's gain arrived in the
   final 20 steps — so run 2 bounds the effect from below.

## 4. Planned analyses

- **Longer training** (run 3, 300 steps, in progress) with the collapse
  dashboard active; larger eval n for the banana–orange delta comparison.
- **P2 controls:** nonsense fruit names ("blorple") and shuffled
  name↔dynamics pairings, separating semantic transfer from slot-filling;
  a no-language policy (small network on raw state, trained with the
  identical loss) as the no-prior baseline — it learns the task easily and
  cannot transfer by construction.
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
  implicitly acquired dynamics became linguistically recoverable.

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
al. 2023); RT-2 (Brohan et al. 2023); Schulman 2020 (KL estimators).)*
