# catch-rl

Outcome-based RL on an LLM's chain-of-thought, from scratch, in plain PyTorch.
A tiny falling-fruit environment, a small Qwen model, and a hand-rolled GRPO
loop — no TRL, no verifiers, no veRL. The only libraries are `torch` (math)
and `transformers` (loading pretrained weights; that part isn't the lesson).

The question being poked at: if you train a language model *through language*
to act in a (toy) physical world, does the language prior carry? Concretely —
train it to catch strawberries, apples, and oranges, then throw it a **banana**
(a word it never saw during RL) and see if it catches that too.

📄 **[Working paper draft](paper/draft.md)** — hypothesis, method, current
results, and planned analyses in one place.

## The hypothesis

**Text alone yields knowledge without competence — the base model, evaluated
before any RL, is the control for exactly that. The claim: outcome RL against
a (toy) physical world, on top of a text prior, produces a property neither
ingredient produces alone: grounded competence that generalizes along semantic
lines laid down by text. Text supplies the map of meanings; real outcomes
forge the skill; the emergent thing to detect is skill that travels the map.**

Note what is *not* being tested: that language carries physical knowledge
(settled — frontier models write you a working simulator today, but that is
knowing-by-testimony, not competence-by-experience), and that transformers can
emit actions (settled — VLAs are transformers). The contested part is the
integration: the same weights ingesting both testimony (pretraining) and
experience (outcome RL), with the words acting as the index that lets
experience generalize.

Predictions, in decreasing order of comfort:

- **P1 — slot transfer.** Post-RL, banana's improvement over its own base
  rate tracks orange's (Δbanana ≈ Δorange > 0), despite "banana" never
  appearing in an RL episode. (Delta-from-base, not raw rates: the fruits
  start from different base catch rates.) Honest scope note: banana's
  assigned dynamics are *semantically arbitrary* — nothing in the meaning of
  "banana" predicts drifts-right — so P1 tests whether the learned policy is
  name-conditional lookup or object-general skill, **not** semantic
  inheritance of dynamics. The semantic-inheritance test is Experiment 2
  (below), where meaning actually predicts physics.
- **P2 — the prior is load-bearing.** Replace fruit names with nonsense tokens
  ("blorple") or shuffle name↔dynamics pairings and the transfer should
  weaken or vanish — separating semantic generalization from slot-filling.
- **P3 — fragility.** If RL strip-mines the prior (entropy/reasoning
  collapse), banana transfer *decays* across checkpoints even while training
  fruits improve. Eval banana at every saved checkpoint: this erosion curve
  makes the language-vs-embodiment debate a measurable quantity, and nobody
  has published it.

**Falsification:** if orange → ~1.0 while banana ≈ base model, the policy
learned lexical slot-filling, not semantics — a clean point for the
grounding-first side of the debate. Either outcome is a result.

(Post-training probe, free to run: ask the trained model to *state the rules*
— "describe how each fruit falls." It only ever saw positions and rewards; if
it can verbalize dynamics it learned through outcomes, implicit competence
became explicit, linguistically recoverable knowledge.)

---

## The whole idea in one loop

```
for each training step:
    sample G rollouts of the SAME episode        # same fruit, same positions
        each turn: model reads text state → thinks → "ACTION: LEFT"
        environment moves fruit, returns new text state
        at the end: reward = 1 caught, 0 missed
    advantage_i = reward_i − mean(rewards)        # "was this rollout better
                                                  #  than my usual self?"
    loss = −Σ advantage_i · logprob(tokens the model generated)
    backprop, small optimizer step
```

That's genuinely all of it. Everything below is the *why* behind each line.

## Why this is REINFORCE wearing a GRPO badge

The fundamental policy-gradient idea (REINFORCE, 1992): make the tokens that
led to good outcomes more likely, tokens that led to bad outcomes less likely.
`∇J = E[ R · ∇log π(trajectory) ]`. The problem is variance: if every rollout
gets reward 1, you'd push up on everything, learning nothing about *relative*
quality. So you subtract a **baseline** — "how well do I usually do here?" —
and only the *difference* drives learning.

Classic PPO learns that baseline with a second network (a value model/critic).
**GRPO's whole trick is to skip the critic**: sample the same episode G times
and use the group's mean reward as the baseline. Caught it in 4 of 6 tries →
the 4 successes get advantage +⅓, the 2 failures −⅔. No value network, no
extra memory — though the group mean is a *noisy estimate*, not an oracle,
and one subtlety matters: a rollout's own reward must be left out of its
baseline (we average the *siblings*), because including it makes the
baseline action-dependent and silently rescales the gradient by (G−1)/G. This
is why the G rollouts in a group must share the *identical* initial state —
otherwise the mean mixes "easy episode" with "bad policy" and the baseline is
polluted.

Two footnotes worth internalising:

- **We don't divide by the std** (the original GRPO does; the Dr.GRPO paper
  showed it biases learning toward near-deterministic groups — dividing a
  tiny advantage by a tiny std manufactures a big gradient out of noise).
  Mean-subtraction only.
- **PPO's famous clipped ratio is a no-op here.** Clipping only matters when
  you take multiple gradient steps on stale rollouts (the ratio π_new/π_old
  drifts from 1). We do one update per batch of fresh rollouts, so the ratio
  is identically 1 and this is exactly REINFORCE-with-a-group-baseline. Most
  of the notational fog around GRPO evaporates once you see this.

## The multi-turn part (where the real subtleties live)

**Token masking.** An episode's transcript interleaves text the *model* wrote
(its reasoning + ACTION lines) with text the *environment* wrote (observations,
chat-template scaffolding). The gradient must only flow through tokens the
model actually sampled — you can't credit or blame the policy for the fruit's
position being announced. So the rollout keeps a boolean mask alongside the
token ids, and the loss touches masked-True positions only. This is the #1
source of silent bugs in hand-rolled agent RL, which is why `--check-mask`
exists: it decodes the masked positions and verifies they reconstruct exactly
what the model generated.

**Generation builds no graph.** `.generate()` runs under no-grad — you can't
backprop through it. So after rollouts, we re-forward the full episode token
sequence *with* grad to recover the logprobs at the model's own tokens. Rollout
and learning are two separate passes over the same tokens. (This also means
sampling and learning could happen on different machines — which is exactly
how the big distributed RL stacks work; ours just does both in one process.)

**Credit assignment is deliberately crude.** One scalar reward at the end gets
smeared uniformly across every token the model generated over all 5 turns. The
brilliant early move and the turn-4 blunder get the same credit. This is the
central crudeness of outcome-based RL — the group baseline reduces variance
*across* rollouts but does nothing *within* a trajectory. The literature's
answers (turn-level advantages à la GiGPO, learned critics, reward shaping)
all trade simplicity for resolution. We start crude on purpose: watching what
crude credit assignment can and can't learn is the education.

**The KL leash.** A small penalty keeps the policy near a frozen copy of the
starting model. Without it, a sparse-reward policy gradient happily strip-mines
the language prior: the chain-of-thought degenerates into repeated filler that
happens to correlate with reward ("reasoning collapse" — RAGEN documented this
failure mode), entropy collapses, and the very prior you're trying to leverage
— the thing that would let *banana* inherit from *orange* — gets destroyed.
The KL term is literally "stay a language model while you learn the task."

## Things to watch in the logs

Ordered roughly by how early each one warns (per RAGEN's collapse anatomy —
the mean reward is the *last* thing to move):

- **`rstd`** (within-group reward std) — the earliest indicator: the spread
  of outcomes dies before the mean does. Zero `rstd` at ceiling reward is
  victory; zero `rstd` below ceiling is premature convergence — a corpse
  that still moves.
- **`ent`** (Monte-Carlo entropy: mean −logp of the sampled tokens, free
  since the loss already computes those logprobs) — falling is normal
  (learning *is* spending entropy); falling fast while reward stays flat
  means the exploration budget is buying nothing.
- **`gnorm`** (pre-clip gradient norm) — spikes arrive late: ∇log π = ∇π/π
  divides by the sampled token's probability, so once the distribution is
  nearly deterministic, rare tokens detonate the batch gradient. By then
  recovery usually means reloading a checkpoint.
- **`frac_zero_var_groups`** — with a binary reward, a group that's all-0 or
  all-1 has zero advantage everywhere: *no learning signal at all*. Early on
  (policy nearly always misses) this is the practical failure mode of sparse
  outcome rewards. If it sits near 1.0, nothing is being learned; remedies in
  rough order of principle-preservation: bigger G, easier curriculum (slower
  fruit), then shaped rewards (partial credit for proximity) as a last resort
  — shaping works but dilutes the "pure outcome" premise.
- **Mean generated tokens per turn** — if it trends toward the minimum, the
  CoT is atrophying toward bare ACTION lines. Interesting either way: is the
  reasoning load-bearing or decorative? You can test this directly (see below).
- **KL** — a slow rise is learning; a spike is the policy bolting.
- **Parser hacking** — the model only needs to end with a parseable line;
  degenerate outputs that game the parser are miniature reward hacking. Sample
  transcripts are printed every 10 steps: read them.

## The experiment

| fruit      | drift               | seen in RL? |
|------------|---------------------|-------------|
| strawberry | falls straight      | yes         |
| apple      | drifts left         | yes         |
| orange     | drifts right        | yes         |
| **banana** | drifts right        | **never**   |

Banana shares orange's dynamics but the *word* never appeared in training.
If post-RL catch-rate(banana) ≈ catch-rate(orange) ≫ base model, the policy
learned something like "track the object and anticipate drift" in a way that
rides on the language prior, rather than memorising "the token `straw`+`berry`
means press STAY." That's the language-vs-embodiment argument in miniature,
run as an experiment. A stricter follow-up: invent a *nonsense* fruit ("a blorple is at
column 3…") — banana tests transfer via real-world semantics; blorple tests
whether the behaviour generalises over the object slot itself.

### Experiment 2 (planned): counterbalanced semantic routing

The catch env's drift assignments are deliberately counterfactual — that's
what makes the *learning* real (no corpus says apples drift left; only
experience teaches it). But it means held-out-word transfer can't run
through meaning: banana's dynamics are arbitrary, so a transfer result shows
slot-generalization, not semantic inheritance. Experiment 2 closes the loop
with **semantic clusters under counterbalanced arbitrary dynamics**: take
{rock, anvil, brick} and {feather, leaf, petal}, assign each cluster's
dynamics *arbitrarily*, and **reverse the assignment in matched runs** — in
half the runs, "feathers" fall fast and straight. Train on some members
(rock, feather); evaluate held-out neighbors (anvil, leaf), crossed
misleading names, and nonce names, at aligned states where different
dynamics demand different first actions. The counterbalancing is the causal
teeth: if anvil inherits whatever rock was taught *in that run*, and the
inheritance flips when the assignment flips, neither corpus physics
("anvils are heavy" is in the text) nor generic competence can explain it —
only the newly-taught binding traveling through pretrained similarity.
Two requirements for the reversal logic to hold: several seeds per
assignment (one pair of runs can't distinguish reversal from run variance)
and replication across ≥2 independent cluster pairs, so nothing rests on a
single lexical axis. Design principle throughout: the *binding being
learned* sits outside the prior; the *transfer cue* sits inside it. Both
ingredients, or the experiment tests nothing.

Other cheap experiments once the loop works:
- **Is the CoT load-bearing?** Retrain (or just eval) with thinking forbidden —
  format is `ACTION: X` only. If catch rates match, the reasoning was
  decoration; if they drop, the tokens were computation.
- Vary drift magnitude to make anticipation (not just tracking) necessary.
- `--kl 0` and watch the collapse happen for real. Educational carnage.

## Phase 2: the MRI plan (internals, not just outcomes)

The behavioral predictions above all have internal correlates, and the tooling
for our exact model exists (Anthropic's circuit-tracer + BluelightAI's
cross-layer transcoders for Qwen3-0.6B; Delta-Crosscoder for diffing narrow
fine-tunes). Ranked by evidence-per-effort:

0. **Sanity first**: base-model pass@k on catch (does the prior already
   contain the skill, occasionally? RL may consolidate rather than create —
   measure before claiming emergence), plus per-checkpoint KL from base.
   RL's-Razor prediction: banana transfer decays in lockstep with KL drift.
1. **Fruit-token geometry** — residual streams for all four fruits in
   identical board states, every layer, every checkpoint: does banana stay
   glued to orange while dynamics structure gets carved? Forward hooks only.
2. **Cross-fruit activation patching** (causal): patch orange-context
   activations into banana episodes (strawberry as control). A near-no-op
   proves banana rides orange's machinery.
3. **Linear probes** for world-state variables (fruit col, drift direction,
   signed offset) per layer, base vs trained — did RL build a world-state
   readout that text alone lacks?
4. **Crosscoder diff** base-vs-final: hunt for RL-born features firing for
   {orange, banana} but not strawberry. The associative smoking gun; needs
   the studio.
5. Garnish: logit lens at the action position across checkpoints; weight-space
   diffing (which matrices moved, when).

No published precedent exists for this package (mechanistic diff of a
multi-turn GRPO policy + held-out novel-word transfer probe) as of July 2026.

## Notebooks

Three interactive marimo notebooks accompany the code
(`uv run marimo edit <file>`), forming a ladder — math → core ML → RL:

- **`math_refresher.py`** — the math underneath everything: derivatives →
  the log-derivative identity → gradients → expectation → variance →
  distributions → KL → a guided re-derivation of the score-function trick.
  Every concept in three aligned representations (scrubbable picture,
  annotated notation, executable numpy), with exercises.
- **`ml_primer.py`** — a brisk core-ML pass: networks as bent linear maps →
  backprop as chain rule with bookkeeping (manual numpy matched against
  autograd to machine precision) → softmax/cross-entropy → attention → a
  language model as a next-token classifier, mapped line-by-line onto this
  repo's `batched_logprobs`.
- **`grpo_notes.py`** — the RL, derived: REINFORCE → why baselines →
  GRPO-as-critic-replaced-by-sampling → token masking → where PPO's clip
  went → the KL leash. Capstone: a small numpy policy trained with exactly
  this repo's loss, learning the catch game live, with baseline and
  group-size toggles that demonstrate why each piece exists.

## Lab notes

**2026-07-22 — run 2 (100 steps, 2 groups × G=8, ~45 s/step, M4 Max).**
First completed run (run 1 died at step 40 to an MPS allocator pathology —
variable sequence lengths defeating the caching allocator; fixed by
bucketing padded lengths and flushing the cache per step, see commit
history). Greedy evals, n=100/fruit, base → ckpt-0100: strawberry
0.21 → 0.37, apple 0.08 → 0.34, orange 0.36 → 0.51, **banana (held out)
0.42 → 0.50**. Reading: outcome RL works on all training fruits (apple's
+0.26 against the base model's rightward bias is the cleanest effect,
>4σ); **no evidence of task-level erosion** — banana never fell
meaningfully below base across five checkpoints (transient ckpt-0020 dip,
~1σ; note the logged KL is measured on policy trajectories, not a
generic-text retention probe, and staying near base is *expected* at this
KL coefficient and step count); **held-out improvement positive but
non-specific** — Δbanana +0.08 (~1σ) sits below the train-fruit deltas
(+0.15 to +0.26), i.e. consistent with a general competence lift rather
than anything orange-specific; the name-conditioning diagnostic and the
tangerine/blorple battery are what can separate those stories. No
collapse signatures: KL settled near 0.005, CoT length stable ~90–100
tok/ep, dead-group fraction halved in the final 20 steps. The run ended
while still accelerating (most of orange's gain arrived in the last 20
steps) — undertrained. Next: 300 steps with the rstd/ent/gnorm dashboard.

**2026-07-22 — verification (M4 Max studio).** Mask check exact (295/678
tokens generated in test episode). Smoke: 8.7–18.3 s/step at 4 eps/step;
step 3 hit an all-fail group (`zero-var 1.00`, zero gradient) — dead groups
are real, hence G=8 for run 1. Base-model greedy control (n=30/fruit):
strawberry 0.20, apple 0.03, orange 0.43, banana 0.57; scripted ceiling
≈0.95–0.98. Reading: big headroom; marked directional bias in the base
policy (apple, the left-drifter, ≈0 — rightward prior?); the banana-vs-orange
gap is ~1σ at n=30, i.e. noise until proven otherwise. P1 is a delta-from-base
measurement, and real evals need n ≥ 100.

## Getting started

Requirements: [uv](https://docs.astral.sh/uv/) and a machine with ≥24GB of
RAM — Apple Silicon (MPS) is the tested path; CPU works but is slow. The
first training/eval run downloads Qwen3-0.6B (~1.5GB) from HuggingFace.

```bash
git clone https://github.com/NiallBrickell/catch-rl && cd catch-rl
uv sync                               # installs Python 3.12 + torch + transformers + marimo
```

**Notebooks** (each opens in the browser; no GPU needed):

```bash
uv run marimo edit math_refresher.py  # the math: derivatives → score-function trick
uv run marimo edit ml_primer.py       # core ML: backprop → attention → LM-as-classifier
uv run marimo edit grpo_notes.py      # the RL: REINFORCE → GRPO, capstone learns catch live
```

**Sanity checks** (run these before believing anything else):

```bash
uv run python catch_env.py            # env self-test: scripted-policy catch rates (~0.95)
uv run python train.py --check-mask   # verify the gradient mask reconstructs generated text
```

**Train and evaluate:**

```bash
uv run python train.py --steps 100 --group-size 8   # ~1-1.5h on an M4-class GPU
uv run python train.py --eval --eval-n 100          # base model catch-rate table (the control)
uv run python train.py --eval runs/ckpt-0100 --eval-n 100   # a checkpoint, incl. held-out banana
```

Training prints one line per step (reward, `rstd`, `ent`, `gnorm`, KL,
dead-group fraction, tokens/episode — see *Things to watch in the logs*) and
appends JSON to `runs/log.jsonl`; checkpoints land in `runs/ckpt-NNNN` every
20 steps, one per point on the erosion curve.

Hardware notes: Qwen3-0.6B full-finetunes comfortably in 24GB on MPS (weights
+ grads + Adam moments + a frozen reference copy ≈ 8–9GB before activations;
logits are the spike, hence micro-batched re-forwards). The M4 Max studio is
~2× on memory bandwidth ≈ ~2× on decode speed — worth using for longer runs or
for stepping up to Qwen3-1.7B, at which point LoRA (a ~40-line from-scratch
addition, and with adapters disabled the base model doubles as the KL
reference for free) is the natural next lesson.
