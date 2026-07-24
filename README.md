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
- **`mi_retrieval_acc` / `mi_zscore`** (every 10 steps) — RAGEN-2's insight:
  entropy can't tell healthy convergence from *input-agnostic* reasoning,
  because a policy can keep its traces diverse while they stop depending on
  the input. So we cross-score: each turn-1 completion's logprob against
  every sampled turn-1 prompt in the batch. If a trace can't identify its
  own input (retrieval accuracy falling toward chance, z-score toward 0),
  the "reasoning" has become a template — a signal token counts only catch
  later, once the template also shrinks.
- **KL** — a slow rise is learning; a spike is the policy bolting.
- **Parser hacking** — the model only needs to end with a parseable line;
  degenerate outputs that game the parser are miniature reward hacking. Sample
  transcripts are printed every 10 steps: read them.

## The experiment

| word          | drift    | trained? | role in the battery                          |
|---------------|----------|----------|----------------------------------------------|
| strawberry    | none     | yes      | train                                        |
| apple         | left     | yes      | train (fights the base model's right bias)   |
| orange        | right    | yes      | train                                        |
| **banana**    | right    | never    | real word, matched dynamics                  |
| **tangerine** | left     | never    | real word *near orange*, misleading dynamics |
| **plum**      | none     | never    | real word, neutral                           |
| **blorple**   | right    | never    | nonce — generic-skill floor                  |
| **quorf**     | left     | never    | nonce — generic-skill floor                  |

No single holdout is diagnostic on its own — a banana improvement is equally
consistent with transfer, with a generic play-the-game-better lift, or with
sharpening a favorable prior bias. The battery separates the stories: the
nonce words floor what name-free skill achieves; tangerine predicts a
*decrement* under name-mediated anticipation (semantically adjacent to
orange, drifting the other way) that no generic story can produce; and the
turn-1 diagnostic (`analysis/diag_name_conditioning.py`) probes aligned
states where the fruit hasn't moved yet — a reactive policy says STAY there,
so any name-keyed lean is the prior acting before evidence exists.
(Verdict from run 3: the battery came back **name-blind** — everything
converged to the same band, the turn-1 leans collapsed to zero, and the
reason turned out to be a design lesson. See Lab notes.)

Prior-art honesty: grounding LLMs through online RL and testing unseen and
invented nouns dates to GLAM (Carta et al., 2023) — this experiment is best
read as a pilot on a modern decoder + GRPO stack whose instruments feed
Experiment 2, where the aimed contribution lives (see the paper draft's
positioning section).

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

**Third ingredient, learned the hard way from run 3: the reward must be
unreachable name-blind.** The catch env's drift is observable turn-to-turn
and correctable within the horizon, so a purely reactive policy hits ≈0.95
— and that is precisely the policy RL found, washing out even the base
model's name-linked biases along the way (see Lab notes). Experiment 2's
environment must therefore make anticipation *decisive*: e.g. drift that
lands on the final fall (after the last action, so the only way to be under
the fruit is to have committed early) or drift magnitude that outruns the
remaining moves. Acceptance test before any training run: the scripted
reactive ceiling must sit well below the scripted name-aware ceiling
(target: ≤0.5 vs ≥0.9). If reactive play can reach the reward, the prior is
dead weight and the experiment tests nothing.

**Fourth ingredient, learned from run 4 (and then corrected): rare wins
must be able to consolidate.** A reward gap is necessary but not
sufficient. The first reading of run 4 was an "exploration desert" — the
behavior between "stand at c" and "stand at c+2" earns zero, so the far
basin is never even sampled. The training log falsifies the strong
version: butterfly's *sampled* reward sat at 0.11–0.15 all run (≈ one
caught-butterfly episode per group, ~400 positively-rewarded examples),
yet the greedy policy never moved. The wins arrived; they didn't
*consolidate*. Each lucky catch is a different wobble path (incoherent
gradient), the straight cluster supplies a dense coherent gradient
pulling the other way, the KL leash taxes any drift, and the policy's
falling entropy shrinks the wobble budget over time. So the fix is a
trainer knob, not an env concession: an **entropy bonus** (`--ent-bonus`,
exact differentiable entropy on sampled positions — the sampler stays
untouched, so the sampled-vs-learned invariant holds) keeps exploration
pressure alive long enough for the name to soak up the credit. The env
keeps shift 2; `--exp2-shift 1` exists as a later ablation (same 0.5
reactive box, denser accidental wins) to separate signal-density effects
from exploration-pressure effects if needed.

**Reading the counterbalance (A = congruent assignment: light cluster
drifts; B = incongruent: rock cluster drifts).** Given run 6's probe shows
the A-split, assignment B's outcome decides the story:

| B outcome | reading |
|---|---|
| rock-cluster learns its shift, stone/granite inherit | **Semantic routing of an arbitrary binding** — travels the lexicon regardless of congruence (the headline claim) |
| trained rock learns, siblings don't inherit | Congruence gates *transfer*, not learning — the prior mediates generalization only |
| rock-cluster barely learns the shift at all | The prior gates what reward can teach — RL-finetuning is congruence-limited (a different, equally publishable claim) |
| feather-cluster drifts anyway despite B's training | The prior dominates reward — "transfer" was prior-unlocking all along |

**Status: the v2 environment exists and passes.** `catch2_env.py`
implements the final-fall shift (object drops straight through every
observed row, then lands `+2` columns keyed to its name-cluster, after the
last action — zero in-episode evidence, magnitude 2 so hedging one column
off catches nothing). Acceptance test (`uv run python catch2_env.py`):
scripted reactive 0.500 — it catches the straight cluster at 1.000 and the
shifted cluster at 0.000 — scripted name-aware oracle 1.000. The
cluster→shift assignment is a constructor argument (`make_shift_map`), so
a counterbalanced run pair is two configs, not two envs. The trainer is
wired: `uv run python train.py --exp2-pair 1 --exp2-assign A` trains on
one validated member per cluster (rock + butterfly for pair 1; see
`analysis/cluster-validation.md` for how the word lists were chosen and
what got rejected), holding out the four sibling words plus two fresh
nonce words (cromlet, torgim). Next: a pilot run to size G against the
sparser early reward (a reactive policy earns ~0.5 here, so
shifted-cluster groups will be all-fail more often than v1's).

Other cheap experiments once the loop works:
- ~~**Is the CoT load-bearing?**~~ Answered incidentally by run 3: ckpt-0300
  catches at 0.86–0.98 with literally empty `<think>` blocks. For this task,
  at convergence, the reasoning was decoration.
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

**2026-07-24 — run 6 (Exp 2, assignment A, shift 2, --ent-bonus 0.02) —
first name-conditioned behavior, mid-run probe.** The entropy bonus
changed the dynamics qualitatively: entropy_exact *rises* (0.29 → 0.38 by
step 120), CoT length rises (74 → 159 tok/ep — the model talks again),
zero-var groups at record lows, and butterfly's sampled reward climbs in
lockstep with rock instead of flatlining. A reboot killed the run at step
~124 (see the one-model-at-a-time convention — that outage was
self-inflicted by a concurrent eval); greedy probe of the surviving
ckpt-0120 (n=50, shared seeds → paired): **butterfly 0.32, held-out
feather 0.32, moth 0.30 — vs shifted nonce torgim 0.10**, all four
identical (0.19) at base. Same dynamics, same spawns; the ~4σ (pooled)
split is carried entirely by what the word *is*, and it extends to
cluster members never seen in training. First genuine name-conditioning
of the project, and it appeared at the trained word's *siblings*.
Essential caveat: assignment A is the semantically congruent one —
"light things drift" is corpus physics — so this may be reward unlocking
a congruent prior rather than an arbitrary binding traveling the lexicon.
Assignment B (rocks shift, feathers fall true) is queued as the decisive
control; the four-way outcome table in the Experiment 2 section says what
each result would mean. Straight side of the probe: rock 0.42 > stone
0.30 > granite 0.20 = cromlet 0.20 — possibly inheritance ordered by
embedding proximity to the trained word (rock~stone 0.50, rock~granite
0.32); n=50 is too small to lean on, the final n=100 eval will say.

**2026-07-24 — run 4 (Exp 2, pair 1, assignment A, shift 2, 400 steps,
2×G=8).** The box held all the way to convergence pressure — and revealed
its own flaw. Battery at ckpt-0400 (greedy, n=100): rock 0.89, stone 0.72,
granite 0.72, cromlet 0.72 vs butterfly 0.11, feather 0.11, moth 0.10,
torgim 0.06. The policy became an excellent name-blind tracker (nonce =
held-out real words at 0.72, exactly the signature) and never once found
the +2 behavior — *including on the trained shifted word*, 400 steps of
reward available and untouched. Training curve confirms: rock's sampled
reward 0.23 → 0.69 while butterfly pinned at 0.11–0.15 (one transient
flicker to ~0.25 near step 110–130 that died). One intriguing wrinkle:
rock (0.89) beat its held-out siblings (0.72, ~3σ) — a possible trained-
word-specific effect worth revisiting once any binding exists to compare
against. Diagnosis (corrected same day): NOT a pure exploration desert —
butterfly's sampled reward was 0.11–0.15 throughout, so ~400 positive
examples arrived and failed to consolidate against the straight cluster's
coherent gradient + the KL tax + falling entropy. See the fourth
ingredient above. Next: run 6 — shift stays 2, add `--ent-bonus 0.02`
(watch `entropy_exact`: the bonus should hold it up; if it keeps falling
by ~step 100, the coefficient is too small).

**2026-07-23 — Experiment 2 pilot (pair 1, assignment A: rock-cluster lands
true, butterfly-cluster shifts +2; 100 steps, 2×G=8, M4 Max).** Purpose was
operational — does G=8 get gradient through the sparser reward? Yes:
zero-variance fraction bounced 0.2–0.6, never pinned. Battery (greedy,
n=100, base → ckpt-0100): every straight-landing word improved — rock
+0.18, held-out stone +0.24, granite +0.09, and nonce cromlet +0.12 — and
every shifted word stayed on the floor, *including trained butterfly*
(+0.03; feather −0.07, moth +0.04, torgim −0.05). Clean reading: 100 steps
bought better name-blind tracking (catches whatever lands where it fell,
nonce included) and no name→shift binding yet, even for the trained word.
The env's box is holding — reactive play is worth ~0.5 and nothing more.
Also notable: under the v2 prompt the policy is terse from step 1 (~27
tok/ep, near-bare ACTION lines), so any name-use that develops will be
silent in the weights — the turn-1 probe and battery carry the detection
load, as designed. Next: run 4, same config, 400 steps (v1's transition
started ~step 160; 100 steps was never the real attempt).

**2026-07-23 — run 3 (300 steps, fresh from base, M4 Max) + full battery +
turn-1 diagnostic. The three-story question is adjudicated: story (b) wins.**
Greedy evals, n=100/fruit, base → ckpt-0300: strawberry 0.21 → 0.98, apple
0.08 → 0.86, orange 0.36 → 0.90, banana 0.42 → 0.86, tangerine 0.04 → 0.89,
plum 0.21 → 0.98, **blorple 0.35 → 0.90, quorf 0.04 → 0.86**. Read that
battery carefully: the nonce words — which *cannot* know their dynamics —
converge to the same band as the training fruits, and tangerine (the
misleading condition, predicted to *decrease* under name-mediated
anticipation) posts the largest gain on the board (+0.85). Held-out mean
0.90 vs train mean 0.91: name identity became irrelevant. The turn-1
diagnostic agrees from the other side: the base model carries a generic
rightward lean for every name (+0.11 to +0.42, no name structure beyond
noise at K=72), and by ckpt-0300 every lean has collapsed to ~0 (STAY
0.83–0.99 at aligned states — the reactive-correct answer). Training didn't
sharpen name-conditioning; it *erased* what little name-linked bias existed.
Meanwhile the chain-of-thought atrophied to literally empty `<think>`
blocks (122 → 35 tok/ep; entropy 0.29 → 0.17; KL from base climbed to ~0.2)
while rstd stayed healthy (0.24–0.33) — no collapse, just compilation:
outcome RL distilled the policy into a name-blind reactive controller and
threw the reasoning away. **Why this happened is the real finding:** drift
here is observable turn-to-turn and correctable within the 5-turn horizon
(the scripted *reactive* policy already hits ≈0.95), so the reward never
paid a single point for using the name. The KL-cheapest solution consistent
with the reward is name-blind, and that's exactly what GRPO found.
Retrospectively this adjudicates run 2's banana +0.08 as story (b), a
generic competence lift. It also answers "is the CoT load-bearing?" for
this task: 0.86–0.98 catch rates with empty think blocks — decoration. And
it hands Experiment 2 its missing design requirement (see that section):
the environment must be rebuilt so that a reactive policy *caps out* below
a name-using one, or the same washout will happen again.

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
uv run python train.py --eval runs/ckpt-0100 --eval-n 100   # a checkpoint, full holdout battery
uv run python analysis/diag_name_conditioning.py Qwen/Qwen3-0.6B runs/ckpt-0100   # turn-1 name-lean tables
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
