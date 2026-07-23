# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A from-scratch educational RL project: multi-turn GRPO on Qwen3-0.6B playing a
text-based falling-fruit catch game, in plain PyTorch. Deliberately **no RL
libraries** (no TRL, no veRL, no verifiers) — the hand-rolled loop *is* the
point. Only `torch` and `transformers` (for pretrained weights) are allowed as
ML dependencies; don't introduce RL frameworks to "simplify" things.

The experiment: train on strawberry/apple/orange, then eval on a held-out
battery — banana (real, matched), tangerine (real, near-orange, misleading),
plum (real, neutral), blorple/quorf (nonce floors). All five stay out of
training (`TRAIN_FRUITS` vs `EVAL_FRUITS` in `catch_env.py`); that split is
the entire experiment. The README explains the hypothesis, battery logic,
and predictions; `paper/draft.md` carries positioning and results.

## Commands

```bash
uv run python catch_env.py            # env self-test: scripted-policy catch rates
uv run python train.py --check-mask   # verify the gradient mask is exact
uv run python train.py --steps 100    # train (logs → runs/log.jsonl, ckpt → runs/ckpt-NNNN every 20 steps)
uv run python train.py --eval         # greedy catch-rate table incl. banana (base model)
uv run python train.py --eval runs/ckpt-0100   # same, from a checkpoint
```

There is no test suite or linter; `--check-mask` and the env self-test are the
verification tools. Run `--check-mask` after any change touching tokenization,
templating, rollout, or masking. Training runs on MPS (bfloat16) if available,
else CPU (float32).

## Architecture

Two files. `catch_env.py` is the environment (7 columns, 5 turns, plain-text
observations; drift applies after the fall on turns 2 and 4). `train.py` is
everything else, in four pieces:

1. **Chat-format glue (`derive_segments`)** — Qwen3's chat template strips
   earlier turns' `<think>` blocks on re-render, so re-templating mid-episode
   would silently change the sequence. Instead the template's glue strings are
   probed once via sentinel and the transcript is concatenated manually, so the
   training-time token sequence is byte-identical to rollout-time.

2. **Rollout (`rollout`)** — batched `generate()` per game turn with
   left-padding, no grad. Each episode keeps ONE token list plus a parallel
   boolean mask: True exactly on tokens the model sampled (including its EOS),
   False on observations and scaffolding. This mask discipline is the #1
   source of silent bugs in multi-turn RL; `check_mask` verifies that decoding
   the mask=True positions reconstructs the generated text exactly.

3. **GRPO update (`grpo_step` / `batched_logprobs`)** — `generate()` builds no
   graph, so full episodes are re-forwarded (right-padded, micro-batched by
   `MICRO=2` because the [B, L, ~150k-vocab] logits are the memory spike) to
   get logprobs with grad. Advantage = reward − group mean; all rollouts in a
   group share one (fruit, seed) so the baseline is exact for that state.
   Deliberate choices, don't "fix" them: no std division (Dr.GRPO), no PPO
   clipping (one update per fresh batch → ratio ≡ 1, so this is REINFORCE with
   a group baseline), plus a k3-estimator KL penalty against a frozen copy of
   the initial model. Note the off-by-one in `batched_logprobs`: logits[:, t]
   scores token t+1, so both logprobs and mask are shifted.

4. **Eval (`evaluate`)** — greedy rollouts over `EVAL_FRUITS` with fixed seeds
   (10_000+i), reporting per-fruit catch rate with banana tagged held-out.

## Invariants to preserve

- **Sampling distribution == learning distribution.** `main()` resets Qwen's
  default sampler (temp/top_p/top_k) to neutral because logprobs are computed
  for the tokens as if sampled from the raw distribution. Any change to
  generation settings must keep these consistent.
- **Gradient flows only through mask=True tokens.** Anything the environment
  or template contributed must be mask=False.
- **Groups share identical initial state** (same fruit + seed), or the group
  mean stops being a valid baseline.
- Unparseable model output falls back to `STAY` (last `ACTION:` match wins).

Beyond the two core files: `analysis/` holds diagnostics (the turn-1
name-conditioning probe), and three standalone Marimo notebooks form a
teaching ladder — `math_refresher.py` (calculus/probability up to the
score-function trick), `ml_primer.py` (backprop → attention → LM-as-
classifier), `grpo_notes.py` (REINFORCE → GRPO, capstone learns this env).
Notebooks deliberately import nothing from the trainer — keep them
self-contained.

## Watching a run

Per-step JSONL logs go to `runs/log.jsonl`. Key signals (see README for
detail): `frac_zero_var_groups` near 1.0 means no learning signal (all-0/all-1
groups); shrinking `gen_tokens_per_ep` means the chain-of-thought is
atrophying; a KL spike means the policy is bolting. Sample transcripts print
every 10 steps — read them for parser-hacking/degenerate outputs.

## Conventions

- **Checkpoint with git, always.** Commit after every meaningful change —
  code, docs, or experiment configuration — with a message saying *why*.
  History is the changelog; small frequent commits over big silent ones.
- **Docs travel with code.** Any design or behavior change updates README.md
  and this file in the same commit.
- `runs/` (logs, model checkpoints) is gitignored — record notable results by
  writing them into the README/commit messages, not by committing checkpoints.

## Roadmap

- **Next experiment: counterbalanced semantic routing** (Experiment 2 in the
  README/paper) — semantic clusters with arbitrary cluster→dynamics
  assignments *reversed across matched runs*, several seeds per assignment,
  ≥2 independent cluster pairs. ~15 training runs: a studio-week, or ~$20 of
  rented 4090 time. **Run-3 verdict (2026-07-23) unblocked this and added a
  hard requirement:** the converged catch policy is name-blind (nonce words
  hit 0.86–0.90, tangerine +0.85, turn-1 leans → 0, CoT empty) because the
  env is reactively solvable. Exp 2's env must make anticipation decisive —
  drift on the final fall (after the last action) or magnitude that outruns
  remaining moves — with an acceptance test *before* training: scripted
  reactive ceiling ≤0.5, scripted name-aware ceiling ≥0.9.
- Later: `gym_env.py` — a Gymnasium adapter (text-serialized state vectors,
  action-repeat so the horizon stays LLM-sized, no vision). CartPole first as
  validation that 0.6B can control real physics through text at all, then
  LunarLander with wind/gravity settings *described in language* in the
  prompt, eval on novel descriptions — the real-physics transfer experiment.
- Bigger runs (Qwen3-1.7B+, LoRA): the M4 Max studio on the local network,
  `ssh studio`.
- Engineering debt (from external review): proper run directories — config,
  optimizer/RNG/global-step state for exact resumption, per-checkpoint eval
  results stored beside checkpoints; currently resuming resets optimizer
  moments and step numbering and appends to the same log.
- **Write-up (TODO, once results exist):** an arXiv-style paper (PDF via
  mdprint/LaTeX, typeset properly — typography is part of the deliverable)
  plus an
  interactive distill.pub-style HTML version. Core figures: per-fruit learning
  curves, the banana-vs-orange delta-from-base bars, the checkpoint erosion
  curve (behavioral + KL-from-base overlaid, per RL's Razor), sample
  transcripts before/after, and phase-2 internals plots (fruit-token geometry
  across checkpoints, patching results). All figure data comes from
  `runs/log.jsonl` + checkpoint evals — keep those logs.
