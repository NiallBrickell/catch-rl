# Experiment 2 cluster validation — Qwen3-0.6B input-embedding audit

**Date:** 2026-07-23
**Model:** Qwen/Qwen3-0.6B (input embedding matrix only, `model.embed_tokens.weight`, 151936 x 1024, loaded on CPU via partial safetensors read — no forward passes, no generation).
**Script:** `analysis/validate_exp2_clusters.py` reproduces every number below (the candidate-search passes that led here used throwaway variants of the same method).
**Method:** each word embedded as its space-prefixed mid-sentence form (verified identical inside the frame "The {word} is at column 3"), input embeddings mean-pooled over the word's tokens, cosine similarity on the pooled vectors.

## Final recommendation

```python
EXP2_WORDS = {
    "pair1_heavy": ["rock", "stone", "granite"],
    "pair1_light": ["feather", "moth", "butterfly"],
    "pair2_fast":  ["missile", "bullet", "jet"],
    "pair2_slow":  ["tortoise", "turtle", "worm"],
    "nonce":       ["cromlet", "torgim"],
}
```

Headline geometry (input-embedding cosine, mean over pairs):

| pair | within mean | cross mean | gap | weakest word (margin) |
|---|---|---|---|---|
| pair 1 heavy/light | **0.324** | 0.074 | **+0.251** | feather (+0.128) |
| pair 2 fast/slow | **0.202** | 0.114 | **+0.088** | worm (+0.072) |

Every individual word sits closer to its own cluster than to the opposite one (min per-word margin +0.072); both nonce words sit near zero cosine to all four clusters (max |cos| = 0.050).

## What changed from the original candidates, and why

**Pair 1.** The proposed {rock, anvil, brick} vs {feather, leaf, petal} technically separates (gap +0.031) but two words fail the per-word test in this embedding space: **anvil** (margin +0.006; tokenizes as ` an`+`vil`, and its pooled vector is dominated by the generic piece ` an`) and **petal** (margin **−0.008** — pooled ` pet`+`al` sits marginally closer to the *heavy* cluster than to feather/leaf). A brute-force search over 13 heavy / 15 light candidates found {rock, stone, granite} vs {feather, moth, butterfly} with 8x the gap and all margins ≥ +0.128. rock~stone = 0.503 is the strongest within-cluster edge anywhere in the design — ideal for the trained-member → held-out-neighbor inheritance test.

**Pair 2.** The proposed fast/slow axis is sound, but three candidate words had to go:

- **rocket** — hard sanity failure: superstring of pair-1's "rock" (the substring rule exists precisely because a shared surface form is a transfer confound across *pairs*).
- **tortoise + snail together with missile/bullet/arrow** — marginal geometry: tortoise margin +0.001, snail +0.013 (missile~tortoise = 0.128 and arrow~tortoise = 0.107 eat the separation).
- **slug** — passes numerically but is slang for a bullet, i.e. direct polysemy leakage into the opposite cluster's semantic field. Excluded on design grounds.

Targeted search over 6 fast / 16 slow candidates (min-margin objective) yields {missile, bullet, jet} vs {tortoise, turtle, worm}: gap +0.088, min margin +0.072, all words ≤ 2 tokens, no shared pieces, no polysemy hazards.

**Nonce words.** Tested 8 fresh candidates. **cromlet** (` c`+`rom`+`let`) and **torgim** (` t`+`org`+`im`) are the two whose pooled vectors sit nearest zero against *all four* clusters (max |cos| 0.038 and 0.050 respectively). Neither is an English word, neither carries any direction/speed/weight connotation, neither shares a token piece with any cluster word, and neither collides with Exp 1's blorple/quorf.

## Tokenization (mid-sentence, space-prefixed)

All cluster words verified to tokenize identically inside "The {word} is at column 3".

| word | n | pieces | notes |
|---|---|---|---|
| rock | 1 | ` rock` | |
| stone | 1 | ` stone` | |
| granite | 1 | ` granite` | |
| feather | 1 | ` feather` | |
| moth | 1 | ` moth` | |
| butterfly | 1 | ` butterfly` | |
| missile | 1 | ` missile` | |
| bullet | 1 | ` bullet` | |
| jet | 1 | ` jet` | mild polysemy (jet black / jet stream); dominant sense is the fast one |
| tortoise | 2 | ` tort`+`oise` | only 2-token cluster word |
| turtle | 1 | ` turtle` | |
| worm | 1 | ` worm` | |
| cromlet | 3 | ` c`+`rom`+`let` | nonce — 3 pieces is desirable here (tokenizes as gibberish) |
| torgim | 3 | ` t`+`org`+`im` | nonce |

Shared-subword-piece check across opposite clusters of each pair: **none** (pair 1 and pair 2). Nonce pieces (≥2 chars) share nothing with any cluster word's pieces.

## Full geometry

### Pair 1 — heavy/light

Within heavy: rock~stone 0.503, rock~granite 0.322, stone~granite 0.389.
Within light: feather~moth 0.182, feather~butterfly 0.267, moth~butterfly 0.281.
Cross (all 9): 0.012 – 0.122.

| word | within | cross | margin |
|---|---|---|---|
| rock | 0.412 | 0.035 | +0.377 |
| stone | 0.446 | 0.090 | +0.356 |
| granite | 0.355 | 0.095 | +0.260 |
| feather | 0.225 | 0.096 | +0.128 |
| moth | 0.232 | 0.052 | +0.179 |
| butterfly | 0.274 | 0.072 | +0.203 |

**Within 0.3240 vs cross 0.0735 — gap +0.2506.**

### Pair 2 — fast/slow

Within fast: missile~bullet 0.269, missile~jet 0.177, bullet~jet 0.139.
Within slow: tortoise~turtle 0.254, tortoise~worm 0.186, turtle~worm 0.186.
Cross (all 9): 0.055 – 0.178 (highest: missile~turtle 0.178, bullet~turtle 0.173).

| word | within | cross | margin |
|---|---|---|---|
| missile | 0.223 | 0.149 | +0.075 |
| bullet | 0.204 | 0.119 | +0.085 |
| jet | 0.158 | 0.075 | +0.083 |
| tortoise | 0.220 | 0.088 | +0.132 |
| turtle | 0.220 | 0.141 | +0.079 |
| worm | 0.186 | 0.114 | +0.072 |

**Within 0.2018 vs cross 0.1141 — gap +0.0877.**

Pair 2's separation is genuinely weaker than pair 1's on this proxy — expected: fast/slow is a behavioral property cutting across taxonomic categories, while heavy/light here rides a materials-vs-flying-things split. Every word still clears its own margin comfortably, and having one tight pair and one looser pair is arguably informative for the replication claim (transfer strength can be read against cluster tightness).

### Nonce placement

| nonce | vs heavy | vs light | vs fast | vs slow | max abs |
|---|---|---|---|---|---|
| cromlet | −0.007 | −0.038 | −0.020 | −0.029 | 0.038 |
| torgim | −0.017 | −0.046 | −0.050 | −0.013 | 0.050 |

Both are an order of magnitude below the within-cluster means. (cromlet~torgim = 0.202 — pooled multi-token gibberish vectors correlate with each other; harmless, since the nonces are controls, not a cluster.)

## Sanity checks

- **Exp 1 collisions:** none of the 14 chosen words equals or substring-relates to strawberry, apple, orange, banana, tangerine, plum, blorple, quorf.
- **Substring relations among chosen words:** none (this is what eliminated rocket/rock).
- **Mid-sentence tokenization:** identical to the space-prefixed form for every word.

## Backup sets (validated, ready if a swap is ever needed)

- **Pair 2 alternative:** {arrow, dart, jet} vs {snail, slug, sloth} — gap +0.099, min margin +0.061, lowest cross-mean of any fast/slow combo (0.036). Rejected as primary only because of slug's bullet polysemy and snail/sloth being 2-token.
- **Pair 1 light alternative:** {balloon, bubble, butterfly} — gap +0.256, min margin +0.136, marginally better numbers than feather/moth/butterfly. Rejected as primary because balloons and bubbles *rise*, which reads oddly in a falling-object prompt frame.
- Nonce runners-up: zellik, vantrix (max cluster |cos| ≤ 0.055 in the first-pass screen).

## Suggested train/held-out split (optional)

If one trained member per cluster: **rock** (margin +0.377; strongest edges to both held-out neighbors), **butterfly** (+0.203), **bullet** (+0.085; lower cross than missile), **tortoise** (+0.132; highest slow-cluster within, lowest cross). Held-out neighbors then span a similarity gradient (e.g. stone at 0.503 vs granite at 0.322 from rock), which gives the inheritance measurement dynamic range.

## Caveats

1. **Input-embedding cosine is a weak proxy.** These are layer-0 static vectors: no context, none of the model's deeper semantic geometry where the transfer (if any) will actually be mediated. This audit certifies tokenization hygiene, absence of surface-form leakage, and coarse lexical-neighborhood structure — nothing more. The phase-2 residual-stream probes (mid-layer representations in actual game states) are the real measurement; a quick mid-layer version of this same check would be a cheap upgrade if wanted.
2. **Anisotropy:** absolute cosines in this space are small and offset (the embedding space is anisotropic); only the within-vs-cross *differences* are meaningful, not the raw values.
3. **Mean-pooling multi-token words is crude** — it dilutes multi-token words toward their pieces' generic vectors (exactly the failure that sank anvil and petal). The final set keeps 13 of 14 words at ≤2 tokens (only tortoise at 2, plus the deliberately-3-token nonces).
4. **Pair-2 cross similarities are nonzero** (missile~turtle 0.178 is the worst). If Exp 2's aligned-state probe shows the slow cluster inheriting fast-cluster dynamics above the nonce floor, check this before invoking deeper explanations.
