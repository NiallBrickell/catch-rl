# Experiment 3 cluster validation — Qwen3-0.6B input-embedding audit (valence routing)

**Date:** 2026-07-24
**Model:** Qwen/Qwen3-0.6B (input embedding matrix only, `model.embed_tokens.weight`, 151936 x 1024, loaded on CPU via partial safetensors read — no forward passes, no generation).
**Script:** `validate_exp3_clusters.py` (this directory) reproduces every number below (the candidate-search pass used a throwaway variant of the same method over 12 treat / 10 sting candidates).
**Method:** identical to the Exp 2 audit — each word embedded as its space-prefixed mid-sentence form (verified identical inside the frame "The {word} is at column 3"), input embeddings mean-pooled over the word's tokens, cosine similarity on the pooled vectors. New for Exp 3: cosine to the anchors ` good` / ` bad` as a crude valence-prior readout.

## Final recommendation

```python
EXP3_WORDS = {
    "treat": ["candy", "cake", "cookie"],
    "sting": ["viper", "scorpion", "snake"],
    "nonce": ["zellik", "vantrix"],
}
```

Headline geometry (input-embedding cosine):

| pair | within mean | cross mean | gap | weakest word (margin) |
|---|---|---|---|---|
| treat/sting | **0.260** | 0.089 | **+0.171** | snake (+0.111) |

Every word sits closer to its own cluster than to the opposite one with min per-word margin **+0.111** — stronger than Exp 2's pair 2 (gap +0.088, min margin +0.072) and approaching pair 1 (+0.251). Both nonces are essentially orthogonal to both clusters (max |cos| = **0.006** — an order of magnitude cleaner than cromlet/torgim were against the Exp 2 clusters).

Valence axis (cos to ` good` minus cos to ` bad`): all three treats positive, all three stings negative, nonces in between — the clusters separate on the axis and the ordering is total (every treat g−b > every sting g−b).

## What changed from the original candidates, and why

**The proposed {grape, cherry, melon} vs {wasp, hornet, bee} fails outright: gap = −0.018** (cross mean 0.064 *exceeds* within mean 0.046). Four of the six words fail the per-word test:

- **bee** — worst failure (margin **−0.135**): it sits in the *treat* semantic field, not with wasp/hornet (bee~honey 0.357, bee~peach 0.248, bee~grape 0.189 vs bee~wasp **−0.069**). The honey~bee leakage flagged in advance is real and is only the visible tip: bee is a pollinator/food-adjacent word in this space, not a stinger.
- **wasp** — margin −0.021, and *negative* within-cluster mean (−0.029: wasp~hornet 0.010, wasp~bee −0.069). Tokenizes as ` was`+`p` — the pooled vector is dominated by the generic function-word piece ` was` (the exact failure mode that sank anvil in Exp 2).
- **hornet** — margin −0.025; ` horn`+`et`, pooled toward ` horn`. The "insects that sting" category simply has no lexical cohesion at layer 0.
- **melon** — margin **−0.089**: near-zero ties to its own cluster (grape~melon 0.039, cherry~melon −0.019; ` mel`+`on` is 2-token) while melon~hornet = 0.141.
- **grape, cherry** — individually positive margins (+0.084, +0.075) but they can't carry a cluster whose third member is repelled, cherry's valence prior points the wrong way (g−b = −0.010), and a fruit-treat cluster overlaps categorically with Exp 1's all-fruit battery (surface-form rules pass, but re-using the fruit category for "treat" would blur the Exp 1 / Exp 3 story). peach and mango have the same categorical problem plus high cross-similarity to the animal stings (peach~bee 0.248, peach~viper 0.177).

**Replacement search.** Brute force over all 3v3 combinations from 12 treat candidates {grape, cherry, melon, peach, mango, candy, honey, toffee, caramel, cake, cookie, sweet} x 10 sting candidates {wasp, hornet, bee, viper, nettle, thorn, scorpion, cobra, spider, snake} (min-margin objective, shared-piece combos excluded). Desserts vs venomous animals dominate the leaderboard. The raw min-margin winner is {candy, caramel, cake} vs {scorpion, spider, snake} (+0.155), but it fails the valence-direction check: **caramel** g−b = −0.039 and **spider** g−b = +0.029, both wrong-signed. Filtering for correct valence sign on every word gives **{candy, cake, cookie} vs {viper, scorpion, snake}**: min margin +0.111, gap +0.171, all words correct-signed on the good/bad axis, all nouns, no cross-cluster polysemy.

Other rejections from the pool:

- **berry** — banned before scoring (substring of Exp 1's strawberry).
- **thorn** — wrong-signed valence prior (g−b +0.026), weak margin, ` th`+`orn`.
- **nettle** — margin ≈ 0 against the pools; 2-token.
- **toffee** — within-cluster mean only 0.069 (` to`+`ffee` pooled toward ` to`).
- **honey** — passes numerically (pool margin +0.074, correct valence sign) but is the designated bee-leakage hazard and the weakest of the viable treats; not needed once bee is gone, excluded on design grounds.
- **cobra** — tokenizes ` c`+`obra` (shares the 1-char piece ` c` with Exp 2's cromlet; below the salience bar, but the within edges are also weaker than scorpion's).
- **sweet** — passes everything (see backup below) but rejected as primary because it *is* a valence adjective: putting a near-synonym of "good" inside the treat cluster makes the valence-prior readout circular, and "The sweet is at column 3" reads as a Briticism.

## Tokenization (mid-sentence, space-prefixed)

All words verified to tokenize identically inside "The {word} is at column 3".

| word | n | pieces | notes |
|---|---|---|---|
| candy | 1 | ` candy` | |
| cake | 1 | ` cake` | "piece of cake" idiom is valence-congruent; harmless |
| cookie | 1 | ` cookie` | browser-cookie polysemy — neutral domain, no cross-cluster leak; g−b ≈ 0 likely reflects it |
| viper | 1 | ` viper` | minor polysemy (Dodge Viper; F-16 nickname) — see cross-experiment notes |
| scorpion | 2 | ` sc`+`orpion` | only 2-token cluster word (Exp 2 had one too: tortoise) |
| snake | 1 | ` snake` | |
| zellik | 3 | ` z`+`ell`+`ik` | nonce — 3-piece gibberish, as desired |
| vantrix | 3 | ` van`+`tr`+`ix` | nonce |

Shared-subword-piece check across the opposite clusters: **none**. Nonce pieces (≥2 chars) share nothing with either cluster or with any Exp 1/Exp 2 word.

## Full geometry

Within treat: candy~cake 0.313, candy~cookie 0.290, cake~cookie 0.286 — the tightest, most uniform cluster in the whole design (Exp 2's best single edge was rock~stone 0.503 but its cluster means were 0.20–0.32; here every edge is ≥ 0.286).
Within sting: viper~snake 0.300, viper~scorpion 0.183, scorpion~snake 0.186.
Cross (all 9): 0.008 – 0.133.

| word | within | cross | margin |
|---|---|---|---|
| candy | 0.302 | 0.099 | +0.202 |
| cake | 0.299 | 0.083 | +0.216 |
| cookie | 0.288 | 0.084 | +0.205 |
| viper | 0.241 | 0.107 | +0.135 |
| scorpion | 0.184 | 0.028 | +0.157 |
| snake | 0.243 | 0.132 | +0.111 |

**Within 0.2596 vs cross 0.0887 — gap +0.1709.**

If one trained member per cluster is wanted: **candy** (or cake — margins are near-identical, edges to both held-out neighbors ≈ 0.29–0.31) and **viper** (its two held-out neighbors span a similarity gradient: snake 0.300 vs scorpion 0.183, mirroring the rock→stone/granite dynamic-range argument from Exp 2).

## Valence axis (cos to ` good` / ` bad` anchors)

| word | good | bad | g−b |
|---|---|---|---|
| candy | +0.020 | −0.002 | **+0.022** |
| cake | +0.058 | +0.040 | **+0.018** |
| cookie | −0.009 | −0.010 | **+0.001** |
| viper | −0.023 | −0.012 | **−0.011** |
| scorpion | −0.012 | +0.064 | **−0.076** |
| snake | +0.010 | +0.040 | **−0.030** |
| zellik | +0.022 | +0.011 | +0.010 |
| vantrix | +0.003 | −0.015 | +0.018 |

Cluster means: **treat +0.014, sting −0.039** (nonce +0.014). Every treat is g−b-positive, every sting negative, and the ordering is total: min treat (cookie +0.001) > max sting (viper −0.011). As expected the prior lives mostly on the *sting* side (dangerous animals are lexically "bad"; desserts are only weakly "good" at layer 0). Honest caveats: (a) the magnitudes are tiny relative to anisotropy noise (good~bad themselves cosine at 0.554 — sentiment antonyms are near-synonyms in embedding space), so this is a directional sanity check, not a measurement; (b) the nonces' g−b values (+0.010/+0.018) numerically overlap the treat range — the correct reading is "nonces don't separate from zero", not "nonces are treats"; their |cos| to good and bad individually is ≤ 0.022. The real valence-prior measurement in Exp 3 is behavioral (avoidance onset lag per assignment), not this.

## Nonce placement

zellik and vantrix (Exp 2's validated runners-up) re-verified against the *new* clusters:

| nonce | vs treat | vs sting | max abs | g−b |
|---|---|---|---|---|
| zellik | +0.000 | −0.005 | **0.005** | +0.010 |
| vantrix | −0.005 | −0.006 | **0.006** | +0.018 |

Both are indistinguishable from orthogonal — far cleaner against these clusters (max |cos| 0.006) than cromlet/torgim were against Exp 2's (0.038/0.050). zellik~vantrix = 0.114 (multi-token gibberish vectors correlate; harmless, they are controls, not a cluster). Prefer them as recommended: already screened once, and no piece overlap with cromlet/torgim, so all four nonces across Exp 2/Exp 3 stay mutually surface-disjoint.

## Sanity checks

- **Exp 1 + Exp 2 collisions:** none of the 8 chosen words equals, substring-relates to, or shares any ≥2-char token piece with strawberry, apple, orange, banana, tangerine, plum, blorple, quorf, rock, stone, granite, feather, moth, butterfly, missile, bullet, jet, tortoise, turtle, worm, cromlet, torgim (script check: `problems: none`).
- **Substring relations among chosen words:** none.
- **Mid-sentence tokenization:** identical to the space-prefixed form for every word.
- **Cross-experiment semantic adjacency (informational — no rule violated, different training runs):** snake~turtle 0.310, snake~worm 0.239, snake~tortoise 0.226 (the sting cluster leans reptile, Exp 2's slow cluster was reptiles); viper~missile 0.202 (weapon-adjacent naming). If Exp 3 internals are ever compared *across* experiments' checkpoints, remember the sting cluster is not lexically independent of Exp 2's slow cluster.

## Backup set (validated, ready if a swap is ever needed)

- **Treat alternative:** {candy, cake, sweet} vs same stings — gap +0.147, min margin +0.124, and a wider valence separation (treat mean g−b +0.030, min treat +0.018). Rejected as primary only because "sweet" is itself a sentiment adjective (circular valence readout) and an awkward noun in the game frame.
- Raw-geometry winner {candy, caramel, cake} vs {scorpion, spider, snake} (min margin +0.155) is available if the valence-direction criterion is ever dropped — caramel and spider are wrong-signed on g−b.

## Caveats

Same four as the Exp 2 audit, unchanged: input-embedding cosine is a layer-0 proxy (certifies tokenization hygiene, surface-form independence, and coarse neighborhood structure — the residual-stream probes are the real measurement); the space is anisotropic so only within-vs-cross *differences* mean anything; mean-pooling dilutes multi-token words (only scorpion at 2 tokens here, plus the deliberately-3-token nonces); and the nonzero cross floor (worst: candy~snake / snake-column ≈ 0.13) should be re-checked before invoking deep explanations if the sting cluster ever appears to inherit treat-cluster value above the nonce floor. Exp-3-specific addition: the good/bad anchor readout is directional only — anchors themselves cosine at 0.554, so absolute g−b magnitudes below ~0.02 (cookie, viper) are sign-consistent noise; the designed measurement of valence priors is the behavioral onset-lag comparison across assignments.
