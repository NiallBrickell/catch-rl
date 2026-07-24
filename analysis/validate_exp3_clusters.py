"""Final validation of the chosen Experiment-3 word sets (embedding-only, CPU).

Valence routing: treat/sting cluster pair + 2 nonce words. Same method as
analysis/validate_exp2_clusters.py — Qwen3-0.6B input embeddings only,
mean-pooled over space-prefixed mid-sentence tokens, cosine similarity.
Extra Exp-3 check: cosine to the anchors " good" / " bad" as a crude
valence-prior readout.
"""

import itertools

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
# every word already used by Exp 1 + Exp 2 (collision list)
USED = ["strawberry", "apple", "orange", "banana", "tangerine", "plum",
        "blorple", "quorf",
        "rock", "stone", "granite", "feather", "moth", "butterfly",
        "missile", "bullet", "jet", "tortoise", "turtle", "worm",
        "cromlet", "torgim"]

FINAL = {
    "treat": ["candy", "cake", "cookie"],
    "sting": ["viper", "scorpion", "snake"],
    "nonce": ["zellik", "vantrix"],
}
VARIANTS = {
    "ORIGINAL proposal (rejected)": (
        ["grape", "cherry", "melon"], ["wasp", "hornet", "bee"]),
    "backup treat_alt (sweet)": (
        ["candy", "cake", "sweet"], ["viper", "scorpion", "snake"]),
}
ANCHORS = ["good", "bad"]

tok = AutoTokenizer.from_pretrained(MODEL)
path = hf_hub_download(MODEL, "model.safetensors")
with safe_open(path, framework="pt", device="cpu") as f:
    emb = f.get_slice("model.embed_tokens.weight")[:].to(torch.float32)

INFO = {}
ALL = [w for ws in FINAL.values() for w in ws] + \
      [w for a, b in VARIANTS.values() for w in a + b] + ANCHORS + USED
for w in sorted(set(ALL)):
    ids = tok.encode(" " + w, add_special_tokens=False)
    sent = tok.encode(f"The {w} is at column 3", add_special_tokens=False)
    consistent = any(sent[i:i + len(ids)] == ids for i in range(len(sent)))
    v = emb[torch.tensor(ids)].mean(dim=0)
    INFO[w] = {"ids": ids, "pieces": [tok.decode([i]) for i in ids],
               "consistent": consistent, "v": v / v.norm()}


def cos(a, b):
    return float(INFO[a]["v"] @ INFO[b]["v"])


print("== TOKENIZATION (space-prefixed / mid-sentence) ==")
for w in sorted(set([x for ws in FINAL.values() for x in ws] +
                    [x for a, b in VARIANTS.values() for x in a + b] +
                    ANCHORS)):
    i = INFO[w]
    flag = "" if i["consistent"] else "  [MID-SENTENCE DIFFERS!]"
    print(f"  {w:10s} n={len(i['ids']):d} pieces={i['pieces']}{flag}")


def pair_report(name, A, B):
    within = [cos(x, y) for x, y in itertools.combinations(A, 2)] + \
             [cos(x, y) for x, y in itertools.combinations(B, 2)]
    cross = [cos(x, y) for x in A for y in B]
    wm, cm = sum(within) / len(within), sum(cross) / len(cross)
    print(f"\n== {name}: {A} vs {B} ==")
    for x, y in itertools.combinations(A, 2):
        print(f"  within-A {x}~{y} = {cos(x, y):.3f}")
    for x, y in itertools.combinations(B, 2):
        print(f"  within-B {x}~{y} = {cos(x, y):.3f}")
    for x in A:
        for y in B:
            print(f"  cross    {x}~{y} = {cos(x, y):.3f}")
    print(f"  WITHIN mean={wm:.4f}  CROSS mean={cm:.4f}  GAP={wm - cm:.4f}")
    for w in A + B:
        own, other = (A, B) if w in A else (B, A)
        wi = sum(cos(w, o) for o in own if o != w) / (len(own) - 1)
        cr = sum(cos(w, o) for o in other) / len(other)
        print(f"    {w:10s} within={wi:.3f} cross={cr:.3f} margin={wi - cr:+.3f}")
    return wm, cm


pair_report("TREAT/STING (final)", FINAL["treat"], FINAL["sting"])
for name, (A, B) in VARIANTS.items():
    pair_report(f"VARIANT {name}", A, B)

print("\n== VALENCE AXIS (cos to ' good' / ' bad' anchors) ==")
for group in ("treat", "sting", "nonce"):
    gs = []
    for w in FINAL[group]:
        g, b = cos(w, "good"), cos(w, "bad")
        gs.append(g - b)
        print(f"  {group:5s} {w:10s} good={g:+.3f} bad={b:+.3f} g-b={g - b:+.3f}")
    print(f"    {group} mean g-b = {sum(gs) / len(gs):+.4f}")
print(f"  anchor sanity: good~bad = {cos('good', 'bad'):.3f}")

print("\n== NONCE placement (mean cos to each final cluster) ==")
for n in FINAL["nonce"]:
    sims = {k: sum(cos(n, w) for w in ws) / len(ws)
            for k, ws in FINAL.items() if k != "nonce"}
    mx = max(abs(v) for v in sims.values())
    print(f"  {n:8s} " + "  ".join(f"{k}={v:+.3f}" for k, v in sims.items()) +
          f"   max|cos|={mx:.3f}")
print(f"  nonce~nonce: zellik~vantrix = {cos('zellik', 'vantrix'):.3f}")

print("\n== SANITY ==")
chosen = [w for ws in FINAL.values() for w in ws]
problems = []
for w in chosen:
    if w in USED:
        problems.append(f"{w}: Exp1/Exp2 collision")
    for e in USED:
        if w != e and (w in e or e in w):
            problems.append(f"{w}/{e}: Exp1/Exp2 substring relation")
for a, b in itertools.combinations(chosen, 2):
    if a in b or b in a:
        problems.append(f"substring pair {a}/{b}")
# salient (>=2 char) token-piece overlap with any already-used word
for w in chosen:
    wp = {p.strip() for p in INFO[w]["pieces"] if len(p.strip()) >= 2}
    for e in USED:
        ep = {p.strip() for p in INFO[e]["pieces"] if len(p.strip()) >= 2}
        if wp & ep:
            problems.append(f"{w} shares pieces with used {e}: {wp & ep}")
# token-piece overlap across the opposite clusters, and nonce vs both
sa = {p.strip() for w in FINAL["treat"] for p in INFO[w]["pieces"]}
sb = {p.strip() for w in FINAL["sting"] for p in INFO[w]["pieces"]}
if sa & sb:
    problems.append(f"treat/sting shared pieces: {sa & sb}")
np_ = {p.strip() for w in FINAL["nonce"] for p in INFO[w]["pieces"]
       if len(p.strip()) >= 2}
if np_ & (sa | sb):
    problems.append(f"nonce shares pieces with clusters: {np_ & (sa | sb)}")
print("problems:", problems or "none")

print("\n== CROSS-EXPERIMENT semantic adjacency (informational, not a rule) ==")
for a, b in [("snake", "worm"), ("snake", "turtle"), ("snake", "tortoise"),
             ("viper", "jet"), ("viper", "missile"), ("candy", "plum"),
             ("cookie", "apple"), ("cake", "strawberry")]:
    print(f"  {a}~{b} = {cos(a, b):+.3f}")
