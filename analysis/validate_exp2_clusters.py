"""Final validation of the chosen Experiment-2 word sets (embedding-only, CPU)."""

import itertools

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
EXP1 = ["strawberry", "apple", "orange", "banana", "tangerine", "plum",
        "blorple", "quorf"]

FINAL = {
    "pair1_heavy": ["rock", "stone", "granite"],
    "pair1_light": ["feather", "moth", "butterfly"],
    "pair2_fast": ["missile", "bullet", "jet"],
    "pair2_slow": ["tortoise", "turtle", "worm"],
    "nonce": ["cromlet", "torgim"],
}
VARIANTS = {
    "pair2_alt (arrow/dart/jet vs snail/slug/sloth)": (
        ["arrow", "dart", "jet"], ["snail", "slug", "sloth"]),
    "pair1_light_alt (balloon)": (["rock", "stone", "granite"],
                                  ["balloon", "bubble", "butterfly"]),
}

tok = AutoTokenizer.from_pretrained(MODEL)
path = hf_hub_download(MODEL, "model.safetensors")
with safe_open(path, framework="pt", device="cpu") as f:
    emb = f.get_slice("model.embed_tokens.weight")[:].to(torch.float32)

INFO = {}
ALL = [w for ws in FINAL.values() for w in ws] + \
      [w for a, b in VARIANTS.values() for w in a + b]
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
for w in sorted(INFO):
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


pair_report("PAIR 1 heavy/light", FINAL["pair1_heavy"], FINAL["pair1_light"])
pair_report("PAIR 2 fast/slow", FINAL["pair2_fast"], FINAL["pair2_slow"])
for name, (A, B) in VARIANTS.items():
    pair_report(f"VARIANT {name}", A, B)

print("\n== NONCE placement (mean cos to each final cluster) ==")
for n in FINAL["nonce"]:
    sims = {k: sum(cos(n, w) for w in ws) / len(ws)
            for k, ws in FINAL.items() if k != "nonce"}
    mx = max(abs(v) for v in sims.values())
    print(f"  {n:8s} " + "  ".join(f"{k}={v:+.3f}" for k, v in sims.items()) +
          f"   max|cos|={mx:.3f}")
print(f"  nonce~nonce: cromlet~torgim = {cos('cromlet', 'torgim'):.3f}")

print("\n== SANITY ==")
chosen = [w for ws in FINAL.values() for w in ws]
problems = []
for w in chosen:
    if w in EXP1:
        problems.append(f"{w}: Exp1 collision")
    for e in EXP1:
        if w != e and (w in e or e in w):
            problems.append(f"{w}/{e}: Exp1 substring relation")
for a, b in itertools.combinations(chosen, 2):
    if a in b or b in a:
        problems.append(f"substring pair {a}/{b}")
# token-piece overlap across opposite clusters of each pair, and nonce vs all
for pa, pb in [("pair1_heavy", "pair1_light"), ("pair2_fast", "pair2_slow")]:
    sa = {p.strip() for w in FINAL[pa] for p in INFO[w]["pieces"]}
    sb = {p.strip() for w in FINAL[pb] for p in INFO[w]["pieces"]}
    if sa & sb:
        problems.append(f"{pa}/{pb} shared pieces: {sa & sb}")
np = {p.strip() for w in FINAL["nonce"] for p in INFO[w]["pieces"] if len(p.strip()) >= 2}
cp = {p.strip() for k, ws in FINAL.items() if k != "nonce"
      for w in ws for p in INFO[w]["pieces"]}
if np & cp:
    problems.append(f"nonce shares pieces with clusters: {np & cp}")
print("problems:", problems or "none")
