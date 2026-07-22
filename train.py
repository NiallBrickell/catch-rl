"""From-scratch multi-turn GRPO on Qwen3-0.6B playing catch_env.

No RL libraries: rollouts via model.generate(), then a re-forward to get
logprobs with a graph, then REINFORCE-with-a-group-baseline. The whole trick
of multi-turn RL on a chat model is bookkeeping: we keep ONE token list per
episode plus a parallel boolean mask that is True only on tokens the model
itself sampled. Env observations and chat-template scaffolding are False.
The policy gradient touches masked-True tokens and nothing else.

Usage:
  uv run python train.py                    # train
  uv run python train.py --check-mask       # verify the generated-token mask
  uv run python train.py --eval [ckpt_dir]  # greedy eval (base model if no dir)
"""

import argparse, gc, json, os, random, re, time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from catch_env import CatchEnv, TRAIN_FRUITS, EVAL_FRUITS, TURNS

SYSTEM_PROMPT = """You are playing a fruit-catching game on a grid with 7 columns, numbered 0 to 6.
A fruit starts at row 0 and falls one row after each of your moves. Your basket is at row 5.
You catch the fruit if your basket is in the fruit's column when the fruit reaches row 5.
Different fruits may drift sideways as they fall: watch how the fruit's column changes
from turn to turn, and aim for where the fruit is going, not where it is now.
Each turn, think briefly (a sentence or two), then answer on a final line with exactly one of:
ACTION: LEFT
ACTION: RIGHT
ACTION: STAY
LEFT moves your basket one column toward 0, RIGHT one column toward 6."""

ACTION_RE = re.compile(r"ACTION:\s*(LEFT|RIGHT|STAY)")
MICRO = 2  # episodes per training forward; bounds the logits tensor (~150k vocab)


def bucket(n, size=128):
    """Round a padded length up to a fixed bucket. The MPS caching allocator
    keeps a cached buffer per distinct tensor shape it has ever served; with
    naturally varying sequence lengths every step allocates fresh ~GB-scale
    logits buffers and the cache grows without bound (run 1 hit 42GB by step
    40). Bucketing makes shapes repeat so buffers get reused."""
    return -(-n // size) * size


# ---------------------------------------------------------------------------
# Chat format. Qwen3's template re-renders can strip earlier turns' content
# (e.g. old <think> blocks), so re-templating the transcript each turn would
# silently change what the model saw mid-episode. Instead we derive the glue
# strings ONCE by probing apply_chat_template, then concatenate manually so
# the training-time sequence is byte-identical to the rollout-time sequence.
# ---------------------------------------------------------------------------

def derive_segments(tok):
    S = "\x00SENTINEL\x00"  # never survives templating by accident
    sys_txt = tok.apply_chat_template([{"role": "system", "content": S}], tokenize=False)
    sys_pre, turn_end = sys_txt.split(S)  # "<|im_start|>system\n" , "<|im_end|>\n"
    both = tok.apply_chat_template(
        [{"role": "system", "content": "A"}, {"role": "user", "content": S}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    head = sys_pre + "A" + turn_end
    assert both.startswith(head), "unexpected chat template shape"
    user_pre, rest = both[len(head):].split(S)
    gen_pre = rest[len(turn_end):]  # includes the empty <think></think> block
    assert rest.startswith(turn_end)
    seg = {"sys_pre": sys_pre, "user_pre": user_pre, "turn_end": turn_end, "gen_pre": gen_pre}
    seg["turn_end_ids"] = tok(turn_end, add_special_tokens=False)["input_ids"]
    seg["im_end"] = tok.convert_tokens_to_ids("<|im_end|>")
    assert seg["turn_end_ids"][0] == seg["im_end"]
    return seg


def new_episode(tok, seg, fruit, seed, group):
    env = CatchEnv(fruit, seed)
    text = (seg["sys_pre"] + SYSTEM_PROMPT + seg["turn_end"]
            + seg["user_pre"] + env.reset() + seg["turn_end"] + seg["gen_pre"])
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return {"env": env, "fruit": fruit, "group": group, "ids": list(ids),
            "mask": [False] * len(ids), "gen_texts": [], "actions": [], "reward": 0.0}


def parse_action(text):
    hits = ACTION_RE.findall(text)  # take the LAST match; unparseable -> STAY
    return hits[-1] if hits else "STAY"


# ---------------------------------------------------------------------------
# Rollout. All episodes in a batch have the same number of turns (always 5),
# so we can run them in lockstep: one generate() call per game turn, with
# left-padding so every row's next sampled token lines up on the right edge.
# ---------------------------------------------------------------------------

@torch.no_grad()
def rollout(model, tok, seg, episodes, device, greedy=False, max_new=80):
    eos_ids = {seg["im_end"], tok.eos_token_id}
    n_gen, t0 = 0, time.time()
    for _ in range(TURNS):
        maxlen = bucket(max(len(e["ids"]) for e in episodes))
        input_ids = torch.full((len(episodes), maxlen), tok.pad_token_id, dtype=torch.long)
        attn = torch.zeros((len(episodes), maxlen), dtype=torch.long)
        for i, e in enumerate(episodes):
            input_ids[i, maxlen - len(e["ids"]):] = torch.tensor(e["ids"])
            attn[i, maxlen - len(e["ids"]):] = 1
        out = model.generate(input_ids=input_ids.to(device), attention_mask=attn.to(device),
                             max_new_tokens=max_new, do_sample=not greedy,
                             pad_token_id=tok.pad_token_id)
        for i, e in enumerate(episodes):
            gen = []
            for t in out[i, maxlen:].tolist():
                gen.append(t)
                if t in eos_ids:  # keep the EOS: the model sampled it, so it is
                    break         # part of the trajectory the gradient sees
            # Everything the model sampled is mask=True -- and nothing else.
            e["ids"].extend(gen)
            e["mask"].extend([True] * len(gen))
            n_gen += len(gen)
            gtxt = tok.decode(gen)
            e["gen_texts"].append(gtxt)
            action = parse_action(gtxt)
            e["actions"].append(action)
            obs, r, done = e["env"].step(action)
            e["reward"] = r
            # Close the assistant turn: if the model already emitted <|im_end|>
            # we only add the template's trailing newline; if it hit the token
            # cap we close the turn for it. Either way it's scaffolding (False).
            scaffold = seg["turn_end_ids"][1:] if (gen and gen[-1] == seg["im_end"]) \
                else seg["turn_end_ids"]
            if not done:
                scaffold = scaffold + tok(seg["user_pre"] + obs + seg["turn_end"]
                                          + seg["gen_pre"], add_special_tokens=False)["input_ids"]
            e["ids"].extend(scaffold)
            e["mask"].extend([False] * len(scaffold))
    return {"gen_tokens": n_gen, "gen_time": time.time() - t0}


def check_mask(model, tok, seg, device):
    """The #1 multi-turn RL bug is a misaligned mask (scaffolding marked as
    generated, or sampled tokens dropped). Verify that decoding exactly the
    mask=True positions reconstructs the model's own text and nothing else."""
    eps = [new_episode(tok, seg, "apple", 123, 0)]
    rollout(model, tok, seg, eps, device)
    e = eps[0]
    assert len(e["ids"]) == len(e["mask"])
    got = tok.decode([t for t, m in zip(e["ids"], e["mask"]) if m])
    want = "".join(e["gen_texts"])  # captured turn-by-turn from generate() output
    print(f"--- decode(mask=True tokens) ({sum(e['mask'])} of {len(e['ids'])}) ---")
    print(got)
    assert got == want, "mask=True tokens do not reconstruct the generated text"
    unmasked = tok.decode([t for t, m in zip(e["ids"], e["mask"]) if not m])
    # the only ACTION: strings on the scaffold side should be the system
    # prompt's own examples -- any extra means model output leaked out of the mask
    assert unmasked.count("ACTION:") == SYSTEM_PROMPT.count("ACTION:"), \
        "model output leaked into mask=False tokens"
    assert SYSTEM_PROMPT in unmasked and "Turn 5 of 5" in unmasked
    print(f"--- OK: mask exactly reconstructs the generated text; "
          f"actions={e['actions']} reward={e['reward']} ---")


# ---------------------------------------------------------------------------
# GRPO update. generate() builds no autograd graph, so we re-forward each full
# episode to recompute logprobs WITH grad -- that re-forward is where the
# policy gradient actually comes from.
# ---------------------------------------------------------------------------

def batched_logprobs(tok, model, episodes, device, grad):
    """Right-pad a micro-batch, one forward, per-token logprob of each token
    actually in the sequence. logits[:, t] scores token t+1, hence the shift;
    the returned mask is likewise shifted so it selects predictions OF
    generated tokens (this off-by-one is a classic silent bug)."""
    maxlen = bucket(max(len(e["ids"]) for e in episodes))
    ids = torch.full((len(episodes), maxlen), tok.pad_token_id, dtype=torch.long)
    attn = torch.zeros_like(ids)
    gmask = torch.zeros((len(episodes), maxlen), dtype=torch.bool)
    for i, e in enumerate(episodes):
        L = len(e["ids"])
        ids[i, :L] = torch.tensor(e["ids"])
        attn[i, :L] = 1
        gmask[i, :L] = torch.tensor(e["mask"])
    ids, attn, gmask = ids.to(device), attn.to(device), gmask.to(device)
    with torch.enable_grad() if grad else torch.no_grad():
        logits = model(input_ids=ids, attention_mask=attn, use_cache=False).logits[:, :-1]
        logp = torch.log_softmax(logits.float(), dim=-1)
        tok_logp = logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)  # [B, L-1]
    return tok_logp, gmask[:, 1:]


def grpo_step(model, ref, opt, tok, episodes, kl_coef, device):
    # Advantage = reward minus the group's mean reward. The group shares one
    # (fruit, seed), so the mean is a baseline for the SAME initial state:
    # "did this rollout do better than my siblings on this exact episode?"
    # Dr.GRPO-style, we do NOT divide by the group std: with binary rewards
    # std-scaling inflates near-deterministic groups (std -> 0) and generally
    # reweights episodes by difficulty rather than by learning signal.
    groups = {}
    for e in episodes:
        groups.setdefault(e["group"], []).append(e["reward"])
    zero_var = sum(1 for rs in groups.values() if max(rs) == min(rs))
    # RAGEN's earliest collapse indicator: within-group reward std. Mean reward
    # lags; the spread of outcomes dies first (the policy stops producing
    # *different* attempts). zero_var is this signal binarized; log both.
    def _std(rs):
        m = sum(rs) / len(rs)
        return (sum((r - m) ** 2 for r in rs) / len(rs)) ** 0.5
    r_std = sum(_std(rs) for rs in groups.values()) / len(groups)
    for e in episodes:
        rs = groups[e["group"]]
        e["adv"] = e["reward"] - sum(rs) / len(rs)
    # With one update per batch of rollouts, the PPO ratio pi/pi_old == 1
    # identically (the weights haven't moved since sampling), so clipping
    # would be a no-op: this is plain REINFORCE with a group baseline, plus a
    # KL leash to the frozen initial model so the policy stays a language model.
    opt.zero_grad()
    N, tot_loss, tot_kl = len(episodes), 0.0, 0.0
    ent_sum, ent_n = 0.0, 0  # -logp of sampled tokens = MC estimate of entropy
    for k in range(0, N, MICRO):  # micro-batch: the [B, L, 150k] logits are the memory hog
        chunk = episodes[k:k + MICRO]
        tok_logp, gm = batched_logprobs(tok, model, chunk, device, grad=True)
        ref_logp, _ = batched_logprobs(tok, ref, chunk, device, grad=False)
        loss = torch.zeros((), device=device)
        for j, e in enumerate(chunk):
            lp = tok_logp[j][gm[j]]          # logprobs of this episode's sampled tokens
            # Entropy = E_{t~pi}[-log pi(t)], and these tokens ARE samples from
            # pi: their mean -logp is an unbiased entropy estimate, for free
            # (the exact entropy would need the full 151k-vocab distribution).
            ent_sum += float(-lp.detach().sum())
            ent_n += lp.numel()
            delta = ref_logp[j][gm[j]].detach() - lp
            k3 = delta.exp() - delta - 1.0   # k3 KL estimator: unbiased-ish, always >= 0
            loss = loss + (-e["adv"] * lp.mean() + kl_coef * k3.mean()) / N
            tot_kl += k3.mean().item() / N
        tot_loss += loss.item()
        loss.backward()  # grads accumulate across micro-batches
    # Pre-clip gradient norm: RAGEN's third early-warning signal (spikes).
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return {"loss": tot_loss, "kl": tot_kl,
            "frac_zero_var_groups": zero_var / len(groups),
            "reward_group_std": r_std,
            "entropy_mc": ent_sum / max(ent_n, 1),
            "grad_norm": float(gnorm)}


def evaluate(model, tok, seg, device, n=30):
    print(f"greedy eval, {n} episodes per fruit:")
    for fruit in EVAL_FRUITS:  # banana never appears in training
        eps = [new_episode(tok, seg, fruit, 10_000 + i, 0) for i in range(n)]
        rollout(model, tok, seg, eps, device, greedy=True)
        rate = sum(e["reward"] for e in eps) / n
        tag = " (held out)" if fruit not in TRAIN_FRUITS else ""
        print(f"  {fruit:10s} {rate:5.2f}{tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--groups", type=int, default=2)
    ap.add_argument("--group-size", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--kl", type=float, default=0.02)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--eval", nargs="?", const="", default=None,
                    help="greedy eval; optional checkpoint dir (default: --model)")
    ap.add_argument("--check-mask", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "mps" else torch.float32
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    load_from = args.eval if args.eval else args.model
    model = AutoModelForCausalLM.from_pretrained(load_from, dtype=dtype).to(device)
    # Kill Qwen's default sampler settings (temp 0.6, top_p 0.95, top_k 20):
    # we must sample from the SAME distribution we later compute logprobs of,
    # or the policy gradient is for a policy we never actually ran.
    model.generation_config.temperature, model.generation_config.top_p, \
        model.generation_config.top_k = 1.0, 1.0, 0
    seg = derive_segments(tok)

    if args.check_mask:
        return check_mask(model, tok, seg, device)
    if args.eval is not None:
        return evaluate(model, tok, seg, device)

    ref = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    ref.requires_grad_(False)
    ref.eval()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    os.makedirs("runs", exist_ok=True)
    rng, fruit_i = random.Random(args.seed), 0

    for step in range(1, args.steps + 1):
        t0, episodes = time.time(), []
        for g in range(args.groups):  # one (fruit, seed) per group; cycle fruits
            fruit = TRAIN_FRUITS[fruit_i % len(TRAIN_FRUITS)]
            fruit_i += 1
            seed = rng.randrange(1_000_000)
            episodes += [new_episode(tok, seg, fruit, seed, g)
                         for _ in range(args.group_size)]
        gstats = rollout(model, tok, seg, episodes, device)
        stats = grpo_step(model, ref, opt, tok, episodes, args.kl, device)

        by_fruit = {}
        for e in episodes:
            by_fruit.setdefault(e["fruit"], []).append(e["reward"])
        rec = {"step": step,
               "reward": sum(e["reward"] for e in episodes) / len(episodes),
               "reward_by_fruit": {f: sum(rs) / len(rs) for f, rs in by_fruit.items()},
               **stats,
               "gen_tokens_per_ep": gstats["gen_tokens"] / len(episodes),
               "gen_tok_per_s": gstats["gen_tokens"] / gstats["gen_time"],
               "step_seconds": time.time() - t0,
               "mem_gb": torch.mps.driver_allocated_memory() / 2**30
                         if device == "mps" else 0.0}
        print(f"step {step:4d} | reward {rec['reward']:.3f} | rstd {rec['reward_group_std']:.2f} | "
              f"ent {rec['entropy_mc']:.2f} | gnorm {rec['grad_norm']:.2f} | "
              f"kl {rec['kl']:.4f} | zero-var {rec['frac_zero_var_groups']:.2f} | "
              f"gen tok/ep {rec['gen_tokens_per_ep']:.0f} | mem {rec['mem_gb']:.1f}G | "
              f"{rec['step_seconds']:.1f}s")
        with open("runs/log.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        if step % 10 == 0:  # eyeball a full transcript now and then
            print("--- sample transcript " + "-" * 40)
            print(tok.decode(episodes[0]["ids"]))
            print(f"--- reward {episodes[0]['reward']} " + "-" * 40)
        if step % 20 == 0:
            ckpt = f"runs/ckpt-{step:04d}"  # keep every checkpoint: the erosion
            model.save_pretrained(ckpt)     # curve needs the whole trajectory
            tok.save_pretrained(ckpt)
        if device == "mps":  # release cached buffers the bucketing didn't dedupe
            gc.collect()
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
