"""Name-conditioning diagnostic: does the policy act on the WORD before the
world has shown it anything?

On turn 1 the fruit has not moved yet -- drift is unobservable -- so any
systematic difference in the action distribution across fruit NAMES (at
identical positions) is name-conditioned anticipation, not in-episode
inference. A policy that preemptively leans RIGHT for orange/banana/blorple
and LEFT for apple/quorf is using the name; a purely reactive policy shows
the same turn-1 distribution for every name.

Usage:
  uv run python analysis/diag_name_conditioning.py MODEL [MODEL ...]
  # MODEL is a HF id or a checkpoint dir; each gets its own table.
"""

import sys, collections

import torch

sys.path.insert(0, ".")  # run from the repo root
from catch_env import CatchEnv, EVAL_FRUITS, BOTTOM, TURNS
from train import SYSTEM_PROMPT, derive_segments, parse_action

from transformers import AutoModelForCausalLM, AutoTokenizer

K = 24            # sampled turn-1 completions per (fruit, position)
POSITIONS = [(1, 3), (3, 3), (5, 3)]  # (fruit_col, basket_col): left/center/right of basket
MAX_NEW = 80


def turn1_prompt(tok, seg, fruit, fruit_col, basket_col):
    env = CatchEnv(fruit, seed=0)
    env.reset()
    env.fruit_col, env.basket_col = fruit_col, basket_col  # fixed, not seeded
    obs = env._obs()
    text = (seg["sys_pre"] + SYSTEM_PROMPT + seg["turn_end"]
            + seg["user_pre"] + obs + seg["turn_end"] + seg["gen_pre"])
    return tok(text, add_special_tokens=False)["input_ids"]


def main():
    models = sys.argv[1:] or ["Qwen/Qwen3-0.6B"]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "mps" else torch.float32
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    seg = derive_segments(tok)

    for name in models:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device)
        model.generation_config.temperature, model.generation_config.top_p, \
            model.generation_config.top_k = 1.0, 1.0, 0
        torch.manual_seed(0)
        print(f"\n===== {name} =====")
        print(f"turn-1 action distribution, K={K} samples x {len(POSITIONS)} positions "
              f"(fruit has NOT moved yet; drift is unobservable)")
        print(f"{'fruit':11s} {'LEFT':>6s} {'STAY':>6s} {'RIGHT':>6s}   P(RIGHT)-P(LEFT)")
        for fruit in EVAL_FRUITS:
            counts = collections.Counter()
            for fcol, bcol in POSITIONS:
                ids = turn1_prompt(tok, seg, fruit, fcol, bcol)
                batch = torch.tensor([ids] * K, device=device)
                attn = torch.ones_like(batch)
                out = model.generate(input_ids=batch, attention_mask=attn,
                                     max_new_tokens=MAX_NEW, do_sample=True,
                                     pad_token_id=tok.pad_token_id)
                for row in out[:, len(ids):]:
                    counts[parse_action(tok.decode(row, skip_special_tokens=True))] += 1
            n = sum(counts.values())
            lean = (counts["RIGHT"] - counts["LEFT"]) / n
            print(f"{fruit:11s} {counts['LEFT']/n:6.2f} {counts['STAY']/n:6.2f} "
                  f"{counts['RIGHT']/n:6.2f}   {lean:+.2f}")
        del model
        if device == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
