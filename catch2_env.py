"""Experiment 2 environment: catch with a name-keyed landing shift.

Why v2 exists: run 3 showed the v1 env is reactively solvable — drift is
observable turn-to-turn and correctable within the horizon, so RL learned a
name-blind tracker and threw the language away. v2 moves ALL of the dynamics
into the unobservable final fall: the object drops straight through every
observed row, then, after the agent's LAST action, lands shifted sideways by
an amount keyed to its NAME. In-episode evidence about the shift is zero by
construction, so the name is the only channel — the reward is unreachable
name-blind. That is the third design ingredient (see README): a policy that
stands under the object catches only the straight-falling cluster (~0.5 on a
balanced word mix); a policy that reads the name clears 0.9.

The shift magnitude is 2 columns so a hedging policy (stand one column off)
catches nothing, and sloppy tracking cannot luck into shifted landings.

Which cluster of words gets the shift is an ASSIGNMENT, passed in as a
mapping — the counterbalancing lever. In half the training runs the "light"
cluster gets the shift, in the other half the "heavy" one does. If held-out
cluster members inherit whatever their trained sibling was taught IN THAT
RUN, and the inheritance flips with the assignment, only the trained binding
travelling through pretrained similarity can explain it.

Acceptance test (run this file): scripted reactive ceiling ≤ 0.5 and
scripted name-aware oracle ≥ 0.9, on the same spawns. If that gap closes,
the env is broken and training on it would just repeat run 3.
"""

import random

WIDTH = 7    # columns 0..6
BOTTOM = 5   # the basket's row; the object lands when it reaches this row
TURNS = 5    # decision turns per episode (object falls rows 0 -> 5)
SHIFT = 2    # landing shift, applied after the final action

# Provisional clusters (final word lists pend the embedding-validation
# report): pair 1 is a heavy/light axis, pair 2 fast/slow. Per pair, train
# on ONE member of each cluster; siblings are held out for the transfer
# probe, and the nonce words floor what name-free skill achieves.
PAIR1 = (["rock", "anvil", "brick"], ["feather", "leaf", "petal"])
PAIR2 = (["rocket", "bullet", "arrow"], ["snail", "slug", "tortoise"])
NONCE = ["zeph", "grolt"]


def make_shift_map(zero_cluster, shift_cluster, nonce_shifts=None):
    """One counterbalanced assignment: every word in `shift_cluster` lands
    SHIFT columns right of where it fell; `zero_cluster` lands true. Flip
    the two arguments to build the matched reversed run."""
    m = {w: 0 for w in zero_cluster}
    m.update({w: SHIFT for w in shift_cluster})
    m.update(nonce_shifts or {})
    return m


def _clamp(col):
    return max(0, min(WIDTH - 1, col))


class Catch2Env:
    """One episode. Same text interface as v1's CatchEnv (train.py needs no
    structural changes), same (word, seed) group-baseline contract. The
    object's column never changes while observable; the shift lands with it."""

    def __init__(self, word, seed, shift_map):
        assert word in shift_map, f"word {word!r} not in this assignment"
        self.word = word
        self.seed = seed
        self.shift = shift_map[word]

    def reset(self):
        rng = random.Random(self.seed)
        # Spawn so every landing hypothesis is honest and reachable: the
        # object's column keeps col+SHIFT on the grid, and the basket starts
        # within 3 of the object, so both col and col+SHIFT are inside the
        # 5 moves the episode allows. Without this, feasibility would differ
        # between clusters and confound the reward gap.
        self.obj_col = rng.randrange(0, WIDTH - SHIFT)
        self.basket_col = rng.randrange(max(0, self.obj_col - 3),
                                        min(WIDTH - 1, self.obj_col + 3) + 1)
        self.obj_row = 0
        self.turn = 1
        self.done = False
        return self._obs()

    def _obs(self):
        return (f"Turn {self.turn} of {TURNS}. "
                f"The {self.word} is at column {self.obj_col}, row {self.obj_row}. "
                f"Your basket is at column {self.basket_col}, row {BOTTOM}.")

    def step(self, action):
        """action in {"LEFT", "RIGHT", "STAY"} -> (obs_text, reward, done)."""
        assert not self.done, "episode is over; call reset()"
        self.basket_col = _clamp(self.basket_col + {"LEFT": -1, "RIGHT": +1, "STAY": 0}[action])
        self.obj_row += 1
        if self.obj_row == BOTTOM:  # the final fall: the shift happens here,
            self.obj_col += self.shift  # after the last action, unobserved
            self.done = True
            caught = self.obj_col == self.basket_col
            text = (f"The {self.word} landed at column {self.obj_col}. "
                    f"Your basket is at column {self.basket_col}. "
                    + ("You caught it!" if caught else "You missed it."))
            return text, (1.0 if caught else 0.0), True
        self.turn += 1
        return self._obs(), 0.0, False


def _scripted(env, target_fn):
    env.reset()
    done, r = False, 0.0
    while not done:
        target = target_fn(env)
        a = "LEFT" if target < env.basket_col else "RIGHT" if target > env.basket_col else "STAY"
        _, r, done = env.step(a)
    return r


if __name__ == "__main__":
    # Acceptance test — MUST pass before anything trains on this env.
    # Reactive = chase the observed column (the policy run 3 converged to).
    # Oracle = chase observed column + the word's true shift.
    words = PAIR1[0] + PAIR1[1]
    shift_map = make_shift_map(PAIR1[0], PAIR1[1])
    N = 500
    print(f"{'word':10s} {'reactive':>9s} {'oracle':>9s}")
    tot_r = tot_o = 0.0
    for w in words:
        r = sum(_scripted(Catch2Env(w, s, shift_map), lambda e: e.obj_col)
                for s in range(N)) / N
        o = sum(_scripted(Catch2Env(w, s, shift_map), lambda e: e.obj_col + e.shift)
                for s in range(N)) / N
        tot_r += r / len(words)
        tot_o += o / len(words)
        print(f"{w:10s} {r:9.3f} {o:9.3f}")
    print(f"{'OVERALL':10s} {tot_r:9.3f} {tot_o:9.3f}")
    ok = tot_r <= 0.5 and tot_o >= 0.9
    print("ACCEPTANCE:", "PASS" if ok else "FAIL",
          f"(need reactive <= 0.5, oracle >= 0.9)")
