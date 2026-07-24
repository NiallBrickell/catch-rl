"""Experiment 3 environment: catch with name-keyed VALENCE.

Experiment 2 asks whether a reward-taught *dynamics* binding routes through
the lexicon. Experiment 3 moves the variable from physics to utility — the
child's version of the problem: a bee arcs through the air much like a
grape, but you want exactly one of them in your hand. Here every object
falls identically (straight down, fully observable), so tracking skill is
value-neutral; the ONLY thing the name determines is whether catching pays
+1 (treat cluster) or -1 (sting cluster). Missing pays 0 either way.

The box logic (ingredient 3, valence form): any name-blind policy behaves
identically on both clusters, so its expected reward on a balanced word mix
is exactly 0 — always-catch earns (+1-1)/2, always-dodge earns 0, and every
mixture sits in between. A name-reading policy catches treats and dodges
stings for 0.5. The gap is the experiment. Note the learning asymmetry:
avoid-bindings only receive gradient when the policy actually catches a
sting word (negative advantage needs a catch to punish), so early training
teaches catching before it can teach discrimination — the onset lag of
avoidance ("how many stings does it take?") is itself a measurement.

Valence is assigned per CLUSTER and counterbalanced across runs: in the
congruent assignment the sting-cluster words hurt (as the corpus already
says); in the incongruent one the grapes sting and the wasps are treats.
Text priors about valence are the strongest priors language has — the
incongruent arm is the harshest prior-vs-reward fight this project can
stage. Word lists pend embedding validation (see analysis/).

Acceptance test (run this file): best name-blind scripted policy <= 0.05
mean reward; name-aware oracle >= 0.45.
"""

import random

WIDTH = 7    # columns 0..6
BOTTOM = 5   # the basket's row; the object lands when it reaches this row
TURNS = 5    # decision turns per episode (object falls rows 0 -> 5)

# Provisional clusters (final lists pend the Exp 3 embedding validation).
TREAT = ["grape", "cherry", "melon"]
STING = ["wasp", "hornet", "bee"]
NONCE = ["zellik", "vantrix"]  # validated runners-up from the Exp 2 audit


def make_valence_map(plus_cluster, minus_cluster, nonce_valence=None):
    """One counterbalanced assignment: catching a `plus_cluster` word pays
    +1, a `minus_cluster` word pays -1. Flip the arguments for the matched
    reversed run."""
    m = {w: +1.0 for w in plus_cluster}
    m.update({w: -1.0 for w in minus_cluster})
    m.update(nonce_valence or {})
    return m


def _clamp(col):
    return max(0, min(WIDTH - 1, col))


class Catch3Env:
    """One episode. Same text interface and (word, seed) group contract as
    the other envs. Objects fall straight and observably; the reward at
    landing is valence_map[word] if caught, else 0."""

    def __init__(self, word, seed, valence_map):
        assert word in valence_map, f"word {word!r} not in this assignment"
        self.word = word
        self.seed = seed
        self.valence = valence_map[word]

    def reset(self):
        rng = random.Random(self.seed)
        self.obj_col = rng.randrange(WIDTH)
        # Basket within 3 of the object: catching is always feasible in the
        # 5 available moves, and dodging is always feasible in one.
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
        if self.obj_row == BOTTOM:
            self.done = True
            caught = self.obj_col == self.basket_col
            # Landing text stays valence-neutral: the reward carries the
            # valence, and this text never conditions any generated token
            # anyway (it lands after the last action).
            text = (f"The {self.word} landed at column {self.obj_col}. "
                    f"Your basket is at column {self.basket_col}. "
                    + ("You caught it." if caught else "You missed it."))
            return text, (self.valence if caught else 0.0), True
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


def _dodge_target(e):
    # step one column to whichever side has room; never equals obj_col
    return e.obj_col + 1 if e.obj_col < WIDTH - 1 else e.obj_col - 1


if __name__ == "__main__":
    # Acceptance test. Name-blind candidates behave identically on every
    # word: always-catch and always-dodge bracket the name-blind family.
    # Oracle: chase treats, sidestep stings.
    words = TREAT + STING
    vmap = make_valence_map(TREAT, STING)
    N = 500
    def run(policy):
        tot = 0.0
        for w in words:
            tot += sum(_scripted(Catch3Env(w, s, vmap), policy)
                       for s in range(N)) / N / len(words)
        return tot
    catch_all = run(lambda e: e.obj_col)
    dodge_all = run(_dodge_target)
    oracle = run(lambda e: e.obj_col if e.valence > 0 else _dodge_target(e))
    blind_best = max(catch_all, dodge_all)
    print(f"name-blind always-catch: {catch_all:+.3f}")
    print(f"name-blind always-dodge: {dodge_all:+.3f}")
    print(f"name-aware oracle:       {oracle:+.3f}")
    ok = blind_best <= 0.05 and oracle >= 0.45
    print("ACCEPTANCE:", "PASS" if ok else "FAIL",
          "(need best name-blind <= 0.05, oracle >= 0.45)")
    if not ok:
        raise SystemExit(1)
