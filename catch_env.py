"""A falling-fruit catch game with plain-text observations.

The grid has 7 columns (0-6). A fruit starts at row 0 and falls one row after
each action; the basket lives at row 5, so an episode is exactly 5 decision
turns. Some fruits drift sideways as they fall: drift is applied AFTER the
fall on even-numbered turns (turns 2 and 4). That makes the dynamics
learnable-but-not-trivial: a policy that just chases the fruit's current
column is good, but a policy that understands "apples drift left" is better.

"banana" is held out of training. It has the same dynamics as "orange" but a
novel name, so evaluating on it tests whether the model learned "this WORD
means drift +1" versus a transferable "watch how the column changes" habit.
"""

import random

WIDTH = 7    # columns 0..6
BOTTOM = 5   # the basket's row; the fruit lands when it reaches this row
TURNS = 5    # decision turns per episode (fruit falls rows 0 -> 5)

DRIFT = {"strawberry": 0, "apple": -1, "orange": +1, "banana": +1}
TRAIN_FRUITS = ["strawberry", "apple", "orange"]
EVAL_FRUITS = TRAIN_FRUITS + ["banana"]


def _clamp(col):
    return max(0, min(WIDTH - 1, col))


class CatchEnv:
    """One episode of catch. The seed fixes the starting fruit/basket columns,
    so a group of rollouts built with the same (fruit, seed) shares an
    identical initial state -- which is what makes a GRPO group's mean reward
    a fair baseline for every member of the group."""

    def __init__(self, fruit_name, seed):
        assert fruit_name in DRIFT, f"unknown fruit {fruit_name!r}"
        self.fruit = fruit_name
        self.seed = seed

    def reset(self):
        rng = random.Random(self.seed)
        self.fruit_col = rng.randrange(WIDTH)
        self.basket_col = rng.randrange(WIDTH)
        self.fruit_row = 0
        self.turn = 1
        self.done = False
        return self._obs()

    def _obs(self):
        return (f"Turn {self.turn} of {TURNS}. "
                f"The {self.fruit} is at column {self.fruit_col}, row {self.fruit_row}. "
                f"Your basket is at column {self.basket_col}, row {BOTTOM}.")

    def step(self, action):
        """action in {"LEFT", "RIGHT", "STAY"} -> (obs_text, reward, done)."""
        assert not self.done, "episode is over; call reset()"
        self.basket_col = _clamp(self.basket_col + {"LEFT": -1, "RIGHT": +1, "STAY": 0}[action])
        self.fruit_row += 1
        if self.turn % 2 == 0:  # drift kicks in after the fall on turns 2 and 4
            self.fruit_col = _clamp(self.fruit_col + DRIFT[self.fruit])
        if self.fruit_row == BOTTOM:
            self.done = True
            caught = self.fruit_col == self.basket_col
            text = (f"The {self.fruit} landed at column {self.fruit_col}. "
                    f"Your basket is at column {self.basket_col}. "
                    + ("You caught it!" if caught else "You missed it."))
            return text, (1.0 if caught else 0.0), True
        self.turn += 1
        return self._obs(), 0.0, False


if __name__ == "__main__":
    # Self-test: a scripted greedy policy (step toward the fruit's CURRENT
    # column) over 200 random starts per train fruit. Greedy should be strong
    # but imperfect on drifting fruits -- the fruit can walk away from a
    # basket that is chasing where it was, not where it is going. That gap is
    # the headroom that reasoning about drift can close.
    for fruit in TRAIN_FRUITS:
        caught = 0.0
        for seed in range(200):
            env = CatchEnv(fruit, seed)
            env.reset()
            done = False
            while not done:
                if env.fruit_col < env.basket_col:
                    a = "LEFT"
                elif env.fruit_col > env.basket_col:
                    a = "RIGHT"
                else:
                    a = "STAY"
                _, r, done = env.step(a)
            caught += r
        print(f"greedy catch rate on {fruit:10s}: {caught / 200:.3f}")
