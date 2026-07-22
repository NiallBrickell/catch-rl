import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "font.size": 10.5,
            "axes.titlesize": 11,
        }
    )

    # Validated categorical palette (fixed slot order 1..5), plus ink tones.
    C_BLUE, C_ORANGE, C_AQUA, C_YELLOW, C_MAGENTA = (
        "#2a78d6",
        "#eb6834",
        "#1baf7a",
        "#eda100",
        "#e87ba4",
    )
    C_INK, C_MUTED, C_FAINT = "#0b0b0b", "#52514e", "#c9c8c0"

    # Bandit constants (sections 1-2).
    BANDIT_MEANS = np.array([0.2, 0.5, 0.8])
    SHIFTED_MEANS = np.array([0.85, 0.90, 0.95])

    # Catch env constants (section 7) -- mirror of catch_env.py.
    CATCH_WIDTH = 7
    CATCH_TURNS = 5
    CATCH_FRUITS = ["strawberry", "apple", "orange"]
    CATCH_DRIFT = np.array([0, -1, +1])
    FRUIT_COLORS = [C_BLUE, C_ORANGE, C_AQUA]


@app.cell
def _():
    mo.md(r"""
    # Policy gradients → GRPO, from first principles

    These are working notes for `train.py` in this repo: a hand-rolled multi-turn GRPO
    loop that teaches Qwen3-0.6B to catch falling fruit in a text world. The claim of
    this notebook is that every line of that loop is forced by about four pieces of
    math, and that each piece is small enough to *play with*. Every section below is a
    derivation followed by a live demo — the sliders re-run everything downstream of
    them, so drag things and watch.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## 1 · The problem: maximize \( \mathbb{E}[R] \) when \( R \) is a black box

    We have a policy \( \pi_\theta \) that emits trajectories \( \tau \) (for us: five turns
    of chain-of-thought and `ACTION:` lines), and a simulator that scores each one with a
    reward \( R(\tau) \). The objective is

    \[ J(\theta) \;=\; \mathbb{E}_{\tau \sim \pi_\theta}\!\left[ R(\tau) \right]. \]

    The catch — pun fully intended — is that \( R \) is *code*. The fruit lands in the basket
    or it doesn't; there is no \( \partial R / \partial \theta \) to backprop through. And yet
    \( J \) genuinely depends on \( \theta \), because \( \theta \) shapes *which* trajectories
    get sampled. That dependence lives entirely in the distribution, so differentiate there:

    \[ \nabla_\theta J \;=\; \nabla_\theta \int \pi_\theta(\tau)\, R(\tau)\, d\tau
       \;=\; \int \nabla_\theta \pi_\theta(\tau)\, R(\tau)\, d\tau . \]

    \( R \) passed through untouched — it is a constant with respect to \( \theta \). Now the one
    genuinely clever step, the **log-derivative identity**
    \( \nabla_\theta \pi = \pi \, \nabla_\theta \log \pi \) (just the chain rule on \( \log \), read
    right-to-left). Substituting it manufactures a \( \pi_\theta \) out of thin air, and a
    \( \pi_\theta \) inside an integral is an expectation:

    \[ \nabla_\theta J \;=\; \int \pi_\theta(\tau)\, \nabla_\theta \log \pi_\theta(\tau)\, R(\tau)\, d\tau
       \;=\; \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\, R(\tau)\, \nabla_\theta \log \pi_\theta(\tau) \,\right]. \]

    That is the whole trick, and it is worth staring at. The gradient of a
    **non-differentiable** objective became a **sampleable expectation**: roll out
    trajectories, compute \( \nabla_\theta \log \pi_\theta(\tau) \) (which we *can* backprop —
    it's just the log-probability our own network assigned to what it did), and weight each one
    by its reward. \( R \) is never differentiated; it only ever appears as a **multiplier**
    telling the log-likelihood gradient how hard to push, and in which direction. REINFORCE is
    exactly this, one sample at a time. Here it is for a 3-armed bandit with a softmax policy
    over logits \( \theta \in \mathbb{R}^3 \):

    ```python
    probs = np.exp(theta) / np.exp(theta).sum()      # π_θ
    a     = rng.choice(3, p=probs)                   # sample an action
    r     = rng.normal(means[a], noise)              # black-box reward
    grad_logpi = np.eye(3)[a] - probs                # ∇_θ log π_θ(a)  (softmax identity)
    theta += lr * r * grad_logpi                     # ascend E[R ∇log π]
    ```

    Arm rewards below are \( 0.2, 0.5, 0.8 \) (plus noise). Watch probability mass migrate to
    the best arm as you grant it more updates.
    """)
    return


@app.function
def bandit_train(means, steps, lr, baseline=0.0, noise=0.05, seed=0):
    """REINFORCE on a softmax bandit. Returns (steps+1, 3) array of arm probs."""
    rng = np.random.default_rng(seed)
    theta = np.zeros(3)
    hist = np.empty((steps + 1, 3))
    for t in range(steps + 1):
        z = np.exp(theta - theta.max())
        probs = z / z.sum()
        hist[t] = probs
        if t == steps:
            break
        a = rng.choice(3, p=probs)
        r = rng.normal(means[a], noise)
        theta += lr * (r - baseline) * (np.eye(3)[a] - probs)
    return hist


@app.cell
def _():
    s1_steps = mo.ui.slider(0, 500, step=10, value=150, label="REINFORCE updates")
    s1_lr = mo.ui.slider(0.05, 1.0, step=0.05, value=0.3, label="learning rate")
    mo.hstack([s1_steps, s1_lr], justify="start", gap=2)
    return s1_lr, s1_steps


@app.cell
def _(s1_lr, s1_steps):
    plt.close("all")
    _hist = bandit_train(BANDIT_MEANS, s1_steps.value, s1_lr.value, noise=0.1, seed=0)
    _fig, _ax = plt.subplots(figsize=(8.5, 3.4), layout="constrained")
    for _i, _c in enumerate([C_BLUE, C_ORANGE, C_AQUA]):
        _ax.plot(_hist[:, _i], color=_c, lw=2, label=f"arm {_i}  (r̄={BANDIT_MEANS[_i]:.1f})")
        _ax.annotate(
            f"arm {_i}",
            (len(_hist) - 1, _hist[-1, _i]),
            textcoords="offset points",
            xytext=(6, 0),
            color=_c,
            fontsize=9,
        )
    _ax.set(
        xlabel="update",
        ylabel="π(arm)",
        ylim=(0, 1),
        title=f"softmax policy under REINFORCE — after {s1_steps.value} updates, "
        f"π(best) = {_hist[-1, 2]:.2f}",
    )
    _ax.legend(loc="center left", frameon=False)
    _fig
    return


@app.cell
def _():
    mo.md(r"""
    ## 2 · Variance, and why baselines are free

    The estimator \( R \, \nabla \log \pi \) is unbiased but can be atrociously noisy, and the
    classic villain is *uniformly shifted rewards*. Suppose every trajectory earns between 0.8
    and 1.0. Then every sample says "push this one up" — the signal that arm A beats arm B is a
    whisper riding on a shout. The fix exploits a lovely identity: for any constant \( b \)
    (more generally, anything not depending on the action),

    \[ \mathbb{E}\!\left[ b \, \nabla_\theta \log \pi_\theta(\tau) \right]
       \;=\; b \int \pi_\theta \nabla_\theta \log \pi_\theta \, d\tau
       \;=\; b \int \nabla_\theta \pi_\theta \, d\tau
       \;=\; b \, \nabla_\theta \!\! \int \pi_\theta \, d\tau
       \;=\; b \, \nabla_\theta 1 \;=\; 0 . \]

    Probabilities always sum to one, so the gradient of their sum is zero — which means we can
    subtract **any** action-independent baseline from the reward without biasing the gradient.
    (Hold on to that proviso — *action-independent* — because section 3 nearly trips over it.)
    It changes only the *variance* of the estimate. The centered quantity
    \( A = R - b \) is the **advantage**: not "was this good?" but "was this **better than my
    usual**?" — and crucially its *sign* now drives the direction of the update. Score above the
    baseline, push the trajectory up; below, push it down.

    The demo uses the shifted bandit (arm means \( 0.85, 0.90, 0.95 \)). Left: the distribution
    of single-sample gradient estimates for a frozen policy, as you move \( b \). The dashed
    line is the true expected gradient — it does **not move**, no matter what you do to the
    slider; only the spread around it breathes. Right: the standard deviation as a function of
    \( b \), minimized almost exactly at \( b = \mathbb{E}[R] = 0.9 \).
    """)
    return


@app.cell
def _():
    s2_b = mo.ui.slider(0.0, 1.2, step=0.02, value=0.0, label="baseline b")
    s2_b
    return (s2_b,)


@app.cell
def _(s2_b):
    plt.close("all")
    _rng = np.random.default_rng(7)
    _N = 4000
    _a = _rng.integers(0, 3, _N)
    _r = SHIFTED_MEANS[_a] + _rng.normal(0, 0.02, _N)
    _score = (_a == 2).astype(float) - 1 / 3  # ∂ log π / ∂ θ_2 at uniform π
    _g = (_r - s2_b.value) * _score
    _true_mean = (1 / 3) * (SHIFTED_MEANS[2] - SHIFTED_MEANS.mean())

    _fig, (_axh, _axs) = plt.subplots(
        1, 2, figsize=(9.5, 3.4), layout="constrained", width_ratios=[1.4, 1]
    )
    _axh.hist(_g, bins=80, range=(-1.2, 1.2), color=C_BLUE, edgecolor="white", lw=0.3)
    _axh.axvline(
        _true_mean, color=C_INK, ls="--", lw=1.5, label=f"true E[g] = {_true_mean:.3f}  (fixed)"
    )
    _axh.set(
        xlim=(-1.2, 1.2),
        xlabel="single-sample estimate of ∂J/∂θ₂",
        ylabel="count",
        title=f"b = {s2_b.value:.2f}:  mean {_g.mean():+.3f},  std {_g.std():.3f}",
    )
    _axh.legend(frameon=False, loc="upper left", fontsize=9)

    _bs = np.linspace(0, 1.2, 61)
    _stds = [((_r - _bb) * _score).std() for _bb in _bs]
    _axs.plot(_bs, _stds, color=C_BLUE, lw=2)
    _axs.scatter([s2_b.value], [_g.std()], color=C_BLUE, zorder=5, s=45)
    _axs.axvline(SHIFTED_MEANS.mean(), color=C_MUTED, ls=":", lw=1.2)
    _axs.annotate("E[R]", (SHIFTED_MEANS.mean(), max(_stds) * 0.95), color=C_MUTED, fontsize=9)
    _axs.set(xlabel="baseline b", ylabel="std of estimate", title="variance vs baseline")
    _fig
    return


@app.cell
def _(s2_b):
    plt.close("all")
    _runs0, _runsb = [], []
    for _s in range(8):
        _runs0.append(bandit_train(SHIFTED_MEANS, 400, 0.35, 0.0, 0.03, seed=_s)[:, 2])
        _runsb.append(bandit_train(SHIFTED_MEANS, 400, 0.35, s2_b.value, 0.03, seed=_s)[:, 2])
    _runs0, _runsb = np.array(_runs0), np.array(_runsb)

    _fig, _ax = plt.subplots(figsize=(8.5, 3.4), layout="constrained")
    for _run in _runs0:
        _ax.plot(_run, color=C_ORANGE, alpha=0.18, lw=0.8)
    for _run in _runsb:
        _ax.plot(_run, color=C_BLUE, alpha=0.18, lw=0.8)
    _ax.plot(_runs0.mean(0), color=C_ORANGE, lw=2.4, label="b = 0 (raw reward)")
    _ax.plot(_runsb.mean(0), color=C_BLUE, lw=2.4, label=f"b = {s2_b.value:.2f}")
    _ax.set(
        xlabel="update",
        ylabel="π(best arm)",
        ylim=(0, 1),
        title="same sample budget, 8 seeds each — only the baseline differs",
    )
    _ax.legend(frameon=False, loc="upper left")
    _fig
    return


@app.cell
def _():
    mo.md(r"""
    ## 3 · GRPO: the critic replaced by sampling

    So the baseline should approximate "how well do I usually do *from this state*" —
    \( b(s) \approx \mathbb{E}[R \mid s] \). PPO's answer is to *learn* it: a whole second
    network (the critic / value model) trained alongside the policy, with its own loss, its
    own memory footprint, its own failure modes. GRPO's answer is almost insolently simpler:
    **just measure it**. Roll out the *same* state \( G \) times and use the group's mean
    reward as the baseline:

    \[ A_i \;=\; r_i \;-\; \frac{1}{G} \sum_{j=1}^{G} r_j . \]

    Caught it in 4 of 6 tries → the successes get \( A = +\tfrac{1}{3} \), the failures
    \( -\tfrac{2}{3} \). No critic, and the baseline is *measured at exactly the right state* —
    which is precisely why the \( G \) rollouts **must share an identical initial state** (same
    fruit, same starting columns; in the repo, same seed). Mix states within a group and the
    baseline conflates "this was an easy episode" with "the policy played well": a mediocre
    rollout on an easy start would steal advantage from a brilliant rollout on a hard one.

    **Fine print (a real 7/8).** Section 2's proof required \( b \) to be independent of the
    sample it multiplies — and the plain group mean is not, because \( \bar r \) contains
    \( r_i \) itself. One line of algebra shows what the self-contamination costs:

    \[ r_i - \bar r \;=\; \tfrac{G-1}{G} \left( r_i - \bar r_{-i} \right),
       \qquad \bar r_{-i} = \tfrac{1}{G-1} \textstyle\sum_{j \neq i} r_j . \]

    So the naive version doesn't bias the gradient's *direction* — it silently rescales the
    expected gradient by \( (G-1)/G \) (at \( G = 8 \), a harmless-but-real 7/8, absorbable
    into the learning rate but not "exact"). The clean fix is the **leave-one-out baseline**:
    \( A_i = r_i - \bar r_{-i} \), each rollout judged against its *siblings'* mean only.
    That baseline never sees rollout \( i \)'s own reward, so it is action-independent and
    section 2's proof applies verbatim. `train.py` and the capstone below both use it.

    With a binary reward either estimator has one sharp edge. If the whole group succeeds or
    the whole group fails, every \( A_i = 0 \) and the batch contributes **exactly zero
    gradient**. The probability of that, for success rate \( p \), is \( p^G + (1-p)^G \) —
    plotted below.
    """)
    return


@app.cell
def _():
    s3_G = mo.ui.slider(2, 32, step=1, value=8, label="group size G")
    s3_p = mo.ui.slider(0.0, 1.0, step=0.02, value=0.25, label="success prob p")
    s3_resample = mo.ui.button(
        value=0, on_click=lambda value: value + 1, label="resample group"
    )
    mo.hstack([s3_G, s3_p, s3_resample], justify="start", gap=2)
    return s3_G, s3_p, s3_resample


@app.cell
def _(s3_G, s3_p, s3_resample):
    plt.close("all")
    _rng = np.random.default_rng(1000 + s3_resample.value)
    _G = s3_G.value
    _r = (_rng.random(_G) < s3_p.value).astype(float)
    _A = _r - (_r.sum() - _r) / (_G - 1)  # leave-one-out advantage

    _fig, (_axl, _axr) = plt.subplots(
        1, 2, figsize=(9.5, 3.5), layout="constrained", width_ratios=[1, 1.25]
    )
    _cols = [C_BLUE if _ri > 0 else C_ORANGE for _ri in _r]
    _axl.bar(np.arange(_G), _A, color=_cols, width=0.72)
    _axl.axhline(0, color=C_INK, lw=1)
    _zero = np.all(_r == _r[0])
    _axl.set(
        xlabel="rollout i in group",
        ylabel="advantage Aᵢ (leave-one-out)",
        ylim=(-1.05, 1.05),
        title=(
            f"{int(_r.sum())}/{_G} caught → "
            + ("ZERO-VARIANCE GROUP: A ≡ 0, no gradient" if _zero else f"A ∈ {{{_A.min():+.2f}, {_A.max():+.2f}}}")
        ),
    )
    _axl.title.set_color(C_ORANGE if _zero else C_INK)

    _ps = np.linspace(0, 1, 201)
    _blues = plt.cm.Blues(np.linspace(0.35, 0.95, 5))
    for _g, _c in zip([2, 4, 8, 16, 32], _blues):
        _axr.plot(_ps, _ps**_g + (1 - _ps) ** _g, color=_c, lw=1.8, label=f"G = {_g}")
    _pz = s3_p.value**_G + (1 - s3_p.value) ** _G
    _axr.scatter([s3_p.value], [_pz], color=C_ORANGE, zorder=5, s=55)
    _axr.annotate(
        f"  you: G={_G}, P(dead group)={_pz:.2f}",
        (s3_p.value, _pz),
        color=C_ORANGE,
        fontsize=9,
    )
    _axr.set(
        xlabel="success prob p",
        ylabel="P(all-same group)",
        ylim=(0, 1.05),
        title="P(zero gradient) = p^G + (1−p)^G",
    )
    _axr.legend(frameon=False, fontsize=8.5)
    _fig
    return


@app.cell
def _():
    mo.callout(
        mo.md(
            r"""
    **This is THE practical failure mode of sparse-reward GRPO.** Early in training the policy
    almost always misses, so \( p \approx 0 \) and nearly every group is all-fail → advantage
    zero everywhere → *no learning signal at all*, forever, in expectation. The training loop
    runs, the loss prints, nothing moves. (`train.py` logs `frac_zero_var_groups` for exactly
    this reason — if it sits near 1.0, you are burning tokens.) The escape routes, in rough
    order of principle-preservation: a **bigger G** (drag the slider left-panel-wards and watch
    \( p^G + (1-p)^G \) fall), an easier **curriculum** (raise \( p \) itself), or **reward
    shaping** (partial credit that makes the reward non-binary — effective, but it dilutes the
    pure-outcome premise).
    """
        ),
        kind="warn",
    )
    return


@app.cell
def _():
    mo.md(r"""
    ### 3b · Dr.GRPO: why we do *not* divide by the group std

    The original GRPO paper normalizes the advantage: \( A_i = (r_i - \bar r) / \sigma_{\text{group}} \).
    It looks like innocent standardization, but the Dr.GRPO paper pointed out the bias it smuggles
    in: dividing by \( \sigma \) makes the gradient magnitude *independent of how decided the
    group was*. A group where rewards barely differ — advantage a hair above noise — gets its
    tiny signal inflated by \( 1/\sigma \) into a full-sized gradient. Near-deterministic
    groups (σ → 0) should contribute almost nothing, because there is almost nothing to learn
    from them; the division instead **manufactures huge updates out of noise**, and biases
    training toward exactly the states where the policy has already made up its mind. Drag σ
    toward zero and watch the two curves disagree about physics.
    """)
    return


@app.cell
def _():
    s3_sigma = mo.ui.slider(0.01, 0.5, step=0.01, value=0.1, label="group reward std σ")
    s3_sigma
    return (s3_sigma,)


@app.cell
def _(s3_sigma):
    plt.close("all")
    _rng = np.random.default_rng(3)
    _z = _rng.normal(size=(400, 16))  # 400 groups of G=16, unit-scale noise
    _sigmas = np.logspace(-3, np.log10(0.5), 40)
    _raw, _norm = [], []
    for _sig in _sigmas:
        _r = 0.5 + _sig * _z
        _A = _r - _r.mean(1, keepdims=True)
        _raw.append(np.abs(_A).mean())
        _norm.append((np.abs(_A) / (_r.std(1, keepdims=True) + 1e-4)).mean())

    _rs = 0.5 + s3_sigma.value * _z
    _As = _rs - _rs.mean(1, keepdims=True)
    _raw_here = np.abs(_As).mean()
    _norm_here = (np.abs(_As) / (_rs.std(1, keepdims=True) + 1e-4)).mean()

    _fig, _ax = plt.subplots(figsize=(8.5, 3.5), layout="constrained")
    _ax.loglog(_sigmas, _raw, color=C_BLUE, lw=2.2, label="mean |A|  (no std division)")
    _ax.loglog(_sigmas, _norm, color=C_ORANGE, lw=2.2, label="mean |A/σ̂|  (GRPO's std-division)")
    _ax.axvline(s3_sigma.value, color=C_MUTED, ls=":", lw=1.2)
    _ax.set(
        xlabel="group reward std σ (log)",
        ylabel="gradient-weight magnitude (log)",
        title=f"at σ = {s3_sigma.value:.2f}:  |A| = {_raw_here:.3f}   vs   |A/σ̂| = {_norm_here:.3f}",
    )
    _ax.legend(frameon=False, loc="upper left")
    _fig
    return


@app.cell
def _():
    mo.md(r"""
    The blue curve is honest: signal shrinks as the group's outcomes converge, exactly as it
    should. The orange curve is flat — a group that is 99.9% decided gets the same gradient
    heft as a genuinely informative 50/50 group. This is why `train.py` uses
    **baseline-subtraction only**: \( A_i = r_i - \bar r_{-i} \), no denominator.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## 4 · From trajectories to tokens: why the mask is math, not bookkeeping

    For an LLM agent, a "trajectory" is a transcript: the model's tokens interleaved with the
    environment's. The probability of a full multi-turn episode factorizes as

    \[ \log P(\tau) \;=\; \underbrace{\sum_{t \,\in\, \text{model tokens}} \log \pi_\theta(y_t \mid y_{<t})}_{\text{depends on } \theta}
       \;+\; \underbrace{\sum_{t \,\in\, \text{observation tokens}} \log P_{\text{env}}(y_t \mid \cdot)}_{\text{no } \theta \text{ anywhere}} \]

    and the environment terms **vanish under** \( \nabla_\theta \). The generated-token mask in
    the training loop is not an implementation convenience — it *is* this derivative. The
    fruit's position is not a function of your weights, so no gradient may flow through the
    text that announces it. Below, the tokens whose log-probs belong in the loss are tinted
    orange (the model sampled them, including its own end-of-turn token); everything blue was
    written by the environment or the chat template. Flip the toggle to see what the loss
    becomes if you get the mask wrong.
    """)
    return


@app.cell
def _():
    s4_leak = mo.ui.switch(value=False, label="leak observation tokens into the loss")
    s4_leak
    return (s4_leak,)


@app.cell
def _(s4_leak):
    _env = (
        "background:rgba(42,120,214,0.14);border-radius:3px;padding:0 3px;"
        "box-decoration-break:clone;-webkit-box-decoration-break:clone;"
    )
    _gen = (
        "background:rgba(235,104,52,0.22);border-radius:3px;padding:0 3px;"
        "box-decoration-break:clone;-webkit-box-decoration-break:clone;"
    )

    def _e(txt):
        return f'<span style="{_env}">{txt}</span>'

    def _g(txt):
        return f'<span style="{_gen}">{txt}</span>'

    _lines = [
        _e("&lt;|im_start|&gt;system<br>You control a basket. Each turn reply with ACTION: LEFT, STAY or RIGHT.&lt;|im_end|&gt;"),
        _e("&lt;|im_start|&gt;user<br>Turn 1 of 5. The apple is at column 5, row 0. Your basket is at column 2, row 5.&lt;|im_end|&gt;"),
        _e("&lt;|im_start|&gt;assistant<br>") + _g("The apple is at 5 and I'm at 2 — three to the right, four turns of movement left. Apples drift left on even turns, so it will land nearer 3. Still: close the gap. ACTION: RIGHT&lt;|im_end|&gt;"),
        _e("&lt;|im_start|&gt;user<br>Turn 2 of 5. The apple is at column 5, row 1. Your basket is at column 3, row 5.&lt;|im_end|&gt;"),
        _e("&lt;|im_start|&gt;assistant<br>") + _g("Drift hits after this fall — expect it at 4, then 3 by landing. I'm at 3. Hold... no, one more step to be safe. ACTION: RIGHT&lt;|im_end|&gt;"),
    ]
    _transcript = mo.Html(
        '<div style="font-family:ui-monospace,SFMono-Regular,Menlo,monospace;'
        'font-size:0.82rem;line-height:2.0;max-width:46rem;">'
        + "<br>".join(_lines)
        + '<div style="margin-top:0.8rem;font-family:inherit;font-size:0.8rem;">'
        + f'<span style="{_gen}">model-generated → in the loss</span> &nbsp;&nbsp; '
        + f'<span style="{_env}">environment / template → masked out</span></div></div>'
    )

    if s4_leak.value:
        _loss = mo.callout(
            mo.md(
                r"""
    **Leaked (wrong):**
    \[ \mathcal{L} = -A \sum_{t \,\in\, \text{gen} \,\cup\, \text{obs}} \log \pi_\theta(y_t \mid y_{<t}) \]
    The observation terms are *not* zero — they are the model's probability of predicting the
    environment's text, and that very much depends on \( \theta \). You are now doing
    advantage-weighted language modeling of the simulator: successful episodes teach the model
    to *recite fruit positions more confidently*, failed episodes teach it to recite them less
    confidently. Neither is in \( \nabla J \). This is the #1 silent bug in hand-rolled agent
    RL, and why `train.py --check-mask` exists.
    """
            ),
            kind="danger",
        )
    else:
        _loss = mo.callout(
            mo.md(
                r"""
    **Masked (right):**
    \[ \mathcal{L} = -A \sum_{t \,\in\, \text{gen}} \log \pi_\theta(y_t \mid y_{<t}) \]
    Only tokens the model actually sampled carry gradient. The env terms dropped out on paper
    (\( \nabla_\theta \log P_{\text{env}} = 0 \)), so they must drop out in code.
    """
            ),
            kind="success",
        )
    mo.vstack([_transcript, _loss], gap=1)
    return


@app.cell
def _():
    mo.md(r"""
    ## 5 · Where PPO's clip went

    PPO's famous objective looks nothing like what we've built so far:

    \[ \mathcal{L}^{\text{clip}} = \mathbb{E}\big[ \min\!\big( \rho A,\; \text{clip}(\rho,\, 1-\varepsilon,\, 1+\varepsilon)\, A \big) \big],
       \qquad \rho = \frac{\pi_{\text{new}}(a \mid s)}{\pi_{\text{old}}(a \mid s)} . \]

    Both pieces exist to solve one problem: **reusing stale rollouts**. Sampling from an LLM is
    expensive, so the big systems take several gradient steps per batch of rollouts; after the
    first step the data was generated by a policy you no longer have. The importance ratio
    \( \rho \) reweights each sample to correct for that mismatch, and the clip caps the
    correction so a single sample whose probability exploded can't dominate the batch.

    Our loop takes **one fresh update per batch**. At the moment of that update,
    \( \pi_{\text{new}} = \pi_{\text{old}} \), so \( \rho \equiv 1 \), the clip brackets an
    interval that contains 1, the `min` is a tie — and the entire apparatus collapses to
    \( \mathbb{E}[A \log \pi] \)'s gradient: plain REINFORCE with a baseline. Most of the
    notational fog around GRPO evaporates once you see where you sit on this plot.
    """)
    return


@app.cell
def _():
    s5_eps = mo.ui.slider(0.05, 0.5, step=0.05, value=0.2, label="clip ε")
    s5_eps
    return (s5_eps,)


@app.cell
def _(s5_eps):
    plt.close("all")
    _rho = np.linspace(0.0, 2.5, 501)
    _eps = s5_eps.value
    _fig, _axes = plt.subplots(1, 2, figsize=(9.5, 3.5), layout="constrained", sharex=True)
    for _ax, _Aval, _c in zip(_axes, [+1.0, -1.0], [C_BLUE, C_ORANGE]):
        _clipped = np.minimum(_rho * _Aval, np.clip(_rho, 1 - _eps, 1 + _eps) * _Aval)
        _ax.plot(_rho, _rho * _Aval, color=C_FAINT, ls="--", lw=1.4, label="unclipped ρA")
        _ax.plot(_rho, _clipped, color=_c, lw=2.2, label="min(ρA, clip(ρ)A)")
        _ax.scatter([1.0], [_Aval], color=C_INK, zorder=5, s=45)
        _ax.annotate(
            "you are here\n(on-policy, ρ=1)",
            (1.0, _Aval),
            textcoords="offset points",
            xytext=(10, -24 if _Aval > 0 else 12),
            fontsize=8.5,
            color=C_INK,
        )
        _ax.axvspan(1 - _eps, 1 + _eps, color=C_FAINT, alpha=0.25, lw=0)
        _ax.set(xlabel="ρ = π_new / π_old", title=f"A = {_Aval:+.0f}", ylim=(-2.6, 2.6))
        _ax.legend(frameon=False, fontsize=8.5, loc="lower right" if _Aval > 0 else "upper right")
    _axes[0].set(ylabel="surrogate objective")
    _fig
    return


@app.cell
def _():
    mo.md(r"""
    ## 6 · The KL leash

    The full loss carries one more term: \( \beta \, \mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}}) \),
    a penalty tying the policy to a frozen copy of the model it started as. Two things deserve
    first-principles attention.

    **Why this direction of KL.** \( \mathrm{KL}(\pi_\theta \| \pi_{\text{ref}}) =
    \mathbb{E}_{x \sim \pi_\theta} \log \frac{\pi_\theta}{\pi_{\text{ref}}} \) is *reverse* KL —
    the expectation runs over the **policy's own samples**. It charges you for putting mass
    where the reference has little (leaving the language manifold: gibberish, degenerate
    filler) but charges nothing for *abandoning* modes of the reference (you may stop saying
    most things a base model would say). Mode-seeking is exactly the right shape for a leash:
    sharpening onto good behavior is allowed; inventing non-language is punished.

    **Why estimate it at all.** Per token, the exact KL is a sum over the ~150k-entry vocab —
    at every position of every rollout, against a second model's logits. Instead we evaluate it
    only at the tokens actually sampled, using Schulman's estimators. With
    \( \rho = \pi_{\text{ref}}(x)/\pi_\theta(x) \) at a sampled \( x \sim \pi_\theta \):

    \[ k_1 = -\log \rho, \qquad k_2 = \tfrac{1}{2} (\log \rho)^2, \qquad k_3 = \rho - 1 - \log \rho . \]

    \( k_1 \) is the textbook unbiased estimator but is high-variance and swings negative on
    individual samples; \( k_2 \) is low-variance but biased; \( k_3 \) — a control-variate
    combination of the two — is unbiased *and* pointwise non-negative (it is
    \( f(\rho) = \rho - 1 - \log\rho \), convex, zero at \( \rho = 1 \)). That's the one to use.

    **Why the leash matters *here*.** A sparse binary reward is a strip-mining license: the
    cheapest way to correlate tokens with reward is to collapse the chain-of-thought into
    repeated filler and let entropy die ("reasoning collapse"). But this project's entire bet —
    that a policy trained on *orange* catches *banana* — rides on the language prior, because
    the prior is the mechanism by which "banana" inherits behavior from "orange". Destroy the
    prior and you've won the toy task while losing the experiment. The KL term is literally
    "stay a language model while you learn."
    """)
    return


@app.cell
def _():
    s6_temp = mo.ui.slider(0.3, 3.0, step=0.05, value=0.7, label="policy temperature T")
    s6_shift = mo.ui.slider(0.0, 4.0, step=0.1, value=0.0, label="logit shift on one token")
    s6_n = mo.ui.slider(8, 256, step=8, value=64, label="samples per estimate n")
    mo.hstack([s6_temp, s6_shift, s6_n], justify="start", gap=2)
    return s6_n, s6_shift, s6_temp


@app.cell
def _(s6_n, s6_shift, s6_temp):
    plt.close("all")
    _ref_logits = np.array([2.0, 1.6, 1.2, 0.9, 0.6, 0.3, 0.0, -0.3, -0.7, -1.1, -1.6, -2.2])
    _K = len(_ref_logits)

    def _softmax(z):
        _e = np.exp(z - z.max())
        return _e / _e.sum()

    def _policy(T, shift):
        _z = _ref_logits / T
        _z = _z.copy()
        _z[5] += shift
        return _softmax(_z)

    _q = _softmax(_ref_logits)
    _p_now = _policy(s6_temp.value, s6_shift.value)

    _rng = np.random.default_rng(11)
    _Ts = np.logspace(np.log10(0.3), np.log10(3.0), 21)
    _true, _est = [], {"k1": [], "k2": [], "k3": []}
    _R, _n = 200, s6_n.value
    for _T in _Ts:
        _p = _policy(_T, s6_shift.value)
        _true.append(np.sum(_p * np.log(_p / _q)))
        _idx = _rng.choice(_K, size=(_R, _n), p=_p)
        _logrho = np.log(_q[_idx] / _p[_idx])  # log ρ, ρ = π_ref/π_θ
        _est["k1"].append((-_logrho).mean(1))
        _est["k2"].append((0.5 * _logrho**2).mean(1))
        _est["k3"].append((np.exp(_logrho) - 1 - _logrho).mean(1))

    _fig, (_axd, _axk) = plt.subplots(
        1, 2, figsize=(9.8, 3.6), layout="constrained", width_ratios=[1, 1.5]
    )
    _x = np.arange(_K)
    _axd.bar(_x - 0.2, _q, width=0.4, color=C_FAINT, label="π_ref (frozen)")
    _axd.bar(_x + 0.2, _p_now, width=0.4, color=C_BLUE, label="π_θ (yours)")
    _axd.set(
        xlabel="token id",
        ylabel="probability",
        title=f"KL(π_θ‖π_ref) = {np.sum(_p_now * np.log(_p_now / _q)):.3f} nats",
    )
    _axd.legend(frameon=False, fontsize=8.5)

    for _name, _c in zip(["k1", "k2", "k3"], [C_BLUE, C_ORANGE, C_AQUA]):
        _m = np.array([_v.mean() for _v in _est[_name]])
        _sd = np.array([_v.std() for _v in _est[_name]])
        _axk.plot(_Ts, _m, color=_c, lw=1.8, label=_name)
        _axk.fill_between(_Ts, _m - _sd, _m + _sd, color=_c, alpha=0.15, lw=0)
    _axk.plot(_Ts, _true, color=C_INK, lw=2.4, ls="--", label="true KL")
    _axk.axvline(s6_temp.value, color=C_MUTED, ls=":", lw=1.2)
    _axk.axhline(0, color=C_MUTED, lw=0.8)
    _axk.set(
        xscale="log",
        xlabel="policy temperature T (sharpen ←  → flatten)",
        ylabel="nats",
        title=f"true KL vs Monte-Carlo estimators (n={_n}, ±1σ over {_R} resamples)",
    )
    _axk.legend(frameon=False, fontsize=8.5)
    _fig
    return


@app.cell
def _():
    mo.md(r"""
    Watch the bands as the policy drifts from the reference (T away from 1, or crank the
    shift): \( k_1 \)'s band balloons and dips below zero — a *negative* KL estimate for a
    divergence — while \( k_3 \) hugs the dashed truth with a fraction of the spread and never
    goes negative. Shrink \( n \) to make the contrast brutal. `train.py` uses \( k_3 \).
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## 7 · Capstone: the identical algorithm, learning to catch

    Everything above, assembled: REINFORCE with a group baseline — *exactly* the section-3
    loss, \( A_i = r_i - \bar r_{-i} \) (leave-one-out, per the fine print), no std division,
    no clip (section 5 says it's a no-op
    on-policy), and here no KL because there's no prior worth protecting in a randomly
    initialized 300-parameter net. The environment is an inline numpy mirror of
    `catch_env.py`: 7 columns, the fruit falls for 5 turns, drift after the fall on turns 2 and
    4 (strawberry 0, apple −1, orange +1), reward 1 if the basket is under the fruit at the
    end. The policy is a one-hidden-layer softmax net over {LEFT, STAY, RIGHT}, reading
    (fruit col, basket col, offset, fruit one-hot, turn). Each training step samples one
    episode (fruit + seed) and rolls it out \( G \) times **from the identical initial state**.

    Try: baseline off (watch learning get noisier and slower — section 2 live), shaped reward
    on (partial credit \( -0.1 \cdot |\text{fruit} - \text{basket}| \) at the end — the
    zero-variance escape hatch from section 3), tiny G vs big G.
    """)
    return


@app.function
def catch_init_params(seed=0, hidden=32):
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0.0, 0.2, (8, hidden)),
        "b1": np.zeros(hidden),
        "W2": rng.normal(0.0, 0.2, (hidden, 3)),
        "b2": np.zeros(3),
    }


@app.function
def catch_features(fruit_cols, basket_cols, fruit_idx, turn):
    """Featurize a batch of states. fruit_cols scalar or (N,), basket_cols (N,)."""
    n = len(basket_cols)
    X = np.zeros((n, 8))
    X[:, 0] = np.asarray(fruit_cols) / 6.0
    X[:, 1] = basket_cols / 6.0
    X[:, 2] = (np.asarray(fruit_cols) - basket_cols) / 6.0
    X[:, 3 + fruit_idx] = 1.0
    X[:, 6] = (turn - 1) / 4.0
    X[:, 7] = 1.0
    return X


@app.function
def catch_forward(params, X):
    H = np.tanh(X @ params["W1"] + params["b1"])
    logits = H @ params["W2"] + params["b2"]
    logits = logits - logits.max(axis=1, keepdims=True)
    P = np.exp(logits)
    P /= P.sum(axis=1, keepdims=True)
    return H, P


@app.function
def catch_rollout_group(params, fruit_idx, seed, G, rng, shaped=False):
    """G rollouts of ONE episode (same fruit, same seed => identical initial state).
    Returns (tape, rewards): tape holds per-turn (X, H, P, actions) for the backprop."""
    ep = np.random.default_rng(seed)
    fruit_col = int(ep.integers(CATCH_WIDTH))
    baskets = np.full(G, int(ep.integers(CATCH_WIDTH)), dtype=float)
    tape = []
    for turn in range(1, CATCH_TURNS + 1):
        X = catch_features(fruit_col, baskets, fruit_idx, turn)
        H, P = catch_forward(params, X)
        acts = (rng.random((G, 1)) > np.cumsum(P, axis=1)).sum(axis=1)  # 0/1/2
        tape.append((X, H, P, acts))
        baskets = np.clip(baskets + acts - 1, 0, CATCH_WIDTH - 1)
        if turn % 2 == 0:  # drift after the fall, turns 2 and 4 — as in catch_env.py
            fruit_col = int(np.clip(fruit_col + CATCH_DRIFT[fruit_idx], 0, CATCH_WIDTH - 1))
    rewards = (baskets == fruit_col).astype(float)
    if shaped:
        rewards = rewards - 0.1 * np.abs(baskets - fruit_col)
    return tape, rewards


@app.function
def catch_policy_gradient(params, tape, advantages):
    """Accumulate  Σ_i A_i Σ_t ∇_θ log π(a_t | s_t)  by hand (one hidden layer)."""
    grads = {k: np.zeros_like(v) for k, v in params.items()}
    for X, H, P, acts in tape:
        D = -P.copy()
        D[np.arange(len(acts)), acts] += 1.0  # ∇_logits log π(a) = onehot(a) − P
        D *= advantages[:, None]
        grads["W2"] += H.T @ D
        grads["b2"] += D.sum(0)
        dH = (D @ params["W2"].T) * (1.0 - H * H)
        grads["W1"] += X.T @ dH
        grads["b1"] += dH.sum(0)
    return grads


@app.function
def catch_eval(params, n_eps=300, seed=123):
    """Greedy (argmax) catch rate per fruit over n_eps random starts, vectorized."""
    rng = np.random.default_rng(seed)
    rates = np.zeros(3)
    for fi in range(3):
        fruit = rng.integers(0, CATCH_WIDTH, n_eps).astype(float)
        baskets = rng.integers(0, CATCH_WIDTH, n_eps).astype(float)
        for turn in range(1, CATCH_TURNS + 1):
            _, P = catch_forward(params, catch_features(fruit, baskets, fi, turn))
            baskets = np.clip(baskets + P.argmax(1) - 1, 0, CATCH_WIDTH - 1)
            if turn % 2 == 0:
                fruit = np.clip(fruit + CATCH_DRIFT[fi], 0, CATCH_WIDTH - 1)
        rates[fi] = (baskets == fruit).mean()
    return rates


@app.function
def catch_train(G=8, steps=1000, lr=0.15, use_baseline=True, shaped=False, seed=0, eval_every=25):
    """The section-3 loss, verbatim: group rollouts on a shared seed, leave-one-out
    advantage A_i = r_i − mean(siblings' r), ascend  Σ A_i ∇log π  — nothing else."""
    params = catch_init_params(seed)
    rng = np.random.default_rng(seed + 1)
    xs, curves = [], []
    for step in range(steps + 1):
        if step % eval_every == 0 or step == steps:
            xs.append(step)
            curves.append(catch_eval(params))
        if step == steps:
            break
        fruit_idx = step % 3
        ep_seed = int(rng.integers(1 << 30))
        tape, rewards = catch_rollout_group(params, fruit_idx, ep_seed, G, rng, shaped)
        if use_baseline:  # leave-one-out: baseline for rollout i = mean of its siblings
            adv = rewards - (rewards.sum() - rewards) / (G - 1)
        else:
            adv = rewards.copy()
        grads = catch_policy_gradient(params, tape, adv)
        for k in params:
            params[k] += lr * grads[k] / G
    return params, np.array(xs), np.array(curves)


@app.cell
def _():
    s7_G = mo.ui.slider(2, 32, step=1, value=8, label="group size G")
    s7_steps = mo.ui.slider(100, 2000, step=100, value=1000, label="training steps")
    s7_baseline = mo.ui.switch(value=True, label="leave-one-out group baseline")
    s7_shaped = mo.ui.switch(value=False, label="shaped reward (−0.1·|distance|)")
    mo.hstack([s7_G, s7_steps, s7_baseline, s7_shaped], justify="start", gap=1.5)
    return s7_G, s7_baseline, s7_shaped, s7_steps


@app.cell
def _(s7_G, s7_baseline, s7_shaped, s7_steps):
    plt.close("all")
    _params, _xs, _curves = catch_train(
        G=s7_G.value,
        steps=s7_steps.value,
        use_baseline=s7_baseline.value,
        shaped=s7_shaped.value,
        seed=0,
    )
    _fig, _ax = plt.subplots(figsize=(8.5, 3.6), layout="constrained")
    for _fi, (_name, _c) in enumerate(zip(CATCH_FRUITS, FRUIT_COLORS)):
        _ax.plot(_xs, _curves[:, _fi], color=_c, lw=2, label=_name)
        _ax.annotate(
            f" {_name} {_curves[-1, _fi]:.2f}",
            (_xs[-1], _curves[-1, _fi]),
            color=_c,
            fontsize=9,
        )
    _ax.axhline(_curves[0].mean(), color=C_FAINT, ls="--", lw=1.2)
    _ax.annotate(
        f" random init ≈ {_curves[0].mean():.2f}", (0, _curves[0].mean()), color=C_MUTED, fontsize=8.5
    )
    _ax.set(
        xlabel="training step",
        ylabel="greedy catch rate (300 eval episodes)",
        ylim=(0, 1.05),
        xlim=(0, _xs[-1] * 1.12),
        title=(
            f"G={s7_G.value}, baseline {'ON' if s7_baseline.value else 'OFF'}, "
            f"reward {'shaped' if s7_shaped.value else 'binary'} — "
            f"mean catch rate {_curves[-1].mean():.2f}"
        ),
    )
    _ax.legend(frameon=False, loc="lower right")

    mo.vstack(
        [
            _fig,
            mo.md(
                r"""
    **This is the identical algorithm `train.py` runs on Qwen3-0.6B.** Line for line: group
    rollouts from a shared initial state, leave-one-out sibling-mean advantage, \( -\sum_i A_i \log
    \pi_\theta(\text{what } i \text{ did}) \), one fresh update per batch. The only differences
    are that the policy is a 600M-parameter transformer instead of a 300-parameter numpy net,
    an "action" is a whole sampled token sequence (so the log-prob is a masked sum over
    generated tokens, section 4), and there's a KL leash to a reference model (section 6)
    because *that* policy starts with a prior worth protecting.
    """
            ),
        ],
        gap=0.5,
    )
    return


@app.cell
def _():
    mo.md(r"""
    ## 8 · Coda: mapping the notebook onto `train.py`

    | notebook section | the line(s) of `train.py` it explains |
    |---|---|
    | 1 · score-function trick | the loss itself: `-(A * logprobs).sum()` — reward appears only as a multiplier on log-probs re-computed with grad |
    | 2 · baselines are free | the leave-one-out advantage `A[i] = r[i] - siblings.mean()` — unbiased by \( \nabla \int \pi = 0 \) *because* the baseline excludes rollout i's own reward (section 3's fine print), there purely for variance |
    | 3 · group baseline | the `G` rollouts of the same `(fruit, seed)` episode per step; `frac_zero_var_groups` in the logs is the \( p^G + (1-p)^G \) failure mode made observable |
    | 3b · Dr.GRPO | the *absence* of `/ rewards.std()` — subtract a baseline, divide by nothing |
    | 4 · the mask | the boolean mask carried beside token ids; `--check-mask` verifies the masked positions decode to exactly what the model generated |
    | 5 · no clip | the *absence* of an importance ratio: one update per fresh batch ⇒ \( \rho \equiv 1 \) |
    | 6 · KL leash | the \( \beta \cdot k_3 \) penalty against the frozen reference copy |

    **One caveat that is objective, not bookkeeping: token aggregation.** Whether you divide
    each trajectory's masked log-prob *sum* by its own token count (token-mean) or not
    (token-sum) changes what you optimize, not how you report it. Per-trajectory token-mean
    quietly down-weights every token in long chains of thought — a 200-token rollout's
    brilliant move gets 1/200th the per-token credit of a 10-token rollout's — which is a
    length bias with a documented effect on CoT (it's half of what Dr.GRPO fixes, alongside
    the std division). Decide the normalizer as part of the objective — e.g. a constant, or
    sum-with-global-normalization — and then leave it alone.

    **Two sanity experiments before trusting any RL result:**

    1. **Overfit one episode.** Fix a single `(fruit, seed)` and train on it alone. The policy
       should saturate to catch rate 1.0 within tens of steps. If it can't memorize *one*
       episode, no claim about learning is meaningful — the gradient path is broken (mask,
       sign, or logprob indexing), and no amount of hyperparameter search will fix arithmetic.
    2. **Zero-advantage inertness.** Force \( A = 0 \) for every rollout and train. The loss
       should be exactly zero and the weights should not move (up to numerics). If anything
       changes, some term in your loss is not advantage-gated — a leaked observation token, a
       stray KL sign, an un-detached value — and it was silently steering every "real" run too.

    Both take five minutes. Both catch real bugs in hand-rolled RL loops far more often
    than seems fair.
    """)
    return


if __name__ == "__main__":
    app.run()
