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

    # Visual grammar — the same color means the same thing in EVERY figure:
    #   C_INK / C_FAINT : fixed scenery (the function, contours, the truth P, the reference)
    #   C_ORANGE        : whatever your slider is currently moving (probe, secant, path, sample)
    #   C_BLUE          : the exact object the orange thing converges to, or the thing you control
    #   C_AQUA          : reward
    C_BLUE, C_ORANGE, C_AQUA, C_YELLOW, C_MAGENTA = (
        "#2a78d6",
        "#eb6834",
        "#1baf7a",
        "#eda100",
        "#e87ba4",
    )
    C_INK, C_MUTED, C_FAINT = "#0b0b0b", "#52514e", "#c9c8c0"


@app.function
def ref_softmax(z):
    """Numerically stable softmax of a 1-D array."""
    e = np.exp(np.asarray(z, dtype=float) - np.max(z))
    return e / e.sum()


@app.cell
def _():
    mo.md(r"""
    # A math refresher, aimed like a rifle at `grpo_notes.py`

    This notebook has one design principle. Fluency in mathematics lives in the
    *round trips between representations* — reading a line of notation and seeing
    the picture it encodes, or the five lines of numpy it compiles to — and those
    round trips are exactly what decays without use: the individual pieces
    survive, the crossings silt up. So the whole notebook is built out of
    crossings. **Every core idea appears three times, aligned:**

    1. **the picture** — a plot you can drag;
    2. **the notation** — with each symbol explicitly pinned to something in the
       picture above it ("this term is that arrow");
    3. **the numpy** — a handful of lines on concrete numbers, shown in the cell,
       verifying that the notation and the picture are the same fact.

    Three costumes, one fact, and every section below forces the changes of
    costume — because that, not any one representation, is the skill.

    Two conventions run through everything. First, **every slider is a scrubbable
    animation** — a 3Blue1Brown film you play by hand. Dragging Δx shrinks the
    secant into the tangent; dragging n walks the sample mean down the funnel;
    dragging a step counter marches a ball uphill. When a demo says *drag*, the
    dragging is the lesson. Second, **colors mean the same thing in every
    figure**: black/grey is fixed scenery (the function, the truth, the
    reference), **orange** is whatever your slider is currently moving (probes,
    secants, sample paths), **blue** is the exact object the orange thing
    converges to — or the thing you control — and **green** is reward.

    This is not a general math course. Every section exists because a specific step
    in `grpo_notes.py` (the REINFORCE → baselines → GRPO derivation for the
    fruit-catching project) leans on it directly, and says so at the end. Each
    section ends with **exercises** — including "translate" problems that hand you
    one representation and demand another — with worked solutions hidden behind
    accordions. Genuinely attempt before peeking; reading math feels like knowing
    math, and only doing math is.

    At the end you re-derive the central trick of the whole project yourself, then
    watch numpy confirm it to nine decimal places.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## 1 · Derivatives as sensitivity

    ### The picture

    A function is a hillside. The derivative at a point answers one question:
    **standing here, if I shuffle one step in x, how much does my altitude change?**
    Below, the orange dots are two probes on the hillside, a horizontal gap
    \( \Delta x \) apart; the orange line through them (the *secant*) has slope =
    rise ÷ run between the probes. Drag \( \Delta x \) toward zero and watch the
    secant rotate into the blue dashed line (the *tangent*): the local slope, the
    thing the derivative *is*.
    """)
    return


@app.cell
def _():
    r1_x0 = mo.ui.slider(-3.0, 3.0, step=0.05, value=-1.2, label="point x₀")
    r1_dx = mo.ui.slider(0.01, 2.0, step=0.01, value=1.6, label="wiggle Δx")
    mo.hstack([r1_x0, r1_dx], justify="start", gap=2)
    return r1_dx, r1_x0


@app.cell
def _(r1_dx, r1_x0):
    plt.close("all")

    def _f(x):
        return np.sin(x) + x**2 / 8.0

    def _fp(x):
        return np.cos(x) + x / 4.0

    _x0, _dx = r1_x0.value, r1_dx.value
    _xs = np.linspace(-4.2, 4.2, 400)
    _sec_slope = (_f(_x0 + _dx) - _f(_x0)) / _dx
    _tan_slope = _fp(_x0)

    _fig, _ax = plt.subplots(figsize=(8.5, 3.7), layout="constrained")
    _ax.plot(_xs, _f(_xs), color=C_INK, lw=2)
    _ax.annotate("f(x) = sin x + x²/8", (2.35, _f(2.35) + 0.4), color=C_INK, fontsize=9.5)
    _loc = np.linspace(_x0 - 1.8, _x0 + max(_dx, 0.4) + 1.4, 2)
    _sec_y = _f(_x0) + _sec_slope * (_loc - _x0)
    _tan_y = _f(_x0) + _tan_slope * (_loc - _x0)
    _ax.plot(_loc, _sec_y, color=C_ORANGE, lw=2)
    _ax.plot(_loc, _tan_y, color=C_BLUE, lw=2, ls="--")
    _ax.annotate(
        f"secant, slope {_sec_slope:+.3f}", (_loc[-1], _sec_y[-1]),
        textcoords="offset points", xytext=(-6, 9), color=C_ORANGE, fontsize=9, ha="right",
    )
    _ax.annotate(
        f"tangent, slope {_tan_slope:+.3f}", (_loc[0], _tan_y[0]),
        textcoords="offset points", xytext=(2, -14), color=C_BLUE, fontsize=9,
    )
    _ax.scatter([_x0, _x0 + _dx], [_f(_x0), _f(_x0 + _dx)], color=C_ORANGE, zorder=5, s=40)
    _ax.scatter([_x0], [_f(_x0)], color=C_BLUE, zorder=6, s=40)
    _ybr = min(_f(_x0), _f(_x0 + _dx)) - 0.35
    _ax.annotate(
        "", xy=(_x0 + _dx, _ybr), xytext=(_x0, _ybr),
        arrowprops=dict(arrowstyle="<->", color=C_ORANGE, lw=1.2),
    )
    _ax.annotate(
        "Δx", ((2 * _x0 + _dx) / 2, _ybr - 0.1), color=C_ORANGE,
        fontsize=10, ha="center", va="top",
    )
    _ax.set(
        xlim=(-4.2, 4.2), ylim=(-2.0, 3.4), xlabel="x", ylabel="f(x)",
        title=f"Δx = {_dx:.2f}:  secant slope {_sec_slope:+.3f}  →  f′(x₀) = {_tan_slope:+.3f}"
        f"   (gap {abs(_sec_slope - _tan_slope):.3f})",
    )
    _fig
    return


@app.cell
def _():
    mo.md(r"""
    ### The notation, pinned to the picture

    \[ f'(x_0) \;=\; \lim_{\Delta x \to 0} \frac{f(x_0 + \Delta x) - f(x_0)}{\Delta x} \]

    - \( \Delta x \) — the horizontal gap between the two orange dots. Your right
      slider.
    - \( f(x_0 + \Delta x) - f(x_0) \) — the vertical rise between them.
    - the fraction — rise over run: the **orange secant's slope**, exactly the
      number in the plot legend.
    - \( \lim_{\Delta x \to 0} \) — the act of dragging the slider left. The limit
      is the slope the secant settles on as it rotates into the blue tangent.
    - \( f'(x_0) \) — the blue tangent's slope: **output wiggle per unit of input
      wiggle**, right here. \( f' = 3 \) means output currently moves 3× as fast as
      input; \( f' = 0 \) means the output, right now, doesn't care.

    That "how much does the output care about this input" reading is the whole
    reason gradients run machine learning: training *is* asking, for each of 600M
    parameters, "how much does the objective care about you?"

    ### The chain rule is wiggle-propagation

    Compose two functions into a pipeline, \( x \to g(x) \to f(g(x)) \). Wiggle
    \( x \) by a hair: stage \( g \) amplifies the wiggle by its local slope
    \( g'(x) \); the wiggled value then enters \( f \), which amplifies *its* input
    wiggle by *its* local slope \( f'(g(x)) \). Amplifications in sequence
    **multiply** — a 2× stage feeding a 3× stage is a 6× pipe:

    \[ \underbrace{\frac{d}{dx} f(g(x))}_{\text{whole pipe}}
       \;=\; \underbrace{f'(g(x))}_{\substack{\text{outer stage's slope,}\\ \text{measured at the inner value}}}
       \cdot \underbrace{g'(x)}_{\text{inner stage's slope}} \]

    That's backpropagation: a network is a long pipeline, and the gradient anywhere
    is the product of all local slopes downstream. When \( \nabla_\theta \log
    \pi_\theta \) shows up later, this multiplication — chained through every layer
    — is how it gets computed.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    **Scrub a wiggle through the pipeline.** Three number lines below: the input
    \( x \), the intermediate \( u = x^2 \), the output \( y = \sin(x^2) \), each
    drawn relative to its rest position at \( x_0 = 0.8 \). The slider is time —
    drag it back and forth and *you are* the wiggle. Watch the amplitudes: stage
    one multiplies the sway by \( g'(x_0) = 1.6 \), stage two by
    \( f'(u_0) = \cos(0.64) \approx 0.80 \). The orange bands are exactly those
    amplification factors, and their product, \( 1.6 \times 0.80 = 1.28 \), is
    the chain rule — visible as motion.
    """)
    return


@app.cell
def _():
    r1_t = mo.ui.slider(0.0, 1.0, step=0.01, value=0.25, label="scrub the wiggle (time)")
    r1_t
    return (r1_t,)


@app.cell
def _(r1_t):
    plt.close("all")
    _x0, _A = 0.8, 0.25
    _d = _A * np.sin(2 * np.pi * r1_t.value)       # the input wiggle right now
    _x = _x0 + _d
    _u0, _y0 = _x0**2, np.sin(_x0**2)
    _g1 = 2 * _x0                                  # g'(x0) = 1.6
    _f1 = np.cos(_x0**2)                           # f'(u0) ≈ 0.80

    _fig, _ax = plt.subplots(figsize=(8.5, 3.4), layout="constrained")
    _levels = [2.0, 1.0, 0.0]
    _amps = [_A, _A * _g1, _A * _g1 * _f1]         # linearized amplitude per stage
    _offs = [_d, _x**2 - _u0, np.sin(_x**2) - _y0]
    _labs = [
        f"x   (around {_x0:.2f})",
        f"u = x²   (around {_u0:.2f})",
        f"y = sin x²   (around {_y0:.2f})",
    ]
    for _lv, _amp, _off, _lab in zip(_levels, _amps, _offs, _labs):
        _ax.plot([-0.62, 0.62], [_lv, _lv], color=C_FAINT, lw=1.4)
        _ax.fill_betweenx([_lv - 0.09, _lv + 0.09], -_amp, _amp, color=C_ORANGE, alpha=0.15, lw=0)
        _ax.scatter([_off], [_lv], color=C_ORANGE, s=90, zorder=5)
        _ax.annotate(_lab, (-0.61, _lv + 0.13), color=C_MUTED, fontsize=9)
    _ax.annotate("", xy=(0.42, 1.12), xytext=(0.42, 1.88),
                 arrowprops=dict(arrowstyle="->", color=C_MUTED, lw=1.3))
    _ax.annotate(f"× g′(x₀) = {_g1:.2f}", (0.46, 1.46), color=C_INK, fontsize=10)
    _ax.annotate("", xy=(0.42, 0.12), xytext=(0.42, 0.88),
                 arrowprops=dict(arrowstyle="->", color=C_MUTED, lw=1.3))
    _ax.annotate(f"× f′(u₀) = {_f1:.2f}", (0.46, 0.46), color=C_INK, fontsize=10)
    _ax.annotate(
        f"whole pipe:  × {_g1 * _f1:.2f}  =  the chain rule", (-0.61, -0.44),
        color=C_BLUE, fontsize=10,
    )
    _ax.set(xlim=(-0.65, 0.95), ylim=(-0.55, 2.45))
    _ax.axis("off")
    _fig
    return


@app.cell
def _():
    _f = lambda x: np.sin(x) + x**2 / 8
    _x0, _h = -1.2, 1e-6
    _secant = (_f(_x0 + _h) - _f(_x0)) / _h
    _analytic = np.cos(_x0) + _x0 / 4

    _comp = lambda x: np.sin(x**2)  # the pipeline x → x² → sin
    _chain_fd = (_comp(_x0 + _h) - _comp(_x0)) / _h
    _chain_an = np.cos(_x0**2) * 2 * _x0

    mo.md(
        r"""
    ### The same fact in numpy

    The derivative is not exotic — it's a subtraction and a division with a small
    \( \Delta x \). The analytic formula is just what that computation converges to:

    ```python
    f  = lambda x: np.sin(x) + x**2/8
    x0, h = -1.2, 1e-6
    secant   = (f(x0 + h) - f(x0)) / h    # the orange line, Δx shrunk to 1e-6
    analytic = np.cos(x0) + x0/4          # the blue line, from the rules

    comp     = lambda x: np.sin(x**2)     # pipeline: x → x² → sin
    chain_fd = (comp(x0 + h) - comp(x0)) / h
    chain_an = np.cos(x0**2) * 2*x0       # outer slope · inner slope
    ```
    """
        + f"""
    ```
    secant   = {_secant:+.9f}      chain_fd = {_chain_fd:+.9f}
    analytic = {_analytic:+.9f}      chain_an = {_chain_an:+.9f}
    ```

    Same numbers from both directions. Whenever a symbolic derivative feels shaky,
    this two-line finite-difference check is the tiebreaker — you will use it on
    the project's central identity in section 2.
    """
    )
    return


@app.cell
def _():
    mo.vstack([
        mo.md(r"""
    ### Exercises 1

    **1.1 (warm-up).** Differentiate \( f(x) = 3x^2 + 5x \).

    **1.2.** Differentiate \( f(x) = \sin(x^2) \) — and say in one sentence why the
    two factors *multiply* rather than add.

    **1.3 (translate: picture → symbols → numpy).** In the demo, set
    \( \Delta x = 0.5 \). The orange line passes through \( (x_0, f(x_0)) \) and
    \( (x_0 + 0.5,\, f(x_0 + 0.5)) \). (a) Write the symbolic expression its slope
    computes. (b) Write one line of numpy for it. (c) Which single symbol of the
    limit definition does it approximate, and what operation makes it exact?

    **1.4 (stinger).** For \( f(x) = e^{-x^2} \): compute \( f'(x) \), then find
    where the *sensitivity* \( |f'(x)| \) is greatest. (Not at the peak. Why not —
    what does the hillside look like at a peak?)
    """),
        mo.accordion({
            "Solution 1.1": mo.md(r"""
    Power rule, term by term: \( f'(x) = 6x + 5 \). Sensitivity reading: at
    \( x = 0 \) the output moves 5× as fast as the input; at \( x = 10 \), 65×.
    """),
            "Solution 1.2": mo.md(r"""
    Outer \( \sin(u) \), inner \( u = x^2 \):
    \[ f'(x) = \cos(x^2) \cdot 2x. \]
    They multiply because amplifications compound: the inner stage scales the
    wiggle by \( 2x \), then the outer stage scales *whatever it receives* by
    \( \cos(x^2) \) — a scaling of a scaling is a product. Adding would describe
    two independent machines whose outputs you sum (that's the sum rule).
    """),
            "Solution 1.3": mo.md(r"""
    (a) \[ \frac{f(x_0 + 0.5) - f(x_0)}{0.5} \]
    (b) `slope = (f(x0 + 0.5) - f(x0)) / 0.5`
    (c) It approximates \( f'(x_0) \) — the whole left-hand side. The operation
    that makes it exact is the limit \( \Delta x \to 0 \): the slider-drag,
    performed to completion. (Finite \( \Delta x \) = secant; limit = tangent.)
    """),
            "Solution 1.4": mo.md(r"""
    Chain rule, outer \( e^u \), inner \( u = -x^2 \):
    \( f'(x) = -2x\,e^{-x^2} \).

    At the peak \( x = 0 \), \( f' = 0 \): a peak is locally *flat* — the output is
    momentarily indifferent to the input, which is exactly why "set the gradient to
    zero" finds maxima. Sensitivity peaks where \( |f'| = 2|x|e^{-x^2} \) is
    maximal: \( \frac{d}{dx}(2x e^{-x^2}) = (2 - 4x^2)e^{-x^2} = 0 \Rightarrow
    x = \pm 1/\sqrt{2} \) — on the *shoulders* of the bump, where the hillside is
    steepest. Corollary you'll feel in RL: gradient ascent slows to a crawl
    precisely as it arrives anywhere worth being.
    """),
        }),
        mo.md(
            r"**Where you'll use this:** the entire project is one derivative, "
            r"\( \nabla_\theta J \) — how sensitive is expected reward to each of "
            r"600M weights — computed by exactly this chain rule through the network "
            r"(grpo_notes §1)."
        ),
    ], gap=1)
    return


@app.cell
def _():
    mo.md(r"""
    ## 2 · Logs and exps: the machinery of probability

    ### The picture

    Below: the *same curve* seen through two lenses. Left, \( f(\theta) \) in raw
    coordinates. Right, \( \log f(\theta) \) — the log lens, which squashes tall
    values hard and small values barely at all. A slope measured *after* squashing
    is the raw slope **deflated by the current height**: where \( f \) is tall, the
    log lens flattens its climb; where \( f \) is short, it exaggerates it. Slide
    \( \theta_0 \) and watch the suptitle: left slope ÷ right slope = the height of
    the left curve, everywhere, always.
    """)
    return


@app.cell
def _():
    r2_theta = mo.ui.slider(0.3, 3.0, step=0.05, value=1.5, label="point θ₀")
    r2_theta
    return (r2_theta,)


@app.cell
def _(r2_theta):
    plt.close("all")
    _t0 = r2_theta.value
    _ts = np.linspace(0.15, 3.2, 400)
    _f = _ts**2 + 0.5
    _f0 = _t0**2 + 0.5
    _fp0 = 2 * _t0                 # f'(θ0)
    _lp0 = _fp0 / _f0              # (log f)'(θ0)

    _fig, (_axf, _axl) = plt.subplots(1, 2, figsize=(9.5, 3.5), layout="constrained")
    _loc = np.linspace(max(_t0 - 0.9, 0.15), _t0 + 0.9, 2)
    _axf.plot(_ts, _f, color=C_INK, lw=2)
    _axf.plot(_loc, _f0 + _fp0 * (_loc - _t0), color=C_BLUE, lw=2)
    _axf.scatter([_t0], [_f0], color=C_ORANGE, zorder=5, s=45)
    _axf.set(xlabel="θ", ylabel="f(θ)", title=f"f = θ² + ½ :  slope {_fp0:.3f},  height {_f0:.3f}")
    _axl.plot(_ts, np.log(_ts**2 + 0.5), color=C_INK, lw=2)
    _axl.plot(_loc, np.log(_f0) + _lp0 * (_loc - _t0), color=C_BLUE, lw=2)
    _axl.scatter([_t0], [np.log(_f0)], color=C_ORANGE, zorder=5, s=45)
    _axl.set(xlabel="θ", ylabel="log f(θ)", title=f"log f :  slope {_lp0:.3f}")
    _fig.suptitle(
        f"slope ratio {_fp0:.3f} / {_lp0:.3f} = {_fp0 / _lp0:.3f}   =   height f(θ₀) = {_f0:.3f}   ✓",
        fontsize=11,
    )
    _fig
    return


@app.cell
def _():
    mo.md(r"""
    ### The notation, pinned to the picture

    First, why logs own probability at all: \( \log(ab) = \log a + \log b \) —
    **products become sums**. The probability of an LLM rollout is a product of
    hundreds of per-token probabilities, each below 1; the raw product underflows
    to zero, while the sum of logs is a comfortable number like −230. That is the
    entire reason log-likelihoods exist (numpy proof in the next cell). And
    \( e^x \) is the inverse lens, with the famous property of being **its own
    derivative** — its growth rate equals its current value, compounding embodied.

    Now the identity the whole project stands on. Derivation, two lines, each with
    its picture-gloss:

    \[ \frac{d}{d\theta} \log f(\theta)
       \;=\; \underbrace{\frac{1}{f(\theta)}}_{\substack{\text{outer stage: log's slope,} \\ \text{evaluated at height } f}}
       \cdot \underbrace{f'(\theta)}_{\text{inner stage: raw slope}}
       \;=\; \frac{f'(\theta)}{f(\theta)} \]

    *Line 1 is spatially:* the log lens squashes by \( 1/\text{height} \), so the
    right panel's slope is the left panel's slope ÷ the left panel's height. That is
    the suptitle of the demo, as an equation.

    \[ \text{Read right-to-left:} \qquad
       \underbrace{\nabla f}_{\text{raw slope}}
       \;=\; \underbrace{f}_{\text{height}} \cdot \underbrace{\nabla \log f}_{\text{log-lens slope}} \]

    *Line 2 is computationally:* you can reconstruct a function's raw gradient from
    its **log**-gradient by multiplying back the height. Useless-looking — until the
    function is a probability \( \pi_\theta \) sitting inside an integral, where the
    spare factor of \( \pi_\theta \) is exactly what turns the integral back into a
    sampleable expectation. That move is the **log-derivative trick**, the single
    load-bearing identity of `grpo_notes.py`. You re-derive its consequences
    yourself in section 8; stare at it now until it looks obvious.
    """)
    return


@app.cell
def _():
    _f = lambda t: t**2 + 0.5
    _t0, _h = 1.5, 1e-6
    _slope_f = (_f(_t0 + _h) - _f(_t0 - _h)) / (2 * _h)
    _slope_logf = (np.log(_f(_t0 + _h)) - np.log(_f(_t0 - _h))) / (2 * _h)

    _rng = np.random.default_rng(0)
    _tok_p = _rng.uniform(0.05, 0.6, 200)   # 200 per-token probabilities
    _prod = np.prod(_tok_p)
    _logsum = np.log(_tok_p).sum()

    mo.md(
        r"""
    ### The same facts in numpy

    ```python
    f, t0, h   = lambda t: t**2 + 0.5,  1.5,  1e-6
    slope_f    = (f(t0+h) - f(t0-h)) / (2*h)                  # left panel's slope
    slope_logf = (np.log(f(t0+h)) - np.log(f(t0-h))) / (2*h)  # right panel's slope
    # the identity claims:  slope_f / f(t0) == slope_logf

    tok_p  = rng.uniform(0.05, 0.6, 200)   # 200 per-token probabilities
    prod   = np.prod(tok_p)                # rollout probability, computed naively
    logsum = np.log(tok_p).sum()           # same quantity, log lens
    ```
    """
        + f"""
    ```
    slope_f / f(t0) = {_slope_f / _f(_t0):.9f}
    slope_logf      = {_slope_logf:.9f}          # identity confirmed

    prod   = {_prod:.6g}                      # underflowed to nothing
    logsum = {_logsum:.3f}                        # perfectly healthy
    ```

    The product of 200 modest probabilities is `0.0` in float64 — the number
    exists, the representation doesn't. Its log is a routine −230. Every loss in
    `train.py` is a *sum of per-token log-probs* for exactly this reason.
    """
    )
    return


@app.cell
def _():
    _big = np.array([1000.0, 1001.0, 1002.0])
    with np.errstate(over="ignore", invalid="ignore"):
        _naive = np.exp(_big) / np.exp(_big).sum()
    _stable = ref_softmax(_big)
    mo.md(
        r"""
    ### Stability teaser: logsumexp

    Same failure, opposite direction. Softmax is \( e^{z_i} / \sum_j e^{z_j} \);
    feed it logits around 1000 — unremarkable for an unnormalized network — and
    \( e^{1000} \) overflows to `inf`, giving `inf/inf = nan`:

    ```python
    z = np.array([1000., 1001., 1002.])
    naive  = np.exp(z) / np.exp(z).sum()          # overflow
    stable = np.exp(z - z.max()) / np.exp(z - z.max()).sum()
    ```
    """
        + """
    ```
    naive  = """ + np.array2string(_naive, precision=3) + """
    stable = """ + np.array2string(_stable, precision=3) + """
    ```

    Subtracting the max first is legal by a shift-invariance you'll prove in
    exercise 2.4, and it makes every exponent ≤ 0 — nothing can blow up. Every
    softmax in this project (and inside every transformer you've ever used) does
    this dance; the same idea under the name **logsumexp** computes
    \\( \\log \\sum_i e^{z_i} \\) without ever exponentiating anything big.
    """
    )
    return


@app.cell
def _():
    mo.vstack([
        mo.md(r"""
    ### Exercises 2

    **2.1 (warm-up).** Expand \( \log\!\left( \dfrac{a\,b^2}{c} \right) \) into
    simple log terms.

    **2.2 (symbols → numpy).** Differentiate \( \log(x^2 + 1) \) using
    \( (\log f)' = f'/f \). Then write three lines of numpy that verify your answer
    at \( x = 2 \) by finite difference.

    **2.3 (guided, the important one).** Let
    \( p = \operatorname{softmax}(\theta) \), \( \theta \in \mathbb{R}^3 \), so
    \( p_i = e^{\theta_i} / \sum_k e^{\theta_k} \). Show
    \[ \frac{\partial}{\partial \theta_j} \log p_i \;=\; \delta_{ij} - p_j,
       \qquad \text{i.e.} \qquad \nabla_\theta \log p_i = e_i - p . \]
    *Hint: write \( \log p_i = \theta_i - \log \sum_k e^{\theta_k} \) first.*

    **2.4 (stinger).** Prove softmax is shift-invariant:
    \( \operatorname{softmax}(z - m) = \operatorname{softmax}(z) \) for any
    constant \( m \) — then say in one sentence why \( m = \max_i z_i \) makes the
    computation safe.
    """),
        mo.accordion({
            "Solution 2.1": mo.md(r"""
    Products → sums, powers → multipliers, quotients → differences:
    \[ \log\frac{a b^2}{c} = \log a + 2 \log b - \log c. \]
    This is why a 200-token rollout's log-probability is a *sum* of 200 per-token
    terms — the form every loss in the project takes.
    """),
            "Solution 2.2": mo.md(r"""
    \( f = x^2 + 1 \), \( f' = 2x \), so
    \[ \frac{d}{dx} \log(x^2+1) = \frac{2x}{x^2+1} \;\xrightarrow{\,x=2\,}\; \frac{4}{5} = 0.8. \]

    ```python
    g = lambda x: np.log(x**2 + 1)
    fd = (g(2 + 1e-6) - g(2 - 1e-6)) / 2e-6   # → 0.800000000
    print(fd, 2*2 / (2**2 + 1))               # both 0.8
    ```

    Note the shape of the answer: raw gradient \( 2x \), *deflated by the current
    height* \( x^2{+}1 \). Log-gradients always measure **relative** change.
    """),
            "Solution 2.3": mo.md(r"""
    From the hint:
    \[ \log p_i = \theta_i - \log \sum_k e^{\theta_k}. \]
    Differentiate w.r.t. \( \theta_j \). First term: \( \delta_{ij} \) (1 iff
    \( i = j \)). Second term, chain rule with outer \( \log u \), inner
    \( u = \sum_k e^{\theta_k} \):
    \[ \frac{\partial}{\partial \theta_j} \log \sum_k e^{\theta_k}
       = \frac{e^{\theta_j}}{\sum_k e^{\theta_k}} = p_j
       \qquad\Longrightarrow\qquad
       \frac{\partial \log p_i}{\partial \theta_j} = \delta_{ij} - p_j. \]
    Geometric sanity checks: the entries sum to \( 1 - \sum_j p_j = 0 \) (pushing
    one logit up must pull the others' shares down — probability is a zero-sum
    simplex); and if \( p_i \to 1 \) the gradient → 0 (you cannot get more
    confident than certain). Numpy check, four lines:

    ```python
    th, i, h = np.array([0.4, -0.2, 0.1]), 0, 1e-6
    p  = np.exp(th) / np.exp(th).sum()
    fd = [(np.log(np.exp(th + h*np.eye(3)[j]) / np.exp(th + h*np.eye(3)[j]).sum())[i]
           - np.log(p[i])) / h for j in range(3)]
    print(np.round(fd, 6), np.round(np.eye(3)[i] - p, 6))   # identical
    ```
    """),
            "Solution 2.4": mo.md(r"""
    \[ \frac{e^{z_i - m}}{\sum_j e^{z_j - m}}
       = \frac{e^{z_i}\, e^{-m}}{e^{-m} \sum_j e^{z_j}}
       = \frac{e^{z_i}}{\sum_j e^{z_j}} . \]
    The \( e^{-m} \) factors out of the sum and cancels — sliding all logits
    together moves nothing, because softmax only ever sees *differences*. With
    \( m = \max_i z_i \) every exponent is ≤ 0, so every \( e^{z_i - m} \in (0, 1] \):
    no overflow, and the denominator contains a term equal to 1, so no division by
    zero either.
    """),
        }),
        mo.md(
            r"**Where you'll use this:** \( \nabla f = f\,\nabla\!\log f \) *is* the "
            r"log-derivative trick that makes REINFORCE possible (grpo_notes §1), and "
            r"2.3's softmax gradient \( e_a - p \) appears verbatim in its bandit code "
            r"as `np.eye(3)[a] - probs`."
        ),
    ], gap=1)
    return


@app.cell
def _():
    mo.md(r"""
    ## 3 · Vectors, dot products, gradients

    ### The picture

    A dot product measures **agreement between directions**. Below, \( a \)
    (black, fixed scenery) stays put while \( b \) (orange — you drag it)
    rotates. Project \( b \) onto \( a \)'s line — drop the perpendicular, take
    the **shadow** (blue: the quantity being computed). The dot product is
    **(length of \( a \)) × (signed length of the shadow)**: biggest when aligned,
    zero at 90° (no shadow — nothing to say to each other), negative past 90°
    (the shadow points backwards). Divide out both lengths and only the angle
    survives: cosine similarity, pure agreement on a −1..1 scale.
    """)
    return


@app.cell
def _():
    r3_phi = mo.ui.slider(0, 360, step=5, value=40, label="direction of b (degrees)")
    r3_phi
    return (r3_phi,)


@app.cell
def _(r3_phi):
    plt.close("all")
    _a = np.array([2.0, 1.0])
    _phi = np.deg2rad(r3_phi.value)
    _b = 1.5 * np.array([np.cos(_phi), np.sin(_phi)])
    _dot = float(_a @ _b)
    _cos = _dot / (np.linalg.norm(_a) * np.linalg.norm(_b))
    _shadow = (_dot / (_a @ _a)) * _a   # projection of b onto a

    _fig, _ax = plt.subplots(figsize=(7.2, 4.1), layout="constrained")
    _ax.axhline(0, color=C_FAINT, lw=0.8)
    _ax.axvline(0, color=C_FAINT, lw=0.8)
    _ua = _a / np.linalg.norm(_a)
    _ax.plot([-2.6 * _ua[0], 2.6 * _ua[0]], [-2.6 * _ua[1], 2.6 * _ua[1]],
             color=C_FAINT, lw=1, ls=":")
    _ax.annotate("", xy=_a, xytext=(0, 0),
                 arrowprops=dict(arrowstyle="->", color=C_INK, lw=2.2))
    _ax.annotate("", xy=_b, xytext=(0, 0),
                 arrowprops=dict(arrowstyle="->", color=C_ORANGE, lw=2.2))
    _ax.plot([_b[0], _shadow[0]], [_b[1], _shadow[1]], color=C_MUTED, lw=1.1, ls="--")
    _ax.plot([0, _shadow[0]], [0, _shadow[1]], color=C_BLUE, lw=4, alpha=0.85,
             solid_capstyle="round")
    _ax.annotate("a", _a * 1.08, color=C_INK, fontsize=12, fontweight="bold")
    _ax.annotate("b", _b * 1.12, color=C_ORANGE, fontsize=12, fontweight="bold")
    _ax.annotate("shadow = a·b / ‖a‖", _shadow * 0.5 + np.array([0.06, -0.25]),
                 color=C_BLUE, fontsize=9)
    _ax.set(
        xlim=(-2.4, 2.9), ylim=(-2.0, 2.2), aspect="equal", xlabel="x", ylabel="y",
        title=f"a·b = {_dot:+.2f}    cos∠(a,b) = {_cos:+.2f}    "
        f"signed shadow length = {_dot / np.linalg.norm(_a):+.2f}",
    )
    _fig
    return


@app.cell
def _():
    mo.md(r"""
    ### The notation, pinned to the picture

    \[ a \cdot b \;=\; \underbrace{\sum_i a_i b_i}_{\text{how you compute it}}
       \;=\; \underbrace{\|a\|}_{\text{length of } a}
       \underbrace{\|b\| \cos \angle(a, b)}_{\text{signed shadow of } b \text{ on } a} ,
    \qquad
    \cos \angle(a, b) \;=\; \frac{a \cdot b}{\|a\|\,\|b\|} . \]

    The left form is the code (`a @ b`); the right form is the picture; the
    equals sign between them is the useful theorem. File the cosine away with
    intent: the fruit-catch project's endgame is to take the model's internal
    vector for the token "banana" and ask how much it **agrees** with the vector
    for "orange" — literally this formula, in 1024 dimensions instead of 2. Same
    shadow, more axes.

    ### The gradient is an arrow, and it points uphill

    For \( f \) of many inputs, stack all the partial sensitivities from §1 into a
    vector: \( \nabla f = \left( \frac{\partial f}{\partial x},
    \frac{\partial f}{\partial y}, \dots \right) \). Its defining property — worth
    pinning to the picture below — is that a small step \( \Delta \) changes the
    altitude by a **dot product with the gradient**:

    \[ f(p + \Delta) - f(p) \;\approx\;
       \underbrace{\nabla f(p)}_{\text{the arrow at } p}
       \cdot
       \underbrace{\Delta}_{\text{your step}} . \]

    Which step direction gains the most altitude? The one whose *shadow on the
    gradient* is longest — the gradient's own direction. **The gradient is the
    locally steepest-ascent arrow.** Minimizing a loss means stepping against it
    (descent); but RL *maximizes* expected reward, so every update in this project
    steps **with** it — gradient **ascent**, `θ += lr * grad`, plus sign. It reads
    wrong to supervised-learning eyes until this picture is loaded.

    Below: the hill \( f(x, y) = -(x^2 + 2y^2) \), peak at the origin (★). The
    grey arrows are \( \nabla f \) on a grid — all pointing uphill, growing with
    steepness. The **steps taken** slider is the film: scrub it and walk the
    orange ball uphill one gradient step at a time. Then the experiment: the
    \( y \)-curvature is twice the \( x \)-curvature, so the safe step size is set
    by the *steepest* direction — crank \( \eta \) past 0.5, scrub the walk
    again, and watch \( y \) ping-pong across the ridge while \( x \) is still
    fine.
    """)
    return


@app.cell
def _():
    r3_eta = mo.ui.slider(0.02, 0.62, step=0.02, value=0.10, label="step size η")
    r3_k = mo.ui.slider(0, 12, step=1, value=4, label="steps taken (scrub to walk)")
    mo.hstack([r3_eta, r3_k], justify="start", gap=2)
    return r3_eta, r3_k


@app.cell
def _(r3_eta, r3_k):
    plt.close("all")
    _eta, _k = r3_eta.value, r3_k.value
    _gx, _gy = np.meshgrid(np.linspace(-2.3, 2.3, 200), np.linspace(-1.7, 1.7, 200))
    _Z = -(_gx**2 + 2 * _gy**2)

    _pts = [np.array([-1.8, 1.2])]
    for _ in range(12):
        _p = _pts[-1]
        _pts.append(_p + _eta * np.array([-2 * _p[0], -4 * _p[1]]))
    _pts = np.array(_pts)

    _fig, _ax = plt.subplots(figsize=(8.5, 4.2), layout="constrained")
    _ax.contour(_gx, _gy, _Z, levels=14, colors=C_FAINT, linewidths=0.9)
    _qx, _qy = np.meshgrid(np.linspace(-2, 2, 9), np.linspace(-1.5, 1.5, 7))
    _ax.quiver(
        _qx, _qy, -2 * _qx, -4 * _qy, color=C_MUTED, alpha=0.55,
        angles="xy", scale=42, width=0.0035,
    )
    _ax.annotate("∇f", (1.62, 1.28), color=C_MUTED, fontsize=10)
    _ax.plot(_pts[: _k + 1, 0], _pts[: _k + 1, 1], color=C_ORANGE, lw=1.6,
             marker="o", ms=4, zorder=5)
    _ax.scatter([_pts[_k, 0]], [_pts[_k, 1]], color=C_ORANGE, s=110, zorder=6)
    _ax.annotate("you", _pts[_k] + np.array([0.07, 0.07]), color=C_ORANGE, fontsize=9.5)
    _ax.scatter([_pts[0, 0]], [_pts[0, 1]], color=C_MUTED, s=45, zorder=5)
    _ax.annotate("start", _pts[0] + np.array([-0.32, 0.09]), color=C_MUTED, fontsize=9)
    _ax.scatter([0], [0], color=C_INK, marker="*", s=140, zorder=6)
    _ax.annotate("peak", (0.08, 0.06), color=C_INK, fontsize=9)
    _f_now = -(_pts[_k, 0] ** 2 + 2 * _pts[_k, 1] ** 2)
    _ax.set(
        xlabel="x", ylabel="y", xlim=(-2.3, 2.3), ylim=(-1.7, 1.7),
        title=f"gradient ascent on f = −(x² + 2y²), η = {_eta:.2f} :  "
        f"after {_k} steps,  f = {_f_now:+.4f}"
        + ("   ← overshooting in y!" if _eta > 0.5 else ""),
    )
    _ax.title.set_color(C_ORANGE if _eta > 0.5 else C_INK)
    _fig
    return


@app.cell
def _():
    _a = np.array([1.0, 2.0, 2.0])
    _b = np.array([2.0, 0.0, 1.0])
    _dot = float(_a @ _b)
    _cos = _dot / (np.linalg.norm(_a) * np.linalg.norm(_b))

    _f = lambda p: -(p[0] ** 2 + 2 * p[1] ** 2)
    _p = np.array([1.0, 1.0])
    _d = np.array([0.01, -0.02])
    _grad = np.array([-2 * _p[0], -4 * _p[1]])
    _actual = _f(_p + _d) - _f(_p)
    _predicted = float(_grad @ _d)

    mo.md(
        r"""
    ### The same facts in numpy

    ```python
    a, b = np.array([1., 2., 2.]), np.array([2., 0., 1.])
    dot  = a @ b                                          # sum of products
    cos  = dot / (np.linalg.norm(a) * np.linalg.norm(b))  # shadow / lengths

    f    = lambda p: -(p[0]**2 + 2*p[1]**2)               # the hill
    p, d = np.array([1., 1.]), np.array([0.01, -0.02])    # point, small step
    grad = np.array([-2*p[0], -4*p[1]])                   # the arrow at p
    actual    = f(p + d) - f(p)                           # altitude change, measured
    predicted = grad @ d                                  # ∇f · Δ, the dot product claim
    ```
    """
        + f"""
    ```
    dot = {_dot:.1f}    cos = {_cos:.4f}
    actual    = {_actual:+.6f}
    predicted = {_predicted:+.6f}      # ∇f·Δ nails the altitude change
    ```

    That last comparison is the gradient's *definition* doing its job: one dot
    product predicts what any small step does to the function. Every optimizer
    ever written is a bet that this local prediction is worth following.
    """
    )
    return


@app.cell
def _():
    mo.vstack([
        mo.md(r"""
    ### Exercises 3

    **3.1 (warm-up).** For \( a = (1, 2, 2) \), \( b = (2, 0, 1) \): compute
    \( a \cdot b \), \( \|a\| \), \( \|b\| \), and the cosine — by hand, then check
    against the numpy above.

    **3.2.** One step of gradient **ascent** on \( f(x, y) = -(x^2 + 2y^2) \) from
    \( (1, 1) \), step size \( \eta = 0.25 \). New point, and did \( f \) improve?

    **3.3 (stinger).** On \( f(x) = -x^2 \), ascent iterates
    \( x \leftarrow x + \eta(-2x) = (1 - 2\eta)\,x \). For which \( \eta \) does it
    (a) converge smoothly, (b) converge while ping-ponging across the peak, (c)
    diverge? Reconcile with what the slider did to the demo's \( y \)-coordinate.

    **3.4 (translate: numpy → symbols → picture).** A line from the project's
    analysis code: `v @ w / (np.linalg.norm(v) * np.linalg.norm(w))`. (a) Write it
    in symbols. (b) Describe the picture it measures. (c) What value does it take
    if \( w = -v \)? If \( w \perp v \)?
    """),
        mo.accordion({
            "Solution 3.1": mo.md(r"""
    \( a \cdot b = 1{\cdot}2 + 2{\cdot}0 + 2{\cdot}1 = 4 \);
    \( \|a\| = \sqrt{1+4+4} = 3 \); \( \|b\| = \sqrt{5} \approx 2.236 \);
    \[ \cos = \frac{4}{3\sqrt{5}} \approx 0.596 . \]
    Moderate agreement — about 53° apart, shadow a bit over half of \( b \)'s
    length. Matches the numpy cell to the digit.
    """),
            "Solution 3.2": mo.md(r"""
    \( \nabla f = (-2x, -4y) \big|_{(1,1)} = (-2, -4) \). Ascent:
    \( (1,1) + 0.25 \cdot (-2, -4) = (0.5,\, 0) \).
    \( f(1,1) = -3 \to f(0.5, 0) = -0.25 \): much higher. Note the arithmetic said
    "add the gradient" yet the step moved *toward the origin* — on the side of a
    hill, uphill points inward. In numpy: `p = p + 0.25 * grad`.
    """),
            "Solution 3.3": mo.md(r"""
    \( x_k = (1 - 2\eta)^k x_0 \), so everything hangs on \( c = 1 - 2\eta \):

    - (a) \( 0 < c < 1 \iff \eta < \tfrac12 \): smooth exponential approach.
    - (b) \( -1 < c < 0 \iff \tfrac12 < \eta < 1 \): converges, alternating sign —
      hopping over the peak each step.
    - (c) \( |c| > 1 \iff \eta > 1 \): diverges.

    The demo's \( y \)-direction has curvature 4 (factor \( 1 - 4\eta \)), so its
    thresholds sit at \( \eta = 0.25 \) and \( 0.5 \) — exactly where the
    trajectory started ping-ponging vertically while \( x \) stayed calm. The
    steepest direction sets the ceiling on the learning rate for *all* directions.
    That is the entire personality of the `lr` knob in `train.py`.
    """),
            "Solution 3.4": mo.md(r"""
    (a) \( \dfrac{v \cdot w}{\|v\|\,\|w\|} = \cos \angle(v, w) \) — cosine
    similarity. (b) The picture: rotate to the plane spanned by \( v \) and
    \( w \); this is the shadow of one unit arrow on the other — pure direction
    agreement, lengths divided out. (c) \( w = -v \): −1 (perfect disagreement).
    \( w \perp v \): 0 (no shadow, orthogonal, nothing shared). This exact line,
    run on the model's embedding vectors for *banana* and *orange*, is how the
    project will test whether transfer had a geometric substrate.
    """),
        }),
        mo.md(
            r"**Where you'll use this:** every update in grpo_notes and `train.py` "
            r"is gradient *ascent* on \( \mathbb{E}[R] \), and the project's final "
            r"question — did banana inherit orange's behavior? — is answered with "
            r"3.4's cosine between embedding vectors."
        ),
    ], gap=1)
    return


@app.cell
def _():
    mo.md(r"""
    ## 4 · Expectation: the weighted average

    ### The picture

    Put probability-weights on a massless ruler, one at each outcome's position.
    The expected value is **where the ruler balances** — the center of mass.
    Below: a reward distribution from a catch-like world (usually 0, sometimes
    0.5, rarely 1); the balance point sits at 0.25, dragged toward the heavy pile
    at zero. Note that 0.25 is *not a possible outcome* — the same way a fair
    die balances at 3.5, a face it doesn't have.
    """)
    return


@app.cell
def _():
    plt.close("all")
    _vals = np.array([0.0, 0.5, 1.0])
    _probs = np.array([0.6, 0.3, 0.1])
    _E = float(_vals @ _probs)

    r4_fig, _axm = plt.subplots(figsize=(7.6, 3.1), layout="constrained")
    _axm.bar(_vals, _probs, width=0.09, color=C_BLUE)
    for _v, _pr in zip(_vals, _probs):
        _axm.annotate(f"P = {_pr:.1f}", (_v, _pr + 0.02), ha="center",
                      color=C_BLUE, fontsize=9)
    _axm.plot([-0.12, 1.12], [0, 0], color=C_INK, lw=2.5)
    _axm.scatter([_E], [-0.045], marker="^", s=170, color=C_ORANGE, zorder=5, clip_on=False)
    _axm.annotate(f"balance point  E[R] = {_E:.2f}", (_E, -0.115), ha="center",
                  color=C_ORANGE, fontsize=10, annotation_clip=False)
    _axm.set(xlabel="reward r", ylabel="P(r)", ylim=(-0.14, 0.78),
             title="probability as mass on a ruler — E[R] is where it balances")
    _axm.grid(False)
    r4_fig
    return


@app.cell
def _():
    mo.md(r"""
    ### The notation, pinned to the picture

    \[ \mathbb{E}[X] \;=\; \underbrace{\sum_x}_{\text{over the bars}}
       \underbrace{x}_{\text{bar's position}} \,
       \underbrace{P(x)}_{\text{bar's height}}
    \qquad\text{or, continuous,}\qquad
    \mathbb{E}[X] = \int x \, p(x) \, dx \]

    — position × mass, summed: the center-of-mass formula, verbatim. (The integral
    is the same recipe with infinitely thin bars.) Two properties do nearly all the
    work in this project:

    **E of a constant is the constant:** \( \mathbb{E}[c] = c \). All the mass at
    one point balances at that point. This is the licence for pulling
    action-independent things out of expectations.

    **Linearity — needing *no independence*:** for *any* random variables, however
    correlated or adversarially entangled,

    \[ \mathbb{E}[aX + bY] \;=\; a\,\mathbb{E}[X] + b\,\mathbb{E}[Y]. \]

    This is worth staring at, because intuition keeps whispering "…if X and Y are
    independent," and intuition is wrong. Products need independence
    (\( \mathbb{E}[XY] \neq \mathbb{E}[X]\,\mathbb{E}[Y] \) in general); **sums
    never do**, because an expectation is itself a sum, and sums slide through
    sums unconditionally.

    ### Monte Carlo: when you can't do the sum, sample it

    For an LLM policy, \( \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] \) is a sum
    over every possible transcript — more terms than atoms. But you don't need the
    sum, only its value:

    \[ \hat{\mu}_n \;=\; \underbrace{\frac{1}{n} \sum_{i=1}^n X_i}_{\text{average of } n \text{ samples}}
       \;\longrightarrow\; \mathbb{E}[X],
    \qquad
    \text{typical error} \;\sim\; \underbrace{\frac{\sigma}{\sqrt{n}}}_{\text{the funnel below}} . \]

    In the demo, the jagged line is \( \hat\mu_n \) as rolls accumulate; the blue
    funnel is \( 3.5 \pm \sigma/\sqrt{n} \). The estimate is unbiased at every
    \( n \) (exercise 4.3), and the funnel narrows like \( 1/\sqrt{n} \) —
    **slowly**: 10× the accuracy costs 100× the samples. That brutal exchange rate
    is why so much of RL is variance management.
    """)
    return


@app.cell
def _():
    r4_n = mo.ui.slider(10, 3000, step=10, value=300, label="number of rolls n")
    r4_redraw = mo.ui.button(value=0, on_click=lambda value: value + 1, label="new dice")
    mo.hstack([r4_n, r4_redraw], justify="start", gap=2)
    return r4_n, r4_redraw


@app.cell
def _(r4_n, r4_redraw):
    plt.close("all")
    _rng = np.random.default_rng(40 + r4_redraw.value)
    _n = r4_n.value
    _rolls = _rng.integers(1, 7, _n)
    _k = np.arange(1, _n + 1)
    _running = np.cumsum(_rolls) / _k
    _sigma = np.sqrt(35 / 12)  # std of a fair die

    _fig, _ax = plt.subplots(figsize=(8.5, 3.5), layout="constrained")
    _ax.fill_between(
        _k, 3.5 - _sigma / np.sqrt(_k), 3.5 + _sigma / np.sqrt(_k),
        color=C_BLUE, alpha=0.12, lw=0,
    )
    _ax.plot(_k, _running, color=C_ORANGE, lw=1.6)
    _ax.axhline(3.5, color=C_INK, ls="--", lw=1.3)
    _ax.annotate("E[X] = 3.5", (_k[-1], 3.53), color=C_INK, fontsize=9,
                 ha="right", va="bottom")
    _ax.annotate("the funnel: 3.5 ± σ/√n",
                 (_k[-1], 3.5 + _sigma / np.sqrt(_k[-1]) + 0.09),
                 color=C_BLUE, fontsize=9, ha="right")
    _i0 = min(2, _n - 1)
    _ax.annotate("running mean μ̂ₙ", (_k[_i0], _running[_i0]),
                 textcoords="offset points", xytext=(8, 10),
                 color=C_ORANGE, fontsize=9)
    _ax.set(
        xlabel="rolls so far (log scale)", ylabel="mean of rolls", ylim=(2.3, 4.7),
        xscale="log",
        title=f"after {_n} rolls: mean = {_running[-1]:.3f}   "
        f"(theory: typical error ≈ {_sigma / np.sqrt(_n):.3f})",
    )
    _fig
    return


@app.cell
def _():
    _faces = np.arange(1, 7)
    _exact = float(_faces.mean())
    _rng = np.random.default_rng(4)
    _sample = _rng.integers(1, 7, 100_000).mean()

    _vals = np.array([0.0, 0.5, 1.0])
    _probs = np.array([0.6, 0.3, 0.1])
    _Eexact = float(_vals @ _probs)
    _Esample = float(_rng.choice(_vals, 100_000, p=_probs).mean())

    mo.md(
        r"""
    ### The same facts in numpy

    Two roads to the same number — the weighted sum (the formula) and the sample
    mean (Monte Carlo):

    ```python
    faces = np.arange(1, 7)
    E_die_exact  = (faces * (1/6)).sum()                        # Σ x·P(x)
    E_die_sample = rng.integers(1, 7, 100_000).mean()           # 1/n Σ xᵢ

    vals, probs   = np.array([0., .5, 1.]), np.array([.6, .3, .1])
    E_r_exact     = vals @ probs                                # note: a dot product!
    E_r_sample    = rng.choice(vals, 100_000, p=probs).mean()
    ```
    """
        + f"""
    ```
    E_die_exact = {_exact:.4f}     E_die_sample = {_sample:.4f}
    E_r_exact   = {_Eexact:.4f}     E_r_sample   = {_Esample:.4f}
    ```

    Notice `vals @ probs`: **an expectation is literally a dot product** between
    the outcome vector and the probability vector — section 3 and section 4 are
    the same machinery wearing different clothes.
    """
    )
    return


@app.cell
def _():
    mo.vstack([
        mo.md(r"""
    ### Exercises 4

    **4.1 (warm-up).** A reward is 0 with probability 0.5, 0.5 with probability
    0.25, and 1 with probability 0.25. Compute \( \mathbb{E}[R] \), and say where
    the fulcrum sits relative to the picture's 0.25 and why.

    **4.2 (linearity puzzle).** Roll one die; let \( X \) be the top face and
    \( Y = 7 - X \) the *bottom* face — perfectly dependent, knowing one fixes the
    other. What is \( \mathbb{E}[X + Y] \), and what did the calculation *not*
    require?

    **4.3.** Show the Monte Carlo estimator \( \hat\mu_n = \frac{1}{n}\sum_i X_i \)
    is unbiased: \( \mathbb{E}[\hat\mu_n] = \mu \) exactly, for every \( n \), even
    \( n = 1 \).

    **4.4 (translate: code → symbols → picture).** In `train.py`, the line
    `baseline = rewards.mean()` runs on the \( G = 8 \) rewards of one group.
    (a) Which mathematical object is it *estimating*, in symbols? (b) Which picture
    from this section describes its error, and how big is that error roughly?
    """),
        mo.accordion({
            "Solution 4.1": mo.md(r"""
    \[ \mathbb{E}[R] = 0 \cdot 0.5 + 0.5 \cdot 0.25 + 1 \cdot 0.25 = 0.375. \]
    Fulcrum to the *right* of the picture's 0.25: this distribution moved mass from
    the pile at zero out to 1, so the ruler balances further out. In numpy:
    `np.array([0, .5, 1]) @ np.array([.5, .25, .25])`.
    """),
            "Solution 4.2": mo.md(r"""
    Linearity: \( \mathbb{E}[X+Y] = \mathbb{E}[X] + \mathbb{E}[Y] = 3.5 + 3.5 = 7 \)
    (also obvious directly — top plus bottom of a die is always 7). The
    calculation never asked whether \( X \) and \( Y \) were independent; here they
    are *maximally* dependent and linearity doesn't blink. Any proof built purely
    from linearity is immune to correlations — a superpower in RL, where
    everything is correlated with everything.
    """),
            "Solution 4.3": mo.md(r"""
    Linearity, and nothing else:
    \[ \mathbb{E}\!\left[\frac{1}{n}\sum_i X_i\right]
       = \frac{1}{n}\sum_i \mathbb{E}[X_i] = \frac{1}{n} \cdot n\mu = \mu. \]
    Unbiased means *centered on the truth* — it says nothing about spread. A
    single die roll is an unbiased estimate of 3.5 and also a terrible one. The
    gap between "unbiased" and "useful" is entirely a variance story: next section.
    """),
            "Solution 4.4": mo.md(r"""
    (a) It estimates \( \mathbb{E}_{\tau \sim \pi_\theta}[R \mid s] \) — the
    expected reward *from this specific initial state* — via
    \( \hat\mu_8 = \frac{1}{8}\sum_i r_i \). (b) The running-mean funnel, frozen at
    \( n = 8 \): the estimate is unbiased but sits inside an error band of width
    \( \sim \sigma/\sqrt{8} \approx 0.35\,\sigma \). For binary rewards with
    \( p = 0.5 \), that's about ±0.18 — a noisy baseline, and the price GRPO pays
    for not training a critic network. (Why noisy-but-unbiased is an acceptable
    trade is §5 and grpo_notes §2–3.)
    """),
        }),
        mo.md(
            r"**Where you'll use this:** the objective itself is "
            r"\( J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)] \); every "
            r"gradient in the project is a Monte Carlo average over \( G \) rollouts; "
            r"and the baseline proof in grpo_notes §2 is one long exercise in "
            r"linearity."
        ),
    ], gap=1)
    return


@app.cell
def _():
    mo.md(r"""
    ## 5 · Variance, and why it ruins everything

    ### The picture

    Two estimators of the same quantity (truth = 0.9, the dashed line). Each
    histogram below is "what answer would an \( n \)-sample experiment report,"
    replayed 6000 times. Both are **centered on the truth at every \( n \)** —
    unbiased — but one is a tight spike and the other a sprawl. The default
    \( n = 8 \) is not an accident: it is the group size in `train.py`. Look at
    how often the orange estimator lands on the wrong side of **zero** — in policy
    gradients, the sign of the estimate is the *direction of the update*.
    """)
    return


@app.cell
def _():
    r5_n = mo.ui.slider(1, 200, step=1, value=8, label="samples per experiment n")
    r5_n
    return (r5_n,)


@app.cell
def _(r5_n):
    plt.close("all")
    _rng = np.random.default_rng(5)
    _n = r5_n.value
    _trials = 6000
    _mu = 0.9
    _means_hi = _rng.normal(_mu, 0.5, (_trials, _n)).mean(axis=1)
    _means_lo = _rng.normal(_mu, 0.05, (_trials, _n)).mean(axis=1)

    _fig, _ax = plt.subplots(figsize=(8.5, 3.6), layout="constrained")
    _bins = np.linspace(_mu - 1.1, _mu + 1.1, 121)
    _ax.hist(
        _means_hi, bins=_bins, color=C_ORANGE, alpha=0.55,
        label=f"estimator A: σ = 0.5   (SE = {0.5 / np.sqrt(_n):.3f})",
    )
    _ax.hist(
        _means_lo, bins=_bins, color=C_BLUE, alpha=0.65,
        label=f"estimator B: σ = 0.05  (SE = {0.05 / np.sqrt(_n):.3f})",
    )
    _ax.axvline(_mu, color=C_INK, ls="--", lw=1.5, label="truth = 0.9")
    _ax.axvline(0, color=C_MUTED, ls=":", lw=1.2)
    _ax.annotate(" 0 — sign flips here", (0, _trials * 0.035), color=C_MUTED, fontsize=8.5)
    _ax.set(
        xlabel="value reported by an n-sample experiment", ylabel="count",
        title=f"same mean, different σ — {_trials} replays of an n = {_n} experiment. "
        f"A lands below zero {100 * (_means_hi < 0).mean():.1f}% of the time",
    )
    _ax.legend(frameon=False, fontsize=9)
    _fig
    return


@app.cell
def _():
    mo.md(r"""
    ### The notation, pinned to the picture

    \[ \operatorname{Var}[X] \;=\;
       \mathbb{E}\big[\underbrace{(X - \mathbb{E}[X])^2}_{\substack{\text{horizontal distance from} \\ \text{the dashed line, squared}}}\big],
    \qquad
    \sigma = \sqrt{\operatorname{Var}[X]}
       \;=\; \text{the histogram's width, in original units} . \]

    Averaging \( n \) independent samples shrinks the width by \( \sqrt{n} \) —
    that's the **standard error of the mean**, and it is what your slider
    controls:

    \[ \operatorname{SE} \;=\; \frac{\sigma}{\sqrt{n}} . \]

    The fact this section exists for: **two estimators can have identical means
    and wildly different usefulness.** Unbiasedness is table stakes; variance
    decides what you can afford at the sample sizes you actually have. An
    unbiased, high-variance gradient estimator doesn't fail on average — it fails
    on *every individual step* while being right in expectation, which your
    optimizer experiences as noise it must average away over many updates. Halving
    σ is worth exactly as much as buying 4× the samples. That equivalence is the
    entire economic case for baselines.
    """)
    return


@app.cell
def _():
    _rng = np.random.default_rng(55)
    _x = _rng.normal(0.9, 0.5, 8)
    _mean, _std = _x.mean(), _x.std()
    _se = _std / np.sqrt(8)
    _y = 3 * _x + 7
    _ratio = _y.var() / _x.var()

    mo.md(
        r"""
    ### The same facts in numpy

    ```python
    x = rng.normal(0.9, 0.5, 8)        # one 8-sample experiment with estimator A
    x.mean()                           # its report — one draw from the orange histogram
    x.std()                            # σ̂: histogram width
    x.std() / np.sqrt(8)               # SE: how far the report typically sits from 0.9

    y = 3*x + 7                        # scale by a=3, shift by b=7
    y.var() / x.var()                  # → a² = 9, and the +7 vanished (exercise 5.1)
    ```
    """
        + f"""
    ```
    x.mean() = {_mean:.4f}      x.std() = {_std:.4f}      SE = {_se:.4f}
    y.var() / x.var() = {_ratio:.4f}
    ```

    This particular experiment reported {_mean:.2f} for a true value of 0.90 —
    off by {abs(_mean - 0.9):.2f}, about one SE, exactly as the funnel predicted.
    An honest estimator can still be a bad day.
    """
    )
    return


@app.cell
def _():
    mo.vstack([
        mo.md(r"""
    ### Exercises 5

    **5.1.** From the definition
    \( \operatorname{Var}[X] = \mathbb{E}[(X - \mathbb{E}X)^2] \), show
    \( \operatorname{Var}[aX + b] = a^2 \operatorname{Var}[X] \). Then interpret
    the \( a^2 \) and the vanished \( b \) in one sentence each — the vanished
    \( b \) is about to become load-bearing.

    **5.2.** Your reward estimates have \( \sigma = 0.4 \). What is the standard
    error of a mean over \( n = 64 \) rollouts? How many rollouts to halve it?

    **5.3 (conceptual — this IS the baseline story).** Every trajectory in a batch
    earns a reward between 0.8 and 1.0. REINFORCE weights each trajectory's
    \( \nabla \log \pi \) by its reward. What goes wrong if you push up the
    probability of *all* of them — and what single subtraction fixes it without
    changing the expected gradient?
    """),
        mo.accordion({
            "Solution 5.1": mo.md(r"""
    Let \( \mu = \mathbb{E}[X] \). Linearity gives
    \( \mathbb{E}[aX + b] = a\mu + b \), so
    \[ \operatorname{Var}[aX+b] = \mathbb{E}\big[(aX + b - a\mu - b)^2\big]
       = \mathbb{E}\big[a^2 (X-\mu)^2\big] = a^2 \operatorname{Var}[X]. \]
    The \( a^2 \): variance lives in squared units, so scaling the data by
    \( a \) scales the spread by \( a^2 \) (the numpy cell measured exactly 9 for
    \( a = 3 \)). The vanished \( b \): **shifting a distribution slides the
    histogram without changing its width** — which is precisely why subtracting a
    baseline from rewards may change variance while it *provably cannot* change
    anything else.
    """),
            "Solution 5.2": mo.md(r"""
    \( \operatorname{SE} = 0.4/\sqrt{64} = 0.05 \). Halving it needs
    \( \sqrt{n} \) twice as large: \( n = 256 \), four times the rollouts. The
    square root always charges quadratically for precision — with LLM rollouts at
    real GPU prices, variance reduction by *algebra* (baselines) is massively
    cheaper than variance reduction by *sampling*.
    """),
            "Solution 5.3": mo.md(r"""
    With all rewards in \( [0.8, 1.0] \), every sample says "push this
    trajectory's probability up" — including the *worst* one in the batch. The
    actual information ("0.8 was worse than 1.0") is a whisper riding on a shout
    of common offset; and since probabilities sum to 1, pushing everything up
    mostly cancels into noise. Still unbiased — but the variance is enormous, and
    any finite batch mostly reinforces whatever it happened to sample.

    Subtract the batch mean (≈ 0.9): weights become ±0.1-ish, above-average
    trajectories get pushed up, below-average ones **actively pushed down**.
    Spatially: you slid the whole reward histogram left until it straddles zero —
    5.1's vanished \( b \) — changing no expectation (§8 proves it: exactly zero
    bias) while collapsing the variance. GRPO's group mean is this subtraction,
    computed fresh per group.
    """),
        }),
        mo.md(
            r"**Where you'll use this:** grpo_notes §2 is this section weaponized — "
            r"the baseline exists *only* to shrink the variance of an "
            r"already-unbiased estimator, and the histogram you just squinted at is "
            r"its Figure 1."
        ),
    ], gap=1)
    return


@app.cell
def _():
    mo.md(r"""
    ## 6 · The probability distributions we actually use

    Only two families matter for this project, and neither is exotic.

    ### Categorical + softmax: the picture

    A **categorical** distribution is a die with unequal faces — bars on outcomes,
    heights summing to 1. An LLM's next-token distribution is a categorical over
    ~150k faces. Networks emit unconstrained reals ("logits"), so we need a bridge
    from \( \mathbb{R}^K \) to a valid set of bars: **softmax**. The temperature
    slider below is the knob to internalize — watch equal logit *gaps* stay equal
    while the bars redistribute.
    """)
    return


@app.cell
def _():
    r6_T = mo.ui.slider(0.05, 5.0, step=0.05, value=1.0, label="temperature T")
    r6_T
    return (r6_T,)


@app.cell
def _(r6_T):
    plt.close("all")
    _z = np.array([2.0, 1.0, 0.0, -1.0])
    _p = ref_softmax(_z / r6_T.value)

    _fig, _ax = plt.subplots(figsize=(8.5, 3.2), layout="constrained")
    _x = np.arange(len(_z))
    _ax.bar(_x, _p, width=0.62, color=C_BLUE)
    for _xi, _pi in zip(_x, _p):
        _ax.annotate(f"{_pi:.3f}", (_xi, _pi), ha="center", va="bottom", fontsize=9)
    _ax.set(
        xticks=_x, xticklabels=[f"logit {_v:+.0f}" for _v in _z],
        ylabel="probability", ylim=(0, 1.08),
        title=f"softmax(z / T) at T = {r6_T.value:.2f}   —   "
        + ("nearly argmax" if r6_T.value < 0.3 else "nearly uniform" if r6_T.value > 3.5 else "in between"),
    )
    _fig
    return


@app.cell
def _():
    mo.md(r"""
    ### The notation, pinned to the picture

    \[ p_i \;=\; \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}
    \qquad
    \begin{aligned}
    &z_i && \text{— the raw logit under bar } i \text{ (the x-axis labels)} \\
    &/T && \text{— your slider: divides all logit \emph{gaps} before anything else} \\
    &e^{(\cdot)} && \text{— §2's lens run backwards: makes everything positive,}\\
    & && \phantom{\text{—}}\ \text{turns logit \emph{differences} into probability \emph{ratios}} \\
    &\textstyle\sum_j && \text{— normalize so the bar heights sum to 1}
    \end{aligned} \]

    \( T \to 0 \): gaps get amplified to infinity, winner takes every bar —
    argmax. \( T \to \infty \): gaps shrink to nothing, bars level out — uniform.
    (You'll prove both limits in exercise 6.3.)

    ### Bernoulli and binomial, with the project's own numbers

    A **Bernoulli(p)** is one biased coin: 1 with probability \( p \), else 0. One
    catch attempt is exactly this — fruit in basket or not. Flip it \( G \) times
    independently (a **binomial**) and the corner case we care about is:

    \[ P(\text{all } G \text{ rollouts fail})
       \;=\; \underbrace{(1 - p)}_{\text{one miss}}{}^{\underbrace{G}_{\text{independent tries}}} . \]

    Why care: GRPO's baseline is the group mean. If all \( G \) fail, every reward
    equals the mean, every advantage is zero, and the group contributes **no
    gradient at all** — a *dead group*. Early in training \( p \) is small, and
    this curve decides whether the run learns or just heats the room. The
    `frac_zero_var_groups` column in `runs/log.jsonl` is this quantity, measured
    live.
    """)
    return


@app.cell
def _():
    r6_p = mo.ui.slider(0.0, 1.0, step=0.01, value=0.1, label="per-rollout success prob p")
    r6_p
    return (r6_p,)


@app.cell
def _(r6_p):
    plt.close("all")
    _ps = np.linspace(0, 1, 201)
    _blues = plt.cm.Blues(np.linspace(0.35, 0.95, 5))

    _fig, _ax = plt.subplots(figsize=(8.5, 3.5), layout="constrained")
    for _G, _c in zip([2, 4, 8, 16, 32], _blues):
        _ax.plot(_ps, (1 - _ps) ** _G, color=_c, lw=1.8, label=f"G = {_G}")
        _ax.scatter([r6_p.value], [(1 - r6_p.value) ** _G], color=_c, s=30, zorder=5)
    _here = (1 - r6_p.value) ** 8
    _ax.annotate(
        f"  p = {r6_p.value:.2f}, G = 8 → all fail with prob {_here:.3f}",
        (r6_p.value, _here), color=C_ORANGE, fontsize=9,
    )
    _ax.set(
        xlabel="success probability p", ylabel="P(all G rollouts fail) = (1−p)^G",
        ylim=(0, 1.04),
        title="the dead-group curve: how often a whole group strikes out",
    )
    _ax.legend(frameon=False, fontsize=8.5)
    _fig
    return


@app.cell
def _():
    _z = np.array([1.5, 0.5, -0.5])
    _num = np.exp(_z)
    _p = _num / _num.sum()
    _dead = 0.75**8
    _rng = np.random.default_rng(6)
    _mc_dead = float(((_rng.random((100_000, 8)) < 0.25).sum(axis=1) == 0).mean())

    mo.md(
        r"""
    ### The same facts in numpy

    ```python
    z = np.array([1.5, 0.5, -0.5])
    p = np.exp(z) / np.exp(z).sum()        # softmax: exponentiate, normalize

    dead_exact = (1 - 0.25)**8             # all 8 rollouts fail at p = 0.25
    groups     = rng.random((100_000, 8)) < 0.25       # 100k simulated groups
    dead_mc    = (groups.sum(axis=1) == 0).mean()      # fraction with zero catches
    ```
    """
        + f"""
    ```
    numerators = {np.array2string(_num, precision=3)}     p = {np.array2string(_p, precision=3)}
    dead_exact = {_dead:.4f}          dead_mc = {_mc_dead:.4f}
    ```

    Note the numerator ratios: logit gaps of 1.0 became probability ratios of
    e ≈ 2.718 each — §2's products↔sums bridge, crossed in the other direction.
    And the Monte Carlo estimate of the dead-group rate agrees with the formula to
    three digits, which is §4 keeping its promises.
    """
    )
    return


@app.cell
def _():
    mo.vstack([
        mo.md(r"""
    ### Exercises 6

    **6.1 (warm-up).** For \( p = 0.1 \), \( G = 8 \): compute the probability the
    group is dead in *either* direction — all fail or all succeed:
    \( (1-p)^G + p^G \). Which term dominates, and what does the number mean for a
    training run at this stage?

    **6.2.** Compute \( \operatorname{softmax}([2, 1, 0]) \) by hand, three
    significant figures (\( e \approx 2.718 \), \( e^2 \approx 7.389 \)).

    **6.3.** Prove the two temperature limits: as \( T \to 0 \),
    \( \operatorname{softmax}(z/T) \to \) one-hot on \( \arg\max_i z_i \) (unique
    max assumed); as \( T \to \infty \), it \( \to \) uniform.
    """),
        mo.accordion({
            "Solution 6.1": mo.md(r"""
    \( 0.9^8 = 0.430 \) and \( 0.1^8 = 10^{-8} \): total ≈ **0.430**, entirely the
    all-fail term. Meaning: at a 10% catch rate with \( G = 8 \), about **43% of
    groups produce exactly zero gradient** — nearly half the compute evaporates.
    One line of numpy to feel it: `(np.random.rand(100_000, 8) < 0.1).sum(1)` —
    count how many rows are all zeros. The escape hatches (bigger \( G \), easier
    curriculum, shaped reward) are graded against this number in grpo_notes §3.
    """),
            "Solution 6.2": mo.md(r"""
    Numerators \( e^2 \approx 7.389 \), \( e^1 \approx 2.718 \), \( e^0 = 1 \);
    sum \( 11.107 \):
    \[ p \approx (0.665,\; 0.245,\; 0.090). \]
    Equal logit *gaps* (2→1→0) gave equal probability *ratios* (each ≈ 2.72× the
    next) — additive differences became multiplicative ones, the log/exp bridge
    in reverse. Check: `np.exp([2,1,0]) / np.exp([2,1,0]).sum()`.
    """),
            "Solution 6.3": mo.md(r"""
    Let \( z_\star = \max_i z_i \), unique. Shift-invariance (exercise 2.4) lets
    us write
    \[ \operatorname{softmax}(z/T)_i
       = \frac{e^{(z_i - z_\star)/T}}{\sum_j e^{(z_j - z_\star)/T}} . \]
    \( T \to 0 \): every exponent with \( z_i < z_\star \) goes to \( -\infty \)
    (numerator → 0) while the max's exponent is exactly 0 (numerator = 1): one-hot
    on the argmax. \( T \to \infty \): all exponents → 0, all numerators → 1,
    every \( p_i \to 1/K \): uniform. Temperature interpolates between greedy
    decoding and monkeys with typewriters.
    """),
        }),
        mo.md(
            r"**Where you'll use this:** the policy \( \pi_\theta \) *is* a softmax "
            r"over ~150k tokens (`train.py` deliberately samples at neutral "
            r"temperature so the learning and sampling distributions match), and the "
            r"dead-group curve is grpo_notes §3's central failure mode — the "
            r"`frac_zero_var_groups` column of the training logs."
        ),
    ], gap=1)
    return


@app.cell
def _():
    mo.md(r"""
    ## 7 · KL divergence, gently

    ### The picture

    Two bar charts on the same outcomes: \( P \) (grey, fixed — the truth) and
    \( Q \) (blue — yours, on sliders). KL divergence measures **how surprised
    you'll be, on average, betting with \( Q \) while reality draws from \( P \)**
    — the expected *extra* log-loss beyond the surprise you can't avoid. Slide
    \( Q \) onto \( P \) and both KLs hit zero together. Then starve outcome 1
    (drag its logit to −3) and watch the asymmetry: \( \mathrm{KL}(P\|Q) \)
    rockets while \( \mathrm{KL}(Q\|P) \) stays polite. *Under-covering the truth
    is catastrophic in one direction and nearly free in the other.*
    """)
    return


@app.cell
def _():
    r7_z1 = mo.ui.slider(-3.0, 3.0, step=0.1, value=0.9, label="Q logit for outcome 1")
    r7_z2 = mo.ui.slider(-3.0, 3.0, step=0.1, value=0.4, label="Q logit for outcome 2")
    mo.hstack([r7_z1, r7_z2], justify="start", gap=2)
    return r7_z1, r7_z2


@app.cell
def _(r7_z1, r7_z2):
    plt.close("all")
    _P = np.array([0.5, 0.3, 0.2])
    _Q = ref_softmax(np.array([r7_z1.value, r7_z2.value, 0.0]))
    _kl_pq = float(np.sum(_P * np.log(_P / _Q)))
    _kl_qp = float(np.sum(_Q * np.log(_Q / _P)))

    _fig, _ax = plt.subplots(figsize=(8.5, 3.3), layout="constrained")
    _x = np.arange(3)
    _ax.bar(_x - 0.19, _P, width=0.38, color=C_FAINT, label="P (fixed truth)")
    _ax.bar(_x + 0.19, _Q, width=0.38, color=C_BLUE, label="Q (yours — slide me)")
    for _xi in _x:
        _ax.annotate(f"{_P[_xi]:.2f}", (_xi - 0.19, _P[_xi]), ha="center", va="bottom", fontsize=8.5, color=C_MUTED)
        _ax.annotate(f"{_Q[_xi]:.2f}", (_xi + 0.19, _Q[_xi]), ha="center", va="bottom", fontsize=8.5, color=C_BLUE)
    _ax.set(
        xticks=_x, xticklabels=["outcome 1", "outcome 2", "outcome 3"],
        ylabel="probability", ylim=(0, 1.0),
        title=f"KL(P‖Q) = {_kl_pq:.4f} nats      KL(Q‖P) = {_kl_qp:.4f} nats"
        + ("      (matched!)" if _kl_pq < 0.001 else ""),
    )
    _ax.legend(frameon=False, fontsize=9)
    _fig
    return


@app.cell
def _():
    mo.md(r"""
    ### The notation, pinned to the picture

    \[ \mathrm{KL}(P \,\|\, Q)
       \;=\; \sum_x
       \underbrace{P(x)}_{\substack{\text{grey bar:} \\ \text{how often } x \text{ happens}}}
       \,
       \underbrace{\log \frac{P(x)}{Q(x)}}_{\substack{\text{log of grey÷blue height:} \\ \text{extra surprise when it does}}}
       \;=\; \mathbb{E}_{x \sim P}\!\left[ \log \frac{P(x)}{Q(x)} \right] \]

    Read the recipe off the bars: for each outcome, take the log-ratio of the two
    heights (your under-estimate factor), weight by how often *truth* serves that
    outcome, sum. Three facts fall out of the shape of the formula:

    - **\( Q = P \) ⟹ KL = 0**: every ratio 1, every log 0. And it never goes
      below zero — no book of odds beats the true one on average (Jensen's
      inequality; the honest one-line proof is exercise 7.3's sting).
    - **Not symmetric**: swapping which distribution *samples* and which *scores*
      is a different experiment, so
      \( \mathrm{KL}(P\|Q) \ne \mathrm{KL}(Q\|P) \) in general. KL is a
      divergence, not a distance.
    - **Zeros are lethal on one side only**: if \( Q(x) = 0 \) where
      \( P(x) > 0 \), the term \( P(x)\log\frac{P(x)}{0} = \infty \) — infinite
      penalty for calling something impossible and then watching it happen. But
      where \( P(x) = 0 \), the weight out front is 0 and \( Q \) wastes mass
      unpunished (well, lightly punished, through normalization).

    ### Forward vs reverse: cover the modes, or seize one

    Scale that asymmetry up and it becomes a personality difference. Fit one
    Gaussian \( Q \) to a two-humped truth \( P \): **forward** KL
    \( (P\|Q) \) averages over *P's* samples — wherever truth puts mass, you'd
    better cover, so \( Q \) stretches across both humps (**mode-covering**), even
    parking its center over the empty valley. **Reverse** KL \( (Q\|P) \)
    averages over *Q's own* samples — you're only charged where *you* put mass,
    so the cheap strategy is to seize one hump, fit it snugly, and pretend the
    other doesn't exist (**mode-seeking**). Drag \( \mu \) between the humps and
    watch the two curves disagree; shrink \( \sigma \) and watch forward KL panic
    while reverse KL applauds.
    """)
    return


@app.cell
def _():
    r7_mu = mo.ui.slider(-4.0, 4.0, step=0.1, value=2.0, label="Q mean μ")
    r7_sigma = mo.ui.slider(0.3, 3.0, step=0.05, value=0.7, label="Q std σ")
    mo.hstack([r7_mu, r7_sigma], justify="start", gap=2)
    return r7_mu, r7_sigma


@app.cell
def _(r7_mu, r7_sigma):
    plt.close("all")
    _xg = np.linspace(-7, 7, 1401)
    _dx = _xg[1] - _xg[0]

    def _gauss(x, m, s):
        return np.exp(-((x - m) ** 2) / (2 * s**2)) / (s * np.sqrt(2 * np.pi))

    _p = 0.5 * _gauss(_xg, -2, 0.6) + 0.5 * _gauss(_xg, 2, 0.6)
    _p /= _p.sum() * _dx

    def _kls(m, s):
        _q = np.maximum(_gauss(_xg, m, s), 1e-300)
        _fwd = float(np.sum(_p * np.log(_p / _q)) * _dx)          # KL(P‖Q)
        _rev = float(np.sum(_q * np.log(_q / _p)) * _dx)          # KL(Q‖P)
        return _fwd, _rev

    _fwd0, _rev0 = _kls(r7_mu.value, r7_sigma.value)
    _mus = np.linspace(-4, 4, 161)
    _curves = np.array([_kls(_m, r7_sigma.value) for _m in _mus])

    _fig, (_axd, _axk) = plt.subplots(
        1, 2, figsize=(9.8, 3.6), layout="constrained", width_ratios=[1.15, 1]
    )
    _axd.fill_between(_xg, _p, color=C_FAINT, alpha=0.7, lw=0, label="P: two-mode truth")
    _axd.plot(_xg, _gauss(_xg, r7_mu.value, r7_sigma.value), color=C_BLUE, lw=2.2, label="Q: your Gaussian")
    _axd.set(
        xlabel="x", ylabel="density", xlim=(-7, 7),
        title=f"KL(P‖Q) = {_fwd0:.2f}      KL(Q‖P) = {_rev0:.2f}",
    )
    _axd.legend(frameon=False, fontsize=9)

    _axk.plot(_mus, _curves[:, 0], color=C_ORANGE, lw=2, label="forward KL(P‖Q): cover!")
    _axk.plot(_mus, _curves[:, 1], color=C_BLUE, lw=2, label="reverse KL(Q‖P): seize a mode!")
    _axk.scatter([r7_mu.value], [_fwd0], color=C_ORANGE, zorder=5, s=40)
    _axk.scatter([r7_mu.value], [_rev0], color=C_BLUE, zorder=5, s=40)
    _axk.set(
        xlabel="Q mean μ", ylabel="nats", yscale="log",
        title=f"both KLs vs μ (at σ = {r7_sigma.value:.2f})",
    )
    _axk.legend(frameon=False, fontsize=8.5)
    _fig
    return


@app.cell
def _():
    _P = np.array([0.5, 0.5])
    _Q = np.array([0.75, 0.25])
    _kl_pq = float((_P * np.log(_P / _Q)).sum())
    _kl_qp = float((_Q * np.log(_Q / _P)).sum())

    mo.md(
        r"""
    ### The same facts in numpy

    The formula is one line, and the asymmetry is visible in which array gets to
    stand out front (= which distribution's samples pay the bill):

    ```python
    P, Q  = np.array([0.5, 0.5]), np.array([0.75, 0.25])   # two coins
    kl_pq = (P * np.log(P / Q)).sum()      # P samples, Q scores  (forward)
    kl_qp = (Q * np.log(Q / P)).sum()      # Q samples, P scores  (reverse)
    ```
    """
        + f"""
    ```
    kl_pq = {_kl_pq:.4f} nats      kl_qp = {_kl_qp:.4f} nats      # not equal
    ```

    Same two coins, two different numbers — the experiment you run (who samples,
    who scores) is part of the definition. Exercise 7.1 has you produce these by
    hand.
    """
    )
    return


@app.cell
def _():
    mo.vstack([
        mo.md(r"""
    ### Exercises 7

    **7.1.** Coins: \( P = \mathrm{Bernoulli}(0.5) \),
    \( Q = \mathrm{Bernoulli}(0.75) \). Compute \( \mathrm{KL}(P\|Q) \) and
    \( \mathrm{KL}(Q\|P) \) in nats by hand
    (\( \ln 2 \approx 0.693 \), \( \ln 3 \approx 1.099 \)) and check against the
    numpy above.

    **7.2.** Explain from the formula why \( \mathrm{KL}(P\|Q) = \infty \) whenever
    \( Q(x) = 0 \) for some \( x \) with \( P(x) > 0 \) — then say what that
    implies for a KL penalty whose job is "stay a language model."

    **7.3 (stinger).** Prove \( \mathrm{KL}(P\|Q) \ge 0 \) using Jensen's
    inequality (for concave \( \log \):
    \( \mathbb{E}[\log Y] \le \log \mathbb{E}[Y] \)).
    *Hint: start from \( -\mathrm{KL}(P\|Q) = \mathbb{E}_{P}[\log(Q/P)] \).*
    """),
        mo.accordion({
            "Solution 7.1": mo.md(r"""
    Forward:
    \[ \mathrm{KL}(P\|Q) = 0.5 \ln\frac{0.5}{0.75} + 0.5 \ln\frac{0.5}{0.25}
       = 0.5(\ln 2 - \ln 3) + 0.5 \ln 2 \approx -0.203 + 0.347 = 0.144 \text{ nats}. \]
    Reverse:
    \[ \mathrm{KL}(Q\|P) = 0.75 \ln 1.5 + 0.25 \ln 0.5
       \approx 0.304 - 0.173 = 0.131 \text{ nats}. \]
    Both match the numpy cell. Different numbers for the same pair of coins:
    which one you optimize decides *whose samples pay the bill*.
    """),
            "Solution 7.2": mo.md(r"""
    The term \( P(x) \log\frac{P(x)}{Q(x)} \) contains
    \( \log\frac{P(x)}{0} = +\infty \), weighted by \( P(x) > 0 \): the sum is
    infinite. In words: \( Q \) declared an event impossible, the event has real
    probability, and the average surprise of watching the impossible happen is
    unbounded.

    For the leash: `train.py` penalizes
    \( \mathrm{KL}(\pi_\theta \| \pi_{\text{ref}}) \) — **reverse** KL, the policy
    in the sampling slot, expectation over the *policy's own tokens*. The penalty
    explodes exactly when the policy generates text the frozen reference finds
    (near-)impossible — gibberish, degenerate filler, non-language — but charges
    almost nothing for *abandoning* things the reference could have said.
    Mode-seeking is precisely the right shape for a leash: sharpen onto good
    behavior freely; never leave the language manifold.
    """),
            "Solution 7.3": mo.md(r"""
    \[ -\mathrm{KL}(P\|Q) = \mathbb{E}_{x\sim P}\!\left[\log \frac{Q(x)}{P(x)}\right]
       \;\le\; \log \mathbb{E}_{x\sim P}\!\left[\frac{Q(x)}{P(x)}\right]
       \quad \text{(Jensen: log of average ≥ average of log)}. \]
    Now the trailing expectation collapses:
    \[ \mathbb{E}_{x\sim P}\!\left[\frac{Q(x)}{P(x)}\right]
       = \sum_x P(x) \frac{Q(x)}{P(x)} = \sum_x Q(x) = 1
       \;\Longrightarrow\; -\mathrm{KL} \le \log 1 = 0. \]
    *What the collapse IS computationally:* multiplying by \( P \) inside a
    \( \sum_x P(x) \cdot (\,\cdot\,) \) cancels the \( P \) in the denominator and
    leaves a bare distribution summing to 1. Mark this move — "a distribution
    appearing under a sum makes the sum trivial" — it is the mirror image of the
    trick you're about to pull in the capstone, where you *create* the
    distribution under the integral instead of cancelling it.
    """),
        }),
        mo.md(
            r"**Where you'll use this:** grpo_notes §6 — the "
            r"\( \beta\,\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}}) \) leash is "
            r"*reverse* KL for exactly the mode-seeking reason above, and its "
            r"\( k_3 \) Monte-Carlo estimator is §4's sampling idea applied to the "
            r"KL integral."
        ),
    ], gap=1)
    return


@app.cell
def _():
    mo.md(r"""
    ## 8 · Capstone: re-derive the score-function trick yourself

    Everything is on the table: derivatives as sensitivity (§1), the
    log-derivative identity (§2), gradients and ascent (§3), expectations and
    sampling (§4), the variance stakes (§5). Time to collect.

    **The problem.** A policy \( \pi_\theta \) produces trajectories \( \tau \); a
    black-box program scores them with \( R(\tau) \). We want

    \[ \nabla_\theta J(\theta), \qquad
       J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]
       = \int \pi_\theta(\tau)\, R(\tau)\, d\tau , \]

    with **no** \( \partial R/\partial \tau \) available — \( R \) is code, not
    calculus. Spatially: \( \pi_\theta \) is a lump of probability mass spread
    over trajectory-space, \( R \) is a fixed altitude map over that same space,
    and \( J \) is the average altitude *of the mass*. Turning \( \theta \) does
    not move the mountains; it **moves the mass** — that is the only channel
    through which \( \theta \) matters, so that is where the derivative must go.

    Four lines. Each posed as a question below; commit to an answer (out loud, on
    paper — whatever makes it honest) before opening the reveal. Each reveal ends
    with a one-line gloss of what the step *is*, spatially or computationally.

    **But feel it before you prove it.** Three trajectories, each with a fixed
    reward (green — the mountains), and a policy spreading probability mass over
    them (blue — the mass). The slider below is one component of \( \theta \).
    Scrub it: the green levels never move; \( J \) changes *only* because mass
    flows between them. Every line of the derivation is bookkeeping for exactly
    this motion.
    """)
    return


@app.cell
def _():
    r8_th = mo.ui.slider(-3.0, 3.0, step=0.1, value=0.0, label="scrub θ (one logit)")
    r8_th
    return (r8_th,)


@app.cell
def _(r8_th):
    plt.close("all")
    _r = np.array([0.2, 0.5, 0.8])
    _p = ref_softmax(np.array([0.3, -0.1, r8_th.value]))
    _J = float(_p @ _r)

    _fig, _ax = plt.subplots(figsize=(8.5, 3.4), layout="constrained")
    _x = np.arange(3)
    _ax.bar(_x, _p, width=0.55, color=C_BLUE)
    for _xi in _x:
        _ax.annotate(f"π = {_p[_xi]:.2f}", (_xi, _p[_xi] + 0.02), ha="center",
                     color=C_BLUE, fontsize=9)
        _ax.plot([_xi - 0.38, _xi + 0.38], [_r[_xi], _r[_xi]], color=C_AQUA, lw=2.5)
        _ax.annotate(f"R = {_r[_xi]:.1f}", (_xi + 0.4, _r[_xi]), va="center",
                     color=C_AQUA, fontsize=9)
    _ax.set(
        xticks=_x, xticklabels=["trajectory A", "trajectory B", "trajectory C"],
        ylabel="probability mass  π_θ(τ)", ylim=(0, 1.06),
        title=f"J(θ) = Σ π·R = {_J:.3f}   —   the mountains never move; only the mass does",
    )
    _fig
    return


@app.cell
def _():
    mo.vstack([
        mo.md(r"""
    **Step 1.** Starting from
    \( \nabla_\theta J = \nabla_\theta \int \pi_\theta(\tau) R(\tau)\, d\tau \):
    the first move pushes \( \nabla_\theta \) inside the integral. Why is that
    legitimate, and what does the integrand become?
    """),
        mo.accordion({
            "Reveal step 1": mo.md(r"""
    \[ \nabla_\theta J = \int \nabla_\theta \big[ \pi_\theta(\tau)\, R(\tau) \big]\, d\tau . \]
    The integral runs over \( \tau \), the gradient over \( \theta \) — different
    variables, so the operations commute (dominated convergence for the lawyers;
    every distribution here is smooth enough that no one loses sleep).

    *Computationally, this step IS:* swapping two loops —
    `grad(sum(...))` → `sum(grad(...))`. Nothing clever has happened yet.
    """),
        }),
        mo.md(r"""
    **Step 2.** The integrand is \( \nabla_\theta [\pi_\theta(\tau) R(\tau)] \).
    What happens to \( R(\tau) \), and why is this the step that lets us use a
    non-differentiable reward at all?
    """),
        mo.accordion({
            "Reveal step 2": mo.md(r"""
    \[ \nabla_\theta \big[ \pi_\theta(\tau)\, R(\tau) \big]
       = R(\tau)\, \nabla_\theta \pi_\theta(\tau) . \]
    Inside the integral, \( \tau \) is a *fixed* trajectory and \( R(\tau) \) is
    just the number attached to it — no \( \theta \) anywhere in it, so under
    \( \nabla_\theta \) it is a constant and rides along as a multiplier. We will
    never differentiate \( R \).

    *Spatially, this step IS:* the altitude map is bolted down; only the mass
    under it moves when you turn \( \theta \). \( R \)'s job from here on is
    merely to say *how much each parcel of moving mass matters*.
    """),
        }),
        mo.md(r"""
    **Step 3.** We're stuck with \( \nabla_\theta \pi_\theta(\tau) \), which is
    not an expectation of anything — nothing in the integrand is a probability
    density anymore, so sampling can't estimate it. Section 2 gave you an identity
    that rewrites it. Apply it. What structural problem does it solve?
    """),
        mo.accordion({
            "Reveal step 3": mo.md(r"""
    The log-derivative identity, read right-to-left
    (\( \nabla \pi = \pi\, \nabla \log \pi \)):
    \[ \nabla_\theta J
       = \int \pi_\theta(\tau)\, \nabla_\theta \log \pi_\theta(\tau)\, R(\tau)\, d\tau . \]
    It **manufactures a factor of \( \pi_\theta \) out of thin air** — and a
    density multiplying the rest of an integrand is exactly the shape of an
    expectation.

    *Computationally, this step IS:* multiply-and-divide by \( \pi_\theta \) so
    the integral regains a "sample from me" factor — the exact mirror of the
    cancellation move you flagged in exercise 7.3.
    """),
        }),
        mo.md(r"""
    **Step 4.** The integral now has the form
    \( \int \pi_\theta(\tau)\, [\,\cdot\,]\, d\tau \). Finish it. What can you now
    *do* that you couldn't before?
    """),
        mo.accordion({
            "Reveal step 4": mo.md(r"""
    \[ \nabla_\theta J
       = \mathbb{E}_{\tau \sim \pi_\theta}\big[ R(\tau)\, \nabla_\theta \log \pi_\theta(\tau) \big]
       \;\approx\; \frac{1}{n} \sum_{i=1}^{n} R(\tau_i)\, \nabla_\theta \log \pi_\theta(\tau_i) . \]
    An expectation is something §4 taught you to estimate by sampling: roll out
    \( n \) trajectories, compute each one's \( \nabla_\theta \log \pi_\theta \)
    (your own network's log-prob of what it did — fully differentiable, §1's
    chain rule through every layer), weight by reward, average.

    *Computationally, this step IS:* the for-loop over rollouts in `train.py`.
    The gradient of a **non-differentiable** objective became a **sampleable
    expectation** — that is REINFORCE, and you just derived it.
    """),
        }),
        mo.md(r"""
    ### Victory lap: baselines are free

    One more, unassisted — this is grpo_notes §2, and after §2, §4 and §5 you own
    every ingredient. Show that for any constant \( b \):

    \[ \mathbb{E}_{\tau \sim \pi_\theta}\big[ b\, \nabla_\theta \log \pi_\theta(\tau) \big] = 0 , \]

    which licenses replacing \( R \) with the advantage \( R - b \) — §5's
    variance surgery — at exactly zero cost in bias.
    *Hint: run step 3 in reverse, then remember what all probabilities sum to.*
    """),
        mo.accordion({
            "Solution (victory lap)": mo.md(r"""
    \[ \mathbb{E}\big[ b\, \nabla \log \pi_\theta \big]
       = b \int \pi_\theta\, \nabla_\theta \log \pi_\theta \, d\tau
       = b \int \nabla_\theta \pi_\theta \, d\tau
       = b\, \nabla_\theta \!\!\int \pi_\theta \, d\tau
       = b\, \nabla_\theta 1 = 0 . \]
    Move by move, with glosses:
    constants exit expectations (§4 — *pull the scalar out of the loop*);
    the log-derivative identity runs *left-to-right* this time, collapsing
    \( \pi \nabla \log \pi \) into \( \nabla \pi \) (§2);
    gradient and integral swap back (step 1 — *the loops again*);
    and probabilities integrate to 1, whose gradient is zero (§1 — *the total
    mass is a constant altitude; a flat function has slope 0 in every direction*).
    Spatially: squeezing probability toward some trajectories necessarily drains
    it from others, and \( b \) pays both sides equally — the net push is zero.
    """),
        }),
    ], gap=1)
    return


@app.cell
def _():
    _r = np.array([0.2, 0.5, 0.8])          # black-box rewards per arm
    _th = np.array([0.3, -0.1, 0.2])
    _h = 1e-6

    def _J(t):
        return float(ref_softmax(t) @ _r)   # E[R]: only 3 outcomes, so cheat with the exact sum

    _fd = np.array(
        [(_J(_th + _h * np.eye(3)[k]) - _J(_th - _h * np.eye(3)[k])) / (2 * _h) for k in range(3)]
    )
    _p = ref_softmax(_th)
    _score = sum(_p[a] * _r[a] * (np.eye(3)[a] - _p) for a in range(3))
    _base = sum(_p[a] * 0.9 * (np.eye(3)[a] - _p) for a in range(3))

    mo.md(
        r"""
    ### The whole derivation, verified in numpy

    A 3-armed softmax bandit is small enough to compute \( \mathbb{E}[R] \)
    exactly (it's `p @ r` — a dot product, §4), which means we can check the
    capstone against brute-force finite differences (§1) with no sampling noise.
    The score-function estimator uses 2.3's \( \nabla \log p_a = e_a - p \):

    ```python
    r, th, h = np.array([.2, .5, .8]), np.array([.3, -.1, .2]), 1e-6
    J  = lambda t: ref_softmax(t) @ r                      # E[R] as an exact sum
    fd = [(J(th + h*np.eye(3)[k]) - J(th - h*np.eye(3)[k])) / (2*h)
          for k in range(3)]                               # ∇J by wiggling (§1)

    p     = ref_softmax(th)
    score = sum(p[a] * r[a]  * (np.eye(3)[a] - p) for a in range(3))  # E[R ∇log π]
    base  = sum(p[a] * 0.9   * (np.eye(3)[a] - p) for a in range(3))  # E[b ∇log π]
    ```
    """
        + f"""
    ```
    fd    = {np.array2string(_fd, precision=6)}     ← ∇J, by wiggling θ
    score = {np.array2string(_score, precision=6)}     ← E[R ∇log π], the capstone
    base  = {np.array2string(_base, precision=6, suppress_small=True)}     ← E[b ∇log π], the victory lap
    ```

    Line one and line two agree to six decimals: the score-function identity is
    not a metaphor, it is an equality between two computable arrays. Line three is
    zero: the baseline really is free. Replace the exact sums with `1/n Σ` over
    sampled arms and you have, character for character, the REINFORCE update at
    the top of `grpo_notes.py`.
    """
    )
    return


@app.cell
def _():
    mo.md(r"""
    ## The map: you now own every ingredient of `grpo_notes.py` §1–6

    Open `uv run marimo edit grpo_notes.py` and read it against this key:

    - **§1 (score-function trick)** — the capstone you just derived, verbatim:
      steps 1–4 are its equations, and 2.3's softmax gradient \( e_a - p \) is
      literally the `np.eye(3)[a] - probs` line in its bandit code.
    - **§2 (baselines are free)** — your victory-lap proof, plus §5's variance
      economics explaining *why* anyone bothers adding a provably-zero term.
    - **§3 (GRPO's group baseline)** — §4's Monte Carlo (the group mean is a
      sample mean standing in for \( \mathbb{E}[R \mid s] \), error
      \( \sigma/\sqrt{G} \)) plus §6's dead-group curve \( p^G + (1-p)^G \).
    - **§3b (Dr.GRPO, no std division)** — §5's machinery: dividing by
      \( \sigma_{\text{group}} \) rescales gradient weights by a noise-inflating
      \( 1/\sigma \), and you now know exactly what \( \sigma \) measures.
    - **§4 (the token mask)** — §2's products-become-sums: an episode's log-prob
      is a *sum* over tokens, the environment's terms contain no \( \theta \), and
      §1 says a \( \theta \)-free term has derivative zero — masking them out *is*
      the math, not bookkeeping.
    - **§5 (where PPO's clip went)** — §6's sampling-distribution discipline:
      on-policy means the importance ratio is identically 1 and the clip is a
      no-op.
    - **§6 (the KL leash)** — §7 wholesale: reverse KL for mode-seeking, the
      infinity at unsupported outputs as the "stay a language model" enforcement,
      and §4's sampling to estimate it per token.

    Three representations, round trips between them, thirty problems counting
    the capstone's guided steps. Every ingredient is in hand. Go read the real
    thing.
    """)
    return


if __name__ == "__main__":
    app.run()
