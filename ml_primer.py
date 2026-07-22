import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as F

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

    # Visual grammar — same as the sibling notebooks:
    #   C_INK / C_FAINT : fixed scenery (data, the truth, the reference)
    #   C_ORANGE        : whatever the slider is currently moving
    #   C_BLUE          : the limit / the controlled object the orange thing tracks
    #   C_AQUA          : reward / target
    C_BLUE, C_ORANGE, C_AQUA, C_YELLOW, C_MAGENTA = (
        "#2a78d6",
        "#eb6834",
        "#1baf7a",
        "#eda100",
        "#e87ba4",
    )
    C_INK, C_MUTED, C_FAINT = "#0b0b0b", "#52514e", "#c9c8c0"

    # Section 4 toy sequence; section 5 toy vocabulary.
    ATTN_TOKENS = ["the", "banana", "drifts", "right", "catch", "it"]
    LM_VOCAB = list("ABCDEFGH")


@app.function
def softmax_rows(z):
    """Numerically stable softmax along the last axis."""
    z = np.asarray(z, dtype=float)
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


@app.cell
def _():
    mo.md(r"""
    # Core ML in five demos

    A cache-warmer sitting between `math_refresher.py` (the calculus and probability)
    and `grpo_notes.py` (REINFORCE → GRPO). Five facts, each stated, poked with a
    slider, and verified in numpy — brisk on purpose, because none of it is new,
    only cold. The color grammar is the siblings': grey is fixed scenery, **orange**
    is what the slider moves, **blue** is the limit or the controlled object,
    **green** is the target. The closing section maps each fact onto the lines of
    `grpo_notes.py` and `train.py` it unlocks.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## 1 · A network is a stack of bent linear maps

    A linear layer \( h = Wx + b \) does exactly three things to space: rotate,
    shear/scale (that's \( W \)), and translate (that's \( b \)). The unit grid maps
    to a grid of parallelograms; straight lines stay straight; parallel stays
    parallel. And — fatally for depth — linear maps compose into linear maps:
    \( W_2(W_1 x + b_1) + b_2 \) collapses to a single \( W' x + b' \). A hundred
    stacked linear layers can draw exactly one straight decision boundary.

    The nonlinearity is the bend. \( \tanh \) (or ReLU) applied elementwise squashes
    each coordinate independently, which *curves* the grid — and once the grid can
    curve, the next linear layer's straight cut through the *bent* space is a curved
    cut through the original space. Capacity comes from composition: bend, cut,
    bend, cut. That is the entire architectural idea.

    The demo trains a 2-layer numpy net ( \( 2 \to 16\ \tanh \to 2 \) ) on
    two-moons with plain full-batch gradient descent, snapshotting weights every
    epoch. Drag the epoch slider and watch the boundary start as a near-straight
    cut (the best a linear map could ever do) and progressively bend around the
    moons. The dashed blue curve is the epoch-200 boundary — the thing the orange
    one is converging to.
    """)
    return


@app.function
def moons_data(n=240, noise=0.10, seed=1):
    """Two interleaved half-moons in 2D. Returns X (n,2), y (n,) in {0,1}."""
    rng = np.random.default_rng(seed)
    m = n // 2
    t = rng.uniform(0, np.pi, m)
    X0 = np.c_[np.cos(t), np.sin(t)]
    X1 = np.c_[1 - np.cos(t), 0.5 - np.sin(t)]
    X = np.vstack([X0, X1]) + rng.normal(0, noise, (n, 2))
    y = np.r_[np.zeros(m, int), np.ones(m, int)]
    return X, y


@app.function
def net_forward(params, X):
    H = np.tanh(X @ params["W1"] + params["b1"])
    P = softmax_rows(H @ params["W2"] + params["b2"])
    return H, P


@app.function
def net_train(X, y, hidden=16, epochs=200, lr=1.5, seed=0):
    """Full-batch GD on softmax cross-entropy; returns a snapshot per epoch."""
    rng = np.random.default_rng(seed)
    p = {
        "W1": rng.normal(0, 0.6, (2, hidden)),
        "b1": np.zeros(hidden),
        "W2": rng.normal(0, 0.6, (hidden, 2)),
        "b2": np.zeros(2),
    }
    Y = np.eye(2)[y]
    snaps = [{k: v.copy() for k, v in p.items()}]
    for _ in range(epochs):
        H, P = net_forward(p, X)
        D = (P - Y) / len(X)               # ∂CE/∂logits, section 3's identity
        gW2, gb2 = H.T @ D, D.sum(0)
        dH = (D @ p["W2"].T) * (1 - H * H)
        gW1, gb1 = X.T @ dH, dH.sum(0)
        for k, g in zip(["W1", "b1", "W2", "b2"], [gW1, gb1, gW2, gb2]):
            p[k] -= lr * g
        snaps.append({k: v.copy() for k, v in p.items()})
    return snaps


@app.cell
def _():
    moons_X, moons_y = moons_data()
    net_snaps = net_train(moons_X, moons_y)
    return moons_X, moons_y, net_snaps


@app.cell
def _(net_snaps):
    p1_epoch = mo.ui.slider(0, len(net_snaps) - 1, step=2, value=0, label="training epoch")
    p1_epoch
    return (p1_epoch,)


@app.cell
def _(moons_X, moons_y, net_snaps, p1_epoch):
    plt.close("all")
    _p = net_snaps[p1_epoch.value]
    _gx, _gy = np.meshgrid(np.linspace(-1.8, 2.8, 90), np.linspace(-1.4, 1.9, 90))
    _G = np.c_[_gx.ravel(), _gy.ravel()]
    _, _P = net_forward(_p, _G)
    _Z = _P[:, 1].reshape(_gx.shape)
    _, _Pfin = net_forward(net_snaps[-1], _G)
    _Zfin = _Pfin[:, 1].reshape(_gx.shape)
    _, _Ptr = net_forward(_p, moons_X)
    _acc = (_Ptr.argmax(1) == moons_y).mean()

    _fig, _ax = plt.subplots(figsize=(8.5, 3.8), layout="constrained")
    _ax.contourf(_gx, _gy, _Z, levels=[0, 0.5, 1], colors=["#f4f4f2", "#fdeee6"])
    _ax.contour(_gx, _gy, _Zfin, levels=[0.5], colors=[C_BLUE], linewidths=1.6, linestyles="--")
    _cs = _ax.contour(_gx, _gy, _Z, levels=[0.5], colors=[C_ORANGE], linewidths=2.4)
    _ax.scatter(*moons_X[moons_y == 0].T, s=16, facecolors="none", edgecolors=C_MUTED, lw=1.1)
    _ax.scatter(*moons_X[moons_y == 1].T, s=16, color=C_INK)
    _ax.annotate("p(class 1) = ½ now", (-1.7, 1.7), color=C_ORANGE, fontsize=9)
    _ax.annotate("epoch-200 boundary", (-1.7, 1.45), color=C_BLUE, fontsize=9)
    _ax.grid(False)
    _ax.set(
        xlabel="x₁", ylabel="x₂",
        title=f"epoch {p1_epoch.value}:  train accuracy {_acc:.0%}"
        + ("  — still (almost) a straight cut" if p1_epoch.value <= 4 else ""),
    )
    _fig
    return


@app.cell
def _():
    mo.vstack([
        mo.md(r"""
    **Exercises 1**

    **1.1.** Prove the collapse claim: show \( W_2(W_1x + b_1) + b_2 = W'x + b' \),
    identifying \( W' \) and \( b' \). What single fact about \( \tanh \) breaks the
    proof when it is inserted between the layers?

    **1.2.** With hidden width 1 ( \( 2 \to 1\ \tanh \to 2 \) ), what shapes can the
    decision boundary take? Why does the demo need width \( \gtrsim 4 \) to wrap two
    moons?
    """),
        mo.accordion({
            "Solution 1.1": mo.md(r"""
    \( W_2(W_1x + b_1) + b_2 = (W_2W_1)x + (W_2b_1 + b_2) \), so \( W' = W_2W_1 \),
    \( b' = W_2b_1 + b_2 \) — matrix products and affine shifts are closed under
    composition. \( \tanh \) breaks it because it is not affine:
    \( \tanh(u + v) \neq \tanh u + \tanh v \), so \( W_2\tanh(W_1x + b_1) \) cannot
    be rewritten as any single \( W'x + b' \). One elementwise bend is what makes
    depth non-trivial.
    """),
            "Solution 1.2": mo.md(r"""
    Width 1 computes \( \tanh(w^\top x + b) \) — a function of the single scalar
    \( w^\top x \), constant along every line perpendicular to \( w \). The final
    layer can only threshold it, so the boundary is a straight line (or a pair of
    parallel lines, using the two flat tails of one tanh ridge). Each hidden unit
    contributes one such ridge, one "fold" of the space; wrapping a moon needs
    several folds in different directions, hence several units.
    """),
        }),
    ], gap=1)
    return


@app.cell
def _():
    mo.md(r"""
    ## 2 · Backprop is the chain rule with bookkeeping

    Write the network as a computation graph — every intermediate value a node,
    every operation an edge. The chain rule then has a purely local reading:

    \[ \frac{\partial L}{\partial (\text{node})}
       \;=\; \underbrace{\frac{\partial L}{\partial (\text{node's output})}}_{\text{upstream, arrives from the right}}
       \times \underbrace{\frac{\partial (\text{output})}{\partial (\text{node})}}_{\text{local, known from the op alone}} \]

    Backprop is one right-to-left sweep applying that multiplication at every node,
    caching each result so it is computed once — which is why the backward pass
    costs about the same as the forward pass, and why *all* parameter gradients
    arrive in a single sweep. Parameters need no special treatment: \( w \) and
    \( c \) below are just leaf nodes, so the same rule deposits
    \( \partial L / \partial w \) on them. "Training" reads the gradients off the
    leaves that happen to be adjustable.

    The graph below computes \( L = (w \cdot x + c)^2 \) with \( w = 1.5 \),
    \( c = -2 \). Drag \( x \): forward values (orange, above each node) flow left
    to right, gradients (blue, below) flow right to left, and every blue number is
    the blue number to its right times the grey local derivative on the edge.
    """)
    return


@app.cell
def _():
    p2_x = mo.ui.slider(-3.0, 3.0, step=0.1, value=2.0, label="input x")
    p2_x
    return (p2_x,)


@app.cell
def _(p2_x):
    plt.close("all")
    _w, _c, _x = 1.5, -2.0, p2_x.value
    _a = _w * _x
    _z = _a + _c
    _L = _z**2
    _dz = 2 * _z          # ∂L/∂z
    _da = _dz * 1.0       # ∂L/∂a
    _dx = _da * _w        # ∂L/∂x
    _dw = _da * _x        # ∂L/∂w
    _dc = _dz * 1.0       # ∂L/∂c

    _fig, _ax = plt.subplots(figsize=(9.5, 3.4), layout="constrained")
    _ax.axis("off")
    _ax.set(xlim=(-0.7, 7.0), ylim=(-2.3, 1.9))

    def _node(x, y, name, fv, gv):
        _ax.annotate(name, (x, y), ha="center", va="center", fontsize=10.5, color=C_INK,
                     bbox=dict(boxstyle="round,pad=0.45", fc="white", ec=C_MUTED, lw=1.2))
        _ax.annotate(f"{fv:+.2f}", (x, y + 0.52), ha="center", color=C_ORANGE, fontsize=10)
        _ax.annotate(f"∂L = {gv:+.2f}", (x, y - 0.55), ha="center", color=C_BLUE, fontsize=9.5)

    def _edge(x0, y0, x1, y1, local):
        _ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                     arrowprops=dict(arrowstyle="->", color=C_MUTED, lw=1.3))
        _ax.annotate(local, ((x0 + x1) / 2, (y0 + y1) / 2 + 0.16),
                     ha="center", color=C_MUTED, fontsize=8.5)

    _node(0.0, 0.9, "x", _x, _dx)
    _node(0.0, -1.5, "w (param)", _w, _dw)
    _node(2.2, -0.2, "a = w·x", _a, _da)
    _node(4.4, -1.5, "c (param)", _c, _dc)
    _node(4.4, -0.2, "z = a + c", _z, _dz)
    _node(6.4, -0.2, "L = z²", _L, 1.0)
    _edge(0.45, 0.75, 1.55, 0.05, "∂a/∂x = w")
    _edge(0.65, -1.35, 1.55, -0.5, "∂a/∂w = x")
    _edge(2.9, -0.2, 3.7, -0.2, "∂z/∂a = 1")
    _edge(4.4, -1.15, 4.4, -0.72, "∂z/∂c = 1")
    _edge(5.1, -0.2, 5.85, -0.2, "∂L/∂z = 2z")
    _ax.set_title(
        f"x = {_x:+.1f}:  L = {_L:.2f};  every blue number = (blue to its right) × (grey local)"
    )
    _fig
    return


@app.cell
def _():
    mo.md(r"""
    Same idea at network scale, with receipts. Below: a 2-layer net
    ( \( 3 \to 8\ \tanh \to 2 \), MSE loss), gradients computed twice on identical
    float64 weights — once by a ten-line manual backward (numpy), once by
    `torch.autograd`. The chain-rule sweep and the framework are the same
    algorithm, so the numbers agree to machine precision.
    """)
    return


@app.function
def backprop_check(seed=0):
    """Manual numpy backward vs torch.autograd on identical weights.
    Returns (loss, {param: max |numpy_grad - torch_grad|})."""
    rng = np.random.default_rng(seed)
    X, Y = rng.normal(size=(6, 3)), rng.normal(size=(6, 2))
    W1, b1 = rng.normal(size=(3, 8)) * 0.5, np.zeros(8)
    W2, b2 = rng.normal(size=(8, 2)) * 0.5, np.zeros(2)

    # -- manual: forward, then the ten-line backward -------------------------
    H = np.tanh(X @ W1 + b1)
    Yh = H @ W2 + b2
    L = ((Yh - Y) ** 2).mean()
    dYh = 2 * (Yh - Y) / Yh.size          # ∂L/∂Yh
    gW2 = H.T @ dYh                       # local (H) × upstream
    gb2 = dYh.sum(0)
    dH = (dYh @ W2.T) * (1 - H * H)       # through W2, then through tanh
    gW1 = X.T @ dH
    gb1 = dH.sum(0)

    # -- autograd on the same float64 weights --------------------------------
    tW1, tb1, tW2, tb2 = (torch.tensor(v, requires_grad=True) for v in (W1, b1, W2, b2))
    tYh = torch.tanh(torch.tensor(X) @ tW1 + tb1) @ tW2 + tb2
    ((tYh - torch.tensor(Y)) ** 2).mean().backward()
    diffs = {
        k: float(np.abs(g - t.grad.numpy()).max())
        for k, g, t in [("W1", gW1, tW1), ("b1", gb1, tb1), ("W2", gW2, tW2), ("b2", gb2, tb2)]
    }
    return L, diffs


@app.cell
def _():
    _L, _diffs = backprop_check()
    _rows = "\n".join(f"| `{_k}` | {_v:.2e} |" for _k, _v in _diffs.items())
    mo.md(
        f"loss = {_L:.6f}; max absolute gradient disagreement per parameter:\n\n"
        f"| parameter | max \\|numpy − autograd\\| |\n|---|---|\n{_rows}\n\n"
        "Float64 epsilon is ≈ 2.2e-16 — the two computations differ only in "
        "floating-point association order."
    )
    return


@app.cell
def _():
    mo.vstack([
        mo.md(r"""
    **Exercises 2**

    **2.1.** In the graph demo, set \( x = 2 \) and verify all five gradients by
    hand ( \( w = 1.5 \), \( c = -2 \) ).

    **2.2.** Reverse-mode AD (backprop) computes \( \partial L/\partial\theta \)
    for *all* parameters in one backward sweep. Forward-mode computes
    \( \partial(\text{everything})/\partial x_i \) for *one* input per sweep. Why
    is reverse mode the only sane choice for training, and when would forward mode
    win?
    """),
        mo.accordion({
            "Solution 2.1": mo.md(r"""
    Forward: \( a = 3 \), \( z = 1 \), \( L = 1 \). Backward:
    \( \partial L/\partial z = 2z = 2 \);
    \( \partial L/\partial a = 2 \cdot 1 = 2 \);
    \( \partial L/\partial c = 2 \cdot 1 = 2 \);
    \( \partial L/\partial x = 2 \cdot w = 3 \);
    \( \partial L/\partial w = 2 \cdot x = 4 \). Each is upstream × local, nothing
    else.
    """),
            "Solution 2.2": mo.md(r"""
    Training has millions of inputs-to-the-derivative (the parameters) and **one**
    scalar output (the loss). Reverse mode costs one sweep per *output*: one
    backward pass yields every \( \partial L/\partial\theta_i \). Forward mode
    costs one sweep per *input* — 600M sweeps for Qwen3-0.6B. Forward mode wins in
    the transposed regime: few inputs, many outputs (e.g. sensitivity of a full
    simulation trajectory to one physical constant), or when storing the forward
    tape is the binding constraint, since forward mode needs no tape.
    """),
        }),
    ], gap=1)
    return


@app.cell
def _():
    mo.md(r"""
    ## 3 · Softmax + cross-entropy = negative log-likelihood

    A classifier's last linear layer emits *logits* \( z \in \mathbb{R}^K \) —
    unnormalized scores. Softmax turns them into a distribution,
    \( p_k = e^{z_k} / \sum_j e^{z_j} \), and cross-entropy scores it against the
    label: \( L = -\log p_t \) for correct class \( t \). That *is* negative
    log-likelihood — "how surprised was the model by the truth" — and its gradient
    is famously clean. Since \( -\log p_t = -z_t + \log\sum_j e^{z_j} \),

    \[ \frac{\partial L}{\partial z_k}
       \;=\; -\mathbb{1}[k = t] + \frac{e^{z_k}}{\sum_j e^{z_j}}
       \;=\; p_k - \mathbb{1}[k = t] . \]

    Prediction minus target, componentwise. The correct class's gradient
    \( p_t - 1 \le 0 \) always pushes its logit up; every wrong class gets
    \( +p_k \), pushed down in proportion to the mass it stole; the components sum
    to zero, so the update only moves probability *around*. Drag the correct
    class's logit and watch the gradient bars shrink toward zero exactly as the
    green bar absorbs the probability mass.

    This object is the project's entire interface to the LLM: a policy emitting
    `ACTION: LEFT` is this classifier over the vocabulary, once per token, and
    `train.py`'s `gather` of token logprobs is \( -\text{CE} \), unreduced.
    """)
    return


@app.cell
def _():
    p3_zt = mo.ui.slider(-3.0, 5.0, step=0.1, value=0.0, label="logit of the correct class")
    p3_zt
    return (p3_zt,)


@app.cell
def _(p3_zt):
    plt.close("all")
    _z = np.array([p3_zt.value, 1.2, 0.4, -0.6])
    _t = 0
    _p = softmax_rows(_z)
    _grad = _p - np.eye(4)[_t]
    _ce = -np.log(_p[_t])

    _fig, (_axp, _axg) = plt.subplots(1, 2, figsize=(9.5, 3.4), layout="constrained")
    _cols = [C_AQUA if _k == _t else C_FAINT for _k in range(4)]
    _axp.bar(range(4), _p, color=_cols, width=0.7)
    _axp.annotate("correct class", (_t, _p[_t]), textcoords="offset points",
                  xytext=(0, 6), ha="center", color=C_AQUA, fontsize=9)
    _axp.set(ylim=(0, 1.05), xticks=range(4), xlabel="class k", ylabel="pₖ",
             title=f"softmax(z):  p(correct) = {_p[_t]:.3f},  CE = −log p = {_ce:.3f}")
    _axg.bar(range(4), _grad, color=C_ORANGE, width=0.7)
    _axg.axhline(0, color=C_INK, lw=1)
    _axg.set(ylim=(-1.05, 1.05), xticks=range(4), xlabel="class k", ylabel="∂CE/∂zₖ",
             title=f"gradient = p − onehot   (sums to {_grad.sum():+.1e})")
    _fig
    return


@app.function
def ce_grad_check(eps=1e-6):
    """Analytic (p − onehot) vs central finite differences on fixed logits.
    Returns (analytic, numeric, max abs diff)."""
    z, t = np.array([0.8, 1.2, 0.4, -0.6]), 0
    analytic = softmax_rows(z) - np.eye(4)[t]

    def ce(zz):
        return -np.log(softmax_rows(zz)[t])

    numeric = np.array([
        (ce(z + eps * np.eye(4)[k]) - ce(z - eps * np.eye(4)[k])) / (2 * eps)
        for k in range(4)
    ])
    return analytic, numeric, float(np.abs(analytic - numeric).max())


@app.cell
def _():
    _an, _num, _d = ce_grad_check()
    mo.md(
        "Verified numerically — central differences on \\( -\\log p_t \\) versus "
        "the identity, at fixed logits \\( z = (0.8, 1.2, 0.4, -0.6) \\), "
        "\\( t = 0 \\):\n\n"
        f"```\nanalytic  p - onehot : {np.array2string(_an, precision=8)}\n"
        f"numeric   central FD : {np.array2string(_num, precision=8)}\n"
        f"max |difference|     : {_d:.2e}\n```"
    )
    return


@app.cell
def _():
    mo.vstack([
        mo.md(r"""
    **Exercises 3**

    **3.1.** Show that adding a constant \( c \) to every logit leaves \( p \)
    unchanged, and that this is equivalent to the gradient identity's
    \( \sum_k \partial L/\partial z_k = 0 \).

    **3.2.** In the demo's left panel, the loss is \( -\log p_t \). Sketch (or
    reason out) \( L \) as a function of \( p_t \) near \( p_t = 1 \) and near
    \( p_t = 0 \). What does the shape imply about which examples dominate a
    training batch's gradient?
    """),
        mo.accordion({
            "Solution 3.1": mo.md(r"""
    \( e^{z_k + c} / \sum_j e^{z_j + c} = e^c e^{z_k} / (e^c \sum_j e^{z_j}) = p_k \):
    softmax only reads logit *differences*. Invariance along the direction
    \( \mathbf{1} = (1,\dots,1) \) means the directional derivative
    \( \nabla L \cdot \mathbf{1} \) must vanish — which is exactly
    \( \sum_k (p_k - \mathbb{1}[k{=}t]) = 1 - 1 = 0 \). Same fact, geometric and
    algebraic costume.
    """),
            "Solution 3.2": mo.md(r"""
    Near \( p_t = 1 \), \( -\log p_t \approx 1 - p_t \): flat, loss and gradient
    both \( \to 0 \) — confident correct examples contribute almost nothing. Near
    \( p_t = 0 \) the loss diverges like \( -\log p_t \to \infty \) and the
    correct-class gradient saturates at \( p_t - 1 \to -1 \). So a batch's gradient
    is dominated by the examples the model currently gets *wrong* — and a single
    confidently-wrong example keeps pulling with full force until fixed.
    """),
        }),
    ], gap=1)
    return


@app.cell
def _():
    mo.md(r"""
    ## 4 · Embeddings and attention

    A token id is just an index; the embedding table turns it into a learned vector,
    and from then on *geometry is meaning*: the dot product \( a \cdot b \) measures
    agreement in direction, so tokens used in similar contexts drift toward similar
    vectors. (This project's phase-2 bet lives here: "banana" never appears in RL
    training, but its embedding sits near "orange" in exactly this cosine sense —
    that proximity is the proposed transfer mechanism.)

    One attention head is a soft dictionary lookup built from three linear maps.
    Each token emits a **query** ("what am I looking for"), a **key** ("what do I
    contain"), and a **value** ("what I contribute if selected"):

    \[ \text{out} \;=\; \operatorname{softmax}\!\Big( \frac{QK^\top}{\sqrt d} \Big) V . \]

    \( QK^\top \) is every query dotted with every key — an all-pairs relevance
    table. Softmax per row turns relevance into weights, and each output is a
    *weighted average of value vectors*: a dictionary lookup where every entry is
    returned a little, in proportion to key match. The \( \sqrt d \) keeps the
    dots in softmax's soft regime (exercise 4.2).

    Below: six tokens, random embeddings, one head. The slider is the score
    divisor \( s \) — the demo generalizes the \( \sqrt d \) into a temperature.
    Small \( s \) sharpens rows toward hard one-entry lookups; large \( s \)
    flattens them toward a uniform average. The right panel shows the row for
    **"catch"**: its output (blue) is visibly the weighted blend of the value
    vectors (grey), with the weights it chose (orange).
    """)
    return


@app.cell
def _():
    p4_s = mo.ui.slider(0.5, 10.0, step=0.1, value=2.8, label="score divisor s   (√d = 2.83)")
    p4_s
    return (p4_s,)


@app.cell
def _(p4_s):
    plt.close("all")
    _rng = np.random.default_rng(4)
    _n, _d = len(ATTN_TOKENS), 8
    _E = _rng.normal(0, 1, (_n, _d))
    _Wq, _Wk, _Wv = (_rng.normal(0, 0.5, (_d, _d)) for _ in range(3))
    _Q, _K, _V = _E @ _Wq, _E @ _Wk, _E @ _Wv
    _A = softmax_rows(_Q @ _K.T / p4_s.value)
    _q = ATTN_TOKENS.index("catch")
    _out = _A[_q] @ _V

    _fig, (_axh, _axb) = plt.subplots(
        1, 2, figsize=(9.8, 3.7), layout="constrained", width_ratios=[1, 1.25]
    )
    _axh.imshow(_A, cmap="Oranges", vmin=0, vmax=1)
    _axh.set(xticks=range(_n), yticks=range(_n),
             xticklabels=ATTN_TOKENS, yticklabels=ATTN_TOKENS,
             title=f"attention = softmax(QKᵀ / {p4_s.value:.1f}), one row per query")
    _axh.tick_params(axis="x", rotation=45)
    _axh.grid(False)
    for _i in range(_n):
        for _j in range(_n):
            _axh.annotate(f"{_A[_i, _j]:.2f}", (_j, _i), ha="center", va="center",
                          fontsize=7.5, color="white" if _A[_i, _j] > 0.5 else C_MUTED)

    _dims = np.arange(_d)
    for _j in range(_n):
        _axb.plot(_dims, _A[_q, _j] * _V[_j], color=C_FAINT, lw=1.1)
    _axb.plot(_dims, _out, color=C_BLUE, lw=2.4, marker="o", ms=4, label="output = Σ wⱼ vⱼ")
    _top = int(_A[_q].argmax())
    _axb.plot(_dims, _A[_q, _top] * _V[_top], color=C_ORANGE, lw=1.6,
              label=f"biggest term: w={_A[_q, _top]:.2f} × v('{ATTN_TOKENS[_top]}')")
    _axb.axhline(0, color=C_MUTED, lw=0.8)
    _axb.set(xlabel="embedding dimension", ylabel="component value",
             title=f"the row for 'catch': output as a blend of {_n} value vectors")
    _axb.legend(frameon=False, fontsize=8.5, loc="upper right")
    _fig
    return


@app.cell
def _():
    mo.vstack([
        mo.md(r"""
    **Exercises 4**

    **4.1.** Take the limits: what does one head compute as \( s \to \infty \), and
    as \( s \to 0 \)? Name each limit in dictionary terms.

    **4.2.** For \( q, k \in \mathbb{R}^d \) with i.i.d. \( \mathcal{N}(0,1) \)
    entries, compute \( \operatorname{Var}(q \cdot k) \). Why does that make
    \( \sqrt d \) the natural divisor, and what goes wrong in the softmax without
    it at \( d = 4096 \)?
    """),
        mo.accordion({
            "Solution 4.1": mo.md(r"""
    As \( s \to \infty \) every row of scores collapses toward zero, softmax goes
    uniform, and every output is the *unweighted mean of all values* — a lookup
    that returns the whole dictionary averaged, ignoring the query. As
    \( s \to 0 \) each row concentrates on its argmax: a *hard* lookup returning
    exactly one value vector per query. Attention is the differentiable
    interpolation between those two.
    """),
            "Solution 4.2": mo.md(r"""
    \( q \cdot k = \sum_{i=1}^d q_i k_i \); each term has mean 0 and variance 1,
    terms independent, so \( \operatorname{Var}(q \cdot k) = d \) — std \( \sqrt d \).
    Dividing by \( \sqrt d \) normalizes scores to unit scale regardless of width.
    Without it, at \( d = 4096 \) the scores have std 64: softmax of numbers that
    spread saturates to near-one-hot rows, and its gradients (which look like
    \( p - \text{onehot} \), section 3) vanish — the head freezes into a hard,
    untrainable lookup from initialization onward.
    """),
        }),
    ], gap=1)
    return


@app.cell
def _():
    mo.md(r"""
    ## 5 · A language model is a next-token classifier over the vocabulary

    All four previous sections, stapled together. An LM reads a token sequence
    (embeddings, §4), mixes context (attention, §4), and at **every position**
    emits logits over the vocabulary — one section-3 classifier per position, each
    predicting the *next* token. The training objective is nothing more than
    cross-entropy at every position, summed; backprop (§2) delivers the gradient
    through the stack of bent linear maps (§1).

    Generation is the classifier run in a loop: sample a token from
    \( \operatorname{softmax}(z/T) \), append, re-predict. Temperature \( T \)
    rescales logits before softmax — \( T \to 0 \) is argmax, \( T \) large is
    uniform — changing the sampling distribution without touching the model.

    Where logprobs live, concretely: final layer → logits `[B, L, V]` →
    `log_softmax` → `gather` the entry of the token that actually came next. That
    is `train.py`'s `batched_logprobs`, off-by-one included: `logits[:, t]` scores
    token \( t{+}1 \), so the gathered logprobs and the generated-token mask are
    both shifted by one.

    The demo trains a bigram LM — the smallest model that is honestly this
    pipeline: a \( V \times V \) logit table, row = context token, trained by
    torch autograd with exactly the position-wise CE above, on pairs sampled from
    a hidden Markov chain over tokens `A…H` (each token mostly followed by its
    successor, sometimes by the token 3 ahead). Then scrub the temperature.
    """)
    return


@app.function
def lm_true_transitions():
    """Ground-truth next-token distribution: 8×8 row-stochastic matrix."""
    V = len(LM_VOCAB)
    P = np.full((V, V), 0.1 / (V - 2))
    for i in range(V):
        P[i, (i + 1) % V] = 0.7
        P[i, (i + 3) % V] = 0.2
    return P


@app.function
def lm_train(n_pairs=4096, steps=250, lr=0.1, seed=0):
    """Bigram LM as a V×V logit table, trained with torch on sampled pairs.
    Returns (logit table as numpy, loss_before, loss_after, ce_true, entropy_floor):
    losses are training-sample CE; ce_true is CE against the true chain."""
    P = lm_true_transitions()
    V = len(LM_VOCAB)
    rng = np.random.default_rng(seed)
    x = rng.integers(0, V, n_pairs)
    y = (rng.random(n_pairs)[:, None] > np.cumsum(P[x], axis=1)).sum(1)

    torch.manual_seed(seed)
    tx, ty = torch.tensor(x), torch.tensor(y)
    table = torch.zeros(V, V, requires_grad=True)
    opt = torch.optim.Adam([table], lr=lr)
    with torch.no_grad():
        loss_before = float(F.cross_entropy(table[tx], ty))
    for _ in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(table[tx], ty)   # CE at every position — the LM objective
        loss.backward()
        opt.step()
    with torch.no_grad():
        loss_after = float(F.cross_entropy(table[tx], ty))
    tab = table.detach().numpy()
    logq = tab - np.log(np.exp(tab - tab.max(1, keepdims=True)).sum(1, keepdims=True)) \
        - tab.max(1, keepdims=True)
    ce_true = float(-(P * logq).sum(1).mean())
    floor = float(-(P * np.log(P)).sum(1).mean())
    return tab, loss_before, loss_after, ce_true, floor


@app.cell
def _():
    lm_table, lm_before, lm_after, lm_ce_true, lm_floor = lm_train()
    mo.md(
        f"Trained in-cell just now: training CE **{lm_before:.4f}** (uniform init, "
        f"= ln 8 = {np.log(8):.4f}) → **{lm_after:.4f}** after 250 Adam steps. "
        f"Against the *true* chain the model scores {lm_ce_true:.4f} nats, a whisker "
        f"above the chain's conditional entropy {lm_floor:.4f} — the irreducible "
        "surprise no model can beat. (Training CE lands slightly *below* the floor: "
        "the model fits the finite sample's empirical frequencies, not the source — "
        "the same gap as any train-vs-population loss.)"
    )
    return lm_table, lm_floor


@app.cell
def _():
    p5_T = mo.ui.slider(0.1, 3.0, step=0.05, value=1.0, label="sampling temperature T")
    p5_T
    return (p5_T,)


@app.cell
def _(lm_table, p5_T):
    plt.close("all")
    _V = len(LM_VOCAB)
    _ctx = 0  # condition on 'A'
    _P_true = lm_true_transitions()
    _p1 = softmax_rows(lm_table[_ctx])                 # model at T = 1
    _pT = softmax_rows(lm_table[_ctx] / p5_T.value)    # what sampling actually uses

    _rng = np.random.default_rng(5)
    _seq, _tok = [_ctx], _ctx
    for _ in range(23):
        _tok = int(_rng.choice(_V, p=softmax_rows(lm_table[_tok] / p5_T.value)))
        _seq.append(_tok)
    _text = " ".join(LM_VOCAB[_t] for _t in _seq)

    _x = np.arange(_V)
    _fig, _ax = plt.subplots(figsize=(8.5, 3.5), layout="constrained")
    _ax.bar(_x - 0.28, _P_true[_ctx], width=0.26, color=C_FAINT, label="true chain")
    _ax.bar(_x, _p1, width=0.26, color=C_BLUE, label="model, T = 1 (learned)")
    _ax.bar(_x + 0.28, _pT, width=0.26, color=C_ORANGE, label=f"model, T = {p5_T.value:.2f} (sampled)")
    _ent = float(-(_pT * np.log(_pT + 1e-12)).sum())
    _ax.set(xticks=_x, xticklabels=LM_VOCAB, ylim=(0, 1.02),
            xlabel="next token", ylabel="p(next | 'A')",
            title=f"softmax(z/T) given context 'A' — entropy {_ent:.2f} nats at T = {p5_T.value:.2f}")
    _ax.legend(frameon=False, fontsize=9)
    mo.vstack([_fig, mo.md(f"24 tokens sampled autoregressively at this T:  `{_text}`")], gap=0.5)
    return


@app.cell
def _():
    mo.vstack([
        mo.md(r"""
    **Exercises 5**

    **5.1.** Show that \( T \to 0^+ \) turns sampling into argmax (assume a unique
    maximum logit), and that \( T \to \infty \) turns it uniform. Which direction
    does each move the sampled sequences' diversity, and why does `train.py` insist
    on neutral sampling settings during rollout?

    **5.2.** A batch of token ids has shape `[B, L]` and the model returns logits
    `[B, L, V]`. Write the two-line indexing that extracts the logprob of each
    *actually-occurring* token, and explain the off-by-one.
    """),
        mo.accordion({
            "Solution 5.1": mo.md(r"""
    Write \( p_k(T) \propto e^{z_k/T} = (e^{z_k})^{1/T} \). As \( T \to 0^+ \),
    ratios \( p_k/p_{k^*} = e^{(z_k - z_{k^*})/T} \to 0 \) for every
    \( k \neq k^* = \arg\max z \): all mass collapses onto the max — greedy
    decoding, zero diversity. As \( T \to \infty \), \( z/T \to 0 \) and softmax of
    zeros is uniform — maximum diversity, no model left. `train.py` samples at the
    raw distribution ( \( T = 1 \), no top-p/top-k) because the logprobs used in
    the loss are computed *as if* tokens came from the raw softmax; sampling from
    anything else makes \( \log \pi_\theta(\text{what happened}) \) a logprob under
    the wrong distribution, silently biasing the policy gradient.
    """),
            "Solution 5.2": mo.md(r"""
    ```python
    logp = torch.log_softmax(logits[:, :-1], dim=-1)          # [B, L-1, V]
    tok_logp = logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    ```
    The logits at position \( t \) are a prediction of token \( t{+}1 \) — the
    model has read tokens \( 0..t \) and is guessing what comes next. So
    predictions `[:, :-1]` align with targets `ids[:, 1:]`, and any mask over
    "which tokens were generated" must shift identically (`gmask[:, 1:]` in
    `batched_logprobs`). Misalign by one and every token is scored by the wrong
    classifier — the classic silent bug the docstring there warns about.
    """),
        }),
    ], gap=1)
    return


@app.cell
def _():
    mo.md(r"""
    ## 6 · The map: what unlocks what

    | primer section | what it unlocks downstream |
    |---|---|
    | 1 · bent linear maps | what \( \pi_\theta \) *is*: `grpo_notes` §1's softmax policy and §7's one-hidden-layer numpy policy are exactly this section's net with a different last-layer meaning |
    | 2 · backprop | why \( \nabla_\theta \log \pi_\theta \) is computable at all — the one quantity REINFORCE needs; it is what `loss.backward()` produces in `train.py`'s `grpo_step`, and what §7's `catch_policy_gradient` does by hand |
    | 3 · softmax + CE | `grpo_notes`' softmax identity `np.eye(3)[a] − probs` **is** \( p - \text{onehot} \) with a sign flip (ascent vs descent); `train.py`'s gathered token logprobs are \( -\text{CE} \) unreduced, so the RL loss `−(A · logprobs)` is advantage-weighted cross-entropy on the model's own actions |
    | 4 · embeddings + attention | the machinery inside Qwen3-0.6B that `train.py` treats as a black-box \( \pi_\theta \); and the phase-2 experiment — banana/orange cosine geometry across checkpoints — is a direct measurement of §4's "geometry is meaning" claim |
    | 5 · LM = next-token classifier | `batched_logprobs` end to end: `logits[:, :-1]` → `log_softmax` → `gather`, with the shifted mask; plus the sampling==learning-distribution invariant behind `main()` resetting Qwen's sampler to neutral |

    Reading order for the whole repo: `math_refresher.py` (why gradients and
    expectations behave), this file (what the model is), `grpo_notes.py` (why the
    training loop is forced), then `train.py` (the loop itself, at 600M
    parameters).
    """)
    return


if __name__ == "__main__":
    app.run()
