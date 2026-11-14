"""Every figure in the notebook.

Static matplotlib figures in one consistent style: thin marks, recessive axes,
a fixed categorical colour order, one y-axis per chart, and direct labels where
they help.
"""

import matplotlib.pyplot as plt
import numpy as np

# Reference categorical palette, fixed order (never cycled).
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, RED = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#ff0000",
)
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"


def set_style():
    """Matplotlib defaults used by every figure."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2,
            "font.family": "sans-serif",
        }
    )


set_style()


# --------------------------------------------------------------------------
# Section 3: causal graph
# --------------------------------------------------------------------------


def plot_dag():
    """Causal DAG of the design (Cai et al. 2015): randomization blocks the confounder.

    Nine nodes. Randomized nodes (shaded) have no arrows into them, so nothing
    unobserved can bias them. The unobserved node (village trust, homophily,
    norms) drives peers' take-up and own take-up: the reflection problem. The
    village default is an instrument because it has no arrow from the unobserved
    node and, by the paper's tests, no direct arrow to take-up.
    """
    from matplotlib.patches import FancyBboxPatch

    W = 2.3
    # (x, y, name, body, kind) with kind in {rand, obs, hid, out}
    nodes = {
        "net": (
            1.3,
            5.0,
            "NET",
            "share of listed friends in a\nround-1 intensive session",
            "rand",
        ),
        "own": (1.3, 3.9, "Own session", "Intensive vs simple", "rand"),
        "default": (1.3, 1.7, "Village default", "Buy vs not buy", "rand"),
        "know": (4.6, 4.45, "Knowledge", "Test score", "obs"),
        "peers": (4.6, 1.7, "Peers' take-up", "Round-1 rate", "obs"),
        "u": (4.6, 0.1, "Unobserved", "Trust, Ties", "hid"),
        "price": (8.0, 5.0, "Price", "7 random levels", "rand"),
        "take": (8.0, 3.3, "Take-up", "Buys insurance", "out"),
        "info": (
            8.0,
            1.5,
            "Info revealed",
            "None / overall rate /\nindividual list",
            "rand",
        ),
    }
    # Edges: (from, to, style).
    edges = [
        ("net", "know", "solid"),
        ("own", "know", "solid"),
        ("know", "take", "solid"),
        ("default", "peers", "solid"),
        ("peers", "take", "solid"),
        ("info", "take", "solid"),
        ("price", "take", "solid"),
        ("u", "peers", "dashed"),
        ("u", "take", "dashed"),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    ax.set_axis_off()
    ax.set_xlim(-0.2, 9.6)
    ax.set_ylim(-0.7, 5.8)
    ax.set_aspect("equal")

    patches = {}
    heights = {}
    for key, (x, y, name, body, kind) in nodes.items():
        rand, hid, out = kind == "rand", kind == "hid", kind == "out"
        H = 0.62 + 0.2 * body.count("\n")
        heights[key] = H
        patch = FancyBboxPatch(
            (x - W / 2, y - H / 2),
            W,
            H,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            fc=(*plt.matplotlib.colors.to_rgb(BLUE), 0.12) if rand else "white",
            ec=BLUE if rand else MUTED,
            lw=2.2 if out else 1.4,
            ls="--" if hid else "-",
            zorder=2,
        )
        ax.add_patch(patch)
        patches[key] = patch
        top = y + H / 2 - 0.2
        ax.text(
            x,
            top,
            name,
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color=INK,
            zorder=3,
        )
        ax.text(
            x,
            top - 0.1 - 0.1 * (body.count("\n") + 1),
            body,
            ha="center",
            va="center",
            fontsize=8,
            color=INK_2,
            zorder=3,
            linespacing=1.15,
        )

    for a, b, style in edges:
        (xa, ya, *_), (xb, yb, *_) = nodes[a], nodes[b]
        dashed = style == "dashed"
        ax.annotate(
            "",
            xy=(xb, yb),
            xytext=(xa, ya),
            arrowprops=dict(
                arrowstyle="-|>",
                mutation_scale=13,
                color=MUTED if dashed else INK_2,
                lw=1.3,
                ls="--" if dashed else "-",
                shrinkA=0,
                shrinkB=0,
                patchA=patches[a],
                patchB=patches[b],
            ),
            zorder=1,
        )

    ax.text(
        0.0,
        5.75,
        "Causual Directed Acyclic Graph (DAG)",
        fontsize=11,
        fontweight="bold",
        color=INK,
        va="top",
    )
    # Key
    ky = -0.55
    for x, label, fc, ec, ls in [
        (
            0.0,
            "Randomized by design",
            (*plt.matplotlib.colors.to_rgb(BLUE), 0.12),
            BLUE,
            "-",
        ),
        (3.0, "Observed, not randomized", "white", MUTED, "-"),
        (6.0, "Unobserved confounding variables", "white", MUTED, "--"),
    ]:
        ax.add_patch(
            FancyBboxPatch(
                (x, ky - 0.1),
                0.3,
                0.2,
                boxstyle="round,pad=0.01,rounding_size=0.05",
                fc=fc,
                ec=ec,
                lw=1.2,
                ls=ls,
            )
        )
        ax.text(x + 0.42, ky, label, va="center", fontsize=8.5, color=INK_2)
    return fig


# --------------------------------------------------------------------------
# Section 1: experimental design
# --------------------------------------------------------------------------


def plot_household_design(df=None):
    """Figure 1.1: within-village, household-level randomization tree.

    Households are randomized to one of four sessions (simple/intensive, first or
    second round); second-round households are further randomized into what they
    are told about first-round take-up. Counts are computed from ``df``
    (``load_analysis_data``, loaded if omitted).
    """
    if df is None:
        from auxiliary.data import load_analysis_data

        df = load_analysis_data()

    def n(**cond):
        m = np.ones(len(df), dtype=bool)
        for col, val in cond.items():
            m &= (df[col] == val).to_numpy()
        return int(m.sum())

    fig, ax = plt.subplots(figsize=(13, 4.3))
    ax.set_axis_off()
    ax.set_xlim(-0.2, 28.8)
    ax.set_ylim(-0.3, 8.9)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    kw = dict(color="black", lw=1.1, zorder=1)

    def box(x, y, text, w=3.0, h=1.5):
        ax.add_patch(
            plt.Rectangle(
                (x - w / 2, y - h / 2), w, h, fc="white", ec="black", lw=1.1, zorder=2
            )
        )
        ax.text(x, y, text, ha="center", va="center", fontsize=9, zorder=3)

    def tree(px, py, ph, xs, cy, ch):
        """Parent box bottom -> bus -> each child's top."""
        top, bus = py - ph / 2, (py - ph / 2 + cy + ch / 2) / 2
        ax.plot([px, px], [top, bus], **kw)
        ax.plot([min(xs), max(xs)], [bus, bus], **kw)
        for x in xs:
            ax.plot([x, x], [bus, cy + ch / 2], **kw)

    # Level 0: the two rounds.
    box(3.0, 8.0, "First round\nsessions", w=3.4, h=1.4)
    box(18.0, 8.0, "Second round\nsessions", w=3.4, h=1.4)
    ax.plot([4.7, 16.3], [8.0, 8.0], **kw)
    ax.text(10.5, 8.25, "3 days later", ha="center", fontsize=10)

    # Level 1: session type.
    r1 = [
        (1.6, "Simple1", dict(delay=0, intensive=0)),
        (5.4, "Intensive1", dict(delay=0, intensive=1)),
    ]
    for x, label, c in r1:
        box(x, 5.4, f"{label}\n({n(**c):,} HHs)", w=3.2)
    tree(3.0, 8.0, 1.4, [1.6, 5.4], 5.4, 1.5)

    r2 = [(12.6, "Simple2", 0), (23.4, "Intensive2", 1)]
    for x, label, inten in r2:
        box(x, 5.4, f"{label}\n({n(delay=1, intensive=inten):,} HHs)", w=3.2)
    tree(18.0, 8.0, 1.4, [12.6, 23.4], 5.4, 1.5)

    # Level 2: information arms for second-round households.
    for x0, short, inten in [(12.6, "Simple2", 0), (23.4, "Intens2", 1)]:
        xs = [x0 - 3.6, x0, x0 + 3.6]
        arms = [
            ("NoInfo", dict(info_none=1)),
            ("Overall", dict(info_none=0, info_takeup_list=0)),
            ("Indiv", dict(info_none=0, info_takeup_list=1)),
        ]
        for x, (label, c) in zip(xs, arms, strict=True):
            box(
                x,
                1.7,
                f"{short}-{label}\n({n(delay=1, intensive=inten, **c):,} HHs)",
                w=3.2,
                h=1.5,
            )
        tree(x0, 5.4, 1.5, xs, 1.7, 1.5)

    ax.text(
        14.3,  # centre of the tree: from Simple1's left edge to Intens2-Indiv's right
        -0.1,
        "Figure 1.1. Experimental Design: Within-Village, Household-Level Randomization",
        fontsize=12,
        fontweight="bold",
        ha="center",
    )
    return fig


def plot_village_design(df=None, price=None):
    """Figure 1.2: village-level randomization tree.

    Villages are split into Type I (no price variation) and Type II (price
    variation); Type I villages are randomized to a first-round default of buy
    (IA) or not buy (IB). Counts come from the survey (``df``) and price
    (``price``) files, loaded if omitted. A village counts as default-buy if any
    household is coded ``default == 1``. This reproduces the paper's 85/88 split;
    one village (xinlian1314) has 1 of 23 households coded buy, and a majority
    rule would give 84/89.
    """
    if df is None or price is None:
        from auxiliary.data import load_analysis_data, load_price_data

        df = load_analysis_data() if df is None else df
        price = load_price_data() if price is None else price

    default = df.groupby("address")["default"].max().astype(int)
    n_i, n_ii = len(default), price["address"].nunique()
    n_buy, n_not = int((default == 1).sum()), int((default == 0).sum())

    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.set_axis_off()
    ax.set_xlim(-0.2, 28.2)
    ax.set_ylim(-0.4, 9.6)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    kw = dict(color="black", lw=1.1, zorder=1)

    def box(x, y, text, w=8.0, h=1.9):
        ax.add_patch(
            plt.Rectangle(
                (x - w / 2, y - h / 2), w, h, fc="white", ec="black", lw=1.1, zorder=2
            )
        )
        ax.text(x, y, text, ha="center", va="center", fontsize=11, zorder=3)

    def tree(px, py, xs, cy, h=1.9):
        top, bus = py - h / 2, (py - h / 2 + cy + h / 2) / 2
        ax.plot([px, px], [top, bus], **kw)
        ax.plot([min(xs + [px]), max(xs + [px])], [bus, bus], **kw)
        for x in xs:
            ax.plot([x, x], [bus, cy + h / 2], **kw)

    box(14.2, 8.4, f"Sample villages\n({n_i + n_ii} villages)", w=8.4)
    box(9.0, 5.0, f"Type I ({n_i} villages)\nprice variation = no", w=8.6)
    box(23.2, 5.0, f"Type II ({n_ii} villages)\nprice variation = yes", w=8.6)
    tree(14.2, 8.4, [9.0, 23.2], 5.0)
    box(4.3, 1.6, f"Type IA ({n_buy} villages)\n1st round default = buy", w=8.4)
    box(13.7, 1.6, f"Type IB ({n_not} villages)\n1st round default = not buy", w=8.4)
    tree(9.0, 5.0, [4.3, 13.7], 1.6)

    ax.text(
        14.0,
        -0.1,
        "Figure 1.2. Experimental Design: Village-Level Randomization",
        fontsize=12,
        fontweight="bold",
        ha="center",
    )
    return fig


# --------------------------------------------------------------------------
# Section 6: Monte Carlo simulation
# --------------------------------------------------------------------------


def plot_simulated_estimates(estimates, true_effect):
    """Figure 6.1: sampling distribution of the Knowledge slope under random assignment.

    Parameters
    ----------
    estimates : array-like
        Slope estimates, one per simulation.
    true_effect : float
        The coefficient built into the simulated outcomes.
    """
    estimates = np.asarray(estimates)

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.hist(
        estimates, bins=30, color=AQUA, alpha=0.75, label="Estimates", edgecolor="black"
    )
    ax.axvline(
        estimates.mean(), color=RED, lw=2.5, label=f"Mean = {estimates.mean():.3f}"
    )
    ax.axvline(
        true_effect,
        color=INK,
        ls="--",
        lw=1.6,
        label=f"True effect = {true_effect:.3f}",
    )
    ax.grid(False)
    ax.set_xlabel(r"Estimated effect of Knowledge on take-up ($\beta_0$)")
    ax.set_ylabel("Number of simulations")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_sample_size_convergence(sample_sizes, estimates_by_size, true_effect):
    """Figure 6.2: the estimates concentrate on the truth as the sample grows.

    Left: one histogram per sample size, drawn on common bins. Right: the standard
    deviation of the estimates against a 1/sqrt(n) reference line anchored at the
    largest sample size.

    Parameters
    ----------
    sample_sizes : list of int
    estimates_by_size : dict
        ``{n: array of estimates}``.
    true_effect : float
        The coefficient built into the simulated outcomes.
    """
    data = [np.asarray(estimates_by_size[n]) for n in sample_sizes]
    bins = np.linspace(min(d.min() for d in data), max(d.max() for d in data), 130)
    # Viridis from purple (small samples) to green (large samples).
    colors = plt.cm.viridis(np.linspace(0.1, 1.0, len(data)))

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.7, 1]}
    )
    for n, d, color in zip(sample_sizes, data, colors, strict=True):
        ax.hist(
            d,
            bins=bins,
            color=color,
            alpha=0.5,
            edgecolor=INK,
            linewidth=0.4,
            label=f"n = {n:,}",
        )
    ax.axvline(
        true_effect,
        color="#e8312f",
        ls="--",
        lw=2,
        label=f"True effect = {true_effect:.3f}",
    )
    ax.grid(False)
    ax.set_title("Sampling distributions by sample size")
    ax.set_xlabel(r"Estimated effect of Knowledge on take-up ($\beta_0$)")
    ax.set_ylabel("Number of simulations")
    ax.legend(loc="upper right")

    n = np.array(sample_sizes, dtype=float)
    sds = np.array([d.std() for d in data])
    ax2.plot(n, sds, marker="o", ms=8, color=BLUE, label="Simulated")
    ax2.plot(
        n,
        sds[-1] * np.sqrt(n[-1] / n),
        ls="--",
        lw=1.5,
        color=MUTED,
        label=r"$\propto 1/\sqrt{n}$",
    )
    ax2.set_title("Uncertainty decreases with sample size")
    ax2.set_xlabel("Sample size (n)")
    ax2.set_ylabel("Standard deviation of estimate")
    ax2.legend(loc="upper right")

    fig.suptitle(
        "Figure 6.2. Convergence as the sample size grows",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig
