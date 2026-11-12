"""Every figure in the notebook.

Static matplotlib figures in one consistent style: thin marks, recessive axes,
a fixed categorical colour order, one y-axis per chart, and direct labels where
they help.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullFormatter

# Reference categorical palette, fixed order (never cycled).
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
)
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
SERIES = [BLUE, ORANGE, AQUA, YELLOW]


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


def _finish(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    return ax


# --------------------------------------------------------------------------
# Section 3: causal graph
# --------------------------------------------------------------------------


def plot_dag():
    """Causal graph of the design: Default -> PreTakeup -> Takeup, U confounds."""
    pos = {
        "Default\n(randomized)": (0, 0),
        "Peer take-up\n(pre_takeup_rate)": (1.6, 0),
        "Own take-up": (3.2, 0),
        "U: village traits\n(unobserved)": (2.4, 1.05),
        "X: household\ncontrols": (4.1, -1.05),
        "Intensive\n(randomized)": (2.3, -1.05),
    }
    edges = [
        ("Default\n(randomized)", "Peer take-up\n(pre_takeup_rate)", "solid"),
        ("Peer take-up\n(pre_takeup_rate)", "Own take-up", "solid"),
        (
            "U: village traits\n(unobserved)",
            "Peer take-up\n(pre_takeup_rate)",
            "dashed",
        ),
        ("U: village traits\n(unobserved)", "Own take-up", "dashed"),
        ("X: household\ncontrols", "Own take-up", "solid"),
        ("Intensive\n(randomized)", "Own take-up", "solid"),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.set_axis_off()
    ax.set_xlim(-0.6, 4.7)
    ax.set_ylim(-1.5, 1.45)
    boxes = {}
    for node, (x, y) in pos.items():
        unobserved = node.startswith("U")
        boxes[node] = ax.text(
            x,
            y,
            node,
            ha="center",
            va="center",
            fontsize=9.5,
            color=INK,
            zorder=3,
            bbox=dict(
                boxstyle="round,pad=0.45",
                fc="white",
                ec=MUTED if unobserved else BLUE,
                lw=1.4,
                ls="--" if unobserved else "-",
            ),
        )
    fig.canvas.draw()  # box patches need a layout before arrows can clip to them

    def arrow(a, b, color, ls, rad=0.0):
        ta, tb = boxes[a], boxes[b]
        ax.annotate(
            "",
            xy=(0.5, 0.5),
            xycoords=tb,
            xytext=(0.5, 0.5),
            textcoords=ta,
            arrowprops=dict(
                arrowstyle="-|>",
                mutation_scale=14,
                color=color,
                lw=1.4,
                ls=ls,
                shrinkA=2,
                shrinkB=2,
                patchA=ta.get_bbox_patch(),
                patchB=tb.get_bbox_patch(),
                connectionstyle=f"arc3,rad={rad}",
            ),
            zorder=4,
        )

    for a, b, style in edges:
        dashed = style == "dashed"
        arrow(a, b, MUTED if dashed else INK_2, "--" if dashed else "-")
    # The excluded path: Default must not reach Own take-up directly.
    arrow("Default\n(randomized)", "Own take-up", ORANGE, ":", rad=0.35)
    ax.text(
        0.55,
        -0.95,
        "excluded: no direct path\nfrom Default (δ = 0)",
        color=ORANGE,
        ha="center",
        fontsize=9,
    )
    ax.set_title("Figure 0. Causal graph of the design")
    return fig


# --------------------------------------------------------------------------
# Section 1: experimental design
# --------------------------------------------------------------------------


def plot_design_flow(df=None):
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
            plt.Rectangle((x - w / 2, y - h / 2), w, h, fc="white", ec="black", lw=1.1, zorder=2)
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
    r1 = [(1.6, "Simple1", dict(delay=0, intensive=0)), (5.4, "Intensive1", dict(delay=0, intensive=1))]
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
            box(x, 1.7, f"{short}-{label}\n({n(delay=1, intensive=inten, **c):,} HHs)", w=3.2, h=1.5)
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
# Section 5: data and results
# --------------------------------------------------------------------------


def plot_takeup_by_arm(cells):
    """Figure 1: take-up across the 2 x 2 of default and intensive, with 95% CIs."""
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    width = 0.34
    for j, (inten, label) in enumerate(
        [(0, "Simple session"), (1, "Intensive session")]
    ):
        sub = cells[cells["intensive"] == inten].sort_values("default")
        x = np.arange(len(sub)) + (j - 0.5) * (width + 0.02)
        ax.bar(x, sub["mean"], width=width, color=SERIES[j], label=label, zorder=2)
        ax.errorbar(
            x,
            sub["mean"],
            yerr=1.96 * sub["se"],
            fmt="none",
            ecolor=INK_2,
            elinewidth=1.2,
            capsize=3,
            zorder=3,
        )
        for xi, m in zip(x, sub["mean"], strict=True):
            ax.text(
                xi,
                0.02,
                f"{m:.2f}",
                ha="center",
                va="bottom",
                color="white",
                fontsize=9,
                fontweight="bold",
                zorder=4,
            )
    ax.set_xticks([0, 1], ["Default: not buy", "Default: buy"])
    ax.set_ylim(0, 0.7)
    ax.legend(loc="upper left", ncols=2)
    return _finish(
        ax,
        "Figure 1. Take-up by arm (2nd-round, informed farmers)",
        ylabel="Share who bought insurance",
    ).figure


def plot_first_stage(df):
    """Natural-village first-round take-up by the village's default."""
    vil = df.groupby("address").agg(
        default=("default", "first"), pre=("pre_takeup_rate", "first")
    )
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for d in (0, 1):
        v = vil.loc[vil["default"] == d, "pre"]
        ax.scatter(
            d + rng.uniform(-0.14, 0.14, len(v)),
            v,
            s=22,
            color=SERIES[d],
            alpha=0.55,
            edgecolor="white",
            linewidth=0.6,
            zorder=2,
        )
        ax.hlines(v.mean(), d - 0.25, d + 0.25, color=INK, lw=2, zorder=3)
        ax.text(
            d + 0.28,
            v.mean(),
            f"mean {v.mean():.2f}",
            va="center",
            fontsize=9,
            color=INK_2,
        )
    ax.set_xticks([0, 1], ["Default: not buy", "Default: buy"])
    ax.set_xlim(-0.5, 1.75)
    return _finish(
        ax,
        "Figure 2. First stage: 1st-round take-up by village default",
        ylabel="1st-round take-up rate\n(one dot per natural village)",
    ).figure


def plot_coefficients(table, title, xlabel="Estimate (95% CI)"):
    """Horizontal coefficient plot for any table with Estimate / CI columns."""
    t = table.iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.8, 0.42 * len(t) + 1.3))
    y = np.arange(len(t))
    ax.hlines(y, t["CI low"], t["CI high"], color=BLUE, lw=2, zorder=2)
    ax.scatter(
        t["Estimate"], y, s=40, color=BLUE, edgecolor="white", linewidth=1.5, zorder=3
    )
    ax.axvline(0, color=MUTED, lw=1, zorder=1)
    ax.set_yticks(y, t.index)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    return _finish(ax, title, xlabel=xlabel).figure


def plot_leave_one_out(loo, full_estimate):
    """Leave-one-admin-village-out 2SLS estimates, sorted."""
    s = loo.sort_values("Estimate").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    x = np.arange(len(s))
    ax.vlines(
        x,
        s["Estimate"] - 1.96 * s["Std. Error"],
        s["Estimate"] + 1.96 * s["Std. Error"],
        color=GRID,
        lw=2,
        zorder=1,
    )
    ax.scatter(x, s["Estimate"], s=18, color=BLUE, zorder=2)
    ax.axhline(full_estimate, color=ORANGE, lw=1.5, ls="--", zorder=3)
    ax.text(
        len(s) - 1,
        full_estimate + 0.05,
        f"full sample {full_estimate:.2f}",
        color=INK_2,
        ha="right",
        fontsize=9,
    )
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks([])
    return _finish(
        ax,
        "Figure 11. 2SLS dropping one admin village at a time",
        xlabel="Dropped admin village (sorted by estimate)",
        ylabel="Peer effect β",
    ).figure


# --------------------------------------------------------------------------
# Section 6: Monte Carlo
# --------------------------------------------------------------------------

COVERAGE_SERIES = {
    "coverage address": ("2SLS Wald, cluster natural village", BLUE, "-"),
    "coverage village": ("2SLS Wald, cluster admin village", ORANGE, "-"),
    "coverage ar_village": ("Anderson-Rubin, cluster admin village", AQUA, "-"),
    "coverage iid": ("2SLS Wald, iid", YELLOW, "-"),
}


def plot_coverage_vs_f(summary, observed_f, reps):
    """Headline Monte Carlo figure: CI coverage against first-stage F."""
    s = summary.sort_values("mean F")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    band = 1.96 * np.sqrt(0.95 * 0.05 / reps)
    ax.axhspan(0.95 - band, 0.95 + band, color=GRID, alpha=0.6, lw=0, zorder=0)
    ax.axhline(0.95, color=MUTED, lw=1, ls="--", zorder=1)
    for col, (label, color, ls) in COVERAGE_SERIES.items():
        if col in s:
            ax.plot(
                s["mean F"],
                s[col],
                color=color,
                ls=ls,
                marker="o",
                ms=5,
                label=label,
                zorder=3,
            )
    ax.axvline(observed_f, color=INK_2, lw=1, zorder=2)
    ax.text(
        observed_f * 1.04,
        ax.get_ylim()[1],
        f"our data: F = {observed_f:.1f}",
        color=INK_2,
        fontsize=9,
        va="top",
    )
    ax.set_xscale("log")
    ax.set_xticks([2, 5, 10, 20, 50], ["2", "5", "10", "20", "50"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncols=2)
    return _finish(
        ax,
        "Figure 4. 95% CI coverage against first-stage strength",
        xlabel="Mean first-stage F (log scale)",
        ylabel="Share of CIs containing true β",
    ).figure


def plot_bias(summary, x, xlabel, title, extra=None):
    """Median bias of OLS and 2SLS across one parameter grid."""
    s = summary.sort_values(x)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.axhline(0, color=MUTED, lw=1)
    ax.plot(s[x], s["OLS bias"], color=ORANGE, marker="o", ms=5, label="OLS")
    ax.plot(
        s[x],
        s["2SLS median bias"],
        color=BLUE,
        marker="o",
        ms=5,
        label="2SLS (median bias)",
    )
    if extra is not None:
        ax.plot(s[x], extra(s), color=MUTED, ls=":", lw=1.5, label="δ / π (theory)")
    ax.legend(loc="upper left")
    return _finish(ax, title, xlabel=xlabel, ylabel="Bias in β").figure


def plot_power(summary):
    """Rejection rate of H0: beta = 0 against the true beta, by sample size."""
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    for j, (scale, g) in enumerate(summary.groupby("n_scale")):
        g = g.sort_values("beta")
        n = int(round(1378 * scale))
        ax.plot(
            g["true beta"],
            g["reject b=0 ar_village"],
            color=SERIES[j],
            marker="o",
            ms=5,
            label=f"n ≈ {n:,}",
        )
    ax.axhline(0.8, color=MUTED, lw=1, ls="--")
    ax.text(
        ax.get_xlim()[1],
        0.81,
        "80% power",
        color=INK_2,
        fontsize=9,
        va="bottom",
        ha="right",
    )
    ax.axhline(0.05, color=MUTED, lw=1, ls=":")
    ax.legend(loc="center right", title="Sample size", title_fontsize=9)
    ax.set_ylim(0, 1.02)
    return _finish(
        ax,
        "Figure 6. Power of the Anderson-Rubin test (admin clusters)",
        xlabel="True peer effect β",
        ylabel="Share rejecting β = 0",
    ).figure


# --------------------------------------------------------------------------
# Section 7: causal forest
# --------------------------------------------------------------------------


def plot_cate_distribution(cate, ate):
    """Histogram of out-of-fold CATE estimates."""
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.hist(cate, bins=30, color=BLUE, edgecolor="white", linewidth=1, zorder=2)
    ax.axvline(ate, color=ORANGE, lw=1.5, ls="--", zorder=3)
    ax.text(
        ate,
        ax.get_ylim()[1] * 0.95,
        f"  AIPW average {ate:.3f}",
        color=INK_2,
        fontsize=9,
        va="top",
    )
    return _finish(
        ax,
        "Figure 8. Forest estimates of the default's effect (out of fold)",
        xlabel="Estimated intent-to-treat effect on take-up",
        ylabel="Households",
    ).figure


def plot_gates(g):
    """Average effect by quartile of predicted CATE, with 95% CIs."""
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    x = np.arange(len(g))
    ax.bar(x, g["Estimate"], color=BLUE, width=0.55, zorder=2)
    ax.errorbar(
        x,
        g["Estimate"],
        yerr=[g["Estimate"] - g["CI low"], g["CI high"] - g["Estimate"]],
        fmt="none",
        ecolor=INK_2,
        elinewidth=1.2,
        capsize=3,
        zorder=3,
    )
    ax.scatter(
        x,
        g["Mean predicted CATE"],
        color=ORANGE,
        marker="D",
        s=30,
        zorder=4,
        label="Forest's predicted mean",
    )
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(
        x,
        [
            f"{q}\n(lowest)" if i == 0 else f"{q}\n(highest)" if i == len(g) - 1 else q
            for i, q in enumerate(g.index)
        ],
    )
    ax.legend(loc="upper left")
    return _finish(
        ax,
        "Figure 9. Effect by quartile of predicted effect (GATES)",
        ylabel="AIPW estimate of the default's effect",
    ).figure


def plot_cate_tree(interpreter, feature_names):
    """Draw the shallow CATE tree from ``SingleTreeCateInterpreter``."""
    fig, ax = plt.subplots(figsize=(13, 4.3))
    interpreter.plot(ax=ax, feature_names=feature_names, fontsize=9, precision=3)
    ax.set_title(
        "Figure 10. Which farmers respond most to a buy default? (depth-2 tree)"
    )
    return fig


def plot_validation(summary):
    """Share of simulated runs in which the forest reports heterogeneity."""
    s = summary["Share reporting heterogeneity"]
    fig, ax = plt.subplots(figsize=(6.6, 0.5 * len(s) + 1.4))
    y = np.arange(len(s))[::-1]
    ax.barh(y, s.to_numpy(), color=BLUE, height=0.55, zorder=2)
    for yi, v in zip(y, s.to_numpy(), strict=True):
        ax.text(v + 0.01, yi, f"{v:.0%}", va="center", fontsize=9, color=INK_2)
    ax.axvline(0.05, color=MUTED, lw=1, ls=":")
    ax.set_yticks(y, s.index)
    ax.set_xlim(0, 1)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    return _finish(
        ax,
        "Figure 7. Does the forest find heterogeneity when it exists?",
        xlabel="Share of simulated runs with BLP p < 0.05",
    ).figure


def as_percent_table(df, cols):
    """Format selected columns of a summary table as percentages."""
    out = df.copy()
    for c in cols:
        out[c] = out[c].map(lambda v: f"{v:.1%}" if pd.notna(v) else "")
    return out
