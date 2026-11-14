"""Robustness checks that Cai et al. (2015) report alongside the main equations.

Each ``robust_eqN`` function returns ``{column label: fit}`` for one of the four
equations in Section 5.2 of the notebook, ready for ``stacked_regression_table``.
The first column is always the baseline estimate from ``auxiliary.estimation``,
so each table shows how the headline coefficient moves across specifications.
Column labels name the paper table and column being reproduced; the specs follow
``analysis.do`` line by line (cited in each docstring).

``PAPER`` stores the published coefficient of interest for every column, and
``compare_to_paper`` puts our estimates next to them.
"""

import numpy as np
import pandas as pd
import pyfixest as pf

from auxiliary.estimation import (
    CLUSTER,
    FE,
    HH_X,
    HH_X_IV,
    NETSIZE,
    PRICE_NETSIZE,
    PRICE_X,
    STATA_SSC,
    _as_result,
    _no_info,
    _rhs,
    _round2,
    _tag,
    eq1_network_model,
    eq2_price_model,
    eq3_knowledge_model,
    eq4_info_iv_model,
    two_sls,
)

ROBUST_LABELS = {
    "network_rate_presimple": "Share of friends in 1st-round simple session",
    "inter_network_educ": "Network x Intensive",
    "network_onlyone": "Exactly 1 friend in 1st-round intensive",
    "network_onlytwo": "Exactly 2 friends in 1st-round intensive",
    "network_twomore": "2+ friends in 1st-round intensive",
    "inter_network_educ1": "1 friend x Intensive",
    "inter_network_educ2": "2 friends x Intensive",
    "inter_network_educ3": "2+ friends x Intensive",
    "inter_default_network": "Network x Default",
    "delay": "Second round (delay)",
    "inter_inten_delay": "Intensive x Second round",
    "network_yes": "Has a friend in 1st-round intensive",
    "inter_inten_netyes": "Intensive x Has such a friend",
    "knowledge_network": "Friends' knowledge score (IV)",
    "price_high_ratio": "Share of friends facing a higher price",
    "price_low_ratio": "Share of friends facing a lower price",
    "inter_pre_takeup_iv": "Default x network in 1st round session",
    "network_rate_presession": "Share of friends in any 1st-round session",
    "inter_pre_none_iv": "Default x No info",
    "inter_prenetwork_none_iv1": "Default x network x No info",
}


def _ols(df, y, regressors, controls=(), netsize=(), fe=FE, cluster=CLUSTER):
    """OLS with Stata-matching clustered SEs, tagged for the table helpers."""
    rhs = _rhs([*regressors, *controls, *netsize])
    fit = pf.feols(f"{y} ~ {rhs} | {fe}", df, vcov={"CRV1": cluster}, ssc=STATA_SSC)
    return _tag(fit, controls, netsize, fe)


def _iv(df, y, endog, instruments, exog, controls, netsize=(), fe=FE, small=True):
    """2SLS tagged for the table helpers; ``small`` as in ``two_sls``."""
    fit = two_sls(df, y, endog, instruments, exog=[*exog, *controls, *netsize], fe=fe, small=small)
    fit.coefs = fit.coefs.drop(index="Intercept")  # absorbed by the FE in the paper
    return _tag(fit, controls, netsize, fe, iv=True)


# --------------------------------------------------------------------------
# Eq. 1: network effect on take-up (paper Tables 2, 3 and A4)
# --------------------------------------------------------------------------


def robust_eq1(df_full):
    """Robustness of the network effect, ``analysis.do`` lines 77-99, 109 and 210.

    * Simple-session spillover (Table 2 col. 3): add the share of friends in a
      first-round *simple* session. The paper drops X here, so we do too.
    * Network x own Intensive (Table 2 col. 4).
    * Nonlinear network effect (Table 2 col. 5, Table 3 col. 3): dummies for
      1, 2 and 2+ friends in a first-round intensive session.
    * Network x Default (Table A4 col. 1).
    * Non-friend spillover (Table 2 col. 6): does an intensive session matter
      less in round 2 for households with no informed friend? If round-2 farmers
      learned from people outside their friend list, Intensive x Delay < 0.
    """
    sub = _no_info(df_full).copy()
    sub["inter_network_educ"] = sub["network_rate_preintensive"] * sub["intensive"]
    for k in (1, 2, 3):
        dummy = ["network_onlyone", "network_onlytwo", "network_twomore"][k - 1]
        sub[f"inter_network_educ{k}"] = sub[dummy] * sub["intensive"]
    sub["inter_default_network"] = sub["network_rate_preintensive"] * sub["default"]

    nonlinear = ["network_onlyone", "network_onlytwo", "network_twomore"]
    fits = {
        "Baseline (T2 c2)": eq1_network_model(df_full),
        "Simple spillover (T2 c3)": _ols(
            sub, "takeup_survey", ["network_rate_preintensive", "network_rate_presimple", "intensive"],
            netsize=NETSIZE,
        ),
        "x Intensive (T2 c4)": _ols(
            sub, "takeup_survey", ["network_rate_preintensive", "intensive", "inter_network_educ"],
            HH_X, NETSIZE,
        ),
        "Nonlinear x Int. (T2 c5)": _ols(
            sub, "takeup_survey",
            [*nonlinear, "intensive", "inter_network_educ1", "inter_network_educ2", "inter_network_educ3"],
            HH_X, NETSIZE,
        ),
        "Nonlinear (T3 c3)": _ols(sub, "takeup_survey", [*nonlinear, "intensive"], HH_X, NETSIZE),
        "x Default (TA4 c1)": _ols(
            sub, "takeup_survey",
            ["network_rate_preintensive", "default", "inter_default_network", "intensive"],
            HH_X, NETSIZE,
        ),
    }

    # Table 2 col. 6: round 1 plus round-2 no-info households with no friend
    # in a first-round intensive session (``nofriend`` in analysis.do).
    d = df_full.copy()
    informed_friend = (d["delay"] == 1) & (d["info_none"] == 1) & (d["network_yes"] == 1)
    d = d[(d["delay"] == 0) | ((d["delay"] == 1) & (d["info_none"] == 1) & ~informed_friend)]
    d["inter_inten_delay"] = d["intensive"] * d["delay"]
    fits["Non-friend spillover (T2 c6)"] = _ols(
        d, "takeup_survey", ["intensive", "delay", "inter_inten_delay"], HH_X
    )
    return fits


# --------------------------------------------------------------------------
# Eq. 2: price equivalence (paper Tables 4 and 7)
# --------------------------------------------------------------------------


def _price_sample(price_df):
    d = price_df.copy()
    d["price_x_net"] = d["price"] * d["network_rate_preintensive"]
    for k in range(1, 9):
        d[f"friend{k}"] = (d["network_obs"] == k).astype(float)
    d.loc[d["network_obs"].isna(), PRICE_NETSIZE] = np.nan
    return d


def robust_eq2(price_df):
    """Robustness of the price model, ``analysis.do`` lines 142-156.

    * No interaction (Table 4 col. 1): the average network effect in price terms.
    * Friends' prices (Table 4 col. 3): adds the shares of friends who faced a
      higher or a lower price, to rule out farmers anchoring on friends' prices.
    * Friends' take-up, IV (Table 7 col. 9): friends' take-up instrumented by
      the average price friends faced (``ivregress``, so ``small=False``).
    """
    d = _price_sample(price_df)
    fe = "address"
    base = ["price", "network_rate_preintensive"]
    return {
        "Baseline (T4 c2)": eq2_price_model(price_df),
        "No interaction (T4 c1)": _ols(
            d, "takeup", [*base, "intensive"], PRICE_X, PRICE_NETSIZE, fe=fe
        ),
        "Friends' prices (T4 c3)": _ols(
            d, "takeup", [*base, "price_x_net", "price_high_ratio", "price_low_ratio", "intensive"],
            PRICE_X, PRICE_NETSIZE, fe=fe,
        ),
        "Friends' take-up IV (T7 c9)": _iv(
            d, "takeup", ["network_rate_pretakeup"], ["network_avgprice"],
            ["price", "intensive"], PRICE_X, PRICE_NETSIZE, fe=fe, small=False,
        ),
    }


# --------------------------------------------------------------------------
# Eq. 3: knowledge channel (paper Table 5)
# --------------------------------------------------------------------------


def robust_eq3(df_full):
    """Robustness of the knowledge channel, ``analysis.do`` lines 166-175.

    * Intensive x Delay (Table 5 col. 1): all no-info households; the intensive
      session raises knowledge much less in round 2, once friends have attended.
    * Intensive x Has-a-friend (Table 5 col. 2): the same idea within round 2.
    * Friends' knowledge, IV (Table 5 col. 4): own knowledge on friends' average
      knowledge score, instrumented by the network share (``ivreg``, ``small=True``).
    """
    no_info = df_full[df_full["info_none"] == 1].copy()
    no_info["inter_inten_delay"] = no_info["intensive"] * no_info["delay"]
    sub = _no_info(df_full).copy()
    sub["inter_inten_netyes"] = sub["intensive"] * sub["network_yes"]
    return {
        "Baseline (T5 c3)": eq3_knowledge_model(df_full),
        "Intensive x Delay (T5 c1)": _ols(
            no_info, "understanding", ["intensive", "delay", "inter_inten_delay"], HH_X
        ),
        "x Has friend (T5 c2)": _ols(
            sub, "understanding", ["intensive", "network_yes", "inter_inten_netyes"], HH_X
        ),
        "Friends' knowledge IV (T5 c4)": _iv(
            sub, "understanding", ["knowledge_network"], ["network_rate_preintensive"],
            ["intensive"], HH_X,
        ),
    }


# --------------------------------------------------------------------------
# Eq. 4: friends' decisions vs. overall take-up (paper Tables 6 and 7)
# --------------------------------------------------------------------------


def _eq4_sample(df_full):
    """Round 2 without the households told only the overall rate (Table 7)."""
    d = _round2(df_full).copy()
    d["inter_pre_none"] = d["pre_takeup_rate"] * d["info_none"]
    d["inter_pre_none_iv"] = d["default"] * d["info_none"]
    d["inter_pre_takeup_iv"] = d["network_rate_presession"] * d["default"]
    d["inter_prenetwork_none"] = d["network_rate_pretakeup"] * d["info_none"]
    d["inter_prenetwork_none_iv1"] = d["inter_pre_takeup_iv"] * d["info_none"]
    return d


def robust_eq4(df_full):
    """Robustness of the friends-vs-overall result, ``analysis.do`` lines 224-269.

    * First stage (Table 7 col. 1): friends' take-up on the instrument
      default x share of friends in a first-round session. Its strength is
      what makes the IV columns credible.
    * Pooled OLS (Table 7 col. 2) and pooled IV (Table 7 col. 3): informed and
      no-info households together, with No-info interactions. The interaction
      terms test the gap between the two arms directly.
    * Pooled reduced form (Table 7 col. 6).
    * Overall rate only (Table 6 cols. 2-5): the same exercise without the
      friends' rate, OLS and IV, pooled and by arm.

    The pooled IV uses ``ivreg`` (``small=True``); the by-arm IVs use
    ``ivregress`` (``small=False``), as in the do-file.
    """
    d = _eq4_sample(df_full)
    t7 = d[d["info_takeup_rate"] != 1]
    X, X6 = HH_X_IV, HH_X

    fits = {
        "Baseline IV informed (T7 c4)": eq4_info_iv_model(df_full)["(4) IV, informed"],
        "First stage (T7 c1)": _ols(t7, "network_rate_pretakeup", ["inter_pre_takeup_iv", "intensive"], X),
        "Pooled OLS (T7 c2)": _ols(
            t7, "takeup_survey",
            ["pre_takeup_rate", "network_rate_pretakeup", "info_none", "inter_pre_none",
             "inter_prenetwork_none", "intensive"], X,
        ),
        "Pooled IV (T7 c3)": _iv(
            t7, "takeup_survey",
            ["network_rate_pretakeup", "pre_takeup_rate", "inter_pre_none", "inter_prenetwork_none"],
            ["inter_pre_takeup_iv", "network_rate_presession", "default", "inter_pre_none_iv",
             "inter_prenetwork_none_iv1"],
            ["info_none", "intensive"], X,
        ),
        "Pooled RF (T7 c6)": _ols(
            t7, "takeup_survey",
            ["inter_pre_takeup_iv", "default", "inter_pre_none_iv", "inter_prenetwork_none_iv1",
             "info_none", "intensive"], X,
        ),
    }

    overall = {
        "Overall OLS pooled (T6 c2)": _ols(
            d, "takeup_survey", ["pre_takeup_rate", "info_none", "inter_pre_none", "intensive"], X6
        ),
        "Overall IV pooled (T6 c3)": _iv(
            d, "takeup_survey", ["pre_takeup_rate", "inter_pre_none"], ["default", "inter_pre_none_iv"],
            ["info_none", "intensive"], X6,
        ),
        "Overall IV informed (T6 c4)": _iv(
            d[d["info_none"] == 0], "takeup_survey", ["pre_takeup_rate"], ["default"],
            ["intensive"], X6, small=False,
        ),
        "Overall IV no info (T6 c5)": _iv(
            d[d["info_none"] == 1], "takeup_survey", ["pre_takeup_rate"], ["default"],
            ["intensive"], X6, small=False,
        ),
    }
    return fits, overall


# --------------------------------------------------------------------------
# Comparison with the published numbers
# --------------------------------------------------------------------------

PAPER = {
    # column label: (regressor, published estimate, published SE, published N)
    "Baseline (T2 c2)": ("network_rate_preintensive", 0.291, 0.0820, 1255),
    "Simple spillover (T2 c3)": ("network_rate_preintensive", 0.278, 0.0845, 1274),
    "x Intensive (T2 c4)": ("network_rate_preintensive", 0.444, 0.109, 1255),
    "Nonlinear x Int. (T2 c5)": ("network_onlyone", 0.0970, 0.0425, 1255),
    "Nonlinear (T3 c3)": ("network_onlyone", 0.0531, 0.0315, 1255),
    "x Default (TA4 c1)": ("network_rate_preintensive", 0.286, 0.121, 1255),
    "Non-friend spillover (T2 c6)": ("inter_inten_delay", -0.0525, 0.0468, 2756),
    "Baseline (T4 c2)": ("price_x_net", 0.125, 0.0489, 433),
    "No interaction (T4 c1)": ("network_rate_preintensive", 0.353, 0.112, 433),
    "Friends' prices (T4 c3)": ("price_x_net", 0.121, 0.0500, 433),
    "Friends' take-up IV (T7 c9)": ("network_rate_pretakeup", 0.0990, 0.205, 405),
    "Baseline (T5 c3)": ("network_rate_preintensive", 0.290, 0.0488, 1255),
    "Intensive x Delay (T5 c1)": ("inter_inten_delay", -0.250, 0.0200, 3262),
    "x Has friend (T5 c2)": ("inter_inten_netyes", -0.231, 0.0331, 1255),
    "Friends' knowledge IV (T5 c4)": ("knowledge_network", 0.414, 0.0799, 958),
    "Baseline IV informed (T7 c4)": ("network_rate_pretakeup", 0.579, 0.216, 613),
    "First stage (T7 c1)": ("inter_pre_takeup_iv", 0.302, 0.0604, 1500),
    "Pooled OLS (T7 c2)": ("network_rate_pretakeup", 0.106, 0.0495, 1500),
    "Pooled IV (T7 c3)": ("network_rate_pretakeup", 0.556, 0.259, 1500),
    "Pooled RF (T7 c6)": ("inter_prenetwork_none_iv1", -0.246, 0.114, 1930),
    "Overall OLS pooled (T6 c2)": ("pre_takeup_rate", 0.378, 0.0680, 2674),
    "Overall IV pooled (T6 c3)": ("pre_takeup_rate", 0.719, 0.235, 2674),
    "Overall IV informed (T6 c4)": ("pre_takeup_rate", 0.791, 0.267, 1378),
    "Overall IV no info (T6 c5)": ("pre_takeup_rate", 0.0171, 0.325, 1296),
}


def robustness_summary(groups):
    """One row per robustness check: the key coefficient, its SE, p-value and N,
    with the published estimate alongside.

    ``groups`` is ``{equation label: {column label: fit}}``; the equation label
    becomes the outer index level.
    """
    out = pd.concat({eq: compare_to_paper(fits) for eq, fits in groups.items()})
    out = out.rename(columns={"Ours": "Estimate", "Ours SE": "Std. Error"})
    return out[["Coefficient", "Estimate", "Std. Error", "P-Value", "N", "Paper"]]


def compare_to_paper(fits, labels=None):
    """Our estimate of each column's key coefficient next to the published one."""
    from auxiliary.data import VARIABLE_LABELS
    from auxiliary.estimation import REGRESSOR_LABELS

    labels = {**VARIABLE_LABELS, **REGRESSOR_LABELS, **ROBUST_LABELS, **(labels or {})}
    rows = []
    for name, fit in fits.items():
        var, b, se, n = PAPER[name]
        res = _as_result(fit)
        rows.append(
            {
                "Specification": name,
                "Coefficient": labels.get(var, var),
                "Ours": res.coefs.loc[var, "Estimate"],
                "Paper": b,
                "Ours SE": res.coefs.loc[var, "Std. Error"],
                "Paper SE": se,
                "P-Value": res.coefs.loc[var, "P-Value"],
                "N": res.n,
                "Paper N": n,
            }
        )
    return pd.DataFrame(rows).set_index("Specification")
