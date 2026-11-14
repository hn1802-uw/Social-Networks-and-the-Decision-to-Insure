"""Tables and regressions used in ``project.ipynb``.

OLS models use ``pyfixest`` with village fixed effects and natural-village
clustered standard errors; ``two_sls`` wraps ``linearmodels.IV2SLS`` for the
2SLS with several endogenous regressors that ``pyfixest`` cannot run.
"""

import numpy as np
import pandas as pd
import pyfixest as pf
from linearmodels.iv import IV2SLS
from scipy import stats

FE = "village"
CLUSTER = "address"


# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------


def color_pvalues(value):
    """Colour p-values: red < 0.001, orange < 0.01, yellow < 0.05, else none."""
    if not isinstance(value, (int, float)) or pd.isna(value):
        return ""
    if value < 0.001:
        return "color: red"
    if value < 0.01:
        return "color: darkorange"
    if value < 0.05:
        return "color: goldenrod"
    return ""


def style_table(table, pcol="P-Value", digits=3):
    """Round numeric columns and colour the p-value column."""
    def _num_or_text(v):
        if isinstance(v, str):
            return v
        if pd.isna(v):
            return ""
        return f"{v:.{digits}f}"

    fmt = {c: f"{{:.{digits}f}}" for c in table.select_dtypes("number").columns}
    fmt.update({c: _num_or_text for c in table.select_dtypes("object").columns})
    for col in ("N", "Missing", "missing"):
        if col in table.columns:
            fmt[col] = lambda v: "" if pd.isna(v) else f"{v:.0f}"
    # Min / Max: show whole numbers without decimals (0, 1, 86), others as usual.
    def _whole_or_float(v):
        if pd.isna(v):
            return ""
        return f"{v:.0f}" if float(v).is_integer() else f"{v:.{digits}f}"

    for col in ("Min", "Max"):
        if col in table.columns:
            fmt[col] = _whole_or_float
    styler = table.style.format(fmt, na_rep="")
    if pcol in table.columns:
        styler = styler.map(color_pvalues, subset=[pcol])
    return styler


def _rhs(controls):
    return " + ".join(controls) if controls else "1"


# --------------------------------------------------------------------------
# Descriptive and balance tables
# --------------------------------------------------------------------------


def summary_table(df, columns, labels=None):
    """Table 1: count, mean, SD, min, max and missing values."""
    labels = labels or {}
    out = df[columns].agg(["count", "mean", "std", "min", "max"]).T
    out["missing"] = df[columns].isna().sum().astype(int)
    out.columns = ["N", "Mean", "SD", "Min", "Max", "Missing"]
    out[["N", "Missing"]] = out[["N", "Missing"]].astype(int)
    out.index = [labels.get(c, c) for c in out.index]
    return out


def balance_table(df, treatment, covariates, labels=None, fe=None, cluster=CLUSTER):
    """Table 2: covariate means by treatment arm, with clustered p-values.

    The p-value comes from regressing each covariate on ``treatment`` (plus
    ``fe`` fixed effects if given) with standard errors clustered by ``cluster``.
    A small difference with a large p-value is what randomization predicts.
    """
    labels = labels or {}
    rows = []
    for cov in covariates:
        sub = df.dropna(subset=[cov, treatment])
        fml = f"{cov} ~ {treatment}" + (f" | {fe}" if fe else "")
        fit = pf.feols(fml, data=sub, vcov={"CRV1": cluster})
        t = fit.tidy().loc[treatment]
        rows.append(
            {
                "Variable": labels.get(cov, cov),
                f"{treatment} = 0": sub.loc[sub[treatment] == 0, cov].mean(),
                f"{treatment} = 1": sub.loc[sub[treatment] == 1, cov].mean(),
                "Difference": t["Estimate"],
                "Std. Error": t["Std. Error"],
                "P-Value": t["Pr(>|t|)"],
            }
        )
    return pd.DataFrame(rows).set_index("Variable")


PRICE_BALANCE_VARS = ["age", "agpop", "earlyarea", "literacy", "male", "risk_averse", "intensive"]


def price_balance_table(price_df, covariates=PRICE_BALANCE_VARS, labels=None):
    """Price randomization check: ``X_ij = a0 + a1 Price_ij + eta_j + e_ij``.

    ``eta_j`` is a natural-village fixed effect (price is randomized within
    natural village) and standard errors are clustered by natural village. The
    ``Price`` row of each regression is the test; a large p-value is what random
    price assignment predicts.
    """
    out = balance_table(
        price_df, "price", covariates, labels=labels, fe="address", cluster="address"
    )
    # Price is continuous, so arm means do not apply; the slope is the effect.
    return out.drop(columns=["price = 0", "price = 1"]).rename(
        columns={"Difference": "Slope on price"}
    )


# --------------------------------------------------------------------------
# The four equations of Section 4
# --------------------------------------------------------------------------

NETSIZE = [f"friend{k}" for k in range(1, 6)]
"""NetSize: dummies for the number of friends listed (1 to 5)."""

STATA_SSC = pf.ssc(k_fixef="full")
"""Count every fixed effect in the small-sample correction, as Stata's dummy-variable
regressions in ``analysis.do`` do, so standard errors match the paper exactly."""

PRICE_NETSIZE = [f"friend{k}" for k in range(1, 9)]
"""NetSize in the price sample, where farmers listed up to 8 friends."""

HH_X = ["male", "age", "agpop", "ricearea_2010", "literacy", "risk_averse", "disaster_prob"]
"""Household characteristics X in the paper's Tables 2, 5 and 6 (``analysis.do``)."""

HH_X_IV = ["male", "age", "agpop", "ricearea_2010", "literacy"]
"""Household characteristics X in the paper's Table 7 (no risk or disaster measures)."""

PRICE_X = ["male", "age", "agpop", "earlyarea", "literacy"]
"""X in the price sample (Table 4): the price file records rice area as ``earlyarea``."""

REGRESSOR_LABELS = {
    "Intercept": "Constant",
    "network_rate_preintensive": "Network (share of friends in 1st-round intensive)",
    "price": "Price (RMB/mu)",
    "price_x_net": "Price x Network",
    "info_none": "No info (told nothing)",
    "pre_takeup_rate": "TakeupRate (village)",
    "network_rate_pretakeup": "TakeupRateNetwork (friends)",
    "inter_pre_none": "TakeupRate x No info",
    "inter_prenetwork_none": "TakeupRateNetwork x No info",
}


STACK_LABELS = {
    "inter_pre_takeup_iv": "Default x network in 1st round session",
    "default": "1st round default (1 = buy, 0 = not buy)",
    "network_rate_pretakeup": "1st round network's take-up rate",
    "pre_takeup_rate": "1st round overall take-up rate",
}


class RegResult:
    """Minimal fit container for estimators pyfixest cannot run (e.g. 2SLS with
    several endogenous regressors), so ``regression_table`` treats all fits alike."""

    def __init__(self, coefs, n, r2, adj_r2):
        self.coefs = coefs  # DataFrame: Estimate, Std. Error, P-Value
        self.n = n
        self.r2 = r2
        self.adj_r2 = adj_r2


def _as_result(fit):
    """Convert a pyfixest fit to a ``RegResult``."""
    if isinstance(fit, RegResult):
        return fit
    t = fit.tidy()
    coefs = t[["Estimate", "Std. Error", "Pr(>|t|)"]].rename(columns={"Pr(>|t|)": "P-Value"})
    return RegResult(coefs, int(fit._N), float(fit._r2), float(fit._adj_r2))


def _tag(fit, controls=(), netsize=(), fe=None, iv=False):
    """Record which regressors ``regression_table`` should collapse into Yes/No rows."""
    fit.table_spec = dict(controls=list(controls), netsize=list(netsize), fe=fe, iv=iv)
    return fit


def regression_table(fit, labels=None):
    """Results table for one fit, laid out like the paper's tables.

    One row per regressor of interest with Estimate, Std. Error, P-Value and N.
    Household characteristics, number-of-friends dummies and fixed effects are
    estimated but not listed: they are summarised as Yes/No rows instead. The
    last two rows are R-squared and adjusted R-squared; for 2SLS they are left
    blank, as Stata does, since they are not a measure of fit there (and are
    often negative). Pass the result to ``style_table`` to colour p-values.
    """
    from auxiliary.data import VARIABLE_LABELS

    res = _as_result(fit)
    spec = getattr(fit, "table_spec", dict(controls=[], netsize=[], fe=None, iv=False))
    labels = {**VARIABLE_LABELS, **REGRESSOR_LABELS, **(labels or {})}
    hidden = set(spec["controls"]) | set(spec["netsize"])

    table = res.coefs.loc[[v for v in res.coefs.index if v not in hidden]].astype(object)
    table["N"] = res.n
    table.index = [labels.get(v, v) for v in table.index]

    r2 = (np.nan, np.nan) if spec["iv"] else (res.r2, res.adj_r2)
    footer = pd.DataFrame(
        {
            "Estimate": [
                *("Yes" if spec[k] else "No" for k in ("controls", "netsize", "fe")),
                *r2,
            ]
        },
        index=[
            "Household characteristics",
            "Number-of-friends dummies",
            "Village fixed effects",
            "R-squared",
            "Adjusted R-squared",
        ],
        dtype=object,
    )
    return pd.concat([table, footer])[["Estimate", "Std. Error", "P-Value", "N"]]


def two_sls(df, y, endog, instruments, exog=(), fe=None, cluster=CLUSTER, small=True):
    """2SLS with any number of endogenous regressors, via ``linearmodels.IV2SLS``.

    Includes a constant; ``fe`` enters as dummies. Standard errors are clustered
    with the Stata small-sample correction, and p-values use a t distribution
    with G - 1 degrees of freedom, so the output reproduces Stata's
    ``ivreg ..., cluster()``. R-squared uses the structural residuals, so it can
    be negative.

    ``small=False`` drops the small-sample correction and uses normal p-values,
    as Stata's ``ivregress 2sls`` does (the paper's Table 7 cols. 4 and 6).
    """
    endog, instruments, exog = list(endog), list(instruments), list(exog)
    cols = [y, *endog, *instruments, *exog, cluster] + ([fe] if fe else [])
    d = df.dropna(subset=cols)
    W = d[exog].astype(float)
    W.insert(0, "Intercept", 1.0)
    if fe:
        W = pd.concat(
            [W, pd.get_dummies(d[fe], prefix="fe", drop_first=True, dtype=float)], axis=1
        )
    # Stata silently omits collinear regressors (e.g. a friend-count dummy that
    # is all zero in the estimation sample); do the same so the fit runs.
    _, r = np.linalg.qr(W.to_numpy())
    W = W.loc[:, np.abs(np.diag(r)) > 1e-10 * np.abs(r).max()]
    clusters = pd.factorize(d[cluster])[0]
    res = IV2SLS(
        d[y].astype(float), W, d[endog].astype(float), d[instruments].astype(float)
    ).fit(cov_type="clustered", clusters=clusters, debiased=small)

    # Account for the difference between IV2SLS in Python and Strata (used by paper)
    G = clusters.max() + 1
    coefs = pd.DataFrame(
        {
            "Estimate": res.params,
            "Std. Error": res.std_errors,
            "P-Value": 2 * (stats.t.sf(np.abs(res.tstats), G - 1) if small else stats.norm.sf(np.abs(res.tstats))),
        }
    )
    coefs = coefs.loc[[v for v in coefs.index if not v.startswith("fe_")]]
    return RegResult(coefs, int(res.nobs), float(res.rsquared), float(res.rsquared_adj))


def _round2(df_full, informed=None):
    """Second-round households; ``informed`` = True/False picks the info arm."""
    r2 = df_full[df_full["delay"] == 1]
    if informed is None:
        return r2
    return r2[r2["info_none"] == (0 if informed else 1)]


def _no_info(df_full):
    return _round2(df_full, informed=False)


def eq1_network_model(df_full, controls=HH_X, fe=FE, cluster=CLUSTER):
    """Eq. (1), paper Table 2 col. 2: Takeup on Network, Intensive, X, NetSize, eta_j.

    Sample: second-round households given no take-up information
    (Simple2-NoInfo and Intensive2-NoInfo). ``analysis.do`` line 77.
    """
    sub = _no_info(df_full)
    rhs = _rhs(["network_rate_preintensive", "intensive", *controls, *NETSIZE])
    fit = pf.feols(f"takeup_survey ~ {rhs} | {fe}", sub, vcov={"CRV1": cluster}, ssc=STATA_SSC)
    return _tag(fit, controls, NETSIZE, fe)


def eq2_price_model(price_df, controls=PRICE_X, fe="address", cluster=CLUSTER):
    """Eq. (2), paper Table 4 col. 2: Takeup on Price, Network, Price x Network,
    Intensive, X, NetSize and eta_j.

    Sample: type II (price-randomized) villages. Price is randomized within
    natural village, so ``eta_j`` is the natural-village (``address``) effect.
    ``analysis.do`` line 144.
    """
    d = price_df.copy()
    d["price_x_net"] = d["price"] * d["network_rate_preintensive"]
    for k in range(1, 9):
        d[f"friend{k}"] = (d["network_obs"] == k).astype(float)
    d.loc[d["network_obs"].isna(), PRICE_NETSIZE] = np.nan
    rhs = _rhs(
        ["price", "network_rate_preintensive", "price_x_net", "intensive", *controls, *PRICE_NETSIZE]
    )
    fit = pf.feols(f"takeup ~ {rhs} | {fe}", d, vcov={"CRV1": cluster}, ssc=STATA_SSC)
    return _tag(fit, controls, PRICE_NETSIZE, fe)


def eq3_knowledge_model(df_full, controls=HH_X, fe=FE, cluster=CLUSTER):
    """Eq. (3), paper Table 5 col. 3: Knowledge on Network, Intensive, X, eta_j.

    Knowledge is the ten-question test score (``understanding``). Sample:
    second-round households given no take-up information. ``analysis.do`` line 172.
    """
    sub = _no_info(df_full)
    rhs = _rhs(["network_rate_preintensive", "intensive", *controls])
    fit = pf.feols(f"understanding ~ {rhs} | {fe}", sub, vcov={"CRV1": cluster}, ssc=STATA_SSC)
    return _tag(fit, controls, (), fe)


def eq4_info_iv_model(df_full, controls=HH_X_IV, fe=FE, cluster=CLUSTER):
    """Eq. (4), paper Table 7 cols. 4-7: friends' decisions vs. overall take-up.

    Returns ``{label: fit}`` for four second-round regressions of take-up:

    * (4) IV, households shown the first-round list of decisions
      (``info_takeup_list == 1``). Endogenous: TakeupRate and TakeupRateNetwork;
      instruments: default and default x share of friends in a first-round session.
    * (5) Reduced form on that subsample: take-up on the two instruments.
    * (6) IV, households told nothing (``info_none == 1``).
    * (7) Reduced form on that subsample.

    The reduced forms do not use the network rate, so they keep households with
    a missing one and have more observations than the IV columns, as in the
    paper. ``analysis.do`` lines 259-269 (the do-file's column comments are off
    by one; the layout here follows the published table).
    """
    d = _round2(df_full)
    d["inter_pre_takeup_iv"] = d["network_rate_presession"] * d["default"]
    arms = {"informed": d[d["info_takeup_list"] == 1], "no info": d[d["info_none"] == 1]}
    endog = ["network_rate_pretakeup", "pre_takeup_rate"]
    instr = ["inter_pre_takeup_iv", "default"]

    fits = {}
    for (iv_col, rf_col), (arm, sub) in zip([(4, 5), (6, 7)], arms.items()):
        iv = two_sls(
            sub, "takeup_survey", endog, instr, exog=["intensive", *controls], fe=fe,
            cluster=cluster, small=False,
        )
        iv.coefs = iv.coefs.drop(index="Intercept")  # absorbed by the FE in the paper
        fits[f"({iv_col}) IV, {arm}"] = _tag(iv, [*controls, "intensive"], (), fe, iv=True)
        rhs = _rhs([*instr, "intensive", *controls])
        rf = pf.feols(f"takeup_survey ~ {rhs} | {fe}", sub, vcov={"CRV1": cluster}, ssc=STATA_SSC)
        fits[f"({rf_col}) RF-OLS, {arm}"] = _tag(rf, [*controls, "intensive"], (), fe)
    return fits


def stacked_regression_table(fits, labels=None, digits=3):
    """Table with one Estimate / P-Value column pair per fit (cf. the paper's Table 7).

    ``fits`` is ``{column label: fit}``. Standard errors and significance stars
    are left out; the p-values are coloured with the same code as ``style_table``
    (red < 0.001, orange < 0.01, yellow < 0.05). Observations, fixed effects,
    controls and R-squared (blank for 2SLS) follow the regressors. Returns a
    styled table ready to display.
    """
    from auxiliary.data import VARIABLE_LABELS

    labels = {**VARIABLE_LABELS, **REGRESSOR_LABELS, **STACK_LABELS, **(labels or {})}
    results = {k: _as_result(f) for k, f in fits.items()}
    specs = {k: getattr(f, "table_spec", {}) for k, f in fits.items()}
    hidden = {v for sp in specs.values() for v in (*sp.get("controls", []), *sp.get("netsize", []))}

    order = []
    for r in results.values():
        order += [v for v in r.coefs.index if v not in hidden and v not in order]

    cols = pd.MultiIndex.from_product([list(results), ["Estimate", "P-Value"]])
    table = pd.DataFrame(index=[labels.get(v, v) for v in order], columns=cols, dtype=object)
    for k, r in results.items():
        for v in order:
            if v in r.coefs.index:
                table.loc[labels.get(v, v), (k, "Estimate")] = r.coefs.loc[v, "Estimate"]
                table.loc[labels.get(v, v), (k, "P-Value")] = r.coefs.loc[v, "P-Value"]

    def yes(k, key):
        return "Yes" if specs[k].get(key) else "No"

    footer = pd.DataFrame(index=["Observations", "Village fixed effects", "Household characteristics", "R-squared"], columns=cols, dtype=object)
    for k, r in results.items():
        footer[(k, "Estimate")] = [
            f"{r.n:,}", yes(k, "fe"), yes(k, "controls"),
            "" if specs[k].get("iv") else f"{r.r2:.{digits}f}",
        ]
    table = pd.concat([table, footer])

    def _fmt(v):
        if isinstance(v, str):
            return v
        return "" if pd.isna(v) else f"{v:.{digits}f}"

    pcols = [c for c in table.columns if c[1] == "P-Value"]
    return table.style.format(_fmt).map(color_pvalues, subset=pcols)
