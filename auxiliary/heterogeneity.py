import numpy as np
import pandas as pd
import pyfixest as pf
from econml.cate_interpreter import SingleTreeCateInterpreter
from econml.dml import CausalForestDML, LinearDML
from sklearn.linear_model import LassoCV

from auxiliary.estimation import CLUSTER, FE, HH_X, NETSIZE, STATA_SSC, _no_info

Y_COL = "takeup_survey"
T_COL = "network_rate_preintensive"

HET_X = [
    "male",
    "age",
    "agpop",
    "ricearea_capped",
    "literacy",
    "risk_averse",
    "disaster_prob",
    "intensive",
    "general_trust",
]
"""Candidate moderators X: the paper's household controls, own session, trust."""

HET_LABELS = {
    "male": "Male household head",
    "age": "Age of household head",
    "agpop": "Household size",
    "ricearea_capped": "Rice area (mu, capped)",
    "literacy": "Literate",
    "risk_averse": "Risk aversion",
    "disaster_prob": "Perceived disaster probability (%)",
    "intensive": "Own session intensive",
    "general_trust": "Trusts others",
}

SEED = 481


def heterogeneity_sample(df_full, cap_quantile=0.99):
    """The Eq. (1) sample with complete controls and rice area capped at p99."""
    sub = _no_info(df_full).dropna(subset=[Y_COL, T_COL, *HH_X, *HET_X[-2:]])
    sub = sub.reset_index(drop=True).copy()
    cap = sub["ricearea_2010"].quantile(cap_quantile)
    sub["ricearea_capped"] = sub["ricearea_2010"].clip(upper=cap)
    return sub


def _arrays(sample):
    """Y, T, X, W and the cluster groups for EconML."""
    Y = sample[Y_COL].to_numpy(float)
    T = sample[T_COL].to_numpy(float)
    X = sample[HET_X].astype(float)
    W = pd.concat(
        [
            sample[NETSIZE].astype(float),
            pd.get_dummies(sample[FE], prefix="v", drop_first=True, dtype=float),
        ],
        axis=1,
    )
    return Y, T, X, W, sample[CLUSTER].to_numpy()


def _forest(seed):
    # Lasso nuisances absorb the village dummies; outcome is a linear probability
    # model, as in the paper. Leaves of >= 30 households keep the small sample stable.
    return CausalForestDML(
        model_y=LassoCV(),
        model_t=LassoCV(),
        n_estimators=2000,
        min_samples_leaf=30,
        cv=5,
        random_state=seed,
    )


def fit_causal_forest(sample, seed=SEED):
    """Fit ``CausalForestDML``; cross-fitting folds keep natural villages together."""
    Y, T, X, W, groups = _arrays(sample)
    return _forest(seed).fit(Y, T, X=X, W=W, groups=groups)


def fit_linear_dml(sample, seed=SEED):
    """Fit ``LinearDML``: theta(X) = X'beta, the linear benchmark."""
    Y, T, X, W, groups = _arrays(sample)
    est = LinearDML(model_y=LassoCV(), model_t=LassoCV(), cv=5, random_state=seed)
    return est.fit(Y, T, X=X, W=W, groups=groups)


def ate_table(forest, sample, eq1_fit):
    """Average effect from the forest next to the Eq. (1) estimate."""
    X = sample[HET_X].astype(float)
    inf = forest.ate_inference(X)
    lo, hi = inf.conf_int_mean()
    eq1 = eq1_fit.tidy().loc[T_COL]
    rows = {
        "Causal forest (average of theta(X))": [
            inf.mean_point,
            inf.stderr_mean,
            lo,
            hi,
            len(sample),
        ],
        "Eq. 1 OLS (Section 5.2.1)": [
            eq1["Estimate"],
            eq1["Std. Error"],
            eq1["2.5%"],
            eq1["97.5%"],
            int(eq1_fit._N),
        ],
    }
    return pd.DataFrame(
        rows, index=["Estimate", "Std. Error", "CI low", "CI high", "N"]
    ).T


def cate_distribution(forest, sample):
    """Summary of the household effects theta(X) and how many are significant."""
    X = sample[HET_X].astype(float)
    te = forest.effect(X)
    lo, hi = forest.effect_interval(X)
    out = pd.Series(te).describe()[["mean", "std", "min", "25%", "50%", "75%", "max"]]
    out["share with CI above 0"] = np.mean(lo > 0)
    return out.to_frame("theta(X)")


def importance_table(forests):
    """Mean and range of forest split importance across several seeds."""
    imp = pd.DataFrame([f.feature_importances_ for f in forests], columns=HET_X)
    out = pd.DataFrame(
        {"Importance": imp.mean(), "Min": imp.min(), "Max": imp.max()}
    ).sort_values("Importance", ascending=False)
    out.index = [HET_LABELS[v] for v in out.index]
    return out


def linear_dml_table(est):
    """``LinearDML`` coefficients: how theta(X) moves with each X."""
    s = est.coef__inference().summary_frame()
    out = pd.DataFrame(
        {
            "Estimate": s["point_estimate"].to_numpy(),
            "Std. Error": s["stderr"].to_numpy(),
            "P-Value": s["pvalue"].to_numpy(),
        },
        index=[HET_LABELS[v] for v in HET_X],
    )
    return out


def interpret_tree(forest, sample, max_depth=2, min_samples_leaf=100):
    """A shallow tree that summarises theta(X) into a few household types."""
    X = sample[HET_X].astype(float)
    intp = SingleTreeCateInterpreter(
        max_depth=max_depth, min_samples_leaf=min_samples_leaf
    )
    return intp.interpret(forest, X)


def _eq1_rhs(extra=()):
    return " + ".join([T_COL, *extra, "intensive", *HH_X, *NETSIZE])


MODERATORS = {
    "intensive": "Own session intensive",
    "small_farm": "Rice area at or below median",
    "old": "Age above median",
    "risk_hi": "Risk aversion above 0",
    "literacy": "Literate",
}


def moderator_tests(sample, moderators=MODERATORS):
    """Eq. (1) with Network x moderator, one binary moderator at a time.

    Clustered by natural village like every other table, so these are the
    conventional test of whether the forest's splits are statistically real.
    """
    d = sample.copy()
    d["small_farm"] = (d["ricearea_capped"] <= d["ricearea_capped"].median()).astype(
        float
    )
    d["old"] = (d["age"] > d["age"].median()).astype(float)
    d["risk_hi"] = (d["risk_averse"] > 0).astype(float)

    rows = []
    for m, label in moderators.items():
        d["net_x_m"] = d[T_COL] * d[m]
        extra = ["net_x_m"] + ([] if m in [*HH_X, "intensive"] else [m])
        fit = pf.feols(
            f"{Y_COL} ~ {_eq1_rhs(extra)} | {FE}",
            d,
            vcov={"CRV1": CLUSTER},
            ssc=STATA_SSC,
        )
        t = fit.tidy()
        base, inter = t.loc[T_COL], t.loc["net_x_m"]
        rows.append(
            {
                "Moderator (= 1)": label,
                "Effect when 0": base["Estimate"],
                "Effect when 1": base["Estimate"] + inter["Estimate"],
                "Difference": inter["Estimate"],
                "Std. Error": inter["Std. Error"],
                "P-Value": inter["Pr(>|t|)"],
            }
        )
    return pd.DataFrame(rows).set_index("Moderator (= 1)")
