"""Load the Cai, de Janvry & Sadoulet (2015) replication data and build variables.

The replication package (openICPSR 113593) ships raw Stata files only. This module
rebuilds, in pandas, the network variables the authors construct in ``rawnet.do``
and the village take-up rate they construct in ``analysis.do``, so that the
notebook can work from the ``.dta`` files alone.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

CONTROLS = ["age", "agpop", "ricearea_2010", "literacy", "male", "risk_averse"]
"""Household controls that must be non-missing (``disaster_prob`` excluded)."""

MAIN_CONTROLS = ["age", "agpop", "ricearea_capped", "literacy", "male", "risk_averse"]
"""Controls X in the main specification, with rice area capped by ``clean``."""

"""The paper's controls in Table 6 (``analysis.do``): raw rice area, plus ``disaster_prob``."""

VARIABLE_LABELS = {
    "takeup_survey": "Insurance take-up",
    "pre_takeup_rate": "1st-round take-up rate (natural village)",
    "default": "Default = buy",
    "intensive": "Intensive session",
    "delay": "2nd-round session",
    "info_none": "No take-up information",
    "age": "Age of household head",
    "agpop": "Household size",
    "ricearea_2010": "Rice area (mu)",
    "ricearea_capped": "Rice area, capped at p99 (mu)",
    "literacy": "Literate",
    "male": "Male household head",
    "risk_averse": "Risk aversion",
    "disaster_prob": "Perceived disaster probability (%)",
    "understanding": "Insurance knowledge score",
    "network_obs": "Number of named friends",
    "network_rate_preintensive": "Share of friends in 1st-round intensive session",
    "network_rate_pretakeup": "1st-round friends' take-up rate",
}


def _read(name, data_dir=DATA_DIR):
    return pd.read_stata(Path(data_dir) / f"{name}.dta", convert_categoricals=False)


def build_network_variables(rawnet):
    """Household-level network measures, following section 1 of ``rawnet.do``.

    Parameters
    ----------
    rawnet : pandas.DataFrame
        ``0422allinforawnet``: one row per (household, named friend) pair, with the
        friend's session assignment and take-up.

    Returns
    -------
    pandas.DataFrame
        One row per household ``id``.
    """
    net = rawnet[~((rawnet["network_missname"] == 1) & rawnet["network_id"].isna())]
    net = net.copy()

    first_round = net["delay"] == 0
    net["pre_intensive"] = (first_round & (net["intensive"] == 1)).astype(float)
    net["presession"] = first_round.astype(float)
    net["pretakeup"] = (first_round & (net["takeup_survey"] == 1)).astype(float)
    net["pretakeup_obs"] = (first_round & net["takeup_survey"].notna()).astype(float)
    net["understanding_pre"] = net["understanding"].where(first_round)

    g = net.groupby("id")
    out = pd.DataFrame(
        {
            "network_obs": g.size(),
            "sum_preintensive": g["pre_intensive"].sum(),
            "sum_presession": g["presession"].sum(),
            "sum_pretakeup": g["pretakeup"].sum(),
            "n_pretakeup_obs": g["pretakeup_obs"].sum(),
            "knowledge_network": g["understanding_pre"].mean(),
        }
    )
    out.loc[out["n_pretakeup_obs"] == 0, "sum_pretakeup"] = np.nan

    out["network_rate_preintensive"] = out["sum_preintensive"] / out["network_obs"]
    out["network_rate_presession"] = out["sum_presession"] / out["network_obs"]
    # Stata divides by zero to missing; pandas gives inf/nan, so mask explicitly.
    out["network_rate_pretakeup"] = out["sum_pretakeup"] / out["sum_presession"].where(
        out["sum_presession"] > 0
    )
    out["network_rate_presimple"] = (
        out["network_rate_presession"] - out["network_rate_preintensive"]
    )

    for k in range(1, 6):
        out[f"friend{k}"] = (out["network_obs"] == k).astype(float)

    n_pre = out["sum_preintensive"]
    out["network_onlyone"] = (n_pre == 1).astype(float)
    out["network_onlytwo"] = (n_pre == 2).astype(float)
    out["network_twomore"] = (n_pre > 1).astype(float)
    out["network_yes"] = (out["network_rate_preintensive"] > 0).astype(float)

    keep = [
        "network_obs",
        "network_rate_preintensive",
        "network_rate_presimple",
        "network_rate_presession",
        "network_rate_pretakeup",
        "knowledge_network",
        *[f"friend{k}" for k in range(1, 6)],
        "network_onlyone",
        "network_onlytwo",
        "network_twomore",
        "network_yes",
    ]
    return out[keep].reset_index()


def add_village_takeup(df):
    """Add ``pre_takeup_rate``: mean first-round take-up in each natural village.

    Mirrors section 7 of ``analysis.do``. The rate is computed on all first-round
    households in the natural village (``address``) and assigned to everyone in it.
    """
    rate = df.loc[df["delay"] == 0].groupby("address")["takeup_survey"].mean()
    out = df.copy()
    out["pre_takeup_rate"] = out["address"].map(rate)
    return out


def load_analysis_data(data_dir=DATA_DIR):
    """Survey data merged with network variables and the village take-up rate.

    This is the Python equivalent of ``0422analysis.dta`` (the network
    structure measures computed in UCINET are left out).
    """
    survey = _read("0422survey", data_dir)
    rawnet = _read("0422allinforawnet", data_dir)
    df = survey.merge(build_network_variables(rawnet), on="id", how="left")
    df = add_village_takeup(df)
    df["session"] = df["delay"].astype(int) * 10 + df["intensive"].astype(int)
    return df


def load_price_data(data_dir=DATA_DIR):
    """The price-randomization sample (``0422price.dta``), used in Table 4 of the paper."""
    return _read("0422price", data_dir)


def main_sample(df):
    """Second-round households that were told about first-round take-up.

    ``delay == 1`` and ``info_none == 0``: 1,410 households in 166 natural
    villages. This is the estimation sample of the paper's Table 6, column 5, and
    exactly the ``causaldata.social_insure`` extract.
    """
    out = df[(df["delay"] == 1) & (df["info_none"] == 0)].copy()
    return out.reset_index(drop=True)


def placebo_sample(df):
    """Second-round households told nothing about first-round take-up.

    ``delay == 1`` and ``info_none == 1`` (the paper's Table 6, column 7). The
    default still shifts their village's first-round take-up, but they never
    learn it, so a peer effect here would point to an exclusion violation.
    """
    out = df[(df["delay"] == 1) & (df["info_none"] == 1)].copy()
    return out.reset_index(drop=True)


def clean(df, controls=CONTROLS, cap_var="ricearea_2010", cap_quantile=0.99):
    """Cap rice-area outliers and drop rows with a missing control.

    Parameters
    ----------
    df : pandas.DataFrame
    controls : list of str
        Columns that must be non-missing.
    cap_var : str
        Variable whose upper tail is capped. A capped copy is stored as
        ``ricearea_capped`` and the original column is left untouched.
    cap_quantile : float
        Quantile at which to cap.

    Returns
    -------
    clean_df : pandas.DataFrame
    report : pandas.DataFrame
        Missing values per variable before dropping, and rows kept.
    """
    missing = df[controls].isna().sum()
    report = pd.DataFrame({"missing": missing})
    report.loc["rows dropped (any missing)"] = int(
        df[controls].isna().any(axis=1).sum()
    )

    out = df.dropna(subset=controls).copy()
    cap = out[cap_var].quantile(cap_quantile)
    out["ricearea_capped"] = out[cap_var].clip(upper=cap)
    report.loc[f"{cap_var} cap (p{int(cap_quantile * 100)})"] = round(cap, 1)
    report.loc["rows kept"] = len(out)
    return out.reset_index(drop=True), report


def check_data(df):
    """Assertions that the constructed data have the structure the design needs.

    Raises ``AssertionError`` with a message if a check fails; returns a small
    table of the checks that passed otherwise.
    """
    checks = {}

    varying = df.groupby("address")["default"].transform("nunique") > 1
    n_vary = int(df.loc[varying, "address"].nunique())
    assert n_vary <= 1, f"default varies within {n_vary} natural villages"
    checks[f"default constant within natural village (exceptions: {n_vary})"] = True

    within = df.groupby("address")["pre_takeup_rate"].nunique().max()
    assert within <= 1, "pre_takeup_rate varies within a natural village"
    checks["pre_takeup_rate constant within natural village"] = True

    nested = df.groupby("address")["village"].nunique().max()
    assert nested == 1, "a natural village belongs to more than one admin village"
    checks["natural villages nested in admin villages"] = True

    for col in ["takeup_survey", "default", "intensive", "delay", "info_none"]:
        assert set(df[col].dropna().unique()) <= {0, 1}, f"{col} is not binary"
    checks["treatment and outcome indicators binary"] = True

    rates = df["pre_takeup_rate"].dropna()
    assert rates.between(0, 1).all(), "pre_takeup_rate outside [0, 1]"
    checks["pre_takeup_rate in [0, 1]"] = True

    return pd.Series(checks, name="passed").to_frame()
