"""Unit tests for the data build and the estimation helpers."""

import numpy as np
import pandas as pd
import pyfixest as pf
import pytest

from auxiliary.data import build_network_variables
from auxiliary.estimation import two_sls


@pytest.fixture
def iv_data():
    rng = np.random.default_rng(0)
    G, m = 120, 8
    cl = np.repeat(np.arange(G), m)
    fe = cl // 4
    z = rng.binomial(1, 0.5, G)[cl].astype(float)
    u = rng.normal(0, 0.2, G)[cl]
    d = 0.3 + 0.3 * z + u + rng.normal(0, 0.1, G)[cl]
    w = rng.normal(size=len(cl))
    y = 0.5 * d + 0.4 * u + 0.1 * w + rng.normal(0, 0.5, len(cl))
    return pd.DataFrame({"y": y, "d": d, "z": z, "w": w, "fe": fe, "cl": cl})


def test_two_sls_matches_pyfixest(iv_data):
    df = iv_data
    fit = pf.feols("y ~ w | fe | d ~ z", df, vcov={"CRV1": "cl"})
    res = two_sls(df, "y", ["d"], ["z"], ["w"], fe="fe", cluster="cl")
    assert res.coefs.loc["d", "Estimate"] == pytest.approx(fit.coef()["d"], rel=1e-6)


def test_network_variables_follow_rawnet_do():
    raw = pd.DataFrame(
        {
            "id": [1, 1, 1, 2, 2],
            "network_id": [10, 11, 12, 10, np.nan],
            "network_missname": [0, 0, 0, 0, 1],
            "delay": [0, 0, 1, 0, np.nan],
            "intensive": [1, 0, 1, 1, np.nan],
            "takeup_survey": [1, 0, 1, 1, np.nan],
            "understanding": [0.5, 0.2, 0.9, 0.5, np.nan],
        }
    )
    out = build_network_variables(raw).set_index("id")
    # Household 1: three friends, two in round 1, one of them intensive.
    assert out.loc[1, "network_obs"] == 3
    assert out.loc[1, "network_rate_preintensive"] == pytest.approx(1 / 3)
    assert out.loc[1, "network_rate_pretakeup"] == pytest.approx(1 / 2)
    # Household 2: the unnamed friend is dropped, as in rawnet.do.
    assert out.loc[2, "network_obs"] == 1
    assert out.loc[2, "network_rate_preintensive"] == 1
