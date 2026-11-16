"""Tests for the heterogeneous-effects extension."""

import pytest

from auxiliary.data import load_analysis_data
from auxiliary.heterogeneity import (
    HET_X,
    T_COL,
    fit_causal_forest,
    heterogeneity_sample,
    moderator_tests,
)


@pytest.fixture(scope="module")
def sample():
    return heterogeneity_sample(load_analysis_data())


def test_sample_matches_eq1(sample):
    assert len(sample) == 1255
    assert sample[[T_COL, *HET_X]].notna().all().all()


def test_forest_average_close_to_eq1(sample):
    forest = fit_causal_forest(sample)
    assert forest.ate(sample[HET_X].astype(float)) == pytest.approx(0.29, abs=0.08)


def test_moderator_effects_add_up(sample):
    t = moderator_tests(sample)
    assert (t["Effect when 1"] - t["Effect when 0"]).to_numpy() == pytest.approx(
        t["Difference"].to_numpy()
    )
