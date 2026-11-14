"""The robustness columns reproduce the paper's published estimates."""

import warnings

import pytest

from auxiliary.data import load_analysis_data, load_price_data
from auxiliary.robustness import (
    PAPER,
    compare_to_paper,
    robust_eq1,
    robust_eq2,
    robust_eq3,
    robust_eq4,
)


@pytest.fixture(scope="module")
def comparison():
    warnings.filterwarnings("ignore")
    full, price = load_analysis_data(), load_price_data()
    eq4, eq4_overall = robust_eq4(full)
    fits = {**robust_eq1(full), **robust_eq2(price), **robust_eq3(full), **eq4, **eq4_overall}
    return compare_to_paper(fits)


def test_every_paper_column_is_estimated(comparison):
    assert set(comparison.index) == set(PAPER)


@pytest.mark.parametrize("spec", list(PAPER))
def test_matches_published_table(comparison, spec):
    row = comparison.loc[spec]
    # The paper prints three significant digits.
    assert row["Ours"] == pytest.approx(row["Paper"], abs=6e-4)
    assert row["Ours SE"] == pytest.approx(row["Paper SE"], abs=6e-4)
    assert row["N"] == row["Paper N"]
