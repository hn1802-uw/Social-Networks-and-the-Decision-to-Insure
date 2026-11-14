"""Unit tests for the Monte Carlo helpers."""

import numpy as np
import pytest

from auxiliary.simulation import group_means, ols_within, permute_within


def test_permute_within_keeps_each_villages_values():
    g = np.array([0, 0, 0, 1, 1, 1])
    v = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
    out = permute_within(v, g, np.random.default_rng(0))
    assert sorted(out[:3]) == [1.0, 2.0, 3.0] and sorted(out[3:]) == [10.0, 20.0, 30.0]


def test_ols_within_recovers_coefficients_with_village_effects():
    rng = np.random.default_rng(1)
    g = np.repeat(np.arange(20), 15)
    X = rng.normal(size=(g.size, 2))
    y = X @ np.array([2.0, -1.0]) + rng.normal(size=20)[g] * 5
    assert ols_within(y, X, g) == pytest.approx([2.0, -1.0])


def test_group_means_broadcasts_2d():
    out = group_means(
        np.array([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]]), np.array([0, 0, 1])
    )
    assert out.tolist() == [[2.0, 20.0], [2.0, 20.0], [5.0, 50.0]]
