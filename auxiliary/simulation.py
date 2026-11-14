"""Monte Carlo simulation: sampling variation of the Knowledge slope.

The simulated world keeps the real households of the paper's main sample (second-round
households shown their friends' first-round decisions) and builds outcomes with a
known truth:

    y_i = Y0_i + beta0 * Knowledge_i

``Knowledge`` is the insurance knowledge score. ``Y0_i`` is observed take-up with the
real link to ``Knowledge`` projected out, and ``beta0`` is the coefficient on the real
data, treated as the truth. In every simulation the observed ``Knowledge`` values are
randomly permuted among households of the same village, so ``Knowledge`` is unrelated
to ``Y0``, and ``beta0`` is re-estimated with village fixed effects.

Outcomes are linear (a linear probability model) and are not forced into [0, 1].
"""

import numpy as np


def group_means(values, groups):
    """Group means broadcast back to each row (works for 1-D and 2-D ``values``)."""
    codes = np.unique(groups, return_inverse=True)[1]
    counts = np.bincount(codes)
    values = np.asarray(values, float)
    if values.ndim == 1:
        return (np.bincount(codes, values) / counts)[codes]
    cols = [group_means(values[:, k], groups) for k in range(values.shape[1])]
    return np.column_stack(cols)


def ols_within(y, X, groups):
    """OLS coefficients of ``y`` on the columns of ``X`` after removing group means."""
    X = np.asarray(X, float).reshape(len(y), -1)
    X_dm = X - group_means(X, groups)
    y_dm = np.asarray(y, float) - group_means(y, groups)
    return np.linalg.lstsq(X_dm, y_dm, rcond=None)[0]


def permute_within(values, groups, rng):
    """Randomly permute ``values`` among the households of each group."""
    codes = np.unique(groups, return_inverse=True)[1]
    who = np.lexsort((rng.random(len(values)), codes))
    out = np.empty(len(values))
    out[who] = np.asarray(values, float)[np.lexsort((values, codes))]
    return out


def _setup(sample):
    """Village codes, Knowledge, the baseline Y0 and the true beta0 of a sample."""
    village = sample["village"].to_numpy()
    takeup = sample["takeup_survey"].to_numpy(dtype=float)
    knowledge = sample["understanding"].to_numpy(dtype=float)
    beta0 = ols_within(takeup, knowledge, village)[0]
    return village, knowledge, takeup - beta0 * knowledge, beta0


def run_simulation(sample, n_simulations=500, seed=481):
    """Slope estimates when Knowledge is randomly assigned, and the truth.

    Parameters
    ----------
    sample : pandas.DataFrame
        Main sample with ``takeup_survey``, ``understanding`` and ``village``.
    n_simulations : int
        Number of draws (500 keeps the sampling distribution stable and the run
        fast).
    seed : int

    Returns
    -------
    estimates : numpy.ndarray
        One estimate of the slope on ``Knowledge`` per simulation.
    beta0 : float
        The true effect built into the outcomes.
    """
    rng = np.random.default_rng(seed)
    village, knowledge, y0, beta0 = _setup(sample)

    estimates = []
    for _ in range(n_simulations):
        k = permute_within(knowledge, village, rng)
        estimates.append(ols_within(y0 + beta0 * k, k, village)[0])
    return np.array(estimates), beta0


def run_simulation_by_size(sample, sample_sizes, n_simulations=500, seed=481):
    """Repeat the simulation on random subsamples of different sizes.

    In every simulation ``n`` households are drawn without replacement from
    ``sample``, their ``Knowledge`` values are randomly permuted within villages, and
    the slope is re-estimated with village fixed effects. The true effect and each
    household's baseline ``Y0`` are those of the full ``sample``.

    Returns
    -------
    estimates_by_size : dict
        ``{n: array of n_simulations estimates}``.
    beta0 : float
        The true effect built into the outcomes.
    """
    rng = np.random.default_rng(seed)
    village, knowledge, y0, beta0 = _setup(sample)

    estimates_by_size = {}
    for n in sample_sizes:
        estimates = []
        for _ in range(n_simulations):
            i = rng.choice(len(sample), n, replace=False)
            k = permute_within(knowledge[i], village[i], rng)
            estimates.append(ols_within(y0[i] + beta0 * k, k, village[i])[0])
        estimates_by_size[n] = np.array(estimates)
    return estimates_by_size, beta0
