"""Likelihood-ratio coverage tests; no regulatory traffic-light classification.

Kupiec uses all hits. Christoffersen independence conditions on the first hit
and counts adjacent observations. Combined CC = LR_uc + LR_ind (usual
asymptotic implementation). Small/degenerate samples need special care.
"""
from numbers import Real
import numpy as np
from scipy.special import xlogy
from scipy.stats import chi2


def _probability(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or not 0 < value < 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return float(value)


def _hits(exceptions):
    values = np.asarray(exceptions)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("exceptions must contain at least two binary observations")
    if not np.isin(values, [0, 1]).all():
        raise ValueError("exceptions must contain only binary observations")
    return values.astype(int)


def _ll(zero, one, p):
    # xlogy(0, 0) = 0: the limiting likelihood for empty categories.
    return float(xlogy(zero, 1 - p) + xlogy(one, p))


def _result(lr, df, significance):
    lr = max(0.0, float(lr))
    pvalue = float(chi2.sf(lr, df))
    return {"lr_statistic": lr, "degrees_of_freedom": df, "p_value": pvalue,
            "significance": significance, "reject_null": pvalue < significance,
            "status": "reject" if pvalue < significance else "do_not_reject"}


def coverage_tests(exceptions, expected_exception_rate=0.05, significance=0.05):
    """Return POF, first-order independence and combined coverage diagnostics.

    A constant sequence has no identified transition model: independence and
    conditional-coverage p-values are unavailable, not automatic passes.
    """
    hits = _hits(exceptions)
    p = _probability(expected_exception_rate, "expected_exception_rate")
    significance = _probability(significance, "significance")
    n = len(hits)
    x = int(hits.sum())
    lr_uc = 2 * (_ll(n - x, x, x / n) - _ll(n - x, x, p))
    counts = np.bincount(2 * hits[:-1] + hits[1:], minlength=4)
    n00, n01, n10, n11 = map(int, counts)
    transitions = dict(zip(["n00", "n01", "n10", "n11"], map(int, counts)))
    warnings = []
    if min(n * p, n * (1 - p)) < 5:
        warnings.append("Small expected hit count; chi-square approximation may be unreliable.")
    pof = _result(lr_uc, 1, significance)
    if n00 + n01 == 0 or n10 + n11 == 0 or len(np.unique(hits[1:])) < 2:
        unavailable = {"lr_statistic": None, "p_value": None, "reject_null": None,
                       "significance": significance, "status": "not_identifiable"}
        independence = dict(unavailable, degrees_of_freedom=1)
        conditional = dict(unavailable, degrees_of_freedom=2)
        warnings.append("Transition model is degenerate; independence and combined CC are not reported.")
    else:
        pi = (n01 + n11) / (n - 1)
        pi0 = n01 / (n00 + n01)
        pi1 = n11 / (n10 + n11)
        lr_ind = 2 * (_ll(n00, n01, pi0) + _ll(n10, n11, pi1)
                      - _ll(n00 + n10, n01 + n11, pi))
        independence = _result(lr_ind, 1, significance)
        conditional = _result(lr_uc + max(0, lr_ind), 2, significance)
        if min(counts) < 5:
            warnings.append("Sparse transition cells; interpret asymptotic independence/CC p-values cautiously.")
    return {"observations": n, "exceptions": x, "exception_rate": x / n,
            "expected_exception_rate": p, "transition_counts": transitions,
            "kupiec_pof": pof, "christoffersen_independence": independence,
            "christoffersen_conditional_coverage": conditional, "warnings": warnings}
