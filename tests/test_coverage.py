"""Independent reference values, temporal dependence, and boundary cases."""
import math
import numpy as np
import pandas as pd
import pytest
from src.risk_engine.coverage import coverage_tests
from src.risk_engine.backtest import rolling_historical_var


def test_pof_matches_published_mathworks_example():
    # MathWorks POF documentation: N=1043, x=57, expected rate=.05.
    hits = [0] * (1043 - 57) + [1] * 57
    result = coverage_tests(hits)["kupiec_pof"]
    assert result["lr_statistic"] == pytest.approx(.46147, abs=5e-6)
    assert result["p_value"] == pytest.approx(.49694, abs=5e-6)


def test_same_exception_rate_different_order_changes_independence():
    interspersed = [0] * 20 + [1] * 20 + [0] * 20 + [1] * 20 + [0] * 20
    dispersed = [0, 0, 1, 0, 1] * 20
    a, b = coverage_tests(interspersed, .4), coverage_tests(dispersed, .4)
    assert a["kupiec_pof"] == b["kupiec_pof"]
    assert a["christoffersen_independence"]["lr_statistic"] != pytest.approx(b["christoffersen_independence"]["lr_statistic"])


def test_independence_matches_hand_calculated_likelihood():
    # 0,0,1,0,1,1,0 gives n00=1,n01=2,n10=2,n11=1.
    result = coverage_tests([0, 0, 1, 0, 1, 1, 0], .5)
    lr = 2 * (2 * math.log(1/3) + 4 * math.log(2/3) - 6 * math.log(.5))
    assert result["transition_counts"] == {"n00": 1, "n01": 2, "n10": 2, "n11": 1}
    assert result["christoffersen_independence"]["lr_statistic"] == pytest.approx(lr)
    assert result["christoffersen_independence"]["p_value"] == pytest.approx(math.erfc(math.sqrt(lr/2)))


@pytest.mark.parametrize("value", [0, 1])
def test_constant_hits_have_finite_pof_but_no_independence_pvalue(value):
    r = coverage_tests([value] * 250)
    expected = -2 * 250 * math.log(.05 if value else .95)
    assert r["kupiec_pof"]["lr_statistic"] == pytest.approx(expected)
    assert r["christoffersen_independence"]["p_value"] is None
    assert r["christoffersen_conditional_coverage"]["reject_null"] is None


@pytest.mark.parametrize("invalid", [[], [1], [0, 2], [0, np.nan], [[0, 1]], ["0", "1"]])
def test_invalid_hit_sequences_rejected(invalid):
    with pytest.raises(ValueError):
        coverage_tests(invalid)


@pytest.mark.parametrize("invalid", [0, 1, -.1, np.nan, True])
def test_invalid_probabilities_rejected(invalid):
    with pytest.raises(ValueError):
        coverage_tests([0, 1, 0], invalid)


def test_forecasts_require_unique_chronological_observations():
    for index in [[2, 1, 3], [1, 1, 2]]:
        with pytest.raises(ValueError, match="unique, ascending"):
            rolling_historical_var(pd.Series([.01, .02, -.01], index=index), window=2)
