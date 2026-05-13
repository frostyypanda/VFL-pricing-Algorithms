"""Unit tests for v3 role-mean EB shrinkage with B-floor."""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v3.eb_role import eb_shrink_role, compute_v3_eb_estimates, MIN_B


def test_b_floor_keeps_low_n_player_above_role_mean_when_observed_is_above():
    # Given: a low-n player with observed=12 against role_mu=7, large sigma2/tau2 ratio
    # When: EB shrinkage applied with B-floor
    role_mu = 7.0
    observed = 12.0
    n = 3
    tau2 = 1.0
    sigma2 = 30.0  # raw B = 1/(1+10) = 0.09, floor lifts to 0.30
    # Then: shrunk value should be 0.30*12 + 0.70*7 = 8.5, not collapse to ~7.45
    result = eb_shrink_role(observed, n, role_mu, tau2, sigma2)
    assert result == 0.30 * 12.0 + 0.70 * 7.0


def test_n_zero_returns_role_mean():
    # Given: a brand-new player with no prior data
    # When: EB shrink is called with n=0
    # Then: result is exactly the role mean
    assert eb_shrink_role(observed=10.0, n=0, role_mu=8.0, tau2=1.0, sigma2=5.0) == 8.0


def test_b_floor_is_at_least_min_b():
    # Given: any small n where raw B would be below MIN_B
    # When: EB shrink is called
    # Then: effective B is exactly MIN_B
    result = eb_shrink_role(observed=10.0, n=1, role_mu=5.0,
                            tau2=0.1, sigma2=100.0)
    expected = MIN_B * 10.0 + (1 - MIN_B) * 5.0
    assert abs(result - expected) < 1e-9


def test_v3_eb_estimates_differ_by_role():
    # Given: two players with identical Pts, one a duelist, one a sentinel,
    # where duelists have a different role mean than sentinels
    # When: compute_v3_eb_estimates runs
    # Then: the two players receive different shrunk values
    played = pd.DataFrame([
        {"Player": "duelist1", "Pts": 12.0},
        {"Player": "duelist2", "Pts": 11.0},
        {"Player": "duelist3", "Pts": 13.0},
        {"Player": "duelist4", "Pts": 10.0},
        {"Player": "duelist5", "Pts": 11.5},
        {"Player": "sentinel1", "Pts": 6.0},
        {"Player": "sentinel2", "Pts": 7.0},
        {"Player": "sentinel3", "Pts": 5.5},
        {"Player": "sentinel4", "Pts": 6.5},
        {"Player": "sentinel5", "Pts": 7.5},
        # The two players under test:
        {"Player": "test_duelist", "Pts": 9.0},
        {"Player": "test_sentinel", "Pts": 9.0},
    ])
    roster = {
        **{f"duelist{i}": {"team": "T", "region": "AMER", "role": "D"}
           for i in range(1, 6)},
        **{f"sentinel{i}": {"team": "T", "region": "AMER", "role": "S"}
           for i in range(1, 6)},
        "test_duelist": {"team": "T", "region": "AMER", "role": "D"},
        "test_sentinel": {"team": "T", "region": "AMER", "role": "S"},
    }
    estimates, debug = compute_v3_eb_estimates(played, roster)
    assert estimates["test_duelist"] != estimates["test_sentinel"]
    assert debug["role_means"]["D"] > debug["role_means"]["S"]
