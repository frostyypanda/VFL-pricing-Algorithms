"""Tests for the pricing algorithm."""
import pytest
import numpy as np
from v2.data_loader import load_all_data, load_manual_prices, load_pickrate_summary, split_walk_forward
from v2.expected_points import calibrate, compute_expected_pts
from v2.pricing import compute_prices, verify_distribution, price_summary
from v2.constants import VP_MIN, VP_MAX, TARGET_MEAN, SQUAD_SIZE


@pytest.fixture(scope="module")
def priced():
    all_data = load_all_data()
    cal = calibrate(all_data)
    train, _ = split_walk_forward(all_data, "Stage 1", 2026)
    train = train[train["P?"] == 1]
    mp = load_manual_prices()
    roster = {
        r["Player"]: {"team": r["Team"], "region": r["Region"], "role": r["Position"]}
        for _, r in mp.iterrows()
    }
    ep = compute_expected_pts(train, roster, cal)
    pr = load_pickrate_summary()
    ep = compute_prices(ep, pr)
    ep["ManualVP"] = ep["Player"].map(dict(zip(mp["Player"], mp["Stage1_Price"])))
    return ep


def test_all_distribution_checks_pass(priced):
    checks = verify_distribution(priced)
    for name, passed, detail in checks:
        assert passed, f"Distribution check failed: {name} = {detail}"


def test_price_range(priced):
    assert priced["SuggestedVP"].min() >= VP_MIN
    assert priced["SuggestedVP"].max() <= VP_MAX


def test_half_vp_increments(priced):
    remainders = priced["SuggestedVP"] % 0.5
    assert (remainders == 0).all(), "All prices must be 0.5 increments"


def test_mean_near_target(priced):
    mean = priced["SuggestedVP"].mean()
    assert abs(mean - TARGET_MEAN) < 0.3, f"Mean {mean:.3f} too far from target {TARGET_MEAN}"


def test_correlation_with_manual(priced):
    corr = priced["SuggestedVP"].corr(priced["ManualVP"])
    assert corr > 0.65, f"Correlation {corr:.3f} below 0.65 threshold"


def test_mae_with_manual(priced):
    mae = (priced["SuggestedVP"] - priced["ManualVP"]).abs().mean()
    assert mae < 2.0, f"MAE {mae:.2f} too high"


def test_no_dead_zone_in_middle(priced):
    """Ensure no 2VP gap without any players in the 7-12 range."""
    prices = sorted(priced["SuggestedVP"].unique())
    middle = [p for p in prices if 7.0 <= p <= 12.0]
    for i in range(len(middle) - 1):
        gap = middle[i + 1] - middle[i]
        assert gap <= 1.5, f"Dead zone: gap of {gap} between {middle[i]} and {middle[i+1]}"


def test_stars_expensive_enough(priced):
    """Top 11 players must cost more than budget (forces tradeoffs)."""
    top11 = priced["SuggestedVP"].nlargest(SQUAD_SIZE).sum()
    assert top11 > 120, f"Top 11 cost only {top11} — stars not expensive enough"
