"""Tests for the expected points model."""
import pytest
import numpy as np
from v2.data_loader import load_all_data, load_played, split_walk_forward, load_manual_prices
from v2.expected_points import (
    estimate_eb_params, eb_shrink, compute_eb_estimates,
    grid_search_ema, calibrate, compute_expected_pts,
)


class TestEmpiricalBayes:
    def test_eb_params_estimated_from_data(self):
        played = load_played([2025])
        _, params = compute_eb_estimates(played)
        assert params["mu"] > 0, "Population mean should be positive"
        assert params["tau2"] > 0, "Between-player variance should be positive"
        assert params["sigma2"] > 0, "Within-player variance should be positive"

    def test_shrinkage_full_for_zero_games(self):
        result = eb_shrink(observed=20.0, n=0, mu=7.0, tau2=4.0, sigma2=16.0)
        assert result == 7.0, "Zero games should return pure prior"

    def test_shrinkage_minimal_for_many_games(self):
        result = eb_shrink(observed=12.0, n=100, mu=7.0, tau2=4.0, sigma2=16.0)
        assert result > 11.5, "Many games should barely shrink"

    def test_shrinkage_moderate_for_few_games(self):
        result = eb_shrink(observed=12.0, n=3, mu=7.0, tau2=4.0, sigma2=16.0)
        assert 7.0 < result < 12.0, "Few games should shrink toward prior"


class TestCalibration:
    @pytest.fixture(scope="class")
    def cal(self):
        all_data = load_all_data([2024, 2025])
        return calibrate(all_data)

    def test_no_hand_tuned_params(self, cal):
        """Every parameter should be learned, not default."""
        assert cal["best_ema_alpha"] != 0.3, "EMA alpha should be learned, not default 0.3"
        assert cal["ridge_model"].alpha_ in [0.01, 0.1, 1.0, 10.0, 100.0]

    def test_ema_alpha_reasonable(self, cal):
        alpha = cal["best_ema_alpha"]
        assert 0.05 <= alpha <= 0.95, f"EMA alpha {alpha} outside reasonable range"

    def test_opponent_beta_reasonable(self, cal):
        beta = cal["opponent_beta"]
        assert 0.0 <= beta <= 1.0, f"Opponent beta {beta} outside range"

    def test_ridge_has_coefficients(self, cal):
        coefs = cal["ridge_coefs"]
        assert len(coefs) == 6
        # avg_recent or avg_old should be the dominant feature
        top_feat = max(coefs, key=lambda k: abs(coefs[k]))
        assert "avg" in top_feat, f"Top feature should be an avg_pts, got {top_feat}"

    def test_walk_forward_results_exist(self, cal):
        assert len(cal["window_results"]) >= 1


class TestExpectedPointsMatrix:
    @pytest.fixture(scope="class")
    def ep(self):
        all_data = load_all_data()
        cal = calibrate(all_data)
        train, _ = split_walk_forward(all_data, "Stage 1", 2026)
        train = train[train["P?"] == 1]
        mp = load_manual_prices()
        roster = {
            r["Player"]: {"team": r["Team"], "region": r["Region"], "role": r["Position"]}
            for _, r in mp.iterrows()
        }
        return compute_expected_pts(train, roster, cal)

    def test_correct_player_count(self, ep):
        assert len(ep) == 180

    def test_gw_columns_exist(self, ep):
        for gw in range(1, 7):
            assert f"GW{gw}" in ep.columns

    def test_amer_zero_gw1(self, ep):
        amer = ep[ep["Region"] == "AMER"]
        assert (amer["GW1"] == 0).all(), "AMER players should have 0 pts in GW1"

    def test_non_amer_zero_gw6(self, ep):
        # G2 Esports moved to AMER in 2026 but manual sheet says EMEA
        # Filter them out — schedule is source of truth
        non_amer = ep[(ep["Region"] != "AMER") & (ep["Team"] != "G2 Esports")]
        assert (non_amer["GW6"] == 0).all(), "Non-AMER should have 0 pts in GW6"

    def test_top_player_reasonable(self, ep):
        top = ep.nlargest(1, "BasePts").iloc[0]
        assert 9.0 <= top["BasePts"] <= 16.0, f"Top player BasePts={top['BasePts']} unreasonable"

    def test_population_mean_reasonable(self, ep):
        mean = ep["BasePts"].mean()
        assert 5.0 <= mean <= 10.0, f"Population mean {mean:.2f} unreasonable"

    def test_season_value_positive(self, ep):
        # Every player should have some season value (unless truly new)
        assert (ep["SeasonValue"] >= 0).all()
        assert ep["SeasonValue"].mean() > 20, "Mean season value too low"
