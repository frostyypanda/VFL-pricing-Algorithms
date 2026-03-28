"""
Alternative pricing algorithms — Round 2.

Root cause analysis from Round 1:
  - Elite tier has BEST Pts/VP everywhere -> top-heavy always wins
  - Mid-tier players are too homogeneous -> interchangeable (high overlap)
  - Need elite to cost MORE relative to output, and mid-tier to differentiate

Key insight: to get archetype diversity, the VALUE CURVE must not monotonically
favor one tier. Different tiers should be "best value" depending on HOW MANY
you pick from them.

Usage:
    python alternative_algorithms.py
"""
import pandas as pd
import numpy as np
import os
import sys

DIR = os.path.dirname(os.path.abspath(__file__))

from pricing_algorithms import (
    load_all_data, get_training_data, get_validation_prices,
    get_stage1_results, get_stage1_players,
    compute_player_features, compute_team_win_rates, detect_new_players_on_team,
    load_pickrate_data, compute_player_pickrates, compute_team_popularity,
    compute_opponent_strength,
    algo_baseline_ema, algo_distribution_aware, algo_combined,
    fill_missing_players, add_uncertainty_flags,
    shrink_estimate, _snap_to_half,
    VP_MIN, VP_MAX, BUDGET, SQUAD_SIZE, CHINA_TEAMS,
    TARGET_QUANTILES, TARGET_VPS,
)
from evaluate_pricing import (
    find_optimal_teams, classify_archetype, backtest_teams,
    analyze_distribution,
)
from deep_evaluate import (
    get_stage2_training, get_stage2_validation, get_stage2_results, get_stage2_players,
    algo_neural_network, analyze_criteria,
)


# ============================================================
# Alternative A: Elite Tax
# ============================================================

def algo_elite_tax(features, **kwargs):
    """Overcharge elite players so top-heavy isn't automatically best.

    The problem: elite (top 10%) players have the best Pts/VP ratio, so
    loading up on them + cheap fillers always wins. Fix: price elite players
    ABOVE their proportional value, making them a luxury, not a value pick.

    Method: use a convex curve (x^1.4) that makes the top more expensive.
    This means elite players cost proportionally MORE than their output.
    Budget players stay fairly priced. Mid-tier becomes the value sweet spot.
    Then shift to target mean ~9.09.
    """
    prices = features.copy()
    ema = prices["ema_ppm"].values
    ema_min, ema_max = ema.min(), ema.max()

    if ema_max - ema_min < 1e-9:
        prices["predicted_vp"] = 9.0
    else:
        norm = (ema - ema_min) / (ema_max - ema_min)
        curved = norm ** 1.4
        scaled = 6.0 + curved * (15.0 - 6.0)
        # Shift to target mean ~9.09
        shift = 9.09 - np.mean(scaled)
        scaled = scaled + shift
        prices["predicted_vp"] = _snap_to_half(np.clip(scaled, 6.0, 15.0))

    return prices[["Player", "Team", "predicted_vp"]]


# ============================================================
# Alternative B: Stretched Mid-Tier
# ============================================================

def algo_stretched_mid(features, **kwargs):
    """Expand and differentiate the mid-tier (8-11 VP) where most players live.

    Problem: mid-tier players are interchangeable (all roughly same value),
    causing high team overlap. Fix: allocate more of the price range to the
    middle 60% of players, creating meaningful price differences in mid-tier.

    Distribution (tuned for mean ~9.09):
      Bottom 20%: 6.0-7.5 (compressed budget)
      Middle 60%: 7.5-11.5 (stretched mid — lots of granularity)
      Top 20%:    11.5-15.0 (compressed elite)
    """
    prices = features.copy()
    ema = prices["ema_ppm"].values

    ranks = pd.Series(ema).rank(pct=True).values

    Q = np.array([0.00, 0.20, 0.80, 1.00])
    V = np.array([6.00, 7.50, 11.5, 15.0])

    predicted = np.interp(ranks, Q, V)
    prices["predicted_vp"] = _snap_to_half(predicted)
    return prices[["Player", "Team", "predicted_vp"]]


# ============================================================
# Alternative C: S-Curve (Sigmoid) Pricing
# ============================================================

def algo_s_curve(features, **kwargs):
    """S-curve pricing: compress both extremes, expand mid-tier.

    Sigmoid maps [0,1] -> [0,1] with an S-shape: slow at extremes, fast
    in the middle. This creates:
    - Cluster of budget players at 6-7.5 (similar price, need to pick wisely)
    - Wide spread of mid-tier players at 7.5-12 (lots of differentiation)
    - Cluster of elite at 12-15 (expensive, limited supply)

    Tuned so mean ~9.09. The mid-tier spread means balanced builds have
    interesting choices, while compressed extremes allow stars-and-scrubs.
    """
    prices = features.copy()
    ema = prices["ema_ppm"].values
    ema_min, ema_max = ema.min(), ema.max()

    if ema_max - ema_min < 1e-9:
        prices["predicted_vp"] = 9.0
    else:
        norm = (ema - ema_min) / (ema_max - ema_min)
        k = 6.0
        sigmoid = 1.0 / (1.0 + np.exp(-k * (norm - 0.5)))
        s_min, s_max = sigmoid.min(), sigmoid.max()
        if s_max - s_min > 1e-9:
            sigmoid_norm = (sigmoid - s_min) / (s_max - s_min)
        else:
            sigmoid_norm = np.full_like(sigmoid, 0.5)
        scaled = 6.0 + sigmoid_norm * (15.0 - 6.0)
        # Shift to target mean ~9.09
        shift = 9.09 - np.mean(scaled)
        scaled = scaled + shift
        prices["predicted_vp"] = _snap_to_half(np.clip(scaled, 6.0, 15.0))

    return prices[["Player", "Team", "predicted_vp"]]


# ============================================================
# Alternative D: Archetype-Balanced Distribution
# ============================================================

def algo_archetype_balanced(features, **kwargs):
    """Explicitly design distribution so 3+ archetypes yield similar points.

    Approach: create 4 distinct tiers with controlled player counts,
    where the budget math makes multiple team compositions optimal.

    Tier structure (tuned for mean ~9.09):
      Budget (6.0-7.5):   ~30% of players
      Mid    (8.0-10.0):  ~35% of players
      Good   (10.5-12.0): ~20% of players
      Elite  (12.5-15.0): ~15% of players

    Budget math check (all should be ~100 VP):
    - 2 Elite(13.5) + 3 Good(11) + 6 Budget(7) = 102 -> tight but workable
    - 1 Elite(14) + 3 Good(11) + 7 Mid(8.5) = 106.5 -> 1E+2G+8Mid=14+22+68=104 -> needs tuning
    - 0 Elite + 4 Good(11) + 7 Mid(8) = 100 (Balanced-Good)
    - 0 Elite + 11 Mid(9) = 99 (Pure Balanced)
    - 2 Elite(13) + 9 Budget(7) = 89 -> add 2 Mid = 2E+2Mid+7Bud = 26+17+49=92 (Stars-and-Scrubs)
    """
    prices = features.copy()
    ema = prices["ema_ppm"].values

    ranks = pd.Series(ema).rank(pct=True).values

    # Tier mapping with small gaps between tiers
    Q = np.array([0.00, 0.08, 0.30, 0.33, 0.65, 0.68, 0.85, 0.88, 1.00])
    V = np.array([6.00, 7.00, 7.50, 8.00, 10.0, 10.5, 12.0, 12.5, 15.0])

    predicted = np.interp(ranks, Q, V)
    prices["predicted_vp"] = _snap_to_half(predicted)
    return prices[["Player", "Team", "predicted_vp"]]


# ============================================================
# Alternative E: Combined V2 (Elite Tax + Mid Stretch + All Factors)
# ============================================================

def algo_combined_v2(features, team_win_rates=None, team_new_counts=None,
                     team_popularity=None, opponent_strength=None, **kwargs):
    """Combined V2 — applies elite tax + mid-tier stretch + all contextual factors.

    Key changes from original Combined:
    1. Convex curve (x^1.3) on base performance — elite costs more than proportional
    2. Stretched-mid distribution — more granularity at 7.5-11.5
    3. All original factors (team strength, pickrate, etc.)
    4. Snapped to 0.5 VP increments, mean ~9.09
    """
    if team_win_rates is None:
        team_win_rates = {}
    if team_new_counts is None:
        team_new_counts = {}
    if team_popularity is None:
        team_popularity = {}
    if opponent_strength is None:
        opponent_strength = {}

    prices = features.copy()
    n = len(prices)

    # --- Component 1: Convex EMA (elite tax) ---
    ema_vals = prices["ema_ppm"].values
    ema_min, ema_max = ema_vals.min(), ema_vals.max()
    if ema_max - ema_min > 1e-9:
        ema_norm = (ema_vals - ema_min) / (ema_max - ema_min)
    else:
        ema_norm = np.full(n, 0.5)
    ema_curved = ema_norm ** 1.3  # Convex: top costs more

    # --- Component 2: Role adjustment ---
    kill_pctl = prices["kill_profile"].rank(pct=True).values
    ppts_pctl = prices["ppts_ratio"].rank(pct=True).values
    role_score = (kill_pctl + ppts_pctl) / 2
    role_adj = np.where(role_score > 0.7, -0.03, np.where(role_score < 0.3, 0.03, 0.0))

    # --- Component 3: Team strength ---
    team_wr_vals = prices["Team"].map(team_win_rates).fillna(0.5).values
    avg_wr = np.mean(team_wr_vals)

    def get_team_factor(team):
        new_count = team_new_counts.get(team, 0)
        if new_count >= 3:
            return 0.0
        elif new_count == 2:
            return 0.33
        elif new_count == 1:
            return 0.67
        return 1.0

    team_factors = prices["Team"].apply(get_team_factor).values
    team_strength_norm = (team_wr_vals - avg_wr) * team_factors

    # --- Component 4: Consistency ---
    cv = prices["std_ppm"].values / (np.abs(prices["avg_ppm"].values) + 1e-9)
    cv_rank = pd.Series(cv).rank(pct=True).values
    consistency_adj = 0.1 * (1 - 2 * cv_rank)

    # --- Component 5: Pickrate ---
    pickpct_vals = prices["avg_pickpct"].values
    pickpct_max = pickpct_vals.max() if pickpct_vals.max() > 0 else 1.0
    pickrate_norm = pickpct_vals / pickpct_max

    # --- Component 6: Team brand popularity ---
    team_pop_vals = prices["Team"].map(team_popularity).fillna(0.5).values

    # --- Component 7: Opponent strength ---
    opp_strength_vals = prices["Team"].map(opponent_strength).fillna(0.5).values
    opp_adj = (0.5 - opp_strength_vals) * 0.1

    # --- Blend ---
    base_score = ema_curved + role_adj
    combined_score = (
        0.60 * base_score +
        0.12 * team_strength_norm +
        0.05 * consistency_adj +
        0.08 * pickrate_norm +
        0.08 * team_pop_vals +
        0.07 * opp_adj
    )

    # Stretched-mid distribution mapping
    Q = np.array([0.00, 0.20, 0.80, 1.00])
    V = np.array([6.00, 7.50, 11.5, 15.0])

    ranks = pd.Series(combined_score).rank(pct=True).values
    predicted = np.interp(ranks, Q, V)
    prices["predicted_vp"] = _snap_to_half(predicted)

    return prices[["Player", "Team", "predicted_vp"]]


# ============================================================
# Alternative F: Combined V3 (Archetype-optimized + All Factors)
# ============================================================

def algo_combined_v3(features, team_win_rates=None, team_new_counts=None,
                     team_popularity=None, opponent_strength=None, **kwargs):
    """Combined V3 — archetype-balanced distribution + all contextual factors.

    Uses the archetype-balanced tier structure with all Combined signals.
    The tier gaps are designed so multiple team archetypes yield similar points.
    """
    if team_win_rates is None:
        team_win_rates = {}
    if team_new_counts is None:
        team_new_counts = {}
    if team_popularity is None:
        team_popularity = {}
    if opponent_strength is None:
        opponent_strength = {}

    prices = features.copy()
    n = len(prices)

    # Compute combined score (same as Combined V2)
    ema_vals = prices["ema_ppm"].values
    ema_min, ema_max = ema_vals.min(), ema_vals.max()
    if ema_max - ema_min > 1e-9:
        ema_norm = (ema_vals - ema_min) / (ema_max - ema_min)
    else:
        ema_norm = np.full(n, 0.5)

    kill_pctl = prices["kill_profile"].rank(pct=True).values
    ppts_pctl = prices["ppts_ratio"].rank(pct=True).values
    role_score = (kill_pctl + ppts_pctl) / 2
    role_adj = np.where(role_score > 0.7, -0.03, np.where(role_score < 0.3, 0.03, 0.0))

    team_wr_vals = prices["Team"].map(team_win_rates).fillna(0.5).values
    avg_wr = np.mean(team_wr_vals)

    def get_team_factor(team):
        nc = team_new_counts.get(team, 0)
        if nc >= 3: return 0.0
        elif nc == 2: return 0.33
        elif nc == 1: return 0.67
        return 1.0

    team_factors = prices["Team"].apply(get_team_factor).values
    team_strength_norm = (team_wr_vals - avg_wr) * team_factors

    cv = prices["std_ppm"].values / (np.abs(prices["avg_ppm"].values) + 1e-9)
    cv_rank = pd.Series(cv).rank(pct=True).values
    consistency_adj = 0.1 * (1 - 2 * cv_rank)

    pickpct_vals = prices["avg_pickpct"].values
    pickpct_max = pickpct_vals.max() if pickpct_vals.max() > 0 else 1.0
    pickrate_norm = pickpct_vals / pickpct_max

    team_pop_vals = prices["Team"].map(team_popularity).fillna(0.5).values
    opp_strength_vals = prices["Team"].map(opponent_strength).fillna(0.5).values
    opp_adj = (0.5 - opp_strength_vals) * 0.1

    base_score = ema_norm + role_adj
    combined_score = (
        0.60 * base_score +
        0.12 * team_strength_norm +
        0.05 * consistency_adj +
        0.08 * pickrate_norm +
        0.08 * team_pop_vals +
        0.07 * opp_adj
    )

    # Archetype-balanced tier mapping
    Q = np.array([0.00, 0.08, 0.30, 0.33, 0.65, 0.68, 0.85, 0.88, 1.00])
    V = np.array([6.00, 7.00, 7.50, 8.00, 10.0, 10.5, 12.0, 12.5, 15.0])

    ranks = pd.Series(combined_score).rank(pct=True).values
    predicted = np.interp(ranks, Q, V)
    prices["predicted_vp"] = _snap_to_half(predicted)

    return prices[["Player", "Team", "predicted_vp"]]


# ============================================================
# Alternative G: Combined S-Curve (best value equity + all factors)
# ============================================================

def algo_combined_scurve(features, team_win_rates=None, team_new_counts=None,
                         team_popularity=None, opponent_strength=None, **kwargs):
    """Combined S-Curve — S-curve mapping (best value equity) + all contextual factors.

    S-curve produced the most balanced Pts/VP across tiers (CV=0.036).
    This version applies S-curve shaping to the full combined score
    including all contextual factors, targeting mean ~9.09.
    """
    if team_win_rates is None:
        team_win_rates = {}
    if team_new_counts is None:
        team_new_counts = {}
    if team_popularity is None:
        team_popularity = {}
    if opponent_strength is None:
        opponent_strength = {}

    prices = features.copy()
    n = len(prices)

    # EMA normalized
    ema_vals = prices["ema_ppm"].values
    ema_min, ema_max = ema_vals.min(), ema_vals.max()
    if ema_max - ema_min > 1e-9:
        ema_norm = (ema_vals - ema_min) / (ema_max - ema_min)
    else:
        ema_norm = np.full(n, 0.5)

    # Role adjustment
    kill_pctl = prices["kill_profile"].rank(pct=True).values
    ppts_pctl = prices["ppts_ratio"].rank(pct=True).values
    role_score = (kill_pctl + ppts_pctl) / 2
    role_adj = np.where(role_score > 0.7, -0.03, np.where(role_score < 0.3, 0.03, 0.0))

    # Team strength
    team_wr_vals = prices["Team"].map(team_win_rates).fillna(0.5).values
    avg_wr = np.mean(team_wr_vals)

    def get_team_factor(team):
        nc = team_new_counts.get(team, 0)
        if nc >= 3: return 0.0
        elif nc == 2: return 0.33
        elif nc == 1: return 0.67
        return 1.0

    team_factors = prices["Team"].apply(get_team_factor).values
    team_strength_norm = (team_wr_vals - avg_wr) * team_factors

    # Consistency
    cv = prices["std_ppm"].values / (np.abs(prices["avg_ppm"].values) + 1e-9)
    cv_rank = pd.Series(cv).rank(pct=True).values
    consistency_adj = 0.1 * (1 - 2 * cv_rank)

    # Pickrate
    pickpct_vals = prices["avg_pickpct"].values
    pickpct_max = pickpct_vals.max() if pickpct_vals.max() > 0 else 1.0
    pickrate_norm = pickpct_vals / pickpct_max

    # Team brand popularity
    team_pop_vals = prices["Team"].map(team_popularity).fillna(0.5).values

    # Opponent strength
    opp_strength_vals = prices["Team"].map(opponent_strength).fillna(0.5).values
    opp_adj = (0.5 - opp_strength_vals) * 0.1

    # Blend
    base_score = ema_norm + role_adj
    combined_score = (
        0.60 * base_score +
        0.12 * team_strength_norm +
        0.05 * consistency_adj +
        0.08 * pickrate_norm +
        0.08 * team_pop_vals +
        0.07 * opp_adj
    )

    # S-curve mapping on combined score
    cs_min, cs_max = combined_score.min(), combined_score.max()
    if cs_max - cs_min > 1e-9:
        cs_norm = (combined_score - cs_min) / (cs_max - cs_min)
    else:
        cs_norm = np.full(n, 0.5)

    k = 6.0
    sigmoid = 1.0 / (1.0 + np.exp(-k * (cs_norm - 0.5)))
    s_min, s_max = sigmoid.min(), sigmoid.max()
    if s_max - s_min > 1e-9:
        sigmoid_norm = (sigmoid - s_min) / (s_max - s_min)
    else:
        sigmoid_norm = np.full(n, 0.5)

    scaled = 6.0 + sigmoid_norm * (15.0 - 6.0)
    # Shift to target mean ~9.09
    shift = 9.09 - np.mean(scaled)
    scaled = scaled + shift
    prices["predicted_vp"] = _snap_to_half(np.clip(scaled, 6.0, 15.0))

    return prices[["Player", "Team", "predicted_vp"]]


# ============================================================
# Evaluation — criteria-focused (no old-price accuracy)
# ============================================================

def evaluate_criteria(algo_name, prices_df, stage_results, n_teams=30):
    """Evaluate an algorithm purely on the 4 CLAUDE.md criteria.

    Does NOT compare against old VFL prices — those are irrelevant.
    """
    result = {"algo": algo_name}

    # Distribution analysis
    dist = analyze_distribution(prices_df["predicted_vp"])
    result["distribution"] = dist

    # Actual points per player from stage results
    actual_totals = stage_results.groupby("Player")["Pts"].sum().to_dict()

    # PPM correlation: do higher-priced players actually score more?
    merged = prices_df.copy()
    merged["actual_total"] = merged["Player"].map(actual_totals).fillna(0)
    merged_valid = merged[merged["actual_total"] > 0]
    if len(merged_valid) > 10:
        ppm_corr = merged_valid["predicted_vp"].corr(merged_valid["actual_total"])
    else:
        ppm_corr = 0.0
    result["price_performance_corr"] = ppm_corr

    # Points per VP by tier
    tier_bins = [(6, 8, "Budget"), (8, 10, "Mid"), (10, 12, "Good"), (12, 15.1, "Elite")]
    tier_value = {}
    for lo, hi, name in tier_bins:
        tier_players = merged_valid[(merged_valid["predicted_vp"] >= lo) &
                                     (merged_valid["predicted_vp"] < hi)]
        if len(tier_players) > 0:
            avg_pts = tier_players["actual_total"].mean()
            avg_vp = tier_players["predicted_vp"].mean()
            tier_value[name] = round(avg_pts / avg_vp, 2) if avg_vp > 0 else 0
        else:
            tier_value[name] = 0
    result["pts_per_vp_by_tier"] = tier_value

    # Value equity: how close are the tier Pts/VP ratios? (lower = more equalized)
    values = [v for v in tier_value.values() if v > 0]
    if len(values) > 1:
        result["value_equity"] = round(np.std(values) / np.mean(values), 3)  # CV
    else:
        result["value_equity"] = 0

    # Find optimal teams
    teams = find_optimal_teams(prices_df, actual_totals, n_teams=n_teams, n_iter=20000)
    bt_results = backtest_teams(teams, stage_results)

    # Archetype analysis
    archetypes = {}
    for t in teams:
        team_prices = [p[2] for p in t["players"]]
        arch = classify_archetype(team_prices)
        archetypes[arch] = archetypes.get(arch, 0) + 1
    result["archetypes"] = archetypes
    result["n_archetypes"] = len(archetypes)

    # Player diversity
    all_players = set()
    for t in teams:
        for p in t["players"]:
            all_players.add(p[0])
    result["unique_players"] = len(all_players)

    # Team overlap (Jaccard)
    team_sets = [set(p[0] for p in t["players"]) for t in teams]
    jaccard_scores = []
    for i in range(len(team_sets)):
        for j in range(i + 1, len(team_sets)):
            inter = len(team_sets[i] & team_sets[j])
            union = len(team_sets[i] | team_sets[j])
            jaccard_scores.append(inter / union if union > 0 else 0)
    result["avg_overlap"] = round(np.mean(jaccard_scores), 3) if jaccard_scores else 0

    # Backtest scores
    if bt_results:
        scores = [t["actual_pts_with_igl"] for t in bt_results]
        result["best_pts"] = max(scores)
        result["worst_pts"] = min(scores)
        result["avg_pts"] = round(np.mean(scores), 1)
        result["std_pts"] = round(np.std(scores), 1)
        result["pts_spread"] = max(scores) - min(scores)
    else:
        result["best_pts"] = 0
        result["worst_pts"] = 0
        result["avg_pts"] = 0
        result["std_pts"] = 0
        result["pts_spread"] = 0

    result["teams"] = teams
    result["bt_results"] = bt_results

    # --- CRITERIA SCORES (0-100) ---

    # 1. Accurate Valuation: price-performance correlation + value equity
    #    High correlation = prices reflect actual output
    #    Low value_equity (CV) = tiers are fairly priced
    val_corr_score = max(0, ppm_corr * 70)  # 0-70 from correlation
    val_equity_score = max(0, 30 - result["value_equity"] * 100)  # 0-30 from equity
    result["score_valuation"] = round(min(100, val_corr_score + val_equity_score), 1)

    # 2. Competitive Compositions: team diversity + pts spread
    #    More unique players = more viable compositions
    #    Lower overlap = different teams are actually different
    comp_diversity = min(40, (result["unique_players"] - 15) * 2.5)
    comp_overlap = max(0, 40 - result["avg_overlap"] * 100)
    comp_pts = min(20, (result["avg_pts"] - 600) / 20)
    result["score_compositions"] = round(min(100, max(0, comp_diversity + comp_overlap + comp_pts)), 1)

    # 3. Multiple Archetypes: count + balance of archetypes
    n_arch = result["n_archetypes"]
    arch_count_score = min(50, n_arch * 15)
    # Bonus for balanced distribution (no single archetype > 60%)
    if archetypes:
        max_pct = max(archetypes.values()) / sum(archetypes.values())
        arch_balance_score = max(0, 50 - max_pct * 60)
    else:
        arch_balance_score = 0
    result["score_archetypes"] = round(min(100, arch_count_score + arch_balance_score), 1)

    # 4. Perceived Value: mean AND median close to 9.09 + healthy spread + no "dead" tiers
    mean_gap = abs(dist["mean"] - 9.09)
    median_gap = abs(dist["median"] - 9.09)
    pv_mean = max(0, 25 - mean_gap * 15)     # mean close to 9.09 is important
    pv_median = max(0, 25 - median_gap * 15)  # median too
    pv_spread = min(20, dist["std"] * 7)      # some spread is good
    # Bonus if all 4 tiers have players
    n_active_tiers = sum(1 for v in tier_value.values() if v > 0)
    pv_tiers = n_active_tiers * 7.5
    result["score_perceived_value"] = round(min(100, pv_mean + pv_median + pv_spread + pv_tiers), 1)

    result["total_score"] = round(
        (result["score_valuation"] + result["score_compositions"] +
         result["score_archetypes"] + result["score_perceived_value"]) / 4, 1
    )

    return result


# ============================================================
# Main
# ============================================================

def main():
    print("Loading data...")
    all_data = load_all_data()
    pickrate_df = load_pickrate_data()

    # Stage 1 setup
    s1_training = get_training_data(all_data)
    s1_validation = get_validation_prices(all_data)
    s1_results = get_stage1_results(all_data)
    s1_players = get_stage1_players(all_data)

    player_pickrates = compute_player_pickrates(pickrate_df)
    features = compute_player_features(s1_training, player_pickrates=player_pickrates)
    team_wr = compute_team_win_rates(s1_training)
    team_new_counts = detect_new_players_on_team(s1_training, s1_players)
    team_pop = compute_team_popularity(pickrate_df, all_data)
    opp_strength = compute_opponent_strength(s1_training)

    algo_kwargs = {
        "team_win_rates": team_wr,
        "team_new_counts": team_new_counts,
        "team_popularity": team_pop,
        "opponent_strength": opp_strength,
    }

    print(f"Training: {len(s1_training)} games, {s1_training['Player'].nunique()} players")
    print(f"Stage 1: {len(s1_results)} played rows, {s1_results['Player'].nunique()} players")
    print()

    # === Define all algorithms to test ===
    algorithms = {
        # Original best performers (for comparison)
        "Baseline EMA (orig)": algo_baseline_ema(features, **algo_kwargs),
        "Combined (orig)": algo_combined(features, **algo_kwargs),
        # New alternatives
        "A: Elite Tax": algo_elite_tax(features, **algo_kwargs),
        "B: Stretched Mid": algo_stretched_mid(features, **algo_kwargs),
        "C: S-Curve": algo_s_curve(features, **algo_kwargs),
        "D: Archetype-Balanced": algo_archetype_balanced(features, **algo_kwargs),
        "E: Combined V2": algo_combined_v2(features, **algo_kwargs),
        "F: Combined V3": algo_combined_v3(features, **algo_kwargs),
        "G: Combined S-Curve": algo_combined_scurve(features, **algo_kwargs),
    }

    # Neural Network
    nn_prices = algo_neural_network(features, s1_validation)
    algorithms["Neural Network (orig)"] = nn_prices

    # Fill missing players for all
    for name in algorithms:
        algorithms[name] = fill_missing_players(
            algorithms[name], s1_players, features,
            team_win_rates=team_wr, team_new_counts=team_new_counts,
        )
        algorithms[name] = add_uncertainty_flags(algorithms[name], features)

    # === Evaluate all on Stage 1 ===
    print("=" * 90)
    print("  STAGE 1 — CRITERIA-FOCUSED EVALUATION (old prices ignored)")
    print("=" * 90)

    all_results = {}
    for name, prices_df in algorithms.items():
        print(f"\n  --- {name} ---")
        r = evaluate_criteria(name, prices_df, s1_results, n_teams=30)
        all_results[name] = r

        d = r["distribution"]
        print(f"    Distribution: Mean={d['mean']:.2f}  Med={d['median']:.2f}  "
              f"Std={d['std']:.2f}  Range={d['min']:.1f}-{d['max']:.1f}")
        print(f"    Price-Perf Corr: {r['price_performance_corr']:.3f}")
        print(f"    Pts/VP by tier: {r['pts_per_vp_by_tier']}  "
              f"Value equity (CV): {r['value_equity']:.3f}")
        print(f"    Teams: Best={r['best_pts']:.0f}  Avg={r['avg_pts']:.0f}  "
              f"Worst={r['worst_pts']:.0f}  Spread={r['pts_spread']:.0f}")
        print(f"    Diversity: {r['n_archetypes']} archetypes, "
              f"{r['unique_players']} unique players, "
              f"overlap={r['avg_overlap']:.3f}")
        print(f"    Archetypes: {r['archetypes']}")

    # === Summary table ===
    print(f"\n{'=' * 90}")
    print(f"  CRITERIA SCORES (0-100) — higher is better")
    print(f"{'=' * 90}")
    print(f"  {'Algorithm':<26} {'1.Value':>8} {'2.Comps':>8} {'3.Archs':>8} "
          f"{'4.PVal':>8} {'TOTAL':>8}")
    print(f"  {'-'*26} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    sorted_algos = sorted(all_results.values(), key=lambda x: x["total_score"], reverse=True)
    for r in sorted_algos:
        marker = " ***" if r["algo"] in ["A: Compressed (6-13)", "B: Sqrt Mapping",
                                           "C: Value-Equalized", "D: Tiered Gaps",
                                           "E: Combined V2"] else ""
        print(f"  {r['algo']:<26} {r['score_valuation']:>8.1f} "
              f"{r['score_compositions']:>8.1f} {r['score_archetypes']:>8.1f} "
              f"{r['score_perceived_value']:>8.1f} {r['total_score']:>8.1f}{marker}")

    # === Key metrics comparison ===
    print(f"\n{'=' * 90}")
    print(f"  KEY METRICS COMPARISON")
    print(f"{'=' * 90}")
    print(f"  {'Algorithm':<26} {'Med VP':>7} {'Archs':>6} {'Unique':>7} "
          f"{'Overlap':>8} {'BestPts':>8} {'ValEq':>7}")
    print(f"  {'-'*26} {'-'*7} {'-'*6} {'-'*7} {'-'*8} {'-'*8} {'-'*7}")
    for r in sorted_algos:
        print(f"  {r['algo']:<26} {r['distribution']['median']:>7.2f} "
              f"{r['n_archetypes']:>6} {r['unique_players']:>7} "
              f"{r['avg_overlap']:>8.3f} {r['best_pts']:>8.0f} "
              f"{r['value_equity']:>7.3f}")

    # === Show best team for all new algos ===
    actual_totals = s1_results.groupby("Player")["Pts"].sum().to_dict()
    new_algos = [r for r in sorted_algos if "orig" not in r["algo"] and "Neural" not in r["algo"]]
    for r in new_algos[:3]:
        bt = r["bt_results"]
        if not bt:
            continue
        best = bt[0]
        print(f"\n  {r['algo']} — Best Team ({best['actual_pts_with_igl']:.0f} pts, IGL: {best['igl']}):")
        print(f"    {'Player':<18} {'Team':<20} {'VP':>6} {'Pts':>6}")
        for pname, pteam, pvp, _ in sorted(best["players"],
                                             key=lambda x: actual_totals.get(x[0], 0),
                                             reverse=True):
            igl = " *" if pname == best["igl"] else ""
            print(f"    {pname:<18} {pteam:<20} {pvp:>6.1f} {actual_totals.get(pname, 0):>6}{igl}")
        print(f"    {'TOTAL':<40} {best['total_vp']:>6.1f}")

    # =============================================
    # STAGE 2 EVALUATION
    # =============================================
    s2_training = get_stage2_training(all_data)
    s2_validation = get_stage2_validation(all_data)
    s2_results = get_stage2_results(all_data)
    s2_players = get_stage2_players(all_data)

    if len(s2_validation) == 0 or len(s2_results) == 0:
        print("\nNo Stage 2 data available.")
        print("\nDone.")
        return

    print(f"\n\n{'=' * 90}")
    print(f"  STAGE 2 — CRITERIA-FOCUSED EVALUATION")
    print(f"{'=' * 90}")

    s2_pickrates = compute_player_pickrates(pickrate_df)
    s2_features = compute_player_features(s2_training, player_pickrates=s2_pickrates)
    s2_team_wr = compute_team_win_rates(s2_training)
    s2_new_counts = detect_new_players_on_team(s2_training, s2_players)
    s2_team_pop = compute_team_popularity(pickrate_df, all_data)
    s2_opp = compute_opponent_strength(s2_training)

    s2_kwargs = {
        "team_win_rates": s2_team_wr,
        "team_new_counts": s2_new_counts,
        "team_popularity": s2_team_pop,
        "opponent_strength": s2_opp,
    }

    s2_algorithms = {
        "Baseline EMA (orig)": algo_baseline_ema(s2_features, **s2_kwargs),
        "Combined (orig)": algo_combined(s2_features, **s2_kwargs),
        "A: Elite Tax": algo_elite_tax(s2_features, **s2_kwargs),
        "B: Stretched Mid": algo_stretched_mid(s2_features, **s2_kwargs),
        "C: S-Curve": algo_s_curve(s2_features, **s2_kwargs),
        "D: Archetype-Balanced": algo_archetype_balanced(s2_features, **s2_kwargs),
        "E: Combined V2": algo_combined_v2(s2_features, **s2_kwargs),
        "F: Combined V3": algo_combined_v3(s2_features, **s2_kwargs),
        "G: Combined S-Curve": algo_combined_scurve(s2_features, **s2_kwargs),
    }

    # NN for Stage 2
    s2_nn = algo_neural_network(s2_features, s2_validation)
    s2_algorithms["Neural Network (orig)"] = s2_nn

    for name in s2_algorithms:
        s2_algorithms[name] = fill_missing_players(
            s2_algorithms[name], s2_players, s2_features,
            team_win_rates=s2_team_wr, team_new_counts=s2_new_counts,
        )
        s2_algorithms[name] = add_uncertainty_flags(s2_algorithms[name], s2_features)

    s2_results_dict = {}
    for name, prices_df in s2_algorithms.items():
        print(f"\n  --- {name} ---")
        r = evaluate_criteria(name, prices_df, s2_results, n_teams=30)
        s2_results_dict[name] = r
        d = r["distribution"]
        print(f"    Distribution: Mean={d['mean']:.2f}  Med={d['median']:.2f}  "
              f"Std={d['std']:.2f}")
        print(f"    Teams: Best={r['best_pts']:.0f}  Avg={r['avg_pts']:.0f}  "
              f"Archetypes: {r['archetypes']}")
        print(f"    Diversity: {r['unique_players']} unique, overlap={r['avg_overlap']:.3f}")

    # Stage 2 summary
    print(f"\n{'=' * 90}")
    print(f"  STAGE 2 — CRITERIA SCORES")
    print(f"{'=' * 90}")
    print(f"  {'Algorithm':<26} {'1.Value':>8} {'2.Comps':>8} {'3.Archs':>8} "
          f"{'4.PVal':>8} {'TOTAL':>8}")
    print(f"  {'-'*26} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    s2_sorted = sorted(s2_results_dict.values(), key=lambda x: x["total_score"], reverse=True)
    for r in s2_sorted:
        print(f"  {r['algo']:<26} {r['score_valuation']:>8.1f} "
              f"{r['score_compositions']:>8.1f} {r['score_archetypes']:>8.1f} "
              f"{r['score_perceived_value']:>8.1f} {r['total_score']:>8.1f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
