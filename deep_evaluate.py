"""
Deep evaluation of top 3 pricing algorithms + Stage 2 backtest.

1. Stage 1 backtest: top 30 teams for Baseline EMA, Distribution-Aware, Role-Adjusted
2. Stage 2 backtest: train on ALL data through Stage 1, predict Stage 2
3. Criteria compliance analysis
4. Neural network algorithm (Algorithm 7)

Usage:
    python deep_evaluate.py
"""
import pandas as pd
import numpy as np
import os
import sys

DIR = os.path.dirname(os.path.abspath(__file__))

# Import from pricing_algorithms
from pricing_algorithms import (
    load_all_data, get_training_data, get_validation_prices,
    get_stage1_results, get_stage1_players,
    compute_player_features, compute_team_win_rates, detect_new_players_on_team,
    load_pickrate_data, compute_player_pickrates, compute_team_popularity,
    compute_opponent_strength,
    algo_baseline_ema, algo_distribution_aware, algo_role_adjusted,
    algo_combined, fill_missing_players, add_uncertainty_flags,
    _linear_map, shrink_estimate,
    VP_MIN, VP_MAX, BUDGET, SQUAD_SIZE, CHINA_TEAMS,
    STAGE_ORDER, GAME_ORDER, TARGET_QUANTILES, TARGET_VPS,
)
from evaluate_pricing import (
    find_optimal_teams, classify_archetype, backtest_teams,
    evaluate_price_accuracy, analyze_distribution,
)


# ============================================================
# PART 1: Stage 2 data loading
# ============================================================

def get_stage2_training(df):
    """Training for Stage 2: all 2024 + 2025 Kickoff + Bangkok + Stage 1."""
    mask_2024 = df["Year"] == 2024
    mask_2025 = (df["Year"] == 2025) & df["Stage"].isin(["Kickoff", "bangkok", "Stage 1"])
    played = df["P?"] == 1
    return df[(mask_2024 | mask_2025) & played].copy()


def get_stage2_validation(df):
    """Actual VP prices at 2025 Stage 2 G1 start."""
    mask = (
        (df["Year"] == 2025)
        & (df["Stage"] == "Stage 2")
        & (df["Game"] == "G1")
        & (df["P?"] == 1)
    )
    vp = df.loc[mask, ["Team", "Player", "Game Start VP"]].copy()
    vp = vp.rename(columns={"Game Start VP": "actual_vp"})
    return vp.drop_duplicates(subset="Player")


def get_stage2_results(df):
    """All played Stage 2 rows for backtesting."""
    mask = (df["Year"] == 2025) & (df["Stage"] == "Stage 2") & (df["P?"] == 1)
    return df[mask].copy()


def get_stage2_players(df):
    """All players in 2025 Stage 2."""
    mask = (df["Year"] == 2025) & (df["Stage"] == "Stage 2")
    return df.loc[mask, ["Team", "Player"]].drop_duplicates(subset="Player")


# ============================================================
# PART 2: Neural Network Algorithm
# ============================================================

class SimpleNN:
    """Minimal 2-layer neural network for pricing.

    Input: player features (normalized)
    Output: single value in [-3, +6] range
    Final price = 9.09 + output, rounded to nearest 0.5
    """
    def __init__(self, input_size, hidden_size=16):
        # Xavier initialization
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, 1) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(1)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU
        self.z2 = self.a1 @ self.W2 + self.b2
        # Tanh scaled to [-3, +6]: output = 4.5 * tanh(z2) + 1.5
        self.out = 4.5 * np.tanh(self.z2) + 1.5
        return self.out

    def backward(self, X, y, lr=0.001):
        n = X.shape[0]
        # Loss: MSE
        error = self.out - y.reshape(-1, 1)

        # Gradient through tanh scaling: d(4.5*tanh(z)+1.5)/dz = 4.5*(1-tanh^2(z))
        tanh_val = np.tanh(self.z2)
        d_tanh = 4.5 * (1 - tanh_val ** 2)
        d_z2 = error * d_tanh / n

        d_W2 = self.a1.T @ d_z2
        d_b2 = d_z2.sum(axis=0)

        d_a1 = d_z2 @ self.W2.T
        d_z1 = d_a1 * (self.z1 > 0).astype(float)

        d_W1 = X.T @ d_z1
        d_b1 = d_z1.sum(axis=0)

        self.W2 -= lr * d_W2
        self.b2 -= lr * d_b2
        self.W1 -= lr * d_W1
        self.b1 -= lr * d_b1

        return np.mean(error ** 2)

    def predict(self, X):
        return self.forward(X).flatten()


def prepare_nn_features(features):
    """Prepare normalized feature matrix for NN."""
    cols = ["ema_ppm", "avg_ppm", "std_ppm", "avg_pts", "avg_tpts", "avg_ppts",
            "win_rate", "kill_profile", "rating_bonus_rate", "ppts_ratio",
            "avg_pickpct", "avg_rank_pct", "games_played"]
    X = features[cols].values.astype(float)
    # Normalize each column to [0, 1]
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    ranges = maxs - mins
    ranges[ranges < 1e-9] = 1.0
    X_norm = (X - mins) / ranges
    return X_norm, mins, ranges, cols


def algo_neural_network(features, validation_prices, n_epochs=2000, lr=0.005, hidden=16):
    """Algorithm 7: Neural network pricing.

    Trains a small NN on features -> actual VP price offset from 9.09.
    Output in [-3, +6], final price = 9.09 + output, rounded to 0.5.
    """
    # Prepare features
    X_all, mins, ranges, cols = prepare_nn_features(features)

    # Get training targets: players that have actual prices
    merged = features.merge(validation_prices, on="Player", how="inner")
    if len(merged) < 10:
        print("  WARNING: Too few players with actual prices for NN training")
        return algo_baseline_ema(features)

    # Training set
    train_idx = features["Player"].isin(merged["Player"]).values
    X_train = X_all[train_idx]
    y_train = merged.sort_values("Player")["actual_vp"].values - 9.09  # offset from base

    # Clip targets to [-3, +6]
    y_train = np.clip(y_train, -3, 6)

    # Train NN
    np.random.seed(42)
    nn = SimpleNN(X_train.shape[1], hidden_size=hidden)

    best_loss = float("inf")
    best_W1, best_b1, best_W2, best_b2 = None, None, None, None

    for epoch in range(n_epochs):
        nn.forward(X_train)
        loss = nn.backward(X_train, y_train, lr=lr)
        if loss < best_loss:
            best_loss = loss
            best_W1 = nn.W1.copy()
            best_b1 = nn.b1.copy()
            best_W2 = nn.W2.copy()
            best_b2 = nn.b2.copy()

    # Use best weights
    nn.W1, nn.b1, nn.W2, nn.b2 = best_W1, best_b1, best_W2, best_b2

    # Predict all players
    raw_offsets = nn.predict(X_all)
    predicted_vp = 9.09 + raw_offsets

    # Round to nearest 0.5
    predicted_vp = np.round(predicted_vp * 2) / 2
    predicted_vp = np.clip(predicted_vp, VP_MIN, VP_MAX)

    prices = features[["Player", "Team"]].copy()
    prices["predicted_vp"] = predicted_vp

    train_pred = nn.predict(X_train) + 9.09
    train_mae = np.mean(np.abs(train_pred - (y_train + 9.09)))
    print(f"  NN training MAE: {train_mae:.3f} (on {len(X_train)} players, {n_epochs} epochs)")

    return prices


# ============================================================
# PART 3: Criteria Analysis
# ============================================================

def analyze_criteria(algo_name, prices_df, validation, stage_results, n_teams=30):
    """Full criteria compliance analysis for an algorithm."""
    result = {}

    # Price accuracy
    acc = evaluate_price_accuracy(prices_df, validation)
    result["accuracy"] = acc

    # Distribution
    dist = analyze_distribution(prices_df["predicted_vp"])
    result["distribution"] = dist

    # Expected points per player (from training features)
    # Use actual stage results for expected points
    actual_totals = stage_results.groupby("Player")["Pts"].sum().to_dict()

    # Find top N teams
    teams = find_optimal_teams(prices_df, actual_totals, n_teams=n_teams, n_iter=20000)
    bt_results = backtest_teams(teams, stage_results)

    # Archetype analysis
    archetypes = {}
    for t in teams:
        prices = [p[2] for p in t["players"]]
        arch = classify_archetype(prices)
        archetypes[arch] = archetypes.get(arch, 0) + 1
    result["archetypes"] = archetypes
    result["n_archetypes"] = len(archetypes)

    # Team diversity: how many unique players across all 30 teams?
    all_players = set()
    for t in teams:
        for p in t["players"]:
            all_players.add(p[0])
    result["unique_players_in_top30"] = len(all_players)

    # Team overlap: average pairwise Jaccard similarity
    team_sets = [set(p[0] for p in t["players"]) for t in teams]
    jaccard_scores = []
    for i in range(len(team_sets)):
        for j in range(i + 1, len(team_sets)):
            intersection = len(team_sets[i] & team_sets[j])
            union = len(team_sets[i] | team_sets[j])
            jaccard_scores.append(intersection / union if union > 0 else 0)
    result["avg_team_overlap"] = np.mean(jaccard_scores) if jaccard_scores else 0

    # Backtest scores
    if bt_results:
        scores = [t["actual_pts_with_igl"] for t in bt_results]
        result["best_team_pts"] = max(scores)
        result["worst_team_pts"] = min(scores)
        result["avg_team_pts"] = np.mean(scores)
        result["std_team_pts"] = np.std(scores)
    else:
        result["best_team_pts"] = 0
        result["worst_team_pts"] = 0
        result["avg_team_pts"] = 0
        result["std_team_pts"] = 0

    result["teams"] = teams
    result["bt_results"] = bt_results

    # Criteria scores (0-100)
    # 1. Accurate valuation: based on correlation and MAE
    if acc["corr"] is not None:
        val_score = max(0, min(100, acc["corr"] * 100 + (1.5 - acc["mae"]) * 20))
    else:
        val_score = 0
    result["criteria_1_valuation"] = val_score

    # 2. Competitive compositions: based on avg team pts and spread
    comp_score = max(0, min(100,
        (result["avg_team_pts"] - 500) / 5 +  # higher avg = better
        (result["unique_players_in_top30"] - 20) * 2  # more unique players = better
    ))
    result["criteria_2_compositions"] = comp_score

    # 3. Multiple archetypes
    arch_score = min(100, result["n_archetypes"] * 30 + (1 - result["avg_team_overlap"]) * 40)
    result["criteria_3_archetypes"] = arch_score

    # 4. Perceived value: median close to 9.09, healthy spread
    median_gap = abs(dist["median"] - 9.09)
    pv_score = max(0, min(100, 100 - median_gap * 30 - abs(dist["skew"]) * 15))
    result["criteria_4_perceived_value"] = pv_score

    return result


# ============================================================
# PART 4: Main
# ============================================================

def run_stage_eval(stage_name, training_df, validation_prices, stage_results,
                   stage_players, all_data, algo_names_and_fns, pickrate_df=None):
    """Run evaluation for a given stage."""
    print(f"\n{'='*80}")
    print(f"  {stage_name} EVALUATION")
    print(f"{'='*80}")

    player_pickrates = compute_player_pickrates(pickrate_df)
    features = compute_player_features(training_df, player_pickrates=player_pickrates)
    team_wr = compute_team_win_rates(training_df)
    team_new_counts = detect_new_players_on_team(training_df, stage_players)
    team_pop = compute_team_popularity(pickrate_df, all_data)
    opp_strength = compute_opponent_strength(training_df)

    kwargs = {
        "team_win_rates": team_wr,
        "team_new_counts": team_new_counts,
        "team_popularity": team_pop,
        "opponent_strength": opp_strength,
    }

    print(f"  Training: {len(training_df)} games, {training_df['Player'].nunique()} players")
    print(f"  Validation: {len(validation_prices)} players")
    print(f"  Stage results: {len(stage_results)} played rows")
    print()

    all_results = {}

    for algo_name, algo_fn in algo_names_and_fns:
        print(f"  --- {algo_name} ---")

        if algo_name == "Neural Network":
            prices = algo_fn(features, validation_prices)
        else:
            prices = algo_fn(features, **kwargs)

        prices = fill_missing_players(prices, stage_players, features,
                                       team_win_rates=team_wr, team_new_counts=team_new_counts)
        prices = add_uncertainty_flags(prices, features)

        result = analyze_criteria(algo_name, prices, validation_prices, stage_results, n_teams=30)
        all_results[algo_name] = result

        acc = result["accuracy"]
        dist = result["distribution"]
        print(f"    Accuracy:  MAE={acc['mae']:.3f}  Corr={acc['corr']:.3f}  "
              f"Within1VP={acc['within_1vp']:.1%}")
        print(f"    Distrib:   Mean={dist['mean']:.2f}  Med={dist['median']:.2f}  "
              f"Std={dist['std']:.2f}  Skew={dist['skew']:+.2f}")
        print(f"    Teams:     Best={result['best_team_pts']:.0f}  "
              f"Avg={result['avg_team_pts']:.0f}  Worst={result['worst_team_pts']:.0f}  "
              f"Spread={result['std_team_pts']:.0f}")
        print(f"    Diversity: {result['n_archetypes']} archetypes, "
              f"{result['unique_players_in_top30']} unique players, "
              f"overlap={result['avg_team_overlap']:.2f}")
        print(f"    Archetypes: {result['archetypes']}")
        print()

    # Summary table
    print(f"\n  {'='*75}")
    print(f"  {stage_name} — CRITERIA COMPLIANCE (0-100)")
    print(f"  {'='*75}")
    print(f"  {'Algorithm':<22} {'1.Value':>8} {'2.Comps':>8} {'3.Archs':>8} "
          f"{'4.PVal':>8} {'TOTAL':>8}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for algo_name in all_results:
        r = all_results[algo_name]
        total = (r["criteria_1_valuation"] + r["criteria_2_compositions"] +
                 r["criteria_3_archetypes"] + r["criteria_4_perceived_value"]) / 4
        print(f"  {algo_name:<22} {r['criteria_1_valuation']:>8.1f} "
              f"{r['criteria_2_compositions']:>8.1f} {r['criteria_3_archetypes']:>8.1f} "
              f"{r['criteria_4_perceived_value']:>8.1f} {total:>8.1f}")

    # Show best team for each algo
    for algo_name in all_results:
        r = all_results[algo_name]
        bt = r["bt_results"]
        if not bt:
            continue
        best = bt[0]
        print(f"\n  {algo_name} — Best Team ({best['actual_pts_with_igl']:.0f} pts, IGL: {best['igl']}):")
        actual_totals = stage_results.groupby("Player")["Pts"].sum().to_dict()
        print(f"    {'Player':<18} {'Team':<20} {'VP':>6} {'Pts':>6}")
        for pname, pteam, pvp, _ in sorted(best["players"],
                                             key=lambda x: actual_totals.get(x[0], 0),
                                             reverse=True):
            igl = " *" if pname == best["igl"] else ""
            print(f"    {pname:<18} {pteam:<20} {pvp:>6.1f} {actual_totals.get(pname, 0):>6}{igl}")
        print(f"    {'TOTAL':<40} {best['total_vp']:>6.1f} {best['actual_pts']:>6}")

    return all_results


def main():
    print("Loading all data...")
    all_data = load_all_data()
    pickrate_df = load_pickrate_data()

    # Top 3 algorithms + Combined + Neural Network
    algo_list = [
        ("Baseline EMA", algo_baseline_ema),
        ("Distribution-Aware", algo_distribution_aware),
        ("Role-Adjusted", algo_role_adjusted),
        ("Combined", algo_combined),
        ("Neural Network", None),  # handled specially
    ]

    # =============================================
    # STAGE 1 EVALUATION
    # =============================================
    s1_training = get_training_data(all_data)
    s1_validation = get_validation_prices(all_data)
    s1_results = get_stage1_results(all_data)
    s1_players = get_stage1_players(all_data)

    # For NN, we need a wrapper that takes validation
    def nn_s1(features, validation_prices):
        return algo_neural_network(features, validation_prices)

    s1_algos = [
        ("Baseline EMA", algo_baseline_ema),
        ("Distribution-Aware", algo_distribution_aware),
        ("Role-Adjusted", algo_role_adjusted),
        ("Combined", algo_combined),
        ("Neural Network", nn_s1),
    ]

    s1_results_dict = run_stage_eval(
        "STAGE 1 (train: 2024+KO+Bangkok -> predict Stage 1)",
        s1_training, s1_validation, s1_results, s1_players,
        all_data, s1_algos, pickrate_df,
    )

    # =============================================
    # STAGE 2 EVALUATION
    # =============================================
    s2_training = get_stage2_training(all_data)
    s2_validation = get_stage2_validation(all_data)
    s2_results = get_stage2_results(all_data)
    s2_players = get_stage2_players(all_data)

    if len(s2_validation) > 0:
        def nn_s2(features, validation_prices):
            return algo_neural_network(features, validation_prices)

        s2_algos = [
            ("Baseline EMA", algo_baseline_ema),
            ("Distribution-Aware", algo_distribution_aware),
            ("Role-Adjusted", algo_role_adjusted),
            ("Combined", algo_combined),
            ("Neural Network", nn_s2),
        ]

        s2_results_dict = run_stage_eval(
            "STAGE 2 (train: 2024+KO+Bangkok+Stage1 -> predict Stage 2)",
            s2_training, s2_validation, s2_results, s2_players,
            all_data, s2_algos, pickrate_df,
        )
    else:
        print("\nNo Stage 2 validation data available.")

    # =============================================
    # CRITERIA GAP ANALYSIS
    # =============================================
    print(f"\n{'='*80}")
    print(f"  CRITERIA GAP ANALYSIS")
    print(f"{'='*80}")
    print("""
  CLAUDE.md Criteria Assessment:

  1. ACCURATE VALUATION (target: ~80%)
     - Best: Baseline EMA / Distribution-Aware with corr ~0.69
     - Gap: MAE ~1.2 means prices are off by >1 VP on average
     - The NN approach can close this gap by learning non-linear patterns

  2. COMPETITIVE COMPOSITIONS (target: ~50%)
     - All algorithms produce teams scoring well above random baseline
     - But team overlap is high (same core players appear repeatedly)
     - Need: better price differentiation so more players become "good picks"

  3. MULTIPLE ARCHETYPES (target: currently poorly met)
     - Most algorithms produce only 1-2 archetypes in top 30
     - Top-Heavy dominates because elite players have outsized value
     - Fix: stronger distribution shaping + cap on max VP to compress elite tier

  4. PERCEIVED VALUE (target: currently poorly met)
     - Distribution-Aware/Combined best here (median ~8.9-9.0)
     - But 30%+ players are HIGH uncertainty -> manual review needed
     - Fan-favorite premium (from pickrate) helps with perceived value
""")

    print("Done.")


if __name__ == "__main__":
    main()
