"""
VFL Pricing Evaluation Framework.

Evaluates pricing algorithms by:
1. Price accuracy vs actual 2025 Stage 1 VP prices
2. Distribution analysis
3. Team composition optimization (top 15 teams)
4. Archetype diversity
5. Backtest: how those teams would have scored in Stage 1

Usage:
    python evaluate_pricing.py
"""
import pandas as pd
import numpy as np
from pricing_algorithms import run_all_algorithms, VP_MIN, VP_MAX, BUDGET, SQUAD_SIZE


# --- Price Accuracy ---


def evaluate_price_accuracy(predicted, actual):
    """Compare predicted vs actual VP prices.

    Args:
        predicted: DataFrame with Player, predicted_vp
        actual: DataFrame with Player, actual_vp

    Returns dict with MAE, RMSE, correlation, bias, bucket_accuracy.
    """
    merged = predicted.merge(actual, on="Player", how="inner")
    if len(merged) == 0:
        return {"mae": None, "rmse": None, "corr": None, "bias": None,
                "bucket_acc": None, "n_compared": 0}

    diff = merged["predicted_vp"] - merged["actual_vp"]
    mae = diff.abs().mean()
    rmse = np.sqrt((diff ** 2).mean())
    corr = merged["predicted_vp"].corr(merged["actual_vp"])
    bias = diff.mean()

    # Bucket accuracy: same integer VP bucket
    pred_bucket = merged["predicted_vp"].apply(np.floor).astype(int)
    actual_bucket = merged["actual_vp"].apply(np.floor).astype(int)
    bucket_acc = (pred_bucket == actual_bucket).mean()

    # Within 1 VP accuracy
    within_1 = (diff.abs() <= 1.0).mean()

    return {
        "mae": mae, "rmse": rmse, "corr": corr, "bias": bias,
        "bucket_acc": bucket_acc, "within_1vp": within_1,
        "n_compared": len(merged),
    }


# --- Distribution Analysis ---


def analyze_distribution(prices_series):
    """Analyze price distribution shape."""
    vals = prices_series.values
    return {
        "mean": np.mean(vals),
        "median": np.median(vals),
        "std": np.std(vals),
        "min": np.min(vals),
        "max": np.max(vals),
        "skew": pd.Series(vals).skew(),
        "pct_below_9": np.mean(vals < 9.0) * 100,
        "pct_6_8": np.mean((vals >= 6) & (vals < 8)) * 100,
        "pct_8_10": np.mean((vals >= 8) & (vals < 10)) * 100,
        "pct_10_12": np.mean((vals >= 10) & (vals < 12)) * 100,
        "pct_12_15": np.mean((vals >= 12) & (vals <= 15)) * 100,
    }


# --- Composition Optimizer ---


def find_optimal_teams(prices_df, expected_pts, n_teams=15, n_iter=10000,
                       budget=BUDGET, max_per_team=2):
    """Find top N teams using greedy + random restarts.

    Args:
        prices_df: DataFrame with Player, Team, predicted_vp
        expected_pts: dict mapping Player -> expected total Stage 1 points
        n_teams: number of best teams to return
        n_iter: number of random restart iterations
        budget: total VP budget
        max_per_team: max players from same VCT team

    Returns list of dicts, each with 'players', 'total_vp', 'expected_pts'.
    """
    rng = np.random.default_rng(42)

    # Build player array for fast iteration
    players = prices_df.to_dict("records")
    for p in players:
        p["exp_pts"] = expected_pts.get(p["Player"], 0)

    # Value metric: expected points per VP
    for p in players:
        p["value"] = p["exp_pts"] / (p["predicted_vp"] + 0.1)

    best_teams = []

    for iteration in range(n_iter):
        # Add random noise to value for diversity
        noise = rng.normal(0, 0.3, len(players))
        scored = [(p, p["value"] + noise[i]) for i, p in enumerate(players)]
        scored.sort(key=lambda x: x[1], reverse=True)

        team = []
        team_vp = 0.0
        team_counts = {}

        for p, _ in scored:
            if len(team) >= SQUAD_SIZE:
                break

            vp = p["predicted_vp"]
            t = p["Team"]

            # Check constraints
            if team_vp + vp > budget:
                continue
            if team_counts.get(t, 0) >= max_per_team:
                continue
            # Check if remaining budget can fill remaining slots at min price
            remaining_slots = SQUAD_SIZE - len(team) - 1
            if remaining_slots > 0 and (budget - team_vp - vp) < remaining_slots * VP_MIN:
                continue

            team.append(p)
            team_vp += vp
            team_counts[t] = team_counts.get(t, 0) + 1

        if len(team) == SQUAD_SIZE:
            total_exp = sum(p["exp_pts"] for p in team)
            # IGL: double the best player's expected points
            best_player_pts = max(p["exp_pts"] for p in team)
            total_with_igl = total_exp + best_player_pts  # original + doubled = +1x

            team_info = {
                "players": [(p["Player"], p["Team"], p["predicted_vp"], p["exp_pts"])
                            for p in team],
                "total_vp": round(team_vp, 2),
                "expected_pts": round(total_exp, 1),
                "expected_pts_with_igl": round(total_with_igl, 1),
            }
            best_teams.append(team_info)

    # Deduplicate and sort by expected pts with IGL
    seen = set()
    unique_teams = []
    for t in sorted(best_teams, key=lambda x: x["expected_pts_with_igl"], reverse=True):
        key = tuple(sorted(p[0] for p in t["players"]))
        if key not in seen:
            seen.add(key)
            unique_teams.append(t)

    return unique_teams[:n_teams]


# --- Archetype Classification ---


def classify_archetype(team_prices):
    """Classify a team composition into an archetype.

    Archetypes:
    - Stars & Scrubs: 2+ expensive (>=12) + 3+ cheap (<=7)
    - Top-Heavy: 1 star (>=13) + cheap filler (min<=7)
    - Balanced: low variance (std < 1.2), everyone similar price
    - Spread: all mid-range (7-11), no extremes
    - Dual-Star: exactly 2 premium (>=11) + rest mid/cheap
    - Mid-Heavy: mostly mid (8-10.5), 0-1 premium picks
    - Budget-Heavy: mean < 8.5, mostly cheap with 1-2 splurge picks
    """
    prices = np.array(team_prices)
    std = np.std(prices)
    mean_p = np.mean(prices)
    max_p = np.max(prices)
    min_p = np.min(prices)

    n_expensive = np.sum(prices >= 12)
    n_premium = np.sum(prices >= 11)
    n_cheap = np.sum(prices <= 7)
    n_mid = np.sum((prices >= 8) & (prices <= 10.5))

    if n_expensive >= 2 and n_cheap >= 3:
        return "Stars & Scrubs"
    if max_p >= 13 and min_p <= 7:
        return "Top-Heavy"
    if std < 1.2:
        return "Balanced"
    if np.all((prices >= 7) & (prices <= 11)):
        return "Spread"
    if n_premium == 2 and n_mid >= 5:
        return "Dual-Star"
    if n_mid >= 7 and n_expensive <= 1:
        return "Mid-Heavy"
    if mean_p < 8.5 and n_cheap >= 5:
        return "Budget-Heavy"
    return "Mixed"


# --- Backtest ---


def backtest_teams(teams, stage1_results):
    """Score generated teams against actual Stage 1 results.

    For each team, sum actual points across all Stage 1 games.
    IGL = player with highest total actual points (hindsight optimal).
    """
    # Pre-compute total Stage 1 points per player
    player_totals = stage1_results.groupby("Player")["Pts"].sum().to_dict()
    player_games = stage1_results.groupby("Player")["Pts"].count().to_dict()

    results = []
    for team in teams:
        player_names = [p[0] for p in team["players"]]
        player_pts = {p: player_totals.get(p, 0) for p in player_names}
        player_gp = {p: player_games.get(p, 0) for p in player_names}

        total_pts = sum(player_pts.values())
        best_player = max(player_pts, key=player_pts.get)
        igl_bonus = player_pts[best_player]  # IGL doubles best player
        total_with_igl = total_pts + igl_bonus

        results.append({
            "players": team["players"],
            "total_vp": team["total_vp"],
            "expected_pts": team["expected_pts"],
            "actual_pts": total_pts,
            "actual_pts_with_igl": total_with_igl,
            "igl": best_player,
            "igl_pts": player_pts[best_player],
            "avg_games_played": np.mean(list(player_gp.values())),
        })

    return sorted(results, key=lambda x: x["actual_pts_with_igl"], reverse=True)


# --- Random Baseline ---


def random_teams_baseline(prices_df, stage1_results, n_teams=100, n_iter=5000):
    """Generate random valid teams as a baseline comparison."""
    rng = np.random.default_rng(123)
    players = prices_df.to_dict("records")
    player_totals = stage1_results.groupby("Player")["Pts"].sum().to_dict()

    valid_teams = []
    for _ in range(n_iter):
        rng.shuffle(players)
        team = []
        team_vp = 0.0
        team_counts = {}

        for p in players:
            if len(team) >= SQUAD_SIZE:
                break
            vp = p["predicted_vp"]
            t = p["Team"]
            remaining = SQUAD_SIZE - len(team) - 1
            if team_vp + vp > BUDGET:
                continue
            if team_counts.get(t, 0) >= 2:
                continue
            if remaining > 0 and (BUDGET - team_vp - vp) < remaining * VP_MIN:
                continue
            team.append(p)
            team_vp += vp
            team_counts[t] = team_counts.get(t, 0) + 1

        if len(team) == SQUAD_SIZE:
            pts = sum(player_totals.get(p["Player"], 0) for p in team)
            best_pts = max(player_totals.get(p["Player"], 0) for p in team)
            valid_teams.append(pts + best_pts)

    if not valid_teams:
        return 0, 0
    valid_teams.sort(reverse=True)
    top_n = valid_teams[:n_teams]
    return np.mean(top_n), np.mean(valid_teams)


# --- Main Report ---


def print_separator(char="=", width=80):
    print(char * width)


def main():
    algorithms, features, validation, stage1_results = run_all_algorithms()

    # Pre-compute expected points per player (average Pts * expected games)
    # Stage 1 has ~7 games for playoff teams, ~5 for group stage
    player_avg_pts = features.set_index("Player")["avg_pts"].to_dict()
    # For expected points, use avg_pts * 5 (group stage games)
    expected_pts = {p: v * 5 for p, v in player_avg_pts.items()}

    # Also get actual total points per player for comparison
    actual_totals = stage1_results.groupby("Player")["Pts"].sum().to_dict()

    print_separator()
    print("VFL PRICING ALGORITHM EVALUATION REPORT")
    print_separator()
    print()

    # Summary table header
    summary_rows = []

    for algo_name, predictions in algorithms.items():
        print_separator("-")
        print(f"  ALGORITHM: {algo_name}")
        print_separator("-")

        # 1. Price Accuracy
        accuracy = evaluate_price_accuracy(predictions, validation)
        print(f"\n  Price Accuracy (vs actual 2025 Stage 1 VP):")
        print(f"    MAE: {accuracy['mae']:.3f}   RMSE: {accuracy['rmse']:.3f}   "
              f"Corr: {accuracy['corr']:.3f}   Bias: {accuracy['bias']:+.3f}")
        print(f"    Bucket accuracy: {accuracy['bucket_acc']:.1%}   "
              f"Within 1 VP: {accuracy['within_1vp']:.1%}")
        print(f"    Players compared: {accuracy['n_compared']}")

        # 2. Distribution Analysis
        dist = analyze_distribution(predictions["predicted_vp"])
        print(f"\n  Distribution:")
        print(f"    Mean: {dist['mean']:.2f}   Median: {dist['median']:.2f}   "
              f"Std: {dist['std']:.2f}   Skew: {dist['skew']:+.2f}")
        print(f"    Below 9 VP: {dist['pct_below_9']:.0f}%   "
              f"[6-8]: {dist['pct_6_8']:.0f}%   [8-10]: {dist['pct_8_10']:.0f}%   "
              f"[10-12]: {dist['pct_10_12']:.0f}%   [12-15]: {dist['pct_12_15']:.0f}%")

        # 3. Composition Optimization
        print(f"\n  Finding optimal teams (10K iterations)...")
        teams = find_optimal_teams(predictions, expected_pts)
        print(f"    Found {len(teams)} unique valid teams")

        if teams:
            # Archetype diversity
            archetypes = {}
            for t in teams:
                prices = [p[2] for p in t["players"]]
                arch = classify_archetype(prices)
                archetypes[arch] = archetypes.get(arch, 0) + 1

            print(f"\n  Archetype Diversity (top {len(teams)} teams):")
            for arch, count in sorted(archetypes.items(), key=lambda x: -x[1]):
                print(f"    {arch}: {count}")

            # 4. Backtest
            print(f"\n  Backtest (actual Stage 1 performance):")
            bt_results = backtest_teams(teams, stage1_results)

            if bt_results:
                actual_scores = [t["actual_pts_with_igl"] for t in bt_results]
                print(f"    Best team:  {max(actual_scores):.0f} pts (with IGL)")
                print(f"    Worst team: {min(actual_scores):.0f} pts")
                print(f"    Average:    {np.mean(actual_scores):.0f} pts")

                # Show best team composition
                best = bt_results[0]
                print(f"\n  Best Team (actual {best['actual_pts_with_igl']:.0f} pts, "
                      f"IGL: {best['igl']}):")
                print(f"    {'Player':<18} {'Team':<22} {'VP':>6} {'Actual Pts':>10}")
                for pname, pteam, pvp, _ in sorted(best["players"],
                                                     key=lambda x: actual_totals.get(x[0], 0),
                                                     reverse=True):
                    actual = actual_totals.get(pname, 0)
                    igl_mark = " *IGL*" if pname == best["igl"] else ""
                    print(f"    {pname:<18} {pteam:<22} {pvp:>6.1f} {actual:>10}{igl_mark}")
                print(f"    {'':_<60}")
                print(f"    {'TOTAL':<42} {best['total_vp']:>6.1f} {best['actual_pts']:>10}")

        print()

        summary_rows.append({
            "Algorithm": algo_name,
            "MAE": accuracy["mae"],
            "Corr": accuracy["corr"],
            "Mean VP": dist["mean"],
            "Median VP": dist["median"],
            "Std VP": dist["std"],
            "Best Team Pts": max(actual_scores) if teams and bt_results else 0,
            "Avg Team Pts": np.mean(actual_scores) if teams and bt_results else 0,
            "Archetypes": len(archetypes) if teams else 0,
        })

    # Random baseline
    print_separator("-")
    print("  BASELINE: Random Valid Teams")
    print_separator("-")
    # Use baseline EMA prices for random team generation
    baseline_prices = algorithms["Baseline EMA"]
    top_random, avg_random = random_teams_baseline(baseline_prices, stage1_results)
    print(f"    Top 100 random teams avg: {top_random:.0f} pts")
    print(f"    All random teams avg:     {avg_random:.0f} pts")
    print()

    # Summary comparison table
    print_separator("=")
    print("  SUMMARY COMPARISON")
    print_separator("=")
    summary = pd.DataFrame(summary_rows)
    print(f"\n  {'Algorithm':<22} {'MAE':>6} {'Corr':>6} {'Mean':>7} {'Med':>6} "
          f"{'Std':>5} {'Best':>6} {'Avg':>6} {'Arch':>5}")
    print(f"  {'':-<22} {'':-<6} {'':-<6} {'':-<7} {'':-<6} "
          f"{'':-<5} {'':-<6} {'':-<6} {'':-<5}")
    for _, row in summary.iterrows():
        print(f"  {row['Algorithm']:<22} {row['MAE']:>6.3f} {row['Corr']:>6.3f} "
              f"{row['Mean VP']:>7.2f} {row['Median VP']:>6.2f} {row['Std VP']:>5.2f} "
              f"{row['Best Team Pts']:>6.0f} {row['Avg Team Pts']:>6.0f} "
              f"{row['Archetypes']:>5}")
    print(f"\n  Random baseline: best={top_random:.0f}, avg={avg_random:.0f} pts")
    print()

    # Export predicted prices to CSV
    print("Exporting predicted prices to CSV...")
    all_prices = []
    for algo_name, predictions in algorithms.items():
        p = predictions.copy()
        p["algorithm"] = algo_name
        all_prices.append(p)
    pd.concat(all_prices).to_csv(
        "predicted_prices.csv", index=False, encoding="utf-8"
    )
    print("Saved: predicted_prices.csv")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
