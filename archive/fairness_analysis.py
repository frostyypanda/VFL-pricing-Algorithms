"""
Fairness Analysis of VFL Pricing Algorithms.

For the top 3 algorithms (Combined orig, S-Curve, Combined S-Curve):
1. Compute prices for Stage 1 players
2. Group into tiers, flag "pulled-down" players, check if they outperform tier avg
3. Compute fairness metric per tier (% within 1 std dev)
4. Pick rate equity across top-30 optimal teams
5. Opponent schedule analysis

Usage:
    python fairness_analysis.py
"""
import pandas as pd
import numpy as np
import os, sys

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DIR)

from pricing_algorithms import (
    load_all_data, get_training_data, get_validation_prices,
    get_stage1_results, get_stage1_players, compute_player_features,
    compute_team_win_rates, detect_new_players_on_team, load_pickrate_data,
    compute_player_pickrates, compute_team_popularity, compute_opponent_strength,
    algo_combined, fill_missing_players, add_uncertainty_flags,
    _snap_to_half, VP_MIN, VP_MAX,
)
from alternative_algorithms import algo_s_curve, algo_combined_scurve
from evaluate_pricing import find_optimal_teams, classify_archetype, backtest_teams


# ── Tier definitions ──
TIERS = [
    ("Budget", 6.0, 7.5),
    ("Mid",    8.0, 9.5),
    ("Good",  10.0, 11.5),
    ("Elite", 12.0, 15.0),
]


def assign_tier(vp):
    for name, lo, hi in TIERS:
        if lo <= vp <= hi:
            return name
    return "Other"


def sep(title, char="=", width=80):
    print()
    print(char * width)
    print(f"  {title}")
    print(char * width)


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    # ── Load data ──
    print("Loading data...")
    all_data   = load_all_data()
    training   = get_training_data(all_data)
    stage1_res = get_stage1_results(all_data)
    stage1_pl  = get_stage1_players(all_data)

    pickrate_df     = load_pickrate_data()
    player_pickrates = compute_player_pickrates(pickrate_df)
    features        = compute_player_features(training, player_pickrates=player_pickrates)
    team_wr         = compute_team_win_rates(training)
    team_new_counts = detect_new_players_on_team(training, stage1_pl)
    team_pop        = compute_team_popularity(pickrate_df, all_data)
    opp_strength    = compute_opponent_strength(training)

    algo_kwargs = dict(
        team_win_rates=team_wr,
        team_new_counts=team_new_counts,
        team_popularity=team_pop,
        opponent_strength=opp_strength,
    )

    # ── Run top-3 algorithms ──
    algos = {
        "Combined (orig)":   algo_combined(features, **algo_kwargs),
        "S-Curve":           algo_s_curve(features, **algo_kwargs),
        "Combined S-Curve":  algo_combined_scurve(features, **algo_kwargs),
    }

    # Fill missing players + uncertainty
    for name in algos:
        algos[name] = fill_missing_players(
            algos[name], stage1_pl, features,
            team_win_rates=team_wr, team_new_counts=team_new_counts,
        )
        algos[name] = add_uncertainty_flags(algos[name], features)

    # ── Actual Stage 1 total points per player ──
    actual_totals = stage1_res.groupby("Player")["Pts"].sum().to_dict()

    # ── EMA PPM rank (from features) — the "raw performance rank" ──
    ema_rank = features.set_index("Player")["ema_ppm"].rank(ascending=False).to_dict()
    n_features = len(features)

    # ==================================================================
    # SECTION 1 + 2 + 3: Tier analysis per algorithm
    # ==================================================================
    for algo_name, prices_df in algos.items():
        sep(f"ALGORITHM: {algo_name}")

        prices = prices_df.copy()
        prices["actual_pts"] = prices["Player"].map(actual_totals).fillna(0)
        prices["tier"] = prices["predicted_vp"].apply(assign_tier)

        # Price rank (1 = most expensive)
        prices["price_rank"] = prices["predicted_vp"].rank(ascending=False, method="min")
        # EMA PPM rank from features
        prices["ema_rank"] = prices["Player"].map(ema_rank)

        # "Pulled-down" flag: EMA rank much higher (lower number) than price rank
        # i.e. they are better performers than their price suggests
        # Threshold: their ema_rank is at least 20 percentile positions better
        prices["rank_gap"] = prices["price_rank"] - prices["ema_rank"].fillna(prices["price_rank"])
        threshold = 0.15 * n_features  # ~15% of player pool
        prices["pulled_down"] = prices["rank_gap"] > threshold

        for tier_name, tier_lo, tier_hi in TIERS:
            tier_df = prices[(prices["predicted_vp"] >= tier_lo) &
                             (prices["predicted_vp"] <= tier_hi) &
                             (prices["actual_pts"] > 0)].copy()

            if len(tier_df) == 0:
                print(f"\n  [{tier_name} tier ({tier_lo}-{tier_hi} VP)] — no players with actual pts")
                continue

            tier_avg = tier_df["actual_pts"].mean()
            tier_std = tier_df["actual_pts"].std() if len(tier_df) > 1 else 0

            # Fairness metric: % within 1 std dev of tier average
            if tier_std > 0:
                within_1sd = ((tier_df["actual_pts"] >= tier_avg - tier_std) &
                              (tier_df["actual_pts"] <= tier_avg + tier_std)).mean() * 100
            else:
                within_1sd = 100.0

            pulled = tier_df[tier_df["pulled_down"]]
            n_pulled = len(pulled)
            pulled_outperform = (pulled["actual_pts"] > tier_avg).sum() if n_pulled > 0 else 0

            print(f"\n  [{tier_name} tier ({tier_lo}-{tier_hi} VP)]  "
                  f"Players: {len(tier_df)}  |  Avg pts: {tier_avg:.1f}  |  Std: {tier_std:.1f}")
            print(f"  FAIRNESS: {within_1sd:.1f}% of players scored within 1 std dev of tier avg")
            if n_pulled > 0:
                print(f"  PULLED-DOWN players: {n_pulled}  "
                      f"({pulled_outperform}/{n_pulled} outperformed tier avg)")

            # Show individual players sorted by actual pts descending
            show_df = tier_df.sort_values("actual_pts", ascending=False)
            print(f"  {'Player':<20} {'Team':<22} {'VP':>5} {'Pts':>6} {'EMA Rk':>7} {'Pr Rk':>6} {'Gap':>5} {'Pulled':>7} {'vs Avg':>8}")
            print(f"  {'-'*20} {'-'*22} {'-'*5} {'-'*6} {'-'*7} {'-'*6} {'-'*5} {'-'*7} {'-'*8}")
            for _, r in show_df.iterrows():
                ema_r = r["ema_rank"]
                ema_str = f"{ema_r:.0f}" if pd.notna(ema_r) else "N/A"
                flag = " <<" if r["pulled_down"] else ""
                vs_avg = r["actual_pts"] - tier_avg
                vs_str = f"+{vs_avg:.0f}" if vs_avg >= 0 else f"{vs_avg:.0f}"
                print(f"  {r['Player']:<20} {r['Team']:<22} {r['predicted_vp']:>5.1f} "
                      f"{r['actual_pts']:>6.0f} {ema_str:>7} {r['price_rank']:>6.0f} "
                      f"{r['rank_gap']:>5.0f} {flag:>7} {vs_str:>8}")

    # ==================================================================
    # SECTION 4: Pick Rate Equity
    # ==================================================================
    sep("PICK RATE EQUITY (Top-30 Optimal Teams)")

    for algo_name, prices_df in algos.items():
        prices = prices_df.copy()
        # Expected pts = actual stage 1 total (hindsight-optimal)
        teams = find_optimal_teams(prices, actual_totals, n_teams=30, n_iter=20000)

        if not teams:
            print(f"\n  {algo_name}: No valid teams found")
            continue

        # Count appearances
        player_counts = {}
        total_slots = 0
        for t in teams:
            for p_name, p_team, p_vp, p_pts in t["players"]:
                player_counts[p_name] = player_counts.get(p_name, 0) + 1
                total_slots += 1

        n_unique = len(player_counts)
        total_possible = len(prices)
        max_appearances = max(player_counts.values())
        min_appearances = min(player_counts.values())
        counts = list(player_counts.values())
        appear_at_least_once_pct = n_unique / total_possible * 100

        # How many players appear in 50%+ of teams?
        heavy_use = sum(1 for c in counts if c >= 15)

        print(f"\n  {algo_name}:")
        print(f"    Total player pool:                {total_possible}")
        print(f"    Players appearing at least once:   {n_unique} ({appear_at_least_once_pct:.1f}%)")
        print(f"    Max appearances by single player:  {max_appearances} / 30 teams")
        print(f"    Min appearances (among selected):  {min_appearances} / 30 teams")
        print(f"    Players in 50%+ of teams (>=15):   {heavy_use}")
        print(f"    Mean appearances (among selected): {np.mean(counts):.1f}")

        # Top 10 most-picked
        top_picked = sorted(player_counts.items(), key=lambda x: -x[1])[:10]
        print(f"    Top-10 most picked:")
        for p, c in top_picked:
            print(f"      {p:<22} {c:>2}/30 teams")

    # ==================================================================
    # SECTION 5: Opponent Schedule Analysis
    # ==================================================================
    sep("OPPONENT SCHEDULE ANALYSIS")

    print("""
  Examining compute_opponent_strength() from pricing_algorithms.py:

  The function computes opponent strength as follows:
    1. De-duplicates training data to one row per (Team, Year, Stage, Wk, Game)
    2. Computes each team's overall W/L win rate across ALL training data
    3. Normalizes to 0-1 range

  FINDING: It does NOT use actual group-stage matchup data.
  It uses each team's OVERALL historical win rate as a proxy for how tough
  their schedule is. This means:
    - A team with a 70% win rate is ASSUMED to face tougher opponents
      (the logic is "strong teams face strong opponents in competitive groups")
    - But this is backwards / circular: it penalizes good teams rather than
      analyzing who they actually play against
    - The actual group draw (which teams are in which group) is not considered
    - Per-gameweek opponent identity is not factored in

  The opponent strength component has only 7% weight in Combined algorithms,
  so the impact is modest. But it's still a flawed signal:
    - It conflates team quality with schedule difficulty
    - Two teams in different groups (easy vs hard) get the same adjustment
      if they have the same win rate
    - It cannot capture that e.g. Team A faces 3 weak opponents while
      Team B faces 3 strong opponents in Stage 1 groups

  RECOMMENDATION: Replace with actual group-stage matchup data when available.
  For each team, compute average opponent win rate from the actual draw,
  not the team's own win rate.
""")

    # Show the actual opponent strength values
    print("  Current opponent_strength values (team own win rate, normalized 0-1):")
    sorted_opp = sorted(opp_strength.items(), key=lambda x: -x[1])
    for team, score in sorted_opp[:15]:
        print(f"    {team:<28} {score:.3f}")
    if len(sorted_opp) > 15:
        print(f"    ... and {len(sorted_opp) - 15} more teams")

    # ==================================================================
    # SUMMARY TABLE
    # ==================================================================
    sep("SUMMARY: FAIRNESS METRICS BY ALGORITHM")

    print(f"\n  {'Algorithm':<22} {'Tier':<10} {'Players':>8} {'Avg Pts':>8} {'Std':>8} "
          f"{'%in1SD':>8} {'Pulled':>8} {'Outperf':>8}")
    print(f"  {'-'*22} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for algo_name, prices_df in algos.items():
        prices = prices_df.copy()
        prices["actual_pts"] = prices["Player"].map(actual_totals).fillna(0)
        prices["tier"] = prices["predicted_vp"].apply(assign_tier)
        prices["ema_rank"] = prices["Player"].map(ema_rank)
        prices["price_rank"] = prices["predicted_vp"].rank(ascending=False, method="min")
        prices["rank_gap"] = prices["price_rank"] - prices["ema_rank"].fillna(prices["price_rank"])
        threshold = 0.15 * n_features
        prices["pulled_down"] = prices["rank_gap"] > threshold

        for tier_name, tier_lo, tier_hi in TIERS:
            tier_df = prices[(prices["predicted_vp"] >= tier_lo) &
                             (prices["predicted_vp"] <= tier_hi) &
                             (prices["actual_pts"] > 0)]
            if len(tier_df) == 0:
                continue
            tier_avg = tier_df["actual_pts"].mean()
            tier_std = tier_df["actual_pts"].std() if len(tier_df) > 1 else 0
            if tier_std > 0:
                within_1sd = ((tier_df["actual_pts"] >= tier_avg - tier_std) &
                              (tier_df["actual_pts"] <= tier_avg + tier_std)).mean() * 100
            else:
                within_1sd = 100.0
            n_pulled = tier_df["pulled_down"].sum()
            n_outperf = (tier_df[tier_df["pulled_down"]]["actual_pts"] > tier_avg).sum()

            print(f"  {algo_name:<22} {tier_name:<10} {len(tier_df):>8} {tier_avg:>8.1f} "
                  f"{tier_std:>8.1f} {within_1sd:>7.1f}% {n_pulled:>8} {n_outperf:>8}")

    print("\n  Legend:")
    print("    %in1SD  = % of tier players scoring within 1 std dev of tier avg (higher = fairer)")
    print("    Pulled  = players whose EMA rank >> price rank (repriced downward by curve)")
    print("    Outperf = of those pulled-down players, how many beat the tier average")
    print()


if __name__ == "__main__":
    main()
