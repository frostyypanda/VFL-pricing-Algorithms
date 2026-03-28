"""
VFL 2026 Stage 1 — Ensemble Pricing & Optimal Team Recommender.

Generates player prices via a Bayesian bootstrap ensemble, compares with manual
prices, runs auto-pick analysis, and recommends per-gameweek teams.

Usage:
    python vfl_2026_stage1.py
"""
import pandas as pd
import numpy as np
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Force UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DIR = os.path.dirname(os.path.abspath(__file__))

from pricing_algorithms import (
    load_all_data, compute_player_features, compute_team_win_rates,
    load_pickrate_data, compute_player_pickrates, compute_team_popularity,
    _snap_to_half, CHINA_TEAMS, STAGE_ORDER, GAME_ORDER,
)
from schedule_2026 import (
    SCHEDULE, GROUPS, GW_REGIONS, TEAM_ALIASES,
    get_team_opponent, get_playing_teams, get_team_region,
)
from evaluate_pricing import find_optimal_teams, classify_archetype, analyze_distribution


# ============================================================================
#  CONSTANTS
# ============================================================================

VP_MIN, VP_MAX = 5.0, 15.0   # Range for 2026 (wider than 2025)
BUDGET = 100
SQUAD_SIZE = 11
N_BOOTSTRAP = 100
TARGET_MEAN = 100.0 / 11     # ~9.09
NUM_GWS = 6
MAX_TRANSFERS = 3
RNG = np.random.default_rng(2026)

# Role constraints: 2D, 2C, 2I, 2S, 3W (wildcard = any)
ROLE_SLOTS = {"D": 2, "C": 2, "I": 2, "S": 2}
WILDCARD_SLOTS = 3

# Chronological training events with base recency weights (higher = more recent)
TRAINING_EVENTS = [
    # (year, stage, base_weight)
    (2024, "Kickoff",   0.30),
    (2024, "Madrid",    0.32),
    (2024, "Stage 1",   0.35),
    (2024, "Shanghai",  0.38),
    (2024, "Stage 2",   0.42),
    (2024, "Champions", 0.48),
    (2025, "Kickoff",   0.55),
    (2025, "bangkok",   0.60),
    (2025, "Stage 1",   0.65),
    (2025, "Toronto",   0.70),
    (2025, "Stage 2",   0.78),
    (2025, "Champions", 0.85),
    (2026, "Kickoff",   0.92),
    (2026, "Santiago",  1.00),
]

# S-curve quantile targets for distribution shaping (5.0-15.0 range)
TARGET_QUANTILES = np.array([0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
                              0.60, 0.70, 0.80, 0.90, 0.95, 1.0])
TARGET_VPS      = np.array([5.0, 5.5,  6.0,  7.0,  7.5,  8.5,  9.0,
                              9.5, 10.0, 11.0, 12.5, 13.5, 15.0])


# ============================================================================
#  SECTION 1 — DATA LOADING
# ============================================================================

def _read_csv_safe(path):
    """Read CSV with encoding fallback."""
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def _normalize_team_name(name):
    """Normalize team names across CSV / schedule / manual sources."""
    if not isinstance(name, str):
        return name
    # Fix known encoding mismatches
    # CSV has KR\x9a Esports, schedule has KRÜ Esports, manual has KRÜ Esports
    for old, new in [
        ("KR\x9a Esports", "KRU Esports"),
        ("KRÜ Esports",    "KRU Esports"),
        ("KR\xdc Esports", "KRU Esports"),
        ("LEVIAT\xb5N",    "LEVIATAN"),
        ("LEVIATÁN",       "LEVIATAN"),
        ("LEVIAT\xc1N",    "LEVIATAN"),
        ("PCIFIC Espor",   "PCIFIC Esports"),
        ("ULF Esports",    "Eternal Fire"),
    ]:
        if name == old:
            return new
    # Also apply TEAM_ALIASES from schedule
    if name in TEAM_ALIASES:
        return TEAM_ALIASES[name]
    return name


def load_all_data_with_2026():
    """Load 2024 + 2025 + 2026 CSVs, normalize teams, filter China."""
    frames = []
    for year, filename in [(2024, "2024 VFL.csv"), (2025, "2025 VFL.csv"),
                           (2026, "2026 VFL.csv")]:
        path = os.path.join(DIR, filename)
        df = _read_csv_safe(path)
        # 2024 CSV has unnamed first two columns (Team, Player)
        if "Team" not in df.columns:
            cols = list(df.columns)
            if cols[0].startswith("Unnamed") and cols[1].startswith("Unnamed"):
                cols[0], cols[1] = "Team", "Player"
                df.columns = cols
        df["Year"] = year
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined["Team"] = combined["Team"].apply(_normalize_team_name)
    combined = combined[~combined["Team"].isin(CHINA_TEAMS)].copy()
    return combined


def load_manual_prices():
    """Load manual_prices_2026.csv, normalize team names, build role map."""
    path = os.path.join(DIR, "manual_prices_2026.csv")
    df = _read_csv_safe(path)
    df["Team"] = df["Team"].apply(_normalize_team_name)
    df["Player_lower"] = df["Player"].str.lower()
    return df


def load_schedule_opponent_strength(team_wr):
    """Compute per-team average opponent win rate across 5 group-stage GWs.

    Returns dict: normalized_team_name -> avg_opponent_win_rate.
    """
    result = {}
    all_teams_in_schedule = set()
    for gw in range(1, 7):
        all_teams_in_schedule |= get_playing_teams(gw)

    for team in all_teams_in_schedule:
        team_n = _normalize_team_name(team)
        opp_wrs = []
        for gw in range(1, 7):
            opp = get_team_opponent(team, gw)
            if opp is not None:
                opp_n = _normalize_team_name(opp)
                wr = team_wr.get(opp_n, 0.5)
                opp_wrs.append(wr)
        if opp_wrs:
            result[team_n] = np.mean(opp_wrs)
        else:
            result[team_n] = 0.5
    return result


# ============================================================================
#  SECTION 2 — ENSEMBLE PRICING ALGORITHM
# ============================================================================

def compute_event_ppm(all_data, year, stage):
    """Compute average PPM per player for a given event, played games only."""
    mask = (all_data["Year"] == year) & (all_data["Stage"] == stage) & (all_data["P?"] == 1)
    sub = all_data[mask]
    if len(sub) == 0:
        return {}
    return sub.groupby("Player")["PPM"].mean().to_dict()


def compute_event_consistency(all_data, year, stage):
    """Compute PPM std-dev per player for a given event."""
    mask = (all_data["Year"] == year) & (all_data["Stage"] == stage) & (all_data["P?"] == 1)
    sub = all_data[mask]
    if len(sub) == 0:
        return {}
    return sub.groupby("Player")["PPM"].std().fillna(0).to_dict()


def bootstrap_ensemble_prices(all_data, manual_df, pickrate_dict, team_pop,
                               team_wr, schedule_opp_strength):
    """Run N_BOOTSTRAP iterations with Dirichlet-weighted events.

    Returns DataFrame with Player, Team, generated_vp.
    """
    # ---- Identify target players: those in manual_prices_2026 ----
    target_players = manual_df[["Player", "Team", "Position"]].copy()
    player_lower_to_name = {p.lower(): p for p in target_players["Player"]}

    # Build role map: player_lower -> position
    role_map = dict(zip(manual_df["Player_lower"], manual_df["Position"]))

    # ---- Pre-compute per-event PPMs ----
    event_ppms = {}
    event_consistency = {}
    for year, stage, _ in TRAINING_EVENTS:
        key = (year, stage)
        event_ppms[key] = compute_event_ppm(all_data, year, stage)
        event_consistency[key] = compute_event_consistency(all_data, year, stage)

    # ---- Base recency weights ----
    base_weights = np.array([w for _, _, w in TRAINING_EVENTS])

    # ---- Pre-compute team strength for each player ----
    # team_wr already normalized

    # ---- Collect all bootstrap predictions ----
    all_predictions = []  # list of dicts: player -> raw combined score

    for b in range(N_BOOTSTRAP):
        # Dirichlet noise over events
        dirichlet = RNG.dirichlet(np.ones(len(TRAINING_EVENTS)) * 2.0)
        weights = base_weights * dirichlet
        weights = weights / weights.sum()

        player_scores = {}

        for player in target_players["Player"]:
            pl = player.lower()
            team = target_players.loc[target_players["Player"] == player, "Team"].iloc[0]
            role = role_map.get(pl, "I")  # default Initiator (no adjustment)

            # a. Weighted EMA PPM (60%)
            ppm_vals = []
            ppm_weights = []
            for idx, (year, stage, _) in enumerate(TRAINING_EVENTS):
                key = (year, stage)
                ppm = event_ppms[key].get(player)
                if ppm is None:
                    # Case-insensitive fallback
                    for p_name, p_ppm in event_ppms[key].items():
                        if p_name.lower() == pl:
                            ppm = p_ppm
                            break
                if ppm is not None:
                    ppm_vals.append(ppm)
                    ppm_weights.append(weights[idx])

            if ppm_vals:
                w_arr = np.array(ppm_weights)
                if w_arr.sum() > 0:
                    weighted_ppm = np.average(ppm_vals, weights=w_arr)
                else:
                    weighted_ppm = np.mean(ppm_vals)
            else:
                weighted_ppm = 3.5  # population fallback

            # b. Role adjustment
            if role == "D":
                weighted_ppm *= 0.95   # duelists score more kills naturally
            elif role == "S":
                weighted_ppm *= 1.05   # sentinels undervalued
            elif role == "C":
                weighted_ppm *= 1.03   # controllers slight boost
            # I: no adjustment

            # c. Team strength with new-player discount (12%)
            tw = team_wr.get(team, 0.5)
            avg_tw = np.mean(list(team_wr.values())) if team_wr else 0.5
            team_str = (tw - avg_tw) * 2.0  # scale to roughly -1..+1

            # d. Opponent strength from schedule (7%)
            opp_str = schedule_opp_strength.get(team, 0.5)
            avg_opp = np.mean(list(schedule_opp_strength.values())) if schedule_opp_strength else 0.5
            opp_adj = (avg_opp - opp_str) * 2.0  # weaker opponents = positive

            # e. Pickrate popularity (8%)
            pick_info = pickrate_dict.get(pl, {})
            avg_pickpct = pick_info.get("avg_pickpct", 0.0)
            pick_max = 80.0  # approximate max
            pick_norm = min(avg_pickpct / pick_max, 1.0)

            # f. Team brand popularity (8%)
            team_pop_val = team_pop.get(team, 0.5)

            # g. Consistency premium (5%)
            consistency_vals = []
            for idx, (year, stage, _) in enumerate(TRAINING_EVENTS):
                key = (year, stage)
                std_val = event_consistency[key].get(player)
                if std_val is None:
                    for p_name, s_val in event_consistency[key].items():
                        if p_name.lower() == pl:
                            std_val = s_val
                            break
                if std_val is not None:
                    consistency_vals.append(std_val)
            if consistency_vals and weighted_ppm > 0:
                avg_std = np.mean(consistency_vals)
                cv = avg_std / (weighted_ppm + 1e-9)
                consistency_score = max(0, 1 - cv)  # 0-1, higher = more consistent
            else:
                consistency_score = 0.5

            # Normalize PPM to 0-1 range (approximate)
            ppm_norm = np.clip((weighted_ppm - 1.5) / (7.0 - 1.5), 0, 1)

            # Weighted combination
            combined = (
                0.60 * ppm_norm +
                0.12 * (team_str * 0.5 + 0.5) +  # rescale to 0-1 approx
                0.07 * (opp_adj * 0.5 + 0.5) +
                0.08 * pick_norm +
                0.08 * team_pop_val +
                0.05 * consistency_score
            )

            player_scores[player] = combined

        all_predictions.append(player_scores)

    # ---- Average across all bootstrap samples ----
    players_list = list(target_players["Player"])
    avg_scores = {}
    for player in players_list:
        vals = [pred.get(player, 0) for pred in all_predictions]
        avg_scores[player] = np.mean(vals)

    # ---- Distribution shaping with S-curve ----
    score_series = pd.Series(avg_scores)
    ranks = score_series.rank(pct=True)

    # Map percentile ranks to target VP via S-curve interpolation
    mapped_vp = {}
    for player in players_list:
        r = ranks[player]
        vp = np.interp(r, TARGET_QUANTILES, TARGET_VPS)
        mapped_vp[player] = vp

    # ---- Shift to target mean ----
    vp_series = pd.Series(mapped_vp)
    current_mean = vp_series.mean()
    shift = TARGET_MEAN - current_mean
    vp_series = vp_series + shift

    # ---- Snap to 0.5, clip to [5.0, 15.0] ----
    vp_series = np.round(vp_series * 2) / 2
    vp_series = vp_series.clip(VP_MIN, VP_MAX)

    # ---- Build result DataFrame ----
    result = target_players.copy()
    result["generated_vp"] = result["Player"].map(vp_series)
    result["combined_score"] = result["Player"].map(score_series)

    return result


# ============================================================================
#  SECTION 3 — AUTO-PICK ANALYSIS
# ============================================================================

def find_optimal_teams_with_roles(prices_df, expected_pts, role_map,
                                   n_teams=30, n_iter=10000, budget=BUDGET):
    """Find top N teams respecting role constraints: 2D, 2C, 2I, 2S, 3W.

    Uses greedy optimizer with random noise restarts.
    """
    players = prices_df.to_dict("records")
    for p in players:
        p["exp_pts"] = expected_pts.get(p["Player"], 0)
        p["role"] = role_map.get(p["Player"].lower(), "I")
        p["value"] = p["exp_pts"] / (p["price"] + 0.1)

    best_teams = []

    for iteration in range(n_iter):
        noise = RNG.normal(0, 0.3, len(players))
        scored = [(p, p["value"] + noise[i]) for i, p in enumerate(players)]
        scored.sort(key=lambda x: x[1], reverse=True)

        team = []
        team_vp = 0.0
        team_counts = {}
        role_counts = {"D": 0, "C": 0, "I": 0, "S": 0}
        wildcards_used = 0

        for p, _ in scored:
            if len(team) >= SQUAD_SIZE:
                break

            vp = p["price"]
            t = p["Team"]
            role = p["role"]

            # Budget check
            if team_vp + vp > budget:
                continue
            # Max 2 per team
            if team_counts.get(t, 0) >= 2:
                continue
            # Remaining budget check
            remaining_slots = SQUAD_SIZE - len(team) - 1
            if remaining_slots > 0 and (budget - team_vp - vp) < remaining_slots * VP_MIN:
                continue

            # Role constraint check
            if role_counts.get(role, 0) < ROLE_SLOTS.get(role, 0):
                # Fill dedicated role slot
                role_counts[role] = role_counts.get(role, 0) + 1
            elif wildcards_used < WILDCARD_SLOTS:
                # Use wildcard slot
                wildcards_used += 1
            else:
                continue

            team.append(p)
            team_vp += vp
            team_counts[t] = team_counts.get(t, 0) + 1

        if len(team) == SQUAD_SIZE:
            total_exp = sum(p["exp_pts"] for p in team)
            best_player_pts = max(p["exp_pts"] for p in team)
            total_with_igl = total_exp + best_player_pts

            team_info = {
                "players": [(p["Player"], p["Team"], p["price"], p["exp_pts"], p["role"])
                            for p in team],
                "total_vp": round(team_vp, 2),
                "expected_pts": round(total_exp, 1),
                "expected_pts_with_igl": round(total_with_igl, 1),
            }
            best_teams.append(team_info)

    # Deduplicate
    seen = set()
    unique = []
    for t in sorted(best_teams, key=lambda x: x["expected_pts_with_igl"], reverse=True):
        key = tuple(sorted(p[0] for p in t["players"]))
        if key not in seen:
            seen.add(key)
            unique.append(t)
    return unique[:n_teams]


def auto_pick_analysis(prices_df, expected_pts, role_map):
    """Find auto-picks and measure their value."""
    print("\n" + "=" * 80)
    print("  SECTION 3: AUTO-PICK ANALYSIS")
    print("=" * 80)

    print("\n  Finding top 30 optimal teams (10K iterations)...")
    teams = find_optimal_teams_with_roles(prices_df, expected_pts, role_map, n_teams=30)
    print(f"  Found {len(teams)} unique teams")

    if not teams:
        print("  No valid teams found!")
        return

    # Count player appearances
    player_counts = {}
    for t in teams:
        for p in t["players"]:
            player_counts[p[0]] = player_counts.get(p[0], 0) + 1

    # Auto-picks: appear in 25+ of 30 teams
    auto_picks = {p for p, c in player_counts.items() if c >= 25}
    print(f"\n  Auto-picks (in 25+/30 teams): {len(auto_picks)}")
    for p in sorted(auto_picks):
        count = player_counts[p]
        row = prices_df[prices_df["Player"] == p].iloc[0]
        print(f"    {p:<18} {row['Team']:<22} {row['price']:>6.1f} VP  "
              f"(in {count}/30 teams)")

    # Average expected points of top-30 teams
    avg_pts_with = np.mean([t["expected_pts_with_igl"] for t in teams])
    print(f"\n  Avg expected pts (top 30 with auto-picks): {avg_pts_with:.1f}")

    # Now find top-30 excluding auto-picks
    if auto_picks:
        filtered_df = prices_df[~prices_df["Player"].isin(auto_picks)].copy()
        print(f"\n  Finding top 30 teams EXCLUDING auto-picks...")
        teams_no_ap = find_optimal_teams_with_roles(
            filtered_df, expected_pts, role_map, n_teams=30
        )
        if teams_no_ap:
            avg_pts_without = np.mean([t["expected_pts_with_igl"] for t in teams_no_ap])
            print(f"  Avg expected pts (top 30 without auto-picks): {avg_pts_without:.1f}")
            diff = avg_pts_with - avg_pts_without
            print(f"  Point difference: {diff:.1f} pts ({diff/avg_pts_with*100:.1f}% loss)")
        else:
            print("  No valid teams found without auto-picks")

    return auto_picks


# ============================================================================
#  SECTION 4 — PER-GAMEWEEK TEAM RECOMMENDER
# ============================================================================

def get_player_expected_pts_gw(player, team, gw, player_avg_ppm, team_wr,
                                schedule_opp_strength):
    """Estimate expected points for a player in a specific gameweek."""
    playing_teams = get_playing_teams(gw)
    norm_playing = {_normalize_team_name(t) for t in playing_teams}

    if team not in norm_playing:
        return 0.0

    base_ppm = player_avg_ppm.get(player, 3.0)
    avg_maps = 2.3

    # Opponent strength adjustment
    opp = None
    for t_raw in playing_teams:
        if _normalize_team_name(t_raw) == team:
            opp_raw = get_team_opponent(t_raw, gw)
            if opp_raw:
                opp = _normalize_team_name(opp_raw)
            break

    opp_mult = 1.0
    if opp:
        opp_wr = team_wr.get(opp, 0.5)
        if opp_wr >= 0.6:
            opp_mult = 0.85  # strong opponent
        elif opp_wr <= 0.4:
            opp_mult = 1.15  # weak opponent
        else:
            opp_mult = 1.0 - (opp_wr - 0.5) * 0.6  # linear interpolation

    return base_ppm * avg_maps * opp_mult


def optimize_gw_team(available_players, gw, player_avg_ppm, team_wr,
                      schedule_opp_strength, role_map,
                      current_squad=None, max_transfers=3,
                      n_iter=5000, budget=BUDGET):
    """Optimize team for a single gameweek.

    If current_squad is None (GW1), build from scratch.
    Otherwise, make up to max_transfers changes.
    """
    # Compute expected pts for this GW for all players
    player_gw_pts = {}
    for _, row in available_players.iterrows():
        player = row["Player"]
        team = row["Team"]
        exp = get_player_expected_pts_gw(
            player, team, gw, player_avg_ppm, team_wr, schedule_opp_strength
        )
        player_gw_pts[player] = exp

    players = available_players.to_dict("records")
    for p in players:
        p["gw_pts"] = player_gw_pts.get(p["Player"], 0)
        p["role"] = role_map.get(p["Player"].lower(), "I")

    if current_squad is None:
        # GW1: build from scratch
        return _build_initial_team(players, gw, budget, n_iter)
    else:
        # GW2+: transfer up to max_transfers
        return _optimize_transfers(players, current_squad, gw, budget,
                                    max_transfers, n_iter)


def _check_role_constraints(team_players, role_map):
    """Check if a list of player dicts satisfies role constraints."""
    role_counts = {"D": 0, "C": 0, "I": 0, "S": 0}
    wildcards = 0
    for p in team_players:
        role = p.get("role", role_map.get(p["Player"].lower(), "I"))
        if role_counts.get(role, 0) < ROLE_SLOTS.get(role, 0):
            role_counts[role] += 1
        elif wildcards < WILDCARD_SLOTS:
            wildcards += 1
        else:
            return False
    return len(team_players) == SQUAD_SIZE


def _build_initial_team(players, gw, budget, n_iter):
    """Build initial team from scratch using greedy + random restarts."""
    best_team = None
    best_score = -1

    for iteration in range(n_iter):
        noise = RNG.normal(0, 0.2, len(players))
        scored = [(p, p["gw_pts"] / (p["price"] + 0.1) + noise[i])
                  for i, p in enumerate(players)]
        scored.sort(key=lambda x: x[1], reverse=True)

        team = []
        team_vp = 0.0
        team_counts = {}
        role_counts = {"D": 0, "C": 0, "I": 0, "S": 0}
        wildcards_used = 0

        for p, _ in scored:
            if len(team) >= SQUAD_SIZE:
                break

            vp = p["price"]
            t = p["Team"]
            role = p["role"]

            if team_vp + vp > budget:
                continue
            if team_counts.get(t, 0) >= 2:
                continue
            remaining = SQUAD_SIZE - len(team) - 1
            if remaining > 0 and (budget - team_vp - vp) < remaining * VP_MIN:
                continue

            # Role check
            if role_counts.get(role, 0) < ROLE_SLOTS.get(role, 0):
                role_counts[role] += 1
            elif wildcards_used < WILDCARD_SLOTS:
                wildcards_used += 1
            else:
                continue

            team.append(p)
            team_vp += vp
            team_counts[t] = team_counts.get(t, 0) + 1

        if len(team) == SQUAD_SIZE:
            total_pts = sum(p["gw_pts"] for p in team)
            best_player = max(team, key=lambda p: p["gw_pts"])
            total_with_igl = total_pts + best_player["gw_pts"]

            if total_with_igl > best_score:
                best_score = total_with_igl
                best_team = {
                    "players": team,
                    "total_vp": team_vp,
                    "expected_pts": total_pts,
                    "expected_pts_with_igl": total_with_igl,
                    "igl": best_player["Player"],
                    "transfers": [],
                }

    return best_team


def _optimize_transfers(all_players, current_squad, gw, budget,
                         max_transfers, n_iter):
    """Find optimal transfers for a gameweek."""
    current_names = {p["Player"] for p in current_squad["players"]}
    current_vp = current_squad["total_vp"]

    # Build lookup
    player_lookup = {p["Player"]: p for p in all_players}

    # Update current squad with this GW's expected points
    for p in current_squad["players"]:
        if p["Player"] in player_lookup:
            p["gw_pts"] = player_lookup[p["Player"]]["gw_pts"]
        else:
            p["gw_pts"] = 0

    best_result = None
    best_score = -1

    for iteration in range(n_iter):
        # Start from current squad
        squad = list(current_squad["players"])
        squad_names = set(current_names)
        squad_vp = current_vp
        transfers_made = []

        # Try up to max_transfers
        n_transfers = RNG.integers(0, max_transfers + 1)

        for _ in range(n_transfers):
            # Pick a random player to transfer out (prefer low gw_pts)
            squad_pts = [(i, p["gw_pts"]) for i, p in enumerate(squad)]
            squad_pts.sort(key=lambda x: x[1] + RNG.normal(0, 2))
            out_idx = squad_pts[0][0]
            out_player = squad[out_idx]

            # Find best replacement
            freed_vp = squad_vp - out_player["price"]
            remaining_budget = budget - freed_vp + out_player["price"]

            candidates = [p for p in all_players
                         if p["Player"] not in squad_names
                         and p["price"] <= budget - freed_vp + out_player["price"]]

            if not candidates:
                continue

            # Add noise for diversity
            noise = RNG.normal(0, 1.0, len(candidates))
            scored_candidates = [(c, c["gw_pts"] + noise[i])
                                for i, c in enumerate(candidates)]
            scored_candidates.sort(key=lambda x: x[1], reverse=True)

            # Try top candidates
            replaced = False
            for cand, _ in scored_candidates[:5]:
                new_vp = squad_vp - out_player["price"] + cand["price"]
                if new_vp > budget:
                    continue
                # Team constraint
                team_count = sum(1 for p in squad if p["Team"] == cand["Team"]
                               and p["Player"] != out_player["Player"])
                if team_count >= 2:
                    continue

                # Role constraint check (rebuild)
                test_squad = [p for p in squad if p["Player"] != out_player["Player"]]
                test_squad.append(cand)
                role_counts = {"D": 0, "C": 0, "I": 0, "S": 0}
                wildcards = 0
                valid = True
                for tp in test_squad:
                    role = tp["role"]
                    if role_counts.get(role, 0) < ROLE_SLOTS.get(role, 0):
                        role_counts[role] += 1
                    elif wildcards < WILDCARD_SLOTS:
                        wildcards += 1
                    else:
                        valid = False
                        break
                if not valid:
                    continue

                # Make the transfer
                squad = test_squad
                squad_names = {p["Player"] for p in squad}
                squad_vp = new_vp
                transfers_made.append((out_player["Player"], cand["Player"]))
                replaced = True
                break

        # Score this squad
        total_pts = sum(p["gw_pts"] for p in squad)
        if squad:
            best_player = max(squad, key=lambda p: p["gw_pts"])
            total_with_igl = total_pts + best_player["gw_pts"]
        else:
            total_with_igl = 0
            best_player = {"Player": "N/A"}

        if total_with_igl > best_score:
            best_score = total_with_igl
            best_result = {
                "players": list(squad),
                "total_vp": squad_vp,
                "expected_pts": total_pts,
                "expected_pts_with_igl": total_with_igl,
                "igl": best_player["Player"],
                "transfers": list(transfers_made),
            }

    return best_result


def run_gw_recommender(prices_df, player_avg_ppm, team_wr,
                        schedule_opp_strength, role_map, label=""):
    """Run per-gameweek team recommendations."""
    print(f"\n  --- {label} Per-Gameweek Team Recommendations ---")

    current_squad = None

    for gw in range(1, 7):
        # Determine which players are available
        playing_teams = get_playing_teams(gw)
        norm_playing = {_normalize_team_name(t) for t in playing_teams}

        if gw == 1:
            # GW1: EMEA + PAC only
            available = prices_df[prices_df["Team"].apply(
                lambda t: get_team_region_normalized(t) in ["EMEA", "PAC"]
            )].copy()
        else:
            available = prices_df.copy()

        result = optimize_gw_team(
            available, gw, player_avg_ppm, team_wr,
            schedule_opp_strength, role_map,
            current_squad=current_squad,
            max_transfers=MAX_TRANSFERS if gw > 1 else SQUAD_SIZE,
            n_iter=5000,
            budget=BUDGET,
        )

        if result is None:
            print(f"\n  GW{gw}: Failed to find valid team!")
            continue

        # Print GW result
        print(f"\n  GW{gw} ({', '.join(GW_REGIONS[gw])}):")
        if result["transfers"]:
            out_names = [t[0] for t in result["transfers"]]
            in_names = [t[1] for t in result["transfers"]]
            print(f"    Transfers: OUT {', '.join(out_names)} -> IN {', '.join(in_names)}")

        print(f"    {'Player':<18} {'Team':<22} {'VP':>6} {'Role':>4} {'Exp Pts':>8}")
        print(f"    {'-'*60}")
        sorted_players = sorted(result["players"],
                                key=lambda p: p["gw_pts"], reverse=True)
        for p in sorted_players:
            igl_mark = " *IGL*" if p["Player"] == result["igl"] else ""
            print(f"    {p['Player']:<18} {p['Team']:<22} {p['price']:>6.1f} "
                  f"{p['role']:>4} {p['gw_pts']:>8.1f}{igl_mark}")
        print(f"    {'-'*60}")
        print(f"    Total VP: {result['total_vp']:.1f}  |  "
              f"Expected: {result['expected_pts']:.1f}  |  "
              f"With IGL: {result['expected_pts_with_igl']:.1f}")

        current_squad = result


def get_team_region_normalized(team):
    """Get region for a normalized team name."""
    for region, groups in GROUPS.items():
        for gname, tlist in groups.items():
            for t in tlist:
                if _normalize_team_name(t) == team:
                    return region
    return None


# ============================================================================
#  SECTION 5 — OUTPUT & MAIN
# ============================================================================

def print_separator(char="=", width=80):
    print(char * width)


def main():
    print_separator()
    print("  VFL 2026 STAGE 1 — ENSEMBLE PRICING & OPTIMAL TEAM RECOMMENDER")
    print_separator()

    # ----------------------------------------------------------------
    #  SECTION 1: Data Loading
    # ----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  SECTION 1: DATA LOADING")
    print("=" * 80)

    print("\n  Loading 2024 + 2025 + 2026 CSVs...")
    all_data = load_all_data_with_2026()
    print(f"  Combined: {len(all_data)} rows, {all_data['Player'].nunique()} players, "
          f"{all_data['Team'].nunique()} teams")
    print(f"  Years: {sorted(all_data['Year'].unique())}")

    print("\n  Loading manual prices...")
    manual_df = load_manual_prices()
    print(f"  Manual prices: {len(manual_df)} players, "
          f"mean={manual_df['Price'].mean():.2f}, "
          f"range=[{manual_df['Price'].min()}, {manual_df['Price'].max()}]")

    print("\n  Loading pickrate data...")
    pickrate_df = load_pickrate_data()
    pickrate_dict = compute_player_pickrates(pickrate_df) if pickrate_df is not None else {}
    print(f"  Pickrate data: {len(pickrate_dict)} players")

    print("\n  Loading schedule data...")
    for gw in range(1, 7):
        teams = get_playing_teams(gw)
        print(f"    GW{gw}: {len(teams)} teams ({', '.join(GW_REGIONS[gw])})")

    # Compute training features
    # Training data: everything up to and including 2026 Santiago
    training_stages = set()
    for year, stage, _ in TRAINING_EVENTS:
        training_stages.add((year, stage))

    train_mask = all_data.apply(
        lambda r: (r["Year"], r["Stage"]) in training_stages and r["P?"] == 1, axis=1
    )
    training_df = all_data[train_mask].copy()
    print(f"\n  Training data: {len(training_df)} played games, "
          f"{training_df['Player'].nunique()} players")

    # Compute features
    team_wr = compute_team_win_rates(training_df)
    team_pop = compute_team_popularity(pickrate_df, all_data) if pickrate_df is not None else {}
    schedule_opp_strength = load_schedule_opponent_strength(team_wr)

    # Build role map from manual prices
    role_map = dict(zip(manual_df["Player_lower"], manual_df["Position"]))

    # Player avg PPM for GW recommendations (from training data, recent bias)
    # Use weighted average: 2026 data 2x, 2025 1x, 2024 0.5x
    def _compute_player_avg_ppm(training_df):
        result = {}
        for player, grp in training_df.groupby("Player"):
            ppm_vals = grp["PPM"].values
            year_vals = grp["Year"].values
            weights = np.where(year_vals == 2024, 0.5,
                      np.where(year_vals == 2025, 1.0, 2.0))
            if weights.sum() > 0:
                result[player] = np.average(ppm_vals, weights=weights)
            else:
                result[player] = np.mean(ppm_vals)
        return result

    player_avg_ppm = _compute_player_avg_ppm(training_df)

    # ----------------------------------------------------------------
    #  SECTION 2: Ensemble Pricing Algorithm
    # ----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  SECTION 2: ENSEMBLE PRICING ALGORITHM")
    print("=" * 80)

    print(f"\n  Running {N_BOOTSTRAP} bootstrap iterations...")
    generated = bootstrap_ensemble_prices(
        all_data, manual_df, pickrate_dict, team_pop, team_wr,
        schedule_opp_strength
    )
    print(f"  Generated prices for {len(generated)} players")

    # Distribution stats
    dist = analyze_distribution(generated["generated_vp"])
    print(f"\n  Distribution:")
    print(f"    Mean:   {dist['mean']:.2f} (target: {TARGET_MEAN:.2f})")
    print(f"    Median: {dist['median']:.2f}")
    print(f"    Std:    {dist['std']:.2f}")
    print(f"    Range:  [{dist['min']:.1f}, {dist['max']:.1f}]")

    # ---- SECTION 5a: Print all generated prices ----
    print("\n  --- Generated Prices (sorted by price desc) ---")
    gen_sorted = generated.sort_values("generated_vp", ascending=False)
    print(f"  {'#':>3} {'Player':<18} {'Team':<22} {'Pos':>3} {'VP':>6}")
    print(f"  {'-'*55}")
    for i, (_, row) in enumerate(gen_sorted.iterrows(), 1):
        print(f"  {i:>3} {row['Player']:<18} {row['Team']:<22} "
              f"{row['Position']:>3} {row['generated_vp']:>6.1f}")

    # ---- SECTION 5b: Comparison with manual prices ----
    print("\n\n  --- Generated vs Manual Prices (side by side) ---")
    comparison = generated.merge(manual_df[["Player", "Price"]], on="Player", how="inner")
    comparison["diff"] = comparison["generated_vp"] - comparison["Price"]
    comparison = comparison.sort_values("diff", ascending=False)

    print(f"  {'Player':<18} {'Team':<22} {'Gen VP':>7} {'Man VP':>7} {'Diff':>6}")
    print(f"  {'-'*62}")
    for _, row in comparison.iterrows():
        sign = "+" if row["diff"] >= 0 else ""
        print(f"  {row['Player']:<18} {row['Team']:<22} "
              f"{row['generated_vp']:>7.1f} {row['Price']:>7.1f} "
              f"{sign}{row['diff']:>5.1f}")

    # ---- SECTION 5c: Biggest disagreements ----
    big_diff = comparison[comparison["diff"].abs() > 2.0].sort_values("diff",
                                                                       key=abs,
                                                                       ascending=False)
    print(f"\n\n  --- Biggest Disagreements (|diff| > 2.0 VP) ---")
    if len(big_diff) > 0:
        for _, row in big_diff.iterrows():
            direction = "OVERPRICED by algo" if row["diff"] > 0 else "UNDERPRICED by algo"
            print(f"  {row['Player']:<18} {row['Team']:<22} "
                  f"Gen={row['generated_vp']:.1f}  Man={row['Price']:.1f}  "
                  f"({direction}, {abs(row['diff']):.1f} VP)")
    else:
        print("  No disagreements > 2.0 VP")

    # Summary stats
    mae = comparison["diff"].abs().mean()
    corr = comparison["generated_vp"].corr(comparison["Price"])
    print(f"\n  Overall: MAE={mae:.2f}, Correlation={corr:.3f}, "
          f"Mean diff={comparison['diff'].mean():+.2f}")

    # ----------------------------------------------------------------
    #  SECTION 3: Auto-Pick Analysis
    # ----------------------------------------------------------------
    # Prepare prices DF for optimizer
    # Use expected total points = avg_ppm * 5 games * 2.3 maps
    expected_pts_total = {p: ppm * 5 * 2.3 for p, ppm in player_avg_ppm.items()}

    # For generated prices
    gen_prices_for_opt = generated[["Player", "Team", "generated_vp"]].copy()
    gen_prices_for_opt.rename(columns={"generated_vp": "price"}, inplace=True)

    auto_picks_gen = auto_pick_analysis(gen_prices_for_opt, expected_pts_total, role_map)

    # For manual prices
    print("\n  --- Auto-Pick Analysis (Manual Prices) ---")
    man_prices_for_opt = manual_df[["Player", "Team", "Price"]].copy()
    man_prices_for_opt.rename(columns={"Price": "price"}, inplace=True)
    auto_picks_man = auto_pick_analysis(man_prices_for_opt, expected_pts_total, role_map)

    # ----------------------------------------------------------------
    #  SECTION 4: Per-Gameweek Team Recommender
    # ----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  SECTION 4: PER-GAMEWEEK TEAM RECOMMENDER")
    print("=" * 80)

    # Generated prices recommender
    run_gw_recommender(gen_prices_for_opt, player_avg_ppm, team_wr,
                        schedule_opp_strength, role_map,
                        label="GENERATED PRICES")

    # Manual prices recommender
    run_gw_recommender(man_prices_for_opt, player_avg_ppm, team_wr,
                        schedule_opp_strength, role_map,
                        label="MANUAL PRICES")

    # ----------------------------------------------------------------
    #  Save generated prices to CSV
    # ----------------------------------------------------------------
    output_path = os.path.join(DIR, "generated_prices_2026.csv")
    gen_sorted[["Player", "Team", "Position", "generated_vp", "combined_score"]].to_csv(
        output_path, index=False, encoding="utf-8"
    )
    print(f"\n  Saved generated prices to: {output_path}")

    # ----------------------------------------------------------------
    #  ALGORITHM DOCUMENTATION
    # ----------------------------------------------------------------
    print("\n\n" + "=" * 80)
    print("  ALGORITHM DOCUMENTATION")
    print("=" * 80)
    print("""
  1. HOW THE ENSEMBLE WORKS (Bayesian Bootstrap Over Events)
  ----------------------------------------------------------
  The algorithm creates 100 bootstrap samples. Each sample assigns random
  Dirichlet-distributed weights over 14 chronological training events
  (2024 Kickoff through 2026 Santiago). These random weights are multiplied
  by base recency weights (0.30 for 2024 Kickoff up to 1.00 for 2026
  Santiago), so more recent events always have higher influence on average
  but each bootstrap sample explores a different weighting.

  For each sample, every player's weighted-average PPM is computed across
  all events they participated in. This produces a distribution of 100
  price estimates per player. The final price is the average across all
  bootstrap samples, then shaped and snapped.

  2. FACTORS AND WEIGHTS
  ----------------------
  a. Weighted EMA PPM (60%): Core performance metric, recency-biased.
  b. Team strength (12%): Team win rate vs average, indicating T.Pts boost.
  c. Opponent strength from schedule (7%): Average opponent win rate across
     the player's 5 group-stage matchups. Weaker opponents = slight boost.
  d. Pickrate popularity (8%): Higher historical pick rate = more demand,
     which should increase price (supply/demand pricing).
  e. Team brand popularity (8%): Fan-favorite teams have inflated demand.
  f. Consistency premium (5%): Low coefficient of variation in PPM = premium.

  3. OPPONENT STRENGTH FROM ACTUAL SCHEDULE
  ------------------------------------------
  For each team, we look up their 5 group-stage opponents from the 2026
  Stage 1 schedule. Each opponent's historical win rate becomes a proxy for
  opponent quality. The average opponent win rate across all 5 matchups
  determines the schedule difficulty adjustment:
  - Easy schedule (low avg opponent WR) -> price boost
  - Hard schedule (high avg opponent WR) -> price discount

  4. ROLE ADJUSTMENTS
  -------------------
  Using actual position data from manual_prices_2026.csv:
  - D (Duelist): PPM * 0.95 — duelists naturally score more kills, so their
    raw PPM overstates their fantasy premium relative to other roles.
  - S (Sentinel): PPM * 1.05 — sentinels are undervalued by raw PPM since
    they contribute through team play and information denial.
  - C (Controller): PPM * 1.03 — slight boost for similar reasons.
  - I (Initiator): No adjustment (baseline role).

  5. UPSIDES, DOWNSIDES, AND RISKS
  ---------------------------------
  Upsides:
  + Bootstrap ensemble captures uncertainty: instead of one point estimate,
    we average over 100 different event-weighting scenarios.
  + Schedule-aware: actual matchup data informs pricing rather than generic
    opponent strength.
  + Multiple signals blended: performance, team strength, popularity, roles.
  + Distribution shaping ensures healthy price spread with median near 9.0.

  Downsides:
  - Relies on historical data that may not predict future performance well
    (meta shifts, roster changes, form fluctuations).
  - Dirichlet noise can be too uniform; extreme event weightings are rare.
  - Role adjustments are fixed multipliers rather than learned.
  - Pickrate data reflects past VFL manager behavior, which may be biased.

  Risks:
  - Newly transferred players with limited data get uncertain prices.
  - Teams with major roster changes may have unreliable team strength signals.
  - The 0.5 VP snapping can move many players to the same price point.

  6. WHAT IT OPTIMIZES FOR
  -------------------------
  The algorithm optimizes for a price list where:
  - Mean price is close to 9.09 (100 VP / 11 players)
  - Distribution follows an S-curve that enables multiple team archetypes
  - Prices reflect both individual skill and contextual factors (team, role,
    schedule, popularity)
  - Bootstrap averaging smooths out noise from any single event's results
""")


if __name__ == "__main__":
    main()
