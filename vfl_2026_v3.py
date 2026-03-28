"""
VFL 2026 Stage 1 — v3 Improved Pricing Algorithm.

Key improvements over v1 (vfl_2026_stage1.py):
  1. Adaptive recency weighting (based on 2026 data availability)
  2. Single tournament run dampening (outlier events don't dominate)
  3. Potential floor (peak performance lifts base price)
  4. Pickrate-weighted sentiment (community signal, stronger when data is scarce)
  5. Low data uncertainty flags for manual review

Usage:
    python vfl_2026_v3.py
"""
import pandas as pd
import numpy as np
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DIR = os.path.dirname(os.path.abspath(__file__))

from pricing_algorithms import (
    load_all_data, compute_team_win_rates, load_pickrate_data,
    compute_player_pickrates, compute_team_popularity, _snap_to_half,
    CHINA_TEAMS, STAGE_ORDER, GAME_ORDER,
)
from schedule_2026 import (
    SCHEDULE, GROUPS, GW_REGIONS, get_team_opponent, get_playing_teams,
)

# ============================================================================
#  CONSTANTS
# ============================================================================

VP_MIN, VP_MAX = 5.0, 15.0
BUDGET = 100
SQUAD_SIZE = 11
TARGET_MEAN = BUDGET / SQUAD_SIZE  # ~9.09

# S-curve quantile targets for distribution shaping
TARGET_QUANTILES = np.array([0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
                              0.60, 0.70, 0.80, 0.90, 0.95, 1.00])
TARGET_VPS      = np.array([5.0,  5.5,  6.0,  7.0,  7.5,  8.5,  9.0,
                              9.5, 10.0, 11.0, 12.5, 13.5, 15.0])

# Team name aliases for historical lookups
TEAM_HISTORY = {
    "FULL SENSE": ["FULL SENSE", "TALON"],
    "Eternal Fire": ["Eternal Fire", "ULF Esports"],
    "PCIFIC Esports": ["PCIFIC Esports", "PCIFIC Espor"],
    "LEVIATAN": ["LEVIATAN", "LEVIATÁN", "LEVIATµN", "LEVIAT\xb5N", "LEVIAT\xc1N"],
    "KRU Esports": ["KRU Esports", "KRÜ Esports", "KR Esports", "KR\x9a Esports", "KR\xdc Esports"],
}


# ============================================================================
#  DATA LOADING
# ============================================================================

def _read_csv_safe(path):
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def _normalize_team_name(name):
    """Normalize team names to consistent forms."""
    if not isinstance(name, str):
        return name
    replacements = [
        ("KR\x9a Esports", "KRU Esports"),
        ("KRÜ Esports",    "KRU Esports"),
        ("KR\xdc Esports", "KRU Esports"),
        ("LEVIAT\xb5N",    "LEVIATAN"),
        ("LEVIATÁN",       "LEVIATAN"),
        ("LEVIAT\xc1N",    "LEVIATAN"),
        ("PCIFIC Espor",   "PCIFIC Esports"),
        ("ULF Esports",    "Eternal Fire"),
    ]
    for old, new in replacements:
        if name == old:
            return new
    return name


def load_all_data_3years():
    """Load 2024 + 2025 + 2026 CSVs, normalize teams, filter China."""
    frames = []
    for year, filename in [(2024, "2024 VFL.csv"), (2025, "2025 VFL.csv"),
                           (2026, "2026 VFL.csv")]:
        path = os.path.join(DIR, filename)
        df = _read_csv_safe(path)
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
    """Load manual_prices_2026.csv, normalize team names."""
    path = os.path.join(DIR, "manual_prices_2026.csv")
    df = _read_csv_safe(path)
    df["Team"] = df["Team"].apply(_normalize_team_name)
    df["Player_lower"] = df["Player"].str.lower()
    return df


def load_schedule_opponent_strength(team_wr):
    """Compute per-team average opponent win rate across group-stage GWs."""
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
        result[team_n] = np.mean(opp_wrs) if opp_wrs else 0.5
    return result


# ============================================================================
#  CORE v3 ALGORITHM
# ============================================================================

def compute_player_event_ppms(all_data):
    """Compute per-player, per-event average PPM.

    Returns dict: player_lower -> list of (year, stage, avg_ppm, n_games)
    """
    played = all_data[all_data["P?"] == 1].copy()
    result = {}
    for (player, year, stage), grp in played.groupby(["Player", "Year", "Stage"]):
        pl = player.lower()
        if pl not in result:
            result[pl] = []
        result[pl].append({
            "player": player,
            "year": year,
            "stage": stage,
            "avg_ppm": grp["PPM"].mean(),
            "n_games": len(grp),
            "avg_pts": grp["Pts"].mean(),
            "avg_tpts": grp["T.Pts"].mean(),
            "std_ppm": grp["PPM"].std() if len(grp) > 1 else 0.0,
        })
    return result


def dampen_outlier_events(event_list):
    """Change 2: Single Tournament Run Dampening.

    If any event's PPM is > mean + 1.5*std, dampen it to keep 60% of the excess.
    Also dampen outlier LOWS (< mean - 1.5*std) to prevent one bad event from
    tanking an otherwise elite player.
    Returns (dampened_event_list, peak_event_ppm, had_outlier).
    """
    if len(event_list) < 2:
        peak = event_list[0]["avg_ppm"] if event_list else 0
        return event_list, peak, False

    ppms = np.array([e["avg_ppm"] for e in event_list])
    mean_ppm = ppms.mean()
    std_ppm = ppms.std()
    peak_ppm = ppms.max()

    if std_ppm < 0.01:
        return event_list, peak_ppm, False

    high_threshold = mean_ppm + 1.5 * std_ppm
    low_threshold = mean_ppm - 1.5 * std_ppm
    had_outlier = False

    dampened = []
    for e in event_list:
        e_copy = dict(e)
        if e_copy["avg_ppm"] > high_threshold:
            had_outlier = True
            dampened_ppm = mean_ppm + 0.6 * (e_copy["avg_ppm"] - mean_ppm)
            e_copy["avg_ppm"] = dampened_ppm
        elif e_copy["avg_ppm"] < low_threshold:
            # Dampen outlier low: pull toward mean, keeping 60% of the drop
            dampened_ppm = mean_ppm + 0.6 * (e_copy["avg_ppm"] - mean_ppm)
            e_copy["avg_ppm"] = dampened_ppm
        dampened.append(e_copy)

    return dampened, peak_ppm, had_outlier


def get_adaptive_year_weights(games_2026):
    """Change 1: Adaptive Recency Weighting based on 2026 data availability."""
    if games_2026 >= 8:
        return {2024: 0.15, 2025: 0.35, 2026: 1.0}
    elif games_2026 >= 4:
        return {2024: 0.35, 2025: 0.55, 2026: 0.85}
    elif games_2026 >= 1:
        return {2024: 0.50, 2025: 0.70, 2026: 0.65}
    else:
        return {2024: 0.60, 2025: 0.80, 2026: 0.0}


def compute_weighted_ppm(event_list, year_weights):
    """Compute adaptive-weighted average PPM across events."""
    if not event_list:
        return 3.5  # population fallback

    weighted_sum = 0.0
    weight_sum = 0.0
    for e in event_list:
        yr_w = year_weights.get(e["year"], 0.5)
        if yr_w <= 0:
            continue
        # Weight by year factor * number of games (more games = more reliable)
        w = yr_w * np.sqrt(e["n_games"])
        weighted_sum += e["avg_ppm"] * w
        weight_sum += w

    if weight_sum > 0:
        return weighted_sum / weight_sum
    return 3.5


def get_pickrate_signal(avg_pickpct):
    """Change 4: Pickrate-Weighted Sentiment signal."""
    if avg_pickpct > 30:
        return 0.15   # premium: managers wanted them
    elif avg_pickpct > 20:
        return 0.05   # slight premium
    elif avg_pickpct < 8:
        return -0.15  # discount: managers avoided them
    elif avg_pickpct < 15:
        return -0.05  # slight discount
    return 0.0


def get_uncertainty_flag(games_total, games_2026):
    """Change 5: Low Data Warning flags."""
    if games_total < 5:
        return "HIGH UNCERTAINTY - MANUAL REVIEW NEEDED"
    elif games_2026 == 0 and games_total < 10:
        return "HIGH UNCERTAINTY"
    elif games_2026 < 3 and games_total < 8:
        return "MEDIUM UNCERTAINTY"
    return ""


def ppm_to_raw_vp(ppm, ppm_min, ppm_max):
    """Map a PPM value to VP range linearly, for potential floor calculation."""
    if ppm_max - ppm_min < 0.01:
        return TARGET_MEAN
    frac = (ppm - ppm_min) / (ppm_max - ppm_min)
    return VP_MIN + frac * (VP_MAX - VP_MIN)


def generate_v3_prices(all_data, manual_df, pickrate_dict, team_pop,
                        team_wr, schedule_opp_strength):
    """Main v3 pricing pipeline.

    Returns DataFrame with Player, Team, Position, generated_vp, and metadata.
    """
    target_players = manual_df[["Player", "Team", "Position"]].copy()
    role_map = dict(zip(manual_df["Player_lower"], manual_df["Position"]))

    # Compute per-player, per-event PPMs
    event_ppms = compute_player_event_ppms(all_data)

    # Global PPM stats for shrinkage
    played = all_data[all_data["P?"] == 1]
    global_mean_ppm = played["PPM"].mean()

    avg_tw = np.mean(list(team_wr.values())) if team_wr else 0.5
    avg_opp = np.mean(list(schedule_opp_strength.values())) if schedule_opp_strength else 0.5

    records = []

    for _, row in target_players.iterrows():
        player = row["Player"]
        team = row["Team"]
        position = row["Position"]
        pl = player.lower()

        # Find player's event history (case-insensitive)
        events = event_ppms.get(pl, [])
        if not events:
            # Try exact match
            for key, val in event_ppms.items():
                if key == pl:
                    events = val
                    break

        # Count games per year
        games_2024 = sum(e["n_games"] for e in events if e["year"] == 2024)
        games_2025 = sum(e["n_games"] for e in events if e["year"] == 2025)
        games_2026 = sum(e["n_games"] for e in events if e["year"] == 2026)
        games_total = games_2024 + games_2025 + games_2026
        n_events = len(events)

        # --- Step c: Dampen outlier events (Change 2) ---
        dampened_events, peak_event_ppm, had_outlier = dampen_outlier_events(events)

        # --- Step d: Adaptive-weighted EMA (Change 1) ---
        year_weights = get_adaptive_year_weights(games_2026)
        weighted_ppm = compute_weighted_ppm(dampened_events, year_weights)

        # Bayesian shrinkage for low sample — stronger for fewer games
        shrinkage_games = 6
        weight_factor = games_total / (games_total + shrinkage_games)
        weighted_ppm = weight_factor * weighted_ppm + (1 - weight_factor) * global_mean_ppm

        # --- Step e: Role adjustment ---
        # Applied AFTER S-curve as a VP-level tweak (see below), not here.
        # This avoids distorting the rank ordering which the S-curve relies on.

        # --- Step f: Team strength (12%) ---
        # Scale: (tw - avg) ranges ~-0.3 to +0.3, so * 0.12 * weighted_ppm
        # gives ~-3.6% to +3.6% of PPM
        tw = team_wr.get(team, 0.5)
        team_str_adj = (tw - avg_tw) * 0.12 * weighted_ppm

        # --- Step g: Opponent schedule strength (7%) ---
        opp_str = schedule_opp_strength.get(team, 0.5)
        opp_adj = (avg_opp - opp_str) * 0.07 * weighted_ppm

        # --- Step h: Pickrate sentiment (Change 4) ---
        pick_info = pickrate_dict.get(pl, {})
        avg_pickpct = pick_info.get("avg_pickpct", 0.0)
        pickrate_signal = get_pickrate_signal(avg_pickpct)
        # Weight increases as 2026 data decreases
        pickrate_weight = max(0.05, 0.25 - games_2026 / 40)
        pickrate_adj = pickrate_signal * pickrate_weight * weighted_ppm

        # --- Step i: Team brand popularity (5%) ---
        team_pop_val = team_pop.get(team, 0.5)
        brand_adj = (team_pop_val - 0.5) * 0.05 * weighted_ppm

        # --- Step j: Consistency premium (3%) ---
        if events and games_total > 2:
            ppms_all = [e["avg_ppm"] for e in events]
            cv = np.std(ppms_all) / (np.mean(ppms_all) + 1e-9)
            consistency_score = max(0, 1 - cv)  # 0-1, higher = more consistent
        else:
            consistency_score = 0.5
        consistency_adj = (consistency_score - 0.5) * 0.03 * weighted_ppm

        # --- Additional: Small sample / recency-concentrated penalty ---
        # Players whose data is concentrated in few events get pulled toward
        # the global mean more, even if their game count is decent.
        # This handles cases like NSH where 27 games in 5 events all 2025-2026
        # produce consistently high PPMs that may not be sustainable.
        #
        # Also: players with above-average PPM and limited event history
        # get "hot streak dampening" -- we trust long track records more.
        weighted_ppm_after = weighted_ppm

        # Step 1: Event count shrinkage (pulls toward global mean)
        if n_events <= 3:
            event_shrink = 0.12 * (4 - n_events) / 3
            weighted_ppm_after = weighted_ppm * (1 - event_shrink) + global_mean_ppm * event_shrink

        # Step 2: Hot streak dampening for above-average players with few events
        # A player with 12+ events who scores high is proven; one with 3-5 events may be on a streak.
        # Scale: 8 events = baseline "proven", below that = progressive dampening
        PROVEN_EVENTS = 8
        if n_events < PROVEN_EVENTS and weighted_ppm_after > global_mean_ppm * 1.1:
            excess = weighted_ppm_after - global_mean_ppm
            # Dampening factor: 0 for 8+ events, up to 0.25 for 1 event
            dampen_strength = 0.25 * (PROVEN_EVENTS - n_events) / PROVEN_EVENTS
            weighted_ppm_after = weighted_ppm_after - excess * dampen_strength

        # Combine into a single adjusted PPM (all adjustments are proportional)
        adjusted_ppm = (weighted_ppm_after + team_str_adj + opp_adj +
                        pickrate_adj + brand_adj + consistency_adj)

        records.append({
            "Player": player,
            "Team": team,
            "Position": position,
            "adjusted_ppm": adjusted_ppm,
            "raw_weighted_ppm": weighted_ppm,
            "peak_event_ppm": peak_event_ppm,
            "had_outlier": had_outlier,
            "games_2024": games_2024,
            "games_2025": games_2025,
            "games_2026": games_2026,
            "games_total": games_total,
            "avg_pickpct": avg_pickpct,
            "pickrate_signal": pickrate_signal,
            "consistency_score": consistency_score,
            "uncertainty": get_uncertainty_flag(games_total, games_2026),
        })

    df = pd.DataFrame(records)

    # --- S-curve distribution shaping ---
    ranks = df["adjusted_ppm"].rank(pct=True).values
    mapped_vp = np.interp(ranks, TARGET_QUANTILES, TARGET_VPS)
    df["base_vp"] = mapped_vp

    # --- Step k: Potential floor (Change 3) ---
    # For each player, if they had an outlier peak, lift their base price
    ppm_min = df["adjusted_ppm"].min()
    ppm_max = df["adjusted_ppm"].max()

    final_vps = []
    for _, r in df.iterrows():
        base_price = r["base_vp"]

        if r["had_outlier"] and r["peak_event_ppm"] > 0:
            # Map peak PPM to VP range
            peak_price = ppm_to_raw_vp(r["peak_event_ppm"], ppm_min, ppm_max)
            peak_price = np.clip(peak_price, VP_MIN, VP_MAX)

            if peak_price > base_price:
                potential_boost = (peak_price - base_price) * 0.25
                final_vps.append(base_price + potential_boost)
            else:
                final_vps.append(base_price)
        else:
            final_vps.append(base_price)

    df["pre_shift_vp"] = final_vps

    # --- Step e (post-S-curve): Role adjustment on VP ---
    # Apply VP-level adjustment that fades toward the extremes.
    # Top players shouldn't be penalized for being duelists (they ARE top players).
    # Low players shouldn't be boosted for being sentinels (they ARE low players).
    # Full adjustment in the middle (7-12 VP), fading to 0 at extremes (5, 15).
    base_role_adj = {"D": -0.5, "S": 0.5, "C": 0.3, "I": 0.0}
    role_adjs = []
    for _, r in df.iterrows():
        base_adj = base_role_adj.get(r["Position"], 0.0)
        vp = r["pre_shift_vp"]
        # Fade factor: 1.0 in the 7-12 range, fading to 0 at 5 and 15
        if vp < 7:
            fade = max(0, (vp - 5) / 2)
        elif vp > 12:
            fade = max(0, (15 - vp) / 3)
        else:
            fade = 1.0
        role_adjs.append(base_adj * fade)
    df["role_adj_vp"] = role_adjs
    df["pre_shift_vp"] = df["pre_shift_vp"] + df["role_adj_vp"]

    # --- Shift to target mean ---
    current_mean = df["pre_shift_vp"].mean()
    shift = TARGET_MEAN - current_mean
    df["shifted_vp"] = df["pre_shift_vp"] + shift

    # --- Snap to 0.5, clip ---
    df["generated_vp"] = np.round(df["shifted_vp"] * 2) / 2
    df["generated_vp"] = df["generated_vp"].clip(VP_MIN, VP_MAX)

    return df


# ============================================================================
#  OUTPUT
# ============================================================================

def print_separator(char="=", width=80):
    print(char * width)


def main():
    print_separator()
    print("  VFL 2026 STAGE 1 — v3 IMPROVED PRICING ALGORITHM")
    print_separator()

    # --- Load data ---
    print("\n  Loading 2024 + 2025 + 2026 CSVs...")
    all_data = load_all_data_3years()
    print(f"  Combined: {len(all_data)} rows, {all_data['Player'].nunique()} players, "
          f"{all_data['Team'].nunique()} teams")

    print("\n  Loading manual prices...")
    manual_df = load_manual_prices()
    print(f"  Manual prices: {len(manual_df)} players, "
          f"mean={manual_df['Price'].mean():.2f}, "
          f"range=[{manual_df['Price'].min()}, {manual_df['Price'].max()}]")

    print("\n  Loading pickrate data...")
    pickrate_df = load_pickrate_data()
    pickrate_dict = compute_player_pickrates(pickrate_df) if pickrate_df is not None else {}
    print(f"  Pickrate data: {len(pickrate_dict)} players")

    # Compute team features
    played = all_data[all_data["P?"] == 1]
    team_wr = compute_team_win_rates(played)
    team_pop = compute_team_popularity(pickrate_df, all_data) if pickrate_df is not None else {}
    schedule_opp_strength = load_schedule_opponent_strength(team_wr)

    # --- Generate prices ---
    print("\n  Running v3 pricing algorithm...")
    result = generate_v3_prices(
        all_data, manual_df, pickrate_dict, team_pop,
        team_wr, schedule_opp_strength
    )
    print(f"  Generated prices for {len(result)} players")

    # ================================================================
    # 1. Print all prices sorted by VP desc
    # ================================================================
    print("\n" + "=" * 80)
    print("  ALL GENERATED PRICES (sorted by VP desc)")
    print("=" * 80)
    gen_sorted = result.sort_values("generated_vp", ascending=False)
    print(f"  {'#':>3} {'Player':<18} {'Team':<22} {'Pos':>3} {'VP':>6} {'Games':>6} {'Unc'}")
    print(f"  {'-'*80}")
    for i, (_, row) in enumerate(gen_sorted.iterrows(), 1):
        unc_str = f"  [{row['uncertainty']}]" if row['uncertainty'] else ""
        print(f"  {i:>3} {row['Player']:<18} {row['Team']:<22} "
              f"{row['Position']:>3} {row['generated_vp']:>6.1f} "
              f"{row['games_total']:>4}g{unc_str}")

    # ================================================================
    # 2. Compare to manual prices — print side by side with diff
    # ================================================================
    print("\n" + "=" * 80)
    print("  GENERATED vs MANUAL PRICES (side by side)")
    print("=" * 80)
    comparison = result.merge(manual_df[["Player", "Price"]], on="Player", how="inner")
    comparison["diff"] = comparison["generated_vp"] - comparison["Price"]
    comparison = comparison.sort_values("diff", ascending=False)

    print(f"  {'Player':<18} {'Team':<20} {'Pos':>3} {'Gen':>6} {'Man':>6} {'Diff':>6}")
    print(f"  {'-'*65}")
    for _, row in comparison.iterrows():
        sign = "+" if row["diff"] >= 0 else ""
        print(f"  {row['Player']:<18} {row['Team']:<20} "
              f"{row['Position']:>3} {row['generated_vp']:>6.1f} "
              f"{row['Price']:>6.1f} {sign}{row['diff']:>5.1f}")

    # ================================================================
    # 3. Biggest disagreements (>2 VP)
    # ================================================================
    print("\n" + "=" * 80)
    print("  BIGGEST DISAGREEMENTS (|diff| > 2.0 VP)")
    print("=" * 80)
    big_diff = comparison[comparison["diff"].abs() > 2.0].sort_values("diff", key=abs, ascending=False)
    if len(big_diff) == 0:
        print("  None! All prices within 2.0 VP of manual.")
    else:
        for _, row in big_diff.iterrows():
            sign = "+" if row["diff"] >= 0 else ""
            print(f"  {row['Player']:<18} {row['Team']:<20} "
                  f"Gen={row['generated_vp']:.1f}  Man={row['Price']:.1f}  "
                  f"Diff={sign}{row['diff']:.1f}")

    # ================================================================
    # 4. Uncertainty flags
    # ================================================================
    print("\n" + "=" * 80)
    print("  UNCERTAINTY FLAGS")
    print("=" * 80)
    unc_players = result[result["uncertainty"] != ""].sort_values("uncertainty")
    if len(unc_players) == 0:
        print("  No uncertainty flags.")
    else:
        for _, row in unc_players.iterrows():
            print(f"  {row['Player']:<18} {row['Team']:<20} "
                  f"VP={row['generated_vp']:.1f}  "
                  f"Games={row['games_total']}  "
                  f"2026={row['games_2026']}  "
                  f"[{row['uncertainty']}]")

    # ================================================================
    # 5. Distribution stats
    # ================================================================
    print("\n" + "=" * 80)
    print("  DISTRIBUTION STATISTICS")
    print("=" * 80)
    vps = result["generated_vp"]
    print(f"  Mean:   {vps.mean():.2f}  (target: {TARGET_MEAN:.2f})")
    print(f"  Median: {vps.median():.2f}")
    print(f"  Std:    {vps.std():.2f}")
    print(f"  Min:    {vps.min():.1f}")
    print(f"  Max:    {vps.max():.1f}")

    # Histogram buckets
    print("\n  Price Distribution:")
    buckets = pd.cut(vps, bins=np.arange(4.75, 15.75, 0.5))
    counts = buckets.value_counts().sort_index()
    for bucket, count in counts.items():
        if count > 0:
            bar = "#" * count
            print(f"  {bucket.left+0.25:>5.1f}: {count:>3} {bar}")

    # ================================================================
    # Key player sanity checks
    # ================================================================
    print("\n" + "=" * 80)
    print("  KEY PLAYER SANITY CHECKS")
    print("=" * 80)
    checks = {
        "aspas": "should be 13-15 (top player OAT)",
        "BABYBAY": "should be 8-10 (role change, low pickrate)",
        "something": "top player, should be 13-15",
        "Kaajak": "top EMEA duelist",
        "mada": "top NRG player",
    }
    # Also check NSH players
    nsh_players = ["Xross", "Ivy", "Francis", "Dambi", "Rb"]

    for p_name, desc in checks.items():
        match = result[result["Player"].str.lower() == p_name.lower()]
        if len(match) > 0:
            vp = match.iloc[0]["generated_vp"]
            man_match = manual_df[manual_df["Player"].str.lower() == p_name.lower()]
            man_vp = man_match.iloc[0]["Price"] if len(man_match) > 0 else "N/A"
            print(f"  {p_name:<18} Gen={vp:.1f}  Man={man_vp}  ({desc})")

    print(f"\n  Nongshim RedForce players (should NOT all be 12+):")
    for p_name in nsh_players:
        match = result[result["Player"] == p_name]
        if len(match) > 0:
            vp = match.iloc[0]["generated_vp"]
            man_match = manual_df[manual_df["Player"] == p_name]
            man_vp = man_match.iloc[0]["Price"] if len(man_match) > 0 else "N/A"
            print(f"    {p_name:<18} Gen={vp:.1f}  Man={man_vp}")

    # Check for duelists at 5-6
    low_duelists = result[(result["Position"] == "D") & (result["generated_vp"] <= 6.0)]
    if len(low_duelists) > 0:
        print(f"\n  WARNING: {len(low_duelists)} duelists at 5.0-6.0 VP:")
        for _, row in low_duelists.iterrows():
            print(f"    {row['Player']:<18} {row['Team']:<20} {row['generated_vp']:.1f} VP "
                  f"({row['games_total']} games)")
    else:
        print(f"\n  OK: No duelists priced at 5.0-6.0 VP")

    # ================================================================
    # Save to CSV
    # ================================================================
    out_path = os.path.join(DIR, "generated_prices_2026_v3.csv")
    save_cols = ["Player", "Team", "Position", "generated_vp", "games_total",
                 "games_2026", "uncertainty"]
    result[save_cols].sort_values("generated_vp", ascending=False).to_csv(
        out_path, index=False
    )
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
