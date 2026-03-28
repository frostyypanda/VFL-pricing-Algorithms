"""
VFL Pricing Algorithms — 6 approaches to player pricing for Valorant Fantasy League.

Training data: All 2024 + 2025 Kickoff + 2025 Bangkok (Masters 1)
Validation:    2025 Stage 1 (predict prices, simulate teams, backtest)

Factors considered:
  - EMA performance (PPM, recency-weighted)
  - Distribution shaping (force healthy VP spread)
  - Role proxy (kill patterns -> duelist/support adjustment)
  - Team strength (win rate, discounted by roster changes)
  - Variance/consistency premium
  - Pickrate popularity (from VFL manager data)
  - Opponent strength (for stage play matchups)
  - Team brand popularity (fan-favorite bias)

Usage:
    python pricing_algorithms.py
"""
import pandas as pd
import numpy as np
import os

DIR = os.path.dirname(os.path.abspath(__file__))

# --- Constants ---

VP_MIN, VP_MAX = 6.0, 15.0
BUDGET = 100
SQUAD_SIZE = 11

CHINA_TEAMS = [
    "Bilibili Gaming", "Dragon Ranger Gaming", "Edward Gaming",
    "FunPlus Phoenix", "JD Mall JDG Esports", "Nova Esports",
    "Titan Esports Club", "Trace Esports", "Wolves Esports",
    "Xi Lai Gaming", "TYLOO", "All Gamers",
]

# Chronological order for sorting games
STAGE_ORDER = {
    "Kickoff": 0, "Madrid": 1, "Stage 1": 2, "Shanghai": 3,
    "Stage 2": 4, "Champions": 5, "bangkok": 1,
}

GAME_ORDER = {
    "G1": 0, "G2": 1, "G3": 2, "G4": 3, "G5": 4,
    "SR1": 5, "SR2": 6, "SR3": 7,
    "UR1": 8, "UR2": 9, "UQF": 10, "USF": 11, "UF": 12,
    "LR1": 13, "LR2": 14, "LR3": 15, "LR4": 16, "LF": 17, "GF": 18,
    "MR1": 19, "MR2": 20,
}

# Target quantile breakpoints for distribution-aware pricing
# Tuned for CLAUDE.md criteria: median ~9.0, healthy spread, multiple archetypes
TARGET_QUANTILES = np.array([0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
                              0.60, 0.70, 0.80, 0.90, 0.95, 1.0])
TARGET_VPS =      np.array([6.0, 6.3,  6.5,  7.2,  8.0,  8.6,  9.0,
                              9.5, 10.2, 11.0, 12.5, 13.5, 15.0])

# Uncertainty thresholds
LOW_GAMES_THRESHOLD = 3      # < 3 games = high uncertainty
NEW_PLAYER_THRESHOLD = 0     # 0 games in training = new player

# --- Data Loading ---


def load_all_data():
    """Load and combine 2024 + 2025 CSVs, add Year column, filter China."""
    frames = []
    for year, filename in [(2024, "2024 VFL.csv"), (2025, "2025 VFL.csv")]:
        path = os.path.join(DIR, filename)
        try:
            df = pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin-1")
        df["Year"] = year
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined[~combined["Team"].isin(CHINA_TEAMS)].copy()
    return combined


def get_training_data(df):
    """Training set: all 2024 + 2025 Kickoff + 2025 Bangkok, played games only."""
    mask_2024 = df["Year"] == 2024
    mask_2025_train = (df["Year"] == 2025) & df["Stage"].isin(["Kickoff", "bangkok"])
    played = df["P?"] == 1
    return df[(mask_2024 | mask_2025_train) & played].copy()


def get_validation_prices(df):
    """Actual VP prices at 2025 Stage 1 G1 start (ground truth for evaluation)."""
    mask = (
        (df["Year"] == 2025)
        & (df["Stage"] == "Stage 1")
        & (df["Game"] == "G1")
        & (df["P?"] == 1)
    )
    vp = df.loc[mask, ["Team", "Player", "Game Start VP"]].copy()
    vp = vp.rename(columns={"Game Start VP": "actual_vp"})
    return vp.drop_duplicates(subset="Player")


def get_stage1_results(df):
    """All played Stage 1 rows for backtesting."""
    mask = (df["Year"] == 2025) & (df["Stage"] == "Stage 1") & (df["P?"] == 1)
    return df[mask].copy()


def get_stage1_players(df):
    """All players who appear in 2025 Stage 1 (including P?=0), non-China."""
    mask = (df["Year"] == 2025) & (df["Stage"] == "Stage 1")
    players = df.loc[mask, ["Team", "Player"]].drop_duplicates(subset="Player")
    return players


# --- Pickrate & Popularity Data ---


def load_pickrate_data():
    """Load pickrate summary if available. Returns DataFrame or None."""
    path = os.path.join(DIR, "pickrate_summary.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def compute_team_popularity(pickrate_df, all_data):
    """Compute team brand popularity from pickrate data.

    Teams like Paper Rex, Sentinels etc. have inflated pick rates due to fan loyalty.
    Returns dict: team -> popularity_score (0-1 normalized).
    """
    if pickrate_df is None:
        return {}

    # Map players to teams from game data
    player_teams = all_data.groupby("Player")["Team"].last().to_dict()

    pick_with_team = pickrate_df.copy()
    pick_with_team["Team"] = pick_with_team["Player"].map(player_teams)
    pick_with_team = pick_with_team.dropna(subset=["Team"])

    # Team popularity = mean pick% of all players on that team
    team_pop = pick_with_team.groupby("Team")["avg_pickpct"].mean()

    # Normalize to 0-1
    if team_pop.max() - team_pop.min() > 0:
        normalized = (team_pop - team_pop.min()) / (team_pop.max() - team_pop.min())
    else:
        normalized = team_pop * 0 + 0.5
    return normalized.to_dict()


def compute_player_pickrates(pickrate_df):
    """Get per-player pickrate stats.

    Returns dict: player_name_lower -> {avg_pickpct, avg_rank_pct, events_appeared}
    """
    if pickrate_df is None:
        return {}
    result = {}
    for _, row in pickrate_df.iterrows():
        result[row["Player"].lower()] = {
            "avg_pickpct": row["avg_pickpct"],
            "avg_rank_pct": row["avg_rank_pct"],
            "events_appeared": row["events_appeared"],
            "total_picks": row["total_picks"],
        }
    return result


def compute_opponent_strength(training_df):
    """Compute opponent strength ratings for stage play.

    In group stages, each team plays 5 games against known opponents.
    A player's expected points are affected by opponent quality:
    - Strong opponents -> lower expected T.Pts (more likely to lose maps)
    - Weak opponents -> higher expected T.Pts

    Returns dict: team -> opponent_strength_score (0-1, higher = tougher schedule).
    For now, we use overall team win rates as a proxy for opponent strength.
    """
    df = training_df.drop_duplicates(subset=["Team", "Year", "Stage", "Wk", "Game"])
    team_wr = df.groupby("Team")["W/L"].apply(lambda x: (x == "W").mean()).to_dict()

    # Normalize to 0-1
    values = np.array(list(team_wr.values()))
    if values.max() - values.min() > 0:
        for team in team_wr:
            team_wr[team] = (team_wr[team] - values.min()) / (values.max() - values.min())
    return team_wr


# --- Feature Engineering ---


def shrink_estimate(player_val, n_games, pop_mean, shrinkage_games=4):
    """Bayesian shrinkage toward population mean for small samples."""
    weight = n_games / (n_games + shrinkage_games)
    return weight * player_val + (1 - weight) * pop_mean


def detect_new_players_on_team(training_df, stage1_players):
    """Detect how many new players each team has in Stage 1 vs training data.

    Returns dict: team -> number of new players (0-5).
    """
    trained_by_team = training_df.groupby("Team")["Player"].apply(set).to_dict()
    stage1_by_team = stage1_players.groupby("Team")["Player"].apply(set).to_dict()

    team_new_counts = {}
    for team, s1_players in stage1_by_team.items():
        trained_players = trained_by_team.get(team, set())
        new_count = len(s1_players - trained_players)
        team_new_counts[team] = new_count
    return team_new_counts


def compute_player_features(training_df, player_pickrates=None):
    """Compute per-player features from training data."""
    if player_pickrates is None:
        player_pickrates = {}
    df = training_df.copy()
    df["_stage_ord"] = df["Stage"].map(STAGE_ORDER).fillna(99)
    df["_game_ord"] = df["Game"].map(GAME_ORDER).fillna(99)
    df = df.sort_values(["Year", "_stage_ord", "Wk", "_game_ord"])

    pop_mean_ppm = df["PPM"].mean()

    records = []
    for player, grp in df.groupby("Player"):
        team = grp["Team"].iloc[-1]
        games = len(grp)
        ppm_vals = grp["PPM"].values
        pts_vals = grp["Pts"].values
        tpts_vals = grp["T.Pts"].values
        ppts_vals = grp["P.Pts"].values
        year_vals = grp["Year"].values

        # Year discount: 2024 games get 0.5 weight
        year_weights = np.where(year_vals == 2024, 0.5, 1.0)

        # EMA with year discount
        alpha = 0.3
        n = len(ppm_vals)
        ema_weights = np.array([(1 - alpha) ** (n - 1 - i) for i in range(n)])
        combined_weights = ema_weights * year_weights
        if combined_weights.sum() > 0:
            ema_ppm = np.average(ppm_vals, weights=combined_weights)
        else:
            ema_ppm = ppm_vals.mean()

        # Shrink toward population mean
        ema_ppm = shrink_estimate(ema_ppm, games, pop_mean_ppm)

        avg_ppm = np.mean(ppm_vals)
        std_ppm = np.std(ppm_vals) if games > 1 else 0.0
        avg_pts = np.mean(pts_vals)
        avg_tpts = np.mean(tpts_vals)
        avg_ppts = np.mean(ppts_vals)

        wl = grp["W/L"].values
        wins = np.sum(wl == "W")
        win_rate = wins / games if games > 0 else 0.5

        # Kill profile: fraction of maps in 20k+ brackets
        kill_cols = ["20k", "25k", "30k", "35k", "40k", "45k", "50k"]
        total_maps = grp[["0k", "5k", "10k", "15k", "20k", "25k", "30k",
                          "35k", "40k", "45k", "50k"]].sum().sum()
        high_kills = grp[kill_cols].sum().sum()
        kill_profile = high_kills / total_maps if total_maps > 0 else 0.0

        rating_bonuses = grp[["TOP3", "TOP2", "TOP1", "1.5R2", "1.75R2", "2.0R2"]].sum().sum()
        rating_bonus_rate = rating_bonuses / games if games > 0 else 0.0

        ppts_ratio = avg_ppts / (avg_pts + 1e-9)

        # Pickrate features
        pick_info = player_pickrates.get(player.lower(), {})
        avg_pickpct = pick_info.get("avg_pickpct", 0.0)
        avg_rank_pct = pick_info.get("avg_rank_pct", 50.0)

        # Uncertainty assessment
        uncertainty_reasons = []
        if games < LOW_GAMES_THRESHOLD:
            uncertainty_reasons.append(f"few games ({games})")

        records.append({
            "Player": player,
            "Team": team,
            "games_played": games,
            "ema_ppm": ema_ppm,
            "avg_ppm": avg_ppm,
            "std_ppm": std_ppm,
            "avg_pts": avg_pts,
            "avg_tpts": avg_tpts,
            "avg_ppts": avg_ppts,
            "win_rate": win_rate,
            "kill_profile": kill_profile,
            "rating_bonus_rate": rating_bonus_rate,
            "ppts_ratio": ppts_ratio,
            "avg_pickpct": avg_pickpct,
            "avg_rank_pct": avg_rank_pct,
            "uncertainty_reasons": uncertainty_reasons,
        })

    features = pd.DataFrame(records)
    return features


def compute_team_win_rates(training_df):
    """Compute team win rates from training data."""
    df = training_df.drop_duplicates(subset=["Team", "Year", "Stage", "Wk", "Game"])
    team_records = df.groupby("Team")["W/L"].agg(
        total="count", wins=lambda x: (x == "W").sum()
    )
    team_records["win_rate"] = team_records["wins"] / team_records["total"]
    return team_records["win_rate"].to_dict()


# --- Pricing Algorithms ---


def _linear_map(values, target_min=VP_MIN, target_max=VP_MAX):
    """Linear map from value range to VP range, clipped."""
    vmin, vmax = values.min(), values.max()
    if vmax - vmin < 1e-9:
        return np.full_like(values, (target_min + target_max) / 2)
    scaled = target_min + (values - vmin) / (vmax - vmin) * (target_max - target_min)
    return np.clip(scaled, target_min, target_max)


def algo_baseline_ema(features, **kwargs):
    """Algorithm 1: Baseline EMA pricing.
    Linear map of EMA PPM to [6, 15] VP."""
    prices = features.copy()
    prices["predicted_vp"] = _linear_map(prices["ema_ppm"].values)
    return prices[["Player", "Team", "predicted_vp"]]


def algo_distribution_aware(features, **kwargs):
    """Algorithm 2: Distribution-aware pricing.
    Rank players by EMA PPM, then map percentiles to target distribution."""
    prices = features.copy()
    ranks = prices["ema_ppm"].rank(pct=True).values
    predicted = np.interp(ranks, TARGET_QUANTILES, TARGET_VPS)
    prices["predicted_vp"] = np.clip(predicted, VP_MIN, VP_MAX)
    return prices[["Player", "Team", "predicted_vp"]]


def algo_role_adjusted(features, role_data=None, **kwargs):
    """Algorithm 3: Role-adjusted pricing.
    Proxy roles from scoring patterns, adjust PPM before pricing."""
    prices = features.copy()

    if role_data is not None:
        # TODO: use actual role data when available
        pass

    kill_pctl = prices["kill_profile"].rank(pct=True)
    ppts_pctl = prices["ppts_ratio"].rank(pct=True)
    role_score = (kill_pctl + ppts_pctl) / 2

    adjustment = np.where(
        role_score > 0.7, 0.93,
        np.where(role_score < 0.3, 1.05, 1.0)
    )
    adjusted_ppm = prices["ema_ppm"] * adjustment
    prices["predicted_vp"] = _linear_map(adjusted_ppm.values)
    return prices[["Player", "Team", "predicted_vp"]]


def algo_team_strength(features, team_win_rates=None, team_new_counts=None, **kwargs):
    """Algorithm 4: Team strength multiplier.
    Decompose into personal + team components, predict each separately.

    New player discount on team strength factor:
    - 1 new player on team: team factor *= 0.67
    - 2 new players: team factor *= 0.33
    - 3+ new players: ignore team strength entirely (factor = 0)
    """
    if team_win_rates is None:
        team_win_rates = {}
    if team_new_counts is None:
        team_new_counts = {}

    prices = features.copy()
    prices["team_wr"] = prices["Team"].map(team_win_rates).fillna(0.5)

    # Apply new-player discount to team strength
    def team_strength_factor(team):
        new_count = team_new_counts.get(team, 0)
        if new_count >= 3:
            return 0.0
        elif new_count == 2:
            return 0.33
        elif new_count == 1:
            return 0.67
        return 1.0

    prices["team_factor"] = prices["Team"].apply(team_strength_factor)

    avg_team_wr = prices["team_wr"].mean()
    avg_maps = 2.3
    # Team strength adjustment, discounted by new player factor
    tpts_adjustment = (prices["team_wr"] - avg_team_wr) * 4.0 / avg_maps * prices["team_factor"]
    adjusted_ppm = prices["ema_ppm"] + tpts_adjustment

    prices["predicted_vp"] = _linear_map(adjusted_ppm.values)
    return prices[["Player", "Team", "predicted_vp"]]


def algo_variance_aware(features, consistency_premium=0.10, **kwargs):
    """Algorithm 5: Variance-aware pricing.
    Consistent players get a premium, volatile players get a discount."""
    prices = features.copy()

    cv = prices["std_ppm"] / (prices["avg_ppm"].abs() + 1e-9)
    cv_norm = cv.rank(pct=True)

    base_vp = _linear_map(prices["ema_ppm"].values)
    adjustment_factor = 1 + consistency_premium * (1 - 2 * cv_norm)
    adjusted_vp = base_vp * adjustment_factor

    prices["predicted_vp"] = np.clip(adjusted_vp, VP_MIN, VP_MAX)
    return prices[["Player", "Team", "predicted_vp"]]


def algo_combined(features, team_win_rates=None, team_new_counts=None,
                  team_popularity=None, opponent_strength=None, **kwargs):
    """Algorithm 6: Combined pricing — uses ALL available signals.

    Blends:
    - EMA performance (base score, 50% weight)
    - Distribution shaping (forces healthy spread)
    - Role adjustment (duelist discount / support boost)
    - Team strength with new-player discount (15% weight)
    - Variance premium (5% weight)
    - Pickrate popularity premium (15% weight — popular players cost more)
    - Team brand popularity adjustment (10% weight)
    - Opponent strength awareness (5% weight)
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

    # --- Component 1: EMA base score (normalized to 0-1) ---
    ema_vals = prices["ema_ppm"].values
    ema_min, ema_max = ema_vals.min(), ema_vals.max()
    if ema_max - ema_min > 1e-9:
        ema_norm = (ema_vals - ema_min) / (ema_max - ema_min)
    else:
        ema_norm = np.full(n, 0.5)

    # --- Component 2: Role adjustment ---
    kill_pctl = prices["kill_profile"].rank(pct=True).values
    ppts_pctl = prices["ppts_ratio"].rank(pct=True).values
    role_score = (kill_pctl + ppts_pctl) / 2
    # Duelists (>0.7): slight discount, supports (<0.3): slight boost
    role_adj = np.where(role_score > 0.7, -0.03, np.where(role_score < 0.3, 0.03, 0.0))

    # --- Component 3: Team strength (with new-player discount) ---
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
    team_strength_norm = (team_wr_vals - avg_wr) * team_factors  # -0.5 to +0.5 range

    # --- Component 4: Consistency premium ---
    cv = prices["std_ppm"].values / (np.abs(prices["avg_ppm"].values) + 1e-9)
    cv_rank = pd.Series(cv).rank(pct=True).values
    consistency_adj = 0.1 * (1 - 2 * cv_rank)  # -0.1 to +0.1

    # --- Component 5: Pickrate popularity ---
    # Higher pickrate = more demand = should cost more (supply/demand pricing)
    pickpct_vals = prices["avg_pickpct"].values
    pickpct_max = pickpct_vals.max() if pickpct_vals.max() > 0 else 1.0
    pickrate_norm = pickpct_vals / pickpct_max  # 0-1

    # --- Component 6: Team brand popularity ---
    team_pop_vals = prices["Team"].map(team_popularity).fillna(0.5).values

    # --- Component 7: Opponent strength ---
    # Players on teams facing weaker opponents get a slight boost
    # (they're more likely to win maps -> more T.Pts)
    # For stage pricing, we don't know exact matchups yet, so use inverse team strength
    # as a proxy (strong teams face strong opponents in competitive groups)
    opp_strength_vals = prices["Team"].map(opponent_strength).fillna(0.5).values
    opp_adj = (0.5 - opp_strength_vals) * 0.1  # facing weaker opponents = slight boost

    # --- Blend all components ---
    # Base performance score (EMA + role adjustment)
    base_score = ema_norm + role_adj

    # Weighted combination — performance-dominant with lighter secondary signals
    # Pickrate and team popularity are "perceived value" adjustments, not price drivers
    combined_score = (
        0.60 * base_score +         # core performance (dominant)
        0.12 * team_strength_norm +  # team win rate impact
        0.05 * consistency_adj +     # consistency premium
        0.08 * pickrate_norm +       # popularity/demand (light touch)
        0.08 * team_pop_vals +       # brand popularity (light touch)
        0.07 * opp_adj               # opponent strength
    )

    # Map to VP range using distribution-aware quantile mapping
    ranks = pd.Series(combined_score).rank(pct=True).values
    predicted = np.interp(ranks, TARGET_QUANTILES, TARGET_VPS)
    prices["predicted_vp"] = np.clip(predicted, VP_MIN, VP_MAX)

    return prices[["Player", "Team", "predicted_vp"]]


# --- Handle Missing Players + Uncertainty ---


def fill_missing_players(predictions, stage1_players, features, team_win_rates=None,
                         team_new_counts=None, default_vp=9.0):
    """Add predictions for players in Stage 1 who aren't in training data.

    For new players:
    - Use team context (team avg VP - 1) as baseline
    - Consider team new-player count for team strength discount
    - Mark as HIGH UNCERTAINTY
    """
    if team_win_rates is None:
        team_win_rates = {}
    if team_new_counts is None:
        team_new_counts = {}

    predicted_players = set(predictions["Player"])
    missing = stage1_players[~stage1_players["Player"].isin(predicted_players)]

    if len(missing) == 0:
        return predictions

    team_avg = predictions.groupby("Team")["predicted_vp"].mean().to_dict()

    new_rows = []
    for _, row in missing.iterrows():
        team = row["Team"]
        team_vp = team_avg.get(team)
        if team_vp is not None:
            vp = max(VP_MIN, team_vp - 1.0)
        else:
            vp = default_vp

        reasons = ["new player (no training data)"]
        new_count = team_new_counts.get(team, 0)
        if new_count >= 2:
            reasons.append(f"team has {new_count} new players")

        new_rows.append({
            "Player": row["Player"],
            "Team": team,
            "predicted_vp": round(vp, 2),
            "uncertainty": "HIGH",
            "uncertainty_reasons": "; ".join(reasons),
        })

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        predictions = pd.concat([predictions, new_df], ignore_index=True)
    return predictions


def add_uncertainty_flags(predictions, features):
    """Add uncertainty flags to all predictions.

    HIGH uncertainty:
    - New players (not in training data)
    - Players with < 3 games in training
    - Players on teams with 3+ new players

    MEDIUM uncertainty:
    - Players with 3-5 games
    - Players who recently changed teams (team in training != team in Stage 1)

    LOW uncertainty:
    - Players with 6+ games on same team
    """
    features_dict = {}
    for _, row in features.iterrows():
        features_dict[row["Player"]] = row

    uncertainties = []
    uncertainty_reasons_list = []

    for _, row in predictions.iterrows():
        player = row["Player"]

        # Already flagged as HIGH (new player from fill_missing_players)
        if "uncertainty" in predictions.columns and row.get("uncertainty") == "HIGH":
            uncertainties.append("HIGH")
            uncertainty_reasons_list.append(row.get("uncertainty_reasons", "new player"))
            continue

        feat = features_dict.get(player)
        if feat is None:
            uncertainties.append("HIGH")
            uncertainty_reasons_list.append("no training data")
            continue

        reasons = list(feat.get("uncertainty_reasons", []))
        games = feat["games_played"]

        # Check if team changed
        if feat["Team"] != row["Team"]:
            reasons.append("team change detected")

        if games < LOW_GAMES_THRESHOLD:
            uncertainties.append("HIGH")
            if not reasons:
                reasons.append(f"only {games} games")
            uncertainty_reasons_list.append("; ".join(reasons))
        elif games <= 5:
            uncertainties.append("MEDIUM")
            reasons.append(f"{games} games (limited sample)")
            uncertainty_reasons_list.append("; ".join(reasons))
        else:
            if reasons:
                uncertainties.append("MEDIUM")
                uncertainty_reasons_list.append("; ".join(reasons))
            else:
                uncertainties.append("LOW")
                uncertainty_reasons_list.append("")

    predictions = predictions.copy()
    predictions["uncertainty"] = uncertainties
    predictions["uncertainty_reasons"] = uncertainty_reasons_list
    return predictions


# --- Main Entry Point ---


def run_all_algorithms():
    """Run all 5 pricing algorithms and return results.

    Returns:
        dict mapping algorithm name -> DataFrame with Player, Team, predicted_vp, uncertainty
        features: player features DataFrame
        validation: actual VP prices DataFrame
        stage1_results: all Stage 1 played rows
    """
    print("Loading data...")
    all_data = load_all_data()
    training = get_training_data(all_data)
    validation = get_validation_prices(all_data)
    stage1_results = get_stage1_results(all_data)
    stage1_players = get_stage1_players(all_data)

    print(f"Training: {len(training)} played games, {training['Player'].nunique()} players")
    print(f"Validation: {len(validation)} players with actual VP prices")

    # Load pickrate data
    pickrate_df = load_pickrate_data()
    if pickrate_df is not None:
        print(f"Pickrate data: {len(pickrate_df)} players with pick history")
    else:
        print("Pickrate data: not found (run parse_pickrates.py first)")
    print()

    print("Computing features...")
    player_pickrates = compute_player_pickrates(pickrate_df)
    features = compute_player_features(training, player_pickrates=player_pickrates)
    team_wr = compute_team_win_rates(training)
    team_new_counts = detect_new_players_on_team(training, stage1_players)
    team_pop = compute_team_popularity(pickrate_df, all_data)
    opp_strength = compute_opponent_strength(training)
    print(f"Features computed for {len(features)} players")

    # Report new player counts per team
    teams_with_new = {t: c for t, c in team_new_counts.items() if c > 0}
    if teams_with_new:
        print(f"Teams with new players: {teams_with_new}")

    # Report top popular teams
    if team_pop:
        top_pop = sorted(team_pop.items(), key=lambda x: -x[1])[:5]
        print(f"Top popular teams: {', '.join(f'{t} ({v:.2f})' for t, v in top_pop)}")
    print()

    algo_kwargs = {
        "team_win_rates": team_wr,
        "team_new_counts": team_new_counts,
        "team_popularity": team_pop,
        "opponent_strength": opp_strength,
    }

    algorithms = {
        "Baseline EMA": algo_baseline_ema(features, **algo_kwargs),
        "Distribution-Aware": algo_distribution_aware(features, **algo_kwargs),
        "Role-Adjusted": algo_role_adjusted(features, **algo_kwargs),
        "Team Strength": algo_team_strength(features, **algo_kwargs),
        "Variance-Aware": algo_variance_aware(features, **algo_kwargs),
        "Combined": algo_combined(features, **algo_kwargs),
    }

    # Fill in missing players and add uncertainty flags
    for name in algorithms:
        algorithms[name] = fill_missing_players(
            algorithms[name], stage1_players, features,
            team_win_rates=team_wr, team_new_counts=team_new_counts,
        )
        algorithms[name] = add_uncertainty_flags(algorithms[name], features)

    return algorithms, features, validation, stage1_results


def main():
    algorithms, features, validation, _ = run_all_algorithms()

    for name, preds in algorithms.items():
        merged = preds.merge(validation, on="Player", how="inner", suffixes=("", "_actual"))
        if len(merged) == 0:
            print(f"\n=== {name} === (no overlap with validation)")
            continue

        mae = (merged["predicted_vp"] - merged["actual_vp"]).abs().mean()
        corr = merged["predicted_vp"].corr(merged["actual_vp"])

        print(f"=== {name} ===")
        print(f"  Players priced: {len(preds)}")
        print(f"  Validated against: {len(merged)} players")
        print(f"  MAE: {mae:.3f}  Correlation: {corr:.3f}")
        print(f"  VP range: {preds['predicted_vp'].min():.1f} - {preds['predicted_vp'].max():.1f}")
        print(f"  VP mean: {preds['predicted_vp'].mean():.2f}  median: {preds['predicted_vp'].median():.2f}")

        # Uncertainty breakdown
        unc = preds["uncertainty"].value_counts()
        print(f"  Uncertainty: {unc.to_dict()}")

        # Show HIGH uncertainty players
        high_unc = preds[preds["uncertainty"] == "HIGH"]
        if len(high_unc) > 0:
            print(f"  HIGH uncertainty players ({len(high_unc)}):")
            for _, row in high_unc.iterrows():
                print(f"    {row['Player']:<18} {row['Team']:<22} {row['predicted_vp']:>6.1f} VP  "
                      f"({row['uncertainty_reasons']})")

        # Distribution buckets
        buckets = pd.cut(preds["predicted_vp"], bins=range(5, 17), right=False)
        print(f"  Distribution: {buckets.value_counts().sort_index().to_dict()}")
        print()


if __name__ == "__main__":
    main()
