"""
VFL Pricing Algorithms — 5 approaches to player pricing for Valorant Fantasy League.

Training data: All 2024 + 2025 Kickoff + 2025 Bangkok (Masters 1)
Validation:    2025 Stage 1 (predict prices, simulate teams, backtest)

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
TARGET_QUANTILES = np.array([0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
                              0.60, 0.70, 0.80, 0.90, 0.95, 1.0])
TARGET_VPS =      np.array([6.0, 6.3,  6.5,  7.2,  8.0,  8.6,  9.0,
                              9.5, 10.2, 11.0, 12.5, 13.5, 15.0])

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
    # Filter out China teams
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


# --- Feature Engineering ---


def compute_ema(values, alpha=0.3):
    """Compute exponential moving average over a sequence of values.
    More recent values (end of array) get higher weight."""
    if len(values) == 0:
        return 0.0
    weights = np.array([(1 - alpha) ** i for i in range(len(values))])[::-1]
    # Most recent value gets weight alpha, previous gets alpha*(1-alpha), etc.
    # Actually: standard EMA from oldest to newest
    weights = np.array([(1 - alpha) ** (len(values) - 1 - i) for i in range(len(values))])
    return np.average(values, weights=weights)


def shrink_estimate(player_val, n_games, pop_mean, shrinkage_games=4):
    """Bayesian shrinkage toward population mean for small samples."""
    weight = n_games / (n_games + shrinkage_games)
    return weight * player_val + (1 - weight) * pop_mean


def compute_player_features(training_df):
    """Compute per-player features from training data.

    Returns DataFrame indexed by Player with feature columns.
    """
    # Sort chronologically
    df = training_df.copy()
    df["_stage_ord"] = df["Stage"].map(STAGE_ORDER).fillna(99)
    df["_game_ord"] = df["Game"].map(GAME_ORDER).fillna(99)
    df = df.sort_values(["Year", "_stage_ord", "Wk", "_game_ord"])

    pop_mean_ppm = df["PPM"].mean()

    records = []
    for player, grp in df.groupby("Player"):
        team = grp["Team"].iloc[-1]  # most recent team
        games = len(grp)
        ppm_vals = grp["PPM"].values
        pts_vals = grp["Pts"].values
        tpts_vals = grp["T.Pts"].values
        ppts_vals = grp["P.Pts"].values
        year_vals = grp["Year"].values

        # Apply year discount: 2024 games get 0.5 weight
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

        # Basic stats
        avg_ppm = np.mean(ppm_vals)
        std_ppm = np.std(ppm_vals) if games > 1 else 0.0
        avg_pts = np.mean(pts_vals)
        avg_tpts = np.mean(tpts_vals)
        avg_ppts = np.mean(ppts_vals)

        # Win rate
        wl = grp["W/L"].values
        wins = np.sum(wl == "W")
        win_rate = wins / games if games > 0 else 0.5

        # Kill profile: fraction of maps in 20k+ brackets
        kill_cols = ["20k", "25k", "30k", "35k", "40k", "45k", "50k"]
        total_maps = grp[["0k", "5k", "10k", "15k", "20k", "25k", "30k",
                          "35k", "40k", "45k", "50k"]].sum().sum()
        high_kills = grp[kill_cols].sum().sum()
        kill_profile = high_kills / total_maps if total_maps > 0 else 0.0

        # Rating bonus rate
        rating_bonuses = grp[["TOP3", "TOP2", "TOP1", "1.5R2", "1.75R2", "2.0R2"]].sum().sum()
        rating_bonus_rate = rating_bonuses / games if games > 0 else 0.0

        # Personal-to-total ratio (proxy for role)
        ppts_ratio = avg_ppts / (avg_pts + 1e-9)

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


def algo_baseline_ema(features):
    """Algorithm 1: Baseline EMA pricing.
    Linear map of EMA PPM to [6, 15] VP."""
    prices = features.copy()
    prices["predicted_vp"] = _linear_map(prices["ema_ppm"].values)
    return prices[["Player", "Team", "predicted_vp"]]


def algo_distribution_aware(features):
    """Algorithm 2: Distribution-aware pricing.
    Rank players by EMA PPM, then map percentiles to target distribution."""
    prices = features.copy()
    # Percentile rank (0 to 1)
    ranks = prices["ema_ppm"].rank(pct=True).values
    # Interpolate target VP from quantile breakpoints
    predicted = np.interp(ranks, TARGET_QUANTILES, TARGET_VPS)
    prices["predicted_vp"] = np.clip(predicted, VP_MIN, VP_MAX)
    return prices[["Player", "Team", "predicted_vp"]]


def algo_role_adjusted(features, role_data=None):
    """Algorithm 3: Role-adjusted pricing.
    Proxy roles from scoring patterns, adjust PPM before pricing."""
    prices = features.copy()

    if role_data is not None:
        # TODO: use actual role data when available
        pass

    # Proxy role classification based on scoring patterns
    # Duelist-like: high kill profile + high ppts ratio
    # Support-like: low kill profile + low ppts ratio
    kill_pctl = prices["kill_profile"].rank(pct=True)
    ppts_pctl = prices["ppts_ratio"].rank(pct=True)
    role_score = (kill_pctl + ppts_pctl) / 2  # 0=most support-like, 1=most duelist-like

    # Adjustment: duelists get discounted (they score high naturally),
    # supports get boosted (their PPM understates their value)
    # Duelist (role_score > 0.7): -7% PPM adjustment
    # Support (role_score < 0.3): +5% PPM adjustment
    # Middle: no adjustment
    adjustment = np.where(
        role_score > 0.7, 0.93,
        np.where(role_score < 0.3, 1.05, 1.0)
    )
    adjusted_ppm = prices["ema_ppm"] * adjustment
    prices["predicted_vp"] = _linear_map(adjusted_ppm.values)
    return prices[["Player", "Team", "predicted_vp"]]


def algo_team_strength(features, team_win_rates):
    """Algorithm 4: Team strength multiplier.
    Decompose into personal + team components, predict each separately."""
    prices = features.copy()

    # Average T.Pts per game by team win rate (from training data analysis):
    # ~30% WR teams: ~1.0 T.Pts/game, ~50%: ~2.5, ~70%: ~3.5
    # Linear approximation: expected_tpts = 1.0 + 4.0 * team_wr
    prices["team_wr"] = prices["Team"].map(team_win_rates).fillna(0.5)
    expected_tpts_per_game = 1.0 + 4.0 * prices["team_wr"]

    # Personal component from EMA (subtract average T.Pts contribution)
    pop_avg_tpts = features["avg_tpts"].mean()
    personal_ema = prices["ema_ppm"]  # PPM includes both components

    # Estimate maps per game from training data (typically 2-2.5)
    avg_maps = 2.3

    # Recompose: adjust the team component based on actual team strength
    # If a player's team is stronger/weaker than average, shift their expected PPM
    avg_team_wr = prices["team_wr"].mean()
    tpts_adjustment = (prices["team_wr"] - avg_team_wr) * 4.0 / avg_maps
    adjusted_ppm = personal_ema + tpts_adjustment

    prices["predicted_vp"] = _linear_map(adjusted_ppm.values)
    return prices[["Player", "Team", "predicted_vp"]]


def algo_variance_aware(features, consistency_premium=0.10):
    """Algorithm 5: Variance-aware pricing.
    Consistent players get a premium, volatile players get a discount."""
    prices = features.copy()

    # Coefficient of variation
    cv = prices["std_ppm"] / (prices["avg_ppm"].abs() + 1e-9)
    # Normalize CV to [0, 1] range for adjustment
    cv_norm = cv.rank(pct=True)  # 0=most consistent, 1=most volatile

    # Start from baseline EMA price
    base_vp = _linear_map(prices["ema_ppm"].values)

    # Adjust: consistent (low CV) gets premium, volatile (high CV) gets discount
    # Max adjustment: +/- consistency_premium * VP
    adjustment_factor = 1 + consistency_premium * (1 - 2 * cv_norm)
    adjusted_vp = base_vp * adjustment_factor

    prices["predicted_vp"] = np.clip(adjusted_vp, VP_MIN, VP_MAX)
    return prices[["Player", "Team", "predicted_vp"]]


# --- Handle Missing Players ---


def fill_missing_players(predictions, stage1_players, features, default_vp=9.0):
    """Add predictions for players in Stage 1 who aren't in training data.

    Strategy: use team average minus 1 VP, or default_vp if no teammates.
    """
    predicted_players = set(predictions["Player"])
    missing = stage1_players[~stage1_players["Player"].isin(predicted_players)]

    if len(missing) == 0:
        return predictions

    # Team average VP from predictions
    team_avg = predictions.groupby("Team")["predicted_vp"].mean().to_dict()

    new_rows = []
    for _, row in missing.iterrows():
        team_vp = team_avg.get(row["Team"])
        if team_vp is not None:
            vp = max(VP_MIN, team_vp - 1.0)
        else:
            vp = default_vp
        new_rows.append({
            "Player": row["Player"],
            "Team": row["Team"],
            "predicted_vp": round(vp, 2),
        })

    if new_rows:
        predictions = pd.concat([predictions, pd.DataFrame(new_rows)], ignore_index=True)
    return predictions


# --- Main Entry Point ---


def run_all_algorithms():
    """Run all 5 pricing algorithms and return results.

    Returns:
        dict mapping algorithm name -> DataFrame with Player, Team, predicted_vp
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
    print()

    print("Computing features...")
    features = compute_player_features(training)
    team_wr = compute_team_win_rates(training)
    print(f"Features computed for {len(features)} players")
    print()

    # Run all algorithms
    algorithms = {
        "Baseline EMA": algo_baseline_ema(features),
        "Distribution-Aware": algo_distribution_aware(features),
        "Role-Adjusted": algo_role_adjusted(features),
        "Team Strength": algo_team_strength(features, team_wr),
        "Variance-Aware": algo_variance_aware(features),
    }

    # Fill in missing players for each algorithm
    for name in algorithms:
        algorithms[name] = fill_missing_players(
            algorithms[name], stage1_players, features
        )

    return algorithms, features, validation, stage1_results


def main():
    algorithms, features, validation, _ = run_all_algorithms()

    for name, preds in algorithms.items():
        # Merge with validation to compute quick accuracy
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

        # Distribution buckets
        buckets = pd.cut(preds["predicted_vp"], bins=range(5, 17), right=False)
        print(f"  Distribution: {buckets.value_counts().sort_index().to_dict()}")
        print()


if __name__ == "__main__":
    main()
