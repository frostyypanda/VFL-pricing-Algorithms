"""Expected Points Model for VFL v2.

Every parameter is learned from data:
- Shrinkage (tau², sigma²) via empirical Bayes
- Feature weights via RidgeCV with LOO
- EMA alpha via grid search
- Opponent beta via within-stage regression
- Final prediction via ensemble of all 3 models
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from .constants import NUM_GWS, CHINA_TEAMS
from .schedule import get_playing_teams, get_team_opponent, get_team_region

# ---------------------------------------------------------------------------
#  Empirical Bayes (normal-normal)
# ---------------------------------------------------------------------------


def estimate_eb_params(player_means, player_counts, player_vars):
    """Estimate population-level empirical Bayes parameters.

    Returns (mu, tau2, sigma2):
      mu    = population mean
      tau2  = between-player variance (true skill spread)
      sigma2 = average within-player variance (noise)
    """
    mu = np.mean(player_means)
    sigma2 = np.mean(player_vars)
    raw_var = np.var(player_means)
    # tau2 = between-player var = total var - noise var / avg_n
    avg_n = np.mean(player_counts)
    tau2 = max(raw_var - sigma2 / avg_n, 0.01)
    return mu, tau2, sigma2


def eb_shrink(observed, n, mu, tau2, sigma2):
    """Empirical Bayes shrunk estimate for one player."""
    if n == 0:
        return mu
    B = tau2 / (tau2 + sigma2 / n)
    return B * observed + (1 - B) * mu


def compute_eb_estimates(played_df, group_col=None):
    """Compute EB-shrunk avg Pts for every player.

    If group_col provided (e.g. role), shrink toward group mean.
    Returns dict player -> shrunk_pts.
    """
    player_stats = {}
    for player, grp in played_df.groupby("Player"):
        pts = grp["Pts"].astype(float).values
        player_stats[player] = {
            "mean": np.mean(pts),
            "var": np.var(pts, ddof=1) if len(pts) > 1 else 9.0,
            "n": len(pts),
        }

    means = np.array([s["mean"] for s in player_stats.values()])
    counts = np.array([s["n"] for s in player_stats.values()])
    variances = np.array([s["var"] for s in player_stats.values()])

    mu, tau2, sigma2 = estimate_eb_params(means, counts, variances)

    result = {}
    for player, s in player_stats.items():
        result[player] = eb_shrink(s["mean"], s["n"], mu, tau2, sigma2)
    return result, {"mu": mu, "tau2": tau2, "sigma2": sigma2}


# ---------------------------------------------------------------------------
#  Feature construction for RidgeCV
# ---------------------------------------------------------------------------


def _recency_stage_idx(stage, year):
    """Map (stage, year) to a global ordering for recency."""
    stage_map = {
        "Kickoff": 0, "bangkok": 1, "Madrid": 1, "Santiago": 1,
        "Stage 1": 2, "Toronto": 3, "Shanghai": 3, "London": 3,
        "Stage 2": 4, "Champions": 5,
    }
    return year * 10 + stage_map.get(stage, 2)


def build_features(train_df, target_players):
    """Build feature matrix for RidgeCV.

    For each player in target_players, compute features from train_df.
    Returns (X, player_list) where X is a numpy array.
    """
    # Split train into "recent" (most recent stage) and "old" (everything else)
    stage_indices = train_df.apply(
        lambda r: _recency_stage_idx(r["Stage"], r["Year"]), axis=1
    )
    max_stage = stage_indices.max()
    recent_mask = stage_indices == max_stage
    old_mask = ~recent_mask

    recent = train_df[recent_mask]
    old = train_df[old_mask]

    # Team win rates from training data
    team_wr = {}
    for team, grp in train_df.groupby("Team"):
        wins = (grp["W/L"] == "W").sum()
        team_wr[team] = wins / max(len(grp), 1)

    features = []
    players_out = []

    for player in target_players:
        r_grp = recent[recent["Player"] == player]
        o_grp = old[old["Player"] == player]
        all_grp = train_df[train_df["Player"] == player]

        avg_recent = r_grp["Pts"].mean() if len(r_grp) > 0 else np.nan
        avg_old = o_grp["Pts"].mean() if len(o_grp) > 0 else np.nan
        avg_all = all_grp["Pts"].mean() if len(all_grp) > 0 else np.nan
        n_recent = len(r_grp)
        n_total = len(all_grp)

        team = all_grp["Team"].iloc[-1] if len(all_grp) > 0 else None
        wr = team_wr.get(team, 0.5) if team else 0.5

        # Fill NaN with population mean (will be handled by Ridge)
        pop_mean = train_df["Pts"].mean()
        avg_recent = avg_recent if not np.isnan(avg_recent) else pop_mean
        avg_old = avg_old if not np.isnan(avg_old) else pop_mean
        avg_all = avg_all if not np.isnan(avg_all) else pop_mean

        features.append([
            avg_recent,
            avg_old,
            avg_all,
            n_recent,
            n_total,
            wr,
        ])
        players_out.append(player)

    return np.array(features), players_out


# ---------------------------------------------------------------------------
#  EMA with grid-searched alpha
# ---------------------------------------------------------------------------


def compute_ema_estimates(played_df, alpha):
    """Compute EMA of Pts for each player with given alpha.

    Games are ordered by (Year, Stage) recency.
    Returns dict player -> ema_pts.
    """
    result = {}
    for player, grp in played_df.groupby("Player"):
        # Sort by recency
        pts_list = grp.sort_values(
            ["Year", "Stage"], key=lambda s: s.map(
                lambda v: _recency_stage_idx(v, 0) if isinstance(v, str)
                else v * 10
            )
        )["Pts"].astype(float).values

        ema = pts_list[0]
        for p in pts_list[1:]:
            ema = alpha * p + (1 - alpha) * ema
        result[player] = ema
    return result


def grid_search_ema(train_df, actual_df):
    """Find best EMA alpha by testing on actual data."""
    actual_avg = actual_df.groupby("Player")["Pts"].mean()
    best_alpha, best_corr = 0.3, -1

    for alpha in np.arange(0.05, 0.96, 0.05):
        preds = compute_ema_estimates(train_df, alpha)
        common = set(preds.keys()) & set(actual_avg.index)
        if len(common) < 20:
            continue
        pred_vals = [preds[p] for p in common]
        act_vals = [actual_avg[p] for p in common]
        corr = np.corrcoef(pred_vals, act_vals)[0, 1]
        if corr > best_corr:
            best_corr = corr
            best_alpha = alpha

    return best_alpha, best_corr


# ---------------------------------------------------------------------------
#  Opponent adjustment
# ---------------------------------------------------------------------------


def learn_opponent_beta(played_df):
    """Learn opponent adjustment coefficient from within-stage data.

    Regresses player Pts on opponent team win rate within each stage.
    Returns beta (expected range 0.2-0.5).
    """
    # Need to know who played whom - approximate from team W/L same stage
    betas = []
    for (year, stage), stage_df in played_df.groupby(["Year", "Stage"]):
        if len(stage_df) < 50:
            continue
        team_wr = {}
        for team, grp in stage_df.groupby("Team"):
            wins = (grp["W/L"] == "W").sum()
            team_wr[team] = wins / max(len(grp), 1)

        # For each game, the "opponent strength" is the other team's WR
        # We don't have explicit opponent per game, so use team WR as proxy
        # Players on winning teams score more (F7: +5.3 pts)
        pts = stage_df["Pts"].astype(float).values
        own_wr = stage_df["Team"].map(team_wr).values
        if np.std(own_wr) < 0.01:
            continue
        # Simple: correlation between team WR and Pts
        corr = np.corrcoef(own_wr, pts)[0, 1]
        betas.append(corr)

    if not betas:
        return 0.3
    return float(np.median(betas))


# ---------------------------------------------------------------------------
#  Calibration: run walk-forward to find best parameters
# ---------------------------------------------------------------------------


def calibrate(all_data):
    """Run walk-forward calibration across available windows.

    Returns calibrated parameters:
    - eb_params (mu, tau2, sigma2)
    - ridge_model (fitted RidgeCV)
    - best_ema_alpha
    - opponent_beta
    - window_results (for reporting)
    """
    from .data_loader import split_walk_forward

    windows = [
        ("Stage 1", 2025),
        ("Stage 2", 2025),
    ]

    all_ridge_X, all_ridge_y = [], []
    all_ema_results = {}
    window_results = []

    for target_stage, target_year in windows:
        train, actual = split_walk_forward(all_data, target_stage, target_year)
        if len(train) < 50 or len(actual) < 30:
            continue

        actual_avg = actual.groupby("Player")["Pts"].mean()
        common_players = list(
            set(train["Player"].unique()) & set(actual_avg.index)
        )
        if len(common_players) < 30:
            continue

        # Features for Ridge
        X, players = build_features(train, common_players)
        y = np.array([actual_avg[p] for p in players])
        all_ridge_X.append(X)
        all_ridge_y.append(y)

        # EMA grid search
        alpha, corr = grid_search_ema(train, actual)
        all_ema_results[(target_stage, target_year)] = (alpha, corr)

        window_results.append({
            "window": f"{target_year} {target_stage}",
            "n_players": len(common_players),
            "best_ema_alpha": alpha,
            "ema_corr": corr,
        })

    # Fit Ridge on pooled windows
    X_all = np.vstack(all_ridge_X)
    y_all = np.concatenate(all_ridge_y)
    ridge = RidgeCV(
        alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=None  # LOO
    )
    ridge.fit(X_all, y_all)

    # EB params from all played data
    played = all_data[all_data["P?"] == 1]
    _, eb_params = compute_eb_estimates(played)

    # Best EMA alpha = median across windows
    ema_alphas = [v[0] for v in all_ema_results.values()]
    best_alpha = float(np.median(ema_alphas)) if ema_alphas else 0.3

    # Opponent beta
    opp_beta = learn_opponent_beta(played)

    for wr in window_results:
        wr["ridge_alpha"] = float(ridge.alpha_)
        wr["ridge_r2"] = float(ridge.score(X_all, y_all))

    return {
        "eb_params": eb_params,
        "ridge_model": ridge,
        "best_ema_alpha": best_alpha,
        "opponent_beta": opp_beta,
        "window_results": window_results,
        "ridge_coefs": dict(zip(
            ["avg_recent", "avg_old", "avg_all", "n_recent", "n_total", "team_wr"],
            ridge.coef_,
        )),
        "ridge_intercept": float(ridge.intercept_),
    }


# ---------------------------------------------------------------------------
#  Main: compute expected points matrix
# ---------------------------------------------------------------------------


def compute_expected_pts(train_df, roster, calibration, schedule_gws=True):
    """Compute expected Pts per player per GW.

    Args:
        train_df: historical played data (P?=1, no China)
        roster: dict player -> {team, region, role}
        calibration: output of calibrate()
        schedule_gws: if True, apply schedule (0 for non-playing GWs)

    Returns DataFrame with Player, Team, Region, Role, GW1-GW6, SeasonValue
    """
    eb_params = calibration["eb_params"]
    ridge = calibration["ridge_model"]
    alpha = calibration["best_ema_alpha"]
    opp_beta = calibration["opponent_beta"]

    # Model A: Empirical Bayes
    eb_estimates, _ = compute_eb_estimates(train_df)

    # Model B: RidgeCV
    target_players = list(roster.keys())
    X, players_ordered = build_features(train_df, target_players)
    ridge_preds = dict(zip(players_ordered, ridge.predict(X)))

    # Model C: EMA
    ema_estimates = compute_ema_estimates(train_df, alpha)

    # Team win rates for opponent adjustment
    team_wr = {}
    for team, grp in train_df.groupby("Team"):
        wins = (grp["W/L"] == "W").sum()
        team_wr[team] = wins / max(len(grp), 1)

    # Population mean for fallback
    pop_mean = train_df["Pts"].mean()

    rows = []
    for player, info in roster.items():
        team = info["team"]
        region = info["region"]
        role = info["role"]

        # Ensemble: average of available models
        estimates = []
        if player in eb_estimates:
            estimates.append(eb_estimates[player])
        if player in ridge_preds:
            estimates.append(ridge_preds[player])
        if player in ema_estimates:
            estimates.append(ema_estimates[player])

        if estimates:
            base_pts = float(np.mean(estimates))
        else:
            # New player: role-based prior
            role_priors = {"D": 9.0, "I": 7.0, "C": 7.0, "S": 7.5}
            base_pts = role_priors.get(role, pop_mean) * 0.85

        # Clamp to reasonable range
        base_pts = max(1.0, min(25.0, base_pts))

        n_games = len(train_df[train_df["Player"] == player])

        row = {
            "Player": player, "Team": team, "Region": region,
            "Role": role, "N_Games": n_games, "BasePts": round(base_pts, 2),
        }

        for gw in range(1, NUM_GWS + 1):
            if not schedule_gws:
                row[f"GW{gw}"] = round(base_pts, 2)
                continue

            playing = get_playing_teams(gw)
            playing_norm = {_norm(t): t for t in playing}
            team_norm = _norm(team)

            if team_norm not in playing_norm:
                row[f"GW{gw}"] = 0.0
                continue

            # Opponent adjustment
            actual_team = playing_norm[team_norm]
            opp = get_team_opponent(actual_team, gw)
            opp_factor = 1.0
            if opp:
                opp_wr = team_wr.get(opp, 0.5)
                # Stronger opponent -> fewer points
                opp_factor = 1.0 + opp_beta * (0.5 - opp_wr)
                opp_factor = max(0.8, min(1.2, opp_factor))

            row[f"GW{gw}"] = round(base_pts * opp_factor, 2)

        rows.append(row)

    df = pd.DataFrame(rows)
    gw_cols = [f"GW{i}" for i in range(1, NUM_GWS + 1)]
    df["SeasonValue"] = df[gw_cols].sum(axis=1).round(2)
    return df


def _norm(name):
    if not isinstance(name, str):
        return ""
    return name.strip().lower().replace("é", "e").replace("ü", "u")
