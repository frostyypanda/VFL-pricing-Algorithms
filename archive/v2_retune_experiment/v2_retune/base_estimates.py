"""Compute v2 base estimates (EB, Ridge, EMA) parameterized by cal_windows + recency_map.

We avoid modifying v2 source. We re-implement calibration with overridable params,
calling into v2's internals where safe.
"""
import numpy as np
from sklearn.linear_model import RidgeCV

from v2.expected_points import (
    compute_eb_estimates, compute_ema_estimates,
    learn_opponent_beta,
)
from v2.data_loader import split_walk_forward


def calibrate(all_data, cal_windows, recency_map):
    """v2-style calibration with overridable cal windows and recency map.

    Returns dict with eb_params, ridge_model, best_ema_alpha, opp_beta.
    """
    X_all, y_all = _pool_ridge_data(all_data, cal_windows, recency_map)
    ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    ridge.fit(X_all, y_all)
    alpha = _grid_search_ema_pooled(all_data, cal_windows, recency_map)
    played = all_data[all_data["P?"] == 1]
    _, eb_params = compute_eb_estimates(played)
    opp_beta = learn_opponent_beta(played)
    return {
        "ridge_model": ridge,
        "best_ema_alpha": float(alpha),
        "eb_params": eb_params,
        "opponent_beta": opp_beta,
    }


def _pool_ridge_data(all_data, cal_windows, recency_map):
    X_list, y_list = [], []
    for stage, year in cal_windows:
        train, actual = split_walk_forward(all_data, stage, year)
        if len(train) < 50 or len(actual) < 30:
            continue
        actual_avg = actual.groupby("Player")["Pts"].mean()
        common = list(set(train["Player"].unique()) & set(actual_avg.index))
        if len(common) < 30:
            continue
        X, players = _build_features(train, common, recency_map)
        y = np.array([actual_avg[p] for p in players])
        X_list.append(X)
        y_list.append(y)
    return np.vstack(X_list), np.concatenate(y_list)


def _grid_search_ema_pooled(all_data, cal_windows, recency_map):
    best_alpha, best_corr = 0.3, -1.0
    for alpha in np.arange(0.05, 0.96, 0.05):
        corrs = []
        for stage, year in cal_windows:
            train, actual = split_walk_forward(all_data, stage, year)
            if len(train) < 50 or len(actual) < 30:
                continue
            preds = compute_ema_estimates(train, alpha)
            actual_avg = actual.groupby("Player")["Pts"].mean()
            common = set(preds) & set(actual_avg.index)
            if len(common) < 20:
                continue
            p_vals = [preds[p] for p in common]
            a_vals = [actual_avg[p] for p in common]
            c = np.corrcoef(p_vals, a_vals)[0, 1]
            corrs.append(c)
        if corrs and np.mean(corrs) > best_corr:
            best_corr = float(np.mean(corrs))
            best_alpha = float(alpha)
    return best_alpha


def _build_features(train_df, target_players, recency_map):
    """v2 feature builder with overridable recency map."""
    stage_indices = train_df.apply(
        lambda r: r["Year"] * 10 + recency_map.get(r["Stage"], 2), axis=1)
    max_stage = stage_indices.max()
    recent = train_df[stage_indices == max_stage]
    old = train_df[stage_indices != max_stage]
    team_wr = _team_wr_map(train_df)
    pop_mean = train_df["Pts"].mean()
    features, players_out = [], []
    for player in target_players:
        feat = _player_features(
            player, train_df, recent, old, team_wr, pop_mean)
        features.append(feat)
        players_out.append(player)
    return np.array(features), players_out


def _player_features(player, train_df, recent, old, team_wr, pop_mean):
    r_grp = recent[recent["Player"] == player]
    o_grp = old[old["Player"] == player]
    a_grp = train_df[train_df["Player"] == player]
    avg_recent = float(r_grp["Pts"].mean()) if len(r_grp) > 0 else pop_mean
    avg_old = float(o_grp["Pts"].mean()) if len(o_grp) > 0 else pop_mean
    avg_all = float(a_grp["Pts"].mean()) if len(a_grp) > 0 else pop_mean
    team = a_grp["Team"].iloc[-1] if len(a_grp) > 0 else None
    wr = team_wr.get(team, 0.5) if team else 0.5
    return [avg_recent, avg_old, avg_all, len(r_grp), len(a_grp), wr]


def _team_wr_map(train_df):
    out = {}
    for team, grp in train_df.groupby("Team"):
        wins = float((grp["W/L"] == "W").sum())
        out[team] = wins / max(len(grp), 1)
    return out


def compute_base_estimates(train_df, target_players, calibration, recency_map):
    """Per-player (eb, ridge, ema) estimates for the target stage."""
    eb_estimates, _ = compute_eb_estimates(train_df)
    X, players_ord = _build_features(train_df, target_players, recency_map)
    ridge_preds = dict(zip(players_ord, calibration["ridge_model"].predict(X)))
    ema_estimates = compute_ema_estimates(train_df, calibration["best_ema_alpha"])
    out = {}
    pop_mean = float(train_df["Pts"].mean())
    for player in target_players:
        out[player] = {
            "eb": float(eb_estimates.get(player, pop_mean)),
            "ridge": float(ridge_preds.get(player, pop_mean)),
            "ema": float(ema_estimates.get(player, pop_mean)),
        }
    return out
