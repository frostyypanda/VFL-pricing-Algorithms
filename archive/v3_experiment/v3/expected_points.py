"""v3 expected-points: extends v2 ensemble with role-EB, team-form, continuity."""
import numpy as np
import pandas as pd

from v2.expected_points import (
    build_features, compute_ema_estimates, _norm,
    compute_eb_estimates,
)
from v2.constants import NUM_GWS
from v2.schedule import get_playing_teams, get_team_opponent

from .eb_role import compute_v3_eb_estimates
from .team_form import compute_team_form_ratio
from .continuity import compute_team_continuity, continuity_adjusted_estimate
from .flags import V3Flags


def compute_v3_expected_pts(train_df, roster, calibration, schedule_gws=True,
                             flags=None):
    """v3 expected points. Returns DataFrame with Player, Team, Region, Role,
    N_Games, BasePts, GW1..GW6, SeasonValue, plus v3-specific cols.
    """
    flags = flags or V3Flags()
    eb_estimates, role_means, global_mu = _compute_eb(train_df, roster, flags)
    ridge = calibration["ridge_model"]
    alpha = calibration["best_ema_alpha"]
    opp_beta = calibration["opponent_beta"]

    target_players = list(roster.keys())
    X, players_ordered = build_features(train_df, target_players)
    ridge_preds = dict(zip(players_ordered, ridge.predict(X)))
    ema_estimates = compute_ema_estimates(train_df, alpha)

    team_form_ratio = (compute_team_form_ratio(train_df)
                       if flags.team_form_decay else {})
    continuity = (compute_team_continuity(train_df, roster)
                  if flags.continuity else {p: 1.0 for p in roster})
    rows = _build_rows(
        roster, eb_estimates, ridge_preds, ema_estimates,
        role_means, global_mu, continuity, team_form_ratio,
        opp_beta, train_df, schedule_gws,
    )
    df = pd.DataFrame(rows)
    gw_cols = [f"GW{i}" for i in range(1, NUM_GWS + 1)]
    df["SeasonValue"] = df[gw_cols].sum(axis=1).round(2)
    return df


def _compute_eb(train_df, roster, flags):
    """Return (estimates, role_means, global_mu) honoring role/B-floor flags."""
    if flags.role_mean_eb:
        eb_estimates, debug = compute_v3_eb_estimates(train_df, roster)
        return eb_estimates, debug["role_means"], debug["mu"]
    if flags.b_floor:
        return _eb_population_with_floor(train_df)
    estimates, params = compute_eb_estimates(train_df)
    return estimates, {}, params["mu"]


def _eb_population_with_floor(train_df):
    """v2 EB with B-floor only (no role target)."""
    from .eb_role import MIN_B, estimate_eb_population
    stats = {}
    for player, grp in train_df.groupby("Player"):
        pts = grp["Pts"].astype(float).values
        stats[player] = {
            "mean": float(np.mean(pts)),
            "var": float(np.var(pts, ddof=1)) if len(pts) > 1 else 9.0,
            "n": int(len(pts)),
        }
    means = np.array([s["mean"] for s in stats.values()])
    counts = np.array([s["n"] for s in stats.values()])
    vars_ = np.array([s["var"] for s in stats.values()])
    mu, tau2, sigma2 = estimate_eb_population(means, counts, vars_)
    out = {}
    for player, s in stats.items():
        if s["n"] == 0:
            out[player] = mu
            continue
        raw_B = tau2 / (tau2 + sigma2 / s["n"])
        B = max(raw_B, MIN_B)
        out[player] = B * s["mean"] + (1 - B) * mu
    return out, {}, mu


def _build_rows(roster, eb, ridge, ema, role_means, global_mu,
                continuity, team_form, opp_beta, train_df, schedule_gws):
    pop_mean = float(train_df["Pts"].mean()) if len(train_df) else 7.5
    rows = []
    for player, info in roster.items():
        rows.append(_row_for_player(
            player, info, eb, ridge, ema, role_means, global_mu,
            continuity, team_form, opp_beta, train_df, pop_mean, schedule_gws,
        ))
    return rows


def _row_for_player(player, info, eb, ridge, ema, role_means, global_mu,
                    continuity, team_form, opp_beta, train_df, pop_mean,
                    schedule_gws):
    role = info["role"]
    role_mu = role_means.get(role, global_mu)
    base = _ensemble_base(player, eb, ridge, ema, role_mu)
    cont = continuity.get(player, 1.0)
    if continuity:
        adjusted = continuity_adjusted_estimate(base, role_mu, cont)
    else:
        adjusted = base
    team_factor = team_form.get(info["team"], 1.0) if team_form else 1.0
    final_base = max(1.0, min(25.0, adjusted * team_factor))
    n_games = int(len(train_df[train_df["Player"] == player]))
    row = _base_row(player, info, n_games, final_base, cont, team_factor)
    _fill_gw_pts(row, info["team"], final_base, opp_beta, train_df, schedule_gws)
    return row


def _ensemble_base(player, eb, ridge, ema, role_mu):
    estimates = [v for v in (eb.get(player), ridge.get(player), ema.get(player))
                 if v is not None]
    if not estimates:
        return role_mu
    return float(np.mean(estimates))


def _base_row(player, info, n_games, base, continuity, team_factor):
    return {
        "Player": player, "Team": info["team"], "Region": info["region"],
        "Role": info["role"], "N_Games": n_games,
        "BasePts": round(base, 2),
        "Continuity": round(continuity, 2),
        "TeamForm": round(team_factor, 2),
    }


def _fill_gw_pts(row, team, base, opp_beta, train_df, schedule_gws):
    team_wr = _team_wr_map(train_df)
    for gw in range(1, NUM_GWS + 1):
        row[f"GW{gw}"] = _gw_value(gw, team, base, opp_beta, team_wr, schedule_gws)


def _team_wr_map(train_df):
    out = {}
    for team, grp in train_df.groupby("Team"):
        wins = float((grp["W/L"] == "W").sum())
        out[team] = wins / max(len(grp), 1)
    return out


def _gw_value(gw, team, base, opp_beta, team_wr, schedule_gws):
    if not schedule_gws:
        return round(base, 2)
    playing = get_playing_teams(gw)
    playing_norm = {_norm(t): t for t in playing}
    team_norm = _norm(team)
    if team_norm not in playing_norm:
        return 0.0
    actual_team = playing_norm[team_norm]
    opp = get_team_opponent(actual_team, gw)
    opp_factor = 1.0
    if opp:
        opp_wr = team_wr.get(opp, 0.5)
        opp_factor = 1.0 + opp_beta * (0.5 - opp_wr)
        opp_factor = max(0.8, min(1.2, opp_factor))
    return round(base * opp_factor, 2)
