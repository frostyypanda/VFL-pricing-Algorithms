"""v3 Empirical Bayes — role-mean shrinkage target + B-floor (spec §5.1)."""
import numpy as np

MIN_B = 0.30
MIN_ROLE_PLAYERS = 5


def role_means(played_df, roster):
    """Pts mean per role across players present in roster.

    Returns {role: mean_pts}. Roles with <MIN_ROLE_PLAYERS distinct players fall back to global mean.
    """
    global_mean = float(played_df["Pts"].mean())
    out = {}
    by_player_role = _player_role_map(roster)
    player_means = played_df.groupby("Player")["Pts"].mean()
    by_role = _group_means_by_role(player_means, by_player_role)
    for role, vals in by_role.items():
        if len(vals) < MIN_ROLE_PLAYERS:
            out[role] = global_mean
        else:
            out[role] = float(np.mean(vals))
    return out, global_mean


def _player_role_map(roster):
    return {p: info["role"] for p, info in roster.items()}


def _group_means_by_role(player_means, by_player_role):
    by_role = {}
    for player, mean in player_means.items():
        role = by_player_role.get(player)
        if role is None:
            continue
        by_role.setdefault(role, []).append(float(mean))
    return by_role


def estimate_eb_population(player_means, player_counts, player_vars):
    """Same population EB params as v2 (mu, tau2, sigma2)."""
    mu = float(np.mean(player_means))
    sigma2 = float(np.mean(player_vars))
    raw_var = float(np.var(player_means))
    avg_n = float(np.mean(player_counts))
    tau2 = max(raw_var - sigma2 / avg_n, 0.01)
    return mu, tau2, sigma2


def eb_shrink_role(observed, n, role_mu, tau2, sigma2):
    """EB shrinkage toward role mean with B floored at MIN_B."""
    if n == 0:
        return role_mu
    raw_B = tau2 / (tau2 + sigma2 / n)
    B = max(raw_B, MIN_B)
    return B * observed + (1 - B) * role_mu


def compute_v3_eb_estimates(played_df, roster):
    """Per-player EB-shrunk Pts using role-mean target and B-floor.

    Returns (estimates_dict, debug_info_dict).
    """
    stats = _per_player_stats(played_df)
    means = np.array([s["mean"] for s in stats.values()])
    counts = np.array([s["n"] for s in stats.values()])
    vars_ = np.array([s["var"] for s in stats.values()])
    mu, tau2, sigma2 = estimate_eb_population(means, counts, vars_)
    role_mu_map, global_mu = role_means(played_df, roster)
    role_lookup = {p: info["role"] for p, info in roster.items()}
    result = _shrink_each(stats, role_lookup, role_mu_map, global_mu, tau2, sigma2)
    debug = {"mu": mu, "tau2": tau2, "sigma2": sigma2, "role_means": role_mu_map}
    return result, debug


def _per_player_stats(played_df):
    stats = {}
    for player, grp in played_df.groupby("Player"):
        pts = grp["Pts"].astype(float).values
        stats[player] = {
            "mean": float(np.mean(pts)),
            "var": float(np.var(pts, ddof=1)) if len(pts) > 1 else 9.0,
            "n": int(len(pts)),
        }
    return stats


def _shrink_each(stats, role_lookup, role_mu_map, global_mu, tau2, sigma2):
    out = {}
    for player, s in stats.items():
        role = role_lookup.get(player)
        target_mu = role_mu_map.get(role, global_mu) if role else global_mu
        out[player] = eb_shrink_role(s["mean"], s["n"], target_mu, tau2, sigma2)
    return out
