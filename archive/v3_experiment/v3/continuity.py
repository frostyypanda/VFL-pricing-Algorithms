"""v3 Team-continuity (spec §5.5).

For each player, fraction of prior games on their *current* team.
Used to shrink low-continuity players harder toward role-mean.
"""


def compute_team_continuity(played_df, roster):
    """Return {player: continuity_fraction in [0, 1]}.

    continuity = games_on_current_team / total_prior_games.
    Players with 0 prior games → 1.0 (no history, no penalty).
    """
    out = {}
    by_player = played_df.groupby("Player")
    for player, info in roster.items():
        current_team = info["team"]
        out[player] = _player_continuity(player, current_team, by_player)
    return out


def _player_continuity(player, current_team, by_player):
    try:
        grp = by_player.get_group(player)
    except KeyError:
        return 1.0
    total = len(grp)
    if total == 0:
        return 1.0
    same = int((grp["Team"] == current_team).sum())
    return float(same) / float(total)


def continuity_adjusted_estimate(personal_est, role_mu, continuity):
    """Blend personal_est toward role_mu when continuity is low.

    continuity ≥ 0.7 → full personal estimate (no adjustment).
    continuity ≤ 0.3 → 50/50 personal/role.
    Linearly interpolated between.
    """
    if continuity >= 0.7:
        return personal_est
    if continuity <= 0.3:
        return 0.5 * personal_est + 0.5 * role_mu
    blend_weight = (continuity - 0.3) / 0.4
    role_weight = (1.0 - blend_weight) * 0.5
    return (1.0 - role_weight) * personal_est + role_weight * role_mu
