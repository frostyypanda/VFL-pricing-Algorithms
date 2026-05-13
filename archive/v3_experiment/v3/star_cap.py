"""v3 Star cap (spec §5.4).

If a player's recent-stage avg / historic avg < 0.85 AND historic > pop_mean,
cap their suggested price at 11.0.
"""
RECENT_DECLINE_THRESHOLD = 0.85
STAR_PRICE_CAP = 11.0


def compute_recent_vs_historic(played_df, target_players):
    """For each player, return (recent_avg_pts, historic_avg_pts).

    recent = most recent stage in played_df.
    historic = all stages combined.
    """
    if len(played_df) == 0:
        return {p: (None, None) for p in target_players}
    most_recent = _most_recent_stage_key(played_df)
    out = {}
    for player in target_players:
        out[player] = _player_recent_historic(player, played_df, most_recent)
    return out


def _most_recent_stage_key(played_df):
    from .team_form import stage_rank
    df = played_df.copy()
    df["_rank"] = df.apply(lambda r: stage_rank(r["Year"], r["Stage"]), axis=1)
    max_rank = df["_rank"].max()
    row = df[df["_rank"] == max_rank].iloc[0]
    return (int(row["Year"]), row["Stage"])


def _player_recent_historic(player, played_df, most_recent):
    p_df = played_df[played_df["Player"] == player]
    if len(p_df) == 0:
        return (None, None)
    year, stage = most_recent
    recent = p_df[(p_df["Year"] == year) & (p_df["Stage"] == stage)]
    recent_avg = float(recent["Pts"].mean()) if len(recent) > 0 else None
    historic_avg = float(p_df["Pts"].mean())
    return (recent_avg, historic_avg)


def apply_star_cap(price, recent_avg, historic_avg, pop_mean):
    """Return capped price if star is in decline, else original price."""
    if recent_avg is None or historic_avg is None:
        return price
    if historic_avg <= pop_mean:
        return price
    ratio = recent_avg / historic_avg
    if ratio >= RECENT_DECLINE_THRESHOLD:
        return price
    return min(price, STAR_PRICE_CAP)
