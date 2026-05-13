"""v3 Team-form decay (spec §5.2).

Weight each team's WR by stage recency. Most recent stage = weight 1.0,
each stage back = multiplied by DECAY_BASE.
"""
DECAY_BASE = 0.6

STAGE_ORDER_GLOBAL = [
    "Kickoff", "bangkok", "Madrid", "Santiago",
    "Stage 1", "Toronto", "Shanghai", "London",
    "Stage 2", "Champions",
]


def stage_rank(year, stage):
    """Global stage ordering: higher = more recent."""
    if stage in STAGE_ORDER_GLOBAL:
        idx = STAGE_ORDER_GLOBAL.index(stage)
    else:
        idx = 2
    return year * 10 + idx


def compute_team_form(played_df):
    """Return {team: decayed_wr}. Recent stages weighted heavily.

    decayed_wr = Σ w_i * wr_i / Σ w_i, where w_i = DECAY_BASE^(distance_from_max).
    """
    if len(played_df) == 0:
        return {}
    df = played_df.copy()
    df["_rank"] = df.apply(lambda r: stage_rank(r["Year"], r["Stage"]), axis=1)
    max_rank = df["_rank"].max()
    return _decayed_wr_by_team(df, max_rank)


def _decayed_wr_by_team(df, max_rank):
    out = {}
    for team, grp in df.groupby("Team"):
        out[team] = _team_decayed_wr(grp, max_rank)
    return out


def _team_decayed_wr(grp, max_rank):
    by_stage = grp.groupby("_rank")
    weighted_sum = 0.0
    weight_sum = 0.0
    for rank_val, sgrp in by_stage:
        wins = float((sgrp["W/L"] == "W").sum())
        total = float(len(sgrp))
        wr = wins / total if total > 0 else 0.5
        steps_back = int(max_rank - rank_val) // 1
        w = DECAY_BASE ** steps_back
        weighted_sum += w * wr
        weight_sum += w
    return weighted_sum / weight_sum if weight_sum > 0 else 0.5


def compute_team_form_ratio(played_df):
    """Return {team: form_ratio = decayed_wr / pooled_wr}.

    Ratio < 1.0 → team is in decline (recent worse than historical).
    Ratio > 1.0 → team is improving.
    Clamped to [0.7, 1.3] to avoid over-correction.
    """
    decayed = compute_team_form(played_df)
    pooled = _pooled_wr(played_df)
    out = {}
    for team, d_wr in decayed.items():
        p_wr = pooled.get(team, 0.5)
        if p_wr < 0.05:
            out[team] = 1.0
            continue
        ratio = d_wr / p_wr
        out[team] = max(0.7, min(1.3, ratio))
    return out


def _pooled_wr(played_df):
    out = {}
    for team, grp in played_df.groupby("Team"):
        wins = float((grp["W/L"] == "W").sum())
        total = float(len(grp))
        out[team] = wins / total if total > 0 else 0.5
    return out
