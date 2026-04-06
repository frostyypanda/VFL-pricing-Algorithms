"""Compute VFL fantasy points from VLR match data."""


def kill_bracket_pts(kills):
    """Kill bracket points for a single map."""
    if kills == 0:
        return -3
    if kills <= 4:
        return -1
    if kills <= 9:
        return 0
    if kills <= 14:
        return 1
    if kills <= 19:
        return 2
    if kills <= 24:
        return 3
    if kills <= 29:
        return 4
    return 4 + (kills - 29) // 5


def map_team_pts(winner, score_w, score_l):
    """Team points for a map. winner=True if this player's team won."""
    margin = score_w - score_l if winner else score_l - score_w
    if winner:
        base = 1
        if margin >= 10:
            return base + 2
        if margin >= 5:
            return base + 1
        return base
    if margin <= -10:
        return -1
    return 0


def rating_bonuses(player_rating, all_ratings):
    """Rating-based bonuses (TOP3/2/1 + threshold bonuses)."""
    pts = 0
    sorted_r = sorted(all_ratings, reverse=True)
    rank = sorted_r.index(player_rating) + 1
    if rank == 1:
        pts += 3
    elif rank == 2:
        pts += 2
    elif rank == 3:
        pts += 1

    if player_rating >= 2.0:
        pts += 3
    elif player_rating >= 1.75:
        pts += 2
    elif player_rating >= 1.5:
        pts += 1
    return pts


def compute_match_points(match):
    """Compute VFL fantasy points for all players in a match.

    Returns dict of {player_name: {"pts": total, "team": str, ...}}.
    """
    results = {}
    map_scores = match["map_scores"]
    per_map = match["per_map"]
    agg = match["aggregate"]
    multikills = match.get("multikills", {})

    all_ratings = []
    for side in ["team1", "team2"]:
        for p in agg[side]:
            all_ratings.append(p["rating"])

    for side_idx, side in enumerate(["team1", "team2"]):
        team_name = match["team1"] if side_idx == 0 else match["team2"]

        for p_agg in agg[side]:
            name = p_agg["name"]
            total = 0

            for map_idx, ms in enumerate(map_scores):
                if map_idx >= len(per_map):
                    break
                t1_won = ms["team1_won"]
                won = t1_won if side_idx == 0 else not t1_won
                sw = max(ms["score1"], ms["score2"])
                sl = min(ms["score1"], ms["score2"])

                kills = _find_kills(per_map[map_idx], side, name)
                total += kill_bracket_pts(kills)
                total += map_team_pts(won, sw, sl)

            total += rating_bonuses(p_agg["rating"], all_ratings)

            # Exact multi-kill bonuses from Performance tab
            mk = multikills.get(name, {"4k": 0, "5k": 0})
            total += mk["4k"] * 1   # +1 per 4K round
            total += mk["5k"] * 3   # +3 per 5K (ace)

            results[name] = {
                "pts": total, "team": team_name,
                "rating": p_agg["rating"], "kills": p_agg["kills"],
                "deaths": p_agg["deaths"], "maps": len(map_scores),
                "4k": mk["4k"], "5k": mk["5k"],
            }
    return results


def _find_kills(map_data, side, name):
    """Find a player's kills in a specific map."""
    for p in map_data.get(side, []):
        if p["name"] == name:
            return p["kills"]
    return 0


