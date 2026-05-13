"""Merge W1-W6 scraped JSON results into 2026 VFL.csv (Stage 1).

Replaces P?=0 placeholder rows (Stage 1, G1-G5) with actual scraped stats
for the matched (Team, Player, Game). Keeps playoff placeholder rows untouched.

Region/week mapping:
  w1: EMEA W1, PAC W1
  w2: AMER W1, EMEA W2, PAC W2
  w3: AMER W2, EMEA W3, PAC W3
  w4: AMER W3, EMEA W4, PAC W4
  w5: AMER W4, EMEA W5, PAC W5
  w6: AMER W5

Decisions:
- VP cols (Adj.VP, Game Start VP, Game End VP) <- player manual price (constant).
- PR Avg. left blank (display field, not used downstream).
- Player names matched case-insensitively; CSV's canonical (Team, Player) form is kept.
"""
import sys
import os
import io
import json

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
CSV_PATH = os.path.join(DATA, "2026 VFL.csv")


def week_label_for(wk_n, region):
    if region == "AMER":
        return f"W{wk_n - 1}"
    return f"W{wk_n}"


def load_manual_prices_map():
    from v2.data_loader import load_manual_prices
    mp = load_manual_prices()
    return {p.lower(): float(price) if pd.notna(price) else 0.0
            for p, price in zip(mp["Player"], mp["Stage1_Price"])}


def kill_bucket_index(kills):
    """0k=0; 5k=1-4; 5-9 not bucketed; 10k=10-14;...50k=50+.
    Returns col index 0..10 or None for 5-9 kills."""
    if kills == 0:
        return 0
    if kills <= 4:
        return 1
    if kills <= 9:
        return None
    return min(2 + (kills - 10) // 5, 10)


def compute_team_pts_for_player(side_idx, map_scores):
    t = 0
    for ms in map_scores:
        won = ms["team1_won"] if side_idx == 0 else not ms["team1_won"]
        sw = max(ms["score1"], ms["score2"])
        sl = min(ms["score1"], ms["score2"])
        margin = sw - sl
        if won:
            pts = 1 + (1 if margin >= 5 else 0) + (2 if margin >= 10 else 0)
            t += pts
        elif margin >= 10:
            t -= 1
    return t


def compute_brackets(side, player_name, per_map):
    brackets = [0] * 11
    kill_pts = 0
    for mp in per_map:
        kills = _find_kills_in_map(mp, side, player_name)
        if kills is None:
            continue
        idx = kill_bucket_index(kills)
        if idx is not None:
            brackets[idx] += 1
        kill_pts += _kill_pts(kills)
    return brackets, kill_pts


def _kill_pts(kills):
    if kills == 0:
        return -3
    if kills <= 4:
        return -1
    if kills <= 9:
        return 0
    return 1 + (kills - 10) // 5


def _find_kills_in_map(map_data, side, name):
    for p in map_data.get(side, []):
        if p["name"] == name:
            return p["kills"]
    return None


def rating_flags_and_pts(player_rating, all_ratings):
    sorted_r = sorted(all_ratings, reverse=True)
    rank = sorted_r.index(player_rating) + 1
    t3 = 1 if rank == 3 else 0
    t2 = 1 if rank == 2 else 0
    t1 = 1 if rank == 1 else 0
    r15 = r175 = r20 = 0
    if player_rating >= 2.0:
        r20 = 1
    elif player_rating >= 1.75:
        r175 = 1
    elif player_rating >= 1.5:
        r15 = 1
    pts = t3 + 2 * t2 + 3 * t1 + r15 + 2 * r175 + 3 * r20
    return (t3, t2, t1, r15, r175, r20), pts


def build_match_rows(match, wk_n, player_lookup, manual_prices):
    """Yield row dicts for each resolvable player in the match."""
    wk = week_label_for(wk_n, match["region"])
    game = f"G{int(wk[1:])}"
    map_scores = match["map_scores"]
    per_map = match["per_map"]
    agg = match["aggregate"]
    multikills = match.get("multikills", {})
    all_ratings = [p["rating"] for side in ("team1", "team2") for p in agg[side]]

    for side_idx, side in enumerate(["team1", "team2"]):
        team_wins = sum(1 for ms in map_scores
                        if (ms["team1_won"] if side_idx == 0 else not ms["team1_won"]))
        result = "W" if team_wins * 2 > len(map_scores) else "L"
        t_pts = compute_team_pts_for_player(side_idx, map_scores)
        for p_agg in agg[side]:
            row = _row_for_player(
                p_agg, side, per_map, multikills, all_ratings,
                t_pts, len(map_scores), wk, game, result,
                manual_prices, player_lookup,
            )
            if row is not None:
                yield row


def _row_for_player(p_agg, side, per_map, multikills, all_ratings,
                    t_pts, maps_played, wk, game, result,
                    manual_prices, player_lookup):
    json_name = p_agg["name"]
    resolved = player_lookup.get(json_name.lower())
    if resolved is None:
        return None
    csv_team, csv_player = resolved
    brackets, kill_pts = compute_brackets(side, json_name, per_map)
    flags, rating_pts = rating_flags_and_pts(p_agg["rating"], all_ratings)
    mk = multikills.get(json_name, {"4k": 0, "5k": 0})
    mk_pts = mk["4k"] + mk["5k"] * 3
    p_pts = kill_pts + rating_pts + mk_pts
    total = p_pts + t_pts
    ppm = round(total / max(maps_played, 1), 2)
    price = manual_prices.get(csv_player.lower(), 0.0)
    return {
        "Team": csv_team, "Player": csv_player, "Stage": "Stage 1",
        "Wk": wk, "Game": game,
        "Pts": total, "T.Pts": t_pts, "P.Pts": p_pts, "PPM": ppm,
        "Adj.VP": price, "P?": 1,
        "0k": brackets[0], "5k": brackets[1], "10k": brackets[2],
        "15k": brackets[3], "20k": brackets[4], "25k": brackets[5],
        "30k": brackets[6], "35k": brackets[7], "40k": brackets[8],
        "45k": brackets[9], "50k": brackets[10],
        "4ks": mk["4k"], "5ks": mk["5k"], "6ks": 0, "7ks": 0,
        "TOP3": flags[0], "TOP2": flags[1], "TOP1": flags[2],
        "1.5R2": flags[3], "1.75R2": flags[4], "2.0R2": flags[5],
        "PR Avg.": "", "W/L": result,
        "Game Start VP": price, "Game End VP": price,
    }


def load_all_json():
    by_gw = {}
    for wk in range(1, 7):
        path = os.path.join(DATA, f"w{wk}_vlr_results.json")
        with open(path, encoding="utf-8") as f:
            by_gw[wk] = json.load(f)
    return by_gw


def build_player_lookup(df):
    """Case-insensitive {player_lower: (canonical_team, canonical_player)} from Stage 1 rows."""
    s1 = df[df["Stage"] == "Stage 1"]
    out = {}
    for _, row in s1[["Team", "Player"]].drop_duplicates().iterrows():
        out[row["Player"].lower()] = (row["Team"], row["Player"])
    return out


def main():
    print("[1/4] Loading CSV + manual prices...")
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    print(f"  {len(df)} rows in 2026 VFL.csv")
    manual_prices = load_manual_prices_map()
    print(f"  {len(manual_prices)} manual prices loaded")

    print("[2/4] Building player lookup (case-insensitive)...")
    player_lookup = build_player_lookup(df)
    print(f"  {len(player_lookup)} unique player keys")

    print("[3/4] Loading + computing all gameweek JSONs...")
    all_json = load_all_json()
    updates = []
    for wk_n in range(1, 7):
        for match in all_json[wk_n]["matches"]:
            updates.extend(build_match_rows(match, wk_n, player_lookup, manual_prices))
    print(f"  {len(updates)} player-game updates computed")

    print("[4/4] Applying updates to CSV...")
    _apply_updates(df, updates)
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"  Wrote {CSV_PATH}")
    _print_summary(df)


def _apply_updates(df, updates):
    idx_map = {}
    for i, r in df.iterrows():
        if r["Stage"] == "Stage 1":
            idx_map[(r["Team"], r["Player"], r["Game"])] = i
    applied = 0
    missed = []
    for u in updates:
        key = (u["Team"], u["Player"], u["Game"])
        idx = idx_map.get(key)
        if idx is None:
            missed.append(key)
            continue
        for k, v in u.items():
            df.at[idx, k] = str(v)
        applied += 1
    print(f"  Applied {applied} row updates")
    if missed:
        print(f"  WARN: {len(missed)} updates missed a matching placeholder row")
        for k in missed[:10]:
            print(f"    {k}")


def _print_summary(df):
    s1 = df[df["Stage"] == "Stage 1"]
    played = s1[s1["P?"] == "1"]
    print(f"\nStage 1 played rows after merge: {len(played)}")
    print("By Wk:")
    print(played["Wk"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
