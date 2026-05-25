"""Merge Stage 1 playoff scrapes (AMER/EMEA/PAC) and full China Stage 1 into 2026 VFL.csv.

AMER/EMEA/PAC: updates existing P?=0 playoff placeholder rows (matched by Team,Player,Game).
China:        replaces all Stage 1 China placeholder rows with new rows using
              actual game IDs (G1-G5 round-robin, SR1 seeding, UQF, USF, UF, LR1-3, LF, GF).

Re-uses the row-building helpers from scripts/merge_results_to_csv.py.
"""
import sys
import os
import io
import json

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from scripts.merge_results_to_csv import (
    compute_team_pts_for_player, compute_brackets, rating_flags_and_pts,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
CSV_PATH = os.path.join(DATA, "2026 VFL.csv")


TEAM_ALIASES_VLR_TO_CSV = {
    "Kiwoom DRX": "DRX",
    "Wuxi Titan Esports Club(Titan Esports Club)": "Titan Esports Club",
    "Guangzhou Huadu Bilibili Gaming(Bilibili Gaming)": "Bilibili Gaming",
    "JD Mall JDG Esports(JDG Esports)": "JD Mall JDG Esports",
    "JDG Esports": "JD Mall JDG Esports",
    "DetonatioN FocusMe": "Detonation FocusMe",
    "LEVIATÁN": "LEVIATµN",
}


def canon_team(name):
    """VLR team name -> canonical team name used in the CSV."""
    return TEAM_ALIASES_VLR_TO_CSV.get(name, name)


def build_match_rows_for_event(match, manual_prices, player_lookup, stage_label):
    """Build CSV rows for one match. Game label comes from match['game']."""
    map_scores = match["map_scores"]
    per_map = match["per_map"]
    agg = match["aggregate"]
    multikills = match.get("multikills", {})
    game_label = match["game"]
    all_ratings = [p["rating"] for side in ("team1", "team2") for p in agg[side]]

    team1 = canon_team(match["team1"])
    team2 = canon_team(match["team2"])

    rows = []
    for side_idx, side in enumerate(["team1", "team2"]):
        team_csv = team1 if side_idx == 0 else team2
        team_wins = sum(
            1 for ms in map_scores
            if (ms["team1_won"] if side_idx == 0 else not ms["team1_won"])
        )
        result = "W" if team_wins * 2 > len(map_scores) else "L"
        t_pts = compute_team_pts_for_player(side_idx, map_scores)

        for p_agg in agg[side]:
            row = _build_row(
                p_agg, side, per_map, multikills, all_ratings, t_pts,
                len(map_scores), game_label, result, manual_prices,
                player_lookup, team_csv, stage_label,
            )
            if row is not None:
                rows.append(row)
    return rows


def _build_row(p_agg, side, per_map, multikills, all_ratings, t_pts,
               maps_played, game_label, result, manual_prices,
               player_lookup, team_csv, stage_label):
    json_name = p_agg["name"]
    resolved = player_lookup.get((team_csv, json_name.lower()))
    if resolved is None:
        # try team-agnostic lookup as fallback (player may exist on this team in CSV)
        resolved = player_lookup.get(("__any__", json_name.lower()))
    if resolved is None:
        # Final fallback: synthesize a row using the JSON name (China case)
        csv_team, csv_player = team_csv, json_name
    else:
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
        "Team": csv_team, "Player": csv_player, "Stage": stage_label,
        "Wk": "", "Game": game_label,
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


def build_player_lookup(df, stage):
    """{(team, player_lower): (team, player_canonical)} from stage rows."""
    s = df[df["Stage"] == stage]
    out = {}
    for _, row in s[["Team", "Player"]].drop_duplicates().iterrows():
        out[(row["Team"], row["Player"].lower())] = (row["Team"], row["Player"])
        out[("__any__", row["Player"].lower())] = (row["Team"], row["Player"])
    return out


def load_manual_prices_map():
    from v2.data_loader import load_manual_prices
    mp = load_manual_prices()
    return {p.lower(): float(price) if pd.notna(price) else 0.0
            for p, price in zip(mp["Player"], mp["Stage1_Price"])}


def apply_playoffs_to_csv(df, playoff_matches, manual_prices):
    """Update existing Stage 1 playoff placeholder rows in-place for AMER/EMEA/PAC."""
    lookup = build_player_lookup(df, "Stage 1")
    updates = []
    for m in playoff_matches:
        updates.extend(build_match_rows_for_event(
            m, manual_prices, lookup, "Stage 1"
        ))
    idx_map = {}
    for i, r in df.iterrows():
        if r["Stage"] == "Stage 1":
            idx_map[(r["Team"], r["Player"], r["Game"])] = i

    applied = 0
    inserted_rows = []
    for u in updates:
        idx = idx_map.get((u["Team"], u["Player"], u["Game"]))
        if idx is None:
            inserted_rows.append(u)
            continue
        for k, v in u.items():
            df.at[idx, k] = str(v)
        applied += 1
    print(f"  AMER/EMEA/PAC playoffs updated: {applied}, new rows: {len(inserted_rows)}")

    if inserted_rows:
        ins_df = pd.DataFrame(inserted_rows)
        for col in df.columns:
            if col not in ins_df.columns:
                ins_df[col] = ""
        ins_df = ins_df[df.columns]
        df = pd.concat([df, ins_df], ignore_index=True)
    return df


def replace_china_stage1(df, china_matches, manual_prices):
    """Drop existing China Stage 1 rows, insert real rows from scrapes."""
    china_teams = sorted(set(canon_team(m["team1"]) for m in china_matches) |
                         set(canon_team(m["team2"]) for m in china_matches))
    print(f"  China teams (canonicalized): {china_teams}")

    # Build minimal player lookup from China rows in CSV (for canonical names)
    cn_rows = df[(df["Stage"] == "Stage 1") & (df["Team"].isin(china_teams))]
    lookup = {}
    for _, row in cn_rows[["Team", "Player"]].drop_duplicates().iterrows():
        lookup[(row["Team"], row["Player"].lower())] = (row["Team"], row["Player"])
        lookup[("__any__", row["Player"].lower())] = (row["Team"], row["Player"])

    new_rows = []
    for m in china_matches:
        new_rows.extend(build_match_rows_for_event(
            m, manual_prices, lookup, "Stage 1"
        ))
    print(f"  China new rows to insert: {len(new_rows)}")

    # Drop existing China Stage 1 rows
    mask_drop = (df["Stage"] == "Stage 1") & (df["Team"].isin(china_teams))
    print(f"  Dropping {mask_drop.sum()} existing China Stage 1 placeholders")
    df = df[~mask_drop].copy()

    # Add the new rows. Match column order to the existing DataFrame.
    new_df = pd.DataFrame(new_rows)
    for col in df.columns:
        if col not in new_df.columns:
            new_df[col] = ""
    new_df = new_df[df.columns]

    df = pd.concat([df, new_df], ignore_index=True)
    return df


def main():
    print("[1/3] Loading CSV + manual prices...")
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig", dtype=str,
                     keep_default_na=False)
    print(f"  {len(df)} rows initial")
    manual_prices = load_manual_prices_map()
    print(f"  {len(manual_prices)} manual prices loaded")

    print("[2/3] Applying AMER/EMEA/PAC Stage 1 playoffs...")
    with open(os.path.join(DATA, "stage1_playoffs_results.json")) as f:
        pj = json.load(f)
    df = apply_playoffs_to_csv(df, pj["matches"], manual_prices)

    print("[3/3] Replacing China Stage 1 with real data...")
    with open(os.path.join(DATA, "stage1_china_results.json")) as f:
        cj = json.load(f)
    df = replace_china_stage1(df, cj["matches"], manual_prices)

    print(f"\nWriting back to {CSV_PATH} ({len(df)} rows)")
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

    print("\nSummary after merge:")
    s1 = df[df["Stage"] == "Stage 1"]
    played = s1[s1["P?"] == "1"]
    print(f"  Stage 1 played: {len(played)}")
    print(f"  Stage 1 by Game:")
    print(played["Game"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
