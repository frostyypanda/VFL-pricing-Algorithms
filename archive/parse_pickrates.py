"""
Parse VFL pickrate JSON files into a consolidated CSV.

Event mapping (confirmed via player overlap analysis):
  Event 1 = 2025 Toronto (Masters 2)     — 60 players, 3 gameweeks
  Event 2 = 2025 Stage 2                 — 183 players, 5 gameweeks
  Event 3 = 2025 Champions               — 135 players, 3 gameweeks
  Event 4 = 2026 Kickoff                 — 180 players, 1 gameweek
  Event 5 = 2026 Santiago (Masters 1)    — 60 players, 3 gameweeks

Output: pickrate_data.csv with columns:
  Player, Event, Gameweek, Pickcount, PickPct (% of max picks that GW)
"""
import json
import os
import pandas as pd

DOWNLOAD_DIR = r"D:\Downloads"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

EVENT_MAP = {
    1: ("2025 Toronto", "Toronto"),
    2: ("2025 Stage 2", "Stage 2"),
    3: ("2025 Champions", "Champions"),
    4: ("2026 Kickoff", "Kickoff"),
    5: ("2026 Santiago", "Santiago"),
}


def parse_all_files():
    rows = []
    for fname in os.listdir(DOWNLOAD_DIR):
        if not fname.startswith("most_picked") or not fname.endswith(".json"):
            continue
        parts = fname.replace("most_pickedgw", "").replace(".json", "")
        gw_str, event_str = parts.split("event")
        gw = int(gw_str)
        event_id = int(event_str)

        if event_id not in EVENT_MAP:
            print(f"Warning: unknown event {event_id} in {fname}")
            continue

        event_name, stage_name = EVENT_MAP[event_id]
        path = os.path.join(DOWNLOAD_DIR, fname)
        with open(path) as f:
            data = json.load(f)

        for entry in data:
            rows.append({
                "Player": entry["Name"],
                "Event": event_name,
                "Stage": stage_name,
                "Gameweek": gw,
                "Pickcount": int(entry["pickcount"]),
            })

    df = pd.DataFrame(rows)

    # Compute pick percentage within each event+gameweek
    # (percentage of maximum possible picks, approximated by max pickcount)
    for (event, gw), grp in df.groupby(["Event", "Gameweek"]):
        max_picks = grp["Pickcount"].max()
        df.loc[grp.index, "MaxPicks"] = max_picks
    df["PickPct"] = (df["Pickcount"] / df["MaxPicks"] * 100).round(2)

    df = df.sort_values(["Event", "Gameweek", "Pickcount"], ascending=[True, True, False])
    return df[["Player", "Event", "Stage", "Gameweek", "Pickcount", "PickPct"]]


def compute_player_pickrate_summary(df):
    """Compute per-player pickrate summary across all events.

    Returns DataFrame with:
      Player, avg_pickpct, max_pickpct, events_appeared, total_gameweeks,
      avg_rank (average rank within each gameweek)
    """
    # Compute rank within each event+gameweek
    df = df.copy()
    df["Rank"] = df.groupby(["Event", "Gameweek"])["Pickcount"].rank(ascending=False, method="min")
    df["PoolSize"] = df.groupby(["Event", "Gameweek"])["Player"].transform("count")
    df["RankPct"] = (df["Rank"] / df["PoolSize"] * 100).round(2)

    summary = df.groupby("Player").agg(
        avg_pickpct=("PickPct", "mean"),
        max_pickpct=("PickPct", "max"),
        avg_rank_pct=("RankPct", "mean"),
        events_appeared=("Event", "nunique"),
        total_gameweeks=("Gameweek", "count"),
        total_picks=("Pickcount", "sum"),
    ).round(2)

    return summary.sort_values("avg_pickpct", ascending=False)


def main():
    print("Parsing pickrate files...")
    df = parse_all_files()
    print(f"Loaded {len(df)} rows across {df['Event'].nunique()} events")

    # Save raw data
    raw_path = os.path.join(OUT_DIR, "pickrate_data.csv")
    df.to_csv(raw_path, index=False, encoding="utf-8")
    print(f"Saved: {raw_path}")

    # Compute and save summary
    summary = compute_player_pickrate_summary(df)
    summary_path = os.path.join(OUT_DIR, "pickrate_summary.csv")
    summary.to_csv(summary_path, encoding="utf-8")
    print(f"Saved: {summary_path}")

    print(f"\nTop 20 most picked players (by avg pick %):")
    print(summary.head(20).to_string())

    print(f"\n\nPer-event stats:")
    for event in sorted(df["Event"].unique()):
        edf = df[df["Event"] == event]
        gws = edf["Gameweek"].nunique()
        players = edf["Player"].nunique()
        print(f"  {event}: {players} players, {gws} gameweeks, "
              f"avg picks/player/gw: {edf['Pickcount'].mean():.0f}")


if __name__ == "__main__":
    main()
