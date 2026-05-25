"""Analyze how regional performance translates to international events.

Goal: quantify a regional adjustment factor that maps a player's regional Pts/game
to expected international Pts/game. Especially important for China (weaker domestic
field) and to calibrate the smaller scaling for AMER/EMEA/PAC.

For each player who appeared in both a regional Stage AND a same-year international
event (Kickoff/Santiago/Toronto/Shanghai/London/Champions), compute:
  ratio_i = pts_intl_per_game / pts_regional_per_game

Aggregate by source region (where the player normally plays).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from v2.data_loader import normalize_team

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

REGIONAL_STAGES = ["Stage 1", "Stage 2"]
INTL_STAGES = ["Kickoff", "bangkok", "Madrid", "Santiago",
               "Toronto", "Shanghai", "London", "Champions"]

REGIONS = {
    "AMER": ["100 Thieves", "Cloud9", "ENVY", "Evil Geniuses", "FURIA",
             "G2 Esports", "KRÜ Esports", "KR Esports", "KRU Esports",
             "LEVIATµN", "LEVIATÁN", "Leviatan",
             "LOUD", "MIBR", "NRG Esports", "Sentinels", "The Guard",
             "2Game Esports"],
    "EMEA": ["BBL Esports", "Eternal Fire", "FNATIC", "FUT Esports",
             "FunPlus Phoenix", "GIANTX", "Gentle Mates", "Karmine Corp",
             "KOI", "Natus Vincere", "PCIFIC Esports", "Team Heretics",
             "Team Liquid", "Team Vitality"],
    "PAC":  ["DRX", "Detonation FocusMe", "DetonatioN FocusMe",
             "FULL SENSE", "Gen.G", "Global Esports", "Nongshim RedForce",
             "Paper Rex", "Rex Regum Qeon", "T1", "Talon",
             "Team Secret", "VARREL", "ZETA DIVISION", "Zeta Division"],
    "CN":   ["All Gamers", "Bilibili Gaming", "Dragon Ranger Gaming",
             "EDward Gaming", "Edward Gaming", "FunPlus Phoenix",
             "JD Mall JDG Esports", "Nova Esports", "Titan Esports Club",
             "Trace Esports", "TYLOO", "Wolves Esports", "Xi Lai Gaming"],
}

# Build team -> region
TEAM_REGION = {}
for region, teams in REGIONS.items():
    for t in teams:
        TEAM_REGION[normalize_team(t)] = region


def _read_csv_any(path):
    for enc in ["utf-8-sig", "utf-8", "latin-1"]:
        try:
            return pd.read_csv(path, encoding=enc,
                                dtype={"P?": str}, keep_default_na=False)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise IOError(f"could not decode {path}")


def load_combined():
    frames = []
    for year in [2024, 2025, 2026]:
        path = os.path.join(DATA, f"{year} VFL.csv")
        if not os.path.exists(path):
            continue
        df = _read_csv_any(path)
        if "Team" not in df.columns:
            cols = list(df.columns)
            df = df.rename(columns={cols[0]: "Team", cols[1]: "Player"})
        df["Year"] = year
        df["Team"] = df["Team"].apply(normalize_team)
        df["P?"] = pd.to_numeric(df["P?"], errors="coerce").fillna(0).astype(int)
        df["Pts"] = pd.to_numeric(df["Pts"], errors="coerce").fillna(0)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def compute_player_intl_ratio(played):
    """For each player-year, compute (regional_avg, intl_avg) per stage type."""
    rows = []
    for (player, year), grp in played.groupby(["Player", "Year"]):
        reg = grp[grp["Stage"].isin(REGIONAL_STAGES)]
        intl = grp[grp["Stage"].isin(INTL_STAGES)]
        if len(reg) < 3 or len(intl) < 2:
            continue
        team = reg["Team"].mode().iloc[0] if len(reg) else \
               intl["Team"].mode().iloc[0]
        region = TEAM_REGION.get(team, "?")
        rows.append({
            "Player": player, "Year": year, "Team": team, "Region": region,
            "reg_avg": float(reg["Pts"].mean()),
            "reg_n": len(reg),
            "intl_avg": float(intl["Pts"].mean()),
            "intl_n": len(intl),
        })
    return pd.DataFrame(rows)


def summarize_region(ratio_df):
    """Per-region: median/mean intl_avg/reg_avg ratio, and absolute diff."""
    print(f"\n{'Region':<8} {'N':>4} {'reg_avg':>10} {'intl_avg':>10} "
          f"{'ratio_med':>10} {'ratio_mean':>11} {'diff_med':>10}")
    summary = {}
    for region, grp in ratio_df.groupby("Region"):
        ratios = grp["intl_avg"] / grp["reg_avg"]
        ratios = ratios.replace([np.inf, -np.inf], np.nan).dropna()
        diffs = grp["intl_avg"] - grp["reg_avg"]
        summary[region] = {
            "n": len(grp),
            "reg_avg": float(grp["reg_avg"].mean()),
            "intl_avg": float(grp["intl_avg"].mean()),
            "ratio_med": float(ratios.median()),
            "ratio_mean": float(ratios.mean()),
            "diff_med": float(diffs.median()),
        }
        s = summary[region]
        print(f"{region:<8} {s['n']:>4} {s['reg_avg']:>10.2f} {s['intl_avg']:>10.2f} "
              f"{s['ratio_med']:>10.3f} {s['ratio_mean']:>11.3f} "
              f"{s['diff_med']:>10.2f}")
    return summary


def main():
    print("[1/3] Loading data...")
    all_df = load_combined()
    played = all_df[all_df["P?"] == 1].copy()
    played["Region"] = played["Team"].map(TEAM_REGION).fillna("?")
    print(f"  {len(played)} played rows total")

    print("\n[2/3] Computing per-player regional vs intl avg points...")
    ratio_df = compute_player_intl_ratio(played)
    print(f"  {len(ratio_df)} player-years with both regional and intl data")

    print("\n[3/3] Regional scaling factors (intl_avg / regional_avg):")
    summary = summarize_region(ratio_df)

    print("\n--- Interpretation ---")
    if "CN" in summary:
        cn_r = summary["CN"]["ratio_med"]
        print(f"China median intl/reg ratio: {cn_r:.3f}")
        if cn_r < 0.85:
            print("  -> Strong evidence: Chinese players score notably less internationally")
            print(f"     Suggested CN regional discount factor: {cn_r:.3f}")
        else:
            print("  -> Chinese players translate roughly 1:1")

    # Save for downstream pricing
    out_path = os.path.join(ROOT, "output", "regional_scaling.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    import json
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_path}")

    # Also save the raw per-player ratios for QC
    csv_path = os.path.join(ROOT, "output", "regional_scaling_perplayer.csv")
    ratio_df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
