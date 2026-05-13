"""v3 ablation: which of the 5 proposed changes actually helps?

For each holdout, run v2 baseline + v3 with each single flag ON
(rest OFF, on top of v2 baseline). Compare Pearson r vs actual PPG.
"""
import sys
import os
import io
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from v2.data_loader import load_all_data, load_manual_prices, load_pickrate_summary
from v2.expected_points import calibrate, compute_expected_pts
from v2.pricing import compute_prices
from v3.expected_points import compute_v3_expected_pts
from v3.pricing import compute_v3_prices
from v3.flags import V3Flags

HOLDOUTS = [
    (2024, "Stage 1"), (2024, "Stage 2"),
    (2025, "Kickoff"), (2025, "Stage 1"), (2025, "Stage 2"),
    (2026, "Kickoff"), (2026, "Santiago"), (2026, "Stage 1"),
]

ABLATIONS = [
    ("v2_baseline", None),
    ("v3_role_eb_only", V3Flags.only("role_mean_eb")),
    ("v3_b_floor_only", V3Flags.only("b_floor")),
    ("v3_team_form_only", V3Flags.only("team_form_decay")),
    ("v3_continuity_only", V3Flags.only("continuity")),
    ("v3_star_cap_only", V3Flags.only("star_cap")),
    ("v3_region_q_only", V3Flags.only("region_quantile")),
    ("v3_full", V3Flags()),
]

OUT_DIR = os.path.join(ROOT, "output")


def main():
    print("[init] Loading data and calibrating v2...")
    all_data = load_all_data()
    mp_df = load_manual_prices()
    pickrate_df = load_pickrate_summary()
    cal = calibrate(all_data)
    region_lookup = dict(zip(mp_df["Team"], mp_df["Region"]))

    rows = []
    for year, stage in HOLDOUTS:
        print(f"\n=== {year} {stage} ===")
        train, actual = _split(all_data, year, stage)
        if len(train) < 100 or len(actual) < 30:
            print("  skip")
            continue
        roster = _build_roster(actual, region_lookup, mp_df)
        actual_ppg = (actual.groupby("Player")["Pts"].mean()
                      .rename("actual_ppg").reset_index())
        actual_ppg = actual_ppg[actual_ppg["Player"].isin(roster)]
        if len(actual_ppg) < 30:
            print("  skip (overlap)")
            continue
        for label, flags in ABLATIONS:
            r = _score(label, flags, train, roster, actual_ppg,
                       pickrate_df, cal, year, stage)
            if r:
                rows.append(r)
    _write_results(rows)


def _split(all_data, target_year, target_stage):
    from v3.team_form import stage_rank
    target_r = stage_rank(target_year, target_stage)
    played = all_data[all_data["P?"] == 1].copy()
    played["_rank"] = played.apply(
        lambda r: stage_rank(r["Year"], r["Stage"]), axis=1)
    train = played[played["_rank"] < target_r].drop(columns=["_rank"])
    actual = played[(played["Year"] == target_year)
                    & (played["Stage"] == target_stage)].drop(columns=["_rank"])
    return train, actual


def _build_roster(actual, team_region, mp_df):
    role_map = dict(zip(mp_df["Player"], mp_df["Position"]))
    out = {}
    for player, grp in actual.groupby("Player"):
        team = grp["Team"].iloc[0]
        out[player] = {
            "team": team,
            "region": team_region.get(team, "AMER"),
            "role": role_map.get(player, "D"),
        }
    return out


def _score(label, flags, train, roster, actual_ppg, pickrate_df, cal,
           year, stage):
    try:
        if flags is None:
            ep = compute_expected_pts(train, roster, cal, schedule_gws=False)
            priced = compute_prices(ep, pickrate_df)
        else:
            ep = compute_v3_expected_pts(train, roster, cal,
                                          schedule_gws=False, flags=flags)
            priced = compute_v3_prices(ep, train, flags=flags)
        prices = priced[["Player", "SuggestedVP"]].rename(
            columns={"SuggestedVP": "price"})
    except Exception as e:
        print(f"  {label} FAILED: {e}")
        return None
    merged = prices.merge(actual_ppg, on="Player", how="inner")
    if len(merged) < 10:
        return None
    p = merged["price"].astype(float).values
    a = merged["actual_ppg"].astype(float).values
    r, _ = pearsonr(p, a)
    mae = float(np.mean(np.abs(p - a)))
    print(f"  {label:24s} n={len(merged):3d} r={r:+.3f} MAE={mae:.2f}")
    return {"year": year, "stage": stage, "label": label,
            "n": int(len(merged)), "pearson_r": float(r), "mae": mae}


def _write_results(rows):
    df = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, "v3_ablation_summary.csv")
    df.to_csv(out_path, index=False)
    print(f"\n[done] Wrote {out_path}")
    pivot = df.pivot_table(index=["year", "stage"], columns="label",
                            values="pearson_r").round(3)
    pivot["best"] = pivot.idxmax(axis=1)
    pivot_path = os.path.join(OUT_DIR, "v3_ablation_pearson_pivot.csv")
    pivot.to_csv(pivot_path)
    print(f"[done] Wrote {pivot_path}")
    print("\n=== Pearson r pivot ===")
    print(pivot.to_string())


if __name__ == "__main__":
    main()
