"""Multi-holdout backtest: Manual / Algo1 / Algo2 / v3 vs actual PPG.

Holdouts (all regular stages + recent Kickoffs):
  - 2024 Stage 1, 2024 Stage 2
  - 2025 Stage 1, 2025 Stage 2, 2025 Kickoff
  - 2026 Kickoff, 2026 Santiago, 2026 Stage 1
"""
import sys
import os
import io
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "archive"))

from v2.data_loader import load_all_data, load_manual_prices, load_pickrate_summary
from v2.expected_points import calibrate, compute_expected_pts
from v2.pricing import compute_prices
from v3.expected_points import compute_v3_expected_pts
from v3.pricing import compute_v3_prices

HOLDOUTS = [
    (2024, "Stage 1"), (2024, "Stage 2"),
    (2025, "Kickoff"), (2025, "Stage 1"), (2025, "Stage 2"),
    (2026, "Kickoff"), (2026, "Santiago"), (2026, "Stage 1"),
]

OUT_DIR = os.path.join(ROOT, "output")


def main():
    print("[init] Loading data...")
    all_data = load_all_data()
    mp_df = load_manual_prices()
    pickrate_df = load_pickrate_summary()
    region_lookup_team = _build_team_region_lookup(mp_df, all_data)
    cal = _calibrate_once(all_data)
    print(f"[init] Calibrated v2 base model (alpha={cal['best_ema_alpha']:.2f})")

    summary_rows = []
    for year, stage in HOLDOUTS:
        print(f"\n=== Holdout: {year} {stage} ===")
        result = _run_holdout(year, stage, all_data, mp_df,
                              pickrate_df, region_lookup_team, cal)
        if result is None:
            print("  skipped (insufficient data)")
            continue
        summary_rows.extend(result)
    _write_summary(summary_rows)


def _calibrate_once(all_data):
    return calibrate(all_data)


def _run_holdout(year, stage, all_data, mp_df, pickrate_df, region_team, cal):
    train, actual = _split_for_holdout(all_data, year, stage)
    if len(train) < 100 or len(actual) < 30:
        return None
    roster = _build_holdout_roster(actual, region_team, mp_df)
    target_players = set(roster.keys())
    actual_ppg = (actual.groupby("Player")["Pts"].mean()
                  .rename("actual_ppg").reset_index())
    actual_ppg = actual_ppg[actual_ppg["Player"].isin(target_players)]
    if len(actual_ppg) < 30:
        return None
    return _score_all_models(
        year, stage, train, actual_ppg, roster, mp_df, pickrate_df, cal,
    )


def _split_for_holdout(all_data, target_year, target_stage):
    from v3.team_form import stage_rank
    target_r = stage_rank(target_year, target_stage)
    played = all_data[all_data["P?"] == 1].copy()
    played["_rank"] = played.apply(
        lambda r: stage_rank(r["Year"], r["Stage"]), axis=1)
    train = played[played["_rank"] < target_r].drop(columns=["_rank"])
    actual = played[(played["Year"] == target_year)
                    & (played["Stage"] == target_stage)].drop(columns=["_rank"])
    return train, actual


def _build_holdout_roster(actual, team_region, mp_df):
    role_map = dict(zip(mp_df["Player"], mp_df["Position"]))
    roster = {}
    for player, grp in actual.groupby("Player"):
        team = grp["Team"].iloc[0]
        roster[player] = {
            "team": team,
            "region": team_region.get(team, "AMER"),
            "role": role_map.get(player, "D"),
        }
    return roster


def _build_team_region_lookup(mp_df, all_data):
    out = dict(zip(mp_df["Team"], mp_df["Region"]))
    return out


def _score_all_models(year, stage, train, actual_ppg, roster, mp_df,
                       pickrate_df, cal):
    rows = []
    rows.append(_score_manual(year, stage, actual_ppg, mp_df, train))
    rows.append(_score_algo1(year, stage, train, actual_ppg, roster, mp_df,
                              pickrate_df))
    rows.append(_score_algo2(year, stage, train, actual_ppg, roster, mp_df,
                              pickrate_df, cal))
    rows.append(_score_v3(year, stage, train, actual_ppg, roster, mp_df, cal))
    for r in rows:
        if r is None:
            continue
        _print_row(r)
    return [r for r in rows if r is not None]


def _print_row(r):
    print(f"  {r['model']:14s} n={r['n']:3d} r={r['pearson_r']:+.3f} "
          f"rho={r['spearman_rho']:+.3f} MAE={r['mae']:.2f}")


def _score_manual(year, stage, actual_ppg, mp_df, train):
    if (year, stage) == (2026, "Stage 1"):
        prices = mp_df[["Player", "Stage1_Price"]].rename(
            columns={"Stage1_Price": "price"})
    else:
        prices = _proxy_manual_from_csv(year, stage)
    if prices is None:
        return None
    return _metrics("Manual", actual_ppg, prices, year, stage)


def _proxy_manual_from_csv(year, stage):
    """Use Game Start VP of the first played game in this stage as Manual."""
    path = os.path.join(ROOT, "data", f"{year} VFL.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")
    df = df[(df["Stage"] == stage)
            & (pd.to_numeric(df["P?"], errors="coerce") == 1)].copy()
    if "Game Start VP" not in df.columns or len(df) == 0:
        return None
    df["Game Start VP"] = pd.to_numeric(df["Game Start VP"], errors="coerce")
    df = df.dropna(subset=["Game Start VP"])
    if len(df) == 0:
        return None
    first = df.sort_values(["Player", "Wk", "Game"]).groupby("Player").first()
    out = first.reset_index()[["Player", "Game Start VP"]].rename(
        columns={"Game Start VP": "price"})
    return out[out["price"] > 0]


def _score_algo1(year, stage, train, actual_ppg, roster, mp_df, pickrate_df):
    try:
        prices = _algo1_prices(train, mp_df, pickrate_df)
        return _metrics("Algo1_Combined", actual_ppg, prices, year, stage)
    except Exception as e:
        print(f"  Algo1 failed: {e}")
        return None


def _algo1_prices(training_df, mp_df, pickrate_df):
    import pricing_algorithms as A
    A.STAGE_ORDER.setdefault("Santiago", 1)
    player_picks = {}
    if pickrate_df is not None:
        for _, r in pickrate_df.iterrows():
            player_picks[r["Player"].lower()] = {
                "avg_pickpct": r.get("avg_pickpct", 0.0),
                "avg_rank_pct": r.get("avg_rank_pct", 50.0),
            }
    features = A.compute_player_features(training_df, player_picks)
    team_wr = A.compute_team_win_rates(training_df)
    stage1_players = mp_df[["Team", "Player"]].drop_duplicates()
    team_new = A.detect_new_players_on_team(training_df, stage1_players)
    team_pop = (A.compute_team_popularity(pickrate_df, training_df)
                if pickrate_df is not None else {})
    opp_strength = A.compute_opponent_strength(training_df)
    pred = A.algo_combined(
        features, team_win_rates=team_wr, team_new_counts=team_new,
        team_popularity=team_pop, opponent_strength=opp_strength,
    )
    pred = A.fill_missing_players(
        pred, stage1_players, features,
        team_win_rates=team_wr, team_new_counts=team_new,
    )
    return pred.rename(columns={"predicted_vp": "price"})[["Player", "price"]]


def _score_algo2(year, stage, train, actual_ppg, roster, mp_df, pickrate_df, cal):
    try:
        ep = compute_expected_pts(train, roster, cal, schedule_gws=False)
        priced = compute_prices(ep, pickrate_df)
        prices = priced[["Player", "SuggestedVP"]].rename(
            columns={"SuggestedVP": "price"})
        return _metrics("Algo2_v2", actual_ppg, prices, year, stage)
    except Exception as e:
        print(f"  Algo2 failed: {e}")
        return None


def _score_v3(year, stage, train, actual_ppg, roster, mp_df, cal):
    try:
        ep = compute_v3_expected_pts(train, roster, cal, schedule_gws=False)
        priced = compute_v3_prices(ep, train)
        prices = priced[["Player", "SuggestedVP"]].rename(
            columns={"SuggestedVP": "price"})
        return _metrics("v3", actual_ppg, prices, year, stage)
    except Exception as e:
        print(f"  v3 failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def _metrics(label, actual_ppg, prices, year, stage):
    merged = prices.merge(actual_ppg, on="Player", how="inner")
    if len(merged) < 10:
        return None
    p = merged["price"].astype(float).values
    a = merged["actual_ppg"].astype(float).values
    r, _ = pearsonr(p, a)
    rho, _ = spearmanr(p, a)
    return {
        "year": year, "stage": stage, "model": label,
        "n": int(len(merged)),
        "pearson_r": float(r), "spearman_rho": float(rho),
        "mae": float(np.mean(np.abs(p - a))),
        "mean_price": float(np.mean(p)), "median_price": float(np.median(p)),
    }


def _write_summary(rows):
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, "v3_backtest_summary.csv")
    df.to_csv(out_path, index=False)
    print(f"\n[done] Wrote {out_path}")
    print("\n=== Backtest summary ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
