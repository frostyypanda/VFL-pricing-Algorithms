"""Leave-one-stage-out hyperparameter tuning for v2.

For each of 8 historical holdouts:
  - Train hyperparams on the OTHER 7 stages
  - Evaluate winning config on the held-out 8th
Aggregate the 8 LOO r values → honest cross-validated v2*.
"""
import sys
import os
import io
import json
import numpy as np
import pandas as pd

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from v2.data_loader import load_all_data, load_manual_prices, load_pickrate_summary
from v2_retune.config import (
    CAL_WINDOWS, RECENCY_MAPS, ENSEMBLE_WEIGHTS, PICKRATE_WEIGHTS, HOLDOUTS,
)
from v2_retune.base_estimates import calibrate, compute_base_estimates
from v2_retune.score import score_holdout
from v3.team_form import stage_rank

OUT_DIR = os.path.join(ROOT, "output")


def main():
    print("[init] Loading data...")
    all_data = load_all_data()
    mp_df = load_manual_prices()
    pickrate_df = load_pickrate_summary()
    region_lookup = dict(zip(mp_df["Team"], mp_df["Region"]))
    role_map = dict(zip(mp_df["Player"], mp_df["Position"]))
    pickrate_map = _pickrate_map(pickrate_df)

    cache = _precompute_all_holdouts(all_data, mp_df, region_lookup, role_map)
    print(f"[cache] {len(cache)} (cal, recency) configs precomputed")

    loo_results = _run_loo(cache, pickrate_map, region_lookup)
    _write_results(loo_results)


def _precompute_all_holdouts(all_data, mp_df, region_lookup, role_map):
    cache = {}
    for cal_name, cal_w in CAL_WINDOWS.items():
        for rec_name, rec_map in RECENCY_MAPS.items():
            print(f"[calibrating] {cal_name} | {rec_name}")
            try:
                cal = calibrate(all_data, cal_w, rec_map)
            except Exception as e:
                print(f"  failed: {e}")
                continue
            for year, stage in HOLDOUTS:
                key = (cal_name, rec_name, year, stage)
                cache[key] = _holdout_data(all_data, year, stage, cal,
                                           rec_map, mp_df, region_lookup,
                                           role_map)
    return cache


def _holdout_data(all_data, year, stage, calibration, rec_map, mp_df,
                   region_lookup, role_map):
    train, actual = _split(all_data, year, stage)
    if len(train) < 100 or len(actual) < 30:
        return None
    roster = _build_roster(actual, region_lookup, role_map)
    target_players = list(roster.keys())
    base = compute_base_estimates(train, target_players, calibration, rec_map)
    actual_ppg = (actual.groupby("Player")["Pts"].mean()
                  .rename("actual_ppg").reset_index())
    actual_ppg = actual_ppg[actual_ppg["Player"].isin(roster)]
    return {"base": base, "actual_ppg": actual_ppg, "roster": roster}


def _split(all_data, target_year, target_stage):
    played = all_data[all_data["P?"] == 1].copy()
    played["_rank"] = played.apply(
        lambda r: stage_rank(r["Year"], r["Stage"]), axis=1)
    target_r = stage_rank(target_year, target_stage)
    train = played[played["_rank"] < target_r].drop(columns=["_rank"])
    actual = played[(played["Year"] == target_year)
                    & (played["Stage"] == target_stage)].drop(columns=["_rank"])
    return train, actual


def _build_roster(actual, region_lookup, role_map):
    roster = {}
    for player, grp in actual.groupby("Player"):
        team = grp["Team"].iloc[0]
        roster[player] = {
            "team": team,
            "region": region_lookup.get(team, "AMER"),
            "role": role_map.get(player, "D"),
        }
    return roster


def _pickrate_map(pickrate_df):
    if pickrate_df is None:
        return {}
    return dict(zip(pickrate_df["Player"], pickrate_df["avg_pickpct"]))


def _run_loo(cache, pickrate_map, region_lookup):
    results = []
    for i, held_out in enumerate(HOLDOUTS):
        train_holdouts = [h for h in HOLDOUTS if h != held_out]
        print(f"\n[LOO {i+1}/8] hold out {held_out}")
        best = _search_best_config(cache, train_holdouts, pickrate_map,
                                    region_lookup)
        eval_r = _evaluate(cache, held_out, best, pickrate_map, region_lookup)
        baseline_r = _evaluate(cache, held_out, _v2_default_config(),
                                pickrate_map, region_lookup)
        print(f"  best train mean r={best['train_r']:.3f} "
              f"-> held-out r={eval_r:.3f} (v2 baseline {baseline_r:.3f})")
        print(f"  config: {best['cal']} | {best['rec']} | "
              f"ens={best['ens']} | pr={best['pr']}")
        results.append({
            "held_out": f"{held_out[0]} {held_out[1]}",
            "train_mean_r": best["train_r"],
            "v2_retune_r": eval_r, "v2_baseline_r": baseline_r,
            "delta": eval_r - baseline_r,
            "cal": best["cal"], "rec": best["rec"],
            "ens_eb": best["ens"][0], "ens_ridge": best["ens"][1],
            "ens_ema": best["ens"][2], "pickrate_w": best["pr"],
        })
    return results


def _v2_default_config():
    return {"cal": "v2_default", "rec": "v2_default",
            "ens": (1/3, 1/3, 1/3), "pr": 0.05, "train_r": float("nan")}


def _search_best_config(cache, train_holdouts, pickrate_map, region_lookup):
    best = {"train_r": -2.0}
    for cal_name in CAL_WINDOWS:
        for rec_name in RECENCY_MAPS:
            cfg = _try_config_grid(cache, train_holdouts, cal_name, rec_name,
                                    pickrate_map, region_lookup)
            if cfg["train_r"] > best["train_r"]:
                best = cfg
    return best


def _try_config_grid(cache, train_holdouts, cal_name, rec_name,
                      pickrate_map, region_lookup):
    best = {"train_r": -2.0, "cal": cal_name, "rec": rec_name}
    for ens in ENSEMBLE_WEIGHTS:
        for pr_w in PICKRATE_WEIGHTS:
            mean_r = _eval_on_set(cache, train_holdouts, cal_name, rec_name,
                                   ens, pr_w, pickrate_map, region_lookup)
            if mean_r > best["train_r"]:
                best = {"train_r": mean_r, "cal": cal_name, "rec": rec_name,
                        "ens": ens, "pr": pr_w}
    return best


def _eval_on_set(cache, holdouts, cal, rec, ens, pr_w, pickrate_map,
                  region_lookup):
    rs = []
    for year, stage in holdouts:
        key = (cal, rec, year, stage)
        item = cache.get(key)
        if item is None:
            continue
        r, n = score_holdout(item["base"], item["actual_ppg"], ens,
                              pr_w, pickrate_map, item["roster"],
                              region_lookup)
        if not np.isnan(r):
            rs.append(r)
    return float(np.mean(rs)) if rs else float("nan")


def _evaluate(cache, held_out, cfg, pickrate_map, region_lookup):
    year, stage = held_out
    key = (cfg["cal"], cfg["rec"], year, stage)
    item = cache.get(key)
    if item is None:
        return float("nan")
    r, _ = score_holdout(item["base"], item["actual_ppg"], cfg["ens"],
                          cfg["pr"], pickrate_map, item["roster"],
                          region_lookup)
    return r


def _write_results(rows):
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, "v2_retune_loo.csv")
    df.to_csv(out_path, index=False)
    print(f"\n[done] Wrote {out_path}")
    print("\n=== LOO summary ===")
    print(df.to_string(index=False))
    print(f"\n  baseline mean r: {df['v2_baseline_r'].mean():.3f}")
    print(f"  retune   mean r: {df['v2_retune_r'].mean():.3f}")
    print(f"  delta:           {df['delta'].mean():+.3f}")


if __name__ == "__main__":
    main()
