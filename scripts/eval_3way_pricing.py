"""3-way Stage 1 2026 pricing eval: Manual vs Algo1 (archive Combined) vs Algo2 (v2/pricing).

All algorithms use ONLY pre-Stage-1-2026 data (= 2024 + 2025 + 2026 Kickoff + 2026 Santiago).
Evaluation = predicted price vs actual 2026 Stage 1 PPG.

Reports overall metrics + factor breakdowns:
  - rookies (no prior data) vs veterans (30+ prior games)
  - roster stability (new players on team)
  - region (AMER / EMEA / PAC)
"""
import sys
import os
import io
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "archive"))

from v2.data_loader import load_all_data, load_manual_prices, load_pickrate_summary

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "output")


def get_pre_stage1_training(all_data):
    """All played games BEFORE 2026 Stage 1."""
    df = all_data[all_data["P?"] == 1].copy()
    is_old = (df["Year"] < 2026)
    is_2026_pre = (df["Year"] == 2026) & df["Stage"].isin(["Kickoff", "Santiago", "bangkok"])
    return df[is_old | is_2026_pre].copy()


def get_actual_stage1_2026(all_data):
    """Actual 2026 Stage 1 played rows."""
    df = all_data[(all_data["Year"] == 2026) & (all_data["Stage"] == "Stage 1")
                  & (all_data["P?"] == 1)].copy()
    return df


def player_actual_ppg(stage1):
    """Per-player avg Pts across all 2026 Stage 1 games played."""
    return stage1.groupby("Player").agg(
        actual_ppg=("Pts", "mean"),
        actual_ppm=("PPM", "mean"),
        games=("Pts", "count"),
        team=("Team", "last"),
    ).reset_index()


# -----------------------------------------------------------------------------
# Pricing model 1: Manual
# -----------------------------------------------------------------------------


def manual_prices(mp_df):
    return mp_df[["Player", "Team", "Stage1_Price"]].rename(
        columns={"Stage1_Price": "price"}
    ).copy()


# -----------------------------------------------------------------------------
# Pricing model 2: archive Combined
# -----------------------------------------------------------------------------


def algo1_prices(training_df, mp_df, pickrate_df):
    """Run archive's algo_combined on pre-Stage-1-2026 training."""
    import pricing_algorithms as A
    A.STAGE_ORDER.setdefault("Santiago", 1)
    player_picks = _build_player_pickrates(pickrate_df)
    features = A.compute_player_features(training_df, player_picks)
    team_wr = A.compute_team_win_rates(training_df)
    stage1_players = mp_df[["Team", "Player"]].drop_duplicates()
    team_new = A.detect_new_players_on_team(training_df, stage1_players)
    team_pop = A.compute_team_popularity(pickrate_df, training_df) if pickrate_df is not None else {}
    opp_strength = A.compute_opponent_strength(training_df)
    pred = A.algo_combined(
        features, team_win_rates=team_wr, team_new_counts=team_new,
        team_popularity=team_pop, opponent_strength=opp_strength,
    )
    pred = A.fill_missing_players(
        pred, stage1_players, features,
        team_win_rates=team_wr, team_new_counts=team_new,
    )
    pred = pred.rename(columns={"predicted_vp": "price"})
    return pred[["Player", "Team", "price"]]


def _build_player_pickrates(pickrate_df):
    if pickrate_df is None:
        return {}
    out = {}
    for _, r in pickrate_df.iterrows():
        out[r["Player"].lower()] = {
            "avg_pickpct": r.get("avg_pickpct", 0.0),
            "avg_rank_pct": r.get("avg_rank_pct", 50.0),
        }
    return out


# -----------------------------------------------------------------------------
# Pricing model 3: v2 pipeline
# -----------------------------------------------------------------------------


def algo2_prices(training_df, mp_df, pickrate_df, all_data_for_calib):
    from v2.expected_points import calibrate, compute_expected_pts
    from v2.pricing import compute_prices
    cal = calibrate(all_data_for_calib)
    roster = {r["Player"]: {"team": r["Team"], "region": r["Region"],
                            "role": r["Position"]}
              for _, r in mp_df.iterrows()}
    ep = compute_expected_pts(training_df, roster, cal)
    priced = compute_prices(ep, pickrate_df)
    return priced[["Player", "Team", "SuggestedVP"]].rename(
        columns={"SuggestedVP": "price"}
    )


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------


def evaluate(price_df, actual_df, label, mp_df, training_df):
    """Return dict of metrics for one price set."""
    merged = price_df.merge(actual_df, on="Player", how="inner")
    if len(merged) < 10:
        return {"label": label, "n": len(merged), "error": "too few overlap"}

    prices = merged["price"].values.astype(float)
    actual = merged["actual_ppg"].values.astype(float)

    r, _ = pearsonr(prices, actual)
    rho, _ = spearmanr(prices, actual)
    mae = float(np.mean(np.abs(prices - actual)))
    return {
        "label": label, "n": int(len(merged)),
        "pearson_r": float(r), "spearman_rho": float(rho), "mae": mae,
        "mean": float(np.mean(prices)), "median": float(np.median(prices)),
        "min": float(np.min(prices)), "max": float(np.max(prices)),
        "n_unique": int(pd.Series(prices).nunique()),
        "top11_sum": float(np.sort(prices)[-11:].sum()),
        "merged": merged,
    }


def factor_breakdown(price_results, training_df, mp_df):
    """Compute Pearson r within subgroups (rookies / veterans / roster stable / region)."""
    prior_games = training_df.groupby("Player").size().to_dict()
    region_lookup = dict(zip(mp_df["Player"], mp_df["Region"]))

    breakdowns = []
    cohorts = _build_cohorts(price_results[0]["merged"], prior_games, region_lookup,
                              training_df, mp_df)
    for cohort_name, mask_fn in cohorts.items():
        row = {"cohort": cohort_name}
        for res in price_results:
            m = res["merged"]
            mask = mask_fn(m, prior_games, region_lookup, training_df, mp_df)
            sub = m[mask]
            n = len(sub)
            if n < 5:
                row[res["label"]] = f"n={n} (skip)"
                continue
            r, _ = pearsonr(sub["price"].astype(float), sub["actual_ppg"].astype(float))
            row[res["label"]] = f"r={r:+.3f} (n={n})"
        row["n_total"] = int(mask_fn(
            price_results[0]["merged"], prior_games, region_lookup, training_df, mp_df
        ).sum())
        breakdowns.append(row)
    return breakdowns


def _build_cohorts(merged, prior_games, region_lookup, training_df, mp_df):
    def rookie(m, pg, *_):
        return m["Player"].map(lambda p: pg.get(p, 0) == 0)
    def green(m, pg, *_):
        return m["Player"].map(lambda p: 1 <= pg.get(p, 0) <= 10)
    def veteran(m, pg, *_):
        return m["Player"].map(lambda p: pg.get(p, 0) >= 30)
    def amer(m, _pg, rl, *_):
        return m["Player"].map(lambda p: rl.get(p) == "AMER")
    def emea(m, _pg, rl, *_):
        return m["Player"].map(lambda p: rl.get(p) == "EMEA")
    def pac(m, _pg, rl, *_):
        return m["Player"].map(lambda p: rl.get(p) == "PAC")

    return {
        "rookies (0 prior games)": rookie,
        "green (1-10 prior games)": green,
        "veterans (30+ prior games)": veteran,
        "AMER": amer,
        "EMEA": emea,
        "PAC": pac,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    print("[1/5] Loading data...")
    all_data = load_all_data()
    mp_df = load_manual_prices()
    pickrate_df = load_pickrate_summary()
    training = get_pre_stage1_training(all_data)
    actual_s1 = get_actual_stage1_2026(all_data)
    print(f"  training rows: {len(training)} | actual S1 rows: {len(actual_s1)}")

    actual_ppg = player_actual_ppg(actual_s1)
    print(f"  unique S1 players w/ data: {len(actual_ppg)}")

    print("[2/5] Computing Manual prices...")
    p_manual = manual_prices(mp_df)
    print("[3/5] Computing Algo1 (archive Combined)...")
    p_algo1 = algo1_prices(training, mp_df, pickrate_df)
    print("[4/5] Computing Algo2 (v2 pipeline)...")
    all_for_cal = all_data[~((all_data["Year"] == 2026) & (all_data["Stage"] == "Stage 1"))]
    p_algo2 = algo2_prices(training, mp_df, pickrate_df, all_for_cal)

    print("[5/5] Evaluating...")
    res_manual = evaluate(p_manual, actual_ppg, "Manual", mp_df, training)
    res_algo1 = evaluate(p_algo1, actual_ppg, "Algo1_Combined", mp_df, training)
    res_algo2 = evaluate(p_algo2, actual_ppg, "Algo2_v2", mp_df, training)

    factor_rows = factor_breakdown([res_manual, res_algo1, res_algo2], training, mp_df)

    _print_results([res_manual, res_algo1, res_algo2], factor_rows)
    _write_report([res_manual, res_algo1, res_algo2], factor_rows, training, mp_df)


def _print_results(results, factor_rows):
    print()
    print("=" * 80)
    print("OVERALL METRICS")
    print("=" * 80)
    cols = ["label", "n", "pearson_r", "spearman_rho", "mae",
            "mean", "median", "min", "max", "n_unique", "top11_sum"]
    rows = [{k: r.get(k) for k in cols} for r in results]
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print()
    print("=" * 80)
    print("FACTOR BREAKDOWN (Pearson r within cohort)")
    print("=" * 80)
    fdf = pd.DataFrame(factor_rows)
    print(fdf.to_string(index=False))


def _write_report(results, factor_rows, training, mp_df):
    os.makedirs(OUT_DIR, exist_ok=True)
    summary_path = os.path.join(OUT_DIR, "Stage1_2026_pricing_eval_summary.json")
    payload = {
        "overall": [
            {k: (round(v, 4) if isinstance(v, float) else v)
             for k, v in r.items() if k != "merged"}
            for r in results
        ],
        "factors": factor_rows,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {summary_path}")

    merged_path = os.path.join(OUT_DIR, "Stage1_2026_per_player_prices.csv")
    out_merge = results[0]["merged"][["Player", "Team", "actual_ppg", "games"]].copy()
    out_merge = out_merge.rename(columns={"price": "Manual"})
    out_merge["Manual"] = results[0]["merged"]["price"].values
    for r in results[1:]:
        out_merge = out_merge.merge(
            r["merged"][["Player", "price"]].rename(columns={"price": r["label"]}),
            on="Player", how="left",
        )
    out_merge.to_csv(merged_path, index=False)
    print(f"Wrote {merged_path}")


if __name__ == "__main__":
    main()
