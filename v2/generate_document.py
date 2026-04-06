"""Generate the 2026 Stage 1 pricing document."""
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v2.data_loader import (
    load_all_data, load_manual_prices, load_pickrate_summary, split_walk_forward,
)
from v2.expected_points import calibrate, compute_expected_pts
from v2.pricing import compute_prices, verify_distribution, price_summary
from v2.schedule import GW_REGIONS
from v2.constants import VP_MIN, VP_MAX, BUDGET, SQUAD_SIZE, TARGET_MEAN

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")


def main():
    print("=" * 60)
    print("VFL v2 - 2026 Stage 1 Pricing Document Generator")
    print("=" * 60)

    # --- Step 1: Load & calibrate ---
    print("\n[1/4] Loading data and calibrating model...")
    all_data = load_all_data()
    cal = calibrate(all_data)

    print(f"  Learned parameters:")
    print(f"    EB: mu={cal['eb_params']['mu']:.2f}, "
          f"tau2={cal['eb_params']['tau2']:.2f}, "
          f"sigma2={cal['eb_params']['sigma2']:.2f}")
    print(f"    EMA alpha: {cal['best_ema_alpha']:.2f}")
    print(f"    Opponent beta: {cal['opponent_beta']:.3f}")
    print(f"    Ridge alpha: {cal['ridge_model'].alpha_}")
    for feat, coef in cal["ridge_coefs"].items():
        print(f"    Ridge {feat}: {coef:.4f}")

    # --- Step 2: Compute expected points ---
    print("\n[2/4] Computing expected points...")
    train, _ = split_walk_forward(all_data, "Stage 1", 2026)
    train = train[train["P?"] == 1]

    mp = load_manual_prices()
    roster = {
        r["Player"]: {
            "team": r["Team"], "region": r["Region"], "role": r["Position"],
        }
        for _, r in mp.iterrows()
    }
    ep = compute_expected_pts(train, roster, cal)

    # --- Step 3: Compute prices ---
    print("\n[3/4] Computing prices...")
    pr = load_pickrate_summary()
    ep = compute_prices(ep, pr)

    manual_map = dict(zip(mp["Player"], mp["Stage1_Price"]))
    ep["ManualVP"] = ep["Player"].map(manual_map)
    ep["Diff"] = ep["SuggestedVP"] - ep["ManualVP"]

    s = price_summary(ep)
    print(f"  Mean: {s['mean']:.3f}, Median: {s['median']}, "
          f"Range: [{s['min']}, {s['max']}], Unique: {s['n_unique']}")

    corr = ep["SuggestedVP"].corr(ep["ManualVP"])
    mae = ep["Diff"].abs().mean()
    print(f"  vs Manual: corr={corr:.3f}, MAE={mae:.2f}")

    checks = verify_distribution(ep)
    for name, passed, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")

    # --- Step 4: Generate document ---
    print("\n[4/4] Generating document...")
    doc = _build_document(ep, cal)

    doc_path = os.path.join(OUT, "VFL_2026_Stage1_v2.md")
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"  Saved: {doc_path}")

    csv_path = os.path.join(OUT, "suggested_prices_2026_v2.csv")
    out_cols = [
        "Player", "Team", "Region", "Role", "N_Games", "BasePts",
        "GW1", "GW2", "GW3", "GW4", "GW5", "GW6",
        "SeasonValue", "SuggestedVP", "ManualVP", "Diff",
    ]
    ep[out_cols].sort_values("SuggestedVP", ascending=False).to_csv(
        csv_path, index=False
    )
    print(f"  Saved: {csv_path}")
    print("\n  DONE.")


def _build_document(ep, cal):
    gw_cols = [f"GW{i}" for i in range(1, 7)]

    # Region expected points per GW
    region_lines = []
    for region in ["AMER", "EMEA", "PAC"]:
        rp = ep[ep["Region"] == region]
        vals = []
        for gw in range(1, 7):
            col = f"GW{gw}"
            mean = rp[col].mean()
            top5 = rp[col].nlargest(5).mean()
            if mean == 0:
                vals.append("--")
            else:
                vals.append(f"{mean:.1f} (top5: {top5:.1f})")
        region_lines.append(f"| {region} | {' | '.join(vals)} |")

    # Price comparison table (all 180 players)
    ep_sorted = ep.sort_values("SuggestedVP", ascending=False)
    player_rows = []
    for rank, (_, r) in enumerate(ep_sorted.iterrows(), 1):
        diff = r["Diff"]
        d_str = f"+{diff:.1f}" if diff >= 0 else f"{diff:.1f}"
        m = f"{r['ManualVP']:.1f}" if pd.notna(r["ManualVP"]) else "--"
        gws = " | ".join(f"{r[f'GW{g}']:.1f}" for g in range(1, 7))
        player_rows.append(
            f"| {rank} | {r['Player']} | {r['Team']} | {r['Region']} | "
            f"{r['Role']} | {m} | {r['SuggestedVP']:.1f} | {d_str} | "
            f"{gws} | {r['SeasonValue']:.1f} |"
        )

    corr = ep["SuggestedVP"].corr(ep["ManualVP"])
    mae = ep["Diff"].abs().mean()
    w1 = (ep["Diff"].abs() <= 1).mean()
    w2 = (ep["Diff"].abs() <= 2).mean()

    doc = f"""# VFL 2026 Stage 1 -- v2 Pricing Document

## Method

All parameters learned from data via walk-forward cross-validation. No hand-tuned weights.

| Component | Method | Learned Value |
|-----------|--------|---------------|
| Shrinkage | Empirical Bayes (normal-normal) | mu={cal['eb_params']['mu']:.2f}, tau2={cal['eb_params']['tau2']:.2f}, sigma2={cal['eb_params']['sigma2']:.2f} |
| Recency | EMA alpha grid search | alpha={cal['best_ema_alpha']:.2f} |
| Feature weights | RidgeCV with LOO | alpha={cal['ridge_model'].alpha_} |
| Opponent | Within-stage regression | beta={cal['opponent_beta']:.3f} |
| Ensemble | Equal-weight average of EB + Ridge + EMA | -- |
| Pricing | Quantile mapping + pickrate adjustment | 5.0--15.0 VP |

### Ridge Coefficients (learned feature weights)
| Feature | Coefficient |
|---------|------------|
"""
    for feat, coef in cal["ridge_coefs"].items():
        doc += f"| {feat} | {coef:.4f} |\n"

    doc += f"""
## Price Summary

| Metric | Suggested | Manual |
|--------|-----------|--------|
| Mean | {ep['SuggestedVP'].mean():.3f} | {ep['ManualVP'].mean():.3f} |
| Median | {ep['SuggestedVP'].median():.1f} | {ep['ManualVP'].median():.1f} |
| Std | {ep['SuggestedVP'].std():.2f} | {ep['ManualVP'].std():.2f} |
| Min | {ep['SuggestedVP'].min():.1f} | {ep['ManualVP'].min():.1f} |
| Max | {ep['SuggestedVP'].max():.1f} | {ep['ManualVP'].max():.1f} |

| Metric | Value |
|--------|-------|
| Correlation | {corr:.3f} |
| MAE | {mae:.2f} VP |
| Within 1 VP | {w1:.0%} |
| Within 2 VP | {w2:.0%} |

## Expected Points by Region per Gameweek

| Region | GW1 | GW2 | GW3 | GW4 | GW5 | GW6 |
|--------|-----|-----|-----|-----|-----|-----|
{chr(10).join(region_lines)}

*GW1: EMEA + PAC only. GW6: AMER only.*

## Full Player List

| # | Player | Team | Region | Role | Manual | Suggested | Diff | GW1 | GW2 | GW3 | GW4 | GW5 | GW6 | Season |
|---|--------|------|--------|------|--------|-----------|------|-----|-----|-----|-----|-----|-----|--------|
{chr(10).join(player_rows)}

---

*Generated by VFL Pricing Algorithm v2.0 -- all parameters data-driven*
"""
    return doc


if __name__ == "__main__":
    main()
