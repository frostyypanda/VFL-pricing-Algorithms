"""Pricing algorithm for VFL v2.

Season value -> pickrate adjustment -> quantile mapping -> VP prices.
"""
import numpy as np
import pandas as pd
from .constants import VP_MIN, VP_MAX, VP_STEP, BUDGET, SQUAD_SIZE, TARGET_MEAN


# Target quantile curve (percentile -> VP)
QUANTILE_ANCHORS = [
    (0.00, 5.0),
    (0.05, 5.5),
    (0.10, 6.0),
    (0.18, 7.0),
    (0.30, 7.5),
    (0.42, 8.0),
    (0.52, 8.5),
    (0.60, 9.0),
    (0.68, 9.5),
    (0.75, 10.0),
    (0.82, 11.0),
    (0.88, 12.0),
    (0.93, 13.0),
    (0.97, 14.0),
    (1.00, 15.0),
]


def compute_prices(ep_matrix, pickrate_df=None):
    """Full pricing pipeline: season value -> VP prices.

    Args:
        ep_matrix: DataFrame with SeasonValue column
        pickrate_df: optional DataFrame with Player, avg_pickpct columns

    Returns ep_matrix with SuggestedVP column added.
    """
    df = ep_matrix.copy()

    # Step 1: Start from season value
    df["AdjValue"] = df["SeasonValue"].copy()

    # Step 2: Pickrate adjustment (small)
    if pickrate_df is not None:
        df = _apply_pickrate(df, pickrate_df)

    # Step 3: Quantile mapping
    df = _quantile_map(df)

    # Step 4: Budget calibration
    df = _calibrate_mean(df)

    return df


def _apply_pickrate(df, pickrate_df):
    """Add small premium for highly-picked players."""
    pr_map = dict(zip(pickrate_df["Player"], pickrate_df["avg_pickpct"]))
    df["pickrate"] = df["Player"].map(pr_map).fillna(0)

    # Normalize to [0, 1]
    pr_max = df["pickrate"].max()
    if pr_max > 0:
        df["pr_norm"] = df["pickrate"] / pr_max
    else:
        df["pr_norm"] = 0

    # Small multiplier: 0-5% boost for popular players
    # The weight is small because pickrate->performance r=0.11
    df["AdjValue"] = df["AdjValue"] * (1.0 + 0.05 * df["pr_norm"])
    df = df.drop(columns=["pickrate", "pr_norm"])
    return df


def _quantile_map(df):
    """Map adjusted season value to VP via quantile curve."""
    # Rank by adjusted value
    df["pct_rank"] = df["AdjValue"].rank(method="first", pct=True)

    pcts = [a[0] for a in QUANTILE_ANCHORS]
    vps = [a[1] for a in QUANTILE_ANCHORS]

    raw = np.interp(df["pct_rank"].values, pcts, vps)
    df["SuggestedVP"] = _snap(raw)
    df = df.drop(columns=["pct_rank"])
    return df


def _snap(prices):
    """Snap to 0.5 increments within [VP_MIN, VP_MAX]."""
    clipped = np.clip(prices, VP_MIN, VP_MAX)
    return np.round(clipped * 2) / 2


def _calibrate_mean(df):
    """Iteratively adjust prices to hit target mean ~9.09."""
    for _ in range(20):
        current = df["SuggestedVP"].mean()
        diff = TARGET_MEAN - current
        if abs(diff) < 0.05:
            break
        # Shift proportionally — larger prices shift more
        raw = df["SuggestedVP"].values.astype(float)
        shift = diff * (raw / raw.mean())
        df["SuggestedVP"] = _snap(raw + shift)

    return df


def verify_distribution(df):
    """Run automated distribution checks. Returns list of (check, pass/fail, detail)."""
    prices = df["SuggestedVP"]
    checks = []

    mean_ok = 8.9 <= prices.mean() <= 9.3
    checks.append(("Mean in [8.9, 9.3]", mean_ok, f"{prices.mean():.3f}"))

    med_ok = 8.0 <= prices.median() <= 9.5
    checks.append(("Median in [8.0, 9.5]", med_ok, f"{prices.median():.1f}"))

    min_ok = prices.min() == VP_MIN
    checks.append((f"Min = {VP_MIN}", min_ok, f"{prices.min():.1f}"))

    max_ok = prices.max() == VP_MAX
    checks.append((f"Max = {VP_MAX}", max_ok, f"{prices.max():.1f}"))

    # No bucket > 20%
    counts = prices.value_counts(normalize=True)
    max_bucket = counts.max()
    bucket_ok = max_bucket <= 0.20
    checks.append(("No bucket > 20%", bucket_ok, f"max={max_bucket:.1%}"))

    # Unique prices
    n_unique = prices.nunique()
    unique_ok = n_unique >= 15
    checks.append((">=15 unique prices", unique_ok, f"{n_unique}"))

    # Top 11 sum > 120 (stars expensive enough)
    top11 = prices.nlargest(SQUAD_SIZE).sum()
    top_ok = top11 > 120
    checks.append(("Top 11 sum > 120", top_ok, f"{top11:.1f}"))

    return checks


def price_summary(df):
    """Summary stats for the price distribution."""
    p = df["SuggestedVP"]
    return {
        "mean": round(p.mean(), 3),
        "median": p.median(),
        "std": round(p.std(), 2),
        "min": p.min(),
        "max": p.max(),
        "n_unique": p.nunique(),
        "top11_sum": round(p.nlargest(SQUAD_SIZE).sum(), 1),
    }
