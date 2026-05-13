"""v3 pricing — quantile + region calibration + star cap."""
import numpy as np
import pandas as pd

from v2.constants import VP_MIN, VP_MAX, TARGET_MEAN, SQUAD_SIZE
from v2.pricing import QUANTILE_ANCHORS, _snap, _calibrate_mean

from .star_cap import compute_recent_vs_historic, apply_star_cap
from .flags import V3Flags

MIN_REGION_PLAYERS = 30


def compute_v3_prices(ep_matrix, played_df, flags=None):
    """Pricing pipeline: SeasonValue -> region quantile -> star cap -> calibrate.

    Args:
        ep_matrix: DataFrame with Player, Team, Region, SeasonValue
        played_df: training data for star-cap recent/historic computation
        flags: V3Flags toggling star_cap / region_quantile
    """
    flags = flags or V3Flags()
    df = ep_matrix.copy()
    df["AdjValue"] = df["SeasonValue"].copy()
    if flags.region_quantile:
        df = _region_quantile_map(df)
    else:
        df["SuggestedVP"] = _quantile_map_subset(df["AdjValue"])
    if flags.star_cap:
        df = _apply_star_caps(df, played_df)
    df = _calibrate_mean(df)
    return df


def _region_quantile_map(df):
    """Quantile-map within each region if region has enough players, else global."""
    df["SuggestedVP"] = np.nan
    for region in df["Region"].dropna().unique():
        region_mask = df["Region"] == region
        n = int(region_mask.sum())
        if n >= MIN_REGION_PLAYERS:
            df.loc[region_mask, "SuggestedVP"] = _quantile_map_subset(
                df.loc[region_mask, "AdjValue"]
            )
        else:
            df.loc[region_mask, "SuggestedVP"] = _quantile_map_subset(
                df["AdjValue"]
            ).loc[df.loc[region_mask].index]
    leftover_mask = df["SuggestedVP"].isna()
    if leftover_mask.any():
        df.loc[leftover_mask, "SuggestedVP"] = _quantile_map_subset(
            df.loc[leftover_mask, "AdjValue"]
        )
    return df


def _quantile_map_subset(series):
    """Quantile-map a single series through QUANTILE_ANCHORS."""
    pct_rank = series.rank(method="first", pct=True)
    pcts = [a[0] for a in QUANTILE_ANCHORS]
    vps = [a[1] for a in QUANTILE_ANCHORS]
    raw = np.interp(pct_rank.values, pcts, vps)
    snapped = _snap(raw)
    return pd.Series(snapped, index=series.index)


def _apply_star_caps(df, played_df):
    """For each row, cap price if recent-vs-historic indicates decline."""
    if len(played_df) == 0:
        return df
    pop_mean = float(played_df["Pts"].mean())
    recent_hist = compute_recent_vs_historic(played_df, df["Player"].tolist())
    capped = []
    for _, row in df.iterrows():
        r_avg, h_avg = recent_hist.get(row["Player"], (None, None))
        new_price = apply_star_cap(row["SuggestedVP"], r_avg, h_avg, pop_mean)
        capped.append(new_price)
    df["SuggestedVP"] = _snap(np.array(capped, dtype=float))
    return df


def price_summary(df):
    p = df["SuggestedVP"]
    return {
        "mean": round(float(p.mean()), 3),
        "median": float(p.median()),
        "std": round(float(p.std()), 2),
        "min": float(p.min()), "max": float(p.max()),
        "n_unique": int(p.nunique()),
        "top11_sum": round(float(p.nlargest(SQUAD_SIZE).sum()), 1),
    }
