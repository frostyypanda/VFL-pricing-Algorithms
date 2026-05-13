"""Score a hyperparameter config against a (pre-computed) holdout estimate cache."""
import numpy as np
from scipy.stats import pearsonr

from v2.pricing import compute_prices


def blend_and_price(base_per_player, ensemble_w, pickrate_w,
                     pickrate_map, roster, region_lookup):
    """Convert (eb, ridge, ema) per player → SuggestedVP via v2 pricing."""
    w_eb, w_ridge, w_ema = ensemble_w
    rows = []
    for player, est in base_per_player.items():
        base = w_eb * est["eb"] + w_ridge * est["ridge"] + w_ema * est["ema"]
        base = max(1.0, min(25.0, base))
        info = roster.get(player, {})
        rows.append({
            "Player": player,
            "Team": info.get("team", ""),
            "Region": info.get("region", "AMER"),
            "SeasonValue": base,
        })
    import pandas as pd
    df = pd.DataFrame(rows)
    pickrate_df = _shape_pickrate(pickrate_map, df["Player"])
    priced = compute_prices(df, pickrate_df if pickrate_w > 0 else None)
    return priced


def _shape_pickrate(pickrate_map, players):
    if not pickrate_map:
        return None
    import pandas as pd
    rows = [{"Player": p, "avg_pickpct": pickrate_map.get(p, 0.0)}
            for p in players]
    return pd.DataFrame(rows)


def score_holdout(base_per_player, actual_ppg, ensemble_w, pickrate_w,
                   pickrate_map, roster, region_lookup):
    """Return Pearson r for one holdout given precomputed bases + config."""
    priced = blend_and_price(base_per_player, ensemble_w, pickrate_w,
                              pickrate_map, roster, region_lookup)
    merged = priced.merge(actual_ppg, on="Player", how="inner")
    if len(merged) < 10:
        return float("nan"), 0
    p = merged["SuggestedVP"].astype(float).values
    a = merged["actual_ppg"].astype(float).values
    r, _ = pearsonr(p, a)
    return float(r), int(len(merged))
