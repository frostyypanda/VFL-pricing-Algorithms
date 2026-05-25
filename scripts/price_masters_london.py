"""Generate VFL prices for Masters London 2026.

Format (per user):
  - 6 players per squad: 1D + 1C + 1I + 1S + 2 wildcards
  - Budget: 50 VP
  - VP range: 5.0 - 13.0 (0.5 increments)

Pipeline (re-uses v2 building blocks where possible):
  1. Roster: 12 qualified teams, ~60 players from vlr.gg team pages.
  2. Per-player base Pts: empirical Bayes shrunk avg from all 2025+2026 data.
  3. Recency boost: weight Stage 1 2026 (incl. playoffs) + Santiago heavily.
  4. Regional adjustment from output/regional_scaling.json
     (CN slight discount; AMER/EMEA/PAC neutral).
  5. Uncertainty score from N_games + regional history + role mismatch.
  6. Quantile-map adjusted value -> VP in [5, 13], snap to 0.5.
  7. Calibrate mean toward 50 / 6 = 8.33 VP.

Writes:
  output/Masters_London_2026_prices.csv
  output/Masters_London_2026_pricing_summary.md
"""
import sys
import os
import io
import json
import math

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from v2.data_loader import normalize_team
from v2.expected_points import (
    estimate_eb_params, eb_shrink, compute_ema_estimates,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "output")
os.makedirs(OUT, exist_ok=True)


VP_MIN, VP_MAX = 5.0, 13.0
SQUAD_SIZE = 6
BUDGET = 50
TARGET_MEAN = BUDGET / SQUAD_SIZE  # 8.333


# Qualified team rosters for Masters London 2026 (from vlr.gg / Liquipedia).
# Role labels follow our schema: D=Duelist, C=Controller, I=Initiator, S=Sentinel.
# Where roles are ambiguous we mark with 'W' (wildcard-only treatment).
QUALIFIED_TEAMS = {
    # AMER
    "G2 Esports":            [("valyn", "I"), ("trent", "D"), ("leaf", "S"),
                              ("jawgemo", "D"), ("babybay", "C")],
    "LEVIATµN":              [("spikeziN", "D"), ("Sato", "C"), ("Neon", "I"),
                              ("kiNgg", "I"), ("blowz", "S")],
    "NRG Esports":           [("brawk", "I"), ("Ethan", "I"), ("Keiko", "C"),
                              ("mada", "D"), ("skuba", "S")],
    # EMEA
    "Team Heretics":         [("Wo0t", "D"), ("RieNs", "I"), ("koshmaras", "C"),
                              ("Boo", "I"), ("benjyfishy", "S")],
    "Team Vitality":         [("Sayonara", "S"), ("PROFEK", "I"), ("Jamppi", "I"),
                              ("Derke", "D"), ("Chronicle", "S")],
    "FUT Esports":           [("Yetujey", "S"), ("xeus", "C"), ("sociablEE", "I"),
                              ("s0pp", "I"), ("KROSTALY", "D")],
    # PAC
    "Paper Rex":             [("invy", "I"), ("d4v41", "S"), ("f0rsakeN", "C"),
                              ("Jinggg", "D"), ("something", "D")],
    "FULL SENSE":            [("Leviathan", "I"), ("Jitboys", "S"), ("killua", "D"),
                              ("Primmie", "C"), ("Crws", "I")],
    "Global Esports":        [("xavi8k", "D"), ("Udotan", "S"), ("PatMen", "I"),
                              ("Kr1stal", "C"), ("autumn", "I")],
    # CN
    "EDward Gaming":         [("ZmjjKK", "D"), ("Smoggy", "I"), ("nobody", "C"),
                              ("Jieni7", "I"), ("CHICHOO", "S")],
    "Xi Lai Gaming":         [("WsLeo", "S"), ("Rarga", "D"), ("NoMan", "D"),
                              ("Lysoar", "S"), ("happywei", "C")],
    "Dragon Ranger Gaming":  [("vo0kashu", "D"), ("Life", "C"), ("Nicc", "I"),
                              ("SpiritZ1", "I"), ("Flex1n", "S")],
}


PLAYER_NAME_ALIASES = {
    # roster_name (Liquipedia/vlr) -> DB historical name
    "spikeziN": "Spike",
    "blowzin": "blowz",
}


REGION_OF_TEAM = {
    "G2 Esports": "AMER", "LEVIATµN": "AMER", "NRG Esports": "AMER",
    "Team Heretics": "EMEA", "Team Vitality": "EMEA", "FUT Esports": "EMEA",
    "Paper Rex": "PAC", "FULL SENSE": "PAC", "Global Esports": "PAC",
    "EDward Gaming": "CN", "Xi Lai Gaming": "CN", "Dragon Ranger Gaming": "CN",
}


def _read_csv_any(path):
    for enc in ["utf-8-sig", "utf-8", "latin-1"]:
        try:
            return pd.read_csv(path, encoding=enc,
                                dtype={"P?": str}, keep_default_na=False)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise IOError(f"could not decode {path}")


def load_history():
    """Load all years' data, return played-only DataFrame with Pts numeric."""
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
    df = pd.concat(frames, ignore_index=True)
    return df[df["P?"] == 1].copy()


def _norm(s):
    return s.strip().lower()


def find_player_history(played, player_name, team_csv):
    """Find a player's historical rows. Match case-insensitively, prefer team.

    Tries the given name first, then the PLAYER_NAME_ALIASES mapping.
    """
    candidates = [player_name]
    if player_name in PLAYER_NAME_ALIASES:
        candidates.append(PLAYER_NAME_ALIASES[player_name])
    for cand in candidates:
        pn = _norm(cand)
        same_team = played[(played["Player"].str.lower() == pn) &
                           (played["Team"] == team_csv)]
        if len(same_team) > 0:
            return same_team
        any_team = played[played["Player"].str.lower() == pn]
        if len(any_team) > 0:
            return any_team
    return played.iloc[0:0]


STAGE_RECENCY = {
    "Stage 1": 5, "Santiago": 4, "Kickoff": 3,
    "Stage 2": 4, "Champions": 5, "London": 6,
    "Toronto": 4, "Shanghai": 4, "bangkok": 3, "Madrid": 3,
}


def recency_weight(stage, year):
    """How much to weight a game. 2026 Stage 1 max; older years less."""
    base = STAGE_RECENCY.get(stage, 2)
    year_w = {2024: 0.4, 2025: 0.7, 2026: 1.0}.get(year, 0.5)
    return base * year_w


def weighted_avg_pts(player_rows):
    if len(player_rows) == 0:
        return 0.0, 0
    w = player_rows.apply(lambda r: recency_weight(r["Stage"], r["Year"]),
                          axis=1).values
    pts = player_rows["Pts"].astype(float).values
    if w.sum() <= 0:
        return float(pts.mean()), len(pts)
    return float((pts * w).sum() / w.sum()), len(pts)


def stage1_avg_pts(player_rows):
    """Average Pts in 2026 Stage 1 + Santiago only (most recent intl form)."""
    recent = player_rows[
        (player_rows["Year"] == 2026) &
        (player_rows["Stage"].isin(["Stage 1", "Santiago"]))
    ]
    if len(recent) == 0:
        return 0.0, 0
    return float(recent["Pts"].mean()), len(recent)


INTL_STAGES = ["Kickoff", "bangkok", "Madrid", "Santiago",
               "Toronto", "Shanghai", "London", "Champions"]


def intl_game_count(player_rows):
    """How many international-event games does this player have on record?"""
    if len(player_rows) == 0:
        return 0
    return len(player_rows[player_rows["Stage"].isin(INTL_STAGES)])


def role_prior(role):
    return {"D": 9.0, "I": 7.0, "C": 7.0, "S": 7.5}.get(role, 7.5)


def load_regional_factor():
    path = os.path.join(OUT, "regional_scaling.json")
    if not os.path.exists(path):
        return {"AMER": 1.0, "EMEA": 1.0, "PAC": 1.0, "CN": 0.95}
    with open(path) as f:
        s = json.load(f)
    # Use median ratio, with floors/ceilings to keep it sane
    out = {}
    for region in ["AMER", "EMEA", "PAC", "CN"]:
        r = s.get(region, {}).get("ratio_med", 1.0)
        out[region] = max(0.85, min(1.10, r))
    # Additional CN-specific competition-strength discount (5%) — the median ratio
    # is biased because only top CN players make internationals (selection bias).
    # We're pricing a population that includes mid/bottom CN starters.
    out["CN"] = out["CN"] * 0.95
    return out


def build_player_rows(played, eb_params):
    """For each qualified player, compute base estimates + uncertainty."""
    mu, tau2, sigma2 = eb_params["mu"], eb_params["tau2"], eb_params["sigma2"]
    regional_factor = load_regional_factor()
    rows = []
    for team, players in QUALIFIED_TEAMS.items():
        team_csv = normalize_team(team)
        region = REGION_OF_TEAM[team]
        for player, role in players:
            hist = find_player_history(played, player, team_csv)
            n = len(hist)
            if n > 0:
                obs_mean = float(hist["Pts"].mean())
            else:
                obs_mean = role_prior(role)
            shrunk = eb_shrink(obs_mean, n, mu, tau2, sigma2)
            w_avg, _ = weighted_avg_pts(hist)
            recent, n_recent = stage1_avg_pts(hist)
            n_intl = intl_game_count(hist)
            # Ensemble: blend shrunk, weighted, recent (when available)
            parts = [shrunk]
            if w_avg > 0:
                parts.append(w_avg)
            if recent > 0:
                parts.append(recent * 1.15)  # slight extra recency premium
            base = float(np.mean(parts))
            # Apply regional adjustment
            adj = base * regional_factor.get(region, 1.0)
            # Uncertainty
            unc, flag, reason = _uncertainty(n, n_recent, n_intl, region)
            rows.append({
                "Player": player, "Team": team_csv, "Region": region,
                "Role": role, "N_Games": n, "N_Stage1": n_recent,
                "N_Intl": n_intl,
                "ObsMean": round(obs_mean, 2),
                "Shrunk": round(shrunk, 2),
                "WeightedAvg": round(w_avg, 2),
                "RecentMean": round(recent, 2),
                "BaseValue": round(base, 2),
                "RegionFactor": round(regional_factor[region], 3),
                "AdjValue": round(adj, 2),
                "Uncertainty": unc,
                "UncertainFlag": flag,
                "UncertaintyReason": reason,
            })
    return pd.DataFrame(rows)


def _uncertainty(n_total, n_recent, n_intl, region):
    """Returns (score 0-1, flag Y/N, reason).

    Score components:
      +0.50 if <5 total games (very thin sample)
      +0.25 if 5-11 total games (thin sample)
      +0.25 if <3 recent (2026 S1+Santiago) games
      +0.30 if region=CN and <5 international games (weak translation prior)
      +0.15 if region=CN and >=5 intl games (residual competition gap)
      +0.20 if <3 international games seen historically (regardless of region)
    """
    score = 0.0
    reasons = []
    if n_total < 5:
        score += 0.50
        reasons.append(f"only_{n_total}_games")
    elif n_total < 12:
        score += 0.25
        reasons.append(f"thin_sample_{n_total}")
    if n_recent < 3:
        score += 0.25
        reasons.append(f"only_{n_recent}_recent")
    if region == "CN":
        if n_intl < 5:
            score += 0.30
            reasons.append("CN_low_intl_history")
        else:
            score += 0.15
            reasons.append("CN_competition_gap")
    elif n_intl < 3:
        score += 0.20
        reasons.append(f"only_{n_intl}_intl_games")
    score = min(1.0, score)
    flag = "Y" if score >= 0.30 else "N"
    reason = ";".join(reasons) if reasons else "-"
    return round(score, 3), flag, reason


QUANTILE_ANCHORS = [
    (0.00, VP_MIN),
    (0.05, 5.5),
    (0.12, 6.0),
    (0.22, 6.5),
    (0.32, 7.0),
    (0.42, 7.5),
    (0.52, 8.0),
    (0.62, 8.5),
    (0.72, 9.0),
    (0.80, 9.5),
    (0.86, 10.0),
    (0.91, 10.5),
    (0.95, 11.0),
    (0.97, 11.5),
    (0.99, 12.0),
    (1.00, VP_MAX),
]


def quantile_to_vp(rank_pct):
    pcts = [a[0] for a in QUANTILE_ANCHORS]
    vps = [a[1] for a in QUANTILE_ANCHORS]
    raw = np.interp(rank_pct, pcts, vps)
    return raw


def snap(prices):
    return np.round(np.clip(prices, VP_MIN, VP_MAX) * 2) / 2


def compute_vp(df):
    df = df.copy()
    df["pct_rank"] = df["AdjValue"].rank(method="first", pct=True)
    raw = quantile_to_vp(df["pct_rank"].values)
    df["SuggestedVP"] = snap(raw)
    # Calibrate mean toward TARGET_MEAN
    for _ in range(20):
        cur = df["SuggestedVP"].mean()
        diff = TARGET_MEAN - cur
        if abs(diff) < 0.05:
            break
        shift = diff * (df["SuggestedVP"].values / df["SuggestedVP"].mean())
        df["SuggestedVP"] = snap(df["SuggestedVP"].values + shift)
    return df.drop(columns=["pct_rank"])


def write_outputs(df, regional_factor, eb_params):
    df = df.sort_values(["Region", "Team", "SuggestedVP"], ascending=[True, True, False])
    csv_cols = [
        "Player", "Team", "Region", "Role",
        "SuggestedVP", "Uncertainty", "UncertainFlag", "UncertaintyReason",
        "N_Games", "N_Stage1", "N_Intl",
        "BaseValue", "AdjValue", "RegionFactor",
        "ObsMean", "Shrunk", "WeightedAvg", "RecentMean",
    ]
    csv_path = os.path.join(OUT, "Masters_London_2026_prices.csv")
    df[csv_cols].to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\nWrote {csv_path}")
    _write_summary(df, regional_factor, eb_params)


def _write_summary(df, regional_factor, eb_params):
    p = df["SuggestedVP"]
    lines = [
        "# Masters London 2026 — VFL Pricing Summary",
        "",
        f"- **Players priced**: {len(df)}",
        f"- **Squad format**: {SQUAD_SIZE} players, {BUDGET} VP budget, "
        f"{VP_MIN}-{VP_MAX} VP range (0.5 increments)",
        f"- **Role slots**: 1D + 1C + 1I + 1S + 2 Wildcards",
        "",
        "## Distribution",
        "",
        f"- Mean: **{p.mean():.2f}** (target {TARGET_MEAN:.2f})",
        f"- Median: **{p.median():.1f}**",
        f"- Min: **{p.min():.1f}**, Max: **{p.max():.1f}**",
        f"- Unique prices: **{p.nunique()}**",
        "",
        "## Regional scaling factors applied",
        "",
        "| Region | Median ratio (intl/reg) | Effective factor |",
        "|--------|-------------------------|------------------|",
    ]
    for r, f in regional_factor.items():
        lines.append(f"| {r} | from analysis | {f:.3f} |")
    lines += [
        "",
        "## High-uncertainty players (Flag=Y)",
        "",
        "| Player | Team | Region | Role | VP | Reason |",
        "|--------|------|--------|------|----|--------|",
    ]
    for _, r in df[df["UncertainFlag"] == "Y"].iterrows():
        lines.append(
            f"| {r['Player']} | {r['Team']} | {r['Region']} | {r['Role']} | "
            f"{r['SuggestedVP']:.1f} | {r['UncertaintyReason']} |"
        )
    lines += [
        "",
        "## Price distribution by 0.5 bucket",
        "",
        "| VP | Count |",
        "|-----|-------|",
    ]
    counts = p.value_counts().sort_index()
    for vp_val, c in counts.items():
        lines.append(f"| {vp_val:.1f} | {c} |")
    lines += [
        "",
        "## EB params (used for shrinkage)",
        f"- mu={eb_params['mu']:.3f}, tau²={eb_params['tau2']:.3f}, sigma²={eb_params['sigma2']:.3f}",
    ]
    path = os.path.join(OUT, "Masters_London_2026_pricing_summary.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {path}")


def main():
    print("[1/5] Loading history...")
    played = load_history()
    print(f"  {len(played)} played rows")

    print("[2/5] Estimating empirical Bayes population params...")
    from v2.expected_points import compute_eb_estimates
    _, eb_params = compute_eb_estimates(played)
    print(f"  mu={eb_params['mu']:.3f}, tau²={eb_params['tau2']:.3f}, "
          f"sigma²={eb_params['sigma2']:.3f}")

    print("[3/5] Building player rows for 12 qualified teams...")
    df = build_player_rows(played, eb_params)
    print(f"  {len(df)} players")
    print(df[["Player", "Team", "Region", "Role", "N_Games", "N_Stage1",
              "BaseValue", "AdjValue"]].to_string(index=False))

    print("\n[4/5] Mapping AdjValue -> VP via quantile + calibration...")
    df = compute_vp(df)
    print(f"  mean={df['SuggestedVP'].mean():.2f}, median={df['SuggestedVP'].median():.1f}, "
          f"min={df['SuggestedVP'].min():.1f}, max={df['SuggestedVP'].max():.1f}")

    print("\n[5/5] Writing outputs...")
    rf = load_regional_factor()
    write_outputs(df, rf, eb_params)


if __name__ == "__main__":
    main()
